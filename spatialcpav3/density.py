"""
Learned continuous 3D cell-density field for SpatialCPA v3.

The v3 generator's most important step is placing cells: cell types and
expression are only meaningful once positions are right. The original
positions came from a 2-slice occupancy histogram, which cannot generalise to
an arbitrary z (or an arbitrary cutting plane) and does not learn any global
prior over tissue architecture.

This module replaces that with a **deep prior over 3D tissue architecture**
learned from ALL training slices at once. Cell positions are modelled as an
inhomogeneous Poisson point process with intensity ``lambda(x, y, z)``; a
coordinate network (Fourier features -> residual backbone -> optional k-NN
spatial self-attention -> softplus rate head) regresses the expected cell count
per unit area at any 3D location. It is trained on grid-binned cell counts
pooled across every training section, with empty bins acting as negatives.

Because the field is continuous in (x, y, z), it can be queried:
  * at any z (interpolating between or extrapolating beyond observed sections),
  * along any oriented plane (enabling in-silico tissue sectioning at arbitrary
    angles),
and it reflects the whole tissue's architecture rather than just two flanking
slices.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

from spatialcpav3.fourier import FourierFeatureEncoder
from spatialcpav3.backbone import SpatialBackbone


# ===================================================================
# k-NN spatial self-attention
# ===================================================================

class KNNSelfAttention(nn.Module):
    """
    Local multi-head self-attention over each point's k spatial neighbours.

    For every query point, the k nearest points (within the current batch, by
    3D coordinate) act as keys/values; a learned relative-position bias injects
    geometry. This lets the density (and, when reused, cell-type) prediction at
    a location depend on its learned neighbourhood — an attention mechanism on
    the prediction head that encourages spatially coherent fields.

    The neighbour graph is built with a KD-tree on detached coordinates (used
    only for gather indices), so gradients still flow through the gathered
    features.
    """

    def __init__(self, dim, n_heads=4, k=16, coord_dim=3):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert self.head_dim * n_heads == dim, "dim must be divisible by n_heads"
        self.k = k
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        # Relative-position bias: (dx, dy, dz) -> per-head scalar bias.
        self.rel = nn.Sequential(
            nn.Linear(coord_dim, dim), nn.GELU(), nn.Linear(dim, n_heads),
        )

    def forward(self, h, coords):
        """
        Parameters
        ----------
        h : (N, dim) per-point features.
        coords : (N, 3) point coordinates (same order as h).

        Returns
        -------
        (N, dim) attention-refined features (residual + LayerNorm).
        """
        N = h.shape[0]
        k = min(self.k + 1, N)  # +1 to include self
        with torch.no_grad():
            pts = coords.detach().cpu().numpy()
            tree = cKDTree(pts)
            _, idx = tree.query(pts, k=k)
            if k == 1:
                idx = idx[:, None]
        idx_t = torch.as_tensor(idx, dtype=torch.long, device=h.device)  # (N, k)

        q = self.q(h).view(N, self.n_heads, self.head_dim)              # (N, H, d)
        kv = self.kv(h)                                                 # (N, 2*dim)
        nbr_kv = kv[idx_t]                                              # (N, k, 2*dim)
        key, val = nbr_kv.split(self.dim, dim=-1)
        key = key.view(N, k, self.n_heads, self.head_dim)
        val = val.view(N, k, self.n_heads, self.head_dim)

        # Attention logits + relative-position bias.
        attn = torch.einsum('nhd,nkhd->nhk', q, key) * self.scale       # (N, H, k)
        rel = coords[idx_t] - coords.unsqueeze(1)                       # (N, k, 3)
        bias = self.rel(rel).permute(0, 2, 1)                          # (N, H, k)
        attn = torch.softmax(attn + bias, dim=-1)

        out = torch.einsum('nhk,nkhd->nhd', attn, val).reshape(N, self.dim)
        return self.norm(h + self.proj(out))


# ===================================================================
# Density field model
# ===================================================================

class DensityFieldModel(nn.Module):
    """
    Continuous 3D cell-intensity field lambda(x, y, z).

    Parameters
    ----------
    xy_scale, z_scale : float
        Characteristic spatial scales (as in the main model).
    n_freq_xy, n_freq_z : int
        Fourier frequencies.
    hidden : int
        Backbone width / feature dim.
    n_layers : int
        Backbone residual layers.
    use_attention : bool
        If True, apply k-NN spatial self-attention before the rate head.
    attn_heads, attn_k : int
        Attention configuration.
    """

    def __init__(self, xy_scale, z_scale, n_freq_xy=32, n_freq_z=24,
                 hidden=192, n_layers=4, use_attention=True,
                 attn_heads=4, attn_k=16):
        super().__init__()
        self.fourier = FourierFeatureEncoder(
            n_freq_xy=n_freq_xy, n_freq_z=n_freq_z,
            xy_scale=xy_scale, z_scale=z_scale)
        self.backbone = SpatialBackbone(
            input_dim=self.fourier.output_dim, hidden_dim=hidden,
            output_dim=hidden, n_layers=n_layers, dropout=0.0)
        self.attn = KNNSelfAttention(hidden, n_heads=attn_heads, k=attn_k) \
            if use_attention else None
        self.rate_head = nn.Linear(hidden, 1)

    def forward(self, coords):
        """Return the log-rate (pre-softplus) per coordinate, shape (N,)."""
        h = self.backbone(self.fourier(coords))
        if self.attn is not None:
            h = self.attn(h, coords)
        return self.rate_head(h).squeeze(-1)

    def rate(self, coords):
        """Return the non-negative expected cell count per bin, shape (N,)."""
        return F.softplus(self.forward(coords)) + 1e-6


# ===================================================================
# Training data: grid-binned counts across all slices
# ===================================================================

def build_bin_counts(sections, bin_size):
    """
    Pool all training sections into grid-binned cell counts.

    Every section is binned on a common ``bin_size`` grid over its own bounding
    box; empty bins are kept as zero-count negatives so the field learns tissue
    support, not just where cells are.

    Returns
    -------
    coords : (M, 3) float32 bin centres (x, y, z).
    counts : (M,) float32 cell counts per bin.
    bin_size : float (echoed).
    """
    all_coords = []
    all_counts = []
    for sec in sections:
        xy = sec.coords_xy
        z = float(np.median(sec.z_values))
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        nx = max(int(np.ceil((x_max - x_min) / bin_size)), 1)
        ny = max(int(np.ceil((y_max - y_min) / bin_size)), 1)
        x_edges = x_min + bin_size * np.arange(nx + 1)
        y_edges = y_min + bin_size * np.arange(ny + 1)
        hist, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[x_edges, y_edges])
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        gx, gy = np.meshgrid(xc, yc, indexing='ij')
        coords = np.column_stack([gx.ravel(), gy.ravel(),
                                  np.full(gx.size, z)]).astype(np.float32)
        all_coords.append(coords)
        all_counts.append(hist.ravel().astype(np.float32))
    return (np.concatenate(all_coords, axis=0),
            np.concatenate(all_counts, axis=0), bin_size)


def estimate_bin_size(sections, cells_per_bin=2.0):
    """Median nearest-neighbour spacing across sections, scaled by cells_per_bin."""
    spac = []
    for sec in sections:
        if sec.n_cells < 2:
            continue
        d, _ = cKDTree(sec.coords_xy).query(sec.coords_xy, k=2)
        spac.append(np.median(d[:, 1]))
    s = float(np.mean(spac)) if spac else 1.0
    return max(s * cells_per_bin, 1e-6)


class DensityFieldTrainer:
    """
    Poisson-regression trainer for :class:`DensityFieldModel`.

    Trains lambda(x, y, z) to match grid-binned counts from all sections with a
    Poisson negative log-likelihood (rate - count*log(rate)), using per-section
    full-batch steps so the k-NN attention sees a coherent spatial neighbourhood.
    """

    def __init__(self, model, sections, device='cpu', lr=2e-3,
                 cells_per_bin=2.0, z_jitter=0.3):
        self.model = model.to(device)
        self.device = device
        self.z_jitter = z_jitter
        self.bin_size = estimate_bin_size(sections, cells_per_bin)
        # Per-section binned tensors (kept separate for coherent attention).
        self.batches = []
        for sec in sections:
            coords, counts, _ = build_bin_counts([sec], self.bin_size)
            self.batches.append((
                torch.tensor(coords, device=device),
                torch.tensor(counts, device=device),
            ))
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def train(self, n_epochs=60, verbose=False):
        hist = []
        for ep in range(n_epochs):
            self.model.train()
            order = np.random.permutation(len(self.batches))
            losses = []
            for bi in order:
                coords, counts = self.batches[bi]
                # Jitter z within the slab so the field is smooth across z.
                c = coords.clone()
                if self.z_jitter > 0:
                    c[:, 2] = c[:, 2] + (torch.rand(c.shape[0], device=self.device)
                                         - 0.5) * 2 * self.z_jitter
                self.opt.zero_grad()
                rate = self.model.rate(c)
                # Poisson NLL (drop the constant log(count!) term).
                loss = (rate - counts * torch.log(rate)).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                losses.append(loss.item())
            hist.append(float(np.mean(losses)))
            if verbose and (ep % 10 == 0 or ep == n_epochs - 1):
                print(f"  [density] epoch {ep}: poisson_nll={hist[-1]:.4f}")
        return hist


# ===================================================================
# Sampling positions from the learned field
# ===================================================================

class DensitySampler:
    """
    Draw cell positions from a trained :class:`DensityFieldModel`.

    Wraps the field with the training bounding box + bin size so it can be
    queried on a regular grid at any z, or along an arbitrary oriented plane.
    """

    def __init__(self, model, sections, bin_size, device='cpu', pad=1):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.bin_size = bin_size
        xy = np.vstack([s.coords_xy for s in sections])
        self.x_min, self.y_min = xy.min(axis=0) - pad * bin_size
        self.x_max, self.y_max = xy.max(axis=0) + pad * bin_size
        z = np.concatenate([s.z_values for s in sections])
        self.z_min, self.z_max = float(z.min()), float(z.max())

    # -- grid helpers -------------------------------------------------------

    def _grid_xy(self):
        nx = max(int(np.ceil((self.x_max - self.x_min) / self.bin_size)), 1)
        ny = max(int(np.ceil((self.y_max - self.y_min) / self.bin_size)), 1)
        xc = self.x_min + self.bin_size * (np.arange(nx) + 0.5)
        yc = self.y_min + self.bin_size * (np.arange(ny) + 0.5)
        gx, gy = np.meshgrid(xc, yc, indexing='ij')
        return gx, gy

    @torch.no_grad()
    def _rate_on(self, coords3d, batch=8192):
        out = np.zeros(len(coords3d), dtype=np.float32)
        for s in range(0, len(coords3d), batch):
            e = min(s + batch, len(coords3d))
            c = torch.tensor(coords3d[s:e], dtype=torch.float32, device=self.device)
            out[s:e] = self.model.rate(c).cpu().numpy()
        return out

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def plane_rate_grid(self, z):
        """
        Evaluate the learned rate on the axial grid at height ``z``.

        Returns
        -------
        centers : (M, 2) bin-centre xy.
        rate : (M,) predicted cell count per bin.
        """
        gx, gy = self._grid_xy()
        centers = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)
        coords = np.column_stack([centers,
                                  np.full(len(centers), z, np.float32)]).astype(np.float32)
        return centers, self._rate_on(coords)

    @torch.no_grad()
    def expected_count(self, z):
        """Total expected number of cells on the z-plane (sum of the rate grid)."""
        _, rate = self.plane_rate_grid(z)
        return float(rate.sum())

    @torch.no_grad()
    def sample_plane(self, z, n_cells=None, rng=None, relax_iters=2,
                     relax_strength=0.4, floor_frac=0.02, local_prob=None,
                     field_weight=1.0):
        """
        Sample de-novo (x, y) positions on the axial plane at height ``z``.

        Parameters
        ----------
        local_prob : (M,) or None
            Optional per-bin probability from a local source (e.g. the two
            flanking slices' occupancy), aligned to :meth:`plane_rate_grid`.
            When given, the sampling density is
            ``field_weight * field + (1-field_weight) * local`` — the learned
            global prior refined by local observations.
        field_weight : float
            Blend weight for the learned field vs ``local_prob``.

        Returns
        -------
        positions : (n, 2) float32.
        """
        rng = rng or np.random.default_rng()
        centers, rate = self.plane_rate_grid(z)
        if n_cells is None:
            n_cells = max(int(round(rate.sum())), 1)
        thr = floor_frac * rate.max() if rate.max() > 0 else 0.0
        rate = np.where(rate >= thr, rate, 0.0)
        field_p = rate / max(rate.sum(), 1e-12)
        if local_prob is not None:
            lp = np.clip(np.asarray(local_prob, dtype=np.float64), 0, None)
            lp = lp / max(lp.sum(), 1e-12)
            w = float(np.clip(field_weight, 0.0, 1.0))
            p = w * field_p + (1.0 - w) * lp
            p = p / max(p.sum(), 1e-12)
        else:
            p = field_p
        return _sample_from_prob(centers, p, self.bin_size, n_cells,
                                 rng, relax_iters, relax_strength)

    @torch.no_grad()
    def sample_section(self, point, normal, n_cells=None, extent=None,
                       n_grid=160, rng=None, floor_frac=0.02):
        """
        Sample positions on an ARBITRARY oriented plane (in-silico sectioning).

        Parameters
        ----------
        point : (3,) a point the cutting plane passes through.
        normal : (3,) plane normal (any non-zero vector; sagittal/coronal/oblique).
        n_cells : int or None
            If None, inferred from the integrated rate over the plane patch.
        extent : float or None
            Half-size of the square patch sampled on the plane (in xy units).
            Defaults to the tissue's xy diagonal.
        n_grid : int
            Grid resolution along each in-plane axis.

        Returns
        -------
        coords3d : (n, 3) float32 positions in the tissue volume, lying on the
            requested plane.
        uv : (n, 2) float32 in-plane coordinates (for 2D visualisation).
        """
        rng = rng or np.random.default_rng()
        point = np.asarray(point, dtype=np.float64)
        normal = np.asarray(normal, dtype=np.float64)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        # Two orthonormal in-plane axes.
        a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = a - normal * (a @ normal)
        u = u / (np.linalg.norm(u) + 1e-12)
        v = np.cross(normal, u)
        if extent is None:
            extent = 0.5 * np.hypot(self.x_max - self.x_min, self.y_max - self.y_min)
        ts = np.linspace(-extent, extent, n_grid)
        uu, vv = np.meshgrid(ts, ts, indexing='ij')
        flatu, flatv = uu.ravel(), vv.ravel()
        pts = point[None, :] + flatu[:, None] * u[None, :] + flatv[:, None] * v[None, :]
        # Keep only points inside the trained volume (bounding box in xy/z).
        inside = ((pts[:, 0] >= self.x_min) & (pts[:, 0] <= self.x_max) &
                  (pts[:, 1] >= self.y_min) & (pts[:, 1] <= self.y_max) &
                  (pts[:, 2] >= self.z_min - self.bin_size) &
                  (pts[:, 2] <= self.z_max + self.bin_size))
        rate = np.zeros(len(pts), dtype=np.float32)
        if inside.any():
            rate[inside] = self._rate_on(pts[inside].astype(np.float32))
        cell_area = (2 * extent / n_grid) ** 2
        if n_cells is None:
            n_cells = max(int(round(rate.sum() * cell_area / (self.bin_size ** 2))), 1)
        thr = floor_frac * rate.max() if rate.max() > 0 else 0.0
        rate = np.where(rate >= thr, rate, 0.0)
        tot = rate.sum()
        if tot <= 0:
            return (np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32))
        p = rate / tot
        step = 2 * extent / n_grid
        pick = rng.choice(len(pts), size=n_cells, p=p)
        ju = (rng.random(n_cells) - 0.5) * step
        jv = (rng.random(n_cells) - 0.5) * step
        uv = np.column_stack([flatu[pick] + ju, flatv[pick] + jv]).astype(np.float32)
        coords3d = (point[None, :] + uv[:, 0:1] * u[None, :]
                    + uv[:, 1:2] * v[None, :]).astype(np.float32)
        return coords3d, uv


def _sample_from_prob(bin_centers, prob, bin_size, n_cells, rng,
                      relax_iters, relax_strength):
    """Multinomial bin sampling + jitter + blue-noise relaxation (2D)."""
    pick = rng.choice(len(bin_centers), size=n_cells, p=prob)
    jitter = (rng.random((n_cells, 2)) - 0.5) * bin_size
    positions = (bin_centers[pick] + jitter).astype(np.float64)
    if relax_iters > 0 and n_cells > 3:
        area = bin_size ** 2 * np.count_nonzero(prob)
        target = np.sqrt(max(area, 1e-12) / max(n_cells, 1))
        for _ in range(relax_iters):
            d, idx = cKDTree(positions).query(positions, k=2)
            vec = positions - positions[idx[:, 1]]
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm[norm < 1e-9] = 1e-9
            deficit = np.clip(target - d[:, 1], 0, None)[:, None]
            positions = positions + relax_strength * (vec / norm) * deficit
    return positions.astype(np.float32)
