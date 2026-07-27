"""Phase 2.1/2.2 — the multi-channel structure field.

The proposal is explicit that this must **not** be an RGB "coloured by label"
image: RGB distances are meaningless and a generative model produces invalid
intermediate colours.  The field here is

    lambda_c(x, y, z) = sum_{i : t_i = c} K_h(p - p_i)

one non-negative channel per type group, plus a total-density channel, plus
**density-weighted mean type-embedding** channels (Phase 2.1(b)) which carry
within-group identity in a space where distances are meaningful.  Channels are
commensurate (all are densities or density-weighted sums), so every intermediate
state a generative model can produce is a valid field.

Three details from the proposal are honoured:

* **channel budget** — per-type channels are used only while the number of types
  is small; above ``FieldConfig.max_group_channels`` the channels drop to the
  Phase 1.3 hierarchy (class/subclass) and the embedding channels carry the rest;
* **anisotropy** — the in-plane grid is coarsened to a multiple of the median cell
  spacing, so voxel aspect ratios stay sane;
* **slabs, not planes** — a section is the z-integral of the field over its
  thickness, so rasterizing a section accumulates *all* of its cells (whatever
  their exact z) into that section's plane.  That is the forward operator, and
  it is what the interpolator is trained to invert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import FieldConfig


@dataclass
class FieldSpec:
    """Geometry + channel layout of the structure field."""

    grid_h: int
    grid_w: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    bandwidth: float
    group_of_type: np.ndarray        # (C,) type -> channel group
    n_groups: int
    type_embedding: np.ndarray       # (C, E) embedding used for the mean-emb channels

    @property
    def n_channels(self) -> int:
        return 1 + self.n_groups + self.type_embedding.shape[1]

    def bin_of(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        xb = np.clip(np.digitize(u[:, 0], self.x_edges) - 1, 0, self.grid_w - 1)
        yb = np.clip(np.digitize(u[:, 1], self.y_edges) - 1, 0, self.grid_h - 1)
        return yb, xb

    def centers(self) -> Tuple[np.ndarray, np.ndarray]:
        xc = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        yc = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        return yc, xc


def build_field_spec(u_all: np.ndarray, type_rep, spacing_norm: float,
                     cfg: FieldConfig) -> FieldSpec:
    """Choose the grid and the channel layout (Phase 2.1)."""
    lo = u_all.min(axis=0)
    hi = u_all.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = 0.02 * span
    lo, hi = lo - pad, hi + pad
    target = max(cfg.bin_spacings * spacing_norm, 1e-6)
    gw = int(np.clip(round(float((hi[0] - lo[0]) / target)), cfg.min_grid, cfg.max_grid))
    gh = int(np.clip(round(float((hi[1] - lo[1]) / target)), cfg.min_grid, cfg.max_grid))

    C = type_rep.n_types
    if C <= cfg.max_group_channels:
        group_of = np.arange(C)
        n_groups = C
    elif type_rep.subclass_of.max() + 1 <= cfg.max_group_channels:
        group_of = type_rep.subclass_of.copy()
        n_groups = int(group_of.max() + 1)
    else:
        group_of = type_rep.class_of.copy()
        n_groups = int(group_of.max() + 1)

    E = type_rep.embedding
    ne = int(min(cfg.embedding_channels, E.shape[1]))
    emb = E[:, :ne] if ne > 0 else np.zeros((C, 0))

    return FieldSpec(grid_h=gh, grid_w=gw,
                     x_edges=np.linspace(lo[0], hi[0], gw + 1),
                     y_edges=np.linspace(lo[1], hi[1], gh + 1),
                     bandwidth=cfg.kde_bandwidth_bins, group_of_type=group_of,
                     n_groups=n_groups, type_embedding=emb)


def rasterize_section(u: np.ndarray, types: Optional[np.ndarray],
                      spec: FieldSpec) -> np.ndarray:
    """Rasterize one section (a *slab*: all of its cells) to (C_ch, H, W).

    Channel 0 is total density; channels ``1..n_groups`` are per-group densities
    ``lambda_g``; the remaining channels are the density-weighted embedding sums
    ``sum_i K_h(p - p_i) e_{c(i)}`` — deliberately *unnormalized*, so every
    channel accumulates linearly and the slab z-integral is exact.  The mean
    embedding at a voxel is recovered by dividing by channel 0.
    """
    H, W = spec.grid_h, spec.grid_w
    ne = spec.type_embedding.shape[1]
    F = np.zeros((spec.n_channels, H, W), dtype=np.float32)
    if u.shape[0] == 0:
        return F
    yb, xb = spec.bin_of(u)
    flat = yb * W + xb
    np.add.at(F[0].reshape(-1), flat, 1.0)
    if types is not None:
        g = spec.group_of_type[np.clip(types, 0, len(spec.group_of_type) - 1)]
        for gi in range(spec.n_groups):
            m = g == gi
            if m.any():
                np.add.at(F[1 + gi].reshape(-1), flat[m], 1.0)
        if ne:
            emb = spec.type_embedding[np.clip(types, 0, spec.type_embedding.shape[0] - 1)]
            for e in range(ne):
                np.add.at(F[1 + spec.n_groups + e].reshape(-1), flat, emb[:, e])
    if spec.bandwidth > 0:
        for c in range(F.shape[0]):
            F[c] = gaussian_filter(F[c], sigma=spec.bandwidth, mode="nearest")
    return F


def rasterize_stack(stack, frame, spec: FieldSpec) -> np.ndarray:
    """Rasterize every observed section -> (Z, C_ch, H, W)."""
    out = []
    for s in stack.slices:
        u, _ = frame.to_normalized(s.coords_xy, s.z_values, s.section_id)
        out.append(rasterize_section(u, s.cell_type_indices, spec))
    return np.stack(out).astype(np.float32)


def linear_field(fields: np.ndarray, a: int, b: int, t: float) -> np.ndarray:
    """The linear-interpolation baseline the Phase 2.5 stop condition compares to."""
    return (1.0 - t) * fields[a] + t * fields[b]


# ── Phase 2.2 — compression (automatic; a 3D VAE is overkill at these sizes) ──

@dataclass
class ChannelCompressor:
    """Channel-space PCA used only when the field has more channels than the
    interpolator should carry.  ``enabled=False`` is the identity."""

    enabled: bool
    mean: Optional[np.ndarray] = None
    comps: Optional[np.ndarray] = None      # (k, C_ch)

    def encode(self, F: np.ndarray) -> np.ndarray:
        """(..., C_ch, H, W) -> (..., k, H, W)."""
        if not self.enabled:
            return F
        V = np.moveaxis(F, -3, -1)                       # (..., H, W, C_ch)
        Z = (V - self.mean) @ self.comps.T                # (..., H, W, k)
        return np.moveaxis(Z, -1, -3)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """(..., k, H, W) -> (..., C_ch, H, W)."""
        if not self.enabled:
            return Z
        V = np.moveaxis(Z, -3, -1)                        # (..., H, W, k)
        F = V @ self.comps + self.mean
        return np.moveaxis(F, -1, -3)


def fit_compressor(fields: np.ndarray, cfg: FieldConfig) -> ChannelCompressor:
    """Automatic rule: compress only when the channel count exceeds the budget."""
    n_ch = fields.shape[1]
    if n_ch <= cfg.compress_max_channels:
        return ChannelCompressor(enabled=False)
    V = fields.transpose(0, 2, 3, 1).reshape(-1, n_ch)
    mu = V.mean(axis=0)
    Vc = V - mu
    k = int(cfg.compress_max_channels)
    _, _, Vt = np.linalg.svd(Vc[np.random.default_rng(0).choice(
        Vc.shape[0], min(20000, Vc.shape[0]), replace=False)], full_matrices=False)
    return ChannelCompressor(enabled=True, mean=mu, comps=Vt[:k])
