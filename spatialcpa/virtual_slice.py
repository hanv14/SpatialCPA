"""
SpatialCPA v3 — Virtual Slice Generation from neighboring sections.

This module implements *true* virtual-slice generation: given a target
z-coordinate and the two flanking real sections (already registered / aligned),
it synthesizes a complete virtual slice — cell positions, cell types, and gene
expression — WITHOUT ever looking at the slice being generated.

Why v3 exists
-------------
The v2 generator (:mod:`spatialcpa.inference`) takes the held-out slice as a
reference and predicts expression at each of its *true* cell (x, y) positions,
usually conditioned on its *true* cell types. That is reconstruction at known
locations, not generation — it cannot exist without the very slice it is meant
to invent, so it cannot build a virtual slice at a novel z.

v3 removes that dependency entirely. The only inputs are:

    * a target z,
    * two neighboring real sections (z_below < z < z_above), aligned,
    * the trained continuous field h(x, y, z) -> (cell type, expression).

and the pipeline is:

    A. Estimate how many cells the virtual slice should contain.
    B. Build an interpolated cell-density field from the two neighbors and
       sample de-novo cell positions from it (generative, blue-noise spaced).
    C. Assign each position a cell type by sampling from the learned spatial
       classifier, blended with the local neighbor composition and smoothed
       into coherent spatial domains.
    D. Generate each cell's expression by SAMPLING the model's generative
       decoder (Gaussian or ZINB), or — for a deterministic ('mse') model —
       by adding empirical same-type residuals drawn from the neighbors.

The result is biologically plausible, coherent with its neighbors, and
genuinely generative (stochastic, with cell-to-cell variability) rather than a
linear interpolation of the flanking slices.
"""

import numpy as np
import torch
import anndata as ad
import pandas as pd
from scipy.spatial import cKDTree


# ===================================================================
# Neighbor context
# ===================================================================

class NeighborContext:
    """
    Holds the two flanking real sections and derives everything the generator
    needs from them: z-distance weights, an interpolated density field, cell
    counts, cell-type composition priors, and (optionally) expression-residual
    pools for deterministic models.

    Parameters
    ----------
    section_below, section_above : SpatialSection
        The two aligned real sections flanking the target z. ``section_below``
        must have the smaller z-center. They are assumed already registered so
        their (x, y) coordinate frames are comparable.
    target_z : float
        The z-coordinate of the virtual slice to generate.
    """

    def __init__(self, section_below, section_above, target_z):
        if section_below.z_center > section_above.z_center:
            section_below, section_above = section_above, section_below
        self.below = section_below
        self.above = section_above
        self.target_z = float(target_z)

        z_lo = self.below.z_center
        z_hi = self.above.z_center
        span = max(z_hi - z_lo, 1e-8)
        # Linear interpolation weight: t=0 at below, t=1 at above.
        self.t = float(np.clip((self.target_z - z_lo) / span, 0.0, 1.0))
        self.w_below = 1.0 - self.t
        self.w_above = self.t

        # Median nearest-neighbor spacing (cell scale) from both neighbors.
        self.cell_spacing = self._estimate_spacing()

    def _estimate_spacing(self):
        spacings = []
        for sec in (self.below, self.above):
            xy = sec.coords_xy
            if len(xy) < 2:
                continue
            tree = cKDTree(xy)
            d, _ = tree.query(xy, k=2)
            spacings.append(np.median(d[:, 1]))
        if not spacings:
            return 1.0
        return float(np.mean(spacings))

    def target_n_cells(self, jitter=0.0, rng=None):
        """
        Interpolated number of cells for the virtual slice.

        Parameters
        ----------
        jitter : float
            Relative Gaussian jitter on the count (e.g. 0.05 = +/-5%). 0 gives
            the deterministic interpolated count.
        """
        n = self.w_below * self.below.n_cells + self.w_above * self.above.n_cells
        if jitter > 0:
            rng = rng or np.random
            n = n * (1.0 + rng.normal(0.0, jitter))
        return max(int(round(n)), 1)

    # -- density field ------------------------------------------------------

    def build_density_grid(self, cells_per_bin=1.5):
        """
        Build an interpolated 2D cell-density field from the two neighbors.

        Returns
        -------
        grid : dict with keys
            'prob'   (H, W) normalized probability per bin,
            'x_edges', 'y_edges' bin edges,
            'bin_w', 'bin_h' bin sizes.
        """
        all_xy = np.vstack([self.below.coords_xy, self.above.coords_xy])
        x_min, y_min = all_xy.min(axis=0)
        x_max, y_max = all_xy.max(axis=0)

        # Bin size chosen so each bin holds a few cells on average.
        bin_size = max(self.cell_spacing * cells_per_bin, 1e-6)
        nx = max(int(np.ceil((x_max - x_min) / bin_size)), 1)
        ny = max(int(np.ceil((y_max - y_min) / bin_size)), 1)
        x_edges = np.linspace(x_min, x_max, nx + 1)
        y_edges = np.linspace(y_min, y_max, ny + 1)

        h_below, _, _ = np.histogram2d(
            self.below.coords_xy[:, 0], self.below.coords_xy[:, 1],
            bins=[x_edges, y_edges])
        h_above, _, _ = np.histogram2d(
            self.above.coords_xy[:, 0], self.above.coords_xy[:, 1],
            bins=[x_edges, y_edges])

        # Normalize each neighbor to a probability field, then z-interpolate.
        p_below = h_below / max(h_below.sum(), 1e-8)
        p_above = h_above / max(h_above.sum(), 1e-8)
        prob = self.w_below * p_below + self.w_above * p_above
        total = prob.sum()
        if total <= 0:
            prob = np.ones_like(prob) / prob.size
        else:
            prob = prob / total

        return {
            'prob': prob,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'bin_w': x_edges[1] - x_edges[0] if nx > 0 else bin_size,
            'bin_h': y_edges[1] - y_edges[0] if ny > 0 else bin_size,
        }


# ===================================================================
# Position sampling
# ===================================================================

def sample_positions(grid, n_cells, rng=None, relax_iters=2, relax_strength=0.4):
    """
    Sample ``n_cells`` de-novo (x, y) positions from an interpolated density
    grid, then apply a light blue-noise relaxation so spacing resembles real
    tissue rather than a regular lattice or clumps.

    Parameters
    ----------
    grid : dict
        Output of :meth:`NeighborContext.build_density_grid`.
    n_cells : int
    rng : np.random.Generator or None
    relax_iters : int
        Number of Lloyd-style repulsion passes (0 disables).
    relax_strength : float
        Fraction of the local spacing deficit corrected per pass.

    Returns
    -------
    positions : (n_cells, 2) float32 array.
    """
    rng = rng or np.random.default_rng()
    prob = grid['prob'].ravel()
    nx, ny = grid['prob'].shape
    bin_w, bin_h = grid['bin_w'], grid['bin_h']

    # Multinomial assignment of cells to bins.
    flat_idx = rng.choice(prob.size, size=n_cells, p=prob)
    ix = flat_idx // ny
    iy = flat_idx % ny

    x0 = grid['x_edges'][ix]
    y0 = grid['y_edges'][iy]
    # Uniform jitter within the chosen bin.
    xs = x0 + rng.random(n_cells) * bin_w
    ys = y0 + rng.random(n_cells) * bin_h
    positions = np.column_stack([xs, ys]).astype(np.float64)

    if relax_iters > 0 and n_cells > 3:
        target = np.sqrt((bin_w * bin_h * nx * ny) / max(n_cells, 1))
        for _ in range(relax_iters):
            tree = cKDTree(positions)
            d, idx = tree.query(positions, k=2)
            nn_d = d[:, 1]
            nn_i = idx[:, 1]
            # Vector away from the nearest neighbor, scaled by how much closer
            # than the target spacing the pair currently is.
            vec = positions - positions[nn_i]
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            norm[norm < 1e-9] = 1e-9
            deficit = np.clip(target - nn_d, 0, None)[:, None]
            positions = positions + relax_strength * (vec / norm) * deficit

    return positions.astype(np.float32)


# ===================================================================
# Cell-type assignment
# ===================================================================

def neighbor_celltype_prior(positions, ctx, n_cell_types, k=12):
    """
    Local cell-type composition prior from the two neighbor sections.

    For each virtual position, gather the ``k`` nearest cells in each neighbor
    (in xy), weight them by z-proximity and inverse xy-distance, and accumulate
    a probability vector over cell types.

    Returns
    -------
    prior : (n_cells, n_cell_types) float32, rows sum to 1.
    """
    n = len(positions)
    prior = np.zeros((n, n_cell_types), dtype=np.float64)

    for sec, w_z in ((ctx.below, ctx.w_below), (ctx.above, ctx.w_above)):
        if w_z <= 0 or sec.n_cells == 0:
            continue
        kk = min(k, sec.n_cells)
        tree = cKDTree(sec.coords_xy)
        d, idx = tree.query(positions, k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = w_z / (d + 1e-6)
        ct = sec.cell_type_indices[idx]  # (n, kk)
        for j in range(kk):
            np.add.at(prior, (np.arange(n), ct[:, j]), w[:, j])

    row = prior.sum(axis=1, keepdims=True)
    row[row <= 0] = 1.0
    return (prior / row).astype(np.float32)


def smooth_probability_field(prob, positions, k=8, iterations=1):
    """
    Spatially smooth a per-cell probability field over the generated points'
    k-NN graph so that sampled cell types form coherent domains instead of a
    salt-and-pepper mixture. Renormalized after each pass.
    """
    if iterations <= 0 or len(positions) <= k:
        return prob
    tree = cKDTree(positions)
    _, idx = tree.query(positions, k=min(k + 1, len(positions)))
    out = prob
    for _ in range(iterations):
        out = out[idx].mean(axis=1)
        out = out / np.clip(out.sum(axis=1, keepdims=True), 1e-8, None)
    return out


# ===================================================================
# Generator
# ===================================================================

class VirtualSliceGeneratorV3:
    """
    Generate a virtual slice at an arbitrary z from its two neighboring
    real sections and a trained SpatialCPA model.

    Parameters
    ----------
    model : SpatialCPA
        Trained model. A ``gaussian`` or ``zinb`` model gives fully generative
        expression; an ``mse`` model falls back to empirical residual sampling.
    cell_type_names : list of str
    gene_names : list of str
    device : str
    """

    def __init__(self, model, cell_type_names, gene_names, device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.cell_type_names = list(cell_type_names)
        self.gene_names = list(gene_names)
        self.n_cell_types = len(cell_type_names)
        self.device = device

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, section_below, section_above, target_z,
                 n_cells=None, count_jitter=0.0,
                 ct_model_weight=0.5, ct_smooth_k=8, ct_smooth_iters=1,
                 ct_temperature=1.0,
                 expr_temperature=1.0, expr_smooth_k=0,
                 relax_iters=2, batch_size=4096, seed=None,
                 var=None):
        """
        Generate a complete virtual slice.

        Parameters
        ----------
        section_below, section_above : SpatialSection
            The two aligned neighbors flanking ``target_z``.
        target_z : float
            z of the virtual slice.
        n_cells : int or None
            Number of cells to generate. If None, interpolated from neighbors.
        count_jitter : float
            Relative jitter on the interpolated cell count (ignored if
            ``n_cells`` is given).
        ct_model_weight : float
            Blend between the learned classifier and the neighbor composition
            prior for cell types: ``w*P_model + (1-w)*P_neighbor``. 1.0 trusts
            the model fully; 0.0 uses only the neighbor composition.
        ct_smooth_k, ct_smooth_iters : int
            k-NN smoothing of the cell-type probability field into domains.
        ct_temperature : float
            Softmax temperature for cell-type sampling. 0 → argmax.
        expr_temperature : float
            Sampling temperature for expression (0 → decoder mean).
        expr_smooth_k : int
            If > 0, light spatial smoothing of expression (off by default so
            spatial autocorrelation is preserved).
        relax_iters : int
            Blue-noise relaxation passes for positions.
        batch_size : int
            Inference batch size.
        seed : int or None
            Seed for reproducible generation.
        var : pd.DataFrame or None
            Optional var frame for the output AnnData (e.g. gene metadata).

        Returns
        -------
        virtual_adata : AnnData
            With ``obsm['spatial']`` (x, y), ``obs['z']``, ``obs['cell_class']``,
            ``obs['cell_type_idx']`` and ``X`` (expression).
        """
        rng = np.random.default_rng(seed)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        ctx = NeighborContext(section_below, section_above, target_z)

        # --- A. cell count -------------------------------------------------
        if n_cells is None:
            n_cells = ctx.target_n_cells(jitter=count_jitter, rng=rng)

        # --- B. positions --------------------------------------------------
        grid = ctx.build_density_grid()
        positions = sample_positions(grid, n_cells, rng=rng,
                                     relax_iters=relax_iters)
        z_col = np.full((len(positions), 1), ctx.target_z, dtype=np.float32)
        coords_3d = np.hstack([positions, z_col]).astype(np.float32)

        # --- C. cell types -------------------------------------------------
        p_model = self._model_celltype_prob(coords_3d, batch_size)
        p_nbr = neighbor_celltype_prior(positions, ctx, self.n_cell_types)
        w = float(np.clip(ct_model_weight, 0.0, 1.0))
        p_ct = w * p_model + (1.0 - w) * p_nbr
        p_ct = p_ct / np.clip(p_ct.sum(axis=1, keepdims=True), 1e-8, None)
        p_ct = smooth_probability_field(p_ct, positions, k=ct_smooth_k,
                                        iterations=ct_smooth_iters)
        cell_types = self._sample_categorical(p_ct, ct_temperature, rng)

        # --- D. expression -------------------------------------------------
        expr = self._generate_expression(
            coords_3d, cell_types, ctx,
            temperature=expr_temperature, batch_size=batch_size, gen=gen, rng=rng)

        if expr_smooth_k > 0:
            expr = _spatial_smooth(expr, positions, k=expr_smooth_k)

        # --- assemble AnnData ---------------------------------------------
        cell_type_labels = [self.cell_type_names[i] for i in cell_types]
        obs = pd.DataFrame({
            'cell_class': cell_type_labels,
            'cell_type_idx': cell_types,
            'z': np.full(len(positions), ctx.target_z, dtype=np.float32),
        })
        obs.index = [f'virtual_z{ctx.target_z:g}_{i}' for i in range(len(positions))]

        if var is None:
            var = pd.DataFrame(index=self.gene_names)

        virtual_adata = ad.AnnData(X=expr.astype(np.float32), obs=obs, var=var)
        virtual_adata.obsm['spatial'] = positions.astype(np.float32)
        virtual_adata.uns['target_z'] = ctx.target_z
        virtual_adata.uns['z_below'] = ctx.below.z_center
        virtual_adata.uns['z_above'] = ctx.above.z_center
        return virtual_adata

    # -- internals ----------------------------------------------------------

    def _model_celltype_prob(self, coords_3d, batch_size):
        probs = []
        for start in range(0, len(coords_3d), batch_size):
            end = min(start + batch_size, len(coords_3d))
            c = torch.tensor(coords_3d[start:end], device=self.device)
            p = self.model.predict_cell_type(c).cpu().numpy()
            probs.append(p)
        return np.concatenate(probs, axis=0).astype(np.float32)

    @staticmethod
    def _sample_categorical(prob, temperature, rng):
        n = len(prob)
        if temperature <= 0.0:
            return prob.argmax(axis=1).astype(np.int64)
        logits = np.log(np.clip(prob, 1e-12, None)) / max(temperature, 1e-6)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        # Vectorized categorical sampling via inverse-CDF.
        cdf = np.cumsum(p, axis=1)
        u = rng.random(n)[:, None]
        return (u < cdf).argmax(axis=1).astype(np.int64)

    def _generate_expression(self, coords_3d, cell_types, ctx,
                             temperature, batch_size, gen, rng):
        n = len(coords_3d)
        n_genes = len(self.gene_names)
        out = np.zeros((n, n_genes), dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = torch.tensor(coords_3d[start:end], device=self.device)
            ct = torch.tensor(cell_types[start:end], dtype=torch.long,
                              device=self.device)
            if self.model.expression_mode in ('gaussian', 'zinb'):
                e = self.model.sample_expression(
                    c, ct, temperature=temperature, generator=gen)
            else:
                e = self.model.predict_expression(c, ct)
            out[start:end] = e.cpu().numpy()

        # For a deterministic ('mse') model there is no learned noise: inject
        # biologically realistic variability by adding same-type residuals
        # sampled from the neighbor cells.
        if self.model.expression_mode == 'mse' and temperature > 0.0:
            out = self._add_empirical_residuals(out, cell_types, ctx,
                                                temperature, batch_size, rng)
        return out

    def _add_empirical_residuals(self, mean_expr, cell_types, ctx,
                                 temperature, batch_size, rng):
        """Add per-cell-type expression residuals drawn from the neighbors."""
        # Build residual pools: real_expr - model_mean, per cell type.
        pools = {}
        for sec in (ctx.below, ctx.above):
            if sec.n_cells == 0:
                continue
            coords = sec.get_3d_coords().astype(np.float32)
            resid = np.zeros_like(sec.expression, dtype=np.float32)
            for start in range(0, sec.n_cells, batch_size):
                end = min(start + batch_size, sec.n_cells)
                c = torch.tensor(coords[start:end], device=self.device)
                ct = torch.tensor(sec.cell_type_indices[start:end],
                                  dtype=torch.long, device=self.device)
                m = self.model.predict_expression(c, ct).cpu().numpy()
                resid[start:end] = sec.expression[start:end] - m
            for ct_id in np.unique(sec.cell_type_indices):
                mask = sec.cell_type_indices == ct_id
                pools.setdefault(ct_id, []).append(resid[mask])
        pools = {k: np.concatenate(v, axis=0) for k, v in pools.items()}

        out = mean_expr.copy()
        for ct_id in np.unique(cell_types):
            if ct_id not in pools or len(pools[ct_id]) == 0:
                continue
            tgt = np.where(cell_types == ct_id)[0]
            pool = pools[ct_id]
            pick = rng.integers(0, len(pool), size=len(tgt))
            out[tgt] += temperature * pool[pick]
        return out


def _spatial_smooth(expression, positions, k=8):
    """Gaussian-weighted k-NN smoothing (kept light; see inference.py)."""
    tree = cKDTree(positions)
    kk = min(k + 1, len(positions))
    d, idx = tree.query(positions, k=kk)
    sigma = np.median(d[:, 1]) * 1.5 + 1e-8
    w = np.exp(-d ** 2 / (2 * sigma ** 2))
    w /= w.sum(axis=1, keepdims=True)
    return np.einsum('nk,nkg->ng', w, expression[idx]).astype(np.float32)
