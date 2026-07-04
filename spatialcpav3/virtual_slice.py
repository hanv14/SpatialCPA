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


def neighbor_expression_mean(positions, cell_types, ctx, n_genes, k=12, eps=1e-6):
    """
    Cell-type-conditioned inverse-distance expression estimate from the two
    flanking sections.

    For each generated cell, average the ``k`` nearest **same-type** real cells
    (pooled from both neighbors, weighted by z-proximity and inverse
    xy-distance). This grounds the generated expression in the real tissue that
    actually surrounds the target z — a much stronger signal than the coordinate
    field alone — while remaining a de-novo synthesis (no held-out access).

    Returns
    -------
    mean : (n, n_genes) float32 neighbor-anchored mean (rows without any
        same-type neighbor are left at zero).
    have : (n,) bool, True where a same-type estimate was available.
    """
    n = len(positions)
    xy = np.vstack([ctx.below.coords_xy, ctx.above.coords_xy])
    expr = np.vstack([ctx.below.expression, ctx.above.expression]).astype(np.float32)
    ctv = np.concatenate([ctx.below.cell_type_indices,
                          ctx.above.cell_type_indices])
    zw = np.concatenate([
        np.full(ctx.below.n_cells, ctx.w_below, dtype=np.float64),
        np.full(ctx.above.n_cells, ctx.w_above, dtype=np.float64),
    ])

    out = np.zeros((n, n_genes), dtype=np.float32)
    have = np.zeros(n, dtype=bool)
    for t in np.unique(cell_types):
        tgt = np.where(cell_types == t)[0]
        src = np.where(ctv == t)[0]
        if len(src) == 0:
            continue
        kk = min(k, len(src))
        tree = cKDTree(xy[src])
        d, idx = tree.query(positions[tgt], k=kk)
        if kk == 1:
            d = d[:, None]
            idx = idx[:, None]
        w = zw[src][idx] / (d + eps)
        w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        out[tgt] = np.einsum('nk,nkg->ng', w.astype(np.float32), expr[src][idx])
        have[tgt] = True
    return out, have


def neighbor_local_residuals(ctx, k=12, eps=1e-6):
    """
    Per-cell-type pool of **calibrated local expression fluctuations** measured
    on the real flanking tissue.

    For each real neighbor cell, subtract its own leave-self-out same-type
    inverse-distance mean (the same estimator :func:`neighbor_expression_mean`
    uses for generated cells). The residual is the cell's genuine local
    deviation from the smooth field — it carries the *real* magnitude and
    (near-absent) spatial structure of biological + technical variability.

    Adding a randomly drawn residual of the right type to a generated cell's
    neighbor-anchored mean therefore reproduces the real slice's spatial
    statistics (and hence its per-gene Moran's I) far more faithfully than
    injecting the model's learned Gaussian noise, whose scale is only loosely
    calibrated.

    Returns
    -------
    pools : dict {cell_type_index -> (m_t, n_genes) residual array}.
    """
    xy = np.vstack([ctx.below.coords_xy, ctx.above.coords_xy])
    expr = np.vstack([ctx.below.expression, ctx.above.expression]).astype(np.float32)
    ctv = np.concatenate([ctx.below.cell_type_indices,
                          ctx.above.cell_type_indices])
    zw = np.concatenate([
        np.full(ctx.below.n_cells, ctx.w_below, dtype=np.float64),
        np.full(ctx.above.n_cells, ctx.w_above, dtype=np.float64),
    ])

    pools = {}
    for t in np.unique(ctv):
        src = np.where(ctv == t)[0]
        e = expr[src]
        if len(src) < 3:
            # Too few to estimate a local mean — use deviation from the type mean.
            pools[t] = (e - e.mean(axis=0, keepdims=True)).astype(np.float32)
            continue
        pts = xy[src]
        kk = min(k + 1, len(src))
        tree = cKDTree(pts)
        d, idx = tree.query(pts, k=kk)
        # Drop the self match (first column).
        d = d[:, 1:]
        idx = idx[:, 1:]
        w = zw[src][idx] / (d + eps)
        w /= np.clip(w.sum(axis=1, keepdims=True), 1e-12, None)
        local_mean = np.einsum('nk,nkg->ng', w.astype(np.float32), e[idx])
        pools[t] = (e - local_mean).astype(np.float32)
    return pools


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
    density_sampler : DensitySampler or None
        Optional trained 3D density field (:mod:`spatialcpav3.density`). When
        provided, cell positions and counts are drawn from the learned global
        tissue-architecture prior (works at any z and along any plane); when
        None, positions fall back to the 2-neighbour occupancy histogram.
    """

    def __init__(self, model, cell_type_names, gene_names, device='cpu',
                 density_sampler=None):
        self.model = model.to(device)
        self.model.eval()
        self.cell_type_names = list(cell_type_names)
        self.gene_names = list(gene_names)
        self.n_cell_types = len(cell_type_names)
        self.device = device
        self.density_sampler = density_sampler

    # -- public API ---------------------------------------------------------

    @torch.no_grad()
    def generate(self, section_below, section_above, target_z,
                 n_cells=None, count_jitter=0.0,
                 ct_model_weight=0.3, ct_smooth_k=8, ct_smooth_iters=3,
                 ct_temperature=0.4,
                 expr_temperature=0.35, expr_model_weight=0.1, expr_noise='empirical',
                 expr_neighbor_k=2, expr_smooth_k=0,
                 relax_iters=2, density_field_weight=0.5,
                 batch_size=4096, seed=None,
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
            Sampling temperature for expression (0 → grounded mean, no noise).
        expr_model_weight : float
            Blend between the model's predicted expression mean and the
            neighbor-anchored (real flanking tissue) mean:
            ``beta*mu_model + (1-beta)*mu_neighbor``. 1.0 uses only the learned
            field; lower values ground expression more in the real neighbors
            (higher fidelity). Generative noise is added on top either way.
        expr_noise : str
            Noise model for the generative residual: ``'empirical'`` (default)
            draws real same-type local fluctuations from the flanking tissue
            (calibrated to the real slice → best spatial-autocorrelation match);
            ``'model'`` uses the decoder's learned Gaussian scale.
        expr_neighbor_k : int
            Neighbors used for the cell-type-conditioned inverse-distance
            expression mean. Larger k → smoother, more strongly grounded mean.
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

        # --- A/B. positions (+ count) --------------------------------------
        # Prefer the learned 3D density field (global tissue-architecture prior,
        # valid at any z). When two flanking slices exist, refine it with their
        # local occupancy (hybrid); otherwise fall back to the histogram.
        if n_cells is None:
            n_cells = ctx.target_n_cells(jitter=count_jitter, rng=rng)
        if self.density_sampler is not None:
            local_prob = self._neighbor_occupancy_on_grid(ctx)
            positions = self.density_sampler.sample_plane(
                target_z, n_cells=n_cells, rng=rng, relax_iters=relax_iters,
                local_prob=local_prob, field_weight=density_field_weight)
        else:
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
            temperature=expr_temperature, expr_model_weight=expr_model_weight,
            batch_size=batch_size, gen=gen, rng=rng, noise=expr_noise,
            neighbor_k=expr_neighbor_k)

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

    @torch.no_grad()
    def generate_section(self, point, normal, n_cells=None, extent=None,
                         n_grid=160, ct_temperature=0.4, ct_smooth_k=8,
                         ct_smooth_iters=3, expr_temperature=0.35,
                         batch_size=4096, seed=None, var=None):
        """
        In-silico tissue sectioning: generate a virtual slice on an ARBITRARY
        oriented plane (sagittal, coronal, oblique) cut through the volume.

        Requires a ``density_sampler`` (learned 3D field). Because an arbitrary
        plane has no two parallel observed neighbours, positions come from the
        learned density field and cell types / expression come purely from the
        trained neural field (no neighbour anchoring).

        Parameters
        ----------
        point : (3,) point the plane passes through.
        normal : (3,) plane normal (defines the cut orientation).
        n_cells : int or None
            Number of cells (default inferred from the integrated density).
        extent, n_grid : float, int
            In-plane patch half-size and grid resolution.
        Returns
        -------
        AnnData with ``obsm['spatial']`` = in-plane (u, v) coords, ``obsm['xyz']``
        = 3D positions on the plane, ``obs['cell_class']`` and ``X``.
        """
        if self.density_sampler is None:
            raise ValueError("generate_section requires a density_sampler "
                             "(train a DensityFieldModel first).")
        rng = np.random.default_rng(seed)
        gen = None
        if seed is not None:
            gen = torch.Generator(device=self.device).manual_seed(int(seed))

        coords_3d, uv = self.density_sampler.sample_section(
            point, normal, n_cells=n_cells, extent=extent, n_grid=n_grid, rng=rng)
        if len(coords_3d) == 0:
            raise ValueError("No tissue intersects the requested plane.")

        # Cell types from the learned classifier (model-only), smoothed + sampled.
        p_ct = self._model_celltype_prob(coords_3d, batch_size)
        p_ct = smooth_probability_field(p_ct, uv, k=ct_smooth_k,
                                        iterations=ct_smooth_iters)
        cell_types = self._sample_categorical(p_ct, ct_temperature, rng)

        # Expression from the model only (no parallel neighbours to anchor to).
        n = len(coords_3d)
        n_genes = len(self.gene_names)
        expr = np.zeros((n, n_genes), dtype=np.float32)
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            c = torch.tensor(coords_3d[s:e], device=self.device)
            ct = torch.tensor(cell_types[s:e], dtype=torch.long, device=self.device)
            if self.model.expression_mode in ('gaussian', 'zinb'):
                expr[s:e] = self.model.sample_expression(
                    c, ct, temperature=expr_temperature, generator=gen).cpu().numpy()
            else:
                expr[s:e] = self.model.predict_expression(c, ct).cpu().numpy()

        labels = [self.cell_type_names[i] for i in cell_types]
        obs = pd.DataFrame({'cell_class': labels, 'cell_type_idx': cell_types})
        obs.index = [f'section_{i}' for i in range(n)]
        if var is None:
            var = pd.DataFrame(index=self.gene_names)
        adata = ad.AnnData(X=expr.astype(np.float32), obs=obs, var=var)
        adata.obsm['spatial'] = uv.astype(np.float32)
        adata.obsm['xyz'] = coords_3d.astype(np.float32)
        adata.uns['plane_point'] = np.asarray(point, dtype=np.float32)
        adata.uns['plane_normal'] = np.asarray(normal, dtype=np.float32)
        return adata

    # -- internals ----------------------------------------------------------

    def _neighbor_occupancy_on_grid(self, ctx):
        """z-weighted occupancy of the two flanking slices on the field grid.

        Returns a per-bin probability aligned to ``DensitySampler.plane_rate_grid``,
        or None if no density sampler is set.
        """
        if self.density_sampler is None:
            return None
        s = self.density_sampler
        gx, gy = s._grid_xy()
        nx, ny = gx.shape
        x0 = s.x_min
        y0 = s.y_min
        bs = s.bin_size
        occ = np.zeros(nx * ny, dtype=np.float64)
        for sec, w_z in ((ctx.below, ctx.w_below), (ctx.above, ctx.w_above)):
            if w_z <= 0 or sec.n_cells == 0:
                continue
            ix = np.clip(((sec.coords_xy[:, 0] - x0) / bs).astype(int), 0, nx - 1)
            iy = np.clip(((sec.coords_xy[:, 1] - y0) / bs).astype(int), 0, ny - 1)
            flat = ix * ny + iy
            np.add.at(occ, flat, w_z)
        total = occ.sum()
        return occ / total if total > 0 else None

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
                             temperature, expr_model_weight, batch_size, gen, rng,
                             noise='empirical', neighbor_k=12):
        n = len(coords_3d)
        n_genes = len(self.gene_names)
        mode = self.model.expression_mode

        # ZINB (raw counts): keep pure generative count sampling — blending
        # counts with a continuous neighbor mean is not well defined.
        if mode == 'zinb':
            out = np.zeros((n, n_genes), dtype=np.float32)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                c = torch.tensor(coords_3d[start:end], device=self.device)
                ct = torch.tensor(cell_types[start:end], dtype=torch.long,
                                  device=self.device)
                out[start:end] = self.model.sample_expression(
                    c, ct, temperature=temperature, generator=gen).cpu().numpy()
            return out

        # --- 1. model-predicted mean (and learned std for gaussian) ----------
        mu_model = np.zeros((n, n_genes), dtype=np.float32)
        std_model = np.zeros((n, n_genes), dtype=np.float32) if mode == 'gaussian' else None
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            c = torch.tensor(coords_3d[start:end], device=self.device)
            ct = torch.tensor(cell_types[start:end], dtype=torch.long,
                              device=self.device)
            mu, std = self.model.predict_expression_dist(c, ct)
            mu_model[start:end] = mu.cpu().numpy()
            if std_model is not None:
                std_model[start:end] = std.cpu().numpy()

        # --- 2. anchor the mean in the real flanking tissue ------------------
        beta = float(np.clip(expr_model_weight, 0.0, 1.0))
        mu = mu_model
        if beta < 1.0:
            mu_nbr, have = neighbor_expression_mean(
                coords_3d[:, :2], cell_types, ctx, n_genes, k=neighbor_k)
            blended = beta * mu_model + (1.0 - beta) * mu_nbr
            mu = np.where(have[:, None], blended, mu_model).astype(np.float32)

        # --- 3. generative noise around the grounded mean --------------------
        if temperature <= 0.0:
            return mu
        # 'empirical' (default): draw real local fluctuations from the neighbors
        # — matches the real slice's noise magnitude/structure, so per-gene
        # Moran's I lines up. 'model': use the learned Gaussian scale.
        if noise == 'model' and mode == 'gaussian':
            eps = std_model * temperature * \
                rng.standard_normal(mu.shape).astype(np.float32)
            return (mu + eps).astype(np.float32)
        return self._add_calibrated_residuals(mu, cell_types, ctx,
                                              temperature, rng)

    def _add_calibrated_residuals(self, mean_expr, cell_types, ctx,
                                  temperature, rng):
        """Add real same-type local fluctuations (see neighbor_local_residuals)."""
        pools = neighbor_local_residuals(ctx)
        out = mean_expr.copy()
        for ct_id in np.unique(cell_types):
            pool = pools.get(ct_id)
            if pool is None or len(pool) == 0:
                continue
            tgt = np.where(cell_types == ct_id)[0]
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
