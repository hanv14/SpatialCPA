"""The virtual section's geometry: where the cells are, before what they express.

The spectral model in ``harmonics.py`` generates *signals on a graph*, so it
needs a graph — which needs positions. This module supplies them, and it is
deliberately the least clever part of the method: everything v16 claims lives in
what is put *on* the geometry, not in the geometry itself.

Two things still have to be right, because two of the five target metrics read
position. ``paper_marker_depth_r`` integrates expression along an axis derived
from the ground truth, so the tissue's **extent and outline** must be right or
the profile is computed over the wrong support. ``paper_celltype_localization``
compares each type's spatial distribution to the truth on tissue-normalized
coordinates, so the **interior density** must be right too.

How it is done, and what it deliberately is not
-----------------------------------------------
The outline is interpolated as a **signed distance function**. Each bracketing
section's occupancy is rasterized to a mask, the mask is converted to an SDF
(negative inside, positive outside), the two SDFs are blended linearly at the
target depth, and the zero level set of the blend is the virtual section's
outline. This is the standard shape-interpolation construction from geometry
processing and it has the property a benchmark needs: the intermediate outline is
genuinely *between* the two — a tissue that is growing produces an intermediate
size, not a union, not an average of two binary masks (which would be a blur with
a fractional boundary), and not a copy of either.

Interior intensity is the z-interpolated smoothed density of the brackets,
restricted to that outline, and points are drawn from it by stratified sampling
so the draw reproduces the intensity rather than a multinomial approximation to
it.

**No optimal transport anywhere.** There is no coupling matrix, no transport
plan, no barycentric or McCann displacement of matched cells, and no velocity
field advecting an anchor slice. Nothing here matches a cell in one section to a
cell in another — the two sections meet only as *fields*, never as point sets in
correspondence. Nor is any flanking slice copied, alpha-blended, or resampled:
the cell count is emergent from the intensity and every position is new.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter


def _grid(xy_all, n_bins, pad=0.04):
    lo = xy_all.min(axis=0)
    hi = xy_all.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    lo = lo - pad * span
    hi = hi + pad * span
    xe = np.linspace(lo[0], hi[0], n_bins + 1)
    ye = np.linspace(lo[1], hi[1], n_bins + 1)
    return xe, ye


def _density(xy, xe, ye, smooth):
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[xe, ye])
    return gaussian_filter(H, sigma=smooth, mode="constant")


def _sdf(mask, spacing=1.0):
    """Signed distance to the mask boundary: negative inside, positive outside."""
    if not mask.any():
        return np.full(mask.shape, 1e3, dtype=np.float64)
    if mask.all():
        return np.full(mask.shape, -1e3, dtype=np.float64)
    inside = distance_transform_edt(mask) * spacing
    outside = distance_transform_edt(~mask) * spacing
    return outside - inside


def interpolate_geometry(xy_lo, xy_hi, w, n_cells, n_bins=96, smooth=2.0,
                         occupancy_quantile=0.55, seed=0, jitter=0.6):
    """Positions for a virtual section at fractional depth ``w`` between two.

    ``w = 0`` reproduces the low bracket's *field* (not its cells), ``w = 1`` the
    high bracket's. Returns ``(xy (n_cells, 2), intensity, xe, ye)``.
    """
    rng = np.random.default_rng(seed)
    xy_lo = np.asarray(xy_lo, dtype=np.float64)
    xy_hi = np.asarray(xy_hi, dtype=np.float64)
    xe, ye = _grid(np.vstack([xy_lo, xy_hi]), n_bins)
    spacing = float(np.mean([np.diff(xe).mean(), np.diff(ye).mean()]))

    d_lo = _density(xy_lo, xe, ye, smooth)
    d_hi = _density(xy_hi, xe, ye, smooth)

    # Outline: blend the two signed distance functions, take the zero level set.
    def mask_of(d):
        pos = d[d > 0]
        thr = np.quantile(pos, occupancy_quantile) if pos.size else 0.0
        return d > max(thr, 1e-12)

    sdf = (1.0 - w) * _sdf(mask_of(d_lo), spacing) + w * _sdf(mask_of(d_hi), spacing)
    mask = sdf <= 0.0
    if not mask.any():                                     # degenerate: fall back
        mask = mask_of((1.0 - w) * d_lo + w * d_hi)

    intensity = np.clip((1.0 - w) * d_lo + w * d_hi, 0.0, None) * mask
    if intensity.sum() <= 0:
        intensity = mask.astype(np.float64)

    xy = _stratified_sample(intensity, xe, ye, n_cells, rng, jitter=jitter)
    return xy, intensity, xe, ye


def _stratified_sample(intensity, xe, ye, n_cells, rng, jitter=0.6):
    """Draw ``n_cells`` positions reproducing ``intensity``.

    Proportional allocation with a randomized remainder, rather than a
    multinomial draw. At the ~1 cell per bin these grids run at, multinomial
    counting noise is comparable to the density signal itself, and it shows up
    directly in the metrics that read local density.
    """
    flat = intensity.ravel().astype(np.float64)
    total = flat.sum()
    if total <= 0:
        flat = np.ones_like(flat)
        total = flat.sum()
    exact = flat * (n_cells / total)
    base = np.floor(exact).astype(int)
    short = int(n_cells - base.sum())
    if short > 0:
        frac = exact - base
        p = frac / frac.sum() if frac.sum() > 0 else None
        extra = rng.choice(len(flat), size=short, replace=True, p=p)
        np.add.at(base, extra, 1)
    elif short < 0:
        full = np.where(base > 0)[0]
        drop = rng.choice(full, size=min(-short, len(full)), replace=False)
        np.add.at(base, drop, -1)

    cells = np.repeat(np.arange(len(flat)), base)
    ix, iy = np.unravel_index(cells, intensity.shape)
    cx = 0.5 * (xe[ix] + xe[ix + 1])
    cy = 0.5 * (ye[iy] + ye[iy + 1])
    dx = np.diff(xe).mean()
    dy = np.diff(ye).mean()
    cx = cx + rng.uniform(-jitter, jitter, size=len(cx)) * dx
    cy = cy + rng.uniform(-jitter, jitter, size=len(cy)) * dy
    return np.column_stack([cx, cy])


def target_cell_count(n_lo, n_hi, w):
    """Cell count at fractional depth ``w`` — linear in the bracket counts.

    Emergent by design: the benchmark never tells a method how many cells the
    held-out section has, and ``paper_cell_count_ratio`` is reported as a
    diagnostic rather than ranked precisely because the count is not a quality
    score.
    """
    return int(round((1.0 - w) * n_lo + w * n_hi))
