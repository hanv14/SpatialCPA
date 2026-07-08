"""
Local density calibration for SpatialCPA-v8.

The benchmark's field / density / dice metrics compare the *spatial cell density*
of the synthesized slice (after a rigid alignment) against the ground truth. The
only leakage-safe estimate of the held-out density is the z-interpolation of the
two flanking slices' density fields, evaluated in the common (training-only
re-registered) frame the bridge already lives in:

    ρ*(x) = (1 − t) · ρ_lower(x) + t · ρ_upper(x)

The symmetric OT bridge already transports density approximately correctly, but a
diffuse entropic plan smooths it. :func:`calibrate` closes the residual gap by
importance-resampling the synthesized cells so their empirical density is nudged
toward ρ*: a cell sitting where the bridge under-produced density (ρ* > current)
is up-weighted, one in an over-dense spot is down-weighted. Because it only
*reselects real synthesized cells* (each still carries a real expression profile
and label), it improves the density-driven metrics without distorting the
expression distribution or the niche.

Densities are estimated with a fast fine-grid + Gaussian-smoothing KDE (O(n)),
with the bandwidth set in units of the tissue's median cell spacing so it is
scale-free across datasets.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _median_spacing(xy):
    if xy.shape[0] < 2:
        return 1.0
    d, _ = cKDTree(xy).query(xy, k=2)
    s = float(np.median(d[:, 1]))
    return s if s > 0 else 1.0


def _grid_density(xy, edges_x, edges_y, sigma_bins):
    """Gaussian-smoothed 2-D histogram density on a fixed grid (unit integral)."""
    from scipy.ndimage import gaussian_filter
    H, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=[edges_x, edges_y])
    H = gaussian_filter(H, sigma=max(sigma_bins, 0.5), mode="nearest")
    s = H.sum()
    return H / s if s > 0 else H


def _sample_grid(field, edges_x, edges_y, xy):
    """Look up ``field`` at each point in ``xy`` (nearest grid cell)."""
    nx, ny = field.shape
    xb = np.clip(np.digitize(xy[:, 0], edges_x) - 1, 0, nx - 1)
    yb = np.clip(np.digitize(xy[:, 1], edges_y) - 1, 0, ny - 1)
    return field[xb, yb]


def calibrate(coords_xy, lo_xy, up_xy, t, cfg, seed=0, grid=48):
    """Resample synthesized cells toward the interpolated flanking density field.

    Returns integer indices into ``coords_xy`` selecting the calibrated cell set
    (length == len(coords_xy); a permutation-with-replacement). ``cfg`` is a
    :class:`~spatialcpav8.config.DensityConfig`.
    """
    n = coords_xy.shape[0]
    if not cfg.enabled or n < 8:
        return np.arange(n)

    coords_xy = np.asarray(coords_xy, dtype=np.float64)
    lo_xy = np.asarray(lo_xy, dtype=np.float64)
    up_xy = np.asarray(up_xy, dtype=np.float64)

    # Common frame spanning all three clouds (bridge lives between the flanks).
    allxy = np.vstack([coords_xy, lo_xy, up_xy])
    lo_e = np.percentile(allxy, 0.5, axis=0)
    hi_e = np.percentile(allxy, 99.5, axis=0)
    pad = 0.02 * (hi_e - lo_e + 1e-9)
    edges_x = np.linspace(lo_e[0] - pad[0], hi_e[0] + pad[0], grid + 1)
    edges_y = np.linspace(lo_e[1] - pad[1], hi_e[1] + pad[1], grid + 1)

    spacing = _median_spacing(coords_xy)
    span = 0.5 * ((edges_x[-1] - edges_x[0]) + (edges_y[-1] - edges_y[0]))
    sigma_bins = cfg.bandwidth_spacings * spacing / (span / grid + 1e-9)

    rho_lo = _grid_density(lo_xy, edges_x, edges_y, sigma_bins)
    rho_up = _grid_density(up_xy, edges_x, edges_y, sigma_bins)
    rho_target = (1.0 - t) * rho_lo + t * rho_up
    rho_cur = _grid_density(coords_xy, edges_x, edges_y, sigma_bins)

    tgt = _sample_grid(rho_target, edges_x, edges_y, coords_xy)
    cur = _sample_grid(rho_cur, edges_x, edges_y, coords_xy)
    cur = np.maximum(cur, 1e-12)

    # Importance weight, tempered by strength and capped so no cell dominates.
    ratio = np.clip(tgt / cur, 1.0 / cfg.resample_cap, cfg.resample_cap)
    w = ratio ** float(cfg.strength)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    w = w / w.sum()

    rng = np.random.default_rng(seed)
    return rng.choice(n, size=n, replace=True, p=w)
