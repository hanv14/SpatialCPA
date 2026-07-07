"""
Optimal-transport displacement interpolation between two flanking slices.

The biological premise: between two adjacent physical sections, tissue changes
*continuously*. The principled model of the in-between slice is therefore the
displacement interpolation (McCann geodesic) of the two flanking cell
distributions under optimal transport — not a copy of either slice (v5 / SpatialZ)
nor an unstructured union of both.

We compute an entropic-OT coupling ``P`` between the lower- and upper-slice cells
under a cost that combines **physical** distance (the slices are re-registered
into a common frame, so ``(x, y)`` is directly comparable) and **molecular**
distance in the shared cell-state embedding. ``P`` is a soft matching: mass
``P[i, j]`` says lower-cell ``i`` corresponds to upper-cell ``j``. Sampling pairs
from ``P`` and placing each synthesized cell at the interpolated position
``(1-t)·x_i + t·x_j`` yields the geodesic midpoint slice, whose footprint, density
gradient and regional layout lie *between* the two flanking slices — exactly what
the held-out slice is.

A joint spatial+molecular cost matters: pure spatial OT would match a cell to
whatever sits at the same ``(x, y)`` even across a tissue boundary; adding the
embedding term keeps matches molecularly coherent, so the interpolated cell's
type and profile are consistent with its neighbors.

Everything here uses the two *training* flanking slices and the scalar target z
only — no held-out information — so it is leakage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


@dataclass
class TransportResult:
    """Sampled OT pairs and the interpolated in-between cloud."""

    lo_idx: np.ndarray        # (N,) index into the (subsampled) lower slice
    up_idx: np.ndarray        # (N,) index into the (subsampled) upper slice
    coords_xy: np.ndarray     # (N, 2) interpolated positions
    t: float                  # interpolation fraction toward the upper slice
    sub_lo: np.ndarray        # indices of the lower-slice subsample into the full slice
    sub_up: np.ndarray        # indices of the upper-slice subsample into the full slice


def interpolation_fraction(z: float, z_lo: float, z_hi: float) -> float:
    """``t = (z - z_lo) / (z_hi - z_lo)`` clamped to [0, 1] (0.5 if degenerate)."""
    if z_hi == z_lo:
        return 0.5
    return float(np.clip((float(z) - z_lo) / (z_hi - z_lo), 0.0, 1.0))


def interpolated_count(n_lo: int, n_hi: int, t: float, mode: str = "interpolate") -> int:
    """Emergent cell count of the virtual slice (never uses the held-out count)."""
    if mode == "lower":
        n = n_lo
    elif mode == "upper":
        n = n_hi
    elif mode == "mean":
        n = 0.5 * (n_lo + n_hi)
    else:  # interpolate
        n = (1.0 - t) * n_lo + t * n_hi
    return max(int(round(n)), 1)


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, cap, replace=False))


def sinkhorn_plan(cost: np.ndarray, epsilon: float, n_iter: int,
                  a: Optional[np.ndarray] = None,
                  b: Optional[np.ndarray] = None) -> np.ndarray:
    """Entropic-OT plan for a cost matrix with (uniform) marginals ``a``, ``b``.

    Returns the coupling ``P`` (normalized to sum 1). The cost is median-scaled
    so ``epsilon`` is dimensionless and behaves consistently across datasets.
    """
    n, m = cost.shape
    a = np.full(n, 1.0 / n) if a is None else a / a.sum()
    b = np.full(m, 1.0 / m) if b is None else b / b.sum()
    scale = np.median(cost) + 1e-9
    K = np.exp(-cost / (scale * epsilon)) + 1e-300
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
    P = u[:, None] * K * v[None, :]
    s = P.sum()
    return P / (s if s > 0 else 1.0)


def _deshrink(coords_xy, lo_xy, up_xy, t, strength):
    """Rescale the interpolated cloud to the z-interpolated flanking footprint.

    Barycentric interpolation of a diffuse OT plan can contract the cloud toward
    its centroid (shrinking the footprint and inflating local density). We
    correct that by matching the cloud's per-axis mean and standard deviation to
    the z-interpolated statistics of the two flanking slices — the leakage-safe
    estimate of the true intermediate extent. This keeps the density/field
    metrics well-posed and stops the evaluation aligner from settling on a
    degenerate scaled pose.
    """
    target_mean = (1.0 - t) * lo_xy.mean(axis=0) + t * up_xy.mean(axis=0)
    target_std = (1.0 - t) * lo_xy.std(axis=0) + t * up_xy.std(axis=0)
    cur_mean = coords_xy.mean(axis=0)
    cur_std = coords_xy.std(axis=0)
    cur_std = np.where(cur_std > 1e-8, cur_std, 1.0)
    scaled = (coords_xy - cur_mean) / cur_std * target_std + target_mean
    return (1.0 - strength) * coords_xy + strength * scaled


def barycentric_interpolate(
    anchor_xy: np.ndarray, other_xy: np.ndarray,
    anchor_e: np.ndarray, other_e: np.ndarray,
    w: float, cfg, seed: int = 0,
):
    """Coherent single-sheet placement via the barycentric OT map.

    Random real-cell *mixing* interleaves two offset lattices, so a synthesized
    cell's local neighborhood contains cells from both flanking slices — which
    attenuates spatial autocorrelation and gene-gene structure (the ``morans`` /
    ``coexpression`` / ``nhood`` metrics) whenever adjacent sections are already
    near-identical (e.g. thin volumetric z-planes). This instead morphs the
    *anchor* slice into the other along the optimal-transport map, producing **one**
    coherent cell sheet:

    1. entropic-OT plan ``P`` (anchor × other) under the joint spatial+molecular
       cost, peaked so the map is near-deterministic;
    2. barycentric image of each anchor cell ``i``:
       ``x̂_i = Σ_j P(j|i)·other_xy[j]`` (its OT-matched location in the other slice);
    3. interpolated position ``(1-w)·x_i + w·x̂_i``.

    The displacement ``w·(x̂_i − x_i)`` scales with how different the two slices are,
    so the method **auto-adapts**: ≈ a coherent copy of the anchor when the slices
    are near-identical (matching a single-slice copy on the coherence metrics), and
    a genuine morph toward the intermediate footprint when they differ. One cell per
    anchor cell → no density doubling. Returns interpolated ``coords_xy`` and the
    anchor-cell index each came from (so expression/labels stay real and coherent).
    """
    rng = np.random.default_rng(seed)
    n_a = anchor_xy.shape[0]
    sub_a = _subsample(n_a, cfg.max_ot_cells, rng)
    sub_b = _subsample(other_xy.shape[0], cfg.max_ot_cells, rng)
    Axy, Bxy = anchor_xy[sub_a], other_xy[sub_b]
    Ae, Be = anchor_e[sub_a], other_e[sub_b]

    Csp = cdist(Axy, Bxy, metric="sqeuclidean")
    Csp = Csp / (np.median(Csp) + 1e-9)
    if cfg.embed_weight > 0 and Ae.shape[1] > 0:
        Cem = cdist(Ae, Be, metric="sqeuclidean")
        Cem = Cem / (np.median(Cem) + 1e-9)
        cost = (1.0 - cfg.embed_weight) * Csp + cfg.embed_weight * Cem
    else:
        cost = Csp

    P = sinkhorn_plan(cost, cfg.epsilon, cfg.n_iter)
    row = P / (P.sum(axis=1, keepdims=True) + 1e-300)     # P(j | i)
    image = row @ Bxy                                     # (n_sub_a, 2) OT image

    # Dissimilarity signal for the "adaptive" placement: how far each anchor cell is
    # transported by the OT map, in units of the anchor's cell spacing. On thin
    # volumetric z-planes (STARmap) adjacent sections are nearly registered, so cells
    # barely move (small value) and a coherent morph is best; on distinct tissue
    # sections (IMC) cells map far (large value) and the barycentric map contracts,
    # so both-slice interpolation is the better estimate. Logged by the wrapper so
    # the threshold can be verified per dataset.
    if Axy.shape[0] >= 2:
        spacing = float(np.median(cKDTree(Axy).query(Axy, k=2)[0][:, 1])) or 1.0
    else:
        spacing = 1.0
    dissimilarity = float(np.median(np.linalg.norm(image - Axy, axis=1)) / spacing)

    coords = ((1.0 - w) * Axy + w * image).astype(np.float32)
    if cfg.deshrink and coords.shape[0] >= 3:
        # Guard against barycentric contraction: match the interpolated footprint.
        coords = _deshrink(coords, Axy, Bxy, w, cfg.deshrink_strength)
    return coords, sub_a, dissimilarity


def transport_interpolate(
    lo_xy: np.ndarray, up_xy: np.ndarray,
    lo_embed: np.ndarray, up_embed: np.ndarray,
    t: float, n_target: int, cfg, seed: int = 0,
) -> TransportResult:
    """Build the OT plan, sample pairs, and place ``n_target`` interpolated cells.

    ``cfg`` is a :class:`~spatialcpav6.config.TransportConfig`.
    """
    rng = np.random.default_rng(seed)
    n_lo, n_hi = lo_xy.shape[0], up_xy.shape[0]
    sub_lo = _subsample(n_lo, cfg.max_ot_cells, rng)
    sub_up = _subsample(n_hi, cfg.max_ot_cells, rng)
    Alo_xy, Aup_xy = lo_xy[sub_lo], up_xy[sub_up]
    Alo_e, Aup_e = lo_embed[sub_lo], up_embed[sub_up]

    # Joint cost: median-normalized spatial + embedding squared distances.
    Csp = cdist(Alo_xy, Aup_xy, metric="sqeuclidean")
    Csp = Csp / (np.median(Csp) + 1e-9)
    if cfg.embed_weight > 0 and Alo_e.shape[1] > 0:
        Cem = cdist(Alo_e, Aup_e, metric="sqeuclidean")
        Cem = Cem / (np.median(Cem) + 1e-9)
        cost = (1.0 - cfg.embed_weight) * Csp + cfg.embed_weight * Cem
    else:
        cost = Csp

    P = sinkhorn_plan(cost, cfg.epsilon, cfg.n_iter)

    # Sample n_target matched pairs proportional to the transport mass.
    flat = P.ravel()
    flat = flat / flat.sum()
    m = Aup_xy.shape[0]
    draws = rng.choice(flat.size, size=n_target, replace=True, p=flat)
    li = draws // m
    ui = draws % m

    coords = (1.0 - t) * Alo_xy[li] + t * Aup_xy[ui]
    if cfg.deshrink and coords.shape[0] >= 3:
        coords = _deshrink(coords, Alo_xy, Aup_xy, t, cfg.deshrink_strength)

    return TransportResult(
        lo_idx=sub_lo[li], up_idx=sub_up[ui],
        coords_xy=coords.astype(np.float32), t=float(t),
        sub_lo=sub_lo, sub_up=sub_up,
    )
