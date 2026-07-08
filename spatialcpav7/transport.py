"""
Fused Gromov-Wasserstein displacement interpolation between two flanking slices.

The biological premise (shared with v6): between two adjacent physical sections
tissue changes *continuously*, so the in-between slice is the displacement
interpolation of the two flanking cell distributions under optimal transport —
not a copy of either.

What is new in v7 is the coupling. v6 used entropic OT under a spatial+molecular
*feature* cost, which matches the two marginals but is blind to the intra-slice
relational geometry: two cells that are neighbours in the lower slice can be sent
to distant cells in the upper slice. The neighbourhood structure is exactly what
Moran's-I agreement and cell-type neighbourhood agreement measure, so v7 solves a
**fused Gromov-Wasserstein (FGW)** problem instead:

    T* = argmin_T  (1-α)·<M, T>  +  α·Σ_ijkl |C_lo[i,k] − C_hi[j,l]|² T_ij T_kl

where ``M`` is the feature cost (physical + cell-state) and ``C_lo``, ``C_hi``
are the intra-slice pairwise-distance matrices. The Gromov term penalizes
distorting the neighbourhood graph, so the FGW map is *relation-preserving*: the
morphed / interpolated slice keeps the tissue's local geometry, not just its
density. Solved with the entropic proximal-gradient scheme (Peyré et al. 2016;
Vayer et al. 2019) — pure numpy/scipy, no POT dependency.

Everything uses the two *training* flanking slices and the scalar target z only,
so it is leakage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


@dataclass
class TransportResult:
    lo_idx: np.ndarray
    up_idx: np.ndarray
    coords_xy: np.ndarray
    t: float
    sub_lo: np.ndarray
    sub_up: np.ndarray


def interpolation_fraction(z: float, z_lo: float, z_hi: float) -> float:
    if z_hi == z_lo:
        return 0.5
    return float(np.clip((float(z) - z_lo) / (z_hi - z_lo), 0.0, 1.0))


def interpolated_count(n_lo: int, n_hi: int, t: float, mode: str = "interpolate") -> int:
    if mode == "lower":
        n = n_lo
    elif mode == "upper":
        n = n_hi
    elif mode == "mean":
        n = 0.5 * (n_lo + n_hi)
    else:
        n = (1.0 - t) * n_lo + t * n_hi
    return max(int(round(n)), 1)


def _subsample(n: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    return np.sort(rng.choice(n, cap, replace=False))


def sinkhorn_plan(cost: np.ndarray, epsilon: float, n_iter: int,
                  a: Optional[np.ndarray] = None,
                  b: Optional[np.ndarray] = None) -> np.ndarray:
    """Entropic-OT plan for ``cost`` with (uniform) marginals ``a``, ``b``.

    The cost is median-scaled so ``epsilon`` is dimensionless.
    """
    n, m = cost.shape
    a = np.full(n, 1.0 / n) if a is None else a / a.sum()
    b = np.full(m, 1.0 / m) if b is None else b / b.sum()
    scale = np.median(np.abs(cost)) + 1e-9
    logK = -cost / (scale * epsilon)
    logK -= logK.max()                     # log-domain shift: prevents exp overflow
    K = np.exp(logK) + 1e-300              # (constant shift cancels in Sinkhorn)
    u = np.ones(n)
    v = np.ones(m)
    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
    P = u[:, None] * K * v[None, :]
    s = P.sum()
    return P / (s if s > 0 else 1.0)


def _feature_cost(Axy, Bxy, Ae, Be, embed_weight):
    """Median-normalized spatial + embedding squared-distance feature cost."""
    Csp = cdist(Axy, Bxy, metric="sqeuclidean")
    Csp = Csp / (np.median(Csp) + 1e-9)
    if embed_weight > 0 and Ae.shape[1] > 0:
        Cem = cdist(Ae, Be, metric="sqeuclidean")
        Cem = Cem / (np.median(Cem) + 1e-9)
        return (1.0 - embed_weight) * Csp + embed_weight * Cem
    return Csp


def _intra_distance(xy):
    """Median-normalized intra-slice euclidean distance matrix (relational geometry)."""
    C = cdist(xy, xy, metric="euclidean")
    return C / (np.median(C) + 1e-9)


def fgw_plan(M, C1, C2, epsilon, alpha, outer_iter, sinkhorn_iter):
    """Entropic fused Gromov-Wasserstein plan (uniform marginals).

    Proximal-gradient outer loop: linearize the Gromov term at the current plan
    ``T`` (square-loss factorization), add the feature cost, and re-solve the
    resulting entropic-OT subproblem. ``alpha`` fuses structure (Gromov) with the
    feature cost; ``alpha = 0`` recovers plain entropic OT.
    """
    n, m = M.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    if alpha <= 0:
        return sinkhorn_plan(M, epsilon, sinkhorn_iter, a, b)
    C1sq, C2sq = C1 ** 2, C2 ** 2
    constC = (C1sq @ a)[:, None] + (C2sq @ b)[None, :]     # (n, m)
    T = a[:, None] * b[None, :]
    for _ in range(outer_iter):
        # gradient of the square-loss Gromov term at T (h1=C1, h2=2·C2).
        tens = constC - C1 @ T @ (2.0 * C2)
        cost = (1.0 - alpha) * M + alpha * tens
        T = sinkhorn_plan(cost, epsilon, sinkhorn_iter, a, b)
    return T


def _deshrink(coords_xy, lo_xy, up_xy, t, strength):
    """Rescale the interpolated cloud to the z-interpolated flanking footprint."""
    target_mean = (1.0 - t) * lo_xy.mean(axis=0) + t * up_xy.mean(axis=0)
    target_std = (1.0 - t) * lo_xy.std(axis=0) + t * up_xy.std(axis=0)
    cur_mean = coords_xy.mean(axis=0)
    cur_std = coords_xy.std(axis=0)
    cur_std = np.where(cur_std > 1e-8, cur_std, 1.0)
    scaled = (coords_xy - cur_mean) / cur_std * target_std + target_mean
    return (1.0 - strength) * coords_xy + strength * scaled


def fgw_morph(
    anchor_xy: np.ndarray, other_xy: np.ndarray,
    anchor_e: np.ndarray, other_e: np.ndarray,
    w: float, cfg, seed: int = 0,
):
    """Coherent single-sheet placement via the fused-GW barycentric map.

    1. Solve the entropic FGW plan ``T`` (anchor × other) under the joint
       feature cost + the Gromov structure term.
    2. Barycentric image of each anchor cell ``i``:
       ``x̂_i = Σ_j P(j|i)·other_xy[j]`` (its FGW-matched location in the other slice).
    3. interpolated position ``(1-w)·x_i + w·x̂_i``.

    Because FGW keeps neighbours together, the morphed sheet preserves the anchor
    slice's local graph while flowing toward the other slice's footprint. One cell
    per anchor cell (no density doubling); expression/labels stay the anchor's
    (real, coherent). Returns ``(coords_xy, anchor_idx, dissimilarity)``.
    """
    rng = np.random.default_rng(seed)
    n_a = anchor_xy.shape[0]
    sub_a = _subsample(n_a, cfg.max_ot_cells, rng)
    sub_b = _subsample(other_xy.shape[0], cfg.max_ot_cells, rng)
    Axy, Bxy = anchor_xy[sub_a], other_xy[sub_b]
    Ae, Be = anchor_e[sub_a], other_e[sub_b]

    M = _feature_cost(Axy, Bxy, Ae, Be, cfg.embed_weight)
    C1 = _intra_distance(Axy)
    C2 = _intra_distance(Bxy)
    P = fgw_plan(M, C1, C2, cfg.epsilon, cfg.alpha_gw, cfg.gw_iter, cfg.n_iter // 4 or 1)

    row = P / (P.sum(axis=1, keepdims=True) + 1e-300)     # P(j | i)
    image = row @ Bxy                                     # OT image of each anchor cell

    if Axy.shape[0] >= 2:
        spacing = float(np.median(cKDTree(Axy).query(Axy, k=2)[0][:, 1])) or 1.0
    else:
        spacing = 1.0
    dissimilarity = float(np.median(np.linalg.norm(image - Axy, axis=1)) / spacing)

    coords = ((1.0 - w) * Axy + w * image).astype(np.float32)
    if cfg.deshrink and coords.shape[0] >= 3:
        coords = _deshrink(coords, Axy, Bxy, w, cfg.deshrink_strength)
    return coords, sub_a, dissimilarity


def fgw_geodesic(
    lo_xy: np.ndarray, up_xy: np.ndarray,
    lo_embed: np.ndarray, up_embed: np.ndarray,
    t: float, n_target: int, cfg, seed: int = 0,
) -> TransportResult:
    """Sample matched pairs from the FGW plan and place at the McCann midpoint."""
    rng = np.random.default_rng(seed)
    n_lo, n_hi = lo_xy.shape[0], up_xy.shape[0]
    sub_lo = _subsample(n_lo, cfg.max_ot_cells, rng)
    sub_up = _subsample(n_hi, cfg.max_ot_cells, rng)
    Alo_xy, Aup_xy = lo_xy[sub_lo], up_xy[sub_up]
    Alo_e, Aup_e = lo_embed[sub_lo], up_embed[sub_up]

    M = _feature_cost(Alo_xy, Aup_xy, Alo_e, Aup_e, cfg.embed_weight)
    C1 = _intra_distance(Alo_xy)
    C2 = _intra_distance(Aup_xy)
    P = fgw_plan(M, C1, C2, cfg.epsilon, cfg.alpha_gw, cfg.gw_iter, cfg.n_iter // 4 or 1)

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
