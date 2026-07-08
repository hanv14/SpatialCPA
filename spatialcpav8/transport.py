"""
Symmetric optimal-transport bridge between two flanking slices (SpatialCPA-v8).

Biological premise (shared with v6): between two adjacent physical sections the
tissue changes *continuously*, so the principled model of the in-between slice is
the displacement interpolation (McCann geodesic) of the two flanking cell
distributions under optimal transport.

What is new in v8 is *how the geodesic midpoint is discretized into cells*. v6
had to pick, per holdout, between two imperfect discretizations:

* **one-sided morph** — displace every cell of the *nearest* slice along the OT
  map. Coherent (a single sheet, so spatial autocorrelation and local
  neighborhoods survive), but the synthesized population is drawn from **one**
  flanking slice only, so mixture-sensitive metrics (composition, co-expression,
  distribution) reflect that single slice rather than the true intermediate mix.
* **random interpolation** — sample matched pairs and jitter between them. Right
  mixture, but incoherent: a cell's neighbors come from unrelated parts of the two
  slices, which attenuates spatial structure.

v8's :func:`symmetric_bridge` removes the trade-off. It builds the barycentric
image of **both** slices under the *same* entropic-OT plan and draws cells from
the two coherent projected sheets in the z-interpolated ratio ``(1-t) : t``:

    lower cell i  ->  x_i + t · (barycentric_image_of_i_in_upper − x_i)
    upper cell j  ->  x_j + (1−t) · (barycentric_image_of_j_in_lower − x_j)

Each projected sheet is coherent (a smooth barycentric displacement of a real
slice, one cell per source cell — no density doubling), and their union in the
right ratio is the correct mixture. This is exactly the two "halves" of the
McCann interpolant that a one-sided morph throws away, recombined.

Everything uses the two *training* flanking slices and the scalar target z only —
no held-out information — so it is leakage-safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


@dataclass
class BridgeResult:
    """A synthesized in-between sheet: coherent positions from both slices."""

    coords_xy: np.ndarray     # (M, 2) interpolated positions
    lo_src: np.ndarray        # (M,) index into the lower slice (−1 if from upper)
    up_src: np.ndarray        # (M,) index into the upper slice (−1 if from lower)
    from_upper: np.ndarray    # (M,) bool: True if the cell's source is the upper slice
    dissimilarity: float      # OT-map displacement in cell-spacings (diagnostic)


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


def sinkhorn_plan(cost: np.ndarray, epsilon: float, n_iter: int) -> np.ndarray:
    """Entropic-OT plan for a cost matrix with uniform marginals (sum-1 coupling).

    The cost is median-scaled so ``epsilon`` is dimensionless and behaves
    consistently across datasets.
    """
    n, m = cost.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
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


def _joint_cost(Axy, Bxy, Ae, Be, embed_weight):
    """Median-normalized spatial + embedding squared-distance cost."""
    Csp = cdist(Axy, Bxy, metric="sqeuclidean")
    Csp = Csp / (np.median(Csp) + 1e-9)
    if embed_weight > 0 and Ae.shape[1] > 0 and Be.shape[1] > 0:
        Cem = cdist(Ae, Be, metric="sqeuclidean")
        Cem = Cem / (np.median(Cem) + 1e-9)
        return (1.0 - embed_weight) * Csp + embed_weight * Cem
    return Csp


def _deshrink(coords_xy, lo_xy, up_xy, t, strength):
    """Rescale the interpolated cloud to the z-interpolated flanking footprint.

    Barycentric projection of a diffuse OT plan can contract the cloud toward its
    centroid (shrinking the footprint, inflating local density). We correct that
    by matching the cloud's per-axis mean and standard deviation to the
    z-interpolated statistics of the two flanking slices — the leakage-safe
    estimate of the true intermediate extent.
    """
    target_mean = (1.0 - t) * lo_xy.mean(axis=0) + t * up_xy.mean(axis=0)
    target_std = (1.0 - t) * lo_xy.std(axis=0) + t * up_xy.std(axis=0)
    cur_mean = coords_xy.mean(axis=0)
    cur_std = coords_xy.std(axis=0)
    cur_std = np.where(cur_std > 1e-8, cur_std, 1.0)
    scaled = (coords_xy - cur_mean) / cur_std * target_std + target_mean
    return (1.0 - strength) * coords_xy + strength * scaled


def _barycentric_image(P_rows, target_xy):
    """Barycentric image ``x̂_i = Σ_j P(j|i)·target_xy[j]`` for each source row."""
    row = P_rows / (P_rows.sum(axis=1, keepdims=True) + 1e-300)
    return row @ target_xy


def symmetric_bridge(lo_xy, up_xy, lo_e, up_e, t, n_target, tcfg, bcfg, seed=0):
    """Bidirectional McCann barycentric bridge (the v8 default placement).

    Projects both flanking slices through the *same* entropic-OT plan and draws
    ``n_target`` cells from the two coherent projected sheets in the ratio
    ``(1-t) : t``. Returns a :class:`BridgeResult` whose ``lo_src`` / ``up_src``
    index the *full* flanking slices so expression and labels stay real.
    """
    rng = np.random.default_rng(seed)
    n_lo, n_up = lo_xy.shape[0], up_xy.shape[0]
    sub_lo = _subsample(n_lo, tcfg.max_ot_cells, rng)
    sub_up = _subsample(n_up, tcfg.max_ot_cells, rng)
    Alo, Aup = lo_xy[sub_lo].astype(np.float64), up_xy[sub_up].astype(np.float64)
    Elo, Eup = lo_e[sub_lo], up_e[sub_up]

    cost = _joint_cost(Alo, Aup, Elo, Eup, tcfg.embed_weight)
    P = sinkhorn_plan(cost, tcfg.epsilon, tcfg.n_iter)              # (nlo, nup)

    # Barycentric images under the shared plan (both directions).
    img_lo = _barycentric_image(P, Aup)              # lower cell -> its upper location
    img_up = _barycentric_image(P.T, Alo)            # upper cell -> its lower location

    # McCann displacement toward the target z from each side.
    coords_lo = (1.0 - t) * Alo + t * img_lo         # lower cells moved by fraction t
    coords_up = (1.0 - t) * img_up + t * Aup         # upper cells moved by fraction 1−t

    # Diagnostic: OT-map displacement between the flanking slices, in cell-spacings.
    if Alo.shape[0] >= 2:
        spacing = float(np.median(cKDTree(Alo).query(Alo, k=2)[0][:, 1])) or 1.0
    else:
        spacing = 1.0
    dissimilarity = float(np.median(np.linalg.norm(img_lo - Alo, axis=1)) / spacing)

    # Draw from the two coherent sheets in the z-interpolated ratio, guarding the
    # minority side so a near-integer t still carries a few far-slice cells.
    fmin = float(bcfg.symmetric_min_fraction)
    frac_up = float(np.clip(t, fmin, 1.0 - fmin)) if 0.0 < t < 1.0 else float(np.clip(t, 0.0, 1.0))
    take_up = int(round(frac_up * n_target))
    take_lo = n_target - take_up

    def _draw(coords, sub_full, take):
        if take <= 0 or coords.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros(0, int)
        pick = rng.integers(0, coords.shape[0], size=take)
        return coords[pick], sub_full[pick]

    c_lo, src_lo = _draw(coords_lo, sub_lo, take_lo)
    c_up, src_up = _draw(coords_up, sub_up, take_up)

    coords = np.concatenate([c_lo, c_up], axis=0)
    from_upper = np.concatenate([
        np.zeros(c_lo.shape[0], bool), np.ones(c_up.shape[0], bool)])
    lo_src = np.concatenate([src_lo, np.full(c_up.shape[0], -1, int)])
    up_src = np.concatenate([np.full(c_lo.shape[0], -1, int), src_up])

    if tcfg.deshrink and coords.shape[0] >= 3:
        coords = _deshrink(coords, Alo, Aup, t, tcfg.deshrink_strength)

    return BridgeResult(coords_xy=coords.astype(np.float32), lo_src=lo_src,
                        up_src=up_src, from_upper=from_upper,
                        dissimilarity=dissimilarity)


def _smooth_field(xy, vec, k, n_iter):
    """Spatially smooth a per-cell vector field over the kNN graph.

    Averaging each cell's vector with its spatial neighbors a few times turns a
    noisy per-cell field into a coherent one: neighboring cells share almost the
    same value, so applying it as a displacement moves local patches *together*
    (a near-isometric tissue deformation) instead of scattering cells
    independently. That is what preserves local neighborhoods — and hence Moran's
    I, co-expression and niche structure — while still morphing the global shape.
    """
    n = xy.shape[0]
    if n < k + 1 or n_iter <= 0:
        return vec
    _, nn = cKDTree(xy).query(xy, k=min(k + 1, n))
    out = vec
    for _ in range(n_iter):
        out = out[nn].mean(axis=1)          # include self (nn[:,0]) for stability
    return out


def smooth_morph(anchor_xy, other_xy, anchor_e, other_e, w, tcfg, bcfg, seed=0):
    """Smoothed-OT morph of the anchor slice (the v8 default placement).

    Like a one-sided barycentric morph — copy the *anchor* slice's real cells (so
    their expression, labels and local structure stay exactly real) and displace
    them along the entropic-OT map toward the other slice — but the barycentric
    displacement field is **spatially smoothed** first (:func:`_smooth_field`) so
    the morph is a coherent, near-isometric tissue deformation rather than a
    per-cell scatter. This keeps the position-coupled structure metrics
    (``morans`` / ``coexpression`` / ``nhood``) at copy-quality while the smooth
    warp still moves the global footprint toward the interpolated in-between shape
    (helping the ``field`` / ``density`` metrics). One cell per anchor cell → no
    density doubling.

    Returns ``(coords_xy, anchor_index, dissimilarity)``.
    """
    rng = np.random.default_rng(seed)
    sub_a = _subsample(anchor_xy.shape[0], tcfg.max_ot_cells, rng)
    sub_b = _subsample(other_xy.shape[0], tcfg.max_ot_cells, rng)
    Axy, Bxy = anchor_xy[sub_a].astype(np.float64), other_xy[sub_b].astype(np.float64)
    Ae, Be = anchor_e[sub_a], other_e[sub_b]
    cost = _joint_cost(Axy, Bxy, Ae, Be, tcfg.embed_weight)
    P = sinkhorn_plan(cost, tcfg.epsilon, tcfg.n_iter)
    image = _barycentric_image(P, Bxy)                    # OT target location
    disp = image - Axy                                    # raw per-cell displacement
    if Axy.shape[0] >= 2:
        spacing = float(np.median(cKDTree(Axy).query(Axy, k=2)[0][:, 1])) or 1.0
    else:
        spacing = 1.0
    dissimilarity = float(np.median(np.linalg.norm(disp, axis=1)) / spacing)
    # Spatially smooth the displacement into a coherent deformation, then morph.
    disp_s = _smooth_field(Axy, disp, bcfg.smooth_k, bcfg.smooth_iters)
    coords = (Axy + w * disp_s).astype(np.float32)
    if tcfg.deshrink and coords.shape[0] >= 3:
        coords = _deshrink(coords, Axy, Bxy, w, tcfg.deshrink_strength)
    return coords, sub_a, dissimilarity


def one_sided_morph(anchor_xy, other_xy, anchor_e, other_e, w, tcfg, seed=0):
    """One-sided barycentric morph of the anchor slice (v6-style; ablation).

    Returns ``(coords_xy, anchor_index, dissimilarity)``.
    """
    rng = np.random.default_rng(seed)
    sub_a = _subsample(anchor_xy.shape[0], tcfg.max_ot_cells, rng)
    sub_b = _subsample(other_xy.shape[0], tcfg.max_ot_cells, rng)
    Axy, Bxy = anchor_xy[sub_a].astype(np.float64), other_xy[sub_b].astype(np.float64)
    Ae, Be = anchor_e[sub_a], other_e[sub_b]
    cost = _joint_cost(Axy, Bxy, Ae, Be, tcfg.embed_weight)
    P = sinkhorn_plan(cost, tcfg.epsilon, tcfg.n_iter)
    image = _barycentric_image(P, Bxy)
    if Axy.shape[0] >= 2:
        spacing = float(np.median(cKDTree(Axy).query(Axy, k=2)[0][:, 1])) or 1.0
    else:
        spacing = 1.0
    dissimilarity = float(np.median(np.linalg.norm(image - Axy, axis=1)) / spacing)
    coords = ((1.0 - w) * Axy + w * image).astype(np.float32)
    if tcfg.deshrink and coords.shape[0] >= 3:
        coords = _deshrink(coords, Axy, Bxy, w, tcfg.deshrink_strength)
    return coords, sub_a, dissimilarity
