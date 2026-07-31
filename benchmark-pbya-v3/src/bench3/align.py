"""Pose selection for v3's alignment-dependent metrics and figures.

Two of the paper metrics compare *positions* — the binned marker field and the
per-cell-type localization — so the prediction has to be brought into the ground
truth's frame first. v2's ``align_prediction_to_gt`` does that by trying a set of
rotations plus a reflection and keeping whichever pose covers the most GT cells.

That works when the prediction sits nearly on top of the GT cells, but it fails on
tissue whose outline is symmetric. A bilaterally symmetric coronal section (MERFISH
hypothalamus) covers itself equally well under identity, 180° and both mirrors, so
occupancy cannot separate them and the choice falls to noise. Measured on
hypothalamus-shaped clouds, a wrong pose was picked in 9 of 12 runs, driving
``marker_field_r`` to about −0.96 where the correct pose gives +0.97 — and the
marker figures came out visibly flipped.

This module fixes both halves of that:

**Expression breaks the symmetry that geometry cannot.** Candidate poses are scored
by how well the *marker fields* agree, not by how many cells overlap. A 180° flip
maps the outline onto itself but inverts the marker gradient, so it scores about
−1 instead of +1 and is rejected outright.

**Reflections are not candidates.** Physical tissue has fixed handedness: no rigid
re-registration of the same specimen can mirror it, so a mirrored pose is wrong by
construction however well it fits. Only proper rotations (det = +1) are considered.

Ties within ``TIE_TOL`` resolve toward the smallest rotation, so an already-aligned
dataset keeps the identity pose instead of drifting into an equivalent one.

Like v2's, this is an evaluation-side operation: it reads the ground truth, which
evaluation is allowed to do, and feeds nothing back to the method.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .config import ALIGN_ANGLES, ALIGN_ICP_ITERS, ALIGN_MAX_POINTS, FIELD_GRID

TIE_TOL = 0.01          # score difference treated as a tie, resolved toward identity
REFINE_MARGIN = 0.005   # gain the fine angular scan must beat to move the pose


# ── binned fields (shared with evaluate_paper) ───────────────────────────────

def binned_gene_field(xy, values, xe, ye, grid):
    """Mean value per occupied bin of a ``grid`` x ``grid`` lattice."""
    xb = np.clip(np.digitize(xy[:, 0], xe) - 1, 0, grid - 1)
    yb = np.clip(np.digitize(xy[:, 1], ye) - 1, 0, grid - 1)
    flat = yb * grid + xb
    sums = np.zeros(grid * grid)
    cnts = np.zeros(grid * grid)
    np.add.at(sums, flat, values)
    np.add.at(cnts, flat, 1.0)
    occ = cnts > 0
    means = np.zeros(grid * grid)
    means[occ] = sums[occ] / cnts[occ]
    return means, occ


def _field_agreement(pred_xy, pred_V, gt_xy, gt_V, grid):
    """Mean Pearson r between binned fields, over the scoring channels."""
    allxy = np.vstack([pred_xy, gt_xy])
    xe = np.linspace(allxy[:, 0].min(), allxy[:, 0].max(), grid + 1)
    ye = np.linspace(allxy[:, 1].min(), allxy[:, 1].max(), grid + 1)
    rs = []
    for j in range(pred_V.shape[1]):
        pm, po = binned_gene_field(pred_xy, pred_V[:, j], xe, ye, grid)
        gm, go = binned_gene_field(gt_xy, gt_V[:, j], xe, ye, grid)
        both = po & go
        if both.sum() >= 4 and pm[both].std() > 0 and gm[both].std() > 0:
            rs.append(float(np.corrcoef(pm[both], gm[both])[0, 1]))
    return float(np.mean(rs)) if rs else np.nan


def _coverage(pred_xy, gt_xy, radius):
    """Fraction of GT cells with a predicted cell within ``radius`` — the tie-break."""
    if len(pred_xy) == 0:
        return 0.0
    d, _ = cKDTree(pred_xy).query(gt_xy)
    return float((d <= radius).mean())


# ── rigid fitting, proper rotations only ─────────────────────────────────────

def _kabsch(A, B):
    """Rotation with det=+1 taking centred ``A`` onto centred ``B``."""
    U, _, Vt = np.linalg.svd(A.T @ B)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, d]) @ U.T


def _icp(src, dst, iters, with_scale=True):
    """Point-to-point ICP restricted to proper rotations. Returns the moved points.

    The transform is recovered afterwards by fitting source -> moved in one step
    (:func:`_fit_similarity`), which is exact because a composition of
    similarities is a similarity — and far less error-prone than accumulating
    rotations, scales and translations across iterations.
    """
    tree = cKDTree(dst)
    cur = np.asarray(src, dtype=np.float64)
    for _ in range(iters):
        _, idx = tree.query(cur)
        tgt = dst[idx]
        cm, tm = cur.mean(axis=0), tgt.mean(axis=0)
        A, B = cur - cm, tgt - tm
        R = _kabsch(A, B)
        denom = float((A * A).sum())
        s = float((B * (A @ R.T)).sum() / denom) if (with_scale and denom > 0) else 1.0
        cur = (A @ R.T) * s + tm
    return cur


def _fit_similarity(src, moved, with_scale=True):
    """The (R, s, t) taking ``src`` onto ``moved``; rotation only, never a mirror."""
    sc, mc = src.mean(axis=0), moved.mean(axis=0)
    A, B = src - sc, moved - mc
    R = _kabsch(A, B)
    denom = float((A * A).sum())
    s = float((B * (A @ R.T)).sum() / denom) if (with_scale and denom > 0) else 1.0
    t = mc - (sc @ R.T) * s
    return R, s, t


def _apply(xy, R, s, t):
    return (xy @ R.T) * s + t


def _rotate_about(xy, centre, deg):
    th = np.radians(deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    return (xy - centre) @ R.T + centre


# ── the aligner ───────────────────────────────────────────────────────────────

def align_by_expression(pred_xy, pred_V, gt_xy, gt_V, n_angles=ALIGN_ANGLES,
                        grid=FIELD_GRID, icp_iters=ALIGN_ICP_ITERS,
                        max_points=ALIGN_MAX_POINTS, with_scale=True, seed=0):
    """Align ``pred_xy`` onto ``gt_xy``, choosing the pose by field agreement.

    ``pred_V`` / ``gt_V`` are the (n, k) scoring channels — the marker genes,
    already rank-normalized and column-matched between prediction and GT.

    Returns ``(aligned_xy, info)``; ``info`` records the chosen rotation in
    degrees, the scale, the winning score and the runner-up, so a marginal
    decision is visible rather than silent.
    """
    pred_xy = np.asarray(pred_xy, dtype=np.float64)
    gt_xy = np.asarray(gt_xy, dtype=np.float64)
    if len(pred_xy) < 3 or len(gt_xy) < 3:
        # Too few cells to fit anything; still move the prediction onto the GT
        # centroid, so it is not left in whatever frame the method emitted.
        shift = gt_xy.mean(axis=0) - pred_xy.mean(axis=0)
        return pred_xy + shift, {"align_rotation_deg": 0.0, "align_scale": 1.0,
                                 "align_score": None, "align_runner_up": None,
                                 "align_method": "centroid-only (too few cells)"}

    rng = np.random.default_rng(seed)

    def sub(xy, V):
        if len(xy) <= max_points:
            return xy, V
        i = rng.choice(len(xy), max_points, replace=False)
        return xy[i], V[i]

    p_xy, p_V = sub(pred_xy, np.asarray(pred_V, dtype=np.float64))
    g_xy, g_V = sub(gt_xy, np.asarray(gt_V, dtype=np.float64))

    p_c, g_c = p_xy.mean(axis=0), g_xy.mean(axis=0)
    p_rad = float(np.sqrt(((p_xy - p_c) ** 2).sum(axis=1)).mean())
    g_rad = float(np.sqrt(((g_xy - g_c) ** 2).sum(axis=1)).mean())
    s0 = (g_rad / p_rad) if (with_scale and p_rad > 0) else 1.0

    # tie-break radius: a few times the GT nearest-neighbour spacing
    d_gt, _ = cKDTree(g_xy).query(g_xy, k=2)
    radius = 3.0 * float(np.median(d_gt[:, 1]))

    candidates = []
    for ang in np.linspace(0.0, 360.0, int(n_angles), endpoint=False):
        th = np.radians(ang)
        R0 = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        t0 = g_c - (p_c @ R0.T) * s0
        start = _apply(p_xy, R0, s0, t0)

        # Both the un-refined start and its ICP refinement are scored: ICP can
        # drift away from a correct pose into a nearby wrong one, so neither
        # should be trusted on its own.
        poses = [start]
        if icp_iters > 0:
            poses.append(_icp(start, g_xy, icp_iters, with_scale=with_scale))

        for moved in poses:
            score = _field_agreement(moved, p_V, g_xy, g_V, grid)
            if not np.isfinite(score):
                continue
            R, s, t = _fit_similarity(p_xy, moved, with_scale=with_scale)
            candidates.append({
                "angle": float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360.0),
                "score": score,
                "coverage": _coverage(moved, g_xy, radius),
                "R": R, "s": s, "t": t,
            })

    if not candidates:      # no usable scoring channel — keep the prediction as-is
        R, s = np.eye(2), s0
        t = g_c - (p_c @ R.T) * s
        return _apply(pred_xy, R, s, t), {
            "align_rotation_deg": 0.0, "align_scale": float(s),
            "align_score": None, "align_runner_up": None,
            "align_method": "centroid-only (no scoring channel)"}

    best = max(c["score"] for c in candidates)
    near = [c for c in candidates if c["score"] >= best - TIE_TOL]
    # among statistically indistinguishable poses prefer the smallest rotation,
    # so an already-aligned dataset is not nudged into an equivalent pose
    pick = min(near, key=lambda c: (min(c["angle"], 360.0 - c["angle"]),
                                    -c["coverage"]))
    runner = sorted((c["score"] for c in candidates), reverse=True)

    # The candidate grid is coarse (360/n_angles), and ICP has little rotational
    # signal on a near-elliptical outline, so the winner can sit up to half a grid
    # step off. A short fine scan, scored the same way, removes that quantization.
    # A refinement has to *earn* its way in (``REFINE_MARGIN``): accepting any
    # improvement would let bin-level noise nudge an already-correct pose off
    # true, which costs the depth profile more than the field gains.
    moved = _apply(p_xy, pick["R"], pick["s"], pick["t"])
    half = 180.0 / int(n_angles)
    best_extra, floor = 0.0, pick["score"] + REFINE_MARGIN
    for delta in np.arange(-half, half + 1e-9, half / 5.0):
        if abs(delta) < 1e-9:
            continue
        cand = _rotate_about(moved, moved.mean(axis=0), delta)
        sc = _field_agreement(cand, p_V, g_xy, g_V, grid)
        if np.isfinite(sc) and sc > max(floor, pick["score"]):
            pick["score"], best_extra = sc, delta
    if best_extra:
        moved = _rotate_about(moved, moved.mean(axis=0), best_extra)
        R, s, t = _fit_similarity(p_xy, moved, with_scale=with_scale)
        pick.update({"R": R, "s": s, "t": t,
                     "angle": float(np.degrees(np.arctan2(R[1, 0], R[0, 0])) % 360.0),
                     "coverage": _coverage(moved, g_xy, radius)})

    info = {
        "align_rotation_deg": round(min(pick["angle"], pick["angle"] - 360.0,
                                        key=abs), 2),
        "align_scale": round(float(pick["s"]), 4),
        "align_score": round(float(pick["score"]), 4),
        "align_runner_up": round(float(runner[1]), 4) if len(runner) > 1 else None,
        "align_coverage": round(float(pick["coverage"]), 4),
        "align_method": "expression-scored rotation (no reflections)",
    }
    return _apply(pred_xy, pick["R"], pick["s"], pick["t"]), info


def scoring_columns(gene_names, markers, gt_xy, gt_R, k=3, spatial_k=10):
    """Columns to score poses on: the marker genes, else the most structured ones.

    Falling back to the top-k genes by ground-truth Moran's I keeps the aligner
    working for a dataset whose marker panel resolved empty — those genes carry
    the strongest spatial signal available, which is exactly what a correct pose
    has to reproduce and a flipped one destroys.
    """
    names = list(gene_names)
    cols = [names.index(g) for g in markers if g in names]
    if cols:
        return cols, "markers"

    from benchmark.evaluate_generation import _morans_i
    mi = _morans_i(gt_xy, gt_R, spatial_k)
    if not np.isfinite(mi).any():
        return [], "none"
    order = np.argsort(np.where(np.isfinite(mi), mi, -np.inf))[::-1]
    return [int(i) for i in order[:k]], "top-morans"
