"""
Leakage-safe internal cross-validation for placement selection (SpatialCPA-v8).

Which placement is best is genuinely dataset-dependent: on near-identical
volumetric z-planes a coherent single-sheet morph reproduces one clean real slice
(winning the structure/density metrics), whereas on distinct tissue sections
real-cell interpolation from both slices better estimates the intermediate
(winning the field/cell-matched metrics). A fixed rule (v6/v8's displacement
threshold) can only approximate this boundary.

Instead we *measure* it. We hold out a **middle training slice** — one that itself
has two flanking training slices — regenerate it from those flanks with each
candidate placement, and score the reconstruction against the real (held-out-only-
from-training) slice. The placement that best reconstructs a *known* training slice
is used for the actual target. This is ordinary model selection on the training
data: the benchmark's held-out slice is never touched, so it introduces no
leakage, and it automatically adapts to each dataset's geometry.

The score aggregates the same quantities the benchmark rewards, computed here
without any rigid alignment because both the reconstruction and the real training
slice live in the *same* training-only re-registered frame:

  * co-expression agreement  (gene-gene correlation structure; scale-fair),
  * density agreement        (binned cell-count correlation),
  * field agreement          (binned mean-expression correlation),
  * composition agreement    (cell-type proportion overlap, when labels exist).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, rankdata


def _rank_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return X
    R = np.empty_like(X)
    for g in range(X.shape[1]):
        R[:, g] = rankdata(X[:, g], method="average")
    return (R - 0.5) / X.shape[0]


def _coexpr_agreement(pred_X, real_X, max_genes=150):
    G = pred_X.shape[1]
    if G < 3:
        return np.nan
    if G > max_genes:
        sel = np.linspace(0, G - 1, max_genes).astype(int)
        pred_X, real_X = pred_X[:, sel], real_X[:, sel]

    def upper(X):
        C = np.corrcoef(_rank_normalize(X), rowvar=False)
        return C[np.triu_indices_from(C, k=1)]

    a, b = upper(pred_X), upper(real_X)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 3 or a[m].std() == 0 or b[m].std() == 0:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def _binned(xy, edges_x, edges_y, values=None):
    nx, ny = len(edges_x) - 1, len(edges_y) - 1
    xb = np.clip(np.digitize(xy[:, 0], edges_x) - 1, 0, nx - 1)
    yb = np.clip(np.digitize(xy[:, 1], edges_y) - 1, 0, ny - 1)
    flat = yb * nx + xb
    counts = np.bincount(flat, minlength=nx * ny).astype(float)
    if values is None:
        return counts
    sums = np.bincount(flat, weights=values, minlength=nx * ny)
    means = np.zeros_like(sums)
    occ = counts > 0
    means[occ] = sums[occ] / counts[occ]
    return counts, means, occ


def _density_and_field(pred_xy, pred_X, real_xy, real_X, grid=20):
    allxy = np.vstack([pred_xy, real_xy])
    ex = np.linspace(allxy[:, 0].min(), allxy[:, 0].max(), grid + 1)
    ey = np.linspace(allxy[:, 1].min(), allxy[:, 1].max(), grid + 1)
    pc = _binned(pred_xy, ex, ey)
    rc = _binned(real_xy, ex, ey)
    dens = (float(pearsonr(pc, rc)[0]) if pc.std() > 0 and rc.std() > 0 else np.nan)
    # field: per-gene binned-mean correlation on rank-normalized expression, median
    pR, rR = _rank_normalize(pred_X), _rank_normalize(real_X)
    _, pm, po = _binned(pred_xy, ex, ey, values=None) if False else (None, None, None)
    # recompute means per gene
    def gene_means(xy, X):
        nx, ny = grid, grid
        xb = np.clip(np.digitize(xy[:, 0], ex) - 1, 0, nx - 1)
        yb = np.clip(np.digitize(xy[:, 1], ey) - 1, 0, ny - 1)
        flat = yb * nx + xb
        cnt = np.bincount(flat, minlength=nx * ny).astype(float)
        occ = cnt > 0
        means = np.zeros((nx * ny, X.shape[1]))
        np.add.at(means, flat, X)
        means[occ] /= cnt[occ, None]
        return means, occ
    pmm, po = gene_means(pred_xy, pR)
    rmm, ro = gene_means(real_xy, rR)
    both = po & ro
    if both.sum() < 4:
        field = np.nan
    else:
        rs = []
        for g in range(pmm.shape[1]):
            a, b = pmm[both, g], rmm[both, g]
            if a.std() > 0 and b.std() > 0:
                rs.append(pearsonr(a, b)[0])
        field = float(np.median(rs)) if rs else np.nan
    return dens, field


def _field_ssim(pred_xy, pred_X, real_xy, real_X, grid=20):
    """SSIM of the binned mean rank-expression image (favours both-slice mixing)."""
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        return np.nan
    pR, rR = _rank_normalize(pred_X).mean(1), _rank_normalize(real_X).mean(1)
    allxy = np.vstack([pred_xy, real_xy])
    ex = np.linspace(allxy[:, 0].min(), allxy[:, 0].max(), grid + 1)
    ey = np.linspace(allxy[:, 1].min(), allxy[:, 1].max(), grid + 1)

    def img(xy, v):
        nx, ny = grid, grid
        xb = np.clip(np.digitize(xy[:, 0], ex) - 1, 0, nx - 1)
        yb = np.clip(np.digitize(xy[:, 1], ey) - 1, 0, ny - 1)
        flat = yb * nx + xb
        cnt = np.bincount(flat, minlength=nx * ny).astype(float)
        s = np.bincount(flat, weights=v, minlength=nx * ny)
        occ = cnt > 0
        s[occ] /= cnt[occ]
        return s.reshape(nx, ny)

    a, b = img(pred_xy, pR), img(real_xy, rR)
    dr = max(b.max() - b.min(), a.max() - a.min(), 1e-10)
    try:
        return float(ssim(b, a, data_range=dr))
    except Exception:
        return np.nan


def _matched_fidelity(pred_xy, pred_X, real_xy, real_X):
    """Median per-gene Pearson over nearest-neighbour-matched cells (same frame).

    Rewards a placement whose cells land where the real cells are with the right
    expression — the cell-matched fidelity the benchmark's ``pearson_median`` and
    cell-type accuracy reward, and where both-slice interpolation tends to win.
    """
    if pred_xy.shape[0] < 5 or real_xy.shape[0] < 5:
        return np.nan
    _, nn = cKDTree(pred_xy).query(real_xy, k=1)
    P = _rank_normalize(pred_X[nn])
    R = _rank_normalize(real_X)
    rs = []
    for g in range(P.shape[1]):
        if P[:, g].std() > 0 and R[:, g].std() > 0:
            rs.append(pearsonr(P[:, g], R[:, g])[0])
    return float(np.median(rs)) if rs else np.nan


def _composition_agreement(pred_types, real_types):
    if pred_types is None or real_types is None:
        return np.nan
    labels = sorted(set(map(int, pred_types)) | set(map(int, real_types)))
    p = np.array([np.mean(np.asarray(pred_types) == l) for l in labels])
    q = np.array([np.mean(np.asarray(real_types) == l) for l in labels])
    return float(1.0 - 0.5 * np.abs(p - q).sum())


def score_reconstruction(pred_coords, pred_X, pred_types, real_xy, real_X, real_types):
    """Aggregate reconstruction score of a synthesized slice vs a real one.

    Both are in the same training frame, so no alignment is needed. Returns the
    mean of the available agreement terms (higher is better).
    """
    pred_xy = np.asarray(pred_coords)[:, :2]
    real_xy = np.asarray(real_xy)
    real_X = np.asarray(real_X, dtype=np.float64)
    # Each term is a correlation-like agreement in [-1, 1]; averaging them gives a
    # balanced proxy for the benchmark that spans both sides of the placement
    # trade-off (single-slice structure/density vs both-slice field/cell fidelity).
    terms = []
    for v in (
        _coexpr_agreement(pred_X, real_X),                       # structure
        _matched_fidelity(pred_xy, pred_X, real_xy, real_X),     # cell-matched fidelity
        _composition_agreement(pred_types, real_types),          # cell-type mix
    ):
        if np.isfinite(v):
            terms.append(v)
    dens, field = _density_and_field(pred_xy, pred_X, real_xy, real_X)
    for v in (dens, field, _field_ssim(pred_xy, pred_X, real_xy, real_X)):
        if np.isfinite(v):
            terms.append(v)
    return float(np.mean(terms)) if terms else -np.inf


def pick_cv_slice(stack):
    """Index of a middle training slice that has both a lower and an upper flank."""
    n = stack.n_slices
    if n < 3:
        return None
    centres = stack.z_centers()
    order = np.argsort(centres)
    # choose the interior slice whose flanks are most balanced (closest to its z)
    mid = order[n // 2]
    return int(mid)
