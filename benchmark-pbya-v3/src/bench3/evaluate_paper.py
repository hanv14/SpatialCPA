"""The SpatialZ-paper validation strategy, as metrics (benchmark-pbya-v3).

The paper validates a reconstructed STARmap section four ways:

1. **UMAP continuity** — do reconstructed cells land where real cells of the
   held-out section land, or do they form their own island?
2. **Marker-gene spatial patterns** — Flt1, Pcp4, Cux2. In visual cortex these
   are laminar (Cux2 upper layers, Pcp4 deep layers) or vascular (Flt1), so the
   right question is whether the *pattern* is reproduced, not whether individual
   cells match.
3. **Gene expression similarity and spatial autocorrelation** — Moran's I *and*
   Geary's C per gene.
4. **Preservation of cell spatial localization** — are cells of a given type in
   the right part of the tissue?

Every metric here is **correspondence-free**: de-novo generation synthesizes
cells, it does not place them on ground-truth cells, so there is no cell-to-cell
pairing to correlate over (see benchmark-pbya-v2/README.md for why the
cell-matched metrics are the wrong measurement for generation).

Every metric is also **scale-fair**: methods emit expression on different scales
(raw counts, log1p, arbitrary), so the primary computations run on *per-gene
rank-normalized* expression, which is invariant to any monotonic per-gene
transform. The same normalization is applied to the ground truth, so raw-count GT
and log-scale predictions are directly comparable. (Rank-normalization is
imported from v2's evaluator so v3 and v2 use bit-identical definitions.)

Only the binned-field and localization metrics need the prediction and the GT to
share a coordinate frame; those use v2's orientation-robust
``align_prediction_to_gt``. Alignment is an evaluation-side operation — it reads
the GT and feeds nothing back to the method — so it is not leakage.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

from ._v2bridge import align_prediction_to_gt, load_ground_truth, load_prediction
from .config import (
    DEPTH_BINS, EMBED_NEIGHBORS, FIELD_GRID, LAYER_DEEP, LAYER_SUPERFICIAL,
    MARKER_GENES, RANDOM_SEED, SPATIAL_K,
)

# Same definitions v2's generation evaluator uses — imported rather than
# re-implemented so the two benchmarks cannot silently drift apart.
from benchmark.evaluate_generation import (  # noqa: E402
    _morans_i, _normalize_counts, _rank_normalize,
)


# --------------------------------------------------------------------------- #
# Spatial autocorrelation                                                      #
# --------------------------------------------------------------------------- #
def gearys_c(xy, X, k=SPATIAL_K, gene_chunk=16):
    """Per-gene Geary's C under row-standardized kNN spatial weights.

    C = (n-1) · Σᵢⱼ wᵢⱼ(xᵢ-xⱼ)² / (2·S₀·Σᵢ(xᵢ-x̄)²).  With row-standardized kNN
    weights (wᵢⱼ = 1/k for each of i's k neighbours) S₀ = n, so

        C = (n-1) · Σᵢ (1/k)Σⱼ∈N(i)(xᵢ-xⱼ)² / (2·n·Σᵢ(xᵢ-x̄)²)

    C ≈ 1 means no spatial autocorrelation; C < 1 positive autocorrelation
    (neighbouring cells are similar); C > 1 negative. Geary's C is the local
    counterpart to Moran's I — it is driven by pairwise differences rather than
    by covariance with the neighbourhood mean, so it is markedly more sensitive
    to a method that over-smooths, which is exactly why the paper reports both.

    The same weights (:func:`benchmark.evaluate_generation._morans_i` uses a
    row-standardized kNN graph too) so I and C are directly comparable.
    """
    X = np.asarray(X, dtype=np.float64)
    n, G = X.shape
    if n < k + 2:
        return np.full(G, np.nan)
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    idx = idx[:, 1:]                                    # drop self
    denom = ((X - X.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)

    numer = np.empty(G, dtype=np.float64)
    for s in range(0, G, gene_chunk):                   # chunked: (n, k, g) buffer
        Xc = X[:, s:s + gene_chunk]
        diff = Xc[:, None, :] - Xc[idx]                 # (n, k, g)
        numer[s:s + gene_chunk] = (diff ** 2).sum(axis=(0, 1)) / k

    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.where(denom > 0, (n - 1) * numer / (2.0 * n * denom), np.nan)
    return c


def _agreement(pred_vals, gt_vals, name):
    """Pearson/Spearman agreement + mean absolute deviation across genes."""
    p = np.asarray(pred_vals, dtype=np.float64)
    g = np.asarray(gt_vals, dtype=np.float64)
    ok = ~(np.isnan(p) | np.isnan(g))
    out = {
        f"{name}_median_pred": float(np.median(p[ok])) if ok.any() else np.nan,
        f"{name}_median_gt": float(np.median(g[ok])) if ok.any() else np.nan,
        f"{name}_mae": float(np.mean(np.abs(p[ok] - g[ok]))) if ok.any() else np.nan,
        f"{name}_pearson": np.nan,
        f"{name}_spearman": np.nan,
    }
    if ok.sum() >= 3 and p[ok].std() > 0 and g[ok].std() > 0:
        out[f"{name}_pearson"] = float(pearsonr(p[ok], g[ok])[0])
        out[f"{name}_spearman"] = float(spearmanr(p[ok], g[ok])[0])
    return out


def spatial_autocorrelation_metrics(pred_xy, pred_R, gt_xy, gt_R, k=SPATIAL_K):
    """Moran's I and Geary's C agreement between prediction and GT (per gene).

    Both are computed *within* each slice using its own spatial graph, so this is
    alignment-free: it asks whether the same genes carry the same amount of
    spatial structure, not whether they do so at the same coordinates.

    ``*_pearson``/``*_spearman`` capture whether the ranking of genes by spatial
    structure is preserved; ``*_mae`` captures whether the *level* is right,
    which is what catches over-smoothing (a blurred reconstruction inflates
    Moran's I and deflates Geary's C while keeping the gene ranking intact).
    """
    out = {}
    out.update(_agreement(_morans_i(pred_xy, pred_R, k),
                          _morans_i(gt_xy, gt_R, k), "morans"))
    out.update(_agreement(gearys_c(pred_xy, pred_R, k),
                          gearys_c(gt_xy, gt_R, k), "gearys"))
    return out


# --------------------------------------------------------------------------- #
# UMAP continuity between real and reconstructed cells                         #
# --------------------------------------------------------------------------- #
def _knn_mixing(embedding, labels, k=EMBED_NEIGHBORS):
    """Normalized kNN mixing between two groups in an embedding.

    For each cell, the fraction of its k nearest neighbours drawn from the other
    group, averaged over all cells and divided by the value expected under
    perfect mixing (2·n_a·n_b / (n·(n-1))). 1 = the two groups are locally
    indistinguishable; 0 = they occupy disjoint regions of the embedding. This is
    the quantitative version of "do the reconstructed cells overlay the real
    ones in the UMAP".
    """
    labels = np.asarray(labels)
    n = len(labels)
    kk = int(min(k, n - 1))
    if kk < 1:
        return np.nan
    _, idx = cKDTree(embedding).query(embedding, k=kk + 1)
    idx = idx[:, 1:]
    observed = float((labels[idx] != labels[:, None]).mean())
    n_a = int((labels == labels[0]).sum())
    n_b = n - n_a
    expected = 2.0 * n_a * n_b / (n * (n - 1)) if n > 1 else 0.0
    if expected <= 0:
        return np.nan
    return float(np.clip(observed / expected, 0.0, 1.0))


def embedding_continuity(pred_R, gt_R, n_neighbors=EMBED_NEIGHBORS,
                         seed=RANDOM_SEED, max_cells=4000, use_umap=True):
    """UMAP (and PCA) continuity between reconstructed and real held-out cells.

    Both matrices are already per-gene rank-normalized, so the shared embedding
    cannot be driven apart by a mere difference in output scale — a method is
    only separated from the GT if its *expression structure* differs.

    Returns ``umap_mixing`` (the paper's comparison), ``umap_centroid_dist``
    (separation of the two clouds' centroids in units of the GT cloud's spread;
    lower is better) and ``embedding_mixing_pca`` — the same mixing score in
    linear PCA space, which is deterministic and therefore the number to trust
    when UMAP's stochasticity is a concern.
    """
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(seed)

    def sub(M):
        if M.shape[0] > max_cells:
            return M[rng.choice(M.shape[0], max_cells, replace=False)]
        return M

    P, G = sub(pred_R), sub(gt_R)
    if P.shape[0] < 10 or G.shape[0] < 10:
        return {"umap_mixing": np.nan, "umap_centroid_dist": np.nan,
                "embedding_mixing_pca": np.nan}

    Z = np.vstack([P, G])
    labels = np.r_[np.zeros(len(P), dtype=int), np.ones(len(G), dtype=int)]
    n_comp = int(min(20, Z.shape[1], Z.shape[0] - 1))
    pcs = PCA(n_components=n_comp, random_state=seed).fit_transform(Z)

    out = {"embedding_mixing_pca": _knn_mixing(pcs, labels, n_neighbors)}

    emb = None
    if use_umap:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import umap
                emb = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                                random_state=seed).fit_transform(pcs)
        except Exception as e:                      # pragma: no cover - env dependent
            print(f"    [paper] UMAP unavailable ({e}); reporting PCA mixing only")
            emb = None

    if emb is None:
        out["umap_mixing"] = np.nan
        out["umap_centroid_dist"] = np.nan
        return out

    out["umap_mixing"] = _knn_mixing(emb, labels, n_neighbors)
    e_p, e_g = emb[labels == 0], emb[labels == 1]
    spread = float(np.sqrt((e_g.var(axis=0)).sum()))
    out["umap_centroid_dist"] = (
        float(np.linalg.norm(e_p.mean(axis=0) - e_g.mean(axis=0)) / spread)
        if spread > 0 else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Marker-gene spatial patterns                                                 #
# --------------------------------------------------------------------------- #
def _zscore(M):
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    return (M - mu) / np.where(sd > 0, sd, 1.0)


def laminar_axis(gt_xy, gt_R, gene_names, k=SPATIAL_K, layers=None):
    """In-plane unit vector pointing from deep layers toward the pial surface.

    Cortical marker expression is organized along one axis, so the honest
    "did you reproduce the marker pattern" test is whether the *depth profile*
    matches. The axis is derived from the **ground truth only** (evaluation-side)
    as the spatial gradient of a signed layer score: mean z-scored superficial
    markers minus mean z-scored deep markers (``config.LAYER_SUPERFICIAL`` /
    ``LAYER_DEEP``). If the panel carries neither group, it falls back to the
    gradient of the single most spatially structured gene (highest Moran's I).

    Returns ``None`` when no usable gradient exists (e.g. a flat score).
    """
    names = list(gene_names)
    sup_genes, deep_genes = layers or (LAYER_SUPERFICIAL, LAYER_DEEP)
    sup = [names.index(g) for g in sup_genes if g in names]
    deep = [names.index(g) for g in deep_genes if g in names]

    if sup and deep:
        score = (_zscore(gt_R[:, sup]).mean(axis=1)
                 - _zscore(gt_R[:, deep]).mean(axis=1))
    else:
        mi = _morans_i(gt_xy, gt_R, k)
        if not np.isfinite(mi).any():
            return None
        score = _zscore(gt_R[:, [int(np.nanargmax(mi))]])[:, 0]

    if score.std() == 0:
        return None
    A = np.column_stack([gt_xy - gt_xy.mean(axis=0), np.ones(len(gt_xy))])
    coef, *_ = np.linalg.lstsq(A, score, rcond=None)
    u = coef[:2]
    norm = float(np.linalg.norm(u))
    return None if norm == 0 else u / norm


def depth_profile(xy, values, axis_u, edges):
    """Mean ``values`` per bin along the laminar axis (NaN for empty bins)."""
    d = xy @ axis_u
    b = np.clip(np.digitize(d, edges) - 1, 0, len(edges) - 2)
    n_bins = len(edges) - 1
    sums = np.zeros(n_bins)
    cnts = np.zeros(n_bins)
    np.add.at(sums, b, values)
    np.add.at(cnts, b, 1.0)
    out = np.full(n_bins, np.nan)
    occ = cnts > 0
    out[occ] = sums[occ] / cnts[occ]
    return out


def _binned_gene_field(xy, values, xe, ye, grid):
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


def marker_metrics(pred_xy_aligned, pred_R, gt_xy, gt_R, gene_names,
                   markers=MARKER_GENES, grid=FIELD_GRID, depth_bins=DEPTH_BINS,
                   k=SPATIAL_K, layers=None):
    """Per-marker spatial-pattern agreement (Flt1 / Pcp4 / Cux2 by default).

    Three complementary views of "is the marker's spatial pattern reproduced":

    ``<gene>_field_r``
        Pearson r between the binned 2-D expression fields (needs alignment).
    ``<gene>_depth_r``
        Pearson r between the profiles along the cortical laminar axis. This is
        the laminar-organization test and is far more robust than the 2-D field
        to residual in-plane misalignment, because it integrates over the axis
        parallel to the layers.
    ``<gene>_morans_delta``
        |Moran's I(pred) − Moran's I(GT)| for that marker — alignment-free, and
        it catches a reconstruction that has the right pattern shape but the
        wrong amount of spatial structure.

    Also returns the unweighted means over markers as ``marker_field_r``,
    ``marker_depth_r`` and ``marker_morans_mae``.
    """
    names = list(gene_names)
    present = [g for g in markers if g in names]
    out = {}
    if not present:
        return {"marker_field_r": np.nan, "marker_depth_r": np.nan,
                "marker_morans_mae": np.nan}

    cols = [names.index(g) for g in present]
    mi_p = _morans_i(pred_xy_aligned, pred_R[:, cols], k)
    mi_g = _morans_i(gt_xy, gt_R[:, cols], k)

    all_xy = np.vstack([pred_xy_aligned, gt_xy])
    xe = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid + 1)
    ye = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid + 1)

    axis_u = laminar_axis(gt_xy, gt_R, names, k=k, layers=layers)
    if axis_u is not None:
        d_gt = gt_xy @ axis_u
        edges = np.linspace(d_gt.min(), d_gt.max(), depth_bins + 1)

    field_rs, depth_rs, morans_deltas = [], [], []
    for j, (g, col) in enumerate(zip(present, cols)):
        pm, po = _binned_gene_field(pred_xy_aligned, pred_R[:, col], xe, ye, grid)
        gm, go = _binned_gene_field(gt_xy, gt_R[:, col], xe, ye, grid)
        both = po & go
        fr = np.nan
        if both.sum() >= 4 and pm[both].std() > 0 and gm[both].std() > 0:
            fr = float(pearsonr(pm[both], gm[both])[0])
        out[f"marker_{g}_field_r"] = fr
        field_rs.append(fr)

        dr = np.nan
        if axis_u is not None:
            pp = depth_profile(pred_xy_aligned, pred_R[:, col], axis_u, edges)
            gp = depth_profile(gt_xy, gt_R[:, col], axis_u, edges)
            ok = ~(np.isnan(pp) | np.isnan(gp))
            if ok.sum() >= 4 and pp[ok].std() > 0 and gp[ok].std() > 0:
                dr = float(pearsonr(pp[ok], gp[ok])[0])
        out[f"marker_{g}_depth_r"] = dr
        depth_rs.append(dr)

        delta = (float(abs(mi_p[j] - mi_g[j]))
                 if np.isfinite(mi_p[j]) and np.isfinite(mi_g[j]) else np.nan)
        out[f"marker_{g}_morans_delta"] = delta
        morans_deltas.append(delta)

    out["marker_field_r"] = float(np.nanmean(field_rs)) if np.any(~np.isnan(field_rs)) else np.nan
    out["marker_depth_r"] = float(np.nanmean(depth_rs)) if np.any(~np.isnan(depth_rs)) else np.nan
    out["marker_morans_mae"] = (float(np.nanmean(morans_deltas))
                                if np.any(~np.isnan(morans_deltas)) else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Preservation of cell spatial localization                                    #
# --------------------------------------------------------------------------- #
def _sinkhorn(A, B, scale, eps=0.05, n_iter=200):
    """Entropic-OT transport cost between 2-D point sets, on a shared scale."""
    C = cdist(A, B, metric="sqeuclidean") / scale
    K = np.exp(-C / eps) + 1e-300
    n, m = C.shape
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    v = np.ones(m)
    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)
    return float(((u[:, None] * K * v[None, :]) * C).sum())


def _sinkhorn_divergence(A, B, scale, eps=0.05):
    """Debiased Sinkhorn divergence (0 for identical distributions)."""
    return max(_sinkhorn(A, B, scale, eps)
               - 0.5 * (_sinkhorn(A, A, scale, eps) + _sinkhorn(B, B, scale, eps)),
               0.0)


def celltype_localization(pred_xy_aligned, pred_types, gt_xy, gt_types,
                          seed=RANDOM_SEED, min_gt_cells=20, min_pred_cells=5,
                          max_n=250):
    """Is each cell type in the *right part of the tissue*?

    For every ground-truth cell type, the spatial distribution of that type's
    predicted cells is compared with the ground-truth one via a debiased
    Sinkhorn (optimal-transport) divergence on tissue-normalized coordinates.
    The raw divergence has no natural unit, so it is calibrated against a
    within-tissue null: the divergence between the true type-c cloud and an
    equally sized *random* draw from all GT cells of that section — i.e. what
    you would score by scattering the type anywhere in the tissue.

        score_c = 1 − D(pred_c, gt_c) / D(random, gt_c)

    1 = the type's spatial distribution is reproduced; 0 = no better than
    scattering it at random. Types the method never produces score 0 (a type
    that is absent has no localization to preserve); types whose null divergence
    is negligible (already tissue-wide, so there is nothing to localize) are
    skipped. Scores are averaged weighted by GT type frequency.

    Also returns ``celltype_ot`` — the raw weighted divergence over types present
    in both, in units of the tissue radius squared (lower is better).
    """
    rng = np.random.default_rng(seed)
    pred_types = np.asarray(pred_types).astype(str)
    gt_types = np.asarray(gt_types).astype(str)

    centre = gt_xy.mean(axis=0)
    radius = float(np.sqrt(((gt_xy - centre) ** 2).sum(axis=1)).mean())
    if radius <= 0:
        return {"celltype_localization": np.nan, "celltype_ot": np.nan}
    G_all = (gt_xy - centre) / radius
    P_all = (pred_xy_aligned - centre) / radius

    # One cost scale for every Sinkhorn call in this section — the median squared
    # distance between GT cells. Sharing it keeps the entropic regularization at a
    # sensible strength AND keeps the observed and null divergences on the same
    # ruler, which is what makes their ratio meaningful.
    ref = G_all[rng.choice(len(G_all), min(len(G_all), 400), replace=False)]
    scale = float(np.median(cdist(ref, ref, metric="sqeuclidean"))) or 1.0

    def sub(M):
        return M[rng.choice(len(M), max_n, replace=False)] if len(M) > max_n else M

    scores, ots, weights = [], [], []
    for t in np.unique(gt_types):
        g_mask = gt_types == t
        if g_mask.sum() < min_gt_cells:
            continue
        w = float(g_mask.mean())
        Gc = sub(G_all[g_mask])

        null_draw = sub(G_all[rng.choice(len(G_all), int(g_mask.sum()), replace=False)])
        d_null = _sinkhorn_divergence(null_draw, Gc, scale)
        if d_null < 1e-4:          # type already spread tissue-wide: nothing to test
            continue

        p_mask = pred_types == t
        if p_mask.sum() < min_pred_cells:
            scores.append(0.0)     # never produced -> no localization preserved
            weights.append(w)
            continue

        d_obs = _sinkhorn_divergence(sub(P_all[p_mask]), Gc, scale)
        scores.append(float(np.clip(1.0 - d_obs / d_null, 0.0, 1.0)))
        ots.append(d_obs)
        weights.append(w)

    if not scores:
        return {"celltype_localization": np.nan, "celltype_ot": np.nan}
    w = np.asarray(weights)
    return {
        "celltype_localization": float(np.average(scores, weights=w)),
        "celltype_ot": float(np.mean(ots)) if ots else np.nan,
    }


# --------------------------------------------------------------------------- #
# Gene expression similarity                                                   #
# --------------------------------------------------------------------------- #
def expression_similarity(pred_L, gt_L):
    """Per-gene mean/variance agreement (Spearman — rank-based, robust).

    Computed on log-normalized expression, matching the SpatialZ paper audit's
    ``gene_mean_pearson``/``gene_var_pearson`` but with Spearman so a single
    dominant gene cannot carry the correlation.
    """
    out = {}
    for stat, fn in (("mean", np.mean), ("var", np.var)):
        p = fn(pred_L, axis=0)
        g = fn(gt_L, axis=0)
        out[f"gene_{stat}_spearman"] = (
            float(spearmanr(p, g)[0]) if p.std() > 0 and g.std() > 0 else np.nan)
    return out


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def evaluate_paper(prediction_path, h5ad_path, output_path=None,
                   markers=None, k=SPATIAL_K, grid=FIELD_GRID,
                   depth_bins=DEPTH_BINS, seed=RANDOM_SEED, use_umap=True,
                   layers=None):
    """Compute the paper's validation metrics, per held-out section and pooled.

    Section-level results are kept in ``per_section`` (used by the figures and
    the disaggregated table); the top-level values are averages over held-out
    sections weighted by ground-truth cell count.
    """
    pred = load_prediction(prediction_path)
    holdout_sections = [str(s) for s in pred["holdout_sections"]]
    gt = load_ground_truth(h5ad_path, holdout_sections)

    # Marker and laminar-axis genes come from the dataset itself (written by
    # ``prepare_dataset`` into uns['paper_protocol']), so a second dataset is
    # never scored against STARmap's panel. Explicit arguments still win, and the
    # config defaults apply to a file built before this was recorded.
    pp = dict(gt.uns.get("paper_protocol") or {})
    if markers is None:
        markers = [str(g) for g in pp.get("marker_genes", [])] or MARKER_GENES
    if layers is None:
        sup = [str(g) for g in pp.get("layer_superficial", [])] or LAYER_SUPERFICIAL
        deep = [str(g) for g in pp.get("layer_deep", [])] or LAYER_DEEP
        layers = (sup, deep)

    common = np.intersect1d(pred["gene_names"], gt.var_names.values)
    metrics = {
        "method": pred["method_name"],
        "holdout_sections": holdout_sections,
        "n_holdout_cells_gt": int(gt.n_obs),
        "n_predicted_cells": int(pred["X"].shape[0]),
        "n_common_genes": int(len(common)),
        "eval": "paper (SpatialZ STARmap validation strategy)",
    }
    if len(common) == 0:
        metrics["error"] = "no common genes"
        return _finish(metrics, output_path)

    pgi = np.array([int(np.where(pred["gene_names"] == g)[0][0]) for g in common])
    ggi = np.array([int(np.where(gt.var_names.values == g)[0][0]) for g in common])
    gene_names = [str(g) for g in common]

    pred_X_all = pred["X"][:, pgi]
    pred_X_all = pred_X_all.toarray() if sp.issparse(pred_X_all) else np.asarray(pred_X_all)
    gt_sections = gt.obs["section"].values.astype(str)
    gt_spatial = np.asarray(gt.obsm["spatial"])
    gt_has_types = "cell_type" in gt.obs.columns

    per_section, weights = {}, []
    for sec in holdout_sections:
        pm = pred["section"] == sec
        gm = gt_sections == sec
        if pm.sum() < 10 or gm.sum() < 10:
            per_section[sec] = {"error": f"too few cells "
                                         f"(pred={int(pm.sum())}, gt={int(gm.sum())})"}
            continue

        pred_X = pred_X_all[pm]
        gt_X = gt.X[gm][:, ggi]
        gt_X = gt_X.toarray() if sp.issparse(gt_X) else np.asarray(gt_X)

        pR, gR = _rank_normalize(pred_X), _rank_normalize(gt_X)   # scale-fair
        pL, gL = _normalize_counts(pred_X), _normalize_counts(gt_X)

        gt_xy = gt_spatial[gm, :2]
        pred_xy = np.column_stack([pred["x"][pm], pred["y"][pm]])
        pred_xy_al = align_prediction_to_gt(pred_xy, gt_xy, with_scale=True)

        m = {}
        m.update(spatial_autocorrelation_metrics(pred_xy, pR, gt_xy, gR, k=k))
        m.update(embedding_continuity(pR, gR, seed=seed, use_umap=use_umap))
        m.update(marker_metrics(pred_xy_al, pR, gt_xy, gR, gene_names,
                                markers=markers, grid=grid,
                                depth_bins=depth_bins, k=k, layers=layers))
        m.update(expression_similarity(pL, gL))
        if gt_has_types:
            m.update(celltype_localization(
                pred_xy_al, pred["cell_type"][pm],
                gt_xy, gt.obs["cell_type"].values[gm], seed=seed))
        m["cell_count_ratio"] = float(pm.sum()) / float(gm.sum())
        m["n_gt_cells"] = int(gm.sum())
        m["n_pred_cells"] = int(pm.sum())

        per_section[sec] = m
        weights.append(int(gm.sum()))

    usable = [(s, v) for s, v in per_section.items() if "error" not in v]
    if not usable:
        metrics["error"] = "no evaluable sections"
        metrics["per_section"] = per_section
        return _finish(metrics, output_path)

    w = np.asarray(weights, dtype=float)
    keys = sorted({key for _, v in usable for key in v
                   if key not in ("n_gt_cells", "n_pred_cells")})
    for key in keys:
        vals = np.array([v.get(key, np.nan) for _, v in usable], dtype=float)
        ok = ~np.isnan(vals)
        metrics[f"paper_{key}"] = (float(np.average(vals[ok], weights=w[ok]))
                                   if ok.any() else None)

    metrics["per_section"] = per_section
    return _finish(metrics, output_path)


def _finish(metrics, output_path):
    metrics = _clean(metrics)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    return metrics


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(float(obj)) else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def main():
    ap = argparse.ArgumentParser(
        description="SpatialZ-paper validation metrics for one prediction")
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output")
    ap.add_argument("--no-umap", action="store_true",
                    help="skip the UMAP embedding (PCA mixing is still reported)")
    args = ap.parse_args()
    res = evaluate_paper(args.prediction, args.ground_truth, args.output,
                         use_umap=not args.no_umap)
    print(json.dumps({k: v for k, v in res.items() if k != "per_section"}, indent=2))


if __name__ == "__main__":
    main()
