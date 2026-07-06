"""Correspondence-free evaluation for de-novo virtual-slice generation (v2).

Per-cell nearest-neighbor-matched metrics (see ``evaluate.py``) assume a
cell-to-cell correspondence between prediction and ground truth. De-novo
generation does not produce one — cells are synthesized, not placed on GT cells —
so matched per-cell correlation is fragile and measures the wrong thing.

This module instead measures whether the generated slice reproduces the true
slice's **spatial expression structure and distribution**, using metrics that
need either no alignment at all, or only a coarse one that is robust to exact
cell placement. Expression is normalized identically for both slices first, so
scale differences (log-pred vs raw-GT) don't distort error/distribution metrics.

Primary generation metrics
--------------------------
* ``field_pearson`` / ``field_ssim`` — bin both slices onto a shared spatial
  grid (aligned frame), per-gene mean per bin, then compare the per-gene spatial
  fields. Robust to exact cell placement; the honest "spatial pattern" metric.
* ``morans_agreement`` — Pearson r between per-gene Moran's I of prediction and
  of GT (each computed within its own slice via kNN). Alignment-free; tests
  whether the *same genes* are spatially structured. (This is what the v1
  ``morans_i_median`` docstring described but did not compute.)
* ``coexpression_agreement`` — Pearson r between the gene-gene correlation
  matrices (upper triangles) of prediction and GT. Alignment-free; tests
  gene-gene relationships.
* ``gene_mean_pearson`` / ``gene_var_pearson`` — per-gene mean/variance
  agreement across the population. Alignment-free.
* ``density_pearson`` — bin-wise cell-count agreement (aligned frame).

Alignment (for the binned field/density metrics only) reuses the
orientation-robust ``leakage_guard.align_prediction_to_gt``; the alignment-free
metrics above are computed on raw coordinates and are immune to any residual
orientation ambiguity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import pearsonr

from .evaluate import load_prediction, load_ground_truth
from .leakage_guard import align_prediction_to_gt


# --------------------------------------------------------------------------- #
# Normalization (per-cell; makes pred and GT scale-comparable)                 #
# --------------------------------------------------------------------------- #
def _normalize_counts(X, target_sum=1e4):
    """Per-cell total-count normalize + log1p (numpy; no scanpy dependency)."""
    X = np.asarray(X, dtype=np.float64)
    totals = X.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return np.log1p(X / totals * target_sum)


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def field_metrics(pred_xy, pred_X, gt_xy, gt_X, grid=20):
    """Binned spatial-field agreement (per-gene Pearson across bins) + SSIM."""
    all_xy = np.vstack([pred_xy, gt_xy])
    xe = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid + 1)
    ye = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid + 1)

    def binned_means(xy, X):
        xb = np.clip(np.digitize(xy[:, 0], xe) - 1, 0, grid - 1)
        yb = np.clip(np.digitize(xy[:, 1], ye) - 1, 0, grid - 1)
        flat = yb * grid + xb
        n_bins = grid * grid
        sums = np.zeros((n_bins, X.shape[1]))
        cnts = np.zeros(n_bins)
        np.add.at(sums, flat, X)
        np.add.at(cnts, flat, 1.0)
        occ = cnts > 0
        means = np.zeros_like(sums)
        means[occ] = sums[occ] / cnts[occ, None]
        return means, occ

    pm, po = binned_means(pred_xy, pred_X)
    gm, go = binned_means(gt_xy, gt_X)
    both = po & go
    if both.sum() < 4:
        return {"field_pearson": np.nan, "field_ssim": np.nan}

    pm, gm = pm[both], gm[both]
    rs = []
    for g in range(pm.shape[1]):
        if pm[:, g].std() > 0 and gm[:, g].std() > 0:
            rs.append(pearsonr(pm[:, g], gm[:, g])[0])
    field_pearson = float(np.median(rs)) if rs else np.nan

    # A single global SSIM-like structural agreement over the mean field.
    try:
        from skimage.metrics import structural_similarity as ssim
        p_img = pm.mean(axis=1)
        g_img = gm.mean(axis=1)
        dr = max(g_img.max() - g_img.min(), p_img.max() - p_img.min(), 1e-10)
        field_ssim = float(ssim(g_img, p_img, data_range=dr))
    except Exception:
        field_ssim = np.nan
    return {"field_pearson": field_pearson, "field_ssim": field_ssim}


def _morans_i(xy, X, k=10):
    """Per-gene Moran's I via kNN spatial weights (row-standardized)."""
    n = X.shape[0]
    if n < k + 1:
        return np.full(X.shape[1], np.nan)
    nn = cKDTree(xy)
    _, idx = nn.query(xy, k=k + 1)
    idx = idx[:, 1:]  # drop self
    Z = X - X.mean(axis=0, keepdims=True)
    denom = (Z ** 2).sum(axis=0)
    # numerator: sum_i z_i * mean_j(z_j over neighbors)
    neigh_mean = Z[idx].mean(axis=1)          # (n, G)
    numer = (Z * neigh_mean).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.where(denom > 0, numer / denom, np.nan)
    return mi


def morans_agreement(pred_xy, pred_X, gt_xy, gt_X, k=10):
    """Pearson r between per-gene Moran's I of prediction and GT (alignment-free)."""
    mp = _morans_i(pred_xy, pred_X, k)
    mg = _morans_i(gt_xy, gt_X, k)
    valid = ~(np.isnan(mp) | np.isnan(mg))
    if valid.sum() < 3 or mp[valid].std() == 0 or mg[valid].std() == 0:
        return {"morans_agreement": np.nan,
                "morans_i_pred_median": float(np.nanmedian(mp)) if np.isfinite(mp).any() else np.nan}
    return {"morans_agreement": float(pearsonr(mp[valid], mg[valid])[0]),
            "morans_i_pred_median": float(np.nanmedian(mp[valid]))}


def coexpression_agreement(pred_X, gt_X, max_genes=200):
    """Pearson r between gene-gene correlation matrices (alignment-free)."""
    G = pred_X.shape[1]
    if G < 3:
        return {"coexpression_agreement": np.nan}
    if G > max_genes:  # subsample genes for tractability on large panels
        sel = np.linspace(0, G - 1, max_genes).astype(int)
        pred_X, gt_X = pred_X[:, sel], gt_X[:, sel]
    def corr_upper(X):
        C = np.corrcoef(X, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        return C[iu]
    pu, gu = corr_upper(pred_X), corr_upper(gt_X)
    valid = ~(np.isnan(pu) | np.isnan(gu))
    if valid.sum() < 3 or pu[valid].std() == 0 or gu[valid].std() == 0:
        return {"coexpression_agreement": np.nan}
    return {"coexpression_agreement": float(pearsonr(pu[valid], gu[valid])[0])}


def gene_level_agreement(pred_X, gt_X):
    """Per-gene mean/variance agreement across the population (alignment-free)."""
    out = {}
    pm, gm = pred_X.mean(axis=0), gt_X.mean(axis=0)
    pv, gv = pred_X.var(axis=0), gt_X.var(axis=0)
    out["gene_mean_pearson"] = (float(pearsonr(pm, gm)[0])
                                if pm.std() > 0 and gm.std() > 0 else np.nan)
    out["gene_var_pearson"] = (float(pearsonr(pv, gv)[0])
                               if pv.std() > 0 and gv.std() > 0 else np.nan)
    return out


def density_agreement(pred_xy, gt_xy, grid=20):
    all_xy = np.vstack([pred_xy, gt_xy])
    xe = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid + 1)
    ye = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid + 1)
    ph, _, _ = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1], bins=[xe, ye])
    gh, _, _ = np.histogram2d(gt_xy[:, 0], gt_xy[:, 1], bins=[xe, ye])
    pf, gf = ph.ravel(), gh.ravel()
    if pf.std() == 0 or gf.std() == 0:
        return {"density_pearson": np.nan}
    return {"density_pearson": float(pearsonr(pf, gf)[0])}


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def evaluate_generation(prediction_path, h5ad_path, output_path=None,
                        grid=20, moran_k=10):
    """Compute correspondence-free generation metrics per held-out section.

    Metrics are averaged over held-out sections (weighted by GT cells).
    """
    pred = load_prediction(prediction_path)
    holdout_sections = [str(s) for s in pred["holdout_sections"]]
    gt = load_ground_truth(h5ad_path, holdout_sections)

    # Common gene space.
    common = np.intersect1d(pred["gene_names"], gt.var_names.values)
    metrics = {
        "method": pred["method_name"],
        "holdout_sections": holdout_sections,
        "n_holdout_cells_gt": int(gt.n_obs),
        "n_predicted_cells": int(pred["X"].shape[0]),
        "n_common_genes": int(len(common)),
        "eval": "generation (correspondence-free)",
    }
    if len(common) == 0:
        metrics["error"] = "no common genes"
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump(_clean(metrics), open(output_path, "w"), indent=2)
        return _clean(metrics)

    pgi = np.array([np.where(pred["gene_names"] == g)[0][0] for g in common])
    ggi = np.array([np.where(gt.var_names.values == g)[0][0] for g in common])

    pred_X_all = pred["X"][:, pgi]
    pred_X_all = pred_X_all.toarray() if sp.issparse(pred_X_all) else np.asarray(pred_X_all)
    gt_sections = gt.obs["section"].values.astype(str)
    gt_spatial = gt.obsm["spatial"]

    per_section = []
    weights = []
    for sec in holdout_sections:
        pm = pred["section"] == sec
        gm = gt_sections == sec
        if pm.sum() < 5 or gm.sum() < 5:
            continue
        # Normalize BOTH slices identically (scale-comparable).
        pX = _normalize_counts(pred_X_all[pm])
        gX_raw = gt.X[gm][:, ggi]
        gX_raw = gX_raw.toarray() if sp.issparse(gX_raw) else np.asarray(gX_raw)
        gX = _normalize_counts(gX_raw)

        gt_xy = gt_spatial[gm, :2]
        pred_xy = np.column_stack([pred["x"][pm], pred["y"][pm]])
        # Orientation-robust alignment for the binned (field/density) metrics only.
        pred_xy_al = align_prediction_to_gt(pred_xy, gt_xy, with_scale=True)

        m = {}
        m.update(field_metrics(pred_xy_al, pX, gt_xy, gX, grid=grid))
        m.update(density_agreement(pred_xy_al, gt_xy, grid=grid))
        m.update(morans_agreement(pred_xy, pX, gt_xy, gX, k=moran_k))  # alignment-free
        m.update(coexpression_agreement(pX, gX))                        # alignment-free
        m.update(gene_level_agreement(pX, gX))                          # alignment-free
        per_section.append(m)
        weights.append(int(gm.sum()))

    if not per_section:
        metrics["error"] = "no evaluable sections"
    else:
        w = np.array(weights, dtype=float)
        keys = per_section[0].keys()
        for kname in keys:
            vals = np.array([s[kname] for s in per_section], dtype=float)
            ok = ~np.isnan(vals)
            metrics[kname] = float(np.average(vals[ok], weights=w[ok])) if ok.any() else None

    metrics = _clean(metrics)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(metrics, open(output_path, "w"), indent=2)
    return metrics


def _clean(d):
    return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser(description="Correspondence-free generation evaluation")
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--ground-truth", required=True)
    ap.add_argument("--output")
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--moran-k", type=int, default=10)
    args = ap.parse_args()
    print(json.dumps(evaluate_generation(args.prediction, args.ground_truth,
                                         args.output, args.grid, args.moran_k), indent=2))


if __name__ == "__main__":
    main()
