"""Compute evaluation metrics: prediction.h5 vs ground-truth h5ad.

Metrics
-------
1. Per-gene expression correlation (Pearson, Spearman)
2. Expression error (RMSE, MAE)
3. Cell-type recovery (accuracy, macro-F1)
4. Spatial structure (SSIM on binned grids)
5. Cell density recovery (Pearson r of bin-wise counts)
6. Matching rate (fraction of GT cells matched within threshold)
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score

from .config import (
    METRIC_NAMES,
    NN_MATCH_THRESHOLD_UM,
    SSIM_GRID_SIZE,
    SSIM_TOP_GENES,
)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_prediction(prediction_path):
    """Load prediction.h5 into dict with X (sparse CSR), obs, var, uns."""
    pred = {}
    with h5py.File(prediction_path, "r") as f:
        # X as CSR
        data = f["X/data"][:]
        indices = f["X/indices"][:]
        indptr = f["X/indptr"][:]
        shape = tuple(f["X/shape"][:])
        pred["X"] = sp.csr_matrix((data, indices, indptr), shape=shape)

        pred["cell_id"] = f["obs/cell_id"][:].astype(str)
        pred["x"] = f["obs/x"][:]
        pred["y"] = f["obs/y"][:]
        pred["z"] = f["obs/z"][:]
        pred["section"] = f["obs/section"][:].astype(str)
        pred["cell_type"] = f["obs/cell_type"][:].astype(str)
        pred["gene_names"] = f["var/gene_name"][:].astype(str)

        pred["holdout_sections"] = json.loads(f["uns/holdout_sections"][()])
        pred["method_name"] = f["uns/method_name"][()].decode() if isinstance(
            f["uns/method_name"][()], bytes) else str(f["uns/method_name"][()])
    return pred


def load_ground_truth(h5ad_path, holdout_sections):
    """Load ground-truth cells from held-out sections of original h5ad."""
    import anndata as ad
    adata = ad.read_h5ad(h5ad_path)
    mask = adata.obs["section"].isin(holdout_sections)
    gt = adata[mask].copy()
    return gt


# ── Cell matching ─────────────────────────────────────────────────────────────

def match_cells(pred, gt_adata, threshold_um=NN_MATCH_THRESHOLD_UM):
    """Match predicted cells to ground-truth via NN in (x,y) per section.

    Returns
    -------
    pred_idx, gt_idx : arrays of matched indices
    """
    pred_idx_all = []
    gt_idx_all = []
    gt_sections = gt_adata.obs["section"].values
    gt_spatial = gt_adata.obsm["spatial"]  # (n, 3)

    for sec in np.unique(gt_sections):
        # ground truth indices for this section
        gt_mask = gt_sections == sec
        gt_sec_idx = np.where(gt_mask)[0]
        gt_xy = gt_spatial[gt_sec_idx, :2]

        # prediction indices for this section
        pred_mask = pred["section"] == sec
        pred_sec_idx = np.where(pred_mask)[0]
        if len(pred_sec_idx) == 0:
            continue
        pred_xy = np.column_stack([pred["x"][pred_mask], pred["y"][pred_mask]])

        # Align predicted to GT coordinates using center+scale to handle
        # coordinate frame mismatches (methods may synthesize in flanking space)
        gt_center = gt_xy.mean(axis=0)
        pred_center = pred_xy.mean(axis=0)
        gt_range = gt_xy.max(axis=0) - gt_xy.min(axis=0)
        pred_range = pred_xy.max(axis=0) - pred_xy.min(axis=0)
        scale = np.where(pred_range > 0, gt_range / pred_range, 1.0)
        # Only scale if ratio > 1.5x off (avoid spurious scaling for aligned data)
        scale = np.where(np.abs(scale - 1.0) > 0.5, scale, 1.0)
        pred_xy_aligned = (pred_xy - pred_center) * scale + gt_center

        # build tree on predicted, query ground truth
        tree = cKDTree(pred_xy_aligned)
        dists, nn_idx = tree.query(gt_xy, k=1)

        within = dists <= threshold_um
        gt_idx_all.append(gt_sec_idx[within])
        pred_idx_all.append(pred_sec_idx[nn_idx[within]])

    if len(gt_idx_all) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(pred_idx_all), np.concatenate(gt_idx_all)


# ── Metric computations ──────────────────────────────────────────────────────

def _gene_correlations(pred_X, gt_X):
    """Per-gene Pearson and Spearman correlations.

    Parameters: pred_X, gt_X are (n_matched_cells, n_genes) arrays.
    Returns dict with per-gene arrays and summary stats.
    """
    n_genes = pred_X.shape[1]
    pearsons = np.full(n_genes, np.nan)
    spearmans = np.full(n_genes, np.nan)
    for g in range(n_genes):
        p = pred_X[:, g]
        t = gt_X[:, g]
        if np.std(t) > 0 and np.std(p) > 0:
            pearsons[g] = pearsonr(p, t)[0]
            spearmans[g] = spearmanr(p, t)[0]

    valid_p = pearsons[~np.isnan(pearsons)]
    valid_s = spearmans[~np.isnan(spearmans)]
    return {
        "pearson_per_gene": pearsons,
        "spearman_per_gene": spearmans,
        "pearson_median": float(np.median(valid_p)) if len(valid_p) else np.nan,
        "pearson_mean": float(np.mean(valid_p)) if len(valid_p) else np.nan,
        "pearson_frac_gt05": float(np.mean(valid_p > 0.5)) if len(valid_p) else np.nan,
        "spearman_median": float(np.median(valid_s)) if len(valid_s) else np.nan,
        "spearman_mean": float(np.mean(valid_s)) if len(valid_s) else np.nan,
    }


def _expression_error(pred_X, gt_X):
    """Per-gene RMSE and MAE."""
    diff = pred_X - gt_X
    rmse_per_gene = np.sqrt(np.mean(diff ** 2, axis=0))
    mae_per_gene = np.mean(np.abs(diff), axis=0)
    return {
        "rmse_per_gene": rmse_per_gene,
        "mae_per_gene": mae_per_gene,
        "rmse_median": float(np.median(rmse_per_gene)),
        "mae_median": float(np.median(mae_per_gene)),
    }


def _celltype_metrics(pred_types, gt_types):
    """Accuracy and macro-F1 for cell type prediction."""
    valid = (pred_types != "NA") & (pred_types != "")
    if valid.sum() == 0:
        return {"celltype_accuracy": np.nan, "celltype_f1_macro": np.nan}
    p = pred_types[valid]
    t = gt_types[valid]
    return {
        "celltype_accuracy": float(accuracy_score(t, p)),
        "celltype_f1_macro": float(f1_score(t, p, average="macro", zero_division=0)),
    }


def _ssim_metric(pred, gt_adata, pred_idx, gt_idx, gene_names_pred, gt_gene_names,
                 grid_size=SSIM_GRID_SIZE, n_top_genes=SSIM_TOP_GENES):
    """Spatial SSIM on binned expression grids for top variable genes."""
    from skimage.metrics import structural_similarity as ssim

    # find common genes
    common = np.intersect1d(gene_names_pred, gt_gene_names)
    if len(common) == 0:
        return {"ssim_median": np.nan}

    # select top variable genes from ground truth
    gt_X = gt_adata.X
    if sp.issparse(gt_X):
        gt_X = gt_X.toarray()
    gene_idx_gt = np.array([np.where(gt_gene_names == g)[0][0] for g in common])
    gene_var = np.var(gt_X[:, gene_idx_gt], axis=0)
    top_idx = np.argsort(gene_var)[-min(n_top_genes, len(common)):]
    top_genes = common[top_idx]

    # get matched expression for top genes
    gene_idx_gt_top = np.array([np.where(gt_gene_names == g)[0][0] for g in top_genes])
    gene_idx_pred_top = np.array([np.where(gene_names_pred == g)[0][0] for g in top_genes])

    gt_spatial = gt_adata.obsm["spatial"]
    gt_xy = gt_spatial[gt_idx, :2]
    pred_xy = np.column_stack([pred["x"][pred_idx], pred["y"][pred_idx]])

    # combined extent for consistent binning
    all_xy = np.vstack([gt_xy, pred_xy])
    x_edges = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid_size + 1)

    gt_x_bin = np.clip(np.digitize(gt_xy[:, 0], x_edges) - 1, 0, grid_size - 1)
    gt_y_bin = np.clip(np.digitize(gt_xy[:, 1], y_edges) - 1, 0, grid_size - 1)
    pred_x_bin = np.clip(np.digitize(pred_xy[:, 0], x_edges) - 1, 0, grid_size - 1)
    pred_y_bin = np.clip(np.digitize(pred_xy[:, 1], y_edges) - 1, 0, grid_size - 1)

    gt_expr = gt_adata.X[gt_idx][:, gene_idx_gt_top]
    if sp.issparse(gt_expr):
        gt_expr = gt_expr.toarray()
    pred_expr_full = pred["X"][pred_idx][:, gene_idx_pred_top]
    if sp.issparse(pred_expr_full):
        pred_expr_full = pred_expr_full.toarray()

    ssim_values = []
    for gi in range(len(top_genes)):
        gt_grid = np.zeros((grid_size, grid_size))
        gt_count = np.zeros((grid_size, grid_size))
        pred_grid = np.zeros((grid_size, grid_size))
        pred_count = np.zeros((grid_size, grid_size))

        for i in range(len(gt_idx)):
            gt_grid[gt_y_bin[i], gt_x_bin[i]] += gt_expr[i, gi]
            gt_count[gt_y_bin[i], gt_x_bin[i]] += 1
        for i in range(len(pred_idx)):
            pred_grid[pred_y_bin[i], pred_x_bin[i]] += pred_expr_full[i, gi]
            pred_count[pred_y_bin[i], pred_x_bin[i]] += 1

        gt_count[gt_count == 0] = 1
        pred_count[pred_count == 0] = 1
        gt_mean = gt_grid / gt_count
        pred_mean = pred_grid / pred_count

        data_range = max(gt_mean.max() - gt_mean.min(),
                         pred_mean.max() - pred_mean.min(), 1e-10)
        try:
            s = ssim(gt_mean, pred_mean, data_range=data_range)
            ssim_values.append(s)
        except Exception:
            continue

    return {
        "ssim_median": float(np.median(ssim_values)) if ssim_values else np.nan,
        "ssim_per_gene": np.array(ssim_values),
    }


def _density_metric(pred, gt_adata, holdout_sections, grid_size=SSIM_GRID_SIZE):
    """Cell density recovery: Pearson r of bin-wise counts."""
    gt_mask = gt_adata.obs["section"].isin(holdout_sections)
    gt_xy = gt_adata.obsm["spatial"][gt_mask.values, :2]
    pred_mask = np.isin(pred["section"], holdout_sections)
    pred_xy = np.column_stack([pred["x"][pred_mask], pred["y"][pred_mask]])

    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return {"density_pearson": np.nan}

    all_xy = np.vstack([gt_xy, pred_xy])
    x_edges = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid_size + 1)

    gt_hist, _, _ = np.histogram2d(gt_xy[:, 0], gt_xy[:, 1], bins=[x_edges, y_edges])
    pred_hist, _, _ = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1],
                                      bins=[x_edges, y_edges])

    gt_flat = gt_hist.ravel()
    pred_flat = pred_hist.ravel()
    if np.std(gt_flat) > 0 and np.std(pred_flat) > 0:
        r = pearsonr(gt_flat, pred_flat)[0]
    else:
        r = np.nan
    return {"density_pearson": float(r)}


def _matching_rate(gt_adata, pred, holdout_sections, threshold_um=NN_MATCH_THRESHOLD_UM):
    """Fraction of ground-truth cells matched to a predicted cell within threshold."""
    gt_mask = gt_adata.obs["section"].isin(holdout_sections)
    gt_spatial = gt_adata.obsm["spatial"][gt_mask.values]
    pred_mask = np.isin(pred["section"], holdout_sections)
    pred_xyz = np.column_stack([
        pred["x"][pred_mask], pred["y"][pred_mask], pred["z"][pred_mask]
    ])

    if len(pred_xyz) == 0:
        return {"matching_rate": 0.0}

    total_matched = 0
    total_gt = 0
    gt_sections = gt_adata.obs.loc[gt_mask, "section"].values

    for sec in holdout_sections:
        sec_gt = gt_sections == sec
        sec_pred = pred["section"][pred_mask] == sec
        gt_xy = gt_spatial[sec_gt, :2]
        p_xy = np.column_stack([
            pred["x"][pred_mask][sec_pred],
            pred["y"][pred_mask][sec_pred],
        ])
        if len(p_xy) == 0:
            total_gt += len(gt_xy)
            continue
        tree = cKDTree(p_xy)
        dists, _ = tree.query(gt_xy, k=1)
        total_matched += np.sum(dists <= threshold_um)
        total_gt += len(gt_xy)

    rate = total_matched / total_gt if total_gt > 0 else 0.0
    return {"matching_rate": float(rate)}


def _gene_level_stats(pred_X, gt_X):
    """Per-gene mean and variance correlation (FEAST's primary metric)."""
    pred_mean = pred_X.mean(axis=0)
    gt_mean = gt_X.mean(axis=0)
    pred_var = pred_X.var(axis=0)
    gt_var = gt_X.var(axis=0)

    result = {}
    if np.std(gt_mean) > 0 and np.std(pred_mean) > 0:
        result["gene_mean_pearson"] = float(pearsonr(pred_mean, gt_mean)[0])
    else:
        result["gene_mean_pearson"] = np.nan
    if np.std(gt_var) > 0 and np.std(pred_var) > 0:
        result["gene_var_pearson"] = float(pearsonr(pred_var, gt_var)[0])
    else:
        result["gene_var_pearson"] = np.nan
    return result


def _morans_i_metric(pred_X, pred_xy, gt_X, gt_xy, n_top_genes=20, k=10):
    """Moran's I comparison (SpatialZ/FEAST's primary spatial metric).

    Computes Moran's I for top variable genes on both predicted and GT,
    then returns Pearson r between the two vectors of Moran's I values.
    """
    from sklearn.neighbors import NearestNeighbors

    n_genes = pred_X.shape[1]
    n_use = min(n_top_genes, n_genes)

    # Select top variable genes from GT
    gene_var = np.var(gt_X, axis=0)
    top_idx = np.argsort(gene_var)[-n_use:]

    def compute_morans_i(X, xy, gene_indices, k_nn):
        """Compute Moran's I for each gene using kNN spatial weights."""
        n = X.shape[0]
        if n < k_nn + 1:
            return np.full(len(gene_indices), np.nan)

        nn = NearestNeighbors(n_neighbors=min(k_nn, n - 1))
        nn.fit(xy)
        _, indices = nn.kneighbors(xy)

        morans = []
        for gi in gene_indices:
            z = X[:, gi] - X[:, gi].mean()
            denom = np.sum(z ** 2)
            if denom == 0:
                morans.append(0.0)
                continue
            numer = 0.0
            w_sum = 0.0
            for i in range(n):
                for j_idx in indices[i]:
                    numer += z[i] * z[j_idx]
                    w_sum += 1.0
            morans.append(float(n * numer / (w_sum * denom)) if w_sum > 0 else 0.0)
        return np.array(morans)

    # Subsample for speed
    max_cells = 2000
    if len(pred_xy) > max_cells:
        idx_p = np.random.choice(len(pred_xy), max_cells, replace=False)
        pred_X_sub, pred_xy_sub = pred_X[idx_p], pred_xy[idx_p]
    else:
        pred_X_sub, pred_xy_sub = pred_X, pred_xy
    if len(gt_xy) > max_cells:
        idx_g = np.random.choice(len(gt_xy), max_cells, replace=False)
        gt_X_sub, gt_xy_sub = gt_X[idx_g], gt_xy[idx_g]
    else:
        gt_X_sub, gt_xy_sub = gt_X, gt_xy

    mi_pred = compute_morans_i(pred_X_sub, pred_xy_sub, top_idx, k)
    mi_gt = compute_morans_i(gt_X_sub, gt_xy_sub, top_idx, k)

    valid = ~(np.isnan(mi_pred) | np.isnan(mi_gt))
    if valid.sum() < 3:
        return {"morans_i_median": np.nan}

    return {"morans_i_median": float(np.median(mi_pred[valid]))}


def _dice_density(pred, gt_adata, holdout_sections, grid_size=SSIM_GRID_SIZE):
    """Dice coefficient on binarized density maps (isoST's primary metric)."""
    gt_mask = gt_adata.obs["section"].isin(holdout_sections)
    gt_xy = gt_adata.obsm["spatial"][gt_mask.values, :2]
    pred_mask = np.isin(pred["section"], holdout_sections)
    pred_xy = np.column_stack([pred["x"][pred_mask], pred["y"][pred_mask]])

    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return {"dice_density": 0.0}

    all_xy = np.vstack([gt_xy, pred_xy])
    x_edges = np.linspace(all_xy[:, 0].min(), all_xy[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(all_xy[:, 1].min(), all_xy[:, 1].max(), grid_size + 1)

    gt_hist, _, _ = np.histogram2d(gt_xy[:, 0], gt_xy[:, 1], bins=[x_edges, y_edges])
    pred_hist, _, _ = np.histogram2d(pred_xy[:, 0], pred_xy[:, 1], bins=[x_edges, y_edges])

    # Binarize at median
    gt_bin = (gt_hist > np.median(gt_hist[gt_hist > 0])).astype(float) if gt_hist.max() > 0 else gt_hist
    pred_bin = (pred_hist > np.median(pred_hist[pred_hist > 0])).astype(float) if pred_hist.max() > 0 else pred_hist

    intersection = np.sum(gt_bin * pred_bin)
    dice = 2 * intersection / (np.sum(gt_bin) + np.sum(pred_bin)) if (np.sum(gt_bin) + np.sum(pred_bin)) > 0 else 0.0
    return {"dice_density": float(dice)}


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(prediction_path, h5ad_path, output_path=None):
    """Run all metrics and return/write results dict."""
    pred = load_prediction(prediction_path)
    holdout_sections = pred["holdout_sections"]
    gt = load_ground_truth(h5ad_path, holdout_sections)

    # match cells
    pred_idx, gt_idx = match_cells(pred, gt)
    n_matched = len(pred_idx)

    metrics = {
        "method": pred["method_name"],
        "holdout_sections": holdout_sections,
        "n_holdout_cells_gt": int(gt.n_obs),
        "n_predicted_cells": int(pred["X"].shape[0]),
        "n_matched_cells": n_matched,
    }

    if n_matched > 0:
        # align gene space
        common_genes = np.intersect1d(pred["gene_names"], gt.var_names.values)
        if len(common_genes) == 0:
            metrics["error"] = "no common genes"
            metrics.update({m: np.nan for m in METRIC_NAMES})
        else:
            pred_gene_idx = np.array([
                np.where(pred["gene_names"] == g)[0][0] for g in common_genes
            ])
            gt_gene_idx = np.array([
                np.where(gt.var_names.values == g)[0][0] for g in common_genes
            ])

            pred_X = pred["X"][pred_idx][:, pred_gene_idx]
            if sp.issparse(pred_X):
                pred_X = pred_X.toarray()
            gt_X = gt.X[gt_idx][:, gt_gene_idx]
            if sp.issparse(gt_X):
                gt_X = gt_X.toarray()

            metrics["n_common_genes"] = len(common_genes)
            metrics["frac_genes_predicted"] = len(common_genes) / gt.n_vars

            # 1. correlations
            corr = _gene_correlations(pred_X, gt_X)
            for k in ["pearson_median", "pearson_mean", "pearson_frac_gt05",
                       "spearman_median", "spearman_mean"]:
                metrics[k] = corr[k]

            # 2. expression error
            err = _expression_error(pred_X, gt_X)
            metrics["rmse_median"] = err["rmse_median"]
            metrics["mae_median"] = err["mae_median"]

            # 3. cell type
            gt_cell_types = gt.obs["cell_type"].values[gt_idx] if "cell_type" in gt.obs else np.array([""] * n_matched)
            ct = _celltype_metrics(pred["cell_type"][pred_idx],
                                   gt_cell_types.astype(str))
            metrics.update(ct)

            # 4. SSIM
            ssim_res = _ssim_metric(pred, gt, pred_idx, gt_idx,
                                     pred["gene_names"], gt.var_names.values)
            metrics["ssim_median"] = ssim_res["ssim_median"]

            # 5. density
            dens = _density_metric(pred, gt, holdout_sections)
            metrics["density_pearson"] = dens["density_pearson"]

            # 6. matching rate
            mr = _matching_rate(gt, pred, holdout_sections)
            metrics["matching_rate"] = mr["matching_rate"]

            # 7. gene-level stats (FEAST's metrics)
            gls = _gene_level_stats(pred_X, gt_X)
            metrics.update(gls)

            # 8. Moran's I (SpatialZ/FEAST spatial autocorrelation)
            try:
                pred_xy_matched = np.column_stack([
                    pred["x"][pred_idx], pred["y"][pred_idx]])
                gt_xy_matched = gt.obsm["spatial"][gt_idx, :2]
                mi = _morans_i_metric(pred_X, pred_xy_matched,
                                       gt_X, gt_xy_matched)
                metrics.update(mi)
            except Exception:
                metrics["morans_i_median"] = np.nan

            # 9. Dice density (isoST's metric)
            dice = _dice_density(pred, gt, holdout_sections)
            metrics["dice_density"] = dice["dice_density"]
    else:
        metrics["error"] = "no matched cells"
        metrics.update({m: np.nan for m in METRIC_NAMES})

    # clean NaN for JSON serialization
    def _clean(v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v

    metrics_clean = {k: _clean(v) for k, v in metrics.items()}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics_clean, f, indent=2)

    return metrics_clean


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction vs ground truth")
    parser.add_argument("--prediction", required=True, help="Path to prediction.h5")
    parser.add_argument("--ground-truth", required=True, help="Path to original data.h5ad")
    parser.add_argument("--output", help="Output metrics.json path")
    args = parser.parse_args()

    metrics = evaluate(args.prediction, args.ground_truth, args.output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
