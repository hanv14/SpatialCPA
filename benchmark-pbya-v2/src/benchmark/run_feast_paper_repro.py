"""Reproduce FEAST paper Fig 5c on Zhuang-ABCA-1 slices 5-9.

Paper: Chen, Xie, Hu et al. 2025 (github.com/maiziezhoulab/FEAST)
Protocol:
  - Slices 005-009 from Zhuang-ABCA-1 (1122 genes, log2_normalized)
  - Holdout slices 006, 007, 008; train on 005 + 009
  - Alignment: Spateo morpho_align (PCA 30 comps, cosine dissimilarity)
  - Interpolation: FEAST with sigma=0, ot_method=sinkhorn, ot_reg=0.05
  - t = 0.25 (slice 6), 0.50 (slice 7), 0.75 (slice 8)

Paper Fig 5c targets:
  Slice 6: mean_corr=0.908, var_corr=0.922, morans_i_corr=0.960
  Slice 7: mean_corr=0.977, var_corr=0.923, morans_i_corr=0.943
  Slice 8: mean_corr=0.985, var_corr=0.956, morans_i_corr=0.934

Usage:
    conda run -n bench_feast python src/benchmark/run_feast_paper_repro.py
"""

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

SUBSET_PATH = Path("results/paper_validation_subsets/feast_abca1_slices5to9.h5ad")
OUT_DIR = Path("results/feast_paper_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_morans_i(adata, genes, k=5):
    """Compute per-gene Moran's I (FEAST's implementation from benchmark_scripts)."""
    coords = adata.obsm["spatial"]
    X = adata[:, genes].X
    if sp.issparse(X):
        X = X.toarray()

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(coords))).fit(coords)
    W = nbrs.kneighbors_graph(coords).toarray()
    np.fill_diagonal(W, 0)

    W_sum = W.sum()
    if W_sum == 0:
        return np.full(len(genes), np.nan)

    n = W.shape[0]
    e = X - X.mean(axis=0)
    num = (n / W_sum) * np.einsum("ij,ik,jk->k", W, e, e)
    den = (e ** 2).sum(axis=0) / n
    den[den == 0] = 1.0
    return num / den


def compute_alignment_spateo(sl_below, sl_above, device="cpu"):
    """Compute Spateo morpho_align between two slices (paper's method)."""
    import spateo as st

    # Prepare PCA representation (paper: 30 components, cosine dissimilarity)
    genes_common = sorted(set(sl_below.var_names) & set(sl_above.var_names))
    X_below = sl_below[:, genes_common].X
    X_above = sl_above[:, genes_common].X
    if sp.issparse(X_below):
        X_below = X_below.toarray()
    if sp.issparse(X_above):
        X_above = X_above.toarray()

    X_stack = np.vstack([X_below, X_above])
    n_comp = min(30, X_stack.shape[1], X_stack.shape[0] - 1)
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_stack)

    sl_below.obsm["X_expr_pca"] = X_pca[: sl_below.n_obs]
    sl_above.obsm["X_expr_pca"] = X_pca[sl_below.n_obs :]

    aligned, pis = st.align.morpho_align(
        models=[sl_below, sl_above],
        rep_layer="X_expr_pca",
        rep_field="obsm",
        dissimilarity="cos",
        verbose=True,
        spatial_key="spatial",
        key_added="align_feast",
        device=device,
        SVI_mode=False,
    )

    pi = pis[0]
    if hasattr(pi, "cpu"):
        pi = pi.cpu().numpy()
    return np.array(pi)


def main():
    from FEAST import interpolate_slices, InterpolationConfig

    print("Loading subset...")
    adata = ad.read_h5ad(str(SUBSET_PATH))
    expression_type = adata.uns.get("expression_type", "log2_normalized")
    print(f"  {adata.shape}, expression_type={expression_type}")

    # Identify sections sorted by z
    secs = adata.obs["section"].values.astype(str)
    unique_secs = np.unique(secs)
    sec_z = {}
    for s in unique_secs:
        mask = secs == s
        sec_z[s] = float(np.median(adata.obsm["spatial"][mask, 2]))
    sorted_secs = sorted(unique_secs, key=lambda s: sec_z[s])
    print(f"  Sections: {sorted_secs}")
    for s in sorted_secs:
        print(f"    {s}: z={sec_z[s]:.0f}, n={np.sum(secs == s)}")

    # Extract 2D slices
    def extract_slice(section_label):
        mask = secs == section_label
        slc = adata[mask].copy()
        slc.obsm["spatial"] = slc.obsm["spatial"][:, :2].copy()
        # Reverse log2 for counts layer (FEAST needs raw in layers['counts'])
        X_dense = slc.X.toarray() if sp.issparse(slc.X) else slc.X.copy()
        slc.layers["counts"] = sp.csr_matrix(np.clip(np.power(2, X_dense) - 1, 0, None))
        # .X stays as log2-normalized for FEAST's query averaging
        return slc

    sl_005 = extract_slice(sorted_secs[0])  # anchor below
    sl_009 = extract_slice(sorted_secs[4])  # anchor above

    # Compute Spateo morpho_align
    print("\nComputing Spateo morpho_align...")
    t0 = time.time()
    alignment = compute_alignment_spateo(sl_005, sl_009, device="cpu")
    print(f"  Alignment: {alignment.shape}, took {time.time()-t0:.1f}s")

    # Holdout slices 006, 007, 008 with t = 0.25, 0.50, 0.75
    holdouts = [
        (sorted_secs[1], 0.25),  # slice 006
        (sorted_secs[2], 0.50),  # slice 007
        (sorted_secs[3], 0.75),  # slice 008
    ]

    results = {}
    genes = adata.var_names.tolist()

    for holdout_sec, t_val in holdouts:
        gt = adata[secs == holdout_sec].copy()
        gt.obsm["spatial"] = gt.obsm["spatial"][:, :2].copy()
        gt_X = gt.X.toarray() if sp.issparse(gt.X) else gt.X

        print(f"\n{'='*60}")
        print(f"Holdout {holdout_sec}: t={t_val}")

        config = InterpolationConfig(
            t=t_val,
            random_seed=42,
            use_normalized=True,
            verbose=False,
            boundary_multiplier=1.1,
            sigma=0,  # paper: sigma=0
            feature_weights={"mean": 1.0, "variance": 1.0, "zero_prop": 1.0},
        )

        interpolated = interpolate_slices(sl_005, sl_009, alignment, config)

        pred_X = interpolated.X
        if sp.issparse(pred_X):
            pred_X = pred_X.toarray()

        # Metrics (matching FEAST benchmark_scripts/interpolation_benchmark.py)
        gt_mean = gt_X.mean(axis=0)
        pred_mean = pred_X.mean(axis=0)
        mask_mean = (gt_mean > 1e-10) & (pred_mean > 1e-10)
        mean_corr = pearsonr(gt_mean[mask_mean], pred_mean[mask_mean])[0]

        gt_var = gt_X.var(axis=0)
        pred_var = pred_X.var(axis=0)
        mask_var = (gt_var > 1e-10) & (pred_var > 1e-10)
        var_corr = pearsonr(gt_var[mask_var], pred_var[mask_var])[0]

        # Moran's I correlation
        gt_morans = compute_morans_i(gt, genes)
        # Build spatial for interpolated
        if "spatial" in interpolated.obsm:
            interp_for_morans = interpolated.copy()
        else:
            interp_for_morans = ad.AnnData(X=pred_X, var=gt.var)
            interp_for_morans.obsm["spatial"] = interpolated.obsm.get(
                "spatial", np.zeros((pred_X.shape[0], 2))
            )
        pred_morans = compute_morans_i(interp_for_morans, genes)

        valid = ~np.isnan(gt_morans) & ~np.isnan(pred_morans)
        morans_corr = pearsonr(gt_morans[valid], pred_morans[valid])[0]

        results[holdout_sec] = {
            "t": t_val,
            "mean_correlation": float(mean_corr),
            "variance_correlation": float(var_corr),
            "morans_i_correlation": float(morans_corr),
            "n_gt_cells": gt.n_obs,
            "n_pred_cells": pred_X.shape[0],
        }

        print(f"  mean_corr:  {mean_corr:.4f}  (paper: {[0.908, 0.977, 0.985][holdouts.index((holdout_sec, t_val))]})")
        print(f"  var_corr:   {var_corr:.4f}  (paper: {[0.922, 0.923, 0.956][holdouts.index((holdout_sec, t_val))]})")
        print(f"  morans_i:   {morans_corr:.4f}  (paper: {[0.960, 0.943, 0.934][holdouts.index((holdout_sec, t_val))]})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: FEAST Paper Fig 5c Reproduction")
    print(f"{'Slice':>20} {'mean_corr':>10} {'var_corr':>10} {'morans_i':>10}")
    for sec, r in results.items():
        print(f"{'Ours '+sec:>20} {r['mean_correlation']:>10.4f} {r['variance_correlation']:>10.4f} {r['morans_i_correlation']:>10.4f}")
    paper = {"006": (0.908, 0.922, 0.960), "007": (0.977, 0.923, 0.943), "008": (0.985, 0.956, 0.934)}
    for sec_suffix, (mc, vc, mi) in paper.items():
        sec = [s for s in results if sec_suffix in s][0] if any(sec_suffix in s for s in results) else sec_suffix
        print(f"{'Paper '+sec_suffix:>20} {mc:>10.3f} {vc:>10.3f} {mi:>10.3f}")

    out_path = OUT_DIR / "feast_fig5c_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
