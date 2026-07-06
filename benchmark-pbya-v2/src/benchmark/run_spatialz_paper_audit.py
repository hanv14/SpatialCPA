"""Audit SpatialZ on the paper's exact shipped data.

Runs two evaluations:
  1. Holdout: for each interior section (0.09, 0.14, 0.19), hold it out,
     generate from flanking pair, compare with ground truth.
  2. Dense generation: paper's exact protocol — Generate_multiple_slices
     with num_sim=3 between each pair, then compute Moran's I / Geary's C
     on both original sparse and densified data.

Uses the 5 pre-aligned MERFISH hypothalamus sections shipped with SpatialZ.
"""

import json
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "spatialz" / "SpatialZ_code"
DATA_DIR = TOOLS_DIR / "data"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "spatialz_paper_audit"


def load_paper_data():
    """Load the 5 sections shipped with SpatialZ demo."""
    files = {
        "0.04": DATA_DIR / "merfish_4_paste.h5ad",
        "0.09": DATA_DIR / "merfish_9_paste.h5ad",
        "0.14": DATA_DIR / "merfish_14_paste.h5ad",
        "0.19": DATA_DIR / "merfish_19_paste.h5ad",
        "0.24": DATA_DIR / "merfish_24_paste.h5ad",
    }
    adatas = {}
    for sid, path in files.items():
        a = ad.read_h5ad(str(path))
        # Ensure dense X (SpatialZ requires it)
        if sp.issparse(a.X):
            a.X = a.X.toarray()
        adatas[sid] = a
        print(f"  Loaded {sid}: {a.shape[0]} cells, {a.shape[1]} genes")
    return adatas


def compute_spatial_metrics(adata, genes=None, n_neighbors=6):
    """Compute Moran's I and Geary's C per gene.

    These are the paper's primary metrics for evaluating spatial coherence.
    """
    if genes is None:
        genes = adata.var_names.tolist()

    # Build spatial graph
    temp = adata.copy()
    sc.pp.neighbors(temp, n_neighbors=n_neighbors, use_rep="spatial",
                    key_added="spatial_neighbors")

    morans = {}
    gearys = {}
    for gene in genes:
        try:
            mi = sc.metrics.morans_i(temp, vals=temp[:, gene].X.flatten(),
                                      neighbors_key="spatial_neighbors")
            gc = sc.metrics.gearys_c(temp, vals=temp[:, gene].X.flatten(),
                                      neighbors_key="spatial_neighbors")
            morans[gene] = float(mi)
            gearys[gene] = float(gc)
        except Exception:
            morans[gene] = float("nan")
            gearys[gene] = float("nan")

    return morans, gearys


def run_holdout_evaluation(adatas):
    """Hold out each interior section, predict from flanking pair."""
    sys.path.insert(0, str(TOOLS_DIR))
    from SpatialZ import Generate_spatialz

    ordered_ids = ["0.04", "0.09", "0.14", "0.19", "0.24"]
    holdout_results = {}

    for i in range(1, len(ordered_ids) - 1):
        target_id = ordered_ids[i]
        below_id = ordered_ids[i - 1]
        above_id = ordered_ids[i + 1]

        gt = adatas[target_id]
        sl_below = adatas[below_id].copy()
        sl_above = adatas[above_id].copy()

        print(f"\n  Holdout {target_id}: generating from {below_id} + {above_id}")

        t0 = time.time()
        synthesized = Generate_spatialz(
            adata1=sl_above,
            adata2=sl_below,
            adata1_id=above_id,
            adata2_id=below_id,
            alpha=0.5,
            device="cuda:0",
            seed=42,
            lr=1e-5,
            nb_iter_max=1000,
            k_sam=50,
            syn_mode="default",
            cell_type_key="cell_class",
            verbose=True,
        )
        wall_time = time.time() - t0

        # Compute metrics: per-gene Pearson between synthesized and ground truth means
        gt_means = gt.X.mean(axis=0).flatten()
        syn_means = synthesized.X.mean(axis=0).flatten()
        if hasattr(gt_means, 'A1'):
            gt_means = gt_means.A1
        if hasattr(syn_means, 'A1'):
            syn_means = syn_means.A1

        gene_mean_r = float(np.corrcoef(gt_means, syn_means)[0, 1])

        gt_vars = gt.X.var(axis=0).flatten()
        syn_vars = synthesized.X.var(axis=0).flatten()
        if hasattr(gt_vars, 'A1'):
            gt_vars = gt_vars.A1
        if hasattr(syn_vars, 'A1'):
            syn_vars = syn_vars.A1
        gene_var_r = float(np.corrcoef(gt_vars, syn_vars)[0, 1])

        # Cell-type proportions
        gt_ct = gt.obs["cell_class"].value_counts(normalize=True).sort_index()
        ct_key_syn = "cell_class" if "cell_class" in synthesized.obs else "cell_type"
        syn_ct = synthesized.obs[ct_key_syn].value_counts(normalize=True).sort_index()
        common_ct = gt_ct.index.intersection(syn_ct.index)
        if len(common_ct) > 2:
            ct_r = float(np.corrcoef(
                gt_ct.reindex(common_ct, fill_value=0),
                syn_ct.reindex(common_ct, fill_value=0)
            )[0, 1])
        else:
            ct_r = float("nan")

        # Moran's I on synthesized slice
        syn_morans, syn_gearys = compute_spatial_metrics(
            synthesized, genes=gt.var_names.tolist()[:20])
        gt_morans, gt_gearys = compute_spatial_metrics(
            gt, genes=gt.var_names.tolist()[:20])

        holdout_results[target_id] = {
            "n_gt_cells": gt.n_obs,
            "n_syn_cells": synthesized.n_obs,
            "gene_mean_pearson": gene_mean_r,
            "gene_var_pearson": gene_var_r,
            "celltype_proportion_r": ct_r,
            "morans_i_median_syn": float(np.nanmedian(list(syn_morans.values()))),
            "morans_i_median_gt": float(np.nanmedian(list(gt_morans.values()))),
            "gearys_c_median_syn": float(np.nanmedian(list(syn_gearys.values()))),
            "gearys_c_median_gt": float(np.nanmedian(list(gt_gearys.values()))),
            "wall_time": wall_time,
        }

        print(f"    gene_mean_r={gene_mean_r:.4f}, gene_var_r={gene_var_r:.4f}, "
              f"ct_r={ct_r:.4f}")
        print(f"    Moran's I: GT={holdout_results[target_id]['morans_i_median_gt']:.4f}, "
              f"Syn={holdout_results[target_id]['morans_i_median_syn']:.4f}")
        print(f"    Geary's C: GT={holdout_results[target_id]['gearys_c_median_gt']:.4f}, "
              f"Syn={holdout_results[target_id]['gearys_c_median_syn']:.4f}")

    return holdout_results


def run_dense_generation(adatas):
    """Paper's exact protocol: Generate_multiple_slices with num_sim=3."""
    sys.path.insert(0, str(TOOLS_DIR))
    from SpatialZ import Generate_multiple_slices

    ordered_ids = ["0.04", "0.09", "0.14", "0.19", "0.24"]
    adata_list = [adatas[sid].copy() for sid in ordered_ids]

    print("\n  Running Generate_multiple_slices (paper protocol, num_sim=3)...")
    t0 = time.time()
    dense_adata = Generate_multiple_slices(
        adata_list=adata_list,
        num_sim_list=[3, 3, 3, 3],  # 3 interpolated between each pair
        adatas_id_list=ordered_ids,
        save_path=str(RESULTS_DIR / "dense_output"),
        device="cuda:0",
        nb_iter_max=1000,
        cell_type_key="cell_class",
        k_sam=50,
        syn_mode="default",
        add_obs_list=["neuron_class", "domain", "Region", "annotation"],
        include_raw=True,
    )
    wall_time = time.time() - t0
    print(f"  Done in {wall_time:.1f}s: {dense_adata.shape}")

    # Compute Moran's I per gene on original sparse vs densified
    # Original: just the 5 real sections concatenated
    original = ad.concat([adatas[sid] for sid in ordered_ids], join="outer")
    original.obsm["spatial"] = np.vstack([
        adatas[sid].obsm["spatial"] for sid in ordered_ids
    ])
    # Add z-coord for 3D structure
    z_map = {sid: 400 * (i + 1) for i, sid in enumerate(ordered_ids)}
    z_coords = []
    for sid in ordered_ids:
        z_coords.extend([z_map[sid]] * adatas[sid].n_obs)
    original.obs["Z"] = z_coords

    # Select top-variable genes for Moran's I comparison
    gene_var = np.var(original.X, axis=0)
    if hasattr(gene_var, 'A1'):
        gene_var = gene_var.A1
    top_genes = original.var_names[np.argsort(gene_var)[-30:]].tolist()

    print(f"\n  Computing Moran's I on original ({original.n_obs} cells) "
          f"and dense ({dense_adata.n_obs} cells) for {len(top_genes)} genes...")

    orig_morans, orig_gearys = compute_spatial_metrics(original, genes=top_genes)

    # Dense: real + synthetic cells
    dense_morans, dense_gearys = compute_spatial_metrics(dense_adata, genes=top_genes)

    return {
        "n_original_cells": original.n_obs,
        "n_dense_cells": dense_adata.n_obs,
        "n_slices_original": 5,
        "n_slices_dense": len(dense_adata.obs["slice_id"].unique()),
        "wall_time": wall_time,
        "per_gene_morans_i": {
            g: {"original": orig_morans[g], "dense": dense_morans[g]}
            for g in top_genes
        },
        "per_gene_gearys_c": {
            g: {"original": orig_gearys[g], "dense": dense_gearys[g]}
            for g in top_genes
        },
        "morans_i_median_original": float(np.nanmedian(list(orig_morans.values()))),
        "morans_i_median_dense": float(np.nanmedian(list(dense_morans.values()))),
        "gearys_c_median_original": float(np.nanmedian(list(orig_gearys.values()))),
        "gearys_c_median_dense": float(np.nanmedian(list(dense_gearys.values()))),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "dense_output").mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SpatialZ Paper Audit — Using Paper's Exact Shipped Data")
    print("=" * 60)

    print("\nLoading paper data...")
    adatas = load_paper_data()

    # ── Holdout evaluation ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 1: Holdout Evaluation (interior sections)")
    print("=" * 60)
    holdout_results = run_holdout_evaluation(adatas)

    # ── Dense generation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PART 2: Dense Generation (paper protocol)")
    print("=" * 60)
    dense_results = run_dense_generation(adatas)

    # ── Save results ──────────────────────────────────────────────
    all_results = {
        "holdout": holdout_results,
        "dense_generation": dense_results,
    }
    out_path = RESULTS_DIR / "spatialz_paper_audit.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")

    # ── Print summary ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n--- Holdout Evaluation ---")
    print(f"{'Section':<10} {'gene_mean_r':>12} {'gene_var_r':>12} {'ct_r':>8} "
          f"{'MI_gt':>8} {'MI_syn':>8} {'GC_gt':>8} {'GC_syn':>8}")
    for sid, r in holdout_results.items():
        print(f"{sid:<10} {r['gene_mean_pearson']:>12.4f} {r['gene_var_pearson']:>12.4f} "
              f"{r['celltype_proportion_r']:>8.4f} "
              f"{r['morans_i_median_gt']:>8.4f} {r['morans_i_median_syn']:>8.4f} "
              f"{r['gearys_c_median_gt']:>8.4f} {r['gearys_c_median_syn']:>8.4f}")

    print("\n--- Dense Generation (Paper Protocol) ---")
    print(f"  Original: {dense_results['n_original_cells']} cells, "
          f"{dense_results['n_slices_original']} slices")
    print(f"  Dense:    {dense_results['n_dense_cells']} cells, "
          f"{dense_results['n_slices_dense']} slices")
    print(f"  Moran's I median: original={dense_results['morans_i_median_original']:.4f}, "
          f"dense={dense_results['morans_i_median_dense']:.4f}")
    print(f"  Geary's C median: original={dense_results['gearys_c_median_original']:.4f}, "
          f"dense={dense_results['gearys_c_median_dense']:.4f}")


if __name__ == "__main__":
    main()
