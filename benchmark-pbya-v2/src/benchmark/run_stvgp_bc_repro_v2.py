"""Reproduce stVGP BC tutorial exactly (example_bc.ipynb).

Key aspects:
  - Preprocess: concat -> normalize -> log1p -> per-slice HVG 6000 -> intersect
  - Alignment gene selection: R-pipeline replicated in Python
    1. Split each slice into 4 quadrants (spot_make=2)
    2. FindAllMarkers equivalent (scanpy rank_genes_groups per quadrant)
    3. Per-subregion Moran's I (kNN k=6), variance across subregions
    4. Top 4 genes per section by Moran's I variance
  - Alignment: gene_rigid_alignment(ref_label=1)
  - Training: hidden=[512,30], lr=0.0001, epochs=3000, n_neighbors=4, all_gat=False
  - GP prediction: train on all slices, predict all slices (paper protocol)
  - Decode: gene_prediction (reparametrize with logvar from original data)
  - Spatial coords: raw array indices (not micrometers)
  - Gene prediction: model_layer=[n, 512, 30, 1]
"""

import sys
import time
import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "stvgp"))
from stVGP.stVGP import (train_stVGP, get_3D_prediction, gene_prediction,
                          adata_preprocess_adjnet, gene_rigid_alignment, select_gene)

RAW_DIR = Path("data/raw/st_breast_cancer_stvgp")
OUT_DIR = Path("results/stvgp_bc_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_GENES = ["FN1", "COL3A1", "LUM", "COL1A1", "SPARC", "PRSS23"]


def load_raw_data():
    """Load exactly as the tutorial does -- from TSV, raw array coords."""
    slices = []
    for layer_num in [1, 2, 3, 4]:
        df = pd.read_csv(RAW_DIR / f"Layer{layer_num}_BC_count_matrix-1.tsv", sep="\t")
        spot_names = df["Unnamed: 0"].str.split("x")
        x = spot_names.str[0].astype(float).values
        y = spot_names.str[1].astype(float).values
        spatial = np.column_stack([x, y]).astype(np.float32)

        count_matrix = df.values[:, 1:].astype(np.float32)
        var_names = np.array(df.columns[1:])
        obs_names = np.array(df["Unnamed: 0"])

        adata = ad.AnnData(X=count_matrix)
        adata.obs.index = obs_names
        adata.var.index = var_names
        adata.obs["samples_idx"] = f"Layer{layer_num}"
        adata.obsm["spatial"] = spatial
        adata.obsm["loc_use"] = spatial.copy()
        adata.obsm["raw_count"] = count_matrix.copy()
        slices.append(adata)
        print(f"  Layer{layer_num}: {adata.shape}")
    return slices


def preprocess_tutorial(slices):
    """Exactly follow tutorial Cell 1 preprocessing."""
    adata_concat = ad.concat(slices)
    sc.pp.filter_cells(adata_concat, min_genes=1)
    sc.pp.filter_genes(adata_concat, min_cells=1)
    sc.pp.normalize_total(adata_concat, inplace=True)
    sc.pp.log1p(adata_concat)

    slice_names = ["Layer1", "Layer2", "Layer3", "Layer4"]
    slice_list = [adata_concat[adata_concat.obs["samples_idx"] == s].copy() for s in slice_names]

    # Per-slice HVG 6000 -> intersection
    hvg_lists = []
    for sl in slice_list:
        sc.pp.highly_variable_genes(sl, flavor="seurat", n_top_genes=6000)
        hvg_lists.append(set(sl.var_names[sl.var["highly_variable"]]))

    common_genes = hvg_lists[0]
    for h in hvg_lists[1:]:
        common_genes = common_genes & h
    common_genes = sorted(common_genes)
    print(f"  Common HVG genes: {len(common_genes)}")

    adata_list = [sl[:, common_genes].copy() for sl in slice_list]
    for a in adata_list:
        sc.pp.filter_cells(a, min_genes=1)
    return adata_list, common_genes


def _compute_morans_i_manual(values, coords, k=6):
    """Compute Moran's I using kNN spatial weights (manual implementation).

    Replicates spdep::moran.mc with k nearest neighbors.
    """
    n = len(values)
    if n < k + 1:
        return 0.0
    values = np.asarray(values, dtype=np.float64)
    mean_val = values.mean()
    z = values - mean_val
    ss = np.sum(z ** 2)
    if ss == 0:
        return 0.0

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Moran's I = (n / W) * (sum_i sum_j w_ij * z_i * z_j) / (sum_i z_i^2)
    # Vectorized: sum of z[i] * z[neighbors_of_i] for all i
    numerator = np.sum(z[:, None] * z[indices])
    W = n * k
    morans_i = (n / W) * (numerator / ss)
    return morans_i


def _assign_quadrants(coords, spot_make=2):
    """Assign spots to spatial quadrants, replicating stVGP select_gene().

    spot_make=2 yields 4 quadrants. Returns integer labels 1..spot_make^2.
    """
    x, y = coords[:, 0], coords[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_interval = (x_max - x_min) / spot_make
    y_interval = (y_max - y_min) / spot_make

    x_idx = np.floor((x - x_min) / x_interval).astype(int)
    y_idx = np.floor((y - y_min) / y_interval).astype(int)
    # Clamp edge cases (exactly at max boundary)
    x_idx = np.clip(x_idx, 0, spot_make - 1)
    y_idx = np.clip(y_idx, 0, spot_make - 1)

    # Same formula as stVGP: (X_indices * spot_make) + (Y_indices + 1)
    labels = (x_idx * spot_make) + (y_idx + 1)
    return labels


def align_tutorial(adata_list):
    """Use R (Seurat + spdep) for gene selection, exactly as the paper does.

    Pipeline:
      1. Python: select_gene(save_data=True) → writes select_gene_4.txt per slice
      2. R: Seurat FindAllMarkers → spdep moran.mc → gene_morans_4.txt
      3. Python: read top 4 genes → gene_rigid_alignment
    """
    import subprocess
    import tempfile

    spot_make = 2
    n_quadrants = spot_make * spot_make

    tmp_dir = tempfile.mkdtemp(prefix="stvgp_bc_genes_")
    r_script = str(Path(__file__).parent / "stvgp_select_genes.R")

    # Step 1: Python select_gene to produce select_gene_4.txt per slice
    print("  Step 1: select_gene (Python)...")
    for ref_index in range(len(adata_list)):
        save_dir = os.path.join(tmp_dir, str(ref_index))
        os.makedirs(save_dir, exist_ok=True)
        adata_list = select_gene(
            adata_list, ref_adata_num=ref_index,
            spot_make=spot_make, save_data=True,
            key_words="loc_use", savepath=save_dir + "/")

    # Verify files were created
    for i in range(len(adata_list)):
        f = os.path.join(tmp_dir, str(i), f"select_gene_{n_quadrants}.txt")
        if os.path.exists(f):
            lines = sum(1 for _ in open(f))
            print(f"    Slice {i}: {f} ({lines} lines)")
        else:
            print(f"    Slice {i}: MISSING {f}")

    # Step 2: R script for FindAllMarkers + Moran's I
    print("  Step 2: R FindAllMarkers + Moran's I...")
    cmd = [
        "conda", "run", "-n", "bench_stvgp", "--no-capture-output",
        "Rscript", r_script, tmp_dir, tmp_dir, str(len(adata_list)), str(spot_make)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    print(result.stdout)
    if result.returncode != 0:
        print(f"  R STDERR: {result.stderr[:500]}")

    # Step 3: Read top 4 genes per slice
    print("  Step 3: Reading Moran's I results...")
    gene_input_list = []
    for i in range(len(adata_list)):
        morans_file = os.path.join(tmp_dir, str(i), f"gene_morans_{n_quadrants}.txt")
        if os.path.exists(morans_file):
            data = np.genfromtxt(morans_file, dtype=str, skip_header=1, delimiter="\t")
            if data.size == 0 or (data.ndim == 1 and data.shape[0] < 2):
                print(f"    Layer{i+1}: R returned empty results, using Python fallback")
                gene_input_list.append(_python_morans_fallback(adata_list[i], spot_make))
                continue
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # Sort by value (column 1) descending, take top 4
            sorted_idx = np.argsort(data[:, 1].astype(float))[::-1]
            top_idx = sorted_idx[:4]
            top_genes = data[top_idx, 0]
            top_vals = data[top_idx, 1].astype(float)
            gene_input_list.append(np.array(top_genes))
            print(f"    Layer{i+1}: {list(top_genes)} = {[round(float(v),4) for v in top_vals]}")
        else:
            # Fallback: use Python Moran's I
            print(f"    Layer{i+1}: R failed, using Python fallback")
            gene_input_list.append(_python_morans_fallback(adata_list[i], spot_make))

    # Step 4: Alignment
    print("  Step 4: gene_rigid_alignment (ref_label=1)...")
    for a in adata_list:
        a.obsm["spatial"] = a.obsm["loc_use"].copy()

    try:
        aligned = gene_rigid_alignment(
            gene_input=gene_input_list[0],
            stadata_input=adata_list,
            ini_spatial="loc_use",
            add_spatial="align_spatial",
            align_model="single_template_alignment",
            gene_input_list=gene_input_list,
            ref_label=1,
            align_method="optimize",
            icp_iterations=20,
            maxiter=300)
        if aligned is not None:
            adata_list = aligned
        print("  Alignment done")
    except Exception as e:
        print(f"  Alignment failed ({e}), using raw coords")
        for a in adata_list:
            a.obsm["align_spatial"] = a.obsm["loc_use"].copy()

    return adata_list


def _python_morans_fallback(adata, spot_make=2):
    """Fallback: compute Moran's I variance in Python if R fails."""
    coords = adata.obsm["loc_use"]
    labels = _assign_quadrants(coords, spot_make=spot_make)
    unique_labels = sorted(set(labels))

    gene_variance = {}
    for gene in adata.var_names:
        vals = adata[:, gene].X
        if sp.issparse(vals):
            vals = vals.toarray()
        vals = vals.flatten().astype(np.float64)

        morans_per_quad = []
        for lab in unique_labels:
            mask = labels == lab
            sub_vals = vals[mask]
            sub_coords = coords[mask]
            if np.all(sub_vals == 0) or len(sub_vals) < 7:
                morans_per_quad.append(0.0)
            else:
                morans_per_quad.append(_compute_morans_i_manual(sub_vals, sub_coords, k=6))
        arr = np.array(morans_per_quad)
        gene_variance[gene] = np.sum((arr - arr.mean()) ** 2)

    top4 = sorted(gene_variance.keys(), key=lambda g: gene_variance[g], reverse=True)[:4]
    return np.array(top4)


def main():
    print("Loading raw BC data...")
    slices = load_raw_data()

    print("\nPreprocessing (tutorial protocol)...")
    adata_list, common_genes = preprocess_tutorial(slices)

    print("\nAligning (Moran's I variance gene selection)...")
    adata_list = align_tutorial(adata_list)

    # Check eval genes
    available_eval = [g for g in EVAL_GENES if g in common_genes]
    print(f"\n  Eval genes in common HVGs: {available_eval}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build adjacency and train (tutorial Cell 7 params)
    print("\nBuilding adjacency (n_neighbors=4, ref_label=1)...")
    slice_matrix, adj_sp = adata_preprocess_adjnet(
        input_adata=adata_list, align_model="single_template_alignment",
        spatial_label="align_spatial", n_neighbors=4, ref_label=1, no_cross=False)
    adj_coo = adj_sp.tocoo()
    if not isinstance(slice_matrix, np.ndarray):
        slice_matrix = np.array(slice_matrix, dtype=np.float32)
    print(f"  slice_matrix: {slice_matrix.shape}, edges: {adj_coo.nnz}")

    # 3D coords (tutorial Cell 12)
    z_vals = [0., 10., 20., 30.]
    all_coords = []
    for i, a in enumerate(adata_list):
        xy = a.obsm["align_spatial"]
        all_coords.append(np.column_stack([xy, np.full(len(a), z_vals[i])]))
    spatial_coords = np.vstack(all_coords)

    # Section boundaries
    boundaries = [0]
    for a in adata_list:
        boundaries.append(boundaries[-1] + a.n_obs)

    # Train (tutorial Cell 7)
    print(f"\nTraining stVGP: hidden=[512,30], lr=0.0001, epochs=3000, seed=512...")
    t0 = time.time()
    result = train_stVGP(
        ST_need_reconstruction_matrix=slice_matrix, all_spatial_net=adj_coo,
        lr=0.0001, weight_decay=0.0001, training_epoch=3000, num_heads=1,
        device=device, hidden_embedding=[512, 30], random_seed=512,
        all_gat=False, VAE_model_select="GAT_VAE")
    embedding = result[1]
    model_params = result[2]
    print(f"  Trained in {time.time()-t0:.0f}s")

    # Gene prediction (tutorial Cell 13-16 protocol)
    n_genes = slice_matrix.shape[1]
    model_layer = [n_genes, 512, 30, 1]
    edge_list = [adj_coo.row.tolist(), adj_coo.col.tolist()]
    adj_tensor = torch.LongTensor(edge_list)

    # Paper protocol: GP-predict ALL slices' embeddings (train on all, predict all),
    # then decode with gene_prediction (which reparametrizes with logvar from
    # original slice_matrix). This matches the tutorial's Cell 13-16 flow.
    # The GP with RBF(512) on array-index coords produces smoothed embeddings;
    # the decoder reconstructs expression using logvar from the original data.
    N_SAMPLES = 50

    print(f"\n  GP: train on all slices, predict all (Rbf=512)...")
    all_pred_emb = get_3D_prediction(
        train_coordinates=spatial_coords, embedding=embedding,
        spatial_pred=spatial_coords, noise=False, constant_value=1.0, Rbf_value=512)

    # Diagnostic: GP prediction quality
    for i in range(4):
        start, end = boundaries[i], boundaries[i+1]
        true_slice = embedding[start:end]
        pred_slice = all_pred_emb[start:end]
        corrs = [pearsonr(true_slice[:, d], pred_slice[:, d])[0]
                 for d in range(embedding.shape[1])]
        print(f"    Layer{i+1}: emb_corr={np.nanmean(corrs):.4f}, "
              f"pred_var={pred_slice.var(axis=0).mean():.6f}, "
              f"true_var={true_slice.var(axis=0).mean():.6f}")

    # Decode with gene_prediction, averaged over N_SAMPLES for stability
    prediction_embedding_t = torch.tensor(all_pred_emb, dtype=torch.float32)
    print(f"\n  Decoding (N_SAMPLES={N_SAMPLES})...")
    preds = []
    for sample_i in range(N_SAMPLES):
        torch.manual_seed(512 + sample_i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(512 + sample_i)
        pred_expr_i = gene_prediction(
            slice_matrix=torch.tensor(slice_matrix, dtype=torch.float32),
            prediction_embedding=prediction_embedding_t,
            adj_matrix=adj_tensor,
            checkpoint=model_params,
            model_layer=model_layer,
            all_gat=False, device=device, VAE_model_select="GAT_VAE")
        if isinstance(pred_expr_i, torch.Tensor):
            pred_expr_i = pred_expr_i.cpu().numpy()
        preds.append(pred_expr_i)
    pred_avg = np.mean(preds, axis=0)

    # Evaluate per slice
    results = {}
    for holdout_idx in range(4):
        sec_name = f"Layer{holdout_idx+1}"
        start, end = boundaries[holdout_idx], boundaries[holdout_idx + 1]
        pred_holdout = pred_avg[start:end]
        gt = slice_matrix[start:end]

        gene_results = {}
        for gene in EVAL_GENES:
            if gene not in common_genes:
                continue
            g_idx = common_genes.index(gene)
            gt_vals = gt[:, g_idx]
            pred_vals = pred_holdout[:, g_idx]
            pcc = pearsonr(gt_vals, pred_vals)[0]
            scc = spearmanr(gt_vals, pred_vals)[0]
            rmse = np.sqrt(np.mean((gt_vals - pred_vals) ** 2))
            gene_results[gene] = {"pearson": float(pcc), "spearman": float(scc), "rmse": float(rmse)}

        results[sec_name] = gene_results
        print(f"\n  {sec_name}:")
        for g, r in gene_results.items():
            print(f"    {g}: PCC={r['pearson']:.3f} SCC={r['spearman']:.3f} RMSE={r['rmse']:.3f}")

    # Paper approximate Pearson values (read from Supp Fig 39 bar plots)
    paper_pearson = {
        "Layer1": {"FN1": 0.72, "COL3A1": 0.68, "LUM": 0.65, "COL1A1": 0.70, "SPARC": 0.62, "PRSS23": 0.55},
        "Layer2": {"FN1": 0.65, "COL3A1": 0.60, "LUM": 0.58, "COL1A1": 0.62, "SPARC": 0.55, "PRSS23": 0.50},
        "Layer3": {"FN1": 0.68, "COL3A1": 0.62, "LUM": 0.60, "COL1A1": 0.65, "SPARC": 0.58, "PRSS23": 0.52},
        "Layer4": {"FN1": 0.75, "COL3A1": 0.70, "LUM": 0.68, "COL1A1": 0.72, "SPARC": 0.65, "PRSS23": 0.58},
    }

    # Summary table: Paper vs Ours
    print("\n" + "=" * 80)
    print("stVGP BC Reproduction -- Paper vs Ours (Pearson correlation)")
    print("=" * 80)
    print(f"\n{'Gene':>10} {'Slice':>8} {'Paper':>8} {'Ours':>8} {'Diff':>8}")
    print("-" * 50)
    all_paper, all_ours = [], []
    for sec in ["Layer1", "Layer2", "Layer3", "Layer4"]:
        for gene in EVAL_GENES:
            p_val = paper_pearson.get(sec, {}).get(gene, float("nan"))
            o_val = results.get(sec, {}).get(gene, {}).get("pearson", float("nan"))
            diff = o_val - p_val if not (np.isnan(o_val) or np.isnan(p_val)) else float("nan")
            print(f"{gene:>10} {sec:>8} {p_val:>8.3f} {o_val:>8.3f} {diff:>+8.3f}")
            if not np.isnan(p_val):
                all_paper.append(p_val)
            if not np.isnan(o_val):
                all_ours.append(o_val)
    print("-" * 50)
    if all_paper and all_ours:
        print(f"{'MEAN':>10} {'':>8} {np.mean(all_paper):>8.3f} {np.mean(all_ours):>8.3f} {np.mean(all_ours)-np.mean(all_paper):>+8.3f}")
    print()

    # Full results with Spearman and RMSE
    print(f"{'Gene':>10} {'Slice':>8} {'Pearson':>8} {'Spearman':>9} {'RMSE':>6}")
    print("-" * 45)
    for sec in sorted(results.keys()):
        for gene in EVAL_GENES:
            if gene in results[sec]:
                r = results[sec][gene]
                print(f"{gene:>10} {sec:>8} {r['pearson']:>8.3f} {r['spearman']:>9.3f} {r['rmse']:>6.3f}")

    out_path = OUT_DIR / "stvgp_bc_results_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
