"""Reproduce stVGP paper Supp Fig 39 on breast cancer ST dataset.

Paper: Wang et al. 2026, Advanced Science
Protocol from supplementary:
  - 4 breast cancer ST slices (~250-264 spots each)
  - Mask each slice, predict gene expression from remaining 3
  - Evaluate 6 genes: FN1, COL3A1, LUM, COL1A1, SPARC, PRSS23
  - Metrics: Pearson correlation, Spearman correlation, SSIM, RMSE

Paper Supp Fig 39 values (approximate from bar plots):
  Pearson: ~0.5-0.8 per gene per slice
  Spearman: ~0.4-0.7
  SSIM: ~0.7-0.85
  RMSE: ~0.6-1.0

Usage:
    conda run -n bench_stvgp python src/benchmark/run_stvgp_bc_repro.py
"""

import sys
import time
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim_func

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "stvgp"))
from stVGP.stVGP import (train_stVGP, get_3D_prediction, gene_prediction,
                          adata_preprocess_adjnet, gene_rigid_mapping_alignment)

DATA_PATH = Path("data/processed/st_breast_cancer_stvgp/data.h5ad")
OUT_DIR = Path("results/stvgp_bc_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper params from Tutorial/example_bc.ipynb
HIDDEN = [512, 24]
EPOCHS = 1200
N_HVG = 6000  # paper uses 6000 for BC
N_NEIGHBORS = 10
SEED = 112

# Paper's evaluation genes
EVAL_GENES = ["FN1", "COL3A1", "LUM", "COL1A1", "SPARC", "PRSS23"]


def compute_ssim_gene(gt_coords, gt_expr, pred_coords, pred_expr, grid_size=20):
    """Compute SSIM between two spatial gene expression patterns on a grid."""
    all_coords = np.vstack([gt_coords, pred_coords])
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)

    def to_grid(coords, expr):
        grid = np.zeros((grid_size, grid_size))
        counts = np.zeros((grid_size, grid_size))
        x_norm = (coords[:, 0] - x_min) / (x_max - x_min + 1e-8) * (grid_size - 1)
        y_norm = (coords[:, 1] - y_min) / (y_max - y_min + 1e-8) * (grid_size - 1)
        for xi, yi, val in zip(x_norm.astype(int), y_norm.astype(int), expr):
            xi = min(xi, grid_size - 1)
            yi = min(yi, grid_size - 1)
            grid[xi, yi] += val
            counts[xi, yi] += 1
        counts[counts == 0] = 1
        return grid / counts

    g1 = to_grid(gt_coords, gt_expr)
    g2 = to_grid(pred_coords, pred_expr)
    data_range = max(g1.max() - g1.min(), g2.max() - g2.min(), 1e-8)
    return ssim_func(g1, g2, data_range=data_range)


def run_holdout(adata, holdout_sec, all_sections, device):
    """Mask one slice, train on rest, predict, evaluate per-gene."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = sections == holdout_sec
    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()
    target_z = float(np.median(adata.obsm["spatial"][holdout_mask, 2]))

    print(f"\n  Holdout {holdout_sec}: {holdout_mask.sum()} spots")

    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)

    train_secs = train_adata.obs["section"].values.astype(str)
    unique_secs = sorted(np.unique(train_secs),
                         key=lambda s: np.median(train_adata.obsm["spatial"][train_secs == s, 2]))
    section_adatas = []
    section_z_values = []
    for sec in unique_secs:
        mask = train_secs == sec
        sa = train_adata[mask].copy()
        sa.obsm["spatial_2d"] = sa.obsm["spatial"][:, :2].copy()
        section_adatas.append(sa)
        section_z_values.append(float(np.median(sa.obsm["spatial"][:, 2])))

    sc.pp.highly_variable_genes(train_adata, n_top_genes=min(N_HVG, train_adata.n_vars),
                                flavor="seurat")
    hvg_mask = train_adata.var["highly_variable"].values
    gene_names = train_adata.var_names[hvg_mask].tolist()
    section_adatas = [a[:, hvg_mask].copy() for a in section_adatas]

    # Alignment
    for a in section_adatas:
        a.obsm["spatial"] = a.obsm["spatial_2d"].copy()
    try:
        align_genes = gene_names[:10]
        section_adatas = gene_rigid_mapping_alignment(
            gene_input=align_genes, stadata_input=section_adatas,
            ini_spatial="spatial", add_spatial="align_spatial",
            align_model="single_template_alignment", ref_label=0,
            angle_params=[-60, -40, -20, 0, 20, 40, 60])
        spatial_key = "align_spatial"
    except Exception:
        spatial_key = "spatial"

    # Adjacency
    try:
        slice_matrix, adj_sp = adata_preprocess_adjnet(
            input_adata=section_adatas, align_model="single_template_alignment",
            ref_label=0, spatial_label=spatial_key,
            n_neighbors=N_NEIGHBORS, no_cross=False)
        adj_coo = adj_sp.tocoo()
    except Exception:
        all_X, edge_rows, edge_cols = [], [], []
        offset = 0
        for sa in section_adatas:
            X = sa.X.toarray() if sp.issparse(sa.X) else sa.X
            all_X.append(X.astype(np.float32))
            coords = sa.obsm.get(spatial_key, sa.obsm["spatial_2d"])
            nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, len(sa) - 1))
            nn.fit(coords)
            _, indices = nn.kneighbors(coords)
            for j in range(len(sa)):
                for k in indices[j]:
                    if k != j:
                        edge_rows.append(offset + j)
                        edge_cols.append(offset + k)
            offset += len(sa)
        slice_matrix = np.vstack(all_X)
        n_total = slice_matrix.shape[0]
        adj_coo = sp.coo_matrix((np.ones(len(edge_rows)), (edge_rows, edge_cols)),
                                shape=(n_total, n_total))

    if not isinstance(slice_matrix, np.ndarray):
        slice_matrix = np.array(slice_matrix, dtype=np.float32)

    all_coords = []
    for i, sa in enumerate(section_adatas):
        c2d = sa.obsm.get(spatial_key, sa.obsm["spatial_2d"])
        if c2d.shape[1] > 2:
            c2d = c2d[:, :2]
        all_coords.append(np.hstack([c2d, np.full((len(sa), 1), section_z_values[i])]))
    spatial_coords = np.vstack(all_coords)

    n_total = slice_matrix.shape[0]
    n_genes = slice_matrix.shape[1]
    model_layer = [n_genes, HIDDEN[0], HIDDEN[1], 1]

    # Train
    t0 = time.time()
    result = train_stVGP(
        ST_need_reconstruction_matrix=slice_matrix, all_spatial_net=adj_coo,
        lr=0.001, weight_decay=0.0001, training_epoch=EPOCHS, num_heads=1,
        device=device, hidden_embedding=HIDDEN, random_seed=SEED,
        all_gat=True, VAE_model_select="GAT_VAE")
    embedding = result[1]
    model_params = result[2]
    print(f"    Trained in {time.time()-t0:.0f}s")

    # GP predict
    target_xy = holdout_adata.obsm["spatial"][:, :2]
    n_target = target_xy.shape[0]
    target_3d = np.hstack([target_xy, np.full((n_target, 1), target_z)])

    embedding_pred = get_3D_prediction(
        train_coordinates=spatial_coords, embedding=embedding,
        spatial_pred=target_3d, noise=False, constant_value=1.0, Rbf_value=1024)

    # Decode — full embedding
    full_emb = np.vstack([embedding, embedding_pred])
    n_train = embedding.shape[0]
    pred_nn = NearestNeighbors(n_neighbors=min(N_NEIGHBORS, n_train - 1))
    pred_nn.fit(spatial_coords[:, :2])
    _, pred_indices = pred_nn.kneighbors(target_xy)
    new_rows, new_cols = list(adj_coo.row), list(adj_coo.col)
    for j in range(n_target):
        for k in pred_indices[j]:
            new_rows.append(n_train + j)
            new_cols.append(int(k))
            new_rows.append(int(k))
            new_cols.append(n_train + j)
    adj_combined = sp.coo_matrix(
        (np.ones(len(new_rows)), (new_rows, new_cols)),
        shape=(n_train + n_target, n_train + n_target))
    sm_combined = np.vstack([slice_matrix, np.zeros((n_target, n_genes), dtype=np.float32)])

    try:
        pred_full = gene_prediction(
            slice_matrix=torch.tensor(sm_combined, dtype=torch.float32),
            prediction_embedding=torch.tensor(full_emb, dtype=torch.float32),
            adj_matrix=torch.LongTensor([adj_combined.row.tolist(), adj_combined.col.tolist()]),
            checkpoint=model_params, model_layer=model_layer,
            all_gat=True, device=device, VAE_model_select="GAT_VAE")
        if isinstance(pred_full, torch.Tensor):
            pred_full = pred_full.cpu().numpy()
        pred_expr = pred_full[n_train:]
    except Exception as e:
        print(f"    gene_prediction failed ({e}), linear decode")
        W3 = model_params.get("gat3.weight", model_params.get("gat3.att_src"))
        W4 = model_params.get("gat4.weight", model_params.get("gat4.att_src"))
        z = torch.tensor(embedding_pred, dtype=torch.float32)
        h = torch.relu(z @ model_params["gat3.weight"].T + model_params.get("gat3.bias", 0))
        pred_expr = (h @ model_params["gat4.weight"].T + model_params.get("gat4.bias", 0)).numpy()

    # Evaluate per-gene
    gt = holdout_adata.copy()
    sc.pp.normalize_total(gt, target_sum=1e4)
    sc.pp.log1p(gt)

    gene_results = {}
    for gene in EVAL_GENES:
        if gene not in gt.var_names or gene not in gene_names:
            print(f"    {gene}: not found")
            continue
        gt_idx = list(gt.var_names).index(gene)
        pred_idx = gene_names.index(gene)
        gt_vals = gt.X[:, gt_idx]
        if sp.issparse(gt_vals):
            gt_vals = gt_vals.toarray().flatten()
        else:
            gt_vals = np.asarray(gt_vals).flatten()
        pred_vals = pred_expr[:, pred_idx]

        pcc = pearsonr(gt_vals, pred_vals)[0]
        scc = spearmanr(gt_vals, pred_vals)[0]
        rmse = np.sqrt(np.mean((gt_vals - pred_vals) ** 2))
        ssim_val = compute_ssim_gene(
            holdout_adata.obsm["spatial"][:, :2], gt_vals,
            holdout_adata.obsm["spatial"][:, :2], pred_vals)

        gene_results[gene] = {
            "pearson": float(pcc),
            "spearman": float(scc),
            "rmse": float(rmse),
            "ssim": float(ssim_val),
        }
        print(f"    {gene}: PCC={pcc:.3f} SCC={scc:.3f} RMSE={rmse:.3f} SSIM={ssim_val:.3f}")

    return gene_results


def main():
    print("Loading breast cancer ST...")
    adata = ad.read_h5ad(str(DATA_PATH))
    all_sections = sorted(adata.obs["section"].unique())
    print(f"  {adata.shape}, sections: {all_sections}")

    # Check eval genes exist
    available = [g for g in EVAL_GENES if g in adata.var_names]
    print(f"  Eval genes available: {available} / {EVAL_GENES}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results = {}
    for sec in all_sections:
        results[sec] = run_holdout(adata, sec, all_sections, device)

    # Summary table
    print("\n" + "=" * 80)
    print("stVGP Breast Cancer — Paper Supp Fig 39 Reproduction")
    print("=" * 80)
    print(f"\n{'Gene':>10} {'Slice':>8} {'Pearson':>8} {'Spearman':>9} {'SSIM':>6} {'RMSE':>6}")
    print("-" * 50)
    for sec in all_sections:
        for gene in EVAL_GENES:
            if gene in results[sec]:
                r = results[sec][gene]
                print(f"{gene:>10} {sec:>8} {r['pearson']:>8.3f} {r['spearman']:>9.3f} "
                      f"{r['ssim']:>6.3f} {r['rmse']:>6.3f}")

    print("\nPaper Supp Fig 39 (approximate from bar plots):")
    print("  Pearson: ~0.5-0.8, Spearman: ~0.4-0.7, SSIM: ~0.7-0.85, RMSE: ~0.6-1.0")

    out_path = OUT_DIR / "stvgp_bc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
