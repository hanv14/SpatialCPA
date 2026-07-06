"""Reproduce stVGP paper Supp Fig 16 on DLPFC (slices 151674-151675).

Paper: Wang et al. 2026, Advanced Science
Protocol from supplementary:
  - 4 DLPFC slices (151673-151676)
  - Mask each interior slice (151674, 151675), predict from remaining
  - Evaluate MOBP gene: Pearson correlation, RMSE, SSIM
  - Paper Supp Fig 16: stVGP MOBP PCC=0.72, RMSE=0.81
  - Paper Supp Fig 17: stVGP MOBP PCC=0.682, RMSE=0.806, SSIM=0.553

Usage:
    conda run -n bench_stvgp python src/benchmark/run_stvgp_dlpfc_repro.py
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
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "stvgp"))
from stVGP.stVGP import (train_stVGP, get_3D_prediction, gene_prediction,
                          adata_preprocess_adjnet, gene_rigid_mapping_alignment)

DATA_PATH = Path("data/processed/visium_dlpfc_stvgp/data.h5ad")
OUT_DIR = Path("results/stvgp_dlpfc_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper params from Tutorial/example_dlpfc.ipynb
HIDDEN = [512, 24]
EPOCHS = 1200
N_HVG = 5000
N_NEIGHBORS = 10
SEED = 112


def run_one_holdout(adata, holdout_sec, all_sections, device):
    """Mask one slice, train on rest, predict, evaluate."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = sections == holdout_sec
    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()
    target_z = float(np.median(adata.obsm["spatial"][holdout_mask, 2]))

    print(f"\n  Holdout {holdout_sec}: {holdout_mask.sum()} cells, z={target_z:.0f}")

    # Normalize
    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)

    # Per-section AnnData
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

    # HVG
    sc.pp.highly_variable_genes(train_adata, n_top_genes=N_HVG, flavor="seurat")
    hvg_mask = train_adata.var["highly_variable"].values
    gene_names = train_adata.var_names[hvg_mask].tolist()
    section_adatas = [a[:, hvg_mask].copy() for a in section_adatas]

    # Alignment
    align_genes = gene_names[:10]
    for a in section_adatas:
        a.obsm["spatial"] = a.obsm["spatial_2d"].copy()
    try:
        section_adatas = gene_rigid_mapping_alignment(
            gene_input=align_genes, stadata_input=section_adatas,
            ini_spatial="spatial", add_spatial="align_spatial",
            align_model="single_template_alignment", ref_label=0,
            angle_params=[-60, -40, -20, 0, 20, 40, 60])
        spatial_key = "align_spatial"
    except Exception as e:
        print(f"    Alignment failed ({e}), using raw coords")
        spatial_key = "spatial"

    # Adjacency
    try:
        slice_matrix, adj_sp = adata_preprocess_adjnet(
            input_adata=section_adatas, align_model="single_template_alignment",
            ref_label=0, spatial_label=spatial_key,
            n_neighbors=N_NEIGHBORS, no_cross=False)
        adj_coo = adj_sp.tocoo()
    except Exception:
        # Manual fallback
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

    # 3D coords
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
    print(f"    Training: {n_total} cells, {n_genes} genes, hidden={HIDDEN}, epochs={EPOCHS}")
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

    gp_coords = spatial_coords
    gp_embedding = embedding
    if len(spatial_coords) > 5000:
        idx = np.random.choice(len(spatial_coords), 5000, replace=False)
        gp_coords = spatial_coords[idx]
        gp_embedding = embedding[idx]

    embedding_pred = get_3D_prediction(
        train_coordinates=gp_coords, embedding=gp_embedding,
        spatial_pred=target_3d, noise=False, constant_value=1.0, Rbf_value=1024)

    # Decode — use full embedding approach
    full_emb = np.vstack([embedding, embedding_pred])
    n_train = embedding.shape[0]
    # Add edges for prediction cells
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
    n_combined = n_train + n_target
    adj_combined = sp.coo_matrix(
        (np.ones(len(new_rows)), (new_rows, new_cols)), shape=(n_combined, n_combined))
    sm_combined = np.vstack([slice_matrix, np.zeros((n_target, n_genes), dtype=np.float32)])
    ml_combined = [n_genes, HIDDEN[0], HIDDEN[1], 1]

    try:
        pred_full = gene_prediction(
            slice_matrix=torch.tensor(sm_combined, dtype=torch.float32),
            prediction_embedding=torch.tensor(full_emb, dtype=torch.float32),
            adj_matrix=torch.LongTensor([adj_combined.row.tolist(), adj_combined.col.tolist()]),
            checkpoint=model_params, model_layer=ml_combined,
            all_gat=True, device=device, VAE_model_select="GAT_VAE")
        if isinstance(pred_full, torch.Tensor):
            pred_full = pred_full.cpu().numpy()
        pred_expr = pred_full[n_train:]
        print(f"    gene_prediction OK: {pred_expr.shape}")
    except Exception as e:
        print(f"    gene_prediction failed ({e}), linear decode fallback")
        from stVGP.stVGP import GP_VAE_all
        model = GP_VAE_all(
            in_channels=ml_combined[0], hidden_channels=ml_combined[1],
            out_channels=ml_combined[2], num_heads=ml_combined[3],
            n_batch=0, in_channels_image=2048).to(device)
        model.load_state_dict(model_params)
        model.eval()
        W3 = model_params["gat3.att_src"] if "gat3.att_src" in model_params else model_params.get("gat3.weight")
        # Simple linear decode
        z = torch.tensor(embedding_pred, dtype=torch.float32)
        with torch.no_grad():
            # Try decode with kNN graph
            pred_nn2 = NearestNeighbors(n_neighbors=min(10, n_target - 1))
            pred_nn2.fit(target_xy)
            _, pi = pred_nn2.kneighbors(target_xy)
            pr, pc = [], []
            for j in range(n_target):
                for k in pi[j]:
                    if k != j:
                        pr.append(j); pc.append(k)
            pe = torch.LongTensor([pr, pc]).to(device)
            out = model.decode(z.to(device), pe)
            pred_expr = (out[0].cpu().numpy() if isinstance(out, tuple) else out.cpu().numpy())

    # Evaluate
    gt = holdout_adata.copy()
    sc.pp.normalize_total(gt, target_sum=1e4)
    sc.pp.log1p(gt)

    # All-gene metrics
    gt_hvg = gt[:, gene_names].X
    if sp.issparse(gt_hvg):
        gt_hvg = gt_hvg.toarray()
    gt_means = gt_hvg.mean(axis=0)
    pred_means = pred_expr.mean(axis=0)
    gene_mean_r = pearsonr(gt_means, pred_means)[0]
    gene_var_r = pearsonr(gt_hvg.var(axis=0), pred_expr.var(axis=0))[0]

    # MOBP gene (paper's primary metric)
    mobp_result = {}
    if "MOBP" in gt.var_names:
        mobp_idx_gt = list(gt.var_names).index("MOBP")
        gt_mobp = gt.X[:, mobp_idx_gt]
        if sp.issparse(gt_mobp):
            gt_mobp = gt_mobp.toarray().flatten()
        else:
            gt_mobp = np.asarray(gt_mobp).flatten()

        if "MOBP" in gene_names:
            mobp_idx_pred = gene_names.index("MOBP")
            pred_mobp = pred_expr[:, mobp_idx_pred]
            pcc = pearsonr(gt_mobp, pred_mobp)[0]
            rmse = np.sqrt(np.mean((gt_mobp - pred_mobp) ** 2))
            mobp_result = {"pcc": float(pcc), "rmse": float(rmse)}
            print(f"    MOBP PCC={pcc:.4f}, RMSE={rmse:.4f}")
        else:
            print(f"    MOBP not in HVG list")
    else:
        print(f"    MOBP not in dataset")

    return {
        "gene_mean_r": float(gene_mean_r),
        "gene_var_r": float(gene_var_r),
        "mobp": mobp_result,
        "n_target": n_target,
    }


def main():
    print("Loading DLPFC...")
    adata = ad.read_h5ad(str(DATA_PATH))
    print(f"  {adata.shape}, sections: {sorted(adata.obs['section'].unique())}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    all_sections = sorted(adata.obs["section"].unique())

    # Paper masks 151674 and 151675 (interior slices)
    results = {}
    for holdout in ["151674", "151675"]:
        results[holdout] = run_one_holdout(adata, holdout, all_sections, device)

    # Summary
    print("\n" + "=" * 70)
    print("stVGP DLPFC Reproduction — Paper Supp Fig 16/17")
    print("=" * 70)
    print(f"\n{'Slice':>10} {'gene_mean_r':>12} {'gene_var_r':>12} {'MOBP_PCC':>10} {'MOBP_RMSE':>10}")
    print("-" * 58)
    for sec, r in results.items():
        mobp_pcc = r["mobp"].get("pcc", float("nan"))
        mobp_rmse = r["mobp"].get("rmse", float("nan"))
        print(f"{sec:>10} {r['gene_mean_r']:>12.4f} {r['gene_var_r']:>12.4f} {mobp_pcc:>10.4f} {mobp_rmse:>10.4f}")

    # Paper values
    print("-" * 58)
    print(f"{'Paper':>10} {'':>12} {'':>12} {'0.682':>10} {'0.806':>10}")
    print(f"  (Supp Fig 17: PCC=0.682, RMSE=0.806, SSIM=0.553)")

    out_path = OUT_DIR / "stvgp_dlpfc_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
