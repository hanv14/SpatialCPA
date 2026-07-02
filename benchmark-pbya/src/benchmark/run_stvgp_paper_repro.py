"""Reproduce stVGP on ADMB (st_mouse_brain_ortiz) with paper's exact parameters.

Paper: Wang et al. 2026, Advanced Science (github.com/wzdrgi/stVGP)
Parameters from Tutorial/example_admb.ipynb:
  hidden_embedding = [128, 28], training_epoch = 600
  n_hvg = 8000, n_neighbors = 15, all_gat = False
  random_seed = 502, align_model = sequential_alignment
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

OUT_DIR = Path("results/stvgp/st_mouse_brain_ortiz")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    holdout = "22A"

    print("Loading st_mouse_brain_ortiz...")
    adata = ad.read_h5ad("data/processed/st_mouse_brain_ortiz/data.h5ad")
    print(f"  {adata.shape}")

    sections = adata.obs["section"].values.astype(str)
    holdout_mask = sections == holdout
    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()
    target_z = float(np.median(adata.obsm["spatial"][holdout_mask, 2]))
    print(f"  Holdout {holdout}: {holdout_mask.sum()} cells, z={target_z:.0f}")

    # Preprocess
    sc.pp.normalize_total(train_adata, target_sum=1e4)
    sc.pp.log1p(train_adata)

    # Per-section AnnData
    train_sections = train_adata.obs["section"].values.astype(str)
    unique_secs = sorted(np.unique(train_sections),
                         key=lambda s: np.median(train_adata.obsm["spatial"][train_sections == s, 2]))

    section_adatas = []
    section_z_values = []
    for sec in unique_secs:
        mask = train_sections == sec
        sec_adata = train_adata[mask].copy()
        sec_adata.obsm["spatial_2d"] = sec_adata.obsm["spatial"][:, :2].copy()
        section_adatas.append(sec_adata)
        section_z_values.append(float(np.median(sec_adata.obsm["spatial"][:, 2])))

    # HVG: n_hvg=8000 (paper)
    sc.pp.highly_variable_genes(train_adata, n_top_genes=8000, flavor="seurat")
    hvg_mask = train_adata.var["highly_variable"].values
    gene_names = train_adata.var_names[hvg_mask].tolist()
    section_adatas = [a[:, hvg_mask].copy() for a in section_adatas]
    print(f"  Selected {len(gene_names)} HVGs")

    # Alignment
    align_genes = gene_names[:10]
    for a in section_adatas:
        a.obsm["spatial"] = a.obsm["spatial_2d"].copy()

    try:
        section_adatas = gene_rigid_mapping_alignment(
            gene_input=align_genes,
            stadata_input=section_adatas,
            ini_spatial="spatial",
            add_spatial="align_spatial",
            align_model="sequential_alignment",
            ref_label=0,
            angle_params=[-60, -40, -20, 0, 20, 40, 60])
        spatial_key = "align_spatial"
        print(f"  Aligned {len(section_adatas)} sections")
    except Exception as e:
        print(f"  Alignment failed ({e}), using raw coords")
        spatial_key = "spatial"

    # Adjacency: n_neighbors=15, no_cross=False (paper)
    n_total = sum(a.n_obs for a in section_adatas)
    print(f"  Total training cells: {n_total}")

    # For 74 sections with no_cross=False and ~34K cells, this may be large
    # Paper uses 35 slices. If too many edges, fall back to intra-section only.
    try:
        slice_matrix, adj_matrix_sp = adata_preprocess_adjnet(
            input_adata=section_adatas,
            align_model="sequential_alignment",
            ref_label=0,
            spatial_label=spatial_key,
            n_neighbors=15,
            no_cross=False)
        adj_coo = adj_matrix_sp.tocoo()
        print(f"  Adjacency: {adj_coo.nnz} edges")
    except Exception as e:
        print(f"  adata_preprocess_adjnet failed ({e}), manual fallback (intra-section only)")
        all_X = []
        edge_rows, edge_cols = [], []
        offset = 0
        for sec_adata in section_adatas:
            X = sec_adata.X
            if sp.issparse(X):
                X = X.toarray()
            all_X.append(X.astype(np.float32))
            coords_2d = sec_adata.obsm.get(spatial_key, sec_adata.obsm["spatial_2d"])
            nn = NearestNeighbors(n_neighbors=min(15, len(sec_adata) - 1))
            nn.fit(coords_2d)
            _, indices = nn.kneighbors(coords_2d)
            for j in range(len(sec_adata)):
                for k_idx in indices[j]:
                    if k_idx != j:
                        edge_rows.append(offset + j)
                        edge_cols.append(offset + k_idx)
            offset += len(sec_adata)
        slice_matrix = np.vstack(all_X)
        adj_coo = sp.coo_matrix(
            (np.ones(len(edge_rows)), (edge_rows, edge_cols)),
            shape=(n_total, n_total))

    if not isinstance(slice_matrix, np.ndarray):
        slice_matrix = np.array(slice_matrix, dtype=np.float32)

    # 3D coords
    all_coords_3d = []
    for i, sec_adata in enumerate(section_adatas):
        coords_2d = sec_adata.obsm.get(spatial_key, sec_adata.obsm["spatial_2d"])
        if coords_2d.shape[1] > 2:
            coords_2d = coords_2d[:, :2]
        z_col = np.full((len(sec_adata), 1), section_z_values[i])
        all_coords_3d.append(np.hstack([coords_2d, z_col]))
    spatial_coords = np.vstack(all_coords_3d)

    n_genes_used = slice_matrix.shape[1]
    # Paper params: hidden=[128, 28], epochs=600, all_gat=False
    hidden_embedding = [128, 28]
    model_layer = [n_genes_used, 128, 28, 1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"  Training stVGP: {n_total} cells, {n_genes_used} genes, "
          f"hidden={hidden_embedding}, epochs=600, all_gat=False")

    t0 = time.time()
    train_result = train_stVGP(
        ST_need_reconstruction_matrix=slice_matrix,
        all_spatial_net=adj_coo,
        lr=0.001,
        weight_decay=0.0001,
        training_epoch=600,
        num_heads=1,
        device=device,
        hidden_embedding=hidden_embedding,
        random_seed=502,
        all_gat=False,
        VAE_model_select="GAT_VAE")

    recon_x, embedding, model_params = train_result[0], train_result[1], train_result[2]
    print(f"  Trained in {time.time()-t0:.0f}s")

    # GP prediction
    target_xy = holdout_adata.obsm["spatial"][:, :2]
    n_target = target_xy.shape[0]
    target_3d = np.hstack([target_xy, np.full((n_target, 1), target_z)])

    max_gp_train = 5000
    if len(spatial_coords) > max_gp_train:
        gp_idx = np.random.choice(len(spatial_coords), max_gp_train, replace=False)
        gp_coords = spatial_coords[gp_idx]
        gp_embedding = embedding[gp_idx]
    else:
        gp_coords = spatial_coords
        gp_embedding = embedding

    embedding_pred = get_3D_prediction(
        train_coordinates=gp_coords,
        embedding=gp_embedding,
        spatial_pred=target_3d,
        noise=False,
        constant_value=1.0,
        Rbf_value=512)

    # Decode using paper's protocol:
    # gene_prediction expects FULL-sized embedding (n_train), not just prediction cells.
    # Paper replaces the held-out slice's embedding with GP-predicted embedding,
    # keeping all other slices' original training embeddings.
    # We identify which cells belong to the holdout section in the training index space
    # and substitute their embeddings with GP-predicted ones.

    # Find the holdout section's position in the sorted training sections
    holdout_sec_z = target_z
    # Insert predicted embedding into full embedding at the right position
    # Since holdout was removed from training, we append predictions and adjust adj
    full_embedding = np.vstack([embedding, embedding_pred])
    n_train_orig = embedding.shape[0]

    # Build combined adjacency (training + prediction cells)
    # Add kNN edges for prediction cells to nearest training cells
    all_coords = np.vstack([spatial_coords, target_3d])
    pred_nn = NearestNeighbors(n_neighbors=min(15, n_train_orig - 1))
    pred_nn.fit(spatial_coords[:, :2])  # 2D spatial
    _, pred_indices = pred_nn.kneighbors(target_xy)
    new_rows, new_cols = list(adj_coo.row), list(adj_coo.col)
    for j in range(n_target):
        for k_idx in pred_indices[j]:
            new_rows.append(n_train_orig + j)
            new_cols.append(int(k_idx))
            new_rows.append(int(k_idx))
            new_cols.append(n_train_orig + j)
    n_total_combined = n_train_orig + n_target
    adj_combined = sp.coo_matrix(
        (np.ones(len(new_rows)), (new_rows, new_cols)),
        shape=(n_total_combined, n_total_combined))

    # Expand slice_matrix with zeros for prediction cells
    slice_matrix_combined = np.vstack([
        slice_matrix,
        np.zeros((n_target, slice_matrix.shape[1]), dtype=np.float32)])

    slice_matrix_t = torch.tensor(slice_matrix_combined, dtype=torch.float32)
    prediction_embedding_t = torch.tensor(full_embedding, dtype=torch.float32)
    edge_list = [adj_combined.row.tolist(), adj_combined.col.tolist()]
    adj_tensor = torch.LongTensor(edge_list)
    model_layer_combined = [slice_matrix_combined.shape[1], 128, 28, 1]

    try:
        pred_expr_full = gene_prediction(
            slice_matrix=slice_matrix_t,
            prediction_embedding=prediction_embedding_t,
            adj_matrix=adj_tensor,
            checkpoint=model_params,
            model_layer=model_layer_combined,
            all_gat=False,
            device=device,
            VAE_model_select="GAT_VAE")
        if isinstance(pred_expr_full, torch.Tensor):
            pred_expr_full = pred_expr_full.cpu().numpy()
        # Extract only the prediction cells (last n_target rows)
        pred_expr = pred_expr_full[n_train_orig:]
        print(f"  gene_prediction succeeded: {pred_expr.shape}")
    except RuntimeError as e:
        print(f"  gene_prediction failed ({e}), using linear decode fallback")
        # Fallback: manual decode through saved weights
        W3 = model_params.get("gat3.weight")
        b3 = model_params.get("gat3.bias", None)
        W4 = model_params.get("gat4.weight")
        b4 = model_params.get("gat4.bias", None)
        z = torch.tensor(embedding_pred, dtype=torch.float32)
        h = z @ W3.T
        if b3 is not None:
            h = h + b3
        h = torch.relu(h)
        out = h @ W4.T
        if b4 is not None:
            out = out + b4
        pred_expr = out.numpy()

    # Evaluate
    gt_adata = holdout_adata.copy()
    sc.pp.normalize_total(gt_adata, target_sum=1e4)
    sc.pp.log1p(gt_adata)
    gt_X = gt_adata[:, gene_names].X
    if sp.issparse(gt_X):
        gt_X = gt_X.toarray()

    gt_means = gt_X.mean(axis=0)
    pred_means = pred_expr.mean(axis=0)
    gmr = pearsonr(gt_means, pred_means)[0]
    gt_vars = gt_X.var(axis=0)
    pred_vars = pred_expr.var(axis=0)
    gvr = pearsonr(gt_vars, pred_vars)[0]

    print(f"\nResults (paper params: hidden=[128,28], epochs=600, n_hvg=8000):")
    print(f"  gene_mean_pearson: {gmr:.4f}")
    print(f"  gene_var_pearson:  {gvr:.4f}")
    print(f"  n_predicted_cells: {n_target}")
    print(f"  Previous (wrong params): gene_mean_pearson=0.818")

    out_path = OUT_DIR / "stvgp_paper_params.json"
    with open(out_path, "w") as f:
        json.dump({
            "gene_mean_r": float(gmr),
            "gene_var_r": float(gvr),
            "n_target": n_target,
            "params": {
                "hidden_embedding": [128, 28],
                "training_epoch": 600,
                "n_hvg": 8000,
                "n_neighbors": 15,
                "all_gat": False,
                "random_seed": 502,
            },
        }, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
