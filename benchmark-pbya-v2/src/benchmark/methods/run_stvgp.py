"""stVGP method wrapper for virtual slice interpolation.

stVGP: Variational Spatial Gaussian Process for multi-slice ST.
Paper: Wang et al. (2026), Advanced Science.
Code: github.com/wzdrgi/stVGP

Pipeline:
  1. Preprocess: HVG selection, alignment (rigid mapping)
  2. Build adjacency + expression matrices
  3. Train GAT-VAE on all training sections
  4. GP interpolation of latent embeddings at target z-coordinates
  5. Decode interpolated embeddings to gene expression

Usage:
    conda run -n bench_stvgp python src/benchmark/methods/run_stvgp.py \
        --input data/processed/imc_breast_cancer/data.h5ad \
        --holdout-sections z7 \
        --output results/stvgp/imc_breast_cancer/loo_z7/prediction.h5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import scipy.sparse as sp
import torch


def check_environment():
    """Verify stVGP is importable."""
    try:
        from stVGP.stVGP import (st_preprocess, train_stVGP,
                                  get_3D_prediction, gene_prediction,
                                  adata_preprocess_adjnet)
        print(f"stVGP imported, CUDA: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return False


def prepare_input(adata, holdout_sections):
    """Remove holdout sections, return training/holdout data and targets."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)

    train_adata = adata[~holdout_mask].copy()
    holdout_adata = adata[holdout_mask].copy()

    target_z = {}
    for sec in holdout_sections:
        sec_mask = sections == sec
        target_z[sec] = float(np.median(adata.obsm["spatial"][sec_mask, 2]))

    return train_adata, holdout_adata, target_z


def run_method(train_adata, holdout_adata, target_z, seed=42,
               training_epoch=1500, hidden_embedding=None, n_hvg=5000,
               n_neighbors=10, all_gat=True, Rbf_value=512,
               device="cuda:0"):
    """Execute stVGP: preprocess → train VAE → GP interpolate → decode."""
    from stVGP.stVGP import (train_stVGP, get_3D_prediction, gene_prediction,
                              get_need_ST_reconstruction, adata_preprocess_adjnet)
    import scanpy as sc
    from sklearn.neighbors import NearestNeighbors

    # Skip datasets with >200K cells (stVGP OOMs on large datasets)
    n_total_cells = train_adata.n_obs + holdout_adata.n_obs
    if n_total_cells > 200_000:
        print(f"  SKIP: {n_total_cells} cells > 200K limit for stVGP")
        return {}

    np.random.seed(seed)
    torch.manual_seed(seed)
    if hidden_embedding is None:
        hidden_embedding = [512, 24]  # DLPFC defaults (paper's most representative)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Split training data into per-section AnnData objects
    train_sections = train_adata.obs["section"].values.astype(str)
    unique_sections = sorted(np.unique(train_sections),
                             key=lambda s: np.median(
                                 train_adata.obsm["spatial"][train_sections == s, 2]))

    section_adatas = []
    section_z_values = []
    for sec in unique_sections:
        mask = train_sections == sec
        sec_adata = train_adata[mask].copy()
        # Store 2D spatial for adjacency
        sec_adata.obsm["spatial_2d"] = sec_adata.obsm["spatial"][:, :2].copy()
        section_adatas.append(sec_adata)
        section_z_values.append(float(np.median(sec_adata.obsm["spatial"][:, 2])))

    # Normalize expression (paper applies normalize_total + log1p before training)
    expr_type = train_adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        sc.pp.normalize_total(train_adata, target_sum=1e4)
        sc.pp.log1p(train_adata)
        for a in section_adatas:
            sc.pp.normalize_total(a, target_sum=1e4)
            sc.pp.log1p(a)
    elif expr_type in ("log1p_normalized", "log2_normalized"):
        pass  # already log-transformed
    elif expr_type in ("normalized", "fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(train_adata)
        for a in section_adatas:
            sc.pp.log1p(a)

    # Select HVGs across all sections (paper uses 5000-6000)
    n_genes = train_adata.n_vars
    if n_genes > n_hvg:
        sc.pp.highly_variable_genes(train_adata, n_top_genes=n_hvg, flavor="seurat")
        hvg_mask = train_adata.var["highly_variable"].values
        gene_names = train_adata.var_names[hvg_mask].tolist()
        section_adatas = [a[:, hvg_mask].copy() for a in section_adatas]
        print(f"  Selected {len(gene_names)} HVGs")
    else:
        gene_names = train_adata.var_names.tolist()

    # Alignment step (paper requires gene_rigid_mapping_alignment before training)
    # Use top HVGs as alignment genes (paper uses Moran's I genes from R, but HVGs are a proxy)
    align_genes = gene_names[:min(10, len(gene_names))]  # top 10 genes for alignment
    for a in section_adatas:
        a.obsm["spatial"] = a.obsm["spatial_2d"].copy()  # alignment uses 'spatial' key

    try:
        from stVGP.stVGP import gene_rigid_mapping_alignment
        section_adatas = gene_rigid_mapping_alignment(
            gene_input=align_genes,
            stadata_input=section_adatas,
            ini_spatial="spatial",
            add_spatial="align_spatial",
            align_model="single_template_alignment",
            ref_label=0,
            angle_params=[-60, -40, -20, 0, 20, 40, 60],
        )
        spatial_key = "align_spatial"
        print(f"  Aligned {len(section_adatas)} sections")
    except Exception as e:
        print(f"  WARNING: alignment failed ({e}), using raw coords")
        spatial_key = "spatial"

    # Build expression matrix and adjacency using adata_preprocess_adjnet
    # Cross-slice edges are critical for GAT but scale O(n_sections * n_cells),
    # causing OOM for datasets with many sections. Disable for >20 sections.
    use_cross = len(section_adatas) <= 20
    try:
        slice_matrix, adj_matrix_sp = adata_preprocess_adjnet(
            input_adata=section_adatas,
            align_model="single_template_alignment",
            ref_label=0,
            spatial_label=spatial_key,
            n_neighbors=n_neighbors,
            no_cross=not use_cross,
        )
        # adj_matrix_sp is scipy COO sparse
        adj_coo = adj_matrix_sp.tocoo()
        print(f"  Adjacency: {adj_coo.nnz} edges (incl. cross-slice)")
    except Exception as e:
        print(f"  WARNING: adata_preprocess_adjnet failed ({e}), falling back to manual")
        # Fallback: manual adjacency (intra-section only)
        n_neighbors_fb = n_neighbors
        all_X = []
        edge_rows, edge_cols = [], []
        offset = 0
        for i, sec_adata in enumerate(section_adatas):
            X = sec_adata.X
            if sp.issparse(X):
                X = X.toarray()
            all_X.append(X.astype(np.float32))
            coords_2d = sec_adata.obsm.get(spatial_key, sec_adata.obsm["spatial_2d"])
            nn = NearestNeighbors(n_neighbors=min(n_neighbors_fb, len(sec_adata) - 1))
            nn.fit(coords_2d)
            _, indices = nn.kneighbors(coords_2d)
            for j in range(len(sec_adata)):
                for k_idx in indices[j]:
                    if k_idx != j:
                        edge_rows.append(offset + j)
                        edge_cols.append(offset + k_idx)
            offset += len(sec_adata)
        slice_matrix = np.vstack(all_X)
        n_total = slice_matrix.shape[0]
        adj_coo = sp.coo_matrix(
            (np.ones(len(edge_rows)), (edge_rows, edge_cols)),
            shape=(n_total, n_total))

    if not isinstance(slice_matrix, np.ndarray):
        slice_matrix = np.array(slice_matrix, dtype=np.float32)

    # Build 3D coords for GP prediction
    all_coords_3d = []
    for i, sec_adata in enumerate(section_adatas):
        coords_2d = sec_adata.obsm.get(spatial_key, sec_adata.obsm["spatial_2d"])
        if coords_2d.shape[1] > 2:
            coords_2d = coords_2d[:, :2]
        z_col = np.full((len(sec_adata), 1), section_z_values[i])
        all_coords_3d.append(np.hstack([coords_2d, z_col]))
    spatial_coords = np.vstack(all_coords_3d)

    n_total = slice_matrix.shape[0]
    print(f"  Training stVGP: {n_total} cells, {slice_matrix.shape[1]} genes, "
          f"{len(unique_sections)} sections, {adj_coo.nnz} edges")

    # Train
    n_genes_used = slice_matrix.shape[1]
    model_layer = [n_genes_used, hidden_embedding[0], hidden_embedding[1], 1]

    train_result = train_stVGP(
        ST_need_reconstruction_matrix=slice_matrix,
        all_spatial_net=adj_coo,
        lr=0.001,
        weight_decay=0.0001,
        training_epoch=training_epoch,
        num_heads=1,
        device=device,
        hidden_embedding=hidden_embedding,
        random_seed=seed,
        all_gat=all_gat,
        VAE_model_select="GAT_VAE",
    )

    # train_stVGP returns (recon_x, embedding, model_params, logvar) — 4 values
    recon_x, embedding, model_params = train_result[0], train_result[1], train_result[2]

    # For each held-out section: GP interpolation + decode
    results = {}
    holdout_sections_arr = holdout_adata.obs["section"].values.astype(str)

    for target_sec, tz in sorted(target_z.items(), key=lambda kv: kv[1]):
        sec_mask = holdout_sections_arr == target_sec
        target_xy = holdout_adata.obsm["spatial"][sec_mask, :2]
        n_target = target_xy.shape[0]

        # Build target 3D coordinates
        target_3d = np.hstack([target_xy, np.full((n_target, 1), tz)])

        print(f"  {target_sec}: GP interpolation for {n_target} cells...")

        try:
            # Subsample training points for GP (full kernel matrix is O(n^2))
            max_gp_train = 5000
            if len(spatial_coords) > max_gp_train:
                gp_idx = np.random.choice(len(spatial_coords), max_gp_train, replace=False)
                gp_coords = spatial_coords[gp_idx]
                gp_embedding = embedding[gp_idx]
            else:
                gp_coords = spatial_coords
                gp_embedding = embedding

            # GP interpolation of embeddings
            embedding_pred = get_3D_prediction(
                train_coordinates=gp_coords,
                embedding=gp_embedding,
                spatial_pred=target_3d,
                noise=False,
                constant_value=1.0,
                Rbf_value=Rbf_value,
            )

            # Decode using gene_prediction (proper reparametrize + training adj)
            slice_matrix_t = torch.tensor(slice_matrix, dtype=torch.float32)
            prediction_embedding_t = torch.tensor(embedding_pred, dtype=torch.float32)
            edge_list = [adj_coo.row.tolist(), adj_coo.col.tolist()]
            adj_tensor = torch.LongTensor(edge_list)

            try:
                pred_expr = gene_prediction(
                    slice_matrix=slice_matrix_t,
                    prediction_embedding=prediction_embedding_t,
                    adj_matrix=adj_tensor,
                    checkpoint=model_params,
                    model_layer=model_layer,
                    all_gat=all_gat,
                    device=device,
                    VAE_model_select="GAT_VAE",
                )
                if isinstance(pred_expr, torch.Tensor):
                    pred_expr = pred_expr.cpu().numpy()
            except RuntimeError as decode_err:
                # gene_prediction fails due to size mismatch between training
                # logvar (n_train) and prediction embedding (n_pred).
                # Fallback: direct decode with kNN graph on prediction cells.
                print(f"      gene_prediction failed ({decode_err}), using kNN-decode fallback")
                from stVGP.stVGP import GP_VAE_all
                model = GP_VAE_all(
                    in_channels=model_layer[0],
                    hidden_channels=model_layer[1],
                    out_channels=model_layer[2],
                    num_heads=model_layer[3],
                    n_batch=0,
                    in_channels_image=2048,
                ).to(device)
                model.load_state_dict(model_params)
                model.eval()

                pred_nn = NearestNeighbors(n_neighbors=min(10, n_target - 1))
                pred_nn.fit(target_xy)
                _, pred_indices = pred_nn.kneighbors(target_xy)
                pred_rows, pred_cols = [], []
                for j in range(n_target):
                    for k_idx in pred_indices[j]:
                        if k_idx != j:
                            pred_rows.append(j)
                            pred_cols.append(k_idx)
                pred_edge = torch.LongTensor([pred_rows, pred_cols]).to(device)

                z = torch.tensor(embedding_pred, dtype=torch.float32).to(device)
                with torch.no_grad():
                    decode_out = model.decode(z, pred_edge)
                    pred_expr = decode_out[0].cpu().numpy() if isinstance(decode_out, tuple) else decode_out.cpu().numpy()

            coords = np.column_stack([target_xy, np.full(n_target, tz)])

            results[target_sec] = {
                "X": sp.csr_matrix(pred_expr.astype(np.float32)),
                "coords": coords,
                "cell_type": np.array(["NA"] * n_target),
                "gene_names": gene_names,
            }
            print(f"    -> {n_target} cells predicted")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def format_output(results, gene_names_full, holdout_sections, method_params,
                  wall_time, output_path):
    """Write prediction.h5."""
    if not results:
        print("No results to write!")
        return

    first = next(iter(results.values()))
    gene_names = first.get("gene_names", gene_names_full)

    all_X, all_ids, all_x, all_y, all_z = [], [], [], [], []
    all_section, all_ct = [], []
    cell_counter = 0

    for sec in holdout_sections:
        if sec not in results:
            continue
        r = results[sec]
        n = r["X"].shape[0]
        all_X.append(r["X"])
        all_ids.extend([f"pred_{cell_counter + i}" for i in range(n)])
        all_x.append(r["coords"][:, 0])
        all_y.append(r["coords"][:, 1])
        all_z.append(r["coords"][:, 2])
        all_section.extend([sec] * n)
        all_ct.extend(r["cell_type"].tolist())
        cell_counter += n

    if cell_counter == 0:
        return

    X = sp.vstack(all_X, format="csr")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        g = f.create_group("X")
        g.create_dataset("data", data=X.data)
        g.create_dataset("indices", data=X.indices)
        g.create_dataset("indptr", data=X.indptr)
        g.create_dataset("shape", data=np.array(X.shape))

        obs = f.create_group("obs")
        obs.create_dataset("cell_id", data=np.array(all_ids, dtype="S"))
        obs.create_dataset("x", data=np.concatenate(all_x))
        obs.create_dataset("y", data=np.concatenate(all_y))
        obs.create_dataset("z", data=np.concatenate(all_z))
        obs.create_dataset("section", data=np.array(all_section, dtype="S"))
        obs.create_dataset("cell_type", data=np.array(all_ct, dtype="S"))

        var = f.create_group("var")
        var.create_dataset("gene_name", data=np.array(gene_names, dtype="S"))

        uns = f.create_group("uns")
        uns.create_dataset("method_name", data="stvgp")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="stVGP virtual slice generation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-epoch", type=int, default=1500)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    train_adata, holdout_adata, target_z = prepare_input(adata, args.holdout_sections)
    del adata

    print(f"Running stVGP...")
    t0 = time.time()
    results = run_method(train_adata, holdout_adata, target_z, seed=args.seed,
                         training_epoch=args.training_epoch, device=args.device)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "training_epoch": args.training_epoch,
        "hidden_embedding": [512, 24], "n_hvg": 5000, "all_gat": True,
        "Rbf_value": 512, "n_neighbors": 10,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
