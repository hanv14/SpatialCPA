"""isoST method wrapper for virtual slice interpolation.

isoST: SDE-based isotropic 3D reconstruction of gene expression.
Paper: Li et al. (2025), bioRxiv.
Code: github.com/deng-ai-lab/isoST

Pipeline:
  1. Preprocess h5ad → PCA → normalized .pt files per section
  2. Train isoST model (SDE-based) on training sections
  3. Run inference to reconstruct at held-out z-positions
  4. Inverse PCA to recover gene expression
  5. Format output as prediction.h5

Usage:
    python src/benchmark/methods/run_isost.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_3 \
        --output results/isost/cosmx_nsclc_3d/loo_section_3/prediction.h5
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import anndata as ad
import h5py
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# v2 shared I/O + guards (sibling module).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _v2_io  # noqa: E402

# v2 shares the downloaded isoST tool with v1.
TOOLS_DIR = (Path(__file__).resolve().parents[4] / "benchmark-pbya"
             / "tools" / "isost")
N_PCS = 50


def check_environment():
    """Verify isoST and its dependencies are available."""
    if not TOOLS_DIR.exists():
        print(f"ERROR: isoST not found at {TOOLS_DIR}", file=sys.stderr)
        print("Clone: git clone https://github.com/deng-ai-lab/isoST tools/isost",
              file=sys.stderr)
        return False

    try:
        import torch
        print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("WARNING: isoST strongly prefers GPU. CPU mode may be very slow.")
    except ImportError:
        print("ERROR: PyTorch not installed", file=sys.stderr)
        return False

    # Check isoST modules
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        from utils.train_ode import biaxial_train
        from utils.inference import fine_inference
        print("isoST train/inference modules available")
        return True
    except ImportError as e:
        print(f"ERROR importing isoST: {e}", file=sys.stderr)
        return False


def prepare_input(adata, holdout_sections):
    """Remove holdout sections, return training data and targets."""
    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    train_adata = adata[~holdout_mask].copy()

    target_z = {}
    for sec in holdout_sections:
        sec_mask = sections == sec
        target_z[sec] = float(np.median(adata.obsm["spatial"][sec_mask, 2]))

    return train_adata, target_z


def _preprocess_to_pt(adata, work_dir, n_pcs=N_PCS):
    """Convert AnnData to isoST's expected .pt format.

    Creates:
      - work_dir/data/{section}_log_PC.pt  (N × (3 + n_pcs))
      - work_dir/min_dic.csv, scale_dic.csv
      - work_dir/zscore_pc_model.pkl
      - work_dir/gene.csv

    Returns:
      - slide_names: list of section names (sorted by z)
      - pca_model: fitted PCA for inverse transform
      - scaler: fitted StandardScaler
      - norm_params: dict with normalization parameters for de-normalization
    """
    import torch

    sections = adata.obs["section"].values.astype(str)
    unique_sections = np.unique(sections)

    # Sort by z
    section_z = {}
    for sec in unique_sections:
        mask = sections == sec
        section_z[sec] = float(np.median(adata.obsm["spatial"][mask, 2]))
    sorted_sections = sorted(unique_sections, key=lambda s: section_z[s])

    # Get dense expression matrix
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Log2 transform (matching isoST authors: file suffix _log_PC, output log2_expr)
    expr_type = adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        X = np.log2(X + 1)
    elif expr_type in ("log1p_normalized",):
        pass  # already log-transformed
    elif expr_type in ("log2_normalized",):
        pass  # already log2
    elif expr_type in ("normalized", "fluorescence_intensity", "mean_intensity"):
        X = np.log2(X + 1)
    # else: use as-is

    # Z-score normalization + PCA (matching authors: zscore_PC50_minmax pipeline)
    actual_pcs = min(n_pcs, min(X.shape) - 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=actual_pcs)
    X_pcs = pca.fit_transform(X_scaled)

    # Min-max normalize PCs
    pc_min = X_pcs.min(axis=0)
    pc_range = X_pcs.max(axis=0) - pc_min
    pc_range[pc_range == 0] = 1.0
    X_pcs_norm = (X_pcs - pc_min) / pc_range

    # Normalize spatial coordinates
    coords = adata.obsm["spatial"]  # (N, 3)
    x_all, y_all, z_all = coords[:, 0], coords[:, 1], coords[:, 2]

    min_x, min_y = x_all.min(), y_all.min()
    width_x = x_all.max() - min_x
    width_y = y_all.max() - min_y
    max_width = max(width_x, width_y)
    if max_width == 0:
        max_width = 1.0

    # z normalization: map to [0, 1]
    min_z = z_all.min()
    z_range = z_all.max() - min_z
    if z_range == 0:
        z_range = 1.0

    # Save .pt files per section (both original and shuffled for training)
    data_dir = Path(work_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    slide_names = []
    for sec in sorted_sections:
        mask = sections == sec
        idx = np.where(mask)[0]

        x_norm = (coords[idx, 0] - min_x) / max_width
        y_norm = (coords[idx, 1] - min_y) / max_width
        z_norm = (coords[idx, 2] - min_z) / z_range

        features = X_pcs_norm[idx]
        # (N, 3 + n_pcs): x, y, z, PC1..PCn
        tensor_data = np.column_stack([x_norm, y_norm, z_norm, features])
        tensor = torch.tensor(tensor_data, dtype=torch.float32)

        fname = f"{sec}_log_PC"
        torch.save(tensor, data_dir / f"{fname}.pt")
        # isoST training and inference both load "shuffled_{name}.pt"
        perm = torch.randperm(tensor.shape[0])
        torch.save(tensor[perm], data_dir / f"shuffled_{fname}.pt")
        slide_names.append(fname)

    # Save metadata
    min_dic = {"x": [min_x], "y": [min_y], "z": [min_z]}
    scale_dic = {"xy": [max_width], "z": [z_range]}
    for i in range(actual_pcs):
        min_dic[f"PC_{i+1}"] = [pc_min[i]]
        scale_dic[f"PC_{i+1}"] = [pc_range[i]]

    pd.DataFrame(min_dic).to_csv(Path(work_dir) / "min_dic.csv", index=False)
    pd.DataFrame(scale_dic).to_csv(Path(work_dir) / "scale_dic.csv", index=False)

    # Save PCA model (scaler + PCA combined)
    model_info = {"scaler": scaler, "pca": pca}
    joblib.dump(model_info, Path(work_dir) / "zscore_pc_model.pkl")

    # Save gene list
    gene_df = pd.DataFrame({"gene_symbol": adata.var_names.tolist()})
    gene_df.to_csv(Path(work_dir) / "gene.csv")

    norm_params = {
        "min_x": float(min_x), "min_y": float(min_y), "min_z": float(min_z),
        "max_width": float(max_width), "z_range": float(z_range),
        "pc_min": pc_min.tolist(), "pc_range": pc_range.tolist(),
        "actual_pcs": actual_pcs,
    }

    return slide_names, sorted_sections, norm_params


def _write_config(work_dir, n_pcs):
    """Write isoST config.yml."""
    import yaml
    config = {
        "trainer": "IsoST",
        "params": {
            "gene_dim": n_pcs,
            "hidden_dim": 64,
            "head_num": 1,
            "K": 8,  # paper uses K=8 for kNN graph (default 5)
            "lr": 0.001,
            "optimizer_name": "NAdam",
            "weight_decay": 1e-8,
            "method": "euler",
            "delta_d": 0.01,
            "stride": 1,
            "std_x": 0.01,
            "std_y": 0.01,
            "std_z": 0.1,
            "std_seq": 0.1,
            "alpha": 0.1,
            "dual": True,
            "beta_start_value": 1,
            "beta_end_value": 0.05,
            "beta_start_iteration": 50,
            "beta_n_iterations": 50,
            "warm_up_rate": 1,
        },
    }
    config_path = Path(work_dir) / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(config_path)


def run_method(train_adata, target_z, seed=42, epochs=None, device="cuda:0", batch_num=5):
    """Execute isoST: preprocess → train → ODE inference → inverse PCA.

    Uses IsoST.fine_infer() which integrates the ODE from each observed section
    forward/backward, producing interpolated points at fine z-steps.
    Then extracts points at target z-coordinates.

    Returns dict mapping section_label -> {X, coords, cell_type}.
    """
    import torch
    import yaml

    sys.path.insert(0, str(TOOLS_DIR))
    from utils.train_ode import biaxial_train
    import model as training_module

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if epochs is None:
        epochs = [100, 100, 100]  # matches authors' mouse_brain config

    work_dir = tempfile.mkdtemp(prefix="isost_bench_")
    print(f"  Working directory: {work_dir}")

    # Step 1: Preprocess
    print("  Preprocessing to .pt format...")
    actual_pcs = min(N_PCS, min(train_adata.X.shape[0], train_adata.n_vars) - 1)
    slide_names, sorted_sections, norm_params = _preprocess_to_pt(
        train_adata, work_dir, n_pcs=actual_pcs)

    # Step 2: Write config
    config_path = _write_config(work_dir, actual_pcs)

    # Step 3: Train
    data_dir = os.path.join(work_dir, "data")
    experiment_dir = os.path.join(work_dir, "experiments")
    result_dir = os.path.join(work_dir, "result")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    print(f"  Training isoST ({len(slide_names)} sections, {actual_pcs} PCs)...")
    biaxial_train(
        experiment_dir=experiment_dir,
        data_dir=data_dir,
        slide_names=slide_names,
        batch_num=batch_num,
        config_file=config_path,
        device=device,
        checkpoint_every=100,
        backup_every=50,
        epoch=epochs,
        mode="joint",
    )

    # Step 4: Load model and run fine_infer (ODE-based, no BiG needed)
    print("  Loading trained model...")
    with open(os.path.join(experiment_dir, "config.yml")) as f:
        config = yaml.safe_load(f)

    TrainerClass = getattr(training_module, config["trainer"])
    trainer = TrainerClass(device=device, **config["params"])
    trainer.load(os.path.join(experiment_dir, "model.pt"))
    trainer.to(device)

    dd = config["params"]["delta_d"]
    print(f"  Running ODE inference (delta_d={dd})...")
    trainer.fine_infer(data_dir, slide_names, "joint", dd, result_dir, 1, device)

    # Step 5: Parse .npy output files, find points at target z-coordinates
    print("  Extracting predictions at target z-coordinates...")
    results = _extract_from_npy(result_dir, target_z, norm_params, actual_pcs,
                                os.path.join(work_dir, "zscore_pc_model.pkl"))

    # Cleanup
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass

    return results


def _extract_from_npy(result_dir, target_z, norm_params, actual_pcs, pca_model_path):
    """Extract predicted cells at target z from isoST .npy output files."""
    result_path = Path(result_dir)
    npy_files = sorted(result_path.glob("*_forward.npy"),
                       key=lambda f: int(f.stem.split("_")[0]))

    if not npy_files:
        print("  WARNING: no .npy result files found")
        return {}

    # Load all interpolated points
    all_data = np.vstack([np.load(f) for f in npy_files])
    z_norm_all = all_data[:, 2]

    min_z = norm_params["min_z"]
    z_range = norm_params["z_range"]
    max_width = norm_params["max_width"]
    min_x = norm_params["min_x"]
    min_y = norm_params["min_y"]
    pc_min = np.array(norm_params["pc_min"])
    pc_range = np.array(norm_params["pc_range"])

    model_info = joblib.load(pca_model_path)
    pca = model_info["pca"]
    scaler = model_info["scaler"]

    results = {}
    for target_sec, tz in target_z.items():
        tz_norm = (tz - min_z) / z_range

        # Find points closest to this z (within half a delta_d step)
        z_dist = np.abs(z_norm_all - tz_norm)
        # Use a small threshold based on spacing between unique z-values
        unique_z = np.unique(z_norm_all.round(4))
        if len(unique_z) > 1:
            min_gap = np.min(np.diff(np.sort(unique_z)))
            threshold = min_gap * 0.6
        else:
            threshold = 0.01
        mask = z_dist < threshold

        if mask.sum() == 0:
            # Fallback: take the closest z-level
            closest_z = unique_z[np.argmin(np.abs(unique_z - tz_norm))]
            mask = np.abs(z_norm_all - closest_z) < 1e-4

        if mask.sum() == 0:
            print(f"    WARNING: no points found near z={tz} (norm={tz_norm:.3f})")
            continue

        selected = all_data[mask]

        # De-normalize coordinates
        x_um = selected[:, 0] * max_width + min_x
        y_um = selected[:, 1] * max_width + min_y
        n_cells = selected.shape[0]
        coords = np.column_stack([x_um, y_um, np.full(n_cells, tz)])

        # Inverse PCA: min-max denorm -> PCA inverse -> z-score inverse -> exponentiate
        pcs_norm = selected[:, 3:3+actual_pcs]
        pcs = pcs_norm * pc_range + pc_min
        X_scaled = pca.inverse_transform(pcs)
        X_log2 = scaler.inverse_transform(X_scaled)  # now in log2(X+1) space
        X_expr = np.power(2, X_log2) - 1  # reverse log2(x+1)
        X_expr = np.clip(X_expr, 0, None)

        results[target_sec] = {
            "X": sp.csr_matrix(X_expr),
            "coords": coords,
            "cell_type": np.array(["NA"] * n_cells),
        }
        print(f"    {target_sec}: {n_cells} predicted cells")

    return results



def format_output(results, gene_names, holdout_sections, method_params,
                  wall_time, output_path):
    """Write prediction.h5 in standardized format."""
    if not results:
        print("No results to write!")
        return

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
        print("No cells predicted!")
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
        uns.create_dataset("method_name", data="isost")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    """benchmark-pbya-v2 generation-only entry point (reuses isoST run_method)."""
    parser = argparse.ArgumentParser(
        description="isoST virtual slice generation (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--epochs", nargs=3, type=int, default=[100, 100, 100],
                        help="Training epochs for 3 phases (authors use 100,100,100)")
    parser.add_argument("--batch-num", type=int, default=5,
                        help="Training batch splits (paper uses 5)")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]

    print(f"Loading training-only input {args.input}...")
    train_adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(train_adata, target_sections)  # defense in depth
    gene_names = train_adata.var_names.tolist()
    target_z = {s: z for s, z in targets}

    print(f"Running isoST (generation-only) for {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(train_adata, target_z, seed=args.seed,
                         epochs=args.epochs, device=args.device,
                         batch_num=args.batch_num)
    wall_time = time.time() - t0

    method_params = {"seed": args.seed, "epochs": args.epochs,
                     "device": args.device, "generation_only": True}
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="isost")


if __name__ == "__main__":
    main()
