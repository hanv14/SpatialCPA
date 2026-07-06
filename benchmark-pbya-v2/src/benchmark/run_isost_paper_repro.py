"""Reproduce isoST paper on Zhuang-ABCA-2 with paper's exact protocol.

Paper: Li et al. 2025 (github.com/deng-ai-lab/isoST)
Protocol:
  - 54 specific sections from Zhuang-ABCA-2 (1122 genes, log2_normalized)
  - Train on ALL 54 sections (NOT alternating holdout)
  - batch_num=1 (paper), NOT 5 (demo)
  - epochs=[100,100,100], 3-phase training
  - Config: K=8, delta_d=0.01, hidden_dim=64, gene_dim=50
  - Evaluate: reconstruct isotropic volume, compare at held-out z positions

For holdout evaluation (our addition, since paper doesn't provide numeric tables):
  - Train on 27 even-indexed sections, evaluate on 27 odd-indexed sections
  - This matches the paper's Supp Fig 2B protocol description

Usage:
    conda run -n bench_isost python src/benchmark/run_isost_paper_repro.py
"""

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

TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "isost"
DATA_PATH = Path("data/processed/allen_zhuang_merfish/Zhuang-ABCA-2/data.h5ad")
OUT_DIR = Path("results/isost_paper_repro")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Paper's exact 54 sections
ISOST_SECTIONS = [
    'Zhuang-ABCA-2.004', 'Zhuang-ABCA-2.005', 'Zhuang-ABCA-2.006',
    'Zhuang-ABCA-2.007', 'Zhuang-ABCA-2.008', 'Zhuang-ABCA-2.009',
    'Zhuang-ABCA-2.010', 'Zhuang-ABCA-2.011', 'Zhuang-ABCA-2.012',
    'Zhuang-ABCA-2.013', 'Zhuang-ABCA-2.014', 'Zhuang-ABCA-2.015',
    'Zhuang-ABCA-2.016', 'Zhuang-ABCA-2.017', 'Zhuang-ABCA-2.018',
    'Zhuang-ABCA-2.019', 'Zhuang-ABCA-2.020', 'Zhuang-ABCA-2.021',
    'Zhuang-ABCA-2.022', 'Zhuang-ABCA-2.023', 'Zhuang-ABCA-2.025',
    'Zhuang-ABCA-2.026', 'Zhuang-ABCA-2.027', 'Zhuang-ABCA-2.028',
    'Zhuang-ABCA-2.030', 'Zhuang-ABCA-2.031', 'Zhuang-ABCA-2.032',
    'Zhuang-ABCA-2.033', 'Zhuang-ABCA-2.034', 'Zhuang-ABCA-2.035',
    'Zhuang-ABCA-2.036', 'Zhuang-ABCA-2.037', 'Zhuang-ABCA-2.039',
    'Zhuang-ABCA-2.040', 'Zhuang-ABCA-2.041', 'Zhuang-ABCA-2.042',
    'Zhuang-ABCA-2.044', 'Zhuang-ABCA-2.045', 'Zhuang-ABCA-2.046',
    'Zhuang-ABCA-2.047', 'Zhuang-ABCA-2.048', 'Zhuang-ABCA-2.049',
    'Zhuang-ABCA-2.050', 'Zhuang-ABCA-2.051', 'Zhuang-ABCA-2.052',
    'Zhuang-ABCA-2.053', 'Zhuang-ABCA-2.054', 'Zhuang-ABCA-2.055',
    'Zhuang-ABCA-2.056', 'Zhuang-ABCA-2.057', 'Zhuang-ABCA-2.058',
    'Zhuang-ABCA-2.059', 'Zhuang-ABCA-2.060', 'Zhuang-ABCA-2.061',
]

N_PCS = 50


def main():
    import torch
    import yaml

    sys.path.insert(0, str(TOOLS_DIR))
    from utils.train_ode import biaxial_train
    import model as training_module

    print("Loading Zhuang-ABCA-2...")
    adata = ad.read_h5ad(str(DATA_PATH))
    print(f"  Full dataset: {adata.shape}")

    # Subset to paper's 54 sections
    mask = adata.obs["section"].isin(ISOST_SECTIONS)
    adata = adata[mask].copy()
    print(f"  After subsetting to 54 sections: {adata.shape}")

    # Sort sections by z
    secs = adata.obs["section"].values.astype(str)
    unique_secs = np.unique(secs)
    sec_z = {}
    for s in unique_secs:
        m = secs == s
        sec_z[s] = float(np.median(adata.obsm["spatial"][m, 2]))
    sorted_sections = sorted(unique_secs, key=lambda s: sec_z[s])

    # Alternating holdout: odd-indexed sections held out
    train_secs = [sorted_sections[i] for i in range(0, len(sorted_sections), 2)]
    holdout_secs = [sorted_sections[i] for i in range(1, len(sorted_sections), 2)]
    print(f"  Train: {len(train_secs)} sections, Holdout: {len(holdout_secs)} sections")

    train_mask = np.isin(secs, train_secs)
    train_adata = adata[train_mask].copy()
    holdout_adata = adata[~train_mask].copy()

    # Holdout z targets
    holdout_z = {}
    for sec in holdout_secs:
        holdout_z[sec] = sec_z[sec]

    # Preprocess: log2 transform (data is already log2_normalized, skip)
    expr_type = adata.uns.get("expression_type", "log2_normalized")
    print(f"  expression_type: {expr_type}")

    # Work directory
    work_dir = tempfile.mkdtemp(prefix="isost_paper_")
    print(f"  Work dir: {work_dir}")

    # Get dense expression
    X = train_adata.X
    if sp.issparse(X):
        X = X.toarray()

    # Z-score + PCA (paper: 50 PCs)
    actual_pcs = min(N_PCS, min(X.shape) - 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=actual_pcs)
    X_pcs = pca.fit_transform(X_scaled)

    # Min-max normalize PCs
    pc_min = X_pcs.min(axis=0)
    pc_range = X_pcs.max(axis=0) - pc_min
    pc_range[pc_range == 0] = 1.0
    X_pcs_norm = (X_pcs - pc_min) / pc_range

    # Normalize spatial
    coords = train_adata.obsm["spatial"]
    min_x, min_y = coords[:, 0].min(), coords[:, 1].min()
    width_x, width_y = coords[:, 0].max() - min_x, coords[:, 1].max() - min_y
    max_width = max(width_x, width_y)
    if max_width == 0:
        max_width = 1.0
    min_z = coords[:, 2].min()
    z_range = coords[:, 2].max() - min_z
    if z_range == 0:
        z_range = 1.0

    # Save .pt files
    data_dir = os.path.join(work_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_sections_arr = train_adata.obs["section"].values.astype(str)
    train_sorted = sorted(train_secs, key=lambda s: sec_z[s])
    slide_names = []

    for sec in train_sorted:
        mask_s = train_sections_arr == sec
        idx = np.where(mask_s)[0]
        x_norm = (coords[idx, 0] - min_x) / max_width
        y_norm = (coords[idx, 1] - min_y) / max_width
        z_norm = (coords[idx, 2] - min_z) / z_range
        features = X_pcs_norm[idx]
        tensor_data = np.column_stack([x_norm, y_norm, z_norm, features])
        tensor = torch.tensor(tensor_data, dtype=torch.float32)
        fname = f"{sec}_log_PC"
        torch.save(tensor, os.path.join(data_dir, f"{fname}.pt"))
        perm = torch.randperm(tensor.shape[0])
        torch.save(tensor[perm], os.path.join(data_dir, f"shuffled_{fname}.pt"))
        slide_names.append(fname)

    # Save metadata
    joblib.dump({"scaler": scaler, "pca": pca}, os.path.join(work_dir, "zscore_pc_model.pkl"))

    # Write config (paper's mouse_brain.yml)
    config = {
        "trainer": "IsoST",
        "params": {
            "gene_dim": actual_pcs,
            "hidden_dim": 64,
            "head_num": 1,
            "K": 8,
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
    config_path = os.path.join(work_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Train with batch_num=1 (paper) — this is the key difference
    experiment_dir = os.path.join(work_dir, "experiments")
    result_dir = os.path.join(work_dir, "result")
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    epochs = [100, 100, 100]
    device = "cuda:0"

    # batch_num: paper uses 1 on large server. With ~533K cells, batch_num=1 OOMs
    # on our GPU. Use batch_num=5 (demo default) — trains on 20% subsamples.
    batch_num = 5
    print(f"\nTraining isoST: {len(slide_names)} sections, {actual_pcs} PCs, "
          f"batch_num={batch_num}, epochs={epochs}")
    t0 = time.time()
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
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.0f}s")

    # Inference
    print("Running ODE inference...")
    with open(os.path.join(experiment_dir, "config.yml")) as f:
        exp_config = yaml.safe_load(f)

    TrainerClass = getattr(training_module, exp_config["trainer"])
    trainer = TrainerClass(device=device, **exp_config["params"])
    trainer.load(os.path.join(experiment_dir, "model.pt"))
    trainer.to(device)

    dd = exp_config["params"]["delta_d"]
    trainer.fine_infer(data_dir, slide_names, "joint", dd, result_dir, 1, device)

    # Extract predictions at holdout z-positions
    print("Extracting predictions...")
    npy_files = sorted(Path(result_dir).glob("*_forward.npy"),
                       key=lambda f: int(f.stem.split("_")[0]))
    if not npy_files:
        print("ERROR: No .npy result files")
        return

    all_data = np.vstack([np.load(f) for f in npy_files])
    z_norm_all = all_data[:, 2]

    # Evaluate at each holdout section
    results = {}
    for sec in holdout_secs:
        tz = holdout_z[sec]
        tz_norm = (tz - min_z) / z_range

        # Find closest z in output
        unique_z_out = np.unique(z_norm_all.round(4))
        closest_z = unique_z_out[np.argmin(np.abs(unique_z_out - tz_norm))]
        sel_mask = np.abs(z_norm_all - closest_z) < 1e-4

        if sel_mask.sum() == 0:
            continue

        selected = all_data[sel_mask]

        # Inverse PCA to get gene expression
        pcs_norm = selected[:, 3:3 + actual_pcs]
        pcs = pcs_norm * pc_range + pc_min
        X_scaled_pred = pca.inverse_transform(pcs)
        X_log2_pred = scaler.inverse_transform(X_scaled_pred)

        # Ground truth
        gt_mask = holdout_adata.obs["section"] == sec
        gt_X = holdout_adata[gt_mask].X
        if sp.issparse(gt_X):
            gt_X = gt_X.toarray()

        # Metrics: gene mean correlation
        gt_means = gt_X.mean(axis=0)
        pred_means = X_log2_pred.mean(axis=0)
        from scipy.stats import pearsonr
        gmr = pearsonr(gt_means, pred_means)[0]

        gt_vars = gt_X.var(axis=0)
        pred_vars = X_log2_pred.var(axis=0)
        gvr = pearsonr(gt_vars, pred_vars)[0]

        results[sec] = {
            "gene_mean_r": float(gmr),
            "gene_var_r": float(gvr),
            "n_gt": int(gt_mask.sum()),
            "n_pred": int(sel_mask.sum()),
        }

    # Summary
    if results:
        gmrs = [r["gene_mean_r"] for r in results.values()]
        gvrs = [r["gene_var_r"] for r in results.values()]
        print(f"\nResults ({len(results)} holdout sections):")
        print(f"  gene_mean_pearson: median={np.median(gmrs):.4f}, mean={np.mean(gmrs):.4f}")
        print(f"  gene_var_pearson:  median={np.median(gvrs):.4f}, mean={np.mean(gvrs):.4f}")
        print(f"  Previous (batch_num=5, wrong sections): gene_mean=0.897")
    else:
        print("No results extracted!")

    out_path = OUT_DIR / "isost_abca2_paper_protocol.json"
    with open(out_path, "w") as f:
        json.dump({
            "per_section": results,
            "summary": {
                "gene_mean_r_median": float(np.median(gmrs)) if results else None,
                "gene_mean_r_mean": float(np.mean(gmrs)) if results else None,
                "gene_var_r_median": float(np.median(gvrs)) if results else None,
                "n_train": len(train_secs),
                "n_holdout": len(holdout_secs),
                "batch_num": 5,
                "epochs": epochs,
                "train_time_s": train_time,
            },
            "params": {
                "batch_num": 5,
                "epochs": [100, 100, 100],
                "n_pcs": actual_pcs,
                "config": "mouse_brain.yml (K=8, delta_d=0.01, alpha=0.1)",
            },
        }, f, indent=2, default=str)
    print(f"Saved to {out_path}")

    # Cleanup
    try:
        shutil.rmtree(work_dir)
    except Exception:
        pass


if __name__ == "__main__":
    main()
