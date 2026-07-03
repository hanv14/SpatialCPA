"""SpatialCPA v2 method wrapper for virtual slice interpolation.

SpatialCPA v2 is a biology-informed 3D neural field that learns the *joint*
spatial-cell-type-expression structure of a tissue from sparsely sampled 2D
sections, then predicts cell type + gene expression at any (x, y, z).

Architecture / inference improvements over v1 (see the ``spatialcpav2`` package
docstrings for detail):
  * calibrated multi-scale positional encoding + anisotropic random Fourier
    features
  * gated residual backbone with dual skip re-injection
  * FiLM cell-type conditioning + posterior-marginalised expression head
  * class-balanced training with Pearson + gene mean/variance losses
  * hybrid neural + 3D k-NN cell-type prediction, and moment-calibrated
    neural/k-NN expression fusion at inference

The model code lives in the ``spatialcpav2/`` package at the repository root
(one level above this benchmark project); it is imported from source rather
than pip-installed. Set SPATIALCPA_ROOT to override auto-discovery.

Usage:
    conda run -n bench_spatialcpa python src/benchmark/methods/run_spatialcpav2.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_10 \
        --output results/spatialcpav2/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
        --seed 42
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ── Locate the spatialcpav2 source package ────────────────────────────────────

def _add_spatialcpa_to_path():
    candidates = []
    env_root = os.environ.get("SPATIALCPA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    for cand in candidates:
        if (cand / "spatialcpav2" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPA_ROOT = _add_spatialcpa_to_path()


def check_environment():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch not available: {e}", file=sys.stderr)
        return False
    if _SPATIALCPA_ROOT is None:
        print("ERROR: could not locate the `spatialcpav2` package. "
              "Set SPATIALCPA_ROOT to the directory that contains "
              "spatialcpav2/__init__.py.", file=sys.stderr)
        return False
    try:
        import torch
        from spatialcpav2 import SpatialCPAv2, SpatialCPAv2Trainer, VirtualSliceGenerator  # noqa: F401
        print(f"spatialcpav2 imported from {_SPATIALCPA_ROOT}, "
              f"CUDA: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav2: {e}", file=sys.stderr)
        return False


# ── Input preparation ─────────────────────────────────────────────────────────

def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    """Normalize .X to a log space suitable for MSE training (mirrors v1)."""
    import scanpy as sc
    expr_type = adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif expr_type in ("log1p_normalized", "log2_normalized", "normalized"):
        pass
    elif expr_type in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return expr_type


def _build_cell_type_indices(adata, seed):
    if "cell_type" in adata.obs.columns:
        labels = adata.obs["cell_type"].values.astype(str)
        names = sorted(pd.unique(labels).tolist())
        idx_map = {n: i for i, n in enumerate(names)}
        ct = np.array([idx_map[l] for l in labels], dtype=np.int64)
        print(f"  Cell types from obs['cell_type']: {len(names)}")
        return ct, names
    try:
        import scanpy as sc
        tmp = adata.copy()
        sc.pp.pca(tmp, n_comps=min(30, tmp.n_vars - 1, tmp.n_obs - 1))
        sc.pp.neighbors(tmp, n_neighbors=15)
        sc.tl.leiden(tmp, resolution=1.0, random_state=seed)
        labels = tmp.obs["leiden"].values.astype(str)
        names = sorted(pd.unique(labels).tolist(), key=lambda s: int(s))
        idx_map = {n: i for i, n in enumerate(names)}
        ct = np.array([idx_map[l] for l in labels], dtype=np.int64)
        names = [f"leiden_{n}" for n in names]
        print(f"  Cell types from Leiden fallback: {len(names)}")
        return ct, names
    except Exception as e:
        print(f"  WARNING: Leiden fallback failed ({e}); using a single cell type")
        return np.zeros(adata.n_obs, dtype=np.int64), ["type_0"]


def prepare_input(adata, holdout_sections, seed):
    """Split into train / holdout and build SpatialCPAv2 training structures."""
    from spatialcpav2.data import SpatialSection

    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    if int(holdout_mask.sum()) == 0:
        raise ValueError(f"No cells found for holdout sections {holdout_sections}")

    gene_names = adata.var_names.tolist()
    expr_type = _normalize_expression(adata)
    print(f"  Expression type: {expr_type}")

    ct_all, cell_type_names = _build_cell_type_indices(adata, seed)
    ct_to_idx = {n: i for i, n in enumerate(cell_type_names)}

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.shape[1] < 3:
        raise ValueError("obsm['spatial'] must have 3 columns (x, y, z)")
    X = _to_dense_f32(adata.X)

    train_sections = []
    train_labels = np.unique(sections[~holdout_mask])
    train_labels = sorted(train_labels, key=lambda s: np.median(coords[sections == s, 2]))
    for sec in train_labels:
        m = sections == sec
        train_sections.append(SpatialSection(
            expression=X[m], coords_xy=coords[m, :2], z_values=coords[m, 2],
            cell_type_indices=ct_all[m], section_id=str(sec)))
    n_train_cells = sum(s.n_cells for s in train_sections)
    print(f"  Training: {len(train_sections)} sections, {n_train_cells} cells")

    holdout_refs = {}
    for sec in holdout_sections:
        m = sections == sec
        if not m.any():
            continue
        n = int(m.sum())
        ref = ad.AnnData(
            X=np.zeros((n, len(gene_names)), dtype=np.float32),
            obs=pd.DataFrame({"z": coords[m, 2].astype(np.float32)},
                             index=[f"{sec}_{i}" for i in range(n)]),
            var=pd.DataFrame(index=gene_names))
        ref.obsm["spatial"] = coords[m, :2].astype(np.float32)
        holdout_refs[sec] = ref

    return train_sections, cell_type_names, ct_to_idx, holdout_refs, gene_names


# ── Method execution ──────────────────────────────────────────────────────────

def run_method(train_sections, cell_type_names, holdout_refs, gene_names,
               seed=42, epochs=60, device=None,
               knn_k=8, knn_z_weight=3.0,
               ct_knn_weight=0.7, expr_knn_alpha=0.3,
               calibrate=False, use_true_celltypes=False):
    """Train SpatialCPAv2 and predict expression at each held-out section."""
    import torch
    from spatialcpav2.model import SpatialCPAv2
    from spatialcpav2.trainer import SpatialCPAv2Trainer
    from spatialcpav2.inference import VirtualSliceGenerator
    from spatialcpav2.fourier import FourierFeatureEncoder

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    n_train_cells = sum(s.n_cells for s in train_sections)
    if n_train_cells < 8:
        print("  SKIP: too few training cells (<8)")
        return {}

    n_genes = len(gene_names)
    n_cell_types = len(cell_type_names)

    all_coords = np.vstack([s.get_3d_coords() for s in train_sections])
    xy_scale, z_scale = FourierFeatureEncoder.estimate_scales(all_coords)
    xy_extent, z_extent = FourierFeatureEncoder.estimate_extent(all_coords)
    print(f"  Estimated scales: xy={xy_scale:.3f}, z={z_scale:.3f}; "
          f"extent xy={xy_extent:.1f}, z={z_extent:.1f}")

    model = SpatialCPAv2(
        n_genes=n_genes, n_cell_types=n_cell_types, n_regions=None,
        n_freq_xy=48, n_freq_z=32, xy_scale=xy_scale, z_scale=z_scale,
        xy_extent=xy_extent, z_extent=z_extent, n_rff=96,
        backbone_hidden=512, backbone_output=256, backbone_layers=8,
        decoder_layers=3, dropout=0.05, use_zinb=False)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = min(1024, max(4, n_train_cells - 1))
    trainer = SpatialCPAv2Trainer(
        model=model, sections=train_sections, device=device, lr=5e-4,
        batch_size=batch_size, n_z_samples=3, z_jitter=0.3, loo_weight=0.3,
        expression_weight=1.0, corr_weight=0.5, moment_weight=0.5,
        marginal_weight=0.3, class_balanced=True)

    print(f"  Training SpatialCPAv2 for {epochs} epochs (batch={batch_size})...")
    history = trainer.train(n_epochs=epochs, verbose=True)
    if history:
        print(f"  Final loss: {history[-1]['total']:.4f}")

    generator = VirtualSliceGenerator(
        model=model, cell_type_names=cell_type_names, gene_names=gene_names,
        region_names=None, device=device, train_sections=train_sections)

    results = {}
    for sec, ref in holdout_refs.items():
        n = ref.n_obs
        print(f"  {sec}: predicting {n} cells (z={float(np.median(ref.obs['z'])):.2f})...")
        true_ct = None
        if use_true_celltypes and "cell_type" in ref.obs.columns:
            true_ct = ref.obs["cell_type"].values
        try:
            sim = generator.generate_matching(
                reference_adata=ref, true_cell_types=true_ct,
                knn_k=knn_k, knn_z_weight=knn_z_weight,
                ct_knn_weight=ct_knn_weight, expr_knn_alpha=expr_knn_alpha,
                calibrate=calibrate, smooth_k=0, batch_size=4096)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        expr = _to_dense_f32(sim.X)
        xy = np.asarray(sim.obsm["spatial"], dtype=np.float64)[:, :2]
        z = ref.obs["z"].values.astype(np.float64)
        coords = np.column_stack([xy, z])
        cell_type = (sim.obs["cell_class"].values.astype(str)
                     if "cell_class" in sim.obs.columns else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(expr.astype(np.float32)),
                        "coords": coords, "cell_type": cell_type}
        print(f"    -> {expr.shape[0]} cells predicted")
    return results


# ── Output ────────────────────────────────────────────────────────────────────

def format_output(results, gene_names, holdout_sections, method_params,
                  wall_time, output_path):
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
        all_ct.extend([str(c) for c in r["cell_type"]])
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
        uns.create_dataset("method_name", data="spatialcpav2")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)
    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SpatialCPA v2 virtual slice generation")
    parser.add_argument("--input", required=True)
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default=None)
    parser.add_argument("--knn-k", type=int, default=8)
    parser.add_argument("--knn-z-weight", type=float, default=3.0)
    parser.add_argument("--ct-knn-weight", type=float, default=0.7,
                        help="weight of k-NN vote in cell-type blend (0=neural,1=kNN)")
    parser.add_argument("--expr-knn-alpha", type=float, default=0.3,
                        help="blend: alpha*neural + (1-alpha)*kNN for expression")
    parser.add_argument("--calibrate", action="store_true",
                        help="enable per-gene moment calibration (off by default)")
    parser.add_argument("--use-true-celltypes", action="store_true")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    (train_sections, cell_type_names, ct_to_idx,
     holdout_refs, gene_names) = prepare_input(adata, args.holdout_sections, args.seed)
    del adata

    print("Running SpatialCPA v2...")
    t0 = time.time()
    results = run_method(
        train_sections, cell_type_names, holdout_refs, gene_names,
        seed=args.seed, epochs=args.epochs, device=args.device,
        knn_k=args.knn_k, knn_z_weight=args.knn_z_weight,
        ct_knn_weight=args.ct_knn_weight, expr_knn_alpha=args.expr_knn_alpha,
        calibrate=args.calibrate, use_true_celltypes=args.use_true_celltypes)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "epochs": args.epochs,
        "n_freq_xy": 48, "n_freq_z": 32, "n_rff": 96,
        "backbone_hidden": 512, "backbone_output": 256, "backbone_layers": 8,
        "decoder_layers": 3, "use_zinb": False,
        "knn_k": args.knn_k, "knn_z_weight": args.knn_z_weight,
        "ct_knn_weight": args.ct_knn_weight, "expr_knn_alpha": args.expr_knn_alpha,
        "calibrate": args.calibrate,
        "use_true_celltypes": args.use_true_celltypes,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
