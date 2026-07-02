"""SpatialCPA method wrapper for virtual slice interpolation.

SpatialCPA: Continuous 3D Spatial transcriptomics Prediction and Atlas.
A coordinate-based neural field that learns a continuous function
h(x, y, z) -> (cell type, region, gene expression) over the tissue volume
from sparsely sampled 2D sections, then predicts expression at any
(x, y, z) coordinate.

Unlike the pairwise-interpolation methods (FEAST, isoST, stVGP), SpatialCPA
does not need flanking slices: it fits a single global model over all
training sections and evaluates it directly at the held-out cells'
coordinates. Predictions are refined with a cell-type-conditioned k-NN
lookup into the training cells (the method's inference stage).

The model code lives in the `spatialcpa/` package at the repository root
(one level above this benchmark project); it is imported from source rather
than pip-installed. Set SPATIALCPA_ROOT to override auto-discovery.

Pipeline per run:
  1. Drop the held-out section(s), keep the rest as training sections
  2. Normalize expression (raw counts -> normalize_total + log1p)
  3. Build one SpatialSection per training section (per-cell z retained)
  4. Train SpatialCPA (Fourier encoder -> MLP backbone -> 3 heads) with
     MSE + Pearson expression loss and gap-aware leave-one-out self-supervision
  5. For each held-out section, predict cell type + expression at the section's
     real (x, y, z) cell coordinates via VirtualSliceGenerator
  6. Write prediction.h5

Usage:
    conda run -n bench_spatialcpa python src/benchmark/methods/run_spatialcpa.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_10 \
        --output results/spatialcpa/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
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


# ── Locate the spatialcpa source package ──────────────────────────────────────

def _add_spatialcpa_to_path():
    """Make the top-level `spatialcpa` package importable.

    Order of resolution:
      1. $SPATIALCPA_ROOT (directory that contains spatialcpa/__init__.py)
      2. Walk parent directories of this file looking for spatialcpa/__init__.py
    """
    candidates = []
    env_root = os.environ.get("SPATIALCPA_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    # Walk up from this file (…/benchmark-pbya/src/benchmark/methods/run_spatialcpa.py)
    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    for cand in candidates:
        if (cand / "spatialcpa" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPA_ROOT = _add_spatialcpa_to_path()


def check_environment():
    """Verify torch and the spatialcpa package are importable."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch not available: {e}", file=sys.stderr)
        return False
    if _SPATIALCPA_ROOT is None:
        print("ERROR: could not locate the `spatialcpa` package. "
              "Set SPATIALCPA_ROOT to the directory that contains "
              "spatialcpa/__init__.py.", file=sys.stderr)
        return False
    try:
        import torch
        from spatialcpa import SpatialCPA, SpatialCPATrainer, VirtualSliceGenerator  # noqa: F401
        print(f"spatialcpa imported from {_SPATIALCPA_ROOT}, "
              f"CUDA: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpa: {e}", file=sys.stderr)
        return False


# ── Input preparation ─────────────────────────────────────────────────────────

def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    """Normalize .X to a log space suitable for MSE training.

    Mirrors the convention used by the other neural wrappers (stVGP): raw
    counts are total-normalized and log1p'd; already-log data is left as is;
    intensity data is log1p'd. Returns the expression_type actually applied.
    """
    import scanpy as sc

    expr_type = adata.uns.get("expression_type", "raw_counts")
    if expr_type == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif expr_type in ("log1p_normalized", "log2_normalized", "normalized"):
        pass  # already on a log / normalized scale
    elif expr_type in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        # Unknown: normalize defensively
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return expr_type


def _build_cell_type_indices(adata, seed):
    """Return (ct_indices int64[n], cell_type_names list[str]).

    Uses obs['cell_type'] when present so predicted labels are comparable to
    ground truth. Falls back to Leiden clustering, then to a single class.
    """
    if "cell_type" in adata.obs.columns:
        labels = adata.obs["cell_type"].values.astype(str)
        names = sorted(pd.unique(labels).tolist())
        idx_map = {n: i for i, n in enumerate(names)}
        ct = np.array([idx_map[l] for l in labels], dtype=np.int64)
        print(f"  Cell types from obs['cell_type']: {len(names)}")
        return ct, names

    # Leiden fallback (same idea as the SpatialZ wrapper)
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
    """Split into train / holdout and build SpatialCPA training structures.

    Returns
    -------
    train_sections : list[SpatialSection]
    cell_type_names : list[str]
    ct_to_idx : dict[str, int]
    holdout_refs : dict[section -> AnnData] reference slices to predict at
    gene_names : list[str]
    """
    from spatialcpa.data import SpatialSection

    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    n_holdout = int(holdout_mask.sum())
    if n_holdout == 0:
        raise ValueError(f"No cells found for holdout sections {holdout_sections}")

    gene_names = adata.var_names.tolist()

    # Normalize expression in place (applies to both train and holdout rows;
    # holdout rows are only used for their coordinates, not their expression).
    expr_type = _normalize_expression(adata)
    print(f"  Expression type: {expr_type}")

    # Cell-type indices computed across ALL cells so the label space is shared.
    ct_all, cell_type_names = _build_cell_type_indices(adata, seed)
    ct_to_idx = {n: i for i, n in enumerate(cell_type_names)}

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.shape[1] < 3:
        raise ValueError("obsm['spatial'] must have 3 columns (x, y, z)")
    X = _to_dense_f32(adata.X)

    # ── Training sections (one SpatialSection per section label) ──────────────
    train_sections = []
    train_labels = np.unique(sections[~holdout_mask])
    # Sort by median z so gap-aware LOO sees them in physical order
    train_labels = sorted(train_labels, key=lambda s: np.median(coords[sections == s, 2]))
    for sec in train_labels:
        m = sections == sec
        train_sections.append(SpatialSection(
            expression=X[m],
            coords_xy=coords[m, :2],
            z_values=coords[m, 2],
            cell_type_indices=ct_all[m],
            section_id=str(sec),
        ))
    n_train_cells = sum(s.n_cells for s in train_sections)
    print(f"  Training: {len(train_sections)} sections, {n_train_cells} cells")

    # ── Held-out reference slices (coordinates only) ──────────────────────────
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
            var=pd.DataFrame(index=gene_names),
        )
        ref.obsm["spatial"] = coords[m, :2].astype(np.float32)
        holdout_refs[sec] = ref

    return train_sections, cell_type_names, ct_to_idx, holdout_refs, gene_names


# ── Method execution ──────────────────────────────────────────────────────────

def run_method(train_sections, cell_type_names, holdout_refs, gene_names,
               seed=42, epochs=50, device=None,
               knn_k=5, knn_z_weight=3.0, knn_alpha=0.0,
               use_true_celltypes=False):
    """Train SpatialCPA and predict expression at each held-out section.

    Returns dict: section_label -> {X (csr), coords (n,3), cell_type (n,)}.
    """
    import torch
    from spatialcpa.model import SpatialCPA
    from spatialcpa.trainer import SpatialCPATrainer
    from spatialcpa.inference import VirtualSliceGenerator
    from spatialcpa.fourier import FourierFeatureEncoder

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

    # Estimate spatial scales from all training coordinates
    all_coords = np.vstack([s.get_3d_coords() for s in train_sections])
    xy_scale, z_scale = FourierFeatureEncoder.estimate_scales(all_coords)
    print(f"  Estimated scales: xy={xy_scale:.3f}, z={z_scale:.3f}")

    model = SpatialCPA(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        n_regions=None,
        n_freq_xy=48,
        n_freq_z=32,
        xy_scale=xy_scale,
        z_scale=z_scale,
        backbone_hidden=512,
        backbone_output=256,
        backbone_layers=8,
        dropout=0.05,
        use_zinb=False,  # MSE mode: expression already log-normalized
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    # Batch size safe for DataLoader(drop_last=True): keep >= 1 full batch.
    batch_size = min(1024, max(4, n_train_cells - 1))

    trainer = SpatialCPATrainer(
        model=model,
        sections=train_sections,
        device=device,
        lr=5e-4,
        batch_size=batch_size,
        n_z_samples=3,
        z_jitter=0.3,
        loo_weight=0.3,
        expression_weight=1.0,
        corr_weight=0.5,
    )

    print(f"  Training SpatialCPA for {epochs} epochs (batch={batch_size})...")
    history = trainer.train(n_epochs=epochs, verbose=True)
    if history:
        print(f"  Final loss: {history[-1]['total']:.4f}")

    # ── Inference ─────────────────────────────────────────────────────────────
    generator = VirtualSliceGenerator(
        model=model,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
        region_names=None,
        device=device,
        train_sections=train_sections,  # enable k-NN refinement
    )

    results = {}
    for sec, ref in holdout_refs.items():
        n = ref.n_obs
        print(f"  {sec}: predicting {n} cells (z={float(np.median(ref.obs['z'])):.2f})...")

        true_ct = None
        if use_true_celltypes and "cell_type" in ref.obs.columns:
            true_ct = ref.obs["cell_type"].values

        try:
            sim = generator.generate_matching(
                reference_adata=ref,
                true_cell_types=true_ct,
                knn_k=knn_k,
                knn_z_weight=knn_z_weight,
                knn_alpha=knn_alpha,
                smooth_k=0,
                batch_size=4096,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        expr = _to_dense_f32(sim.X)
        xy = np.asarray(sim.obsm["spatial"], dtype=np.float64)[:, :2]
        z = ref.obs["z"].values.astype(np.float64)
        coords = np.column_stack([xy, z])

        if "cell_class" in sim.obs.columns:
            cell_type = sim.obs["cell_class"].values.astype(str)
        else:
            cell_type = np.array(["NA"] * n)

        results[sec] = {
            "X": sp.csr_matrix(expr.astype(np.float32)),
            "coords": coords,
            "cell_type": cell_type,
        }
        print(f"    -> {expr.shape[0]} cells predicted")

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def format_output(results, gene_names, holdout_sections, method_params,
                  wall_time, output_path):
    """Write prediction.h5 in the standardized benchmark format."""
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
        uns.create_dataset("method_name", data="spatialcpa")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SpatialCPA virtual slice generation")
    parser.add_argument("--input", required=True, help="Path to data.h5ad")
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output prediction.h5 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--knn-k", type=int, default=5,
                        help="training neighbors for k-NN refinement (0 disables)")
    parser.add_argument("--knn-z-weight", type=float, default=3.0)
    parser.add_argument("--knn-alpha", type=float, default=0.0,
                        help="blend: alpha*NN + (1-alpha)*kNN (0 = pure kNN)")
    parser.add_argument("--use-true-celltypes", action="store_true",
                        help="condition on ground-truth cell types (leaks labels; "
                             "off by default so cell types are predicted)")
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

    print("Running SpatialCPA...")
    t0 = time.time()
    results = run_method(
        train_sections, cell_type_names, holdout_refs, gene_names,
        seed=args.seed, epochs=args.epochs, device=args.device,
        knn_k=args.knn_k, knn_z_weight=args.knn_z_weight, knn_alpha=args.knn_alpha,
        use_true_celltypes=args.use_true_celltypes,
    )
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed,
        "epochs": args.epochs,
        "n_freq_xy": 48,
        "n_freq_z": 32,
        "backbone_hidden": 512,
        "backbone_output": 256,
        "backbone_layers": 8,
        "use_zinb": False,
        "knn_k": args.knn_k,
        "knn_z_weight": args.knn_z_weight,
        "knn_alpha": args.knn_alpha,
        "use_true_celltypes": args.use_true_celltypes,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
