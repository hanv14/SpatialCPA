"""SpatialCPA-v4 (transformer) method wrapper for virtual-slice interpolation.

SpatialCPA-v4 learns ``{Slice(i-1), Slice(i+1)} -> Slice(i)`` with a Transformer:
for each target location it attends over the k nearest neighbors from each
flanking slice and predicts gene expression, cell type / region, and tissue
occupancy.  Unlike the coordinate-neural-field original SpatialCPA, it predicts
the middle slice directly from its two neighbors.

This wrapper matches the benchmark method interface exactly (same CLI, same
`prediction.h5` output) so it runs under `run_benchmark.py` with no changes to
the benchmark framework.  Per held-out section it:

  1. Drops the held-out section(s); keeps the rest as reference slices.
  2. Normalizes expression (raw counts -> normalize_total + log1p).
  3. Builds a SliceStack from the reference sections and precomputes k-NN
     triplet training samples (with occupancy negatives).
  4. Trains the transformer (expression + label + occupancy heads).
  5. For each held-out section, picks the nearest lower/upper reference slices
     and predicts expression + cell type at the section's real (x, y, z)
     coordinates.
  6. Writes prediction.h5.

The `spatialcpav4` package lives at the repository root (one level above this
benchmark project) and is imported from source.  Set SPATIALCPAV4_ROOT to
override auto-discovery.

Usage:
    conda run -n bench_spatialcpa python src/benchmark/methods/run_spatialcpav4.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_10 \
        --output results/spatialcpav4/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
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


# ── Locate the spatialcpav4 source package ────────────────────────────────────

def _add_spatialcpav4_to_path():
    """Make the top-level `spatialcpav4` package importable.

    Resolution order:
      1. $SPATIALCPAV4_ROOT (directory that contains spatialcpav4/__init__.py)
      2. Walk parent directories of this file looking for spatialcpav4/__init__.py
    """
    candidates = []
    env_root = os.environ.get("SPATIALCPAV4_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    for cand in candidates:
        if (cand / "spatialcpav4" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV4_ROOT = _add_spatialcpav4_to_path()


def check_environment():
    """Verify torch and the spatialcpav4 package are importable."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch not available: {e}", file=sys.stderr)
        return False
    if _SPATIALCPAV4_ROOT is None:
        print("ERROR: could not locate the `spatialcpav4` package. "
              "Set SPATIALCPAV4_ROOT to the directory that contains "
              "spatialcpav4/__init__.py.", file=sys.stderr)
        return False
    try:
        import torch
        from spatialcpav4 import SpatialCPATransformer, Trainer, Predictor  # noqa: F401
        print(f"spatialcpav4 imported from {_SPATIALCPAV4_ROOT}, "
              f"CUDA: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav4: {e}", file=sys.stderr)
        return False


# ── Input preparation ─────────────────────────────────────────────────────────

def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    """Normalize .X to a log space suitable for MSE training (mirrors stVGP).

    LEAKAGE SAFETY: every transform used here (normalize_total, log1p) is
    strictly per-cell / per-value — no statistic is pooled across cells — so
    applying it to the full AnnData yields exactly the same training-cell values
    as a train-only pass. No held-out information enters the training features.
    """
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


def _build_label_indices(adata, key, seed, train_mask):
    """Return (indices int64[n_all], names list[str]) for a categorical obs column.

    LEAKAGE SAFETY: the label vocabulary — and, in the Leiden fallback, the
    clustering itself — are derived from TRAINING cells only (``train_mask``), so
    no held-out information reaches training. Held-out cells are mapped into the
    training vocabulary where their label exists and get index -1 otherwise;
    either way held-out entries are never consumed during training (the wrapper
    only reads indices for reference sections), they are returned solely to keep
    the array aligned to ``adata``.

    Uses obs[key] when present so predicted labels are comparable to ground
    truth.  For cell_type, falls back to Leiden clustering then a single class.
    Returns (None, None) if the column is absent and no fallback applies.
    """
    train_idx = np.where(train_mask)[0]

    if key in adata.obs.columns:
        labels = adata.obs[key].values.astype(str)
        # Vocabulary from TRAINING cells only.
        names = sorted(pd.unique(labels[train_mask]).tolist())
        idx_map = {n: i for i, n in enumerate(names)}
        idx = np.array([idx_map.get(l, -1) for l in labels], dtype=np.int64)
        n_unseen = int((idx[~train_mask] == -1).sum()) if (~train_mask).any() else 0
        print(f"  {key} from obs['{key}']: {len(names)} classes (train-only vocab; "
              f"{n_unseen} held-out cells with unseen labels)")
        return idx, names

    if key != "cell_type":
        return None, None

    # Leiden fallback fit on TRAINING cells only.
    try:
        import scanpy as sc
        tmp = adata[train_mask].copy()
        sc.pp.pca(tmp, n_comps=min(30, tmp.n_vars - 1, tmp.n_obs - 1))
        sc.pp.neighbors(tmp, n_neighbors=15)
        sc.tl.leiden(tmp, resolution=1.0, random_state=seed)
        train_labels = tmp.obs["leiden"].values.astype(str)
        names = sorted(pd.unique(train_labels).tolist(), key=lambda s: int(s))
        idx_map = {n: i for i, n in enumerate(names)}
        # Held-out cells default to -1 (never used for training).
        idx = np.full(adata.n_obs, -1, dtype=np.int64)
        idx[train_idx] = np.array([idx_map[l] for l in train_labels], dtype=np.int64)
        names = [f"leiden_{n}" for n in names]
        print(f"  cell_type from Leiden fallback (train-only): {len(names)} classes")
        return idx, names
    except Exception as e:
        print(f"  WARNING: Leiden fallback failed ({e}); using a single cell type")
        return np.zeros(adata.n_obs, dtype=np.int64), ["type_0"]


def prepare_input(adata, holdout_sections, seed):
    """Split into reference / holdout and build SpatialCPA-v4 structures.

    Returns
    -------
    ref_slices : list[Slice]        reference (training) sections
    holdout_refs : dict[section -> dict(coords_xy, z)]
    gene_names : list[str]
    cell_type_names : list[str] or None
    region_names : list[str] or None
    """
    from spatialcpav4 import Slice

    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    if int(holdout_mask.sum()) == 0:
        raise ValueError(f"No cells found for holdout sections {holdout_sections}")

    gene_names = adata.var_names.tolist()
    expr_type = _normalize_expression(adata)
    print(f"  Expression type: {expr_type}")

    # Label spaces are built from TRAINING cells only (~holdout_mask) so no
    # held-out information leaks into training; see _build_label_indices.
    train_mask = ~holdout_mask
    ct_all, cell_type_names = _build_label_indices(adata, "cell_type", seed, train_mask)
    reg_all, region_names = _build_label_indices(adata, "region", seed, train_mask)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.shape[1] < 3:
        raise ValueError("obsm['spatial'] must have 3 columns (x, y, z)")
    X = _to_dense_f32(adata.X)

    # Reference sections (sorted by z-center inside SliceStack later).
    ref_slices = []
    ref_labels = np.unique(sections[~holdout_mask])
    ref_labels = sorted(ref_labels, key=lambda s: np.median(coords[sections == s, 2]))
    for sec in ref_labels:
        m = sections == sec
        ref_slices.append(Slice(
            expression=X[m],
            coords_xy=coords[m, :2],
            z_values=coords[m, 2],
            cell_type_indices=ct_all[m] if ct_all is not None else None,
            region_indices=reg_all[m] if reg_all is not None else None,
            section_id=str(sec),
        ))
    n_ref_cells = sum(s.n_spots for s in ref_slices)
    print(f"  Reference: {len(ref_slices)} sections, {n_ref_cells} cells")

    # Held-out reference coordinates (predict at these).
    holdout_refs = {}
    for sec in holdout_sections:
        m = sections == sec
        if not m.any():
            continue
        holdout_refs[sec] = {
            "coords_xy": coords[m, :2].astype(np.float32),
            "z": coords[m, 2].astype(np.float32),
        }

    return ref_slices, holdout_refs, gene_names, cell_type_names, region_names


# ── Method execution ──────────────────────────────────────────────────────────

def run_method(ref_slices, holdout_refs, gene_names, cell_type_names, region_names,
               args):
    """Train SpatialCPA-v4 and predict expression at each held-out section.

    Returns dict: section_label -> {X (csr), coords (n,3), cell_type (n,)}.
    """
    import torch
    from spatialcpav4 import (
        SliceStack, SpatialCPATransformer, SpatialCPAv4Config, Trainer,
        Predictor, build_triplet_samples, set_seed,
    )

    set_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    n_ref_cells = sum(s.n_spots for s in ref_slices)
    if n_ref_cells < 8:
        print("  SKIP: too few reference cells (<8)")
        return {}
    if len(ref_slices) < 3:
        print(f"  SKIP: need >=3 reference sections to form triplets "
              f"(have {len(ref_slices)})")
        return {}

    stack = SliceStack(ref_slices)
    n_genes = stack.n_genes
    n_cell_types = len(cell_type_names) if cell_type_names is not None else None
    n_regions = len(region_names) if region_names is not None else None

    # ── Config (all hyperparameters come from CLI; nothing hard-coded) ───────
    cfg = SpatialCPAv4Config()
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.num_heads = args.num_heads
    cfg.model.dropout = args.dropout
    cfg.data.n_neighbors = args.neighbors
    cfg.data.negative_ratio = args.negative_ratio
    cfg.loss.expression_weight = args.expression_weight
    cfg.loss.label_weight = args.label_weight
    cfg.loss.occupancy_weight = args.occupancy_weight
    cfg.train.lr = args.lr
    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    cfg.train.device = device
    cfg.inference.occupancy_threshold = args.occupancy_threshold
    # Batch size safe for tiny inputs.
    cfg.train.batch_size = min(args.batch_size, max(8, n_ref_cells))
    cfg.train.checkpoint_dir = str(Path(args.output).parent / "spatialcpav4_ckpt")
    if args.tensorboard:
        cfg.train.tensorboard_dir = str(Path(args.output).parent / "tb")

    coord_scale = stack.estimate_coord_scale()
    print(f"  Estimated coord scale: {coord_scale:.4f}")

    # ── Build training samples (cached KDTree neighbor search) ───────────────
    print(f"  Building triplet samples (k={cfg.data.n_neighbors} per side)...")
    samples = build_triplet_samples(
        stack,
        n_neighbors=cfg.data.n_neighbors,
        negative_ratio=cfg.data.negative_ratio,
        negative_min_dist_factor=cfg.data.negative_min_dist_factor,
        seed=args.seed,
    )
    print(f"  {len(samples)} training samples "
          f"({int(samples.has_target.sum())} tissue, "
          f"{int((samples.has_target == 0).sum())} background)")

    model = SpatialCPATransformer(
        n_genes=n_genes,
        n_cell_types=n_cell_types,
        n_regions=n_regions,
        cfg=cfg.model,
        coord_scale=coord_scale,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    trainer = Trainer(model, stack, samples, cfg)
    print(f"  Training for up to {cfg.train.epochs} epochs "
          f"(batch={cfg.train.batch_size})...")
    history = trainer.train(verbose=True)
    if history:
        print(f"  Final train loss: {history[-1]['train_total']:.4f}")

    # Reload best checkpoint if validation was used.
    best_path = Path(cfg.train.checkpoint_dir) / "best.pt"
    if best_path.exists():
        from spatialcpav4 import load_model
        model = load_model(str(best_path), device=device)

    predictor = Predictor(
        model,
        gene_names=gene_names,
        cell_type_names=cell_type_names,
        region_names=region_names,
        device=device,
        n_neighbors=cfg.data.n_neighbors,
    )

    # ── Inference: predict each held-out section from its flanking slices ────
    results = {}
    for sec, ref in holdout_refs.items():
        coords_xy = ref["coords_xy"]
        z = ref["z"]
        z_center = float(np.median(z))
        coords_3d = np.column_stack([coords_xy, z]).astype(np.float32)
        n = coords_3d.shape[0]

        lower, upper = Predictor._pick_flanking_slices(z_center, stack.slices)
        print(f"  {sec}: predicting {n} cells at z={z_center:.2f} "
              f"(lower={lower.section_id}, upper={upper.section_id})...")

        try:
            pred = predictor.predict(
                coords_3d, lower, upper, batch_size=cfg.inference.batch_size
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        expr = _to_dense_f32(pred.expression)
        cell_type = (
            pred.cell_type.astype(str) if pred.cell_type is not None
            else np.array(["NA"] * n)
        )
        results[sec] = {
            "X": sp.csr_matrix(expr.astype(np.float32)),
            "coords": coords_3d.astype(np.float64),
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
        uns.create_dataset("method_name", data="spatialcpav4")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} predicted cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCPA-v4 (transformer) virtual slice generation")
    parser.add_argument("--input", required=True, help="Path to data.h5ad")
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output prediction.h5 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available else cpu)")
    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Data
    parser.add_argument("--neighbors", type=int, default=10,
                        help="nearest neighbors per flanking slice")
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="background occupancy negatives per real spot")
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    # Loss
    parser.add_argument("--expression-weight", type=float, default=1.0)
    parser.add_argument("--label-weight", type=float, default=1.0)
    parser.add_argument("--occupancy-weight", type=float, default=1.0)
    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tensorboard", action="store_true",
                        help="write TensorBoard logs next to the output")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    (ref_slices, holdout_refs, gene_names,
     cell_type_names, region_names) = prepare_input(adata, args.holdout_sections, args.seed)
    del adata

    print("Running SpatialCPA-v4...")
    t0 = time.time()
    results = run_method(
        ref_slices, holdout_refs, gene_names, cell_type_names, region_names, args,
    )
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "neighbors": args.neighbors,
        "negative_ratio": args.negative_ratio,
        "occupancy_threshold": args.occupancy_threshold,
        "expression_weight": args.expression_weight,
        "label_weight": args.label_weight,
        "occupancy_weight": args.occupancy_weight,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
