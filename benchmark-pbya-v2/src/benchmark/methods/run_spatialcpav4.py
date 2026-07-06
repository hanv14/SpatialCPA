"""SpatialCPA-v4 (transformer) — benchmark-pbya-v2 generation-only wrapper.

Leakage-hardened variant of the v1 wrapper. It consumes the shared v2 contract:
a training-only, already-re-registered input plus a scalar target z per held-out
section. It NEVER receives the held-out (x, y); it synthesizes each slice de novo
via the occupancy head (grid over the flanking training slices' bbox) and keeps
grid points above the occupancy threshold. The cell count is emergent.

All leakage safeguards live in benchmark.leakage_guard / _v2_io:
  * the input file excludes the held-out section (built by run_benchmark);
    ``guard_no_holdout`` re-checks this,
  * labels are built from the (all-training) input only,
  * expression normalization is per-cell (no pooled statistic).

The ``spatialcpav4`` package is imported from the repository root; set
SPATIALCPAV4_ROOT to override auto-discovery.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# Make sibling helpers and the benchmark package importable when run as a script.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # _v2_io
sys.path.insert(0, str(_HERE.parent))     # leakage_guard
import _v2_io  # noqa: E402
import leakage_guard  # noqa: E402


# ── Locate the spatialcpav4 source package ────────────────────────────────────

def _add_spatialcpav4_to_path():
    candidates = []
    env_root = os.environ.get("SPATIALCPAV4_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(__file__).resolve().parents)
    for cand in candidates:
        if (cand / "spatialcpav4" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV4_ROOT = _add_spatialcpav4_to_path()


def check_environment():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch not available: {e}", file=sys.stderr)
        return False
    if _SPATIALCPAV4_ROOT is None:
        print("ERROR: could not locate the `spatialcpav4` package.", file=sys.stderr)
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    """Per-cell normalization (leakage-safe: no statistic pooled across cells)."""
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


def build_slices(adata, cell_type_names, region_names, ct_all, reg_all):
    """Build spatialcpav4 Slice objects from the training-only input."""
    from spatialcpav4 import Slice
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X = _to_dense_f32(adata.X)
    labels = sorted(np.unique(sections),
                    key=lambda s: np.median(coords[sections == s, 2]))
    slices = []
    for sec in labels:
        m = sections == sec
        slices.append(Slice(
            expression=X[m],
            coords_xy=coords[m, :2],
            z_values=coords[m, 2],
            cell_type_indices=ct_all[m] if ct_all is not None else None,
            region_indices=reg_all[m] if reg_all is not None else None,
            section_id=str(sec),
        ))
    return slices


# ── Method execution ──────────────────────────────────────────────────────────

def run_method(adata, targets, gene_names, args):
    """Train on the training-only input and synthesize each target slice."""
    import torch
    from spatialcpav4 import (
        SliceStack, SpatialCPATransformer, SpatialCPAv4Config, Trainer,
        Predictor, build_triplet_samples, set_seed, load_model,
    )

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Labels: the entire input is training, so the vocabulary is train-only.
    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    reg_all, region_names = leakage_guard.build_labels_train_only(
        adata, "region", train_mask, seed=args.seed, leiden_fallback=False)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}; "
          f"regions: {None if region_names is None else len(region_names)}")

    slices = build_slices(adata, cell_type_names, region_names, ct_all, reg_all)
    n_cells = sum(s.n_spots for s in slices)
    if n_cells < 8 or len(slices) < 3:
        print(f"  SKIP: need >=3 training sections and >=8 cells "
              f"(have {len(slices)} sections, {n_cells} cells)")
        return {}

    stack = SliceStack(slices)
    n_genes = stack.n_genes
    n_cell_types = len(cell_type_names) if cell_type_names is not None else None
    n_regions = len(region_names) if region_names is not None else None

    cfg = SpatialCPAv4Config()
    cfg.model.hidden_dim = args.hidden_dim
    cfg.model.num_layers = args.num_layers
    cfg.model.num_heads = args.num_heads
    cfg.model.dropout = args.dropout
    cfg.data.n_neighbors = args.neighbors
    cfg.data.negative_ratio = args.negative_ratio
    cfg.train.lr = args.lr
    cfg.train.epochs = args.epochs
    cfg.train.seed = args.seed
    cfg.train.device = device
    cfg.train.batch_size = min(args.batch_size, max(8, n_cells))
    cfg.train.checkpoint_dir = str(Path(args.output).parent / "spatialcpav4_ckpt")
    cfg.inference.occupancy_threshold = args.occupancy_threshold
    cfg.inference.grid_points = args.grid_points
    cfg.inference.grid_type = args.grid_type

    coord_scale = stack.estimate_coord_scale()
    print(f"  coord scale: {coord_scale:.4f}; building triplet samples "
          f"(k={cfg.data.n_neighbors})...")
    samples = build_triplet_samples(
        stack, n_neighbors=cfg.data.n_neighbors,
        negative_ratio=cfg.data.negative_ratio,
        negative_min_dist_factor=cfg.data.negative_min_dist_factor, seed=args.seed)

    model = SpatialCPATransformer(n_genes, n_cell_types, n_regions,
                                  cfg.model, coord_scale=coord_scale)
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}; "
          f"training up to {cfg.train.epochs} epochs (batch={cfg.train.batch_size})...")
    Trainer(model, stack, samples, cfg).train(verbose=True)

    best = Path(cfg.train.checkpoint_dir) / "best.pt"
    if best.exists():
        model = load_model(str(best), device=device)

    predictor = Predictor(model, gene_names=gene_names,
                          cell_type_names=cell_type_names,
                          region_names=region_names, device=device,
                          n_neighbors=cfg.data.n_neighbors)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: synthesizing virtual slice at z={z:.2f} "
              f"(grid={cfg.inference.grid_points} {cfg.inference.grid_type}, "
              f"occ>{cfg.inference.occupancy_threshold})...")
        try:
            vs = predictor.generate_virtual_slice(
                z=z, slices=stack.slices, xy_bounds=None,
                n_grid_points=cfg.inference.grid_points,
                occupancy_threshold=cfg.inference.occupancy_threshold,
                grid_type=cfg.inference.grid_type,
                batch_size=cfg.inference.batch_size, seed=args.seed)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            print(f"    WARNING: 0 cells passed occupancy threshold for {sec}")
            continue
        cell_type = vs.cell_type.astype(str) if vs.cell_type is not None else np.array(["NA"] * n)
        results[sec] = {
            "X": sp.csr_matrix(_to_dense_f32(vs.expression)),
            "coords": vs.coords.astype(np.float64),
            "cell_type": cell_type,
        }
        print(f"    -> {n} cells synthesized")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCPA-v4 generation-only wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--occupancy-threshold", type=float, default=0.5)
    parser.add_argument("--grid-points", type=int, default=1000)
    parser.add_argument("--grid-type", default="regular", choices=["regular", "random"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]

    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)  # defense in depth
    gene_names = adata.var_names.tolist()
    _normalize_expression(adata)

    print(f"Running SpatialCPA-v4 (generation-only) for targets "
          f"{[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers, "num_heads": args.num_heads,
        "dropout": args.dropout, "neighbors": args.neighbors,
        "negative_ratio": args.negative_ratio,
        "occupancy_threshold": args.occupancy_threshold,
        "grid_points": args.grid_points, "grid_type": args.grid_type,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
        "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav4_gen")


if __name__ == "__main__":
    main()
