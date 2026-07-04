"""SpatialCPA v3 method wrapper — generative virtual-slice construction.

SpatialCPA v3 learns the same continuous 3D field h(x, y, z) -> (cell type,
region, gene expression) as v1/v2, but its inference stage is fundamentally
different. Where the v1/v2 wrapper (`run_spatialcpa.py`) predicts expression at
the held-out section's TRUE (x, y) cell coordinates — i.e. it needs the very
slice it is meant to produce — v3 **generates** the held-out slice from scratch
using only:

  * the target z of the held-out section (the query coordinate), and
  * the two flanking training sections (already registered / aligned).

It never sees the held-out positions, cell counts, or cell types. The pipeline:

  1. Drop the held-out section(s), keep the rest as training sections.
  2. Normalize expression (raw counts -> normalize_total + log1p).
  3. Build one SpatialSection per training section (per-cell z retained).
  4. Train SpatialCPA v3 with a GENERATIVE Gaussian expression head.
  5. For each held-out section, find its two flanking training sections and
     synthesize a virtual slice at the held-out z with VirtualSliceGeneratorV3:
     de-novo positions (interpolated density field + blue-noise), sampled cell
     types (learned classifier blended with neighbor composition, smoothed into
     domains), and sampled expression (learned Gaussian).
  6. Write prediction.h5.

Because generated cells have no 1:1 correspondence with the held-out cells, the
benchmark's evaluator matches predicted cells to ground-truth cells by nearest
neighbour in (x, y) (see benchmark/evaluate.py::match_cells), which already
supports methods that synthesize positions in the flanking coordinate frame.

The model code lives in the `spatialcpav3/` package at the repository root
(one level above this benchmark project); it is imported from source rather
than pip-installed. Set SPATIALCPAV3_ROOT to override auto-discovery.

Usage:
    conda run -n bench_spatialcpav3 python src/benchmark/methods/run_spatialcpav3.py \
        --input data/processed/cosmx_nsclc_3d/data.h5ad \
        --holdout-sections section_10 \
        --output results/spatialcpav3/cosmx_nsclc_3d/loo_section_10/prediction.h5 \
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


# ── Locate the spatialcpav3 source package ────────────────────────────────────

def _add_spatialcpav3_to_path():
    """Make the top-level `spatialcpav3` package importable.

    Order of resolution:
      1. $SPATIALCPAV3_ROOT (directory that contains spatialcpav3/__init__.py)
      2. Walk parent directories of this file looking for spatialcpav3/__init__.py
    """
    candidates = []
    env_root = os.environ.get("SPATIALCPAV3_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)

    for cand in candidates:
        if (cand / "spatialcpav3" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV3_ROOT = _add_spatialcpav3_to_path()


def check_environment():
    """Verify torch and the spatialcpav3 package are importable."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        print(f"ERROR: torch not available: {e}", file=sys.stderr)
        return False
    if _SPATIALCPAV3_ROOT is None:
        print("ERROR: could not locate the `spatialcpav3` package. "
              "Set SPATIALCPAV3_ROOT to the directory that contains "
              "spatialcpav3/__init__.py.", file=sys.stderr)
        return False
    try:
        import torch
        from spatialcpav3 import (  # noqa: F401
            SpatialCPA, SpatialCPATrainer, VirtualSliceGeneratorV3)
        print(f"spatialcpav3 imported from {_SPATIALCPAV3_ROOT}, "
              f"CUDA: {torch.cuda.is_available()}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav3: {e}", file=sys.stderr)
        return False


# ── Input preparation ─────────────────────────────────────────────────────────

def _to_dense_f32(X):
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def _normalize_expression(adata):
    """Normalize .X to a log space suitable for training (mirrors v2 wrapper)."""
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
    """Return (ct_indices int64[n], cell_type_names list[str])."""
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
    """Split into train / holdout and build SpatialCPA v3 training structures.

    Returns
    -------
    train_sections : list[SpatialSection]   (one per training section, sorted by z)
    cell_type_names : list[str]
    holdout_plan : list[dict]  each with keys:
        'section'    held-out section label,
        'target_z'   query z (median z of the held-out section),
        'below_idx'  index into train_sections of the flanking section below,
        'above_idx'  index into train_sections of the flanking section above,
    gene_names : list[str]
    """
    from spatialcpav3.data import SpatialSection

    sections = adata.obs["section"].values.astype(str)
    holdout_mask = np.isin(sections, holdout_sections)
    if int(holdout_mask.sum()) == 0:
        raise ValueError(f"No cells found for holdout sections {holdout_sections}")

    gene_names = adata.var_names.tolist()

    expr_type = _normalize_expression(adata)
    print(f"  Expression type: {expr_type}")

    ct_all, cell_type_names = _build_cell_type_indices(adata, seed)

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.shape[1] < 3:
        raise ValueError("obsm['spatial'] must have 3 columns (x, y, z)")
    X = _to_dense_f32(adata.X)

    # ── Training sections (one SpatialSection per training section) ───────────
    train_labels = np.unique(sections[~holdout_mask])
    train_labels = sorted(train_labels,
                          key=lambda s: np.median(coords[sections == s, 2]))
    train_sections = []
    train_z_centers = []
    for sec in train_labels:
        m = sections == sec
        train_sections.append(SpatialSection(
            expression=X[m],
            coords_xy=coords[m, :2],
            z_values=coords[m, 2],
            cell_type_indices=ct_all[m],
            section_id=str(sec),
        ))
        train_z_centers.append(float(np.median(coords[m, 2])))
    train_z_centers = np.asarray(train_z_centers)
    n_train_cells = sum(s.n_cells for s in train_sections)
    print(f"  Training: {len(train_sections)} sections, {n_train_cells} cells")

    # ── Holdout plan: flanking training sections for each held-out slice ──────
    holdout_plan = []
    for sec in holdout_sections:
        m = sections == sec
        if not m.any():
            continue
        target_z = float(np.median(coords[m, 2]))

        below_candidates = np.where(train_z_centers < target_z)[0]
        above_candidates = np.where(train_z_centers > target_z)[0]
        if len(below_candidates) == 0 or len(above_candidates) == 0:
            print(f"  WARNING: section {sec} (z={target_z:.2f}) has no flanking "
                  f"training section on one side; skipping (cannot interpolate).")
            continue
        below_idx = int(below_candidates[np.argmax(train_z_centers[below_candidates])])
        above_idx = int(above_candidates[np.argmin(train_z_centers[above_candidates])])
        holdout_plan.append({
            "section": str(sec),
            "target_z": target_z,
            "below_idx": below_idx,
            "above_idx": above_idx,
        })
        print(f"  Holdout {sec}: z={target_z:.2f} <- flanked by "
              f"{train_sections[below_idx].section_id} "
              f"(z={train_z_centers[below_idx]:.2f}) & "
              f"{train_sections[above_idx].section_id} "
              f"(z={train_z_centers[above_idx]:.2f})")

    return train_sections, cell_type_names, holdout_plan, gene_names


# ── Method execution ──────────────────────────────────────────────────────────

def run_method(train_sections, cell_type_names, holdout_plan, gene_names,
               seed=42, epochs=50, device=None,
               ct_model_weight=0.3, ct_temperature=0.4,
               expr_temperature=0.35, expr_model_weight=0.1, expr_neighbor_k=2,
               count_jitter=0.0, n_cells=None,
               density_epochs=80, density_attention=False,
               density_field_weight=0.5):
    """Train SpatialCPA v3 and GENERATE each held-out slice from its neighbors.

    Returns dict: section_label -> {X (csr), coords (n,3), cell_type (n,)}.
    """
    import torch
    from spatialcpav3.model import SpatialCPA
    from spatialcpav3.trainer import SpatialCPATrainer
    from spatialcpav3.virtual_slice import VirtualSliceGeneratorV3
    from spatialcpav3.fourier import FourierFeatureEncoder
    from spatialcpav3.density import (DensityFieldModel, DensityFieldTrainer,
                                      DensitySampler)

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
        expression_mode="gaussian",  # generative expression head (v3)
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

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

    print(f"  Training SpatialCPA v3 for {epochs} epochs (batch={batch_size})...")
    history = trainer.train(n_epochs=epochs, verbose=True)
    if history:
        print(f"  Final loss: {history[-1]['total']:.4f}")

    # ── Learn the 3D density field (coordinate prior over all slices) ───────────
    density_sampler = None
    if len(train_sections) >= 2:
        try:
            density_model = DensityFieldModel(
                xy_scale=xy_scale, z_scale=z_scale, n_freq_xy=48, n_freq_z=24,
                hidden=192, n_layers=4, use_attention=density_attention)
            dtrainer = DensityFieldTrainer(density_model, train_sections,
                                           device=device, lr=2e-3, cells_per_bin=1.5)
            print(f"  Training 3D density field for {density_epochs} epochs "
                  f"(attention={density_attention})...")
            dhist = dtrainer.train(n_epochs=density_epochs, verbose=False)
            density_sampler = DensitySampler(density_model, train_sections,
                                             dtrainer.bin_size, device=device)
            print(f"  Density Poisson NLL: {dhist[-1]:.4f}, bin={dtrainer.bin_size:.2f}")
        except Exception as e:
            print(f"  WARNING: density field training failed ({e}); "
                  f"falling back to histogram positions")
            density_sampler = None

    # ── Generation ────────────────────────────────────────────────────────────
    generator = VirtualSliceGeneratorV3(
        model=model,
        cell_type_names=cell_type_names,
        gene_names=gene_names,
        device=device,
        density_sampler=density_sampler,
    )

    results = {}
    for plan in holdout_plan:
        sec = plan["section"]
        below = train_sections[plan["below_idx"]]
        above = train_sections[plan["above_idx"]]
        print(f"  {sec}: generating virtual slice at z={plan['target_z']:.2f} "
              f"from {below.section_id} & {above.section_id}...")

        try:
            virt = generator.generate(
                section_below=below,
                section_above=above,
                target_z=plan["target_z"],
                n_cells=n_cells,
                count_jitter=count_jitter,
                ct_model_weight=ct_model_weight,
                ct_smooth_k=8,
                ct_smooth_iters=3,
                ct_temperature=ct_temperature,
                expr_temperature=expr_temperature,
                expr_model_weight=expr_model_weight,
                expr_neighbor_k=expr_neighbor_k,
                density_field_weight=density_field_weight,
                relax_iters=2,
                seed=seed,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        expr = _to_dense_f32(virt.X)
        xy = np.asarray(virt.obsm["spatial"], dtype=np.float64)[:, :2]
        z = np.full(virt.n_obs, plan["target_z"], dtype=np.float64)
        coords = np.column_stack([xy, z])
        cell_type = virt.obs["cell_class"].values.astype(str)

        results[sec] = {
            "X": sp.csr_matrix(expr.astype(np.float32)),
            "coords": coords,
            "cell_type": cell_type,
        }
        print(f"    -> {expr.shape[0]} cells generated")

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
        print("No cells generated!")
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
        uns.create_dataset("method_name", data="spatialcpav3")
        uns.create_dataset("holdout_sections", data=json.dumps(holdout_sections))
        uns.create_dataset("method_params", data=json.dumps(method_params))
        uns.create_dataset("wall_time_seconds", data=wall_time)

    print(f"Wrote {cell_counter} generated cells to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCPA v3 generative virtual slice construction")
    parser.add_argument("--input", required=True, help="Path to data.h5ad")
    parser.add_argument("--holdout-sections", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Output prediction.h5 path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None,
                        help="torch device (default: cuda if available else cpu)")
    parser.add_argument("--ct-model-weight", type=float, default=0.3,
                        help="blend of learned classifier vs neighbor composition "
                             "for cell types (1=classifier only, 0=neighbors only)")
    parser.add_argument("--ct-temperature", type=float, default=0.4,
                        help="cell-type sampling temperature (0=argmax, coherent "
                             "domains; higher=more diverse)")
    parser.add_argument("--expr-temperature", type=float, default=0.35,
                        help="expression sampling temperature (0=grounded mean, "
                             "1=full learned variance)")
    parser.add_argument("--expr-model-weight", type=float, default=0.1,
                        help="blend of model mean vs neighbor-anchored mean for "
                             "expression (1=model only, lower=more grounded; raise "
                             "for generation far from any neighbor)")
    parser.add_argument("--expr-neighbor-k", type=int, default=2,
                        help="neighbors for the cell-type-conditioned expression "
                             "anchor (small k tracks real local texture -> higher "
                             "spatial-autocorrelation fidelity)")
    parser.add_argument("--density-epochs", type=int, default=80,
                        help="epochs to train the learned 3D density field "
                             "(coordinate prior; 0 disables -> histogram positions)")
    parser.add_argument("--density-attention", action="store_true",
                        help="use k-NN self-attention in the density field "
                             "(better count calibration; slower on CPU)")
    parser.add_argument("--density-field-weight", type=float, default=0.5,
                        help="blend of learned field vs local neighbor occupancy "
                             "for positions (1=field only, 0=neighbor occupancy)")
    parser.add_argument("--count-jitter", type=float, default=0.0,
                        help="relative jitter on interpolated cell count")
    parser.add_argument("--n-cells", type=int, default=None,
                        help="override number of generated cells (default: "
                             "interpolated from the two neighbors)")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)
    gene_names = adata.var_names.tolist()

    print(f"Preparing input (holdout: {args.holdout_sections})...")
    (train_sections, cell_type_names,
     holdout_plan, gene_names) = prepare_input(adata, args.holdout_sections, args.seed)
    del adata

    print("Running SpatialCPA v3...")
    t0 = time.time()
    results = run_method(
        train_sections, cell_type_names, holdout_plan, gene_names,
        seed=args.seed, epochs=args.epochs, device=args.device,
        ct_model_weight=args.ct_model_weight, ct_temperature=args.ct_temperature,
        expr_temperature=args.expr_temperature,
        expr_model_weight=args.expr_model_weight,
        expr_neighbor_k=args.expr_neighbor_k,
        density_epochs=args.density_epochs,
        density_attention=args.density_attention,
        density_field_weight=args.density_field_weight,
        count_jitter=args.count_jitter, n_cells=args.n_cells,
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
        "expression_mode": "gaussian",
        "ct_model_weight": args.ct_model_weight,
        "ct_temperature": args.ct_temperature,
        "expr_temperature": args.expr_temperature,
        "expr_model_weight": args.expr_model_weight,
        "expr_neighbor_k": args.expr_neighbor_k,
        "density_epochs": args.density_epochs,
        "density_attention": args.density_attention,
        "density_field_weight": args.density_field_weight,
        "count_jitter": args.count_jitter,
        "n_cells": args.n_cells,
    }
    format_output(results, gene_names, args.holdout_sections,
                  method_params, wall_time, args.output)


if __name__ == "__main__":
    main()
