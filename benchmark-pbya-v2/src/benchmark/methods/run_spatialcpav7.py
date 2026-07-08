"""SpatialCPA-v7 (fused-transport + 3D communication) — benchmark-pbya-v2 wrapper.

v7 synthesizes each held-out section from its two flanking *training* slices and
a scalar target z, with no held-out (x, y) or content. Like v6 it is
training-free, but it upgrades the two stages the benchmark scores:

  * Placement is a **fused Gromov-Wasserstein** morph/interpolation between the
    flanking slices, which preserves the intra-slice neighbourhood graph across
    the morph (not just the marginals) — helping Moran's-I and neighbourhood
    agreement.
  * Annotation is a foundation-model cell-state prior constrained to the
    interpolated composition and refined by a **2D + 3D cell-cell communication**
    MRF: in-plane niche plus a cross-slice (z-stacking) niche estimated from the
    real cells above/below each virtual cell, and a ligand-receptor flux prior.

Leakage safeguards are shared with the other v2 wrappers:
  * the input file excludes the held-out section (built by run_benchmark);
    ``guard_no_holdout`` re-checks this,
  * labels are built from the (all-training) input only,
  * expression normalization is per-cell (no pooled statistic),
  * every synthesized position/type/profile derives from the flanking training
    slices + the target z.

The ``spatialcpav7`` package is imported from the repository root; set
SPATIALCPAV7_ROOT to override auto-discovery.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))            # _v2_io
sys.path.insert(0, str(_HERE.parent))     # leakage_guard
import _v2_io  # noqa: E402
import leakage_guard  # noqa: E402


# ── Locate the spatialcpav7 source package ────────────────────────────────────
def _add_spatialcpav7_to_path():
    candidates = []
    env_root = os.environ.get("SPATIALCPAV7_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(__file__).resolve().parents)
    for cand in candidates:
        if (cand / "spatialcpav7" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV7_ROOT = _add_spatialcpav7_to_path()


def check_environment():
    if _SPATIALCPAV7_ROOT is None:
        print("ERROR: could not locate the `spatialcpav7` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav7
        from spatialcpav7 import SpatialCPAv7, SpatialCPAv7Config  # noqa: F401
        print(f"spatialcpav7 v{getattr(spatialcpav7, '__version__', '?')} "
              f"imported from {_SPATIALCPAV7_ROOT} "
              f"(default placement: {SpatialCPAv7Config().synthesis.placement})")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav7: {e}", file=sys.stderr)
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


def build_stack(adata, ct_all):
    from spatialcpav7 import Slice, SliceStack
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
            section_id=str(sec),
        ))
    return SliceStack(slices)


# ── Method execution ──────────────────────────────────────────────────────────
def run_method(adata, targets, gene_names, args):
    from spatialcpav7 import SpatialCPAv7, SpatialCPAv7Config
    from spatialcpav7.foundation_assets import load_lr_pairs

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    n_cells = sum(s.n_spots for s in stack.slices)
    if n_cells < 8 or stack.n_slices < 2:
        print(f"  SKIP: need >=2 training sections and >=8 cells "
              f"(have {stack.n_slices} sections, {n_cells} cells)")
        return {}

    cfg = SpatialCPAv7Config()
    cfg.seed = args.seed
    cfg.synthesis.seed = args.seed
    cfg.synthesis.placement = args.placement
    cfg.synthesis.expression_mode = args.expression_mode
    cfg.synthesis.transfer_alpha = args.transfer_alpha
    cfg.synthesis.count_mode = args.count_mode
    cfg.embedding.method = args.embedding
    cfg.embedding.n_components = args.embed_dim
    cfg.embedding.fm_gene_embedding_path = args.fm_gene_embedding
    cfg.embedding.cross_slice_anchor = not args.no_cross_slice_anchor
    cfg.transport.epsilon = args.ot_epsilon
    cfg.transport.embed_weight = args.ot_embed_weight
    cfg.transport.alpha_gw = args.alpha_gw
    cfg.transport.max_ot_cells = args.max_ot_cells
    cfg.transport.deshrink = not args.no_deshrink
    cfg.communication.enabled = not args.no_communication
    cfg.communication.n_sweeps = args.niche_sweeps
    cfg.communication.niche_weight = args.niche_weight
    cfg.communication.k_neighbors = args.niche_k
    cfg.communication.enable_3d = not args.no_3d_communication
    cfg.communication.niche3d_weight = args.niche3d_weight
    cfg.communication.lr_weight = args.lr_weight
    cfg.annotation.enabled = not args.no_annotation
    cfg.annotation.classifier = args.classifier
    cfg.annotation.anchor_weight = args.anchor_weight
    cfg.annotation.constrain_composition = not args.no_composition_constraint

    lr_pairs = load_lr_pairs(args.lr_pairs) if args.lr_weight > 0 else None

    print(f"  placement={cfg.synthesis.placement}, embedding={cfg.embedding.method}, "
          f"alpha_gw={cfg.transport.alpha_gw}, classifier={cfg.annotation.classifier}, "
          f"niche={'on' if cfg.communication.enabled else 'off'} "
          f"(sweeps={cfg.communication.n_sweeps}, 3d={'on' if cfg.communication.enable_3d else 'off'}), "
          f"expr={cfg.synthesis.expression_mode}")

    gen = SpatialCPAv7(stack, gene_names=gene_names,
                       cell_type_names=cell_type_names, cfg=cfg, lr_pairs=lr_pairs)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: synthesizing virtual slice at z={z:.2f} (fused-GW)...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if getattr(gen, "_last_dissimilarity", None) is not None:
            print(f"    adaptive: flanking dissimilarity="
                  f"{gen._last_dissimilarity:.3f} (threshold {cfg.transport.adaptive_threshold}) "
                  f"-> {gen._last_placement}")
        n = vs.coords.shape[0]
        if n == 0:
            print(f"    WARNING: 0 cells synthesized for {sec}")
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
        description="SpatialCPA-v7 generation-only wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    # Placement.
    parser.add_argument("--placement", default="adaptive",
                        choices=["adaptive", "fgw_morph", "interpolate", "backbone",
                                 "fgw_geodesic"],
                        help="adaptive (default — fused-GW morph when the flanking "
                             "slices are near-identical, both-slice interpolation "
                             "when distinct), fgw_morph, interpolate, backbone, or "
                             "fgw_geodesic (ablation)")
    # Embedding / foundation-model prior.
    parser.add_argument("--embedding", default="pca",
                        choices=["pca", "coexpr", "fm_gene", "concat"],
                        help="cell-state embedding (pca default)")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--fm-gene-embedding", default=None,
                        help="path to a pretrained gene-embedding matrix "
                             "(.npz genes/embedding or panel-aligned .npy)")
    parser.add_argument("--no-cross-slice-anchor", action="store_true",
                        help="disable cross-slice mutual-NN embedding anchoring")
    # Fused Gromov-Wasserstein transport.
    parser.add_argument("--alpha-gw", type=float, default=0.35,
                        help="fusion weight on the Gromov (structure) term "
                             "(0 = plain OT, v6 behaviour)")
    parser.add_argument("--ot-epsilon", type=float, default=0.05)
    parser.add_argument("--ot-embed-weight", type=float, default=0.20)
    parser.add_argument("--max-ot-cells", type=int, default=1200)
    parser.add_argument("--no-deshrink", action="store_true")
    parser.add_argument("--count-mode", default="interpolate",
                        choices=["interpolate", "lower", "upper", "mean"])
    # Cell-cell communication (niche) refinement.
    parser.add_argument("--no-communication", action="store_true",
                        help="disable the niche MRF label refinement (ablation)")
    parser.add_argument("--no-3d-communication", action="store_true",
                        help="disable only the 3D cross-slice niche term (ablation)")
    parser.add_argument("--niche-sweeps", type=int, default=8)
    parser.add_argument("--niche-weight", type=float, default=1.0)
    parser.add_argument("--niche3d-weight", type=float, default=1.5)
    parser.add_argument("--niche-k", type=int, default=10)
    parser.add_argument("--lr-weight", type=float, default=0.5,
                        help="ligand-receptor flux prior weight (0 disables)")
    parser.add_argument("--lr-pairs", default=None,
                        help="path to a 2-column ligand,receptor file "
                             "(default: built-in curated set)")
    # Annotation.
    parser.add_argument("--no-annotation", action="store_true",
                        help="keep the morphed real endpoint labels (ablation)")
    parser.add_argument("--classifier", default="spatial",
                        choices=["spatial", "labelprop", "prototype", "knn"],
                        help="spatial (interpolated type field, default), labelprop "
                             "(FM manifold label propagation), or a prototype / knn "
                             "FM-embedding classifier")
    parser.add_argument("--anchor-weight", type=float, default=3.0)
    parser.add_argument("--no-composition-constraint", action="store_true",
                        help="do not pin composition to the interpolated flanking mix")
    # Expression.
    parser.add_argument("--expression-mode", default="endpoint",
                        choices=["endpoint", "transfer", "blend"])
    parser.add_argument("--transfer-alpha", type=float, default=0.5)
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]

    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = adata.var_names.tolist()
    _normalize_expression(adata)

    print(f"Running SpatialCPA-v7 (generation-only) for targets "
          f"{[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "placement": args.placement,
        "embedding": args.embedding, "embed_dim": args.embed_dim,
        "fm_gene_embedding": args.fm_gene_embedding,
        "cross_slice_anchor": not args.no_cross_slice_anchor,
        "alpha_gw": args.alpha_gw, "ot_epsilon": args.ot_epsilon,
        "ot_embed_weight": args.ot_embed_weight, "max_ot_cells": args.max_ot_cells,
        "deshrink": not args.no_deshrink, "count_mode": args.count_mode,
        "annotation": not args.no_annotation, "classifier": args.classifier,
        "anchor_weight": args.anchor_weight,
        "composition_constraint": not args.no_composition_constraint,
        "communication": not args.no_communication,
        "communication_3d": not args.no_3d_communication,
        "niche_sweeps": args.niche_sweeps, "niche_weight": args.niche_weight,
        "niche3d_weight": args.niche3d_weight, "niche_k": args.niche_k,
        "lr_weight": args.lr_weight,
        "expression_mode": args.expression_mode, "transfer_alpha": args.transfer_alpha,
        "training_free": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav7_gen")


if __name__ == "__main__":
    main()
