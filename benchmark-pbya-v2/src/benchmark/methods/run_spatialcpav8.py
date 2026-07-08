"""SpatialCPA-v8 (symmetric OT bridge) — benchmark-pbya-v2 generation-only wrapper.

v8 synthesizes each held-out section from its two flanking *training* slices and a
scalar target z, with no held-out (x, y) or content. It is training-free (pure
numpy/scipy/sklearn, same ``bench_spatialcpa`` env as v4-v6). The headline advance
over v6 is the **bidirectional McCann barycentric bridge**: both flanking
populations are projected through the same entropic-OT map and drawn in the
z-interpolated ratio, producing one coherent sheet that is also the correct
mixture — removing v6's per-holdout morph-vs-interpolate trade-off. A density
calibration then matches the local cell density to the interpolated flanking
field, and a niche-aware annotator sets cell types. See ``spatialcpav8/README.md``.

The harness invokes this wrapper with only the shared generation-only arguments,
so every default in ``SpatialCPAv8Config`` is the intended production setting; the
extra CLI flags below are for ablations / tuning only.

Leakage safeguards are shared with the other v2 wrappers:
  * the input file excludes the held-out section (built by run_benchmark);
    ``guard_no_holdout`` re-checks this,
  * labels are built from the (all-training) input only,
  * expression normalization is per-cell (no pooled statistic),
  * every synthesized position/type/profile derives from the flanking training
    slices + the target z.

The ``spatialcpav8`` package is imported from the repository root; set
SPATIALCPAV8_ROOT to override auto-discovery.
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


# ── Locate the spatialcpav8 source package ────────────────────────────────────

def _add_spatialcpav8_to_path():
    candidates = []
    env_root = os.environ.get("SPATIALCPAV8_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend(Path(__file__).resolve().parents)
    for cand in candidates:
        if (cand / "spatialcpav8" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_SPATIALCPAV8_ROOT = _add_spatialcpav8_to_path()


def check_environment():
    if _SPATIALCPAV8_ROOT is None:
        print("ERROR: could not locate the `spatialcpav8` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav8
        from spatialcpav8 import SpatialCPAv8, SpatialCPAv8Config  # noqa: F401
        print(f"spatialcpav8 v{getattr(spatialcpav8, '__version__', '?')} "
              f"imported from {_SPATIALCPAV8_ROOT} "
              f"(default placement: {SpatialCPAv8Config().bridge.mode})")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav8: {e}", file=sys.stderr)
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
    from spatialcpav8 import Slice, SliceStack
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
    from spatialcpav8 import SpatialCPAv8, SpatialCPAv8Config

    # Labels: the entire input is training, so the vocabulary is train-only.
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

    cfg = SpatialCPAv8Config()
    cfg.seed = args.seed
    cfg.synthesis.seed = args.seed
    cfg.bridge.mode = args.placement
    cfg.bridge.adaptive_threshold = args.adaptive_threshold
    cfg.bridge.smooth_k = args.smooth_k
    cfg.bridge.smooth_iters = args.smooth_iters
    cfg.bridge.symmetric_min_fraction = args.symmetric_min_fraction
    cfg.embedding.method = args.embedding
    cfg.embedding.n_components = args.embed_dim
    cfg.embedding.fm_gene_embedding_path = args.fm_gene_embedding
    cfg.transport.epsilon = args.ot_epsilon
    cfg.transport.embed_weight = args.ot_embed_weight
    cfg.transport.max_ot_cells = args.max_ot_cells
    cfg.transport.deshrink = not args.no_deshrink
    cfg.density.enabled = args.density
    cfg.density.strength = args.density_strength
    cfg.density.bandwidth_spacings = args.density_bandwidth
    cfg.communication.enabled = not args.no_communication
    cfg.communication.n_sweeps = args.niche_sweeps
    cfg.communication.niche_weight = args.niche_weight
    cfg.communication.k_neighbors = args.niche_k
    cfg.annotation.enabled = not args.no_annotation
    cfg.annotation.classifier = args.classifier
    cfg.annotation.anchor_weight = args.anchor_weight
    cfg.annotation.constrain_composition = not args.no_composition_constraint
    cfg.synthesis.expression_mode = args.expression_mode
    cfg.synthesis.transfer_alpha = args.transfer_alpha
    cfg.synthesis.count_mode = args.count_mode

    print(f"  placement={cfg.bridge.mode}, embedding={cfg.embedding.method}, "
          f"density={'on' if cfg.density.enabled else 'off'} (strength={cfg.density.strength}), "
          f"classifier={cfg.annotation.classifier}, niche={'on' if cfg.communication.enabled else 'off'} "
          f"(sweeps={cfg.communication.n_sweeps}), expr={cfg.synthesis.expression_mode}")

    gen = SpatialCPAv8(stack, gene_names=gene_names,
                       cell_type_names=cell_type_names, cfg=cfg)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: synthesizing virtual slice at z={z:.2f} "
              f"(placement={cfg.bridge.mode})...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        if getattr(gen, "_cv_scores", None):
            print(f"    CV placement selection: chose {gen._last_placement} "
                  f"from {{" + ", ".join(f'{k}:{v:.3f}' for k, v in gen._cv_scores.items()) + "}")
        elif getattr(gen, "_last_dissimilarity", None) is not None:
            print(f"    flanking dissimilarity="
                  f"{gen._last_dissimilarity:.3f} (placement={gen._last_placement})")
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
        description="SpatialCPA-v8 generation-only wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    # Placement regime.
    parser.add_argument("--placement", default="smooth_morph",
                        choices=["smooth_morph", "adaptive", "coherent_mix", "symmetric",
                                 "morph", "interpolate", "backbone"],
                        help="smooth_morph (default — coherent smoothed-OT deformation "
                             "of the single nearest clean slice: inherits a single "
                             "slice's structure/density fidelity while its non-rigid "
                             "warp adds the interpolated field the copy lacks), "
                             "adaptive (pick the placement per dataset by leakage-safe "
                             "internal cross-validation on a held-out training slice; "
                             "logs the CV scores), coherent_mix (both-slice expression "
                             "on one coherent density manifold), symmetric "
                             "(bidirectional McCann barycentric bridge), morph "
                             "(un-smoothed barycentric morph; ablation), interpolate "
                             "(random both-slice mixing; ablation), or backbone "
                             "(single nearest slice; SpatialZ-like)")
    parser.add_argument("--adaptive-threshold", type=float, default=0.85,
                        help="displacement (cell-spacings) above which adaptive "
                             "switches from smooth morph to symmetric mixing")
    parser.add_argument("--smooth-k", type=int, default=12,
                        help="kNN degree for smoothing the OT displacement field")
    parser.add_argument("--smooth-iters", type=int, default=3,
                        help="iterations of displacement-field smoothing (morph coherence)")
    parser.add_argument("--symmetric-min-fraction", type=float, default=0.05,
                        help="minimum fraction drawn from the minority flanking slice")
    # Embedding / foundation-model prior.
    parser.add_argument("--embedding", default="pca",
                        choices=["pca", "coexpr", "fm_gene", "concat"],
                        help="cell-state embedding: pca (local, default), coexpr "
                             "(data-derived gene-program prior), fm_gene (pretrained "
                             "gene embedding, needs --fm-gene-embedding), or concat")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--fm-gene-embedding", default=None,
                        help="path to a pretrained gene-embedding matrix "
                             "(.npz genes/embedding or panel-aligned .npy)")
    # Optimal transport (placement).
    parser.add_argument("--ot-epsilon", type=float, default=0.05,
                        help="Sinkhorn entropic regularization (smaller = peaked plan)")
    parser.add_argument("--ot-embed-weight", type=float, default=0.15,
                        help="OT cost blend: (1-w)*spatial + w*embedding")
    parser.add_argument("--max-ot-cells", type=int, default=1500,
                        help="subsample each flanking slice to this for the OT solve")
    parser.add_argument("--no-deshrink", action="store_true",
                        help="disable footprint (covariance) matching after the bridge")
    parser.add_argument("--count-mode", default="interpolate",
                        choices=["interpolate", "lower", "upper", "mean"])
    # Density calibration (opt-in; off by default — see DensityConfig).
    parser.add_argument("--density", action="store_true",
                        help="enable density calibration to the interpolated field "
                             "(off by default; helps only strongly non-stationary density)")
    parser.add_argument("--density-strength", type=float, default=0.7,
                        help="0 = off, 1 = fully re-weight synthesized cells to the target field")
    parser.add_argument("--density-bandwidth", type=float, default=2.0,
                        help="KDE bandwidth in units of median cell spacing")
    # Cell-cell communication (niche) refinement.
    parser.add_argument("--no-communication", action="store_true",
                        help="disable the niche MRF label refinement (ablation)")
    parser.add_argument("--niche-sweeps", type=int, default=8)
    parser.add_argument("--niche-weight", type=float, default=1.0)
    parser.add_argument("--niche-k", type=int, default=10)
    # Annotation.
    parser.add_argument("--no-annotation", action="store_true",
                        help="keep the copied real source labels (ablation)")
    parser.add_argument("--classifier", default="spatial",
                        choices=["spatial", "prototype", "knn"],
                        help="spatial (interpolate the type field from both slices, "
                             "default), or a foundation-model embedding classifier "
                             "(prototype / knn)")
    parser.add_argument("--anchor-weight", type=float, default=3.0,
                        help="weight of the copied-type anchor (higher = more conservative)")
    parser.add_argument("--no-composition-constraint", action="store_true",
                        help="do not pin composition to the interpolated flanking mix")
    # Expression.
    parser.add_argument("--expression-mode", default="endpoint",
                        choices=["endpoint", "transfer", "blend"],
                        help="endpoint (copy real profile; max variance, default), "
                             "transfer (nearest same-type training cell), or blend")
    parser.add_argument("--transfer-alpha", type=float, default=0.5,
                        help="blend weight on the transferred profile (blend mode)")
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

    print(f"Running SpatialCPA-v8 (generation-only) for targets "
          f"{[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "placement": args.placement,
        "symmetric_min_fraction": args.symmetric_min_fraction,
        "embedding": args.embedding, "embed_dim": args.embed_dim,
        "fm_gene_embedding": args.fm_gene_embedding,
        "ot_epsilon": args.ot_epsilon, "ot_embed_weight": args.ot_embed_weight,
        "max_ot_cells": args.max_ot_cells, "deshrink": not args.no_deshrink,
        "count_mode": args.count_mode,
        "adaptive_threshold": args.adaptive_threshold,
        "smooth_k": args.smooth_k, "smooth_iters": args.smooth_iters,
        "density": args.density, "density_strength": args.density_strength,
        "density_bandwidth": args.density_bandwidth,
        "annotation": not args.no_annotation,
        "communication": not args.no_communication, "niche_sweeps": args.niche_sweeps,
        "niche_weight": args.niche_weight, "niche_k": args.niche_k,
        "classifier": args.classifier, "anchor_weight": args.anchor_weight,
        "composition_constraint": not args.no_composition_constraint,
        "expression_mode": args.expression_mode, "transfer_alpha": args.transfer_alpha,
        "training_free": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav8_gen")


if __name__ == "__main__":
    main()
