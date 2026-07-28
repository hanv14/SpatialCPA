"""SpatialCPA-v15 — benchmark-pbya-v2 generation-only wrapper.

v15 implements ``spatialcpa_v15_idea.md`` end to end: a continuous hierarchical
**type embedding** built from signed marker effect sizes (Phase 1.2), a
multi-channel **structure field** completed by a flow-free, query-based
conditional diffusion interpolator with a deterministic DDIM multi-query and a
machine-checked stop condition (Phase 2), an **NB expression VAE** used as a
likelihood plus a **conditional latent diffusion** with neighbourhood
cross-attention over the real bracketing slices (Phase 3), and inference by
**marked spatial point process** layout sampling with mandatory per-cell
provenance (Phase 4).

It shares **no** methodology with SpatialCPA-v8: there is no entropic-OT plan, no
barycentric/McCann bridge, no smoothed-displacement or velocity-field morph of an
anchor slice, no OT expression fusion and no niche MRF.  v15's estimand is the
completed density field, from which cells are *sampled*, not a transported copy
of a real slice.

Every default in ``SpatialCPAv15Config`` is the intended production setting, so
the harness's shared generation-only arguments alone reproduce the proposed
pipeline; the extra CLI flags below are for ablations and tuning.

Leakage safeguards are shared with the other v2 wrappers: the input file excludes
the held-out section (``guard_no_holdout`` re-checks it), the label vocabulary is
built from the training input only, expression normalization is per-cell, and the
only thing tying the output to the held-out slice is the scalar target z.

Runs in ``bench_spatialcpa`` (PyTorch >= 2.0).  The ``spatialcpav15`` package is
imported from the repository root; set SPATIALCPAV15_ROOT to override discovery.
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


def _add_root():
    cands = ([Path(os.environ["SPATIALCPAV15_ROOT"])]
             if os.environ.get("SPATIALCPAV15_ROOT") else [])
    cands += list(Path(__file__).resolve().parents)
    for cand in cands:
        if (cand / "spatialcpav15" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav15` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav15
        import torch
        # Touch the public entry points so a broken package fails here, loudly,
        # rather than half-way through a benchmark run.
        assert spatialcpav15.SpatialCPAv15 and spatialcpav15.SpatialCPAv15Config
        print(f"spatialcpav15 v{spatialcpav15.__version__} from {_ROOT}; "
              f"torch {torch.__version__} "
              f"({'cuda' if torch.cuda.is_available() else 'cpu'})")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav15 (needs torch>=2.0): {e}",
              file=sys.stderr)
        return False


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _to_dense_f32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _prepare_expression(adata):
    """Return (log-normalized matrix, raw count matrix or None).

    Phase 3.1 wants a *count* likelihood, so the raw matrix is kept alongside the
    log-normalized one whenever the dataset actually ships counts.  Normalization
    is per-cell, so no statistic is pooled across cells (leakage-safe).
    """
    import scanpy as sc

    et = adata.uns.get("expression_type", "raw_counts")
    # ``.copy()`` is load-bearing: ``_to_dense_f32`` returns the *same* buffer for
    # an already-dense float32 ``X``, and scanpy normalizes in place — without the
    # copy the "raw counts" become the log-normalized matrix and Phase 3.1
    # silently falls back from the NB likelihood to Gaussian.
    raw = _to_dense_f32(adata.X).copy() if et == "raw_counts" else None
    if et == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif et in ("log1p_normalized", "log2_normalized", "normalized"):
        pass
    elif et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return _to_dense_f32(adata.X), raw, et


def build_stack(adata, X_log, X_raw, ct_all):
    from spatialcpav15 import build_stack_from_arrays

    return build_stack_from_arrays(
        expression=X_log,
        coords_xyz=np.asarray(adata.obsm["spatial"], dtype=np.float64),
        sections=adata.obs["section"].values.astype(str),
        cell_type_indices=ct_all,
        counts=X_raw,
    )


# ── method execution ─────────────────────────────────────────────────────────

def run_method(adata, X_log, X_raw, targets, gene_names, args):
    from spatialcpav15 import SpatialCPAv15, SpatialCPAv15Config

    train_mask = np.ones(adata.n_obs, dtype=bool)   # the whole input is training
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_cells < 8 or stack.n_slices < 2:
        print(f"  SKIP: need >=2 sections and >=8 cells "
              f"(have {stack.n_slices} sections, {stack.n_cells} cells)")
        return {}

    cfg = SpatialCPAv15Config()
    cfg.seed = args.seed
    cfg.types.n_components = args.type_embed_dim
    cfg.types.top_k_markers = args.top_k_markers
    cfg.types.fm_gene_embedding_path = args.fm_gene_embedding
    cfg.types.shrinkage_kappa = args.shrinkage_kappa
    cfg.field_.bin_spacings = args.bin_spacings
    cfg.field_.kde_bandwidth_bins = args.kde_bandwidth
    cfg.field_.max_group_channels = args.max_group_channels
    cfg.field_.max_grid = args.max_grid
    cfg.field_.min_grid = min(args.min_grid, args.max_grid)
    cfg.layout.softmax_tau = args.softmax_tau
    cfg.layout.hardcore_beta = args.hardcore_beta
    cfg.layout.mark_sharpness = args.mark_sharpness
    cfg.structure.enabled = not args.no_structure_model
    cfg.structure.epochs = args.structure_epochs
    cfg.structure.ddim_steps = args.ddim_steps
    cfg.vae.epochs = args.vae_epochs
    cfg.vae.latent_dim = args.latent_dim
    cfg.vae.min_passes = args.vae_min_passes
    cfg.diffusion.enabled = not args.no_expression_diffusion
    cfg.diffusion.epochs = args.expr_epochs
    cfg.diffusion.ddim_steps = args.ddim_steps
    cfg.diffusion.n_context = args.n_context
    cfg.diffusion.decoder_readout = args.readout
    cfg.runtime.device = args.device
    cfg.runtime.cuda_index = args.cuda_index
    cfg.runtime.memory_fraction = args.gpu_memory_fraction
    cfg.runtime.expandable_segments = not args.no_expandable_segments
    cfg.runtime.release_cache_between_stages = not args.no_gpu_cache_release
    cfg.inference.n_structure_seeds = args.structure_seeds
    cfg.inference.n_expression_samples = args.expression_samples
    cfg.inference.output_counts = not args.no_output_counts

    print(f"  config: type_dim={cfg.types.n_components}, "
          f"structure(epochs={cfg.structure.epochs}, seeds="
          f"{cfg.inference.n_structure_seeds}), "
          f"vae(latent={cfg.vae.latent_dim}, epochs={cfg.vae.epochs}), "
          f"expr_diffusion(min_steps={cfg.diffusion.epochs}, "
          f"min_passes={cfg.diffusion.min_passes}, "
          f"ctx={cfg.diffusion.n_context}), readout={cfg.diffusion.decoder_readout}, "
          f"spatial_library={cfg.inference.spatial_library})")

    gen = SpatialCPAv15(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: completing structure + generating expression at z={z:.2f}...")
        try:
            vs = gen.generate_virtual_slice(z=float(z))
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        n = vs.n_cells
        if n == 0:
            print(f"    WARNING: 0 cells synthesized for {sec}")
            continue
        d = vs.diagnostics
        print(f"    -> {n} cells | dz_to_real={d['dist_to_nearest_real_slice']:.2f} "
              f"| struct_spread={d['mean_structural_spread']:.3f} "
              f"| sub_nyquist={d['sub_nyquist']}")
        results[sec] = {
            "X": sp.csr_matrix(np.asarray(vs.expression, dtype=np.float32)),
            "coords": vs.coords.astype(np.float64),
            "cell_type": (vs.cell_type.astype(str) if vs.cell_type is not None
                          else np.array(["NA"] * n)),
        }
    results["_diagnostics"] = gen.diagnostics
    return results


def main():
    parser = argparse.ArgumentParser(
        description="SpatialCPA-v15 generation-only wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    # Phase 1.2 — type representation.
    parser.add_argument("--type-embed-dim", type=int, default=16,
                        help="dimensionality d of the continuous type embedding e_c")
    parser.add_argument("--top-k-markers", type=int, default=30,
                        help="markers kept per type (with signed effect size)")
    parser.add_argument("--shrinkage-kappa", type=float, default=50.0,
                        help="kappa in w_c = n_c/(n_c+kappa): how quickly a type "
                             "stops borrowing from the encoder prior")
    parser.add_argument("--fm-gene-embedding", default=None,
                        help="optional pretrained gene embedding (.npz genes/embedding "
                             "or panel-aligned .npy) used as the Phase 1.2.1 encoder")
    # Phase 2 — structure field + completion.
    parser.add_argument("--bin-spacings", type=float, default=2.0,
                        help="field voxel size in units of median cell spacing")
    parser.add_argument("--kde-bandwidth", type=float, default=1.0,
                        help="Gaussian rasterization kernel K_h, in voxels")
    parser.add_argument("--max-group-channels", type=int, default=48,
                        help="use one density channel per type while the type count "
                             "is <= this; above it the channels collapse to the "
                             "Phase 1.3 hierarchy and within-group identity has to "
                             "be recovered from the blurred embedding field")
    parser.add_argument("--max-grid", type=int, default=56,
                        help="cap on the in-plane field resolution. Dense sections "
                             "with fine structure hit this cap, and the resulting "
                             "blur costs the spatial-organization metrics; raise it "
                             "for large ROIs (cost is quadratic in the U-Net)")
    parser.add_argument("--min-grid", type=int, default=24,
                        help="floor on the in-plane field resolution")
    parser.add_argument("--softmax-tau", type=float, default=0.35,
                        help="temperature of the within-group type posterior")
    parser.add_argument("--hardcore-beta", type=float, default=0.55,
                        help="hardcore radius as a fraction of median NN distance")
    parser.add_argument("--mark-sharpness", type=float, default=1.0,
                        help="exponent on the type posterior before drawing marks")
    parser.add_argument("--structure-epochs", type=int, default=220)
    parser.add_argument("--structure-seeds", type=int, default=8,
                        help="deterministic completions; their voxelwise spread is "
                             "the structural uncertainty map")
    parser.add_argument("--no-structure-model", action="store_true",
                        help="ablation: skip the learned completer (linear field)")
    # Phase 3 — expression.
    parser.add_argument("--vae-epochs", type=int, default=260)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--vae-min-passes", type=int, default=15,
                        help="minimum passes over the cells for the Phase 3.1 VAE")
    parser.add_argument("--expr-epochs", type=int, default=260)
    parser.add_argument("--n-context", type=int, default=16,
                        help="context cells cross-attended per target cell")
    parser.add_argument("--expression-samples", type=int, default=1)
    parser.add_argument("--readout", default="auto",
                        choices=["auto", "mean", "sample"],
                        help="decoder readout; 'auto' lets the Phase 3.3 gate choose")
    parser.add_argument("--no-expression-diffusion", action="store_true",
                        help="ablation: skip the latent diffusion")
    # Shared.
    parser.add_argument("--ddim-steps", type=int, default=20)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="where the three trained models run; 'auto' uses the "
                             "GPU when one is available")
    parser.add_argument("--cuda-index", type=int, default=0,
                        help="which GPU to use when several are visible")
    parser.add_argument("--gpu-memory-fraction", type=float, default=None,
                        help="hard cap on this process's share of the GPU (e.g. 0.5) "
                             "so a co-tenant is never crowded out; default uncapped")
    parser.add_argument("--no-expandable-segments", action="store_true",
                        help="do NOT ask the CUDA allocator for growable segments "
                             "(not recommended on a shared GPU)")
    parser.add_argument("--no-gpu-cache-release", action="store_true",
                        help="do NOT return cached GPU blocks between training "
                             "stages (not recommended on a shared GPU)")
    parser.add_argument("--no-output-counts", action="store_true",
                        help="emit log1p-normalized expression instead of counts")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]

    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)   # defense in depth
    gene_names = adata.var_names.tolist()
    X_log, X_raw, expr_type = _prepare_expression(adata)
    print(f"  expression_type={expr_type}, counts available={X_raw is not None}")

    print(f"Running SpatialCPA-v15 for targets "
          f"{[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, X_log, X_raw, targets, gene_names, args)
    wall_time = time.time() - t0
    diagnostics = results.pop("_diagnostics", {})

    method_params = {
        "seed": args.seed,
        "type_embed_dim": args.type_embed_dim,
        "top_k_markers": args.top_k_markers,
        "shrinkage_kappa": args.shrinkage_kappa,
        "structure_epochs": args.structure_epochs,
        "structure_seeds": args.structure_seeds,
        "vae_epochs": args.vae_epochs,
        "latent_dim": args.latent_dim,
        "expr_epochs": args.expr_epochs,
        "n_context": args.n_context,
        "ddim_steps": args.ddim_steps,
        "readout": args.readout,
        "device": diagnostics.get("runtime", {}).get("device"),
        "gpu_peak_reserved_gib": diagnostics.get("runtime", {}).get("peak_reserved_gib"),
        "structure_learned": bool(
            diagnostics.get("phase2_structure", {}).get("passed", False)),
        "expression_diffusion": bool(
            diagnostics.get("phase3_diffusion", {}).get("trained", False)),
        "phase3_gate": diagnostics.get("phase3_gate", {}).get("beats_type_mean"),
        "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav15_gen")


if __name__ == "__main__":
    main()
