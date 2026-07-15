"""SpatialCPA-v12 (generative continuous 3D neural field) — benchmark-pbya-v2 wrapper.

v12 enhances v11's continuous implicit-field model. Stage-1 ``LayoutField`` (positions +
cell type) is unchanged (distilled from a frozen OmiCLIP / Path2Space teacher, proxy
stand-in when weights absent, plus self-supervised slice reconstruction). Stage-2 is a
**generative factor-analysis expression decoder**: it outputs a per-cell mean and holds
low-rank loadings fit to the real gene-gene covariance, so sampled cells reproduce
realistic covariance/variance instead of collapsing to the mean. Inference draws each
cell with a spatially-coherent latent and calibrates the density, per-gene statistics,
and cell-type composition to the z-interpolated flanking fields (leakage-safe). Runs in
``bench_spatialcpa`` (PyTorch >= 2.0); falls back to a nearest-slice layout if torch is
unavailable. See ``spatialcpav12/README.md``.
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
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
import _v2_io  # noqa: E402
import leakage_guard  # noqa: E402


def _add_root():
    for cand in ([Path(os.environ["SPATIALCPAV12_ROOT"])] if os.environ.get("SPATIALCPAV12_ROOT") else []) \
            + list(Path(__file__).resolve().parents):
        if (cand / "spatialcpav12" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav12` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav12
        from spatialcpav12 import SpatialCPAv12, SpatialCPAv12Config  # noqa: F401
        try:
            import torch
            tinfo = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
        except Exception:
            tinfo = "torch UNAVAILABLE -> nearest-slice fallback"
        print(f"spatialcpav12 v{getattr(spatialcpav12, '__version__', '?')} from {_ROOT}; {tinfo}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav12: {e}", file=sys.stderr)
        return False


def _to_dense_f32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _normalize_expression(adata):
    import scanpy as sc
    et = adata.uns.get("expression_type", "raw_counts")
    if et == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    elif et in ("log1p_normalized", "log2_normalized", "normalized"):
        pass
    elif et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    return et


def _gene_symbols(adata):
    """Return gene SYMBOLS aligned to var order from a common .var column, else None."""
    for col in ("gene_symbol", "gene_symbols", "symbol", "SYMBOL", "Symbol",
                "feature_name", "gene_name", "gene_names", "GeneSymbol"):
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).tolist()
            if any(v and v.lower() != "nan" for v in vals):
                print(f"  using gene symbols from adata.var['{col}'] for the teacher")
                return vals
    return None


def build_stack(adata, ct_all):
    from spatialcpav12 import Slice, SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    X = _to_dense_f32(adata.X)
    labels = sorted(np.unique(sections), key=lambda s: np.median(coords[sections == s, 2]))
    slices = [Slice(expression=X[sections == sec], coords_xy=coords[sections == sec, :2],
                    z_values=coords[sections == sec, 2],
                    cell_type_indices=ct_all[sections == sec] if ct_all is not None else None,
                    section_id=str(sec)) for sec in labels]
    return SliceStack(slices)


def run_method(adata, targets, gene_names, args):
    from spatialcpav12 import SpatialCPAv12, SpatialCPAv12Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}

    cfg = SpatialCPAv12Config()
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.device = args.device
    cfg.teacher.kind = args.teacher
    cfg.teacher.weights_path = args.teacher_weights
    cfg.teacher.gene_embedding_path = args.gene_embedding
    cfg.teacher.model_arch = args.teacher_arch
    cfg.teacher.top_genes = args.teacher_top_genes
    cfg.teacher.symbol_map_path = args.teacher_symbol_map
    cfg.expression.n_factors = args.n_factors
    cfg.inference.expr_decode = args.expr_decode
    cfg.inference.latent_coherence = args.latent_coherence
    cfg.inference.anchor_weight = args.anchor_weight
    cfg.inference.noise_scale = args.noise_scale
    cfg.inference.position_mode = args.position_mode
    cfg.inference.residual_weight = args.residual_weight
    cfg.inference.z_marginalize = args.z_marginalize
    cfg.inference.output_counts = not args.no_output_counts
    cfg.inference.calibrate_gene_stats = args.calibrate_gene_stats
    cfg.inference.calibrate_density = args.calibrate_density
    cfg.inference.density_blend = args.density_blend
    cfg.inference.composition_calibrate = not args.no_calibrate_composition

    print(f"  epochs={cfg.train.epochs}, teacher={cfg.teacher.kind}, "
          f"expr_decode={cfg.inference.expr_decode} (factors={cfg.expression.n_factors}, "
          f"coherence={cfg.inference.latent_coherence}, anchor_w={cfg.inference.anchor_weight}), "
          f"z_marginalize={cfg.inference.z_marginalize}, "
          f"calib[gene={cfg.inference.calibrate_gene_stats},dens={cfg.inference.calibrate_density}"
          f"(blend={cfg.inference.density_blend}),comp={cfg.inference.composition_calibrate}]")

    gene_symbols = args._gene_symbols
    gen = SpatialCPAv12(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                        cfg=cfg, gene_symbols=gene_symbols)
    print(f"  neural fields trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: querying continuous field at z={z:.2f}...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        cell_type = vs.cell_type.astype(str) if vs.cell_type is not None else np.array(["NA"] * n)
        results[sec] = {"X": sp.csr_matrix(_to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64), "cell_type": cell_type}
    return results


def main():
    parser = argparse.ArgumentParser(description="SpatialCPA-v12 wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--teacher", default="auto",
                        choices=["auto", "omiclip", "path2space", "gene_embedding", "proxy"],
                        help="foundation-model teacher (same options as v11)")
    parser.add_argument("--teacher-weights", default=None,
                        help="OmiCLIP checkpoint (open_clip pretrained) for --teacher omiclip")
    parser.add_argument("--gene-embedding", default=None,
                        help="pretrained gene-embedding matrix for --teacher path2space/gene_embedding")
    parser.add_argument("--teacher-arch", default="coca_ViT-L-14",
                        help="open_clip architecture for the OmiCLIP text tower")
    parser.add_argument("--teacher-top-genes", type=int, default=50,
                        help="genes per spot in the OmiCLIP gene-sentence")
    parser.add_argument("--teacher-symbol-map", default=None,
                        help="id->symbol map to translate an Ensembl-ID panel for the OmiCLIP teacher")
    parser.add_argument("--n-factors", type=int, default=24,
                        help="rank of the generative factor-analysis loading matrix")
    parser.add_argument("--expr-decode", default="generative",
                        choices=["generative", "field", "residual"],
                        help="generative (mean + coherent factor noise; default) | field (mean only) | residual (v11 blend)")
    parser.add_argument("--latent-coherence", type=float, default=0.9,
                        help="fraction of the factor code drawn as a spatially-coherent field")
    parser.add_argument("--anchor-weight", type=float, default=0.9,
                        help="weight of the real same-type mean anchor (0 = pure field mean)")
    parser.add_argument("--noise-scale", type=float, default=0.2,
                        help="scale of the additive factor-analysis noise (0 = deterministic mean)")
    parser.add_argument("--position-mode", default="auto",
                        choices=["auto", "field", "flanking", "hybrid"],
                        help="auto (regime-adaptive; default) | field (learned grid) | flanking | hybrid")
    parser.add_argument("--residual-weight", type=float, default=0.7,
                        help="weight of the real profile when --expr-decode residual")
    parser.add_argument("--z-marginalize", type=int, default=3,
                        help="samples in the z window for hybrid inference (1=point query)")
    parser.add_argument("--density-blend", type=float, default=0.5,
                        help="blend of learned occupancy vs z-interpolated flanking density")
    parser.add_argument("--no-output-counts", action="store_true",
                        help="emit log1p-normalized (not count-like) expression")
    parser.add_argument("--calibrate-gene-stats", action="store_true",
                        help="pin per-gene mean/variance to the interpolated target (off by default)")
    parser.add_argument("--calibrate-density", action="store_true",
                        help="blend the learned occupancy with the interpolated density (off by default)")
    parser.add_argument("--no-calibrate-composition", action="store_true",
                        help="disable cell-type composition prior-correction")
    args = parser.parse_args()

    if not check_environment():
        sys.exit(1)

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input}...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = adata.var_names.tolist()
    args._gene_symbols = _gene_symbols(adata)
    _normalize_expression(adata)

    print(f"Running SpatialCPA-v12 for targets {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "epochs": args.epochs, "teacher": args.teacher,
        "n_factors": args.n_factors, "expr_decode": args.expr_decode,
        "latent_coherence": args.latent_coherence, "anchor_weight": args.anchor_weight,
        "z_marginalize": args.z_marginalize, "learned": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav12_gen")


if __name__ == "__main__":
    main()
