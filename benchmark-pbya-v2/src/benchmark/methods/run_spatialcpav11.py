"""SpatialCPA-v11 (two-stage continuous 3D neural field) — benchmark-pbya-v2 wrapper.

v11 is a continuous implicit-field model: Stage-1 LayoutField (positions + cell type)
and Stage-2 ExpressionField (conditioned on the layout), both queryable at arbitrary
continuous z. Stage 1 is trained by distillation from a frozen multimodal foundation-
model teacher (OmiCLIP / Path2Space; data-derived stand-in when weights absent) plus
self-supervised slice reconstruction; Stage 2 by expression reconstruction; with
cross-z consistency and biology-informed constraints. Runs in ``bench_spatialcpa``
(PyTorch >= 2.0); falls back to a nearest-slice layout if torch is unavailable. See
``spatialcpav11/README.md``.
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
    for cand in ([Path(os.environ["SPATIALCPAV11_ROOT"])] if os.environ.get("SPATIALCPAV11_ROOT") else []) \
            + list(Path(__file__).resolve().parents):
        if (cand / "spatialcpav11" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav11` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav11
        from spatialcpav11 import SpatialCPAv11, SpatialCPAv11Config  # noqa: F401
        try:
            import torch
            tinfo = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
        except Exception:
            tinfo = "torch UNAVAILABLE -> nearest-slice fallback"
        print(f"spatialcpav11 v{getattr(spatialcpav11, '__version__', '?')} from {_ROOT}; {tinfo}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav11: {e}", file=sys.stderr)
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


def build_stack(adata, ct_all):
    from spatialcpav11 import Slice, SliceStack
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
    from spatialcpav11 import SpatialCPAv11, SpatialCPAv11Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}

    cfg = SpatialCPAv11Config()
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.device = args.device
    cfg.teacher.kind = args.teacher
    cfg.teacher.weights_path = args.teacher_weights
    cfg.inference.expr_decode = args.expr_decode
    cfg.inference.residual_weight = args.residual_weight
    cfg.inference.z_marginalize = args.z_marginalize

    print(f"  epochs={cfg.train.epochs}, teacher={cfg.teacher.kind}, "
          f"expr_decode={cfg.inference.expr_decode} (residual_w={cfg.inference.residual_weight}), "
          f"z_marginalize={cfg.inference.z_marginalize}")

    gen = SpatialCPAv11(stack, gene_names=gene_names, cell_type_names=cell_type_names, cfg=cfg)
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
    parser = argparse.ArgumentParser(description="SpatialCPA-v11 wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--teacher", default="auto", choices=["auto", "omiclip", "path2space", "proxy"],
                        help="foundation-model teacher (data-derived proxy if weights absent)")
    parser.add_argument("--teacher-weights", default=None, help="path to OmiCLIP/Path2Space weights")
    parser.add_argument("--expr-decode", default="residual", choices=["residual", "field"],
                        help="residual (layout-conditioned real profile; default) or field (pure Stage-2)")
    parser.add_argument("--residual-weight", type=float, default=0.7)
    parser.add_argument("--z-marginalize", type=int, default=3,
                        help="samples in the z window for hybrid inference (1=point query)")
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

    print(f"Running SpatialCPA-v11 for targets {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "epochs": args.epochs, "teacher": args.teacher,
        "expr_decode": args.expr_decode, "residual_weight": args.residual_weight,
        "z_marginalize": args.z_marginalize, "learned": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav11_gen")


if __name__ == "__main__":
    main()
