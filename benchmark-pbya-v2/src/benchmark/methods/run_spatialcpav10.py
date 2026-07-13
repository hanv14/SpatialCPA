"""SpatialCPA-v10 (biologically-constrained) — benchmark-pbya-v2 wrapper.

v10 keeps v8's diffeomorphic single-slice *placement* but **generates** each cell's
expression from an explicit biological model instead of copying: a z-continuous
cell-type gene program, modulated by the ligand-receptor signaling of its spatial
neighbours, plus a real residual so gene-gene structure stays realistic (a balanced
hybrid). Cell-type annotation is the organizing first step and the niche MRF enforces
which types co-localize. Training-free (numpy/scipy/sklearn); same ``bench_spatialcpa``
env. See ``spatialcpav10/README.md``.

Leakage safeguards match the other v2 wrappers.
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
    for cand in ([Path(os.environ["SPATIALCPAV10_ROOT"])] if os.environ.get("SPATIALCPAV10_ROOT") else []) \
            + list(Path(__file__).resolve().parents):
        if (cand / "spatialcpav10" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav10` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav10
        from spatialcpav10 import SpatialCPAv10, SpatialCPAv10Config  # noqa: F401
        print(f"spatialcpav10 v{getattr(spatialcpav10, '__version__', '?')} imported from {_ROOT}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav10: {e}", file=sys.stderr)
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
    from spatialcpav10 import Slice, SliceStack
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
    from spatialcpav10 import SpatialCPAv10, SpatialCPAv10Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}

    cfg = SpatialCPAv10Config()
    cfg.seed = cfg.synthesis.seed = args.seed
    cfg.bridge.mode = "diffeo_morph"
    cfg.biology.enabled = not args.no_biology
    cfg.biology.residual_weight = args.residual_weight
    cfg.biology.program_weight = args.program_weight
    cfg.biology.lr_lambda = args.lr_lambda
    cfg.biology.lr_source = args.lr_source
    cfg.biology.lr_db_path = args.lr_db
    cfg.biology.z_continuity = not args.no_z_continuity
    cfg.annotation.enabled = not args.no_annotation
    cfg.communication.enabled = not args.no_communication

    print(f"  biology={'on' if cfg.biology.enabled else 'off'} "
          f"(residual_w={cfg.biology.residual_weight}, program_w={cfg.biology.program_weight}, "
          f"lr_lambda={cfg.biology.lr_lambda}, lr_source={cfg.biology.lr_source}, "
          f"z_continuity={cfg.biology.z_continuity})")

    gen = SpatialCPAv10(stack, gene_names=gene_names, cell_type_names=cell_type_names, cfg=cfg)

    results = {}
    for sec, z in targets:
        print(f"  {sec}: synthesizing biologically-constrained slice at z={z:.2f}...")
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
    parser = argparse.ArgumentParser(description="SpatialCPA-v10 wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--residual-weight", type=float, default=1.0,
                        help="weight of the real residual (1=max realism/scores, 0=fully mechanistic)")
    parser.add_argument("--program-weight", type=float, default=1.0,
                        help="weight of the cell-type program mean")
    parser.add_argument("--lr-lambda", type=float, default=0.2,
                        help="strength of ligand-receptor expression modulation")
    parser.add_argument("--lr-source", default="auto", choices=["auto", "db", "infer", "off"],
                        help="LR coupling source: auto (DB else inferred), db, infer, or off")
    parser.add_argument("--lr-db", default=None, help="curated LR database (.npz/.tsv)")
    parser.add_argument("--no-z-continuity", action="store_true",
                        help="disable z-interpolation of cell-type programs")
    parser.add_argument("--no-biology", action="store_true",
                        help="disable the biological expression model (copy real profiles)")
    parser.add_argument("--no-annotation", action="store_true")
    parser.add_argument("--no-communication", action="store_true")
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

    print(f"Running SpatialCPA-v10 for targets {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "biology": not args.no_biology,
        "residual_weight": args.residual_weight, "program_weight": args.program_weight,
        "lr_lambda": args.lr_lambda, "lr_source": args.lr_source,
        "z_continuity": not args.no_z_continuity, "annotation": not args.no_annotation,
        "communication": not args.no_communication, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav10_gen")


if __name__ == "__main__":
    main()
