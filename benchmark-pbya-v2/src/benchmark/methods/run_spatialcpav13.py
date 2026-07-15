"""SpatialCPA-v13 (cell-sentence transformer LM) — benchmark-pbya-v2 wrapper.

v13 is an LLM-based generator: each cell is tokenized into a rank-ordered gene-token
"cell-sentence" and a self-attention transformer is trained with a masked
gene-language-model objective + a spatial in-context objective. Virtual slices are
produced by retrieval-augmented in-context generation (cross-attend over retrieved
flanking cells -> sample a real exemplar from the LM-similarity distribution -> emit a
grounded, generated profile). Runs in ``bench_spatialcpa`` (PyTorch >= 2.0); falls back
to a nearest-slice layout if torch is unavailable. See ``spatialcpav13/README.md``.
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
    for cand in ([Path(os.environ["SPATIALCPAV13_ROOT"])] if os.environ.get("SPATIALCPAV13_ROOT") else []) \
            + list(Path(__file__).resolve().parents):
        if (cand / "spatialcpav13" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand)
    return None


_ROOT = _add_root()


def check_environment():
    if _ROOT is None:
        print("ERROR: could not locate the `spatialcpav13` package.", file=sys.stderr)
        return False
    try:
        import spatialcpav13
        from spatialcpav13 import SpatialCPAv13, SpatialCPAv13Config  # noqa: F401
        try:
            import torch
            tinfo = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
        except Exception:
            tinfo = "torch UNAVAILABLE -> nearest-slice fallback"
        print(f"spatialcpav13 v{getattr(spatialcpav13, '__version__', '?')} from {_ROOT}; {tinfo}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav13: {e}", file=sys.stderr)
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
    from spatialcpav13 import Slice, SliceStack
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
    from spatialcpav13 import SpatialCPAv13, SpatialCPAv13Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}

    cfg = SpatialCPAv13Config()
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.device = args.device
    cfg.tokenizer.top_genes = args.top_genes
    cfg.model.n_layers = args.n_layers
    cfg.model.d_model = args.d_model
    cfg.generation.retrieval_temp = args.retrieval_temp
    cfg.generation.edit_weight = args.edit_weight
    cfg.generation.position_mode = args.position_mode
    cfg.generation.output_counts = not args.no_output_counts
    cfg.generation.composition_match = not args.no_composition_match

    print(f"  epochs={cfg.train.epochs}, transformer(d={cfg.model.d_model},L={cfg.model.n_layers},"
          f"heads={cfg.model.n_heads}), top_genes={cfg.tokenizer.top_genes}, "
          f"retrieval_temp={cfg.generation.retrieval_temp}, edit_w={cfg.generation.edit_weight}, "
          f"pos={cfg.generation.position_mode}, comp_match={cfg.generation.composition_match}")

    gen = SpatialCPAv13(stack, gene_names=gene_names, cell_type_names=cell_type_names, cfg=cfg)
    print(f"  transformer LM trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: generating (retrieval-augmented) at z={z:.2f}...")
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
    parser = argparse.ArgumentParser(description="SpatialCPA-v13 wrapper (benchmark-pbya-v2)")
    _v2_io.add_v2_args(parser)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--top-genes", type=int, default=32,
                        help="genes per cell-sentence (rank-ordered)")
    parser.add_argument("--n-layers", type=int, default=4, help="transformer layers")
    parser.add_argument("--d-model", type=int, default=128, help="transformer width")
    parser.add_argument("--retrieval-temp", type=float, default=0.2,
                        help="temperature for sampling the real exemplar (RAG)")
    parser.add_argument("--edit-weight", type=float, default=0.15,
                        help="blend toward the LM-decoded profile (0 = pure exemplar)")
    parser.add_argument("--position-mode", default="flanking", choices=["flanking", "nearest", "auto"],
                        help="retrieval layout: flanking (default) | nearest | auto")
    parser.add_argument("--no-output-counts", action="store_true",
                        help="emit log1p-normalized (not count-like) expression")
    parser.add_argument("--no-composition-match", action="store_true",
                        help="disable cell-type composition matching")
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

    print(f"Running SpatialCPA-v13 for targets {[(s, round(z, 2)) for s, z in targets]}...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, args)
    wall_time = time.time() - t0

    method_params = {
        "seed": args.seed, "epochs": args.epochs, "d_model": args.d_model,
        "n_layers": args.n_layers, "top_genes": args.top_genes,
        "retrieval_temp": args.retrieval_temp, "llm_based": True, "generation_only": True,
    }
    _v2_io.write_prediction_h5(results, gene_names, target_sections,
                               method_params, wall_time, args.output,
                               method_name="spatialcpav13_gen")


if __name__ == "__main__":
    main()
