"""SpatialCPA-v16 (Spectral Tissue Flow) — benchmark wrapper.

This wrapper lives in ``benchmark-pbya-v3`` rather than beside the others in
``benchmark-pbya-v2/src/benchmark/methods/``, because v2 is frozen. It speaks the
same ``_v2_io`` contract as every other wrapper — same CLI, same
``prediction.h5`` format, same leakage guard — so ``run_benchmark`` invokes it
identically and the evaluator reads it unchanged.

Generation-only: the input file physically excludes the held-out sections and the
method receives one scalar target z per section. Nothing else.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# v2's shared I/O helpers. Imported, never modified — the contract is v2's and
# reusing it is what keeps v3's rows comparable with the rest of the benchmark.
_V2_BENCH = (Path(__file__).resolve().parents[4]
             / "benchmark-pbya-v2" / "src" / "benchmark")
sys.path.insert(0, str(_V2_BENCH / "methods"))   # _v2_io
sys.path.insert(0, str(_V2_BENCH))               # leakage_guard
import _v2_io                      # noqa: E402
import leakage_guard               # noqa: E402


ROOT_ENV = "SPATIALCPAV16_ROOT"


def _candidate_roots():
    """Directories that might *contain* the ``spatialcpav16`` package.

    ``$SPATIALCPAV16_ROOT`` wins, matching the convention v14's wrapper uses. It
    names the directory the package sits **in**, not the package itself — but
    pointing it straight at ``.../spatialcpav16`` is the obvious mistake, so that
    spelling is accepted too rather than failing with the package one level away.
    """
    env = os.environ.get(ROOT_ENV)
    if env:
        p = Path(env).expanduser()
        yield p
        # ``$SPATIALCPAV16_ROOT=/path/to/spatialcpav16`` -> try its parent.
        if p.name == "spatialcpav16":
            yield p.parent
    yield from Path(__file__).resolve().parents


def _add_root():
    tried = []
    for cand in _candidate_roots():
        tried.append(str(cand))
        if (cand / "spatialcpav16" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return str(cand), tried
    return None, tried


_ROOT, _TRIED = _add_root()


def check_environment():
    if _ROOT is None:
        env = os.environ.get(ROOT_ENV)
        print("ERROR: could not locate the `spatialcpav16` package.", file=sys.stderr)
        print(f"  Looked for a directory containing spatialcpav16/__init__.py in:",
              file=sys.stderr)
        for t in _TRIED:
            print(f"    {t}", file=sys.stderr)
        if env:
            print(f"  ${ROOT_ENV} is set to {env!r}; it must name the directory "
                  f"the package sits IN, so {Path(env) / 'spatialcpav16' / '__init__.py'} "
                  f"has to exist.", file=sys.stderr)
        else:
            print(f"  ${ROOT_ENV} is not set. Either set it to the directory "
                  f"holding spatialcpav16/, or keep the package at the repository "
                  f"root beside benchmark-pbya-v3/ where the parent walk finds it.",
                  file=sys.stderr)
        return False
    try:
        import spatialcpav16
        try:
            import torch
            t = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
        except Exception:
            t = "torch UNAVAILABLE -> depth-interpolated spectrum (degraded)"
        print(f"spatialcpav16 v{spatialcpav16.__version__} from {_ROOT}; {t}")
        return True
    except ImportError as e:
        print(f"ERROR: failed to import spatialcpav16: {e}", file=sys.stderr)
        return False


def _dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _normalize(adata):
    import scanpy as sc
    et = str(adata.uns.get("expression_type", "raw_counts"))
    if et in ("log1p_normalized", "log2_normalized", "normalized"):
        return et
    if et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
        return et
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return et


def build_stack(adata, ct_all):
    from spatialcpav16 import Slice, SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    X = _dense(adata.X)
    labels = sorted(np.unique(sections),
                    key=lambda s: float(np.median(coords[sections == s, 2])))
    slices = []
    for sec in labels:
        m = sections == sec
        slices.append(Slice(expression=X[m], coords_xy=coords[m, :2],
                            z=float(np.median(coords[m, 2])),
                            cell_type_indices=(ct_all[m] if ct_all is not None
                                               else None),
                            section_id=str(sec)))
    return SliceStack(slices)


def run_method(adata, targets, gene_names, args):
    from spatialcpav16 import SpatialCPAv16, SpatialCPAv16Config

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, ct_all)
    if stack.n_slices < 2 or sum(s.n_cells for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}

    cfg = SpatialCPAv16Config()
    cfg.seed = args.seed
    cfg.harmonics.n_modes = args.n_modes
    cfg.harmonics.k_neighbors = args.k_neighbors
    cfg.flow.gene_components = args.gene_components
    cfg.flow.epochs = args.epochs
    cfg.flow.ode_steps = args.ode_steps
    cfg.flow.n_ensemble = args.ensemble
    cfg.expression.residual_scale = args.residual_scale
    cfg.expression.calibrate = not args.no_calibration
    cfg.expression.residual_by_type = not args.residual_pooled
    cfg.types.composition_match = not args.no_composition_match
    cfg.runtime.device = args.device
    cfg.runtime.cuda_index = args.cuda_index
    cfg.runtime.memory_fraction = args.gpu_memory_fraction
    cfg.runtime.expandable_segments = not args.no_expandable_segments
    cfg.runtime.release_cache = not args.no_gpu_cache_release

    gen = SpatialCPAv16(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    print(f"  spectral flow trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: spectral synthesis at z={z:.2f} ...")
        try:
            vs = gen.generate_virtual_slice(z=z)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
        n = vs.coords.shape[0]
        if n == 0:
            continue
        d = vs.diagnostics
        print(f"    -> {n} cells; modelled variance "
              f"{d['modelled_variance_fraction']:.2f}; Moran median "
              f"{d['predicted_morans_median_before']:+.3f} -> "
              f"{d['predicted_morans_median_after']:+.3f} after calibration")
        names = (np.array([cell_type_names[i] for i in vs.cell_type])
                 if cell_type_names is not None and vs.cell_type is not None
                 else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(vs.expression.astype(np.float32)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": names}
    # Surfaced into method_params so a result records the device it ran on and
    # what it actually held, not just what was asked for.
    run_method.device = gen.diagnostics.get("device", "cpu")
    run_method.peak_gpu_gib = gen.diagnostics.get("peak_gpu_gib")
    return results


def main():
    p = argparse.ArgumentParser(description="SpatialCPA-v16 wrapper (Spectral Tissue Flow)")
    _v2_io.add_v2_args(p)
    p.add_argument("--n-modes", type=int, default=64,
                   help="tissue harmonics modelled (the structure/dispersion cut)")
    p.add_argument("--k-neighbors", type=int, default=10,
                   help="kNN degree of the tissue graph; match the evaluator's")
    p.add_argument("--gene-components", type=int, default=32)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--ode-steps", type=int, default=16)
    p.add_argument("--ensemble", type=int, default=8)
    p.add_argument("--residual-scale", type=float, default=1.0)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="default 'auto': use a GPU whenever one is visible")
    p.add_argument("--cuda-index", type=int, default=0,
                   help="which GPU on a multi-GPU host")
    p.add_argument("--gpu-memory-fraction", type=float, default=None,
                   help="hard ceiling on this process's share of the card "
                        "(e.g. 0.4). Off by default: a cap set too tight turns "
                        "a slow run into a crashed one")
    p.add_argument("--no-expandable-segments", action="store_true",
                   help="stop asking the CUDA allocator for growable segments "
                        "(not recommended on a shared card — pools then reserve "
                        "fixed blocks they never return)")
    p.add_argument("--no-gpu-cache-release", action="store_true",
                   help="keep cached GPU blocks between stages instead of "
                        "returning them to the driver")
    # Ablations
    p.add_argument("--no-calibration", action="store_true",
                   help="ablate the spectral calibration stage")
    p.add_argument("--residual-pooled", action="store_true",
                   help="draw residuals tissue-wide instead of per cell type")
    p.add_argument("--no-composition-match", action="store_true")
    args = p.parse_args()

    if not check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, [s for s, _ in targets])
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")
    et = _normalize(adata)
    print(f"  expression_type={et}")

    t0 = time.time()
    results = run_method(adata, targets, list(adata.var_names), args)
    wall = time.time() - t0
    gen_device = getattr(run_method, "device", "cpu")
    gen_peak = getattr(run_method, "peak_gpu_gib", None)
    if not results:
        print("No sections synthesized.")
        return 1

    _v2_io.write_prediction_h5(
        results, list(adata.var_names), [s for s, _ in targets],
        {"n_modes": args.n_modes, "k_neighbors": args.k_neighbors,
         "gene_components": args.gene_components, "epochs": args.epochs,
         "ode_steps": args.ode_steps, "ensemble": args.ensemble,
         "residual_scale": args.residual_scale,
         "calibration": not args.no_calibration,
         "device": gen_device, "peak_gpu_gib": gen_peak,
         "gpu_memory_fraction": args.gpu_memory_fraction,
         "seed": args.seed},
        wall, args.output, "spatialcpav16_gen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
