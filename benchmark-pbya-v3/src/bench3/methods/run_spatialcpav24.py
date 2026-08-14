"""SpatialCPA-v24 (CTF-Flow: a Continuous Transcriptomic Field) — v3 wrapper.

Like the v16/v18-v23 wrappers, this lives in ``benchmark-pbya-v3`` rather than
beside the others in ``benchmark-pbya-v2/src/benchmark/methods/`` (v2 is frozen),
and speaks the identical ``_v2_io`` contract, so ``run_benchmark`` invokes it like
the rest and the evaluator reads its output unchanged.

v24 is the single-file ``learn_spatialcpav24.py`` at the repository root, which
implements the ``v23_design.md`` "CTF-Flow" design. Its API matches the v14 line
(``Slice`` / ``SliceStack`` / ``VirtualSlice`` / ``build_stack_from_sections`` /
``generate_virtual_slice``) but the model class is ``SpatialCPAv24`` and the config
is ``CTFFlowConfig``.

IMPORTANT — quarantine by design. CTF-Flow's *neural* tier (triplane field, flow
matching with a correlated prior, ZINB decoder, MedCPT text embeddings, metric-
aware LOSO training) is a guarded scaffold in ``learn_spatialcpav24.py`` and is NOT
implemented/validated in this repository state — it needs GPU/torch/MedCPT/data.
So ``SpatialCPAv24`` reports ``trained=False`` and generates with the numpy field
tier (a strict v20 superset). That numpy tier is a legitimate, runnable method, but
it is NOT the CTF-Flow the design claims wins for. Scoring it under the name
``spatialcpav24_gen`` would misrepresent it, so config.py's ``invalid_log_markers``
match ``flow-matching model trained: False`` and the scaffold marker, and the run
is quarantined rather than scored — until the neural tier is implemented and trains.
Implementing ``_train_neural`` / ``_generate_neural`` (and removing the scaffold
guard) is what turns v24 into a scored method.

Generation-only: the input file physically excludes the held-out sections and the
method receives one scalar target z per section.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_V2_BENCH = (Path(__file__).resolve().parents[4]
             / "benchmark-pbya-v2" / "src" / "benchmark")
sys.path.insert(0, str(_V2_BENCH / "methods"))   # _v2_io
sys.path.insert(0, str(_V2_BENCH))               # leakage_guard
import _v2_io                      # noqa: E402
import leakage_guard               # noqa: E402


FILE_ENV = "SPATIALCPAV24_FILE"
ROOT_ENV = "SPATIALCPAV24_ROOT"
_MODULE_NAME = "learn_spatialcpav24"
_FILENAME = "learn_spatialcpav24.py"


def _candidate_files():
    f = os.environ.get(FILE_ENV)
    if f:
        yield Path(f).expanduser()
    r = os.environ.get(ROOT_ENV)
    if r:
        yield Path(r).expanduser() / _FILENAME
    for cand in Path(__file__).resolve().parents:
        yield cand / _FILENAME


def _load_module():
    tried = []
    for cand in _candidate_files():
        tried.append(str(cand))
        if cand.exists():
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, str(cand))
            module = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = module
            spec.loader.exec_module(module)
            return module, str(cand), tried
    return None, None, tried


_V24, _V24_PATH, _TRIED = _load_module()


def check_environment():
    if _V24 is None:
        print(f"ERROR: could not locate {_FILENAME}.", file=sys.stderr)
        for t in _TRIED:
            print(f"    {t}", file=sys.stderr)
        print(f"  Set ${FILE_ENV} to the file, or ${ROOT_ENV} to its directory, or "
              f"keep {_FILENAME} at the repository root beside benchmark-pbya-v3/.",
              file=sys.stderr)
        return False
    try:
        import torch
        t = f"torch {torch.__version__} ({'cuda' if torch.cuda.is_available() else 'cpu'})"
    except Exception:
        t = "torch UNAVAILABLE -> numpy field tier (neural CTF-Flow tier skipped)"
    print(f"spatialcpav24 (learn_spatialcpav24.py, CTF-Flow) from {_V24_PATH}; {t}")
    return True


def _to_dense_f32(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _normalize_expression(adata):
    """Log-normalize the METHOD INPUT, respecting expression_type — same policy as
    every other wrapper, so the comparison stays apples-to-apples."""
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


def build_stack(adata, X_log, X_raw, ct_all):
    Slice, SliceStack = _V24.Slice, _V24.SliceStack
    sections = adata.obs["section"].values.astype(str)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    labels = sorted(np.unique(sections),
                    key=lambda s: float(np.median(coords[sections == s, 2])))
    slices = []
    for sec in labels:
        m = sections == sec
        slices.append(Slice(
            expression=X_log[m], coords_xy=coords[m, :2], z_values=coords[m, 2],
            cell_type_indices=(ct_all[m] if ct_all is not None else None),
            section_id=str(sec),
            raw_expression=(X_raw[m] if X_raw is not None else None)))
    return SliceStack(slices)


def _build_config(args):
    cfg = _V24.CTFFlowConfig()
    cfg.seed = args.seed
    cfg.verbose = True
    cfg.device = args.device
    cfg.loso_select = not args.no_loso
    if args.layout_mode is not None:
        cfg.layout_mode = args.layout_mode
    if args.prior_mode is not None:
        cfg.prior_mode = args.prior_mode
    if args.expr_mode is not None:
        cfg.expr_mode = args.expr_mode
    if args.text_emb is not None:
        cfg.text_emb = args.text_emb
    cfg.retrieval_k = args.retrieval_k
    cfg.corr_modules = args.corr_modules
    cfg.z_proximity = args.z_proximity
    return cfg


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    SpatialCPAv24 = _V24.SpatialCPAv24
    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}

    cfg = _build_config(args)
    gen = SpatialCPAv24(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    print(f"  gates: layout={cfg.layout_mode} prior={cfg.prior_mode} "
          f"expr={cfg.expr_mode} text={cfg.text_emb} loso={cfg.loso_select}")
    # config.invalid_log_markers matches this line: CTF-Flow's neural tier is a
    # scaffold, so trained is False here and the run is quarantined (not scored).
    print(f"  flow-matching model trained: {gen.trained}")

    results = {}
    for sec, z in targets:
        print(f"  {sec}: field query at z={z:.2f} ...")
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
        print(f"    -> {n} cells synthesized")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}
    return results


def main():
    p = argparse.ArgumentParser(
        description="SpatialCPA-v24 wrapper (CTF-Flow continuous transcriptomic field)")
    _v2_io.add_v2_args(p)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # CTF-Flow's headline property is "one command, no flags": LOSO picks the gates
    # per dataset. These override the gates for ablations (design 6 / A1-A6).
    p.add_argument("--no-loso", action="store_true",
                   help="disable the internal LOSO gate selection (use the fixed "
                        "defaults / the flags below)")
    p.add_argument("--layout-mode", default=None,
                   choices=["field", "hybrid", "resample"],
                   help="force the layout gate (default: LOSO-selected)")
    p.add_argument("--prior-mode", default=None, choices=["correlated", "iid"],
                   help="force the prior gate (correlated GRF vs iid noise)")
    p.add_argument("--expr-mode", default=None,
                   choices=["zinb-flow", "cross-mix", "auto-blend"],
                   help="force the expression gate")
    p.add_argument("--text-emb", default=None,
                   choices=["medcpt+residual", "lookup-only"],
                   help="gene-embedding channel (default lookup-only; medcpt needs "
                        "the encoder + transformers)")
    p.add_argument("--retrieval-k", type=int, default=32,
                   help="real neighbours retrieved per query (design 3.2)")
    p.add_argument("--corr-modules", type=int, default=10,
                   help="gene modules for the per-module correlation length")
    p.add_argument("--z-proximity", type=float, default=1.0,
                   help="weight of the z-proximity term SpatialZ omits")
    args = p.parse_args()

    if not check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    X_raw = _to_dense_f32(adata.X)
    et = _normalize_expression(adata)
    X_log = _to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running SpatialCPA-v24 (CTF-Flow) for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed, "loso_select": not args.no_loso,
        "layout_mode": args.layout_mode, "prior_mode": args.prior_mode,
        "expr_mode": args.expr_mode, "text_emb": args.text_emb,
        "retrieval_k": args.retrieval_k, "corr_modules": args.corr_modules,
        "z_proximity": args.z_proximity,
        "design": "CTF-Flow (v23_design.md)", "neural_tier": "scaffold-unvalidated",
        "generation_only": True,
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, "spatialcpav24_gen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
