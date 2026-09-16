"""SpatialCPA-v21 with the expression step replaced by a classical ML regressor.

One wrapper, seven methods
(``--learner ridge|lasso|knn|rf|gbm|xgb|mlp``), registered in
``METHODS`` as ``v21_ridge`` … ``v21_mlp``. Each pins its learner through
``wrapper_args``, the mechanism ``spatialcpav8_gen`` already uses, so a bare
invocation is reproducible from ``config.py`` alone.

WHAT IS ABLATED, AND WHAT IS NOT
--------------------------------
v21's layout, donor selection, cell-type vote, composition matching and every
configuration knob are **untouched** — not re-implemented, not re-tuned.
``generate_virtual_slice`` is called exactly as ``run_spatialcpav21`` calls it,
and the returned ``VirtualSlice`` already carries v21's positions and labels. The
only substitution is its ``expression`` array:

    vs = gen.generate_virtual_slice(z)          # v21, unmodified
    vs.expression  <-  learner.predict(features(vs.coords, vs.cell_type_idx))

So ``learn_spatialcpav21.py`` is never edited and no part of ``_generate`` is
copied. The swap happens at the ``VirtualSlice`` boundary, which is the narrowest
contract that separates "which real cell do I copy" from "what values do I emit".

THE QUESTION IT ANSWERS
-----------------------
Donor copying vs. learned regression at the same interface. **Not** cross-mix:
under every ``paper_*`` design ``alpha`` is exactly zero by construction, so
v21's cross-mix never executes on these rows at all — see
``reports/inert_mechanisms.md``. As benchmarked, v21's whole emitted expression is
``expm1(clip(0.75·X_log[donor] + 0.25·PCA_decode(e_hat), 0, 20))``, and that is
what these five variants are measured against.

FEATURES AND TARGET (framing A)
-------------------------------
Per cell, from information a generated cell actually has:

    [ x, y, z, one-hot(cell_type), morphology_features(...) ]

``morphology_features`` is v21's own (``learn_spatialcpav21.py:531``): a
Gaussian-weighted local cell-type composition plus a local density channel,
computed within one section. Target: that cell's **log-normalized** expression —
the same scale ``expr`` carries at the swap point — and v21's own count tail
(``expm1(clip(·, 0, 20))``) converts it back, so the output path is unchanged.

The learner is fit **once per volume** on the training sections only (the input
file physically excludes the held-out sections; ``guard_no_holdout`` enforces it),
and ``z`` is a feature, so one model serves every held-out section.

EXPECT A DENSE PREDICTION. A squared-loss regressor emits the local conditional
mean, which has no zeros. ``paper_gene_detection_spearman`` reads the raw emitted
matrix precisely because rank-normalization would erase that, so the detection
columns will fall and ``paper_morans_mae`` will rise. That is the measurement,
not a defect: it is where "what donor copying contributes" is written down.

Speaks the identical ``_v2_io`` contract as every other wrapper, so
``run_benchmark`` invokes it exactly as it invokes v21 and the evaluator reads its
output unchanged.
"""

import argparse
import dataclasses
import re
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# The v21 wrapper itself: imported, never modified. Reusing it — rather than
# copying its module loader, its normalization policy and its forty flags — is
# what guarantees these variants run v21's code and v21's configuration. Its
# argparse lives inside main(), so importing it parses nothing.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav21 as _V21W     # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "benchmark-pbya-v2" / "src" / "benchmark" / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "benchmark-pbya-v2" / "src" / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ml_learners as ML        # noqa: E402
import _v2_io                         # noqa: E402
import leakage_guard                  # noqa: E402

_V21 = _V21W._V21



# ── v21 configuration parity ─────────────────────────────────────────────────
def _assert_v21_config_parity():
    """Every knob ``run_spatialcpav21._build_config`` sets equals its V14Config default.

    These variants build the config as ``V14Config()`` plus seed/verbose/device
    rather than by re-declaring v21's forty flags, because a second copy of those
    defaults is a second thing to drift. That shortcut is only valid while v21's
    argparse defaults still mirror ``V14Config``'s — which its own docstring
    asserts but nothing checked. This checks it, by reading both sources, and
    fails the run rather than silently benchmarking a configuration that is no
    longer v21's.

    Returns the list of knobs verified.
    """
    wrapper_src = Path(_V21W.__file__).read_text()
    module_src = Path(_V21W._V21_PATH).read_text()

    argp = {}
    for m in re.finditer(r'p\.add_argument\(\s*"(--[\w-]+)"(.*?)\)\n', wrapper_src, re.S):
        flag, rest = m.group(1), m.group(2)
        if re.search(r'action="store_true"', rest):
            argp[flag] = "False"
        else:
            d = re.search(r'default=([^,)]+)', rest)
            argp[flag] = d.group(1).strip() if d else None

    defaults = {}
    block = module_src.split("class V14Config:")[1].split("\nclass ")[0]
    for line in block.split("\n"):
        m = re.match(r'\s{4}(\w+):\s*[\w\[\].]+\s*=\s*([^#]+?)\s*(#.*)?$', line)
        if m:
            defaults[m.group(1)] = m.group(2).strip()

    body = wrapper_src.split("def _build_config(args):")[1].split("\ndef ")[0]
    checked, bad = [], []
    for name, expr in re.findall(r'cfg\.(\w+)\s*=\s*(.+)', body):
        expr = expr.split("#")[0].strip()
        if name in ("seed", "verbose", "device"):
            continue
        if name == "context_slices_each_side":
            # set only under `if args.context_slices is not None`, and its
            # argparse default IS None, so the V14Config default stands.
            checked.append(name)
            continue
        m = re.fullmatch(r'args\.([\w_]+)', expr)
        n = re.fullmatch(r'not args\.(no_[\w_]+)', expr)
        z = re.fullmatch(r'0\.0 if args\.(no_[\w_]+) else args\.([\w_]+)', expr)
        if m:
            want = argp.get("--" + m.group(1).replace("_", "-"))
        elif n:
            want = "True"                      # `not args.no_x`, store_true default False
        elif z:
            want = argp.get("--" + z.group(2).replace("_", "-"))
        else:
            bad.append(f"{name}: cannot parse `{expr}`")
            continue
        have = defaults.get(name)
        if want is None or have is None or _lit(want) != _lit(have):
            bad.append(f"{name}: wrapper default {want!r} != V14Config default {have!r}")
        else:
            checked.append(name)
    if bad:
        raise SystemExit(
            "ERROR: run_spatialcpav21.py's defaults no longer mirror V14Config's, so "
            "building the config as V14Config() would NOT reproduce v21:\n  "
            + "\n  ".join(bad)
            + "\nFix the drift, or teach this wrapper the flags explicitly. Refusing "
              "to score a configuration that is not the method under test.")
    return checked


def _lit(s):
    try:
        import ast
        return ast.literal_eval(str(s))
    except Exception:
        return str(s)


def check_environment(learner):
    if not _V21W.check_environment():
        return False
    # No silent fallback: the learner's library IS the method under test.
    versions = ML.require_learner_deps(learner)
    if versions is None:
        return False
    knobs = _assert_v21_config_parity()
    print(f"v21 config parity: {len(knobs)} knobs match V14Config defaults")
    print(f"expression learner: {learner} ({versions})")
    return True


# ── features ─────────────────────────────────────────────────────────────────
def build_features(xy, z, ct_idx, n_types, cfg):
    """Per-cell feature matrix for the expression regressor.

    (N, 2) in-plane coords + z + one-hot type + morphology -> (N, F) float64,
    F = 3 + 2 * max(n_types, 1) + 1. Identical construction at fit and predict
    time; this is the only place features are defined.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    z = np.asarray(z, dtype=np.float64)
    z = np.full(n, float(z)) if z.ndim == 0 else z.astype(np.float64)
    nt = max(int(n_types), 1)
    onehot = np.zeros((n, nt), dtype=np.float64)
    if ct_idx is not None and int(n_types) >= 1 and n:
        onehot[np.arange(n), np.clip(np.asarray(ct_idx, dtype=int), 0, nt - 1)] = 1.0
    morph = _V21.morphology_features(xy, ct_idx, int(n_types),
                                     k=cfg.morph_k, density_sigma=cfg.density_sigma)
    F = np.hstack([xy, z[:, None], onehot, np.asarray(morph, dtype=np.float64)])
    return F.astype(np.float64)


def training_matrix(stack, n_types, cfg):
    """(F, Y) over every training cell. Y is log-normalized expression (n, G).

    Features are built per section, because ``morphology_features`` is a
    within-slice quantity (its own docstring) and the generated slice is one
    section. Per-cell z is used, not the section centre — the target z is a
    plane between two training slabs and the model should see the real spread.
    """
    Fs, Ys = [], []
    for s in stack.slices:
        Fs.append(build_features(s.coords_xy, s.z_values, s.cell_type_indices,
                                 n_types, cfg))
        Ys.append(np.asarray(s.expression, dtype=np.float64))
    return np.vstack(Fs), np.vstack(Ys)


# ── run ──────────────────────────────────────────────────────────────────────


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    SpatialCPAv14 = _V21.SpatialCPAv14

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = _V21W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, {}

    # v21's shipped configuration, verified knob-by-knob against the wrapper's
    # own defaults by _assert_v21_config_parity().
    cfg = _V21.V14Config()
    cfg.seed = args.seed
    cfg.verbose = True
    cfg.device = args.device

    if cfg.edit_weight == 0.0:
        raise SystemExit(
            "ERROR: V14Config.edit_weight == 0.0. v21's output tail is then the "
            "raw-donor path (learn_spatialcpav21.py:1792), which has no meaning "
            "for a regressor — there is no donor value to emit. This wrapper "
            "replicates only the count tail expm1(clip(., 0, 20)). Refusing to "
            "emit expression through a path it cannot reproduce.")

    gen = SpatialCPAv14(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    # Matched by config.invalid_log_markers, exactly as for spatialcpav21_gen.
    print(f"  flow-matching model trained: {gen.trained}")

    n_types = int(getattr(gen, "n_types", 0) or 0)
    Ftr, Ytr = training_matrix(stack, n_types, cfg)
    print(f"  learner fit: {args.learner} on {Ftr.shape[0]} training cells x "
          f"{Ftr.shape[1]} features -> {Ytr.shape[1]} genes (log-normalized target)")
    model = ML.make_learner(args.learner, args)
    t_fit = time.time()
    model.fit(Ftr, Ytr)
    fit_seconds = time.time() - t_fit
    ML.freeze_for_determinism(model)
    print(f"    fit_seconds={fit_seconds:.1f}")
    ML.report_degeneracy(model, args.learner)

    tr_xy = tr_z = tr_type = tr_sec = None
    if args.emit == "donor":
        # The field target is the SAME model the *_<learner> variants emit;
        # only its use differs. The pool's own predicted field is cached once
        # here because the noise floor is measured against it.
        args._pool_pred = np.asarray(model.predict(Ftr), dtype=np.float64)
        tr_xy, tr_z, tr_type, tr_sec = ML.training_index(stack)
        print(f"    --emit donor: predictions used as a field target; "
              f"every emitted value stays a real measurement")

    results, predict_seconds = {}, 0.0
    for sec, z in targets:
        print(f"  {sec}: v21 layout/donor selection at z={z:.2f}, then {args.learner} "
              f"expression ...")
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

        t_pred = time.time()
        Fp = build_features(vs.coords[:, :2], vs.coords[:, 2],
                            vs.cell_type_idx, n_types, cfg)
        pred = np.asarray(model.predict(Fp), dtype=np.float64)
        if pred.shape != (n, Ytr.shape[1]):
            raise SystemExit(f"ERROR: learner returned {pred.shape}, expected "
                             f"{(n, Ytr.shape[1])}")
        # v21's own output tail, verbatim (learn_spatialcpav21.py:1783 and :1800).
        if args.emit == "donor":
            incumbent = ML.to_target_space(vs.expression, "log")
            cand = ML.flanking_rows(stack, tr_sec, float(vs.coords[0, 2]))
            pred, dstats = ML.select_donor_by_field(
                pred, incumbent, Ytr, tr_xy, tr_type, cand,
                vs.coords[:, :2], (vs.cell_type_idx
                                   if vs.cell_type_idx is not None
                                   else np.zeros(n, np.int64)), args)
            print(f"    field-guided: {dstats['n_swapped']}/{n} donors swapped "
                  f"({dstats['n_eligible']} eligible), median deviation "
                  f"{dstats['dev_before']:.4f} -> {dstats['dev_after']:.4f}, "
                  f"floor sigma0={dstats['sigma0']:.4f}")
        expr = np.clip(pred, 0.0, None)
        if cfg.output_counts:
            expr = np.expm1(np.clip(expr, 0.0, 20.0))
        expr = expr.astype(np.float32)
        predict_seconds += time.time() - t_pred

        dens = float((expr > 0).mean())
        print(f"    -> {n} cells; emitted density {dens:.4f} "
              f"(a regressor emits no zeros; this is the measurement)")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V21W._to_dense_f32(expr)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}

    timing = {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
              "n_train_cells": int(Ftr.shape[0]), "n_features": int(Ftr.shape[1])}
    return results, {"cfg": dataclasses.asdict(cfg), **timing}


def build_parser():
    """v21's CLI, plus the shared learner flags. Mirrors its siblings."""
    p = argparse.ArgumentParser(
        description="SpatialCPA-v21 with a classical ML expression head "
                    "(layout, donor selection and every v21 flag unchanged)")
    _v2_io.add_v2_args(p)
    ML.add_learner_args(p)
    # v21-specific. Its siblings declare --device inside their host flag
    # tables; v21_ml declares no host flags (it builds V14Config() directly),
    # so this one lives here.
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="device for v21's flow (layout and donor selection are v21's)")
    return p


def main():
    args = build_parser().parse_args()
    # Fail on a wrapper bug before touching any data (see the guard's docstring).
    ML.assert_args_declared(args, __file__)

    if not check_environment(args.learner):
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    ML.check_output_size(adata, len(targets), args.max_dense_gb)
    gene_names = list(adata.var_names)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    X_raw = _V21W._to_dense_f32(adata.X)
    et = _V21W._normalize_expression(adata)
    X_log = _V21W._to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running v21_{args.learner} for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results, info = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed,
        "ablation": "expression head only; v21 layout, donor selection, type vote "
                    "and composition matching unchanged",
        "expression_framing": "regress log-normalized expression on "
                              "(x, y, z, one-hot type, morphology_features)",
        "target_scale": "log1p-normalized; emitted through v21's expm1 count tail",
        "fit_granularity": "once per volume, training sections only",
        "flow_matching": True, "generation_only": True,
        **ML.learner_params(args.learner, args),
        **{k: v for k, v in info.items() if k != "cfg"},
        "v21_config": info.get("cfg", {}),
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, f"v21_{args.learner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
