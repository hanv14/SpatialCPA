"""SpatialCPA-v18 with the expression step replaced by a classical ML regressor.

One wrapper, seven methods
(``--learner ridge|lasso|knn|rf|gbm|xgb|mlp``), registered in
``METHODS`` as ``v18_ridge`` … ``v18_mlp``. Third of the family, after
``run_spatialcpav14_ml.py`` and ``run_spatialcpav21_ml.py``, for the v18
configuration this project runs:

    python -m src.bench3.run_all --methods v18_ridge --dataset easi_fish_lha2 \
        -- --edit-weight 0.0 --ground-blend-flow 1.0 --ground-k 8 \
           --ground-temp 0.25 --ground-keep-margin 1.0 \
           --type-mode vote --type-vote-k 12 --gene-mix-frac 0.15

Same ablation as its siblings: v18's layout, donor selection, type vote and
composition matching are untouched, ``generate_virtual_slice`` is called exactly
as ``run_spatialcpav18`` calls it, and only the returned ``VirtualSlice``'s
``expression`` array is replaced. ``learn_spatialcpav18.py`` is never edited.

TWO THINGS ARE DIFFERENT HERE, AND BOTH MATTER
----------------------------------------------
**1. v18's baseline is already a per-gene chimera, not a single copy.** At
``--edit-weight 0.0`` with ``--gene-mix-frac 0.15``, the v18 gene-mix at
``learn_spatialcpav18.py:1231`` is LIVE: ~15 % of each cell's genes are redrawn
from a second local same-type real cell, and the emitted profile is a blend of
two donors chosen per gene. That is v18's answer to the duplicate-atom / Sinkhorn
problem, and it means this family's baseline is the closest of the three to a
per-gene synthesis method. The ablation replaces the whole emitted matrix, so the
gene-mix is inside the replaced region — it is a per-gene *value* choice, not a
cell-level assignment, so it counts as expression rather than donor selection.
Read the v18 rows as "two-donor per-gene chimera vs learned regression". v14's
(``--edit-weight 0.0``, no gene-mix) is the single-copy contrast; v21's is
muddied by a 0.25 PCA-decode blend (``reports/inert_mechanisms.md``).

**2. v18 emits RAW measurements at these flags, not ``expm1`` of log.** The
raw-output path at ``:1240`` fires when ``raw_output and edit_weight == 0.0`` and
the slices carry ``raw_expression``; it discards the log-scale array entirely and
emits ``pool_raw[pick]``, with the gene-mix re-applied on the raw scale. So the
count tail its siblings replicate is not reached. This wrapper therefore mirrors
v18's own ``raw_ok`` predicate and **trains the learner on whichever scale v18
would have emitted**, so the baseline and the variant are compared on one scale:

    raw_ok  ->  target = X_raw,  emit the prediction verbatim (clipped at 0)
    else    ->  target = X_log,  emit expm1(clip(., 0, 20)) under output_counts

``--target-scale`` overrides the mirror if you want to probe it; ``auto`` is the
default and is what keeps the rows comparable.

⚠️ **Known property, not a defect: on intensity data a squared-loss regressor
trained on the raw scale is dominated by the brightest genes.** The EASI-FISH sets
are exactly that (``fluorescence_intensity`` — the wrapper's normalizer applies
``log1p`` only, no library step, so there ``expm1`` *is* an exact inverse and
``--target-scale log`` is a legitimate cross-check). On a ``raw_counts`` dataset it
is not an inverse, which is the bug v18's raw path exists to fix, so ``auto`` does
not take that shortcut. Report which scale each row ran on — it is in
``method_params``.

Clipping: v18 never clips its raw output because it copies real measurements,
which are already non-negative. A regressor can predict below zero, so the
prediction is clipped at 0. That is a deliberate, minimal deviation, recorded in
``method_params`` as ``raw_clipped_at_zero``.

⚠️ **And it confounds the detection column, for this family only.** The other two
families emit a flat density of 1.000 — no zeros at all, which is the clean
signal that a squared-loss regressor models no sparsity. Here the clip
manufactures zeros wherever the prediction went negative: measured 0.86–1.00
across the learners on a fixture whose real zero fraction was 0.39. So a
``v18_*`` row's ``paper_gene_detection_spearman`` is not as catastrophic as its
siblings', and **the difference is the clip, not the learner.** Do not read it as
the regressor recovering sparsity. ``--target-scale log`` emits through ``expm1``
instead, which cannot produce a zero, and is the cross-check if the distinction
matters for a claim.

Features and target framing are identical to the other two families, so all three
are readable against each other: per cell,
``[x, y, z, one-hot(cell_type), morphology_features(...)]``, fit once per volume
on training sections only.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

# v18's own wrapper: imported, never modified. It gives us its module loader, its
# normalization policy, its `build_stack` AND its `_build_config` — so unlike the
# v14 family this one duplicates no config mapping at all, only the flag table.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18 as _V18W    # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "benchmark-pbya-v2" / "src" / "benchmark" / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]
                       / "benchmark-pbya-v2" / "src" / "benchmark"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ml_learners as ML        # noqa: E402
import _v2_io                        # noqa: E402
import leakage_guard                 # noqa: E402

_V18 = _V18W._V18
_V18_WRAPPER_PATH = Path(_V18W.__file__)


# ── v18 CLI parity ───────────────────────────────────────────────────────────
def _flag_table(src):
    """{flag: default-or-'FLAG'} for every add_argument in a wrapper's source."""
    out = {}
    for m in re.finditer(r'(?:p|parser)\.add_argument\(\s*"(--[\w-]+)"(.*?)\)\n',
                         src, re.S):
        flag, rest = m.group(1), m.group(2)
        if re.search(r'action="store_true"', rest):
            out[flag] = "FLAG"
        else:
            d = re.search(r'default=([^,)]+)', rest)
            out[flag] = d.group(1).strip() if d else "NO-DEFAULT"
    return out


def assert_v18_parity():
    """Fail the run if this wrapper's flag table has drifted from v18's.

    Only the flags are duplicated — the config mapping is v18's own
    ``_build_config``, imported and called, so there is nothing to check there.
    v18's tuning arrives through ``run_all``'s ``extra_args``, so a flag this
    wrapper lacks is an argparse error mid-campaign and a flag whose default
    differs is worse: it runs, and it is not the configuration under test.
    """
    theirs = _flag_table(_V18_WRAPPER_PATH.read_text())
    ours = _flag_table(Path(__file__).read_text())
    problems = []
    for flag, want in theirs.items():
        if flag not in ours:
            problems.append(f"{flag}: declared by v18, missing here")
        elif _lit(ours[flag]) != _lit(want):
            problems.append(f"{flag}: v18 default {want!r} != ours {ours[flag]!r}")
    if problems:
        raise SystemExit(
            f"ERROR: this wrapper has drifted from {_V18_WRAPPER_PATH.name}, so it "
            f"would NOT accept or reproduce v18's configuration:\n  "
            + "\n  ".join(problems)
            + "\nRefusing to score a configuration that is not the method under test.")
    return len(theirs)


def _lit(s):
    try:
        import ast
        return ast.literal_eval(str(s))
    except Exception:
        return str(s)


def check_environment(learner):
    if not _V18W.check_environment():
        return False
    versions = ML.require_learner_deps(learner)
    if versions is None:
        return False
    n_flags = assert_v18_parity()
    print(f"v18 parity: {n_flags} flags match {_V18_WRAPPER_PATH.name}; "
          f"config built by its own _build_config")
    print(f"expression learner: {learner} ({versions})")
    return True


# ── which scale v18 would emit ───────────────────────────────────────────────
def resolve_target_scale(cfg, X_raw, requested):
    """Mirror ``learn_spatialcpav18.py:1240``'s ``raw_ok`` predicate.

    v18 emits the picked cell's RAW measurement (gene-mix re-applied on the raw
    scale) whenever ``raw_output and edit_weight == 0.0`` and both flanking slices
    carry ``raw_expression``. ``build_stack`` attaches ``raw_expression`` to every
    slice iff ``X_raw`` is not None, so the per-slice half of that predicate is a
    single check here. Training the learner on any other scale would compare it to
    the baseline through a transform the baseline never applied.
    """
    raw_ok = bool(cfg.raw_output and cfg.edit_weight == 0.0 and X_raw is not None)
    if requested == "auto":
        return "raw" if raw_ok else "log"
    if requested == "raw" and X_raw is None:
        raise SystemExit("ERROR: --target-scale raw, but the input carries no raw "
                         "expression to train on.")
    return requested


# ── features ─────────────────────────────────────────────────────────────────
def build_features(xy, z, ct_idx, n_types, cfg):
    """Per-cell feature matrix for the expression regressor.

    (N, 2) in-plane coords + z + one-hot type + morphology -> (N, F) float64,
    F = 3 + 2 * max(n_types, 1) + 1. Identical construction at fit and predict
    time. ``morphology_features`` is v18's own (``learn_spatialcpav18.py:319``).
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    z = np.asarray(z, dtype=np.float64)
    z = np.full(n, float(z)) if z.ndim == 0 else z.astype(np.float64)
    nt = max(int(n_types), 1)
    onehot = np.zeros((n, nt), dtype=np.float64)
    if ct_idx is not None and int(n_types) >= 1 and n:
        onehot[np.arange(n), np.clip(np.asarray(ct_idx, dtype=int), 0, nt - 1)] = 1.0
    morph = _V18.morphology_features(xy, ct_idx, int(n_types),
                                     k=cfg.morph_k, density_sigma=cfg.density_sigma)
    return np.hstack([xy, z[:, None], onehot,
                      np.asarray(morph, dtype=np.float64)]).astype(np.float64)


def training_matrix(stack, n_types, cfg, scale):
    """(F, Y) over every training cell, Y on the scale v18 would emit.

    ``Slice.expression`` is the log-normalized matrix and ``Slice.raw_expression``
    the pre-normalization measurement; ``build_stack`` fills both, so the choice
    here is a field selection, not a transform.
    """
    Fs, Ys = [], []
    for s in stack.slices:
        Fs.append(build_features(s.coords_xy, s.z_values, s.cell_type_indices,
                                 n_types, cfg))
        y = s.raw_expression if scale == "raw" else s.expression
        if y is None:
            raise SystemExit(f"ERROR: section {s.section_id!r} carries no "
                             f"{scale} expression to train on.")
        Ys.append(np.asarray(y, dtype=np.float64))
    return np.vstack(Fs), np.vstack(Ys)


def emit(pred, cfg, scale):
    """Put the learner's prediction on v18's own output scale.

    ``raw``: v18 emits the copied measurement verbatim, with no clip — real values
    are already non-negative. A regressor is not, so the prediction is clipped at
    zero; that is the one deviation, and it is recorded in ``method_params``.
    ``log``: v18's count tail, verbatim (``learn_spatialcpav18.py:1245-1247``).
    """
    expr = np.clip(pred, 0.0, None)
    if scale == "log" and cfg.output_counts:
        expr = np.expm1(np.clip(expr, 0.0, 20.0))
    return expr.astype(np.float32)


# ── run ──────────────────────────────────────────────────────────────────────
def run_method(adata, targets, gene_names, X_log, X_raw, args):
    SpatialCPAv14 = _V18.SpatialCPAv14

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, {}

    # v18's own config builder, imported and called — no mapping duplicated here.
    cfg = _V18W._build_config(args)
    scale = resolve_target_scale(cfg, X_raw, args.target_scale)
    print(f"  edit_w={cfg.edit_weight}, blend={cfg.ground_blend_flow}, "
          f"k={cfg.ground_k}, temp={cfg.ground_temp}, "
          f"keep_margin={cfg.ground_keep_margin}, type_mode={cfg.type_mode}"
          f"(k={cfg.type_vote_k}), gene_mix={cfg.gene_mix_frac}, "
          f"raw_output={cfg.raw_output}")
    print(f"  target scale: {scale} ({'mirrors' if args.target_scale == 'auto' else 'OVERRIDES'} "
          f"v18's raw_ok = {cfg.raw_output and cfg.edit_weight == 0.0 and X_raw is not None})")
    if cfg.edit_weight == 0.0 and cfg.gene_mix_frac > 0.0:
        print(f"  v18 baseline here is a TWO-DONOR per-gene chimera "
              f"({cfg.gene_mix_frac:.0%} of genes redrawn), not a single copy; "
              f"the learner replaces all of it")

    gen = SpatialCPAv14(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    # Matched by config.invalid_log_markers, exactly as for spatialcpav18_gen.
    print(f"  flow-matching model trained: {gen.trained}")

    n_types = int(getattr(gen, "n_types", 0) or 0)
    Ftr, Ytr = training_matrix(stack, n_types, cfg, scale)
    print(f"  learner fit: {args.learner} on {Ftr.shape[0]} training cells x "
          f"{Ftr.shape[1]} features -> {Ytr.shape[1]} genes ({scale} target)")
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
        print(f"  {sec}: v18 layout/donor selection at z={z:.2f}, then "
              f"{args.learner} expression ...")
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
        if args.emit == "donor":
            incumbent = ML.to_target_space(vs.expression, scale)
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
        expr = emit(pred, cfg, scale)
        predict_seconds += time.time() - t_pred

        print(f"    -> {n} cells; emitted density {float((expr > 0).mean()):.4f} "
              f"(a regressor emits no zeros; this is the measurement)")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(expr)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}

    return results, {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                     "n_train_cells": int(Ftr.shape[0]),
                     "n_features": int(Ftr.shape[1]), "target_scale": scale,
                     "raw_clipped_at_zero": scale == "raw"}


def build_parser():
    """v18's CLI verbatim, plus the learner. Checked by ``assert_v18_parity``."""
    p = argparse.ArgumentParser(
        description="SpatialCPA-v18 with a classical ML expression head "
                    "(layout, donor selection and every v18 flag unchanged)")
    _v2_io.add_v2_args(p)
    ML.add_learner_args(p)
    p.add_argument("--target-scale", default="auto", choices=["auto", "log", "raw"],
                   help="scale the learner is trained on and emits. 'auto' mirrors "
                        "v18's own raw_ok predicate, which is what keeps the rows "
                        "comparable; override only to probe the choice")
    # ── v18's own flags, defaults identical to run_spatialcpav18.py ──────────
    p.add_argument("--epochs", type=int, default=160, help="Phase B flow epochs")
    p.add_argument("--pretrain-epochs", type=int, default=60, help="Phase A epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--joint-dim", type=int, default=48)
    p.add_argument("--ode-steps", type=int, default=12)
    p.add_argument("--ensemble", type=int, default=4)
    p.add_argument("--context-slices", type=int, default=None)
    p.add_argument("--position-mode", default="flanking",
                   choices=["flanking", "morph", "nearest"])
    p.add_argument("--ground-blend-flow", type=float, default=0.20,
                   help="fraction of cells re-grounded to the flow-latent pick")
    p.add_argument("--ground-k", type=int, default=8,
                   help="local real candidates the flow latent grounds among")
    p.add_argument("--ground-temp", type=float, default=0.25,
                   help="softmax temperature for exemplar sampling")
    p.add_argument("--edit-weight", type=float, default=0.25,
                   help="blend toward the flow-decoded profile (0 = pure exemplar). "
                        "Also selects v18's raw-output path, which this wrapper "
                        "mirrors when choosing the learner's target scale")
    p.add_argument("--no-coherent-source", action="store_true")
    p.add_argument("--no-output-counts", action="store_true",
                   help="emit log1p-normalized (not count-like) expression")
    p.add_argument("--no-composition-match", action="store_true")
    p.add_argument("--type-mode", default="vote", choices=["inherit", "vote"])
    p.add_argument("--type-vote-k", type=int, default=12)
    p.add_argument("--gene-mix-frac", type=float, default=0.15,
                   help="minority fraction of each cell's genes resampled from a "
                        "local same-type real cell. Part of v18's EXPRESSION step, "
                        "so the learner replaces it")
    p.add_argument("--no-gene-mix", action="store_true")
    p.add_argument("--no-raw-output", action="store_true",
                   help="ablate raw output. Also flips this wrapper's target scale "
                        "to log, because it flips v18's")
    p.add_argument("--no-ground-sample", action="store_true")
    p.add_argument("--no-dedup-ground", action="store_true")
    p.add_argument("--ground-keep-margin", type=float, default=1.0)
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

    X_raw = _V18W._to_dense_f32(adata.X)
    et = _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)
    print(f"  expression_type={et}")

    print(f"Running v18_{args.learner} for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results, info = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed,
        "ablation": "expression head only; v18 layout, donor selection, type vote "
                    "and composition matching unchanged. NOTE the v18 gene-mix is "
                    "an expression mechanism and is therefore replaced too",
        "expression_framing": "regress expression on "
                              "(x, y, z, one-hot type, morphology_features)",
        "fit_granularity": "once per volume, training sections only",
        "epochs": args.epochs, "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "joint_dim": args.joint_dim,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "position_mode": args.position_mode,
        "edit_weight": args.edit_weight,
        "ground_blend_flow": args.ground_blend_flow,
        "ground_k": args.ground_k, "ground_temp": args.ground_temp,
        "ground_keep_margin": args.ground_keep_margin,
        "type_mode": args.type_mode, "type_vote_k": args.type_vote_k,
        "gene_mix_frac": 0.0 if args.no_gene_mix else args.gene_mix_frac,
        "raw_output": not args.no_raw_output,
        "ground_sample": not args.no_ground_sample,
        "dedup_ground": not args.no_dedup_ground,
        "coherent_source": not args.no_coherent_source,
        "flow_matching": True, "generation_only": True,
        **ML.learner_params(args.learner, args),
        **info,
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, f"v18_{args.learner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
