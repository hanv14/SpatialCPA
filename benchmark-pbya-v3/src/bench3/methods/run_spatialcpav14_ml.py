"""SpatialCPA-v14 with the expression step replaced by a classical ML regressor.

One wrapper, seven methods
(``--learner ridge|lasso|knn|rf|gbm|xgb|mlp``), registered in
``METHODS`` as ``v14_ridge`` … ``v14_mlp``. The sibling of
``run_spatialcpav21_ml.py``, for the v14 configuration this project actually runs:

    python -m src.bench3.run_all --methods v14_ridge --dataset <ds> \
        -- --edit-weight 0.0 --ground-blend-flow 1.0 --ground-k 2

WHAT IS ABLATED, AND WHAT IS NOT
--------------------------------
v14's layout, donor selection, cell-type placement and composition matching are
untouched. ``generate_virtual_slice`` is called exactly as the v2 wrapper calls
it, and the returned ``VirtualSlice`` already carries v14's positions and labels.
Only its ``expression`` array is replaced:

    vs = gen.generate_virtual_slice(z)          # v14, unmodified
    vs.expression  <-  learner.predict(features(vs.coords, vs.cell_type_idx))

``spatialcpav14/`` is never edited, ``benchmark-pbya-v2`` is never edited, and no
part of ``trainer.generate_slice`` is copied.

WHY THE v14 CONTRAST IS CLEANER THAN v21'S
------------------------------------------
At ``--edit-weight 0.0`` v14 emits a **verbatim** real profile: ``_ground``
returns ``pool_expr[pick]`` (``spatialcpav14/trainer.py:662``) and the decode
blend at ``:532`` is skipped entirely. So the comparison is exactly "emit one real
cell's profile" vs "emit a regressed conditional mean", with no blend in between.
The v21 variants cannot say that — v21 runs at ``edit_weight=0.25``, so a quarter
of its output is already a PCA decode (``reports/inert_mechanisms.md``).

Note also what ``--ground-blend-flow 1.0 --ground-k 2`` makes the baseline. Every
cell is re-grounded (``n_flow == n``, ``trainer.py:655``) but among only its two
spatially-nearest real cells, and ``_pick_candidate`` at ``selection="flow"`` is a
plain ``argmin`` over those two (``:634``). The donor is therefore "the better of
my two nearest real neighbours" — the baseline is close to a local copy, and the
flow's role in it is one binary choice per cell. Worth saying out loud when the
ablation is read.

v14 has **no raw-output path** — ``build_stack`` never carries ``raw_expression``
— so unlike v21 there is nothing that ``edit_weight == 0.0`` diverts the output
through. The tail is ``expm1(clip(·, 0, 20))`` under ``output_counts``, full stop,
and this wrapper reproduces it exactly.

FEATURES AND TARGET
-------------------
Identical framing to the v21 variants, so the two families are readable against
each other: per cell, ``[x, y, z, one-hot(cell_type), morphology_features(...)]``
against a **log-normalized** expression target, fit once per volume on training
sections only, emitted through v14's own count tail. A squared-loss regressor
emits no zeros; the detection columns are supposed to see that.

FLAG PARITY
-----------
Unlike the v21 variants — which inherit ``V14Config()`` because the benchmark
passes v21 no flags — this family IS driven by flags, so it declares v14's entire
CLI and maps it onto ``SpatialCPAv14Config`` the same way. Two source-level guards
check that duplication rather than trusting it: the flag table and the
``cfg.*`` assignment block are both compared against
``benchmark-pbya-v2/.../run_spatialcpav14.py`` at startup, and any drift fails the
run instead of silently benchmarking a different configuration.
"""

import argparse
import importlib.util
import re
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_V2_BENCH = (Path(__file__).resolve().parents[4]
             / "benchmark-pbya-v2" / "src" / "benchmark")
sys.path.insert(0, str(_V2_BENCH / "methods"))
sys.path.insert(0, str(_V2_BENCH))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ml_learners as ML        # noqa: E402
import _v2_io                         # noqa: E402
import leakage_guard                  # noqa: E402

_V14_WRAPPER_PATH = _V2_BENCH / "methods" / "run_spatialcpav14.py"


def _load_v14_wrapper():
    """Import v2's v14 wrapper from disk. Reused, never modified.

    Gives us its ``build_stack``, ``_normalize_expression``, ``_to_dense_f32``
    and package locator, so the input pipeline is v14's own rather than a copy.
    Its argparse lives in ``main()``, so importing it parses nothing and runs
    nothing.
    """
    spec = importlib.util.spec_from_file_location("run_spatialcpav14_v2",
                                                  str(_V14_WRAPPER_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_spatialcpav14_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


_V14W = _load_v14_wrapper()


# ── v14 CLI / config parity ──────────────────────────────────────────────────
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


def _cfg_assignments(src, func):
    """The set of `cfg.<path> = <expr>` lines inside one function, normalized."""
    body = src.split(f"def {func}(")[1].split("\ndef ")[0]
    return {f"{a} = {b.split('#')[0].strip()}"
            for a, b in re.findall(r'\s(cfg\.[\w.]+)\s*=\s*(.+)', body)}


def assert_v14_parity():
    """Fail the run if this wrapper has drifted from v2's v14 wrapper.

    Two checks, because two things are duplicated and both matter:

    1. **Flags.** Every v14 flag must exist here with the identical default. The
       benchmark passes v14's tuning through ``extra_args`` (``--edit-weight``,
       ``--ground-blend-flow``, ``--ground-k``, …), so a flag this wrapper lacks
       is an argparse error mid-campaign, and a flag whose default differs is
       worse — it runs, and it is not the configuration under test.
    2. **Config mapping.** Every ``cfg.<path> = args.<flag>`` line in v14's
       ``run_method`` must appear here verbatim, so a knob cannot be wired to a
       different config field than v14 wires it to.
    """
    v14_src = _V14_WRAPPER_PATH.read_text()
    mine_src = Path(__file__).read_text()

    # Both sides are parsed from source the same way, so they compare like for
    # like; flags this wrapper adds on top (--learner and the learner knobs) are
    # simply not in `theirs` and are ignored.
    theirs, ours = _flag_table(v14_src), _flag_table(mine_src)
    problems = []
    for flag, want in theirs.items():
        if flag not in ours:
            problems.append(f"{flag}: declared by v14, missing here")
        elif _lit(ours[flag]) != _lit(want):
            problems.append(f"{flag}: v14 default {want!r} != ours {ours[flag]!r}")

    theirs_cfg = _cfg_assignments(v14_src, "run_method")
    ours_cfg = _cfg_assignments(mine_src, "build_v14_config")
    for line in sorted(theirs_cfg - ours_cfg):
        problems.append(f"config mapping missing or changed: `{line}`")

    if problems:
        raise SystemExit(
            "ERROR: this wrapper has drifted from "
            f"{_V14_WRAPPER_PATH.name}, so it would NOT run v14's configuration:\n  "
            + "\n  ".join(problems)
            + "\nRefusing to score a configuration that is not the method under test.")
    return len(theirs), len(theirs_cfg)


def _lit(s):
    try:
        import ast
        return ast.literal_eval(str(s))
    except Exception:
        return str(s)


def check_environment(learner):
    if not _V14W.check_environment():
        return False
    versions = ML.require_learner_deps(learner)
    if versions is None:
        return False
    n_flags, n_cfg = assert_v14_parity()
    print(f"v14 parity: {n_flags} flags and {n_cfg} config assignments match "
          f"{_V14_WRAPPER_PATH.name}")
    print(f"expression learner: {learner} ({versions})")
    return True


# ── v14 configuration (mirrors run_spatialcpav14.run_method; guarded above) ──
def build_v14_config(args):
    from spatialcpav14 import SpatialCPAv14Config
    cfg = SpatialCPAv14Config()
    cfg.seed = args.seed
    cfg.train.seed = args.seed
    cfg.train.epochs = args.epochs
    cfg.train.pretrain_epochs = args.pretrain_epochs
    cfg.train.device = args.device
    cfg.latent.expr_latent_dim = args.latent_dim
    cfg.encoder.joint_dim = args.joint_dim
    cfg.flow.n_ode_steps = args.ode_steps
    cfg.flow.n_ensemble = args.ensemble
    cfg.generation.position_mode = args.position_mode
    cfg.generation.displacement_scale = args.displacement_scale
    cfg.generation.ground_blend_flow = args.ground_blend_flow
    cfg.generation.ground_k = args.ground_k
    cfg.generation.edit_weight = args.edit_weight
    cfg.generation.selection = args.selection
    cfg.generation.ground_expression = not args.no_ground
    if args.context_slices is not None:
        cfg.attn.context_slices_each_side = args.context_slices
    if args.gap_dropout is not None:
        cfg.train.gap_dropout = args.gap_dropout
    if args.z_sigma is not None:
        cfg.train.z_sigma = args.z_sigma
    if args.morph_k is not None:
        cfg.latent.morph_k = args.morph_k
    if args.n_context is not None:
        cfg.attn.n_context = args.n_context
    if args.flow_hidden is not None:
        cfg.flow.hidden = args.flow_hidden
    if args.flow_layers is not None:
        cfg.flow.n_layers = args.flow_layers
    if args.type_placement is not None:
        cfg.generation.type_placement = args.type_placement
    if args.no_coherent_source:
        cfg.generation.coherent_source = False
    if args.coherent_freq is not None:
        cfg.generation.coherent_freq = args.coherent_freq
    cfg.generation.output_counts = not args.no_output_counts
    cfg.generation.composition_match = not args.no_composition_match
    if args.no_bio:
        cfg.bio.w_interface = cfg.bio.w_hypoxia = cfg.bio.w_consistency = cfg.bio.w_smooth = 0.0
    if args.no_attention:
        cfg.attn.n_context = 1
        cfg.attn.n_global_tokens = 0
    return cfg


# ── features ─────────────────────────────────────────────────────────────────
def build_features(xy, z, ct_idx, n_types, cfg):
    """Per-cell feature matrix for the expression regressor.

    (N, 2) in-plane coords + z + one-hot type + morphology -> (N, F) float64,
    F = 3 + 2 * max(n_types, 1) + 1. Identical construction at fit and predict
    time; this is the only place features are defined. ``morphology_features`` is
    v14's own (``spatialcpav14/latents.py:67``).
    """
    from spatialcpav14 import morphology_features
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    z = np.asarray(z, dtype=np.float64)
    z = np.full(n, float(z)) if z.ndim == 0 else z.astype(np.float64)
    nt = max(int(n_types), 1)
    onehot = np.zeros((n, nt), dtype=np.float64)
    if ct_idx is not None and int(n_types) >= 1 and n:
        onehot[np.arange(n), np.clip(np.asarray(ct_idx, dtype=int), 0, nt - 1)] = 1.0
    morph = morphology_features(xy, ct_idx, int(n_types),
                                k=cfg.latent.morph_k,
                                density_sigma=cfg.latent.density_sigma)
    return np.hstack([xy, z[:, None], onehot,
                      np.asarray(morph, dtype=np.float64)]).astype(np.float64)


def training_matrix(stack, n_types, cfg):
    """(F, Y) over every training cell. Y is log-normalized expression (n, G)."""
    Fs, Ys = [], []
    for s in stack.slices:
        Fs.append(build_features(s.coords_xy, s.z_values, s.cell_type_indices,
                                 n_types, cfg))
        Ys.append(np.asarray(s.expression, dtype=np.float64))
    return np.vstack(Fs), np.vstack(Ys)


# ── run ──────────────────────────────────────────────────────────────────────


def run_method(adata, targets, gene_names, args):
    from spatialcpav14 import SpatialCPAv14

    train_mask = np.ones(adata.n_obs, dtype=bool)
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", train_mask, seed=args.seed)
    print(f"  cell types: {None if cell_type_names is None else len(cell_type_names)}")

    stack = _V14W.build_stack(adata, ct_all)
    if sum(s.n_spots for s in stack.slices) < 8 or stack.n_slices < 2:
        print("  SKIP: need >=2 sections and >=8 cells")
        return {}, {}

    cfg = build_v14_config(args)
    g = cfg.generation
    print(f"  epochs(A={cfg.train.pretrain_epochs},B={cfg.train.epochs}), "
          f"pos={g.position_mode}, selection={g.selection}, "
          f"edit_w={g.edit_weight}, blend={g.ground_blend_flow}, k={g.ground_k}, "
          f"ground={g.ground_expression}, counts={g.output_counts}")
    if g.edit_weight == 0.0:
        print("  edit_weight=0 -> v14's baseline emits a VERBATIM real profile; "
              "the learner replaces exactly that")

    gen = SpatialCPAv14(stack, gene_names=gene_names,
                        cell_type_names=cell_type_names, cfg=cfg)
    # Matched by config.invalid_log_markers, exactly as for spatialcpav14_gen.
    print(f"  flow-matching model trained: {gen.trained}")

    n_types = int(getattr(gen, "n_types", 0) or 0)
    Ftr, Ytr = training_matrix(stack, n_types, cfg)
    print(f"  learner fit: {args.learner} on {Ftr.shape[0]} training cells x "
          f"{Ftr.shape[1]} features -> {Ytr.shape[1]} genes (log-normalized target)")
    tr_xy = tr_z = tr_type = tr_sec = None
    tr_field = None
    Yfit = Ytr
    if args.emit == "donor":
        tr_xy, tr_z, tr_type, tr_sec = ML.training_index(stack)
        if args.field_mode == "residual":
            # Leave-section-out, because a generated section has no
            # neighbour in its own plane and the training analogue must
            # not either (see ML.training_field).
            tr_field = ML.training_field(stack, Ytr, tr_xy, tr_type,
                                         tr_sec, args.field_k)
            Yfit = Ytr - tr_field
            print(f"    field-mode residual: learner fits the correction "
                  f"to the interpolated flank field")
    model = ML.make_learner(args.learner, args)
    t_fit = time.time()
    model.fit(Ftr, Yfit)
    fit_seconds = time.time() - t_fit
    ML.freeze_for_determinism(model)
    print(f"    fit_seconds={fit_seconds:.1f}")
    ML.report_degeneracy(model, args.learner)

    if args.emit == "donor":
        # The pool's own field, cached once: the noise floor and every
        # per-gene repair band are measured against it.
        args._pool_pred = np.asarray(model.predict(Ftr), dtype=np.float64)
        if tr_field is not None:
            args._pool_pred = args._pool_pred + tr_field
        args._pool_resid = np.abs(Ytr - args._pool_pred)
        print(f"    --emit donor: predictions used as a field target; "
              f"every emitted value stays a real measurement")

    results, predict_seconds = {}, 0.0
    for sec, z in targets:
        print(f"  {sec}: v14 layout/donor selection at z={z:.2f}, then "
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
        # v14's own output tail, verbatim (spatialcpav14/trainer.py:536-539).
        if args.emit == "donor":
            incumbent = ML.to_target_space(vs.expression, "log")
            q_type = (vs.cell_type_idx if vs.cell_type_idx is not None
                      else np.zeros(n, np.int64))
            cand0 = ML.flanking_rows(stack, tr_sec, float(vs.coords[0, 2]))
            if args.field_mode == "residual":
                pred = pred + ML.local_field(
                    vs.coords[:, :2], q_type, tr_xy, tr_type, Ytr,
                    np.where(cand0)[0], args.field_k)
            field = pred
            pred, dstats = ML.select_donor_by_field(
                field, incumbent, Ytr, tr_xy, tr_type, cand0,
                vs.coords[:, :2], q_type, args)
            print(f"    field-guided: {dstats['n_swapped']}/{n} donors swapped "
                  f"({dstats['n_eligible']} eligible), median deviation "
                  f"{dstats['dev_before']:.4f} -> {dstats['dev_after']:.4f}, "
                  f"floor sigma0={dstats['sigma0']:.4f}")
            if args.repair_frac > 0:
                pred, rstats = ML.per_gene_repair(
                    pred, field, Ytr, np.where(cand0)[0], tr_xy, tr_type,
                    vs.coords[:, :2], q_type, args._pool_resid, args,
                    seed=args.seed)
                print(f"    per-gene repair: {rstats['n_repaired']:,}/"
                      f"{rstats['n_entries']:,} entries across "
                      f"{rstats['genes_touched']} genes")
        expr = np.clip(pred, 0.0, None)
        if cfg.generation.output_counts:
            expr = np.expm1(np.clip(expr, 0.0, 20.0))
        expr = expr.astype(np.float32)
        predict_seconds += time.time() - t_pred

        print(f"    -> {n} cells; emitted density {float((expr > 0).mean()):.4f} "
              f"(a regressor emits no zeros; this is the measurement)")
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V14W._to_dense_f32(expr)),
                        "coords": vs.coords.astype(np.float64),
                        "cell_type": cell_type}

    return results, {"fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                     "n_train_cells": int(Ftr.shape[0]),
                     "n_features": int(Ftr.shape[1])}


def build_parser():
    """v14's CLI verbatim, plus the learner. Checked by ``assert_v14_parity``."""
    p = argparse.ArgumentParser(
        description="SpatialCPA-v14 with a classical ML expression head "
                    "(layout, donor selection and every v14 flag unchanged)")
    _v2_io.add_v2_args(p)
    ML.add_learner_args(p)
    # ── v14's own flags, defaults identical to run_spatialcpav14.py ──────────
    p.add_argument("--epochs", type=int, default=160, help="Phase B flow-matching epochs")
    p.add_argument("--pretrain-epochs", type=int, default=60, help="Phase A encoder epochs")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--latent-dim", type=int, default=32, help="expression latent dim d_e")
    p.add_argument("--joint-dim", type=int, default=48, help="joint latent dim d")
    p.add_argument("--ode-steps", type=int, default=12, help="Euler steps for the sampling ODE")
    p.add_argument("--ensemble", type=int, default=4, help="initial noises marginalized per query")
    p.add_argument("--position-mode", default="flanking", choices=["morph", "flanking", "nearest"])
    p.add_argument("--displacement-scale", type=float, default=0.5,
                   help="scale on the flow-decoded displacement field (morph mode)")
    p.add_argument("--ground-blend-flow", type=float, default=0.20,
                   help="fraction of cells re-grounded to the flow-latent pick")
    p.add_argument("--ground-k", type=int, default=8,
                   help="number of spatially-nearest real candidate cells the flow "
                        "latent selects among when grounding each spot")
    p.add_argument("--edit-weight", type=float, default=0.25,
                   help="blend toward the flow-decoded profile (0 = pure real exemplar). "
                        "Affects v14's DONOR-SELECTION arm only; the emitted values come "
                        "from the learner either way")
    p.add_argument("--selection", default="flow", choices=["flow", "nearest", "random"],
                   help="how a flow-driven cell picks among its candidates")
    p.add_argument("--gap-dropout", type=float, default=None)
    p.add_argument("--z-sigma", type=float, default=None)
    p.add_argument("--morph-k", type=int, default=None,
                   help="kNN for the per-cell morphology features — note this also "
                        "shapes the learner's morphology columns, as it shapes v14's")
    p.add_argument("--n-context", type=int, default=None)
    p.add_argument("--flow-hidden", type=int, default=None)
    p.add_argument("--flow-layers", type=int, default=None)
    p.add_argument("--context-slices", type=int, default=None)
    p.add_argument("--type-placement", default=None, choices=["exemplar", "flow_smooth"])
    p.add_argument("--no-coherent-source", action="store_true")
    p.add_argument("--coherent-freq", type=float, default=None)
    p.add_argument("--no-ground", action="store_true", help="disable real-profile grounding")
    p.add_argument("--no-bio", action="store_true", help="ablate biology-informed losses")
    p.add_argument("--no-attention", action="store_true", help="ablate 3D attention")
    p.add_argument("--no-output-counts", action="store_true",
                   help="emit log1p-normalized (not count-like) expression")
    p.add_argument("--no-composition-match", action="store_true",
                   help="disable cell-type composition matching")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
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
    gene_names = adata.var_names.tolist()
    et = _V14W._normalize_expression(adata)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections; expression_type={et}")

    print(f"Running v14_{args.learner} for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results, info = run_method(adata, targets, gene_names, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed,
        "ablation": "expression head only; v14 layout, donor selection, type "
                    "placement and composition matching unchanged",
        "expression_framing": "regress log-normalized expression on "
                              "(x, y, z, one-hot type, morphology_features)",
        "target_scale": "log1p-normalized; emitted through v14's expm1 count tail",
        "fit_granularity": "once per volume, training sections only",
        "epochs": args.epochs, "pretrain_epochs": args.pretrain_epochs,
        "latent_dim": args.latent_dim, "joint_dim": args.joint_dim,
        "ode_steps": args.ode_steps, "ensemble": args.ensemble,
        "position_mode": args.position_mode,
        "edit_weight": args.edit_weight,
        "ground_blend_flow": args.ground_blend_flow,
        "ground_k": args.ground_k, "selection": args.selection,
        "gap_dropout": args.gap_dropout, "z_sigma": args.z_sigma,
        "morph_k": args.morph_k, "n_context": args.n_context,
        "flow_hidden": args.flow_hidden, "flow_layers": args.flow_layers,
        "coherent_source": not args.no_coherent_source,
        "flow_matching": True, "generation_only": True,
        **ML.learner_params(args.learner, args),
        **info,
    }
    _v2_io.write_prediction_h5(
        results, gene_names, target_sections, method_params, wall,
        args.output, f"v14_{args.learner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
