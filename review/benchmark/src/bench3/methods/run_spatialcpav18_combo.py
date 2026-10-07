"""SpatialCPA-v18-combo — stack3d_flow_cv and the published v18 as one family
(methods ``spatialcpav18_gen_flow_combo*``).

The published v18 configuration ("stack3d") and ``spatialcpav18_gen_flow_cv``
("flow_cv") differ in two mechanisms only; everything else (type vote,
composition matching, gene mix, raw output) is shared:

* **flank weight w** — the share of tissue area taken from the flank flow_cv
  chooses. w = 1 copies that flank whole (flow_cv); w < 1 lays the section out
  with v18's own coherent-patch field (smooth patches from both flanks, the
  published layout), giving the chosen flank area w. w = 0.5 is stack3d's
  50/50 split at the alternating hold-out.
* **re-grounding strength beta** — the fraction of cells offered to v18's
  flow-guided re-grounding (query = the flow's prediction ê, also used by the
  type vote and composition matching, as published). beta = 0 keeps every copied
  cell (flow_cv); beta = 1 is the published ``--ground-blend-flow 1.0``.

So flow_cv is the corner (w = 1, beta = 0) and the published configuration's two
mechanisms are the corner (w = 0.5, beta = 1). The direction (which flank is
"chosen") is flow_cv's own decision: the flow's flank margin against the
threshold delta calibrated on training folds.

Selection, on the same leave-one-training-section-out folds (each fold model
kept for both passes):

1. delta, exactly as flow-cv (``calibrate_delta``).
2. w in ``COMBO_W_GRID`` at beta = 0, then beta in ``COMBO_BETA_GRID`` at the
   chosen w. A step is taken only if its total fold score beats the current
   point by more than ``SNR_MIN``; ties keep the point closer to flow_cv.

The fold score is a signal-to-noise composite, fixed before any run:
``sum_k sign_k * (metric_k - metric_k(current)) / SIGMA[k]`` over the flow-cv
metrics, with SIGMA the run-to-run sd of each metric measured on five seeds of
the paired baseline (REVIEW_NOTES §12 runs). Unlike wins − losses it keeps effect
size and down-weights noisy metrics (gene detection's sd is ~10-40x the others').

Every random draw after the layout comes from a stage-specific generator
(common random numbers, as in h-cv); re-grounding draws from its own
(``SplitRNG``). The paired baseline is this method at (w = 1, beta = 0):
flow_cv with these streams.

Arms (``--combo-arm``):
``flow``       the method above
``untrained``  every flow use (flank margins, ê) from a re-initialised flow
``noflow``     no network: direction = v18's rule flank, re-grounding query =
               the t-weighted mean latent of the 8 nearest cells of each flank

Built on ``run_spatialcpav18_flow.py`` and v18, neither of which is edited.
"""

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18_flow as _F    # noqa: E402  (the flow wrapper; not modified)

_V18W, _v2_io, leakage_guard, torch = _F._V18W, _F._v2_io, _F.leakage_guard, _F.torch
_FlowBase = _F.SpatialCPAv18Flow        # bound once: _fold_pass rebinds the module name

METHOD_NAME = {"flow": "spatialcpav18_gen_flow_combo",
               "untrained": "spatialcpav18_gen_flow_combo_untrained",
               "noflow": "spatialcpav18_gen_flow_combo_noflow"}
COMBO_W_GRID = (1.0, 0.75, 0.5)            # 1 = flow_cv's whole-flank copy
COMBO_BETA_GRID = (0.0, 0.25, 0.5, 1.0)    # 0 = flow_cv's no re-grounding
# Run-to-run sd of each flow-cv metric: mean over the 4 designs of the sd across
# seeds 1-5 of the paired rho = 0 baseline (reproduced/seed_layout/seed*/base,
# REVIEW_NOTES §12), computed before any combo run.
SIGMA = {
    "embedding_mixing_pca": 0.001616,
    "morans_pearson": 0.001116,
    "morans_mae": 0.0009241,
    "marker_depth_r": 0.004012,
    "marker_field_r": 0.001512,
    "celltype_localization": 0.003078,
    "rare_celltype_localization": 0.006684,
    "gene_detection_spearman": 0.0488,
}
# A step must beat the current point by this much total fold score.
SNR_MIN = 1.0
INTERP_K = 8


def snr_score(m, base):
    """Signal-to-noise composite of metrics ``m`` against ``base`` (flow-cv metric
    keys, directions from ``_F.CV_METRICS``); missing values contribute 0."""
    s = 0.0
    for k, sign in _F.CV_METRICS:
        a, b = m.get(k), base.get(k)
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            continue
        s += sign * (float(a) - float(b)) / SIGMA[k]
    return s


def choose_step(current, totals):
    """The grid point to move to: the best total score if it beats ``current``'s
    by more than ``SNR_MIN``; otherwise ``current``. ``totals``: {point: score}
    (scores relative to the same reference). Ties keep the earlier point in the
    grid order (closer to flow_cv)."""
    best = current
    for p, v in totals.items():
        if v - totals[current] > SNR_MIN and v > totals[best]:
            best = p
    return best


class SpatialCPAv18Combo(_FlowBase):
    """flow_cv with a coherent-patch flank weight w and v18 re-grounding beta."""

    flank_select = "combo"           # not a parent mode
    direction = "flow"               # "flow" (flow-cv) | "rule" | "lower" | "upper"
    query_source = "flow"            # "flow" (ê) | "interp" (no network)
    combo_w = 1.0
    combo_beta = 0.0

    def use_untrained_flow(self):
        """Replace the trained flow (vector field + context module) by a seeded
        random re-initialisation, for every later flow use."""
        self.vfield, self.ctxmod = self._untrained_flow()

    def _generate(self, z):
        self._z_current = float(z)
        self._query = None
        self._stage_rng = {k: np.random.default_rng([int(self.cfg.seed), v])
                           for k, v in _F.STAGE_STREAMS.items()}
        return super(_FlowBase, self)._generate(z)                # v18's own

    # -- layout: direction from flow-cv, then w ----------------------------------
    def _resample_layout(self, lower, upper, t, n_target, rng):
        rule_lower = t <= 0.5
        if self.direction == "flow":
            chose_lower, info = self._flow_cv_choose_flank(lower, upper, rule_lower)
        elif self.direction in ("lower", "upper"):
            chose_lower, info = self.direction == "lower", {"forced": self.direction}
        else:
            chose_lower, info = rule_lower, {"rule": True}
        w = float(self.combo_w)
        if w >= 1.0:                                   # whole chosen flank (flow_cv)
            saved = self.flank_select
            self.flank_select = "lower" if chose_lower else "upper"
            try:
                anchor, src = super()._resample_layout(lower, upper, t, n_target, rng)
            finally:
                self.flank_select = saved
            entry = self.flank_log[-1]
        else:                                          # v18 coherent patches
            t_eff = (1.0 - w) if chose_lower else w    # area share of the upper flank
            anchor, src = super(_FlowBase, self)._resample_layout(lower, upper, t_eff,
                                                                  n_target, rng)
            self._flanks, self._t_current = (lower, upper), float(t)
            entry = {}
            self.flank_log.append(entry)
        entry.update(info)
        entry.update(z=self._z_current, t=float(t), rule_lower=bool(rule_lower),
                     chose_lower=bool(chose_lower), combo_w=w,
                     combo_beta=float(self.combo_beta), direction=self.direction,
                     query_source=self.query_source)
        return anchor, src

    # -- re-grounding: beta -------------------------------------------------------
    def _ground(self, anchor, anchor_src, e_hat, pool_nxy, pool_e, pool_expr, pool_type, rng):
        beta = float(self.combo_beta)
        if beta == 0.0:                                # flow_cv: keep every copied cell
            return super()._ground(anchor, anchor_src, e_hat, pool_nxy, pool_e,
                                   pool_expr, pool_type, rng)
        q = e_hat if self.query_source == "flow" else self._interp_e(anchor, pool_nxy, pool_e)
        self._query = np.asarray(q, dtype=np.float64)  # the vote / composition query too
        saved = self.cfg.ground_blend_flow
        self.cfg.ground_blend_flow = beta
        try:
            out = super(_FlowBase, self)._ground(anchor, anchor_src, self._query, pool_nxy,
                                                 pool_e, pool_expr, pool_type,
                                                 _F.SplitRNG(rng, int(self.cfg.seed)))
        finally:
            self.cfg.ground_blend_flow = saved
        if self.flank_log:
            self.flank_log[-1]["regrounded_frac"] = float(np.mean(out[2] != anchor_src))
        return out

    def _interp_e(self, anchor, pool_nxy, pool_e):
        lower, _upper = self._flanks
        n_lo, t = lower.n_spots, self._t_current
        k = min(INTERP_K, n_lo, len(pool_nxy) - n_lo)
        _, a = cKDTree(pool_nxy[:n_lo]).query(anchor, k=k)
        _, b = cKDTree(pool_nxy[n_lo:]).query(anchor, k=k)
        a, b = np.asarray(a).reshape(len(anchor), k), np.asarray(b).reshape(len(anchor), k)
        return (1.0 - t) * pool_e[:n_lo][a].mean(1) + t * pool_e[n_lo:][b].mean(1)


def _configure(model, arm):
    if arm == "untrained":
        model.use_untrained_flow()
    if arm == "noflow":
        model.direction, model.query_source = "rule", "interp"


def _fold_pass(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg, arm,
               forced_wb=None):
    """Both selection steps on shared fold models. Returns (delta, w, beta, record).
    With ``forced_wb`` only delta is calibrated."""
    _F.SpatialCPAv18Flow = SpatialCPAv18Combo      # _fold_models builds by this name
    try:
        folds = list(_F._fold_models(adata, gene_names, X_log, X_raw, ct_all,
                                     cell_type_names, cfg))
    finally:
        _F.SpatialCPAv18Flow = _FlowBase
    rec = {"arm": arm, "n_folds": len(folds), "sigma": SIGMA, "snr_min": SNR_MIN}

    def synth(fold, z, score, w, beta):
        fold.combo_w, fold.combo_beta = w, beta
        return score(fold.generate_virtual_slice(z=z))

    # step 1 — delta (flow-cv), only when the direction comes from the flow
    delta = float("inf")
    if arm != "noflow":
        fl = []
        for sec, z, fold, score in folds:
            assert isinstance(fold, SpatialCPAv18Combo)
            _configure(fold, arm)
            lower, upper = fold.stack.pick_flanking_slices(z)
            t = fold._tp_frac(z, lower.z_center, upper.z_center)
            rule_lower = t <= 0.5
            fold._z_current = z
            dist = fold._flow_flank_distances(lower, upper, generator=fold._cv_generator())
            rule, other = ("lower", "upper") if rule_lower else ("upper", "lower")
            scored = {}
            for side in (rule, other):
                fold.direction = side
                scored[side] = synth(fold, z, score, 1.0, 0.0)
            gain, _ = _F.fold_gain(scored[rule], scored[other])
            fl.append(_F.Fold(sec, z, dist[rule] - dist[other], gain, {}))
            fold.direction = "flow"
            print(f"    fold {sec}: flank margin {dist[rule] - dist[other]:+.4f}, "
                  f"other-flank gain {gain:+d}")
        delta, dgain = _F.calibrate_delta(fl)
        rec["delta"] = _F._json_num(delta)
        rec["delta_gain"] = int(dgain)
        print(f"  combo ({arm}): delta={delta:.6g} (validated gain {dgain:+d})")
    else:
        for _, _, fold, _ in folds:
            _configure(fold, arm)
    if forced_wb is not None or not folds:
        w, beta = forced_wb if forced_wb is not None else (1.0, 0.0)
        return delta, w, beta, rec
    for _, _, fold, _ in folds:
        fold.cv_delta = delta

    # step 2 — w at beta 0, then beta at the chosen w (scores vs the current point)
    base = {sec: synth(fold, z, score, 1.0, 0.0) for sec, z, fold, score in folds}
    tot_w = {1.0: 0.0}
    for w in COMBO_W_GRID[1:]:
        tot_w[w] = sum(snr_score(synth(fold, z, score, w, 0.0), base[sec])
                       for sec, z, fold, score in folds)
    w_star = choose_step(1.0, tot_w)
    print(f"  combo ({arm}): fold score by w {tot_w} -> w={w_star}")
    ref = (base if w_star == 1.0 else
           {sec: synth(fold, z, score, w_star, 0.0) for sec, z, fold, score in folds})
    tot_b = {0.0: 0.0}
    for b in COMBO_BETA_GRID[1:]:
        tot_b[b] = sum(snr_score(synth(fold, z, score, w_star, b), ref[sec])
                       for sec, z, fold, score in folds)
    b_star = choose_step(0.0, tot_b)
    print(f"  combo ({arm}): fold score by beta at w={w_star} {tot_b} -> beta={b_star}")
    rec.update(score_by_w={str(k): v for k, v in tot_w.items()},
               score_by_beta={str(k): v for k, v in tot_b.items()})
    return delta, w_star, b_star, rec


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", np.ones(adata.n_obs, dtype=bool), seed=args.seed)
    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, [], None
    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"
    forced = None
    if args.combo_w is not None or args.combo_beta is not None:
        forced = (1.0 if args.combo_w is None else float(args.combo_w),
                  0.0 if args.combo_beta is None else float(args.combo_beta))
        print(f"  combo: (w, beta)={forced} FORCED (delta still calibrated; a diagnostic)")
    delta, w, beta, rec = _fold_pass(adata, gene_names, X_log, X_raw, ct_all,
                                     cell_type_names, cfg, args.combo_arm, forced)
    rec.update(w=w, beta=beta, forced=forced is not None)
    gen = SpatialCPAv18Combo(stack, gene_names=gene_names, cell_type_names=cell_type_names,
                             cfg=cfg)
    gen.flank_log = []
    _configure(gen, args.combo_arm)
    gen.cv_delta, gen.combo_w, gen.combo_beta = delta, w, beta
    print(f"  flow-matching model trained: {gen.trained}")
    results = {}
    for sec, z in targets:
        vs = gen.generate_virtual_slice(z=z)
        f = gen.flank_log[-1]
        print(f"  {sec}: z={z:.2f} chose {'lower' if f['chose_lower'] else 'upper'} "
              f"(rule {'lower' if f['rule_lower'] else 'upper'}), w={w}, beta={beta}, "
              f"re-grounded {100 * f.get('regrounded_frac', 0.0):.1f}% "
              f"-> {vs.coords.shape[0]} cells")
        n = vs.coords.shape[0]
        if n == 0:
            continue
        cell_type = (vs.cell_type.astype(str) if vs.cell_type is not None
                     else np.array(["NA"] * n))
        results[sec] = {"X": sp.csr_matrix(_V18W._to_dense_f32(vs.expression)),
                        "coords": vs.coords.astype(np.float64), "cell_type": cell_type}
    return results, gen.flank_log, rec


def main():
    p = _F.build_parser()
    p.description = "SpatialCPA-v18-combo wrapper"
    p.add_argument("--combo-arm", default="flow", choices=["flow", "untrained", "noflow"])
    p.add_argument("--combo-w", type=float, default=None,
                   help="force the flank weight (diagnostic; delta still calibrated)")
    p.add_argument("--combo-beta", type=float, default=None,
                   help="force the re-grounding strength (diagnostic)")
    args = p.parse_args()
    if args.flank_select != "flow" or args.position_mode != "flanking":
        print("ERROR: this method sets its own layout; do not pass --flank-select or "
              "--position-mode", file=sys.stderr)
        return 2
    if not _V18W.check_environment():
        return 1
    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    gene_names = list(adata.var_names)
    X_raw = _V18W._to_dense_f32(adata.X)
    _V18W._normalize_expression(adata)
    X_log = _V18W._to_dense_f32(adata.X)
    t0 = time.time()
    results, flank_log, rec = run_method(adata, targets, gene_names, X_log, X_raw, args)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1
    params = {"seed": args.seed, "combo": rec,
              "flank_choices": [{k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in f.items()} for f in flank_log],
              "flow_matching": True, "generation_only": True}
    _v2_io.write_prediction_h5(results, gene_names, target_sections, params, wall,
                               args.output, METHOD_NAME[args.combo_arm])
    return 0


if __name__ == "__main__":
    sys.exit(main())
