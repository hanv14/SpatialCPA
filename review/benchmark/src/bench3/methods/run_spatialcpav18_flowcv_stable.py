"""SpatialCPA-v18-flow-cv-stable — flow_cv with a lower-variance switch decision
(method ``spatialcpav18_gen_flow_cv_stable``).

flow_cv switches a held-out section to the non-rule flank when the flow's flank
margin clears a threshold delta calibrated on leave-one-training-section-out
folds. Across seeds that decision was unstable (REVIEW_NOTES §13): the flow's
margins barely move between seeds, but each fold's evidence — "is the other
flank better here?" — came from ONE synthesis per side, scored by wins − losses
over 8 metrics, and any positive total switched. Here only that evidence
changes:

* **replicates** — each fold synthesizes its left-out section from the rule
  flank and from the other flank ``STABLE_REPS`` times; replicate r reseeds the
  generation randomness (``(seed, r)``) and is shared by the two sides, so each
  replicate is a paired comparison;
* **signal-to-noise score** — a replicate's gain is the composite
  ``sum_k sign_k * (other_k - rule_k) / SIGMA[k]`` (the run-to-run sds fixed in
  ``run_spatialcpav18_combo.SIGMA``), so noisy metrics no longer decide signs;
* **confidence gate** — a fold's gain is the replicate mean with its standard
  error; a threshold that switches the folds S is admissible only if the lower
  one-sided bound ``sum_S mean - Z * sqrt(sum_S se^2)`` is positive. delta is the
  largest threshold with the best admissible bound; none admissible → never
  switch (the rule flank, "nearest + no flow").

Output synthesis and everything else are flow_cv's, with the combo class's
stage-specific random streams; with the same decision the output equals
``spatialcpav18_gen_flow_combo`` at (w = 1, beta = 0) for the same seed.
Built on run_spatialcpav18_combo.py / run_spatialcpav18_flow.py and v18, none
of which is edited.
"""

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpav18_combo as _C   # noqa: E402  (not modified)

_F = _C._F
_V18W, _v2_io, leakage_guard = _F._V18W, _F._v2_io, _F.leakage_guard

METHOD_NAME = "spatialcpav18_gen_flow_cv_stable"
STABLE_REPS = 5          # paired replicate syntheses per fold and side
Z = 1.645                # one-sided 95% lower bound


def calibrate_delta_lcb(margins, means, ses, z=Z):
    """delta from fold (margin, mean gain, se). For each candidate threshold just
    below a fold margin, the switched folds are those with margin > threshold;
    their lower bound is ``sum(mean) - z * sqrt(sum(se^2))``. Returns
    ``(delta, lower_bound)`` for the largest threshold with the best positive
    bound, or ``(inf, 0.0)`` when no bound is positive."""
    best_lb, delta = 0.0, float("inf")
    margins = np.asarray(margins, float)
    means, ses = np.asarray(means, float), np.asarray(ses, float)
    for m in sorted(set(margins.tolist()), reverse=True):
        thr = float(np.nextafter(m, -np.inf))
        s = margins > thr
        lb = float(means[s].sum() - z * np.sqrt((ses[s] ** 2).sum()))
        if lb > best_lb:
            best_lb, delta = lb, thr
    return delta, best_lb


def _fold_evidence(adata, gene_names, X_log, X_raw, ct_all, cell_type_names, cfg):
    _F.SpatialCPAv18Flow = _C.SpatialCPAv18Combo
    try:
        folds = list(_F._fold_models(adata, gene_names, X_log, X_raw, ct_all,
                                     cell_type_names, cfg))
    finally:
        _F.SpatialCPAv18Flow = _C._FlowBase
    margins, means, ses, recs = [], [], [], []
    for sec, z, fold, score in folds:
        lower, upper = fold.stack.pick_flanking_slices(z)
        t = fold._tp_frac(z, lower.z_center, upper.z_center)
        rule, other = ("lower", "upper") if t <= 0.5 else ("upper", "lower")
        fold._z_current = z
        dist = fold._flow_flank_distances(lower, upper, generator=fold._cv_generator())
        margin = dist[rule] - dist[other]
        seed0 = int(fold.cfg.seed)
        gains = []
        try:
            for r in range(STABLE_REPS):
                fold.cfg.seed = seed0 * 1000 + r          # replicate r: both sides share it
                m = {}
                for side in (rule, other):
                    fold.direction, fold.combo_w, fold.combo_beta = side, 1.0, 0.0
                    m[side] = score(fold.generate_virtual_slice(z=z))
                gains.append(_C.snr_score(m[other], m[rule]))
        finally:
            fold.cfg.seed = seed0
        g = np.asarray(gains)
        mean, se = float(g.mean()), float(g.std(ddof=1) / np.sqrt(len(g)))
        print(f"    fold {sec}: flank margin {margin:+.4f}; other-flank SNR gain "
              f"{mean:+.1f} ± {se:.1f} (replicates {', '.join(f'{x:+.1f}' for x in g)})")
        margins.append(margin)
        means.append(mean)
        ses.append(se)
        recs.append({"section": sec, "margin": margin, "gain_mean": mean, "gain_se": se,
                     "gains": [float(x) for x in g]})
    return margins, means, ses, recs


def run_method(adata, targets, gene_names, X_log, X_raw, args):
    ct_all, cell_type_names = leakage_guard.build_labels_train_only(
        adata, "cell_type", np.ones(adata.n_obs, dtype=bool), seed=args.seed)
    stack = _V18W.build_stack(adata, X_log, X_raw, ct_all)
    if stack.n_slices < 2 or sum(s.n_spots for s in stack.slices) < 8:
        print("  SKIP: need >= 2 sections and >= 8 cells")
        return {}, [], None
    cfg = _V18W._build_config(args)
    cfg.position_mode = "flanking"
    margins, means, ses, recs = _fold_evidence(adata, gene_names, X_log, X_raw, ct_all,
                                               cell_type_names, cfg)
    delta, lb = calibrate_delta_lcb(margins, means, ses)
    print(f"  flow-cv-stable: delta={delta:.6g} (lower bound of the validated SNR gain "
          f"{lb:+.1f}; {'never switch' if delta == float('inf') else 'switch when margin > delta'})")
    rec = {"delta": _F._json_num(delta), "lower_bound": lb, "reps": STABLE_REPS, "z": Z,
           "folds": recs}
    gen = _C.SpatialCPAv18Combo(stack, gene_names=gene_names,
                                cell_type_names=cell_type_names, cfg=cfg)
    gen.flank_log = []
    gen.cv_delta, gen.combo_w, gen.combo_beta, gen.direction = delta, 1.0, 0.0, "flow"
    print(f"  flow-matching model trained: {gen.trained}")
    results = {}
    for sec, z in targets:
        vs = gen.generate_virtual_slice(z=z)
        f = gen.flank_log[-1]
        print(f"  {sec}: z={z:.2f} chose {'lower' if f['chose_lower'] else 'upper'} "
              f"(rule {'lower' if f['rule_lower'] else 'upper'}) -> {vs.coords.shape[0]} cells")
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
    p.description = "SpatialCPA-v18-flow-cv-stable wrapper"
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
    params = {"seed": args.seed, "flow_cv_stable": rec,
              "flank_choices": [{k: (round(v, 6) if isinstance(v, float) else v)
                                 for k, v in f.items()} for f in flank_log],
              "flow_matching": True, "generation_only": True}
    _v2_io.write_prediction_h5(results, gene_names, target_sections, params, wall,
                               args.output, METHOD_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
