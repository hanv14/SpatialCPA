"""T09's measurements: the joint gate, the calibrators, and the boundary regime.

Runs the whole of ``specs/09`` at the budgets the paper needs, on the synthetic fixture, and
writes ``reports/config_selection_synthetic.md``:

1. **the joint gate** — all four cells of ``{1x, 2x} x {weights off, spec weights}``, each
   fitted at the budget it names, scored on internal LOSO over training sections. This is the
   decision T08 deferred here, and every cell is reported, not only the winner;
2. **coordinate descent** over ``layout_mode`` / ``prior_mode`` / ``expr_mode`` /
   ``text_emb_mode`` from the winning cell, at a reduced budget;
3. **the calibrators** on the selected model — ``ell_xy`` against the flanking sections'
   Moran's I, ``ell_z`` against the observed between-section correlation (open risk R1), the
   derived retrieval z window, and the per-module Moran's I diagnostic (SPEC_QUESTIONS A2);
4. **the boundary regime** (open risk R3) — the uncertainty gate's own latent variance at the
   stack's ends against its interior, which ``specs/09`` §1 asks to be reported separately
   and says is itself a finding if it is *not* elevated;
5. **the definition of done** — the selected config against ``resample`` mode and against the
   independent-donor baseline on the six metrics.

    python scripts/t09_report.py [--steps 1200] [--folds 3] [--out reports/...]

Nothing here touches a held-out section: every fit, every score and every calibration takes
the ``TrainingVolume`` that ``split_holdout`` produced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import to_xyz
from spatialcpav25_gen.eval.baselines import IndependentDonorBaseline
from spatialcpav25_gen.infer.calibrate import (
    CalibrationNotAppliedWarning,
    apply_lengthscale,
    calibrate_anchor_weight,
    calibrate_detection,
    calibrate_lengthscale,
    calibrate_retrieval_window,
)
from spatialcpav25_gen.infer.generate import (
    emitted_counts,
    generate_section,
    latent_uncertainty,
)
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.losses.metric_aware import profile_axis
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow
from spatialcpav25_gen.train.select import (
    METRIC_NAMES,
    ScoreCache,
    calibration_chunks,
    full_budget_gate_cells,
    module_morans_agreement,
    run_selection,
    section_scores,
    selection_folds,
    write_selection_report,
)
from tests.fixtures.synthetic import make_synthetic_volume
from tests.test_expression import build_embeddings, expression_cfg

SEED = 20260819


def fit(cfg: Config, training, volume, *, steps: int) -> CTFFlow:
    """Fit one model on the training volume at ``steps`` optimiser steps."""
    data = TrainingData.build(training, cfg)
    model = CTFFlow(cfg, data, build_embeddings(cfg, volume), grf_seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(model, cfg, steps=steps, seed=SEED)
        if cfg.repulsion:
            model.repulsion = fit_repulsion(training, cfg, seed=SEED + 1)
    return model


def boundary_regime(model: CTFFlow, training, cfg: Config) -> dict[str, float]:
    """Latent variance at the stack's ends against its interior (open risk R3).

    The uncertainty gate already estimates a per-cell latent variance; ``specs/09`` §1 asks
    whether it is *elevated* where the evidence is one-sided, and says that if it is not, that
    is itself a finding. Measured at the real cells of the first, last and middle sections,
    each with its own section excluded from retrieval — the same exclusion generation uses.
    """
    out: dict[str, float] = {}
    for name, index in (("first", 0), ("middle", training.n_sections // 2), ("last", -1)):
        section = training.sections[index]
        xyz = to_xyz(section)[:512].astype(np.float64)
        cell_type = torch.from_numpy(np.asarray(section.cell_type[:512], dtype=np.int64))
        region = (
            None
            if section.region is None
            else torch.from_numpy(np.asarray(section.region[:512], dtype=np.int64))
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            neighbours, _ = model.data.index.query(xyz, {float(section.z)}, seed=SEED)
            uncertainty = latent_uncertainty(
                model, xyz, cell_type, region, neighbours, cfg, seed=SEED
            )
        out[name] = float(np.mean(uncertainty.variance))
    return out


def baseline_scores(model: CTFFlow, training, cfg: Config) -> dict[str, dict[str, float]]:
    """The definition-of-done comparison: the selected config vs `resample` vs donor mixing.

    All three are scored on the *same* LOSO folds with the same metric code. The
    independent-donor baseline is given the method's own generated layout, so the comparison
    is about the expression path alone — which is what T06 built that baseline to isolate.
    """
    axis = profile_axis(training, cfg)
    arms: dict[str, list[dict[str, float]]] = {"method": [], "resample": [], "donor": []}
    resample_cfg = cfg.replace(layout_mode="resample", expr_mode="cross-mix")
    for index, hidden in enumerate(selection_folds(training, cfg)):
        plane = section_plane(hidden)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            generated = generate_section(
                model, plane, training, cfg, SEED + index, exclude_z={float(hidden.z)}
            )
            resampled = generate_section(
                model, plane, training, resample_cfg, SEED + index, exclude_z={float(hidden.z)}
            )
        types = np.asarray(
            [training.celltype_names.index(v) for v in generated.obs[cfg.celltype_key]],
            dtype=np.int64,
        )
        xyz = np.asarray(generated.obsm["xyz"], dtype=np.float64)
        order = np.argsort(np.abs(training.z_values.astype(np.float64) - float(hidden.z)))
        pool = [
            training.sections[int(i)]
            for i in order
            if abs(float(training.sections[int(i)].z) - float(hidden.z)) > 1e-6
        ][:2]
        donor = IndependentDonorBaseline.from_sections(pool, cfg)
        donor_counts = donor.generate(xyz, seed=SEED + index, cfg=cfg).numpy().astype(np.float64)
        arms["method"].append(
            section_scores(
                emitted_counts(generated),
                np.asarray(generated.obsm["xyz"], dtype=np.float64)[:, :2],
                types,
                hidden,
                axis,
                cfg,
                n_types=len(training.celltype_names),
            )
        )
        resample_types = np.asarray(
            [training.celltype_names.index(v) for v in resampled.obs[cfg.celltype_key]],
            dtype=np.int64,
        )
        arms["resample"].append(
            section_scores(
                emitted_counts(resampled),
                np.asarray(resampled.obsm["xyz"], dtype=np.float64)[:, :2],
                resample_types,
                hidden,
                axis,
                cfg,
                n_types=len(training.celltype_names),
            )
        )
        arms["donor"].append(
            section_scores(
                donor_counts,
                np.asarray(generated.obsm["xyz"], dtype=np.float64)[:, :2],
                types,
                hidden,
                axis,
                cfg,
                n_types=len(training.celltype_names),
            )
        )
    return {
        arm: {name: float(np.mean([f[name] for f in folds])) for name in METRIC_NAMES}
        for arm, folds in arms.items()
    }


def main() -> None:
    """Run the selection, the calibration and the two risk measurements; write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200, help="the 1x budget")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--out", default="reports/config_selection_synthetic.md")
    parser.add_argument(
        "--checkpoint",
        default="reports/selection_checkpoint_synthetic.csv",
        help=(
            "Per-cell checkpoint CSV. The merged full-budget gate is 18 fits at the selected "
            "budget, so a run that loses them to an interrupted session is not usable: each "
            "scored cell is flushed here as it completes and a re-run skips what is already "
            "recorded. Re-run the identical command to resume."
        ),
    )
    parser.add_argument(
        "--commit-checkpoint",
        action="store_true",
        help=(
            "Commit and push the checkpoint after every scored cell, so the run survives "
            "losing the machine rather than merely the process."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the cells still to run and exit without fitting anything.",
    )
    parser.add_argument(
        "--expr-mode",
        default="zinb-flow",
        help=(
            "The expression path to calibrate under. 'cross-mix' never evaluates the flow, so "
            "the calibration objective is constant in ell and the calibrators refuse it; the "
            "default runs the arm on a path where ell is live."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Skip the selection and calibrate an already-selected config (the YAML the "
            "selector persists). Used to re-measure the calibration after the first run "
            "showed the selected prior_mode makes ell inert."
        ),
    )
    args = parser.parse_args()

    volume, _ = make_synthetic_volume(seed=0)
    base = expression_cfg().replace(
        train_steps=int(args.steps),
        selection_n_folds=int(args.folds),
        selection_passes=int(args.passes),
    )
    training, _ = split_holdout(volume, "alternating", 0, base)

    def commit_checkpoint(label: str) -> None:
        """Commit and push the checkpoint CSV after a cell, so a lost container costs one cell."""
        path = str(args.checkpoint)
        subprocess.run(["git", "add", path], check=False)
        subprocess.run(
            ["git", "commit", "-q", "-m", f"T09: selection checkpoint — {label}"], check=False
        )
        subprocess.run(
            ["git", "push", "-q", "origin", "claude/task-t09-implementation-5zgrb7"], check=False
        )

    checkpoint = ScoreCache(
        args.checkpoint, on_write=commit_checkpoint if args.commit_checkpoint else None
    )
    print(f"checkpoint {args.checkpoint}: {len(checkpoint)} cell(s) already recorded", flush=True)

    if args.dry_run:
        cells = full_budget_gate_cells(base)
        steps = round(float(base.selection_budget_multiple) * int(base.train_steps))
        remaining = [
            label
            for label, overrides in cells
            if checkpoint.get(base.replace(**overrides, train_steps=steps), steps) is None
        ]
        print(f"{len(remaining)} of {len(cells)} full-budget cells remain (at {steps} steps):")
        for label in remaining:
            print(f"  {label}")
        return

    result = None
    if args.config is None:
        started = time.time()
        result = run_selection(
            training,
            base,
            seed=SEED,
            embeddings=lambda cfg: build_embeddings(cfg, volume),
            dataset="synthetic",
            checkpoint=checkpoint,
        )
        print(f"selection finished in {time.time() - started:.0f} s", flush=True)
        for candidate in result.joint:
            print(
                f"joint {candidate.label}: steps={candidate.steps} rank={candidate.rank:.1f} "
                + json.dumps({k: round(v, 4) for k, v in candidate.scores.items()}),
                flush=True,
            )
        selected = result.config
    else:
        selected = Config.from_yaml(args.config)
        print(f"loaded the selected config from {args.config}", flush=True)

    # `ell` parameterises the GRF, so a calibration run under `prior_mode="iid"` measures
    # nothing (`calibrate_lengthscale` now refuses it outright). When the selector switches
    # the correlated prior off, the calibration arm is the same config with the prior it
    # parameterises switched back on — and the report says so.
    calibration_cfg = selected.replace(prior_mode="correlated", expr_mode=args.expr_mode)
    if calibration_cfg.prior_mode != selected.prior_mode:
        print(
            f"calibrating under prior_mode='correlated' (selected: {selected.prior_mode!r}) — "
            "ell has no effect on an i.i.d. prior",
            flush=True,
        )
    if calibration_cfg.expr_mode != selected.expr_mode:
        print(
            f"calibrating under expr_mode={args.expr_mode!r} (selected: "
            f"{selected.expr_mode!r}) — 'cross-mix' emits donor counts verbatim, so the flow "
            "and therefore ell never reach the output",
            flush=True,
        )
    model = fit(calibration_cfg, training, volume, steps=int(selected.train_steps))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calibration = calibrate_lengthscale(model, training, calibration_cfg, seed=SEED)
        window = calibrate_retrieval_window(training, selected)
        detection = calibrate_detection(model, training, calibration_cfg, seed=SEED, n_folds=3)
        anchor = calibrate_anchor_weight(model, training, calibration_cfg, seed=SEED, n_folds=2)
        modules = module_morans_agreement(model, training, calibration_cfg, seed=SEED)
        boundary = boundary_regime(model, training, calibration_cfg)
        arms = baseline_scores(model, training, calibration_cfg)

    # The step C30 added: a calibration only means something once it reaches the prior, and
    # only a converged axis is allowed to.
    with warnings.catch_warnings(record=True) as dropped:
        warnings.simplefilter("always", CalibrationNotAppliedWarning)
        applied_cfg = apply_lengthscale(calibration_cfg, calibration)
    print("calibration:", calibration.ell, calibration.status, calibration.ell_z_status)
    print(
        f"  applied: ell_xy {calibration_cfg.ell_xy:g} -> {applied_cfg.ell_xy:g} um, "
        f"ell_z {calibration_cfg.ell_z:g} -> {applied_cfg.ell_z:g} um"
    )
    for entry in dropped:
        if issubclass(entry.category, CalibrationNotAppliedWarning):
            print(f"  dropped: {entry.message}")
    print(
        f"  fitted ell {calibration.ell_fitted} | Moran I gen {calibration.i_gen:.4f} "
        f"vs target {calibration.i_target:.4f} in {calibration.iterations} iterations | "
        f"between-section r gen {calibration.z_achieved:.4f} vs observed "
        f"{calibration.z_target:.4f} | sections {list(calibration.section_ids)}"
    )
    print("window:", window.window, "max gap", window.max_gap_um)
    print("boundary latent variance:", json.dumps({k: round(v, 6) for k, v in boundary.items()}))
    print(
        "detection MAD before correction:",
        float(np.mean(np.abs(detection.detection_gen - detection.detection_real))),
    )
    print("anchor w(v):", list(np.round(anchor.y, 3)), "at v", list(np.round(anchor.x, 5)))
    for arm, scores in arms.items():
        print(f"{arm}: " + json.dumps({k: round(v, 4) for k, v in scores.items()}), flush=True)

    path = Path(args.out)
    if result is not None:
        path = write_selection_report(
            result, args.out, calibration=calibration, window=window, modules=modules
        )
    extra = [
        "",
        f"*Calibration arm: `prior_mode={calibration_cfg.prior_mode}` "
        f"(selected: `{selected.prior_mode}`), `train_steps={selected.train_steps}`.*",
        # In --config mode no selection report was written, so the calibration tables that
        # write_selection_report would have emitted are rendered here from the same helper.
        *(calibration_chunks(calibration, window, modules) if result is None else []),
        "",
        "## Open risk R3 — the boundary is a different regime",
        "",
        "Mean latent variance of the uncertainty gate at the stack's ends against its "
        "interior, on real cells with each section excluded from its own retrieval pool.",
        "",
        "| position | mean latent variance |",
        "|---|---|",
        *[f"| {name} | {value:.6f} |" for name, value in boundary.items()],
        "",
        "## Definition of done — the selected config against the two fallbacks",
        "",
        "| arm | " + " | ".join(METRIC_NAMES) + " |",
        "|" + "|".join(["---"] * (len(METRIC_NAMES) + 1)) + "|",
        *[
            f"| {arm} | " + " | ".join(f"{scores[name]:.4f}" for name in METRIC_NAMES) + " |"
            for arm, scores in arms.items()
        ],
        "",
        f"Detection / dispersion calibration fitted on {list(detection.section_ids)}; "
        f"per-gene detection MAD before it "
        f"{float(np.mean(np.abs(detection.detection_gen - detection.detection_real))):.4f}.",
        "",
        f"Anchor weight w(v): {list(np.round(anchor.y, 3))} at variances "
        f"{list(np.round(anchor.x, 5))} — non-increasing by construction.",
    ]
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write("\n".join(extra) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
