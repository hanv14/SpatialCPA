"""Measure a gate where it is **live**, per fold, at full budget.

The selection run answered two of its four gates with a separation of exactly ``0.0000``.
That is not a tie: the merged gate had selected ``expr_mode="cross-mix"``, and
``infer/generate.py::_expression`` returns from ``_cross_mix`` before ``prior_latent``, the
flow, the decoder and the gene embeddings are ever reached — so ``prior_mode`` and
``text_emb_mode`` cannot change a single emitted count under it. The gate built to test the
open-vocabulary channel ran in the one configuration where it cannot be tested.

``train/select.py`` now refuses or re-orders that automatically (``inert_gates``). This script
is the *measurement* it displaces: one gate, scored under an incumbent the caller names, with
**per-fold** numbers rather than their mean.

Two things it reports that the selector's own table cannot.

**Per fold.** ``selection_folds`` returns the *interior* sections, so tier-1 STARmap's
four-section training stack gives **two** folds however large ``Config.selection_n_folds`` is.
Every gate decision in the selection therefore rests on a mean of two numbers, and a mean of
two hides which fold moved. ``fold_scores`` keeps them.

**Against the envelope.** R10's across-seed envelope is **0.0335** and
``Config.claim_tie_break_envelope`` rounds it to 0.04. A margin below that is not a result,
and a margin carried entirely by one of two folds is not one either.

Usage::

    # the measurement the selection could not make: medcpt vs lookup, decoder live
    python scripts/t09_audit_starmap.py --gate text_emb_mode --under expr_mode=zinb-flow

    # the substantive finding, checked per fold: does copying really beat generating?
    python scripts/t09_audit_starmap.py --gate expr_mode --options cross-mix zinb-flow

Fits are shared with the selection run through the same ``ScoreCache``-style keying: pass the
selection's ``--out`` directory and a cell already scored there at the same budget and config
is not refitted. Per-fold numbers always require the model, so a cell whose *scores* are
cached but whose weights are gone is refitted once; that is stated per cell as it runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import yaml
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow
from spatialcpav25_gen.train.select import (
    ALL_GATES,
    METRIC_NAMES,
    average_folds,
    fold_scores,
    inert_gates,
    selection_folds,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import (
    base_config,
    clamp_config_to_input,
    describe_text_channel,
    embeddings_factory,
    load_training_volume,
)

ENVELOPE = 0.0335
"""R10's measured across-seed envelope (``reports/envelope_synthetic.md``), 9 fits, 3 cells x
3 seeds. ``Config.claim_tie_break_envelope`` rounds it up to 0.04; the raw number is used here
so the margin can be read against both."""


def parse_under(values: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--under expects gate=option, got {item!r}")
        gate, option = item.split("=", 1)
        out[gate.strip()] = option.strip()
    return out


def fit(cfg: Config, volume, embeddings, *, seed: int, checkpoint: Path | None) -> CTFFlow:
    model = CTFFlow(cfg, TrainingData.build(volume, cfg), embeddings(cfg), grf_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(
            model,
            cfg,
            steps=int(cfg.train_steps),
            seed=seed,
            checkpoint=None if checkpoint is None else str(checkpoint),
        )
    if cfg.repulsion and cfg.layout_mode != "resample":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.repulsion = fit_repulsion(volume, cfg, seed=seed + 1)
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gate", required=True, choices=[g for g, _ in ALL_GATES])
    ap.add_argument("--options", nargs="*", default=None, help="default: the gate's own options")
    ap.add_argument(
        "--under",
        nargs="*",
        default=["expr_mode=zinb-flow"],
        help="gate=option fixing the incumbent this gate is measured under. The default puts "
        "the decoder live, which is where an expression-path gate can be measured at all",
    )
    ap.add_argument("--selected", default=None, help="selected.yaml to start from")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workdir", default="runs/audit")
    ap.add_argument("--out", default=None, help="default: reports/t09_audit_<gate>.md")
    ap.add_argument("--train-steps", type=int, default=None)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--gene-meta", default=None)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    ap.add_argument(
        "--fit-only",
        action="store_true",
        help="fit each option and stop, without scoring. The fit checkpoint is a resume "
        "point, so a later full run of the same command re-enters a finished fit as a no-op "
        "and goes straight to scoring — which is what lets the four arms of the two audits "
        "run as four concurrent processes instead of two sequential pairs",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="report whether the gate is live under --under, and the fold count, then exit",
    )
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    overrides: dict = {}
    for flag, field_name in (
        (args.text_cache, "text_cache_dir"),
        (args.gene_meta, "gene_meta_path"),
    ):
        if flag:
            overrides[field_name] = flag
    if args.train_steps is not None:
        overrides["train_steps"] = int(args.train_steps)
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)

    if args.selected:
        payload = yaml.safe_load(Path(args.selected).read_text())
        cfg = Config(**payload["config"]).replace(seed=args.seed, **overrides)
        source = str(args.selected)
    else:
        cfg = base_config(args.seed, **overrides)
        source = "Config defaults"
    cfg = clamp_config_to_input(cfg, paths.input)
    under = parse_under(args.under)
    cfg = cfg.replace(**under)

    volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    embeddings = embeddings_factory(volume)
    folds = selection_folds(volume, cfg)
    options = tuple(args.options) if args.options else dict(ALL_GATES)[args.gate]

    print(f"  config from {source}; measured under {under}")
    print(f"  gate {args.gate} over {list(options)} at {cfg.train_steps} steps, seed {args.seed}")
    print(f"  LOSO folds ({len(folds)}): {[s.section_id for s in folds]}")
    if len(folds) < 3:
        print(
            f"  ⚠ only {len(folds)} interior sections, so Config.selection_n_folds="
            f"{cfg.selection_n_folds} cannot be honoured. Every number below is a mean of "
            f"{len(folds)}; the per-fold table is the one to read."
        )
    print("  text channel: " + json.dumps(describe_text_channel(cfg, volume), default=str))

    if args.preflight:
        print("\n── is the gate live under this incumbent? ──")
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        from spatialcpav25_gen.train.select import FitScorer

        scorer = FitScorer(volume, embeddings, needs_repulsion=cfg.layout_mode != "resample")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dead = inert_gates(
                lambda c: scorer.inertness_probe(c, seed=args.seed),
                cfg,
                [(args.gate, options)],
                seed=args.seed,
            )
        if args.gate in dead:
            print(
                f"  ✗ {args.gate} is INERT under {under}: {list(dead[args.gate])} emit "
                "bitwise-identical counts. Measuring it here would report 0.0000 and mean "
                "nothing. Choose a different --under."
            )
            return 1
        print(f"  ✓ {args.gate} is live under {under}; the measurement is meaningful")
        print("\npreflight OK — nothing was fitted.")
        return 0

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for option in options:
        arm = cfg.replace(**{args.gate: option})
        t0 = time.time()
        print(f"\n── {args.gate}={option}  ({arm.content_hash()}) ──", flush=True)
        model = fit(
            arm,
            volume,
            embeddings,
            seed=args.seed,
            checkpoint=workdir / f"fit_{args.gate}_{option}_seed{args.seed}.pt",
        )
        if args.fit_only:
            print(f"  fitted in {time.time() - t0:.0f}s; --fit-only, not scoring", flush=True)
            continue
        anchor = None
        if arm.expr_mode == "auto-blend":
            from spatialcpav25_gen.infer.calibrate import calibrate_anchor_weight

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anchor = calibrate_anchor_weight(model, volume, arm, seed=args.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_fold = fold_scores(model, volume, arm, seed=args.seed, anchor=anchor)
        mean = average_folds(per_fold)
        rows.append(
            {
                "option": option,
                "dataset": paths.dataset,
                "holdout": paths.holdout,
                "n_cells": int(volume.n_cells),
                "n_genes": int(volume.n_genes),
                "n_sections": int(volume.n_sections),
                "train_steps": int(arm.train_steps),
                "expr_pca_dim": int(arm.expr_pca_dim),
                "under": dict(under),
                "config_hash": arm.content_hash(),
                "seconds": round(time.time() - t0, 1),
                "mean": mean,
                "per_fold": per_fold,
                "fold_ids": [s.section_id for s in folds],
            }
        )
        print(f"  fitted and scored in {time.time() - t0:.0f}s")
        for name in METRIC_NAMES:
            each = "  ".join(f"{f[name]:+.4f}" for f in per_fold)
            print(f"    {name:<24} mean {mean[name]:+.4f}   per fold [{each}]")

    if args.fit_only:
        print("\n--fit-only: every arm is fitted and checkpointed. Re-run without it to score.")
        return 0

    lines = _report(rows, args, cfg, under, folds, source, paths, volume)
    text = "\n".join(lines)
    print()
    print(text)
    out = Path(args.out or f"reports/t09_audit_{args.gate}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def _fold_spread(rows, metric: str) -> float:
    """The largest spread *within* one arm across folds, for the metric.

    The statistic the envelope cannot supply. R10's 0.0335 is an across-**seed** envelope
    measured on the fixture; it says nothing about how much a metric moves between the folds of
    *this* dataset. At n = 2 folds a between-arm margin smaller than the within-arm fold spread
    is not a result — the two arms are being separated by less than one arm moves on its own —
    and that is a distinct failure from the folds disagreeing in sign, which is what the report
    checked before. Measured on ``deep_starmap``: ``text_emb_mode``'s ``marker_depth_r`` margin
    is 0.1850 while ``lookup`` alone swings 0.2033 between ``section_3`` and ``section_5``.
    """
    worst = 0.0
    for r in rows:
        values = [f[metric] for f in r["per_fold"]]
        worst = max(worst, max(values) - min(values))
    return worst


def _report(rows, args, cfg, under, folds, source, paths, volume) -> list[str]:
    fold_ids = [s.section_id for s in folds]
    lines = [
        f"# T09 audit — `{args.gate}` measured where it is live, per fold",
        "",
        f"Dataset **`{paths.dataset}`**, holdout **`{paths.holdout}`** — "
        f"{volume.n_cells} training cells x {volume.n_genes} genes over "
        f"{volume.n_sections} sections. Config from {source}, "
        f"measured under **{', '.join(f'`{k}={v}`' for k, v in under.items())}**, "
        f"{cfg.train_steps} steps, seed {args.seed}.",
        "",
        "The selection could not make this measurement: it scored this gate under "
        "`expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the "
        "prior, the flow, the decoder and the gene embeddings are reached — so both options "
        "emitted bitwise-identical counts and the gate reported a separation of exactly "
        "**0.0000**. That is an absence of measurement, not a tie.",
        "",
        f"**Folds: {len(fold_ids)}** — {', '.join(f'`{s}`' for s in fold_ids)}. "
        "`selection_folds` takes the *interior* sections, so a four-section training stack "
        f"gives {len(fold_ids)} however large `Config.selection_n_folds` "
        f"({cfg.selection_n_folds}) is set. Read the per-fold columns, not the mean.",
        "",
        "| metric | " + " | ".join(f"`{r['option']}` mean" for r in rows),
    ]
    header = lines[-1]
    for sid in fold_ids:
        header += " | " + " | ".join(f"`{r['option']}` {sid}" for r in rows)
    lines[-1] = header + " | margin (mean) | vs 0.0335 | vs fold spread |"
    lines.append("|---" * (1 + len(rows) * (1 + len(fold_ids)) + 3) + "|")
    for name in METRIC_NAMES:
        cells = [f"{r['mean'][name]:+.4f}" for r in rows]
        for i in range(len(fold_ids)):
            cells += [f"{r['per_fold'][i][name]:+.4f}" for r in rows]
        margin = (
            abs(rows[0]["mean"][name] - rows[1]["mean"][name]) if len(rows) == 2 else float("nan")
        )
        verdict = (
            "—"
            if margin != margin
            else ("**inside**" if margin < ENVELOPE else f"{margin / ENVELOPE:.1f}x")
        )
        spread = _fold_spread(rows, name)
        vs_spread = (
            "—"
            if margin != margin or spread <= 0
            else (
                f"**{margin / spread:.1f}x**"
                if margin / spread >= 2.0
                else f"⚠ {margin / spread:.1f}x"
            )
        )
        lines.append(
            f"| `{name}` | "
            + " | ".join(cells)
            + f" | {'—' if margin != margin else f'{margin:.4f}'} | {verdict} | {vs_spread} |"
        )

    if len(rows) == 2:
        worst = max(METRIC_NAMES, key=lambda n: abs(rows[0]["mean"][n] - rows[1]["mean"][n]))
        margin = abs(rows[0]["mean"][worst] - rows[1]["mean"][worst])
        agree = all(
            (rows[0]["per_fold"][i][worst] - rows[1]["per_fold"][i][worst])
            * (rows[0]["mean"][worst] - rows[1]["mean"][worst])
            > 0
            for i in range(len(fold_ids))
        )
        lines += [
            "",
            f"**Largest separation: `{worst}` at {margin:.4f}** "
            f"({margin / ENVELOPE:.1f}x the 0.0335 envelope, "
            f"{margin / max(_fold_spread(rows, worst), 1e-12):.1f}x the worst within-arm fold "
            f"spread). The two folds "
            + (
                "**agree in sign**, so the gap is not carried by one of them."
                if agree
                else "**disagree in sign** — the mean is the average of a win and a loss, and "
                "at n = 2 that is not a result."
            ),
            "",
            "",
            "**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an "
            "across-*seed* envelope measured on the fixture; it says nothing about how much a "
            "metric moves between *this* dataset's folds. A margin smaller than the worst "
            "within-arm fold spread (**⚠** below 2x) separates the two arms by less than one "
            "arm moves on its own, and is not a result however far it clears the envelope.",
            "",
            "**One seed.** `specs/09` §3's repeated-seed rule asks for "
            f"`claim_min_seeds` = {cfg.claim_min_seeds} before this reaches a paper claim.",
        ]
    return lines


if __name__ == "__main__":
    sys.exit(main())
