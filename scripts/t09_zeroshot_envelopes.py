"""``specs/10`` §4.2d: both envelope constructions for the replication, and the rule that if they
disagree nothing is established.

§4.2a settles *which arm's* variance an envelope is measured on; §4.2b settles which envelope a
*clearance* takes. Neither says **how the across-seed spread is constructed**, and there are two:

**(a) fold-mean** — average the folds within a seed, then spread across seeds. What
``t09_zeroshot_aggregate.py`` computes and what every seed claim in this project has used.

**(b) per-fold** — spread across seeds *within* each fold, then the worst fold.

(b) is always >= (a), because averaging folds removes variance. So an effect that clears (a) and
not (b) has a verdict that is a property of the aggregation choice, and §4.2d's rule is that such
a result **is not established**. This script computes both and applies the replication's
pre-registered criteria under each.

**Why a separate instrument rather than a flag on the aggregator.**
``t09_zeroshot_aggregate.py`` implements ``deep_starmap``'s *original* pre-registration — primary
``marker_depth_r``, contrast A1-A3 — and its per-contrast tables take §4.2a envelopes. The
replication pre-registered a different primary (``morans_pearson``, **A2-A3**, held-out) against a
single **shared** §4.2b envelope: *"the largest across-seed spread among all four arms and the
referents, on this metric and this dataset"*. Two pre-registrations, two verdicts; folding the
second into the script that states the first is how a reader ends up applying the wrong one.

⚠️ **The shared envelope, not the contrast's own.** Criterion 2 of Part 1 is a contrast, which
§4.2a would measure against the arm that carries the variance. The replication overrides that
explicitly: it defines one envelope for every criterion, and uninformative condition **(c)** is
stated on *"the shared envelope ... exceeds 0.2514, the deep_starmap A2-A3 effect"* — a guard
that only means anything if the shared envelope is what A2-A3 is read against. Both readings are
printed, because the difference decides the verdict and a reader is entitled to see it; the
pre-registered one is the shared envelope and is the one the verdict is taken from.

Usage::

    python scripts/t09_zeroshot_envelopes.py reports/t09_zeroshot_cosmx_seed*.json \\
        --out reports/t09_envelopes_cosmx.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from spatialcpav25_gen.train.select import METRIC_NAMES

ARMS = ("A1", "A2", "A3", "A4")
ENVELOPE_MEMBERS = ("A1", "A2", "A3", "A4", "shuffled")
"""Everything the shared envelope is taken over. ``constant_field`` is excluded by the
pre-registration on ``specs/10`` §4.2c's authority — and its spread is zero anyway, so the
exclusion changes no number; it is stated so the set is not mistaken for an oversight."""

PRIMARY = "morans_pearson"
DEEP_HELD_OUT_A2A3 = 0.2514
DEEP_KEPT_A2A3 = 0.1322
"""The ``deep_starmap`` effects being replicated; also uninformative thresholds (c) and (e)."""


def load(paths: list[str]) -> dict[tuple[str, str, int, str], dict[str, float]]:
    """``(side, arm, seed, fold) -> {metric: value}`` over every seed file, refusing duplicates."""
    out: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for path in paths:
        for row in json.loads(Path(path).read_text()):
            key = (str(row["side"]), str(row["arm"]), int(row["seed"]), str(row["fold"]))
            if key in out:
                raise SystemExit(f"duplicate scored cell {key}; the inputs overlap")
            out[key] = {m: float(row[m]) for m in METRIC_NAMES if m in row}
    return out


def envelopes(cells, side: str, metric: str, seeds, folds) -> tuple[float, float, str, str]:
    """``(env_a, env_b, arm_setting_a, arm_setting_b)`` — the two §4.2d constructions.

    ``env_a`` averages folds within a seed then spreads across seeds; ``env_b`` spreads across
    seeds within each fold and takes the worst fold. Both are maxima over
    :data:`ENVELOPE_MEMBERS`, which is what makes them *shared* envelopes (§4.2b): one threshold
    per metric per pool, so no arm's verdict can be bought with a low variance.
    """
    per_a: dict[str, float] = {}
    per_b: dict[str, float] = {}
    for arm in ENVELOPE_MEMBERS:
        means = [float(np.mean([cells[(side, arm, s, f)][metric] for f in folds])) for s in seeds]
        per_a[arm] = max(means) - min(means)
        per_b[arm] = max(
            max(cells[(side, arm, s, f)][metric] for s in seeds)
            - min(cells[(side, arm, s, f)][metric] for s in seeds)
            for f in folds
        )
    a = max(per_a, key=lambda k: per_a[k])
    b = max(per_b, key=lambda k: per_b[k])
    return per_a[a], per_b[b], a, b


def effect(cells, side: str, metric: str, left: str, right: str, seeds, folds) -> float:
    """The fold-averaged, seed-averaged contrast ``left - right``. Reported as a fold average,
    which under §4.2d is the estimator construction (a) matches; (b) is the stricter check the
    same number must also survive."""
    per_seed = [
        float(
            np.mean(
                [
                    cells[(side, left, s, f)][metric] - cells[(side, right, s, f)][metric]
                    for f in folds
                ]
            )
        )
        for s in seeds
    ]
    return float(np.mean(per_seed))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("seed_files", nargs="+")
    ap.add_argument("--out", default=None, help="destination .json")
    args = ap.parse_args(argv)

    cells = load(args.seed_files)
    seeds = sorted({k[2] for k in cells})
    folds = sorted({k[3] for k in cells})
    sides = sorted({k[0] for k in cells})
    print(f"seeds {seeds}, folds {folds}, sides {sides}\n")

    report: dict[str, Any] = {"seeds": seeds, "folds": folds, "metric": PRIMARY, "sides": {}}
    disagreements: list[str] = []

    for side in sides:
        env_a, env_b, who_a, who_b = envelopes(cells, side, PRIMARY, seeds, folds)
        floor = float(
            np.mean(
                [np.mean([cells[(side, "shuffled", s, f)][PRIMARY] for f in folds]) for s in seeds]
            )
        )
        print(f"## {side} — `{PRIMARY}`")
        print(f"  envelope (a) fold-mean : {env_a:.4f}   set by {who_a}")
        print(
            f"  envelope (b) per-fold  : {env_b:.4f}   set by {who_b}   ({env_b / env_a:.2f}x (a))"
        )
        print(f"  `shuffled` floor       : {floor:+.4f}")

        side_report: dict[str, Any] = {
            "envelope_fold_mean": env_a,
            "envelope_per_fold": env_b,
            "envelope_set_by_fold_mean": who_a,
            "envelope_set_by_per_fold": who_b,
            "shuffled_floor": floor,
            "checks": {},
        }

        checks: list[tuple[str, float, str]] = []
        for left, right in (("A2", "A3"), ("A2", "A4")):
            checks.append(
                (
                    f"{left}-{right}",
                    abs(effect(cells, side, PRIMARY, left, right, seeds, folds)),
                    "contrast",
                )
            )
        for arm in ARMS:
            mean = float(
                np.mean(
                    [np.mean([cells[(side, arm, s, f)][PRIMARY] for f in folds]) for s in seeds]
                )
            )
            checks.append((f"{arm} over floor", abs(mean - floor), "clearance"))

        for name, value, kind in checks:
            ratio_a, ratio_b = value / env_a, value / env_b
            passes_a, passes_b = value > env_a, value > env_b
            agree = passes_a == passes_b
            flag = "" if agree else "   ⚠ CONSTRUCTIONS DISAGREE — NOT ESTABLISHED"
            if not agree:
                disagreements.append(f"{side} {name}")
            print(
                f"    {name:16s} {value:.4f}   (a) {ratio_a:5.2f}x "
                f"{'clears' if passes_a else 'inside':6s}   (b) {ratio_b:5.2f}x "
                f"{'clears' if passes_b else 'inside':6s}{flag}"
            )
            side_report["checks"][name] = {
                "kind": kind,
                "value": value,
                "vs_fold_mean": ratio_a,
                "vs_per_fold": ratio_b,
                "clears_fold_mean": passes_a,
                "clears_per_fold": passes_b,
                "established": agree,
            }
        report["sides"][side] = side_report
        print()

    held = report["sides"].get("held_out", {})
    kept = report["sides"].get("kept", {})
    print("## Uninformative conditions, under both constructions")
    for label, side_rep, threshold in (
        ("(c) held-out", held, DEEP_HELD_OUT_A2A3),
        ("(e) kept", kept, DEEP_KEPT_A2A3),
    ):
        if not side_rep:
            continue
        for tag, key in (("a", "envelope_fold_mean"), ("b", "envelope_per_fold")):
            env = side_rep[key]
            fires = env > threshold
            note = (
                "🚨 FIRES — the design cannot detect the effect being replicated"
                if fires
                else "does not fire"
            )
            print(f"  {label} envelope ({tag}) {env:.4f} vs {threshold:.4f} -> {note}")
    print()

    if disagreements:
        print("🚨 §4.2d: the two constructions disagree on " + ", ".join(disagreements) + ".")
        print("   Every one of those results is a property of the aggregation choice and is NOT")
        print("   ESTABLISHED. A verdict may not be read from them.")
    else:
        print("✅ §4.2d: every check lands on the same side of both constructions. The verdicts")
        print("   do not depend on the aggregation choice.")
    report["disagreements"] = disagreements

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return 1 if disagreements else 0


if __name__ == "__main__":
    sys.exit(main())
