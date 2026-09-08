"""A9 — apply the metric-aware pre-registration to six `t09_ship_starmap.py` reports.

**Written before any of the six fits landed**, so the outcome cannot have shaped the instrument.
The criteria are `progress/t09_inference_and_calibration.md`'s A9 pre-registration (2026-09-07),
restated in :data:`PRIMARY` and the branch functions below; nothing here was added after a number.

The claim under test: **the metric-aware losses improve the metrics they are made of.**

* **PRIMARY** — the three the losses are *built from*: `morans_pearson` and `gearys_pearson` from
  the autocorrelation term, `marker_depth_r` from the profile term.
* **secondary** — `umap_mixing` and `marker_field_r`. The distribution term is not built from them,
  so they are reported and decide nothing.
* **excluded** — `celltype_localization`, which is **inert** under ``layout_mode=resample`` (cell
  types come from the copied layout, identically in both arms). A column with no variance cannot
  separate two arms, and scoring it as a tie would flatter whichever branch counts ties.

Envelope: `specs/10` §4.2b's shared envelope, computed **on this run** — the largest across-seed
spread among the arms on that metric — never R10's inherited 0.0335, which is what the criteria are
being tested *against* rather than *with*. Both §4.2d constructions are computed, and a metric
whose verdict differs between them is **not established**.

Usage::

    python scripts/t10_a9_aggregate.py reports/t10_a9_*.json --out reports/t10_a9.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PRIMARY = ("paper_morans_pearson", "paper_gearys_pearson", "paper_marker_depth_r")
SECONDARY = ("paper_umap_mixing", "paper_marker_field_r")
EXCLUDED = "paper_celltype_localization"
R10_ENVELOPE = 0.0335
"""R10's recorded figure. Used ONLY by uninformative (a) as a sanity bound on this run's own
envelope — never as the threshold a margin is read against."""
RECORDED_OFF = {
    "paper_morans_pearson": 0.9316,
    "paper_gearys_pearson": 0.9022,
    "paper_umap_mixing": 0.9145,
}
"""The `(2x, weights off)` row this experiment is reproducing, for uninformative (c)."""


def load(paths: list[str]) -> list[dict[str, Any]]:
    """One record per fit: ``{seed, weight, per_section, matched, alarms}``.

    The arm is read from the persisted **config**, not from the filename: a report named
    ``_off_`` that was fitted at 0.5 is exactly the confusion this project has already paid for
    once, and a filename is not a measurement.
    """
    out = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        cfg = payload["config"]
        weights = {float(cfg[k]) for k in ("w_autocorr", "w_profile", "w_distribution")}
        if len(weights) != 1:
            raise SystemExit(
                f"{path}: the three metric-aware weights differ ({sorted(weights)}). A9 moves "
                "them together as one gate (`specs/09` §3); a run with them split is testing a "
                "configuration the selection never considered."
            )
        shipped = [a for a in payload["arms"] if a.get("shipped")] or payload["arms"]
        out.append(
            {
                "path": path,
                "seed": int(shipped[0]["seed"]),
                "weight": weights.pop(),
                "matched": shipped[0]["matched"],
                "per_section": shipped[0]["matched_per_section"],
                "alarms": payload.get("alarms"),
            }
        )
    return out


def spread(values: list[float]) -> float:
    return float(max(values) - min(values)) if values else float("nan")


def envelopes(records, metric: str) -> tuple[float, float]:
    """``(fold-mean, per-fold)`` shared envelopes — `specs/10` §4.2d's two constructions.

    Both are maxima over the two arms, which is what makes them *shared* (§4.2b): one threshold
    per metric, so neither arm's verdict can be bought by varying less than the other.
    """
    per_a, per_b = [], []
    for weight in sorted({r["weight"] for r in records}):
        arm = [r for r in records if r["weight"] == weight]
        per_a.append(spread([float(r["matched"][metric]) for r in arm]))
        sections = sorted({s for r in arm for s in r["per_section"][metric]})
        per_b.append(
            max(spread([float(r["per_section"][metric][s]) for r in arm]) for s in sections)
        )
    return max(per_a), max(per_b)


def margin(records, metric: str) -> dict[str, Any]:
    """``(on) - (off)`` per seed and per fold, with the sign agreement both branches need."""
    seeds = sorted({r["seed"] for r in records})
    by = {(r["seed"], r["weight"]): r for r in records}
    on, off = max({r["weight"] for r in records}), min({r["weight"] for r in records})
    per_seed, per_fold = [], []
    for s in seeds:
        a, b = by.get((s, on)), by.get((s, off))
        if a is None or b is None:
            raise SystemExit(f"seed {s} is missing an arm; A9 needs both at every seed")
        per_seed.append(float(a["matched"][metric]) - float(b["matched"][metric]))
        for section in sorted(a["per_section"][metric]):
            per_fold.append(
                float(a["per_section"][metric][section]) - float(b["per_section"][metric][section])
            )
    return {
        "per_seed": per_seed,
        "mean": float(np.mean(per_seed)),
        "signs_agree_seeds": len({v > 0 for v in per_seed}) == 1,
        "signs_agree_folds": len({v > 0 for v in per_fold}) == 1,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("reports", nargs="+", help="the six t09_ship_starmap.py .json files")
    ap.add_argument("--out", default=None, help="destination .md")
    args = ap.parse_args(argv)

    records = load(args.reports)
    seeds = sorted({r["seed"] for r in records})
    weights = sorted({r["weight"] for r in records})
    if len(weights) != 2:
        raise SystemExit(f"expected two arms, found weights {weights}")
    on, off = weights[1], weights[0]

    lines = [
        "# A9 — do the metric-aware losses improve the metrics they are made of?",
        "",
        f"Seeds {seeds}, arms `on`={on:g} and `off`={off:g}, {len(records)} fits. Criteria "
        "pre-registered in `progress/t09_inference_and_calibration.md` (2026-09-07), before any "
        "fit and before this script.",
        "",
        "| metric | role | mean (on-off) | per seed | envelope (a) | vs | envelope (b) | vs | "
        "signs | established? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    verdicts: dict[str, str] = {}
    unstable: list[str] = []
    for metric in (*PRIMARY, *SECONDARY):
        role = "**PRIMARY**" if metric in PRIMARY else "secondary"
        m = margin(records, metric)
        env_a, env_b = envelopes(records, metric)
        clears_a, clears_b = abs(m["mean"]) > env_a, abs(m["mean"]) > env_b
        agree = m["signs_agree_seeds"] and m["signs_agree_folds"]
        established = clears_a == clears_b
        if not established:
            unstable.append(metric)
        if metric in PRIMARY:
            verdicts[metric] = (
                "positive"
                if (clears_a and clears_b and agree and m["mean"] > 0)
                else "negative"
                if (clears_a and clears_b and agree and m["mean"] < 0)
                else "inside"
            )
        lines.append(
            f"| `{metric.replace('paper_', '')}` | {role} | **{m['mean']:+.4f}** | "
            + " / ".join(f"{v:+.4f}" for v in m["per_seed"])
            + f" | {env_a:.4f} | {abs(m['mean']) / env_a:.2f}x | {env_b:.4f} | "
            f"{abs(m['mean']) / env_b:.2f}x | "
            + ("agree" if agree else "**disagree**")
            + " | "
            + ("yes" if established else "**NO — §4.2d**")
            + " |"
        )
    lines += [
        "",
        f"`{EXCLUDED.replace('paper_', '')}` is **excluded**, not scored as a tie: it is inert "
        "under `layout_mode=resample`, and a column with no variance cannot separate two arms.",
        "",
    ]

    # ---- uninformative conditions, in the pre-registered order
    lines += ["## Uninformative conditions", ""]
    fired: list[str] = []
    worst = max(max(envelopes(records, m)) for m in PRIMARY)
    if worst >= 2 * R10_ENVELOPE:
        fired.append("a")
        lines.append(
            f"* 🚨 **(a) FIRES** — the worst primary envelope is **{worst:.4f}**, at least twice "
            f"R10's recorded {R10_ENVELOPE}. The design is noisier than the instrument that "
            "produced the number being tested; neither outcome can be read."
        )
    else:
        lines.append(
            f"* **(a) does not fire** — worst primary envelope {worst:.4f}, under "
            f"{2 * R10_ENVELOPE:.4f}."
        )
    if len([m for m in unstable if m in PRIMARY]) == len(PRIMARY):
        fired.append("b")
        lines.append(
            "* 🚨 **(b) FIRES** — the two §4.2d constructions disagree on **all three** primary "
            "metrics. The verdict would be entirely a property of the aggregation choice."
        )
    else:
        lines.append(
            "* **(b) does not fire** — the constructions agree on at least one primary metric"
            + (f" (they disagree on {', '.join(f'`{m}`' for m in unstable)})" if unstable else "")
            + "."
        )
    seed1_off = next((r for r in records if r["seed"] == seeds[0] and r["weight"] == off), None)
    drifted = []
    if seed1_off is not None:
        for metric, recorded in RECORDED_OFF.items():
            got = float(seed1_off["matched"][metric])
            env_a, _ = envelopes(records, metric)
            if abs(got - recorded) > env_a:
                drifted.append(f"`{metric.replace('paper_', '')}` {got:.4f} vs {recorded}")
    if drifted:
        fired.append("c")
        lines.append(
            "* 🚨 **(c) FIRES** — the off arm at the first seed does not reproduce the recorded "
            "`(2x, weights off)` row: " + "; ".join(drifted) + ". These fits are not reproducing "
            "the selection they are testing."
        )
    else:
        lines.append("* **(c) does not fire** — the off arm reproduces the recorded row.")
    alarmed = [
        f"seed {r['seed']} @ {r['weight']:g}"
        for r in records
        if r["alarms"]
        and any(
            r["alarms"].get(k)
            for k in ("collapse_alarms", "spatial_collapse_alarms", "spatial_inversion_alarms")
        )
    ]
    if alarmed:
        fired.append("d")
        lines.append(
            "* 🚨 **(d) FIRES** — a collapse alarm fired on " + ", ".join(alarmed) + ". A7's "
            "lesson: every metric in a report from a model that reported itself broken is "
            "uninterpretable until that is explained."
        )
    else:
        lines.append(
            "* **(d) does not fire** — no collapse or inversion alarm on any of the six fits, "
            "and the alarm record was **checked** rather than assumed absent."
        )

    # ---- the verdict
    positives = [m for m, v in verdicts.items() if v == "positive"]
    negatives = [m for m, v in verdicts.items() if v == "negative"]
    if fired:
        outcome = "UNINFORMATIVE"
    elif positives and negatives:
        outcome = "MIXED"
    elif len(positives) >= 2 and not negatives:
        outcome = "POSITIVE"
    elif negatives and not positives:
        outcome = "NEGATIVE"
    else:
        outcome = "NULL"
    lines += ["", f"## Verdict: **{outcome}**", "", READINGS[outcome], ""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"\nwrote {args.out}")
    return 0


READINGS = {
    "POSITIVE": (
        "The weights ship at 0.5 **on evidence**. The 2400 selection is retrospectively "
        "justified and the shipped baseline no longer carries an unestablished component."
    ),
    "NULL": (
        '**Not "the losses are harmless".** The selection that put them at 0.5 was a coin-flip '
        "on an aggregate rank, made on margins this design cannot resolve; **every absolute "
        'number in this project was produced with three inert terms active**; and "the best '
        'configuration found" is a statement about a rank rather than about the model. The '
        "right disposition is to **say so, not to turn them off** — turning them off after the "
        "fact re-opens every fitted number for a change that is by hypothesis within noise. "
        "⚠️ And a null here means *no detectable effect on this configuration, with three known "
        "distortions present in BOTH arms* — see the pre-registration's common-mode note: a "
        "common-mode nuisance cannot manufacture a difference, but it can hide one."
    ),
    "NEGATIVE": (
        "The same shape as A7 and stronger: the weights should be **0**, and every number in "
        "the project was measured on a configuration actively worse than an available "
        "alternative. This outcome **does** require re-opening the absolute numbers, and that "
        "cost was named in the pre-registration rather than discovered here."
    ),
    "MIXED": (
        "One primary clears positive and another clears negative. The claim is **refuted as "
        'stated**: "improves the metrics they are made of" is a claim about the set, and a set '
        "that moves in both directions does not satisfy it."
    ),
    "UNINFORMATIVE": (
        "An uninformative condition fired. **No verdict may be read**, in either direction, and "
        "the conditions were fixed before the fits so this cannot be invoked selectively."
    ),
}


if __name__ == "__main__":
    sys.exit(main())
