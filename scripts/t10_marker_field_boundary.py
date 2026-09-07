"""Does `marker_field_r`'s deficit concentrate at the volume boundary, or is it uniform?

`marker_field_r` is v25's weakest metric against its copy floor — **0.247 below it on tier-1,
7.4x R10's 0.0335 envelope** — and it has been weak twice before in v25 (its single loss at T09,
0.1611 in the smoke run). This localises that deficit along the stack, or eliminates the boundary
as its home. Criteria are pre-registered in `progress/t09_inference_and_calibration.md`
(2026-09-07) and restated in :data:`OUTCOMES`; nothing here was written after seeing a number.

⚠️ **A withdrawn claim, because it framed this question.** `marker_field_r` was recorded as v25's
weakness for a *fourth* time on "the pooled loss for v20 and v21 against SpatialZ". That figure is
a **cross-dataset average, which `specs/10` §4.2a forbids by name**: per dataset the comparison is
**9-9**, on tier-1 v20 (0.8804) and v21 (0.8881) both **beat** SpatialZ (0.8522), and v20 wins
**7 of 7** in the wide regime. Only the pooled average favours SpatialZ. Two clean v25 appearances,
not four.

**The quantity is the deficit below the floor, per section — not the raw score.**
`reports/pilot.md` §3 measured the **model-free** copy floor scoring `section_2` worst on 6 of 6
metrics (`marker_field_r` 0.8470 there against 0.8857 / 0.8873), because `section_2` sits beside
`section_1`, the stack's first, so its flanking evidence is **one-sided** — R3's regime. A raw v25
deficit at `section_2` would therefore be partly a property of where the paper protocol put its
held-out sections. Subtracting each section's **own** floor removes that, and what is left is the
model's:

    deficit(s) = flanking_copy(s) - arm(s)

**Zero fits, and zero generation.** Both sides are already on disk: `arms[*].matched_per_section`
and `referents.flanking_copy` in `reports/r11_starmap_layout_modes.json`, written by
`scripts/t10_rescore_saved.py`. This reads them and applies the criteria.

⚠️ **No outcome changes any verdict, and none makes `marker_field_r` a positive.** Three sections,
**one seed**, one dataset. `claim_min_seeds = 3` is about seeds, and this has one. It is worth
running because it is free, and because *eliminating* a candidate is as useful as localising to
one when the list is this short.

Usage::

    python scripts/t10_marker_field_boundary.py \\
        --scores reports/r11_starmap_layout_modes.json \\
        --out reports/t10_marker_field_boundary.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

METRIC = "paper_marker_field_r"
BOUNDARY = "section_2"
"""The stack's first held-out section: `section_1` is the volume's end, so `section_2`'s flanking
evidence is one-sided. R3's regime, and the only boundary section among tier-1's three."""
INTERIOR = ("section_4", "section_6")
ENVELOPE = 0.0335
"""R10's across-seed reproducibility envelope on this instrument. One seed here, so it is the only
scale available; §4.2i's warning applies — it is an estimate, not a constant."""
RECORDED_POOLED = {
    ("field", "rejection"): 0.5763,
    ("hybrid", "rejection"): 0.6384,
    ("resample", "rejection"): 0.6692,
}
"""``(mode, sampler) -> `` the median this project recorded, from ``reports/t10_rescore_exp.json``.

⚠️ **Per arm, because the first version of condition (b) was not.** It compared *every* arm against
``hybrid``'s 0.6384 and duly "fired" on ``field`` and ``resample``, which is a category error: each
mode has its own recorded median. And ``t10_rescore_exp.json`` predates the grid sampler, so a
**grid** arm has no prior value on its own sampler at all — (b) is then **not evaluable**, which is
not the same as passing, and the report says which of the two it is."""

DENSITY_SPREAD_MAX = 2.0
"""⚠️ **POST-HOC, and labelled so.** The pre-registration's condition (c) was binary — "matched
density is not achieved per section" — and by that reading it fires on **all five** arms, since
every one has at least one section emitting fewer cells than the ground truth and subsampling
cannot go up. That is a condition written without asking how the data would fall (`specs/10`
§4.2h's failure, in a new place). The magnitude is what the condition was *for*: a spread of
1.05x across sections cannot confound a comparison between them and a spread of 71x plainly can.
This cut is therefore reported **beside** the pre-registered binary result, never in place of it,
and any arm it admits is admitted on a judgement made after seeing the numbers."""

OUTCOMES = {
    "localised_to_boundary": (
        "deficit(section_2) exceeds the interior mean by MORE than one envelope. The weakness "
        "concentrates where the intensity integral is least stable and the evidence is one-sided "
        "- a pointer at T05's intensity head and at the boundary-clamp geometry (C33, R3). Not "
        "causation, on one seed and three sections."
    ),
    "boundary_eliminated": (
        "the three deficits agree WITHIN one envelope. The weakness is uniform along the stack, "
        "so R3's boundary regime is NOT where it lives and the intensity head's boundary "
        "behaviour is removed as a candidate. The remaining search is the interior mechanism - "
        "arrangement at every depth, not at the edges."
    ),
    "inverted": (
        "deficit(section_2) is LOWER than the interior mean by more than an envelope. Reported "
        "as-is and not explained: no candidate on the table predicts the model doing relatively "
        "better where evidence is one-sided, and it would be a finding about the metric rather "
        "than the model."
    ),
}


def deficits(arm: dict[str, Any], floor: dict[str, float]) -> dict[str, float]:
    """``{section: floor(section) - arm(section)}`` on the density-matched scores.

    Matched, never raw: the layout over-produces cells and a denser point set puts kNN neighbours
    closer together, inflating every graph-based metric. R11 measured `cell_count_ratio` spanning
    500x *between these three sections*, so a raw comparison across them is confounded by exactly
    the quantity being controlled for.
    """
    per_section = arm["matched_per_section"][METRIC]
    return {s: float(floor[s]) - float(per_section[s]) for s in (BOUNDARY, *INTERIOR)}


def classify(d: dict[str, float]) -> tuple[str, float, float]:
    """``(outcome, boundary deficit, interior mean)`` under the pre-registered criteria."""
    interior = float(np.mean([d[s] for s in INTERIOR]))
    gap = d[BOUNDARY] - interior
    if gap > ENVELOPE:
        return "localised_to_boundary", d[BOUNDARY], interior
    if gap < -ENVELOPE:
        return "inverted", d[BOUNDARY], interior
    return "boundary_eliminated", d[BOUNDARY], interior


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", required=True, help="a t10_rescore_saved.py .json")
    ap.add_argument("--arm", default="hybrid-grid", help="which arm to stratify")
    ap.add_argument("--out", default=None, help="destination .md (a .json is written beside it)")
    args = ap.parse_args(argv)

    payload = json.loads(Path(args.scores).read_text())
    arms = {a["arm"]: a for a in payload["arms"]}
    if args.arm not in arms:
        raise SystemExit(f"no arm {args.arm!r} in {args.scores}; have {sorted(arms)}")
    arm = arms[args.arm]
    floor = payload["referents"]["flanking_copy"][METRIC]

    # Uninformative (a): three points is already the minimum; two cannot support the comparison.
    scores = arm["matched_per_section"][METRIC]
    missing = [
        s
        for s in (BOUNDARY, *INTERIOR)
        if scores.get(s) is None or floor.get(s) is None or not np.isfinite(float(scores[s]))
    ]
    if missing:
        raise SystemExit(
            f"UNINFORMATIVE (a): no usable score for {missing} on {METRIC}. Three sections is the "
            "minimum this comparison has; two cannot support it."
        )

    d = deficits(arm, floor)
    outcome, boundary, interior = classify(d)
    pooled = float(arm["matched"][METRIC])
    reference = RECORDED_POOLED.get((arm["mode"], arm["layout_sampler"]))
    drift = None if reference is None else abs(pooled - reference)

    # Uninformative (c), as pre-registered: matching to the ground-truth count is achieved only
    # where the arm emitted at least as many cells as the truth, because subsampling cannot add.
    ratios = {s: float(arm["n_pred"][s]) / float(arm["n_gt"][s]) for s in (BOUNDARY, *INTERIOR)}
    under = [s for s, r in ratios.items() if r < 1.0]
    spread = max(ratios.values()) / min(ratios.values())

    lines = [
        f"# `marker_field_r` — boundary stratification, arm `{args.arm}`",
        "",
        "Deficit below **each section's own** `flanking_copy` floor, on the density-matched "
        "scores. Criteria pre-registered in `progress/t09_inference_and_calibration.md` "
        "(2026-09-07) before this script existed.",
        "",
        "| section | regime | `flanking_copy` | arm | deficit |",
        "|---|---|---|---|---|",
    ]
    for s in (BOUNDARY, *INTERIOR):
        regime = "**boundary** (one-sided evidence)" if s == BOUNDARY else "interior"
        lines.append(
            f"| `{s}` | {regime} | {float(floor[s]):.4f} | {float(scores[s]):.4f} | "
            f"**{d[s]:.4f}** |"
        )
    lines += [
        "",
        f"Boundary deficit **{boundary:.4f}**, interior mean **{interior:.4f}**, gap "
        f"**{boundary - interior:+.4f}** against a **{ENVELOPE}** envelope "
        f"(**{abs(boundary - interior) / ENVELOPE:.2f}x**).",
        "",
        f"## Verdict: **{outcome.replace('_', ' ').upper()}**",
        "",
        OUTCOMES[outcome],
        "",
        "⚠️ **This changes no verdict and makes `marker_field_r` no less a weakness.** Three "
        "sections, one seed, one dataset; `claim_min_seeds = 3` is about seeds and this has one. "
        "It is a pointer for whoever picks the metric up.",
        "",
    ]
    lines += ["## Uninformative conditions", ""]
    if reference is None:
        lines += [
            f"* **(b) NOT EVALUABLE** — no prior median is recorded for `{arm['mode']}` on the "
            f"`{arm['layout_sampler']}` sampler (`reports/t10_rescore_exp.json` predates it). "
            "That is not the same as passing: nothing confirms these are the scores an earlier "
            "run produced, because there is no earlier run on this sampler.",
        ]
    elif drift is not None and drift > ENVELOPE:
        lines += [
            f"* 🚨 **(b) FIRES** — pooled median **{pooled:.4f}** against the recorded "
            f"**{reference:.4f}** for `{arm['mode']}`/`{arm['layout_sampler']}`, a drift of "
            f"{drift:.4f}, more than one envelope. Not the run the recorded number came from; "
            "the verdict above must not be read.",
        ]
    else:
        lines += [
            f"* **(b) does not fire** — pooled median **{pooled:.4f}** against the recorded "
            f"**{reference:.4f}** (drift {drift:.4f}, inside the envelope).",
        ]
    ratio_text = ", ".join(f"`{s}` {ratios[s]:.2f}x" for s in (BOUNDARY, *INTERIOR))
    if under:
        lines += [
            f"* 🚨 **(c) FIRES as pre-registered** — matched density is not achieved at "
            f"{', '.join(f'`{s}`' for s in under)} (emitted below the ground-truth count, and "
            f"subsampling cannot add cells). Per-section ratios: {ratio_text}.",
        ]
    else:
        lines += [f"* **(c) does not fire** — per-section ratios {ratio_text}."]
    lines += [
        f"  * ⚠️ **Magnitude, POST-HOC**: the across-section density spread is "
        f"**{spread:.2f}x**"
        + (
            f", above the {DENSITY_SPREAD_MAX:.1f}x cut — the sections were scored on point sets "
            "of different quality and a comparison between them is confounded by exactly the "
            "quantity (c) names. **Do not read the verdict.**"
            if spread > DENSITY_SPREAD_MAX
            else f", below the {DENSITY_SPREAD_MAX:.1f}x cut — small enough that it cannot "
            "carry a difference between sections."
        )
        + " This cut is a judgement made after seeing the numbers; the pre-registered condition "
        "was binary and fires above regardless.",
        "",
    ]

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "metric": METRIC,
                    "envelope": ENVELOPE,
                    "floor_per_section": {s: float(floor[s]) for s in (BOUNDARY, *INTERIOR)},
                    "arm_per_section": {s: float(scores[s]) for s in (BOUNDARY, *INTERIOR)},
                    "deficits": d,
                    "boundary_deficit": boundary,
                    "interior_mean_deficit": interior,
                    "gap": boundary - interior,
                    "outcome": outcome,
                    "pooled_median": pooled,
                    "recorded_pooled": reference,
                    "density_ratios": ratios,
                    "density_spread": spread,
                    "uninformative_c": bool(under),
                    "uninformative_b": None if drift is None else drift > ENVELOPE,
                },
                indent=2,
            )
        )
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
