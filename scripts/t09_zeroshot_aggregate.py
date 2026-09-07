"""Read the zero-shot seed files and apply the pre-registration to them.

Aggregates ``reports/t09_zeroshot_deep_seed*.json`` — one per seed, four arms x two folds x
two gene pools plus the two referents — into the table the pre-registration is stated against,
and prints the verdict it implies rather than leaving the reader to assemble it.

**Not every referent is a floor, and the script refuses to pretend otherwise.** Two of them are
degenerate by construction, both established by reproduction rather than by argument:

``constant_field`` on ``morans_pearson`` / ``gearys_pearson``
    A constant field has **exactly zero** per-gene variance after normalisation, so Moran's I
    and Geary's C are ``0/0``. What comes back is float32 round-off — magnitudes of order 1e-7 —
    and it is not random: it scales with the gene's own magnitude, so it correlates with the
    real per-gene statistic. Measured on the fixture: normalised per-gene std exactly 0.0,
    generated Moran's I std 3.5e-8, and ``morans_pearson`` **+0.22**. On ``deep_starmap`` the
    same referent reads **+0.53** and ``gearys_pearson`` **-0.49**. Those are not information
    a generated section has to beat; they are the metric's behaviour on a degenerate input.
``shuffled`` on ``umap_mixing``
    :func:`~spatialcpav25_gen.train.select._mixing` never sees a coordinate — two clouds are
    mixed in expression space — so permuting positions returns the arm's own score **exactly**.
    In the seed files it agrees with A1 to every printed digit, because it *is* A1.

``celltype_localization`` is gene-free and, under ``layout_mode=resample``, identical across
every arm and every seed. It is reported once as a constant and excluded from every contrast:
a column with no variance cannot separate two arms.

Usage::

    python scripts/t09_zeroshot_aggregate.py \\
        reports/t09_zeroshot_deep_seed2.json reports/t09_zeroshot_deep_seed3.json \\
        reports/t09_zeroshot_deep_seed4.json --out reports/t09_zeroshot_deep.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from spatialcpav25_gen.train.select import METRIC_NAMES

ARMS = ("A1", "A2", "A3", "A4")
REFERENTS = ("constant_field", "shuffled")

ARM_LABEL = {
    "A1": "medcpt + distill",
    "A2": "medcpt, pure text",
    "A3": "lookup + distill",
    "A4": "lookup, pure text",
}

CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("A1", "A3", "PRIMARY — does W.t add anything over a distillation head that sees the text?"),
    (
        "A2",
        "A3",
        "the two routes to an unseen gene, head to head: text alone vs distillation alone. "
        "Neither reads the free residual, which is zero for a held-out gene either way",
    ),
    ("A2", "A4", "pure text vs no text at all: the text channel against a gene-blind arm"),
    ("A1", "A2", "what the distillation head does to the text arm"),
    ("A3", "A4", "the distillation head with no text channel to help it"),
)

VALID_FLOOR: dict[str, tuple[str, ...]] = {
    "morans_pearson": ("shuffled",),
    "gearys_pearson": ("shuffled",),
    "umap_mixing": (),
    "marker_field_r": ("shuffled",),
    "marker_depth_r": ("shuffled",),
    "celltype_localization": (),
}
"""Which referents mean something for which metric.

⚠️ **`constant_field` is a floor for nothing, on any metric.** A referent answers "what does this
metric return when there is nothing to find?" only if its input contains nothing to find. The
constant field gives every cell the same value for a gene, so its across-cell coefficient of
variation should be zero and measures **2.6e-07** — float32 epsilon — against **16.1** for the
`shuffled` referent on the same rows. Eight orders of magnitude, and identical for both metrics,
because it is a property of the input rather than of the metric. An input with no variation holds
no spatial signal, so whatever a metric returns for it is that metric's behaviour on a degenerate
input, at any magnitude.

⚠️ **Two earlier versions of this note were wrong, in different ways.** The first kept the
constant field for the two profile metrics, arguing that `soft_depth_profile`'s bin normalisation
makes it a real function of cell density; that was reasoning where a measurement was available.
The second replaced it with a **precision-drift threshold** — recompute at float64, call a
referent degenerate if it moves more than 0.01 — which separated cleanly on the synthetic fixture
(1e-2 against 1e-8) and **failed on real data**: `deep_starmap`'s eight constant-field rows drift
0.0042, 0.0092, 0.0092, 0.0111, 0.0438, 0.0545, 0.0738, 0.1905, a continuum with no gap, so the
cut fell between two rows of identical construction and called one stable and the other not. Drift
is now reported as corroboration and decides nothing.

The reason a large float64 value is still round-off: the error in the centring step is one ulp of
each value, so its *pattern across genes* tracks expression level at every precision, and
expression level is what real Moran's I correlates with. `section_5`'s +0.5780 reads +0.3875 at
double precision — a different number, the same artifact.

`shuffled` is the floor for the four gene-dependent metrics: real counts, permuted positions, so
the pairing is destroyed and both marginals survive. `umap_mixing` has none — `_mixing` reads no
coordinates, so shuffling returns the arm's own score."""

FLOOR_NOTE = {
    "morans_pearson": "constant field carries no variation in its input, so it is not a floor",
    "gearys_pearson": "constant field carries no variation in its input, so it is not a floor",
    "marker_field_r": "constant field carries no variation in its input, so it is not a floor",
    "marker_depth_r": "constant field carries no variation in its input, so it is not a floor",
    "umap_mixing": "shuffled is the arm's own score (`_mixing` reads no coordinates); "
    "constant field is degenerate like every other. No usable floor.",
    "celltype_localization": "gene-free and constant across arms and seeds; not a contrast",
}

PRIMARY_METRIC = "marker_depth_r"
PRIMARY_SIDE = "held_out"
SECONDARY_METRIC = "morans_pearson"
"""Which metric and side the two clearance tables and the ``--ceiling`` flags address.

⚠️ **These names are a reporting convenience, not a claim.** They were `deep_starmap`'s
pre-registered primary and the metric its only positive landed on, and while this script also
printed a verdict under those names it was asserting one experiment's criteria over any seed
files it was handed — including the `cosmx` replication, whose primary is ``morans_pearson`` on
the **A2-A3** contrast against a shared §4.2b envelope. The verdict block is gone (2026-09-07);
what survives is arithmetic, and no metric here is "the primary" of anything."""
FOLD_BALANCE_MIN = 0.25
"""``min|diff| / max|diff|`` across folds. Below this the gap is carried by one fold."""


def load(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text())
        if not isinstance(payload, list) or not payload:
            raise SystemExit(f"{path} does not hold a non-empty list of scored rows")
        rows.extend(payload)
    seeds = sorted({int(r["seed"]) for r in rows})
    if len(seeds) != len(paths):
        raise SystemExit(
            f"{len(paths)} files but {len(seeds)} distinct seeds {seeds}. Two files from the "
            "same seed would count one fit twice and narrow every spread."
        )
    return rows


def indexed(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int, str], dict[str, float]]:
    """``(side, arm, seed, fold) -> {metric: value}``, refusing a duplicated cell."""
    out: dict[tuple[str, str, int, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["side"]), str(row["arm"]), int(row["seed"]), str(row["fold"]))
        if key in out:
            raise SystemExit(f"duplicate scored cell {key}; the inputs overlap")
        out[key] = {m: float(row[m]) for m in METRIC_NAMES}
    return out


def fold_mean(cells, side, arm, seed, folds, metric) -> float:
    return float(np.mean([cells[(side, arm, seed, f)][metric] for f in folds]))


def spread(values: list[float]) -> float:
    """Across-seed spread: max - min. The envelope this arm carries on this metric."""
    return float(max(values) - min(values))


def _room(path: str, metric: str, arms: dict, floor: float) -> list[str]:
    """Place the arms against the model-free ceiling: what fraction of the room is being used.

    The ceiling and the floor bracket what any method could achieve on these genes, so a score
    means little until it is read against them. ``0.27`` is a strong result against a ceiling of
    0.30 and a weak one against 0.995, and only the second of those is a statement about the
    method rather than about the metric.
    """
    ceiling = json.loads(Path(path).read_text())
    held = [c["held_out"]["noiseless_ceiling"] for c in ceiling]
    top = float(np.median(held))
    room = top - floor
    best_arm, best = max(arms.items(), key=lambda kv: kv[1]["mean"])
    lines = [
        f"**Against the room actually available** (`{path}`): the model-free ceiling on these "
        f"genes is **{top:.4f}** and the floor {floor:+.4f}, so the room is **{room:.4f}**. "
        f"The best arm, {best_arm}, reaches {best['mean']:+.4f} — "
        f"**{100 * (best['mean'] - floor) / room:.0f}%** of it.",
        "",
    ]
    copy = float(np.median([c["held_out"]["best_other_section"] for c in ceiling]))
    lines += [
        f"For scale on the same rows, copying a whole real section scores {copy:.4f}, or "
        f"{100 * (copy - floor) / room:.0f}% of the room — **context only; no zero-shot arm "
        "may copy.**",
        "",
    ]
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("seed_files", nargs="+")
    ap.add_argument("--out", default=None, help="destination .md (a .json is written beside it)")
    ap.add_argument(
        "--ceiling",
        default=None,
        help=f"the `--metric {PRIMARY_METRIC}` ceiling .json, to place the primary result "
        "against the room actually available",
    )
    ap.add_argument(
        "--ceiling-secondary",
        default=None,
        help=f"the `--metric {SECONDARY_METRIC}` ceiling .json. With it, the secondary result "
        "is reported as a fraction of the room above that metric's usable floor",
    )
    args = ap.parse_args(argv)

    rows = load(args.seed_files)
    cells = indexed(rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    folds = sorted({str(r["fold"]) for r in rows})
    sides = sorted({str(r["side"]) for r in rows})
    metrics = [m for m in METRIC_NAMES if VALID_FLOOR[m] or m != "celltype_localization"]

    lines: list[str] = [
        "# The four-arm zero-shot comparison",
        "",
        f"Seeds {seeds}, folds {folds}, arms {list(ARMS)}. "
        f"{len(rows)} scored cells. **Arithmetic only — this file states no verdict.** Each "
        "experiment's pre-registration and its reading live in "
        "`progress/t09_inference_and_calibration.md`, and "
        "`scripts/t09_zeroshot_envelopes.py` applies the replication's.",
        "",
        "| arm | `text_emb_mode` | `use_distill` | unseen gene's embedding |",
        "|---|---|---|---|",
        "| A1 | `medcpt` | yes | `norm(W t + gamma psi(t))` — the full claim |",
        "| A2 | `medcpt` | no | `norm(W t)` — the designed channel alone |",
        "| A3 | `lookup` | yes | `norm(gamma psi(t))` — **the real competitor** |",
        "| A4 | `lookup` | no | `norm(0)` — one vector per gene; the void condition |",
        "",
    ]

    report: dict[str, Any] = {"seeds": seeds, "folds": folds, "sides": sides, "arms": list(ARMS)}

    # ---------------------------------------------------------------- per-arm scores
    for side in sides:
        lines += [
            f"## Scores — `{side}` genes",
            "",
            "Per-seed fold means, then the arm's own across-seed spread.",
            "",
        ]
        header = (
            "| metric | "
            + " | ".join(f"{a} s{s}" for a in ARMS for s in seeds)
            + " | "
            + " | ".join(f"{a} spread" for a in ARMS)
            + " |"
        )
        lines += [header, "|" + "---|" * (1 + len(ARMS) * len(seeds) + len(ARMS))]
        for metric in metrics:
            per_arm = {
                arm: [fold_mean(cells, side, arm, s, folds, metric) for s in seeds] for arm in ARMS
            }
            report.setdefault("scores", {}).setdefault(side, {})[metric] = {
                arm: {"per_seed": values, "spread": spread(values)}
                for arm, values in per_arm.items()
            }
            cellsline = " | ".join(f"{v:+.4f}" for arm in ARMS for v in per_arm[arm])
            spreads = " | ".join(f"{spread(per_arm[arm]):.4f}" for arm in ARMS)
            lines.append(f"| `{metric}` | {cellsline} | {spreads} |")
        lines.append("")

        # ------------------------------------------------------------ referents
        lines += [
            "### Referents",
            "",
            "| metric | constant field | shuffled | usable floor |",
            "|---|---|---|---|",
        ]
        for metric in metrics:
            values = {}
            for name in REFERENTS:
                per_seed = [fold_mean(cells, side, name, s, folds, metric) for s in seeds]
                values[name] = float(np.mean(per_seed))
            valid = VALID_FLOOR[metric]
            note = (
                ", ".join(f"`{v}`" for v in valid) if valid else "**none** — " + FLOOR_NOTE[metric]
            )
            if valid and metric in FLOOR_NOTE:
                note += f" ({FLOOR_NOTE[metric]})"
            report.setdefault("referents", {}).setdefault(side, {})[metric] = values
            lines.append(
                f"| `{metric}` | {values['constant_field']:+.4f} | {values['shuffled']:+.4f} "
                f"| {note} |"
            )
        lines.append("")

    # ---------------------------------------------------------------- contrasts
    for high, low, why in CONTRASTS:
        lines += [
            f"## `{high}` - `{low}` — {why}",
            "",
            "| side | metric | "
            + " | ".join(f"s{s}" for s in seeds)
            + " | mean | envelope | vs it | signs | fold balance | verdict |",
            "|" + "---|" * (9 + len(seeds)),
        ]
        for side in sides:
            for metric in metrics:
                diffs = {
                    s: [
                        cells[(side, high, s, f)][metric] - cells[(side, low, s, f)][metric]
                        for f in folds
                    ]
                    for s in seeds
                }
                per_seed = [float(np.mean(diffs[s])) for s in seeds]
                mean = float(np.mean(per_seed))
                env = max(
                    spread([fold_mean(cells, side, arm, s, folds, metric) for s in seeds])
                    for arm in (high, low)
                )
                flat = [d for s in seeds for d in diffs[s]]
                agree = all(d > 0 for d in flat) or all(d < 0 for d in flat)
                balances = [
                    min(abs(d) for d in diffs[s]) / max(max(abs(d) for d in diffs[s]), 1e-12)
                    for s in seeds
                ]
                balance = min(balances)
                ratio = abs(mean) / env if env > 0 else float("inf")
                verdict = (
                    "signs disagree"
                    if not agree
                    else "inside envelope"
                    if ratio <= 1.0
                    else "one fold carries it"
                    if balance < FOLD_BALANCE_MIN
                    else "STANDS"
                )
                report.setdefault("contrasts", {}).setdefault(f"{high}-{low}", {}).setdefault(
                    side, {}
                )[metric] = {
                    "per_seed": per_seed,
                    "mean": mean,
                    "envelope": env,
                    "ratio": ratio,
                    "signs_agree": agree,
                    "fold_balance": balance,
                    "verdict": verdict,
                }
                lines.append(
                    f"| `{side}` | `{metric}` | "
                    + " | ".join(f"{v:+.4f}" for v in per_seed)
                    + f" | **{mean:+.4f}** | {env:.4f} | {ratio:.1f}x | "
                    + ("agree" if agree else "**disagree**")
                    + f" | {balance:.2f}"
                    + ("" if balance >= FOLD_BALANCE_MIN else " ⚠")
                    + (" | **STANDS** |" if verdict == "STANDS" else f" | {verdict} |")
                )
        lines.append("")

    # ------------------------------------------------ where the verdicts live, and why not here
    lines += [
        "## Where the verdict lives — deliberately not in this script",
        "",
        "This script computes **arithmetic**: per-arm scores, per-arm across-seed spreads, every "
        "pairwise contrast, and the referents, on both gene pools. It states no verdict, and it "
        "used to.",
        "",
        "⚠️ **It carried `deep_starmap`'s pre-registration hardcoded** — primary "
        "`marker_depth_r`, contrast A1-A3, clearances read against §4.2a per-contrast envelopes "
        "— and printed a SUPPORT / REFUTATION / void reading under it. Run on the `cosmx` "
        "replication, whose pre-registration names a **different** primary (`morans_pearson`, "
        "contrast **A2-A3**, held-out) against a single **shared** §4.2b envelope, it produced a "
        "confident verdict for the wrong experiment, under a title naming the wrong dataset. "
        "**Two pre-registrations in one script is how a reader applies the wrong one**, which is "
        "the same argument that kept the envelope computation in its own instrument. Removed "
        "2026-09-07.",
        "",
        "Each experiment's criteria and verdict are stated in "
        "`progress/t09_inference_and_calibration.md`, beside the pre-registration they belong "
        "to. For the replication, `scripts/t09_zeroshot_envelopes.py` applies them — including "
        "§4.2d's two envelope constructions, which this script never computed.",
        "",
    ]

    # ------------------------------------------------- clearances against the usable floor
    floor_name = VALID_FLOOR[SECONDARY_METRIC][0]
    floor = report["referents"][PRIMARY_SIDE][SECONDARY_METRIC][floor_name]
    shared2 = max(
        [report["scores"][PRIMARY_SIDE][SECONDARY_METRIC][a]["spread"] for a in ARMS]
        + [
            spread([fold_mean(cells, PRIMARY_SIDE, r, s, folds, SECONDARY_METRIC) for s in seeds])
            for r in REFERENTS
        ]
    )
    secondary = {}
    for arm in ARMS:
        mean = float(np.mean(report["scores"][PRIMARY_SIDE][SECONDARY_METRIC][arm]["per_seed"]))
        secondary[arm] = {
            "mean": mean,
            "over_floor": mean - floor,
            "envelope": shared2,
            "ratio": abs(mean - floor) / max(shared2, 1e-12),
        }
    report["secondary"] = {
        "metric": SECONDARY_METRIC,
        "floor_name": floor_name,
        "floor": floor,
        "shared_envelope": shared2,
        "arms": secondary,
    }
    lines += [
        f"## Clearances against the usable floor — `{SECONDARY_METRIC}`, `{PRIMARY_SIDE}` genes",
        "",
        "Arithmetic, not a verdict: each arm's distance above the floor, in units of the shared "
        "envelope. Whether a clearance means anything depends on which pre-registration is being "
        "read and whether that metric was named in it in advance — a metric promoted to primary "
        "*because* it produced a result is not a test, and this table cannot tell you which case "
        "you are in. See the section above.",
        "",
        f"Floor is `{floor_name}` ({floor:+.4f}) — the constant field is degenerate on this "
        f"metric. Shared envelope **{shared2:.4f}** (`specs/10` §4.2b).",
        "",
        "| arm | mean | over floor | shared envelope | vs it |",
        "|---|---|---|---|---|",
    ]
    for arm, c in secondary.items():
        lines.append(
            f"| {arm} ({ARM_LABEL[arm]}) | {c['mean']:+.4f} | {c['over_floor']:+.4f} | "
            f"{c['envelope']:.4f} | **{c['ratio']:.2f}x** |"
        )
    lines.append("")
    if args.ceiling_secondary:
        lines += _room(args.ceiling_secondary, SECONDARY_METRIC, secondary, floor)

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        Path(args.out).with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out} and {Path(args.out).with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
