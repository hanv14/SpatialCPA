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
    "marker_field_r": ("constant_field", "shuffled"),
    "marker_depth_r": ("constant_field", "shuffled"),
    "celltype_localization": (),
}
"""Which referents mean something for which metric. See the module docstring for the two
exclusions and how each was established."""

FLOOR_NOTE = {
    "morans_pearson": "constant field is float32 round-off on a zero-variance input, not a floor",
    "gearys_pearson": "constant field is float32 round-off on a zero-variance input, not a floor",
    "umap_mixing": "shuffled is the arm's own score (`_mixing` reads no coordinates); "
    "constant field is a degenerate cloud. No usable floor.",
    "celltype_localization": "gene-free and constant across arms and seeds; not a contrast",
}

PRIMARY_METRIC = "marker_depth_r"
PRIMARY_SIDE = "held_out"
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("seed_files", nargs="+")
    ap.add_argument("--out", default=None, help="destination .md (a .json is written beside it)")
    ap.add_argument("--ceiling", default=None, help="the t09_zeroshot_ceiling .json, for context")
    args = ap.parse_args(argv)

    rows = load(args.seed_files)
    cells = indexed(rows)
    seeds = sorted({int(r["seed"]) for r in rows})
    folds = sorted({str(r["fold"]) for r in rows})
    sides = sorted({str(r["side"]) for r in rows})
    metrics = [m for m in METRIC_NAMES if VALID_FLOOR[m] or m != "celltype_localization"]

    lines: list[str] = [
        "# The four-arm zero-shot comparison on `deep_starmap`",
        "",
        f"Seeds {seeds}, folds {folds}, arms {list(ARMS)}. "
        f"{len(rows)} scored cells. The pre-registration is in "
        "`progress/t09_inference_and_calibration.md`; this file applies it and nothing else.",
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

    # ---------------------------------------------------------------- the verdict
    lines += ["## The pre-registered verdict", ""]
    primary = report["contrasts"]["A1-A3"][PRIMARY_SIDE][PRIMARY_METRIC]
    band = report["referents"][PRIMARY_SIDE][PRIMARY_METRIC]["constant_field"]
    clearances = {}
    # specs/10 §4.2b: a clearance against a referent is read against the LARGEST across-seed
    # envelope in the comparison — every arm plus the referents — and not against the arm's own.
    # One threshold per metric per experiment, so the arm that clears is the one that scored
    # highest rather than the one that happened to vary least.
    shared = max(
        [report["scores"][PRIMARY_SIDE][PRIMARY_METRIC][a]["spread"] for a in ARMS]
        + [
            spread([fold_mean(cells, PRIMARY_SIDE, r, s, folds, PRIMARY_METRIC) for s in seeds])
            for r in REFERENTS
        ]
    )
    for arm in ("A1", "A3", "A4"):
        per_seed = report["scores"][PRIMARY_SIDE][PRIMARY_METRIC][arm]
        mean = float(np.mean(per_seed["per_seed"]))
        clearances[arm] = {
            "mean": mean,
            "over_band": mean - band,
            "own_spread": per_seed["spread"],
            "envelope": shared,
            "ratio": abs(mean - band) / max(shared, 1e-12),
        }
    report["verdict"] = {
        "primary": primary,
        "band": band,
        "shared_envelope": shared,
        "clearances": clearances,
    }

    lines += [
        f"**Primary**: `{PRIMARY_METRIC}` on the `{PRIMARY_SIDE}` genes, A1 - A3 = "
        f"**{primary['mean']:+.4f}**, envelope {primary['envelope']:.4f} "
        f"({primary['ratio']:.1f}x), signs "
        + ("agree" if primary["signs_agree"] else "**disagree**")
        + f", fold balance {primary['fold_balance']:.2f} -> **{primary['verdict']}**.",
        "",
        f"**Against the constant-field band** ({band:+.4f}), read against the shared envelope "
        f"**{shared:.4f}** — the largest across-seed spread in the comparison, arms and "
        "referents together (`specs/10` §4.2b). Each arm's own spread is shown, and does not "
        "set its threshold:",
        "",
        "| arm | mean | over band | own spread | shared envelope | over band / envelope |",
        "|---|---|---|---|---|---|",
    ]
    for arm, c in clearances.items():
        lines.append(
            f"| {arm} ({ARM_LABEL[arm]}) | {c['mean']:+.4f} | {c['over_band']:+.4f} | "
            f"{c['own_spread']:.4f} | {c['envelope']:.4f} | **{c['ratio']:.2f}x** |"
        )
    supported = primary["verdict"] == "STANDS" and primary["mean"] > 0
    # "Clears the band" = above it by more than the SHARED envelope (§4.2b). The two
    # refutation branches are not complements: the architecture branch needs *both* arms clear
    # (text works, W.t is redundant), the idea branch needs *neither* (no route works). One arm
    # clear and the other not satisfies neither, and the pre-registration did not name that
    # case, so it is reported as unresolved rather than rounded into whichever branch is nearer.
    clear = {
        a: clearances[a]["over_band"] > 0 and clearances[a]["ratio"] > 1.0 for a in ("A1", "A3")
    }
    architecture_refuted = not supported and all(clear.values())
    idea_refuted = not any(clear.values())
    report["verdict"]["clears_band"] = clear
    report["verdict"]["outcome"] = (
        "SUPPORT"
        if supported
        else "REFUTATION_ARCHITECTURE"
        if architecture_refuted
        else "REFUTATION_IDEA"
        if idea_refuted
        else "UNRESOLVED"
    )
    a4 = clearances["A4"]
    lines += [
        "",
        "**SUPPORT** requires A1 > A3 with signs agreeing, the margin over the envelope and over "
        "the fold spread, and A1 clearing the band by more than the envelope: "
        + ("**met**" if supported else "**not met**")
        + ".",
        "",
        "**REFUTATION of the architecture** (text works, `W.t` is redundant) would be A1 - A3 "
        "inside its envelope while *both* clear the band: "
        + ("**this is the case**" if architecture_refuted else "**not the case**")
        + ".",
        "",
        "**REFUTATION of the idea** (no route from text to an unseen gene) is neither arm "
        "clearing the band by more than the envelope: "
        + ("**this is the case**" if idea_refuted else "**not the case**")
        + ".",
        "",
        f"**Void condition** — A4 must sit inside the band. A4 is {a4['over_band']:+.4f} from it "
        f"against the shared {a4['envelope']:.4f} envelope ({a4['ratio']:.2f}x): "
        + (
            # A ratio against a vanishing envelope is arithmetic, not evidence: three seeds that
            # agree to the fourth decimal make any offset look infinitely significant. Said
            # rather than divided through.
            "**cannot be read** — the shared across-seed envelope is ~0, so the ratio is a "
            "division by noise. Report the offset itself and get more seeds"
            if shared < 1e-6
            else "**holds**, no leak detected"
            if a4["ratio"] <= 1.0
            else "**FAILS — the run means nothing**"
        )
        + ".",
        "",
    ]
    if report["verdict"]["outcome"] == "UNRESOLVED":
        yes = [a for a in ("A1", "A3") if clear[a]]
        no = [a for a in ("A1", "A3") if not clear[a]]
        lines += [
            "**UNRESOLVED — none of the three pre-registered outcomes applies.** "
            f"{', '.join(yes)} clears the band and {', '.join(no)} does not, so the architecture "
            "branch (which needs *both* clear) and the idea branch (which needs *neither*) are "
            "both false, and support was not met. The pre-registration did not name this case "
            "and it is recorded as unresolved rather than rounded into the nearer branch. What "
            "it says substantively: a route from text to an unseen gene may survive through the "
            f"arm that clears ({', '.join(yes)}), but not through the full architecture.",
            "",
        ]

    if args.ceiling:
        ceiling = json.loads(Path(args.ceiling).read_text())
        held = [c["held_out"] for c in ceiling]
        best = float(np.mean([h["noiseless_ceiling"] for h in held]))
        lines += [
            f"**For scale**: the model-free ceiling on these genes is **{best:.4f}** "
            f"(`{args.ceiling}`). The best arm reaches "
            f"{max(c['mean'] for c in clearances.values()):+.4f}.",
            "",
        ]

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
