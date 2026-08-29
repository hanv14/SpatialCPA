"""Does a gate's margin survive repeated seeds? — `specs/09` §3's repeated-seed rule, applied.

The rule says any measurement that reaches a paper claim runs at least ``Config.claim_min_seeds``
seeds and **reports the spread, not a point estimate**. This aggregates the per-seed audit JSONs
and does that, then applies a decision rule that is deliberately stricter than the spec's text:

A margin is reported as **STANDS** only when

1. every seed agrees on the **sign** — one seed disagreeing means the two arms are not separated;
2. the mean margin exceeds the **across-seed spread** (max - min) of that margin — otherwise the
   seeds move the margin by more than the gate does, which is the exact failure the rule exists
   to catch and the one a single-seed table cannot show;
3. the mean margin exceeds the R10 reproducibility envelope, and
4. the mean margin exceeds the largest **within-arm fold spread** — the check
   `t09_audit_starmap._fold_spread` added, kept here so a margin cannot pass on seeds while
   failing on folds.

Anything short of all four is **NOT ESTABLISHED**, which is not the same as refuted and is
reported as such. `specs/09` only requires reporting the spread; the four conditions are this
file's, stated here so a reader can disagree with them and recompute from the JSON.

Every input JSON must carry its own ``seed``. A file written before `t09_audit_starmap.py`
recorded that field raises, naming it, rather than having its seed guessed from a filename
(Convention 6) — re-score that seed instead, which needs no refit because the fit checkpoint is
a resume point.

Usage::

    python scripts/t09_seed_claim.py \\
        reports/t09_audit_deep_text_emb_mode_seed{1,2,3}.json \\
        --out reports/t09_seed_claim_deep_text_emb_mode.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.train.select import METRIC_NAMES

# R10's across-seed reproducibility envelope, measured on the synthetic fixture.
R10_ENVELOPE = 0.0335


def load(paths: list[str]) -> list[dict]:
    """Read the audit JSONs, refusing anything that cannot be compared seed to seed."""
    runs = []
    for name in paths:
        rows = json.loads(Path(name).read_text())
        for r in rows:
            if "seed" not in r:
                raise SystemExit(
                    f"{name} carries no 'seed' field. It was written before "
                    "scripts/t09_audit_starmap.py recorded one, and this comparison is between "
                    "seeds, so the seed cannot be inferred from the filename. Re-score that "
                    "seed — the fit checkpoint is a resume point, so it costs scoring only."
                )
        runs.append({"path": name, "rows": rows})
    return runs


def coherent(runs: list[dict]) -> dict:
    """Refuse to average across runs that differ in anything but the seed."""
    keys = ("dataset", "holdout", "under", "train_steps", "expr_pca_dim", "n_cells", "n_genes")
    seen: dict[str, set] = {k: set() for k in keys}
    options: set[tuple[str, ...]] = set()
    seeds: list[int] = []
    for run in runs:
        options.add(tuple(sorted(str(r["option"]) for r in run["rows"])))
        for r in run["rows"]:
            for k in keys:
                seen[k].add(json.dumps(r.get(k), sort_keys=True))
            if int(r["seed"]) not in seeds:
                seeds.append(int(r["seed"]))
    if len(options) != 1:
        raise SystemExit(f"the runs compare different options: {sorted(options)}")
    for k, values in seen.items():
        if len(values) > 1:
            raise SystemExit(
                f"the runs disagree on '{k}': {sorted(values)}. Only the seed may differ, or "
                "the numbers are not a seed spread."
            )
    if len(seeds) != len(runs):
        raise SystemExit(f"expected one seed per run, got seeds {sorted(seeds)} from {len(runs)}")
    first = runs[0]["rows"][0]
    return {
        # Report in the order the audit measured the arms, not alphabetically: the audit's
        # tables read "won minus lost", and a silently re-sorted pair flips every sign.
        "order": tuple(str(r["option"]) for r in runs[0]["rows"]),
        "dataset": first.get("dataset"),
        "holdout": first.get("holdout"),
        "under": first.get("under"),
        "train_steps": first.get("train_steps"),
        "options": sorted(options)[0],
        "seeds": sorted(seeds),
        "fold_ids": first.get("fold_ids"),
    }


def margins(runs: list[dict], options: tuple[str, ...], metric: str) -> dict:
    """Per-seed signed margin ``a - b`` for one metric, plus the within-arm fold spread."""
    per_seed, spread = {}, 0.0
    for run in runs:
        by_option = {str(r["option"]): r for r in run["rows"]}
        a, b = (by_option[o] for o in options)
        seed = int(a["seed"])
        per_seed[seed] = float(a["mean"][metric]) - float(b["mean"][metric])
        for r in (a, b):
            values = [f[metric] for f in r["per_fold"]]
            spread = max(spread, max(values) - min(values))
    return {"per_seed": per_seed, "fold_spread": spread}


def arm_spreads(runs: list[dict], options: tuple[str, ...], metric: str) -> dict[str, float]:
    """Across-seed spread of **each arm's own score**, per metric. ``{option: max - min}``.

    This is the like-for-like replacement for R10's pooled 0.0335: that number answered "how far
    does re-running one configuration move its score", and this answers it per metric *and* per
    arm. Both refinements matter and the second was not anticipated — a copying arm barely uses
    the fitted weights, so its score is nearly seed-invariant, while a generative arm's is not.
    Measured on tier-1 at seeds 2/3: ``cross-mix`` moves 0.0008-0.0084 across seeds where
    ``zinb-flow`` moves 0.0130-0.0368, up to 27x more. A single pooled envelope applied to both
    is therefore wrong in two directions at once.
    """
    out: dict[str, float] = {}
    for option in options:
        values = [
            float({str(r["option"]): r for r in run["rows"]}[option]["mean"][metric])
            for run in runs
        ]
        out[option] = float(max(values) - min(values))
    return out


def duplicate_gap(base_rows: list[dict], dup_rows: list[dict], metric: str) -> dict[str, float]:
    """Same-seed, different-process absolute difference per arm. ``{option: |a - b|}``.

    R10 recorded that fitting one configuration twice — same config, same seed, different
    process — moved its scores by up to 0.0120. That is the signature of the salted-``hash()``
    seeding bug fixed by ``data.schema.section_seed``, and this measures what is left. A figure
    at or near **zero** says the old 0.0120 was the bug rather than genuine variation, which
    decides how much of the pooled envelope was ever real and how many "inside the envelope"
    verdicts need re-reading.
    """
    a = {str(r["option"]): r for r in base_rows}
    b = {str(r["option"]): r for r in dup_rows}
    return {
        option: abs(float(a[option]["mean"][metric]) - float(b[option]["mean"][metric]))
        for option in sorted(set(a) & set(b))
    }


def verdict(per_seed: dict[int, float], fold_spread: float, envelope: float) -> dict:
    """The four conditions, each reported separately so a reader can disagree with any one."""
    values = np.array([per_seed[s] for s in sorted(per_seed)], dtype=np.float64)
    mean = float(values.mean())
    seed_spread = float(values.max() - values.min())
    signs = {float(np.sign(v)) for v in values if v != 0.0}
    checks = {
        "signs_agree": len(signs) <= 1 and bool(signs),
        "beats_seed_spread": abs(mean) > seed_spread,
        "beats_envelope": abs(mean) > envelope,
        "beats_fold_spread": abs(mean) > fold_spread,
    }
    return {
        "mean": mean,
        "sd": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        "seed_spread": seed_spread,
        "fold_spread": fold_spread,
        "checks": checks,
        "stands": all(checks.values()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("json_paths", nargs="+", help="one audit .json per seed")
    ap.add_argument(
        "--pooled-envelope",
        type=float,
        default=R10_ENVELOPE,
        dest="envelope",
        help="R10's single pooled figure, kept only for the verdict-change comparison. The "
        "envelope this file decides against is measured per metric from the runs themselves.",
    )
    ap.add_argument(
        "--duplicate",
        default=None,
        help="an audit .json for one of the same seeds, fitted in a SEPARATE process. Measures "
        "what is left of R10's same-seed cross-process figure after the section_seed fix.",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs = load(args.json_paths)
    meta = coherent(runs)
    cfg = Config()
    options = meta["order"]
    if len(options) != 2:
        raise SystemExit(f"this compares exactly two arms, got {options}")
    a, b = options
    n_seeds = len(meta["seeds"])
    print(
        f"{meta['dataset']} / {meta['holdout']}: `{a}` vs `{b}` under {meta['under']}, "
        f"{meta['train_steps']} steps, seeds {meta['seeds']}"
    )
    if n_seeds < int(cfg.claim_min_seeds):
        print(
            f"  ⚠ {n_seeds} seeds, below Config.claim_min_seeds={cfg.claim_min_seeds}. Nothing "
            "below reaches a paper claim however it comes out."
        )

    dup_rows = json.loads(Path(args.duplicate).read_text()) if args.duplicate else None
    dup_seed = int(dup_rows[0]["seed"]) if dup_rows else None
    base_rows: list[dict] | None = None
    if dup_rows is not None:
        base_rows = [r for run in runs for r in run["rows"] if int(r["seed"]) == dup_seed]
        if not base_rows:
            raise SystemExit(
                f"--duplicate was fitted at seed {dup_seed}, which is not among the runs "
                f"{meta['seeds']}. A same-seed comparison needs the same seed on both sides."
            )

    results = {}
    for metric in METRIC_NAMES:
        m = margins(runs, options, metric)
        spreads = arm_spreads(runs, options, metric)
        # The per-metric envelope: how far the worse-behaved arm moves across seeds. This is
        # what replaces the pooled 0.0335 as the thing a margin has to beat.
        envelope = max(spreads.values())
        v = verdict(m["per_seed"], m["fold_spread"], envelope)
        v_pooled = verdict(m["per_seed"], m["fold_spread"], args.envelope)
        results[metric] = {
            **v,
            "per_seed": m["per_seed"],
            "arm_spreads": spreads,
            "envelope": envelope,
            "stands_under_pooled": v_pooled["stands"],
            "verdict_changed": v["stands"] != v_pooled["stands"],
            "duplicate_gap": (
                duplicate_gap(base_rows, dup_rows, metric)
                if dup_rows is not None and base_rows is not None
                else None
            ),
        }

    header = " | ".join(f"seed {s}" for s in meta["seeds"])
    lines = [
        f"# Repeated-seed verdict — `{a}` vs `{b}` on `{meta['dataset']}`",
        "",
        f"Holdout **`{meta['holdout']}`**, measured under **{meta['under']}**, "
        f"{meta['train_steps']} steps, LOSO folds "
        f"`{'`, `'.join(meta['fold_ids'] or [])}` ({len(meta['fold_ids'] or [])}), seeds "
        f"{meta['seeds']} ({n_seeds} of `Config.claim_min_seeds`={cfg.claim_min_seeds}). "
        f"Margins are `{a}` minus `{b}`; positive favours `{a}`.",
        "",
        f"| metric | {header} | mean margin | **per-metric envelope** | vs it | margin's own "
        "seed spread | fold spread | verdict | was, vs pooled {0} |".format(args.envelope),
        "|---" * (n_seeds + 8) + "|",
    ]
    for metric in METRIC_NAMES:
        v = results[metric]
        cells = " | ".join(f"{v['per_seed'][s]:+.4f}" for s in meta["seeds"])
        env = v["envelope"]
        ratio = abs(v["mean"]) / env if env else float("inf")
        mark = "**STANDS**" if v["stands"] else "not established"
        was = ("STANDS" if v["stands_under_pooled"] else "not established") + (
            " ← **CHANGED**" if v["verdict_changed"] else ""
        )
        lines.append(
            f"| `{metric}` | {cells} | {v['mean']:+.4f} | **{env:.4f}** | {ratio:.1f}x "
            f"| {v['seed_spread']:.4f} | {v['fold_spread']:.4f} | {mark} | {was} |"
        )

    lines += [
        "",
        "### The envelope, per metric and per arm",
        "",
        "R10's **0.0335** was one pooled figure, measured on the synthetic fixture, applied to "
        "all six metrics and both arms. Measured here on real data it is neither constant across "
        "metrics nor across arms:",
        "",
        "| metric | "
        + " | ".join(f"`{o}` across-seed spread" for o in options)
        + " | envelope used |",
        "|---|" + "---|" * (len(options) + 1),
    ]
    for metric in METRIC_NAMES:
        v = results[metric]
        lines.append(
            f"| `{metric}` | "
            + " | ".join(f"{v['arm_spreads'][o]:.4f}" for o in options)
            + f" | **{v['envelope']:.4f}** |"
        )
    lines += [
        "",
        "A **copying** arm barely uses the fitted weights, so its score is nearly seed-invariant; "
        "a **generative** arm's is not. The margin therefore inherits almost all of its seed "
        "variance from one side, which a pooled envelope cannot express.",
    ]

    if dup_rows is not None:
        lines += [
            "",
            f"### Same seed ({dup_seed}), separate process — what is left of R10's 0.0120",
            "",
            "| metric | " + " | ".join(f"`{o}`" for o in options) + " |",
            "|---|" + "---|" * len(options),
        ]
        worst = 0.0
        for metric in METRIC_NAMES:
            gap = results[metric]["duplicate_gap"] or {}
            worst = max(worst, max(gap.values(), default=0.0))
            lines.append(
                f"| `{metric}` | "
                + " | ".join(f"{gap.get(o, float('nan')):.6f}" for o in options)
                + " |"
            )
        lines += [
            "",
            f"**Largest same-seed cross-process difference: {worst:.6f}.** "
            + (
                "Bitwise identical — R10's 0.0120 was the salted-`hash()` seeding bug entirely, "
                "not run-to-run variation, and every 'inside the envelope' verdict decided "
                "against a figure inflated by it should be re-read."
                if worst == 0.0
                else "Not zero, so something beyond the seeding bug is nondeterministic across "
                "processes. That is a finding in its own right and must be attributed before "
                "this envelope is trusted."
            ),
        ]
    lines += [
        "",
        "**The four conditions.** A margin STANDS only if every seed agrees on the sign, the "
        "mean margin exceeds the spread of the margin itself across seeds, exceeds **that "
        "metric's own envelope** — the worse arm's across-seed spread from the table above, "
        f"not R10's pooled {args.envelope} — and exceeds the largest within-arm fold spread. "
        "Which condition failed, per metric:",
        "",
        "| metric | signs agree | > seed spread | > envelope | > fold spread |",
        "|---|---|---|---|---|",
    ]
    for metric in METRIC_NAMES:
        c = results[metric]["checks"]
        lines.append(
            f"| `{metric}` | "
            + " | ".join(
                "yes" if c[k] else "**no**"
                for k in ("signs_agree", "beats_seed_spread", "beats_envelope", "beats_fold_spread")
            )
            + " |"
        )
    lines += [
        "",
        "`specs/09` §3 requires only that the spread be reported rather than a point estimate; "
        "the four conditions are this file's and are stricter, so a reader who disagrees with "
        "any one of them can recompute from the per-seed columns above.",
        "",
        '**"Not established" is not "refuted."** A margin that fails a condition has not '
        "been shown; it may still be real and under-powered at this number of seeds and folds.",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
        out.with_suffix(".json").write_text(
            json.dumps(
                {"meta": meta, "envelope": args.envelope, "metrics": results}, indent=2, default=str
            )
        )
        print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
