"""M3 — one table: does flooring `theta` move variance from dispersion into `mu`?

A1 measured the model's **total** between-cell count variance as correct — implied
``sd(log mu)`` 1.4131 against the tissue's 1.3699 on ``deep_starmap`` — with roughly 90 % of it
carried by ``theta`` and ``pi``, which are spatially independent, rather than by ``mu``, which is
not. The ZINB likelihood is close to indifferent to that split at the margin. So the question is
not whether the model can make more variance; it is whether the variance can be **moved**.

``Config.decoder_theta_floor`` removes the cheap half of the trade. This script reads a baseline
and a floored arm and puts every number the pre-registration names in one table.

**Why one table and not two reports.** ``reports/a1_escalation_review.md`` §6.4: a repair that
raises Moran's ``I`` while worsening the benchmark is not a repair, and ``I`` is the metric that
would hide it — the model already beats real tissue on tier-1 ``I`` (+0.5134 against +0.4635) with
a latent 1.28x too smooth. So the pinned ``bench3`` ``paper_*`` metrics sit **beside** ``I``, in
the same row, and the verdict reads both.

Nothing is fitted or generated here. It joins artifacts the two verified instruments already
write:

* ``scripts/t10_chain_diagnostic.py`` -> ``<out>.json``: stage ``I``, ``sd(log mu)``, the
  ``mu_spread`` table, and (with ``--report-theta``) the ``theta`` distribution and binding
  fraction;
* ``scripts/t10_rescore_saved.py`` -> ``<out>.json``: the pinned ``paper_*`` metrics at
  ground-truth-matched density.

Criteria are fixed in ``reports/m3_preregistration.md`` and applied by :func:`m3_verdict`, not by
the reader.

Usage::

    python scripts/t10_reallocation_table.py \\
        --baseline-chain reports/m3_tier1_baseline.json \\
        --floored-chain  reports/m3_tier1_floored.json \\
        --baseline-bench reports/m3_bench_baseline.json \\
        --floored-bench  reports/m3_bench_floored.json \\
        --out reports/m3_reallocation.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

METRICS = (
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_umap_mixing",
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_celltype_localization",
    "paper_gene_mean_spearman",
)

FIDELITY_TOLERANCE = 0.02
"""How far a `paper_*` metric may fall before the repair is refused. `reports/m3_preregistration.md`
§3. Absolute, on metrics that are all correlations or shares in [0, 1]."""


def stage_i(chain: dict, prefix: str) -> float:
    """Median Moran's I of the stage whose name starts with ``prefix``. NaN if absent."""
    for row in chain.get("stages", []):
        if str(row.get("stage", "")).startswith(prefix):
            return float(row["median_I"])
    return float("nan")


def spread(chain: dict, quantity_prefix: str) -> float:
    """``sd(log mu)`` of the named row of the chain report's N2 table. NaN if absent."""
    for row in (chain.get("emission_ablation") or {}).get("mu_spread") or []:
        if str(row.get("quantity", "")).startswith(quantity_prefix):
            return float(row["sd_log_mu"])
    return float("nan")


def structured_share(sd_mu: float, sd_counts: float) -> float:
    """``CV^2(mu) / CV^2(counts)``, both from ``sd(log mu) = sqrt(log(1 + CV^2))``.

    The quantity R12 is stated in — *"only 9-19 % of the emitted count variance survives as
    between-cell structure"* — and the one `reports/a1_escalation_review.md` §3 recomputed
    model-free at 10.4 % against the tissue's >= 42.5 % on `deep_starmap`. It stacks the lognormal
    approximation twice and is labelled as an approximation wherever it is printed.
    """
    if not (math.isfinite(sd_mu) and math.isfinite(sd_counts)):
        return float("nan")
    num, den = math.exp(sd_mu**2) - 1.0, math.exp(sd_counts**2) - 1.0
    return num / den if den > 0 else float("nan")


def bench_metrics(bench: dict) -> dict[str, float]:
    """The pinned ``paper_*`` medians at matched density, from a ``t10_rescore_saved`` sidecar.

    Reads the **matched** pass, never the raw one: a denser point set puts kNN neighbours closer
    and inflates every graph-based metric, so the raw rows are not comparable between arms that
    emit different cell counts.
    """
    rows = bench.get("rows") if isinstance(bench, dict) else bench
    if not rows:
        raise SystemExit("no rows in the rescore sidecar; was it written by t10_rescore_saved.py?")
    if len(rows) > 1:
        raise SystemExit(
            f"the rescore sidecar holds {len(rows)} arms ({[r.get('arm') for r in rows]}); M3 "
            "compares one layout_mode at a time. Re-run it with a single --modes value."
        )
    return {m: rows[0]["matched"].get(m) for m in METRICS}


def m3_verdict(
    base: dict,
    floor: dict,
    binding: float,
    *,
    i_real: float,
    bracket: tuple[float, float],
    role: str,
    escalated: bool,
) -> tuple[str, list[str]]:
    """Apply `reports/m3_preregistration.md` §3's criteria. Returns the verdict and its reasons.

    Two things this encodes that the first version did not, both corrected before M3 ran
    (`reports/n5_and_m3_review.md` §4):

    **`I` rising is not the goal.** On tier-1 the model already sits *above* the tissue
    (+0.5134 against +0.4635), so `d I > 0` there is movement **away** from it — and the
    emission-free ceiling is ~+0.78, so every point a `theta` floor can reach increases the
    distance. The benefit statistic is therefore `d abs(I - I_real)`, which is signed correctly on
    both datasets, and it is a **criterion only in the `benefit` role**. Tier-1 runs as
    `mechanism`: it has a converged fit and no deficit, so it can answer *does flooring `theta`
    move variance into `mu`* and nothing else.

    **CAPACITY-LIMITED is not declarable from one dose.** It closes §10, so it requires
    ``escalated`` — the 75th-percentile floor having been run too. A median floor binds on half
    the pairs and moves them up about 1.9x; "the dose was too small" is a live explanation for a
    null at that dose, and it is a different claim from "the head cannot".
    """
    reasons: list[str] = []
    d_sd = floor["sd_log_mu"] - base["sd_log_mu"]
    d_i = floor["i_counts"] - base["i_counts"]
    d_share = floor["share"] - base["share"]
    d_distance = abs(floor["i_counts"] - i_real) - abs(base["i_counts"] - i_real)
    worst = min(
        (
            (floor["bench"][m] - base["bench"][m], m)
            for m in METRICS
            if isinstance(floor["bench"].get(m), (int, float))
            and isinstance(base["bench"].get(m), (int, float))
        ),
        default=(float("nan"), "—"),
    )
    lo, hi = bracket
    reasons.append(
        f"`sd(log mu)` {base['sd_log_mu']:.4f} -> {floor['sd_log_mu']:.4f} ({d_sd:+.4f}); "
        f"the tissue's bracket is [{lo:.4f}, {hi:.4f}]"
    )
    reasons.append(f"`I(counts)` {base['i_counts']:+.4f} -> {floor['i_counts']:+.4f} ({d_i:+.4f})")
    reasons.append(
        f"distance to the tissue `abs(I - {i_real:+.4f})` "
        f"{abs(base['i_counts'] - i_real):.4f} -> {abs(floor['i_counts'] - i_real):.4f} "
        f"({d_distance:+.4f}) — "
        + ("a criterion" if role == "benefit" else "reported, not a criterion")
    )
    reasons.append(f"structured share {base['share']:.1%} -> {floor['share']:.1%} ({d_share:+.1%})")
    reasons.append(f"worst `paper_*` move {worst[0]:+.4f} on `{worst[1]}`")
    reasons.append(
        f"floor binds on {binding:.1%} of (cell, gene) pairs; role={role}, "
        f"75th-percentile escalation {'RUN' if escalated else 'NOT RUN'}"
    )

    if binding < 0.05:
        return "NULL EXPERIMENT — the floor bound on almost nothing", reasons
    if d_sd < 0.05:
        if not escalated:
            return (
                "UNDER-DOSED — capacity-limited is NOT declarable until the 75th-percentile "
                "floor has run",
                reasons,
            )
        return "CAPACITY-LIMITED — the mu head, not the objective", reasons
    if floor["sd_log_mu"] > hi:
        return "OVERSHOT — mu's spread passed the tissue's upper bound", reasons
    if not (math.isnan(worst[0]) or worst[0] >= -FIDELITY_TOLERANCE):
        return "HARMED — a pinned benchmark metric fell past the tolerance", reasons
    if role == "benefit" and d_distance >= 0.0:
        return "NO BENEFIT — I did not move toward the tissue", reasons
    return ("MECHANISM CONFIRMED" if role == "mechanism" else "REALLOCATION WORKS"), reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-chain")
    ap.add_argument("--floored-chain")
    ap.add_argument("--baseline-bench")
    ap.add_argument("--floored-bench")
    ap.add_argument(
        "--role",
        default="mechanism",
        choices=["mechanism", "benefit"],
        help="mechanism (the default, and what tier-1 can answer): success is sd(log mu) moving "
        "into the tissue's bracket, and the distance to the tissue is reported but is not a "
        "criterion, because tier-1's model already sits above the tissue. benefit: the distance "
        "becomes a criterion. See m3_preregistration.md §3.",
    )
    ap.add_argument(
        "--escalated",
        action="store_true",
        help="the 75th-percentile floor has also been run. Required before CAPACITY-LIMITED may "
        "be declared, since that outcome closes §10 and a median floor is a modest dose.",
    )
    ap.add_argument("--out", default="reports/m3_reallocation.md")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="assert m3_verdict's four branches and the share arithmetic, then exit. No data.",
    )
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()
    missing = [
        flag
        for flag, value in (
            ("--baseline-chain", args.baseline_chain),
            ("--floored-chain", args.floored_chain),
            ("--baseline-bench", args.baseline_bench),
            ("--floored-bench", args.floored_bench),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    def arm(chain_path: str, bench_path: str) -> dict:
        chain = json.loads(Path(chain_path).read_text())
        sd_mu = spread(chain, "mu decoded from the generated h")
        sd_counts = spread(chain, "model counts")
        return {
            "chain": chain,
            "i_counts": stage_i(chain, "4. sampled counts"),
            "i_mu": stage_i(chain, "3. decoded mu"),
            "sd_log_mu": sd_mu,
            "sd_log_counts": sd_counts,
            "share": structured_share(sd_mu, sd_counts),
            "bench": bench_metrics(json.loads(Path(bench_path).read_text())),
            "theta_floor": float((chain.get("run") or {}).get("decoder_theta_floor", float("nan"))),
        }

    base = arm(args.baseline_chain, args.baseline_bench)
    floor = arm(args.floored_chain, args.floored_bench)
    theta = (floor["chain"].get("theta") or {}) or (base["chain"].get("theta") or {})
    binding = float(theta.get("fraction_at_or_below_floor", float("nan")))
    i_real = stage_i(base["chain"], "REF real counts")
    bracket = (
        spread(base["chain"], "mu_oracle"),
        spread(base["chain"], "real counts (Poisson-deconvolved)"),
    )
    if not all(math.isfinite(x) for x in (i_real, *bracket)):
        raise SystemExit(
            "the baseline chain sidecar carries no `REF real counts` stage or no N2 spread table, "
            "so neither the distance to the tissue nor its sd(log mu) bracket can be read. Re-run "
            "the baseline with --emission-ablation (m3_preregistration.md §6)."
        )
    verdict, reasons = m3_verdict(
        base,
        floor,
        binding,
        i_real=i_real,
        bracket=bracket,
        role=args.role,
        escalated=args.escalated,
    )
    lines = [
        "# M3 — the reallocation test",
        "",
        f"Baseline against `decoder_theta_floor = {floor['theta_floor']:g}`, role **{args.role}**."
        " Criteria fixed in",
        "`reports/m3_preregistration.md` before either arm ran; applied here by `m3_verdict`,",
        "not by the reader. In the `mechanism` role the distance to the tissue is **reported and",
        "is not a criterion**, because a dataset whose model already sits above the tissue cannot",
        "answer whether a repair helped.",
        "",
        f"## Verdict: **{verdict}**",
        "",
        *[f"* {r}" for r in reasons],
        "",
        "## The one table",
        "",
        "Moran's `I` and the pinned `bench3` `paper_*` metrics side by side, because a repair that",
        "raises `I` while worsening the benchmark is not a repair and `I` is the metric that would",
        "hide it (`a1_escalation_review.md` §6.4). `paper_*` are medians at ground-truth-matched",
        "density from the pinned evaluator.",
        "",
        "| quantity | baseline | floored | change |",
        "|---|---|---|---|",
        f"| `I(sampled counts)` | {base['i_counts']:+.4f} | {floor['i_counts']:+.4f} | "
        f"{floor['i_counts'] - base['i_counts']:+.4f} |",
        f"| `I(decoded mu)` | {base['i_mu']:+.4f} | {floor['i_mu']:+.4f} | "
        f"{floor['i_mu'] - base['i_mu']:+.4f} |",
        f"| `I(real counts)` — the reference | {i_real:+.4f} | {i_real:+.4f} | — |",
        f"| `sd(log mu)` of the decoded mean | {base['sd_log_mu']:.4f} | {floor['sd_log_mu']:.4f} "
        f"| {floor['sd_log_mu'] - base['sd_log_mu']:+.4f} |",
        f"| `sd(log mu)` implied by the counts | {base['sd_log_counts']:.4f} | "
        f"{floor['sd_log_counts']:.4f} | {floor['sd_log_counts'] - base['sd_log_counts']:+.4f} |",
        f"| **count-level structured share** | **{base['share']:.1%}** | **{floor['share']:.1%}** "
        f"| {floor['share'] - base['share']:+.1%} |",
    ]
    for metric in METRICS:
        b, f = base["bench"].get(metric), floor["bench"].get(metric)
        if not isinstance(b, (int, float)) or not isinstance(f, (int, float)):
            lines.append(f"| `{metric}` | — | — | — |")
            continue
        flag = " 🚩" if f - b < -FIDELITY_TOLERANCE else ""
        lines.append(f"| `{metric}` | {b:.4f} | {f:.4f} | {f - b:+.4f}{flag} |")
    lines += [
        "",
        f"🚩 marks a `paper_*` metric that fell by more than the pre-registered "
        f"{FIDELITY_TOLERANCE:.2f}.",
        "",
        "## What the floor actually bound on",
        "",
        f"**{binding:.1%}** of (cell, gene) pairs, "
        f"{theta.get('genes_with_any_binding', '—')} of {theta.get('n_genes', '—')} genes.",
        "",
        "🚩 **A floor that binds on nothing is a null experiment, not a null result** "
        "(`emission_repair_options.md` §5 names it as candidate A's failure mode). This figure",
        "travels with the verdict.",
    ]
    text = "\n".join(lines)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps(
            {
                "verdict": verdict,
                "reasons": reasons,
                "binding_fraction": binding,
                "role": args.role,
                "escalated": bool(args.escalated),
                "i_real": i_real,
                "bracket": list(bracket),
                "baseline": {k: v for k, v in base.items() if k != "chain"},
                "floored": {k: v for k, v in floor.items() if k != "chain"},
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")
    return 0


def _self_check() -> int:
    """Assert `m3_verdict`'s four branches and the share arithmetic. Seconds, no data."""
    checks: list[tuple[str, bool]] = []

    def arm(sd, i, share, bench):
        return {
            "sd_log_mu": sd,
            "i_counts": i,
            "share": share,
            "bench": dict.fromkeys(METRICS, bench),
        }

    base = arm(0.70, 0.10, 0.10, 0.60)
    kw = {"i_real": 0.30, "bracket": (0.72, 0.95), "role": "mechanism", "escalated": True}
    cases = [
        ("MECHANISM CONFIRMED", arm(0.90, 0.20, 0.30, 0.60), 0.40, kw),
        (
            "CAPACITY-LIMITED — the mu head, not the objective",
            arm(0.71, 0.10, 0.10, 0.60),
            0.40,
            kw,
        ),
        (
            "UNDER-DOSED — capacity-limited is NOT declarable until the 75th-percentile floor "
            "has run",
            arm(0.71, 0.10, 0.10, 0.60),
            0.40,
            {**kw, "escalated": False},
        ),
        (
            "OVERSHOT — mu's spread passed the tissue's upper bound",
            arm(1.10, 0.20, 0.30, 0.60),
            0.40,
            kw,
        ),
        (
            "HARMED — a pinned benchmark metric fell past the tolerance",
            arm(0.90, 0.20, 0.30, 0.50),
            0.40,
            kw,
        ),
        (
            "NULL EXPERIMENT — the floor bound on almost nothing",
            arm(0.90, 0.20, 0.30, 0.60),
            0.01,
            kw,
        ),
    ]
    for want, floored, binding, kwargs in cases:
        got, _ = m3_verdict(base, floored, binding, **kwargs)
        checks.append((f"m3_verdict returns {want.split(' —')[0]}", got == want))

    benefit = {**kw, "role": "benefit"}
    # i_real 0.30, baseline I 0.10 -> distance 0.20. Moving to 0.25 shrinks it, to 0.05 grows it.
    toward, _ = m3_verdict(base, arm(0.90, 0.25, 0.30, 0.60), 0.40, **benefit)
    away, _ = m3_verdict(base, arm(0.90, 0.05, 0.30, 0.60), 0.40, **benefit)
    # the SAME arm that moved away reads as a pass in the mechanism role, which is the point
    away_mech, _ = m3_verdict(base, arm(0.90, 0.05, 0.30, 0.60), 0.40, **kw)
    checks += [
        ("in the benefit role, moving toward the tissue passes", toward == "REALLOCATION WORKS"),
        ("and moving away from it does not", away.startswith("NO BENEFIT")),
        (
            "while the mechanism role reads that same arm as confirmed — the correction that "
            "stops tier-1 scoring a move away from the tissue as a win",
            away_mech == "MECHANISM CONFIRMED",
        ),
        (
            "the null-experiment check comes FIRST, so a bound-on-nothing win is not a win",
            m3_verdict(base, arm(0.90, 0.20, 0.30, 0.60), 0.01, **kw)[0].startswith("NULL"),
        ),
        (
            "a fidelity drop inside the tolerance does not refuse",
            m3_verdict(base, arm(0.90, 0.20, 0.30, 0.59), 0.40, **kw)[0] == "MECHANISM CONFIRMED",
        ),
        (
            "structured_share is CV2(mu)/CV2(counts), not a ratio of sds",
            abs(structured_share(0.7118, 1.4131) - 0.1036) < 1e-3,
        ),
        (
            "and it reproduces the tissue's deep figure",
            abs(structured_share(1.0994, 1.3699) - 0.4247) < 1e-3,
        ),
    ]
    failed = 0
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
        failed += 0 if ok else 1
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
