#!/usr/bin/env python3
"""Run GATE 1 on the synthetic fixture and write `reports/gate1.md` with plots.

    python scripts/gate1_report.py                 # full report, including the diagnostics
    python scripts/gate1_report.py --no-diagnostics
    python scripts/gate1_report.py --seed 1 --out reports/gate1_seed1.md

Every criterion, its measured value, its threshold and its verdict come from
`tests/gate1_criteria.py` — the same module `tests/test_noise.py` asserts against, so the
report and the test suite cannot disagree about whether the gate passed.

Exits non-zero when the gate fails. A failed gate is a result, not a crash: the report is
written either way, and the exit code is there so nobody automates past it.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatialcpav25_gen.config import Config  # noqa: E402
from spatialcpav25_gen.model.noise import matern_correlation  # noqa: E402
from tests.fixtures.synthetic import make_synthetic_volume  # noqa: E402
from tests.gate1_criteria import (  # noqa: E402
    ELL_SWEEP_FACTORS,
    GateSection,
    measure_fov_diagnostic,
    measure_g1_1,
    measure_g1_2,
    measure_g1_3,
    measure_g1_4,
    measure_latent_sweep,
)

# The interpretation below is written by hand against the numbers this script produced on
# 2026-08-15. It is kept here, rather than generated, because "what the numbers mean" is a
# judgement; if the measurements move, this text has to be re-read and rewritten.
INTERPRETATION = """\
### What the numbers mean

The prior works. Three of the four gate sections pass outright; the fourth passes the two
criteria that test the mechanism and fails the two that test the *shape* of the
length-scale response.

**What works.** The field is the object the design says it is. Its realised covariance
tracks the analytic anisotropic Matern to a mean absolute error of {g11a:.4f} — well inside
the 0.03 the spec asks for — and that error falls like `1/sqrt(M)` across
`M = 1024 -> 2048 -> 4096` ({g11_trend}), which is the signature of a random-Fourier-feature
approximation converging to its kernel rather than of an implementation that happens to
look smooth. Two planes crossing inside the volume receive *bit-identical* noise along the
line they share ({n_line} points, maximum difference exactly 0.0), so the mutual coherence
of two virtual sections is exact by construction and not something a loss has to negotiate
later. The same seed in a separate process reproduces every value bit for bit, and a
million queries at `M = 4096`, `d_h = 64` take {g14b:.2f} s on this 4-core CPU.

**And the decisive part passes.** Substituting the GRF for the i.i.d. prior in the
fixture's own generative map cuts the median per-gene Moran's I error to {g13a:.0%} of what
the i.i.d. prior gives ({grf_err:.4f} vs {iid_err:.4f}) — the spec asks for less than 50% —
and the per-gene correlation between generated and real Moran's I rises from
r = {iid_r:.2f} (i.i.d.) to r = {grf_r:.2f} (GRF), against a threshold of 0.7. Geary's C
moves the same way ({g13f:.0%}). This is the mechanism the whole method rests on, and on
this fixture it does what the design claims: a spatially correlated prior survives the
generative map and shows up as preserved spatial autocorrelation in the counts.

**What fails.** Median `I_gen` is *not* monotone across the full `0.25x - 4x` sweep of
`ell` ({sweep}). It rises to a peak near `1.6x` the fitted length-scale and then falls; the
worst step is {g13c:+.4f} where the criterion requires every step to be positive. The
related criterion — that the fitted `ell` land within 25% of the value that best matches
`I_real` — misses at {g13d:.0%}, because the best match sits at that peak
({best_factor:.2f}x the fitted value) and, on this fixture, median `I_gen` never quite
reaches median `I_real` ({peak_i:.4f} at the peak against {median_i_real:.4f} real).

**Why.** Not because the prior's autocorrelation is non-monotone: measured directly, the
median Moran's I of the field's own channels rises monotonically with `ell` and saturates
near 1 ({latent_sweep}). The non-monotonicity enters with the observable. Moran's I of
expression is a *ratio* — spatially structured variance over total variance — and a
stationary unit-variance field observed through a finite window loses within-window
variance once its correlation length approaches the window size. Over the fixture's 1000 um
section the field's within-section standard deviation falls from
{within_sd_first:.2f} at `0.25x` to {within_sd_last:.2f} at `4x`, so the latent contributes
progressively less of the total expression variance even as its neighbour-scale correlation
approaches 1, and the ratio turns over. Rebuilding the fixture with a 3000 um field of view
and the same cell density — the D1 diagnostic — flattens the effect almost completely
({fov_sweep}), which locates the cause in the ratio of `ell` to the field of view rather
than in the field.

**What this means for what comes next.** The consequence the spec cares about is stated in
its own words: "this is what makes the T09 calibration loop well-posed — if `I_gen` is not
monotone in `ell`, bisection will fail". That risk is real and now quantified. `I_gen(ell)`
is unimodal, with its maximum at roughly `0.2x` the section's in-plane extent; bisection is
well-posed only when bracketed below that maximum, and when the target `I_real` exceeds the
achievable maximum there is no root to find at all — which is precisely the situation on
this fixture. T09's calibrator therefore needs two things it does not currently specify: a
bracket capped at a fraction of the field of view, and an explicit "target unreachable,
returning the maximiser" branch instead of a bisection that walks into a boundary.
"""

RECOMMENDATION = """\
### Recommendation

The gate is failed, and per `CLAUDE.md` the project stops here rather than building T04 on
top of it. Three ways forward, in the order I would take them:

1. **Make T09's calibrator maximum-aware** (my recommendation). The finding is a real
   property of the statistic being calibrated, not a defect in T03: cap the bisection
   bracket at ~0.2x the in-plane extent, detect the maximum, and report "target
   unreachable" rather than converging to a bound. Then re-run G1.3c over the range where
   the calibration actually operates and record the restriction in the spec.
2. **Widen the fixture.** The D1 diagnostic shows a 3000 um field of view nearly removes
   the effect, and real sections are 5-10 mm, where `4x` a fitted `ell` of ~150 um is far
   from the window scale. But changing the fixture to make a gate pass is changing the
   measurement, and that is a decision for a human, not a fix to apply quietly.
3. **Accept the criterion as written and treat the mechanism as unproven.** I do not think
   the evidence supports this: G1.3a and G1.3b, the criteria that test whether a correlated
   prior preserves autocorrelation at all, pass by a wide margin.

Nothing here should be resolved by moving a threshold.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="gate seed (default 0)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "gate1.md")
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="skip the field-of-view diagnostic, which rebuilds a larger fixture",
    )
    return parser.parse_args(argv)


def _table(sections: list[GateSection]) -> str:
    rows = [
        "| Criterion | What was measured | Measured | Required | Verdict |",
        "|---|---|---|---|---|",
    ]
    for section in sections:
        for c in section.criteria:
            required = "-" if c.threshold is None else f"`{c.comparison} {c.threshold:g}`"
            # Pipes appear inside |absolute values|; unescaped they would end the cell.
            note = f"<br><sub>{c.note.replace('|', chr(92) + '|')}</sub>" if c.note else ""
            description = c.description.replace("|", chr(92) + "|")
            rows.append(
                f"| **{c.key}** | {description}{note} | {c.measured:.6g}{c.unit} | "
                f"{required} | **{c.verdict}** |"
            )
    return "\n".join(rows)


def _plot_covariance(section: GateSection, path: Path) -> None:
    art = section.artifacts
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = np.argsort(art["radius"])
    grid = np.linspace(0.0, float(art["radius"].max()), 200)
    axes[0].plot(grid, matern_correlation(grid, 1.5), "k-", lw=2, label="analytic Matern")
    for n_rff, realised in art["realised"].items():
        axes[0].plot(
            art["radius"][order], realised[order], ".", ms=2, alpha=0.35, label=f"M = {n_rff}"
        )
    axes[0].set_xlabel(r"scaled distance $\|p-q\|_\ell$")
    axes[0].set_ylabel("covariance")
    axes[0].set_title("G1.1 realised vs analytic covariance")
    axes[0].legend(markerscale=4)

    ms = list(art["errors"])
    errs = [art["errors"][m] for m in ms]
    axes[1].loglog(ms, errs, "o-", label="measured")
    axes[1].loglog(ms, [errs[0] * (ms[0] / m) ** 0.5 for m in ms], "k--", label=r"$1/\sqrt{M}$")
    axes[1].set_xlabel("M (random Fourier features)")
    axes[1].set_ylabel("mean |realised - analytic|")
    axes[1].set_title("G1.1 error vs M")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_intersection(section: GateSection, path: Path) -> None:
    art = section.artifacts
    fig, ax = plt.subplots(figsize=(7, 3.5))
    t = np.arange(art["xi"].shape[0])
    for channel in range(min(4, art["xi"].shape[1])):
        ax.plot(t, art["xi"][:, channel], lw=1)
    ax.set_xlabel("point along the intersection line")
    ax.set_ylabel(r"$\xi_c(p)$")
    ax.set_title(
        f"G1.2 noise along the line where the planes cross ({art['angle_deg']:.0f} deg apart)\n"
        "both plane pathways give identical values"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_autocorrelation(section: GateSection, latent: GateSection | None, path: Path) -> None:
    art = section.artifacts
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    factors = np.asarray(art["crossing_factors"])
    axes[0].plot(factors, art["crossing_median_i"], "o-", label="median $I_{gen}$")
    axes[0].axhline(art["median_i_real"], color="k", ls="--", label="median $I_{real}$")
    axes[0].axvline(1.0, color="C2", ls=":", label="fitted $\\ell$")
    axes[0].axvline(art["best_factor"], color="C3", ls=":", label="best match")
    if latent is not None:
        axes[0].plot(
            ELL_SWEEP_FACTORS,
            latent.artifacts["medians"],
            "s--",
            color="C4",
            alpha=0.7,
            label="median $I$ of the field itself",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\ell$ / fitted $\ell$")
    axes[0].set_ylabel("median Moran's I")
    axes[0].set_title("G1.3c/d Moran's I vs length-scale")
    axes[0].legend(fontsize=8)

    axes[1].bar(
        [0, 1],
        [float(np.mean(art["iid_errors"])), float(np.mean(art["grf_errors"]))],
        color=["C1", "C0"],
    )
    axes[1].set_xticks([0, 1], ["i.i.d. prior", "GRF prior"])
    axes[1].set_ylabel(r"median $|I_{gen} - I_{real}|$")
    axes[1].set_title("G1.3a autocorrelation error by prior")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_variogram(section: GateSection, cfg: Config, path: Path) -> None:
    fit = section.artifacts["fit"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, lags, gamma, ell, nugget, sill, title in (
        (
            axes[0],
            fit.lags_xy,
            fit.gamma_xy,
            fit.ell[0],
            fit.nugget_xy,
            fit.sill_xy,
            f"in-plane, fitted $\\ell_{{xy}}$ = {fit.ell[0]:.0f} um",
        ),
        (
            axes[1],
            fit.lags_z,
            fit.gamma_z,
            fit.ell[2],
            fit.nugget_z,
            fit.sill_z,
            f"along z, fitted $\\ell_z$ = {fit.ell[2]:.0f} um",
        ),
    ):
        grid = np.linspace(0.0, float(lags.max()) * 1.05, 200)
        ax.plot(lags, gamma, "o", label="empirical")
        ax.plot(
            grid,
            nugget + sill * (1.0 - matern_correlation(grid / ell, cfg.matern_nu)),
            "-",
            label="fitted Matern",
        )
        ax.set_xlabel("lag (um)")
        ax.set_ylabel("semivariance")
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _interpretation(sections: dict[str, GateSection], diagnostics: dict[str, GateSection]) -> str:
    """Fill the hand-written interpretation with the numbers actually measured."""
    by_key: dict[str, Any] = {c.key: c for section in sections.values() for c in section.criteria}
    g13 = sections["G1.3"].artifacts
    g11 = sections["G1.1"].artifacts
    latent = diagnostics.get("D2")
    fov = diagnostics.get("D1")
    return INTERPRETATION.format(
        g11a=by_key["G1.1a"].measured,
        g11_trend=" -> ".join(f"{g11['errors'][m]:.4f}" for m in g11["errors"]),
        n_line=sections["G1.2"].artifacts["line"].shape[0],
        g14b=by_key["G1.4b"].measured,
        g13a=by_key["G1.3a"].measured,
        grf_err=float(np.mean(g13["grf_errors"])),
        iid_err=float(np.mean(g13["iid_errors"])),
        iid_r=by_key["G1.3e"].measured,
        grf_r=by_key["G1.3b"].measured,
        g13f=by_key["G1.3f"].measured,
        sweep=", ".join(
            f"{f:g}x: {v:.4f}"
            for f, v in zip(g13["sweep_factors"], g13["sweep_median_i"], strict=False)
        ),
        g13c=by_key["G1.3c"].measured,
        g13d=by_key["G1.3d"].measured,
        best_factor=g13["best_factor"],
        peak_i=max(g13["crossing_median_i"]),
        median_i_real=g13["median_i_real"],
        latent_sweep=(
            ", ".join(
                f"{f:g}x: {v:.3f}"
                for f, v in zip(ELL_SWEEP_FACTORS, latent.artifacts["medians"], strict=False)
            )
            if latent
            else "diagnostic not run"
        ),
        within_sd_first=latent.artifacts["within_sd"][0] if latent else float("nan"),
        within_sd_last=latent.artifacts["within_sd"][-1] if latent else float("nan"),
        fov_sweep=(
            ", ".join(
                f"{f:g}x: {v:.4f}"
                for f, v in zip(ELL_SWEEP_FACTORS, fov.artifacts["sweep_median_i"], strict=False)
            )
            if fov
            else "diagnostic not run"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print("building the synthetic fixture ...", flush=True)
    vol, gt = make_synthetic_volume(seed=0)

    sections: dict[str, GateSection] = {}
    for name, thunk in (
        ("G1.1", lambda: measure_g1_1(cfg, seed=args.seed)),
        ("G1.2", lambda: measure_g1_2(cfg, vol, seed=args.seed)),
        ("G1.3", lambda: measure_g1_3(cfg, vol, gt, seed=args.seed)),
        ("G1.4", lambda: measure_g1_4(cfg, seed=args.seed)),
    ):
        print(f"measuring {name} ...", flush=True)
        sections[name] = thunk()
        for criterion in sections[name].criteria:
            print("   ", criterion.summary(), flush=True)

    diagnostics: dict[str, GateSection] = {}
    if not args.no_diagnostics:
        print("measuring D2 (the field's own autocorrelation) ...", flush=True)
        diagnostics["D2"] = measure_latent_sweep(cfg, vol, gt, seed=args.seed)
        print("measuring D1 (wider field of view; rebuilds the fixture) ...", flush=True)
        diagnostics["D1"] = measure_fov_diagnostic(cfg, seed=args.seed)
        for section in diagnostics.values():
            for criterion in section.criteria:
                print("   ", criterion.summary(), "|", criterion.note, flush=True)

    figures = out.parent / "gate1_figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_covariance(sections["G1.1"], figures / "g1_1_covariance.png")
    _plot_intersection(sections["G1.2"], figures / "g1_2_intersection.png")
    _plot_autocorrelation(
        sections["G1.3"], diagnostics.get("D2"), figures / "g1_3_autocorrelation.png"
    )
    _plot_variogram(sections["G1.3"], cfg, figures / "g1_3_variogram.png")

    passed = all(section.passed for section in sections.values())
    failing = [c.key for section in sections.values() for c in section.criteria if not c.passed]
    fit = sections["G1.3"].artifacts["fit"]
    verdict = "PASS" if passed else "FAIL"
    failed_note = "" if passed else " — failing criteria: " + ", ".join(failing)
    body = f"""# GATE 1 — 3D Gaussian random field prior (T03)

**Verdict: {verdict}**{failed_note}

Generated by `python scripts/gate1_report.py --seed {args.seed}` on
{datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}, Python {platform.python_version()} on
{platform.platform()}. Fixture: `make_synthetic_volume(seed=0)` —
{vol.n_sections} sections x {vol.sections[0].n_cells} cells x {vol.n_genes} genes,
{vol.median_spacing:.0f} um spacing, {vol.bbox[1, 0] - vol.bbox[0, 0]:.0f} um in-plane extent.
Fitted length-scale: `ell = ({fit.ell[0]:.1f}, {fit.ell[1]:.1f}, {fit.ell[2]:.1f})` um
(ground truth {gt.autocorr_length_xy:.0f} in-plane, {gt.autocorr_length_z:.0f} along z).

## Criteria

{_table(list(sections.values()))}

## Figures

![G1.1 covariance](gate1_figures/g1_1_covariance.png)

![G1.2 intersection](gate1_figures/g1_2_intersection.png)

![G1.3 autocorrelation](gate1_figures/g1_3_autocorrelation.png)

![length-scale fit](gate1_figures/g1_3_variogram.png)

## Diagnostics (not gate criteria)

{_table(list(diagnostics.values())) if diagnostics else "_skipped (`--no-diagnostics`)._"}

{_interpretation(sections, diagnostics)}
{RECOMMENDATION}
## Reproducing

```
python scripts/gate1_report.py            # this report
pytest tests/test_noise.py -m gate        # the same numbers, as assertions
```
"""
    out.write_text(body, encoding="utf-8")
    print(f"\nwrote {out} — gate {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
