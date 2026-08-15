#!/usr/bin/env python3
"""Run GATE 2 on the synthetic gate fixture and write `reports/gate2.md` with plots.

    python scripts/gate2_report.py                    # full report
    python scripts/gate2_report.py --steps 480        # a longer probe
    python scripts/gate2_report.py --seed 1 --out reports/gate2_seed1.md

Every criterion, its measured value, its threshold and its verdict come from
`tests/gate2_criteria.py` — the same module `tests/test_field.py` asserts against, so the
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
from tests.fixtures.synthetic import GATE_EXTENT_UM, make_synthetic_volume  # noqa: E402
from tests.gate1_criteria import GateSection, describe_hardware  # noqa: E402
from tests.gate2_criteria import (  # noqa: E402
    ANGLES_DEG,
    FRACTIONAL_DEPTHS,
    G23_Z_WINDOW,
    PROBE_BATCH,
    PROBE_LR,
    PROBE_STEPS,
    gate2_probes,
    measure_g2_1,
    measure_g2_2,
    measure_g2_3,
    measure_g2_4,
)

# The interpretation below is written by hand against the numbers this script produced on
# 2026-08-15. It is kept here, rather than generated, because "what the numbers mean" is a
# judgement; if the measurements move, this text has to be re-read and rewritten.
INTERPRETATION = """\
### What the numbers mean

All four criteria pass. In plain language: the backbone reconstructs cells lying on an
oblique plane about as well as it reconstructs cells lying on a coronal one, it does not
fall over at a depth it never saw, the z-proximity term does something the competing
method's score cannot, and the retrieval branch is averaging its evidence rather than
copying its nearest neighbour.

**Oblique parity, the gate.** Reconstruction R^2 across the six dihedral angles spans
{r2_min:.4f} to {r2_max:.4f}, and the worst angle ({worst_angle:g} deg) reaches
**{ratio:.1%}** of the axis-aligned value against a required 90%. Every angle is measured on
exactly {common_n} cells drawn with seed {subsample_seed}, and every evaluated cell has its
own source section excluded from retrieval, so neither sample size nor a trivially near
donor is doing the work. The variation across angles is small and **not monotone in the
angle** — 90 deg ({r2_90:.4f}) is the *best* angle, not the worst, and the minimum sits in
the middle of the sweep at {worst_angle:g} deg. That is the signature of sampling scatter
across six {common_n}-cell subsets rather than of a directional bias in the basis: a
triplane whose oblique resolution lagged would degrade steadily from 0 to 90 deg, which is
exactly the failure this gate was written to catch, and it is not what the fixture shows.

**How to read R^2 = {r2_0:.2f}.** This is a *linear* read-out of 32 expression PCs from
[field, retrieval-context] after {steps} steps — a deliberately weak probe, so that the
number reflects the backbone and not a head that could compensate for it. The gate is a
ratio between angles measured with one probe, and both arms of every comparison see the same
budget, so the absolute level is not the quantity under test. It is reported because a probe
that explained nothing would make the ratio meaningless, and {r2_0:.2f} of the variance of
32 PCs is comfortably away from that.

**Depth is interpolated, not memorised.** With the middle section removed from training
entirely, the probe reconstructs its cells at **{g22:.1%}** of what it achieves on the two
neighbouring training sections — above 100%, not merely above the required 80%. There is no
dip at the held-out z, so `fourier_bands_z = {bands_z}` and the TV_z penalty are doing their
job: the field has not carved a step at each section's depth. (The held-out section is also
the *middle* of the stack, the easiest place to interpolate; a consecutive-run holdout is
T10's regime, not this gate's.)

**The z term earns its place — in the wide-gap regime specifically.** Setting
`retrieval_w_z = 0`, which reproduces the competing method's omission exactly, costs at
least {g23_min:.4f} R^2 at the asymmetric fractional depths ({g23_02:+.4f} at 0.2,
{g23_08:+.4f} at 0.8) and only {g23_05:+.4f} at the symmetric depth 0.5, where the two
flanking sections are equidistant and the term has nothing to say. The pattern is exactly
the one the design predicts, and it is a genuine ablation: both arms share a training seed,
so initialisation, batch order and per-step rotations are identical and the retrieval score
is the only difference. Diagnostic G2.3c is the other side — with the *whole* stack
admissible the three deltas are {g23c}, inside the noise, because the nearest section is
always in the pool and in-plane distance alone already ranks it first. The term buys
accuracy when the evidence is far and asymmetric, which is the regime in-silico sectioning
lives in.

**Retrieval has not collapsed to copying.** Mean attention entropy is {g24:.3f} nats against
the required 0.5 log K = {g24_threshold:.3f}. Read the margin the right way round: the
criterion is one-sided, and this probe sits at the opposite extreme — {g24_frac:.1%} of
log K, i.e. near-uniform. The attention is *averaging* its {k} donors, not selecting among
them. That is safe for the gate and unsurprising for a linear probe trained for a few
hundred steps, but it means GATE 2 has shown only that the attention has not collapsed, not
that it has learned to be selective.

### One bug this gate found

`retrieval_candidates_per_section` defaulted to 16 against `retrieval_k = 32`. Only the top
16 in-plane cells of each admissible section entered the ranking, so whenever just two
sections were admissible — a held-out run, the gap-aware dropout, any wide-gap inference —
the candidate union was exactly K, the top-K selected all of it, and **the retrieval score
decided nothing**. The z term was silently inert in precisely the regime it exists for.
G2.3 measured the ablation as a no-op, which is what exposed it. The default is now 64, and
`Config.validate` refuses `retrieval_candidates_per_section < retrieval_k` with the reason
written out. Nothing about the gate's thresholds was touched.
"""

RECOMMENDATION = """\
### Recommendation

GATE 2 passes; T05 may start. Four things carry forward:

1. **The oblique-parity number is {ratio:.3f}, and that is the number for the paper.** It is
   measured on equal-`n` evaluation sets with own-section retrieval excluded, on the 3000 um
   synthetic fixture. It is *not* yet a statement about real tissue, and T10's E3 is where it
   becomes one.
2. **This gate constrains the backbone, not the generator.** The probe is a linear read-out;
   the flow-matching head (T06) and the SEFL losses (T07) can still break oblique parity, and
   T07's `L_cross` is where intersection consistency gets its own measurement. Re-check the
   ratio after T07 rather than assuming it survives.
3. **The attention is near-uniform** ({g24_frac:.1%} of log K). G2.4 is satisfied at the
   opposite extreme from collapse. T06 should watch this number move *down* as the head
   learns to select, and treat a drop below the 0.5 log K line as the collapse alarm the
   criterion was written for.
4. **Open risk R1 (`ell_z` reads high) is untouched by this gate.** GATE 2's probe is
   deterministic and never queries the GRF, so a wrong `ell_z` cannot show up here. It stays
   open, owed to T07, exactly as `reports/gate1.md` left it.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="gate seed (default 0)")
    parser.add_argument(
        "--steps", type=int, default=PROBE_STEPS, help=f"probe steps (default {PROBE_STEPS})"
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "gate2.md")
    return parser.parse_args(argv)


def _table(sections: list[GateSection]) -> str:
    rows = [
        "| Criterion | What was measured | Measured | Required | Verdict |",
        "|---|---|---|---|---|",
    ]
    for section in sections:
        for c in section.criteria:
            required = "-" if c.threshold is None else f"`{c.comparison} {c.threshold:g}`"
            note = f"<br><sub>{c.note.replace('|', chr(92) + '|')}</sub>" if c.note else ""
            description = c.description.replace("|", chr(92) + "|")
            rows.append(
                f"| **{c.key}** | {description}{note} | {c.measured:.6g}{c.unit} | "
                f"{required} | **{c.verdict}** |"
            )
    return "\n".join(rows)


def _plot_angles(section: GateSection, path: Path) -> None:
    art = section.artifacts
    scores = art["r2"]
    sets = art["sets"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    angles = list(ANGLES_DEG)
    axes[0].plot(angles, [scores[a] for a in angles], "o-", label="probe $R^2$")
    axes[0].axhline(
        0.9 * scores[0.0], color="C3", ls="--", label=r"$0.90 \times R^2(0^\circ)$ (the gate)"
    )
    axes[0].axhline(scores[0.0], color="k", ls=":", label=r"$R^2(0^\circ)$")
    axes[0].set_xlabel("dihedral angle to the sectioning plane (deg)")
    axes[0].set_ylabel(r"$R^2$ on the equal-$n$ evaluation set")
    axes[0].set_title(f"G2.1 oblique parity (n = {sets.common_n} per angle)")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(angles)), [sets.pre_subsample_n[a] for a in angles], color="C0")
    axes[1].axhline(sets.common_n, color="C3", ls="--", label=f"common n = {sets.common_n}")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(len(angles)), [f"{a:g}" for a in angles])
    axes[1].set_xlabel("dihedral angle (deg)")
    axes[1].set_ylabel("cells within thickness/2 of the plane")
    axes[1].set_title("Evaluation-set size before subsampling")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_wz(section: GateSection, path: Path) -> None:
    art = section.artifacts
    fractions = list(FRACTIONAL_DEPTHS)
    width = 0.35
    positions = np.arange(len(fractions))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        positions - width / 2,
        [art["with_z"][f] for f in fractions],
        width,
        label=r"with $w_z$",
    )
    ax.bar(
        positions + width / 2,
        [art["without_z"][f] for f in fractions],
        width,
        label=r"$w_z = 0$ (ablation A5)",
    )
    ax.set_xticks(positions, [f"{f:g}" for f in fractions])
    ax.set_xlabel("fractional depth between the two flanking sections")
    ax.set_ylabel(r"probe $R^2$")
    ax.set_title("G2.3 the z-proximity term at asymmetric vs symmetric depths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_losses(sections: dict[str, GateSection], path: Path) -> None:
    losses = sections["G2.1"].artifacts["losses"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(losses)) * 10, losses, "-")
    ax.set_xlabel("probe step")
    ax.set_ylabel("training loss (MSE on 32 PCs + TV$_z$)")
    ax.set_title("The probe's convergence")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _interpretation(sections: dict[str, GateSection], cfg: Config, steps: int) -> str:
    """Fill the hand-written interpretation with the numbers actually measured."""
    by_key: dict[str, Any] = {c.key: c for section in sections.values() for c in section.criteria}
    g21 = sections["G2.1"].artifacts
    g23 = sections["G2.3"].artifacts
    scores = g21["r2"]
    return INTERPRETATION.format(
        r2_min=min(scores.values()),
        r2_max=max(scores.values()),
        r2_0=scores[0.0],
        r2_90=scores[90.0],
        worst_angle=g21["worst_angle"],
        ratio=g21["ratio"],
        common_n=g21["sets"].common_n,
        subsample_seed=g21["sets"].seed,
        steps=steps,
        g22=by_key["G2.2a"].measured,
        bands_z=cfg.fourier_bands_z,
        g23_min=by_key["G2.3a"].measured,
        g23_02=g23["deltas"][0.2],
        g23_05=g23["deltas"][0.5],
        g23_08=g23["deltas"][0.8],
        g23c=", ".join(
            f"{g23['full_stack_deltas'][f]:+.4f} at f = {f:g}" for f in FRACTIONAL_DEPTHS
        ),
        g24=by_key["G2.4a"].measured,
        g24_threshold=by_key["G2.4a"].threshold,
        g24_frac=by_key["G2.4b"].measured,
        k=cfg.retrieval_k,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = Config()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"building the gate fixture ({GATE_EXTENT_UM:.0f} um) ...", flush=True)
    vol, gt = make_synthetic_volume(seed=0, extent_xy=GATE_EXTENT_UM)

    print(f"training the probes ({args.steps} steps each) ...", flush=True)
    gate2_probes(cfg, vol, seed=args.seed, steps=args.steps)

    sections: dict[str, GateSection] = {}
    for name, thunk in (
        ("G2.1", lambda: measure_g2_1(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.2", lambda: measure_g2_2(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.3", lambda: measure_g2_3(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.4", lambda: measure_g2_4(cfg, vol, seed=args.seed, steps=args.steps)),
    ):
        print(f"measuring {name} ...", flush=True)
        sections[name] = thunk()
        for criterion in sections[name].criteria:
            print("   ", criterion.summary(), flush=True)

    figures = out.parent / "gate2_figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_angles(sections["G2.1"], figures / "g2_1_oblique_parity.png")
    _plot_wz(sections["G2.3"], figures / "g2_3_z_proximity.png")
    _plot_losses(sections, figures / "g2_probe_convergence.png")

    passed = all(section.passed for section in sections.values())
    failing = [c.key for section in sections.values() for c in section.criteria if not c.passed]
    verdict = "PASS" if passed else "FAIL"
    failed_note = "" if passed else " — failing criteria: " + ", ".join(failing)
    sets = sections["G2.1"].artifacts["sets"]
    ratio = sections["G2.1"].artifacts["ratio"]
    entropy_frac = next(c for c in sections["G2.4"].criteria if c.key == "G2.4b").measured

    # The evaluation-set contract's table rows, built here so the report body below stays
    # inside the line limit. Every one of them is required by the spec's definition of done.
    membership_row = (
        "every training-section cell within `thickness / 2` of the query plane, "
        f"pooled across all {vol.n_sections} sections"
    )
    pre_row = ", ".join(f"{a:g} deg: {sets.pre_subsample_n[a]}" for a in ANGLES_DEG)
    floor_row = (
        f"`Config.gate2_min_cells_per_angle = {cfg.gate2_min_cells_per_angle}` — the common "
        "`n` is above it, so the fixture's slabs were **not** thickened and **no angle was "
        "dropped**"
    )
    exclusion_row = (
        "**excluded at every angle** (`Config.retrieval_exclude_source_section = "
        f"{cfg.retrieval_exclude_source_section}`); criterion G2.1c is the standing check "
        "that the exclusion is still plumbed through"
    )
    z_window_row = (
        "G2.3's 0.2 and 0.8 configurations put one flank four spacings away, outside the "
        f"default window of {cfg.retrieval_z_window:g}; left there the ablation would be "
        "measuring `retrieval_z_window` instead. Applied identically to both arms"
    )
    fractions_row = ", ".join(f"{f:g}" for f in FRACTIONAL_DEPTHS)

    body = f"""# GATE 2 — anatomical field + retrieval cross-attention (T04)

**Verdict: {verdict}**{failed_note}

Generated by `python scripts/gate2_report.py --seed {args.seed} --steps {args.steps}` on
{datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")}, Python {platform.python_version()} on
{platform.platform()} ({describe_hardware()}).

**Fixture:** `make_synthetic_volume(seed=0, extent_xy={GATE_EXTENT_UM:.0f})` —
{vol.n_sections} sections x {vol.sections[0].n_cells} cells x {vol.n_genes} genes,
{vol.median_spacing:.0f} um spacing, {vol.sections[0].thickness:.0f} um slabs,
{vol.bbox[1, 0] - vol.bbox[0, 0]:.0f} um in-plane extent, median nearest-neighbour distance
{vol.median_nn_dist:.1f} um, ground-truth correlation lengths {gt.autocorr_length_xy:.0f} um
in-plane and {gt.autocorr_length_z:.0f} um along z. The same fixture GATE 1 was measured on.

**Probe:** `TriplaneField` + `RetrievalAttention` -> a **linear** head predicting the top
{cfg.expr_pca_dim} expression PCs. {args.steps} Adam steps, batch {PROBE_BATCH}, lr {PROBE_LR},
rotation augmentation live, TV_z weighted at `Config.tv_z_weight = {cfg.tv_z_weight:g}`.
Deliberately weak: a nonlinear head could compensate for a directionally biased field, and
the gate would then be measuring the head. The full generative heads do not exist yet — this
isolates the backbone's representational quality, which is what T04 is responsible for.

## The evaluation-set contract (SPEC_QUESTIONS C1, stated in full as the spec requires)

| | |
|---|---|
| Membership rule | {membership_row} |
| Slab half-thickness | **{sets.half_thickness:g} um** |
| Angles | {", ".join(f"{a:g}" for a in ANGLES_DEG)} degrees to the sectioning plane |
| Pre-subsample `n` per angle | {pre_row} |
| Common `n` after subsampling | **{sets.common_n}** (the 90 deg set, the thinnest) |
| Subsample seed | **{sets.seed}** |
| Floor | {floor_row} |
| Own-section retrieval | {exclusion_row} |

**What the angle does and does not change.** The probe is a function of *position* — the
query plane is not an input to it — so the angle enters through membership only: which cells
are evaluated. That is what makes the criterion a statement about the field rather than
about a plane-conditioned decoder. R^2 is variance explained, and a 0 deg strip is a single
section (its target variance is entirely in-plane) while a 90 deg strip spans the whole stack
(much of its target variance is along z, the axis the triplane resolves at
`triplane_res_z = {cfg.triplane_res_z}` rather than `triplane_res_xy = {cfg.triplane_res_xy}`).
A backbone whose z resolution lagged its in-plane resolution would therefore explain less of
the 90 deg strip and the ratio would fall. It does not.

## Criteria

{_table(list(sections.values()))}

## Figures

![G2.1 oblique parity](gate2_figures/g2_1_oblique_parity.png)

![G2.3 the z-proximity term](gate2_figures/g2_3_z_proximity.png)

![probe convergence](gate2_figures/g2_probe_convergence.png)

## Measurement constants

Properties of the *measurement*, not of the model, so they live in `tests/gate2_criteria.py`
rather than in `Config` — a gate that read its own sample size out of the config could be
made to pass by editing the config.

| Constant | Value | Why |
|---|---|---|
| `PROBE_STEPS` | {args.steps} | both arms of every comparison see the same budget |
| `PROBE_BATCH` / `PROBE_LR` | {PROBE_BATCH} / {PROBE_LR} | |
| `SUBSAMPLE_SEED` | {sets.seed} | the equal-`n` draw |
| `G23_Z_WINDOW` | {G23_Z_WINDOW:g} x median spacing | {z_window_row} |
| `FRACTIONAL_DEPTHS` | {fractions_row} | two flanking sections left in the pool |

{_interpretation(sections, cfg, args.steps)}
{RECOMMENDATION.format(ratio=ratio, g24_frac=entropy_frac)}
## Reproducing

```
python scripts/gate2_report.py                 # this report
pytest tests/test_field.py -m gate             # the same numbers, as assertions
```
"""
    out.write_text(body, encoding="utf-8")
    print(f"\nwrote {out} — gate {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
