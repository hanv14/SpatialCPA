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

**G2.1d fails at {fixed_ratio:.3f} against a required 0.90, and the failure is in the
evaluation contract, not in the backbone.** That claim is not an excuse; it is the
conclusion of two measurements reported below, either of which could have gone the other
way. Everything else passes.

**The two denominators, and why the verdict flipped.** G2.1a divides each angle's R^2 by its
**own** set's variance — the spec's ratio as literally written — and reports
**{ratio:.3f}**. Those denominators are not the same quantity, so the ratio partly answers
"how much variance was there to explain". G2.1d fixes the denominator to the per-cell target
variance over all {n_train_cells} training cells (V = {fixed_variance:.4f}) and reports
**{fixed_ratio:.3f}**. The move from {ratio:.3f} to {fixed_ratio:.3f} is {movement}, and it
is the fixed-denominator number that a paper can quote, so **the gate does not pass on the
number that matters**.

**Where the remaining gap comes from — measured, not argued.** Fixing the denominator removed
one confound and exposed a larger one. Under the C1 membership rule a 0 degree query plane
through the volume's centre selects **exactly one section**, the middle one, while every
oblique plane draws about {edge_pct:.0f}% of its cells from the two **edge** sections. The
per-section fixed R^2 is

{per_section_table}

— the edge sections at {edge_lo:.3f} and {edge_hi:.3f} against {interior_lo:.3f}-{interior_hi:.3f}
for the interior. A cell at the top or bottom of the stack has training and retrieval evidence
on one side only, which is a fact about *depth*, not about the angle of the query plane. So
`R^2(theta) / R^2(0 deg)` is comparing the best-supported depth in the stack against a
depth-representative sample.

The attribution is quantitative, not a story. Predicting each angle's R^2 from its section
mix alone — weighting the per-section values by the mix, with the angle playing no part —
gives {predicted_list}. Those are flat across every oblique angle to within {predicted_spread:.4f},
and they reproduce the measured values. **The angle-dependence in G2.1d is depth mix.**

**Diagnostic G2.1e removes it**: take the 0 degree arm over the coronal planes at *every*
section rather than the single central one, so both sides of the ratio are
depth-representative. The ratio is then **{depth_ratio:.3f}**. Nothing about the model
changed between G2.1d and G2.1e — only which cells the denominator was measured on.

**The spec's own remedy was run and did nothing.** specs/04 says that on a G2.1 failure the
first step is to raise `n_plane_orientations` 4 -> 8 and re-run. Measured, on the same
fixture and seed (`n_plane_orientations=8`, 343 s against 215 s for the doubled parameter
count):

| | P = 4 | P = 8 |
|---|---|---|
| G2.1a per-set ratio | 0.9410 | 0.9419 |
| G2.1d fixed ratio | **0.8858** | **0.8867** |
| G2.1e depth-matched | 0.9596 | 0.9601 |

Doubling the orientation ensemble moves the gate number by **+0.0009**. If oblique parity
were limited by the basis concentrating capacity on axis-aligned planes — the failure this
gate exists to catch — doubling the ensemble is precisely the intervention that should have
moved it. It did not. The spec's remedy 2, that rotation augmentation reaches coords, planes,
retrieval and GRF queries, is enforced by construction: `RotationContext` refuses to exit
with a required channel un-transformed, and `test_rotation_augmentation_is_not_inert` asserts
the triplane lookup actually moves.

**How to read R^2 = {r2_0:.2f}.** A *linear* read-out of 32 expression PCs from
[field, retrieval-context] after {steps} steps — deliberately weak, so the number reflects
the backbone and not a head that could compensate for it. The gate is a ratio, and both arms
of every comparison see the same budget, so the absolute level is not the quantity under
test.

**Depth is interpolated, not memorised.** With the middle section removed from training
entirely, the probe reconstructs its cells at **{g22:.1%}** of what it achieves on the two
neighbouring training sections ({g22_fixed:.1%} on the fixed denominator — the three sets
here are whole sections of one volume, so their variances are close and the choice barely
matters, unlike G2.1). Above 100%, not merely above the required 80%: there is no dip at the
held-out z, so `fourier_bands_z = {bands_z}` and the TV_z penalty are doing their job.

**The z term earns its place — in the wide-gap regime specifically.** Setting
`retrieval_w_z = 0`, which reproduces the competing method's omission exactly, costs at
least {g23_min:.4f} R^2 at the asymmetric fractional depths ({g23_02:+.4f} at 0.2,
{g23_08:+.4f} at 0.8) and only {g23_05:+.4f} at the symmetric depth 0.5, where the two
flanking sections are equidistant and the term has nothing to say. It is a genuine ablation:
both arms share a training seed, so initialisation, batch order and per-step rotations are
identical and the retrieval score is the only difference. Diagnostic G2.3c is the other side
— with the *whole* stack admissible the three deltas are {g23c}, inside the noise, because
the nearest section is always in the pool and in-plane distance alone already ranks it first.
This is why specs/10 now requires ablation A5 to be run in the wide-gap regime: run
whole-stack, A5 would report a null result for a term that demonstrably works.

**Retrieval has not collapsed to copying.** Mean attention entropy is {g24:.3f} nats against
the required 0.5 log K = {g24_threshold:.3f}. Read the margin the right way round: the
criterion is one-sided, and this probe sits at the opposite extreme — {g24_frac:.1%} of
log K, i.e. near-uniform. The attention is *averaging* its {k} donors, not selecting among
them. GATE 2 has shown only that the attention has not collapsed. specs/06 now requires T06
to drive this number **down** by at least 0.05 log K while staying above the collapse line.

### Two bugs this gate found

**1. `retrieval_candidates_per_section` was 16 against `retrieval_k = 32`.** Only the top 16
in-plane cells of each admissible section entered the ranking, so whenever just two sections
were admissible — a held-out run, the gap-aware dropout, any wide-gap inference — the
candidate union was exactly K, the top-K selected all of it, and **the retrieval score
decided nothing**. The z term was silently inert in precisely the regime it exists for. G2.3
measured the ablation as a no-op, which is what exposed it. The default is now 64, and
`Config.validate` refuses `retrieval_candidates_per_section < retrieval_k`.

That rule is necessary but not sufficient, because **the invariant is about the union, not
the per-section cap**: what the top-K selects from is
`candidates_per_section x n_admissible_sections`, and the number of admissible sections is
not a config field — `exclude_z`, the z window, the own-section exclusion and above all the
gap-aware dropout shrink it *per query*, at inference. `RetrievalIndex.query` now also warns
at runtime (`InertScoreWarning`) with the count of affected queries, naming every exclusion
that could have caused it.

**2. The probe cache ignored the config.** `gate2_probes` keyed on
`(id(vol), seed, steps)`, so the first P = 4 vs P = 8 comparison returned the P = 4 probes
for both arms and reported "no change" for a change that was never made — the remedy the
spec mandates on failure would have been silently unrunnable. `Config` is a frozen dataclass
and hashes by value, so it is now part of the key. The P = 8 numbers above are from the
fixed version.

Nothing about the gate's thresholds was touched.
"""

RECOMMENDATION = """\
### Recommendation

**GATE 2 does not pass as specified. T05 does not start until the contract question below is
settled.** specs/04's "Do NOT" is explicit: do not proceed to T05 without G2.1 passing.

The decision that has to be made is *which measurement G2.1 is*, and it is not mine to make
unilaterally, because it changes the number the paper quotes:

1. **Accept G2.1d = {fixed_ratio:.3f} as the verdict.** The gate has failed, and the next step
   is specs/04's remedy 3 — a steerable / equivariant backbone, a design change. **The
   evidence argues against this reading**: the spec's remedy 1 (P = 4 -> 8) moved the number
   by +0.0009, which is what says the residual is not directional capacity, and the section-mix
   attribution accounts for essentially all of it.
2. **Amend the C1 evaluation contract so the 0 degree arm is depth-representative** — pooled
   over the coronal planes at every section rather than the single central one — and re-run.
   That is diagnostic G2.1e, which reads **{depth_ratio:.3f}**. The amendment is defensible on
   its own terms: C1 already exists to stop the ratio measuring sample size, and this is the
   same defect one level down (it measures depth support instead). But it is a change to a
   settled contract, made after seeing the number it changes, so it needs to be a decision on
   the record and not a quiet edit.

My recommendation is **2**, on the strength of the P = 8 null result and the mix attribution,
with the amendment written into SPEC_QUESTIONS as a dated decision and both numbers
({fixed_ratio:.3f} single-plane, {depth_ratio:.3f} depth-matched) reported in the paper. What
I have **not** done is pick 2 and declare a pass.

Four things carry forward either way:

1. **Quote the fixed-denominator ratio, and say which one it is.** G2.1a's {ratio:.3f} is the
   spec's literal formula and is not comparable across angles, because each angle's
   denominator is its own strip's variance. Measured on the 3000 um synthetic fixture; not
   yet a statement about real tissue, which is T10's E3.
2. **This gate constrains the backbone, not the generator.** The probe is a linear read-out;
   the flow-matching head (T06) and the SEFL losses (T07) can still break oblique parity, and
   T07's `L_cross` is where intersection consistency gets its own measurement. Re-check the
   ratio after T07 rather than assuming it survives.
3. **The attention is near-uniform** ({g24_frac:.1%} of log K). G2.4 is satisfied at the
   opposite extreme from collapse. specs/06 now requires T06 to drive it down by at least
   0.05 log K while staying above the 0.5 log K collapse line.
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


def _per_section_table(depth: Any) -> str:
    """The per-section fixed R^2, as a markdown table row pair."""
    order = sorted(depth.per_section)
    header = "| section z (um) | " + " | ".join(f"{depth.section_z[s]:.0f}" for s in order) + " |"
    rule = "|---" * (len(order) + 1) + "|"
    values = "| fixed R^2 | " + " | ".join(f"{depth.per_section[s]:.3f}" for s in order) + " |"
    return "\n".join([header, rule, values])


def _interpretation(sections: dict[str, GateSection], cfg: Config, steps: int) -> str:
    """Fill the hand-written interpretation with the numbers actually measured."""
    by_key: dict[str, Any] = {c.key: c for section in sections.values() for c in section.criteria}
    g21 = sections["G2.1"].artifacts
    g23 = sections["G2.3"].artifacts
    scores = g21["r2"]
    fits = g21["fits"]
    per_cell_sst = [f.sst_set / f.n for f in fits.values()]
    movement = (
        "materially lower"
        if g21["fixed_ratio"] < g21["ratio"] - 0.02
        else "materially higher"
        if g21["fixed_ratio"] > g21["ratio"] + 0.02
        else "essentially unchanged"
    )
    depth = g21["depth"]
    interior = [
        v for s_, v in depth.per_section.items() if s_ not in (0, len(depth.per_section) - 1)
    ]
    edges = [depth.per_section[0], depth.per_section[len(depth.per_section) - 1]]
    oblique_predicted = [v for a, v in depth.predicted.items() if a != 0.0]
    return INTERPRETATION.format(
        r2_min=min(scores.values()),
        r2_max=max(scores.values()),
        r2_0=scores[0.0],
        r2_90=scores[90.0],
        worst_angle=g21["worst_angle"],
        ratio=g21["ratio"],
        edge_pct=100 * float(np.mean([v for a, v in depth.edge_share.items() if a != 0.0])),
        per_section_table=_per_section_table(depth),
        edge_lo=min(edges),
        edge_hi=max(edges),
        interior_lo=min(interior),
        interior_hi=max(interior),
        predicted_list=", ".join(
            f"{a:g} deg {depth.predicted[a]:.4f}" for a in ANGLES_DEG if a != 0.0
        ),
        predicted_spread=max(oblique_predicted) - min(oblique_predicted),
        depth_ratio=(min(v for a, v in g21["r2_fixed"].items() if a != 0.0) / depth.coronal_pooled),
        fixed_ratio=g21["fixed_ratio"],
        fixed_worst_angle=g21["fixed_worst_angle"],
        fixed_variance=g21["fixed_variance"],
        n_train_cells=g21["n_train_cells"],
        spread=max(per_cell_sst) / min(per_cell_sst),
        movement=movement,
        g22_fixed=sections["G2.2"].artifacts["fixed_ratio"],
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
    fixed_ratio = sections["G2.1"].artifacts["fixed_ratio"]
    depth_artifact = sections["G2.1"].artifacts["depth"]
    depth_ratio = (
        min(v for a, v in sections["G2.1"].artifacts["r2_fixed"].items() if a != 0.0)
        / depth_artifact.coronal_pooled
    )
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

    recommendation = RECOMMENDATION.format(
        ratio=ratio,
        fixed_ratio=fixed_ratio,
        depth_ratio=depth_ratio,
        g24_frac=entropy_frac,
    )

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
{recommendation}
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
