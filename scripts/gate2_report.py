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
    measure_augmentation_completeness,
    measure_coronal_arms,
    measure_escalation,
    measure_g2_1,
    measure_g2_2,
    measure_g2_3,
    measure_g2_4,
    measure_ratio_stability,
    measure_support_attribution,
)

# The interpretation below is written by hand against the numbers this script produced on
# 2026-08-15. It is kept here, rather than generated, because "what the numbers mean" is a
# judgement; if the measurements move, this text has to be re-read and rewritten.
INTERPRETATION = """\
### What the numbers mean

**G2.1d fails at {fixed_ratio:.3f} against a required 0.90. The gate has not passed and T05
does not start.** Everything else passes. The escalation section below reports what specs/04
prescribes on a failure, and what it found.

**The two denominators.** G2.1a divides each angle's R^2 by its **own** set's variance — the
spec's ratio as literally written — and reports **{ratio:.3f}**. Those denominators are not
the same quantity across angles, so part of that ratio answers "how much variance was there
to explain". G2.1d fixes the denominator to the per-cell target variance over all
{n_train_cells} training cells (V = {fixed_variance:.4f}), shared by every angle, and reports
**{fixed_ratio:.3f}**. The move is {movement}. The fixed-denominator number is the one a paper
can quote, and it is below threshold.

**How to read R^2 = {r2_0:.2f}.** A *linear* read-out of 32 expression PCs from
[field, retrieval-context] after {steps} steps — deliberately weak, so the number reflects
the backbone and not a head that could compensate for it. The gate is a ratio, and both arms
of every comparison see the same budget, so the absolute level is not the quantity under
test.

**The profile is U-shaped, not monotone in angle.** Fixed-denominator R^2 runs
{r2_fixed_list}: highest at 0 deg, falling to a minimum at {fixed_worst_angle:g} deg, then
recovering monotonically to 90 deg, which is the *second best* angle. This matters for the
diagnosis, because the two failure modes it could indicate make opposite predictions:

* A triplane basis concentrating capacity on axis-aligned planes would be worst at the
  *intermediate* angles and better at both ends — 0 deg is carried by the XY plane, 90 deg by
  XZ/YZ, and 30-45 deg by neither. **The observed U-shape is superficially exactly this
  signature**, and it is the reason the escalation below had to be run rather than argued
  about.
* Depth mix predicts something different: 0 deg is high because it is a single interior
  section, and every oblique angle should then be roughly equal to every other. Measured, the
  section-mix prediction is {predicted_list} — flat to {predicted_spread:.4f}, i.e. it
  accounts for the 0-deg-vs-oblique step and none of the U among the oblique angles.

Neither account is complete on its own, which is why the escalation section separates them.

**Depth is interpolated, not memorised.** With the middle section removed from training
entirely, the probe reconstructs its cells at **{g22:.1%}** of what it achieves on the two
neighbouring training sections ({g22_fixed:.1%} on the fixed denominator). Above 100%, not
merely above the required 80%: there is no dip at the held-out z, so
`fourier_bands_z = {bands_z}` and the TV_z penalty are doing their job.

**The z term earns its place — in the wide-gap regime specifically.** Setting
`retrieval_w_z = 0`, which reproduces the competing method's omission exactly, costs at
least {g23_min:.4f} R^2 at the asymmetric fractional depths ({g23_02:+.4f} at 0.2,
{g23_08:+.4f} at 0.8) and only {g23_05:+.4f} at the symmetric depth 0.5, where the two
flanking sections are equidistant and the term has nothing to say. It is a genuine ablation:
both arms share a training seed, so initialisation, batch order and per-step rotations are
identical and the retrieval score is the only difference. Diagnostic G2.3c is the other side
— with the *whole* stack admissible the three deltas are {g23c}, inside the noise, because
the nearest section is always in the pool and in-plane distance alone already ranks it first.
This is why specs/10 now requires ablation A5 to be run in the wide-gap regime.

**Retrieval has not collapsed to copying.** Mean attention entropy is {g24:.3f} nats against
the required 0.5 log K = {g24_threshold:.3f}. Read the margin the right way round: the
criterion is one-sided, and this probe sits at the opposite extreme — {g24_frac:.1%} of
log K, i.e. near-uniform. The attention is *averaging* its {k} donors, not selecting among
them. specs/06 now requires T06 to drive this **down** by at least 0.05 log K while staying
above the collapse line.

### Two bugs this gate found

**1. `retrieval_candidates_per_section` was 16 against `retrieval_k = 32`.** Only the top 16
in-plane cells of each admissible section entered the ranking, so whenever just two sections
were admissible — a held-out run, the gap-aware dropout, any wide-gap inference — the
candidate union was exactly K, the top-K selected all of it, and **the retrieval score
decided nothing**. G2.3 measured the ablation as a no-op, which is what exposed it. Default
now 64; `Config.validate` refuses a cap below `retrieval_k`; and because the invariant is
about the *union* rather than the per-section cap, `RetrievalIndex.query` also warns at
runtime (`InertScoreWarning`) when a query's admissible union falls to K or below.

**2. The probe cache ignored the config.** `gate2_probes` keyed on `(id(vol), seed, steps)`,
so the first P = 4 vs P = 8 comparison returned the P = 4 probes for both arms and reported
"no change" for a change that was never made — the remedy specs/04 mandates on failure would
have been silently unrunnable. `Config` is frozen and hashes by value; it is now in the key.

Nothing about the gate's thresholds was touched.
"""

ESCALATION = r"""
## Escalation — what specs/04 prescribes on a G2.1 failure, and what it found

### 1. Raise `n_plane_orientations` 4 -> 8, re-measure G2.1d unchanged

**{esc_ratio:.4f}** — still below 0.90. Doubling the orientation ensemble, and with it the
triplane's parameter count, moved the gate number by **{esc_delta:+.4f}**.

That is the direct test of the directional-capacity hypothesis the U-shape suggests, and it
comes back negative. If oblique parity were limited by the basis concentrating capacity on
axis-aligned planes, adding four more maximally-separated orientations is precisely the
intervention that should have moved it.

### 2. Verify the augmentation reaches all four channels

A *partial* rotation — one channel left in the wrong frame while the others move — would
produce this signature, so it is checked by **mutation** rather than by assertion: leave each
channel un-rotated in turn and measure whether anything changes. A channel whose omission
changes nothing is a channel that is not wired, and an invariance assertion cannot detect
that, because an unwired channel satisfies invariance trivially.

| channel left un-rotated | effect |
|---|---|
| coords | {aug_coords:.4f} mean \|delta\| per field feature |
| GRF query points | {aug_grf:.4f} mean \|delta\| per noise channel |
| retrieval neighbourhoods | {aug_retrieval:.1f} of K = {k} neighbours change per cell |
| plane normals | {aug_planes:.4f} max component change |

All four register. The augmentation is complete.

**And it achieved what it exists for.** The trained probe's prediction for a fixed cell,
evaluated under {pose_samples} random poses with the rotation bound to the field, varies by
**{aug_pose:.4f}** — **{aug_pose_frac:.2%}** of the spread of the targets it is predicting.
That is the spec's "a full forward pass is equivariant … (approximately) the same result",
answered with a number. It cannot be exactly zero: the triplane lookup is not invariant by
construction, and that non-invariance *is* the augmentation (SPEC_QUESTIONS B5). Under 1% is
what "the augmentation worked" looks like.

*(Writing this measurement is itself how the partial-rotation failure mode was confirmed to
be detectable: the first version compared an **unbound** field against model-frame
coordinates, so the field read them as data-frame positions, decided 85% of them were outside
the bounding box, and clamped. It reported a coords effect of 0.086 instead of the correct
{aug_coords:.4f}, and a pose spread of 0.070 instead of {aug_pose:.4f} — an order of
magnitude, from exactly the mistake this section exists to rule out.)*

### 3. R^2 for the coronal plane at each of the nine sections

The criterion's own construction, applied nine times instead of once, each at the common
n = {common_n}:

{arms_table}

They **do not** cluster tightly: spread **{arms_spread:.4f}**, against the {tight:g} that
would have rejected the depth-mix account and left {fixed_ratio:.4f} standing on its own. But
read the shape, not just the range — the *interior* seven span {interior_lo:.4f}-{interior_hi:.4f}
(spread {interior_spread:.4f}, just inside that same {tight:g}), and the whole of the spread
is the two **edge** sections at {edge_lo:.4f} and {edge_hi:.4f}. A cell at the top or bottom
of the stack has evidence on one side only, which is a fact about depth, not about the angle
of a query plane.

GATE 2's actual 0 deg arm is the central section at {central_arm:.4f}, against a mean over
all nine of {arms_mean:.4f} — the single-plane baseline flatters the denominator by
**{arms_flatter:.1%}**. Against that mean the worst oblique angle reads
**{arms_ratio:.4f}**. **This is a finding, not a substitute criterion**: it says the C1
contract has a defect, and amending a settled contract after seeing the number it changes is
a decision for the spec's owner.

### 4. Is the residual U real? — the measurement's own resolution

Two further attributions, both reported because they bound how much of the profile is
mechanism and how much is the draw.

**Which cells were sampled.** Stratifying by section alone leaves an unexplained range of
{res_section:.4f} across the oblique angles ({res_section_list}). Adding a
6x6 in-plane grid cuts it to **{res_grid:.4f}** ({res_grid_list}) — a 40% reduction, not an
elimination. An in-plane distance-to-boundary stratification was tried first and **rejected**:
it moved the residual by 0.0008, so the in-plane analogue of the edge-section effect is not
what is happening.

**The draw.** Re-running G2.1d on {repeats} independent equal-`n` draws of the same evaluation
sets, with the probe untouched and only the subsample seed changing, gives a ratio of
**{stab_mean:.4f} +/- {stab_sd:.4f}**, range {stab_min:.4f}-{stab_max:.4f}, with
**{stab_below} of {repeats} draws falling below the 0.90 threshold**. Per angle the standard
deviation reaches {stab_angle_sd:.4f}.

**This is the number that governs how the gate should be read.** The shortfall being judged
is {shortfall:.4f} against a draw-to-draw standard deviation of {stab_sd:.4f}. At the sample
size this fixture supports — n = {common_n}, set by the 90 deg strip, which is every cell it
has rather than a subsample — **the criterion cannot resolve 0.886 from 0.90**. The residual
U among the oblique angles ({res_grid:.4f} after stratification) is the same size as the draw
noise ({stab_angle_sd:.4f} per angle), so it is not evidence of a directional mechanism
either.

That does **not** convert the failure into a pass. It says the fixture is underpowered for
the criterion as written, which is a third finding, alongside the denominator and the depth
mix.
"""

RECOMMENDATION = """\
### Recommendation — stop here

**GATE 2 fails at {fixed_ratio:.4f}. specs/04's escalation is exhausted: eight orientations
changed nothing, and the augmentation is verified complete. Per specs/04 the next step is a
steerable / equivariant backbone, and that is a design decision, not a fix to apply.** T05
does not start.

What the escalation established, so the decision can be made on evidence:

1. **It is not directional capacity.** P = 4 -> 8 moved the gate number by {esc_delta:+.4f}.
2. **It is not a partial rotation.** All four channels register under mutation, and the
   trained probe's prediction for a fixed cell varies by {aug_pose_frac:.2%} of the target
   spread across {pose_samples} random poses.
3. **The 0 deg denominator is unrepresentative.** The nine coronal arms spread
   {arms_spread:.4f}; the central one GATE 2 uses flatters the denominator by
   {arms_flatter:.1%}, and against the mean of all nine the worst oblique angle reads
   {arms_ratio:.4f}.
4. **The criterion cannot resolve the shortfall at this sample size.** Across {repeats}
   equal-`n` draws the ratio is {stab_mean:.4f} +/- {stab_sd:.4f}, with {stab_below} of
   {repeats} below threshold.

Points 3 and 4 are defects in the measurement; point 1 and 2 rule out the two mechanisms the
gate was written to catch. **Taken together they say the honest reading of {fixed_ratio:.4f}
is "the fixture cannot answer this question", not "the backbone is not equivariant enough".**
A steerable backbone is a large commitment to make against a number that is
{stab_below}-of-{repeats} coin flips, so before spending it, the cheaper options are worth
pricing:

* **Thicken the fixture's slabs.** C1 already names this as the response to a common `n`
  below the floor; here `n` clears the floor but not the resolution the criterion needs. It
  is the one change that raises `n` at every angle without touching the contract, and specs/04
  already sanctions it.
* **Amend C1's 0 deg arm to be depth-representative** (SPEC_QUESTIONS C16), which is a
  contract change and needs to be recorded as one.
* **Accept and escalate to a steerable backbone**, which is the spec's literal instruction.

I have not chosen among these. Four things carry forward whichever is chosen:

1. **Quote the fixed-denominator ratio and say which one it is.** G2.1a's {ratio:.3f} is the
   spec's literal formula and is not comparable across angles.
2. **This gate constrains the backbone, not the generator.** T06's flow-matching head and
   T07's SEFL losses can still break oblique parity; re-measure after T07.
3. **The attention is near-uniform** ({g24_frac:.1%} of log K) — specs/06 now requires T06 to
   drive it down by at least 0.05 log K while staying above the collapse line.
4. **Open risk R1 (`ell_z` reads high) is untouched by this gate**, since the probe is
   deterministic and never queries the GRF. Still owed to T07.
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


def _escalation(
    sections: dict[str, GateSection], follow_ups: dict[str, GateSection], cfg: Config
) -> str:
    """Fill the hand-written escalation section with the numbers actually measured."""
    from tests.gate2_criteria import (
        CORONAL_ARM_SPREAD_TIGHT,
        OBLIQUE_PARITY_THRESHOLD,
        POSE_INVARIANCE_SAMPLES,
        SUBSAMPLE_REPEAT_SEEDS,
    )

    by_key: dict[str, Any] = {
        c.key: c for group in (sections, follow_ups) for s_ in group.values() for c in s_.criteria
    }
    arms = follow_ups["G2.1f"].artifacts["arms"]
    index_max = max(arms)
    interior = [v for k, v in arms.items() if 0 < k < index_max]
    values = np.asarray(list(arms.values()))
    stability = np.asarray(follow_ups["G2.1i"].artifacts["ratios"])
    g21 = sections["G2.1"].artifacts
    worst_oblique = min(v for a, v in g21["r2_fixed"].items() if a != 0.0)
    residual_section = follow_ups["G2.1g"].artifacts["residual_section"]
    residual_grid = follow_ups["G2.1g"].artifacts["residual_two_way"]
    section_z = follow_ups["G2.1f"].artifacts.get("section_z", g21["depth"].section_z)

    header = "| section z (um) | " + " | ".join(f"{section_z[k]:.0f}" for k in sorted(arms)) + " |"
    rule = "|---" * (len(arms) + 1) + "|"
    row = "| coronal arm R^2 | " + " | ".join(f"{arms[k]:.4f}" for k in sorted(arms)) + " |"

    return ESCALATION.format(
        esc_ratio=by_key["G2.1-esc-a"].measured,
        esc_delta=by_key["G2.1-esc-b"].measured,
        aug_coords=by_key["G2.1h-a"].measured,
        aug_grf=by_key["G2.1h-b"].measured,
        aug_retrieval=by_key["G2.1h-c"].measured,
        aug_planes=by_key["G2.1h-d"].measured,
        aug_pose=follow_ups["G2.1h"].artifacts["pose_spread"],
        aug_pose_frac=by_key["G2.1h-e"].measured,
        pose_samples=POSE_INVARIANCE_SAMPLES,
        k=cfg.retrieval_k,
        common_n=g21["sets"].common_n,
        arms_table="\n".join([header, rule, row]),
        arms_spread=by_key["G2.1f-a"].measured,
        tight=CORONAL_ARM_SPREAD_TIGHT,
        fixed_ratio=g21["fixed_ratio"],
        interior_lo=min(interior),
        interior_hi=max(interior),
        interior_spread=max(interior) - min(interior),
        edge_lo=min(arms[0], arms[index_max]),
        edge_hi=max(arms[0], arms[index_max]),
        central_arm=arms[index_max // 2],
        arms_mean=float(values.mean()),
        arms_flatter=by_key["G2.1f-b"].measured - 1.0,
        arms_ratio=worst_oblique / float(values.mean()),
        res_section=by_key["G2.1g-a"].measured,
        res_grid=by_key["G2.1g-b"].measured,
        res_section_list=", ".join(f"{a:g} deg {v:+.4f}" for a, v in residual_section.items()),
        res_grid_list=", ".join(f"{a:g} deg {v:+.4f}" for a, v in residual_grid.items()),
        repeats=SUBSAMPLE_REPEAT_SEEDS,
        stab_mean=float(stability.mean()),
        stab_sd=by_key["G2.1i-a"].measured,
        stab_min=float(stability.min()),
        stab_max=float(stability.max()),
        stab_below=int((stability < OBLIQUE_PARITY_THRESHOLD).sum()),
        stab_angle_sd=by_key["G2.1i-b"].measured,
        shortfall=OBLIQUE_PARITY_THRESHOLD - float(stability.mean()),
    )


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
        r2_fixed_list=", ".join(f"{a:g} deg {g21['r2_fixed'][a]:.4f}" for a in ANGLES_DEG),
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

    # The follow-ups a failed G2.1 owes: specs/04's escalation path, plus the diagnosis.
    # Reported, never asserted — none of them is a substitute criterion.
    follow_ups: dict[str, GateSection] = {}
    for name, thunk in (
        (
            "G2.1h",
            lambda: measure_augmentation_completeness(cfg, vol, seed=args.seed, steps=args.steps),
        ),
        ("G2.1f", lambda: measure_coronal_arms(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.1g", lambda: measure_support_attribution(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.1i", lambda: measure_ratio_stability(cfg, vol, seed=args.seed, steps=args.steps)),
        ("G2.1-esc", lambda: measure_escalation(cfg, vol, seed=args.seed, steps=args.steps)),
    ):
        print(f"measuring {name} ...", flush=True)
        follow_ups[name] = thunk()
        for criterion in follow_ups[name].criteria:
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

    from tests.gate2_criteria import POSE_INVARIANCE_SAMPLES, SUBSAMPLE_REPEAT_SEEDS

    esc = {c.key: c for c in follow_ups["G2.1-esc"].criteria}
    arms_stats = {c.key: c for c in follow_ups["G2.1f"].criteria}
    stability_values = np.asarray(follow_ups["G2.1i"].artifacts["ratios"])
    arm_values = np.asarray(list(follow_ups["G2.1f"].artifacts["arms"].values()))
    recommendation = RECOMMENDATION.format(
        ratio=ratio,
        fixed_ratio=fixed_ratio,
        g24_frac=entropy_frac,
        esc_delta=esc["G2.1-esc-b"].measured,
        aug_pose_frac=next(c for c in follow_ups["G2.1h"].criteria if c.key == "G2.1h-e").measured,
        pose_samples=POSE_INVARIANCE_SAMPLES,
        arms_spread=arms_stats["G2.1f-a"].measured,
        arms_flatter=arms_stats["G2.1f-b"].measured - 1.0,
        arms_ratio=(
            min(v for a, v in sections["G2.1"].artifacts["r2_fixed"].items() if a != 0.0)
            / float(arm_values.mean())
        ),
        repeats=SUBSAMPLE_REPEAT_SEEDS,
        stab_mean=float(stability_values.mean()),
        stab_sd=float(stability_values.std(ddof=1)),
        stab_below=int((stability_values < 0.90).sum()),
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
{_escalation(sections, follow_ups, cfg)}
## Escalation and diagnostic measurements

None of these is a gate criterion. They are reported because a failed gate owes an
explanation, and every one of them could have come out the other way.

{_table(list(follow_ups.values()))}

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
