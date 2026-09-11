# 5. Off-axis evaluation, and why it could not be resolved

*Every number here comes from `reports/oblique_demo.md` and its pre-registration. The angles, the
gates, the arms, the preconditions, the outcomes and the rule applied in §5.5 were all fixed before
the scores existed; the seven amendments and nine retractions that got them there are listed in
`reports/oblique_demonstration_preregistration.md` and `reports/retractions.md`.*

## 5.1 The construction

No dataset we know of contains an obliquely-cut section, so both sides of the comparison are built
from real cells and the construction has to be stated before the numbers.

**The evaluation set** is the real cells within a slab of thickness `t` about the tilted plane —
`N` strata covering `fill = t·cos θ / s` of the plane (§3.1).

**The donors** are a *flanking slab*: the same orientation, origin offset along the normal by one
**section spacing**. At 0° that is exactly the adjacent sections, so this is `flanking_copy`'s
construction generalised to an arbitrary orientation, and it is disjoint from the evaluation set by
construction. Both arms draw **only** from cells outside the evaluation slab, and every arm's emitted
coordinates are checked against the ground truth's at run time.

| arm | what it is | what it tests |
|---|---|---|
| `copy-nearest-z` | the nearest section's face pasted on the plane, minus the cells inside the slab | the previous method off-axis. **The baseline.** |
| `resample-pd` | the cells the flanking slab actually contains | the plane's own footprint. **Ours.** |
| `null` | `resample-pd`'s positions, types permuted | the floor |

All three reproduce real cells, so **no fit is involved**: `celltype_localization` touches generated
expression only through pose estimation (§6.3). This section is a claim about layout.

## 5.2 The angle

θ\* = **60°**, the largest angle clearing three criteria none of which reads a score: **G1** (scorable
types ≥ 60% of the coronal plane's — the 60% is ours and labelled so), **G2** (largest type ≥ the
metric's own `max_n` = 250), and **F2** (each stratum at least as wide as the volume's median
nearest-neighbour distance, 8.0 µm — so 75°, 85° and 90° are excluded, their strata being 7.7, 2.5
and 1.8 × 10⁻¹⁵ µm).

Two margins are thin and are printed rather than left as arithmetic: **G1 clears 60° by 0.6 of one
cell type** (6 against a threshold of 5.4), and **the precondition that ultimately excludes 60°
fails by 0.29σ of its own noise.**

## 5.3 The footprint: the baseline is not a section at these angles

Pre-registered with its bands before the measurement was written, the middle band defaulting against
us:

| θ | plane's footprint | `copy-nearest-z` | ratio | outside | `resample-pd` | ratio | outside |
|---|---|---|---|---|---|---|---|
| 30° | 390 µm | 1626 µm | **4.17** | **71%** | 390 µm | 1.00 | 35% |
| 45° | 270 µm | 1328 µm | **4.92** | **75%** | 270 µm | 1.00 | 22% |
| 60° | 213 µm | 939 µm | **4.40** | **72%** | 213 µm | 1.00 | 22% |

**The previous method's off-axis output spans four to five times the plane's own footprint, with
roughly three-quarters of its cells at in-plane positions the plane does not pass through.** It is
not a section at that angle; it is a coronal face pasted onto one. That is the capability gap this
work set out to close, and it is real.

**Our own arm is not co-located either, and the `1.00` column must not be read as saying it is.**
`resample-pd`'s extent *ratio* is 1.00 at every angle, yet **22–35% of its cells lie outside the
ground truth's range**: the two ribbons are the same width and are *offset*, because the donor slab
sits one section-spacing away and the tissue it cuts there is displaced. At 30° a third of our cells
fall outside the target's footprint. This is the honest analogue of `flanking_copy` at a coronal
plane, and it is a real cost of holding the donors out.

## 5.4 Scores

| θ | fill | `copy-nearest-z` | `resample-pd` | difference | ± bound (ours) | `null` |
|---|---|---|---|---|---|---|
| 30° | 0.43 | +0.2485 | +0.1302 | −0.1183 | ± 0.1180 | +0.1424 |
| 45° | 0.35 | +0.3340 | +0.1167 | −0.2173 | ± 0.2841 | +0.0290 |
| 60° | 0.25 | +0.4516 | +0.4054 | −0.0462 | ± 0.5477 | +0.2461 |

60° is **not readable**: its permuted-type null exceeds the ceiling set for it by the calibration in
§5.6. 30° and 45° pass every precondition.

The ± figures are a leave-one-scorable-type-out jackknife. **They are an upper bound on precision,
not confidence intervals, and they are not narrowed.** With 6–9 scorable types, leaving one out
removes 11–17% of the data *and* re-normalises the metric's own `radius`, `scale` and null draws —
not the small, smooth perturbation a jackknife's asymptotics assume. They replaced a cell bootstrap
that shifted the estimate it was bracketing by 0.118 (`retractions.md` R14); the jackknife cannot
shift it, because a jackknife estimates the variance of a statistic and never replaces it.

## 5.5 The result

| θ | difference | combined bound | separation |
|---|---|---|---|
| 30° | −0.1183 | 0.2999 | **0.39σ** |
| 45° | −0.2173 | 0.4171 | **0.52σ** |

**Not one difference reaches a single standard error.**

> **The evaluation cannot distinguish the two arms. This is not a success: the capability was not
> demonstrated, and this section reports why it could not be.**

Two things license reading it that way rather than as a defeat, and both were in place beforehand.
The rule — *a difference whose interval spans zero is reported as not distinguishable, whatever its
point estimate* — was **fixed before any of these numbers existed**. And it is **predicted
independently** by §5.6: if a section with completely randomised cell types scores 0.03–0.24, then
differences of 0.05–0.22 were always going to be inside the noise. Two separate measurements agree.

What is **not** available from this table is the claim that our layout beats the previous method
off-axis, or that it loses to it. Neither is supported.

## 5.6 Why it could not be resolved: four measurements

Three of these are properties of the data and the standard metric rather than of any method, and we
have not found any of them stated in the literature.

1. **The comb limit** (§3.1). An oblique ground truth from `N` serial sections is `N` strata; here
   `fill` runs 0.43 → 0.25 across the scored angles and is exactly 0 at 90°.
2. **The metric's resolution** (§3.2). `blur / radius = √(eps·scale) ≈ 0.26–0.30` — a constant of
   `celltype_localization`, not of any tissue. It distinguishes three to four locations along a
   radius on any dataset.
3. **The scrambled-section floor.** Permuting the ground truth's cell types **among its own cells**
   and scoring it against itself — no method, no donor, no arm — gives **0.03–0.24**, and it does
   **not fall with cell count** (it is as high at n = 1 906 as at n = 250). *We predicted this was
   small-`n` noise and were wrong; the hypothesis is withdrawn as `retractions.md` R17.* **Any
   difference below roughly 0.2 on this metric is inside the range a section carrying no type
   information at all can reach.**
4. **Alignment is underdetermined on elongated clouds.** `align_by_expression` aligned our arm at
   **174°** at 30° — unchanged across two runs whose baselines differed. An oblique strip is a
   ribbon; a ribbon maps onto itself under a half-turn; the rotation search therefore has two
   near-equivalent optima. This is reported as a diagnostic rather than a gate, because
   `evaluate_paper` aligns **every** prediction independently and so every published comparison in
   this benchmark is already cross-pose.

Points 2 and 3 say quantitatively why an evaluation of this kind is not powered to separate two
plausible methods. Point 4 says the pose it assigns them is not stable. Point 1 says the object being
scored is a comb. **Point 5.3 adds that even a resolved comparison here would have been between a
section and an object that is not one.**

## 5.7 What this section claims, and what it does not

**Claims.** That off-axis section generation is **well-defined** under a continuous field and is not
under `nearest-z` — evidenced by §5.3, where the previous method's output is four to five times the
plane's footprint with three-quarters of its cells outside it. That an oblique evaluation on serial
sections is bounded in three independent, quantified ways. That, within those bounds, **the standard
metric cannot separate the two approaches at any angle it can score.**

**Does not claim.** That our layout is better off-axis. That it is worse. That the demonstration
succeeded. One specimen, one metric, four sections, and a comparison the instrument cannot resolve.

**What would resolve it.** Nothing in modelling: a specimen cut as contiguous slabs (raising `fill`),
a holdout that does not remove alternate sections (halving `s`), or a metric whose resolution is
finer than 0.3 of the tissue radius and whose floor under randomised labels is nearer zero. The first
two are preparation; the third is a measurement problem this work has characterised and not solved.
