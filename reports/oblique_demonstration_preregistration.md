# Pre-registration — the oblique demonstration on `merfish_thick_hypothalamus`

**Committed before the demonstration is built or run.** The arms, the evaluation set, the baseline,
the preconditions and the four outcomes are fixed here. Nothing below may be revised after a number
exists; a revision, if one is needed, is appended as a dated amendment that says what it changes and
why, the way `layout_split_preregistration.md` §6-bis does.

## 0. Why this specimen, and the one thing it does not license

`scripts/angle_budget.py --datasets all` reports `merfish_thick_hypothalamus` clearing **90°** with
**1988 cells** at **10.3 : 1** in-plane to depth, both gates by wide margins. Tier-1 clears **5°** at
**21.6 : 1**. The budget table is the pre-registered instrument, its gates are
`celltype_localization`'s own constants, and it was run across every built dataset at once — so this
specimen was not chosen after seeing which one scored well. It was chosen by geometry, before any
score exists.

**It does not license a cross-dataset comparison.** `specs/10` §4.2a: no number here may be averaged
with, or set against, a tier-1 number. This specimen's `flanking_copy` floor, its oracle and its
null are its own and must all be measured on it. Tier-1 appears in this report only as the
*geometry* contrast — 21.6 : 1 against 10.3 : 1 — never as a score.

## 1. What is being demonstrated

**Claim A — well-definedness.** The method produces a coherent section at an arbitrary orientation.
Carries no number; shown, and labelled as shown.

**Claim B — evaluability.** That section scores, against a baseline measured under identical
constraints, at an angle no published method has reported. This document governs claim B.

## 2. The angles

Five planes, all through the same origin (the volume's in-plane centre at its median **real**
section's z, per `retractions.md` R1):

| θ | role |
|---|---|
| **0°** | the coronal control. `plane-distance` must reproduce `nearest-z` here, on real data, as `tests/test_layout.py` asserts on the fixture. |
| **30°** | first genuinely oblique angle |
| **60°** | |
| **90°** | the budget's limit — orthogonal to the sectioning plane, the hardest case and the one the claim is about |
| **45°** | claim A's figure only, if 45° is not already covered above — **not** re-scored |

Angles are fixed now. **The reported angle is 90° whatever the scores are**, unless a precondition
below fails at 90°, in which case the largest angle passing every precondition is reported *and the
failure is printed beside it*. Picking the best-scoring angle afterwards is forbidden.

## 3. The evaluation set, and the exclusion that makes it honest

`oblique_layout_cost.md` §3c: there is no real obliquely-cut section, but an oblique plane passes
near real cells and **their own measurements are the ground truth for that plane**. That set is
`cells_near_plane(sections, plane)` — the threshold form, not the expanded one.

**The trap §3c names, closed explicitly.** The donors and the ground truth are the same cells, so:

1. The ground truth `G` is the cells in the slab, pooled across sections.
2. Every method arm generates with `exclude_sections = {every section contributing to G}`.
3. **Precondition L1:** `set(donor.section_id) ∩ set(G.section_id) == ∅`, asserted at run time on
   the returned `section_id` arrays — not argued from the call signature. A run that cannot assert
   it does not report a score.
4. **Precondition L2:** no generated coordinate is within `1e-6` µm of a ground-truth coordinate.
   L1 is about provenance; L2 catches a copy arriving by another route.

At 90° on this specimen, `G` is ~1988 cells drawn from many sections, so the exclusion removes most
of the volume. **That is the point, and it is also the demonstration's main risk** — see §6.

## 2-ter. AMENDMENT (2026-09-11, second) — θ\* needs a fill criterion, because G1 and G2 cannot see the comb

*Written before any arm is scored. §2-bis defined θ\* as "the largest angle clearing G1 and G2"; the
runner implemented that faithfully and returned **90°**, at `fill = 0.00`, where the evaluation set
is four parallel lines. That is the angle `the_comb_limit.md` says is unavailable, so the two
documents contradicted each other and the fault is in §2-bis's definition — G1 and G2 are **counts**,
and the comb document had just finished proving counts are blind to the comb. Mine to fix.*

### What the metric actually does to the comb — measured, not assumed

`celltype_localization`'s Sinkhorn kernel is `exp(−C/eps)` with `C = d²/scale` and `eps = 0.05`,
where `scale` is the median squared distance between ground-truth cells on coordinates normalised by
the tissue radius. In micrometres that is a Gaussian of
`σ = radius·√(eps·scale/2)`. Convolving the comb with that kernel and measuring what survives:

| θ | fill | period | gap | metric's blur | **residual modulation** |
|---|---|---|---|---|---|
| 30° | 0.43 | 115 µm | 65 µm | 119 µm | **1.7 × 10⁻⁴** |
| 45° | 0.35 | 81 µm | 53 µm | 114 µm | **2.3 × 10⁻⁸** |
| 60° | 0.25 | 66 µm | 50 µm | 109 µm | **3.1 × 10⁻¹²** |
| 90° | **0.00** | 58 µm | 58 µm | 106 µm | **zero measure** |

*(At the externally-sourced 28.6 µm thickness. The modulation **falls** with angle because the
strata crowd together — `period = s / sin θ` shrinks — so a fixed blur suppresses them harder.)*

⚠️ **CORRECTED before publication, and worth recording.** The first version of this table read
0.01% / 0.00% / 0.03%, from an FFT convolution on a discrete grid. At 60° that implementation
returned 1.8 × 10⁻⁵, 9.4 × 10⁻⁴, 2.0 × 10⁻⁴ and 7.7 × 10⁻⁵ as the grid went 10⁵ → 8 × 10⁵ points,
against a true value of 3 × 10⁻¹². **A quantity that moves two orders of magnitude with an
implementation parameter is floating-point noise wearing a measurement's clothes**, and it was one
edit from being published as one. The closed form — a square comb's first harmonic
`2|sin(πf)|/(πf)` times the Gaussian's `exp(−2π²σ²/p²)` — is exact, stable and now what the runner
computes.

**Two conclusions, and they point in opposite directions.**

1. **The comb does not damage this metric at any angle — 90° included.** The modulation limit as
   `f → 0` is `2·exp(−2π²σ²/p²)`, also negligible here. So **F1 is withdrawn**: a continuous-fill
   arm is not penalised for filling gaps the statistic cannot see, and `field` returns to the table
   as an ordinary ablation. **And this quantity therefore discriminates nothing** — it cannot be
   the criterion, which is why F2 below rests on measure zero instead.
2. **That is because the metric cannot resolve anything below ~110 µm**, on a tissue of radius
   ~400 µm. This is a much larger fact than the oblique question and it cuts against us; see §8-bis
   and `reports/metric_resolution.md`.

### F2 — the criterion, and the honest limits of it

> **F2.** θ\* is the largest angle clearing G1 and G2 whose evaluation set has **non-zero measure**
> in the plane — i.e. `fill > 0`. **θ = 90° is excluded**: `fill = t·cos 90° / s = 0` exactly, so
> the evaluation set is `N` lines with zero area, and no score computed on it is a score on a
> section. Every scored angle reports its fill in the same row as its score.

**Why this and not a fill floor.** I looked for a principled floor on `fill` and **there is not
one.** The metric supplies no coverage constant; `min_gt_cells` and `max_n` are counts. Any floor I
picked now — 0.25, 0.33, 0.40 — would be a number chosen after seeing the table, and it would move
θ\* between 60°, 45° and 30°. Inventing one and calling it derived is exactly what this campaign
keeps catching, so it is not done. `fill = 0` is the **only sharp line available**, and it is sharp
for a reason that is not about thresholds: a set of measure zero is not a section.

**What F2 therefore selects: θ\* = 60°, with `fill = 0.25`.** That is *not* the 30–45° previously
accepted, and the difference must not be smuggled in. The derivation excludes 90° and says nothing
about 60° versus 45°. **Choosing a headline below 60° is a judgement about how much coverage a claim
needs, and it is the author's to make explicitly**, not mine to launder through a threshold. Recorded
here so that whichever is chosen, the reason is on the record.

**What is scored regardless: every angle clearing G1 and G2 with `fill > 0`** — 30°, 45° and 60° —
each reported with its fill beside its score. The curve is the result; θ\* is a label on it.

**One nuance, because it qualifies the sharp line.** `fill = 0` at 90° is a property of the
*representation*: `load_volume` requires one depth per section, so cells are points at their
section's `z` and the teeth have zero width. Real cells have finite extent, so a physical 90° section
would have `fill ≈ cell diameter / s ≈ 0.17`. The degeneracy is in the data as built, not in the
tissue — and since the data as built is what anyone can score, the exclusion stands.

## 2-quater. AMENDMENT (2026-09-12) — F2's SECOND repair: measure in micrometres, not in `fill`

*Dated and recorded as a second repair of the same rule, not as a bugfix. F2 has now failed twice,
both times admitting the one angle it exists to exclude, and a rule with that history should carry
it visibly.*

**What happened.** §2-ter's F2 reads *"the evaluation set must have non-zero measure"*, and the
runner implemented it as `fill > 0.0`. But `fill = t·cos θ / s` and `np.cos(np.deg2rad(90))` is
**6.12 × 10⁻¹⁷**, not zero, so at 90° `fill = 3.05 × 10⁻¹⁷`, `has_measure` returned `True`, and θ\*
came back as **90°** — into the headline, into `scored_angles`, into everything. The criterion was
right and the comparison was against exact zero in binary arithmetic.

This is the third floating-point boundary defect in this work (the `−0.05` band, `fill(90°) = 0` in
a rendered table, and now this) and the only consequential one: the other two were cosmetic, and
this one silently restored a claim that had twice been ruled out.

**The repair: measure the stratum in micrometres against the volume's own resolution.**

> **F2 (second form).** Each section contributes a stratum of width `t·cos θ / sin θ` micrometres
> in the plane. The evaluation set has non-zero measure **in practice** when that width is at least
> the volume's median nearest-neighbour distance — `TrainingVolume.median_nn_dist`, which the loader
> already computes. A stratum narrower than the spacing between neighbouring cells contains no
> spatial extent the data could resolve; it is a line drawn through a point cloud.

On `merfish_thick_hypothalamus` (`t` = 28.6 µm, median NN ≈ 8 µm):

| θ | stratum width | verdict |
|---|---|---|
| 30° | 49.5 µm | passes |
| 45° | 28.6 µm | passes |
| 60° | 16.5 µm | passes |
| 70° | 10.4 µm | passes |
| **75°** | **7.7 µm** | **fails** |
| **85°** | **2.5 µm** | **fails** |
| **90°** | **1.8 × 10⁻¹⁵ µm** | **fails** |

**Why this is not gerrymandered toward an angle.** Every term is measured: `t` is the dataset's
stated slab thickness, `s` and the nearest-neighbour distance come from the volume, and the width is
trigonometry. Nothing is chosen. The test that it is not reverse-engineered to a preferred answer is
that **it excludes 85° as well as 90°** — a rule tuned to exclude only 90° would have admitted 85°,
which is 90° in all but name, and the first form of F2 did exactly that. A loophole I had flagged
and then left open is closed by the repair rather than by a second rule.

**What it selects: θ\* = 60° on the swept angles, or ~70° if that angle is swept.** Both are below
75°, and both are reported with fill and stratum width beside them.

**A second precondition defect, found in the same run and repaired here.** At 0° the donor slab came
back **empty**: it is offset along the normal by one slab *thickness* (28.6 µm), and with sections
57.5 µm apart the offset band contains no section at all. So **P1, the coronal control that gates
every other number in this report, could not run.** `flanking_copy`'s donor is the adjacent
*section*, so the correct offset is **one section spacing**, which also keeps the slabs disjoint at
oblique angles (57.5 − 28.6 = 28.9 µm of clear space between them). §3-bis's flanking-slab
construction is amended accordingly: **offset by `s`, not by `t`.**

Worth recording how it was caught: by *"and neither set is empty"*, the third and most trivial-looking
of the three leak assertions. L1 and L2 both pass vacuously on an empty donor set — it coincides with
nothing and is disjoint from everything. **A leak check without an emptiness check is a check that
passes hardest when there is nothing to check.**

## 3-bis. AMENDMENT (2026-09-11) — the exclusion rule in §3 makes the demonstration impossible

*Found while building the runner, before it ran. §3 steps 2–3 are superseded by this section; §3
step 1 and preconditions L1/L2 stand, restated below against the corrected construction.*

**What §3 said.** Exclude every *section* contributing to the ground truth `G`.

**Why it cannot work.** At 90° through a 10.3 : 1 block, `G` draws from **all 7 sections**. Excluding
them removes the entire volume:

```
90deg ground truth G: 373 cells from 7 sections
PRE-REGISTERED RULE (exclude whole sections contributing to G):
  donors = 0 cells NONE   <-- the whole volume is gone
```

This is prediction 2 of §7 arriving early and worse than predicted: not "at least one angle fails
P3", but *every* oblique angle fails by construction. **The fix is not to loosen the exclusion.**
Section-level exclusion is simply the wrong granularity off-axis — at a coronal plane one section
holds the whole evaluation set, and off-axis `G` is a band drawn from all of them.

**A first correction, also rejected.** Cell-level exclusion within a guard band keeps donors alive,
but the empty-slab fallback then returns **one cell**: a flat section meets an oblique plane in a
*line*, so no two cells share a perpendicular distance and widening to the minimum admits exactly
one. That degeneracy now **raises** rather than returning a one-cell donor set
(`cells_near_plane`), and its message names the construction below.

**The corrected construction: a FLANKING SLAB.** Donors come from a slab of the same orientation
with its origin offset along the normal by one slab thickness, read with the ordinary threshold
rule. Two of them, `±1`, exactly as `flanking_copy` has two flanking sections. Measured on a
7-slab / 27 µm / 1000 × 1000 µm block:

| θ | `G` | flank −1 | flank +1 | shared cells |
|---|---|---|---|---|
| 0° | 2000 from 1 section | 2000 from 1 | 2000 from 1 | **0** |
| 30° | 795 from 7 | 762 from 7 | 757 from 7 | **0** |
| 60° | 440 from 7 | 466 from 7 | 468 from 7 | **0** |
| 90° | 373 from 7 | 401 from 7 | 385 from 7 | **0** |

Three properties, all of which the §3 rule lacked:

1. **Leak-free by construction**, not by assertion — the slabs are disjoint in the perpendicular
   coordinate, so no cell can appear in both. L1/L2 remain as run-time assertions, now as checks on
   a construction that should pass them rather than as the mechanism itself.
2. **It reduces to `flanking_copy` at 0°** — same cells, same two donors. So the coronal control P1
   is a comparison against the real baseline, not against a differently-constructed one.
3. **The donor is one full slab from the plane at every angle**, which is the relationship a
   flanking section has to a held-out section coronally. An oblique score is therefore comparable
   with a coronal one rather than flattered by donors half a slab away.

**§3 steps 2–3, restated.** Every arm generates from the flanking slabs at `±1 × thickness`;
`exclude_within_um = half thickness` is set as a belt-and-braces guard so a leak would raise rather
than be detected afterwards; L1 and L2 are asserted on the returned arrays at run time regardless.

**What this changes in §7.** Prediction 2 is **superseded**: it named the right risk and the wrong
remedy. I said "if an angle fails P3 after the exclusion, that is a finding, not a reason to loosen
the exclusion" — and the finding turned out to be that the exclusion was mis-specified, which is a
third thing, found by building the runner rather than by running it. Predictions 1, 3 and 4 stand
unchanged and are **not** revised in the light of the table above.

## 2-bis. AMENDMENT (2026-09-11) — the claim is made at 30–45°, not at 90°, and why

*Supersedes §0's "clears 90°" and §2's "the reported angle is 90° whatever the scores are". Written
before the true-thickness budget runs and before any demonstration score exists.*

**Two independent reasons, and the second is the one that settles it.**

**R5 — the 90° clearance was measured on a slab 2.1× too thick.** `angle_budget.py` defaulted slab
thickness to the training volume's median spacing, 57.5 µm, where the specimen's slabs are ~27 µm;
`paper_2_4_6` removes every other section, so the training spacing is a multiple of the real pitch.
Fixed at source. The true-thickness budget does not exist yet and my estimate is marginal.

**The comb limit — 90° was never available, at any thickness.** `reports/the_comb_limit.md`: the
plane's second in-plane coordinate is `v = (y − y₀)cos θ − (z − z₀) sin θ`, which at 90° is
`−(z − z₀)` exactly. `z` takes one value per section, so a 90° "oblique section" from `N` serial
sections is **`N` parallel lines**. On a 4-section training volume it is four lines. This is
geometry, not a shortcoming of any method, and no specimen in the cross-dataset sweep escapes it.

**So the claim is made at the largest angle clearing G1 and G2 at the true thickness, with the fill
ratio `t·cos θ / s` stated beside it. On present evidence that is 30–45°.** This is fixed now,
before the number exists, and it is *weaker* than what §0 claimed — which is the direction that
makes it safe to fix in advance.

**What this changes in §2.** The angle table stands; "the reported angle is 90° whatever the scores
are" is struck. The rule that replaces it: **the reported angle is the largest that clears G1 and G2
at the true thickness** — determined by the budget, which is pre-registered, gate-driven and carries
no score. Picking the best-*scoring* angle remains forbidden. 90° is still measured and still
reported, with its fill of 0.00 printed beside it, because the comb limit is a result.

**What this changes in §4–§6: one new rule, and it restricts us rather than helping us.**

> **F1 — the fill rule.** Arms that reproduce real cells (`copy`, `resample-pd`, `null`) are combs
> with the same teeth as the ground truth and may be compared at any fill. An arm that generates a
> **continuous fill** (`field`) places cells between the teeth, where the ground truth has none by
> construction, and an optimal-transport metric charges it for the specimen's sampling rather than
> for its own error. **`field` is therefore excluded from the headline comparison and reported
> separately, with the fill printed beside it, at every angle.**

This forbids the one arm whose poor showing would have flattered the claim. It is recorded here for
that reason: `field` losing at low fill would not have been evidence, and we would have had it in
the table.

**§6's outcomes are unchanged in form**, with `90°` replaced throughout by **`θ*`**, the largest
angle clearing G1 and G2 at the true thickness. DEMONSTRATED still requires
`resample-pd(θ*) ≥ copy(θ*) − 0.05` with `|d(θ*)|` set against the across-seed spread.

## 4. The baseline: `flanking_copy` under the same constraints

**The baseline is `flanking_copy` restricted to the same slab under the same exclusion.** Not the
published tier-1 floor (a different dataset, §4.2a), and not this specimen's coronal `flanking_copy`
(a different and much easier geometry). It is re-measured at each angle, under L1 and L2, against
the same `G`.

Arms, all at each angle, all under the same exclusion:

| arm | positions | what it tests |
|---|---|---|
| `copy` | the nearest *permitted* section's, pasted on the plane | the baseline: what the previous best method can do off-axis |
| `resample-pd` | `plane-distance`, the cells the plane actually cuts, minus the excluded | the method's layout |
| `field` | the intensity field | reported as ablation A4, not as the claim |
| `null` | permuted types on `resample-pd`'s positions | the floor; the scale of the metric on this slab |

Scored through the pinned `evaluate_paper`, metric `paper_celltype_localization`, three generation
seeds, **reported per seed and per angle** — never a median of medians
(`test1b_estimator_discrepancy.md`).

## 5. Preconditions — any failing at an angle and that angle is NOT READABLE

- **L1, L2** as above.
- **P1 — the coronal control.** At 0°, `resample-pd` must equal `copy` **bitwise**. This is the
  fixture's load-bearing test repeated on real data; if it fails, `plane-distance` is not a strict
  generalisation on this specimen and nothing else in this report may be read.
- **P2 — the null is a floor.** `null ≤ 0.10` at every angle, or the metric is not responding to the
  type–position association on this slab and the scale is unknown.
- **P3 — the slab is scorable.** The budget's own G1 and G2, re-checked at run time on the actual
  `G` after the exclusion, because the exclusion changes the counts the budget measured.
- **P4 — pose.** Per **(angle, seed, arm)** `align_rotation_deg` recorded and compared per arm-pair
  actually being differenced, never as a median over anything (`layout_split_preregistration.md`
  §6-bis: a precondition on a compressed aggregate measures the compression).
- **P5 — the spread is reported beside the score.** Across-seed spread at each angle, printed in the
  same table. §4.2o: a criterion that ignores a spread the report has already computed.

## 6. Outcomes, fixed now

Let `d(θ) = resample-pd(θ) − copy(θ)`, both under the same exclusion, medians over three seeds, with
the across-seed spread `s(θ)` reported beside it.

- **DEMONSTRATED** — P1–P5 and L1–L2 hold at 90°, and `resample-pd(90°) ≥ copy(90°) − 0.05` with
  `|d(90°)|` set against `s(90°)`. The method generates a scorable section orthogonal to the
  sectioning plane, at no meaningful cost against the best available baseline. This is the paper's
  central claim.
- **DEMONSTRATED WITH A COST** — as above but `d(90°) < −0.05`. The capability is real and its price
  is stated as a number, in the abstract, not in a limitations paragraph.
- **PARTIAL** — 90° fails a precondition but a smaller angle (≥ 30°) passes everything. The claim is
  made at that angle, with the failing precondition at 90° printed beside it.
- **NOT DEMONSTRATED** — no angle ≥ 30° passes. Claim B is withdrawn; claim A stands alone as a
  shown-not-scored figure, and the budget table stands as the contribution.

## 7. My predictions, recorded now

1. **PARTIAL or DEMONSTRATED WITH A COST**, not DEMONSTRATED. At 90° the exclusion removes nearly
   every section, so `copy` is reduced to a distant donor and `resample-pd` to the cells of whatever
   survives the exclusion. Both degrade; which degrades less I do not know.
2. **The exclusion, not the angle, is what will bite.** §3's step 2 removes every section
   contributing to `G`, and at 90° a plane through a 10.3 : 1 volume touches most of them. I expect
   at least one angle to fail **P3** after the exclusion even though it passed the budget before it.
   If it does, that is a finding to report, not a reason to loosen the exclusion.
3. `field` stays below `resample-pd` at every angle, consistent with Test 1's refutation.
4. **P1 passes.** The fixture asserts it bitwise, and this specimen's sections are evenly spaced, so
   the tie-break that the rule now handles is exercised on the real data too.

## 8. What this cannot decide

One metric (`paper_celltype_localization`), one specimen, one holdout design. It says nothing about
expression fidelity — `test1b` established that this metric is nearly blind to the expression head
(`retractions.md` R2) — and nothing about whether a *second* specimen would replicate. The
replication question is `reports/oblique_replication_candidates.md`.
