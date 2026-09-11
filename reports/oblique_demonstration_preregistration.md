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
