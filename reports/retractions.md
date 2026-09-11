# Retractions

Claims this campaign made and has since withdrawn, each with what disproved it. Recorded as
retractions rather than quiet corrections: a claim that reached a report, a review or the advisor
has to be visibly withdrawn in the same place it was made.

---

## R1 — the 0° reference row was a double-thickness plane

**Claimed** (`reports/angle_budget.md`, first version): tier-1's coronal reference plane contains
**8279 cells** and **19 scorable types**, and G1's threshold is 60% of that 19.

**Withdrawn.** The plane's origin was `0.5 * (lo + hi)`, the volume's z-**midpoint** — 52.0 µm on
tier-1, exactly midway between the sections at z = 41 and z = 63. With a half-thickness of 11.0 µm
the band is precisely [41.0, 63.0], and `<=` admitted **both** sections on an exact floating-point
tie. The reference row was two sections where every other row is a slab through one.

**What disproved it**, independently and arithmetically:

- `test1b`'s own `n_cells` records ~**4165** cells per real tier-1 section. 8279 ≈ 2 × 4140.
- Strip area scales correctly only after halving the row: area(5°)/area(0°) = (1008×1412)/(1300×1514)
  = **0.72**, while cells = 3248/8279 = **0.39** — off by exactly 2×. Halved: 3248/4140 = **0.78**.
- Every tilted row draws from 4 sections; only the 0° row reports 2. Once tilted, the tie is gone.

**Class.** `specs/10` §4.2n — a statistic whose reference was inherited from where the code happened
to put the plane rather than chosen. Aggravated: the number sat on an exact float tie, one
recompilation away from being 0 or 8279.

**What survives.** The **budget of 5° is unchanged.** G2 is absolute (largest type ≥ 250) and a
single real section gives ~546, still clearing; re-centred, G1's denominator falls to ~18 and 10°
still fails on G2 at 159 < 250, 45° still fails G1 at 9 types. The defect is in the reported
reference, not the verdict.

**Fixed at source.** The reference plane is now centred on the volume's median **real section**, and
`reports/angle_budget.md` says so in the table's own caption.

---

## R2 — "the layout's positions are unstable across sections"

**Claimed** (review of `test1b_layout_split.md`, last round): the ~25× ratio between model-position
and copy-position arms' across-section spread shows the model's *placement* is unstable.

**Withdrawn.** It was a post-hoc reading of a table already seen, it was flagged as such, and the
three-seed run disproves it.

**What disproved it.** `both_oracle`'s across-seed spread is **0.0 on all three sections** — exactly
zero, not small. `celltype_localization` touches generated expression only through
`align_by_expression`; with positions fixed, types fixed and pose 0°, nothing stochastic remains.
So the metric is nearly **blind to the expression head**, and the across-section spread it reports
cannot be attributed to placement instability by that argument. `null_types` (model positions,
permuted types) has an across-section spread of 0.054 — but it sits at 0.03–0.08 and a metric
bounded below cannot spread at its floor, so it is uninterpretable in either direction.

**What replaces it**, on a clean single-variable contrast both of whose arms are far from any floor:

| | positions | types | across-section spread |
|---|---|---|---|
| `fix_types` | model | oracle | **0.4306** |
| `both_oracle` | copy | oracle | **0.0140** |

Types held identical, one variable changed, **31×**. Worst across-seed spreads are 0.095 and 0.000,
so the variation is across **sections**, not across seeds. The statement the paper can make is
**"the model's layout is wildly section-dependent"**, not "its placement is unstable" — and nothing
in `test1b` may be read as a statement about expression at all.

---

## R3 — `both_oracle` beating the copy floor

**Claimed** (implicit in reading `test1b`'s 125.5% recovered-both, and in the temptation to quote
0.8375 > 0.7765): giving the model the copy's positions and true types produces a section that
**beats `flanking_copy`**.

**Withdrawn.** `both_oracle` is donor positions carrying **ground-truth** cell types NN-transferred
onto them. `celltype_localization` scores, per type, the Sinkhorn divergence between predicted and
GT point clouds — so GT types on donor positions is a **partial oracle on precisely the quantity
being scored**, while the copy carries the *donor's* types. 0.8375 > 0.7765 is what that oracle buys
and is not evidence about the method.

**And a coincidence not to be seduced by.** 0.8375 sits next to the chain diagnostic's A1b ceiling
of 0.8369. **These are different metrics.** The ladder is `paper_morans_pearson` (its `oracle` is
1.0000); this table is `celltype_localization` (its `oracle` is 0.9808). Reading the two as
corroborating each other is the cross-scope comparison this campaign has been caught by four times
and will not make a fifth.

---

## R4 — "an exact tie is not a case the old rule handled meaningfully"

**Claimed** (`cells_near_plane`'s docstring, and `progress/t09_inference_and_calibration.md`, one
commit ago): the `plane-distance` rule's one divergence from `nearest-z` is that a plane sitting
*exactly midway* between two sections returns both, where `nearest-z` broke the tie on `section_id`
and took one — and this was "documented rather than hidden" as a curiosity about exact ties.

**Withdrawn.** It is not a curiosity. **With evenly spaced sections it is every target plane there
is.** `tests/test_layout.py::target_plane` says so in its own docstring — *"A plane halfway between
two training sections: what generation actually asks for"* — and on tier-1 the two flanking sections
sit at exactly ±22 µm from every held-out plane. I read the central case as an edge case and wrote a
paragraph excusing it.

**What disproved it.** The load-bearing test, on the second try:

```
test_plane_distance_is_BITWISE_identical_to_nearest_z_at_a_coronal_plane
  AssertionError: positions must be bitwise identical
test_the_empty_slab_means_two_different_things_and_the_flag_says_which
  expanding reached {'synthetic_s00', 'synthetic_s01'}, expected {'synthetic_s00'}
```

Two failures, one cause. Until they passed, `plane-distance` was not a strict generalisation and
every tier-1 number measured under `nearest-z` would have become cross-construction.

**Fixed at source.** The empty-slab fallback now selects a **section**, not a distance stratum, and
breaks its tie exactly as `nearest-z` does — on `(perpendicular distance, section_id)`. Verified
against `nearest-z`'s key transcribed verbatim, on all **8** midway planes of the fixture's stack,
plus the on-section and post-exclusion cases.

**And one assertion of mine was simply false**, on the same wrong premise: the exclusion test
asserted the band widens to cells *strictly further* from the plane (`min > max`). At a midway plane
the section the exclusion falls through to is at **the same** distance — 50.0 against 50.0 on the
fixture. Corrected to `>=`, with the equality asserted positively instead: the result must be
exactly the section `nearest-z` would pick under the same exclusion.

**Class.** Not §4.2n or §4.2o. This one is: *a divergence I documented instead of measuring.* Writing
the exception down is not the same as checking how often it fires, and the paragraph excusing it was
doing the work a test should have done.

---

## R5 — "`merfish_thick_hypothalamus` clears 90°"

**Claimed** (`reports/angle_budget.md`, cross-dataset sweep; carried into
`oblique_demonstration_preregistration.md` §0 and `oblique_replication_candidates.md`): the
specimen clears **90°** with **1988 cells**, both gates by wide margins, so the oblique
demonstration is scored at 90°.

**Withdrawn.** It was measured on a slab **2.1× too thick.** `angle_budget.py` defaulted the slab
thickness to the volume's **median section spacing** — the right concept, *"what a real section
represents"*, and the wrong quantity on a leakage-guarded input. `paper_2_4_6` removes sections 2, 4
and 6, so the *training* volume's spacing is **57.5 µm** where the specimen's slabs are **~27 µm**
(`specs/10` §8: a 200 µm block cut into 7 slabs). The run measured a 57.5 µm slab.

**What disproved it.** The cell count at 90° is proportional to slab thickness, so the two runs pin
the thickness the first one used:

```
486 / 1988  = 0.2445     (thin run / original run, measured)
13.5 / 57.5 = 0.2348     (the ratio those thicknesses imply)
```

4% apart — density variation. The original run's `t` was 57.5 µm.

**Fixed at source.** The runner now takes `Section.thickness` where the loader recorded it as
**measured** (`thickness_is_assumed=False`), falls back to spacing only when the file carries none,
and **prints which it used and why** in the report's own table. The fallback's text now says
explicitly that on a leakage-guarded input the spacing overstates the slab.

**What replaces it.** The budget at the true thickness, which does not exist yet. My estimate is
~933 cells and ~5 scorable types against a threshold of 5.4 — **marginal**, and I am not calling it
in advance. The claim is expected to land at **30–45°**, and `reports/the_comb_limit.md` shows 90°
was never available on a 4-section volume for reasons that have nothing to do with thickness.

**Class.** `specs/10` §4.2n, and the second instance in this campaign after R1 — a default inherited
from a general rule that does not hold on this particular input. Both were in the same runner and
both were in the *reference* the gates are read against, not in the gates.

---

## R6 — my own `f(90°)` screen arithmetic

**Claimed** (`oblique_replication_candidates.md` §2): a strip retains ~2% of a volume at 90°, from
`starmap` 304/16 600 = 1.8% and `merfish_thick_hypothalamus` 1988/**79 000** = 2.5%.

**Withdrawn.** The 79 000 is the **full dataset**; 1988 was measured on the **training** volume, which
holds 47 189. The second figure is `1988/47 189` = **4.2%**, and the screen's threshold moves from
~12 000 cells in the largest type to ~**8 300**.

**The conclusion is unchanged.** `exseq_visual_cortex` (1 130 cells in its entire volume) and
`exseq_breast_cancer` (1 979) are still ruled out by an order of magnitude. But the arithmetic was
wrong and a cross-scope mix of a full-dataset count with a training-volume count is exactly the
error `specs/10` §4.2a exists for — committed in the document that was ruling other things out for
being unmeasured.

---

## R7 — the first residual-modulation table was floating-point noise

**Claimed** (`oblique_demonstration_preregistration.md` §2-ter, first version, and reported in
conversation): the comb's residual modulation under the metric's kernel is 0.01% at 30°, 0.00% at
45°, 0.03% at 60°.

**Withdrawn.** Those came from an FFT convolution on a discrete grid. At 60° that implementation
returned **1.8 × 10⁻⁵, 9.4 × 10⁻⁴, 2.0 × 10⁻⁴ and 7.7 × 10⁻⁵** as the grid went 10⁵ → 8 × 10⁵
points. The true value is **3 × 10⁻¹²** — far below what a double-precision convolution can
represent, so the grid was reporting its own rounding.

**A quantity that moves two orders of magnitude with an implementation parameter is not a
measurement.** It was one edit from being published as one, and the only reason it was not is that
the self-check asserted a *margin* ("four orders of magnitude") rather than a value, and the margin
failed.

**Fixed at source.** The closed form is exact and stable: a square comb of duty cycle `f` has
first-harmonic amplitude `sin(πf)/π` against a mean of `f`, and a Gaussian of width `σ` multiplies
it by `exp(−2π²σ²/p²)`, so

```
modulation = (2·|sin(π f)| / (π f)) · exp(−2 π² σ² / p²)
```

Correct values: **1.7 × 10⁻⁴ (30°), 2.3 × 10⁻⁸ (45°), 3.1 × 10⁻¹² (60°)**.

**And the corrected numbers change the reading.** The modulation *falls* with angle, because the
strata crowd together as `period = s / sin θ` shrinks. The limit as `f → 0` is
`2·exp(−2π²σ²/p²)`, also negligible. **So the metric cannot see the comb at any angle, 90° included**
— which means this quantity discriminates nothing and cannot be the criterion. F2 rests on measure
zero instead, and it is a better criterion for having been forced off the wrong one.

**A second defect from the same function**, caught in a dry run of the renderer: at a coronal plane
`period` is infinite, `exp(0) = 1`, and it reported **128% modulation for a comb that does not
exist**. Now returns 0 when the period is not finite.

**Class.** New. §4.2 has rules about criteria, references and estimators; this is *a number whose
value depends on an implementation knob nobody varied*. The general form: **vary the knob before
reporting the number.** A convergence check is cheap and this one took four lines.
