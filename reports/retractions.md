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

---

## R8 — F2's first form admitted the angle it existed to exclude

**Claimed** (`oblique_demonstration_preregistration.md` §2-ter, and reported in conversation): F2
excludes 90° because its evaluation set has zero measure, so θ\* = 60°.

**Withdrawn as implemented.** F2 read `fill > 0.0`, and `np.cos(np.deg2rad(90))` is **6.12 × 10⁻¹⁷**,
so `fill = 3.05 × 10⁻¹⁷ > 0` is `True`. The run returned **θ\* = 90°**, into the headline and into
`scored_angles`. The criterion was correct in words and compared against exact zero in binary.

**Third floating-point boundary defect in this work** — the `−0.05` band, `fill(90°) = 0` in a
rendered table, and this — and the only consequential one: the other two were cosmetic, this one
silently restored a claim ruled out twice.

**Fixed at source, as F2's second form** (§2-quater): the stratum's width `t·cos θ / sin θ` is tested
in **micrometres** against `TrainingVolume.median_nn_dist`, which the loader already computes. Every
term is measured; none is chosen. It excludes 90° (1.8 × 10⁻¹⁵ µm), **85° (2.5 µm)** and 75°
(7.7 µm), and admits 30/45/60/70. Excluding 85° is the test that it is not gerrymandered: a rule
tuned to exclude only 90° admits 85°, which is 90° in all but name — and the first form did exactly
that, leaving open a loophole I had flagged and not closed.

**Class.** A criterion stated in exact arithmetic and implemented in floating point. The general
form: **a rule that turns on a quantity being exactly zero must be tested in units where zero is
physically meaningful.** `fill` is dimensionless and rounds; a stratum width in micrometres does not.

---

## R9 — the coronal control had no donors, and the blur figure understated the limit

**Two defects from the same run, both repaired in §2-quater and the metric-resolution document.**

**The donor slab was offset by one slab thickness (28.6 µm) where sections sit 57.5 µm apart**, so at
0° the offset band contained no section and the donor set was **empty**. P1 — the coronal control
gating every other number in the report — could not run. `flanking_copy`'s donor is the adjacent
*section*, so the offset is one **spacing**. Caught by *"and neither set is empty"*, the third and
most trivial-looking leak assertion: **L1 and L2 both pass vacuously on an empty set** — it coincides
with nothing and is disjoint from everything.

**And `metric_resolution.md` gave the blur as "≈ 110 µm", read off the oblique rows.** Every
published score in this literature is computed on a **full coronal section**, where the radius is
621 µm and the blur is **186 µm** — 70% larger. The figure understated the limit on exactly the
numbers the limit most applies to.

**Replaced by the dimensionless form**, which cannot be misquoted that way: `scale` is computed on
radius-normalised coordinates, so `blur / radius = √(eps·scale)` ≈ **0.26–0.30 is a constant of the
metric, not a property of any tissue**. The statistic distinguishes roughly **three to four locations
along a radius, on any dataset at any magnification**. Corrected in all five propagated documents and
in paper §3.2.

---

## R10 — the scoring path emitted 2-column coordinates

**Not a withdrawn claim — a crash, recorded here because the check that should have caught it was
written one round earlier and did not.**

`arm_prediction` passed the plane's `(u, v)` straight to `_v2_io.write_prediction_h5`, which reads
`r["coords"][:, 0]`, `[:, 1]` **and `[:, 2]`**:

```
IndexError: index 2 is out of bounds for axis 1 with size 2
```

**Why §4.2p's check missed it.** The previous round added a wiring check for the *scoring* path, and
it verified the **evaluator's** contract — that the ground truth is still subset by section label,
that `celltype_localization` is still emitted, that `NearPlaneCells` carries `counts`. It never
verified the **writer's**. A signature check could not have caught it either: `coords` is one key of
an untyped dict, so every call type-checks.

**What is checkable, and now is:** *which columns the writer indexes*. The check reads
`_v2_io.py` and extracts them by pattern — `{0, 1, 2}` — then asserts `arm_prediction` emits `(n, 3)`
and that its third column is zero. Verified by reintroduction: with the two-column version restored,
`--self-check` drops 72/72 → 70/72 and **reproduces the identical `IndexError` message without any
data**.

### And the question the crash raised, settled by evidence rather than judgement

*What should `z` be for an oblique slab — the cells' real depth, or a constant in the plane's frame?*

**Neither choice can affect a score, because the evaluator never reads the third column.**
`load_prediction` loads `obs/z` into `pred["z"]`; no metric in `evaluate_paper` or `align.py` touches
it, and the only indexing of a third column anywhere is `[:, :2]`, which excludes it.

**The real decision is which two columns go in `x` and `y`, and it is forced.** Every metric —
Moran's I, Geary's C, the marker field, `celltype_localization` — is computed from `gt_xy` and
`pred_xy`, both two-dimensional, and the `SPATIAL_K` kNN graph is built from them. So both sides
carry the **plane's own `(u, v)`**: a section's geometry is its in-plane geometry, and the
one-section ground truth uses the same frame. Writing the cells' real `(x, y)` would compare them in
the *volume's* frame — at 90° the plane's `v` is `−(z − z₀)`, so real-`y` would collapse to a band
one slab wide and the section's geometry would be destroyed.

`z = 0` is then the honest constant: a generated section lies *in* its plane, so its depth in that
plane's own frame is zero everywhere. It is a **format requirement of the writer, not a modelling
choice**, and the report says so where a reader will meet it.

**Class.** §4.2p again, one layer out: *a wiring check must cover every interface the path crosses,
not the one whose contract was most recently on your mind.* This path crosses two — the writer and
the evaluator — and the check covered the second.

---

## R11 — the baseline arm was copying the evaluation set

**Withdrawn: every score in the first oblique run.** `copy-nearest-z` emitted the **whole** of the
section nearest the plane's origin — the section the plane passes through the middle of. That
section's cells inside the slab **are** part of the ground truth, so roughly **475 of the 1 906 GT
cells at 30° (≈25%)** were present in the baseline verbatim, at identical coordinates.

**The baseline beat us at every angle**, by 0.11 to 0.27, and this is why the direction is
uninterpretable.

**How L1 missed it.** L1 is an exact-coordinate test and would have fired instantly — it was pointed
at the wrong object. `leak_checks(best.xyz, truth.xyz, …)` tested the **donor slab**, which
`resample-pd` uses and which does not leak. `copy-nearest-z` drew from `nearest`, never checked. All
fifteen green ticks in that run's leakage table were about an arm that does not leak.

**Fixed at source.** Both arms now draw **only from cells outside the evaluation slab**, symmetric
and leak-free by construction, and `arm_leak_checks` runs on **every arm's emitted coordinates**. The
self-check asserts it fires on a planted coincidence, on an empty arm, and that the label names which
arm failed.

**Class.** A check applied to the wrong operand. §4.2p says a wiring check must cover every interface
the path crosses; this is its sibling — **a correctness check must cover every object the claim
rests on**, and "we ran L1" is not the same as "we ran L1 on each arm".

---

## R12 — three seeds bought no interval

**Claimed** (the first run's verdict line): "…across-seed spread 0.0000", offered as the uncertainty
beside a difference of −0.1110.

**Withdrawn.** The spread is exactly zero because **both compared arms are deterministic** — they
reproduce real cells and nothing in them varies with a generation seed. Only the permuted null
varies. So the seeds were spent entirely on the one arm not in the comparison, and **the difference
had no uncertainty attached to it at all** while appearing to have one.

**Fixed at source.** An interval from **resampling the prediction's cells**, `R = 40` replicates at
the 16th–84th percentile, fixed in the pre-registration before it ran. A difference whose interval
spans zero is reported as *not distinguishable*, whatever its point estimate.

**Class.** §4.2o — a criterion quoting a spread that cannot vary. Worse than ignoring a spread: it
displayed one and it was structurally zero.

---

## R13 — the verdict outranked the preconditions it printed

**Withdrawn:** "DEMONSTRATED WITH A COST" in the first scored run.

Three lines below that verdict the same report printed **P2 ❌ FAILED**, and §5 of the
pre-registration says *"any failing and that angle is NOT READABLE."* `verdict()` computed the
outcome from the scores alone and never consulted the preconditions. And **P4 was pre-registered and
never implemented at all** — the poses were recorded in the JSON and never tested. Every angle failed
it, once by **166.5°**.

**Replayed through the corrected logic, the first run's own numbers give NOT READABLE at every
angle**: P2 fails at 30° and 60°, P4 fails at 30°, 45° and 60°.

**About that 166.5°.** An oblique strip is a ribbon — 440 × 1612 µm at 30° — and a ribbon maps onto
itself under a half-turn, so `align_by_expression`'s rotation search has two near-equivalent optima
and picks between them arbitrarily. It is compounded by the two arms having different *footprints*: a
near-full section face against the plane's own ribbon. **`align_by_expression` is unstable on
elongated point clouds**, which is what every oblique evaluation set is, and that is a third bound on
oblique evaluation alongside the comb limit and the metric's resolution.

**Fixed at source.** An angle with any failing precondition is NOT READABLE and no score can
overwrite it; P4 is computed between the two arms actually being differenced; and the report prints
every precondition as a row.

---

## R14 — the bootstrap shifted the estimate it was bracketing

**Withdrawn:** every interval in the second scored run. At 45° the bootstrap median was **0.2349**
against a point estimate of **0.1167** — a shift of 0.118, larger than two of the three differences
the study measures. Widths ran from 0.021 to 0.407 across three angles of one arm.

**Cause.** Resampling cells with replacement creates **duplicate coordinates**, and both the Sinkhorn
transport and the metric's own `max_n = 250` per-type subsampling behave differently on a cloud with
ties. **A resampling scheme that moves the estimate is not measuring uncertainty about it.**

**Replaced by a leave-one-cell-type-out jackknife**, chosen for a property the bootstrap lacked
rather than by retuning until the shift vanished — which would have been choosing a method by its
effect on the answer. A jackknife estimates the *variance* of a statistic and never replaces it, so
**the point estimate is the full-sample score by construction** and this defect cannot recur. It also
matches the statistic's own unit: `celltype_localization` is a frequency-weighted mean over cell
types. If fewer than three types survive, **no interval is reported** and the report says why.

---

## R15 — one angle's noise floor was applied to all three

**Withdrawn:** the single `null_ceiling = 0.1688` in the second run. `read_self_null` took the
**first** calibrated row — 30° — and looked up the entry nearest **θ\*'s** cell count, 1011. It mixed
**30°'s noise floor with 60°'s `n`**, and applied the result to every angle. Each angle's own
calibration sat unused in the same JSON.

Per-angle, as it should have been: 30° (n = 1906) ceiling 0.2843, passes; **45° (n = 1311) self-null
0.0333 — the *clean* branch, so the original 0.10 stands** and the run never reported that; 60°
(n = 1011) ceiling 0.1878, fails at 0.2461. Same verdicts, correct reasoning, and one branch that
went unseen.

**Class.** §4.2a's scope mix — committed inside the machinery built to settle a scope question.

---

## R16 — the calibration pass scored the arms, and then said it had not

`oblique_null_calibration.md` is **byte-identical to `oblique_demo.md`**, score table included,
while its JSON records `"scoring": "NOT RUN — geometry and preconditions only"`.

`score_arms` is entered on `args.score or args.calibrate_null` and then scores the arms
unconditionally. So the pass whose entire purpose was to settle P2 **before any arm was scored**
scored every arm — and recorded a **false provenance field** saying otherwise. The ordering §5-ter
exists to guarantee was never enforced.

**Fixed:** `--calibrate-null` without `--score` now calibrates and returns. The calibration's own
`n`-sweep, which the first pass computed and rendered nowhere, is now a table in the report.

---

## R17 — the small-`n` hypothesis is refuted

**Withdrawn:** my explanation for P2's failure — that `G2` constrains only the largest type, so a
Sinkhorn ratio on 30-point clouds is noisy and the null was measuring that.

**It does not fall with `n`:**

| n | 30° | 45° | 60° |
|---|---|---|---|
| 250 | 0.0980 | 0.0532 | 0.0849 |
| 500 | **0.2360** | 0.0822 | 0.0338 |
| 1000 | 0.1070 | 0.0897 | 0.0891 |
| full | 0.1714 | **0.0333** | 0.1305 |

No trend at any angle; at 45° the largest sample gives the lowest value. The floor is as present at
n = 1906 as at n = 250, so cell count is not the mechanism and the hypothesis is withdrawn.

**And the test found something larger than what it was testing.** A section whose cell types have
been **completely scrambled** — the ground truth against itself with its labels randomised, no
method, no donor, no arm — scores **0.03 to 0.24**. That is a property of `celltype_localization`,
recorded in `reports/metric_resolution.md` beside the blur.

The spreads are comparable to the values (0.1129 against 0.1714; 0.1142 against 0.0338), so **three
seeds cannot place this floor precisely** and every ceiling derived from it inherits that.

## R18 — R5 came back: §4's sweep predates its own fix

**Withdrawn:** the sentence in §4.7 warning that `starmap_visual_cortex`'s row alone was measured
before two corrections, and the figure of *159 cells against 250* it cited — which appears in no
committed report. The reference-plane correction (R1) is applied in **every** row of
`reports/angle_budget.md`; singling out one dataset was wrong, and the count was invented.

**And the thing it should have warned about instead is R5, reappearing.** R5 withdrew
"`merfish_thick_hypothalamus` clears 90°" because the slab was measured **2.1× too thick** — the
runner defaulted `t` to the *training* volume's section spacing, 57.5 µm, where the specimen's slabs
are ~28.6 µm. R5 says it was *"fixed at source"*. The cross-dataset sweep in
`reports/angle_budget.md` nonetheless reports **90°** again, on `t` = **57.5 µm**.

**The committed artifacts say why, and the test is one field.** The fixed runner writes a
`thickness_source` into every record and prints a *"thickness came from …"* line into the report.
`reports/angle_budget.json` carries `thickness_source: null` on **all four** datasets, and
`reports/angle_budget.md` carries no such line. **The sweep was run by the pre-R5 runner.** The fix
is in `scripts/angle_budget.py`; it was never applied to the numbers §4 was written from.

**And re-running it does not change the number — there is a post-fix run in `reports/` that proves
it.** `reports/angle_budget_true.json` carries `thickness_source`, so it was written by the fixed
runner. It reports `t` = **57.5 µm** and a budget of **90°**, identically, because the fix takes
`Section.thickness` only where the loader marked it *measured* and this build marks it assumed on
every section. What the fix added was the runner saying so, in the record, in its own words:

> *"the volume's median section spacing — `Section.thickness` is assumed on every section, so the
> file carries no measured slab thickness. **On a leakage-guarded input this OVERSTATES the slab**:
> held-out sections are removed, so the spacing between the ones that remain is a multiple of the
> real pitch."*

**The artifact announced its own defect and was read as a budget anyway.** The number moves only if
someone passes `--thickness 28.6`. A fix that makes a wrong default *visible* is not a fix that makes
it *right*, and R5's "fixed at source" overstated what had been done — the runner was corrected, the
measurement was not.

**What the sweep's `t = s` default does to each row.** A slab cannot be thicker than its own spacing,
so `t = s` is the **most generous** geometry available: it makes the strip `(D + t)/sin θ` as wide as
it can be, and every cell count, scorable-type count and `fill` in §4.4 an **upper bound**.

| row | direction | verdict |
|---|---|---|
| the three 5° budgets | conservative — they fail G1/G2 on the most generous geometry | **stand**; a truer `t` only makes them fail harder |
| the 90° budget | optimistic | **withdrawn again** |

The same sweep at `t` = 13.5 µm (`reports/angle_budget_thin.json`) gives that specimen **30°**,
failing G1 at 45° with 5 scorable types of 9. At the protocol's 28.6 µm the answer lies between 30°
and 90°, and §5's own run — which used 28.6 µm and added the stratum-width gate — measures **60°**.
**The paper quotes 60°** and reports 90° only as what the default gives.

**What is unaffected.** §4's argument does not rest on 90°: it rests on the gap between a stack of
thin sections and a block cut into slabs, and 60° against 5° is the same gap. The three 5° budgets,
the bimodality, the pre-build screen and §4.6.1's `fill` halving are unchanged — §4.6.1 uses the
protocol thickness (28.6 / 57.5 = 0.497) and never used the default.

**The check that was missing, and now exists.** `angle_budget.py --audit` reads the committed
`angle_budget*.json` and names every record written before a fix that landed in this file, by looking
for a field only the fixed runner writes. Pointed at `reports/` it returns:

```
  STALE reports/angle_budget.json          (all four datasets)
  STALE reports/angle_budget_thin.json     (merfish_thick_hypothalamus)
  ok    reports/angle_budget_true.json
```

Its four self-checks assert the auditor itself — that a record with no field is flagged, that an
explicit `null` is flagged, that the *fallback's own explanation* counts as post-fix, and that this
runner's own records audit clean. The existing check on this subject asserted that the **renderer**
prints the provenance line, on a record the check itself constructed with the field already set.
That is `specs/10` §4.2p once more: **the renderer was right and the committed sweep was old.**

**How it was found.** Drawing figure F3 required reading each dataset's thickness out of the JSON
rather than off the table, at which point `thickness_source` was `null` on all four rows while the
demonstration's JSON had been carrying its provenance for three rounds. **A retraction is not closed
by a patch to the script; it is closed by a re-run whose output shows the patch ran.** No artifact in
`reports/` distinguished the two until a field that only the fixed runner writes was looked for.
