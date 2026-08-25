# T05 — Layout head

Part of [PROGRESS.md](../PROGRESS.md).

### T05 — layout head: intensity field, Strauss sampler, Potts marks (2026-08-16)

`model/layout.py`, `losses/reconstruction.py` (the layout NLL), `infer/planes.py` (the minimal
`Plane` T05 needs; T07/T09 add the rest beside it), `tests/test_layout.py` (20 tests), 16 new
`Config` fields. `make check` green; the fast suite is **151 tests in 56 s**.

**All eight of the spec's acceptance tests, with numbers.**

| Test | Criterion | Measured |
|---|---|---|
| `test_poisson_nll_recovers_intensity` | Pearson r > 0.9 on a grid | **0.989** total, **0.950 / 0.921** per type — **measured with a reduced spatial basis** (`fourier_bands_xy = 2`, the scale this intensity varies on), *not* the default 8. At the default the Poisson MLE overfits the point pattern: r decays **0.97 → 0.28** as steps grow while the NLL keeps falling. T05 specifies no regulariser and this task does not invent one — **open item owed to T06** (SPEC_QUESTIONS B10) |
| `test_expected_count_matches` | mean N over 50 seeds within 5 % of `N_expected` | **0.50 %** (543.50 vs 540.79) |
| `test_hardcore_respected` | no pair closer than `r0` | min pair **7.900 µm** ≥ `r0` = **7.897 µm** |
| `test_pcf_matches_real` | max abs g(r) difference < 0.15 over **`[0, 3R]`** (range amended at T05 — see below) | **0.093** |
| `test_potts_improves_purity` | closer to real than before, not above it | **0.490 → 0.649**, tissue **0.688** |
| `test_rare_types_survive` | 2 % type keeps ≥ 50 % of its expected count | **59.1 %** |
| `test_layout_deterministic` | same seed → identical layout | bitwise, coords and marks |
| `test_all_three_modes_run` | valid `Layout`, plausible N | field / hybrid within 5σ of `N_expected`, resample = the reused section's count |

**Definition of done — cell-type localization, both comparisons, and a failure.**
`paper_celltype_localization` (transcribed from `bench3/evaluate_paper.py`; T10 vendors the pinned
copy), on **all three** held-out sections rather than the one this task reported first time:

| held-out | **self** (section vs itself) | **generated** (`field`) | **ideal** (independent draw from the fixture's *true* law) | **flanking** (nearest real section = `resample`) |
|---|---|---|---|---|
| s02, z = 100 | 0.8730 | 0.7933 | 0.7144 | 0.4797 |
| s04, z = 200 | 0.9353 | 0.5732 | 0.6298 | 0.4966 |
| s06, z = 300 | 0.9581 | 0.7719 | 0.8091 | 0.6193 |
| **mean** | **0.9221** | **0.7128** | **0.7178** | **0.5319** |

* **Generated vs the held-out section's own value: 0.776 — this FAILS the 10 % criterion**, passing
  on one section of three (0.909 / 0.613 / 0.806). Plainly: the generated layout is materially below
  the held-out section's own localization. The first report of this task quoted 1.654 from s02
  alone, which was the best of the three.
* **Generated vs the flanking real section: 1.35×** (1.654 / 1.154 / 1.246) — better than the
  real-data alternative on every section.
* **Generated vs an ideal intensity: 0.994.** The `ideal` arm draws positions and marks from the
  fixture's *own* generative composition — an independent draw from the process that produced the
  held-out section — and reaches only **0.779** of the self-score, statistically the same as the
  layout head's 0.776. The metric normalises a Sinkhorn divergence against a within-tissue null, so
  a different *realisation* of the same law is already ~22 % of the way from the section to that
  null. The gap to the ceiling is the metric penalising realisation noise rather than the sampler
  losing localization — and a 0.90-of-self criterion asks the layout head to beat the process that
  produced the data.

**Decided 2026-08-16: the criterion is `generated ≥ 0.90 × ideal`**, stated on the mean over
held-out sections — measured **0.994**. `specs/05` is amended, and the superseded reading stays in
the suite as a **strict xfail** holding its failing 0.776, so the shortfall against the real
section is not reworded away and a later task that closes it breaks the suite until the record is
updated. On real data there is no ideal draw, so the referent is the flanking baseline, reported by
T10 (SPEC_QUESTIONS B15).

**The 0.613 outlier is `synthetic_s04` — the exact centre of the nine-section stack** (index 4 of 9,
four sections from either end). **This is not open risk R3**, which predicts a deficit at the *ends*;
`alternating` never holds out an end section, so the boundary regime does not appear in this table
at all. The cause is the metric, and the `ideal` arm carries it too, so it is not the sampler:
`celltype_localization` scores a type as `1 − d_obs / d_null`, and `d_null` — the divergence to an
equally sized random draw from the whole section — collapses to **0.072–0.087** for type 0, which is
34 % of the cells and therefore already nearly tissue-wide, against **0.079–0.573** for the localised
minority types. The same realisation noise costs the abundant type four to eight times as many score
points: type 0 scores **0.332** at s04 (`d_obs` 0.058) against **0.841** at s06 (`d_obs` 0.011), and
weighted by 0.34 that is essentially the whole 0.18 spread. `evaluate_paper` guards this only at
`d_null < 1e-4`, three orders of magnitude below where it bites. Two consequences, both written into
the specs: the T05 criterion is a **LOSO mean** rather than per-section, and **T10 reports per-type
ceilings** (SPEC_QUESTIONS B15a).

**Generalised into `specs/10` §1 — the achievable ceiling, for every metric.** A metric's stated
range is not its achievable range: every target metric compares a *generated* section with a *real*
one, so a perfect model still scores below the top of the scale because a different realisation of
the same law is not the same point cloud. T10 must now measure a ceiling for all six target metrics
and the control metrics by drawing from the fixture's `GroundTruthField` directly, report every
method / ablation / baseline number **both raw and as a fraction of that ceiling**, report the
ceiling's own spread over `Config.ceiling_n_draws` draws, treat a method *above* the ceiling as a
finding to investigate, and report per-part ceilings where a metric averages over parts.

**The fitted parameters, for the methods section.** On the synthetic fixture's six training
sections: `r0` = **7.897 µm** at the **1st percentile** of pooled nearest-neighbour distances
(`Config.repulsion_r0_percentile = 1.0`; the 5th-percentile alternative gives **8.386 µm**, and
**which one was used has to be reported** — SPEC_QUESTIONS B6), `R` = **19.176 µm**, `gamma` =
**1.000**, in-plane density **1.502e-3 cells/µm²**, median nearest-neighbour distance **13.946 µm**.
`gamma = 1` is not a failure to fit: the fixture's own point process is a *pure* hard core with no
soft repulsion, so "no soft repulsion" is the right answer and the fit finds it. On real tissue it
will not be 1, and the 1-D search is what will say so.
Potts coupling `beta` = **0.278** on the fixture as it stands, and **0.144** when a 2 % cell type is
injected — the rare-type constraint binding is the difference (below).

**Negative controls, as assertions.**

* *Pure Poisson (ablation A4) vs the pair-correlation criterion.* Running the control is what
  exposed a defect in the criterion itself. Over the spec's original range `[r0, 3R]`, field mode
  scores 0.093 and **pure Poisson scores 0.070 — it passes**. It has to: a hard-core process differs
  from a Poisson one only *inside* the correlation hole, and the hole ends at about `r0`, because
  `r0` **is** a low percentile of the nearest-neighbour distances. The stated range began exactly
  where the signal stops. Over `[0, 3R]` — the same statistic, a superset of the range, therefore a
  strictly harder test — field mode still scores **0.093** and Poisson scores **0.994**. The test
  asserts all four numbers, so the blindness is pinned rather than described.
  **Accepted and amended (2026-08-16):** `specs/05` now states the range as `[0, 3R]` with the
  measurement as its justification, `specs/10` §4 carries the matching warning for **ablation A4**
  (reported over `[r0, 3R]` the ablation table would claim the repulsion buys nothing — a false
  null), and `specs/11_COVERAGE_MATRIX.md` notes both (SPEC_QUESTIONS B12).
* *Hard core.* With the interaction switched off the same intensity produces a closest pair at
  **0.205 µm** against `r0` = 7.897 µm, so `test_hardcore_respected` is not vacuous.
* *Over-smoothing.* At `beta = potts_beta_max` with 8 rounds the 2 % type retains **0.000** of its
  expected count, against **0.591** at the fitted coupling.
* *ICM (the update rule T05 names).* See below; retention **0.000**.

**Deviations from the spec, and why.**

1. **`potts_update = "gibbs"` is the default; ICM is kept as the negative control.** T05 §3 says
   ICM, and ICM takes the `argmax` — it seeks the *mode*, so its first sweep is essentially
   `argmax_c lambda_c` whatever `beta` is. Measured with a 2 % type injected, at the **smallest
   coupling the fit can choose** (0.02): ICM takes the rare type to **0.000** and purity to
   **0.785** against the tissue's **0.688**, i.e. it violates T05's own "Do NOT" and overshoots the
   fit's target before the coupling does anything, leaving no `beta` to fit. Gibbs samples the same
   conditional instead: at that coupling the rare type retains **1.004**, and purity becomes
   monotone and fittable in `beta`. `test_icm_erases_rare_types` asserts the ICM numbers, so the
   spec's variant stays visible and measured. (SPEC_QUESTIONS B11.)
2. **`fit_potts_beta` takes the intensity** (T05 writes `fit_potts_beta(vol)`), and enforces the
   rare-type floor **as a constraint on the fit**. `beta` closes the gap between a draw from
   `lambda_c` and the tissue, so it is not a property of the tissue alone; fitting it from a
   structureless i.i.d. draw asks the coupling to do all the organising work and over-estimates it.
   The floor (`potts_rare_retention = 0.5` in *every* section, for types below
   `potts_rare_prevalence = 0.05`, which is the benchmark's own `RARE_CELLTYPE_FRAC`) turns T05's
   "Do NOT" into something the code guarantees: purity matching alone would choose **0.278** on the
   fixture-with-rare-type, and the floor takes it to **0.144**.
3. **The `beta` grid is geometric, not linear.** `beta` enters an exponent; on a linear grid of the
   same size the first non-zero candidate (0.25) already sits past the rare-type floor, so the fit
   has to choose between "no smoothing at all" and "over-smoothed". Grid: 0, 0.020, 0.039, 0.075,
   0.144, 0.278, 0.537, 1.036, 2.0.
4. **Positions on the mid-plane, count from the slab volume.** T05 fixes the count's domain (the
   slab volume, explicitly not the area — verified: doubling the thickness doubles `N_expected`) and
   leaves the positions' domain open. They are sampled on the mid-plane, because the section reports
   in-plane coordinates, every benchmark metric is an in-plane kNN statistic, and `r0`/`R`/`gamma`
   are fitted to an in-plane `g(r)`.
5. **`fit_intensity_head` jitters cell depths and redraws the MC points every step.** Without the
   jitter the fit is degenerate: every cell is recorded at its section's nominal `z` while the
   integral runs over a continuous slab, so intensity concentrated in thin sheets at those depths
   scores arbitrarily well — measured, the NLL fell **three nats below its value at the true
   intensity** while the correlation with that truth stayed at **0.00**. Both changes are statements
   the data already makes, not regularisers. (SPEC_QUESTIONS B10.)
6. **`infer/planes.py` exists early**, holding only `Plane`, `plane_from_normal`, `section_plane`,
   `uniform_plane_points`, `uniform_slab_points`. T05 is package code and needs a plane type; T07/T09
   still own `intersect`, `random_plane_pair` and the curved surfaces, to be added beside these.
   (SPEC_QUESTIONS B13.)
7. **`sample_layout` takes keyword-only `repulsion` and `flanking`.** The spec's positional
   signature is unchanged. `Config.repulsion=True` with no fitted `RepulsionParams` **raises** —
   there is no default hard core, because a hand-set one is what T05 forbids — and `hybrid` /
   `resample` raise without the flanking sections.

**Two findings that are not deviations.**

* *The fixture's "~2 %" rare type is 6.3 %.* `tests/fixtures/synthetic.py` claims one, T05 needs one,
  and `type_bias = linspace(0.6, -2.4, 6)` does not produce one. The fixture is left alone (every
  earlier task's numbers were measured on it) and `test_rare_types_survive` injects a genuine 2 %
  type — a stripe varying 0.2 %–3.8 %, i.e. interspersed rather than a compact niche, which is the
  hard case. The 6.3 % is pinned by a test. (SPEC_QUESTIONS B14.)
* *The Poisson MLE of a flexible intensity overfits, and T05 specifies no regulariser.* With the
  default `fourier_bands_xy = 8` the recovered correlation decays from **0.97 at 300 steps to 0.28
  at 1200** while the NLL keeps falling. The acceptance test lowers the head's spatial basis to the
  scale the intensity varies on and says so; **T06's trainer owes an explicit answer** (early
  stopping, a smoothness penalty, or a basis tied to the fitted length-scale). Recorded as an open
  item in SPEC_QUESTIONS B10.

**Coverage matrix.** All six T05 rows are implemented: per-type intensity + Poisson NLL; `r0` at the
1st percentile with the 5th selectable and recorded; Strauss repulsion fitted to `g(r)` with A4 as
the ablation; Potts smoothing with `beta` fitted, not set; the `layout_mode` gate (field / hybrid /
resample, all three exercised); and the slab-volume integral. Nothing in the design docs is missing
from the matrix for this task.

**Both gates re-run after the change** (`pytest tests/ -m gate`): **no criterion changed verdict** — GATE 1 and GATE 2 pass
exactly as at T03/T04. T05 adds `Config` fields and three modules but touches no existing code path.

> Scope, restated 2026-08-25: `pytest tests/ -m gate` asserts **criteria**, not the values a gate report prints, so this is a verdict-level claim. A criterion whose threshold is `> 0` can have its value move a long way and stay green — `G2.1h-c` did, 40.28 -> 63.97, for an unrelated reason (SPEC_QUESTIONS C32). The verdicts here are correct and were re-measured 2026-08-24; "unchanged" in this log never means "no number moved". Audit: `progress/t04_field_and_retrieval.md`.

---

### T05 addendum — the grid-multinomial position sampler (2026-08-24)

`reports/r11_fix_options.md`'s **option D**, implemented. The rejection sampler is kept, selectable
at `Config.layout_sampler="rejection"`, and is no longer the default.

**Why.** `reports/r11_envelope.md` established that the rejection sampler's envelope — the maximum
of `sum_c lambda_c` over a `layout_n_mc` sample of the mid-plane, times `layout_envelope_slack` — is
a *sampled* maximum with a 140-853x spread across sections, and that the acceptance ratio
`lambda / envelope` is never clamped. Where the true intensity exceeds the sampled maximum the ratio
passes 1 and the point is accepted with certainty, so the realised draw is from
`min(lambda, envelope)`. That is a **biased** point pattern, not merely a starved one. Option A (an
analytic Lipschitz bound) was costed and rejected on measurement: `L = 74.69` gives a bound 7-82x
above the true supremum, which would convert the bias into universal starvation.

**What was built.**

| | |
|---|---|
| `mid_plane_grid(plane, cfg)` | the sampler's cell centres `(K, 2)` and cell size `(2,)`. Public, because the convergence check and the per-cell expected count are stated in terms of it |
| `_propose_points_grid` | evaluate `sum_c lambda_c` once per cell centre, draw cell indices with probability proportional to it, jitter uniformly inside the drawn cell |
| `Config.layout_sampler` | `"grid"` (new default) / `"rejection"`, in `LAYOUT_SAMPLERS` |
| `Config.layout_grid_cells` | 128 along the window's longer axis; the shorter axis gets the count that keeps a cell near-square |
| `tests/test_layout_sampler.py` | 16 tests, **no fit and no data** |

There is no envelope, no acceptance ratio and no proposal budget on the intensity. The only
approximation left is the grid's own resolution — a midpoint rule with an `O(h^2)` error and a
convergence check, rather than a sampled maximum. With `Config.repulsion=False` one multinomial draw
of `N` indices places all `N` points; with the repulsion on, candidates are drawn from the grid in
batches and thinned by the same sequential Strauss test as before, against the same
`layout_max_proposal_factor` budget and the same `ProposalBudgetWarning`. The interaction is the
only thing that budget can now be exhausted by, which is what its warning always claimed.

**The validation the sampler never had.** `sample_layout` takes an `intensity_fn`, not a model, so
the whole sampler can be validated against a closed-form intensity. `tests/test_layout_sampler.py`
hands it a sum of Gaussian bumps on a floor, whose integral over any axis-aligned rectangle is a
product of two `erf` differences — so every expected count asserted is exact arithmetic, not a
second Monte-Carlo estimate.

| Test | Criterion | Measured |
|---|---|---|
| `test_the_regime_is_the_one_that_broke_the_sampler` | max/mean in the hundreds | **235** |
| `test_grid_total_density_is_correct` | slab integral within 5 % of exact; `N` within 5σ; every point placed | **0.6 %**; within 5σ; `n_cells == n_proposals` |
| `test_grid_spatial_density_is_correct` | every reference bin inside `0.06 * expected + 4 sqrt(expected)` | worst bin **0.28** tolerances |
| `test_grid_per_type_mix_is_correct` | global mix within 0.01 of the exact per-type shares; per-bump composition within 0.03 | passes at 3 bumps |
| `test_grid_converges_with_resolution` | in tolerance at 32 / 64 / 128 / 256 cells | passes at all four |
| `test_grid_error_falls_as_h_squared` | halving `h` at least halves the quadrature error | passes; **< 1e-3** at 256 |

The reference grid is 12x12 and deliberately not a divisor of `layout_grid_cells`: a reference grid
aligned to the sampler's own would integrate over exactly the cells carrying a systematic
within-cell error and hide it.

**The negative control, because a test both samplers pass measures nothing.** Three of the file's
tests assert the retained rejection sampler is *wrong* on the same closed-form intensity.

* `test_the_envelope_is_a_sampled_maximum` — with no sampling at all, the envelope rebuilt at eight
  seeds spans **9.3x** and its smallest value is **below half** the true supremum. `r11`'s defect 3,
  reproduced with no fit and no data.
* `test_rejection_sampler_starves_on_the_bump_field` — acceptance is `mean / envelope`, so at a
  max/mean of 235 the budget of `20 N` proposals places **under 1 %** of what it draws and the
  layout comes back truncated.
* `test_rejection_sampler_fails_the_same_criterion` — on a needle intensity (`sigma = 3 µm`, whose
  peaks a default `layout_n_mc = 4096` sample misses entirely) the share of points landing on the
  bumps spans **most of the unit interval across three seeds** and misses the exact value by more
  than 0.3, while the grid sampler holds the same quantity to **0.02** on the same three seeds.

**T05's own acceptance tests, re-measured.** All still pass. The numbers move, and the recorded ones
were the biased sampler's; both are now in the test docstrings, and the rejection column is
reproducible today with `layout_sampler="rejection"`.

| | rejection (recorded) | grid (now) |
|---|---|---|
| generated / ideal, mean | 0.7128 / 0.7178 = **99.4 %** | 0.7075 / 0.7235 = **97.8 %** |
| generated / ideal, per section | 1.110 / 0.910 / 0.954 | 1.046 / 0.949 / 0.945 |
| generated / flanking, per section | 1.654 / 1.154 / 1.246 | 1.491 / 1.514 / 1.059 |
| generated / self, mean (the superseded criterion, strict xfail) | 0.776 | 0.769 |
| ideal / self, mean | 0.779 | 0.785 |

Both arms move because the `ideal` arm is drawn by the same sampler as the `generated` one, which is
what keeps the comparison fair. `resample` is untouched: it reuses a flanking section's coordinates
and never calls a position sampler.

**What this does not do.** A correct sampler removes a bias; it does not supply a better intensity
field. R11's finding stands until re-measured: on tier-1 STARmap, `field` scored 0.4252 on
`celltype_localization` against `resample`'s 0.7008 and a model-free floor of 0.7765. **The
`layout_mode` decision waits for the three modes to be re-measured on STARmap with this sampler**,
and that fit does not run in this container.

**Cost.** One intensity evaluation per grid cell per section — 16384 at the default, against the
4096 the envelope alone used to spend plus up to `20 N` proposals. On the T05 fixture the whole of
`sample_layout` goes 0.36 s (rejection) to 0.76 s (grid), because the fixture's intensity is a GRF
lookup; on a trained `IntensityHead` the grid is one batched MLP forward.

**Both gates re-run after the change.** GATE 1 **PASSES**, every criterion value identical to the
recorded run (error ratio 0.130, r 0.917, smallest step +0.028, unimodality 0.000); only the
timestamp and the machine-dependent throughput move. GATE 2 **PASSES**, all criteria identical to
6 significant figures — G2.1a **0.9547**, G2.1b edge-excluded **0.9795**, z-interpolation 1.097,
`w_z` +0.030 / +0.049 against +0.003, entropy unchanged.

One **diagnostic** moved, is not this change, and is **not yet attributed**: `G2.1h-c`, the
mutation check that querying retrieval in the model frame perturbs the neighbour sets, reads
**63.97** against the recorded **40.28**. It passes either way (threshold `> 0`), and at 2K = 64 the
new value says the two neighbour sets are fully disjoint — a stronger signal that the channel is
wired, not a weaker one. What is *not* established is why the recorded number is different, and the
first explanation offered for it was wrong.

**Withdrawn: it is not the host.** The 40.28 was generated on macOS / arm / Python 3.12 and this
container is Linux / x86 / Python 3.11, so a cross-platform float difference in a discrete
symmetric-difference count looked like the answer. The spec owner regenerated it on that same macOS
platform and got **63.9697**, identical to this container. The platform account is false and is
recorded here only so it is not offered again.

**Not in this repository's history either.** The measurement was isolated — reproduced without a
trained probe, with the imported module paths printed per arm so an arm that silently loaded the
wrong tree would be visible. Comparing this commit against its parent was uninformative rather than
reassuring: the **oldest commit in the repository** (`70076ad`) already gives **63.9697**. That
bounds the transition outside the 52-commit history rather than inside it, which is why bisecting
cannot locate it — `reports/gate2.md` carries a generation date of 2026-08-16 and the oldest commit
is dated 2026-08-20, so the state that produced 40.28 was never committed.

**What the diagnostic actually measures.** It is close to a pure function of how far the mutation
moves the query points, and it saturates almost immediately. Measured on rotations about z (the
mildest axis, since z is preserved and only the in-plane position moves):

| rotation | 0° | 0.5° | 1° | 2° | 5° | 10° | 20° | 45° |
|---|---|---|---|---|---|---|---|---|
| `G2.1h-c` | 0.00 | 9.88 | 19.28 | 35.68 | 57.89 | 62.43 | 63.58 | 63.93 |

The gate's own rotation, `random_rotation(_seed_for(seed, 50))`, is **179.68°**, so **63.97 is the
correct value for the rotation the gate draws**, and 40.28 corresponds to an effective displacement
of roughly 2°. The value is invariant to every knob that could plausibly reach it —
`retrieval_z_window` over 0.05-3.0, `retrieval_z_window_gap_factor`, `retrieval_candidates_per_section`,
`rotation_bias` and its tilt cap — and there are **no PAD slots** in either arm, because the
per-query gap term (C1c) makes the candidate pool non-empty by construction, which closes the other
route to a smaller symmetric difference (a truncated pool). So 40.28 is not reproducible from any
committed state under any reachable configuration.

**Attributed, 2026-08-25 — it is the pre-C1c retrieval window.** The candidate named here as
untestable turned out to be testable after all, by substituting the old window rule rather than the
old code: the pre-C1c bound is the single absolute term `retrieval_z_window x median_spacing`, with
no gap-relative widening. Substituting it and re-running the measurement returns **40.2783**,
bit-for-bit the recorded number, with **23.69 of 32 slots padded** in the mutation's broken arm.

The mechanism is C1c's own, seen from the other side. The broken arm queries deliberately off-stack
points, so under the pre-C1c window they fall outside `3 x median_spacing` of every section,
retrieve nothing, and come back mostly padded — and a truncated pool makes the symmetric difference
*smaller*. C1c's per-query gap term makes the pool non-empty by construction, so the same mutation
now returns two fully populated, disjoint sets at 63.97. The recorded 40.28 is therefore a
**pre-amendment** number that the report carried forward un-regenerated, which is consistent with
its 2026-08-16 generation date. C32 is closed with the reproduction, and
`progress/t04_field_and_retrieval.md` carries the audit of what C1c's "gate numbers unchanged"
claim was measured on.

A second observation the sweep makes unavoidable, recorded because it is what makes the 40 -> 64
move consequence-free: **`G2.1h-c` saturates at 2K for any real rotation**, so as a *graded*
diagnostic it carries almost no information. It is a binary "the channel is wired" check wearing a
continuous number, and its value should not be read as a measurement of anything.

---

## T05 addendum — `layout_mode` ships as `resample` (2026-08-25)

**The decision is T05's headline claim not surviving real data, and it is recorded as a negative
result rather than a defect.** `Config.layout_mode` now defaults to `"resample"`; `field` and
`hybrid` stay implemented, tested and selectable, and are reported as ablation A4 (`specs/10` §6).
`specs/05` §4a carries the statement, the numbers and the reason; R11 in `progress/risks.md` carries
the decision and its evidence; `reports/r11_starmap_layout_modes.md` is the measurement.

### What T05 got right, and what it did not

| claim | verdict |
|---|---|
| Strauss/hard-core thinning reproduces real nearest-neighbour spacing | **holds** — `g(r)` over `[0, 3R]` matches at 0.093 against pure Poisson's 0.994 |
| Potts marks reproduce neighbourhood type purity, rare types retained | **holds** — unchanged by this decision |
| `N = ∫ Σ_c λ_c` gives the cell number *emergently and correctly* | **fails on real tissue** — 63.9x / 5.4x / 0.895x ground truth across three sections of one volume, and **3.7x unstable between refits of one configuration** (48 343 then 179 495 cells for a section whose truth is 4 187) |
| the field layout beats copying a flanking section | **fails on real tissue** — 0.6607 against 0.7546 for `resample` and a 0.7765 model-free copy floor; holds on the fixture at 1.35x, for a reason that is about the fixture (below) |

The point *pattern* is not what failed. The *number of points* is, and the metrics see the number.

### Why the fixture said otherwise

Not a power problem, and not a wrong measurement — an unrepresentative referent. The fixture's
flanking sections score **0.5319** against a 0.9221 self-score (58 % of ceiling); real serial
sections at 50 µm score **0.7765** against a 0.9808 oracle (79 %). The fixture's law decorrelates
fast in z, so copying a neighbour is weak there and nearly anything beats it. `specs/05`'s
`test_localization_beats_the_real_data_baseline` still passes at 1.35x and is now annotated to say
what it does and does not mean. Recorded in `progress/fixture_limitations.md` §2.

### Two things measured while deciding it, both reusable

* **`layout_mode` does not affect the fitted weights.** Fitting the fixture at `field`, `hybrid` and
  `resample` with one seed gives **bitwise identical** state dicts across all 96 parameter and
  buffer tensors — `sample_layout` is never called in training, and `_layout_term` evaluates the
  intensity at the *real* cells' positions. So one fit serves all three arms, which is how both the
  STARmap re-measurement and the fixture gate re-run were done, and it means T09's merged gate
  refits 3x more than it needs to on this axis (`specs/09` §3).
* **The grid sampler's effect is small and does not change the ordering.** Same weights, sampler
  swapped: `field` +0.0599 (1.8x the envelope), `hybrid` +0.0158 (inside it). The bias
  `reports/r11_envelope.md` found was real and worth fixing; it was not what put the layout head
  behind a copy.

### What is still open, and it is T05's

The intensity integral's scale. It is wrong by 1.5x-64x depending on section, unstable 3.7x across
refits of one configuration, and coupled to `decoder_mu_link` through the shared triplane although
`IntensityHead` never sees `mu` (`reports/r11_coupling.md`). A generative layout cannot ship until
that is fixed, and fixing it is not a tuning exercise: a quantity that varies 3.7x between refits
carries no headline number at any score.
