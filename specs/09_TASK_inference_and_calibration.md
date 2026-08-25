# T09 — Inference, leakage-free calibration, and automatic configuration

**Goal.** Generate a section at any plane/thickness; calibrate the free statistical parameters using
**training sections only**; and choose the per-dataset configuration by internal LOSO so the user
never tunes a flag. The last point is both a usability claim in the paper and the guarantee that the
method cannot regress below the previous version.

**Files:** `spatialcpav25_gen/infer/generate.py`, `spatialcpav25_gen/infer/calibrate.py`, `spatialcpav25_gen/train/select.py`,
`tests/test_generate.py`, `tests/test_calibrate.py`

**Dependencies:** T01–T08.

---

## 1. Generation — `spatialcpav25_gen/infer/generate.py`

> ⚠️ **Open risk R3, raised at T04: reconstruction is far worse near the stack's ends.** Measured on
> the gate fixture, per-section R² of the T04 probe was **0.2912** at the first section and
> **0.3642** at the last, against an interior mean of **0.4474** — a 20–35 % deficit at the boundary.
> The mechanism is one-sided evidence: a cell at z = 0 has training sections and retrieval donors
> above it only. This is **real-volume geometry, not a fixture artefact** — every serial-section
> dataset has two ends — and it lands squarely here, because `generate_section` is routinely asked
> for planes at or beyond the outermost sections, where the model is extrapolating rather than
> interpolating. Treat generation outside the interior as a distinct, worse regime: report it
> separately, and consider surfacing an explicit boundary flag on the emitted AnnData (`uns`) rather
> than letting a caller assume uniform quality across the stack. The uncertainty gate (§below) is the
> natural place for it — the latent variance it already estimates should be elevated there, and if it
> is *not*, that is itself a finding.

```python
def generate_section(model, plane: Plane, vol: TrainingVolume, cfg: Config, seed: int) -> AnnData
```

```
1. Query the field on the plane's slab.
2. Layout head -> N, positions, types                      (T05)
3. h0 = GRF(positions)   -- continuous 3D field            (T03)
4. Retrieval: K real neighbours, excluding any section at the target z
5. Flow ODE 0->1 -> h                                      (T06)
6. ZINB decode + SAMPLE counts                             (T06)
7. Uncertainty-gated anchoring (below)
8. Emit AnnData: X = counts (raw), obsm['spatial'] = (x, y) in the plane's basis,
   obsm['spatial_3d'] = (x,y,z), obs[cell_type, region], uns[plane, seed, config hash]
```

**Uncertainty-gated anchoring.** Run `M = 8` flow samples with different GRF seeds; per-cell latent
variance `v_i` estimates uncertainty. Where `v_i` is low, blend toward a retrieval-anchored real
profile (Bernoulli per-gene mixing, preserving count-ness); where high, trust the generative sample.
The blend weight is a **learned** function `w(v_i)` fitted on LOSO reconstructions — a 1-D isotonic
regression from `v` to the mixing weight that minimised reconstruction error. This replaces the
hand-tuned gap heuristics of the previous version entirely; there is no `gap_scale`, no `alpha_tol`,
no `edit_weight`.

Also provide:
```python
def generate_stack(model, z_values, vol, cfg, seed) -> AnnData     # dense virtual stack
def generate_oblique(model, normal, n_sections, vol, cfg, seed) -> AnnData
def generate_curved(model, surface: Surface, vol, cfg, seed) -> AnnData
```
All share one GRF realisation per call, so the produced sections are **mutually coherent** — slices
of one object. Expose `grf_seed` explicitly and document that reusing it across calls is what
guarantees coherence.

### `retrieval_z_window` must scale with the gap, not be fixed (measured at T06)

`Config.retrieval_z_window = 3.0` is in units of the median section spacing, which is the right
*unit* and the wrong *constant* once the holdout is wide. Measured at T06 on the `consecutive-3`
holdout: the first training section is 200 µm from the next admissible one, so after the own-section
exclusion (`retrieval_exclude_source_section`, load-bearing for GATE 2) its cells have **no admissible
donor at all** inside 3 × 50 µm, and `EmptyCandidatePoolWarning` fires on **100–110 of every 512
cells**. Their neighbour sets are fully masked and the attention returns its bias — i.e. the retrieval
branch is silently absent for a fifth of the batch, in exactly the wide-gap regime the branch exists
for.

The warning did its job; the constant is the defect. T09 must make the window a function of the
generation or holdout geometry rather than a fixed multiple:

* the window has to admit at least the nearest admissible section *after* every exclusion, so its
  floor is `(gap to the nearest admissible section) / median_spacing`, rounded up, not 3;
* it is a calibration-time quantity like `ell`, and it belongs beside the length-scale calibrator in
  §2 — leakage-free by the same construction, fitted on flanking training sections only;
* `EmptyCandidatePoolWarning` firing on more than a negligible fraction of queries must be a
  **failure** of the generation path, not a warning, once the window is derived: with a derived window
  an empty pool means the geometry is genuinely impossible and the caller needs to know.

`Config.retrieval_z_window` stays as the fallback and the ablation handle. T10 §4 carries the matching
requirement for the ablation table: **A5 (`retrieval_w_z = 0`) must not be run at a fixed window**,
or the ablation measures the window instead of the z term — the same trap T04's G2.3 fell into and
recorded.

## 2. Calibration — `spatialcpav25_gen/infer/calibrate.py`, all leakage-free

```python
def calibrate_lengthscale(model, vol: TrainingVolume, cfg) -> LengthscaleCalibration
```
Bisection on `ell_xy` so the **generated** section's mean Moran's I matches the mean Moran's I of the
**flanking training sections**. 6–8 iterations. `ell_z` from between-section correlation decay
(T03's `fit_lengthscale_from_sections`).

Ground truth is never consulted. Assert in a test that the calibrator's signature cannot accept
held-out sections.

**The objective is unimodal, not monotone (T03/GATE 1, measured).** Mean `I_gen` rises with `ell`,
turns over, and falls: Moran's I of expression is structured variance over total variance, and a
stationary field loses within-window variance once its correlation length approaches the section.
The maximiser was measured at 0.086 of the in-plane extent on the 3000 µm gate fixture and 0.112 on
the 1000 µm one — and at 2.5× and 0.8× the *fitted* `ell` respectively, because the variogram fit is
itself window-biased at a narrow field of view. A bisection that ignores this walks into a boundary
and returns it as if it were an answer. The calibrator therefore:

1. **Caps the bracket** at
   `ell_max = min(cfg.calibration_ell_max_extent_frac * extent, cfg.calibration_ell_max_fitted_multiple * ell_fitted)`
   (0.2 × extent and 2 × fitted by default) and starts from `ell_min = cfg.variogram_ell_min_factor *
   median_nn_dist`. Neither cap is a guarantee — 0.2 × extent is about twice the measured maximiser —
   so they bound the search, they do not define it.
2. **Locates the maximum** on the bracket before bisecting: evaluate `I_gen` on a
   `cfg.bisection_grid_size`-point log grid, take the maximiser, and bisect only on
   `[ell_min, ell_argmax]`, which is monotone by construction.
3. **Returns a status.** `LengthscaleCalibration` carries `ell`, `status ∈ {"converged",
   "target_unreachable", "boundary"}`, the achieved `I_gen` and the target. When the target mean
   Moran's I exceeds `max(I_gen)` over the bracket there is **no root**: return the maximiser with
   `status="target_unreachable"` and log a warning naming both numbers. Never return a bracket
   endpoint as though bisection had converged (Convention 6 — no silent fallbacks).
4. **Is written back by an explicit writer** (amended after T09; the original spec named the
   return value but not who applies it, and the two were never connected — the calibrator
   measured an `ell` and generation went on using the config's own).

```python
def apply_lengthscale(cfg: Config, calibration: LengthscaleCalibration) -> Config
```
   `ell` reaches the GRF only as `Config.ell_xy` / `Config.ell_z`, which `generate_section`
   swaps into the field through `with_lengthscale`. `apply_lengthscale` is the only sanctioned
   way to get a calibrated length-scale into generation.

   **Only a `"converged"` axis is applied.** `target_unreachable` and `boundary` both mean the
   search found no root, and T09 measured what such a value is worth: on an objective that is
   constant in `ell` — which `expr_mode="cross-mix"` produces, being flat to ten decimal places
   across a 15× `ell_z` sweep — the returned number is whichever grid point tied first.
   Applying it would ship a tie-break as a length-scale. A non-converged axis is **dropped with
   a `CalibrationNotAppliedWarning` naming the achieved and target values**, and the config's
   existing value stands.

   The two axes are decided **separately**, on `status` and `ell_z_status`: an in-plane
   calibration that converged is not made worthless by a stack too short to constrain `ell_z`
   (open risk R1), and discarding it would throw away a real measurement. The cost — a
   half-applied result whose anisotropy ratio mixes a calibrated axis with a default one — is
   why each dropped axis warns rather than passing silently.

Acceptance tests:
- `test_calibrator_recovers_a_reachable_target` — plant a target inside the achievable range;
  `status == "converged"` and the achieved `I_gen` is within tolerance.
- `test_apply_lengthscale_writes_a_converged_result_through_to_the_prior` — a converged
  calibration reaches the field the generator builds, not just the returned object.
- `test_apply_lengthscale_drops_a_non_converged_axis_with_both_numbers` — an unreachable axis
  leaves `Config` alone and warns, naming achieved and target.
- `test_apply_lengthscale_decides_the_two_axes_separately` — a converged `ell_xy` survives an
  unresolvable `ell_z`.
- `test_calibrator_reports_unreachable_target` — plant a target above `max(I_gen)`; the calibrator
  returns `status == "target_unreachable"`, the returned `ell` **is** the grid maximiser, and it does
  not sit at either bracket endpoint. This is the branch T03 measured and the one a naive bisection
  gets wrong.
- `test_calibrator_bracket_respects_the_caps` — on a narrow-field-of-view volume the bracket's upper
  end is the extent cap, not the fitted-multiple cap.

**One global `ell`, and per-module agreement as a diagnostic only** (settled, SPEC_QUESTIONS A2).
The original spec calibrated `ell` per gene-module (Leiden clusters of gene embedding space, ~10
modules), which is not implementable as written: `ell` parameterises the **latent** field (`d_h = 64`
channels queried at cell positions) and gene modules only exist downstream of the decoder, so "the
`ell` for module *m*" would require the field to know which latent channels that module reads from —
not a property the decoder is constrained to have. Calibrate **one global**
`ell = (ell_x, ell_y, ell_z)` — which is also what GATE 1's monotonicity and unimodality criteria are
defined on — and report per-module Moran's I agreement as a **diagnostic table** in
`reports/config_selection_*.md`, not as a target.

*Escalation, if the diagnostic is poor.* The cheap version is to partition the `d_h` latent channels
into groups with their own `ell` and add a loss tying gene modules to channel groups. `T03`'s
`with_lengthscale` already makes the field side free (a grouped field is a channel-wise
concatenation of rescaled copies of one realisation, no redraw and no new state), so the cost is all
in the tying loss. **That is a design change and must be decided explicitly, with the diagnostic
table as the evidence — not improvised inside the calibration loop.**

```python
def calibrate_detection(model, vol, cfg) -> DetectionCalibration
```
Per-gene affine correction on the `pi` logit so generated detection rates match the flanking
sections'. Fit on LOSO reconstructions; apply at generation.

**Calibrate the mean–variance relation too, not just `pi`** (settled; `design/v23_design.md` §3.5
asks for both and `specs/` had only `pi` — SPEC_QUESTIONS D-table). Add a per-gene correction on
`log theta` fitted the same way, so the generated mean–variance curve matches the flanking sections'
rather than only their detection rates. The two are not substitutes: `pi` moves the zeros and
`theta` moves the spread of the non-zeros, and T06's `test_mean_variance_relation` (log-log slope
within 15%) is the property this protects at inference. `DetectionCalibration` carries both
corrections and records which sections it was fitted on.

#### How the `theta` correction is tested: on the premise, not on a tolerance (added at T10)

`test_detection_calibration_matches_the_fold_it_was_fitted_on` originally asserted, without
condition, that applying the correction moves the mean-variance slope *towards* the real section's.
That assertion has a premise — **that there is an error to correct** — and the premise stopped
holding when `decoder_mu_link` became `exp` at T10: on the synthetic fixture the decoded slope is
then within **0.005** of real, and the correction, solving for detection, moves it by 0.031 and the
test goes red. Relaxing the threshold would have hidden both halves of the question. State them
separately instead.

1. **No headroom -> do no harm.** Measure the estimator's own draw-to-draw spread first — repeated
   `sample_counts` from *one* `(mu, theta, pi)` — and compare the uncalibrated error against it.
   When the error is at or below that floor there is nothing to correct, and the requirement is
   only that the calibrated slope stays inside the same floor. Nothing may be asserted below the
   noise of the estimator doing the asserting; this is the device already used for the covariance
   ceiling (B16) and the oblique-correlation ceiling.

2. **Headroom -> take it.** The fixture cannot supply this arm under `exp`, so **construct** the
   condition real STARmap showed — a decoded slope of **2.121** against a real **1.738**, the
   decoder putting variance in the count draw rather than in `mu`. Shrinking `theta` is what
   over-dispersion is in a ZINB, so the arm is built by shrinking it, and the requirement is that
   the solver removes **at least half** the constructed headroom. The test first asserts the
   construction actually created headroom above the floor, so it cannot silently degenerate.

Arm 2 needs no model and no fit: `calibrate.solve_detection_shifts(mu, theta, pi, real, cfg)` is
the array half of `_fold_statistics`, split out at T10 for exactly this. Keep it that way — a
solver assertion that costs a training run gets run rarely and therefore protects little.

```python
def calibrate_anchor_weight(model, vol, cfg) -> IsotonicRegressor
```
As described in §1.

## 3. Automatic per-dataset configuration — `spatialcpav25_gen/train/select.py`

```python
def select_config(vol: TrainingVolume, base_cfg: Config) -> Config
```

Coordinate-descent over a small gate grid, scored by the six target metrics on **LOSO
reconstructions of training sections**:

| Gate | Options |
|---|---|
| `layout_mode` | field / hybrid / resample — **but see the note below: the fixture cannot rank this gate** |
| `prior_mode` | correlated / iid |
| `expr_mode` | zinb-flow / cross-mix / auto-blend |
| `text_emb` | medcpt+residual / lookup-only |
| **`train_steps` × metric weights** | **one joint gate, four cells** — see below |

> ⚠️ **`layout_mode` is not selectable on the synthetic fixture, at any budget or seed count
> (2026-08-25).** The gate compares a generated layout against a referent, and on the fixture that
> referent is unrepresentative: the fixture's flanking sections score 0.5319 against a 0.9221
> self-score (58 % of ceiling) where real serial sections at 50 µm score 0.7765 against a 0.9808
> oracle (79 %). Its generative law decorrelates fast in z, so copying a neighbour is a weak
> strategy there and a generative layout is systematically over-rewarded. This is not a power
> problem — more seeds do not fix an unrepresentative referent — and it is why the fixture's
> tie-break shipped `hybrid` while real data ships `resample` (`specs/05` §4a,
> `progress/fixture_limitations.md` §2). A per-dataset selection run on **real** data decides this
> gate; the fixture's verdict is recorded for comparison only
> (`reports/t09_layout_mode_gate_grid.md`).
>
> **Cost note for the merged gate.** `layout_mode` is read only at generation time — fitting the
> fixture at all three modes with one seed gives bitwise identical weights over all 96 parameter
> and buffer tensors — so the 18-cell full-budget gate refits 3x more than it needs to: 6 fits
> (`prior_mode` × `expr_mode`) serve all 18 cells if the `layout_mode` axis reuses one fit. The
> scores are unchanged; the contrast is cleaner, because it carries no fit-to-fit noise.

- Fit a **reduced-epoch** model (25% of `cfg.epochs`) per candidate — **except for the budget
  gate, which is scored at its own budget** (see below), and except for any gate the
  **training-free-option rule** below disqualifies.
- Score = median rank across the six metrics over LOSO folds.
- Coordinate descent, 2 passes, over the gates the rule leaves eligible.
- Persist the chosen config and the full score table to `reports/config_selection_{dataset}.md`.

### The training-free-option rule (added at T09, from a measured failure)

> **A gate is scored at the selected budget when either of two conditions holds.**
>
> 1. **It has a training-free option.** If any option of a gate reaches its final behaviour
>    without training — because it copies real data rather than generating it — then that option
>    is already at full strength at any budget while its rivals are not, and a reduced-budget
>    comparison measures the budget rather than the gate.
> 2. **The incumbent is unconverged at the reduced budget.** Even when every option trains, a
>    gate decided on a model that behaves nothing like the shipped one is not decided. If the
>    incumbent's own score at the reduced budget falls short of its score at the selected budget
>    by more than `Config.selection_convergence_tol` on at least
>    `Config.selection_convergence_min_metrics` of the six metrics, the reduced budget is not a
>    usable proxy for *any* remaining gate and all of them are escalated.
>
> Gates qualifying under (1) are scored **jointly**, because their errors compound through
> coordinate descent's ordering. Condition (2) is a property of the run, not of a gate, so it is
> **measured** each time: the incumbent is scored once at the reduced budget and compared with
> the selected-budget score the search already has for it.

This is a rule for every future gate, not a patch for the ones that failed. **When a gate is added
to the table above, classify each of its options as training-free or trained and record the
classification** — in `train/select.py`'s `TRAINING_FREE_OPTIONS`, which is the machine-readable
form of condition (1) and what the selector reads to decide a gate's budget. A gate whose options
are all trained keeps the reduced budget *unless condition (2) fires*, which no static declaration
can predict: `incumbent_is_unconverged` measures it at run time and the report says which rule
escalated which gate.

*The measurement behind condition (2)* (open risk **R9**): with the rule at condition (1) alone,
`text_emb_mode` kept the reduced budget — both its options train — and was decided at 600 steps
under a `zinb-flow` incumbent scoring **0.5997 / 0.6523** on `morans_pearson` against **0.96** at
the selected budget. Its winner changed from `medcpt` to `lookup`, and `lookup` **disables the
MedCPT channel**, which is the paper's open-vocabulary claim. A gate that can switch off a headline
capability, decided on a model six-tenths of the way to the shipped one's behaviour, is not
decided. Both options are handicapped equally, so this is not condition (1)'s bias — it is the
separate failure that the proxy itself is invalid.

*The measurement behind the rule* (`reports/r8_budget_grid.md`, open risk **R8**): at 25% of the
budget `expr_mode="cross-mix"` won under both priors and at full budget it came **last** under
both; `prior_mode="iid"` won at 25% on exactly the two expression paths where the prior can act and
lost on both at full budget. The cause is visible in one column — from 600 to 2400 steps
`morans_pearson` gains **+0.3432** for `zinb-flow` and **−0.0180** for `cross-mix`, because
`cross-mix` copies donor counts and needs no training. The shipped configuration ranked **fifth of
six** at the budget it was actually trained at. The two gates also compounded: fixing
`prior_mode="iid"` first dropped `zinb-flow` from rank 2.5 to 3.0 *before* the `expr_mode` gate was
scored, which is why qualifying gates are scored jointly rather than one after another.

### The repeated-seed rule (added at T09, from a measured envelope)

> **Any measurement that reaches a paper claim runs at least
> `Config.claim_min_seeds` seeds and reports the spread, not a point estimate.** A single-seed
> number cannot distinguish "wins" from "wins by less than the run-to-run variation", and a
> benchmark whose purpose is claiming wins cannot leave those two indistinguishable.

*Why it exists, measured.* Fitting one configuration twice — same config, same seed, different
process — moved its scores by up to **0.0120** (`umap_mixing`), while the largest difference
between the two `text_emb_mode` options at the selected budget was **0.0110**. Re-running the
identical configuration moved the score as much as changing the gate did. That gate was therefore
undecidable at one seed at any budget, which no amount of extra training would have revealed.

This belongs in the paper's methods as a **strength**, not a caveat: the campaign states its own
reproducibility envelope and reports every claim against it, which is strictly more than a
single-seed table can support.

### The capability tie-break (added at T09)

> **When two options are separated by less than the reproducibility envelope, prefer the option
> whose headline capability is *exercised on this dataset*.** A capability that is present but
> **inert** does not count, and neither does the nominal rank winner: below the envelope the rank
> ordering is not evidence.

Two worked examples, both from the fixture, and the second is the reason for the word *exercised*:

* `text_emb_mode`. `lookup` outranks `medcpt` (1.2 against 1.8) on margins of at most 0.011 —
  inside the envelope. `lookup` **disables the MedCPT channel**, which is the open-vocabulary
  claim; `medcpt` exercises it, since the text embeddings are live on every gene. The tie-break
  selects **`medcpt`**. Shipping `lookup` here would switch off a headline capability on a margin
  smaller than the noise floor.
* `expr_mode`, and the trap. `auto-blend` and `zinb-flow` scored **identically** — max difference
  0.0000 — so the tie-break applies. `auto-blend` is nominally the richer capability (T09's
  uncertainty-gated anchoring). But on this fixture the fitted `w(v)` is **0 at every knot**, so
  the blend passes the flow's draw through unmixed and the two cells are the *same model* under
  two labels. The capability is present and does nothing, so it does not count, and the tie-break
  selects **`zinb-flow`** — the honest label for what the model actually is. Preferring
  `auto-blend` would ship a feature claim that no emitted count depends on.

A gate resolved by the tie-break is **reported as tie-broken**, with both scores and the envelope,
so a reader can see the choice was made on capability rather than on measurement.

*Applying the rule to the current table:*

| Gate | Training-free option | Budget |
|---|---|---|
| `layout_mode` | `resample` — reuses real cell positions | **selected** |
| `prior_mode` | `iid` — never queries the fitted field | **selected** |
| `expr_mode` | `cross-mix` — emits donor counts verbatim | **selected** |
| `text_emb` | none; both options train | reduced (25%) **unless condition (2) fires** |

The first three form **one joint gate of 3 × 2 × 3 = 18 cells** at the selected budget. `text_emb`
stays on coordinate descent at the reduced budget. Note that `layout_mode` did *not* reverse at the
reduced budget on the fixture — it qualifies by the rule regardless, because the rule is about
whether the comparison is sound, not about whether it happened to come out wrong this time.

*Cost, and why it is accepted.* 18 full-budget fits replace 8 reduced-budget ones — on the
synthetic fixture roughly 8.75 h against 40 min. Selection runs once per dataset and decides what
T10 benchmarks, so a selection that is cheap and wrong is worth less than one that is expensive and
right. Because the run is long, the scorer is **checkpointed per cell**: every scored cell is
appended to a CSV keyed by the candidate's config hash and budget, and a re-run skips what is
already there, so an interrupted selection resumes rather than restarting.

### The training budget and the metric-aware weights are gates, and they must be selected *together* (added at T08)

`train_steps` and T08's `w_autocorr` / `w_profile` / `w_distribution` are **not constants**. T08
measured them and found that their effect **reverses with the budget**, so any fixed value for
either is a value fitted to whatever budget happened to be in use when it was chosen:

| statistic | off@1200 | on@1200 | off@2400 | on@2400 |
|---|---|---|---|---|
| reconstruction (nats/pair) | 1.5901 | 1.6843 | **1.5703** | 1.5885 |
| gene–gene Frobenius | 9.000 | 11.154 | 9.049 | **8.489** |
| Moran's MAE | 0.0287 | 0.0408 | 0.0339 | **0.0279** |
| marker-depth r | 0.978 | 0.967 | 0.983 | **0.990** |

T08 ships all three weights at **0** because that is what the measurement supports *at T06's
1200-step budget*, and `specs/10`'s A2 is written as an addition experiment against it. But 1200
steps is not a neutral reference point — it is where T06 stopped **because the arm without the
terms starts degrading there**, which is open risk R4's own symptom. **So the shipped 0 is
calibrated to an undertrained model, and it is this task's job to stop it being a constant.**

Three requirements, and the second is the one a naive implementation gets wrong:

1. **The budget is a `Config` field and a gate.** `Config.train_steps` (added at T08) is the
   value `train_ctfflow` is called with, so a selected budget is persisted, hashed into the run,
   and reported like every other gate. Options: `1×` and `2×` the base budget at minimum.
2. **The budget and the weights are ONE gate with four cells, not two gates visited in turn.**
   Coordinate descent varies one gate at a time from the incumbent, so starting at
   `(1200, weights off)` it would score `(1200, weights on)` — which loses, by the table above —
   conclude the weights are harmful, and never reach `(2400, weights on)`, which is the cell that
   wins on four of six statistics. **Coordinate descent over interacting gates reproduces exactly
   the error this amendment exists to prevent.** Score all four cells of the
   `{1×, 2×} × {off, spec weights}` grid jointly and take the best; it costs four fits, once.
3. **The budget gate is scored at its own budget.** The "25% of `cfg.epochs`" reduction above is a
   cost control for gates whose effect is visible early. It is invalid here by construction: a
   reduced-epoch fit of a `2×` candidate *is* the `1×` candidate, so the reduction would compare
   a budget against itself and always return "no difference". Fit each cell of the joint gate at
   the budget it names.

The leakage discipline is unchanged and binds harder here, because the budget is the gate most
easily fitted to a test set: the selection runs on **internal LOSO over training sections only**
(T08's `LOSOScheduler`), never against held-out sections. T06's `TRAIN_STEPS = 1200` was chosen by
reading the fixture's own degradation curve, which is the mistake this replaces — recorded in
`SPEC_QUESTIONS` B10/R4 and now in the risk table as the thing T09 closes.

**This is the "no regression" guarantee:** `layout_mode=resample` + `expr_mode=cross-mix` reproduces
the previous version's behaviour, so if the new machinery does not help on a dataset it is switched
off automatically. Add `test_selector_can_recover_v20_config` asserting that config is reachable and
selected when the new components are artificially degraded.

## Acceptance tests

- `test_generate_shapes_and_dtypes` — AnnData valid; X integer-valued; no NaN; `uns` carries plane,
  seed, config hash.
- `test_generate_deterministic` — same seed → identical output.
- `test_stack_coherence` — two sections from `generate_stack` at nearby z have expression correlation
  decaying smoothly with |Δz|, with no discontinuity at training-section z values. A spike at real-
  section z means the model is copying rather than interpolating.
- `test_oblique_intersection_agreement` — two oblique sections intersecting inside the tissue agree
  along their intersection: cell-type concordance > 0.8, expression correlation > 0.85. **This is
  the headline capability experiment (E5); make it a first-class test.**
- `test_calibration_no_leakage` — held-out sections cannot be passed (type error), and calibration
  results are identical whether or not held-out data exists in the parent object.
- `test_calibration_converges` — bisection reaches |I_gen − I_flank| < 0.02 within 8 iterations on
  the fixture.
- `test_anchor_weight_monotone` — the fitted isotonic map is non-increasing in `v`.
- `test_selector_runs_and_persists` — selection completes on the fixture and writes the report.
- `test_selector_can_recover_v20_config` — described above.
- `test_budget_and_metric_weights_are_selected_jointly` — the four cells of
  `{1×, 2×} × {weights off, spec weights}` are all scored, and the selector can return
  `(2×, weights on)`. Asserted **by construction on a stub scorer** that scores that cell best and
  every other cell worse: a one-gate-at-a-time selector cannot return it, so this test fails on the
  implementation the amendment above forbids.
- `test_budget_gate_is_not_scored_at_a_reduced_budget` — the fits issued for the two budget
  candidates use *different* step counts. Pins §3's third requirement, without which the budget gate
  silently compares a candidate against itself.
- `test_selection_never_sees_heldout` — `select_config` takes a `TrainingVolume` and raises
  `TypeError` on `HeldOutSections`; the chosen budget is identical whether or not held-out sections
  exist in the parent object. The budget is the gate most easily fitted to a test set.

## Definition of done

On the fixture, LOSO reconstruction beats both `resample`-mode and the independent-donor baseline on
≥ 4 of the 6 target metrics; `reports/config_selection_synthetic.md` exists. `PROGRESS.md` records
the calibrated `ell` values and the selected config — **including the selected `train_steps` and
whether the metric-aware weights came out on or off**, since that is the decision T08 deferred here.

## Do NOT

- Do not consult held-out sections anywhere in calibration or selection.
- Do not expose gap/alpha/edit flags to users — the whole point is that these are inferred.
- Do not draw a fresh GRF per section inside a stack (destroys coherence).
- **Do not hardcode the training budget or the metric-aware weights, and do not visit them as
  separate coordinate-descent gates.** Both are selected, jointly, per dataset. A fixed value for
  either is a value fitted to one budget on one fixture — see §3.
- **Do not score the budget gate at a reduced budget.** It compares a candidate against itself and
  returns a null result for a gate that demonstrably moves four of six statistics.
