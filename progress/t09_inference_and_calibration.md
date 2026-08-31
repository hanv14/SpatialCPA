# T09 — Inference, leakage-free calibration, and automatic configuration

Part of [PROGRESS.md](../PROGRESS.md).

### T09 — generation, calibration and the selector (2026-08-19)

Built `spatialcpav25_gen/infer/generate.py` (the single generation entry point and the three
multi-section wrappers), `spatialcpav25_gen/infer/calibrate.py` (four leakage-free calibrators),
`spatialcpav25_gen/train/select.py` (the per-dataset selector, the six selection metrics and the
report writer), and `scripts/t09_report.py`. `CTFFlow.generate` now delegates to
`generate_section` (SPEC_QUESTIONS C9). 42 tests, 26 of them in the fast suite.

Six results are worth reading before the tables:

1. **The joint gate decided the budget and the metric-aware weights**, and it decided them the way
   T08's measurement said it would: **`train_steps = 2400`, all three weights at 0.5**. The weights
   *lose* at 1x and *win* at 2x, so a selector that visited the two gates in turn could not have
   reached the winning cell. All four cells are below.
2. **The selector then switched two of the new components off** — `prior_mode = iid`,
   `expr_mode = cross-mix` — which is the no-regression guarantee firing on the fixture rather
   than in a test. It also exposed the limit of the reduced-budget heuristic those gates are scored
   under, which is the finding in §7.
3. **`ell_z` is calibrated, not fitted** (open risk R1's remedy 2, with remedy 3 as the guard) —
   and both `ell` calibrations came back **`target_unreachable`**, i.e. the generated section is
   *less* autocorrelated than the tissue at every admissible length-scale. The guard did its job:
   no bracket endpoint is returned dressed as a fitted value.
4. **A calibration bug the run found: `ell` cannot be calibrated under `prior_mode = "iid"`.** The
   selector chose `iid`, under which the prior is white noise and never queries the field, so the
   objective is flat and the status was an answer to a question that could not be asked.
   `calibrate_lengthscale` now **refuses** that config, and the calibration is re-measured under
   the prior `ell` parameterises.
5. **E5's expression criterion is above the achievable ceiling** and is now a strict xfail with the
   ceiling measured beside it; its cell-type half passes as written.
6. **The detection / dispersion calibration has no headroom on this fixture** and is therefore not
   applied by default. Both halves work on the fold they are fitted on; neither transfers.

---

#### 1. Generation (`infer/generate.py`)

`generate_section(model, plane, vol, cfg, seed)` runs the eight steps of `specs/09` §1 and emits an
AnnData carrying the plane, both seeds, the config hash, the retrieval diagnostics and the
**boundary flag**. `generate_stack`, `generate_oblique` and `generate_curved` resolve one
`grf_seed` per call and pass it to every section, so a stack is slices of one object; reusing the
seed across calls extends that between calls, which is why it is a parameter and not an internal.

Three things the spec asks for in prose and that are now mechanisms:

* **the retrieval z window is derived from the geometry.** `required_z_window(vol, plane, cfg,
  exclude_z)` returns `max(Config.retrieval_z_window, ceil(gap to the nearest admissible section /
  median spacing))`, and `calibrate_retrieval_window(vol, cfg)` does the same for a whole training
  stack. On the `consecutive-3` holdout the fixture's training stack is z = 0, 200, 250, 300, 350,
  400 µm: the largest surviving gap is **200 µm = 4 spacings**, so the derived window is **4** where
  the constant is 3 — the exact case T06 measured `EmptyCandidatePoolWarning` firing on 100–110 of
  every 512 cells. `Config.retrieval_z_window` stays as the fallback and as T10's A5 handle
  (`generate_section(..., z_window=...)` pins it).
* **an empty candidate pool is a failure of the generation path**, not a warning: above
  `Config.generation_empty_pool_tol` (1 % of cells) `generate_section` raises
  `EmptyCandidatePoolError` naming the plane and the fraction.
* **uncertainty-gated anchoring** is `Config.expr_mode = "auto-blend"`: `M =
  Config.n_uncertainty_samples` flow samples under *different GRF realisations* give a per-cell
  latent variance, the fitted isotonic `w(v)` turns it into a Bernoulli mixing weight, and the mix
  is the v20 cross-mix per gene, so every emitted value is still a count. Sample 0 is drawn under
  the call's **own** realisation and is the one kept — averaging the M samples would break stack
  coherence, because the average of M realisations is not a realisation. There is no `gap_scale`,
  no `alpha_tol` and no `edit_weight` anywhere in the package, and `auto-blend` without a fitted
  `w(v)` raises rather than defaulting to one.

#### 2. Acceptance tests

| test | criterion | measured | status |
|---|---|---|---|
| `test_generate_shapes_and_dtypes` | valid AnnData, integer X, no NaN, `uns` carries plane/seed/config hash | passes; `uns` also carries the boundary flag and the derived window | pass |
| `test_generate_deterministic` | same seed → identical output | bitwise on counts **and** positions; a different seed differs | pass |
| `test_stack_coherence` | correlation decays with `|dz|`, no spike at a training section's depth | decay confirmed; on-section coherence within **`COPY_SPIKE_MAX` = 0.10** of its neighbours' median | pass |
| `test_oblique_intersection_agreement` (E5) | concordance > 0.8, expression r > 0.85 | concordance **0.814** (ceiling 0.781); expression r **0.724** against a measured ceiling of **0.726** — 99.7 % of achievable | pass (ceiling-relative), literal criterion a **strict xfail** |
| `test_calibration_no_leakage` | held-out cannot be passed; result independent of the parent | `TypeError` at all five calibrator doors, for `HeldOutSections` *and* for a plain `Volume`; identical result from a standalone training volume | pass |
| `test_calibration_converges` | \|I_gen − I_flank\| < 0.02 within 8 iterations | see §4 | pass |
| `test_anchor_weight_monotone` | the fitted isotonic map is non-increasing | non-increasing on the fitted map and on a randomised PAVA check (200 random inputs, both directions) | pass |
| `test_selector_runs_and_persists` | selection completes and writes the report | end-to-end on the real scorer at toy budgets | pass |
| `test_selector_can_recover_v20_config` | `resample` + `cross-mix` reachable and selected | selected when the new components are degraded | pass |
| `test_budget_and_metric_weights_are_selected_jointly` | all four cells scored; `(2×, on)` returnable | all four scored; the conjunctive stub's winning cell is returned | pass |
| `test_budget_gate_is_not_scored_at_a_reduced_budget` | the two budget candidates fitted at different step counts | 1× fitted at 1×, 2× at 2×, per candidate | pass |
| `test_selection_never_sees_heldout` | `TypeError`; budget independent of the holdout's existence | both | pass |

#### 3. E5, and a criterion above its ceiling

`specs/09`'s intersection test asks for cell-type concordance > 0.8 **and** expression correlation
> 0.85 between the cells two crossing oblique sections place at the same physical point. Both
sections emit *draws* and their layouts are independent, so two matched cells carry independent
ZINB noise on one mean. The achievable ceiling is what two draws of **one** plane reach under one
realisation — the same field, the same model, nothing differing but the draw — and the test measures
it with the same code:

| arm (600-step fit) | cell-type concordance | expression r |
|---|---|---|
| oblique pair, along the intersection | **0.814** | **0.724** |
| ceiling: two draws of one plane | 0.781 | 0.726 |
| the same at a 60-step fit | 0.450 (ceiling 0.515) | 0.148 (ceiling 0.153) |

So the concordance criterion passes as written and the expression one is **unreachable by any
model**: at 99.7 % of the ceiling there is 0.002 of headroom and the spec asks for 0.126 more. The
headline test therefore asserts concordance absolutely and expression correlation at ≥ 0.95 × the
measured ceiling, and `test_oblique_expression_correlation_reaches_the_spec_threshold` keeps the
literal 0.85 as a **strict xfail** — the pattern T06 used for its covariance criterion (B16).
Recorded as SPEC_QUESTIONS **C27**.

The 60-step row is why the E5 test pays for its own longer fit: at that budget the model has no
cell-type field yet and the test would be measuring the budget.

#### 4. Calibration (`infer/calibrate.py`)

**`ell_xy`.** Bisection against the flanking *training* sections' mean Moran's I, on a bracket
capped at `min(0.2 × extent, 2 × fitted)`, with the maximum located on a
`Config.bisection_grid_size`-point log grid first and the bisection run only on the monotone branch
below it. Three statuses (`converged` / `target_unreachable` / `boundary`), and the unreachable
branch returns the **grid maximiser** with a warning naming both numbers — never a bracket endpoint
dressed as an answer.

**`ell_z` — open risk R1, closed as a mechanism.** Remedy 2 is implemented: the target is the
*observed* adjacent-section correlation of the real training sections (grid-averaged profiles, the
same construction the along-z variogram uses), and the objective is that statistic on a two-section
virtual stack generated at the same depths with both excluded from retrieval. Remedy 3 is the guard:
the variogram's `ell_z` enters as the bracket's **upper endpoint**
(`Config.calibration_ell_z_max_fitted_multiple = 1.0`), never as a value, and a volume that cannot
close the bracket reports `target_unreachable`, which
`LengthscaleCalibration.ell_z_is_upper_bound` exposes and the report prints. The fixture's fitted
`ell_z` is **365 µm** against a 200 µm ground truth (the T03 measurement on this fixture, with
`LengthscaleFitWarning` firing at 58 % saturation), so the bracket runs to 365 µm and the calibrated
value is whatever matches the observed decay below it.

**Detection and dispersion.** `DetectionCalibration` carries **both** corrections (design §3.5, the
D-table): an affine on the `pi` logit and an affine on `log theta`, slope fixed at 1 and the
per-gene intercept **solved exactly** by bisection on a monotone closed form. Two things had to be
got right, and both were measured wrong first:

* the two are **not orthogonal** — `theta` enters the detection rate through `P_NB(0)`, so solving
  them independently made the per-gene detection MAE *worse*, **0.055 → 0.230**, while each matched
  its own target exactly. The solves now alternate twice;
* the `theta` target is the **mean–variance relation** (`Var = mean + phi mean²` at the model's own
  mean), not the absolute variance. Matching the absolute variance asks `theta` to repair a wrong
  *mean*: on the dense genes (model mean 14 against a section variance of 2.7e4) it drove `theta` to
  0.007 and collapsed their detection rate from 0.68 to 0.06.

**And then the correction has no headroom on this fixture** (SPEC_QUESTIONS C28). At a 600-step fit:

| quantity | value |
|---|---|
| model's own per-gene detection MAE | **0.0217** |
| real per-gene rate variation *between training sections* | **0.0397** |
| after a correction fitted on other sections (5 folds) | 0.0326 |
| after a correction fitted on the section it is applied to (oracle, unreachable) | **0.0191** |

The model is already inside the tissue's own section-to-section noise, so a leakage-free correction
imports more of that noise than it removes model error. Both halves work where they are fitted —
the oracle improves detection MAE and moves the mean–variance slope toward the real one — and
neither transfers. `generate_section(..., calibration=None)` therefore does **not** apply them by
default; `test_detection_calibration_does_not_transfer_between_sections` asserts the diagnosis, so
the day a dataset gives the correction something to do the test fails and this measurement is
re-run. T10 decides on real data.

**`w(v)`.** A 1-D isotonic regression (PAVA, written here rather than taken from sklearn so the
monotonicity test tests this code) from per-cell latent variance to Bernoulli mixing weight, fitted
on LOSO folds by scoring a grid of weights with a **per-cell profile correlation** against the real
section. Deliberately not an element-wise error: the expected element-wise error of a Bernoulli
mixture is *linear* in the weight, so its minimiser is always 0 or 1 and an isotonic map would have
nothing to say.

*A bug worth recording.* The first PAVA merged into the wrong block —
`block[-2] += block.pop()` resolves `-2` **after** the pop — and still passed a five-point example.
It is now a randomised test over 200 inputs in both directions, and the guard against a NaN knot is
explicit, because every comparison against NaN is False and a NaN would have made a non-monotone map
report itself as monotone.

#### 5. The joint gate, and what the selector chose

`scripts/t09_report.py --steps 1200`, synthetic fixture, `alternating` holdout, three interior
LOSO folds, 19 fits, 9195 s. **All four cells, as `specs/09` requires — not just the winner:**

| cell | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|
| 1x, weights off | 1200 | 0.9197 | 0.8710 | 0.8501 | −0.0527 | 0.0382 | −0.0595 | 3.0 |
| 1x, weights on | 1200 | 0.8813 | 0.8589 | 0.6560 | −0.0387 | **0.0599** | −0.0577 | 3.5 |
| 2x, weights off | 2400 | 0.9316 | 0.9022 | 0.9145 | −0.0519 | 0.0195 | −0.0502 | 2.0 |
| **2x, weights on** | **2400** | **0.9368** | **0.9123** | **0.9163** | **−0.0370** | 0.0554 | **−0.0410** | **1.0** |

**T08's interaction reproduces exactly.** At 1200 steps the metric-aware weights *lose* (rank 3.5
against 3.0) and at 2400 they *win* (1.0 against 2.0), taking four of the six metrics outright.
A coordinate-descent selector starting from `(1200, off)` scores `(1200, on)`, loses, and stops —
it cannot reach the cell that wins, which is the whole reason `specs/09` §3 makes this one gate.
**Selected: `train_steps = 2400`, `w_autocorr = w_profile = w_distribution = 0.5`.** That is the
decision T08 deferred, and the 0 it shipped is now a per-dataset choice rather than a constant.

Coordinate descent then ran two passes over the four remaining gates at
`selection_reduced_epoch_frac` x 2400 = **600** steps and chose:

| gate | selected | runner-up | margin (median rank) |
|---|---|---|---|
| `layout_mode` | **field** | hybrid | 1.5 vs 1.5 (tie on rank; field leads on 4 of 6 metrics) |
| `prior_mode` | **iid** | correlated | 1.0 vs 2.0 |
| `expr_mode` | **cross-mix** | auto-blend, then zinb-flow | 1.5 vs 2.0 vs 2.5 |
| `text_emb_mode` | **medcpt** | lookup | 1.0 vs 2.0 |

Two of those deserve to be read as results rather than as settings:

* **`expr_mode = cross-mix` is the no-regression guarantee firing on real numbers.** At 600 steps
  the v20 expression path scores 0.8495 / 0.7914 / 0.6811 on the first three metrics against the
  flow path's 0.6494 / 0.5769 / 0.2057. The selector switched the new machinery off by itself,
  which is exactly what `specs/09` says must happen when it does not help — and
  `test_selector_can_recover_v20_config` is the test of the mechanism that just did it.
* **`prior_mode = iid` beating `correlated`** is the reduced-budget heuristic biting a gate it
  should not. See §7.

#### 6. Calibration on the fixture

**And the run found a defect in this task's own code before it found anything about the fixture.**
The selector chose `prior_mode = "iid"`; under it the prior is white noise that never queries the
GRF, so `ell` has *no effect at all* on the generated section. The first calibration therefore
bisected a flat objective and returned `ell_xy = 7.0 um` (the bracket's low end) with status
`target_unreachable` — a correct-looking answer to a question that could not be asked.
`calibrate_lengthscale` and `calibrate_ell_z` now **refuse** a non-GRF prior with a message that
says why (`test_calibrator_refuses_a_prior_that_ignores_ell`), and the calibration arm is the
selected config with `prior_mode = "correlated"` restored, stated as such in the report.

The corrected arm — the selected config with `prior_mode = "correlated"` restored, `train_steps
= 2400`, targets measured on flanking **training** sections `synthetic_s03` and `synthetic_s07`:

| quantity | value | status |
|---|---|---|
| `ell_xy` | 7.0 um | `target_unreachable` |
| `ell_z` | 25.0 um | `target_unreachable` |
| fitted `ell` (variogram) | 136.8 / 364.6 um | bracket endpoint, never a value |
| Moran's I | gen **0.2390** vs flanking **0.4102** | 0 bisection iterations |
| between-section r | gen **0.6734** vs observed **0.9182** | R1 remedy 2 |
| derived `retrieval_z_window` | 3 spacings | largest gap 100 um |

**Both axes are `target_unreachable`, and `0 iterations` is the informative part**: the objective's
maximum over the log grid already sits below the target, so the bisector has nothing to bisect
towards. It is not a search that ran out of budget; it is a target outside the reachable set.

`ell_z = 25.0 um` is the bracket's **lower** endpoint (`calibration_ell_z_min_factor = 0.25` x the
100 um median spacing), not its upper one — and `specs/09` §2 explicitly requires an unreachable
target to return the grid maximiser and **not** an endpoint. An endpoint coming back is the tell.

**Diagnosis: the objective is not backwards, it is exactly constant.** Three sweeps, run to settle
whether this is a defect in the objective or a real limit of a 400 um stack:

| sweep | 25 | 60 | 137 | 200 | 275 | 364.6 um |
|---|---|---|---|---|---|---|
| GRF field, r between two planes 100 um apart | −0.005 | 0.202 | 0.654 | 0.794 | 0.874 | **0.921** |
| generated sections, `expr_mode="zinb-flow"` | 0.887 | 0.913 | 0.919 | 0.924 | 0.927 | **0.932** |
| generated sections, `expr_mode="cross-mix"` | 0.6728 | 0.6728 | 0.6728 | 0.6728 | 0.6728 | 0.6728 |

*Neither* candidate explanation survives. The **field** is correct: its between-plane correlation
rises strictly and monotonically over the bracket, and `with_lengthscale` — the rescale path the
calibrator uses, which carries the draws over rather than redrawing — reproduces it value for value.
The **stack** is adequate: on this same 6-section, 400 um volume the whole generation pipeline
inherits that monotonicity under `zinb-flow`, reaching 0.932 against an observed target of 0.918, so
100 um spacing resolves this correlation length perfectly well. What is actually happening is that
under the **selected** `expr_mode = "cross-mix"` the objective is identical to **ten decimal
places** (0.6727617248) across a 15x sweep. `_cross_mix` copies each emitted count verbatim from a
donor cell and never evaluates the flow — the donor weights come from the retrieval score, not from
the latent — so the GRF is absent from the expression path entirely and `ell_z` cannot move a single
count. A constant function has no maximiser; the grid argmax is whichever point ties first, which is
the endpoint.

That is the same failure as `prior_mode = "iid"`, one stage further down the pipeline, and the guard
now covers both: `calibrate_lengthscale` and `calibrate_ell_z` refuse `expr_mode="cross-mix"` as
well as a non-GRF prior (`test_calibrator_refuses_an_expression_path_that_ignores_ell`, which also
asserts `zinb-flow` and `auto-blend` are *not* refused — under `auto-blend` the fitted `w(v)` is 0
everywhere on this fixture, so the flow's draw passes through unblended and `ell` acts in full).
**So the 25.0 um and its `target_unreachable` status are artefacts of a flat objective, not
measurements of this tissue.** The honest reading is that `ell_z` was never measurable on the
selected config — not that it is small. Correcting an earlier claim in this entry's first draft: I
described the response as falling with `ell_z`; it does not fall, it does not move at all.

R1's remedy 3 is untouched by any of this: the variogram's own `ell_z` (364.6 um here, 561 um on the
gate fixture against a 200 um truth) enters only as the bracket's upper endpoint and is never
returned as a value, so the 561 um over-fit still cannot propagate into a shipped number.

**C30 is closed: the writer exists, and only a converged axis gets through it.** The gap the
diagnosis found was that nothing applied a `LengthscaleCalibration` at all —
`generate_section`'s `calibration=` argument is the *detection* calibration, and `ell` reaches the
prior only as `cfg.ell_xy` / `cfg.ell_z`. `apply_lengthscale(cfg, calibration) -> Config` is now the
only sanctioned route, `specs/09` §2 names it, and it applies **only** a `"converged"` axis:
`target_unreachable` and `boundary` are dropped with a `CalibrationNotAppliedWarning` naming the
achieved and target values, and the config's own value stands. The two axes are decided separately
on `status` and `ell_z_status`, because an in-plane result that converged is not made worthless by a
stack too short to constrain `ell_z` — which is exactly the fixture's case, below.

#### 6b. The live arm — the first real measurement of the anisotropy

With both severing gates put back (`prior_mode="correlated"`, `expr_mode="zinb-flow"`) at the
selected 2400 steps:

| quantity | value | status | applied? |
|---|---|---|---|
| `ell_xy` | **86.4 um** | `converged`, 2 iterations | **yes**, `Config.ell_xy` 100 → 86.4 um |
| `ell_z` | 364.6 um | `target_unreachable` | **no**, dropped; `Config.ell_z` stays 100 um |
| Moran's I | gen **0.4051** vs flanking **0.4102** | gap 0.0051 against a 0.02 tolerance | |
| between-section r | gen **0.8706** vs observed **0.9182** | R1 remedy 2 | |

`ell_xy` converges in two iterations to 86.4 um, **13.6 % below** the variogram's 136.8 um and in
the direction T03 predicted for a window-biased fit. It is written through to the prior; the arm's
per-module diagnostic under it reads |I_gen − I_real| = 0.0585 / 0.0577 / 0.0586 / 0.0491 — flat
across modules and now slightly *over*-shooting, which is what a global `ell` tuned to the mean
should look like, and evidence **against** the per-channel-group escalation. On the dead `cross-mix`
arm the same table read 0.1241 / 0.1056 / 0.1467 / 0.1298: uniformly worse, and uniformly.

**`ell_z` fails upward, and that inverts a claim this entry made twice.** The objective is monotone
increasing, so the search terminates at the bracket's **top** — 364.6 um, the variogram fit — and
still undershoots. What the data support is `ell_z >= 364.6 um`: a **lower** bound on the tissue,
not an upper one. The "upper bound" phrasing inherited from R1 described where the *search* stopped,
not what the *parameter* can be, and is corrected in the code, the warning text and the report.
Remedy 3's actual protection is unaffected and is doing its job: the fit never becomes a value
because `apply_lengthscale` drops it.

**Why remedy 2's target may be unreachable by construction.** The objective generates both sections
with **both excluded from retrieval**, so their correlation comes from the field alone — without
that exclusion a shared donor pool would correlate them at any `ell_z` and the curve would be flat,
which is the trap the whole calibrator is built around. But real adjacent sections draw much of
their 0.9182 from precisely that shared anatomy. Target and objective are not measuring the same
quantity, so there need not exist an `ell_z` that closes the gap. That is a question for the spec
owner, and it is why **R1 stays open on the measurement even though every mechanism it asked for now
works**.

**The anisotropy the oblique claim depends on**, stated as measured: `ell_z / ell_xy >= 364.6 / 86.4
= 4.2`, against the fixture's generative truth of 200/120 = 1.7 and the variogram's own 2.7. All
three agree the field is elongated along z by at least a factor of two, which is what makes an
oblique cut a different sampling problem from an axis-aligned one. The bound form is the honest one:
this stack constrains the ratio from below, not from above.

**And the live arm changes the definition of done.** Method vs the two fallbacks on the six metrics:
0.9386 / 0.9067 / 0.9211 / −0.0467 / 0.0350 / −0.0502, against `resample` at 0.8278 / 0.7683 /
0.6844 / −0.0374 / 0.0195 / −0.0660 and the independent-donor sampler at 0.8694 / 0.8347 / 0.7256 /
−0.0373 / 0.0255 / −0.0502. That is **5 of 6 against `resample` and 4 of 6 against the donor
baseline**, where the selected `cross-mix` config managed 5 of 6 and only **2 of 6**. The three
distribution-level metrics the method loses as selected are the three it wins by the widest margin
once the flow is on. The definition of done is met on the configuration that exercises the method
and not on the one the selector shipped — which is R8 with a price attached rather than a separate
finding.

**Calibration headroom is a property of the `prior_mode` x `expr_mode` pair, not of `ell` alone** —
that is R8, and it is the honest answer to "why is R1 still open after T09": the calibrators are
correct, and now refuse the configurations where they would be meaningless, but the configuration
this fixture selects leaves them no mechanism to act through.

**The per-module diagnostic (A2)** — one global `ell`, per-module agreement reported and *not*
targeted — shows the deficit is **global rather than per-module**: |I_gen − I_real| is 0.1241 /
0.1056 / 0.1467 / 0.1298 across the four gene modules (55 / 55 / 54 / 36 genes), all in the same
direction. There is no module the single `ell` serves badly and the others well, so this is not
evidence for the per-channel-group escalation `specs/09` §2 describes; it is evidence that the
generated section is uniformly under-structured.

**The derived retrieval window** on this holdout is **3** spacings (largest surviving gap 100 um =
1 spacing, so the configured floor binds). The gate the derivation exists for is the
`consecutive-3` holdout, where the same code returns **4** — see §1.

#### 7. Open risk R3, and a new one the selector raised

**R3 — the boundary.** The uncertainty gate *does* notice it. Mean latent variance over
`n_uncertainty_samples` GRF realisations, on real cells with each section excluded from its own
retrieval pool: **first 0.04762, middle 0.04207, last 0.04576** — the ends run **+13.2 %** and
**+8.8 %** above the interior. `specs/09` §1 asked for this to be measured and said that a gate
which does *not* elevate there would itself be a finding; it elevates, in the right direction,
by roughly a tenth. Every emitted section also carries `uns["boundary"]`, so a caller cannot
mistake the regime.

The `prior_mode = "correlated"` calibration arm, however, **inverts the sign**: first 0.78378,
middle 0.84673, last 0.82079, i.e. the ends run **7.4 %** and **3.1 %** *below* the interior, with
the absolute variance ~20x larger because the prior is itself correlated. So "the gate elevates at
the boundary" is true of the shipped `iid` configuration and *not* a fixed property of the gate.
Recorded as such in the report rather than generalised: R3 is answered for what ships, and the
`correlated` regime needs its own measurement before anyone leans on it.

**New: the reduced-budget heuristic is not safe for `prior_mode`.** `specs/09` §3 exempts the
budget gate from the 25 % reduction because a reduced fit of a `2x` candidate *is* the `1x`
candidate. The fixture shows the exemption is drawn too narrowly: at 600 steps `iid` beats
`correlated` (rank 1.0 vs 2.0), while GATE 1 established that the correlated prior is what makes
Moran's I of the generated section track the tissue's at all — a mechanism that needs training to
express. The selector is choosing a *prior* on a quarter-budget model and applying it to a
full-budget one. This is the same class of error as the budget gate's, one gate over, and it is
recorded rather than patched: fixing it means either scoring `prior_mode` at its own budget (four
more fits) or folding it into the joint gate, and both are design changes for the spec's owner.
The immediately available evidence against the choice is in this run: the joint gate's cells were
all fitted with `prior_mode = correlated` at 1200/2400 steps and reached morans_pearson **0.9368**,
where the 600-step `iid` winner reaches **0.6494**.

#### 8. Definition of done

`specs/09`: *LOSO reconstruction beats both `resample`-mode and the independent-donor baseline on
≥ 4 of the 6 target metrics.* Measured on the same three folds, same metric code, at the selected
config:

| arm | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| method (selected) | 0.8329 | 0.7527 | 0.7028 | **−0.0293** | **0.0254** | **−0.0480** |
| `resample` + `cross-mix` (v20) | 0.8278 | **0.7683** | 0.6844 | −0.0374 | 0.0195 | −0.0660 |
| independent-donor baseline | **0.8454** | **0.8218** | **0.7358** | −0.0345 | 0.0218 | **−0.0480** |

**Half met, and the half that fails is the one that matters.** Against `resample` the method wins
**5 of 6** (all but Geary's). Against the **independent-donor baseline** it wins **2 of 6** with one
tie — the baseline leads on all three correlation-style metrics. The reason is the one T06 already
recorded: that baseline emits **real counts** drawn from real donors, so per-gene autocorrelation
and a shared-embedding mixing score are exactly where it is strongest and where a generative head
has the least to add. `reports/config_selection_synthetic.md` carries the table; T10's A2/A3 re-score
it with the vendored metric code, and the paper cannot claim the definition of done as met on this
fixture.

**The anchor gate switched itself off.** `w(v)` came back **0.0 in every variance bin** (bin centres
0.0278 … 0.0694): at the selected config the flow's own draw beat every blend with the
retrieval-anchored profile on the per-cell profile correlation, at every level of uncertainty. The
map is trivially non-increasing, `auto-blend` therefore reduces to `zinb-flow`, and this is
consistent with the selector's own choice of `cross-mix` over `auto-blend` one section up.

#### 9. `text_emb_mode` was a dead gate

`Config.text_emb_mode` has existed since T01, `specs/09` §3 lists it as one of the four gates and
`specs/10`'s A3 ablates it — and nothing read it. The selector would have scored four identical
candidates and reported a decision it had not made. Implemented once, in
`TextGroundedEmbedding._text_channel`: `"lookup"` zeroes the projected text vector on the seen and
the zero-shot path alike and pins the residual gate at 1. SPEC_QUESTIONS **C25**; T10's A3 uses the
same switch.

#### 10. The six selection metrics are T10's names on T08's kernels

`specs/09` §3 scores candidates on the six target metrics; `specs/10` requires those to be vendored
verbatim from `bench3/evaluate_paper.py`, and that module is T10's. `train/select.py` computes the
six under T10's names with T08's kernels, plus a PCA-space kNN mixing score in place of the UMAP
one (a stochastic embedding inside a selector that must be reproducible from a seed is not
something to add). The substitution is printed in every report the selector writes, and T10
re-scores the selected config with the vendored code — `Scorer` is a protocol, so the swap is one
line. SPEC_QUESTIONS **C26**.

#### 11. R8 resolved — the training-free-option rule, and a new selected config

`specs/09` §3 now carries the **rule**, not the patch: *a gate with a training-free option cannot
be scored at a reduced budget.* If an option reaches its final behaviour without training — because
it copies real data rather than generating it — it is at full strength at any budget while its
rivals are not, so a reduced-budget comparison measures the budget. Qualifying gates are scored at
the selected budget, and **jointly** when more than one qualifies, because their errors compound
through coordinate descent's ordering.

The rule is enforced, not just written: `TRAINING_FREE_OPTIONS` classifies every gate and
`_check_gate_classification` raises at import for a gate that is unclassified or that names an
option it does not have. An empty tuple is the positive statement "all options train". The two gate
sets are **derived** from that classification rather than hand-maintained, so they cannot drift
apart from the rule. On the current table it merges `layout_mode` x `prior_mode` x `expr_mode` into
one **18-cell** gate at full budget; `text_emb_mode` keeps coordinate descent.

**The new selected configuration** — 23 fits, 21 330 s of selection:

| gate | old | **new** |
|---|---|---|
| `layout_mode` | field | **resample** |
| `prior_mode` | iid | **correlated** |
| `expr_mode` | cross-mix | **zinb-flow** |
| `text_emb_mode` | medcpt | **lookup** |
| `train_steps` | 2400 | 2400 |
| `w_autocorr` / `w_profile` / `w_distribution` | 0.5 | 0.5 |
| config hash | `fe49ea9f8ad54bb2` | **`00ef4a19a2f576b8`** |

**Every one of the three merged gates changed its answer**, and the 18-cell table shows why: all
six `cross-mix` cells occupy the **bottom six ranks** (13.0–17.0) and every non-`cross-mix` cell
beats them. At a quarter budget `cross-mix` won this gate. The separation is total — there is no
overlap between the two groups on any of the three distribution-level metrics.

`layout_mode` earned its place in the merged gate. It was the leg of the criterion with no observed
reversal, included because the rule is about whether a comparison is sound rather than whether it
happened to come out wrong — and it **changed the answer**, from `field` to `resample` (ranks 3.0
vs 7.5). Scoping the fix to the two gates that had visibly failed would have left it wrong.

The winner ties with `resample + correlated + auto-blend` at rank 3.0, and the tie is not
arbitrary: on this fixture the fitted `w(v)` is 0 at every knot, so `auto-blend` passes the flow's
draw through unblended and the two cells are **bit-identical**. `min()` takes the first, which is
`zinb-flow`; either label denotes the same model.

**Definition of done, on the new config** — and `reports/config_selection_synthetic.md`'s previous
numbers are **superseded**, not merely updated:

| arm | morans | gearys | umap_mixing | field_r | depth_r | ct_loc |
|---|---|---|---|---|---|---|
| **method (new)** | **0.9517** | **0.9142** | **0.9730** | −0.0425 | **0.0519** | −0.0660 |
| resample (v20 fallback) | 0.8278 | 0.7683 | 0.6844 | −0.0374 | 0.0195 | −0.0660 |
| independent donor | 0.8716 | 0.8273 | 0.7388 | −0.0404 | 0.0405 | −0.0660 |

The method beats `resample` on 4 of 6, ties `celltype_localization` and loses `marker_field_r`; and
it beats the **independent-donor sampler on 4 of 6** with the same tie and the same single loss.
The old config managed **2 of 6** against the donor bar. The three distribution-level statistics it
used to lose are now won by 0.08, 0.09 and 0.23. **The definition of done goes from half met to met
on four of six against both fallbacks**, and it is the selector's own choice that did it.

`ell_xy` still calibrates to **86.4 um, `converged`** on the selected config, and `ell_z` is still
`target_unreachable` at the bracket's top — R1 is unchanged by the new config, as expected, since
its obstacle is the target rather than the gates.

**A residual the run surfaced, not covered by the rule as written.** `text_emb_mode` passes the
rule — both options train — but it is scored at 600 steps *under a `zinb-flow` incumbent*, where
the flow is far from converged (morans 0.5997 / 0.6523 against 0.96 at full budget), and its winner
changed from `medcpt` to `lookup`. Both options are handicapped equally, so this is not the R8
defect; but the gate is being decided on a model that behaves nothing like the shipped one. Whether
the rule should be widened from "has a training-free option" to "is scored at a budget where the
incumbent is unconverged" is a spec question, recorded rather than answered here.

**T10 benchmarks the new config** (`00ef4a19a2f576b8`, persisted as
`reports/config_selection_synthetic.yaml`), not the old one. Every headline number T10 reports —
and every arm of its A2 ablation — must be produced under it.


#### 12. R9 closed — the rule widened, and a gate that turns out to be undecidable

`specs/09` §3's rule now has **two conditions**, and a gate is scored at the selected budget when
either holds: (1) it has a training-free option, or (2) **the incumbent is unconverged at the
reduced budget**. Condition (2) cannot be declared in advance — it depends on the incumbent the
search arrives at — so it is *measured*: `incumbent_is_unconverged` compares the incumbent's own
score at the two budgets and escalates every remaining gate when at least
`selection_convergence_min_metrics` (2) of the six fall by more than `selection_convergence_tol`
(0.05). The report says which rule escalated what, and two tests pin both directions — a
budget-sensitive incumbent escalates, a flat one keeps the cheap descent, or the reduced budget
would be dead code.

**It fires on this run's own numbers**, no new fitting required: the incumbent scores 0.9606 /
0.9308 / 0.9744 at 2400 and 0.5997 / 0.5048 / 0.1649 at 600 — shortfalls of 0.36, 0.43 and 0.81
against a 0.05 tolerance.

**Re-scored at 2400, `lookup` still wins — 1.2 against 1.8 — so the winner does not flip back.**
But the gate stops being decidable in the process. At 600 steps `lookup` led by 0.053 on
`morans_pearson`; at 2400 the two split the metrics 3–2 with one tie, and no metric separates them
by more than 0.011:

| text_emb_mode | morans | gearys | umap_mixing | field_r | depth_r | ct_loc | rank |
|---|---|---|---|---|---|---|---|
| medcpt | 0.9535 | 0.9288 | 0.9624 | −0.0469 | **0.0570** | −0.0660 | 1.8 |
| **lookup** | 0.9511 | **0.9334** | **0.9688** | **−0.0425** | 0.0460 | −0.0660 | **1.2** |

**And the margin is inside the reproducibility envelope — the finding that matters most here.**
`medcpt` at 2400 was fitted twice, same config, same seed, different process, and moved by up to
**0.0120** (`umap_mixing`). The largest difference between the two *options* is **0.0110**. Re-running
the identical configuration moves the score by as much as changing the gate does. So this gate is
not resolved at one seed at any budget: the reduced budget made it look decided in the wrong
direction, and the selected budget makes it visibly undecided. The cause is **not established** —
both fits take explicit seeds, and nondeterministic float reduction under different thread
scheduling is the obvious suspect but was not confirmed. Recorded as **R10**, and it is not local to
this gate: every number T10 reports inherits the same envelope.

`text_emb_mode = "lookup"` therefore stands as selected, on rank, with nothing in the measurement
overturning it — while disabling the MedCPT channel on a margin smaller than the noise floor. Two
proposals go to the spec's owner rather than being taken here: a **tie-break rule** preferring the
capability-preserving option when the separation is below the reproducibility envelope, and
**repeated seeds** for any gate that reaches a headline claim.

The selected config's **hash moves to `00ef4a19a2f576b8`** — the gate choices are unchanged, but
`Config` gained the two fields condition (2) needs, and the hash covers every field. `T10 benchmarks
`00ef4a19a2f576b8`.


#### 13. R10 settled — the envelope measured, and two gate choices were inside it

Both remedies are in, as rules rather than as fixes for the gates that exposed them.

**The repeated-seed rule** (`specs/09` §3, scoped in `specs/10` §3): any measurement reaching a
paper claim runs `Config.claim_min_seeds` (3) seeds and reports the spread. Scoped so it attaches to
*claims*, not measurements — the headline table, the claim-bearing ablation arms, E1–E5 and the
boundary stratification pay for three seeds; ceilings, diagnostics, calibration statuses and the
selection itself stay at one. `specs/10` now requires T10 to carry `CLAIM_BEARING` and a
`_check_claim_coverage` that refuses a headline table containing an unclassified measurement — the
same derived enforcement the budget rule has. **Estimated bill: ~2.4x a single-seed campaign**, not
3x, because the diagnostics and ceilings are the long tail and stay at one seed.

**The capability tie-break** needs a claim *level*, not a flag, because "prefer the capability"
pulls in opposite directions in the two cases measured. `CAPABILITY_CLAIM` gives every option an
integer, and `capability_tie_break` applies two rules: an exactly-identical rival proves a higher
claim is **inert** and drops it, then among what survives the highest claim wins. `lookup` (0) vs
`medcpt` (1) differ but are inside the envelope, so `medcpt` wins; `auto-blend` (2) is *identical*
to `zinb-flow` (1), so it is dropped and `zinb-flow` is the honest label. One bug worth recording:
the first implementation returned early when the rank winner was capable, which credited
`auto-blend` whenever it happened to sort first — equal ranks make the order arbitrary, and the
test now asserts both orderings.

**The envelope, measured** (`reports/envelope_synthetic.md`): 3 cells x 3 seeds, 9 fits, 3.4 h.
Largest across-seed spread **0.0335** — **nearly 3x the 0.0120 that two fits had suggested**. It is
**not score-dependent** (0.0299 / 0.0335 / 0.0270 at score levels 0.95 / 0.94 / 0.83, which is what
the far arm was for) but it is strongly **metric-dependent**: `celltype_localization` reproduces to
0.0068 while `gearys_pearson` moves 0.0335 — a 5x range. `claim_tie_break_envelope` is set to
**0.04**, rounded up, because a maximum over nine samples is a lower bound on the true spread.

**And the re-check found a gate nobody suspected.** Against 0.04:

| gate | selected | closest rival | margin | verdict |
|---|---|---|---|---|
| `layout_mode` | resample | **hybrid** | **0.0344** | **inside** |
| `prior_mode` | correlated | iid | 0.0731 | safe (1.8x) |
| `expr_mode` | zinb-flow | cross-mix | 0.2900 | safe (7.3x) |
| `text_emb_mode` | lookup | **medcpt** | **0.0110** | **inside** |

> ⚠️ **REVISIT — flagged by T10's pilot (2026-08-21), open risk R11.** This tie-break shipped
> `hybrid` on a within-noise margin, and real-data evidence now exists that points the other way and
> is **not** inside any envelope. On tier-1 STARmap, holding everything else fixed and swapping only
> `layout_mode`: `celltype_localization` **0.4252** for `field` against **0.7008** for `resample`,
> where the model-free `flanking_copy` floor is **0.7765** — the learned layout scores *below the
> floor* on the metric it exists to win, by ~29x the across-seed envelope. `cell_count_ratio` moves
> 0.83 -> 0.99 in the same swap, and at 2400 steps the field layout gets *less* stable, emitting 146
> cells for a section whose ground truth has 4 102.
>
> Read against the table below, the fixture and the real data agree in direction: **`resample` was
> the rank winner here too (3.0), `hybrid` followed (4.2), and `field`'s best cell ranked 7.0 — it
> won nothing.** The fixture result was underpowered, not wrong.
>
> **The decision to revisit is not "which of the three ships" but "should this gate be inherited at
> all".** `specs/09` §3 selects per dataset; this tie-break was made on the fixture and then carried
> forward. It should be re-selected on each dataset with real-data evidence rather than inherited,
> and `specs/05`'s headline layout claim re-examined (`specs/10` §4.5b). T10 measures; it does not
> retune the layout head.
>
> ### ✅ CLOSED 2026-08-25 — the tie-break is overturned; `resample` ships
>
> Re-measured on the corrected grid sampler, five arms off one refit
> (`reports/r11_starmap_layout_modes.md`): `field` **0.6607**, `hybrid` **0.6692**, `resample`
> **0.7546**, copy floor **0.7765**. The capability tie-break that shipped `hybrid` is reversed —
> not by a re-weighting but by a measurement that is 3.2x the envelope where the original was
> 1.03x it — and `Config.layout_mode` now defaults to `"resample"`. The count is what decides it:
> `field`/`hybrid` swing 3.7x between refits of one configuration on the section-2 integral, which
> no density-matched score can see.
>
> **Both questions this note raised are answered.** *Which of the three ships*: `resample`, on real
> data, as a recorded negative result (`specs/05` §4a). *Whether the gate may be inherited from the
> fixture at all*: **no** — and the reason is now specific rather than statistical. The fixture's
> flanking baseline sits at 58 % of its ceiling where real tissue's sits at 79 %, so the fixture
> systematically over-rewards a generative layout no matter how many seeds it is given
> (`progress/fixture_limitations.md` §2). The fixture gate was nonetheless re-run on the corrected
> sampler so its verdict is on the record beside the real-data one:
> `reports/t09_layout_mode_gate_grid.md`.
>
> One thing that re-run established about **this gate's cost**: `layout_mode` is read only at
> generation time, and fitting the fixture at all three modes with one seed gives **bitwise
> identical** weights across all 96 parameter and buffer tensors. The merged 18-cell gate therefore
> refits 3x more than it needs to for this leg — 6 fits would serve 18 cells — and a future
> selection run should reuse one fit across the `layout_mode` axis rather than refitting per cell.
> That is a cost fix, not a correctness one: the scores are the same either way, but the contrast
> is cleaner because it carries no fit-to-fit noise.

**`layout_mode` was decided inside the noise** — 0.0344 against an envelope of 0.0335, a margin
1.03x the noise floor. The tie-break selects **`hybrid`**, because `resample` reuses real cell
positions and is the v20 fallback, so shipping it switches the learned continuous layout off. This
is borderline and is flagged as such in the report: at the raw 0.0335 the margin clears by 3 %, and
only the rounding puts it inside. The rounding is justified in the unsafe direction — 0.0335 is a
maximum over nine fits, and the same statistic at two fits read 0.0120.

**Shipped after both tie-breaks:** `hybrid` + `correlated` + `zinb-flow` + `medcpt`, 2400 steps,
weights 0.5 — hash **`00ef4a19a2f576b8`**. Two of the four gates are now decided by capability
rather than by measurement, and the report says so per gate, which is the point of the rule.

**What this leaves T10.** The definition-of-done arms are superseded twice over — wrong config, and
one seed. A single-seed replacement would not be admissible under the rule this task just wrote, so
they are **not** re-run here: T10 produces them under `00ef4a19a2f576b8` at three seeds, reports
min–max beside every median, and treats any effect below the campaign's envelope as a tie.

---

### T09 on real data — the machinery for a STARmap tier-1 run (2026-08-26)

Everything above was measured on the synthetic fixture. `specs/09` §3 says the configuration is
selected **per dataset**, and `progress/fixture_limitations.md` §2 established that at least one
gate cannot be decided on the fixture at all. This entry is the run being *prepared* on tier-1
STARmap — the code, the tests and the campaign plan. **No campaign has been run**; every number
below is from the synthetic fixture, the recorded reports, or a bench3-shaped smoke dataset built
for the purpose. Nothing here is a STARmap measurement.

#### 1. The finding that made the run necessary, and that no test would have caught

**`text_emb_mode = "medcpt"` has never been exercised on real data, and could not have been.**
Five callers built entity embeddings and every one of them made the text channel dead or hollow:

| caller | what it passed as text vectors | effect |
|---|---|---|
| `scripts/t10_chain_diagnostic.build_embeddings` | **zeros** | `W t = 0`; the channel is off |
| `scripts/t10_rescore_saved` (imports the above) | **zeros** | same |
| bench3's `run_spatialcpav25_gen.build_embeddings`, `medcpt` arm | `gene_descriptor(symbol, **None**)` | encodes the string `"Slc17a7."` — MedCPT applied to a token |
| the same, `lookup` arm | **zeros** | the two A3 arms differ in *two* things, not one |
| `scripts/t09_layout_mode_gate`, `scripts/t06_expression_report` | fixture stand-ins | fine there, not a real-data path |

So `progress/numbers.md`'s "**Still ablation A3**: `text_emb_mode=lookup` (MedCPT 403'd in the
container)" understates it: even with the encoder reachable, the wrapper would have encoded bare
symbols and reported it as `medcpt`. **The panel's own table has the content all along** —
`resources/gene_meta.parquet` covers **28/28** STARmap symbols and **28/28 carry a summary**, so
`gene_descriptor` yields e.g. *"Slc17a7. solute carrier family 17 …, member 7. Human orthologue
SLC17A7: The protein encoded by this gene is a vesicle-bound, sodium-dependent phosphate
transporter…"* against the bare `"Slc17a7."` the wrapper was sending.

`model/embeddings.py::build_entity_embeddings` is now the single builder: descriptors from
`Config.gene_meta_path` through `TextEncoder`, refusing a missing table rather than degrading a
whole panel to bare symbols. **Both arms of the gate get the same vectors** — `"lookup"` is applied
inside `TextGroundedEmbedding._text_channel`, so withholding the vectors as well made A3 differ in
two things at once and changed `distillation_loss`, which reads `text_vecs` directly.
`describe_entity_descriptors` reports `n_bare` so a run that says `medcpt` with nothing behind it
says so in its own provenance line. `t10_chain_diagnostic` is deliberately **not** switched over —
that would move `reports/chain_2400*.md` — but it now prints `text channel: ZERO VECTORS` on every
run rather than leaving the fact in a docstring.

#### 2. Four defects that would each have stopped or spoiled the run

* **Selection through the bench3 wrapper was unreachable.** `run_selection_for` passed neither a
  scorer nor an embeddings factory, and `run_selection` refuses both-absent by design — so every
  `--select-only` invocation raised `SelectionError` at the first line of the search. That is why
  no per-dataset selection had ever run on any dataset. Fixed, along with the base config's obs
  keys, which were `Config`'s defaults rather than bench3's.
* **`Config.expr_pca_dim = 32` exceeds STARmap's 28-gene panel**, so `validate_config_against_volume`
  refuses every fit. This was a recorded **owed fix** (`specs/10` §0, `specs/11`, `progress/decisions.md`)
  and it is now the rule the spec asked for: `data/schema.py::clamp_config_to_volume` narrows
  `expr_pca_dim` and `retrieval_k` to the volume's own size, warns with both numbers because the
  clamped value lands in the content hash, and `run_selection` applies it. **The rule gives 28 on
  tier 1**; `reports/pilot.md`'s 16 was a hand-picked stand-in, so anything meant to be comparable
  with the pilot's recorded row must pass `--expr-pca-dim 16` explicitly. The drivers apply the same
  rule from the input file's *header*, because building a volume to clamp against runs the very
  check the clamp exists to satisfy.
* **`fit_repulsion` was being paid for where nothing could read it.** `sample_layout` returns
  `_resample_layout` before it looks at the interaction, so under the shipped
  `layout_mode="resample"` the fit is dead work — and worse than dead: it raises `LayoutError` on a
  point pattern with no soft-repulsion range and would abort a whole search over a quantity it is
  not using. `FitScorer.needs_repulsion` is set by `run_selection` from
  `repulsion_is_reachable(cfg, pinned)`. It is a property of the **search**, not of the candidate,
  because the fit cache normalises `layout_mode` out — deciding per candidate would make the shared
  model depend on which cell was fitted first.
* **One under-trained candidate would have killed a nine-hour search.** `assert_detection_rate`'s
  band is measured against the training sections, and STARmap's median per-gene detection is
  **0.9999** (`specs/10` §0) — so a reduced-budget candidate genuinely misses it, and the reduced-budget
  cells of any selection *are* under-trained. It was observed on the smoke dataset: at 60 steps the
  emitted rate was 0.47 against 1.0000 and the process died. Such a candidate now **ranks last**,
  and does so loudly: `SCORING_FAILURES` is deliberately narrow (`ExpressionError` only — a shape
  error or a leakage refusal still aborts), the failure warns, is carried in
  `SelectionResult.failures`, is printed in the report, survives a resume through the checkpoint,
  and **a gate every option of which failed is refused outright** rather than shipping the first
  label.

#### 3. Two mechanisms `specs/09` §3 describes and nothing implemented

* **The capability tie-break was never wired into the search.** `capability_tie_break` existed and
  was tested, but `run_selection` took `min(rank)` at every gate and §13's shipped configuration was
  produced by applying the rule *by hand* to a printed table. A persisted `selected.yaml` would
  therefore have carried the rank winner while the record said the tie-break decided two of the four
  gates. `review_gates` now re-checks every gate after the search, on the candidates that differ
  from the selected config **in that gate alone**, and the result applies and reports it.

  Two bugs the smoke run found in that reviewer, both of which produced a *plausible* report:
  **(a)** median rank is not consistent under subsetting — a merged 6-cell gate and a 2-cell slice
  of it named different winners, so the review printed `prior_mode ships correlated` beside a
  selected config saying `iid`. The review is now anchored on the option that actually ships, and
  answers only the question the rule poses: is that option's margin larger than the envelope.
  **(b)** a candidate that failed in a `--prewarm` shard was read back from the checkpoint by the
  final process, which then reported "every candidate scored" about a table containing a failure.

* **Gates can now be pinned.** `run_selection(..., pinned={"layout_mode": "resample"},
  pinned_reason=...)` fixes a gate in the base config, drops it from the merged gate's product and
  from coordinate descent, and reports it as pinned. A reason is **required**: a gate removed from
  selection without one is indistinguishable in the report from a gate that was never a gate.
  Refusals are Convention 6's — an unknown gate, an option that gate does not have, or pinning
  everything.

  ⚠️ **Correcting the rationale, not the decision.** Pinning `layout_mode` was asked for on the
  grounds that "layout_mode does not affect the fitted weights". That fact is real
  (`FIT_INVARIANT_GATES`) but it argues the *other* way: because the gate is fit-invariant, six fits
  already served the merged gate's 18 cells, so pinning it removes **12 LOSO scorings and zero
  fits**. `test_pinning_costs_scorings_and_not_fits` asserts exactly that. The good reason to pin it
  is the one recorded at R11: real data settled the gate at 3.2x the envelope, and re-opening it at
  one seed inside a search whose own margins are envelope-sized can only lose to noise.

#### 4. What was built

| file | what |
|---|---|
| `model/embeddings.py` | `build_entity_embeddings`, `describe_entity_descriptors` |
| `data/schema.py` | `clamp_config_to_volume`, `ConfigClampWarning` |
| `train/select.py` | `pinned` / `pinned_reason`, `full_budget_gates`, `descent_gates`, `GateReview`, `review_gates`, `rank_candidates`, `repulsion_is_reachable`, `SCORING_FAILURES`, `CandidateFailedWarning`, `SelectionResult.{reviews, failures, pinned}` |
| `scripts/_starmap_run.py` | the shared real-data runtime: volume, base config, embeddings factory, header-level clamp, encoder preflight |
| `scripts/t09_select_starmap.py` | the selection driver — `--preflight`, `--prewarm {joint,full-budget} --index K --of N`, `--run`, `--pin` |
| `scripts/t09_ship_starmap.py` | one shipped-config fit, the whole `specs/09` §2 chain, generation, and the six metrics against `oracle` and `flanking_copy` |
| bench3's `run_spatialcpav25_gen.py` | the selection call fixed, real descriptors, `--pin-gate` |

**21 new tests** (362 in the fast suite, 2 min 11 s on CPU), `make lint` and `make typecheck` clean.

#### 5. The campaign plan, and what it rests on

`run_selection` is sequential — the merged gate needs the joint gate's budget, coordinate descent
needs the merged gate's incumbent — but the cells *within* a stage are independent and `ScoreCache`
is the seam. `--prewarm` scores one shard of one stage and flushes it; the final `--run` finds them
cached and issues only the fits that genuinely depend on a prior decision.

With `layout_mode` pinned, the search is **11 fits**: 4 for the joint gate (2 at 1x, 2 at 2x), 5 new
for the merged `prior_mode` x `expr_mode` gate at the selected budget (the sixth cell is the joint
winner's own config and is a cache hit), 1 convergence probe at the reduced budget, and 1 for
`text_emb_mode`'s rival. At the pilot's measured **28.5 min per 1200-step fit** that is ~8.8 h
serial and **~3 h** in three parallel waves of at most 6.

**Two things about tier 1 that constrain the calibration, established here rather than assumed.**
The training stack is **four** sections (1/3/5/7 at z = 19/41/63/85), giving exactly **three**
distinct z lags — the bare minimum for `_fit_matern_variogram`'s nugget, sill and length-scale. At
~4176 cells per section every one of the 8x8 grid cells clears
`variogram_z_min_cells_per_cell = 5` (~65 cells each), so all three lags survive
`variogram_min_pairs_per_bin = 32`; the margin is real but not large, and `ell_z` on tier 1 will be
weakly constrained whatever it returns. That is R1's obstacle appearing on real data for the first
time. The ship driver therefore treats `VariogramError` and `CalibrationError` the same way — the
fit is already paid for, so `ell` is recorded as refused with the reason verbatim and the run
continues.

#### 6. What is verified, and what is not

Verified end to end on a **bench3-shaped smoke dataset** (7 sections, 12 genes, 4900 cells, its own
`paper_protocol`, a matching gene-meta table and a pre-seeded text cache): preflight; four
concurrent joint-gate shards; six concurrent merged-gate shards; the final `--run` resuming all ten
from the checkpoint and issuing only the two dependent fits; the pinned gate and the tie-break
review in the report; a candidate failing and ranking last without killing its stage; a fit killed
mid-way and resumed from `fit_seed1.pt`; the whole `specs/09` §2 chain including
`calibrate_lengthscale` **refusing** `prior_mode="iid"` and being recorded as refused; both
`apply_lengthscale` axes dropped as `target_unreachable` with both numbers named; the anchor `w(v)`
fitted under `auto-blend` and skipped with a reason otherwise; generation of the three targets at
raw and ground-truth-matched density; and the six-metric table scored on `bench3.evaluate_paper`
against `oracle` and `flanking_copy` produced by `bench3.selftest`.

**Not verified, and it cannot be here:** that the MedCPT weights load. `huggingface.co` is 403 from
this container, so the smoke run used a pre-seeded text cache. `--preflight` is exactly that check
and costs seconds; it must pass on the campaign machine before anything is fitted.

**No STARmap number is reported in this entry.** The selected config, the calibration statuses and
the six-metric row against `oracle` and `flanking_copy` are what the run produces; whether `medcpt`
or `lookup` wins the gate, and by how much against R10's **0.0335** envelope, is what the run
answers.

---

### T09 — the inert-gate rule, and three findings from the first real selection run (2026-08-26)

The tier-1 STARmap selection ran. **Two of its four gates measured nothing**, and the run
surfaced two further defects. This entry is the fix and the diagnosis; the measurements the
run owes are named at the end and have not been made.

#### 1. `prior_mode` and `text_emb_mode` were scored where they cannot act

Both came back with a separation of **exactly 0.0000** under the incumbent the merged gate
selected, `expr_mode="cross-mix"`. That is not a tie. `infer/generate.py::_expression` returns
from `_cross_mix` **before** `prior_latent`, `flow.sample`, the decoder and the gene embeddings
are reached, so neither the GRF nor the text channel can change a single emitted count. The
gate built to test the open-vocabulary claim ran in the one configuration where it cannot be
tested, and `calibrate_lengthscale` already refuses that path for precisely this reason.

**This is the fourth control-that-cannot-fire in this project** — after `text_emb_mode` being a
dead gate nothing read (§9), the reduced-budget rule (R8/R9), and the capability tie-break
never being wired into the search.

**The rule, enforced the way the reduced-budget one is.** `specs/09` §3 now carries: *a gate
must not be scored under an incumbent that makes its options inert.* And the relation is
**derived, not declared** — `inert_gates(probe, cfg, gates)` *runs* each option and compares
emitted counts bitwise. Nothing writes down "cross-mix kills prior_mode"; a future edit that
creates a new inert path is caught by the person who introduces it rather than by a reversal
months later. `test_inertness_is_derived_by_running_the_path_not_declared` asserts it on the
real code in both directions: inert under `cross-mix`, live under `zinb-flow` and `auto-blend`.

Inertness is a property of the **code path, not the weights**, so the probe uses an *untrained*
model: no fit, one generation per option. It runs *before* the gate is scored, so the fits are
never spent on a measurement that cannot mean anything.

On detection the selector **re-orders**: the gate is measured under the best-ranked cell that
can decide it (`live_incumbent_for`), the choice is recorded in `SelectionResult.inert_notes`,
a `InertGateWarning` fires, and the report carries a section saying **the gate's value is
evidence from there, not from the shipped cell**. A gate inert under *every* scored
configuration raises `InertGateError` rather than shipping a 0.0000 tie.

#### 2. Every tier-1 gate decision rests on **two** folds, not three

`selection_folds` takes the *interior* sections — a boundary fold would decide gates on the
worst regime (R3). `paper_2_4_6` leaves four training sections (1/3/5/7), so the interior is
two, and `Config.selection_n_folds = 3` cannot be honoured. Nothing said so: the six columns
look identical whether they average 2 folds or 30.

`fold_scores` now returns the unaveraged per-fold six beside `selection_scores`'s mean, and
`test_selection_folds_are_two_on_a_four_section_stack` pins the count.

#### 3. The bbox clamp is the boundary sections' own slabs — diagnosed, not fixed

~50 % of query points clamped, "repeatedly". Traced by instrumenting `_warn_if_outside`:
**every outside point is on the z axis**, the pose's rotation and centre are correct, and the
model→data round trip is exact to **1.6e-5 µm** (float32 noise). It is not a frame bug.

`CTFFlow._layout_targets` jitters each cell's depth uniformly within its slab
(`z += U(-thickness/2, +thickness/2)`), but `Volume.bbox` is the box of section **centres**,
not of slabs. The first section sits *at* `z_min`, so half its jittered cells fall below the
bbox floor; likewise at `z_max`. That is ~50 % of one section's cells regardless of the point
set — matching the fixture (749/1500 cells, 2103/4096 MC points) and tier 1 (**2067 of
section_1's 4073**, against a predicted ~2036).

**The consequence is real.** For those cells the intensity head is queried at the tissue's
surface instead of at the depth the jitter chose, so the layout term's Poisson integral is
evaluated on a support that is half surface-clamped **at exactly the boundary sections** —
R3's regime, and a candidate contributor to R11's intensity integral being unstable 3.7x
between refits.

**Not fixed here.** Inflating the bbox by half a section thickness changes the support every
fitted number in this project was produced on; that is a T03/T04 design decision, not this
task's. `test_the_bbox_excludes_the_boundary_sections_own_slabs` asserts the arithmetic so the
day the bbox changes, it fails and the record is updated. Recorded as a spec question.

#### 4. What this leaves owed

Three measurements, none of them made:

* **`medcpt` vs `lookup` under `expr_mode="zinb-flow"`**, where the decoder is live. This is
  the measurement that was asked for and the selection could not make.
* **`cross-mix` beating `zinb-flow` by 0.0885 (2.6x the 0.0335 envelope) at full budget on
  real data** — copying beating generating on STARmap. Substantive enough to deserve scrutiny
  rather than acceptance, and at n = 2 folds a mean can be a win plus a loss.
* Both **per fold**, because a mean of two numbers is not a result.

`scripts/t09_audit_starmap.py` makes them: one gate, an incumbent the caller names, per-fold
columns, and each margin printed against 0.0335. `--preflight` refuses an incumbent under which
the gate is inert before anything is fitted.

---

### T09 — C33 and C34 decided; the fold count made visible (2026-08-26)

Both spec questions the previous entry raised came back accepted. This is what changed.

#### C33 — the bbox now spans the sections' slabs. Fixed.

`Volume._compute_bbox` takes `z ± thickness/2` at each end instead of the section centres. The
in-plane axes are untouched: `x` and `y` are real cell coordinates and a cell on the face is a
real cell there, which is what `field._BBOX_TOLERANCE_FRAC` exists for.

**What it changes, measured on the fixture (`scripts/` diagnostic, 8 training steps):**

| | before | after |
|---|---|---|
| bbox z (fixture volume) | `[0, 400]` | **`[−12.5, 412.5]`** |
| cells outside, `_layout_term` | **749 / 1500 (49.9 %)** | **0** |
| MC points outside, `_layout_term` | **2103 / 4096 (51.3 %)** | **2 / 4096 (0.05 %)** |
| which axis | z, entirely | **y** — in-plane layout proposals |

The 2 survivors are what the warning's own docstring calls expected ("layout proposals, planes
grazing the boundary"). The z axis is clean.

**One committed number moves, and it is a test:** `tests/test_schema.py::test_volume_derived_fields`
asserted `bbox[:, 2] == [0, 400]` and now asserts `[0 − half, 400 + half]`. Nothing else in the
suite changed value — 368 fast tests pass. No *report* number is invalidated that was not already
superseded: every fitted STARmap number to date is one seed on a config the selection has since
replaced, and every fixture number in `reports/` was produced before the boundary sections' slabs
entered the support.

**What it does not fix.** The clamp was a symptom; R11's intensity integral being unstable 3.7x
between refits is still open. What can now be said is that the layout term is no longer evaluated
against a half-surface-clamped support at the two boundary sections, so a re-measurement of R11
is no longer confounded by it.

#### C34 — a gate decided elsewhere is UNDETERMINED, not shipped

When a gate is inert under the incumbent that ships, the search still measures it (re-ordered
onto a live cell) — but **the winner is no longer adopted**. `SelectionResult.undetermined`
carries the gate and why; `SelectionResult.elsewhere_winner` carries what won there;
`selected.yaml` gains `undetermined:` and `undetermined_won_elsewhere:`; and the report's
selected-configuration table prints **UNDETERMINED** in place of a value.

The reasoning is the user's and it is the right one: under the configuration that ships, the gate
changes no emitted count, so writing a winner into `selected.yaml` would claim a decision the
shipped model cannot express. The evidence is not discarded — it is labelled as evidence about
the configuration it was taken under.

#### The two-fold constraint, made visible

`specs/09` §3 now carries **"A four-section training stack cannot honour `selection_n_folds = 3`"**
as a rule of its own. `selection_folds` takes the interior sections; `paper_2_4_6` leaves four
training sections; the interior is two.

* `GateReview.n_folds` is filled in for every gate and the report prints a **`folds`** column
  beside every margin, with a paragraph saying why the count is on the page.
* `selected.yaml` records `n_loso_folds`.
* `scripts/t09_select_starmap.py` prints `margin 0.0813 (n=2 folds)` per gate.
* `fold_scores` returns the unaveraged per-fold six, and `scripts/t09_audit_starmap.py` reports
  them per metric with an explicit sign-agreement check — at n = 2 a mean can be the average of
  a win and a loss.

#### Two things the fix surfaced that were not the fix

**A stale-cache hazard, closed.** C33 changed every score while leaving every `Config` hash
identical, so `ScoreCache` — keyed on the config — served **11 pre-fix cells straight into a
post-fix run** in the smoke test. `ScoreCache` now takes a `volume_cache_key` (sections, cell
and gene counts, **bbox**, flattened flag) and a changed volume misses instead of silently
hitting. **Any `scores.csv` from before 2026-08-26 is stale and will now correctly miss.**

**A latent test flakiness, exposed rather than caused.**
`test_sefl.py::test_cross_terms_are_the_specified_four` asserted every cross term `>= 0.0`.
`type` is a KL divergence, so it is non-negative *mathematically* — but on an untrained model
the student and the EMA teacher are identical, the KL is exactly zero, and float32 rounding
scatters it about zero: measured over 8 seeds, **4 negative, all |v| <= 9.6e-9** against a
float32 eps of 1.19e-7. The assertion was a coin flip and the bbox change happened to land on
tails. It now allows a float32 tolerance, with the measurement in the comment.

#### `--fit-only`, so the two audits parallelise

`scripts/t09_audit_starmap.py --fit-only` fits each arm and stops. The fit checkpoint is a
resume point and re-entering a finished fit is a no-op, so the four arms of the two audits run
as **four concurrent processes** and the two scoring runs afterwards resume instantly.

---

### T09 — both STARmap audits, both negative, and what they actually establish (2026-08-26)

`reports/t09_audit_text_emb_mode.md` and `reports/t09_audit_expr_mode.md` (+ their `.json`), run
on tier-1 STARmap at 2400 steps, seed 1, two LOSO folds (`section_3`, `section_5`), fits resumed
from `--fit-only` checkpoints. Both answers are **negative**, and the second one is more
interesting than "copying wins".

#### 1. `medcpt` loses to `lookup` — the open-vocabulary channel does not help here

| metric | `medcpt` | `lookup` | margin | vs 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | 0.7510 | **0.7943** | 0.0433 | 1.3x |
| `gearys_pearson` | 0.7519 | **0.7944** | 0.0425 | 1.3x |
| `umap_mixing` | 0.6564 | 0.6716 | 0.0152 | inside |
| `marker_field_r` | −0.1355 | −0.1196 | 0.0159 | inside |
| `marker_depth_r` | −0.3084 | −0.3117 | 0.0033 | inside |
| `celltype_localization` | 0.0256 | 0.0256 | 0.0000 | inert |

Folds agree in sign on both metrics that clear the envelope. **This is the first real-data
measurement of the gate at all** — every prior STARmap number was `lookup` under another name
(the five embedding builders that passed zeros or a bare symbol). The answer on a 28-gene panel
is that the text channel costs about 1.3x the envelope on the two autocorrelation metrics and
does nothing measurable on the other four.

#### 2. `cross-mix` beats `zinb-flow` — but on exactly the three metrics the record predicted

| metric | `cross-mix` | `zinb-flow` | margin | vs 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | **0.9255** | 0.7510 | 0.1745 | **5.2x** |
| `gearys_pearson` | **0.9293** | 0.7519 | 0.1774 | **5.3x** |
| `umap_mixing` | **0.8098** | 0.6564 | 0.1534 | **4.6x** |
| `marker_field_r` | −0.1412 | −0.1355 | 0.0057 | inside |
| `marker_depth_r` | −0.3086 | −0.3084 | 0.0002 | inside |
| `celltype_localization` | 0.0256 | 0.0256 | 0.0000 | inert |

**The three it wins are the three T06/T09 already said a copy would win.** The record's own words
about the independent-donor baseline: *"that baseline emits **real counts** drawn from real
donors, so per-gene autocorrelation and a shared-embedding mixing score are exactly where it is
strongest and where a generative head has the least to add."* `morans_pearson` and
`gearys_pearson` **are** per-gene autocorrelation; `umap_mixing` **is** the shared-embedding
mixing score. `cross-mix` emits donor counts verbatim, so it is a copy by construction.

On the three metrics that measure **spatial arrangement** rather than count realism, the two are
tied inside the envelope — and on `marker_field_r`, the project's standing weakness, `zinb-flow`
is nominally *ahead*.

So the honest statement is not "copying beats generating on STARmap". It is: **copying beats
generating on the three metrics that reward emitting real counts, by 4.6–5.3x, and neither path
can be distinguished on the three that would show a generative advantage.** That is a much
weaker claim for copying and a much sharper indictment of the generative path: it pays a large
price on count realism and buys nothing measurable in arrangement.

#### 3. Three cross-checks the two JSONs support, none of them planned

* **Determinism across processes.** The `zinb-flow` arm of the `expr_mode` audit and the
  `medcpt` arm of the `text_emb_mode` audit are the *same configuration* — hash
  `9d1ce6c0ff7cfb15` in both — and their means and per-fold values are **identical to full
  double precision** (`0.7509865177838673`, …). Two separate processes, same answer. Convention
  3 holds on real data, unprompted.
* **A fifth inert control.** `celltype_localization` is **constant at 0.0255776032059378 across
  all four arms**. Under `layout_mode="resample"` the cell types are copied from the flanking
  real section, so that metric cannot respond to *any* expression-path gate. The effective
  comparison is over **five** metrics, not six — and median rank over six with one guaranteed
  tie compresses every rank difference, because both candidates take 1.5 on it.
* **`--fit-only` worked.** Per-arm wall times are 5.1–7.8 s against a ~57 min 2400-step fit, so
  every arm re-entered a finished checkpoint as a no-op and paid only for the two fold scorings.

#### 4. The instrument, stated because it is not bench3's

These are `fold_scores` → `section_scores`, i.e. **T08's kernels under T10's names**
(`specs/09` §10), scored on internal LOSO over *training* sections. They are **not**
`bench3.evaluate_paper` and **not** the held-out 2/4/6 sections, so they must not be put beside
`reports/pilot.md`'s or `progress/numbers.md`'s bench3 numbers. The clearest tell is
`marker_field_r`, negative here (−0.12 to −0.17) where the pilot's bench3 figure was +0.1611:
different estimator, different sections.

#### 5. What this does and does not establish

**Establishes**, at one seed, on two folds, on this instrument: on tier-1 STARmap the text
channel does not help, and the flow path does not beat a copy on any metric while losing badly
on three.

**Does not establish** that either loses in general. The reading to test — and it is the
dataset's own properties, not a hope — is that **tier-1 STARmap is close to the most favourable
case a copy could be handed**: 28 genes, median per-gene detection **0.9999**, 22 µm spacing.
A panel that dense and that finely sectioned makes the neighbouring section an excellent
estimate of the held-out one, and leaves a 28-dimensional text channel very little to say.
`deep_starmap` — 1017 genes, `raw_counts`, 137 types, the same `paper_2_4_6` design — varies
both at once, and is the next measurement.

**One seed.** `specs/09` §3 asks for `claim_min_seeds` = 3 before any of this reaches a paper
claim, and the two margins that clear the envelope do so by 1.3x.

---

### T09 — `deep_starmap`: both mechanisms finally do something, and one of the two results is not robust (2026-08-26)

`reports/t09_audit_deep_expr_mode.md` and `reports/t09_audit_deep_text_emb_mode.md` (+ `.json`).
`deep_starmap`, `paper_2_4_6`, tier 2 — 1017 genes, mouse brain, `raw_counts`, 137 cell types —
at 2400 steps, seed 1, folds `section_3` / `section_5`. The hypothesis under test was that
tier-1 STARmap is close to the most favourable case a copy could be handed (28 genes, 0.9999
detection, 22 µm spacing), so the STARmap negatives are about *that dataset*, not about the
mechanisms.

#### 0. Which dataset these are, established rather than asserted

The reports' headers said `starmap_visual_cortex`, because `_report` hardcoded the tier-1 name.
That is a provenance defect and it is fixed — the header now comes from the resolved paths and
the JSON records `dataset`, `holdout`, `n_cells`, `n_genes`, `n_sections`, `train_steps`,
`expr_pca_dim` and `under`, none of which it carried before.

**The numbers are `deep_starmap`'s, proven from the config hashes.** Reconstructing the audit's
config gives all four arms exactly under `expr_pca_dim=32` (1017 genes, unclamped) and none
under `28` (STARmap's clamp) — and that same reconstruction reproduces the *STARmap* reports'
hashes exactly, which is a second, unplanned confirmation of both runs:

| arm | reported | `expr_pca_dim=32` | `=28` |
|---|---|---|---|
| expr / `cross-mix` | `f31b0764c4f27de1` | **match** | `d068229e5608c7e8` (= the STARmap report) |
| expr / `zinb-flow` | `336cbc6a491faa51` | **match** | `9d1ce6c0ff7cfb15` (= the STARmap report) |
| text / `medcpt` | `336cbc6a491faa51` | **match** | ” |
| text / `lookup` | `b21d98fed9b7b958` | **match** | `12fc94609f4d6c7a` (= the STARmap report) |

Only the headers were corrected; every number is as emitted.

#### 1. Nothing reversed. Every margin moved the same way, and further

| gate | metric | STARmap | deep_starmap |
|---|---|---|---|
| `expr_mode` (`cross-mix` − `zinb-flow`) | morans | +0.1745 | **+0.4321** |
| | gearys | +0.1774 | **+0.5533** |
| | umap_mixing | +0.1534 | **+0.2817** |
| | **marker_depth_r** | −0.0002 (tie) | **−0.2598** |
| `text_emb_mode` (`medcpt` − `lookup`) | morans | −0.0433 | **−0.1234** |
| | gearys | −0.0425 | **−0.1288** |
| | **marker_depth_r** | +0.0033 (tie) | **+0.1850** |

No metric changes sign. The sparse panel does not rescue either mechanism on the
count-realism metrics — it makes `cross-mix`'s lead **larger** (12.9x / 16.5x / 8.4x the
envelope, against 5.2 / 5.3 / 4.6 on STARmap) and `lookup`'s lead larger too (3.7x / 3.8x
against 1.3x).

#### 2. But `marker_depth_r` comes alive, and both mechanisms win it

The metric that was a dead tie on STARmap (0.0002 and 0.0033, both arms *negative* at −0.31)
separates on `deep_starmap`, and **in favour of the machinery both times**:

* **`zinb-flow` beats `cross-mix`: +0.2745 vs +0.0147**, 7.8x the envelope. The first real-data
  metric on which the generative path beats copying.
* **`medcpt` beats `lookup`: +0.2745 vs +0.0895**, 5.5x the envelope. The first real-data metric
  on which the text channel does anything at all.

Mechanistically this is the metric one would predict: `marker_depth_r` asks whether marker genes
carry the right laminar depth profile — whether the model knows which genes are superficial and
which are deep. With 1017 genes and **966 carrying summaries**, that is exactly what a gene
description can supply and what 28 curated genes never needed supplying.

#### 3. One of those two is not robust at n = 2, and the report did not say so

The tooling checked whether the *folds agree in sign* on the gap. That is necessary and not
sufficient. The missing statistic is the spread **within one arm** across folds:

| gate | metric | margin | worst within-arm fold spread | ratio |
|---|---|---|---|---|
| `expr_mode` | morans | 0.4321 | 0.0319 | 13.6x |
| `expr_mode` | gearys | 0.5533 | 0.0287 | 19.3x |
| `expr_mode` | umap_mixing | 0.2817 | 0.0242 | 11.7x |
| `expr_mode` | **marker_depth_r** | 0.2598 | 0.0597 | **4.4x** |
| `text_emb_mode` | morans | 0.1234 | 0.0319 | 3.9x |
| `text_emb_mode` | gearys | 0.1288 | 0.0284 | 4.5x |
| `text_emb_mode` | **marker_depth_r** | 0.1850 | **0.2033** | **0.9x** |

**`medcpt`'s `marker_depth_r` win is inside its own fold noise.** `lookup` alone swings
**−0.0121 → +0.1911** between `section_3` and `section_5` — a range wider than the 0.1850 that
separates it from `medcpt`. The two arms are being told apart by less than one arm moves on its
own. It clears the across-seed envelope by 5.5x and is still not a result.

`zinb-flow`'s win on the same metric is 4.4x its fold spread and **does** stand.

`_fold_spread` is now a column in every audit (**⚠** below 2x), with the reasoning in the report:
R10's 0.0335 is an across-*seed* envelope measured on the fixture and says nothing about how far
a metric moves between *this* dataset's folds.

#### 4. What this leaves

**Established** (one seed, two folds, T08 kernels on internal LOSO — *not* `bench3.evaluate_paper`):

* the generative path beats copying on **`marker_depth_r`**, on a sparse panel, robustly to the
  fold check — and loses count realism by 8–16x;
* the text channel's only positive is on the same metric and is **not** robust at n = 2.

**The reading that survives.** Copying wins wherever the metric rewards emitting real counts,
and that advantage *grows* with panel width rather than shrinking. What the generative path buys
is laminar structure — one metric, one dataset, one seed. `celltype_localization` was inert
again (constant −0.005087 across all four arms, because `resample` copies cell types), so the
effective comparison remains **five** metrics.

**The measurement this now justifies** is the one `specs/09` §3 already requires and neither
dataset has had: **`claim_min_seeds` = 3** on `marker_depth_r` for both gates, which is the only
thing that can separate the 4.4x from the 0.9x. Three seeds x two arms x two gates = 12 fits;
at `deep_starmap`'s measured ~500 s per scored arm plus the fit, that is the cheapest decisive
experiment left in T09.

---

### 2026-08-27 — the instrument for the `marker_depth_r` mechanism question

`scripts/t09_depth_mechanism.py`. The two `deep_starmap` audits agree on one metric and
disagree on every other, and the same metric splits **both** gates the same way. That is either
a mechanism or a coincidence of two two-fold means, and the margin cannot tell them apart — for
`text_emb_mode` the margin (0.1850) is *inside* its own worst within-arm fold spread (0.2033).

**Why a decomposition is a stronger test than the margin it explains.** `marker_depth_r` is a
mean over `Config.metric_marker_genes` (32) genes of a per-gene depth-profile correlation, so it
decomposes exactly: **32 internal degrees of freedom against the 2 LOSO folds**. A *structured*
pattern across genes is far harder to obtain by chance than a difference of two means, so this
can carry weight the n = 2 margin cannot. That is the whole reason for building it rather than
waiting on the three-seed run — which is still the decisive experiment and is not replaced.

**The hypothesis under test** (the user's): semantic gene embeddings group functionally related
genes, so a gene borrows laminar structure from its text neighbours, while a lookup table
memorises each gene and wins per-gene autocorrelation instead. Prediction: the per-gene gain
concentrates on genes whose *text neighbours* carry strong depth gradients.

**The confound, named before measuring.** A gene with a flat real depth profile conditions its
own Pearson r badly, so *any* arm difference has more room where the gradient is strong. Gain
will correlate with gradient strength with no semantics involved. Three controls separate them:

1. a **partial** Spearman of gain against the *neighbours'* gradient, holding the gene's own
   trend **and** its own bin-to-bin contrast fixed;
2. a **permutation null** (2000 shuffles of the text vectors across the marker genes, rebuilding
   the neighbourhood predictor) — does MedCPT's *actual* geometry matter, or only the shape of
   the gradient distribution?
3. **`expr_mode` as a control gate.** It does not touch the text channel. A neighbourhood effect
   on both gates is a property of the metric, not of text space; only a `text_emb_mode` effect
   *without* an `expr_mode` effect supports the hypothesis.

**Built on the metric's own code**, and the identity is asserted rather than assumed: the
per-gene terms are checked to average to `section_scores`' own `marker_depth_r` for that arm and
fold to 1e-6, and the run aborts naming the drift if they do not. A decomposition that does not
add back up explains a different number than the audit reported.

**No refit.** The audit's fit checkpoints are resume points; re-entering a finished fit restores
it. The script refuses with a named path if a checkpoint is missing, and `require_compatible`
refuses a checkpoint written under a different config — so `--train-steps` and `--expr-pca-dim`
must match the audit that wrote it, and a mismatch raises instead of silently refitting.

Not run here: `deep_starmap` and its four fits are on the user's server.

**Two changes the calibration forced.** `scripts/t09_depth_mechanism_calibration.py` plants three
worlds at the diagnostic's own scale (32 genes, kNN 10) and asks how often each rejects:

| test | folds | `null` | `text-carries-trend` (the confound) | `borrowing` (power) |
|---|---|---|---|---|
| two-sided | single fold | 8% | 5% | 18% |
| two-sided | pooled over 2 folds | 7% | 5% | 35% |
| one-sided | single fold | 8% | 8% | 28% |
| one-sided | pooled over 2 folds | 5% | 8% | 42% |

The **partial works**: the confound world — text space encodes the depth gradient, but the gain
depends only on each gene's *own* gradient — rejects at the null rate, 5–8%, while its raw
`rho(gain, neighbour trend)` sits at +0.42. Without the partial that world would have been read
as a mechanism.

**Power is the problem, not calibration.** 18–28% per fold. So two things were added rather than
left implicit: a **pooled row** (the genes both folds select as markers, gains averaged — the
same measurement at lower noise, not a third replicate) and a **one-sided** p against the sign
the hypothesis predicts, printed beside the two-sided one. Together 18% → 42%. That is still
weak enough that the report says so in the body: **a null result here is not evidence of
absence**, and does not replace `claim_min_seeds` = 3.

**Verified.** The identity holds *exactly* — per-gene mean vs `section_scores`' `marker_depth_r`,
drift **0.0e+00** on both folds of the synthetic fixture, and the run aborts on any drift above
1e-6. End to end on a sparse bench3-shaped fixture (2799 cells x 12 genes, 200 steps): four
restores from checkpoints, six rows, JSON and markdown, `make lint` / `mypy --strict` /
`pytest -m "not slow"` (371 passed) clean. Every partial on that fixture is noise, which is the
right answer — its text vectors are deterministic hashes of the descriptors and carry no
geometry.

### T09 — the `marker_depth_r` mechanism: hypothesis rejected, but the metric survives the test that killed it (2026-08-27)

`reports/t09_depth_mechanism_deep.md` (+ `.json`). `deep_starmap`, `paper_2_4_6`, 115 830 cells x
1017 genes, 2 folds, 32 marker genes each, 2400 steps, seed 1, 2000 permutations. No refit — the
audits' four fits restored from their checkpoints. The identity held at every one of the eight
arm x fold combinations or the run would have aborted.

#### 1. Semantic borrowing is not what is happening

The hypothesis was that MedCPT groups functionally related genes, so a gene borrows laminar
structure from its text neighbours. It predicts a **positive partial** — gain against the
neighbours' gradient strength, holding the gene's own gradient fixed.

| gate | fold | rho(gain, own trend) | rho(gain, nbr trend) | **partial** | p (1-sided) |
|---|---|---|---|---|---|
| `expr_mode` | `section_3` | +0.082 | +0.085 | **+0.178** | 0.171 |
| `expr_mode` | `section_5` | +0.411 | −0.247 | **−0.208** | 0.892 |
| `expr_mode` | pooled | +0.128 | −0.101 | **−0.041** | 0.613 |
| `text_emb_mode` | `section_3` | +0.016 | +0.187 | **+0.218** | 0.131 |
| `text_emb_mode` | `section_5` | +0.071 | −0.044 | **−0.096** | 0.714 |
| `text_emb_mode` | pooled | −0.064 | −0.080 | **−0.113** | 0.730 |

Nothing reaches significance, the two folds **disagree in sign** on both gates, and the pooled
partial is ≈ 0 on both. The two sub-questions answer separately:

* **does the gain concentrate on the strongest-gradient genes?** No. `rho(gain, own trend)` is
  +0.082 / +0.411 and +0.016 / +0.071 — one number out of four, on one fold.
* **does it track text-space similarity?** No, and not even before the partial: the *raw*
  neighbour correlation is +0.085 / −0.247 and +0.187 / −0.044. There is no signal here that the
  partial is stripping away — the control is not what is killing the hypothesis.

**Power keeps this from being a disproof**: 18–28% per fold, 35–42% pooled
(`reports/t09_depth_mechanism_calibration.md`). This is "not supported", not "refuted".

#### 2. But the decomposition establishes something the margin could not

The per-gene terms answer three questions the 2-fold mean cannot reach
(`scripts/t09_depth_mechanism_summary.py`, which reads the diagnostic's JSON — no refit):

**The advantage is broad, not carried by a few genes.** `expr_mode`: **24/32 and 25/32** genes
improve (Wilcoxon p = 1.1e−3, 5.1e−4). `text_emb_mode`: **26/32 and 20/32** (9.1e−4, 4.3e−2).
Per-gene gains within a fold share a model and a section, so these are optimistic — but a mean
over two folds could not have shown breadth either way.

**Which gene benefits is reproducible — for one gate only.**

| gate | Pearson(gain in `section_3`, gain in `section_5`) | p | improved in both |
|---|---|---|---|
| `expr_mode` | **+0.645** | 0.000 | 19/29 |
| `text_emb_mode` | +0.259 | 0.175 | 16/29 |

**This independently reproduces the fold-spread split, at 29 degrees of freedom instead of 2.**
The fold-spread column said `expr_mode`'s `marker_depth_r` win stands (4.4x its within-arm
spread) and `text_emb_mode`'s does not (0.9x). The per-gene pattern says the same thing by a
different route: the flow path helps a *stable, identifiable set of genes* on both held-out
sections, and the text channel's per-gene pattern is not distinguishable from noise. Two
statistics with almost nothing in common agreeing on which of the two wins is real is the
strongest thing in this run.

**A correlation that is not a finding.** The two gates appear to help the same genes (+0.502,
+0.532; p ≈ 0.002–0.003). They share their winning arm — the same fitted config is `zinb-flow`
for one gate and `medcpt` for the other — so both gains carry the same `+r(shared arm)` term and
would correlate with unrelated losing arms. Recorded with the caveat so it is not later read as
the two gates corroborating each other.

#### 3. What this does to the framing

The scope statement is available for the **generative path** and not for the **text channel**:

* `zinb-flow` has a real advantage on laminar depth structure — broad across genes, reproducible
  in *which* genes, robust to the fold check, on a 1017-gene panel — while losing count realism
  by 8–16x. "Better at a different thing" is supportable here.
* `medcpt`'s only positive remains **unestablished** on every test applied to it: inside its own
  fold spread, and its per-gene pattern does not reproduce across folds. The open-vocabulary
  channel has still not been shown to do anything on real data.
* The mechanism is **not** the proposed one. Whatever `zinb-flow` is doing for depth profiles, it
  is not gene-neighbourhood borrowing in text space — which is consistent with the effect being
  present at all under `text_emb_mode=lookup`, where there is no text geometry to borrow from.

**Unchanged**: `claim_min_seeds` = 3 on `marker_depth_r` (12 fits) remains the decisive
experiment. This run narrows what it needs to settle — `expr_mode`, not both gates.

#### 4. Two corrections to how these two gates were being read (2026-08-27)

**(a) The two gates were never independent evidence, and the record said they were.** `zinb-flow`
under `prior_mode=correlated` and `medcpt` under `expr_mode=zinb-flow` are the **same fitted
config**, hash `336cbc6a491faa51`. So the winning arm is *one model measured against two different
losers*, not two gates agreeing. The per-gene numbers show it directly — the winner's rows are
identical in both gates:

| gate | fold | arm | median per-gene r | mean | fraction of genes with r > 0 |
|---|---|---|---|---|---|
| `expr_mode` | `section_3` | `zinb-flow` | +0.2530 | +0.2447 | 0.69 |
| `text_emb_mode` | `section_3` | `medcpt` | +0.2530 | +0.2447 | 0.69 |
| `expr_mode` | `section_5` | `zinb-flow` | +0.2739 | +0.3044 | 0.75 |
| `text_emb_mode` | `section_5` | `medcpt` | +0.2739 | +0.3044 | 0.75 |

A single favourable draw of that one fit inflates **both** margins simultaneously. This was the
stated motivation for building the mechanism diagnostic at all ("one metric splitting two
independent gates the same way looks like a mechanism") and that motivation was weaker than
claimed. The docstring of `scripts/t09_depth_mechanism.py` now carries the correction; the earlier
"help the same genes" correlation (+0.50, +0.53) was already flagged as inflated by the shared
term, and this is the same defect stated at full strength. **It also decides what the repeated-seed
run is for**: the quantity most likely to be wrong is `r(336cbc6a)` itself, and only new seeds
move it.

**(b) The gain lifts three quarters of the panel, which is the opposite of selective borrowing.**
75–81% of individual marker genes improve. A mechanism that helps most of the panel at once is
not genes borrowing from semantic neighbours — the partial correlation says the same thing, but
the breadth says it without needing the statistic. Any surviving explanation has to be something
that shifts *all* genes' depth profiles together, not something that routes information between
particular genes.

**What the arm-level numbers point at.** `cross-mix`'s per-gene profile correlation is a **coin
flip**: mean −0.0039 / +0.0333, with **47% / 50%** of genes above zero. `zinb-flow`'s is only
modestly positive (+0.2447 / +0.3044, 69% / 75% above zero). Both arms share `layout_mode=resample`,
so the *positions* are real and identical in both; only the expression assignment differs. So the
gate is not "the flow head models laminar depth well" — it is "copying donors' counts lands laminar
structure no better than chance, and the flow head is weakly better than chance". Whether +0.27 is
close to what is achievable on this metric is **not known**, and that missing number is the cheapest
thing left to measure (§5).

#### 5. The two instruments the corrections call for (2026-08-27)

**`scripts/t09_depth_ceiling.py` — what is `marker_depth_r` capable of?** No model, no fit, no
generation: the metric's own kernels applied to the built input. Four reference points per target
section, all on the target's own ruler (its markers, its bounds, its `z`) exactly as
`section_scores` builds them:

* `self` — the target against itself, which **must** be 1.0. A correctness check on the file; the
  run aborts otherwise. Measured 1.000000 on the synthetic fixture.
* `split_half` — the target's cells split at random, halves correlated, Spearman-Brown corrected.
  The **reliability** R of a whole-section profile.
* `noiseless_ceiling` — `sqrt(R)`. Correction for attenuation: a method with *no* noise of its own
  still cannot exceed this against a target that is itself a finite sample. The hard bound.
* `other_section` — another **real** section on the target's ruler, by `|dz|`. The **copying
  ceiling**: the most a donor-copying method could score, which is what `cross-mix` competes
  against rather than 1.0.
* `shuffled` — the target's counts with the cell-to-position assignment permuted. The floor.

This is the number the whole `expr_mode` reading turns on and it has been missing all along. If
the copying ceiling is near zero, `cross-mix`'s +0.0147 is not a model failure and the margin says
nothing about the flow head. The audit numbers quoted in the report are **read out of the audit
JSON**, not transcribed — this project has twice shipped a report carrying a hand-copied number
for the wrong dataset.

**`scripts/t09_seed_claim.py` — `specs/09` §3's repeated-seed rule, applied.** Aggregates one
audit JSON per seed and reports the spread rather than a point estimate. A margin is **STANDS**
only under four conditions: every seed agrees in sign; the mean margin exceeds the across-seed
spread; it exceeds the 0.0335 R10 envelope; and it exceeds the largest within-arm fold spread.
`specs/09` requires only the third and the reporting — the other three are this file's and are
stated in its output so a reader can disagree and recompute. "Not established" is reported as
distinct from "refuted".

Two refusals, both exercised: an audit JSON with no `seed` field raises naming it rather than
guessing from the filename, and runs that differ in anything but the seed (`train_steps`,
`expr_pca_dim`, `under`, dataset, cell or gene count) raise rather than being averaged.
`t09_audit_starmap.py` now records `seed` in every row — it did not, which a seed comparison
cannot work without.

`make lint`, `mypy --strict`, `pytest -m "not slow"` (371 passed) clean.

### T09 — the ceiling test answered its question and exposed a coordinate-frame defect (2026-08-27)

`reports/t09_depth_ceiling_deep.md` (+ `.json`), run on `deep_starmap` / `paper_2_4_6`, all four
training sections. No model, no fit, no generation.

| target | cells | split-half R | noiseless ceiling √R | nearest other section | best other | shuffled floor |
|---|---|---|---|---|---|---|
| `section_1` | 39 327 | +0.9914 | 0.9957 | +0.9859 | +0.9859 | −0.0217 |
| `section_3` | 29 842 | +0.9924 | 0.9962 | +0.9631 | +0.9861 | +0.0036 |
| `section_5` | 28 654 | +0.9900 | 0.9950 | +0.8578 | +0.9739 | +0.0304 |
| `section_7` | 18 007 | +0.9871 | 0.9935 | +0.9293 | +0.9293 | +0.0029 |

`self` was exactly 1.000000 on all four, so the profile code reproduces the metric.

**The metric is almost perfectly reproducible, and copying a whole real section scores +0.86 to
+0.99.** Against that, the audits' `zinb-flow` at **+0.2745** and `cross-mix` at **+0.0147** are
both at the **shuffled floor** (−0.022 … +0.030). `cross-mix` *literally copies donor counts*; it
cannot be 60x worse than copying a section unless something is broken. That is a defect
signature, not a modelling result — and it was.

#### The defect

`generate_section` documents its output as ``obsm[cfg.coord_key]`` = the **plane-local** `(u, v)`
and `obsm["xyz"]` = the physical `(x, y, z)`. `_fold_scores` passed `obsm[cfg.coord_key]` into
`section_scores`, which compares it against `Section.coords` — **physical**. For
`section_plane` the plane basis is `e1 = (0, 1, 0)`, `e2 = (-1, 0, 0)`, so

    u = y - y_c ,   v = -(x - x_c)

— a 90-degree rotation *and* a re-centring on the section's bounding box. Verified on real code
to **1.5e-5 µm**: `obsm['spatial']` ranged `u[-399, +399] v[-398, +398]` where the real section
ranged `x[2, 799] y[1, 800]`, and `obsm['xyz'][:, :2]` matched the real range exactly.

**Why it was silent for the whole of T09.** Three of the six metrics are built on each side's
**own kNN graph** and are invariant to a rigid transform — `morans_pearson`, `gearys_pearson`,
`umap_mixing` came out **numerically identical** either way. The other three place both sides on
a ruler built from the *real* section's extent and collapse to the floor:

| metric | as shipped (u, v) | using `xyz` |
|---|---|---|
| `morans_pearson` | +0.9780 | +0.9780 |
| `gearys_pearson` | +0.9823 | +0.9823 |
| `umap_mixing` | +0.9674 | +0.9674 |
| `marker_field_r` | +0.0311 | **+0.4048** |
| `marker_depth_r` | +0.1090 | **+0.4373** |
| `celltype_localization` | −0.1670 | **+1.0000** |

Half the table moving and half not is not distinguishable from a modelling result. This is the
fifth control-that-cannot-fire class in this project and the most expensive: it produced numbers
that were reported, interpreted, and built on for four separate investigations.

**It also corrects an earlier explanation.** `celltype_localization` was recorded as "inert
(constant −0.005087 across all four arms) because `resample` copies cell types". The *constancy*
had that cause; the *value* was the frame defect holding it at the floor. On the fixture it is
+1.0000 once the frame is right.

#### Scope, checked rather than assumed

**Not affected.**

* **bench3's published `paper_*` metrics.** `run_spatialcpav25_gen.py` already passes
  `emitted.obsm["xyz"]` to `_v2_io`, with a comment saying exactly why; `t09_ship_starmap` and
  `t10_rescore_saved` write `xyz` into the prediction files too. Every benchmark number stands.
* **The training losses.** `train/loso.py` calls `loss_profile(x_gen, recon.coords, x_real,
  recon.coords, ...)` — *one* coordinate array for both sides, so the comparison is frame-free.
  No model was trained against a rotated target.
* **`module_morans_agreement`** and **`calibrate_ell_xy`** — both build a kNN graph inside the
  generated section only, so both are invariant. `calibrate_ell_z` already used `xyz`.
* **`scripts/t09_depth_ceiling.py`** — real coordinates on both sides. The table above stands.

**Affected — every number from the T08-kernel internal LOSO six, on three of the six metrics:**
the per-dataset selection (`FitScorer` → `selection_scores`), every `t09_audit_starmap` run
(tier-1 STARmap **and** `deep_starmap`), `t09_layout_mode_gate` (the **R11 table that made
`resample` the shipped default**), `t09_report`, and `t09_depth_mechanism`.

#### The fix, and the guard

`_fold_scores` now passes `obsm["xyz"][:, :2]`; so do the three `section_scores` calls in
`t09_report.py` and the one in `t09_depth_mechanism.py`. `section_scores` gained
`assert_same_frame` / `FrameMismatchError`, which refuses coordinates whose centroid sits more
than 25% of the real section's bounding-box diagonal away, or whose per-axis extents differ by
more than 25%. Documented as **not a proof** — a pure rotation about the true centroid of a
square region changes neither statistic — so the call sites keep the explicit comment too. A
first attempt using "centroid inside the bounding box" was too weak and is recorded here because
it *passed* on the synthetic fixture, whose sections span [0, 1000]² so the rotated centroid
lands just inside.

Two regression tests: the guard fires on the exact `(u, v)` transform, and `fold_scores` drives
the real path (fit, generate, score) and reaches the guard with coordinates it accepts. 373 pass,
`ruff` and `mypy --strict` clean.

On the synthetic fixture at 200 steps the `expr_mode` gate **reverses** once the frame is right —
`cross-mix` +0.4081 vs `zinb-flow` +0.0198 on `marker_depth_r`, where the shipped code had them
at +0.0509/+0.1090. That is an under-trained fixture and predicts nothing about `deep_starmap`,
but it establishes that the sign of the gate is not safe.

#### What this invalidates

Every conclusion in §1–§4 above that rests on `marker_depth_r`, `marker_field_r` or
`celltype_localization` — which is most of them:

* "the generative path beats copying on `marker_depth_r`" — the margin was between two arms both
  pinned at the floor by the defect;
* "`medcpt` beats `lookup` by 5.5x on `marker_depth_r`" — same;
* the per-gene decomposition and the borrowing test, which decomposed those gains;
* the fold-spread and per-gene-reproducibility split between the two gates.

`morans_pearson`, `gearys_pearson` and `umap_mixing` are unchanged, so **"copying wins count
realism, by 8–16x on a wide panel" survives intact.** The three-seed run on `text_emb_mode` was
**not started** — it would have spent four fits establishing a margin on a broken metric.

### T09 — the re-score: the frame fix validated exactly, the `marker_depth_r` result reversed, and a second defect underneath (2026-08-27)

Both `deep_starmap` audits re-scored on the existing fits (no refit; `config_hash` unchanged on
all four arms, so the checkpoints were accepted).

#### 1. The frame fix behaves exactly as predicted

Full-precision, pre-fix → post-fix, per arm:

| metric | change | verdict |
|---|---|---|
| `umap_mixing` | **0.00e+00** on all four arms | bitwise identical — expression-space only |
| `morans_pearson` | −1.8e−06 … +2.5e−05 | ~1e−5 |
| `gearys_pearson` | −1.1e−05 … +6.8e−05 | ~1e−5 |
| `marker_field_r` | +0.36 … **+1.03** | moved |
| `marker_depth_r` | +0.11 … **+0.97** | moved |
| `celltype_localization` | **+1.01** on all four arms | moved |

`umap_mixing` identical to the last bit and the two graph metrics moving only at 1e−5 is the
predicted signature: the two graph metrics are isometry-invariant *mathematically* but the
coordinates round-trip through `float32` in two different frames, so a handful of kNN ties flip.
The scope analysis holds.

#### 2. The headline result **reversed**

| metric | `cross-mix` | `zinb-flow` | margin | was (pre-fix) |
|---|---|---|---|---|
| `morans_pearson` | +0.9657 | +0.5337 | 0.4321 | same |
| `gearys_pearson` | +0.9229 | +0.3695 | 0.5534 | same |
| `umap_mixing` | +0.7902 | +0.5084 | 0.2817 | same |
| `marker_field_r` | **+0.9659** | +0.2738 | 0.6921 | −0.0658 / −0.0853 |
| `marker_depth_r` | **+0.9875** | +0.3888 | 0.5988 | +0.0147 / **+0.2745** |
| `celltype_localization` | +1.0000 | +1.0000 | 0.0000 | −0.0051 both |

**`cross-mix` now wins every live metric.** The one real-data result where the generative path
beat copying — `zinb-flow` over `cross-mix` on `marker_depth_r` by 7.8x the envelope — was an
artifact of the frame defect and is **gone and reversed**: copying wins that metric by 0.5988.

And `text_emb_mode`: `medcpt` +0.3888 vs `lookup` +0.4087 on `marker_depth_r` — margin **0.0199,
inside the envelope, 0.6x the fold spread**, and now favouring `lookup`. `medcpt`'s only positive
on real data (+0.1850) is gone. **The text channel loses or ties on all six.**

`cross-mix`'s +0.9875 landing inside the copying-ceiling band the model-free ceiling test
measured independently (+0.86 … +0.99) is a cross-validation of both instruments.

#### 3. `celltype_localization` = exactly 1.0 exposed a second defect

Exactly `1.0` to full double precision on all four arms is not a metric. It is not:
`layout_mode="resample"` reuses **the nearest flanking section's coordinates and cell types**,
and `_flanking` took the two nearest sections of `vol` **with no exclusion**. Generating at a
training section's own depth — which is what internal LOSO does on every fold — returns that
section itself. Measured on the fixture:

```
section_3 (z=41.0): _flanking -> [section_3 (41.0), section_1 (19.0)]   nearest is THE FOLD ITSELF
  max |generated coord - hidden coord| = 0.000e+00 um
  cell types identical to the hidden section: True (100.0% match)
```

**The retrieval pool honoured `exclude_z` all along**, which is exactly what let this survive:
the *expression* was generated honestly while the *positions and cell types* were handed over.
`celltype_localization` is a function of positions and types only, so it read 1.0 — and the frame
defect had it pinned at −0.005087, so the tell was invisible until the first fix landed. Two
defects, the second masked by the first.

**Scope.** Internal LOSO only, where the hidden fold is still a member of `vol`:
`_fold_scores` / `selection_scores` (the per-dataset selection and every audit),
`t09_layout_mode_gate`, and any calibration probe generating at a training section's depth.
**R11 is safe** — `scripts/t10_layout_modes_table.py` scores the *held-out* sections through the
bench3 instrument, and a `TrainingVolume` does not contain them, so `_flanking` cannot return
them; `resample` 0.7546 against a `flanking_copy` floor of 0.7765 is what an honest flanking copy
looks like. The **shipped `layout_mode=resample` default stands.** bench3's published `paper_*`
numbers are unaffected for the same reason.

**Fixed**: `_flanking` takes `exclude_z` and applies the same thousandth-of-a-spacing tolerance
as `_default_exclusions`; `generate_section` passes its own exclusion set through `_layout_on`.
Deployment is unchanged — at a novel depth nothing is excluded and the two nearest sections
genuinely flank the plane. Two regression tests: the layout is no longer an exact copy of the
excluded section, and a fully excluded stack raises naming the specimen. 375 pass in 2m55s,
inside the 3-minute budget.

#### 4. Where this leaves the record

Everything measured through internal LOSO needs a **third** pass. What survives all of it:
`morans_pearson`, `gearys_pearson` and `umap_mixing` are untouched by either defect, so
**copying beats the generative path on count realism by 8–16x on a wide panel** still stands, and
is now joined rather than opposed by the two coordinate metrics.

What is gone: every claim that the generative path or the text channel wins *anything* on real
data. The three-seed run was never started, which is the one piece of luck here — it would have
spent four fits on a margin that has since changed sign and then changed instrument.

### T09 — third re-score, both defects fixed: the instruments agree and the result is clean (2026-08-27)

`deep_starmap` / `paper_2_4_6`, 2400 steps, seed 1, 2 folds, scoring only on the existing fits.
`celltype_localization` is **+0.6028**, no longer exactly 1.0 — the leak is closed.

#### 1. Two independently written instruments agree to 0.001

`layout_mode=resample` copies the **nearest admissible** flanking section. The ceiling test
measured every donor independently, **with no model at all**. They should therefore report the
same number for `cross-mix`, which copies donor counts onto that layout:

| fold | `cross-mix` `marker_depth_r` | nearest donor | ceiling test (model-free) | difference |
|---|---|---|---|---|
| `section_3` | +0.9622 | `section_5` (42.0 µm) | +0.9631 | **−0.0009** |
| `section_5` | +0.8677 | `section_7` (40.6 µm) | +0.8578 | **+0.0099** |

Written days apart, sharing no code path, agreeing to 0.001–0.01. That validates the frame fix,
the leak fix, and the ceiling instrument at once, and it settles what `cross-mix` *is* on this
dataset: **copying the nearest other section**, to within a hundredth.

Against the same ruler, `zinb-flow` reaches **36% and 35% of what copying achieves** — the same
fraction on both folds:

| fold | `zinb-flow` | copying | noiseless ceiling √R | shuffled floor |
|---|---|---|---|---|
| `section_3` | +0.3422 | +0.9631 | 0.9962 | +0.0036 |
| `section_5` | +0.2982 | +0.8578 | 0.9950 | +0.0304 |

Well clear of the floor — the flow head does model laminar depth — and about a third of the way
to what a copy gets for free.

#### 2. `expr_mode` — copying wins four of five, on both checks

Recomputed from the JSON (margin `cross-mix` − `zinb-flow`; "spread" is the worst within-arm
fold spread):

| metric | margin | fold spread | ratio | vs 0.0335 | folds agree |
|---|---|---|---|---|---|
| `morans_pearson` | +0.3930 | 0.0657 | **6.0x** | 11.7x | yes |
| `gearys_pearson` | +0.5502 | 0.0910 | **6.0x** | 16.4x | yes |
| `umap_mixing` | +0.2364 | 0.1547 | ⚠ 1.5x | 7.1x | yes |
| `marker_field_r` | +0.6614 | 0.1105 | **6.0x** | 19.7x | yes |
| `marker_depth_r` | +0.5948 | 0.0945 | **6.3x** | 17.8x | yes |
| `celltype_localization` | 0.0000 | 0.1430 | — | — | — |

**Copying beats the generative path on every live metric**, four of them clearing both the
envelope and the fold spread. The earlier reading — "copying wins count realism and buys nothing
in arrangement" — is **inverted**: the two arrangement metrics are now copying's *largest*
margins (0.66 and 0.59), because they were the two the frame defect had pinned at the floor.

#### 3. `text_emb_mode` — one established result, and it is negative for `medcpt`

| metric | margin (`medcpt` − `lookup`) | fold spread | ratio | vs 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | **−0.1252** | 0.0126 | **10.0x** | 3.7x |
| `gearys_pearson` | −0.1364 | 0.1058 | ⚠ 1.3x | 4.1x |
| `umap_mixing` | −0.0177 | 0.0661 | ⚠ 0.3x | 0.5x |
| `marker_field_r` | −0.0562 | 0.0637 | ⚠ 0.9x | 1.7x |
| `marker_depth_r` | −0.0080 | 0.0440 | ⚠ 0.2x | 0.2x |
| `celltype_localization` | 0.0000 | 0.1430 | — | — |

Every sign favours `lookup`, and exactly **one** metric clears both thresholds:
`morans_pearson`, at 10.0x the fold spread with both folds agreeing. So the one thing established
about the open-vocabulary channel on real data is that **MedCPT embeddings make per-gene spatial
autocorrelation worse.** `marker_depth_r`, which two rounds ago was `medcpt`'s headline win at
5.5x the envelope, is now −0.0080 — 0.2x its own fold spread, i.e. nothing.

#### 4. `celltype_localization` is structurally inert for every expression-side gate

+0.6028388701933918, **bit-identical across all four arms**. That is correct, not a defect:
under `layout_mode=resample` the cell types come from the copied layout, which is the same in
both arms of a gate that changes only the expression path. The metric measures the **layout**,
so it can separate `layout_mode` and nothing else.

Recording it explicitly because it has now been misread twice — once as "inert because resample
copies cell types" (true but incomplete: the *value* was floored by the frame defect) and once as
a leak signature (true, at 1.0). **Under the shipped default, every expression-side gate has a
five-metric table, not six.** A 0.0000 margin there is structural and is not evidence of a tie.

#### 5. What stands after three passes

* **Copying beats the generative path on all five live metrics, on a 1017-gene panel**, by
  6.0–6.3x the within-arm fold spread on four of them. One seed, two folds.
* **`cross-mix` under `resample` is exactly "copy the nearest other section"** — confirmed to
  0.001 against a model-free instrument. Its scores are not a model result and should never be
  quoted as one.
* **`zinb-flow` reaches ~35% of copying** on `marker_depth_r`, consistently across folds, well
  clear of the shuffled floor.
* **`medcpt` has no positive on real data**, and one established negative (`morans_pearson`).
* Unchanged by any of this: bench3's published `paper_*` numbers, the training losses, and R11's
  `layout_mode=resample` default.

### T09 — the finding, and the scope limit underneath it (2026-08-27)

#### The finding: `cross-mix` **is** the nearest-section copy, measured two independent ways

`layout_mode=resample` copies the nearest admissible flanking section's coordinates; `cross-mix`
then fills them with donor counts. `scripts/t09_depth_ceiling.py` measures what copying a whole
real section scores, **with no model, no fit and no generation** — a completely separate code
path. On `deep_starmap` they report the same number:

| fold | `cross-mix` `marker_depth_r` | nearest donor | model-free copy | difference |
|---|---|---|---|---|
| `section_3` | +0.9622 | `section_5` (42.0 µm) | +0.9631 | **−0.0009** |
| `section_5` | +0.8677 | `section_7` (40.6 µm) | +0.8578 | **+0.0099** |

**This is the strongest result in this line of work.** It is not a margin between two arms — it
is an identification. `cross-mix` under the shipped `resample` default is a copy of the nearest
other section, to within a hundredth, and its scores are a property of the tissue rather than of
the model. Every table that quotes `cross-mix` as a *method* is quoting a copy.

Beside it, the statement worth carrying instead of either gate's margin:

| fold | `zinb-flow` | copying | noiseless ceiling √R | shuffled floor | share of copying |
|---|---|---|---|---|---|
| `section_3` | +0.3422 | +0.9631 | 0.9962 | +0.0036 | **36%** |
| `section_5` | +0.2982 | +0.8578 | 0.9950 | +0.0304 | **35%** |

**The flow head models laminar depth — well clear of the shuffled floor — and gets about a third
of what a copy gets for free.** The same fraction on both folds. That is more useful than
"copying wins by 0.5948", because it says how much of the gap is real modelling and how much is
the task being easy.

#### The scope limit: on this tissue the reconstruction task is *saturated*

The ceiling test also measured how far real sections' depth profiles stay correlated: **0.78–0.99
across 40–125 µm**. On tissue that stable, copying is near-optimal by construction. Quantified as
the **headroom** — how much a *perfect, noiseless* method could beat the best available copy, on
the same attenuated scale the audits report:

| target | noiseless ceiling √R | best copy | headroom | vs the 0.0335 envelope |
|---|---|---|---|---|
| `section_1` | 0.9957 | 0.9859 (43 µm) | +0.0098 | **0.3x** |
| `section_3` | 0.9962 | 0.9861 (43 µm) | +0.0101 | **0.3x** |
| `section_5` | 0.9950 | 0.9739 (42 µm) | +0.0210 | **0.6x** |
| `section_7` | 0.9935 | 0.9293 (41 µm) | +0.0642 | 1.9x |

**On three of the four sections the entire headroom above copying is smaller than the run-to-run
reproducibility envelope.** Median 0.0156, 0.5x. No method — not v25, not a perfect one — can beat
copying by a measurable amount on this design. `marker_depth_r` under `paper_2_4_6` is not a test
the generative path can pass or fail; it is a test that has no room in it.

Headroom does grow with the gap. Median over all donor pairs, by section spacing:

| gap | median headroom | vs envelope | |
|---|---|---|---|
| ~40 µm | +0.0270 | 0.8x | **saturated — inside the envelope** |
| ~80 µm | +0.0757 | 2.3x | marginal |
| ~120 µm | +0.1655 | 4.9x | real headroom |

`paper_2_4_6` puts every fold in the ~40 µm regime. **This is a property of the experimental
design, not of the method**, and it was invisible until the metric had a measured ceiling.

#### What the generative path's case has to rest on, and what would test it

Reconstruction of an interpolated section cannot separate the methods here. The case has to rest
on the two things copying cannot do at all:

**(a) Unmeasured genes — and note this retires the `text_emb_mode` result as evidence *against*
`medcpt`.** `lookup` has no embedding row for a gene it never saw; `medcpt` projects one from
text. But **every A3 measurement so far has been on genes the model was fitted on**, where
`lookup` memorises per gene and `medcpt` is strictly more constrained (a linear image of a frozen
768-d vector plus a gated residual). *The gate as run cannot favour `medcpt` even in principle.*
Its repeated losses are the expected result of testing an open-vocabulary mechanism on a closed
vocabulary. The machinery to do it properly already exists and has never been run:
`TextGroundedEmbedding.forward_zero_shot`, and `gene_pool` threaded through `train_ctfflow`,
`reconstruct_hidden` and the SEFL losses precisely so a zero-shot run's held-out genes stay held
out.

**(b) Arbitrary planes.** `generate_oblique` and `generate_curved` exist. On real data there is no
ground truth at an oblique angle — no real section to score against — so the comparison is
**categorical**: copying has no output at all, and the measurable quantity is self-consistency
(`stack_pair_correlations`, intersection agreement where two planes cross) rather than a
head-to-head score.

**The cheap screen before either.** `exclude_z` already accepts an arbitrary set, so on the
**existing checkpoints** one can widen the effective gap at inference — reconstruct a fold with
the nearest one or two admissible donors also excluded — and measure whether `zinb-flow` degrades
more slowly than copying does. The copy curve is already known model-free; only v25's is missing.
Scoring only, no refits. **Caveat that must travel with it**: the model was *trained* on the
sections being excluded, so this measures extrapolation when the nearby evidence is withheld at
inference, not when it was never seen. It is a screen, not the experiment — if v25's curve falls
as fast as copying's, the wide-gap refit will fail too and need not be paid for.

### T09 — tier-1 STARmap re-scored, and a third check the report was missing (2026-08-27)

Scoring only on `runs/audit`; all four config hashes unchanged, so no refit. 16 527 cells x 28
genes, `expr_pca_dim` correctly clamped to 28.

#### 1. The two fixes move tier-1 as they moved `deep_starmap`

| arm | morans | gearys | umap | `marker_field_r` | `marker_depth_r` | `celltype_localization` |
|---|---|---|---|---|---|---|
| `cross-mix` | −0.0119 | −0.0057 | −0.0089 | **+0.8734** | **+1.0342** | **+0.7498** |
| `zinb-flow` / `medcpt` | −0.0047 | −0.0039 | −0.0115 | **+0.8152** | **+0.9376** | **+0.7498** |
| `lookup` | −0.0122 | −0.0146 | −0.0132 | **+0.8242** | **+1.0070** | **+0.7498** |

The three coordinate metrics were **negative** before (`marker_depth_r` −0.3086 / −0.3084 /
−0.3117, `marker_field_r` ≈ −0.13) and are now +0.63 … +0.73. The three graph metrics move by
−0.004 … −0.015 — small and *negative*, because unlike the `deep_starmap` frame-only comparison
this run also closed the layout leak, so the generated section is a genuine flanking copy rather
than the fold itself and everything got slightly harder.

**The old tier-1 conclusion was wrong in both halves.** "`marker_field_r` 0.0057, `marker_depth_r`
0.0002 … copying wins count realism and buys nothing in arrangement" — those two margins were
between two arms both sitting at −0.13 and −0.31.

#### 2. `expr_mode` on 28 genes: copying wins, but arrangement barely separates

| metric | margin (`cross-mix` − `zinb-flow`) | vs envelope | vs fold spread | fold balance |
|---|---|---|---|---|
| `morans_pearson` | +0.1673 | 5.0x | **3.9x** | 0.92 |
| `gearys_pearson` | +0.1756 | 5.2x | **4.3x** | 0.94 |
| `umap_mixing` | +0.1559 | 4.7x | **2.5x** | 0.85 |
| `marker_field_r` | +0.0525 | 1.6x | ⚠ 0.3x | 0.77 |
| `marker_depth_r` | +0.0964 | 2.9x | ⚠ 0.5x | ⚠ 0.21 |

**The dataset contrast is the finding.** On 28 genes the two arms are close on arrangement —
`zinb-flow` reaches 0.6797 against copying's 0.7322 on `marker_field_r`, and 0.6292 against
0.7256 on `marker_depth_r`, neither clearing its fold spread. On 1017 genes the same two metrics
were copying's **largest** margins (0.6614 and 0.5948, both 6.0x+). So the generative path's
spatial arrangement **degrades sharply with panel width** — 0.87x of copying's `marker_depth_r`
at 28 genes, **0.35x** at 1017 — while copying holds up. That is the same "the gap widened on the
wide panel" pattern seen before the fixes, but now on metrics that were measuring something.

#### 3. `text_emb_mode` on 28 genes: nothing is established

Every margin is below 2x its within-arm fold spread — 0.5x, 0.4x, 0.4x, 0.2x, 0.3x. Every sign
still favours `lookup`. The one that looked like a result, `marker_depth_r` at 2.0x the envelope,
is the reason for §4.

Set against `deep_starmap`, where `morans_pearson` cleared at **10.0x** the fold spread: the text
gate's single established real-data result is on the wide panel only, and tier-1 adds nothing to
it either way.

#### 4. The report claimed the opposite of the truth, and now has the statistic to see it

`text_emb_mode`'s `marker_depth_r` margin of 0.0661 has per-fold differences of **−0.1322 and
−0.0000** (4.2e-5). The report printed *"The two folds **agree in sign**, so the gap is not
carried by one of them"* — the sign check passed on a difference of 4e-5 and asserted the
opposite of what the numbers say. The mean of two numbers, one of which is zero, is the other one
halved.

`_fold_balance` = `min |per-fold difference| / max |per-fold difference|` is now a column
(**⚠** below 0.25), and the headline sentence prints it and the per-fold differences instead of
resting on sign agreement alone. Measured on the four audits: tier-1 `expr_mode` is 0.77–0.94 on
the metrics that matter and **0.21** on `marker_depth_r`; tier-1 `text_emb_mode` is 0.15–0.36 and
**0.00**; `deep_starmap` is healthy throughout.

This is the third statistic the n = 2 design has needed — after the within-arm fold spread and
the inertness probe — and the same failure each time: **a check that cannot fail in the way it is
meant to.** Sign agreement over two folds has only four outcomes and cannot express "one fold
carried it". 375 tests pass, `ruff` and `mypy --strict` clean.

### T09 — the tier-1 ceiling corrects R13: saturation is dataset-dependent, and the two sit on opposite sides (2026-08-27)

`reports/t09_depth_ceiling_starmap.md`. `self` = 1.000000 on all four; model-free.

| | tier-1 STARmap (28 genes, ~4.1k cells/section) | `deep_starmap` (1017 genes, 18–39k cells/section) |
|---|---|---|
| split-half R | 0.859 – 0.882 | 0.987 – 0.992 |
| noiseless ceiling √R | 0.927 – 0.939 | 0.994 – 0.996 |
| best copy | 0.713 – 0.784 | 0.929 – 0.986 |
| **headroom over the best copy** | **+0.155 … +0.214 (median 5.2x the envelope)** | +0.010 … +0.064 (median **0.5x**) |
| copying, as a share of the ceiling | 81% | 98% |
| `zinb-flow`, as a share of the ceiling | 67% | 32% |
| `zinb-flow`, as a share of copying | **83%** | **33%** |

**R13 said "the interpolation task is saturated". That is true of `deep_starmap` and false of
tier-1**, and the earlier entry generalised from one dataset. Corrected in `PROGRESS.md` and
`progress/numbers.md`. What actually holds:

* On **`deep_starmap` the comparison is uninformative** — a perfect method could beat the best
  copy by 0.016, half the reproducibility envelope. Its huge `expr_mode` margins (0.6614, 0.5948,
  both 6.0x+) separate the arms on a task where copying is already at **98% of the achievable
  ceiling**. They are real, and they are measuring "copy vs not-copy" where copying is nearly
  perfect.
* On **tier-1 there is 0.175 of genuine room above copying, 5.2x the envelope** — and
  **`zinb-flow` uses none of it.** It sits 0.096 *below* copying, at 67% of the ceiling against
  copying's 81%.

That is a **better-founded negative than either dataset alone.** The saturation defence — "the
task had no room, so we could not show an advantage" — is available for `deep_starmap` and **not**
for tier-1. Where the task has room, the generative path still loses.

**And the reversal is the interesting part.** The arms are *closest* where the task is hardest:
tier-1's `expr_mode` margin is 0.0964 and does not clear its fold spread (⚠0.5x), while
`deep_starmap`'s is 0.5948 at 6.3x. `zinb-flow` reaches 83% of copying on the sparse noisy panel
and 33% on the dense one. Consistent with the previous entry's finding that spatial arrangement
degrades with panel width — and it means **tier-1, not `deep_starmap`, is the informative
reconstruction benchmark**, despite being the smaller dataset.

**The copy identification holds on tier-1 too, more loosely.** `cross-mix` against a model-free
copy of the section `_resample_layout` actually picks (nearest by `|dz|`, ties broken by
`section_id`): `section_3` +0.6930 vs +0.7434 (−0.0503), `section_5` +0.7581 vs +0.7839 (−0.0258).
Same direction and order as `deep_starmap`'s −0.0009 / +0.0099, but not the near-exact match —
expected, since tier-1's profile is far noisier (R 0.87 vs 0.99) and `cross-mix` mixes counts from
several retrieved donors rather than copying one section wholesale, which costs more on 28 genes.
The claim to carry is "`cross-mix` tracks the nearest-section copy", quantified per dataset —
**not** the 0.001 figure on its own.

#### What this does to the plan

* **No wide-gap experiment.** It was proposed to manufacture headroom that tier-1 already has at
  22 µm spacing. Score the reconstruction claim on tier-1 and report `deep_starmap`'s margins with
  the saturation caveat attached.
* **The `exclude_z` screen is no longer the priority** for the same reason.
* **Unchanged: the zero-shot gene run is the only test of the text channel's actual claim.** Every
  `text_emb_mode` measurement remains on fitted genes, where `lookup` memorises and `medcpt` is
  strictly more constrained, and tier-1 added nothing either way (every margin below 2x its fold
  spread).

### T09 — the headroom bootstrap, and a Convention 3 violation it turned up (2026-08-27)

#### 1. `scripts/t09_ceiling_bootstrap.py` — is the reversal trustworthy?

The tier-1-vs-`deep_starmap` reversal rests on a split-half reliability estimated from **28 genes
at ~4 100 cells per section**, and a point estimate at that scale is not self-evidently stable.
This resamples it: per replicate, an independent **cell** subsample (80%, without replacement) of
the target *and* the donor, and a **marker-gene** resample with replacement, then the whole chain
recomputed — R, `sqrt(R)`, the copy correlation, the headroom.

Two copy referents, because they answer different questions: the **operational** donor
(`_resample_layout`'s actual pick — nearest by `|dz|`, ties by `section_id`) and the **best**
available donor (the oracle bound the claim was stated against). `--compare` takes the other
dataset's JSON and reports interval overlap, `P(this > other)`, and a bootstrap CI on the
*difference*, with an explicit **"REVERSAL NOT ESTABLISHED"** verdict when the intervals overlap.

Two biases are stated in the report because **both cut against the conclusion this file exists to
test**: R is estimated by splitting the *subsample*, so Spearman-Brown corrects to the
subsample's size and the ceiling comes out slightly low; and the gene resample captures the
spread of the mean over the *selected* markers, not the variance of the selection itself — real
on `deep_starmap` (32 chosen from 1017), absent on tier-1 where all 28 genes are markers, so
tier-1's interval is the more complete of the two.

**Not run here** — both datasets are on the user's server. `specs/10` is **not** being rewritten
until it returns.

#### 2. What the smoke test found: `hash()` on a `str` is salted per process

Two runs of the bootstrap with the **same `--seed`** disagreed. The cause was in my own script and
then, on grepping, in **the package**:

```
spatialcpav25_gen/train/loso.py:281   gen = np.random.default_rng([int(seed), hash(hidden.section_id) % (2**31)])
spatialcpav25_gen/infer/calibrate.py:1169                        (identical line)
```

`hash()` on a `str` is salted by `PYTHONHASHSEED`, random by default since Python 3.3 — measured
here at 1223376786 / 929189150 / 474838271 for `'section_3'` in three consecutive processes. So
**the same declared seed drew a different stream in every run.** That is a direct Convention 3
violation ("two runs with the same seed must be bitwise identical, and a test asserts it"), and
no single-process determinism test can see it — which is why
`test_resumed_fit_is_bitwise_identical` and friends have always passed.

**What it touches**: `reconstruct_hidden`'s cell and gene subsample, drawn **every metric-aware
training step**, and `_decode_hidden`'s subsample in **calibration**. Both are on the shipping
path. **Scoring is not affected** — `_fold_scores` → `generate_section` never calls `hash()` — so
every audit number measured against a fixed checkpoint stands.

**Fixed**: `data.schema.section_seed()` (`zlib.crc32`, stable across processes and platforms),
used at both sites and in the bootstrap script. `test_section_seed_is_stable_across_processes`
spawns interpreters under three different `PYTHONHASHSEED` values and compares — and asserts in
the same subprocesses that the **builtin** is *unstable*, so the test cannot quietly become
vacuous. 376 pass in 3m01s.

#### 3. This implicates R10's envelope, which is the yardstick for every gate margin

R10 records: *"Fitting one configuration twice — same config, same seed, different process — moved
its scores by up to **0.0120**."* That is precisely the signature of this bug: same seed, different
process, different training stream. The 0.0335 figure is an *across-seed* spread and is less
directly implicated, but it was measured with the bug present and both numbers should be
re-measured now that a declared seed means what it says.

This matters beyond tidiness: **0.0335 is the yardstick every gate margin in this project is
quoted against**, and several verdicts are "inside the envelope". If the envelope shrinks, some of
those become measurable — and the two margins currently sitting closest to it are exactly the ones
the campaign turns on. Re-measuring it is now the cheapest high-value item in T09, and it needs
fits rather than scoring.

**Not claimed**: that the whole envelope is this bug. The across-seed component is real by
construction; only the same-seed cross-process component is explained.

### T09 — the bootstrap lands: the reversal is established, and it exposes a defect in the shipped layout (2026-08-27)

`reports/t09_ceiling_bootstrap_{starmap,deep}.md`. 400 replicates each, resampling cells (80%,
without replacement) and marker genes (with replacement).

| dataset | referent | median headroom | 95% CI | vs envelope |
|---|---|---|---|---|
| tier-1 | best copy (oracle) | **+0.1551** | [+0.1075, +0.2186] | 4.6x [3.2x, 6.5x] |
| tier-1 | operational copy | +0.1833 | [+0.1274, +0.2567] | 5.5x |
| `deep_starmap` | best copy (oracle) | **+0.0160** | [+0.0104, +0.0243] | 0.5x [0.3x, 0.7x] |
| `deep_starmap` | operational copy | +0.0855 | [+0.0489, +0.1351] | **2.6x** |

**Intervals disjoint. P(deep > tier-1) = 0.000. Difference −0.1389 [−0.2017, −0.0884].** The
reversal is established and is now `specs/10` **§0a**, ahead of the additivity contract.

**The bootstrap validates itself**: its medians reproduce the independent point estimates to
±0.007 on every one of the four targets (tier-1 `section_3` +0.1520 vs +0.1581; `section_5`
+0.1618 vs +0.1550; deep +0.0106 vs +0.0101 and +0.0215 vs +0.0210).

#### The nuance that narrows the claim — and is a defect, not a footnote

"Saturated" is true of `deep_starmap` **against an oracle copier**. Against the copy the shipped
configuration actually performs it is 2.6x the envelope, and `deep_starmap`'s `section_5` carries
all of it: **4.1x** against the operational donor, **0.6x** against the best.

`_resample_layout` selects `nearest = min(sections, key=(abs(dz), section_id))`. On that section it
takes **`section_7` at 40.6 µm (`marker_depth_r` 0.8578)** over **`section_3` at 42.0 µm
(0.9739)** — **0.116 of score for 1.4 µm of proximity**, 3.5x the envelope, measured model-free so
it is a property of the rule rather than of any fit.

Recorded as **R14**, not as a baselines note: `resample` is the shipped `layout_mode`, so every
"copying wins" number in the campaign is measured against a copier leaving 2.6x the envelope
unclaimed, and a reader may fairly discount those margins as beating a weak baseline. **The better
rule** is donor selection by **profile correlation** to the target, or by **niche similarity**
(mean retrieval-feature distance) — both computable from the training volume at generation time,
neither needing a fit, and the ceiling instrument already measures the first for every donor pair.

**Deliberately not changed now.** Touching `_resample_layout` would invalidate every number the
coordinate-frame and layout-leak fixes have just stabilised — both audits on both datasets and
both ceiling runs. Order: re-measure the envelope, then fix the donor rule, then re-run the audits
against it.

#### R10 reopened

Three things are now known about the 0.0335 every margin is divided by: it was measured **on the
synthetic fixture**; it was measured **with the `hash()` seeding bug present**, which is precisely
the mechanism behind this row's own 0.0120 same-seed cross-process finding; and it is **one pooled
number applied to six metrics** whose spread this row already recorded as *5x metric-dependent*.
The replacement is a **per-metric** across-seed spread measured on **real data**, and every
existing "inside the envelope" verdict then needs re-reading against its own metric's figure.
`t09_seed_claim.py` already reports per-metric seed spread, so this needs fits and no new code.

### T09 — the envelope re-measured on real data: R10's 0.0120 was entirely the seeding bug (2026-08-27)

Tier-1 STARmap, `expr_mode` gate, seeds 2/3/4 fitted post-`section_seed`, plus seed 2 refitted in
a **separate process**. 6 cold fits at ~62 min each, run three-up.

#### 1. Same seed, separate process: **0.000000**, bitwise

**36 of 36 values identical** — all six metrics, both arms, means and per-fold alike, largest
absolute difference exactly `0`. R10 recorded this quantity at **0.0120** and shipped it as part
of the justification for a 0.0335 envelope.

**That 0.0120 was the salted-`hash()` seeding bug in its entirety**, not run-to-run variation. With
`section_seed` in place and threads pinned to one, a declared seed now means what it says: the
fit is reproducible across processes to the last bit. Every "inside the envelope" verdict in this
project was decided against a figure inflated by a defect.

#### 2. The envelope is neither one number nor a property of the metric alone

| metric | `cross-mix` spread | `zinb-flow` spread | envelope | vs pooled 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | 0.0054 | **0.0574** | 0.0574 | pooled too **small**, 1.7x |
| `gearys_pearson` | 0.0027 | **0.0595** | 0.0595 | pooled too **small**, 1.8x |
| `umap_mixing` | 0.0068 | 0.0190 | 0.0190 | pooled too large, 1.8x |
| `marker_field_r` | 0.0049 | 0.0148 | 0.0148 | pooled too large, 2.3x |
| `marker_depth_r` | 0.0084 | **0.0472** | 0.0472 | pooled too **small**, 1.4x |
| `celltype_localization` | 0.0000 | 0.0000 | 0.0000 | inert (copied layout) |

**A 4.0x range across the six**, and the pooled figure errs in *both* directions depending on the
metric — too lenient on `morans`/`gearys`/`marker_depth_r`, too strict on `umap_mixing`/
`marker_field_r`. It is also **arm-dependent**, which was not anticipated: a copying arm barely
uses the fitted weights (0.0027–0.0084 across seeds) where the generative arm moves 0.0148–0.0595,
**up to 22x more**. A margin inherits nearly all its seed variance from one side.

#### 3. Which verdicts change

**No overall verdict flips**, but one sub-test does and every margin of safety moves:

* **`marker_field_r`'s envelope test flips FAIL → pass.** Its margin (0.0206) was "inside" the
  pooled 0.0335 and beats its own 0.0148. It remains *not established* only because it fails the
  within-arm fold spread (0.1741) — a different and much larger obstacle.
* **`morans_pearson` and `gearys_pearson` weaken sharply**: 3.9x/4.8x against the pooled figure
  becomes **2.2x/2.2x** against their own. They still stand, with far less room. Seed 4 is why —
  its `zinb-flow` arm scored 0.8265 on `morans` against 0.7690 and 0.7914 at the other two seeds,
  so one seed nearly trebled the envelope.
* **`umap_mixing` strengthens**: 4.2x → **7.7x**, and it is now the most secure of the six.
* **`marker_depth_r`** stays not established; its margin moves 0.0543–0.1021 across seeds, a range
  larger than the pooled envelope it used to be judged against.

**`cross-mix` beats `zinb-flow` on `morans_pearson`, `gearys_pearson` and `umap_mixing` at three
seeds, against a per-metric real-data envelope** — the first claim in this project to satisfy
`claim_min_seeds` = 3 on real data. The two arrangement metrics remain unestablished, on the fold
spread rather than the envelope.

#### 4. ⚠️ What this envelope may **not** be used for

It was measured on **tier-1**, for the **`expr_mode`** arms. Applying it to `deep_starmap`, or to
the `text_emb_mode` gate, would repeat exactly the error just diagnosed — a figure measured in one
setting applied to another because it was the only one available. `deep_starmap`'s `expr_mode`
margins (0.2364–0.6614) clear even the widest tier-1 envelope by 4x and are safe on any reading;
its `text_emb_mode` margins are **not**, and stay quoted against the pooled figure with that
caveat until an envelope is measured for them.

### T09 — `text_emb_mode` on `deep_starmap`: two established three-seed negatives (2026-08-27)

Six cold fits at **3.3–3.8 h each** (11 826 – 13 724 s), six-up, ~3.8 h wall. All six config
hashes distinct. Scoring 408–414 s per arm. My pre-run estimate of 2.3 h per fit was low by
1.4–1.7x: the tier-1 → `deep_starmap` ratio is **3.2–3.7x**, not the 2.2x per-step gene scaling I
extrapolated from — the extra factor is the 7x cell count feeding the parts of a step that
`genes_per_step` does not cap.

| metric | seed 2 | seed 3 | seed 4 | mean | own envelope | vs it | verdict |
|---|---|---|---|---|---|---|---|
| `morans_pearson` | −0.1108 | −0.1167 | −0.1358 | **−0.1211** | 0.0246 | **4.9x** | **STANDS** |
| `gearys_pearson` | −0.1126 | −0.1022 | −0.1239 | **−0.1129** | 0.0501 | **2.3x** | **STANDS** |
| `umap_mixing` | +0.0021 | +0.0029 | −0.0105 | −0.0018 | 0.0201 | 0.1x | signs disagree |
| `marker_field_r` | −0.0412 | −0.0291 | −0.0485 | −0.0396 | 0.0197 | 2.0x | fails fold spread |
| `marker_depth_r` | −0.0205 | −0.0106 | **+0.0139** | −0.0058 | 0.0427 | 0.1x | signs disagree |

**`lookup` beats `medcpt` on Moran's and Geary's at three seeds, against an envelope measured in
its own setting.** Signs agree on every seed; `morans`'s fold spread is 0.0169, the tightest in
the campaign, and its margin is 7.2x that. This is the **second** claim in the project to satisfy
`claim_min_seeds` = 3 on real data, and the first about the method's headline novelty.

**It came out stronger than the cautious estimate, which is the point of having measured it.** The
previous entry warned that if tier-1's envelope transferred, `morans` would fall from 3.7x to
2.1x. It does not transfer — `deep_starmap`'s own `morans` envelope is **0.0246**, less than half
tier-1's 0.0574 — so the result is **4.9x**. Borrowing the denominator would have understated a
negative result about our own method.

**`marker_depth_r` is settled and it is nothing.** Per-seed −0.0205, −0.0106, **+0.0139**: signs
disagree, mean 0.1x its envelope, one seed favouring `medcpt`. That is the metric this entire line
of inquiry began from — "`medcpt` beats `lookup` by 5.5x the envelope" — which survived a
coordinate-frame defect, a layout leak, a fold-balance defect and finally three seeds, and is now
conclusively noise.

**Standing position on the text channel**: no positive on real data at any dataset or panel width,
and two established negatives on the wide panel. Recorded as a result, not a disappointment — a
three-seed negative against a self-measured envelope is a stronger statement than most of what
this campaign has produced in the positive direction.

#### The correction this run forced

The report's envelope section asserted: *"A copying arm barely uses the fitted weights, so its
score is nearly seed-invariant; a generative arm's is not."* **False here, and printed anyway.**
Both arms of this gate are `zinb-flow` — only the embedding differs, so there is no copying arm —
and the worse arm **alternates by metric**: `medcpt` on `morans` (0.0246 vs 0.0073), `lookup` on
the other four.

Another sentence that could not fail. The *finding* survives and is reinforced — envelopes are
per-arm, and since the worse arm is **not predictable** it must be measured rather than reasoned
about — but the causal gloss was mine and was wrong. `specs/10` §4.2a now states the observation,
tabulates the worse arm per gate and metric, and says plainly that the mechanism is unexplained.
The aggregator prints a worse-arm/steadier-arm table instead of the claim, and reports a tie as a
tie rather than naming whichever key `max()` happened to return.

### T09 — the zero-shot pre-flight: three leaks, a wrong pre-registration, and the ceiling instrument (2026-08-27)

Three code reads before spending four fits. All three found something.

#### A. `cross-mix` would serve the answer key

```python
donors = model.data.counts[safe.reshape(-1)].todense()   # infer/generate.py:467
```

`model.data.counts` is the **full** training count matrix; `_cross_mix` and `cross_mix_counts`
filter by *cell*, never by gene. A gene held out of training is still emitted verbatim from a
donor cell. **`cross-mix` is not a zero-shot method — it is a lookup of the held-out answer** and
is excluded from the experiment rather than handicapped. This was pre-registered as the most
likely way to get a wrong result and it is real.

#### B. The pre-registered support criterion was not a test of anything

It said `lookup` "has no embedding row for an unseen gene, so it structurally cannot do the task."
**False.** `forward_zero_shot` computes `norm(_text_channel(t) + gamma * distill(t))`, and while
`_text_channel` returns zeros under `lookup`, `distill` is a two-layer MLP over the text vector
trained by `distillation_loss` to regress the learned per-gene residual. Under `lookup` that
residual *is* the whole embedding — so the distill head learns text → embedding and **generalises
to unseen genes through the same MedCPT vectors.**

Both arms reach unseen genes through text; they differ in *how*, and `use_distill` splits each
again — which the docstring already anticipated ("both are reported in the zero-shot table"):

| arm | unseen-gene embedding | what it tests |
|---|---|---|
| `medcpt` + distill | `W·t + γ·distill(t)` | the full claim |
| `medcpt`, pure text | `W·t` | the paper's pure-text arm |
| `lookup` + distill | `γ·distill(t)` | **a real competitor**, not a floor |
| `lookup`, pure text | `norm(0)` = the LayerNorm bias, constant | the true degenerate arm |

**The question is now "does `W·t` add anything over a distillation head that also sees the
text".** That is a real test and a harder one; the two-arm criteria are void.

#### C. A third leak, not pre-registered

`RetrievalIndex` stores `self.expr_pca` — PCs of the **full** expression matrix — and the
`zinb-flow` path conditions on the retrieved neighbours' PCs. A held-out gene contributes to those
32 components, so even `medcpt`/`lookup` would see it through conditioning. The split must be
applied in **three** places and only the first exists: `gene_pool` in training (exists),
`expr_pca` refitted on kept genes (does not), `cross-mix` excluded (a design decision).

#### The instrument, and one thing it corrected on its first run

`scripts/_gene_split.py` builds the split as a recorded, seeded, **stratified** object — quantile
bins on mean expression × Moran's I, an equal share drawn from every cell of the grid, written out
gene-by-gene with the statistics behind it. Stratified on those two axes specifically because
`marker_genes` selects by Moran's I with a detection floor: an unstratified draw could leave the
held-out set with no eligible markers and no measurement at all.

`scripts/t09_zeroshot_ceiling.py` asks the pre-fit question, and **its shape is different from
the reconstruction ceiling's.** There the competitor was a copy of another section, so the
question was room *above copying*; here no arm may copy, so the competitor is the **constant
field** — predict each held-out gene's own global mean everywhere.

⚠️ **That referent is not zero, and I asserted it was before measuring it.** The intuition — a
flat field has a flat profile, so `_safe_r` sees no variance and returns 0 — is wrong:
`soft_depth_profile` divides each bin by the kernel weight it received, so a constant field's
profile tracks **where the cells are** along the depth axis, and cell density is itself laminar.
Measured at **+0.013 … +0.049** on the fixture. Small, but it is the number every arm must beat,
and the docstring's reason for computing rather than asserting it paid off on the first run.

The report answers two questions: is √R on the held-out genes clear of that referent, and is the
held-out ceiling comparable to the kept genes' (an unrepresentative split would not generalise to
the panel). `self` = 1.000000 or the run aborts. Fit-free, so unaffected by all three leaks.
`ruff`, `mypy --strict`, 376 tests clean.

### T09 — the zero-shot split clears, and the four-arm experiment is pre-registered (2026-08-27)

`reports/t09_zeroshot_ceiling_deep.md`, `reports/t09_gene_split_deep.json`. Model-free.
**813 kept / 204 held out** of 1017 (20.06%), stratified 5x5 on mean expression x Moran's I.

| target | side | genes | markers | R | **ceiling √R** | constant field | shuffled | room |
|---|---|---|---|---|---|---|---|---|
| `section_3` | held out | 204 | 32 | 0.9610 | **0.9803** | +0.0416 | +0.0189 | 0.9387 |
| `section_3` | kept | 813 | 32 | 0.9907 | **0.9953** | +0.0218 | +0.0046 | 0.9735 |
| `section_5` | held out | 204 | 32 | 0.9711 | **0.9854** | +0.0299 | −0.0021 | 0.9555 |
| `section_5` | kept | 813 | 32 | 0.9888 | **0.9944** | −0.0310 | +0.0394 | 0.9634 |

**Not saturated, and not marginally.** ~0.94 of room above the referent, against a widest measured
per-metric envelope of 0.0595 — **16x**. Unlike the reconstruction task, which `deep_starmap` was
saturated for, this split has room for an arm to demonstrate something.

**The 99% representativeness is conservative, not flattering.** Held-out markers are the top 32 of
**204** (15.7% of their pool); kept markers the top 32 of **813** (3.9%). A shallower selection
pool yields weaker markers and should *depress* the held-out ceiling — and it does, slightly. That
the held-out ceiling still reaches 99% of the kept genes' means the stratified draw did its job
with a handicap working against it.

**The floor is a band, not a point.** The constant-field referent ranges **−0.0310 … +0.0416**
across the four rows, a spread of 0.0726 — **wider than the widest per-metric envelope measured
anywhere in this campaign**. An arm must clear roughly **+0.042**, not 0. Irrelevant against 0.94
of room, but it must be quoted as a band or the floor will be understated.

**Two incidental confirmations.** The split was ranked on `section_1`, which is *not* one of the
two scored folds — good, but that is `_reference_section` picking the largest section and
happening to avoid them, not a guarantee; on another dataset it must be checked. And **R14
reproduces on a disjoint gene set**: on `section_5`, nearest-by-z takes `section_7` (40.6 µm) over
`section_3` (42.0 µm) at a cost of **0.152** on held-out genes and 0.121 on kept, against the
0.116 recorded on the full panel. The donor-selection defect is not an artifact of one gene set.

#### Pre-registration — the FOUR-arm comparison, stated before any fit

The two-arm criteria are void: they rested on `lookup` being unable to embed an unseen gene, which
is false. **The question is whether `W·t` adds anything over a distillation head that also sees
the text.**

| arm | unseen-gene embedding | role |
|---|---|---|
| **A1** `medcpt` + distill | `W·t + γ·distill(t)` | the full claim |
| **A2** `medcpt`, pure text | `W·t` | the designed channel alone |
| **A3** `lookup` + distill | `γ·distill(t)` | **the real competitor** |
| **A4** `lookup`, pure text | `norm(0)` — one embedding for every gene | degenerate; the leak detector |

**Two fits, four arms.** `use_distill` is a parameter of `forward_zero_shot`, not a training
setting, so A1/A2 share one fit and A3/A4 share the other. My "four arms ≈ 14 core-hours" assumed
four fits and was wrong: it is **2 fits per seed**.

**Primary comparison: A1 − A3 on `marker_depth_r`**, over markers selected within the held-out
pool, on both folds. `marker_depth_r` is primary because it is the metric whose ceiling has been
measured on these genes; the other five are reported with the caveat that theirs has not been.

**Referents**: the constant field (band above) and the shuffled floor. **`cross-mix` is excluded**
— it reads the full count matrix and would emit the held-out genes verbatim.

**SUPPORT** for the text channel requires *all* of: A1 > A3, signs agreeing on every seed and both
folds with fold balance ≥ 0.25; the margin exceeding that metric's own per-arm across-seed
envelope measured from these runs; the margin exceeding the largest within-arm fold spread; **and**
A1 clearing the constant-field band by more than that envelope.

**REFUTATION of the architecture, not the idea**: A1 − A3 inside its envelope while *both* clear
the constant-field band. Text helps, the distillation head alone captures it, and `W·t` is
redundant.

**REFUTATION of the idea**: neither A1 nor A3 clears the constant-field band by more than the
envelope. No route from text to an unseen gene works.

**Void conditions — the run means nothing unless both hold.** A4 must sit inside the
constant-field band: it emits one embedding for every gene and *cannot* distinguish them, so a
score above the band proves a leak (`expr_pca` over the full panel, or an incomplete `gene_pool`)
rather than a result. And the `self` check in the ceiling instrument must still pass on the same
split. A2 carries no threshold and is reported as the paper's purest arm.

### T09 — the `expr_pca` restriction, demonstrated rather than declared (2026-08-30)

The instruction was *"show me the expr_pca restriction works, not that it exists"*. It did not, on
the first attempt, and the way it failed is the finding.

**A fourth leak channel, found by the test that was written to confirm the third.** The pre-flight
had identified the retrieval PCA as a route for a held-out gene into the `zinb-flow` conditioning
and the fix was to zero the excluded rows of the basis, so a held-out gene multiplies zero. That
is correct and it is **not sufficient**. `_normalised_expression` library-size normalises before
the PCA — `x / x.sum(axis=1) * median` — and the sum runs over the **whole panel**. A held-out
gene therefore rescales every *kept* gene's normalised value, and reaches the conditioning
through the denominator with the basis rows still exactly zero. The size factor is now restricted
to the pool too, and `ExpressionPCs` carries the pool so a basis reapplied elsewhere cannot be
normalised on a different footing.

**What the leak was worth.** On the fixture, with the basis correctly zeroed and only the size
factor unrestricted, vandalising the held-out columns moves the retrieval PCs the flow conditions
on by **max 5.78, mean 0.617 standardised units** — the tokens are unit-scaled, so that is 5.8 σ.
Not a rounding effect: it is the conditioning vector changing character.

**The test asserts the invariance, not the mechanism.** `tests/test_retrieval.py::
test_gene_pool_makes_the_pcs_invariant_to_the_held_out_genes` replaces the excluded columns with
garbage at scales 0, 1e3 and 1e4–1e6 and requires the conditioning back **bitwise identical**, at
two levels: `ExpressionPCs.project` on a vandalised matrix, and a whole `RetrievalIndex` rebuilt
from a vandalised volume, compared on `neighbour_tokens` — the object the flow actually reads.
Both leak channels were re-broken one at a time to check the test has teeth: removing the size
factor restriction fails it, and adding 1e-9 to the off-pool basis rows fails it. Had the test
been written against the mechanism ("assert the basis rows are zero") it would have passed
throughout and the campaign would have been spent measuring a leak.

**The same leak was live in `tests/test_expression.py`.** `train_model` passed `gene_pool` to
`train_ctfflow` and not to `TrainingData.build`, so `test_zero_shot_gene_decoding` — the fixture's
own capability experiment — conditioned on PCs fitted over the whole panel, held-out genes
included. Fixed. It is the same shape as the defect that test's docstring already records (the
first `train_ctfflow` accepted `gene_pool` and dropped it, and 0.9235 in-sample was reported as
zero-shot): a pool threaded to one consumer of the volume and not to the next.

### T09 — the four-arm scoring path, and a fifth leak of the same shape (2026-08-30)

Built while the fits run: the arms, the pool-restricted metrics, and the referents.

**`ZeroShotView`** (`model/embeddings.py`). Generation calls `embeddings.gene(arange(G))` and
gets `forward` for every gene — right for a fitted gene, wrong for a held-out one, because
`forward` reads the free residual `r_g` that a held-out gene never received a gradient for. The
view substitutes `forward_zero_shot` for named entities and leaves every other row untouched, so
one fit serves two arms. It is a wrapper, not a branch in `forward`: the fitted path stays
bitwise what every other number in this project was measured on.

**A2 and A4 are algebraically plain generation, and not bitwise.** A gene whose residual is still
its zeros init has `forward` = `forward_zero_shot(t, use_distill=False)`. Measured, the two agree
to **1.3e-6** and not to the bit: the view runs `W` over the unseen rows alone, and a float32 GEMM
at two batch sizes rounds differently — 3e-8 on `W·t`, amplified by the `LayerNorm`. Seen rows are
bitwise identical (0.0). That is a rule for the scorer, not a defect: **all four arms go through
the view**, so the only things differing between two arms are `use_distill` and the fit, never
which code path produced the embedding.

**A fifth leak, the same shape as the fourth.** `section_scores` gained `gene_pool`, and the test
written for it — vandalise the *generated* counts outside the pool, require the pool's metrics not
to move — failed on `marker_depth_r` before it failed on anything else. `_normalised` in
`train/select.py` library-size normalises over the whole panel, so what the model emitted for the
**kept** genes rescales the **held-out** genes' scored values. Since the kept-gene emissions are
exactly what differs between two arms, the held-out-gene score would have carried a kept-gene
difference. Same defect as `_normalised_expression`, in a second function, found the same way and
on the same day. Fixed by restricting the size factor to the pool on both sides.

Then it failed again on `umap_mixing`, which has no per-gene decomposition to subset after the
fact — two clouds are mixed in the space the cells are placed in — so `_mixing` now builds that
space from the pool: columns, size factor and PCA together. `celltype_localization` reads no
expression and is unmoved by any pool; it is reported and is not evidence about a gene split.

**One rule for marker selection, not two.** `marker_genes` took a `pool` argument and
`scripts/_gene_split.markers_within` became a three-line adapter over it. It had been a mirrored
copy of the Moran's-I-plus-detection-floor rule, and a copy of a selection rule drifts from it.

**The referents are computed by the scorer, on its own folds and markers.** The pre-flight's
ceiling report measured a constant-field band too, but under its own marker selection; a band from
one script and scores from another are not comparable, and the band is the thing A1 must clear and
A4 must stay inside. Both referents reuse an arm's own generated layout with its counts replaced,
so they isolate the expression rather than the layout.

**The layout must be identical across the four arms** or the gap is not about the text channel.
The scorer asserts it per fold and aborts; `tests/test_generate.py::
test_zero_shot_arms_share_one_layout_and_differ_only_in_expression` drives the same assertion
through a real fit and a real generation on the fixture, so a regression is caught here rather
than four hours into a campaign.

**What the `expr_pca` leak was worth on the fixture.** `test_zero_shot_gene_decoding`, re-run
either side of the fix: **−0.1596 → −0.1808** on the held-out genes (seen genes 0.9343 → 0.9389).
Both are far below the 0.4 the test asks for and it stays a strict xfail, so the fix changes no
verdict — but the direction is the one the leak predicts, and −0.16 was the number with held-out
genes in the conditioning. 382 fast tests pass; `ruff` and `mypy --strict` clean.

### T09 — the zero-shot fits: six clean cold fits on `deep_starmap` (2026-08-31)

Six cold fits, three seeds x two `text_emb_mode` arms, 2400 steps, one thread each, six-up.

| seed | `medcpt` config | s | `lookup` config | s |
|---|---|---|---|---|
| 2 | `2a947e26e6310658` | 13737 | `4a47030a2417aada` | 14712 |
| 3 | `4ceac85c7eba8d80` | 13907 | `9061b48fc78368b7` | 14113 |
| 4 | `c86cb080be73ac87` | 14020 | `40c90b6e4b71dd04` | 14126 |

**Six distinct config hashes**, so seed and `text_emb_mode` both took and no fit was silently
reused. **One split in all six runs** — 813 kept / 204 held out, seed 7, 5x5 strata on
`section_1` — so the ceiling measured on that split applies to these fits. **The text channel is
identical across arms**: 1017/1017 genes with metadata, 0 bare, 966 with a summary, byte-identical
`A2M` example, so `lookup` differs by having the channel zeroed and not by seeing different text.
The pre-fit basis assertion exits before the model is built, so six completed fits is its evidence
— though it prints nothing on success, and it should print the count of non-zero held-out basis
rows it actually found rather than leaving the reader to infer the check from the absence of a
crash.

**Cost: 84 615 s = 23.5 core-hours**, 3.82–4.09 h per fit (mean 3.92), ~4.1 h wall six-up. I
quoted 3.3–3.8 h from the previous deep campaign and this is over it: **the fifth consecutive
low estimate**. The measured record is now unambiguous and is what to quote — a `deep_starmap`
cold fit at 2400 steps has never come in under **3.8 h** on this box, and 6 fits cost ~23.5
core-hours, not the ~21 approved.

### T09 — two of my referents are not floors, and the aggregator now refuses them (2026-08-31)

The scored seed files came back and the first thing they say is about the **referents**, not the
arms. Both defects are mine, both were established by reproduction rather than by argument, and
both would have turned an unusable band into a published claim.

**The constant field is float32 round-off on `morans_pearson` and `gearys_pearson`.** A constant
field has *exactly* zero per-gene variance after normalisation, so Moran's I and Geary's C are
`0/0`. What comes back is round-off — and it is not random, because it scales with the gene's own
magnitude and therefore correlates with the real per-gene statistic. Measured on the fixture:
normalised per-gene std **exactly 0.0**, generated Moran's I std **3.5e-8**, `morans_pearson`
**+0.2207**. On `deep_starmap` the same referent reads **+0.5326** (`section_3`) and **+0.5073**
(`section_5`), with `gearys_pearson` at **−0.4659** / **−0.4883**. Read naively, every arm "fails
to beat a constant field" on Moran's — a sentence about float32, not about the model.

**The shuffled referent is a no-op on `umap_mixing`.** `_mixing` never sees a coordinate — two
clouds are mixed in expression space — so permuting positions returns the arm's own score. In the
seed files it agrees with A1 to every printed digit *because it is A1*; on the fixture,
`_mixing(gen, real)` and `_mixing(gen[shuffled], real)` are both 0.000444.

So the usable floors are: `shuffled` for the two autocorrelation metrics, `constant_field` **and**
`shuffled` for the two profile metrics, and **none at all** for `umap_mixing`. The primary metric
`marker_depth_r` is one of the two with a working band, so the pre-registration's primary
criterion is unaffected. `scripts/t09_zeroshot_aggregate.py` encodes the mapping and prints
"**none** — …" with the reason rather than a number, so the invalid bands cannot be quoted by
accident.

**`celltype_localization` carries no information here and is excluded from every contrast.** It is
gene-free, and under `layout_mode=resample` the generated layout is identical across all four arms
*and* all three seeds — `0.6743207385784847` on `section_3` and `0.5313570018082987` on
`section_5` in every cell of the design. That is a useful fact in its own right: all across-seed
variation in these runs comes from the expression path, so the envelopes measured here are
**narrower than a full envelope** and should not be quoted as one.

The aggregator's verdict logic was exercised on two fabricated seed sets with known answers before
it saw a real one — a positive (A1 beating A3 by 0.30 at every seed and fold: reports SUPPORT) and
a null (every arm on the band: reports REFUTATION OF THE IDEA, void condition holds).

### T09 — the four-arm zero-shot result on `deep_starmap` (2026-08-31)

Three seeds x two fits x four arms x two folds x two gene pools = 72 scored cells,
`reports/t09_zeroshot_deep.md`. The pre-registration applied exactly as written, plus the one
case it did not name.

#### The primary comparison is UNRESOLVED, and the reason is power, not category

`marker_depth_r` on the held-out genes, **A1 - A3 = -0.0044** against a 0.1273 envelope (0.0x),
signs disagreeing across seeds *and* folds (+0.0444, -0.1161, -0.0155, -0.0622, +0.0222, +0.1010).
Support fails at the first clause and is not close.

| arm | mean | over the +0.0017 band | own across-seed envelope | ratio |
|---|---|---|---|---|
| A1 `medcpt` + distill | +0.0322 | +0.0305 | 0.1273 | 0.24x |
| A3 `lookup` + distill | +0.0366 | +0.0349 | 0.0320 | **1.09x** |
| A4 `lookup`, pure text | +0.0171 | +0.0155 | 0.0710 | 0.22x |

A3 clears the band, A1 does not, so **neither refutation branch applies**: the architecture branch
needs both clear, the idea branch needs neither. Recorded as **UNRESOLVED** rather than rounded
into the nearer branch.

**And that verdict is a coin-flip on a convention.** A1 and A3 sit within 0.004 of each other;
what separates them is that A3's across-seed spread is 0.032 and A1's is 0.127. Under the
worse-arm envelope convention this project uses for every *contrast* (specs/10 §4.2a), A3's
clearance would be 0.0349/0.1273 = **0.27x** and the verdict would read REFUTATION OF THE IDEA.
The pre-registration wrote "that metric's own per-arm across-seed envelope" and did not say which
arm's for a single-arm clearance; 1.09 against a threshold of 1.0 does not survive that ambiguity.

The honest headline is **underpowered**: on the primary metric A1's across-seed envelope is
**4x its own distance above the band**. Three seeds cannot resolve this question, and a range
over three draws does not shrink fast enough for a few more to fix it.

#### The one positive, and it is not on the primary metric

**`morans_pearson` on the held-out genes, A2 - A4 = +0.2999**, envelope 0.0532, **5.6x**, signs
agreeing at 6 of 6 seed-fold cells, fold balance 0.38: **STANDS**. Against the *usable* floor for
that metric (`shuffled`, +0.0382) A2's +0.2729 clears by 5.0x its own envelope.

A2 is `norm(W t)` and A4 is `norm(0)` — one vector for every gene. So MedCPT text alone
reproduces the per-gene Moran's I ordering of genes the model never saw, far above an arm that
cannot tell two genes apart. Head to head against the other route, **A2 - A3 = +0.2514** (2.7x,
6/6 signs) — flagged "one fold carries it" because of a single anomalous cell (seed 2,
`section_5`, where A3 scores +0.1493 against its own range of -0.065..+0.063 elsewhere); 5 of the
6 cells show the full gap.

**The caveat that stops this being a claim**: `morans_pearson`'s ceiling on the held-out genes was
never measured. The pre-registration named `marker_depth_r` primary *because* it is the one metric
whose ceiling on these genes is known (0.9803/0.9854). +0.27 against a floor of +0.04 might be
most of the available room or a tenth of it, and nothing here says which.

#### The sign of the text channel's value flips between seen and unseen genes

Within the same fits, in the same run:

* **kept (seen) genes**: `lookup` beats `medcpt` — A1 - A3 = -0.1330 (5.8x, 6/6 signs, balance
  0.83) and A2 - A4 = -0.1312 (8.3x) on `morans_pearson`, with `gearys_pearson`, `umap_mixing`
  and `marker_field_r` all STANDS in the same direction. The two established three-seed negatives
  reproduce on a third gene pool.
* **held-out (unseen) genes**: `medcpt` pure text beats `lookup` pure text by +0.2999 on the same
  metric.

That is the paper's thesis stated as a measurement — a free lookup table wins where it has a row
and loses where it does not — and it is the first within-run evidence for it in this project. It
rests on one metric with an unmeasured ceiling, and should not be written up as more than that
until the ceiling exists.

#### What the distillation head does

`A1 - A2` on held-out `morans_pearson` = **-0.1533** (3.3x, 6/6 signs, flagged on fold balance
0.06). Adding `gamma psi(t)` to the text channel *costs* 0.15 on the metric where the text channel
works. On the kept genes the same contrast is -0.0008 (0.1x, signs disagree) — the head does
nothing there. The head is not free, and on unseen genes it is actively harmful on this metric.

### T09 — the clearance convention fixed, and the primary re-read under it (2026-08-31)

The ambiguity that made the primary verdict a coin-flip is settled in `specs/10` **§4.2b**, argued
from comparability rather than from outcome, and the aggregator implements it.

**The rule.** A clearance against a referent is read against the **largest across-seed envelope in
the comparison** — every arm plus the referents — on that metric and dataset. One threshold per
metric per experiment. The arm's own spread is reported but does not set its bar.

**Why, and it is not the outcome.** A criterion of this shape is never applied to one arm alone;
it is applied to every arm against a shared band and the verdicts are read against each other.
Per-arm thresholds make those verdicts incomparable: the arm that clears becomes the one that
*varied least*, not the one that scored highest. Here A1 and A3 differ by 0.004 and land on
opposite sides of the line only because A3's spread is 0.032 and A1's is 0.127 — the experiment
would have reported that the steadier arm has a capability the higher-scoring one lacks. A rule
that rewards steadiness over score can be satisfied by being boring. Note also that the new rule
is **stricter for every arm in every future experiment**, including ones this project would rather
pass.

**The primary under the fixed rule.** Shared envelope 0.1273; band +0.0017.

| arm | mean | over band | own spread | shared | ratio |
|---|---|---|---|---|---|
| A1 `medcpt`+distill | +0.0322 | +0.0305 | 0.1273 | 0.1273 | 0.24x |
| A3 `lookup`+distill | +0.0366 | +0.0349 | 0.0320 | 0.1273 | 0.27x |
| A4 `lookup` pure text | +0.0171 | +0.0155 | 0.0710 | 0.1273 | 0.12x |

Neither A1 nor A3 clears: the pre-registered **REFUTATION OF THE IDEA** on `marker_depth_r`. Void
condition **holds** (A4 at 0.12x, no leak). The earlier UNRESOLVED reading is superseded.

### T09 — the `morans_pearson` ceiling instrument (2026-08-31)

`scripts/t09_zeroshot_ceiling.py` gained `--metric {marker_depth_r,morans_pearson}`; the default
path is untouched. The autocorrelation branch is the same design with the per-gene Moran's I
vector in place of the depth profile: split-half reliability, Spearman-Brown, `sqrt(R)`, a shuffled
floor, and the constant field **reported and labelled degenerate** rather than quoted as a floor.

**It reproduces the metric bitwise.** A ceiling computed with a different estimator is a ceiling
for a different metric, so this is checked rather than assumed: scoring a donor section against a
target through `section_scores(gene_pool=...)` and through the instrument gives
**0.915040537526 both ways, |delta| exactly 0**.

**A consistency defect found on the way, and it touches the published ceiling.**
`t09_depth_ceiling.profile` normalised with the **whole-panel** size factor while `section_scores`
now restricts it to the pool. The ceiling and the arms were being computed under two different
normalisations, which also feeds the marker selection. `profile` now takes `pool` (default `None`
keeps the reconstruction-ceiling behaviour identical) and the zero-shot ceiling passes it at all
five call sites, including the marker selection. **`reports/t09_zeroshot_ceiling_deep.json` was
written before this fix and should be re-run**; the `marker_depth_r` ceiling of 0.9803/0.9854 is
expected to move slightly.

**Fixture run** (`self` = 1.000000 on every row or the run aborts): held-out ceilings 0.9724–0.9845
against shuffled floors 0.0569–0.1093, with the constant field at +0.27 to +0.44 — visibly
degenerate, exactly as the round-off account predicts. The instrument works; **it has not been run
on `deep_starmap`**, which needs the campaign machine.

### T09 — descriptor coverage: panel-wide measured, split-wise outstanding (2026-08-31)

Run here against `resources/gene_meta.parquet` and `resources/deep_starmap_symbols.txt`:

| | value |
|---|---|
| symbols in the panel | 1017 |
| in the metadata table | 1017 (100%) |
| with a full name | 1017 (100%) |
| with a summary | 966 (**95.0%**) |
| — of which a **human orthologue's** | 805 (**83%**) |
| bare symbol only | **0** |
| descriptor length, median / min | 566 / 37 chars |

Two things follow. **The text channel is not thin**: no gene falls back to a bare symbol, and the
median descriptor is 566 characters. And **83% of the summaries are a human orthologue's biology,
labelled as such in the descriptor** — so what A2 demonstrates, if the ceiling clears, is that
MedCPT can place a mouse gene from mostly *human* orthologue text. That belongs in the write-up.

**The split-wise number is still outstanding** and needs `reports/t09_gene_split_deep.json`, which
is not in this repo. 51 of 1017 genes carry no summary; under a draw blind to metadata the held-out
204 would contain **10.2 ± 2.8** of them (95% of draws 5–16). A count well outside that range would
mean the stratified draw picked up a metadata bias and A1/A2 were handicapped for a reason
unrelated to the text channel. `scripts/t09_zeroshot_text_coverage.py` reports it in seconds.

### T09 — the ceiling clears: the pure-text channel is the contribution (2026-08-31)

Three model-free runs on `deep_starmap`, no fits. `reports/t09_zeroshot_ceiling_morans_deep.json`,
`reports/t09_zeroshot_ceiling_deep.json`, `reports/t09_zeroshot_text_coverage_deep.json`, all
folded into `reports/t09_zeroshot_deep.md` by the aggregator so no number here is transcribed.

**1. `morans_pearson` has room, and A2 uses a quarter of it.** Ceiling **0.9956** (held-out;
`self` 1.0 or 0.9999999999999998, i.e. float, on every row), shuffled floor **+0.0382**, room
**0.9574**. Against it:

| arm | mean | over floor | vs the 0.0930 shared envelope |
|---|---|---|---|
| A1 `medcpt` + distill | +0.1196 | +0.0813 | 0.87x — **does not clear** |
| **A2 `medcpt`, pure text** | **+0.2729** | **+0.2347** | **2.52x — clears** |
| A3 `lookup` + distill | +0.0215 | −0.0167 | 0.18x |
| A4 `lookup`, pure text | −0.0270 | −0.0653 | 0.70x, *below* the floor — correct for a
gene-blind arm |

A2 reaches **25%** of the room; copying a whole real section reaches 98%. So the capability is
real and three quarters of it is unclaimed — the opposite of the saturation that sank the
`deep_starmap` reconstruction comparison.

**2. The primary's failure is not the metric's fault.** `marker_depth_r`'s held-out ceiling is
**0.9823** with room **0.9806**, and the best arm uses **4%** of it against copying's 96%. The
room is there; the arms cannot reach it. REFUTATION OF THE IDEA stands on that metric.

**3. The distillation head is what breaks the claim.** A2 clears at 2.52x and A1 at 0.87x. The
head does not merely cost 0.1533 on this metric — it takes the arm from clearing the floor to not
clearing it. On kept genes it does nothing (−0.0008, signs disagree). **`W t` is the contribution;
`gamma psi(t)` is not.**

**4. The split is clean, so the result is not an artifact of it.** Held-out 192/204 with a summary
(94.1%) against kept 774/813 (95.2%) — gap −1.1 points, zero bare symbols either side, median
descriptor 546 vs 572 chars. Twelve held-out genes lack a summary where a metadata-blind draw
predicts **10.2 ± 2.8** — dead centre. And 83% of the summaries are a **human orthologue's**, so
A2's demonstration is MedCPT placing a *mouse* gene from mostly *human* text.

**5. The normalisation fix was real, and the two instruments now agree.** Re-running
`marker_depth_r`'s ceiling with the pool-restricted size factor moves the ceiling barely
(0.9803 → 0.9808, 0.9854 → 0.9838) but moves the constant field a lot (+0.0416 → −0.0188,
+0.0299 → +0.0124). The instrument's held-out constant field and the *scorer's* band were
**0.0341** apart before the fix and are **0.0049** apart after — a 7x reduction, which is the
independent check that the two were computed under one normalisation rather than two.

**Status of the positive.** It is a **strong observation, not a claim.** `marker_depth_r` was
pre-registered as primary before any fit; `morans_pearson` is reported because the run's only
positive landed there, and a metric promoted to primary because it produced a result is not a
test. What would make it a claim is a **pre-registered replication on `morans_pearson`**, ideally
on a second dataset. That is now the cheapest high-value thing left in E1.

### T09 — ⚠️ CORRECTION: the degeneracy flag was an assertion, and it was wrong about which metric (2026-08-31)

Raised in review: the JSON carries `constant_field_is_degenerate: true` while `section_5`'s
held-out constant field scores **+0.5780** against A2's **+0.2147** on the same fold — a null model
outscoring the arm. Three questions, answered from the code rather than from memory.

**1. What set the flag?** A hardcoded literal. `side_ceiling_autocorr` returned
`"constant_field_is_degenerate": True` unconditionally, on the strength of a fixture reproduction I
ran by hand. It was never computed per run, so in the JSON it was **indistinguishable from a
measurement** to anyone reading it. That is the defect, and it is mine.

**2. Where was the exclusion written down?** In `VALID_FLOOR` in the aggregator and in the
progress entry of the same day — **not in `specs/10`, and not in the pre-registration**, which
still says "clear the constant-field band" in all three of its outcome conditions. So the record
carried a rule the pre-registration did not know about.

**3. Is there an UNINFORMATIVE condition?** **No.** The pre-registration has SUPPORT, two
REFUTATION branches, the void conditions, and (added later) UNRESOLVED. "Uninformative" is used
elsewhere in this project — `progress/fixture_limitations.md`, R11 — but it is not a condition of
this experiment, so `section_5` cannot trigger it. Stated plainly because the review asked
directly.

#### The flag is now measured, and the measurement contradicts what I wrote

`constant_field_probe` recomputes each referent end to end at **double precision**. A referent that
is a function of the data does not move; one that is a function of summation order does. Both arms
of the probe are genuine recomputations — the first version cast a float32 profile to float64,
which compares a number with itself and would have called everything stable.

| referent, metric | f32 | f64 | drift |
|---|---|---|---|
| constant field, `marker_depth_r` | −0.004104 | +0.043830 | **4.8e-2** |
| constant field, `morans_pearson` | +0.393130 | +0.376074 | **1.7e-2** |
| shuffled, `morans_pearson` | −0.273051 | −0.273050 | 5.9e-8 |
| shuffled, `marker_depth_r` | +0.031811 | +0.031811 | 3.7e-9 |
| a real donor section, `morans_pearson` | +0.963204 | +0.963204 | 2.2e-8 |

**Six orders of magnitude** separate the constant field from every input carrying real variance.
The normalised constant field's largest per-gene std is 7.45e-09, i.e. numerically zero — it holds
no spatial information at all, so whatever the metric returns for it is the metric's behaviour on
a degenerate input.

**And it is worse on `marker_depth_r` (4.8e-2) than on `morans_pearson` (1.7e-2).** My earlier
account had the constant field valid for the profile metrics — "a real, if small, function of cell
density via the bin normalisation" — and invalid only for the autocorrelation ones. **That was
reasoning where a measurement was available, and the measurement says the opposite.**
`VALID_FLOOR` now lists `constant_field` as a floor for **nothing**, on any metric, and `shuffled`
as the floor for the four gene-dependent metrics.

#### What this does and does not change

**It does not change either verdict, and the report now says so under both referents.** The
primary re-read against the `shuffled` floor (−0.0212): A1 clears by +0.0534 (**0.42x**), A3 by
+0.0578 (**0.45x**), neither above the 0.1273 shared envelope — **REFUTATION OF THE IDEA**, the
same outcome as under the pre-registered constant-field band, and the aggregator prints both so
the reader can see the verdict does not hinge on the referent. Void condition holds either way
(A4 at 0.30x). The reason it is robust is arithmetic, not luck: the band's instability (~0.05) is
a quarter of the envelope the clearance is judged against.

**The secondary is unaffected in value** — A2's 2.52x was already computed against the shuffled
floor — **but its justification changes**. It rested on "the constant field is degenerate *for
this metric*"; it now rests on "the constant field is degenerate for every metric, measured", and
the shuffled referent is the floor because it is precision-stable at 1e-8, not because it is what
was left over.

**`section_5`'s +0.5780 is excluded from every floor computation and reported with its drift.** It
is not a null model beating an arm; it is the metric's output on an input with no variance in it.

⚠️ **The probe has run on the fixture, not on `deep_starmap`.** The fixture establishes the
mechanism decisively; the deep numbers are the ones in the write-up, and their drift is unmeasured
until the two ceiling commands are re-run. **Until that re-run the degeneracy of the deep constant
fields is inference from a mechanism, not a measurement on those rows**, and the write-up should
not claim otherwise.

### T09 — ⚠️ the drift threshold failed on real data; the test is now the input (2026-08-31)

The precision probe ran on `deep_starmap` and **did not reproduce the fixture's clean separation.**
The eight constant-field rows drift, sorted:

`0.0042  0.0092  0.0092  0.0111  0.0438  0.0545  0.0738  0.1905`

A **continuum with no gap.** The 0.01 threshold fell between `morans s3 kept` (0.0092, called
stable) and `morans s3 held_out` (0.0111, called degenerate) — two rows of identical construction
on the same section. The flag came back mixed, 5 of 8, and that is a defect in my instrument, not
a property of the data. The fixture's six-order separation (1e-2 against 1e-8) was real and was
**not evidence that a fixed cut would hold on a real panel**; I generalised from it and should not
have.

**The sound test needs no threshold on the output, because it is a test of the input.** A
referent answers "what does this metric return when there is nothing to find?" only if its input
contains nothing to find — measurable without reference to the metric. `input_information` reports
the largest per-gene coefficient of variation across cells, scale-free, with the `shuffled`
referent measured beside it as the control **on the same rows**:

| referent | input CV | verdict |
|---|---|---|
| constant field | **2.609e-07** (float32 epsilon) | no information; not a floor |
| shuffled positions | **1.614e+01** | the panel's full variation; a floor |

Eight orders of magnitude, and **identical for `marker_depth_r` and `morans_pearson`**, as it must
be — it is the input being judged, not the metric. `INFORMATION_TOL = 1e-6` sits an order above
float32 epsilon and seven below anything real, so unlike the drift cut it is not near either side.

**The objection this has to answer, and does.** At double precision `section_5`'s held-out
constant field reads **+0.3875**, not zero — against A2's +0.2147 on that fold. So "it is
round-off" is not established by the value shrinking. The mechanism is that round-off in the
centring step is **one ulp of each value**, so its pattern across genes tracks expression
magnitude at *every* precision, and expression magnitude is what real Moran's I correlates with.
Changing the mantissa changes the number without changing what it is. The input CV is what settles
it: there is no variation in that input to correlate with anything.

**Verdicts unchanged, and now reported under both referents.** Primary against `shuffled`
(−0.0212): A1 0.42x, A3 0.45x, neither above the 0.1273 envelope — REFUTATION OF THE IDEA, the
same as under the pre-registered constant-field band. Void condition holds (A4 0.30x). Secondary
unchanged: A2 at 2.52x over the shuffled floor, 25% of the measured room.

⚠️ **The input CV has been measured on the fixture, not on `deep_starmap`.** The deep run reported
the *absolute* std (7.45e-09 to 2.98e-08), which is float32-epsilon scale but not scale-free. The
next re-run emits `constant_field_input_cv_max` and `shuffled_input_cv_max` per row, and until it
does, the deep constant fields' degeneracy rests on the mechanism plus an absolute std, not on the
scale-free measurement.

### T09 — the referent test, third version: exact, and no threshold at all (2026-08-31)

The deep re-run returned the pre-stated values — constant-field CV 2.39e-07 to 2.95e-07 against
shuffled 27.7 to 63.7, a ratio of ~1e8, all eight rows flagged degenerate and all eight shuffled
rows not. **The disconfirming outcome I stated in advance did not occur.** The CVs are also
identical between the two ceiling files for corresponding rows, which is the consistency check
passing: `input_information` reads the input, not the metric.

**But the number I thresholded was my own arithmetic.** A constant field's normalised input has
**bitwise identical rows** — verified directly: `np.all(x == x[0])` is True, the row totals have
exactly one unique value, and the per-gene std measured in **float64 is exactly 0.0**. The
2.4e-07 came from computing a standard deviation *in float32* over identical values. The test was
correct in its verdict and wrong in its instrument, and a cut was being placed around a rounding
artifact.

**The test is now a boolean**: are all rows of the normalised, pool-restricted input bitwise
equal? True for the constant field by construction — one row broadcast over every cell, one size
factor, one result — and False for real counts. No threshold, no tolerance, no dataset on which it
could differ.

| referent | rows identical | float64 per-gene std | verdict |
|---|---|---|---|
| constant field | **yes** | **0.0** | no variation; not a floor |
| shuffled positions | no | 0.155 | a floor |

`tests/test_select.py::test_a_constant_field_normalises_to_bitwise_identical_rows` pins it in the
suite, including an assertion that the float32 spread is **non-zero**, so the next person tempted
to threshold it sees why.

**Three instruments, two of them thresholds, both failed** — and the failures generalise, so they
are in `specs/10` §4.2c rather than only here: (1) an argument where a measurement was available;
(2) a precision-drift cut that separated by six orders on the fixture and hit a gapless continuum
on real data — *a fixture that separates cleanly is not evidence that a threshold transfers*;
(3) a CV cut around a float32 artifact — *before thresholding a small number, check it is not your
own arithmetic.*

**No verdict has changed at any point in this sequence.** Primary REFUTATION OF THE IDEA (A1
0.42x, A3 0.45x against the 0.1273 shared envelope), reported under both the pre-registered
constant-field band and the shuffled floor; void condition holds; secondary A2 at 2.52x over the
shuffled floor, 25% of the measured room. What changed three times is the justification, which is
the part that had to be right.
