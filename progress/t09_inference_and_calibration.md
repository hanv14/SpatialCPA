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
