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
