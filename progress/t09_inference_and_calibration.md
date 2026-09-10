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
| `test_stack_coherence` | correlation decays with `|dz|`, no spike at a training section's depth | decay confirmed; on-section coherence no more than **`COPY_SPIKE_MAX` = 0.10** ABOVE its neighbours' median (one-sided, as the code asserts; "within" was wrong here — §4.2h) | pass |
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

## PRE-REGISTRATION — replicating `morans_pearson` on a second dataset (written 2026-08-31, before any number)

### Why this replication is required and not optional

**`morans_pearson` is being promoted to primary because it produced a result.** On `deep_starmap`
the pre-registered primary was `marker_depth_r` — chosen before any fit, because it was the one
metric whose held-out ceiling had been measured — and it returned REFUTATION OF THE IDEA. Six
metrics were reported; the only positive landed on `morans_pearson`, which carried no
pre-registered threshold at all.

Selecting the best of six after the fact inflates the effective false-positive rate roughly in
proportion to the number examined, and no correction applied afterwards recovers what
pre-registration would have given. **A metric promoted because it produced a result is not a
test.** This replication is the thing that converts the deep observation into a claim, or fails
to. If it is not run, the correct status of the A2 result in the paper is *observation*, and it
must be written that way — which is how `specs/10` §7 currently carries it.

### Dataset

**`cosmx_nsclc_3d`** — human NSCLC tumour, 960 genes, `raw_counts`, 18 cell types, ~227 k
training cells (`specs/10` §5.4).

Chosen because it varies **tissue, species and spatial organisation** while holding the thing the
metric depends on nearly constant: 960 genes gives ~192 held out against `deep_starmap`'s 204, so
the per-gene correlation is computed over a comparable number of points. That is what makes it a
replication rather than a re-run. `deep_starmap` is mouse cortex and the result could be a
property of laminar organisation; a tumour has none, and if the text channel still places unseen
genes there, the claim is about genes and text rather than about layers.

**Rejected, with reasons.** `merfish_thick_cortex` (254 genes, ~29 k cells) is 15x cheaper but
gives only ~51 held-out genes — a correlation over 51 points has a much wider across-seed
envelope, and it is the same tissue class as `deep_starmap`, so it varies little. It would very
likely trip uninformative condition (c) below. `exseq_breast_cancer` has 1 979 cells with a
minimum of 57 per section, far too few for a per-section kNN Moran's I to mean anything.
`merfish_thick_hypothalamus` gives ~31 held-out genes. **If the cost of `cosmx_nsclc_3d` is
refused, the honest answer is to run no replication and leave the result labelled an observation
— not to run a cheaper dataset that cannot answer.**

### Split, arms, referents

* **Split**: `stratified_gene_split`, seed 7, `frac=0.2`, 5x5 strata on mean expression x Moran's I
  ranked on the volume's largest section — the identical procedure, not the identical genes.
  Written to `reports/t09_gene_split_cosmx.json` and checksummed into the log.
* **Arms**: A1 `medcpt`+distill, A2 `medcpt` pure text, A3 `lookup`+distill, A4 `lookup` pure text.
  Two fits per seed; **all four generated through `ZeroShotView`**, so the only things differing
  between arms are `use_distill` and the fit.
* **Referents**: **`shuffled` only.** The constant field is excluded by `specs/10` §4.2c, and the
  run must confirm on this dataset that its normalised input has bitwise identical rows — if it
  does not, the referent analysis does not transfer and that is uninformative condition (d).
  `best_other_section` is context; **no arm may use it.**
* **Envelope**: the shared envelope of `specs/10` §4.2b — the largest across-seed spread among all
  four arms and the referents, on this metric and this dataset.

### The contrasts, and which one decides

| role | contrast | what it asks |
|---|---|---|
| **PRIMARY** | **A2 − A3** | the two routes to an unseen gene head to head: `W t` against `gamma psi(t)`. Neither reads the free residual, which is zero for a held-out gene either way. |
| confirmatory | A2 − A4 | does text do anything at all, against an arm that emits one vector for every gene |
| confirmatory | A1 − A2 | is the distillation head a cost on this dataset too |

A2 − A3 is primary and A2 − A4 is not, deliberately: A4 is degenerate by construction, so beating
it is the weaker claim, and `specs/10` §7's sentence — "the pure-text path is the contribution and
the machinery around it is not" — is a statement about A2 against A3. On `deep_starmap` A2 − A3
was **+0.2514**, 2.7x the envelope, 6/6 signs, but flagged *one fold carries it* (balance 0.02 from
a single anomalous A3 cell). Pre-registering it as primary sets a test `deep_starmap` only partly
passed. That is intended.

### SUPPORT — the claim replicates. All of:

1. A2 − A3 > 0 on `morans_pearson`, held-out genes, with signs agreeing at **every seed and every
   fold**, and fold balance ≥ 0.25 **at every seed**;
2. |A2 − A3| exceeds the shared envelope;
3. A2 clears the `shuffled` floor by more than the shared envelope;
4. the void condition holds: A4 sits within one shared envelope of the `shuffled` floor.

### PARTIAL — text works, the route does not replicate

A2 − A4 meets criteria 1–4 with A3 in place of A4, but A2 − A3 does not. Reading: something in the
text channel reaches an unseen gene, but `W t` is not distinguishable from `gamma psi(t)` as the
thing that does it. `specs/10` §7's mechanism sentence would have to be withdrawn and the
A2 − A4 claim kept.

### REFUTATION — the deep result does not replicate

A2 does not clear the `shuffled` floor by more than the shared envelope. Reading: the
`deep_starmap` positive was a property of that dataset, or of selecting one metric from six, and
the paper carries a negative on the zero-shot claim entire.

### UNINFORMATIVE — the run cannot answer, decided by these conditions and no others

* **(a)** the model-free held-out ceiling minus the `shuffled` floor is **< 0.50**. *Checkable
  before any fit, and it gates the spend.*
* **(b)** the held-out ceiling is **< 0.80x** the kept ceiling — the stratified draw took
  systematically harder genes. *Also model-free and pre-fit.*
* **(c)** the shared envelope on `morans_pearson` held-out **exceeds 0.2514**, the `deep_starmap`
  A2 − A3 effect. Then the design could not detect an effect the size of the one being replicated
  even if it were present, and neither support nor refutation may be read from it. *This is the
  condition the `deep_starmap` primary failed on `marker_depth_r`, where the envelope was 4x the
  effect, and it is named in advance so it cannot be invoked selectively.*
* **(d)** any validity check fails: `self` != 1.0 in the ceiling instrument; the four arms' layouts
  are not identical; the retrieval PCA basis is non-zero on a held-out row; or the constant field's
  normalised input is not bitwise row-identical on this dataset.

### Order of operations, and the stop points

1. Build the split; run the **model-free ceiling** (`--metric morans_pearson`) and the
   descriptor-coverage check. **Stop and report if (a) or (b) fires — no fits.**
2. **One fit, timed**, before the other five. Cost below is an extrapolation, not a promise.
3. Five more fits; score; aggregate; read the verdict.

### Cost — an extrapolation from two measured points, stated as such

| dataset | training cells | measured per cold fit |
|---|---|---|
| `starmap_visual_cortex` | 16.5 k | **62 min** |
| `deep_starmap` | ~113 k | **3.82–4.09 h** (mean 3.92) |
| `cosmx_nsclc_3d` | ~227 k | **~8 h — extrapolated, unmeasured** |

Cell count has driven the scaling on both measured points more than gene count. At ~2x
`deep_starmap`'s cells and 0.94x its genes, six fits project to **~47 core-hours**, ~8 h wall
six-up, plus scoring. ⚠️ **I have under-estimated fit cost on five consecutive occasions in this
project.** Treat 47 as a lower bound and let step 2 replace it before the remaining five are
committed.

## T09 CLOSE-OUT — what is open, and what this project can honestly claim

> ⚠️ **SUPERSEDED 2026-09-01 by the FINAL CLOSE-OUT at the end of this file.** Kept in place
> because its assessment was right and its spend list is the record of what was chosen; four
> of its five open items have since been resolved or withdrawn, and its headline paragraph
> still carries the `decoder_mu_link` "may be a defaulting error" claim that the addendum
> below it corrects. **Read the final one for current state.**

Written at the request of a direct question: with every capability claim now tested, what is the
headline? This is an assessment, not a summary.

### The answer, plainly

**This is a negative-results paper with a methods contribution, and the methods contribution is
the stronger half.**

Every *comparative* claim the method was built to make has been tested on real data and has come
back negative or unresolved. What survives as positive is two gates measured on a fixture or on a
reconstruction task, one localised diagnostic finding, and one unreplicated result on a metric
that was not pre-registered. What is genuinely new and defensible is a body of work about **how to
evaluate a generative spatial-transcriptomics model** — most of which exists because a measurement
in this project was wrong first and got caught.

That is a publishable paper. It is not the paper the design documents describe.

### The capability claims, and what happened to each

| claim | status | evidence |
|---|---|---|
| Oblique planes reconstruct as well as axis-aligned | **PASSES** | GATE 2: depth-matched parity **0.955**, edge-excluded 0.979, against a ≥ 0.90 criterion |
| A 3D GRF prior controls per-gene spatial autocorrelation | **PASSES, on the fixture** | GATE 1: error ratio 0.130, per-gene r 0.917, monotone in `ell` |
| The learned intensity field places cells better than copying | **REFUTED on real data** | R11: `field` 0.6607 / `hybrid` 0.6692 against `resample` 0.7546 and a 0.7765 copy floor; `resample` ships |
| Generated expression beats copying a real section | **REFUTED, both datasets** | tier-1 `cross-mix` beats `zinb-flow` by 4.6–5.3x the envelope on three metrics; on `deep_starmap` copying wins all five live metrics |
| Text-grounded embeddings help genes the model was fitted on | **REFUTED, two three-seed negatives** | `lookup` beats `medcpt` on Moran's and Geary's at 4.9x / 2.3x their own envelopes; reproduced a third time on the zero-shot run's kept genes at 5.8x |
| Text-grounded embeddings place genes the model never saw | ⚠️ **PARTIAL, replicated on two datasets** | Pre-registered primary `marker_depth_r` refuted (neither A1 nor A3 clears, 0.42x/0.45x). The `morans_pearson` observation was replicated on `cosmx` under criteria fixed in advance: **A2 clears the `shuffled` floor at 2.08x** against `deep_starmap`'s 2.52x and A2−A4 at 1.96x, so *text reaches an unseen gene* has two independent positives; **A2−A3 = +0.0450, 0.22x the envelope**, so *which path in the text channel does it* has none. `specs/10` §7's mechanism sentence withdrawn |
| SEFL improves anything | 🚨 **REFUTED — it makes things WORSE, 3 seeds x 2 arms** | A7 (2026-09-01): the SEFL arm collapses the anatomical field, `i_gen` at **1.6-2.4 %** of target against the off arm's 95.6-97.5 %; five of seven metrics cost, signs agreeing 3/3 at 1.31x-4.68x. `check_collapse` fired **218 times** across the three ON fits from step 250. The method is named for a mechanism that is measurably harmful on real tissue |
| Metric-aware losses improve the metrics they are made of | ⚠️ **NEGATIVE at 1200; UNRESOLVED at the shipped 2400** — row corrected 2026-09-07 | At 1200 the weights lose (rank 3.5 vs 3.0) and cost on every metric. At **2400 they win the selection** (rank 1.0 vs 2.0, four of six metrics) and **ship at 0.5** — so "negative at the shipped budget" was written when the shipped budget was 1200 and did not survive T09 selecting 2400. But the per-metric margins at 2400 are **0.0052 / 0.0101 / 0.0018** on the autocorrelation metrics, all far inside R10's 0.0335 envelope, on **one seed**: won on aggregate rank, not established on any metric. Neither a negative nor a positive |
| The model reproduces gene–gene covariance | **DOWNGRADED to a mechanism claim** | criterion unsatisfiable as stated; 9.316 against an independent-donor 7.783 |

Two things read as positives and should not be oversold. **GATE 2 is a reconstruction result on
held-in sections**, not a demonstration that the generated expression is right — and the expression
path is exactly where every comparison since has failed. **GATE 1 is a fixture result**; R12 is the
same quantity on real data, and there the model carries **15%** of real tissue's structured
between-cell variance against 62%.

### The one substantive mechanism finding

**R12 / `decoder_mu_link`.** The autocorrelation collapse was localised to a single operation — the
count draw loses 0.73 of Moran's I where real tissue loses 0.27 — and then to a candidate cause:
switching the decoder's mean link from `softplus` to `exp` takes the structured share of
between-cell variance from **15.1% to 61.4%** against tissue's 62.2%, with counts Moran's I
+0.1297 → +0.4782 against tissue's +0.4635.

⚠️ **Two sentences that stood here were wrong and are corrected below** (2026-08-31). They said
the `softplus`/`exp` mismatch "may be a defaulting error" and that `exp` "has never been carried
through a full fit and re-scored". **`Config.decoder_mu_link` has defaulted to `exp` since
2026-08-21**, and every real-data audit in this project is dated **2026-08-25 or later**. The
defaulting error was found and fixed before the campaign; the campaign fits used `exp`. See the
close-out addendum.

What is genuinely open is narrower: **the structured share has never been read off a current
real-data fit.** The 15.1% is a tier-1 *pilot* number under `softplus`; the 61.4% came from a
saved model with caveats stated at the time (48 343 emitted cells against a ground truth of
4 187, `sd(log mu)` 0.777 against tissue's 1.213). Neither describes what the shipped decoder does
on the fits the negatives were actually measured from.

### What is genuinely new — the methods contribution

Each of these came out of a measurement that was wrong first, which is why they are stated as
rules rather than observations.

1. **Reproducibility envelopes are per-metric *and* per-arm, and the worse arm alternates**
   (`specs/10` §4.2a). A 4.0x range across metrics and up to 22x across arms; a pooled figure errs
   in both directions depending on which metric is read. Not reported anywhere in this literature.
   The mechanism is unexplained and is stated as an observation.
2. **A clearance against a referent takes the worst envelope in the comparison** (§4.2b). Per-arm
   thresholds make arms' verdicts incomparable and reward the arm that varied least. Found because
   two arms 0.004 apart landed on opposite sides of a line.
3. **A referent is not a floor until its input is shown to carry nothing** (§4.2c). Three
   instruments, two of them thresholds, both failed: a drift cut that separated by six orders on a
   fixture and met a gapless continuum on real data; and a CV cut placed around a float32
   artifact. The working test is a boolean.
4. **A leak taxonomy for held-out-gene experiments.** Five distinct channels, each found by an
   invariance test and several *after* the obvious fix: training batches; the retrieval PCA basis;
   the retrieval PCA's **size factor** (zeroing the basis is necessary and not sufficient); the
   **metric's** size factor; and the copying arm's donor payload. Worth 5.78 standardised units on
   the conditioning vector.
5. **Two silent defects that inverted published-shaped results.** Scoring against plane-local
   coordinates floored three of six metrics while leaving three bitwise identical — half a table
   moving reads as a modelling result. And a missing exclusion made a LOSO fold's "generated"
   section an exact copy of itself.
6. **A seeding bug masquerading as irreducible noise.** `hash(section_id)` is salted per process;
   the "reproducible only to 0.012" envelope was that bug. Post-fix: 36 of 36 values bitwise
   identical. **Every "inside the envelope" verdict in this project had been decided against a
   figure inflated by a defect.**
7. **Ceiling-first discipline.** Measure what the metric could achieve on these rows before
   spending fits. It is what showed `deep_starmap`'s reconstruction task is saturated against an
   oracle copier (0.5x headroom) while tier-1 has 4.6x — so *which dataset a comparison runs on
   decides whether it can say anything*, and this project spent a campaign learning that.
8. **R14: the shipped donor rule costs 0.116.** `_resample_layout` picks by `|dz|` alone and on
   `deep_starmap` pays 0.116 of `marker_depth_r` for 1.4 µm of proximity — so every "copying wins"
   number is measured against a copier leaving 2.6x the envelope on the table. Note this makes the
   negatives **stronger**: a better copier would win by more.

### What remains open, in the order I would spend on it

1. **R4 — the likelihood/fidelity inversion.** Four instances across three heads with one shape: a
   likelihood term reduced by moving explanatory power from the structured component into the
   unstructured one, with nothing in the objective opposing it. This is the single largest open
   question and it is a design decision, not a tuning one.
2. **R12's structured share, read off the existing fits** — see the addendum; it needs no refit
   and it is the cheapest open item in the project.
3. **The `morans_pearson` replication** pre-registered above — the only route by which this project
   gets a positive capability claim.
4. **A7 (SEFL's net contribution)** — currently the paper ships three losses at zero weight and
   cannot say whether the mechanism it is named for does anything.
5. **R14's donor rule** — deliberately not changed yet, because it would invalidate every number
   stabilised by the coordinate-frame and layout-leak fixes.

### The honest headline

> A continuous-field formulation reconstructs oblique planes at 95% of axis-aligned quality. On
> real tissue, every generative component built on top of it — the intensity-field layout, the
> flow-matching expression head, the text-grounded gene embedding — loses to copying a real
> section, and the text channel helps only for genes with no training data, on one metric, pending
> replication. The autocorrelation collapse behind the expression failure is localised to the
> decoder's mean link and may be a defaulting error. The evaluation methodology developed to
> establish these results — per-arm envelopes, referent-validity tests, and a leak taxonomy for
> held-out-gene experiments — is the transferable contribution.

**What would change this assessment**: the replication above returning SUPPORT, which would give
the open-vocabulary claim one dataset-independent leg. (An earlier version of this sentence also
named a `decoder_mu_link` refit; that was based on the two mistaken claims corrected above.)

## ⚠️ CLOSE-OUT ADDENDUM — the `exp` refit is not owed, because it already happened (2026-08-31)

The close-out above put `decoder_mu_link` second on the spend list and said the negatives might
have been fitted under a mis-defaulted decoder. Asked to plan that refit first — correctly, on the
close-out's own logic — I checked the code before planning around it. **The premise does not
hold**, and it was my close-out that put it there.

**What the code and history say.**

* `Config.decoder_mu_link` is **`"exp"`** (`config.py:942`), and `ZINBDecoder`'s docstring records
  it as "**`exp` by default since T10**". There is no docstring/config mismatch to find.
* The default changed in commit `1968c09`, **2026-08-21**: *"decoder_mu_link defaults to exp:
  T06's own revisit condition, met by T10"*.
* Every real-data audit is later: `layout_mode` ships as `resample` **2026-08-25**; the tier-1
  audits **2026-08-26**; the `deep_starmap` audits **2026-08-27**; the envelope re-measurement
  **2026-08-27**; the zero-shot campaign **2026-08-31**.
* `scripts/_starmap_run.BENCH3_KEYS` sets only the four dataset key names and does **not** override
  the link, so every campaign fit took the default.

So the layout comparison, both expression comparisons and the text results were fitted under
`exp`. **No refit is owed and nothing in the close-out's negative column is attributable to a
`softplus` decoder.**

**That is an argument from dates and defaults, and it is settled by measurement, not by me.**
`decoder_mu_link` is inside `Config.content_hash()` — `b6fb1c71844ffe7f` against `079785968f93ec11`
on an otherwise identical config — and `FitCheckpoint` stores the writing run's hash.
`scripts/t09_checkpoint_config.py` rebuilds the campaign config both ways and reports which the
checkpoint matches, or **NEITHER**, in which case the link cannot be read off the hash and the
script says so instead of guessing.

### What is actually open, and it costs no fits

**R12's headline number is stale on both sides.** The **15.1%** structured share is a *tier-1
pilot* measurement under `softplus`; the **61.4%** is from a saved model under `exp` with caveats
recorded at the time (48 343 emitted cells against a ground truth of 4 187; `sd(log mu)` 0.777
against tissue's 1.213). Neither is a measurement of what the **shipped** decoder does on a
**current** real-data fit — and six such fits exist on the campaign machine.

`scripts/t09_structured_share.py` reads it off them: load a zero-shot checkpoint (11–12 s, the
scorer's measured reload), generate each fold, and report `share_shape`, `sd_log_mu`, the emitted
counts' median Moran's I and the real section's, on `t10_chain_diagnostic`'s own estimators so the
numbers are comparable to the record. **Restricted to the kept genes**, because a share computed
over genes that never entered a batch is not evidence about the shipped configuration.

### Pre-registered before the run — what "the shipped decoder is sound" means

Stated now, with the same discipline as the replication, and against R12's own numbers.

**SOUND** — the emission model is not the bottleneck the pilot measured, and R12's expression half
closes: **`share_shape` ≥ 0.50 on every fold and every arm** (tissue 62.2%; the pilot's failing
value 15.1%), **and** retention — the emitted counts' median Moran's I over the real section's —
**≥ 0.50 on every fold**.

**STILL BROKEN** — `share_shape` < 0.30 anywhere. The `exp` link did not carry its saved-model
result into a real fit, R12's expression half stays open, and the emission model is a live
candidate for the negatives after all. **This is the only outcome that would owe a refit**, and it
would be a refit of the *decoder design*, not of the link.

**PARTIAL** — between 0.30 and 0.50, or the two folds disagreeing across that band. R12 stays open
with a sharper number than it has now, and the next step is the decoder design question T06 is
owed rather than another link sweep.

⚠️ **Whatever it returns, it does not reopen the negatives.** They were fitted under the shipped
link; a low share would mean the shipped link is *insufficient*, not that the comparisons used the
wrong one. Those are different claims and the report must not blur them.

### Cost

| step | cost |
|---|---|
| `t09_checkpoint_config.py` over all six checkpoints | seconds, no generation |
| `t09_structured_share.py`, one arm x one seed | one checkpoint load (~12 s) + 2 fold generations |
| the same over three seeds x two arms, if the first disagrees across arms | still minutes |

**No fits. Compare with the 47 core-hours the replication needs**, which is the reason to do this
first — not the reason I gave in the close-out, which was wrong.

### ⚠️ T09 — the structured-share run is VOID: three defects, all in my instrument (2026-08-31)

The run returned `share_shape` **1.2136 / 1.2190** and retention **39.4% / 44.4%**. Under the
pre-registration, ≥ 0.50 on both would read SOUND. **I am not reading it that way, because the
run cannot be read at all.** Three independent defects, none of them the campaign's.

**1. `share_shape` is not bounded by 1, and my thresholds assumed it was.**
`Var(log mu) = Var(shape) + Var(log s) + 2 Cov`. Here the covariance is **negative** — median
−0.018605 against `var_shape` 0.114824 and `var_logsize` 0.016368 — so the total (0.093983) is
*smaller* than `Var(shape)` alone and the ratio is **1.2218**. A "share" above 1 is not a share.
Pre-registering ≥ 0.50 / < 0.30 on a statistic I had not checked was bounded is the **third time
in this session** I have put a threshold on a quantity without first establishing its range
(after the referent drift cut and the CV cut). The pattern is now explicit enough to name: *before
thresholding a statistic, establish its range.*

**2. The decomposition was over all 1017 genes, not the kept 813.** `mu_variance_decomposition`
indexes `arange(len(vol.gene_names))`. The script's docstring and the addendum both claimed the
restriction; only the Moran's I figures had one. A share computed partly over genes that never
entered a batch is not evidence about the shipped configuration.

**3. The Moran's I numbers are on the wrong scale *and* the wrong panel.** Two separate problems:

* *Scale*: I passed raw counts where `t10_chain_diagnostic` rank-normalises before every spatial
  statistic. Measured on the fixture this is small — median I **+0.4315** raw against **+0.3966**
  rank-normalised, a 0.9x factor — so it is an inconsistency, **not** the explanation for the
  numbers below, and I would have mis-attributed it had I not measured.
* *Panel*: `real_morans` came back **+0.0182**, against R12's tissue figure of **0.4635**. That
  0.4635 is a median over STARmap's **28-gene marker panel**, where every gene is spatially
  structured by construction. `deep_starmap` has **1017** genes, most carrying no spatial signal,
  so a median over them sits near zero — and `retention` at 39–44% is then two small numbers
  divided, dominated by genes with no signal on either side. **I transferred a threshold across a
  panel boundary it does not cross**, which is the same error as (1) wearing different clothes.

#### What the run does establish

**`decoder_mu_link` = `"exp"`, config hash `2a947e26e6310658`** — matching the seed-2 `medcpt`
line in the campaign log exactly. The link question is closed by the fit's own output: the weights
behind the `deep_starmap` zero-shot numbers were trained under `exp`. Nothing about the negatives
is attributable to `softplus`, which is what the addendum claimed and this confirms independently
of the date argument.

#### The instrument, fixed

* the decomposition is restricted to the kept pool, as a local function rather than a call to one
  that indexes the whole panel;
* `share_shape` is kept for comparability with R12's record and **flagged** when it leaves [0, 1],
  with the covariance reported beside it;
* **`share_shape_bounded` = `Var(shape) / (Var(shape) + Var(log s))`** is added — in [0, 1] by
  construction, and it answers what the share was meant to answer: is the between-cell dynamic
  range the latent's or the size factor's;
* Moran's I is rank-normalised, and reported **both** over all kept genes and over the
  `metric_marker_genes` most structured genes of the real section — the second being the quantity
  comparable to a marker-panel figure.

#### Re-pre-registration, and what I have already seen

⚠️ **Stated after seeing one run.** What I have seen: `share_shape` 1.21/1.22 (unbounded, void),
`sd_log_mu` 0.304, and near-zero median Moran's I on both sides. I have **not** seen
`share_shape_bounded`, the rank-normalised figures, or anything restricted to the kept pool — the
statistics the criteria below are stated on did not exist when the run was made. That is the
honest position; a reader should discount accordingly and the alternative — quietly restating
thresholds — is worse.

On **`share_shape_bounded`**, over the kept pool, both folds, and on `retention_top` (the
structured-gene retention, the panel-comparable one):

* **SOUND** — `share_shape_bounded` ≥ 0.70 on every fold **and** `retention_top` ≥ 0.50 on every
  fold. The latent carries the dynamic range and the emission keeps half the structure of the
  genes that have any.
* **STILL BROKEN** — `share_shape_bounded` < 0.40 anywhere, **or** `retention_top` < 0.25
  anywhere. R12's expression half stays open and the emission model is a live candidate for the
  negatives.
* **PARTIAL** — anything between, or the folds disagreeing across a band.

The 0.70 / 0.40 bounds are set from the *decomposition's* own arithmetic rather than transferred:
`var_shape` 0.1148 against `var_logsize` 0.0164 is a bounded share of **0.875**, so a decoder
whose dynamic range is genuinely latent-driven should sit high, and 0.40 is the point below which
the size factor is carrying a comparable amount. **These are not R12's 15.1% / 61.4% thresholds
and must not be reported as continuous with them** — that statistic is `share_shape`, which is
void here.

⚠️ **Unchanged: no outcome reopens the negatives.** They were fitted under the shipped link.

### T09 — R12's structured share on a current fit: the latent carries the range, the emission still loses it (2026-08-31)

`reports/t09_structured_share_deep.json`, `medcpt`, seed 2, both folds, `decoder_mu_link="exp"`,
config `2a947e26e6310658` — the seed-2 line from the campaign log. `n_genes_decomposed: 813`
confirms the kept-pool restriction is now real.

| fold | bounded share | `retention_top` | counts I (top-32) | real I (top-32) |
|---|---|---|---|---|
| `section_3` | **0.8681** | **0.2457** | 0.0714 | 0.2906 |
| `section_5` | **0.8499** | **0.3413** | 0.0966 | 0.2830 |

#### The pre-registered verdict, and why the bin is not the finding

By the letter: **STILL BROKEN**. `retention_top` on `section_3` is **0.24568** against the 0.25
cut — it fires by **0.00432, 1.7% of the threshold**, on one fold of one seed, against a
**between-fold spread of 0.0956** on the same statistic. The spread is **22x the margin**. I will
not defend that bin, and reporting it as a categorical result would be the fourth threshold
failure in this session rather than a finding.

The pre-registration also has a structural gap the run exposed: SOUND **ANDs** its two components
while STILL BROKEN **ORs** them, so a split verdict — one component emphatic on the sound side,
the other on the broken side — lands in BROKEN by construction. That is exactly what happened, and
it flattens a two-part answer into one word.

**The substantive reading is the same under any bin, and it is a two-part answer:**

1. **The decoder's between-cell dynamic range is latent-driven, decisively.**
   `share_shape_bounded` is **0.85–0.87** — the latent shape carries ~six times the variance of
   the size factor (`var_shape` 0.104–0.108 against `var_logsize` 0.016–0.018). This confirms on a
   **current real-data fit** what R12 had ruled out on a saved model: the size factor is not the
   problem. The covariance is negative (−0.018), which is why the unbounded `share_shape` reads
   1.21 and why it is void as a statistic.
2. **The emission still loses two thirds to three quarters of the spatial structure.** Over the 32
   most structured kept genes, the model emits Moran's I of **0.071–0.097** where the real section
   has **0.283–0.291**. R12's expression half is **open**, and now localised further: not the size
   factor, not the link — `exp` is in place and confirmed by the fit's own config hash — but the
   **count draw itself**, which is where the pilot's chain measurement put it.

#### ⚠️ The saved-model `exp` result does not reproduce here

R12's record has `exp` taking counts Moran's I to **+0.4782** against tissue's **+0.4635** — a
retention of **~103%**. On this fit it is **25–34%**. Two candidate explanations and no
measurement separating them yet:

* **Dataset.** The 0.4782/0.4635 pair is tier-1 STARmap's **28-gene marker panel**, where every
  gene is spatially structured; this is `deep_starmap`'s top 32 of 813, a different selection on a
  different tissue. The panel-transfer problem that voided the previous run applies to the
  *record's* numbers too, not only to mine.
* **The saved model's own caveats**, recorded at the time: 48 343 emitted cells against a ground
  truth of 4 187, and `sd(log mu)` 0.777 against tissue's 1.213. Here `sd_log_mu` is **0.287–0.295**
  — lower still.

**Until one of those is established, R12's "candidate 1 recovered it" must not be quoted as a
property of the shipped decoder.** It is a property of one saved model on one panel.

#### What is measured next, and why

The `lookup` arm on the same seed and folds. It shares the decoder architecture and differs only
in the gene embedding, so:

* if `share_shape_bounded` and `retention_top` land close to `medcpt`'s, they are **decoder
  properties** and the two-part reading above stands as a statement about the emission model;
* if they differ materially, retention depends on the embedding, and "the emission loses the
  structure" is the wrong attribution.

It also doubles the `retention_top` sample from two fold-values to four — which, given the verdict
turned on 0.00432 against a 0.0956 fold spread, is the cheapest thing that can be done about the
one number the reading is thinnest on. **It is still one seed**, and a spread over four folds of
one seed is not an envelope; the honest ceiling on what this can establish is a direction and a
magnitude, not a categorical verdict.

### T09 — the `lookup` arm: one half of the reading confirmed, the other half withdrawn (2026-08-31)

`reports/t09_structured_share_deep_lookup.json`, config `4a47030a2417aada` — the seed-2 `lookup`
line from the campaign log, `exp` again.

| arm | fold | bounded share | `retention_top` | counts I | `var_shape` | `sd_log_mu` |
|---|---|---|---|---|---|---|
| `medcpt` | `section_3` | 0.8681 | 0.2457 | 0.0714 | 0.1077 | 0.2954 |
| `medcpt` | `section_5` | 0.8499 | 0.3413 | 0.0966 | 0.1043 | 0.2870 |
| `lookup` | `section_3` | 0.8209 | 0.2858 | 0.0830 | 0.0748 | 0.2201 |
| `lookup` | `section_5` | 0.8141 | 0.4534 | 0.1283 | 0.0802 | 0.2199 |

**Part 1 confirmed, and it is a decoder property.** `share_shape_bounded` ranges **0.8141–0.8681**
across two arms and two folds — a spread of **0.0540**, and the two arms are separate trainings
with different gene embeddings. The between-cell dynamic range of `log mu` is latent-driven, at
~5–6x the size factor's variance, and it does not depend on the embedding. This is the question
R12's candidate 2 asked, answered on a current real-data fit.

**Part 2 withdrawn as stated.** I wrote that "the emission loses two thirds to three quarters of
the structure", as a statement about the emission model. `retention_top` ranges **0.2457–0.4534**,
a spread of **0.2077** — nearly **4x** the bounded share's — with two systematic effects:

* **fold**: `section_5` > `section_3` in **both** arms (+0.0956 `medcpt`, +0.1676 `lookup`);
* **arm**: `lookup` > `medcpt` on **both** folds (+0.0401, +0.1121).

So retention is not a property of the emission model alone; it depends on which section is being
reconstructed and on which fit is doing it. **With one seed the arm effect cannot be separated
from fit-to-fit variation** — the two arms are different trainings, and +0.04/+0.11 is well inside
the scale that a seed change moves comparable statistics in this project (the zero-shot campaign's
`marker_depth_r` across-seed envelope was 0.1273). The direction is robust and the magnitude is
not: **every one of the four values is below 0.50, and the lowest is 0.246**, so the emission
loses more than half the structure everywhere — that much stands.

**The verdict is now clearly an artifact of where the cut fell.** STILL BROKEN still fires, and
still on `0.24568` — the **single lowest of four values**, against a range of **0.2077**. The 0.25
threshold sits at the very bottom of the observed distribution. Fourth threshold in this session
placed too close to the data; the difference is that this time the pre-registration named the
statistic before the range was known, which is the failure mode itself and not a lapse in applying
it.

#### An unexplained observation, stated as one

`lookup` has **lower** `var_shape` (0.0775 against 0.1060) and **lower** `sd_log_mu` (0.2200
against 0.2912) — a narrower dynamic range in `log mu` — and **higher** retention on both folds. A
narrower mean retaining *more* spatial structure is counterintuitive and I have no mechanism for
it. Recorded as an observation, following the same rule as the per-arm envelope asymmetry
(`specs/10` §4.2a): what can be claimed is the measurement, not a cause.

⚠️ It is, however, **consistent with an established negative measured by a different instrument**:
in the zero-shot campaign `lookup` beat `medcpt` on `morans_pearson` over the **kept** genes at
0.62–0.65 against 0.48–0.52, 5.8x the shared envelope. Two independent measurements, same
direction — the `lookup` arm emits more spatially structured counts on genes it was fitted on.
That strengthens the established negative rather than adding a new claim.

#### R12's status, restated

**Expression half OPEN.** Retention 0.25–0.45 against a real section's own structure. **Not** the
size factor (bounded share 0.81–0.87 across arms), **not** the link (`exp`, confirmed twice by
config hash). The count draw remains the locus the pilot's chain measurement identified — but
"the emission model" is an incomplete attribution while retention moves 0.21 across folds and
fits.

#### The cheap next step, and it is nearly free

Seeds 3 and 4, both arms — **four more script runs, minutes each, no fits**, since those
checkpoints already exist. That gives 3 seeds x 2 arms x 2 folds = **12 values** and, for the
first time, an **across-seed envelope for `retention_top`**, which is the one number every reading
here is thin on. Only then can the fold effect, the arm effect and fit-to-fit variation be told
apart, and only then is a threshold on this statistic worth placing — after its range is known,
which is the rule this session has now had to learn four times.

### T09 — 12 values, an envelope at last, and R12's stated mechanism contradicted (2026-08-31)

3 seeds x 2 arms x 2 folds, all `exp`, all config hashes matching the campaign log.
`reports/t09_structured_share_deep*.json`.

| statistic | 12 values | medcpt | lookup | across-seed envelope | arm effect |
|---|---|---|---|---|---|
| `share_shape_bounded` | 0.7545–0.8681 | 0.8453–0.8681 | 0.7545–0.8209 | medcpt **0.0026**, lookup 0.0584 | **−0.0718**, 6/6, **1.23x**, balance 0.76 → **STANDS** |
| `retention_top` | 0.2375–0.5266 | 0.2375–0.3413 | 0.2858–0.5266 | medcpt **0.0138**, lookup 0.0847 | **+0.0952**, 6/6, **1.12x**, balance 0.36 → **STANDS** |

**Part 1 holds, robustly.** `share_shape_bounded` is **≥ 0.70 on 12 of 12** and `medcpt`'s
across-seed envelope is **0.0026** — three seeds agreeing to the third decimal. The dynamic range
of `log mu` is latent-driven and the size factor is not the problem. R12's candidate 2 is closed
on a current fit.

**Part 2: the emission retains a quarter to a half, never more.** 11 of 12 values are below 0.50;
the single exception is `lookup` seed 4 `section_5` at 0.5266. The pre-registered SOUND condition
(≥ 0.50 on **every** fold) fails by a wide margin, not a hair.

**The verdict, and why the bin still does not matter.** STILL BROKEN fires on **2 of 12**
(0.2375, 0.2457) — both `medcpt`/`section_3`, a cell whose own across-seed range is **0.0186**, so
it straddles the 0.25 cut inside its own noise. But this time the *SOUND* side fails decisively,
and my two remaining bins say nearly the same thing: STILL BROKEN says "R12's expression half
stays open and the decoder design is the next question"; PARTIAL says "R12 stays open and the next
step is the decoder design question T06 is owed". **The bin does not change the action**, and the
finding below is what the run is actually for.

#### ⚠️ R12's stated mechanism does not survive this measurement

R12 records the mechanism as: *"`mu` is spatially smooth but too flat in **amplitude** for its
structure to survive sampling."* If amplitude is what binds, a larger `sd(log mu)` should retain
more. **The opposite holds, 6 of 6 by seed and fold:**

| | `sd(log mu)` | `retention_top` |
|---|---|---|
| `lookup` (lower amplitude) | 0.189–0.220 | 0.286–0.527, mean **0.3811** |
| `medcpt` (higher amplitude) | 0.283–0.297 | 0.238–0.341, mean **0.2859** |

The arm with **half** the latent-shape variance (`var_shape` 0.053–0.080 against 0.100–0.108) and
the **narrower** mean retains **more** spatial structure, at 1.12x the shared envelope with signs
agreeing everywhere. **The amplitude account predicts the wrong sign.**

That does not refute R12 outright — the two arms differ in more than amplitude, and this is one
dataset — but the simple version of the mechanism is contradicted by the first measurement that
could test it. **R12's mechanism sentence should be marked as unsupported rather than repeated**,
and the question it was answering is back open: if not amplitude, what does determine whether
`mu`'s structure survives the count draw?

⚠️ **Stated as an observation.** I have no mechanism for why less variance in the structured mean
buys more spatial fidelity, and the rule this project has adopted for exactly this situation
(`specs/10` §4.2a) is to claim the measurement and not a cause.

**It is connected to R4's shape** — a model putting more variance into its structured component
and getting less structure out of it is the same trade R4 names, seen from the emission side — but
it is not a fifth instance of R4: R4 is about a *likelihood* improving while fidelity degrades,
and nothing here measures a likelihood. The connection is worth a sentence in the paper and not a
claim.

#### The third independent measurement of the same arm ordering

`lookup` > `medcpt` on emitted spatial structure now appears in three instruments: the tier-1 and
`deep_starmap` `text_emb_mode` audits, the zero-shot campaign's kept-gene `morans_pearson`
(0.62–0.65 against 0.48–0.52, 5.8x), and now `retention_top` (+0.0952, 1.12x). Different
statistics, same direction, three times. **The established negative on the text channel for seen
genes is as well supported as anything in this project.**

### ⚠️ T09 — CORRECTION: the retention arm effect does not stand, and my table hid which envelope it used (2026-08-31)

Raised in review, and correct on both counts.

**The presentation defect.** My table's "across-seed envelope" column showed `medcpt`'s **0.0138**
beside a ratio of **1.12x**. The ratio was computed against the *worse arm's* 0.0847, which is the
right convention — but the table reads as though 0.0138 were the denominator. The arithmetic was
right and the presentation was misleading, which for a number this marginal is the same thing.

**The substantive defect.** There are two defensible constructions of an across-seed spread and I
used one without saying so:

| | effect | (a) fold-mean env | vs it | (b) per-fold env | vs it | verdict |
|---|---|---|---|---|---|---|
| `retention_top`, arm | +0.0952 | 0.0847 | 1.12x | **0.1268** | **0.75x** | **NOT ESTABLISHED** |
| `share_shape_bounded`, arm | −0.0718 | 0.0584 | 1.23x | 0.0596 | 1.20x | stands under both |

`lookup`/`section_5`'s across-seed range is **0.1268** — 9x `medcpt`'s fold-mean spread. Under
construction (b) the retention arm effect is **0.75x and does not stand**. A result that clears
one aggregation of its own noise and not the other is a result about the aggregation.
**`retention_top`'s arm effect is withdrawn.** The rule is now `specs/10` **§4.2d**: build the
envelope at the effect's own aggregation level, report both, and treat disagreement as not
established.

The bounded-share arm effect survives both (1.23x, 1.20x) **and its arm ranges do not overlap at
all** (medcpt 0.8453–0.8681, lookup 0.7545–0.8209). That is what an established difference looks
like beside one that is not.

#### What this does to the amplitude finding — weakened, not withdrawn

I wrote that R12's amplitude account is *contradicted*. That was too strong, because it rested on
the arm effect in retention. What survives, and does not depend on any envelope:

* **`sd(log mu)` separates the arms perfectly**: medcpt 0.2827–0.2972, lookup 0.1893–0.2201, a gap
  of **+0.0626** with **no overlap**.
* **`retention_top` does not separate**: medcpt 0.2375–0.3413, lookup 0.2858–0.5266, with an
  overlap of 0.0555 containing **6 of the 12 values**.
* The sign of the association is **6/6 opposite** to what the amplitude account predicts.

A predictor that separates two groups perfectly, while the outcome it is supposed to govern
overlaps across half the sample, **is not sufficient to explain that outcome** — and the residual
association runs backwards. So: **amplitude does not determine retention.** What cannot be said,
and what I did say, is that lower amplitude *causes* higher retention; that is the arm effect, and
the arm effect is not established.

R12's mechanism sentence — *"too flat in amplitude for its structure to survive sampling"* — is
**unsupported** and should be marked so rather than repeated. The question it answered is open.

### T09 — the three-instrument result, stated plainly (2026-08-31)

**`lookup` emits more spatially structured counts than `medcpt` on genes the model was fitted on.
Three instruments, three different statistics, same direction.**

| instrument | statistic | measurement |
|---|---|---|
| tier-1 STARmap + `deep_starmap` `text_emb_mode` audits | `morans_pearson`, `gearys_pearson` | `lookup` wins at 4.9x and 2.3x their own envelopes, three seeds |
| zero-shot campaign, **kept** genes | `morans_pearson` | 0.62–0.65 against 0.48–0.52, **5.8x** the shared envelope, 6/6 signs, fold balance 0.83 |
| structured-share run, **kept** genes | `retention_top` | +0.0952, 6/6 signs — **1.12x / 0.75x**, not established on its own |

The third is the weakest and is reported as not established; it agrees in direction and adds
nothing to the strength. **The first two are what carry it**, and they are independent
measurements on two datasets and two gene pools. This is the best-supported negative result in the
project: **the text-grounded embedding costs spatial fidelity on genes with training data.**

## HANDOFF TO T06 — R12's expression half: what is known, what governs retention, and what would test it

Ordering decision (2026-08-31): **R12 goes to T06 before the `morans_pearson` replication.** The
amplitude account failing is upstream of every expression result in the campaign; the replication
tests one observation on one metric for 47 core-hours. The replication stays held.

### What R12 now says

Established, on `deep_starmap`, 12 measurements over 3 seeds x 2 arms x 2 folds, all under the
shipped `decoder_mu_link="exp"` and all with config hashes matching the campaign log:

1. **`mu`'s spatial pattern is not the problem.** The pilot measured Moran's I of **0.861** at the
   decoded mean, *above* real tissue's own latent at 0.745.
2. **The emitted counts retain a quarter to a half of it.** `retention_top` — emitted median
   Moran's I over the 32 most structured kept genes, against the real section's — is
   **0.2375–0.5266**, with 11 of 12 below 0.50.
3. **The size factor is not the cause.** `share_shape_bounded` is **0.7545–0.8681**, 12/12 above
   0.70, with `medcpt`'s across-seed envelope at **0.0026**. The dynamic range of `log mu` is
   latent-driven at roughly 5x the size factor's variance.
4. **The mean link is not the cause.** `exp` has shipped since 2026-08-21 and every real-data
   audit postdates it; confirmed per checkpoint by `config_hash`.
5. **Amplitude is not sufficient.** `sd(log mu)` separates the two arms with **no overlap**
   (0.1893–0.2201 against 0.2827–0.2972) while `retention_top` overlaps across half the sample,
   and the residual association runs **backwards**, 6/6.

### What R12 does not say

* **It does not say what governs retention.** The recorded mechanism — *"`mu` is spatially smooth
  but too flat in amplitude for its structure to survive sampling"* — is **unsupported** by (5)
  and must be marked so, not repeated.
* **The saved-model recovery does not reproduce.** The record's `exp` result reads retention
  **~103%** (counts I +0.4782 against tissue's +0.4635); here it is **24–53%**. Unresolved between
  a panel difference (that pair is STARmap's 28-gene marker panel; this is 32 of `deep_starmap`'s
  813) and the saved model's own recorded caveats. **"Candidate 1 recovered it" must not be quoted
  as a property of the shipped decoder.**
* The **arm effect** in retention is **not established** (§4.2d: 1.12x / 0.75x).

### The candidate mechanism, and why it is the next thing to test

For a ZINB draw the emitted counts' spatial autocorrelation is diluted by sampling noise in
proportion to how much of the **count** variance is structured:

```
I(counts)  ~  I(mu) * Var_cells(mu) / ( Var_cells(mu) + E_cells[ Var(count | mu, theta, pi) ] )
```

`I(mu)` is already near-perfect (1), so retention is governed by that **ratio** — which depends on
the learned **dispersion `theta`** and **zero-inflation `pi`** as much as on `Var(mu)`. That is
exactly why amplitude alone fails to predict it: a model can widen `theta` and lose structure at
any `sd(log mu)`.

**This is R4's shape seen from the emission side** — the ZINB NLL at means in the thousands can be
reduced by widening dispersion rather than sharpening `mu`, with nothing in the objective opposing
the trade. R4 is the project's largest open question; this would be its first *measured* instance
in the decoder rather than an inferred one.

### Step 1 — identify the governing quantity. **No fits.**

Compute, per gene and per cell, the structured share of **count** variance

```
s_g = Var_cells(mu_g) / ( Var_cells(mu_g) + mean_cells[ Var(count | mu_g, theta_g, pi_g) ] )
```

median over the top-32 structured kept genes, for each of the **12** existing arm x seed x fold
cells. Then ask whether `s` predicts `retention_top` across those 12.

**Pre-registered, before the measurement:**

* **IDENTIFIED** — Spearman |r(s, retention_top)| ≥ **0.7** across the 12 cells, **and** `s`
  orders the two arms in the same direction `retention_top` does. The governing quantity is the
  structured share of count variance, and step 2 is justified.
* **NOT IDENTIFIED** — |r| < **0.4**. Dispersion and zero-inflation do not explain retention
  either; step 2 is **not** justified and the next move is another measurement, not a refit.
* **AMBIGUOUS** — between 0.4 and 0.7. Report and do not spend.

Cost: a decoder forward pass per fold on checkpoints that already exist. **Minutes.**

### Step 2 — test it causally. Only if step 1 says IDENTIFIED.

**The change**: fix `theta_g` per gene to a **moment-matched** value from the training counts
instead of learning it — one `Config` field, one branch in `ZINBDecoder`. This removes the
optimiser's freedom to make the trade R4 names, which is the cleanest causal test available; a
`theta` *floor* is the softer alternative if moment-matching destabilises the fit.

**Pre-registered, before the change is written:**

* **ANSWERED** — `retention_top` rises by more than the shared across-seed envelope under **both**
  constructions of §4.2d, with signs agreeing on every seed and every fold. Dispersion governs
  retention, and R4 has its first measured instance in the emission model.
* **NOT ANSWERED** — the rise is inside either envelope, or retention falls. Dispersion is not the
  lever; R12's expression half stays open with one more candidate eliminated.
* **UNINFORMATIVE, and checked first** — `I(mu)` on the constrained fit drops below **0.90x** the
  baseline's, or the reconstruction NLL degrades by more than the across-seed spread of the
  baseline's own NLL. Either means the change broke the thing that was already working rather than
  testing what survives it, and no retention number from it may be read.

**Cost.** The six existing checkpoints are **gene-split** fits and are not a clean full-panel
baseline, so this needs its own: **3 baseline + 3 variant fits**, one arm (`lookup`, the selected
config), full panel, `deep_starmap`. At the measured 3.82–4.09 h per cold fit that is
**~24 core-hours**, ~4 h wall six-up — **about half the replication**. ⚠️ Extrapolated from the
same two measured points, and **one timed fit gates the other five**, as with every estimate in
this project since five consecutive ones came in low.

### Why this ordering

Step 1 costs minutes and can stop step 2 entirely. Step 2 costs half the replication and, if it
answers, gives R4 — the project's largest open question — its first measured instance in the
decoder. The replication gives one observation on one metric a second dataset. Both are worth
doing; this one is worth doing first, and step 1 is worth doing before either.

### T09 — step 1 built, and what NOT IDENTIFIED would rule out (2026-08-31)

`scripts/t09_retention_mechanism.py`. Two things verified before it touches the campaign, because
either would have silently measured the wrong quantity:

* **The latent capture is the real one.** `s` must be computed on the *generated* cells or it
  cannot predict a retention measured there. Rather than re-deriving the chain, `_flow_counts` —
  the single point every emitted count passes through — is patched for one call and the capture
  checked against the emitted counts. On the fixture: **1500 latent rows against 1500 emitted
  cells**, and the run aborts if the count is not exactly one or the shapes disagree.
* **The ZINB variance formula matches the sampler.** `Var(X) = (1-pi) mu (1 + mu/theta + pi mu)`
  against a 60-draw Monte-Carlo of `sample_counts` on the same parameters: **correlation
  0.999862**, median ratio 0.984 (the Monte-Carlo low at 60 draws, as expected).

**Retention is reported as the product it is.** `retention_top = mean_vs_real x draw_retention`,
where the first factor is the *generated* conditional mean's Moran's I over the real section's and
the second is what the count draw then costs. This was added before running, and it matters more
than the correlation: **R12's 0.861 — "`mu` is spatially smooth" — was measured on the ENCODER's
latent for a real section, never on the flow's latent at generated positions.** If `mean_vs_real`
comes back low, retention was never a sampling problem and the whole account has been looking at
the wrong stage. One Moran's call, so the run answers something whichever way the correlation goes.

#### Pre-committed: what NOT IDENTIFIED would rule out, stated before the run

Asked for, and right to ask for. If |Spearman r| < 0.4:

**Ruled out.** The dilution account — that retention is governed by the fraction of *count*
variance that is between-cell structure. Under the law of total variance that is the **only**
route by which spatially-independent sampling noise can reduce autocorrelation, so ruling it out
removes dispersion and zero-inflation as explanations, not just as parameters to tune. Combined
with what is already eliminated — the size factor (12/12 latent-driven at 0.75–0.87), the mean
link (`exp`, confirmed per checkpoint), and amplitude (a perfectly separating predictor with an
overlapping outcome and a backwards sign) — **nothing in the emission model's own parameterisation
would remain as a candidate.**

**What would still stand, and it is not a mechanism.** Two possibilities that are not about the
decoder: (a) the generated conditional mean is itself less structured than the real section, which
`mean_vs_real` measures directly in the same run and which would relocate the problem to the flow
or the conditioning; (b) a metric artifact — rank-normalising a sparse integer count vector with
many ties is not the same operation as rank-normalising a continuous mean, and ties can depress
Moran's I independently of any variance share.

**And then the honest answer is that we do not know what governs retention, and the paper says
so.** That is a legitimate result: R12 would be *"the emission loses more than half the spatial
structure of the structured genes; it is not the size factor, not the link, not the amplitude, and
not the structured share of count variance; the mechanism is unidentified"* — four candidates
eliminated by measurement, with the elimination itself being the contribution. **I will not
propose a fifth measurement without being asked**, and a paper that names an unexplained effect
precisely is worth more than one that attributes it to the first surviving candidate.

### T09 — step 1, `medcpt` (6 of 12 cells): the draw dilutes by `s`, and the mean is 2.7x too smooth (2026-08-31)

| seed | fold | `s` | `draw` | `s − draw` | `mean/real` | I(mean) | I(real) | retention |
|---|---|---|---|---|---|---|---|---|
| 2 | `section_3` | 0.0934 | 0.0916 | +0.0019 | 2.6835 | 0.7798 | 0.2906 | 0.2457 |
| 2 | `section_5` | 0.1304 | 0.1251 | +0.0053 | 2.7285 | 0.7723 | 0.2830 | 0.3413 |
| 3 | `section_3` | 0.1007 | 0.0952 | +0.0055 | 2.6883 | 0.7812 | 0.2906 | 0.2560 |
| 3 | `section_5` | 0.1163 | 0.1173 | −0.0009 | 2.6681 | 0.7552 | 0.2830 | 0.3129 |
| 4 | `section_3` | 0.1044 | 0.0895 | +0.0149 | 2.6517 | 0.7706 | 0.2906 | 0.2375 |
| 4 | `section_5` | 0.1181 | 0.1193 | −0.0012 | 2.6984 | 0.7638 | 0.2830 | 0.3219 |

**No verdict yet** — the pre-registered test is over all 12 cells with the arm-ordering condition,
and `lookup` is outstanding.

#### 1. The dilution relation holds, and that is weaker evidence than it looks

`s` tracks `draw_retention` to a maximum absolute difference of **0.0149** and a mean ratio of
**0.960**. `I(counts) ≈ I(conditional mean) × s` describes the sampler to about one part in ten.

⚠️ **But this is close to arithmetic, not discovery.** Given spatially-independent sampling noise,
the law of total variance *makes* the draw dilute autocorrelation by the structured share — it
would hold for any correct ZINB sampler. And because `mean_vs_real` turns out to be nearly
constant here (2.65–2.73, a 2.8% spread), `retention_top` is proportional to `draw_retention`,
which is ≈ `s`. So the pre-registered Spearman test is **close to tautological**: its positive
outcome was near-guaranteed if the sampler is correct.

**That is a defect in my pre-registration**, and it is the same one this session has now produced
five times: a criterion placed on a quantity whose properties I had not established first. The
Spearman number will still be reported, and it should be read as *"the sampler behaves as the
theory says"* rather than as *"the mechanism is identified"*.

**The non-tautological content is the magnitude.** `s` is **0.09–0.13**: only about a tenth of the
emitted count variance is between-cell structure. That is what a 90% loss at the draw *is*, and it
locates the question precisely — in the ZINB conditional variance
`(1−π)μ(1 + μ/θ + πμ)`, whose `μ²/θ` term dominates at these means. **Small `θ` — high learned
overdispersion — is what makes `s` small**, which is exactly what step 2 was designed to test, so
the chain to a causal experiment survives.

#### 2. The generated mean is **2.7x more autocorrelated than the real section**

`mean_vs_real` is **2.65–2.73** on all six cells: I(generated conditional mean) 0.755–0.781
against I(real counts) 0.283–0.291. Not "smooth enough" — **far too smooth**. And `retention_top`
at 0.24–0.34 is the product of a 2.7x overshoot and a 0.09–0.13 draw: **two large errors in
opposite directions that the single retention number had been hiding.**

⚠️ **The first factor is not interpretable as "how good is the generated mean", and I built it
that way.** Its numerator is a **noiseless** quantity — the conditional mean is a smooth function
of the latent, with no cell-level draw in it — and its denominator is **real counts, which carry
the tissue's own sampling noise and are depressed by it**. A ratio above 1 is therefore expected
and says little on its own. The pilot's 0.861-vs-0.745 comparison was between two *latents* and
did not have this problem; mine does.

The referent that would make it interpretable is the real section's **noise-free** Moran's I —
estimable model-free by split-half plus Spearman–Brown, the same construction
`t09_zeroshot_ceiling.py` already uses for the correlation. **I am not proposing to run it**, per
the standing agreement not to chain measurements unasked; it is recorded so the decision is
available and so the 2.7x is not quoted as a finding about the mean's quality. What can be said
without it: the generated mean is smoother than noisy real data, and the loss at the draw is real
and large.

### T09 — step 1, all 12 cells: IDENTIFIED fires, and it does not justify step 2 (2026-08-31)

`Spearman(s, retention_top) = +0.9720` over 12 cells, arms ordered the same way by both. **The
pre-registered IDENTIFIED condition is met.** I am not treating that as a licence to spend, for
two independent reasons.

#### 1. The test had almost no power to fail

| quantity | value |
|---|---|
| Spearman(`s`, `retention_top`) — the pre-registered test | **+0.9720** |
| Spearman(`s`, `draw_retention`) — the arithmetic identity | **+0.9650** |
| `draw/s` ratio across 12 cells | 0.961 ± 0.047, max abs difference 0.0149 |
| spread of `mean_vs_real` | **7.0%** of its own mean (2.6517–2.8456) |

`retention = mean_vs_real x draw`, `draw ≈ s` by the law of total variance for any correct
sampler, and `mean_vs_real` varies by 7%. So **`retention ≈ 2.75 x s` by construction** and a high
Spearman was near-guaranteed. The criterion I wrote tests that the ZINB sampler obeys the law of
total variance — which is worth confirming once, and is not the question.

**Fifth instance of the same error**, and the clearest: a criterion placed on a quantity whose
properties were not established first. The previous four were thresholds too close to the data;
this one is a correlation that could barely have come out otherwise. The rule that would have
caught it: *before pre-registering a test, ask what would have to be true for it to fail.*

#### 2. The premise step 2 rests on is unmeasured, and plausibly false

Step 2 is a **moment-matched `theta`**. Its whole logic is: retention ∝ `s`; `s` is small because
`E[Var(count | cell)]` is large; the `mu^2/theta` term dominates that; so constraining `theta`
raises `s`. **The middle link is not measured anywhere in this run.**

```
E[Var(count|cell)] / mu^2  ~  1/mu  +  1/theta  +  pi
                              ^^^^     ^^^^^^^     ^^
                          Poisson   overdispersion  zero-inflation
```

From `s` and `sd(log mu)` that sum is **~0.68 (medcpt)** and **~0.30 (lookup)** — but nothing here
says which term carries it. **If per-gene means are of order 1, `1/mu` alone accounts for the
whole thing and `theta` is not a lever at all**; the 24 core-hours would buy nothing measurable.
R12's "means in the thousands" was tier-1 STARmap's **28-gene** panel; `deep_starmap` spreads
comparable tissue over **1017** genes, so its per-gene means are plausibly small enough for the
Poisson floor to dominate.

**The check that decides it**: report the three terms of `E[Var(count|cell)]` separately. Same
checkpoints, same forward pass, one more array reduction — minutes, no fits. It is the difference
between a 24-core-hour experiment that can work and one that cannot.

Per the standing agreement I am **not chaining measurements unasked** — this is put as a decision
rather than a plan, because it governs whether an already-approved spend is worth making.

#### What is newly established, under both envelope constructions

**The generated conditional mean is 2.65–2.85x more autocorrelated than the real section's
counts**, on all 12 cells, and the arm difference in that overshoot is the one arm effect in this
analysis that survives both constructions:

| effect | (a) fold-mean | (b) per-fold | ranges | verdict |
|---|---|---|---|---|
| `mean_vs_real`, arm | +0.1307, **4.22x** | **2.16x** | medcpt 2.6517–2.7285, lookup 2.7962–2.8456 | **NON-OVERLAPPING — stands** |
| `s`, arm | +0.0297, 1.11x | **0.57x** | overlap 0.0226 | not established |

So `lookup`'s generated mean is *systematically the smoother of the two*, while its `s` advantage
is not established — consistent with the retention arm effect being withdrawn earlier, and it
means the established arm difference in the emission path is about the **mean**, not the draw.

⚠️ The `mean_vs_real` caveat stands and is not resolved by having 12 cells: its numerator is
noiseless and its denominator carries the tissue's own sampling noise, so a ratio above 1 is
expected and the *level* is not interpretable as mean quality. What the arm comparison uses is the
**difference between two arms against the same denominator**, which is unaffected by that.

## PRE-REGISTRATION — the conditional-variance decomposition (written 2026-08-31, before the run)

**The question**: is `theta` a lever on retention at all, or does the Poisson floor account for the
sampling variance? This decides whether step 2's 24 core-hours can produce a detectable effect.

**The quantity.** For each of the 32 structured kept genes, split the ZINB conditional variance
into its three additive terms and report each as a fraction of the total:

```
E[Var(count|cell)] = E[(1-pi) mu]  +  E[(1-pi) mu^2/theta]  +  E[(1-pi) pi mu^2]
                     f_poisson         f_overdispersion         f_zero_inflation
```

median over the 32 genes, for each of the 12 arm x seed x fold cells.

**The criterion is an upper bound, deliberately.** Removing overdispersion *entirely*
(`theta -> inf`) would take the structured share from `s` to `s' = s / (s + (1-s)(1-f_od))`, and
retention to `mean_vs_real x s'`. A moment-matched `theta` cannot do better than that, so **if the
idealised gain is undetectable, the real experiment certainly is.** It assumes the mean is
unchanged, which is what "is `theta` the lever" means at first order and is stated as the
assumption it is.

Calibrated against the measured across-seed envelopes for `retention_top` — (a) 0.0847,
(b) 0.1268 — at the `medcpt` baseline (`s` 0.1106, `mean_vs_real` 2.70, retention 0.2986):

| `f_overdispersion` | `s'` | retention' | gain | vs the strict (b) envelope |
|---|---|---|---|---|
| 0.10 | 0.1214 | 0.3278 | +0.0292 | 0.23x |
| **0.20** | 0.1345 | 0.3632 | +0.0646 | **0.51x** |
| 0.30 | 0.1508 | 0.4073 | +0.1087 | 0.86x |
| **0.50** | 0.1992 | 0.5378 | +0.2391 | **1.89x** |
| 0.70 | 0.2930 | 0.7912 | +0.4926 | 3.88x |

The idealised gain crosses the strict envelope between `f_od` 0.2 and 0.3, so the two thresholds
below bracket *"undetectable even if `theta` were removed entirely"* and *"comfortably
detectable"*. **The thresholds are set from the envelope, not from where the answer is expected to
fall** — which is the failure the previous four pre-registrations made.

### THETA IS A LEVER

`f_overdispersion` >= **0.50** on **every** cell. Constraining `theta` targets the majority of the
sampling variance, and the idealised gain is ~1.9x the strict envelope, so a real moment-matched
fit has room to produce a detectable effect. **Step 2's 24 core-hours are justified.**

### THETA IS NOT A LEVER

`f_overdispersion` < **0.20** on **every** cell. Then even eliminating overdispersion completely
moves retention by less than **0.51x** the envelope it would have to clear. Step 2 **cannot**
produce a detectable effect and must not be run.

### AMBIGUOUS

Anything else — including cells falling on both sides of either threshold. Report and do not spend.

**What I will do on NOT A LEVER**: state what it rules out, and stop. **No substitute experiment
will be proposed in that message.** Option 3's open state is an acceptable ending and reaching it
deliberately is better than reaching it after one more measurement.

### T09 — the decomposition: THETA IS A LEVER, and what that verdict does and does not say (2026-08-31)

| | 12 cells | threshold |
|---|---|---|
| `f_overdispersion` | **0.5677–0.6143** | ≥ 0.50 on every cell — **met, minimum clears by +0.0677 = 1.5x the across-cell spread** |
| `f_poisson` | 0.2896–0.3359 | — the floor I expected might dominate. It does not. |
| `f_zero_inflation` | 0.0795–0.1014 | — |
| idealised gain if `theta -> inf` | **0.2848–0.5364** | **2.25x–4.23x** the strict per-fold envelope |

**Verdict: THETA IS A LEVER.** No cell is near the boundary and the reported gain reproduces
independently (medcpt seed 2 `section_3`: `s'` 0.1995, gain 0.2847 against the script's 0.2848).

The medians of the three fractions sum to 0.983–1.002 rather than exactly 1. That is the median
operation, not an error — a median of sums is not a sum of medians; per gene they sum to 1 to
**1.8e-06** (fixture check).

**My stated worry is disconfirmed.** I argued the Poisson floor might account for the whole
sampling variance if `deep_starmap`'s per-gene means were of order 1, in which case `theta` would
be no lever and step 2 would buy nothing. `f_poisson` is **0.29–0.34**. The means are high enough
that overdispersion dominates, and the concern is answered by measurement rather than left as a
caveat.

#### ⚠️ What A LEVER does **not** say, and the asymmetry is in my own design

The criterion is an **upper bound**: it assumes overdispersion is removed *entirely*
(`theta -> inf`) and the conditional mean is unchanged. That construction was chosen so a
**negative would be rigorous** — "even in the ideal case the gain is undetectable" rules the
experiment out. **The positive is therefore only permissive**: it says the experiment is *not
ruled out*, with a gain somewhere in `[0, 2.25x–4.23x the envelope]`.

A moment-matched `theta` reaches `theta -> inf` nowhere. What it actually buys depends on how far
the *learned* `theta` sits from the moment-matched one — and if the model has already learned
something close to it, the change does nothing. **That comparison belongs inside step 2 as a
diagnostic it reports on the first fit, not as a new gate before it**: the pre-registration said A
LEVER justifies the spend, the user pre-accepted that, and adding a condition now — after a result
I like — would be the same discipline failure as a threshold moved after seeing the data, in the
direction that flatters the plan.

So: **step 2 is justified as pre-registered, and its first fit should report `theta_learned` vs
`theta_moment_matched` so a null result can be told apart from a null change.**

#### R12's state after step 1

The expression half is now localised as far as measurement without a refit can take it:

* `mu`'s spatial pattern is **not** the problem — the generated conditional mean is 2.65–2.85x
  *more* autocorrelated than the real section's counts (with the noiseless-numerator caveat).
* The loss is at the **count draw**, which retains only 9–19% — because only 9–19% of the emitted
  count variance is between-cell structure.
* That share is small **because of learned overdispersion**: `mu^2/theta` carries **57–61%** of the
  conditional variance, against 29–34% for the Poisson floor and 8–10% for zero-inflation.
* Eliminated by measurement: the size factor, the mean link, and amplitude.

**This is R4's trade, measured in the emission model for the first time.** The decoder reduces its
ZINB likelihood by widening dispersion rather than sharpening the mean, and the spatial structure
that the mean carries almost perfectly is then thrown away by the draw. R4 has had four inferred
instances; this is the first where the mechanism is decomposed and the responsible term named.

### T09 — the `medcpt` decomposition files, reviewed (2026-08-31)

Three files, six cells. They reproduce the summary table exactly. Two things the summary does not
show.

**1. The re-run is a reproduction check, and it passes.** 29 of 30 previously-reported values are
**bitwise identical** to the pre-decomposition run. The single difference is `s_max` on seed 4
`section_5`: `0.4272283911705017 -> 0.4272284209728241`, **delta 3.0e-08**. Cause: adding the
decomposition regrouped the arithmetic from `(1-pi)·mu·(1 + mu/theta + pi·mu)` to the three terms
summed separately — algebraically identical, so only the last bits of one gene's share moved.
Every reported figure is unchanged at the precision it is quoted to. `real_morans_top` is also
identical across all six cells per fold (0.2906032145674943 / 0.28304612687800995), as it must be:
it is a property of the real data and a fixed gene selection.

**2. The per-gene `s` spans two orders of magnitude, and the median hides it.**

| seed | fold | `s_min` | median | `s_max` | ratio |
|---|---|---|---|---|---|
| 2 | `section_3` | 0.0036 | 0.0934 | 0.4279 | **119x** |
| 2 | `section_5` | 0.0093 | 0.1304 | 0.4278 | 46x |
| 3 | `section_3` | 0.0042 | 0.1007 | 0.4208 | **100x** |
| 3 | `section_5` | 0.0076 | 0.1163 | 0.4266 | 56x |
| 4 | `section_3` | 0.0040 | 0.1044 | 0.4232 | **106x** |
| 4 | `section_5` | 0.0077 | 0.1181 | 0.4272 | 55x |

"The draw retains ~10%" is a **median over genes whose structured shares differ by 50–120x**. Some
genes emerge at `s` 0.43 and some at 0.004. That belongs in any write-up of this result: the
emission does not lose structure uniformly, it loses nearly all of it for some genes and little
for others, and nothing here says which genes or why.

**Does the median distort the projection?** Checked rather than assumed. `s' = s/((1-f) + s·f)` is
**monotone increasing in `s`**, so `median(s') = s'(median s)` *exactly*. The gain `mv·(s' − s)` is
monotone up to `s = 0.391` and only the handful of genes above that are past the turnover, so
`median(gain) ≈ gain(median s)` to good accuracy. **The median-based projection is sound in the
`s` dimension.** It is approximate only insofar as `f_overdispersion` varies across genes *and*
correlates with `s` — which is not measured, and is the one way the 2.25x–4.23x could be off.
Recorded as a limitation of the projection, not proposed as another measurement.

Awaiting the three `lookup` files before anything further.

### T09 — the `lookup` decomposition files, reviewed; the LEVER verdict is arm-independent (2026-08-31)

**1. Reproduction, with a wrinkle worth stating.** For `lookup`, **4 of 6** `s` medians moved from
the pre-decomposition run, by **3.7e-09 to 7.5e-09**; `medcpt` had **0 of 6** move (one `s_max`
did). Same cause — regrouping the conditional variance into three summed terms — and the
difference is only which gene happens to sit at each median. Across all 12 cells **5 values moved,
none by more than 7.5e-09**, and nothing changes at the precision anything is reported to. Every
other field (`retention_top`, `mean_vs_real`, `draw_retention`, the Moran's figures, `s_min`,
`s_max`) is bitwise identical.

**2. `f_overdispersion` is arm-independent, and that strengthens the verdict.**

| | mean | range |
|---|---|---|
| `medcpt` | 0.6003 | 0.5866–0.6140 |
| `lookup` | 0.5961 | 0.5677–0.6143 |
| difference | **−0.0041** | ranges overlap almost entirely |

The two arms are separate trainings with different gene embeddings and they put the *same
fraction* of their conditional variance in overdispersion. **The overdispersion share is a
property of the decoder, not of the embedding** — so THETA IS A LEVER is not an artefact of one
arm, and step 2 can be run on either.

**3. `lookup` lifts the floor, not just the median.** Its per-gene `s` spans **26x–65x** against
`medcpt`'s **46x–119x**, and its `s_min` is **0.0052–0.0157** against `medcpt`'s 0.0036–0.0093 —
roughly 2x higher. So `lookup`'s *worst-retained* genes are meaningfully better retained, which is
a different fact from its higher median and was not visible in any earlier statistic.

**4. A fold effect in `f_overdispersion`: `section_5` higher by +0.0207, 6/6.** `section_5` also
has the higher `s` and the higher retention, and `Spearman(s, f_overdispersion)` across the 12
cells is **+0.4965**.

⚠️ **That correlation is not the one I flagged as a threat to the projection.** The concern
recorded for the `medcpt` files is a **per-gene** correlation *within* a cell, which would make a
median-of-parameters projection wrong. This is an **across-cell** association, and each cell's
projection uses its own `s` and its own `f_od`, so it is harmless. Stated explicitly so the
+0.4965 is not later misread as the defect. And there is no contradiction in `section_5` having
both a higher `f_od` and a higher `s`: `f_od` is a fraction *within* the sampling variance, `s`
compares that variance to the structured one — independent dimensions.

**Nothing here disturbs THETA IS A LEVER.** All 12 cells sit at `f_overdispersion` 0.5677–0.6143,
the minimum clears 0.50 by 1.5x the across-cell spread, and the fraction is the same in both arms.

### T09 — step 2 built: `Config.decoder_theta_mode`, and three things recorded first (2026-09-01)

#### 1. The closure question was a reader's error, not an instrument defect

I had raised the three fraction medians summing to 0.983–1.002 rather than to 1 and the user had
classed it with the drift threshold of §4.2c. **It does not belong there and the record should say
so.** §4.2c's threshold was a *defect in an instrument I built*: a cut placed at 0.01 that
separated by six orders of magnitude on the fixture, failed on real data, and fell between two
identically-constructed rows — the instrument was wrong and had to be replaced twice. The closure
question is the opposite: the instrument is correct and the reading of it was not. `f_poisson`,
`f_overdispersion` and `f_zero_inflation` sum to 1 **per gene**, to 1.8e-06 on the fixture check;
the reported numbers are per-gene *medians*, and a median of a sum is not the sum of the medians.
Nothing was wrong with the decomposition, the JSON, or the code that produced them.

The distinction matters for the write-up because the two failures have different remedies. A
defective instrument has to be rebuilt and the numbers it produced discarded. A misread output has
to be labelled — which is done: the summary prints the fractions as medians, and this paragraph is
the note a reader needs. **Filed as a reader's error. It is not a fifth entry in §4.2's list, and
the four envelope rules plus the fifth (ask what would have to be true for the test to fail) stand
at five, not six.**

#### 2. Per-gene `s` spans 46–119x within a single cell — an open question in its own right

Not a footnote to the aggregate. `retention_top` and every projection built on it use
`median_g(s)`, and that median summarises a distribution whose ends differ by **two orders of
magnitude**: `medcpt` spans 46x–119x per cell (`s_min` 0.0036–0.0093 against `s_max` up to 0.43),
`lookup` 26x–65x with an `s_min` roughly 2x higher.

**"The draw retains ~10%" is therefore false of almost every individual gene.** Some genes come
through the emission with 43 % of their between-cell structure intact and some with 0.4 %, and
**nothing measured in this campaign says which genes, or why.** The obvious candidates — mean
expression, detection rate, the learned `theta`, the gene's own real-tissue Moran's I — are all
unmeasured against `s`. That is a question about what the decoder does to a gene, and it may
matter more than the aggregate: a method that retains structure for the 30 genes an atlas is read
on and loses it for the other 800 is a different method from one that loses 90 % of everything,
and the two are indistinguishable in every number reported so far.

It is recorded here as **R12's second open question**, alongside the amplitude account's failure,
and not as a proposed measurement. Whether to spend on it is a separate decision from step 2.

#### 3. `f_overdispersion`'s arm-independence is the strongest form of R4 the project has

R4 — likelihood improving while distributional fidelity degrades — has had four inferred
instances. Step 1 gave it a decomposed one: the decoder pays for its ZINB likelihood by widening
dispersion rather than by sharpening the mean, and `mu^2/theta` carries 57–61 % of the conditional
variance that then destroys the spatial structure `mu` had almost perfectly.

**The arm-independence is what makes it a claim about the decoder rather than about a fit.**
`medcpt` 0.6003 (0.5866–0.6140) against `lookup` 0.5961 (0.5677–0.6143): two separate trainings,
different gene embeddings, different text channels — one of which works and one of which is a
`norm(0)` void — and the same fraction of conditional variance in the same term. An arm-dependent
number would have been a property of one embedding and would have supported nothing general. This
one says the trade is structural to the ZINB emission model under this objective, which is R4's
own claim, stated for the first time about a named term with a measured share.

⚠️ **Two arms is two, and both are `deep_starmap` at one `decoder_mu_link`.** The claim is
"arm-independent", not "dataset-independent" or "objective-independent", and the write-up must not
round the first up to the others.

#### 4. What was built

`Config.decoder_theta_mode` (`"learned"` default | `"moment_matched"`), one branch in
`ZINBDecoder`, and `scripts/t09_theta_mode.py`.

**The plumbing decision, because it touched seven call sites.** A fixed per-gene dispersion has to
be *indexed* by gene, and `ZINBDecoder.forward` receives gene **embeddings**, not panel columns —
that is the open-vocabulary property and it is deliberate. Three options: thread an optional
`gene_idx` through every decoder call; derive `theta` from the embedding through a frozen map; or
impose a penalty instead of a fixed value. The second is indirect enough that a null result would
be uninterpretable and the third is not what was pre-registered, so `gene_idx` is threaded — an
optional trailing argument, **ignored under `"learned"`** (a test asserts the two calls are
bitwise equal) and **required under `"moment_matched"`**, where its absence raises rather than
reaching for a default.

**`head_theta` is still constructed under `"moment_matched"` even though nothing reads it.** Not
an oversight: `_init_linear` consumes the generator in a fixed order, so removing the head would
shift the draw for `head_pi` and the two modes would differ by more than the dispersion. A test
asserts `mu` and `pi` are **bitwise identical** between the two modes on the same inputs.

**What is under test is not the value, it is the freedom.** The mechanism is the removal of the
*per-cell* degree of freedom — the learned head can widen dispersion cell by cell to absorb
structure it failed to predict, and a single dispersion per gene cannot, so any cell-to-cell
variation then has to be carried by `mu`.

#### 5. The estimator was wrong, the fixture said so, and the correction runs the way I wanted

The obvious estimator is the **marginal** moment `theta_g = m_g^2 / (v_g - m_g)` on
size-factor-normalised counts, and that is what I wrote first. Its bias is elementary and I
recorded it in the field's docstring **before running anything**: `v_g` marginally contains every
between-cell difference the model is supposed to *predict*, so it over-states the variance and
under-states `theta`.

The synthetic fixture then measured what that costs. Median `theta` over the 200-gene panel:

| estimator | median `theta` | 10-90% |
|---|---|---|
| marginal | **0.213** | 0.092-0.446 |
| pooled within cell type | **0.469** | 0.197-0.900 |
| the learned head, at 40 steps | 0.371 | 0.296-0.542 |

**The marginal estimate is 1.7x *more* over-dispersed than what the head had already learned.**
Pinning `theta` there does not remove the trade under test, it *deepens* it — and a decoder pinned
there emits a median per-gene detection rate of **0.286** against the training sections' 0.522,
tripping `assert_detection_rate` on a path that had nothing to do with this change. That guard,
written for a different failure at T06, is what turned an argument into a measurement.

The estimator is now the same moment match **pooled within cell type** —
`theta_g = sum_k n_k m_gk^2 / sum_k n_k (v_gk - m_gk)`, which reduces *exactly* to the marginal
form when there is one group, so the two are one function rather than two. It lands at 1.27x
*less* over-dispersed than the learned head: a constraint rather than a handicap.

⚠️ **The correction runs in the direction that flatters the hypothesis, and that has to be said
first, not buried.** What makes it a correction and not a thumb on the scale:

1. the direction was derived from the estimator's algebra and **written down before** the fixture
   measured it — it is a prediction that came true, not a rationalisation offered afterwards;
2. the pooled form is simply closer to the quantity `theta` denotes — the dispersion of counts
   *given the cell's state* — and conditioning on cell type is the coarsest honest approximation
   to that state available without a model;
3. it is **still a lower bound**: within-type spatial variation stays in the estimate, so the
   fixed value is if anything still too over-dispersed;
4. the first fit's diagnostic **reports both vectors** against the learned head, so the size of
   the choice sits in the same table as the result it feeds. A test asserts the model uses the
   conditional one and that the two differ.

A referee is entitled to read this as the estimator being chosen for its answer. The record above
is what I have against that reading, and it is deliberately stated as a defence rather than as a
settled point.

**The diagnostic is reported, not thresholded.** The first fit prints `within_gene_sd_log` (how
much per-cell freedom the constraint removes), `log_ratio_median` (how far the learned dispersion
sits from the moment estimate) and their Spearman. If both are near zero the constraint is a no-op
and a null result is a null *change*, not a null *effect*. **It is not a gate**: the
pre-registration said A LEVER justifies the spend, that was pre-accepted, and adding a condition
now — after a result that flattered the plan — is the threshold-after-the-fact failure this
campaign has recorded five times.

**Tests** (`tests/test_expression.py`, all fast): the moment estimator recovers a known dispersion
(4.0 drawn, 3.2–5.0 accepted) and pins an under-dispersed column at `zinb_theta_max` rather than at
an `inf` or a `nan`; a Poisson column gets a huge but *finite* value, which is why the guard cannot
be an equality test; `"learned"` ignores `gene_idx` bitwise; `"moment_matched"` `theta` is constant
across cells and equals the clamped table while `mu` and `pi` stay bitwise identical; and every way
of reaching `"moment_matched"` without a real per-gene value raises — no vector at construction,
a vector under `"learned"`, no `gene_idx`, a mis-sized `gene_idx`, an out-of-range `gene_idx`, and
an unknown mode string.

#### 6. The UNINFORMATIVE clause is amended — before any fit, and the fifth rule is why

As pre-registered, UNINFORMATIVE fired on either of two conditions: `I(mu)` below 0.90x the
baseline's, **or** reconstruction NLL degrading by more than the baseline's own across-seed spread.

Applying the fifth rule to my own criterion — *before pre-registering a test, ask what would have
to be true for it to fail* — the NLL clause fails almost by construction. A constraint that removes
a degree of freedom will cost some likelihood; the baseline's across-seed NLL spread is a few
thousandths; and **a likelihood that gets slightly worse while spatial fidelity improves is R4's
signature.** That is not a broken fit. It is precisely the outcome the experiment exists to find.
The clause as written would have thrown the finding away and called it a null.

**Replaced with `assert_detection_rate`**, which is a genuine breakage test on the same side of the
model, is already enforced in code, and is the guard that caught the marginal estimator two hours
earlier in this same session. **NLL is still measured and reported on every fit; it is no longer a
criterion.** The driver catches the detection failure and records it as an outcome rather than
dying, so a broken fit produces a row saying so instead of a missing file.

⚠️ **Amended before any fit ran, and the trigger was a dry run of `--summarise` on fabricated
rows** — deliberately fabricated so that a criterion could be exercised without a measurement being
seen first. Nothing from the real experiment has been looked at. The `ANSWERED` / `NOT ANSWERED`
criteria are **unchanged**. If the original NLL clause is preferred it is one line to restore, and
that is the user's call, not mine — but it should be settled now rather than after a number exists.

**Full panel, no gene split.** The six zero-shot checkpoints are gene-split fits and are not a
clean baseline. It is also the only honest configuration for this field: a gene held out of the
panel has no legitimate per-gene dispersion, and inventing one is exactly the silent fallback
Convention 6 forbids.

### T09 — step 2, fit 1 of 6: the diagnostic stops the other five (2026-09-01)

`lookup`, seed 2, full panel (1017 genes), `deep_starmap`/`paper_2_4_6`, `decoder_theta_mode="learned"`,
`decoder_mu_link="exp"`, config `9cac532971466d11`. `reports/t09_theta_learned_s2.json`.

#### 1. The cost estimate held — the first one in this project that did not come in low

**3.74 h (13 477 s)** against the 3.82–4.09 h quoted. The remaining five would be **18.7 h**, total
**22.5 core-hours**, inside the 24 approved. The gate did its job in the cheap direction for once,
and then the diagnostic closed the spend for a different reason.

#### 2. The diagnostic: the value the variant would be pinned at is 7x too low

| estimator | fixture | `deep_starmap`, this fit |
|---|---|---|
| marginal moment | 0.213 | **0.0186** |
| pooled within cell type | 0.469 | **0.0489** |
| the learned head | 0.371 | **0.366** |

`log_ratio_median` **1.934** — the learned dispersion is `e^1.934` = **6.9x above** the pooled
estimate (7.5x on the ratio of medians; 19.7x against the marginal one). `theta_moment_at_max` is
0.0, so nothing is degenerate: the estimate is well defined on all 1017 genes and simply lands far
below what the model learned.

**Pinning `theta` there would make the decoder ~7x *more* over-dispersed.** `f_overdispersion`
rises, `s` falls, retention falls. The variant would test the **opposite** of the hypothesis, and a
NOT ANSWERED from it would be uninterpretable — indistinguishable from the constraint working
exactly as the mechanism predicts, in reverse. **The remaining five fits must not run as
configured.**

#### 3. My defence of the pooled estimator does not transfer, and that is the finding

Earlier today I replaced the marginal estimator with the cell-type-pooled one and defended the
choice at length: conditioning on cell type removes the between-cell structure that biases `v_g`,
the direction was predicted before the fixture measured it, and the fixture confirmed it (marginal
0.213 -> pooled 0.469, crossing the learned 0.371 from below to above).

**On real data the correction moves the same way and is nowhere near enough.** Marginal 0.0186 ->
pooled 0.0489 is the predicted direction — and still **7x short**. The reason is now obvious and
should have been argued before the fixture was trusted: on the fixture, expression is largely
cell-type driven, so grouping by type removes most of the structure from `v_gk`. On real tissue,
cells of one type span the whole volume, so `v_gk` still contains nearly all the spatial variation
the model is supposed to *predict*. The bias I named is an order of magnitude larger on real data,
**and the fixture cannot calibrate it** — it reports the correction as sufficient when it is not.
Another entry for `progress/fixture_limitations.md`: the fixture agreed with the argument and the
agreement was worthless.

⚠️ For the record: the direction I was accused-in-advance of choosing — the one that flatters the
hypothesis — is the direction real data says is *unavailable*. The defence I wrote was sincere and
the estimator is still wrong.

#### 4. What the fit establishes anyway, and it is not nothing

* **The per-cell freedom is real and large.** `within_gene_sd_log` **0.418** (p90 0.507): the
  learned `theta` swings **1.52x** across cells within one gene (1.66x at the 90th percentile). The
  constraint would bind. This is the number that separates "a null effect" from "a null change",
  and it says the change would not have been null.
* **`log_ratio_spearman` = 0.068 — the learned dispersion is essentially *uncorrelated* with the
  data's own.** Across 1017 genes, how over-dispersed the head makes a gene has almost no relation
  to how over-dispersed that gene is within its cell type (IQR of the log ratio 1.891, a 6.6x
  spread between the quartiles). **This is a second, independent statement of R4's trade, and a
  stronger one than "theta is too wide": `theta` is not a dispersion estimate at all.** It is
  absorbing mean variation the model failed to predict, and the fact that it does so
  gene-indiscriminately is what a moment-matched value was never going to fix.
* **THETA IS A LEVER reproduces outside the configuration it was measured in.**
  `f_overdispersion` **0.611 / 0.630** here, against 0.5677–0.6143 on the six gene-split fits —
  full panel instead of an 813-gene pool, a new config hash, one arm. The verdict does not depend
  on the gene split.
* **The law of total variance holds on real data at full panel.** `s` 0.1605 against
  `draw_retention` 0.1614 (`section_3`), 0.2016 against 0.1999 (`section_5`) — within ~1 %. The
  identity the whole step-1 argument rests on is now checked on the object it is used on.
* `mean_vs_real` **2.64 / 2.76** reproduces step 1's 2.65–2.85, with the noiseless-numerator caveat
  unchanged.

#### 5. ⚠️ A side effect of the field I should have flagged when I added it

`Config.content_hash()` hashes **every** field, so adding `decoder_theta_mode` changed the hash of
every config, and `CheckpointState.require_compatible` refuses a resume on a mismatch. **The six
existing zero-shot `deep_starmap` checkpoints (23.5 core-hours) can no longer be resumed at HEAD.**
Nothing measured is lost — their JSON reports are written and reviewed — but re-scoring them now
requires checking out the pre-change commit. This is a general property of adding any `Config`
field and it will recur; it belongs in the record so the next person does not discover it with a
stranded checkpoint.

#### 6. The decision, which is not mine

**Option A — stop.** Record what is established: the mechanism is localised to `theta`
(57–63 % of the conditional variance, arm-independent and split-independent), the per-cell freedom
is real (1.52x), and the learned dispersion is uncorrelated with the data's own. Then state that
**no data-derived fixed value exists that would test it in the right direction**, so the causal
test is unavailable and the paper says so. This is Option 3's honest open state, reached
deliberately.

**Option B — remove the per-cell freedom without choosing the value**: `theta` **per gene, still
learned**. The optimiser picks `theta_g`; it cannot vary it cell to cell. Same five fits, same
~18.7 h, and the ANSWERED / NOT ANSWERED criteria stay **exactly** as pre-registered because
nothing about the outcome changes. The pre-registration's own sentence named this as the mechanism
— *"removes the optimiser's freedom to make the trade R4 names"* — and moment-matching was the
means, not the target.

⚠️ **Option B is weaker in one specific way and it must be named before anyone chooses it.** It
removes only the **per-cell** half of the freedom. If the trade also runs at the per-gene level —
the optimiser simply picking a low `theta_g` for every gene — the constraint does not stop it, and
`log_ratio_spearman` = 0.068 is consistent with exactly that. `within_gene_sd_log` = 0.418 says
there *is* per-cell freedom to remove; **nothing measured says it is the half that matters.**

⚠️ **Is switching to B a design moved after a result?** The case that it is not: the diagnostic was
pre-registered as *reported, not thresholded*, precisely so a null could be told apart from a null
change; it fired **before any variant fit existed**; and what it found is that the experiment as
configured tests the opposite of its hypothesis. That is a mis-specification caught before
spending, not a threshold moved after a disliked result. The case that it is: I am changing the
intervention after seeing a number, and the replacement happens to be the one that can still
produce a positive. **Both readings are on the record and the choice is the user's.**

---

## R12 CLOSE-OUT — `theta` is not a dispersion estimate (2026-09-01)

R12 opened as "spatial autocorrelation collapses in the expression path on real data" and is closed
here as far as measurement can take it without a design change. **Step 2's first fit is where it
ends, and the number that ends it is a diagnostic that was never meant to be the finding.**

### 1. The finding: `log_ratio_spearman` = 0.068

Across **1017 genes** on a full-panel `deep_starmap` fit, how over-dispersed the decoder's `theta`
head makes a gene has **almost no relation** to how over-dispersed that gene actually is within its
own cell type. Spearman **+0.068**; the IQR of the log ratio is 1.891, a **6.6x spread** between
the quartile genes; the median ratio is 6.9x with essentially no ordering agreement underneath it.

**`theta` is not estimating dispersion.** It is absorbing mean variation the model failed to
predict, and it does so **gene-indiscriminately** — not preferentially on the genes that are
genuinely noisy. That is a stronger and more specific statement than the one R12 carried for two
weeks ("`mu` is spatially smooth but too flat in amplitude", and later "`theta` is too wide"),
and it is what makes the whole moment-matching family of fixes unavailable: **there is no
data-derived value to match to, because the quantity the head produces is not the quantity the
data defines.**

It also explains, after the fact, why the two estimators failed in the direction they did. A
marginal or cell-type-pooled moment is a real dispersion; the learned `theta` is a residual
absorber; the two are different quantities and their disagreement is not an error to be corrected
by a better estimator.

### 2. Beside it: the freedom is real, and the mechanism reproduces

* **`within_gene_sd_log` = 0.418** (p90 0.507). The learned `theta` swings **1.52x** across cells
  within one gene, 1.66x at the 90th percentile. The per-cell degree of freedom is not vestigial —
  it is used, substantially, and a constraint removing it would have bound. This is the number
  that would have separated a null *effect* from a null *change*, and it says the change would not
  have been null.
* **`f_overdispersion` = 0.611 / 0.630** on a full-panel `lookup` fit at a new config hash,
  against **0.5677–0.6143** across the six gene-split fits of both arms. THETA IS A LEVER
  reproduces **outside the configuration it was measured in** — different gene set, different
  panel width, different config. Together with the earlier arm-independence (`medcpt` 0.6003 vs
  `lookup` 0.5961), the overdispersion share is now shown independent of the embedding, the text
  channel, the gene split and the panel width. It is a property of the ZINB emission under this
  objective.
* **The law of total variance holds on the object it is used on.** `s` 0.1605 against
  `draw_retention` 0.1614, and 0.2016 against 0.1999 — within ~1 % on real data at full panel. The
  identity every step-1 conclusion rests on is now checked, not assumed.

### 3. What R12 is, finally

The chain is fully localised and the responsible term is named:

| stage | median Moran's I |
|---|---|
| GRF prior | 0.9714 |
| after the flow | 0.9015 |
| decoded `mu` | 0.8607 (2.64–2.76x the real section's counts, on this fit) |
| **sampled counts** | **0.13–0.16** |

The count draw loses it, and it loses it because only 16–20 % of the emitted count variance is
between-cell structure, and **57–63 % of the rest is overdispersion**. Eliminated by measurement
along the way: the length-scale, the layout head, the step budget, `text_emb_mode`, the size
factor (0.2 % of `Var(log mu)`), calibration (retention 14.4 % -> 14.6 %), and amplitude as a
sufficient account (`sd(log mu)` separates the arms with no overlap while retention overlaps, and
the residual association runs backwards 6/6).

**The remaining gap is a design decision, not a measurement.** The ZINB objective admits a
degenerate trade — the same marginal variance is reachable by structured `mu` or by unstructured
`theta`, and nothing in the loss prefers the first. Removing that requires changing the objective
or the emission model, and this project has established which term to change and that no fixed
value derived from the data can substitute for the change.

### 4. R12's two open questions, stated as questions

1. **Per-gene `s` spans 46–119x within one cell** (`s_min` 0.0049–0.0093, `s_max` 0.401–0.420 on
   this fit). "The draw retains ~16 %" is a median over a distribution two orders of magnitude
   wide, and nothing measured says which genes are retained or why. It may matter more than the
   aggregate.
2. **Why is the learned `theta` uncorrelated with the data's?** 0.068 is the finding; the
   mechanism behind it — whether the head is capacity-limited, whether the trade is gene-specific,
   whether a different likelihood would break it — is untouched.

---

## ⚠️ STANDING HAZARD — any new `Config` field invalidates every existing checkpoint's resume

`Config.content_hash()` is `sha256` over the YAML of **every** field, and
`CheckpointState.require_compatible` refuses a resume when it differs. **Adding one field with a
default therefore strands every checkpoint ever written**, whether or not the field affects the
fit.

Measured cost so far: adding `decoder_theta_mode` — a field that is a no-op under its own default
— stranded the **six zero-shot `deep_starmap` checkpoints, 23.5 core-hours**. Nothing measured was
lost, because their JSON reports were already written and reviewed; what was lost is the ability to
re-score them at HEAD. Recovering that means checking out the pre-change commit.

**It will recur, on every field this project adds from here.** Stated as a hazard rather than
fixed, because the fix is a design decision with real trade-offs: hashing only fit-affecting
fields needs a per-field classification that will drift and whose errors are silent resumes of
incompatible checkpoints — which is the failure the hash exists to prevent, and it is worse than
the one it would cure. The cheap mitigation is procedural: **add `Config` fields in batches, and
re-run anything that must stay resumable before adding the next one.**

---

## T09 FINAL CLOSE-OUT (2026-09-01)

Supersedes the close-out above. That one was written before the structured-share audit, step 1,
the variance decomposition, and step 2's first fit; four of its five open items are now resolved or
withdrawn, and its headline still carried a `decoder_mu_link` claim its own addendum corrects.

### What changed since it was written

| item | then | now |
|---|---|---|
| R12's structured share on a current fit | "the cheapest open item in the project" | **done** — 16–20 %, and the mechanism localised to `theta` |
| R4 | four *inferred* instances | **a fifth, measured**: `f_od` 0.57–0.63, arm-, split- and panel-independent |
| `decoder_mu_link` refit | second on the spend list | **not owed** — `exp` shipped 2026-08-21, before every real-data audit |
| a moment-matched `theta` as the causal test | pre-registered, 24 core-hours approved | **stopped after one fit** — `theta` is not a dispersion estimate, so no data-derived value exists to match to |
| the `morans_pearson` replication | third on the spend list | **the only live decision left** |

### The honest headline, corrected

> A continuous-field formulation reconstructs oblique planes at **95 %** of axis-aligned quality.
> On real tissue, every generative component built on top of it — the intensity-field layout, the
> flow-matching expression head, the text-grounded gene embedding — **loses to copying a real
> section**, and the text channel helps only for genes with no training data, on one metric,
> pending replication. The expression failure is localised to the count draw and, within it, to the
> decoder's dispersion: `theta` carries 57–63 % of the conditional variance and is **uncorrelated
> with the data's own dispersion** (Spearman 0.068 over 1017 genes), so it is absorbing unpredicted
> mean variation rather than estimating noise. That is a property of the ZINB objective, not a
> tuning error, and no data-derived value can substitute for changing it. The evaluation
> methodology developed to establish these results — per-arm envelopes, referent-validity tests,
> aggregation-level rules, and a five-channel leak taxonomy for held-out-gene experiments — is the
> transferable contribution.

The assessment is unchanged and is now better supported: **this is a negative-results paper with a
methods contribution, and the methods contribution is the stronger half.** What the last week added
is that the largest negative is no longer an unexplained collapse — it is a named, reproduced,
mechanism-level finding about the ZINB emission model, which is a considerably better thing to
publish than "the expression head underperforms".

### What remains open

1. **R4 / the ZINB trade** — now with a measured instance and a named term, and a design change
   (objective or emission model) as the only route. The largest open question, and the one a
   follow-up paper is built on rather than a follow-up measurement.
2. **R12's two questions** — the 46–119x per-gene spread in `s`, and why `theta` is uncorrelated
   with the data's dispersion.
3. **The `morans_pearson` replication** — the only route to a positive capability claim. Decision
   below.
4. **A7 (SEFL's net contribution)** — three losses ship at zero weight and the paper cannot say
   whether the mechanism it is named for does anything. ⚠️ This is a hole in the *method's own
   identity*, not just an unrun ablation: SEFL is in the title.
5. **R14's donor rule** — costs 0.116 of `marker_depth_r`; deliberately unchanged because fixing
   it makes the negatives **stronger**, not weaker.

### The replication: my recommendation is **spend it**

⚠️ **The case against, first, because it is real.** `morans_pearson` was promoted to primary
*because it produced a result* — the textbook forking path. The metric chosen in advance,
`marker_depth_r`, **refuted** the claim (neither A1 nor A3 clears the floor: 0.42x, 0.45x). This
campaign's base rate on real-data capability tests is close to zero. And ~47 core-hours — now
~8 h/fit on ~227 k cells, extrapolated from a `deep_starmap` anchor that for the first time did
**not** come in low (3.74 h against 3.82–4.09 h quoted) — buys one observation.

**The case for, which I find decisive.**

1. **The current value of the 2.52x result is approximately zero.** A single unreplicated positive
   on a post-hoc metric is a result a referee discounts entirely, and correctly. Replication is not
   an enhancement of that result; it is the difference between having it and not.
2. **A pre-registered replication on a second dataset is the standard, correct remedy for a
   post-hoc metric.** It does not erase the selection, and the paper must still say the metric was
   chosen after the fact — but it converts "we found this while looking" into "we found this while
   looking and then tested it".
3. **It tests the seen/unseen sign flip at no extra cost**, and that is the more interesting claim.
   `lookup` wins by 0.133 where it has a learned row; `medcpt` wins by 0.300 where it does not.
   That is coherent, mechanistically interpretable, and already recorded as E1's finding in
   `specs/10` §7 **before** this decision — so testing it again is confirmation, not another
   forking path. The same six fits score it.
4. **Both outcomes are publishable and the pre-registration already commits to reporting either.**
   Unlike the mechanism hunts of the last week, this has a defined stopping point and cannot
   generate a follow-on measurement.

**Conditions I would attach.**

* The `morans_pearson` criteria stay **exactly** as pre-registered. Nothing about them moves.
* **The sign-flip criterion must be written down before the fits run**, in the same document, or
  it does not count — the same rule that has governed everything else here. It is not yet written.
* The staged gate holds: build the split, run the model-free ceiling and the descriptor-coverage
  check, and **stop with no fits** if either fires. Then **one timed fit** before the other five.
* ~47 core-hours is a lower bound. One favourable extrapolation does not undo five unfavourable
  ones.

**If the answer is no, that is also a defensible ending** and it should be recorded as a decision
rather than an omission: the paper reports the 2.52x as an observation with its pre-registration
attached and states that the replication was scoped, costed and declined. That is honest. It is
just weaker than doing it, and the difference is one dataset.

---

## REPLICATION — PRE-REGISTRATION, PART 2: the seen/unseen sign flip (2026-09-01)

Written **before any `cosmx_nsclc_3d` fit, and before the staged gate has been run.** Part 1 — the
`morans_pearson` criteria — is above and is **unchanged**; nothing in it moves. This adds the
second criterion, at the user's instruction that the flip counts only if it is pre-registered:
*"it is currently in exactly the position `morans_pearson` was in before this replication was
proposed. If it is not pre-registered it does not count, and I would rather lose it than launder
it."*

### The finding being replicated

`deep_starmap`, three seeds x two fits x four arms x two folds x two gene pools, on
`morans_pearson`. Within one run and one scoring pass, the sign of the text channel's value
reverses with nothing changing but which genes are scored:

| gene pool | winner | margin | envelope | signs |
|---|---|---|---|---|
| kept (fitted on) | `lookup` — a free per-gene table | **−0.1330** (A1−A3) | 0.0230 | 5.8x, 6/6 |
| held out (never in a batch) | `medcpt` — text alone | **+0.2999** (A2−A4) | 0.0532 | 5.6x, 6/6 |

### The contrast, and why it is A2 − A3 on both pools

The table above uses **two different contrasts**, one per pool, which is not a sign flip *within* a
comparison. The criterion is stated on **A2 − A3** — `medcpt` pure text against `lookup`+distill —
**on both gene pools**, because:

* it is the one contrast in which **neither arm is degenerate on either pool**. A4 emits `norm(0)`
  for a held-out gene, so A2 − A4 on the held-out side is a comparison against a void, which
  `specs/10` §7 and Part 1 both already call the weaker claim;
* A2 − A3 on held-out genes **is already Part 1's PRIMARY** (+0.2514 on `deep_starmap`), so the
  flip criterion adds only the kept half — free, on the same fits and the same scoring pass;
* A3 is the **shipped** lookup configuration, so the flip is stated against what the method
  actually ships rather than against an arm built to fail.

⚠️ **`deep_starmap`'s kept-pool A2 − A3 is derived, not directly reported.** From the recorded
A1 − A3 = −0.1330 and A1 − A2 = −0.0008 it is **−0.1322** exactly, and the two directly measured
neighbours bracket it tightly — A2 − A4 = −0.1312, A1 − A3 = −0.1330, all three inside 0.002. The
**point estimate** is therefore safe to use as a magnitude scale; the **envelope** for that contrast
is not derivable and must be read from the run. The `cosmx` scorer reports every pairwise contrast
directly, so the replication measures both halves without any arithmetic.

### FLIP REPLICATES — all of:

1. **kept** genes: A2 − A3 **< 0**, signs agreeing at every seed and every fold;
2. **held-out** genes: A2 − A3 **> 0**, signs agreeing at every seed and every fold;
3. **both** magnitudes exceed the shared envelope computed **on their own gene pool** — the pools
   are different gene populations and §4.2a's per-metric/per-arm rule extends to them;
4. fold balance >= 0.25 at every seed, on **both** halves.

### FLIP DOES NOT REPLICATE

Either half's sign disagrees with `deep_starmap`'s, **or** either magnitude sits inside its own
pool's envelope. Reading: the reversal was a property of `deep_starmap` — mouse cortex, laminar —
and the open-vocabulary claim has no within-run evidence.

### ONE-SIDED — exactly one half meets 1–4, the other's magnitude is inside its envelope

Not sign-reversed, just undetectable. Report **which** half carries it and read no flip. If the
**held-out** half is the one that survives, that is Part 1's claim and nothing more; if the **kept**
half is the one that survives, the flip is not established and the paper keeps only the negative it
already had three times over.

### UNINFORMATIVE — conditions (a)–(d) of Part 1 apply unchanged, plus:

* **(e)** the **kept**-pool shared envelope on `morans_pearson` exceeds **0.1322** — the design
  could not detect a kept-half effect the size of the one being replicated. Symmetric with Part 1's
  condition (c), which does the same on the held-out half against 0.2514. The threshold is robust
  to the derivation above: the three candidate values span 0.0018.

### ⚠️ Three things this criterion does NOT fix, stated so nobody reads it as more than it is

1. **The flip inherits `morans_pearson`'s selection problem entire.** The *contrast* and its
   criteria are now fixed in advance; the *metric* was still chosen from six because it produced a
   result. No replication on that metric erases that, and the paper must say so in the same
   sentence that reports the flip.
2. **The kept half is not an independent test.** `lookup` beating `medcpt` on genes with rows has
   already reproduced three times (tier-1, `deep_starmap`, and the zero-shot run's own kept pool at
   5.8x). Replicating it is *expected*, and the flip's novelty rests almost entirely on the
   held-out half — which is the same quantity Part 1's primary tests. **The flip is therefore
   close to, but not identical with, Part 1**: what it adds is the *reversal* requirement, which is
   a stronger and more interpretable statement than a clearance against a floor, and which is
   robust to the floor being wrong.
3. **The two verdicts can disagree, and neither overrides the other.** The flip can REPLICATE while
   Part 1 returns REFUTATION (A2 − A3 reverses cleanly but A2 fails to clear the `shuffled` floor
   by an envelope), and Part 1 can return SUPPORT while the flip is ONE-SIDED (the kept half
   undetectable). Both are reported, in full, with whichever combination occurs stated plainly.

### Reported beside it, as context and **not** as a criterion

The same flip on **`marker_depth_r`** — the metric that *was* pre-registered. On `deep_starmap` it
is already null there (held-out A1 − A3 = −0.0044 against a 0.1273 envelope, signs disagreeing), so
replicating a null is not a test of anything and it carries no threshold. It is reported because a
reader will ask, and because a flip appearing there would be a genuine surprise worth recording.

---

## A7 — cost, and whether it beats the replication (2026-09-01)

### What A7 is

An **addition** experiment, not an ablation: the shipped model (all three SEFL weights at 0)
against the same model with `w_thick = w_prog = 0.2`. `w_cross` stays at 0 in both arms — it is
redundant by construction in v25 (asserted bitwise on an untrained model) and harmful when trained
(generated per-gene variance 0.067 against 0.711 with SEFL off, R6). The question is **"is SEFL
used at all"**, and it is currently unanswerable: three losses ship at zero weight, SEFL is in the
design's title, and T07 established the losses are *correct* — never that they *help*.

### Cost, on the two candidate datasets

Three seeds x two arms = **6 fits**, which is the minimum §4.2 allows for an envelope.

| dataset | measured per cold fit | 6 fits | headroom on the reconstruction task |
|---|---|---|---|
| `starmap_visual_cortex` (tier-1, 16.5 k cells) | **62 min** | **~6.2 core-hours** | **4.6x** |
| `deep_starmap` (~113 k cells) | **3.74–4.09 h** | **~23 core-hours** | **0.5x — saturated** |

**Tier-1 is the right dataset and the reason is this project's own ceiling-first rule.** A7 has to
*detect an effect*; `deep_starmap`'s reconstruction task is saturated against an oracle copier at
0.5x headroom, which is what sank the `deep_starmap` reconstruction comparison, and spending 23
core-hours to ask a detection question on a saturated task is the mistake the rule exists to
prevent. Tier-1 has 4.6x.

⚠️ **One check must run before committing to tier-1, and it is free.** `L_thick` is a
thick-section consistency term. If tier-1's sections are too thin for it to bind, the loss charges
~0 there and A7 on tier-1 tests **`L_prog` alone** — which would make the dataset choice wrong, not
merely weaker. One forward pass on an untrained model settles it: **does `L_thick` charge a
non-zero loss on tier-1?** If it does not, A7 goes to `deep_starmap` at ~23 core-hours and the
saturation caveat is stated in the result.

⚠️ **And a tier-1-only A7 is a weaker claim about the shipped model**, since the negatives that
define this paper were measured on both datasets. State it as "on the dataset where the effect is
detectable", not as "on the shipped configuration".

**Possible reuse, unverified**: step 2's `learned` baseline is a full-panel `lookup` `deep_starmap`
fit at SEFL-off, so it could serve as one of three SEFL-off seeds *if* A7's config matches it
exactly. That would take a `deep_starmap` A7 from 6 fits to 5 (~19 core-hours). It does not change
the recommendation and should not be counted on until the hashes are compared.

### Which is the better spend, if only one

**A7 — and it is not close, because it costs 13 % of the replication** (6.2 against 47
core-hours). Three reasons:

1. **It answers a question about what the method *is*.** The paper cannot currently say whether the
   mechanism it is named for contributes anything. That is not an unrun ablation, it is a hole in
   the method's identity, and no amount of evaluation methodology fills it.
2. **It converts a defaulting decision into a measured one.** SEFL ships at zero weight on the
   strength of T07 measurements about the losses' *own* criteria and T06's acceptance tests —
   never about the six target metrics. Right now the paper defends a default with evidence that
   does not address it.
3. **Both outcomes are publishable and one is interesting.** If SEFL loses, "a continuous-field
   model needs less self-supervision than a point-cloud one" is a real finding about the
   formulation, and it is the finding the by-construction `w_cross` result already points at.

⚠️ **The strongest argument against, stated fairly**: the prior favours SEFL losing — every
distributional statistic T07 could measure moved the wrong way at 0.2 — so A7 most likely adds
*another negative* to a negative-results paper, while the replication is the only thing that could
add a positive. That is true, and it is why the replication is the better spend **per hour of
result-that-changes-the-headline**. It is not why A7 wins: A7 wins on the ratio.

### ⚠️ The dilemma is probably false, and that is the more useful answer

At 6.2 against 47 core-hours these are not competing for the same budget. Running A7 first costs
**13 %** of the replication and delays it by well under a day, and its result cannot change what
the replication measures. **The sequencing that dominates both single choices: run A7 on tier-1
now, then run the replication.** If the budget genuinely admits only one *large* spend, that spend
is the replication — and A7 should still be run, because at 6.2 core-hours it is not the thing
competing with it.

---

## A7 — the `L_thick` binding check built, and two versions of it were wrong (2026-09-01)

`scripts/t10_a7_thick_binding.py`. It runs before A7's six fits and answers one question: **on
this dataset, how large a student–teacher disagreement must exist before `L_thick` charges
anything at all?** It fits nothing and takes no training step.

**Two versions failed on the synthetic fixture before either reached tier-1**, and both failures
are the same shape — an instrument reporting "no signal" for a reason that has nothing to do with
the thing being measured.

1. **"An untrained model answers it: a term that is structurally zero is zero at initialisation
   too."** False in the direction that matters. `L_thick` compares the **student** on a `3h` slab
   against the **teacher** on three `h` slabs over common random numbers whose estimator error
   cancels exactly; at initialisation the EMA teacher is a deep copy, so both branches integrate
   the same field over the same points and the term is **exactly zero by construction**. Measured
   on the fixture: `count` 0.0, `count_by_type` 0.0, `state` −1.1e-13, on a geometry where the
   thick slab occupies 18 % of the z-extent and nothing grazes. An untrained model returns zero
   *whether or not the term binds*.
2. **Perturbing the student's parameters by 5 % of each tensor's sd.** The teacher then genuinely
   differed (max parameter delta 0.0129, teacher object distinct), and `count` was **still exactly
   zero**. The cause is `_poisson_consistency` = `relu(z² − 1)/expected`: a disagreement inside one
   Poisson standard deviation costs **exactly zero**, which is what `specs/07` asks for and what
   `test_thick_counts_add` checks. A 5 % move on 26 182 intensity-head weights changed the
   *integrated count* by far less than `1/sqrt(N)`.

**What the instrument measures now.** The student branch's **output** is scaled by a known factor
— `_intensity_at` patched for the duration of a sweep, restored in a `finally`, the same discipline
`t09_retention_mechanism` applies to `_flow_counts` — so the sweep reads in units of **relative
count error**. It reports, per error size, the fraction of draws charging above `ZERO`, the median
and max charge, and the grazing fraction; the headline is **the smallest relative error the term
notices**.

On the fixture that is **3 %**, with 1 % and below at exact zero — consistent with a hinge at
`1/sqrt(N)` and a thick slab of roughly 1 100 cells.

**This is the number that decides A7's dataset**, and it is a data property, not a modelling one:
a volume whose slabs hold few cells needs a large disagreement before `L_thick` has any gradient;
a denser one registers a much smaller one. Tier-1 has ~4.1 k cells/section against `deep_starmap`'s
18–39 k, so the hinge should bite at a **larger** relative error on tier-1 — which is a reason the
cheaper dataset might be the wrong one, and it is measured rather than argued.

⚠️ **A third defect, found and fixed in the same pass**: `fraction_grazed` counts draws where every
part is exactly 0.0 — `_zero_terms`' signature — but below the hinge `_poisson_consistency` returns
exact zeros too, so the two causes are indistinguishable there. On the fixture it reads **0.375 at
a 1 % error and 0.000 at 3 %**, and the 3 % figure is the true one. The verdict now reads grazing
off the **largest** factor in the sweep, and `draw_terms`' docstring says why.

⚠️ **A non-zero charge is necessary, not sufficient.** It says the geometry admits the term. It
does not say `L_thick` helps, which is what A7's six fits measure and what nothing short of them
can say.

`tests/test_sefl.py::test_thick_binding_instrument_reads_the_poisson_hinge` pins both failure
modes so a third version cannot regress into either: zero at agreement, zero at 0.1 %, charging at
30 %, monotone between, and grazing read only where the hinge is escaped.

### The three verdicts, and what each means for A7

* **BINDS** — the geometry admits the term. A7 runs on tier-1 at ~6.2 core-hours.
* **DOES NOT BIND** — a 30 % student-side error charges nothing. A7 on tier-1 would test
  **`L_prog` alone**, which is a different experiment. **Stop and report; the user chooses it
  rather than having it happen.**
* **ILL-POSED** — sections thicker than the spacing between them, so the coarse-graining identity
  is false and the same tissue is observed twice (`OverlappingSlabsWarning`). Not a matter of
  degree; A7 does not run on that dataset at all.

Reported beside the verdict: `thickness_is_assumed`. If tier-1's thickness was defaulted from
median spacing rather than measured, `L_thick` still binds but **every A7 number rests on that
assumption** and the result must say so.

### A7's arm switches, added to the tier-1 driver

`scripts/t09_ship_starmap.py` gains `--w-thick` and `--w-prog`. `specs/10` §9's CLI table already
lists them as A7's switches; the script did not have them. Each overrides exactly one `Config`
field **after** the selected config is loaded, so the arm differs from the shipped fit in the
weight and nothing else, and the override lands in the config hash the report prints. The banner
now names all three SEFL weights and marks the arm `<- A7 arm (SEFL ON)` or `(shipped: SEFL off)`,
because two runs whose only difference is a weight are exactly the pair a log has to distinguish.

`--w-cross` is deliberately **not** added. `w_cross` stays at 0 in both arms — redundant by
construction in v25 and harmful when trained (R6) — so A7 tests **two** losses, not three, and the
write-up has to say so. A switch that made it settable would invite an arm nobody pre-registered.

---

## A7 — STOPPED at the binding check. Both candidate datasets fail, for different reasons (2026-09-01)

`reports/t10_a7_thick_binding_tier1.json`, `..._deep.json`, seed 2, 32 draws each. The
pre-registered stop has fired on **both**, so no A7 fit runs and the choice returns to the user,
as committed: *"stop and let the user choose it rather than have it happen."*

| | tier-1 `starmap_visual_cortex` | `deep_starmap` |
|---|---|---|
| sections | 4 | 4 |
| thickness (µm) | 22, **assumed** | 42, **assumed** |
| spacing median / min (µm) | 22 / 22 | 42 / **40.6** |
| slabs overlap | no (exactly tangent) | **YES** |
| `3h` slab (µm) | 66 | 126 |
| z-extent (µm) | **88** | **167.3** |
| `3h` / z-extent | **0.75** | **0.75** |
| grazed draws at a 30 % error | **0.969** | 0.938 |
| smallest error noticed | **never, up to 30 %** | never, up to 30 % |
| verdict | **MARGINAL** | **ILL-POSED** |

### 1. `deep_starmap` is ILL-POSED, and the cause is a defaulting rule, not the tissue

`spacing_min` is **40.6 µm** against an assumed thickness of **42 µm**, so one section pair
overlaps by 1.4 µm and the coarse-graining identity `L_thick` rests on is false — the same tissue
observed twice.

**The 42 µm is not measured.** `loaders` defaults `Section.thickness` to `vol.median_spacing` and
sets `thickness_is_assumed`. **That rule guarantees `OverlappingSlabsWarning` on any stack whose
spacing is not uniform**, because half the gaps are below the median by construction. It is not a
fact about the sections; it is a default choosing a value that breaks a downstream loss.

⚠️ **`min_spacing` would be the safe rule and I am not proposing the change.** `Section.thickness`
is load-bearing for T05's intensity integral — cells per unit *volume* — so altering it moves every
layout number in this project, including R11's shipped `layout_mode` decision. That is a
project-wide invalidation to unblock one ablation, and the trade is not worth it without a separate
decision. **Recorded as a defect in the default; the fix is not costed here.**

(The 1.4 µm is the same 1.4 µm as R14's donor rule. One irregular gap in this stack shows up in
two unrelated places.)

### 2. Tier-1 is MARGINAL, and the cause is undetermined between two possibilities

**97 % of random planes are refused.** The one draw in 32 that is not charges strongly — total
1.70 at a 30 % error, 0.102 at 10 %, 6.3e-4 at 3 % — so `L_thick` is not structurally vacuous here.
It is *starved of usable planes*.

A draw is refused when `_slab_sample` gathers fewer than `Config.layout_n_mc` = 4096 in-box points
within `Config.sefl_rejection_max_rounds` = 16. **Two very different causes produce that and the
sweep cannot distinguish them:**

* **The sampler is under-budgeted.** The loop proposes `4 x (n - found)` per round and accepts a
  fraction `a`, so `remaining` shrinks by `(1 - 4a)` and the rounds needed grow as
  `log(n) / -log(1 - 4a)`. Worked out at the shipped budget:

  | acceptance `a` | rounds needed | 16 enough? |
  |---|---|---|
  | 0.02 | 96 | no |
  | 0.05 | 37 | no |
  | 0.10 | 17 | **no — by one round** |
  | **0.11** | 16 | **yes, the threshold** |
  | 0.20 | 6 | yes |

  **A slab that fits perfectly well is refused whenever acceptance sits below about a tenth.**
* **The geometry is genuinely starved.** An oblique `3h` slab through a volume only `4h` deep
  intersects little of it, and no round budget recovers that.

`3h / z-extent = 0.75` on **both** datasets is consistent with either. The fixture, where
everything works, has 9 sections and a 425 µm extent — `3h` is **0.18** of it — which is why
`thickness_ratio = 3` and a 16-round budget were never stressed. **Another
`progress/fixture_limitations.md` entry: the fixture's stack is 4.8x deeper in units of its own
slab than the real data's, and both SEFL constants were set against it.**

### 3. The instrument that would settle it, and what each answer rules out

`scripts/t10_a7_thick_binding.py` now also reports `_slab_sample`'s **acceptance rate** directly —
patched for the probe, restored in a `finally`. On the synthetic fixture as a control: acceptance
**0.187–0.458, median 0.237**, 3 rounds needed, **100 %** of draws reach `n`. That is what a
healthy geometry looks like.

**If acceptance on tier-1 is >= ~0.05**: the geometry is fine and the 16-round budget is the
binding constraint. Raising `sefl_rejection_max_rounds` is a **sampler-budget** change — it alters
how a fixed integral is estimated, not what `L_thick` compares — the same class of fix as R11's
grid-multinomial sampler replacing the biased rejection one. A7 then runs on tier-1 at ~6.2
core-hours. ⚠️ ~~It also means every SEFL measurement in T07 was taken on a sampler that silently
refused most planes.~~ **That sentence is wrong and is corrected in the section below**: T07
measured on the synthetic fixture, where acceptance is 0.237 and every draw reaches `n`. The
crippling is a real-data property and T07 never saw real data.

**If acceptance is << 0.01**: no budget recovers it. `L_thick`'s random-plane construction is
**inoperative on 4-section stacks**, which is the shape of every volume this project has. That is
not a failure to run the experiment — **it is a partial answer to the question A7 was going to
ask**, obtained for zero core-hours: the thickness half of SEFL cannot contribute on this data, and
the paper says so. A7 would then be `L_prog` alone, which is the different experiment the
pre-registration said the user chooses rather than has happen.

**Between 0.01 and 0.05**: raising the budget helps some planes and not most. Report and decide;
do not tune the ratio to make an experiment runnable.

⚠️ **What I will not do**: lower `Config.thickness_ratio` from 3 to 2 to make the slab fit. It
would raise the acceptance rate and it would also change what `L_thick` asserts, and changing a
loss's definition to make its ablation runnable is the tuning-to-enable failure this project has
recorded five times in other forms.

---

## A7 — the acceptance probe: the sampler is under-budgeted, the geometry is not (2026-09-01)

`reports/t10_a7_thick_binding_tier1.json`, tier-1, seed 2, 32 probes.

| | value |
|---|---|
| acceptance, median | **0.0372** |
| acceptance, min – max | **0.0280 – 0.1848** |
| draws reaching `n` at the shipped budget | **0.0625** (2/32) |
| rounds needed, median | **51.7** against a budget of **16** |
| planes with acceptance >= 0.05 | 0.344 |
| **fixture control** | **0.237** median, 3 rounds, **1.000** reach `n` |

### The reading

**Nothing is starved.** The lowest acceptance over 32 planes is **2.8 %** — an oblique 66 µm slab
through an 88 µm-deep box really does intersect the tissue; it just occupies a small fraction of
its own bounding window. The integral `L_thick` needs is well defined and estimable everywhere.
**The estimator is impatient, not the geometry empty.**

Simulating `_slab_sample`'s own loop at the observed rates:

| acceptance | rounds needed | proposals | x `n` | |
|---|---|---|---|---|
| 0.0280 | **68** | 146 297 | 35.7 | observed **min** |
| 0.0372 | **50** | 110 112 | 26.9 | observed **median** |
| 0.1848 | 7 | 22 174 | 5.4 | observed max |
| 0.2369 | 4 | 17 304 | 4.2 | fixture control |

A budget of **96** covers every plane observed; 16 covers only the top decile. At 5 % acceptance —
a third of tier-1's planes — 37 rounds are needed and 16 are given, which is why
`fraction_acceptance_above_0.05` is 0.344 while `fraction_reaching_n` is 0.0625.

### ⚠️ This is my pre-registered middle band, and its stated reading is wrong

I wrote three bands before the numbers: `>= 0.05` budget-bound, `<< 0.01` starved, and **between**
— *"raising the budget helps some planes and not most. Report and decide."* The median is 0.037, so
this is the middle band and **the pre-registered action is "report and decide", which is what this
is.** But the band's *description* is contradicted by the numbers behind it: raising the budget to
96 helps **every** plane, not some, because the minimum acceptance is 0.028 and 0.028 needs a
finite 68 rounds.

The threshold was a **proxy** for "is the budget the binding constraint". The probe measures that
directly, and a direct measurement supersedes a proxy for the thing it stood for. I am not moving
the threshold to reach a verdict I like — the band's own action was to report — but the distinction
matters enough to state, because it is the same shape as a threshold moved after a result and a
reader is entitled to check it.

### ⚠️ A claim I put in the record yesterday is wrong

I flagged in advance that a budget-bound answer *"would also mean every SEFL measurement in T07 was
taken on a sampler that silently refused most planes."* **It does not.** T07 measured on the
synthetic fixture — CLAUDE.md Convention 7 — where acceptance is **0.237** and **100 %** of draws
reach `n`. The crippling is a property of the real volumes, and T07 never saw one. So:

* T07's `w_cross` finding (generated per-gene variance 0.067 against 0.711) **stands**.
* T07's `w_thick = w_prog = 0.2` finding (three of T06's acceptance tests failing) **stands**, and
  is **not** reattributable to `L_prog` alone as I had begun to reason — because on the fixture
  nothing was crippled, so there is nothing to reattribute.

  ⚠️ **The reason I gave for that was also wrong, and is corrected below**: I wrote that
  `_slab_sample` is called only by `thick_terms` and that "`cross_terms` and `prog_terms` draw
  planes but not slab points", so a crippled sampler would have isolated the damage to `L_thick`.
  The first clause is true of `_slab_sample` specifically; **the inference is not.** `prog_terms`
  calls `Plane.sample_points`, a different function running the *same* rejection loop off the
  *same* constant. The conclusion survives — the fixture is not starved — but it survives for one
  reason, not two.

The corrected sentence is patched in place above rather than deleted.

### What this means for A7

**A7 on real data would have trained with `L_thick` inert on ~97 % of steps**, and no output of it
would have said so — the term returns exact zeros and the loss simply reads low. **That is what the
binding check was built to catch, and it caught it.** It is also the first time this project has
established anything about SEFL on real tissue.

**The unblock is one constant**: `Config.sefl_rejection_max_rounds` 16 -> 96 (or 128 for margin).

* It is a **budget** change: it alters how a fixed integral is estimated, not what `L_thick`
  compares. The same class as R11's grid-multinomial sampler replacing the biased rejection one.
* It **invalidates nothing measured**. `_slab_sample` is reached only through `thick_terms`, which
  `sefl_terms` skips entirely at `w_thick = 0` — and all three SEFL weights ship at 0, so the
  function is never called in any shipped fit. The fixture's own draws are unaffected because 4
  rounds are already enough there and the loop breaks on `found >= n`.
* It **costs** ~146 k uniform proposals per `_slab_sample` call at the worst observed acceptance,
  against ~50 k already spent failing. Roughly 3x a sampling step that is negligible beside the
  intensity forward pass on 4096 points.
* The guard still guards: a plane that genuinely misses the tissue (acceptance ~0.001) needs ~2000
  rounds and still raises. The threshold moves from "the sampler is impatient" to "the slab does
  not intersect the volume", which is what the guard was for.

⚠️ **The better fix is not this one, and I am not proposing it now.** The proposal schedule
`4 x (n - found)` is what makes convergence geometric — `remaining` shrinks by `(1 - 4a)` per round,
so the rounds needed scale as `1/a`. A schedule that proposed `(n - found) / a_observed` would
converge in two or three rounds at **any** acceptance. That is a strictly better sampler and a
larger change to shipped code; raising the budget is sufficient to unblock A7 and is the smaller
thing to be wrong about.

⚠️ **And `deep_starmap` stays ILL-POSED regardless.** This unblocks tier-1 only. The overlapping
slabs there are a separate defect with a separate cause (the median-spacing thickness default) and
a fix that would invalidate every layout number in the project.

### Two smaller notes

* The sweep says 1/32 planes charge; the probe says 2/32 reach `n`. Different plane draws — the two
  routines consume the generator differently — so this is agreement within one draw, not a
  discrepancy.
* `AssumedThicknessWarning` is still on. If the budget is raised and A7 runs, **every A7 number
  rests on a thickness defaulted from median spacing**, and the result has to say so.

---

## `sefl_rejection_max_rounds` 16 -> 96 — applied, with two corrections to the record (2026-09-01)

`Config.sefl_rejection_max_rounds` is now **96**. The field's docstring carries the measurement,
the arithmetic, and the fix that was declined.

### The old docstring's arithmetic was wrong, and that is the whole bug

> "Each round proposes four times the shortfall, so a slab covering a tenth of the bounding box
> fills in one or two."

Four times the **shortfall** means `remaining` shrinks by `(1 - 4a)` per round, so the rounds needed
scale as `log(n) / -log(1 - 4a)` — **`1/a`, not `O(1)`**. At a tenth acceptance that is **17**
rounds, not one or two, which the shipped budget of 16 already missed by one. The constant was set
against a sentence that did not hold, on a fixture where acceptance is 0.237 and the error never
showed.

### ⚠️ Correction 1 — restated, because the wrong version has been said more often than the right one

The claim **"every T07 SEFL measurement was taken on a broken sampler"** is **false**. I wrote it
once as a conditional flag, the user repeated it back twice as a finding, and it was then in the
record three times against one correction. Stating it plainly here so the count runs the other way:

* **T07 measured on the synthetic fixture** (CLAUDE.md Convention 7 — tests run without real data).
* **On the fixture the sampler is healthy**: acceptance **0.237** median (0.187–0.458), 3–4 rounds
  needed against a budget of 16, and **100 %** of draws reach `n`.
* **The starvation is a property of the real volumes only.** T07 never saw one.
* Therefore T07's `w_cross` result (generated per-gene variance 0.067 against 0.711) and its
  `w_thick = w_prog = 0.2` result (three of T06's acceptance tests failing) **both stand,
  unqualified.**

What *is* true, and is the useful version of the claim: **A7 on real data would have trained with
its SEFL terms starved on ~97 % of steps, and nothing in its output would have said so.** That is a
statement about an experiment that had not run, not about measurements already in the record.

### ⚠️ Correction 2 — `prog_terms` uses the same rejection loop, so `L_prog` was starved too

I wrote that `_slab_sample` is called only by `thick_terms` and that "`cross_terms` and `prog_terms`
draw planes but not slab points". The first clause is true of `_slab_sample` *specifically*. **The
inference drawn from it is wrong**: `prog_terms` (and `prog_wrong_terms`) call
`Plane.sample_points`, a different function running the *same* rejection loop off the *same*
constant, at `n = sefl_patch_cells = 2000`.

| call site | `n` | rounds at the observed tier-1 median (a = 0.037) | at the worst plane (a = 0.028) |
|---|---|---|---|
| `thick_terms` MC | 4096 | 51.6 | **70.0** |
| `thick_terms` state | 512 | 38.7 | 52.5 |
| `prog_terms` patch | 2000 | 47.2 | **64.0** |

Three consequences:

1. **`L_prog` is subject to the identical failure**, less severely only because `n` is smaller. Its
   planes are also thinner (`h`, not `3h`), which should *raise* acceptance — **unmeasured**: the
   binding check probes the thick path only, so `L_prog`'s real-data acceptance is an open number.
2. **"A7 on `L_prog` alone" was never a viable fallback.** The pre-registration offered it as the
   different experiment the user could choose if `L_thick` did not bind. At budget 16 `L_prog`
   would have been starved too, so that fallback was measuring nothing either. It is only viable
   *after* this change.
3. **Correction 1's conclusion survives but loses a leg.** I gave two reasons the T07 numbers stand
   — "the fixture is not starved" and "a crippled sampler would have isolated the damage to
   `L_thick`". The second is void. The first is sufficient and is the one that carries it.

96 covers `thick_terms` at the worst observed plane (70 rounds) with margin, and `prog_terms` (64)
with more.

### The better fix, declined, and where to look if A7 turns on it

`4 x (n - found)` is what makes convergence geometric. A schedule proposing
`(n - found) / a_observed` — estimating `a` from the first round — converges in **two or three
rounds at any acceptance**, and would not need this constant tuned per dataset at all. It is a
larger change to shipped sampling code, and raising the budget is sufficient to unblock A7, so it
is deliberately not taken.

**If an A7 result turns on sampler behaviour, this is the first place to look.** The budget makes
the failure rarer; it does **not** remove the `1/a` scaling. A denser panel, a thinner volume, or a
larger `thickness_ratio` walks straight back into it, and the next person will meet it as
"`L_thick` mysteriously reads low" rather than as an error.

### ⚠️ The standing hazard fires again

`content_hash` covers every field, so this change strands **step 2's `learned` baseline checkpoint
(3.74 h)** against `require_compatible`, on top of the six zero-shot checkpoints already stranded by
`decoder_theta_mode`. Nothing measured is lost — that fit's JSON is written and reviewed, and step 2
is stopped — but the hazard has now cost **27.2 core-hours of resumability across two fields**, and
the procedural mitigation (batch the fields, re-run what must stay resumable first) was not applied
here because the alternative was leaving A7 blocked.

---

## A7 — the SEFL-off arm, seed 1 of 3 (2026-09-01). Timing gate NOT satisfied.

`reports/t10_a7_off_s1.{md,json}`, tier-1 `starmap_visual_cortex`/`paper_2_4_6`, config
`19cf1544f0cc5fcc`, defaults, `w_cross = w_thick = w_prog = 0`, `sefl_rejection_max_rounds = 96`.

**This is the baseline arm only.** A7 is a two-arm comparison and nothing about SEFL can be read
until the `w_thick = w_prog = 0.2` arm exists.

### ⚠️ The timing gate cannot be evaluated from what was sent

The gate was "one fit before the other five". **Neither artifact carries a duration.**
`t09_ship_starmap.py` has timed the fit since T09 and prints `fit: N steps in Xs` — to the
console, which is not what gets sent. My instruction ("send the measured minutes") assumed a
number that no artifact contained.

**Fixed**: `fit_seconds` is now persisted into the run's JSON (`null` when a fit is reused, which
is not the same as free). The wall clock for this run has to come off the console line.

### The numbers, single seed, uncalibrated (the shipped arm)

| metric | uncalibrated | detection-calibrated | delta |
|---|---|---|---|
| `paper_morans_pearson` | +0.2340 | +0.3316 | **+0.0976** |
| `paper_gearys_pearson` | +0.2357 | +0.3365 | **+0.1008** |
| `paper_umap_mixing` | +0.7976 | +0.7765 | −0.0211 |
| `paper_marker_field_r` | +0.4472 | +0.4677 | +0.0205 |
| `paper_marker_depth_r` | +0.6289 | +0.6431 | +0.0142 |
| `paper_celltype_localization` | +0.7591 | +0.7601 | +0.0010 |
| `paper_gene_mean_spearman` | +0.9825 | +0.9611 | −0.0214 |
| `paper_cell_count_ratio` | 0.988 | 0.988 | — |

Consistent with the established negative: `morans_pearson` +0.234 against tier-1's **best
available copy at 0.713–0.784** and a noiseless ceiling of 0.927–0.939. The model is far below the
model-free floor, which is what every real-data comparison in this project has said.

**Internal consistency checked** rather than assumed: every `matched` figure is the median of its
three per-section values, and `raw == matched` exactly on `gearys`, `umap_mixing` and
`marker_field_r` because those medians land on `section_2` or `section_6`, where `n_pred < n_gt`
and the density match is a no-op. Not a bug.

### ⚠️ A reporting defect in the per-module table, and it is not small

The `|diff|` column is **`mean_g |I_gen(g) − I_real(g)|`** over the module's genes, while `I_gen`
and `I_real` beside it are **means over the same genes**. A mean of absolute differences is not the
absolute difference of the means, so subtracting the two displayed columns — which is what the
header `|diff|` invites — gives a different number:

| module | \|columns\| | reported | apart |
|---|---|---|---|
| 0 | 0.1315 | 0.1694 | 1.3x |
| **1** | **0.0114** | **0.1639** | **14.4x** |
| 2 | 0.0354 | 0.0364 | 1.0x |
| 3 | 0.1253 | 0.1369 | 1.1x |

Module 1's genes miss in **both directions** and nearly cancel in the mean: the module looks
almost perfectly calibrated by the columns and is the second-worst by the per-gene statistic. The
computation is correct; the header was wrong.

**Fixed**: the columns are now `mean I_gen` / `mean I_real` / `mean |I_gen - I_real|`, with a note
carrying module 1's 14x as the worked example. This diagnostic feeds the per-channel-group `ell`
escalation (SPEC_QUESTIONS A2), so a reader who subtracts the columns would under-state the case
for a design change by an order of magnitude on the module where it is strongest.

### An observation that is not A7's, and must not be read as a result

**Detection calibration moves the two autocorrelation metrics by ~+0.10 on tier-1** while costing
`gene_mean_spearman` 0.0214 and `umap_mixing` 0.0211. The project ships it **off**, on a T09
finding that it had **no headroom** — a fixture measurement, later reproduced on `deep_starmap`
where it moved retention 14.4 % -> 14.6 %.

⚠️ **This is one seed and I am not calling it anything.** The right envelope for it is per-metric
and per-arm on *this* dataset (§4.2a) and has never been measured on tier-1; the 0.0335 in the
report's own footer is the **fixture** envelope and using it here would be exactly the
cross-dataset envelope substitution §4.2a exists to forbid. What is fair to say: the shipped
decision to leave detection calibration off rests on two measurements, **neither on tier-1**, and
this run is a reason to check that before the paper states it as settled. Recorded as an
observation with its own caveat, not as a finding, and **not** as a new experiment.

### What A7 still needs

The five remaining fits: seeds 2 and 3 off, seeds 1–3 with `--w-thick 0.2 --w-prog 0.2`. Plus the
wall clock for this one, which decides whether they run at all.

⚠️ **`AssumedThicknessWarning` is on**: tier-1's 22 µm thickness is defaulted from median spacing,
not measured, so every A7 number rests on it and the write-up says so.

⚠️ **`provenance.source = "defaults"`, not a selection.** Both arms will share it, so the
comparison is valid, but A7 then measures SEFL's contribution **at `Config` defaults**, not to a
selected configuration. The gates match what ships (`resample` / `correlated` / `zinb-flow` /
`medcpt`), so this is a caveat on the wording rather than on the comparison — the result is "SEFL
adds nothing at the default budget of 1200 steps", not "SEFL adds nothing to the shipped model".

---

## Replication — the cosmx staged gate: conditions pass, the text channel does not (2026-09-01)

`reports/t09_zeroshot_ceiling_morans_cosmx.{md,json}`, `reports/t09_text_coverage_cosmx.json`.
`cosmx_nsclc_3d`, holdout `paper_2_4`, 225 981 cells x 960 genes, split **769 kept / 191 held
out**, seed 7. No fits.

### 1. Every pre-registered stop condition passes, cleanly

| condition | threshold | measured | |
|---|---|---|---|
| **(a)** held-out ceiling − shuffled floor | >= 0.50 | ceiling **0.9960**, floor **0.0397**, room **0.9563** | pass |
| **(b)** held-out ceiling / kept ceiling | >= 0.80 | 0.9955/0.9969 and 0.9965/0.9972 = **~1.00** | pass |
| **(d)** `self` = 1.0 | exact | 1.0 on all four rows | pass |
| **(d)** constant field's normalised input bitwise row-identical | required | `input_rows_identical: true`, `input_std_max: 0.0`, all four rows | pass |

**§4.2c's instrument transferred to a new dataset and gave the same answer.** The constant field
is degenerate here too — exact zero per-gene variance, Moran's I is `0/0`, and what comes back is
float32 round-off (`precision_drift` **0.0758–0.1509**, and float64 disagrees with float32 by more
than the value itself on `section_5/kept`: 0.2929 vs 0.1595). The **boolean** test settles it
where the two thresholds that failed on `deep_starmap` would have been guesswork again. The
shuffled floor is justified by measurement on this dataset, not inherited.

The room, 0.9563, is within 0.001 of `deep_starmap`'s 0.9574 — which is what the pre-registration
wanted when it chose this dataset for holding the metric's geometry nearly constant. Split-half R
is **0.991–0.994**, higher than deep's, so the envelope should be no worse.

### 2. The split is clean by the criterion I wrote

Held-out summary rate **13.6 %** against kept **12.0 %** — gap **1.6 points**. The pre-registered
coverage check was about a *differential*: "a count well outside [the metadata-blind range] would
mean the stratified draw picked up a metadata bias and A1/A2 were handicapped for a reason
unrelated to the text channel." There is no differential. **By the check I actually wrote, this
split passes.**

### 3. ⚠️ And the dataset fails a check I never wrote. The replication cannot run.

| | `deep_starmap` (where the effect was found) | `cosmx_nsclc_3d` |
|---|---|---|
| held-out genes | 204 | 191 |
| in the metadata table | **1017 / 1017 (100 %)** | **121 / 960 (12.6 %)** |
| held-out with a summary | **192 (94.1 %)** | **26 (13.6 %)** |
| held-out bare-symbol-only | **0** | **163 (85.3 %)** |
| descriptor length, median | **546 chars** | **6 chars** |

**Six characters is the gene symbol.** For 85 % of the held-out genes there is no text at all.

**Why that is fatal rather than merely weakening.** A2 is `medcpt` pure text; on a bare symbol it
is MedCPT encoding a *symbol string*. `specs/10` §7 already names this exact condition — "every
prior STARmap number was produced with embeddings built from zeros or from a bare symbol, so
`text_emb_mode=medcpt` was `lookup` in all but name (ablation A3)."

**And the failure mode is the dangerous one: a false SUPPORT.** A2 on a bare symbol still yields a
consistent, gene-specific vector — a hash of the string — while A4 is `norm(0)`, a void. "An
arbitrary but consistent per-gene vector beats a void" would clear **every** pre-registered
criterion, including the sign flip's, while supporting a claim far weaker than *"text places genes
the model never saw"*. **The criteria as written cannot tell the two apart.** Running this would
have produced a number that looked like the replication and was not.

⚠️ **This is a hole in my pre-registration and it is mine.** I put the coverage check in the order
of operations and wrote thresholds only for (a) and (b). The only coverage criterion I set was the
differential one — because `deep_starmap` had 100 % absolute coverage and it did not occur to me to
bound it. A check without a threshold is a step in a recipe, not a gate.

### 4. The cause is a resource mismatch, not a property of the dataset

`resources/gene_meta.parquet` is **2 155 rows, every one `species_resolved = 10090` (mouse)**.
`cosmx_nsclc_3d` is **human NSCLC**. The table does not contain the panel.

It is worse than absent for the 121 that do hit. **1 020 of the table's rows carry ALL-UPPERCASE
(human-style) symbols with `ENSMUSG` Ensembl ids** — mouse genes fetched under uppercase spellings,
`species_requested: mouse` on every one. Symbol lookup is case-insensitive
(`text.py`'s `match_symbol` compares `casefold()`), so a cosmx symbol like `A2M` resolves to the
**mouse** gene `A2m`'s record: mouse `full_name`, mouse `ensembl_id`.

Stated fairly, because it cuts both ways: for **807** of those 1 020 the *summary text* is already
the **human orthologue's** (`summary_source_taxid: 9606`), so the biology prose would be roughly
right for a human panel by accident. For **161** it is native mouse text attached to a human gene.
Either way the table is declared mouse, keyed to mouse genes, and is not the panel's table.

`deep_starmap` is unaffected: its symbols are title-case mouse symbols and matched title-case rows,
which is why its coverage was 1017/1017 with zero bare symbols.

### 5. What this costs, and what it does not

**The replication is blocked on a metadata fetch, not on 47 core-hours.** The dataset is fine — the
ceiling, the floor, the split balance and the referent validity all pass, and its geometry matches
`deep_starmap`'s to within 0.001. What is missing is a human gene-metadata table.

`scripts/build_gene_meta.py` is the project's one sanctioned online step, and its own docstring
fixes the constraint: **"One organism per table."** A human table must be built **separately** and
`Config.gene_meta_path` pointed at it for cosmx runs — merging human symbols into the mouse table
would let one symbol resolve to two genes, which is the thing that rule exists to prevent.

    python scripts/build_gene_meta.py --species human --symbols-from <cosmx panel .h5ad> \
        --out resources/gene_meta.cosmx_human.parquet

Then re-run the coverage check with `--gene-meta resources/gene_meta.cosmx_human.parquet` and
**stop again** to read it.

### 6. Pre-registered NOW, before the rebuilt numbers exist

Absolute coverage, on the **held-out** side, anchored on `deep_starmap`'s measured values with
slack, and written before any rebuild has run:

* **PROCEED** — all of: summary rate **>= 0.80** (deep: 0.941); bare-symbol-only **<= 0.10** of
  held-out genes (deep: 0.000); median held-out descriptor **>= 200 characters** (deep: 546).
* **STOP** — any of those fails. The text channel is then too thin for A2 to be a statement about
  text, and a positive would be uninterpretable in the specific way described in §3.
* **Unchanged**: the differential check. Held-out and kept summary rates must not differ by more
  than a metadata-blind draw predicts.
* **NEW, and it is the defect found here** — a **species check**: every row used must resolve to
  **human (9606)** with an `ENSG` Ensembl prefix. A rebuilt table that still returns mouse rows for
  a human panel means the text channel is describing the wrong organism, and the run does not
  happen regardless of how good the coverage looks.

200 characters is roughly one sentence of description — far below `deep_starmap`'s 546 and far
above a 6-character symbol. The thresholds are set from the dataset where the effect was found, not
from what a rebuild is expected to produce, and they are on the record before it runs.

---

## Replication — the human rebuild resolved nothing. STOP, and the numbers are not about the panel (2026-09-01)

`reports/t09_text_coverage_cosmx_human.json`.

### Against the thresholds pre-registered before it ran

| criterion | threshold | measured | |
|---|---|---|---|
| held-out summary rate | >= 0.80 | **0.000** | **STOP** |
| held-out bare-symbol-only | <= 0.10 | **1.000** (191/191) | **STOP** |
| median held-out descriptor | >= 200 chars | **6** | **STOP** |
| species check (9606 / `ENSG`) | required | **not evaluable — no resolved rows** | — |
| differential (kept vs held-out) | metadata-blind | gap **0.000** | pass |

All three fire. **The verdict is STOP and it is not close.**

### ⚠️ But these numbers are a property of the build, not of the panel

**The rebuild went backwards.** The *mouse* table — the wrong organism, matching by accident —
resolved **121/960** symbols and gave **26** held-out summaries. The purpose-built *human* table
resolved **0/960**. A correct human table cannot do worse than an accidental mouse one on a human
panel: human is the best-covered organism in mygene.info.

**`n_full_name` is 0 across all 960 rows.** In every real table in this project a row that exists
carries a full name (`n_full_name == n_in_table`; the mouse table: 28/28 and 93/93). Zero full
names over 960 rows does not mean "these genes have no description" — it means **no row resolved
at all**. `n_in_table: 960/960` is the giveaway read the wrong way: every symbol is present, as a
**symbol-only row**.

**Reproduced locally.** With `mygene` absent, `build_gene_meta` warns
`GeneMetaUnavailableWarning: mygene.info lookup for N symbol(s) failed (ModuleNotFoundError: No
module named 'mygene'); falling back to symbol-only rows`, writes the table, and **exited 0**.

**Leading hypothesis, to be confirmed from the build's console output, not asserted**: `mygene` is
declared in the `extra` optional-dependency group, **not** in `dev` — so `make install` ("editable
install with dev extras") does not provide it, and the project's one sanctioned online step is
unrunnable after a standard install. The string to look for is
`GeneMetaUnavailableWarning ... No module named 'mygene'`. If instead the warning names a network
error, the cause is outbound access; if it names unmatched symbols, the cause is `--species`.

    pip install -e '.[extra]'          # what the online step needs, and what dev does not give

⚠️ **I am not moving `mygene` into `dev`.** `dev` is test and lint tooling and `extra` is the right
group for it semantically; the gap is that nothing says which install the online step needs. Flagged
for the user's decision, not applied.

### The defect that let this reach a coverage report: a silent fallback (Convention 6)

`build_gene_meta.py` printed `with full_name 0/960` and **returned 0**. Its only failure paths were
row-count mismatches — an `--overwrite` count, or rows lost in a merge — so a build in which
*nothing resolved* looked exactly like a success. The table is written, every gene is "in the
table", and the sole signal is a printed line nobody is obliged to read.

**Fixed.** `MIN_RESOLVED_FRACTION = 0.5`: a build that is not `--offline` and resolves fewer than
half its rows to a full name now **exits 1** and names the three causes in order — `--offline` not
passed, outbound access, `--species`. It points at `--dump-raw <SYMBOL>`, which prints mygene's raw
response for one symbol and writes nothing.

The constant is **a structural-failure detector, not a quality bar**: a panel that reaches
mygene.info resolves near 100 % (`deep_starmap` 1017/1017, the mouse table 2155/2155), so half is
far below any real build and far above zero. `--offline` is exempt, because symbol-only rows are
what it is for.

Verified both ways: `--offline` on three symbols exits **0** with `full_name 0/3`; a non-offline
build that resolves nothing exits **1** with the diagnostic.

### Where the replication stands

**Not refuted, and not run.** Every scientific gate on `cosmx_nsclc_3d` passed — room 0.9563,
ceiling ratio ~1.00, `self` = 1.0, constant field degenerate by §4.2c's boolean test, split
balanced. The dataset is good and its geometry matches `deep_starmap`'s to within 0.001. What is
missing is still a resolved human gene-metadata table, and the pre-registered thresholds stand
unchanged for the next attempt — they have now been applied once and stopped a run, which is what
they were written for.

---

## Replication — the human table resolves, and the coverage check gains `--species` (2026-09-01)

**The rebuild worked once `mygene` was installed**, which confirms the leading hypothesis recorded
above: the failure was the missing dependency, degrading silently to symbol-only rows. The
`MIN_RESOLVED_FRACTION` guard added for it would have caught the bad build at source.

Reported by the user from the build: **949/960 full names, 946/960 summaries (98.5 %), all
`native` human (0 orthologue), all `ENSG`, taxid 9606.**

⚠️ **That is a better table than `deep_starmap`'s**, and the difference matters for what A2 would
be demonstrating. `deep_starmap`'s summaries are **83 % human-orthologue** text describing a
*mouse* gene — so its A2 result is "MedCPT places a mouse gene from mostly human prose". cosmx's
are **100 % native**: same organism as the panel. **If the replication runs, its A2 is a cleaner
test than the original**, and the write-up must say the two are not the same demonstration.

### `--species`, added because the guard was right

The check refused the human table:

    GeneMetaError: ... holds rows resolved to species ['9606'] but 'mouse' (taxid 10090) was
    requested.

`t09_zeroshot_text_coverage.py` took the species from `Config.mygene_species` with no way to say
otherwise, so a `--gene-meta` pointing at another organism could only fail. **The guard is correct
and is not relaxed.** What was missing was a way for the caller to state both halves of one
statement.

* `--species` added, defaulting to `Config.mygene_species`.
* **Required whenever `--gene-meta` is given.** Inferring the organism from the table would be the
  silent fallback Convention 6 forbids, so the script refuses and its message names
  `Config.mygene_species`, the table's actual `species_resolved`, and what to do.
* A mismatch is caught **before** `load_gene_meta`, so the error names *both arguments the caller
  passed* rather than only the mismatch it found.

Verified on all four paths: default (mouse table, no flags) **exit 0**; `--gene-meta` without
`--species` **exit 1**; `--gene-meta` with the wrong `--species` **exit 1**, naming both; correct
pair **exit 0**.

### The 11 unresolved symbols, and where they land

`HLA.A` for `HLA-A` — **the panel writes a dot where a hyphen belongs**, so a handful of HLA
symbols are mangled at source and mygene cannot match them.

An unresolved symbol is written as a **symbol-only row**, which is the trap: it counts as "in the
table" while carrying no text at all. Previously nothing distinguished it from a resolved row with
a thin description, and nothing said which **side of the split** it landed on — which is the
question that matters, because a bare symbol on the held-out side is a bare symbol *in the arm
under test*.

The coverage report now carries `n_unresolved_in_table` and `unresolved_symbols` **per side**, and
prints them. It is a different quantity from `n_bare_symbol_only`, which counts both causes
together: a symbol **absent** from the table and a symbol **present but unresolved** produce the
same empty descriptor and had the same count. Verified against a deliberately built symbol-only
table.

**The held-out count is not yet known** — the cosmx coverage check has to be re-run under the new
flag to produce it. If any of the 11 fall in the held-out 191 they are bare symbols in A1/A2's own
arm, and the number belongs in the report beside the summary rate.

    python scripts/t09_zeroshot_text_coverage.py --split reports/t09_gene_split_cosmx.json \
        --gene-meta resources/gene_meta.cosmx_human.parquet --species human \
        --out reports/t09_text_coverage_cosmx_human.json

⚠️ **The pre-registered thresholds are unchanged and are not yet met by a measurement.** 98.5 % is
a property of the *table*; the criteria are on the **held-out side of the split**, and until that
run exists there is no number to check them against. On the panel-wide figures they would pass
comfortably — which is a reason to run the check, not a substitute for it.

---

## A7 — seed 1, both arms. The SEFL arm collapsed the field (2026-09-01)

`reports/t10_a7_off_s1.{md,json}` (`19cf1544f0cc5fcc`, `w_thick = w_prog = 0`) and
`reports/t10_a7_on_s1.{md,json}` (`9209091d5d27d668`, `w_thick = w_prog = 0.2`, `w_cross = 0` in
both). Tier-1, seed 1. **Seeds 2 and 3 outstanding — no verdict here.**

### 1. The uncalibrated table, and it is a trap

| metric | SEFL off | SEFL on | delta |
|---|---|---|---|
| `paper_morans_pearson` | +0.2340 | **+0.2858** | **+0.0519** |
| `paper_gearys_pearson` | +0.2357 | **+0.3028** | **+0.0671** |
| `paper_umap_mixing` | +0.7976 | +0.5090 | **−0.2886** |
| `paper_marker_field_r` | +0.4472 | +0.1171 | **−0.3301** |
| `paper_marker_depth_r` | +0.6289 | +0.2154 | **−0.4135** |
| `paper_celltype_localization` | +0.7591 | +0.4167 | **−0.3424** |
| `paper_gene_mean_spearman` | +0.9825 | +0.6546 | **−0.3279** |

⚠️ **Reading that table alone gives the wrong answer.** It says SEFL helps the two autocorrelation
metrics and hurts everything else. What actually happened is that the SEFL arm's **anatomical field
went flat**, and `paper_morans_pearson` is a *correlation across genes*, not an amplitude — so it
stays positive on a dead field by correlating noise. R12 recorded exactly this trap: *"the predicted
values are squashed into a narrow band, so `paper_morans_pearson` is noise about zero and its
negative sign is meaningless — the interpretable quantity is `morans_median_pred` vs
`morans_median_gt`."*

### 2. Three independent witnesses to the collapse

| witness | SEFL off | SEFL on |
|---|---|---|
| calibration `i_gen` against a target of 0.2611 | 0.2547 — **97.5 %** of target | **0.0042 — 1.6 %** |
| `status` / `ell_z_status` | `converged` / `converged`, 3 iterations | **`target_unreachable` / `target_unreachable`, 0 iterations** |
| per-module `mean I_gen` (4 modules) | 0.2556, 0.3243, 0.1968, 0.3646 | **−0.0012, 0.0013, 0.0005, 0.0057** |

against a real `mean I_real` of 0.3871 / 0.3357 / 0.1614 / 0.4899 in both. **Every module's
generated Moran's I is zero to three decimal places.** The field carries no spatial structure at
all, and the length-scale search reported it in the only way it can: no `ell` reaches the target,
because the problem is not the length-scale.

**And the calibrated arm confirms the correlation is noise**: applying detection calibration takes
the SEFL arm's `morans_pearson` from +0.2858 to **+0.0704** and `gearys_pearson` to **−0.0208**. A
real signal does not do that; a correlation of near-zero values does.

### 3. ⚠️ The comparison is confounded, and the confound is downstream of the collapse

`apply_lengthscale` applies **only a `converged` axis** — a documented refusal, added after T09
measured that a non-converged value is "whichever grid point tied first". So:

* SEFL **off** generated at the calibrated `ell_xy = 77.0`, `ell_z = 48.0`;
* SEFL **on** generated at the `Config` defaults `ell_xy = ell_z = 100.0`.

**The two arms therefore differ in two things, not one.** That is not a clean A7 and the write-up
cannot claim it is.

It is also not a fixable confound: **no `ell` produces structure in a flat field**, and the
calibrator failed *because of* the collapse rather than alongside it. The honest statement is that
the SEFL arm cannot be given a calibrated length-scale, and that this is itself the result.

### 4. ⚠️ T07's collapse alarm cannot see this collapse

`check_collapse` fires when the mean per-gene **variance of drawn counts** falls below
`sefl_collapse_warn_fraction` of the real cells'. That is variance **across cells**, not spatial
structure. A field can be **uniform in space while retaining per-cell variance** — flat in
*position*, not flat in *value* — and passes the alarm untouched.

This run is exactly that shape. **Worth asking whether the fit printed a `CollapseWarning` at all**;
if it did not, the alarm has a gap that T07 could not have found on the fixture, and the gap is
that it watches the wrong axis for this failure mode. Recorded as a defect to confirm from the
console, not asserted.

### 5. R6's scope needs correcting

R6 recorded field flattening for **`w_cross`** and explicitly exonerated the other two: *"`w_thick`
and `w_prog` — these two are **not** broken."* On real tissue at the spec's own 0.2 they flatten the
field as thoroughly as `w_cross` did on the fixture. **R6 is a property of the SEFL consistency
block, not of `L_cross` alone**, and the sentence clearing the other two was a fixture result that
does not transfer.

### 6. What can and cannot be said yet

* **One seed, and the arms are confounded.** No verdict. Seeds 2 and 3 are outstanding.
* The deltas on the four interpretable metrics are **8.6x to 12.3x** the 0.0335 figure the report
  footer quotes — but that is the **fixture** envelope and §4.2a forbids carrying it to tier-1. The
  right envelope is per-metric and per-arm on this dataset and does not exist. The magnitudes are
  large enough that this is unlikely to change the direction; it does mean **no clearance may be
  quoted**.
* **The layout is identical across arms** (`n_pred` 4073/4169/4110 in both) because
  `layout_mode="resample"` copies real coordinates. So the entire difference — including
  `celltype_localization` — is in the **field and expression path**, not in cell placement. That
  narrows what A7 is measuring and is worth stating in the write-up.
* Both arms still carry `provenance.source = "defaults"` and `AssumedThicknessWarning`, so the
  result is about SEFL **at the default 1200-step budget on a defaulted section thickness**.

⚠️ **Timing**: neither JSON carries `fit_seconds`, so the runs predate the persistence fix or ran
from an un-updated tree. The ON arm ran anyway, so the "one fit before the other five" gate is
already spent; with 2 of 6 done the remaining question is only the last four.

---

## A7 — seed 2. The collapse reproduces, and the seed-1 "improvement" flips sign (2026-09-01)

`reports/t10_a7_{off,on}_s2.{md,json}`. Both arms at `sefl_rejection_max_rounds = 96`, so **SEFL's
terms were actually charging** — before that fix `L_thick` was inert on ~97 % of planes and
`L_prog` starved by the same loop. **Two seeds; `claim_min_seeds` is 3. No verdict.**

### 1. The collapse reproduces exactly

| witness | OFF s2 | ON s2 |
|---|---|---|
| `i_gen` against a 0.2577 target | 0.2463 — **95.6 %** | **0.0061 — 2.4 %** |
| `status` / `ell_z_status` | `converged` / `converged`, 2 iterations | **`target_unreachable` both, 0 iterations** |
| per-module `mean I_gen` | 0.2488, 0.3098, 0.1983, 0.3705 | **0.0059, 0.0037, −0.0007, 0.0012** |
| applied `ell_xy` / `ell_z` | 41.4 / 26.9 (calibrated) | **100 / 100 (Config defaults)** |

Seed 1 was 1.6 % of target; seed 2 is 2.4 %. **Two independent fits, same dead field.**

### 2. ⚠️ The seed-1 autocorrelation "gain" does not reproduce — it reverses

| metric | delta seed 1 | delta seed 2 | |
|---|---|---|---|
| `morans_pearson` | **−0.0519** (SEFL better) | **+0.2804** (SEFL worse) | **signs differ** |
| `gearys_pearson` | **−0.0671** (SEFL better) | **+0.3070** (SEFL worse) | **signs differ** |
| `umap_mixing` | +0.2886 | +0.2957 | agree |
| `marker_field_r` | +0.3301 | +0.3904 | agree |
| `marker_depth_r` | +0.4135 | +0.3734 | agree |
| `celltype_localization` | +0.3424 | +0.2140 | agree |
| `gene_mean_spearman` | +0.3279 | +0.2206 | agree |

(positive = SEFL off is better)

**This is the demonstration, not the argument.** After seed 1 I said the two autocorrelation
metrics rising on a collapsed field was R12's documented trap — a correlation *across genes* rather
than an amplitude, correlating noise at `I_gen ≈ 0.004`. A noise correlation can land anywhere, and
on the second seed it landed on the other side by 0.28–0.31. **Anyone who had read seed 1's table
alone and concluded "SEFL helps autocorrelation" would have been refuted by one more fit.**

### 3. The first per-arm across-seed spread measured on tier-1 — and the fixture envelope is 5.5x wrong

⚠️ **Two seeds give a *range*, not an envelope: these are lower bounds** (`claim_min_seeds` = 3).

| metric | OFF range | ON range | worst (§4.2b) | mean delta | vs worst |
|---|---|---|---|---|---|
| `morans_pearson` | **0.1839** | 0.1483 | 0.1839 | +0.1143 | 0.6x — **not readable** |
| `gearys_pearson` | 0.1848 | 0.1893 | 0.1893 | +0.1200 | 0.6x — **not readable** |
| `umap_mixing` | 0.0517 | 0.0447 | 0.0517 | +0.2921 | **5.6x** |
| `marker_field_r` | 0.0844 | 0.0241 | 0.0844 | +0.3602 | **4.3x** |
| `marker_depth_r` | 0.0812 | 0.1212 | 0.1212 | +0.3934 | **3.2x** |
| `celltype_localization` | 0.0093 | 0.1192 | 0.1192 | +0.2782 | **2.3x** |
| `gene_mean_spearman` | 0.0033 | 0.1040 | 0.1040 | +0.2742 | **2.6x** |

**The `OFF`-arm `morans_pearson` range alone is 0.1839 — 5.5x the 0.0335 figure the report footer
quotes.** That footer figure is the **fixture** envelope, and §4.2a says an envelope is per-metric
and per-arm; this is the first direct measurement of how wrong carrying it across datasets would
be. It also vindicates §4.2a's other half: the worse arm **alternates by metric** here too — `OFF`
is worse on `morans`/`umap`/`field`, `ON` on `gearys`/`depth`/`loc`/`gms`.

**Provisionally**: five metrics show SEFL costing, signs agreeing on both seeds, at **2.3x–5.6x**
the worse arm's spread. The two autocorrelation metrics are **not readable** — their signs disagree
*and* their effect is inside the spread, which is exactly what an uninterpretable statistic on a
dead field looks like.

### 4. A finding about the calibration, not about SEFL

The `OFF` arm's **calibrated** `ell` moved a long way between seeds while the **fitted** `ell` — a
property of the data, identical in both — barely moved:

| | seed 1 | seed 2 | apart |
|---|---|---|---|
| fitted `ell_xy` (variogram) | 116.3 | 125.6 | **7 %** |
| **applied** `ell_xy` (bisection) | 77.0 | 41.4 | **1.86x** |
| **applied** `ell_z` | 48.0 | 26.9 | **1.78x** |

**The bisection amplifies seed noise by an order of magnitude relative to the fit it starts from**,
and that is a large part of why the `OFF` arm's `morans`/`gearys` spread is 0.18. It belongs to R1,
not to A7, and it means a shipped `ell` on tier-1 is a considerably less determinate quantity than
a single run's "converged" suggests. Recorded here because A7 is where it became visible; it is not
this experiment's result.

### 5. What the budget fix did to this experiment

Both arms ran with `sefl_rejection_max_rounds = 96`. Before that, `L_thick` returned exact zeros on
~97 % of planes and `L_prog` was starved by the same loop at the same rate. **So the fix is what
made A7 a real experiment — and the real experiment says that SEFL, once its terms actually charge,
destroys the anatomical field.** Had the budget stayed at 16, A7 would have measured two arms that
barely differed and reported "SEFL does nothing", which would have been wrong for a reason nobody
could have seen from the six-metric table.

It also means the two terms cannot be separated: both were starved before and both are active now.

### 6. Standing caveats, unchanged

* **Two seeds.** `claim_min_seeds` = 3. Seed 3 outstanding.
* **The arms remain confounded**: `OFF` generates at a calibrated `ell`, `ON` at the `Config`
  defaults, because `apply_lengthscale` correctly refuses a non-converged axis. Not fixable — no
  `ell` produces structure in a flat field — but it means this is not a single-variable comparison.
* Layout identical across arms (`n_pred` 4073/4169/4110 everywhere), so the whole difference is in
  the field and expression path.
* `provenance.source = "defaults"` and `AssumedThicknessWarning` on every run.
* Still no `fit_seconds` in any artifact.

---

## A7 — COMPLETE. Three seeds, both arms. SEFL is harmful on real tissue (2026-09-01)

`reports/t10_a7_{off,on}_s{1,2,3}.{md,json}`. Tier-1 `starmap_visual_cortex`/`paper_2_4_6`,
`w_cross = 0` in both arms, `w_thick = w_prog = 0.2` in the `on` arm, all six fits at
`sefl_rejection_max_rounds = 96`. `claim_min_seeds = 3` **satisfied**.

### 1. The collapse reproduces on every seed

| | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| **ON** `i_gen` as % of its own target | **1.61 %** | **2.38 %** | **1.35 %** |
| **ON** largest \|module `I_gen`\| (4 modules) | 0.0057 | 0.0059 | 0.0028 |
| **ON** `status` / `ell_z_status` | `target_unreachable` both, **0 iterations** | same | same |
| **OFF** `i_gen` as % of target | 97.5 % | 95.6 % | 94.1 % |
| **OFF** `status` | `converged`, 3 it. | `converged`, 2 it. | `converged`, 3 it. |

**Three independent fits, three dead fields.** Every gene module's generated Moran's I is under
0.006 in absolute value against real values of 0.16–0.49. The length-scale search reports it the
only way it can, and the `ell_xy` it returns when it fails is **56.5, 20.1, 163.3** across the three
seeds — an **8.1x** spread, which is precisely the "whichever grid point tied first" that
`apply_lengthscale`'s refusal exists to keep out of a shipped config. The refusal worked.

### 2. The verdict, under §4.2's own construction

Medians across seeds, min–max spread per arm, effect against the **worse arm's** spread (§4.2b):

| metric | OFF median | OFF spread | ON median | ON spread | worst | signs | median Δ | vs worst |
|---|---|---|---|---|---|---|---|---|
| `paper_umap_mixing` | 0.8493 | 0.0528 | 0.5090 | 0.0632 | 0.0632 | **3/3** | +0.2957 | **4.68x** |
| `paper_marker_field_r` | 0.5316 | 0.1040 | 0.1171 | 0.0450 | 0.1040 | **3/3** | +0.3904 | **3.76x** |
| `paper_gene_mean_spearman` | 0.9797 | 0.0033 | 0.6546 | 0.1193 | 0.1193 | **3/3** | +0.3279 | **2.75x** |
| `paper_celltype_localization` | 0.7499 | 0.0181 | 0.4167 | 0.2210 | 0.2210 | **3/3** | +0.3424 | **1.55x** |
| `paper_marker_depth_r` | 0.6973 | 0.0812 | 0.2154 | 0.3168 | 0.3168 | **3/3** | +0.4135 | **1.31x** |
| `paper_gearys_pearson` | 0.4205 | 0.2685 | 0.1135 | 0.3687 | 0.3687 | 2/3 | +0.3070 | 0.83x |
| `paper_morans_pearson` | 0.4179 | 0.2684 | 0.1375 | 0.3607 | 0.3607 | 2/3 | +0.2804 | 0.78x |

(positive Δ = SEFL **off** is better)

**Five of seven metrics: SEFL costs, signs agreeing on every seed, at 1.31x–4.68x the worse arm's
own spread.** The two autocorrelation metrics do **not** clear (0.78x, 0.83x) and their signs
disagree 2/3 — exactly as predicted once the field is dead, and the reason is in §3.

**A7's answer: SEFL, as specified at `w_thick = w_prog = 0.2`, is harmful on real tissue. It ships
at zero because it should.** The paper's SEFL section can now be written, and it is a negative.

### 3. The seed-1 reading was a trap and the full set proves it

`morans_pearson` delta by seed: **−0.0519, +0.2804, +0.5772.** Seed 1 said SEFL helps; seeds 2 and
3 say it hurts, by ten times as much. `gearys_pearson` does the same: −0.0671, +0.3070, +0.5701.

These are **correlations across genes, not amplitudes**. On a field whose `I_gen` is 0.003–0.006
they correlate noise, so they wander over a 0.63-wide range and their arm spread (0.36–0.37) is
**seven times** the other five metrics'. R12 recorded the trap; A7 demonstrates it three times over.

⚠️ **This is the reporting lesson of the whole ablation.** A single-seed six-metric table said
"SEFL improves autocorrelation and hurts everything else". Two of those seven numbers were
unreadable, and the two were the ones a reader would have led with.

### 4. What the budget fix did, and why A7 could not have been run before it

Every fit here used `sefl_rejection_max_rounds = 96`. At the shipped 16, `L_thick` returned exact
zeros on ~97 % of planes and `L_prog` was starved by the same loop at the same rate. **A7 at the old
budget would have compared two nearly identical models and reported "SEFL does nothing" — a false
negative invisible in every number the run produces.** The fix is what made this a real experiment,
and the real experiment returns a strong negative.

It also means **the two terms cannot be separated**: both were starved before, both are active now,
and A7 tests them jointly. `w_thick` alone and `w_prog` alone are unmeasured.

### 5. R6's scope, corrected

R6 recorded field flattening for **`w_cross`** and cleared the other two: *"`w_thick` and `w_prog`
— these two are **not** broken."* That was a fixture result. On real tissue at the spec's own 0.2
they flatten the field as thoroughly as `w_cross` did, on three seeds out of three. **R6 is a
property of the SEFL consistency block as a whole.**

### 6. Two findings that are not A7's

**The collapse alarm cannot see this.** `check_collapse` compares the per-gene **variance of drawn
counts** to the real cells'. That is variance *across cells*, not spatial structure: a field
uniform in **position** but not in **value** passes untouched. Three collapsed fits and — as far as
the artifacts show — no `CollapseWarning`. **The alarm watches the wrong axis for this failure
mode**, which is a T07 defect that only real data could expose. Worth confirming from a console log.

**The calibrated `ell` is seed-unstable, and that belongs to R1.** On the healthy `off` arm the
**fitted** `ell_xy` (a property of the data) is 116.3 / 125.6 / 116.3 — within **7 %** — while the
**applied** value is 77.0 / 41.4 / 56.5, a **1.86x** spread, and `ell_z` 48.0 / 26.9 / 23.3, a
**2.06x** spread. **The bisection amplifies seed noise by an order of magnitude relative to the fit
it starts from.** A shipped `ell` is far less determinate than one run's `converged` suggests, and
this is part of why `morans_pearson`'s off-arm spread is 0.27.

### 7. ⚠️ Standing caveats on A7's own verdict

* **The arms are confounded on `ell`.** `off` generates at a calibrated length-scale, `on` at the
  `Config` defaults (100/100), because `apply_lengthscale` correctly refuses a non-converged axis.
  Not fixable — no `ell` gives structure to a flat field, and the calibrator failed *because of* the
  collapse — but the comparison is **SEFL-on-at-defaults vs SEFL-off-at-calibrated**, and the
  write-up must say so.
* **`provenance.source = "defaults"`** on all six fits: this is SEFL at `Config`'s 1200-step budget,
  not against a selected configuration.
* **`AssumedThicknessWarning`** on all six: tier-1's 22 µm thickness is defaulted from median
  spacing, and `L_thick` is a thickness loss. Every A7 number rests on that default.
* The layout is identical across all six fits (`n_pred` 4073/4169/4110), so the entire difference is
  in the field and expression path, not in cell placement.
* No `fit_seconds` in any artifact — the runs predate the persistence fix.

### 8. The fixture envelope, measured wrong by 8x

The report footer quotes **0.0335** from `reports/envelope_synthetic.md` and says a difference
smaller than that is a tie. Measured here on tier-1, per arm: spreads run **0.0033 to 0.3687**, a
**112x** range across metric and arm, and `morans_pearson`'s off-arm spread alone is **0.2684 — 8x
the fixture figure**. §4.2a said an envelope is per-metric and per-arm; this is the direct
measurement, and it also shows the worse arm **alternating** (`off` worse on `field`; `on` worse on
the other six). ⚠️ **The `t09_ship_starmap.py` footer should stop quoting a fixture envelope for a
real-data run** — it invites exactly the comparison §4.2a forbids.

---

## Replication — the staged gate is CLEARED. Every pre-registered condition met (2026-09-01)

`reports/t09_text_coverage_cosmx_human.json`, against
`resources/gene_meta.cosmx_human.parquet` with `--species human`.

### 1. The coverage thresholds, written before this run existed

| criterion | threshold | measured | |
|---|---|---|---|
| held-out summary rate | >= 0.80 | **0.9843** | **PASS** |
| held-out bare-symbol fraction | <= 0.10 | **0.0157** (3/191) | **PASS** |
| held-out median descriptor | >= 200 chars | **709** | **PASS** |
| differential, held-out vs kept | metadata-blind | gap **−0.0014** | **PASS** |
| species: human / 9606 | required | enforced — the run could not have produced output otherwise | **PASS** |

⚠️ **The species check is only half-enforced by this artifact.** `t09_zeroshot_text_coverage.py`
refuses a taxid mismatch, so the run completing *is* the taxid half of the check. The `ENSG`
Ensembl-prefix half rests on the **build's** console report, not on this file. Stated so the
provenance of each half is clear.

Together with the ceiling gate already passed — room **0.9563**, held-out/kept ceiling ratio
**~1.00**, `self` = 1.0 on every row, constant field degenerate by §4.2c's boolean test — **the
replication is cleared to run.** Both criteria (Part 1 `morans_pearson` A2−A3, Part 2 the sign flip
on A2−A3 across both pools) were written before any cosmx number existed and are unchanged.

### 2. The 11 unresolved symbols are one biological family, and the split is benign

| | held-out | kept |
|---|---|---|
| unresolved | **3** (1.57 %) | **8** (1.04 %) |
| symbols | `HLA.DPA1`, `HLA.DRB1`, `HLA.DRB5` | `HLA.A`, `HLA.B`, `HLA.C`, `HLA.DPB1`, `HLA.DQA1`, `HLA.DQB1`, `HLA.DRA`, `HLA.E` |

**All eleven are HLA genes, and all eleven fail for one reason: the panel writes a dot where a
hyphen belongs** (`HLA.A` for `HLA-A`). Nothing about the metadata table is wrong — the symbols are
mangled at source.

The numbers clear the threshold comfortably, and the split across sides is even (1.6 % vs 1.0 %).
⚠️ **But these are not a random eleven.** They are the MHC class I and II loci, and on an NSCLC
tumour panel that is the family whose spatial structure — immune infiltration, antigen-presentation
zoning — is most likely to be strong. If those three held-out genes carry high real Moran's I, A1
and A2 are handicapped on genes that contribute disproportionately to `morans_pearson`. Three of
191 is small; being a coherent, plausibly high-signal family is not the same as being three random
genes, and the write-up should name them rather than report "1.6 % bare symbols".

**Cheap to fix, and worth doing before the fits, not after.** A symbol repair — if a symbol fails to
resolve and replacing `.` with `-` resolves it, use that — recovers all eleven. It costs a metadata
rebuild and a coverage re-run, minutes; recovering them after six ~8-hour fits costs the fits.
**This is a judgement call, not a gate**: the gate passes either way, and it is the user's decision
whether to spend the minutes. Not implemented.

### 3. ⚠️ cosmx's text channel is *stronger* than `deep_starmap`'s, in two independent ways

| | `deep_starmap` (where the effect was found) | `cosmx_nsclc_3d` |
|---|---|---|
| held-out summary rate | 0.941 | **0.984** |
| held-out bare symbols | 0 | 3 |
| median descriptor | 546 chars | **709 chars** |
| summaries that are an **orthologue's** | **83 %** | **0 % — all native human** |

`deep_starmap`'s A2 demonstrates *"MedCPT places a **mouse** gene from mostly **human** orthologue
prose"*. cosmx's would demonstrate *"MedCPT places a **human** gene from **native human** prose"*,
with descriptors 30 % longer. **These are not the same demonstration and the paper must not let the
second silently stand in for the first.**

**The asymmetry has a consequence for how each outcome reads**, and it is worth fixing in advance:

* A **REFUTATION** on cosmx is the **more informative** outcome. The text channel there is
  unambiguously richer, so failing with better text is a stronger negative than failing with worse
  text would have been.
* A **SUPPORT** is correspondingly **weaker than it looks**. Part of any success could be the better
  descriptors rather than the mechanism transferring, and nothing in the design separates those.
  The honest write-up of a SUPPORT says "replicated on a dataset whose text channel is richer",
  not "replicated".

Recorded now, before the fits, so it cannot be read as a post-hoc discount of an unwelcome result
or a post-hoc inflation of a welcome one.

### 4. Where the replication stands

**Cleared, costed, not launched.** ~47 core-hours for 6 fits on 225 981 cells — extrapolated from
`deep_starmap`'s measured 3.74 h at 113 k cells and 1017 genes, against cosmx's 2x the cells and
0.94x the genes. The staged gate's remaining step is unchanged: **one timed fit before the other
five.** ⚠️ That gate has been bypassed twice now (step 2's and A7's), both times because the runs
went out before a duration was reported. `fit_seconds` is persisted in the run JSON since the A7
review; it will only help if the fits use an updated tree.

---

## Six follow-ups from A7 and the cosmx gate, applied (2026-09-01)

### 1. The symbol repair — dot for hyphen, before the fits

`spatialcpav25_gen.data.text.repair_symbol`: a symbol the first lookup failed on is retried with
`.` replaced by `-`. **General, not an HLA special case**, and consulted **only** for symbols the
first query already failed on, so it can never displace a genuine hit. The row is filed under the
**panel's own spelling**, so every downstream lookup still finds it, and a
`GeneMetaUnavailableWarning` names how many were repaired and with what substitution.

The rule is what R's `make.names()` does to a column header, which is how it reaches a panel. On
`cosmx_nsclc_3d` it recovers all **11 of 960** — every one an HLA locus, three of them held out.
**No new column**: `GENE_META_COLUMNS` is unchanged, because adding to it would make every existing
table "missing" a field and trip the rebuild refusal. The provenance lives in the warning.

⚠️ **The table has not been rebuilt yet.** The repair is code; the numbers are still the 3-held-out
/ 8-kept ones. Rebuild and re-run coverage before the fits.

### 2. The spatial collapse alarm — the second alarm in this project that could not fire

**Confirmed from code, not from a log**: `check_collapse` reads
`terms[DIAG_gene_variance_gen]`, which is `drawn.var(dim=0).mean()` — the variance of drawn counts
**across cells**, averaged over genes. Spatial organisation does not enter it. A field uniform in
*position* while retaining per-cell variance is invisible to it, definitionally.

⚠️ **The "never fired" half still needs the console log.** I can prove the statistic is blind; I
cannot prove no warning was printed without seeing the run's output. Send it if it is kept.

Added: `spatial_structure_ratio(generated, real, coords, cfg)` — median Moran's I of the generated
counts over the real cells' **on the same positions and one shared kNN graph** — plus
`check_spatial_collapse` and `Config.sefl_spatial_collapse_warn_fraction`. Both alarms now run at
every logged step and the ratio is recorded in `TrainHistory.spatial_ratio` on every run.

⚠️ **The threshold is 0.05 and is deliberately blunt.** The only anchors are A7's, and they come
from the *calibration's* construction (0.016 collapsed, ~0.95 healthy), while this field is read in
a **different** one — generated against real counts on the same batch cells — where no healthy
value has been measured. R12 is the warning against assuming they agree: a healthy `deep_starmap`
model carried only **15 %** of real tissue's counts-level Moran's I, so a threshold at the
calibration's healthy figure would cry wolf on a working model. 0.05 sits an order below the
healthy anchor and 3x above the collapsed one: **it would have caught A7 and cannot fire on
anything that still has structure.** Tighten it only once `spatial_ratio` has been logged on
healthy runs — which is why it is logged unconditionally. Setting it from the two numbers available
today would be the threshold-without-a-measurement this project has already got wrong twice
(§4.2c's drift cut and its CV cut).

`tests/test_sefl.py::test_spatial_collapse_alarm_catches_what_the_variance_alarm_cannot` constructs
the field that separates them — a real spatial gradient, spatially scrambled, so **per-gene variance
is preserved exactly** — and asserts the variance alarm stays silent while the spatial one fires.

### 3. The timing gate, enforced in code

**A gate you can skip by accident is not a gate.** It has now been bypassed three times: twice
because runs went out before a duration was reported, once because the loop was started early.

`t09_zeroshot_run.py` gains `timing_gate` / `write_timing`. The fit designated `--first-fit`
(default `2:medcpt`) runs unconditionally and **writes** `fit_timing.json`; every other fit
**refuses to start** until that file exists and carries a positive `fit_seconds`. Fit 1 also prints
the six-fit projection from its own measured time. `--no-timing-gate` exists and prints a refusal
line naming what was never measured, so a deliberate skip is visible in the artifact rather than
silent.

Verified on all four paths: fit 1 allowed; fit 2 **refused** with no record; fit 2 allowed once the
record exists; fit 2 **refused** when the record is present but carries no duration.

### 4. The fixture envelope is out of the real-data footer

`t09_ship_starmap.py` quoted 0.0335 from `reports/envelope_synthetic.md` and called smaller
differences ties, on every real-data run. Removed, and replaced with the measurement that shows why:
on tier-1 the per-arm across-seed spreads run **0.0033 to 0.3687**, a **112x** range, and
`morans_pearson`'s alone is **0.2684 — 8x** the fixture figure. A tie declared against 0.0335 there
is the cross-dataset comparison §4.2a forbids, printed by the project's own tooling.

### 5. `specs/10` §4.2e — the worked example, in the methods

A7's `morans_pearson` deltas **−0.0519, +0.2804, +0.5772** are now a methods subsection, not a risk
note: one seed inverted the sign of the headline on two of seven metrics, the mechanism is stated
(a correlation across genes correlating noise on a dead field, arm spread 0.36 — seven times the
other five metrics'), and three consequences are drawn — `claim_min_seeds` = 3 is not conservatism;
a correlation-shaped metric needs an amplitude reported beside it; and the five readable metrics all
agreed 3/3 while the two that disagreed were the two that were uninterpretable, which is a signature
rather than a coincidence.

### 6. `specs/10` — the replication's asymmetry, before the fits

Recorded beside E1's finding: `cosmx`'s text channel is **richer** than `deep_starmap`'s in two
independent ways (709 vs 546 median chars; **0 % vs 83 %** orthologue summaries), so a REFUTATION
there is the stronger negative and a SUPPORT must be written as **"replicated on a dataset whose
text channel is richer"**. In the spec before any fit, so it cannot read as a post-hoc discount.

---

## Replication — the repair worked; every gate now clears (2026-09-01)

`resources/gene_meta.cosmx_human.parquet` rebuilt with the dot-to-hyphen retry: **960/960 full
names**, all 11 HLA symbols resolved through the repair, and the warning names the substitution
with provenance.

### The held-out side is now clean by every measure

| criterion | threshold | before repair | after repair |
|---|---|---|---|
| held-out summary rate | >= 0.80 | 0.9843 | **1.0000** (191/191) |
| held-out bare-symbol fraction | <= 0.10 | 0.0157 | **0.0000** |
| held-out median descriptor | >= 200 chars | 709 | **720** |
| held-out **minimum** descriptor | — | 9 | **176** |
| orthologue summaries | — | 0 | **0 — all native** |
| summary-rate gap (held-out − kept) | metadata-blind | −0.0014 | **+0.004** |

**The minimum matters as much as the median here.** At 176 characters the *shortest* held-out
descriptor is still a real description, so there is no tail of effectively-bare genes hiding under
a healthy median — the failure mode the whole coverage check exists to catch.

### ⚠️ A third asymmetry, and it cuts the same way as the other two

The gap now runs **+0.4 points in the held-out side's favour**: the arm under test is, if anything,
slightly *advantaged* rather than handicapped. With the descriptors already longer (720 vs 546) and
native rather than orthologue prose (0 % vs 83 %), **all three differences push in the same
direction**, and `specs/10` now records the set: a REFUTATION on `cosmx` is the stronger negative,
and a SUPPORT must read **"replicated on a dataset whose text channel is richer"**.

None of the three is a defect. Together they mean the replication is a *more favourable* test than
the original, which is exactly why its failure would say more than its success.

## ⚠️ The `check_collapse` question is resolvable after all — correcting the conclusion

The user reported that `grep`ping for "collapse" across `runs/a7_on_s*/` returns nothing, the
console was not kept, and proposed recording the "never fired" half as **unresolvable**. **It is
not unresolvable, and the grep is weak evidence rather than none: warnings go to stderr, which was
never in that directory in the first place.**

`TrainHistory.collapse_alarms` — the list of steps at which the alarm fired — **is persisted**.
`write_checkpoint` stores `history=dataclasses.asdict(history)` and `save_checkpoint` writes the
whole payload with `torch.save`, so it is inside the resumable fit checkpoint, in a binary a text
`grep` cannot read:

    python - <<'PY'
    import torch
    for seed in (1, 2, 3):
        h = torch.load(f"runs/a7_on_s{seed}/fit_seed{seed}.pt", weights_only=False)["history"]
        print(seed, "collapse_alarms:", h["collapse_alarms"],
              " variance_ratio last:", h["variance_ratio"][-3:])
    PY

An **empty** `collapse_alarms` on a fit whose SEFL block was live past
`sefl_collapse_min_steps` is the positive result: the alarm was *checked* and did not fire. That is
the half the code-level proof cannot supply, and it is one command away as long as those
checkpoints still exist.

⚠️ **The qualifier in that sentence turned out to be the whole thing (2026-09-07).** "*on a fit
whose SEFL block was live*" is correct for A7, which ran SEFL on. On a **SEFL-off** fit — which is
every shipped run — the alarm was **never armed**, so an empty list there is not a negative result
of any kind. I wrote the qualifier without noticing it excluded the configuration that ships.
`_log_sefl` now arms on the step floor alone.

### And the artifact gap that caused the confusion is now closed

`model.pt` carries weights and config, never the history, so once a fit checkpoint is cleaned up
the alarm record is gone. **An alarm nobody can show fired later is half an alarm.**
`t09_ship_starmap.py` now writes an `alarms` block into every run's JSON:

```json
{"collapse_alarms": [], "spatial_collapse_alarms": [250, 300],
 "variance_ratio": {"min": 0.85, "median": 0.88, "last": 0.88, "n": 3},
 "spatial_ratio":  {"min": 0.004, "median": 0.02, "last": 0.004, "n": 3}}
```

Both alarm lists **and the trajectories they read**, so a later reader can tell *"did not fire"*
from *"was never checked"* — the distinction A7's artifacts could not make. A reused fit records
`null`, which says "not measured on this run" and is not the same as "did not fire". Exactly this
record on an A7 ON fit would have shown `collapse_alarms: []` beside `spatial_ratio.min ≈ 0.004`:
the variance alarm silent, the field dead, in one artifact.

---

## Replication — fit 1 timed at 3.50 h, and a hole the gate did not cover (2026-09-01)

`runs/fit_timing.json`: **12 602.8 s = 3.50 h**, seed 2, arm `medcpt`. The timing gate worked —
fit 1 ran, wrote the record, and the other five are unblocked by it.

### 1. ⚠️ The estimate was 2.2x too HIGH, and the cost model behind it was wrong

| | projected | measured |
|---|---|---|
| per fit | ~8 h | **3.50 h** |
| six fits | **~47 core-hours** | **21.0 core-hours** |

**The first over-estimate in this project, after five consecutive under-estimates.** That is not
an improvement in my estimating — it is the same error in the other direction, and the reason is
worth having:

I wrote *"cell count has driven the scaling on both measured points more than gene count. At ~2x
`deep_starmap`'s cells and 0.94x its genes, six fits project to ~47 core-hours."* **That premise is
structurally wrong at a fixed step budget.** `sample_batch` draws `Config.batch_cells` cells per
step regardless of how many the volume holds, so training cost is
`train_steps x batch_cells x genes_per_step` — **fixed** — plus a setup term that does scale with
the volume (load, retrieval index, PCA). At the same 2400 steps:

| dataset | cells | genes | measured |
|---|---|---|---|
| `deep_starmap` | 113 k | 1017 | 3.74 h |
| `cosmx_nsclc_3d` | **226 k** | 960 | **3.50 h** |

**Twice the cells, slightly less time.** The cost model should be **steps-driven, not
cell-driven**, and the tier-1 comparison that seemed to support the old model (16.5 k cells,
62 min) confounded cell count with a step budget half the size.

**Remaining five: 17.5 core-hours**, ~3.5 h wall five-up.

### 2. ⚠️ But the gate had a hole, and fit 1 may have fallen through it

`t09_zeroshot_run.py` had **no `--gene-meta` flag**. `base_config` sets only `BENCH3_KEYS`, so the
fit read `Config.gene_meta_path` — **`resources/gene_meta.parquet`, the mouse table** — under
`Config.mygene_species = "mouse"`, for a **human** panel.

**Nothing upstream would have raised.** `load_gene_meta` checks the table's rows against the
requested species and both say mouse, so the pairing is internally consistent and silent. The
human symbols then resolve **121 of 960** by case-insensitive accident (`A2M` matching mouse
`A2m`), leaving 839 bare.

**And the coverage gate cannot catch it, by construction**: `t09_zeroshot_text_coverage.py` reads
the table `--gene-meta` points *it* at, while the fit reads `Config.gene_meta_path`. **The gate can
pass on one table while the experiment runs on another** — which is precisely what it is for, and
precisely what it could not see.

⚠️ **This is unresolved for fit 1 and it decides whether that 3.50 h is a measurement or a void
run.** The run printed `text channel: {...}` with `gene_meta_path`, `n_with_meta` and `n_bare` in
it. **`n_with_meta` near 960 means the fit is sound; near 121 means it must be discarded and
re-run, along with the timing.** The remaining five must not start until that line is read.

### 3. The fix, so it cannot recur

* **`--gene-meta` and `--species` added** to `t09_zeroshot_run.py`, required together — the path
  and the organism are one statement, and inferring the second from the first is the fallback
  Convention 6 forbids.
* **`check_text_channel` refuses the fit** when under `MIN_PANEL_WITH_META = 0.5` of the panel
  carries metadata, on **either** arm — `lookup` is built from the same descriptors and applies
  its gate inside the embedding, so a `lookup` arm against the wrong table is wrong the same way,
  and A4's `norm(0)` void only means what it should if the text channel is what we think.
  The message names the likely cause and says explicitly that the coverage check cannot catch it.
* Verified: 960/960 allowed, 1017/1017 allowed, **121/960 refused**.

The constant is a **structural-mismatch detector, not a quality bar** — the same shape as
`MIN_RESOLVED_FRACTION` in the build script, and set the same way: half separates 12.6 % from
100 % with room on both sides and cannot fire on a real pairing.

---

## 🚨 CORRECTION — `check_collapse` fired on A7. 218 times. Nothing surfaced it (2026-09-01)

Read out of the fit checkpoints:

| seed | alarms | first | `variance_ratio` (last three logged) |
|---|---|---|---|
| 1 | **72** | step 250 | 0.153 / 0.193 / 0.188 |
| 2 | **74** | step 250 | 0.141 / 0.131 / 0.129 |
| 3 | **72** | step 250 | 0.113 / 0.105 / 0.107 |

Against a `sefl_collapse_warn_fraction` of **0.25**. The alarm fired from step **250 of 1200** on
every seed and kept firing.

### What I got wrong, in both halves

I wrote that `check_collapse` **"did not fire and could not"**. Both halves are false.

* **"Did not fire"** — it fired 218 times. My evidence was a `grep` for "collapse" over the run
  directories, which was never going to find anything: warnings go to stderr and the alarm history
  goes into a binary `.pt`. I treated the absence of a signal from an instrument I had not checked
  as the absence of the signal.
* **"Could not"** — the *general* argument was sound and is still why the spatial alarm exists: the
  two statistics are orthogonal, and
  `test_spatial_collapse_alarm_catches_what_the_variance_alarm_cannot` constructs the field that
  proves it (a real gradient scrambled in space, per-gene variance preserved **exactly**, Moran's I
  to zero). But **it does not describe A7**. There the field lost per-cell variance *as well as*
  spatial structure — ratio 0.105-0.193 — so the existing alarm caught it. I applied a valid
  general claim to a specific run without checking whether the run matched it.

**The spatial alarm is still worth having**, and its justification is now stated correctly in both
`Config.sefl_spatial_collapse_warn_fraction` and `check_spatial_collapse`: nothing guarantees the
two collapse together, and in A7 they happened to. **Justified by the orthogonality, not by a miss.**

### The real defect, and it is worse than the one I diagnosed

`history.collapse_alarms` was **populated**, **persisted**, and **read by no report**. Three fits
were reviewed here in detail across three rounds, and the model's own alarm — screaming from step
250 — never entered the review, because the artifacts I was reading do not carry it.

**An alarm that fires where nobody looks is worse than an alarm that cannot fire.** A blind spot is
at least honest about being one; this created the impression of a watched run. The whole "the
six-metric table is a trap" analysis was reconstructed from `i_gen` and the module tables, and the
model had been saying so directly for 950 steps.

⚠️ **It also means A7's ON arm was diagnosable at step 250 of 1200.** Nothing about A7's verdict
changes — SEFL is still harmful on three seeds — but "the run told us and the report did not print
it" is the finding, and it belongs beside the result rather than in a footnote.

### The fix

`t09_ship_starmap.py` now:

* prints the count to the console at fit time, with `⚠️ N COLLAPSE ALARM(S) FIRED ... Every number
  below is from a model that reported itself broken while training`;
* carries a **`## Collapse alarms`** section in the **markdown report**, not only the JSON. A count
  buried in JSON is barely better than one buried in a `.pt`. When it fired the section leads with
  🚨 and the sentence that no number in the report should be read as a property of the method until
  it is explained; when it did not, it said **"a measured silence rather than an absent
  measurement"** — ⚠️ **which was false for every SEFL-off run and is corrected 2026-09-07**: the
  alarms were armed only while SEFL was on, so on the shipped configuration that line described a
  check that never ran. The section now says so and directs the reader to the trajectory;
* renders **"not measured on this run"** for a reused fit — because `model.pt` carries weights and
  config but never the history, and that is **not** the same as "did not fire".

Rendered against A7 seed 1's real history, the section it would have produced reads:

    🚨 **72 ALARM(S) FIRED. Every metric in this report comes from a model that reported itself
    broken while training, and no number here should be read as a property of the method until
    that is explained.**

    * per-gene **variance** collapse: **72** firing(s), first at step 250

## Replication — fit 1 is void, as predicted (2026-09-01)

`n_with_meta` **121 of 960**, 839 bare, example descriptor `"AATK."`, `gene_meta_path
resources/gene_meta.parquet`. The fit read the **mouse** table for a human panel, exactly the
failure `check_text_channel` now refuses. The 3.50 h and its timing record are discarded.

**The measurement that survives is the cost model**, because it is a property of the code path
rather than of the metadata: at a fixed step budget the fit is steps-driven, not cell-driven. The
re-run will confirm it, and 21.0 core-hours remains the projection to beat.

## The healthy `spatial_ratio` trajectory — first measurement in the alarm's own construction (2026-09-04)

The re-run's log gives what `Config.sefl_spatial_collapse_warn_fraction`'s docstring had been
asking for since T10: the spatial ratio on a **real-data** fit that did not collapse. Until now the
only anchors were A7's, and they came from the *calibration's* construction (generated section vs
flanking real sections), which R12 says need not agree with this one.

**n = 241 logged steps.** Median **1.299**. First two samples 0.0168 and **-0.2010**. Last two
1.451 / 1.384. Restricted to after step 100 — the warm-up `Config.sefl_collapse_min_steps` already
skips — **n = 231, minimum 0.5467, zero samples below 0.05**. Four samples anywhere in the run were
negative, minimum **-0.2451**, all of them before step ~100.

So the threshold has a **10.9x margin on the measured floor** and 26x on the median. That is the
first evidence that 0.05 is not merely a guess bounded below by A7's 0.016, and it is why 0.05
**stays where it is**: moving a threshold on a single observation is the mistake this project has
recorded twice already, and a 10.9x margin is a real safety factor, not a reason to spend one.

⚠️ **It is the right construction and the wrong population, and the docstring now says so.** The
fit that produced these numbers had **SEFL off**. This alarm only arms when SEFL is **on**. The
population it will actually be applied to — SEFL-on, running, not collapsed — has **never been
observed on real data**: every SEFL-on real-data fit to date is A7, and A7 collapsed. 0.547
therefore bounds what a healthy *decoder* does on real tissue; it says nothing about what the
consistency losses do to the ratio while they train. Anyone reading 0.547 as headroom for the
armed alarm is reading a proxy.

### The sign handling was wrong, and the fix is not the obvious one

Four negative samples exposed a defect: `check_spatial_collapse` treated every ratio below 0.05 as
a collapse and printed **"The field is flat in SPACE"**. A negative ratio is not a flat field. It
is signal of the **opposite sign** — neighbouring generated cells *less* alike than random ones,
where the tissue's are more alike. Anti-correlated structure is not absent structure, and the
single message also ranked `-0.25` as a *worse* collapse than `+0.01`, which is backwards.

It is latent, not harmless: on this run every negative excursion fell inside
`sefl_collapse_min_steps`, so nothing fired. A longer warm-up, or a mid-training dip past step 100,
and the first message anyone reads is a wrong diagnosis of a real fault.

`check_spatial_collapse` now returns `"collapse" | "inversion" | None` and the trainer files them in
separate `TrainHistory` lists; `t09_ship_starmap.py` prints them as separate lines.

**The boundary is minus the threshold, not zero** — and the module's own fixture is why. A field
scrambled in space, the textbook collapse and the shape three A7 fits took, scores **-0.0133**: not
by accident, because Moran's I under a permutation null is `-1/(n-1)`. A boundary at zero would
have filed the canonical collapse as an inversion. Reading
`sefl_spatial_collapse_warn_fraction` **symmetrically** says what is meant — `|ratio|` inside it is
absent structure whichever side of zero the noise landed, and only anti-correlation beyond the
alarm's own resolution is an inversion. The healthy run's -0.2451 is five times outside it, so the
two classes are well separated in the data we have.

Reusing the existing field also avoids the standing `content_hash` hazard: `Config().content_hash()`
is **unchanged at `445414903717491c`**, so the re-run's checkpoint still resumes. Every change in
this entry is a docstring, a return type, a history list and a report line.

## `t09_zeroshot_score.py` had the same gene-metadata hole (2026-09-04)

Asked directly: does the scorer read gene text, and from where? **It does**, and the reasoning
that says it cannot is the reasoning that would have shipped six wrong scorings.

The tempting argument is that scoring only consumes checkpoints — the four arms differ in
`use_distill` and in which of two fits they load, and the embeddings are in the `.pt`. Both
halves of that are false where it counts:

* `build_and_fit` — shared with the fitter, by design — calls `embeddings_factory(volume)(cfg)`,
  which is `build_entity_embeddings(cfg, ...)`. That reads `cfg.gene_meta_path` and
  `cfg.mygene_species`, loads the table, and encodes the panel's descriptors through MedCPT
  **before** `load_state_dict` puts the fitted buffers over the top.
* `ZeroShotView` embeds an unseen gene as `forward_zero_shot(base.text_vecs[g])`. `text_vecs` is
  not an incidental buffer; on the held-out side it **is** the zero-shot input, and it is the
  quantity the whole experiment is about.

`arm_config` takes no metadata arguments, so the scorer built every config at
`gene_meta_path="resources/gene_meta.parquet"`, species `mouse` — the same default that voided
replication fit 1.

### It is not the identical failure, and the difference is worth stating

`gene_meta_path` and `mygene_species` are **both in `content_hash`** (verified: changing either
moves it off `445414903717491c`). So a scorer pointed at the mouse table would have failed the
checkpoint's `require_compatible` rather than producing a quiet wrong number. The run script's
hole was **silent**; this one was **loud but unreadable** — the failure arrives after a full
MedCPT encode of the wrong descriptors, and the message names a hash and no field. A reader
would have to diff two configs to learn that the gene table was the problem. It also means the
six scoring commands could not have run at all without the flags.

### The fix

Same shape as the fitter's, deliberately:

* `--gene-meta` / `--species`, **required together**, refused before any path resolution — the
  path alone leaves `mygene_species` at `mouse`, which makes `load_gene_meta`'s organism check
  agree with itself while the panel is another organism's.
* `check_text_channel` on the shared config, at the same `MIN_PANEL_WITH_META = 0.5`, run
  **before** `build_and_fit`, so the wrong table is named by a message about the table.
* The module docstring states why the scorer touches text at all, so the "it only scores"
  argument does not get re-derived by the next reader.
* `tests/test_text.py::test_the_zero_shot_scorer_reads_the_gene_table_and_refuses_the_wrong_one`
  pins all three: the flags reject either half alone, 121/960 is refused with the
  ANOTHER-ORGANISM message, and the guard is actually wired into the scorer rather than merely
  importable from it.

**The general lesson, since this is the second instance:** a script that shares a builder with
the fitter inherits every config field that builder reads. "It only scores" describes the
intent, not the code path.

## The zero-total guess — PRE-REGISTERED, before the instrument is written (2026-09-07)

**Status: a guess.** It is written here before `scripts/t09_zeroshot_pool_sparsity.py` exists, and
before any number, because a candidate explanation proposed after seeing its own measurement is
not a candidate explanation.

### What it claims

`select._normalised` library-size normalises **within the scored gene pool**: the size factor is
the sum over the pool's columns, not the panel's. A cell whose counts across the 191 held-out
genes sum to zero therefore gets an all-zero normalised row — `eps`-guarded, so no error, just a
row carrying nothing. Every held-out-gene metric is then computed over fewer *effective* cells
than the fold contains, and neither the metric nor the seed files say so.

If that fraction is large on `cosmx` and small on `deep_starmap`, it is a candidate for **both**
open oddities in the replication at once:

* the held-out A2−A3 effect shrinking **5.6×** (+0.2514 → +0.0450) — a sparser field gives noisier
  per-gene Moran's I, which attenuates any correlation across genes;
* **A4's 0.2015 across-seed swing**, six to eighteen times every informative arm's, which is the
  spread that set the shared envelope and decided both verdicts. A4's held-out genes all share one
  embedding, so its `morans_pearson` is already a correlation over near-identical fields; on a
  sparse pool it is that correlation computed on a handful of cells.

### The thresholds, fixed now

The quantity to explain is a **difference between two datasets**, so the decisive comparison is
`cosmx` held-out against `deep_starmap` held-out. The kept pool is the within-dataset control.

**SUPPORTS the guess** — `cosmx`'s held-out zero-total fraction is **≥ 0.20** *and* at least **3×**
`deep_starmap`'s held-out fraction.

**RULES IT OUT** — `cosmx`'s held-out fraction is **< 0.05**, *or* within **1.5×** of
`deep_starmap`'s. Either way the pool sparsity is not what differs between the two runs, and this
explanation is dead rather than weakened.

**INCONCLUSIVE** — anything between. A real difference that is too small to carry a 5.6× effect
shrinkage on its own, and it would then be reported as one contributing factor among unknown
others rather than as the cause.

### Reported beside it, as corroboration only and with no threshold

* the **per-gene detection rate** on each pool (fraction of cells where the gene is non-zero,
  median across the pool). A pool of near-undetected genes produces the same attenuation through a
  different door, and the two are worth telling apart.
* the **effective cell count** `n_eff` = cells with a non-zero pool total. If the guess is right,
  A4's across-seed swing should scale roughly as `1/sqrt(n_eff)` between the two datasets — but
  that is one comparison on two points and cannot test anything. It is written down so it is not
  later presented as a prediction that was confirmed.

⚠️ **Whatever this returns, no verdict moves.** Part 1 is PARTIAL and Part 2 DOES NOT REPLICATE
under criteria fixed before the fits, and an explanation for *why* an effect was undetectable is
not a licence to re-read it as detected. If the guess is supported, what it buys is a stated
reason to expect the replication to have been underpowered on this dataset — which belongs in the
paper's limitations, not in its results.

---

# REPLICATION — THE VERDICTS (2026-09-07)

Three seeds (2, 3, 4) x two fits x four arms x two folds x two gene pools on `cosmx_nsclc_3d`,
holdout `paper_2_4`, folds `section_3` / `section_5`, 191 held out of 960. Every gate the
pre-registration names is measured. Both verdicts are read against criteria fixed before any fit.

## Every gate, closed

| gate | threshold | measured | |
|---|---|---|---|
| **(a)** ceiling − floor | >= 0.50 | room **0.9563** | pass (staged gate) |
| **(b)** held-out / kept ceiling | >= 0.80 | **~1.00** | pass (staged gate) |
| **(c)** held-out shared envelope | <= 0.2514 | **0.2015** (a) / **0.2327** (b) | does not fire — **by 7.4 % under (b)** |
| **(d)** `self` = 1.0 | exact | 1.0, four rows | pass |
| **(d)** layouts identical across arms | exact | scorer's own assert | pass |
| **(d)** retrieval PCA zero on held-out | exact | `build_and_fit`'s assert | pass |
| **(d)** constant field bitwise row-identical | required | **true**, deviation exactly **0.0**, all four (fold, pool) cells | pass |
| **(e)** kept shared envelope | <= 0.1322 | **0.0646** (a) / **0.1103** (b) | does not fire |
| **§4.2d** | both constructions agree | every check, both pools | nothing withheld |

Envelope (a) is the fold-mean construction, (b) per-fold. (b) is 1.15x (a) on the held-out pool
and 1.71x on the kept one, and **no check changes side between them**, so neither verdict is a
property of the aggregation choice. That is the first time this project has been able to say so:
nothing computed (b) until `scripts/t09_zeroshot_envelopes.py` existed.

⚠️ **(c) is the thin one.** Under the fold-mean construction the design cleared its own
cannot-detect-it condition by 20 %; under the per-fold construction by **7.4 %**. The replication
is inside its stated power by a margin I would not defend as comfortable, and the write-up says so
rather than reporting "(c) did not fire" as though it were clear.

## Part 1 — **PARTIAL**

`morans_pearson`, held-out genes, primary contrast **A2 − A3**, against the shared envelope
**0.2015**.

| criterion | measured | |
|---|---|---|
| 1. A2 − A3 > 0, signs at every seed **and** fold, balance >= 0.25 | +0.0573 / +0.0546 / +0.0231; **3/3 seeds, 6/6 folds**; balances 0.61 / 0.81 / 0.39 | ✓ |
| 2. \|A2 − A3\| exceeds the envelope | 0.0450 = **0.22x** (a), 0.19x (b) | ✗ |
| 3. A2 clears the `shuffled` floor by more than the envelope | +0.4189 over −0.0071 = **2.08x** (a), 1.80x (b) | ✓ |
| 4. void condition: A4 within one envelope of the floor | +0.0240 = **0.12x** | ✓ |

A2 − A4 meets 1–4 with A4 in A3's place: **+0.3949 = 1.96x** (a), 1.70x (b), 3/3 seeds, 6/6 folds,
balances 0.76 / 0.84 / 0.65. That is the pre-registered **PARTIAL** branch exactly:

> *text works, the route does not replicate. Something in the text channel reaches an unseen gene,
> but `W t` is not distinguishable from `gamma psi(t)` as the thing that does it.*

**`specs/10` §7's mechanism sentence is withdrawn** — "the pure-text path is the contribution and
the machinery around it is not" is a statement about A2 against A3, and A2 against A3 is
undetectable here. **The A2 − A4 claim is kept**: text does reach a gene the model never saw.

**The headline transfers; the mechanism does not.** `deep_starmap`'s A2 cleared its floor at
**2.52x**; here at **2.08x**. That is the 2.52x observation replicating on a second dataset, and
it is the one thing in the zero-shot experiment that now has two independent positives. Against
it, A2 − A3 went **+0.2514 → +0.0450, 5.6x smaller**, and the kept half **−0.1322 → −0.0327,
4.0x smaller**.

## Part 2 — **DOES NOT REPLICATE**, and the verdict's stated reading is wrong

| pool | A2 − A3 | signs | balances | vs its own envelope |
|---|---|---|---|---|
| held-out | **+0.0450** | 3/3 seeds, 6/6 folds | 0.61 / 0.81 / 0.39 | 0.22x (a), 0.19x (b) |
| kept | **−0.0327** | 3/3 seeds, 6/6 folds | 0.77 / 0.56 / 0.54 | 0.51x (a), 0.30x (b) |

Criterion 3 — *both magnitudes exceed the shared envelope computed on their own gene pool* —
fails on **both** halves, so it is not ONE-SIDED either. **DOES NOT REPLICATE**, as written.

⚠️ **The reading I attached to that verdict does not describe this data, and I am correcting it
rather than the verdict.** I pre-registered DOES NOT REPLICATE to mean *"the reversal was a
property of `deep_starmap` — mouse cortex, laminar — and the open-vocabulary claim has no
within-run evidence."*

**The reversal happened.** Sign-negative on kept and sign-positive on held-out, on **every seed
and every fold on both pools**, 12 of 12 cells, with fold balances never below 0.39. What failed
is **detectability**: neither magnitude clears an envelope set by a degenerate quantity. I wrote
the reading assuming a magnitude failure would arrive with a sign failure, and it did not. The
correct reading of this outcome is: **the direction replicated and the effect size did not, so
the flip is not established as a measurement even though nothing about it pointed the wrong way.**

That is a weaker claim than SUPPORT and a *different* claim from "it was a `deep_starmap`
artifact". The verdict stands; the sentence under it is replaced by this one.

## §4.2g — not every arm's spread is an envelope

**On both pools the threshold was set by a quantity that carries no information**, and on neither
did any informative arm's variance enter it:

| pool | shared envelope | set by | the four arms' own spreads |
|---|---|---|---|
| held-out | **0.2015** | **A4** (−0.0820 / +0.1195 / +0.0133) | A1 0.0318, A2 0.0111, A3 0.0234 |
| kept | **0.0646** | **`shuffled`** (+0.0528 / −0.0118 / +0.0337) | 0.0108 – 0.0134 |

A4 gives every held-out gene the **same** embedding, so its `morans_pearson` is a correlation
across genes whose generated fields are draws from one distribution — noise about zero. Its
*level* is exactly what certifies the void condition (0.12x from the floor, criterion 4's pass);
its *spread* is not a measurement scale for anything. Both facts are the same fact, stated twice:
A4 is noise centred near the floor.

§4.2b's rule — *"the largest across-seed envelope among everything the comparison contains ... so
none can be bought with a low variance"* — was written against an arm buying a verdict by being
**steady**. This is the mirror failure it did not anticipate: a **degenerate** arm inflating the
threshold **6–18x** above every real arm's, so that an effect present on 12 of 12 cells cannot
clear it. It is §4.2c's lesson one level up — *not every referent is a floor* becomes *not every
arm's spread is an envelope*.

⚠️ **This is an open methods question, and it is NOT grounds to re-read either verdict.** The
criteria were fixed before the fits and the envelope construction was named in them. Recording a
defect in a rule I wrote is not a licence to apply a different rule to the run that exposed it —
that is the fifth-instance failure §4.2 already describes, arriving from the other direction. It
belongs in the paper's methods as the next instance of the series and in the next
pre-registration as a construction to settle in advance.

## Cross-metric — an observation, and only that

**Not pre-registered.** The criteria are on `morans_pearson`; these five columns were computed
from the same seed files afterwards and no threshold applies to them.

| metric | held-out A2 − A3 | kept A2 − A3 | sign reverses |
|---|---|---|---|
| `morans_pearson` | +0.0450 | −0.0327 | **yes** |
| `gearys_pearson` | +0.0451 | −0.0396 | **yes** |
| `marker_field_r` | +0.0473 | −0.0092 | **yes** |
| `marker_depth_r` | +0.0855 | +0.0104 | no — positive on both, 8x smaller on kept |
| `umap_mixing` | −0.0052 | −0.0076 | no — negative on both; **no usable floor** (`_mixing` reads no coordinates) |

**Three of five reverse**, and the two that do not fail differently: `marker_depth_r` keeps its
sign while collapsing 8x, and `umap_mixing` is the metric §4.2c already excludes from referent
analysis entirely. The direction of Part 2's finding is therefore broader than the one metric it
was stated on — which makes the magnitude failure more interesting, not less, since a coincidence
would have to be a coincidence across four metrics at once.

It changes no verdict and cannot: a metric that was not named in advance cannot rescue one that
was. It is recorded because a reader will ask whether the reversal was a single-metric fluke, and
the answer is on this page rather than in a re-analysis someone else has to run.

## What the degeneracy check actually contributed — narrower than I implied

I asked for uninformative (d)'s constant-field clause as though it were open. **The staged gate
had already measured it on `cosmx`** on 2026-09-01 — `input_rows_identical: true`,
`input_std_max: 0.0`, all four rows — and the ceiling instrument's `input_information` calls
**the same `train.select._normalised`** on the same pools. I did not recall that when I raised it.

What `scripts/t09_zeroshot_degeneracy.py` adds is genuinely narrow, and it is worth having:

* **The ceiling path was established; the scorer's was not.** They are separately constructed —
  the ceiling broadcasts the *target section's* own mean over that section's cells, the scorer
  broadcasts the *whole training volume's* mean over the *generated* section's. The referent the
  arms were actually scored against had not been checked, and it is now: bitwise row-identical,
  deviation exactly **0.0**, on 55–58 k cells, all four (fold, pool) cells.
* **The CV comparison against real counts is new on this dataset**: constant field **8.8e-08 to
  1.03e-07** against real counts **5.42 to 6.07** — a **5.3e7x** separation, against
  `deep_starmap`'s 6.2e7x. Same order, so §4.2c's instrument gave the same answer on a second
  dataset by measurement rather than by inheritance.

⚠️ **And my own route to the conclusion was invalid regardless.** In the first review I wrote that
the constant field's across-seed spread of exactly 0.0000 showed bitwise-identical input. It shows
nothing of the kind: the constant field is a deterministic function of the training volume, so its
score is identical across seeds whatever its rows contain. A reproducible referent and a
degenerate one are indistinguishable in that statistic. The conclusion happened to be right and
the reasoning could not have established it.

Two readings from the same numbers, neither a finding: `cosmx`'s real-counts CV (5.4–6.1) sits
about **3x below** `deep_starmap`'s 16.1, which is what a denser 960-plex panel should look like;
and real counts hit a maximum row deviation of exactly **1.0** on both held-out folds, meaning
some cell puts all of its held-out mass on one gene while another puts none there — which is what
a 191-gene subset of a 960-plex panel should look like, and confirms the size factor is
pool-restricted as `_normalised` intends.

---

# T09 CLOSE-OUT, with the replication resolved (2026-09-07)

Supersedes the 2026-09-01 close-out. That one had five open items and named the replication as
"the only live decision left". It has been run, and item 4 (A7) has been run too. This states what
is left and where the project stands.

## What changed since 2026-09-01

| item | then | now |
|---|---|---|
| the `morans_pearson` replication | "the only live decision left" | **run and read** — Part 1 **PARTIAL**, Part 2 **DOES NOT REPLICATE** |
| A7 — SEFL's net contribution | "a hole in the method's own identity" | **answered, negatively**: SEFL is *harmful* on real tissue, 3 seeds, and ships at zero because it should |
| §7's mechanism sentence | the paper's one positive mechanism claim | **withdrawn** — it is A2 vs A3, and A2 vs A3 is undetectable on a second dataset |
| "text reaches an unseen gene" | one dataset, one metric, unreplicated | **two datasets**: A2 clears the floor at 2.52x and 2.08x; A2 − A4 at 1.96x |
| the evaluation methodology | four rules (§4.2a–d) | **five** — §4.2g, found by the replication, plus §4.2e/f from A7 |

## The honest headline, updated

> A continuous-field formulation reconstructs oblique planes at **95 %** of axis-aligned quality.
> On real tissue every generative component built on it — the intensity-field layout, the
> flow-matching expression head, the text-grounded gene embedding — **loses to copying a real
> section**, and the sectioning-equivariant losses the method is named for make it **worse** when
> switched on. One capability claim survives replication on a second dataset: **text places genes
> the model never saw**, at 2.52x and 2.08x over a measured floor. Which *path* in the text channel
> does it is **not** established — the pre-registered contrast between the pure-text projection and
> the distillation head reproduced in direction on 12 of 12 cells and at a fifth of the effect
> size, inside the noise. The expression failure is localised to the count draw and within it to
> the decoder's dispersion: `theta` carries 57–63 % of the conditional variance and is
> uncorrelated with the data's own (Spearman 0.068 over 1017 genes), so it absorbs unpredicted mean
> variation rather than estimating noise — a property of the ZINB objective, not a tuning error.
> The evaluation methodology developed to establish all of this — per-arm envelopes, referent- and
> arm-validity tests, aggregation-level rules, a five-channel leak taxonomy, and pre-registration
> with named uninformative conditions — is the transferable contribution.

## Where the project stands

**Unchanged in kind, stronger in evidence: a negative-results paper with a methods contribution,
and the methods contribution is the stronger half.** What the replication changed is the *shape*
of the positive column, and it is worth being precise because it is easy to report either too
well or too badly.

**The positive got narrower and much harder to dismiss.** Before, the paper had one unreplicated
positive on a post-hoc metric — worth approximately nothing to a referee, correctly. Now it has a
pre-registered replication on a second dataset in which the claim *text reaches an unseen gene*
cleared its floor twice, and the *mechanism* claim did not. That is a smaller claim than the
architecture was designed to make and it is the first thing in this project that has survived
being tested twice under criteria written in advance.

**The negative got a second confirmed instance.** A7 answered the identity hole: the three SEFL
losses do not merely fail to help, they collapse the anatomical field. The method is named for a
mechanism that is measurably harmful on real tissue, and the paper says so with three seeds.

**The methods contribution grew by the two ways this campaign failed itself**, and both are more
transferable than the results they were found in: §4.2f (an alarm that fires 218 times and reaches
no report is worse than one that was never built) and §4.2g (an envelope taken over a degenerate
arm sets a bar no real effect can clear). Neither is in this literature.

⚠️ **The one thing I would not oversell.** The replication cleared uninformative condition (c) by
**7.4 %** under the per-fold envelope construction. It is inside its stated power, and barely. A
referee entitled to ask "was this design able to detect the effect you were replicating?" gets the
answer "yes, by a margin of 7 %", and the paper should print that rather than "condition (c) did
not fire".

## What remains open

1. **R4 / the ZINB trade.** Unchanged and still the largest: a measured instance, a named term
   (`theta` absorbing mean variation), and a design change — objective or emission model — as the
   only route. A follow-up paper, not a follow-up measurement.
2. **R12's two questions.** The 46–119x per-gene spread in `s`, and why `theta` is uncorrelated
   with the data's dispersion. Both owed to T06.
3. **§4.2g's construction question.** How an envelope should be taken when some members of a
   comparison are degenerate by design. Recorded, deliberately *not* applied to the run that
   exposed it, and it must be settled **before** the next pre-registration rather than after the
   next verdict.
4. **The zero-total guess.** Pre-registered above with its thresholds fixed; the instrument exists
   and has not been run. Model-free, one minute per dataset, and it is the only live candidate for
   why the held-out effect shrank 5.6x and why A4 swung 0.2015. **It cannot move a verdict** — if
   supported it is a limitation, not a re-reading.
5. **R14's donor rule.** Unchanged: costs 0.116 of `marker_depth_r`, deliberately not fixed
   because fixing it makes the negatives stronger.

**Removed from the list:** the replication (run), A7 (run), the `decoder_mu_link` refit (not owed),
and the moment-matched `theta` (stopped after one fit — no data-derived value exists to match to).

## What would change this assessment

Only one thing, and it is not a measurement this project can run: a change to the emission model
or its objective that closes R4, followed by a re-run of the six-metric comparison. Everything
short of that produces another number in the negative column. The zero-shot claim is now as strong
as three seeds on two datasets can make it, and the mechanism question underneath it needs a
design change rather than more seeds.

## The zero-total guess — **RULED OUT**, and it points the other way (2026-09-07)

Measured on both datasets with `scripts/t09_zeroshot_pool_sparsity.py`, against thresholds fixed
before the instrument was written.

| | `cosmx_nsclc_3d` | `deep_starmap` |
|---|---|---|
| held-out zero-total cells | **329 / 225 981 = 0.00146** | **417 / 115 830 = 0.00360** |
| kept zero-total cells | **0** | **0** |
| per-gene detection rate (median) | **0.0933** | **0.0164** |
| median pool total, held out | 37–56 counts | 12–23 counts |

**RULES IT OUT fires on the first clause: 0.00146 < 0.05.** Not marginally — the fraction is
**34x below** the threshold, and the pool-restricted normalisation drops **0.15 %** of cells on
`cosmx`'s held-out side and **none at all** on either kept pool. `n_eff` is `n_cells` to three
decimal places everywhere. The mechanism I proposed does not exist at a scale that could affect
anything.

**And the direction is backwards, which is worth more than the threshold.** `cosmx` is **2.5x
LESS** sparse than `deep_starmap` by zero-total fraction and **5.7x DENSER** per gene by detection
rate — 9.3 % of cells detect the median gene against 1.6 %. The dataset where the held-out effect
was **5.6x smaller** is the one with substantially **more** data per cell. If field sparsity
mattered at all it predicts the opposite of what was observed, so this is not a weakened
explanation, it is an eliminated one — and the second candidate mechanism, per-gene sparsity, is
eliminated the same way and harder: `deep_starmap`'s least-detected held-out gene appears in
**0.04 %** of cells, and that is the dataset with the *larger* effect.

⚠️ **A hole in my own threshold set, recorded because it nearly mattered.** The RULES OUT clause
"within 1.5x of `deep_starmap`'s" is a **two-sided band**, `[0.67, 1.50]`. The measured ratio is
**0.404** — outside it, on the low side — so that clause returned *false* on a result that refutes
the guess more strongly than a ratio of 1.0 would. Had the fraction landed above 0.05 (say `cosmx`
0.30 against `deep` 0.90), **no branch would have fired**: not SUPPORTS (ratio 0.33 < 3), not RULES
OUT, and the run would have been called INCONCLUSIVE when it should have been ruled out. Clause 1
saved it here. The defect is that I wrote a symmetric band for an asymmetric hypothesis: the guess
predicts `cosmx` **sparser**, so the refuting region is everything below, not a neighbourhood of
parity. Next pre-registration of this shape states the one-sided form.

### Two things this bought that the guess did not

**1. A4's swing is not anomalous — it is exactly the sampling distribution of a null correlation,
and that makes §4.2g worse.** Under the null, a Pearson correlation over `n` genes has
`sd ≈ 1/sqrt(n-1)`, and the **range of three draws** has expectation `1.693 sd` and its own
`sd = 0.888 sd`:

| quantity | n | sd(r) | E[range of 3] | observed | z |
|---|---|---|---|---|---|
| `cosmx` held-out **A4** | 191 | 0.0725 | 0.1228 | **0.2015** | +1.22 |
| `deep_starmap` held-out shared envelope | 204 | 0.0702 | 0.1188 | **0.0930** | −0.41 |
| `cosmx` kept **`shuffled`** | 769 | 0.0361 | 0.0611 | **0.0646** | +0.11 |

All three are unremarkable draws from the same null. So the two datasets' envelopes — 0.2015 and
0.0930, a 2.2x difference that reads like a fact about the datasets — are **noise in the noise
estimate**, and the kept-pool envelope lands within 6 % of its theoretical expectation.

**§4.2g is therefore stronger than recorded.** It is not only that a degenerate member sets the
envelope; it is that a degenerate member's spread is a draw from a wide distribution that **three
seeds cannot estimate**. The threshold every criterion is read against is itself a random variable
with a coefficient of variation over 50 %. That belongs in the spec beside the rule.

**2. The verdicts are robust to it, and that can now be shown rather than hoped.** The decisive
criterion fails under every plausible envelope, not just the one measured:

| | vs measured | vs the other dataset's | vs the theoretical E[range] |
|---|---|---|---|
| held-out A2 − A3 = 0.0450 | 0.22x (0.2015) | 0.48x (0.0930) | 0.37x (0.1228) |
| kept A2 − A3 = 0.0327 | 0.51x (0.0646) | — | 0.54x (0.0611) |

Inside, every time. **Part 1 stays PARTIAL and Part 2 stays DOES NOT REPLICATE** whichever
envelope is used, so §4.2g's defect did not decide either verdict — it only made the margin look
larger than it is. That is the good case: a rule found to be wrong which turns out not to have
changed the answer it produced.

### What is left unexplained, stated plainly

**Nothing in this measurement accounts for the 5.6x shrinkage, and two candidates are dead.** What
remains is what the instrument cannot see: tissue organisation (mouse cortex is laminar, NSCLC
tumour is not), panel composition, and the text channel's content.

⚠️ **A nuance about my corrected reading, and I want it stated carefully rather than used.** The
pre-registered reading of DOES NOT REPLICATE was *"a property of `deep_starmap` — mouse cortex,
laminar"*. I corrected it because the **signs** replicated on 12 of 12 cells, and that correction
stands. But the laminar hypothesis was about *magnitude*, and on magnitude it is still live and now
has less competition. The honest statement is: **the direction is not a `deep_starmap` property;
the effect size may well be.** That is a different sentence from the one I withdrew, it is not
evidence for anything yet, and it does not touch either verdict.

## The density difference — an open candidate, and deliberately NOT pursued (2026-09-07)

The sparsity measurement ruled out its own hypothesis and produced a different one on the way out:
`cosmx` detects the median gene in **9.3 %** of cells against `deep_starmap`'s **1.6 %** — **5.7x
denser** — and its held-out A2 − A3 effect is **5.6x smaller**. Two numbers, opposite directions,
one dataset pair.

**It is a live candidate and it is being left alone.** This would be the fourth mechanism hunt of
this campaign (the length-scale calibration, the step budget, `decoder_mu_link`, the pool
sparsity), and each has cost more than it returned — three of the four ended in "not this either",
and the one that found something, `decoder_mu_link`, arrived after the fits it would have changed.
The base rate is what governs here, not how interesting the number looks.

**What would test it, so the next person does not have to re-derive it.** The mechanism would be
that a *denser* field makes per-gene Moran's I easier to estimate for **every** arm, compressing
the spread between arms — an attenuation of contrasts rather than of any one score. Two
predictions follow, and both are checkable without new fits:

1. **On the same dataset, split the held-out pool by detection rate** and score the arms
   separately on the sparse and dense halves. If density compresses contrasts, A2 − A3 is larger
   on the **sparse** half of `cosmx`'s pool than on the dense half. One scoring pass, no fitting.
2. **The arm scores themselves should not move much** — an attenuated *contrast* with unchanged
   *levels* is the signature; if the levels move together with the contrast, it is a difference in
   difficulty rather than in resolution.

**And a third dataset would settle it properly**, which is why this is a candidate and not a
finding: two points cannot separate "density" from every other way `cosmx` and `deep_starmap`
differ — tissue organisation, panel composition, tumour versus cortex.

⚠️ **Two datasets, one comparison, and no test.** Nothing above is evidence. It is written down so
that it is a recorded open question rather than an observation someone rediscovers and treats as
new.

---

# T09 CLOSE-OUT — the campaign is finished (2026-09-07, supersedes the entry above)

Written after the replication, A7, and the sparsity test. Nothing is pending measurement; the open
items below are design questions, not experiments this project can run.

## What I actually think the paper says

The 2026-09-01 close-out said "a negative-results paper with a methods contribution, and the
methods contribution is the stronger half". I think that is true and too vague to be useful, and
that the results now support a sharper claim which is also a more uncomfortable one.

**The generative hypothesis failed at every level that was tested, and it failed to the same
trivial baseline each time.** Not "some components underperformed" — *copying*:

* the learned intensity-field layout loses to **copying the coordinates** (R11: `field` 0.6607,
  `hybrid` 0.6692 against `resample` 0.7546), so `resample` ships;
* the flow-matching expression head loses to **copying the counts** — on `deep_starmap`
  `cross-mix` wins **every live metric**, and the one metric where generation had won was an
  artifact of the frame defect and reversed when it was fixed;
* the sectioning-equivariant losses the method is **named for** make it worse on three seeds, and
  ship at zero weight;
* the metric-aware losses cost on every metric they are built from.

**So the shipped configuration is beaten by the previous version on real data.** v25 ships
`resample` + `zinb-flow`: real positions with generated expression. v20's fallback is `resample` +
`cross-mix`: the same real positions with the donor's counts copied. The difference between them
is exactly the contribution v25 makes to that pairing, and it is **negative on every live metric**.
That sentence has not been written down before and it should be in the paper, because a reader who
works it out for themselves will trust nothing else in it.

**Against that, one thing works and it is not the generator — it is the representation.** GATE 2
is a real, clean, unqualified pass: oblique planes reconstruct at **95.5 %** of axis-aligned
quality (edge-excluded 97.9 %) against a pre-registered 90 %. And the text channel places genes the
model never saw above a measured floor, on **two datasets** at **2.52x** and **2.08x**, under
criteria written before the second run. Both of those are statements about **encoding** — that the
continuous field represents the volume off-axis, and that MedCPT text carries enough to position an
unseen gene. Neither is a statement about generation.

**My honest headline, then, is not "the method fails".** It is:

> A continuous 3D field is a good **representation** of a tissue volume and a bad **generator**
> from it. Reconstruction off-axis reaches 95 % of on-axis quality, and text embeddings place genes
> the model never saw above a measured floor on two datasets — but every generative component built
> on that representation loses to copying a real section, including the sectioning-equivariant
> losses the method is named for. The failure is localised and mechanical rather than diffuse: the
> decoder reproduces the *pattern* of between-cell variation almost perfectly (`mu`'s Moran's I
> 0.861, above the tissue's own latent at 0.745) at a fraction of its *amplitude*, and the ZINB
> objective closes the gap with dispersion instead — `theta` carries 57–63 % of the conditional
> variance while correlating with the data's own dispersion at Spearman **0.068** over 1017 genes.
> That is a property of the emission model's objective, not a tuning error, and no data-derived
> value of `theta` can substitute for changing it.

The part I would defend hardest is the last sentence, because it is the difference between "our
expression head underperformed" and a finding. A likelihood that can be reduced by moving
explanatory power out of the structured mean and into the noise term will be, and this project
measured it happening four separate ways (R4 i–iv).

## The methods half — and why I think it is the stronger contribution

Nine rules now (§4.2a–i), and the reason they are worth a paper is not that they are individually
clever. It is the pattern: **every claim in this literature has the form "the margin exceeds the
noise", and that sentence hides six independent choices — in this project each one silently decided
a verdict before anyone noticed it was a choice.** Which arm's variance is the noise (§4.2a). Which
envelope a clearance takes (§4.2b). Which referent is a floor at all (§4.2c). How the spread is
aggregated (§4.2d). Which arms may contribute a spread (§4.2g). Which side of the threshold refutes
(§4.2h). Plus two failures of a different kind — a metric that is a correlation on a dead field
(§4.2e) and an alarm that fired 218 times where no report read it (§4.2f) — and one standing
reporting requirement that falls out of all of them (§4.2i: a three-seed envelope is an estimate
with a CV above 50 %, and none of ours was ever reported as one).

I think this is the stronger half because it is the part that transfers. The negative result is
about one method on two datasets; the methodology is about how anyone reads a repeated-seed
benchmark, and none of it is reported in this literature.

⚠️ **The honest discount on that claim.** Every one of these was found *because a criterion in this
project nearly returned the wrong answer*, and in four cases it had already returned one that was
published internally and later withdrawn. That is the right provenance for a methods paper and it
should be stated as such rather than presented as foresight. The paper should say: we found these
by getting them wrong first.

## What remains open

1. **R4 / the ZINB trade.** Unchanged and still the largest. A measured mechanism, a named term,
   and a design change — objective or emission model — as the only route. A follow-up paper.
2. **R12's two questions.** The 46–119x per-gene spread in `s`, and why `theta` is uncorrelated
   with the data's dispersion. Both owed to T06.
3. **§4.2g's construction question.** How to take an envelope when some members are degenerate by
   design, and how to report one that three seeds cannot pin (§4.2i). Must be settled **before**
   the next pre-registration, not after the next verdict.
4. **The `cosmx` effect-size shrinkage.** The direction replicated on 12 of 12 cells and the
   magnitude fell 5.6x, and nothing explains it. Two candidates eliminated (pool sparsity, per-gene
   sparsity); the density difference is recorded above as a candidate with a stated test and is
   **deliberately not being chased**. The magnitude half of the withdrawn "property of
   `deep_starmap`" reading is also still live and also unused.
5. **R14's donor rule.** Costs 0.116 of `marker_depth_r`, deliberately unfixed because fixing it
   makes the negatives stronger.

**Closed since 2026-09-01:** the replication (run — PARTIAL / DOES NOT REPLICATE), A7 (run — SEFL
harmful), the `decoder_mu_link` refit (not owed), the moment-matched `theta` (no data-derived value
exists to match to), and the pool-sparsity guess (refuted, direction backwards).

## What would change this assessment

One thing, and this project cannot run it: a change to the emission model or its objective that
closes R4, followed by a re-run of the six-metric comparison against copying. If generated
expression beat `cross-mix` on real tissue after that change, the headline inverts from "a good
representation and a bad generator" to a method paper. Everything short of it produces another row
in the negative column.

**And one thing that would not.** More seeds. The zero-shot claim is as strong as three seeds on
two datasets can make it, and §4.2i is the reason to say so plainly: the envelopes those seeds
produce are estimates with a coefficient of variation above 50 %, so a fourth and fifth seed buy
precision in the noise estimate, not evidence about the method.

---

# CLOSE-OUT AMENDMENTS (2026-09-07)

Two corrections to the close-out above, at the user's direction. The second one did not survive
checking in the form it was raised, and the corrected version is stronger.

## Amendment 1 — intersection consistency is the SECOND validated claim, not residue

The close-out listed exactly one thing that works — the representation — and put mutual
consistency between crossing sections nowhere. That is a misplacement, and the property is the
cleanest positive in the project.

**Two crossing sections emit bitwise identical expression along their intersection**, on an
**untrained** model, with no consistency loss applied
(`tests/test_sefl.py::test_generation_is_intersection_consistent_by_construction`). Not
approximately, not as a consequence of optimisation, and not at a particular checkpoint: the 3D
noise field is continuous (T03) so both branches draw the identical realisation along the
intersection, and every conditioning pathway — retrieval, the GRF, the Fourier encoding — is
queried at **physical** points while `CTFFlow.generate` conditions with the identity pose whatever
plane it is handed. The property therefore holds for **every model of this architecture, at every
checkpoint, exactly**.

**Three things make this a result rather than an implementation note.**

1. **It is a stronger claim than the loss written to earn it.** `L_cross` exists in `specs/07` to
   *train* approximate agreement between crossing planes. The architecture already has exact
   agreement, so the loss is **vacuous in v25** — and worse than vacuous: the only plane-dependent
   channel left for it to act on is the augmentation **pose**, which T04 made pose-dependent on
   purpose, so minimising it destroys the anatomical field. At `specs/07`'s `w_cross = 0.3` the
   generated section's per-gene variance falls to **0.065** of the real one's, against **0.711**
   with SEFL off (R6). `w_cross` ships at **0**.
2. **So it survived the strongest possible test of a claimed property**: the mechanism written to
   enforce it was shown to be both unnecessary *and* harmful, and the property was unaffected —
   because it never depended on that mechanism.
3. **No competing method has it.** A method that generates sections independently per plane has no
   reason for two crossing sections to agree anywhere, and none of the baselines in `reference/`
   makes the guarantee. It is a property of committing to a **continuous 3D field** as the
   representation, which is exactly the thesis the generative half failed to support.

**Where it belongs.** Beside GATE 1's correlated prior and GATE 2's oblique reconstruction, as the
third statement about the **representation** — and the second that is exact rather than measured.
The headline should read: the continuous field is a good representation, and here is what
"good" means concretely — off-axis reconstruction at 95.5 %, controllable per-gene spatial
autocorrelation, and mutual consistency that is exact by construction rather than trained for.

## Amendment 2 — the budget question, and the premise it was raised on does not hold

Raised as: *every negative was measured at `train_steps` 1200 or on a defaults-provenance config,
while T09's gate selected 2400 and T08 showed the metric-aware terms reverse sign between those
budgets.* **The concern is real and the scope is much narrower than that.** Checked against the
record, one negative of the four was measured at 1200:

| negative | budget it was measured at |
|---|---|
| **Layout (R11)** | **2400** for the headline `layout_mode` swap; the count error confirmed at **both** (4332/1372/3866 at 1200, 11168/146/3788 at 2400, against ~4160) |
| **SEFL (A7)** | 🚨 **1200 only** — `Config.train_steps`'s default, not the selected 2400 |
| **Metric-aware (T08/T09)** | **both**, and it is the one case where the budget flips the sign |
| **Zero-shot (E1 + replication)** | **2400** — `t09_zeroshot_run.py --train-steps` defaults to 2400, and all six fits of both campaigns ran there |

**What this licenses, precisely.**

* **A7's verdict carries the budget caveat and always did** — the record already states it as
  *"SEFL adds nothing at the default budget of 1200 steps", not "SEFL adds nothing to the shipped
  model"*, and the close-out should repeat it rather than leave it in a prior entry. It is the one
  headline negative whose budget does not match the shipped configuration, and the metric-aware
  result is the proof that the objection is not hypothetical: **the same three-weight comparison
  reverses between 1200 and 2400.** A reviewer asking "would SEFL have helped at 2400?" has a
  measured precedent for the question and the honest answer is that nobody knows.
* **The claims table was stale and is corrected.** "Metric-aware losses: NEGATIVE at the shipped
  budget" was written when the shipped budget was 1200. At 2400 the weights **win** the selection
  (rank 1.0 against 2.0, four of six metrics) and **ship at 0.5** — but the per-metric margins are
  **0.0052 / 0.0101 / 0.0018**, all far inside R10's 0.0335 envelope, on one seed. The claim is now
  **UNRESOLVED at the shipped budget**: won on aggregate rank, established on nothing.

**What it does NOT license, and this is the part to state plainly.**

* **It does not rescue the layout or the zero-shot negatives**, which were measured at the selected
  budget. Those two are the bulk of the negative column and the budget objection does not touch
  them.
* **It does not rescue A7 either — it only widens A7's stated scope.** A7's SEFL arm did not merely
  underperform; it **collapsed**, with `i_gen` at 1.6–2.4 % of target and `check_collapse` firing
  218 times from step 250 of 1200. A budget that ends *after* the collapse has already run for 950
  steps is not obviously a budget that would have escaped it, and the burden is on the claim that
  more steps recover a dead field, not on the claim that they do not.
* **It does not make the negatives provisional.** "Measured at a budget the project itself did not
  select" is a scope statement on one result, not a general discount. Writing it as a general
  discount would be the reverse of the §4.2 failures: instead of a threshold moved to help, a
  caveat applied broadly to soften an answer that was measured correctly.

**The cost of closing it**, stated because a reviewer will ask and because it bears on what is left
to spend: A7 at 2400 is six fits at roughly double the per-fit cost of the 1200-step arms. That is
the largest single item on any remaining list, and it buys a scope extension on a result that is
already negative on three seeds — not a new result.

## Amendment 3 — a component that ships ON on no evidence (2026-09-07)

The corrected metric-aware row is not a bookkeeping fix and should not sit in the claims table
where a reader scans past it. **`w_autocorr = w_profile = w_distribution = 0.5` ship ON, and
nothing establishes them.**

* At **1200** steps they lose — rank 3.5 against 3.0, a cost on every metric.
* At **2400**, the budget T09 selected, they win the selection on aggregate rank, 1.0 against 2.0,
  taking four of six metrics. That is why they ship at 0.5.
* The per-metric margins at 2400 are **0.0052 / 0.0101 / 0.0018** on the autocorrelation metrics,
  against R10's **0.0335** envelope. Every one is inside it, by factors of 3 to 19. On **one seed**.

So the selection is sound as a *selection* — an aggregate rank over six metrics is a legitimate way
to pick a cell when the individual margins are small — and it is **not evidence that the losses do
what they are named for**. "Metric-aware losses improve the metrics they are made of" is
**UNRESOLVED**, and `claim_min_seeds = 3` says one seed cannot resolve it in either direction.

**Why this is a different kind of problem from the rest of the negative column, and belongs in the
paper as such.** Every other component here is in one of two defensible states:

| state | components | what the paper can say |
|---|---|---|
| **ships OFF, with evidence against** | SEFL's three weights (A7: harmful, 3 seeds), `w_cross` (vacuous *and* destructive, R6), the intensity-field layout (R11: below the copy floor by 3.2-3.5x the envelope) | "we tried it, it does not work, here is the measurement, it is disabled" |
| **ships ON, with evidence for** | `prior_mode="correlated"` (GATE 1), `layout_mode="resample"` (beats both field modes), `decoder_mu_link="exp"` (structured share 15.1 % -> 61.4 %) | "we tried it, it works, here is the measurement" |
| 🚨 **ships ON, established by nothing** | the three **metric-aware** weights at 0.5 | — |

A component that ships **off** with evidence against it is an honest negative result: the reader
learns something and the shipped model does not depend on it. A component that ships **on** while
established by nothing is a different failure — **every number this project reports on the shipped
configuration was produced with those three losses active**, so an unestablished component is
inside the baseline that all of the negatives are measured against. It is not a claim the paper
makes; it is a claim the paper's *setup* makes silently.

**What honesty requires here**, and it is cheap: say it. The paper should state that the
metric-aware weights ship at 0.5 because they won a one-seed aggregate-rank selection at 2400
steps, that the per-metric margins were inside the reproducibility envelope, and that their
contribution is therefore unresolved rather than demonstrated. A reader can then discount the
shipped configuration accordingly, which is the point.

**What would resolve it**, so the option is costed rather than gestured at: the same
`(2400, on)` vs `(2400, off)` comparison at **three seeds** — two fits per seed, six fits, the same
order as A7. It is the cheapest unresolved item on the list and the only one that touches the
configuration every other number was measured on. I am **not** recommending it over the R4 work;
I am recording that it is the one place where a small spend would convert a silent assumption into
a stated result.

⚠️ **And it does not weaken the negatives.** If the metric-aware losses turn out to do nothing,
every negative was measured on a model carrying three inert terms, which changes no comparison —
each of R11, A7 and the zero-shot arms is a *within-configuration* contrast with the same weights
on both sides. The exposure is to the **absolute** numbers and to the claim that the shipped model
is the best configuration found, not to any of the differences.

## Close-out amendments 4–6 (2026-09-07)

**4. §2.4's header now carries its own qualification.** It read "Text places genes the model never
saw — replicated" while §3 records the same experiment as PARTIAL and DOES NOT REPLICATE. A header
that will be quoted alone must not need the body to be honest. It now reads **"Text beats a
gene-blind arm on unseen genes — replicated. *Which* text path does it — not."** What replicated is
A2 over the `shuffled` floor (2.52x, 2.08x) and A2 − A4 (1.96x); what did not is A2 − A3, the
contrast that says *how*.

**5. §9 was self-contradicting on its face and the two statements are now separated.** "More seeds
would not change the assessment" and "six fits at three seeds is the only small spend that touches
something real" are about different objects:

* *More seeds on a comparison already run* buys precision in the **noise estimate**, which §4.2i
  shows is a CV-above-50 % quantity. It cannot change a verdict on the zero-shot claim.
* *Three seeds on the metric-aware comparison* is not a rerun at higher precision — the existing
  evidence there is **one** seed, so three is the **first** measurement `claim_min_seeds` accepts.

Different objects, opposite conclusions, and the section now says so rather than leaving a reader to
reconcile them.

**6. `marker_field_r` is promoted to an open item (§8.6).** It had no entry anywhere in the
close-out and it has appeared four times: the pooled v20/v21 loss against SpatialZ, v25's single
losing metric on the fixture at T09, the smoke run's 0.1611, and tier-1's **0.6384 against a
`flanking_copy` floor of 0.8857 — 0.247 below it, 7.4x the envelope**, the worst of the six.

⚠️ **Stated at its real strength, which is lower than the count implies.** Appearances 1–2 are a
**cross-dataset pool**, which `specs/10` §4.2a forbids: per dataset the comparison is **9–9**, on
tier-1 v20 (0.8804) and v21 (0.8881) both beat SpatialZ (0.8522), and in the wide regime v20 wins
**7 of 7**. Only the pooled figure favours SpatialZ. The honest count is **two clean v25 appearances
plus a pooled comparison this project's own methodology rejects** — still a pattern across three
generations, and not the four-for-four it reads as.

**Why it is an open item and not a complaint**: the deficit is **arrangement, not magnitude**
(`gene_mean_spearman` is 0.0033 off its copy floor, inside the envelope), and it is one of only two
**pose-dependent** metrics with tier-1 rotations of 0.0–1.5 degrees — so alignment is measurably not
confounding it and a deficit at ~0 degrees points at **T05's intensity head**. **The first step
costs zero fits**: the alignment diagnostics are already reported beside it, and stratifying the
existing tier-1 scores by distance to the volume boundary (R3's regime, where the intensity integral
is least stable) is a scoring pass on fits that already exist.

## `marker_field_r` boundary stratification — PRE-REGISTERED, before the instrument (2026-09-07)

**Status: a localisation test, not a hypothesis test.** It cannot make `marker_field_r` better or
worse; it can only say **where** the deficit lives. Written before
`scripts/t10_rescore_saved.py` is touched.

### The correction this rests on, first

The advisor framing — *"its fourth appearance as this project's standing weakness (v20 and v21's
pooled loss vs SpatialZ, …)"* — is **wrong and is withdrawn** in `progress/numbers.md`,
`scripts/t10_rescore_saved.py`'s docstring, and the close-out. The v20/v21 figure is a
**cross-dataset pool, which `specs/10` §4.2a forbids by name**: per dataset the comparison is
**9-9**, on tier-1 v20 (**0.8804**) and v21 (**0.8881**) both **beat** SpatialZ (**0.8522**), and
in the wide regime v20 wins **7 of 7**. Only the pooled average favours SpatialZ. The honest count
is **two clean v25 appearances** — the single loss at T09 and 0.1611 in the smoke run — **plus a
pooled comparison our own methodology rejects.** A real pattern in v25, not a three-generation one.

### What is measured

Tier-1's three held-out sections split into **one boundary section and two interior**:
`section_2` sits next to `section_1`, the stack's first, so its flanking evidence is **one-sided**
— R3's regime. `section_4` and `section_6` are interior.

The quantity is **not** v25's raw per-section score. `reports/pilot.md` §3 already established that
the **model-free copy floor is itself worst at `section_2`** on 6 of 6 metrics
(`marker_field_r`: **0.8470** there against 0.8857 / 0.8873), so a raw v25 deficit at `section_2`
would be partly a property of the protocol rather than of the model. The quantity is v25's
**deficit below its own per-section `flanking_copy` floor**:

    deficit(s) = flanking_copy_marker_field_r(s) − v25_marker_field_r(s)

pooled median deficit **0.247** (0.8857 − 0.6384), **7.4x** R10's 0.0335 envelope.

### The three outcomes, fixed now

**LOCALISED TO THE BOUNDARY** — `deficit(section_2)` exceeds the mean of
`deficit(section_4)`, `deficit(section_6)` by **more than one envelope (0.0335)**. Reading: the
weakness concentrates where the intensity integral is least stable and the evidence is one-sided.
That is a **pointer at T05's intensity head**, and it makes the boundary-clamp geometry (C33's
`z ± thickness/2` fix and R3) the first place to look. It does **not** prove causation from one
seed and three sections.

**BOUNDARY ELIMINATED** — the three deficits agree **within one envelope**. Reading: the weakness
is uniform across the stack, so R3's boundary regime is **not** where it lives, and the intensity
head's boundary behaviour is **removed as a candidate**. That is the more useful outcome for
narrowing, and it points the remaining search at the interior mechanism — arrangement at every
depth, not at the edges.

**INVERTED** — `deficit(section_2)` is **lower** than the interior mean by more than an envelope.
Reading: reported as-is and not explained. It would mean the model does relatively *better* where
evidence is one-sided, which no candidate on the table predicts, and it would be a finding about
the metric rather than the model.

### Conditions that make it uninformative, named in advance

* **(a)** any section's `flanking_copy` or v25 score is missing or NaN — three points is already
  the minimum and two cannot support the comparison.
* **(b)** the re-scored pooled median `marker_field_r` differs from the recorded **0.6384** by more
  than one envelope. The saved weights, the generation seed and the evaluator would then not be the
  ones that produced the recorded number, and the stratification would describe a different run.
* **(c)** ground-truth-matched density is not achieved per section — the raw layout over-produces,
  and a denser point set inflates every graph-based metric, so an unmatched comparison across
  sections is confounded by exactly the quantity R11 found spanning 500x between these sections.

### What no outcome licenses

**None of the three changes any verdict in the close-out**, and none makes `marker_field_r` a
positive. `n = 3` sections, **one seed**, one dataset: this is a pointer for the next person, and
`claim_min_seeds = 3` applies to seeds, which this has one of. It is worth running because it is
**free** — a re-score of saved weights, no fit — and because "eliminated" is as useful as
"localised" when the candidate list is this short.

## `marker_field_r` boundary stratification — RESULT: the boundary is eliminated, on the only readable arm (2026-09-07)

Run with **zero fits and zero generation**: both sides were already on disk — `arms[*]`'s
`matched_per_section` and `referents.flanking_copy` in `reports/r11_starmap_layout_modes.json`.
`scripts/t10_marker_field_boundary.py` reads them and applies the pre-registered criteria.
The pre-registration named no arm, so **all five** are reported rather than one chosen.

| arm | deficit `section_2` | interior mean | gap vs 0.0335 | verdict | readable? |
|---|---|---|---|---|---|
| `resample-grid` | 0.1729 | 0.1960 | −0.0231, **0.69x** | **BOUNDARY ELIMINATED** | ✅ density spread **1.04x** |
| `hybrid-grid` | 0.1678 | 0.2265 | −0.0587, 1.75x | INVERTED | ❌ spread **71x** |
| `field-grid` | 0.2431 | 0.3612 | −0.1182, 3.53x | INVERTED | ❌ spread **71x** |
| `hybrid-rejection` | 0.1957 | 0.2929 | −0.0972, 2.90x | INVERTED | ❌ spread **1079x** |
| `field-rejection` | 0.2213 | 0.3064 | −0.0851, 2.54x | INVERTED | ❌ spread **1079x** |

### The four INVERTED verdicts are a density artifact, and condition (c) caught it

Pre-registered condition **(c)** — *"ground-truth-matched density is not achieved per section"* —
**fires on all five arms**, because every one emits fewer cells than the truth in at least one
section and subsampling cannot add. But the *magnitude* differs by three orders of magnitude, and
that is what decides readability. The field-based modes emit **63.9x** the truth at `section_2` and
**0.90x** at `section_6`: the boundary section is scored on a 64x-thinned draw — many candidates,
well-conditioned — and the interior on the raw emission. A better-conditioned point set scores
better, so `section_2` looks better, and the "inversion" is the sampler's count error read through
the metric. `hybrid` shares `field`'s count draw bit-for-bit (R11), so it inherits the same artifact
despite passing (b).

**`resample-grid` is the only arm where the sections are comparable** — ratios 0.97 / 1.02 / 0.99, a
spread of **1.04x** — and it is also the **shipped** `layout_mode`. It returns **BOUNDARY
ELIMINATED**: the deficits are 0.1729 / 0.1877 / 0.2043, and the boundary-vs-interior gap is
**0.69x** the envelope, comfortably inside it.

### The reading

**The boundary is not where `marker_field_r`'s weakness lives.** On the shipped configuration the
deficit below the copy floor is **uniform along the stack** — if anything marginally *smaller* at
the boundary — so R3's one-sided-evidence regime and the boundary-clamp geometry are **removed as
candidates**. That was the pre-registered "boundary eliminated" branch, and it is the more useful
of the three for narrowing: the remaining search is an **interior** mechanism, arrangement at every
depth, not an edge effect.

⚠️ **The pointer at T05's intensity head is weakened, not confirmed.** The close-out's §8.6 argued
that a deficit at ~0 degrees rotation points at the intensity head *via the boundary regime*. That
route is now closed. The intensity head is not exonerated — `resample` does not use it for
placement at all, and the deficit is still 0.19 there, which says the weakness survives **removing
the layout head from the picture entirely**. That is itself informative and points away from T05
and toward the expression path — R12's territory, not R11's.

### Two defects in my own instrument, both found by running it

1. **Condition (b) was written for one arm and applied to five.** `RECORDED_POOLED` was `hybrid`'s
   0.6384, compared against `field` and `resample` too, and duly "fired" on both — a category
   error, not evidence. It is now a `(mode, sampler)` map, and a **grid** arm has no prior value on
   its own sampler at all, so (b) is reported **NOT EVALUABLE** there, which is not the same as
   passing.
2. **Condition (c) was binary where it needed a magnitude.** As written it fires on all five and
   would have thrown the readable arm away with the unreadable ones. The 2.0x spread cut that
   separates them is **post-hoc** and is labelled as such everywhere it appears — the report prints
   the pre-registered binary result first and the magnitude judgement beside it, never instead of
   it. This is `specs/10` §4.2h's failure in a new place: a condition written without asking how the
   data would fall.

### What this does not do

It does not change a verdict, and `marker_field_r` is exactly as weak as it was: **0.247 below its
copy floor at 7.4x the envelope**, v25's worst metric, on two clean appearances. Three sections,
one seed, one dataset. It eliminates one candidate from a short list, for free.

---

# A9 — THE METRIC-AWARE WEIGHTS: PRE-REGISTRATION (2026-09-07, before any fit)

Written before `scripts/t09_ship_starmap.py` gains its switches and before a single fit runs. This
is the last spend that resolves anything.

## Why this and not something else

`w_autocorr = w_profile = w_distribution = **0.5**` **ship ON**, and nothing establishes them. At
1200 steps they lose (rank 3.5 against 3.0, a cost on every metric); at **2400** — the budget T09
selected — they win the selection on aggregate rank, 1.0 against 2.0, on four of six metrics. The
per-metric margins there are **0.0052 / 0.0101 / 0.0018** against R10's **0.0335** envelope: inside
it by factors of 3 to 19, on **one seed**.

That makes this the only component in the project that **ships on** while established by nothing,
and it sits **inside the baseline every other number was measured against**. It is not a claim the
paper makes; it is a claim the paper's setup makes silently.

## The design

**Arms.** `(2400, on)` = the shipped `0.5 / 0.5 / 0.5`, against `(2400, off)` = `0 / 0 / 0`. One
gate, two levels, nothing else moved. **Three seeds**, two fits each: **six fits.**

**Dataset and instrument.** Tier-1 STARmap, holdout `paper_2_4_6`, the pinned
`bench3.evaluate_paper`, ground-truth-matched density, `layout_mode=resample` (shipped).

**Envelope.** The shared envelope of `specs/10` §4.2b, computed **on this run** — the largest
across-seed spread among the arms on that metric — never R10's inherited 0.0335. §4.2i applies: the
null's sampling distribution is not analytically available for these metrics, so the envelope is
reported as the estimate it is, with the measured spread of **both** arms shown beside it.

**Both §4.2d constructions.** Fold-mean and per-fold, on every metric. If they disagree on any
metric, that metric is **not established** and is reported as such.

## The claim under test

> The metric-aware losses improve the metrics they are made of.

Three of the six metrics are the ones the losses are *built from* — `morans_pearson`,
`gearys_pearson` (autocorrelation), `marker_depth_r` (profile). Those are the **primary** three.
`umap_mixing` and `marker_field_r` are secondary: the distribution term is not built from them
directly. `celltype_localization` is **inert** under `resample` and is excluded, not scored as a
tie.

## POSITIVE — the losses do what they are named for. All of:

1. `(on) − (off) > 0` on **at least two of the three primary metrics**, with signs agreeing at
   **every seed and every fold**;
2. each of those margins **exceeds the shared envelope** under **both** §4.2d constructions;
3. no primary metric shows a **negative** margin that clears the envelope — an improvement on two
   bought by a real regression on the third is not "improves the metrics they are made of".

**Reading.** The weights ship at 0.5 on evidence, the selection is retrospectively justified, and
the shipped baseline is no longer carrying an unestablished component.

## NULL — and this is stated as explicitly as the positive, because it is the likelier outcome

**NULL fires when every primary margin sits inside the shared envelope under either construction**,
whatever the signs.

**The reading is not "the losses are harmless".** It is:

* **The selection that put them at 0.5 was a coin-flip on a rank**, made on margins the design
  cannot resolve, and the paper must say the shipped configuration includes three terms whose
  contribution is **zero to within measurement**.
* **Every absolute number in this project was produced with three inert terms active.** That
  changes no *contrast* — R11, A7 and the zero-shot arms all carry the same weights on both sides —
  but it means "the best configuration found" is a statement about a rank, not about the model.
* **The right disposition is to say so, not to turn them off.** Turning them off after the fact
  would re-open every fitted number in the project for a change that is by hypothesis within noise,
  and would be a threshold moved after seeing where the data fell. They ship as they are, with the
  null attached.
* **It is publishable.** "A three-term auxiliary objective, selected by aggregate rank, contributes
  nothing measurable at three seeds" is a real result about metric-aware training, and it is the
  same shape as A7's — a mechanism that was supposed to help and does not.

## NEGATIVE — the losses cost

At least one primary metric shows `(on) − (off) < 0` clearing the shared envelope under both
constructions, with signs agreeing at every seed and fold, and **no** primary metric clears it
positive. **Reading:** the same shape as A7 and a stronger version of it — the weights should be
set to 0, and every number in the project was measured on a configuration that was actively worse
than an available alternative. This outcome **would** require re-opening the absolute numbers, and
that cost is named here so it cannot be a surprise.

## MIXED — one primary clears positive and another clears negative

Reported as-is, with both, and the claim is **refuted as stated**: "improves the metrics they are
made of" is a claim about the set, and a set that moves in both directions does not satisfy it.

## UNINFORMATIVE — decided by these conditions and no others

* **(a)** the shared envelope on any primary metric **exceeds 0.0335**, R10's recorded figure by a
  factor of two or more. The design would then be noisier than the instrument that produced the
  number being tested, and neither outcome could be read. *Checkable only after the fits.*
* **(b)** the two §4.2d constructions disagree on **all three** primary metrics — a verdict that is
  entirely a property of the aggregation choice.
* **(c)** the `(2400, off)` arm's scores differ from the recorded `(2x, weights off)` row
  (`morans_pearson` **0.9316**, `gearys_pearson` **0.9022**, `umap_mixing` **0.9145**) by more than
  one envelope at seed 1. The fits would then not be reproducing the selection they are testing.
* **(d)** any fit fires `check_collapse` or `check_spatial_collapse`. A7's lesson: an alarm that
  fires and is not read invalidates every metric in the report, and the run report now prints it.

## What no outcome licenses

**None of the four changes any verdict in the close-out.** R11, A7 and the zero-shot results are
within-configuration contrasts with the same weights on both sides; their *differences* are
untouched whatever this returns. The exposure is to the **absolute** numbers and to the sentence
"the shipped model is the best configuration found" — nothing else.

**Three seeds is the minimum, not a comfortable margin.** `claim_min_seeds = 3`, and §4.2i's warning
stands: a three-seed envelope is an estimate. Two seeds agreeing would not have been reportable;
three is the floor at which it becomes so.

## Order of operations

1. **One fit, timed** — seed 1, `(2400, off)` — before the other five. It writes the timing record
   and prints the six-fit projection; every later fit **refuses to start** without it.
2. Report the projection and **stop**, for the same reason the last two campaigns did.
3. Five more fits, then score, aggregate under both constructions, and read the verdict.

## A9 — fit 1 timed, and the common-mode note the warnings require (2026-09-07)

**Fit 1: ~56 min, six fits projected at 5.6 core-hours, ~4.7 remaining.** The cheapest campaign in
the project by a wide margin — A7 cost roughly four times this and the replication twenty. The
timing gate has now been honoured on three consecutive campaigns.

### The three warnings, and the asymmetry they carry

Fit 1 raised three, all known: `expr_pca_dim` clamping 32 to 28 (`specs/10` §0's owed fix),
thickness defaulting to 22 µm under the assumed flag, and 81 coincident coordinates from
z-flattening.

**Both arms carry all three, so none of them can separate the arms.** That is right and it is the
important half — a common-mode nuisance cannot manufacture a difference between two arms that
share it, so **none of these can produce a false POSITIVE or a false NEGATIVE.**

⚠️ **But the protection is one-sided, and the pre-registration should have said so.** A common-mode
distortion cannot *create* a difference; it can *attenuate* one. Two of the three plausibly do:

* **Thickness at an assumed 22 µm** feeds the slab geometry that `soft_depth_profile` reads, and
  `w_profile` is built on that profile. If the assumed thickness is wrong, the profile term is
  mis-scaled in **both** arms — which leaves the contrast unbiased in sign and **smaller in
  magnitude** than it would be at the true thickness.
* **81 coincident coordinates** put zero-distance pairs into every kNN graph, which is what
  `morans_pearson` and `gearys_pearson` are computed on. Again identical in both arms, again a
  compression of whatever difference exists.

**So the caveat attaches to the NULL, not to the positive.** A POSITIVE under these conditions is
if anything understated. A NULL means *no detectable effect on this configuration, with three known
distortions present in both arms* — **not** "no effect". That is a materially weaker statement than
the one A9's null branch was written to support, and it is being recorded **before the results**
rather than discovered in them.

It does not change the pre-registration's branches, thresholds or readings. It changes one sentence
in what a null licenses, and `scripts/t10_a9_aggregate.py`'s NULL reading now carries it.

### The aggregator exists before the data does

`scripts/t10_a9_aggregate.py` is written and lint-clean **before any of the six reports landed**,
so the outcome cannot have shaped the instrument. It reads the arm from each report's persisted
**config**, not from its filename — a filename is not a measurement, and this project has already
paid once for a run whose name and contents disagreed. It refuses a report whose three weights
differ, computes both §4.2d constructions, marks any metric where they disagree as **not
established**, excludes `celltype_localization` rather than scoring it as a tie, and checks the
persisted alarm record for uninformative (d) instead of assuming silence.

## A9 fit 1 — clean, and it withdraws my own uninformative (c) (2026-09-07)

`runs/a9/s1_off`, config `defd0eec6a179c82`, **3342 s (55.7 min)**, matching the projection.

### What checks out

* **Arms.** `w_autocorr = w_profile = w_distribution = 0.0` — the `off` arm, all three together.
  `w_cross = w_thick = w_prog = 0.0`, so SEFL is off in both A9 arms as intended and this is not
  A7 by accident. `train_steps = 2400`, `layout_mode = resample`, `prior_mode = correlated`,
  `expr_mode = zinb-flow`, `decoder_mu_link = exp` — the shipped configuration.
* ⚠️ **Uninformative (d) — I wrote that it "does not fire, and it was *checked*". That was wrong,
  and the six-fit aggregate found out why.** The three alarm lists are empty and persisted, but on
  this run they were **never armed**: `train_ctfflow` gated both alarms on `sefl_teacher is not
  None`, and A9 runs SEFL off in both arms. An absent measurement, reported by me as a measured
  silence — §4.2f's own distinction, in the entry that invoked §4.2f. See the A9 result below and
  the fix in `_log_sefl`.
* **Density is comparable across sections**: 4073/4187, 4169/4102, 4110/4162 — ratios 0.97 / 1.02 /
  0.99, a spread of **1.05x**. The same regime the boundary stratification found readable, so the
  per-section numbers can carry a §4.2d per-fold envelope.
* **A second healthy `spatial_ratio` trajectory**, n = 241: median **1.644**, last 1.876, minimum
  **−0.0342**. Against the first fit's median 1.299 and floor 0.5467. Both are **SEFL-off**, so the
  caveat in `Config.sefl_spatial_collapse_warn_fraction` is unchanged — the population the alarm
  will actually be applied to is still unobserved — but the 0.05 threshold now has two independent
  real-data anchors rather than one.

### 🚨 Uninformative (c) is withdrawn as malformed, and the reason is the finding

(c) required the `off` arm to reproduce the recorded `(2x, weights off)` row: `morans_pearson`
**0.9316**, `gearys_pearson` **0.9022**, `umap_mixing` **0.9145**. Fit 1 returned **0.5438 /
0.5419 / 0.8201**.

**That is not a reproduction failure. The two are not the same quantity.** The recorded row comes
from `scripts/t09_report.py --steps 1200` on the **synthetic fixture**, an `alternating` holdout and
three interior LOSO folds, scored with **T08 kernels**. A9's fits are **real tier-1 STARmap**, the
held-out `paper_2_4_6` sections, scored with **`bench3.evaluate_paper`**. Different dataset,
different holdout, different instrument — three ways. A condition comparing them could only ever
fire, on any run, whatever the model did.

**The condition is removed, not relaxed.** There is no same-instrument prior to reproduce.
Conditions (a), (b) and (d) stand exactly as written. This is recorded **on fit 1, before the other
five landed**, so it cannot be a condition dropped once it became inconvenient — and the aggregator
prints the withdrawal and its reason rather than silently omitting a check.

### ⚠️ And the premise of A9 was understated — the weights were selected on the FIXTURE

This is what (c)'s malformation exposed, and it is more important than the condition.

The pre-registration said the weights ship at 0.5 "established by one seed with margins inside the
envelope". **That is too kind.** The selection that put them at 0.5 ran on the **synthetic
fixture** — 19 fits in 9195 s, roughly 8 minutes each, against the 56 minutes a real tier-1 fit
takes. So the true statement is:

> `w_autocorr = w_profile = w_distribution = 0.5` ship **on**, selected by an aggregate rank over
> six metrics on the **synthetic fixture**, with per-metric margins inside the envelope, on **one
> seed** — and **never measured on real data at all.**

That is the same pattern R11 already burned. There, a fixture signal put `hybrid` ahead on a
tie-break whose margin sat inside R10's envelope, and real data reversed it decisively: the fixture
was **"underpowered, not wrong"**, because its flanking baseline sits at 58 % of its ceiling against
real tissue's 79 %, so it over-rewards a generative component at any number of seeds.
`progress/fixture_limitations.md` records two more cases where a fixture measurement could not have
surfaced what real data did.

**So A9 is not a confirmation run.** It is the **first real-data measurement** of a component that
ships on, chosen on a fixture that is documented to over-reward exactly this kind of addition. That
raises the prior on a null or a negative and it raises the value of the experiment. **No branch,
threshold or reading changes** — the criteria were fixed before the fits and stay fixed. What
changes is one sentence of framing, in the direction that makes the shipped configuration look
*less* well established, which is the direction that costs rather than helps.

---

# A9 — RESULT: **UNINFORMATIVE**, and the two things it found instead (2026-09-07)

Six fits, three seeds, both arms. `reports/t10_a9.md`.

## The pre-registered verdict: UNINFORMATIVE, condition (a)

| metric | role | mean (on−off) | per seed | env (a) | vs | env (b) | vs | signs |
|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | PRIMARY | +0.0057 | −0.1685 / +0.0782 / +0.1073 | 0.2894 | 0.02x | 0.3973 | 0.01x | **disagree** |
| `gearys_pearson` | PRIMARY | +0.0032 | −0.1662 / +0.0684 / +0.1075 | 0.2861 | 0.01x | 0.3929 | 0.01x | **disagree** |
| `marker_depth_r` | PRIMARY | −0.0709 | −0.0251 / −0.0117 / −0.1759 | 0.1225 | 0.58x | 0.4323 | 0.16x | agree |
| `umap_mixing` | secondary | −0.1857 | −0.2134 / −0.2121 / −0.1317 | 0.1281 | **1.45x** | 0.1337 | 1.39x | agree |
| `marker_field_r` | secondary | −0.1278 | −0.0707 / −0.1533 / −0.1593 | 0.0596 | **2.14x** | 0.0989 | 1.29x | agree |

**Condition (a) fires and nothing may be read from the primaries.** The worst primary envelope is
**0.4323**, against the 2x0.0335 = 0.067 the condition names — **6.5x over**. The design is far
noisier than the instrument that produced the number being tested, exactly the situation (a) was
written to catch, and it was written before the fits so it cannot be invoked selectively. Both
autocorrelation primaries also have signs disagreeing across seeds (one seed at −0.17, two at
+0.07/+0.11), which would have blocked POSITIVE independently.

**So the question A9 asked is not answered.** The weights remain **unestablished**, and the §4
close-out disposition is unchanged: they ship at 0.5 on a fixture-selected aggregate rank, and the
paper must say so. What A9 adds is that a three-seed real-data design **could not resolve it** —
which is itself worth reporting, because it means "run more seeds" is not a cheap path either.

⚠️ **And the common-mode note applies with full force.** The pre-registration recorded before these
fits that the three known distortions cannot separate the arms but *can* attenuate a real
difference. With an envelope this wide, that caveat is not decorative.

## 🚨 Finding 1 — the collapse alarm is disarmed whenever SEFL is off

`train_ctfflow` passes `alarm=(sefl_teacher is not None and sefl_ramp(...) > 0 and step >= ...)`,
and `sefl_teacher` is built **only when `max(w_cross, w_thick, w_prog, w_prog_wrong) > 0`**. The
shipped configuration has all four at **zero**. Therefore:

> **On the shipped model, `check_collapse` and `check_spatial_collapse` never run.** Both alarm
> lists are empty on every fit of every campaign that shipped SEFL off — and that emptiness means
> **"never armed"**, not "did not fire".

My own aggregator printed the wrong one of those two on its first run: *"(d) does not fire — no
collapse or inversion alarm on any of the six fits, and the alarm record was **checked** rather
than assumed absent."* That sentence is false, and it is `specs/10` §4.2f's exact distinction — the
rule I wrote after A7 — failing in a new place, in an instrument I wrote to enforce it. Fixed: (d)
now reports **CANNOT BE EVALUATED** and counts as **fired**, because a condition that cannot fail
is not a condition.

**This is a defect in the model code, not only in the report.** The alarm watches a statistic —
per-gene variance of generated expression against real — that has nothing to do with SEFL. Gating
it on the teacher's existence means the diagnostic is armed only in the configuration that does not
ship. Recorded as an owed fix; not fixed here, because changing when an alarm arms mid-campaign
would change what every subsequent run reports.

## 🚨 Finding 2 — and the disarmed alarm would have fired on all three ON fits

The trajectory is persisted whether or not the alarm is armed, so the threshold can be applied by
hand:

| arm | median `variance_ratio` (3 seeds) | vs 0.25 |
|---|---|---|
| `off` (0.0) | **0.8555 / 0.7707 / 0.8129** | 3x above, all three |
| `on` (0.5) | **0.1404 / 0.0825 / 0.1705** | 🚨 **below, all three** |

**A7's collapsed SEFL arm settled at 0.105–0.193 on this same statistic.** The metric-aware arm
sits at **0.083–0.171** — the same regime — on all three seeds, at the weights that **ship**.

⚠️ **Stated at exactly its strength and no more.** This is not a pre-registered test, it decides no
A9 branch, and the A9 verdict stays UNINFORMATIVE. It is a **diagnostic trajectory**, reported
because a number that would have raised an alarm had the alarm been armed must not reach a reader
as silence — which is the whole content of §4.2f.

**A candidate mechanism, and it is R4's shape again.** The `spatial_ratio` medians run
**2.43 / 2.54 / 2.17** on the ON arm against **1.64 / 1.39 / 1.39** off. So with the autocorrelation
loss active the generated field's Moran's I relative to real goes **up** — past the target, to
2.2–2.5x — while the per-gene variance the field carries goes **down** by 5–10x. Moran's I is
normalised by variance, so a field can score well on it while its amplitude drains. That is a
likelihood-shaped term being satisfied by removing the structure it was meant to preserve, which is
R4 (i)–(iv)'s pattern in a fifth place. **Candidate, not established**: it rests on two diagnostic
trajectories at one budget, and the six metrics cannot separate it from the noise (a) already
flagged.

## Wall clock — an observation the pre-registration did not name

Arm `off` averages **57 min**, arm `on` **93 min** — **1.63x**. Whatever else the three losses do,
they cost 63 % more compute per fit. That is not a criterion and changes no verdict; it belongs
beside the disposition, because "ships on, established by nothing" now reads "ships on, established
by nothing, and 1.63x the compute".

## 🚨 Adding a buffer stranded every earlier checkpoint (2026-09-08)

The `umap_mixing` re-score failed on `runs/pilot/model_exp_2400.pt`:

    RuntimeError: Error(s) in loading state_dict for CTFFlow:
            Missing key(s) in state_dict: "decoder.gene_theta".

**A defect I introduced.** `gene_theta` arrived with `Config.decoder_theta_mode` (2026-09-01),
registered **unconditionally** on `ZINBDecoder` with a comment claiming it was done that way "so a
checkpoint from either mode loads into either decoder". That comment was wrong in **both**
directions: an *earlier* checkpoint has no such key and strict loading rejects it, and a
moment-matched checkpoint loaded into a `learned` decoder would hit a size mismatch instead.

⚠️ **This is the `content_hash` stranding hazard's hard form, and I recorded the mild one twice
without noticing the severe one.** A `content_hash` change refuses a *resume* and is worked around
by re-fitting or by not resuming. A **state_dict** key cannot be worked around at all: the weights
exist, they are correct, and no flag loads them. `runs/pilot/model_exp_2400.pt` is the fit behind
`reports/pilot.md` §13 and `reports/r11_starmap_layout_modes.json` — months of compute — and it was
unloadable for a week without anyone noticing, because nothing tried to reuse it until this
re-score.

### The fix, and why it is not `strict=False`

`ZINBDecoder._load_from_state_dict` now resolves it from what the buffer **is** in each mode:

* under **`learned`** it is `zeros(0)` and **nothing reads it** — `_fixed_theta` is reachable only
  from the moment-matched branch — so its presence, absence or width in a checkpoint is
  immaterial. A missing key is filled in; a full `(G,)` table from a moment-matched fit is
  dropped rather than allowed to raise.
* under **`moment_matched`** it **is** the emitted dispersion, so a checkpoint without it is
  refused with a message naming the reason and the two ways out. Convention 6: `strict=False`
  would have "fixed" the error by loading a zero-width dispersion table into a decoder that reads
  it — the silent fallback, which is the failure this project has paid for repeatedly.

`test_a_checkpoint_written_before_gene_theta_existed_still_loads` pins all three: the legacy load
reports no missing keys, a moment-matched table loads into a `learned` decoder without a size
error, and the moment-matched-without-the-buffer case **raises**.

### The rule this earns

**A new buffer or parameter is a breaking change to every existing checkpoint, and the breakage is
silent until something tries to load one.** `Config` fields at least announce themselves through
`content_hash`; `state_dict` keys announce nothing. Any future buffer needs a
`_load_from_state_dict` path decided at the same time it is added, stating what an older checkpoint
should do — and the decision has to distinguish a buffer that is **read** from one that is merely
**present**. Belongs beside the `content_hash` hazard in the standing-risks list, as the more
serious half.

---

## The 0.0335 correction — every clearance figure re-read against its own envelope (2026-09-08)

Full derivation: **[reports/envelope_correction.md](../reports/envelope_correction.md)**. Zero fits,
zero generation; every number from artifacts already in the branch.

**What was wrong.** `specs/10` §4.2a says an envelope is per-metric, per-arm, per-dataset and
per-gate. The practice was to divide every margin by **0.0335**, which is none of those: it is the
**maximum over six metrics** of a spread measured on the **synthetic fixture** by
`train/select.py::section_scores`. Defect (1), pooling, was avoidable from day one —
`reports/envelope_synthetic.md` carries the per-metric table and says in its own text that it *"is
the right thing for T10 to quote per claim"*. It was pooled anyway, for three weeks, across three
documents. Defect (2) is applying a fixture figure to real data.

**Verified against the record before trusting the method.** Recomputing §4.2a's tier-1 table from
`t09_envelope_starmap_seed{2,3,4}.json` reproduces every cell (0.0054/0.0574, 0.0027/0.0595,
0.0068/0.0190, 0.0049/0.0148, 0.0084/0.0472); the `deep_starmap` "worse arm alternates by metric"
row reproduces on all five metrics; A7's six ratios reproduce **exactly** (4.68x / 3.76x / 2.75x /
1.55x / 1.31x, and 0.78x/0.83x on the two unreadable ones); A9's envelope (a) of 0.1225 reproduces
to four decimals.

**Recomputed, verdicts unchanged, numbers moved:**

| figure | was | is |
|---|---|---|
| `expr_mode` tier-1, three metrics | 4.6–5.3x | **2.2x / 2.3x / 7.4x** (per-metric per-arm, 3 seeds) |
| tier-1 reconstruction headroom | 4.6x | **3.3x** |
| `deep_starmap` headroom over the best copy | 0.5x | **0.37x** |
| `deep_starmap` headroom over the operational copy (R14) | 2.6x | **2.0x** |
| R14's donor-rule cost | 3.5x | **2.7x** |
| metric-aware selection margins | inside by 3–19x | inside by **1.6–19x** |

Note the `expr_mode` row: the error runs in **opposite directions** on different metrics — the two
autocorrelation metrics were overstated by more than 2x and `umap_mixing` was understated. That is
§4.2a's own claim arriving in this project's own headline numbers.

**Two verdicts changed.**

1. 🚨 **The fixture `layout_mode` gate was NOT "decided inside the noise".** *"0.0344 against a
   0.0335 envelope, 1.03x the noise floor"* compared a max-over-metrics margin to a
   max-over-metrics envelope. Per metric, from the same nine fits: `umap_mixing` +0.0401 against
   0.0115 = **3.49x toward `resample`**, `celltype_localization` −0.0193 against 0.0068 = **2.84x
   toward `hybrid`**, both 3/3 on sign, the other four inside. The fixture was not too blunt to see
   a difference — it saw **two, and they disagreed**. The recorded gloss *"underpowered, not wrong"*
   is withdrawn wherever it appears; *over-rewards a generative addition* survives as the objection.
2. 🚩 **BOUNDARY ELIMINATED → NOT READABLE.** `t10_marker_field_boundary.json` hard-codes
   `"envelope": 0.0335` on a `bench3.evaluate_paper` number. Against A9's `paper_marker_field_r`
   (0.0596) the same −0.0231 gap is **0.39x** (⚠️ **corrected** — an earlier revision put 1.56x
   here, which is the ratio against *instrument A's* 0.0148, not A9's), while against instrument A
   it is **1.56x**: the two candidate divisors give **opposite** answers. Neither envelope is
   itself inadmissible (fold-aggregated where the effect is per section; wrong arm). The **raw**
   finding is untouched and still carries the redirect: the boundary section has the **smallest** of
   the three deficits (0.1729 vs 0.1877, 0.2043).

**And the finding that was not anticipated: the corpus contains two instruments, and every envelope
is on the wrong one.** `train/select.py::section_scores` (internal LOSO, folds `section_3/5`) versus
`bench3.evaluate_paper` (`paper_2_4_6`). `specs/10` §5 forbids mixing them by name. **Every
three-seed envelope ever quoted is the first; the headline table, R11, the marker deficits and the
boundary work are all the second.** On tier-1 their per-metric envelopes differ by **2.6x–6.7x**.

✅ **The missing envelope existed inside A9.** `t10_a9_{0,05}_s{1,2,3}` are three seeds, two arms,
2400 steps, `paper_2_4_6`, pinned evaluator — per-metric spreads from **0.0009 to 0.2894**, and they
carry `paper_gene_mean_spearman`, which no instrument-A envelope has. A9 was filed UNINFORMATIVE
about the question it was built for and nobody looked at what else it had measured.

🚩 **Six figures could not be recomputed and are flagged, not substituted** (correction §3): R11's
two layout deficits, `marker_field_r`'s tier-1 deficit, the boundary gap, `gene_mean_spearman`'s
position against its floor, and §13.2's prior-campaign localization lead. **Five share one blocker
and it is not compute**: `runs/pilot/model_exp_2400.pt`'s configuration is recorded in none of the
artifacts that cite it — `r11_starmap_layout_modes.json` and `r11_{resample_grid,determinism}*.json`
carry `model`, `decoder_mu_link`, `train_steps` and `seed`, and no `config_hash`, no
`text_emb_mode`, no metric-aware weights. So A9's envelope cannot be matched to that arm, and §4.2a
forbids assuming: on the `deep_starmap` `text_emb_mode` gate the arms' envelopes differ by up to
2.6x and the worse arm alternates by metric.

**Two rules earned**, added to `specs/10` as **§4.2a-i** (an envelope is per *instrument*; where no
matching one exists, report the raw margin and say the multiple is unavailable — the nearest
available figure is not a fallback) and **§4.2a-ii** (an artifact records the arm it describes, or
its envelope can never be matched later). Both are in the §4.2* meta table, which now names nine
choices rather than seven.

⚠️ **Two things this correction did not touch and should not be read as having fixed.**
`scripts/t10_marker_field_boundary.py` still hard-codes 0.0335, and the per-seed audit reports
(`t09_envelope_starmap_seed*.md`, `t09_audit_*.md`) still carry a `vs 0.0335` column. They are
measurement artifacts and were left as written; anyone re-running or re-reading them gets the old
divisor back.

⚠️ **Estimator.** Margins are the **median** across seeds (§4.6). The record's own R10 recomputation
used the **mean**; the two differ by at most 0.3x here (`marker_depth_r` 1.21x vs 1.51x) and no
verdict turns on it. Stated so the choice is visible.

### Follow-ups to the correction (2026-09-08, same day)

**1. The r11 config was never lost — and that is the sharper version of §4.2a-ii.** The first
revision of the correction implied the arm behind the six-metric table could no longer be
identified. Wrong: `scripts/t10_rescore_saved.py` writes and reads the model file as
`{"config": <every Config field>, "state_dict": ...}` and its own `--preflight` branch already
prints five gates from it. The whole `Config` has been in `runs/pilot/model_exp_2400.pt` throughout.
**The defect is that the reporting scripts never copied the block into their output** — ordinary,
cheap to fix, and easier to miss precisely because nothing is absent.

New: **`scripts/t09_recover_checkpoint_config.py`**. Reads both payload shapes — a saved fit
(`{"config", "state_dict"}`, every gate a direct read) and a resumable `FitCheckpoint`
(`config_hash` only, so the gates are *inferred* from evidence and labelled as inference:
`history.terms` omits `autocorr`/`profile`/`distribution` **iff** all three metric-aware weights are
zero, since `train_ctfflow` builds no `LOSOScheduler` at zero weight and `metric_aware_terms` is
never called; `teacher is None` **iff** every SEFL weight is zero). `--patch` writes a
`recovered_config` block into named artifacts. The command:

```
python scripts/t09_recover_checkpoint_config.py runs/pilot/model_exp_2400.pt \
    --patch reports/r11_starmap_layout_modes.json \
    --patch reports/r11_resample_grid.json --patch reports/r11_resample_grid_umap.json \
    --patch reports/r11_determinism_a.json --patch reports/r11_determinism_b.json
```

⚠️ **Not run here — this container has no `torch` and no `numpy`.** What *was* verified: the
payload logic against both shapes and both error paths (`--self-check`, passes), and the patcher
against a copy of `r11_determinism_a.json` — idempotent, refuses overwrite without `--force`, every
original field byte-equal afterwards.

**2. Both hard-coded divisors are closed.** A correction that only rewrites prose leaves the defect
in the code that produced it.

* `scripts/t10_marker_field_boundary.py` hard-coded `ENVELOPE = 0.0335`, and **every branch** of its
  pre-registered criterion compared against it. `--envelope` is now required with no default; a
  numeric value also requires `--envelope-source` (§4.2g); `--envelope unavailable` returns a new
  **NOT READABLE** outcome rather than falling through to `boundary_eliminated`, because a criterion
  that cannot fail must not be scored as passed (§4.2j). Uninformative (b) gained the same
  three-way distinction. `reports/t10_marker_field_boundary.{md,json}` regenerated under it:
  **every measured field byte-identical** to the superseded pair — `deficits`, `gap`,
  `floor_per_section`, `arm_per_section`, `density_*`, `pooled_median`, both uninformative
  conditions — with only `envelope` (0.0335 → `null`) and `outcome` moving. The `.md` carries a
  header saying so, because a regeneration that replaces a verdict must not read as a
  re-measurement. ⚠️ Produced with a two-function `numpy` stub (`mean` over two floats, `isfinite`
  on a float — the script's only `numpy` uses); the deficits reproduce the committed ones exactly,
  which is checkable from the file.
* `scripts/t09_audit_starmap.py` printed a `vs 0.0335` column on every gate audit it ever wrote.
  Now `vs own envelope`, populated only from `--envelopes` (per metric, with a required `source`
  naming dataset, holdout, instrument, arms and seeds) and `—` otherwise. `vs fold spread` —
  always the honest column at n = 2 — is unchanged.

The eleven `.md` artifacts carrying the old column are **annotated, not regenerated** (regenerating
needs fits). Two different notes, because they are two different situations: the seven post-fix seed
reports get the per-metric per-arm replacement, derivable from those same files; the four
`t09_audit_*.md` are marked **superseded outright**, since they are one seed *and* pre-frame-fix —
visible in their own tables as a negative `marker_field_r` — so their margins are as retired as
their divisor.

**3. The two-scorer finding is out of the correction report and into the documents.** It was the
most consequential thing in it and it was sitting in one file. Now `reports/advisor_report.md`
**§4b** (a peer of §4a's collapse-alarm finding) and `reports/t09_closeout.md` §6 as **§4.2a-i**,
both stating it plainly: every envelope in this project was measured by a different scorer, on a
different design, from the numbers it was used to judge; tier-1's per-metric envelopes differ
**2.6x–6.7x** between the two, and on the autocorrelation metrics the real instrument-B envelope is
**~9x** the pooled figure that was used — in the direction that flatters every clearance. A third
axis of error on top of pooled-across-metrics and pooled-across-arms.

**4. The A9 retrieval failure is recorded as its own finding** — advisor **§4c**, close-out §6,
`specs/10` **§4.2a-iii**. A9's instrument-B envelope existed for a month and nobody looked, because
the run was **filed by its verdict rather than by what it measured**. "The project has no real-data
envelope on the pinned instrument" and "A9 measured one" were both true, and only the first was
written down. **Retrieval, not measurement**, and the kind that recurs: a null result's measurements
stay valid long after its verdict stops being interesting, and indexing a run by the question it was
designed to answer makes them unfindable to anyone asking a different one. §4.2f one level up.

### Recovery ran (2026-09-09) — the arms do not match, and it found two things it was not looking for

`scripts/t09_recover_checkpoint_config.py runs/pilot/model_exp_2400.pt --patch ...` was run on the
campaign machine; the five r11 artifacts now carry a `recovered_config` block on every arm. Every
gate is a **direct read**:

```
text_emb_mode lookup | expr_mode zinb-flow | prior_mode correlated | layout_mode field
layout_sampler grid  | decoder_mu_link exp | train_steps 2400      | expr_pca_dim 16
w_autocorr/profile/distribution 0.0/0.0/0.0 | w_cross/thick/prog 0.0 | seed 1
```

**1. It closes none of the six flagged figures, and my prediction that it would close four was
wrong.** A9's arms are `medcpt` at `expr_pca_dim=28`; the r11 checkpoint is `lookup` at 16. They
agree on the metric-aware weights — the gate the flag was *about* — and differ on two other
**fit-time** gates, so A9 is not an admissible donor under §4.2a. The way the prediction was wrong
is the useful part: **I checked the gate in question and did not ask what else would have to agree**,
which is §4.2a's own failure mode committed while writing the correction for it.

What it did settle: each row moves from *"no candidate donor known"* to *"the one candidate is
measurably the wrong arm"* — determinate, with a costed remedy (three seeds of the shipped arm on
instrument B). And the **within-r11** comparison got firmer: all five arms are provably one fit
differing only in generation-time gates, and `layout_mode` is in `FIT_INVARIANT_GATES`, asserted
bitwise across all 96 tensors by `test_layout_mode_does_not_enter_the_fit`. R11's ordering is a
clean within-fit contrast; what it lacks is an across-seed spread, which one checkpoint cannot
supply.

**2. 🚨 The six-metric "v25 shipped" column is mislabelled.** It sources
`r11_resample_grid_umap.json`, i.e. this checkpoint. Gate by gate: `layout_mode` is **sound**
(generation-time, provably fit-invariant, so the arm really does generate under `resample`);
`expr_mode`, `prior_mode`, `decoder_mu_link`, `train_steps`, `layout_sampler` all match. But
**`text_emb_mode=lookup` is ablation A3** — `specs/10` §3 says so in as many words — and
**`expr_pca_dim=16`** is the pilot stand-in the clamp rule replaced with 28. Both are fit-time and
not overridable.

⚠️ **This is the same failure the table was corrected FOR.** The 2026-09-08 revision moved it from
the superseded `hybrid` pilot row to the `resample` arm and called the result shipped. The layout
label became right; the expression labels stayed wrong, because only the layout gate was in
question. **A column is not the shipped configuration until every fit-time gate has been read out of
the artifact.** Corrected in `reports/advisor_report.md` §5 with the gate-by-gate table; the
measurement is valid and every within-r11 comparison built on it stands.

**3. 🚨 "Every absolute number was produced with the metric-aware weights active" is backwards.**
`config.py` declares `w_autocorr = w_profile = w_distribution = **0.0**`; `_starmap_run.base_config`
overrides only `BENCH3_KEYS`; `t09_ship_starmap.py --w-metric-aware` defaults to unset. Read off the
artifacts: the r11 checkpoint **0/0/0**, A7 both arms **0/0/0**, A9's `0` arm **0/0/0**, and A9's
`05` arm the **only** run in the corpus at 0.5 — the experiment that tested them.

So the absolute numbers are *cleaner* than the close-out claimed, not dirtier, and the defect is the
other one: **a value the selection chose, recorded everywhere as shipped, that no code path
produces.** The standing recommendation to zero the weights is now nearly a no-op, because `Config`
already does; what the next campaign needs is for the record to stop saying they are on.

⚠️ **And the fixture envelope ran the opposite way**: `scripts/t09_envelope.py` sets all three to
**0.5** for its nine fits, so R10's 0.0335 was measured on the weights-**on** arm while every number
it judged was weights-**off** — a fourth axis of mismatch (metric, dataset, instrument, arm) in the
one place it is most embarrassing.

⚠️ **Not checkable from this checkout**: whether a persisted `selected.yaml` carrying 0.5 exists on
the campaign machine (`runs/` is not in the repo). What is checkable is that **no run whose artifact
is in this repository resolved one** — A9's provenance reads `source: "defaults"`,
`selection_path: null`. Worth confirming before the next campaign, since `--require-config` reads it.

**Rule earned**, added to `specs/10` §4.2a-ii beside §10.1's
`test_bare_invocation_reproduces_shipped_config`: *a table may be labelled with a configuration only
when every fit-time gate has been read out of the artifact and compared — not just the gate under
test.*

### The shipped configuration has now been measured — and it was A9 again (2026-09-09)

**1. The stated absence.** No draft of the six-metric table ever measured the shipped configuration.
The earlier draft was `hybrid` + `lookup` + `expr_pca_dim=16`; the 2026-09-08 correction moved the
layout gate to `resample` and inherited the label on the two gates that were not in dispute. **Both
are ablation A3** — the pilot ran in the container, where MedCPT is unreachable, and `specs/10` §3
names that arm outright. Stated as an absence in `reports/advisor_report.md` §5.0, not as a caveat.

**2. The remedy already existed.** A9's six fits are the shipped configuration on every gate except
the disputed weights: `medcpt`, `expr_pca_dim=28`, `resample`+`grid`, `correlated`, `zinb-flow`,
`exp`, 2400 steps, SEFL zero, three seeds, `paper_2_4_6`, pinned evaluator. Referents come from the
r11 probes, which are **model-free** (they copy real cells) and were re-measured on 2026-09-08 to
four decimals, so they transfer across arms. **Cost: zero fits** — 2.85 core-hours at `w=0`
(56/59/56 min) and 4.65 at `w=0.5` (91/92/97 min), already spent and already committed.

⚠️ Corrected mid-derivation: A9's `referents` key exists but is **empty**; I first read `'referents'
in d` as carrying them. The probes come from `r11_starmap_layout_modes.json`. Also
`paper_gene_mean_spearman` is recorded pooled only in A9, with no per-section breakdown, where
`specs/10` §4.6 wants per-section values beside every tier-1 median.

**3. 🚨 The shipped configuration is WORSE than the arm labelled as it, on six of seven metrics.**

| metric | A3 column (1 seed) | shipped `w=0` (3 seeds) | change | A3 − floor | shipped − floor |
|---|---|---|---|---|---|
| `morans_pearson` | 0.6541 | 0.5574 | −0.097 | −0.3294 | **−0.4262** |
| `gearys_pearson` | 0.6535 | 0.5543 | −0.099 | −0.3306 | **−0.4297** |
| `umap_mixing` | 0.9262 | 0.8318 | −0.094 | — | — |
| `marker_field_r` | 0.6824 | 0.5655 | −0.117 | −0.2032 | **−0.3201** |
| `marker_depth_r` | 0.8331 | 0.7228 | −0.110 | −0.1464 | **−0.2566** |
| `celltype_localization` | 0.7546 | 0.7591 | +0.005 | −0.0219 | −0.0174 |
| `gene_mean_spearman` | 0.9901 | 0.9721 | −0.018 | **+0.0038** | **−0.0142** |

Every deficit below the copy floor grows, so **the negative column is stronger on the real shipped
arm than on the stand-in.** And 🚨 **`gene_mean_spearman` flips sign against its floor** — the
close-out's *"one genuinely solved thing"* is **withdrawn**: per-gene magnitude was solved on A3, not
on the method. `celltype_localization`'s +0.005 is inside its own 0.0061 envelope, i.e. a tie.

**What is still missing from a headline table** is the **comparator set** — SpatialZ, FEAST, isoST,
v20 on the same instrument and holdout — which `specs/10` §3 already establishes is not in this
repository and §12 already budgets. That, not more v25 fits, is the remaining cost.

**4. A9 has now yielded three things it was not filed under**: the instrument-B envelope, the
shipped-configuration table, and the metric-aware weight evidence. Same six files, all three
unlooked-at for a month, because the run was indexed by its verdict. §4.2a-iii has now cost three
separate retrievals.

### The selected.yaml question — PENDING, with both branches pre-registered

`scripts/t09_find_selection.py` (new; reads, never writes) searches for `selected.yaml`,
`selection_report.md` and `scores.csv`, reports the weights each carries, and **prints the roots it
searched** so a negative can be quoted at its real breadth. Run here it finds none, but this
checkout has no `runs/`, so that is not the answer:

```
python scripts/t09_find_selection.py --root "$SPATIALCPAV25_SELECT_DIR" --root /data/han/projects/Spatial3D
```

Both branches are written before the answer arrives, so it cannot be fitted afterwards:

* **(a) a `selected.yaml` carrying 0.5 exists** → 0.5 is a real persisted selection that no campaign
  run resolved; `--require-config` exists and nothing used it. A **wiring gap**, and the record
  should read "selected at 0.5, persisted, applied by no run in the corpus".
* **(b) no such file, under complete roots** → 0.5 entered from a printed selection report and was
  written into `Config`'s docstrings, `specs/10`, both reports and `--w-metric-aware`'s own help
  text as *shipped* without ever being persisted or applied. 🚨 **A seventh provenance failure mode:
  a value that was chosen, recorded as shipped, and never ran** — distinct from all six in §8c,
  because nothing is missing, nothing is mislabelled and nothing was regenerated. Every artifact is
  internally consistent and the claim is false anyway.

⚠️ Branch (b) is where the current evidence points — `Config` carries 0.0, which is where an applied
selection would show — but pointing at is not showing, and the search costs one command.

### The 0.5 has no source anywhere — the seventh failure mode (2026-09-09)

**The finder's negative was wrong, and the answer is neither pre-registered branch.**
`runs/select/starmap_visual_cortex/selected.yaml` **exists** — added at `3d57725` ("T10 pilot:
halted at step 6"), deleted at `5cd1fd6`, still in history. It carries the three weights at **0.0**,
with `train_steps: 20`, `decoder_mu_link: softplus`, `expr_pca_dim: 32`, `text_emb_mode: lookup`,
`layout_mode: field` — a 20-step smoke artifact from the halted pilot, predating the link fix, the
clamp rule and R11. Not a selection anyone would ship, and not 0.5.

🚨 **So: `0.5` appears in NO machine-readable artifact this project produced.** Not `Config`; not any
fit's recorded config (the six-metric checkpoint, A7's two arms, A9's control all read `0.0`); not
the one persisted selection that exists. It lives only in prose — docstrings, `specs/10`, both
reports, `PROGRESS.md`, and `--w-metric-aware`'s own help text, each citing the others.

**Written as the seventh provenance failure mode** (`specs/10` §4.2a-iv, advisor §6b, close-out §4).
It is stronger than "never persisted": the six in §8c are all about a *file*, and this one has no
file at its centre. Nothing is missing because nothing was ever written; **every artifact is
internally consistent and every document is wrong**, and no artifact-to-artifact provenance check can
catch it — they all agree at `0.0`. It is caught only by comparing the documents to the code.

**Instrument defect, and it is §4.2j a second time in this thread.** The first
`t09_find_selection.py` walked filesystem roots and reported NONE FOUND for a file **tracked in the
repository it was run from** — present in history, absent from the tree. Its scope caveat warned
about *roots*, i.e. space, while the file was missing in *time*, and it quoted §8c's reflog lesson in
its own output while committing it. Fixed with `scan_history` over `git rev-list --all --reflog`, so
reset-away commits are covered too; **the self-check now carries a regression on the false negative
itself**, asserting the history scan finds that exact deleted file with `w_autocorr = 0.0`. The
verdict logic is three-way (a 0.5 carrier / artifacts none of which carry 0.5 / nothing found) and
flags a low `train_steps` as a likely smoke artifact — an observation with the gates printed beside
it, not a threshold.

#### What it does to the two arguments reasoned from the 0.5

**1. The standing recommendation is WITHDRAWN as vacuous.** "Set the three weights to zero in the
next campaign" recommends changing a state that does not exist — they are zero in `Config`, in every
recorded fit config, and in the only persisted selection. ⚠️ Its first reason goes too: *"they were
selected on the fixture by an aggregate rank"* rests on the §8b-**unrecoverable** selection table,
and the one recoverable selection carries 0.0 — so **even the claim that a selection ever chose 0.5
is prose-only**. Reasons 3 (1.63x compute) and 4 (collapse regime) are measured from A9 and stand as
measurements, but they describe an arm the project ran nowhere else. What replaces the
recommendation is documentary: the record stops asserting 0.5.

🚨 **And A9 was framed backwards.** Its pre-registration reads *"Unlike A7 this is a REMOVAL
experiment: the three ship at 0.5, so the arms are `--w-metric-aware 0.5` (shipped) and 0."* With
0.5 unsourced, **A9 is an ADDITION experiment** — the same inversion A2, A4 and A7 each went through
— and **its `0` arm is the shipped configuration, its `05` arm the addition.** The UNINFORMATIVE
verdict is untouched (a two-arm contrast does not care which arm is called the baseline); what
inverts is which column of the §5.1 table is v25, and it is `w = 0`.

**2. R10's envelope was measured on an arm with no other instance in the project.**
`t09_envelope.py` sets all three weights to 0.5 for its nine fits — presumably because 0.5 was
believed shipped. So the 0.0335 **and its per-metric decomposition** describe a configuration that
appears nowhere else. Last revision called it "the wrong arm on the very gate §6 is about"; it is
stronger than that — **there is nothing it is the envelope *of*.** Every other mismatch in §4b is
between two arms that both exist; this one has a single instance, created by a script on the
strength of a value with no source. `reports/envelope_correction.md` §2.3 and §2.4 inherit it: §2.4's
tie-break is internally consistent and stands **as a statement about that arm**; §2.3's margins come
from the unrecoverable table, so their arm is unestablished. Neither conclusion moves.

✅ **What is unaffected, and it is now the figure to quote.** A9's spreads are measured on A9's own
arms, and **A9's `0` arm is the shipped configuration** — so `paper_morans_pearson` 0.0202,
`paper_gearys_pearson` 0.0241, `paper_umap_mixing` 0.0464, `paper_marker_field_r` 0.0307,
`paper_marker_depth_r` 0.1225, `paper_celltype_localization` 0.0009, `paper_gene_mean_spearman`
0.0400 are **the only envelope this project has ever measured on the configuration it actually
ships**, on the pinned instrument, at the right design, at three seeds.

### The comparator re-score, and the envelope replacement propagated (2026-09-09)

**1. The predictions were kept, so the last tier-1 gap is a scoring pass.** Every
`results/<method>/<dataset>/<holdout>/` carries `prediction.h5` beside `metrics.json`, so §13.1's
"re-run the comparators" becomes `evaluate_all --force` — **no method re-runs**. Command, checks and
cost are `specs/10` §13.1a. Three things in it are not obvious:

* ✅ **The pin holds today.** `benchmark-pbya-v3/src/bench3/evaluate_paper.py` is 764 lines and
  hashes to `7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992` — §0's value,
  verified in this checkout.
* 🚨 **Copy the tree first.** `evaluate_prediction` writes `metrics.json` **in place**, so `--force`
  destroys the prior campaign's scores. Those are the only record of what the older evaluator
  revisions returned, and §13.1 states the drift between revisions on the *shared* metrics is
  **unknown and cannot be determined from the CSV**. Re-scoring into a copy makes it determinable —
  old and new, per section, per metric. **That delta is itself a result**, and overwriting in place
  leaves §13.1's caveat permanently unresolvable.
* ⚠️ **No per-prediction scoring time exists in this project's artifacts** — every recorded duration
  is a fit or a calibration, and the r11 re-scores were never timed. So the cost is not modelled: run
  `--methods spatialz` alone first, time it, multiply. §11's own discipline, applied to the one line
  item that still has a model instead of a number. Keep UMAP **on**: `--no-umap` is what produced
  the two-`SIX` hole on the r11 arms.
* Require **`failed 0`** — `evaluate_all` counts failures and continues, and a partially re-scored
  tree is a cross-instrument tree again, silently.

What it buys beyond comparability: §13.1's specific defect is `paper_marker_field_ssim`,
`paper_gene_detection_spearman` and `paper_rare_celltype_localization` at 132/132 rows for
v18/v20/v21 and **0/3** for `spatialz` on tier-1. One pass populates every column for every method,
which is what makes a tier-1 headline table possible — and `field_ssim` beside `field_r` is §13.4's
own requirement.

**2. The envelope replacement, propagated.** A9's `w=0` arm is the shipped configuration, so its
spreads are the only admissible envelope for a shipped-arm clearance. Substituted where the
instrument and arm match, flagged where they do not — full table in
`reports/envelope_correction.md` §5b. **Only one family was substitutable**: the shipped table's
deficits below the copy floor. §2.1 and §2.2 are instrument A and keep their own run's envelopes;
§2.3/§2.4 are the fixture's orphan `w=0.5` arm; §3's six rows are r11's `lookup`/`pca16` arm and
stay flagged. A7 keeps its own 1200-step per-arm spreads.

✅ **The result is the first set of deficits in this project expressible as an admissible envelope
multiple** (§4.2b: the referent is a fixed probe, so the arm's own envelope decides):

| metric | deficit | vs the retired 0.0335 | vs the shipped arm's own envelope |
|---|---|---|---|
| `paper_morans_pearson` | −0.4262 | 12.7x | **21.1x** |
| `paper_gearys_pearson` | −0.4297 | 12.8x | **17.8x** |
| `paper_marker_field_r` | −0.3201 | 9.6x | **10.4x** |
| `paper_marker_depth_r` | −0.2566 | 7.7x | **2.1x** |
| `paper_gene_mean_spearman` | −0.0142 | 0.4x | **0.4x — inside, a tie** |
| `paper_celltype_localization` | −0.0174 | 0.5x | 🚩 **not readable** |

The pooled figure was mis-scaling in **both** directions — the autocorrelation pair reads far higher
against its own arm, `marker_depth_r` far lower — which is §4.2a's claim in the headline numbers.

🚩 **`celltype_localization` is where the replacement must not be applied, and the reason is new.**
Its `w=0` spread is **0.0009**, with `n_pred` **bitwise identical across all three seeds** (4073 /
4169 / 4110) because `resample` copies the donor's layout and cell types; two of three seeds return
exactly 0.7591. Dividing by that manufactures an 18.9x deficit from an arm that barely moves. §4.2g
says a member **degenerate by construction** contributes its level and not its spread — it was
written about a *referent*, and **here the degenerate member is the arm under test**, a case the
rule does not cover. Reported unreadable, with the raw −0.017 beside it.

⚠️ **And a correction to this same day's earlier entry.** `gene_mean_spearman`'s deficit is **0.4x**
its own envelope, so "flips sign against its floor" overstates it: the point estimate crosses, the
deficit is a **tie**. *"The one genuinely solved thing"* stays withdrawn — it is not established as
solved — but it is **not established as worse** either, and the record should not say it is.

### v20/v21 evidence audit, and the honesty question (2026-09-09)

Full report: **`reports/v20_v21_evidence_audit.md`**. Direction taken is v20/v21 as the method, v25's
field representation and negative result as the methodological contribution. The audit holds the
v20/v21 numbers to `specs/10` §4.2a–j, §4.6 and §5 — the rules v25's own claims were withdrawn
under.

**The base is thinner than the spec makes it look.** The prior-campaign `per_section_metrics.csv`
that §13.2–13.4 are derived from **has never been in this repository** — searched tree and
`--all --reflog`. So the entire v20/v21 case is attested by the spec document alone (§8b, and
§4.2a-iv's shape). The methods themselves are fine: v20 in `reference/`, v21 at the repo root, both
with `METHODS` entries, **neither carrying a `wrapper_args` key** (verified — the phrase "No
``wrapper_args``" appears only in the explanatory comment; my first check matched that comment and
had to be redone against the actual key).

**Survives**: the instrument (re-score closed it for tier-1), the runnable bare-invoked methods,
§13.3's already-caught pooling caveat, and the direction of every signal.
**Unestablished**: every margin (zero seed replication anywhere in the prior campaign; v20/v21's
determinism never measured), every level (no floor, no ceiling in §13), every number (prose-only).
**Three defects**: §13.2's "cell-count-weighted" header contradicts §13's median preamble — the
numbers are medians (0.9704 is §4.6's median against its 0.8970 mean), so the label is wrong on the
one metric §4.6 uses to show the estimators invert; §13's panel is 7 metrics, unnamed; and the
numpy-fallback screening is a run-time log check `evaluate_all` cannot re-perform, so the re-score
inherits it.
**Not fixable by measurement**: fourteen registered versions, with bench3's own notes reading
"benchmark-driven fixes" and "recalibrated on the real benchmark" — v20/v21 were selected on the
protocol they would be reported on.

**Cost**: four of six items are reading only. Two cost runs — 9 at tier-1 (three seeds of v20, v21,
SpatialZ) and 9 at `wide_3_4_5`, the second discharging the version-selection exposure *and*
answering C2, which has never been run. No comparator timing exists here; time one first.

**Honesty (§7 of the report).** The framing is sound; it is not yet honestly evidenced. The correction
that matters most is that the usual phrasing is slightly wrong: copying is near-optimal on
`deep_starmap` (98 % of ceiling) and *not* on tier-1 (81 %, headroom 3.3x its own envelope). The
sharper point is that **v20 is itself a structured copier** — R13 showed `cross-mix` under `resample`
*is* a copy — so the honest claim is *"the winning design is a copier, on a task where copying
reaches 81–98 % of what is achievable, and v25's negative explains why that is the winning design
rather than an embarrassment."* Publishable and interesting; dishonest only if reported without the
copy floor beside it, which the re-score now makes computable for the first time.

Reviewer objections in order: single-seed headline in a paper about repeated-seed benchmarks (nine
runs fix it); version selection on the reporting protocol (disclose, and report the wide regime that
did not drive development); v25 characterised on ablation A3 rather than its shipped configuration
(A9 already fixes this at zero cost, and the negative is *stronger* there); and asymmetric rigour,
which generalises the first two. **The one thing that would make it dishonest is spending v25's audit
as credibility while exempting v20/v21 from the audit** — §4.2j one level up, on the paper itself.

### Emission-repair design comparison — and the premise it does not survive (2026-09-09)

Report: **`reports/emission_repair_options.md`**. Design work only; nothing built.

🚨 **The brief's headline chain is the superseded `softplus` arm.** `reports/chain_2400.md` (and
`_calibrated`) is the arm the mu-link candidate was run *against*. `t10_chain_diagnostic.py` carries
`--decoder-mu-link {softplus,exp}` documented as *"T10 candidate 1: softplus compresses dynamic
range"*; `Config.decoder_mu_link` has defaulted to **`exp`** since 2026-08-21; and the record states
the swap as **+0.1297 → +0.4782** against tissue's +0.4635. On the shipped link the same step reads
**0.9008 → 0.5253** (`chain_2400_explink.md`) and **0.9098 → 0.5408** (`chain_2400_grid.md`) —
retention **60.5 / 61.0 %** against real tissue's **79.8 / 81.1 %**, with the generated counts *more*
autocorrelated than the real section's. So *"one operation destroys the result"* describes a link
this project stopped shipping a month ago.

🚨 **And "0.09–0.19 against 0.62" is a cross-panel comparison the record itself rejects.** The
0.09–0.19 is `deep_starmap`'s 1017 genes, most carrying no spatial signal; the 0.62 is tier-1's
28-gene marker panel where every gene is structured by construction. `progress/` names this error
against its own voided run *and* against the record's numbers: *"The panel-transfer problem that
voided the previous run applies to the record's numbers too."*

**The panel-matched failure that does exist**: `deep_starmap`, exp link, top-32 structured kept
genes — model I **0.071–0.097** against real **0.283–0.291**, retention **25–34 %**. Tier-1 at exp
shows ~103 %, which the record forbids quoting as a shipped-decoder property. **No measurement
separates dataset from saved-model artifact**, and the record already names that as the next step.
⚠️ Also unremarked until now: every chain artifact runs at `expr_pca_dim=16`, the pilot stand-in.

**An arithmetic bound rules out two candidates from data already held.** With `s = V/(V+N)` and R4
(v)'s decomposition (overdispersion 0.568–0.614, Poisson 0.290–0.336, ZI 0.080–0.101), scaling the
noise by `f` gives `s' = s/(s + f(1-s))`. Removing overdispersion entirely (`f = 0.41`) reaches
0.194–0.364; **pure Poisson (`f = 0.31`) reaches 0.242–0.431**. Reaching 0.62 needs `f = 0.061–0.144`
— **2.2× to 5.6× below the Poisson floor**, i.e. a Fano factor near 0.3. So **candidate A (theta
prior/floor) and candidate C (Poisson + dispersion floor) cannot be sufficient**, by algebra, not by
opinion. Equivalently: with noise at Poisson, **`Var(mu)` must rise 2.2–5.1×** — and the independent
dynamic-range statistic (`sd(log mu)` 0.777 vs tissue 1.213, a variance ratio of 2.44×) agrees in
order of magnitude. Two routes, same answer.

**Candidates costed** (A–D as briefed, plus three the brief omits): **E** raise `Var(mu)` — the
binding constraint, and nothing in A–D targets it except B if B is specified against `Var(mu)` rather
than the share; **F** emit the conditional mean instead of counts — a framing decision, not a repair,
and `duplicate_profile_rate` exists to catch it; **G** replace the ZINB likelihood with a proper
scoring rule — the only candidate aimed at R4's stated root cause.

⚠️ **B and D each carry A9's gaming surface.** B penalises a *variance-normalised* ratio, which can
be satisfied by shrinking the denominator — exactly what drove `variance_ratio` to 0.08–0.17 while
`spatial_ratio` rose. D breaks the conditional independence the retention statistic is *defined*
under, so it can raise retention with no per-cell gain; both need that failure mode pre-registered as
the expected one, with fidelity controls reported beside the target.

**Recommended order: two measurements before any design.** (0) re-run the chain on the shipped
configuration — exp link, `expr_pca_dim=28`, GT-matched density, both datasets, one panel definition
— generation only, no fits; (1) measure `Var(mu)` against the real latent's per gene, hours. Then A
as a *cheap test of the diagnosis*, B as the first candidate fix, D and G after.

**Honest view, recorded because it was asked for plainly**: the 0.09→0.62 gap is not a measured gap
and I would not fund work against it. Against the real 25–34 % gap, **no single candidate closes it**
— the noise side is bounded at ~0.43 and the target needs `Var(mu)` to rise as well, so the fix is a
combination (bound the dispersion **and** raise `Var(mu)`), which is a T06 redesign and should be
costed as one. And nothing should start before step 0: weeks of design against a superseded arm is
the expensive version of the mistake this project has spent five rounds catching cheaply.

### Steps 0 and 1 specified; the combination costed; two premises withdrawn (2026-09-09)

Added as §§7–10 of `reports/emission_repair_options.md`.

**Two corrections, both from the brief and both recorded as findings.** (1) `chain_2400.md` was
handed over as the diagnosis without checking its arm — it is `softplus`; the shipped `exp` link
gives 0.9008 → 0.5253 and emits counts **1.13× more autocorrelated than the real section's**.
(2) The 0.09-vs-0.62 comparison was repeated across panels after this project had already voided a
run for that error and extended the criticism to its own numbers. ⚠️ **Both are the same shape and
it is the third instance this week**: neither was a mistake about a number, both were **an arm or a
panel inherited with a number** — the r11 six-metric table (§4.2a-ii) and the 0.5 weights
(§4.2a-iv) are the other two. The companion rule: *a number carries its panel and its arm, and
quoting it without them is quoting a different number.*

🚨 **Step 0 cannot be run as specified.** `t10_chain_diagnostic.py:341–345` **hardcodes**
`text_emb_mode="lookup"` and `expr_pca_dim=16` with no CLI override, so **every chain artifact in
`reports/` is a `lookup`/`pca16` arm** — the six-metric table's mislabelling, in a second instrument.
Four flags are needed first (~half a day): `--text-emb-mode`, `--expr-pca-dim`, `--match-density`
(the artifacts emitted 11k/48k/268k cells against a GT of 4 187, and kNN Moran's I rises with
density), and **`--top-k-by real`** — selecting the panel by the *real* section's I, identically on
both datasets, which is what §0a's error was made of and which must never select on the model side.
Commands are in §8.2. Step 1 (`Var(mu)` against the encoder latent's, per gene) is one more block in
the same run and **costs nothing extra if added before step 0 launches**.

**The number step 0 exists to produce** is `I(model counts)` **and** `I(real counts)` on the same
top-32 real-selected panel on both datasets — not the model's retention, which has different
denominators on the two sides.

**Judgement, asked for plainly.** If step 0 returns 60 % tier-1 and 25–34 % `deep_starmap`, that is a
**dataset property to characterise and report, not a defect worth a redesign**: tier-1 is already at
1.13× real tissue and is the dataset §0a calls the informative reconstruction benchmark; the deficit
sits on a dataset §0a says **cannot carry a reconstruction claim** (copying reaches 98 % of its
ceiling); there is an untested mechanism that predicts the split (`deep_starmap`'s median gene is
detected in **1.6 %** of cells against tier-1's **0.9999** detection rate, and sparse counts are
Poisson-bounded for real tissue too); and adopting a new emission re-opens every absolute number in
the project. 🚩 **Reversed if** step 0 shows real tissue's own I on that panel is high (≥ 0.28) while
the model's stays at 0.07–0.10 — then sparsity does not explain it and it is the emission's defect.
Either branch is publishable as *"retention is panel-dependent: at parity on a 28-gene curated panel,
3–4× below on a 1017-gene panel whose median gene is detected in 1.6 % of cells."*

**The combination, costed as one T06 redesign.** The two halves are **not equal partners**: the
dispersion floor buys a noise factor of 1.0 → 0.41 with high certainty (the term is a measured
57–61 % of conditional variance, arm-independent), while raising `Var(mu)` must then buy **2.9×–6.8×**
in the signal with low certainty — it may be small because the model cannot predict more. **The
uncertain half does most of the work.** Validation is a 2×2 (floor × `Var(mu)` term) at three seeds
= 12 fits: **11.5 core-hours tier-1** (18.7 with A9's measured 1.63× penalty if half 2 is a loss
term), **45.6 on `deep_starmap`**; implementation 3–4 days. **Three pre-registered gates, each able
to stop the work**: (1) after half 1 alone, `f_overdispersion` → ~0 and `s` must land in **0.19–0.36**
— §2 predicts that interval, so landing outside it means the decomposition is wrong; one fit, ~1
hour, and the most informative hour in the plan; (2) step 1's `Var(mu)` ratio must be ≤ 0.4, since
≥ 0.8 means half 2 has nothing to fix; (3) after half 2, `Var(mu)` up **and** fidelity held, or the
term is manufacturing unconditioned variance — A9's failure in a new place. **Stop rule: if gate 1
or gate 2 fails, half 2 is not built.** Beyond the fits, adoption re-opens every absolute number,
re-derives T06's acceptance criteria, and needs a calibration branch (`calibrate.py` is ZINB-only).

---

## 2026-09-09 — step 0 / step 1 instrumentation: four flags on the chain diagnostic

Built, on instruction, and **only** this: the four flags step 0 needs plus step 1's block in the
same run. The 2×2 and both halves of §10's redesign are **not** built — gates 1 and 2 come first
and either can stop the work.

**Why it was needed.** `scripts/t10_chain_diagnostic.py::main` hardcoded `text_emb_mode="lookup"`
and `expr_pca_dim=16`, so every artifact under `reports/chain_*` is an ablation-A3 `lookup` arm at
the pilot's PCA dim whatever it is labelled — the six-metric table's mislabelling, in a second
instrument. It could not express the shipped configuration at all.

**What went in** (`scripts/t10_chain_diagnostic.py`, +713 / −72):

| flag | default | note |
|---|---|---|
| `--text-emb-mode {medcpt,lookup}` | omitted | fits with a **live** channel through `build_entity_embeddings`; raises rather than falling back to zeros |
| `--expr-pca-dim N` | 16 | plus `specs/10` §0's `clamp_config_to_input`, so the width is the rule's, not a hand-picked value |
| `--match-density` / `--density-seed` | off / run seed | thins **before** the neighbour query, conditioning and metric |
| `--top-k-by {all,real,model}` / `--top-k` | `all` / 32 | `real` is the one to quote; `model` prints its own 🚩 |
| `--self-check` | — | 20 assertions on the panel and density logic, seconds, no fit, no data |

**The text channel has three states, not two**, and this is the correction that matters most for
the record. The old default supplies **zero vectors**; A3's `lookup` arm supplies the MedCPT
vectors and zeroes the *projection* inside `TextGroundedEmbedding._text_channel`. They are not the
same run — `distillation_loss` reads `text_vecs` directly and sees something in one and nothing in
the other. Calling the old default "the lookup arm", as this script's docstring did, was a fourth
instance of a run being filed by what it was assumed to be. The docstring, the console line and the
report now all say "zero vectors (neither A3 arm)".

**Four defects fixed while in there**, each of which could have produced a wrong number silently:

1. An unknown `--section` produced an all-False mask and a NaN reference **after** the fit. It now
   raises before the fit and lists the ids the file carries. `--section section_2` is tier-1's
   default and would not have transferred to `deep_starmap`.
2. The ground truth's gene order was assumed equal to the training volume's — every stage indexes
   `model.embeddings.gene` by column number. It is now checked, and names the first difference.
3. `--target-z` was unchecked against the volume. Outside the z bbox every GRF query is clamped to
   a bounding-box face under a `BBoxClampWarning` this script suppresses; a boundary plane carries
   R3's one-sided-evidence deficit. Both are now flagged in the report.
4. The ground truth was read three times (reference, verdict, and once per caller). It is read once,
   before the fit.

**Step 1** is folded into the same run rather than given a `--load-model` reader, as decided:
`real_section_reference` now returns `h1 = encoder(real counts)` beside its rows, and the same
decoder, size head and panel are applied to it. The "Candidate 2" block gained a **real latent**
column, which is the matched tissue-side `sd(log mu)` the record has been quoting as *"tissue's
1.213"* with no source. It also reports `share_shape_bounded` beside the unbounded share, since the
unbounded one exceeds 1 under a negative covariance (`t09_structured_share.py` measured 1.21) and
only the bounded one can carry a threshold. The pre-registered ratio
`Var(log mu_gen) / Var(log mu_real)` is computed per gene and printed: ≥ 0.8 means §2's binding
constraint does not exist, ≤ 0.4 confirms it, between is uninformative.

**Scope honesty.** The JSON sidecar changed shape — an object carrying the arm, the density, the
panel and its gene names, with the stage list under `stages`, rather than a bare list of stages.
Nothing reads it programmatically (checked). A stage table that does not say what it measured is
what §4.2a-iii is about. The `.md` gained a "What this run is" block for the same reason, so a
default-flag run reproduces the same **numbers** but not a byte-identical file.

**Verification, and its limit.** This container has no torch. `ruff check` and `ruff format` are
clean; the module imports and all three command lines parse; `--self-check`'s **20/20** assertions
pass on the numpy/scipy logic — panel size, ordering, determinism, tie-breaking by column index,
constant columns sorting last, `rank_normalize` commuting with column selection (so the panel does
not itself move a stage's `I`), density subsampling's size/uniqueness/seed-determinism/no-upsampling,
and the variance summary's median-of-ratios and bounded share. It also confirms the mechanism the
density flag exists for: the same field measured at 4 000 and at 400 cells gives `I` 0.1037 vs
0.0739. **Not verified: anything touching torch** — the live MedCPT channel, the clamp against a
real header, the encoder path, both decompositions. `--self-check` should be run on the campaign
machine before the fits; it exercises the import as well.

The two commands are in `reports/emission_repair_options.md` §8.2.

---

## 2026-09-09 (b) — `--load-model`, A1 built and pre-registered, two §4.2 rules recorded

**Step 0 reported, and it moved three things.** Tier-1 ran clean and emits counts at **1.11x** the
real section's Moran's I (+0.5134 against +0.4635) — no deficit to repair on the headline dataset.
🚩 **Corrected by A1 (2026-09-09 c): that 1.11x is over-smoothing, not fidelity.** On the same
emission the tissue's own latent yields +0.4053 and the model's +0.5134; the model's latent is 1.28x
smoother than the tissue's on tier-1 and 2.46x on deep. It is not a reconstruction claim.
`deep_starmap` came back at **+0.0729 against +0.3236**, which is §9's pre-registered reversal
condition — and the number is **inadmissible on two independent grounds**, both accepted:

1. **The fit never converged spatially.** `check_spatial_collapse` fired at **122 of the checked
   steps**, 79 of them *inversions* (worst ratio −0.8314), and was **still firing at step 2399, the
   last checked**, at −0.0129. The alarm's own healthy reference floors at **+0.5467**. Tier-1 from
   the same script the same day was silent after step 360.
2. **It ran under tier-1's GRF length-scale.** `main()` hardcodes `ell=(116.3, 116.3, 132.0)`, which
   `reports/pilot.md` records as the fit *on tier-1*. `CTFFlow` reads `cfg.ell_*` straight into the
   GRF; T03 ships `fit_lengthscale_from_sections` and the script bypasses it. The prior's `I` is
   0.9284 on deep against 0.9362 on tier-1 — identical, because `ell` is identical — while deep's
   own tissue latent is at 0.2415 against tier-1's 0.6253.

**Gate 2 reads NOT EVALUATED, and §10 is SUSPENDED, not re-opened.** Both sides of §8.3's
`Var(log mu)` ratio pass through the same decoder, so a narrow `mu` head pins both and the ratio is
~1 by construction. Proof it was pinned: generated `sd(log mu)` is **0.7269** on 28 genes at ~100 %
detection and **0.7254** on 1017 genes at 1.6 % detection — three decimals, two unrelated datasets.
A gate that could not be read is not a gate that passed: neither half of the redesign is built,
costed further or resumed on the reclassification, and the suspension banner is on §10 itself.

**Built** (`scripts/t10_chain_diagnostic.py`):

* `--load-model` — reads a `--save-model` checkpoint and skips the fit. **The config comes from the
  checkpoint, not the command line**, and `--text-emb-mode` / `--expr-pca-dim` /
  `--decoder-mu-link` are *refused* with it (§4.2a-ii: honouring them would report one arm's config
  over another arm's weights). The guard fires before path resolution, so it fails on the command
  line rather than behind a missing-file error. `--save-model` with `--load-model` is refused too.
  Text vectors need no re-encode: `text_vecs` is a registered buffer, so a strict `load_state_dict`
  restores the fitted MedCPT vectors into a zero-vector construction.
* `--emission-ablation` (A1) + `--ablation-seed` — four arms drawn **at the real cells**, so the kNN
  graph, cell count and density are the ground truth's and identical between arms: `A1c`
  Poisson(`mu_oracle`) — **model-free**; `A1b` ZINB(`mu_oracle`, model `theta`/`pi`); `A1a`
  ZINB(decode(`h1`)); `A1n` permutation null. Plus `I(mu_oracle)` and `I(decode(h1))` so the reader
  sees what went into each draw. `knn_mean_field` is a sparse row-stochastic product — the
  `values[idx].mean(axis=1)` form is a 2.4 GB intermediate on `deep_starmap`.

**Pre-registered before the run** in `reports/a1_preregistration.md`, committed ahead of the
command: the recovery statistic `R = (I(X) − I_model)/(I_real − I_model)` with bands
**>= 0.70 / <= 0.30** and a deliberate 40-point dead band; a four-row decision table read A1c first;
a **level guard** ([0.5, 2.0] on per-gene means); the stated asymmetry that `mu_oracle` is a kNN mean
and so every oracle arm is an **upper bound** (the self-check measures the size of that: I
0.5494 -> 0.9704, so *low is conclusive, high is permissive*); the stated contamination ordering
(A1c model-free and admissible; A1b uses only `theta`/`pi`; **A1a provisional** on the failed deep
fit); tier-1 as the **instrument control** whose failure voids every deep row; and §7's list of the
four reachable ways the test can fail — the check the previous gate did not get.

**Two rules into `specs/10` §4.2, both from these runs:**

* **§4.2f-i — an alarm that fires into the run whose own report says nothing about it.** §4.2f was a
  signal recorded in a checkpoint nobody read back. This is tighter: the alarm and the artifact are
  produced by the same process seconds apart, and the artifact selects against it. The rule: every
  report carries its run's alarm count, trajectory and **last-checked value**, and **refuses to
  print a verdict** when the last checked step is past the threshold.
* **§4.2k — a report describing an operation that did not happen.** Tier-1's `--top-k 32` on a
  28-gene panel kept all 28 and the report called it *"top 28 by Moran's I on the real side"*;
  `--match-density` asked for 4 187, the layout produced 4 073, no subsampling ran, and the report
  said *"matched ... 4073 kept"*. Neither is a wrong number; the defect is that a comparison built
  to make two datasets alike described an unselected panel and a top-3.1 % selection in the same
  words. The rule: a provenance line states what happened, not what was requested.

Also recorded, against §4.2's closing rule and as its second instance: the gate-2 mis-specification
is mine, and it is the same failure — a criterion whose reference could not disagree with it.

**Standing instruction:** every future `deep_starmap` run uses `--section section_4 --target-z 68.6`.
`section_2` at 30.8 sits between `section_1` (6.3) and `section_3` (43.4) and is boundary-adjacent;
R3 is a 20–35 % deficit there. A1 therefore re-measures its own anchors and does not inherit
`section_2`'s.

**B1 is not run until A1 reports.** If A1 exonerates the emission, B1's `ell` refit is the right next
step; if A1 convicts it, B1 tests the wrong thing. `+0.0729` is not quoted anywhere, including as a
negative result.

**Verification.** No torch in this container. `ruff` clean; the module imports; every command line
parses and all three refusal paths fire with the right message; `--self-check` **25/25**, now
covering `knn_mean_field` (shape, row-stochasticity preserving column means, non-negativity,
determinism, and that smoothing *raises* `I`). Nothing touching torch was executed: the checkpoint
load, the decode arms and the draws are unrun here.

---

## 2026-09-09 (c) — N1–N4 read out: the mechanism is located, and R12 is withdrawn from refutation

**A1b came back UNRESOLVED**, exactly as `a1_escalation_preregistration.md` §4 predicted. Per-seed
`R` = +0.29, +0.33, +0.28 — two seeds DOES NOT RECOVER, one UNINFORMATIVE, so the stability override
fires and the decision table returns **row 5 again**. N5 is the escalation. A1a is DOES NOT RECOVER
stably (−0.37/−0.39/−0.37); A1c RECOVERS stably (+1.95..+1.97), so row 1 stays refuted. The tier-1
control passes at three seeds. **§10 remains SUSPENDED.**

**N2 answered, and it locates the mechanism.** The decoder's `mu` head produces `sd(log mu)`
**0.6728** on tier-1 and **0.6725** on `deep_starmap` — 28 genes against 1017, three decimals — while
the tissue's model-free bracket is **0.72–0.94** (tier-1) and **1.10–1.37** (deep). Pre-registered
reading: **SUPPORTED on deep** (lower bound 1.54x the decoder, whole bracket 1.54–1.93x),
**UNTESTED STILL on tier-1** (0.99x–1.30x). The deep pass clears the 1.5x line by only 3 %, which is
acceptable here for a structural reason and not a convenient one: **N2's quantities carry no draw
randomness** — `mu_oracle` is a deterministic kNN mean, the real counts are fixed — so unlike A1b
there is no seed noise to straddle the line, and the criterion was placed on the pessimistic end of
the bracket by design. The estimator is verified on the real data too: the A1c check row recovers
`mu_oracle`'s own spread from Poisson draws of it, 0.7162 vs 0.7165 and 1.1005 vs 1.0994.

**Two facts that together name the failure.** (1) It is not the latent: on deep, `h1` is faithful
(`I(h1)` +0.3171 against real counts +0.3123) and decoding it gives **less** `mu` spread (0.6035)
than decoding the model's own over-smooth latent (0.6725). (2) It is not a shortage of variance: the
model's implied total is **1.4131** against the tissue's **1.3699** — slightly more. The variance is
present and ~90 % of it sits in `theta`/`pi`, which is spatially unstructured, instead of in `mu`,
which is not. The ZINB likelihood is near-indifferent to that split at the margin, which is R4's
trade with the mechanism measured rather than inferred.

**Consequence for §10, and it lowers its cost:** the two halves are one constrained move, not two
independent ones. The total variance is already right, so bounding `theta` forces the likelihood to
put it into `mu` — **half 1 may deliver half 2 for free**, and that is a one-hour tier-1 experiment
rather than a 3–4 day redesign. It still needs its own pre-registration and it does not lift the
suspension.

🚨 **R12's refutation is WITHDRAWN.** Last round I recorded *"R12 is refuted on the shipped arm"* on
the strength of the bounded share `Var(shape)/(Var(shape)+Var(log s))` reading 97.4 % / 93.3 %. That
statistic decomposes `Var(log mu)` into latent-driven and size-factor parts and **never touches the
sampling noise**. R12's claim is about counts — *"only 9–19 % of the emitted count variance survives
as between-cell structure"*. Computed model-free from N2, the count-level structured share is
**10.4 %** for the model against **>= 42.5 %** for the tissue on `deep_starmap`: **inside R12's own
range**. R12 stands, on an estimator that does not pass through the decoder. This is the second time
in three rounds I read a statistic by its name rather than its definition — gate 2 was the first —
and the rule it earns is: **a refutation must restate the claim in the claim's own terms before it
counts.**

**The deficit keeps shrinking as confounds come out**: 4.44x (`section_2`, z = 30.8) → 3.06x
(`section_4`, z = 68.6) → **2.71x** (`section_4` at its own z = 73.5, +0.1154 against +0.3123). None
of the three is a modelling change. 2.71x is the figure to quote, with the caveat that the trend
means more confounds may remain and the underlying deep fit is still the one whose alarm fired last.

**N1 is half-done and published a contradiction.** The report line now reads the `text_vecs` buffer
and correctly says "live, medcpt, 28/28 non-zero — from the checkpoint"; the **console** line in
`build_embeddings` still prints "ZERO VECTORS" because it fires at construction, before
`load_state_dict`. One run, two artifacts, opposite claims — §4.2k's fifth instance, all mine.

**Also recorded:** `chain_shipped_review.md` §6's prediction that the tissue's `sd(log mu)` would be
`>= 2` (3–5x the decoder) was **too large**; measured 1.5–1.9x. Direction and criterion hold on deep,
the magnitude did not. And tier-1's chain reproduced bitwise for a third time, across a code change —
a real determinism check on `--load-model`.

Next, in order: fix the console line; run N5 (pre-registered, minutes); pre-register and run the
reallocation test on tier-1 (~1 h) — which is §10 half 1 and a candidate for the gate that could lift
the suspension; add the pinned `bench3` `paper_*` metrics beside Moran's `I` before any repair is
judged, since the over-smoothing correction proved `I` alone can be raised by making the model worse;
then B1, with its expected value revised **down** — `ell` fixes the latent and the latent is not the
bottleneck. Full reasoning in `reports/a1_escalation_review.md`.

---

## 2026-09-09 (d) — M1, M2 and M3 built; two record items filed

**M1.** `build_embeddings` printed `ZERO VECTORS ... neither A3 arm` at construction, before
`load_state_dict` restores the `text_vecs` buffer, so the A1 runs' **console** asserted one arm while
the **report** they wrote asserted the other. Under `--load-model` it now says the vectors are about
to be replaced and declines to name an arm it is about to lose; the arm is printed after the load
from `describe_text_state`, which reads the tensor. §4.2k's fifth instance, closed.

**M2 — N5's split, pre-registered in `a1_escalation_preregistration.md` §2 before it was built.**
`A1b-t` is `NB(mu_oracle, theta)` with `pi` forced to zero; `A1b-p` takes **A1c's own Poisson
realisation** and applies the model's `pi` to it, so `A1c -> A1b-p` is an exact within-realisation
contrast and the difference is the dropout and nothing else. The decision rule is `n5_verdict()`
rather than prose — shares of the `A1c -> A1b` loss at the pre-registered 0.70/0.30 criteria, an
explicit *not decomposable* outcome, the additivity gap and the multiplicative prediction of `I(A1b)`
reported either way — because a pre-registered rule executed by hand is a rule that drifts. Its four
outcomes are asserted in `--self-check`. The report also carries `A1b-t`'s level ratio as an
instrument check that can fail: `theta` cannot move the mean, so anything away from 1.00x means that
arm is not what it claims.

**M3 — built and pre-registered; the run is one fit.** New `Config.decoder_theta_floor` (a lower
bound on `theta`, i.e. an upper bound on over-dispersion), kept **separate from `zinb_theta_min`**
because one is a numerical guard and the other an experimental constraint that enters the content
hash. Chosen over `decoder_theta_mode="moment_matched"` for a recorded reason: the moment-matched
estimator is **7.5x smaller** than the learned head, so it pulls dispersion *up* and deepens the
trade — candidate A's named failure. A floor can only move it down.

`--theta-floor` fits under it; `--report-theta` reads the learned `theta` at the real cells and, given
a candidate floor, the fraction of `(cell, gene)` pairs it would bind on — because *a floor that
binds on nothing is a null experiment, not a null result*, and that is invisible unless measured.
`scripts/t10_reallocation_table.py` joins the two chain sidecars and the two `t10_rescore_saved`
sidecars into **one table** and applies `m3_verdict()`.

`reports/m3_preregistration.md` fixes, before either arm runs: the floor's **selection rule** (the
median learned `theta` over `(cell, gene)` on the panel, a deterministic read of the existing
baseline checkpoint — no choice, and it binds on ~half by construction); four ordered outcomes with
**NULL EXPERIMENT checked first**, so a favourable result resting on a floor that did nothing cannot
be reported as a win; and §5's four reachable failures.

⚠️ **Outcome 1, "capacity-limited — the `mu` head, not the objective", is written as a result and not
a disappointment**, and §3a says why: it converts *"the ZINB objective allows the trade"* from a
mechanism into a **bound**, explains `sd(log mu)` coming back at 0.6728 / 0.6725 on 28 genes and 1017
and a faithful latent decoding to **less** spread (0.6035) than an over-smooth one, and makes every
loss-side candidate in §10 unnecessary to try. **It closes §10 rather than failing it.**

**M4 is a condition on M3, not a later step.** Every pinned `bench3` `paper_*` metric sits in the
same table as `I`, at matched density, and outcome 3 refuses the repair on a drop of more than 0.02
in any of them. The reason is measured: the model already beats real tissue on tier-1's `I` (+0.5134
against +0.4635) with a latent 1.28x smoother than the tissue's, so **`I` can be raised by making the
model worse** and a table reporting `I` alone would hide the most likely way this test fails.

**M5 (B1) is held**, as agreed: `ell` fixes the latent and the latent is not the bottleneck. It buys
trustworthy deep `theta`/`pi` and nothing else, at 3.3 h.

**Two record items filed.**

* **`specs/10` §4.2l — a refutation that answers a differently-defined question of the same name.**
  R12's claim is `Var(mu)/Var(y)`; it was refuted with `Var(shape)/(Var(shape)+Var(log s))`, which
  never touches the sampling noise. In R12's own terms the model reads **10.4 %** against the
  tissue's **>= 42.5 %** — inside R12's range. Refutation withdrawn. The rule: *a refutation must
  restate the claim in the claim's own terms before it counts* — write both as expressions and show
  they are the same expression. Second instance a round apart, after the gate whose two sides shared
  a decoder; both this project's.
* **The deficit trend now sits beside the number** in `chain_shipped_review.md`, as a table rather
  than a note: **4.44x -> 3.06x -> 2.71x**, three corrections, none of them a modelling change, each
  shrinking the gap. The reading travels with it: **the remaining number may still carry confounds
  nobody has found.**

**Verification.** No torch here. `ruff` clean on all three files; `t10_chain_diagnostic --self-check`
**40/40**, `t10_reallocation_table --self-check` **9/9** (including that the null-experiment guard is
checked *first*, and that `structured_share` reproduces both the 10.4 % and the 42.5 %). Nothing
touching torch ran: the new `Config` field, the decoder clamp, `--theta-floor`, `--report-theta` and
both N5 arms are unexecuted here.

---

## 2026-09-09 (e) — `decoder_theta_floor` made the default config unconstructible

**The defect.** The new field's range check was written into `_check_positive`'s `positive` dict
instead of being left out of it. That dict rejects zero, and the field's default **is** zero —
a sentinel meaning *no floor*. So `Config().replace(...)` raised
`ConfigError: Config.decoder_theta_floor=0.0 must be > 0`, and **all three real-data runs died in
`replace()` before opening a file**. Not a wrong number; a project that would not start.

**How the line got there.** `to_dict` is `dataclasses.asdict(self)` and needs no maintenance, but I
believed it was a hand-maintained dict because a grep had shown me `"zinb_theta_min": self.zinb_theta_min,`
in a listing I read as `to_dict`. It was `_check_positive`. My edit asserted the anchor occurred
exactly once — it did — and put the new line beside it, in the wrong method. The assertion was
satisfied and the placement was still wrong: **a unique anchor proves the edit is unambiguous, not
that it is in the right place.**

**Why nothing caught it.** `ruff`, `py_compile` and both `--self-check` suites pass, and **not one of
them constructs a `Config`**. `tests/test_config.py` *would* have caught it — `test_config_roundtrip`
and `test_config_gates_validate` both fail on the broken form — and it runs in this container in
0.3 s with `python3 -m pytest tests/test_config.py --noconftest` (the package's `config` module
imports without torch or pandas; `conftest.py` does not). **That channel existed all session and I
never used it.** It is now part of the loop, and it is the only test module collectable here — every
other one needs torch or pandas.

**The rule this earns**, and it is about my own instrument rather than the project's: *a self-check
that never constructs the object under test verifies the code path it exercises, not the change.*
Both `--self-check` suites assert numerics over numpy arrays; the change was a dataclass field, and
they could not have failed.

**The fix.** The line is out of `positive`. The range check stays written out — `>= 0`, and refused
at or above `zinb_theta_max`, where every `theta` would be pinned to one value and the run would
measure the floor rather than the model. The field's docstring now says the default is a **sentinel,
not a value**, that it therefore belongs in neither the positive nor the fraction registry, and that
putting it in the positive dict is exactly what was done first.

**The guard**, in `tests/test_config.py`:

* `test_shipped_defaults_validate_field_by_field` — asserts `Config().validate()`, that
  `Config().replace()` with no arguments returns an equal config, and then walks **every** field
  asserting its own declared default survives `replace()`, so the failure names the field. Verified
  to fail on a scratch copy with the defect reintroduced, and to pass on the fix.
* `test_decoder_theta_floor_sentinel_and_range` — 0 disables, 0.25 is accepted, negatives and a floor
  at `zinb_theta_max` are refused.

Verification now: `ruff` clean; `pytest tests/test_config.py --noconftest` **7 passed**;
`t10_chain_diagnostic --self-check` **40/40**; `t10_reallocation_table --self-check` **9/9**. Still
no torch here, so the decoder's use of the floor, `--theta-floor` and `--report-theta` remain
unexecuted — but the config now constructs, which is what the three runs needed.

---

## 2026-09-09 (f) — N5 read out; M3's criterion is wrong and M3 has not run

**M1 verified in production.** Both A1 reports read `live, text_emb_mode=medcpt, 28/28 gene rows
non-zero — from runs/chain/shipped_tier1.pt`, the console agrees, and the vacuous density and panel
lines fire correctly on tier-1. §4.2k's instances are closed on real data.

**N5, as pre-registered.** Tier-1: **over-dispersion**, `L_theta` 100.0 %, `L_pi` 0.0 %, additivity
gap 0.0000, multiplicative prediction exact, `A1b-t` level 1.00x. `A1b-t` equals `A1b` to four
decimals across all three seeds because `pi` is already ~0 there. **On a converged fit, the
emission's entire cost is over-dispersion** — M3's floor is aimed at the right term.
`deep_starmap`: **not decomposable**, gap 0.1100 against a 0.0200 criterion. The practical reading is
sharper than the verdict: as retention against `A1c`, **`theta` costs 45.7 % and `pi` costs 41.9 %**,
both together 65.6 %. Neither is a passenger.

⚠️ **The additivity clause was placed in the wrong space.** Losses in `I` cannot be additive — `I` is
a bounded ratio, and independent noise sources add *variances*, so retentions compose closer to
multiplicatively. The multiplicative prediction lands within **0.0142** of the measured `I(A1b)`,
inside the tolerance the additivity clause was given, and it under-predicts, which is the direction
the interaction term requires. Verdict unaffected; the criterion should have been on retention.

**🚩 A defect in M3's criteria, caught before M3 runs.** `m3_verdict`'s outcome 4 requires
`d I(counts) > 0`. **On tier-1 the model is already above the tissue** (+0.5134 against +0.4635), so
`I` rising is movement *away* from it, and M3 would report REALLOCATION WORKS for it. Removing
`theta` entirely from the model's own mean field is predicted to give ~**+0.78** on tier-1 — 1.68x
the tissue — so every point of the achievable range increases the distance. This is the
over-smoothing correction, in the record since round 11 and the reason M4 is a condition, not applied
to my own gate. **The numbers were on the table when I wrote the criterion**; M3 has not run.

Proposed fix, to be pre-registered before M3: **tier-1 is a mechanism test only** — success is
`sd(log mu)` rising from 0.6728 toward the tissue's bracket [0.7165, 0.9438] without overshooting,
`I` reported but not a criterion there — and the benefit statistic on both datasets becomes
**`d abs(I - I_real)`**, which is signed correctly on each and encodes *be like the tissue* rather
than a proxy for it.

**The picture: two defects that partly cancel.** Measured, all on deep: a correct latent with the
current emission gives **+0.0426** — *worse* than the model's own +0.1154 — because `decode(h1)` is
+0.3399 against `decode(h)`'s +0.7451 and the model's latent is **2.45x smoother** than the tissue's;
while removing the emission noise from the model's own mean field is predicted to **overshoot** at
+0.41 against +0.3123. Fix either alone and it gets worse in one direction or the other. On tier-1
the two errors cancel to within 11 %, **which is why tier-1 looked healthy for three rounds**.

🚩 **This corrects my assessment of B1, in the user's favour.** I read *"`decode(h1)` gives less `mu`
spread than `decode(h)`"* — a statement about `mu`'s **spread** — as settling that the latent is not
the bottleneck. The latent's **smoothness** is a different quantity, wrong by 2.45x on deep, and
`ell` is the only handle on it. **B1 is not a 3.3 h purchase of trustworthy `theta`/`pi`; it is the
only test of the second of two coupled defects.** Its priority goes up.

**The floor is 2.599** (the median `theta` at the real cells; percentiles 1/25/50/75/99 =
0.369/1.394/2.599/5.165/13.65), binding on **50 %** of pairs by construction, so the null-experiment
guard passes without a further read. ⚠️ It is a **modest dose** — the floored half moves up ~1.9x and
the other half is untouched — so if `sd(log mu)` does not move, *"the dose was too small"* is a live
explanation alongside *"capacity-limited"*. Proposed: one higher floor at the **75th percentile
(5.165)** before outcome 1 may be declared. Outcome 1 is a strong claim and should not be reachable
from one weak intervention.

**The missing measurement, one draw:** `I(counts ~ Poisson(mu_gen))` — the model's own mean field
with all emission noise removed, i.e. the **ceiling for any emission repair on the model as it
actually is**. Nothing in the campaign has it; every arm is either on `mu_oracle` or keeps the
emission. Pre-registered prediction before measuring: **+0.7775** tier-1 and **+0.4116** deep, i.e.
**1.68x and 1.32x the tissue**. At or above those, no emission repair reaches the tissue without the
latent being fixed too; at or below the tissue on deep, the transfer is invalid and the emission
repair stands alone.

**Order proposed:** P1 the `Poisson(mu_gen)` arm (minutes) — it tells you what M3 is aiming at; P2
amend M3's pre-registration; P3 M3 at floor 2.599; P4 M3 at 5.165 if the dose condition fires; P5 B1.
**P1 and P2 come before P3**, both cheap, and both change what P3 means. Full reasoning in
`reports/n5_and_m3_review.md`. **§10 stays suspended** — N5 answered the mechanism question it was
asked and did not return a readable answer to the decision table, which still reads row 5 for `A1b`.

---

## 2026-09-09 (g) — P1 and P2 built; the B1 reversal and the cancelling-defects table filed

**P1 — the emission-free ceiling.** `4p. counts ~ Poisson(mu) — emission noise removed`, a new chain
stage. It belongs at the **generated** cells, not in the ablation: `mu` lives there, and every A1 arm
is at the real ones. Drawn on the panel only (Moran's `I` is per column, so selecting before or after
the draw is identical) and at the run seed like stage 4, so the two differ in the emission and in
nothing else. It is the ceiling any repair to `theta`/`pi` can reach **on the model as it actually
is** — the first such bound in the campaign; every previous arm either used the tissue's mean field
or kept the emission. Predicted before measuring, in `n5_and_m3_review.md` §7: **+0.7775** tier-1 and
**+0.4116** deep, i.e. 1.68x and 1.32x the tissue.

**The cancelling-defects explanation now prints beside the numbers**, not only in a report the reader
may not have — `specs/10` §4.2f-i's lesson applied to an explanation rather than to an alarm. When a
run comes back with `I(model counts)` above its tissue reference, `cancelling_defects_block` emits the
latent's smoothness ratio, `mu`'s spread against the tissue's model-free bracket, and the emission's
direction, from the run's own rows so it cannot disagree with the table above it. Stage 4p's ceiling
prints either way. Both branches are asserted in `--self-check`, because report-generating code that
silently produces nothing is what §4.2k is about.

**P2 — M3's pre-registration amended before either arm ran**, with an AMENDED banner stating that
both corrections come from numbers already on the table when the first version was written.

* **`I` rising is not the goal, and on tier-1 it is the wrong direction.** The old outcome 4 would
  have reported REALLOCATION WORKS for a run that moved the model *away* from the tissue. The benefit
  statistic is now `d abs(I - I_real)`; **tier-1 runs as `--role mechanism`, in which it is reported
  and is not a criterion** (§3c). Success on tier-1 is `sd(log mu)` rising from 0.6728 toward the
  tissue's bracket [0.7165, 0.9438] without passing its upper bound — a new **OVERSHOT** outcome,
  since more spread than the tissue has is not a repair.
* **The dose condition is a precondition, in the outcome table** (§3d). CAPACITY-LIMITED closes §10,
  so it is not declarable from one modest intervention: a median floor binds on 50 % of pairs and
  moves them up ~1.9x while leaving the rest untouched, and a null there has two explanations. Until
  the 75th-percentile floor (5.165) has also run, the verdict is **UNDER-DOSED**, which is not a
  result and closes nothing. `m3_verdict` refuses CAPACITY-LIMITED without `--escalated`.

The floor is **2.599** and binds on 50 % by construction, so the null-experiment guard passes without
a further read. Step 5 of §6 is the escalation command, run only if step 4 reads UNDER-DOSED.

**🔄 The B1 reversal, recorded as a reversal.** *What I read*: a faithful latent decodes to **less**
`mu` spread (0.6035) than the model's over-smooth one (0.6725), so the latent is not the bottleneck.
*What that covers*: `mu`'s **spread** — still true, the `mu` head is not spread-limited by its latent.
*What I now read*: the latent's **smoothness** is a different quantity — `I(h)` +0.7782 against the
tissue's +0.3171, **2.45x** on deep and 1.28x on tier-1 — and nothing in the spread measurement bears
on it. *The correction*: B1 is not a 3.3 h purchase of trustworthy `theta`/`pi`; it is the only
handle on the second of two coupled defects, and its priority goes **up**. Filed both where B1 was
first costed (`chain_shipped_review.md` §9) and where I revised it down
(`a1_escalation_review.md` §6.5). The error was conflating two properties of one object, measured by
different statistics, and reading one measurement as if it covered both.

**The cancelling-defects table is now in `chain_shipped_review.md` §4a**, where a reader meets
tier-1's numbers rather than only in the review that found it. It states the whole of tier-1's
apparent health: the latent is 1.28x too smooth (pushes `I` up), the emission adds 40.7 % of
retention in unstructured noise (pushes `I` down), and **the two cancel to within 11 %**. Both
directions are measured, not inferred — a faithful latent drops `I` to +0.4053, removing the
emission's noise raises it to a predicted +0.7775 — so **repairing either alone moves `I` away from
the tissue.**

**Verification.** No torch. `ruff` clean; `pytest tests/test_config.py --noconftest` 7 passed;
`t10_chain_diagnostic --self-check` **47/47** (7 new, covering both branches of the explanation block
and its empty case); `t10_reallocation_table --self-check` **13/13**, including that the same arm
reads MECHANISM CONFIRMED in one role and NO BENEFIT in the other, which is the correction P2 exists
for. Stage 4p and the amended verdict are unexecuted here.

---

## 2026-09-09 (h) — the ceiling bounds the emission programme, and exposes a transform mismatch

**The ceiling (stage 4p), against the prediction committed in `n5_and_m3_review.md` §7.**

| | model now | ceiling | tissue | predicted ceiling |
|---|---|---|---|---|
| tier-1 | +0.5134 (1.11x) | **+0.8358 (1.80x)** | +0.4635 | +0.7775 (1.68x) |
| `deep_starmap` | +0.1154 (0.37x) | **+0.2594 (0.83x)** | +0.3123 | +0.4116 (1.32x) |

Tier-1 was close and the conclusion is unchanged. **Deep was wrong in the direction that changes the
conclusion** — predicted an overshoot, measured an undershoot. Both pre-registered branches fired,
one per dataset: on tier-1 the coupled reading is **established** (an emission repair alone cannot
land it on the tissue; it goes to 1.80x), and on deep the emission does **not** overshoot — it closes
**73.1 %** of the deficit and leaves 17 % to the mean field.

**This bounds §10's whole programme without a redesign.** A *perfect* emission repair gives a
**regression** on tier-1 (1.11x -> 1.80x) and **83 %** of the tissue on deep. §10 stays suspended —
a bound is not the readable gate answer it waits on — but the bound belongs beside the costing.

**🚩 The tier-1 run contradicted itself, and that is how the defect was found.** `4p` is
`Poisson(mu)`: independent noise dilutes autocorrelation and cannot create it, so `I(4p) <= I(3)`
must hold. It does not — **+0.8358 against +0.7920**. Confirmed in the code afterwards: **every
mean-field and latent stage is raw; every count stage is rank-normalised.** Rank-normalising a
heavy-tailed field raises its Moran's I, and `mu` is log-normal, so **every `counts / mu` or
`counts / latent` retention in the campaign is overstated** by an unmeasured amount — including the
"three numbers" table (64.1 %, 14.8 %, and the tissue's 74.1 % and 98.5 %) and
`a1_escalation_review.md` §5's table, now flagged in place.

**What survives**, which is most of the record: everything comparing a count stage with a count
stage — the ceiling itself, every A1 arm and every `R`, N5's shares, the deep 2x2, and the
cancelling-defects table (raw-vs-raw latents, and `sd(log mu)` is not Moran's I).

**The rule:** *a ratio is only a ratio if both sides were computed under the same transform.* The
stage names disclosed it — ranked rows say "(rank-normalised)", raw rows say nothing — but a label is
not a guard. The invariant that catches it is `I(4p) <= I(3)`, it is free, and it should be asserted
at runtime rather than noticed by a reader.

**The deep 2x2, now fully measured, every cell count-vs-count** (tissue +0.3123): model `mu` +
model emission **0.37x**; `mu_oracle` + model emission 0.55x; model `mu` + Poisson **0.83x**;
`mu_oracle` + Poisson 1.60x. So the **emission is the larger lever on deep** (closes 73 % against
`mu`'s 29 %) and **the two together over-correct**. The `mu_oracle` column is optimistic — a kNN mean
creates autocorrelation — so the left column is the trustworthy one.

**The narrow-`mu` mechanism, quantified end-to-end for the first time**, and with neither side
passing through the decoder: the same Poisson draw on the model's mean field against the tissue's
gives **0.917** on tier-1 and **0.518** on deep, tracking `sd(log mu)` against the tissue's bracket
exactly (0.6728 just below [0.7165, 0.9438]; 0.6725 far below [1.0994, 1.3699]). Caveat: the two
fields differ in spatial pattern as well as spread, so 0.518 is an upper bound on what spread alone
costs.

**M3's value goes UP.** The narrow `mu` costs 48 % of the achievable `I` on deep *even with a perfect
emission*, so an intervention that bounds `theta` **and** widens `mu` attacks both measured defects at
once — and M3 is the test of whether one lever does both. Its verdict is also **unaffected** by the
transform defect: `m3_verdict` reads `sd(log mu)`, stage 4, `REF real counts`, their distance, the
`CV²` share and the `paper_*` metrics, every one count-vs-count or not an `I` ratio.

**Order proposed:** Q1 fix the transform mismatch and assert the invariant (free); Q2 run M3 at 2.599
in the mechanism role (~1 h); Q3 B1 (~3.3 h); Q4 M3 on deep in the benefit role. Q1 before Q2 only
because it is free and Q2's report would otherwise print contaminated retention rows beside a clean
verdict.

**Where I was wrong.** The deep prediction transferred `A1c`'s Poisson retention at `mu_oracle`
(0.5525) onto `mu_gen`; the measured retention there is **0.3481**. Retention is not transferable
across mean fields of different spread. Part of the error was the transform mismatch itself — the
transferred factor divided a ranked count stage by a raw `mu` stage. **The failed prediction measured
the mechanism more directly than a correct one would have**: had the transfer held, the 0.518 would
never have been computed.

---

## 2026-09-10 — Q1 built, and Q1.5 pre-registered: the campaign has been measuring a median, the paper is scored on a correlation

**The gap, raised by the user and correct.** Every number in this programme is a **median** per-gene
Moran's I. `paper_morans_pearson` is a **correlation across genes**, and a model can match the
tissue's median exactly while getting every individual gene wrong. v25 scores **0.557** against
SpatialZ's **0.932** and nothing in the chain work has touched it. Before 4.3 h on Q2/Q3, the
question is whether the quantity the chain moves is coupled to the quantity the paper is judged on.

**The definition was read from the source, not recalled** — `specs/10` §4.2l was earned one round ago
for exactly that failure. `evaluate_paper.py::_agreement` (line 102) and
`spatial_autocorrelation_metrics` (line 120), with `pR, gR = _rank_normalize(...)` at line 670 and
`SPATIAL_K = 10`. **What matches the chain exactly**: ranked counts on both sides, `k = 10` equal to
`Config.metric_knn_k`, each side on its **own** spatial graph (the evaluator calls it
"alignment-free", which is how the chain measures stage 4 at the generated cells and `REF real
counts` at the real ones), NaN genes dropped pairwise. **What does not**: bench3 scores **all** shared
genes and medians over sections **2/4/6**; a chain run is one section and possibly a panel. So the
number is **the statistic, not the score** — comparable between stages of one run, never to a
published `paper_*` value. Fixed in `reports/q15_preregistration.md` §2 before anything was computed.

🚩 **A finding from reading the source.** `_agreement` emits five keys and the project's `METRICS`
tuple scores one. **`paper_morans_mae` is emitted, is documented in the evaluator's own docstring as
the metric that catches over-smoothing** — *"a blurred reconstruction inflates Moran's I ... while
keeping the gene ranking intact"* — **and is not in the scored set.** That is round 11's
over-smoothing finding, written into the benchmark before this campaign began, with its detector
named and unscored. Three statistics, and the campaign has measured only the first: the median pair
(emitted, unscored), `mae` (emitted, unscored), `pearson` (scored). All three are now reported
together.

**Pre-registered readings**, fixed before computing: **(a)** `d r = r(4p) - r(4)`, bands `>= +0.15`
MOVES IT / `<= +0.05` DOES NOT; **(b)** `r(4p)` absolute, bands `>= 0.85` / `<= 0.70` — the ceiling
argument applied to the scored statistic, and the more decisive of the two. `mae` and the median pair
are reported, not criteria. §6 lists four ways it can fail, including the one that would waste it:
if `r(4)` is already high while the score is 0.557, the single-section panel-restricted
reconstruction is not measuring what the benchmark measures — checkable from the numbers themselves.

**Q1 — the transform mismatch, fixed structurally rather than by a note.** `summarise` now takes the
**untransformed** array and ranks it itself, returning `median_I_raw` **and** `median_I_rank` for
every stage, with `primary` naming which one `median_I` carries so existing artifacts keep their
figures. Every call site was changed to stop pre-ranking. `rank_normalize` now appears only inside
`summarise`, in the self-check, and in panel selection. The report prints both columns and says a
ratio must take both sides from the same one.

**The invariant is in the code.** `I_rank(4p) <= I_rank(3) + 0.02` — 4p is an independent draw from
stage 3's field and independent noise cannot raise Moran's I. A violation prints to stderr **and**
opens the report with `🚨 INVARIANT VIOLATED`, above which no stage-to-stage ratio may be read. This
is the check that would have caught the mismatch instead of a reader noticing +0.8358 above +0.7920.

**Q1.5 built.** `morans_agreement` reproduces `_agreement`'s construction; `summarise` now carries
each stage's per-gene ranked `I` vector, so the panel figures are free. On `deep_starmap` the panel
is the top 3.1 % by the real section's own `I`, which compresses its `I` range and attenuates the
correlation, so an **all-1017-gene** pass also runs and governs — via `morans_i_ranked_blocked`,
which is exactly `morans_i(xy, rank_normalize(v), k)` in gene blocks, because the direct call forms
an `(N, k, G)` array that is **24 GB** at 29 544 cells and 1017 genes.

**Verification.** No torch. `ruff` clean; `pytest tests/test_config.py --noconftest` 7 passed;
`--self-check` **61/61**, 14 of them new — including that the blocked estimator equals the direct one
exactly, that a constant offset keeps `pearson` at 1.0 while `mae` rises (the ranking/level split
Q1.5 exists to expose), that NaN genes drop pairwise as the evaluator does, and that `summarise`
refuses an unknown `primary`. The agreement numbers themselves are unrun here.

**Q2 and Q3 do not start until Q1.5 is read.**
