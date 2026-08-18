# T08 — Metric-aware LOSO losses

Part of [PROGRESS.md](../PROGRESS.md).

### T08 — Metric-aware losses via internal leave-one-section-out (2026-08-18)

Built `spatialcpav25_gen/losses/metric_aware.py` (the three loss families and the statistics they
are built on), `spatialcpav25_gen/train/loso.py` (the internal LOSO schedule and the reconstruction
the losses are computed on), `TrainingVolume.principal_axis` (SPEC_QUESTIONS C10), the trainer's
metric-aware block, and `scripts/t08_metric_report.py`. 22 tests, 18 of them in the fast suite.

**Both headline criteria are red, and the two decisions they force are taken rather than deferred:
the terms ship at 0, and the paper's gene-gene covariance claim is downgraded to a mechanism
claim.** The rest of this entry is what was measured and why the decisions are what they are.

#### What the acceptance tests say

| test | criterion | measured | status |
|---|---|---|---|
| `test_morans_matches_reference` | 1e-5 vs an independent dense reference | max abs diff **4.4e-16** (float64) and **5.3e-8** (float32), against I in [0.57, 0.75] | pass |
| `test_gearys_matches_reference` | 1e-5, likewise | same order; C max **0.415** (a flat statistic would read 1) and I/C correlate **−0.977** across genes | pass |
| `test_morans_differentiable` | gradient finite and non-zero, graph carries none | passes; `W.grad_fn is None` by construction | pass |
| `test_soft_profile_approaches_hard` | relative error **< 2 %** at `sigma = 0.1 x` bin width | **max 1.24 %**, mean 0.30 % over occupied bins; at the working `sigma` the same profile is measurably smoother | pass |
| `test_profile_axis_stable` | identical across epochs | bitwise identical over 5 reads; recomputing per LOSO fold **does** move it, which is the control | pass |
| `test_loso_excludes_from_retrieval` | structural | no donor and no batch row from the hidden section, its layout target excluded, and the same call **without** `hide` does reach it | pass |
| `test_metric_aware_rejects_heldout` | `TypeError` | raised for `HeldOutSections` **and** for a plain `Volume`, at all three doors | pass |
| `test_metric_losses_improve_metrics` | all three families better with the terms on | **RED** — see below | strict xfail |
| `test_metric_losses_close_the_covariance_loss` | below the baseline on **both** holdouts | **RED** — see below | strict xfail |

#### Ablation A2: the terms cost at T06's budget and pay at twice it

Four arms, alternating holdout, everything else T06's (`scripts/t08_metric_report.py`). The two
1200-step arms are the comparison `specs/08` asks for; the 2400-step pair is the one open risk R4 is
actually about.

| statistic | off@1200 | on@1200 | off@2400 | on@2400 |
|---|---|---|---|---|
| reconstruction (nats/pair) | 1.5901 | 1.6843 | **1.5703** | 1.5885 |
| gene-gene Frobenius (baseline 7.70-8.00, ceiling 5.601) | 9.000 | 11.154 | 9.049 | **8.489** |
| Moran's MAE, held-in LOSO reconstruction | 0.0287 | 0.0408 | 0.0339 | **0.0279** |
| marker-depth r | 0.978 | 0.967 | 0.983 | **0.990** |
| cell-type localization | **0.967** | 0.962 | 0.958 | 0.957 |
| detection-rate MAD (T06 criterion < 0.05) | 0.0185 | 0.0314 | 0.0218 | 0.0227 |
| mean-variance slope (real 1.741, criterion 15 %) | 1.762 | 1.734 | 1.773 | **1.722** |

At **1200** steps the terms lose on every statistic they are made of. The control that says why:
a **schedule-only** arm — internal LOSO hiding a section from the epoch's batches, retrieval and
layout targets, with nothing charged for it — scores Moran's MAE 0.0340, depth r 0.9719,
localization 0.9696, i.e. between the two arms and much nearer `off`. So the cost is the loss
terms', not the hidden section's, and `specs/08`'s instruction ("report it — a loss that does not
improve its own metric is either mis-specified or mis-weighted") applies to the terms.

At **2400** steps the ordering reverses on four of six, including both metrics the terms are named
for and the covariance. The reading that fits all of it: the terms add a constraint and converge
more slowly, and 1200 steps is not a neutral reference point — it is where T06 stopped *because the
unconstrained arm starts degrading there*. That is visible in this table too: `off` buys 0.0198
nats of likelihood between 1200 and 2400 steps and pays for it in covariance (9.000 → 9.049),
Moran's MAE (0.0287 → 0.0339) and localization (0.967 → 0.958), which is R4 in miniature; `on`
improves on likelihood *and* covariance *and* Moran's *and* depth over the same interval.

Unlike T07's SEFL terms, these break **no** T06 acceptance criterion at either budget (detection
MAD 0.0314 / 0.0227 < 0.05; mean-variance slope error 0.4 % / 1.1 % < 15 %; detection r > 0.95). The
decision to ship them off is therefore about the measured cost at the documented budget, not about
breakage, and it is reversible by one config override — which is exactly what `specs/10`'s **A2**
now is: an *addition* experiment, to be run **at two budgets**, reporting the six target metrics.

**Amended 2026-08-18, and it is the right correction: a 0 chosen at 1200 steps is calibrated to an
undertrained model, so it is no longer hardcoded.** The reversal above says the budget and the
weights interact, and 1200 steps is not a neutral reference point — it is where T06 stopped
*because the arm without these terms starts degrading there*, which is R4's own symptom. Leaving
the weights at a flat 0 would bake that symptom into the shipped model. So `train_steps` becomes a
`Config` field and, with the three weights, **one joint selection gate** in `specs/09` §3's
`select_config`, scored per dataset on internal LOSO over training sections. The four cells of
`{1x, 2x} x {off, spec weights}` are scored **together**, because coordinate descent visiting them
separately would score `(1200, on)` against a `(1200, off)` incumbent, find it worse — which it is
— and never reach `(2400, on)`, the cell that wins on four of six statistics. The 0 in `Config` is
now where that search starts, not what it concludes; `specs/11`'s matrix and both "Do NOT" lists
carry the rule.

#### The R4 verdict: the covariance claim is a mechanism claim

`specs/08` requires the model's Frobenius covariance error to fall below the independent-donor
baseline's on the default holdout **and** hold at `consecutive-3`. With the terms on:

| regime | model | independent-donor baseline | achievable ceiling | T06, terms off |
|---|---|---|---|---|
| `alternating` | **11.022** | 7.732 | 5.601 | 9.316 |
| `consecutive-3` | **13.391** | 11.383 | 5.513 | 17.7 |

Both above their baseline, so the criterion fails on both, and per `specs/08`'s own instruction the
claim is **downgraded**: the paper may say that per-gene independent draws destroy within-cell
covariance and a shared latent cannot — the mechanism, established at T06 with the donors held
fixed — and may not say that this method preserves gene-gene covariance better than the competing
method. Written into `specs/10` §2 with the numbers. The terms do move the wide-gap arm a long way
(17.7 → 13.4) and, at 2400 steps on the default holdout, take the model to **8.489** against a 7.948
baseline — the closest this project has come, and still a loss.

#### Four things `specs/08` does not specify, each of which had a wrong answer that trains

These were found by measurement, not by reading, and each is a term whose minimiser was not the one
it was written for. Full table in SPEC_QUESTIONS **C24**; the summary is that the naive reading of
"compare the generated section with the real one" diverges three different ways.

| what looked right | what it does | measured |
|---|---|---|
| compare the decoder's **mean field** with the real **draw** | biases every family the same way; nothing can train it away | marker-depth r 0.985 → 0.642, Frobenius 9.3 → **26.0** |
| a **straight-through** count draw everywhere | `log1p` is concave; its slope at a zero draw is 1 against a far smaller true derivative, so every step overstates what raising the mean buys | generated mean normalised count 4 → **761** against a real 6.8 |
| normalise each side by **its own** row sum | every gene then shares one denominator, and the panel's autocorrelation can be raised at once by making that denominator spatially structured | autocorr term 0.33 → 0.16 while the largest value pinned at `log1p(reference)`; parameter gradient 6 → **1.6e8** |
| read `lambda_c` for the per-type histogram **at the cells** | a field that classifies each cell correctly averages to the bin's label fractions, so unbounded confidence is free | `max_c lambda_c / sum_c` 0.62 → **0.9975**, taking the shared anatomical field with it |

The shipped shape, one line per family: **profiles** take the mean field on a linear
library-normalised scale (a bin mean of the mean is an unbiased estimate of the bin mean of a draw);
**autocorrelation** takes the mean field plus the draw's *analytic* variance in Moran's and Geary's
denominators (`draw_mean_variance`), so the model's statistic estimates the same quantity the real
side measures rather than its noiseless limit; **distribution** takes a reparameterised
moment-matched surrogate in the `log1p` PCA basis it is stated in. Both sides are divided by the
**real** cell's library size, the generated side uses the real cell's size factor, the per-type
composition is read at uniform slab points, and markers are chosen among genes detected in ≥ 5 % of
real cells.

#### Deviations from `specs/08`, all recorded in SPEC_QUESTIONS

* **C20** — `specs/08` §3 names `geomloss`. Using it would make a paper number depend on whether an
  optional dependency is installed; the divergence is always the in-repo debiased Sinkhorn T07 wrote,
  with `Config.metric_distribution_kind` selecting the spec's MMD fallback.
* **C21** — `loss_profile`'s stated signature has nowhere to put the cell types its own third bullet
  needs. Kept verbatim, plus keyword-only `types_gen` / `types_real` / `types_coords`, and `bounds`
  on both profile functions so two profiles land on the same ruler.
* **C22** — "reconstruct at the hidden section's true plane" is read as the expression pathway at the
  section's own cells: a sampled layout has no gradient to the intensity head and would add the
  sampler's draw noise to all of §1-§3. The layout still enters, differentiably, through the
  per-type histogram. **T08's terms do not train the point pattern**, and nothing here claims they do.
* **C23** — eight constants become documented `Config` fields; the Sinkhorn iteration count is
  *shared* with T07's rather than duplicated, because it is a property of the solver.
* **C24** — the model-versus-measurement question above.
* **Budget**: the arms run at T06's `TRAIN_STEPS = 1200` rather than `specs/08`'s 1000, so that the
  `off` arm is the model already on the record and the R4 comparison is against a measurement rather
  than a re-measurement. The 2400-step pair is an addition, not a substitution.
* The distribution term is normalised **per PC dimension**. Its raw magnitude scales with
  `expr_pca_dim` — around 10 at 16 components against a reconstruction NLL of 1.6 — and
  "do not weight these losses above reconstruction" is a statement about the *weighted* term, so the
  term has to be on a stated scale before a weight can mean anything.

#### Cost, and the guard on `specs/08`'s last "Do NOT"

The block runs every `Config.loso_every_k_steps` (4) on ≤ `loso_max_cells` cells and costs **+57 %**
per step on the fixture (365 s → 572 s per 1200 steps; +61 % at 2400). The
metric/reconstruction ratio is logged at every block and warned above
`Config.metric_dominance_ratio_warn`: median **0.187**, max **0.506** over 300 blocks at the spec's
weights. The denominator is the reconstruction term alone — an earlier version divided by
reconstruction *plus* layout and read 0.46 while the metric terms were sitting at twice
reconstruction, which is precisely the state the guard exists to catch.
