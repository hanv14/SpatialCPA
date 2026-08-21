# T10 pilot — STARmap tier 1, run 2026-08-20

**Status: RESUMED and run to a complete tier-1 v25 prediction** after the schema decision.
The headline number the 3-vs-5 dataset decision rests on is measured (§5). Two environmental
blockers remain and are not resolvable in this container (§1, §2).

*Original run (2026-08-20, first pass): halted at step 6 of 12 by the schema conflict in §4. That
is now decided and implemented, and §4 records the resolution.*

The two blockers the run was ordered to surface first both hit walls in the first minute, which is
the ordering working as intended.

`evaluate_paper.py` SHA-256 asserted **before and after**: `7362669200bb…8992`, unchanged.
bench3 footprint: `config.py` +32/−0 (one appended `METHODS` entry, `METHOD_ORDER` untouched) and
one new wrapper file. Nothing else.

---

## What ran, and what it establishes

| step | result |
|---|---|
| 1 · evaluator SHA | ✅ matches the pin, before and after |
| 2 · environment + tier-1 build | ✅ `pip install -e .[dev]`; STARmap builds **exactly to protocol** — 32 845 cells → trim 3 867 (11.8 %) → 77 planes → 7 × 11, sections at z = 19/30/41/52/63/74/85, 4 073–4 187 cells each, all three markers resolved. Reproduces the README's table row for row |
| 3 · `selftest` on the real build | ✅ **all 12 checks pass**, 2 m 53 s. oracle 0.982–1.000, `spatial_scramble` collapses on every spatial metric while holding 1.000 on the distributional ones |
| 3b · `flanking_copy` / `oracle` referents | ✅ **measured — the tier-1 referents now exist** (§3) |
| 4 · wrapper + `METHODS` entry | ✅ written; `--require-config`, stale-selection and bare-config paths all exercised |
| 5 · comparator re-runs | ❌ **BLOCKED — no conda** (§1) |
| 6 · v25 fit | ✅ **MEASURED — 1200 steps in 1712.7 s (28.5 min), 1709 MB peak, CPU** (§5) |
| 6b · end-to-end prediction + scoring | ✅ 9 570 cells written, scored on the pinned evaluator (§5) |
| 7–10 · headline, statistics, E5, A1, C1/C2 | not reached — they need a *selected* config, and selection never ran (§5) |
| 11 · re-sectioning geometry | not reached |
| 12 · `deep_starmap` gene coverage | ⚠️ **partly blocked — no data** (§2); the casing half is measured |

---

## 1. BLOCKER — the comparators cannot be run here (no conda)

`which conda` → not found. `run_benchmark.run_single` invokes every wrapper as
`conda run -n <env> python <wrapper>`, and the four environments (`bench_spatialcpa`,
`bench_spatialz`, `bench_feast`, `bench_isost`) do not exist. The v1 YAMLs are present
(`benchmark-pbya/envs/`), so the environments are *creatable*, but not in this container.

**Consequence.** SpatialZ, FEAST, isoST and v20 cannot be re-scored here, and §13.1 established
that re-scoring them is not optional: the recovered results tree is evaluator-heterogeneous, with
SpatialZ's STARmap rows carrying **0/3** of the newer evaluator's columns against v20's 3/3.

**Not fatal to the comparison**, because of §3: the two probes are method-independent and needed no
conda, so tier 1 already has its floor and its ceiling on the pinned instrument.

## 2. BLOCKER — `deep_starmap` is not in this repository

No raw, no processed, nothing under any name (`find` across the tree returns nothing). Only the
STARmap volume ships here. So E1's 1 017-symbol coverage question cannot be answered.

**The casing half is measured and confirmed.** `resources/gene_meta.parquet` is 1 138 mouse-cased
symbols (`summary_sources`: native 148 / ortholog 896 / none 94 — matching the documented 148/1138).
Against `deep_starmap`'s known uppercase marker and layer genes:

| match | result |
|---|---|
| exact, against the mouse-cased table | **0 / 6** |
| after case folding | **6 / 6** |

So case folding is **mandatory and, on the symbols testable here, sufficient**. Whether the full
1 017-gene panel is covered remains unmeasurable, and a shortfall would need a `mygene` build that
is 403'd in this container (C14).

## 3. ✅ The tier-1 referents are measured — and they confirm the boundary caveat independently

`selftest`'s probes go through the real prediction contract and the pinned evaluator, so they are
the ceiling (`oracle`) and the flanking-section floor (`flanking_copy` = `run_nearest_copy`) that
`specs/10` §2 requires beside every method number. **Medians over the three held-out sections:**

| metric | `oracle` | `flanking_copy` |
|---|---|---|
| `celltype_localization` | 0.9808 | **0.7765** |
| `marker_field_r` | 0.9997 | **0.8857** |
| `marker_depth_r` | 1.0000 | **0.9794** |
| `morans_pearson` | 1.0000 | **0.9836** |
| `gearys_pearson` | 1.0000 | **0.9840** |
| `gene_mean_spearman` | 1.0000 | **0.9863** |

**And the per-section split settles the boundary question with a method-independent probe.**
`flanking_copy` scores `section_2` **worst on 6 of 6 metrics**, by wide margins, while `oracle` —
the real cells — shows no such pattern and sits at ceiling everywhere:

| `flanking_copy` | section_2 | section_4 | section_6 |
|---|---|---|---|
| `celltype_localization` | **0.7008** | 0.7765 | 0.7868 |
| `marker_depth_r` | **0.8713** | 0.9820 | 0.9794 |
| `morans_pearson` | **0.9517** | 0.9913 | 0.9836 |
| `marker_field_r` | **0.8470** | 0.8857 | 0.8873 |

This is the cleanest available evidence for `specs/10` §1's caveat: the `section_2` deficit is **not
a property of any model**. A probe with no model at all shows it, because `section_2`'s flanking
evidence is one-sided — `section_1` is the stack's first section. It is a property of where the
paper protocol put its held-out sections.

**It also sharpens C1.** `flanking_copy`'s localization drops **0.086** from `section_4` to
`section_2` — larger than SpatialZ's entire **0.061** tier-1 lead over v20. So a method's pooled
localization score is materially sensitive to how it handles one-sided evidence, and C1's
per-section split (`specs/10` §11.1) is not a refinement but the whole measurement.

## 4. ⛔ SUBSTANTIVE BLOCKER — the protocol dataset fails T01's schema validation

A v25 fit on the real tier-1 input aborts in `load_volume`:

```
SchemaError: Section.coords for section_id='section_1' contains 34 duplicate
coordinate row(s) (Config.coord_key='spatial'); coincident cells corrupt every
kNN-graph metric downstream
```

**Diagnosed, not guessed.** bench3 flattens each 11-plane slab to 2-D — every cell keeps its real
`(x, y)` and takes the slab's centre z, with the original plane preserved in `obs['z_plane']`. Two
cells at the same `(x, y)` in *different* z-planes of one slab therefore become exactly coincident.
Confirmed: the colliding cells in `section_1` span z-planes 14–21.

Measured across the tier-1 build:

| section | cells | duplicate rows | % |
|---|---|---|---|
| section_1 | 4 073 | 34 | 0.83 |
| section_2 | 4 187 | 27 | 0.64 |
| section_3 | 4 169 | 16 | 0.38 |
| section_4 | 4 102 | 13 | 0.32 |
| section_5 | 4 110 | 16 | 0.39 |
| section_6 | 4 162 | 22 | 0.53 |
| section_7 | 4 175 | 15 | 0.36 |
| **total** | **28 978** | **143** | **0.49** |

**This is not STARmap-specific.** `flattened_z` is true for **all 18** bench3 datasets, so the same
collision mechanism applies to every one; STARmap merely happens to be the one that was built.

**Both sides are behaving correctly**, which is why this needs a decision rather than a fix:

* T01's invariant is right for a 2-D section — genuinely coincident cells do corrupt a kNN graph,
  and `evaluate_paper`'s own Moran's/Geary's graphs have the same exposure, unchecked.
* bench3's flattening is right for the protocol — the paper calls these *sections*, and a section is
  what a microtome produces.

**Four options, with the one I would not take marked.**

| option | effect | verdict |
|---|---|---|
| **A.** De-duplicate at the wrapper's input boundary (drop or merge coincident cells) | changes the cells **v25** trains on relative to every other method — breaks the shared-input guarantee that makes the comparison apples-to-apples | ⛔ **do not** |
| **B.** Build with `--no-flatten-z` | keeps true 3-D positions and the collision disappears, but it is **a different dataset from the protocol build** and cannot be tier 1 | tier-2 only |
| **C.** Scope T01's check: allow coincident coordinates when the volume is flagged as flattened serial sections, with the duplicate count reported and carried into `uns` | keeps the protocol build intact and every method on identical input; costs an explicit, documented weakening of a Convention-6 guarantee | **recommended** |
| **D.** Deterministic sub-µm jitter of duplicates inside the model's own input | silent data modification | ⛔ **do not** |

**Resolved: option C, implemented.** `Volume.flattened_sections` permits exact ties,
`Volume.n_coincident_coords` records the count, `validate_volume` warns once with it, and the flag
propagates through `split_holdout` and `loso_folds`. An unflattened volume keeps the hard check.
Verified on the real input: 81 ties (0.49 % of 16 527), warned once, validation passes.

---

## 5. ✅ The v25 fit cost — the number the 3-vs-5 decision rests on

**One 1200-step fit on tier-1 STARmap: 1712.7 s of training (28.5 min), 1729.4 s total wall,
1709 MB peak RSS, on CPU, 16 527 cells x 28 genes.** Exit 0, 9 570 cells written, prediction scored.

| measurement | value |
|---|---|
| training, 1200 steps | **1712.7 s = 28.5 min** (1.43 s/step) |
| total wall (load + fit + 3 generations + write) | 1729.4 s |
| peak RSS | 1709 MB |
| a contended run, sharing cores with `make test` | 2224 s — **30 % slower**, so quote the clean figure |

**Against the §12 cost model, which assumed ~25 min per fixture-equivalent fit at 2400 steps:** a
1200-step fit costs 28.5 min here, so a **2400-step fit is ~57 min — roughly 2.3x the model.**
STARmap's FEF weight should be **~2.3, not 1.0**, and every FEF figure in `specs/10` §12 scales with
it. On that basis the three-dataset campaign is closer to **220 CPU-hours than 96**. Re-derive §12
before the 3-vs-5 decision rather than reusing the modelled number.

### What the prediction scores, and why it is not a v25 result

Scored on the pinned evaluator, medians over the three held-out sections:

| metric | s2 | s4 | s6 | median | `random` probe | `flanking_copy` |
|---|---|---|---|---|---|---|
| `celltype_localization` | 0.4111 | 0.2923 | 0.5178 | **0.4111** | 0.065 | 0.7765 |
| `marker_field_r` | 0.1611 | 0.0738 | 0.1813 | **0.1611** | 0.058 | 0.8857 |
| `marker_depth_r` | 0.3888 | 0.1364 | 0.3503 | **0.3503** | 0.163 | 0.9794 |
| `morans_pearson` | −0.1242 | −0.2469 | −0.1999 | **−0.1999** | −0.062 | 0.9836 |
| `gearys_pearson` | −0.1380 | −0.2486 | −0.2054 | **−0.2054** | −0.058 | 0.9840 |
| `gene_mean_spearman` | 0.8002 | 0.8227 | 0.8248 | **0.8227** | −0.013 | 0.9863 |
| `cell_count_ratio` | 1.0346 | **0.3345** | 0.9289 | 0.9289 | — | — |

**These numbers are a smoke result and must not be quoted as v25's performance.** The run used a
**hand-written stand-in configuration**, because per-dataset selection never ran:

* `text_emb_mode = "lookup"`, not the shipped `medcpt` — the encoder needs `transformers` and
  weights unavailable here, so this run is effectively **ablation A3**, not the shipped method;
* `expr_pca_dim = 16`, forced, because the default 32 exceeds the 28-gene panel (§4's second finding);
* `train_steps = 1200`, while T09's joint gate selected **2400** on the fixture;
* **none of T09's calibration ran** — no `apply_lengthscale`, no detection calibration, no fitted
  anchor `w(v)`.

The last one is the likely explanation of the headline failure. **Moran's and Geary's correlations
are negative**, below even the `random` probe, and GATE 1 established that the correlated prior's
entire mechanism runs through the fitted length-scale `ell`. An uncalibrated `ell` is exactly the
configuration in which the prior would impose the wrong spatial structure rather than none. The
`cell_count_ratio` of 0.33 on `section_4` says the layout head under-produced badly there too.

**So the honest reading is narrow and worth stating: the pipeline runs end to end on real data, the
cost is known, and the fixture-tuned defaults do not transfer to STARmap's 28-gene, near-dense
(median detection 0.9999) panel without selection and calibration.** That is an argument *for* the
per-dataset selection step, not evidence about the method — and it means **selection is on the
critical path for any quality number**, not an optimisation to defer.

---

## What is still unknown

- **The real-data fit cost** — the number the 3-vs-5 dataset decision depends on. Blocked by §4.
- **The ~23-fit selection cost on real data.** Blocked by §4.
- **C1 and C2.** Blocked by §4; C1's method is now sharper because of §3.
- **`deep_starmap` panel coverage.** Blocked by §2.
- **Comparator numbers on the pinned instrument.** Blocked by §1 — though §3 supplies the floor and
  ceiling without them.

---

## 6. Attribution of the autocorrelation failure (2026-08-21)

The smoke run's headline failure — Moran's and Geary's correlations **negative**, below the `random`
probe — was attributed by direct experiment. **My calibration hypothesis was wrong**, and the cause
is now localised.

### 6.0 The requested selection run cannot run in this container

Per-dataset selection and a shipped-config fit are **campaign-machine work**. `Config`'s default is
`text_emb_mode="medcpt"`, and the MedCPT encoder is unreachable here (`huggingface.co` → 403 through
the proxy, verified). `text_emb_mode` is also one of selection's gates, so a local selection cannot
score the `medcpt` arm and **cannot return the shipped configuration**. Running it here would have
cost ~11 h and produced a config that is A3 by construction.

So the question — *"attribute the failure; the smoke config differed from shipped in four ways at
once"* — was answered by varying the factors **one at a time** instead, which is both sharper and
~20x cheaper. Three of the four are testable locally.

### 6.1 The failure is a magnitude collapse, not an inversion

Per-gene Moran's I on `section_2`, 28 genes:

| | min | median | max | sd |
|---|---|---|---|---|
| ground truth | +0.1995 | **+0.4635** | +0.7755 | 0.1405 |
| v25 (smoke) | +0.0210 | **+0.0703** | +0.1706 | 0.0365 |

The emitted field carries **15 % of the tissue's autocorrelation**, and the residual spread is small
enough that the correlation over 28 genes is noise about zero — which lands at −0.12 by chance.
**The negative sign is not an inversion and should not be read as one.** `paper_morans_pearson` is a
*ranking* metric and it is uninformative once the predicted values are squashed into a narrow band;
the interpretable quantity is `morans_median_pred` against `morans_median_gt`.

### 6.2 Calibration: RULED OUT

The fitted length-scale on this volume is **ell = (116.3, 116.3, 132.0) µm** against the default
(100, 100, 100) — only **0.86x / 0.76x** off, not the order of magnitude a 7x collapse would need.
Tested directly anyway (arm A, one full fit):

| | `morans_median_pred` | `morans_pearson` |
|---|---|---|
| default `ell` | 0.0649 | −0.1999 |
| **calibrated `ell`** | **0.0577** | −0.0998 |
| ground truth | 0.4102 | — |

Calibration moves nothing, and moves the median slightly the *wrong* way. **My stated hypothesis is
refuted.**

### 6.3 The layout head: a large part of the localization failure, NOT of the collapse

Arm B swaps `layout_mode="field"` for `"resample"`, which places cells at real flanking-section
positions. One factor changed, one full fit.

| metric | A: field layout | **B: resample layout** | `flanking_copy` | GT |
|---|---|---|---|---|
| `celltype_localization` | 0.4252 | **0.7008** | 0.7765 | — |
| `cell_count_ratio` | 0.8283 | **0.9875** | — | 1.0 |
| `marker_field_r` | 0.1244 | 0.1781 | 0.8857 | — |
| **`morans_median_pred`** | 0.0577 | **0.0904** | — | **0.4102** |
| `morans_pearson` | −0.0998 | −0.1202 | 0.9836 | — |

**Two findings, and they point in different directions.**

* **Localization and cell count are largely the layout head.** Given real positions, localization
  goes 0.425 → 0.701 and lands *on* the `flanking_copy` floor per section (B: 0.7008 / 0.7760 /
  0.6808 against the floor's 0.7008 / 0.7765 / 0.7868). The intensity-field layout is producing
  positions materially worse than copying a neighbouring slice, and `cell_count_ratio` recovers to
  0.99.
* **The autocorrelation collapse is NOT the layout.** With *correct real positions*, the emitted
  expression still carries only **22 %** of the tissue's Moran's I (0.0904 vs 0.4102), and the
  correlation stays negative.

### 6.4 Where that leaves the cause

Ruled out by direct experiment: **length-scale calibration** (6.2) and **the layout head** (6.3) —
the latter matters a great deal for C1 but does not explain the collapse. `expr_pca_dim` is not a
free variable: 32 exceeds the 28-gene panel, so every run must use ≤ 28.

**The collapse is in the expression path** — the flow/decoder emits values that are close to
spatially unstructured even when placed at correct positions. Two candidates remain, and they split
by machine:

| candidate | testable locally? |
|---|---|
| **budget** — 1200 steps, where T09's gate chose 2400 on the fixture, and STARmap is larger | ✅ one 57-min fit |
| **`text_emb_mode="medcpt"`** — every local run is forced to `lookup`, i.e. ablation A3 | ❌ **server only** |

### 6.5 Recommendation: the campaign waits

By the standard set for this run — *"if the numbers recover, the campaign is a cost question; if
they do not, we have a real-data problem the fixture never showed and the campaign waits"* — **the
numbers did not recover.** A 200+ hour campaign against a method emitting 15–22 % of the tissue's
spatial autocorrelation would measure the defect at scale.

**The cheap next step is the 2400-step arm** (one fit, ~57 min, local): it closes the last locally
testable factor. If the collapse survives it, the remaining explanation is the text channel or a
genuine expression-path defect, and both are investigations rather than campaign runs.

⚠️ **All numbers in §6 come from a non-shipped configuration** (`text_emb_mode=lookup`,
`expr_pca_dim=16`, 1200 steps, no T09 calibration beyond the arm under test). They attribute a
failure; they are **not** v25's performance, and none may be quoted as such.

---

## 7. The budget: RULED OUT (2026-08-21)

Arm C is arm A with one factor changed — 2400 steps instead of 1200, the budget T09's joint gate
selected on the fixture. **2312.7 s of training (38.5 min), 2331.7 s wall, 1722 MB peak.**

Reported as the magnitude, not the ranking metric (§6.1: `paper_morans_pearson` is uninformative
once the predicted values are squashed):

| | `morans_median_pred` | as a fraction of GT | `gearys_median_pred` |
|---|---|---|---|
| A — 1200 steps, field layout | 0.0577 | **14.1 %** | 0.9432 |
| B — 1200 steps, resample layout | 0.0904 | **22.1 %** | 0.9089 |
| **C — 2400 steps, field layout** | **0.0753** | **18.4 %** | 0.9225 |
| ground truth | **0.4102** | 100 % | **0.5884** |

**Doubling the budget buys 4.3 percentage points against an ~82-point gap.** That is not a budget
effect. `morans_mae` barely moves (0.3495 → 0.3344), and Geary's C confirms it from the other
direction: 0.92 against the tissue's 0.588, where 1.0 is the no-autocorrelation value. **The emitted
field is close to spatially random however long it trains.**

Two further observations from arm C, both against the longer budget being a fix:

* `celltype_localization` improves (0.4252 → 0.5822) but stays well below the `flanking_copy` floor
  of 0.7765 — consistent with §6.3: the layout is a separate, real defect.
* **The layout gets less stable, not more.** Arm C emitted **146 cells for `section_4` against
  4 102 in the ground truth**, and 11 168 for `section_2` against 4 187. Per-section
  `morans_median_pred` scatters accordingly (0.1297 / 0.0157 / 0.0753). A longer fit is making the
  intensity field's cell-count integral worse.

### Where that leaves it

Ruled out by direct experiment, each with one factor changed and a full fit:

| factor | verdict |
|---|---|
| length-scale calibration (§6.2) | **ruled out** — fitted `ell` is 0.86x/0.76x of default; no movement |
| the layout head (§6.3) | **ruled out as the cause of the collapse** — 22 % even at real positions. A separate, first-class defect (`specs/10` §4.5b, R11) |
| **the step budget (§7)** | **ruled out** — 14.1 % → 18.4 % at 2x |
| `expr_pca_dim` | not a free variable — 32 exceeds the 28-gene panel |

**So the collapse is not a budget effect, and the fixture is not representative of this regime.**
GATE 1 measured the correlated prior surviving the fixture's generative map at r = 0.92 and T09
measured the calibration objective live under `zinb-flow`; neither holds up here, on real tissue,
without the pipeline losing ~80 % of the structure somewhere between the prior and the counts.

**Next, and running: `scripts/t10_chain_diagnostic.py --steps 2400`** — the same chain measured
stage by stage (GRF prior at the generated positions → latent after the flow → decoded `mu` →
sampled counts, against the real section's counts and the latent the encoder makes of them), one
estimator throughout. That localises the loss to the prior, the flow or the decoder instead of
leaving it as a whole-pipeline number. **`text_emb_mode="medcpt"` is not blamed until the chain has
been read** — it is the last untested factor, but it is a gene-embedding channel and the collapse is
spatial, so it is not the first suspect.

---

## 8. The chain: the structure is destroyed at the count-sampling step (2026-08-21)

`scripts/t10_chain_diagnostic.py --steps 2400`, `section_2`, 11 168 generated cells. One estimator
throughout — row-standardised kNN Moran's I at k = 10 — median over channels.

| stage | median I | p25 | p75 | channels |
|---|---|---|---|---|
| 1. prior `h0` = GRF at the generated positions | **+0.9714** | +0.9661 | +0.9742 | 64 |
| 2. latent `h` after the flow | **+0.9015** | +0.8856 | +0.9162 | 64 |
| 3. decoded `mu`, before sampling | **+0.8607** | +0.8508 | +0.8722 | 28 |
| 4. **sampled counts** | **+0.1297** | +0.0791 | +0.1779 | 28 |
| REF — real counts | +0.4635 | +0.3587 | +0.5637 | 28 |
| REF — real latent, `encoder(real counts)` | +0.7449 | +0.6752 | +0.7912 | 64 |

**The prior, the flow and the decoder's mean are all fine. The count draw destroys it.**

* Stages 1 → 3 lose **0.11** in total. The correlated GRF delivers at the generated positions
  (0.971), the flow preserves it (0.902), and `mu` is still strongly structured (0.861) — *more*
  structured than the real tissue's own latent (0.745).
* Stage 3 → 4 loses **0.73 in one operation.**
* And the real data shows what that step *should* cost. Real tissue also loses structure from latent
  to counts — that is Poisson noise, and it is why real counts sit at 0.464 rather than 0.745. But
  it **retains 62 %** across that step. The model retains **14 %**. The generated draw is throwing
  away **4.4x more structure than the tissue's own sampling noise does.**

### What the counts look like

Per-gene, `section_2`, raw counts:

| | median mean | median variance | median var/mean | log-log slope |
|---|---|---|---|---|
| real | 5835.0 | 2.88e8 | 12 951 | **1.738** |
| generated | 5119.7 | 4.67e7 | 7 429 | **2.120** |

Zero fraction is right (0.0075 against 0.0068) and the mean is close (0.89x). **The output is not
globally over-dispersed** — its variance is 0.57x the real, *below* it. So the collapse is not "too
much noise" in aggregate.

It is the **shape**: the mean-variance slope is **2.120 against the tissue's 1.738**, a 22 %
relative error against T06's own < 0.15 criterion. Read with stage 3 → 4, that says `mu` is
spatially smooth but **too flat in amplitude across cells** — its between-cell variation is small
relative to the per-cell draw, so sampling buries it.

### The next experiment, and it is cheap

**T09's mean-variance calibration is built, fitted, and not applied by default.** `specs/09` §2
solves `log theta` per gene against **the mean-variance relation at the model's own mean**, jointly
with `pi` — which is precisely the quantity measured wrong above. It ships unapplied because on the
synthetic fixture it had **no headroom** (model error 0.0217 against a real between-section
variation of 0.0397, SPEC_QUESTIONS C28). On real STARmap it has 22 % of headroom.

So: **fit and apply `calibrate_detection` on tier-1 STARmap and re-measure stage 4.** One fit plus a
calibration, and it tests the one part of T09's machinery that has never run against data with room
for it. If stage 4 recovers toward the tissue's 62 % retention, the collapse is a calibration gap
that the fixture could not have shown. If it does not, the ZINB decoder's dispersion parameterisation
is the defect, and that is a T06 question.

⚠️ **`text_emb_mode="medcpt"` is now a weak suspect and should not be blamed.** The chain shows the
loss happening *after* the decoder's mean, and the gene-embedding channel feeds `mu` — which is
measured healthy at 0.861. Whatever the text channel contributes, it is not where the structure is
going.

---

## 9. The calibration arm: the calibrator works, and it does not fix the collapse (2026-08-21)

`scripts/t10_chain_diagnostic.py --steps 2400 --calibrate`. One fit (2423.3 s), then T09 §2's
`calibrate_detection` fitted in **7.4 s** on the four training sections, and the emission stages
re-measured from the *same* latent — so the only thing that changes between the two rows is the
calibration.

| | `morans_median_pred` (counts I) | latent I | **retention** | mean-variance slope | rel. error vs tissue's 1.738 |
|---|---|---|---|---|---|
| **real tissue** | +0.4635 | +0.7449 | **62.2 %** | 1.738 | — |
| uncalibrated | +0.1297 | +0.9015 | **14.4 %** | 2.121 | **22.0 %** |
| **calibrated** | +0.1317 | +0.9015 | **14.6 %** | **1.840** | **5.9 %** |

**The calibrator does exactly what it was built to do.** The mean-variance slope goes 2.121 → 1.840
against the tissue's 1.738 — a relative error of **22.0 % → 5.9 %**, which **passes T06's < 15 %
criterion** where the uncalibrated arm failed it. The quantity §8 measured wrong is now right, and
it cost 7.4 seconds. `calibrate_detection` is validated on real data, and its "no headroom on the
fixture" default (C28) is confirmed as a fixture artifact.

**And the collapse does not move: retention 14.4 % → 14.6 %, against the tissue's 62.2 %.**
`mu` is untouched (0.8607 both arms, as expected — the calibration corrects `pi` and `theta`, not
the mean).

### This is the design branch, not the tuning branch

By the standard set for this run — *"if it does not, the emission model itself is wrong for
near-dense panels and that is a design finding, not a tuning one"* — **it is a design finding.**

The measurements support a specific statement of it. The count draw is spatially uncorrelated by
construction, so the ratio `I(counts) / I(mu)` is the share of the emitted between-cell variance
that lives in the *structured mean* rather than in the draw:

| | structured share of between-cell variance |
|---|---|
| model, uncalibrated | **15.1 %** |
| model, calibrated | **15.3 %** |
| **real tissue** | **62.2 %** |

**The decoder puts ~15 % of its output variance into `mu` and ~85 % into the sampling
distribution. Real tissue is roughly 62/38.** Real cells genuinely differ from one another by a
lot, and those differences are spatially organised; the model reproduces the *pattern* of that
variation almost perfectly (`mu` at I = 0.861, higher than the tissue's own latent at 0.745) but at
a fraction of its *amplitude*, and then fills the gap with dispersion.

**That is why calibrating `theta` cannot help.** The calibrator matches the *marginal* mean-variance
relation, and it does — but matching a marginal only reallocates variance within the same
mean/dispersion decomposition. It has no term that moves variance *from* the dispersion *into* `mu`,
which is what would be needed. The defect is in the decomposition, not its parameters.

⚠️ **One assumption is load-bearing and should be stated wherever this is quoted**: that the ZINB
draw contributes no spatial autocorrelation, so `I(counts)/I(mu)` reads as a variance share. That is
true by construction here — the draw is i.i.d. per cell given `mu`.

### What this does and does not implicate

* **Not the prior, the flow, or the budget** — all three measured healthy or ruled out (§6, §7, §8).
* **Not the layout** — 22 % retention even at real positions (§6.3). Still a separate real defect
  (R11).
* **Not calibration** — now measured, works, insufficient.
* **Not `text_emb_mode`** — the loss is downstream of `mu`, which is healthy.
* **The emission model's variance decomposition**, on a panel where per-gene means are in the
  thousands and detection is 0.9999. `progress/fixture_limitations.md` §1 records why no fixture
  measurement could have caught it.

### Recommendation

**The campaign stays on hold, and the next step is a T06 design question rather than another
pilot run.** The question to put to it: *what makes `mu` carry only 15 % of between-cell variance
when its spatial pattern is correct?* Candidates worth checking before any new fit —

1. **The size-factor path.** At a median 5 120 counts per cell, if `mu` is largely set by the size
   factor and the latent modulates it only weakly, that is exactly this signature.
2. **`Config.decoder_mu_link`.** T06 kept softplus as the default with `exp` selectable and
   measured; a softplus link compresses dynamic range at large means, which is the regime here and
   is not the regime the fixture tested.
3. **The reconstruction loss's own optimum.** A ZINB NLL at these means can be reduced by widening
   dispersion rather than sharpening `mu`, and nothing in the objective penalises that trade — which
   would also connect to R4's likelihood/fidelity divergence and to T06's unresolved gene-gene
   covariance loss.

Item 2 is a one-line config change and is the cheapest of the three to test.

---

## 10. Candidate 2 — the size-factor path, and what real tissue requires (2026-08-21)

**The structural half needs no fit.** The decoder builds

```
mu = link(MLP_mu(u)) * size_factor          # model/expression.py::ZINBDecoder.forward
```

so `mu` is **multiplicative in the size factor by construction**, and in logs the split is exact:
`log mu = shape + log s`, `Var(log mu) = Var(shape) + Var(log s) + 2 Cov`. `scripts/t10_chain_diagnostic.py`
now reports those shares (`mu_variance_decomposition`), so the model's own split comes with every
run rather than needing its own experiment.

**The real-data half is measurable now, and it sets the bar.** On `section_2`, regressing each
gene's `log(count + 1)` on the cell's `log(total)`:

| | value |
|---|---|
| median library size | 226 580 counts |
| `sd(log total)` across cells | 0.4725 |
| `sd(log(count + 1))` per gene, median | **1.2127** |
| **share of per-gene between-cell variance explained by library size alone** | **21.8 %** (p25 12.9 %, p75 32.3 %) |
| **share the latent must therefore supply** | **78.2 %** |

**Real cells differ from each other by a great deal — `sd(log(count+1))` of 1.21 is a factor of
~3.4 in each direction — and only about a fifth of that is library size.** A decoder whose `mu`
tracks the size factor and modulates it weakly can reproduce at most ~22 % of what the tissue does,
which is the same order as the 15.3 % structured share measured in §9.

⚠️ **The two numbers are not the same quantity and must not be quoted as if they were.** 21.8 % is
a decomposition of the *real* data (how much a decoder cannot get from library size); 15.3 % is the
share of the *model's emitted* variance that is spatially structured. They are consistent with one
story — `mu` lacking amplitude — but neither implies the other. The model's own `Var(shape)` vs
`Var(log s)` split, which *is* the directly comparable quantity, comes from the run in §11.

---

## 11. Candidate 1 — the `mu` link function. **This is the fix.**

`Config.decoder_mu_link` ships as **`softplus`**, and softplus compresses dynamic range at large
pre-activations — at a median 226 580 counts per cell and per-gene means in the thousands, that is
exactly this regime, and it is not the regime the fixture tested. `exp` is selectable and is the
one-line change.

⚠️ **Discrepancy found while reading it, worth a fix either way**: `ZINBDecoder`'s docstring says
*"``link`` is ``Config.decoder_mu_link``: ``exp`` by default and ``softplus`` — the spec's own
choice — selectable"*, but `Config.decoder_mu_link: str = "softplus"`. **The docstring and the
default disagree**, so a reader of the decoder believes the opposite of what ships. One of the two
is wrong and T06 should say which.

The arm runs `--steps 2400 --calibrate --decoder-mu-link exp` and reports the structured share, the
retention, the mean-variance slope and the `Var(shape)` / `Var(log s)` split, so candidates 1 and 2
are answered by one fit. It also **saves the fitted model** (`--save-model`), so any further
analysis of this configuration costs no refit — four 2 400-step fits have been spent re-deriving
the same model.

### Result

`--steps 2400 --calibrate --decoder-mu-link exp`, one fit (model saved to
`runs/pilot/model_exp_2400.pt`). Against the shipped `softplus` arm:

| | `softplus` (shipped) | **`exp`** | real tissue |
|---|---|---|---|
| counts I | +0.1297 | **+0.5253** | +0.4635 |
| `mu` I | +0.8607 | +0.9008 | — |
| **structured share** `I(counts)/I(mu)` | **15.1 %** | **58.3 %** | ~62 % |
| mean-variance slope | 2.121 | **1.807** | 1.738 |

**Changing one line takes the structured share from 15 % to 58 %.**

### The density confound — tested, and the result survives

The `exp` arm emitted **48 343 cells for a section whose ground truth has 4 187** — 11.5x. Denser
sampling puts kNN neighbours closer together and inflates Moran's I mechanically, so the gain had to
be separated from the fix. Re-measured on random subsamples of the *same* generated section, from
the saved model (no refit):

| n cells | density vs GT | counts I | `mu` I | structured share |
|---|---|---|---|---|
| 48 343 | 11.5x | +0.5253 | +0.9008 | 58.3 % |
| 16 748 | 4.0x | +0.5192 | +0.8756 | 59.3 % |
| 8 374 | 2.0x | +0.4967 | +0.8386 | 59.2 % |
| **4 187** | **1.0x — GT-matched** | **+0.4782** | +0.7792 | **61.4 %** |
| *real tissue* | *1.0x* | *+0.4635* | — | *~62 %* |

**At GT-matched density the model reaches +0.4782 against the tissue's +0.4635 — 1.03x — and a
structured share of 61.4 % against ~62 %.** Density accounts for about 0.05 of the 0.40 gain; the
structured share is nearly flat across densities (58-61 %), which is the correct behaviour for a
ratio. **The fix is real.**

### Candidate 2 — ruled out

| quantity | value |
|---|---|
| `Var(shape)` — latent-driven | 0.60376 |
| `Var(log s)` — size factor | 0.00123 |
| **share of `Var(log mu)` from the latent** | **100.1 %** |
| **share from the size factor** | **0.2 %** |

`mu`'s dynamic range is **not** the size factor — the latent supplies essentially all of it. The
hypothesis was reasonable given `mu = link(MLP_mu(u)) * size_factor`, and it is wrong.

### Four things to carry, and one of them is a caution

1. **The cell count is badly wrong in this arm** — 48 343 against 4 187. That is the layout head
   (R11) and is *independent* of the emission fix; the two defects need separating before any
   headline number. It also means the `exp` arm's raw scored metrics are not usable as-is.
2. **Amplitude is improved, not matched.** `sd(log mu)` = 0.7767 against the tissue's
   `sd(log(count+1))` = 1.2127 (§10). The *share* is recovered; the absolute spread is still below
   real. Whether that matters is a T06 question.
3. **Calibration is now unnecessary and mildly harmful here.** It takes the slope 1.807 → 1.611
   against the tissue's 1.738 — it *overshoots*, because it was correcting a defect the `exp` link
   removes. If the link changes, T09's calibrator should be re-evaluated, not inherited.
4. ⚠️ **Retention is not comparable across arms.** Its denominator is the *real latent*, which is
   produced by that arm's own encoder: 0.7449 in the `softplus` fit, 0.5812 in the `exp` fit, which
   moves the real-tissue retention row from 62.2 % to 79.8 % without anything about the tissue
   changing. **Cross-arm, use counts I and the within-arm structured share.** The retention rows in
   §9 and the `exp` report are each valid within their own fit and must not be lined up.

### This does not close R4

The `exp` link changes the **link function**, not the objective. R4's other instances are untouched:
the layout head's Poisson NLL falling three nats below the truth's, the 1200 → 2400 likelihood/
fidelity divergence, and the gene-gene covariance loss. What this result *adds* to R4 is that the
inversion can be very sensitive to a parameterisation choice — softplus compressing dynamic range at
means in the thousands gave the NLL an easy way to buy likelihood with dispersion, and removing that
compression took back three quarters of the gap without touching the objective at all.

**Resolved since this was written.** §11's docstring discrepancy — `ZINBDecoder` saying `exp`,
`Config` shipping `softplus` — was investigated and is **not** a defaulting error. `softplus` was
chosen deliberately at T06, on the fixture, where it won T06's deciding statistic (per-gene
mean-expression correlation), and T06 pre-specified the condition on which to revisit it. STARmap
meets that condition. `Config.decoder_mu_link` now defaults to `exp` with `softplus` still
selectable, and the field's own docstring carries both measurements so the comparison is not a
cross-reference to a file. Recorded in `specs/06` and `progress/t06_expression_head.md` as T06's
own condition being met by T10, not as a bug fix. Only the docstring's *stale* half was wrong.

---

## 12. Where this leaves the campaign

**Stopping here, as agreed.** Candidate 1 recovered the structured share, so the emission collapse
has a one-line candidate fix that is validated on real data at GT-matched density. What remains is
**not** more pilot iteration:

* **A T06 design decision** on the `mu` link default and the docstring discrepancy, with R4's four
  instances as context.
* **The layout head (R11)** — independently broken, and now the dominant defect in the `exp` arm.
* Only then a re-costed campaign, since the corrected cost model (§7, ~2.3x) still stands.

