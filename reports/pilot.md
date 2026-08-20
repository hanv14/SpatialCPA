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
