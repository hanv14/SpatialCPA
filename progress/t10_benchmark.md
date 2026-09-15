# T10 — Benchmark and baselines

Task log. `PROGRESS.md` is the index; this file holds T10's entries.

---

## 2026-09-11 — the tier-1 comparator re-score, and the instrument pin becomes a test

### What arrived

`specs/10` step 5 — *"Comparators on tier 1: SpatialZ, FEAST, isoST, v20, plus `oracle` and
`flanking_copy` — **required**, because the existing numbers are not in this repo"* — **was run.**
`evaluate_all --force` into `benchmark-pbya-v3/results_rescored/` for
`starmap_visual_cortex/paper_2_4_6`, six methods, **0 failures**, on the pinned evaluator. The table
is inlined in the paper's §8.2 with `flanking_copy`, `oracle` and v25's own row beside it.

Three reports that said the run could not be done here are stale on that point and are marked, not
rewritten: `reports/pilot.md` §44 (annotated in place), `reports/spatialz_claim_struck.md` §1–2
(superseded by its own §6 amendment), and the paper's earlier §8.3 (rewritten as §8.4).

### What the table says

1. **On four of the five metrics with a floor, no method reaches it** — not SpatialZ, FEAST, isoST,
   v18, v20, v21 or v25. Best in column is −0.0025. Copying the flanking sections is at or above the
   state of the art on four of the five readable metrics. **This is §6.1 generalised past our own
   method and is the most valuable thing the run produced.**
2. **SpatialZ beats v25 on five of six.** Our single win is `umap_mixing`, the one metric with no
   floor and no ceiling — unreadable by this project's own rule, and not claimed.
3. **v20 sits +0.0001 from `flanking_copy`** on `celltype_localization` and −0.0025 on both
   autocorrelation metrics. R13's "`cross-mix` under `resample` *is* a copy" now holds on a second
   dataset, on the pinned instrument, to four decimals.

### The struck claim stays struck

"v20 beats SpatialZ 5 of 6" is now **arithmetically supported** by this run and **remains withdrawn**.
`spatialz_claim_struck.md` had two objections; the run resolves the evidence objection (§1–2) and
leaves the circularity objection (§3) untouched — and §3 was the load-bearing one. Recorded as a
dated §6 amendment to that file, with the sentence the situation needed: **getting a number does not
unstrike a claim.**

### RESOLVED — v18 and v20 are one prediction on tier-1, and it is a finding

Raised as an open flag, settled the same day by diffing the prediction files directly. On
`paper_2_4_6` the two are **identical array by array** — `X/data`, `X/indices`, `X/indptr`,
`cell_id`, `cell_type`, `section`, `x`, `y`, `z`, across **12 403 cells / 344 361 non-zeros**. Only
`/uns` differs, and `/uns` is not scored.

They are not the same method: in the **wide** regime they differ, and **only in expression** — on
`allen_merfish_brain/wide_26_…_34`, `X` differs while every coordinate, `cell_id`, `cell_type` and
`section` matches. **v20's changes over v18 are expression-path only and gated on the section gap
exceeding the median spacing, which `paper_2_4_6` never reaches.**

So the tier-1 protocol does not fail to *resolve* the two versions — **it never executes the code
that separates them.** The six comparator columns are five distinct predictions. Promoted out of the
flag list into §8.2's fourth reading, because two rows identical to four decimals reads as a
transcription error and is not one.

#### 2026-09-13 — the mechanism, and a correction to the sentence above

The empirical finding stands unchanged. The *explanation* — "gated on the section gap exceeding the
median spacing, which `paper_2_4_6` never reaches" — is right in direction, weaker than the truth,
and misses half the cause. Audited in `reports/inert_mechanisms.md`; the two corrections are now in
§8.2.

1. **`alpha` is zero by construction, not by a threshold that was missed.**
   `alpha = clip((this_gap/med_gap − 1)/(gap_scale − 1), 0, 1)`
   (`learn_spatialcpav21.py:1573-1580`), and `med_gap` is the median spacing of the **training**
   sections. `config.held_out_indices` holds out every other section, so the flanking pair bracketing
   any held-out section is a pair of consecutive training sections and `this_gap ≡ med_gap`. The
   numerator is identically zero — on every dataset, at any section thickness, under any
   `--gap-scale`. Only `--design wide` reaches `alpha > 0`. "Never reaches that gap" implies a margin;
   there is no margin.

2. **A second, unrelated default disables four more mechanisms, in both versions.** No `METHODS`
   entry passes `wrapper_args`, so `--edit-weight` keeps its wrapper default of `0.25`
   (`run_spatialcpav21.py:335`; `V14Config.edit_weight`, `:1038`). v18's gene-mix (`:1715`), v18's
   raw-output path (`:1792`), v20's `edit_gap_extra` (`:1770`) and v21's field repair (`:1778`) are
   each gated on `edit_weight == 0.0`. None has run in a scored row. It bears equally on both
   versions so it does not affect the identity — but **neither version ever executed its own
   expression mechanisms**, and that includes the raw-output path written specifically to fix
   `gene_var_spearman` on the three EASI-FISH volumes.

Pinned by a symbol-level diff rather than by reading: 79 shared top-level symbols between v18 and
v20, **75 byte-identical**; only `V14Config` (added fields, **no shared default changed**),
`_generate` and `_phase_b` differ. `_phase_b`'s only difference is the `curriculum_flow` branch,
whose RNG draw short-circuits when the flag is off, so training consumes an identical stream. At
`alpha = 0` the cross-mix and `edit_gap_extra` blocks are skipped **without consuming any RNG**.
Bitwise identity is the only outcome the code permits, at any seed.

Consequence for the record: **v21 vs v20 on tier-1 is a layout-and-donor-selection comparison, not an
expression one.** The only two v21 changes reachable at `alpha = 0` are its narrow-gap adaptations —
patch frequency `4.0 → 8.0` (`:1538-1541`) and re-grounding margin `1.0 → 5.0` (`:1666-1669`).
v21's coherent mix, multi-partner draws, field alignment and field repair are all in the dead set,
so no `paper_*` row is evidence about them.

### 2026-09-14 — the expression ablation, two families

`v21_<learner>` and `v14_<learner>`, five learners each (`ridge`, `knn`, `rf`, `gbm`, `mlp`),
registered in `METHODS` and appended to `METHOD_ORDER`. One new wrapper per host method under
`benchmark-pbya-v3/src/bench3/methods/`; `learn_spatialcpav21.py`, `spatialcpav14/` and
`benchmark-pbya-v2/` are all untouched, `config.py` is insertions only, and `evaluate_paper.py`
still hashes to `7362669…38992`.

The swap point is the **`VirtualSlice` boundary**: both host methods return
`VirtualSlice(coords, expression, cell_type, cell_type_idx)` with layout, donor selection, type
placement and composition matching already applied, so replacing `.expression` ablates the
expression step and nothing else — without editing either method or copying its `_generate`.

**v14's contrast is the cleaner of the two, and it is worth stating which is which.** At
`--edit-weight 0.0` — the configuration this project runs v14 at — v14 emits a *verbatim* real
profile (`spatialcpav14/trainer.py:662`; the decode blend at `:532` is skipped), so the comparison
is "one real cell's profile" vs "a regressed conditional mean" with nothing in between. v21 runs
at `edit_weight=0.25`, so a quarter of its output is already a PCA decode (see the 2026-09-13
entry above) and its contrast is muddier. Neither family measures the cross-mix; `alpha` is zero
by construction on every `paper_*` design.

Two things duplication guards catch rather than trust. The v21 wrapper builds `V14Config()`
directly, justified by a startup check that **51 knobs** equal their wrapper defaults. The v14
wrapper cannot do that — v14 *is* driven by flags (`-- --edit-weight 0.0 --ground-blend-flow 1.0
--ground-k 2` arrive through `run_all`'s `extra_args`) — so it declares v14's whole CLI and
verifies **28 flags and 31 `cfg.*` assignments** against `benchmark-pbya-v2`'s wrapper. Both
guards were exercised in the failing direction (drifted default, added flag, rewired config line).

Two findings from building it:

* **Random forest predicts non-deterministically** at `n_jobs=-1` even with `random_state` fixed —
  the fit is bit-identical (checked on `tree_.value`), but joblib threads accumulate per-tree
  contributions in varying order. Single-threaded prediction after the parallel fit; all five
  learners now reproduce bitwise.
* **`openst_lymph_node` cannot run these variants at all.** A regressor emits no zeros (measured
  density 1.000), and at ~20 000 genes over ~474 k held-out cells that is 37.9 GB dense / 75.8 GB
  as CSR, per learner. Both wrappers now estimate this *before training* and refuse
  (`--max-dense-gb`, default 8). Every other dataset is under 0.6 GB. `reports/ml_ablation_cost.md`
  has the table and the three ways out — none of which is thresholding predictions to zero, which
  would contaminate the detection column the ablation exists to read.

Marginal cost measured rather than modelled: ~2–6 CPU-hours per family on 4 cores, dominated by
`rf` and `gbm`. The host methods' own training (170 complete runs) is the real bill and is not
measured here. Not run.

### 2026-09-14 (2) — the ablation's third family, `v18_<learner>`

Same five learners, same `VirtualSlice` swap point. v18's wrapper is a v3 wrapper with a
module-level `_build_config`, so this one imports and CALLS it: no config mapping is duplicated at
all and the startup guard checks the **24 flags** only (exercised both ways — drifted default,
added flag). `learn_spatialcpav18.py` untouched, `config.py` insertions only, evaluator hash
unchanged.

Two properties specific to v18 at `--edit-weight 0.0`, both verified against the source rather
than assumed:

* **The baseline is already a per-gene chimera.** The gene-mix at `:1231` is live at
  `edit_weight == 0`, so at `--gene-mix-frac 0.15` ~15 % of each cell's genes come from a second
  local same-type donor. It is a per-gene *value* choice, so it is expression, so the ablation
  replaces it. That makes v18 the closest of the three baselines to a per-gene synthesis method,
  and v14 (no gene-mix, verbatim single copy) the cleanest single-copy contrast.
* **v18 emits RAW measurements, not `expm1` of log** (`:1240`). The wrapper mirrors v18's own
  `raw_ok` predicate and trains the learner on whichever scale v18 would emit, so baseline and
  variant are scored on one scale. `--target-scale` overrides; `auto` is the default and is what
  keeps the rows comparable.

⚠️ **The raw path's zero-clip confounds the detection column, for this family only.** v18 never
clips (it copies real, non-negative measurements); a regressor can predict below zero, so the
prediction is clipped at 0 — and that manufactures zeros. Measured density **0.86–1.00** across the
five learners (ridge 0.963, knn 0.999, rf 1.000, gbm 0.999, mlp 0.864) against a fixture zero
fraction of 0.39, where v14/v21 emit a flat 1.000 through `expm1`. So a `v18_*`
`paper_gene_detection_spearman` is not comparable to a `v14_*`/`v21_*` one, and the difference is
the clip rather than the learner. Recorded in `method_params` as `raw_clipped_at_zero`, in the
wrapper docstring, in `METHODS`, and in `reports/ml_ablation_cost.md`. `--target-scale log` is the
cross-check.

Campaign is now three families of five over 17 datasets: 255 runs, ~6–18 CPU-hours marginal, ~45 GB
of predictions, host training on top. Not run.

### 2026-09-15 — two more learners, and the registry moves to one file

`lasso` and `xgb` added to all three ablation families, on request. Seven learners x three hosts x
17 datasets. `config.py` still has **0 deletions** against the pre-work baseline (`f637730^`);
`evaluate_paper.py` unchanged.

**The learner registry is now `benchmark-pbya-v3/src/bench3/methods/_ml_learners.py`**, imported by
all three wrappers. Adding two learners to three copies was the moment to stop: the copies had
*already* drifted (identical behaviour, divergent comments — checked by hashing each block), and a
`v14_rf` row is only comparable to a `v21_rf` row if they are the same learner. That is now true by
construction rather than by inspection. Each wrapper keeps what is genuinely its own: its parity
guard, its feature builder, its target-scale decision, its output tail. v21 also gained a
`build_parser()` so all three read alike.

Three things the work turned up, all measured:

* **Lasso's conventional `alpha=1.0` is a silent trap.** On standardized features with log-scale
  targets it leaves **0 of 640 coefficients non-zero**, R² = 0.000 — every prediction becomes the
  per-gene training mean, and the row still *scores*, indistinguishable from a fitted learner in the
  results tree. Worse, v18 trains on the RAW scale where EASI-FISH intensities are ~10³ larger, so
  one alpha cannot mean the same thing in both families. The penalty is therefore set as a fraction
  of `alpha_max = max|X'y|/n` (`FracAlphaLasso`), which is scale-free, and an all-zero fit is
  **reported at run time** (`report_degeneracy`) rather than quietly ranked. At the default
  `--lasso-alpha-frac 0.01` the fit keeps 98.1 % of coefficients — so lasso lands near ridge, which
  is what was predicted when it was argued against; raise the fraction for genuine sparsity. The
  scale-free form is demonstrated, not just argued: the same fraction gives **98.1 %** non-zero on
  log targets (alpha_max 0.747) and **97.5 %** on targets 4000× larger (alpha_max 2992). A fixed
  alpha cannot do that, and the alarm was verified firing at `--lasso-alpha-frac 1.0`.
* **XGBoost is deterministic** at `n_jobs=-1` under both `multi_strategy` settings (verified across
  two fits), so it needs none of the single-threaded-prediction fix the random forest required.
* **`multi_output_tree` is not the cheap option**, which was the guess. Measured at n = 4000:
  G = 40 → 7.97 s vs 24.52 s; G = 300 → 64.18 s vs 238.90 s. `one_output_per_tree` is the default,
  and the alternative stays a flag because they are different models, not two implementations of one.

One real bug caught by testing rather than review: `FracAlphaLasso` as a plain class fits fine but
raises inside `Pipeline.predict` on scikit-learn 1.9 — it needs `BaseEstimator`/`RegressorMixin` for
the tag machinery. Fixed, and the hand-rolled `get_params`/`set_params` deleted as redundant.

Campaign is now 357 runs. Not run.

### 2026-09-15 (2) — a regression I introduced, and the guard that now catches its class

Moving the learner flags into `_ml_learners.py` sliced **v21's `--device` out of its parser**: the
flag sat between two learner flags, and the edit took the whole span. Every `v21_*` run then died at
`cfg.device = args.device` — *after* loading the input and building the cell-type vocabulary, so it
looked like a data problem rather than a wrapper bug. v14 and v18 were untouched: their flag-parity
guards compare against their host wrapper's entire flag table, so a deleted host flag cannot survive
startup there. v21_ml declares no host flags at all (it builds `V14Config()` directly), so it had no
equivalent check — the one wrapper without that guard is the one that broke.

My own post-refactor test missed it because it passed each wrapper only the flags I listed, and for
v21 that list was empty. A test that exercises the flags you remembered cannot catch the flag you
forgot.

Fixed, and the class closed: `ML.assert_args_declared(args, __file__)` now runs in all three wrappers
immediately after `parse_args`, before any data is touched. It walks the wrapper's own AST for
attribute reads on the name `args` — exact, so a mention in a docstring or comment cannot cause a
false failure — and fails naming the missing flag. Verified on all three (33 / 26 / 6 distinct
`args.X` reads, all declared) and verified firing when `--device` is removed again.

No results were affected: the failure was before `write_prediction_h5`, so nothing was written and
`--skip-existing` will re-run the `v21_*` cells cleanly. All seven `v21_*` learners were blocked, not
just `lasso`; `v14_*` and `v18_*` were never affected.

### Three cells reported as returned rather than smoothed

- FEAST and isoST return **exactly `0.0000`** on `celltype_localization`: the not-scorable value, not
  a measurement. Read as blank, shown as returned (Convention 6).
- isoST's **0.9956** on `umap_mixing` — largest number in the table, from the method otherwise last
  or second-last in four of five other columns, on the one column with no probe. Three reasons to
  read nothing from that column, ours included.
- FEAST returns **0.7742** on both `morans_pearson` and `umap_mixing`. Noted, unexplained.

### `tests/test_instrument_pin.py` — a specified acceptance test that did not exist

`specs/10`'s acceptance list names `test_evaluate_paper_sha256_unchanged`. **It had never been
written.** The pin lived in prose in `specs/10` §0 and in a `sha256sum` command a person was expected
to run by hand — for the file every published number in this work was measured with.

Written now, four tests, no data / GPU / network:

| test | what it holds |
|---|---|
| `test_evaluate_paper_sha256_unchanged` | the committed evaluator hashes to the pin; the failure message names the file and says **not** to update the constant |
| `test_pinned_hash_agrees_with_the_spec_that_pins_it` | the constant in the test is the one `specs/10` §0 states — the duplication is deliberate and this keeps it honest |
| `test_line_count_matches_the_pin_so_a_mismatch_is_readable` | 764 lines, a second descriptor, so a failure says roughly *how* different |
| `test_the_hasher_rejects_a_changed_byte` | positive control, and it runs **without** the bench3 tree, so it still holds in the checkout where the other three skip |

Verified: the run's reported hash, `specs/10` §0's pin and the committed file all give
`7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992`, 764 lines.

**The general lesson, and it is §7.3's rule again.** An acceptance test listed in a spec and never
written is indistinguishable, from inside the repository, from one that is written and passing.
Nothing failed; the check simply was not there, and the pin held only because nobody had changed the
file. This is the same shape as R18 — a control that exists on paper and not where a consumer reads
it.

### Still open on T10

- `results_rescored/` is not committed; §8.2's 42 numbers are transcribed. The **instrument** is
  reproducible from this repository (hash-asserted); the **measurement** is not.
- `paper_umap_mixing` has **no probe on any run** — no floor, no ceiling, one of the protocol's own
  six columns unreadable.
- The floor/ceiling columns come from the probes tree rather than from the same `evaluate_all` call.
  Legitimate (the probes are model-free and arm-independent) but a join of two invocations.
