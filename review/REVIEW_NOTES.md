# Review notes — SpatialCPA-v18

Where to look, in the order it matters. The file/line references are to this tree.
Everything under "measured" was run in this repository (4-core CPU, the
`envs/lock/cpu-verified.txt` environment). Anything else is marked.

---

## 1. The configuration is not the wrapper's defaults

The published v18 rows ran under eight flags passed as `run_all` extras (README,
"Effective configuration"). Two of them differ from the argparse defaults, and
both change **which code runs**, not just a number:

| flag | ran | default | consequence |
|---|---|---|---|
| `--edit-weight` | `0.0` | `0.25` | At 0 the decoded-flow blend (`learn_spatialcpav18.py:1219-1221`) is skipped, **and** it is the gate for v18's two headline mechanisms: gene-mix (`:1225`, `gene_mix_frac > 0 and edit_weight == 0`) and the raw-output path (`:1240`, `raw_output and edit_weight == 0`). At the default 0.25 neither runs, and v18 emits `expm1` of a 75/25 blend of a real profile and a PCA decode. |
| `--ground-blend-flow` | `1.0` | `0.2` | Every generated cell is offered for flow re-grounding in `_ground` (`:1256`), not a random 20 %. |

In the parent project the `spatialcpav18_gen` registry entry carried **no
`wrapper_args`**, so these flags existed only on the command line of each run.
A bare `run_all --methods spatialcpav18_gen` there runs the *default*
configuration, where gene-mix and raw output are dead code.

This repo pins them (`config.V18_ARGS`), but the question for the published rows
remains: **were all of them run with the extras?** The prediction records it.
Every published `prediction.h5` should carry `uns/method_params` with
`"edit_weight": 0.0` and `"ground_blend_flow": 1.0`. That is the first thing to
check in the lab `results/` tree (§7).

`--ground-temp` is **not** written to `method_params`, and neither is `--device`.
A published row's temperature can only be read from its logged command line
(`run_all`'s stdout, `cmd: …`). The method log's config line doesn't print it
either.

## 2. What v18 actually emits under these flags

Read `_generate` (`learn_spatialcpav18.py:1122-1254`) with the flags above in mind:

- **Positions are resampled real positions.** `_resample_layout` (`:1100`) draws
  `n_target` coordinates from the two flanking training sections, in coherent
  patches, plus 5 %-of-spacing jitter. `n_target` interpolates the flanks' cell
  counts, so `paper_cell_count_ratio ≈ 1` (0.996 measured) follows from the
  construction rather than being learned.
- **Every emitted expression value is a verbatim training measurement.** With
  `edit_weight == 0` no decoded expression reaches the output. With raw
  measurements present (STARmap: `expression_type=raw_counts`), `raw_ok` holds and
  the output is `pool_raw[pick]` with the gene-mix re-applied on the raw scale
  (`:1240-1246`).
- **The trained flow acts only through `pick`**: which real flanking cell each
  generated cell copies. `_ground` offers each cell the `ground_k = 8` nearest
  real cells. It moves the cell off its inherited source only if the flow latent
  prefers a candidate by at least `ground_keep_margin = 1.0` (standardized latent
  distance), then samples among the candidates, with a spatial prior and a reuse
  penalty. The type vote (`_vote_types`) and composition matching then re-pick
  again.

**How much the flow changes, measured** (`scripts/probe_flow_contribution.py`
instruments the four stages without changing them; its prediction is bitwise
identical to the committed one). STARmap `paper_2_4_6`:

| section | cells | re-grounded by flow | re-picked by type vote | by composition | final ≠ inherited | genes mixed |
|---|---|---|---|---|---|---|
| 2 | 4121 | 1775 (43.1 %) | 1246 | 95 | 1346 (32.7 %) | 12.3 % |
| 4 | 4140 | 1928 (46.6 %) | 1299 | 127 | 1522 (36.8 %) | 12.0 % |
| 6 | 4142 | 1949 (47.1 %) | 1355 | 89 | 1454 (35.1 %) | 11.8 % |

So about two thirds of emitted cells are the profile of the flanking cell the
layout drew, and the rest are a different real cell from the same 8-cell
neighbourhood. A fair one-line description of v18 at the published flags is
**learned local donor selection plus per-gene two-donor recombination**. It is
not generative expression synthesis, and the paper's wording should be checked
against that. The nominal `gene_mix_frac = 0.15` yields ~12 % mixed genes, which
suggests some cells find no same-type partner (`_gene_mix`, `:1365`). Worth a
look.

## 3. Reproducibility — measured

- **The STARmap row runs from a fresh clone with no environment variables**:
  `make starmap-row`, 2 min 48 s on 4 CPU cores. v18 itself takes 36 s and
  986 MB peak RSS; the rest is scoring.
- **Deterministic.** Two independent runs gave bitwise-identical `prediction.h5`
  (all of X, obs, method_params; only `uns/wall_time_seconds` differs), and all
  50 scalar metrics were identical at tolerance 0, UMAP included.
- **Not yet compared with the published row.** `expected/published/` is empty
  until the lab files arrive (§7). Two things can legitimately make them differ:
  - `--device auto` picks CUDA when present, and GPU flow training isn't bitwise
    equal to CPU;
  - UMAP (`paper_umap_*`) moves with umap-learn/numba versions.

  `scripts/compare_metrics.py` holds the UMAP pair to a separate tolerance.
  `paper_embedding_mixing_pca` is the deterministic stand-in for the same
  question.

## 4. The scorer — where a number could move without the method changing

- **`evaluate_paper.py` is pinned** (sha256 `7362669200bb…9538992`, README). Its
  dependencies are pinned in `MANIFEST.sha256`. Two config files were edited
  (registry only), and their constants are asserted in `tests/test_pins.py`.
- **Pose.** `paper_marker_field_r` and `paper_celltype_localization` are computed
  after `align.align_by_expression` (`align.py`), which picks the rotation by
  binned-marker-field agreement. Anything that smooths the marker field can
  improve the chosen pose, and so these two scores, without placing cells better.
  `metrics.json` records `align_rotation_deg`, `align_score` and
  `align_runner_up` per section; a marginal pose is visible there.
- **Scale fairness is rank-normalization.** All primary metrics use per-gene
  rank-normalized expression, imported from v2's `evaluate_generation.py`. So
  v18's raw-output fix cannot help them directly. It shows only in
  `paper_gene_var_spearman`, and in `paper_gene_detection_spearman`, which is
  computed on raw emitted values.
- **Not ranked:** `paper_cell_count_ratio` (by construction ≈ 1 here, §2) and
  `paper_rare_celltype_recall`.
- **The v2 `matched`/`gen_*` blocks** are in `metrics.json` for continuity with the
  v2 sweep. The cell-matched ones (`pearson_median`, `celltype_accuracy`, …) are
  reference only: generation has no cell correspondence.
- The harness's own discrimination check (`make selftest`: oracle / flanking copy
  / spatial scramble / random) passes on this tree (measured, 4 min 20 s).
  `flanking_copy` is the natural baseline for a method that copies flanking
  cells; see `results/summary/selftest_metrics.json` after running it.

## 5. The expression ablation (`v18_*`)

`run_spatialcpav18_ml.py` calls v18's own `generate_virtual_slice` and replaces
only the returned `expression` array with a learner's prediction. v18's layout,
donor selection, type vote and composition matching are untouched, and
`learn_spatialcpav18.py` is not edited. Each row answers: *what does emitting
copied real measurements buy over emitting a regressed conditional mean from the
same cells' features* (x, y, z, cell type, local morphology)?

Read these before reading the rows:

- **The baseline is already a two-donor per-gene chimera** (§2). The ablation
  replaces the gene-mix too, so a `v18_*` row compares against "two-donor
  recombination", not "one copied cell".
- **The target scale mirrors v18's `raw_ok`.** With raw data present the learner
  trains on raw values, and its negative predictions are **clipped to 0**. That
  clip *manufactures zeros*, so a `v18_*` `paper_gene_detection_spearman` is not a
  learner recovering sparsity. `--target-scale log` is the cross-check.
  `method_params` records which scale ran.
- **`lasso`'s penalty is a fraction of `alpha_max`** (`_ml_learners.FracAlphaLasso`),
  because scikit-learn's default collapses the fit to the per-gene mean. An
  all-zero fit is reported at run time.
- **`*_gbmfield` / `*_lgbm{field,repair,band,balance}` are donor variants.** The
  learner's prediction is a *target*, and each cell emits the real local same-type
  profile closest to it. They change neither coordinates nor cell types. So any
  movement in `paper_celltype_localization` is a pose effect via the aligner
  (§4), not better placement (`config.py`, comment above `_GBMFIELD_HOSTS`).
- The registry comments in `config.py` are the design record for each variant,
  including measured trade-offs (e.g. the per-gene repair costs Moran's MAE in
  every configuration tried).

### Ablation run times (measured, STARmap, 4-core CPU)

Method step only (`resources.json`); scoring adds ~2 min per row.

ABLATION_TABLE_PLACEHOLDER

## 6. What was changed from the published tree

Cut from the parent project at `08c385ce` (2026-09-23), trimmed by deletion, then
edited in two ways: the harness changes this review needed, and a rename that
makes SpatialCPA-v18 a standalone method with no trace of the versions it
descends from. **Neither changes a single emitted number**, and both are proven,
not asserted.

### Harness changes

| file | change | affects numbers? |
|---|---|---|
| `benchmark-pbya-v3/src/bench3/config.py` | METHODS / METHOD_ORDER / `_GBMFIELD_HOSTS` cut to v18 + the three comparators; `V18_ARGS` added and applied to every v18-hosted entry | no: scorer constants unchanged (tested); the evaluated METHODS equal the published ones plus `V18_ARGS`, apart from `notes` and the log-marker prefix. It changes *what runs by default*, which is the point |
| `benchmark-pbya-v2/src/benchmark/config.py` | METHODS cut to spatialz/feast/isost | no (constants tested) |
| `benchmark-pbya-v3/src/bench3/methods/run_spatialcpav18.py` | v18 lookup stops at the repo root; loaded file's sha256 and `temp`/`device` logged | no — resolution and log text only |
| `benchmark-pbya-v3/src/bench3/run_benchmark.py` | optional `$BENCH_V3_PYTHON` launcher | no: unset = published behaviour |
| `run_all.py`, `plot_*.py` | usage examples | no |

### The standalone rename

| was | is | where |
|---|---|---|
| class `SpatialCPAv14` | `SpatialCPAv18` | `learn_spatialcpav18.py`, both wrappers, the probe |
| class `V14Config` | `V18Config` | same |
| log prefix `[v14] …` | `[v18] …` | `learn_spatialcpav18.py`; the matching `invalid_log_markers` in `config.py` |
| docstrings and comments describing v18 as "v14 + fixes", citing a `spatialcpav14/` package, or pointing at sibling versions' code | v18 described on its own terms | `learn_spatialcpav18.py`, both wrappers, `_ml_learners.py`, `config.py`, and comments in `design.py`, `assets.py`, v2's `evaluate_generation.py` and `leakage_guard.py` |

**Proof 1, mechanical.** `scripts/verify_rename.py ORIGINAL RENAMED` parses both
files, drops docstrings and comments, maps the two class names and the `[v14]`
literal in the original, and masks only `print(...)` text and `help=`/
`description=` strings. Then it requires the two ASTs to be **identical**. Every
constant, default, choice string, call, argument and branch must match. All seven
renamed files pass against the parent project's originals: `learn_spatialcpav18.py`,
`run_spatialcpav18_ml.py`, `_ml_learners.py`, `design.py`, `assets.py`,
`evaluate_generation.py`, `leakage_guard.py`. `run_spatialcpav18.py` passes against
its post-path-fix version; against the parent's copy, it differs by exactly that
fix. `config.py` is checked by evaluating `METHODS`, since its `notes` text
changed. The original hashes are
in `MANIFEST.original.sha256`, and the check re-runs against any original, including
the lab's (§7).

**Proof 2, empirical.** The STARmap row re-run on the renamed code produces a
`prediction.h5` **bitwise identical** to the one committed before the rename
(`expected/cpu-verified/…/prediction.h5`, unchanged), and all 50 metrics are
equal at tolerance 0. Only `method_log.txt` was re-committed: its text now reads
`[v18]` and prints `temp`/`device`.

Byte-identical to the parent, untouched: `evaluate_paper.py`, `align.py`,
`_v2bridge.py`, `prepare_dataset.py`, `sources.py`, `evaluate_all.py`,
`aggregate_results.py`, `rank_methods.py`, `_field_fixture.py` (the synthetic
fixture the ablation's registry comments were measured on), v2's `evaluate.py` /
`_v2_io.py` / `resource_monitor.py` and the comparator wrappers.

**What stripping the other versions broke: nothing that v18 or the comparators
use.** `_ml_learners.py` was shared with other versions' ablations and is kept
whole. `_GBMFIELD_HOSTS` holds v18 alone. The upstream v3 README is replaced by
`docs/`. Some ablation comments in `config.py` cite `reports/ml_ablation_*.md`
from the parent project; those reports are not carried, because they analyse
several versions together.

## 7. Open provenance items (need the lab machine)

`scripts/export_lab_envs.sh` collects all of these in one run:

1. **The published row itself.** Copy
   `results/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/{metrics.json,prediction.h5,method_log.txt,resources.json}`
   into `expected/published/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/`,
   then run `make manifest`. `make starmap-row` then compares against it
   automatically.
2. **Is this the v18 that ran?** The published runs used
   `/data/han/projects/Spatial3D/src/learn_spatialcpav18.py`. This repo's file is
   the parent project's root copy (sha256 `cf89792a…`, in
   `MANIFEST.original.sha256`), renamed (§6). If `provenance.txt` shows the same
   hash, it is the same file. If it doesn't, run
   `python scripts/verify_rename.py <lab file> learn_spatialcpav18.py`. EQUIVALENT
   means this repo computes exactly what the lab file did. Anything else means the
   lab file is the one to ship.
3. The same for `evaluate_paper.py` (hash must match exactly: it is not renamed),
   `align.py`, and the two v18 wrappers (via `verify_rename.py`).
4. **Exact environments** (`envs/lock/*.lab.*`). The `.yml` files here are ranges.
5. **The comparators' code**: the isoST commit (`fetch_tools.sh` defaults to HEAD
   as of 2026-09-24, `805981c3`), and whether the Zenodo SpatialZ code equals the
   `reference/SpatialZ.py` hashed in `MANIFEST.tools.sha256`.
6. **Device.** Whether the published v18 rows trained on GPU. A published
   prediction's method log shows `torch … (cuda|cpu)`.
