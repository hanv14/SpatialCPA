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
| `--edit-weight` | `0.0` | `0.25` | At 0 the decoded-flow blend (`learn_spatialcpav18.py:1254-1256`) is skipped, **and** it is the gate for v18's two headline mechanisms: gene-mix (`:1260`, `gene_mix_frac > 0 and edit_weight == 0`) and the raw-output path (`:1275`, `raw_output and edit_weight == 0`). At the default 0.25 neither runs, and v18 emits `expm1` of a 75/25 blend of a real profile and a PCA decode. |
| `--ground-blend-flow` | `1.0` | `0.2` | Every generated cell is offered for flow re-grounding in `_ground` (`:1291`), not a random 20 %. |

In the parent project the `spatialcpav18_gen` registry entry carried **no
`wrapper_args`**, so these flags existed only on the command line of each run.
A bare `run_all --methods spatialcpav18_gen` there runs the *default*
configuration, where gene-mix and raw output are dead code.

This repo pins them (`config.V18_ARGS`), but the question for the published rows
remains: **were all of them run with the extras?** The prediction records it.
Every published `prediction.h5` should carry `uns/method_params` with
`"edit_weight": 0.0` and `"ground_blend_flow": 1.0`. That is the first thing to
check in the lab `results/` tree (§6).

`--ground-temp` is **not** written to `method_params`, and neither is `--device`.
A published row's temperature can only be read from its logged command line
(`run_all`'s stdout, `cmd: …`). The method log's config line doesn't print it
either.

## 2. What v18 actually emits under these flags

Read `_generate` (`learn_spatialcpav18.py:1157-1289`) with the flags above in mind:

- **Positions are resampled real positions.** `_resample_layout` (`:1135`) draws
  `n_target` coordinates from the two flanking training sections, in coherent
  patches, plus 5 %-of-spacing jitter. `n_target` interpolates the flanks' cell
  counts, so `paper_cell_count_ratio ≈ 1` (0.996 measured) follows from the
  construction rather than being learned.
- **Every emitted expression value is a verbatim training measurement.** With
  `edit_weight == 0` no decoded expression reaches the output. With raw
  measurements present (STARmap: `expression_type=raw_counts`), `raw_ok` holds and
  the output is `pool_raw[pick]` with the gene-mix re-applied on the raw scale
  (`:1275-1281`).
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
neighbourhood. The accurate description of v18 at the published flags, now used
throughout this repository, is **retrieval-based virtual-slice synthesis with a
learned query**: real cells are retrieved, and the flow-matching model only ranks
them. It is not generative expression synthesis, and the paper's wording should
match. Whether the learned query improves on keeping the layout's cell (a
"no-flow" ablation) is the open question for the paper's claims. The nominal `gene_mix_frac = 0.15` yields ~12 % mixed genes, which
suggests some cells find no same-type partner (`_gene_mix`, `:1400`). Worth a
look.

## 3. Reproducibility — measured

- **The STARmap row runs from a fresh clone with no environment variables**:
  `make starmap-row`, 2 min 48 s on 4 CPU cores. v18 itself takes 36 s and
  986 MB peak RSS; the rest is scoring.
- **Deterministic.** Every independent run gave a bitwise-identical `prediction.h5`
  (all of X, obs, method_params; only `uns/wall_time_seconds` differs). That
  holds across runs before and after each restructuring of this tree (§5), and
  across two different machines.
- **The scorer is deterministic up to floating-point rounding.** On one machine
  with one thread count, all 50 scalar metrics were identical at tolerance 0,
  UMAP included. Re-scoring the *same* prediction on another CPU, or with 1
  instead of 4 BLAS threads, moved exactly two metrics:
  `paper_rare_celltype_localization` (≤ 2.2 × 10⁻¹⁶) and `paper_celltype_ot`
  (1.4 × 10⁻¹⁷). Both come from the Sinkhorn optimal-transport computation, whose
  summation order follows the BLAS kernel and thread count. That is why
  `compare_metrics.py` defaults to `--atol 1e-6`; any real difference is many
  orders of magnitude larger.
- **Not yet compared with the published row.** `expected/published/` is empty
  until the lab files arrive (§6). Two things can legitimately make them differ:
  - `--device auto` picks CUDA when present, and GPU flow training isn't bitwise
    equal to CPU;
  - UMAP (`paper_umap_*`) moves with umap-learn/numba versions.

  `scripts/compare_metrics.py` holds the UMAP pair to a separate tolerance.
  `paper_embedding_mixing_pca` is the deterministic stand-in for the same
  question.

## 4. The scorer — where a number could move without the method changing

- **`evaluate_paper.py` is pinned** (sha256 `7362669200bb…9538992`, README). Its
  dependencies are pinned in `MANIFEST.sha256`, and every file's relation to its
  parent-project original is recorded in `MANIFEST.original.sha256` (§5). The
  two config files were edited (registry and paths only), and their constants are
  asserted in `tests/test_pins.py`.
- **Pose.** `paper_marker_field_r` and `paper_celltype_localization` are computed
  after `align.align_by_expression` (`align.py`), which picks the rotation by
  binned-marker-field agreement. Anything that smooths the marker field can
  improve the chosen pose, and so these two scores, without placing cells better.
  `metrics.json` records `align_rotation_deg`, `align_score` and
  `align_runner_up` per section; a marginal pose is visible there.
- **Scale fairness is rank-normalization.** All primary metrics use per-gene
  rank-normalized expression, from `src/benchmark/evaluate_generation.py`. So
  v18's raw-output fix cannot help them directly. It shows only in
  `paper_gene_var_spearman`, and in `paper_gene_detection_spearman`, which is
  computed on raw emitted values.
- **Not ranked:** `paper_cell_count_ratio` (by construction ≈ 1 here, §2) and
  `paper_rare_celltype_recall`.
- **The `gen_*` and cell-matched blocks** are in `metrics.json` as well. The
  cell-matched ones (`pearson_median`, `celltype_accuracy`, …) are
  reference only: generation has no cell correspondence.
- The harness's own discrimination check (`make selftest`: oracle / flanking copy
  / spatial scramble / random) passes on this tree (measured, 4 min 20 s).
  `flanking_copy` is the natural baseline for a method that copies flanking
  cells; see `benchmark/results/summary/selftest_metrics.json` after running it.

## 5. What was changed from the published tree

Cut from the parent project at `08c385ce` (2026-09-23) and then reorganised in
three ways. **None of them changes a single emitted number**, and each is proven,
not asserted.

1. **Scope.** Only SpatialCPA-v18 and the three comparators remain: other
   SpatialCPA versions and v18's `ml` expression ablation are gone, along with
   their wrappers, registry entries, learners, fixtures and environment
   packages.
2. **One folder.** The parent project's three benchmark folders were merged into
   `benchmark/`, following the v3 layout:

   | was | is |
   |---|---|
   | `benchmark-pbya-v3/src/bench3/` (harness, scorer) | `benchmark/src/bench3/` |
   | `benchmark-pbya-v2/src/benchmark/` (evaluators, leakage guard) | `benchmark/src/benchmark/` |
   | `benchmark-pbya-v2/src/benchmark/methods/` (comparator wrappers, `_v2_io`) | `benchmark/src/bench3/methods/`, beside v18's wrapper |
   | `benchmark-pbya/src/data/` (download/process scripts) | `benchmark/src/data/` |
   | `benchmark-pbya/tools/` | `benchmark/tools/` |
   | `benchmark-pbya/data/{raw,processed}/` | `benchmark/data/{raw,processed}/` |
   | `benchmark-pbya-v3/data/processed/` (the built paper-protocol datasets) | `benchmark/data/sections/`, so a build never shares a path with its source |
   | `data/starmap/…h5ad` | `benchmark/data/raw/starmap_visual_cortex/…h5ad` |

   Run commands are unchanged apart from `cd benchmark`:
   `python -m src.bench3.<module>`.
3. **Standalone naming.** Every reference to other SpatialCPA versions, and to the
   old folder names, was removed from code, comments, docstrings and messages.
   v18's classes are now `SpatialCPAv18` / `V18Config`, and its log prefix is
   `[v18]`.

   Two files were deliberately left **byte-identical**, docstrings included:
   - `evaluate_paper.py`, so its published sha256 still identifies it;
   - `align.py`.

   Their docstrings still say `benchmark-pbya-v3`. Two names were kept for the
   same reason:
   - the evaluators' package is still imported as `benchmark`;
   - the bridge module is still `_v2bridge`.

   `evaluate_paper.py` imports both.

### How "no number moved" is proven

**Per file.** `MANIFEST.original.sha256` lists all 70 files that came from the
parent project, with the original's path and sha256:

- **40 `identical`**: byte-for-byte the original. This includes
  `evaluate_paper.py`, `align.py`, `resource_monitor.py`, `sanitize_checkpoint.py`
  and every dataset script.
- **21 `equivalent`**: `scripts/verify_rename.py` parses both files and shows the
  ASTs are identical once docstrings, comments, message text (`print`, `raise`,
  `warnings.warn`, `parser.error`, `help=`/`description=`/`epilog=`) and v18's two
  class names are set aside. Every constant, default, choice string, call,
  argument and branch must match. This includes `learn_spatialcpav18.py`,
  `evaluate.py`, `evaluate_generation.py`, `leakage_guard.py`, `sources.py`,
  `prepare_dataset.py`, `design.py`, `rank_methods.py` and `aggregate_results.py`.
- **9 `changed`**, each an executable change outside the computation:

| file | change |
|---|---|
| `src/bench3/config.py` | paths for the one-folder layout (`DATA_ROOT`, `data/sections/`); METHODS cut to the four methods; `V18_ARGS` pinned on `spatialcpav18_gen`; `[v18]` fallback markers. Scorer constants unchanged (tested); the evaluated METHODS equal the published ones plus `V18_ARGS`, apart from `notes` |
| `src/benchmark/config.py` | cut to the four constants `evaluate.py` imports, with values unchanged (tested against the original) |
| `src/bench3/_v2bridge.py` | finds `src/benchmark` at its new place |
| `src/bench3/methods/run_spatialcpav18.py` | v18 lookup stops at the repo root; sha256, `temp` and `device` logged; finds `_v2_io`/`leakage_guard` at their new place |
| `src/bench3/methods/run_spatialz.py`, `run_isost.py` | `TOOLS_DIR` → `benchmark/tools/` |
| `src/bench3/run_benchmark.py` | optional `$BENCH_V3_PYTHON` launcher (unset = published behaviour); one error string |
| `src/bench3/selftest.py` | finds `_v2_io` at its new place |
| `src/bench3/survey_datasets.py` | `--root` default → `benchmark/` |

`tests/test_rename.py` re-checks every `identical` and `equivalent` row against
the originals whenever they sit beside this tree. It also fails if the set of
`changed` files ever differs from these nine.

**End to end.** After each restructuring, the STARmap row was re-run in this
tree:
- after the standalone rename;
- after the merge into `benchmark/` together with the `ml` removal, in the
  environment with the ablation packages uninstalled.

Each `prediction.h5` was **bitwise identical** to the committed
`expected/cpu-verified/…/prediction.h5`, which has not been re-committed since it
was first produced, and all 50 metrics matched within `1e-6` (bitwise, except
for the last-bit Sinkhorn rounding described in §3 once this session moved to a
different machine). The harness
selftest exercises the scorer on four synthetic reconstructions of known
quality.

Selftest, measured (4-core CPU, 4 min): **all checks passed**, and all 172
numeric values in `selftest_metrics.json` (oracle / flanking_copy /
spatial_scramble / random × every metric) are **identical** to the run made
before the merge, in the parent layout.

## 6. Open provenance items (need the lab machine)

`scripts/export_lab_envs.sh` collects all of these in one run:

1. **The published row itself.** Copy
   `results/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/{metrics.json,prediction.h5,method_log.txt,resources.json}`
   into `expected/published/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/`,
   then run `make manifest`. `make starmap-row` then compares against it
   automatically.
2. **Is this the v18 that ran?** The published runs used
   `/data/han/projects/Spatial3D/src/learn_spatialcpav18.py`. This repo's file is
   the parent project's root copy (sha256 `cf89792a…`, in
   `MANIFEST.original.sha256`), renamed (§5). If `provenance.txt` shows the same
   hash, it is the same file. If it doesn't, run
   `python scripts/verify_rename.py <lab file> learn_spatialcpav18.py`. EQUIVALENT
   means this repo computes exactly what the lab file did. Anything else means the
   lab file is the one to ship.
3. The same for `evaluate_paper.py` and `align.py` (hashes must match exactly:
   neither was touched) and for the v18 wrapper (via `verify_rename.py`; it differs
   from the published wrapper only by the lookup fix).
4. **Exact environments** (`envs/lock/*.lab.*`). The `.yml` files here are ranges.
5. **The comparators' code**: the isoST commit (`fetch_tools.sh` defaults to HEAD
   as of 2026-09-24, `805981c3`), and whether the Zenodo SpatialZ code equals the
   `reference/SpatialZ.py` hashed in `MANIFEST.tools.sha256`.
6. **Device.** Whether the published v18 rows trained on GPU. A published
   prediction's method log shows `torch … (cuda|cpu)`.

## 7. Experimental: `spatialcpav18_gen_flow` (flow-chosen flank)

**What it is.** `benchmark/src/bench3/methods/run_spatialcpav18_flow.py` subclasses
v18's class from the unchanged `learn_spatialcpav18.py`, overriding four methods.
The flow is used for **one** decision: which flanking section to retrieve the
output section from. The flow is queried at each flank's own cell positions and the
target depth, and the flank whose real cells match its prediction better (lower
mean latent distance) is copied, at its exact positions, as in v18's `nearest`
layout. Donor reranking, the type-vote replacement and the composition replacement
use each cell's own source cell as the query, as in the "no flow" ablation. Flags:
`V18_ARGS` plus `--flank-select flow` (or `rule` / `lower` / `upper` for
diagnostics).

**Identity with "nearest + no flow", proven.** `--flank-select rule` (v18's rule:
the lower flank when `t ≤ 0.5`) was run against an independently built reference:
the unchanged `learn_spatialcpav18.py` with the 5-line `STACK3D_AUDIT_NOFLOW`
switch inserted before `_ground`, run with `--position-mode nearest`. All 11 X /
obs / var arrays of the two `prediction.h5` files are **bitwise identical**. So this
method can differ from "nearest + no flow" only through the flank the flow picks.

**STARmap `paper_2_4_6`, measured (4-core CPU, seed 42).** The flow chose the
**lower** flank for all three sections (latent distance lower vs upper: 4.46 vs
5.04, 5.05 vs 5.18, 5.19 vs 5.33), the same as the rule. The output is therefore
identical to "nearest + no flow":

| metric | published v18 | `spatialcpav18_gen_flow` (= nearest + no flow here) | forced upper flank |
|---|---|---|---|
| UMAP mixing | 0.926 | 0.948 | 0.973 |
| Moran's I r / MAE | 0.980 / 0.034 | 0.981 / 0.024 | 0.973 / 0.042 |
| marker depth r / field r | 0.850 / 0.870 | 0.945 / 0.874 | 0.967 / 0.883 |
| cell-type / rare-type localization | 0.780 / 0.626 | 0.754 / 0.619 | 0.814 / 0.699 |
| gene detection ρ | 0.860 | 0.973 | 0.856 |

**What this shows, and what it does not.**
- The flank matters a lot, and neither flank wins everything. Per section, the
  upper flank is better on most metrics for sections 2 and 6 and mixed for 4,
  while the lower flank is better on Moran's I and detection.
- The flow's criterion chose the lower flank every time, so on this dataset it
  **adds nothing** over the fixed rule. One dataset and one seed say nothing about
  its value elsewhere, where flanks are unequally spaced or genuinely differ.
- The "forced upper" column is a **diagnostic, not a method**. It was chosen
  after seeing the test sections, so it must not be reported as a result.
- A defensible way to choose a flank must not look at the targets. One candidate
  is internal validation: on interior *training* sections, measure whether copying
  the section above or below reproduces them better, and use that to choose for
  the held-out ones. That is the next thing to test, across datasets and seeds.

### 7a. `spatialcpav18_gen_flow_cv` (the flow's choice, validated on training sections)

**What it is.** The same wrapper under `--flank-select flow-cv`. The flow still
scores both flanks of every held-out section and makes the choice. What changes is
the threshold it must clear before leaving v18's rule flank, which is fixed by
internal validation on the training sections only:

1. **Folds.** Each interior training section i (both neighbours are training
   sections; at most 8, evenly spaced) is left out, and a fresh model is trained on
   the rest. That fold's flow scores i's two flanks at z_i, giving
   `margin = d(rule flank) − d(other flank)`.
2. **Truth on the fold.** The fold model synthesizes section i from each flank, and
   both are scored against the real section i with `evaluate_paper`'s own
   per-section functions, called as `evaluate_paper()` calls them, with UMAP off.
   `gain` = wins − losses of the other flank on 8 metrics fixed in advance: PCA
   embedding mixing, Moran's I r and MAE, marker depth r and field r, cell-type and
   rare-type localization, and gene-detection ρ. Differences ≤ 1e-9 are ties, so
   Sinkhorn last-bit noise cannot decide a fold.
3. **Threshold.** `delta` is the largest value maximizing the total gain of the
   folds with `margin > delta`. Never switching (`delta = +inf`, gain 0) wins every
   tie, so a switch needs strictly positive validated gain.
4. **Held-out.** The flow takes the other flank iff `margin > delta`.

The input file holds training sections only (`guard_no_holdout`), and every fold
target is one of them. Fold models are trained before the main one, and v18
re-seeds torch at the start of every training run, so the main model is the same
as under `flow`. Each flow-cv flank score draws its noise from its own generator
seeded with `--seed`, so a decision depends only on the model and z\*. Under the
shared stream, copying a different flank for an earlier section changed the cell
count, and with it the noise later scores saw. Cost on STARmap: two extra
trainings plus four scorings, about 3 min on 4 CPU cores.

**STARmap `paper_2_4_6` (seed 42).** Two folds:

| fold | margin | upper vs lower on the fold |
|---|---|---|
| section_3 | −0.709 | gain +6: upper better on 7 of 8, worse only on Moran's MAE |
| section_5 | −0.324 | gain 0: 4 wins, 4 losses |

`delta = −0.709` (validated gain +6). All three held-out margins (−0.585, −0.138,
−0.138) clear it, so the flow switched to the **upper** flank for every section.
The X / obs arrays are bitwise identical to the forced-upper diagnostic, and to a
second flow-cv run. Only the logged float32 distances move, at about 1e-6, and
every held-out margin is at least 0.12 from `delta`. This time the choice was made
from training sections only:

| metric | `spatialcpav18_gen_flow` (= nearest + no flow) | `spatialcpav18_gen_flow_cv` |
|---|---|---|
| UMAP mixing / PCA mixing | 0.948 / 0.919 | **0.973 / 0.950** |
| Moran's I r / MAE | **0.981 / 0.024** | 0.973 / 0.042 |
| marker depth r / field r | 0.945 / 0.874 | **0.967 / 0.883** |
| cell-type / rare-type localization | 0.754 / 0.619 | **0.814 / 0.699** |
| gene detection ρ | **0.973** | 0.856 |

On the pre-registered 8-metric composite the held-out result is +2 for flow-cv
(5 wins, 3 losses), the same direction as the folds predicted.

**Read this carefully:**
- **Uniform choice.** Every held-out margin cleared `delta`, so the flow's ranking
  did not separate the sections. The switch rests mainly on one fold
  (section_3).
- **Thin validation.** STARmap gives 2 folds, and they are at a 22 µm neighbour
  gap, while each target is 11 µm from its flanks.
- **Trade-off.** flow-cv gains continuity and localization and gives up Moran's I
  level and detection. Whether that counts as "better" is the composite's
  verdict, which was fixed before the run.
- **Next.** Run it across datasets and seeds before any claim. The composite and
  the fold rule are fixed in `CV_METRICS` / `calibrate_delta`, and must not be
  tuned against held-out results.

### 7b. `spatialcpav18_gen_flow_patch` (flank chosen per spatial patch) and its random control

**What it is.** The same wrapper under `--flank-select patch-cv`. The flank is
chosen per spatial patch instead of per section:
- **Patches.** Both flanks' cells go on one square grid, with side 8 median cell
  spacings (about 270–300 patches per STARmap section).
- **Margin.** The flow is queried at every cell of both flanks at z\*. A patch's
  margin is the rule flank's mean latent distance there minus the other flank's.
- **Switching.** The top fraction q of patches by margin are copied from the other
  flank, cells at their exact positions. Whole patches move, so neighbourhoods stay
  intact, unlike the published per-cell reranking.
- **Who decides what.** The flow decides which patches; training folds decide how
  many. Each fold synthesizes its left-out section at q ∈ {0, 0.1, 0.25, 0.5, 0.75,
  1} and scores it on the flow-cv metrics against q = 0. q = 0 wins ties.
- **Control.** `spatialcpav18_gen_flow_patch_random` (`--patch-rank random`) is
  calibrated the same way, but ranks patches in a seeded random order. The gap
  between the two is what the flow's ranking adds.

**Checks.** `--patch-q 0` (forced, no folds) is bitwise identical to "nearest + no
flow". After the refactor this mode needed, `flow` is still bitwise identical to
its earlier run, and `flow-cv`'s arrays are too.

**STARmap `paper_2_4_6` (seed 42): negative result.** Fold gain against q = 0,
summed over the two folds:

| q | 0.1 | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|---|
| flow-ranked patches | −6 | −6 | −6 | −2 | +4 |
| random patches (control) | −4 | −2 | −6 | −2 | +4 |

- **Mixing patches hurts.** Every partial switch loses on the folds. Seams between
  two sections cost more than any patch choice gains.
- **The flow's ranking is no better than random**, slightly worse at q = 0.1 and
  0.25. On this dataset the per-patch margins carry no usable signal.
- **Both settle on q = 1, and the outputs are identical.** At q = 1 all ranked
  patches switch, about 95 % of cells. The 4–5 % of rule-flank cells left (patches
  where the other flank has under 5 cells, mostly at the tissue edge) cost
  accuracy against whole-section flow-cv:

| metric | nearest + no flow | flow-cv | patch-cv (q = 1) |
|---|---|---|---|
| UMAP / PCA mixing | 0.948 / 0.919 | **0.973 / 0.950** | 0.962 / 0.944 |
| Moran's I r / MAE | **0.981 / 0.024** | 0.973 / 0.042 | 0.970 / 0.045 |
| marker depth r / field r | 0.944 / 0.874 | **0.967 / 0.883** | 0.888 / 0.880 |
| cell-type / rare-type localization | 0.754 / 0.619 | **0.814 / 0.699** | 0.804 / 0.676 |
| gene detection ρ | **0.973** | 0.856 | 0.857 |

So on STARmap the flow does not earn a patch-level role. If the flow ranking beats
the random control on other datasets, especially ones where flanks differ within
the section (damage, uneven coverage, wider gaps), that would be evidence for the
flow. Run it there before drawing a conclusion either way. Both registry entries
are kept so that comparison can be run as is.
