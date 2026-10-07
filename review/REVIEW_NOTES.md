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

### 7c. Transport of retrieved cells (`spatialcpav18_gen_flow_transport*`) and the STARmap gap sweep

**What it is.** `--flank-select transport-cv`. Retrieval decides what each output
cell is: v18's nearest layout, with the rule flank's real cells and real
expression. A transport decides where the cell goes: it is moved by λ × a
displacement from its section's depth to z\*.

| method | displacement | role |
|---|---|---|
| `spatialcpav18_gen_nearest_noflow` | none | baseline (bitwise "nearest + no flow") |
| `_flow_transport` | learned velocity field v(x, y, z), trained by flow matching on exact-OT-matched cells of every consecutive pair of training sections; integrated from the source depth to z\* | the method |
| `_flow_transport_ot` | exact OT (assignment) between the two flanks, moved the fraction (z\*−z_src)/(z_far−z_src) of the way | control: is learning needed? |
| `_flow_transport_zshuffle` | the field, with each section pair's depth interval rotated to the next pair's | negative control: does it use depth? |
| `_flow_transport_pair` | the field, trained on the two flanks only | negative control: does the wider stack matter? |

λ ∈ {0, 0.25, 0.5, 0.75, 1} is chosen per method by the flow-cv folds. The fold
model and the transport are both trained without the left-out section. λ = 0
wins ties, and with no interior training section (block 5) there are no folds,
so λ stays 0. λ = 0 is bitwise "nearest + no flow" (checked).

**Sweep.** `scripts/gap_sweep_starmap.sh` runs the existing designs; there is no
change to `prepare_dataset`. The designs are `paper` (hold out 2, 4, 6) and
`wide` blocks 1, 3 and 5 (hold out 4; 3–5; 2–6), so the farthest target is 11,
11, 22 and 33 µm from a real section. A second pass forces λ = 1 for the flow and
for OT, as a diagnostic of raw transport. `scripts/summarize_gap_sweep.py` writes
the tables.

**Result (seed 42): the transport does not help, and the learned field does not
beat OT.** Composite against "nearest + no flow" (wins − losses over the 8 pooled
metrics; 0 means the same output):

| gap | flow | OT | flow, depth rotated | flow, two flanks only | λ chosen (flow / OT / rot. / pair) |
|---|---|---|---|---|---|
| 11 µm, alternate | 0 | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
| 11 µm, block 1 | 0 | 0 | 0 | 0 | 0 / 0 / 0 / 0 |
| 22 µm, block 3 | **−6** | 0 | 0 | −4 | 0.25 / 0 / 0 / 0.25 |
| 33 µm, block 5 | 0 | 0 | 0 | 0 | no folds: 0 everywhere |

Forced λ = 1 (diagnostic, not a method):

| gap | flow | OT |
|---|---|---|
| 11 µm, alternate | −6 | −2 |
| 11 µm, block 1 | −4 | −5 |
| 22 µm, block 3 | −4 | −2 |
| 33 µm, block 5 | 0 | −2 |

- **Moving retrieved cells makes them worse at every gap.** Forced full transport
  loses at every gap except one tie. The folds saw this and kept λ = 0 in 14 of 16
  calibrations.
- **Where the folds chose λ = 0.25** (22 µm, flow and flow-pair; fold gains of
  only +1 and +2), the held-out sections lost (−6 and −4). The folds are too few
  (2) to separate a small real effect from noise.
- **No evidence the learned field is better than classical OT.** The flow loses to
  OT at 3 of 4 gaps when forced and matches it once (33 µm: 0 vs −2, one dataset,
  one seed). The negative controls do no worse than the flow. So nothing here
  shows that the field's use of depth or of the wider stack adds anything.
- **Likely cause.** Neighbouring sections don't contain the same cells. An exact
  assignment between two sections therefore produces mostly matching noise
  (mean shifts of 8–25 µm, comparable to the 11–22 µm section spacing), not
  tissue deformation. A field fitted to that noise moves cells away from where
  real cells are. STARmap's sections are also already well aligned, so there is
  little real deformation to find.

**Conclusion for the paper.** On STARmap the transport adds nothing, and the
evidence does not support claiming that a learned flow improves the method.
Datasets with real deformation across depth (developing tissue, curved
structures, thick serial sections), on more seeds, are where this could still
show something. The sweep and summary scripts run unchanged there. A transport
learned from matched *cell identities* (expression-aware coupling) rather than
positions alone would be the next design to try.

## 8. Vascular-architecture metrics (`evaluate_vascular.py`, evaluation-only)

**What it is.** A new scorer module, `benchmark/src/bench3/evaluate_vascular.py`,
run over existing predictions. It reruns no method and no `prepare_dataset`, and
it doesn't touch the pinned `evaluate_paper.py` or `metrics.json`. It writes
`vascular_metrics.json` per prediction. It measures how vascular cells are
organized, not whether vessels are functional; expression can't show perfusion.

**Which cells are vascular.** One rule, applied identically to prediction and GT,
and fixed by the dataset alone:
- **Markers:** declared per dataset in `VASCULAR_SPEC`. STARmap: `Flt1`.
- **Vascular fraction f:** the fraction of the dataset's most marker-enriched cell
  type. STARmap: cluster 8, 4.7 % of cells, mean Flt1 rank 0.97 against 0.81 for
  the next cluster.
- **Per section:** the vascular cells are the top f fraction by marker rank.
  Detection can't be used, because Flt1 is nonzero in ~100 % of STARmap cells.
- **Consequence:** the rank rule scores placement, not density. Density is scored
  only by the cell-type rule (`vasc_type_frac_ratio`).

**Metrics.** All are calibrated as `1 − error / error(null)`. The null is the GT
section with its vascular labels moved to random cells, so 1 = real organization
and 0 = vessels scattered at random. Distances are in units of the section's own
cell spacing.
- `vasc_dist`: distance from each cell to its nearest vessel cell.
- `vasc_nn`: vascular clustering / chaining.
- `vasc_niche`: the expression of the non-vascular cells next to vessels.
- `vasc_field_r`: the vascular density field, posed as the paper metrics are.
- `vasc_type_dist`: per cell type, distance to the vascular type.
- `vasc_type_frac_ratio`: predicted vascular-type fraction / GT fraction.

`ref_neighbour_*` scores the nearest real input section as if it were the
prediction.

**Validated before use.**
- **Synthetic section (`tests/test_evaluate_vascular.py`).** The GT scores
  exactly 1. log1p×3 output scores the same. A spatial scramble scores ~0. A
  different section of the same tissue scores in between.
- **Real STARmap section 4.** Oracle 1.00 on all; scramble −0.09 to 0.14 on every
  calibrated metric.
- **One design fix made during validation.** An all-cell niche is dominated by
  vascular-vascular neighbours, because vessels chain. So the niche uses
  non-vascular neighbours only. The Pearson form (`vasc_niche_r`) swings ±0.4
  under a scramble, so `vasc_niche`, which uses the L2 error against the null, is
  the primary score.

**First reading on existing STARmap results (seed 42; no reruns).**

| hold-out | method | dist | nn | niche | field r | type dist |
|---|---|---|---|---|---|---|
| paper 2,4,6 | published v18 | **0.855** | 0.875 | **0.889** | 0.829 | 0.734 |
| paper 2,4,6 | nearest + no flow | 0.831 | **0.879** | 0.855 | 0.869 | 0.647 |
| paper 2,4,6 | flow-cv (upper flank) | 0.739 | 0.815 | 0.861 | 0.849 | **0.748** |
| paper 2,4,6 | patch-cv | 0.807 | 0.837 | 0.877 | 0.858 | 0.721 |
| paper 2,4,6 | copy nearest section (ref.) | 0.790 | 0.820 | 0.854 | **0.897** | 0.706 |
| wide 4 (11 µm) | nearest + no flow | 0.775 | 0.894 | 0.856 | 0.921 | 0.793 |
| wide 4 | copy nearest section (ref.) | 0.872 | 0.804 | 0.870 | 0.931 | 0.845 |
| wide 3–5 (22 µm) | nearest + no flow | 0.635 | 0.823 | 0.807 | 0.874 | 0.702 |
| wide 3–5 | flow transport (λ 0.25) | 0.606 | 0.809 | 0.780 | 0.848 | 0.702 |
| wide 3–5 | copy nearest section (ref.) | 0.854 | 0.867 | 0.856 | 0.910 | 0.768 |
| wide 2–6 (33 µm) | nearest + no flow | 0.479 | 0.691 | 0.695 | 0.741 | 0.768 |
| wide 2–6 | copy nearest section (ref.) | 0.698 | 0.835 | 0.749 | 0.839 | 0.739 |

- **The metrics track difficulty.** `vasc_dist` for nearest + no flow falls from
  0.83 to 0.48 as the gap widens. They also separate methods the paper metrics
  rank differently: flow-cv's upper flank, which wins most paper metrics, loses on
  vessel distance and clustering.
- **v18's post-copy steps hurt at wide gaps.** At the wide gaps, copying the
  nearest real section beats "nearest + no flow" on vessel distance by 0.10–0.22,
  with the same source sections. The only differences are v18's post-copy steps:
  cell subsampling, the type vote, composition matching and the 15 % gene mix,
  which can move Flt1 between cells. That is worth a targeted ablation before it
  is read as a finding.
- **Scope of this reading.** It covers one dataset, one seed and one marker
  (Flt1). Treat it as a first reading, not a result.

## 9. Grounding in the flow's joint latent h (`spatialcpav18_gen_flow_h*`)

**What it is.** `--flank-select h-cv`. v18 samples a 48-d joint latent `h_star`,
covering expression, neighbourhood and type, and then keeps only its expression
part ê for grounding. The type head is evaluated and never read, and the
neighbourhood decoder is never called at inference. This mode re-grounds v18's
nearest layout with v18's own `_ground`, unchanged, but in h-space:
- **Pool:** the flanks' stored real-cell h.
- **Query:** `(1−γ)·h_source + γ·h*`.
- **Distances:** rescaled into ê-distance units, so the published margin 1.0 and
  temperature 0.25 keep their meaning. γ is the only new parameter, chosen per arm
  by the flow-cv folds; γ = 0 wins ties.

| arm | h\* comes from | question |
|---|---|---|
| `_flow_h` | the trained flow at z\* | the method |
| `_flow_h_untrained` | the same sampler, flow re-initialised at random | does training matter? |
| `_flow_h_interp` | no network: (1−t)·kNN-mean h (lower) + t·kNN-mean h (upper) | is it the flow, or blending the flanks? |
| `_flow_h_srcdepth` | the trained flow queried at the source section's depth | does the target depth matter? |

**A confound found and fixed: common random numbers.** In v18 every stage draws
from one numpy stream. A single re-grounding draw, even one that lands back on
the same cell, shifts every later draw of the type vote, composition matching and
gene mix. In the first sweep, 3374 of 4140 output rows changed with about one cell
swapped, and gene detection fell from 1.000 to 0.911 with no swap at all. That
first sweep was discarded.
- **The fix.** Re-grounding draws from its own generator (`SplitRNG`), and under
  h-cv the vote, composition and gene-mix stages each get their own seeded
  generator (`STAGE_STREAMS`).
- **Check.** γ = 0.25 with no swap is now identical to γ = 0, and γ = 1 changes
  625 of 4140 rows.
- **Cost.** h-cv's baseline is now h-cv at γ = 0, which is "nearest + no flow"
  with re-drawn random numbers. Every row below is compared with it. Its gap to
  the registry "nearest + no flow" row is a direct measure of run-to-run noise.

Paired composite against h-cv γ = 0 (8 pooled metrics, wins − losses; seed 42;
`scripts/gap_sweep_starmap.sh` with `SWEEP=h`):

| gap | flow | untrained | interp | srcdepth | registry nearest + no flow (= noise) | γ chosen (flow / untr. / interp / src) |
|---|---|---|---|---|---|---|
| 11 µm, paper | 0 | +3 | 0 | 0 | −2 | 0 / 0.25 / 0 / 0 |
| 11 µm, block 1 | −1 | +1 | 0 | −1 | +2 | 1 / 0.25 / 0 / 1 |
| 22 µm, block 3 | −3 | −1 | −4 | +1 | 0 | 0.5 / 0.25 / 0.75 / 0.5 |
| 33 µm, block 5 | 0 | 0 | 0 | 0 | +2 | no folds |

Forced γ = 1 (full h\* query; diagnostic): flow −4 / −1 / −6 / −6, untrained
−2 / −3 / −2 / −6, interp +2 / −1 / −6 / −2, srcdepth −3 / −1 / −4 / −6.

**Reading.**
- **No gain from the trained flow.** It gains nothing at any gap (0, −1, −3, 0)
  and does no better than the untrained control (+3, +1, −1, 0) or the no-network
  blend.
- **Noise is ±2.** Re-drawing random numbers alone moves the composite by ±2 (the
  registry column). Every fold-calibrated difference here is within about ±3, so
  none is distinguishable from noise. Part of that noise is UMAP mixing, the one
  stochastic metric, which accounts for several of the single-metric wins and
  losses.
- **Full h\* grounding hurts,** most at wide gaps (−6 at 22 and 33 µm), as the
  published ê-space reranking did. Replacing copied cells with what the flow
  predicts moves the output away from the real tissue.
- **Conclusion on STARmap.** Using `h*` instead of ê does not make the flow
  useful. The neighbourhood and type information in h\* doesn't pick better
  donors than the copied cell itself.
- **Wider lesson for every ablation in §7.** Any variant that changes a cell
  shifts v18's shared random stream. Single-run differences of ±2 on the 8-metric
  composite should be read as noise unless they come from a paired (common random
  numbers) comparison or hold over several seeds.

## 10. The flow used generatively: a calibrated uncertainty map? (`spatialcpav18_gen_uq`)

**What it is.** A new wrapper, `methods/run_spatialcpav18_uq.py`. It imports the
flow wrapper and v18 without editing either. The synthesized section is v18
"nearest + no flow", bitwise (checked), so its paper metrics are those of plain
retrieval. It also writes `uncertainty.npz` with three per-cell scores:
- `u_flow`: the spread of 8 independent single-noise flow draws, decoded to the
  expression latent. v18 averages its noise draws, which is what discards this
  information.
- `u_untrained`: the same spread from a re-initialised flow.
- `d_flank`: the latent disagreement of the two flanking sections at the cell, a
  retrieval-only proxy with no network.

A new evaluation-only scorer, `evaluate_uncertainty.py`, poses the prediction
onto the GT as `evaluate_paper` does. It bins both on the `FIELD_GRID` lattice
and takes each patch's error as the L2 distance between predicted and real mean
rank-expression. For each score it reports Spearman ρ with the error; the partial
ρ with patch cell count, the `sampling` baseline sqrt(1/n_pred + 1/n_gt),
regressed out; AUROC for the worst quarter of patches; and mean error per score
quintile. Tested in `tests/test_evaluate_uncertainty.py`: a score that tracks the
error ranks it, noise doesn't, and a pure cell-count score is caught by the
partial ρ.

**STARmap, seed 42.** Grid 20 is primary; grid 10 is a sensitivity analysis
declared before looking.

| design (patches) | u_flow ρ / partial / AUROC | u_untrained ρ / partial | d_flank ρ / partial / AUROC | sampling ρ / AUROC |
|---|---|---|---|---|
| paper (935) | +0.17 / −0.01 / 0.55 | −0.01 / −0.02 | +0.02 / +0.10 / 0.50 | +0.50 / 0.75 |
| wide 4 (317) | +0.24 / +0.04 / 0.61 | −0.06 / −0.04 | +0.08 / +0.20 / 0.55 | +0.51 / 0.75 |
| wide 3–5 (962) | +0.16 / −0.01 / 0.56 | +0.04 / +0.04 | +0.10 / +0.09 / 0.55 | +0.49 / 0.75 |
| wide 2–6 (1545) | +0.04 / −0.02 / 0.49 | −0.02 / −0.01 | +0.14 / +0.10 / 0.58 | +0.45 / 0.73 |
| grid 10, partial ρ (paper / w4 / w3–5 / w2–6) | −0.03 / +0.13 / −0.12 / −0.11 | −0.15 / −0.10 / +0.06 / −0.08 | −0.03 / +0.12 / +0.05 / +0.15 | — |

**Reading.**
- **The flow's spread mostly tracks cell count.** It correlates weakly with the
  error (ρ up to 0.24, where the untrained flow gets about 0), but almost all of
  that is how few cells a patch has. With cell count regressed out it is ≈ 0 at
  grid 20 (−0.02 to +0.04) and inconsistent in sign at grid 10. The trained flow
  has learned to be less certain in sparse regions, and nothing beyond that.
- **The retrieval-only flank disagreement is better.** It has a small but
  consistent partial ρ (+0.09 to +0.20 at grid 20), yet is still a weak predictor.
- **Most patch error here is sampling noise.** Cell count alone reaches
  ρ ≈ 0.5 and AUROC ≈ 0.75, so the error that any uncertainty map could explain
  is small at this resolution.
- **Conclusion on STARmap.** Used generatively, the flow gives no calibrated
  uncertainty beyond cell count, and does no better than comparing the two flanks.
  This doesn't support an uncertainty-quantification claim for the flow.

## 11. The flow generates the layout's density (`spatialcpav18_gen_flow_layout*`)

**What it is.** A new wrapper, `methods/run_spatialcpav18_layout.py`. It imports
the flow wrapper and v18 without editing either. v18 copies one flank's layout;
here a density source decides how many cells each region of the output holds.
- **Bins and targets.** The rule flank is copied, and both flanks' cells are
  binned on one grid of 5 median spacings. Each bin's count moves from the copied
  count c_b towards `N · w_b / Σw` by a fraction ρ; the total N is kept up to the
  caveat below.
- **Moving cells.** Bins over target drop copied cells; bins under target take
  unused real cells of either flank in that bin. Every cell is still a real cell
  at a real position.
- **Strength.** ρ ∈ {0, 0.25, 0.5, 0.75, 1} is chosen by the flow-cv folds, and
  ρ = 0 (the copy) wins ties.
- **Random numbers.** As in h-cv, they are paired (common random numbers). The
  check passed: ρ = 0 is bitwise h-cv's γ = 0 baseline.

| arm | w_b comes from |
|---|---|
| `_flow_layout` | the trained flow: decoded density channel of v18's neighbourhood head at the bin's cell centroid, squared (density is an inverse spacing) |
| `_flow_layout_untrained` | the same from a re-initialised flow |
| `_flow_layout_interp` | no network: (1−t)·lower-flank count + t·upper-flank count |

**Caveat.** Where the flow asks for more cells than the flanks have unused cells
in that bin, the bin stays short. At ρ = 1 the flow arms emit about 4 % fewer
cells; the interpolation arm loses none.

**STARmap, seed 42.** Paired composite against ρ = 0 (8 pooled metrics, wins −
losses; `SWEEP=layout scripts/gap_sweep_starmap.sh`):

| gap | flow | untrained | interp (no network) | registry nearest + no flow (= noise) | ρ chosen (flow / untr. / interp) |
|---|---|---|---|---|---|
| 11 µm, paper | 0 | 0 | 0 | −2 | 0 / 0 / 0 |
| 11 µm, block 1 | +2 | 0 | 0 | +2 | 0.25 / 0 / 0.25 |
| 22 µm, block 3 | 0 | 0 | **+4** | 0 | 0 / 0 / 1.0 |
| 33 µm, block 5 | 0 | 0 | 0 | +2 | no folds |

Forced ρ = 1 (diagnostic): flow 0 / −2 / −4 / −6; untrained −6 / −4 / −4 / −6;
interp +2 / −1 / +4 / −2.

**Reading.**
- **The trained flow's density is better than an untrained one** (forced: 0 to
  −6 against −4 to −6; untrained folds always keep ρ = 0). So the flow has learned
  something about where cells are.
- **It is no better than interpolating the two flanks' counts, which needs no
  network.** The interpolation is the only arm with a gain beyond the ±2 noise
  floor (+4 at 22 µm, under both calibrated and forced ρ). The flow's best is +2
  at 11 µm, inside the noise.
- **Pushing the layout fully to the flow's density hurts at wide gaps** (−4 and
  −6).
- **Conclusion on STARmap.** A flow-generated density does not beat a classical
  interpolation, so it supports no claim that the flow is useful. If anything
  here is worth following up, it is the no-network interpolated layout at wide
  gaps: one dataset, one seed, +4, to be checked over seeds before it means
  anything.

## 12. Seed replication of the interpolated layout (`scripts/seed_sweep_layout_interp.sh`)

**Design, fixed before running.**
- **Seeds:** 1–5, all fresh. Seed 42 produced the §11 lead and is excluded.
- **Runs:** per seed and design, the fold-calibrated
  `spatialcpav18_gen_flow_layout_interp`, its ρ = 0 baseline and forced ρ = 1, on
  shared random streams. 60 runs, all completed.
- **Primary endpoint:** the 22 µm block-3 paired 8-metric composite (wins −
  losses against the same seed's ρ = 0). `scripts/summarize_seed_sweep.py` writes
  the tables.

| gap | composite, fold-calibrated: mean ± sd (seeds > 0 / < 0) | forced ρ = 1: mean ± sd | per-seed calibrated composite |
|---|---|---|---|
| 11 µm, paper | +2.0 ± 2.4 (4 / 1) | +2.0 ± 2.0 | +4, +2, −2, +2, +4 |
| 11 µm, block 1 | +2.4 ± 2.6 (3 / 0) | +2.6 ± 1.3 | +4, 0, +6, 0, +2 |
| **22 µm, block 3 (primary)** | **+0.4 ± 0.9 (1 / 0)** | +0.8 ± 1.1 | 0, +2, 0, 0, 0 |
| 33 µm, block 5 | 0 (no folds) | +0.4 ± 2.6 | — |

**Primary result: not replicated.** At 22 µm the folds kept ρ = 0 for three seeds
of five, and the composite averaged +0.4. The seed-42 +4 that motivated this was
noise.

**Exploratory, not pre-registered: a consistent small gain at 11 µm.** At both
11 µm designs the interpolated layout is ≥ 0 in 9 of 10 calibrated runs. Forced
ρ = 1 is ≥ 0 in all 10 and > 0 in 8. The per-metric means show the gain:
- **Gains:** cell-type localization +0.023 to +0.025, rare-type localization up
  to +0.022, and field r and detection up slightly.
- **Costs:** Moran's I MAE is slightly worse (+0.001 to +0.005).

Read this carefully:
1. **It was not the hypothesis tested.** It needs its own confirmation on new
   seeds and other datasets before any claim.
2. **The effect is small.** It is about +2 on the composite with an sd of about
   2.5, at a noise floor of ±2.
3. **It is not a flow result.** The interpolated density uses no network. If it
   holds up, it is a retrieval improvement: set each region's cell count from both
   flanks, depth-weighted, rather than copying one flank's layout.
