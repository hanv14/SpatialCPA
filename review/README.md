# SpatialCPA-v18 — review repository

A self-contained copy of what it takes to re-run and re-score **SpatialCPA-v18**
(`learn_spatialcpav18.py`) on the SpatialZ STARmap paper protocol, next to the
three published comparators it was benchmarked against: **SpatialZ**, **FEAST**
and **isoST**. The benchmark harness, the scorer, the method wrappers and the
dataset scripts all live in one folder, [`benchmark/`](#layout).

No other SpatialCPA version is in this tree: no wrapper, registry entry, method
file or result. `tests/test_scope.py` and `tests/test_rename.py` enforce that.

Start with [`REVIEW_NOTES.md`](REVIEW_NOTES.md) for what deserves attention. This
file is the how-to.

---

## Contents

1. [What "reproduced" means here](#what-reproduced-means-here)
2. [Quick start: clone → STARmap v18 row](#quick-start-clone--starmap-v18-row)
3. [Effective configuration of the v18 rows](#effective-configuration-of-the-v18-rows)
4. [Pinned files and provenance](#pinned-files-and-provenance)
5. [Environments](#environments)
6. [How v18 is located](#how-v18-is-located)
7. [Data](#data)
8. [Full benchmark: all four methods, all datasets](#full-benchmark-all-four-methods-all-datasets)
9. [Tests](#tests)
10. [Layout](#layout)

---

## What "reproduced" means here

Two committed reference rows can sit under `expected/`, and `make starmap-row`
compares against whichever is present:

| row | what it is | status |
|---|---|---|
| `expected/published/…/metrics.json` | the metrics.json behind the published v18 STARmap row, copied from the lab machine | **not yet committed.** `make starmap-row` uses it automatically once it is |
| `expected/cpu-verified/…/{prediction.h5,metrics.json,method_log.txt,resources.json}` | this repository's own run, on CPU, in the environment pinned by `envs/lock/cpu-verified.txt` | committed; reproduced **bitwise** in every independent run since, before and after each restructuring of this tree (all 50 scalar metrics identical at tolerance 0, UMAP included) |

So today `make starmap-row` proves three things:

- the code in this tree runs end to end from a fresh clone;
- it runs under the pinned configuration and loads the pinned `learn_spatialcpav18.py`;
- it is deterministic.

It does not yet prove equality with the published number. That needs the lab
files (`scripts/export_lab_envs.sh` collects them; see REVIEW_NOTES, "Open
provenance items"). If the published row was trained on a GPU, expect the flow
latent, and so the prediction, to differ from the CPU run. The comparison then
shows by how much, metric by metric.

---

## Quick start: clone → STARmap v18 row

Two routes. Both end at `REPRODUCED` or a per-metric diff table.

### Route A — conda, the way the benchmark was run

```bash
git clone <this repo> && cd <repo>/review
make envs                                  # 5 envs; only bench_eval + bench_spatialcpa are
                                           # needed for this row (~10 min on a warm conda cache)
conda activate bench_eval
make verify                                # ~15 s
make starmap-row                           # ~3 min on 4 CPU cores; faster on a GPU
```

### Route B — one virtualenv, no conda (the route this repo was verified on)

```bash
git clone <this repo> && cd <repo>/review
python3.10 -m venv .venv && .venv/bin/pip install -r envs/lock/cpu-verified.txt   # ~3 min, ~5 GB (torch + CUDA wheels)
export BENCH_V3_PYTHON=$PWD/.venv/bin/python      # wrapper runs here instead of `conda run`
make verify PYTHON=.venv/bin/python
make starmap-row PYTHON=.venv/bin/python
```

`BENCH_V3_PYTHON` is the one environment variable this repo adds. It changes only
the launcher: `run_benchmark` runs the wrapper with that interpreter instead of
`conda run -n bench_spatialcpa python`. Leave it unset for Route A. No
environment variable is needed to find v18 (see [How v18 is located](#how-v18-is-located)).

### What `make starmap-row` does (`scripts/reproduce_starmap_v18.sh`)

| step | what | wall time (4-core CPU, measured) |
|---|---|---|
| 1 | `pytest tests/test_pins.py`: every file in `MANIFEST.sha256`, and `evaluate_paper.py` by literal hash | ~2 s |
| 2 | `prepare_dataset --dataset starmap_visual_cortex`: committed raw volume → 7 paper sections (4073/4187/4169/4102/4110/4162/4175 cells) | ~4 s |
| 3 | `run_all --methods spatialcpav18_gen --dataset starmap_visual_cortex` into `reproduced/`: v18 train + generate, then `evaluate_paper` | 24–36 s method (~0.9–1.0 GB peak RSS) + ~2 min scoring |
| 4 | method log shows `review/learn_spatialcpav18.py` with the manifest's sha256, and the logged command carries the pinned flags | <1 s |
| 5 | `compare_predictions.py` against the committed `prediction.h5` (every array, bitwise) | <1 s |
| 6 | `compare_metrics.py` against the committed `metrics.json` (`--atol 1e-6`; UMAP pair `--umap-atol 0.02`) | <1 s |

Exit 0 means reproduced. Everything is written under `reproduced/`, which is
git-ignored.

---

## Effective configuration of the v18 rows

Every published v18 row was produced with these flags passed as `run_all` extras:

```bash
python -m src.bench3.run_all --methods spatialcpav18_gen --dataset <name> \
  -- --edit-weight 0.0 --ground-blend-flow 1.0 --ground-k 8 --ground-temp 0.25 \
     --ground-keep-margin 1.0 --type-mode vote --type-vote-k 12 --gene-mix-frac 0.15
```

In this repo they are **pinned in the registry** as `V18_ARGS`
(`benchmark/src/bench3/config.py`), set as `spatialcpav18_gen`'s `wrapper_args`.
So a bare `run_all --methods spatialcpav18_gen` runs the published configuration.
The command above still works and produces the same argv; an extra repeats a
pinned value and argparse keeps the last.

Pinning also fixes a trap. `run_all` forwards `--` extras to **every** method in the
invocation, so the command above with `--methods spatialz spatialcpav18_gen` would
hand `--edit-weight` to SpatialZ and fail it.

Every flag the wrapper accepts, with the value that actually ran. This table is
checked against the wrapper's argparse and against `V18_ARGS` by
`tests/test_effective_config.py`:

<!-- effective-config:begin -->
| flag | ran with | wrapper default | note |
|---|---|---|---|
| `--seed` | `42` | `42` | `_v2_io`; run_benchmark passes `config.RANDOM_SEED` |
| `--edit-weight` | `0.0` | `0.25` | **pinned, differs from default.** 0 = emit the grounded exemplar verbatim. It is also the gate for v18's gene-mix (`learn_spatialcpav18.py:1225`) and raw-output path (`:1240`), so both are live in the published rows |
| `--ground-blend-flow` | `1.0` | `0.2` | **pinned, differs from default.** every cell is re-grounded to the flow-latent pick |
| `--ground-k` | `8` | `8` | pinned (restates default) |
| `--ground-temp` | `0.25` | `0.25` | pinned (restates default). Not recorded in `method_params`; see REVIEW_NOTES |
| `--ground-keep-margin` | `1.0` | `1.0` | pinned (restates default) |
| `--type-mode` | `vote` | `vote` | pinned (restates default) |
| `--type-vote-k` | `12` | `12` | pinned (restates default) |
| `--gene-mix-frac` | `0.15` | `0.15` | pinned (restates default) |
| `--epochs` | `160` | `160` | default |
| `--pretrain-epochs` | `60` | `60` | default |
| `--device` | `auto` | `auto` | default: CUDA when present, else CPU. The published rows' device is not recorded |
| `--latent-dim` | `32` | `32` | default |
| `--joint-dim` | `48` | `48` | default |
| `--ode-steps` | `12` | `12` | default |
| `--ensemble` | `4` | `4` | default |
| `--context-slices` | `None` | `None` | default → `V18Config.context_slices_each_side`, which resolves to 1 (method log: `ctx_slices=1`) |
| `--position-mode` | `flanking` | `flanking` | default |
| `--no-coherent-source` | `False` | `False` | default (coherent source on) |
| `--no-output-counts` | `False` | `False` | default (count-like output) |
| `--no-composition-match` | `False` | `False` | default (composition matching on) |
| `--no-gene-mix` | `False` | `False` | default (gene-mix on) |
| `--no-raw-output` | `False` | `False` | default (raw output on) |
| `--no-ground-sample` | `False` | `False` | default (temperature sampling on) |
| `--no-dedup-ground` | `False` | `False` | default (exemplar-reuse penalty on) |
<!-- effective-config:end -->

Everything v18 does that is not a CLI flag is a `V18Config` default inside
`learn_spatialcpav18.py`, which is pinned by sha256. What ran is also written into
every prediction as `uns/method_params`. For the committed CPU row:

```json
{"seed": 42, "epochs": 160, "pretrain_epochs": 60, "latent_dim": 32, "joint_dim": 48,
 "ode_steps": 12, "ensemble": 4, "position_mode": "flanking", "ground_blend_flow": 1.0,
 "ground_k": 8, "edit_weight": 0.0, "type_mode": "vote", "type_vote_k": 12,
 "gene_mix_frac": 0.15, "raw_output": true, "ground_sample": true, "dedup_ground": true,
 "ground_keep_margin": 1.0, "coherent_source": true, "flow_matching": true,
 "generation_only": true}
```

`python -m src.bench3.run_all --methods spatialcpav18_gen --dry-run` (from
`benchmark/`) prints the full command line.

---

## Pinned files and provenance

**`evaluate_paper.py`** — `benchmark/src/bench3/evaluate_paper.py`

```
sha256  7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992
```

It is byte-identical to the file in the parent project (commit `08c385ce`), and so
is `align.py`. Both were left untouched on purpose, docstrings included, so their
hashes can be compared directly with the lab copies. That is why they still
mention the old folder name `benchmark-pbya-v3`.
`tests/test_pins.py::test_evaluate_paper_is_the_pinned_file` asserts the literal,
and another test asserts that this README, the test and the manifest all carry
the same value.

Pinning `evaluate_paper.py` alone would not pin the numbers. It imports the pose
search (`align.py`), the constants in `src/bench3/config.py`, the rank normalizer
and Moran's I from `src/benchmark/evaluate_generation.py`,
`load_prediction`/`load_ground_truth` from `src/benchmark/evaluate.py`, and the
leakage guard. So **`MANIFEST.sha256`** pins:

- every `.py` under `benchmark/src/`, plus `learn_spatialcpav18.py`;
- the STARmap source volume;
- the committed `expected/` rows.

`make verify` checks all of them.

**Provenance of every file: `MANIFEST.original.sha256`.** For each of the 70
files that came from the parent project, it records the original's path and
sha256, the file's path here, and one of three statuses:

| status | count | meaning | how it's checked |
|---|---|---|---|
| `identical` | 40 | byte-for-byte the original (the scorer, the dataset scripts, …) | `tests/test_rename.py` compares bytes |
| `equivalent` | 21 | only docstrings, comments, message text or v18's two class names differ | `scripts/verify_rename.py`: identical AST once those are set aside |
| `changed` | 9 | an executable change: paths for the one-folder layout, the v18 lookup fix, the optional launcher, the method registry | listed with its reason in REVIEW_NOTES §5; all 9 are outside the computation |

The rename and equivalence checks run whenever the parent project's originals sit
beside this tree (a clone of the full project). Against the lab's copies, run
`python scripts/verify_rename.py <lab file> <file here>` by hand.

On top of the per-file checks, every restructuring of this tree was followed by a
re-run of the STARmap row. Each run's `prediction.h5` was **bitwise identical** to
the committed one and all 50 metrics matched at tolerance 0. The same
restructurings were checked with the harness selftest; see REVIEW_NOTES §5.

The two config files are `changed`, so the scorer's constants are also asserted
directly (`test_scoring_constants_unchanged`,
`test_v2_scoring_constants_unchanged`): `SPATIAL_K=10`, `FIELD_GRID=20`,
`DEPTH_BINS=20`, `EMBED_NEIGHBORS=15`, `RARE_CELLTYPE_FRAC=0.05`,
`ALIGN_*=(24, 12, 3000)`, the STARmap trim/partition, the markers, `SSIM_*`,
`NN_MATCH_THRESHOLD_UM`, the 27 evaluator metric names, seed 42.

---

## Environments

Each method runs in its own conda env, launched by the harness with
`conda run -n <env>`. The harness itself runs in `bench_eval`. **Python 3.10
everywhere.**

| env | file | runs | verified here |
|---|---|---|---|
| `bench_eval` | `envs/bench_eval.yml` | prepare / run_all / evaluate_all / aggregate / rank / plots / tests | yes (merged into the CPU lock) |
| `bench_spatialcpa` | `envs/bench_spatialcpa.yml` | `spatialcpav18_gen` | yes (merged into the CPU lock) |
| `bench_spatialz` | `envs/bench_spatialz.yml` | `spatialz` (+ SpatialZ code from Zenodo, `make tools`) | no: Zenodo unreachable from the build environment |
| `bench_feast` | `envs/bench_feast.yml` | `feast` | no |
| `bench_isost` | `envs/bench_isost.yml` | `isost` (+ isoST from GitHub at a pinned commit, `make tools`); **needs a CUDA GPU** for practical run times | no |

Locks in `envs/lock/`:

- `cpu-verified.txt`: the exact package set (70 packages) that reproduces
  `expected/cpu-verified/`, as one venv. Key versions: numpy 1.26.4,
  scipy 1.15.3, scikit-learn 1.7.2, scanpy 1.11.5, anndata 0.11.4,
  umap-learn 0.5.12, numba 0.67.0, torch 2.4.1 (PyPI wheel, run on CPU),
  h5py 3.16.0.
- `*.lab.yml` / `*.lab.pip.txt` / `provenance.txt`: **not yet present.**
  `scripts/export_lab_envs.sh`, run once on the lab machine, writes them.

The `.yml` files are version *ranges*. The originals they descend from could not
have built a working env from scratch, and the fixes are marked `REVIEW COPY` in
each file:

- `bench_spatialz`: SpatialZ.py imports torch, POT, plotly, MENDER and tqdm, and
  none of them was listed.
- `bench_isost`: the PyG extension wheels were pinned but torch, which they are
  ABI-bound to, was not.
- `bench_feast`: paste2 was implicit.

---

## How v18 is located

`benchmark/src/bench3/methods/run_spatialcpav18.py` resolves, in order:

1. `$SPATIALCPAV18_FILE`, the file itself;
2. `$SPATIALCPAV18_ROOT`, a directory containing it;
3. `<repo root>/learn_spatialcpav18.py`. **Nothing else.**

A fresh clone needs **no environment variable**: the file is committed at the
repository root. The published wrapper instead walked every parent directory, so
with the file missing it would silently load whatever `learn_spatialcpav18.py` an
enclosing directory held. A file kept outside the repository (the lab layout
keeps it in `Spatial3D/src/`) is not on that walk at all. The walk now stops at
the repository root, and an explicit override never falls through to it. The
method log prints the resolved path and the file's sha256
(`learn_spatialcpav18.py sha256 …`), and `make starmap-row` checks both.
`tests/test_path_resolution.py` covers: no env vars, any cwd, a bad `FILE`
override not falling through, and the `ROOT` override.

Everything else a wrapper needs is inside `benchmark/`:
- the shared wrapper I/O `_v2_io.py` sits beside the wrappers;
- the leakage guard is in `src/benchmark/`;
- SpatialZ and isoST are expected in `benchmark/tools/`, where `make tools` puts them.

---

## Data

| what | where | size | in git? | how to get it |
|---|---|---|---|---|
| STARmap raw volume (Wang et al. 2018; 32 845 cells × 28 genes, 89 z-planes) | `benchmark/data/raw/starmap_visual_cortex/STARmap_Wang2018three_data_3D_data.h5ad` | 30 MB | **yes** (sha256 in manifest) | — |
| STARmap paper-protocol dataset | `benchmark/data/sections/starmap_visual_cortex/data.h5ad` | 9.9 MB | no (built) | `prepare_dataset --dataset starmap_visual_cortex`, **4 s** |
| training-only input per holdout | `benchmark/results/_inputs/…` | 5.7 MB (STARmap) | no (built on first run, cached, mtime-invalidated) | automatic |
| SpatialZ code | `benchmark/tools/spatialz/` | small | no | `make tools` (Zenodo 10.5281/zenodo.17416727) |
| isoST code | `benchmark/tools/isost/` | small | no | `make tools` (GitHub, pinned commit) |
| the 17 analogue volumes | `benchmark/data/raw/`, `benchmark/data/processed/` | 50 MB – 15 GB each, ~50 GB raw in total | no, too large | [below](#0-datasets) |

Everything the STARmap row needs is committed or built in seconds. The analogue
datasets are not committed: several are GB-scale, and all are redistributable
only from their original sources.

---

## Full benchmark: all four methods, all datasets

The complete campaign is **4 methods × 18 datasets** under `--design paper` (one
holdout per dataset). Stages:

```
0 datasets ──► 1 inputs ──► 2 method runs (4 envs, parallel) ──► 3 evaluate_all ──► 4 aggregate ──► 5 rank ──► 6 figures
```

All commands run from `benchmark/` in `bench_eval`, unless stated otherwise.

Wall times below are **measured** only where they say so (STARmap v18 on a 4-core
CPU). Everything else is marked *not measured*. The comparator envs and the
analogue datasets could not be built where this repo was assembled.
`results/<method>/<dataset>/<holdout>/resources.json` records wall time, peak RSS
and GPU memory for every run you do make, and `aggregate_results` collects them.

### 0. Datasets

STARmap is committed. For the analogues:

1. Download each source with its script in `src/data/download/`: one
   self-contained script per source, resumable, writing to `data/raw/<source>/`.
   Each script's own header is the authority on the source; the table in
   `src/data/download/README.md` is stale in places.
2. Six sources also need their processor in `src/data/process/`, which writes
   `data/processed/<source>/…`: `merfish_thick_*`, `easi_fish_lha*`,
   `exseq_breast_cancer`, `st_mouse_brain_ortiz`, `visium_mouse_brain_c2l`. The
   rest are read raw by `src/bench3/sources.py`.
3. Cut each one into its paper-protocol sections (`data/sections/<dataset>/`).

```bash
cd benchmark
python src/data/download/download_<source>.py             # or download_all.py
python src/data/process/process_<source>.py               # only the six above
python -m src.bench3.prepare_dataset --dataset <name>     # all: loop over the names below
python -m src.bench3.describe_datasets                    # one-line check of every built dataset
```

| dataset | download script | raw size (script README's estimate) | held out |
|---|---|---|---|
| `starmap_visual_cortex` | committed | 30 MB | 2, 4, 6 |
| `exseq_visual_cortex` | `download_exseq_visual_cortex.py` | ~1 GB | 2, 4, 6 |
| `imc_breast_cancer` | `download_imc_breast_cancer_kuett.py` | ~200 MB | 2,4,…,14 |
| `cosmx_nsclc_3d` | `download_cosmx_nsclc_3d.py` | ~2 GB | 2, 4 |
| `deep_starmap` | `download_deep_starmap.py` | ~50 MB | 2, 4, 6 |
| `merfish_hypothalamus` | `download_merfish_hypothalamus.py` | ~2 GB | 2,4,…,10 |
| `openst_lymph_node` | `download_openst_lymph_node.py` | ~15 GB | 2,4,…,18 — **known to fail**, see below |
| `allen_merfish_brain` | `download_allen_merfish_brain.py` | ~12 GB | alternate |
| `allen_zhuang_abca1`, `allen_zhuang_abca2` | `download_allen_zhuang_merfish.py` | ~8 GB (both) | alternate |
| `merfish_thick_cortex`, `merfish_thick_hypothalamus` | `download_merfish_thick_tissue.py` + process | ~500 MB | alternate |
| `easi_fish_lha1`, `lha2`, `lha3` | `download_easi_fish_hypothalamus.py` + process | ~3 GB | alternate |
| `exseq_breast_cancer` | `download_exseq_breast_cancer.py` + process | ~200 MB | alternate (5 sections) |
| `st_mouse_brain_ortiz` | `download_st_mouse_brain_ortiz.py` + process | ~200 MB | alternate (spot) |
| `visium_mouse_brain_c2l` | `download_visium_mouse_brain_cell2location.py` + process | ~500 MB | 1 of 3 (spot) |

**Disk: ~50 GB raw**, plus processed copies of the same order for the Allen and
Open-ST volumes. Download time depends on the network, *not measured*.
`prepare_dataset` took 4 s for STARmap; the million-cell Allen and Open-ST builds
are dominated by reading the source h5ad, *not measured*. The per-dataset
protocol (partition, trims, caps, hold-out pattern) is in
[`docs/DATASETS.md`](docs/DATASETS.md).

### 1. Build every training-only input once, and see the plan

```bash
for d in $(python -c "from src.bench3.config import DATASET_SPECS; print(*DATASET_SPECS)"); do
  python -m src.bench3.run_all --dataset $d --dry-run
done
```

**`--dry-run`** prints the holdout design, the run count and, for each (method,
holdout), the exact wrapper command, including `conda run -n <env>` and the
pinned `V18_ARGS`. It runs no method and scores nothing. It does **build (or
reuse) the training-only input** for each holdout, because the printed command
names that file. That makes a dry-run pass the right way to pre-build every input
before launching methods in parallel. Inputs are cached under
`results/_inputs/<dataset>/<holdout>/` and rebuilt automatically if the dataset
file is newer.

### 2. Method runs — one process per method, in parallel

Each method needs its own env, and `run_all` launches every wrapper with `conda run
-n <env>`. So parallelism means one `run_all` process **per method**, each started
from `bench_eval`. They write disjoint trees (`results/<method>/…`) and read the
shared cached inputs:

```bash
DATASETS="$(python -c "from src.bench3.config import DATASET_SPECS; print(*DATASET_SPECS)")"
for m in spatialz feast isost spatialcpav18_gen; do
  ( for d in $DATASETS; do
      python -m src.bench3.run_all --methods $m --dataset $d --skip-existing --no-eval
    done ) > results/campaign_$m.log 2>&1 &
done
wait
```

- **`--skip-existing`** skips a (method, holdout) whose `prediction.h5` already
  exists, and skips building its input. It looks only at `prediction.h5`, not
  `metrics.json`. So a run whose scoring failed is *not* redone; score it in stage
  3. A run quarantined as degraded (`prediction.h5.degraded`, below) has no
  `prediction.h5`, so it *is* retried.
- **`--no-eval`** defers scoring to stage 3. Scoring is CPU-bound and runs in
  `bench_eval`, so it doesn't need to hold a GPU slot.
- `results/summary/run_log_paper.json` is rewritten by **every** `run_all`
  process, so the last one wins. The per-method `campaign_<m>.log` above is the
  durable record.
- Run `isost` (GPU) and `spatialcpav18_gen` (GPU if present) on different devices
  with `CUDA_VISIBLE_DEVICES`. SpatialZ and FEAST are CPU.

A run that exits 0 but whose log shows the method fell back to something that isn't
the method (v18's numpy fallback when torch is missing or the flow fails to train)
is **failed**, not scored. Its prediction is moved to `prediction.h5.degraded`.
The markers are in `METHODS[...]["invalid_log_markers"]`.

| per run | wall time | peak RSS | disk |
|---|---|---|---|
| `spatialcpav18_gen`, STARmap, 4-core CPU | **24–36 s** (measured, 3 uncontended runs) | 0.9–1.0 GB | 3.4 MB prediction |
| `spatialz` / `feast` / `isost`, STARmap | *not measured* | | similar (same cell count) |
| any method, million-cell datasets | *not measured*; scales with cells × genes | | tens–hundreds of MB |

### 3. Score — `evaluate_all`

```bash
python -m src.bench3.evaluate_all                    # every prediction lacking metrics.json
python -m src.bench3.evaluate_all --force            # re-score everything (e.g. after a scorer change)
python -m src.bench3.evaluate_all --methods spatialcpav18_gen --dataset-name starmap_visual_cortex --force
```

Each prediction is scored against **its own** dataset's ground truth, resolved from
its results path. **~2 min per STARmap prediction on 4 CPU cores (measured)**, most
of it UMAP and the per-type Sinkhorn OT. `--no-umap` skips UMAP but leaves
`paper_umap_*` empty, and those metrics are in the ranking. Writes
`metrics.json` (~10 KB) beside each prediction.

### 4. Aggregate — `aggregate_results`

```bash
python -m src.bench3.aggregate_results
```

Writes `results/summary/all_metrics.csv` (one row per method × dataset × holdout),
`per_section_metrics.csv` and `summary_by_method.csv`. Seconds.

### 5. Rank — `rank_methods`

```bash
python -m src.bench3.rank_methods                                   # one table per dataset
python -m src.bench3.rank_methods --dataset-name starmap_visual_cortex --csv results/summary/rank_starmap.csv
```

Methods are ranked **within** each of five groups (continuity, autocorrelation,
markers, localization, expression), and the composite is the mean of the group
ranks. Ranks are per dataset and never pooled across datasets.
`paper_cell_count_ratio` and `paper_rare_celltype_recall` are reported but not
ranked. Spot datasets are flagged. `--include-gen` adds the `gen_*` metrics as a
sixth group. Seconds.

### 6. Figures (optional)

```bash
python -m src.bench3.plot_paper_figures --methods spatialz feast isost spatialcpav18_gen
python -m src.bench3.plot_cross_dataset --method-order spatialz feast isost spatialcpav18_gen
```

### Known failures

| dataset | methods | why |
|---|---|---|
| `openst_lymph_node` | `spatialcpav18_gen` | **OOM.** Whole-transcriptome (~10⁶ cells × ~2×10⁴ genes) with no `n_hvg` cap, and the wrapper densifies the training matrix (`run_spatialcpav18.py:348`), which is order 80 GB; a run with a wrapper that densifies the same way was OOM-killed on this dataset. Adding `"n_hvg": 3000` to its spec would fix it, but that changes the panel a reported dataset was measured on, so it is left as it was |
| `visium_mouse_brain_c2l` | all | not a failure, but degenerate: 3 sections, 1 held out, spot resolution. Its localization group scores deconvolved composition; read only against `st_mouse_brain_ortiz` |
| `st_mouse_brain_ortiz` | all | spot resolution, same caveat |

Other datasets' comparator outcomes (SpatialZ / FEAST / isoST crashes or time-outs,
if any) are not recorded in this repository; the lab `results/` tree is the
authority.

---

## Tests

```bash
make verify            # = pytest -q tests      (~15 s; needs numpy/h5py only for the path test)
make selftest          # the harness's own check: oracle / flanking copy / scramble / random (~4-5 min CPU)
```

| test file | asserts |
|---|---|
| `test_pins.py` | `evaluate_paper.py` sha256 = the literal above; every `MANIFEST.sha256` entry matches; the manifest covers the whole scoring chain; the scorer's constants equal their published values |
| `test_effective_config.py` | `V18_ARGS` = the published flags and is what `spatialcpav18_gen` runs with; comparators get none; the table above = argparse defaults ⊕ `V18_ARGS` |
| `test_path_resolution.py` | v18 resolves to `review/learn_spatialcpav18.py` with no env vars from any cwd, looks nowhere else, and the overrides don't fall through |
| `test_rename.py` | nothing in the tree names another SpatialCPA version; the `[v18]` fallback markers match strings v18 prints; every file carried from the parent project is identical or AST-equivalent to its original, except the 9 documented `changed` files; the checker rejects a one-constant change |
| `test_scope.py` | the registry is exactly {spatialz, feast, isost, spatialcpav18_gen}; the only wrappers are those four; one `benchmark/` folder, no `benchmark-pbya*` |

---

## Layout

```
review/
  README.md, REVIEW_NOTES.md, Makefile, pytest.ini
  MANIFEST.sha256              sha256 of every source file, the STARmap volume, expected/
  MANIFEST.original.sha256     provenance: each file's parent-project original + identical/equivalent/changed
  MANIFEST.tools.sha256        SpatialZ.py / Synthesize.py as kept in the parent project's reference/
  learn_spatialcpav18.py       THE METHOD (single file)
  benchmark/                   everything benchmark-related — run commands from here
    src/bench3/                the harness package (python -m src.bench3.<module>)
      config.py                  protocol constants, dataset specs, METHODS (+ V18_ARGS)
      prepare_dataset.py, sources.py, prepare_starmap.py, design.py
      run_all.py, run_benchmark.py, _v2bridge.py, assets.py, sanitize_checkpoint.py
      evaluate_paper.py, align.py, evaluate_all.py        ← the scorer (pinned)
      aggregate_results.py, rank_methods.py, describe_datasets.py, survey_datasets.py
      selftest.py, selftest_datasets.py, plot_*.py, nature_theme.py
      methods/                 the four method wrappers + their shared I/O
        run_spatialcpav18.py, run_spatialz.py, run_feast.py, run_isost.py, _v2_io.py
    src/benchmark/             shared evaluators + leakage guard, imported as `benchmark`
      evaluate.py, evaluate_generation.py, leakage_guard.py, resource_monitor.py, config.py
    src/data/download/, src/data/process/   one script per dataset source
    data/raw/                  downloaded sources (only STARmap is committed)
    data/processed/            (built) processed sources
    data/sections/             (built) paper-protocol datasets
    results/                   (built) predictions, metrics, summaries, figures
    tools/                     (fetched) SpatialZ + isoST code — make tools
  expected/
    cpu-verified/spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6/
                               prediction.h5, metrics.json, method_log.txt, resources.json
    published/                 (slot) the lab's files for the same row — see REVIEW_NOTES
  envs/                        bench_{eval,spatialcpa,spatialz,feast,isost}.yml
    lock/cpu-verified.txt      exact set that reproduces expected/cpu-verified
  scripts/
    reproduce_starmap_v18.sh   make starmap-row
    compare_metrics.py         metrics.json vs metrics.json, per metric, two tolerances
    compare_predictions.py     prediction.h5 vs prediction.h5, every array, bitwise
    verify_rename.py           AST equivalence of a file against its original
    probe_flow_contribution.py how much v18's flow changes what is emitted (REVIEW_NOTES §2)
    fetch_tools.sh             SpatialZ (Zenodo) + isoST (GitHub @ pinned commit)
    export_lab_envs.sh         run on the lab machine: envs + provenance hashes → envs/lock/
    write_manifest.sh          regenerate MANIFEST.sha256
  tests/                       see above
  docs/PROTOCOL.md             the protocol, metrics, pose search and leakage policy
  docs/DATASETS.md             the 18 datasets: partitions, caps, sources, hold-outs
```

Two names keep their history because the pinned scorer depends on them:
- `src/benchmark/` is imported as `benchmark`: `evaluate_paper.py` does
  `from benchmark.evaluate_generation import …`.
- `_v2bridge.py` is imported by that name.

`_v2_io.py`, the wrapper I/O contract, keeps its name to match.
