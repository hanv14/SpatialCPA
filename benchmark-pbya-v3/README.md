# benchmark-pbya-v3 — the SpatialZ STARmap paper protocol, run on every method

A **paper-faithful reproduction benchmark**. Where `benchmark-pbya-v2` sweeps 17
datasets with leave-one-out and scores generation with its own metrics, v3 does
one thing: it reproduces the STARmap visual-cortex evaluation from the SpatialZ
paper (Lin et al. 2025, *Nature Methods*) exactly, and puts every method through
it — so a published method's numbers are reproducible on their own terms, and the
SpatialCPA variants are measured against them on identical footing.

Methods benchmarked: **spatialcpav8_gen, spatialcpav11_gen, spatialcpav14_gen,
spatialcpav15_gen, SpatialZ, FEAST, isoST**.

v3 does not modify v1 or v2; all three coexist.

---

## The protocol

Straight from the paper's description of the dataset:

| Step | What the paper says | What v3 does |
|---|---|---|
| Source | STARmap mouse visual cortex, 3-D, single-cell resolution | `data/starmap/STARmap_Wang2018three_data_3D_data.h5ad` (32 845 cells × 28 genes, 89 z-planes) |
| Trim | remove uppermost `z = 6–13` and lowermost `z = 91–94` | drops 3 867 cells (11.8 %); 77 planes remain (`z = 14–90`) |
| Partition | divide the remainder into **seven consecutive 2-D sections** | 77 / 7 = **exactly 11 planes per section** — the partition is even, no fudging |
| Split | hold out sections **2, 4, 6**; input sections **1, 3, 5, 7** | one holdout config, `paper_2_4_6` |
| Objective | reconstruct the missing sections as virtual slices | generation-only; the held-out sections are the ground truth |

```
                 z(µm)   planes    cells
  section_1       19     14–24      4073   input
  section_2       30     25–35      4187   HELD OUT
  section_3       41     36–46      4169   input
  section_4       52     47–57      4102   HELD OUT
  section_5       63     58–68      4110   input
  section_6       74     69–79      4162   HELD OUT
  section_7       85     80–90      4175   input
```

Two implementation choices worth stating outright:

* **Sections are flattened to 2-D.** Each partition is an 11-plane slab, but the
  protocol calls it a *section* — the thing a microtome produces and the thing a
  virtual slice has to be. So each slab's cells keep their real `(x, y)` and take
  the slab's centre z. The original plane survives in `obs['z_plane']`;
  `--no-flatten-z` opts out.
* **All three sections are held out at once**, not one at a time. That is the
  paper's design and it is materially harder: half the volume is missing and no
  reconstruction ever sees an adjacent real slice. Each held-out section is still
  bracketed by two input sections (1|3, 3|5, 5|7), so the task stays well-posed
  for every interpolation method. A `--design loo` robustness check exists, but
  it is an *easier* task and its numbers are not comparable.

---

## What is measured

The paper validates a reconstruction four ways. Each becomes a metric group in
`evaluate_paper.py`, and `rank_methods.py` ranks methods within each group and
averages the group ranks — so no criterion dominates just because it happens to
contribute more individual numbers.

| Paper's validation | Metrics | Notes |
|---|---|---|
| **UMAP continuity** between real and reconstructed slices | `paper_umap_mixing` ↑, `paper_umap_centroid_dist` ↓, `paper_embedding_mixing_pca` ↑ | kNN mixing in a shared embedding, normalized by the value expected under perfect mixing. 1 = the two clouds are locally indistinguishable, 0 = disjoint islands. The PCA variant is deterministic and is the number to trust if UMAP's stochasticity is a concern. |
| **Marker-gene spatial patterns** (Flt1, Pcp4, Cux2) | `paper_marker_field_r` ↑, `paper_marker_depth_r` ↑, `paper_marker_morans_mae` ↓, plus per-gene breakdowns | Binned 2-D field; profile along the cortical **laminar axis**; and the deviation in each marker's own Moran's I. The laminar axis is derived from the ground truth as the spatial gradient of a signed layer score (superficial minus deep markers) — for cortex the depth profile is the honest "did you reproduce the pattern" test, and it is far more robust than the 2-D field to residual in-plane misalignment. |
| **Spatial autocorrelation** — Moran's I *and* Geary's C | `paper_morans_pearson`/`_spearman`/`_mae`, `paper_gearys_*`, plus pred/GT medians | Both computed per gene on a row-standardized kNN graph *within* each slice, so they are alignment-free. The correlations say whether the *ranking* of genes by spatial structure survives; the MAEs say whether the *level* is right — which is what catches over-smoothing (blur inflates Moran's I and deflates Geary's C while leaving the ranking intact). |
| **Preservation of cell spatial localization** | `paper_celltype_localization` ↑, `paper_celltype_ot` ↓ | Per cell type, a debiased Sinkhorn (OT) divergence between the predicted and true spatial distributions, calibrated against a within-tissue null: scattering that type anywhere in the tissue. 1 = localization reproduced, 0 = no better than random placement. |
| **Gene expression similarity** | `paper_gene_mean_spearman`, `paper_gene_var_spearman` | Per-gene mean/variance agreement on log-normalized expression. |

Also written, for continuity with the rest of the repo: v2's correspondence-free
`gen_*` metrics (identical code, so v3 rows read alongside the v2 sweep) and v2's
cell-matched metrics as **reference only** — de-novo generation produces no
cell-to-cell correspondence, so those are not a valid score here.

`paper_cell_count_ratio` is reported but deliberately **not** ranked: the number
of cells is emergent in generation-only mode, so it is a diagnostic ("did the
method produce a plausible amount of tissue?"), not a quality score.

### Two properties that make the comparison fair

**Scale fairness.** Methods emit expression on different scales (raw counts,
log1p, arbitrary). Every primary metric is computed on **per-gene
rank-normalized** expression — invariant to any monotonic per-gene transform — so
two methods differing only in output scale get identical scores. The
rank-normalizer is imported from v2's evaluator rather than re-implemented, so
the two benchmarks cannot silently drift apart.

**Correspondence freedom.** Nothing here correlates prediction against ground
truth cell-by-cell. Generation synthesizes cells; it does not place them on GT
cells, so a manufactured correspondence measures alignment noise, not fidelity.

---

## Leakage policy

Inherited wholesale from v2 (`_v2bridge.py` re-exports `leakage_guard`), and it
matters more here, not less, because three sections are missing at once:

1. **Membership** — the held-out cells are physically absent from the file a
   method receives. `split_holdout` builds it, `assert_no_leakage` checks it, and
   the wrapper re-checks with `guard_no_holdout` before touching the data.
2. **Geometry** — methods get a *scalar target z* per held-out section and
   nothing else. Never the held-out `(x, y)`; the cell count is emergent.
3. **Registration** — the training slices are re-registered into a common frame
   using training slices only. For STARmap the policy is `none`: it is a single
   3-D imaging block whose z-planes are inherently co-registered, so
   re-registering would only introduce distortion. (Same call v2 makes for every
   volumetric dataset.)
4. **Global statistics** — label vocabularies are built from the training input
   only; expression normalization is per-cell.

The evaluation side *is* allowed to read the ground truth — that is what
evaluation means. The prediction→GT rigid alignment used by the binned-field and
localization metrics is an evaluation-side operation that feeds nothing back to
the method.

The `train_registered.h5ad` is built **once per holdout and reused by every
method**, so the comparison is apples-to-apples by construction.

---

## Verification status

Unlike v2 — which could not run anything in its authoring environment — the v3
harness was **executed and validated end to end** here, on the real STARmap data.
The methods themselves still need their conda environments (see below), but
everything v3 adds was run.

`python -m src.bench3.selftest` feeds the evaluator four synthetic
reconstructions of known quality and asserts the metrics order correctly.
Measured (paper design, all three held-out sections):

| metric | `oracle` | `flanking_copy` | `spatial_scramble` | `random` |
|---|---|---|---|---|
| `paper_umap_mixing` | +1.000 | +0.963 | +1.000 | +0.489 |
| `paper_morans_pearson` | +1.000 | +0.975 | +0.271 | −0.062 |
| `paper_gearys_pearson` | +1.000 | +0.976 | +0.282 | −0.058 |
| `paper_marker_field_r` | +1.000 | +0.862 | +0.004 | +0.021 |
| `paper_marker_depth_r` | +1.000 | +0.939 | +0.019 | +0.099 |
| `paper_gene_mean_spearman` | +1.000 | +0.985 | +1.000 | −0.013 |
| `paper_celltype_localization` | +0.982 | +0.754 | +0.049 | +0.008 |

* `oracle` = the real held-out cells → every metric at its ceiling.
* `flanking_copy` = the nearest *training* slice, copied → a strong but clearly
  sub-oracle baseline, which is what a method that ignores interpolation gets.
* `spatial_scramble` = real cells with coordinates permuted → every marginal
  perfect, every spatial relationship destroyed.
* `random` = expression from the pooled value distribution, uniform positions,
  random types → the floor everywhere.

The `spatial_scramble` column is the one that earns its keep. It scores **1.000**
on UMAP mixing and gene-mean similarity while collapsing on every spatial metric
— confirming that the distributional and spatial families measure genuinely
different things, and that no method can pass the panel by matching marginals
alone. This is exactly why the paper pairs a UMAP comparison with Moran's I and
Geary's C, and it is why v3 scores the four criteria as separate groups.

The figures reproduce the same story: `random` and `spatial_scramble` land flat
at Moran's I ≈ 0 and Geary's C ≈ 1 — the theoretical no-autocorrelation values —
while `oracle` and `flanking_copy` sit on the diagonal.

**Also verified here:** the dataset build (77 planes → 7 × 11), the leakage guard
on the real split (16 527 training cells from sections 1/3/5/7, 12 451 held out),
the shared-input cache, the prediction-format writer (the wrappers' own
`_v2_io.write_prediction_h5`), the merged three-evaluator path, aggregation,
ranking and all four figures.

**Not verified here:** the seven method wrappers, which need `bench_spatialcpa`,
`bench_spatialz`, `bench_feast` and `bench_isost`. They are v2's wrappers,
invoked unchanged.

---

## Usage

### Input data

`prepare_starmap` resolves the STARmap volume itself, trying in order:

```
$BENCH_V3_RAW_STARMAP
../data/starmap/STARmap_Wang2018three_data_3D_data.h5ad
../benchmark-pbya/data/raw/starmap_visual_cortex/STARmap_Wang2018three_data_3D_data.h5ad
../benchmark-pbya/data/processed/starmap_visual_cortex/data.h5ad
../benchmark-pbya/data/processed/starmap_visual_cortex.h5ad
```

So if the raw file sits in v1's standard `data/raw/` location, `prepare_starmap`
needs no arguments. Otherwise point it at the file:

```bash
python -m src.bench3.prepare_starmap --raw /path/to/STARmap_Wang2018three_data_3D_data.h5ad
# or
export BENCH_V3_RAW_STARMAP=/path/to/STARmap_Wang2018three_data_3D_data.h5ad
```

**Either the raw volume or the v1-*processed* `data.h5ad` works** — they are the
same cells and the same 89 z-planes, differing only in whether coordinates have
been converted to micrometres. `prepare_starmap` determines the units from the
file (native `obs['x','y','z']` are voxel indices; an obsm-only file is trusted
when `uns['spatial_metadata']['coordinate_units']` says micrometres), prints
which it used, and converts at most once. *Verified: raw, v1-processed, and an
obsm-only µm variant all produce bit-identical output.* Getting this wrong would
be silent — re-applying the 0.859 µm/voxel calibration to micrometres shrinks x
and y while leaving z alone — hence the explicit detection rather than an
assumption.

### Pipeline

```bash
cd benchmark-pbya-v3

# 1. build the paper dataset (once)
python -m src.bench3.prepare_starmap

# 2. inspect the design

python -m src.bench3.design

# 3. validate the harness without any conda env
python -m src.bench3.selftest

# 4. run the campaign (all 7 methods, shared input)
python -m src.bench3.run_all
python -m src.bench3.run_all --dry-run              # print the plan first
python -m src.bench3.run_all --methods spatialz feast
python -m src.bench3.run_all --design loo --skip-existing

# 5. results
python -m src.bench3.aggregate_results
python -m src.bench3.rank_methods --include-gen
python -m src.bench3.plot_paper_figures
```

Re-evaluate without re-running the methods (prediction and evaluation are
decoupled):

```bash
python -m src.bench3.evaluate_all --force
```

### Environments

The conda environments are v1's and are **shared**, not duplicated:

```bash
conda env create -f ../benchmark-pbya/envs/spatialcpa.yml   # bench_spatialcpa
conda env create -f ../benchmark-pbya/envs/spatialz.yml     # bench_spatialz
conda env create -f ../benchmark-pbya/envs/feast.yml        # bench_feast
conda env create -f ../benchmark-pbya/envs/isost.yml        # bench_isost
```

The orchestrator and evaluators need `numpy scipy pandas scikit-learn anndata
scanpy h5py matplotlib` plus `umap-learn` for the UMAP comparison (without it,
`paper_umap_mixing` is `None` and only the deterministic PCA mixing is reported).

Path overrides: `BENCH_V3_RAW_STARMAP`, `BENCH_V3_DATA`, `BENCH_V3_RESULTS`.

---

## Layout

```
src/bench3/
  config.py              protocol constants, method registry, metric names
  prepare_starmap.py     raw volume -> the 7-section paper dataset
  design.py              the paper holdout (2/4/6) and the LOO robustness check
  _v2bridge.py           imports v2's leakage guard / evaluators / resource monitor
  run_benchmark.py       one method: shared input -> wrapper -> merged metrics.json
  run_all.py             the 7-method campaign
  evaluate_paper.py      the paper's validation strategy, as metrics
  evaluate_all.py        re-evaluate predictions without re-running methods
  aggregate_results.py   all_metrics / per_section_metrics / summary_by_method CSVs
  rank_methods.py        rank by criterion + composite
  plot_paper_figures.py  UMAP, marker maps, Moran/Geary scatter, summary heatmap
  selftest.py            validate the harness with known-quality reconstructions

data/processed/starmap_visual_cortex/data.h5ad   built by prepare_starmap (gitignored)
results/                                         predictions, metrics, figures (gitignored)
```

The dataset follows v1/v2's `data/processed/<dataset>/data.h5ad` convention but
lives under `benchmark-pbya-v3/`, because it is a *differently partitioned* view
of the same volume — seven paper sections rather than the raw 89 z-planes — and
must not be confused with, or written over, v1's processed copy. Which protocol
produced a result stays visible in the holdout id (`paper_2_4_6`) and in
`uns['paper_protocol']`. Override the location with `BENCH_V3_DATA`.

The package is `bench3`, not `benchmark`, on purpose: v2's package *is* `benchmark`
and v3 imports it, so the two must not shadow each other.

### Why the method wrappers live in v2

`config.METHODS` points at `benchmark-pbya-v2/src/benchmark/methods/run_*.py`.
Those are plain CLI scripts over the stable `_v2_io` contract, so reusing them
rather than forking guarantees v3 and v2 exercise the *same* synthesis code — a
fix or a tuning change propagates to both, and a difference between the two
benchmarks can only come from the protocol, never from the method.

What is *not* shared is the wrappers' ablation flags: their defaults follow
whichever variant v2 was last tuning, so inheriting them would let a v3 run change
— or fail outright — because of an unrelated edit in v2. A method may therefore
pin the configuration v3 runs it in via `wrapper_args` in `config.METHODS`
(`spatialcpav8_gen` pins `--placement smooth_morph --expression-mode endpoint`,
the packaged v8's own defaults — which v2's wrapper now also defaults to, so the
pin is a no-op that simply keeps the run reproducible from `config.py` if that
default moves again). `run_benchmark` appends those after the shared
`_v2_io` arguments and before any extras you pass on the command line, so an
explicit flag still overrides the pin:

```bash
python -m src.bench3.run_benchmark --method spatialcpav8_gen -- --placement backbone
```

`env` in `config.METHODS` does the same job for knobs that live in the *method
package* and have no wrapper flag, so v3 configures a run without editing v2.
`spatialcpav11_gen` pins `SPATIALCPAV11_FALLBACK_ON_ERROR=0`: v11 degrades to a
deterministic OT-morph layout when training fails, which is reasonable for a
library and wrong for a benchmark — the fallback is not v11, but it still writes
a prediction, and its numbers look plausible enough to be ranked without anyone
noticing. With the pin, a training failure fails the run. A value already set in
your shell wins over the pin, so `SPATIALCPAV11_FALLBACK_ON_ERROR=1 python -m
src.bench3.run_all …` restores the old behaviour if you want to benchmark the
fallback deliberately.

### Running spatialcpav11_gen with the real OmiCLIP teacher

```bash
python -m src.bench3.run_all --methods spatialcpav11_gen \
  -- --teacher omiclip \
     --teacher-weights /path/to/omiclip/checkpoint.pt \
     --gpu-mem-fraction 0.4
```

OmiCLIP ships an open_clip *training* checkpoint, which pickles numpy scalars
alongside the tensors; torch ≥ 2.6 loads with `weights_only=True` and rejects
those. The teacher loader allowlists exactly those numpy globals and retries, so
this works unattended — you should see `checkpoint contains numpy scalars; loaded
with those allowlisted` followed by `neural fields trained: True`. If a checkpoint
needs the *full* pickle, set `SPATIALCPAV11_TRUST_CHECKPOINT=1` (it can execute
code from the file), or convert it once to a plain `state_dict` and point
`--teacher-weights` at that.

## Reading the results next to v2

v3 and v2 answer different questions and their numbers are not interchangeable:

* v2 asks *"which method generalizes across 17 datasets under leave-one-out?"*
* v3 asks *"under the published STARmap protocol, on the criteria that paper
  validated, how do these seven methods compare?"*

v3 is a harder holdout (three sections missing at once) on one dataset. Use v2 for
breadth and v3 for reproducing and extending a published result. The shared
`gen_*` columns are the bridge between them.
