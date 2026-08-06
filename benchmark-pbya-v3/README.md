# benchmark-pbya-v3 — the SpatialZ STARmap paper protocol, run on every method

A **paper-faithful reproduction benchmark**. Where `benchmark-pbya-v2` sweeps 17
datasets with leave-one-out and scores generation with its own metrics, v3 does
one thing: it reproduces the STARmap visual-cortex evaluation from the SpatialZ
paper (Lin et al. 2025, *Nature Methods*) exactly, and puts every method through
it — so a published method's numbers are reproducible on their own terms, and the
SpatialCPA variants are measured against them on identical footing.

Methods benchmarked: **spatialcpav8_gen, spatialcpav11_gen, spatialcpav14_gen,
spatialcpav15_gen, spatialcpav16_gen, spatialcpav17_gen, SpatialZ, FEAST,
isoST**.

Datasets: **STARmap visual cortex** (the paper's, `kind=paper`) plus **17
analogues** — every volume in `benchmark-pbya` that can be cut into sections at
all (`kind=analogue` — see [Datasets](#datasets)).

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

### Choosing the pose (alignment-dependent metrics only)

Two metrics compare *positions* — `paper_marker_field_r` and
`paper_celltype_localization` — so the prediction has to be rotated into the
ground truth's frame first. Everything else (both autocorrelation families, the
embedding mixing, the distributional metrics, `paper_marker_depth_r` up to the
laminar axis) is computed within each slice and is pose-invariant.

`src/bench3/align.py` picks that pose, on two rules:

* **Expression breaks the symmetry that geometry cannot.** Candidate rotations
  are scored by agreement between the *binned marker fields*, not by how many
  cells overlap. On tissue with a symmetric outline — a bilaterally symmetric
  coronal section, say — identity and 180° cover the ground truth equally well,
  so an occupancy score cannot separate them and the pose falls to noise; a flip
  inverts the marker gradient, so field agreement scores it about −1 and rejects
  it. When a dataset's marker panel resolves empty the aligner falls back to the
  genes with the highest ground-truth Moran's I.
* **Reflections are not candidates.** Physical tissue has fixed handedness, so a
  mirrored pose is wrong by construction however well it fits. Only proper
  rotations (det = +1) are searched.

Ties resolve toward the smallest rotation, so an already-registered dataset keeps
the identity pose. The chosen rotation, scale, score and runner-up score are
written into each section's entry in `metrics.json` (`align_rotation_deg`,
`align_score`, `align_runner_up`, …), with the largest rotation applied surfaced
at the top level as `align_rotation_deg_max` — a marginal decision is visible
rather than silent. The figures use the same aligner, so a marker map is drawn at
the orientation its reported `paper_marker_field_r` was computed at.

Measured on synthetic sections whose correct pose is known: on symmetric geometry
the occupancy aligner picked a wrong pose in 3 of 12 runs (`field_r` ≈ −0.94
where the correct pose gives +0.94); the expression-scored one, 0 of 12. Scored
end to end through `evaluate_paper` on a prediction delivered in an arbitrary
frame, `paper_marker_field_r` holds at ≈ 0.89 for rotations of 0°, 40° and 150°,
against 0.44–0.48 before.

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
the method. How that pose is chosen is described under
[Choosing the pose](#choosing-the-pose-alignment-dependent-metrics-only).

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

The pose selector has its own fixtures — two synthetic datasets in v3 format, one
with a symmetric outline and one asymmetric — which the self-test passes on both,
with the `oracle` probe holding the identity pose exactly (`align_rotation_deg` =
0 on every section) and `paper_marker_field_r` at +1.000. See
[Choosing the pose](#choosing-the-pose-alignment-dependent-metrics-only) for the
old-vs-new numbers.

**Also verified here:** every one of the eighteen registered datasets builds.
`python -m src.bench3.selftest_datasets` synthesizes a source shaped like each
dataset's real one and runs the real `prepare_dataset`, asserting the section
count, the hold-out id, that every held-out section is bracketed, that the marker
panel resolves, and that a gene cap never drops a marker or layer gene. All 18
pass. It also runs the coordinate-guard negative control described under [The
coordinate trap](#the-coordinate-trap-these-datasets-introduced). A dataset
registered without a case there fails the run, so a spec cannot be added and left
unexercised.

**Not verified here:** the nine method wrappers, which need `bench_spatialcpa`,
`bench_spatialz`, `bench_feast` and `bench_isost`. They are v2's wrappers,
invoked unchanged — except `spatialcpav16_gen`, whose wrapper lives here (see
[Why the method wrappers live in v2](#why-the-method-wrappers-live-in-v2)). And,
for the eleven datasets added after the original seven, the *readers against
their real files* — `selftest_datasets` proves the specs and
the partitioning, not that a column is spelled the way a distribution spells it.
Build each one once and read the printed section table before trusting a row.

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
python -m src.bench3.selftest            # the evaluator, on known-quality probes
python -m src.bench3.selftest_datasets   # every registered dataset's build

# 4. run the campaign (all 9 methods, shared input)
python -m src.bench3.run_all
python -m src.bench3.run_all --dry-run              # print the plan first
python -m src.bench3.run_all --methods spatialz feast
python -m src.bench3.run_all --design loo --skip-existing

# 5. results
python -m src.bench3.aggregate_results
python -m src.bench3.rank_methods --include-gen
python -m src.bench3.plot_paper_figures     # per dataset
python -m src.bench3.plot_cross_dataset     # every method x every dataset
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
  prepare_dataset.py     any registered volume -> its 7-section dataset
  sources.py             readers for source volumes (h5ad; ExSeq/Deep-STARmap
                         CSVs; IMC/Open-ST per-section h5ads; CosMx zips;
                         Allen ABC h5ad + CCF metadata CSV)
  prepare_starmap.py     the STARmap entry point (thin wrapper, unchanged CLI)
  design.py              the paper holdout (2/4/6) and the LOO robustness check
  _v2bridge.py           imports v2's leakage guard / evaluators / resource monitor
  assets.py              method assets v3 prepares (torch checkpoints it can load)
  sanitize_checkpoint.py runs in the method's conda env; tensors-only rewrite
  run_benchmark.py       one method: shared input -> wrapper -> merged metrics.json
  run_all.py             the 9-method campaign
  evaluate_paper.py      the paper's validation strategy, as metrics
  evaluate_all.py        re-evaluate predictions without re-running methods
  aggregate_results.py   all_metrics / per_section_metrics / summary_by_method CSVs
  rank_methods.py        rank by criterion + composite
  plot_paper_figures.py  UMAP, marker maps, Moran/Geary scatter, summary heatmap
                         — one dataset at a time
  plot_cross_dataset.py  every method x every dataset: bars, heatmap, rank profile
  nature_theme.py        the Nature/Springer figure theme the above draws in
  selftest.py            validate the harness with known-quality reconstructions
  selftest_datasets.py   validate every DATASET_SPECS entry builds, on synthetic
                         sources — plus the coordinate-guard negative control
  survey_datasets.py     screen benchmark-pbya's datasets for protocol fitness

data/processed/<dataset>/data.h5ad              built by prepare_dataset (gitignored)
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

The one exception is `spatialcpav16_gen`, whose wrapper is
`src/bench3/methods/run_spatialcpav16.py`: v2 never ran v16, and adding it there
would mean editing a frozen benchmark. It speaks the identical `_v2_io` contract
— same CLI, same `prediction.h5`, same leakage guard — so `run_benchmark` cannot
tell the difference. `spatialcpav17_gen` needs no such exception: v2 already
carries `run_spatialcpav17.py`, so v3 invokes v2's copy like every other method.

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

Two more per-method fields keep v3 self-sufficient, so that making a method run
correctly here never means editing v2 or a method package:

* **`sanitize_weight_args`** — flags whose value is a torch checkpoint. Before the
  wrapper runs, v3 checks the file loads under the method env's torch and, if not,
  substitutes a tensors-only copy in `results/_assets/` (same weights; `assets.py`).
* **`invalid_log_markers`** — strings in the method log that mean the run silently
  degraded to a fallback. v3 renames the prediction to `prediction.h5.degraded`,
  skips evaluation and fails the run. Three methods carry one: `spatialcpav11_gen`
  (OT-morph fallback when the neural fields fail to train), `spatialcpav16_gen`
  (depth-interpolated spectrum without torch) and `spatialcpav17_gen` (flanking
  resample-and-copy when torch is missing, or when training or generation
  raises). Each fallback is a sensible library default and an invalid benchmark
  row — v17's in particular is close to the `flanking_copy` probe the self-test
  uses as a *baseline*, so a degraded run would score respectably rather than
  visibly failing.

### Running spatialcpav11_gen with the real OmiCLIP teacher

```bash
python -m src.bench3.run_all --methods spatialcpav11_gen \
  -- --teacher omiclip \
     --teacher-weights /path/to/omiclip/checkpoint.pt \
     --gpu-mem-fraction 0.4
```

OmiCLIP ships an open_clip *training* checkpoint: it pickles numpy scalars and
optimizer state next to the tensors, and torch ≥ 2.6 loads with
`weights_only=True`, which refuses those globals (`Unsupported global: GLOBAL
numpy.core.multiarray.scalar`). open_clip loads through `torch.load` with that
default, so the teacher never builds — and v11 then quietly degrades to its
OT-morph fallback and still writes a prediction.

v3 handles both halves without touching the method. It writes a tensors-only copy
of the checkpoint (identical weights, nothing executable pickled, cached per
file+mtime) and passes *that* to the wrapper, so you should see:

```
  weights: converted tensors=… -> results/_assets/checkpoint.safe.<hash>.pt
```

followed by `teacher: real omiclip` and `neural fields trained: True` in the
method log. If the teacher still fails for some other reason, the second half
catches it: the run fails instead of contributing fallback numbers to the table.

Nothing here modifies the checkpoint you supplied.

## Datasets

Eighteen volumes. The first seven build from `data/raw/`; the rest are read from
`benchmark-pbya`'s **processed** `data.h5ad` (see [Where each source comes
from](#where-each-source-comes-from)).

| dataset | `kind` | partition | source | status |
|---|---|---|---|---|
| `starmap_visual_cortex` | `paper` | `planes` — trim z 6–13 / 91–94, split 77 planes into 7 × 11 | the paper's own volume | the reproduction |
| `exseq_visual_cortex` | `analogue` | `z_width` — cut the z range into 7 equal-width slabs (0.2 % outlier clip, not a noise trim) | `benchmark-pbya/data/raw/exseq_visual_cortex` (or v1's processed h5ad) | the same protocol, a second volume |
| `imc_breast_cancer` | `analogue` | `sections` — all 15 real serial sections at 10 µm, used as-is | `benchmark-pbya/data/raw/imc_breast_cancer` (15 h5ads, or v1's processed) | protein panel, human tumour |
| `cosmx_nsclc_3d` | `analogue` | `sections` — all 6 real cryosections, **30 µm** apart | `benchmark-pbya/data/raw/cosmx_nsclc_3d` (2 zips, or v1's processed) | widest *uniform* gaps |
| `deep_starmap` | `analogue` | `planes` — 0.70 µm optical planes grouped into 7 slabs, no trim | `benchmark-pbya/data/raw/deep_starmap` (3 CSVs, or v1's processed) | dense volume, mouse brain |
| `merfish_hypothalamus` | `analogue` | `sections` — 12 coronal sections, **50 µm** apart (animal 1) | `benchmark-pbya/data/raw/merfish_hypothalamus` (one CSV, all animals) | new tissue, wide gaps |
| `openst_lymph_node` | `analogue` | `sections` — 19 cryosections | `benchmark-pbya/data/raw/openst_lymph_node` (19 h5ad.gz) | human lymphoid tissue |
| `allen_merfish_brain` | `analogue` | `sections` — 59 sections, centred window | Allen ABC `MERFISH-C57BL6J-638850` | whole mouse brain, most sections |
| `allen_zhuang_abca1` | `analogue` | `sections` — centred window of 15 | Allen ABC `Zhuang-ABCA-1` | whole mouse brain |
| `allen_zhuang_abca2` | `analogue` | `sections` — centred window of 15 | Allen ABC `Zhuang-ABCA-2` | whole mouse brain, 2nd parcellation |
| `merfish_thick_cortex` | `analogue` | `z_width` — 100 µm block into 7 slabs (~14 µm) | Fang et al. 2023, Dryad | thick-tissue volume |
| `merfish_thick_hypothalamus` | `analogue` | `z_width` — 200 µm block into 7 slabs (~29 µm) | Fang et al. 2023, Dryad | thickest slabs in the benchmark |
| `easi_fish_lha1/2/3` | `analogue` | `z_width` — each sample into 7 slabs | Figshare 13749154 | three independent volumes, **tiny panel** |
| `exseq_breast_cancer` | `analogue` | `z_width` — **5** sections (the volume is small) | Alon et al. 2021, Zenodo | human tumour, ~3 100 cells |
| `st_mouse_brain_ortiz` | `analogue`, **spot** | `sections` — centred window of 15 of 75 | GEO GSE147747 | spot array, whole transcriptome |
| `visium_mouse_brain_c2l` | `analogue`, **spot** | `sections` — all 3 (mouse 1) | E-MTAB-11114 | spot array, thinnest experiment |

```bash
python -m src.bench3.prepare_dataset --dataset exseq_visual_cortex
python -m src.bench3.run_all --dataset exseq_visual_cortex
```

### Two flags that are not decoration

**`resolution: "spot"`** — `st_mouse_brain_ortiz` and `visium_mouse_brain_c2l`
are spot arrays, not single cells. Everything mechanical works, but a spot pools
tens of cells, so `paper_celltype_localization` scores *deconvolved composition*
rather than cells and the binned marker field is already binned by the array
geometry before v3 bins it. `rank_methods` prints the flag and says so. Read the
two spot rows against each other, never against a single-cell row — the same
discipline `kind` imposes between the paper dataset and the analogues.

**Size caps.** Three datasets arrive at a scale the earlier seven never reached,
so their specs carry caps that are applied **when the dataset is built** — so the
ground truth and every method's input hold the same cells and the same genes:

* `n_hvg` (ST, Visium: 3 000) — whole-transcriptome sources arrive at 20–35 k
  genes against the 28–1 000 of the targeted panels. Uncapped, the per-gene
  Moran's/Geary's families average over tens of thousands of near-empty genes,
  and any method that materializes a dense cell-by-gene matrix cannot load the
  input at all. The dataset's own marker and layer genes are force-included, so
  the selection can never silence `paper_marker_*`.
* `max_cells_per_section` (the three Allen atlases: 20 000) — those volumes run
  to millions of cells. The subsample is spatially stratified, so it keeps the
  tissue outline and the density gradient the field metrics read.

Both are recorded in `uns['paper_protocol']['gene_selection']` /
`['cell_subsample']` and overridable per run with `--n-hvg` /
`--max-cells-per-section`. `openst_lymph_node` is whole-transcriptome too and is
**not** capped here — see [Known gap](#known-gap-openst-is-still-uncapped).

Both write to `benchmark-pbya-v3/data/processed/<dataset>/data.h5ad` (override with
`$BENCH_V3_DATA`; the build prints the destination). ExSeq needs no arguments: it
resolves `benchmark-pbya/data/raw/exseq_visual_cortex` first, then v1's processed
h5ad — `$BENCH_V3_RAW_EXSEQ` or `--raw` override. The raw form is the spacejam2
cell-by-gene CSV, read directly by `sources.read_exseq_csv`, so v1's processing
pipeline does not have to have been run. Cell types come from `results_adata.h5ad`
beside the CSV when it is present and row-aligned; without it they stay `unknown`
and the `paper_celltype_*` group is unavailable, which the build says out loud.

Results, inputs and figures are keyed by dataset (`results/<method>/<dataset>/…`),
so the two never mix. The summary stage handles every dataset present in one go:

```bash
python -m src.bench3.evaluate_all       # each prediction against ITS own ground truth
python -m src.bench3.aggregate_results  # all_metrics/per_section carry a `dataset` column;
                                        # summary_by_method is per (dataset, method)
python -m src.bench3.rank_methods       # one ranking table per dataset — never pooled
python -m src.bench3.plot_paper_figures # results/summary/figures/<dataset>/…
```

`--dataset` takes a registered **name** or a path to a built `data.h5ad`, on every
stage that has it. Ranks are **within** a dataset: a composite is a position among the methods run on
that volume, so averaging STARmap's and ExSeq's composites would compare places in
two different races. Restrict any stage to one dataset with `--dataset-name`.

### Where each source comes from

**The original seven build from `data/raw/`.** v1's processed files are accepted
as an alternative, but none is required: `sources.py` reads each raw distribution
in its own form — ExSeq's cell-by-gene CSV, IMC's per-section h5ads,
Deep-STARmap's expression/spatial CSVs, and CosMx's two zips (the shipped h5ad
carries STIM coordinates in arbitrary units, so it is joined to the per-section
flat files for physical micrometres). Those readers mirror v1's processors rather
than calling them, so v3 stays self-contained.

**The Allen atlases build from either.** `sources.read_allen_ccf` reads the raw
distribution directly — an expression `.h5ad` beside the cell-metadata CSV that
carries the reconstructed CCF position and the cluster annotation — including the
mm → µm conversion, which is the step that is silently wrong if skipped, since
every other v3 dataset is micrometres. `reader_region` selects which Zhuang
parcellation.

**The remaining six read v1's *processed* `data.h5ad`.** `merfish_thick_*`,
`easi_fish_lha*`, `exseq_breast_cancer`, `st_mouse_brain_ortiz` and
`visium_mouse_brain_c2l` have no raw reader in v3, because their raw forms are a
Dryad archive with an R/Seurat fallback path, a MATLAB `.mat` of transcript
positions, and a per-slide Visium ZIP respectively — re-implementing those
faithfully is a much larger job than the CSV and h5ad readers above, and getting
one subtly wrong is the failure mode this benchmark is least able to detect. So
they require v1's processing pipeline to have been run, and the build says so if
the file is missing. Adding a raw reader later changes nothing else: it is one
entry in `sources.READERS` plus a `reader` key.

### The coordinate trap these datasets introduced

The Allen atlases copy their whole metadata CSV into `obs`, which puts
*section-local* `x`/`y`/`z` right next to the reconstructed volume position in
`obsm['spatial']`. `extract_xyz` prefers `obs['x','y','z']` — correct for
STARmap, catastrophic here: the build would assemble the stack out of unrelated
per-section frames and produce a dataset with the right section count, the right
cell counts, monotone section centres, and geometry that is noise.

Two things stop it. Those specs set `"coords": "obsm"`, which says which array to
believe rather than relying on a priority order. And `prepare_dataset` now checks
the property that actually distinguishes the two for any `partition="sections"`
dataset: in real serial sections every cell in a section shares that section's z,
so the spread *within* a section must be small against the gap *between*
sections. Comparable values mean z is a per-cell quantity and the build stops.

That second check matters because the existing z-monotonicity assertion in
`verify` cannot catch this — for `sections` the section index is *assigned* by
sorting on the section centres, so "centres increase with index" is true by
construction. Verified both ways: with `coords: "obsm"` the eight new datasets
build correctly, and flipping `allen_zhuang_abca1` back to `auto` on the same
source is refused by the new check rather than silently succeeding, which is what
it did before.

### Known gap: Open-ST is still uncapped

`openst_lymph_node` is whole-transcriptome like the two spot datasets, and it is
**not** given an `n_hvg` cap here. A `spatialcpav14_gen` run on it was killed by
the OOM killer: the wrapper densifies the training matrix, which at ~10⁶ cells ×
~2×10⁴ genes is order 80 GB. Capping it is the same one-line spec change the ST
and Visium entries already carry, but it changes the panel a *previously reported*
dataset was measured on, so it is left as a deliberate decision rather than folded
into this change. Until then, expect that run to fail.

**Why ExSeq.** Same tissue as STARmap — mouse visual cortex — so the marker genes
and the laminar-axis composite carry over unchanged, and it is the only candidate
in `benchmark-pbya` for which none of the metric definitions have to be
reinterpreted. It is an independent technology on independent tissue, which is
what a second dataset is *for*.

**About the IMC dataset.** It is the one case where the metrics change meaning, so
read its rows differently from the other two:

* *Protein, not RNA.* The markers are **panCK** (tumour/epithelial compartment)
  and **CD3** (T-cell infiltrate), following the published analysis of this volume.
  Matching ignores case and punctuation, so `panCK` finds `PanCK` or `pan-CK`; it
  is deliberately *not* a prefix match, because `CD3` would then silently select
  `CD31`. If a marker does not match, the build says so and suggests the closest
  panel names.
* *A compartment axis, not a laminar one.* A tumour has no cortical layers, but
  those same two markers define the axis that matters: the signed score is
  `z(CD3) − z(panCK)`, so its in-plane gradient points from tumour toward immune
  infiltrate. `paper_marker_depth_r` therefore reads as **"profile across the
  tumour–immune axis"** — derived from the ground truth exactly as the cortical
  version is. Empty both layer lists in `config.DATASET_SPECS` to fall back to the
  generic axis (the gradient of whichever channel is most spatially structured).
* *Real serial sections, so registration matters.* These are cut, mounted and
  imaged independently — not one imaging block — so the policy is `rigid`, not
  `none`. The training slices move into a common frame while the held-out ground
  truth stays in the original one, and the evaluation-side prediction→GT alignment
  absorbs the difference. That makes the alignment-dependent metrics
  (`paper_marker_field_r`, `paper_celltype_localization`) noisier here than on the
  two co-registered volumes; the autocorrelation and distributional families are
  unaffected.
* *Half the volume is unused.* The paper design needs 7 consecutive sections and
  the dataset has 15, so the centred window is kept and 8 sections are dropped.

### Sections have to be big enough to run

The build refuses a dataset with a section under 50 cells (`--min-cells-per-section`,
or `--allow-small-sections` to force it), and prints `cells/section: min/median/max`
with an imbalance warning otherwise. This is not a metric threshold — it is what the
*task* needs. A section that thin is useless as a method input (SpatialZ's
flanking-slice PCA asks for 20 components and fails below 20 cells; every
interpolation method needs a neighbourhood) and useless as ground truth (the binned
field, the depth profile and the per-type OT all become noise).

It matters most for `z_width`, where equal-width slabs on a volume whose cell
density falls off toward one end can leave the last slab nearly empty. The error
names the offending sections and suggests concrete `--z-trim-quantile` /
`--n-sections` combinations that would work.

### Trimming

Only STARmap's trim is protocol. Dropping `z = 6–13` and `91–94` comes from the
SpatialZ paper's own analysis of that volume, so it is always applied and
`--no-trim` is refused for it. The analogue datasets have no published trim and v3
does not invent one:

* `exseq_visual_cortex` clips 0.2 % of cells at each end of z — not to remove
  noise, but because equal-width binning takes its edges from `min(z)`/`max(z)`,
  where a few segmentation outliers would skew all seven slab boundaries.
  `--no-trim` uses the raw range; `--z-trim-quantile` sets it.
* `imc_breast_cancer` trims nothing: it uses all 15 of its sections. Pass
  `--n-sections` to take a smaller window, and `--section-trim low|center|high` to
  choose which.

### Section count and hold-out pattern

**Seven sections and the 2/4/6 split are STARmap's, because they are the paper's.**
They are pinned for that dataset and derived for every other one:

| dataset | sections | held out | holdout id |
|---|---|---|---|
| `starmap_visual_cortex` | 7 (pinned — the published design) | 2, 4, 6 | `paper_2_4_6` |
| `exseq_visual_cortex` | 7 (a choice: ≈11 µm slabs, like a cryosection) | 2, 4, 6 | `paper_2_4_6` |
| `imc_breast_cancer` | 15 (all it has) | 2, 4, …, 14 | `paper_alt7of15` |
| `cosmx_nsclc_3d` | 6 (all it has) | 2, 4 | `paper_2_4` |
| `deep_starmap` | 7 (a choice: ~14 µm slabs of 0.70 µm planes) | 2, 4, 6 | `paper_2_4_6` |
| `merfish_hypothalamus` | 12 (all animal 1 has) | 2, 4, …, 10 | `paper_alt5of12` |
| `openst_lymph_node` | 19 (all it has) | 2, 4, …, 18 | `paper_alt9of19` |

What actually carries over from the paper is the **alternating hold-out**, not the
number seven: hold out every even section, keep the first and last as input. At
n = 7 that is exactly 2/4/6; at n = 15 it is 2/4/…/14. Either way every held-out
section is bracketed by two input sections — so the task stays well-posed — and
about half the volume is missing, which is what makes it hard. Set `held_out` to an
explicit tuple in `config.DATASET_SPECS` to pin a different split, and
`--n-sections` to change the count.

**One more caveat.** `kind=analogue` is not decoration: the SpatialZ paper validated
this protocol on STARmap, so an ExSeq row is v3's extension of it. Report the two
separately and never pool their ranks.

### Adding another one

Two things decide whether a candidate is worth it, and `survey_datasets.py`
measures both against `benchmark-pbya`'s processed tree:

```bash
python -m src.bench3.survey_datasets                 # screen every v1 dataset
python -m src.bench3.survey_datasets --csv survey.csv
```

**Can it be cut into sections?** It must carry 3-D coordinates at single-cell
resolution, with either enough optical planes to slab (`kind=volume`, as STARmap:
77 planes → 7 × 11) or enough real serial sections to use directly
(`kind=serial`). Spot-resolution and 2-D datasets are out.

**Would it discriminate?** `flank_r` is the per-gene Moran's I correlation
between the two sections flanking a held-out one — what a method scores by
ignoring the interpolation and copying a neighbour. **STARmap measures 0.98**, so
on the autocorrelation family this benchmark already has almost no room between a
trivial copy and a perfect reconstruction. A second dense volume with ~1 µm
planes would land in the same place and add breadth without adding
discrimination; a dataset with *wider* section spacing would add both. Read that
column before anything else.

Mechanically, adding one means a new entry in `config.DATASET_SPECS` — source
paths, partition mode, trim, marker and layer genes, registration policy. Nothing
else changes: `prepare_dataset` writes the resolved protocol into
`uns['paper_protocol']`, and `design.py`, `evaluate_paper` and the figures all
read the sections and the marker panel from the built file, so a dataset can never
be scored against another one's genes.

The genes are the part that needs judgement: `MARKER_GENES` and the `LAYER_*`
composites are mouse cortex, and a non-cortex dataset needs its own chosen before
`paper_marker_*` means anything. The `markers` column reports how much of the
current panel a candidate carries; `prepare_dataset` records the intersection and
warns when it is empty.

## Cross-dataset figures

`plot_paper_figures` draws inside one volume — a UMAP, a marker map and an
autocorrelation scatter only mean anything against that volume's own ground
truth. `plot_cross_dataset` draws the other view, **every method across every
dataset**, which is the figure a paper reports:

```bash
python -m src.bench3.plot_cross_dataset
```

| figure | what it shows |
|---|---|
| `fig_cross_dataset_bars` | one panel per metric; datasets along x, a bar per method, each held-out section overlaid as a dot |
| `fig_cross_dataset_heatmap` | one panel per metric, methods × datasets, annotated — the compact overview |
| `fig_cross_dataset_ranks` | each method's composite rank *within* each dataset, side by side, with no average across them |

With all eighteen datasets built these get wide. There is no cap — the tick
layout adapts — but a figure for publication generally wants a chosen subset:
use `--dataset-order` to pick and order it (see below).

Each run also writes `cross_dataset_values.csv` — exactly the numbers drawn.

### Choosing and labelling what goes on the axes

Directory names make poor axis labels, so selection and labelling are the same
flag. **The keys of whichever flag you pass select and order that axis:**

```bash
python -m src.bench3.plot_cross_dataset \
  --method-order spatialz feast isost spatialcpav15_gen \
  --method-names spatialz=SpatialZ feast=FEAST isost=isoST \
                 'spatialcpav15_gen=SpatialCPA v15' \
  --dataset-names starmap_visual_cortex='STARmap V1' \
                  exseq_visual_cortex='ExSeq V1' \
                  cosmx_nsclc_3d='CosMx NSCLC' \
  --width 120
```

* `--method-order` / `--dataset-order` — membership and order.
* `--method-names` / `--dataset-names` — labels; and when the matching `*-order`
  flag is absent, these select and order too, so one flag does both jobs.
* Either form takes `NAME` or `NAME=Label`; an inline label on the order flag
  wins. A name with no results fails with the list of names that do have some.
* `--metrics` / `--metric-names` do the same for the metric panels.

Ranks are recomputed among exactly the methods plotted — a rank is a position in
a field, so narrowing the field changes it.

### The theme

`nature_theme.py` holds the Nature-family constraints in one place: the fixed
column widths (89 / 120 / 183 mm, `--width`, never above the 247 mm page), 5–7 pt
sans at final size, 0.5 pt hairlines, outward ticks, and PDF output with embedded
editable type. **The saved PDF is exactly the width asked for** — no tight
bounding box, because recropping to the ink would change the width and
production would then rescale the type. A 600 dpi PNG is written alongside for
viewing.

Colours are Okabe-Ito (the Color Universal Design set) minus black, with the pale
yellow swapped for a dark gold that survives print. The slot ordering is not a
preference: all 5040 orderings were scored against a white surface and this one
clears every gate on the adjacent pairlist — worst adjacent colourblind ΔE 15.8
against a target of 8. Two consequences are handled rather than ignored: past
four slots the palette no longer separates *every* pair, so the rank figure adds
a distinct marker per method and labels each line at its end; and two hues sit
below 3:1 on white, which is why every figure writes its CSV.

Seven methods is the palette's depth. An eighth raises an error rather than
cycling a hue, because two methods sharing a colour would still render.

**The default campaign is nine methods, so `plot_cross_dataset` needs a chosen
subset.** With results for all nine in the tree it stops with the count and the
remedy rather than drawing; pass `--method-order` (or `--method-names`, which
selects too) to name at most seven. Nothing else is affected: the tables, the
rankings and `plot_paper_figures` order by `METHOD_ORDER` but do not colour by
it, so they carry all nine.

## Reading the results next to v2

v3 and v2 answer different questions and their numbers are not interchangeable:

* v2 asks *"which method generalizes across 17 datasets under leave-one-out?"*
* v3 asks *"under the published STARmap protocol, on the criteria that paper
  validated, how do these nine methods compare?"*

v3 is a harder holdout (three sections missing at once) on one dataset. Use v2 for
breadth and v3 for reproducing and extending a published result. The shared
`gen_*` columns are the bridge between them.
