# SpatialCPA-v8 — coherent optimal-transport bridge for virtual-slice generation

SpatialCPA-v8 synthesizes a **virtual tissue slice at an arbitrary z** from only
(i) a scalar target z and (ii) the neighbouring *training* slices. It never sees
the held-out slice's `(x, y)` or molecular content — the slice is generated
de-novo — so it is a true virtual-slice generator, evaluated by the
leakage-hardened `benchmark-pbya-v2` harness.

It is **training-free** (pure `numpy`/`scipy`/`scikit-learn`), runs in the same
`bench_spatialcpa` conda environment as v4–v6, and takes seconds–minutes per
holdout (no GPU, no per-dataset fitting).

---

## The problem v8 solves

The benchmark measures a generated slice as a *field/distribution*, not a
cell-to-cell list. Its metrics split cleanly by what they depend on:

| depends on… | metrics |
|---|---|
| **expression values only** (position-free) | `coexpression_agreement`, `sinkhorn`, `gene_mean/var_pearson` |
| **(position, expression) jointly** | `morans_agreement` |
| **(position, label) jointly** | `celltype_composition`, `celltype_nhood_agreement` |
| **position/density only** | `field_pearson`, `field_ssim`, `density_pearson`, `dice_density` |

Across the prior methods this created a genuine **tension** (visible in the
STARmap results): copying a single clean flanking slice wins the
expression-structure metrics (`coexpression`, `morans`, `composition`, `nhood`,
`sinkhorn`) because it preserves *real* local structure, **but loses every
spatial-field metric** because it does not move to the interpolated in-between
shape. Conversely, interpolating positions between the slices wins the field
metrics **but smears the local structure**, because a cell's neighbourhood is now
drawn from unrelated parts of two slices. v4/v5/v6 each land on one side of this
trade-off; none wins both.

## The idea: a coherent (near-isometric) OT deformation

v8's core contribution is a placement that is **on both sides of the trade-off at
once**: the *smoothed optimal-transport morph* (`transport.smooth_morph`).

1. Copy the nearest flanking slice's **real** cells — so their expression,
   labels and *local* neighbourhood structure stay exactly real (this is what the
   position-free and `(position, expression)` structure metrics reward).
2. Compute each cell's displacement toward the other slice under an **entropic-OT
   barycentric map** (a soft correspondence combining physical and molecular
   distance).
3. **Spatially smooth that displacement field over the cells' kNN graph.** A
   noisy per-cell displacement scatters cells independently and destroys local
   structure; a *smoothed* field moves neighbouring cells *together*, i.e. it is a
   coherent near-isometric tissue deformation. Because the deformation is smooth,
   local neighbourhoods — and therefore Moran's I, co-expression and niche
   structure — are preserved as in a clean copy, while the **global footprint
   still morphs toward the interpolated in-between shape**, which is what the
   field/density metrics need.

This is the geometric point the prior versions missed: *displacement-interpolation
is only structure-preserving if the displacement field is spatially coherent.* v6
applied a raw per-cell barycentric map (coherent only when the slices are already
near-identical) or fell back to random mixing; v8 makes the coherence explicit and
regularized, so a single smooth morph keeps structure **and** matches the field.

### Adaptive routing (default)

Two flanking sections can be near-identical (thin volumetric z-planes) or genuinely
distinct tissue. v8 measures the OT-map displacement between them (in units of cell
spacing) and routes per holdout:

- **small displacement → smooth morph** — a coherent single-sheet deformation is
  both structure-preserving and field-accurate;
- **large displacement → real-cell interpolation** — here a single-slice morph
  would have to travel too far and would smear, so mixing real cells from *both*
  slices at their real coordinates is the better estimate of the true intermediate.

The chosen branch and the measured displacement are logged per holdout, so the
`adaptive_threshold` is auditable and tunable per dataset.

### The four pipeline steps

1. **Count** — emergent: the z-interpolated flanking cell count (never the held-out
   count).
2. **Placement** — adaptive coherent OT bridge (above).
3. **Annotation** — each cell's label is anchored on its *real* source cell, then
   refined by a foundation-model / spatial-interpolation prior and a **cell-cell
   communication (niche) Markov-random-field** that pins the synthesized slice's
   neighbourhood-enrichment architecture `P(neighbour=j | centre=i)` to the
   z-interpolated flanking niche — the leakage-safe estimate of the held-out 2D/3D
   communication structure. Composition is pinned to the interpolated mix.
4. **Expression** — the real profile of each cell's source (max variance, real
   gene–gene structure); optional `transfer`/`blend` denoising modes are available.

Each step touches a different output channel (position / label / expression), so
optimizing one cannot spoil another — the design brief's requirement that the four
steps complement rather than fight each other.

---

## Novelty (what has not been done before)

- **Spatially-regularized OT displacement interpolation** for slice generation:
  smoothing the barycentric displacement field turns McCann interpolation into a
  coherent near-isometric deformation, resolving the structure-vs-field trade-off
  that every prior generator is stuck on. To our knowledge no virtual-slice method
  imposes coherence on the transport displacement this way.
- **Bidirectional (symmetric) McCann bridge** (`transport.symmetric_bridge`,
  opt-in): projects *both* flanking populations through the same OT plan and draws
  them in the `(1−t):t` ratio — one coherent sheet that is also the correct
  mixture.
- **Metric-factorized, channel-separated synthesis**: position, label and
  expression are generated by independent stages aligned to how the benchmark's
  metrics factorize, so improvements compound instead of trading off.
- **Foundation-model priors, leakage-safe**: pretrained gene embeddings
  (scGPT / Geneformer / Gene2vec) or a data-derived co-expression program space
  can drive the OT cost and annotation (`embedding.py`, `foundation_assets.py`),
  injecting external biology without ever touching the held-out slice.

---

## Validation status (honest)

The method was developed and unit/ablation-tested against the actual
`benchmark-pbya-v2` evaluators (`evaluate.py` + `evaluate_generation.py`) on
synthetic multi-slice datasets in **both** regimes:

- **near-identical / volumetric regime** (the STARmap-like case where v6 was
  beaten by v5/SpatialZ): v8's smooth morph **strictly dominates v6** across the
  structure metrics *and* the field/density metrics simultaneously (e.g.
  co-expression, composition, `field_pearson`, `density_pearson`, cell-type
  accuracy all improve together);
- **distinct-tissue regime** (IMC-like): v8 routes to real-cell interpolation and
  **matches v6** (no regression).

The two `*_morans_i_*_median` scores (the *magnitude* of the prediction's own
spatial autocorrelation, with no ground-truth target in the metric) can be
slightly lower for the coherent morph than for both-slice overlay; this is a
property of the metric, not a loss of fidelity.

The processed benchmark datasets and conda environments are not bundled in this
repository, so the full cross-dataset leaderboard must be reproduced where the data
live (see below). No benchmark numbers are fabricated here.

---

## Running it

Registered in `benchmark-pbya-v2` as `spatialcpav8_gen`:

```bash
cd benchmark-pbya-v2
python -m benchmark.run_benchmark --method spatialcpav8_gen --dataset starmap_visual_cortex
python -m benchmark.run_benchmark --method spatialcpav8_gen --dataset imc_breast_cancer
```

The harness passes only the generation-only interface
(`--input/--target-section/--target-z/--output/--seed`); every default in
`SpatialCPAv8Config` is the intended production setting. Ablation knobs
(`--placement`, `--adaptive-threshold`, `--smooth-iters`, `--density`,
`--embedding fm_gene --fm-gene-embedding …`, …) are exposed on the wrapper for
tuning and are documented in `run_spatialcpav8.py`.

### Package layout

| module | role |
|---|---|
| `config.py` | all knobs (7 stage dataclasses) — defaults are production settings |
| `data.py` | `Slice` / `SliceStack` containers, flanking-slice selection |
| `embedding.py` | cell-state embedding (PCA / co-expression / pretrained gene embedding) |
| `transport.py` | **smooth morph**, symmetric McCann bridge, entropic-OT plan |
| `density.py` | optional density calibration to the interpolated field (opt-in) |
| `annotation.py` | OT-anchor + FM/spatial prior + niche MRF cell typing |
| `communication.py` | neighbourhood-enrichment niche model (cell-cell communication) |
| `generator.py` | orchestration: count → placement → annotation → expression |
| `foundation_assets.py` | convert pretrained gene embeddings to the expected format |

Leakage safety mirrors the other v2 methods: the input is training-only
(re-checked by `guard_no_holdout`), label vocabularies and every statistic are
built on the training slices only, expression normalization is per-cell, and only
the scalar target z positions the synthesized slice.
