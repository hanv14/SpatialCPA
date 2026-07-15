# SpatialCPA-v12 — generative continuous 3D virtual-slice generation

v12 is a **generative** enhancement of v11's continuous implicit-field model for aligned
serial spatial transcriptomics **with no paired H&E**. It keeps v11's two coordinate
networks, queried at **arbitrary continuous `(x, y, z)`** — between *or beyond* the real
slices — and replaces v11's mean-regressing expression field with a **conditional
factor-analysis decoder** that samples cells with realistic gene–gene covariance:

```
Stage 1   LayoutField(x, y, z | neighbouring slices)      ->  occupancy + cell-type/region
Stage 2   GenerativeExpressionField(x, y, z | layout)     ->  N(mu, L Lᵀ + Ψ)  sampler
```

v12 targets the exact places where v11 loses to the training-free OT/copy family: v11's
MSE field collapses toward the conditional mean, so it loses gene–gene co-expression and
per-gene variance; and its coarse occupancy sampling gives an unstable density/field. v12
fixes both while staying **completely inside the neural-field paradigm** — it uses no
optimal transport, no barycentric / diffeomorphic morph, no two-slice OT fusion, and no
niche Markov-random-field (i.e. none of v8's machinery).

## What is new in v12 (vs v11)

1. **Conditional factor-analysis expression decoder** (`nets.GenerativeExpressionField`,
   `losses.factor_analysis_nll`). Stage 2 outputs a per-cell mean `mu(x,y,z,layout)` and
   holds a shared low-rank loading matrix `L` (G×r) plus per-gene idiosyncratic noise `Ψ`.
   The implied per-cell law is the Gaussian `N(mu, L Lᵀ + Ψ)`; `L, Ψ` are **warm-started
   from the real gene–gene covariance** and refined by a factor-analysis likelihood
   (Woodbury identity, so the G×G covariance is never formed). Sampling `x = mu + L·s`
   yields cells that carry **realistic gene–gene covariance and per-gene variance** — the
   structure a pure MSE field cannot represent — while remaining genuinely *generated*.

2. **Mean + structured-noise decode with a real, field-selected anchor.** The mean is the
   field `mu` blended with a *real* same-type profile retrieved by the learned layout
   (`anchor_weight`); the factor noise is **added** on top (`noise_scale`) rather than
   blended into the mean, so the biological covariance is injected without smearing the
   spatial mean field. This is a proper generative sampler (mean + calibrated noise), not
   a copy.

3. **Spatially-coherent latent field** (`_coherent_latent`). The factor code `s` is drawn
   as a smooth field over the synthesized cells' kNN graph (plus an idiosyncratic part),
   so spatially-variable genes stay spatially autocorrelated (Moran's I) instead of being
   scattered by i.i.d. per-cell noise.

4. **Quota cell-typing to the interpolated composition** (`_quota_assign`). Types are
   assigned so the population matches the z-interpolated flanking composition *exactly*,
   filling each type from its highest-confidence cells — matching composition while
   preserving the spatial niche arrangement. (A prior-correction, not an MRF.)

5. **Count-space output + robust decoding.** The field trains on log1p-normalized
   expression but emits count-like values (`output_counts`), so the evaluator's per-cell
   count normalization behaves as it does for the raw-count ground truth — which the
   scale-sensitive per-gene mean/variance metrics need. Finer occupancy grid and
   leakage-safe density/gene-stat calibration hooks are available.

Everything is **leakage-safe**: training, the teacher, rasters, the covariance
warm-start, and every interpolation target use the training slices only; only the scalar
target z queries the fields.

## Stage 1 — Layout Generator (unchanged from v11)

`LayoutField` outputs occupancy, a cell-type/region distribution, and a layout code, and
is trained by **knowledge distillation from a frozen multimodal foundation model**
(OmiCLIP / Path2Space text/expression tower; a data-derived proxy stands in when no FM
asset is supplied — clearly logged) plus **self-supervised slice reconstruction**. See
`teacher.py` (`--teacher omiclip|path2space|gene_embedding|proxy`).

## Stage 2 — Generative Expression Decoder (new)

`GenerativeExpressionField` is the conditional factor-analysis decoder described above.
Trained by expression **mean reconstruction** (MSE, anchors `mu`) **+ factor-analysis
NLL** (fits `L, Ψ` to the real covariance), with cross-z consistency and the v11
biology-informed constraints (interface preservation, within-domain gradient smoothness,
domain coherence).

## Inference (continuous z, generative)

`generate_virtual_slice(z)`: encode the flanking slices → sample the occupancy field
(z-marginalized) for `n_target` positions → assign types by quota to the interpolated
composition → evaluate the mean field `mu` → **sample** `expr = [(1-a)·mu + a·anchor] +
noise_scale·(L·s_coherent)` → emit count-like expression. Decode modes (`--expr-decode`):

| mode | expression |
|---|---|
| `generative` (default) | mean + additive coherent factor-analysis noise (covariance-preserving sampler) |
| `field` | deterministic mean `mu` only |
| `residual` | v11-style blend of `mu` with a real same-type profile |

## Validation status (honest)

Developed and validated end-to-end through the **real** `benchmark-pbya-v2` generation
evaluator (`evaluate_generation.py`) on synthetic multi-slice data in both regimes
(distinct-tissue and near-identical/volumetric), head-to-head against **v8's default**
(the strongest prior SpatialCPA generator), averaged over three held-out sections.

* **vs v11 (the version v12 enhances):** a clear, consistent improvement — v12 fixes
  v11's expression mean-collapse (co-expression, gene variance) and its unstable density.
* **vs v8:** **parity on distinct tissue** (v12 wins the expression-distribution metrics:
  gene mean/variance, composition, field-SSIM, sinkhorn) and **competitive on
  near-identical planes** (ties the gene-level metrics, wins composition/field-SSIM). The
  two honest residues are `morans_agreement` and `density_pearson` — the spatial-coherence
  metrics a real-slice copy + coherent warp (v8's design) is intrinsically built to
  dominate, the same "Pareto residue against a copy" v8's own README documents. v12 pays a
  little of that residue to remain a fully generative, non-copy model.

The processed real benchmark datasets and conda environments are not bundled here, so the
full cross-dataset leaderboard must be reproduced where the data live. No benchmark numbers
are fabricated. See `validation/VALIDATION.md` for the per-metric tables and how to
reproduce them.

## Running it

Registered in `benchmark-pbya-v2` as `spatialcpav12_gen` (needs PyTorch, in
`bench_spatialcpa`):

```bash
python -m benchmark.run_benchmark --method spatialcpav12_gen --dataset starmap_visual_cortex
# real OmiCLIP teacher:      ... --teacher omiclip --teacher-weights /path/omiclip.pt
# real gene-embedding:       ... --teacher path2space --gene-embedding /path/genes.npz
# deterministic mean field:  ... --expr-decode field
```

### Package layout

| module | role |
|---|---|
| `config.py` | all architecture / loss / training / inference hyperparameters |
| `nets.py` | Fourier features, context encoder, LayoutField, GenerativeExpressionField |
| `teacher.py` | FM teacher (OmiCLIP / Path2Space hook + data-derived stand-in) |
| `losses.py` | reconstruction, factor-analysis NLL, distillation, consistency, biology |
| `trainer.py` | rasterization, covariance warm-start, LOO training, generative inference |
| `model.py` | `SpatialCPAv12` — orchestration, normalization, fallback |
| `data.py` | `Slice` / `SliceStack` containers |
