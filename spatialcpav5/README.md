# SpatialCPA-v5 — Transformer for 3D Virtual Slice Generation

A **new, self-contained** implementation of SpatialCPA that predicts a tissue
section directly from its two neighboring sections using a Transformer:

```
{ Slice(i-1), Slice(i+1) }  ──►  Slice(i)
```

This is *separate* from and does **not** modify the original coordinate-field
`spatialcpa/` package. The two can coexist and be benchmarked side by side.

---

## What's new in v5 (vs v4)

v5 keeps v4's architecture, training, and losses **unchanged** and improves the
three generation steps the correspondence-free benchmark actually scores — *how
many* cells the virtual slice has, *where* they sit, and *what type* they are —
purely on the inference/synthesis side:

- **Interpolation placement (`position_source="interpolate"`, new default).**
  Instead of integrating a density field (mis-calibrated cell counts) or stacking
  both flanking slices (doubled density), v5 matches each lower-slice cell to its
  nearest upper-slice cell and places **one** synthesized cell per pair at the
  z-interpolated position. The cell **count** is the z-interpolated flanking count
  `N ≈ (1-t)·N_lo + t·N_hi`, and the density/morphology track the true tissue
  (adjacent sections change slowly). Uses only the training flanking slices — no
  held-out information — so it stays leakage-safe.
- **Cell-type transfer (`celltype_source="transfer"`, new default).** Each
  synthesized cell inherits the type of the same nearest training cell its
  expression is transferred from, so type, expression, and neighborhood are
  mutually consistent. The predicted cell-type composition and spatial
  organization then track the flanking slices (which closely match the held-out
  slice for adjacent sections), instead of the label head's noisier argmax.
- **Transfer expression by default (`expression_mode="transfer"`).** Preserves
  real cell-to-cell variance and gene–gene structure.

On the STARmap volumetric holdout these lift the correspondence-free primary
metrics substantially (e.g. gene-gene coexpression 0.71→0.90, Moran's-I agreement
0.51→0.85, cell-type composition 0.59→0.89, cell-type neighborhood 0.37→0.69) and
bring the synthesized cell count from badly mis-calibrated to within a few percent
of ground truth — while leaving the trained model identical. All new behavior is
config/CLI-gated, so v4's regimes remain available for ablation.

---

## Why a transformer?

The original SpatialCPA fits a continuous neural field `h(x, y, z) → (labels,
expression)` over the whole volume. SpatialCPA-v5 instead frames interpolation
as a **set-to-vector** problem: for each target location, its immediate
neighbors in the flanking slices are the most informative evidence, so we let a
Transformer *attend* over them and pool the result into a per-target latent.

Concretely, for every target spot we:

1. find its `k` nearest neighbors in the lower slice and `k` in the upper slice
   (Euclidean distance on physical `(x, y, z)`, via a KDTree);
2. turn each neighbor into a **token** (expression + labels + relative
   position);
3. run a Transformer encoder over the `2k` tokens plus a learnable `CLS` token;
4. decode the `CLS` latent with three heads.

---

## Architecture

```
 neighbor spots (2k)
        │
        ▼
  ┌───────────────┐     each token =  expr_proj(ExprEncoder(expr))
  │ TokenEmbedder │                 + RelCoordEncoder([Δx,Δy,Δz,‖Δ‖])
  │               │                 + side_embed(lower/upper)
  └───────────────┘                 [+ cell_type_embed] [+ region_embed]
        │
        ▼
  [CLS] + tokens ──►  Transformer Encoder (pre-LN, GELU)  ──►  CLS latent
                                                                  │
             ┌───────────────────────┬──────────────────────────┤
             ▼                       ▼                            ▼
      ExpressionHead            LabelHead                  OccupancyHead
     (full gene vector)   (cell type / region)        (tissue vs background)
```

### Module map

| File | Responsibility |
|------|----------------|
| `config.py`      | All hyperparameters as dataclasses (`SpatialCPAv5Config`). Nothing is hard-coded elsewhere. |
| `data.py`        | `Slice`, `SliceStack` (flattened global spot table), KDTree neighbor search, background/negative sampling, triplet-sample construction with on-disk caching. |
| `dataset.py`     | `TripletTokenDataset` — gathers neighbor tokens lazily from the shared table. |
| `encoders.py`    | `ExpressionEncoder` interface + `linear`/`mlp` implementations (registry-based, swappable), and `RelativeCoordEncoder`. |
| `tokens.py`      | `TokenEmbedder` — sums the typed component embeddings into `hidden_dim` tokens. |
| `transformer.py` | `TransformerAggregator` — CLS token + `nn.TransformerEncoder`. |
| `heads.py`       | `ExpressionHead`, `LabelHead`, `OccupancyHead`. |
| `losses.py`      | Masked MSE, Pearson loss, masked cross-entropy, occupancy BCE, weighted total. |
| `model.py`       | `SpatialCPATransformer` — assembles everything. |
| `trainer.py`     | `Trainer` — mini-batch, val split, AMP, schedulers, grad clip, checkpoints, early stopping, TensorBoard. |
| `inference.py`   | `Predictor` — `predict_slice` and `generate_virtual_slice`. |

---

## Tokens

Each neighbor token is the **sum** of component embeddings (all projected to
`hidden_dim`), which keeps a fixed width and lets components be toggled by data
availability:

- **expression** — a learnable `Linear` (default) projects the raw gene vector
  into a latent embedding. The gene dimension never enters the transformer. The
  encoder is chosen from a registry (`encoders.EXPRESSION_ENCODER_REGISTRY`) so
  it can be replaced by an autoencoder or a pretrained model (scGPT / Geneformer)
  by registering a new builder — no model changes required.
- **cell type / region** — `nn.Embedding` (index 0 reserved for unknown/pad).
  Included only if the data provides those annotations.
- **relative coordinates** — an MLP over `[Δx, Δy, Δz, ‖Δ‖]`, normalised by an
  estimated `coord_scale` buffer so inputs are unit-agnostic.
- **side** — a 2-way embedding distinguishing the lower vs upper slice.

---

## Prediction heads & losses

| Head | Output | Loss |
|------|--------|------|
| Expression | full gene vector (regression) | `mse_weight · MSE + pearson_weight · (1 − Pearson r)` |
| Label      | cell type and/or region logits | Cross-entropy (masked to supervised samples) |
| Occupancy  | tissue-vs-background logit | Binary cross-entropy |

**Total loss** (weights all in `LossConfig`):

```
total = expression_weight · (mse_weight·MSE + pearson_weight·Pearson)
      + label_weight      · (cell_type_weight·CE_ct + region_weight·CE_reg)
      + occupancy_weight  · BCE
```

**Occupancy negatives** are generated automatically: random coordinates inside
the XY bounding box but at least `negative_min_dist_factor × median_spacing`
away from any real spot (label = background = 0). Real spots have occupancy = 1.
Expression/label losses are masked off for background samples.

---

## Training features

`Trainer` implements: mini-batch `DataLoader`, train/val split, **mixed
precision** (`torch.cuda.amp`) on CUDA, AdamW with **cosine (warmup) or
plateau** scheduler, gradient clipping, **checkpoint saving** (`best.pt` /
`last.pt`), **early stopping** on validation loss, and optional **TensorBoard**
logging. KDTree neighbor indices are computed once and cached (in memory, and
optionally on disk keyed by a geometry hash).

---

## Quick start

```python
from spatialcpav5 import (
    Slice, SliceStack, SpatialCPAv5Config, SpatialCPATransformer,
    build_triplet_samples, Trainer, Predictor,
)

# 1. Wrap each aligned section as a Slice.
slices = [Slice(expression=Xi, coords_xy=xyi, z_values=zi,
                cell_type_indices=cti, section_id=f"s{i}") for i, ...]
stack = SliceStack(slices)

# 2. Config — every knob lives here.
cfg = SpatialCPAv5Config()
cfg.data.n_neighbors = 10
cfg.train.epochs = 100

# 3. Build samples + model + train.
samples = build_triplet_samples(stack, n_neighbors=cfg.data.n_neighbors,
                                negative_ratio=cfg.data.negative_ratio)
model = SpatialCPATransformer(
    n_genes=stack.n_genes, n_cell_types=..., n_regions=None,
    cfg=cfg.model, coord_scale=stack.estimate_coord_scale())
Trainer(model, stack, samples, cfg).train()

# 4. Predict an existing (held-out) slice from its neighbors.
pred = Predictor(model, gene_names, cell_type_names,
                 n_neighbors=cfg.data.n_neighbors
        ).predict_slice(target_slice, lower_slice, upper_slice)

# 5. Generate a brand-new virtual slice at arbitrary z.
vslice = Predictor(model, gene_names, cell_type_names,
                   n_neighbors=cfg.data.n_neighbors).generate_virtual_slice(
    z=3.5, slices=stack.slices, n_grid_points=1000, occupancy_threshold=0.5)
```

---

## Benchmark integration

`benchmark-pbya-v2/src/benchmark/methods/run_spatialcpav5.py` implements the v2
**generation-only** contract (training-only, re-registered input + a scalar
target z; the held-out `(x, y)` are never passed) and writes the standard
`prediction.h5`, so it runs under the v2 `run_benchmark.py` unchanged:

```bash
python -m src.benchmark.run_benchmark \
    --method spatialcpav5_gen --dataset starmap_visual_cortex \
    --holdout-json one_holdout.json
```

It is registered as method `spatialcpav5_gen` in `benchmark-pbya-v2`'s
`config.py`. All hyperparameters are CLI flags (`--help`). The wrapper trains the
transformer on the training slices, then synthesizes each held-out section de novo
at its target z. The cell count is **emergent** (interpolated from the flanking
slices), so only the target z — a position, not content — comes from the held-out
section and no held-out information leaks.

Key generation knobs (v5 defaults in **bold**): `--position-source`
{**interpolate**, density, flanking, grid}, `--celltype-source` {**transfer**,
predict}, `--expression-mode` {**transfer**, regress, blend}, and
`--occupancy-threshold` (default 0.5).

---

## Reducing over-smoothing (expression variance)

MSE regression predicts the conditional mean of the neighbors, so a purely
regressed slice is over-smooth: it reproduces gene-gene structure and mean levels
but collapses cell-to-cell variance (near-zero per-gene variance agreement at
evaluation). Three knobs address this:

- **Non-negative expression head** — `ModelConfig.expression_activation`
  (`"softplus"` default) keeps predictions ≥ 0 (a linear head can emit
  unphysical negatives).
- **Variance-matching loss** — `LossConfig.variance_weight` (default 0.5)
  penalizes mismatch between the per-gene std of predictions and targets.
- **Expression transfer at generation** — `InferenceConfig.expression_mode`:
  `"transfer"` copies real profiles from the nearest training cells (as SpatialZ
  and the original SpatialCPA do), restoring full cell-to-cell variance;
  `"blend"` mixes it with the regression (`transfer_alpha`); `"regress"` is the
  smooth baseline. Transfer uses only training cells → no leakage.

## Cell placement, count, and annotation (v5)

Where synthesized cells sit, **how many** there are, and **what type** they are —
the three steps that dominate the correspondence-free score — are set by
`InferenceConfig.position_source` and `celltype_source`:

- **`"interpolate"` (v5 default)** — match each lower-slice cell to its nearest
  upper-slice cell and place **one** synthesized cell per pair at the
  z-interpolated position `(1-t)·p_lo + t·p_match`, `t = (z-z_lo)/(z_hi-z_lo)`.
  The **count** is the z-interpolated flanking count `N ≈ (1-t)·N_lo + t·N_hi`
  and the density/morphology track the true tissue. On the STARmap holdout this
  hits 216 synthesized cells against a ground truth of 211, versus a badly
  mis-calibrated density-integral count. Uses only the training flanking slices —
  leakage-safe.
- **`"density"`** — a `DensityHead` predicts a continuous intensity field `λ(x)`;
  positions are sampled `∝ λ` and the count is the integral `N ≈ Σλ·A_cell`. Fully
  de-novo, but the integral is easily mis-calibrated (over- or under-counts).
- **`"flanking"`** — the real `(x, y)` of **both** flanking slices stacked:
  realistic per-slice density but roughly double the true count.
- **`"grid"`** — a uniform lattice (uniform density; ablation baseline).

**Cell-type annotation** — `celltype_source`:

- **`"transfer"` (v5 default)** — each synthesized cell inherits the type of the
  same nearest training cell its expression is transferred from, so type,
  expression, and neighborhood are mutually consistent. Composition and spatial
  organization then track the flanking slices. On STARmap this lifts cell-type
  composition 0.69→0.89 and neighborhood agreement 0.52→0.69 over the label-head
  argmax, at no cost to the other metrics.
- **`"predict"`** — the label head's argmax (v4 behavior).

The occupancy head still gates the footprint, and everything derives from training
cells + the target z only (no leakage).

## Designed-in extensibility

The following can be added **without major refactoring**:

- **Attention-based neighbor selection** — replace the fixed k-NN gather in
  `data.py` / `dataset.py`; tokens/model are agnostic to how neighbors are chosen.
- **Cross-attention / graph / Perceiver aggregation** — swap
  `TransformerAggregator` only; heads consume any `(B, hidden_dim)` latent.
- **Uncertainty / variational latent** — add a head in `heads.py` (e.g. predict
  variance) or make the latent a distribution; other modules are unaffected.
- **More than two neighboring slices** — the token/side machinery generalises;
  extend sample construction to gather from N flanking slices and widen the
  side embedding.
- **Pretrained expression encoders (scGPT, Geneformer)** — register a builder
  with `encoders.register_expression_encoder(...)` and set
  `ModelConfig.expression_encoder`.
```
