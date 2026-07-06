# SpatialCPA-v4 — Transformer for 3D Virtual Slice Generation

A **new, self-contained** implementation of SpatialCPA that predicts a tissue
section directly from its two neighboring sections using a Transformer:

```
{ Slice(i-1), Slice(i+1) }  ──►  Slice(i)
```

This is *separate* from and does **not** modify the original coordinate-field
`spatialcpa/` package. The two can coexist and be benchmarked side by side.

---

## Why a transformer?

The original SpatialCPA fits a continuous neural field `h(x, y, z) → (labels,
expression)` over the whole volume. SpatialCPA-v4 instead frames interpolation
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
| `config.py`      | All hyperparameters as dataclasses (`SpatialCPAv4Config`). Nothing is hard-coded elsewhere. |
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
from spatialcpav4 import (
    Slice, SliceStack, SpatialCPAv4Config, SpatialCPATransformer,
    build_triplet_samples, Trainer, Predictor,
)

# 1. Wrap each aligned section as a Slice.
slices = [Slice(expression=Xi, coords_xy=xyi, z_values=zi,
                cell_type_indices=cti, section_id=f"s{i}") for i, ...]
stack = SliceStack(slices)

# 2. Config — every knob lives here.
cfg = SpatialCPAv4Config()
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

`benchmark-pbya/src/benchmark/methods/run_spatialcpav4.py` exposes the **same
CLI and `prediction.h5` output** as every other benchmark method, so it runs
under `run_benchmark.py` unchanged:

```bash
conda run -n bench_spatialcpa python \
    src/benchmark/methods/run_spatialcpav4.py \
    --input data/processed/<dataset>/data.h5ad \
    --holdout-sections <section> \
    --output results/spatialcpav4/<dataset>/loo_<section>/prediction.h5 \
    --seed 42
```

It is registered as method `spatialcpav4` in `config.py`. All hyperparameters
are CLI flags (`--help`).

---

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
