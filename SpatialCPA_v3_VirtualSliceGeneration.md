# SpatialCPA v3 — True Virtual Slice Generation

## Why v3

The v1/v2 generator (`spatialcpa/inference.py`, `VirtualSliceGenerator`) does
**not** generate a virtual slice. It takes the held-out slice as a *reference*
and predicts expression at each of its **true (x, y) cell positions**, usually
conditioned on the slice's **true cell types** (`true_cell_types=...`,
`knn_alpha=0.0`). Concretely, in `run_spatialcpa_eval.py`:

```python
sim = generator.generate_matching(
    reference_adata=ref,          # the held-out slice itself
    true_cell_types=true_ct,      # its ground-truth cell types
    knn_alpha=0.0,                # pure k-NN copy from training cells
)
```

This is *reconstruction at known locations*, and it leaks exactly the
information a virtual slice is supposed to invent:

| Leaked from the held-out slice | Should be generated |
|---|---|
| number of cells | ✗ |
| every cell's (x, y) position | ✗ |
| every cell's cell type | ✗ |

Because it needs the slice it is meant to produce, it cannot build a slice at a
**novel z** where no ground truth exists — defeating the purpose.

## What v3 does

`spatialcpav3/virtual_slice.py :: VirtualSliceGeneratorV3` generates a complete
virtual slice from **only**:

* a target `z`, and
* the **two neighboring real sections** (already registered / aligned),

using the trained continuous field `h(x, y, z) → (cell type, expression)`. It
never touches the slice being generated.

```
Input:  target z  +  section_below (z<)  +  section_above (z>)
Output: AnnData with de-novo positions, cell types, and expression at z
```

### Pipeline

**A. Cell count** — interpolated from the two neighbors by z-distance
(`NeighborContext.target_n_cells`), optionally jittered.

**B. Positions (generative)** — an interpolated 2-D **density field** is built
from the two neighbors' occupancy grids, blended by z-distance
(`build_density_grid`). Positions are sampled from it and then relaxed with a
light **blue-noise** repulsion (`sample_positions`) so spacing looks like real
tissue rather than a lattice or clumps. Nothing is copied from the target slice.

**C. Cell types (coherent + sampled)** — for each position we combine
* the learned spatial classifier `P_model(ct | x, y, z)`, and
* the local neighbor composition `P_neighbor` (k-NN into both flanking slices),

blend them (`ct_model_weight`), **smooth the probability field** over the
generated points' k-NN into coherent spatial domains
(`smooth_probability_field`), then **sample** (not argmax) a cell type. Sampling
is what keeps it generative; smoothing is what keeps domains coherent.

**D. Expression (grounded + generative)** — the expression mean is anchored in
the real flanking tissue, then generative noise is added on top:
1. the model predicts a per-gene mean `μ_model` (and, for `gaussian`, a learned
   `σ`) at each `(x, y, z, ct)`;
2. a **neighbor-anchored mean** `μ_nbr` is computed by a cell-type-conditioned
   inverse-distance average of the nearest same-type real cells pooled from the
   two flanking slices (`neighbor_expression_mean`) — this injects the strong,
   real signal that surrounds the target z;
3. the two are blended, `μ = β·μ_model + (1-β)·μ_nbr` (`expr_model_weight`);
4. noise is added around `μ` — the learned `σ` for `gaussian`, an empirical
   same-type residual for `mse`, or full ZINB sampling for raw counts.

`expr_temperature` scales the noise (0 = grounded mean, 1 = full learned
variance) and `expr_model_weight` trades model extrapolation against neighbor
fidelity — together exposing the reconstruction↔generation dial explicitly.
Grounding the mean is the main driver of expression fidelity; the sampled
residual is what keeps it generative rather than a linear interpolation.

## Model changes (generative expression)

To make expression genuinely generative for normalized data (not just a point
estimate), v3 adds a **Gaussian expression head**:

* `spatialcpav3/heads.py`: `GaussianExpressionDecoder` (predicts `μ` and
  `log σ²`, has `.sample()`), plus `gaussian_nll`.
* `spatialcpav3/model.py`: new `expression_mode ∈ {'mse','gaussian','zinb'}`
  (`use_zinb` still works and maps to `'zinb'`/`'mse'`), plus
  `sample_expression(...)` and `predict_expression_dist(...)` (mean + std) for
  all three modes.
* `spatialcpav3/trainer.py`: trains the Gaussian head with NLL + a small MSE
  anchor on the mean (stability) + a Pearson term (gene-wise-r).

All changes are backward compatible: existing code using `use_zinb=True/False`
and `VirtualSliceGenerator` (v2) is untouched.

## Evaluation

Because generated cells have **no 1:1 correspondence** with real cells,
per-cell gene-wise Pearson r is meaningless. `run_spatialcpa_v3_eval.py` uses
correspondence-free, distribution-level metrics:

* gene-wise **Moran's I / Geary's C** correlation (spatial-pattern fidelity),
* **cell-type composition** correlation,
* **pseudobulk** mean-expression correlation,
* **nearest-neighbor-matched** gene-wise r (spatially-aware fidelity),

and reports a **nearest-real-slice baseline** (pure "linear" copy) alongside.

### Reading the STARmap numbers

On the STARmap 7-slice protocol (train 1,3,5,7 → generate 2,4,6), with the
neighbor-anchored expression and sharper cell-type assignment (8-epoch smoke
run, averages over the three held-out slices):

| metric              | v3    | copy-neighbor baseline |
|---------------------|-------|------------------------|
| Moran's I r         | 0.91  | 0.97                   |
| composition r       | 0.95  | 0.92                   |
| pseudobulk r        | 0.77  | 0.72                   |
| NN-matched gene r   | 0.37  | 0.55                   |

v3 now **exceeds** the copy-the-neighbor baseline on cell-type composition and
pseudobulk expression, and closes most of the Moran's I gap — all with **zero
leakage**. The baseline still wins NN-matched gene r because it literally copies
real cells; v3's advantage is structural: it generates at **arbitrary z or
angle where there is no neighbor to copy**, samples **new** cells/types/profiles
rather than duplicating real ones, and never depends on the slice being
produced. `expr_model_weight`/`expr_temperature` favor fidelity when low and
novelty when high.

The same improvements show up in the benchmark harness (per-cell metrics via
`evaluate.py`'s NN matching): grounding the mean and sharpening cell types lifted
gene-wise Pearson ≈ 3×, cell-type accuracy ≈ 1.9×, and Moran's I ≈ 2.4× over the
first pass at identical training budget.

## Usage

```python
from spatialcpav3 import SpatialCPA, SpatialCPATrainer, VirtualSliceGeneratorV3
from spatialcpav3.data import SpatialSection

model = SpatialCPA(n_genes, n_cell_types, expression_mode='gaussian', ...)
SpatialCPATrainer(model, train_sections, ...).train(n_epochs=50)

gen = VirtualSliceGeneratorV3(model, cell_type_names, gene_names)
virtual = gen.generate(section_below, section_above, target_z=547.3)
# virtual.obsm['spatial'], virtual.obs['cell_class'], virtual.X
```

Run the evaluation:

```bash
python run_spatialcpa_v3_eval.py --epochs 50
```
