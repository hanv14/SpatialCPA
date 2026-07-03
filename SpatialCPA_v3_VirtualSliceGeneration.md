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

`spatialcpa/virtual_slice.py :: VirtualSliceGeneratorV3` generates a complete
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

**D. Expression (generative)** — expression is **sampled**, not copied:
* `gaussian` model → sample from the learned per-gene `N(μ, σ²)`
  (`GaussianExpressionDecoder.sample`);
* `zinb` model → sample from the zero-inflated negative binomial;
* `mse` model (deterministic) → predicted mean **plus** an empirical same-type
  residual drawn from the neighbor cells, so variability is still injected.

`expr_temperature` scales the noise (0 = mean, 1 = full learned variance),
exposing the reconstruction↔generation trade-off explicitly.

## Model changes (generative expression)

To make expression genuinely generative for normalized data (not just a point
estimate), v3 adds a **Gaussian expression head**:

* `spatialcpa/heads.py`: `GaussianExpressionDecoder` (predicts `μ` and
  `log σ²`, has `.sample()`), plus `gaussian_nll`.
* `spatialcpa/model.py`: new `expression_mode ∈ {'mse','gaussian','zinb'}`
  (`use_zinb` still works and maps to `'zinb'`/`'mse'`), plus
  `sample_expression(...)` for all three modes.
* `spatialcpa/trainer.py`: trains the Gaussian head with NLL + a Pearson term
  on the mean.

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

On the STARmap 7-slice protocol (train 1,3,5,7 → generate 2,4,6) v3 produces
biologically plausible slices (cell-type composition r ≈ 0.93, Moran r ≈ 0.74)
with **zero leakage**. The copy-the-neighbor baseline scores higher on
point-fidelity metrics — expected, because these sections are nearly adjacent
(`z_scale ≈ 1`), so copying an almost-identical neighbor is a very strong
baseline. v3's advantage is structural, not on this metric: it can generate at
**arbitrary z and arbitrary angle where there is no neighbor to copy**, samples
**new** cells/types/profiles rather than duplicating real ones, and never
depends on the slice being produced. Higher `expr_temperature` favors novelty;
lower favors fidelity.

## Usage

```python
from spatialcpa import SpatialCPA, SpatialCPATrainer, VirtualSliceGeneratorV3
from spatialcpa.data import SpatialSection

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
