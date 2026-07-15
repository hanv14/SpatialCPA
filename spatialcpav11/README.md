# SpatialCPA-v11 — two-stage continuous 3D virtual-slice generation

v11 is a **continuous implicit-field** deep-learning model for aligned serial spatial
transcriptomics **with no paired H&E**. Two coordinate-network fields are queried at
**arbitrary continuous `(x, y, z)`** — between *or beyond* the real slices:

```
Stage 1   LayoutField(x, y, z | neighbouring slices)  ->  occupancy + cell-type/region
Stage 2   ExpressionField(x, y, z | Stage-1 layout)   ->  gene-expression profile
```

The layout is generated first; expression is generated **conditioned on that layout**.
Both fields are conditioned on (i) a permutation-invariant encoding of the aligned
neighbouring real slices (features + a low-res spatial rasterization sampled at the
query point) and (ii) a **Fourier encoding of the continuous z**, which is what lets
the model interpolate/extrapolate smoothly along the axis.

## Stage 1 — Layout Generator

`LayoutField` (implicit MLP over Fourier-encoded `(x,y,z)` + context) outputs, at any
query point, an **occupancy** logit, a **cell-type/region** distribution, and a
**layout code** handed to Stage 2. It is trained by:

- **Knowledge distillation from a frozen multimodal foundation model** (OmiCLIP /
  Path2Space), used as a *teacher* even though no images are present:
  - *feature alignment* — the student layout code is aligned (cosine) to the teacher's
    per-spot embedding of the real slice;
  - *pseudo-layout targets* — the teacher's spatial-domain clustering supervises the
    layout code via a domain head.

  Two **real** teachers are implemented (`teacher.py`) and selected via `--teacher`:
  - **`omiclip`** — the actual OmiCLIP mechanism: each spot's expression → a *sentence*
    of its top-N expressed gene symbols → OmiCLIP's CLIP/CoCa **text tower** (loaded
    with `open_clip`), the same way OmiCLIP/Loki embed ST without images. Needs
    `pip install open_clip_torch` and the OmiCLIP checkpoint (`--teacher-weights`).
  - **`path2space` / `gene_embedding`** — expression projected through a pretrained
    **gene-embedding matrix** `cell = X_norm @ W` (scGPT / Geneformer / Gene2vec, or a
    Path2Space-derived gene-program matrix); `--gene-embedding matrix.npz`.

  When no FM asset is supplied a **data-derived proxy** stands in (spatial-domain
  embedding + clustering) so training always runs — clearly logged as a stand-in. Add
  your own teacher with `teacher.register_teacher(name, builder)`.

  **Gene symbols (OmiCLIP).** OmiCLIP's text tower keys on gene *symbols*. If your
  panel's `var_names` are Ensembl IDs, symbols are resolved automatically in this
  order: (1) a symbol column already in `adata.var` (`gene_symbol`/`symbol`/
  `feature_name`/… — used automatically); (2) an explicit id→symbol map
  (`--teacher-symbol-map file.npz|tsv`, Ensembl version suffixes stripped); (3) the
  panel names as-is, with a warning. Symbols are used only for the gene-sentence, not
  for the expression columns.
- **Self-supervised layout reconstruction** — querying real slices reconstructs their
  occupancy (BCE on real spots vs. empty locations) and cell-type field (CE).

## Stage 2 — Expression Generator

`ExpressionField` takes `(x, y, z)`, the **Stage-1 layout code**, the context, and the
sampled flanking expression rasters, and outputs the gene-expression profile. Trained
by **expression reconstruction** (MSE) on real spots.

## Losses (all implemented, `losses.py` + `LossConfig`)

- **Layout reconstruction / distillation** — occupancy BCE + type CE + teacher feature
  distillation (cosine) + teacher pseudo-layout distillation (CE).
- **Expression reconstruction** — MSE on real spots.
- **Cross-z consistency** — finite-difference smoothness of the layout and expression
  fields across `z` (query at `z` and `z ± dz`), for coherent interpolation.
- **Biology-informed constraints** —
  - *interface preservation*: the predicted type neighbourhood-enrichment matches the
    flanking slices' interpolated `P(neighbour=j | centre=i)` (domain interfaces),
  - *within-domain gradient smoothness*: autograd penalty on `∂expr/∂xy` (the native
    microenvironment varies smoothly inside a spatial domain),
  - *spatial-domain coherence*: nearby query points share a soft type (CRF-style).

## Inference (continuous z, hybrid)

`generate_virtual_slice(z)`: encode the flanking slices → sample the **occupancy field**
on a grid (**z-marginalized** over a small z-window for hybrid inference between/beyond
slices) → draw `n_target` positions → read the **type field** → evaluate the
**expression field** conditioned on the layout code. Two decode modes:

| `--expr-decode` | expression |
|---|---|
| `residual` (default) | layout-conditioned: blend the field output with a *real same-type* cell's profile (robust; realistic gene–gene structure) |
| `field` | pure Stage-2 output (fully generative; needs the pretraining regime to be non-degenerate) |

## Status (honest)

Fully implemented and **validated end-to-end** through the real `benchmark-pbya-v2`
evaluators on synthetic data: the two fields train, the FM-teacher distillation runs
(proxy stand-in), continuous z-querying works (between and beyond slices), and the
default `residual` decode produces valid, non-degenerate metrics. As with any neural
field on a few hundred cells per holdout, the **pure `field` decode collapses toward the
mean** without large-scale pretraining — the model's intended regime is *pretraining
across many 3-D datasets* (and a real OmiCLIP/Path2Space teacher), which the
architecture is built for but which needs the full data + GPU to demonstrate and is not
validated here. No benchmark numbers are fabricated.

## Running it

Registered as `spatialcpav11_gen` (needs PyTorch, in `bench_spatialcpa`):

```bash
python -m benchmark.run_benchmark --method spatialcpav11_gen --dataset imc_breast_cancer
# real OmiCLIP teacher (needs: pip install open_clip_torch):
#   ... spatialcpav11_gen ... --teacher omiclip --teacher-weights /path/omiclip.pt
# real gene-embedding (scGPT/Geneformer/Path2Space program) teacher:
#   ... spatialcpav11_gen ... --teacher path2space --gene-embedding /path/genes.npz
# pure generative expression field: --expr-decode field
```

### Package layout

| module | role |
|---|---|
| `config.py` | all architecture / loss / training / inference hyperparameters |
| `nets.py` | Fourier features, context encoder (DeepSets), LayoutField, ExpressionField |
| `teacher.py` | FM teacher (OmiCLIP/Path2Space hook + data-derived stand-in) |
| `losses.py` | reconstruction, distillation, cross-z consistency, biology constraints |
| `trainer.py` | rasterization, leave-one-slice-out training loop, continuous inference |
| `model.py` | `SpatialCPAv11` — orchestration, normalization, fallback |
| `data.py` | `Slice` / `SliceStack` containers |

Leakage-safe: training, teacher, rasters and normalization all use the training slices
only; only the scalar target z queries the field. Falls back to a nearest-slice layout
if PyTorch is unavailable.
