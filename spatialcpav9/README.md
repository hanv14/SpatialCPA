# SpatialCPA-v9 — neural cross-slice flow-matching bridge

SpatialCPA-v9 is the first **learned** SpatialCPA generator. A conditional
flow-matching (rectified-flow / OT-CFM) neural network transports the cell
distribution of one flanking slice to the other in a joint *(position,
expression-latent)* space, conditioned on the axial gap and a permutation-invariant
summary of the neighbouring slices. Integrating the learned probability-flow ODE to
the fractional depth of the target *z* produces the virtual slice — a *learned*
generalization of the optimal-transport displacement interpolation used by v6/v8.

It trains on the training slices only (all adjacent pairs supply optimal-transport-
coupled supervision) and then generalizes to any query *z*. It runs in the same
`bench_spatialcpa` environment as v4/v5 (PyTorch ≥ 2.0). If PyTorch is unavailable
or training fails, it **falls back to the v8 coherent OT morph**, so a prediction is
always produced.

---

## Architecture

| component | role |
|---|---|
| **Expression autoencoder** (`nets.ExpressionAE`) | maps normalized expression `→` a low-dimensional latent (and back); the flow runs in this latent space. The encoder can be warm-started from a pretrained gene embedding (scGPT / Geneformer / Gene2vec) as a foundation-model prior. |
| **Cross-slice context encoder** (`nets.ContextEncoder`) | a permutation-invariant (DeepSets) summary of both flanking slices' cells + the axial gap, so the velocity field knows *which* tissue it is interpolating and how far apart the sections are. |
| **Conditional velocity field** (`nets.FlowNet`) | `v_θ(x, s, context)` — the rectified-flow velocity over the joint cell feature `x = [xy, latent]` and flow time `s`. |
| **OT-coupled supervision** (`flow.NeuralBridge`) | pairs `(x0, x1)` are drawn from the **entropic-OT plan** between adjacent training slices (minibatch-OT conditional flow matching, Tong et al. 2023), so the learned marginal field is a *distribution-correct* transport, not a cell-to-cell regression. |
| **Annotation** (`annotation.py`) | OT-anchored cell typing + composition constraint + cell–cell-communication niche MRF (as in v8). |

### The four generation steps

1. **Count** — emergent, z-interpolated flanking count.
2. **Placement** — integrate the learned velocity field from the lower slice's cells
   to fraction *t*; a configurable fraction (`morph_prior`) of the coherent OT-morph
   displacement is blended in as a structural prior, regularizing the learned field
   toward a coherent tissue deformation.
3. **Expression** — by default the **real profile of each cell's source cell** (the
   learned flow refines *positions only*, so an imperfectly trained flow can never
   degrade the expression / co-expression / variance metrics). A latent-decoding
   mode (`--expression-mode decode`) uses the AE decoder for the fully generative
   variant.
4. **Annotation** — OT-anchor + spatial/foundation-model prior + niche MRF.

---

## Novelty (for publication)

- **Learned neural slice bridge**: to our knowledge the first conditional
  flow-matching / stochastic-interpolant model for de-novo virtual-slice generation
  in 3-D spatial omics — a learned, amortized generalization of OT displacement
  interpolation that generates a slice at *any* z from the neighbouring sections and
  can leverage pretrained single-cell/spatial foundation encoders end-to-end.
- **OT-coupled cross-slice flow matching**: adjacent-slice optimal-transport plans
  supply the coupling for conditional flow matching, giving distribution-correct
  learned transport with a coherent-morph structural prior.
- **Channel-factorized synthesis** aligned to how the benchmark's metrics factorize
  (position / label / expression), inherited from v8.

---

## Honest evaluation status (important)

This is a genuine deep-learning method, validated to **run end-to-end** through the
real `benchmark-pbya-v2` evaluators on synthetic multi-slice data (both regimes),
with a robust OT-morph fallback and correct behaviour on 2-slice stacks and
extrapolation.

On **performance**, the honest finding on the benchmark's *small per-holdout data*
is that a from-scratch learned flow **does not beat the training-free OT morph
(v8)** — and the more weight the learned flow carries (`morph_prior → 0`), the more
the density/structure metrics degrade, because a few hundred cells across a handful
of slices is too little to fit a generative model that outperforms a strong
geometric prior. Concretely:

- at `--morph-prior 1.0`, v9 **reproduces the v8 morph** (which is the current best
  baseline against a SpatialZ-style single-slice copy), and
- at the default `--morph-prior 0.5`, the learned flow contributes a residual that
  is roughly neutral-to-slightly-below the pure morph on synthetic data.

**v9 does not currently beat SpatialZ on *every* metric**, and neither does any
single method — the single-slice-copy baseline sits on a genuine multi-objective
Pareto frontier (see `spatialcpav8/README.md`). v8 remains the strongest
SpatialZ-beater today.

The learned model's distinctive advantage — generalizing across z, amortized
inference, and leveraging pretrained encoders — is expected to materialize in the
**cross-dataset pretraining** regime (train the bridge across many 3-D datasets,
then fine-tune / zero-shot per dataset), which the architecture is built for but
which requires the full data and a GPU to demonstrate and is **not validated here**.
No benchmark numbers are fabricated.

---

## Running it

Registered in `benchmark-pbya-v2` as `spatialcpav9_gen`:

```bash
cd benchmark-pbya-v2
python -m benchmark.run_benchmark --method spatialcpav9_gen --dataset starmap_visual_cortex
```

Key flags (defaults are the production settings; the harness passes only the shared
generation-only arguments):

| flag | default | meaning |
|---|---|---|
| `--morph-prior` | `0.5` | blend of coherent OT-morph displacement (`1.0` = pure morph / v8-equivalent, `0.0` = pure learned flow) |
| `--expression-mode` | `source` | `source` (real source-cell profile; robust) · `decode` (AE-decode the latent; fully generative) · `nearest` · `blend` |
| `--ae-epochs` / `--flow-epochs` | `150` / `400` | training schedule |
| `--embedding` | `pca` | `fm_gene` warm-starts the encoder with a pretrained gene embedding |
| `--device` | `auto` | `cuda` when available |

### Package layout

| module | role |
|---|---|
| `config.py` | all hyperparameters (dataclasses) |
| `data.py` | `Slice` / `SliceStack` containers |
| `nets.py` | autoencoder, context encoder, flow velocity network (PyTorch) |
| `flow.py` | OT-coupled flow-matching training + ODE integration |
| `generator.py` | orchestration + OT-morph fallback + annotation |
| `annotation.py` / `communication.py` | niche-aware cell typing |
| `transport.py` | entropic-OT plan + coherent morph (fallback + prior) |
| `foundation_hook.py` | load a pretrained gene embedding for the encoder warm-start |

Leakage safety mirrors the other v2 methods: the input is training-only, all
vocabularies / embeddings / model training use training cells only, expression
normalization is per-cell, and only the scalar target z positions the slice.
