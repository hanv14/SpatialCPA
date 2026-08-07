# SpatialCPA-v17 — a fully-generative VAE + NB-decoder latent flow-matching atlas

v17 keeps v14's **conditional flow matching in a joint molecular–morphological latent
space** (3D positional-attention context, gap-aware training, z-marginalization,
noise-ensemble uncertainty) but replaces v14's **linear PCA expression code** with a
**learned variational autoencoder** carrying a **negative-binomial (count-aware)
decoder** (scVI-style). Because the latent is KL-regularized toward a standard normal,
the flow-generated samples lie on the decoder's manifold and can be **decoded directly
into gene expression** — no retrieval from real cells required.

This is the single design difference from v14. v14 uses the flow latent only as a
*retrieval key* to copy a real neighboring cell's profile; v17 makes the pipeline
**fully generative**: `flow → h* → decode(h*) → expression`. Retrieval is retained as an
ablation (`--retrieval`).

```
per-cell counts x ──log-norm──► x̃ ┐
                                   ├─ VAE encoder ─► (μ, logσ²) ─reparam─► joint latent h
per-cell pseudo-image channels m  ─┘                                              │  KL(q(h|x)‖N(0,I))
(soft cell-type maps + local density)                                            │
                                                                                 ▼
  real slices' {h=μ, pos} + Fourier(x,y,z) ──► 3D positional attention ──► context C(z*)
                                                                                 │
  noise h0 ~ N(0,I) ──► ∫ v_t(h_t | t, C(z*), z*) dt  (conditional flow matching)│
                                                                                 ▼
                       generated latent h*(z*) ──► NB decoder ──► ρ=softmax, μ=ℓ·ρ ──► counts
                                                                                 │
                       library size ℓ taken from the cell's coherent-patch anchor ┘
```

## Two-phase training

| phase | what | trains | frozen |
|---|---|---|---|
| **A** | Joint VAE: NB (or Gaussian) reconstruction + KL warmup + auxiliary morphology-reconstruction and cell-type heads. Posterior means `h=μ` are then cached as flow targets. | VAE | — |
| **B** | Conditional flow matching in the **frozen** latent: velocity field `v_t(h_t\|t,C,z)` on the OT straight-line path, gap-aware + z-marginalized. Identical to v14. | attention + flow field | VAE |

Both phases train to careful convergence: Phase A runs `pretrain_epochs=200`, Phase B
`epochs=260`, KL is warmed up over `40` epochs, and the learning rate follows a **cosine
decay** to a small floor (`lr_min_ratio=0.05`) so the model settles finely rather than
oscillating at a fixed step size. The architecture is unchanged — this is a deeper training
schedule, not a bigger model.

## Pipeline stages (one module per stage)

| stage | what | module |
|---|---|---|
| **1** Pseudo-image channels | per-cell kNN soft cell-type composition + local density `m` | `latents.morphology_features` |
| **2** Variational autoencoder | encoder `[x̃ ⊕ m] → (μ, logσ²)`; **NB decoder** `h → ρ=softmax`, `μ = library·ρ`, per-gene dispersion `θ`; aux morphology + type heads; Gaussian likelihood for non-count inputs | `nets.JointVAE` |
| **3.1** 3D attention context | Fourier `(x,y,z)` query cross-attends over local flanking cells + per-slice global tokens → `C(z)` | `nets.ContextAttention` |
| **3.2** Conditional flow matching | velocity field trained with the CFM loss on the OT straight-line path | `nets.VectorField`, `trainer._phase_b` |
| **3.3** Gap-aware + z-marginalized | whole context slices randomly dropped; z jittered during conditioning | `trainer._context`, `_phase_b` |
| **4** Generation | coherent-patch sheet layout; flow ODE (noise-ensemble) → `h*`; **decode** `h*` to expression; library from the anchor real cell; optional cell-type composition matching | `trainer.generate_slice` |
| **5** Retrieval (ablation) | `--retrieval`: use `h*` as a key to pick 1-of-K spatially-nearest real cells (v14 behavior) | `trainer.generate_slice` |

## Likelihood selection

- **`auto`** (default): negative-binomial for raw-count inputs, Gaussian (standardized-MSE)
  for fluorescence / already-normalized inputs. The benchmark wrapper passes **raw counts**
  for count datasets so the NB decoder sees true counts and a real library size.
- Force with `--likelihood {nb,gaussian}`.

## Usage

```python
from spatialcpav17 import SpatialCPAv17, SpatialCPAv17Config, Slice, SliceStack
gen = SpatialCPAv17(stack, gene_names=genes, cell_type_names=types, is_counts=True)
vs = gen.generate_virtual_slice(z=target_z)   # vs.expression are decoded expected counts
```

Benchmark (benchmark-pbya-v2), decode-only default:

```
python src/benchmark/methods/run_spatialcpav17.py \
    --input <train_only.h5ad> --target-section <sec> --target-z <z> --output pred.h5
```

Retrieval ablation: add `--retrieval`. All defaults in `config.py` are the intended
production settings; v14 is left completely untouched.
