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

## Pipeline stages (one module per stage)

| stage | what | module |
|---|---|---|
| **1** Pseudo-image channels | per-cell kNN soft cell-type composition + local density `m` | `latents.morphology_features` |
| **2** Variational autoencoder | encoder `[x̃ ⊕ m] → (μ, logσ²)`; **NB decoder** `h → ρ=softmax`, `μ = library·ρ`, per-gene dispersion `θ`; aux morphology + type heads; Gaussian likelihood for non-count inputs | `nets.JointVAE` |
| **3.1** 3D attention context | Fourier `(x,y,z)` query cross-attends over local flanking cells + per-slice global tokens → `C(z)` | `nets.ContextAttention` |
| **3.2** Conditional flow matching | velocity field trained with the CFM loss on the OT straight-line path | `nets.VectorField`, `trainer._phase_b` |
| **3.3** Gap-aware + z-marginalized | whole context slices randomly dropped; z jittered during conditioning | `trainer._context`, `_phase_b` |
| **4** Generation | coherent-patch sheet layout; one flow trajectory per cell → `h*`; **decode** `h*` and draw a **posterior-predictive sample** (NB `~NB(μ,θ)` for counts, Gaussian `~N(μ,σ)` for intensity); library from the anchor real cell; optional cell-type composition matching | `trainer.generate_slice` |
| **5** Retrieval (ablation) | `--retrieval`: use `h*` as a key to pick 1-of-K spatially-nearest real cells (v14 behavior) | `trainer.generate_slice` |

## Why sample the decoder instead of emitting its mean

The decoder learns not just the mean but the **observation-noise scale** — per-gene
dispersion `θ` (NB) or per-gene standard deviation `σ` (Gaussian). A real cell is a **draw**
from that distribution, not its mean. Emitting the mean directly under-states per-gene
variance and inflates gene-gene correlation (everything is driven by one low-dim latent) —
the low-rank-decode pathology that made v14 fall back to copying real cells. v17 instead
emits a **posterior-predictive draw** (`emit="sample"`, the default): Gamma-Poisson `~NB(μ,θ)`
on the count path, `μ + σ·ε` on the Gaussian path. This restores realistic marginals and
de-correlates genes, while the flow-conditioned mean still provides smooth spatial gradients —
the realistic-variance behavior v14 gets for free from real cells, now produced generatively.
For the same reason, generation uses a **single flow trajectory per cell** (no ensemble
averaging), so cells span the conditional distribution rather than collapsing to its mean.
`--emit mean` recovers the point-estimate behavior as an ablation (both likelihoods).

## Likelihood selection

- **`auto`** (default): negative-binomial for raw-count inputs, Gaussian (learned per-gene
  `σ`, NLL) for fluorescence / already-normalized inputs. The benchmark wrapper passes **raw
  counts** for count datasets so the NB decoder sees true counts and a real library size.
- Force with `--likelihood {nb,gaussian}`. Both paths sample by default (`--emit sample`).

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

## Diagnostic: is there a manifold gap?

Decode-generation only works if the flow's generated latents `h*` land where the decoder was
trained (the encoder's aggregate posterior). `reconstruction_gap(model)` measures this on the
interior training slices, comparing **encode→decode** (`decode(encode(x))`) against
**flow→decode** at the same positions, in latent space (a kNN off-manifold ratio) and in
expression space (how well each reproduces real per-gene mean/variance). The verdict is driven
by the **fidelity drop** recon→gen — not the raw ratio, which is `>1` partly by construction —
so a robust decoder that absorbs the flow's spread reads as *no gap*.

```python
from spatialcpav17 import reconstruction_gap
reconstruction_gap(gen)          # prints per-slice + aggregate metrics and a verdict
```

Or on a dataset via the standalone runner (diagnostic only — writes no predictions):

```
python src/benchmark/methods/diagnose_spatialcpav17.py --input <train_only.h5ad>
```

If a gap is detected, a Phase-C decoder fine-tune (training the decoder on flow-produced
latents) is the targeted fix; if not, decode-generation is already on-manifold and Phase C is
unnecessary.
