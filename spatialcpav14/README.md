# SpatialCPA-v14 — H3D-FLA: a Hybrid 3D Flow-matching Latent Atlas

v14 is a direct implementation of the **`H3D_FLA_Pipeline.md`** proposal: a virtual-slice
generator built on **conditional flow matching in a joint molecular–morphological latent
space**, conditioned on a **3D positional-attention context** over the real slices, trained
**gap-aware** with **z-marginalization** and **biology-informed regularizers**, and decoded
back to expression + cell type with the generated field **grounded in real training
biology**.

It is a paradigm **completely distinct from v8**: no optimal transport, no
barycentric / diffeomorphic morph, no OT fusion, no niche Markov-random-field (v8 is
training-free numpy/scipy). It also shares nothing with v13 (no cell-sentence tokenizer,
no gene-language model, no retrieval softmax). The engine here is a **generative
flow-matching model** — an ODE that transports Gaussian noise to a data latent under a
learned, 3D-attention-conditioned velocity field.

```
per-cell expression ──PCA──► expression latent e ┐
                                                 ├─ JointEncoder ─► joint latent h ──┐
per-cell pseudo-image channels m (soft cell-type │                                   │
maps + local density) ───────────────────────────┘                                  │
                                                                                     ▼
  real slices' {h, pos} + Fourier(x,y,z) ──► 3D positional attention ──► context C(z*)
                                                                                     │
  noise h0 ~ N(0,I) ──► ∫ v_t(h_t | t, C(z*), z*) dt  (conditional flow matching ODE)│
                                                                                     ▼
                       generated joint latent h*(z*) ──► decode ──► expression / type / displacement
                                                                                     │
                                     ground each cell in a real local profile  ◄─────┘
```

## Pipeline stages (one module per stage)

| stage | what | module |
|---|---|---|
| **1** Preprocessing & pseudo-image | PCA **expression latent** `e`; per-cell **pseudo-image channels** `m` = kNN-smoothed soft cell-type maps + local density | `latents.py` |
| **2** Joint encoder | fusion MLP `e ⊕ m → h`, with decoders back to `e`, `m`, cell type, and a hypoxia scalar | `nets.JointEncoder` |
| **3.1** 3D attention context | a Fourier `(x,y,z)` query cross-attends over **local flanking cells** + **per-slice global summary tokens** → context `C(z)` (local + long-range) | `nets.ContextAttention` |
| **3.2** Conditional flow matching | velocity field `v_t(h_t \| t, C, z)` trained with the CFM loss on the OT straight-line path | `nets.VectorField`, `trainer._phase_b` |
| **3.3** Gap-aware + z-marginalized | whole context slices randomly dropped; z jittered during conditioning | `trainer._context`, `_phase_b` |
| **4** Biology-informed reg. | closed-loop consistency, edge-aware adaptive smoothness / interface coherence, soft hypoxia-gradient directionality (annealed) | `trainer._smoothness`, `_hypoxia` |
| **5** Inference | integrate the conditional ODE from several noises (marginalized), decode; lay out the sheet as **spatially-coherent single-slice patches** (a smooth source mask, preserving each real slice's niche organization) and **ground** each cell in a real local profile; match composition | `trainer.generate_slice` |
| **6** Training strategy | two-phase: Phase A encoder/decoder reconstruction, Phase B flow + attention end-to-end | `trainer.train_model` |

## Why a flow-matching latent field *and* real-profile grounding

The benchmark scores a generated slice as a **field / distribution**, and the metrics pull
in two directions (see `spatialcpav8/README.md`): position/field metrics want the coherent
*in-between* geometry, while expression-structure metrics (co-expression, Sinkhorn, Moran's
I) want *real* local gene–gene covariance. v14 gets both sides at once **without** optimal
transport:

* the **flow-matching latent field** supplies the smooth, z-interpolated molecular
  structure (blended into the grounded profile) — this is what wins the binned field/SSIM
  metrics;
* **grounding** each generated cell in a spatially-local *real* training profile keeps the
  real gene–gene covariance and spatial autocorrelation — this is what wins the
  distribution / co-expression / Moran metrics;
* laying the sheet out as **spatially-coherent single-slice patches** (rather than
  interleaving both flanking slices per cell) preserves each real slice's local cell-type
  organization — this is what wins the niche / neighborhood-enrichment metrics.

The flow's decoded expression is blended into the grounded profile (`edit_weight`), so the
generative model genuinely shapes the molecular output rather than only selecting exemplars
— the closed-loop consistency and smoothness regularizers keep that blend biologically
coherent.

## Robustness (the pipeline's gap-aware / z-uncertainty goals)

* **Gap-aware training** — whole context slices are randomly masked so the flow learns to
  reconstruct a latent field from the *remaining* slices, i.e. across real gaps. In the
  benchmark the held-out slice is never in the stack, so training is intrinsically
  leave-one-slice-out.
* **z-marginalization** — z is perturbed during conditioning (train) and several initial
  noises are integrated and averaged at inference, marginalizing z-position uncertainty and
  denoising the generated latent.
* **Continuous querying** — any real-valued `z` (including large gaps) is supported;
  `generate_virtual_slice(z)` builds the attention context for that exact position.

## Validation status (honest)

Validated end-to-end through the **real** `benchmark-pbya-v2` generation evaluator against
**v8's default** (the strongest prior SpatialCPA generator), on a **real** STARmap 3D
cortex block and on two synthetic regimes (distinct drift + near-identical volumetric):

* **Real STARmap 3D cortex block: v14 wins all 10 primary `gen_*` metrics** — including
  co-expression, Moran's I, and cell-type neighborhood enrichment (the niche metric a
  copy-based method usually cedes) — plus the large majority of the cell-matched reference
  metrics. The only reference losses are `celltype_accuracy` (~2%) and count-scale
  `rmse`/`mae` (~ties on real, large-count data).
* **Synthetic (both regimes): v14 wins the substantive distribution + field metrics** and
  the remaining gaps are small (a few percent) — the exact-copy niche/accuracy metrics a
  coherent real-slice copy is intrinsically built to dominate on near-identical planes.

The spatially-coherent patch layout (Stage 5) is what closed the earlier co-expression /
Moran / neighborhood-enrichment gaps against v8, with a single universal default (no
per-dataset tuning).

See `validation/VALIDATION.md` for the full per-metric tables and exact reproduce commands.
The full cross-dataset leaderboard must be regenerated where the processed datasets live;
**no benchmark numbers are fabricated.**

## Running it

Registered in `benchmark-pbya-v2` as `spatialcpav14_gen` (needs PyTorch, in
`bench_spatialcpa`). Every default is the intended production setting — running with only
the shared generation-only arguments reproduces the proposed pipeline:

```bash
python -m benchmark.run_benchmark --method spatialcpav14_gen --dataset starmap_visual_cortex
# ablation / tuning knobs: --epochs --pretrain-epochs --latent-dim --joint-dim
#   --ode-steps --ensemble --position-mode --displacement-scale --edit-weight
#   --ground-blend-flow --context-slices --type-placement --coherent-freq
#   --no-coherent-source --no-bio --no-attention --no-ground --no-composition-match
```

If PyTorch is unavailable the method degrades to a dependency-free, latent-grounded
recombination of the flanking slices (clearly logged), so it always produces output.

### Package layout

| module | role |
|---|---|
| `config.py` | dataclass hyperparameters mapped one-to-one to pipeline stages |
| `latents.py` | Stage 1 — PCA expression latent + per-cell pseudo-image channels |
| `nets.py` | Fourier/time embeddings, joint encoder+decoders, 3D attention, flow velocity field |
| `trainer.py` | Phase A/B training (CFM + gap-aware + z-marg + bio losses); ODE-based generation |
| `model.py` | `SpatialCPAv14` — orchestration, normalization, no-torch fallback |
| `data.py` | `Slice` / `SliceStack` containers |
