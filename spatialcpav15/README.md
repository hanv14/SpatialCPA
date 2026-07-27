# SpatialCPA-v15 — dense 3D atlas reconstruction from sparse 2D sections

SpatialCPA-v15 is a complete, from-scratch implementation of
[`spatialcpa_v15_idea.md`](../spatialcpa_v15_idea.md). It synthesizes a **virtual
tissue section at an arbitrary depth `z`** from the neighbouring *training*
sections and a scalar target `z` — it never sees the held-out section's `(x, y)`
or molecular content, so it is a true de-novo generator and is scored by the
leakage-hardened `benchmark-pbya-v2` harness.

Run it:

```bash
python run_spatialcpa_v15.py          # full pipeline + head-to-head vs v8
```

---

## What it is

Four phases, each instrumented and gated independently, exactly as the proposal
lays out.

### Phase 1 — preprocessing and common coordinates (`phase1_types.py`, `phase1_frame.py`)

The deliverable is a registered stack in which every cell carries a normalized
anatomical position, a **hierarchical** label, a **continuous type embedding**,
raw counts, and its 3D neighbours with each edge's `Δz`.

The centrepiece is the type embedding `e_c`, because it is the widened bottleneck
that everything downstream depends on. Discrete IDs carry no similarity structure
and are what make a conditional generator collapse to type-means. The proposal's
recipe is followed literally:

| step | what happens here |
|---|---|
| **1.2.0** | The label vocabulary is *classified first* — numbered clusters, text names, or mixed. Mixed is harmonized to clusters rather than embedded as a mix of conventions. |
| **1.2.A/B** | Markers per type are identified as **signed effect sizes** (Cohen's *d* vs. the rest), keeping direction and magnitude, not a rank list. Text labels additionally contribute their own name tokens (`"{name}; markers: {…}"`). |
| **1.2.1** | The description is encoded **order-invariantly**: the marker component is pooled over the marker *set* with signed effect weights, so it cannot depend on ranking noise. |
| **1.2.2** | The **backbone is data-driven** — batch-corrected pseudobulk → PCA. The encoder vector is kept as an **external prior**. Their agreement is reported (Mantel *r* + kNN overlap), and they are blended **per type** with a shrinkage weight `w_c = n_c/(n_c+κ)`: well-sampled types follow the data, rare types borrow the prior. Before blending, the prior is rotated into the backbone's frame by weighted Procrustes — two independently fitted embeddings otherwise live in unrelated bases. |
| **1.3** | Types are shared across the whole stack by construction; a Ward hierarchy over `e_c` gives class → subclass → cluster. |
| **1.4/1.5** | Landmark drift stabilization (section centroid + tissue outline, both derivable from points alone) applied *only* when the residual drift exceeds tolerance, then **normalized anatomical coordinates** so nothing memorizes absolute microns. Label-field registration is deliberately **not** used — it is the circular option the proposal warns about. |
| **1.6** | 3D kNN graph, every edge storing its own `Δz`. |

The proposal's caveat that batch effects poison *both* embeddings is handled
upstream: pseudobulk is section-centred before pooling.

### Phase 2 — structure completion (`phase2_field.py`, `phase2_structure.py`, `phase2_layout.py`)

**2.1** The field is a **multi-channel density**, never an RGB label image:

```
λ_c(x, y, z) = Σ_{i: t_i = c} K_h(p − p_i)
```

channel 0 is total density, then one channel per type group, then
**density-weighted mean type-embedding** channels. Channels are commensurate and
non-negative by construction, so there are no invalid intermediate states. When
there are more types than the channel budget, the group channels drop to the
Phase 1.3 hierarchy and the embedding channels carry within-group identity. The
in-plane grid is coarsened to a multiple of the median cell spacing so voxel
aspect ratios stay sane, and a section is rasterized as a **slab** — the
`z`-integral over its thickness, which is the forward operator the interpolator
learns to invert.

**2.2** Channel-space compression kicks in automatically above the channel
budget. A full 3D VAE is unnecessary at the volume sizes this method targets;
the rule is size-based, not a hand switch.

**2.3** The interpolator is trained as **query-based video-frame interpolation**:
tuples `(I_{a−1}, I_a, I_b, I_{b+1}, t, Δz) → I_t` sampled from the training
stack with a real intermediate as the target. **Two slices per side** so the
derivative is available; **flow-free and directly generative** (no warping — flow
assumes conserved translating content and cannot represent branching, merging or
termination); conditioned on **both** normalized `t` and absolute `Δz`; augmented
by random gap widths, random `t`, random crops and **z-reversal** (an exact
symmetry, hence free data). In-plane mirroring is off by default.

The denoiser predicts the *residual from linear bracket interpolation*, so the
model's zero-effort mean is exactly the baseline it must beat, and its capacity is
spent on the non-linear part of the structure.

**2.4** Sampling is a **deterministic DDIM** with the **same initial noise reused
for every query in a gap**, so `t ↦ I_t` is a continuous function and
independently issued queries compose into one coherent volume instead of
flickering. One seed = one hypothesis; `N` seeds = `N` coherent volumes; the
voxelwise spread across seeds is the structural uncertainty map. Direct query is
preferred over recursive subdivision.

**2.5** The stop condition is **machine-checked, not documented**: on internally
held-out real slices the completion must beat linear interpolation of the field,
measured on the **seed-ensemble mean** (a single draw carries the full conditional
variance, so scoring one draw would measure sampling, not accuracy). A failing
model is not used — the pipeline degrades to the linear field and says so. The
orthogonal-reslice test is reported as a number (`reslice_striping`: z-direction
total variation over in-plane total variation; ≈1 means tissue-like from every
axis) rather than as a look.

**Layout** is a **marked spatial point process** (Phase 4.3), not a raster →
peak-detect round trip. Positions are drawn from the completed intensity with
sub-voxel jitter after a bilinear refinement; allocation is **stratified**, so
the process keeps the intensity without the multinomial counting noise that at
~1 cell per bin would exceed the density signal itself; a soft **hardcore**
relaxation restores the regularity of a real point pattern. Marks are drawn from
the local type posterior implied by the group-density and embedding channels, and
the composition is nudged onto the field-implied composition by re-labelling the
least-confident cells.

### Phase 3 — expression (`phase3_expression.py`)

**3.1** An expression VAE used as a **likelihood, not a generator**: an NB
decoder (ZINB and Gaussian available; the choice is automatic from whether the
input is count-like), a **library-free profile** (softmax over genes) with library
size as an **explicit scalar** input, a deliberately **low KL weight**
(near-autoencoder — diffusion supplies the prior), trained on **all cells pooled**
with `z` ignored. Posterior collapse is checked and reported.

**3.2** A **conditional latent diffusion in the frozen latent**, conditioned on

* the **continuous hierarchical type embedding** `[e_cluster, e_subclass, e_class]`
  from Phase 1.2 — never a bare label,
* normalized position (Fourier features),
* **neighbourhood context**: cross-attention over the encoded latents of nearby
  cells in the **real** bracketing slices,
* `Δz` to the nearest real slice.

Training uses **slice dropout** of consecutive runs and randomized `Δz`, so the
model learns to widen its distribution as context recedes. The **whole gap block
is denoised jointly**, with self-attention across cells and across `z` — never
independently per slice.

**3.3** The gate runs on an internally held-out **training** slice with its
**ground-truth layout**, against the three baselines the proposal names —
nearest-slice copy, linear interpolation between brackets, and **cell-type-mean
assignment** — judged on co-expression agreement, per-gene Moran's agreement and
the **within-type variance ratio** (penalized in both directions). Failing the
type-mean bar means the model is a label propagator; the result is reported either
way and selects the decoder readout (NB mean vs. NB draw).

### Phase 4 — inference (`generator.py`)

Register → `N` deterministic structure completions → marked-point-process layout →
context from the real bracketing slices → joint-block latent diffusion → decode to
counts → **provenance**. Every synthesized cell carries its **distance to the
nearest real slice** and its **cross-seed structural spread**, and cells whose
depth is beyond the Nyquist limit are flagged **unresolved**. Observed and
generated sections are visually indistinguishable — that is the point of the
method and its main hazard, so the flags travel with the data
(`VirtualSlice.dist_to_real_slice`, `.structural_spread`, `.unresolved`).

---

## How it differs from SpatialCPA-v8

v8 is training-free and transport-based. v15 shares **none** of its machinery:

| | v8 | v15 |
|---|---|---|
| core object | an entropic-OT plan between two flanking slices | a completed density field `λ_c(x, y, z)` |
| placement | McCann/barycentric bridge, smoothed-OT displacement morph, velocity-field advection of a single anchor slice | cells **sampled** from the field as a marked point process |
| expression | real profiles copied / OT-fused across the two flanking slices | generated by conditional latent diffusion, decoded through an NB VAE |
| typing | niche MRF refinement of copied labels | marks drawn from the field's own type posterior |
| type representation | PCA / co-expression embedding used as an OT cost | marker-effect-size embedding, prior-blended by sample size, hierarchical, used as *conditioning* |
| learning | none (closed-form transport) | three trained models (structure diffusion, VAE, latent diffusion) |
| uncertainty | none | per-cell structural + expression spread, sub-Nyquist flags |

The estimands differ, not just the implementations: v8 answers *"where does this
real slice's tissue move to?"*, v15 answers *"what is the tissue at this depth?"*.

---

## Package layout

| module | role |
|---|---|
| `config.py` | every knob, grouped by proposal phase; defaults **are** the production settings |
| `data.py` | `Slice` / `SliceStack` / `VirtualSlice`, bracket selection (two per side) |
| `phase1_types.py` | QC, label-kind branch, signed markers, prior encoder, data backbone, shrinkage blend, hierarchy |
| `phase1_frame.py` | landmark stabilization, normalized anatomical coordinates, 3D kNN graph with edge `Δz` |
| `phase2_field.py` | multi-channel rasterization (slab forward operator), automatic channel compression |
| `phase2_structure.py` | query-based VFI diffusion, deterministic multi-query, the Phase 2.5 gate, reslice diagnostic |
| `phase2_layout.py` | marked spatial point process (stratified intensity sampling, hardcore, marks, composition repair) |
| `phase3_expression.py` | NB VAE, conditional latent diffusion with context cross-attention, the Phase 3.3 gate |
| `nets.py` | `QueryUNet`, `ExpressionVAE`, `ContextDenoiser`, cosine-schedule DDPM/DDIM |
| `generator.py` | Phase 4 orchestration and provenance |
| `validation/` | head-to-head against v8 through the real benchmark evaluator |

---

## Running it in the benchmark

Registered in `benchmark-pbya-v2` as `spatialcpav15_gen`:

```bash
cd benchmark-pbya-v2
python -m src.benchmark.run_benchmark --method spatialcpav15_gen --dataset starmap_visual_cortex
```

The harness passes only the shared generation-only interface
(`--input/--target-section/--target-z/--output/--seed`), and every default in
`SpatialCPAv15Config` is the production setting, so that command runs the proposed
pipeline as-is. The extra wrapper flags (`--no-structure-model`,
`--no-expression-diffusion`, `--readout`, `--bin-spacings`, …) exist for ablations.

Requires `torch >= 2.0` (present in the `bench_spatialcpa` environment) plus
numpy/scipy/scikit-learn/anndata/scanpy.

---

## Deviations from the proposal

Three, all forced by what is available rather than by preference, and all
localized:

1. **Phase 1.2.1 encoder.** The proposal says "feed the description to a text /
   gene-aware encoder". No pretrained text or gene encoder is bundled and this
   environment has no model download, so the default encoder is a deterministic,
   order-invariant **hashed gene-symbol encoder** (character n-grams, signed
   hashing, effect-weighted set pooling). It is a genuine *external* prior — it is
   computed from names only and knows nothing about the specimen's expression,
   which is the property Phase 1.2.2 actually relies on — but it is weaker than a
   real gene-aware encoder. Supplying `--fm-gene-embedding <scGPT/Gene2vec/…>`
   swaps in a pretrained encoder; nothing else changes.
2. **Phase 2.2 compression.** A 3D VAE over the structural field is implemented as
   an automatic **channel-space PCA** that engages only above a channel budget.
   The proposal itself makes this stage conditional ("skip only if volumes are
   genuinely small"); at the volume sizes here it is not the bottleneck.
3. **Phase 4.2 seed count.** The proposal suggests `N ≈ 20` structure
   completions; the default is `N = 8` for tractability on CPU. It is one config
   value (`InferenceConfig.n_structure_seeds`) and raising it only costs time.

Everything else — including the Phase 2.5 stop condition and the Phase 3.3
baseline gate — is implemented as specified and runs by default.
