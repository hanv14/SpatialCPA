# Dense 3D Spatial Atlas Reconstruction from Sparse 2D Slices
## Detailed Methodology — Phases 1 to 4

**Input assumption:** cell/spot positions and labels only (no histology / no image channel).
Sections are sparse along *z* (spacing `Δz`), near-continuous in-plane.

**Define up front:** `R = Δz / target_output_spacing`. This is the single number that says how hard your instance is. `R ≲ 4` is inside demonstrated interpolation range; `R ≳ 8` is a research contribution in itself. Report it.

---

## Phase 1 — Preprocessing and Common Coordinates

**Deliverable:** a registered 3D volume where each cell carries: normalized anatomical position, a hierarchical label, a *continuous type embedding*, raw counts, and its 3D neighbors (with edge `Δz`).

### 1.1 QC and segmentation
Standard per-section QC (counts, genes, doublets, segmentation quality). Nothing 3D-specific here, but do it per section so batch structure stays visible.

### 1.2 Type representation

The goal of this step is a **continuous embedding `e_c` per label**, in which biologically similar types sit close together. That embedding is what flows into Phase 2 (structure field) and Phase 3 (expression conditioning); it is the widened bottleneck that keeps within-type gradients from collapsing to type-means. Discrete IDs will not do this — they carry no similarity structure.

#### 1.2.0 First, inspect what the input labels actually are
Branch on the label type before doing anything else:

- **Numbered / clustering IDs** (`0, 1, 2, …`, `Leiden_4`, `cluster_12`) — no semantic content. Take the **marker path** (1.2.A).
- **Text cell-type names** (`"CA1 pyramidal"`, `"Sst interneuron"`, `"astrocyte"`) — semantic content present. Take the **text + marker path** (1.2.B).
- **Mixed** across sections (some named, some numbered) — harmonize first: either map names onto clusters, or treat everything as clusters and re-annotate. Do not embed a mix of ID conventions directly.

#### 1.2.A Numbered clusters → markers
1. **Marker identification per cluster.** Differential expression vs. the rest (Wilcoxon / logistic regression / a signed statistic). Keep the top *k* markers per cluster (e.g. 20–50) with **effect size and direction**, not just a ranked name list.
2. **Represent each type by its marker set** — *"the population defined by high {GENE1, GENE2, …}"* — not by its integer ID.
3. Proceed to 1.2.1 to embed.

#### 1.2.B Text labels → text + markers
Even with names, still run marker identification, and embed **both** the text label and its markers together:
1. **Marker identification** as in 1.2.A (the name alone is coarse and dataset-agnostic; the markers ground it in your data).
2. **Compose a single description per type**, e.g. *"{cell-type name}; markers: {GENE1, GENE2, …}"*.
3. Embedding both is strictly better than either alone: the name carries prior knowledge and cross-dataset comparability, the markers carry your dataset's actual signal. Proceed to 1.2.1.

#### 1.2.1 Encode to a continuous vector
Feed the description (marker set, or name + markers) to a text / gene-aware encoder → `e_c ∈ ℝ^d`, `d ≈ 8–32`.
- Encode the marker component **order-invariantly** — it is a *set*. Sort deterministically, or pool per-gene embeddings (mean/attention); do not feed a raw ordered string whose embedding depends on ranking noise.
- This `e_c` is reused directly in Phase 2.1 (structure) and Phase 3.2 (expression).

#### 1.2.2 Data-driven backbone + external prior (the anchor, done right)
Do **not** ship the encoder embedding as the backbone — its similarity structure partly reflects gene-name token statistics and literature co-mention, not your tissue. Instead:

1. **Backbone — data-driven embedding.** Per type, pseudobulk → log-normalize → PCA (or a small pseudobulk autoencoder) → `e_c^data`. Aligned with your tissue by construction: two types are close iff they actually express similarly in your specimens.
2. **Prior — encoder embedding.** The `e_c` from 1.2.1 → `e_c^prior`.
3. **Agreement diagnostic.** Correlate the two pairwise-distance matrices (Mantel test) and/or compute per-type k-NN overlap. This localizes *where* the two disagree.
4. **Sample-size-weighted blend, not a global pass/fail.** Divergence concentrates in **rare / under-sampled** types, whose `e_c^data` is noisy for lack of cells — exactly where the prior should carry more weight. Blend per type:

   `e_c = w_c · e_c^data + (1 − w_c) · e_c^prior`,  with `w_c` increasing in the number of cells of type *c* (e.g. a shrinkage weight `n_c / (n_c + κ)`).

   Well-sampled types → data-driven; rare types → borrow from the prior. This turns the anchor from a decorative correlation number into an operational rule, and answers the reviewer question "why the LLM?" with "only where our own data is too thin to speak."

> **Caveat that neither embedding catches:** both are computed from markers / pseudobulk that carry **batch effects**. If a type's markers are partly technical, *both* embeddings inherit the artifact and the agreement check will happily confirm it. Batch-correct before pseudobulk — this is upstream of everything here.

### 1.3 Joint cell typing across all sections
Type **jointly across every section at once**, so labels/embeddings are consistent across the stack (integrate first if needed, then type). Keep a **hierarchy**: class → subclass → cluster. The coarse level drives structure; the fine level (and the continuous embedding) drives expression.

### 1.4 Registration into one 3D frame
The hardest step given no histology. In rough order of preference:

1. **Nuclear/DAPI channel** if it survives from segmentation — check for it; it is a far better target than labels.
2. **Blockface imaging** if sectioning hasn't happened yet — cheap, and it removes drift.
3. **Atlas registration, section-to-atlas** (not sequential pairwise) — avoids accumulated drift / the "banana" straightening artifact.
4. **Label-field registration** (mutual information or optimal transport on the rasterized type fields) as a last resort. **State the circularity explicitly:** you assume label geometry is consistent in order to align, then study how it varies across *z*. Misalignment is absorbed as apparent biology.

Constrain with landmarks extractable from points alone: midline, ventricle boundaries, tissue outline.

### 1.5 Normalized coordinates
Convert positions to **normalized anatomical coordinates** (atlas- or region-relative), not raw microns. This is what makes cross-specimen transfer possible and prevents the model memorizing absolute geometry.

### 1.6 Neighborhood graph
Build a 3D kNN / radius graph within and across sections, storing each edge's `Δz` explicitly.

---

## Phase 2 — Structure Completion (Layout Generator)

**Goal:** from sparse sections, produce a dense, sliceable field of cell-type density at any *z*.
**Deliverable:** a completed field `λ_c(x,y,z)` (or its latent), from which layouts are sampled.

### 2.1 Rasterize to a multi-channel field — not a color image
Do **not** render "colored by label" RGB; RGB distances are meaningless and diffusion produces invalid intermediate colors. Instead:

`λ_c(x,y,z) = Σ_{i: t_i = c} K_h( p − p_i )`   — one channel per type, smoothing kernel `K_h`.

Nonnegative, commensurate channels, no invalid states.

- **Many types (60–100+):** don't use one channel each. Either (a) a coarse hierarchy of ~10–20 channels, or (b) a **density-weighted mean type-embedding per voxel** using the `e_c` from Phase 1.2 — fewer channels, and the space respects type similarity, so interpolation between regions is meaningful.
- **Anisotropy:** coarsen in-plane to ~50 µm so voxel aspect ratios stay sane (raw ratios reach the hundreds and break 3D convolutions). You lose nothing — the field is smooth at that scale.
- **Slabs, not planes:** a section is a *z*-integrated projection over its thickness `∫λ dz`. Model it that way in the forward operator.

### 2.2 Compress (optional but usually required)
A hemisphere at cellular resolution is ~10⁹ voxels. Train a 3D VAE → structural latent, or use a cascade. Skip only if volumes are genuinely small.

### 2.3 Train as query-based interpolation (VFI framing, flow-free)
From **densely sectioned specimens**, sample tuples with brackets held out and a real intermediate as target:

`(I_{a−1}, I_a, I_b, I_{b+1}, t, Δz) → I_t`

Design choices:

- **Two+ slices per side.** Carries the derivative (growing vs. shrinking vs. peaking mid-gap); two brackets alone force a monotone morph.
- **Flow-free, directly generative.** No optical flow — it assumes conserved, translating content and cannot represent branching / merging / termination, which are the interesting events. Let attention find soft correspondence. (Only warp-style option worth considering: a smoothness-constrained *diffeomorphic* field, and only for large smooth structures at small `Δz`.)
- **Condition on both** normalized `t ∈ [0,1]` **and** absolute `Δz` (µm). Same `t`, different gap width = different difficulty; the model must widen its distribution for wide gaps.
- **Augmentation:** randomize gap width and `t`; use **z-reversal** (an *exact* symmetry here — free data); allow asymmetric bracket counts. Be cautious with in-plane mirroring if any biology is lateralized.

### 2.4 Deterministic multi-query for a coherent volume
At `R > 2` you need many intermediates. Independent stochastic queries don't compose (flicker/jitter). Fix it:

- **Deterministic sampler (DDIM / ODE) + fixed noise across all queries in a gap** ⇒ `t ↦ I_t` is a continuous function; nearby queries → nearby outputs.
- **Prefer direct query over recursive subdivision.** Subdivision compounds smoothing error multiplicatively; use it only if the round-trip self-consistency test (Phase-5-style) passes.
- **Uncertainty protocol:** one seed → one coherent volume (one hypothesis); `N` seeds → `N` coherent volumes; voxelwise spread = structural uncertainty.

### 2.5 Gate — orthogonal reslice + metrics
1. **Orthogonal reslice test (do this first, it's free):** generate a completed volume, reslice along the two orthogonal axes, and *look*. Striping/banding = per-slice inconsistency; crisp in-plane but mushy orthogonal = over-smoothing; tissue-like from every axis = good. Anatomy has no preferred slicing axis, so any axis-dependent appearance is an artifact.
2. **Quantitative, on held-out dense volumes:** per-type density error, boundary position along *z*, IoU — **all reported as a function of distance to nearest real slice.**
3. **Stop condition:** if mid-gap accuracy ≤ linear interpolation of the field, fix this before building Phase 3 on top of it.

---

## Phase 3 — Expression Model

**Goal:** given a cell's label/embedding, position, neighborhood, and `Δz`, generate its full expression profile.
**Deliverable:** a conditional generator from (layout entry + context) → counts.

### 3.1 Expression VAE — used as a likelihood, not a generator
- **NB or ZINB decoder** (proper count model).
- **Library size as an explicit scalar input**, kept out of the profile (total counts are technical depth, not biology).
- **Low KL weight** (near-autoencoder). Diffusion supplies the prior; over-regularizing here throws away the resolution you're trying to preserve.
- Train on **all cells pooled**; this stage ignores *z*. Verify **no posterior collapse**.

### 3.2 Conditional latent diffusion (in the frozen VAE latent)
Condition the denoiser on:

- **Type representation = the continuous embedding `e_c` from Phase 1.2** (hierarchical), *not* a bare discrete label. This is the widened bottleneck that lets within-type spatial gradients survive.
- **Normalized position.**
- **Neighborhood context:** cross-attention over the encoded latents of nearby cells in the real bracketing slices.
- **`Δz` to nearest real slice.**

Training:
- **Slice dropout** — hold out random consecutive runs; force generation from remaining context.
- **Randomize `Δz`** so the model learns to widen its output distribution as context recedes.
- **Generate the whole gap block jointly** with attention across *z* — never independent per-slice (that gives z-flicker).

Why diffusion and not the VAE alone: a KL-regularized VAE samples toward the conditional mean → profiles that look like *type-mean + noise*. Diffusion represents the (genuinely multimodal) conditional "type X at position p."

### 3.3 Gate
With **ground-truth layouts** on held-out dense volumes, beat three baselines:
- nearest-slice copy,
- linear interpolation between brackets,
- **cell-type-mean assignment.**

Judge on **spatially variable genes** and on **within-type variation** specifically. Failing the type-mean bar means the model is a label propagator — the fix is conditioning (wider embedding, stronger neighborhood context), not more capacity.

---

## Phase 4 — Inference

1. **Register** the sparse target specimen into the common frame (Phase 1 machinery).
2. **Structure:** run `N ≈ 20` deterministic completions (fixed noise per seed, `N` seeds). Query `λ_c` at any desired *z*. Voxelwise spread across seeds → **structural uncertainty map**.
3. **Layout:** sample cell positions + types from `λ_c` as a **marked spatial point process** (preserves sub-voxel positions; avoids the raster → peak-detect round trip).
4. **Context:** encode context cells from the *real* bracketing slices.
5. **Expression:** joint-block latent diffusion conditioned on layout + context + `Δz`; multiple samples → **expression uncertainty**.
6. **Decode** latents → counts via the VAE decoder.
7. **Provenance (mandatory):** attach two channels to every voxel/cell and carry them into *every* downstream figure and analysis —
   - distance to nearest real slice,
   - sample spread across seeds.
   Flag sub-Nyquist regions (*z*-extent `< 2Δz`) as **unresolved**, not as fact. Observed and generated slices are visually indistinguishable — that's the point of the method and its main hazard.

---

## Cross-cutting notes

- **Instrument each stage independently.** Structure vs. held-out dense volumes; expression vs. ground-truth layouts. If you only ever measure end-to-end, you won't know which stage failed. Errors compound across stages.
- **Two most likely failure points, both cheap to check early:**
  1. no dense supervision (Phase 0b in the fuller plan — needed to train/validate 2.3–2.5);
  2. label-bottleneck collapse (3.3). The marker-embedding design in 1.2 is the primary mitigation for #2.
- **Skip Phase 2 entirely** if the data is fixed-array (Visium / unsegmented Stereo-seq): the layout *is* the array, and only Phase 3 applies.
