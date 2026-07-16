# H3D-FLA: Hybrid 3D Flow-Matching Latent Atlas Pipeline

**Version**: 1.0 (July 2026)  
**Purpose**: Robust, high-impact computational framework for generating accurate pseudo-3D Spatial Transcriptomics atlases from aligned serial tissue sections. Designed for *Nature Methods*-level rigor with strong emphasis on TME applications, biological fidelity, and reproducible validation.

---

## Overview & Novelty

This pipeline combines:
- Joint molecular-morphological latent representations
- Conditional flow matching in latent space (for high-quality, diverse generation of missing slices)
- 3D attention-based context modeling across slices
- Explicit biology-informed constraints tailored to tumor microenvironment (TME) features
- Closed-loop consistency between morphology and molecular output
- Rigorous gap-aware training and strict per-fold validation

**Key Novel Contributions** (for manuscript framing):
- First use of **conditional flow matching in a joint expression-morphology latent space** with 3D positional attention.
- Explicit, quantifiable **TME-specific soft constraints** (tumor-immune interface coherence + hypoxia gradient preservation) applied in both latent and image space.
- Produces both **molecularly accurate virtual ST slices** and **pathologist-friendly continuous pseudo-histology volumes**.
- Strong robustness to variable z-spacing, missing sections, and z-position uncertainty via marginalization + ensembling.

The method is **hybrid** by design: it retains the molecular fidelity of latent-field approaches while leveraging the visual quality and morphological richness of modern generative image models (flow matching).

---

## Assumptions

- Serial tissue sections have already been **aligned** in 3D coordinate space (using PASTE/PASTE2 or equivalent). z-positions are known or estimated with some uncertainty.
- You have, per spot/cell:
  - Gene expression (raw or normalized counts)
  - Spatial coordinates `(x, y, z)`
  - Cell-type or region annotations (or reliable deconvolution results)
- Optional but recommended: Paired H&E images for at least some sections (used to train/improve pseudo-H&E generation).
- Target application focus: Cancer / Tumor Microenvironment (TME), with emphasis on sharp interfaces and gradients.

---

## Stage 1: Preprocessing & Pseudo-Image Generation (Already Partially Done)

Since alignment is complete:

- For each real slice \(k\) at position \(z_k\):
  1. Compute low-dimensional **expression latent** \(\mathbf{e}_i\) per spot/cell (recommended: scVI, PCA on highly variable genes, or a small VAE; target dimension \(d \approx 32-64\)).
  2. Generate **multi-channel pseudo-image stack** \(I_k\):
     - 2 channels: Pseudo H&E (nuclear + cytoplasmic/ECM). Use marker-gene weighted sums or a lightweight supervised model trained on any available paired H&E data.
     - \(C\) channels: Soft probability maps for major cell types/regions (tumor, stroma, immune subtypes, hypoxia signature areas, etc.).
  3. Rasterize to a consistent grid (start at spot resolution or 2–4× upsampled for better visual quality).

**Output per slice**: 
- Aligned coordinates
- Expression latents \(\mathbf{e}\)
- Annotations
- Multi-channel image tensor \(I_k\)

**Recommended tools**: `scanpy` + `squidpy` for spatial handling, `torchvision` or `monai` for image operations.

---

## Stage 2: Joint Latent + Morphological Encoder

Learn a shared compact representation \(\mathbf{h}\) that encodes **both** molecular (expression) and morphological (image) information.

**Architecture**:
- CNN or lightweight Vision Transformer (ViT) encoder on the multi-channel pseudo-image stack \(I\).
- Fusion MLP that combines image features with the expression latent \(\mathbf{e}\).
- Output: Unified latent code \(\mathbf{h} \in \mathbb{R}^d\) per spatial location (or per patch).

**Pre-training objective** (on real slices only):
- Reconstruction loss on both expression and image channels.
- Optional contrastive loss to encourage \(\mathbf{h}\) to be predictive of both modalities.

This stage makes the later decoding step (latent → expression + images) simple and accurate.

---

## Stage 3: 3D-Conditioned Flow Matching in Joint Latent Space (Core)

This is the primary generative component.

### 3.1 Context Module (3D Attention)
- Encode features from all real slices (aggregated \(\mathbf{h}\) + image features) + Fourier positional encoding of \(z_k\).
- Use Transformer-style or cross-attention layers over the set of real slices.
- For any query position \(z^*\), compute a rich context representation \(C(z^*)\) that captures relevant 3D relationships (local + long-range).

### 3.2 Conditional Flow Matching
We perform flow matching **in the joint latent space** \(\mathbf{h}\).

**Training objective** (Conditional Flow Matching loss):

$$
\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{h}_0, \mathbf{h}_1, C} \Bigl[ \bigl\| v_t(\mathbf{h}_t \mid t, C(z), z) - (\mathbf{h}_1 - \mathbf{h}_0) \bigr\|^2 \Bigr]
$$

where:
- \(\mathbf{h}_t = (1-t)\mathbf{h}_0 + t \mathbf{h}_1\) (straight-line OT path)
- \(\mathbf{h}_0 \sim \mathcal{N}(0, I)\)
- \(\mathbf{h}_1\) = joint latent from real (or masked) slices
- \(v_t\) = vector field network (U-Net or MLP backbone), conditioned on noisy latent, time \(t\), context \(C(z)\), and query \(z\) (Fourier encoded)

### 3.3 Gap-Aware + z-Marginalized Training (Robustness)
- Randomly mask entire slices or blocks of consecutive slices.
- Require the model to reconstruct masked \(\mathbf{h}\) (and downstream outputs) using only context from remaining slices.
- During conditioning/sampling, add small noise to \(z\) (\(z' \sim \mathcal{N}(z, \sigma_z)\)) to implement marginalization over z-position uncertainty.

---

## Stage 4: Biology-Informed Regularization (TME Focus)

These auxiliary losses are applied on **decoded** quantities and back-propagate to shape the learned flow field. They are the key to high biological fidelity in cancer/TME applications.

### 4.1 Tumor-Immune / Stroma Interface Coherence
- Decode cell-type probabilities or use gradients from pseudo-H&E.
- Penalize biologically implausible mixing away from morphological boundaries.
- Implementation: Edge-aware variation penalty or soft level-set style loss.

### 4.2 Hypoxia / Morphogen Gradient Preservation
Decode a hypoxia score \(s_h\) from \(\mathbf{h}\). In tumor regions, enforce expected directionality:

$$
\mathcal{L}_{\text{hyp}} = \sum \max\bigl(0,\ s_h(\mathbf{p}_{\text{outer}}) - s_h(\mathbf{p}_{\text{inner}}) + \text{margin}\bigr)
$$

(Computed on sampled point pairs or via finite differences along approximate inward normals.)

Optional: Add a soft PDE-style diffusion-reaction regularizer inside tumor volumes.

### 4.3 Closed-Loop Consistency
- Generated \(\mathbf{h}\) → decode to expression + pseudo-image.
- Re-encode the generated pseudo-image → should recover a similar \(\mathbf{h}\).
- Add small cycle-consistency loss.

### 4.4 Adaptive Smoothness
Latent-space total variation / Laplacian penalty, modulated by edge strength extracted from pseudo-H&E (stronger continuity inside homogeneous regions, relaxed at real interfaces).

**Total training loss** = \(\mathcal{L}_{\text{CFM}} + \lambda_1 \mathcal{L}_{\text{bio}} + \lambda_2 \mathcal{L}_{\text{consistency}} + \lambda_3 \mathcal{L}_{\text{smooth}}\)

---

## Stage 5: Inference for Arbitrary / Missing z (Virtual Slice Generation)

For any query position \(z^*\) (including large gaps or oblique planes):

1. Compute attention-derived context \(C(z^*)\) from all available real slices.
2. Sample multiple initial noises \(\mathbf{h}_0 \sim \mathcal{N}(0, I)\) (for diversity and uncertainty quantification).
3. Solve the conditional ODE:
   $$
   \frac{d\mathbf{h}}{dt} = v_t\bigl(\mathbf{h}(t) \mid C(z^*), z^*\bigr), \quad t \in [0,1]
   $$
   using an ODE solver (e.g., Euler, RK4, or `torchdiffeq`).
4. Decode each resulting \(\mathbf{h}(z^*)\):
   - Expression head → gene expression (or NB parameters)
   - Annotation head → cell-type / region probabilities
   - Image decoder → high-quality pseudo-H&E + detailed cell-type maps
5. Ensemble results (mean expression + variance; keep multiple image realizations or select best by consistency with neighbors).

**Outputs**:
- Virtual ST slice: expression matrix + coordinates + annotations
- Corresponding pseudo-histology images (ready for visualization or pathologist review)
- Uncertainty estimates

The model supports **continuous querying** — any resolution or cutting plane is possible.

---

## Stage 6: Training Strategy (Practical Implementation)

**Recommended two-phase approach**:
1. **Phase A (optional but helpful)**: Pre-train the joint encoder + decoders on real slices only (reconstruction + contrastive objectives).
2. **Phase B (main training)**: Train the flow-matching vector field + attention context module end-to-end with the full loss (CFM + biology + consistency + gap-aware terms).

**Practical tips**:
- Use `AdamW` optimizer with cosine annealing + warmup.
- Mixed precision (`torch.amp`) + gradient checkpointing.
- Expose key hyperparameters in a config file (YAML): bio-loss weights (anneal them), z-perturbation \(\sigma_z\), attention depth, flow path type.
- Log generated image samples and TME-specific metrics to Weights & Biases (or equivalent) during training.
- Libraries: `torch`, `torchdiffeq` (or `torchdyn`), `einops`, `monai`/`torchvision`, `scanpy`/`squidpy`/`anndata`.

---

## Stage 7: Validation & Benchmarking Protocol (Rigorous & Publication-Ready)

Use your existing **PASTE-based Leave-One-Out (LOO)** pipeline with **both** protocols:

- **Global alignment** (simpler, for comparison)
- **Strict per-fold alignment** (recommended for publication — align only training slices, transform held-out query coordinates into that space)

Simulate realistic gaps (single slices + consecutive blocks).

**Core Metrics** (report comprehensively):
- **Molecular fidelity**: Pearson/Spearman correlation per gene, RMSE, cosine similarity on latents.
- **Structural**: SSIM on rasterized expression maps or key genes.
- **Biological / TME-specific**:
  - Interface sharpness (gradient magnitude of immune/cell-type scores across tumor boundaries in generated maps)
  - Hypoxia gradient fidelity (directionality and expected core-to-periphery behavior)
  - 3D neighborhood enrichment and spatial autocorrelation preservation
- **Morphological** (on generated pseudo-images): Visual quality + consistency with neighboring real slices.
- **Downstream utility**: 3D spatial domain detection (ARI/NMI), cell-cell interaction inference fidelity, quality of in silico sectioning.

**Mandatory Ablations**:
- Full model vs. no biology-informed losses
- Full model vs. no 3D attention (local neighbors only)
- Full model vs. pure image-space flow matching (no joint latent)
- Full model vs. gap-aware training disabled
- Comparison baselines: SpatialZ, UniST (when available), simple z-interpolation, plain INR (coordinate → expression)

**Datasets**: Cancer serial-section datasets with paired H&E preferred; supplement with brain/embryo data for breadth. Use any available dense/true 3D ST for near-ground-truth testing.

**Reproducibility**:
- Full YAML config files
- Exact dataset versions and processing scripts
- Fixed random seeds
- Docker / conda environment specification
- Optional: Simple `napari`-based interactive viewer for exploring the continuous 3D atlas (query any z, overlay expression + pseudo-H&E)

---

## Why This Pipeline Has High Impact Potential

- **Novelty**: Combines the strengths of continuous latent fields and modern generative modeling (flow matching) in a 3D-aware, biology-constrained framework — not present in existing literature.
- **Robustness & Accuracy**: Handles real-world challenges (missing sections, z-uncertainty) while preserving critical TME features (sharp interfaces, coherent gradients) that purely data-driven methods often distort.
- **Dual Output**: Delivers both scientifically accurate molecular virtual slices **and** visually interpretable pseudo-histology — highly valuable for biologists and pathologists.
- **Validation Rigor**: Strict LOO protocol + multi-faceted metrics (including explicit biological fidelity) + comprehensive ablations directly address common reviewer concerns.
- **Practical & Extensible**: Modular design, implementable in stages, and naturally extends to multi-sample population atlases or integration with emerging true 3D ST technologies.

---

## Recommended Implementation Order (for Coder)

1. Stage 1 (finish pseudo-image generation) + basic visualization.
2. Stage 2 (joint encoder + decoders) — validate reconstruction quality on real data.
3. Stage 3 (attention context + conditional flow matching) — get sampling working without bio losses first.
4. Stage 4 (add TME biology constraints + closed-loop losses).
5. Stage 5 (inference pipeline + ensembling).
6. Stage 7 (build strict per-fold LOO evaluation harness early — this will save time later).

---

**File generated**: `H3D_FLA_Pipeline.md`  
This document is ready to be used as the foundation for your Methods section, implementation plan, or collaborator/advisor communication.

If you need expansions (detailed pseudocode for specific stages, config file template, loss implementation sketches, or a companion `README.md` with environment setup), just let me know and I will append or create additional files. 

You now have a complete, editor-vetted, high-impact pipeline tailored to your aligned data and TME-focused 3D atlas goals. Ready to start coding!