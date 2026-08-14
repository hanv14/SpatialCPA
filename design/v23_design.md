# SpatialCPA v23 — **CTF-Flow**: a Continuous Transcriptomic Field for in-silico sectioning

*Design document. Target: beat SpatialZ with clear median gaps on Moran's I agreement, Geary's C
agreement, embedding mixing, marker field r, marker depth r, and cell-type localization — in both
the alternating-holdout (narrow gap) and consecutive-holdout (wide gap) regimes.*

---

## 0. Executive summary

**The reframing.** v14–v20 treat virtual-slice generation as *recombination of two flanking
sections*. Every version has been a negotiation between copying (wins structure metrics, loses
distribution metrics) and generating (the reverse). That trade-off is an artifact of the framing,
not of the biology. v23 drops it: the tissue is modelled as a **continuous field over
(x, y, z) × gene-space**, and a section — at any depth, any orientation, even a curved surface — is
a *query* against that field.

**Four claims that make it publishable.**

1. **Open-vocabulary genes.** Genes enter the model through their *text* (symbol + name + NCBI
   summary) encoded by MedCPT, not through a fixed panel index. One decoder serves any gene. This
   enables the headline capability: **generating spatially-resolved expression for genes that were
   never measured in the target tissue**, and transferring a model across panels/technologies.
   Nobody has this for 3D slice interpolation.
2. **Arbitrary-plane in-silico sectioning.** Because the model is a field, it generates oblique
   and anatomically-curved sections natively. SpatialZ can only reach oblique planes by rotating a
   post-hoc dense atlas (`Synthesize.py`) and slicing the point cloud — resolution is capped by
   what was already reconstructed, and the slab is a thick projection, not a section.
3. **Spatially-correlated flow matching.** The generative prior is a Gaussian *random field*, not
   iid noise, with a correlation length calibrated to the tissue. This is the mechanism that
   preserves spatial autocorrelation instead of destroying it — the failure mode that killed v19.
4. **Metric-aware training by internal leave-one-section-out.** The six target metrics are turned
   into differentiable training signals computed on *held-in* sections only. This is the
   engineering reason to expect wins on exactly those six metrics, and it is leakage-free.

**Safety property.** v23 is a strict superset of v20: every new component is gated, and the gates
are chosen per dataset by the internal LOSO loop. If a component doesn't help on a given dataset,
it is switched off automatically and v23 degrades to v20. You cannot regress relative to your
current numbers. This also dissolves the "universal flags" problem — there is no manual tuning.

---

## 1. What SpatialZ actually does, and where it is beatable

From `SpatialZ.py`:

| Stage | SpatialZ mechanism | Structural weakness |
|---|---|---|
| Layout | Points initialised by uniform index sampling from both flanks, then gradient descent on a weighted **sliced Wasserstein distance** to the two flanks' coordinate clouds (`nb_iter_max=3000`) | Matches only the *marginal geometry* of the point cloud. No pair-correlation / repulsion term, so local density statistics and hard-core spacing are not preserved. Positions drift from any real niche. |
| Cell type | Distance-weighted kNN vote over both flanks, `k_neighbors=1` by default, weights `1/(d+ε)` | With k=1 this is nearest-neighbour copying. No z-weighting: a cell at α=0.2 gets the same vote structure as at α=0.8. |
| Niche | MENDER multi-scale composition vector, `scale=6, radius=15` | Fixed physical radius — not adaptive to cell density across datasets. |
| Expression | For each cell, kNN in space → filter to same type → weights `softmax(cosine(MENDER))` → **per-gene independent categorical draw** among ≤ `k_sam=3` donors | This is the key weakness. (a) Per-gene independent draws across donors **destroy gene–gene covariance** within the cell — a chimera. (b) The donor weights contain **no z-proximity term**, so a cell at α=0.2 draws ~50/50 from both flanks; correct only at α=0.5, systematically wrong elsewhere. (c) Only ≤3 donors → heavy atoms. |

**Where each of your six metrics is winnable:**

- *Moran's I / Geary's C agreement* — SpatialZ's per-gene chimerism injects independent per-gene
  noise across neighbouring cells, which **attenuates** spatial autocorrelation. Its I is
  systematically compressed toward 0. A model with a calibrated correlation length wins here.
- *Embedding mixing* — SpatialZ's strength (chimeric profiles are novel points, maximising cloud
  overlap). To beat it we must generate genuinely novel profiles too, but from a *correct*
  conditional distribution. This is the hardest of the six.
- *Marker field / depth profiles* — SpatialZ has no z-interpolation of expression at all, so
  mid-gap marker gradients are a 50/50 blend of two flanks rather than the intermediate state.
  Directly winnable, and the win grows with gap width.
- *Cell-type localization* — SpatialZ's k=1 vote produces a noisy type mosaic with no explicit
  spatial intensity model. An explicit per-type intensity field is a large, direct improvement.

---

## 2. Representation

### 2.1 The observation token

Every measurement in the training data is a token:

```
τ(i, g) = [ e_g , c_i , φ(p_i) , y_ig ]
```

- `e_g` — **gene embedding** (§2.2)
- `c_i` — **cell context embedding**: cell type + region label, text-encoded (§2.2)
- `φ(p_i)` — **anisotropic Fourier encoding** of (x, y, z) (§2.3)
- `y_ig` — measured expression (raw count or intensity, kept on its native scale)

Training never materialises all tokens: at each step we subsample `G' ≈ 128` genes per cell. This
is what makes the model panel-size-agnostic and keeps memory flat.

### 2.2 Text-grounded embeddings (the open-vocabulary mechanism)

For each gene, build a descriptor string:

```
"{symbol}. {full name}. {NCBI Gene summary}. Aliases: {...}"
```

Encode with **MedCPT-Query-Encoder** (frozen) → `t_g ∈ R^768`. Same for cell types (Cell Ontology
label + definition) and regions (e.g. Allen Reference Atlas structure name + parent path) → `t_c`.

The embedding used by the model is

```
e_g = LayerNorm( W t_g  +  γ · r_g ),      r_g ∈ R^{d_g} free, learned per gene
```

`γ` is annealed 0 → 1 over training. **Critical design point:** the free residual `r_g` is what
lets the model learn expression-specific structure that literature text cannot express; but a
freely-learned residual is unavailable for an unseen gene. So we additionally train a small
**text→residual distillation head** `ψ: t_g ↦ r̂_g` with an L2 loss against the learned `r_g`.
At test time on an unseen gene, `r_g := ψ(t_g)`. Report both `r_g = 0` (pure text) and
`r_g = ψ(t_g)` (distilled) in the zero-shot table.

> **Honest risk.** MedCPT embeddings of gene names carry literature co-occurrence semantics, which
> correlate with but do not determine expression covariance. There is a real chance the text
> channel adds little beyond a learned lookup on the seen-gene benchmark. The design is arranged so
> this is *diagnosable rather than fatal*: ablation A3 (§7) replaces `e_g` with a free lookup table.
> If A3 ties on seen genes, the text channel's value is entirely in the zero-shot experiment — which
> is still the novel contribution, just framed as "enables a new capability" rather than "improves
> accuracy".

### 2.3 Anisotropic position encoding

`φ(x,y,z)` uses **B_xy = 8** Fourier bands in-plane but only **B_z = 2** along z. With 4–7 training
sections, high-frequency z basis functions are unconstrained and will overfit to section positions,
producing a field that is spiky exactly where you need it smooth. The z-smoothness prior belongs in
the basis, not only in the regulariser.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
   (x,y,z) ───────► │  TRIPLANE ANATOMICAL FIELD  F(x,y,z) → f    │
                    │  XY / XZ / YZ feature planes + MLP          │
                    └───────────────┬─────────────────────────────┘
                                    │  f ∈ R^{d_f}
      real cells of nearby ─────────┼──► RETRIEVAL CROSS-ATTENTION  ──► ctx
      sections (gap-aware)          │        (K real neighbours)
                                    ▼
                    ┌─────────────────────────────────────────────┐
       HEAD A       │  LAYOUT: per-type intensity λ_c(x,y,z)      │──► positions + types
                    │  + Strauss repulsion sampler                │
                    └─────────────────────────────────────────────┘
                                    │
                    ┌───────────────▼─────────────────────────────┐
       HEAD B       │  EXPRESSION: flow matching in cell latent h │
                    │  h_0 ~ correlated GRF prior (NOT iid)       │──► h_i
                    └───────────────┬─────────────────────────────┘
                                    ▼
                    ┌─────────────────────────────────────────────┐
       DECODER      │  gene-conditioned ZINB head:                │
                    │  (μ,θ,π)_ig = g([h_i , e_g , h_i ⊙ A e_g])  │──► sampled counts
                    └─────────────────────────────────────────────┘
```

### 3.1 Triplane anatomical field

Three learned feature planes (XY at 256², XZ and YZ at 256×32) with bilinear lookup, concatenated
and passed through a 3-layer MLP → `f`. Regularisation: total-variation penalty along the z axis of
the XZ/YZ planes, plus stochastic **whole-section dropout** during training (inherit `gap_dropout`
from v20). The field carries anatomy (laminar structure, region boundaries, density gradients); the
retrieval branch carries realism.

### 3.2 Retrieval cross-attention (keeps the copying strength)

For a query position, retrieve K=32 real cells from sections within a z-window, ranked by a hybrid
score: in-plane distance + niche similarity + **z-proximity** (the term SpatialZ omits). Cross-
attend from the query to these cells' encoded tokens. Under whole-section dropout the model is
forced to reconstruct from remoter sections — this is the wide-gap curriculum, and unlike v19's
version it is *not* inert, because the retrieval branch is load-bearing at inference.

### 3.3 Head A — layout as a marked point process

This is the direct attack on **cell-type localization**.

1. Decode per-type intensity `λ_c(x,y,z) = softplus(MLP_c([f, region-emb]))`, trained by a
   Poisson-process NLL against the training sections' observed points.
2. Total expected count `N = ∫ Σ_c λ_c` over the section's support → gives cell number *emergently*
   and correctly (no more `n_target` interpolation heuristic).
3. Sample positions by **thinning with a Strauss/hard-core repulsion kernel**, with interaction
   radius and strength fitted to the flanking sections' empirical pair-correlation function `g(r)`.
   This reproduces the real nearest-neighbour spacing distribution — SpatialZ's SWD objective does
   not constrain it at all, and coincident/overdense points are exactly what corrupts kNN-graph
   metrics like Moran's I and neighbourhood enrichment.
4. Mark each point with a type sampled from `λ_c(p)/Σ_c λ_c(p)`, then apply one round of
   **Potts smoothing** on the kNN graph to remove isolated singletons (types are spatially
   organised; independent marks are too noisy).

*Fallback gate:* `layout_mode ∈ {field, resample, hybrid}`. `hybrid` = field-sampled positions
followed by a short SWD polish toward the flank marginals (borrowing SpatialZ's one genuine
strength). Chosen by LOSO (§6).

### 3.4 Head B — expression by flow matching with a correlated prior

Per-cell latent `h ∈ R^{64}`. Conditional flow matching on the straight-line path, conditioned on
`[f(p_i), type-emb, ctx, φ_z(z)]`, exactly as in v20 — this part is proven machinery.

**The novelty is the prior.** Instead of `h_0^i ~ N(0, I)` independently per cell:

```
ε ~ N(0, I)^{n×d}
h_0 = S_ℓ ε ,   S_ℓ = graph filter on the generated cells' kNN graph
                      approximating a Matérn(ν=3/2, length-scale ℓ) kernel,
                      row-normalised so Var(h_0^i) = 1
```

Rationale: the flow map `h_0 ↦ h_1` is smooth, so spatially correlated input noise yields spatially
correlated output latents, and hence spatially autocorrelated expression. With iid noise, every
cell's stochastic component is independent, which *attenuates* Moran's I toward the value the
conditioning alone supports — this is precisely v19's collapse and SpatialZ's compression.

**Calibrating ℓ without leakage.** ℓ is fitted by matching the *generated* section's mean Moran's I
to the mean Moran's I of the **flanking training sections** (a 1-D bisection over ℓ, ~6 forward
passes). Ground truth is never touched. Do it per gene-module (Leiden clusters of the gene
embedding space, ~10 modules) rather than globally, since autocorrelation length is gene-dependent.

### 3.5 Decoder — gene-conditioned ZINB

```
u_ig = [ h_i , e_g , h_i ⊙ (A e_g) ]           # bilinear interaction, FiLM-style
(μ, θ, π)_ig = softplus/sigmoid( MLP(u_ig) )
y_ig ~ ZINB(μ_ig, θ_ig, π_ig)   ·  s_i          # s_i = cell size factor, decoded from h_i
```

Two properties this buys, both of which were the actual failure modes of v19:

- **Sparsity and count-ness by construction.** `π` reproduces detection frequency; you cannot get
  the 4.2× densification that convex interpolation produced.
- **Gene–gene covariance from a shared latent.** All genes in a cell are decoded from the same
  `h_i`, so covariance is inherited from the latent, not destroyed as in SpatialZ's independent
  per-gene draws.

Calibrate `π` and the mean–variance relation per gene against the flanking sections
(leakage-free), so per-gene detection frequency matches the local tissue.

---

## 4. Training objective

```
L = L_CFM                                   flow matching (straight-line CFM)
  + λ_rec · L_ZINB                          NLL of real cells under the decoder
  + λ_lay · L_layout                        Poisson NLL + pair-correlation match
  + λ_spa · L_autocorr        ◄─ metric-aware
  + λ_prf · L_profile         ◄─ metric-aware
  + λ_dst · L_distribution    ◄─ metric-aware
  + λ_txt · L_distill                       text → residual embedding distillation
  + λ_reg · (TV_z + weight decay)
```

### The metric-aware terms (§ the strategic core)

At each epoch, pick one **held-in** section `s`, hide it, reconstruct it from the remaining
training sections, and compute:

- **`L_autocorr`** — soft Moran's I and Geary's C on the reconstruction vs. the real section `s`,
  matched per gene by Huber loss on the I/C vectors. (Both statistics are differentiable given a
  fixed kNN graph.) *Directly optimises target metrics 1 and 2.*
- **`L_profile`** — bin cells along the section's principal tissue axes with a **soft** (Gaussian
  kernel) assignment, and match binned marker profiles and 2-D binned expression fields between
  reconstruction and truth. *Directly optimises target metrics 4 and 5.*
- **`L_distribution`** — entropic Sinkhorn divergence (or MMD with a mixture of RBF bandwidths)
  between reconstructed and real cells in a fixed PCA embedding. *Directly optimises target
  metric 3 (embedding mixing).*
- Cell-type localization (metric 6) is optimised by `L_layout` plus a per-type spatial-histogram
  matching term folded into `L_profile`.

This is the honest reason to expect the six wins: **each is a training signal**, computed only
from training sections, and none of the competing methods optimises any of them. It is standard
practice (train on a surrogate of the evaluation criterion via internal CV) and is not leakage —
but the paper must state it explicitly and prominently, and must show the model still wins on
*held-out metrics that were never trained on* (§7, "unoptimised metrics" row), otherwise a reviewer
will read it as benchmark gaming.

---

## 5. Inference

```
1. Query the field on the target plane Π (any normal, or a curved manifold).
2. Head A → N, positions {p_i}, types {c_i}.        [layout]
3. Build correlated prior h_0 = S_ℓ ε on {p_i}.     [ℓ from flank calibration]
4. Retrieve K real neighbours per cell; build ctx.
5. Integrate the flow ODE 0→1 (Heun, 24 steps) → {h_i}.
6. Decode ZINB per (cell, gene) and sample counts.
7. Uncertainty-gated anchoring (below).
8. Emit AnnData with obsm['spatial'], obs[cell_type, region], X counts.
```

### Uncertainty-gated anchoring (replaces v20's hand-tuned `alpha`)

Run `M=8` flow samples; per-cell latent variance `v_i` is an uncertainty estimate. Where `v_i` is
low (narrow gap, well-supported anatomy) blend toward a retrieval-anchored real profile via the v20
**Bernoulli cross-mix** (which is retained — it is a good mechanism and preserves count-ness);
where `v_i` is high (wide gap) trust the generative sample. The blend weight is a *learned*
function of `v_i` fitted on the LOSO reconstructions, so the narrow/wide regime adaptation is
inferred rather than set by `gap_scale`. **This deletes `--gap-scale`, `--alpha-tol`,
`--edit-gap-extra`, `--edit-weight` from the user-facing surface.**

---

## 6. Automatic per-dataset configuration (the "universal flags" answer)

Before the final fit, run a **LOSO model-selection sweep** over a small discrete grid:

| Gate | Options |
|---|---|
| `layout_mode` | field / hybrid / resample (=v20) |
| `prior_mode` | correlated / iid |
| `expr_mode` | zinb-flow / cross-mix (=v20) / auto-blend |
| `text_emb` | medcpt+residual / lookup-only |

Score each configuration by the six target metrics computed on held-in sections, aggregate by
median rank, pick the winner. Cost: the grid is 3×2×3×2 = 36 but the gates are near-separable, so a
coordinate-descent sweep needs ~10 fits of a *reduced-epoch* model. This is the mechanism behind
the "cannot regress below v20" guarantee, and it is a legitimately attractive property to report:
**one command, no flags, per-dataset adaptation by internal cross-validation.**

---

## 7. Experiments

**Datasets.** Your existing 18. Add ≥1 non-brain (embryo / tumour) and ≥1 non-transcriptomic-panel
(EASI-FISH) to show generality — reviewers at high-impact venues will check for brain-only
overfitting.

**Regimes.** (a) alternating holdout; (b) consecutive holdout (3 and 5 sections). Report both
separately — the story is "ties or wins narrow, wins decisively wide."

**Baselines.** SpatialZ (`syn_mode='default'`, its published settings), v14, v18, v20, plus two
honest ablations of your own method, plus a naive nearest-section copy (the floor) and a convex
interpolation (the ceiling for smoothness, floor for realism).

**Statistics.** Per-section metrics pooled across datasets → paired Wilcoxon signed-rank vs.
SpatialZ, Benjamini–Hochberg across the 6 metrics, and report **median difference with a 95%
bootstrap CI** (this is your "clear gap in medians" requirement, stated defensibly).

**Ablations (each isolates one claim).**

| ID | Ablation | Claim tested |
|---|---|---|
| A1 | iid prior instead of correlated GRF | correlated prior is what preserves Moran's/Geary's |
| A2 | remove metric-aware LOSO losses | how much comes from metric-aware training |
| A3 | MedCPT `e_g` → free lookup table | text channel's contribution on seen genes |
| A4 | remove Strauss repulsion in layout | point-process realism → neighbourhood metrics |
| A5 | remove z-proximity in retrieval | the specific SpatialZ flaw you're fixing |
| A6 | ZINB decoder → Gaussian mean regression | sparsity/dispersion preservation |

**Unoptimised-metric check (reviewer defence).** Report ≥4 metrics that were *not* in the training
objective — Sinkhorn distance on raw profiles, co-expression module preservation, neighbourhood
enrichment z-scores, per-gene variance rank correlation, duplicate-profile rate. Winning these too
is what separates "better model" from "metric gaming."

**Capability experiments (the novelty proofs).**

- **E1 Zero-shot genes.** Hold out 20% of genes entirely from training; generate them in held-out
  sections; evaluate Moran's I / depth profile / mean rank correlation. Compare `r_g = 0` vs.
  distilled `ψ(t_g)`. *No competing method can even attempt this.*
- **E2 Cross-panel transfer.** Train on dataset A, generate dataset B's panel in tissue B with no
  retraining of the gene vocabulary. Report degradation vs. a model trained on B.
- **E3 Oblique / curved sectioning.** Generate sagittal, horizontal and anatomically-curved
  (cortical-layer-following) sections from a coronally-sectioned volume; validate against a real
  volume that *was* sectioned in that orientation. This is the figure that sells the paper — and it
  is exactly what `Synthesize.py` approximates by slicing a thick slab out of a rotated point cloud.
- **E4 Throughput.** z-resolution sweep: generate 10× the measured section density and show
  reconstruction of 3-D structures (vasculature, laminae) invisible at native sampling.

---

## 8. Risks, ranked, with mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Generative sampling over-smooths → Moran's collapse (the v19 failure, repeated) | **High** | Correlated prior + ZINB sampling + `L_autocorr` + leakage-free ℓ calibration. Three independent defences; A1/A6 diagnose which is carrying the weight. |
| Triplane overfits z with 4–7 sections | High | Anisotropic Fourier (B_z=2), TV_z penalty, section dropout, LOSO validation of the field itself. |
| MedCPT text adds nothing on seen genes | Medium | Expected outcome is acceptable — reframe as capability (E1/E2) not accuracy. A3 makes this explicit rather than hidden. |
| Field-sampled layout worse than resampling real coordinates | Medium | `layout_mode` gate with `hybrid` and `resample` fallbacks, selected by LOSO. |
| Embedding mixing — SpatialZ's chimerism is genuinely strong here | Medium | ZINB sampling from a correct conditional gives novelty *without* chimerism; `L_distribution` targets it directly. This is the metric most likely to be a tie rather than a win; be prepared to report a tie honestly. |
| Reviewer reads metric-aware training as gaming | Medium | Prominent disclosure + the unoptimised-metric table + A2 showing wins survive without it. |
| Compute | Low | Gene subsampling (G'=128), retrieval K=32, triplane not voxel grid. One A100-hour per dataset fit is realistic. |

---

## 9. Paper framing

**Title (working).** *A continuous transcriptomic field enables open-vocabulary in-silico
sectioning of three-dimensional tissues.*

**The three-sentence pitch.** 3-D spatial transcriptomics is bottlenecked by z-sampling: sections
are expensive, so volumes are sparse along depth and any plane other than the cutting plane is
unrecoverable. We model the tissue as a continuous field over space and an *open vocabulary* of
genes, so that a section at any depth, any orientation, and any gene panel becomes a query rather
than an experiment. The field reconstructs held-out sections more faithfully than the state of the
art across N datasets, generates spatially-resolved expression for genes that were never measured,
and produces oblique sections validated against orthogonally-sectioned tissue.

**Figures.** (1) Concept + field schematic. (2) Benchmark: six metrics × two regimes × N datasets,
with the median-gap forest plot. (3) Ablations. (4) Zero-shot genes. (5) Oblique/curved sectioning
validated against real orthogonal data. (6) Biological application — a 3-D structure recovered at
10× z-resolution that is invisible at native sampling.

**Venue.** The benchmark alone is *Genome Biology* / *Nature Communications*. What lifts it to
*Nature Methods* / *Nature Biotechnology* is **Figure 5 + Figure 6**: a validated new capability
(orientation-free sectioning, unmeasured genes) plus one real biological finding enabled by it. If
you have to cut scope, cut ablations before you cut Figure 6 — a benchmark win alone will not
carry a high-impact venue, and this is the most common failure mode for methods papers in this
space.

---

## 10. Build order

1. **Week 1–2 — scaffolding.** Token representation, MedCPT descriptor caching, triplane field,
   ZINB decoder. Validate: reconstruct *held-in* sections better than v20's exemplar copying.
2. **Week 3 — the prior.** Correlated GRF prior + leakage-free ℓ calibration. Validate on the
   synthetic sparse-count harness you already have. **This is the gate: if A1 doesn't show a clear
   Moran's I effect here, stop and rethink before building anything else.**
3. **Week 4 — Head A.** Intensity field + Strauss sampler + Potts smoothing. Validate on cell-type
   localization and pair-correlation functions.
4. **Week 5 — metric-aware losses + LOSO selection loop.**
5. **Week 6 — full benchmark**, both regimes, all baselines, statistics.
6. **Week 7–8 — capability experiments** E1–E4.

Step 2 is the true decision point. Everything downstream assumes the correlated prior solves the
autocorrelation problem that has defeated every generative variant so far; it is cheap to test in
isolation, and it should be tested before the rest is built.
