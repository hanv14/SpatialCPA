# v23 Addendum — **Sectioning-Equivariant Field Learning (SEFL)**

*Incorporating in-silico sectioning into training as a self-supervised constraint.*

This addendum extends `v23_design.md`. It replaces §3.4's per-section prior and §4's objective with
a volume-level formulation, and adds three new losses plus one architectural correction.

---

## 1. The core principle, stated precisely

A section is an *observation operator* applied to a tissue, not a thing in itself. Write

```
S(Π, τ) [ V ]  →  a section
```

where `V` is the (latent) tissue volume, `Π` a plane (origin + normal), `τ` a thickness. The
training signal is that **all valid observation operators must be consistent with one shared `V`**.
No ground truth is needed to enforce this, which is why it is such a powerful addition when only
4–7 real sections exist.

### 1.1 What is invariant vs. equivariant — the critical distinction

Getting this wrong destroys the metrics you are trying to win. Force invariance only on the left
column.

| **INVARIANT** — constrain | **EQUIVARIANT** — do *not* constrain to be equal |
|---|---|
| The 3D field itself: `F(x,y,z)` is independent of any Π | In-plane Moran's I / Geary's C (depends on angle vs. tissue anisotropy) |
| Cell identity and state at a physical point `p` | Depth / laminar profiles (a tangential cut through cortex has no laminar gradient) |
| `p(expression \| cell type, region)` — the molecular program | Cells per unit *area* (a slab through a fibre tract vs. across it) |
| Gene–gene covariance, module structure, module scores | Apparent structure size and elongation |
| Total cells per unit *volume* (density) | Total cells per section (scales with slab volume and angle) |

The right column is not noise — it is *predictable structure*. Its correct treatment is
**equivariance**: derive how in-plane covariance should transform with plane orientation from the
3D covariance structure, and use agreement with that prediction as a *validation* metric (§5, V3),
not a loss. This turns anisotropy from a confound into evidence that the model has learned genuine
3D structure.

---

## 2. Architectural change: a volume-level stochastic field

**Current (v23 §3.4):** correlated Gaussian prior `h_0 = S_ℓ ε` built on the *generated 2D point
set* of one section. Consequence: two sections generated at different angles have independent noise
and therefore disagree where they cross.

**Replacement:** define a single 3D Gaussian random field over the volume,

```
ξ : R^3 → R^{d_h},     ξ ~ GRF( Matérn(ν=3/2, ℓ) )
```

realised once per generation *episode* and queried at cell positions: `h_0^i = ξ(p_i)`.
Implementation: a fixed-seed random Fourier feature expansion,
`ξ(p) = sqrt(2/M) Σ_m a_m cos(ω_mᵀ p + b_m)` with `ω_m` drawn from the Matérn spectral density.
This is O(1) to query at any point, exactly reproducible from a seed, and **continuous in 3D** —
so two planes crossing at a line receive *identical* noise along that line.

Three consequences:

1. **Intersection consistency becomes exact by construction**, not approximate. The loss in §3.1
   then only has to correct the *conditioning* pathway (field + retrieval), which is a far easier
   optimisation problem than making two independent stochastic processes agree.
2. All virtual sections of one tissue are **mutually coherent** — they are slices of one object.
   This is the property that makes "in-silico sectioning" a real capability rather than a gallery of
   independently-plausible pictures.
3. `ℓ` retains its meaning and its leakage-free calibration (match flanking-section Moran's I), but
   now `ℓ` may be anisotropic: `ℓ = (ℓ_xy, ℓ_z)`. Fit `ℓ_xy` from in-plane autocorrelation of
   training sections, `ℓ_z` from between-section correlation decay. **This anisotropy is the
   parameter that makes oblique sectioning quantitatively correct** — with isotropic `ℓ`, oblique
   sections get in-plane autocorrelation that is wrong by a factor depending on angle.

### 2.1 Rotation equivariance — a correction that must be made

Axis-aligned triplanes (v23 §3.1) are **not rotation-equivariant**. Representational capacity is
concentrated on the XY plane, so coronal sections would be systematically sharper than oblique ones
— which would undercut the paper's central capability claim in the most embarrassing possible way
(reviewers will slice at 45°).

Three fixes, use the first two:

- **(a) Random global rotation augmentation.** Each training step, rotate the entire volume
  (coordinates, planes, retrieval neighbourhoods) by a random `R ∈ SO(3)`, with anatomically
  plausible bias if desired. Forces the field to be orientation-agnostic. Cheap and effective.
- **(b) Multi-orientation plane ensemble.** Instead of one axis-aligned triplane set, use `P = 4`
  triplane sets at fixed, mutually-oblique orientations (e.g. tetrahedral), summed. Reduces
  directional bias in the basis itself at 4× feature-plane memory (still far below a dense voxel grid).
- (c) Steerable/spherical-harmonic feature backbone — principled and fully equivariant, but a large
  implementation cost. Hold in reserve if (a)+(b) prove insufficient on the oblique benchmark.

Note the interaction with v23 §2.3: the anisotropic Fourier encoding (`B_z = 2`) is a *sampling*
prior, justified by having few sections along z, and it is applied in **data space** before
augmentation. After random rotation, the low-frequency axis follows the tissue's true sampling axis,
not the model's z. Implement carefully — this is the easiest place to introduce a silent bug.

---

## 3. The three new losses

All are computed on **synthetic sectioning tasks with no ground truth**, so they can be applied at
unlimited volume. This is the answer to the small-N problem: 5 real sections become tens of
thousands of self-supervised constraints.

### 3.1 `L_cross` — plane-intersection consistency

```
Sample two planes Π₁, Π₂ that intersect inside the tissue support, at a random dihedral angle.
Generate both sections (shared ξ, shared seed).
Let L = Π₁ ∩ Π₂ (a line segment). Sample n_L points along L.
For each point, obtain the model's predicted conditional state from each section's pathway:
    (h, μ, θ, π, type-logits, intensity λ_c)  from branch 1 and branch 2.
L_cross = D_KL( ZINB₁ ‖ ZINB₂ ) + ‖h₁ − h₂‖² + CE(type-logits₁, type-logits₂) + ‖λ¹ − λ²‖²
```

> **Measured at T07 (2026-08-17) — this loss is not needed in v25, and it is harmful here.**
> The table in §5 says intersection consistency is "**Exact by construction** (shared 3D noise
> field), and additionally trained for (`L_cross`)". The first half is right and is now asserted
> bitwise on an *untrained* model: the continuous 3D field gives both branches the identical noise,
> and the conditioning pathway is data-frame (retrieval, GRF and Fourier encoding are all queried at
> physical points), so two crossing sections agree exactly where they cross whatever the model has
> learned. The second half does not survive contact. Once the pathway is plane-independent, the only
> thing two branches can differ by is the **augmentation pose** — and §2.1(a)'s pose-dependent
> triplane is a *capacity* device, so a loss that equalises poses is a loss that asks the backbone
> to give that capacity up. It does: generated per-gene variance falls to **0.067** of the real
> section's at `λ_x = 0.3`, against **0.711** with SEFL off, while `L_cross` itself falls 90 %.
> `w_cross` ships at 0; `L_thick` and `L_prog` are unaffected and stay on.
>
> For a cross-plane loss to earn its keep the branches would have to differ by the **evidence** each
> plane has — its own flanking sections and dropout, as in v20 and the competing method — rather
> than by the pose. That is the version of the loss that still has content in v25, and it is a
> design change rather than a re-weighting.

**Anti-collapse:** compute branch 2 under an **EMA teacher** with stop-gradient (BYOL-style). A
symmetric consistency loss has a trivial minimiser — a constant field — and will find it. This is
not optional.

**Why this earns its keep on your target metrics:** the intersection line runs *through* the depth
axis of the tissue at arbitrary angles, so agreement along it forces the depth gradient of every
marker to be geometrically coherent in 3D rather than fitted independently per plane. That is a
direct, strong regulariser on `paper_marker_depth_r` and `paper_marker_field_r`. Likewise agreement
of `λ_c` along arbitrary lines sharpens 3D cell-type localisation.

### 3.2 `L_thick` — thickness coarse-graining consistency

A thick section is an aggregate of thin ones:

```
S(Π, 3h) ≡ aggregate[ S(Π, h; offset −h), S(Π, h; 0), S(Π, h; +h) ]
```

Enforce that the model's generation at thickness `3h` matches the aggregation of three thin
generations. **Aggregate correctly** — this is where naive implementations go wrong:

- Cell **counts add** (intensity integrates over slab volume). Do not force equal counts.
- For single-cell data, the thick section is the *union* of cells; match the empirical distribution
  (Sinkhorn) of cell states and the per-type counts, not a per-cell correspondence.
- For spot/bin-level data, expression **sums** within a bin; match binned totals.

Payoff beyond regularisation: this is a principled **cross-technology harmonisation** mechanism.
Datasets with 10 µm and 30 µm sections, or single-cell imaging vs. spot-based capture, become
consistent observations of one field under different operators. That is a genuine methodological
contribution and worth its own subsection in the paper.

### 3.3 `L_prog` — molecular-program invariance

Your original phrasing, made precise:

```
For sampling angles θ₁, θ₂ and each (cell type c, region r) present in both:
L_prog = Σ_{c,r} MMD²( { expr of cells with (c,r) sampled at θ₁ },
                        { expr of cells with (c,r) sampled at θ₂ } )
       + ‖ Corr_gene(θ₁) − Corr_gene(θ₂) ‖_F²        # gene–gene covariance invariance
       + Σ_m ‖ mean module-score_m(θ₁) − mean module-score_m(θ₂) ‖²
```

Note it is conditioned on `(c, r)`. Unconditional matching would be wrong — different planes
genuinely sample different *mixtures* of types and regions, and forcing the marginals to match would
force the model to hallucinate a homogeneous tissue. The gene–gene covariance term is what most
directly attacks SpatialZ's per-gene chimerism, and it feeds `paper_umap_mixing`.

### 3.4 Updated objective

```
L = L_CFM + λ_rec L_ZINB + λ_lay L_layout                          # reconstruction (real sections)
  + λ_spa L_autocorr + λ_prf L_profile + λ_dst L_distribution       # metric-aware LOSO (v23 §4)
  + λ_x L_cross + λ_t L_thick + λ_p L_prog                          # SEFL, self-supervised
  + λ_txt L_distill + λ_reg (TV_z + wd)
```

Schedule: warm up on reconstruction alone for ~20% of training (the consistency losses are
satisfiable by degenerate solutions early on), then ramp `λ_x, λ_t, λ_p` linearly. Keep the
reconstruction terms dominant throughout — consistency is a regulariser, not the objective.

---

## 4. Why this is not SpatialZ's dense-atlas story

SpatialZ *does* claim in-silico sectioning, via `Synthesize.py`. The distinction must be made
explicitly and early in the paper, because a reviewer will raise it:

| | SpatialZ | v23 + SEFL |
|---|---|---|
| Object produced | A **discrete point cloud** densified along z by repeated pairwise interpolation | A **continuous field**; sections are queries |
| Oblique sections | Rotate the finished point cloud and keep points within `slice_thickness` of a plane — a **thick slab projection** of pre-existing points | Direct generation on the plane at native resolution |
| Resolution off-axis | Capped by the reconstructed z-density; between-slice regions contain no points | Unbounded; the field is defined everywhere |
| Consistency between two different sections | **None.** Each slice comes from an independent SWD optimisation and independent per-gene sampling. Two oblique sections through the same tissue disagree where they cross. | **Exact by construction** (shared 3D noise field *and* a data-frame conditioning pathway) — asserted **bitwise on an untrained model** at T07. The "additionally trained for (`L_cross`)" this row used to claim was dropped: it is unnecessary, and measured harmful (§3.1) |
| Thickness | Post-hoc slab selection; a thicker slab is just more points | A modelled observation operator, with coarse-graining consistency |
| Sectioning during training | Not used | The central self-supervision signal |

**The falsifiable experiment that establishes this (E5, add to v23 §7).** Generate two oblique
sections intersecting inside the tissue, from both methods. Measure agreement along the intersection
line (expression correlation, cell-type concordance, density agreement). Predict: v23 near-perfect
by construction; SpatialZ substantially below, with the gap widening at larger dihedral angles.
This is a clean, single-panel figure that demonstrates a categorical rather than incremental
difference, and it costs almost nothing to run.

---

## 5. New validations enabled

- **V1 — Virtual re-sectioning cycle.** From a coronally-sectioned volume, generate a full sagittal
  stack; treat that stack as input and regenerate the original coronal sections; compare to the real
  ones. End-to-end, ground-truthed, and impossible to pass by memorisation.
- **V2 — Orthogonal-tissue validation.** Train on a coronally-sectioned specimen, generate sagittal
  sections, compare against a *different* specimen actually sectioned sagittally (distribution-level
  comparison — Sinkhorn on cell-state distributions, laminar profile agreement, cell-type
  localisation — not per-cell, since specimens differ).
- **V3 — Anisotropy prediction.** Predict how in-plane Moran's I varies with section angle from the
  fitted 3D covariance, and verify against real sections cut at different angles. Confirms the model
  learned genuine 3D structure rather than a stack of 2D fits. This is the *right* use of the
  quantities in the equivariant column of §1.1.
- **V4 — Thickness transfer.** Train on thin sections, predict thick-section (spot-level) data, and
  vice versa. Validates `L_thick` and supports the cross-technology harmonisation claim.

---

## 6. Risks specific to SEFL

| Risk | Severity | Mitigation |
|---|---|---|
| Consistency losses collapse to a smooth/constant field | **High** | EMA teacher + stop-gradient; reconstruction terms dominant; warm-up schedule; monitor field entropy and per-gene variance as collapse alarms. |
| Forcing invariance on the equivariant column flattens real anisotropy and *lowers* Moran's/Geary's agreement | **High** | §1.1 table is normative. Constrain only conditional programs and physical-point identity. Ablate: a variant that (wrongly) matches in-plane statistics across angles should be shown to be worse — a useful negative result for the paper. |
| Axis-aligned triplane makes oblique worse than coronal | High | §2.1 (a)+(b); benchmark explicitly stratified by dihedral angle, not just averaged. |
| `L_thick` mis-implemented as per-cell correspondence | Medium | Aggregate at the distribution/bin level; counts add. Unit-test on synthetic data where the ground-truth aggregation is known. |
| Compute — every step now generates 2–3 sections | Medium | Sample small patches of each plane (not full sections) for the consistency terms; `n_L ≈ 256` intersection points is ample. Apply SEFL on a subset of steps (e.g. every 3rd). |
| Reviewer: "isn't this just augmentation?" | Low | It is not — augmentation perturbs *inputs*; SEFL constrains *outputs of different observation operators on a shared latent object*, and is validated by V1/V3, which augmentation cannot address. Say so directly. |

---

## 7. Effect on the build order

Insert after v23 §10 step 3, before the metric-aware losses:

- **Step 3.5 — SEFL (1.5 weeks).** (i) Replace the 2D correlated prior with the 3D RFF noise field;
  verify intersection consistency is exact with conditioning frozen. (ii) Add rotation augmentation
  + multi-orientation planes; verify oblique-vs-coronal reconstruction parity on held-in sections.
  (iii) ~~Add `L_cross` with EMA teacher; watch for collapse.~~ **Done at T07 and reverted on the
  evidence** — the collapse arrives, through the field rather than the decoder, and the loss it
  comes from is redundant with (i). (iv) Add `L_thick`, `L_prog`.
- **Gate:** oblique reconstruction quality must reach ≥90% of coronal on held-in sections before
  proceeding. If it does not, the backbone is not equivariant enough and option (c) — a steerable
  backbone — becomes necessary. Better to learn this in week 4 than in month 3.

---

## 8. What this does to the paper

It converts the contribution from "a better interpolator" to "a **tissue observation model**": the
claim becomes that a tissue has one 3D molecular field, that sectioning is an operator on it, and
that requiring all operators to be mutually consistent is enough to learn the field from very few
observations. That framing carries a high-impact venue in a way that benchmark deltas alone do not,
and it makes the small number of real sections a *feature* of the story rather than a limitation to
be defended.

Revised title candidate: *Sectioning-consistent neural fields learn the three-dimensional molecular
organisation of tissue from a handful of sections.*

---

### ⚠️ Amendment, 2026-08-20 — what this framing may and may not assert

This addendum is written from, and read alongside, `v23_design.md`, whose §7 target sentence has
been amended from measured evidence. Three constraints carry into any paper text written from this
document:

1. **No wide-gap advantage may be asserted.** `v23_design.md` §7 previously stated the story as
   "ties or wins narrow, wins decisively wide". The prior campaign contradicts it on the metrics
   the claim is about — by median, per wide holdout, SpatialZ beats v20 on Moran's in **6 of 7**
   and Geary's in **5 of 7** — and the tier-1 case has **never been measured**. It is an open
   question (`specs/10` §11.1, criterion C2), not a result. This matters here specifically because
   the wide gap is where the **shared 3D field** is supposed to pay off, so it is this document's
   claim that is under test.

2. **§4's E5 prediction is an expectation, and T09 measured half of it above its own ceiling.**
   "v23 near-perfect by construction; SpatialZ substantially below, with the gap widening at larger
   dihedral angles" is the hypothesis. Measured: concordance **0.814** against a ceiling of 0.781,
   but expression correlation **0.724** against a measured ceiling of **0.726** — two independent
   draws of *one* plane under one realisation reach only 0.726, so the literal 0.85 criterion was
   unsatisfiable and is now a strict xfail, with the headline test asserting correlation
   **ceiling-relative** (SPEC_QUESTIONS C27). The *categorical* half of the claim — exact
   intersection consistency by construction, asserted bitwise on an untrained model — stands and is
   the stronger result. Do not quote the incremental half without its ceiling.

3. **`L_cross` is not part of the shipped method, and the reason is a stronger claim than the
   loss.** §3.1's intersection-consistency loss ships at `w_cross = 0`: consistency is exact **by
   construction** in v25, so the loss is redundant, and training it flattens the anatomical field
   (generated per-gene variance **0.067** against **0.711** with SEFL off). Any paper text
   describing "three new losses" must say that the third is unnecessary and why — the
   by-construction result belongs beside E5 as a contribution, not buried as an ablation. Whether
   `L_thick` and `L_prog` are used at all is decided by A7 and is **unverified** until it runs.
