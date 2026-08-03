# SpatialCPA-v16 — Spectral Tissue Flow

v16 generates a virtual tissue section as a **signal on a graph**, and performs
conditional flow matching directly on that graph's **spectral coefficients**.
The per-gene spatial-autocorrelation profile — the quantity two of the
benchmark's five headline metrics measure — stops being something a generative
model might happen to reproduce and becomes a quantity that is written down,
targeted, and checked before anything is written out.

It is an enhancement of **v14** (`H3D_FLA`): same lineage — conditional flow
matching, gap-aware training, z-marginalization — but the state the ODE
transports changes from a *per-cell latent* to a *whole-section spectrum*, and a
calibration stage is added that v14 has no analogue of.

---

## The identity the method is built on

With the row-standardized kNN weights the evaluator uses, for a mean-centred
per-cell signal `z` on a section:

```
Moran's I(z) = zᵀ S z / zᵀ z            S = symmetrized row-standardized kNN
Geary's  C(z) ≈ (n-1)/n · (1 − I(z))
```

Diagonalize `S = Φ Λ Φᵀ`. Writing `ẑ = Φᵀz` — the signal's **graph Fourier
coefficients** — gives

```
Moran's I(z) = Σ_m λ_m ẑ_m²  /  Σ_m ẑ_m²
```

Moran's I *is* the mean harmonic eigenvalue under the signal's normalized
spectral energy. Nothing is approximated. So:

> **Reproduce a gene's normalized graph power spectrum and you reproduce its
> Moran's I exactly — and, through the affine relation, its Geary's C.**

Two consequences, both stated because they cut against the method as much as for
it:

* **`paper_morans_pearson` and `paper_gearys_pearson` are largely redundant on
  this weight matrix.** The repo's own selftest shows it across four regimes:
  oracle 1.000/1.000, flanking_copy 0.975/0.976, spatial_scramble 0.271/0.282,
  random −0.062/−0.058 — the pair never separates by more than 0.011. One
  mechanism addresses both, and a method claiming to win them independently is
  claiming something the metric cannot show.
* **The identity is exact only over a complete basis.** v16 models `M = 64`
  harmonics out of `n` cells, so the restricted form is badly optimistic for a
  gene whose variance is mostly high-frequency: a pure-noise gene projected onto
  48 smooth modes *looks* smooth. The method therefore carries an explicit
  **residual bucket** with its own measured effective eigenvalue, and reports the
  complete form
  `I = (Σ λ_m ẑ_m² + λ_res·V_res) / (Σ ẑ_m² + V_res)`.
  *Verified: on synthetic signals spanning smooth / mid-frequency / pure noise,
  predicted and measured Moran's I agree to 4 decimal places (0.9818/0.9818,
  −0.0081/−0.0081, 0.7831/0.7831).* Getting the bound wrong here is not
  cosmetic — clipping `λ_res` to the modelled range instead of the operator's
  full [−1, 1] range predicts +0.61 for a gene whose true value is −0.01.

---

## The seven stages

| # | stage | what happens |
|---|---|---|
| 1 | **geometry** | the outline is the zero level set of a z-blended **signed distance function** of the two bracketing sections; interior intensity is their z-interpolated density; positions are drawn by stratified sampling |
| 2 | **basis** | tissue harmonics `(Φ*, λ*)` of the generated point set — the graph the section's signals live on |
| 3 | **flow** | conditional flow matching over **harmonic tokens**: one token per mode, carrying that mode's loading on every gene component |
| 4 | **inverse** | spectrum → smooth per-cell expression **and** cell-type indicator fields |
| 5 | **calibration** | the realized per-gene spectral profile is forced onto its z-interpolated target |
| 6 | **residual** | high-frequency dispersion restored as **whole-cell** draws from real training cells of the assigned type |
| 7 | **typing** | the generated type posterior is sampled under a composition constraint |

### Why the section, not the cell, is the unit of generation

v14 flow-matches a per-cell joint latent: every cell is generated from its own
noise draw given an attention context. The section's *collective* spatial
structure — which is what four of the five target metrics read — is never a
variable the model controls. It can only emerge, and when it does not there is no
handle to reach for.

In v16 the ODE transports the whole section's coefficient matrix `Ŷ` (M
harmonics × K gene components). Generating a section is one trajectory in that
space, so spatial structure is produced coherently rather than assembled from
independent per-cell draws.

Two design points follow from what the tokens are:

* **The eigenvalue is the positional encoding.** A harmonic's identity is its
  spatial frequency, not its index — mode 7 of one section and mode 7 of another
  are not the same object, but two modes at λ ≈ 0.9 are both "smooth along the
  dominant axis". Conditioning on Fourier features of λ is what lets one model
  serve sections with different cell counts and different graphs, and it is why
  the model transfers across the z gap at all.
* **Attention runs across harmonics.** Low modes set the anatomy that high modes
  detail. M is 32–96 tokens, so this is cheap — no per-cell attention anywhere.

### Why calibration is separate from generation

Calibration acts per (frequency bin, gene): within a bin the *shape* the
generator produced is kept and only the bin's total energy is set. So the
generative model decides **where** the structure is; the calibration decides
**how much** there is at each spatial scale. Keeping those separate is what lets
the autocorrelation guarantee hold without the calibration dictating the pattern.

### Why residuals are drawn as whole cells

Independent per-gene noise would fix each gene's variance while destroying
gene–gene covariance. The benchmark reads that directly (co-expression
agreement) and indirectly (`paper_umap_mixing`): a cloud of cells with the right
marginals and no covariance structure does not overlay the real one. Drawing a
*row* of the residual pool hands the virtual cell a real cell's joint deviation
across all genes at once.

---

## What is deliberately absent

**Nothing from v8.** No optimal transport, no coupling matrix, no transport plan,
no McCann/barycentric displacement interpolation, no diffeomorphic or smoothed-OT
morph, no velocity-field advection of an anchor slice, no niche Markov-random-field
refinement. The two bracketing sections meet only as **fields** — densities and
spectra — and never as point sets in correspondence. No cell in one section is
ever matched to a cell in another.

**Nothing from SpatialZ.** No alpha-blended interpolation between two flanking
slices, no MENDER niche/domain labelling, no flanking-slice PCA. v16 never copies,
blends or resamples a flanking slice's cells: every position is new, the cell
count is emergent from the intensity field, and expression is synthesized in the
spectral domain.

The one thing v16 takes from real training cells is the **high-frequency
residual** — a dispersion vector, drawn by cell type, added on top of a
synthesized smooth field. That is a variance model, not a copy: it carries no
position, no identity and no low-frequency structure, and stage 5 has already
fixed the spatial autocorrelation before it is added.

---

## How each target metric is addressed

| metric | mechanism |
|---|---|
| `paper_morans_pearson` | stage 5 — per-gene spectral profile forced onto the z-interpolated target; Moran's I is that profile |
| `paper_gearys_pearson` | the same mechanism; C is near-affine in I on these weights |
| `paper_umap_mixing` | stage 6 — whole-cell residual draws restore both single-cell dispersion and gene–gene covariance, so the generated cloud has the real one's local geometry rather than a shrunken version of it |
| `paper_marker_depth_r` | stages 1–3 — the low harmonics *are* the tissue's dominant anatomical axes, so the depth profile is carried by the modes the flow models most accurately; stage 1 puts them on the right support |
| `paper_celltype_localization` | stage 7 — type fields are generated spectrally (they are low-frequency signals) and sampled under a composition constraint, so a type can never be silently dropped, which is the metric's harshest penalty (an absent type scores 0) |

---

## Ablations

Every stage can be removed from the command line, which is how the contribution
is separated from the scaffolding:

```bash
--no-calibration        # stage 5 off: does the spectral guarantee do anything?
--residual-pooled       # stage 6 draws tissue-wide instead of per type
--no-composition-match  # stage 7 samples freely
--residual-scale 0      # smooth field only — the over-smoothing control
--n-modes 16|64|256     # where structure ends and dispersion begins
```

`--residual-scale 0` is the important one: it is the deliberate over-smoothing
control, and it should show inflated Moran's I, deflated Geary's C and collapsed
embedding mixing. A method whose ablations do not move the metrics they are
supposed to move has not demonstrated its mechanism.

---

## Status

See `validation/VALIDATION.md` for measured numbers, what was run, and what was
not. **No benchmark numbers are fabricated**, and in particular SpatialZ itself
has not been run in the environment this was developed in — see that file for
exactly what the comparison does and does not establish.

## Running it

Registered in `benchmark-pbya-v3` as `spatialcpav16_gen`:

```bash
cd benchmark-pbya-v3
python -m src.bench3.run_all --methods spatialcpav16_gen --dataset starmap_visual_cortex
```

The wrapper lives at `benchmark-pbya-v3/src/bench3/methods/run_spatialcpav16.py`
rather than beside the others in `benchmark-pbya-v2`, because v2 is frozen. It
speaks the identical `_v2_io` contract, so `run_benchmark` invokes it the same
way and the evaluator reads it unchanged.

Requires `torch >= 2.0`, numpy/scipy/scikit-learn/anndata/scanpy. Without torch
the method degrades to the depth-interpolated spectrum and says so; v3 is
configured to **fail** that run rather than score it, because the fallback is not
the method.
