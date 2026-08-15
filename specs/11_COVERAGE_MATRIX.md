# Coverage matrix — design → specs

Every component of `v23_design.md` and `v23_sectioning_equivariance.md` (SEFL), mapped to the task
file that implements it. **If you find something in either design doc that is not in this table, it
is an omission — flag it rather than skipping it.**

## From `v23_design.md` (base design, = v25 core)

| Design component | Spec | Notes |
|---|---|---|
| Observation token: gene emb + context emb + Fourier(xyz) + expression | T02, T04, T06 | assembled in `spatialcpav25_gen.py` |
| MedCPT gene/celltype/region descriptors + cache | T02 | frozen encoder |
| Free residual `r_g` + gamma anneal + text→residual distillation | T02 | zeros init, anneal order matters |
| Anisotropic Fourier encoding (B_xy=8, B_z=2) | T04 | axis-order bug flagged |
| Triplane anatomical field + TV_z | T04 | |
| Retrieval cross-attention, density-adaptive niche | T04 | |
| **z-proximity term in retrieval** (the competing method's omission) | T04 | `retrieval_w_z`, ablation A5 |
| Gap-aware section dropout curriculum | T04 | `section_dropout_p` |
| Layout: per-type intensity field, Poisson NLL | T05 | |
| Hard-core radius `r0` at the **1st** percentile (5th selectable, recorded) | T05 | B6; `Config.repulsion_r0_percentile` |
| Strauss/hard-core repulsion fitted to `g(r)` | T05 | ablation A4 |
| Potts mark smoothing, `beta` fitted not set | T05 | |
| `layout_mode` gate (field/hybrid/resample) | T05, T01 | `resample` = previous version |
| Conditional flow matching in cell latent | T06 | straight-line path, Heun |
| Gene-conditioned ZINB decoder (open-vocabulary) | T06 | bilinear `h ⊙ A e_g` |
| Shared latent → gene–gene covariance preserved | T06 | tested vs. independent-donor baseline |
| Sample counts, never emit `mu` | T06 | assertion in generation path |
| Gene subsampling `genes_per_step` | T06 | what makes panel width irrelevant |
| Metric-aware LOSO losses (Moran, Geary, profiles, Sinkhorn) | T08 | leakage enforced by type; **principal axis on `TrainingVolume`**, not `Volume` (C10) |
| Uncertainty-gated anchoring (replaces alpha/gap flags) | T09 | isotonic `w(v)` |
| Leakage-free length-scale + detection calibration | T09 | flanking sections only; **unimodal objective** — bracket capped, maximum located, `target_unreachable` status (T03/GATE 1); one **global** `ell`, per-module agreement a diagnostic only (A2) |
| **Mean–variance calibration** (`log theta` per gene, beside `pi`) | T09 §2 | design §3.5; `DetectionCalibration` carries both (D-table) |
| **v20 Bernoulli cross-mix** (`expr_mode="cross-mix"`) | T06 §4b | design §5/§6; behaviour pinned by `test_cross_mix_matches_v20` (A6) |
| Automatic per-dataset config selection | T09 | coordinate descent, ~10 fits |
| No-regression guarantee | T09, T01 | `test_selector_can_recover_v20_config` |
| Six target metrics, **vendored verbatim from `bench3/evaluate_paper.py`** | T10 | content hash pinned, bitwise agreement asserted; v20's two bugs are a footnote about v20's own tuning signal, not a fix to the benchmark (A3) |
| Unoptimised control metrics | T10 | paper-integrity requirement |
| Baselines incl. SpatialZ at published defaults | T10 | deep-copy guard; **v14/v18 dropped explicitly** with the reason in the methods (D-table) |
| **Dataset requirement**: >= 1 non-brain, >= 1 non-transcriptomic panel | T10 §3 | design §7; the harness refuses a headline table without both (D-table) |
| Ablations A1–A6 | T10 | gates in `Config` |
| Capability experiments E1–E4 | T10 | zero-shot, cross-panel, oblique, throughput; **E1 reports both arms** (`r_g = 0` and `r_g = psi(t_g)`) (D-table) |

## From `v23_sectioning_equivariance.md` (SEFL)

| SEFL component | Spec | Notes |
|---|---|---|
| **3D GRF noise field** replacing the per-section 2D prior | T03 | GATE 1; RFF, Matérn |
| Anisotropic `ell = (ℓx, ℓy, ℓz)` fitted from sections | T03 | what makes oblique quantitatively right |
| Continuity in 3D → exact intersection consistency | T03 (G1.2), T07 | bitwise-equality test |
| `with_lengthscale` without redrawing | T03 | calibration loop stability |
| **Unimodal `I_gen(ell)`; calibration bracket + maximum detection** | T03 (G1.3g), T09 §2 | measured in `reports/gate1.md`; `calibration_ell_max_extent_frac`, `calibration_ell_max_fitted_multiple` |
| Rotation augmentation over the whole volume | T04 | `RotationContext` |
| Multi-orientation triplane ensemble | T04 | `n_plane_orientations=4` |
| **Oblique parity ≥ 0.90 × axis-aligned** | T04 | GATE 2; evaluation set = pooled cells within `thickness/2`, **equal `n` across angles** and **own source section excluded from retrieval** (C1). Criterion amended at T04 (C16): **both arms depth-matched** — the 0° arm is the mean over coronal planes at every section — plus a required interior-only check, and a **fixed** R² denominator. Measured 0.955 / 0.979 |
| Fourier low-frequency axis follows the data frame | T04 | the silent-bug test |
| Plane geometry, `intersect`, `random_plane_pair` | T07 | hand-computed test cases |
| Curved / anatomy-following surfaces | T07, T09 | `generate_curved` |
| `L_cross` intersection consistency | T07 | decoder parameters matched directly (L2 on `log mu`, `log theta`, `pi` logit) — **no KL surrogate** (C2) |
| **EMA teacher + stop-gradient (anti-collapse)** | T07 | disabled-teacher test must fail |
| Collapse alarm on per-gene variance | T07 | |
| `L_thick` coarse-graining; counts add, no per-cell matching | T07 | |
| `L_prog` conditioned on (cell type, region) | T07 | unconditional variant tested as wrong |
| Invariant vs. equivariant table as a code constant | T07 | `INVARIANT_QUANTITIES` |
| `loss_prog_WRONG` negative control | T07 | ablation A8 |
| SEFL warm-up + ramp; consistency must not dominate | T07 | ratio logged, warned |
| SEFL cost control (every-N steps, patches) | T07, T01 | `sefl_*` config |
| `Section.thickness` as a first-class field | T01 | assumed-value warning |
| Slab-volume intensity integral (not area) | T05 | makes `L_thick` coherent |
| One GRF realisation per stack → mutual coherence | T09 | `generate_stack/oblique/curved` |
| E5 intersection agreement vs. the competing method | T10 | the structural-difference figure |
| **V1 re-sectioning cycle** | T10 §5b | |
| **V2 orthogonal-specimen validation** | T10 §5b | overlaps E3 |
| **V3 anisotropy prediction** | T10 §5b | the equivariant-column payoff |
| **V4 thickness transfer** | T10 §5b | validates `L_thick` |
| Ablation A7 (SEFL off) | T10 | |

## Gates, restated

| Gate | Spec | Criterion | If it fails |
|---|---|---|---|
| GATE 1 | T03 | GRF prior halves Moran's I error vs. i.i.d.; `I_gen` monotone in `ell` **over the calibration bracket** and unimodal with its maximiser at or above the fitted `ell` (G1.3g); measured on the 3000 µm gate fixture | **Stop.** The method's core mechanism does not work; report before building anything else. |
| GATE 2 | T04 | **depth-matched** oblique R² ≥ 0.90 (G2.1a) **and** the interior-only check (G2.1b), on equal-`n` sets with own-section retrieval excluded; plus permanent G2.1h (augmentation complete, by mutation) and G2.1i (draw-noise floor) | Raise orientations to 8, verify augmentation covers everything *by mutation*, check the shortfall exceeds the G2.1i floor, then **stop** — a steerable backbone is a design change. **Ran at T04: 8 orientations bought +0.00086, all channels verified wired.** |

## Deliberate negative controls

Two things are built *to fail*, and their failure is a result reported in the paper:

1. `loss_prog_WRONG` (T07) — constrains in-plane Moran's I across angles, i.e. wrongly treats an
   equivariant quantity as invariant. Trained as ablation A8; expected to be worse.
2. Independent-donor sampler (T06, T10) — reimplements the competing method's per-gene independent
   draw. Used as the reference point for the gene–gene covariance claim.

Neither is dead code. Do not delete them as unused.
