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
| Layout: per-type intensity field, Poisson NLL | T05 | the fitted intensity's acceptance number is measured with a **reduced spatial basis**; at the default `fourier_bands_xy = 8` the Poisson MLE overfits the point pattern and T05 specifies no regulariser — open item owed to **T06** (B10) |
| Hard-core radius `r0` at the **1st** percentile (5th selectable, recorded) | T05 | B6; `Config.repulsion_r0_percentile` |
| Strauss/hard-core repulsion fitted to `g(r)` | T05 | ablation A4; the `g(r)` comparison runs over **`[0, 3R]`**, amended at T05 — over the spec's original `[r0, 3R]` a pure-Poisson layout is indistinguishable from the full model (0.070 against 0.093) because the correlation hole ends at `r0` by construction, so A4 would report a false null (B12; `specs/10` §4 amended too) |
| Potts mark smoothing, `beta` fitted not set | T05 | update rule is **Gibbs**, not the spec's ICM, which erases a 2% cell type at any coupling; ICM kept as the asserted negative control (B11). Cell-type localization is reported against **two** references (held-out self-score and flanking section) and the criterion is proposed but **undecided** (B15) |
| `layout_mode` gate (field/hybrid/resample) | T05, T01 | `resample` = previous version |
| Conditional flow matching in cell latent | T06 | straight-line path, Heun; `h1` detached in the CFM loss, `cfm_sigma_min` as the conditional path's width |
| Gene-conditioned ZINB decoder (open-vocabulary) | T06 | bilinear `h ⊙ A e_g`; the **encoder** is open-vocabulary too (a set encoder over `(expression, e_g)` pairs — T06 §2's fixed-width `Enc` would tie the latent to one panel). `Config.decoder_mu_link` keeps the spec's softplus as the default with `exp` selectable and measured |
| Shared latent → gene–gene covariance preserved | T06 | tested vs. the independent-donor baseline — **criterion amended at T06 (B16)**: the spec's "Frobenius error < 50 % of the baseline's" is **below the achievable ceiling** (5.60 measured against a baseline at 7.78, so it asks for < 3.89) and is unpassable by any model. Replaced by (a) the chimerism mechanism isolated with the donors held fixed — which **confirms** the argument, 22 % of the covariance magnitude lost at `D = 3`, monotone through `D = 10` — (b) every arm reported against the measured ceiling, (c) the original criterion held as a strict xfail at 1.20. Retained-magnitude ratio **0.458** (better by 2.2×) at equal pattern fidelity on the default holdout — but **0.995 at `consecutive-3`**, and the decomposition was chosen after seeing which component passed, so the *model-versus-baseline* claim is **not established** by T06. The mechanism (a) and the unsatisfiability (b) are |
| Sample counts, never emit `mu` | T06 | `assert_detection_rate` in the generation path, band `Config.detection_rate_tol`. Detection MAD is **gap-dependent** (0.019 at 50 µm, 0.056 at 100 µm — B17); the acceptance test runs on the default `alternating` holdout |
| Gene subsampling `genes_per_step` | T06 | what makes panel width irrelevant |
| **Trainer**: AdamW, cosine schedule, gradient clipping, EMA teacher | T06 §5 | `train_ctfflow`; `Config.grad_clip`, `lr_min_frac`, `weight_decay`, `log_every`. `forward_train` returns unweighted named terms plus `diag_`-prefixed diagnostics the trainer never weights |
| **Layout intensity basis tied to the fitted length-scale** | T06 (owed by T05, B10) | `fourier_bands_for_lengthscale`; `Config.intensity_basis_ell_multiple`. Recovered r at 300/1200 steps: derived basis 0.979/0.861 against the default 8 bands' 0.835/0.527, i.e. a decay 2.6× smaller. Does **not** abolish the drift — see open risk R4 |
| **Expression head overfits the likelihood** | T06 measurement, owed to T08/T09; **T08 did not close it** (covariance 11.02 vs a 7.73 baseline, 13.39 vs 11.38 — the claim is downgraded in `specs/10` §2). The enable decision is now **T09 §3's joint selection gate** (`train_steps` × metric weights, on internal LOSO), reported by T10's A2 | open risk **R4**: 1200 → 2400 steps lowers the NLL and worsens every distributional statistic of the generated section. The terms that would stop it are T08's (`w_autocorr`, `w_profile`, `w_distribution`) and T09 §2's calibrators. **`TRAIN_STEPS = 1200` is itself the symptom**, so T09 selects the budget rather than inheriting it |
| Metric-aware LOSO losses (Moran, Geary, profiles, Sinkhorn) | T08 | leakage enforced by type; **principal axis on `TrainingVolume`**, not `Volume` (C10). Built and measured; **all three weights start at 0** because at T06's 1200-step budget the terms cost on every metric they are made of (Moran's MAE 0.0287 -> 0.0408, depth r 0.978 -> 0.967, Frobenius 9.00 -> 11.15) — but the ordering **reverses at 2400 steps** on four of six statistics. A 0 calibrated at 1200 steps is calibrated to an undertrained model, so it is **not hardcoded**: the weights and `train_steps` become **one joint selection gate** at `specs/09` §3, chosen per dataset on internal LOSO, and **A2 is an addition experiment run at two budgets** (`specs/10` §4). `specs/08`'s comparison of a model against a measurement is under-specified and all three naive readings diverge (C24). `loss_profile` gains keyword-only cell types (C21); the divergence is the in-repo Sinkhorn, never `geomloss` (C20) |
| Uncertainty-gated anchoring (replaces alpha/gap flags) | T09 | isotonic `w(v)` |
| Leakage-free length-scale + detection calibration | T09 | flanking sections only; **unimodal objective** — bracket capped, maximum located, `target_unreachable` status (T03/GATE 1); one **global** `ell`, per-module agreement a diagnostic only (A2) |
| **Mean–variance calibration** (`log theta` per gene, beside `pi`) | T09 §2 | design §3.5; `DetectionCalibration` carries both (D-table) |
| **v20 Bernoulli cross-mix** (`expr_mode="cross-mix"`) | T06 §4b | design §5/§6; behaviour pinned by `test_cross_mix_matches_v20` (A6) |
| Automatic per-dataset config selection | T09 | coordinate descent, ~10 fits — **plus one joint gate that is not coordinate-descended**: `train_steps` × the metric-aware weights, all four cells of `{1x, 2x} x {off, spec weights}` scored together, each at its own budget (added at T08; visiting them separately would pick "weights off" from a 1200-step incumbent and never reach the cell that wins). `Config.train_steps` is the persisted budget |
| No-regression guarantee | T09, T01 | `test_selector_can_recover_v20_config` |
| Six target metrics, **vendored verbatim from `bench3/evaluate_paper.py`** | T10 | content hash pinned, bitwise agreement asserted; v20's two bugs are a footnote about v20's own tuning signal, not a fix to the benchmark (A3). **Each metric additionally needs its achievable ceiling measured** on the synthetic fixture from an independent draw of the known generative law, with method / ablation / baseline numbers reported relative to it — added at T05, which measured `celltype_localization`'s ceiling at 0.72 against a self-score of 0.92, and per-type because the metric is unstable for abundant tissue-wide types (B15, B15a) |
| Unoptimised control metrics | T10 | paper-integrity requirement |
| Baselines incl. SpatialZ at published defaults | T10 | deep-copy guard; **v14/v18 dropped explicitly** with the reason in the methods (D-table) |
| **Dataset requirement**: >= 1 non-brain, >= 1 non-transcriptomic panel | T10 §3 | design §7; the harness refuses a headline table without both (D-table) |
| Ablations A1–A6 | T10 | gates in `Config` |
| Capability experiments E1–E4 | T10 | zero-shot, cross-panel, oblique, throughput; **E1 reports both arms** (`r_g = 0` and `r_g = psi(t_g)`) (D-table), **and both summary arms** (`summary_source == "native"` vs the orthologue fallback — T06, SPEC_QUESTIONS B20) |

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
| `L_cross` intersection consistency | T07 | decoder parameters matched directly (L2 on `log mu`, `log theta`, `pi` logit) — **no KL surrogate** (C2). Two of the four terms are computed in log/logit coordinates rather than the spec's raw ones (SPEC_QUESTIONS C17): `lambda` because its squared difference is ~1e-10 at any real intensity, and the type term as `KL(p2 || p1)` — the same gradient as the spec's CE, without the teacher-entropy floor that would make the 60 % fall unreachable. The branches differ by **pose** (`plane_pose`), which is the only channel T04 leaves pose-dependent |
| **EMA teacher + stop-gradient (anti-collapse)** | T07 | disabled-teacher test must fail. A separate module (`EMATeacher`), not T06's `EMA.swap_in`: swapping copies into the *live* parameters, and a parameter mutated between a forward and its backward trips autograd's version counter |
| Collapse alarm on per-gene variance | T07 | |
| `L_thick` coarse-graining; counts add, no per-cell matching | T07 | |
| `L_prog` conditioned on (cell type, region) | T07 | unconditional variant tested as wrong |
| Invariant vs. equivariant table as a code constant | T07 | `INVARIANT_QUANTITIES` |
| `loss_prog_WRONG` negative control | T07 | ablation A8 |
| SEFL warm-up + ramp; consistency must not dominate | T07 | ratio logged, warned |
| SEFL cost control (every-N steps, patches) | T07, T01 | `sefl_*` config. `specs/07`'s own two requirements collide: at the full panel the block costs **+62 %** against its own **< 60 %** cap, so the losses take a separate, smaller gene budget (`sefl_genes_per_step = 64`, **+34 %** measured) — every SEFL term is a mean or a covariance over genes and the subsample is redrawn every step (C18) |
| `Section.thickness` as a first-class field | T01 | assumed-value warning |
| Slab-volume intensity integral (not area) | T05 | makes `L_thick` coherent |
| One GRF realisation per stack → mutual coherence | T09 | `generate_stack/oblique/curved` |
| E5 intersection agreement vs. the competing method | T10 | the structural-difference figure |
| **V1 re-sectioning cycle** | T10 §5b | |
| **V2 orthogonal-specimen validation** | T10 §5b | overlaps E3 |
| **V3 anisotropy prediction** | T10 §5b | the equivariant-column payoff |
| **V4 thickness transfer** | T10 §5b | validates `L_thick` |
| Ablation A7 (SEFL off) | T10 | **Inverted at T07: SEFL ships off, so A7 is an *addition* experiment** (`w_thick = w_prog = 0.2` against the shipped default). `w_cross` is 0 because intersection consistency is exact by construction in v25 and training the loss flattens the field (C19, R6); `w_thick` / `w_prog` are 0 because at their spec weights a model trained at T06's own budget fails three T06 acceptance tests (R7). SEFL's net contribution is **unverified** until A7 reports the six target metrics |

## Gates, restated

| Gate | Spec | Criterion | If it fails |
|---|---|---|---|
| GATE 1 | T03 | GRF prior halves Moran's I error vs. i.i.d.; `I_gen` monotone in `ell` **over the calibration bracket** and unimodal with its maximiser at or above the fitted `ell` (G1.3g); measured on the 3000 µm gate fixture | **Stop.** The method's core mechanism does not work; report before building anything else. |
| GATE 2 | T04 | **depth-matched** oblique R² ≥ 0.90 (G2.1a) **and** the interior-only check (G2.1b), on equal-`n` sets with own-section retrieval excluded; plus permanent G2.1h (augmentation complete, by mutation) and G2.1i (draw-noise floor) | Raise orientations to 8, verify augmentation covers everything *by mutation*, check the shortfall exceeds the G2.1i floor, then **stop** — a steerable backbone is a design change. **Ran at T04: 8 orientations bought +0.00086, all channels verified wired.** |

## Deliberate negative controls

Two things are built *to fail*, and their failure is a result reported in the paper:

1. `loss_prog_WRONG` (T07) — constrains in-plane Moran's I across angles, i.e. wrongly treats an
   equivariant quantity as invariant. Trained as ablation A8; expected to be worse.
2. Independent-donor sampler (T06, T10) — `eval/baselines.py`, reimplements the competing method's
   per-gene independent draw. Used as the reference point for the gene–gene covariance claim. It
   shares its mechanism with the v20 cross-mix deliberately: "pick a donor per gene and take its real
   count" is one operation, and the two methods differ in the *weights* (v20 mixes two donors at a
   weight that is zero at narrow gaps; the competing method mixes ≤ 3 uniformly, always).

Neither is dead code. Do not delete them as unused.
