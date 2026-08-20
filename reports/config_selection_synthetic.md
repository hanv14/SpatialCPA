# Config selection — synthetic

Selected by internal LOSO over **training sections only**, seed 20260819. Folds: synthetic_s01, synthetic_s05, synthetic_s07. No held-out section is reachable from `select_config`: it takes a `TrainingVolume`, which only `split_holdout` produces.

## Selected configuration

| gate | selected |
|---|---|
| `layout_mode` | resample |
| `prior_mode` | correlated |
| `expr_mode` | zinb-flow |
| `text_emb_mode` | lookup |
| `train_steps` | 2400 |
| `w_autocorr` | 0.5 |
| `w_profile` | 0.5 |
| `w_distribution` | 0.5 |
| config hash | `cbabb27aa44ee6e4` |
| persisted at | `config_selection_synthetic.yaml` |

## The joint gate: `train_steps` x metric-aware weights

All four cells, each **fitted at the budget it names**. Reported in full because the gate exists for an interaction, and coordinate descent over these two would pick `weights off` from a 1x incumbent and never reach the cell that wins (`specs/09` §3).

| cell | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|
| 1x, weights off | 1200 | 0.9197 | 0.8710 | 0.8501 | -0.0527 | 0.0382 | -0.0595 | 3.0 |
| 1x, weights on | 1200 | 0.8841 | 0.8569 | 0.6402 | -0.0369 | 0.0610 | -0.0391 | 2.5 |
| 2x, weights off | 2400 | 0.9316 | 0.9022 | 0.9145 | -0.0519 | 0.0195 | -0.0502 | 2.0 |
| 2x, weights on | 2400 | 0.9448 | 0.9050 | 0.9251 | -0.0455 | 0.0351 | -0.0591 | 1.5 |

## The merged full-budget gate: `layout_mode` x `prior_mode` x `expr_mode`

All 18 cells, every one fitted at the **selected** budget. These three gates are disqualified from reduced-budget scoring by `specs/09` §3's training-free-option rule — each has an option that reaches its final behaviour without training (`resample`, `iid`, `cross-mix`) and is therefore at full strength at any budget while its rivals are not. They are scored jointly rather than one after another because their errors compound through coordinate descent's ordering (open risk R8, `reports/r8_budget_grid.md`).

| cell | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|
| layout_mode=resample, prior_mode=correlated, expr_mode=zinb-flow | 2400 | 0.9606 | 0.9308 | 0.9744 | -0.0437 | 0.0491 | -0.0660 | 3.0 |
| layout_mode=resample, prior_mode=correlated, expr_mode=auto-blend | 2400 | 0.9606 | 0.9308 | 0.9744 | -0.0437 | 0.0491 | -0.0660 | 3.0 |
| layout_mode=hybrid, prior_mode=correlated, expr_mode=zinb-flow | 2400 | 0.9497 | 0.9248 | 0.9400 | -0.0540 | 0.0503 | -0.0532 | 4.2 |
| layout_mode=hybrid, prior_mode=correlated, expr_mode=auto-blend | 2400 | 0.9497 | 0.9248 | 0.9400 | -0.0540 | 0.0503 | -0.0532 | 4.2 |
| layout_mode=field, prior_mode=iid, expr_mode=auto-blend | 2400 | 0.9235 | 0.8847 | 0.8805 | -0.0349 | 0.0543 | -0.0453 | 7.0 |
| layout_mode=field, prior_mode=correlated, expr_mode=zinb-flow | 2400 | 0.9448 | 0.9050 | 0.9251 | -0.0455 | 0.0351 | -0.0591 | 7.5 |
| layout_mode=field, prior_mode=correlated, expr_mode=auto-blend | 2400 | 0.9448 | 0.9050 | 0.9251 | -0.0455 | 0.0351 | -0.0591 | 7.5 |
| layout_mode=hybrid, prior_mode=iid, expr_mode=auto-blend | 2400 | 0.9327 | 0.8924 | 0.8748 | -0.0448 | 0.0674 | -0.0568 | 8.0 |
| layout_mode=resample, prior_mode=iid, expr_mode=zinb-flow | 2400 | 0.9361 | 0.9022 | 0.9010 | -0.0444 | 0.0470 | -0.0660 | 8.0 |
| layout_mode=resample, prior_mode=iid, expr_mode=auto-blend | 2400 | 0.9315 | 0.9026 | 0.9013 | -0.0485 | 0.0659 | -0.0660 | 8.0 |
| layout_mode=hybrid, prior_mode=iid, expr_mode=zinb-flow | 2400 | 0.9289 | 0.8899 | 0.8707 | -0.0553 | 0.0613 | -0.0568 | 10.0 |
| layout_mode=field, prior_mode=iid, expr_mode=zinb-flow | 2400 | 0.9210 | 0.8635 | 0.8677 | -0.0456 | 0.0702 | -0.0453 | 11.5 |
| layout_mode=hybrid, prior_mode=correlated, expr_mode=cross-mix | 2400 | 0.8477 | 0.7771 | 0.6979 | -0.0445 | 0.0120 | -0.0532 | 13.0 |
| layout_mode=hybrid, prior_mode=iid, expr_mode=cross-mix | 2400 | 0.8365 | 0.7591 | 0.7056 | -0.0512 | 0.0231 | -0.0568 | 13.5 |
| layout_mode=resample, prior_mode=correlated, expr_mode=cross-mix | 2400 | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 | 15.0 |
| layout_mode=resample, prior_mode=iid, expr_mode=cross-mix | 2400 | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 | 15.0 |
| layout_mode=field, prior_mode=correlated, expr_mode=cross-mix | 2400 | 0.8339 | 0.7461 | 0.6829 | -0.0472 | 0.0024 | -0.0591 | 16.0 |
| layout_mode=field, prior_mode=iid, expr_mode=cross-mix | 2400 | 0.8238 | 0.7446 | 0.6799 | -0.0537 | 0.0194 | -0.0453 | 17.0 |

## Coordinate descent (the gates every option of which trains)

| gate | candidate | steps | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization | median rank |
|---|---|---|---|---|---|---|---|---|---|
| text_emb_mode | text_emb_mode=medcpt | 600 | 0.5997 | 0.5048 | 0.1649 | 0.0066 | 0.0868 | -0.0660 | 2.0 |
| text_emb_mode | text_emb_mode=lookup | 600 | 0.6523 | 0.5951 | 0.1794 | -0.0202 | 0.0911 | -0.0660 | 1.0 |

Two passes are run; the second re-scored the same two candidates and reached the same answer, so the duplicate rows are collapsed here.

Fits issued: 23 (1200 steps, 1200 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 2400 steps, 600 steps, 600 steps).

The six metrics are computed with T08's kernels rather than `bench3`'s vendored implementations — `eval/metrics.py` is T10's module. The names match, and T10 re-scores the selected config with the vendored code.

## Calibration (leakage-free, flanking training sections only)

| quantity | value | status |
|---|---|---|
| `ell_xy` | 86.4 um | converged |
| `ell_z` | 364.6 um | target_unreachable (bound, not a fit — R1) |
| fitted `ell` | 136.8 / 364.6 um | variogram |
| Moran's I | gen 0.4096 vs flanking 0.4102 | 2 iterations |
| between-section r | gen 0.8859 vs observed 0.9182 | R1 remedy 2 |

Derived `retrieval_z_window` = **3** spacings (largest section gap 100 um). `Config.retrieval_z_window` remains the fallback and the ablation handle.

## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)

One **global** `ell` is calibrated; this table says whether it serves every gene module equally. It is not a target, and a poor table is evidence for the per-channel-group escalation, which is a design change to be decided explicitly.

| module | genes | I_gen | I_real | |diff| |
|---|---|---|---|---|
| 0 | 55 | 0.3639 | 0.3538 | 0.0419 |
| 1 | 55 | 0.4011 | 0.3849 | 0.0456 |
| 2 | 54 | 0.4279 | 0.4175 | 0.0426 |
| 3 | 36 | 0.4462 | 0.4296 | 0.0531 |

*Calibration arm: `prior_mode=correlated` (selected: `correlated`), `train_steps=2400`.*

## Open risk R3 — the boundary is a different regime

Mean latent variance of the uncertainty gate at the stack's ends against its interior, on real cells with each section excluded from its own retrieval pool.

| position | mean latent variance |
|---|---|
| first | 0.872739 |
| middle | 0.822842 |
| last | 0.815149 |

## Definition of done — the selected config against the two fallbacks

| arm | morans_pearson | gearys_pearson | umap_mixing | marker_field_r | marker_depth_r | celltype_localization |
|---|---|---|---|---|---|---|
| method | 0.9517 | 0.9142 | 0.9730 | -0.0425 | 0.0519 | -0.0660 |
| resample | 0.8278 | 0.7683 | 0.6844 | -0.0374 | 0.0195 | -0.0660 |
| donor | 0.8716 | 0.8273 | 0.7388 | -0.0404 | 0.0405 | -0.0660 |

Detection / dispersion calibration fitted on ['synthetic_s00', 'synthetic_s01', 'synthetic_s03']; per-gene detection MAD before it 0.0073.

Anchor weight w(v): [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] at variances [0.56255, 0.6776, 0.74843, 0.80689, 0.86753, 0.93996, 1.02803, 1.25427] — non-increasing by construction.
