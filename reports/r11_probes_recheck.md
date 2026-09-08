# R11 re-measured — `layout_mode` on STARmap tier 1, grid sampler

One set of weights (`runs/pilot/model_exp_2400.pt`), `decoder_mu_link=exp`, 2400 steps, seed 1. `layout_mode` and `layout_sampler` are generation-time gates
(`CTFFlow.check_generation_cfg`), so no arm below needed a fit and nothing but the
layout varies between them. Same instrument, same three held-out sections and the same
ground-truth-matched density as `reports/pilot.md` §13, so the columns are comparable.

`celltype_localization` is scored at ground-truth-matched density — each section
subsampled to its own true cell count, because a denser point set puts kNN neighbours
closer and inflates every graph-based metric. `cell_count_ratio` is from the raw pass,
where it means something. Medians are over the three held-out sections
(`specs/10` §4.6), never means.

## Headline — the two metrics R11 turns on

| arm | section_2 | section_4 | section_6 | **median** |
|---|---|---|---|---|
| **celltype_localization** | | | | |
| `oracle` — ceiling | +0.9765 | +0.9888 | +0.9808 | **+0.9808** |
| `flanking_copy` — copy floor | +0.7008 | +0.7765 | +0.7868 | **+0.7765** |
| `layout_mode=resample` (`grid`) | +0.7008 | +0.7546 | +0.7868 | **+0.7546** |
| **cell_count_ratio** (raw pass) | | | | |
| `oracle` — ceiling | 1.000 | 1.000 | 1.000 | **1.000** |
| `flanking_copy` — copy floor | 0.973 | 1.016 | 0.988 | **0.988** |
| `layout_mode=resample` (`grid`) | 0.973 | 1.016 | 0.988 | **0.988** |

Emitted cell counts (generated/ground truth):

* `resample-grid`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162

## The six target metrics, medians over sections

| metric | `resample-grid` | `flanking_copy` | `oracle` |
|---|---|---|---|
| `paper_celltype_localization` | +0.7546 | +0.7765 | +0.9808 |
| `paper_marker_field_r` | +0.6824 | +0.8857 | +0.9997 |
| `paper_marker_depth_r` | +0.8331 | +0.9794 | +1.0000 |
| `paper_morans_pearson` | +0.6541 | +0.9836 | +1.0000 |
| `paper_gearys_pearson` | +0.6535 | +0.9840 | +1.0000 |
| `paper_gene_mean_spearman` | +0.9901 | +0.9863 | +1.0000 |
| `paper_cell_count_ratio` (raw) | 0.988 | 0.988 | 1.000 |

## What varies between the arms

| arm | `layout_mode` | `layout_sampler` | weights | seed |
|---|---|---|---|---|
| `resample-grid` | resample | grid | `model_exp_2400.pt` | 1 |

One seed. `reports/envelope_synthetic.md` measured the across-seed envelope at **0.0335** on the
synthetic fixture, and `specs/10` §3's repeated-seed rule wants `claim_min_seeds` = 3 before a
claim rests on any of this: a difference below that envelope is not a difference.
