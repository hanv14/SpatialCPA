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
| `layout_mode=field` (`grid`) | +0.6634 | +0.4437 | +0.6607 | **+0.6607** |
| `layout_mode=hybrid` (`grid`) | +0.6692 | +0.6600 | +0.6948 | **+0.6692** |
| `layout_mode=resample` (`grid`) | +0.7008 | +0.7546 | +0.7868 | **+0.7546** |
| `layout_mode=field` (`rejection`) | +0.6008 | +0.0000 | +0.6628 | **+0.6008** |
| `layout_mode=hybrid` (`rejection`) | +0.6534 | +0.3581 | +0.6790 | **+0.6534** |
| **cell_count_ratio** (raw pass) | | | | |
| `oracle` — ceiling | 1.000 | 1.000 | 1.000 | **1.000** |
| `flanking_copy` — copy floor | 0.973 | 1.016 | 0.988 | **0.988** |
| `layout_mode=field` (`grid`) | 63.904 | 5.362 | 0.895 | **5.362** |
| `layout_mode=hybrid` (`grid`) | 63.904 | 5.362 | 0.895 | **5.362** |
| `layout_mode=resample` (`grid`) | 0.973 | 1.016 | 0.988 | **0.988** |
| `layout_mode=field` (`rejection`) | 42.870 | 0.040 | 0.895 | **0.895** |
| `layout_mode=hybrid` (`rejection`) | 42.870 | 0.040 | 0.895 | **0.895** |

Emitted cell counts (generated/ground truth):

* `field-grid`: section_2=267567/4187, section_4=21993/4102, section_6=3727/4162
* `hybrid-grid`: section_2=267567/4187, section_4=21993/4102, section_6=3727/4162
* `resample-grid`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162
* `field-rejection`: section_2=179495/4187, section_4=163/4102, section_6=3727/4162
* `hybrid-rejection`: section_2=179495/4187, section_4=163/4102, section_6=3727/4162

## The six target metrics, medians over sections

| metric | `field-grid` | `hybrid-grid` | `resample-grid` | `field-rejection` | `hybrid-rejection` | `flanking_copy` | `oracle` |
|---|---|---|---|---|---|---|---|
| `paper_celltype_localization` | +0.6607 | +0.6692 | +0.7546 | +0.6008 | +0.6534 | +0.7765 | +0.9808 |
| `paper_marker_field_r` | +0.6006 | +0.6605 | +0.6830 | +0.5995 | +0.6513 | +0.8857 | +0.9997 |
| `paper_marker_depth_r` | +0.7925 | +0.8208 | +0.8554 | +0.8104 | +0.7838 | +0.9794 | +1.0000 |
| `paper_morans_pearson` | +0.6564 | +0.6524 | +0.6465 | +0.6578 | +0.6690 | +0.9836 | +1.0000 |
| `paper_gearys_pearson` | +0.6588 | +0.6544 | +0.6469 | +0.6599 | +0.6671 | +0.9840 | +1.0000 |
| `paper_gene_mean_spearman` | +0.9803 | +0.9874 | +0.9880 | +0.9617 | +0.9880 | +0.9863 | +1.0000 |
| `paper_cell_count_ratio` (raw) | 5.362 | 5.362 | 0.988 | 0.895 | 0.895 | 0.988 | 1.000 |

## What varies between the arms

| arm | `layout_mode` | `layout_sampler` | weights | seed |
|---|---|---|---|---|
| `field-grid` | field | grid | `model_exp_2400.pt` | 1 |
| `hybrid-grid` | hybrid | grid | `model_exp_2400.pt` | 1 |
| `resample-grid` | resample | grid | `model_exp_2400.pt` | 1 |
| `field-rejection` | field | rejection | `model_exp_2400.pt` | 1 |
| `hybrid-rejection` | hybrid | rejection | `model_exp_2400.pt` | 1 |

One seed. `reports/envelope_synthetic.md` measured the across-seed envelope at **0.0335** on the
synthetic fixture, and `specs/10` §3's repeated-seed rule wants `claim_min_seeds` = 3 before a
claim rests on any of this: a difference below that envelope is not a difference.
