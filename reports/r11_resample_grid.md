# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps, `layout_sampler=grid`, seed 1. `layout_mode` and
`layout_sampler` are both generation-time gates, so every arm shares one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `resample-grid` |
|---|---|
| `paper_morans_pearson` | +0.6465 |
| `paper_gearys_pearson` | +0.6469 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6830 |
| `paper_marker_depth_r` | +0.8554 |
| `paper_celltype_localization` | +0.7546 |
| `paper_gene_mean_spearman` | +0.9880 |
| `paper_cell_count_ratio` (raw pass) | 0.988 |

Per section, on the two metrics R11 turns on — `celltype_localization` matched,
`cell_count_ratio` raw:

| arm | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `resample-grid` localization | +0.7008 | +0.7546 | +0.7868 | +0.7546 |
| `resample-grid` count ratio | 0.973 | 1.016 | 0.988 | 0.988 |

As emitted, without the density control:

| metric | `resample-grid` |
|---|---|
| `paper_morans_pearson` | +0.6465 |
| `paper_gearys_pearson` | +0.6469 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6830 |
| `paper_marker_depth_r` | +0.8554 |
| `paper_celltype_localization` | +0.7716 |
| `paper_gene_mean_spearman` | +0.9880 |

Emitted cell counts (generated/ground truth):

* `resample-grid`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162
