# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps, `layout_sampler=grid`, seed 1. `layout_mode` and
`layout_sampler` are both generation-time gates, so every arm shares one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `hybrid-grid` |
|---|---|
| `paper_morans_pearson` | +0.6524 |
| `paper_gearys_pearson` | +0.6544 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6605 |
| `paper_marker_depth_r` | +0.8208 |
| `paper_celltype_localization` | +0.6692 |
| `paper_gene_mean_spearman` | +0.9874 |
| `paper_cell_count_ratio` (raw pass) | 5.362 |

Per section, on the two metrics R11 turns on — `celltype_localization` matched,
`cell_count_ratio` raw:

| arm | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `hybrid-grid` localization | +0.6692 | +0.6600 | +0.6948 | +0.6692 |
| `hybrid-grid` count ratio | 63.904 | 5.362 | 0.895 | 5.362 |

As emitted, without the density control:

| metric | `hybrid-grid` |
|---|---|
| `paper_morans_pearson` | +0.6365 |
| `paper_gearys_pearson` | +0.6395 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.7034 |
| `paper_marker_depth_r` | +0.8419 |
| `paper_celltype_localization` | +0.6948 |
| `paper_gene_mean_spearman` | +0.9874 |

Emitted cell counts (generated/ground truth):

* `hybrid-grid`: section_2=267567/4187, section_4=21993/4102, section_6=3727/4162
