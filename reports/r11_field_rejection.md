# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps, `layout_sampler=rejection`, seed 1. `layout_mode` and
`layout_sampler` are both generation-time gates, so every arm shares one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `field-rejection` |
|---|---|
| `paper_morans_pearson` | +0.6578 |
| `paper_gearys_pearson` | +0.6599 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.5995 |
| `paper_marker_depth_r` | +0.8104 |
| `paper_celltype_localization` | +0.6008 |
| `paper_gene_mean_spearman` | +0.9617 |
| `paper_cell_count_ratio` (raw pass) | 0.895 |

Per section, on the two metrics R11 turns on — `celltype_localization` matched,
`cell_count_ratio` raw:

| arm | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `field-rejection` localization | +0.6008 | +0.0000 | +0.6628 | +0.6008 |
| `field-rejection` count ratio | 42.870 | 0.040 | 0.895 | 0.895 |

As emitted, without the density control:

| metric | `field-rejection` |
|---|---|
| `paper_morans_pearson` | +0.6333 |
| `paper_gearys_pearson` | +0.6333 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.5995 |
| `paper_marker_depth_r` | +0.8211 |
| `paper_celltype_localization` | +0.6566 |
| `paper_gene_mean_spearman` | +0.9666 |

Emitted cell counts (generated/ground truth):

* `field-rejection`: section_2=179495/4187, section_4=163/4102, section_6=3727/4162
