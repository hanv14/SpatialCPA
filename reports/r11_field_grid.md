# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps, `layout_sampler=grid`, seed 1. `layout_mode` and
`layout_sampler` are both generation-time gates, so every arm shares one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `field-grid` |
|---|---|
| `paper_morans_pearson` | +0.6564 |
| `paper_gearys_pearson` | +0.6588 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6006 |
| `paper_marker_depth_r` | +0.7925 |
| `paper_celltype_localization` | +0.6607 |
| `paper_gene_mean_spearman` | +0.9803 |
| `paper_cell_count_ratio` (raw pass) | 5.362 |

Per section, on the two metrics R11 turns on — `celltype_localization` matched,
`cell_count_ratio` raw:

| arm | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `field-grid` localization | +0.6634 | +0.4437 | +0.6607 | +0.6607 |
| `field-grid` count ratio | 63.904 | 5.362 | 0.895 | 5.362 |

As emitted, without the density control:

| metric | `field-grid` |
|---|---|
| `paper_morans_pearson` | +0.6445 |
| `paper_gearys_pearson` | +0.6440 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6006 |
| `paper_marker_depth_r` | +0.8240 |
| `paper_celltype_localization` | +0.6268 |
| `paper_gene_mean_spearman` | +0.9770 |

Emitted cell counts (generated/ground truth):

* `field-grid`: section_2=267567/4187, section_4=21993/4102, section_6=3727/4162
