# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps, `layout_sampler=rejection`, seed 1. `layout_mode` and
`layout_sampler` are both generation-time gates, so every arm shares one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `hybrid-rejection` |
|---|---|
| `paper_morans_pearson` | +0.6690 |
| `paper_gearys_pearson` | +0.6671 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6513 |
| `paper_marker_depth_r` | +0.7838 |
| `paper_celltype_localization` | +0.6534 |
| `paper_gene_mean_spearman` | +0.9880 |
| `paper_cell_count_ratio` (raw pass) | 0.895 |

Per section, on the two metrics R11 turns on — `celltype_localization` matched,
`cell_count_ratio` raw:

| arm | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `hybrid-rejection` localization | +0.6534 | +0.3581 | +0.6790 | +0.6534 |
| `hybrid-rejection` count ratio | 42.870 | 0.040 | 0.895 | 0.895 |

As emitted, without the density control:

| metric | `hybrid-rejection` |
|---|---|
| `paper_morans_pearson` | +0.6614 |
| `paper_gearys_pearson` | +0.6624 |
| `paper_umap_mixing` | — |
| `paper_marker_field_r` | +0.6749 |
| `paper_marker_depth_r` | +0.7838 |
| `paper_celltype_localization` | +0.6762 |
| `paper_gene_mean_spearman` | +0.9880 |

Emitted cell counts (generated/ground truth):

* `hybrid-rejection`: section_2=179495/4187, section_4=163/4102, section_6=3727/4162
