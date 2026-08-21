# Re-scored on the pinned instrument — `model_exp_2400.pt`, no refit

`decoder_mu_link=exp`, 2400 steps. `layout_mode` is a
generation-time gate, so all arms share one set of weights.

**Ground-truth-matched density** — each section subsampled to its own ground-truth cell
count, because a denser point set puts kNN neighbours closer and inflates every
graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported
from the raw pass instead.

| metric | `field` | `hybrid` | `resample` |
|---|---|---|---|
| `paper_morans_pearson` | +0.5823 | +0.5749 | +0.5623 |
| `paper_gearys_pearson` | +0.5881 | +0.5716 | +0.5632 |
| `paper_umap_mixing` | +0.9169 | +0.9152 | +0.9217 |
| `paper_marker_field_r` | +0.5763 | +0.6384 | +0.6692 |
| `paper_marker_depth_r` | +0.5919 | +0.7478 | +0.8284 |
| `paper_celltype_localization` | +0.6136 | +0.6572 | +0.7601 |
| `paper_cell_count_ratio` (raw pass) | 0.894 | 0.894 | 0.988 |

As emitted, without the density control:

| metric | `field` | `hybrid` | `resample` |
|---|---|---|---|
| `paper_morans_pearson` | +0.5386 | +0.5481 | +0.5623 |
| `paper_gearys_pearson` | +0.5379 | +0.5469 | +0.5622 |
| `paper_umap_mixing` | +0.9056 | +0.9225 | +0.9184 |
| `paper_marker_field_r` | +0.6073 | +0.6503 | +0.6528 |
| `paper_marker_depth_r` | +0.7429 | +0.7478 | +0.8284 |
| `paper_celltype_localization` | +0.6136 | +0.6572 | +0.7659 |

Emitted cell counts (generated/ground truth):

* `field`: section_2=48343/4187, section_4=92/4102, section_6=3719/4162
* `hybrid`: section_2=48343/4187, section_4=92/4102, section_6=3719/4162
* `resample`: section_2=4073/4187, section_4=4169/4102, section_6=4110/4162
