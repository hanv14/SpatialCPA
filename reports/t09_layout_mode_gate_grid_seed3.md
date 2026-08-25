# T09's `layout_mode` gate, re-run on the grid sampler

Synthetic fixture, `alternating` holdout, internal LOSO over training sections, 2400 steps, seeds [3], `layout_sampler=grid`.

**One fit per seed.** `layout_mode` is read only at generation time, and fitting the
fixture at all three modes with one seed gives bitwise identical weights over all 96
parameter and buffer tensors — so the three arms below share one model and the contrast
carries no fit-to-fit noise. `text_emb_mode=lookup` (ablation A3): the encoder needs a
network. It is constant across arms and cannot affect the contrast.

## Median over seeds, per metric (higher is better)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | median rank |
|---|---|---|---|---|---|---|---|
| `field` | +0.7214 | +0.6837 | +0.3032 | -0.0085 | +0.0505 | -0.0853 | **2.5** |
| `hybrid` | +0.7400 | +0.7132 | +0.3071 | -0.0362 | +0.0448 | -0.0929 | **2.5** |
| `resample` | +0.7394 | +0.6844 | +0.3160 | -0.0330 | +0.0673 | -0.0660 | **1.5** |

## Across-seed spread (max minus min over the seeds)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | max |
|---|---|---|---|---|---|---|---|
| `field` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| `hybrid` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| `resample` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |

R10's measured across-seed envelope is **0.0335**. A gap below it is not a gap.

## Median rank per seed

| seed | `field` | `hybrid` | `resample` |
|---|---|---|---|
| 3 | 2.5 | 2.5 | 1.5 |

## Pairwise gaps against `resample`, per metric (median over seeds)

| metric | `field` less `resample` | `hybrid` less `resample` | vs envelope |
|---|---|---|---|
| `morans_pearson` | -0.0180 | +0.0005 | 0.5x |
| `gearys_pearson` | -0.0007 | +0.0289 | 0.9x |
| `umap_mixing` | -0.0128 | -0.0089 | 0.4x |
| `marker_field_r` | +0.0245 | -0.0033 | 0.7x |
| `marker_depth_r` | -0.0168 | -0.0225 | 0.7x |
| `celltype_localization` | -0.0193 | -0.0269 | 0.8x |
