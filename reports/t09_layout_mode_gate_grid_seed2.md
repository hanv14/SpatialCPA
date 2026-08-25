# T09's `layout_mode` gate, re-run on the grid sampler

Synthetic fixture, `alternating` holdout, internal LOSO over training sections, 2400 steps, seeds [2], `layout_sampler=grid`.

**One fit per seed.** `layout_mode` is read only at generation time, and fitting the
fixture at all three modes with one seed gives bitwise identical weights over all 96
parameter and buffer tensors — so the three arms below share one model and the contrast
carries no fit-to-fit noise. `text_emb_mode=lookup` (ablation A3): the encoder needs a
network. It is constant across arms and cannot affect the contrast.

## Median over seeds, per metric (higher is better)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | median rank |
|---|---|---|---|---|---|---|---|
| `field` | +0.7417 | +0.6755 | +0.3052 | +0.0124 | +0.0506 | -0.0898 | **2.5** |
| `hybrid` | +0.7493 | +0.6794 | +0.3105 | +0.0093 | +0.0691 | -0.0864 | **1.5** |
| `resample` | +0.7488 | +0.7052 | +0.3025 | +0.0053 | +0.0433 | -0.0660 | **2.5** |

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
| 2 | 2.5 | 1.5 | 2.5 |

## Pairwise gaps against `resample`, per metric (median over seeds)

| metric | `field` less `resample` | `hybrid` less `resample` | vs envelope |
|---|---|---|---|
| `morans_pearson` | -0.0071 | +0.0005 | 0.2x |
| `gearys_pearson` | -0.0297 | -0.0257 | 0.9x |
| `umap_mixing` | +0.0027 | +0.0081 | 0.2x |
| `marker_field_r` | +0.0071 | +0.0040 | 0.2x |
| `marker_depth_r` | +0.0073 | +0.0258 | 0.8x |
| `celltype_localization` | -0.0238 | -0.0204 | 0.7x |
