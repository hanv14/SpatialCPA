# T09's `layout_mode` gate, re-run on the grid sampler

Synthetic fixture, `alternating` holdout, internal LOSO over training sections, 2400 steps, seeds [1], `layout_sampler=grid`.

**One fit per seed.** `layout_mode` is read only at generation time, and fitting the
fixture at all three modes with one seed gives bitwise identical weights over all 96
parameter and buffer tensors — so the three arms below share one model and the contrast
carries no fit-to-fit noise. `text_emb_mode=lookup` (ablation A3): the encoder needs a
network. It is constant across arms and cannot affect the contrast.

## Median over seeds, per metric (higher is better)

| arm | `morans_pearson` | `gearys_pearson` | `umap_mixing` | `marker_field_r` | `marker_depth_r` | `celltype_localization` | median rank |
|---|---|---|---|---|---|---|---|
| `field` | +0.7292 | +0.6786 | +0.3011 | +0.0166 | +0.0293 | -0.0657 | **1.5** |
| `hybrid` | +0.7304 | +0.6725 | +0.3045 | +0.0052 | +0.0387 | -0.0661 | **2.0** |
| `resample` | +0.7235 | +0.6773 | +0.2989 | +0.0064 | +0.0383 | -0.0660 | **2.0** |

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
| 1 | 1.5 | 2.0 | 2.0 |

## Pairwise gaps against `resample`, per metric (median over seeds)

| metric | `field` less `resample` | `hybrid` less `resample` | vs envelope |
|---|---|---|---|
| `morans_pearson` | +0.0057 | +0.0069 | 0.2x |
| `gearys_pearson` | +0.0013 | -0.0048 | 0.1x |
| `umap_mixing` | +0.0022 | +0.0057 | 0.2x |
| `marker_field_r` | +0.0102 | -0.0012 | 0.3x |
| `marker_depth_r` | -0.0090 | +0.0004 | 0.3x |
| `celltype_localization` | +0.0003 | -0.0002 | 0.0x |
