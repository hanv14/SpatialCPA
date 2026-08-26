# T09 audit — `expr_mode` measured where it is live, per fold

Tier-1 STARmap (`starmap_visual_cortex`, `paper_2_4_6`). Config from Config defaults, measured under **`prior_mode=correlated`**, 2400 steps, seed 1.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `cross-mix` mean | `zinb-flow` mean | `cross-mix` section_3 | `zinb-flow` section_3 | `cross-mix` section_5 | `zinb-flow` section_5 | margin (mean) | vs 0.0335 |
|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.9255 | +0.7510 | +0.9151 | +0.7289 | +0.9359 | +0.7730 | 0.1745 | 5.2x |
| `gearys_pearson` | +0.9293 | +0.7519 | +0.9174 | +0.7374 | +0.9412 | +0.7664 | 0.1774 | 5.3x |
| `umap_mixing` | +0.8098 | +0.6564 | +0.7919 | +0.6553 | +0.8276 | +0.6575 | 0.1534 | 4.6x |
| `marker_field_r` | -0.1412 | -0.1355 | -0.1679 | -0.1552 | -0.1145 | -0.1157 | 0.0057 | **inside** |
| `marker_depth_r` | -0.3086 | -0.3084 | -0.3243 | -0.3030 | -0.2929 | -0.3138 | 0.0002 | **inside** |
| `celltype_localization` | +0.0256 | +0.0256 | +0.0146 | +0.0146 | +0.0365 | +0.0365 | 0.0000 | **inside** |

**Largest separation: `gearys_pearson` at 0.1774** (5.3x the 0.0335 envelope). The two folds **agree in sign**, so the gap is not carried by one of them.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
