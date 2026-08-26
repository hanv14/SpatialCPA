# T09 audit — `text_emb_mode` measured where it is live, per fold

Tier-1 STARmap (`starmap_visual_cortex`, `paper_2_4_6`). Config from Config defaults, measured under **`expr_mode=zinb-flow`**, 2400 steps, seed 1.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `medcpt` mean | `lookup` mean | `medcpt` section_3 | `lookup` section_3 | `medcpt` section_5 | `lookup` section_5 | margin (mean) | vs 0.0335 |
|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.7510 | +0.7943 | +0.7289 | +0.8012 | +0.7730 | +0.7873 | 0.0433 | 1.3x |
| `gearys_pearson` | +0.7519 | +0.7944 | +0.7374 | +0.7995 | +0.7664 | +0.7892 | 0.0425 | 1.3x |
| `umap_mixing` | +0.6564 | +0.6716 | +0.6553 | +0.6691 | +0.6575 | +0.6742 | 0.0152 | **inside** |
| `marker_field_r` | -0.1355 | -0.1196 | -0.1552 | -0.1572 | -0.1157 | -0.0820 | 0.0159 | **inside** |
| `marker_depth_r` | -0.3084 | -0.3117 | -0.3030 | -0.3175 | -0.3138 | -0.3060 | 0.0033 | **inside** |
| `celltype_localization` | +0.0256 | +0.0256 | +0.0146 | +0.0146 | +0.0365 | +0.0365 | 0.0000 | **inside** |

**Largest separation: `morans_pearson` at 0.0433** (1.3x the 0.0335 envelope). The two folds **agree in sign**, so the gap is not carried by one of them.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
