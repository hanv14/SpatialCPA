# T09 audit — `expr_mode` measured where it is live, per fold

Dataset **`starmap_visual_cortex`**, holdout **`paper_2_4_6`** — 16527 training cells x 28 genes over 4 sections. Config from Config defaults, measured under **`prior_mode=correlated`**, 2400 steps, seed 3.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `cross-mix` mean | `zinb-flow` mean | `cross-mix` section_3 | `zinb-flow` section_3 | `cross-mix` section_5 | `zinb-flow` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.9196 | +0.7690 | +0.8935 | +0.7558 | +0.9458 | +0.7823 | 0.1506 | 4.5x | **2.9x** | **0.84** |
| `gearys_pearson` | +0.9295 | +0.7677 | +0.9097 | +0.7529 | +0.9493 | +0.7825 | 0.1618 | 4.8x | **4.1x** | **0.94** |
| `umap_mixing` | +0.8075 | +0.6462 | +0.7757 | +0.6349 | +0.8394 | +0.6575 | 0.1613 | 4.8x | **2.5x** | **0.77** |
| `marker_field_r` | +0.7376 | +0.7237 | +0.6505 | +0.6923 | +0.8247 | +0.7551 | 0.0139 | **inside** | ⚠ 0.1x | **0.60** |
| `marker_depth_r` | +0.7404 | +0.6383 | +0.6867 | +0.5605 | +0.7940 | +0.7161 | 0.1021 | 3.0x | ⚠ 0.7x | **0.62** |
| `celltype_localization` | +0.7754 | +0.7754 | +0.7473 | +0.7473 | +0.8034 | +0.8034 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `gearys_pearson` at 0.1618** (4.8x the 0.0335 envelope, 4.1x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.94), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
