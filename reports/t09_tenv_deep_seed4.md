# T09 audit — `text_emb_mode` measured where it is live, per fold

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 training cells x 1017 genes over 4 sections. Config from Config defaults, measured under **`expr_mode=zinb-flow`**, 2400 steps, seed 4.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `medcpt` mean | `lookup` mean | `medcpt` section_3 | `lookup` section_3 | `medcpt` section_5 | `lookup` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.5157 | +0.6515 | +0.5099 | +0.6486 | +0.5214 | +0.6543 | 0.1358 | 4.1x | **11.8x** | **0.96** |
| `gearys_pearson` | +0.3566 | +0.4805 | +0.3847 | +0.5178 | +0.3284 | +0.4432 | 0.1239 | 3.7x | ⚠ 1.7x | **0.86** |
| `umap_mixing` | +0.4973 | +0.5078 | +0.5294 | +0.5349 | +0.4652 | +0.4806 | 0.0105 | **inside** | ⚠ 0.2x | **0.36** |
| `marker_field_r` | +0.2424 | +0.2909 | +0.2748 | +0.3084 | +0.2100 | +0.2734 | 0.0485 | 1.4x | ⚠ 0.7x | **0.53** |
| `marker_depth_r` | +0.3396 | +0.3256 | +0.3736 | +0.3426 | +0.3055 | +0.3087 | 0.0139 | **inside** | ⚠ 0.2x | ⚠ 0.10 |
| `celltype_localization` | +0.6028 | +0.6028 | +0.6743 | +0.6743 | +0.5314 | +0.5314 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `morans_pearson` at 0.1358** (4.1x the 0.0335 envelope, 11.8x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.96), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
