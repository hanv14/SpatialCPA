# T09 audit — `text_emb_mode` measured where it is live, per fold

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 training cells x 1017 genes over 4 sections. Config from Config defaults, measured under **`expr_mode=zinb-flow`**, 2400 steps, seed 2.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `medcpt` mean | `lookup` mean | `medcpt` section_3 | `lookup` section_3 | `medcpt` section_5 | `lookup` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.5388 | +0.6496 | +0.5349 | +0.6495 | +0.5427 | +0.6498 | 0.1108 | 3.3x | **14.3x** | **0.93** |
| `gearys_pearson` | +0.3879 | +0.5004 | +0.4082 | +0.5264 | +0.3676 | +0.4745 | 0.1126 | 3.4x | **2.2x** | **0.90** |
| `umap_mixing` | +0.4925 | +0.4904 | +0.5236 | +0.5115 | +0.4614 | +0.4694 | 0.0021 | **inside** | ⚠ 0.0x | **0.66** |
| `marker_field_r` | +0.2300 | +0.2712 | +0.2409 | +0.3014 | +0.2191 | +0.2410 | 0.0412 | 1.2x | ⚠ 0.7x | **0.36** |
| `marker_depth_r` | +0.3475 | +0.3680 | +0.3258 | +0.3698 | +0.3692 | +0.3662 | 0.0205 | **inside** | ⚠ 0.5x | ⚠ 0.07 |
| `celltype_localization` | +0.6028 | +0.6028 | +0.6743 | +0.6743 | +0.5314 | +0.5314 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `gearys_pearson` at 0.1126** (3.4x the 0.0335 envelope, 2.2x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.90), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
