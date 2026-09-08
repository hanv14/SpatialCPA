# T09 audit — `expr_mode` measured where it is live, per fold

Dataset **`starmap_visual_cortex`**, holdout **`paper_2_4_6`** — 16527 training cells x 28 genes over 4 sections. Config from Config defaults, measured under **`prior_mode=correlated`**, 2400 steps, seed 2.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `cross-mix` mean | `zinb-flow` mean | `cross-mix` section_3 | `zinb-flow` section_3 | `cross-mix` section_5 | `zinb-flow` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.9227 | +0.7914 | +0.9020 | +0.7637 | +0.9434 | +0.8191 | 0.1313 | 3.9x | **2.4x** | **0.90** |
| `gearys_pearson` | +0.9269 | +0.7965 | +0.9063 | +0.7694 | +0.9474 | +0.8237 | 0.1303 | 3.9x | **2.4x** | **0.90** |
| `umap_mixing` | +0.8007 | +0.6592 | +0.7686 | +0.6506 | +0.8329 | +0.6679 | 0.1415 | 4.2x | **2.2x** | **0.72** |
| `marker_field_r` | +0.7368 | +0.7089 | +0.6506 | +0.6717 | +0.8231 | +0.7460 | 0.0280 | **inside** | ⚠ 0.2x | **0.27** |
| `marker_depth_r` | +0.7320 | +0.6751 | +0.6926 | +0.6129 | +0.7713 | +0.7373 | 0.0569 | 1.7x | ⚠ 0.5x | **0.43** |
| `celltype_localization` | +0.7754 | +0.7754 | +0.7473 | +0.7473 | +0.8034 | +0.8034 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `umap_mixing` at 0.1415** (4.2x the 0.0335 envelope, 2.2x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.72), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
