# T09 audit — `expr_mode` measured where it is live, per fold

> 🚩 **The `vs 0.0335` column in the table below is WITHDRAWN (2026-09-08).** 0.0335 is R10's
> **maximum over six metrics** of a spread measured on the **synthetic fixture** — pooled across
> metrics and measured on a different volume. This file is one of the three seeds that measure the
> **right** envelope for this exact gate, dataset, instrument and arm pair, so the replacement is
> derivable from these files alone:
>
> | metric | `cross-mix` | `zinb-flow` | shared (§4.2b) |
> |---|---|---|---|
> | `morans_pearson` | 0.0054 | **0.0574** | 0.0574 |
> | `gearys_pearson` | 0.0027 | **0.0595** | 0.0595 |
> | `umap_mixing` | 0.0068 | **0.0190** | 0.0190 |
> | `marker_field_r` | 0.0049 | **0.0148** | 0.0148 |
> | `marker_depth_r` | 0.0084 | **0.0472** | 0.0472 |
> | `celltype_localization` | 0.0000 | 0.0000 | inert under `resample` |
>
> Three-seed margins against those: `morans` **2.29x**, `gearys` **2.19x**, `umap_mixing`
> **7.45x**, `marker_field_r` **1.35x**, `marker_depth_r` **1.21x** (medians; §4.6). Derivation in
> `reports/envelope_correction.md` §2.1. `vs fold spread` was always the honest column at n = 2.


Dataset **`starmap_visual_cortex`**, holdout **`paper_2_4_6`** — 16527 training cells x 28 genes over 4 sections. Config from Config defaults, measured under **`prior_mode=correlated`**, 2400 steps, seed 4.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `cross-mix` mean | `zinb-flow` mean | `cross-mix` section_3 | `zinb-flow` section_3 | `cross-mix` section_5 | `zinb-flow` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.9173 | +0.8265 | +0.8950 | +0.7900 | +0.9395 | +0.8629 | 0.0908 | 2.7x | ⚠ 1.2x | **0.73** |
| `gearys_pearson` | +0.9284 | +0.8273 | +0.9114 | +0.7932 | +0.9453 | +0.8613 | 0.1011 | 3.0x | ⚠ 1.5x | **0.71** |
| `umap_mixing` | +0.8016 | +0.6652 | +0.7689 | +0.6526 | +0.8344 | +0.6778 | 0.1364 | 4.1x | **2.1x** | **0.74** |
| `marker_field_r` | +0.7327 | +0.7127 | +0.6466 | +0.6992 | +0.8188 | +0.7261 | 0.0201 | **inside** | ⚠ 0.1x | **0.57** |
| `marker_depth_r` | +0.7398 | +0.6855 | +0.7044 | +0.6818 | +0.7752 | +0.6891 | 0.0543 | 1.6x | ⚠ 0.8x | **0.26** |
| `celltype_localization` | +0.7754 | +0.7754 | +0.7473 | +0.7473 | +0.8034 | +0.8034 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `umap_mixing` at 0.1364** (4.1x the 0.0335 envelope, 2.1x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.74), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
