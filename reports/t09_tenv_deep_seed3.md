# T09 audit — `text_emb_mode` measured where it is live, per fold

> 🚩 **The `vs 0.0335` column in the table below is WITHDRAWN (2026-09-08).** 0.0335 is R10's
> **maximum over six metrics** of a spread measured on the **synthetic fixture**, not on
> `deep_starmap` and not on this gate. This file is one of the three seeds that measure the right
> envelope for it:
>
> | metric | `lookup` | `medcpt` | shared (§4.2b) |
> |---|---|---|---|
> | `morans_pearson` | 0.0073 | **0.0246** | 0.0246 |
> | `gearys_pearson` | **0.0501** | 0.0398 | 0.0501 |
> | `umap_mixing` | **0.0201** | 0.0067 | 0.0201 |
> | `marker_field_r` | **0.0197** | 0.0152 | 0.0197 |
> | `marker_depth_r` | **0.0427** | 0.0182 | 0.0427 |
>
> Note that **the worse arm alternates by metric** — `medcpt` on `morans_pearson`, `lookup` on the
> other four — which is `specs/10` §4.2a's own finding, measured here. Derivation in `reports/envelope_correction.md`.


Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 training cells x 1017 genes over 4 sections. Config from Config defaults, measured under **`expr_mode=zinb-flow`**, 2400 steps, seed 3.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `medcpt` mean | `lookup` mean | `medcpt` section_3 | `lookup` section_3 | `medcpt` section_5 | `lookup` section_5 | margin (mean) | vs 0.0335 | vs fold spread | fold balance |
|---|---|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.5402 | +0.6569 | +0.5318 | +0.6585 | +0.5486 | +0.6554 | 0.1167 | 3.5x | **6.9x** | **0.84** |
| `gearys_pearson` | +0.3481 | +0.4503 | +0.3476 | +0.4672 | +0.3486 | +0.4334 | 0.1022 | 3.1x | **3.0x** | **0.71** |
| `umap_mixing` | +0.4905 | +0.4876 | +0.5226 | +0.5113 | +0.4585 | +0.4639 | 0.0029 | **inside** | ⚠ 0.0x | **0.48** |
| `marker_field_r` | +0.2453 | +0.2743 | +0.3023 | +0.2906 | +0.1883 | +0.2581 | 0.0291 | **inside** | ⚠ 0.3x | ⚠ 0.17 |
| `marker_depth_r` | +0.3577 | +0.3684 | +0.4214 | +0.3748 | +0.2940 | +0.3620 | 0.0106 | **inside** | ⚠ 0.1x | **0.69** |
| `celltype_localization` | +0.6028 | +0.6028 | +0.6743 | +0.6743 | +0.5314 | +0.5314 | 0.0000 | **inside** | ⚠ 0.0x | — |

**Largest separation: `morans_pearson` at 0.1167** (3.5x the 0.0335 envelope, 6.9x the worst within-arm fold spread). The two folds **agree in sign**, and carry it evenly (balance 0.84), so the gap is not carried by one of them.


**`fold balance`** is ``min |per-fold difference| / max |per-fold difference|``. Sign agreement is not enough: two folds can agree in sign while one contributes the whole mean (**⚠** below 0.25). A margin with a balance near zero is one fold's difference halved, whatever its ratio to the envelope.

**`vs fold spread` is the column to read at n = 2.** R10's 0.0335 is an across-*seed* envelope measured on the fixture; it says nothing about how much a metric moves between *this* dataset's folds. A margin smaller than the worst within-arm fold spread (**⚠** below 2x) separates the two arms by less than one arm moves on its own, and is not a result however far it clears the envelope.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
