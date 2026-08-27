# T09 audit — `text_emb_mode` measured where it is live, per fold

> ⚠️ **Header corrected 2026-08-26.** This file was written by a version of
> `scripts/t09_audit_starmap.py` whose report header hardcoded the tier-1 STARmap dataset name,
> so it said `starmap_visual_cortex` on a `deep_starmap` run. **Only this header was corrected;
> every number below is exactly as emitted.** The dataset is proven by the config hashes: all
> four arms reconstruct exactly under `expr_pca_dim=32` (1017 genes, unclamped) and none under
> `expr_pca_dim=28` (STARmap's clamp), which reproduces the STARmap reports' own hashes instead.
> The generator now takes the dataset from the resolved paths and records it in the JSON.

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 1017 genes, mouse brain, `raw_counts`, 137 cell types (`specs/10` §5.4). **Tier 2.** Config from Config defaults, measured under **`expr_mode=zinb-flow`**, 2400 steps, seed 1.

The selection could not make this measurement: it scored this gate under `expr_mode=cross-mix`, where `_expression` returns from `_cross_mix` before the prior, the flow, the decoder and the gene embeddings are reached — so both options emitted bitwise-identical counts and the gate reported a separation of exactly **0.0000**. That is an absence of measurement, not a tie.

**Folds: 2** — `section_3`, `section_5`. `selection_folds` takes the *interior* sections, so a four-section training stack gives 2 however large `Config.selection_n_folds` (3) is set. Read the per-fold columns, not the mean.

| metric | `medcpt` mean | `lookup` mean | `medcpt` section_3 | `lookup` section_3 | `medcpt` section_5 | `lookup` section_5 | margin (mean) | vs 0.0335 |
|---|---|---|---|---|---|---|---|---|
| `morans_pearson` | +0.5336 | +0.6571 | +0.5177 | +0.6503 | +0.5496 | +0.6638 | 0.1234 | 3.7x |
| `gearys_pearson` | +0.3695 | +0.4983 | +0.3556 | +0.4841 | +0.3834 | +0.5125 | 0.1288 | 3.8x |
| `umap_mixing` | +0.5084 | +0.5223 | +0.5205 | +0.5331 | +0.4963 | +0.5114 | 0.0139 | **inside** |
| `marker_field_r` | -0.0853 | -0.0533 | -0.0998 | -0.0737 | -0.0708 | -0.0328 | 0.0320 | **inside** |
| `marker_depth_r` | +0.2745 | +0.0895 | +0.2447 | -0.0121 | +0.3044 | +0.1911 | 0.1850 | 5.5x |
| `celltype_localization` | -0.0051 | -0.0051 | -0.0112 | -0.0112 | +0.0010 | +0.0010 | 0.0000 | **inside** |

**Largest separation: `marker_depth_r` at 0.1850** (5.5x the 0.0335 envelope). The two folds **agree in sign**, so the gap is not carried by one of them.

**One seed.** `specs/09` §3's repeated-seed rule asks for `claim_min_seeds` = 3 before this reaches a paper claim.
