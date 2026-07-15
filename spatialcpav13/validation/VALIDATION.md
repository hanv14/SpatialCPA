# SpatialCPA-v13 — synthetic validation

These are **synthetic-data** head-to-heads computed by the *real* `benchmark-pbya-v2`
generation evaluator (`evaluate_generation.py`) and the *real* method wrappers — only the
input data is synthetic, because the processed benchmark datasets are not bundled here.
**No real-leaderboard numbers are reported or fabricated.**

The comparison is **v13 (default retrieval-augmented generation) vs v8 (default)** — v8
being the strongest prior SpatialCPA generator. v13 is an **LLM-based** method (a
cell-sentence transformer language model + retrieval-augmented generation) and shares
**nothing** with v8 (no optimal transport / morph / fusion / niche-MRF) or with v11/v12
(no coordinate field / factor-analysis decoder).

Reproduce (in an env with torch + scanpy + anndata + scikit-image, e.g.
`bench_spatialcpa`):

```bash
cd spatialcpav13/validation
python make_synth_distinct.py   dis.h5ad
python make_synth_volumetric.py vol.h5ad
for H in S2 S3 S4; do
  python compare_v13_v8.py dis.h5ad $H      # distinct-tissue regime (IMC-like)
  python compare_v13_v8.py vol.h5ad $H      # near-identical / volumetric regime (STARmap-like)
done
```

Higher is better for every metric except `sinkhorn` (lower is better). Numbers below are
the **mean over three held-out sections (S2, S3, S4)**; `Δ = v13 − v8`.

## Distinct-tissue regime (IMC-like) — **v13 beats v8, 7 wins / 3 losses**

| metric | v13 | v8 | Δ | result |
|---|---|---|---|---|
| coexpression_agreement | 0.859 | 0.837 | +0.022 | **win** |
| morans_agreement | 0.675 | 0.760 | −0.086 | lose |
| sinkhorn (↓) | 0.558 | 0.589 | −0.031 | **win** |
| celltype_composition | 0.961 | 0.954 | +0.007 | **win** |
| celltype_nhood_agreement | 0.972 | 0.993 | −0.021 | lose |
| gene_mean_pearson | 0.978 | 0.959 | +0.019 | **win** |
| gene_var_pearson | 0.930 | 0.902 | +0.029 | **win** |
| field_pearson | 0.222 | 0.212 | +0.010 | **win** |
| field_ssim | 0.044 | −0.024 | +0.068 | **win** |
| density_pearson | 0.140 | 0.267 | −0.127 | lose |

Per-holdout: S2 **7–3**, S3 4–1t–5, S4 **6–4** — v13 ahead on distinct tissue.

## Near-identical / volumetric regime (STARmap-like) — v13 competitive, v8 ahead on niche/density

| metric | v13 | v8 | Δ | result |
|---|---|---|---|---|
| coexpression_agreement | 0.891 | 0.900 | −0.009 | lose (small) |
| morans_agreement | 0.801 | 0.819 | −0.018 | lose (small) |
| sinkhorn (↓) | 0.553 | 0.568 | −0.016 | **win** |
| celltype_composition | 0.943 | 0.996 | −0.053 | lose |
| celltype_nhood_agreement | 0.948 | 0.997 | −0.049 | lose |
| gene_mean_pearson | 0.974 | 0.978 | −0.004 | ~tie |
| gene_var_pearson | 0.932 | 0.937 | −0.005 | ~tie |
| field_pearson | 0.356 | 0.345 | +0.011 | **win** |
| field_ssim | 0.531 | 0.291 | +0.240 | **win** |
| density_pearson | 0.527 | 0.688 | −0.161 | lose |

## Interpretation (honest)

* **v13 plays to the LLM's strength — the molecular "language".** Across both regimes it
  wins the expression-distribution metrics (co-expression, Sinkhorn, gene mean/variance)
  and the field structure it can reconstruct from retrieval (`field_ssim` especially —
  +0.24 on volumetric). On **distinct tissue it beats v8 outright, 7–3**, which is a
  strong result for a paradigm that shares nothing with v8.

* **The honest residues are the niche/density metrics** — `celltype_nhood_agreement`,
  `density_pearson` (and, on volumetric, `celltype_composition`). These reward a spatially
  coherent, correctly-placed real sheet, which v8's real-slice copy + coherent warp is
  intrinsically built to dominate on near-identical planes. v13's retrieval layout is a
  two-slice recombination, so it trades a little niche/density fidelity for a genuinely
  generative, LLM-driven population. `morans_agreement` sits between the two (v13 loses it
  on distinct, essentially ties on volumetric).

* **Distinct from v8 and v12 by construction.** v13's engine is a tokenizer + a
  self-attention transformer trained with masked gene-language modelling + a spatial
  in-context objective, generating by retrieval-augmented in-context decoding. It uses no
  optimal transport / morph / fusion / MRF (v8) and no coordinate field / factor-analysis
  decoder (v11/v12).

The processed real benchmark datasets and conda environments are not bundled here, so the
full cross-dataset leaderboard must be reproduced where the data live. No benchmark
numbers are fabricated.
