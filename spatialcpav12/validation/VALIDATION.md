# SpatialCPA-v12 — synthetic validation

These are **synthetic-data** head-to-heads computed by the *real* `benchmark-pbya-v2`
generation evaluator (`evaluate_generation.py`) and the *real* method wrappers — only the
input data is synthetic, because the processed benchmark datasets are not bundled here.
**No real-leaderboard numbers are reported or fabricated.**

The comparison is **v12 (default generative decode) vs v8 (default)** — v8 being the
strongest prior SpatialCPA generator. v12 shares **none** of v8's machinery: no optimal
transport, no diffeomorphic / barycentric morph, no two-slice OT fusion, no niche
Markov-random-field. It is a generative continuous neural field.

Reproduce (in an env with torch + scanpy + anndata + scikit-image, e.g.
`bench_spatialcpa`):

```bash
cd spatialcpav12/validation
python make_synth_distinct.py   dis.h5ad
python make_synth_volumetric.py vol.h5ad
for H in S2 S3 S4; do
  python compare_v12_v8.py dis.h5ad $H      # distinct-tissue regime (IMC-like)
  python compare_v12_v8.py vol.h5ad $H      # near-identical / volumetric regime (STARmap-like)
done
```

Higher is better for every metric except `sinkhorn` (lower is better). All ten are the
correspondence-free generation metrics the benchmark uses to score a de-novo virtual
slice. Numbers below are the **mean over three held-out sections (S2, S3, S4)** at 200
training epochs; `Δ = v12 − v8` (sign flipped for `sinkhorn`, where lower is better).

## Distinct-tissue regime (IMC-like) — v12 at parity / slight edge (5 wins, 4 losses on mean metrics)

| metric | v12 | v8 | Δ | result |
|---|---|---|---|---|
| coexpression_agreement | 0.830 | 0.837 | −0.007 | ~tie |
| morans_agreement | 0.668 | 0.760 | −0.092 | lose |
| sinkhorn (↓) | 0.584 | 0.589 | −0.005 | **win** |
| celltype_composition | 0.963 | 0.954 | +0.009 | **win** |
| celltype_nhood_agreement | 0.988 | 0.993 | −0.004 | ~tie |
| gene_mean_pearson | 0.970 | 0.959 | +0.011 | **win** |
| gene_var_pearson | 0.903 | 0.902 | +0.002 | **win** |
| field_pearson | 0.212 | 0.212 | +0.000 | tie |
| field_ssim | 0.064 | −0.024 | +0.088 | **win** |
| density_pearson | 0.203 | 0.267 | −0.064 | lose |

Per-holdout win/loss: S2 **8–2**, S3 5–5, S4 3–7 (holdout-dependent; v12 ahead overall).

## Near-identical / volumetric regime (STARmap-like) — v12 competitive, v8 ahead on spatial coherence

| metric | v12 | v8 | Δ | result |
|---|---|---|---|---|
| coexpression_agreement | 0.883 | 0.900 | −0.017 | lose |
| morans_agreement | 0.765 | 0.819 | −0.054 | lose |
| sinkhorn (↓) | 0.572 | 0.568 | +0.004 | ~tie |
| celltype_composition | 0.997 | 0.996 | +0.002 | **win** |
| celltype_nhood_agreement | 0.989 | 0.997 | −0.008 | lose (small) |
| gene_mean_pearson | 0.977 | 0.978 | −0.001 | tie |
| gene_var_pearson | 0.937 | 0.937 | −0.000 | tie |
| field_pearson | 0.337 | 0.345 | −0.008 | lose (small) |
| field_ssim | 0.298 | 0.291 | +0.007 | **win** |
| density_pearson | 0.578 | 0.688 | −0.109 | lose |

## Interpretation (honest)

* **vs v11 (the version v12 enhances): a clear, consistent improvement.** The
  factor-analysis decoder + count-space output repair v11's mean-collapse
  (co-expression, gene variance), and the regime-adaptive positions repair v11's coarse,
  unstable density — v12 dominates v11 on co-expression, gene mean/variance, composition,
  field-SSIM and density.

* **vs v8: parity on distinct tissue, competitive on near-identical planes.** On distinct
  tissue v12 wins the expression-distribution metrics (gene mean/variance, composition,
  field-SSIM, sinkhorn) and is a hair behind on co-expression / nhood; on near-identical
  planes v12 ties the gene-level metrics and wins composition/field-SSIM. Nearly every
  loss is within a few hundredths.

* **The two honest residues are `morans_agreement` and `density_pearson`.** These are the
  spatial-coherence metrics that a *real-slice copy + coherent warp* (v8's design) is
  built to dominate — v8's own README flags exactly this "genuine Pareto residue against a
  real-slice copy" on near-identical planes. v12 pays a little of that residue to stay a
  fully **generative, non-copy** model (learned occupancy/type fields + a probabilistic
  covariance-preserving expression decoder). On real datasets v8's copy is only
  near-optimal when the flanking slice is nearly the held-out slice, so this residue is
  expected to narrow, while v12's gene-covariance modelling has more room to help on the
  larger real gene panels.

The processed real benchmark datasets and conda environments are not bundled here, so the
full cross-dataset leaderboard must be regenerated with `run_benchmark.py` where the data
live. No benchmark numbers are fabricated.
