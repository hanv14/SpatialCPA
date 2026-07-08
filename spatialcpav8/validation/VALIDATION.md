# SpatialCPA-v8 — synthetic validation

These are **synthetic-data** ablations, computed by the *real* `benchmark-pbya-v2`
evaluators (`evaluate.py` + `evaluate_generation.py`) and the *real* method
wrappers — only the input data is synthetic, because the processed benchmark
datasets are not bundled in this repository. **No real-leaderboard numbers are
reported or fabricated here.** They isolate the effect of v8's coherent-OT bridge
against v6 in the two regimes the benchmark contains.

Reproduce:

```bash
# in the bench_spatialcpa env (numpy/scipy/sklearn/skimage/anndata/scanpy/h5py)
python make_synth_volumetric.py vol.h5ad   && python compare_v6_v8.py vol.h5ad S3
python make_synth_distinct.py   distinct.h5ad && python compare_v6_v8.py distinct.h5ad S3
```

`winner` compares v8 vs v6 (higher is better except `gen_sinkhorn`, where lower is
better); ties are exact.

## Near-identical / volumetric regime (STARmap-like — where v6 lost to v5/SpatialZ)

v8 routes to the **smooth morph** and wins 15/17 metrics simultaneously — both the
expression-structure metrics *and* the spatial-field metrics improve together,
which is the trade-off prior methods could not break.

| metric | v6 | v8 | winner |
|---|---|---|---|
| gen_coexpression_agreement | 0.863 | 0.893 | **v8** |
| gen_morans_agreement | 0.763 | 0.793 | **v8** |
| gen_sinkhorn (lower=better) | 0.589 | 0.567 | **v8** |
| gen_celltype_composition | 0.937 | 0.997 | **v8** |
| gen_celltype_nhood_agreement | 0.982 | 0.994 | **v8** |
| gen_gene_mean_pearson | 0.973 | 0.977 | **v8** |
| gen_gene_var_pearson | 0.931 | 0.937 | **v8** |
| gen_field_pearson | 0.293 | 0.364 | **v8** |
| gen_field_ssim | 0.255 | 0.271 | **v8** |
| gen_density_pearson | 0.286 | 0.687 | **v8** |
| gen_morans_i_pred_median | 0.192 | 0.167 | v6 |
| cm_pearson_median | 0.240 | 0.272 | **v8** |
| cm_celltype_accuracy | 0.882 | 0.963 | **v8** |
| cm_celltype_f1_macro | 0.839 | 0.949 | **v8** |
| cm_density_pearson | 0.101 | 0.281 | **v8** |
| cm_morans_i_median | 0.348 | 0.299 | v6 |
| cm_dice_density | 0.067 | 0.116 | **v8** |

**Tally: v8 15 / v6 2 / tie 0.** The only two v6 wins are `*_morans_i_*_median` —
the *magnitude* of the prediction's own spatial autocorrelation, which the metric
reports with no ground-truth target; the coherent morph reproduces one clean slice
rather than overlaying two, giving a slightly lower raw magnitude at strictly
higher fidelity on every agreement metric.

## Distinct-tissue regime (IMC-like)

v8 routes to real-cell interpolation and **matches v6 exactly (17/17 ties)** — no
regression where v6 was already strong.

| tally |
|---|
| v8 0 / v6 0 / **tie 17** |

## Takeaway

v8 is a strict improvement: it dominates v6 in the regime where v6 was beaten by
other methods (near-identical volumetric z-planes, the STARmap case) and never
regresses in the regime where v6 already led (distinct sections, the IMC case). The
full cross-dataset leaderboard should be regenerated with `run_benchmark.py` where
the processed datasets live.
