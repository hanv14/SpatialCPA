# `v21_<learner>` — the marginal cost, measured

**2026-09-13.** The bill for the expression ablation, before it runs. Supersedes an
earlier extrapolation that was **roughly an order of magnitude too high** on the
learner that dominates: it was built from a guessed constant, this is built from a
measured one.

Nothing here covers v21's own cost. Every variant runs a complete v21 —
layout, donor selection and the flow are unchanged, so Phase A (60 epochs) plus
Phase B (160) is paid in full, 85 times (5 learners × 17 datasets). **That is the
real bill and it is not measured here**; per-run wall time is recorded as `wall`
in each existing `spatialcpav21_gen/**/prediction.h5`. Everything below is the
*marginal* cost of replacing the expression step.

## Measured

scikit-learn 1.9.1, 4 cores, dense float64 features (F = 16), fit seconds, at the
campaign's shipped hyper-parameters (`rf_trees=100`, `gbm_iters=100`,
`mlp_hidden=256`, `knn_k=15`):

| n cells | G genes | ridge | knn | rf | gbm | mlp |
|---:|---:|---:|---:|---:|---:|---:|
| 5 000 | 28 | 0.01 | 0.00 | 0.85 | 1.82 | 2.06 |
| 5 000 | 500 | 0.04 | 0.00 | 19.58 | 6.38 | 3.99 |
| 16 500 | 28 | 0.01 | 0.00 | 3.62 | 0.69 | 4.85 |
| 16 500 | 500 | 0.08 | 0.01 | **91.80** | 10.46 | 9.98 |
| 8 000 | 3 000 | 0.15 | 0.03 | **324.39** | — | 23.60 |

⚠️ **`gbm` and `mlp` are FLOORS, not estimates.** Both use `early_stopping=True`
and the synthetic targets here are pure noise, so stopping fires almost
immediately — visible in `gbm` going *down* from 1.82 s to 0.69 s as `n` rose
6.4×, which is not a scaling law, it is a learner giving up sooner. On real
expression they will fit many more iterations. Treat their real cost as some
multiple of the table, and **measure it on the first dataset** rather than
modelling it. `rf` needs no such caveat: it grows all 100 trees regardless of
signal, so its numbers transfer.

## Scaling, checked rather than assumed

`rf`, the dominant learner:

* in `n`: 0.85 → 3.62 (3.3× n, 4.3× time) and 19.58 → 91.80 (3.3× n, 4.7× time)
  — consistent with `n log n`.
* in `G`: 0.85 → 19.58 (17.9× G, 23× time). Roughly linear with a ~1.3×
  superlinear term, as expected: sklearn evaluates MSE impurity over all `G`
  outputs at every split candidate. The `8 000 × 3 000` cell measures 324 s where
  pure linearity predicts ~206 s, so **use the measured points, not the law**, at
  wide panels.

`ridge` and `knn` are free at every size in the benchmark and need no budgeting.

## Projected per dataset, `rf` (the worst case)

Interpolated from the two large measured points. Cell counts for anything but
STARmap are estimates until `describe_datasets` runs.

| dataset class | n, G | `rf` fit |
|---|---|---|
| STARmap (exact: 16 527 train, 28 genes) | 16.5 k × 28 | **~4 s** |
| small-panel analogues (ExSeq, IMC, EASI-FISH, MERFISH-thick) | ≤ 50 k × ≤ 300 | seconds to ~2 min |
| CosMx | ~50 k × 960 | ~10 min |
| ST / Visium (`n_hvg` 3 000) | ~5–8 k × 3 000 | ~5 min (measured) |
| Allen ×3 (`max_cells_per_section` 20 000) | ~160 k × 500 | ~17 min |

**Total marginal, all five learners × 17 datasets: ~2–6 CPU-hours on 4 cores**,
dominated by `rf` and `gbm`, with `gbm`'s share the uncertain half for the reason
above. Proportionally less on a larger box. The earlier 40–60 hour figure is
withdrawn.

`openst_lymph_node` is excluded — it is uncapped whole-transcriptome and already
OOM-kills the v14 family (bench3 README, "Known gap: Open-ST is still uncapped"),
which these variants inherit. 17 datasets, not 18.

## Two non-CPU costs, which do not shrink

**Disk.** v21 emits sparse donor copies; a squared-loss regressor emits the
conditional mean, which has **no zeros** — measured at density 1.000 for all five
learners on the smoke fixture. `write_prediction_h5` stores CSR, so a fully dense
`Q × G` block costs roughly 3× its dense size. Allen: `Q ≈ 140 k × 500` → ~840 MB
*per variant per dataset*. Budget **~15–20 GB** of new `prediction.h5` across the
campaign, against a few hundred MB for the existing tree.

**Evaluation.** 85 new predictions through `evaluate_all`, at the same per-run cost
the existing methods pay. `specs/10` §13.1a is explicit that no per-prediction
scoring time exists in this project's artifacts and that it must be measured
rather than modelled — that applies here unchanged. Keep UMAP **on**; `--no-umap`
is what left `paper_umap_mixing` unmeasured on the r11 arms.

## Reproducing this table

```bash
python - <<'PY'
import numpy as np, time
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
rng = np.random.default_rng(0)
print(f"{'n':>7} {'G':>5} | {'ridge':>7} {'knn':>7} {'rf':>8} {'gbm':>9} {'mlp':>7}")
for n, G in ((5000,28),(5000,500),(16500,28),(16500,500),(8000,3000)):
    X = rng.normal(size=(n,16)); Y = np.abs(rng.normal(size=(n,G))); row=[]
    for name, mk in (
        ("ridge", lambda: Ridge(alpha=1.0)),
        ("knn",   lambda: KNeighborsRegressor(15, weights="distance", n_jobs=-1)),
        ("rf",    lambda: RandomForestRegressor(100, max_features="sqrt",
                          min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ("gbm",   lambda: MultiOutputRegressor(HistGradientBoostingRegressor(
                          max_iter=100, early_stopping=True, random_state=42), n_jobs=-1)),
        ("mlp",   lambda: MLPRegressor(hidden_layer_sizes=(256,256), max_iter=300,
                          early_stopping=True, random_state=42))):
        if name == "gbm" and G > 500: row.append(float("nan")); continue
        t = time.time(); mk().fit(X, Y); row.append(time.time()-t)
    print(f"{n:>7} {G:>5} | {row[0]:7.2f} {row[1]:7.2f} {row[2]:8.2f} {row[3]:9.2f} {row[4]:7.2f}")
PY
```

Note `MLPRegressor(hidden_layer_sizes=...)` must be passed by keyword: in
scikit-learn 1.9 the first positional parameter is `loss`, so a positional
`(256, 256)` raises `InvalidParameterError`. The wrapper already uses the keyword.
