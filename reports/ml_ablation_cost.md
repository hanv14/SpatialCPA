# `v21_<learner>`, `v14_<learner>`, `v18_<learner>` — the marginal cost, measured

**2026-09-13.** The bill for the expression ablation, before it runs. Supersedes an
earlier extrapolation that was **roughly an order of magnitude too high** on the
learner that dominates: it was built from a guessed constant, this is built from a
measured one.

Covers **all three** ablation families. The learners, features and swap point are
identical between them — enforced in code, since all three wrappers import
`methods/_ml_learners.py` — so the measured table applies to each; only the host
method and the target scale differ. Three families of **seven** learners over 17
datasets is **357 runs**.

Nothing here covers the host method's own cost. Every variant runs a complete v21
(or v14) —
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

**`lasso` and `xgb`, measured on the same box.** `lasso` is free — it is a
coordinate-descent linear fit on 16 features, in the same class as `ridge`
(< 1 s at every size in the benchmark). `xgb` is **the new dominant cost**, well
above `rf`: at n = 4000 it takes **7.97 s at G = 40** and **64.18 s at G = 300**,
which is linear in G (8.05× time for 7.5× genes) and several times what
scikit-learn's `gbm` costs at comparable size. Budget for it accordingly — the
projections below put it at roughly an hour per Allen atlas.

`--xgb-multi-strategy` was measured, not guessed, and the guess was wrong:
`multi_output_tree` (one ensemble of vector-leaf trees) is **slower**, not
faster — 24.52 s vs 7.97 s at G = 40, and 238.90 s vs 64.18 s at G = 300. So
`one_output_per_tree` is the default. The alternative stays available as a flag
because the two are different models, not two implementations of one.

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

`ridge`, `lasso` and `knn` are free at every size in the benchmark and need no
budgeting.

**One thing to check on a `lasso` row before reading it.** The penalty is set as
`--lasso-alpha-frac × alpha_max`, where `alpha_max = max|X'y|/n` is the smallest
penalty that zeroes every coefficient. This is deliberate: a fixed `alpha` is not
portable here, and fails silently — at scikit-learn's conventional `alpha=1.0`,
**0 of 640 coefficients survive** (R² = 0.000), so the row becomes the per-gene
training mean and still scores, looking like a fitted method. The fractional form
also means one number works for v14/v21's log targets and v18's raw EASI-FISH
intensities, which differ by ~10³. At the default `0.01` the fit keeps **98.1 %**
of coefficients, so `lasso` will land close to `ridge` — that was the prediction
when it was argued against, and it is now quantified rather than asserted. Raise
the fraction if you want genuine sparsity. Every run prints its non-zero
fraction, and an all-zero fit raises a run-time warning rather than being
silently ranked.

## Projected per dataset, `rf` (the worst case)

Interpolated from the two large measured points. Cell counts for anything but
STARmap are estimates until `describe_datasets` runs.

| dataset class | n, G | `rf` fit | `xgb` fit |
|---|---|---|---|
| STARmap (exact: 16 527 train, 28 genes) | 16.5 k × 28 | **~4 s** | ~30 s |
| small-panel analogues (ExSeq, IMC, EASI-FISH, MERFISH-thick) | ≤ 50 k × ≤ 300 | seconds to ~2 min | ~1–10 min |
| CosMx | ~50 k × 960 | ~10 min | ~50 min |
| ST / Visium (`n_hvg` 3 000) | ~5–8 k × 3 000 | ~5 min (measured) | ~25 min |
| Allen ×3 (`max_cells_per_section` 20 000) | ~160 k × 500 | ~17 min | ~1.5 h |

`xgb` is interpolated from the two measured points above (linear in G, ~n log n
in n); `rf` from its own. Both are 4-core numbers.

**Total marginal, seven learners × 17 datasets: ~8–15 CPU-hours on 4 cores**,
now dominated by `xgb`, then `rf` and `gbm` — `gbm`'s share is the uncertain one,
for the reason above. **Triple it for all three families: ~25–45 CPU-hours.**
Proportionally less on a larger box. (The original 40–60 hour estimate, withdrawn
as ~10× too high for five learners, lands near the right magnitude again once
XGBoost is in — for a different reason, and this time from measurements.)

`openst_lymph_node` is excluded — it is uncapped whole-transcriptome and already
OOM-kills the v14 family (bench3 README, "Known gap: Open-ST is still uncapped"),
which these variants inherit. 17 datasets, not 18.

## Two non-CPU costs, which do not shrink

**Disk, and one dataset that cannot run at all.** v14 and v21 emit sparse donor
copies; a squared-loss regressor emits the conditional mean, which has **no
zeros** — measured at density 1.000 for all five learners. `write_prediction_h5`
stores CSR, so every entry costs its value plus its column index on top of a dense
`Q × G` array that must be materialized first.

Both wrappers now estimate this **before training** (`check_output_size`, default
limit `--max-dense-gb 8`) and refuse rather than discovering it after the flow has
trained. Measured against the real dataset shapes:

| dataset | est. Q × G | dense | as CSR |
|---|---|---|---|
| `starmap_visual_cortex` | 12 419 × 28 | 0.00 GB | 0.00 GB |
| `st_mouse_brain_ortiz` | 7 000 × 3 000 | 0.08 GB | 0.17 GB |
| `cosmx_nsclc_3d` | 20 000 × 960 | 0.08 GB | 0.15 GB |
| `allen_*` (each) | 140 000 × 500 | 0.28 GB | 0.56 GB |
| **`openst_lymph_node`** | **473 684 × 20 000** | **37.9 GB** | **75.8 GB** |

So budget roughly **20 GB per family** of new `prediction.h5` at seven learners —
about 60 GB for all three — and treat
`openst_lymph_node` as **out of scope for these variants**. It is uncapped
whole-transcriptome, and at ~20 000 genes a dense prediction is tens of gigabytes
per learner per dataset. The three ways out, in the order I would take them:

1. Leave it out and say so — the targeted panels answer the question.
2. Build a gene-capped copy (`prepare_dataset --n-hvg 3000`) registered as a
   **separate** dataset; capping in place changes the panel that previously
   reported `openst` rows were measured on.
3. Raise `--max-dense-gb` deliberately, with the disk to back it.

**Never** threshold small predictions to zero to shrink the file: that
contaminates `paper_gene_detection_spearman`, which is the column the whole
ablation exists to read.

## One asymmetry between the families, on exactly that column

v14 and v21 emit through `expm1`, which cannot produce a zero, so their density is
a flat **1.000** — the clean signal that a squared-loss regressor models no
sparsity at all.

The `v18_*` variants emit on the **raw** scale (mirroring v18's own `raw_ok` path
at `learn_spatialcpav18.py:1240`), and the raw path clips negative predictions to
zero. That *manufactures* zeros: measured density **0.86–1.00** across the five
learners on a fixture whose real zero fraction was 0.39 — ridge 0.963, knn 0.999,
rf 1.000, gbm 0.999, mlp 0.864.

So a `v18_*` detection score will look less catastrophic than a `v14_*` or
`v21_*` one, **and the difference is the clip, not the learner.** Do not compare
`paper_gene_detection_spearman` across families without saying so, and do not read
a v18 row as the regressor recovering sparsity. `--target-scale log` routes the
prediction through `expm1` instead and is the cross-check when the distinction
bears on a claim.

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
