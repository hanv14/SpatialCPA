# T10 — Metrics, baselines, and the benchmark harness

**Goal.** Produce the paper's numbers: six target metrics plus unoptimised control metrics, across
two holdout regimes and all datasets, against the competing method and ablations, with proper
statistics.

**Files:** `spatialcpav25_gen/eval/metrics.py`, `spatialcpav25_gen/eval/baselines.py`, `spatialcpav25_gen/eval/benchmark.py`,
`spatialcpav25_gen/cli.py`, `tests/test_metrics.py`, `tests/test_baselines.py`

**Dependencies:** T01–T09.

---

## 1. Metrics — `spatialcpav25_gen/eval/metrics.py`

Port the six target metrics from `reference/learn_spatialcpav20.py`, **fixing two bugs found in that
implementation**:

1. **`gene_mean_spearman` / `gene_var_spearman` are computed with Pearson correlation
   (`np.corrcoef`), not Spearman.** Either compute a real Spearman via `scipy.stats.spearmanr`, or
   rename to `_pearson`. Do not silently keep the mismatch — any published number under that name
   would be wrong.
2. **`rank_normalize` uses `argsort`, which assigns distinct ranks to tied values.** With sparse
   count data the mass of zeros gets an arbitrary rank spread, differently in prediction vs. truth.
   Use `scipy.stats.rankdata(method="average")`. This directly affects the mixing and autocorrelation
   comparisons on sparse data.

Both fixes must be applied to **all methods equally**, including the baselines, and noted in
`PROGRESS.md` and the paper's methods.

The six target metrics:

```python
def morans_pearson(gen, real) -> float        # r between per-gene Moran's I vectors
def gearys_pearson(gen, real) -> float
def umap_mixing(gen, real) -> float           # kNN mixing in a shared embedding
def marker_field_r(gen, real) -> float        # 2-D binned marker field agreement
def marker_depth_r(gen, real) -> float        # depth-profile agreement
def celltype_localization(gen, real) -> float # per-type spatial distribution agreement
```

**Unoptimised control metrics** — required for paper integrity, since six of the metrics are trained
against (T08). Report at least five:

```python
def sinkhorn_profile_distance(gen, real) -> float
def coexpression_module_preservation(gen, real) -> float
def neighbourhood_enrichment_agreement(gen, real) -> float
def gene_variance_rank_corr(gen, real) -> float
def duplicate_profile_rate(gen) -> float        # fraction of exactly-repeated profiles
def detection_rate_agreement(gen, real) -> float
```

Every metric: fixed random seeds, documented normalisation, and a docstring stating whether higher
is better and its range. Build `METRIC_REGISTRY: dict[str, MetricSpec]` so the harness is
data-driven.

## 2. Baselines — `spatialcpav25_gen/eval/baselines.py`

```python
def run_spatialz(vol, target_z, cfg) -> AnnData      # wraps reference/SpatialZ.py
def run_nearest_copy(vol, target_z) -> AnnData       # floor
def run_convex_interp(vol, target_z) -> AnnData      # smooth ceiling / realism floor
def run_independent_donor(vol, target_z, cfg) -> AnnData   # from T06; isolates chimerism
def run_v20(vol, target_z, cfg) -> AnnData           # previous version
```

For the competing method, use its published defaults (`syn_mode='default'`, `k_sam=3`,
`k_neighbors=1`, `nb_iter_max=3000`, `num_projections=80`) and its own MENDER-based niche pipeline.
Do not tune it; do not cripple it. Record the exact settings in the report — reviewers check this.

⚠️ It mutates `adata.obs_names` in place (appends slice ids). **Deep-copy inputs before calling it**
or subsequent baselines silently receive corrupted data. This has bitten people before.

For the alternating/consecutive regimes, generate at the same `alpha` positions the held-out sections
occupy, so the comparison is like-for-like.

## 3. Harness — `spatialcpav25_gen/eval/benchmark.py`

```python
def run_benchmark(datasets, methods, regimes, folds, out_dir) -> pd.DataFrame
```

- Regimes: `alternating`, `consecutive-3`, `consecutive-5`. Report **separately** — the expected
  story is "ties or wins at narrow gaps, wins decisively at wide gaps", and averaging destroys it.
- Long-format output: one row per (dataset, regime, fold, section, method, metric, value). Everything
  downstream is a groupby.
- Cache per-(dataset, method, fold) generations to disk so metrics can be recomputed without
  regeneration.
- Resumable: skip completed cells; a benchmark that cannot resume will not survive a 3-day run.

**Statistics** (`benchmark.py` or `stats.py`):
- Paired Wilcoxon signed-rank vs. the competing method, per metric, paired by section.
- Benjamini–Hochberg across the six metrics.
- **Median difference with 95% bootstrap CI** (10 000 resamples, stratified by dataset) — this is
  the "clear gap in medians" claim, stated defensibly.
- Cliff's delta as a nonparametric effect size.
- Forest plot per metric: median difference ± CI, one row per dataset. This is paper Figure 2.

## 4. Ablations

Wire as config overrides so each is a one-line entry:

| ID | Override | Claim tested |
|---|---|---|
| A1 | `prior_mode=iid` | correlated prior preserves autocorrelation |
| A2 | `w_autocorr=w_profile=w_distribution=0` | contribution of metric-aware training |
| A3 | `text_emb=lookup-only` | text channel's value on seen genes |
| A4 | repulsion off (Poisson layout) | point-process realism |
| A5 | `w_z=0` in retrieval | the specific competing-method flaw |
| A6 | Gaussian mean decoder | sparsity/dispersion preservation |
| A7 | `w_cross=w_thick=w_prog=0` | SEFL's contribution |
| A8 | `loss_prog_WRONG` enabled | **negative control** — wrongly constraining equivariant quantities should be *worse* |

## 5. Capability experiments

```python
def exp_zero_shot_genes(...)      # E1: hold out 20% of genes entirely
def exp_cross_panel(...)          # E2: train on A, generate B's panel
def exp_oblique_validation(...)   # E3: vs. orthogonally-sectioned specimen
def exp_throughput(...)           # E4: 10x z-density, recover fine 3D structure
def exp_intersection_agreement(...)# E5: mutual coherence vs. the competing method
```

**E5 is the cheapest and most decisive.** Generate two intersecting oblique sections with each
method and measure agreement along the intersection line as a function of dihedral angle. The
competing method optimises each slice independently, so its two sections have no mechanism forcing
agreement where they cross; ours share one 3D noise field and are trained for it. Expect a
categorical rather than incremental gap. One panel, minimal compute — run it early, as soon as T09
lands, because it is the figure that establishes the contribution is structural.

## 5b. SEFL validations (V1–V4)

These validate the sectioning-equivariance claims specifically. They are what justify SEFL as a
scientific contribution rather than a regulariser, so they are not optional.

```python
def val_resectioning_cycle(...)     # V1
def val_orthogonal_specimen(...)    # V2
def val_anisotropy_prediction(...)  # V3
def val_thickness_transfer(...)     # V4
```

**V1 — virtual re-sectioning cycle.** From a coronally-sectioned volume, generate a full sagittal
stack; treat that generated stack as input and regenerate the original coronal sections; compare to
the real ones. End-to-end and ground-truthed, and it cannot be passed by memorisation because the
intermediate representation is entirely synthetic. Report the six target metrics on the round trip
and compare against a single-pass generation as the ceiling. Degradation over the cycle is the
quantity of interest — report it, do not hide it.

**V2 — orthogonal-specimen validation.** Train on a coronally-sectioned specimen, generate sagittal
sections, compare against a *different* specimen actually sectioned sagittally. Comparison must be
**distribution-level** (Sinkhorn on cell-state distributions, laminar profile agreement, cell-type
localization), never per-cell — the specimens are different animals. Overlaps with E3; implement
once and reference from both.

**V3 — anisotropy prediction (the equivariant-column payoff).** From the fitted 3D covariance
structure, *predict* how in-plane Moran's I should vary with section angle; verify against real
sections cut at different angles. This is the correct use of the quantities T07 forbids constraining:
they are predicted, not matched. A model that had merely memorised a stack of 2D fits cannot pass
this. Report predicted-vs-observed r across angles.

**V4 — thickness transfer.** Train on thin sections and predict thick-section (spot/bin-level) data,
and the reverse. Validates `L_thick` and supports the cross-technology harmonisation claim. Metric:
agreement of binned expression totals and per-type counts. Include an ablation with `w_thick=0` to
show the loss is what buys the transfer.

## 6. CLI — `spatialcpav25_gen/cli.py`

```
spatialcpav25_gen fit      --data X.h5ad --out runs/foo          # includes select_config + calibration
spatialcpav25_gen generate --run runs/foo --plane oblique --angle 45 --n 20 --out slices.h5ad
spatialcpav25-gen bench    --config bench.yaml --out reports/
spatialcpav25_gen report   --results reports/results.parquet --out reports/figures/
```

`fit` takes **no method flags** — configuration is selected internally (T09 §3). That is a claim in
the paper; make sure it is literally true of the CLI.

## Acceptance tests

- `test_metrics_match_reference_after_fixes` — each metric reproduces the v20 implementation on
  fixed inputs *except* the two documented bug fixes, which are asserted to differ in the expected
  direction.
- `test_rankdata_ties` — a vector with 60% zeros gets identical average ranks for all zeros.
- `test_spearman_is_spearman` — a monotone-nonlinear transform leaves the value unchanged.
- `test_spatialz_wrapper_no_mutation` — input AnnData is byte-identical after the call.
- `test_baselines_run_on_fixture` — all five produce valid AnnData.
- `test_benchmark_resumable` — kill and restart; results identical, completed cells skipped.
- `test_stats_bootstrap_ci` — on synthetic data with a known median difference, the CI covers truth
  in ≥ 94% of 200 simulations.
- `test_metric_registry_complete` — all six target and ≥ 5 control metrics registered with
  direction and range.

## Definition of done

`reports/results.parquet` + a rendered `reports/benchmark.md` with: the six-metric table by regime,
forest plots, the ablation table (A1–A8), the control-metric table, the E5 intersection-agreement
figure, and the V1–V4 validation results. `PROGRESS.md` records the headline median gaps and the V1
cycle degradation.

## Do NOT

- Do not tune the competing method's hyperparameters, in either direction.
- Do not average across regimes.
- Do not report only the metrics that were trained against — the control table is not optional.
- Do not let the two metric bug-fixes apply to some methods and not others.
