# T10 — Metrics, baselines, and the benchmark harness

**Goal.** Produce the paper's numbers: six target metrics plus unoptimised control metrics, across
two holdout regimes and all datasets, against the competing method and ablations, with proper
statistics.

**Files:** `spatialcpav25_gen/eval/metrics.py`, `spatialcpav25_gen/eval/baselines.py`, `spatialcpav25_gen/eval/benchmark.py`,
`spatialcpav25_gen/cli.py`, `tests/test_metrics.py`, `tests/test_baselines.py`

**Dependencies:** T01–T09.

---

## 1. Metrics — `spatialcpav25_gen/eval/metrics.py`

**Do not port, and do not reimplement.** The scoreboard is
`benchmark-pbya-v3/src/bench3/evaluate_paper.py`, and every published v20/v22 number came out of it.
A reimplementation that "agrees closely" is not comparable: the paper's claim is a *difference*
between methods measured on one instrument, and two instruments that agree to 1e-3 turn a 0.01 median
gap into an argument. So (settled, SPEC_QUESTIONS A3):

1. **Vendor or import it verbatim.** Either import `bench3.evaluate_paper` directly, or vendor the
   file into `spatialcpav25_gen/eval/_bench3_evaluate_paper.py` **byte for byte**, with a header
   comment saying where it came from and that it must not be edited. Fixing anything in it is a
   change to the scoreboard and needs its own decision, not a drive-by edit.
2. **Pin it with a content hash.** `eval/metrics.py` records
   `BENCH3_EVALUATE_PAPER_SHA256 = "7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992"`
   (`benchmark-pbya-v3/src/bench3/evaluate_paper.py` as of 2026-08-15, 764 lines) and checks it at
   import. A changed hash raises and names the file — silently scoring against a different instrument
   is exactly the failure this pin exists to prevent (Convention 6).
3. **Assert bit-identical output on fixed inputs.** `test_metrics_match_bench3_bitwise` runs both the
   wrapper and `evaluate_paper` on a small fixed synthetic pair and asserts every metric is `==`, not
   `allclose`. Any difference at all means the wrapper is doing something of its own.

`eval/metrics.py` is therefore a thin adapter: it maps our `AnnData` pairs onto `evaluate_paper`'s
call signature, unpacks its result dict into `METRIC_REGISTRY`, and adds the control metrics below.
It owns no metric arithmetic.

*Footnote on v20's two bugs.* `reference/learn_spatialcpav20.py` computes `gene_mean_spearman` /
`gene_var_spearman` with `np.corrcoef` (Pearson under a Spearman name, `:1876`) and rank-normalises
with `argsort`, which gives tied zeros distinct ranks (`:1810`). Both are real, and both matter for
reading v20's *internal* tuning signal — that is what its own development was steered by. Neither is
present in `bench3/evaluate_paper.py`, which already uses `scipy.stats.spearmanr` and
`rankdata(method="average")`, so there is nothing to fix on the scoreboard and no "bug fix" to apply
to the baselines. Say this in the paper's methods rather than claiming to have fixed the benchmark.

The six target metrics, as named by the scoreboard:

```python
def morans_pearson(gen, real) -> float        # r between per-gene Moran's I vectors
def gearys_pearson(gen, real) -> float
def umap_mixing(gen, real) -> float           # kNN mixing in a shared embedding
def marker_field_r(gen, real) -> float        # 2-D binned marker field agreement
def marker_depth_r(gen, real) -> float        # depth-profile agreement
def celltype_localization(gen, real) -> float # per-type spatial distribution agreement
```

⚠️ `marker_field_r` and `celltype_localization` do **not** exist in v20 under those names; they come
from `evaluate_paper`'s `marker_metrics` and `celltype_localization`. Map the names in the adapter
and record the mapping in the report, so a reader can trace a paper number back to the function that
produced it.

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

### The achievable ceiling — required for every metric (added at T05)

**A metric's stated range is not its achievable range.** Every one of these metrics compares a
*generated* section with a *real* one, so a perfect model — one that samples from exactly the right
distribution — still scores below the top of the scale, because a different **realisation** of the
same law is not the same point cloud. Measured at T05 for `celltype_localization` on the synthetic
fixture: the held-out section scored against itself reaches 0.9221, while an independent draw from
the fixture's own generative law reaches **0.7178**. A method scoring 0.71 there is not mediocre; it
is at **99%** of what is achievable, and reporting the raw number alone says the opposite.

So, on the synthetic fixture (the only dataset with a known generative law):

1. **Measure a ceiling for all six target metrics and all control metrics.** Generate the *ideal*
   arm by drawing from `tests.fixtures.synthetic`'s `GroundTruthField` directly — positions from the
   true intensity, marks from the true composition, expression from `expression_mu` + `sample_counts`
   — never from the trained model. Same held-out sections, same seeds, same metric code path.
2. **Report every method number twice**: raw, and as a fraction of that ceiling. This applies to the
   headline table, the **ablation** table and the **baseline** table alike — an ablation that costs
   0.02 raw on a metric whose ceiling is 0.72 has cost 3% of the achievable range, and that is the
   number a reader needs.
3. **Report the ceiling's own spread** across held-out sections and across seeds. It is a Monte-Carlo
   quantity: at least `Config.ceiling_n_draws` independent draws, mean and standard
   deviation, so a method-vs-ceiling gap can be read against the ceiling's own noise.
4. **A method above the ceiling is a finding, and usually a bug.** It means either the ideal arm is
   not drawing from the true law, or the method is copying real cells (check `duplicate_profile_rate`
   and `layout_mode`). T05 measured one such case legitimately — `field` mode scored 1.110× the
   ideal on one section, inside the ceiling's own per-section spread — which is exactly why 3 exists.
5. **Where a metric averages over parts, report per-part ceilings**, not only the aggregate. For
   `celltype_localization` that means **per cell type**: T05 measured the ceiling for the most
   abundant type (34% of cells) at a score of 0.33–0.84 across sections while localised minority
   types sat at 0.60–0.91, because the metric normalises by the divergence to a within-tissue null
   and that null collapses (`d_null` ≈ 0.08) for a type which is already spread tissue-wide. The
   abundant types are where the headroom is smallest and the variance largest, and a weighted
   average hides both. See `specs/05`'s "Why the criterion is a LOSO mean".

On the real datasets there is no generative law and therefore no ceiling. The referent there is the
**flanking-section baseline** — `run_nearest_copy`, which is what a real neighbouring section
achieves on the same held-out section — reported beside every metric for the same reason.

## 2. Baselines — `spatialcpav25_gen/eval/baselines.py`

```python
def run_spatialz(vol, target_z, cfg) -> AnnData      # wraps reference/SpatialZ.py
def run_nearest_copy(vol, target_z) -> AnnData       # floor
def run_convex_interp(vol, target_z) -> AnnData      # smooth ceiling / realism floor
def run_independent_donor(vol, target_z, cfg) -> AnnData   # from T06; isolates chimerism
def run_v20(vol, target_z, cfg) -> AnnData           # previous version
```

**v14 and v18 are dropped as baselines** (settled; they are listed in `design/v23_design.md` §7).
Reason: both are superseded by v20 on every metric of the existing bench3 campaign, so they add two
more columns without adding a comparison anyone would read — v20 is the version the no-regression
guarantee is stated against, and it is the one that has to be beaten. Say so in one line in the
paper's methods rather than leaving their absence unexplained.

For the competing method, use its published defaults (`syn_mode='default'`, `k_sam=3`,
`k_neighbors=1`, `nb_iter_max=3000`, `num_projections=80`) and its own MENDER-based niche pipeline.
Do not tune it; do not cripple it. Record the exact settings in the report — reviewers check this.

⚠️ It mutates `adata.obs_names` in place (appends slice ids). **Deep-copy inputs before calling it**
or subsequent baselines silently receive corrupted data. This has bitten people before.

For the alternating/consecutive regimes, generate at the same `alpha` positions the held-out sections
occupy, so the comparison is like-for-like.

### E1 must load the gene table with its species, and report the table's own coverage

`load_gene_meta(path, species=...)` **raises** on a table of the wrong organism (added after a mouse
panel's table came back holding four other mammals' genes and nothing noticed). E1 must pass
`Config.mygene_species`, and the headline text must quote the table's own coverage —
`gene_meta_summary` reports rows, resolved taxid, how many rows carry a summary, and the Ensembl-id
prefix histogram — because "the model decodes unseen genes at r = X" means nothing without knowing
that the descriptors were real. **One table per organism**, at different `Config.gene_meta_path`s: the
mouse and human datasets do not share one.

### ⛔ The gene–gene covariance comparison is a LOSS as of T06. Framing rule for the paper.

`run_independent_donor` exists to isolate chimerism, and T06 measured both halves of that comparison.
They point opposite ways and the write-up must say so:

| what | status |
|---|---|
| **the mechanism** — per-gene independent draws destroy covariance, a shared latent cannot | **established.** Donors held fixed, draw varied: retained |off-diag| 0.978 / 0.920 / 0.897 / 0.884 / 0.844 at D = 1/2/3/5/10, monotone, on both holdout regimes |
| **the model beating the baseline on the correlation matrix** | **NOT established — it loses.** Frobenius error: model **9.316**, independent-donor **7.783**, nearest-copy 6.743, achievable ceiling **5.601**. Worse at `consecutive-3` (17.7 vs 11.3) |

**Until T08's `test_metric_losses_close_the_covariance_loss` passes on both regimes, no headline
table, figure, abstract or methods sentence may claim that this method preserves gene–gene covariance
*better than* the competing method.** The claim it may make is the mechanism claim, and the chimerism
table is the evidence for it. If T08 closes the loss, this section is updated with the numbers that
closed it; if T08 cannot, the paper makes the smaller claim and says why — that is a decision to
record in `PROGRESS.md`, not a gap to leave ambiguous.

The reason this needs writing down rather than trusting: T06's *first* reading of its own measurement
found a decomposition on which the model did win by 2.2×, and it took an out-of-sample check
(`consecutive-3`, ratio 0.995) to establish that the decomposition had been chosen after seeing which
component passed. The same temptation will exist when this table is assembled.

## 3. Harness — `spatialcpav25_gen/eval/benchmark.py`

```python
def run_benchmark(datasets, methods, regimes, folds, out_dir) -> pd.DataFrame
```

**Dataset requirement (settled; from `design/v23_design.md` §7, previously unstated in `specs/`).**
The campaign must include **at least one non-brain dataset** (embryo or tumour) and **at least one
non-transcriptomic panel** (e.g. EASI-FISH). This is a reviewer defence, not a nicety: every version
of this project so far has been tuned on brain sections with a transcriptomic panel, and a method
whose oblique-sectioning claim rests on laminar structure would look excellent on brain and fail
silently elsewhere. `run_benchmark` refuses to produce a headline table unless both are present, and
names what is missing (Convention 6). A campaign run without them is a development run, and the
report says so on its first line.

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
| A4 | repulsion off (Poisson layout) | point-process realism — **the `g(r)` comparison must run over `[0, 3R]`, see below** |
| A5 | `w_z=0` in retrieval | the specific competing-method flaw — **must be run in the wide-gap regime, see below** |
| A6 | Gaussian mean decoder | sparsity/dispersion preservation |
| A7 | `w_cross=w_thick=w_prog=0` | SEFL's contribution |
| A8 | `loss_prog_WRONG` enabled | **negative control** — wrongly constraining equivariant quantities should be *worse* |

### A4's pair-correlation comparison must run over `[0, 3R]` (measured at T05)

**Do not report A4 against `g(r)` restricted to `[r0, 3R]`.** T05 originally stated the
pair-correlation criterion over that range and it **cannot fail**: a hard-core process differs from a
Poisson one only *inside* the correlation hole, and the hole ends at about `r0`, because `r0` **is**
a low percentile of the nearest-neighbour distances. Measured on the synthetic fixture, with the real
`g` pooled over the training sections and the simulated one over three seeds:

| Range | `field` mode | pure Poisson (A4) |
|---|---|---|
| `[r0, 3R]` | 0.093 | **0.070 — indistinguishable from the full model** |
| `[0, 3R]` | 0.093 | **0.994** |

Reported over `[r0, 3R]`, A4 is a **false null**: the ablation table would say the repulsion buys
nothing while `g(r)` below `r0` says it is the difference between tissue and confetti. `specs/05` is
amended to `[0, 3R]` and `tests/test_layout.py` asserts both ranges, the second of them precisely so
that the blindness stays visible. Any A4 number in `reports/benchmark.md` states its range.

The same caution applies to every A4 companion metric: choose statistics that can see inside the
hole (nearest-neighbour distance distribution, `g(r)` from 0) rather than ones evaluated only where
the two processes agree by construction.

### A5 must be run in the wide-gap regime (measured at T04's GATE 2)

**...and not at a fixed `retrieval_z_window`** (measured at T06). On the `consecutive-3` holdout the
default window of 3 × median spacing leaves 100–110 of every 512 cells with **no admissible donor at
all** after the own-section exclusion, so the retrieval branch is silently absent for a fifth of them
and an ablation of `retrieval_w_z` would be measuring the window. `specs/09` §1 requires the window to
be derived from the gap; A5 must be run against the derived window, with the empty-pool fraction
reported beside the ablation delta. This is the same trap G2.3 fell into with
`retrieval_candidates_per_section` and recorded — the ablation read as a no-op, with the wrong sign,
until the cap was raised.

**Do not report A5 from the `alternating` holdout, and do not report it with the whole stack
admissible.** GATE 2's G2.3 measured the ablation both ways on the synthetic fixture, with the two
arms sharing a training seed so that initialisation, batch order and per-step rotations were
identical and the retrieval score was the only difference:

| Candidate pool | R² lost by `w_z = 0` at fractional depth 0.2 / 0.5 / 0.8 |
|---|---|
| Two flanking sections, the near one 1 spacing away and the far one 4 (the wide-gap regime) | **+0.0303 / +0.0034 / +0.0486** |
| Whole stack admissible | +0.0004 / +0.0034 / +0.0019 — **inside the noise** |

The reason is mechanical, not statistical. With every section admissible, the *nearest* section is
always in the pool and in-plane distance alone already ranks it first, so the z term has nothing
left to decide. It earns its place only when the evidence is far and asymmetric — which is the
regime in-silico sectioning actually lives in, and the one the competing method's score cannot see.
Run whole-stack and A5 reports a **null result for a term that demonstrably works**, which would be
a false negative in the paper's own ablation table.

Concretely, A5 must be reported at `consecutive-3` and `consecutive-5` (the regimes where the gap to
the nearest real section is 2–3 spacings), and `reports/benchmark.md` must state which holdout
regime each A5 number came from. Reporting it at `alternating` as well is fine as a second row,
labelled as the dense-evidence control, but it is not the headline.

**Check `retrieval_candidates_per_section` before trusting any A5 number.** The invariant that makes
the retrieval score do anything at all is about the candidate **union**:
`candidates_per_section × n_admissible_sections` must exceed `retrieval_k`, or the top-K returns the
whole pool and the score decides nothing. A wide-gap holdout is exactly where the number of
admissible sections is smallest, so this is exactly where it bites. `Config.validate` enforces
`retrieval_candidates_per_section >= retrieval_k` and `RetrievalIndex.query` warns at runtime
(`InertScoreWarning`) when a query's union falls to `K` or below. **An A5 run that emits that
warning is void.**

### Stratify every headline metric by distance to the stack boundary

**Open risk R3, raised at T04.** The T04 probe reconstructed the two **edge** sections at R²
**0.2912** and **0.3642** against an interior mean of **0.4474**. The cause is one-sided evidence at
the volume boundary, it is a property of serial sectioning rather than of the fixture, and it is
large enough to move a pooled average.

Two consequences for this task:

1. **Report boundary and interior separately** for the six `paper_*` metrics, not just pooled. A
   method that is strong in the interior and weak at the ends is a different claim from one that is
   uniformly mediocre, and the pooled number cannot distinguish them. If the baselines degrade at the
   boundary too — SpatialZ interpolates between flanking slices and has no flanks there either — that
   comparison is itself a result worth a row.
2. **Check the holdout regimes for boundary loading.** `alternating` holds out interior sections by
   construction (`split_holdout` never holds out the first or last), so it under-samples the regime
   where the model is weakest, while `consecutive-5` on a short stack pushes the held-out run close
   to an end. State which regime each headline number came from and how much boundary tissue it
   contained; otherwise a regime change silently moves the metric.

## 5. Capability experiments

```python
def exp_zero_shot_genes(...)      # E1: hold out 20% of genes entirely (both arms, see below)
def exp_cross_panel(...)          # E2: train on A, generate B's panel
def exp_oblique_validation(...)   # E3: vs. orthogonally-sectioned specimen
def exp_throughput(...)           # E4: 10x z-density, recover fine 3D structure
def exp_intersection_agreement(...)# E5: mutual coherence vs. the competing method
```

**E1 reports both arms** (settled; `design/v23_design.md` §2.2 / §7, previously unstated here). A
held-out gene has no learned residual `r_g`, so the zero-shot table must show *both*
`forward_zero_shot(use_distill=False)` — the pure-text arm, `r_g = 0` — and `use_distill=True`, the
distilled `r_g = psi(t_g)`. Both exist and are shape-tested in T02. One arm alone cannot separate
"the text channel carries the gene" from "the distillation head guessed a residual", which is the
whole claim of open-vocabulary generation.

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

- `test_metrics_match_bench3_bitwise` — the adapter and `bench3.evaluate_paper` return `==` values,
  not `allclose`, on a fixed synthetic pair; and the pinned SHA-256 of `evaluate_paper.py` matches.
- `test_metrics_match_reference_after_fixes` — *superseded by the above* (SPEC_QUESTIONS A3): the
  reference is bench3, not v20, and agreement with it is asserted bitwise. Kept named here only so a
  reader of the original spec can find where it went. Each metric reproduces the v20 implementation on
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
