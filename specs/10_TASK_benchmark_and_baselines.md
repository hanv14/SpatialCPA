# T10 — Scoring SpatialCPA-v25-Gen on bench3

**Goal.** Produce the paper's numbers on the **existing** instrument. `benchmark-pbya-v3` is a
complete benchmark — the SpatialZ STARmap paper protocol, the leakage machinery, the metrics, the
published baselines, aggregation, ranking and the Nature-themed figures. T10 does **not** build a
benchmark. It adds a wrapper, a driver that runs bench3 from outside, the statistics bench3 lacks,
a preprocessor for the derived datasets the extension claims need, and the experiments.

**Files (all new, all ours):** `spatialcpav25_gen/eval/metrics.py`, `eval/baselines.py`,
`eval/bench3_driver.py`, `eval/resection.py`, `eval/stats.py`, `eval/ceiling.py`,
`eval/experiments.py`, `spatialcpav25_gen/cli.py`, plus `tests/test_metrics.py`,
`tests/test_baselines.py`, `tests/test_bench3_driver.py`, `tests/test_resection.py`,
`tests/test_stats.py`.

**Dependencies:** T01–T09.

---

## 0a. ⚠️ READ FIRST — which dataset can carry a reconstruction claim, and which cannot

**Measured 2026-08-27, model-free, and it inverts how this campaign has been prioritised.**
`reports/t09_depth_ceiling_{starmap,deep}.md` and `reports/t09_ceiling_bootstrap_{starmap,deep}.md`.

Every reconstruction metric compares a generated section with a real one, and §2's *achievable
ceiling* argument says the top of the scale is not reachable. The ceiling is now **measured on
real data** rather than argued, and with it the **headroom**: how much a perfect, noiseless method
could beat a copy of another real section — which is what the `resample` + `cross-mix` baseline
is (R13: it *is* that copy, to −0.0009 / +0.0099 on `deep_starmap`).

| | tier-1 `starmap_visual_cortex` (28 genes, ~4.1k cells/section) | `deep_starmap` (1017 genes, 18–39k cells/section) |
|---|---|---|
| split-half reliability R | 0.859 – 0.882 | 0.987 – 0.992 |
| noiseless ceiling √R | 0.927 – 0.939 | 0.994 – 0.996 |
| best available copy | 0.713 – 0.784 | 0.929 – 0.986 |
| **headroom over the best copy** | **+0.1551 [+0.1075, +0.2186]** | **+0.0160 [+0.0104, +0.0243]** |
| headroom over the **operational** copy | +0.1833 [+0.1274, +0.2567] | +0.0855 [+0.0489, +0.1351] |
| copying, as a share of the ceiling | 81 % | **98 %** |

400-replicate bootstrap over cell subsamples and marker-gene resamples. The two intervals are
**disjoint**, P(deep > tier-1) = **0.000**, difference **−0.1389 [−0.2017, −0.0884]**.

**Consequences, and they are binding on this task:**

1. **`deep_starmap` cannot carry a reconstruction claim against an optimal copier.** Its entire
   headroom is **half the reproducibility envelope**. Its large `expr_mode` margins (0.6614 and
   0.5948 on the two arrangement metrics, both 6.0x+ the within-arm fold spread) are real, and
   they separate the arms on a task where copying already reaches **98 %** of what is achievable.
   Any such number must be reported **with the ceiling beside it**, never alone.
2. **Tier-1 is the informative reconstruction benchmark** — 4.6x the envelope of genuine room —
   **despite being the smaller and sparser dataset.** This is the reverse of the campaign's
   working assumption that tier 2 is the more demanding test.
3. **The "saturated" verdict is against an *oracle* copier.** Against the copy the shipped
   configuration actually performs, `deep_starmap` has **2.6x** the envelope of headroom — see
   **R14**, which is a defect in that copier, not a property of the tissue.
4. **Report every reconstruction number as a fraction of the measured headroom**, not only as a
   raw value or an envelope multiple. §2's ceiling protocol was written for the synthetic fixture
   because it was "the only dataset with a known generative law"; `scripts/t09_depth_ceiling.py`
   supplies a model-free ceiling for any dataset and that limitation no longer holds.

**What this does not say.** It does not say the generative path is untestable — it says
*reconstruction of an interpolated section* is the wrong instrument on a dense panel. The claims
copying cannot address at all — **unmeasured genes** (zero-shot; `forward_zero_shot` and
`gene_pool` exist and have never been run) and **arbitrary planes** (where copying has no output
and the comparison is categorical) — are unaffected by any of this and are where the method's case
now rests.

---

## 0. The additivity contract

**Existing bench3 results must never need re-running.** Every change is additive; nothing in
`benchmark-pbya-v3` is edited in place. If any part of this task cannot be done additively,
**stop and report it** — do not edit and explain afterwards.

### The complete list of bench3 files T10 touches

| File | Change | Why it is additive |
|---|---|---|
| `src/bench3/methods/run_spatialcpav25_gen.py` | **NEW FILE** | Nothing reads it but the `METHODS` entry below. |
| `src/bench3/config.py` | **One appended `METHODS` dict entry.** Nothing else — not `METHOD_ORDER`, not `DATASET_SPECS`, not a constant, not a metric name. | `config.py`'s own comment: a method in `METHODS` but absent from `METHOD_ORDER` "is never run unless it is named explicitly". So a bare `run_all` behaves exactly as before, and `_method_sort_key` sorts v25 last by its index fallback without reordering anything. |

**That is the entire footprint.** In particular:

- `evaluate_paper.py` is **not touched at all.** Its SHA-256 is asserted **before and after** every
  campaign run and recorded in the report:
  `7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992` (764 lines, re-verified
  2026-08-20). A mismatch at either end aborts and names the file.
- **No new `DATASET_SPECS` entry.** Every derived dataset (§9) is passed **by path** —
  `resolve_dataset_arg` accepts a path as readily as a registered name, on every stage that has
  `--dataset`. `run_benchmark.dataset_meta` falls back to `REGISTRATION = "none"` for an
  unregistered name, and our driver passes `registration=` explicitly anyway. Nothing in
  `DATASET_SPECS` changes, so no existing build changes.
- **No new design function.** `run_benchmark.run_single(method, holdout_config, ...)` takes the
  holdout config as a plain dict. The boundary holdout (§4.3) is a dict our driver constructs;
  `design.py` is never called for it and never edited.
- **No path or schema change.** Seeds and arms are separated by `BENCH_V3_RESULTS`, one results
  root per (tier, arm, seed). `build_input` caches under `INPUTS_CACHE/<dataset>/<holdout_id>/`,
  so a new holdout id creates a new cache directory and touches no existing one.

### Two data-contract facts the pilot established about every bench3 dataset

**1. Coincident coordinates are normal here, and permitted narrowly.** bench3 collapses each
multi-plane slab to its centre z, so two cells at the same `(x, y)` in different planes become
exactly coincident. Measured on the tier-1 build: **143 of 28 978 cells (0.49 %)**, in every
section; `flattened_z` is true for **all 18** datasets, so the mechanism is universal.

T01's duplicate-coordinate check is therefore **scoped, not removed**: `Volume.flattened_sections`
permits exact ties, `Volume.n_coincident_coords` records the count for any downstream code, and
`validate_volume` warns once per volume with the count. **An unflattened volume keeps the hard
check** (`test_duplicate_coords_are_permitted_only_for_flattened_sections`), and the flag propagates
through `split_holdout` and `loso_folds` so it cannot lapse on the volume the model actually trains
on. The flag is read from the data, never inferred from the coordinates — an inferred exemption
would silence the check on a dataset with a genuine problem. bench3 records it at
`uns['paper_protocol']['flattened_z']`, which the wrapper passes explicitly.

Wrapper-side de-duplication and coordinate jitter were both rejected: the first would change the
cells v25 trains on relative to every other method, breaking the shared-input guarantee; the second
is silent data modification.

⚠️ **Report it as a metric caveat.** A zero-distance pair is real for any kNN-graph quantity —
Moran's I, Geary's C, the layout's neighbourhood terms — and `evaluate_paper` has the same exposure
without checking for it. Quote `n_coincident_coords` in the methods.

**2. STARmap's panel is narrower and denser than any default assumed.** 28 genes, and a median
per-gene detection rate of **0.9999** in the training sections — a curated panel with almost no
zeros, unlike the 200-gene synthetic fixture every default was tuned on. Two consequences the pilot
hit directly:

* `Config.expr_pca_dim = 32` **exceeds the panel width** and `validate_config_against_volume`
  refuses the fit. This is a hard constraint, not a preference: **`specs/09`'s selector must clamp
  `expr_pca_dim` to the panel width**, or the protocol dataset can never be fitted. Recorded as an
  owed fix; the pilot worked around it with an explicit value.
* `assert_detection_rate`'s band is measured against ~1.0 here, so the guard is unusually tight on
  tier 1. It fired correctly on an under-trained model (0.529 against 0.9999) — treat a detection
  failure on STARmap as a signal about the budget before suspecting the decoder.

### ⚠️ `summary_by_method.csv` averages across holdout ids — never read it

`aggregate_results.summarize` groups by `(dataset, method)` and **means every metric across
holdout ids**. Point it at a results root holding both `paper_2_4_6` and `wide_3_4_5` and it will
average the alternating and consecutive regimes into one number — precisely the thing this spec's
**Do NOT** forbids. Two consequences, both mandatory:

1. **`eval/stats.py` reads `all_metrics.csv` and `per_section_metrics.csv` only.** Those keep one
   row per holdout and per section respectively. `summary_by_method.csv` is never an input to any
   published number.
2. **One results root per tier, arm and seed** — this is what keeps the averaging harmless, and it
   is the same isolation the seed rule needs anyway.

---

## 1. The two tiers — every number is labelled

**STARmap is protocol-faithful and stays untouched.** It follows the SpatialZ paper exactly: the
same trim (`z = 6–13`, `91–94`, always applied, `--no-trim` refused), the same 7 × 11-plane
partition, the same 2/4/6 holdout. That is the headline comparison, and nothing about it may be
varied.

| Tier | What it is | What may appear in it |
|---|---|---|
| **Tier 1 — protocol-faithful** | `starmap_visual_cortex`, holdout `paper_2_4_6`, unmodified build, unmodified evaluator | The headline six-metric table; the direct SpatialZ / FEAST / isoST / v20 comparison; the control-metric table; the statistics (Wilcoxon, BH, bootstrap CI, Cliff's delta) computed on it |
| **Tier 2 — extensions** | Anything needing a modified design: the boundary rows (R3), the `wide` regimes, the re-sectioned datasets (E3/V1/V2/V3), the analogue datasets, E1's wide-panel home, V4's thickness pair | Reported in its own tables and figures, always labelled with the design that produced it |

### ⚠️ Tier 1 carries boundary contamination — a caveat on the protocol itself

**The paper protocol's held-out set includes a section adjacent to the stack boundary.** With
sections 1/3/5/7 as input, `section_2` is bracketed by `section_1` — the **first section of the
volume** — so its evidence is one-sided in a way `section_4` and `section_6` are not.

It is measurable, large, and consistent across methods. In the prior campaign's tier-1 rows,
`section_2` is the **worst-scoring held-out section in 37 of 49 (method, metric) cells**, and in 6
or 7 of 7 metrics for every SpatialCPA method — including SpatialZ, FEAST and isoST. This is open
risk **R3** appearing *inside* the design that was supposed to exclude it: bench3's `split_holdout`
never holds out the first or last section, and that is usually described as making `paper_2_4_6` a
purely interior test. It is not.

**Three consequences.**

1. **It is a caveat on the protocol, not on any method.** Every method pays it equally — it is a
   property of where the sections sit, not of who reconstructed them. So it does not bias the
   tier-1 *comparison*, and the headline table stands. State it once in the methods, and report
   per-section values beside every tier-1 median so a reader can see it.
2. **It is why the estimator is a median** (§4.6). A fixed structural penalty on one of three
   sections is exactly what a mean over n = 3 launders into the headline number, and it already
   inverted one verdict.
3. **It may be part of what C1 measures, and that must be tested rather than assumed.** SpatialZ's
   `celltype_localization` lead is its only tier-1 win, and `section_2` is the section where every
   method is weakest. **A method that is merely more robust at a one-sided boundary would show
   exactly the same pooled lead as one with a better layout model.** So when C1 is measured
   (§11.1), report the localization gap **per section**, not only pooled, and state which of the
   two readings the data supports:

   | if the gap is… | the reading is |
   |---|---|
   | concentrated in `section_2` | **boundary robustness** — SpatialZ degrades less with one-sided evidence, and v25's layout head is not the thing being tested |
   | uniform across `section_2/4/6` | **layout quality** — the intensity-field/Potts/repulsion stack is the right place to attack it |

   The tier-2 boundary rows (§4.3), which hold out `section_1` and `section_7` outright, are the
   clean version of the same question and should be read together with this split.

**Rules, enforced in code (`eval/stats.py::assert_tier_purity`):**

- A tier-1 table may contain **only** rows whose `dataset == "starmap_visual_cortex"` **and**
  `holdout_id == "paper_2_4_6"`. Anything else raises and names the offending row.
- Tier-2 numbers are **never merged into a tier-1 table**, never averaged with tier-1 numbers, and
  never used to compute a tier-1 rank. A rank is a position in a field; changing the field changes
  it.
- Every table, figure caption and report line states its tier and its holdout id. A number without
  a tier is not publishable.

Why this matters beyond bookkeeping: the tier-1 claim is *reproduction of a published protocol*.
The moment an average includes a section that the paper did not hold out, or a volume it did not
use, the sentence "measured under the SpatialZ STARmap protocol" stops being true.

---

## 2. Metrics — `spatialcpav25_gen/eval/metrics.py`

**Do not port, and do not reimplement** (settled, SPEC_QUESTIONS A3). The scoreboard is
`benchmark-pbya-v3/src/bench3/evaluate_paper.py`, and it is the instrument every comparable number
came out of. A reimplementation that "agrees closely" is not comparable: the claim is a *difference*
between methods measured on one instrument, and two instruments agreeing to 1e-3 turn a 0.01 median
gap into an argument.

1. **Vendor or import it verbatim.** Either import `bench3.evaluate_paper` directly, or vendor it
   into `eval/_bench3_evaluate_paper.py` **byte for byte**, with a header saying where it came from
   and that it must not be edited.
2. **Pin it with a content hash.** `BENCH3_EVALUATE_PAPER_SHA256 =
   "7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992"`, checked at import and again
   after every campaign run (§0). A changed hash raises and names the file.
3. **Assert bit-identical output.** `test_metrics_match_bench3_bitwise` runs the adapter and
   `evaluate_paper` on a fixed synthetic pair and asserts every metric is `==`, not `allclose`.

`eval/metrics.py` is a thin adapter: it maps our `AnnData` pairs onto `evaluate_paper`'s signature,
unpacks the result into `METRIC_REGISTRY`, and adds the one control metric bench3 lacks. **It owns
no metric arithmetic.**

*Footnote on v20's two bugs.* `reference/learn_spatialcpav20.py` computes `gene_mean_spearman` /
`gene_var_spearman` with `np.corrcoef` (Pearson under a Spearman name, `:1876`) and rank-normalises
with `argsort`, giving tied zeros distinct ranks (`:1810`). Both are real and both matter for
reading v20's *internal* tuning signal — that is what its own development was steered by. Neither
is present in `bench3/evaluate_paper.py`, which uses `scipy.stats.spearmanr` and
`rankdata(method="average")`. There is nothing to fix on the scoreboard and no "bug fix" to apply to
the baselines. Say this in the methods rather than claiming to have fixed the benchmark.

The six target metrics, as the scoreboard names them:

```
paper_morans_pearson          r between per-gene Moran's I vectors
paper_gearys_pearson          r between per-gene Geary's C vectors
paper_umap_mixing             kNN mixing in a shared embedding
paper_marker_field_r          2-D binned marker field agreement
paper_marker_depth_r          laminar depth-profile agreement
paper_celltype_localization   per-type spatial distribution agreement
```

⚠️ `marker_field_r` and `celltype_localization` do **not** exist in v20 under those names; they come
from `evaluate_paper`'s `marker_metrics` and `celltype_localization`. Record the mapping in the
report so a reader can trace a paper number to the function that produced it.

### Control metrics — build one, not six

Five of the six unoptimised controls the original spec asked for are already columns bench3
computes on every run. Reimplementing them would create a second instrument for no gain:

| Control (original spec) | bench3 column | Action |
|---|---|---|
| `sinkhorn_profile_distance` | `gen_sinkhorn` | read it |
| `coexpression_module_preservation` | `gen_coexpression_agreement` | read it |
| `neighbourhood_enrichment_agreement` | `gen_celltype_nhood_agreement` | read it |
| `gene_variance_rank_corr` | `paper_gene_var_spearman` (+ `gen_gene_var_pearson`) | read it |
| `detection_rate_agreement` | `paper_gene_detection_spearman` | read it |
| `duplicate_profile_rate` | **absent** | **T10 builds this one** |

`duplicate_profile_rate(gen) -> float` is the fraction of emitted profiles that exactly repeat
another emitted profile. It is the metric that catches a method copying real cells, and it is what
§2's "a method above the ceiling is usually a bug" check reads. Fixed seed, documented
normalisation, docstring stating direction and range, registered in `METRIC_REGISTRY` like the rest.

The control table is **not optional**: six of the target metrics are trained against (T08).

### The achievable ceiling

**A metric's stated range is not its achievable range.** Every metric compares a *generated* section
with a *real* one, so a perfect model — one sampling from exactly the right law — still scores below
the top of the scale, because a different **realisation** of the same law is not the same point
cloud. Measured at T05 for `celltype_localization` on the synthetic fixture: the held-out section
scored against itself reaches 0.9221; an independent draw from the fixture's own generative law
reaches **0.7178**. A method scoring 0.71 there is at **99%** of what is achievable, and the raw
number alone says the opposite.

**On the synthetic fixture** (the only dataset with a known generative law) the full protocol
stands, unchanged from the original spec:

1. **Measure a ceiling for all six target metrics and every control.** Generate the *ideal* arm by
   drawing from `tests.fixtures.synthetic`'s `GroundTruthField` directly — positions from the true
   intensity, marks from the true composition, expression from `expression_mu` + `sample_counts` —
   never from the trained model. Same held-out sections, same seeds, same metric code path.
2. **Report every method number twice**: raw, and as a fraction of the ceiling. Headline table,
   ablation table and baseline table alike — an ablation costing 0.02 raw on a metric whose ceiling
   is 0.72 has cost 3% of the achievable range, and that is the number a reader needs.
3. **Report the ceiling's own spread** across held-out sections and seeds: at least
   `Config.ceiling_n_draws` (8) independent draws, mean and standard deviation, so a
   method-vs-ceiling gap can be read against the ceiling's own noise.
4. **A method above the ceiling is a finding, and usually a bug.** Either the ideal arm is not
   drawing from the true law, or the method is copying real cells — check `duplicate_profile_rate`
   and `layout_mode`. T05 measured one legitimate case (`field` mode at 1.110× on one section,
   inside the ceiling's per-section spread), which is why rule 3 exists.
5. **Where a metric averages over parts, report per-part ceilings.** For `celltype_localization`
   that means **per cell type**: T05 measured the most abundant type (34% of cells) at 0.33–0.84
   across sections while localised minority types sat at 0.60–0.91, because the metric normalises by
   the divergence to a within-tissue null and that null collapses (`d_null` ≈ 0.08) for a type
   already spread tissue-wide. The abundant types are where the headroom is smallest and the
   variance largest, and a weighted average hides both. See `specs/05`, "Why the criterion is a LOSO
   mean".

**On real datasets there is no generative law and therefore no ceiling** — and the referent the
original spec names is already implemented. bench3's `selftest.make_probe` writes four
known-quality reconstructions through the *real* prediction contract:

| probe | role here |
|---|---|
| `oracle` | the real held-out cells — the self-score upper bound |
| `flanking_copy` | the nearest **training** slice, copied — this **is** `run_nearest_copy`, the flanking-section referent |
| `spatial_scramble` | perfect marginals, destroyed spatial structure — the discriminating control |
| `random` | the floor |

So on real data the ceiling protocol reduces to **scoring `oracle` and `flanking_copy` as two extra
methods in each results root** and reporting every method number against them. Do not rebuild them.

---

## 3. Baselines — `spatialcpav25_gen/eval/baselines.py`

Most of the baseline set already runs on the instrument. Build only what is missing.

| Baseline | Status | Action |
|---|---|---|
| `run_spatialz` | bench3 `METHODS["spatialz"]`, published defaults, `bench_spatialz` env | **run it, do not wrap it** |
| `run_v20` | bench3 `METHODS["spatialcpav20_gen"]` | **run it** |
| `run_nearest_copy` | `selftest.make_probe("flanking_copy")` | **score the probe as a method** |
| `run_independent_donor` | built at T06 (`eval/baselines.py::IndependentDonorBaseline`) | needs a bench3 wrapper so it is scored on the instrument; training-free, no fit |
| `run_convex_interp` | **absent** | **T10 builds it** — smooth ceiling / realism floor; training-free |

FEAST and isoST come free as additional published comparators.

### Who is a competitor and who is development history

**The competitors are the three published methods: SpatialZ, FEAST and isoST.** They are what the
paper is measured against, and they are what the tier-1 headline table compares.

**Prior SpatialCPA versions (v14, v18, v19, v20, v21, v22, v23, v24) are internal development
history, not comparators.** They belong in a **development table**, reported separately and
labelled as such, and they serve exactly one formal role: **v20 is the no-regression reference**
(`test_selector_can_recover_v20_config`; `layout_mode=resample` + `expr_mode=cross-mix`). v25 has to
beat it, but beating it is not the paper's claim.

Concretely: no headline table, figure, abstract or methods sentence is framed around beating an
internal version. "v25 improves on v21" is a development result; "v25 improves on SpatialZ" is the
claim. `eval/stats.py::assert_tier_purity` also refuses a headline table whose comparator set
contains a `spatialcpav*` method other than v25 itself.

**v14 and v18 are dropped as baselines** (settled; listed in `design/v23_design.md` §7). Both are
superseded by v20 on every metric of the existing bench3 campaign, so they add two columns nobody
would read — v20 is the version the no-regression guarantee is stated against and the one that has
to be beaten. Say so in one line in the methods rather than leaving their absence unexplained.

For the competing method, use its published defaults (`syn_mode='default'`, `k_sam=3`,
`k_neighbors=1`, `nb_iter_max=3000`, `num_projections=80`) and its own MENDER-based niche pipeline.
Do not tune it; do not cripple it. Record the exact settings in the report — reviewers check this.
bench3 already pins the invocation in `METHODS`, so quote that entry rather than restating it.

⚠️ SpatialZ mutates `adata.obs_names` in place (appends slice ids). **Deep-copy inputs before
calling it** or subsequent baselines silently receive corrupted data. bench3 isolates each method in
its own subprocess so this cannot bite there; it bites in `eval/baselines.py`, which does not.

### Where each step can run — two machines, and it is not symmetric

This task spans two environments and the split decides what can be built versus measured. **Where a
step needs the server, build it to run there rather than working around it.**

| | development container | campaign server |
|---|---|---|
| package, tests, synthetic fixture | ✅ | ✅ |
| tier-1 STARmap volume | ✅ | ✅ |
| **`deep_starmap` and the other real volumes** | ❌ | ✅ |
| **conda / the four `bench_*` envs** | ❌ | ✅ |
| **MedCPT encoder** (`huggingface.co`, 403 via proxy) | ❌ | ✅ presumed |
| `mygene` network | ❌ | — no longer needed (§7) |

**Three consequences that are structural, not temporary.**

1. **The comparator re-runs are server-only** — every wrapper is invoked as `conda run -n <env>`.
2. **E1 is server-only** (§7), because `deep_starmap` lives there.
3. ⚠️ **A shipped-config run is server-only too, and this one is easy to miss.** `Config`'s default
   is `text_emb_mode="medcpt"`, which needs the MedCPT encoder. In the container that encoder is
   unreachable, so **any local run is forced to `text_emb_mode="lookup"` — which is ablation A3, not
   the shipped method.** Per-dataset selection is affected the same way: `text_emb_mode` is one of
   its gates, so a local selection cannot score the `medcpt` arm and cannot return the shipped
   configuration. **Selection and any headline fit are campaign-machine work.**

What the container *can* do is everything that is not a measurement: build the code, run the tests,
exercise every path on the synthetic fixture, and run **attribution experiments** on tier-1 STARmap
under an explicitly non-shipped configuration — provided every number from those is labelled with
the configuration that produced it.

### ⚠️ The existing v20/v22 numbers are NOT in this repository

Verified 2026-08-20: `benchmark-pbya-v3/results/`, `benchmark-pbya-v2/results/` and
`benchmark-pbya/results/` **do not exist**; there is no `metrics.json`, no `all_metrics.csv` and no
`summary_by_method.csv` anywhere in the tree, and `progress/numbers.md` carries no bench3 rows. The
results tree is gitignored and was never committed.

**Consequence: comparability requires re-running the comparators.** A v25 row is only comparable to
a v20 row if both came off the same instrument, and the v20 row does not exist here. Budget it
(§12): the tier-1 comparator set is SpatialZ, FEAST, isoST and v20 on STARmap under `paper_2_4_6`,
plus the two probes. These are cheap relative to a v25 fit and they are a **prerequisite of the
pilot**, not a follow-up. If the numbers exist on another machine, confirming that and copying the
tree in is the cheaper path — check before running.

---

## 4. The driver — `spatialcpav25_gen/eval/bench3_driver.py`

bench3 already provides everything §3 of the original spec asked a harness to provide: long-format
per-section rows (`aggregate_results`), per-(dataset, method, holdout) caching (`results/…`),
resumability (`--skip-existing`), a shared training-only input built once per holdout and reused by
every method, and decoupled re-evaluation (`evaluate_all --force`). **Do not rebuild any of it.**

The driver's whole job is the three dimensions bench3's results path does not carry — seed, arm and
tier — plus the boundary holdout, and it does that entirely from outside.

### 4.1 One results root per (tier, arm, seed)

`results/<method>/<dataset>/<holdout_id>/` has no seed and no arm dimension, so a second seed would
overwrite the first. The driver sets `BENCH_V3_RESULTS` per cell:

```
runs/t1/headline/seed_{1,2,3}/          tier 1, shipped config
runs/t1/ablation_a2_2400/seed_{1,2,3}/  tier 1, one arm
runs/t2/wide3/seed_{1,2,3}/             tier 2
runs/t2/boundary/seed_{1,2,3}/
```

Each root is an ordinary bench3 results tree, aggregated by bench3's own `aggregate_results`, then
read across roots by `eval/stats.py`. No path change, no schema change, and a root containing a
single design cannot be mis-averaged by `summarize`.

### 4.2 The repeated-seed rule, and exactly which measurements pay for it

`specs/09` §3's rule: any measurement reaching a paper claim runs at least `Config.claim_min_seeds`
(**3**) seeds and reports the spread, not a point estimate. T09 measured why — refitting one
configuration at the same seed in a different process moved its scores by up to **0.0120**, while
the gap between the two `text_emb_mode` options was **0.0110**, so "wins" and "wins by less than the
run-to-run variation" were indistinguishable. A benchmark whose purpose is claiming wins cannot
leave them that way (open risk **R10**).

**Scoped, because three seeds on everything is not what the rule is for.** The rule attaches to
*claims*, not measurements. `eval/bench3_driver.py` provides **`CLAIM_BEARING`** — the
machine-readable form of the table below — and `_check_claim_coverage`, which refuses to emit a
headline table containing a measurement classified neither way. Same derived enforcement
`TRAINING_FREE_OPTIONS` and `CAPABILITY_CLAIM` carry in `train/select.py`, for the same reason: the
classification is made when a measurement is added, not rediscovered after a reviewer asks.

| measurement | claim-bearing? | seeds |
|---|---|---|
| headline six-metric table, per regime | **yes** — this *is* the claim | **3** |
| ablation arms that carry a claim (A2 on/off, A7 SEFL net contribution, A8 `loss_prog_WRONG`) | **yes** — each is stated as an effect | **3** |
| capability experiments E1–E5 | **yes** — each is a claim of a capability | **3** |
| boundary stratification of the headline metrics | **yes** — reported as a gap | **3** |
| SEFL validations V1–V4 | **yes** | **3** (V4: see §8) |
| achievable-ceiling measurements | no — a bound on interpretation, not a claim of superiority | 1 |
| diagnostics (per-module Moran's agreement, detection MAD, `w(v)`, retrieval-window derivation) | no — they inform, they do not claim | 1 |
| calibration statuses and their achieved-vs-target numbers | no — reported as statuses, not compared against a baseline | 1 |
| config selection itself | no — `specs/09` §3's rules govern it, and its margins are checked against the envelope | 1 |

Report the spread as **min–max across seeds** beside every claim-bearing median. **A claim whose
effect is smaller than its own envelope is not a claim** — report it as a tie, with the numbers.
"Its own" is doing real work in that sentence; see §4.2a.

⚠️ **The 0.0120 figure above is retired.** Re-measured 2026-08-27 on tier-1 after
`data.schema.section_seed` replaced a salted builtin `hash()` in two RNG seeds: refitting one
configuration at the same seed in a separate process now agrees **bitwise — 36 of 36 values,
largest difference exactly 0**. The original 0.0120 was that defect, not run-to-run variation, so
the envelope it helped justify was inflated by a bug. The sentences above are kept because the
*rule* they motivate survives; the number does not.

### 4.2a The envelope is per-metric **and per-arm** — a methods finding, not a caveat

**State this in the paper's methods. It is not reported anywhere in this literature, and it
changes how any repeated-seed benchmark should be read.**

A single pooled reproducibility envelope is wrong in two independent directions at once.
Measured on tier-1 STARmap, `expr_mode` gate, three post-fix seeds:

| metric | `cross-mix` (copying) | `zinb-flow` (generative) | envelope | vs a pooled 0.0335 |
|---|---|---|---|---|
| `morans_pearson` | 0.0054 | **0.0574** | 0.0574 | pooled too **small**, 1.7x |
| `gearys_pearson` | 0.0027 | **0.0595** | 0.0595 | pooled too **small**, 1.8x |
| `umap_mixing` | 0.0068 | 0.0190 | 0.0190 | pooled too large, 1.8x |
| `marker_field_r` | 0.0049 | 0.0148 | 0.0148 | pooled too large, 2.3x |
| `marker_depth_r` | 0.0084 | **0.0472** | 0.0472 | pooled too **small**, 1.4x |
| `celltype_localization` | 0.0000 | 0.0000 | 0.0000 | inert under `resample` |

**1. Across metrics — a 4.0x range**, and a pooled figure errs in *both* directions: too lenient
on three of the six, too strict on two. A pooled envelope does not merely lose resolution, it
gives the wrong answer in a direction that depends on which metric is being read.

**2. Across arms — up to 22x, and this is the part nobody reports.** On the tier-1 `expr_mode`
gate `cross-mix` moves 0.0027–0.0084 across seeds where `zinb-flow` moves 0.0148–0.0595, so a
margin between them inherits **nearly all** of its run-to-run variance from one side.

⚠️ **State this as an observation. The mechanism is unexplained.** The obvious reading — that a
copying baseline barely uses the fitted weights and is therefore seed-invariant — was written
into an earlier draft of this section and **does not survive the second measurement.** On the
`deep_starmap` `text_emb_mode` gate *both* arms are `zinb-flow` and only the gene embedding
differs, yet the arms still move unequally **and the worse arm alternates by metric**:

| gate | `morans` | `gearys` | `umap_mixing` | `marker_field_r` | `marker_depth_r` |
|---|---|---|---|---|---|
| tier-1 `expr_mode` — worse arm | `zinb-flow` | `zinb-flow` | `zinb-flow` | `zinb-flow` | `zinb-flow` |
| `deep_starmap` `text_emb_mode` — worse arm | `medcpt` | `lookup` | `lookup` | `lookup` | `lookup` |

So the asymmetry is **real, per-metric, and not a fixed property of "the copying arm"**. What can
be claimed is the observation and its consequence; what cannot is a mechanism.

The consequence for any benchmark comparing two arms — which is all of them — is that **the
envelope must be measured on the arm that carries the variance, per metric and per gate**, and
quoting a margin against a pooled figure systematically flatters whichever comparison happens to
involve the steadier arm. Because the worse arm is not predictable, it cannot be reasoned about
in advance: it has to be measured, which means **seeding both arms and reporting them
separately**. That is the methods claim.

**The envelope is also dataset- and gate-specific.** The table above is tier-1's, for the
`expr_mode` arms. It may not be applied to `deep_starmap`, nor to the `text_emb_mode` gate:
reusing a figure measured in one setting because it is the only one available is exactly the
error this section exists to retire. Each claim-bearing comparison pays for its own envelope, and
`scripts/t09_seed_claim.py` computes it from the runs.

### 4.2b A clearance against a *referent* takes the worst envelope in the comparison, not the arm's own

§4.2a settles the envelope for a **contrast** between two arms: the arm that carries the variance.
It says nothing about the other shape of criterion, which the zero-shot experiment needed and
which had to be settled separately — **"does arm X sit above a referent by more than noise?"**
There is only one arm in that sentence, so "the worse arm" has no referent, and the obvious
reading is to use arm X's own across-seed spread.

**That reading is wrong, and the reason is comparability.** A criterion of this shape is never
applied to one arm in isolation; it is applied to every arm of an experiment against a shared
band, and the verdicts are then read against each other. Give each arm its own threshold and the
verdicts stop being comparable: the arm that clears is the one that varied *least* across seeds,
not the one that scored highest. Two arms 0.004 apart can land on opposite sides of the line
because one has a spread of 0.032 and the other 0.127 — and the experiment then reports that the
steadier arm has a capability the higher-scoring one lacks, which is not a statement about
capability at all. A rule that rewards steadiness over score is a rule that can be satisfied by
being boring.

**The rule.** A clearance is read against the **largest across-seed envelope among everything the
comparison contains** on that metric and that dataset: every arm being compared, and the referent
itself. One threshold per metric per experiment, so every arm's verdict is on the same footing
and none can be bought with a low variance. On a fixed layout the referent's own envelope is
exactly zero and the arms decide it; on a design where the referent moves between seeds, it does
not get to be free.

**Stated before the number it changes.** The zero-shot run's primary comparison had already been
scored when this ambiguity surfaced, and the two readings give different verdicts, so the rule is
argued above from comparability alone. Its consequence is then declared rather than discovered:
under the arm's-own reading A3 clears at 1.09x and A1 at 0.24x, giving a case the pre-registration
never named; under the rule above the shared threshold is 0.1273 and **neither** arm clears
(A1 0.24x, A3 0.27x), which is the pre-registered **REFUTATION OF THE IDEA**. Anyone who thinks
the rule was chosen for that outcome should note that it also makes the criterion *harder* for
every arm in every future experiment, including the ones this project would rather pass.


### 4.2c A referent is not a floor until its input is shown to carry nothing

A no-information referent answers "what does this metric return when there is nothing to find?"
It answers that only if its input contains nothing to find — **a property of the input, decidable
without reference to the metric, and for the constant field decidable exactly.**

**The test is a boolean.** The constant field broadcasts one row over every cell, so the
pool-restricted size factor is the same number for every row and the normalised input is
**bitwise identical down every column**. Measured on the same rows, same run:

| referent | rows bitwise identical | per-gene std (float64) | verdict |
|---|---|---|---|
| constant field | **yes** | **exactly 0.0** | no variation; not a floor |
| shuffled positions | no | 0.155 | the panel's variation; a floor |

No threshold, no tolerance, and no dataset on which it could come out differently — it follows
from how the referent is constructed. `tests/test_select.py::
test_a_constant_field_normalises_to_bitwise_identical_rows` pins it.

**Three instruments were tried; the first two were thresholds and both failed.** Recorded because
the failures generalise.

1. **An argument, not a measurement.** The constant field was kept as a floor for the profile
   metrics because `soft_depth_profile`'s bin normalisation makes it track cell density — reasoning
   where a measurement was available.
2. **A precision-drift cut** (recompute at float64; degenerate above 0.01 drift). It separated by
   six orders of magnitude on the synthetic fixture and **failed on `deep_starmap`**, whose eight
   constant-field rows drift 0.0042, 0.0092, 0.0092, 0.0111, 0.0438, 0.0545, 0.0738, 0.1905 — a
   continuum with no gap, so the cut fell between two rows of identical construction and called
   one stable. **A fixture that separates cleanly is not evidence that a threshold transfers.**
3. **A coefficient-of-variation cut** at 1e-6. It gave the right answer on both datasets — and the
   quantity it thresholded was **an artifact of the measurement**. A float32 standard deviation
   over bitwise-identical values returns ~2.4e-07 rather than zero; the same data in float64 gives
   exactly 0.0. The instrument was reporting its own rounding and a cut was being placed around
   it. **Before thresholding a small number, check it is not your own arithmetic.**

Drift and the float64 spread are still reported as corroboration and decide nothing. Drift also
answers the obvious objection: `section_5`'s held-out constant field reads +0.5780 at single
precision and **+0.3875 at double** — smaller but nowhere near zero, so "it is round-off" needs
the mechanism. Round-off in the centring step is one ulp of each value, so its *pattern across
genes* tracks expression magnitude at every precision, and expression magnitude is what real
Moran's I correlates with. Doubling the mantissa changes the number without changing what it is.

**Why this is a spec rule.** The zero-shot pre-registration wrote "clear the constant-field band"
into all three of its outcome conditions, and the referent it named carries no variation at all.
The verdict survived — both readings give REFUTATION OF THE IDEA, and the aggregator prints it
under both — but it survived by margin, not by design. Any referent entering a claim pays for this
test first, and the run reports the usable referent's figures beside it so the comparison is on
its own rows rather than another dataset's.

### 4.3 The boundary holdout (R3) — additive, no new dataset, no new design function

**Open risk R3, raised at T04, re-surfaced at T09.** The T04 probe reconstructed the two **edge**
sections at R² **0.2912** and **0.3642** against an interior mean of **0.4474**; at T09 the
uncertainty gate elevated at the ends (+13.2% / +8.8% latent variance). It appeared in two
independent measurements and generation near stack ends is routine, so it is measured, not assumed.

**bench3 cannot currently measure it.** `paper_design` holds out 2/4/6, and
`loo_design(exclude_boundary=True)` is hardwired in `design.py`'s `main()` — no exposed design ever
holds out section 1 or section 7. But `run_benchmark.run_single` takes the holdout config as a
**plain dict**, so the driver constructs it directly:

```python
{"holdout_id": "boundary_1", "design": "boundary",
 "holdout_sections": ["section_1"],
 "remaining_sections": ["section_2", ..., "section_7"],
 "holdout_z": {"section_1": <median z>}}
```

`design.py` is never called and never edited; `build_input` caches under a fresh
`_inputs/<dataset>/boundary_1/` directory; results land in `<root>/<method>/<dataset>/boundary_1/`.
Additive in every direction.

**A boundary section is extrapolation, not interpolation.** It has evidence on one side only, which
is a different task from every other row in this spec. It is **Tier 2**, it gets its own table, and
it is never pooled with `paper_2_4_6`. Report `boundary_1` and `boundary_7` separately from each
other too — they are the two ends of a stack with different neighbours. If the baselines degrade
there as well (SpatialZ interpolates between flanking slices and has no flank at an end either),
that comparison is itself a result worth a row.

Also state, for every headline number, **which regime it came from and how much boundary tissue it
contained.** `paper_2_4_6` holds out interior sections by construction and so under-samples the
regime where the model is weakest; `consecutive-5` on a 7-section stack pushes the held-out run
against both ends. A regime change silently moves the metric otherwise.

### 4.4 Regimes — all three already exist

Corrected: `paper_2_4_6` **is** the alternating design. bench3's README states the generalisation
outright — "hold out every even section, keep the first and last as input"; at n = 7 that is exactly
2/4/6. And `design.py::consecutive_design` provides the wide-gap regimes.

| `specs/10` regime | bench3 invocation | holdout id | tier |
|---|---|---|---|
| `alternating` | `--design paper` | `paper_2_4_6` | **1** |
| `consecutive-3` | `--design wide` (`DEFAULT_WIDE_BLOCK = 3`) | `wide_3_4_5` | 2 |
| `consecutive-5` | `--design wide --holdout-block 5` | `wide_2_3_4_5_6` | 2 |

`DEFAULT_WIDE_BLOCK = 3` is chosen so `paper` and `wide` remove the *same number* of slices and
differ only in adjacency, which isolates gap width itself. Report the three **separately** — a
regime average would destroy the only thing the comparison is for.

⚠️ **The design docs' expectation — "ties or wins at narrow gaps, wins decisively at wide gaps" — is
not supported by any evidence this project holds.** §13.3 audits what the prior campaign actually
measured: there is **no head-to-head wide comparison on STARmap at all**, the per-dataset wide
results are mixed rather than favourable, and the pooled number that looks decisive is an
illegitimate cross-dataset average dominated by one volume. **Rewrite the claim as an open question
the pilot answers**, and do not let any figure caption, abstract or methods sentence assert the
wide-gap advantage until a tier-1 wide comparison exists. This is **pilot criterion C2** (§11.1).

⚠️ **`consecutive-5` on STARmap clamps to n − 2 = 5**, leaving only sections 1 and 7 as input —
about **8.2 k training cells**, half the tier-1 input. Report it as the thin-evidence extreme, not
as a peer of the other two regimes, and state the training cell count beside every number from it.

⚠️ The number of held-out sections **varies by dataset** (`paper_2_4_6` on STARmap,
`paper_alt7of15` on the Allen atlases, `paper_2_4` on CosMx). That changes the *n* of the
per-section Wilcoxon, so `eval/stats.py` reports n per test and never compares a statistic computed
at different n as if it were the same test.

### 4.5 Statistics — `spatialcpav25_gen/eval/stats.py`

bench3 has **none** of this: `rank_methods` averages group ranks, which is not significance. T10
owns it entirely, reading `all_metrics.csv` and `per_section_metrics.csv` from outside.

- Paired **Wilcoxon signed-rank** vs. the competing method, per metric, **paired by section**.
- **Benjamini–Hochberg** across the six metrics.
- **Median difference with 95% bootstrap CI** (10 000 resamples, stratified by dataset) — this is
  the "clear gap in medians" claim, stated defensibly.
- **Cliff's delta** as a nonparametric effect size.
- **Forest plot** per metric: median difference ± CI, one row per dataset. This is paper Figure 2.
- `assert_tier_purity` (§1) runs before any table is emitted.

### 4.5b ⚠️ RESULT — the intensity-field layout is worse than copying. **Resolved: `resample` ships**

**This is a first-class finding, not a diagnostic aside, and it bears directly on the layout head's
headline claim.** Two independent measurements, on different data, point the same way.

**RESOLVED 2026-08-25.** Re-measured on the corrected grid sampler with five arms off one refit
(`reports/r11_starmap_layout_modes.md`), the ordering held and `Config.layout_mode` now defaults to
**`resample`**. The current numbers — `celltype_localization` at ground-truth-matched density,
median over the three held-out sections — are:

| arm | median | vs the 0.7765 copy floor |
|---|---|---|
| `resample` (shipped) | **0.7546** | −0.0219, *inside* R10's 0.0335 envelope |
| `hybrid` | **0.6692** | −0.1073, **3.2x** the envelope |
| `field` | **0.6607** | −0.1158, **3.5x** the envelope |

and the deciding measurement is the **count**, not localization: `field`/`hybrid` emitted 267 567,
21 993 and 3 727 cells against ground truths of 4 187, 4 102 and 4 162, and the same configuration
refitted swung one section 48 343 → 179 495 (3.7x) while every density-matched score moved less
than 0.013. `specs/05` §4a carries the full statement; A4 (§6) reports `field` and `hybrid` as an
addition experiment. **The two tables below are the pilot-era measurements that led here and are
superseded by that report** — they are kept because the reasoning under them is unchanged and
because the fixture half has its own re-run (`reports/t09_layout_mode_gate_grid.md`).

**Signal 1 — real data, tier 1 (T10 pilot, `reports/pilot.md` §6.3), superseded.** Holding
everything else fixed and swapping only `layout_mode`:

| `celltype_localization` | section_2 | section_4 | section_6 | median |
|---|---|---|---|---|
| `layout_mode="field"` (shipped) | 0.5068 | 0.1490 | 0.4252 | **0.4252** |
| `layout_mode="resample"` (v20 fallback: real flanking positions) | 0.7008 | 0.7760 | 0.6808 | **0.7008** |
| `flanking_copy` — the model-free floor | 0.7008 | 0.7765 | 0.7868 | 0.7765 |

`cell_count_ratio` moves 0.8283 → 0.9875 in the same swap. **The learned continuous layout scores
0.4252 where copying a neighbouring slice scores 0.7765** — it is below the floor on the metric it
exists to win, by 0.35, which is ~29x the 0.0120 across-seed envelope.

**Signal 2 — the synthetic fixture (T09's merged 18-cell gate).** The rank winner was
**`resample`** (median rank 3.0); `hybrid` followed at 4.2; **`field`'s best cell ranked 7.0 and it
won nothing.** `hybrid` ships only because it *won a tie-break* — `resample` reuses real positions
and is the v20 fallback, so shipping it switches the learned layout off. And the margin, 0.0344
against a 0.0335 envelope, was **decided inside the noise** (R10).

**Why the pair matters.** On the fixture the preference against `field` was real but undecidable —
inside the reproducibility envelope, which is why `hybrid` shipped at all. On real data the same
preference is **far outside** it. The fixture result was not wrong; it was underpowered, and the
real-data measurement resolves it in the same direction.

**Consequences for T10.**

* **A4** is restated (§6): repulsion-off alone became a no-op once `resample` shipped, so A4 is now
  the ablation of the generative layout as a whole — `field`, `hybrid`, and repulsion-off *within*
  `field` — and it answers "does the intensity-field layout beat copying at all" directly rather
  than only "does repulsion buy realism".
* **C1 cannot be attributed to the expression path** until the layout is fixed or `resample` is
  used as the layout for the C1 measurement — most of the localization gap is positional.
* **`specs/05`'s headline claim did not hold** and is amended there (§4a) with the numbers, as a
  negative result rather than a defect: the generative marked point process is built, measured, and
  loses to copying real coordinates on real tissue. `specs/09`'s `layout_mode` gate was re-run on
  the fixture with the corrected sampler (`reports/t09_layout_mode_gate_grid.md`) so the fixture's
  verdict sits on the record beside the real-data one.

⚠️ **This does not explain the autocorrelation collapse.** With real positions the emitted
expression still carries only 22 % of the tissue's Moran's I (`reports/pilot.md` §6.3). The two are
separate failures and must not be conflated.

### 4.6 The estimator is the MEDIAN. Stated once, used everywhere.

**Every statistic over sections is a median with a bootstrap CI. Never a mean.** This is the
original requirement and it is restated here as a single rule because it was violated once already
and the violation changed a conclusion.

| quantity | estimator |
|---|---|
| a method's score on a (dataset, holdout) | **median over held-out sections**, with the 95% bootstrap CI |
| a method's score across seeds | **median across seeds**, with min–max spread (§4.2) |
| method-vs-competitor gap | **median difference**, 95% bootstrap CI, 10 000 resamples |
| the achievable ceiling and its spread (§2) | median over draws; the spread stays mean ± sd, because it is a Monte-Carlo *noise* estimate, not a score |

**Why this is not a style preference.** The paper design holds out **three** sections, so every
tier-1 number is a statistic on n = 3. At that n a single bad section moves a mean by more than the
gaps being claimed. Measured on the prior campaign's tier-1 rows (§13):

| `paper_marker_depth_r` | section_2 | section_4 | section_6 | median | mean |
|---|---|---|---|---|---|
| spatialcpav20_gen | **0.7422** | 0.9704 | 0.9783 | **0.9704** | 0.8970 |
| spatialz | 0.8966 | 0.9267 | 0.9367 | **0.9267** | 0.9200 |

Median says v20 wins by 0.044; mean says SpatialZ wins by 0.023. **One outlier section inverts the
verdict.** Exactly one of the seven tier-1 metrics flips this way for SpatialZ vs v20 — which is the
point: the flips are rare enough to be invisible and frequent enough to be wrong.

**And the outlier is not random.** Across the prior campaign's tier-1 rows, `section_2` is the
worst-scoring held-out section in **37 of 49** (method, metric) cells, and in 6 or 7 of 7 metrics for
every SpatialCPA method. `section_2` is bracketed by `section_1` — the **first section of the
stack** — so its evidence is boundary-adjacent. **Even `paper_2_4_6`, which holds out interior
sections by construction, contains one boundary-adjacent section, and it is systematically the
hardest.** That is open risk **R3** appearing *inside* the tier-1 design, it is a fixed structural
effect rather than noise, and a mean over three sections silently lets it dominate. Report the
per-section values beside every tier-1 median so the effect stays visible.

`eval/stats.py::assert_no_mean_over_sections` enforces this: any headline or claim-bearing statistic
computed by averaging over the `section` axis raises, naming the metric. The corresponding test is
`test_no_headline_statistic_is_a_mean_over_sections`.

---

## 5. Datasets — which may carry a claim-bearing number, and why

bench3 registers eighteen datasets. They are **not interchangeable**, and two properties decide
what a row from each may be used for.

### 5.1 `expression_type` decides which decoder runs — and only one decoder has been run

`Config.decoder` has three options (`zinb`, `zigamma`, `gaussian`). Every measurement this project
has ever made ran **`zinb`**. `ZIGammaDecoder` is implemented (`model/expression.py:536`,
`sample_zigamma`, `zigamma_log_prob`) but has **never been trained, generated from, calibrated or
benchmarked** — its only test coverage is one parametrised `log_prob` shape check
(`tests/test_expression.py:405`).

⚠️ **And it is not merely untested — T09's calibration is ZINB-only.** `calibrate_detection`'s
core computes the zero-probability through `_zinb_detection` (`infer/calibrate.py:1086`) and solves
the mean–variance intercept on the **negative binomial's** variance. There is no ZIGamma branch.
Under `decoder="zigamma"` that path would compute an NB zero-probability for a Gamma model and
report it as a calibration. Only `_draw_counts` (`calibrate.py:1381`) dispatches on the decoder.

**Making ZIGamma claim-bearing is T06/T09 work, not T10 work.** T10 does not silently do it. Until
it is done:

| `expression_type` | decoder | may carry a claim-bearing number? |
|---|---|---|
| `raw_counts` | `zinb` | **yes** |
| `normalized`, `log2_normalized`, `fluorescence_intensity` | `zigamma` | **no** — diagnostic / qualitative rows only, labelled `decoder=zigamma (unvalidated)` |

Six datasets fall on the wrong side: `imc_breast_cancer`, `merfish_hypothalamus`,
`easi_fish_lha1/2/3`, `allen_zhuang_abca1/2`.

### 5.2 The non-transcriptomic requirement collides with 5.1 — state it, do not paper over it

`design/v23_design.md` §7 requires the campaign to include **at least one non-brain** dataset and
**at least one non-transcriptomic panel**. It is a reviewer defence: every version of this project
has been tuned on brain sections with a transcriptomic panel, and a method whose oblique-sectioning
claim rests on laminar structure would look excellent on brain and fail silently elsewhere.

`imc_breast_cancer` satisfies **both** halves alone — human HER2+ breast cancer, 25-channel protein
panel, 15 real serial sections. It is also `fluorescence_intensity`, i.e. exactly the dataset whose
decoder path has never been run.

**Resolution, stated in the paper rather than hidden:**

- The **non-brain** half is discharged claim-bearingly by `cosmx_nsclc_3d` (human NSCLC tumour,
  960 genes, `raw_counts`, 18 cell types) or `exseq_breast_cancer` (human breast cancer, 297 genes,
  `raw_counts`, 13 types, ~2 k cells).
- The **non-transcriptomic** half is `imc_breast_cancer`, reported as a **Tier-2 diagnostic row,
  explicitly not claim-bearing**, with the reason (`zigamma` unvalidated, calibration ZINB-only)
  stated in the caption.
- `run_benchmark` refuses to emit a headline table unless both halves are present and names what is
  missing (Convention 6). A campaign run without them is a development run and the report says so on
  its first line.

If ZIGamma is validated later, the IMC row is promoted and this section is re-opened. Nothing else
changes.

### 5.3 Cell-type annotations — two datasets have none

`openst_lymph_node` and `visium_mouse_brain_c2l` are **100% unannotated** (`n_cell_types = 0`).

What actually happens: bench3 writes a single `"unknown"` category rather than omitting the column,
so `data/schema.py` — which **requires** `Section.cell_type` as an `int32` code array and shape-checks
it (`schema.py:495`) — loads them without raising, with one type. `IntensityHead` permits
`n_types >= 1` (`layout.py:1213`). So the pipeline runs. But:

- **Potts mark smoothing is degenerate.** With one type, neighbourhood purity is identically 1.0,
  `fit_potts_beta`'s purity constraint is vacuous and the rare-type floor has no rare type to
  protect. The fitted `beta` is meaningless, not merely small.
- **A whole ranked metric group is lost.** `paper_celltype_localization`,
  `paper_rare_celltype_localization` and `paper_rare_celltype_recall` are unavailable or degenerate,
  so `rank_methods`'s `localization` group is empty and the composite is computed over four groups
  instead of five — **not comparable** with any other dataset's composite.

**Decision: both are out of scope for T10.** They are not claim-bearing, not diagnostic, and not
run. `openst_lymph_node` additionally carries bench3's own documented OOM gap (uncapped
whole-transcriptome, ~1.55 M cells; a v14 run was killed by the OOM killer). `visium_mouse_brain_c2l`
additionally has 3 sections with 1 held out and **spot** resolution.

**What would have to happen first**, if they are ever wanted: a type-free path in the layout head
(intensity without marks, `fit_potts_beta` skipped rather than fitted degenerately), a test that
exercises it, and a stated rule for how a four-group composite may be compared with a five-group
one. That is T05 work.

### 5.4 The dataset picks, revised against the table

| dataset | role | tier | claim-bearing | why |
|---|---|---|---|---|
| `starmap_visual_cortex` | **headline** | **1** | ✅ | the protocol; 28 978 cells, 28 genes, 19 types, `raw_counts`, `paper_2_4_6` |
| `deep_starmap` | **E1 home** (§7) | 2 | ✅ | 1 017 genes, mouse brain, `raw_counts`, 137 types, **`paper_2_4_6` — the same design as the headline** |
| `cosmx_nsclc_3d` | non-brain | 2 | ✅ | human NSCLC, 960 genes, `raw_counts`; 340 k cells / 57 k per section — expensive |
| `merfish_thick_cortex` | second brain volume | 2 | ✅ | 28.8 k cells, 254 genes, `raw_counts`, `paper_2_4_6`, 13 µm slabs — **cheapest claim-bearing analogue** |
| `merfish_thick_hypothalamus` | **V4 home** (§8) | 2 | ✅ | 79 k cells, 156 genes, `raw_counts`, 27 µm slabs from a 200 µm block |
| `exseq_breast_cancer` | non-brain, cheap | 2 | ✅ | 1 979 cells, 297 genes, `raw_counts` — but min 57 cells/section, just over bench3's 50-cell floor; treat as a small-n row |
| `imc_breast_cancer` | non-transcriptomic | 2 | ❌ diagnostic | `fluorescence_intensity` → `zigamma` (§5.1) |
| `exseq_visual_cortex` | E2 partner (§7) | 2 | ⚠️ | same tissue as STARmap, `raw_counts` — but 1 130 cells, 5 sections, 28% unannotated, only 1 of 3 markers resolved |
| `allen_zhuang_abca1/2`, `merfish_hypothalamus`, `easi_fish_lha*` | — | 2 | ❌ | `zigamma` (§5.1) |
| `allen_merfish_brain` | — | — | ✅ in principle | 1.17 M cells, 59 sections — cost-prohibitive; excluded unless the pilot says otherwise |
| `openst_lymph_node`, `visium_mouse_brain_c2l`, `st_mouse_brain_ortiz` | — | — | ❌ | no cell types (§5.3); `st_mouse_brain_ortiz` is **spot** resolution |

**Changed from the earlier proposal, because of the table:**

- **E1 moves from `allen_zhuang_abca2` to `deep_starmap`.** ABCA-2 is `log2_normalized` (→ ZIGamma,
  §5.1) and its design is `paper_alt7of15`, so its numbers could not sit beside the headline.
  `deep_starmap` is `raw_counts` and runs `paper_2_4_6` — **the same design as tier 1** — so the
  zero-shot result is read against a headline-shaped row. See §7 for its gene-table blocker.
- **`allen_zhuang_abca2` is dropped from the campaign entirely.**
- **`merfish_thick_cortex` is added** as the cheap claim-bearing analogue (28.8 k cells, the same
  size as STARmap) in place of a second expensive volume.
- **`imc_breast_cancer` is retained but demoted** to a labelled non-claim-bearing diagnostic.
- **`openst_lymph_node`, `visium_mouse_brain_c2l`, `st_mouse_brain_ortiz`, `allen_merfish_brain`
  are out.**

---

## 6. Ablations A1–A8

Each is a config override, wired as a wrapper flag so an arm is one line. Each arm gets its own
results root (§4.1); nothing about the arm mechanism touches bench3.

| ID | Override | Claim tested |
|---|---|---|
| A1 | `prior_mode=iid` | correlated prior preserves autocorrelation |
| A2 | `w_autocorr=w_profile=w_distribution=0.5` (an **addition** — the terms ship off) | contribution of metric-aware training — **at two step budgets**, below |
| A3 | `text_emb=lookup-only` | text channel's value on seen genes |
| A4 | **the generative layout, which ships off** — `layout_mode=field`, `layout_mode=hybrid`, and repulsion off within `field` (an **addition**, restated 2026-08-25, below) | does a learned marked point process beat copying real coordinates — and, inside it, point-process realism, **`g(r)` over `[0, 3R]`** |
| A5 | `w_z=0` in retrieval | the competing method's specific flaw — **wide-gap regime only**, below |
| A6 | Gaussian mean decoder | sparsity/dispersion preservation |
| A7 | `w_thick=w_prog=0.2` (an **addition** — SEFL ships off) | SEFL's contribution — **two losses, not three**; decides whether SEFL is used at all |
| A8 | `loss_prog_WRONG` enabled | **negative control** — wrongly constraining an equivariant quantity should be *worse* |

Ablations run on **STARmap (tier 1)** and **one non-brain claim-bearing dataset**, at `alternating`,
**except A5** (below). They **inherit the headline dataset's selected config** and override one
gate — an ablation is by definition an override of the shipped configuration, so an arm never
re-runs config selection.

### ⛔ Before the ablation table: the gene–gene covariance comparison is a LOSS

`run_independent_donor` exists to isolate chimerism, and T06 measured both halves. They point
opposite ways and the write-up must say so:

| what | status |
|---|---|
| **the mechanism** — per-gene independent draws destroy covariance, a shared latent cannot | **established.** Donors held fixed, draw varied: retained \|off-diag\| 0.978 / 0.920 / **0.897** / 0.884 / 0.844 at D = 1/2/3/5/10 on `alternating`, and 0.955 / 0.818 / **0.783** / 0.714 at D = 1/2/3/10 on `consecutive-3`. Monotone on both. The quantity the claim rests on is the step from a verbatim per-cell copy (D = 1) to the competing method's own **D = 3**, donors held fixed so only the draw varies: **−8 pp** of the real covariance magnitude at a 50 µm gap, **−17 pp** at 100 µm |
| **the model beating the baseline on the correlation matrix** | **NOT established — it loses.** Frobenius error: model **9.316**, independent-donor **7.783**, nearest-copy 6.743, achievable ceiling **5.601**. Worse at `consecutive-3` (17.7 vs 11.3) |

**T08 ran, and it did not close it. The claim is a mechanism claim.** (Decided 2026-08-18; numbers
in `progress/t08_metric_aware.md`, criterion held as a strict xfail carrying them.)

| regime | model, terms on | independent-donor baseline | achievable ceiling | T06, terms off |
|---|---|---|---|---|
| `alternating` | **11.022** | 7.732 | 5.601 | 9.316 |
| `consecutive-3` | **13.391** | 11.383 | 5.513 | 17.7 |

The terms help at the wide gap (17.7 → 13.4) and cost at the narrow one (9.3 → 11.0), and neither
arm reaches its baseline. **Both rows are at 1200 steps, and that qualifier is load-bearing.** At
2400 steps the same terms reach **8.489** against a 7.948 baseline on `alternating` — still a loss,
and the closest this project has come. `specs/09` §3 selects the budget jointly with the metric
weights per dataset, so **A2 reports this table at whichever budget the selector chose, alongside
the 1200-step numbers**, and a run whose selected budget differs from 1200 may not quote these rows
as its own.

**Until T08's `test_metric_losses_close_the_covariance_loss` passes on both regimes, no headline
table, figure, abstract or methods sentence may claim that this method preserves gene–gene
covariance *better than* the competing method.** The claim it may make is the mechanism claim, and
the chimerism table is the evidence. The headline table reports the model-versus-baseline Frobenius
numbers **as a loss**, in both regimes, beside the ceiling. A later task that closes it re-opens
this section; nothing else does.

Why this needs writing down rather than trusting: T06's *first* reading of its own measurement found
a decomposition on which the model won by 2.2×, and it took an out-of-sample check
(`consecutive-3`, ratio 0.995) to establish that the decomposition had been chosen after seeing
which component passed. The same temptation will exist when this table is assembled.

### A2 is an addition experiment, and one budget cannot answer it

**A2 runs the metric-aware terms *on*, because all three ship at 0** — the same inversion T07 made
for SEFL. T08 measured them at `specs/08`'s weights and they cost at T06's 1200-step budget:
Moran's MAE 0.0287 → 0.0408, marker-depth r 0.978 → 0.967, localization 0.967 → 0.962, gene–gene
Frobenius 9.00 → 11.15. A schedule-only control — internal LOSO hiding a section, no terms charged
— sits between the two arms, so the cost is the terms', not the hidden section's.

**But the effect is a budget effect.** The terms add a constraint and converge more slowly, and 1200
steps is where T06 stopped *because the arm without them starts degrading there* — the early stop is
R4's symptom, not a neutral reference. At 2400 steps the ordering reverses on four of six:

| statistic | off@1200 | on@1200 | off@2400 | on@2400 |
|---|---|---|---|---|
| reconstruction (nats/pair) | 1.5901 | 1.6843 | **1.5703** | 1.5885 |
| gene–gene Frobenius (baseline ≈ 7.9) | 9.000 | 11.154 | 9.049 | **8.489** |
| Moran's MAE | 0.0287 | 0.0408 | 0.0339 | **0.0279** |
| marker-depth r | 0.978 | 0.967 | 0.983 | **0.990** |
| mean–variance slope (real 1.741) | 1.762 | 1.734 | 1.773 | **1.722** |
| cell-type localization | **0.967** | 0.962 | 0.958 | 0.957 |

The arm without the terms buys likelihood with the longer budget and pays covariance for it
(1.5901 → 1.5703 while Frobenius goes 9.000 → 9.049); the arm with them improves on both. **A2
reports all six target metrics at both budgets**, and the decision it makes is whether the terms are
on for the headline table. Reproduce with `python scripts/t08_metric_report.py`.

### A7 tests two losses, and until it runs SEFL's net contribution is unverified

**A7 runs SEFL *on*, because all three SEFL weights ship at 0.** Two separate T07 measurements put
them there, and they are different findings:

* `w_cross = 0` — intersection consistency is exact **by construction** in v25 (asserted bitwise on
  an untrained model), so the loss is redundant; and training it flattens the anatomical field,
  because the only plane-dependent channel it can compare is T04's deliberately pose-dependent
  triplane (generated per-gene variance **0.067** against **0.711** with SEFL off). `specs/07` §2's
  amendment, SPEC_QUESTIONS C19, open risk R6.
* `w_thick = w_prog = 0` — these two are *not* broken, but at 0.2 a model trained at **T06's own
  budget and configuration fails three of T06's acceptance tests**: detection-rate MAD 0.0551
  against < 0.05 (T06 recorded 0.0191), mean–variance slope relative error **0.2838** against < 0.15
  (T06 recorded 0.0084), and gene–gene Frobenius **20.301** against T06's 9.316 and the
  independent-donor baseline's 7.563. A default that breaks the previous task's acceptance criteria
  is not a default.

So A7 is an **addition** experiment: the shipped model (SEFL off) against the same model with
`w_thick = w_prog = 0.2`. The question is not "how much does SEFL buy" but "is SEFL used at all".
Describe it that way — as testing **`L_thick` and `L_prog`**, with the methods stating that the
third SEFL loss is unnecessary in v25 and why, because the by-construction result is a *stronger*
claim than the loss it replaces and belongs beside E5.

**Until A7 has run on the six target metrics, SEFL's net contribution is unverified and the paper's
SEFL section cannot be written.** T07 established the losses are correct, bounded in cost and
non-collapsing at their own weights; it did **not** establish that they help, and every
distributional statistic it could measure moved the wrong way:

| term | status after T07 |
|---|---|
| `L_cross` | **not used** — redundant by construction, harmful when trained |
| `L_thick` | verified against its own criterion (counts add at 3.000×, loss charging 0.00); effect on the target metrics **unmeasured** |
| `L_prog` | implemented; its conditioning claim **did not reproduce** on the synthetic fixture (SPEC_QUESTIONS B22); effect on the target metrics **unmeasured** |

A7 reports all six metrics, and beside them the reconstruction NLL (**1.738** off against **2.082**
on), the generated per-gene variance ratio (**0.711** off against **1.04–1.33** on, i.e. overshooting
the real tissue rather than undershooting it) and the three T06 statistics above. State the verdict
in the methods whichever way it falls; if SEFL loses, that is a result about a continuous-field model
needing less self-supervision than a point-cloud one, not an embarrassment.

### A4 is an addition experiment, and repulsion-off alone became vacuous

**Restated 2026-08-25, when `layout_mode` shipped as `resample`.** A4 was "repulsion off (Poisson
layout)", an override of a shipped `field` layout. That override no longer does anything to the
shipped configuration: `sample_layout` returns `_resample_layout` before the interaction is read,
so under the default, `repulsion=False` changes nothing and an A4 arm defined that way would
report a difference of exactly zero — a null that says nothing about the layout head. Left
unamended it would have been a silent no-op arm in the ablation table.

So A4 becomes the ablation of **the generative layout as a whole**, and like A2 and A7 it is an
*addition* experiment — the thing under test ships off:

| arm | override | what it answers |
|---|---|---|
| **A4a** | `layout_mode=field` | does sampling positions from the learned intensity beat copying the flanking section |
| **A4b** | `layout_mode=hybrid` | does the sliced-Wasserstein polish toward the flank marginals recover the difference |
| **A4c** | `layout_mode=field`, `repulsion=False` | the original A4, now stated *inside* a field-based mode where it is not a no-op: point-process realism via `g(r)` |

**Every arm A1-A8 must be verified to change behaviour under the *shipped* config before its
number is quoted** — SPEC_QUESTIONS **C33**, raised by this restatement and now a rule. A4 is the
third arm to have been inert or backwards because the thing it overrides ships off; the first two
(A2, A7) were caught because a weight was visibly zero, and this one was not, because
`repulsion=False` is a real change to a real field that a control-flow decision three levels away
makes moot. The check is one generation per arm with no fit: same seed, shipped config against the
arm's override, assert the outputs differ.

A4a and A4b **report the cell count and its per-section spread beside every metric**, not the
metrics alone. `specs/05` §4a: the count is what decided the mode, it is unstable 3.7x between
refits of one configuration, and a density-matched score cannot see it. An A4 table that reports
only the six metrics would show a 0.10 localization gap and hide a 60x count error.

The claim A4 carries is therefore a **negative** one, and `specs/10` §2 frames it as such: v25
implements a generative marked point process, it is available and measured, and on real tissue it
loses to resampling real coordinates. That is a result about the method, not a gap in it — the
mechanism is built, tested (`tests/test_layout.py`, `tests/test_layout_sampler.py`) and reported.

### A4's pair-correlation comparison must run over `[0, 3R]`

**Do not report A4 against `g(r)` restricted to `[r0, 3R]`.** T05 originally stated the criterion
over that range and it **cannot fail**: a hard-core process differs from a Poisson one only *inside*
the correlation hole, and the hole ends at about `r0`, because `r0` **is** a low percentile of the
nearest-neighbour distances. Measured on the fixture, real `g` pooled over training sections and the
simulated one over three seeds:

| Range | `field` mode | pure Poisson (A4) |
|---|---|---|
| `[r0, 3R]` | 0.093 | **0.070 — indistinguishable from the full model** |
| `[0, 3R]` | 0.093 | **0.994** |

Over `[r0, 3R]`, A4 is a **false null**: the table would say the repulsion buys nothing while `g(r)`
below `r0` says it is the difference between tissue and confetti. `specs/05` is amended to
`[0, 3R]`; `tests/test_layout.py` asserts both ranges, the second precisely so the blindness stays
visible. Any A4 number in the report states its range. The same caution applies to every A4
companion metric: choose statistics that can see inside the hole (nearest-neighbour distance
distribution, `g(r)` from 0) rather than ones evaluated only where the two processes agree by
construction.

### A5 must be run in the wide-gap regime, at the derived window

**...and not at a fixed `retrieval_z_window`.** On `consecutive-3` the default window of 3 × median
spacing leaves 100–110 of every 512 cells with **no admissible donor at all** after the own-section
exclusion, so the retrieval branch is silently absent for a fifth of them and an ablation of
`retrieval_w_z` would be measuring the window. `specs/09` §1 requires the window to be derived from
the gap; A5 runs against the derived window with the **empty-pool fraction reported beside the
ablation delta**. This is the trap G2.3 fell into with `retrieval_candidates_per_section`: the
ablation read as a no-op, with the wrong sign, until the cap was raised.

**Do not report A5 from `alternating`, and do not report it with the whole stack admissible.**
GATE 2's G2.3 measured it both ways with the two arms sharing a training seed, so initialisation,
batch order and per-step rotations were identical and the retrieval score was the only difference:

| Candidate pool | R² lost by `w_z = 0` at fractional depth 0.2 / 0.5 / 0.8 |
|---|---|
| Two flanking sections, near one 1 spacing away and far one 4 (the wide-gap regime) | **+0.0303 / +0.0034 / +0.0486** |
| Whole stack admissible | +0.0004 / +0.0034 / +0.0019 — **inside the noise** |

The reason is mechanical, not statistical. With every section admissible the *nearest* section is
always in the pool and in-plane distance alone ranks it first, so the z term has nothing left to
decide. It earns its place only when the evidence is far and asymmetric — which is the regime
in-silico sectioning actually lives in, and the one the competing method's score cannot see. Run
whole-stack and A5 reports a **null result for a term that demonstrably works**.

A5 is therefore reported at `consecutive-3` and `consecutive-5` (**Tier 2**, where the gap to the
nearest real section is 2–3 spacings), and the report states which regime each A5 number came from.
An `alternating` row is fine as a second, labelled dense-evidence control; it is not the headline.

**Check `retrieval_candidates_per_section` before trusting any A5 number.** The invariant that makes
the retrieval score do anything is about the candidate **union**:
`candidates_per_section × n_admissible_sections` must exceed `retrieval_k`, or the top-K returns the
whole pool and the score decides nothing. A wide-gap holdout is exactly where the number of
admissible sections is smallest, so this is exactly where it bites. `Config.validate` enforces
`retrieval_candidates_per_section >= retrieval_k` and `RetrievalIndex.query` warns
(`InertScoreWarning`) when a query's union falls to `K` or below. **An A5 run that emits that
warning is void.**

---

## 7. Capability experiments E1–E5 — `spatialcpav25_gen/eval/experiments.py`

```python
def exp_zero_shot_genes(...)       # E1
def exp_cross_panel(...)           # E2
def exp_oblique_validation(...)    # E3
def exp_throughput(...)            # E4
def exp_intersection_agreement(...)# E5
```

### E1 — zero-shot genes. **BLOCKED, not descoped.** Home: `deep_starmap`, not STARmap

**STARmap's 28-gene panel cannot measure E1.** 20% of 28 is 5–6 genes. Per-gene Moran's and Geary's
correlations over 5 genes are noise, and `MARKER_GENES = ("Flt1", "Pcp4", "Cux2")` are 3 of the 28,
so a random split has a large chance of taking a marker and silently disabling `paper_marker_*`.

**`deep_starmap` is the home** (Tier 2): 1 017 genes, mouse brain, `raw_counts` (so the ZINB path,
§5.1), 137 cell types, 198 675 cells — and crucially `paper_2_4_6`, **the same design as tier 1**,
so the zero-shot row is read against a headline-shaped comparison rather than a differently-shaped
one.

**Gene-subset scoring is free — do not build it.** `evaluate_paper` computes
`common = intersect1d(pred.gene_names, gt.var_names)` and scores on that intersection. Emit a
prediction whose `gene_names` is only the held-out 20%, and every metric lands on exactly the
zero-shot genes, on the pinned instrument, with no bench3 change.

One catch: markers and layer genes come from `uns['paper_protocol']`, so a split that excludes them
makes `paper_marker_*` unavailable on that arm. **Run two gene splits** — markers-held-out and
markers-seen — so the marker family is reportable either way, and say which split each number is
from.

### ⚠️ RESULT — E1's finding is the **seen/unseen sign flip**, not the primary comparison

Measured on `deep_starmap`, `paper_2_4_6`, three seeds x two fits x four arms x two folds x two
gene pools, 2026-08-31. Full table `reports/t09_zeroshot_deep.md`; the arms are A1 `medcpt`+distill,
A2 `medcpt` pure text, A3 `lookup`+distill, A4 `lookup` pure text (`norm(0)`, one vector for every
gene).

**The headline is a sign reversal between seen and unseen genes, within the same fits and the same
run:**

| gene pool | which embedding wins on `morans_pearson` | margin | envelope | signs |
|---|---|---|---|---|
| **kept** (the model was fitted on them) | `lookup` — a free per-gene table | **−0.1330** (A1−A3) | 0.0230 | 5.8x, 6/6 |
| **held out** (never in a batch) | `medcpt` — text alone | **+0.2999** (A2−A4) | 0.0532 | 5.6x, 6/6 |

A free lookup table wins where it has a row and loses where it does not. That is the
open-vocabulary claim stated as a measurement rather than as a motivation, and it is the first
within-run evidence for it in this project: one set of weights, one scoring pass, the sign of the
text channel's value flipping with nothing changing but which genes are being scored. The kept
half also reproduces the two established three-seed negatives on a third gene pool, so the losing
direction is not new — what is new is that the same comparison reverses on unseen genes.

**The ceiling is now measured and the gain survives it.** `morans_pearson`'s model-free ceiling
on the held-out genes is **0.9956** against a shuffled floor of +0.0382 — room **0.9574** — so
+0.2729 is **25% of what any method could reach**, and A2 clears that floor by **2.52x** the
shared envelope. Not most of the room, and not a rounding error either: the task is nowhere near
saturated (copying a real section reaches 98% of it), so there is three quarters of a capability
still unclaimed. `reports/t09_zeroshot_ceiling_morans_deep.json`.

⚠️ **It is still not the pre-registered primary, and that matters.** `marker_depth_r` was named
primary before any fit; `morans_pearson` is reported because the run's only positive landed there,
and a metric promoted to primary *because* it produced a result is not a test. The right status
for this number is a strong observation with its arithmetic shown, and a **pre-registered
replication on `morans_pearson`** — ideally on a second dataset — before it is written as a claim.

**The primary comparison found nothing, and not because the metric could not tell.**
`marker_depth_r` on the held-out genes: A1 − A3 = −0.0044 against a 0.1273 envelope, signs
disagreeing across seeds *and* folds. Read against the shared envelope (§4.2b) neither A1 (0.24x)
nor A3 (0.27x) clears the constant-field band — the pre-registered **REFUTATION OF THE IDEA** on
that metric. The void condition holds (A4 at 0.12x), so no leak.

That metric's ceiling is **0.9823** with room **0.9806**, and the best arm uses **4%** of it while
copying a real section uses 96%. So this is not the saturation that sank the `deep_starmap`
reconstruction comparison: the room is there and the arms cannot reach it. The measurement is also
**underpowered on its own terms** — A1's across-seed envelope is 4x its own distance above the
band, and a range over three draws does not shrink fast enough for a few more seeds to fix.

**The distillation head is a cost, not a contribution — and it is what breaks the claim.**
A1 − A2 on held-out `morans_pearson` = **−0.1533** (3.3x the envelope, 6/6 signs): adding
`gamma psi(t)` to the text channel makes it *worse* on the one metric where the text channel
works. On the kept genes the same contrast is −0.0008 (0.1x, signs disagree) — it does nothing
there either. Against the measured floor this is decisive rather than cosmetic: **A2 clears at
2.52x and A1 at 0.87x**, so the head takes the arm from clearing to not clearing.

The reading, now that the ceiling is in: **the pure-text path `W t` is the contribution and the
machinery around it is not.** That is a smaller claim than the architecture was designed to make
and a better-supported one, and it points at a specific change — either drop the residual path for
unseen entities or gate it on something that knows when it is unreliable, rather than adding it
unconditionally.

**The split is not the explanation.** Descriptor coverage on the two sides is the same: 192/204
held-out genes carry a summary (94.1%) against 774/813 kept (95.2%), a gap of −1.1 points, with
**zero** bare symbols on either side and median descriptors of 546 vs 572 characters. Twelve
held-out genes lack a summary where a metadata-blind draw predicts 10.2 ± 2.8. So A1/A2 were not
handicapped by a thin text channel, and — worth stating in the methods — **83% of the summaries
are a human orthologue's, labelled as such in the descriptor**, so what A2 demonstrates is that
MedCPT places a *mouse* gene from mostly *human* text.
`reports/t09_zeroshot_text_coverage_deep.json`.


### E1 runs on the campaign machine only

**The gene-metadata half is unblocked** (2026-08-20). `resources/gene_meta.parquet` was rebuilt from
the union of all three mouse panels: **2 155 symbols, 2 010 with summaries (309 native, 1 701 human
orthologue, 145 none), all ENSMUSG**, covering STARmap, Zhuang and `deep_starmap`. Verified in this
container: 2 155 rows, **1 020 all-uppercase symbols resolving without folding**, sources
`{native: 309, ortholog: 1 701}`. So the case-folding workaround the pilot identified is no longer
needed — the table carries `deep_starmap`'s spelling natively — and no `mygene` network access is
required to run E1.

**The data half is a campaign-machine dependency, permanently.** `deep_starmap` lives at

```
/data/han/projects/Spatial3D/benchmark-pbya-v3/data/processed/deep_starmap/data.h5ad
```

on the campaign server, and is **not reachable from the development container**. This is not a
blocker to route around: **write E1 to run on the server and treat it as unrunnable locally**, the
same way the comparator re-runs are (§3). Concretely:

* `eval/experiments.py::exp_zero_shot_genes` takes the dataset **by path** and resolves it at run
  time. It never assumes a local file, and it fails with the path it looked for rather than
  silently substituting the fixture.
* Its unit tests run on the **synthetic fixture** — gene-split construction, the two summary arms,
  the two distillation arms, and the scoring adapter are all exercisable without `deep_starmap`.
  What cannot run locally is the measurement, not the code.
* The E1 row in the definition of done stays outstanding until the server produces it.

**Gene-subset scoring is still free**, which is what keeps E1 cheap once it can run: emit a
prediction whose `gene_names` is only the held-out 20 % and `evaluate_paper`'s gene intersection
does the rest, on the pinned instrument, with no bench3 change.

⚠️ **What was measured about the older table**, retained because it is the reason the rebuild was
needed.
`resources/gene_meta.parquet` is exactly `zhuang_abca2_panel_symbols.txt` (1 122) ∪
`starmap_panel_symbols.txt` (28) = **1 138 symbols**, verified by set arithmetic — no other panel is
in it. It is mouse-cased: **3** of its 1 138 symbols are all-uppercase. `deep_starmap`'s symbols are
**uppercase** despite being mouse (`FLT1`, `PCP4`, `CUX2` in its own marker list), so on an exact
match **every one of its 1 017 symbols misses**. Two things follow:

1. **Case normalisation was mandatory against the old table** — measured at the pilot against
   `deep_starmap`'s uppercase marker and layer genes: **0 of 6** exact, **6 of 6** folded.
   **Superseded by the rebuild**, which carries the uppercase spellings natively. Keep the folding
   test as a regression guard, not as a required step.
2. **Coverage over the full 1017 was the open question, and the rebuild answers it.** The old table
   was 1 138 symbols built for Zhuang-ABCA-2 + STARmap only, so a `deep_starmap` build against it
   would have required a `mygene` rebuild — network access that `progress/t02_text_embeddings.md`
   records as 403'd in this container (SPEC_QUESTIONS C14), and therefore the long pole. The
   2 155-symbol union table removes that dependency: **E1 no longer needs the network**, only the
   server's copy of the volume.

**E1 reports both distillation arms** (settled; `design/v23_design.md` §2.2 / §7). A held-out gene
has no learned residual `r_g`, so the table shows *both* `forward_zero_shot(use_distill=False)` —
the pure-text arm, `r_g = 0` — and `use_distill=True`, the distilled `r_g = psi(t_g)`. Both exist
and are shape-tested at T02. One arm alone cannot separate "the text channel carries the gene" from
"the distillation head guessed a residual", which is the whole claim. These are **generation-time**
arms: one fit is scored twice.

**`load_gene_meta(path, species=...)` raises on a table of the wrong organism** (added after a mouse
panel's table came back holding four other mammals' genes and nothing noticed). E1 passes
`Config.mygene_species`, and the headline text quotes the table's own coverage — `gene_meta_summary`
reports rows, resolved taxid, how many rows carry a summary, and the Ensembl-id prefix histogram —
because "the model decodes unseen genes at r = X" means nothing without knowing the descriptors were
real. **One table per organism**, at different `Config.gene_meta_path`s.

**Coverage must be quoted split by `summary_source`, never as one number.** Mouse NCBI summaries
cover 148/1138 (13%) of the current table, so `Config.gene_summary_fallback="ortholog"` (the
default) backfills the rest from the 1:1 human orthologue's summary, labelled in the descriptor
text. `gene_meta_summary` reports `summary_sources` as `native / ortholog / none`; a bare
"N/1138 carry a summary" hides whether the text the encoder read was mouse biology or human. E1
therefore reports **two summary arms**, both filters on the `summary_source` column of one table
rather than two builds:

| arm | descriptors |
|---|---|
| native-only | rows with `summary_source == "native"` keep their summary; the rest are `"{symbol}. {full_name}."` |
| with fallback | as built |

If zero-shot transfer holds only on the fallback arm, the claim is that *human* gene descriptions
transfer to a mouse model — true and interesting, and not the same sentence as the design's.

Fit count: **2 gene splits × 2 summary arms = 4 fits per seed** (the descriptors change what the
model trains on); × 3 seeds = 12. The 2 distillation arms are free.

### E2 — cross-panel. Pair: STARmap ↔ `exseq_visual_cortex`

Train on A's panel, generate B's. bench3 scores a prediction against **one** dataset's ground truth,
so this needs a pairing driver, not a harness — and it only means anything for a **same-tissue**
pair. STARmap (28 genes) and `exseq_visual_cortex` (42 genes) are both mouse visual cortex,
`raw_counts`, and bench3's own README gives that as the reason ExSeq is the chosen analogue: same
tissue, so the marker genes and the laminar axis carry over unchanged.

⚠️ ExSeq is small and thin — **1 130 cells, 5 sections, `paper_2_4`, 28% unannotated, and only 1 of
3 markers resolved (`Cux2`)**. Report E2 with those numbers in the caption; the marker family is
effectively one gene on that side. Both directions, 3 seeds: **6 fits**.

### E3 — oblique validation. Home: the re-sectioned STARmap (§9)

Train on z-sections, generate x-sections, score against the **real** cells of the re-sectioned
volume. This is the oblique claim validated on real data with real ground truth rather than on the
fixture. **Tier 2** — the design is modified by construction. Overlaps V2; implement once, reference
from both. Read §9 before costing it: the geometry has a hard constraint.

### E4 — throughput. A structure-recovery figure, not a scored row

Generate at 10× the training z-density and show the recovered fine 3-D structure. **There is no
ground truth at 10× density** — the volume does not contain sections between its own sections — so
E4 produces **no scored metric and no claim-bearing number**. It is a qualitative figure plus
self-consistency diagnostics (continuity of the generated stack, absence of a spike at a training
section's depth — `test_stack_coherence` already pins both). Generation-only from an existing fit:
**0 extra fits**. Amend the original spec's implication that it is scored.

### E5 — intersection agreement. Cheapest, most decisive, no bench3 dependency

Generate two intersecting oblique sections with each method and measure agreement along the
intersection line as a function of dihedral angle. The competing method optimises each slice
independently, so its two sections have no mechanism forcing agreement where they cross; ours share
one 3D noise field and are trained for it. Expect a categorical rather than incremental gap.

**E5 needs no ground truth and therefore no bench3 at all** — it measures mutual coherence between
two generated slices. One panel, minimal compute, **0 extra fits**. Run it early, as soon as T09
lands, because it is the figure that establishes the contribution is structural.

⚠️ **T09 measured the criterion and half of it is above its ceiling** (SPEC_QUESTIONS C27):
concordance **0.814** against the spec's 0.8 (ceiling 0.781), expression correlation **0.724**
against the spec's 0.85 with a measured ceiling of **0.726** — two independent draws of one plane
under one realisation. The headline test asserts concordance absolutely and correlation
**ceiling-relative**; the literal 0.85 is a strict xfail.

---

## 8. SEFL validations V1–V4

These validate the sectioning-equivariance claims specifically. They are what justify SEFL as a
scientific contribution rather than a regulariser, so they are not optional. All are **Tier 2**.

```python
def val_resectioning_cycle(...)   # V1
def val_orthogonal_specimen(...)  # V2
def val_anisotropy_prediction(...)# V3
def val_thickness_transfer(...)   # V4
```

**V1 — virtual re-sectioning cycle.** From a coronally-sectioned volume, generate a full sagittal
stack; treat that generated stack as input and regenerate the original coronal sections; compare
against the real ones. End-to-end and ground-truthed, and it cannot be passed by memorisation
because the intermediate representation is entirely synthetic. Report the six target metrics on the
round trip against a single-pass generation as the ceiling. **Degradation over the cycle is the
quantity of interest — report it, do not hide it.** Needs the §9 preprocessor for the sagittal
target. 1 fit on the generated stack per seed: **3 fits**.

**V2 — orthogonal-specimen validation.** Train on a coronally-sectioned specimen, generate sagittal
sections, compare against a *different* specimen actually sectioned sagittally. Comparison must be
**distribution-level** (Sinkhorn on cell-state distributions, laminar profile agreement, cell-type
localization), never per-cell — the specimens are different animals. Overlaps E3; implement once.

⚠️ **No bench3 dataset is a second specimen sectioned on a different axis.** V2's literal form has
no data. What §9 provides is the *same* specimen re-sectioned, which is E3, and which is a **stronger**
ground truth (real cells, same animal) but a **weaker** generalisation claim (no cross-animal
transfer). State that distinction in the methods rather than presenting E3 as if it were V2. If a
genuinely sagittal second specimen is ever available, V2 is re-opened; **0 extra fits** either way.

**V3 — anisotropy prediction (the equivariant-column payoff).** From the fitted 3D covariance
structure, *predict* how in-plane Moran's I should vary with section angle; verify against real
sections cut at different angles. This is the correct use of the quantities T07 forbids
constraining: they are predicted, not matched. A model that had merely memorised a stack of 2D fits
cannot pass this. Report predicted-vs-observed r across angles. Reads a fitted model plus §9's
angle sweep: **0 extra fits**.

**V4 — thickness transfer. Reinstated as a real-data experiment, with a decoder caveat and a better
primary design.**

The cross-dataset pair is real: `merfish_hypothalamus` (thin, 12 sections at 50 µm spacing, 155
genes) and `merfish_thick_hypothalamus` (a 200 µm block cut into 7 slabs of ~27 µm, 156 genes) —
same tissue, near-identical panel, same technology family.

⚠️ **But the two sit on different decoders.** `merfish_hypothalamus` is `normalized` → `zigamma`;
`merfish_thick_hypothalamus` is `raw_counts` → `zinb` (§5.1). A thin↔thick transfer measured across
a decoder change is not measuring `L_thick` — it is measuring `L_thick` confounded with an
unvalidated decoder swap.

**So V4 has two rows, and the primary one is not the cross-dataset pair:**

| row | design | decoder | claim-bearing |
|---|---|---|---|
| **V4a (primary)** | `merfish_thick_hypothalamus` re-partitioned by §9 into **14 slabs of ~13.5 µm** (thin) vs its shipped **7 slabs of ~27 µm** (thick) — same volume, same cells, same panel, thickness the only variable | `zinb` both sides | ✅ |
| V4b (secondary) | `merfish_hypothalamus` (thin) ↔ `merfish_thick_hypothalamus` (thick), as originally proposed | `zigamma` ↔ `zinb` | ❌ — labelled, decoder confound stated |

V4a is a clean single-variable experiment and it costs nothing extra: `merfish_thick_hypothalamus`
is `partition="z_width"`, so a 14-slab build is a re-partition of the same point cloud — exactly
what §9's preprocessor does, emitted as a new dataset id, with the shipped 7-slab build untouched.

Metric for both: agreement of binned expression totals and per-type counts. **Include an ablation
with `w_thick=0`** to show the loss is what buys the transfer. 2 arms × 3 seeds on V4a: **6 fits**;
V4b diagnostic at 1 seed: **2 fits**.

---

## 9. The re-sectioning preprocessor — `spatialcpav25_gen/eval/resection.py`

STARmap is a real 3-D point cloud at single-cell resolution, so re-slicing it along a different axis
gives **genuine orthogonal sections with real ground truth**. A model trained on z-sections that
reproduces x-sections is the oblique claim validated on real data — not on the synthetic fixture,
and not by proxy. This is what makes E3, V1 and V3 measurable on the pinned instrument instead of on
a bespoke rig, and it is why it is worth building.

```python
def resection(source_h5ad, out_h5ad, *, normal, n_sections, dataset_id, seed) -> Path
def repartition(source_h5ad, out_h5ad, *, n_sections, dataset_id, seed) -> Path
```

### It emits a new dataset id and nothing else changes

The output is an ordinary bench3 `data.h5ad` — same `obsm['spatial']`, `obs['section']`,
`obs['cell_type']`, and a `uns['paper_protocol']` our preprocessor writes (marker genes, layer
genes, held-out sections) so `evaluate_paper` reads its panel from the file exactly as for any other
dataset. It is written under a **new dataset id** and passed **by path** (`--dataset /path/to/...`),
so:

- `DATASET_SPECS` is not edited; **no existing spec, build or result changes**.
- The tier-1 `starmap_visual_cortex/data.h5ad` build is **byte-identical** — the preprocessor reads
  it (or the raw volume) and never writes to it. `test_resection_leaves_source_bitwise_identical`
  asserts the source SHA-256 before and after.
- `run_benchmark.dataset_meta` falls back to `REGISTRATION = "none"` for an unregistered name, which
  is correct for a re-sliced single imaging block; the driver passes `registration="none"`
  explicitly regardless.

`repartition` is the same machinery with `normal` unchanged — used by V4a (§8) to cut
`merfish_thick_hypothalamus`'s 200 µm block into 14 thin slabs instead of 7 thick ones.

### ⚠️ The geometry constraint — read this before costing E3

**Every bench3 volume is a thin slab, not an isotropic block.** STARmap after trimming is
approximately **1545 × 1545 × 77 µm** (lateral extent from bench3's own calibration comment,
`VOXEL_XY_UM = 0.859` × 1800 px = 1545 µm; 77 retained planes at `VOXEL_Z_UM = 1.0`). Re-slicing at
**90°** therefore produces sections that are:

- **~220 µm thick** (1545 / 7) against the original **11 µm** — a **20×** change in
  `Section.thickness`, which is a first-class field that `L_thick` and the slab-volume intensity
  integral both read. A 90° re-slice confounds the orientation claim with a 20× thickness change.
- **1545 × 77 µm in plane** — an aspect ratio of about **20:1**. `evaluate_paper` bins the marker
  field on a `FIELD_GRID = 20` square grid, so 20 bins would span 77 µm on one axis and 1545 µm on
  the other: `paper_marker_field_r` and `paper_marker_field_ssim` degenerate to a 1-D comparison.

This is a property of the data, not of the method, and it is not fixable by tuning. **So E3 sweeps
the angle rather than jumping to 90°:**

1. **Determine which in-plane axis carries the laminar gradient first.** STARmap's laminar axis is
   in-plane (`Cux2` superficial → `Pcp4` deep) and `evaluate_paper.laminar_axis` derives it from the
   ground truth. If the gradient runs along x, re-slicing along x puts every section at a single
   cortical depth and **destroys the very structure the marker family measures**; if along y, it
   survives. One line from the built dataset settles it. **Measure it before choosing the normal.**
2. **Sweep `normal` over a set of angles** from the original z-normal (e.g. 15°, 30°, 45°, 60°, 90°),
   emitting one dataset id per angle. Ground truth is real cells at every angle, so all of them are
   valid; what varies is footprint and thickness.
3. **`resection` refuses a build whose footprint aspect ratio exceeds
   `Config.resection_max_aspect`** (proposed default 4.0) or whose section thickness departs from
   the source's by more than `Config.resection_max_thickness_ratio`, and the error names the angle
   and the measured values. A silently degenerate field metric is exactly the failure this guard
   exists to prevent (Convention 6). 90° is expected to trip it on STARmap; report that as a
   **result about the data** — "the published volume is too thin to be re-sectioned orthogonally" —
   rather than forcing it through.
4. **Every re-sectioned number states its angle, footprint and thickness**, and the anisotropy
   sweep is what V3 reads.

Also inherited from bench3: the build refuses any section under 50 cells and prints
`cells/section: min/median/max`. STARmap gives 28 978 / 7 ≈ 4 140 per section at any angle, so the
cell-count floor is never the binding constraint — the footprint is.

`deep_starmap` (125 µm z span, 199 k cells) and `merfish_thick_hypothalamus` (170 µm z span) are the
two other true 3-D point clouds worth re-sectioning; both are still slabs, so the same guard
applies. `easi_fish_lha*` have the thickest blocks (213–261 µm) but only 26 genes and are `zigamma`.

---

## 10. CLI — `spatialcpav25_gen/cli.py`

```
spatialcpav25-gen fit      --data X.h5ad --out runs/foo      # includes select_config + calibration
spatialcpav25-gen generate --run runs/foo --plane oblique --angle 45 --n 20 --out slices.h5ad
spatialcpav25-gen bench    --config bench.yaml --out reports/
spatialcpav25-gen report   --results reports/results.parquet --out reports/figures/
```

`fit` takes **no method flags** — configuration is selected internally (T09 §3). That is a claim in
the paper; make sure it is literally true of the CLI.

`bench` is a **driver over bench3**, not a harness: it resolves the campaign matrix from the YAML,
sets `BENCH_V3_RESULTS` per cell, calls `run_benchmark.run_single`, then hands the roots to
`eval/stats.py`. It asserts `evaluate_paper.py`'s SHA-256 before and after (§0).

### 10.1 How v25 is invoked — the same shape as v20, with no tuning flags

**Requirement: v25 runs the way v20 runs.** Same command shape, same `--dataset` and `--design`
arguments, resumable, one command per run.

```bash
# shipped run — the headline invocation, no flags after the --
python -m src.bench3.run_all --methods spatialcpav25_gen --dataset allen_merfish_brain

# wide regime
python -m src.bench3.run_all --methods spatialcpav25_gen --dataset allen_merfish_brain \
    --design wide --holdout-block 7

# one ablation arm (A1) — its own results root, never mixed with the shipped run
BENCH_V3_RESULTS=runs/t2/ablation_a1/seed_1 \
python -m src.bench3.run_all --methods spatialcpav25_gen --dataset allen_merfish_brain \
    -- --prior-mode iid

# one seed
BENCH_V3_RESULTS=runs/t1/headline/seed_2 \
python -m src.bench3.run_all --methods spatialcpav25_gen --dataset allen_merfish_brain --seed 2
```

#### The flag policy: v20 took eleven tuning flags, v25 takes none

v20's wrapper accepted `--edit-weight`, `--gap-scale`, `--alpha-tol` and eight more. **v25 accepts
no tuning flags at all** — configuration is selected internally per dataset (`specs/09` §3), and
*"`fit` takes no method flags"* is a claim in the paper. **A bare invocation must therefore produce
the shipped configuration**, and there must be no flag that could produce a different one by
tuning.

Nothing named `--gap-scale`, `--alpha-tol` or `--edit-weight` exists anywhere in the package (T09
removed the concepts, not just the flags), so there is nothing to re-expose. The flags after `--`
are exactly two kinds, and the wrapper's `argparse` carries no third:

| kind | flags | effect |
|---|---|---|
| **shared `_v2_io`** | `--input`, `--target-section`, `--target-z`, `--output`, `--seed` | supplied by `run_benchmark`; `--seed` is the seed switch |
| **ablation switches** (§6) | `--prior-mode {correlated,iid}` (A1) · `--w-autocorr/--w-profile/--w-distribution` (A2) · `--text-emb-mode {medcpt,lookup-only}` (A3) · `--no-repulsion` (A4) · `--retrieval-w-z` (A5) · `--decoder {zinb,zigamma,gaussian}` (A6) · `--w-thick/--w-prog` (A7) · `--prog-wrong` (A8) · `--layout-mode`, `--expr-mode`, `--train-steps` (gate overrides for the selection-recovery checks) | each overrides exactly one `Config` field **after** the selected config is loaded |

Every ablation switch defaults to `None`, meaning *"do not override"*. `method_params` in the
prediction records the resolved `Config.content_hash()` and every override actually applied, so a
result is self-describing and a bare run is provably bare.

⚠️ **`--results-root` is not a wrapper flag and cannot be.** `run_benchmark.run_single` resolves
`out_dir` from `config.RESULTS_DIR` *before* it invokes the wrapper, so a wrapper flag could not
move where the prediction is written. The results root is the **`BENCH_V3_RESULTS` environment
variable**, set per (tier, arm, seed) by the driver or by hand, as in the examples above (§4.1).

⚠️ **A8 has no `Config` gate yet.** `loss_prog_WRONG` is built (`losses/sefl.py:1139`) and
deliberately excluded from the weighted loss set, so there is no field for `--prog-wrong` to set.
**T10 must add `Config.w_prog_wrong: float = 0.0`** — a one-field addition in T01's style, with the
docstring naming it the A8 negative control — or A8 cannot be run as a config override as §6
requires. Flagged rather than assumed: it is an omission in T07's wiring, not in T10's.

`test_bare_invocation_reproduces_shipped_config` asserts that a bare wrapper run's recorded
`content_hash` equals the hash in the dataset's persisted selection, and that its applied-override
list is empty.

#### Where per-dataset selection runs, and how it is reused

**This is the difference between a one-command run and a day-long one, so it is specified rather
than left to the implementation.** Selection is ~23 fits (`specs/09` §3: a joint budget × weights
gate, a merged 18-cell full-budget gate, then coordinate-descent passes). Running it inside every
invocation would make each bare run cost ~23 fits before producing a single section — on
`allen_merfish_brain` (~45× fixture weight, §12) that is weeks, not a run.

**Selection runs once per dataset, out of band, and is persisted.** It is not part of a benchmark
run; a benchmark run *resolves* it.

```
$SPATIALCPAV25_SELECT_DIR/            (default: runs/select/)
  <dataset_id>/
    selected.yaml           the chosen Config, plus the provenance block below
    scores.csv              the ScoreCache — every scored cell, flushed per row
    selection_report.md     write_selection_report's table (every cell, not the winner)
```

* **Keyed on `dataset_id`** — `uns['dataset_name']` from the built file, falling back to the
  resolved path's stem, so a derived dataset (§9) gets its own directory and can never reuse the
  source's selection.
* ⚠️ **`ScoreCache.key` is `f"{cfg.content_hash()}:{steps}"` and does NOT include the dataset.**
  One cache file shared across datasets would silently return dataset A's score for a dataset B
  cell. **The per-dataset directory is what makes the cache correct, not merely tidy** —
  `test_score_cache_is_per_dataset` pins it.
* **`selected.yaml` carries a provenance block**: `dataset_id`, the base `Config.content_hash()`,
  the selection seed, and a **volume fingerprint** (n cells, n genes, sorted section ids, and the
  built `data.h5ad`'s mtime). The wrapper **refuses a selection whose fingerprint does not match
  the input it was handed**, naming both — the same staleness class bench3 already guards for its
  own `_inputs/` cache, where a stale cache cost a fix being applied twice.
* **Shared across every seed, arm and tier.** Selection is 1-seed by classification (§4.2) and an
  ablation *inherits* the shipped config by definition (§6), so the three headline seed roots and
  every ablation root resolve the same `selected.yaml`. It deliberately does **not** live under
  `BENCH_V3_RESULTS`.

**How the wrapper resolves it**, in order:

1. `selected.yaml` exists for this `dataset_id` and its fingerprint matches → load it, fit once,
   generate. This is the steady state: **one command, one fit.**
2. It is absent → the wrapper **runs selection inline, persists it, then proceeds.** It prints the
   plan and the estimated cost first (`~23 fits at <dataset> scale`), so the bill is visible before
   it is incurred rather than discovered afterwards. `scores.csv` is flushed per cell, so an
   interrupted first run **resumes** rather than restarts — which is what makes a single command
   survive a dataset this size.
3. It is present but **stale** (fingerprint mismatch) → **raise**, naming the dataset, both
   fingerprints, and the command to re-select. Never silently re-select and never silently reuse:
   a config selected against a different build is exactly the silent fallback Convention 6 forbids.

Two switches control step 2, and neither tunes anything:

```bash
--select-only     run selection, persist it, write no prediction   # pre-warm a dataset
--require-config  refuse to select; raise if selected.yaml is absent  # for campaign runs,
                  #  so an unattended job cannot silently start a 23-fit selection
```

**Recommended for a campaign**: pre-warm each headline dataset once with `--select-only`, then run
every seed and arm with `--require-config`. The pilot (§11) does exactly this at step 6, which is
also where the ~23-fit selection cost is measured for real rather than modelled.

---

## 11. The pilot — run this before any campaign

**STARmap only, end to end.** Nothing else is run until the pilot has been read.

| step | what |
|---|---|
| 1 | Assert the `evaluate_paper.py` SHA-256; record it |
| 2 | Install the environment; build `starmap_visual_cortex` (tier 1, unmodified) |
| 3 | `python -m src.bench3.selftest` — the four probes, on the real build |
| 4 | Add `run_spatialcpav25_gen.py` + the one `METHODS` entry; confirm `run_all` with no arguments still plans exactly the previous campaign |
| 5 | Comparators on tier 1: SpatialZ, FEAST, isoST, v20, plus `oracle` and `flanking_copy` — **required, because the existing numbers are not in this repo (§3)** |
| 6 | Config selection on STARmap via `--select-only` (§10.1), persisted to `runs/select/starmap_visual_cortex/`. **Time it** — §12 models it as the single largest line item and the pilot is what replaces the model with a number. Every later step runs `--require-config` |
| 7 | Headline six-metric table at **3 seeds**, three results roots |
| 8 | `eval/stats.py` on those roots: Wilcoxon, BH, bootstrap CI, Cliff's delta, forest plot |
| 9 | **E5** — cheapest and most decisive, no bench3 dependency |
| 10 | **One ablation: A1** (`prior_mode=iid`) — the simplest override, it exercises the per-arm results-root mechanism, and it is the term **C2** rests on |
| 10b | **C1 and C2 read out** (§11.1), including the per-cell-type localization breakdown and the tier-1 `wide_3_4_5` head-to-head |
| 10c | **`marker_field_r` diagnosed, not hoped away** (§13.4) — report the alignment record beside it |
| 11 | Measure the laminar-gradient axis and the re-sectioning footprint (§9) — a measurement, not a run |
| 12 | Measure `deep_starmap` gene-table coverage after case-folding (§7) — a measurement, not a run |

### 11.1 The two named pilot criteria

The pilot does not merely produce six numbers. **Two of them are named criteria**, chosen because
they are where the prior campaign says the method must improve, and the pilot is read against them
first. Both are stated as questions with a measured referent, not as targets to hit — §13 explains
why the referents have to be re-measured rather than quoted.

**C1 — `paper_celltype_localization` on tier 1.** This is the one tier-1 metric where the published
competitor leads the whole SpatialCPA line, and it is precisely what v25's intensity-field layout
head (T05: per-type intensity field, fitted Strauss/hard-core repulsion, fitted Potts mark
smoothing) exists to attack. Measured in the prior campaign (§13.2 — medians per §4.6, and
cross-instrument, see the warning there): SpatialZ **0.8372**, best internal version **0.8227**,
v20 **0.7760**. It is SpatialZ's **only** tier-1 win, against every SpatialCPA version.

> **The question C1 answers:** does v25 close the localization gap to SpatialZ on STARmap under
> `paper_2_4_6`, **per held-out section**, at 3 seeds, with each section's gap read against the
> across-seed envelope *and* against the `flanking_copy`→`oracle` interval measured for it?

#### C1's primary form is per section, against both referents

**A pooled localization number cannot settle C1, and the pilot measured why.** `flanking_copy` —
bench3's own probe, which copies the nearest training slice and has no model at all — scores
`section_2` at **0.7008** against `section_4`'s **0.7765** and `section_6`'s **0.7868**. That is a
**0.086 positional swing**, produced by geometry alone. SpatialZ's entire tier-1 lead over v20 is
**0.061**. A pooled score is therefore dominated by how a method handles one held-out position, and
two methods with identical layout quality can differ by more than the gap being claimed.

**So C1 is defined per section, against both referents, and the pooled number is reported only
beside them:**

| | `section_2` | `section_4` | `section_6` |
|---|---|---|---|
| `oracle` (ceiling, measured) | 0.9765 | 0.9888 | 0.9808 |
| **v25** | — | — | — |
| `flanking_copy` (floor, measured) | **0.7008** | 0.7765 | 0.7868 |

Both referents come from `selftest.make_probe` on the pinned instrument (measured, `reports/pilot.md`
§3), so a v25 row drops straight in. Report, for each section: the raw score, the **fraction of the
`flanking_copy`→`oracle` interval** it covers, and the gap to SpatialZ. Then the pooled median,
labelled as a summary of the three rather than as the result.

The per-section split is what distinguishes the two readings §1 sets out — a gap concentrated in
`section_2` is boundary robustness, a uniform gap is layout quality — and after the pilot this is no
longer a hypothesis: a model-free probe already shows the positional effect is **larger than the
method effect**. A C1 conclusion drawn from a pooled number is not supportable.

Report it **per cell type as well as pooled** (§2's rule 5): the metric normalises by the divergence
to a within-tissue null, and that null collapses for an abundant tissue-wide type, so a pooled move
can come entirely from types with no headroom. Report `paper_rare_celltype_localization` beside it —
rare-niche placement is the half of the claim the layout head most directly addresses.

**C2 — the wide regime.** No SpatialCPA version has established a wide-gap win, and §13.3 shows the
evidence runs the other way on the metrics the claim is about: by median, SpatialZ beats v20 on
`morans_pearson` in **6 of 7** wide holdouts and `gearys_pearson` in **5 of 7**. And **there is no
head-to-head wide comparison on STARmap in the prior campaign at all**, so the tier-1 case has never
been measured. Wide-gap performance is the stated reason earlier versions
were not publishable, and it is what the correlated 3D prior (T03) and the z-proximity retrieval
term (T04, ablation A5) are for.

> **The question C2 answers:** on STARmap `wide_3_4_5`, at 3 seeds, does v25 beat SpatialZ on the six
> target metrics — and is the margin larger than at `paper_2_4_6`? Read the autocorrelation family
> (`morans_pearson`, `gearys_pearson`) and `marker_field_r` separately: in the prior campaign they
> point in **opposite** directions in the wide regime (6/7 and 5/7 to SpatialZ, 0/7 to v20).

**The pilot therefore generates the first tier-1 wide head-to-head this project has ever had.** That
means running SpatialZ (and v20 as the no-regression reference) at `--design wide` as well as at
`paper_2_4_6` — an addition to pilot step 5, costed in §12 as 2 extra comparator runs.

Neither criterion is pass/fail for the pilot. A criterion that does not move is a **result** and is
reported as one; what is not acceptable is a pilot report that does not state where both landed.

**Report:** what one real-data fit actually costs (wall clock, peak RSS, CPU vs GPU), the measured
across-seed envelope, what broke, and the answers to steps 11 and 12. **Then** the 3-vs-5 dataset
decision is taken with real numbers rather than estimates.

---

## 12. Cost

### The compute assumption, stated explicitly

**Every timing this project has ever measured is CPU.** `specs/10`'s "~29 min per full-budget fit"
and T09's selection runs (19 fits / 9 195 s ≈ 8 min each at the reduced budget; 23 fits / 21 330 s
≈ 15.5 min each at 2400 steps) were all measured **on CPU, on the synthetic fixture** — 9 sections ×
1 500 cells = 13 500 cells, 200 genes. `torch==2.2.2` is pinned and CUDA is available in principle,
but **no GPU timing exists for this codebase**, so a GPU figure would be a guess. The budget below
is therefore **CPU**, and any GPU speed-up is unmeasured upside to be established by the pilot.

⚠️ **The pilot measured a real fit and the model is ~2.3x optimistic.** One 1200-step fit on
tier-1 STARmap (16 527 cells x 28 genes) took **1712.7 s = 28.5 min** of training, 1729.4 s total
wall, 1709 MB peak RSS, uncontended on CPU — so a 2400-step fit is **~57 min**, against the 25 min
this model assumes. **STARmap's FEF weight is therefore ~2.3, not 1.0**, and every figure below
scales with it: the three-dataset campaign is closer to **220 CPU-hours than 96**. Re-derive this
table from the pilot's number before taking the 3-vs-5 decision. (A run sharing cores with the test
suite took 2224 s — 30 % slower — so quote uncontended figures only.)

The natural unit is a **fixture-equivalent fit (FEF)**: one full-budget fit at 2400 steps on
fixture-scale data, **≈ 25 min CPU** (T09's 15.5 min measured, rounded up for real-data overhead).
Panel width is nearly free — `genes_per_step` is what makes it so — so datasets scale by **training
cell count**:

| dataset | training cells | FEF weight |
|---|---|---|
| synthetic fixture | 13 500 | 1.0 |
| `starmap_visual_cortex` | 16 527 (sections 1/3/5/7) | **1.0** |
| `merfish_thick_cortex` | ~16 500 | 1.0 |
| `exseq_breast_cancer` | ~1 300 | 0.1 |
| `merfish_thick_hypothalamus` | ~45 000 | 3 |
| `deep_starmap` | ~113 000 | **8** |
| `cosmx_nsclc_3d` | ~227 000 | **17** |
| `allen_merfish_brain` | ~600 000 | ~45 — excluded |

### The bill, by table

Three headline datasets: `starmap_visual_cortex` (tier 1), `merfish_thick_cortex`,
`exseq_breast_cancer`. `deep_starmap` and `merfish_thick_hypothalamus` are experiment homes, not
headline datasets, and **inherit the shipped config** (§6) rather than re-running selection.

| Table / experiment | Arms × regimes × datasets × seeds | fits | **FEF** |
|---|---|---|---|
| **Per-dataset config selection** (specs/09 §3) | ~23 × 3 headline ds × 1 seed | 69 | **48** |
| Headline six-metric table | 3 ds × 3 regimes × 3 seeds | 27 | 19 |
| Boundary rows, R3 (STARmap, both ends) | 2 × 1 ds × 3 seeds | 6 | 6 |
| A1, A3, A4, A6 | 4 arms × 2 ds × 1 seed | 8 | 4 |
| A5 | 1 arm × 2 wide regimes × 2 ds × 1 seed | 4 | 2 |
| A2 (claim; two budgets) | 2 budgets × 2 ds × 3 seeds | 12 | 7 |
| A7 (claim) | 2 ds × 3 seeds | 6 | 3 |
| A8 (claim) | 2 ds × 3 seeds | 6 | 3 |
| **E1 zero-shot** (`deep_starmap`) | 2 gene splits × 2 summary arms × 3 seeds | 12 | **96** |
| E2 cross-panel (STARmap ↔ ExSeq V1) | 2 directions × 3 seeds | 6 | 6 |
| E3 oblique (re-sectioned, ≤3 admissible angles) | 3 angles × 3 seeds | 9 | 9 |
| E4 throughput | — | 0 | 0 |
| E5 intersection agreement | — | 0 | 0 |
| V1 re-sectioning cycle | 1 fit on the generated stack × 3 seeds | 3 | 3 |
| V2 orthogonal specimen | — (folded into E3) | 0 | 0 |
| V3 anisotropy prediction | — (reads a fitted model) | 0 | 0 |
| V4a thickness (re-partition, `w_thick` on/off) | 2 arms × 3 seeds | 6 | 18 |
| V4b thickness (cross-dataset, diagnostic) | 2 × 1 seed | 2 | 6 |
| Achievable ceiling (fixture, 8 draws) | — | 0 | 0 |
| Baselines: convex-interp, independent-donor, nearest-copy | — | 0 | 0 |
| **Total** | | **176 fits** | **≈ 230 FEF** |

**≈ 230 FEF × 25 min ≈ 96 CPU-hours ≈ 4 days serial**, or under a day across five cores — the runs
are independent per (dataset, arm, seed), so wall clock is a scheduling choice.

**Plus the comparators**, which are not v25 fits: SpatialZ, FEAST, isoST and v20 on 3 datasets × 3
regimes = 36 method runs, plus 2 probes × 3 datasets × 3 regimes = 18 probe evaluations. Individually
far cheaper than a fit, and **required** — §3 established that the existing v20/v22 numbers are not
in this repository.

### The three things that dominate, and what to do about them

1. **Config selection: 48 FEF, 21% of the total, and absent from every earlier estimate.**
   `specs/09` §3 selects per dataset (~23 fits: a coordinate descent plus two non-descended joint
   gates). It is 1-seed by classification and `ScoreCache` checkpoints it, so it resumes. The lever
   is dataset count, not the selector: **each additional headline dataset costs ~23 fits of
   selection before it produces a single headline number.** That is the real content of the 3-vs-5
   decision, and it is why the pilot takes it.
2. **E1 on `deep_starmap`: 96 FEF, 42% of the total** — 12 fits at 8× weight. It is the price of a
   panel wide enough to measure zero-shot at all (§7), and the alternative (`allen_zhuang_abca2`) is
   ruled out by `zigamma`. If it proves prohibitive after the pilot, the honest lever is bench3's own
   `--max-cells-per-section` on a **new** `deep_starmap` dataset id (additive, §9's `repartition`
   machinery) — subsampling changes the dataset, so it would be reported as such, not silently.
3. **`cosmx_nsclc_3d` is a cost bomb: 17× weight.** Adding it as the non-brain headline dataset
   would add ~23 × 17 = **391 FEF of selection alone** — more than doubling the campaign for one
   row. `exseq_breast_cancer` (0.1×, human breast cancer, `raw_counts`, 297 genes) discharges the
   non-brain requirement for ~1% of that, at the cost of small-n (1 979 cells, 57 in its thinnest
   section). **Recommendation: `exseq_breast_cancer` for the campaign; `cosmx_nsclc_3d` only if the
   pilot shows fits are much cheaper than modelled.**

A five-dataset campaign adding `deep_starmap` and `cosmx_nsclc_3d` as headline datasets would run
**~240 fits / ~1 000 FEF ≈ 415 CPU-hours**, a 4.3× increase driven almost entirely by those two
volumes' cell counts. Take that decision on pilot numbers.

---

---

## 13. Evidence from the prior campaign — and why it cannot be a baseline

A `per_section_metrics.csv` from an earlier campaign was recovered (the machine holding the full
results tree is down). 829 rows: 18 datasets, 8 methods (`spatialz`, `feast`, `isost`, and
SpatialCPA `v14`, `v18`, `v19`, `v20`, `v21`), across `paper_*` and `wide_*` designs. It is the only
comparator evidence this project currently holds, and it is worth reading carefully — but **none of
its numbers may be quoted as a baseline**, for the reason in §13.1.

Everything below is the **median over held-out sections**, per §4.6. An earlier pass through this
file used means and inverted one finding (`marker_depth_r`); the medians are the numbers, and the
per-section values are given wherever the two disagree.

### 13.1 ⚠️ The file is evaluator-heterogeneous — the tier-1 comparison is cross-instrument

`paper_marker_field_ssim`, `paper_gene_detection_spearman` and `paper_rare_celltype_localization`
are populated on **132/132** rows for `v18`, `v20` and `v21` — every dataset — but on only **33/132**
for `spatialz`, and **14/76** for `feast` and `isost`.

On `starmap_visual_cortex` specifically: `v20` has them on **3/3** rows, `spatialz` on **0/3**.

**So the two sides of the tier-1 head-to-head were scored by different revisions of
`evaluate_paper.py`.** That is precisely the failure SPEC_QUESTIONS A3's SHA-256 pin exists to
prevent: "a reimplementation that merely agrees closely is not comparable — the claim is a
*difference* between methods measured on one instrument." Whether the *shared* metrics moved between
the two revisions is **unknown** and cannot be determined from the CSV; `celltype_localization`
gained its rare-type split in the same change that added these columns, so it is not safe to assume
it did not.

**Consequences, both already in this spec and now doubly justified:**

1. **§3's "re-run the comparators" is not only about missing files.** Even a recovered results tree
   would need every prediction re-scored on the pinned evaluator. bench3's
   `evaluate_all --force` does exactly that **without re-running any method**, so if the predictions
   are ever recovered, re-scoring is cheap and is the first thing to do.
2. **No number in §13.2–13.4 enters a paper table.** They are read as *directional signals* that set
   the pilot's criteria (§11.1), and every one of them is re-measured on the pinned instrument
   before it is quoted.

Also missing from that campaign: **`v22`, `v23` and `v24` were never run**, so the internal
development table has a three-version gap, and `v19` appears on only 17 rows over a different
dataset subset than the others — its apparent wide-regime strength (§13.3) is not comparable with
the rest and must not be read as one.

### 13.2 C1 — cell-type localization is where the competitor leads

Tier 1, `starmap_visual_cortex` / `paper_2_4_6`, cell-count-weighted over the three held-out
sections:

| method | `celltype_localization` | `marker_depth_r` | `marker_field_r` | `morans_pearson` |
|---|---|---|---|---|
| **spatialz** | **0.8372** | 0.9267 | 0.8522 | 0.9316 |
| spatialcpav21_gen | 0.8227 | **0.9817** | **0.8881** | 0.9831 |
| spatialcpav20_gen | 0.7760 | 0.9704 | 0.8804 | **0.9837** |
| spatialcpav14_gen | 0.7733 | 0.8950 | 0.8540 | 0.9411 |
| feast | **0.0000** | 0.7786 | 0.5717 | 0.7693 |
| isost | **0.0000** | 0.6645 | 0.6292 | 0.7754 |

**The signal holds and C1 is well-motivated**: SpatialZ leads the entire SpatialCPA line on
localization, by **+0.015** over the best internal version and **+0.061** over v20.

**`celltype_localization` is SpatialZ's only tier-1 win — against every SpatialCPA version, not
merely the best one.** Under the median it loses `marker_depth_r` to v20 (0.9267 vs **0.9704**) as
well as to v21, and loses `marker_field_r`, `morans_pearson`, `gearys_pearson`, `umap_mixing` and
`gene_mean_spearman` to both. It is a single, isolated, well-defined deficit — which is precisely
what makes it a good criterion to aim a mechanism at.

**Two things about how it must be read.**

* **FEAST and isoST score exactly 0.0000.** By the metric's own normalisation that is "no better than
  scattering the type anywhere in the tissue". Two published methods hitting the floor exactly is a
  **suspicious value, not a competitive one** — the likely cause is that neither emits usable cell
  types, which `evaluate_paper` cannot distinguish from a genuinely random placement. **Check this
  before any localization number is published**: if they emit no types, they are absent from the
  comparison rather than losing it, and the tier-1 `localization` group is a five-method race whose
  ranks change accordingly.
* **The margin is inside twice the reproducibility envelope.** T09 measured refit-at-same-seed drift
  up to **0.0120**. The **+0.015** gap to v21 is barely above it, which is why C1 is stated at
  **3 seeds with the spread reported** and never as a single-run comparison — and why a v25 result
  that merely ties v21 here says nothing at all.

### 13.3 C2 — the wide regime is unestablished in *both* directions

**There is no head-to-head wide comparison on STARmap.** `starmap_visual_cortex` / `wide_3_4_5`
contains **3 rows, all `spatialcpav19_gen`** — one method, no competitor. Every wide-regime statement
in circulation therefore comes from other volumes.

Pooled across the four datasets that do have wide runs (`allen_merfish_brain`, `allen_zhuang_abca2`,
`merfish_hypothalamus`, `st_mouse_brain_ortiz`):

| method | `morans_pearson` | `gearys_pearson` | n |
|---|---|---|---|
| **spatialz** | **0.9193** | **0.9205** | 33 |
| spatialcpav20_gen | 0.9173 | 0.9175 | 33 |
| spatialcpav21_gen | 0.9082 | 0.9105 | 33 |

**That pooled median is not a legitimate comparison** — it is exactly the cross-dataset pooling
bench3's README forbids ("averaging STARmap's and ExSeq's composites would compare places in two
different races") and this spec's own **Do NOT** forbids. **24 of its 33 rows are
`allen_merfish_brain`**, so it is close to a single-volume result wearing four volumes' clothes.

But per dataset and holdout — the legitimate view, and by median — **the autocorrelation deficit is
real and consistent**, not an artifact of the pooling:

| metric | wide holdouts where SpatialZ beats v20 |
|---|---|
| `morans_pearson` | **6 of 7** |
| `gearys_pearson` | **5 of 7** |
| `celltype_localization` | 4 of 7 |
| `marker_field_r` | **0 of 7** — v20 wins every one |

**So the wide-regime finding survives the correct estimator and is strengthened by it.** On the two
autocorrelation metrics — the ones the correlated 3D prior (T03) and the z-proximity retrieval term
(T04/A5) exist to move — the published competitor beats v20 on nearly every wide holdout it was run
on. The design docs' "wins decisively at wide gaps" is not merely unsupported; the available
evidence points the other way on the metrics the claim is about. (`marker_field_r` goes the opposite
way, 0 of 7 — see §13.4.)

**What is still unestablished is the tier-1 case**, and that is what C2 is for: `starmap` /
`wide_3_4_5` holds one method and no competitor, so this project has never measured a wide-gap
head-to-head on the protocol dataset. The pilot produces the first one — one volume, the pinned
instrument, 3 seeds, medians with CIs. A result that confirms the deficit is as publishable as one
that closes it, and either is better than the current state.

(`spatialcpav19_gen` tops several pooled wide columns. Ignore it: 14–17 rows over a **different**
dataset subset — it is the only method present on STARmap's wide design and absent from
`st_mouse_brain_ortiz` — so its average is over a different, easier mixture. This is the same
cross-dataset trap in miniature.)

### 13.4 `marker_field_r` — a known weakness to diagnose, stated precisely

`paper_marker_field_r` was **v25's single loss on the synthetic fixture at T09**, and it is where
SpatialZ is relatively strongest against the SpatialCPA line in the prior campaign. Two generations
pointing at one metric is worth a diagnosis rather than a hope. But the claim needs stating exactly,
because the pooled and per-dataset views disagree:

| view (all medians) | result |
|---|---|
| **Tier 1** (`starmap` / `paper_2_4_6`) | **v20 (0.8804) and v21 (0.8881) BEAT SpatialZ (0.8522).** It is not a tier-1 loss |
| **Wide holdouts, per dataset** | **v20 wins 7 of 7** — the one family where the wide regime favours SpatialCPA |
| Pooled over all 18 datasets and designs | favours SpatialZ — but this is the forbidden cross-dataset average |
| **Per dataset, `paper_*` designs, SpatialZ vs v20** | **9 wins each — a dead tie** |

So: **not a systematic deficit.** It is a metric on which the SpatialCPA line has no margin in the
`paper_*` designs, wins outright in the wide ones, and on which v25 regressed once on the fixture at
T09. Treat it as a watch item with a named diagnostic path, not as a known defect — and note that it
and the autocorrelation family point in **opposite** directions in the wide regime (§13.3), which is
itself worth understanding rather than averaging away.

**The diagnostic, and why it is cheap.** `marker_field_r` is one of only **two pose-dependent
metrics** (with `celltype_localization`): both are computed after `align.py` rotates the prediction
into the ground-truth frame, and `evaluate_paper` writes the whole alignment record into every
section's entry — `align_rotation_deg`, `align_score`, `align_runner_up`, `align_coverage`,
`align_basis`. So a field-r deficit can be an **alignment** artifact rather than a fidelity one, and
the evidence to tell them apart is already in the output.

**Report the alignment record beside every `marker_field_r` number**, and treat a small
`align_score − align_runner_up` margin as a caution flag on that row.

Checked on tier 1 in the prior campaign: rotations are **0.0°–0.3°** for every SpatialCPA method and
**0.0°–1.5°** for SpatialZ, with `align_score` tracking `field_r` closely. **Alignment is not
confounding the tier-1 ordering** — v20 and v21 genuinely beat SpatialZ there. That result also
means the diagnostic has a clean tier-1 reference: if v25's field-r drops while its rotations stay
at 0°, the cause is the field, not the pose, and the place to look is T05's intensity head and the
`FIELD_GRID = 20` binning rather than `align.py`.

Also report `paper_marker_field_ssim` beside `field_r` wherever both exist. `r` is invariant to an
affine rescale of the field, so a marker reproduced with the right *shape* but the wrong *level*
passes `r` and fails SSIM — and per §13.1 the competitor rows largely lack SSIM, which is a further
reason the comparators must be re-scored.

---

## Acceptance tests

- `test_evaluate_paper_sha256_unchanged` — the pinned SHA-256 matches **before and after** a campaign
  run; the assertion names the file on mismatch.
- `test_metrics_match_bench3_bitwise` — the adapter and `bench3.evaluate_paper` return `==` values,
  not `allclose`, on a fixed synthetic pair.
- `test_bench3_footprint_is_two_files` — the only bench3 paths T10 writes are
  `methods/run_spatialcpav25_gen.py` and one appended `METHODS` key; asserted by diffing the bench3
  tree against a recorded manifest.
- `test_method_order_unchanged` — `config.METHOD_ORDER` is byte-identical to its pre-T10 value, so a
  bare `run_all` plans exactly the previous campaign.
- `test_resection_leaves_source_bitwise_identical` — the source `data.h5ad`'s SHA-256 is unchanged
  after `resection` and `repartition`.
- `test_resection_refuses_degenerate_footprint` — an angle exceeding `Config.resection_max_aspect`
  raises and names the angle and the measured ratio.
- `test_tier_purity` — a tier-1 table containing a non-`paper_2_4_6` or non-STARmap row raises.
- `test_stats_never_reads_summary_by_method` — `eval/stats.py`'s inputs are `all_metrics.csv` and
  `per_section_metrics.csv` only.
- `test_claim_coverage` — a measurement classified in neither branch of `CLAIM_BEARING` blocks the
  headline table.
- `test_rankdata_ties` — a vector with 60% zeros gets identical average ranks for all zeros.
- `test_spearman_is_spearman` — a monotone-nonlinear transform leaves the value unchanged.
- `test_spatialz_wrapper_no_mutation` — input AnnData is byte-identical after the call.
- `test_baselines_run_on_fixture` — all five produce valid AnnData.
- `test_benchmark_resumable` — kill and restart; results identical, completed cells skipped
  (exercises bench3's `--skip-existing`, which T10 does not reimplement).
- `test_stats_bootstrap_ci` — on synthetic data with a known median difference, the CI covers truth
  in ≥ 94% of 200 simulations.
- `test_metric_registry_complete` — all six target and ≥ 5 control metrics registered with direction
  and range.
- `test_no_headline_statistic_is_a_mean_over_sections` — `assert_no_mean_over_sections` raises on any
  headline or claim-bearing statistic computed by averaging the `section` axis, naming the metric.
- `test_median_and_mean_disagree_on_the_fixture` — a fixed 3-section case in which the two
  estimators invert the verdict, so the rule's reason stays exercised rather than asserted.
- `test_bare_invocation_reproduces_shipped_config` — a bare wrapper run records a `content_hash`
  equal to the dataset's persisted selection, with an empty applied-override list.
- `test_score_cache_is_per_dataset` — `ScoreCache.key` omits the dataset, so two datasets sharing one
  cache file is a correctness bug; the per-dataset directory is asserted.
- `test_stale_selection_raises` — a `selected.yaml` whose volume fingerprint does not match the input
  raises, naming both fingerprints and the re-selection command.
- `test_no_tuning_flags` — the wrapper's `argparse` exposes only the shared `_v2_io` arguments and the
  §10.1 ablation switches; any flag that could change the fitted configuration without being an
  ablation fails the test.
- `test_no_cross_dataset_pooling` — any statistic computed over rows spanning more than one `dataset`
  raises unless explicitly constructed as a labelled cross-dataset diagnostic.
- `test_headline_comparators_are_published_methods` — a tier-1 headline table whose comparator set
  contains a `spatialcpav*` method other than v25 raises (§3).
- `test_gene_meta_case_folding` — an uppercase mouse symbol resolves against the mouse-cased table,
  and the resolved-only-after-folding count is reported.

## Definition of done

`reports/results.parquet` + a rendered `reports/benchmark.md` carrying, **each labelled with its
tier and holdout id**:

- the tier-1 headline six-metric table (STARmap, `paper_2_4_6`, 3 seeds, min–max spread) with the
  campaign envelope stated;
- the statistics: Wilcoxon + BH, median difference with 95% bootstrap CI, Cliff's delta, and the
  forest plot (Figure 2);
- the tier-2 tables: wide regimes, boundary rows, analogue datasets, each separate;
- the ablation table A1–A8, with A2 at both budgets and A5's regime and empty-pool fraction stated;
- the control-metric table;
- the achievable-ceiling table (fixture) and the `oracle` / `flanking_copy` referents (real data),
  with every method number reported raw **and** ceiling-relative;
- the gene–gene covariance rows reported **as a loss**, in both regimes, beside the ceiling;
- E1 (both distillation arms × both summary arms × both gene splits, with the table's own coverage
  quoted by `summary_source`), E2, the E3 angle sweep, the E4 figure, the E5 figure;
- V1's cycle degradation, V3's predicted-vs-observed r, V4a and the labelled V4b;
- the `evaluate_paper.py` SHA-256, recorded before and after.

- every statistic a **median with its 95% bootstrap CI** (§4.6), with the per-section values shown
  beside every tier-1 median;
- **C1 and C2** (§11.1) read out explicitly: the localization gap to SpatialZ pooled *and* per cell
  type, and the tier-1 `wide_3_4_5` head-to-head — each against the across-seed envelope;
- `paper_marker_field_r` reported with its alignment record and `field_ssim` beside it (§13.4).

`PROGRESS.md` records the headline median gaps and the V1 cycle degradation;
`progress/t10_benchmark.md` carries the full log.

## Do NOT

- Do not edit any bench3 file beyond the two in §0. If something cannot be done additively,
  **stop and report which** — do not edit and explain afterwards.
- Do not touch `evaluate_paper.py`, for any reason, including a bug.
- Do not add to `METHOD_ORDER` or to `DATASET_SPECS`.
- Do not read `summary_by_method.csv` for any published number.
- Do not merge a tier-2 number into a tier-1 table, or average across tiers.
- Do not average across regimes.
- Do not tune the competing method's hyperparameters, in either direction.
- Do not report only the metrics that were trained against — the control table is not optional.
- Do not let the two v20 metric bugs be described as fixed on the benchmark; they were never there.
- Do not report a `zigamma` number as claim-bearing until the decoder and its calibration are
  validated (§5.1).
- Do not report E4 as a scored metric.
- **Do not compute any headline or claim-bearing statistic as a mean over sections** (§4.6). With
  n = 3 the median and the mean disagree often enough to change a conclusion, and `section_2` is a
  systematic outlier, not noise.
- Do not add a tuning flag to the v25 wrapper. The flags after `--` are ablation switches and the
  seed; anything that could change the fitted configuration otherwise breaks the paper's
  "no method flags" claim (§10.1).
- Do not run selection inside a campaign invocation — pre-warm with `--select-only` and run with
  `--require-config` (§10.1).
- Do not frame any headline claim around beating an internal SpatialCPA version — the competitors are
  SpatialZ, FEAST and isoST; v14–v24 are development history and v20 is the no-regression reference
  (§3).
- Do not quote a number from the prior campaign as a baseline: its two sides were scored by different
  `evaluate_paper.py` revisions (§13.1).
- Do not pool metrics across datasets — the wide-regime "result" in the prior campaign is an artifact
  of exactly that (§13.3).
- Do not assert the wide-gap advantage anywhere until a tier-1 wide comparison exists (§4.4, C2).
