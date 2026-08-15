# PROGRESS — SpatialCPA-v25-Gen

Status of every task in `specs/`. Update the row when a task lands, then append an entry under
"Log" with what was built, the test numbers, and any deviation from the spec and why.

Status values: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`.

| # | Task | Spec | Key deliverables | Gate | Status |
|---|---|---|---|---|---|
| T00 | Project scaffolding | — | `CLAUDE.md`, `PROGRESS.md`, `pyproject.toml`, `Makefile`, ruff/mypy/pytest config, `SPEC_QUESTIONS.md` | — | DONE |
| T01 | Config and data contracts | `specs/01_TASK_config_and_data.md` | `config.py`, `data/schema.py`, `data/loaders.py`, synthetic fixture | — | DONE |
| T02 | Text-grounded embeddings | `specs/02_TASK_text_embeddings.md` | `data/text.py`, `model/embeddings.py`, MedCPT cache, distillation head | — | DONE |
| T03 | 3D GRF noise field | `specs/03_TASK_noise_field_GATE1.md` | `model/noise.py`, `scripts/gate1_report.py`, `reports/gate1.md` | **GATE 1** | BLOCKED — built and tested; **GATE 1 failed on G1.3c/G1.3d** |
| T04 | Anatomical field + retrieval | `specs/04_TASK_field_and_retrieval_GATE2.md` | `model/field.py`, `model/retrieval.py`, `scripts/gate2_report.py`, `reports/gate2.md` | **GATE 2** | TODO |
| T05 | Layout head | `specs/05_TASK_layout_head.md` | `model/layout.py`, intensity + Strauss sampler + Potts marks | — | TODO |
| T06 | Expression head + ZINB decoder | `specs/06_TASK_expression_head.md` | `model/expression.py`, `model/spatialcpav25_gen.py`, `losses/reconstruction.py` | — | TODO |
| T07 | SEFL consistency losses | `specs/07_TASK_sefl_losses.md` | `losses/sefl.py`, `infer/planes.py`, EMA teacher, collapse alarm | — | TODO |
| T08 | Metric-aware LOSO losses | `specs/08_TASK_metric_aware_losses.md` | `losses/metric_aware.py`, `train/loso.py` | — | TODO |
| T09 | Inference + calibration | `specs/09_TASK_inference_and_calibration.md` | `infer/generate.py`, `infer/calibrate.py`, `train/select.py` | — | TODO |
| T10 | Benchmark + baselines | `specs/10_TASK_benchmark_and_baselines.md` | `eval/metrics.py`, `eval/baselines.py`, `eval/benchmark.py`, `cli.py` | — | TODO |

`specs/` defines ten implementation tasks (T01–T10); T00 is this scaffolding pass, listed so the
table covers everything that has been done to the repository.

## Gate status

| Gate | Criterion | Status | Report |
|---|---|---|---|
| GATE 1 (T03) | GRF prior halves median Moran's I error vs i.i.d.; per-gene r > 0.7; `I_gen` monotone in `ell` | **FAILED** — error ratio **0.196** (needs < 0.5) ✅, r **0.820** (needs > 0.7) ✅, monotone in `ell` ❌ (worst step **−0.065**), fitted `ell` within 25 % of the best match ❌ (**37 %**) | `reports/gate1.md` |
| GATE 2 (T04) | oblique R² ≥ 0.90 × axis-aligned R² | not reached | `reports/gate2.md` |

## Numbers the paper needs (fill in as tasks land)

| Quantity | Source task | Value |
|---|---|---|
| Text/co-expression Spearman (synthetic, then real) | T02 | synthetic: **+0.0055** (≈ 0, as expected — arbitrary gene names); real: pending a real panel + `resources/gene_meta.parquet` |
| GRF vs i.i.d. Moran's I error ratio | T03 | **0.196** (median \|I_gen − I_real\|: GRF 0.0776, i.i.d. 0.3956); Geary's C ratio 0.199; per-gene r 0.820 (GRF) vs 0.348 (i.i.d.) |
| Fitted `ell = (ℓx, ℓy, ℓz)` | T03 / T09 | synthetic fixture: **(137.6, 137.6, 355.5) µm** vs ground truth (120, 120, 200); `ell_z` is extrapolated (the 400 µm stack reaches only 59 % of the fitted sill) and warns |
| Oblique parity ratio | T04 | — |
| Fitted repulsion `r0`, `R`, `gamma`; Potts `beta` | T05 | — |
| Detection-rate r; gene–gene covariance vs independent-donor; mean–variance slope | T06 | — |
| Consistency/reconstruction loss ratio; collapse-alarm history | T07 | — |
| Metric-aware on/off table (ablation A2) | T08 | — |
| Selected per-dataset config | T09 | — |
| Headline median gaps; V1 cycle degradation | T10 | — |

## Log

### T00 — Project scaffolding (2026-08-14)

Read all of `specs/`, both design documents, and cross-checked the coverage matrix against them.
Created `CLAUDE.md` (conventions + naming + gates), this file, `pyproject.toml` (package
`spatialcpav25_gen`, CLI `spatialcpav25-gen`, pinned deps, ruff + `mypy --strict` + pytest with a
`slow` marker) and a `Makefile` (`test`, `test-all`, `lint`, `typecheck`, `install`, `format`).
No implementation code written by design.

Open spec questions raised before T01 begins: see `SPEC_QUESTIONS.md` — 29 items, of which 6 change
interfaces (§A) and must be settled before or during T01, 9 are acceptance tests that would fail for
reasons unrelated to the model (§B), 9 are under-specified points with a proposed default (§C), and
5 are design-doc components missing from the coverage matrix (§D).

### T01 — Config and data contracts (2026-08-14)

**Built.** `spatialcpav25_gen/config.py` (one frozen `Config`, 94 documented fields, `from_yaml` /
`to_yaml` / `from_dict` / `to_dict` / `replace`, `validate()`), `spatialcpav25_gen/data/schema.py`
(`Section`, `Volume`, `TrainingVolume`, `HeldOutSections`, `to_xyz`, `validate_volume`,
`validate_config_against_volume`, three warning classes),
`spatialcpav25_gen/data/loaders.py` (`load_volume`, `split_holdout`, `loso_folds`),
`tests/fixtures/synthetic.py` (`make_synthetic_volume`, `GroundTruthField`, `volume_to_anndata`),
`tests/conftest.py`, `tests/test_config.py`, `tests/test_schema.py`.

**Test numbers.** `make check` green: ruff clean, `mypy --strict` clean on 5 source files,
**25 tests pass in 6.2 s** on CPU (budget: 3 min). Nothing is marked `slow` yet.

Synthetic fixture at the defaults (9 sections × 1500 cells × 200 genes, 50 µm spacing, seed 0),
against the thresholds in the spec's `test_synthetic_has_structure`:

| Quantity | Required | Measured |
|---|---|---|
| mean Moran's I over genes (log1p, k=10 graph, middle section) | > 0.25 | **0.441** (median 0.472) |
| per-gene detection rate, minimum | < 0.05 | **0.0105** |
| per-gene detection rate, maximum | > 0.95 | **0.9985** (median 0.524) |
| genes with \|corr(mean expr, z)\| > 0.5 | ≥ 15 | **150** |
| neighbourhood cell-type purity above the label-permuted null | ≥ 3σ | **92.5σ** (purity 0.683 vs chance) |

Other fixture properties later tasks will lean on: overall zero fraction 0.480; median in-plane
nearest-neighbour distance 13.95 µm with a hard core at 7.75 µm (no pair closer); latent-field
correlation at a 120 µm lag is 0.502 in-plane vs 0.767 along z, i.e. the field is anisotropic in the
direction T04/T07 need; per-channel latent variance 1.00 ± 0.02; build time 1.8 s.

Leakage tests: `split_holdout` returns `TrainingVolume`/`HeldOutSections`, held-out ids are disjoint
from training ids, the training volume's `median_spacing` is recomputed (50 → 100 µm in the
alternating regime), endpoints are never held out, and `loso_folds` raises `TypeError` for both a
plain `Volume` and a `HeldOutSections`. `test_thickness_defaults_and_flags`: exactly one
`AssumedThicknessWarning`, `thickness == median_spacing` (50 µm), `thickness_is_assumed is True`.

**Deviations from the spec, and why.**

1. `split_holdout` returns `tuple[TrainingVolume, HeldOutSections]`, not `tuple[Volume,
   list[Section]]` (SPEC_QUESTIONS A1). `TrainingVolume` subclasses `Volume` and `HeldOutSections`
   is a `Sequence[Section]`, so spec-shaped call sites are unaffected; T08's
   `test_metric_aware_rejects_heldout` needs a runtime `TypeError`, which a `NewType` cannot give.
2. `split_holdout` takes an additional keyword `cfg: Config | None = None`. The consecutive run
   length is "configurable" in the spec but the signature has nowhere to read it from
   (SPEC_QUESTIONS A4); it now comes from `Config.holdout_consecutive_k` (1/3/5), which is also how
   T10's `consecutive-3` / `consecutive-5` regimes are expressed.
3. `alternating` holds out every other **interior** section, and `consecutive` never includes the
   first or last section. The spec's "flanking gap ≈ 1× median_spacing on each side" is only true
   if endpoints stay in training; holding out an endpoint makes the task extrapolation.
4. `Config.validate()` is self-consistency only; the data-dependent check the spec gives as an
   example (`fourier_bands_z > 4` with fewer than 8 sections) lives in
   `validate_config_against_volume(cfg, vol)`, called by `load_volume` (SPEC_QUESTIONS A5). A
   standalone `Config` has no volume to check against.
5. `make_synthetic_volume` returns `(Volume, GroundTruthField)` rather than stashing the field in
   `vol.uns["gt_field"]` — `Volume` has no `uns` and should not grow an untyped dict
   (SPEC_QUESTIONS C4). `Volume.median_spacing` / `median_nn_dist` / `bbox` are `field(init=False)`
   (C5).
6. `Config` carries 94 fields, not the 60 printed in the spec. The extra ones are the constants
   later task files write inline (SPEC_QUESTIONS A4) plus the ones T01 itself needed
   (`thickness_key`, `section_key`, `min_sections_per_volume`, `holdout_consecutive_k`,
   `small_volume_n_sections`, `max_fourier_bands_z_small_volume`, `metric_knn_k`). Four have no
   value fixed anywhere in `specs/` and are marked *provisional* in their docstring —
   `field_dim=128`, `retrieval_ctx_dim=64`, `retrieval_n_heads=4`, `expr_pca_dim=32` — to be set by
   T04/T08 when those land.
7. Non-integer counts behind a set `counts_layer` **warn** (`NonIntegerCountsWarning`) rather than
   raise, following the spec's prose ("warn loudly") over its test name ("rejects"). The warning
   names the ZINB decoder as what breaks.
8. Scaffolding: `pyproject.toml` now ignores ruff `ANN101`/`ANN102` (annotating `self`/`cls`),
   which are deprecated upstream and would otherwise force unidiomatic signatures.

**Flagged for later.** T08 §2 says the principal tissue axis is "stored on the `Volume`", but T01's
`Volume` has no such field and computing it needs a `TrainingVolume` (leakage). Recorded as
SPEC_QUESTIONS C10; left for T08 rather than added speculatively here.

### T02 — Text-grounded gene and context embeddings (2026-08-15)

**Built.** `spatialcpav25_gen/data/text.py` — `GeneMeta`, `gene_descriptor`,
`celltype_descriptor`, `region_descriptor`, `descriptor_key`, `build_gene_meta`, `load_gene_meta`,
`TextEncoder` (disk cache keyed by `sha256(text_model + descriptor)`, one `.npy` per descriptor,
atomic writes, lazy model load), `TransformerBackend` + the `load_transformer_backend` seam,
`GeneMetaUnavailableWarning`. `spatialcpav25_gen/model/embeddings.py` — `TextGroundedEmbedding`
(`W`, zeros-init `r`, `gamma` buffer, `distill` MLP, `set_progress`, `forward`,
`forward_zero_shot`, `distillation_loss`), `EntityEmbeddings` (the three instances: genes at
`gene_emb_dim`, cell types and regions at `ctx_emb_dim`), `text_embedding_diagnostics` (ablation
A3). `scripts/build_gene_meta.py` (the one-off online build). `tests/test_text.py` (20 tests),
`tests/fixtures/text.py` (a deterministic hash backend — MedCPT cannot be downloaded in the test
environment). 12 new `Config` fields, no constants outside it.

**Test numbers.** `make check` green: ruff clean, `mypy --strict` clean on 8 source files,
**45 tests pass in 9.1 s** on CPU (20 of them T02's; budget 3 min). Nothing marked `slow`.

| Acceptance test | Required | Measured |
|---|---|---|
| `test_descriptor_stability` | alias order irrelevant | exact string equality across 3 alias orderings, incl. duplicates and self-alias |
| `test_cache_hit_avoids_model_load` | second call from cache | `n_model_calls == 0`, vectors bitwise equal; backend constructor asserts if touched |
| `test_missing_gene_meta_degrades` | `"{symbol}."` and encodes | `"Xkr4."`, `(1, 768)` float32, ‖v‖ = 1 |
| `test_zero_shot_shapes` | `(10, out_dim)` | `(10, 128)` both distilled and pure-text arms |
| `test_gamma_anneal` | γ = 0 → output *is* `LayerNorm(W t)`; γ = 1 at end | `torch.equal` (bitwise) with a **non-zero** residual planted, so the gate is what is tested; γ(0.15) = 0.5 |
| `test_distillation_reduces_error` | MSE drops ≥ 50 % in 200 steps | **1.0155 → 1.29e-07 (ratio 1.3e-07)** in 0.4 s |
| `test_offline` | symbol-only rows, no raise | 3/3 rows symbol-only, one `GeneMetaUnavailableWarning`, table round-trips through `load_gene_meta` |

Diagnostics on the synthetic fixture (200 genes × 13 500 cells, descriptors = bare `GeneNNNN.`,
deterministic hash backend standing in for MedCPT):

| Quantity | Value | Reading |
|---|---|---|
| text/co-expression Spearman | **+0.0055** over 19 900 gene pairs | ≈ 0, exactly as the spec predicts for arbitrary gene names |
| kNN purity (k = 10) | **0.2315** vs chance **0.2259** (5 Leiden modules, sizes 57/53/40/33/17) | no module signal, same reading |
| `residual_norm_ratio` | 0.0 at init (zeros residual); 47.7 with a planted N(0, 1) residual | the ratio responds |

That the diagnostic can *detect* signal is tested separately (`test_diagnostics_sees_planted_signal`,
4 planted co-expression modules with text vectors aligned to them): Spearman **0.506**, kNN purity
**0.700**, which is the maximum reachable with 8-gene modules and k = 10, and 4 modules recovered.
So the ≈ 0 on the fixture is a fact about the fixture's gene names, not a broken statistic. **The
number that matters is the real-data one, and it cannot be computed until `resources/gene_meta.parquet`
exists** (SPEC_QUESTIONS C14).

**Deviations from the spec, and why.**

1. **Pooling is CLS, not mean** (SPEC_QUESTIONS C3, settled in T01 as `Config.text_pooling="cls"`).
   The spec says "mean-pool the last hidden state" and then says to use whatever the checkpoint
   specifies; MedCPT-Query-Encoder is trained contrastively on the first-token state. The choice is
   a `Config` field with the justification in a comment at the pooling site, as the spec asks.
2. `build_gene_meta(symbols, cfg=None)` takes the config as an additive keyword — the spec's
   `build_gene_meta(symbols)` has nowhere to read `gene_meta_path`, `text_allow_network` or
   `mygene_species` from. Same shape of deviation as T01's `split_holdout(..., cfg=None)`.
3. `text_embedding_diagnostics(emb, expr, *, seed)` gains a required keyword-only seed
   (SPEC_QUESTIONS C13): the Leiden partition and the gene-pair subsample are stochastic, and
   Convention 3 forbids leaving that to a global RNG. Two calls with the same seed return an equal
   dict, asserted.
4. **Network is opt-in, not "if available".** The spec says to assemble from mygene.info "if network
   is available"; probing availability from inside a test run is exactly what the "Do NOT" section
   forbids. `Config.text_allow_network` defaults to `False`, so the default path is offline and the
   online path is an explicit one-off (`scripts/build_gene_meta.py`). Degradation warns
   (`GeneMetaUnavailableWarning`) rather than being silent (Convention 6).
5. **Descriptor grammar pinned beyond the spec's format string**, because it is the cache key: every
   field is stripped of surrounding whitespace and trailing periods before joining, so a summary
   that already ends in "." does not produce ".." and a different hash. Aliases are sorted,
   de-duplicated, and an alias equal to the symbol is dropped. A region's ancestor path is *not*
   sorted — its order is the hierarchy.
6. `distillation_loss` is a **mean** over entities and components, not a sum (SPEC_QUESTIONS C15), so
   `w_distill` means the same thing on a 200-gene and a 20 000-gene panel.
7. `TextGroundedEmbedding.__init__` initialises `W` and `distill` from a generator seeded with
   `cfg.seed` instead of the global torch RNG (Convention 3); the distribution is the one
   `nn.Linear` would have used. `test_construction_is_deterministic` asserts bitwise equality across
   two constructions and inequality across seeds.
8. `test_distillation_reduces_error` is **not** marked `slow`, against the proposal in
   SPEC_QUESTIONS C6: at this size the 200-step loop is 0.4 s, and the 3-minute budget is nowhere
   near threatened. C6's rule still stands for the loops that actually cost something (T06–T08).
9. `tests/test_schema.py` was reformatted by `ruff format` (pinned 0.4.4). The version in
   `main` used the newer parenthesised-assert style, which the pinned formatter rejects, so
   `make lint` was already failing before this task. Formatting only, no behaviour change.

**Flagged for later.**

* **`resources/gene_meta.parquet` does not exist and cannot be built here** (SPEC_QUESTIONS C14).
  Every descriptor is currently the bare symbol. Someone has to run
  `python scripts/build_gene_meta.py --symbols-from <panel>.h5ad` on a networked machine and commit
  the table before the first real run, or the paper's text channel is symbols only and the A3
  diagnostic has nothing to report.
* Cell-type ontology records and region hierarchies are *accepted* (`celltype_descriptor`,
  `region_descriptor`) but nothing resolves them yet: T04/T06 pass what the dataset carries, and a
  dataset without an ontology gets the raw label, which is what the spec asks for.
* The zero-shot table needs both arms (design §2.2 / §7 E1): `forward_zero_shot(use_distill=False)`
  is the pure-text arm and `use_distill=True` the distilled one. Both exist and are shape-tested
  here; T10 E1 reports them.
* Ablation A3 (`Config.text_emb_mode="lookup"`) is mapped to T10 in the coverage matrix and is not
  wired here; `TextGroundedEmbedding` is unconditionally text-grounded.

### T03 — 3D Gaussian random field prior ⛔ **GATE 1 FAILED** (2026-08-15)

**Built.** `spatialcpav25_gen/model/noise.py` — `GaussianRandomField` (RFF Matérn field over R³:
`__call__` / `forward` torch path, `evaluate_numpy`, `covariance`, `with_lengthscale`, buffers
`directions` / `omega` / `phase` / `amplitude`), `matern_correlation`, `scaled_distance`,
`fit_lengthscale_from_sections` + `fit_lengthscale_details` (`LengthscaleFit`),
`LengthscaleFitWarning`, `VariogramError`. `tests/gate1_criteria.py` (the four gate criteria as
`Criterion` records, measured once and consumed by both the test suite and the report),
`tests/fixtures/planes.py` (a minimal canonical `SamplingPlane` + `intersection_line`, superseded by
T07's `infer/planes.py`), `tests/test_noise.py` (31 tests), `scripts/gate1_report.py`,
`reports/gate1.md` + four figures. 13 new `Config` fields (`variogram_*`), one changed default
(`grf_chunk_points` 65536 → 1024); no constants outside `Config` in the package.

**Test numbers.** `make check` green: ruff clean, `mypy --strict` clean on 9 source files,
**75 tests pass in 40 s** on CPU (31 of them T03's, 5 marked `slow`; budget 3 min). `make test-all`
is **1 failed, 75 passed**: the one failure is `test_gate1_3_autocorrelation_transfer`, which is the
failed gate asserting itself. It is deliberately left red rather than `xfail`-ed — a gate that stops
the project should not be green.

| Acceptance test | Required | Measured |
|---|---|---|
| `test_grf_zero_mean_unit_var` | \|µ\| < 0.02, var ∈ [0.97, 1.03] over 10⁵ points | max \|µ\| **0.0105**, var **[0.983, 1.013]** |
| `test_grf_channels_independent` | mean \|corr\| < 0.02 | **0.0028** (and exactly 0 by construction: max \|AᵀA\| off-diagonal **< 1e-9**) |
| `test_with_lengthscale_preserves_seed` | halve twice == quarter once | bitwise equal on `omega`, `phase`, `amplitude` and on the field values |
| `test_torch_numpy_agree` | max abs diff < 1e-5 | **4.2e-6** |
| `test_fit_lengthscale_recovers_truth` | fitted `ell_xy` within 30 % of 120 µm | **137.6 µm (+14.7 %)** |
| `test_matern_correlation_matches_sklearn` | — | max diff vs `sklearn.Matern` **< 1e-10** at ν = 0.5 / 1.5 / 2.5 / 4.0 |

**GATE 1 — FAILED on G1.3c and G1.3d.** Full report with plots in `reports/gate1.md`;
`scripts/gate1_report.py` exits non-zero.

| Criterion | Required | Measured | Verdict |
|---|---|---|---|
| G1.1a covariance vs analytic anisotropic Matérn, 4000 pairs, M = 4096 | MAE < 0.03 | **0.0121** | PASS |
| G1.1b error decreases with M | every step < 0 | **0.0232 → 0.0159 → 0.0121** (≈ 1/√M) | PASS |
| G1.2 two plane pathways along their intersection line, 256 points | bitwise identical | **max diff exactly 0.0** (coords agree to 2.8e-14 µm and round to the same float32) | PASS |
| G1.3a median \|I_gen − I_real\|, GRF ÷ i.i.d. | < 0.5 | **0.196** (0.0776 vs 0.3956) | PASS |
| G1.3b per-gene Pearson r(I_gen, I_real), GRF | > 0.7 | **0.820** (i.i.d. arm: 0.348) | PASS |
| G1.3c median I_gen monotone over `ell` = 0.25×–4× | every step > 0 | **−0.065** (0.271, 0.390, 0.418, 0.413, 0.348 — a peak at ≈ 1.6×) | **FAIL** |
| G1.3d fitted `ell` vs the best-matching `ell` | < 25 % | **37 %** (fitted 137.6 µm, best match 218.4 µm) | **FAIL** |
| G1.4a same seed in a second process | 0 differing values | **0** | PASS |
| G1.4b 10⁶ points at M = 4096, d_h = 64 | < 5 s | **3.4 s** (4-core Xeon @ 2.10 GHz, no GPU) | PASS |

**What the failure is, and what it is not.** It is *not* a failure of the prior: measured directly,
the median Moran's I of the field's own channels is monotone in `ell` and saturates near 1
(0.487 → 0.764 → 0.900 → 0.973 → 0.990 over the same sweep, diagnostic D2). The non-monotonicity is
in the *observable*. Moran's I of expression is a ratio — spatially structured variance over total
variance — and a stationary unit-variance field seen through a finite window loses within-window
variance once `ell` approaches the window: across the sweep the field's within-section sd falls
0.98 → 0.65 on the fixture's 1000 µm section, so the latent contributes less of the total expression
variance even as its neighbour-scale correlation approaches 1, and the ratio turns over. Rebuilding
the fixture at a 3000 µm field of view with the same density (diagnostic D1) nearly removes the
effect (0.212, 0.363, 0.470, 0.488, 0.4825 — worst step −0.005, inside the ±0.006 count-draw noise)
and makes `I_real` reachable, which locates the cause in `ell` / field-of-view rather than in the
field. G1.3d fails for the same reason: the best-matching `ell` is the argmax of a curve that never
quite reaches median `I_real` (0.445 at the peak vs 0.472 real).

**Consequence, and the recommendation.** The spec states G1.3c's purpose in its own words — "this is
what makes the T09 calibration loop well-posed — if `I_gen` is not monotone in `ell`, bisection will
fail". That risk is now quantified: `I_gen(ell)` is unimodal with its maximum near 0.2× the section's
in-plane extent, so T09's bisection is well-posed only when bracketed below that maximum, and when
the target exceeds the achievable maximum there is no root at all. My recommendation is to make
T09's calibrator maximum-aware (cap the bracket, detect the maximum, report "target unreachable")
and re-run G1.3c over the range calibration actually operates in — *not* to widen the fixture and
*not* to move a threshold. Recorded as SPEC_QUESTIONS **A7**. **T04 is not started.**

**Deviations from the spec, and why.**

1. **Amplitude normalisation is analytic, not measured.** The spec says to "renormalise `A` columns
   so realised marginal variance is 1.0 ± 0.02 (measure on a random sample of points and rescale)".
   The space-averaged marginal variance is exactly `‖A_c‖²/M`, so the columns are orthogonalised
   (QR, sign-canonicalised) and scaled to `‖A_c‖ = √M`. That makes unit variance and zero
   cross-channel covariance *exact* rather than a Monte Carlo approximation of them, and it settles
   SPEC_QUESTIONS **B3** (the 0.02 cross-channel threshold was at the O(1/√M) ≈ 0.016 noise floor;
   it is now 0 by construction). `E[A_c A_cᵀ] = I` still holds, so the covariance function is
   unchanged — G1.1 checks that empirically.
2. **Queries are evaluated in float32, not float64.** Coordinates are float32 by Convention 4, so a
   float64 phase would chase precision the inputs do not carry, and it triples the cost (the
   `(chunk, M)` block stops fitting in cache). The draws stay float64 so `with_lengthscale` does not
   accumulate rounding across a calibration loop. Measured worst-case phase error ~2e-5 rad → ~2e-5
   on a unit-variance field.
3. **`grf_chunk_points` default 65536 → 1024.** Purely performance: at 65536 the feature block is
   1 GB and the query is memory-bound (5.7 s per 10⁵ points); at 1024 it is 16 MB, cache-resident,
   and 10⁶ points take 3.4 s. Settles SPEC_QUESTIONS **B9** — the 5 s target is reachable on this
   CPU after chunking, so nothing had to be loosened.
4. **`fit_lengthscale_from_sections(vol, cfg, *, seed)`** takes a required keyword-only seed
   (Convention 3 — the per-section cell subsample is stochastic), and refuses `HeldOutSections` with
   a `TypeError` naming leakage. Same shape of deviation as T01's `split_holdout(..., cfg=None)` and
   T02's `text_embedding_diagnostics(..., seed)`.
5. **`fit_lengthscale_details` is an added entry point.** Same computation, returning the curves that
   were fitted (`LengthscaleFit`), because the report has to plot them and a surprising `ell` has to
   be diagnosable. `fit_lengthscale_from_sections` is the spec's signature and returns `.ell`.
6. **`covariance(p, q)` is an added method**, and GATE 1's G1.1 uses it. Estimating the covariance by
   Monte Carlo over the `d_h` channels — the obvious reading of "empirical Cov" — has a noise floor
   of 1/√d_h ≈ 0.125 per pair, four times the criterion's own threshold, and no dependence on `M`
   whatsoever: measured that way the error is 0.0092 / 0.0096 / 0.0095 across M = 1024 / 2048 / 4096,
   i.e. the M-trend the criterion is about is invisible under the sampling noise of the estimator.
   `covariance` computes the same quantity exactly over the amplitude draw, which is where the
   approximation error actually lives. Both numbers are in the report (G1.1c carries the MC
   cross-check: 0.140 against the predicted 0.125).
7. **The plane geometry for G1.2 lives in `tests/fixtures/planes.py`**, not in the package: T07 owns
   `infer/planes.py`, and building it early would be skipping ahead. It is deliberately canonical —
   one `(u, v) → (x, y, z)` function — which is what SPEC_QUESTIONS **B1** asked for, and with it
   the spec's bitwise requirement is met outright rather than relaxed: the two float64
   reconstructions agree to 2.8e-14 µm and round to identical float32 coordinates, so the field
   values are bit-identical.
8. **G1.1's monotone-in-M claim is averaged over 8 realisations** (SPEC_QUESTIONS **B2**), and the
   reported number is the mean of the per-realisation errors, not the error of an averaged field —
   averaging fields would have made the criterion easier rather than more reliable.
9. **Two diagnostics were added to the report** (D1: the same sweep at a 3000 µm field of view;
   D2: Moran's I of the field itself). They are labelled "not gate criteria" and carry no
   thresholds; they exist because a failed gate is worth diagnosing.
10. `tests/test_schema.py` was reformatted by the pinned `ruff format` (0.4.4) again — the same
    formatting-only drift T02 recorded as its deviation 9, reintroduced by commit `94dff64`.

**Flagged for later.**

* **T09 must not bisect blindly.** See SPEC_QUESTIONS A7: cap the bracket at a fraction of the
  in-plane extent and handle "target Moran's I unreachable" explicitly. `Config.bisection_grid_size`
  (12) already exists for the fallback; the fallback now has a defined job.
* **`ell_z` is extrapolated on this fixture.** The along-z variogram reaches only 59 % of its fitted
  sill at the largest available lag (400 µm), so the fitted 355 µm against a ground truth of 200 µm
  is a low-confidence number and `fit_lengthscale_from_sections` warns
  (`LengthscaleFitWarning`, `Config.variogram_min_saturation = 0.75`). Real stacks are deeper; a
  9-section 400 µm fixture cannot see a 200 µm correlation decay away. T09's `ell_z` calibration
  should treat the fitted value as a starting point, not a measurement.
* **Per-channel-group `ell`** (SPEC_QUESTIONS A2) is *not* implemented: the spec's interface takes
  one `ell` and nothing in `specs/` consumes a grouped one. The door stays open cheaply —
  `with_lengthscale` builds a rescaled field from the same draws, so a grouped field is a
  channel-wise concatenation of those, no redraw and no new state.
