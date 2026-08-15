# PROGRESS — SpatialCPA-v25-Gen

Status of every task in `specs/`. Update the row when a task lands, then append an entry under
"Log" with what was built, the test numbers, and any deviation from the spec and why.

Status values: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`.

| # | Task | Spec | Key deliverables | Gate | Status |
|---|---|---|---|---|---|
| T00 | Project scaffolding | — | `CLAUDE.md`, `PROGRESS.md`, `pyproject.toml`, `Makefile`, ruff/mypy/pytest config, `SPEC_QUESTIONS.md` | — | DONE |
| T01 | Config and data contracts | `specs/01_TASK_config_and_data.md` | `config.py`, `data/schema.py`, `data/loaders.py`, synthetic fixture | — | DONE |
| T02 | Text-grounded embeddings | `specs/02_TASK_text_embeddings.md` | `data/text.py`, `model/embeddings.py`, MedCPT cache, distillation head | — | DONE |
| T03 | 3D GRF noise field | `specs/03_TASK_noise_field_GATE1.md` | `model/noise.py`, `scripts/gate1_report.py`, `reports/gate1.md` | **GATE 1** | TODO |
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
| GATE 1 (T03) | GRF prior halves median Moran's I error vs i.i.d.; per-gene r > 0.7; `I_gen` monotone in `ell` | not reached | `reports/gate1.md` |
| GATE 2 (T04) | oblique R² ≥ 0.90 × axis-aligned R² | not reached | `reports/gate2.md` |

## Numbers the paper needs (fill in as tasks land)

| Quantity | Source task | Value |
|---|---|---|
| Text/co-expression Spearman (synthetic, then real) | T02 | synthetic: **+0.0055** (≈ 0, as expected — arbitrary gene names); real: pending a real panel + `resources/gene_meta.parquet` |
| GRF vs i.i.d. Moran's I error ratio | T03 | — |
| Fitted `ell = (ℓx, ℓy, ℓz)` | T03 / T09 | — |
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
