# PROGRESS — SpatialCPA-v25-Gen

Status of every task in `specs/`. Update the row when a task lands, then append an entry under
"Log" with what was built, the test numbers, and any deviation from the spec and why.

Status values: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`.

| # | Task | Spec | Key deliverables | Gate | Status |
|---|---|---|---|---|---|
| T00 | Project scaffolding | — | `CLAUDE.md`, `PROGRESS.md`, `pyproject.toml`, `Makefile`, ruff/mypy/pytest config, `SPEC_QUESTIONS.md` | — | DONE |
| T01 | Config and data contracts | `specs/01_TASK_config_and_data.md` | `config.py`, `data/schema.py`, `data/loaders.py`, synthetic fixture | — | DONE |
| T02 | Text-grounded embeddings | `specs/02_TASK_text_embeddings.md` | `data/text.py`, `model/embeddings.py`, MedCPT cache, distillation head | — | TODO |
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
| Text/co-expression Spearman (synthetic, then real) | T02 | — |
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
