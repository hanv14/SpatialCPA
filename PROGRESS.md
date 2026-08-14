# PROGRESS — SpatialCPA-v25-Gen

Status of every task in `specs/`. Update the row when a task lands, then append an entry under
"Log" with what was built, the test numbers, and any deviation from the spec and why.

Status values: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`.

| # | Task | Spec | Key deliverables | Gate | Status |
|---|---|---|---|---|---|
| T00 | Project scaffolding | — | `CLAUDE.md`, `PROGRESS.md`, `pyproject.toml`, `Makefile`, ruff/mypy/pytest config, `SPEC_QUESTIONS.md` | — | DONE |
| T01 | Config and data contracts | `specs/01_TASK_config_and_data.md` | `config.py`, `data/schema.py`, `data/loaders.py`, synthetic fixture | — | TODO |
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
