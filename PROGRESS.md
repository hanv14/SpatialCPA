# PROGRESS — SpatialCPA-v25-Gen

Status of every task in `specs/`. Update the row when a task lands, then append an entry under
"Log" with what was built, the test numbers, and any deviation from the spec and why.

Status values: `TODO` | `IN PROGRESS` | `BLOCKED` | `DONE`.

| # | Task | Spec | Key deliverables | Gate | Status |
|---|---|---|---|---|---|
| T00 | Project scaffolding | — | `CLAUDE.md`, `PROGRESS.md`, `pyproject.toml`, `Makefile`, ruff/mypy/pytest config, `SPEC_QUESTIONS.md` | — | DONE |
| T01 | Config and data contracts | `specs/01_TASK_config_and_data.md` | `config.py`, `data/schema.py`, `data/loaders.py`, synthetic fixture | — | DONE |
| T02 | Text-grounded embeddings | `specs/02_TASK_text_embeddings.md` | `data/text.py`, `model/embeddings.py`, MedCPT cache, distillation head | — | DONE |
| T03 | 3D GRF noise field | `specs/03_TASK_noise_field_GATE1.md` | `model/noise.py`, `scripts/gate1_report.py`, `reports/gate1.md` | **GATE 1** | DONE — **GATE 1 passes** on the 3000 µm gate fixture |
| T04 | Anatomical field + retrieval | `specs/04_TASK_field_and_retrieval_GATE2.md` | `model/field.py`, `model/retrieval.py`, `scripts/gate2_report.py`, `reports/gate2.md` | **GATE 2** | DONE — **GATE 2 passes**, depth-matched oblique parity **0.955** (edge-excluded check **0.979**); amended 2026-08-16 with the per-query z window (SPEC_QUESTIONS C1c), gate numbers unchanged |
| T05 | Layout head | `specs/05_TASK_layout_head.md` | `model/layout.py`, `losses/reconstruction.py` (layout NLL), `infer/planes.py` (minimal `Plane`), intensity + Strauss sampler + Potts marks | — | DONE — all eight acceptance tests pass, both negative controls fail as they must |
| T06 | Expression head + ZINB decoder | `specs/06_TASK_expression_head.md` | `model/expression.py`, `model/spatialcpav25_gen.py` (`CTFFlow` + trainer), `losses/reconstruction.py`, `eval/baselines.py` | — | DONE — with three recorded failures: the covariance criterion is **unsatisfiable as stated** (below the ceiling, B16) and the model half of the amendment does **not** hold out of sample; zero-shot decoding is **r = −0.368** (B18); T05's intensity overfit is answered at trainer level but not abolished (R4) |
| T07 | SEFL consistency losses | `specs/07_TASK_sefl_losses.md` | `losses/sefl.py`, `infer/planes.py`, EMA teacher, collapse alarm | — | DONE — **SEFL ships opt-in (all three weights 0)**, with one result that reframes the loss it belongs to: intersection consistency is **exact by construction** in v25 (bitwise, untrained), so `L_cross` has only the augmentation pose left to constrain and minimising that flattens the field (generated per-gene variance **0.065** against **0.711** with SEFL off). `w_cross` ships at **0**, the failure is a strict xfail, and the decision is R6 / SPEC_QUESTIONS C19. `L_thick` and `L_prog` land as specified but are **also off**, because at their spec weights they break three of T06's acceptance criteria (R7); A7 at T10 decides whether SEFL is used at all |
| T08 | Metric-aware LOSO losses | `specs/08_TASK_metric_aware_losses.md` | `losses/metric_aware.py`, `train/loso.py`, `scripts/t08_metric_report.py` | — | DONE — with **both** headline criteria red and the decisions they force taken: the terms **ship at 0** (opt-in, A2 becomes an addition experiment) because at T06's 1200-step budget they cost on every metric they are made of, and open risk **R4's covariance criterion is not met on either holdout** (11.02 vs a 7.73 baseline at `alternating`, 13.39 vs 11.38 at `consecutive-3`), so the paper's covariance claim is **downgraded to a mechanism claim** in `specs/10` §2. The finding that qualifies both: at **2400** steps the ordering reverses on four of six statistics — the terms are slower, not worse, and 1200 steps is R4's own symptom |
| T09 | Inference + calibration | `specs/09_TASK_inference_and_calibration.md` | `infer/generate.py`, `infer/calibrate.py`, `train/select.py` | — | TODO |
| T10 | Benchmark + baselines | `specs/10_TASK_benchmark_and_baselines.md` | `eval/metrics.py`, `eval/baselines.py`, `eval/benchmark.py`, `cli.py` | — | TODO |

`specs/` defines ten implementation tasks (T01–T10); T00 is this scaffolding pass, listed so the
table covers everything that has been done to the repository.

## Gate status

| Gate | Criterion | Status | Report |
|---|---|---|---|
| GATE 1 (T03) | GRF prior halves median Moran's I error vs i.i.d.; per-gene r > 0.7; `I_gen` monotone in `ell` over the calibration bracket; `I_gen(ell)` unimodal with its maximiser ≥ the fitted `ell` | **PASSED** — error ratio **0.130** (< 0.5), r **0.917** (> 0.7), smallest step over the bracket **+0.028** (> 0), fitted vs best-matching `ell` **8 %** (< 25 %), unimodality violation **0.000** (< 0.0069), maximiser **2.52×** the fitted `ell` (≥ 1×) | `reports/gate1.md` |
| GATE 2 (T04) | **depth-matched** oblique parity ≥ 0.90 (both arms), plus the interior-only check; held-out z ≥ 0.8 × neighbouring z; `w_z = 0` costs R² at fractional depths 0.2/0.8 but not 0.5; attention entropy > 0.5 log K; augmentation complete (G2.1h) and the draw-noise floor measured (G2.1i) | **PASSED** — G2.1a **0.9547** (≥ 0.90), G2.1b edge-excluded **0.9795**, z-interpolation **1.097** (≥ 0.80), `w_z` ablation **+0.030** at 0.2 / **+0.049** at 0.8 vs **+0.003** at 0.5 (< 0.01), entropy **3.422** nats (> 1.733), all four rotation channels wired, draw σ **0.0168**. Criterion amended in `specs/04` after the escalation came back null — see SPEC_QUESTIONS C16 | `reports/gate2.md` |

## Open risks carried forward

Full table and the R1 / R3 / R4 narratives: **[progress/risks.md](progress/risks.md)**.

| # | State |
|---|---|
| R1 | `ell_z` under-determined by a short stack — **decided** at T07 (calibrate at inference); implementation owed at T09 |
| R2 | Retrieval attention near-uniform — **closed** at T06 (0.987 → 0.856 × log K) |
| R3 | The stack's ends reconstruct far worse than its interior — **open**, owed to T09/T10 |
| R4 | Likelihood/fidelity divergence — **open**. T08 built and measured the cure; it is not closed, and the enable decision is T09's `select_config` + T10's A2 |
| R5 | C14 blocks the open-vocabulary claim — **open**, needs one networked run |
| R6 | `L_cross` vacuous in v25 and harmful when trained — **open**, `w_cross` ships at 0 |
| R7 | SEFL's net contribution unverified — **open**, decided by T10's A7 |

## Numbers the paper needs

**[progress/numbers.md](progress/numbers.md)** — the per-task table, filled in as tasks land.

## Log

One file per task. Each records what was built, the measured numbers, and every
deviation from the spec with its reason.

| Task | Entry |
|---|---|
| T00 | [progress/t00_scaffolding.md](progress/t00_scaffolding.md) |
| T01 | [progress/t01_config_and_data.md](progress/t01_config_and_data.md) |
| T02 | [progress/t02_text_embeddings.md](progress/t02_text_embeddings.md) — includes the two `build_gene_meta` repairs and the B20 orthologue fallback |
| T03 | [progress/t03_noise_field.md](progress/t03_noise_field.md) — GATE 1, failed then re-run and passed |
| T04 | [progress/t04_field_and_retrieval.md](progress/t04_field_and_retrieval.md) — GATE 2, the escalation, and the per-query z window |
| T05 | [progress/t05_layout_head.md](progress/t05_layout_head.md) |
| T06 | [progress/t06_expression_head.md](progress/t06_expression_head.md) — includes the covariance demotion and C14's escalation |
| T07 | [progress/t07_sefl_losses.md](progress/t07_sefl_losses.md) |
| T08 | [progress/t08_metric_aware.md](progress/t08_metric_aware.md) |
| — | [progress/decisions.md](progress/decisions.md) — spec decisions settled with no code change |

**Appending to this log:** add the entry to the task's own file under `progress/`,
update the row in the status table above, and leave this index short. The file was
split at T08 because a 2131-line PROGRESS.md could no longer be rewritten in one pass,
which is how a task's findings came to be missing from it.
