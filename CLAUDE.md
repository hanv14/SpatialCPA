# CLAUDE.md — SpatialCPA-v25-Gen

Project-wide rules. Read this before touching anything. It is a condensation of
`specs/00_START_HERE.md`; that file is the authority, this one is the always-in-context copy.

## Naming (get this right or the project argues with itself)

| Thing | Name |
|---|---|
| Python package | `spatialcpav25_gen` |
| Method / paper name | **SpatialCPA-v25-Gen** |
| CLI entrypoint | `spatialcpav25-gen` |
| Top-level model class | `CTFFlow` (per T06) |

- The design documents in `design/` are named `v23_*.md` and use the working name "v23 + SEFL".
  **They describe this system.** `v23`, `v25`, `CTF-Flow`, `SpatialCPA-v25-Gen` are all the same
  thing. Do not treat them as different systems.
- **`v20` is a different thing**: the previous released version, kept in `reference/` as a baseline
  and as the no-regression fallback (`layout_mode=resample` + `expr_mode=cross-mix`).
- The system is v25 (continuous field, open-vocabulary genes, flow-matching expression head) **plus
  SEFL** (Sectioning-Equivariant Field Learning). SEFL is not optional: it makes the noise prior a
  continuous 3D field (T03), imposes rotation-equivariance on the backbone (T04), and adds three
  consistency losses (T07).

## Conventions (non-negotiable)

1. **Config is a single frozen dataclass** in `spatialcpav25_gen/config.py`. No magic numbers
   anywhere else. Need a constant? Add a documented field with a default. This includes constants
   the task files write inline (kNN `k`, grid sizes, kernel widths, iteration caps) — they become
   `Config` fields.
2. **Every tensor-returning function documents its shape** in the docstring as `(B, N, D)`-style,
   and asserts it at runtime under `if cfg.debug_shapes:`.
3. **Determinism.** Every stochastic function takes an explicit `seed: int` or
   `generator: torch.Generator`. Never call global `torch.rand` / `np.random`. Two runs with the
   same seed must be bitwise identical, and a test asserts it.
4. **Coordinates are physical units (µm), never pixel indices**, `float32`, shape `(N, 3)`, ordered
   `(x, y, z)`. Exception, documented in T01: `Section.coords` is stored as `(N, 2)` in-plane with
   the section's `z` held separately; `to_xyz(section)` is the only sanctioned way to get `(N, 3)`.
5. **Expression is always raw counts** (or native intensity) on the model's output path. Any
   log/normalisation is an *input-side* transform only. Normalised values must never reach the
   decoder target.
6. **No silent fallbacks.** A missing required field raises, and the message names the field *and*
   the AnnData key it was expected at. Fallbacks that hide bugs have already cost this project one
   version.
7. **Tests run without a GPU, without real data, and without network.** Every module ships a
   synthetic fixture. `pytest tests/ -m "not slow"` must pass on CPU in under 3 minutes; anything
   that trains a loop gets `@pytest.mark.slow`.
8. **Type hints everywhere.** `make lint` (ruff) and `make typecheck` (`mypy --strict` on
   `spatialcpav25_gen/`) must pass before a task is done.

## Build order and the two hard gates

| Task | Module | Gate |
|---|---|---|
| T01 | Config + data contracts | — |
| T02 | Text embeddings (MedCPT) | — |
| **T03** | **3D GRF noise field** | **GATE 1 — stop if failed** |
| **T04** | **Anatomical field + retrieval** | **GATE 2 — stop if failed** |
| T05 | Layout head | — |
| T06 | Expression head + ZINB decoder | — |
| T07 | SEFL consistency losses | — |
| T08 | Metric-aware LOSO losses | — |
| T09 | Inference + calibration | — |
| T10 | Benchmark + baselines | — |

**GATE 1 (end of T03).** On the synthetic fixture, the GRF prior must halve the median Moran's I
error relative to an i.i.d. prior, per-gene `I_gen` vs `I_real` must correlate at r > 0.7, and
median `I_gen` must move monotonically as `ell` sweeps 0.25×–4× the fitted value. If the correlated
prior shows no clear Moran's I effect: **stop, write `reports/gate1.md`, report back.** Do not build
T04+. The whole method rests on this mechanism.

**GATE 2 (end of T04).** Reconstruction quality on oblique planes must reach **≥ 90%** of
axis-aligned quality (`min_angle R² ≥ 0.90 × R²(0°)`) on held-in sections. If it fails: raise
`n_plane_orientations` 4 → 8, verify rotation augmentation covers coords *and* planes *and*
retrieval *and* GRF queries, re-run — then **stop and report**. A steerable backbone is a design
change, not a tuning fix.

Both gates produce a report under `reports/` before anything downstream is written.

## How to work

- One task file = one PR-sized unit: implement, write its tests, run them, then move on. Do not
  skip ahead; later tasks depend on earlier ones.
- At the end of each task, update the row in the `PROGRESS.md` status table and append the entry to
  **that task's own file under `progress/`** (`progress/t08_metric_aware.md`, …): what was built, the
  test numbers, and anything that deviated from the spec and why. `PROGRESS.md` itself is a short
  index — status table, gate status, risk summary, links — and is to stay that way. It was split at
  T08 because a 2131-line log could no longer be rewritten in one pass, and a task's findings went
  missing as a result.
- Check `specs/11_COVERAGE_MATRIX.md` before starting a task and again before calling it done. A
  design-doc component missing from the matrix is an omission — flag it, don't silently skip it.
- **When the spec is wrong or underspecified, say so and propose a fix rather than guessing.**
  Open items live in `SPEC_QUESTIONS.md`; add to it rather than inventing an answer quietly.
- Prefer boring, readable implementations. Reviewers will read this code.
- Two things are built *to fail* and are reported as results: `loss_prog_WRONG` (T07, ablation A8)
  and the independent-donor sampler (T06/T10). They are not dead code; do not delete them.

## Leakage discipline

Held-out sections are never touched by training, calibration, or config selection. This is enforced
by type (`TrainingVolume` vs `HeldOutSections`), not by convention. Metric-aware losses (T08),
calibration (T09), and config selection (T09) all run on internal LOSO over *training* sections only.

## Layout

```
spatialcpav25_gen/    config.py, data/, model/, losses/, infer/, eval/, cli.py
tests/                mirrors the package; every task adds tests
specs/                task files 01–10 + coverage matrix (the *what*)
design/               v23_design.md, v23_sectioning_equivariance.md (the *why*)
reference/            v20 (baseline), SpatialZ + Synthesize (competing method) — read, do not copy
benchmark-pbya-v3/    existing benchmark harness; bench3/evaluate_paper.py holds the real `paper_*`
                      metric implementations used for published numbers
reports/              gate1.md, gate2.md, benchmark.md, config_selection_*.md
progress/             the task log, one file per task; PROGRESS.md is its index
```

## Commands

```
make install     # editable install with dev extras
make test        # pytest -m "not slow"   (must stay under 3 minutes on CPU)
make test-all    # including slow tests
make lint        # ruff check + format check
make typecheck   # mypy --strict spatialcpav25_gen
```

## Git

Development happens on `claude/spatialcpa-v25-setup-o4dmyi`. Push with
`git push -u origin claude/spatialcpa-v25-setup-o4dmyi`. Do not push to `main`. Do not open a PR
unless asked.
