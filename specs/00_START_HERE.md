# SpatialCPA-v25-Gen (spatialcpav25_gen) — START HERE

You are implementing **SpatialCPA-v25-Gen**, a continuous 3D transcriptomic field for generating virtual
tissue sections. Read this file fully before writing any code. Then work through the task files
**in numerical order**. Do not skip ahead — later tasks depend on earlier ones and there are two
hard gates that stop the project if they fail.

## What the system does

Given a spatial-transcriptomics volume (a stack of 2D sections, each with cell coordinates, cell
type labels, and gene expression), learn a continuous field over `(x, y, z) × gene` such that a
tissue section at **any depth, any orientation, any thickness** can be generated as a query against
the field. Evaluation: hold out sections, regenerate them, compare to truth.

## Repository layout to create

```
spatialcpav25_gen/
  __init__.py
  config.py            # T01 — dataclass config, single source of truth
  data/
    schema.py          # T01 — AnnData contracts, validation
    loaders.py         # T01 — volume loading, section splitting
    text.py            # T02 — MedCPT descriptors + cache
  model/
    noise.py           # T03 — 3D Gaussian random field prior      [GATE 1]
    field.py           # T04 — triplane anatomical field           [GATE 2]
    retrieval.py       # T04 — retrieval cross-attention
    layout.py          # T05 — intensity field + point process
    expression.py      # T06 — flow matching + ZINB decoder
    spatialcpav25_gen.py         # T06 — top-level module assembling the above
  losses/
    reconstruction.py  # T06
    sefl.py            # T07 — cross-plane / thickness / program consistency
    metric_aware.py    # T08 — differentiable Moran, Geary, profiles, Sinkhorn
  infer/
    generate.py        # T09 — section generation
    calibrate.py       # T09 — leakage-free length-scale + ZINB calibration
    planes.py          # T09 — plane geometry, oblique + curved surfaces
  eval/
    metrics.py         # T10 — the six target metrics (+ unoptimised ones)
    baselines.py       # T10 — SpatialZ and simple baselines
    benchmark.py       # T10 — LOSO harness, statistics
  cli.py               # T10 — entrypoints
tests/                 # mirrors the above; every task adds tests
specs/                 # these files
```

Also create a `CLAUDE.md` at repo root summarising §"Conventions" below so it is always in context.

## Conventions (non-negotiable)

1. **Config is a single frozen dataclass** in `config.py`. No magic numbers anywhere else in the
   codebase. If you need a constant, add a field with a documented default.
2. **Every tensor-returning function documents its shape** in the docstring as
   `(B, N, D)`-style, and asserts it at runtime under `if cfg.debug_shapes:`.
3. **Determinism.** Every stochastic function takes an explicit `seed: int` or
   `generator: torch.Generator`. Never call global `torch.rand`/`np.random` directly. A test
   asserts that two runs with the same seed are bitwise identical.
4. **Coordinates are always physical units** (µm), never pixel indices, and always
   `float32` arrays of shape `(N, 3)` ordered `(x, y, z)`.
5. **Expression is always raw counts** (or native intensity) in the model's output path. Any
   log/normalisation is an *input-side* transform only. Do not let normalised values leak into
   the decoder target.
6. **No silent fallbacks.** If a required field is missing, raise with a message naming the field
   and the AnnData key it was expected at. Fallbacks that hide bugs have already cost this project
   one version.
7. **Tests are runnable without a GPU and without real data.** Every module ships a synthetic
   fixture. `pytest tests/ -m "not slow"` must pass on CPU in under 3 minutes.
8. Type hints everywhere. `ruff` + `mypy --strict` on `spatialcpav25_gen/` must pass.

## Build order and gates

| Task | Module | Gate |
|---|---|---|
| T01 | Config + data contracts | — |
| T02 | Text embeddings (MedCPT) | — |
| **T03** | **3D noise field** | **GATE 1 — stop if failed** |
| **T04** | **Anatomical field + retrieval** | **GATE 2 — stop if failed** |
| T05 | Layout head | — |
| T06 | Expression head + ZINB decoder | — |
| T07 | SEFL consistency losses | — |
| T08 | Metric-aware LOSO losses | — |
| T09 | Inference + calibration | — |
| T10 | Benchmark + baselines | — |

### GATE 1 (end of T03)
On the synthetic fixture, a spatially-correlated prior must reproduce target spatial
autocorrelation within tolerance, and an i.i.d. prior must visibly fail the same test. **If the
correlated prior does not produce a clear Moran's I effect, stop and report. Do not build T04+.**
The entire method rests on this mechanism.

### GATE 2 (end of T04)
Reconstruction quality on oblique planes must reach **≥90%** of the quality on axis-aligned planes,
measured on held-in sections. If not, the backbone is insufficiently rotation-equivariant; stop and
report before building the heads on top of it.

## How to work

- One task file = one PR-sized unit. Implement, write its tests, run them, then move on.
- At the end of each task, append a short entry to `PROGRESS.md`: what was built, what the test
  numbers were, anything that deviated from the spec and why.
- **When the spec is wrong or underspecified, say so and propose a fix rather than guessing.** The
  specs were written before the code; some details will be wrong. Flag them.
- Prefer boring, readable implementations. This is research code that will be read by reviewers.

## Naming

The Python package is **`spatialcpav25_gen`**; the method is **SpatialCPA-v25-Gen**; the CLI is
`spatialcpav25-gen`. The two design documents in `design/` were written under the working name
"v23 + SEFL" and their filenames still say `v23` — **they describe this system.** Do not treat
`v23`, `v25`, `CTF-Flow`, and `SpatialCPA-v25-Gen` as different things. `v20` *is* a different
thing: it is the previous released version, kept as a baseline and as the no-regression fallback.

## What "v25 + SEFL" means

The system is **v25** (continuous transcriptomic field, open-vocabulary genes, flow-matching
expression head) **plus SEFL** (Sectioning-Equivariant Field Learning — in-silico sectioning used as
a self-supervised training signal). SEFL is not an optional add-on: it changes the noise prior from
a per-section 2D construction to a **continuous 3D random field** (T03), adds rotation equivariance
requirements to the backbone (T04), and contributes three consistency losses (T07). The task files
already assume the SEFL variant throughout.

`specs/11_COVERAGE_MATRIX.md` maps every component of both design documents to the task that
implements it. **Read it before starting, and check it again before declaring a task done.** If you
find something in a design doc that is not in the matrix, that is an omission in the specs — flag it
rather than silently skipping it.

## Reference material in this repo

- `specs/` — these task files.
- `design/v23_design.md`, `design/v23_sectioning_equivariance.md` — the two design documents these
  specs implement. Consult them for *why*; the task files are the *what*.
- `reference/learn_spatialcpav20.py` — the previous version (v20). Useful for: the benchmark
  protocol, the metric implementations (which have two bugs, see T10), the gap-aware dropout idea,
  and the ZINB-free cross-mix. Do **not** copy its architecture; v25 is a different design.
- `reference/SpatialZ.py`, `reference/Synthesize.py` — the competing method. Needed as a baseline
  (T10). Read it to understand what it does; do not import its approach into the model.
