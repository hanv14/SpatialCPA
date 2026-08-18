# T00 — Project scaffolding

Part of [PROGRESS.md](../PROGRESS.md).

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
