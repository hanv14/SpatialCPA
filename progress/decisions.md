# Spec decisions settled

Part of [PROGRESS.md](../PROGRESS.md).

### Spec decisions settled (2026-08-15) — no code changed

Nine open items in `SPEC_QUESTIONS.md` were decided and written into the task files they belong to.
Nothing here is implemented yet: each is due at its own task, and the point of recording them now is
that the task file says what to build before someone starts building something else.

| Item | Decision | Landed in |
|---|---|---|
| **C1** GATE 2's evaluation set | Pooled cells within `thickness/2` of the query plane, **plus** (a) each evaluated cell's own source section excluded from retrieval at every angle, (b) seeded subsampling to equal `n` across angles — reporting `n` is not enough. Contract stated in `reports/gate2.md`. | `specs/04`, matrix |
| **A3** T10 metric provenance | **Do not port.** Vendor or import `bench3/evaluate_paper.py` verbatim, pin `sha256 = 7362669…8992`, assert **bitwise** agreement. v20's two bugs become a footnote about v20's *internal* tuning signal. | `specs/10` §1, matrix |
| **A6** v20 Bernoulli cross-mix | Implement in T06 §4b beside the decoder; behaviour pinned by `test_cross_mix_matches_v20`. | `specs/06`, matrix |
| **B6** hard-core radius | `r0` at the **1st** percentile, 5th kept selectable, which one was used recorded in the report. | `specs/05`, matrix |
| **A2** per-module `ell` | **One global `ell`**; per-module Moran's agreement is a diagnostic only. Escalation, if poor at T09, is per-channel-group `ell` — decided explicitly, with the diagnostic as evidence. | `specs/09` §2, matrix |
| **C2** `KL(ZINB‖ZINB)` | **Skip the surrogate.** Match decoder parameters directly: L2 on `log mu`, `log theta`, `pi` logit, branch 2 detached. | `specs/07` §2, matrix |
| **C10** principal tissue axis | On `TrainingVolume`, not `Volume` — leakage-free by construction. T08 adds it. | `specs/08` §2, matrix |
| **D-table** ×5 | v14/v18 dropped explicitly with a reason; dataset requirement (≥ 1 non-brain, ≥ 1 non-transcriptomic) enforced by the harness; mean–variance (`log theta`) calibrated beside `pi`; E1 reports both zero-shot arms; cross-mix → T06. | `specs/06`, `09`, `10`, matrix |

Two `Config` fields are **named but deliberately not added yet** —
`retrieval_exclude_source_section` and `gate2_min_cells_per_angle` — because nothing reads them until
T04 and the floor's value should come from T04's own measurement of how many cells each angle's slab
holds. The spec names both as `Config` fields so Convention 1 still binds when they land.
