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

## 2026-08-20 — T10 pilot decisions

**Coincident coordinates: scope T01's check, do not remove it.** bench3 flattens each multi-plane
slab to its centre z, so two cells at the same `(x, y)` in different planes are exactly coincident —
143 of 28 978 cells (0.49 %) on the tier-1 STARmap build, in every section, and `flattened_z` is
true for all 18 datasets. `Volume.flattened_sections` permits exact ties, `Volume.n_coincident_coords`
records the count, `validate_volume` warns once with it, and the flag propagates through
`split_holdout` and `loso_folds`. An **unflattened** volume keeps the hard check. The flag is read
from the data, never inferred — an inferred exemption would silence the check where there is a real
problem. Rejected: wrapper-side de-duplication (changes the cells v25 trains on relative to every
other method, breaking bench3's shared-input guarantee) and coordinate jitter (silent data
modification).

**C1 is measured per section, not pooled.** `flanking_copy` — a model-free probe — scores
`section_2` at 0.7008 against 0.7765 / 0.7868, a **0.086 positional swing** against SpatialZ's entire
**0.061** tier-1 lead over v20. A pooled localization number is therefore dominated by how a method
handles one held-out position and cannot settle the criterion. C1 reports each held-out section
against both measured referents (`oracle` ceiling, `flanking_copy` floor), with the pooled median
beside them as a summary.

**E1 is blocked, not descoped.** `deep_starmap` is absent from the repository. Case folding is
measured as mandatory and sufficient on the testable symbols (0/6 exact, 6/6 folded). What unblocks
it: the 1017 panel symbols alone answer the coverage question — the long pole, because a shortfall
needs `mygene` network access that is 403'd here — and a path to the source volume unblocks the
rest.

**Owed fix, found by the pilot:** `specs/09`'s selector must clamp `Config.expr_pca_dim` to the
panel width. STARmap has 28 genes against a default of 32, so `validate_config_against_volume`
refuses the fit and the protocol dataset cannot be fitted at the shipped default.

**`build_gene_meta` merges by default; `--overwrite` is explicit.** Second occurrence of the same
data loss: building one panel replaced the whole table, so a `deep_starmap` build destroyed the
STARmap and Zhuang rows including tier-1's. But replace-by-default was **itself** a fix (B19a) —
the original merge reused cached rows for requested symbols, so a corrected `--species` re-run
issued no queries and reported success. The new behaviour avoids all three failures rather than
trading one for another: **unrequested rows are kept** (fixes the loss), **every requested symbol is
re-queried regardless of cache** (keeps B19a fixed), and kept rows are still species-checked (stops
organism mixing). The cache is a fallback only for a symbol the lookup could not reach, which is
what lets an offline re-run keep good rows. `--overwrite` says "this table is exactly this panel",
and the builder prints requested count beside on-disk row count — the pair whose *difference* is
the diagnostic, since "1017 rows for 1017 requested" reads as success unless the prior count is in
view.

