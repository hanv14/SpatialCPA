# T10 — Benchmark and baselines

Task log. `PROGRESS.md` is the index; this file holds T10's entries.

---

## 2026-09-11 — the tier-1 comparator re-score, and the instrument pin becomes a test

### What arrived

`specs/10` step 5 — *"Comparators on tier 1: SpatialZ, FEAST, isoST, v20, plus `oracle` and
`flanking_copy` — **required**, because the existing numbers are not in this repo"* — **was run.**
`evaluate_all --force` into `benchmark-pbya-v3/results_rescored/` for
`starmap_visual_cortex/paper_2_4_6`, six methods, **0 failures**, on the pinned evaluator. The table
is inlined in the paper's §8.2 with `flanking_copy`, `oracle` and v25's own row beside it.

Three reports that said the run could not be done here are stale on that point and are marked, not
rewritten: `reports/pilot.md` §44 (annotated in place), `reports/spatialz_claim_struck.md` §1–2
(superseded by its own §6 amendment), and the paper's earlier §8.3 (rewritten as §8.4).

### What the table says

1. **On four of the five metrics with a floor, no method reaches it** — not SpatialZ, FEAST, isoST,
   v18, v20, v21 or v25. Best in column is −0.0025. Copying the flanking sections is at or above the
   state of the art on four of the five readable metrics. **This is §6.1 generalised past our own
   method and is the most valuable thing the run produced.**
2. **SpatialZ beats v25 on five of six.** Our single win is `umap_mixing`, the one metric with no
   floor and no ceiling — unreadable by this project's own rule, and not claimed.
3. **v20 sits +0.0001 from `flanking_copy`** on `celltype_localization` and −0.0025 on both
   autocorrelation metrics. R13's "`cross-mix` under `resample` *is* a copy" now holds on a second
   dataset, on the pinned instrument, to four decimals.

### The struck claim stays struck

"v20 beats SpatialZ 5 of 6" is now **arithmetically supported** by this run and **remains withdrawn**.
`spatialz_claim_struck.md` had two objections; the run resolves the evidence objection (§1–2) and
leaves the circularity objection (§3) untouched — and §3 was the load-bearing one. Recorded as a
dated §6 amendment to that file, with the sentence the situation needed: **getting a number does not
unstrike a claim.**

### Four cells reported as returned rather than smoothed

- `v18` and `v20` **identical in all six columns to four decimals** — undetermined whether they emit
  the same predictions or the same predictions were scored twice. **Open; worth resolving.**
- FEAST and isoST return **exactly `0.0000`** on `celltype_localization`: the not-scorable value, not
  a measurement. Read as blank, shown as returned (Convention 6).
- isoST's **0.9956** on `umap_mixing` — largest number in the table, from the method otherwise last
  or second-last in four of five other columns, on the one column with no probe. Three reasons to
  read nothing from that column, ours included.
- FEAST returns **0.7742** on both `morans_pearson` and `umap_mixing`. Noted, unexplained.

### `tests/test_instrument_pin.py` — a specified acceptance test that did not exist

`specs/10`'s acceptance list names `test_evaluate_paper_sha256_unchanged`. **It had never been
written.** The pin lived in prose in `specs/10` §0 and in a `sha256sum` command a person was expected
to run by hand — for the file every published number in this work was measured with.

Written now, four tests, no data / GPU / network:

| test | what it holds |
|---|---|
| `test_evaluate_paper_sha256_unchanged` | the committed evaluator hashes to the pin; the failure message names the file and says **not** to update the constant |
| `test_pinned_hash_agrees_with_the_spec_that_pins_it` | the constant in the test is the one `specs/10` §0 states — the duplication is deliberate and this keeps it honest |
| `test_line_count_matches_the_pin_so_a_mismatch_is_readable` | 764 lines, a second descriptor, so a failure says roughly *how* different |
| `test_the_hasher_rejects_a_changed_byte` | positive control, and it runs **without** the bench3 tree, so it still holds in the checkout where the other three skip |

Verified: the run's reported hash, `specs/10` §0's pin and the committed file all give
`7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992`, 764 lines.

**The general lesson, and it is §7.3's rule again.** An acceptance test listed in a spec and never
written is indistinguishable, from inside the repository, from one that is written and passing.
Nothing failed; the check simply was not there, and the pin held only because nobody had changed the
file. This is the same shape as R18 — a control that exists on paper and not where a consumer reads
it.

### Still open on T10

- `results_rescored/` is not committed; §8.2's 42 numbers are transcribed. The **instrument** is
  reproducible from this repository (hash-asserted); the **measurement** is not.
- `paper_umap_mixing` has **no probe on any run** — no floor, no ceiling, one of the protocol's own
  six columns unreadable.
- The floor/ceiling columns come from the probes tree rather than from the same `evaluate_all` call.
  Legitimate (the probes are model-free and arm-independent) but a join of two invocations.
