# R11 re-measurement — protocol for the three `layout_mode` arms on the grid sampler

**Status: the protocol and the scripts. The numbers are not in this repository yet** — the fit and
the three arms run on the campaign machine, and `scripts/t10_layout_modes_table.py` writes them to
`reports/r11_starmap_layout_modes.md`.

## Why it has to be re-measured at all

Every layout number this project has recorded — T05's fixture acceptance numbers, T09's 18-cell
gate, T10's STARmap pilot (`reports/pilot.md` §13, and R11's one-page table in
`progress/risks.md`) — was produced by the rejection sampler, and `reports/r11_envelope.md`
measured that sampler as **biased, not merely starved**: its envelope is a sampled maximum with a
140-853x spread over one draw's size, acceptance is unclamped, so the realised draw came from
`min(lambda, envelope)` rather than from `lambda`. `field` and `hybrid` were therefore never a
faithful draw from the learned intensity, and their deficit against `resample` was not purely a
statement about the intensity head. `reports/r11_fix_options.md`'s option D — grid-multinomial
sampling, no envelope and no acceptance ratio — is now `Config.layout_sampler`'s default, and that
report's own accounting lists this re-measure as still owed.

## The protocol, matching the pilot's §13 table

Deliberately identical to `reports/pilot.md` §13 so the columns line up with the biased-sampler
numbers they replace:

| | |
|---|---|
| dataset | STARmap tier 1, bench3's paper protocol (`paper_2_4_6`), sections 2/4/6 held out |
| instrument | bench3's `evaluate_paper.py`, SHA-pinned, unmodified |
| weights | **one fit**, `decoder_mu_link=exp`, 2400 steps, `text_emb_mode=lookup`, `expr_pca_dim=16`, `ell=(116.3, 116.3, 132.0)`, seed 1 |
| arms | `layout_mode` ∈ {`field`, `hybrid`, `resample`} — a generation-time gate, so no arm needs a fit |
| density | `celltype_localization` at **ground-truth-matched** density; `cell_count_ratio` from the raw pass |
| estimator | median over the three held-out sections (`specs/10` §4.6), **and** every section reported beside it |
| referents | `flanking_copy` (copy floor) and `oracle` (ceiling), bench3's own model-free probes, re-scored on the same machine |

Two things are added to §13's protocol, and both are there to keep the comparison honest:

1. **Per-section, not only the median.** R11 records `cell_count_ratio` spanning 500x across these
   three sections with the median picking the only sane one, and `reports/pilot.md` §3 measured the
   model-free copy floor scoring `section_2` worst on 6 of 6 metrics because the stack's first
   section has flanking evidence on one side only. A pooled layout number that does not carry its
   sections misdescribes the layout.
2. **A rejection-sampler control arm.** The pilot's checkpoint does not exist any more (`*.pt` is
   gitignored; `reports/durability.md`: preserve the measurement, regenerate the model), so the
   grid numbers come from a *refit*. Bitwise determinism (Convention 3) holds for a given machine
   and thread count, not across machines, so a bare grid-vs-pilot comparison confounds **the
   sampler changed** with **the weights are not the pilot's**. Re-running `field` and `hybrid` on
   `--layout-sampler rejection` from the *same* new weights separates them: that pair against the
   pilot's is the refit's effect, and grid-against-rejection on one set of weights is the
   sampler's.

## What the run does not settle

* **One seed.** `reports/envelope_synthetic.md` puts the across-seed envelope at **0.0335**, and
  `specs/10` §3's repeated-seed rule wants `claim_min_seeds` = 3 before a claim rests on any of
  this. A difference below the envelope is not a difference.
* **Still ablation A3.** `text_emb_mode` is `lookup`, not the shipped `medcpt`, and T09's
  per-dataset selection and calibration have still never run on this dataset. These are the
  pilot's standing caveats, unchanged.
* **The grid sampler fixes positions, not the count.** `n_target ~ Poisson(n_expected)` is drawn
  from the slab integral *before* either sampler is reached (`layout.py`), so defect 1 — the
  intensity integral's scale, 1.51x-16.73x against ground truth — is untouched. If
  `cell_count_ratio` stays wrong for `field` and `hybrid`, that is the expected result, not a
  failure of the run.
* **`hybrid` still cannot differ from `field` in count.** The count draw precedes `hybrid`'s
  sliced-Wasserstein polish, so with one seed the two arms' counts remain bit-identical.

## Running it

`scripts/t10_chain_diagnostic.py` fits and saves the weights; `scripts/t10_rescore_saved.py`
scores **one arm per process** (they are independent, so the arms run concurrently, one thread
each); `scripts/t10_layout_modes_table.py` collects the arm JSONs, re-scores the two probes and
writes the table. All three take `--bench3` and resolve the built dataset and the leakage-guarded
input from it, so nothing depends on this checkout's own directory layout. `--preflight` resolves
and checks every path in seconds, which is the thing to run before an hour of fitting.
