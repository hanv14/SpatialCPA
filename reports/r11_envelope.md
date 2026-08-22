# The envelope is a sampled maximum, and that is the starved sampler's real cause

`sample_layout` sets `envelope = max(lam) * slack` over **one** draw of `Config.layout_n_mc` points. Below: 24 independent draws of that same size, the spread of the maximum they find, and the acceptance rate each would imply against a reference mean taken from a 16x larger sample (a mean converges; a maximum does not).

| link | section | mean lam | max: min / median / max | spread | implied acceptance |
|---|---|---|---|---|---|
| `exp` | section_2 | 0.0001129 | 0.0006806 / 0.03779 / 0.5809 | **853.5x** | 0.018% .. 15.073% |
| `exp` | section_4 | 0.001894 | 0.1177 / 3.412 / 21.89 | **185.9x** | 0.008% .. 1.463% |
| `exp` | section_6 | 8.618e-05 | 0.0003815 / 0.003541 / 0.05331 | **139.7x** | 0.147% .. 20.535% |

## What the spread means: the layout is biased, not merely starved

`layout.py:995` accepts a proposal when `u_intensity < lam_total / envelope`. When
`lam_total > envelope` that ratio exceeds 1 and the comparison is **always true**: every proposal
in such a region is accepted unconditionally. Rejection sampling is only valid when
`envelope >= sup lambda`; when the single MC draw under-bounds the true supremum, the sampler
silently draws from `min(lambda, envelope)` rather than from `lambda`.

At the spreads measured above — **853x, 186x, 140x** across draws of the same size — under-bounding
is not an edge case, it is the common case. So the two field-based `layout_mode`s produce a point
pattern that is **flattened wherever the intensity exceeds a randomly chosen ceiling**, and nothing
warns about it: `ProposalBudgetWarning` fires on the *starved* symptom, which is the opposite
failure (an envelope that came out too high).

This is a third defect, distinct from the two in `reports/r11_budget.md`:

| # | defect | evidence | owner |
|---|---|---|---|
| 1 | the intensity integral's scale is wrong, and link-coupled | `n_expected` / GT = 1.51x .. 16.73x; `exp` worsens `section_2` 2.68x -> 16.73x | T05 intensity head |
| 2 | the sampler is starved | 2.4% / 1.0% placed on `section_4`; **not** the interaction — A4 recovers 6 and 11 cells of ~6000 and ~8700 | T05 `_propose_points` |
| 3 | **the sampler is biased** | envelope is a sampled max with 140x-853x spread; acceptance is not clamped, so `lambda` is truncated at a random ceiling | T05 `sample_layout` |

Defect 3 is the one that changes what a `layout_mode` comparison *means*: `field` and `hybrid` are
not a faithful draw from the learned intensity, so their deficit against `resample` is not purely a
statement about the intensity head's quality.

**The fix is not a larger `layout_n_mc`.** A sampled maximum converges slowly and from below, so
raising 4096 buys a constant factor against a quantity with an 853x spread. The standard remedies
are to bound the envelope analytically from the head's own parameterisation, or to replace global
rejection with a partitioned or adaptive envelope, or to clamp the acceptance ratio and *report*
the truncated mass rather than absorb it silently. Choosing between them is T05's call, not T10's;
what T10 owes is this measurement and the note that the current warning names neither defect 2's
cause nor defect 3 at all.

