# Costing the envelope fix, and what it decides

`reports/r11_envelope.md` established three defects in the field-based layout. Defect 3 — the
envelope is a sampled maximum, acceptance is unclamped, so the draw comes from `min(lambda,
envelope)` — is the one that decides whether `layout_mode` can be chosen on a correct sampler at
all. This costs the four available fixes. **Nothing is implemented; this is the decision input.**

## What any fix must preserve

`sample_layout` produces `uv` and everything downstream is independent of how: `xyz = plane.to_xyz(uv)`,
then `lam` at those positions, then `_categorical`, `potts_smooth`, `_build_layout`. So the blast
radius of replacing the position sampler is **one function**, `_propose_points`, plus whatever
`sample_layout` computes to feed it (`envelope`, `n_target`). The marks, the Potts smoothing, the
`Layout` contract and `hybrid`'s sliced-Wasserstein polish are all untouched by every option below.

## A. Analytic bound from the head's own parameterisation — **measured, and counterproductive**

`lambda_c = softplus(MLP_c([field_feat, fourier(xyz), region_emb]))`. An MLP has no closed-form
supremum, so the only analytic route is a Lipschitz bound: `lambda(x) <= softplus(pre(x0) + L *
||z(x) - z(x0)||)` with `L` the product of the trunk's spectral norms.

**Measured on `runs/pilot/model_exp_2400.pt`:** `||W||_2` = 4.3003, 6.5847, 2.6376 over the three
layers, so **L = 74.69**.

The Fourier features are bounded in `[-1, 1]` across 148 input dimensions, so the encoded diameter
over a plane is between about 2 and 24, and the pre-activation swing the bound must allow is
`74.69 * diam` = **+150 to +1790**. Where `softplus` is linear that is the bound on `lambda` itself,
against a measured true supremum of **21.9** on `section_4` — so the bound sits **7x to 82x above
the true sup**.

Acceptance is `mean / envelope`, so a 7-82x looser envelope is 7-82x worse acceptance, applied to a
section already achieving 0.12%. The result is a sampler that is *valid* and places essentially
nothing. **This option converts a bias bug into a universal starvation bug.** It is not expensive to
implement; it is simply the wrong answer, and the measurement above is why.

| | |
|---|---|
| implementation | small — one function computing `L` once per fit, plus an anchor |
| invalidates | nothing structural; every existing layout number would change |
| validated without a fit? | yes, on the saved checkpoint |
| **verdict** | **rejected on measurement** — 7-82x looser envelope, acceptance already at 0.12% |

## B. Partitioned / adaptive envelope

Grid the plane; take a local maximum per cell; sample a cell with probability proportional to
`local_max * cell_area`, then rejection-sample within it. Acceptance becomes governed by the
*local* dynamic range, which for a field whose global max/mean is 532 is dramatically smaller.

| | |
|---|---|
| implementation | moderate — a new `_propose_points_partitioned`, roughly 60-100 lines, plus two `Config` fields (grid resolution, per-cell probe count) |
| invalidates | every `field`/`hybrid` layout number ever measured, including T05's acceptance tests and T09's `layout_mode` gate. `resample` is untouched |
| validated without a fit? | **yes** — see "validation is cheap" below |
| caveat | the per-cell maximum is **still sampled**, so this improves the constant enormously but does not by itself make the envelope a bound. It needs C alongside it to be correct rather than merely better |

## C. Clamp the ratio and report the truncated mass

One line at `layout.py:995` to clamp, plus computing how much intensity mass sat above the envelope
and warning with that number.

| | |
|---|---|
| implementation | **smallest** — ~15 lines including the new warning and its message |
| invalidates | nothing. It changes no sampling behaviour that was not already happening; it only makes the truncation visible and quantified |
| validated without a fit? | yes, trivially |
| caveat | **this does not fix the bias.** It converts a silent defect into a reported one, which is what `CLAUDE.md`'s no-silent-fallbacks convention asks for, and it tells us per section how much of the intensity is being truncated — but the numbers stay wrong until B or D lands |

## D. Grid-multinomial sampling — not on the original list, and the cheapest correct option

Discretise `lambda` onto a grid over the slab, sample cell indices multinomially with probability
proportional to `lambda * cell_volume`, jitter uniformly within the chosen cell. **There is no
envelope, no rejection, and no proposal budget** — defects 2 and 3 both disappear, and the only
error left is the grid's own resolution, which is a tunable with a convergence check rather than a
sampled maximum with an 853x spread.

| | |
|---|---|
| implementation | moderate — comparable to B, roughly 50-80 lines, one `Config` field (grid resolution), and `_propose_points` becomes dead code for the field modes |
| invalidates | same as B: every `field`/`hybrid` number, T05's acceptance tests, T09's `layout_mode` gate |
| validated without a fit? | **yes**, and more strongly than B — with no rejection there is a closed-form expected count per cell to check against |
| caveat | the interaction (Strauss thinning) currently rides on the rejection loop and would need re-siting. A4 measured its contribution at 6 and 11 cells out of ~6000 and ~8700, so this is a small piece of work, not a blocker |

## Validation is cheap, and that is the load-bearing fact

`sample_layout` takes an `intensity_fn`, not a model. So **any** of these can be validated with no
fit and no data, by passing a closed-form spiky intensity whose true density is known and checking
the empirical point density against it — including a case built to have max/mean in the hundreds,
which is the regime that broke the current sampler. That is a fast unit test, not a slow one, and
it is the test the current sampler never had. It also gives a direct correctness check that
distinguishes B from D: D has an exact expected count per grid cell, B does not.

## What this means for the decision

* **Cheap and correct exists.** D is the cheapest *correct* option and is comparable in size to B.
  With C alongside it for reporting, the total is on the order of a day's work plus the re-measure,
  not a rework of the layout head.
* **A is dead on measurement,** and that matters because it was the option that would have required
  no re-measurement of anything.
* **The invalidation cost is the same for B and D** and is not small: T05's acceptance tests and
  T09's `layout_mode` gate were all computed on a biased sampler and would have to be re-run. But
  they are fixture-scale, not campaign-scale.
* **`resample` is untouched by all of this.** It reuses the flanking section's coordinates and never
  calls `_propose_points`, so shipping it needs no fix, no re-measure and no re-validation — at the
  cost of the layout-head claim, and noting that on current numbers `resample` is also the arm that
  scores best (`celltype_localization` 0.7601 against `hybrid`'s 0.6572).

The honest summary: the fix is affordable, but *the case for spending it is not that it will make
the layout head win* — the layout head is behind a model-free copy by 4.9x the reproducibility
envelope, and a correct sampler removes a bias, it does not supply a better intensity field.
