# T09 close-out — SpatialCPA-v25-Gen

**2026-09-07.** Written after the `cosmx` replication, A7, and the pool-sparsity test. Nothing is
pending measurement. This is the consolidated statement; the working record is
`progress/t09_inference_and_calibration.md`, which carries every number's derivation and every
correction made to it.

---

## 1. The headline I would defend

> A continuous 3D field is a good **representation** of a tissue volume and a bad **generator**
> from it.
>
> Oblique planes reconstruct at **95.5 %** of axis-aligned quality; per-gene spatial
> autocorrelation is controllable through a 3D GRF prior; and two crossing sections emit **bitwise
> identical** expression along their intersection, exactly and without training. But every
> generative component built on that representation loses to **copying a real section** — the
> intensity-field layout, the flow-matching expression head, and the sectioning-equivariant losses
> the method is named for, which make it actively worse. One capability claim survives replication
> on a second dataset: text embeddings place genes the model never saw above a measured floor. The
> generative failure is localised and mechanical rather than diffuse: the decoder reproduces the
> *pattern* of between-cell variation almost perfectly and a fraction of its *amplitude*, and the
> ZINB objective closes the gap with dispersion instead of with the structured mean.

The last sentence is the one I would defend hardest. It is the difference between "our expression
head underperformed" and a finding.

---

## 2. What works

Three claims, all about **encoding**. None is about generation.

**2.1 The correlated prior controls spatial autocorrelation (GATE 1).** Error ratio **0.130**
against an i.i.d. prior, per-gene `I_gen` vs `I_real` correlating at **r = 0.917**, and median
`I_gen` monotone as `ell` sweeps 0.25x–4x. On the synthetic fixture — stated as such.

**2.2 Oblique reconstruction reaches parity (GATE 2).** Depth-matched parity **0.955**,
edge-excluded **0.979**, against a pre-registered **≥ 0.90**. A clean, unqualified pass on real
data, and the strongest single result in the project.

**2.3 Intersection consistency is exact by construction.** Two crossing sections emit **bitwise
identical** expression along their intersection on an **untrained** model, with no consistency loss
applied (`test_generation_is_intersection_consistent_by_construction`). Not approximately, not at a
particular checkpoint: the 3D noise field is continuous and every conditioning pathway — retrieval,
the GRF, the Fourier encoding — is queried at **physical** points while `CTFFlow.generate`
conditions with the identity pose whatever plane it is handed. The property therefore holds for
every model of this architecture, at every checkpoint, exactly.

Three things make it a result rather than an implementation note:

* **It is a stronger claim than the loss written to earn it.** `L_cross` exists in `specs/07` to
  *train* approximate agreement. The architecture already has exact agreement.
* **It survived that loss being shown unnecessary *and* harmful.** `L_cross` is vacuous in v25, and
  the only plane-dependent channel left for it to act on is the augmentation pose — which T04 made
  pose-dependent on purpose — so minimising it destroys the field: per-gene variance falls to
  **0.065** of real at `w_cross = 0.3`, against **0.711** with SEFL off. `w_cross` ships at **0**,
  and the property was unaffected, because it never depended on it.
* **No competing method has it.** Independent per-plane generation gives crossing sections no
  reason to agree anywhere. It is a property of committing to a continuous 3D field — the thesis
  the generative half failed to support.

**2.4 Text beats a gene-blind arm on unseen genes — replicated. *Which* text path does it — not.**
The qualification belongs in the header because this line will be quoted alone, and §3 records the
other half of the same experiment as PARTIAL and DOES NOT REPLICATE.

What replicated, under criteria fixed before the fits, on a second dataset: A2 (`medcpt`, pure
text) clears the `shuffled` floor at **2.52x** on `deep_starmap` and **2.08x** on `cosmx`, and
A2 − A4 — text against an arm that sees no gene identity at all — is **+0.3949, 1.96x** the shared
envelope with signs agreeing on 3/3 seeds and 6/6 folds. The void condition holds at 0.12x. **Two
independent positives on two datasets.**

What did **not**: A2 − A3, the pure-text projection against the distillation head, which is the
contrast that would say *how* the text channel reaches an unseen gene. See §3.

---

## 3. What does not work

**Every generative component loses to copying — the same trivial baseline each time.**

| component | result | measurement |
|---|---|---|
| intensity-field layout | **REFUTED** | `field` 0.6607, `hybrid` 0.6692 against `resample` **0.7546**; the model-free copy floor is 0.7765 and the oracle ceiling 0.9808. Both field modes score **below the floor** on the metric the layout head exists to win, by 3.5x and 3.2x the envelope. `resample` ships. |
| flow-matching expression head | **REFUTED, both datasets** | on `deep_starmap` `cross-mix` (copying) wins **every live metric**; on tier-1 by 4.6–5.3x the envelope on three. The one metric where generation had won was an artifact of the frame defect and reversed when it was fixed. |
| SEFL — the mechanism the method is named for | **REFUTED, 3 seeds** | the SEFL arm **collapses** the anatomical field: `i_gen` at **1.6–2.4 %** of target against the off arm's 95.6–97.5 %; five of seven metrics cost with signs agreeing 3/3 at 1.31x–4.68x. `check_collapse` fired **218 times** across the three ON fits, from step 250 of 1200. All three weights ship at **0**. |
| the mechanism half of the zero-shot claim | **PARTIAL** | A2 − A3 — the pure-text projection against the distillation head — is **+0.0450, 0.22x** the shared envelope, where `deep_starmap` had +0.2514 at 2.7x. `specs/10` §7's mechanism sentence is **withdrawn**. |
| the seen/unseen sign flip | **DOES NOT REPLICATE** | signs reversed on **12 of 12** seed x fold cells across both gene pools, and **both** magnitudes sit inside their pool's envelope (0.22x, 0.51x). The direction replicated; the effect size did not. |

**The sentence that follows, and which had not been written down.** v25 ships `resample` +
`zinb-flow`: real positions with generated expression. v20's fallback is `resample` + `cross-mix`:
the same real positions with the donor's counts copied. The difference between them is exactly
v25's contribution to that pairing, and **it is negative on every live metric.** A reader will work
this out; the paper should say it first.

---

## 4. The one component that ships ON while established by nothing

Flagged here rather than left in a table, because it is a different kind of problem from everything
in §3.

`w_autocorr = w_profile = w_distribution = **0.5**` ship **on**. At 1200 steps they lose (rank 3.5
against 3.0, a cost on every metric); at **2400** — the budget T09 selected — they win the
selection on aggregate rank, 1.0 against 2.0, taking four of six metrics. That is why they ship.
The per-metric margins at 2400 are **0.0052 / 0.0101 / 0.0018** against R10's **0.0335** envelope:
every one inside it, by factors of 3 to 19, on **one seed**.

🚨 **And the selection ran on the SYNTHETIC FIXTURE** — 19 fits at ~8 minutes each, against the 56
minutes a real tier-1 fit takes. So the accurate statement is stronger than "one seed with small
margins": these weights ship on **an aggregate rank over the fixture, and have never been measured
on real data at all.** That is the pattern R11 already burned, where a fixture tie-break inside the
envelope put `hybrid` ahead and real data reversed it — the fixture was *"underpowered, not
wrong"*, its flanking baseline sitting at 58 % of its ceiling against real tissue's 79 %, so it
over-rewards a generative addition at any number of seeds.

The selection is sound as a selection. It is **not evidence that the losses do what they are named
for**, and `claim_min_seeds = 3` says one seed cannot resolve it either way.

| state | components |
|---|---|
| ships **off**, evidence against | SEFL's three weights, `w_cross`, the intensity-field layout |
| ships **on**, evidence for | `prior_mode="correlated"`, `layout_mode="resample"`, `decoder_mu_link="exp"` |
| 🚨 ships **on**, established by nothing | the three metric-aware weights at 0.5 |

A component that ships **off** with evidence against it is an honest negative: the reader learns
something and the shipped model does not depend on it. A component that ships **on** while
established by nothing is inside the baseline **every number in this project was measured
against** — it is not a claim the paper makes, it is a claim the paper's *setup* makes silently.

**A9 has now run — six fits, three seeds — and returned UNINFORMATIVE.** Pre-registered condition
(a) fired: the worst primary envelope is **0.4323** against the 0.067 the condition names, **6.5x
over**, and both autocorrelation primaries had signs disagreeing across seeds. **The weights remain
unestablished, and a three-seed real-data design could not resolve them** — so "run more seeds" is
not a cheap path either. Two things the run found instead:

* 🚨 **The collapse alarm is disarmed whenever SEFL is off** — `train_ctfflow` gates it on
  `sefl_teacher is not None`, and the teacher exists only when a SEFL weight is above zero. The
  shipped configuration has all four at zero, so **on the shipped model `check_collapse` never
  runs**, and every empty alarm list in every SEFL-off campaign means *never armed* rather than
  *did not fire*. An owed fix, and §4.2f failing in the instrument written to enforce it.
* 🚨 **Applied by hand, that alarm would have fired on all three ON fits.** Median `variance_ratio`
  is **0.86 / 0.77 / 0.81** off and **0.14 / 0.08 / 0.17** on, against a 0.25 collapse threshold —
  and A7's *collapsed* arm settled at 0.105–0.193 on the same statistic. Meanwhile `spatial_ratio`
  goes **up** (1.4–1.6 → 2.2–2.5), so the autocorrelation term hits its target while the amplitude
  drains: R4's shape in a fifth place. **Candidate, not established** — it decides no A9 branch.
* Wall clock: the ON arm costs **1.63x** (93 min against 57).

**What honesty requires is cheap: say it.** State that the weights ship at 0.5 on a
**fixture-selected** aggregate rank with per-metric margins inside the reproducibility envelope,
that a three-seed real-data test could not resolve their contribution, and that on the diagnostic
the shipped model does not watch they drive per-gene variance into the collapse regime.

**It does not weaken the negatives.** Each of R11, A7 and the zero-shot arms is a
*within-configuration* contrast with the same weights on both sides. The exposure is to the
**absolute** numbers and to the claim that the shipped model is the best configuration found — not
to any of the differences.

---

## 5. The mechanism under the generative failure

**R12, localised.** The chain measures median Moran's I of 0.9714 at the GRF prior, 0.9015 after
the flow, 0.8607 at the decoded `mu` — then **0.1297** at the sampled counts. Stages 1–3 lose 0.11
in total; the count draw loses **0.73 in one operation**. Real tissue retains **62 %** across the
same latent→counts step; the model retains **14 %**.

**R4, the trade that explains it.** The decoder reproduces the *pattern* of between-cell variation
almost perfectly — `mu`'s Moran's I is **0.861**, above the tissue's own latent at 0.745 — at a
fraction of its *amplitude*, and the ZINB objective closes the gap with dispersion. `theta` carries
**57–63 %** of the conditional variance and correlates with the data's own dispersion at Spearman
**0.068** over 1017 genes. It is absorbing unpredicted mean variation, not estimating noise.

That is a property of the objective, not a tuning error: **no data-derived value of `theta` exists
to match to**, which is why the moment-matching experiment was stopped after one fit. The same
shape was measured four separate ways (R4 i–iv) — in each case a likelihood is reduced by moving
explanatory power out of the structured component into the unstructured one, with nothing in the
objective opposing it.

---

## 6. The methods contribution, and why I think it is the stronger half

Nine rules (`specs/10` §4.2a–i). They are worth a paper not individually but as a pattern:

> Every claim in this literature has the form **"the margin exceeds the noise"**, and that sentence
> hides six independent choices. In this project each one silently decided a verdict before anyone
> noticed it was a choice.

* **§4.2a** which arm's variance is the noise — a pooled envelope was too lenient on three metrics
  and too strict on two, and the worse arm alternates by metric.
* **§4.2b** which envelope a clearance takes — two arms 0.004 apart landed on opposite sides
  because one varied less.
* **§4.2c** which referent is a floor at all — the pre-registered constant-field band had
  bitwise-identical input.
* **§4.2d** how the spread is aggregated — an effect read 1.12x under fold-averaged noise and 0.75x
  under per-fold; reported as standing, withdrawn.
* **§4.2g** which arms may contribute a spread — on the replication the envelope was set on **both**
  pools by a **degenerate** member, at 6–18x every informative arm's variance, so an effect
  consistent on 12 of 12 cells could not clear it.
* **§4.2h** which side of the threshold refutes — a two-sided "within 1.5x" band on a one-sided
  hypothesis returned *false* on the result that refuted it most strongly.

Plus two failures of a different kind — **§4.2e**, a metric that is a correlation on a dead field
(one seed inverted a headline's sign on two of seven metrics); **§4.2f**, an alarm that fired 218
times where no report read it — and one standing requirement that falls out of all of them:

* **§4.2i** — a three-seed envelope is an **estimate**, and none of ours was ever reported as one.
  For a correlation over `n` genes the null has `sd ≈ 1/sqrt(n-1)`; the range of three draws has
  expectation `1.693 sd` and its own `sd = 0.888 sd`, a **CV above 50 %**. All three measured
  envelopes are ordinary draws from that null, so the 2.2x gap between two datasets' envelopes —
  which reads like a fact about the datasets — is noise in the noise estimate.

I think this is the stronger half because it transfers. The negative result is about one method on
two datasets; the methodology is about how anyone reads a repeated-seed benchmark, and none of it
is reported in this literature.

⚠️ **The honest discount.** Every one of these was found *because a criterion in this project nearly
returned the wrong answer*, and in four cases it had already returned one that was recorded and
later withdrawn. That is the right provenance for a methods paper and should be stated as such
rather than presented as foresight: **we found these by getting them wrong first.**

---

## 7. Budget provenance — the objection a reviewer will raise

`Config.train_steps` defaults to **1200**; T09's selection gate chose **2400**; and T08 showed the
metric-aware terms **reverse sign** between the two. So "at which budget was this measured?" is a
live question. Checked:

| result | budget measured at |
|---|---|
| layout (R11) | **2400** for the headline swap; count error confirmed at **both** |
| zero-shot (E1 + replication) | **2400** — both campaigns, all twelve fits |
| metric-aware | **both** — and it is the one case where the budget flips the sign |
| **SEFL (A7)** | 🚨 **1200 only** — the default, not the selected budget |

**It touches A7 alone**, and the record already scoped it that way: *"SEFL adds nothing at the
default budget of 1200 steps"*, not "to the shipped model". The metric-aware reversal is proof the
objection is not hypothetical.

**It does not rescue A7.** That arm did not underperform, it **collapsed** — `i_gen` at 1.6–2.4 % of
target, 218 alarms from step 250 of 1200. A budget that ends after 950 steps of a dead field is not
obviously one that escapes the collapse, and the burden is on the claim that more steps recover it.

**It does not make the negatives provisional.** The layout and zero-shot results are the bulk of the
negative column and were measured at the selected budget. Writing the caveat broadly would be the
§4.2 failure in reverse: a discount applied to soften an answer that was measured correctly.

---

## 8. What remains open

1. **R4 / the ZINB trade.** The largest, unchanged. A measured mechanism, a named term, and a
   design change — objective or emission model — as the only route. A follow-up paper, not a
   follow-up measurement.
2. **R12's two questions.** The 46–119x per-gene spread in `s`; why `theta` is uncorrelated with the
   data's dispersion. Both owed to T06.
3. **§4.2g / §4.2i's construction question.** How to take an envelope when some members are
   degenerate by design, and how to report one three seeds cannot pin. **Must be settled before the
   next pre-registration, not after the next verdict.**
4. **The metric-aware weights** (§4). Unresolved, shipping on, inside the baseline everything else
   was measured against.
5. **The `cosmx` effect-size shrinkage.** Direction replicated 12/12, magnitude fell 5.6x, nothing
   explains it. Two candidates eliminated (pool sparsity, per-gene sparsity — both point the wrong
   way). The **density difference** (`cosmx` 5.7x denser per gene) is recorded as a candidate with a
   stated test and is **deliberately not being chased**: fourth mechanism hunt of the campaign, and
   the base rate governs.
6. **`marker_field_r` — a weakness that has survived three method generations.** It has no row
   anywhere above and it has appeared **four times**, which makes it an open item rather than a
   table entry:

   | # | where | what |
   |---|---|---|
   | 1–2 | v20 and v21 against SpatialZ | the **pooled** loss across datasets — **withdrawn as evidence**, see below |
   | 3 | v25 at T09, on the fixture | its **single** losing metric |
   | 4 | v25 at T10, tier-1 STARmap | **0.6384 against a `flanking_copy` floor of 0.8857 — 0.247 below it, 7.4x the envelope**, the worst of the six |

   ⚠️ **State appearances 1–2 at their real strength, which is lower than it sounds.** That loss is
   a **cross-dataset pool**, and §4.2a forbids exactly that: read per dataset it is **9–9**, on
   tier-1 v20 (0.8804) and v21 (0.8881) both **beat** SpatialZ (0.8522), and in the wide regime v20
   wins **7 of 7**. Only the pooled figure favours SpatialZ. So the honest count is **two clean
   appearances in v25 plus a pooled comparison our own methodology rejects** — which is still a
   pattern, and still the metric where this line has been weakest for three generations.

   **What makes it worth opening rather than noting.** The tier-1 deficit is not expression
   magnitude: `gene_mean_spearman` sits **0.0033** off its copy floor, inside the envelope, so
   per-gene magnitude is solved and the residual is **spatial arrangement**. And it is one of only
   two **pose-dependent** metrics, with tier-1 rotations of 0.0–1.5 degrees — so alignment is
   measurably **not** confounding it, and a v25 deficit at ~0 degrees points at **T05's intensity
   head**, not at `align.py`. That is a specific place to look, which is what distinguishes an open
   item from a complaint.

   ✅ **DONE 2026-09-07 — and the redirect is the useful half.** The boundary candidate is
   eliminated, but what the elimination *points at* is worth more than the elimination:
   **`resample` does not use the intensity head to place cells at all, and still carries a 0.19
   deficit.** The weakness survives removing the layout head from the picture entirely, so it is
   not T05's — it is in the **expression path**, R12's territory. §8.6 predicted the intensity head
   via the boundary regime; that route is closed and the prediction is **weakened, not confirmed**.
   The remaining search is arrangement in the emission, at every depth.

   **The measurement.** Zero fits and zero generation — both sides were already on disk. On
   `resample-grid`, the
   **shipped** layout mode and the only arm whose sections are density-comparable (spread
   **1.04x**), the deficit below each section's own copy floor is **0.1729 / 0.1877 / 0.2043** and
   the boundary-vs-interior gap is **0.69x** the envelope: **BOUNDARY ELIMINATED**. The weakness is
   **uniform along the stack**, so R3's one-sided-evidence regime and the boundary-clamp geometry
   are **out**.

   The other four arms return INVERTED and **must not be read**: pre-registered condition (c) fires
   on all five, but the field-based modes emit **63.9x** the truth at `section_2` and **0.90x** at
   `section_6`, so the boundary section is scored on a 64x-thinned, better-conditioned draw. The
   "inversion" is the count error read through the metric.

   `reports/t10_marker_field_boundary.md`. **Closed as far as free measurement goes**: nothing else
   about this metric can be settled without a spend.

7. **R14's donor rule.** Costs 0.116 of `marker_depth_r`, deliberately unfixed because fixing it
   makes the negatives **stronger**.

**Closed since 2026-09-01:** the replication (PARTIAL / DOES NOT REPLICATE), A7 (SEFL harmful), the
`decoder_mu_link` refit (not owed — `exp` shipped before every real-data audit), the moment-matched
`theta` (no value exists to match to), and the pool-sparsity guess (refuted, direction backwards).

---

## 9. What would change this assessment — and what would not

**Would:** a change to the emission model or its objective that closes R4, followed by a re-run of
the six-metric comparison against copying. If generated expression beat `cross-mix` on real tissue
after that change, the headline inverts from "a good representation and a bad generator" to a
method paper. This project cannot run it.

**Would not: more seeds on the results already measured.** §4.2i is the reason. The envelopes three
seeds produce are estimates with a CV above 50 %, so a fourth and fifth seed on an existing
comparison buy precision in the **noise estimate**, not evidence about the method. The zero-shot
claim in particular is as strong as three seeds on two datasets can make it, and adding seeds there
would sharpen an envelope rather than change a verdict.

**The two statements below are about different objects and are not in tension.** The paragraph
above is about **precision on comparisons already run**. The one below is about **resolving a
component that has never been measured at three seeds at all** — an unrun comparison, not a rerun
one. More seeds do not help the first; three seeds are the minimum that can settle the second.

**The cheapest thing that would change something real** is §4's six fits: `(2400, on)` against
`(2400, off)` at **three** seeds, where the existing evidence is **one**. That is not extra
precision on a measured result; it is the first measurement that `claim_min_seeds` would accept, on
a component that ships **on** and sits inside the baseline every other number was produced against.
It converts a silent assumption into a stated result — in either direction, since a null there is
as publishable as a positive.

I am **not** recommending it over the R4 work, which is the larger question and which this project
cannot run anyway. I am recording that it is the only small spend on the list that touches the
configuration every other number was measured on.
