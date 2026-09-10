# The diagnostic programme is closed

**Date of closure: the round that returned `1. FLOOR IS GENUINE` on both datasets.**

## 1. The question, and the answer

The programme existed to answer one question: **is v25's reconstruction deficit closeable inside
this architecture?** Two results answer it, and both were read against criteria fixed before the
numbers existed.

**The measurements are on the published scale.** The chain reproduces bench3's own tier-1
`paper_morans_pearson` — `flanking_copy` at 0.9836 against a published 0.9836 (exact) and v25 at
0.5600 against 0.5574 — so the tier-1 ceiling below is in the units the paper reports
(`reports/scale_anchoring.md`). Deep is not anchored and its figures stay internal.

**The ladder said no.** Grant the model a perfect mean field and keep its own emission — that is
`A1b` — and it reaches **+0.8242** on `deep_starmap`, all 1017 genes. That is the ceiling: no
arrangement of the prior, the flow, `ell`, SEFL, θ or π gets past it, because `A1b` already grants
all of them a perfect result. On tier-1, in published units, the same rung is **0.8369** against a copy at **0.9836**, and the
0.424 gap splits into near-equal thirds — latent 0.139, mean field 0.138, irreducible 0.147. Fixing
the first two perfectly still loses by the third.

⚠️ **A claim in this section has been reversed.** It read *"a better latent is worth nothing"*. That
holds on deep (`A1a` +0.6732 below stage 4's +0.7306, the low rung with the positional advantage on
the wrong side) and is **false on tier-1**, where `4 − A1a` is −0.1851 / −0.2055 / −0.1017, resolved
on all three sections — a better latent is worth **~0.14 in published units** there, one third of
the gap. The permissiveness escape in §2a was available and is declined, because these runs bound
the advantage it rests on: `flanking_copy` scores 0.95–0.99 from an entirely different section's
cells. Full record in `reports/latent_claim_reversal.md`; the corrected reading is
`architecture_ceiling.md` §0 and §3.

**The flanking test removed the benchmark escape.** `flanking_copy` scores **+0.9886** on the same
scope and retains **97.9 – 99.0 %** of it under all three controls, with the three spanning 0.011
against a 0.150 tolerance. `spatial_scramble` is at the null. The copy floor is genuine spatial
fidelity. The deficit is ours.

**So: the deficit is not closeable inside this architecture, and there is no benchmark defence.**

## 2. What that means for the work

**The remaining work is writing, not measuring.** Three things are still owed and none of them is a
diagnostic:

1. the `flanking_copy` replication on sections 2 and 6 of both datasets, required by
   `flanking_copy_preregistration.md` §5d before anything here is stated as a fact;
2. the abundance-floor rescaling (`abundance_floor_preregistration.md`), which changes how the
   negative result is **expressed** and not what it is;
3. the paper.

Everything aimed at the prior, the flow, `ell`, θ, π or SEFL is held. `reports/architecture_ceiling.md`
records the sizes as measured defects and says in its §4 what may not be read as a plan.

## 3. What would reopen it, stated so it can be checked rather than argued

The closure rests on four load-bearing measurements. If any of the following turns up, it is said
**explicitly and separately**, not folded into a rescaling or a framing:

| | would reopen the question |
|---|---|
| `A1b` | a scope where a perfect mean field with the model's own emission reaches the copy floor. The ceiling is a *measured* 0.8242, not an argument. |
| `A1a` vs stage 4 | a scope where `A1a` is **below** stage 4 by more than the bootstrap interval **and** `A1a` is the low rung, i.e. the advantage points the other way. Tier-1's `4 − A1a` = −0.1851 is **not** this: there `A1a` is the *high* rung and §2a makes it permissive. |
| `flanking_copy` | a section where the copy's retained fraction falls below 0.60 under all three controls, or where F3 shows its score is abundance. Two datasets have now said the opposite. |
| the null | any scope whose 20-seed permutation arm is not centred. Three have been centred. |

Absent one of those, further measurement does not change the answer, and proposing it would be
work that cannot alter a conclusion.

## 4. What the programme produced

Not a repair. A localisation, a ceiling with a size, and a benchmark result that stands up:

- on **deep**, the loss is not in the prior, the flow, `ell` or SEFL — the flow's latent scores
  *above* the encoder's posterior on the truth. On **tier-1** the latent is worth ~0.14 of a 0.424
  gap, and neither statement generalises to the other dataset (`latent_claim_reversal.md`);
- the largest single term is the encoder–decoder pair, **0.151**, with `sd(log mu)` from `h1` at
  0.4797 against the tissue's [1.0994, 1.3699]. Recorded as a defect with a size, explicitly not a
  repair route, because a perfect version of it is `A1b` = 0.8242 and still loses to copying;
- on the **scored** metric π costs 0.102 and θ 0.018, inverting N5's median-based verdict and
  showing the θ programme was aimed at the cheap half (`reports/pi_theta_reversal.md`);
- `paper_morans_pearson`'s copy floor is real spatial fidelity on **both** datasets and all six
  sections, and we lose to it;
- the two datasets are different measurement regimes: about half of deep's metric is per-gene
  abundance and almost none of tier-1's is (`reports/panel_regime.md`), so their scores are not the
  same quantity and may not be pooled;
- the instrument reproduces the published tier-1 numbers, which closes an uncertainty every earlier
  result carried (`reports/scale_anchoring.md`).

## 5. The method, since it is the part worth keeping

Every result above was read against criteria committed before its numbers existed, and the
pre-registrations earned their keep by firing against this project's interest at least three times:

- **§6 prediction 3 was wrong in our favour.** I predicted the flanking test would return PARTIAL —
  a result that would have left the benchmark critique alive. It returned outcome 1 by a wide
  margin, and the prediction being on the record is what makes that legible.
- **F3's fixed-point bias** was found while building the abundance floor, and removing it *raises*
  v25's rescaled score. It was removed because the estimator was wrong, and the direction is
  recorded in the same sentence (`abundance_floor_preregistration.md` §3-bis(a)).
- **`spatial_scramble`'s `[:5]`** was mine and unregistered; it manufactured a failed prediction on
  tier-1 that the identical 20-seed construction contradicts in the same report (§2a-bis).
- **The latent claim** was written from deep and stated generally, and the dataset that refutes it
  refutes it 3/3. It is the **third** general claim in this campaign written from one dataset
  (`latent_claim_reversal.md` §"The pattern").

`specs/10` §4.2's rules — §4.2f-i, §4.2k, §4.2l, §4.2m, §4.2n and now §4.2o — were all earned by this
project's own defects, most of them in instruments built to catch the previous one.
