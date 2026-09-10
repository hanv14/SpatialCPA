# Pre-registration — the abundance-floor rescaling

**Committed before the statistic is implemented or computed.**

## 1. The interest, stated first — the same discipline as the flanking test

This rescaling moves v25's headline from *"0.7306 against a copy's 0.9886"* — 27 % short — to
something like *"45 % of the way from the abundance floor to the copy"*. **That is a favourable
reframing of a negative result, and it is the second statistic in a row this project has an
interest in.** So the same two rules apply, and one more:

> **The raw comparison is the default and is reported first, always.** `paper_morans_pearson` as
> the benchmark computes it, v25 against `flanking_copy`, unrescaled. The rescaled figure is
> reported **beside** it and never in place of it. A reader who takes only the first number away
> must not have been given a flattering one.

> **Absent a clear result, the raw comparison is all that is reported.** Every failure mode in §4
> resolves that way.

> **The floor is fixed here, before it is computed** (§3). Three defensible floors exist and they
> will not give the same answer; choosing among them after seeing which is kindest is the failure
> this document exists to prevent.

**This does not reopen the diagnostic programme.** The ladder answered the closeability question
*no* and the flanking test removed the benchmark escape. This changes how the negative result is
*expressed*, not what it is. If the measurement below turns up anything that would reopen the
question, that is said explicitly and separately — it is not folded into a rescaling.

## 2. What is being measured

F3 (`flanking_copy_preregistration.md` §5a) destroys gene-specific spatial identity while keeping
the abundance–`I` relationship exactly. Applied to a prediction's own per-gene `I` vector it gives
**the score that prediction would achieve from abundance alone**. The interval between that floor
and the copy is the part of `paper_morans_pearson` that gene-specific spatial fidelity can move.

F3 is currently computed for `flanking_copy` only. It is extended to stage 4, `A1a`, `A1b`, `A1c`
and `A1n` — every vector already exists; only the relabelling is new.

## 3. The floor, fixed now

Three candidates, and the choice changes the number:

| | floor | why not |
|---|---|---|
| **F3 of the prediction's own vector** | each rung rescaled by its own abundance floor | a rung that is *entirely* abundance rescales to 0 by construction, which flatters any rung that is not — and it makes the rungs incomparable, since each has a different denominator |
| **F3 of `flanking_copy`** | one floor for every rung | ✅ **CHOSEN** |
| the permutation null (≈0) | no rescaling at all | this is the raw comparison, which is reported anyway |

**The floor is F3 of `flanking_copy`, median over the usable stratum widths, on the scope being
reported.** One denominator for every rung, taken from the *reference* prediction rather than from
the one being scored, so no rung can improve its own scale. The rescaled value is

```
rescaled(x) = (r(x) - F3_copy) / (r_copy - F3_copy)
```

reported to two figures, with `F3_copy` and `r_copy` printed beside it every time.

**Each rung's own F3 is still reported**, as a separate column, because it answers a different and
useful question — *how much of this rung's score is abundance?* — and because suppressing it would
leave only the flattering statistic. It is **not** used as a denominator.

## 4. When the rescaling is not reported at all

Any of these and only the raw comparison stands:

1. **F3 has no usable width** on the scope (`null_band_preregistration.md` §2a-ter). Tier-1's 28
   genes may make this permanent there.
2. **F3 is unstable across widths** — the usable widths span more than 0.10 in `F3_copy`.
3. **`F3_copy` is not clearly below `r_copy`** — `r_copy - F3_copy < 0.20` makes the denominator,
   and therefore every rescaled figure, unstable.
4. **`F3_copy` is not clearly above the permutation null** — if the abundance floor is ~0 there is
   nothing to rescale by and the raw comparison already is the answer.
5. The scope's null check fires (§2a).

## 5. Predictions, recorded now

1. `F3_copy` on deep/all-genes lands in **0.50 – 0.55**. The first run measured 0.5461 / 0.5145 /
   0.5197 at widths 10 / 25 / 50, so this is close to a restatement and is recorded as such rather
   than claimed as a prediction.
2. **v25 rescales to 0.40 – 0.50.** From the first run's numbers, (0.7306 − 0.52)/(0.9886 − 0.52)
   = 0.449.
3. **`A1a` rescales BELOW v25** on deep and the ordering of the ladder is unchanged by rescaling.
   If rescaling reorders the ladder, that is a finding about the statistic and the rescaling is
   withdrawn until it is understood.
4. **v25's own F3 is HIGHER than the copy's** — its per-gene ordering is more abundance-determined
   than the copy's (`R2` against detection: **+0.9027** for v25, **+0.6481** for the copy, against
   the tissue's own **+0.6787**). If so, that is a nameable defect of v25 and is reported as one,
   in the same breath as the rescaling that flatters it.

## 6. What the paper may say

**May say.** v25 reaches *X* % of the interval between what abundance alone achieves and what a
model-free copy of a neighbouring section achieves, on `deep_starmap`, all genes, replicated across
sections 2, 4 and 6 — with the raw figures given first.

**May not say.** That the benchmark is flawed, that the gap is smaller than it is, or that the
rescaled figure is the score. It is a decomposition of a metric we lose on, offered so the loss is
legible; the loss is unchanged.

---

## §3-bis — AMENDMENT: two properties of the floor, both found while building it, both before it was computed

### (a) F3 must derange, not merely permute

A permutation within a stratum of width `w` leaves each gene on itself with probability `1/w`, so a
fraction `1/w` of the true agreement survives the relabelling and the floor comes out **too high**.
Measured on a fixture whose control is independent of the truth — where the floor must be zero — a
plain permutation gave **+0.141**.

The direction matters and is recorded here rather than after the fact.
`rescaled(x) = (r_x - f)/(r_copy - f)` **decreases** in `f` whenever `r_x < r_copy`, so the inflated
floor was *understating* v25's rescaled score. **Removing the bias moves the headline in this
project's favour.** It is removed because the estimator was wrong, and the fact that the correction
happens to flatter us is stated in the same sentence as the correction. F3 now uses a derangement;
`_derangement` raises on a stratum of one rather than returning the identity, and a trailing
remainder of fewer than two genes is merged into the previous stratum.

The first run's published F3 figures — deep's 0.5461 / 0.5145 / 0.5197 — were computed with the
biased permutation and are **superseded**, not merely refined.

### (b) The floor is not an upper bound on "what abundance alone can score"

`F3_copy` is *what the copy's own per-gene values achieve when relabelled within abundance strata*.
That is not the same as *the best any abundance-matched prediction can do*, and it can be **lower**:
on a fixture where the truth is half abundance and half gene-specific, `F3_copy` came out at
**+0.290** while a prediction that is a noiseless function of the control alone scored **+0.49** —
above the floor. The floor is the square of the shared loading; a prediction that *is* the loading
is not.

So `rescaled = 0` means *"scores what the copy's values score once gene identity is destroyed"*, not
*"contains no gene-specific information"*, and a rung can rescale above 0 on abundance alone.

§3's choice stands unchanged — one denominator for every rung, taken from the reference rather than
from the rung being scored, is still the property worth having. But **the paper may not write
"rescaled 0 = abundance alone"**. The sentence permitted by §6 is unchanged and already correct:
*the interval between what the copy's values achieve relabelled and what they achieve intact*. Each
rung's **own** F3 column is what answers "how much of this rung is abundance", and §3 already
requires it to be printed.
