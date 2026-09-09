# A1's escalation — pre-registration

**Written before the runs, and before either piece is built.** `reports/a1_preregistration.md`
returned **row 5** on `deep_starmap`: A1b landed at `R = +0.33`, inside the 0.30–0.70 dead band, so
the verdict is NO CONCLUSION and the arm escalates. That pre-registration said *"escalate to three
seeds"* and did not say how to combine three seeds, nor what to do if they disagree. This document
fixes both, and fixes the criteria for the split that follows if seeds do not resolve it.

`specs/10` §4.2's closing rule is applied to each: §4 says what would have to be true for each to
fail.

---

## 1. N4 — the pre-registered escalation: A1b at three seeds

`--ablation-seed 1 2 3`, both datasets, draws only, no refit. The mean-field arms (`A1a'`, `A1b'`)
do not depend on the draw seed and are computed once; `A1a`, `A1b`, `A1c` and the permutation null
are redrawn per seed.

**Combination rule.** For each drawn arm, with `R_1, R_2, R_3` the per-seed recoveries:

| | criterion |
|---|---|
| **the arm's band** | the band containing `median(R_1, R_2, R_3)`, using `a1_preregistration.md` §3's bands unchanged (`>= 0.70` RECOVERS, `<= 0.30` DOES NOT RECOVER, otherwise UNINFORMATIVE) |
| **stability override** | if the three seeds do **not** all fall in the same band, the arm reads **UNRESOLVED** *regardless of the median*, and N4 has not settled it |

The override is the point of running three seeds at all. A median that sits in one band while a
seed sits in another is a verdict that a re-draw could have reversed, and this project has withdrawn
two of those. `min`, `median` and `max` of `I` and of `R` are reported for every drawn arm so the
margin is visible rather than inferred.

**Then, and only then, the decision table in `a1_preregistration.md` §4 is re-read** with A1b's new
band substituted. If A1b reads RECOVERS or DOES NOT RECOVER *and is stable*, the table returns a
conclusion and N5 is not needed for that purpose. If it reads UNINFORMATIVE or UNRESOLVED, the
table still returns row 5 and **N5 becomes the escalation that can resolve it**.

**Scope.** N4 changes no threshold and adds no arm. It is the escalation `a1_preregistration.md`
already required.

---

## 2. N5 — the split, an **addition** to the escalation and not a substitute

**N5 does not replace N4.** N4 is run and reported as written above whatever N5 says; N5 answers the
next question, which is *which* of the two mechanisms `theta`/`pi` lumps together is doing the work.
If they ever disagree, N4's reading governs the decision table and N5's governs the mechanism.

The reason to split: A1b differs from A1c only by the model's `theta` and `pi`, and the level column
already shows those act differently on the two datasets — A1b sits at **1.00x** the real count level
on tier-1 (so `pi ~ 0` and the loss is over-dispersion) and at **0.64x** on `deep_starmap` (so `pi`
is deleting about a third of the transcripts). One arm cannot separate them.

**Two new arms**, drawn at the same real cells, same panel, same seeds:

| arm | drawn from | isolates |
|---|---|---|
| **A1b-θ** | `NB(mu_oracle, theta_model)`, `pi` forced to 0 | over-dispersion alone |
| **A1b-π** | `Poisson(mu_oracle)` then zeroed with `pi_model` | dropout alone |

**Criteria.** With `L_total = I(A1c) - I(A1b)`, `L_θ = I(A1c) - I(A1b-θ)`, `L_π = I(A1c) - I(A1b-π)`,
all at the three-seed median:

| conclusion | criterion |
|---|---|
| **over-dispersion** | `L_θ / L_total >= 0.70` **and** `L_π / L_total <= 0.30` |
| **dropout** | `L_π / L_total >= 0.70` **and** `L_θ / L_total <= 0.30` |
| **both, additively** | each share in `(0.30, 0.70)` **and** `abs(L_θ + L_π - L_total) <= 0.02` in `I` |
| **not decomposable** | anything else — the two interact, and the split does not license attributing the loss to either |

**Reported either way**, so the additivity assumption is visible rather than assumed: the
multiplicative prediction `I(A1c) x (I(A1b-θ)/I(A1c)) x (I(A1b-π)/I(A1c))` against the measured
`I(A1b)`, with their difference in `I`.

**Guards inherited unchanged** from `a1_preregistration.md` §6: the level guard `[0.5, 2.0]` (and
note `A1b-θ` must come back at ~1.00x by construction, since `theta` does not move the mean — a
level away from 1.00x on that arm is an instrument fault, not a result); the oracle-is-an-upper-bound
asymmetry; and the contamination ordering — **`theta` and `pi` on `deep_starmap` come from a fit
whose spatial alarm fired at its final step**, so every N5 conclusion on that dataset is
**provisional** until a converged deep fit exists. Tier-1's fit is converged and its N5 result is not
provisional.

---

## 3. N2 — the two model-free estimates of the tissue's `sd(log mu)`

Not a gate; a measurement the record needs, stated here so the estimator is fixed before the number
exists. `reports/chain_shipped_review.md` §6 predicted the tissue's `sd(log mu)` at ≳ 2 against the
decoder's ~0.72, and `reports/a1_review.md` §7 recorded that prediction as **untested** because the
quantity was never computed.

Both estimates use the lognormal identity `sd(log mu) = sqrt(log(1 + CV²(mu)))`, which is an
approximation and is labelled as one, and neither needs a pseudocount:

* **smoothed-field estimate** — `CV²` of `mu_oracle` directly. Smoothing shrinks variance, so this is
  a **lower** bound on the tissue's spread.
* **Poisson-deconvolved moment estimate** — `CV²(mu) = (Var(y) - mean(y)) / mean(y)²` per gene,
  clipped at zero. Any over-dispersion in the tissue inflates it, so this is an **upper** bound on
  the structured part.

The two bracket the tissue's `sd(log mu)`. The same two estimators are applied to the **model's
emitted counts** as well, so the model and the tissue are compared on the same estimator rather than
against the decoder's own internal figure.

**Pre-registered reading**, fixed here: `chain_shipped_review.md` §6's mechanism is **supported** if
the tissue's lower bound exceeds the decoder's `sd(log mu)` (0.7131 on deep, 0.7269 on tier-1) by
`>= 1.5x`; **refuted** if the tissue's upper bound is below it; **untested still** in between.

⚠️ **How loose the lower bound is depends on the field's own autocorrelation, and this is stated
before any N2 number exists.** The instrument's self-check smooths a lognormal field at `sigma = 0.8`
laid on *random* positions — no spatial structure at all — and the estimate falls to **0.279**, a
shrinkage factor of 0.35. That is the worst case: a kNN mean destroys the variance of a field whose
neighbours are unrelated, and preserves it where they are not. On these panels `I(mu_oracle)` is
**+0.906** (deep) and **+0.928** (tier-1), so the shrinkage there will be far milder — but the
**reported `I(mu_oracle)` is the indicator of how tight the bound is, and must be quoted beside it.**
If the lower bound lands below the decoder's figure, that is **not** evidence against §6: it is
consistent with either a narrow tissue `mu` or a loose bound, and the verdict is "untested still",
never "refuted". Only the **upper** bound can refute.

---

## 4. What would have to be true for each of these to fail

* **N4** fails to settle if the three seeds straddle a band boundary — which is likely, since A1b
  sits **0.0070 of `I`** from the lower edge and a fresh draw at ~30 000 cells plausibly moves it by
  that much. That is the expected outcome, not a surprise, and it is why N5 is pre-registered now
  rather than after.
* **N5** fails if the shares land outside every row — the "not decomposable" case — which happens
  when zero-inflation and over-dispersion interact rather than compose. The additivity check is
  reported precisely so this is visible.
* **N5's** `A1b-θ` level is a self-check that can fail: `theta` cannot move the mean, so a level away
  from 1.00x means the arm is not what it claims.
* **N2** fails to resolve when the two bounds straddle the decoder's figure — a real possibility,
  and the "untested still" verdict exists so that outcome is reportable rather than forced.

None of the four can return a foregone answer, and two of them (N4's straddle, N2's straddle) are
the *likely* outcomes rather than edge cases.
