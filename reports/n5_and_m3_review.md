# N5 read out, and a defect in M3's criteria caught before M3 runs

Sources: `reports/a1_tier1.md`, `reports/a1_deep.md` (three seeds, N5's split arms),
`reports/m3_tier1_baseline.md` (the `theta` distribution). Criteria:
`a1_escalation_preregistration.md` §2, `m3_preregistration.md`. **No code was changed.**

---

## 0. Headline

N5's verdicts are **over-dispersion** on tier-1 (100 % / 0 %, additivity exact) and **not
decomposable** on `deep_starmap` — where `theta` and `pi` each cost about half the retention and
neither dominates. M3's floor is therefore **2.599**, binding on 50 % of pairs by construction.

But the arms now bound something M3's criteria do not encode: **removing the emission noise from the
model's own mean field would overshoot the tissue on both datasets** — a predicted +0.78 against
tier-1's +0.4635 and +0.41 against deep's +0.3123 — while giving the model a *correct* latent with
the current emission **undershoots** it (+0.4053 and +0.0426). **Two defects that partly cancel.**
M3's outcome 4 fires on "`I` rose", which on tier-1 means moving *further from* the tissue. That is a
defect in a gate I wrote, and the numbers that expose it were on the table when I wrote it.

---

## 1. M1 is verified in production

Both reports now read `text channel | live, text_emb_mode=medcpt, 28/28 gene rows non-zero — from
runs/chain/shipped_tier1.pt`, and the console no longer contradicts it. The vacuous density and
vacuous panel lines fire correctly on tier-1 (`requested 4187, produced 4073, NOT density-matched`;
`--top-k 32 >= the panel's 28 genes, no selection took place`). §4.2k's instances are closed and the
fixes work on real data.

---

## 2. N5, as pre-registered

### Tier-1 — **over-dispersion**, and it is exact

| | |
|---|---|
| `L_total` | +0.3709 |
| `L_theta` | +0.3709 (**100.0 %**) |
| `L_pi` | +0.0000 (**0.0 %**) |
| additivity gap | 0.0000 |
| multiplicative prediction of `I(A1b)` | +0.5405 against +0.5405 measured |
| `A1b-t` level self-check | 1.00x ✅ |

`A1b-t` and `A1b` are identical to four decimals **and across all three seeds** — zeroing `pi` changes
nothing because `pi` is already ~0 there, which the level ratio of 1.00x independently says.
`A1b-p` and `A1c` likewise. **On tier-1 the emission's entire cost is over-dispersion**, on a
converged fit. M3's `theta` floor is aimed at exactly the right term.

### `deep_starmap` — **not decomposable**, and the practical reading is sharper than the verdict

| | |
|---|---|
| `L_total` | +0.3284 |
| `L_theta` | +0.2289 (**69.7 %**) |
| `L_pi` | +0.2095 (**63.8 %**) |
| additivity gap | **0.1100** against a 0.0200 criterion |
| multiplicative prediction | +0.1580 against +0.1722 measured, gap **0.0142** |

Not decomposable is the correct verdict: `L_theta/L_total` misses the 0.70 clause by 0.003 *and*
`L_pi/L_total` is 0.638, far past the 0.30 clause it would also need. Neither term is a passenger.

**What the arms actually say**, as retention against `A1c`:

| | tier-1 | `deep_starmap` |
|---|---|---|
| `theta` alone costs | **40.7 %** | **45.7 %** |
| `pi` alone costs | 0.0 % | **41.9 %** |
| both together cost | 40.7 % | **65.6 %** |

**Each of `theta` and `pi` costs about half the retention on deep, and neither dominates.**

🔎 **The additivity criterion was placed in the wrong space, and that is worth recording.** Losses in
`I` cannot be additive: `I` is a bounded ratio, and by the law of total variance two independent
noise sources add *variances*, so retentions compose closer to multiplicatively. The multiplicative
prediction lands within **0.0142** of the measured `I(A1b)` — inside the 0.0200 tolerance the
additivity clause was given — and it under-predicts, which is the direction the interaction term
requires. So the honest statement is: **the loss does not decompose additively in `I`, and the two
terms are close to multiplicative in retention.** The verdict stands as written; the criterion would
have been better stated on retention.

---

## 3. What the arms now bound — every configuration that has been measured

All on `deep_starmap`, at the real cells, same panel and graph. `mu_oracle` is the tissue's own kNN
mean field; `mu_gen` is the model's.

| configuration | `I(counts)` | vs tissue +0.3123 |
|---|---|---|
| correct latent + **current emission** (`A1a`) | **+0.0426** | 0.14x — **worse than the model's own** |
| current latent + current emission (stage 4) | +0.1154 | 0.37x |
| `mu_oracle` + current emission (`A1b`) | +0.1722 | 0.55x |
| `mu_oracle`, `theta` removed (`A1b-t`) | +0.2718 | 0.87x |
| `mu_oracle`, `pi` removed (`A1b-p`) | +0.2911 | 0.93x |
| `mu_oracle`, all emission noise removed (`A1c`) | +0.5007 | **1.60x — overshoots** |
| **`mu_gen`, all emission noise removed** | **not measured** | predicted **+0.41, 1.32x** |

Two things follow, and they are the reason §4 exists:

1. **Removing either `theta` or `pi` alone, on a good mean field, reaches ~90 % of the tissue.**
   Removing both overshoots by 1.60x.
2. **Giving the model the truth as a latent makes it worse** (+0.0426 against its own +0.1154),
   because `decode(h1)` has `I` = +0.3399 against `decode(h)`'s +0.7451 — the model's latent is
   **2.45x smoother** than the tissue's, and the emission noise is partly cancelling that.

---

## 4. 🚩 A defect in M3's criteria — mine, caught before M3 runs

`m3_verdict`'s outcome 4 requires `d I(counts) > 0`, and outcome 2 fires on `d I <= 0`. **On tier-1
the model is already *above* the tissue** (+0.5134 against +0.4635), so `I` rising is movement
**away** from the tissue, and M3 would report REALLOCATION WORKS for it.

The predicted ceiling makes it concrete. Removing `theta` entirely from the model's own mean field
gives ~**+0.78** on tier-1 — 1.68x the tissue. A partial floor lands somewhere between +0.51 and
+0.78, and **every point of that range increases the distance to the tissue**:

| | baseline | `theta` removed (predicted) | tissue |
|---|---|---|---|
| tier-1 `I` | +0.5134 | +0.7775 | +0.4635 |
| distance | 0.0499 | 0.3140 | — |

This is the over-smoothing correction — already in the record since round 11, and the reason M4 is a
condition — applied to my own gate, which I failed to do. **The numbers were all available when I
wrote the criterion.** It is not a reaction to M3's result; M3 has not run.

### 4a. The fix: separate *mechanism* from *benefit*, and let tier-1 test only the first

Tier-1 has a converged fit and no deficit; `deep_starmap` has the deficit and a fit whose alarm fired
at its last step. Neither can carry both questions.

**M3 on tier-1 is a mechanism test.** Its success criterion should be the one thing tier-1 can
answer: does flooring `theta` move variance into `mu`?

* **works** — `sd(log mu)` rises **toward the tissue's bracket [0.7165, 0.9438]** from the baseline's
  0.6728, without overshooting 0.9438;
* **capacity-limited** — `sd(log mu)` does not move (unchanged as a result, per §3a of the
  pre-registration);
* **do-no-harm** — no pinned `paper_*` metric falls by more than 0.02, unchanged;
* **and `I` is reported but is NOT a success criterion on tier-1**, because the direction that would
  count as success there is *down*, toward +0.4635, and the mechanism predicts *up*.

**Benefit is a separate question and it needs `deep_starmap`** — which needs a converged fit, which
is B1. That changes B1's priority (§5).

On `deep_starmap` the distance framing works and `I` rising *is* the goal: baseline distance 0.1969,
and any move toward +0.3123 is a gain until it passes it.

**Proposed replacement statistic, for both datasets: `d |I(counts) - I(real counts)|`.** It is
signed correctly on both, it needs nothing new measured, and it encodes the goal — *be like the
tissue* — rather than a proxy for it.

---

## 5. The picture that is emerging: two defects that partly cancel

| | tier-1 | `deep_starmap` |
|---|---|---|
| latent too smooth | 1.28x the tissue's | **2.45x** |
| `mu` too narrow | 0.6728 vs bracket [0.7165, 0.9438] | 0.6725 vs **[1.0994, 1.3699]** |
| emission adds too much unstructured noise | `theta`, 40.7 % of retention | `theta` **and** `pi`, 65.6 % |
| **net `I`** | +0.5134 vs +0.4635 — **1.11x over** | +0.1154 vs +0.3123 — **0.37x under** |

Fix the emission alone and the model overshoots (predicted 1.68x / 1.32x). Fix the latent alone and
it undershoots (`A1a`: +0.4053 / +0.0426). **On tier-1 the two errors happen to cancel to within
11 %, which is why tier-1 looked healthy for three rounds.**

🚩 **This corrects my own assessment of B1, and the correction is in the user's favour.** I said
*"`ell` fixes the latent and the latent is not the bottleneck"*. That was drawn from `decode(h1)`
producing **less** `mu` spread than `decode(h)` — true, and it is about `mu`'s **spread**. The
latent's **smoothness** is a different quantity and it is wrong by 2.45x on deep. `ell` is a
plausible cause of that and B1 is the test. **B1 is not a 3.3 h purchase of trustworthy `theta`/`pi`;
it is the only handle anyone has on the second of the two coupled defects.** I under-costed its
value by reading one measurement as if it covered both.

---

## 6. The floor, and a dose problem

The baseline's `theta` at the real cells:

| percentile | 1 | 5 | 10 | 25 | 50 | 75 | 90 | 99 |
|---|---|---|---|---|---|---|---|---|
| `theta` | 0.3689 | 0.7087 | 0.8834 | 1.394 | **2.599** | 5.165 | 9.343 | 13.65 |

**The pre-registered floor is the median: 2.599**, and it binds on **50 %** of `(cell, gene)` pairs by
construction — so the null-experiment guard (< 5 %) passes without a further read.

⚠️ **But it is a modest dose.** The floored half has a median `theta` of 1.394 (the 25th percentile),
so it moves up by about 1.9x and its dispersion falls by ~47 %; the other half is untouched. If
`sd(log mu)` does not move at that dose, **"capacity-limited" is not yet the only explanation — "the
dose was too small" is the other**, and outcome 1 is a strong claim that should not be reachable
from one weak intervention.

**Proposed, to be pre-registered before M3 runs:** if the median floor moves `sd(log mu)` by less
than 0.05, run **one** higher floor at the **75th percentile (5.165)** before outcome 1 may be
declared. One extra hour, and it is what makes outcome 1 a result rather than a shrug.

---

## 7. The measurement that is missing, and it is one draw

**`I(counts ~ Poisson(mu_gen))`** — the model's own mean field with **all** emission noise removed.
It is the ceiling for any emission repair, on the model as it actually is, and nothing in the
campaign has it. Every arm so far is either on `mu_oracle` (the tissue's field) or keeps the
emission.

**Pre-registered prediction, before it is measured:** transferring `A1c`'s Poisson retention gives
**+0.7775** on tier-1 and **+0.4116** on `deep_starmap`, i.e. **1.68x and 1.32x the tissue**. If it
comes back at or above those, **no emission repair can reach the tissue without the latent being
fixed too**, and §5's coupled reading is established rather than inferred. If it comes back at or
below the tissue on deep, the transfer is invalid and the emission repair stands on its own.

It costs one draw per dataset inside the existing ablation. It should be added before M3's floored
fit, because it tells you what M3 is aiming at.

---

## 8. What to do, in order

| # | action | cost | what it settles |
|---|---|---|---|
| **P1** | Add the `Poisson(mu_gen)` arm to the ablation and re-run both A1 reads | minutes, no fit | §7 — the ceiling M3 is aiming at, and §5's coupling |
| **P2** | **Amend M3's pre-registration before it runs**: mechanism-only on tier-1, `d abs(I - I_real)` as the benefit statistic, and §6's dose escalation | writing only | §4 — a gate that encodes the goal |
| **P3** | Run M3 at floor **2.599** as the mechanism test | 1 fit, ~1 h | does flooring `theta` move `sd(log mu)` |
| **P4** | If §6's condition fires, M3 at floor **5.165** | 1 fit, ~1 h | separates capacity-limited from under-dosed |
| **P5** | **B1** — `ell` refit on `deep_starmap`, alarms escalated to a hard stop | ~3.3 h | the second defect, and a deep fit whose `theta`/`pi` can be trusted |

**P1 and P2 come before P3** — both are cheap, and both change what P3 means. **P5 is now worth more
than I said**, per §5.

**§10 stays suspended.** N5 resolved the mechanism question it was asked; it did not return a
readable answer to the decision table, which still reads row 5 for `A1b`.

---

## 9. Where I was wrong

* **M3's success criterion.** `d I > 0` is not the goal on a dataset where the model is already above
  the tissue. My own over-smoothing correction says so and I did not apply it to my own gate. The
  numbers were available when I wrote it.
* **N5's additivity clause.** Placed on losses in `I`, which is a bounded ratio and cannot be
  additive under independent noise. Retention composes multiplicatively and the data show it to
  within 0.0142. The verdict is unaffected; the criterion was the wrong shape.
* **B1's value.** I read *"`decode(h1)` gives less `mu` spread than `decode(h)`"* — a statement about
  `mu`'s **spread** — as settling that the latent is not the bottleneck. The latent's **smoothness**
  is a different quantity, wrong by 2.45x on deep, and `ell` is the only handle on it. Corrected in
  §5, and it raises B1's priority rather than lowering it.
