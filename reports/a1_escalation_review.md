# N1–N4 read out, and the shortest path to a defensible paper

Sources: `reports/a1_tier1.md`, `reports/a1_deep.md` (three seeds, interior `section_4`, plane at the
section's own z), and both run logs. Criteria: `a1_preregistration.md`,
`a1_escalation_preregistration.md`. **No code was changed for this report.**

---

## 0. Headline

The mechanism is now **located and measured**, and it is not where the last three rounds put it.
The decoder's `mu` head produces `sd(log mu) = 0.6725` on `deep_starmap` and **0.6728** on tier-1 —
two datasets, 1017 genes against 28, agreeing to three decimals — while the tissue needs **0.72–0.94**
(tier-1) and **1.10–1.37** (deep). The missing spread is absorbed by `theta`/`pi`, so the model's
**total** count variance is right (implied 1.4131 against the tissue's 1.3699 on deep) and only
**~10 %** of it is structured against the tissue's **>= 42 %**. Two consequences: **R12 is
corroborated, not refuted, and my refutation of it was a name collision I have to withdraw**; and
§10's two halves are not independent — the variance is already there, in the wrong place, so the fix
is a **reallocation**, which makes it a much sharper and cheaper test than it was costed as.

A1b came back **UNRESOLVED** exactly as pre-registered, so the decision table still returns row 5 and
N5 is the escalation. **§10 stays suspended.**

---

## 1. 🚩 A defect first: the console and the report disagree about the same run

Both logs, line 19:

```
text channel: ZERO VECTORS (cfg.text_emb_mode='medcpt' is NOT exercised here; ...)
```

Both reports, same run:

```
text channel | live, `text_emb_mode=medcpt`, 28/28 gene rows non-zero — from runs/chain/shipped_tier1.pt
```

The report is right. The console line is printed inside `build_embeddings` at **construction time**,
before `load_state_dict` restores the `text_vecs` buffer, so it is describing a state that exists for
microseconds and never reaches the model. **N1 fixed the report and left the console.** Two artifacts
of one run asserting opposite arms is §4.2k again, and it is the fifth instance, all mine. The fix is
one line: under `--load-model`, that print must say the vectors will be replaced from the checkpoint,
or not claim an arm at all.

Nothing else in either run is affected — tier-1's chain reproduces bitwise for the third time
(+0.9362 / +0.8011 / +0.7920 / +0.5134), across a code change, which is a real determinism check.

---

## 2. The pre-registered verdicts, applied as written

**Tier-1 control: PASSES at three seeds.** Every drawn arm clears `0.6 x I_real` = +0.2781 —
A1a +0.4006..+0.4103, A1b +0.5390..+0.5452, A1c +0.9112..+0.9114 — levels 1.00–1.01x, null +0.0025.
The deep rows are readable.

**A1b: UNRESOLVED.** Per-seed `R` = +0.29, +0.33, +0.28 → bands DOES NOT RECOVER, **UNINFORMATIVE**,
DOES NOT RECOVER. The stability override fires. `a1_escalation_preregistration.md` §4 predicted this
as the likely outcome and it is why N5 was pre-registered in advance. The median (+0.29) and two of
three seeds sit in DOES NOT RECOVER, so the **direction** is not in doubt — but the verdict is
UNRESOLVED and it is not to be reported as "does not recover".

**A1a: DOES NOT RECOVER**, stable (−0.37, −0.39, −0.37). **A1c: RECOVERS**, stable (+1.95, +1.96,
+1.97). Row 1 stays refuted: this is not the sparsity bound.

**Decision table: row 5 again.** Rows 2 and 3 both require A1b to be in a definite band. **No
conclusion; N5 is the escalation.**

**N2 — §6's narrow-`mu` mechanism:**

| | decoder `sd(log mu)` | tissue lower bd | tissue upper bd | verdict |
|---|---|---|---|---|
| tier-1 | 0.7269 | 0.7165 (**0.99x**) | 0.9438 (1.30x) | **UNTESTED STILL** |
| `deep_starmap` | 0.7118 | **1.0994 (1.54x)** | 1.3699 (1.93x) | **SUPPORTED** |

The deep pass clears the 1.5x line by 3 %, which would normally be the kind of margin this project
withdraws verdicts over. It is not, for a reason that is structural rather than convenient: **N2's
numbers carry no draw randomness at all** — `mu_oracle` is a deterministic kNN mean and the real
counts are fixed, so unlike A1b there is no seed noise to straddle a boundary. The only uncertainty
is estimator bias, whose direction is known and downward, and the criterion was placed on the
**pessimistic** end of the bracket by design. The whole bracket, 1.54x–1.93x, clears the line.

**And the estimator is verified on the real data, not just the fixture.** The A1c check row
recovers `mu_oracle`'s own spread from Poisson draws of it: **0.7162 vs 0.7165** (tier-1) and
**1.1005 vs 1.0994** (deep). Three to four decimal places, on the actual panels.

My §6 prediction was directionally right and **quantitatively too large**: I predicted the tissue at
`>= 2` against the decoder's ~0.72, i.e. 3–5x. It is 1.5–1.9x. Recorded as a miss.

---

## 3. 🚨 R12 is corroborated, and my refutation was a name collision — withdrawn

Last round I wrote *"R12 is refuted on the shipped arm"* in `chain_shipped_review.md` §5 and
`a1_review.md` §0, on the strength of the bounded share `Var(shape)/(Var(shape)+Var(log s))` coming
back at 97.4 % / 93.3 % instead of 15.3 % / 62.2 %.

**That statistic answers a different question.** It decomposes `Var(log mu)` into its *latent-driven*
and *size-factor* parts — "is `mu`'s dynamic range the size factor?" — and never touches the sampling
noise. R12's claim, in `progress/risks.md`, is about counts:

> *"only 9–19 % of the emitted count variance survives as between-cell structure … the term that
> eats it is overdispersion"*

That is `Var(mu) / Var(y)`, and N2 is the first thing in the campaign that can compute it model-free.
Derived from the reported numbers (`CV² = exp(sd²) − 1`; **derived, not pre-registered**, and it
stacks the lognormal approximation twice):

| count-level structured share | model | tissue (lower bound) |
|---|---|---|
| tier-1 | 60.7 % | 46.7 % |
| **`deep_starmap`** | **10.4 %** | **>= 42.5 %** |

**10.4 % lands inside R12's 9–19 %**, on the shipped `exp` arm, at matched panel and density, with an
estimator that does not pass through the decoder. R12 stands. What I refuted was a differently-defined
quantity that shares its name because `scripts/t09_structured_share.py` implements the decoder-internal
one.

This is the **second** time in three rounds I have read a statistic by its name rather than its
definition — the first was gate 2, whose two sides shared a decoder. Both belong under `specs/10`
§4.2's closing rule, and I would add the sharper form: **a statistic's name is not its definition,
and a refutation must restate the claim in the claim's own terms before it counts.**

---

## 4. What is now established, and it is one mechanism

**The decoder's `mu` head is pinned, and it is the bottleneck.**

| `sd(log mu)` | tier-1 | `deep_starmap` |
|---|---|---|
| decoded from the **generated** latent | **0.6728** | **0.6725** |
| decoded from the **encoder's** latent `h1` | 0.7225 | **0.6035** |
| what the tissue needs (bracket) | 0.72 – 0.94 | **1.10 – 1.37** |

Two things follow that nothing before could show:

1. **It is not the latent.** On deep, `h1` is a *faithful* latent — `I(h1)` = +0.3171 against the real
   counts' +0.3123 — and decoding it gives **less** spread (0.6035) than decoding the model's own
   over-smooth latent (0.6725). Feeding the decoder the truth does not widen `mu`. A1a's stable
   DOES NOT RECOVER is the same fact seen from the counts side.
2. **It is not a shortage of variance.** The model's implied total is 1.4131 against the tissue's
   1.3699 on deep — slightly **more**. The variance exists; ~90 % of it is in `theta`/`pi`, which is
   spatially unstructured and therefore dilutes `I`, instead of in `mu`, which is structured.

The emission retention table says the same thing from the third side:

> ⚠️ **OVERSTATED — see `reports/ceiling_review.md` §2.** Every ratio in this table divides a
> **rank-normalised** count stage by a **raw** mean-field stage. Rank-normalising a heavy-tailed
> field raises its Moran's I, so the denominators are too small and these percentages are too high
> by an amount nobody has measured. The **ordering** and the **cross-dataset contrast** survive —
> both sides of every comparison carry the same bias — but the percentages themselves do not. The
> arm-vs-arm conclusions elsewhere in this report are count-vs-count and are unaffected.

| step | tier-1 | `deep_starmap` |
|---|---|---|
| `mu_oracle` → Poisson (sparsity alone) | 98.2 % | 55.2 % |
| Poisson → + model `theta`/`pi` | **59.3 %** | **34.4 %** |
| net, `mu_oracle` → model emission | 58.2 % | 19.0 % |

Sparsity is real (45 % on deep, 2 % on tier-1) and is **not** the binding constraint; `theta`/`pi`
cost more, on both datasets, including the one with no deficit and a converged fit. The level column
splits them: A1b at **0.64x** on deep means `pi` deletes ~36 % of transcripts there; at **1.00x** on
tier-1 `pi` is ~0 and the loss is over-dispersion — which is exactly what R12 attributed it to.

**Why the objective allows it:** the ZINB likelihood is close to indifferent to *where* between-cell
variance sits at the margin. A narrow `mu` with a wide `theta` fits the marginal counts about as well
as a wide `mu` with a tight `theta`, and only the second carries spatial structure. Nothing in the
loss prefers the second. That is R4's trade, and it is now measured rather than inferred.

**One consequence for §10, and it lowers its cost.** Its two halves — *bound dispersion* and *raise
`Var(mu)`* — were costed as independent, with the second doing most of the work at low certainty.
The total variance is already correct, so they are **one constrained move**: bound `theta`, and the
likelihood has to put the variance somewhere, and `mu` is the only place left. **Half 1 may deliver
half 2 for free**, and whether it does is a one-hour experiment rather than a 3–4 day redesign.

---

## 5. The deficit keeps shrinking as confounds are removed

| measurement | model | tissue | ratio |
|---|---|---|---|
| `section_2`, z = 30.8 (boundary-adjacent, wrong plane) | +0.0729 | +0.3236 | 4.44x |
| `section_4`, z = 68.6 | +0.1021 | +0.3123 | 3.06x |
| **`section_4`, z = 73.5 (the section's own plane)** | **+0.1154** | **+0.3123** | **2.71x** |

Each provenance fix has moved it toward the tissue, and none of the three is a modelling change.
**2.71x is the current honest figure** and it is the one to quote — with the caveat that the trend
means the remaining number may still carry confounds nobody has found, and that the underlying fit
is the one whose spatial alarm fired at its last step.

---

## 6. What to do, in order, to reach a defensible paper

The goal is a v25 method paper. Two routes: **repair the emission and re-benchmark**, or **publish
the mechanism as the finding**. The cheapest useful thing is the experiment that decides between
them, and it now exists.

### 6.1 Free, and it must land before either report is cited

**M1.** Fix the console line (§1). Under `--load-model`, `build_embeddings` must not claim an arm it
is about to have overwritten.

### 6.2 The pre-registered escalation — run it (minutes, no refit)

**M2 — N5.** The `theta`/`pi` split, already pre-registered in
`a1_escalation_preregistration.md` §2, criteria and all. It resolves A1b and names the term. R12
predicts over-dispersion; the level column predicts `pi` matters on deep and not on tier-1; **those
two predictions can both be checked against one run**, which is worth more than either.

### 6.3 The experiment that decides the route (~1 h, one fit) — **needs a new pre-registration first**

**M3 — the reallocation test.** Refit **tier-1** with a floor on `theta` (equivalently a ceiling on
dispersion), everything else identical, and measure whether the variance moves into `mu`.

Tier-1 because its fit converges, it costs an hour, and the checkpoint to compare against exists. It
is the *insensitive* case — the decoder is already inside tier-1's bracket at 0.99x — so a clear
movement there is strong evidence, and no movement there is not yet fatal.

To be pre-registered before it runs, sketched here, not fixed:

* **works** — `sd(log mu)` rises, `I(counts)` holds or rises, the count-level structured share rises,
  and the NLL degrades by less than a stated bound;
* **the mu head is capacity-limited, not objective-limited** — `sd(log mu)` does not move. Then no
  loss-side fix will work and the finding is *stronger*, not weaker: the objective is not the
  constraint, the head is;
* **manufacturing unconditioned variance** — `sd(log mu)` rises but `I(counts)` does not. A9's
  failure in a new place, and the reason the criterion cannot be `sd(log mu)` alone.

**This is §10 half 1, and running it does not lift the suspension** — §10 stays suspended until a
readable gate says otherwise. M3 *is* a candidate for that gate, and it should be written as one.

### 6.4 The metric gap that must close before any repair is judged

**M4.** Every number in this campaign is Moran's `I`, and the over-smoothing correction proved `I`
can be raised by making the model worse: the model beats the tissue on tier-1 (+0.5134 vs +0.4635)
with a latent 1.28x too smooth. **Any repair must be judged on `I` *and* on the pinned `bench3`
`paper_*` metrics**, pre-registered together. Without that, a repair that raises `I` proves nothing.

### 6.5 B1, with its expected value revised down

> ### 🔄 REVERSED 2026-09-09 — B1's value was under-read, and the reading conflated two quantities
>
> **What I read.** A1 showed `decode(h1)`, the decoder applied to a *faithful* latent, producing
> **less** `mu` spread (0.6035) than `decode(h)` on the model's own over-smooth latent (0.6725). I
> took that as settling that the latent is not the bottleneck, and revised B1 down to *"it buys
> trustworthy deep `theta`/`pi` and nothing else."*
>
> **What that measurement actually covers.** `mu`'s **spread** — `sd(log mu)`, how far apart the
> per-cell means are. It is a statement about the decoder's output range, and it is still true: the
> `mu` head is not spread-limited by its latent.
>
> **What I now read.** The latent's **smoothness** is a different quantity: `I(h)` = +0.7782 against
> the tissue's `I(h1)` = +0.3171, so the model's latent is **2.45x more spatially autocorrelated
> than the tissue's** on `deep_starmap` (1.28x on tier-1). Nothing in the spread measurement bears
> on it. `ell` — fitted on tier-1 and applied unchanged to `deep_starmap` — is a plausible cause,
> and B1 is the only test of it.
>
> **The correction.** B1 is not a 3.3 h purchase of trustworthy `theta`/`pi`. It is **the only
> handle anyone has on the second of two coupled defects** (§4a below), and its priority goes **up**,
> not down. The error was conflating spread with smoothness — two properties of the same object,
> measured by different statistics, and one measurement was read as if it covered both.

**M5 — B1** (`ell` refit on `deep_starmap`, ~3.3 h). Still needed: the deep checkpoint's spatial alarm
fired at its last step, so every `theta`/`pi` conclusion on that dataset is provisional. But §4 has
changed what it can be expected to do — **`ell` fixes the latent, and the latent is not the
bottleneck**. A1a shows a *faithful* latent produces a narrower `mu`, not a wider one. Run B1 to make
the deep numbers trustworthy; **do not expect it to close the 2.71x**, and say so in advance so the
result is not read as a failure of the refit.

**Order: M1, M2, M3 (pre-registered first), then M4 alongside, then M5.** M2 and M3 together cost
about an hour and decide whether the paper is a repair or a finding.

---

## 7. What is publishable if the repair fails

Worth stating now, because it is already substantial and it changes how much risk M3 carries:

1. A stage-by-stage chain diagnostic that **localises** the loss to one step, with a model-free
   floor (Poisson on the tissue's own mean field) proving the step is not information-theoretically
   forced.
2. **The mechanism**: the ZINB objective is indifferent to where between-cell variance lives, so it
   is absorbed by dispersion rather than by the structured mean — the model's total count variance
   is *correct* and ~10 % of it is structured against the tissue's >= 42 %. R12, now on a model-free
   estimator.
3. **A decoder `mu` head pinned at `sd(log mu)` ≈ 0.67 across two datasets, 28 genes and 1017,
   and across both a generated and a faithful latent.** A capacity result about the architecture,
   not a tuning observation.
4. **The over-smoothing correction**: a generative model can beat real tissue on Moran's `I` by
   being smoother than the tissue, so `I` alone cannot certify a spatial generative model. That is a
   methodological result other people's benchmarks need.

That is a paper. The repair would make it a better one; it is not what makes it publishable.

---

## 8. Where I was wrong, again

**R12.** I called it refuted on a statistic that answers a different question and shares its name
(§3). Withdrawn. Second instance in three rounds of reading a statistic by its name — gate 2 was the
first. The rule I would add: *a refutation must restate the claim in the claim's own terms before it
counts.*

**§6's magnitude.** Predicted the tissue's `sd(log mu)` at `>= 2`, i.e. 3–5x the decoder's. Measured
1.5–1.9x. The direction and the pre-registered criterion hold on deep; the number I guessed did not.

**N1.** Fixed the report line and left the console line, so the same run now publishes two contrary
claims about its own arm (§1).
