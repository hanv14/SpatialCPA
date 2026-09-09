# A1 read against its pre-registration, and what follows

Sources: `reports/a1_tier1.md`, `reports/a1_deep.md`. Criteria: `reports/a1_preregistration.md`,
committed before the runs. **No code was changed for this report.**

---

## 0. Headline

The control passed. **Row 1 of the decision table is refuted — this is not the sparsity bound**, so
the branch that would have ended §10 outright is closed by the model-free arm. But **A1b landed in
the dead band at `R = +0.33`**, so the pre-registered verdict on `deep_starmap` is **NO CONCLUSION**,
and I am not rounding it. Two things the table does not gate came out anyway, and both are new: the
model's fitted `theta`/`pi` are **the largest single loss in the emission on both datasets, including
the one with no deficit**; and giving the model the truth as a latent makes its counts **worse**,
which means its Moran's I advantage where it has one comes from a latent smoother than the tissue's
rather than from reconstructing the tissue.

---

## 1. The load is exact — and two provenance lines are wrong, both mine

All six chain stages in `a1_tier1.md` reproduce the fitted step-0 run **bitwise** (+0.9362 / +0.8011
/ +0.7920 / +0.5134 / +0.4635 / +0.6253). `--load-model` works: the strict `load_state_dict`
restored the fitted weights *and* the MedCPT `text_vecs` buffer.

Which is how we know two lines in both reports are false:

| line | says | is |
|---|---|---|
| title | `(1200 steps)` | **2400** — it prints `args.steps`, not the checkpoint's `train_steps` |
| text channel | `**zero vectors** (neither A3 arm)` | **live MedCPT** — restored from the checkpoint buffer; the line reports how the module was *constructed*, not what it holds |

Neither changes a number. Both are **§4.2k — a report describing an operation that did not happen** —
and I introduced them in the same commit that recorded §4.2k. `--load-model` must take every
provenance line from the checkpoint's config, and the text-channel line must read *"from checkpoint
(`text_emb_mode=medcpt`)"*. The tier-1 density line is also still vacuous (4 073 produced against a
target of 4 187, no subsampling, reported as "matched"). **These reports should not be cited until
the labels are fixed**, because a reader who takes the header at face value has a 1200-step
zero-text run in hand.

---

## 2. The control passed

Pre-registered (§5): on tier-1 every drawn arm must reach `0.6 x I_real` = **+0.2781**.

| arm | I | vs +0.2781 | level |
|---|---|---|---|
| A1a | +0.4053 | ✅ | 1.01x |
| A1b | +0.5452 | ✅ | 1.00x |
| A1c | +0.9114 | ✅ | 1.00x |
| A1n (null) | +0.0025 | — | — |

All three clear it, all three levels sit inside the [0.5, 2.0] guard, and the permutation null is
**+0.0025** — the estimator's zero is clean on this panel and this graph. **The `deep_starmap` rows
may be read.**

---

## 3. Row 1 is refuted: this is not the sparsity bound

`A1c` — Poisson on the tissue's own kNN mean field, **model-free**, admissible regardless of the
deep fit's collapse — reaches **+0.4994** on `deep_starmap`, against the real section's **+0.3123**
and the model's **+0.1021**. `R = +1.89`.

**Counting statistics at `deep_starmap`'s sparsity do not bound Moran's I to 0.10.** The decision
table's first row — *no independent per-cell draw can reach the tissue, §10 dies in both halves,
publish the characterisation* — **does not apply**. §9's "dataset property, not a defect" branch is
closed on its own pre-registered terms.

The caveat I fixed in advance holds and I am not dropping it: `mu_oracle` is a kNN mean, smoothing
*creates* autocorrelation (the self-check measures I 0.5494 -> 0.9704 on a synthetic field), so
**A1c high is permissive, not proof**. It shows the draw *can* carry structure at this sparsity given
a good enough mean field. It does not make +0.4994 a target the model must reach.

---

## 4. The pre-registered verdict on `deep_starmap`: NO CONCLUSION

| arm | I | R | band |
|---|---|---|---|
| A1c | +0.4994 | **+1.89** | RECOVERS |
| A1b | +0.1722 | **+0.33** | **UNINFORMATIVE** (0.30 < R < 0.70) |
| A1a | +0.0433 | **−0.28** | DOES NOT RECOVER |

Rows 1–4 of the table need A1b to be either RECOVERS or DOES NOT RECOVER. It is neither, so **row 5
applies: no conclusion, escalate A1b to three seeds.**

**I am not reclassifying it.** A1b sits **0.0335 of `R` above the lower edge** — in `I`, **0.0070**.
Reading it down to DOES NOT RECOVER would name the mechanism (`theta`/`pi` are the fault) on a margin
of seven thousandths, decided after the shape of the data was known. That is the exact move the dead
band was put there to prevent, and this project has already withdrawn two verdicts decided on
narrower margins.

It also means the escalation is real work, not a formality: 0.0070 of `I` is plausibly inside
draw-to-draw variation, so three seeds may well leave it in the band. §7 proposes the escalation that
can actually resolve it.

---

## 5. What the arm-to-arm contrasts establish, with no threshold involved

Retention at a **fixed** mean field — each arm divided by the mean field it was drawn from. No bands,
no anchors, no `R`; these are ratios inside one report.

| step | tier-1 | `deep_starmap` |
|---|---|---|
| `mu_oracle` -> Poisson (A1c / A1b') | **98.2 %** | **55.1 %** |
| Poisson -> + model `theta`/`pi` (A1b / A1c) | **59.8 %** | **34.5 %** |
| net: `mu_oracle` -> model emission (A1b / A1b') | 58.7 % | 19.0 % |
| for comparison, the model end to end (stage 4 / stage 3) | 64.8 % | 13.7 % |

> ⚠️ **The `mu_oracle`-denominated rows are OVERSTATED — `reports/ceiling_review.md` §2.** They
> divide a **rank-normalised** count stage by a **raw** mean-field stage. Rank-normalising a
> heavy-tailed field raises its Moran's I, so the denominator is too small and the percentage is too
> high by an unmeasured amount. The **ordering** and the **cross-dataset contrast** survive, since
> both sides carry the same bias; the percentages do not. The `A1b / A1c` row is count-vs-count and
> is unaffected.


Two multiplicative causes:

1. **Sparsity is real and large on `deep_starmap`** — a bare Poisson draw costs **45 %** of the mean
   field's `I` there, against **2 %** on tier-1. The mechanism §6 of the review proposed is present.
   It is just not sufficient: 55 % of +0.9063 is +0.4994, still well above the tissue.
2. **The fitted `theta`/`pi` cost more than sparsity does — on both datasets.** They multiply
   retention by **0.345** on deep and **0.598** on tier-1. On tier-1 that is a **40 % loss of
   achievable `I` on the dataset with no deficit and a healthy, converged fit.**

That second row is the finding I did not expect. It is measured on a converged fit, it does not
depend on `deep_starmap`, and nothing in the campaign had isolated it — every previous number was
end-to-end, where the over-smooth latent (§6 below) hides it.

**And the level column carries a mechanism.** `A1b` differs from `A1c` only in using the model's
`theta` and `pi`; `theta` does not change the mean, so a level ratio is a direct read on `pi`:

* `deep_starmap`: A1b at **0.64x** — the model's zero-inflation is deleting **~36 %** of transcripts.
* tier-1: A1b at **1.00x** — `pi` is essentially zero there, and the 40 % loss above is **over-dispersion**.

So the two datasets lose `I` to `theta`/`pi` for **different reasons**, and the guard I added to
catch an incomparable arm turned out to measure the thing.

---

## 6. A1a is worse than the model end to end, and that reframes both datasets

`R(A1a) = −0.28`: feeding the model the **truth** as a latent produces counts (+0.0433) *less*
spatially structured than feeding it its own generated latent (+0.1021). The cause is visible one
row up, and it is the same on both datasets:

| | tier-1 | `deep_starmap` |
|---|---|---|
| generated latent `h` | +0.8011 | +0.7786 |
| encoder's latent of real counts `h1` | +0.6253 | **+0.3171** |
| `mu` decoded from `h` | +0.7920 | +0.7469 |
| `mu` decoded from `h1` | +0.6052 | **+0.3399** |

**The model's latent is smoother than the tissue's — 1.28x on tier-1, 2.46x on `deep_starmap`** — and
that extra smoothness survives the draw better than the tissue's own structure does.

Two consequences:

* **Tier-1's headline needs restating.** "The model emits counts at 1.11x the real section's Moran's
  I" is true and is not a reconstruction claim: on the same emission, the tissue's own latent yields
  +0.4053 and the model's yields +0.5134. The gap is over-smoothing, not fidelity. The paper should
  say so — a reader will otherwise take 1.11x as the model beating the tissue at its own task.
* **Repairing the emission would not be uniformly good.** It raises `I` on `deep_starmap`; on tier-1
  it would raise the model's `I` further above the tissue's, i.e. make the over-smoothing more
  visible, not less. Any emission change must be read against **both** datasets and against a
  fidelity metric, not against `I` alone.

---

## 7. §6's narrow-`mu` mechanism is **not** supported, and the one number that would settle it

My review §6 proposed that the model's `mu` is too narrow relative to the tissue's, and predicted a
model-free tissue `sd(log mu)` of ≳ 2 against the model's 0.72. These runs point the other way:

* generated `sd(log mu)` **0.7131** vs the real latent's **0.4797** on deep — the generated is
  **wider**, not narrower;
* `A1a`, whose `mu` comes from the truth and is therefore the more realistic one, does **worse**.

**The prediction is untested, not refuted**, because `sd(log mu_oracle)` was not reported — the
ablation prints `I(mu_oracle)` but not its spread. That is the single number missing from A1 and it
is already computed inside the arm. It is the last piece of §6, and it costs nothing.

---

## 8. The circularity that decides what to run next

A1's sharpest result — `theta`/`pi` are the biggest loss — rests on `theta` and `pi` **from a fit
whose spatial-collapse alarm fired at its final step**, on `deep_starmap`. The pre-registration
flagged A1a as provisional for exactly this reason; A1b inherits a weaker form of it.

The tier-1 half does **not** have this problem. Tier-1's fit is converged, and it shows `theta`
costing 40 % of achievable `I` with `pi` at zero. **That is the uncontaminated version of the
finding**, and it is on the headline dataset.

So `B1` — the `ell` refit on `deep_starmap` — changes status. The decision table did **not** trigger
its row (row 4 needs A1a to recover, and it does not). But B1 is now the only way to obtain
`theta`/`pi` from a **converged** deep fit, and re-running A1 on that checkpoint is the clean test.
**This is my reasoning beyond the table, and I flag it as such**: B1 is motivated, but not by A1's
pre-registered output.

---

## 9. What did not change

**§10 stays SUSPENDED.** The replacement gate returned **NO CONCLUSION** on the dataset it was built
for. Under the standing rule, the suspension lifts only when a readable answer arrives — a favourable
side observation (§5's `theta` finding) is not that, and letting it lift the suspension would be the
reclassification route you ruled out, taken by a different door. If the `theta` finding is to drive
work, it needs **its own** pre-registered gate, proposed in §10.3.

**`+0.0729` is still not quoted.** Note it was also inflated by R3: on the interior `section_4` the
model reads **+0.1021** and the tissue **+0.3123**, so the deficit is **3.06x**, not 4.4x. The
boundary plane was worth about a quarter of the apparent gap — `section_4` was the right call, and
the encoder-latent anomaly cleared with it (real-side retention 134 % -> **98.5 %**).

---

## 10. Proposals

### 10.1 Free, and they unblock everything else

| # | proposal | why |
|---|---|---|
| **N1** | **Fix the two provenance lines** (§1) plus the vacuous-density line: with `--load-model`, every provenance value comes from the checkpoint. | Two §4.2k instances in the commit that added §4.2k. The reports cannot be cited until this is done. |
| **N2** | **Report `sd(log mu_oracle)`** — and the NB moment estimator `CV²_mu = (Var y − ȳ)/ȳ² − 1/θ̂` per gene as an independent route. | §7. Settles §6's narrow-`mu` prediction; already computed inside the arm. |
| **N3** | **Default `--target-z` to the named section's own z.** The deep run generated at z = 68.6 while `section_4` sits at **z = 73.5** — a 4.9 µm offset inside a 20.3 µm gap. | Unforced inconsistency between the generated plane and its reference. |

### 10.2 The pre-registered escalation, and the one that can resolve it

**N4 (pre-registered, do it):** re-run A1b at `--ablation-seed 2` and `3`, both datasets. Draws only,
no refit, minutes. Required by row 5 before the table may be read again.

**N5 (new — must be pre-registered before running):** split A1b, which lumps two mechanisms the level
column already shows are different on the two datasets.

* `A1b-θ`: `NB(mu_oracle, theta_model)`, `pi` forced to 0 — over-dispersion alone.
* `A1b-π`: `Poisson(mu_oracle)` then zeroed with `pi_model` — dropout alone.

Draft criteria, to be committed before the run:

> With `L_total = I(A1c) − I(A1b)`, `L_θ = I(A1c) − I(A1b-θ)`, `L_π = I(A1c) − I(A1b-π)`:
> **over-dispersion** if `L_θ/L_total >= 0.70` and `L_π/L_total <= 0.30`; **dropout** if the reverse;
> **both, additively** if each share is in (0.30, 0.70) **and** `|L_θ + L_π − L_total| <= 0.02` in `I`;
> otherwise **the two interact and the loss is not decomposable**, reported as such.
> Consistency check reported either way: how close the multiplicative prediction lands to `I(A1b)`.

This is more informative than N4 and likely to resolve where N4 may not — but it is **an addition to**
the pre-registered escalation, not a substitute for it. Run N4 as written; N5 answers the next
question.

### 10.3 The gate that could lift the suspension

**N6:** §5's tier-1 finding — `theta` costing **40 %** of achievable `I` with `pi ≈ 0`, on a converged
fit, on the headline dataset — is the first uncontaminated evidence for §10's half 1. It is not a
gate. Turning it into one means pre-registering, before any fit: *what value of the bounded-dispersion
parameter recovers what fraction of `L_θ`, on tier-1 and on a converged deep fit, and what result
would show it manufacturing unconditioned variance instead* — with a fidelity metric beside `I`,
because §6 says `I` alone can be raised by making the model worse.

### 10.4 The one fit, and it is now motivated

**N7: B1** — refit `deep_starmap` with `ell` fitted by `fit_lengthscale_from_sections`, collapse
alarms escalated to a hard stop, ~3.3 h. It clears §1 and §2 of the review, and per §8 it is the only
route to `theta`/`pi` from a converged deep fit. Re-run A1 (with N2's addition) on the resulting
checkpoint.

**Order: N1–N3, then N4, then N5, then N7. N6 is written while N7 runs.**

### 10.5 What I would still not do

Build either half of §10; quote `+0.1021` or `+0.0729` as a result; treat A1a's `−0.28` as settling
the decode path, since it is provisional on the failed fit; or read A1b's `+0.33` as anything other
than the dead band.
