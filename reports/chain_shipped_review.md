# Review of the two shipped-arm chain runs, and what I would change

Sources: `chain_tier1.log`, `chain_deep.log` (the runs of `reports/emission_repair_options.md` §8.2).
**No code was changed for this report.** Everything below is a proposal.

> ## 🚩 Two corrections, from A1 (`reports/a1_review.md`)
>
> **1. The 1.11x is over-smoothing, not fidelity.** This report calls tier-1 "healthy and good" on
> the strength of the model emitting counts at 1.11x the real section's Moran's I, and §4 reads that
> as "no deficit to repair". A1 shows what produces it: on the **same emission**, the tissue's own
> latent yields **+0.4053** and the model's yields **+0.5134**. The model's latent is smoother than
> the tissue's — **1.28x** on tier-1, **2.46x** on `deep_starmap` — and the extra smoothness survives
> the draw better than the tissue's own structure does. **1.11x is not a reconstruction result and
> must not be quoted as one.** Everywhere below that reads it as fidelity is wrong; §4's "no deficit
> to repair" survives only in the narrow sense that the model is not *under* the tissue on `I`.
>
> **2. The deep deficit is 2.71x, and it has moved three times without a modelling change.**
> §0 and §7.1 quote `+0.0729` against `+0.3236` on the boundary-adjacent `section_2`. Each
> provenance fix has moved it toward the tissue:
>
> | measurement | model | tissue | ratio |
> |---|---|---|---|
> | `section_2`, z = 30.8 — boundary-adjacent, and not the section's own plane | +0.0729 | +0.3236 | 4.44x |
> | `section_4`, z = 68.6 — interior, still not its own plane | +0.1021 | +0.3123 | 3.06x |
> | **`section_4` at its own z = 73.5** | **+0.1154** | **+0.3123** | **2.71x** |
>
> **2.71x is the figure to quote, and the trend is part of the figure**: three corrections, none of
> them a modelling change, each one shrinking the gap, which means **the remaining number may still
> carry confounds nobody has found**. R3 alone was worth about a quarter of the original gap — what
> §7.1 predicted and A1 confirmed — and the real-side anomaly cleared with it (encoder-latent
> retention 134% -> 98.5%). None of `+0.0729`, `+0.1021` or `+0.1154` is quoted as a result.

---

## 0. The headline, in one paragraph

Tier-1 came back healthy and good: the model emits counts at **1.11× the real section's** Moran's I
on the full 28-gene panel, and every step-1 quantity matches the tissue-side reference to within 1 %.
`deep_starmap` came back at **0.0729 against the real section's 0.3236** — 4.4× below — which is
exactly §9's pre-registered reversal condition. **But that number is not admissible**, because the
`deep_starmap` fit's own spatial-collapse alarm fired at **122 of the checked training steps,
including the last one**, and because the run used **tier-1's fitted GRF length-scale**. Two of the
three pre-registered readings therefore do not resolve, and one of them — gate 2 — turns out to have
been mis-specified by me. What follows is what each run does establish, what voids the rest, and
eleven proposals ordered by cost.

---

## 1. 🚨 The `deep_starmap` fit never entered the healthy regime

`check_spatial_collapse` fires when the generated field's median Moran's I falls below
`sefl_spatial_collapse_warn_fraction = 0.05` of the real cells' on the same positions, after
`sefl_collapse_min_steps = 100`. Its own docstring records the reference: *on the one healthy
real-data fit measured, all negative excursions were inside `sefl_collapse_min_steps` and the floor
afterwards was **+0.5467***.

| | tier-1 | `deep_starmap` |
|---|---|---|
| variance-collapse alarm, steps fired | 100–360, then silent | 43 steps, 180 → **2399** |
| spatial-collapse alarm | 110–190, then silent | 43 steps, 180 → **2399** |
| **spatial *inversion*** (ratio < −0.05) | **never** | **79 steps**, 340 → 2360 |
| total flagged steps | ~15, all before step 360 | **122**, spread over the whole run |
| ratio at the final check | — (silent) | **−0.0129** at step 2399 |
| worst inversion | — | **−0.8314** at step 2300 |

Tier-1 is textbook: a transient at initialisation, clean for the last 85 % of training. `deep_starmap`
is the opposite — the generated expression field is flat or **anti-correlated in space for the entire
run**, and it is still flat at the last step measured.

**The 0.0729 is therefore a measurement of a model that never learned a spatial field, not a
measurement of the shipped emission.** It cannot be read against §9's reversal threshold, and it
cannot be read against §10's gates.

This is also `specs/10` §4.2's "alarms that fire where nobody looks", in a new place: the alarm was
built, it worked, it fired 122 times — into stderr, while the report the run produced reads as a
clean measurement and says nothing about it.

---

## 2. 🚨 The `deep_starmap` run used tier-1's fitted GRF length-scale

`scripts/t10_chain_diagnostic.py::main` hard-codes:

```python
ell_xy=116.3, ell_z=132.0
```

`reports/pilot.md` records where those came from: *"The fitted length-scale on **this volume**
is ell = (116.3, 116.3, 132.0) µm"* — **this volume** being tier-1 STARmap. `CTFFlow.__init__` reads
`cfg.ell_xy` / `cfg.ell_z` straight into `GaussianRandomField`; it does not fit. T03 ships the
fitter — `noise.fit_lengthscale_from_sections(vol, cfg, seed=...)` — and the chain script bypasses it.

The two volumes are not alike:

| | tier-1 | `deep_starmap` |
|---|---|---|
| cells / genes (training) | 16 527 / 28 | 115 830 / 1017 |
| median section spacing | 22 µm | **42 µm** |
| held-out section gap | 11.0 µm | 20.3 µm |
| real section's own latent I | **+0.6253** | **+0.2415** |
| real section's counts I (panel) | +0.4635 | +0.3236 |
| GRF prior I at generated xyz | +0.9362 | **+0.9284** |

The prior is **identically smooth on both** — it must be, `ell` is identical — while the tissue's own
latent autocorrelation is **2.6× lower** on `deep_starmap`. The model is being asked to fit a
short-correlation tissue from a long-correlation prior, and the flow can only comply by *cancelling*
the prior. Persistent anti-correlation of exactly the kind §1 logs is what cancelling looks like.

I would treat this as the leading causal hypothesis for §1, and it is cheap to test.

---

## 3. 🚨 Step 1's reference was the wrong reference — gate 2 is *not evaluated*, not failed

The pre-registration (§8.3) read: `Var(log mu_generated) / Var(log mu_encoder_real)` ≥ 0.8 means the
structured component is intact. Measured: **0.983** tier-1, **1.917** deep. Taken at face value that
fails gate 2 on both datasets and stops half 2 of §10's redesign.

**It should not be taken at face value, and the fault is in my specification.** Both columns are
`model.decoder` applied to a latent. The ratio asks *"does the generated latent drive this decoder as
hard as the encoder's latent does?"* It does **not** ask *"does this decoder's output range match the
tissue's?"* — and if the decoder's `mu` head is itself the narrow thing, both columns are pinned the
same way and the ratio is ~1 by construction. A ratio of 0.983 is exactly what that looks like.

There is direct evidence the generated side is pinned. `sd(log mu)` on the generated latent:

* tier-1, 28 genes, ~100 % detection: **0.7269**
* `deep_starmap`, 1017 genes, 1.6 % median detection: **0.7254**

Two datasets with nothing in common, and the same number to three decimals. That is not a
data-driven quantity. Meanwhile the *encoder-latent* column does move with the data (0.7333 vs
0.4663), so the decoder can produce different ranges — it is the flow's latent that lands in the same
place both times. The clamp is not the explanation (`zinb_mu_min/max` give a ±18.4 log window).

**Correct verdict: gate 2 reads NOT EVALUATED.** And that is not a reclassification that frees the
work it gated:

> **§10 is SUSPENDED, not re-opened.** A gate that could not be read is **not** a gate that passed.
> Half 2 is neither confirmed nor ruled out, and neither half is built, costed further or resumed on
> the strength of the reclassification. Gate 1 has not been run either. The suspension lifts only
> when a replacement gate — one whose reference is not the model's own decoder — is pre-registered,
> run, and returns a readable answer.

`reports/a1_preregistration.md` is that replacement's first half, and **three of its four decision
rows end §10 rather than resuming it**. Recorded in `emission_repair_options.md` §8.3a and as a
second instance of `specs/10` §4.2's closing rule.

---

## 4. What tier-1 *does* establish, and it is good news

Nothing in §§1–3 touches tier-1. It ran clean, and on it:

| stage | median I |
|---|---|
| 1. GRF prior at generated xyz | +0.9362 |
| 2. latent after the flow | +0.8011 |
| 3. decoded `mu` | +0.7920 |
| 4. sampled counts | **+0.5134** |
| REF real counts | **+0.4635** |
| REF real latent `h1` | +0.6253 |

* **The model emits counts at 1.11× the real section's autocorrelation** — 🚩 **and A1 shows this is
  over-smoothing, not fidelity** (see the correction box at the top). On the same emission the
  tissue's own latent yields +0.4053 against the model's +0.5134. §9's branch 1 holds only in the
  narrow sense that the model is not *under* the tissue on `I`; it is not a reconstruction claim.
* The emission costs 64.1 % against the tissue's own 74.1 % — a 1.16× gap, not a 7× one.
* Mean-variance slope 1.846 against the tissue's 1.738.
* `sd(log mu)` 0.7269 vs the reference's 0.7333; bounded structured share 99.8 % vs 99.7 %.

One caveat that must travel with it: `--top-k 32` on a 28-gene panel selected **all 28 genes**, so
on tier-1 the panel rule was a no-op. Tier-1's number is an unselected full-panel number;
`deep_starmap`'s is a top-3.1 % selection. They are not on the same selection regime (§7.2).

---

## 5. Two entries in the record are settled by these runs

**R12 is refuted on the shipped arm.** R12 says the decoder carries 15.3 % of its between-cell
variance in the structured mean against real tissue's 62.2 %. Measured here, panel-matched,
density-matched, on the shipped `exp` link:

> 🚨 **WITHDRAWN 2026-09-09 (`reports/a1_escalation_review.md` §3).** R12's claim is about
> **counts** — *"only 9-19% of the emitted count variance survives as between-cell structure"* — and
> the bounded share used here decomposes `Var(log mu)` into its latent-driven and size-factor parts,
> which never touches the sampling noise. Measured model-free on the shipped arm, the count-level
> structured share is **10.4%** for the model against **>= 42.5%** for the tissue on `deep_starmap` —
> inside R12's own range. **R12 stands.** What was refuted is a differently-defined quantity that
> shares its name because `t09_structured_share.py` implements the decoder-internal one.


| bounded structured share | generated | real latent |
|---|---|---|
| tier-1 | **99.8 %** | 99.7 % |
| `deep_starmap` | **97.3 %** | 93.7 % |

The size factor contributes essentially nothing (`Var(log s)` = 0.00125 / 0.01471). The 15.3 % was an
artifact of the `softplus` arm and an unmatched panel. R12's premise should be withdrawn, not
adjusted — and note that the whole of §2's arithmetic bound was built on the 0.09-vs-0.62 framing
this replaces.

**"Tissue's 1.213" is now sourced, and it is not 1.213.** The matched tissue-side `sd(log mu)` is
**0.7333** (tier-1) and **0.4663** (`deep_starmap`). The 1.213 figure carried through the written
record should be withdrawn — with the caveat from §3 that neither of these is the *tissue's* value
either; both are the decoder's output on an encoded latent.

---

## 6. One mechanism fits both datasets, and it is testable

Not established — proposed, because it explains every number here with one cause.

The model's `mu` spans `e^{±0.73}` ≈ a 4× range between cells, on both datasets. Real expression is
cell-type-driven and closer to on/off: a marker gene at a few counts inside its type and zero
outside is a spread of `sd(log mu)` in the 2–4 range, not 0.73.

* **Where detection is ~100 % (tier-1's curated 28 genes)**, a 4× `mu` range still survives the draw:
  the Poisson term is small next to `mu`, ranks are informative, and the counts inherit `mu`'s
  structure — retention 64 %, and the model even beats the real section.
* **Where the median gene is detected in 1.6 % of cells (`deep_starmap`)**, `mu` sits around
  hundredths of a count. A 4× range there is 0.01 vs 0.04 counts; nearly every draw is zero, the
  rank-normalised vector is almost all ties, and Moran's I collapses **whatever `mu`'s spatial
  structure is** — which is exactly what stage 3 → stage 4 shows (+0.7966 → +0.0729). Real tissue
  survives the same sparsity because its `mu` spread is large enough that a cell either has the
  transcript or does not.

**Prediction, and it is falsifiable in one afternoon**: the tissue's own `sd(log mu)`, estimated
model-free from the real counts, is ≳ 2 on `deep_starmap` — i.e. **3–5× the model's 0.725**, which is
a variance ratio of 0.04–0.11, well inside gate 2's ≤ 0.4 confirming band. If instead it comes back
near 0.5–0.9, this account is wrong and §1's fit failure is the whole story.

Note what this does to the framing: the deficit would be **narrow `mu` meeting sparsity**, not
"the emission destroys structure". The emission is the amplifier, sparsity is the gain, and the
signal going in is too small.

---

## 7. Confounds that must be cleared before the `deep_starmap` number is quoted

Beyond §§1–2, four more:

**7.1 The plane is a boundary plane.** The run's own flag: *"`--target-z 30.8` is a boundary plane
(within 0.5 median spacings of the stack's end)"*. T04 measured a **20–35 % reconstruction deficit**
there (R3). The stack settles it: `section_2` at z = 30.8 sits between **`section_1` at 6.3** and
**`section_3` at 43.4**, so it is the boundary-adjacent held-out section; **`section_4` at z = 68.6
is interior**. The headline deep number carries R3 uncorrected.

🔁 **Standing instruction for every future `deep_starmap` run: use `--section section_4
--target-z 68.6`.** The script's defaults are tier-1's `section_2` at z = 30.0 and do not transfer.
A1's pre-registration (§6.5 there) already fixes this, and it means A1 re-measures its own anchors —
the `section_2` values in this report are **not** carried across.

**7.2 The two datasets are not on the same panel regime.** Tier-1: all 28 genes. `deep_starmap`: the
top 32 of 1017 = 3.1 %. Selection lifts the real side on `deep_starmap` and not at all on tier-1, so
the cross-dataset comparison is not like-for-like — the §0a error in a new dress.

**7.3 The panel is selected by the statistic it is then scored on.** Choosing the top 32 genes by the
real section's Moran's I and then reporting the real section's Moran's I on those genes is a
selected maximum. It biases the real side upward — the safe direction for a claim *against* the
model, but it means 0.3236 is an upper bound on the tissue, not an estimate of it.

**7.4 The "retention" column is not comparable between the two arms.** Its denominators are the
generated latent (I = 0.7858) and the encoder's latent (I = 0.2415) — **3.3× apart** — and one of
them is panel-selected while neither latent is. Real tissue's "134 % retention" is an artifact of
that mismatch, not a property of tissue. The comparable pair is `I(model counts)` vs
`I(real counts)`; the retention column should be dropped or renamed.

---

## 8. Three defects in the instrument I built

1. **The training alarms never reach the report.** A run whose spatial-collapse alarm fired at the
   final step produced a report with a clean verdict section. The alarm state belongs in the
   provenance block and the sidecar, and a final-step alarm should suppress the verdict.
2. **The density line says "matched" when matching did not happen.** Tier-1's layout produced 4 073
   cells against a target of 4 187; the stderr line said so, and the report's table still read
   *"matched to the real section: 4073 generated -> 4073 kept"*. It should read "requested 4 187,
   produced 4 073, NOT matched".
3. **`--top-k` larger than the panel silently means "all genes"** and the report still calls it
   "top 28 by Moran's I on the real side". It should say the selection was vacuous.

---

## 9. Proposals, ordered by cost

### Tier A — no refit. Both checkpoints are on disk (`runs/chain/shipped_{tier1,deep}.pt`)

Blocked on one missing flag: the script has `--save-model` and **no `--load-model`**. Adding a reader
is the enabling change for all five, and it is small.

| # | proposal | what it decides |
|---|---|---|
| **A1** | **Emission ablation on the real latent.** Encode the real section → `h1`, decode → `mu, theta, pi`, **sample counts**, measure I on the same panel. | The whole question. If `I ≈ 0.07`, the **emission** destroys the structure and the latent is irrelevant. If `I ≈ 0.30`, the emission is fine and the **prior/flow** is the fault. Nothing in the campaign has measured this, and it is one forward pass. |
| **A2** | **Model-free tissue `sd(log mu)`.** Per gene, from the real counts alone: (a) kNN-smoothed counts, `sd(log(smoothed + eps))`; (b) NB moment estimator, `CV²_mu = (Var y − ȳ)/ȳ² − 1/θ̂`. | Replaces §3's broken reference and tests §6's prediction. Re-runs gate 2 against a reference that is not the model's own decoder. |
| **A3** | **`I(mu)` decoded from `h1`**, alongside A1. | Completes the real-side chain, so both sides read prior → latent → `mu` → counts on the same axis. |
| **A4** | **Level and detection check**: per-gene median count and detection rate, generated vs real. | Currently missing entirely. If the model emits systematically fewer counts, the Poisson floor is worse than the tissue's for a reason that has nothing to do with structure. One line, high information. |
| **A5** | **Re-measure `deep_starmap` on `section_4`** (interior) from the same checkpoint. | Clears 7.1. If 0.0729 moves materially, R3 is in the headline number. |

**A1 is the one I would run first.** It splits the emission hypothesis from the latent hypothesis
with a single forward pass, and every design decision downstream depends on which side it lands.

### Tier B — one refit each

| # | proposal | cost | what it decides |
|---|---|---|---|
| **B1** | **Refit `deep_starmap` with `ell` fitted on `deep_starmap`** (`fit_lengthscale_from_sections`), collapse alarms escalated to a hard stop. | ~3.3 h | §§1–2 together. If the alarms clear and counts I rises, the deep deficit was a run defect, not a method defect, and §9's reversal condition has to be re-evaluated from scratch. |
| **B2** | **Tier-1 at seeds 2 and 3.** | ~1 h each | Tier-1's "1.11× real tissue" is **one seed**. A headline claim on one seed is what this project has voided runs for. |
| **B3** | **Tier-1 and `deep_starmap` on a matched panel fraction** (e.g. top 3 % on both, and full-panel on both). | reuses B1/B2 fits | Clears 7.2. |

**B1 is the highest-value fit in the plan**, and I would not run any of §10's redesign fits before it.
Fitting `ell` per dataset is not a design change — it is what T03 exists for, and the run bypassed it.

### Tier C — reporting and process, no compute

| # | proposal |
|---|---|
| **C1** | Alarm state into the report and the sidecar; a final-step spatial-collapse alarm suppresses the verdict section and marks the artifact `not_admissible`. This is the §4.2 rule the campaign keeps re-learning. |
| **C2** | Fix the density and panel provenance lines (§8.2, §8.3) so a vacuous match or a vacuous selection says so. |
| **C3** | **Split-half panel selection**: choose the panel on half the real cells, score both arms on the other half. Removes 7.3 at no compute cost. |
| **C4** | Drop or rename the "retention" column (7.4). |
| **C5** | Withdraw R12's 15.3 %/62.2 % and the "tissue's 1.213" from the written record, citing §5. |
| **C6** | Record gate 2 as **not evaluated** with the reason in §3, rather than as failed. This one matters: the pre-registered stop rule would otherwise cancel half of §10 on a mis-specified reading. |

### Tier D — design changes, and **not yet**

Only if A1 lands on the emission side *and* A2 confirms §6's prediction *and* B1 leaves a deficit.
Then the target is not "raise `Var(mu)`" in the abstract but **match the tissue's `mu` spread**: the
model's `sd(log mu)` is ~0.73 on every dataset it has been run on, and the mechanism that would
produce a tissue-like spread is a `mu` head that can express on/off structure — a per-gene learned
scale on the log-`mu` head, or conditioning `mu` on the cell-type channel that already exists in the
conditioning vector. Both are smaller than §10's two-half redesign and both are testable by A2's
statistic before any fit.

---

## 10. What I would *not* do

* **Not build either half of §10's redesign.** Gate 1 has not been run and gate 2 did not evaluate.
  Adopting a new emission on the strength of a number produced by a fit whose spatial alarm fired at
  the final step would be the largest version of the error this campaign has been correcting.
* **Not quote 0.0729 anywhere**, including as a negative result, until §1 and §2 are cleared. It is
  currently a measurement of a failed fit under a mis-specified prior.
* **Not treat §9 as decided.** Its reversal condition was met numerically — real 0.3236 ≥ 0.28,
  model 0.0729 in the 0.07–0.10 band — but on inadmissible inputs. The defect-or-property question
  is open, and B1 is what reopens it honestly.
* **Not weaken the collapse alarm to get a clean run.** It was right both times.

---

## 11. Where I was wrong

`reports/emission_repair_options.md` §8.3's pre-registration is mine and it is faulty: I specified a
gate whose two sides pass through the same decoder, so it cannot see the failure it was written to
detect. I checked that the comparison was matched in arm, panel, density and estimator — and did not
ask whether the *reference* was a reference to the tissue at all. That is the same shape as the
§4.2a failure recorded in `reports/envelope_correction.md`: the gate I checked was not the gate that
mattered.

It cost nothing this time, because the two runs produced the raw material to see it. It would have
cost §10's stop rule.
