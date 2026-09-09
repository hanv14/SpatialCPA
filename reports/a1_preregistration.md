# A1 — emission ablation: pre-registration

**Written before the run.** The thresholds, the decision table, the level guard, the instrument
control and the two asymmetries below are fixed here and are not to be adjusted once the numbers
exist. `specs/10` §4.2's closing rule — *before pre-registering a test, ask what would have to be
true for it to fail* — is applied to this test in §7, because the last gate this project
pre-registered (`emission_repair_options.md` §8.3) could not fail in the informative direction and
nobody noticed until the data arrived.

---

## 1. The question

`deep_starmap`'s chain shows `I(mu)` = +0.7966 and `I(counts)` = +0.0729 against a real section at
+0.3236. The structure is present in the mean field and absent after the draw. **Which stage of the
emission loses it — the mean model, the dispersion, or counting statistics themselves?**

Nothing in the campaign has measured this. Every number so far is end-to-end.

---

## 2. The design

Four arms, all drawn **at the real section's own cells and positions**, so the kNN graph, the cell
count and the density are the ground truth's and are identical between arms and to the
`REF real counts` row. The layout, the prior and the flow are held out entirely: positions are the
tissue's, and the only thing that varies is what the counts were drawn from.

| arm | drawn from | isolates |
|---|---|---|
| **A1c** | `Poisson(mu_oracle)` — **model-free** | counting statistics alone, on the tissue's own mean field |
| **A1b** | `ZINB(mu_oracle, theta_model, pi_model)` | what the fitted dispersion and dropout cost, on top of A1c |
| **A1a** | `ZINB(decode(h1))`, `h1 = encoder(real counts)` | what the decode path costs, on top of A1b |
| **A1n** | real counts shuffled across cells | the estimator's null on this panel and this graph |

`mu_oracle` is the row-stochastic kNN mean of the real counts (self included, `k = metric_knn_k`),
so its per-gene column means equal the real section's and the arms are at the tissue's count level
by construction.

Reported alongside: `I(mu_oracle)` and `I(decode(h1))`, so the reader sees what went into each draw.

---

## 3. The primary statistic

For each drawn arm `X`, with `I_model` = that run's stage-4 `I(model counts)` and `I_real` = that
run's `REF real counts`:

```
R(X) = ( I(X) - I_model ) / ( I_real - I_model )
```

`R` is the fraction of the end-to-end deficit that this arm recovers. `R = 1` reaches the tissue;
`R = 0` does no better than the model does end-to-end.

**Bands, fixed here:**

| band | criterion | reading |
|---|---|---|
| **RECOVERS** | `R >= 0.70` | the stages upstream of this arm are not where the deficit is |
| **DOES NOT RECOVER** | `R <= 0.30` | the deficit survives even with everything upstream replaced by the truth |
| **UNINFORMATIVE** | `0.30 < R < 0.70` | neither; that row is not read, and the arm is escalated to three seeds |

The 40-point dead band is deliberate and symmetric: a stage carrying between a third and two thirds
of a deficit does not license attributing the deficit to it, and this project has twice attributed
one on a margin narrower than that.

---

## 4. The decision table

Read **A1c first**, then A1b, then A1a. The first row that matches is the conclusion.

| A1c | A1b | A1a | conclusion | what follows |
|---|---|---|---|---|
| DOES NOT RECOVER | — | — | **The sparsity bound.** No independent per-cell draw can reach the tissue at this sparsity, whatever the mean model. | §10 is dead in **both** halves. The result is the characterisation, published as such. B1 is still run for §1's fit failure, but not as a repair. |
| RECOVERS | DOES NOT RECOVER | — | **The fitted dispersion / dropout is the fault.** | §10 **half 1** (bound dispersion) is the lever. Half 2 is not needed and is not built. |
| RECOVERS | RECOVERS | DOES NOT RECOVER | **The decode path is the fault** — `mu` from the latent is too narrow or mis-structured. | §10 **half 2** is the lever, and §6's narrow-`mu` mechanism is supported. Half 1 is re-costed against it, not assumed. |
| RECOVERS | RECOVERS | RECOVERS | **The emission is not the fault.** The deficit is upstream, in the prior or the flow. | No emission redesign. **Run B1** (`ell` refit on `deep_starmap`) as the next step. |
| any UNINFORMATIVE at or before the deciding arm | | | **No conclusion.** | Escalate that arm to three seeds before anything else. |

---

## 5. The instrument control, and it can void everything

Tier-1 has **no deficit**: `I_model` = +0.5134 exceeds `I_real` = +0.4635, so `R`'s denominator is
negative and the statistic is undefined there. Tier-1 is therefore the control, not a second case.

**Pre-registered control criterion: on tier-1, every drawn arm (A1a, A1b, A1c) must reach
`0.6 x I_real` = +0.278.** If any falls below it, the ablation is not measuring what it claims —
on a dataset where the emission demonstrably works end-to-end, an arm given the *tissue's own mean
field* cannot be worse than the model — and **no `deep_starmap` row may be read.**

Run tier-1 first for this reason.

---

## 6. Guards, fixed here

**6.1 Level guard.** For each drawn arm, the median over the panel of (arm's per-gene mean) /
(real section's per-gene mean) must lie in **[0.5, 2.0]**. Outside it the arm sits at a different
count level, its Poisson floor is a different floor, and its `I` is not comparable however it was
drawn: that arm reads **UNINFORMATIVE** regardless of `R`.

**6.2 The oracle is an upper bound — stated in advance.** `mu_oracle` is a kNN mean, and smoothing
*creates* autocorrelation. The instrument's own self-check measures the size of this on a synthetic
field: **I 0.5494 -> 0.9704**. So:

* **A1b and A1c low is conclusive** — even an over-smoothed mean field cannot survive the draw.
* **A1b and A1c high is permissive, not proof** — it shows the draw *can* carry structure at this
  sparsity given a good enough mean field; it does not show the model could produce that field.

The decision table is written to depend on the conclusive direction wherever it can.

**6.3 Contamination by §1's failed fit — stated in advance.** The `deep_starmap` checkpoint's
spatial-collapse alarm fired at the final training step.

* **A1c is model-free** (real counts, the kNN graph, a Poisson draw). Admissible regardless.
* **A1b** uses only the fitted per-gene `theta` / `pi`. Flagged, weaker than A1c, stronger than A1a.
* **A1a** uses the encoder, decoder and size head from that fit. Its `deep_starmap` result is
  **provisional** and must be re-measured after B1 before it settles anything.

A conclusion that rests on A1a alone is therefore provisional by construction. A conclusion that
rests on A1c is not.

**6.4 Same panel, same seed.** Both runs use `--top-k-by real --top-k 32` and `--match-density`, the
same flags the step-0 runs used, and one fixed `--ablation-seed`. The panel is re-derived from the
real section, so it is identical to the step-0 run's on each dataset.

**6.5 `section_4` on `deep_starmap`.** `section_2` at z = 30.8 sits between `section_1` (6.3) and
`section_3` (43.4) and is the boundary-adjacent held-out section — the run's own flag said so, and
R3 is a 20–35 % deficit there. `section_4` at z = 68.6 is interior and is what A1 uses. This changes
the anchors: `I_model` and `I_real` are re-measured on `section_4` by the same run, and the
`section_2` values above are **not** carried over.

---

## 7. What would make this test fail — the §4.2 closing rule, applied to itself

The gate this replaces could not fail informatively: both sides of §8.3's `Var(log mu)` ratio passed
through the same decoder, so a narrow `mu` head pinned both and the ratio was ~1 by construction.
This one has **four** distinct reachable failure modes, three of which overturn a standing position:

1. **A1c comes back low** → §10 dies in both halves and the emission work stops. Reachable: at
   1.6 % median detection a Poisson draw may well not carry `I` = 0.32.
2. **A1a comes back high** → the emission is exonerated and the fault is the prior/flow, which is
   the opposite of the campaign's standing hypothesis since R12.
3. **The tier-1 control fails** → the instrument is wrong and nothing is read. Reachable: an oracle
   arm below 0.278 on a dataset with no deficit would be a straightforward contradiction.
4. **The level guard trips** → an arm is at the wrong count level and drops out.

The arms can also disagree in a way no single hypothesis predicts (A1c high, A1b low, A1a high),
which would say the fitted `theta`/`pi` are worse than the decode path — currently nobody's claim.

---

## 8. What A1 does **not** test

The layout head, the GRF prior, the flow, and the `ell` mismatch of §2 of the review. Positions are
held at the tissue's throughout. A1 answers *"is the emission capable"*; it does not answer *"is the
latent right"* except by elimination in the last row of §4.

---

## 9. The commands

Tier-1 **first** (§5's control). Both read a saved checkpoint; neither fits.

```bash
# CONTROL — tier-1. Every drawn arm must reach +0.278 or nothing else may be read.
python scripts/t10_chain_diagnostic.py \
    --dataset starmap_visual_cortex --load-model runs/chain/shipped_tier1.pt \
    --layout-sampler grid --match-density --top-k-by real --top-k 32 \
    --emission-ablation --ablation-seed 1 \
    --out reports/a1_tier1.md

# TEST — deep_starmap, on the INTERIOR held-out section.
python scripts/t10_chain_diagnostic.py \
    --dataset deep_starmap --load-model runs/chain/shipped_deep.pt \
    --section section_4 --target-z 68.6 \
    --layout-sampler grid --match-density --top-k-by real --top-k 32 \
    --emission-ablation --ablation-seed 1 \
    --out reports/a1_deep.md
```

`--text-emb-mode`, `--expr-pca-dim` and `--decoder-mu-link` are **refused** with `--load-model`: the
weights were fitted under the checkpoint's values and honouring a command-line override would report
one arm's config over another arm's weights. The config comes from the checkpoint.

**Cost**: no fit. One generation pass and four draws per dataset — minutes on tier-1, longer on
`deep_starmap` for the 30 097 x 1017 draws, not hours.

**Run `--self-check` first** (seconds): it now covers `knn_mean_field` as well as the panel and
density logic.

---

## 10. Reporting

`reports/a1_tier1.md` / `a1_deep.md` and their JSON sidecars carry every arm, its level ratio and
its `R`. The conclusion is read off §4 **as written**, in a follow-up that cites this file by commit.
If a reading requires a threshold this document does not contain, the answer is "uninformative",
not a new threshold.
