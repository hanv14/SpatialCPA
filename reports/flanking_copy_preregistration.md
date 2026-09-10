# Pre-registration — putting `flanking_copy` on the ladder and through R3

**Committed before the arm is implemented or run.**

## 1. The interest I have in the answer, stated first

This test asks whether the baseline that beats us is measuring what the benchmark says it measures.
One of its two outcomes is favourable to this project and the other is not. That asymmetry is the
reason for the unusual length of §4 and §5, and for one rule that governs everything below:

> **The default, absent a clear result, is the outcome that does not suit us.** UNINFORMATIVE,
> specification disagreement, a fired null check, or a failed precondition all resolve to
> *"the copy floor stands and the deficit is ours"*. Only an unambiguous result under every
> specification in §3 moves off that default.

And one constraint that is not conditional on any outcome:

> **The negative result is reported either way.** v25 loses to a model-free copy of a neighbouring
> section on `paper_morans_pearson`. That sentence goes in the paper whatever this test returns.
> The benchmark analysis, if it survives §5, is reported *beside* it and never *instead* of it.

## 2. What `flanking_copy` is, exactly

From `benchmark-pbya-v3/src/bench3/selftest.py::make_probe`, read from source:

```
train_secs = np.unique(secs_train)
src = min(train_secs, key=lambda s: abs(z[s] - z_target))
X, xy, ct = train_adata[src].X, train_adata[src].spatial[:, :2], train_adata[src].cell_type
```

It emits the **nearest training section by z, verbatim** — that section's counts, at that section's
own cells, in that section's own cell count. Under the paper design (`paper_2_4_6`: input 1/3/5/7,
sections 2/4/6 held out **simultaneously**) the source for `section_4` is section_3 or section_5,
whichever is nearer in z, `min` breaking a tie toward the lexically first. **The run reports which
section it actually selected**; nothing here assumes it.

**Two asymmetries that must be stated before any number is read:**

**2a. It is not at the target's cells.** Every A1 rung is drawn at the real section's own cells, so
`pred_xy == gt_xy` and §2a of `ladder_preregistration.md` reads a low rung as conclusive. Neither
`flanking_copy` nor stage 4 has that: both are a different cell set scored on its own graph, which
is exactly what `paper_morans_pearson` is built for ("alignment-free"). **So `flanking_copy` is
comparable to stage 4 and to nothing else on the ladder.** It is rendered in its own scope block and
is not to be read against A1a, A1b or A1c.

**2b. 0.9836 is tier-1's number, on 28 genes, medianed over sections 2/4/6.** This test runs on
`deep_starmap` section_4, all 1017 genes — the only scope the corrected null check leaves readable.
`flanking_copy` on that scope is **a different number that nobody has measured**, and the test is
against whatever it scores there, not against 0.9836. If its score on deep's all-genes scope is not
high, the premise of the test — that there is a high model-free floor to explain — does not hold on
this dataset, and the test returns **PREMISE ABSENT** rather than a result in either direction.

## 3. The measurement — identical construction, three control specifications

`r_flank` = `morans_agreement(I(flanking_copy counts, its own graph), I(real section_4 counts, its
own graph))` — the same function, `k=10`, rank-normalised both sides, NaN dropped pairwise, all
1017 genes. The same construction as every other rung.

**R3 for the copy** = partial correlation of `I_flank` and `I_real` controlling for a property of
the **target** section, on a quadratic basis, exactly as R3 already does for stage 4. Retained
fraction = `R3 / r_flank`.

The two existing control specifications disagreed on both datasets (0.247, 0.256, against a 0.150
tolerance), so **two are not enough for a claim of this kind and a third is added now**, before any
of them is computed:

| # | control | 
|---|---|
| C1 | per-gene detection rate in the target section (fraction of non-zero cells) |
| C2 | per-gene log mean count in the target section |
| C3 | per-gene log **variance** of counts in the target section |

C3 is added because C1 and C2 are both location statistics of the same count distribution and their
disagreement may say only that the abundance–`I` relation is not quadratic in either. C3 moves the
control to a different moment. **All three are computed and all three are reported**; none may be
selected after the fact.

## 4. Outcomes, fixed now

Bands are on the **retained fraction** `R3 / r_flank`, and must agree across **all three** controls.

| | retained fraction | reading |
|---|---|---|
| **1. FLOOR IS GENUINE** | ≥ 0.60 under C1, C2 and C3 | The copy floor is spatial fidelity. The deficit is ours. The paper reports a negative result and makes **no** benchmark claim. |
| **2. FLOOR IS ABUNDANCE** | ≤ 0.30 under C1, C2 and C3 | The copy's score is largely abundance matching. **This is the outcome that suits us**, and it is not reportable on this evidence alone — §5 governs. |
| **3. PARTIAL** | any control between 0.30 and 0.60 | Outcome 1's reading stands by default. |
| **4. UNINFORMATIVE** | the three controls disagree by > 0.15, or `r_flank` < 0.20, or the scope's null check fires | Outcome 1's reading stands by default. |
| **5. PREMISE ABSENT** | `r_flank` on this scope is below stage 4's | There is no high model-free floor here to explain (§2b). No reading in either direction. |

Outcomes 3 and 4 resolving to outcome 1's *reading* is deliberate and is the §1 rule applied: a
test we have an interest in does not get to return "unclear" and have that count as support.

## 5. The binding constraint — what the paper needs before it may make the benchmark claim

Outcome 2 is a benchmark critique advanced by the method that loses to the benchmark. One partial
correlation is not enough support for it, and the following are **preconditions, not follow-ups**.
If outcome 2 fires and any of these is missing, the paper reports the negative result and states the
abundance hypothesis as **an open question with the evidence for it**, never as a finding.

**5a. A constructive test that replaces the partial correlation.** Every number in §3 depends on a
control specification we chose. The direct version does not:

> **F3 — the abundance-matched relabelling.** Rank all 1017 genes by the target section's detection
> rate, cut into strata, and permute *which predicted gene is compared to which real gene* **within**
> each stratum. This preserves the abundance–`I` relationship exactly and destroys gene-specific
> spatial identity. If `r` under F3 stays near `r_flank`, the copy's score is carried by abundance
> and the critique is demonstrated without any partial correlation. If `r` collapses toward the
> null, the critique fails and outcome 2 is withdrawn regardless of what R3 said.

F3 is the decisive instrument and R3 is the corroborating one, not the other way round. Its stratum
width is a degree of freedom, so **three widths — 10, 25 and 50 genes — are all reported** and the
reading must be stable across them. F3 is drawn at 20 seeds and reported with its spread.

**5b. The critique must not depend on v25's own numbers.** It must be statable and testable using
only the benchmark's own probes and the real data. Deleting every v25 row from the analysis must
leave the conclusion unchanged. If it does not, it is special pleading.

**5c. A positive control that the metric does respond to position.** `spatial_scramble` keeps every
per-gene marginal — identical detection rates, identical abundances — and destroys only position.
**I predict now that it scores near zero**, because permuting cells drives every gene's `I` to the
permutation null and a near-constant vector cannot correlate with anything. If that prediction
holds, the metric plainly does respond to position, and **the maximal defensible critique is already
narrower than "the metric does not measure spatial fidelity"** — it is:

> *Among predictions that carry realistic within-gene autocorrelation, the across-gene correlation
> `paper_morans_pearson` is dominated by per-gene abundance.*

That narrower sentence is the only one outcome 2 can support, and it is written here so that a
broader one cannot be written later. If `spatial_scramble` instead scores high, that is a much
stronger result than anything else in this document and is reported as the headline.

**5d. Replication.** One section of one dataset is not a benchmark claim. Required: all three
held-out sections (2, 4, 6) on `deep_starmap`, and the same on `starmap_visual_cortex` **reported
even though its 28 genes are underpowered** — reported as underpowered, not omitted. The two
datasets are already known to differ on exactly this axis (`R1` = 0.68 on deep against 0.29 on
tier-1), so a result that holds only on the 1017-gene dataset is a statement about **gene-panel
composition**, not about the metric, and must be written as one.

**5e. The scope's null check must pass** under `reports/null_band_preregistration.md` §2a, and every
rung difference the claim rests on must have a bootstrap interval excluding zero under §2b.

## 6. Predictions, recorded now

So they cannot be claimed as confirmations afterwards, and so being wrong is visible:

1. `spatial_scramble` ≈ 0 (§5c). **High confidence.**
2. `r_flank` on deep's all-genes scope is **above** stage 4's +0.7306 — the premise holds. Moderate
   confidence; deep's sections are 1017 genes and the copy is one z-step away.
3. **Outcome 3 (PARTIAL), not outcome 2.** deep's `R1` = +0.6787 says abundance explains much of the
   tissue's own ordering, but "much" is not "≥ 0.70 of the copy's score". I expect the copy to
   retain more than stage 4 did, because a copy carries real within-gene structure that stage 4's
   emission destroys.
4. F3 collapses at least halfway toward the null. If it does not, prediction 3 is wrong and outcome
   2 is live.

If 2 and 3 both hold, the honest summary is: *the copy floor is partly abundance and partly
fidelity, v25 is behind on both, and the paper reports a negative result.*

## 7. What is reported regardless of outcome

- The ceiling result that closed the repair programme: within this architecture, a **perfect mean
  field with the model's own emission reaches 0.8242** on deep's all-genes scope, against a copy
  floor near 0.98. Even a perfect autoencoder loses to copying. This is a measured defect with a
  size, not a repair route.
- `A1a` (+0.6732) below stage 4 (+0.7306) with the positional advantage on the wrong side — a
  better latent is worth nothing on the scored metric.
- The π/θ reversal: on the scored metric π carries ~70 % of the emission's cost and θ ~12 %,
  inverting N5's median-based verdict on its own quantity.
