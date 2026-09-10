# Q1.5 read out: the emission programme is aimed at the wrong quantity

Sources: `reports/a1_tier1.md`, `reports/a1_deep.md` (Q1's transform fix and Q1.5's agreement block).
Criteria: `reports/q15_preregistration.md`, committed before the statistic existed.
**No code was changed.**

---

## 0. Headline

**The reconstruction tracks the score.** On tier-1 the chain's `r(stage 4)` is **+0.5076** against the
published `paper_morans_pearson` of **0.5574** — a difference of **0.0498**, on the *same* gene set
(tier-1's panel is all 28 genes, which is bench3's), leaving one section vs the median over 2/4/6 as
the only real difference. §2 of the pre-registration warned this might not be measuring what the
benchmark measures. It is.

**And it says the emission programme is aimed the wrong way on the headline dataset.** Removing the
emission's noise *entirely* takes tier-1 from **+0.5076 to +0.3878** — `d r = −0.1198` — and triples
the per-gene level error, `mae` **0.1018 → 0.3529**. Against the model-free copy floor of **0.9836**,
a complete emission repair **widens** the gap by **0.17**.

**Q2 and Q3 should not run**, and §10 is now answered — not by a gate that passed, but by a free
measurement showing the programme's own ceiling is *below its baseline* on the dataset the paper
leads with.

---

## 1. Q1's fix worked, and the anomaly was exactly what it was diagnosed as

| | stage 3 | stage 4p | 4p ≤ 3? |
|---|---|---|---|
| tier-1, **rank** | +0.8365 | +0.8358 | ✅ by 0.0007 |
| tier-1, **raw** | +0.7920 | +0.7911 | ✅ by 0.0009 |
| deep, **rank** | +0.7699 | +0.2594 | ✅ |
| deep, **raw** | +0.7451 | +0.3030 | ✅ |

The invariant holds on **both** transforms and on both datasets; no banner fired. The impossible
+0.8358 > +0.7920 of last round was a raw denominator against a ranked numerator and nothing else.
`I_rank(4p) ≤ I_rank(3)` is now asserted at runtime, so a recurrence stops the report rather than
waiting for a reader.

---

## 2. The pre-registered verdicts, applied as written

| | `r(3)` mu | `r(4)` counts | `r(4p)` emission-free | **(a) `d r`** | **(b) `r(4p)`** |
|---|---|---|---|---|---|
| **tier-1** (all 28 genes) | +0.3847 | **+0.5076** | +0.3878 | **−0.1198 → DOES NOT** | **0.3878 → CEILING LOW** |
| deep, panel (32) | +0.1254 | +0.4055 | +0.6130 | +0.2075 → MOVES IT | 0.6130 → CEILING LOW |
| **deep, all 1017** *(governs)* | +0.4332 | **+0.7306** | +0.8400 | **+0.1094 → PARTIAL** | **0.8400 → PARTIAL** |

The panel row is attenuated exactly as §5 of the pre-registration predicted — the top 3.1 % by the
real section's own `I` has a compressed range — and §2 fixed in advance that the all-genes row
governs on deep.

**No dataset returns "MOVES IT + CEILING IS HIGH".** Tier-1 returns both negatives, with `d r`
*negative*: removing the emission does not merely fail to help, it **hurts**. Deep's governing row is
PARTIAL on both, and `r(4p)` misses the 0.85 threshold by **0.01**.

**`mae`, reported and not a criterion, agrees and is sharper.** On tier-1 it goes **0.1018 → 0.3529**
when the emission is removed. That is the statistic the evaluator's own docstring says *catches
over-smoothing* — and it is **not** in the project's scored set. Round 11's finding, the metric that
detects it, and now a measurement showing it detecting exactly the failure an emission repair would
introduce. The case for scoring `paper_morans_mae` is no longer a docstring quotation.

---

## 3. The gap is 0.426, not 0.375 — and it is against a **model-free copy**

`reports/advisor_report.md` §5.1, tier-1, three seeds, the shipped configuration:

| | `paper_morans_pearson` |
|---|---|
| `oracle` | 1.0000 |
| **`flanking_copy` — the model-free floor** | **0.9836** |
| SpatialZ | 0.932 |
| **v25 shipped** | **0.5574** |

v25 is **−0.4262 below a baseline that copies a neighbouring real section**. My pre-registration §5
scaled its bands to the 0.375 gap against SpatialZ; the real target is the copy floor, and it is
larger. Correcting that makes the bands *more* lenient than they should have been, not less — so the
verdicts in §2 stand and would only be firmer.

**Where a complete emission repair lands on that scale: 0.3878.** It moves v25 **away** from the
floor, by 0.17.

---

## 4. What actually carries the scored correlation, and it is not the latent

| | `mu` alone | + full ZINB draw | + Poisson draw only |
|---|---|---|---|
| tier-1 | +0.3847 | **+0.5076** (+0.1229) | +0.3878 (+0.0031) |
| deep, all genes | +0.4332 | +0.7306 (+0.2974) | **+0.8400** (+0.4068) |

**The mean field alone scores worst on both datasets, and the *draw* is what raises the
correlation.** On tier-1 a Poisson draw adds nothing (+0.0031, counts are dense) and it is
**`theta`/`pi` that add +0.1229**. On deep the Poisson draw alone adds +0.4068.

Read plainly: **the genes `mu` makes spatially structured are largely not the genes the tissue makes
spatially structured** — that is what `r(3)` ≈ 0.38–0.43 says — and most of v25's 0.5574 is earned
by the count-generating process reproducing a per-gene ordering that the latent does not.

That is a different failure from the one this campaign has been chasing. Every chain measurement so
far has been about **how much** structure survives; this one is about **which genes** have it, and the
answer is that the model's mean field gets the *which* substantially wrong.

⚠️ **Not yet established, and §6 says how to settle it.** The obvious mechanism — that the tissue's
per-gene `I` ordering is largely a *sparsity* ordering, which the draw reproduces for reasons
unrelated to the latent — is a hypothesis. It sits awkwardly with v25 scoring **0.9721** on
`paper_gene_mean_spearman`, near the floor's 0.9863, which says the per-gene *level* is matched well.
So sparsity cannot be the whole story, and I am not asserting it.

---

## 5. One caveat on deep's all-genes figure

`median gt` over all 1017 genes is **+0.0180** and `median pred` is **+0.0072** — on that panel the
tissue's typical gene has essentially no spatial structure. A Pearson correlation over such a cloud
is carried by the minority of structured genes plus a large mass of near-zero-against-near-zero
points, and `mae` (0.0296) is small for the same reason. **bench3 computes exactly this**, so it is a
property of the score rather than an artifact of the reconstruction — but deep's +0.7306 and tier-1's
+0.5076 are not the same kind of number and should not be averaged or compared directly.

---

## 6. What to do next

**Do not run Q2 (M3) or Q3 (B1).**

* **Q2 is answered before it runs.** M3's fidelity condition (outcome **HARMED**) is predicted to
  fire on tier-1: a *complete* emission repair drops the scored statistic by 0.12 and triples `mae`,
  and a partial `theta` floor lands somewhere on that path. Its mechanism question — does bounding
  `theta` widen `mu` — remains unanswered and is now worth much less, because the quantity it would
  improve is not the one being scored.
* **Q3's justification is weakened but not void.** B1 fixes the latent's smoothness. §4 says the
  latent is where the gene-wise failure lives, so B1 is still pointed at the right component — but
  smoothness is a *level* property and `r(3)` is a *which-genes* property, and nothing here shows
  `ell` moves the second. It should wait for §6.1.

### 6.1 The free measurement that decides where the deficit is

**One `--load-model` read per dataset, no fit.** Add to the agreement block:

| # | measurement | what it settles |
|---|---|---|
| **R1** | `corr(I_real, detection_rate_real)` and `corr(I_real, mean_count_real)`, per gene | how much of the tissue's own `I` ordering is a **sparsity** ordering |
| **R2** | the same two for the model's stage-4 counts | whether the model's ordering is *more* sparsity-driven than the tissue's |
| **R3** | **partial correlation** `corr(I_4, I_real \| detection_real)` | **the decisive one** — how much of `r(4)` survives once sparsity is controlled for. It separates *"the model structures the right genes"* from *"the model makes the right genes sparse"* |

If `r(4)` collapses under R3, then `paper_morans_pearson` is substantially a sparsity-matching metric
and the route to it is the count distribution, not the spatial field. If it survives, the latent's
gene-wise spatial fidelity is the target and B1 and its successors are the right work.

**Pre-register R3's bands before computing**, as Q1.5 was.

### 6.2 Two record items

* **`paper_morans_mae` should be in the scored set.** It is emitted, documented as the over-smoothing
  detector, and §2 shows it detecting the failure an emission repair introduces, 3.5x, where the
  scored `pearson` moves only 0.12. That is a proposal about the benchmark, not about the model, and
  it should be raised as such.
* **The gap is 0.426 against a model-free copy floor**, not 0.375 against SpatialZ. Correct it
  wherever 0.375 was used, including my own pre-registration §5.

---

## 7. Where I was wrong

**My pre-registration scaled its bands to the wrong gap** — 0.375 against SpatialZ, when the floor
that matters is `flanking_copy` at 0.9836, a gap of 0.426. It made the bands more lenient than they
should have been, so no verdict changes, but the number was in `advisor_report.md` §5.1 the whole
time and I did not look it up.

**And I under-rated §2's own caveats in the other direction.** I wrote that the reconstruction "is
not comparable to 0.557 as a number". On tier-1 it lands within 0.05 of it, on the same gene set —
so the caveat was right to state and wrong to lean on, and the exercise is more informative than I
claimed it would be.
