> ⚠️ **§2's caveat is RETIRED for `starmap_visual_cortex`** (2026-09-10). The chain reproduces
> bench3's published tier-1 `paper_morans_pearson` — `flanking_copy` at **0.9836** against a
> published 0.9836 (exact), v25 at **0.5600** against a published 0.5574 — as the median over
> sections 2/4/6, which is bench3's own construction and, on tier-1's 28 genes, the same gene set.
> Tier-1 numbers in this campaign ARE comparable to a published `paper_*` figure.
> `deep_starmap` is **not** anchored and §2 stands there. See `reports/scale_anchoring.md`.

# Q1.5 — does the emission programme move the metric the paper is scored on?

**Written before the statistic is computed.** The definition below was read out of the pinned
evaluator's source, not recalled — `specs/10` §4.2l was earned one round ago for reading a statistic
by its name, and this is the first place since where that could happen again.

---

## 1. The gap this exists to close

**Every number in this programme is a median.** Median per-gene Moran's I at each chain stage,
median over the panel. The metric the paper is scored on is `paper_morans_pearson`, which is a
**correlation across genes** — and a model can match the tissue's median exactly while getting every
individual gene wrong. v25 scores **0.557** against SpatialZ's **0.932**, and nothing in the chain
work has touched that number.

So before spending 4.3 h on Q2 and Q3: **does removing the emission's noise move the scored
quantity?** If it does not, the emission programme is optimising something the benchmark does not
measure, and the ceiling that "closes 73 % of the deficit" on deep closes 73 % of a deficit nobody
scores.

---

## 2. The definition, read from the source

`benchmark-pbya-v3/src/bench3/evaluate_paper.py`:

```python
def _agreement(pred_vals, gt_vals, name):            # line 102
    ok = ~(np.isnan(p) | np.isnan(g))
    out = {f"{name}_median_pred": median(p[ok]), f"{name}_median_gt": median(g[ok]),
           f"{name}_mae": mean(abs(p[ok] - g[ok])), ...}
    if ok.sum() >= 3 and p[ok].std() > 0 and g[ok].std() > 0:
        out[f"{name}_pearson"]  = pearsonr(p[ok], g[ok])[0]
        out[f"{name}_spearman"] = spearmanr(p[ok], g[ok])[0]

def spatial_autocorrelation_metrics(pred_xy, pred_R, gt_xy, gt_R, k=SPATIAL_K):   # line 120
    out.update(_agreement(_morans_i(pred_xy, pred_R, k),
                          _morans_i(gt_xy,   gt_R,   k), "morans"))
```

with `pR, gR = _rank_normalize(pred_X), _rank_normalize(gt_X)` (line 670) and `SPATIAL_K = 10`.

**What matches the chain exactly:** rank-normalised counts on both sides; `k = 10`, the same as
`Config.metric_knn_k`; **each side on its own spatial graph** — the docstring calls it
*"alignment-free"*, which is exactly how the chain measures stage 4 at the generated cells and
`REF real counts` at the real ones; and NaN genes dropped pairwise.

**What does NOT match, and must travel with every number:**

| | bench3 | the chain |
|---|---|---|
| genes | **all shared genes** (`ggi`) | the run's panel — all 28 on tier-1, **top 32 of 1017** on deep |
| sections | median over held-out **2, 4, 6** | **one** section |
| cells | ground-truth-matched per section | this run's density rule |

So what follows is **a faithful reconstruction of the statistic, on this run's data — not a
reproduction of the score.** It is not comparable to 0.557 as a number. It is comparable **between
stages of the same run**, which is the whole of what Q1.5 asks. On deep it is computed on the panel
**and on all 1017 genes**, and the all-genes figure is the one closer to the benchmark's gene set.

---

## 3. 🚩 A metric the evaluator computes, documents as the over-smoothing detector, and the project does not score

`_agreement` emits **five** keys. The project's `METRICS` tuple in `scripts/t10_rescore_saved.py`
scores `paper_morans_pearson` and **not** `paper_morans_mae`. The evaluator's own docstring, line
127:

> *"`*_pearson`/`*_spearman` capture whether the **ranking** of genes by spatial structure is
> preserved; `*_mae` captures whether the **level** is right, **which is what catches over-smoothing**
> (a blurred reconstruction inflates Moran's I and deflates Geary's C while keeping the gene ranking
> intact)."*

That is the over-smoothing finding of round 11, written into the benchmark before this campaign
began, with the metric that detects it named — and that metric is not in the scored set. It is also
the metric the chain's median-I work is closest in spirit to.

**`morans_mae` is therefore reported beside the correlation in everything below**, and whether it
should join the scored set is a separate question this document does not decide.

Three distinct statistics, and the campaign has been measuring only the first:

| statistic | what it asks | scored? |
|---|---|---|
| `morans_median_pred` vs `morans_median_gt` | is the **overall level** right | emitted, **not scored** |
| `morans_mae` | is **each gene's level** right | emitted, **not scored** |
| `morans_pearson` | is the **ranking across genes** right | **scored** |

---

## 4. What is computed

For stages **3** (decoded `mu`), **4** (sampled counts) and **4p** (Poisson of `mu`, the
emission-free ceiling), against `REF real counts`: `pearson`, `spearman`, `mae`, and both medians —
the five keys `_agreement` emits, by the same construction. Everything is on **rank-normalised**
values on both sides, which Q1's transform fix makes uniform.

---

## 5. The pre-registered reading

**(a) Does removing the emission move the metric?** `d r = r(4p) - r(4)`.

| band | criterion | reading |
|---|---|---|
| **MOVES IT** | `d r >= +0.15` | the emission programme targets the scored quantity; Q2/Q3 are worth their 4.3 h |
| **DOES NOT** | `d r <= +0.05` | removing the emission **entirely** barely moves the metric. The programme is aimed at the wrong quantity and the deficit lives where the chain has not looked |
| **PARTIAL** | in between | it moves it, but not enough to justify 4.3 h on its own |

`0.15` is roughly 40 % of the v25→SpatialZ gap (0.932 − 0.557 = 0.375). That scaling is **rough** —
§2 says this run's `r` is not the score — and the bands are on `d r` itself, not on a fraction.

> ⚠️ **CORRECTED after the run, and it does not change a verdict.** The gap that matters is against
> the **model-free copy floor**, `flanking_copy` = **0.9836** (`advisor_report.md` §5.1, tier-1,
> three seeds), not SpatialZ's 0.932. v25 is **−0.4262** below a baseline that copies a neighbouring
> real section. Scaling the bands to 0.375 made them **more lenient** than they should have been, so
> every verdict in `reports/q15_review.md` §2 stands and would only be firmer. The number was in the
> advisor report the whole time and I did not look it up.

**(b) Where does a perfect emission repair land?** `r(4p)` in absolute terms. This is the ceiling
argument applied to the scored statistic, and it is the more decisive of the two.

| band | criterion | reading |
|---|---|---|
| **CEILING IS HIGH** | `r(4p) >= 0.85` | a perfect emission repair approaches SpatialZ's territory on this statistic |
| **CEILING IS LOW** | `r(4p) <= 0.70` | **even a perfect emission repair leaves a large gap on the scored metric.** The emission is not where the benchmark deficit lives |
| **PARTIAL** | in between | |

**(c) Reported, not a criterion:** `morans_mae` at each stage, and the `median_pred` / `median_gt`
pair — which is the quantity the whole campaign has been measuring, now shown next to the one it is
judged by.

**Both datasets, and they may disagree.** Tier-1 has no median-I deficit and deep has one; if the
correlation behaves the same way on both, that is itself informative — it would mean the scored
metric is insensitive to the thing the two datasets differ in.

---

## 6. What would have to be true for this to fail

1. **`d r` is large and `r(4p)` is low.** Removing the emission helps the metric but nowhere near
   enough — reading (a) says go, reading (b) says it will not be sufficient. Both are reported and
   the disagreement is the finding, not an error.
2. **`r(4)` is already high** (say > 0.85) while the benchmark score is 0.557. Then the chain's
   single-section, panel-restricted reconstruction is not measuring what the benchmark measures, and
   §2's caveats are load-bearing rather than decorative. **This is a real possibility and it would
   invalidate the whole test**, which is why §2 lists the three differences explicitly.
3. **Fewer than 3 genes survive the NaN filter**, or the `I` vector has zero variance across genes —
   the evaluator's own guard. Then `pearson` is NaN and nothing is read.
4. **The panel and all-genes figures on deep disagree sharply.** The panel is the top 3.1 % by the
   real section's own `I`, so its `I` vector has compressed range and its correlation is attenuated.
   The all-genes figure governs.

Outcome 2 is the one that would waste the exercise, and it is checkable from the numbers themselves.

---

## 7. Cost, and what it does not do

**Free.** The per-gene `I` vectors already exist at every stage; the correlation is one call per
stage. No fit, no generation — the same `--load-model` reads that produced `a1_tier1.md` and
`a1_deep.md`.

It does **not** re-run the benchmark. It cannot tell you v25's score under any repair; it tells you
whether the quantity the chain has been moving is coupled to the quantity the score measures. **Q2
and Q3 do not start until this is read.**
