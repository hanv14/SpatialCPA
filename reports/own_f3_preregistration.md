# Pre-registration — the own-F3 ratio

**Committed before the statistic is computed or used as a claim.** It is derived from two quantities
already pre-registered and printed (`flanking_copy_preregistration.md` §5a; `abundance_floor_
preregistration.md` §3), but the **ratio and its reading are new and post-hoc**, and this project's
record now contains three general claims written from one dataset. That it runs *against* our
interest reduces the reason for discipline; it does not remove it.

## 1. The statistic

For a rung `x`:

```
own_F3(x) / r(x)
```

the share of that rung's score which survives destroying gene identity within abundance strata —
*how much of this prediction's score comes from matching per-gene abundance rather than reproducing
which genes are spatially structured*.

**The reading is stated in terms of the DIFFERENCE between rungs, never in absolute terms**, for the
reason in §3.

## 2. What the deep round already showed, quoted as the motivation and not as the result

| own F3 / own raw | s2 | s4 | s6 |
|---|---|---|---|
| `flanking_copy` | 48 % | 51 % | 57 % |
| `A1b` | 76 % | 78 % | 79 % |
| **stage 4** | **91 %** | **89 %** | **93 %** |

Read naively: v25's score is ~90 % abundance-reproducible and a copy's is ~50 %. That is a far
harder negative result than "0.73 against 0.99", and it is the honest counterweight to a rescaling
that reports 0.34–0.47. §3 is why it may not yet be written down.

## 3. The floor effect — the reason this needs a test and not just a table

A rung with a **lower** raw `r` has less room above its own floor, so the ratio may rise as `r`
falls for reasons that have nothing to do with abundance. The deep table is consistent with exactly
that: the ratio increases monotonically as the raw score decreases, down the whole ladder.

**The discriminating test, fixed here before it is run:**

> **T1 — matched-`r` separation.** Degrade `flanking_copy`'s own per-gene `I` vector by adding
> independent noise, in a sweep, until its raw `r` matches stage 4's on the same section and gene
> set. Then compare that degraded copy's own-F3 ratio against stage 4's.
>
> * If the degraded copy's ratio rises to **within 0.10** of stage 4's, the ratio is a **floor
>   effect** and says nothing about abundance. **The statistic is withdrawn.**
> * If the degraded copy's ratio stays **more than 0.20 below** stage 4's at matched `r`, the
>   difference is about abundance and not about headroom.
> * Between the two: **UNINFORMATIVE**, and the statistic is withdrawn.

The degradation must be **spatially independent** noise added to the per-gene `I` vector — the same
kind of insult the emission applies — at ≥ 10 noise levels and ≥ 20 seeds per level, with the
matched level chosen by `|r_degraded − r_4|` and its own interval reported.

**T1 is a precondition, not a follow-up.** Absent it, the deep table stays where it is and is not
described in words.

## 4. Corroboration required alongside T1

**T2 — an independent route to the same claim.** `R2(I_pred, detection)` measures abundance-coupling
without any relabelling. If the own-F3 ratio is measuring abundance, `R2` must order the rungs the
same way. The deep round has `R2(I_4, detection)` = **0.903 / 0.881 / 0.909** against the copy's
**0.648 / 0.704 / 0.652** and the tissue's own **0.679 / 0.699 / 0.618** — consistent, and to be
reported beside T1 rather than instead of it.

**T3 — the tissue's own ratio as the reference point.** The question "is v25's score unusually
abundance-driven" needs a scale, and the tissue is it: `own_F3(real) / r(real, real)` is
degenerate (`r = 1`), so the reference is the **copy**, which is the closest a prediction gets to
the tissue. This is stated so that "90 % is high" is anchored to a measured comparator and not to
intuition.

## 5. Scope, fixed now

**Deep only.** Tier-1's `F3_copy` is indistinguishable from the permutation null on its own interval
(`panel_regime.md`), so no ratio there has a denominator worth taking. **If T1 passes, the paper
says the finding is `deep_starmap` only and names the reason** — the gene panel, not the method.
That sentence is written here so it cannot be omitted later.

## 6. Predictions, recorded now

1. **T1 passes** — the degraded copy's ratio stays well below stage 4's at matched `r`. Moderate
   confidence only: the monotone trend down the ladder is real and I cannot presently distinguish
   its two causes.
2. `R2` orders the rungs as the own-F3 ratio does (T2). High confidence — it already does on three
   sections.
3. The degraded copy's ratio **does** rise with degradation, i.e. there is a genuine floor effect
   of non-zero size. If it does not rise at all, my §3 concern was unfounded and that is recorded
   as such.

## 7. What the paper may say if T1 passes

> On `deep_starmap`, roughly 90 % of v25's `paper_morans_pearson` is reproducible by matching
> per-gene abundance alone, against roughly 50 % for a model-free copy of a neighbouring section
> — at matched raw score. v25's per-gene spatial ordering is more abundance-determined than the
> tissue's own.

**May not say**: any absolute reading of the ratio without the matched-`r` control; anything about
tier-1; or anything that treats the ratio as the score.
