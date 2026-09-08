# Is the ceiling headroom stable? — bootstrap on `starmap_visual_cortex`

Holdout **`paper_2_4_6`** — 16527 cells x 28 genes. 400 replicates, each resampling **cells** (80%, without replacement) and **marker genes** (with replacement), recomputing split-half reliability, the ceiling `sqrt(R)`, the copy correlation and the headroom. Seed 1. No model, no fit.

**Dataset-level median headroom over the best copy: +0.1551 [+0.1075, +0.2186]** = 4.6x the 0.0335 envelope [3.2x, 6.5x].

| target | cells | markers | operational donor | headroom vs operational copy | headroom vs best copy |
|---|---|---|---|---|---|
| `section_3` | 4169 | 28 | `section_1` | +0.2021 [+0.1198, +0.3250] | +0.1520 [+0.0791, +0.2291] |
| `section_5` | 4110 | 28 | `section_3` | +0.1634 [+0.0896, +0.2512] | +0.1618 [+0.0893, +0.2458] |

**Two biases, both against this file's own conclusion.** R is estimated by splitting the *subsample*, so Spearman-Brown corrects to the subsample's size rather than the section's and the ceiling comes out slightly low. And the gene resample captures the spread of the mean over the *selected* markers, not the variance of the selection — real on `deep_starmap` (32 chosen from 1017) and absent on tier-1, where all 28 genes are markers, so tier-1's interval is the more complete of the two. Both understate tier-1's advantage rather than manufacture it.
