# Is the ceiling headroom stable? — bootstrap on `deep_starmap`

Holdout **`paper_2_4_6`** — 115830 cells x 1017 genes. 400 replicates, each resampling **cells** (80%, without replacement) and **marker genes** (with replacement), recomputing split-half reliability, the ceiling `sqrt(R)`, the copy correlation and the headroom. Seed 1. No model, no fit.

**Dataset-level median headroom over the best copy: +0.0160 [+0.0104, +0.0243]** = 0.5x the 0.0335 envelope [0.3x, 0.7x].

| target | cells | markers | operational donor | headroom vs operational copy | headroom vs best copy |
|---|---|---|---|---|---|
| `section_3` | 29842 | 32 | `section_5` | +0.0330 [+0.0152, +0.0585] | +0.0106 [+0.0066, +0.0160] |
| `section_5` | 28654 | 32 | `section_7` | +0.1376 [+0.0693, +0.2329] | +0.0215 [+0.0108, +0.0360] |

### Against `starmap_visual_cortex`

Difference in dataset-level median headroom: **-0.1389 [-0.2017, -0.0884]**, P(this > other) = **0.000**.

**The intervals are disjoint — the reversal holds.**

**Two biases, both against this file's own conclusion.** R is estimated by splitting the *subsample*, so Spearman-Brown corrects to the subsample's size rather than the section's and the ceiling comes out slightly low. And the gene resample captures the spread of the mean over the *selected* markers, not the variance of the selection — real on `deep_starmap` (32 chosen from 1017) and absent on tier-1, where all 28 genes are markers, so tier-1's interval is the more complete of the two. Both understate tier-1's advantage rather than manufacture it.
