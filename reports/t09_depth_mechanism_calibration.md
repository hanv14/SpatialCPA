# `t09_depth_mechanism` — does the test do what it claims?

Planted data at the diagnostic's own scale: 32 marker genes, text kNN 10, 60 replicates x 200 permutations, seed 11, rejecting at p < 0.05. Monte-Carlo s.e. on a 5% rate: +-2.8 points.

`null` and `text-carries-trend` are **false-positive** rates and should sit near 5%. The second is the confound the partial correlation exists to strip: text space encodes the depth gradient, but the per-gene gain depends only on each gene's **own** gradient. `borrowing` is the **power** — the rate at which a real neighbour effect is detected.

| test | folds | `null` | `text-carries-trend` | `borrowing` |
|---|---|---|---|---|
| two-sided | single fold | 8% | 5% | 18% |
| two-sided | pooled over 2 folds | 7% | 5% | 35% |
| one-sided | single fold | 8% | 8% | 28% |
| one-sided | pooled over 2 folds | 5% | 8% | 42% |

**What this licenses.** A significant partial on the real run is informative — the confound cannot manufacture one. A null is **not** evidence of absence at this power, and does not replace the repeated-seed run `specs/09` §3 requires.
