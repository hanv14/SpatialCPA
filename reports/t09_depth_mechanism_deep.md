# Why `marker_depth_r` splits both gates — the per-gene decomposition

Dataset **`deep_starmap`**, holdout **`paper_2_4_6`** — 115830 cells x 1017 genes, 2 LOSO folds (`section_3`, `section_5`). Up to 32 marker genes per fold (the `genes` column is what each fold actually had), 24 depth bins, text kNN 10, 2400 steps, seed 1. No refit: the audit's finished fits were restored from their checkpoints.

`marker_depth_r` is a **mean over marker genes** of a per-gene depth-profile correlation, so it decomposes exactly — asserted here against `section_scores` to 1e-06. That is why this test can carry weight the n = 2 fold margin cannot.

| gate | fold | genes | mean gain | improved | rho(gain, own trend) | rho(gain, nbr trend) | **partial** | p (1-sided) | p (2-sided) |
|---|---|---|---|---|---|---|---|---|---|
| `expr_mode` zinb-flow vs cross-mix | `section_3` | 32 | +0.2486 | 75% | +0.082 | +0.085 | **+0.178** | 0.171 | 0.346 |
| `expr_mode` zinb-flow vs cross-mix | `section_5` | 32 | +0.2711 | 78% | +0.411 | -0.247 | **-0.208** | 0.892 | 0.257 |
| `expr_mode` zinb-flow vs cross-mix | **pooled (2 folds)** | 29 | +0.2660 | 76% | +0.128 | -0.101 | **-0.041** | 0.613 | 0.834 |
| `text_emb_mode` medcpt vs lookup | `section_3` | 32 | +0.2568 | 81% | +0.016 | +0.187 | **+0.218** | 0.131 | 0.257 |
| `text_emb_mode` medcpt vs lookup | `section_5` | 32 | +0.1133 | 62% | +0.071 | -0.044 | **-0.096** | 0.714 | 0.604 |
| `text_emb_mode` medcpt vs lookup | **pooled (2 folds)** | 29 | +0.2162 | 76% | -0.064 | -0.080 | **-0.113** | 0.730 | 0.578 |

**How to read it.** `rho(gain, own trend)` is expected to be positive under *either* explanation — a gene with a flat real profile conditions its own Pearson r badly, so any arm difference has more room where the gradient is strong. The hypothesis is tested by the **partial** column: the gain against the gradient strength of a gene's *text neighbours*, holding the gene's own trend **and** its own bin-to-bin contrast fixed. Borrowing predicts that positive with a small permutation p; the conditioning artefact predicts it near zero whatever the text geometry says.

`p` shuffles the text vectors across the marker genes 2000 times and rebuilds the neighbourhood predictor, so it asks whether MedCPT's *actual* geometry matters or only the shape of the gradient distribution. The **one-sided** p tests the sign the hypothesis predicts (positive partial); the two-sided p is printed beside it so a reader can check that claim against the symmetric test.

**`expr_mode` is the control gate.** It does not touch the text channel. A partial correlation that appears on both gates is a property of the metric, not of text space; only a `text_emb_mode` effect *without* an `expr_mode` effect supports the hypothesis.

**The pooled row is not a third measurement.** The folds are two sections of one volume scored against one fit per arm, so pooling them averages noise out of the *same* measurement; it does not add a replicate. It is reported because of the power limit below, and never instead of the per-fold rows.

**Power — read a null result carefully.** Calibrated on planted data at 32 genes by `scripts/t09_depth_mechanism_calibration.py` (`reports/t09_depth_mechanism_calibration.md`): the false-positive rate is nominal (5-8% at p < 0.05, s.e. +-2.8) and the **confound world** — text space encodes the gradient, but the gain depends only on each gene's own gradient — rejects at **5-8%** too, i.e. the partial does strip it. But power against a real borrowing effect is only **18-28% per fold and 35-42% pooled**. A positive here is informative; a null is **not** evidence of absence, and does not settle what the three-seed run settles.

**One seed, and the 2 folds are not independent replicates** — they are sections of one volume scored against one fit per arm. Consistency across folds is evidence; it is not a second seed.

## Per-gene decomposition — what it establishes on `deep_starmap`

The diagnostic's own table answers the mechanism question. These are the three things its per-gene terms answer that the 2-fold margin cannot.

### 1. Is the advantage broad, or carried by a few genes?

| gate | fold | genes | mean gain | median gain | improved | sign p | Wilcoxon p |
|---|---|---|---|---|---|---|---|
| `expr_mode` | `section_3` | 32 | +0.2486 | +0.2583 | 24/32 | 7.0e-03 | 1.1e-03 |
| `expr_mode` | `section_5` | 32 | +0.2711 | +0.2652 | 25/32 | 2.1e-03 | 5.1e-04 |
| `text_emb_mode` | `section_3` | 32 | +0.2568 | +0.3113 | 26/32 | 5.4e-04 | 9.1e-04 |
| `text_emb_mode` | `section_5` | 32 | +0.1133 | +0.1275 | 20/32 | 2.2e-01 | 4.3e-02 |

**Per-gene gains within one fold are not independent** — one fitted model, one section, one marker set — so these p-values are optimistic and are not a substitute for a repeated-seed run. What they do establish is that the advantage is not carried by a handful of genes, which a mean over 2 folds cannot show either way.

### 2. Is *which* gene benefits reproducible across folds?

| gate | shared genes | Pearson | p | Spearman | p | improved in both |
|---|---|---|---|---|---|---|
| `expr_mode` | 29 | +0.645 | 0.000 | +0.599 | 0.001 | 19/29 |
| `text_emb_mode` | 29 | +0.259 | 0.175 | +0.343 | 0.069 | 16/29 |

This is the question the within-arm fold-spread column raised and could not answer. The two folds are held-out sections of one volume scored against the *same* pair of fitted models, so a correlation here says the per-gene pattern is a stable property of those models on unseen sections — **not** that it survives a new seed.

### 3. Do the two gates help the same genes? (read with the caveat)

| fold | shared genes | Pearson | p |
|---|---|---|---|
| `section_3` | 32 | +0.502 | 0.0034 |
| `section_5` | 32 | +0.532 | 0.0017 |

**Inflated by construction, so not a finding.** The two comparisons share their winning arm — the same fitted config serves as `zinb-flow` for one gate and `medcpt` for the other — so both gains carry the same `+r(shared arm)` term and would correlate even if the two losing arms were unrelated. Printed as a caution against reading the two gates as independent corroboration of each other.
