# Chain diagnostic — where the spatial structure is lost (2400 steps)

`deep_starmap` / `paper_2_4_6`, `section_2` at z=31.5, 29842 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt`, 1017/1017 gene rows non-zero — from `runs/chain/shipped_deep.pt` |
| `expr_pca_dim` | 32 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | 🚩 **vacuous**: requested 30097, the layout produced only 29842, so nothing was subsampled and the arms are **NOT** density-matched |
| gene panel | top 32 of 1017 (3.1%) by Moran's I on the **real** side |
| real section | 30097 cells, z = 31.5 |

**Panel** (real-selected, 32 of 1017): `APOD`, `AQP1`, `CLDN11`, `CLIC6`, `DKK3`, `FOLR1`, `GABBR2`, `GFAP`, `GM5741`, `IGF2`, `IGFBP2`, `KL`, `LAMP5`, `LBP`, `LEF1`, `MBP`, `NEUROD6`, `NPTX1`, `NRGN`, `OLFR558`, `PENK`, `PPP1R1B`, `PRKCD`, `PTGDS`, `PVALB`, `SATB2`, `SPARC`, `SULF1`, `TMEM72`, `TRF`, `TTR`, `VAMP1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

Every stage now carries Moran's I under **both** transforms. A ratio between two stages
must take both sides from the **same** column: the chain used to rank its count stages
and leave its mean-field and latent stages raw, which made every `counts / mu` retention
a cross-transform ratio (`reports/ceiling_review.md` §2). The **primary** column is the
one that stage's earlier artifacts recorded, and is marked `*`.

| stage                                             | median I (raw) | median I (rank) | p25 | p75 | channels |
|---------------------------------------------------|---|---|---|---|---|
| 1. prior h0 = GRF at generated xyz                | **+0.9269** * | +0.9211 | +0.9233 | +0.9310 | 64 |
| 2. latent h after the flow                        | **+0.7866** * | +0.7724 | +0.7640 | +0.8439 | 64 |
| 3. decoded mu (before sampling)                   | **+0.7946** * | +0.8080 | +0.7380 | +0.8229 | 32 |
| 4. sampled counts (rank-normalised)               | +0.0688 | **+0.0788** * | +0.0172 | +0.1817 | 32 |
| 4p. counts ~ Poisson(mu) — emission noise removed | +0.2550 | **+0.2358** * | +0.0653 | +0.3534 | 32 |
| REF real counts (rank-normalised)                 | +0.3422 | **+0.3236** * | +0.2804 | +0.3994 | 32 |
| REF real latent h1 = encoder(real counts)         | **+0.2415** * | +0.2343 | +0.2182 | +0.3594 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

⚠️ **The retention column is OVERSTATED and is not a like-for-like ratio.** Its numerator
is a **rank-normalised** count stage and its denominator a **raw** latent stage; rank-
normalising a heavy-tailed field raises its Moran's I, so the denominator is too small.
Both arms carry the same bias, so the *comparison* between them stands and the
*percentages* do not (`reports/ceiling_review.md` §2).

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.3236 | +0.2415 | **134.0%** | 1.315 | — |
| uncalibrated | +0.0788 | +0.7866 | **10.0%** | 1.273 | 1.315 |

## Q1.5 — the scored statistic, beside the median

`paper_morans_pearson` is the **correlation across genes** between the model's per-gene
Moran's I vector and the tissue's. Every other number in this report is a **median**, and
a model can match the median exactly while getting every gene wrong. Reconstructed here
by bench3's own construction (`evaluate_paper.py::_agreement`, read from source): ranked
counts both sides, `k=10`, each side on its own graph, NaN genes dropped pairwise.

⚠️ **This is the statistic, not the score** — bench3 takes all shared genes and medians
over sections 2/4/6, and this is one section. Comparable **between stages**, not to a
published `paper_*` number (`reports/q15_preregistration.md` §2).

🚩 `mae` is emitted by the evaluator, documented there as **the metric that catches
over-smoothing**, and is **not** in the project's scored `METRICS` tuple.

| gene set | stage | **pearson** | spearman | mae | median pred | median gt | genes |
|---|---|---|---|---|---|---|---|
| panel | 3. decoded mu | **+0.0890** | +0.2082 | 0.4509 | +0.8080 | +0.3236 | 32 |
| panel | 4. sampled counts | **+0.1841** | +0.1701 | 0.2449 | +0.0788 | +0.3236 | 32 |
| panel | 4p. Poisson(mu) — emission-free | **+0.3615** | +0.3460 | 0.1603 | +0.2358 | +0.3236 | 32 |
| all genes | 3. decoded mu | **+0.4489** | +0.6305 | 0.6917 | +0.7268 | +0.0179 | 1017 |
| all genes | 4. sampled counts | **+0.6601** | +0.6576 | 0.0312 | +0.0069 | +0.0179 | 1017 |
| all genes | 4p. Poisson(mu) — emission-free | **+0.7787** | +0.7713 | 0.0230 | +0.0156 | +0.0179 | 1017 |

## The ladder — what each rung holds at the truth

Every arm is drawn at the **real section's own cells**, so `pred_xy == gt_xy`. That is an
advantage stage 4 and bench3 do not have — they compare a generated cell set against the
real one, each on its own graph — so **a LOW rung here is conclusive and a high one is
permissive** (`ladder_preregistration.md` §2a).

Reference points on tier-1: the model-free copy floor `flanking_copy` = **0.9836**,
SpatialZ **0.932**, v25 shipped **0.5574**.

The `A1n` rung carries the **20-seed** figure from the null block below, not its own
3-seed draw: they are the same construction, and at panel size a 3-seed draw of it has
spanned -0.25 to +0.26 between runs (`null_band_preregistration.md` §2a).

| gene set | rung | **pearson** | across seeds | spearman | mae | genes |
|---|---|---|---|---|---|---|
| panel | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.7877** | +0.7724 .. +0.7932 (3) | +0.6990 | 0.1604 | 32 |
| panel | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.3810** | +0.3778 .. +0.3840 (3) | +0.2793 | 0.1969 | 32 |
| panel | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.6127** | +0.6121 .. +0.6173 (3) | +0.4802 | 0.0915 | 32 |
| panel | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.3810** | +0.3744 .. +0.3889 (3) | +0.2881 | 0.1524 | 32 |
| panel | A1a. counts ~ emission(mu \| h1) | **+0.2083** | +0.2024 .. +0.2265 (3) | +0.1800 | 0.3017 | 32 |
| panel | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.1841** | — | +0.1701 | 0.2449 | 32 |
| panel | A1n. permutation null (real counts shuffled) | **+0.0242** | -0.3479 .. +0.2994 (20) | -0.2353 | 0.3551 | 32 |
| all genes | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.9642** | — | +0.8978 | 0.0826 | 1017 |
| all genes | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.7737** | — | +0.6737 | 0.0259 | 1017 |
| all genes | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.9406** | — | +0.8853 | 0.0476 | 1017 |
| all genes | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8135** | — | +0.7528 | 0.0225 | 1016 |
| all genes | A1a. counts ~ emission(mu \| h1) | **+0.6024** | — | +0.5372 | 0.0407 | 1017 |
| all genes | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.6601** | — | +0.6576 | 0.0312 | 1017 |
| all genes | A1n. permutation null (real counts shuffled) | **+0.0027** | -0.0701 .. +0.0869 (20) | +0.0136 | 0.0476 | 1017 |

### Is the null null? — per scope, on the null's CENTRE

The permutation arm shares one shuffle across every gene, so a single draw has a wide
spread and says nothing about the centre. A sound construction has `mean_r` at zero; the
spread is a resolution limit and lives in the bootstrap below, not here
(`null_band_preregistration.md` §2a). Fires on `|mean_r| > 2 * sd_r / sqrt(seeds)`.

**This is the same construction as the `A1n` rung above and as `spatial_scramble` in the
`flanking_copy` block** — permute the real section's cells, correlate against its own
per-gene `I`. All three now run at the same seed count so they cannot read as three
separate findings (§2a-bis).

| scope | seeds | mean r | sd | 2 x se | 2.5% .. 97.5% | verdict |
|---|---|---|---|---|---|---|
| panel | 20 | +0.0242 | 0.1988 | 0.0889 | -0.3479 .. +0.2994 | centred |
| all genes | 20 | +0.0027 | 0.0517 | 0.0231 | -0.0701 .. +0.0869 | centred |

### Which rung differences are real? — paired gene bootstrap

2000 replicates resampling **genes**, the same index applied to every rung so the
difference is paired. This carries the across-gene sampling error only; draw-to-draw
error is the `across seeds` column above and the two are never combined (§2b).

| scope | difference | point | 95% interval | genes | |
|---|---|---|---|---|---|
| panel | A1c - A1b | +0.3946 | +0.1780 .. +0.7250 | 32 | distinguishable |
| panel | A1c - A1b-t | +0.1597 | +0.0279 .. +0.3924 | 32 | distinguishable |
| panel | A1c - A1b-p | +0.3980 | +0.1975 .. +0.6921 | 32 | distinguishable |
| panel | A1b - A1a | +0.1754 | -0.0402 .. +0.3137 | 32 | **contains zero** |
| panel | 4 - A1a | -0.0183 | -0.0877 .. +0.0468 | 32 | **contains zero** |
| panel | A1a - A1n | +0.0552 | -0.3002 .. +0.3895 | 32 | **contains zero** |
| all genes | A1c - A1b | +0.1896 | +0.1367 .. +0.2509 | 1016 | distinguishable |
| all genes | A1c - A1b-t | +0.0236 | +0.0116 .. +0.0365 | 1016 | distinguishable |
| all genes | A1c - A1b-p | +0.1507 | +0.1059 .. +0.1998 | 1016 | distinguishable |
| all genes | A1b - A1a | +0.1723 | +0.1333 .. +0.2103 | 1016 | distinguishable |
| all genes | 4 - A1a | +0.0578 | +0.0306 .. +0.0843 | 1016 | distinguishable |
| all genes | A1a - A1n | +0.6423 | +0.5456 .. +0.7355 | 1016 | distinguishable |

### R1-R3 - is the correlation spatial fidelity, or sparsity matching?

Computed on **all 1017 genes** — the gene set the
agreement table above says governs.

A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model
that matched only *which genes are sparse* would score on `paper_morans_pearson`
without reproducing any spatial fidelity. R3 controls for the tissue's own detection
rate and asks whether the model still orders genes correctly.

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.6179 | +0.6804 | +0.7334 |
| **R2** `corr(I_4, control)` | +0.9089 | +0.7485 | +0.7537 |
| **R3** partial `corr(I_4, I_real given control)` | +0.1897 | -0.0128 | -0.1174 |
| retained fraction of `r` = +0.6601 | 28.7% | -1.9% | -17.8% |

**UNINFORMATIVE — the control specifications disagree** — the 3 specifications span 0.465 against a 0.150 tolerance (`ladder_preregistration.md` §4, `flanking_copy_preregistration.md` §3).

### Stage 4 across whole generations

Each row is a **complete** regeneration — layout, prior, flow, decode, draw — not a
redraw at fixed cells. The A1 arms' `across seeds` column is emission noise alone; this
is the whole pipeline's, and it is the one the verdict leans on
(`null_band_preregistration.md` §2c).

| seed | cells | median I (rank) | r, panel | r, all genes |
|---|---|---|---|---|
| 1 | 29842 | +0.0788 | +0.1841 | +0.6601 |
| 2 | 29842 | +0.0739 | +0.1730 | +0.6595 |
| 3 | 29842 | +0.0768 | +0.1889 | +0.6593 |

**panel: r spans +0.1730 .. +0.1889 across 3 generations** (sd 0.0082). Any rung difference smaller than this is not resolved by a single generation.

**all genes: r spans +0.6593 .. +0.6601 across 3 generations** (sd 0.0004). Any rung difference smaller than this is not resolved by a single generation.

## `flanking_copy` — is the floor that beats us spatial fidelity?

⚠️ **This is the one test whose favourable outcome this project has an interest in.**
`flanking_copy_preregistration.md` §1 fixes two rules before any number here existed:
the default absent a clear result is **the outcome that does not suit us**, and **the
negative result is reported either way** — v25 loses to a model-free copy, and that
sentence goes in the paper whatever this block says.

Source: **section_3** at z=52.5 (29842 cells) — the nearest *training* section to section_2 at z=31.5 (30097 cells), emitted
verbatim, exactly as `bench3/selftest.py::make_probe` does. It is **not** at the
target's cells, so it is comparable to stage 4 and to no other rung (§2a).

| quantity | value |
|---|---|
| `r_flank` on 1017 genes | **+0.9869** |
| stage 4, same scope | +0.6601 |
| spearman / mae | +0.9293 / 0.0070 |

🚩 **0.9836 is tier-1's 28-gene figure and does not transfer here** (§2b). The test is
against whatever the copy scores on *this* scope.

### §5c — the positive control

`spatial_scramble` keeps every per-gene marginal and destroys only position: **+0.0027** (sd 0.0517, 20 seeds).

§6 predicted this scores ~0 **before it ran**. If it does, the metric plainly does
respond to position and the maximal defensible critique is already the narrow one:
*among predictions carrying realistic within-gene autocorrelation, the across-gene
correlation is dominated by per-gene abundance*. A broader sentence than that cannot
be written later. If it scores high instead, that is the headline result.

### §5a — F3, the abundance-matched relabelling (the DECISIVE instrument)

Genes are ranked by the target's detection rate, cut into strata, and the *pairing*
between predicted and real genes is permuted **within** each stratum. The abundance-`I`
relationship survives exactly; gene-specific spatial identity does not. R3 below is the
corroborating instrument, not the other way round — it has a control specification we
chose and this does not.

| stratum width | median r | 2.5% .. 97.5% | as a fraction of `r_flank` | seeds |
|---|---|---|---|---|
| 10 genes | **+0.4824** | +0.4547 .. +0.5387 | 48.9% | 20 |
| 25 genes | **+0.4762** | +0.4322 .. +0.5086 | 48.3% | 20 |
| 50 genes | **+0.4751** | +0.4382 .. +0.5388 | 48.1% | 20 |

The reading must be **stable across all three widths** (§5a); a result that appears at
one width and not the others is a stratum-width artefact.

### §3 — R3 for the copy, under all three controls

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.6179 | +0.6804 | +0.7334 |
| **R2** `corr(I_copy, control)` | +0.6515 | +0.7042 | +0.7534 |
| **R3** partial `corr(I_copy, I_real given control)` | +0.9769 | +0.9722 | +0.9653 |
| retained fraction of `r` = +0.9869 | 99.0% | 98.5% | 97.8% |

The three specifications span **0.012** against a 0.150
tolerance, and the bands must be met by **every** control, not by their mean (§3-§4).

## **1. FLOOR IS GENUINE — the copy's score survives every control. The deficit is ours and the paper reports a negative result with NO benchmark claim**

§5's preconditions — independence from v25's own numbers, the positive control,
stability across F3's widths, replication across sections 2/4/6 and both datasets — are
**preconditions, not follow-ups**. One section of one dataset does not make a benchmark
claim, and this run is one section of one dataset.

## The abundance floor — all genes

⚠️ **The raw comparison is the result and is first.** This rescaling moves v25's headline
in a direction that flatters it, so `abundance_floor_preregistration.md` §1 fixes that the
raw figures are reported first and always, and that every failure mode below returns to
them. **It does not reopen the closeability question** — the ladder answered that and this
changes only how the negative result is expressed.

F3 applied to a rung's own per-gene `I` gives what that rung would score **from abundance
alone**. The denominator is the **copy's** floor for every rung (§3), never the rung's own,
so no rung can improve its own scale.

`F3_copy` = **+0.4762** (median over widths [10, 25, 50], span 0.007); `r_copy` = **+0.9869**; denominator **+0.5107**.

| rung | raw `r` | its own F3 (abundance alone) | **rescaled** |
|---|---|---|---|
| flanking_copy | +0.9869 | +0.4762 | **1.00** |
| A1c | +0.9642 | +0.5026 | **0.96** |
| A1b-t | +0.9406 | +0.5361 | **0.91** |
| A1b-p | +0.8135 | +0.5945 | **0.66** |
| A1b | +0.7737 | +0.5864 | **0.58** |
| 4 | +0.6601 | +0.6032 | **0.36** |
| A1a | +0.6024 | +0.4989 | **0.25** |
| A1n | -0.0418 | -0.0294 | **-1.01** |

The `own F3` column is **not** a denominator (§3). It is reported because it answers a
different question — how much of that rung's score is abundance — and because leaving it
out would print only the flattering statistic.

### Stage 4p — the emission-free ceiling

With the emission's noise removed from the model's **own** mean field, `I` = **+0.2358** against the real section's **+0.3236** (0.73x). That is the most any repair to `theta` or `pi` can reach on this fit — it bounds the emission-side work from above. **If it exceeds the tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and a repair to it alone cannot land the model on the tissue.

## Candidate 2 — is `mu`'s dynamic range the size factor?

`mu = link(MLP_mu(u)) * size_factor`, so `log mu = shape + log s` and the
variance splits exactly. Per gene, medians over the panel above.

The **real latent** column is the same decoder and the same size head applied to
`h1 = encoder(real counts)` instead of to the flow's sample: the two columns differ
in the latent and in nothing else. It is the matched tissue-side quantity the
record has been quoting as "tissue's 1.213" without a source.

| quantity | generated `h` | real latent `h1` |
|---|---|---|
| genes decomposed | 32 | 32 |
| `Var(shape)` — the latent-driven part | 0.52401 | 0.23374 |
| `Var(log s)` — the size-factor part | 0.01468 | 0.01578 |
| `2 Cov` | -0.00226 | -0.01942 |
| share of `Var(log mu)` from the latent (unbounded) | 98.3% | 100.9% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **97.3%** | **93.7%** |
| **`sd(log mu)` across cells** | **0.7221** | **0.4663** |

**`Var(log mu)` generated / real, median per gene: 1.915.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

## A1 — emission ablation, at the real cells

Every arm is drawn at the ground truth's own positions, so the kNN graph, the cell
count and the density are identical to `REF real counts` and to each other. The
layout, the prior and the flow are held out of it entirely: only what the counts
were drawn from varies.

**Read `reports/a1_preregistration.md` before reading these numbers.** The outcome
table, the thresholds, the level guard and the two stated asymmetries were committed
before the run.

| arm                                               | median I | across seeds | ch | level | R | band |
|---------------------------------------------------|---|---|---|---|---|---|
| A1a'. mu decoded from h1                          | **+0.3330** | — | 32 | — | — | — |
| A1a. counts ~ emission(mu \| h1)                  | **+0.0218** | +0.0210 .. +0.0222 (3) | 32 | 0.72x | **-0.23** | DOES NOT RECOVER |
| A1b'. mu_oracle = kNN mean of real counts         | **+0.9267** | — | 32 | — | — | — |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)     | **+0.1143** | +0.1134 .. +0.1178 (3) | 32 | 0.53x | **+0.14** | DOES NOT RECOVER |
| A1b-t. counts ~ NB(mu_oracle, model theta), pi=0  | **+0.2518** | +0.2495 .. +0.2530 (3) | 32 | 1.00x | **+0.71** | RECOVERS |
| A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.2037** | +0.1987 .. +0.2047 (3) | 32 | 0.54x | **+0.51** | UNINFORMATIVE |
| A1c. counts ~ Poisson(mu_oracle)   [model-free]   | **+0.4772** | +0.4745 .. +0.4851 (3) | 32 | 1.00x | **+1.63** | RECOVERS |
| A1n. permutation null (real counts shuffled)      | **-0.0006** | -0.0009 .. -0.0002 (3) | 32 | — | **-0.32** | DOES NOT RECOVER |

Anchors from this run: `I(model counts)` = **+0.0788**, `I(real counts)` = **+0.3236**, deficit = **+0.2448**.

**Three-seed stability** (`a1_escalation_preregistration.md` §1): an arm whose per-seed bands disagree reads **UNRESOLVED** regardless of its median.

| arm | per-seed R | bands | verdict |
|---|---|---|---|
| A1a | -0.23, -0.23, -0.24 | DOES NOT RECOVER | DOES NOT RECOVER |
| A1b | +0.14, +0.14, +0.16 | DOES NOT RECOVER | DOES NOT RECOVER |
| A1b-t | +0.71, +0.70, +0.71 | RECOVERS, UNINFORMATIVE | **UNRESOLVED** (seeds straddle) |
| A1b-p | +0.51, +0.51, +0.49 | UNINFORMATIVE | UNINFORMATIVE |
| A1c | +1.62, +1.66, +1.63 | RECOVERS | RECOVERS |
| A1n | -0.32, -0.33, -0.32 | DOES NOT RECOVER | DOES NOT RECOVER |

### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss

Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were
built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact
within-realisation contrast.

| quantity | value |
|---|---|
| `L_total = I(A1c) - I(A1b)` | +0.3629 |
| `L_theta = I(A1c) - I(A1b-t)` | +0.2254 (**62.1%** of total) |
| `L_pi = I(A1c) - I(A1b-p)` | +0.2734 (**75.4%** of total) |
| additivity gap `abs(L_theta + L_pi - L_total)` | 0.1359 (criterion <= 0.0200) |
| multiplicative prediction of `I(A1b)` | +0.1075 against the measured +0.1143, gap 0.0068 |
| **verdict** | **not decomposable** — the two interact |

🔎 **Instrument self-check**: `A1b-t`'s level ratio is **1.00x**. `theta` cannot move the mean, so anything away from 1.00x means that arm is not what it claims and the verdict above does not stand.

### N2 — the tissue's own `sd(log mu)`, two model-free routes

Both use `sd(log mu) = sqrt(log(1 + CV^2))` — an approximation — so neither
needs a pseudocount, and **neither passes through the decoder**, which is what
the gate in `emission_repair_options.md` §8.3 could not manage. The kNN mean
field shrinks variance, so its figure is a **lower** bound; the
Poisson-deconvolved one counts any tissue over-dispersion as signal, so it is an
**upper** bound. Together they bracket the tissue. Estimator fixed in
`a1_escalation_preregistration.md` §3.

| quantity | `sd(log mu)` | genes |
|---|---|---|
| real counts (Poisson-deconvolved) — UPPER bound | **1.5626** | 32 |
| mu_oracle (kNN mean field) — LOWER bound | **1.3114** | 32 |
| mu decoded from h1 | **0.6126** | 32 |
| mu decoded from the generated h | **0.7034** | 32 |
| model counts (Poisson-deconvolved) | **1.5346** | 32 |
| A1c counts (Poisson-deconvolved) — estimator check | **1.3115** | 32 |

Read against the decoder's own figure in the Candidate 2 block above. `chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper bound falls below it, and **untested still** in between.

⚠️ How loose the lower bound is depends on how autocorrelated the field already is — a kNN mean destroys the variance of a field whose neighbours are unrelated and preserves it where they are not (the self-check measures 0.800 -> 0.279 on an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A lower bound below the decoder's figure is "untested still", never "refuted"** — only the upper bound can refute.
