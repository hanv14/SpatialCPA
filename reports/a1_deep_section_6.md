# Chain diagnostic — where the spatial structure is lost (2400 steps)

`deep_starmap` / `paper_2_4_6`, `section_6` at z=114.8, 23204 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt`, 1017/1017 gene rows non-zero — from `runs/chain/shipped_deep.pt` |
| `expr_pca_dim` | 32 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | matched: 28654 generated -> 23204 kept (seed 1) |
| gene panel | top 32 of 1017 (3.1%) by Moran's I on the **real** side |
| real section | 23204 cells, z = 114.8 |

🚩 --target-z 114.8 is a boundary plane (within 0.5 median spacings of the stack's end): evidence there is one-sided, which T04 measured as a 20-35% reconstruction deficit (R3).

**Panel** (real-selected, 32 of 1017): `APOD`, `CLDN11`, `CRYM`, `DKK3`, `GABBR2`, `GFAP`, `GNG8`, `HPCAL4`, `LEF1`, `LSAMP`, `MBP`, `MEIS2`, `NEFM`, `NEUROD6`, `NPTX1`, `NRGN`, `NTNG1`, `PCP4`, `PENK`, `PPP1R1B`, `PRKCD`, `PTGDS`, `PVALB`, `RGS4`, `RGS9`, `SATB1`, `SATB2`, `SLC6A11`, `SPARC`, `TCF7L2`, `TMSB4X`, `VAMP1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

Every stage now carries Moran's I under **both** transforms. A ratio between two stages
must take both sides from the **same** column: the chain used to rank its count stages
and leave its mean-field and latent stages raw, which made every `counts / mu` retention
a cross-transform ratio (`reports/ceiling_review.md` §2). The **primary** column is the
one that stage's earlier artifacts recorded, and is marked `*`.

| stage                                             | median I (raw) | median I (rank) | p25 | p75 | channels |
|---------------------------------------------------|---|---|---|---|---|
| 1. prior h0 = GRF at generated xyz                | **+0.9163** * | +0.9093 | +0.9129 | +0.9191 | 64 |
| 2. latent h after the flow                        | **+0.7799** * | +0.7669 | +0.7629 | +0.8455 | 64 |
| 3. decoded mu (before sampling)                   | **+0.7330** * | +0.7570 | +0.7251 | +0.7514 | 32 |
| 4. sampled counts (rank-normalised)               | +0.1016 | **+0.1118** * | +0.0641 | +0.1980 | 32 |
| 4p. counts ~ Poisson(mu) — emission noise removed | +0.2741 | **+0.2510** * | +0.1705 | +0.3848 | 32 |
| REF real counts (rank-normalised)                 | +0.2809 | **+0.2819** * | +0.2605 | +0.3445 | 32 |
| REF real latent h1 = encoder(real counts)         | **+0.3189** * | +0.2928 | +0.3078 | +0.3473 | 64 |

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
| **real tissue** | +0.2819 | +0.3189 | **88.4%** | 1.310 | — |
| uncalibrated | +0.1118 | +0.7799 | **14.3%** | 1.295 | 1.310 |

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
| panel | 3. decoded mu | **+0.0493** | +0.3449 | 0.4559 | +0.7570 | +0.2819 | 32 |
| panel | 4. sampled counts | **+0.1522** | +0.1195 | 0.1890 | +0.1118 | +0.2819 | 32 |
| panel | 4p. Poisson(mu) — emission-free | **+0.4183** | +0.3721 | 0.1209 | +0.2510 | +0.2819 | 32 |
| all genes | 3. decoded mu | **+0.3155** | +0.5204 | 0.7013 | +0.7370 | +0.0165 | 1017 |
| all genes | 4. sampled counts | **+0.7021** | +0.7239 | 0.0264 | +0.0077 | +0.0165 | 1017 |
| all genes | 4p. Poisson(mu) — emission-free | **+0.8150** | +0.8267 | 0.0209 | +0.0167 | +0.0165 | 1017 |

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
| panel | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.8226** | +0.8195 .. +0.8306 (3) | +0.7489 | 0.1990 | 32 |
| panel | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.5560** | +0.5514 .. +0.5655 (3) | +0.3394 | 0.1366 | 32 |
| panel | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.8281** | +0.8235 .. +0.8314 (3) | +0.6642 | 0.0509 | 32 |
| panel | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.5175** | +0.5139 .. +0.5241 (3) | +0.3567 | 0.0908 | 32 |
| panel | A1a. counts ~ emission(mu \| h1) | **+0.0992** | +0.0932 .. +0.1052 (3) | +0.0447 | 0.2621 | 32 |
| panel | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.1522** | — | +0.1195 | 0.1890 | 32 |
| panel | A1n. permutation null (real counts shuffled) | **-0.0200** | -0.3650 .. +0.2540 (20) | -0.1045 | 0.3176 | 32 |
| all genes | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.9688** | — | +0.8826 | 0.0811 | 1017 |
| all genes | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.8531** | — | +0.7058 | 0.0219 | 1016 |
| all genes | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.9621** | — | +0.8520 | 0.0465 | 1017 |
| all genes | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8830** | — | +0.7666 | 0.0190 | 1017 |
| all genes | A1a. counts ~ emission(mu \| h1) | **+0.6934** | — | +0.6380 | 0.0358 | 1017 |
| all genes | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.7021** | — | +0.7239 | 0.0264 | 1017 |
| all genes | A1n. permutation null (real counts shuffled) | **+0.0064** | -0.0884 .. +0.0824 (20) | +0.0198 | 0.0433 | 1017 |

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
| panel | 20 | -0.0200 | 0.1922 | 0.0860 | -0.3650 .. +0.2540 | centred |
| all genes | 20 | +0.0064 | 0.0561 | 0.0251 | -0.0884 .. +0.0824 | centred |

### Which rung differences are real? — paired gene bootstrap

2000 replicates resampling **genes**, the same index applied to every rung so the
difference is paired. This carries the across-gene sampling error only; draw-to-draw
error is the `across seeds` column above and the two are never combined (§2b).

| scope | difference | point | 95% interval | genes | |
|---|---|---|---|---|---|
| panel | A1c - A1b | +0.2571 | +0.0848 .. +0.6005 | 32 | distinguishable |
| panel | A1c - A1b-t | -0.0055 | -0.1201 .. +0.1421 | 32 | **contains zero** |
| panel | A1c - A1b-p | +0.3051 | +0.1377 .. +0.5817 | 32 | distinguishable |
| panel | A1b - A1a | +0.4603 | +0.2313 .. +0.6049 | 32 | distinguishable |
| panel | 4 - A1a | +0.0470 | +0.0024 .. +0.0974 | 32 | distinguishable |
| panel | A1a - A1n | +0.2807 | -0.2863 .. +0.7575 | 32 | **contains zero** |
| all genes | A1c - A1b | +0.1157 | +0.0848 .. +0.1542 | 1016 | distinguishable |
| all genes | A1c - A1b-t | +0.0066 | -0.0020 .. +0.0171 | 1016 | **contains zero** |
| all genes | A1c - A1b-p | +0.0858 | +0.0609 .. +0.1148 | 1016 | distinguishable |
| all genes | A1b - A1a | +0.1598 | +0.1246 .. +0.1926 | 1016 | distinguishable |
| all genes | 4 - A1a | +0.0087 | -0.0071 .. +0.0243 | 1016 | **contains zero** |
| all genes | A1a - A1n | +0.7158 | +0.6324 .. +0.8010 | 1016 | distinguishable |

### R1-R3 - is the correlation spatial fidelity, or sparsity matching?

Computed on **all 1017 genes** — the gene set the
agreement table above says governs.

A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model
that matched only *which genes are sparse* would score on `paper_morans_pearson`
without reproducing any spatial fidelity. R3 controls for the tissue's own detection
rate and asks whether the model still orders genes correctly.

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.6989 | +0.7375 | +0.7819 |
| **R2** `corr(I_4, control)` | +0.8810 | +0.7585 | +0.7656 |
| **R3** partial `corr(I_4, I_real given control)` | +0.1610 | -0.0083 | -0.1016 |
| retained fraction of `r` = +0.7021 | 22.9% | -1.2% | -14.5% |

**UNINFORMATIVE — the control specifications disagree** — the 3 specifications span 0.374 against a 0.150 tolerance (`ladder_preregistration.md` §4, `flanking_copy_preregistration.md` §3).

### Stage 4 across whole generations

Each row is a **complete** regeneration — layout, prior, flow, decode, draw — not a
redraw at fixed cells. The A1 arms' `across seeds` column is emission noise alone; this
is the whole pipeline's, and it is the one the verdict leans on
(`null_band_preregistration.md` §2c).

| seed | cells | median I (rank) | r, panel | r, all genes |
|---|---|---|---|---|
| 1 | 23204 | +0.1118 | +0.1522 | +0.7021 |
| 2 | 23204 | +0.1156 | +0.1513 | +0.7039 |
| 3 | 23204 | +0.1144 | +0.1568 | +0.7075 |

**panel: r spans +0.1513 .. +0.1568 across 3 generations** (sd 0.0029). Any rung difference smaller than this is not resolved by a single generation.

**all genes: r spans +0.7021 .. +0.7075 across 3 generations** (sd 0.0027). Any rung difference smaller than this is not resolved by a single generation.

## `flanking_copy` — is the floor that beats us spatial fidelity?

⚠️ **This is the one test whose favourable outcome this project has an interest in.**
`flanking_copy_preregistration.md` §1 fixes two rules before any number here existed:
the default absent a clear result is **the outcome that does not suit us**, and **the
negative result is reported either way** — v25 loses to a model-free copy, and that
sentence goes in the paper whatever this block says.

Source: **section_5** at z=94.5 (28654 cells) — the nearest *training* section to section_6 at z=114.8 (23204 cells), emitted
verbatim, exactly as `bench3/selftest.py::make_probe` does. It is **not** at the
target's cells, so it is comparable to stage 4 and to no other rung (§2a).

| quantity | value |
|---|---|
| `r_flank` on 1017 genes | **+0.9798** |
| stage 4, same scope | +0.7021 |
| spearman / mae | +0.8892 / 0.0082 |

🚩 **0.9836 is tier-1's 28-gene figure and does not transfer here** (§2b). The test is
against whatever the copy scores on *this* scope.

### §5c — the positive control

`spatial_scramble` keeps every per-gene marginal and destroys only position: **+0.0064** (sd 0.0561, 20 seeds).

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
| 10 genes | **+0.5579** | +0.5233 .. +0.6093 | 56.9% | 20 |
| 25 genes | **+0.5663** | +0.5411 .. +0.6127 | 57.8% | 20 |
| 50 genes | **+0.5548** | +0.5076 .. +0.5868 | 56.6% | 20 |

The reading must be **stable across all three widths** (§5a); a result that appears at
one width and not the others is a stratum-width artefact.

### §3 — R3 for the copy, under all three controls

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.6989 | +0.7375 | +0.7819 |
| **R2** `corr(I_copy, control)` | +0.7038 | +0.7306 | +0.7790 |
| **R3** partial `corr(I_copy, I_real given control)` | +0.9534 | +0.9449 | +0.9317 |
| retained fraction of `r` = +0.9798 | 97.3% | 96.4% | 95.1% |

The three specifications span **0.022** against a 0.150
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

`F3_copy` = **+0.5579** (median over widths [10, 25, 50], span 0.011); `r_copy` = **+0.9798**; denominator **+0.4219**.

| rung | raw `r` | its own F3 (abundance alone) | **rescaled** |
|---|---|---|---|
| flanking_copy | +0.9798 | +0.5579 | **1.00** |
| A1c | +0.9688 | +0.5992 | **0.97** |
| A1b-t | +0.9621 | +0.6072 | **0.96** |
| A1b-p | +0.8830 | +0.6770 | **0.77** |
| A1b | +0.8531 | +0.6703 | **0.70** |
| 4 | +0.7021 | +0.6549 | **0.34** |
| A1a | +0.6934 | +0.6448 | **0.32** |
| A1n | -0.0231 | -0.0149 | **-1.38** |

The `own F3` column is **not** a denominator (§3). It is reported because it answers a
different question — how much of that rung's score is abundance — and because leaving it
out would print only the flattering statistic.

### Stage 4p — the emission-free ceiling

With the emission's noise removed from the model's **own** mean field, `I` = **+0.2510** against the real section's **+0.2819** (0.89x). That is the most any repair to `theta` or `pi` can reach on this fit — it bounds the emission-side work from above. **If it exceeds the tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and a repair to it alone cannot land the model on the tissue.

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
| `Var(shape)` — the latent-driven part | 0.54622 | 0.18685 |
| `Var(log s)` — the size-factor part | 0.01448 | 0.02329 |
| `2 Cov` | +0.00435 | +0.00902 |
| share of `Var(log mu)` from the latent (unbounded) | 97.3% | 88.7% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **97.4%** | **88.9%** |
| **`sd(log mu)` across cells** | **0.7456** | **0.4645** |

**`Var(log mu)` generated / real, median per gene: 2.494.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

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
| A1a'. mu decoded from h1                          | **+0.3005** | — | 32 | — | — | — |
| A1a. counts ~ emission(mu \| h1)                  | **+0.0457** | +0.0437 .. +0.0469 (3) | 32 | 0.95x | **-0.39** | DOES NOT RECOVER |
| A1b'. mu_oracle = kNN mean of real counts         | **+0.9132** | — | 32 | — | — | — |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)     | **+0.1815** | +0.1784 .. +0.1816 (3) | 32 | 0.73x | **+0.41** | UNINFORMATIVE |
| A1b-t. counts ~ NB(mu_oracle, model theta), pi=0  | **+0.2627** | +0.2602 .. +0.2666 (3) | 32 | 1.00x | **+0.89** | RECOVERS |
| A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.3006** | +0.2992 .. +0.3030 (3) | 32 | 0.73x | **+1.11** | RECOVERS |
| A1c. counts ~ Poisson(mu_oracle)   [model-free]   | **+0.4946** | +0.4937 .. +0.4997 (3) | 32 | 1.01x | **+2.25** | RECOVERS |
| A1n. permutation null (real counts shuffled)      | **+0.0002** | -0.0001 .. +0.0005 (3) | 32 | — | **-0.66** | DOES NOT RECOVER |

Anchors from this run: `I(model counts)` = **+0.1118**, `I(real counts)` = **+0.2819**, deficit = **+0.1701**.

**Three-seed stability** (`a1_escalation_preregistration.md` §1): an arm whose per-seed bands disagree reads **UNRESOLVED** regardless of its median.

| arm | per-seed R | bands | verdict |
|---|---|---|---|
| A1a | -0.38, -0.39, -0.40 | DOES NOT RECOVER | DOES NOT RECOVER |
| A1b | +0.39, +0.41, +0.41 | UNINFORMATIVE | UNINFORMATIVE |
| A1b-t | +0.87, +0.89, +0.91 | RECOVERS | RECOVERS |
| A1b-p | +1.11, +1.12, +1.10 | RECOVERS | RECOVERS |
| A1c | +2.25, +2.25, +2.28 | RECOVERS | RECOVERS |
| A1n | -0.66, -0.65, -0.66 | DOES NOT RECOVER | DOES NOT RECOVER |

### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss

Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were
built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact
within-realisation contrast.

| quantity | value |
|---|---|
| `L_total = I(A1c) - I(A1b)` | +0.3131 |
| `L_theta = I(A1c) - I(A1b-t)` | +0.2319 (**74.1%** of total) |
| `L_pi = I(A1c) - I(A1b-p)` | +0.1940 (**62.0%** of total) |
| additivity gap `abs(L_theta + L_pi - L_total)` | 0.1128 (criterion <= 0.0200) |
| multiplicative prediction of `I(A1b)` | +0.1597 against the measured +0.1815, gap 0.0218 |
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
| real counts (Poisson-deconvolved) — UPPER bound | **1.3038** | 32 |
| mu_oracle (kNN mean field) — LOWER bound | **1.0225** | 32 |
| mu decoded from h1 | **0.5984** | 32 |
| mu decoded from the generated h | **0.6876** | 32 |
| model counts (Poisson-deconvolved) | **1.3457** | 32 |
| A1c counts (Poisson-deconvolved) — estimator check | **1.0244** | 32 |

Read against the decoder's own figure in the Candidate 2 block above. `chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper bound falls below it, and **untested still** in between.

⚠️ How loose the lower bound is depends on how autocorrelated the field already is — a kNN mean destroys the variance of a field whose neighbours are unrelated and preserves it where they are not (the self-check measures 0.800 -> 0.279 on an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A lower bound below the decoder's figure is "untested still", never "refuted"** — only the upper bound can refute.
