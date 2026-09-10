# Chain diagnostic — where the spatial structure is lost (2400 steps)

`deep_starmap` / `paper_2_4_6`, `section_4` at z=73.5, 29544 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt`, 1017/1017 gene rows non-zero — from `runs/chain/shipped_deep.pt` |
| `expr_pca_dim` | 32 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | matched: 29842 generated -> 29544 kept (seed 1) |
| gene panel | top 32 of 1017 (3.1%) by Moran's I on the **real** side |
| real section | 29544 cells, z = 73.5 |

**Panel** (real-selected, 32 of 1017): `APOD`, `AQP1`, `C4B`, `CLDN11`, `DKK3`, `FOLR1`, `GABBR2`, `GFAP`, `LEF1`, `MBP`, `MEIS2`, `NEFH`, `NEUROD6`, `NPTX1`, `NRGN`, `NTNG1`, `OLFR558`, `PCP4`, `PENK`, `PPP1R1B`, `PRKCD`, `PTGDS`, `PVALB`, `SATB1`, `SATB2`, `SLC6A11`, `SPARC`, `TCF7L2`, `TMSB4X`, `TRF`, `TTR`, `VAMP1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

Every stage now carries Moran's I under **both** transforms. A ratio between two stages
must take both sides from the **same** column: the chain used to rank its count stages
and leave its mean-field and latent stages raw, which made every `counts / mu` retention
a cross-transform ratio (`reports/ceiling_review.md` §2). The **primary** column is the
one that stage's earlier artifacts recorded, and is marked `*`.

| stage                                             | median I (raw) | median I (rank) | p25 | p75 | channels |
|---------------------------------------------------|---|---|---|---|---|
| 1. prior h0 = GRF at generated xyz                | **+0.9272** * | +0.9211 | +0.9239 | +0.9300 | 64 |
| 2. latent h after the flow                        | **+0.7782** * | +0.7628 | +0.7590 | +0.8379 | 64 |
| 3. decoded mu (before sampling)                   | **+0.7451** * | +0.7699 | +0.7286 | +0.8000 | 32 |
| 4. sampled counts (rank-normalised)               | +0.0979 | **+0.1154** * | +0.0613 | +0.2030 | 32 |
| 4p. counts ~ Poisson(mu) — emission noise removed | +0.3030 | **+0.2594** * | +0.1825 | +0.3965 | 32 |
| REF real counts (rank-normalised)                 | +0.2669 | **+0.3123** * | +0.2654 | +0.3433 | 32 |
| REF real latent h1 = encoder(real counts)         | **+0.3171** * | +0.2688 | +0.2892 | +0.4276 | 64 |

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
| **real tissue** | +0.3123 | +0.3171 | **98.5%** | 1.277 | — |
| uncalibrated | +0.1154 | +0.7782 | **14.8%** | 1.268 | 1.277 |

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
| panel | 3. decoded mu | **+0.1254** | +0.3798 | 0.4586 | +0.7699 | +0.3123 | 32 |
| panel | 4. sampled counts | **+0.4055** | +0.4674 | 0.1961 | +0.1154 | +0.3123 | 32 |
| panel | 4p. Poisson(mu) — emission-free | **+0.6130** | +0.6342 | 0.1073 | +0.2594 | +0.3123 | 32 |
| all genes | 3. decoded mu | **+0.4332** | +0.5904 | 0.6953 | +0.7349 | +0.0180 | 1017 |
| all genes | 4. sampled counts | **+0.7306** | +0.6861 | 0.0296 | +0.0072 | +0.0180 | 1017 |
| all genes | 4p. Poisson(mu) — emission-free | **+0.8400** | +0.7929 | 0.0206 | +0.0160 | +0.0180 | 1017 |

## The ladder — what each rung holds at the truth

Every arm is drawn at the **real section's own cells**, so `pred_xy == gt_xy`. That is an
advantage stage 4 and bench3 do not have — they compare a generated cell set against the
real one, each on its own graph — so **a LOW rung here is conclusive and a high one is
permissive** (`ladder_preregistration.md` §2a).

Reference points on tier-1: the model-free copy floor `flanking_copy` = **0.9836**,
SpatialZ **0.932**, v25 shipped **0.5574**.

| gene set | rung | **pearson** | across seeds | spearman | mae | genes |
|---|---|---|---|---|---|---|
| panel | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.8439** | +0.8383 .. +0.8459 (3) | +0.7698 | 0.1891 | 32 |
| panel | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.6140** | +0.6081 .. +0.6255 (3) | +0.5436 | 0.1437 | 32 |
| panel | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.7967** | +0.7875 .. +0.8006 (3) | +0.6829 | 0.0582 | 32 |
| panel | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.5973** | +0.5954 .. +0.5991 (3) | +0.5495 | 0.1008 | 32 |
| panel | A1a. counts ~ emission(mu \| h1) | **+0.3797** | +0.3754 .. +0.3847 (3) | +0.4626 | 0.2659 | 32 |
| panel | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.4055** | — | +0.4674 | 0.1961 | 32 |
| panel | A1n. permutation null (real counts shuffled) | **+0.2578** | -0.0106 .. +0.4465 (3) | +0.1092 | 0.3313 | 32 |
| all genes | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.9699** | — | +0.8885 | 0.0820 | 1017 |
| all genes | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.8242** | — | +0.6837 | 0.0239 | 1017 |
| all genes | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.9522** | — | +0.8841 | 0.0469 | 1017 |
| all genes | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8681** | — | +0.7873 | 0.0203 | 1017 |
| all genes | A1a. counts ~ emission(mu \| h1) | **+0.6732** | — | +0.6138 | 0.0394 | 1017 |
| all genes | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.7306** | — | +0.6861 | 0.0296 | 1017 |
| all genes | A1n. permutation null (real counts shuffled) | **+0.0594** | — | +0.0794 | 0.0472 | 1017 |

### Is the null null? — per scope, on the null's CENTRE

The permutation arm shares one shuffle across every gene, so a single draw has a wide
spread and says nothing about the centre. A sound construction has `mean_r` at zero; the
spread is a resolution limit and lives in the bootstrap below, not here
(`null_band_preregistration.md` §2a). Fires on `|mean_r| > 2 * sd_r / sqrt(seeds)`.

| scope | seeds | mean r | sd | 2 x se | 2.5% .. 97.5% | verdict |
|---|---|---|---|---|---|---|
| panel | 20 | +0.0382 | 0.1784 | 0.0798 | -0.2487 .. +0.3132 | centred |
| all genes | 20 | -0.0035 | 0.0607 | 0.0271 | -0.1463 .. +0.0598 | centred |

### Which rung differences are real? — paired gene bootstrap

2000 replicates resampling **genes**, the same index applied to every rung so the
difference is paired. This carries the across-gene sampling error only; draw-to-draw
error is the `across seeds` column above and the two are never combined (§2b).

| scope | difference | point | 95% interval | genes | |
|---|---|---|---|---|---|
| panel | A1c - A1b | +0.2204 | +0.0826 .. +0.4442 | 32 | distinguishable |
| panel | A1c - A1b-t | +0.0492 | -0.0379 .. +0.1951 | 32 | **contains zero** |
| panel | A1c - A1b-p | +0.2468 | +0.1060 .. +0.4387 | 32 | distinguishable |
| panel | A1b - A1a | +0.2458 | +0.0317 .. +0.3881 | 32 | distinguishable |
| panel | 4 - A1a | +0.0258 | -0.0465 .. +0.0802 | 32 | **contains zero** |
| panel | A1a - A1n | -0.0668 | -0.3724 .. +0.3034 | 32 | **contains zero** |
| all genes | A1c - A1b | +0.1457 | +0.1088 .. +0.1927 | 1017 | distinguishable |
| all genes | A1c - A1b-t | +0.0177 | +0.0071 .. +0.0297 | 1017 | distinguishable |
| all genes | A1c - A1b-p | +0.1017 | +0.0746 .. +0.1350 | 1017 | distinguishable |
| all genes | A1b - A1a | +0.1510 | +0.1117 .. +0.1905 | 1017 | distinguishable |
| all genes | 4 - A1a | +0.0574 | +0.0372 .. +0.0762 | 1017 | distinguishable |
| all genes | A1a - A1n | +0.6137 | +0.5339 .. +0.6961 | 1017 | distinguishable |

### R1-R3 - is the correlation spatial fidelity, or sparsity matching?

Computed on **all 1017 genes** — the gene set the
agreement table above says governs.

A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model
that matched only *which genes are sparse* would score on `paper_morans_pearson`
without reproducing any spatial fidelity. R3 controls for the tissue's own detection
rate and asks whether the model still orders genes correctly.

| quantity | detection rate | log mean count |
|---|---|---|
| **R1** `corr(I_real, control)` - is the tissue's ordering a sparsity ordering? | +0.6787 | +0.7163 |
| **R2** `corr(I_model, control)` | +0.9027 | +0.7663 |
| **R3** partial `corr(I_4, I_real given control)` | +0.2797 | +0.0927 |
| retained fraction of `r(4)` = +0.7306 | 38.3% | 12.7% |

**UNINFORMATIVE — the control specifications disagree** — the two specifications differ by 0.395 against a 0.150 tolerance (`ladder_preregistration.md` §4).

### Stage 4 across whole generations

Each row is a **complete** regeneration — layout, prior, flow, decode, draw — not a
redraw at fixed cells. The A1 arms' `across seeds` column is emission noise alone; this
is the whole pipeline's, and it is the one the verdict leans on
(`null_band_preregistration.md` §2c).

| seed | cells | median I (rank) | r, panel | r, all genes |
|---|---|---|---|---|
| 1 | 29544 | +0.1154 | +0.4055 | +0.7306 |
| 2 | 29544 | +0.1120 | +0.3973 | +0.7340 |
| 3 | 29544 | +0.1087 | +0.4040 | +0.7282 |

**panel: r spans +0.3973 .. +0.4055 across 3 generations** (sd 0.0044). Any rung difference smaller than this is not resolved by a single generation.

**all genes: r spans +0.7282 .. +0.7340 across 3 generations** (sd 0.0029). Any rung difference smaller than this is not resolved by a single generation.

## `flanking_copy` — is the floor that beats us spatial fidelity?

⚠️ **This is the one test whose favourable outcome this project has an interest in.**
`flanking_copy_preregistration.md` §1 fixes two rules before any number here existed:
the default absent a clear result is **the outcome that does not suit us**, and **the
negative result is reported either way** — v25 loses to a model-free copy, and that
sentence goes in the paper whatever this block says.

Source: **section_3** at z=52.5 (29842 cells) — the nearest *training* section to section_4 at z=73.5 (29544 cells), emitted
verbatim, exactly as `bench3/selftest.py::make_probe` does. It is **not** at the
target's cells, so it is comparable to stage 4 and to no other rung (§2a).

| quantity | value |
|---|---|
| `r_flank` on 1017 genes | **+0.9886** |
| stage 4, same scope | +0.7306 |
| spearman / mae | +0.9225 / 0.0069 |

🚩 **0.9836 is tier-1's 28-gene figure and does not transfer here** (§2b). The test is
against whatever the copy scores on *this* scope.

### §5c — the positive control

`spatial_scramble` keeps every per-gene marginal and destroys only position: **+0.0214** (sd 0.0361, 5 seeds).

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
| 10 genes | **+0.5461** | +0.5057 .. +0.5932 | 55.2% | 20 |
| 25 genes | **+0.5145** | +0.4788 .. +0.5596 | 52.0% | 20 |
| 50 genes | **+0.5197** | +0.4788 .. +0.5783 | 52.6% | 20 |

The reading must be **stable across all three widths** (§5a); a result that appears at
one width and not the others is a stratum-width artefact.

### §3 — R3 for the copy, under all three controls

| quantity | detection | log mean | log variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` | +0.6787 | +0.7163 | +0.7690 |
| **R2** `corr(I_copy, control)` | +0.6481 | +0.6905 | +0.7481 |
| **R3** partial `corr(I_copy, I_real given control)` | +0.9788 | +0.9747 | +0.9676 |
| retained fraction of `r_flank` = +0.9886 | 99.0% | 98.6% | 97.9% |

The three specifications span **0.011** against a 0.150
tolerance, and the bands must be met by **every** control, not by their mean (§3-§4).

## **1. FLOOR IS GENUINE — the copy's score survives every control. The deficit is ours and the paper reports a negative result with NO benchmark claim**

§5's preconditions — independence from v25's own numbers, the positive control,
stability across F3's widths, replication across sections 2/4/6 and both datasets — are
**preconditions, not follow-ups**. One section of one dataset does not make a benchmark
claim, and this run is one section of one dataset.

### Stage 4p — the emission-free ceiling

With the emission's noise removed from the model's **own** mean field, `I` = **+0.2594** against the real section's **+0.3123** (0.83x). That is the most any repair to `theta` or `pi` can reach on this fit — it bounds the emission-side work from above. **If it exceeds the tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and a repair to it alone cannot land the model on the tissue.

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
| `Var(shape)` — the latent-driven part | 0.50394 | 0.23870 |
| `Var(log s)` — the size-factor part | 0.01360 | 0.01706 |
| `2 Cov` | +0.00411 | -0.00504 |
| share of `Var(log mu)` from the latent (unbounded) | 97.7% | 95.3% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **97.4%** | **93.3%** |
| **`sd(log mu)` across cells** | **0.7118** | **0.4797** |

**`Var(log mu)` generated / real, median per gene: 2.056.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

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
| A1a'. mu decoded from h1                          | **+0.3399** | — | 32 | — | — | — |
| A1a. counts ~ emission(mu \| h1)                  | **+0.0426** | +0.0395 .. +0.0433 (3) | 32 | 0.93x | **-0.37** | DOES NOT RECOVER |
| A1b'. mu_oracle = kNN mean of real counts         | **+0.9063** | — | 32 | — | — | — |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)     | **+0.1722** | +0.1701 .. +0.1799 (3) | 32 | 0.64x | **+0.29** | DOES NOT RECOVER |
| A1b-t. counts ~ NB(mu_oracle, model theta), pi=0  | **+0.2718** | +0.2716 .. +0.2774 (3) | 32 | 1.00x | **+0.79** | RECOVERS |
| A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.2911** | +0.2873 .. +0.2950 (3) | 32 | 0.62x | **+0.89** | RECOVERS |
| A1c. counts ~ Poisson(mu_oracle)   [model-free]   | **+0.5007** | +0.4994 .. +0.5039 (3) | 32 | 1.00x | **+1.96** | RECOVERS |
| A1n. permutation null (real counts shuffled)      | **+0.0002** | +0.0001 .. +0.0007 (3) | 32 | — | **-0.59** | DOES NOT RECOVER |

Anchors from this run: `I(model counts)` = **+0.1154**, `I(real counts)` = **+0.3123**, deficit = **+0.1969**.

**Three-seed stability** (`a1_escalation_preregistration.md` §1): an arm whose per-seed bands disagree reads **UNRESOLVED** regardless of its median.

| arm | per-seed R | bands | verdict |
|---|---|---|---|
| A1a | -0.37, -0.39, -0.37 | DOES NOT RECOVER | DOES NOT RECOVER |
| A1b | +0.29, +0.33, +0.28 | DOES NOT RECOVER, UNINFORMATIVE | **UNRESOLVED** (seeds straddle) |
| A1b-t | +0.79, +0.79, +0.82 | RECOVERS | RECOVERS |
| A1b-p | +0.91, +0.87, +0.89 | RECOVERS | RECOVERS |
| A1c | +1.95, +1.96, +1.97 | RECOVERS | RECOVERS |
| A1n | -0.58, -0.59, -0.59 | DOES NOT RECOVER | DOES NOT RECOVER |

### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss

Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were
built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact
within-realisation contrast.

| quantity | value |
|---|---|
| `L_total = I(A1c) - I(A1b)` | +0.3284 |
| `L_theta = I(A1c) - I(A1b-t)` | +0.2289 (**69.7%** of total) |
| `L_pi = I(A1c) - I(A1b-p)` | +0.2095 (**63.8%** of total) |
| additivity gap `abs(L_theta + L_pi - L_total)` | 0.1100 (criterion <= 0.0200) |
| multiplicative prediction of `I(A1b)` | +0.1580 against the measured +0.1722, gap 0.0142 |
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
| real counts (Poisson-deconvolved) — UPPER bound | **1.3699** | 32 |
| mu_oracle (kNN mean field) — LOWER bound | **1.0994** | 32 |
| mu decoded from h1 | **0.6035** | 32 |
| mu decoded from the generated h | **0.6725** | 32 |
| model counts (Poisson-deconvolved) | **1.4131** | 32 |
| A1c counts (Poisson-deconvolved) — estimator check | **1.1005** | 32 |

Read against the decoder's own figure in the Candidate 2 block above. `chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper bound falls below it, and **untested still** in between.

⚠️ How loose the lower bound is depends on how autocorrelated the field already is — a kNN mean destroys the variance of a field whose neighbours are unrelated and preserves it where they are not (the self-check measures 0.800 -> 0.279 on an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A lower bound below the decoder's figure is "untested still", never "refuted"** — only the upper bound can refute.
