# Chain diagnostic — where the spatial structure is lost (2400 steps)

`starmap_visual_cortex` / `paper_2_4_6`, `section_6` at z=74.0, 4110 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt`, 28/28 gene rows non-zero — from `runs/chain/shipped_tier1.pt` |
| `expr_pca_dim` | 28 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | 🚩 **vacuous**: requested 4162, the layout produced only 4110, so nothing was subsampled and the arms are **NOT** density-matched |
| gene panel | 🚩 **vacuous**: `--top-k 32` >= the panel's 28 genes, so all 28 were kept and **no selection took place** |
| real section | 4162 cells, z = 74.0 |

**Panel** (**no selection** — all 28 genes): `Slc17a7`, `Mgp`, `Gad1`, `Nov`, `Rasgrf2`, `Rorb`, `Cux2`, `Plcxd2`, `Sulf2`, `Ctgf`, `Pcp4`, `Sema3e`, `Npy`, `Sst`, `Pvalb`, `Vip`, `Calb2`, `Cck`, `Reln`, `Fos`, `Egr1`, `Prok2`, `Egr2`, `Bdnf`, `Gja1`, `Ctss`, `Mbp`, `Flt1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

Every stage now carries Moran's I under **both** transforms. A ratio between two stages
must take both sides from the **same** column: the chain used to rank its count stages
and leave its mean-field and latent stages raw, which made every `counts / mu` retention
a cross-transform ratio (`reports/ceiling_review.md` §2). The **primary** column is the
one that stage's earlier artifacts recorded, and is marked `*`.

| stage                                             | median I (raw) | median I (rank) | p25 | p75 | channels |
|---------------------------------------------------|---|---|---|---|---|
| 1. prior h0 = GRF at generated xyz                | **+0.9372** * | +0.9320 | +0.9294 | +0.9459 | 64 |
| 2. latent h after the flow                        | **+0.8012** * | +0.7719 | +0.7180 | +0.8670 | 64 |
| 3. decoded mu (before sampling)                   | **+0.8175** * | +0.8327 | +0.7439 | +0.8508 | 28 |
| 4. sampled counts (rank-normalised)               | +0.3349 | **+0.4072** * | +0.3330 | +0.5000 | 28 |
| 4p. counts ~ Poisson(mu) — emission noise removed | +0.8169 | **+0.8309** * | +0.7942 | +0.8718 | 28 |
| REF real counts (rank-normalised)                 | +0.2846 | **+0.4102** * | +0.3158 | +0.4835 | 28 |
| REF real latent h1 = encoder(real counts)         | **+0.5631** * | +0.4803 | +0.4240 | +0.6684 | 64 |

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
| **real tissue** | +0.4102 | +0.5631 | **72.8%** | 1.634 | — |
| uncalibrated | +0.4072 | +0.8012 | **50.8%** | 1.814 | 1.634 |

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
| panel | 3. decoded mu | **+0.3663** | +0.4685 | 0.4160 | +0.8327 | +0.4102 | 28 |
| panel | 4. sampled counts | **+0.5826** | +0.6004 | 0.0983 | +0.4072 | +0.4102 | 28 |
| panel | 4p. Poisson(mu) — emission-free | **+0.3746** | +0.4707 | 0.4138 | +0.8309 | +0.4102 | 28 |

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
| panel | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.8704** | +0.8693 .. +0.8705 (3) | +0.8615 | 0.4304 | 28 |
| panel | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.8355** | +0.8351 .. +0.8365 (3) | +0.8621 | 0.0586 | 28 |
| panel | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.8349** | +0.8349 .. +0.8363 (3) | +0.8621 | 0.0587 | 28 |
| panel | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8705** | +0.8689 .. +0.8712 (3) | +0.8560 | 0.4303 | 28 |
| panel | A1a. counts ~ emission(mu \| h1) | **+0.6771** | +0.6680 .. +0.6843 (3) | +0.6886 | 0.1292 | 28 |
| panel | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.5826** | — | +0.6004 | 0.0983 | 28 |
| panel | A1n. permutation null (real counts shuffled) | **+0.0629** | -0.3242 .. +0.5193 (20) | +0.1270 | 0.4095 | 28 |

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
| panel | 20 | +0.0629 | 0.2811 | 0.1257 | -0.3242 .. +0.5193 | centred |

### Which rung differences are real? — paired gene bootstrap

2000 replicates resampling **genes**, the same index applied to every rung so the
difference is paired. This carries the across-gene sampling error only; draw-to-draw
error is the `across seeds` column above and the two are never combined (§2b).

| scope | difference | point | 95% interval | genes | |
|---|---|---|---|---|---|
| panel | A1c - A1b | +0.0353 | -0.0705 .. +0.1461 | 28 | **contains zero** |
| panel | A1c - A1b-t | +0.0354 | -0.0704 .. +0.1464 | 28 | **contains zero** |
| panel | A1c - A1b-p | -0.0002 | -0.0008 .. +0.0002 | 28 | **contains zero** |
| panel | A1b - A1a | +0.1508 | +0.0052 .. +0.3177 | 28 | distinguishable |
| panel | 4 - A1a | -0.1017 | -0.2334 .. -0.0074 | 28 | distinguishable |
| panel | A1a - A1n | +0.4555 | +0.1780 .. +0.7578 | 28 | distinguishable |

### R1-R3 - is the correlation spatial fidelity, or sparsity matching?

Computed on **the scored panel** — the gene set the
agreement table above says governs.

A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model
that matched only *which genes are sparse* would score on `paper_morans_pearson`
without reproducing any spatial fidelity. R3 controls for the tissue's own detection
rate and asks whether the model still orders genes correctly.

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.2780 | +0.4538 | +0.2788 |
| **R2** `corr(I_4, control)` | +0.0618 | +0.5262 | +0.3475 |
| **R3** partial `corr(I_4, I_real given control)` | +0.5873 | +0.4573 | +0.4866 |
| retained fraction of `r` = +0.5826 | 100.8% | 78.5% | 83.5% |

**UNINFORMATIVE — the control specifications disagree** — the 3 specifications span 0.223 against a 0.150 tolerance (`ladder_preregistration.md` §4, `flanking_copy_preregistration.md` §3).

### Stage 4 across whole generations

Each row is a **complete** regeneration — layout, prior, flow, decode, draw — not a
redraw at fixed cells. The A1 arms' `across seeds` column is emission noise alone; this
is the whole pipeline's, and it is the one the verdict leans on
(`null_band_preregistration.md` §2c).

| seed | cells | median I (rank) | r, panel | r, all genes |
|---|---|---|---|---|
| 1 | 4110 | +0.4072 | +0.5826 | — |
| 2 | 4110 | +0.4016 | +0.5474 | — |
| 3 | 4110 | +0.3991 | +0.5644 | — |

**panel: r spans +0.5474 .. +0.5826 across 3 generations** (sd 0.0176). Any rung difference smaller than this is not resolved by a single generation.

## `flanking_copy` — is the floor that beats us spatial fidelity?

⚠️ **This is the one test whose favourable outcome this project has an interest in.**
`flanking_copy_preregistration.md` §1 fixes two rules before any number here existed:
the default absent a clear result is **the outcome that does not suit us**, and **the
negative result is reported either way** — v25 loses to a model-free copy, and that
sentence goes in the paper whatever this block says.

Source: **section_5** at z=63.0 (4110 cells) — the nearest *training* section to section_6 at z=74.0 (4162 cells), emitted
verbatim, exactly as `bench3/selftest.py::make_probe` does. It is **not** at the
target's cells, so it is comparable to stage 4 and to no other rung (§2a).

| quantity | value |
|---|---|
| `r_flank` on 28 genes | **+0.9836** |
| stage 4, same scope | +0.5826 |
| spearman / mae | +0.9792 / 0.0207 |

🚩 **0.9836 is tier-1's 28-gene figure and does not transfer here** (§2b). The test is
against whatever the copy scores on *this* scope.

### §5c — the positive control

`spatial_scramble` keeps every per-gene marginal and destroys only position: **+0.0629** (sd 0.2811, 20 seeds).

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
| 10 genes | **+0.0179** | -0.3695 .. +0.3789 | 1.8% | 20 |
| 25 genes | 🚩 **REFUSED** | a width of 25 leaves fewer than 2 strata in 28 genes, so the relabelling is a full permutation and not an abundance-matched one | — | — |
| 50 genes | 🚩 **REFUSED** | a width of 50 leaves fewer than 2 strata in 28 genes, so the relabelling is a full permutation and not an abundance-matched one | — | — |

The reading must be **stable across all three widths** (§5a); a result that appears at
one width and not the others is a stratum-width artefact.

### §3 — R3 for the copy, under all three controls

| quantity | detection rate | log mean count | log count variance |
|---|---|---|---|
| **R1** `corr(I_real, control)` — is the tissue's ordering a sparsity ordering? | +0.2780 | +0.4538 | +0.2788 |
| **R2** `corr(I_copy, control)` | +0.2082 | +0.4940 | +0.3395 |
| **R3** partial `corr(I_copy, I_real given control)` | +0.9856 | +0.9806 | +0.9826 |
| retained fraction of `r` = +0.9836 | 100.2% | 99.7% | 99.9% |

The three specifications span **0.005** against a 0.150
tolerance, and the bands must be met by **every** control, not by their mean (§3-§4).

## **1. FLOOR IS GENUINE — the copy's score survives every control. The deficit is ours and the paper reports a negative result with NO benchmark claim**

§5's preconditions — independence from v25's own numbers, the positive control,
stability across F3's widths, replication across sections 2/4/6 and both datasets — are
**preconditions, not follow-ups**. One section of one dataset does not make a benchmark
claim, and this run is one section of one dataset.

## The abundance floor — panel

⚠️ **The raw comparison is the result and is first.** This rescaling moves v25's headline
in a direction that flatters it, so `abundance_floor_preregistration.md` §1 fixes that the
raw figures are reported first and always, and that every failure mode below returns to
them. **It does not reopen the closeability question** — the ladder answered that and this
changes only how the negative result is expressed.

F3 applied to a rung's own per-gene `I` gives what that rung would score **from abundance
alone**. The denominator is the **copy's** floor for every rung (§3), never the rung's own,
so no rung can improve its own scale.

🚩 **NOT RESCALED** — the abundance floor is +0.018, indistinguishable from the permutation null: there is nothing to rescale by and the raw comparison already is the answer. Only the raw comparison stands (§4).

### Stage 4p — the emission-free ceiling

With the emission's noise removed from the model's **own** mean field, `I` = **+0.8309** against the real section's **+0.4102** (2.03x). That is the most any repair to `theta` or `pi` can reach on this fit — it bounds the emission-side work from above. **If it exceeds the tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and a repair to it alone cannot land the model on the tissue.

## Candidate 2 — is `mu`'s dynamic range the size factor?

`mu = link(MLP_mu(u)) * size_factor`, so `log mu = shape + log s` and the
variance splits exactly. Per gene, medians over the panel above.

The **real latent** column is the same decoder and the same size head applied to
`h1 = encoder(real counts)` instead of to the flow's sample: the two columns differ
in the latent and in nothing else. It is the matched tissue-side quantity the
record has been quoting as "tissue's 1.213" without a source.

| quantity | generated `h` | real latent `h1` |
|---|---|---|
| genes decomposed | 28 | 28 |
| `Var(shape)` — the latent-driven part | 0.65490 | 0.62687 |
| `Var(log s)` — the size-factor part | 0.00122 | 0.00120 |
| `2 Cov` | -0.00685 | -0.00837 |
| share of `Var(log mu)` from the latent (unbounded) | 100.7% | 100.8% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **99.8%** | **99.8%** |
| **`sd(log mu)` across cells** | **0.8046** | **0.7892** |

**`Var(log mu)` generated / real, median per gene: 1.035.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

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
| A1a'. mu decoded from h1                          | **+0.5507** | — | 28 | — | — | — |
| A1a. counts ~ emission(mu \| h1)                  | **+0.2482** | +0.2480 .. +0.2557 (3) | 28 | 0.99x | **-54.10** | DOES NOT RECOVER |
| A1b'. mu_oracle = kNN mean of real counts         | **+0.8833** | — | 28 | — | — | — |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)     | **+0.4043** | +0.4009 .. +0.4100 (3) | 28 | 1.00x | **-0.99** | DOES NOT RECOVER |
| A1b-t. counts ~ NB(mu_oracle, model theta), pi=0  | **+0.4043** | +0.4008 .. +0.4100 (3) | 28 | 1.00x | **-0.99** | DOES NOT RECOVER |
| A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8412** | +0.8407 .. +0.8424 (3) | 28 | 1.00x | **+147.66** | RECOVERS |
| A1c. counts ~ Poisson(mu_oracle)   [model-free]   | **+0.8424** | +0.8407 .. +0.8424 (3) | 28 | 1.00x | **+148.07** | RECOVERS |
| A1n. permutation null (real counts shuffled)      | **-0.0006** | -0.0013 .. +0.0010 (3) | 28 | — | **-138.76** | DOES NOT RECOVER |

Anchors from this run: `I(model counts)` = **+0.4072**, `I(real counts)` = **+0.4102**, deficit = **+0.0029**.

**Three-seed stability** (`a1_escalation_preregistration.md` §1): an arm whose per-seed bands disagree reads **UNRESOLVED** regardless of its median.

| arm | per-seed R | bands | verdict |
|---|---|---|---|
| A1a | -54.10, -51.55, -54.15 | DOES NOT RECOVER | DOES NOT RECOVER |
| A1b | +0.94, -2.16, -0.99 | RECOVERS, DOES NOT RECOVER | **UNRESOLVED** (seeds straddle) |
| A1b-t | +0.94, -2.19, -0.99 | RECOVERS, DOES NOT RECOVER | **UNRESOLVED** (seeds straddle) |
| A1b-p | +148.06, +147.48, +147.66 | RECOVERS | RECOVERS |
| A1c | +148.07, +147.48, +148.07 | RECOVERS | RECOVERS |
| A1n | -138.20, -138.76, -138.99 | DOES NOT RECOVER | DOES NOT RECOVER |

### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss

Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were
built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact
within-realisation contrast.

| quantity | value |
|---|---|
| `L_total = I(A1c) - I(A1b)` | +0.4381 |
| `L_theta = I(A1c) - I(A1b-t)` | +0.4381 (**100.0%** of total) |
| `L_pi = I(A1c) - I(A1b-p)` | +0.0012 (**0.3%** of total) |
| additivity gap `abs(L_theta + L_pi - L_total)` | 0.0012 (criterion <= 0.0200) |
| multiplicative prediction of `I(A1b)` | +0.4037 against the measured +0.4043, gap 0.0006 |
| **verdict** | **over-dispersion** |

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
| real counts (Poisson-deconvolved) — UPPER bound | **1.0669** | 28 |
| mu_oracle (kNN mean field) — LOWER bound | **0.7448** | 28 |
| mu decoded from h1 | **0.7249** | 28 |
| mu decoded from the generated h | **0.7346** | 28 |
| model counts (Poisson-deconvolved) | **1.0693** | 28 |
| A1c counts (Poisson-deconvolved) — estimator check | **0.7451** | 28 |

Read against the decoder's own figure in the Candidate 2 block above. `chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper bound falls below it, and **untested still** in between.

⚠️ How loose the lower bound is depends on how autocorrelated the field already is — a kNN mean destroys the variance of a field whose neighbours are unrelated and preserves it where they are not (the self-check measures 0.800 -> 0.279 on an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A lower bound below the decoder's figure is "untested still", never "refuted"** — only the upper bound can refute.
