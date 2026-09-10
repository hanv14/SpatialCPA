# Chain diagnostic — where the spatial structure is lost (2400 steps)

`starmap_visual_cortex` / `paper_2_4_6`, `section_2` at z=30.0, 4073 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt`, 28/28 gene rows non-zero — from `runs/chain/shipped_tier1.pt` |
| `expr_pca_dim` | 28 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | 🚩 **vacuous**: requested 4187, the layout produced only 4073, so nothing was subsampled and the arms are **NOT** density-matched |
| gene panel | 🚩 **vacuous**: `--top-k 32` >= the panel's 28 genes, so all 28 were kept and **no selection took place** |
| real section | 4187 cells, z = 30.0 |

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
| 1. prior h0 = GRF at generated xyz                | **+0.9362** * | +0.9328 | +0.9274 | +0.9433 | 64 |
| 2. latent h after the flow                        | **+0.8011** * | +0.7608 | +0.7224 | +0.8563 | 64 |
| 3. decoded mu (before sampling)                   | **+0.7920** * | +0.8365 | +0.6965 | +0.8388 | 28 |
| 4. sampled counts (rank-normalised)               | +0.4158 | **+0.5134** * | +0.3995 | +0.5998 | 28 |
| 4p. counts ~ Poisson(mu) — emission noise removed | +0.7911 | **+0.8358** * | +0.8062 | +0.8775 | 28 |
| REF real counts (rank-normalised)                 | +0.4162 | **+0.4635** * | +0.3587 | +0.5637 | 28 |
| REF real latent h1 = encoder(real counts)         | **+0.6253** * | +0.5056 | +0.5092 | +0.7254 | 64 |

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
| **real tissue** | +0.4635 | +0.6253 | **74.1%** | 1.738 | — |
| uncalibrated | +0.5134 | +0.8011 | **64.1%** | 1.846 | 1.738 |

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
| panel | 3. decoded mu | **+0.3847** | +0.3820 | 0.3535 | +0.8365 | +0.4635 | 28 |
| panel | 4. sampled counts | **+0.5076** | +0.5435 | 0.1018 | +0.5134 | +0.4635 | 28 |
| panel | 4p. Poisson(mu) — emission-free | **+0.3878** | +0.4007 | 0.3529 | +0.8358 | +0.4635 | 28 |

## The ladder — what each rung holds at the truth

Every arm is drawn at the **real section's own cells**, so `pred_xy == gt_xy`. That is an
advantage stage 4 and bench3 do not have — they compare a generated cell set against the
real one, each on its own graph — so **a LOW rung here is conclusive and a high one is
permissive** (`ladder_preregistration.md` §2a).

Reference points on tier-1: the model-free copy floor `flanking_copy` = **0.9836**,
SpatialZ **0.932**, v25 shipped **0.5574**.

| gene set | rung | **pearson** | across seeds | spearman | mae | genes |
|---|---|---|---|---|---|---|
| panel | A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.8332** | +0.8327 .. +0.8333 (3) | +0.8643 | 0.4230 | 28 |
| panel | A1b. counts ~ ZINB(mu_oracle, model theta/pi) | **+0.8369** | +0.8341 .. +0.8421 (3) | +0.8517 | 0.0746 | 28 |
| panel | A1b-t. counts ~ NB(mu_oracle, model theta), pi=0 | **+0.8367** | +0.8341 .. +0.8420 (3) | +0.8517 | 0.0748 | 28 |
| panel | A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.8334** | +0.8330 .. +0.8343 (3) | +0.8643 | 0.4229 | 28 |
| panel | A1a. counts ~ emission(mu \| h1) | **+0.6985** | +0.6927 .. +0.7075 (3) | +0.7269 | 0.0809 | 28 |
| panel | 4.  counts ~ emission(mu \| model latent)   [where we are] | **+0.5076** | — | +0.5435 | 0.1018 | 28 |
| panel | A1n. permutation null (real counts shuffled) | **-0.2472** | -0.3928 .. -0.1294 (3) | -0.3361 | 0.4723 | 28 |

🚨 **NULL RUNG IS NOT NULL** — the permutation arm correlates at -0.2472. `ladder_preregistration.md` §5(1): the construction is wrong and **nothing on this ladder may be read.**

### R1-R3 - is the correlation spatial fidelity, or sparsity matching?

Computed on **the scored panel** — the gene set the
agreement table above says governs.

A sparse gene's Moran's I is bounded low whatever its spatial structure, so a model
that matched only *which genes are sparse* would score on `paper_morans_pearson`
without reproducing any spatial fidelity. R3 controls for the tissue's own detection
rate and asks whether the model still orders genes correctly.

| quantity | detection rate | log mean count |
|---|---|---|
| **R1** `corr(I_real, control)` - is the tissue's ordering a sparsity ordering? | +0.2862 | +0.4171 |
| **R2** `corr(I_model, control)` | +0.1785 | +0.3566 |
| **R3** partial `corr(I_4, I_real given control)` | +0.4847 | +0.3593 |
| retained fraction of `r(4)` = +0.5076 | 95.5% | 70.8% |

**UNINFORMATIVE — the two control specifications disagree** — the two specifications differ by 0.247 against a 0.150 tolerance (`ladder_preregistration.md` §4).

## Why the model sits ABOVE the tissue here, and why that is not fidelity

`I(model counts)` = **+0.5134** against the real section's **+0.4635** — 1.11x. That is **not** a reconstruction result. Two defects point in opposite directions on this dataset and partly cancel:

| defect | this run | direction on `I` |
|---|---|---|
| the latent is **1.28x smoother** than the tissue's (+0.8011 against `h1`'s +0.6253) | too smooth | pushes `I` **up** |
| `mu`'s spread is **0.7269** against the tissue's model-free bracket [0.7165, 0.9438] | too narrow | — |
| the emission adds spatially independent noise (`theta`, and `pi` where it is non-zero) | too much | pushes `I` **down** |

So a number at or above the tissue's says the two happen to cancel, not that either is right. Repairing one alone moves `I` **away** from the tissue: a faithful latent lowers it (A1's `A1a` arm), and removing the emission's noise raises it (stage 4p).

### Stage 4p — the emission-free ceiling

With the emission's noise removed from the model's **own** mean field, `I` = **+0.8358** against the real section's **+0.4635** (1.80x). That is the most any repair to `theta` or `pi` can reach on this fit — it bounds the emission-side work from above. **If it exceeds the tissue, the emission is not the only defect** (`reports/n5_and_m3_review.md` §5), and a repair to it alone cannot land the model on the tissue.

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
| `Var(shape)` — the latent-driven part | 0.51862 | 0.52419 |
| `Var(log s)` — the size-factor part | 0.00125 | 0.00149 |
| `2 Cov` | +0.00159 | -0.00100 |
| share of `Var(log mu)` from the latent (unbounded) | 99.7% | 99.9% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **99.8%** | **99.7%** |
| **`sd(log mu)` across cells** | **0.7269** | **0.7333** |

**`Var(log mu)` generated / real, median per gene: 0.983.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

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
| A1a'. mu decoded from h1                          | **+0.6052** | — | 28 | — | — | — |
| A1a. counts ~ emission(mu \| h1)                  | **+0.4053** | +0.4006 .. +0.4103 (3) | 28 | 1.01x | — | — |
| A1b'. mu_oracle = kNN mean of real counts         | **+0.9284** | — | 28 | — | — | — |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)     | **+0.5405** | +0.5390 .. +0.5452 (3) | 28 | 1.00x | — | — |
| A1b-t. counts ~ NB(mu_oracle, model theta), pi=0  | **+0.5405** | +0.5390 .. +0.5452 (3) | 28 | 1.00x | — | — |
| A1b-p. counts ~ A1c's Poisson draw, then model pi | **+0.9114** | +0.9100 .. +0.9114 (3) | 28 | 1.00x | — | — |
| A1c. counts ~ Poisson(mu_oracle)   [model-free]   | **+0.9114** | +0.9112 .. +0.9114 (3) | 28 | 1.00x | — | — |
| A1n. permutation null (real counts shuffled)      | **+0.0025** | -0.0059 .. +0.0048 (3) | 28 | — | — | — |

Anchors from this run: `I(model counts)` = **+0.5134**, `I(real counts)` = **+0.4635**, deficit = **-0.0498**.

### N5 — which of `theta` and `pi` costs the `A1c -> A1b` loss

Criteria fixed in `a1_escalation_preregistration.md` §2, before these arms were
built. `A1b-p` shares `A1c`'s Poisson realisation, so `A1c -> A1b-p` is an exact
within-realisation contrast.

| quantity | value |
|---|---|
| `L_total = I(A1c) - I(A1b)` | +0.3709 |
| `L_theta = I(A1c) - I(A1b-t)` | +0.3709 (**100.0%** of total) |
| `L_pi = I(A1c) - I(A1b-p)` | +0.0000 (**0.0%** of total) |
| additivity gap `abs(L_theta + L_pi - L_total)` | 0.0000 (criterion <= 0.0200) |
| multiplicative prediction of `I(A1b)` | +0.5405 against the measured +0.5405, gap 0.0000 |
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
| real counts (Poisson-deconvolved) — UPPER bound | **0.9438** | 28 |
| mu_oracle (kNN mean field) — LOWER bound | **0.7165** | 28 |
| mu decoded from h1 | **0.7225** | 28 |
| mu decoded from the generated h | **0.6728** | 28 |
| model counts (Poisson-deconvolved) | **0.8740** | 28 |
| A1c counts (Poisson-deconvolved) — estimator check | **0.7162** | 28 |

Read against the decoder's own figure in the Candidate 2 block above. `chain_shipped_review.md` §6's narrow-`mu` mechanism is **supported** if the tissue's lower bound exceeds it by >= 1.5x, **refuted** if the tissue's upper bound falls below it, and **untested still** in between.

⚠️ How loose the lower bound is depends on how autocorrelated the field already is — a kNN mean destroys the variance of a field whose neighbours are unrelated and preserves it where they are not (the self-check measures 0.800 -> 0.279 on an unstructured field). Read it beside `I(mu_oracle)` in the table above. **A lower bound below the decoder's figure is "untested still", never "refuted"** — only the upper bound can refute.

🚩 `I(model counts)` is at or above `I(real counts)` on this dataset, so there is
no deficit to recover and **`R` is undefined**. This run is the
pre-registration's
**instrument control**: every drawn arm must reach 0.6x `I(real counts)` = **+0.2781**, or the ablation is measuring something other than what
it claims and no result on the other dataset may be read.
