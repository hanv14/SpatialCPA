# Chain diagnostic — where the spatial structure is lost (1200 steps)

`deep_starmap` / `paper_2_4_6`, `section_4` at z=68.6, 29544 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | **zero vectors** (neither A3 arm) |
| `expr_pca_dim` | 32 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | matched to the real section: 29842 generated -> 29544 kept (seed 1) |
| gene panel | top 32 by Moran's I on the **real** side |
| real section | 29544 cells, z = 73.5 |

**Panel** (real-selected, 32 genes): `APOD`, `AQP1`, `C4B`, `CLDN11`, `DKK3`, `FOLR1`, `GABBR2`, `GFAP`, `LEF1`, `MBP`, `MEIS2`, `NEFH`, `NEUROD6`, `NPTX1`, `NRGN`, `NTNG1`, `OLFR558`, `PCP4`, `PENK`, `PPP1R1B`, `PRKCD`, `PTGDS`, `PVALB`, `SATB1`, `SATB2`, `SLC6A11`, `SPARC`, `TCF7L2`, `TMSB4X`, `TRF`, `TTR`, `VAMP1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

| stage                                     | median I | p25 | p75 | channels |
|-------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz        | **+0.9272** | +0.9238 | +0.9300 | 64 |
| 2. latent h after the flow                | **+0.7786** | +0.7584 | +0.8379 | 64 |
| 3. decoded mu (before sampling)           | **+0.7469** | +0.7285 | +0.7992 | 32 |
| 4. sampled counts (rank-normalised)       | **+0.1021** | +0.0551 | +0.1983 | 32 |
| REF real counts (rank-normalised)         | **+0.3123** | +0.2654 | +0.3433 | 32 |
| REF real latent h1 = encoder(real counts) | **+0.3171** | +0.2892 | +0.4276 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.3123 | +0.3171 | **98.5%** | 1.277 | — |
| uncalibrated | +0.1021 | +0.7786 | **13.1%** | 1.258 | 1.277 |

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
| `Var(shape)` — the latent-driven part | 0.50525 | 0.23870 |
| `Var(log s)` — the size-factor part | 0.01369 | 0.01706 |
| `2 Cov` | +0.00463 | -0.00504 |
| share of `Var(log mu)` from the latent (unbounded) | 97.7% | 95.3% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **97.4%** | **93.3%** |
| **`sd(log mu)` across cells** | **0.7131** | **0.4797** |

**`Var(log mu)` generated / real, median per gene: 2.080.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.

## A1 — emission ablation, at the real cells

Every arm is drawn at the ground truth's own positions, so the kNN graph, the cell
count and the density are identical to `REF real counts` and to each other. The
layout, the prior and the flow are held out of it entirely: only what the counts
were drawn from varies.

**Read `reports/a1_preregistration.md` before reading these numbers.** The outcome
table, the thresholds, the level guard and the two stated asymmetries were committed
before the run.

| arm                                             | median I | p25 | p75 | channels | level | recovery R |
|-------------------------------------------------|---|---|---|---|---|---|
| A1a'. mu decoded from h1                        | **+0.3399** | +0.2616 | +0.3679 | 32 | — | **+1.13** |
| A1a. counts ~ emission(mu | h1)                 | **+0.0433** | +0.0170 | +0.0947 | 32 | 0.93x | **-0.28** |
| A1b'. mu_oracle = kNN mean of real counts       | **+0.9063** | +0.8945 | +0.9358 | 32 | — | **+3.83** |
| A1b. counts ~ ZINB(mu_oracle, model theta/pi)   | **+0.1722** | +0.1091 | +0.2690 | 32 | 0.64x | **+0.33** |
| A1c. counts ~ Poisson(mu_oracle)   [model-free] | **+0.4994** | +0.4370 | +0.5795 | 32 | 1.00x | **+1.89** |
| A1n. permutation null (real counts shuffled)    | **+0.0007** | -0.0008 | +0.0021 | 32 | — | **-0.48** |

Anchors from this run: `I(model counts)` = **+0.1021**, `I(real counts)` = **+0.3123**, deficit = **+0.2102**.
