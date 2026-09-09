# Chain diagnostic — where the spatial structure is lost (2400 steps)

`starmap_visual_cortex` / `paper_2_4_6`, `section_2` at z=30.0, 4073 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt` |
| `expr_pca_dim` | 28 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | matched to the real section: 4073 generated -> 4073 kept (seed 1) |
| gene panel | top 28 by Moran's I on the **real** side |
| real section | 4187 cells, z = 30.0 |

**Panel** (real-selected, 28 genes): `Slc17a7`, `Mgp`, `Gad1`, `Nov`, `Rasgrf2`, `Rorb`, `Cux2`, `Plcxd2`, `Sulf2`, `Ctgf`, `Pcp4`, `Sema3e`, `Npy`, `Sst`, `Pvalb`, `Vip`, `Calb2`, `Cck`, `Reln`, `Fos`, `Egr1`, `Prok2`, `Egr2`, `Bdnf`, `Gja1`, `Ctss`, `Mbp`, `Flt1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

| stage                                     | median I | p25 | p75 | channels |
|-------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz        | **+0.9362** | +0.9274 | +0.9433 | 64 |
| 2. latent h after the flow                | **+0.8011** | +0.7224 | +0.8563 | 64 |
| 3. decoded mu (before sampling)           | **+0.7920** | +0.6965 | +0.8388 | 28 |
| 4. sampled counts (rank-normalised)       | **+0.5134** | +0.3995 | +0.5998 | 28 |
| REF real counts (rank-normalised)         | **+0.4635** | +0.3587 | +0.5637 | 28 |
| REF real latent h1 = encoder(real counts) | **+0.6253** | +0.5092 | +0.7254 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.4635 | +0.6253 | **74.1%** | 1.738 | — |
| uncalibrated | +0.5134 | +0.8011 | **64.1%** | 1.846 | 1.738 |

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
