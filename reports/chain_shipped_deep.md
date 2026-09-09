# Chain diagnostic — where the spatial structure is lost (2400 steps)

`deep_starmap` / `paper_2_4_6`, `section_2` at z=30.8, 30097 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

## What this run is

| | |
|---|---|
| text channel | live, `text_emb_mode=medcpt` |
| `expr_pca_dim` | 32 |
| `decoder_mu_link` | `exp` |
| `layout_mode` / `layout_sampler` | `resample` / `grid` |
| cell density | matched to the real section: 39327 generated -> 30097 kept (seed 1) |
| gene panel | top 32 by Moran's I on the **real** side |
| real section | 30097 cells, z = 31.5 |

🚩 --target-z 30.8 is a boundary plane (within 0.5 median spacings of the stack's end): evidence there is one-sided, which T04 measured as a 20-35% reconstruction deficit (R3).

**Panel** (real-selected, 32 genes): `APOD`, `AQP1`, `CLDN11`, `CLIC6`, `DKK3`, `FOLR1`, `GABBR2`, `GFAP`, `GM5741`, `IGF2`, `IGFBP2`, `KL`, `LAMP5`, `LBP`, `LEF1`, `MBP`, `NEUROD6`, `NPTX1`, `NRGN`, `OLFR558`, `PENK`, `PPP1R1B`, `PRKCD`, `PTGDS`, `PVALB`, `SATB2`, `SPARC`, `SULF1`, `TMEM72`, `TRF`, `TTR`, `VAMP1`

The panel restricts the **gene-space stages only** — 3, 4, their calibrated twins, and `REF real counts`. Stages 1 and 2 and `REF real latent h1` are latents with 64 channels that are not genes; their `channels` column shows that, and their `median I` is over all of them.

## The chain

| stage                                     | median I | p25 | p75 | channels |
|-------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz        | **+0.9284** | +0.9253 | +0.9321 | 64 |
| 2. latent h after the flow                | **+0.7858** | +0.7588 | +0.8398 | 64 |
| 3. decoded mu (before sampling)           | **+0.7966** | +0.7374 | +0.8284 | 32 |
| 4. sampled counts (rank-normalised)       | **+0.0729** | +0.0186 | +0.1873 | 32 |
| REF real counts (rank-normalised)         | **+0.3236** | +0.2804 | +0.3994 | 32 |
| REF real latent h1 = encoder(real counts) | **+0.2415** | +0.2182 | +0.3594 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.3236 | +0.2415 | **134.0%** | 1.315 | — |
| uncalibrated | +0.0729 | +0.7858 | **9.3%** | 1.284 | 1.315 |

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
| `Var(shape)` — the latent-driven part | 0.52662 | 0.23374 |
| `Var(log s)` — the size-factor part | 0.01471 | 0.01578 |
| `2 Cov` | -0.00033 | -0.01942 |
| share of `Var(log mu)` from the latent (unbounded) | 98.1% | 100.9% |
| **bounded share** `Var(shape)/(Var(shape)+Var(log s))` | **97.3%** | **93.7%** |
| **`sd(log mu)` across cells** | **0.7254** | **0.4663** |

**`Var(log mu)` generated / real, median per gene: 1.917.** Pre-registered in `reports/emission_repair_options.md` §8.3: >= 0.8 means the structured component is intact and §2's binding constraint does not exist; <= 0.4 confirms it; between the two is uninformative and needs the three-seed version.
