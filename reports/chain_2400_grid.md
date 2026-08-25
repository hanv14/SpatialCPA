# Chain diagnostic — where the spatial structure is lost (2400 steps)

STARmap tier 1, `section_2` at z=30.0, 267567 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

| stage                                     | median I | p25 | p75 | channels |
|-------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz        | **+0.9986** | +0.9984 | +0.9988 | 64 |
| 2. latent h after the flow                | **+0.8866** | +0.8365 | +0.9279 | 64 |
| 3. decoded mu (before sampling)           | **+0.9098** | +0.8679 | +0.9348 | 28 |
| 4. sampled counts (rank-normalised)       | **+0.5408** | +0.4301 | +0.6781 | 28 |
| REF real counts (rank-normalised)         | **+0.4635** | +0.3587 | +0.5637 | 28 |
| REF real latent h1 = encoder(real counts) | **+0.5717** | +0.4821 | +0.7023 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.4635 | +0.5717 | **81.1%** | 1.738 | — |
| uncalibrated | +0.5408 | +0.8866 | **61.0%** | 1.804 | 1.738 |

## Candidate 2 — is `mu`'s dynamic range the size factor?

`mu = link(MLP_mu(u)) * size_factor`, so `log mu = shape + log s` and the
variance splits exactly. Per gene, medians over genes.

| quantity | value |
|---|---|
| `Var(shape)` — the latent-driven part | 0.56008 |
| `Var(log s)` — the size-factor part | 0.00117 |
| `2 Cov` | -0.00333 |
| **share of `Var(log mu)` from the latent** | **100.4%** |
| **share from the size factor** | **0.2%** |
| `sd(log mu)` across cells | 0.7486 |
