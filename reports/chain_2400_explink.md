# Chain diagnostic — where the spatial structure is lost (2400 steps)

STARmap tier 1, `section_2` at z=30.0, 48343 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

| stage                                      | median I | p25 | p75 | channels |
|--------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz         | **+0.9928** | +0.9915 | +0.9936 | 64 |
| 2. latent h after the flow                 | **+0.8677** | +0.8261 | +0.9057 | 64 |
| 3. decoded mu (before sampling)            | **+0.9008** | +0.8577 | +0.9239 | 28 |
| 4. sampled counts (rank-normalised)        | **+0.5253** | +0.3949 | +0.6624 | 28 |
| 3c. decoded mu, CALIBRATED                 | **+0.9008** | +0.8577 | +0.9239 | 28 |
| 4c. sampled counts, CALIBRATED (rank-norm) | **+0.5295** | +0.3503 | +0.6432 | 28 |
| REF real counts (rank-normalised)          | **+0.4635** | +0.3587 | +0.5637 | 28 |
| REF real latent h1 = encoder(real counts)  | **+0.5812** | +0.4771 | +0.6892 | 64 |

## The three numbers

**Retention across the latent -> counts step** — what the emission costs, against what
the tissue's own sampling noise costs.

| arm | counts I | latent I | retention | slope | tissue slope |
|---|---|---|---|---|---|
| **real tissue** | +0.4635 | +0.5812 | **79.8%** | 1.738 | — |
| uncalibrated | +0.5253 | +0.8677 | **60.5%** | 1.807 | 1.738 |
| calibrated | +0.5295 | +0.8677 | **61.0%** | 1.611 | 1.738 |

## Candidate 2 — is `mu`'s dynamic range the size factor?

`mu = link(MLP_mu(u)) * size_factor`, so `log mu = shape + log s` and the
variance splits exactly. Per gene, medians over genes.

| quantity | value |
|---|---|
| `Var(shape)` — the latent-driven part | 0.60376 |
| `Var(log s)` — the size-factor part | 0.00123 |
| `2 Cov` | -0.00037 |
| **share of `Var(log mu)` from the latent** | **100.1%** |
| **share from the size factor** | **0.2%** |
| `sd(log mu)` across cells | 0.7767 |
