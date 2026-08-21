# Chain diagnostic — where the spatial structure is lost (2400 steps)

STARmap tier 1, `section_2` at z=30.0, 11168 generated cells.
Median per-channel Moran's I on a row-standardised kNN graph (k=10), the same estimator at every stage.

| stage                                     | median I | p25 | p75 | channels |
|-------------------------------------------|---|---|---|---|
| 1. prior h0 = GRF at generated xyz        | **+0.9714** | +0.9661 | +0.9742 | 64 |
| 2. latent h after the flow                | **+0.9015** | +0.8856 | +0.9162 | 64 |
| 3. decoded mu (before sampling)           | **+0.8607** | +0.8508 | +0.8722 | 28 |
| 4. sampled counts (rank-normalised)       | **+0.1297** | +0.0791 | +0.1779 | 28 |
| REF real counts (rank-normalised)         | **+0.4635** | +0.3587 | +0.5637 | 28 |
| REF real latent h1 = encoder(real counts) | **+0.7449** | +0.6752 | +0.7912 | 64 |
