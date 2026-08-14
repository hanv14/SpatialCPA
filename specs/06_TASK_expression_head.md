# T06 — Expression head: flow matching + gene-conditioned ZINB decoder

**Goal.** Generate each cell's full expression profile: a per-cell latent produced by conditional
flow matching from the GRF prior, decoded through a **gene-conditioned** ZINB head so that (a) any
gene can be decoded including unseen ones, (b) sparsity and overdispersion are preserved, and
(c) gene–gene covariance survives.

**Files:** `spatialcpav25_gen/model/expression.py`, `spatialcpav25_gen/model/spatialcpav25_gen.py`,
`spatialcpav25_gen/losses/reconstruction.py`, `tests/test_expression.py`

**Dependencies:** T01–T05.

---

## 1. Why this shape

Two failure modes to design against, both observed in earlier versions of this project:

- **Mean regression over-smooths.** Predicting `E[expr]` collapses variance and destroys spatial
  autocorrelation. So: sample from a conditional distribution, never emit the mean.
- **Per-gene independent sampling produces chimeras.** The competing method draws each gene
  independently from among ≤3 donor cells, which destroys within-cell gene–gene covariance. So: all
  genes of a cell decode from **one shared latent** `h_i`.

## 2. Flow matching in the cell latent

```python
class LatentFlow(nn.Module):
    """Conditional flow matching, straight-line path, in R^{d_h}."""
    def velocity(self, h_t: Tensor, t: Tensor, cond: Tensor) -> Tensor: ...
    def cfm_loss(self, h1: Tensor, h0: Tensor, cond: Tensor, gen) -> Tensor: ...
    def sample(self, h0: Tensor, cond: Tensor, steps: int) -> Tensor: ...
```

- Path: `h_t = (1-t) h0 + t h1`, target velocity `u = h1 - h0`. Loss
  `E_t ||v_theta(h_t, t, cond) - u||^2`, `t ~ U(0,1)`.
- `h0` comes from the **GRF** (T03), queried at the cells' physical positions — *not* i.i.d.
  Gaussian. Pass the field in, do not construct noise inside.
- `h1` (the data-side latent) is the output of a small encoder over the cell's observed expression:
  `h1 = Enc(log1p(counts / size_factor))`, `d_h = cfg.latent_dim`. Train `Enc` jointly with the
  decoder by the reconstruction loss, and **detach `h1` in the CFM loss** so the flow chases a
  stable target rather than co-adapting with the encoder.
- `cond = [F(p), fourier(p), type_emb, region_emb, ctx_retrieval, z_embed]`.
- Sampler: Heun (2nd order), `cfg.ode_steps` steps. Deterministic given `h0`.

## 3. Gene-conditioned ZINB decoder

```python
class ZINBDecoder(nn.Module):
    def forward(self, h: Tensor, gene_emb: Tensor, size_factor: Tensor
                ) -> tuple[Tensor, Tensor, Tensor]:   # mu, theta, pi  each (N, G')
```

```
u_ig = [ h_i , e_g , h_i * (A e_g) ]            # bilinear/FiLM interaction
mu    = softplus(MLP_mu(u))  * size_factor      # (N, G')
theta = softplus(MLP_theta(u)) + eps            # dispersion, per (cell, gene)
pi    = sigmoid(MLP_pi(u))                      # zero-inflation
```

The gene enters only through `e_g`, so a single decoder serves an arbitrary gene set — this is what
makes zero-shot genes and cross-panel transfer possible. **Gene subsampling:** each training step
draws `cfg.genes_per_step` genes; the decoder never sees a fixed panel width.

`size_factor` decoded from `h_i` by a small head, trained against the observed per-cell total.

**Loss:** ZINB negative log-likelihood against **raw counts**. Implement carefully in log-space and
unit-test against a reference implementation; a numerically sloppy ZINB is a classic source of silent
NaNs. Guard: `theta` clamped to `[1e-4, 1e6]`, `mu` to `[1e-8, 1e8]`.

For non-count data (e.g. intensity-based assays), provide an alternative
`ZIGammaDecoder` with the same interface, selected by a config field. Detect and warn if the data
is non-integer while a ZINB decoder is selected.

## 4. Sampling counts

```python
def sample_counts(mu, theta, pi, gen) -> Tensor
```
Draw `NB(mu, theta)` then zero out with probability `pi`. **Never return `mu`.** Add an assertion in
the generation path that the emitted matrix has a detection rate within a plausible band of the
training sections' — a silent switch to means would show up here.

## 5. Top-level module — `spatialcpav25_gen/model/spatialcpav25_gen.py`

```python
class CTFFlow(nn.Module):
    """Field + retrieval + layout + expression, one object."""
    def forward_train(self, batch: Batch) -> dict[str, Tensor]   # named loss terms
    def generate(self, plane: Plane, cfg: Config, seed: int) -> AnnData
```

`forward_train` returns a dict of named, unweighted loss terms; the trainer applies weights. This
matters for T07/T08 which add terms, and for logging each term separately (essential for diagnosing
collapse).

Also implement the trainer: AdamW, cosine schedule, gradient clipping at 1.0, EMA of weights
(`cfg.ema_decay`) — the EMA copy is reused as the teacher in T07.

## Acceptance tests

- `test_zinb_nll_matches_reference` — against a hand-derived reference on random inputs, 1e-5.
- `test_zinb_no_nan_extremes` — `mu` 1e-8..1e8, `theta` 1e-4..1e6, `pi` 0..1: finite everywhere.
- `test_cfm_recovers_gaussian` — a toy 2-D target learned to Wasserstein-2 < 0.1 in 2000 steps.
- `test_flow_deterministic` — same `h0` and cond → identical `h1`.
- `test_shared_latent_preserves_covariance` — **the key test.** Generate cells; compare the
  gene–gene correlation matrix to the real section's. Frobenius-norm error must be **< 50%** of the
  error from a per-gene-independent-donor baseline (a reimplementation of the competing method's
  sampler, ~20 lines, put it in `eval/baselines.py`). This is the quantitative claim behind the
  paper's covariance argument.
- `test_sparsity_preserved` — per-gene detection rate: Pearson r > 0.95 vs. real; mean absolute
  difference < 0.05. Guards against the densification failure of earlier versions.
- `test_mean_variance_relation` — the generated mean–variance curve tracks the real one (log-log
  slope within 15%).
- `test_zero_shot_gene_decoding` — hold out 20% of genes from training entirely; decoding them via
  `forward_zero_shot` embeddings yields per-gene mean expression correlating with truth at r > 0.4.
  *(If this fails badly, note it and continue — it is a capability experiment, not a gate.)*
- `test_never_returns_means` — generation output is integer-valued and has non-zero variance
  conditional on cond.

## Definition of done

On the fixture, generated sections match the real held-in section on: detection rate (r > 0.95),
gene–gene covariance (better than the independent-donor baseline by ≥ 2×), and mean–variance slope.
`PROGRESS.md` records these three numbers.

## Do NOT

- Do not emit `mu` as the output anywhere in the generation path.
- Do not decode genes through a fixed-width output layer (breaks open-vocabulary).
- Do not let normalised expression become the ZINB target.
- Do not skip the independent-donor baseline — it is needed here and again in T10.
