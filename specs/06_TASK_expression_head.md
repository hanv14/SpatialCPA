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
mu    = link(MLP_mu(u))      * size_factor      # (N, G'); link = Config.decoder_mu_link
theta = softplus(MLP_theta(u)) + eps            # dispersion, per (cell, gene)
pi    = sigmoid(MLP_pi(u))                      # zero-inflation
```

### The ``mu`` link: ``exp`` since T10, and why the spec's ``softplus`` moved

**This is T06's own revisit condition being met, not a correction.** The formula above was
written as ``softplus`` and T06 kept it, having measured ``exp`` losing on the synthetic
fixture — and having written down, in the same breath, what would justify re-taking the
decision: *"the argument is still right for a panel with a wider dynamic range than this
fixture's; changing the default needs that measurement, not this one"*
(``progress/t06_expression_head.md`` §2).

T10 supplied that measurement. **Both stand, and they are different regimes:**

| | fixture (200 genes, sparse+dense classes) | real STARmap (28 genes, median library 226 580) |
|---|---|---|
| reconstruction NLL | 1.649 -> **1.636** under ``exp`` | — |
| gene-gene Frobenius | 18.02 -> **17.05** under ``exp`` | — |
| per-gene mean-expression r | **0.802 -> 0.576** under ``exp`` (worse) | — |
| counts Moran's I | — | +0.1297 -> **+0.4782** (real tissue +0.4635) |
| structured share of between-cell variance | — | 15.1 % -> **61.4 %** (real ~62 %) |
| mean-variance slope | — | 2.121 -> **1.807** (real 1.738) |
| **verdict** | ``softplus`` wins | **``exp`` wins** |

``softplus`` stays selectable and is the right choice in the fixture's regime. The full
rationale, the failure mode T06 diagnosed (the pre-exponential clamp hiding the gradient, which
is **not fixed, only outweighed**) and both measurement tables live in
``Config.decoder_mu_link``'s own docstring, so the cross-reference is to code rather than to a
progress file.

⚠️ **What this invalidates.** T06's decoder-path acceptance numbers were all measured under
``softplus`` and are re-measured under ``exp`` (see ``progress/t06_expression_head.md``).
**Neither gate is affected** — ``tests/gate1_criteria.py`` and ``tests/gate2_criteria.py``
contain no reference to ``CTFFlow`` or the decoder: GATE 1 pushes latents through the fixture's
own generative map and GATE 2 uses a linear probe over field + retrieval. **T09's selection must
re-run per dataset**, because ``decoder_mu_link`` is not one of ``ALL_GATES`` — it is base
config, so it changes every cell, and ``ScoreCache`` keys on the full config hash and therefore
invalidates automatically.

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

## 4b. The v20 Bernoulli cross-mix — `expr_mode="cross-mix"` (settled, SPEC_QUESTIONS A6)

```python
def cross_mix_counts(donor_counts, weights, gen) -> Tensor
```

Nothing in `specs/` specified building this, and three things depend on it: `Config.expr_mode`
already gates it (T01), T09's no-regression guarantee needs it
(`test_selector_can_recover_v20_config` must be able to select `layout_mode="resample"` +
`expr_mode="cross-mix"` and land on v20's behaviour), and T09's uncertainty-gated anchoring blends
through it (`design/v23_design.md` §5). It belongs here because it shares the count-preserving
output path with the sampler above.

Port it from `reference/learn_spatialcpav20.py` — per gene and per cell, choose one donor by a
Bernoulli/multinomial draw on the donor weights and take that donor's count, so the emitted matrix is
made of real counts and never a blend of two (a blend is neither an integer nor a draw from anything).
~40 lines.

**Pin the behaviour with a test, not with a reading of the code**: `test_cross_mix_matches_v20`
reproduces v20's output bit-for-bit on fixed inputs and a fixed seed. It is a *baseline* — its job is
to be the thing the new path is compared against, so it has to be the old thing exactly. If v20's RNG
consumption order cannot be reproduced under Convention 3's explicit generators, say so in the test's
docstring and assert the distribution instead (same per-gene donor frequencies to within Monte-Carlo
error over 10⁴ draws), rather than quietly accepting a different sampler.

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
- `test_shared_latent_preserves_covariance` — **the key test, and it is AMENDED at T06 (SPEC_QUESTIONS
  B16); read the amendment before quoting a number.** As originally stated: generate cells; compare
  the gene–gene correlation matrix to the real section's; Frobenius-norm error must be **< 50%** of
  the error from a per-gene-independent-donor baseline (a reimplementation of the competing method's
  sampler, ~20 lines, in `eval/baselines.py`).

  **The criterion as stated is below the achievable ceiling and cannot be met by any generator.**
  T05's ceiling protocol, applied here: the *same cells*, the fixture's *true* `mu`, only a fresh
  count draw, gives a Frobenius error of **5.601** on the default holdout (± 0.05 over draws; 5.513
  at the wide gap; 5.705 if the whole generative law is redrawn rather than only the counts) — a
  correlation matrix estimated from ~1500 cells carries that much sampling error whatever produced
  them. The independent-donor baseline on the same section is **7.783**, so "< 50% of the baseline"
  asks for **< 3.892**, which is **1.7 below the ceiling** — thirty-four times the ceiling's own
  draw-to-draw spread. Not a hard criterion: an unsatisfiable one.

  **The replacement criterion is the mechanism test, and only that** — the magnitude/pattern
  decomposition T06 first proposed is **withdrawn as a criterion** (it was chosen after seeing which
  component the model won on, and it gives 0.995 at `consecutive-3`, i.e. no advantage; kept only as
  a reported diagnostic in `reports/benchmark.md`).

  1. **`test_per_gene_independence_destroys_covariance` is the key test.** Hold the donors *fixed*
     and vary only whether the draw is per **cell** or per **gene** — that is the whole of the
     chimerism claim with the positional confound removed, it needs no trained model, and the
     direction and monotonicity are predicted by the paper's argument *before* any measurement rather
     than selected after. Required: mean |off-diagonal correlation| falls monotonically as the number
     of mixed donors grows, a verbatim per-cell copy retains ≥ 0.95 of the real section's value, and
     the per-gene draw over `Config.independent_donor_k` donors loses at least 0.05 more. Measured:
     **0.978 / 0.920 / 0.897 / 0.884 / 0.844** at D = 1/2/3/5/10 against a real 0.1466, on the default
     holdout, and **0.955 / 0.818 / 0.783 / 0.714** at `consecutive-3`. **This is where the paper's
     covariance argument is established.**
  2. **Every arm is reported against the measured ceiling**, with the systematic part
     `sqrt(err² − ceiling²)` beside the raw Frobenius number, because noise and bias add in
     quadrature.
  3. **The original criterion keeps its name and statistic as a strict `xfail`**, so the model's
     shortfall stays on the record and a later task that closes it breaks the suite until the record
     is updated (T05's precedent).

  ### ⛔ The model-versus-baseline covariance comparison is currently a LOSS. Do not quote it.

  | arm | Frobenius error vs the held-out section |
  |---|---|
  | **model (`zinb-flow`)** | **9.316** |
  | independent-donor baseline (`D = 3`) | **7.783** |
  | verbatim nearest-donor copy (`D = 1`) | 6.743 |
  | achievable ceiling (ideal draw) | **5.601** |

  The model is **worse than the baseline it is supposed to beat**, on the default holdout and more so
  at `consecutive-3` (17.7 vs 11.3). Until that reverses, the covariance argument is a **mechanism
  claim only** — "per-gene independent draws destroy covariance, a shared latent cannot" — and no
  paper text, figure, abstract or table may claim that this model preserves covariance *better than
  the competing method*. `specs/08` carries the success criterion that would change this and
  `specs/10` §2 carries the framing rule.

- `test_sparsity_preserved` — per-gene detection rate: Pearson r > 0.95 vs. real; mean absolute
  difference < 0.05. Guards against the densification failure of earlier versions. **Measured on the
  default `alternating` holdout** (SPEC_QUESTIONS B17): the same model scores MAD 0.036 there and
  0.056 at `consecutive-3`, one side of the threshold each, and T06's spec names no regime.
- `test_mean_variance_relation` — the generated mean–variance curve tracks the real one (log-log
  slope within 15%).
- `test_zero_shot_gene_decoding` — hold out 20% of genes from training entirely; decoding them via
  `forward_zero_shot` embeddings yields per-gene mean expression correlating with truth at r > 0.4.
  *(If this fails badly, note it and continue — it is a capability experiment, not a gate.)*
  **It fails badly, and the parenthesis is taken up: r = −0.368** for the 40 never-trained genes
  against +0.946 for the seen ones (SPEC_QUESTIONS B18). Not noise — negative. The fixture's gene
  names are arbitrary (`Gene0042`), for which T02 measured a text/co-expression Spearman of +0.0055,
  and a held-out gene's free residual `r_g` is exactly zero by construction, so there is no channel
  left to transfer through. Kept by name and threshold as a **strict xfail**; the real test is T10's
  capability experiment E1 on a real panel, which needs `resources/gene_meta.parquet` (C14).
- `test_never_returns_means` — generation output is integer-valued and has non-zero variance
  conditional on cond.
- `test_cross_mix_matches_v20` — the ported Bernoulli cross-mix reproduces
  `reference/learn_spatialcpav20.py` on fixed inputs and a fixed seed (or, failing bitwise
  reproduction, matches its donor-frequency distribution — see §4b).
- `test_cross_mix_emits_real_counts` — every emitted value equals some donor's count for that gene;
  no value is a blend of two donors.
- `test_retrieval_attention_becomes_selective` — **carried over from GATE 2 (T04), which could not
  test it.** G2.4 requires mean attention entropy over the `K` retrieved donors to exceed
  `0.5 log K`; it is a one-sided criterion, written to catch *collapse* onto a single donor. T04's
  linear probe passed it at **0.987 × log K** — i.e. at the opposite extreme, near-uniform: the
  attention was **averaging** its 32 donors, not selecting among them. Averaging is a safe default
  and a useless one: it is what makes the retrieval branch equivalent to a fixed kernel smoother,
  and it is the reason G2.4 alone cannot show the branch is load-bearing.

  So T06's requirement is a *direction*, not a floor. With the flow-matching head trained, mean
  attention entropy must move **DOWN** from T04's 0.987 × log K — record the number and assert a
  fall of at least 0.05 × log K — while staying above the `0.5 log K` collapse line. Both bounds
  matter and they are different failures: no movement means the head never learned that some donors
  are better evidence than others, and a fall through 0.5 log K means it has collapsed to copying
  its nearest neighbour and will fail on wide gaps. Log the value every epoch beside the collapse
  alarm on per-gene variance (T07 §3), and put the trajectory in `reports/benchmark.md`.

## Definition of done

On the fixture, generated sections match the real held-in section on: detection rate (r > 0.95),
gene–gene covariance (better than the independent-donor baseline by ≥ 2×), and mean–variance slope.
`PROGRESS.md` records these three numbers.

**Amended at T06, with the measurements, in the same two places the acceptance tests were:** the
covariance clause is unachievable as stated (the ≥ 2× is below the ceiling — SPEC_QUESTIONS B16) and
is replaced by the isolated chimerism measurement plus a ceiling-relative report of every arm; the
detection clause is stated on the default `alternating` holdout (B17). The three numbers
`PROGRESS.md` must record become five: detection rate `r` and MAD, the chimerism table, the
ceiling-relative Frobenius numbers for the model and both baselines, and the mean–variance slope.

**The model's own shortfall is not amended away.** At the T06 test budget the model's Frobenius error
is *worse* than the baseline's, and the reason is now measured: the expression head **overfits the
likelihood**. Doubling the budget 1200 → 2400 steps lowers the reconstruction NLL (1.589 → 1.578
nats/pair) while every distributional statistic of the generated section deteriorates — Frobenius
17.7 → 21.3, detection MAD 0.056 → 0.069, covariance magnitude 0.173 → 0.175 against a real 0.1425.
This is B10's shape (a likelihood that keeps improving while the fit walks away from the truth) on
the expression head rather than the intensity head, and T06's loss set has nothing that could stop
it: the terms that constrain *distributional agreement* are T08's (`w_autocorr`, `w_profile`,
`w_distribution` — weights that exist with no terms yet to weight) and the mean–variance and
detection calibrators are T09 §2's. Carried as open risk **R4**.

## Do NOT

- Do not emit `mu` as the output anywhere in the generation path.
- Do not decode genes through a fixed-width output layer (breaks open-vocabulary).
- Do not let normalised expression become the ZINB target.
- Do not skip the independent-donor baseline — it is needed here and again in T10.
