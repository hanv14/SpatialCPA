# T05 — Layout head: where the cells are and what they are

**Goal.** Generate the held-out section's *layout* — cell count, positions, and type labels — as a
marked point process driven by a continuous intensity field. This is the direct attack on the
`paper_celltype_localization` metric, and it fixes two structural weaknesses of the competing method
(which optimises a sliced-Wasserstein objective over point coordinates and assigns types by a k=1
nearest-neighbour vote, so it constrains neither local spacing nor spatial type organisation).

**Files:** `spatialcpav25_gen/model/layout.py`, `spatialcpav25_gen/losses/reconstruction.py` (layout part),
`tests/test_layout.py`

**Dependencies:** T01, T03, T04.

---

## 1. Intensity field

```python
class IntensityHead(nn.Module):
    """Per-cell-type spatial intensity lambda_c(x,y,z), in cells per micrometre^3."""
    def forward(self, xyz: Tensor, field_feat: Tensor) -> Tensor:  # (N, n_types), positive
```

`lambda_c = softplus(MLP_c([field_feat, fourier(xyz), region_emb]))`.

**Training loss — inhomogeneous Poisson process NLL:**

```
L_layout = -sum_i log lambda_{c_i}(p_i)  +  integral over the section slab of sum_c lambda_c
```

Estimate the integral by Monte Carlo over `n_mc = 4096` uniform points in the slab volume times the
slab volume. Use the **slab volume**, not area — this is what makes thickness handling correct in
T07, and it makes cell count emerge from the model instead of being an interpolated heuristic.

Compute this on training sections, treating each as a slab of thickness = the true section thickness
(a `Volume` field; default to `median_spacing` if unknown, and record the assumption).

## 2. Sampling positions with repulsion

Real tissue has a hard-core minimum spacing — cells cannot overlap. A pure Poisson sample produces
coincident and clumped points, which corrupts every kNN-graph metric downstream (Moran's I, Geary's
C, neighbourhood enrichment). The competing method's SWD objective does not constrain this either;
this is where you beat it.

```python
def sample_layout(intensity_fn, plane: Plane, cfg: Config, seed: int) -> Layout
```

Algorithm (sequential thinning / Strauss process):

1. `N_expected = MC-integrate(sum_c lambda_c)` over the slab. Draw `N ~ Poisson(N_expected)`.
2. Propose points by rejection sampling proportional to `sum_c lambda_c(p)` (envelope = max over
   an MC sample × 1.1).
3. Accept a proposal `p` with probability `exp(-sum_j phi(||p - p_j||))` over already-accepted
   points, where

```
phi(d) = +inf          if d < r0          (hard core)
       = -log(gamma)   if r0 <= d < R     (soft repulsion, 0 < gamma <= 1)
       = 0             if d >= R
```

4. Stop at `N` accepted points or `20*N` proposals (then warn — this means `r0` is too large for
   the requested density and the intensity and repulsion are inconsistent).

**Fitting `r0`, `R`, `gamma` — leakage-free, from training sections only:**
- `r0` = **1st** percentile of nearest-neighbour distances pooled over training sections
  (`Config.repulsion_r0_percentile`, default `1.0`; `5.0` is kept as the selectable alternative).
  Settled, SPEC_QUESTIONS B6: at the 5th percentile, 5% of *real* pairs are by construction closer
  than `r0`, so `test_hardcore_respected` (no generated pair closer than `r0`) forces the generated
  layout to be strictly more regular than the tissue it imitates — which then pushes against
  `test_pcf_matches_real`. The 1st percentile leaves the soft repulsion (`gamma`, `R`) to carry the
  shape of `g(r)`, which is what it is fitted for. **Record which percentile was used** in
  `reports/benchmark.md` and in `PROGRESS.md`: it is a knob that changes a published point-process
  number, so a run that used the alternative has to say so.
- `R` = distance at which the empirical pair-correlation function `g(r)` first reaches 1.0.
- `gamma` fitted by 1-D search so the *simulated* `g(r)` matches the empirical `g(r)` on training
  sections (minimise L2 over `r ∈ [r0, R]`). Provide `fit_repulsion(vol) -> RepulsionParams`.

## 3. Marks (cell types)

1. Sample `c_i ~ Categorical(lambda_c(p_i) / sum_c lambda_c(p_i))`.
2. **Potts smoothing**: `cfg.potts_iters` rounds of ICM on the kNN graph (k=8) with energy
   `-log lambda_c(p_i) - beta * #{neighbours with type c}`, `beta = cfg.potts_beta`.

Independent marks are too noisy — real cell types are spatially organised in patches, and the
localization metric measures exactly that. But over-smoothing erases rare interspersed types, so:
**fit `beta` by matching the training sections' neighbourhood type-purity distribution**, don't
hand-set it. Add `fit_potts_beta(vol) -> float`.

## 4. Mode gate

`cfg.layout_mode`:
- `"field"` — as above.
- `"hybrid"` — field sampling, then a short sliced-Wasserstein polish (200 steps) toward the union
  of flanking-section coordinate marginals. Borrows the one genuine strength of the competing
  method: exact marginal geometry matching.
- `"resample"` — reuse real flanking-section coordinates directly (the v20 behaviour). This is the
  safety fallback that guarantees no regression.

All three must be implemented and selectable; T09's LOSO selector chooses per dataset.

## Acceptance tests

- `test_poisson_nll_recovers_intensity` — on synthetic data drawn from a known `lambda`, the fitted
  intensity correlates with truth at Pearson r > 0.9 on a grid.
- `test_expected_count_matches` — sampled `N` over 50 seeds has mean within 5% of `N_expected`.
- `test_hardcore_respected` — no pair of sampled points closer than `r0`.
- `test_pcf_matches_real` — simulated `g(r)` vs. real `g(r)` on the fixture: max abs difference
  < 0.15 over **`r ∈ [0, 3R]`**. **Also assert that a pure-Poisson sampler fails this test** — it
  documents that repulsion is load-bearing (ablation A4).

  **The range is `[0, 3R]`, not `[r0, 3R]`; amended at T05 because `[r0, 3R]` cannot fail.** A
  hard-core process differs from a Poisson one only *inside* the correlation hole, and the hole
  ends at about `r0` — because `r0` **is** a low percentile of the nearest-neighbour distances,
  i.e. it is defined to sit at the top of the hole. The original range therefore began exactly
  where the discriminating signal stopped. Measured on the synthetic fixture (real `g` pooled over
  the six training sections, simulated over three seeds):

  | | over `[r0, 3R]` (as originally written) | over `[0, 3R]` |
  |---|---|---|
  | `field` mode | 0.093 | **0.093** |
  | pure Poisson, ablation A4 | 0.070 — ***passes*** | **0.994** — fails |

  `[0, 3R]` is strictly harder (it is a superset of the old range), the field sampler passes it
  unchanged, and the control fails it by an order of magnitude. `tests/test_layout.py` measures
  **both** ranges and asserts the old range's blindness as a fact, so this cannot silently
  regress. See SPEC_QUESTIONS B12, which also records that the bin straddling `r0` is
  resolution-sensitive: at 48 bins rather than `Config.pcf_n_bins = 24` the generated pattern reads
  0.315 there against the tissue's 0.646, because a strict hard core at the 1st percentile is still
  stricter than the tissue's own minimum (7.90 µm against 7.75 µm). That is B6 surviving at the 1st
  percentile, and it is a property of the method rather than of the estimator.
- `test_potts_improves_purity` — neighbourhood type-purity after smoothing is closer to the real
  section's than before, and does not exceed it (over-smoothing check).
- `test_rare_types_survive` — a type at 2% prevalence retains ≥ 50% of its expected count after
  Potts smoothing.
- `test_layout_deterministic` — same seed → identical layout.
- `test_all_three_modes_run` — each produces a valid `Layout` with plausible N.

## Definition of done

On the fixture, `field` mode achieves **cell-type localization ≥ 0.90 × the *ideal* draw's** (see the
amendment below) and pair-correlation match per above. `PROGRESS.md` records the three fitted
repulsion parameters and `beta` — they should be reported in the paper's methods.

### The localization criterion — **amended 2026-08-16**

**The criterion is `generated ≥ 0.90 × ideal`**, where *ideal* is an independent draw from the
**known generative law** of the synthetic fixture, evaluated on the same held-out section with the
same metric. It is stated on the **mean over the held-out sections**, not per section, for the
reason measured below (the metric's per-type null normalisation makes a single section's score
noisy). Measured: **0.994** (0.7128 against 0.7178).

It was previously "within 10% of the real section's value", which read against the held-out
section's own score asks the layout head to **beat the process that produced the data**: the ideal
draw itself reaches only 0.779 of that self-score. That reading is kept in the test suite as a
**strict xfail** carrying its failing number — 0.776, on 0.909 / 0.613 / 0.806 per section — so the
shortfall stays in the record rather than being reworded away, and so that a later task which closes
it breaks the suite until this section is updated.

**On real data there is no ideal draw**, because there is no known generative law. The referent
there is the **flanking baseline** — the nearest real section's score on the same held-out section,
which is exactly what `layout_mode="resample"` produces — and it is **T10** that reports it, beside
the achievable-ceiling protocol in `specs/10` §1. Measured here for reference: **1.35×**
(1.654 / 1.154 / 1.246), i.e. better than the real-data alternative on every section.

### The measurement behind the amendment

Measured on all three held-out sections of the synthetic fixture
(`paper_celltype_localization`, transcribed from `bench3/evaluate_paper.py`):

| held-out section | **self** (the section scored against itself) | **generated** (`field`) | **ideal** (an independent draw from the fixture's *true* generative law) | **flanking** (nearest real section = `resample`) |
|---|---|---|---|---|
| s02, z = 100 | 0.8730 | 0.7933 | 0.7144 | 0.4797 |
| s04, z = 200 | 0.9353 | 0.5732 | 0.6298 | 0.4966 |
| s06, z = 300 | 0.9581 | 0.7719 | 0.8091 | 0.6193 |
| **mean** | **0.9221** | **0.7128** | **0.7178** | **0.5319** |

Read against the **held-out section's own value**, `field` mode reaches **0.776** of it (0.909 /
0.613 / 0.806 per section) and **fails** the 10% criterion, passing on one section of three. Said
plainly: the generated layout is materially below the held-out section's own localization.

Read against the **flanking real section**, `field` mode is at **1.35×** (1.654 / 1.154 / 1.246),
i.e. it beats the real-data alternative on every section rather than coming within 10% of it.

The third column is what settles the question. The `ideal` arm samples positions and marks from the
fixture's **own** generative composition — an independent draw from the process that produced the
held-out section, which is the best any intensity head can do — and it reaches only **0.779** of the
self-score, statistically the same as the layout head's 0.776. The gap to the self-score is
therefore the metric penalising **realisation noise**: `celltype_localization` compares point clouds
by a Sinkhorn divergence normalised against a within-tissue null, and a *different draw* from the
same law is already about 22% of the way from the section to that null. A criterion of 0.90 against
the self-score asks the layout head to beat the generative process that produced the data.

All three readings are implemented as tests:
`test_localization_within_10_percent_of_ideal` (**the criterion**, 0.994) and
`test_localization_beats_the_real_data_baseline` (the real-data proxy, 1.35×) pass, and
`test_localization_within_10_percent_of_heldout_self_score` is a **strict xfail** carrying the
failing numbers. See SPEC_QUESTIONS B15.

### Why the criterion is a LOSO mean, and what the per-section spread is

The per-section scores are 0.909 / **0.613** / 0.806 against the self-score, and the low one is
**`synthetic_s04`, the exact centre of the nine-section stack** — index 4 of 9, four sections from
either end, the furthest possible from a boundary. **This is not open risk R3.** R3 predicts a
deficit at the *ends*, and `alternating` never holds out an end section at all, so the boundary
regime is not represented in this table.

What it is instead is the metric's own instability, and the `ideal` arm shows it (per-type breakdown
of that arm, `d_null` = the within-tissue null divergence the score is normalised by):

| | type 0 (weight 0.34) | localised minority types |
|---|---|---|
| `d_null` | **0.072 – 0.087** | 0.079 – 0.573 |
| score at s04 | **0.332** | 0.595 – 0.908 |
| score at s06 | **0.841** | 0.630 – 0.906 |

`celltype_localization` scores a type as `1 - d_obs / d_null`, and `d_null` is the divergence
between the type's cloud and an equally sized random draw from the whole section — so for a type
occupying a third of the tissue, which *is* nearly tissue-wide, the denominator collapses to ~0.08
while a localised type's is 0.16–0.57. The same absolute realisation noise in `d_obs` therefore
costs the abundant type four to eight times as many score points, and it carries a third of the
weight. Type 0's `d_obs` of 0.058 at s04 against 0.011 at s06 is essentially the whole 0.18 spread
between those sections. `evaluate_paper` guards this only at `d_null < 1e-4` ("already spread
tissue-wide: nothing to test"), which is three orders of magnitude below where the instability
actually bites.

Two consequences, both carried into `specs/10` §1: the criterion here is stated on the **mean over
held-out sections**, and T10 reports **per-type** ceilings rather than only the weighted average,
because an abundant tissue-wide type is where the headroom is smallest and the variance largest.

## Do NOT

- Do not force the sampled count to equal a flanking-section count. It must emerge from the
  intensity integral, or the thickness consistency in T07 becomes incoherent.
- Do not hand-tune `beta`, `gamma`, `r0`. All are fitted from training sections.
- Do not smooth types so hard that rare types vanish.
