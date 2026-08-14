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
- `r0` = 5th percentile of nearest-neighbour distances pooled over training sections.
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
  < 0.15 over `r ∈ [r0, 3R]`. **Also assert that a pure-Poisson sampler fails this test** — it
  documents that repulsion is load-bearing (ablation A4).
- `test_potts_improves_purity` — neighbourhood type-purity after smoothing is closer to the real
  section's than before, and does not exceed it (over-smoothing check).
- `test_rare_types_survive` — a type at 2% prevalence retains ≥ 50% of its expected count after
  Potts smoothing.
- `test_layout_deterministic` — same seed → identical layout.
- `test_all_three_modes_run` — each produces a valid `Layout` with plausible N.

## Definition of done

On the fixture, `field` mode achieves cell-type localization within 10% of the real section's value
and pair-correlation match per above. `PROGRESS.md` records the three fitted repulsion parameters
and `beta` — they should be reported in the paper's methods.

## Do NOT

- Do not force the sampled count to equal a flanking-section count. It must emerge from the
  intensity integral, or the thickness consistency in T07 becomes incoherent.
- Do not hand-tune `beta`, `gamma`, `r0`. All are fitted from training sections.
- Do not smooth types so hard that rare types vanish.
