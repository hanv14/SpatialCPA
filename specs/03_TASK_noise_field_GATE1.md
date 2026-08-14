# T03 — 3D Gaussian random field prior  ⛔ **GATE 1**

**Goal.** Replace the usual i.i.d. flow-matching prior with a **continuous, spatially correlated 3D
random field**. This is the single mechanism the whole method rests on. Two prior versions of this
project failed because generated sections lost spatial autocorrelation; a correlated prior is the
fix. It also makes any two virtual sections of the same tissue mutually consistent, because they
are slices of one field.

**Files:** `spatialcpav25_gen/model/noise.py`, `tests/test_noise.py`, `scripts/gate1_report.py`

**Dependencies:** T01.

---

## 1. Why a field and not a graph filter

An obvious cheaper implementation is: build a kNN graph on the generated cells and smooth i.i.d.
noise over it. **Do not do this.** It defines noise only on the sampled point set, so two sections
at different angles get independent noise and disagree where they intersect — which destroys the
central capability claim (mutually coherent in-silico sectioning). The noise must be a function of
*physical position*, evaluable at any point in R³, reproducible from a seed.

## 2. Interface — `spatialcpav25_gen/model/noise.py`

```python
class GaussianRandomField:
    """A continuous, anisotropic Matern GRF over R^3 with d_h output channels.

    xi(p) has zero mean, unit marginal variance per channel, and
    Cov(xi_c(p), xi_c(q)) ~= matern_nu( ||p - q||_ell ),  channels independent.
    """
    def __init__(self, cfg: Config, ell: tuple[float, float, float], seed: int): ...

    def __call__(self, xyz: np.ndarray | Tensor) -> Tensor:
        """xyz: (N, 3) physical coords -> (N, d_h) noise, deterministic given seed."""

    def with_lengthscale(self, ell: tuple[float, float, float]) -> "GaussianRandomField":
        """Same seed/frequencies rescaled — used by the inference-time calibrator."""
```

## 3. Implementation: random Fourier features

For a Matérn kernel with smoothness ν and anisotropic length-scale `ell = (ℓx, ℓy, ℓz)`:

```
Draw, once, from the seeded generator:
    z_m  ~ N(0, I_3)                      m = 1..M           (M = cfg.n_rff)
    g_m  ~ Gamma(shape=nu, scale=1/nu)    (mean 1)
    omega_m = (z_m / sqrt(g_m)) / ell     # elementwise divide by (lx, ly, lz)
    b_m  ~ Uniform(0, 2*pi)
    A    ~ N(0, 1)  of shape (M, d_h)

Then:
    phi(p) = sqrt(2 / M) * cos(omega @ p + b)      # (N, M)
    xi(p)  = phi(p) @ A                            # (N, d_h)
```

The Gamma scale mixture of Gaussians is the Matérn spectral density; ν → ∞ recovers the squared
exponential. **Verify this empirically rather than trusting the derivation** — the acceptance test
below compares the realised covariance against `scipy`'s analytic Matérn.

Notes:
- Store `omega`, `b`, `A` as buffers. `__call__` must be pure.
- Renormalise `A` columns so realised marginal variance is 1.0 ± 0.02 (measure on a random sample
  of points at construction and rescale).
- `with_lengthscale` divides the *original* `z_m/sqrt(g_m)` by the new `ell` — it must not redraw,
  or the calibration loop in T09 will jitter between iterations.
- Provide a `torch` path (differentiable w.r.t. `xyz`, not w.r.t. the random draws) and a numpy path.
- M = 4096 is the default; the test should show the covariance error decreasing with M.

## 4. Anisotropy

`ell` has three components. In-plane (`ℓx = ℓy = ell_xy`) is fitted from the training sections'
in-plane expression autocorrelation; `ℓz` from between-section correlation decay. **This anisotropy
is what makes oblique sections quantitatively correct** — with an isotropic `ell`, a section at 45°
gets an in-plane correlation length wrong by a factor that depends on the angle. Implement
`fit_lengthscale_from_sections(vol, cfg) -> tuple[float,float,float]` here (a simple approach:
compute the empirical semivariogram of the top-50 PCs in-plane and across z, fit a Matérn by least
squares). It is used at inference (T09) and by GATE 1.

---

## ⛔ GATE 1 — acceptance criteria

`scripts/gate1_report.py` runs the following on the synthetic fixture and writes
`reports/gate1.md` with numbers and plots. **All four must pass before T04 begins.**

### G1.1 — Covariance correctness
Sample 4000 random point pairs; compare empirical `Cov(xi(p), xi(q))` against the analytic
anisotropic Matérn. Mean absolute error < 0.03; error decreases monotonically as M goes
1024 → 2048 → 4096.

### G1.2 — Exact intersection consistency
Define two non-parallel planes intersecting inside the bounding box. Sample 256 points along the
intersection line. Query the field via each plane's own coordinate construction. Assert
**bitwise-identical** outputs (`torch.equal`). This is the property that makes multi-angle sectioning
coherent; if it fails, the plane→coords code has a bug, not the field.

### G1.3 — Autocorrelation transfer (**the decisive test**)
This checks the actual claim: correlated prior → preserved spatial autocorrelation after a
generative map.

```
Take the synthetic fixture's ground-truth generative map f: (latent, position) -> expression.
Substitute the latent with (a) i.i.d. N(0,I) noise, (b) GRF noise at the fitted ell.
Generate a section under each. Compute per-gene Moran's I and Geary's C on a fixed kNN graph.
Compare to the real section's I / C.
```

Required:
- **(b) beats (a) by a wide margin**: median |I_gen − I_real| for the GRF prior must be **< 50%** of
  the i.i.d. prior's, and Pearson r between per-gene `I_gen` and `I_real` must be **> 0.7** for GRF
  and materially lower for i.i.d.
- Sweeping `ell` from 0.25× to 4× the fitted value must move median `I_gen` **monotonically**, and
  the fitted `ell` must land within 25% of the value that best matches `I_real`. (This is what makes
  the T09 calibration loop well-posed — if `I_gen` is not monotone in `ell`, bisection will fail.)

### G1.4 — Determinism and scaling
Same seed → identical output across two processes. Querying 10⁶ points takes < 5 s on CPU at
M = 4096.

**If G1.3 fails, stop.** Write up what was observed in `reports/gate1.md` and report back rather
than proceeding. The rest of the design assumes this mechanism works, and there is no cheap
substitute for it.

---

## Other acceptance tests

- `test_grf_zero_mean_unit_var` — over 10⁵ points, per-channel mean |µ| < 0.02, var within [0.97, 1.03].
- `test_grf_channels_independent` — mean |corr| between channels < 0.02.
- `test_with_lengthscale_preserves_seed` — halving `ell` twice equals quartering it once (same draws).
- `test_torch_numpy_agree` — max abs diff < 1e-5.
- `test_fit_lengthscale_recovers_truth` — on the fixture (built with a known `autocorr_length`),
  the fitted `ell_xy` is within 30% of truth.

## Definition of done

`reports/gate1.md` exists with all four gate criteria passing, plots included, and a one-paragraph
plain-language summary of what the numbers mean. `PROGRESS.md` updated.

## Do NOT

- Do not implement the graph-smoothing shortcut (§1).
- Do not make the field depend on the generated point set in any way.
- Do not proceed to T04 on a partial pass.
