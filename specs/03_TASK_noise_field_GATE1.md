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

**The gate fixture is the 3000 µm one** (`make_synthetic_volume(seed=0,
extent_xy=GATE_EXTENT_UM)`), not the 1000 µm volume the fast suite builds. Cell density is
held fixed, so it is the same tissue seen through a wider window. *Why:* real sections are
5–10 mm wide with correlation lengths of 50–200 µm, i.e. `ell / FOV ≈ 0.02`. At a 1000 µm
field of view the top of this section's own `0.25×–4×` sweep sits at **55 % of the extent**,
a regime real data never occupies, and the window artefact described under G1.3 below then
dominates the measurement. The 1000 µm numbers are kept in `reports/gate1.md` as diagnostic
D1 — the finding is real and must not be deleted.

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

> **Known property of the statistic (measured, T03).** Moran's I of *expression* is a ratio:
> spatially structured variance over total variance. As `ell` grows two things happen — the
> field's neighbour-scale correlation rises towards 1 (and saturates), and its variance
> *within a finite window* falls, because a stationary unit-variance field observed through
> a window loses variance once its correlation length approaches that window. On the gate
> fixture the field's within-section standard deviation falls 1.00 → 0.87 across the sweep
> (diagnostic D2b) while the field's own Moran's I rises 0.38 → 0.99 monotonically (D2a).
> Where the two effects cross, `I_gen(ell)` turns over. **`I_gen(ell)` is therefore unimodal,
> not monotone**, and the maximum sits near 0.09–0.11 of the in-plane extent (0.086 on the
> 3000 µm fixture, 0.112 on the 1000 µm one). The prior itself is monotone in `ell`
> throughout; the turnover belongs to the observable.

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
- **G1.3c (restated).** Sweeping `ell` from 0.25× to 4× the fitted value, median `I_gen` must
  move **monotonically over the sweep points that lie inside the calibration bracket**, i.e.
  those with

      ell <= min(Config.calibration_ell_max_extent_frac * in-plane extent,
                 Config.calibration_ell_max_fitted_multiple * fitted ell)

  The original criterion asked for monotonicity over the whole 0.25×–4× range. That is false
  at every field of view — see the box above — because the range is wider than the monotone
  branch, and it was failing on a property of Moran's I rather than of the prior. What the
  criterion is *for* is stated in the spec's own words ("this is what makes the T09
  calibration loop well-posed — if `I_gen` is not monotone in `ell`, bisection will fail"),
  so it is now stated over exactly the range T09 will bisect on.
- **G1.3d.** The fitted `ell` must land within 25% of the value that best matches `I_real`
  (unchanged).
- **G1.3g (new).** `I_gen(ell)` must be **unimodal** — no fall before the maximum and no rise
  after it, beyond twice the standard error of the curve itself — and its **maximiser must be
  at or above the fitted `ell`**, so that a bracket below the maximum is well-posed. This is
  the criterion that carries the calibration content: the caps in G1.3c are a-priori guards,
  whereas this measures where the maximum actually is. Measure the curve on a log grid
  averaged over several field realisations (realisation-to-realisation scatter at fixed `ell`
  is ~0.015, an order of magnitude above the count-draw scatter, so a single draw cannot
  state anything about the *shape* of the curve).

### G1.4 — Determinism and scaling
Same seed → identical output across two processes (bitwise, asserted).

**Throughput is recorded against reference hardware, not asserted as a wall clock.** Report
points/second for 10⁶ queries at M = 4096, d_h = 64, together with the machine it was
measured on. Reference: **2.9 × 10⁵ points/s** (10⁶ points in 3.4 s) on a 4-core Intel Xeon
@ 2.10 GHz with AVX-512, 4 torch threads, torch 2.2.2 CPU, no GPU. The same code and the same
query measured 6.1 s on an Apple-silicon laptop; a 5-second threshold would have made the
gate a statement about whose machine ran it. What *is* asserted is dimensionless and means
the same thing everywhere: **querying 8× the points must cost less than 12× the time** (ideal
8×, anything quadratic 64×), which is what "scaling" in this criterion's title is about.

**If G1.3a or G1.3b fails, stop.** Write up what was observed in `reports/gate1.md` and report back rather
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
