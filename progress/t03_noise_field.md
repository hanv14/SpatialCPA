# T03 — 3D GRF noise field (GATE 1)

Part of [PROGRESS.md](../PROGRESS.md).

### T03 — 3D Gaussian random field prior ⛔ **GATE 1 FAILED** (2026-08-15)

**Built.** `spatialcpav25_gen/model/noise.py` — `GaussianRandomField` (RFF Matérn field over R³:
`__call__` / `forward` torch path, `evaluate_numpy`, `covariance`, `with_lengthscale`, buffers
`directions` / `omega` / `phase` / `amplitude`), `matern_correlation`, `scaled_distance`,
`fit_lengthscale_from_sections` + `fit_lengthscale_details` (`LengthscaleFit`),
`LengthscaleFitWarning`, `VariogramError`. `tests/gate1_criteria.py` (the four gate criteria as
`Criterion` records, measured once and consumed by both the test suite and the report),
`tests/fixtures/planes.py` (a minimal canonical `SamplingPlane` + `intersection_line`, superseded by
T07's `infer/planes.py`), `tests/test_noise.py` (31 tests), `scripts/gate1_report.py`,
`reports/gate1.md` + four figures. 13 new `Config` fields (`variogram_*`), one changed default
(`grf_chunk_points` 65536 → 1024); no constants outside `Config` in the package.

**Test numbers.** `make check` green: ruff clean, `mypy --strict` clean on 9 source files,
**75 tests pass in 40 s** on CPU (31 of them T03's, 5 marked `slow`; budget 3 min). `make test-all`
is **1 failed, 75 passed**: the one failure is `test_gate1_3_autocorrelation_transfer`, which is the
failed gate asserting itself. It is deliberately left red rather than `xfail`-ed — a gate that stops
the project should not be green.

| Acceptance test | Required | Measured |
|---|---|---|
| `test_grf_zero_mean_unit_var` | \|µ\| < 0.02, var ∈ [0.97, 1.03] over 10⁵ points | max \|µ\| **0.0105**, var **[0.983, 1.013]** |
| `test_grf_channels_independent` | mean \|corr\| < 0.02 | **0.0028** (and exactly 0 by construction: max \|AᵀA\| off-diagonal **< 1e-9**) |
| `test_with_lengthscale_preserves_seed` | halve twice == quarter once | bitwise equal on `omega`, `phase`, `amplitude` and on the field values |
| `test_torch_numpy_agree` | max abs diff < 1e-5 | **4.2e-6** |
| `test_fit_lengthscale_recovers_truth` | fitted `ell_xy` within 30 % of 120 µm | **137.6 µm (+14.7 %)** |
| `test_matern_correlation_matches_sklearn` | — | max diff vs `sklearn.Matern` **< 1e-10** at ν = 0.5 / 1.5 / 2.5 / 4.0 |

**GATE 1 — FAILED on G1.3c and G1.3d.** Full report with plots in `reports/gate1.md`;
`scripts/gate1_report.py` exits non-zero.

| Criterion | Required | Measured | Verdict |
|---|---|---|---|
| G1.1a covariance vs analytic anisotropic Matérn, 4000 pairs, M = 4096 | MAE < 0.03 | **0.0121** | PASS |
| G1.1b error decreases with M | every step < 0 | **0.0232 → 0.0159 → 0.0121** (≈ 1/√M) | PASS |
| G1.2 two plane pathways along their intersection line, 256 points | bitwise identical | **max diff exactly 0.0** (coords agree to 2.8e-14 µm and round to the same float32) | PASS |
| G1.3a median \|I_gen − I_real\|, GRF ÷ i.i.d. | < 0.5 | **0.196** (0.0776 vs 0.3956) | PASS |
| G1.3b per-gene Pearson r(I_gen, I_real), GRF | > 0.7 | **0.820** (i.i.d. arm: 0.348) | PASS |
| G1.3c median I_gen monotone over `ell` = 0.25×–4× | every step > 0 | **−0.065** (0.271, 0.390, 0.418, 0.413, 0.348 — a peak at ≈ 1.6×) | **FAIL** |
| G1.3d fitted `ell` vs the best-matching `ell` | < 25 % | **37 %** (fitted 137.6 µm, best match 218.4 µm) | **FAIL** |
| G1.4a same seed in a second process | 0 differing values | **0** | PASS |
| G1.4b 10⁶ points at M = 4096, d_h = 64 | < 5 s | **3.4 s** (4-core Xeon @ 2.10 GHz, no GPU) | PASS |

**What the failure is, and what it is not.** It is *not* a failure of the prior: measured directly,
the median Moran's I of the field's own channels is monotone in `ell` and saturates near 1
(0.487 → 0.764 → 0.900 → 0.973 → 0.990 over the same sweep, diagnostic D2). The non-monotonicity is
in the *observable*. Moran's I of expression is a ratio — spatially structured variance over total
variance — and a stationary unit-variance field seen through a finite window loses within-window
variance once `ell` approaches the window: across the sweep the field's within-section sd falls
0.98 → 0.65 on the fixture's 1000 µm section, so the latent contributes less of the total expression
variance even as its neighbour-scale correlation approaches 1, and the ratio turns over. Rebuilding
the fixture at a 3000 µm field of view with the same density (diagnostic D1) nearly removes the
effect (0.212, 0.363, 0.470, 0.488, 0.4825 — worst step −0.005, inside the ±0.006 count-draw noise)
and makes `I_real` reachable, which locates the cause in `ell` / field-of-view rather than in the
field. G1.3d fails for the same reason: the best-matching `ell` is the argmax of a curve that never
quite reaches median `I_real` (0.445 at the peak vs 0.472 real).

**Consequence, and the recommendation.** The spec states G1.3c's purpose in its own words — "this is
what makes the T09 calibration loop well-posed — if `I_gen` is not monotone in `ell`, bisection will
fail". That risk is now quantified: `I_gen(ell)` is unimodal with its maximum near 0.2× the section's
in-plane extent, so T09's bisection is well-posed only when bracketed below that maximum, and when
the target exceeds the achievable maximum there is no root at all. My recommendation is to make
T09's calibrator maximum-aware (cap the bracket, detect the maximum, report "target unreachable")
and re-run G1.3c over the range calibration actually operates in — *not* to widen the fixture and
*not* to move a threshold. Recorded as SPEC_QUESTIONS **A7**. **T04 is not started.**

**Deviations from the spec, and why.**

1. **Amplitude normalisation is analytic, not measured.** The spec says to "renormalise `A` columns
   so realised marginal variance is 1.0 ± 0.02 (measure on a random sample of points and rescale)".
   The space-averaged marginal variance is exactly `‖A_c‖²/M`, so the columns are orthogonalised
   (QR, sign-canonicalised) and scaled to `‖A_c‖ = √M`. That makes unit variance and zero
   cross-channel covariance *exact* rather than a Monte Carlo approximation of them, and it settles
   SPEC_QUESTIONS **B3** (the 0.02 cross-channel threshold was at the O(1/√M) ≈ 0.016 noise floor;
   it is now 0 by construction). `E[A_c A_cᵀ] = I` still holds, so the covariance function is
   unchanged — G1.1 checks that empirically.
2. **Queries are evaluated in float32, not float64.** Coordinates are float32 by Convention 4, so a
   float64 phase would chase precision the inputs do not carry, and it triples the cost (the
   `(chunk, M)` block stops fitting in cache). The draws stay float64 so `with_lengthscale` does not
   accumulate rounding across a calibration loop. Measured worst-case phase error ~2e-5 rad → ~2e-5
   on a unit-variance field.
3. **`grf_chunk_points` default 65536 → 1024.** Purely performance: at 65536 the feature block is
   1 GB and the query is memory-bound (5.7 s per 10⁵ points); at 1024 it is 16 MB, cache-resident,
   and 10⁶ points take 3.4 s. Settles SPEC_QUESTIONS **B9** — the 5 s target is reachable on this
   CPU after chunking, so nothing had to be loosened.
4. **`fit_lengthscale_from_sections(vol, cfg, *, seed)`** takes a required keyword-only seed
   (Convention 3 — the per-section cell subsample is stochastic), and refuses `HeldOutSections` with
   a `TypeError` naming leakage. Same shape of deviation as T01's `split_holdout(..., cfg=None)` and
   T02's `text_embedding_diagnostics(..., seed)`.
5. **`fit_lengthscale_details` is an added entry point.** Same computation, returning the curves that
   were fitted (`LengthscaleFit`), because the report has to plot them and a surprising `ell` has to
   be diagnosable. `fit_lengthscale_from_sections` is the spec's signature and returns `.ell`.
6. **`covariance(p, q)` is an added method**, and GATE 1's G1.1 uses it. Estimating the covariance by
   Monte Carlo over the `d_h` channels — the obvious reading of "empirical Cov" — has a noise floor
   of 1/√d_h ≈ 0.125 per pair, four times the criterion's own threshold, and no dependence on `M`
   whatsoever: measured that way the error is 0.0092 / 0.0096 / 0.0095 across M = 1024 / 2048 / 4096,
   i.e. the M-trend the criterion is about is invisible under the sampling noise of the estimator.
   `covariance` computes the same quantity exactly over the amplitude draw, which is where the
   approximation error actually lives. Both numbers are in the report (G1.1c carries the MC
   cross-check: 0.140 against the predicted 0.125).
7. **The plane geometry for G1.2 lives in `tests/fixtures/planes.py`**, not in the package: T07 owns
   `infer/planes.py`, and building it early would be skipping ahead. It is deliberately canonical —
   one `(u, v) → (x, y, z)` function — which is what SPEC_QUESTIONS **B1** asked for, and with it
   the spec's bitwise requirement is met outright rather than relaxed: the two float64
   reconstructions agree to 2.8e-14 µm and round to identical float32 coordinates, so the field
   values are bit-identical.
8. **G1.1's monotone-in-M claim is averaged over 8 realisations** (SPEC_QUESTIONS **B2**), and the
   reported number is the mean of the per-realisation errors, not the error of an averaged field —
   averaging fields would have made the criterion easier rather than more reliable.
9. **Two diagnostics were added to the report** (D1: the same sweep at a 3000 µm field of view;
   D2: Moran's I of the field itself). They are labelled "not gate criteria" and carry no
   thresholds; they exist because a failed gate is worth diagnosing.
10. `tests/test_schema.py` was reformatted by the pinned `ruff format` (0.4.4) again — the same
    formatting-only drift T02 recorded as its deviation 9, reintroduced by commit `94dff64`.

**Flagged for later.**

* **T09 must not bisect blindly.** See SPEC_QUESTIONS A7: cap the bracket at a fraction of the
  in-plane extent and handle "target Moran's I unreachable" explicitly. `Config.bisection_grid_size`
  (12) already exists for the fallback; the fallback now has a defined job.
* **`ell_z` is extrapolated on this fixture.** The along-z variogram reaches only 59 % of its fitted
  sill at the largest available lag (400 µm), so the fitted 355 µm against a ground truth of 200 µm
  is a low-confidence number and `fit_lengthscale_from_sections` warns
  (`LengthscaleFitWarning`, `Config.variogram_min_saturation = 0.75`). Real stacks are deeper; a
  9-section 400 µm fixture cannot see a 200 µm correlation decay away. T09's `ell_z` calibration
  should treat the fitted value as a starting point, not a measurement.
* **Per-channel-group `ell`** (SPEC_QUESTIONS A2) is *not* implemented: the spec's interface takes
  one `ell` and nothing in `specs/` consumes a grouped one. The door stays open cheaply —
  `with_lengthscale` builds a rescaled field from the same draws, so a grouped field is a
  channel-wise concatenation of those, no redraw and no new state.

### T03 (amended) — GATE 1 re-run and passed (2026-08-15)

The first T03 pass reported GATE 1 failed on G1.3c (monotonicity) and G1.3d. That verdict was
accepted as a **conditional pass**: the mechanism criteria pass decisively, and the two failures were
a spec defect (a sweep range wider than the statistic's monotone branch) plus a fixture-realism
defect (a 1000 µm field of view that real data never occupies), not implementation defects. Six
amendments, then a re-run.

**1. The gate fixture is now 3000 µm.** `make_synthetic_volume` gained `cell_density`
(default `1.5e-3` cells/µm², the density it already had) and `n_cells_per_section=None`, which
derives the count from `cell_density * extent_xy**2`. The default fixture is bitwise unchanged —
1500 cells, 13.947 µm median NN distance, the numbers T01/T02 measured against — and the gates build
`extent_xy=GATE_EXTENT_UM` (3000 µm, 13 500 cells/section, same density, ~31 s). *No new `extent`
parameter was added:* `extent_xy` already was that parameter, and a second name for one dimension
would be a trap. New session-scoped `gate_volume` / `gate_gt_field` fixtures keep it out of
`make test`.

**2–3. Specs amended.** `specs/03` records the turnover as a known property of the statistic (with
the D2b variance explanation), restates **G1.3c** over the calibration bracket, adds **G1.3g**
(unimodal; maximiser ≥ the fitted `ell`), and states which fixture the gate is measured on and why.
`specs/09` §2 replaces the bisection with: cap the bracket at
`min(calibration_ell_max_extent_frac × extent, calibration_ell_max_fitted_multiple × fitted ell)`,
**locate the maximum on a `bisection_grid_size` log grid and bisect only below it**, and return a
`LengthscaleCalibration` carrying `status ∈ {converged, target_unreachable, boundary}` — never a
bracket endpoint dressed up as convergence. Three acceptance tests are specified, including
`test_calibrator_reports_unreachable_target` for the branch T03 measured.
`specs/11_COVERAGE_MATRIX.md` gains the row and the amended GATE 1 statement.

**4. G1.4b is now throughput against reference hardware, not a wall clock.** It is a REPORT row
(points/s + the machine), because the same code measured 3.4 s here and 6.1 s on an Apple-silicon
laptop — a 5-second threshold made the gate a statement about whose machine ran it. The assertable
half is dimensionless and new: **G1.4c**, 8× the points must cost < 12× the time (measured 6.6–8.0×;
quadratic would be 64×).

**5. The float32 tests.** `evaluate_numpy` now delegates to the torch path instead of
re-implementing the arithmetic in numpy: a second float32 matmul reassociates differently, by a few
1e-6 that vary with the BLAS vendor, so the two paths could only ever have been compared with a
tolerance. They are now equal by construction and `test_torch_numpy_agree` asserts `array_equal`.
The batch-shape assertions are the other half: **identical points in an identically shaped batch are
bitwise identical** (that is the contract G1.2 and T07's `L_cross` rest on, and it is asserted with
`torch.equal`), while a *differently shaped* batch — a one-row query, a different
`grf_chunk_points` — is asserted to 1e-5, because float32 GEMM is free to reassociate its sum over
the M features and torch dispatches a matrix-vector kernel for one row. Padding to fake exactness
would have implied a guarantee no BLAS gives. Split into `test_field_is_pure_and_depends_only_on_position`
(bitwise) and `test_batch_shape_changes_nothing_that_matters` (1e-5).

**6. Lint.** `ANN101`/`ANN102` are gone from the ruff ignore list, and with them the pin had to move:
those rules do not exist in modern ruff (the ignore entry itself is an error), and under the pinned
0.4.4 they fire on every `self`. `ruff==0.4.4 → 0.14.14`, which also ends the formatter tug-of-war
that had `tests/test_schema.py` flipping between machines — 0.14.14 leaves `main`'s version
untouched. `mark-parentheses = false` is now explicit beside `fixture-parentheses`, so
`@pytest.mark.slow` lints clean under both old and new ruff.

**Two implementation changes came out of the re-run, both judged against ground truth, not against
the gate.**

* **Cressie weights in the variogram fit.** Bins are weighted `N(h)/γ(h)²` rather than `N(h)`: the
  estimator's variance grows with `γ(h)²`, so pair counts alone make the fit a fit to the large
  lags. Against the fixture's 120 µm ground truth: 3000 µm FOV 95.2 → **102.9 µm** (−21 % → −14 %),
  1000 µm FOV unchanged at 137.6 → 141.5 µm.
* **`variogram_n_ell_grid` 48 → 128.** At 48 the grid steps are 12 %, so the fitted value visibly
  jumped between two adjacent grid points as the subsample seed changed; 4 % steps are comfortably
  finer than the 25 % tolerance anything reports against.

**GATE 1 — PASS.** `reports/gate1.md`, `scripts/gate1_report.py` exits 0.

| Criterion | Required | Measured | |
|---|---|---|---|
| G1.1a covariance vs analytic anisotropic Matérn | MAE < 0.03 | **0.0121** | PASS |
| G1.1b error decreases with M | every step < 0 | **0.0232 → 0.0159 → 0.0121** | PASS |
| G1.2a–d two plane pathways along the intersection line | bitwise | **max diff exactly 0.0** | PASS |
| G1.3a Moran's I error, GRF ÷ i.i.d. | < 0.5 | **0.130** (0.0552 vs 0.4233) | PASS |
| G1.3b per-gene r(I_gen, I_real) | > 0.7 | **0.917** (i.i.d.: 0.377) | PASS |
| G1.3c monotone over the calibration bracket | every step > 0 | **+0.028** (bracket = 0.25×–2× fitted, 26–206 µm) | PASS |
| G1.3d fitted vs best-matching `ell` | < 25 % | **8.0 %** (102.9 vs 111.8 µm) | PASS |
| G1.3g-a `I_gen(ell)` unimodal | violation < 2 SE = 0.0069 | **0.0000** | PASS |
| G1.3g-b maximiser ≥ fitted `ell` | ≥ 1× | **2.52×** (259 µm = 0.086 × extent) | PASS |
| G1.4a same seed, second process | 0 differing | **0** | PASS |
| G1.4b throughput (recorded) | — | **2.0–3.5 × 10⁵ points/s** across runs, reference 4-core Xeon @ 2.10 GHz | REPORT |
| G1.4c 8× points vs time | < 12× | **6.6–9.7×** | PASS |

`make check` green (ruff, `mypy --strict` on 9 files, **73 fast tests in 32 s**); `make test-all`
**79 passed in 2 min 1 s**, gate tests included.

**What the wider fixture changed, and what it did not.** The mechanism numbers improved
(error ratio 0.196 → 0.130, r 0.820 → 0.917) because the window artefact was suppressing them. The
turnover did **not** go away and was never going to: measured, the maximiser sits at
**0.086 × extent** at 3000 µm and **0.112 × extent** at 1000 µm — close in *window* units, but
**2.52× vs 0.79× the fitted `ell`**, because the variogram fit is itself window-biased at a narrow
field of view. Two consequences are now in `specs/09`: the spec's own 0.25×–4× sweep is wider than
the monotone branch at *any* field of view, and the 0.2 × extent cap is about twice the measured
maximiser, so it does not bind protectively — here the `2 × fitted` cap is what keeps the bracket
below the peak. **Neither cap is a guarantee; T09's maximum detection is the real protection.** I did
not lower `calibration_ell_max_extent_frac` to make the criterion pass on the narrow fixture: the
value is the one specified, and the measurement that would justify changing it is the same one the
criterion checks.

### T03 (amended, second pass) — bitwise batch stability, mypy, and the `ell_z` risk (2026-08-15)

**1. `mypy --strict` is green again.** The redundant cast in `_check_ell` is gone: the triple is now
built by unpacking (`ell_x, ell_y, ell_z = (float(v) for v in raw)`) instead of `tuple(...)` plus a
cast, so the return type is the fixed-length tuple the signature promises under any mypy version.
The two remaining casts (`scaled_distance`, `evaluate_numpy`) are still load-bearing under the pinned
numpy 1.26 stubs.

**2. Batch-size stability is now bitwise, by padding — the reframing was wrong and is reverted.**
`forward` evaluates **fixed-size** chunks and zero-pads the last one, so every matmul in every query
is `(grf_chunk_points, M) @ (M, d_h)`: same shape, same kernel, same reduction order over the M
features, whatever `N` is. Measured on the reference box, a query of **1, 2, 137, 400, 1023, 1024,
1025, 4000 or 4096** points is now `torch.equal` to the same points inside a 4096-point batch —
including the 1-row case that previously landed 1.03e-05 away because torch dispatched a
matrix-*vector* kernel. `test_batch_shape_does_not_change_a_single_value` asserts exactly the sizes
requested, with `torch.equal`. Cost: at most `chunk - 1` wasted rows per query (a 1-point query now
costs what a 1024-point one does, ~3 ms); throughput at 10⁶ points is unchanged.

**One case remains non-bitwise, and it is not the one L_cross depends on.** Two fields whose
`Config.grf_chunk_points` *differ* can disagree by ~2e-6: the chunk size **is** the matmul's row
count, so changing it changes the shape and may change the kernel. Padding cannot remove that — there
is no shape that is simultaneously two shapes. Measured: 0.0 between chunks of 97, 512 and 1024 rows,
2.0e-6 against a single 100000-row matmul. Nothing load-bearing rests on it: `grf_chunk_points` is a
frozen `Config` field, so within one run — and therefore within `L_cross`, which queries one field
object twice — every query goes through the same shape and the values are bitwise identical. G1.2's
"exact by construction" stands as written, and `test_chunk_boundaries_do_not_change_a_single_value`
records both halves: `torch.equal` across chunk *boundaries* at a fixed chunk size, `< 1e-5` across
different chunk sizes.

**3. `ell_z` is recorded as open risk R1** (see the section above), carried to T04 and T07 with the
three candidate remedies and a decision due before T07.

`reports/gate1.md` regenerated; all gate criteria unchanged in verdict. `make check` green
(73 fast tests), `make test-all` green (79 tests).
