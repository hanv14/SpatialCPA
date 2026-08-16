# T09 — Inference, leakage-free calibration, and automatic configuration

**Goal.** Generate a section at any plane/thickness; calibrate the free statistical parameters using
**training sections only**; and choose the per-dataset configuration by internal LOSO so the user
never tunes a flag. The last point is both a usability claim in the paper and the guarantee that the
method cannot regress below the previous version.

**Files:** `spatialcpav25_gen/infer/generate.py`, `spatialcpav25_gen/infer/calibrate.py`, `spatialcpav25_gen/train/select.py`,
`tests/test_generate.py`, `tests/test_calibrate.py`

**Dependencies:** T01–T08.

---

## 1. Generation — `spatialcpav25_gen/infer/generate.py`

> ⚠️ **Open risk R3, raised at T04: reconstruction is far worse near the stack's ends.** Measured on
> the gate fixture, per-section R² of the T04 probe was **0.2912** at the first section and
> **0.3642** at the last, against an interior mean of **0.4474** — a 20–35 % deficit at the boundary.
> The mechanism is one-sided evidence: a cell at z = 0 has training sections and retrieval donors
> above it only. This is **real-volume geometry, not a fixture artefact** — every serial-section
> dataset has two ends — and it lands squarely here, because `generate_section` is routinely asked
> for planes at or beyond the outermost sections, where the model is extrapolating rather than
> interpolating. Treat generation outside the interior as a distinct, worse regime: report it
> separately, and consider surfacing an explicit boundary flag on the emitted AnnData (`uns`) rather
> than letting a caller assume uniform quality across the stack. The uncertainty gate (§below) is the
> natural place for it — the latent variance it already estimates should be elevated there, and if it
> is *not*, that is itself a finding.

```python
def generate_section(model, plane: Plane, vol: TrainingVolume, cfg: Config, seed: int) -> AnnData
```

```
1. Query the field on the plane's slab.
2. Layout head -> N, positions, types                      (T05)
3. h0 = GRF(positions)   -- continuous 3D field            (T03)
4. Retrieval: K real neighbours, excluding any section at the target z
5. Flow ODE 0->1 -> h                                      (T06)
6. ZINB decode + SAMPLE counts                             (T06)
7. Uncertainty-gated anchoring (below)
8. Emit AnnData: X = counts (raw), obsm['spatial'] = (x, y) in the plane's basis,
   obsm['spatial_3d'] = (x,y,z), obs[cell_type, region], uns[plane, seed, config hash]
```

**Uncertainty-gated anchoring.** Run `M = 8` flow samples with different GRF seeds; per-cell latent
variance `v_i` estimates uncertainty. Where `v_i` is low, blend toward a retrieval-anchored real
profile (Bernoulli per-gene mixing, preserving count-ness); where high, trust the generative sample.
The blend weight is a **learned** function `w(v_i)` fitted on LOSO reconstructions — a 1-D isotonic
regression from `v` to the mixing weight that minimised reconstruction error. This replaces the
hand-tuned gap heuristics of the previous version entirely; there is no `gap_scale`, no `alpha_tol`,
no `edit_weight`.

Also provide:
```python
def generate_stack(model, z_values, vol, cfg, seed) -> AnnData     # dense virtual stack
def generate_oblique(model, normal, n_sections, vol, cfg, seed) -> AnnData
def generate_curved(model, surface: Surface, vol, cfg, seed) -> AnnData
```
All share one GRF realisation per call, so the produced sections are **mutually coherent** — slices
of one object. Expose `grf_seed` explicitly and document that reusing it across calls is what
guarantees coherence.

### `retrieval_z_window` must scale with the gap, not be fixed (measured at T06)

`Config.retrieval_z_window = 3.0` is in units of the median section spacing, which is the right
*unit* and the wrong *constant* once the holdout is wide. Measured at T06 on the `consecutive-3`
holdout: the first training section is 200 µm from the next admissible one, so after the own-section
exclusion (`retrieval_exclude_source_section`, load-bearing for GATE 2) its cells have **no admissible
donor at all** inside 3 × 50 µm, and `EmptyCandidatePoolWarning` fires on **100–110 of every 512
cells**. Their neighbour sets are fully masked and the attention returns its bias — i.e. the retrieval
branch is silently absent for a fifth of the batch, in exactly the wide-gap regime the branch exists
for.

The warning did its job; the constant is the defect. T09 must make the window a function of the
generation or holdout geometry rather than a fixed multiple:

* the window has to admit at least the nearest admissible section *after* every exclusion, so its
  floor is `(gap to the nearest admissible section) / median_spacing`, rounded up, not 3;
* it is a calibration-time quantity like `ell`, and it belongs beside the length-scale calibrator in
  §2 — leakage-free by the same construction, fitted on flanking training sections only;
* `EmptyCandidatePoolWarning` firing on more than a negligible fraction of queries must be a
  **failure** of the generation path, not a warning, once the window is derived: with a derived window
  an empty pool means the geometry is genuinely impossible and the caller needs to know.

`Config.retrieval_z_window` stays as the fallback and the ablation handle. T10 §4 carries the matching
requirement for the ablation table: **A5 (`retrieval_w_z = 0`) must not be run at a fixed window**,
or the ablation measures the window instead of the z term — the same trap T04's G2.3 fell into and
recorded.

## 2. Calibration — `spatialcpav25_gen/infer/calibrate.py`, all leakage-free

```python
def calibrate_lengthscale(model, vol: TrainingVolume, cfg) -> LengthscaleCalibration
```
Bisection on `ell_xy` so the **generated** section's mean Moran's I matches the mean Moran's I of the
**flanking training sections**. 6–8 iterations. `ell_z` from between-section correlation decay
(T03's `fit_lengthscale_from_sections`).

Ground truth is never consulted. Assert in a test that the calibrator's signature cannot accept
held-out sections.

**The objective is unimodal, not monotone (T03/GATE 1, measured).** Mean `I_gen` rises with `ell`,
turns over, and falls: Moran's I of expression is structured variance over total variance, and a
stationary field loses within-window variance once its correlation length approaches the section.
The maximiser was measured at 0.086 of the in-plane extent on the 3000 µm gate fixture and 0.112 on
the 1000 µm one — and at 2.5× and 0.8× the *fitted* `ell` respectively, because the variogram fit is
itself window-biased at a narrow field of view. A bisection that ignores this walks into a boundary
and returns it as if it were an answer. The calibrator therefore:

1. **Caps the bracket** at
   `ell_max = min(cfg.calibration_ell_max_extent_frac * extent, cfg.calibration_ell_max_fitted_multiple * ell_fitted)`
   (0.2 × extent and 2 × fitted by default) and starts from `ell_min = cfg.variogram_ell_min_factor *
   median_nn_dist`. Neither cap is a guarantee — 0.2 × extent is about twice the measured maximiser —
   so they bound the search, they do not define it.
2. **Locates the maximum** on the bracket before bisecting: evaluate `I_gen` on a
   `cfg.bisection_grid_size`-point log grid, take the maximiser, and bisect only on
   `[ell_min, ell_argmax]`, which is monotone by construction.
3. **Returns a status.** `LengthscaleCalibration` carries `ell`, `status ∈ {"converged",
   "target_unreachable", "boundary"}`, the achieved `I_gen` and the target. When the target mean
   Moran's I exceeds `max(I_gen)` over the bracket there is **no root**: return the maximiser with
   `status="target_unreachable"` and log a warning naming both numbers. Never return a bracket
   endpoint as though bisection had converged (Convention 6 — no silent fallbacks).

Acceptance tests:
- `test_calibrator_recovers_a_reachable_target` — plant a target inside the achievable range;
  `status == "converged"` and the achieved `I_gen` is within tolerance.
- `test_calibrator_reports_unreachable_target` — plant a target above `max(I_gen)`; the calibrator
  returns `status == "target_unreachable"`, the returned `ell` **is** the grid maximiser, and it does
  not sit at either bracket endpoint. This is the branch T03 measured and the one a naive bisection
  gets wrong.
- `test_calibrator_bracket_respects_the_caps` — on a narrow-field-of-view volume the bracket's upper
  end is the extent cap, not the fitted-multiple cap.

**One global `ell`, and per-module agreement as a diagnostic only** (settled, SPEC_QUESTIONS A2).
The original spec calibrated `ell` per gene-module (Leiden clusters of gene embedding space, ~10
modules), which is not implementable as written: `ell` parameterises the **latent** field (`d_h = 64`
channels queried at cell positions) and gene modules only exist downstream of the decoder, so "the
`ell` for module *m*" would require the field to know which latent channels that module reads from —
not a property the decoder is constrained to have. Calibrate **one global**
`ell = (ell_x, ell_y, ell_z)` — which is also what GATE 1's monotonicity and unimodality criteria are
defined on — and report per-module Moran's I agreement as a **diagnostic table** in
`reports/config_selection_*.md`, not as a target.

*Escalation, if the diagnostic is poor.* The cheap version is to partition the `d_h` latent channels
into groups with their own `ell` and add a loss tying gene modules to channel groups. `T03`'s
`with_lengthscale` already makes the field side free (a grouped field is a channel-wise
concatenation of rescaled copies of one realisation, no redraw and no new state), so the cost is all
in the tying loss. **That is a design change and must be decided explicitly, with the diagnostic
table as the evidence — not improvised inside the calibration loop.**

```python
def calibrate_detection(model, vol, cfg) -> DetectionCalibration
```
Per-gene affine correction on the `pi` logit so generated detection rates match the flanking
sections'. Fit on LOSO reconstructions; apply at generation.

**Calibrate the mean–variance relation too, not just `pi`** (settled; `design/v23_design.md` §3.5
asks for both and `specs/` had only `pi` — SPEC_QUESTIONS D-table). Add a per-gene correction on
`log theta` fitted the same way, so the generated mean–variance curve matches the flanking sections'
rather than only their detection rates. The two are not substitutes: `pi` moves the zeros and
`theta` moves the spread of the non-zeros, and T06's `test_mean_variance_relation` (log-log slope
within 15%) is the property this protects at inference. `DetectionCalibration` carries both
corrections and records which sections it was fitted on.

```python
def calibrate_anchor_weight(model, vol, cfg) -> IsotonicRegressor
```
As described in §1.

## 3. Automatic per-dataset configuration — `spatialcpav25_gen/train/select.py`

```python
def select_config(vol: TrainingVolume, base_cfg: Config) -> Config
```

Coordinate-descent over a small gate grid, scored by the six target metrics on **LOSO
reconstructions of training sections**:

| Gate | Options |
|---|---|
| `layout_mode` | field / hybrid / resample |
| `prior_mode` | correlated / iid |
| `expr_mode` | zinb-flow / cross-mix / auto-blend |
| `text_emb` | medcpt+residual / lookup-only |

- Fit a **reduced-epoch** model (25% of `cfg.epochs`) per candidate.
- Score = median rank across the six metrics over LOSO folds.
- Coordinate descent, 2 passes → ~10 fits, not 36.
- Persist the chosen config and the full score table to `reports/config_selection_{dataset}.md`.

**This is the "no regression" guarantee:** `layout_mode=resample` + `expr_mode=cross-mix` reproduces
the previous version's behaviour, so if the new machinery does not help on a dataset it is switched
off automatically. Add `test_selector_can_recover_v20_config` asserting that config is reachable and
selected when the new components are artificially degraded.

## Acceptance tests

- `test_generate_shapes_and_dtypes` — AnnData valid; X integer-valued; no NaN; `uns` carries plane,
  seed, config hash.
- `test_generate_deterministic` — same seed → identical output.
- `test_stack_coherence` — two sections from `generate_stack` at nearby z have expression correlation
  decaying smoothly with |Δz|, with no discontinuity at training-section z values. A spike at real-
  section z means the model is copying rather than interpolating.
- `test_oblique_intersection_agreement` — two oblique sections intersecting inside the tissue agree
  along their intersection: cell-type concordance > 0.8, expression correlation > 0.85. **This is
  the headline capability experiment (E5); make it a first-class test.**
- `test_calibration_no_leakage` — held-out sections cannot be passed (type error), and calibration
  results are identical whether or not held-out data exists in the parent object.
- `test_calibration_converges` — bisection reaches |I_gen − I_flank| < 0.02 within 8 iterations on
  the fixture.
- `test_anchor_weight_monotone` — the fitted isotonic map is non-increasing in `v`.
- `test_selector_runs_and_persists` — selection completes on the fixture and writes the report.
- `test_selector_can_recover_v20_config` — described above.

## Definition of done

On the fixture, LOSO reconstruction beats both `resample`-mode and the independent-donor baseline on
≥ 4 of the 6 target metrics; `reports/config_selection_synthetic.md` exists. `PROGRESS.md` records
the calibrated `ell` values and the selected config.

## Do NOT

- Do not consult held-out sections anywhere in calibration or selection.
- Do not expose gap/alpha/edit flags to users — the whole point is that these are inferred.
- Do not draw a fresh GRF per section inside a stack (destroys coherence).
