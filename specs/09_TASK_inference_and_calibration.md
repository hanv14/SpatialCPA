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

## 2. Calibration — `spatialcpav25_gen/infer/calibrate.py`, all leakage-free

```python
def calibrate_lengthscale(model, vol: TrainingVolume, cfg) -> tuple[float, float, float]
```
Bisection on `ell_xy` so the **generated** section's mean Moran's I matches the mean Moran's I of the
**flanking training sections**. 6–8 iterations. Per gene-module (Leiden clusters of gene embedding
space, ~10 modules), since autocorrelation length is gene-dependent. `ell_z` from between-section
correlation decay (T03's `fit_lengthscale_from_sections`).

Ground truth is never consulted. Assert in a test that the calibrator's signature cannot accept
held-out sections.

T03/GATE 1 established that mean `I_gen` is monotone in `ell`; if bisection fails to bracket, fall
back to a grid search over 12 values and log a warning.

```python
def calibrate_detection(model, vol, cfg) -> DetectionCalibration
```
Per-gene affine correction on the `pi` logit so generated detection rates match the flanking
sections'. Fit on LOSO reconstructions; apply at generation.

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
