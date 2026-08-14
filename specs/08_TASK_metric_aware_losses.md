# T08 — Metric-aware losses via internal leave-one-section-out

**Goal.** Make the evaluation criteria into differentiable training signals, computed **only on
training sections** through an internal LOSO loop. This is the concrete reason to expect wins on the
six target metrics: no competing method optimises any of them.

**Files:** `spatialcpav25_gen/losses/metric_aware.py`, `spatialcpav25_gen/train/loso.py`, `tests/test_metric_aware.py`

**Dependencies:** T01–T07.

---

## ⚠️ Leakage discipline

These losses are computed by hiding one **training** section and reconstructing it from the others.
The held-out evaluation sections are never touched.

Enforce structurally, not by convention:
- `metric_aware.py` functions take a `TrainingVolume` newtype that `split_holdout` produces only for
  the training portion. The held-out sections live in a different type (`HeldOutSections`) that has
  no method accepted by any loss function. Make the mistake a **type error**, not a code-review
  question.
- Add `test_metric_aware_rejects_heldout` asserting a `TypeError` when held-out data is passed.

This is also a paper-integrity point: the approach must be disclosed prominently in the methods, and
T10 must report unoptimised metrics to show the wins are not narrow overfitting to the scoreboard.

---

## 1. Differentiable spatial autocorrelation

```python
def morans_i(x: Tensor, W: SparseTensor) -> Tensor        # (G,) given (N, G) and fixed kNN graph
def gearys_c(x: Tensor, W: SparseTensor) -> Tensor
```

```
I = (N / S0) * ( sum_ij w_ij (x_i - xbar)(x_j - xbar) ) / sum_i (x_i - xbar)^2
C = ((N-1) / (2 S0)) * ( sum_ij w_ij (x_i - x_j)^2 ) / sum_i (x_i - xbar)^2
```

Both are differentiable w.r.t. `x` for a **fixed** graph `W`. Build `W` once per section from the
generated positions (detached — do not differentiate through graph construction; it is not
meaningfully differentiable and the attempt introduces instability).

```python
def loss_autocorr(x_gen, x_real, W_gen, W_real, cfg) -> Tensor
```
Huber loss between the per-gene `I` vectors, plus the same for `C`. Weight genes by their real
variance so uninformative genes do not dominate.

⚠️ Use `x` on the **same scale** for both (log1p-normalised is fine here — Moran's I is
scale-invariant per gene but not invariant to the nonlinearity, so be consistent and document it).

## 2. Soft spatial profiles

```python
def soft_depth_profile(x, coords, axis, n_bins, sigma) -> Tensor      # (n_bins, G)
def soft_field_profile(x, coords, grid_shape, sigma) -> Tensor        # (H, W, G)
```

Hard binning is non-differentiable; use Gaussian kernel weights
`w_ib = exp(-(proj_i - center_b)^2 / 2 sigma^2)`, row-normalised. `sigma` = 0.75 × bin width.

`axis`: the section's principal tissue axis. Compute it **once per dataset** from the training
sections (first PC of cell coordinates, or a user-supplied anatomical axis) and store it on the
`Volume`. Do not recompute per generated section — a drifting axis makes the loss non-stationary
and the metric incomparable across epochs.

```python
def loss_profile(x_gen, coords_gen, x_real, coords_real, vol, cfg) -> Tensor
```
- Depth profile agreement over marker genes (top-k spatially variable genes from training).
- 2-D field profile agreement (coarse grid, e.g. 24×24).
- **Per-type spatial histogram** agreement — this is where cell-type localization is optimised.

## 3. Distribution matching

```python
def loss_distribution(x_gen, x_real, cfg) -> Tensor
```
Entropic Sinkhorn divergence (via `geomloss`, blur = median NN distance in PC space) between
generated and real cells in a **fixed** PCA basis fitted once on training data. Fallback: multi-
bandwidth RBF MMD if `geomloss` is unavailable.

This targets embedding mixing. Note honestly in `PROGRESS.md`: mixing is the metric where the
competing method is genuinely strong, because its per-gene chimerism maximises cloud overlap almost
by construction. A tie here is an acceptable outcome; a loss is a signal to investigate.

## 4. The LOSO training loop — `spatialcpav25_gen/train/loso.py`

```python
class LOSOScheduler:
    """Each epoch, hide one training section and reconstruct it for metric-aware losses."""
    def epoch_task(self, epoch: int) -> tuple[TrainingVolume, Section]
```

- Round-robin over training sections (deterministic, seeded).
- The hidden section is excluded from the retrieval index and from the field's supervision for that
  epoch — otherwise the model reconstructs it by memorisation and the loss is vacuous. Assert this
  in `test_loso_excludes_from_retrieval`.
- Reconstruct at the hidden section's true plane and thickness, then compute §1–§3.
- Cost control: run the metric-aware block every `k` steps (default 4), on a subsample of ≤ 4000
  cells.

## Acceptance tests

- `test_morans_matches_reference` — against a NumPy/`esda` reference on random data, 1e-5.
- `test_gearys_matches_reference` — likewise.
- `test_morans_differentiable` — gradient w.r.t. `x` is finite and non-zero.
- `test_soft_profile_approaches_hard` — as `sigma -> 0`, the soft profile converges to hard binning
  (relative error < 2% at sigma = 0.1 × bin width).
- `test_profile_axis_stable` — the principal axis is identical across epochs.
- `test_loso_excludes_from_retrieval` — structural assertion, described above.
- `test_metric_aware_rejects_heldout` — `TypeError`, described above.
- `test_metric_losses_improve_metrics` — **the point of the whole task.** Train 1000 steps with the
  metric-aware terms on vs. off; with them on, held-in Moran's agreement, marker-depth r, and
  localization must each be better. If any is not, report it — a loss that does not improve its own
  metric is either mis-specified or mis-weighted.

## Definition of done

The on/off comparison above shows improvement on all three families. `PROGRESS.md` records the
before/after table — it becomes ablation A2 in the paper.

## Do NOT

- Do not differentiate through kNN graph construction.
- Do not let held-out sections reach these functions (make it a type error).
- Do not recompute the principal tissue axis per epoch.
- Do not weight these losses above reconstruction.
