# Open questions and suspected errors in `specs/`

Raised after reading all of `specs/`, `design/v23_design.md`, `design/v23_sectioning_equivariance.md`,
and cross-checking against `reference/learn_spatialcpav20.py` and `benchmark-pbya-v3/`.
Nothing here is a reason to delay T01 except the items in §A, which change interfaces.

Status key: **OPEN** (needs a decision), **PROPOSED** (I will do the stated thing unless told
otherwise), **INFO** (recorded, no action).

---

## A. Contradictions between task files — resolve before the affected task

### A1. `Volume` vs `TrainingVolume` / `HeldOutSections` — **RESOLVED in T01**
T01 specifies `split_holdout(vol, mode, fold) -> tuple[Volume, list[Section]]` and
`loso_folds(vol) -> Iterator[tuple[Volume, Section]]`. T08 and T09 require the training portion to
be a distinct type `TrainingVolume`, the holdout to be `HeldOutSections`, and passing the wrong one
to be a **`TypeError`** (`test_metric_aware_rejects_heldout`). A `typing.NewType` gives a mypy error,
not a runtime `TypeError`.

*Resolution (T01):* done as proposed. `data/schema.py` defines `TrainingVolume(Volume)` and
`HeldOutSections` (a `Sequence[Section]`, deliberately *not* a `Volume` and carrying no
`median_spacing`); `split_holdout` returns `tuple[TrainingVolume, HeldOutSections]` and `loso_folds`
raises `TypeError` for anything else. T08/T09 should runtime-check the same way at their
entrypoints.

### A2. Per-gene-module length-scale calibration is not implementable as written — **RESOLVED (decided 2026-08-15)**
T09 §2 calibrates `ell` "per gene-module (Leiden clusters of gene embedding space, ~10 modules)".
But `ell` parameterises the **latent** GRF (`d_h = 64` channels queried at cell positions), and gene
modules only exist downstream of the decoder. One `ell` per gene module cannot be expressed against
a single latent field: changing `ell` for module *m* would require the field to know which latent
channels that module reads from, which is not a property the decoder is constrained to have.

*Proposal:* calibrate one global `ell = (ℓx, ℓy, ℓz)` (this is also what GATE 1's monotonicity
criterion is defined on), and report per-module Moran's I agreement as a **diagnostic**. If
per-module control turns out to be needed, the cheap version is to partition the `d_h` latent
channels into groups with their own `ell` and add a loss tying gene modules to channel groups —
that is a design change and should be decided explicitly, not improvised at T09.

*Decision:* **one global `ell`**; per-module Moran's I agreement is reported as a **diagnostic only**,
never as a target. If the diagnostic is poor at T09, the escalation is per-channel-group `ell` — and
it is to be **decided explicitly**, with the diagnostic table as evidence, not improvised inside the
calibration loop. Written into `specs/09` §2.

*Status after T03:* the per-channel-group `ell` was **not** added to `GaussianRandomField`. The
spec's interface takes one `ell`, nothing in `specs/` consumes a grouped one, and speculative API is
what CLAUDE.md tells us not to write. The escalation stays cheap: `with_lengthscale` rebuilds a
rescaled field from the same draws without redrawing, so a per-group field is a channel-wise
concatenation of those, and only the loss tying modules to channel groups is new work.

### A3. T10's metric provenance is wrong — **RESOLVED (decided 2026-08-15, strengthened)**
T10 says "port the six target metrics from `reference/learn_spatialcpav20.py`, fixing two bugs".
Checked:

- v20 defines `morans_pearson`, `gearys_pearson`, `embedding_mixing_pca`, `marker_depth_r` — it does
  **not** define `marker_field_r` or `celltype_localization`. Two of the six are not there to port.
- The two bugs are real *in v20*: `_corr` at `learn_spatialcpav20.py:1876` is `np.corrcoef` behind
  the name `gene_*_spearman`, and `rank_normalize` at `:1810` uses `argsort` (distinct ranks for
  tied zeros).
- But the **actual scoreboard** is `benchmark-pbya-v3/src/bench3/evaluate_paper.py`, which produces
  the `paper_*` metrics, and it already uses `scipy.stats.spearmanr` and
  `rankdata(method="average")` (via `benchmark-pbya-v2/.../evaluate_generation.py:69`). Both bugs
  are already fixed there.

So porting from v20 would (a) be missing two metrics, (b) re-fix bugs that the published pipeline
does not have, and (c) risk producing numbers that are **not comparable** to the existing v20/v22
results, which came from bench3.

*Decision — stronger than the proposal: do not port at all.* **Vendor or import
`bench3/evaluate_paper.py` verbatim**, pin it with a content hash
(`7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992`, 764 lines, checked at import),
and assert **bit-identical** output on fixed inputs (`test_metrics_match_bench3_bitwise`, `==` not
`allclose`). The existing v20/v22 numbers came out of bench3, and a reimplementation that merely
agrees closely is **not comparable** — the paper's claim is a difference between methods measured on
one instrument. `eval/metrics.py` becomes a thin adapter that owns no metric arithmetic. T10's bug
note survives as a **footnote about v20's internal tuning signal**, not as a fix applied to the
benchmark (bench3 already uses `spearmanr` and `rankdata(method="average")`). Written into
`specs/10` §1.

### A4. `Config` is missing fields that later tasks depend on — **RESOLVED in T01**
Convention 1 forbids constants outside `Config`, but the task files write several inline. Not in the
T01 field list and needed: `field_dim` (`d_f`, used everywhere from T04 on), `ctx_dim` (`d_ctx`,
retrieval output), retrieval attention head count, `expr_pca_dim` (neighbour tokens, Sinkhorn basis),
`metric_knn_k` (the k in Moran's/Geary's graphs — bench3 uses 10), `potts_knn_k` (T05 hardcodes 8),
`layout_max_proposal_factor` (T05's `20*N`) and `layout_envelope_slack` (its `1.1`),
`swd_polish_steps` (T05 hybrid's 200), profile `n_bins` / `field_grid` (T08's 24×24) and
`profile_sigma_frac` (0.75), `loso_every_k_steps` (4) and `loso_max_cells` (4000),
`n_uncertainty_samples` (T09's M=8), `sefl_min_stratum_cells` (T07's 20), `holdout_consecutive_k`
(T01 calls it "configurable" but gives no field), `bisection_max_iter` / `bisection_grid_size` (T09).

*Resolution (T01):* all added and documented; `Config` is 94 fields rather than the ~60 printed in
the spec. Four have no value fixed anywhere in `specs/` and are marked *provisional* in their
docstrings, to be set by the task that first uses them: `field_dim=128`, `retrieval_ctx_dim=64`,
`retrieval_n_heads=4`, `expr_pca_dim=32` (the last taken from GATE 2's "top-32 expression PCs").
`holdout_consecutive_k` is read by `split_holdout` through a new `cfg` keyword, since the signature
in the spec has nowhere else to get the run length from.

### A5. `Config.validate()` cannot check the example it is given — **RESOLVED in T01**
T01 asks `validate()` to raise on "`fourier_bands_z > 4` with fewer than 8 training sections", but
`Config` is standalone and has no volume. *Resolution (T01):* `Config.validate()` covers set
membership, positivity, fractions and cross-field relations; the data-dependent checks live in
`data/schema.py::validate_config_against_volume(cfg, vol)` (a function rather than a method, so
`config.py` need not import the schema), called by `load_volume`.

### A6. Nothing specifies building the v20 cross-mix — **RESOLVED (decided 2026-08-15)**
`expr_mode ∈ {zinb-flow, cross-mix, auto-blend}` is a `Config` gate (T01), the no-regression
guarantee depends on `cross-mix` (T09 §3, `test_selector_can_recover_v20_config`), and T09's
uncertainty-gated anchoring blends "via the v20 Bernoulli cross-mix" (design §5). No task file
specifies implementing it, and the coverage matrix mentions it only as reference material.
*Decision:* as proposed — **implement it in T06** alongside the decoder (it shares the
count-preserving output path), ~40 lines ported from v20, with its behaviour **pinned by a test**
(`test_cross_mix_matches_v20`: bitwise on fixed inputs and a fixed seed, or the donor-frequency
distribution if v20's RNG consumption order cannot be reproduced under Convention 3 — stated in the
test's docstring either way). Written into `specs/06` §4b and the coverage matrix.

### A7. GATE 1's G1.3c was false as literally specified — **RESOLVED (spec amended, gate re-run)**
Raised by T03: `I_gen` is not monotone as `ell` sweeps 0.25×–4×. It rises to a maximum and falls.

The prior is not the cause — Moran's I of the *field itself* is monotone in `ell` over the same sweep
(0.38 → 0.99) — the observable is: Moran's I of expression is structured variance over total
variance, and a stationary unit-variance field loses within-window variance once its correlation
length approaches the window (within-section sd 1.00 → 0.87 across the sweep at a 3000 µm field of
view, 0.98 → 0.65 at 1000 µm).

*Resolution, after the verdict was accepted as a conditional pass:*
1. **The gate fixture moved to 3000 µm** at constant cell density (`GATE_EXTENT_UM`). Real sections
   are 5–10 mm wide with correlation lengths of 50–200 µm (`ell`/FOV ≈ 0.02); at 1000 µm the top of
   the sweep sat at 55 % of the extent, which real data never does. The 1000 µm measurement is kept
   as diagnostic D1 in `reports/gate1.md`.
2. **G1.3c restated** over the sweep points inside the calibration bracket,
   `ell ≤ min(calibration_ell_max_extent_frac × extent, calibration_ell_max_fitted_multiple ×
   fitted ell)` — the range T09 actually bisects on, which is what the criterion was always for.
3. **G1.3g added**: `I_gen(ell)` is unimodal (violation 0.000 against a 2-SE band of 0.0069) and its
   maximiser is at or above the fitted `ell` (2.52×), so a bracket below the maximum is well-posed.
   This is where the calibration content now lives; G1.3c's caps are a-priori guards.
4. **T09's calibrator amended** (`specs/09` §2): cap the bracket, locate the maximum on a log grid,
   bisect only below it, and return `status ∈ {converged, target_unreachable, boundary}` with a
   test for the unreachable branch.

Measured caveat, recorded rather than papered over: the maximiser sits at 0.086 × extent (3000 µm)
and 0.112 × extent (1000 µm), so the default `calibration_ell_max_extent_frac = 0.2` is about twice
the maximiser and does **not** bind protectively; on the gate fixture it is the `2 × fitted ell` cap
that keeps the bracket below the peak, and on a narrow-FOV dataset neither cap would. T09's maximum
detection is the only real protection. The constant was left at the specified 0.2 rather than tuned
down, because the measurement that would justify a new value is the same one G1.3c checks.

---

## B. Acceptance tests that will fail for reasons unrelated to the model

### B1. Bitwise equality across two plane pathways (G1.2, and `test_noise_identical_along_intersection` in T07) — **RESOLVED in T03**
`torch.equal` on the GRF sampled through plane 1's and plane 2's coordinate constructions only holds
if both produce **bit-identical** `xyz`. `origin + u*e1 + v*e2` with different orthonormal bases
rounds differently in float32, so this can fail while the field is perfectly correct — exactly the
misdiagnosis the spec warns about elsewhere.
*Resolution (T03):* done as proposed, and the worry turned out not to bite. `tests/fixtures/planes.py`
holds one canonical `(u, v) → (x, y, z)` map (T07 supersedes it with `infer/planes.py`); the two
float64 reconstructions agree to **2.8e-14 µm**, which rounds to **identical float32 coordinates**,
so the field values are **bit-identical** and the spec's requirement is met outright. G1.2 asserts
all three: the float64 gap, zero differing float32 coordinates, and `torch.equal` on the field.
Purity is asserted separately. One measured caveat, in the module docstring: a **one-row** query is
dispatched by torch to a matrix-vector kernel and matches the same point inside a batch to ~2e-6
rather than exactly; every batch size from 2 up, and every chunk size, is bit-identical.

### B2. G1.1's monotone-in-M requirement is stochastic — **RESOLVED in T03**
"Error decreases monotonically as M goes 1024 → 2048 → 4096" is a single-draw statement about a
random estimator. *Proposal:* average the covariance MAE over ≥ 5 seeds per M and assert the trend
on the means (plus `err(4096) < err(1024)` outright).

### B3. `test_grf_channels_independent` tolerance is at the noise floor — **RESOLVED in T03**
Finite-M cross-channel correlation is O(1/√M) ≈ 0.016 at M = 4096, before point-sampling noise; the
spec's threshold is 0.02. *Resolution (T03):* done as proposed. The columns are orthogonalised by a sign-canonicalised QR and
scaled to `‖A_c‖ = √M`, which makes the space-averaged cross-channel covariance and the marginal
variance **exact** rather than approximate; the 0.02 threshold stands and measures 0.0028 empirically
(and < 1e-9 algebraically). `E[A_c A_cᵀ] = I` is unchanged, so the covariance function is not
affected — G1.1 confirms.

### B4. `test_relative_position_only` (T04) is false for the full model — **RESOLVED in T04**
"Translating the whole volume by a constant leaves outputs unchanged" cannot hold end-to-end: the
GRF realisation is a function of absolute position (`ξ(p + t) ≠ ξ(p)` by construction — that is the
point of T03), and the triplane is bbox-relative only if the bbox is translated with the data.
*Decision:* as proposed. `test_relative_position_only` is scoped to the retrieval branch's neighbour
encoding — it translates the volume by `(1234.5, −678.25, 900)` µm, rebuilds the index, and asserts
that the retrieved neighbour **set**, the donor weights and the encoded **tokens** are unchanged.
That is where the leakage the spec guards against would appear, and where it would do the damage.
Tolerance 1e-3, not 0: `Section.coords` is float32 (Convention 4), so a 1234.5 µm translation is not
exactly representable and in-plane distances move by ~1e-4 µm; a token carrying an absolute
coordinate would differ by ~1e3.

### B5. Rotation equivariance vs data-frame Fourier encoding (T04) — **RESOLVED in T04 (the proposal was not implementable; see below)**
The spec asks for both "encoding a fixed cell must be invariant to the augmentation rotation" and "a
full forward pass is equivariant: rotate inputs, inverse-rotate outputs, get the same result".

**The proposed contract `F(R·p | volume rotated by R) == F(p | volume unrotated)` cannot be
implemented without making the augmentation an exact no-op**, and that is not a subtlety about
frames — it is arithmetic. The triplane feature planes are fixed arrays indexed by fixed axes. Such
a lookup is invariant to a continuous rotation *only* if it undoes the rotation before indexing; and
a triplane that undoes the rotation is trained identically whether the augmentation is on or off.
The design's fix (a) — "rotate the entire volume … forces the field to be orientation-agnostic",
`design/v23_sectioning_equivariance.md` §2.1 — would then do nothing, GATE 2 would be measuring fix
(b), the multi-orientation ensemble, alone, and it would appear to be testing both. That is the same
class of failure C1(a) identifies for own-section retrieval: a gate that passes while hiding what it
exists to detect.

*Decision (T04):* implement the design's semantics and state the contract per channel, in
`model/field.py`'s docstring and in a table. `RotationContext` carries one rotation `R` and the four
channels behave differently **on purpose**:

| Channel | Under the augmentation |
|---|---|
| `fourier_encode` | **invariant** — evaluated in the data frame, before `R` |
| GRF query points | **invariant** — the context maps them *back* to the data frame, so the realisation stays attached to the tissue and T03's exact intersection consistency survives across steps |
| retrieval neighbourhoods | **invariant** — identity, weights and the tokens' relative positions are data-frame quantities |
| triplane lookup | **not invariant, deliberately** — this *is* the augmentation |

The spec's two tests then assert the two halves: `test_rotation_equivariance` checks the three
invariant channels agree to 1e-3 relative under three random rotations, and
`test_rotation_augmentation_is_not_inert` is a **negative control** asserting the triplane lookup
moves by > 10% relative — so nobody can later "fix" the equivariance test by undoing the rotation.
`RotationContext` additionally refuses to exit with a required channel un-transformed
(`RotationError`), which is the "impossible to rotate one and forget another" the spec asks for.

*Consequence to carry forward:* GATE 2's oblique parity therefore rests on **both** fixes, which is
what the design intends, and the gate is strictly harder than under the proposed contract.

### B6. Hard-core radius fights the pair-correlation test (T05) — **RESOLVED (decided 2026-08-15)**
`r0` = 5th percentile of real nearest-neighbour distances, and `test_hardcore_respected` forbids any
generated pair closer than `r0`. By construction 5% of *real* pairs are closer than `r0`, so the
generated layout is strictly more regular than the tissue it is imitating, which pushes against
`test_pcf_matches_real` (max |Δg(r)| < 0.15 from `r0` up).
*Decision:* as proposed — `r0` at the **1st percentile**, the fitted `gamma`/`R` carrying the soft
repulsion, the **5th percentile kept as a `Config`-selectable alternative**, and **which one was used
recorded** in `reports/benchmark.md` and `PROGRESS.md` (it changes a published point-process number).
`Config.repulsion_r0_percentile` already defaults to 1.0 from T01; written into `specs/05`.

### B7. The layout sampler is a conditional Strauss, not a Strauss process (T05) — **PROPOSED**
Step 1 draws `N ~ Poisson(N_expected)` and step 3 thins; conditioning on `N` and then thinning is
not the same object as a Strauss process (which produces fewer points than its Poisson envelope).
The `20*N`-proposals escape hatch then silently breaks `test_expected_count_matches`.
*Proposal:* implement and document it as a **conditional-on-N** Strauss sampler (which is what makes
the count-from-intensity claim work), make the proposal cap a `Config` field, and raise rather than
warn under test.

### B8. The Matérn RFF parametrisation will not match `scipy`'s `Matern(length_scale=ell)` — **RESOLVED in T03: the worry was unfounded**
`omega = (z/√g)/ell` with `g ~ Gamma(ν, 1/ν)` yields a Matérn *shape* but with an effective
length-scale off by a √(2ν)-type constant relative to `scipy`'s parametrisation, so G1.1's
MAE < 0.03 could fail purely on convention. The spec anticipates this ("verify this empirically
rather than trusting the derivation"). *Resolution (T03):* no constant is needed. The Matérn spectral density in R³ *is* a multivariate
Student-t with 2ν degrees of freedom, and `z/√g` with `g ~ Gamma(ν, 1/ν)` is exactly that, so
`omega = (z/√g)/ell` realises `k(r) = 2^(1−ν)/Γ(ν) (√(2ν) r)^ν K_ν(√(2ν) r)` — the same
parametrisation `sklearn.gaussian_process.kernels.Matern(length_scale=ell, nu=nu)` uses. Verified
rather than trusted: `matern_correlation` agrees with `sklearn` to < 1e-10 at ν = 0.5/1.5/2.5/4.0,
and the realised field covariance sits 0.0121 from the analytic anisotropic kernel at M = 4096. The
derivation is in the `model/noise.py` docstring.

### B9. G1.4's throughput target needs chunking to be possible at all — **RESOLVED in T03; criterion restated**
10⁶ points at M = 4096 is a 16 GB float32 feature matrix if materialised. *Resolution (T03):* chunked, and nothing had to be loosened — **3.4 s for 10⁶ points** at M = 4096,
d_h = 64 on a 4-core Xeon @ 2.10 GHz. The binding constraint turned out to be **cache**, not memory:
the default chunk was lowered 65536 → **1024** so the `(chunk, M)` block of cosines is 16 MB rather
than 1 GB, which is a 3× speed-up on its own.

*Amended:* the same code and query took 6.1 s on an Apple-silicon laptop, i.e. the criterion as
written failed there and passed here for reasons that have nothing to do with the field. G1.4b is now
**throughput recorded against reference hardware** (2.9 × 10⁵ points/s on the reference Xeon) rather
than an asserted wall clock, and the assertable part is the dimensionless **G1.4c**: 8× the points
must cost < 12× the time.

---

## C. Under-specified — I will pick the stated option unless told otherwise

### C1. GATE 2's evaluation set is undefined — **RESOLVED (decided 2026-08-15, with two additions)**
"Reconstruct held-in cells from planes at 0°…90°" — but real cells only exist on the sectioning
planes, so an oblique plane passes through very few of them, and R² at 90° would be computed on a
different (and much smaller) cell set than at 0°. That makes the ratio `R²(θ)/R²(0°)` partly an
artefact of sample size. *Decision:* the pooled `thickness/2` set as proposed, **plus two additions that the proposal missed**:

(a) **Leave-own-section-out retrieval.** For every evaluated cell, its own source section is excluded
from the retrieval candidate pool **at every angle**. Without it, a cell in the 90° strip retrieves
in-plane neighbours a few micrometres away inside its own section, the oblique plane becomes trivially
easy, and GATE 2 passes while hiding exactly the equivariance failure it exists to detect. Needs a
`Config` flag (`retrieval_exclude_source_section`, default `True`, added by T04), and two tests — one
that no returned neighbour shares the query cell's `section_id`, and one that turning the exclusion
*off* measurably raises R²(90°), so the exclusion cannot be quietly dropped later.

(b) **Equal `n` across angles**, by seeded subsampling to the smallest angle's count — reporting `n`
is not enough. R² is a variance-explained ratio; both its sampling error and the mix of tissue it
covers move with `n`, so an unsubsampled comparison partly measures sample size. Report the common
`n`, the pre-subsample `n` per angle, and the seed. Below a floor
(`gate2_min_cells_per_angle`), thicken the fixture's slabs — do not lower the floor, do not drop an
angle.

`reports/gate2.md` must state the whole contract. Written into `specs/04`.

### C2. `KL(ZINB₁ ‖ ZINB₂)` has no closed form (T07) — **RESOLVED (decided 2026-08-15)**
Neither NB–NB nor ZINB–ZINB KL is closed-form (both need an infinite sum over counts). *Decision: skip the surrogate entirely.* Match the decoder parameters directly —
`L2` on `log μ`, `log θ` and the `π` logit, with branch 2 detached. A Gaussian approximation in
`(log μ, log θ)` is a different divergence wearing a KL's name, and a fixed-sample MC estimate adds
variance to the most delicate loss in the system. Two branches that agree on every decoder parameter
agree on the distribution, which is what `L_cross` is actually asserting, and an L2 is scale-stable,
differentiable everywhere and honest in the methods section. Written into `specs/07` §2.

### C3. MedCPT pooling instruction contradicts itself (T02) — **PROPOSED**
The spec says "mean-pool the last hidden state", then says MedCPT-Query-Encoder is trained with CLS
pooling and to use whichever the checkpoint specifies. *Proposal:* use the CLS/first-token
representation (what the MedCPT query encoder is trained for), make it a `Config` field with
mean-pooling as the alternative, and write the justification in a comment as the spec asks. (T01
added `Config.text_pooling`, default `"cls"`.)

### C4. The fixture's ground-truth field has nowhere to live (T01) — **RESOLVED in T01**
The spec suggests `vol.uns["gt_field"]`, but the `Volume` dataclass has no `uns`. *Resolution (T01):* `make_synthetic_volume` returns
`(Volume, GroundTruthField)`; `Volume` has no `uns`. `GroundTruthField` exposes `latent(xyz)`,
`type_logits`, `expression_mu(latent, xyz, cell_type)` and `sample_counts`, so T03's G1.3 can
substitute its own noise for the latent and push it through the same generative map.

### C5. `Volume`'s derived fields need `field(init=False)` (T01) — **RESOLVED in T01**
`median_spacing`, `median_nn_dist`, `bbox` are listed as ordinary fields but described as computed in
`__post_init__`. *Resolution (T01):* they are `field(init=False)`, computed in `__post_init__`, so
they cannot be passed in and go stale; removing sections means constructing a new `Volume`, which is
what makes `split_holdout`'s recomputation of `median_spacing` automatic.

### C6. Which tests are `slow`? — **PROPOSED**
`test_distillation_reduces_error` (200 steps), `test_cfm_recovers_gaussian` (2000),
`test_cross_loss_decreases` (500), `test_no_collapse`, `test_metric_losses_improve_metrics`
(1000 × 2), and both gate reports cannot share a 3-minute CPU budget. *Proposal:* anything that runs
an optimiser loop or a gate report is `@pytest.mark.slow`; `make test` runs the rest and stays under
3 minutes, `make test-all` runs everything.

### C7. Naming drift to reconcile at T01 — **PROPOSED**
`w_z` (T04 §2, T10 A5) vs `retrieval_w_z` (T01 `Config`); `text_emb` (T09 §3, T10) vs
`text_emb_mode`; option strings `medcpt+residual` / `lookup-only` (design §6, T09) vs
`medcpt` / `lookup` (T01). Also T10 §6 prints the CLI as `spatialcpav25_gen fit` (underscore) in
three of four lines; the entrypoint is `spatialcpav25-gen`. I will use the `Config` spellings
everywhere and treat the others as prose.

### C8. Two Moran's I implementations will drift — **PROPOSED**
T08 needs a differentiable torch Moran's/Geary's; T10 needs the numpy scoreboard version. If they
diverge, the training surrogate stops tracking the metric and nobody notices. *Proposal:* one kernel
with thin wrappers plus a test asserting agreement to 1e-6 on shared inputs.

### C9. Two generation entrypoints — **PROPOSED**
`CTFFlow.generate(plane, cfg, seed)` (T06) and `generate_section(model, plane, vol, cfg, seed)`
(T09) do the same job with different signatures. *Proposal:* the method delegates to the function.

---

### C10. T08's principal tissue axis has nowhere to live — **RESOLVED (decided 2026-08-15; due at T08)**
T08 §2 says the principal tissue axis is computed once per dataset and "stored on the `Volume`", but
T01's `Volume` has no such field, and computing it requires a `TrainingVolume` (computing it on the
full volume would consult held-out sections). T01 did **not** add the field speculatively.
*Decision:* as proposed — **`TrainingVolume`, not `Volume`**, as a cached property computed from its
own sections, so it is leakage-free by construction and cannot drift per epoch. **T08 adds the
field**, plus a test that a plain `Volume` does not have it. Written into `specs/08` §2.

### C11. `split_holdout` and endpoint sections — **PROPOSED** (raised in T01)
The spec's alternating regime says "hold out every other section (fold selects the parity)" with a
"flanking gap ≈ 1× `median_spacing` on each side". Those are inconsistent for the first and last
section, which have no flanking pair. T01 holds out only *interior* sections in both regimes, so
every held-out section is an interpolation target. This changes the fold counts slightly (9 sections
give 3 and 4 held-out sections at parities 0 and 1, and 5 consecutive-3 folds); T10's regime
bookkeeping should read them from `split_holdout` rather than assuming.

### C12. `GeneMeta` and the cell-type ontology record are never defined — **RESOLVED in T02** (raised in T02)
T02 types `gene_descriptor(symbol, meta: GeneMeta | None)` and `celltype_descriptor(name, ontology:
dict | None)` but defines neither shape; `GeneMeta` appears nowhere else in `specs/`. *Resolution
(T02):* `GeneMeta` is a frozen dataclass in `data/text.py` with exactly the parquet columns the spec
lists (`symbol, full_name, summary, aliases, ensembl_id`), `aliases` as a `tuple[str, ...]`. The
ontology record is read for optional `"label"` and `"definition"` keys (Cell Ontology's own field
names); a non-empty dict carrying neither raises rather than degrading to the raw label, so a
wrong-shaped record cannot pass silently. If T06/T10 want more ontology fields in the descriptor
(synonyms, term id), adding them changes every cached vector — decide before the first real run.

### C13. `text_embedding_diagnostics` is stochastic but its signature has no seed — **RESOLVED in T02** (raised in T02)
Two of its three numbers are stochastic: the Leiden partition of the co-expression graph, and the
gene-pair subsample when `G` is large. Convention 3 requires an explicit seed. *Resolution (T02):*
the signature gains a required keyword-only `seed: int`; nothing else changes.

### C14. `resources/gene_meta.parquet` is described as "shipped" but nothing can build it offline — **OPEN** (raised in T02)
T02 says `GeneMeta` "comes from a local table shipped in `resources/gene_meta.parquet`", and also
forbids network access at train or test time. The repository has no such table and cannot generate a
real one without going online once. *Resolution so far (T02):* `build_gene_meta(symbols, cfg)`
writes the table at `Config.gene_meta_path`, hitting mygene.info only when
`Config.text_allow_network=True`, and `scripts/build_gene_meta.py` is the one-off online step;
without it every descriptor is the bare symbol, which is legal, warned about
(`GeneMetaUnavailableWarning`) and much weaker. **Someone has to run that script on a networked
machine and commit the resulting table before the first real training run**, or the paper's text
channel is symbols only. Not resolvable inside the offline test environment.

### C15. The distillation loss's reduction is unspecified — **PROPOSED** (raised in T02)
T02 writes `|| distill(t) - stopgrad(r) ||^2` over known entities, which is a sum over V entities and
out_dim components; used directly, its scale rides on the panel width, so `Config.w_distill=0.1`
would mean something different for a 200-gene and a 20 000-gene panel. *Proposal (T02):* it is a
**mean** over entities and components, so the weight transfers across panels. Revisit at T06 if the
term turns out to be too weak at `w_distill=0.1`.

### C16. GATE 2's 0° arm is one section, and it is the best-supported one — **OPEN, blocks T05** (raised in T04)

C1 settled the evaluation set: pooled cells within `thickness/2` of the query plane, equal `n` across
angles, own source section excluded from retrieval. Two defects surfaced only once the R² denominator
was fixed (see below); the first is now fixed, the second needs a decision.

**Defect 1 — the denominator (fixed).** C1 made `n` comparable but not the *denominator*.
`R²(θ)/R²(0°)` with each angle's R² taken about its own set's mean is a ratio of two different
questions: a 0° strip is one section, whose target variance is entirely in-plane, while a 90° strip
spans the stack. Measured, the per-cell denominators differ by 1.07× across the six angles. **Fixed
at T04**: G2.1d divides by the per-cell target variance over *all* training cells, shared by every
angle. Both numbers are reported. The verdict changes: 0.941 per-set → **0.886 fixed**, against a
required 0.90.

**Defect 2 — the depth mix (OPEN).** Under C1's membership rule a 0° plane through the volume's
centre selects **exactly one section**, the middle one — which is the *best-supported depth in the
stack* — while every oblique plane draws ~23% of its cells from the two **edge** sections, which have
training and retrieval evidence on one side only. Measured per-section fixed R²: edges **0.284** and
**0.366**, interior **0.414–0.471**. Predicting each angle's R² from its section mix alone, with the
angle playing no part, gives 0.4179 / 0.4166 / 0.4163 / 0.4189 / 0.4188 at 15/30/45/60/90° — flat to
0.0027, and reproducing the measured values. **The angle dependence in G2.1d is depth mix.**

Corroborating evidence that it is not the backbone: specs/04's own first remedy, raising
`n_plane_orientations` 4 → 8, moves the gate number by **+0.0009** (0.8858 → 0.8867). If oblique
parity were limited by the basis concentrating capacity on axis-aligned planes — the failure GATE 2
exists to catch — that is exactly the intervention that should have moved it.

*Two readings, and the choice changes the number the paper quotes:*

1. **Accept 0.886 as the verdict.** GATE 2 has failed and the next step is specs/04's remedy 3, a
   steerable/equivariant backbone — a design change. The P = 8 null result and the mix attribution
   both argue against this reading.
2. **Amend C1 so the 0° arm is depth-representative** — pooled over the coronal planes at *every*
   section rather than the single central one — and re-run. That is diagnostic G2.1e, which reads
   **0.960**. Defensible on C1's own terms (C1 exists to stop the ratio measuring sample size; this
   is the same defect one level down, measuring depth support), but it is a change to a settled
   contract made *after* seeing the number it changes, so it has to be a decision on the record.

*Recommendation:* **2**, with the amendment dated here and **both** numbers (0.886 single-plane,
0.960 depth-matched) reported in the paper. Not adopted unilaterally: T04 stops at the failed gate,
per specs/04's "Do NOT proceed to T05 without G2.1 passing".

*A third option worth pricing if 2 is rejected:* keep C1's single-plane 0° arm and **drop the edge
sections from every angle's evaluation set**. That equalises support rather than the mix, costs ~22%
of the oblique `n` (the common `n` would fall from 1011 to ~780, still above
`Config.gate2_min_cells_per_angle = 500`), and has the merit of removing the confound at source
rather than averaging over it.

## D. In the design docs but missing from `specs/11_COVERAGE_MATRIX.md` — **all five settled 2026-08-15**

The matrix says an unmapped design component is an omission to be flagged. These were the ones I
found; each is now written into the task file named in the last column and into the coverage matrix,
and each is **due at that task**, not before.

| Design location | Component | Decision | Landed in |
|---|---|---|---|
| `v23_design.md` §7 Baselines | **v14 and v18** are listed as baselines; T10 wires only `run_v20` | **Dropped explicitly**, with the one-line reason in the methods: both are superseded by v20 on every metric of the existing bench3 campaign, and v20 is the version the no-regression guarantee is stated against | `specs/10` §2 |
| `v23_design.md` §7 Datasets | "≥ 1 non-brain (embryo/tumour) and ≥ 1 non-transcriptomic panel (EASI-FISH)" — a reviewer defence against brain-only overfitting | **Required.** `run_benchmark` refuses to produce a headline table without both and names what is missing; a campaign run without them is a development run and says so on the report's first line | `specs/10` §3 |
| `v23_design.md` §3.5 | "Calibrate `π` **and the mean–variance relation** per gene against the flanking sections" — T09 calibrates `π` only | **Both.** A per-gene correction on `log θ` fitted the same way as the `π` one; they are not substitutes (`π` moves the zeros, `θ` the spread of the non-zeros), and T06's `test_mean_variance_relation` is what it protects at inference | `specs/09` §2 |
| `v23_design.md` §2.2 / §7 E1 | Zero-shot table must report **both** `r_g = 0` (pure text) and `r_g = ψ(t_g)` (distilled) | **Both arms.** One arm cannot separate "the text channel carries the gene" from "the distillation head guessed a residual", which is the open-vocabulary claim | `specs/10` §5 (E1) |
| `v23_design.md` §5, §6 | The v20 **Bernoulli cross-mix** itself (see A6) | **Implement in T06**, behaviour pinned by a test | `specs/06` §4b |

---

## E. Recorded, no action needed

- **E1.** Convention 4 says coordinates are always `(N, 3)`; T01 stores `Section.coords` as `(N, 2)`
  plus a scalar `z`. Sanctioned exception, noted in `CLAUDE.md`; `to_xyz()` is the only way to get
  `(N, 3)`.
- **E2.** T06's encoder input `Enc(log1p(counts / size_factor))` is normalised expression, which
  Convention 5 permits (input-side) and forbids as a decoder *target*. Consistent; worth an explicit
  comment at the call site since it looks like a violation at a glance.
- **E3.** The dependency pins in `pyproject.toml` are a mutually-consistent Python 3.11 stack
  (torch 2.2.2 / numpy 1.26.4 / scanpy 1.10.1); every pin was checked to exist on PyPI. They are
  deliberately older than the ambient `ruff`/`mypy` on this machine — reproducibility of paper
  numbers beats recency. Revisit only if a task needs a newer API.
- **E4.** GATE 1 in T03 requires all four criteria (G1.1–G1.4) to pass but the stop instruction
  names only G1.3. Treating G1.3 as the stop-the-project criterion and G1.1/G1.2/G1.4 as
  fix-the-bug criteria, since the latter three are implementation-correctness checks. *After T03 and
  its amendment:* all four pass on the 3000 µm gate fixture. The first pass failed G1.3c/G1.3d on a
  1000 µm fixture; the spec defect behind that is A7, now resolved. The split the stop instruction
  implies turned out to be the right one — G1.3a/G1.3b (the mechanism) never failed.
