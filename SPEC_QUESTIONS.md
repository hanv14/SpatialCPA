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

### B7. The layout sampler is a conditional Strauss, not a Strauss process (T05) — **RESOLVED in T05**
Step 1 draws `N ~ Poisson(N_expected)` and step 3 thins; conditioning on `N` and then thinning is
not the same object as a Strauss process (which produces fewer points than its Poisson envelope).
The `20*N`-proposals escape hatch then silently breaks `test_expected_count_matches`.
*Proposal:* implement and document it as a **conditional-on-N** Strauss sampler (which is what makes
the count-from-intensity claim work), make the proposal cap a `Config` field, and raise rather than
warn under test.

*Resolution (T05):* done as proposed. `sample_layout` is documented as conditional-on-`N` in the
module docstring and in its own; the cap is `Config.layout_max_proposal_factor`; exhausting it warns
`ProposalBudgetWarning` **and** sets `Layout.budget_exhausted`, and `pyproject.toml` promotes that
warning to an error under pytest — so a production run degrades loudly while no acceptance test can
measure a truncated pattern. Measured on the fixture: 1529 points placed from 2048 proposals, and the
mean count over 50 seeds is within **0.50 %** of `N_expected` (limit 5 %).

### B10. Fitting the Poisson intensity by MLE is ill-posed as specified (T05) — **RESOLVED in T05, two fixes; the residual risk is now open risk R4, merged with T06's**
**Merged at T06:** B10's open half and what T06 first filed as a separate risk are the *same*
pathology — a head fitted by its own likelihood improving at that likelihood while the generated
section degrades — on two different heads. They are tracked as the single named risk **R4** in
`PROGRESS.md`, with the success criterion in `specs/08`. Do not close one and assume the class is
closed.

`test_poisson_nll_recovers_intensity` asks a neural intensity to be fitted by the process
log-likelihood. Measured, that fit does **not** recover the truth; it exploits two degeneracies.

1. *Every cell of a section is recorded at that section's nominal `z`*, while the integral runs over
   a continuous slab, so an intensity that puts its mass in thin sheets at the nominal depths scores
   arbitrarily well. Measured: the NLL fell **three nats below its value at the true intensity**
   while the correlation with that truth stayed at 0.00.
2. Against a **fixed** Monte-Carlo integration set the intensity grows spikes between the
   integration points.

*Resolution:* `fit_intensity_head` jitters each cell's depth uniformly within its slab and redraws
the MC points every step. Both are statements the data already makes (a cell is *somewhere* in its
slab; the integral is an expectation), not regularisers.

*Open risk, recorded not fixed:* the Poisson MLE of a flexible intensity still overfits the point
pattern. With the default `fourier_bands_xy = 8` the recovered correlation decays from **0.97 at 300
steps to 0.28 at 1200** while the NLL keeps falling. T05 specifies no regulariser and T05 does not
invent one: the acceptance test lowers the head's spatial basis to the scale the intensity varies on
(`fourier_bands_xy = 2`) and says so. **T06's trainer needs an explicit answer** — early stopping on
a held-in section, a smoothness penalty, or a basis tied to the fitted length-scale.

### B11. T05's Potts smoothing (ICM) erases rare types before `beta` does anything (T05) — **RESOLVED in T05; `Config.potts_update` added**
T05 §3 says "`potts_iters` rounds of **ICM**", and separately forbids smoothing so hard that rare
types vanish, and separately requires `beta` to be *fitted* by matching neighbourhood purity. The
three cannot hold together: ICM takes the `argmax`, i.e. it seeks the **mode** of the Potts
posterior, and its first sweep is essentially `argmax_c lambda_c` whatever `beta` is.

Measured on the fixture, with a 2 % cell type injected, at the **smallest coupling the fit can
choose** (0.02): ICM takes the rare type to **0.000** of its expected count and purity to **0.785**
against a tissue value of **0.688** — overshooting the fit's own target before the coupling does
anything. There is no `beta` left to fit.

*Resolution:* `Config.potts_update` (`"gibbs"` default, `"icm"` selectable). Gibbs **samples** the
same conditional instead of maximising it: the marks stay a draw from the marked point process the
section is supposed to be, purity becomes monotone and fittable in `beta`, and at the same coupling
the rare type retains **1.00**. ICM is kept and asserted as the negative control
(`test_icm_erases_rare_types`), because "T05 said ICM" has to stay visible in the code and in the
methods section. `fit_potts_beta` additionally takes the **intensity** (T05 writes `fit_potts_beta(vol)`)
— `beta` closes the gap between a draw from `lambda_c` and the tissue, so it cannot be fitted without
knowing `lambda_c` — and enforces the rare-type floor as a **constraint**, not a hope: measured, the
purity criterion alone would pick 0.278 and the floor takes it to 0.144.

### B12. `test_pcf_matches_real`'s stated range cannot fail (T05) — **RESOLVED: `specs/05` and `specs/10` amended to `[0, 3R]`**
The criterion is `max |g_sim(r) - g_real(r)| < 0.15` over `r in [r0, 3R]`. A hard-core process differs
from a Poisson one only *inside* the correlation hole — and the hole ends at about `r0`, because `r0`
**is** a low percentile of the nearest-neighbour distances. The stated range therefore begins exactly
where the discriminating signal stops.

Measured on the fixture (real `g` pooled over the six training sections, simulated over three seeds):

| | over `[r0, 3R]` (the spec's range) | over `[0, 3R]` |
|---|---|---|
| field mode | **0.093** | **0.093** |
| pure Poisson (ablation A4) | **0.070** — *passes* | **0.994** — fails |

*Done at T05:* `test_pcf_matches_real` asserts the criterion over both ranges, runs the pure-Poisson
control against `[0, 3R]`, and **pins the old range's blindness itself** as an assertion so it cannot
change unnoticed.
*Amended (accepted 2026-08-16):* `specs/05` now states the range as **`[0, 3R]`** with this table as
its justification, and `specs/10` §4 carries the matching warning for **ablation A4** — reported over
`[r0, 3R]` the ablation table would claim the repulsion buys nothing, which is a false null, so every
A4 number states its range. A caveat: the comparison is also
bin-resolution-sensitive at the bin straddling `r0` — at 48 bins rather than 24 the generated pattern
reads 0.315 there against the tissue's 0.646, because a strict hard core at the 1st percentile is
still stricter than the tissue's own minimum (7.90 µm against 7.75 µm). That is B6 again, surviving
at the 1st percentile, and it is a **property of the method**, not of the estimator.

### B13. T05 needs a `Plane` and T07/T09 own the plane geometry — **RESOLVED in T05 (minimal module)**
`sample_layout(intensity_fn, plane: Plane, cfg, seed)` needs a plane type, but `specs/00` puts plane
geometry in `infer/planes.py` at T07/T09, and T03 deliberately built only a test-tree stand-in
(`tests/fixtures/planes.py`) rather than skipping ahead. T05 is package code, so a stand-in in the
test tree will not do.
*Resolution:* `spatialcpav25_gen/infer/planes.py` exists now, holding **only** what T05 uses — `Plane`
(origin, canonical in-plane basis, window half-extents, slab thickness), `plane_from_normal`,
`section_plane`, `uniform_plane_points`, `uniform_slab_points` — with the same canonical basis
construction as T03's stand-in. `intersect`, `random_plane_pair`, `Surface` and the curved surfaces
are still T07/T09's and are to be **added beside** these, not on top of them.

### B14. The fixture's "~2 %" rare cell type is 6.3 % (T01 fixture, needed by T05) — **RESOLVED in T05 (test-side)**
`tests/fixtures/synthetic.py` documents "one rare (~2%) type, which T05's `test_rare_types_survive`
needs", but `type_bias = linspace(0.6, -2.4, 6)` through the Gumbel argmax realises **6.3 %** for the
rarest of the six. A 6 % type is a perfectly good rare type, but T05's criterion is stated at 2 % and
`Config.potts_rare_prevalence` (5 %, the benchmark's own `RARE_CELLTYPE_FRAC`) would not even
classify it as rare.
*Resolution:* the fixture is left alone — every earlier task's numbers were measured on it — and
`test_rare_types_survive` injects a genuine 2 % type on top of the fixture's own field (a stripe,
0.2 %–3.8 %, i.e. interspersed rather than a compact niche, which is the hard case). The 6.3 % is
pinned by `test_fixture_rarest_type_is_six_percent` so the injection cannot be quietly dropped.

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

### B15. "Within 10% of the real section's value" has two readings, and they disagree (T05) — **RESOLVED (decided 2026-08-16): the referent is an ideal draw**
T05's definition of done is `field` mode "within 10% of the real section's value" for
`paper_celltype_localization`. Measured on all three held-out sections of the fixture:

| held-out | **self** (section vs itself) | **generated** | **ideal** (independent draw from the *true* law) | **flanking** (= `resample`) |
|---|---|---|---|---|
| s02 | 0.8730 | 0.7933 | 0.7144 | 0.4797 |
| s04 | 0.9353 | 0.5732 | 0.6298 | 0.4966 |
| s06 | 0.9581 | 0.7719 | 0.8091 | 0.6193 |
| **mean** | **0.9221** | **0.7128** | **0.7178** | **0.5319** |

Against the held-out section's **own** score the layout head reaches **0.776** and **fails** the
criterion (passing on one section of three). Against the **flanking** real section it is at
**1.35x** and beats it everywhere. Stated plainly: the generated layout *is* materially below the
held-out section's own localization, and that is a fact about the layout head, not a wording choice.

What settles which reading is a criterion: the **ideal** arm — positions and marks drawn from the
fixture's own generative composition, i.e. an independent draw from the process that produced the
held-out section — reaches only **0.779** of the self-score, statistically the same as the layout
head's 0.776. The metric compares point clouds by a Sinkhorn divergence normalised against a
within-tissue null, so a *different realisation* of the same law is already ~22% of the way from the
section to that null. A 0.90-of-self criterion therefore asks the layout head to beat the generative
process that produced the data.

*Decision (2026-08-16): the criterion is `generated >= 0.90 x ideal`* — the independent draw from
the known generative law — **not** 0.90x the held-out section's self-score, since the ideal draw
itself reaches only 0.779 of that and the original therefore asks the layout head to beat the process
that made the data. Measured: **0.994**. Stated on the **mean over held-out sections**, because the
metric's per-type null normalisation makes a single section's score noisy (see B15a below). On real
data there is no ideal draw and the referent is the **flanking baseline**, reported by T10.

The superseded reading stays in the suite as a **strict xfail** carrying its failing number (0.776),
so the shortfall against the real section remains in the record and a later task that closes it
breaks the suite until this entry is updated. `specs/05` amended; the generalisation to every metric
is `specs/10` §1's achievable-ceiling protocol.

### B15a. `celltype_localization` is unstable for abundant, tissue-wide cell types (T05 measurement, owed to T10)
The per-section spread behind B15 — 0.909 / **0.613** / 0.806 — is **not** open risk R3. The low
section is `synthetic_s04`, **the exact centre of the nine-section stack** (index 4 of 9, four
sections from either end), and `alternating` never holds out an end section, so the boundary regime
is not in that table at all.

The cause is the metric. It scores a type as `1 - d_obs / d_null`, where `d_null` is the divergence
between that type's cloud and an equally sized random draw from the whole section. For a type
occupying a third of the tissue — which *is* nearly tissue-wide — `d_null` collapses to ~0.08, while
a localised minority type's is 0.16-0.57. Measured on the **ideal** arm, so this is the metric and
not the sampler:

| | type 0 (weight 0.34) | localised minority types |
|---|---|---|
| `d_null` | 0.072-0.087 | 0.079-0.573 |
| score at s04 | **0.332** (`d_obs` 0.058) | 0.595-0.908 |
| score at s06 | **0.841** (`d_obs` 0.011) | 0.630-0.906 |

The same absolute realisation noise costs the abundant type four to eight times as many score
points, and it carries a third of the weight: type 0 alone is essentially the whole 0.18 spread
between those two sections. `evaluate_paper` guards this only at `d_null < 1e-4`, three orders of
magnitude below where it bites. *Owed to T10:* report **per-type** ceilings, not only the weighted
average (`specs/10` §1, point 5), and read any `celltype_localization` difference against the
ceiling's own spread.

### B16. T06's gene-gene covariance criterion is below the achievable ceiling (T06 measurement) — **OPEN, criterion amended, model shortfall recorded**
`test_shared_latent_preserves_covariance` requires the generated section's gene-gene correlation
matrix to be closer to the held-out section's than the independent-donor baseline's **by a factor
of two in Frobenius norm**. Measured on the synthetic fixture, three separate things are wrong with
that as written, and the first is fatal independently of the model.

**1. The criterion is below the ceiling.** Every arm's error contains the same irreducible term: a
correlation matrix estimated from ~1500 cells has a sampling error of about `1/sqrt(N)` per entry
whatever produced the cells. Measured with T05's ceiling protocol — the **same cells**, the fixture's
**true** `mu`, and only a fresh count draw — the Frobenius error is **5.51 ± 0.05** (5.57 at the
narrow gap; an independent draw of the whole law gives 5.71). Against a baseline at **11.31**
(100 µm gap) the criterion asks for **< 5.65**. A *perfect* model has a coin-flip's chance, and at
the narrow gap (baseline 7.90, criterion < 3.95) it cannot pass at all. This is B15 again: a stated
range is not an achievable range.

**2. The comparison is against a copy, and at a narrow gap a copy is nearly the answer.** With a
120 µm in-plane / 200 µm along-z correlation length, a section 50 µm away shares most of its latent
with the target, so a donor baseline assembled from the immediate flanks is not a weak control — it
is v20's own "the narrow-gap benchmark is saturated: adjacent-slice recombination ties there and
nothing can win it by more than noise". Measured: the baseline's error grows 7.90 → 11.31 as the gap
goes 50 → 100 µm, i.e. the criterion's difficulty is mostly a statement about the holdout regime.

**3. The mechanism the criterion is *about* is real, and it is measurable directly.** Holding the
donors fixed and varying only whether the draw is per **cell** or per **gene** — which is the whole
of the chimerism claim, with the positional confound removed — the mean |off-diagonal correlation|
against a real 0.1425 is:

| donors mixed | 1 (verbatim copy) | 2 | 3 (the competing method) | 5 | 10 |
|---|---|---|---|---|---|
| mean \|off-diag\| | **0.1360** | 0.1165 | **0.1116** | 0.1074 | 0.1017 |
| Frobenius error | 9.90 | 10.57 | 11.24 | 11.96 | 13.06 |

Monotone in the number of donors, 22 % of the covariance magnitude lost at the method's own `D = 3`.
**The paper's covariance argument is confirmed**; what cannot be expressed as "2× in Frobenius norm"
is the *model's* advantage over it.

**The ceiling number, stated plainly.** T05's ceiling protocol on the default `alternating`
holdout, three draws: **5.601** (spread ±0.05; the wide-gap section gives 5.513, and an independent
draw of the *whole* generative law rather than only the counts gives 5.705). The
independent-donor baseline on the same section is **7.783**. Fifty per cent of 7.783 is **3.892**,
which is **below** the ceiling of 5.601 — by 1.7, i.e. 30 % of the ceiling and thirty-four times its
own draw-to-draw spread. **The criterion is unsatisfiable by any generator whatsoever**, including
the fixture's own generative law, and that conclusion involves no model, no training budget and no
choice of mine.

*Amendment, implemented at T06.* Three parts, and they do not all stand on the same footing —
see the pre-registration note below.

(a) The mechanism becomes its own acceptance test,
`test_per_gene_independence_destroys_covariance`, measured on real donors with the confound removed
— it needs no trained model and it is what the paper's covariance section actually claims.
(b) Every arm is reported **relative to the measured ceiling**, with the systematic part
`sqrt(err² − ceiling²)` beside the raw number, because noise and bias add in quadrature.
(c) The spec's own criterion keeps its name and its statistic and is held as a **strict xfail** at
its measured value, so the shortfall is not reworded away and a later task that closes it breaks the
suite until the record is updated (T05's precedent for its localization criterion).

**Pre-registration: was the magnitude/pattern decomposition chosen before or after seeing which
component passed? After. Explicitly after.** The order of work was: measure the Frobenius ratio
(2.06, then 1.20 — failed); hypothesise the decoder's `mu` link and measure it (no gain); measure the
ceiling (which settled that the criterion is unsatisfiable); measure the chimerism isolation (which
confirmed the mechanism); *then* notice that the retained-covariance **magnitude** was the component
on which the model beat the baseline, and adopt magnitude-plus-pattern as the replacement statistic.
The pattern floor at `0.9 ×` the baseline was added as a guard against a model buying magnitude with
random correlations, and it too was set knowing the measured ratio was 0.990.

So the three parts have three different standings, and they should be quoted with them:

* **(b) the ceiling, and therefore the unsatisfiability of the stated criterion, is model-free and
  choice-free.** Nothing about it depends on what passed.
* **(a) the chimerism isolation is a confirmed prediction, not a selected statistic.** The paper's
  argument predicts a loss that is monotone in the number of donors mixed, before any measurement;
  the measurement then produced 0.978 / 0.920 / 0.897 / 0.884 / 0.844 at D = 1/2/3/5/10. That is a
  prediction met, on both holdout gaps.
* **the model-versus-baseline half is post hoc, and it does not survive an out-of-sample check.** On
  the wide-gap (`consecutive-3`) holdout the same decomposition gives a magnitude error of 0.213 for
  the model against 0.214 for the baseline — **ratio 0.995, no advantage at all** — where the default
  holdout gives 0.458. **The claim "the shared latent preserves more covariance than the competing
  method's sampler" is therefore NOT established by T06.** What is established is that the mechanism
  the claim rests on is real (a), and that the criterion as specified could never have shown it (b).

*The model shortfall is recorded as open risk R4, not amended away:* at the wide gap the model's
Frobenius error is **17.7** against the baseline's 11.3 — worse, not better — with the covariance
magnitude 21 % too high where the baseline's is 21 % too low. The head overfits the likelihood, and
the terms that would stop it are T08's.

### B18. `test_zero_shot_gene_decoding` cannot pass on the synthetic fixture (T06 measurement) — **RESOLVED: recorded as a strict xfail; the real test is T10's E1**
Measured, with the gene holdout actually enforced: per-gene mean expression correlates with truth at
**r = −0.368** for the 40 never-trained genes, against **+0.946** for the seen ones. It is not close
and it is not noise — it is negative.

The cause is the fixture and it was predictable from T02's own number. A held-out gene's free
residual `r_g` never receives a gradient (asserted: `max |r_g| == 0.0` exactly), so everything the
decoder can know about it comes through `W t_g` — and the fixture's gene names are arbitrary strings
(`Gene0042`), for which T02 measured a text/co-expression Spearman of **+0.0055**. Zero text signal
in, no transfer out. The two measurements agree, which is itself evidence the channel is wired
correctly rather than broken.

*Resolution:* the spec already permits it ("if this fails badly, note it and continue — it is a
capability experiment, not a gate"), so the test is kept **by name, at its stated `r > 0.4`
threshold, as a strict xfail** holding the measured −0.368. The real test is T10's capability
experiment **E1** on a real panel, which needs `resources/gene_meta.parquet` — still open as **C14**
and still the blocker T02 flagged.

*A bug this found, and it is the reason the number changed.* The first measurement reported
**r = 0.9235** and passed. It was in-sample: `train_ctfflow` accepted `gene_pool`, documented it, and
never forwarded it to `sample_batch`, so the "held-out" genes were trained on. Fixed, and pinned by
`test_trainer_forwards_the_gene_pool`, which asserts by mutation on the one quantity only training
can move (`r_g` outside the pool must be exactly zero, inside must not). The lesson generalises: the
existing test exercised `sample_batch` directly and could not see that the trainer dropped the
argument.

### B17. T06's `test_sparsity_preserved` tolerance is gap-dependent (T06 measurement) — **RESOLVED: measured on the default holdout**
Per-gene detection rate, generated vs held-out real, at 1200 steps: `r = 0.989` and mean absolute
difference **0.036** on the default `alternating` holdout (50 µm to the nearest donor), against
`r = 0.969` and MAD **0.056** on `consecutive-3` (100 µm) — the same model, the same criterion, one
side of `< 0.05` each. T06's spec names no holdout regime. *Resolution:* the acceptance test runs on
`alternating`, which is T01's default and T10's headline regime, and the wide-gap number is reported
as a diagnostic. Detection is what T09 §2 calibrates; closing 0.056 is that task's job and B17 is
the measurement it starts from.

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

### C1c. The retrieval z window is sized off a statistic of the stack, not off the query's evidence — **RESOLVED (implemented 2026-08-16; `gap_factor` default still owed a T09 sweep)** (raised in T04, found chasing a T04 measurement)
`specs/04` states the candidate window as `retrieval_z_window × median_spacing` and says nothing
about irregular stacks. `median_spacing` is a property of the whole volume, so after any holdout,
dropped section or damaged section it sizes the window off tissue that is not there.

**Measured.** On the GATE 2 fixture under a `consecutive`-3 holdout the training stack is
z = 0, 200, 250, 300, 350, 400 µm. Four of five gaps are 50 µm, so the median stays 50 and the window
stays 150 µm — while the section at z = 0 is 200 µm from its nearest neighbour. All **13 500 of
81 000 training cells** (16.7%) retrieved nothing and trained against a fully masked attention row.
Held-out reconstruction inverted: R² at z = 150 µm was **−0.166 with retrieval against +0.187 with
the retrieval branch ablated entirely**. Widening the *training* window to cover the real gap takes
the same depth to **+0.359**. None of this is visible on the gate: held-in G2.1a is 0.937 / 0.926 /
0.924 at window 3 / 4 / 5 and passes throughout.

**Decision.** Two-term, per-query bound:

```
|z_j − z_p| ≤ max(retrieval_z_window × median_spacing,
                  retrieval_z_window_gap_factor × gap_to_nearest(p))
```

with `retrieval_z_window_gap_factor ≥ 1` enforced by `validate`, so the nearest surviving section is
admissible at any gap and the pool is non-empty by construction. Three consequences, all now tested:

(a) **Order is part of the rule.** `gap_to_nearest` is measured *after* `exclude_z`, the own-section
exclusion and the gap-aware dropout. Sized before them, a cell's own section sits at gap 0, the
relative term collapses, and the guarantee is void — silently, and only in the leave-own-section-out
configuration C1(a) makes GATE 2 depend on.

(b) **Bitwise identity on the evaluation path.** On a regular stack the gap is at most one spacing,
`2 × 1 ≤ 3`, the absolute term wins, and nothing moves. Every published number is measured on that
path, and `test_gap_relative_window_is_identity_on_a_regular_stack` asserts it.

(c) **The curriculum path changes, deliberately.** With `apply_dropout=True` the nearest section is
dropped, so the gap — and the window with it — widens. That is the curriculum becoming
self-consistent rather than a side effect: measured on the gate fixture, serving a probe donors from
beyond its training window drove held-out R² from −0.02 to −0.35, so simulating a wide gap under a
narrow window is training in precisely that mismatch. Pinned by
`test_gap_relative_window_follows_the_dropout_gap`.

**Still owed.** `retrieval_z_window_gap_factor = 2.0` is a placeholder, not a swept value — chosen so
a query one gap from its evidence also reaches roughly the next section along. The sweep belongs in
T09's config selection on internal LOSO over *training* sections; choosing it against held-out
sections would be a leak (`CLAUDE.md`, leakage discipline).

**Same bug, patched locally at T04, and *not* fixed by this.** `tests/gate2_criteria.py`'s
`G23_Z_WINDOW = 5.0` overrides the window for G2.3 alone because the 0.2/0.8 fractional depths put
one flank four spacings out, beyond the default window of 3 — the same defect (a window sized off a
statistic rather than off the evidence the query needs), noticed at T04 and worked around in the test
rather than fixed in the model. The gap-relative term does **not** subsume it: it sizes the window off
the *nearest* section, and G2.3's far flank is the *second*. Verified directly — at the default config
the donor sections present are `[near]` only at fractions 0.2 and 0.8, `[near, far]` at 5.0. The
override stays, with that reason recorded at its definition. Whether the window should also carry a
"reach the k-th nearest section" term is a **separate open question**, deferred rather than answered.

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

### B19. `build_gene_meta` returned four species' genes for a one-species request (reported, four defects, all fixed)
Reported after a build of a **1138-symbol mouse panel** with `--species mouse`: the written table held
ENSMUSG 389, ENSMSIG 324 (ground squirrel), ENSNVIG 234 (mink), ENSMPUG 111 (ferret), ENSFALG 73
(falcon), FBgn 2 (fruit fly) and 5 with no id; summaries 144/1138. Two earlier symptoms in the same
session: `Config.mygene_species = "human,mouse"` did not restrict to two species either, and a
`--species mouse` re-run printed "wrote 1122/1122" while the file kept 1138 rows of the old mixed data.

**`species` was reaching the API.** Read from the installed client's source: `querymany` forwards
`species` verbatim into the POST body, so the parameter was sent. The mixing came from four defects
*around* the query, three of them certain from the code alone:

1. **The cache short-circuited every re-run.** `missing = [s for s in symbols if s not in cached]`,
   so after one bad build every symbol was cached and a corrected `--species` run **issued no query
   at all** — the species argument could not filter because nothing was requested. This is the main
   reason it "was not filtering".
2. **The writer merged and never replaced.** `pd.concat([cached, new_rows])` with no removal, so a
   1122-symbol request left 16 stale rows of unknown provenance in a 1138-row file.
3. **The printed count was not a property of the file.** "1122/1122" counted *requested symbols
   carrying a full name*, not rows on disk — a number that cannot detect a wrong file.
4. **Among several hits per symbol, the first won.** `if query in out: continue`, with no species
   check, no exact-match preference and no `_score`; under `scopes="symbol,alias"` another gene's
   *alias* in another species routinely arrives first. `taxid` was not even in `fields`, so what came
   back was never inspected, and there was no species column, so the result could not be audited.

**And the network path had no test at all**, which is how it survived T02. That is fixed too: a
`MyGeneClient` protocol plus `load_mygene_client` seam (the same shape as T02's transformer seam) lets
a fake client reproduce the reported response offline.

*Fixed, with the reported response replayed as the check* — 1138 symbols in, 1138 rows on disk,
`{'ENSMUSG': 1133, 'None': 5}`, one resolved taxid, 749 wrong-species hits dropped loudly:

| # | fix |
|---|---|
| 1 | `taxid` requested and **verified** per hit against the resolved taxid; wrong-species hits dropped, and if *nothing* resolves to the request the query **raises** rather than writing (a systematic filter failure is not a per-gene absence) |
| 2 | `resolve_species` refuses anything but **one** species; `Config.mygene_species` default `"human,mouse"` → `"mouse"`, since a symbol-keyed table describes one organism |
| 3 | `species_requested` and `species_resolved` columns on every row; `gene_meta_summary` reports both plus the Ensembl-prefix histogram that made the bug visible |
| 4 | **replace by default**, `merge=True` for the accumulate case (and a merge across organisms raises); the script prints the row count **of the file on disk** and exits non-zero if it disagrees with the request |
| 5 | `load_gene_meta(path, species=...)` **raises** on a mismatch, and on a pre-species-column table whose rows carry metadata but no species — while still allowing legitimately unresolvable symbols, or the gate would be unusable on any real panel |
| 6 | best hit chosen, not first: right species, then exact symbol match over an alias match, then `_score`; residual same-species ambiguity is counted and warned |

**The one real table that has been committed is the same bug again, in its most deceptive form.**
`b68712d` added `resources/gene_meta.parquet` for "the STARmap panel". Audited with the tooling this
repair added: 28 rows, the right mouse-cased symbols (`Slc17a7`, `Gad1`, …), **Ensembl prefixes
`{'ENSG': 28}` — every row human**, and 28/28 full names, summaries and ids. `Config.mygene_species`
was `"human,mouse"` when it was built, mygene matched the mouse-cased symbols case-insensitively, and
the human hit outranks the mouse one. **Its coverage looks perfect precisely because it is wrong**:
human records are the best-annotated in NCBI, so an accidental human resolution *maximises* summary
coverage — which is why correctness is checked on `species_resolved` and the prefix histogram and never
on coverage. Moved to `resources/gene_meta.human_orthologs.parquet` (kept, not deleted: it is the right
table for a *human* dataset, and T10 needs one), `Config.gene_meta_path` is deliberately absent so
`load_gene_meta` raises "build it", and `resources/README.md` records the audit.
`test_committed_gene_meta_tables_are_species_checkable` pins the state.

*Not measurable here:* the **mouse-only summary coverage**, because mygene.info is 403'd in this
container (see C14). The old 144/1138 is explained — summaries were lost on exactly the 744 rows that
resolved to non-reference species, whose gene records carry no NCBI summary — and the corrected query
keeps the mouse hit, so coverage should rise to whatever fraction of these 1138 mouse genes have an
NCBI summary. `scripts/build_gene_meta.py` now prints that number (`with summary N/rows`) as a
property of the file; it has to be read off a run with network access.

### B19a. `ensembl_id` came from the wrong element of the right hit (reported, fixed)
The second report on the same build: `species_resolved` uniformly `['10090']` — the taxid filter
working — while the Ensembl prefixes were ENSMUSG 390, ENSMSIG 321, ENSNVIG 241, ENSMPUG 111,
ENSFALG 73, FBgn 1.

**The two are not contradictory, and the reason matters.** Every other field of those rows came from
the mouse hit, so the ids can only have come from a **non-mouse element of the mouse hit's own
`ensembl` list**: mygene's `ensembl` field carries cross-species mappings, and `_ensembl_id` took
`value[0]` of it — a coin toss over whatever orthologue mappings the record happened to include. The
`taxid` of a hit says nothing about which element of its `ensembl` field you then read.

*Two fixes, plus one honesty repair:*

1. **`_ensembl_id(value, prefix)` selects by prefix**, not position: the first id whose prefix is the
   requested species' (`SPECIES_ENSEMBL_PREFIX`, mouse `ENSMUSG`). A hit with ids but none of them
   this species' now yields **`None`** — an absent id, which the descriptor never reads — rather than
   another organism's, which every downstream join does. Counted and warned.
2. **`_check_ensembl_prefix` asserts the stored value** at build *and* at load, separately from the
   taxid check, because `ensembl_id` is the field everything downstream reads and it must not inherit
   the hit's credibility.
3. **`species_resolved` is now read from the hit** (`best["taxid"]`), not written as `str(taxid)` from
   the *request*. As written it could never disagree with the argument, so "species_resolved is
   uniformly 10090" was true **by construction** — it read as evidence and was not. That is why the
   report looked self-contradictory.

*Replayed on the reported response* (mouse hits throughout, `ensembl` mixed): 1138 rows, prefixes
`{'None': 748, 'ENSMUSG': 390}`, 747 wrong-prefix hits reported, table accepted by the species gate.
**Whether those 748 become `None` or become real ENSMUSG ids on the live API depends on whether the
mouse id is present further down each list** — which needs the raw response, hence
`scripts/build_gene_meta.py --dump-raw`.

### B20. Mouse summary coverage is 148/1138 — **CONFIRMED genuinely sparse**; orthologue fallback **IMPLEMENTED, default ON**
Coverage fell from **1054/1138 (93%)** under the old human-leaning query to **148/1138 (13%)** once the
query resolved mouse. 93% is the *human* rate; the question is whether 13% is the true mouse rate.

*What is known.* The panel is **not** dominated by unannotatable symbols: of 1138, only **33 (2.9%)**
are RIKEN clones (`*Rik`) or predicted genes (`Gm#####`); **1105 (97.1%)** are conventional named mouse
genes (`A2m`, `Abca8a`, `Abcc9`, …). So the panel's composition does not explain a 7× drop, and 13%
should not be accepted without evidence.

**Confirmed from the committed mouse build itself** (`resources/gene_meta.mouse_prefix_bug.parquet`,
1138 rows), which needs no network to audit. Three findings, and together they answer the question:

| | |
|---|---|
| **summary presence is flat across the wrong-prefix groups** | ENSMUSG **11.5%**, ENSMSIG 10.9%, ENSNVIG 17.0%, ENSMPUG 17.1%, ENSFALG 11.0% |
| **`full_name` present for 1138/1138** | a mouse record *was* found and read for every symbol |
| **by symbol class** | conventional named genes **148/1105 (13.4%)**; clone/predicted symbols **0/33** |

The first says the prefix defect and the summary sparsity are **independent** — had the id defect been
costing summaries, the ENSMUSG rows would carry them at a higher rate, and they do not. The second says
the mouse record was reached in every case, so the 990 absences are a property of *that record*, not of
picking the wrong organism. The third says what the sparsity looks like: the genes that do have
summaries are the well-studied ones (`Abcc9`, `Acta2`, `Adam12`, `Adcyap1`, `Adra1a`, …), which is
exactly the shape of NCBI mouse curation, and 13.4% is consistent with it. **So: genuinely sparse.**

One door remains open, and it is narrow: whether a *different mouse hit for the same symbol* carries a
summary the selected one lacks. That is the only way selection could still be involved, and it is now
counted rather than argued about.

*The discriminator, added rather than argued.* `_query_mygene` now counts symbols where
**the selected hit has no summary while another same-species hit for the same symbol does**, and warns
with the count. That is the whole difference:

* count ≈ 0 → confirms the audit above outright, and the fallback below is the only question left;
* count large → a residual selection effect, and the ranking needs a summary-aware tiebreak among
  otherwise equally good same-species exact matches (a *tiebreak*, never a preference strong enough to
  pick a different gene).

It cannot be measured in this container (mygene.info is 403'd — C14), so it is one run away:
`python scripts/build_gene_meta.py --species mouse --symbols-from resources/mouse_panels_symbols.txt`
prints `with summary N/rows` and emits the selection warning.

*Proposed fallback, if sparsity is confirmed — the human orthologue's summary, explicitly labelled.*
Matching the inclination stated in the report, with four constraints that are not optional:

1. **A separate, recorded provenance field**, not a substituted value: new column `summary_taxid`
   holding the taxid the summary came from (10090 native, 9606 orthologue, null none). Coverage is
   then always reportable **split by source**, so "N/1138 have summaries" can never again mean two
   different things.
2. **The orthologue is resolved through `homologene`, never by uppercasing the symbol.** Uppercasing is
   precisely the mistake that produced the all-human table (B19): `Slc17a7` matched `SLC17A7`
   case-insensitively and silently. Require a **1:1** homologue and skip the gene when the mapping is
   1:many — paralogous families are where an orthologue summary is most wrong and most plausible.
3. **`gene_descriptor` renders the provenance in the text**, e.g.
   `"Pvalb. parvalbumin. Human orthologue PVALB: <summary>"`, so both the frozen encoder and any human
   reading the descriptors see it. Never as though it were the mouse gene's own summary.
4. **T10's E1 reports both arms** (`gene_summary_fallback` off / on), because importing human gene
   descriptions into a mouse model's text channel changes what the open-vocabulary claim is a claim
   *about*. Same discipline as E1's existing two arms for `r_g = 0` vs `psi(t_g)`.

Gated by `Config.gene_summary_fallback ∈ {"none", "ortholog"}`. Never overwrites a native summary.

**IMPLEMENTED at T06, default `"ortholog"`** (approved: 87% bare names is too thin for the
open-vocabulary claim). All four constraints hold, with one deviation from the proposal above:

| proposed | built | why |
|---|---|---|
| one column, `summary_taxid` | **three**: `summary_source` (`native`/`ortholog`/null), `summary_source_taxid`, `summary_source_symbol` | the descriptor label needs the orthologue's *symbol* — `"Human orthologue SLC17A7: …"` — and a taxid cannot supply it. `summary_source` is also what a diagnostic filters on, and `== "native"` reads better than `== "10090"` |
| default `"none"` until T10 measures both arms | default `"ortholog"` | E1's native-only arm is a **filter on `summary_source`**, not a second build, so both arms are measurable from one table and the default costs nothing |

Mechanics: `homologene` is requested on the primary query, so no extra round trip per gene; the 1:1
requirement is enforced **before** the orthologue query (1:many genes are never fetched, let alone
scored); the orthologue query is `scopes="entrezgene", species=human` and its `taxid` is verified the
same way the primary query's is, because an unverified third species would be labelled as human in
the descriptor text. `_read_gene_meta_table` back-fills the three columns as `native` for tables
written before them — the one migration that *is* derivable, unlike the species columns.

**The split cannot be measured in this container** (mygene.info 403, C14). What is measured:

| | |
|---|---|
| `resources/gene_meta.parquet` as committed (pre-fallback) | native **148**, ortholog **0**, none **990** |
| the four-case fallback path, on a fake client | native kept, 1:1 borrowed and labelled, 1:many skipped, no-orthologue left bare (`tests/test_text.py`, 4 tests) |
| expected after the rerun | unknown; bounded below by 148 and above by 1138. The user-reported human rate on this panel is ~93%, so the *plausible* range is high, but the binding quantity is 1:1 HomoloGene coverage for these 1138 symbols, which nothing offline predicts. **Do not quote a number until the run prints one.** |

One run, and it prints the split directly:

```
pip install -e ".[extra]"
python scripts/build_gene_meta.py --species mouse --symbols-from resources/mouse_panels_symbols.txt
#   with summary        N/1138
#     native            148   ortholog M   none K
```

*The consequence to decide with your eyes open.* Human coverage on this panel was **~93%** against
mouse's 13.4%, so with the fallback on roughly **85% of descriptors would carry human text**. The
open-vocabulary claim would then be substantially a claim about *human* gene summaries transferred to a
mouse model — which may well be the right scientific choice (orthologous function is largely conserved,
and it is what a human reader would do), but it is a different claim from the one the design states,
and it is why constraint 4 (E1 reports both arms) is not optional. It also means coverage must always
be quoted split by `summary_taxid`; a single "N/1138 have summaries" would hide the whole issue.

### C14. `resources/gene_meta.parquet` is described as "shipped" but nothing can build it offline — **the table now EXISTS; E1 unblocked, the fallback's numbers still owed** (raised in T02, escalated at T06)
**Resolved as far as this container can take it.** The table was built on a networked machine and
committed at `6f3cdfa`: 1138 mouse symbols, `species_resolved` 10090 uniform, `{'ENSMUSG': 1137,
'None': 1}` by prefix, accepted by `load_gene_meta(..., species="mouse")`. E1 can run on real gene
text. Two things remain, both needing the same one networked run: the table predates
`Config.gene_summary_fallback`, so 990 of its descriptors are still bare names (B20), and the
`native/ortholog/none` split is therefore still unmeasured. The rest of this entry is the record of
why it could not be done here.

**Escalated at T06, with the measurement that makes it blocking.** Zero-shot decoding of
never-trained genes measures **r = −0.368** on the synthetic fixture (B18) — the open-vocabulary claim
is the paper's headline novelty and it currently has *no* positive evidence, because the fixture's
gene names are arbitrary strings and T02 measured their text/co-expression Spearman at +0.0055. There
is nothing wrong with the code: with no text signal in, there is no transfer out. **The claim is
unevidenced until this table exists.**

**It cannot be built in this container, and that is a network-policy fact, not a code problem.**
Measured: the agent proxy records `connect_rejected — gateway answered 403 to CONNECT` for
`mygene.info:443`, and the same 403 applies to `rest.ensembl.org`, `eutils.ncbi.nlm.nih.gov`,
`www.ncbi.nlm.nih.gov`, `api.genenames.org` and `rest.uniprot.org`. No gene-annotation host is
reachable, so no amount of work here produces a real table. Reported rather than worked around; the
one thing that must **not** happen is committing an *offline* (symbol-only) table to
`Config.gene_meta_path`, because `load_gene_meta` would then succeed and C14 would look closed while
every descriptor is still a bare symbol.

**What T06 did do, so that whoever has network access has nothing left to decide:**

* committed **`resources/starmap_panel_symbols.txt`** — the 28 real symbols of the STARmap Wang2018
  3-D panel in `data/starmap/`, which is the real panel this repository has locally and the protocol
  the competing method publishes against;
* fixed a defect in `scripts/build_gene_meta.py` that the file exposed: `read_symbols` did not skip
  blank lines or `#` comments, so a symbol list with a provenance header had its header looked up as
  gene symbols (measured: 38 "symbols" from a 28-gene file);
* verified the offline path degrades loudly — two `GeneMetaUnavailableWarning`s and a printed
  `0/28 symbols carry metadata`.

**The command, to be run on a machine whose policy allows mygene.info:**

```
pip install -e ".[extra]"          # mygene lives in the `extra` group and is not installed by default
python scripts/build_gene_meta.py --symbols-from resources/starmap_panel_symbols.txt
git add resources/gene_meta.parquet && git commit
```

Then T10's capability experiment **E1** can run on real text, in both arms
(`forward_zero_shot(use_distill=False)` and `True`), and B18's threshold is what it reports against.
A wider panel is better than this one — 28 genes is a thin test of open vocabulary — so if a larger
real panel is available, pass its symbols too.
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

### C16. GATE 2's 0° arm is one section, and it is the best-supported one — **RESOLVED (decided 2026-08-15; `specs/04` amended)** (raised in T04)

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

**specs/04's escalation was run in full, and it is exhausted.**

| Escalation step | Result |
|---|---|
| 1. `n_plane_orientations` 4 → 8, criterion unchanged | **0.8867** — still below 0.90; the intervention moved the gate number by **+0.0009** |
| 2. Augmentation reaches coords / planes / retrieval / GRF | **Verified by mutation.** Leaving each channel un-rotated in turn changes the result: coords 0.0117 per field feature, GRF 1.121 per noise channel, retrieval 40.3 of K = 32 neighbours, plane normals 0.890. All four are wired |
| 2b. "A full forward pass is equivariant" | **0.0078**, i.e. **0.78 %** of the target spread, across 16 random poses with the rotation bound to the field. Cannot be 0 by construction (B5); under 1 % is what a working augmentation looks like |
| 3. Steerable backbone | **Not applied.** A design decision for the spec's owner |

**Two further defects in the measurement, both found while running the escalation.**

*The nine coronal arms do not cluster* (the test that could have rejected the depth-mix
account): 0.2912 / 0.4234 / 0.4364 / 0.4280 / 0.4567 / 0.4532 / 0.4625 / 0.4715 / 0.3642, spread
**0.180**. But the shape matters as much as the range — the *interior* seven span 0.4234–0.4715
(spread 0.0481, just inside the 0.05 that would have counted as tight), and the whole spread is the
two edge sections. GATE 2's central arm is 0.4567 against a nine-arm mean of 0.4208: the
single-plane baseline **flatters the denominator by 8.5 %**, and against the mean the worst oblique
angle reads **0.9547**.

*The criterion cannot resolve the shortfall at this `n`.* Re-drawing the equal-`n` evaluation sets
12 times with the probe untouched gives a ratio of **0.8971 ± 0.0168**, range 0.8718–0.9248, with
**6 of 12 draws below 0.90**. The shortfall being judged is 0.0029 against a draw-to-draw σ of
0.0168. `n = 1011` is set by the 90° strip, which is *every cell it has* rather than a subsample, so
this is a property of the fixture's slab thickness, not of the seed.

The residual U among the oblique angles (0.021 after stratifying by section and a 6×6 in-plane grid;
an in-plane distance-to-boundary stratification was tried first and rejected, moving it by 0.0008)
is the same size as the per-angle draw noise (σ up to 0.0075), so it is not evidence of a
directional mechanism either.

*Decision (2026-08-15): amend the criterion, and amend `specs/04` rather than leaving it at report
level.* **G2.1 is now stated on two depth-matched constructions, both required:**

- **G2.1a — the gate.** `min_angle R²_fixed ≥ 0.90 × mean_over_sections R²_fixed(0° at that section)`.
  Both arms depth-representative. Measured **0.9547**.
- **G2.1b — independent check.** The same ratio with both arms restricted to the interior sections,
  re-deriving the common `n` (785). Measured **0.9795** — *higher* than G2.1a, on a construction that
  drops the mechanism instead of averaging over it.

The reasoning is in `specs/04` under "Why G2.1 is stated this way", including the point that an
oblique strip **necessarily** samples the edge sections while a single interior coronal plane never
does — geometry, not sampling. The two superseded constructions are kept there and in
`reports/gate2.md` with their values (0.941 per-set; **0.886** single-central-plane, which failed),
so the record shows what the criterion moved from.

**What made the amendment admissible rather than convenient**, in order:

1. **The escalation was run first and came back null.** `n_plane_orientations` 4 → 8 moved the
   failing number by **+0.00086**; all four rotation channels were verified wired by mutation; the
   full forward pass is equivariant to **0.78 %** of the target spread. Both mechanisms the gate
   exists to catch were excluded by measurement before the contract was touched.
2. **The pre-registered test could have rejected it.** The nine coronal arms were required to
   *spread* for the account to hold; a spread under 0.05 would have left 0.886 standing. Measured
   **0.180** overall — and, decisively, **0.0481 interior-only**, just under that same 0.05. So the
   mechanism is **edge contamination specifically**, not general depth heterogeneity, which is what
   makes G2.1b a meaningful independent check rather than a restatement.
3. **The residual U was explained by the same mechanism.** With edges dropped from both arms the
   angle profile flattens to 0.4517 / 0.4436 / 0.4473 / 0.4396 / 0.4470 / 0.4660 — span 0.026, no
   mid-sweep minimum. The angles that looked worst were the ones drawing the largest edge share.

**`n_plane_orientations` reverted to 4** and the reason recorded in its `Config` docstring: +0.00086
for 2× the feature-plane memory. **G2.1h** (augmentation completeness, by mutation) and **G2.1i**
(draw-noise floor) are now **permanent criteria** in `specs/04` — they are what made the shortfall
diagnosable, and without the noise floor the 0.021 residual would have been uninterpretable.

**Carried forward as open risk R3**: the edge sections themselves reconstruct at 0.2912 and 0.3642
against an interior mean of 0.4474. That is real-volume geometry, and it is now written into
`specs/09` §1 (generation queries planes at or beyond the outermost sections) and `specs/10` §4
(stratify headline metrics by distance to the boundary).

### C17. `specs/07` §2's `lambda` and `CE` terms are written in the wrong coordinates — **RESOLVED in T07 (implemented as stated below)** (raised in T07)

Two of `L_cross`'s four terms are unusable exactly as written, for the reason `specs/07` itself gives
two paragraphs earlier when it argues for matching decoder parameters directly ("scale-stable,
differentiable everywhere"). Both are implemented in the log/logit coordinates the rest of the loss
already lives in; neither changes what the loss asserts.

| Spec | Implemented | Why |
|---|---|---|
| `\|\| lambda_1 - lambda_2 \|\|^2` | `\|\| log lambda_1 - log lambda_2 \|\|^2` | An intensity is ~1e-5 cells/µm³. Its squared difference is ~1e-10, so at `w_cross = 0.3` the term contributes nothing at any plausible weight — it is not a weak term, it is an absent one. The same argument the spec makes for `log mu` |
| `CE(type_logits_1, softmax(type_logits_2))` | `KL(p_2 \|\| p_1)` | Identical gradients — they differ by the teacher's own entropy, and the teacher is detached — but the CE has a floor of `H(p_2)` (~1.5 nats on a 6-type panel), which would sit in `L_cross` for ever and make "`L_cross` falls by 60 %" arithmetically unreachable however well the branches agree |

### C18. `specs/07` leaves four constants and one clip region unspecified — **RESOLVED in T07 (fields added, defaults measured)** (raised in T07)

Each became a documented `Config` field (Convention 1) rather than a literal. Recorded here because
three of them are choices, not transcriptions:

* **`sefl_ramp_frac = 0.2`.** §5 says "ramp `w_cross`, `w_thick`, `w_prog` linearly to their
  configured values" without saying over what horizon. One warm-up length, so the terms reach full
  weight at 40 % of the run.
* **`sefl_genes_per_step = 64`.** §5 caps the block at "< 60 % wall-clock overhead" and §2/§4 compare
  decoder parameters and expression "on the shared genes"; on the full panel the block measured
  **+62 %** — the spec's own two requirements are in tension. Every SEFL term is a mean or a
  covariance over genes, all are fine from a subsample, and the subsample is redrawn every step. At
  64 genes the block measures **+34 %**.
* **`section_granularity = "single-cell"`.** §3 branches on single-cell versus binned/spot data and
  nothing in `specs/01`–`06` records which a dataset is.
* **`sefl_ema_teacher = True`.** The anti-collapse switch has to be *switchable* for §"Acceptance
  tests" to run the disabled arm and assert it fails.
* **`intersect`'s clip region.** §1 says "clipped to the bbox" but gives the signature
  `intersect(p1, p2)`, which has no bbox. Implemented as the intersection of the two planes'
  **windows**, which a `Plane` already carries — and which is the tighter, more meaningful region
  (a segment outside either section's own extent is not on either section). `random_plane_pair`
  builds windows that span the volume, so the two readings agree in the pipeline.
* **`random_plane_pair(..., thickness=)`.** §1's signature predates T05 making thickness part of a
  plane's identity (B13). Keyword-only and required rather than defaulted: a hand-set section
  thickness in the middle of `L_thick` is exactly the constant that must not be invented.

### B21. `test_cross_loss_decreases` has no baseline as written — **RESOLVED in T07 (measured on the run's own trajectory)** (raised in T07)

"500 training steps on the fixture reduce `L_cross` by ≥ 60 %" implicitly compares against an
untrained model. That comparison is empty: `TriplaneField` initialises its feature planes at a
standard deviation of 1e-2, so at step 0 the field is nearly constant, *both* poses return nearly the
same features, and `L_cross` measures **3.9e-9**. Training can only make that number go up.

The criterion is measured instead on the trajectory the optimiser actually sees — the first three
logged values of the `cross` term against the last three of the same run — which is the quantity the
sentence is about. The untrained value is reported beside it as the fact that makes the substitution
necessary, not as a baseline.

### C19. `L_cross` has nothing left to constrain in v25, and what it does constrain is T04's capacity — **OPEN, `w_cross` defaulted to 0, decision owed to the spec's owner** (raised in T07)

`specs/07` §2 builds `L_cross` on the premise that a section's **conditioning pathway depends on the
plane it is cut on**: "this loss only has to correct the conditioning pathway, which is a much easier
optimisation than making two independent stochastic processes agree". That premise is true of the
*previous* architectures — v20 and the competing method both build a section out of its flanking
sections, so two crossing planes genuinely disagree where they cross — and it is **false of v25**,
by v25's own design.

**Measured, not argued.** `CTFFlow`'s expression pathway conditions at *physical points in the data
frame*: retrieval, the GRF and the Fourier encoding are all data-frame channels (T04's contract), and
`generate` passes `(points, points)` — the identity pose — whatever plane it was handed. Two crossing
planes therefore emit **bitwise identical** expression along their intersection on an *untrained*
model with no consistency loss applied, asserted by
`test_generation_is_intersection_consistent_by_construction`. The continuous 3-D field supplies for
free the property `L_cross` was invented to train.

**What is left, and why constraining it is harmful.** The one plane-dependent channel remaining is
the augmentation **pose** — and T04 made the triplane pose-dependent *deliberately*, as the capacity
mechanism GATE 2 rests on (`test_rotation_equivariance` asserts the triplane channel is **not**
invariant, as a live-augmentation control). A two-branch loss can only compare poses, so minimising
it drives the field towards pose-invariance, which for a lookup table indexed by fixed axes means
*constant*. A constant field carries no anatomy, and the generated section goes uniform while the
reconstruction path — which decodes the **encoder**'s latent and never queries the field — still
looks healthy.

Four arms, 500 steps, `specs/07`'s own schedule, differing only in which SEFL terms are live:

| arm | reconstruction (nats/pair) | generated per-gene variance ÷ real | `L_cross` self-consistency |
|---|---|---|---|
| SEFL off | **1.738** | **0.711** | 0.0224 |
| `thick` + `prog` only | 2.082 | **1.331** | 0.0243 |
| + `w_cross = 0.3` | 2.024 | **0.065** | 0.0100 |
| + `w_cross = 0.3`, teacher off | 1.914 | 0.344 | 0.0131 |

`L_cross` falls **90 %** over the run it damages, so this is not a failure to optimise it. The damage
is attributable: with `w_cross = 0` the field survives and nothing collapses.

**Interim decision, and it is reversible.** `Config.w_cross` ships at **0** with the table above in
its docstring; `loss_cross` stays built and tested (T10's A7 and E5 both need it), and
`test_no_collapse_at_the_spec_w_cross` pins the failure as a strict xfail. Carried as open risk
**R6**.

**The two candidate fixes, for the spec's owner.**

1. **Redefine the branch difference as the plane's *evidence*, not its pose.** Branch *i* conditions
   with the retrieval evidence a section cut on plane *i* would actually have — its own flanking
   sections, its own gap-aware dropout — which is a real, plane-dependent pathway that a correct
   model can reconcile without giving up any capacity. This is the version of `L_cross` that would
   still have content in v25, and it is what E5's "intersection agreement vs the competing method"
   figure is really about. **A design change, not a tuning fix** — the same category as GATE 2's
   steerable backbone.
2. **Accept that v25 gets intersection consistency by construction and drop the loss**, reporting
   the by-construction result as the finding it is (it is a *stronger* claim than "we trained for
   it"), and keeping `loss_prog` / `loss_thick` as the SEFL terms that still have content.

My inclination is 2 with the by-construction test promoted into T10's E5, because 1 buys a
constraint the architecture already satisfies; but the choice changes what the paper's SEFL section
claims, so it is not mine to make silently.

### B22. `test_prog_conditioning` does not reproduce on this fixture, and the measured direction is the wrong one — **OPEN, recorded as a strict xfail** (raised in T07)

`specs/07` §4 asserts that matching *marginals* instead of `(cell type, region)` strata "would force
the model to hallucinate a homogeneous tissue", and the acceptance test asks for that to be measured:
"an unconditional variant of `L_prog` measurably homogenises the tissue ...; the conditional version
does not".

**Measured, the effect is absent.** From one trained model, 60 steps of Adam on nothing but
`L_prog`, one arm per variant, three seeds, comparing the between-region expression spread each arm
leaves behind:

| starting model | per-seed ratio (unconditional ÷ conditional) | mean |
|---|---|---|
| trained with SEFL at the shipped weights | 1.60 / 0.59 / 0.71 | **0.97** |
| trained without SEFL | 1.30 / 1.27 / 1.23 | **1.27** |

The claim predicts well under 1 in both rows. On the second the direction is consistently *wrong*;
on the first it is not consistent at all.

**What it is not.** It is not that the two planes have the same composition — their
`(type, region)` mixtures differ by a median total-variation distance of **0.51**, so the marginals
the unconditional variant is matching really are different ones. And it is not one unlucky starting
point: an untrained (warmed-field) model gives 2.06 / 0.53 / 0.70, the same absence of a direction.

**The likely cause is the experiment, not the claim.** The conditional loss carries one MMD, one
correlation and one module term *per stratum* — five to a dozen strata on this fixture — so at equal
learning rate it applies several times the gradient of the single-stratum unconditional variant, and
"same number of steps" is not "same amount of optimisation". A fair version would match the gradient
norms, or train both arms to the same value of their own loss, and neither is specified.

**Kept as written, and the conditioning stays in the loss.** The a priori argument for conditioning
is sound and is not in question — different planes sample different mixtures, and matching marginals
across them asks the model for something false. What this fixture fails to demonstrate is the *harm*
of dropping it. Recorded as a strict xfail carrying the numbers, so a corrected experiment fails
loudly rather than passing quietly.

### C20. `specs/08` §3 names `geomloss`, which would make a paper number depend on the machine — **RESOLVED in T08 (in-repo implementation, always)** (raised in T08)

`specs/08` §3 says "Entropic Sinkhorn divergence (via `geomloss`, blur = median NN distance in PC
space) ... Fallback: multi-bandwidth RBF MMD if `geomloss` is unavailable". Read literally that makes
the loss — and therefore the trained model, and therefore every number in the paper — a function of
whether an optional dependency happens to be installed, and `geomloss` is in `pyproject.toml`'s
`extra` group, i.e. present on some machines and not others. Two different divergences behind one
config is exactly the kind of silent dependency variation the dependency pins exist to prevent.

*Resolution (T08):* the divergence is **always** the in-repo one. T07 already implemented the
debiased entropic Sinkhorn (`losses/sefl.sinkhorn_divergence`, log-domain, plan detached by the
envelope theorem — the same thing `geomloss` computes and for the same reason), so there is no
missing capability to import. `Config.metric_distribution_kind` selects `"sinkhorn"` (default) or
`"mmd"`, the spec's named fallback, and the choice is a *config* setting rather than an accident of
the environment. `geomloss` stays in `extra` unused; if a real run ever needs its GPU kernels the
switch is one function and should be measured against the in-repo one first.

### C21. `loss_profile`'s signature has nowhere to put the labels its own third bullet needs — **RESOLVED in T08 (keyword-only, additive)** (raised in T08)

`specs/08` §2 states `loss_profile(x_gen, coords_gen, x_real, coords_real, vol, cfg)` and then asks
it for three things, the third being "**Per-type spatial histogram** agreement — this is where
cell-type localization is optimised". Cell types are not in the signature, and they are not derivable
from `x` (a per-cell expression vector is not a label).

*Resolution (T08):* the stated positional signature is kept exactly and gains two keyword-only
arguments, `types_gen` and `types_real`, both or neither. `types_real` is the real one-hot labels;
`types_gen` is the **intensity head's** per-type composition `lambda_c / sum_c lambda_c` at the same
points, which is what makes the term differentiable — a *drawn* label carries no gradient, so a
histogram of sampled marks could not optimise anything. Two smaller additive keywords land for the
same reason: `soft_depth_profile` and `soft_field_profile` take a `bounds`, because two profiles that
are going to be compared have to be binned on the same ruler and the spec's signatures have nowhere
to say so.

### C22. What "reconstruct at the hidden section's true plane" means for the layout — **RESOLVED in T08 (expression at the real positions; the layout enters through the intensity)** (raised in T08)

`specs/08` §4 says "Reconstruct at the hidden section's true plane and thickness, then compute
§1–§3", which could mean generating a whole section — layout sampled and all — or evaluating the
expression pathway at the section's own cells.

*Resolution (T08):* the second, and the reason is that the first cannot work. A sampled layout comes
out of a discrete point process (T05's Strauss/Potts sampler), so there is no gradient from any
statistic of the sampled positions back to the intensity head; and a freshly drawn point pattern
would put the sampler's draw noise into every one of §1-§3 on top of what is being measured. The
reconstruction therefore runs the generation path — GRF prior, flow, decoder — at the hidden
section's **own** cell positions, with that section excluded from retrieval, and compares expression.
The layout head is still constrained, differentiably, by the per-type spatial histogram of C21, which
reads `lambda_c` at those positions. The consequence to state plainly: T08's terms **do not** train
the point pattern, and nothing in this task claims they do.

### C23. Constants `specs/08` leaves open — **RESOLVED in T08 (`Config` fields, defaults documented)** (raised in T08)

Convention 1 forbids a magic number outside `Config`, and `specs/08` names several quantities without
values: the Huber's transition point, "top-k spatially variable genes" without a k, the number of
cells the divergence is estimated from, how long a hidden section stays hidden ("each epoch", against
a trainer counted in steps), and the guard on a 0/0 in an empty profile bin. Added as documented
fields: `metric_huber_delta`, `metric_marker_genes`, `metric_distribution_points`,
`metric_sinkhorn_blur_multiple`, `metric_distribution_kind`, `metric_dominance_ratio_warn`,
`loso_epoch_steps` and `metric_eps`. `metric_knn_k` (10, the vendored metric's own degree) and the
four `profile_*` / `loso_*` fields already existed from T01. The Sinkhorn iteration count is **not**
duplicated: `sefl_sinkhorn_iters` is a property of the solver, not of the caller, and both callers
share it.

### C24. `specs/08` compares a *model* against a *measurement* and does not say how — **RESOLVED in T08 (three answers, each measured; two of them are not the obvious one)** (raised in T08)

`specs/08` states every one of its statistics on "the generated section" and "the real section" as
if both were the same kind of object. They are not: the real section is one **draw**, and the model
is a **distribution**. Nothing in the spec says which of the model's two faces — its mean field or a
sample from it — enters each statistic, and it turns out that all three plausible answers are wrong
for at least one of the three families. Each of the following was implemented, trained and measured
on the fixture before the next was tried; the numbers are the reason the final shape is not the
simplest one.

| what enters the statistic | what happens | measured |
|---|---|---|
| the **mean field**, everywhere (log1p, both sides) | matching a tight mean cloud to a dispersed sample cloud can only be closed by distorting the mean | every metric the terms exist to improve got worse: marker-depth r **0.985 → 0.642**, Frobenius covariance **9.3 → 26.0**, Moran's MAE 0.319 → 0.355 |
| a **count draw** on a straight-through estimator, everywhere (log1p) | `log1p` is concave and counts are sparse: at a zero draw its slope is 1 while `dE[log1p(X)]/dm` is a fraction of that, so every step overstates what raising the mean buys | generated mean normalised count **4 → 761** against a real 6.8, the distribution term's own value rising **1.43 → 5.56** while it did it; and on a **linear** scale the same estimator instead buys autocorrelation by out-sampling the noise, **4 → 551** |
| **per family** (shipped) | see below | stable; ratio to reconstruction 0.375 median |

**The shipped answer, one line per family.**

* **Profiles** take the decoder's **mean**, on a **linear** library-normalised scale. A profile is a
  bin mean, and the bin mean of a mean field is an exactly unbiased estimate of the bin mean of a
  draw — no correction, no sampling noise in the gradient. (`specs/08` §1 suggests `log1p`; its
  actual requirement is that both sides share a scale and that the choice is documented, which this
  is. `log1p` is what makes the straight-through estimator biased, and the profiles are the terms
  the scale matters for.)
* **Autocorrelation** takes the mean field **plus the draw's analytic variance**, added to Moran's
  and Geary's denominators (and, for Geary, to its numerator). Sampling noise attenuates a *measured*
  Moran's I; an uncorrected mean field would be asked to be less autocorrelated than the tissue.
  `draw_mean_variance` supplies the closed form for both decoders, and `morans_i` / `gearys_c` /
  `loss_autocorr` gain a keyword-only `noise_var`. Moran's I and Geary's C are invariant to a
  per-gene affine rescaling, so the linear scale costs them nothing.
* **Distribution** takes a **reparameterised surrogate**, `clamp(mean + sqrt(variance) * eps, 0)` at
  a seeded `eps`, in the `log1p` PCA basis it is stated in. A cloud comparison needs dispersed points
  on both sides, and this one has the draw's first two moments with a pathwise gradient.

**Four other things `specs/08` does not say, all decided the same way — by the failure they prevent.**

1. **Both sides are divided by the *real* cell's library size**, not by their own row sums. With each
   row divided by its own generated sum, every gene shares one denominator and the model can raise
   the whole panel's autocorrelation at once by giving that denominator spatial structure — one gene
   takes the library and every other gene inherits `-log(total)`. Measured: the autocorrelation term
   fell 0.33 → 0.16 while the largest normalised value pinned at `log1p(reference)` and the
   parameter gradient went 6 → 1.6e8.
2. **The generated side uses the real cell's size factor**, not `SizeFactorHead`'s. Every statistic
   is library-normalised, so library size is not what these terms measure (it is T06's and T09's);
   leaving the head in the graph left a nearly flat direction with a large derivative through an
   `exp`, and the gradient there reached 53 while every other parameter sat below 5.
3. **The per-type histogram reads `lambda_c` at uniform slab points, not at the cells.** At the cells
   the term has a second minimiser — a field that classifies each individual cell correctly averages,
   over any bin, to exactly that bin's label fractions — so it pays nothing for unbounded confidence:
   `max_c lambda_c / sum_c lambda_c` 0.62 → 0.9975, and the anatomical field the intensity head shares
   with the expression path went with it.
4. **Markers are chosen among genes detected in ≥ 5 % of real cells** (`metric_marker_min_detection`).
   Ranking the whole panel by Moran's I selects sparse genes whose few non-zero cells happen to be
   neighbours; the profile term divides each marker by its own mean, and a mean of order 1e-3 took
   the term from 1.9 to 1766 in forty steps with every model parameter standing still.

None of these is a tuning choice; each is a term that had a minimiser other than the one it was
written for, and each is recorded in the docstring of the function that carries the fix.

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

### C25. `text_emb_mode` was declared at T01 and consumed by nothing — **RESOLVED in T09 (implemented in the embedding)** (raised in T09)

`Config.text_emb_mode ∈ {medcpt, lookup}` has existed since T01, `specs/09` §3 lists `text_emb` as
one of the four coordinate-descended gates, and `specs/10`'s ablation **A3** is
`text_emb=lookup-only`. Nothing read the field: `TextGroundedEmbedding.forward` always applied the
text channel. The selector would therefore have scored four identical candidates and reported a
decision it had not made.

*Resolution (T09):* implemented once, in `TextGroundedEmbedding._text_channel` — `"lookup"` zeroes
the projected text vector on **both** the seen path and the zero-shot path, and pins the residual
gate at 1 (there is no text prior to anneal from, so a warm-up would leave the embedding exactly
zero for the first 30 % of training). T10's A3 uses the same switch.
`test_lookup_only_text_mode_drops_the_text_channel` asserts it.

### C26. T09 §3 scores on "the six target metrics", which `eval/metrics.py` (T10) has not vendored yet — **RESOLVED in T09 (same names, T08's kernels, T10 re-scores)** (raised in T09)

`specs/09` §3 scores every candidate on the six target metrics; `specs/10` §1 requires those to be
**vendored verbatim** from `bench3/evaluate_paper.py` with a pinned content hash, and that module is
T10's. Vendoring it early would pin the scoreboard in two places.

*Resolution (T09):* `train/select.py` computes the six under **T10's names** with T08's own kernels
(`morans_i`, `gearys_c`, `soft_field_profile`, `soft_depth_profile`) plus a PCA-space kNN mixing
score in place of the UMAP one — a linear stand-in chosen so the selector stays reproducible from a
seed alone. The substitution is stated in the module docstring and in every report the selector
writes. **T10 re-scores the selected config with the vendored code**; if the two disagree about a
gate, that disagreement is a T10 finding and the selector's scorer is a one-line swap
(`Scorer` is a protocol).

### C27. E5's expression criterion is above the achievable ceiling — **RECORDED in T09 (strict xfail, ceiling measured)** (raised in T09)

`specs/09`'s `test_oblique_intersection_agreement` requires cell-type concordance > 0.8 **and
expression correlation > 0.85** between the cells two crossing oblique sections place at the same
point. Both sections emit *draws*, and their layouts are independent, so two matched cells carry
independent ZINB noise on the same mean: the highest correlation any model can reach is what two
draws of **one** plane reach under one realisation.

Measured on the fixture at a 600-step fit: ceiling **0.726**, oblique pair **0.724** — 99.7 % of the
achievable range — against the spec's 0.85. Concordance reaches **0.814** against a ceiling of
0.781, so *that* half of the criterion is achievable and is asserted absolutely.

*Resolution (T09), following the pattern T06 used for its covariance criterion (B16):* the headline
test asserts concordance absolutely and expression correlation **relative to the measured ceiling**
(≥ 0.95 × it), and the literal 0.85 is kept by name as a **strict xfail** so the day it becomes
reachable the suite says so. The spec's owner may prefer to restate the criterion in ceiling-relative
terms, which is what `specs/10`'s own "achievable ceiling" section already requires of every metric.

### C28. The `pi` and `theta` corrections are not orthogonal, and the fixture leaves them no headroom — **RESOLVED in T09 (alternating solve); the transfer result is a finding** (raised in T09)

Two things, both measured:

1. **Coupling.** `theta` enters the detection rate through `P_NB(0) = (theta/(theta+mu))^theta`, so a
   `pi` shift solved at the uncorrected `theta` is wrong by however much `theta` then moves. Solved
   independently the two corrections together made the per-gene detection MAE **worse**, 0.055 →
   0.230, while each matched its own target exactly. `_fold_statistics` now alternates the two solves
   twice, and targets the **mean–variance relation** (`Var = mean + phi mean^2` at the model's own
   mean) rather than the absolute variance — matching the absolute variance asks `theta` to repair a
   wrong *mean*, and on the dense genes it drove `theta` to 0.007 and collapsed their detection rate.
2. **Headroom.** On this fixture there is none. At a 600-step fit the model's own per-gene detection
   error is **0.0217** while the *real* per-gene rate varies by **0.040** between training sections,
   so a correction fitted on other sections imports more of that variation than it removes: cross-fold
   **0.0326**, against an oracle (fitted on the very section it is applied to, which no leakage-free
   procedure can reach) of **0.0191**.

*Resolution (T09):* the calibrators ship, `generate_section(..., calibration=None)` does **not** apply
them by default, and the numbers above are recorded in `progress/t09_inference_and_calibration.md`.
`test_detection_calibration_does_not_transfer_between_sections` asserts the *diagnosis* — that the
real between-section variation still exceeds the model's own error — so the day a dataset gives the
correction something to do, the test fails and the measurement is re-run. T10 decides on real data.

### C29. `select_config(vol, base_cfg)` has nowhere to put a seed, a scorer or the embeddings — **RESOLVED in T09 (additive keywords)** (raised in T09)

The spec fixes the signature at `select_config(vol: TrainingVolume, base_cfg: Config) -> Config`, but
the default scorer fits a model per candidate and a model needs T02's `EntityEmbeddings`, which come
from a cache this module has no business knowing about; Convention 3 also requires an explicit seed.

*Resolution (T09):* keyword-only additions — `seed` (defaulting to `base_cfg.seed`, so the run is
reproducible from the config alone rather than from an implicit RNG), `embeddings` (a
`Config -> EntityEmbeddings` **factory**, because every candidate is a fresh fit and reusing one
object would leak the first candidate's training into the rest), `scorer` (the seam the acceptance
tests need), `dataset` and `report_path`. `run_selection` is the same call returning the whole score
table instead of only the winner; `select_config` is `run_selection(...).config`.

### C30. Nothing applies a `LengthscaleCalibration` — the spec names the return value but not the writer — **RESOLVED in T09 (`apply_lengthscale`, spec amended)** (raised in T09)

`specs/09` §2 fixes `calibrate_lengthscale(model, vol, cfg) -> LengthscaleCalibration` and says what
the object carries (`ell`, `status`, achieved and target), but never says who writes the calibrated
`ell` back into the `Config` that generation reads. `generate_section`'s `calibration=` argument is
the **detection** calibration; `ell` reaches the prior only as `cfg.ell_xy` / `cfg.ell_z` through
`_using_field`. So as specified and as built, a calibration is measured, reported, and then not
used: generation runs at the config's own `ell_z` (100 um by default).

Leaving it unwired is the safe half of the bug — GATE 2's oblique parity was measured at the
config's 100 um and not at the fixture's artefactual 25 um. But a calibrator whose result nothing
consumes cannot discharge R1.

*Resolution (accepted by the spec owner, `specs/09` §2 amended to name the writer):*
`apply_lengthscale(cfg, calibration) -> Config` is the only sanctioned way a calibrated `ell`
reaches generation. It applies **only a `"converged"` axis**; `target_unreachable` and `boundary`
are dropped with a `CalibrationNotAppliedWarning` naming the achieved and target values, and the
config's own value stands — so a tie-break on a flat objective can never become the shipped
length-scale. The two axes are decided separately on `status` and `ell_z_status`, because an
in-plane result that converged is not made worthless by a stack too short to constrain `ell_z`
(R1); the cost, a half-applied anisotropy ratio, is why the dropped axis warns rather than passing
silently. Three acceptance tests pin it, including the round trip through to the field the
generator builds.

### C31. `specs/05` says "rejection sampling", and rejection sampling as specified is biased — **RESOLVED 2026-08-24 (grid-multinomial default; the rejection sampler retained and asserted wrong)** (raised at T10, fixed here)

`specs/05` and `design/v23_design.md` both name **rejection sampling** as how the layout's positions
are drawn from `sum_c lambda_c`, and neither says how the envelope is obtained. What was built takes
the maximum of the total intensity over a `Config.layout_n_mc` uniform sample of the mid-plane, times
`Config.layout_envelope_slack`, and tests `u < lambda / envelope` **without clamping the ratio**.

Both halves of that are wrong, and `reports/r11_envelope.md` measured them:

* The envelope is a **sampled maximum**, not a bound — 140-853x spread across sections on the pilot
  checkpoint, and reproducible with no fit at all (`test_the_envelope_is_a_sampled_maximum` rebuilds
  it at eight seeds on a closed-form intensity and gets a 9.3x spread, with the smallest below half
  the true supremum).
* Because the ratio is unclamped, every point where `lambda > envelope` is accepted with certainty,
  so the realised draw is from **`min(lambda, envelope)`**. The peaks are flattened: the pattern is
  wrong in *shape*, not merely short of points.

The obvious repair — an analytic envelope from the head's own parameterisation — was **costed and
rejected on measurement** (`reports/r11_fix_options.md` option A). An MLP has no closed-form
supremum, so the only analytic route is a Lipschitz bound; the trunk's spectral norms give
`L = 74.69`, a bound 7-82x above the true supremum, and a 7-82x looser envelope is 7-82x worse
acceptance applied to a section already accepting 0.12 %. That option converts a bias into universal
starvation.

*Resolution (option D, implemented):* the positions are drawn by a **grid-multinomial** sampler.
`sum_c lambda_c` is evaluated once per cell of a grid over the mid-plane window
(`Config.layout_grid_cells`), a cell is drawn with probability proportional to it, and the point is
jittered uniformly inside the cell. There is no envelope, no acceptance ratio and no proposal budget
on the intensity; the only approximation is the grid's own resolution, which is a midpoint rule with
an `O(h^2)` error and a convergence check rather than a sampled maximum, and which gives every cell a
**closed-form expected count**. The interaction (Strauss thinning) rides on the same sequential test
and the same `layout_max_proposal_factor` budget as before, so `ProposalBudgetWarning` now means what
it always claimed: the intensity and `r0` describe incompatible tissue.

`Config.layout_sampler` selects between `"grid"` (the new default) and `"rejection"`. The rejection
sampler is **kept**, so the two can be compared on one fit and so the biased numbers stay
reproducible, and it is asserted *wrong* in `tests/test_layout_sampler.py` as the negative control.

Two consequences that are not resolved by this and are owed elsewhere. **T05's acceptance tests and
T09's `layout_mode` gate were computed on the biased sampler**; T05's are re-measured in
`progress/t05_layout_head.md` and still pass, T09's are not. And a correct sampler removes a bias, it
does not supply a better intensity field, so **R11's finding stands** until the three modes are
re-measured on tier-1 STARmap.

---

### C32. GATE 2's `G2.1h-c` reads 63.97 against a recorded 40.28, and the recorded value is not reproducible — **OPEN** (raised 2026-08-25)

`reports/gate2.md` records `G2.1h-c` — the mutation check that querying retrieval in the model frame
instead of the data frame perturbs the neighbour sets — at **40.28**. Regenerating the report gives
**63.97**. The criterion passes either way (`> 0`) and every other GATE 2 number agrees to six
significant figures, so nothing about the gate's verdict turns on it. It is raised because a value
that moved 40 to 64 with no identified cause should not sit in a report unexplained.

Two explanations were offered and **both are refuted**:

* *The host.* The recorded value came from macOS / arm / Python 3.12 and the regeneration from
  Linux / x86 / Python 3.11, so a cross-platform float difference in a discrete count looked
  plausible. The spec owner regenerated on the original macOS platform and got 63.9697, identical.
  False.
* *A change in this repository.* The measurement was isolated (reproduced without a trained probe,
  with each arm's imported module paths printed so a mis-resolved import would show). The **oldest
  commit in the repository**, `70076ad`, already gives 63.9697 — which bounds the transition outside
  the 52-commit history rather than inside it, so a bisect cannot locate it. `reports/gate2.md`
  carries a generation date of 2026-08-16; the oldest commit is dated 2026-08-20. The state that
  produced 40.28 was never committed.

**What the diagnostic is.** It is essentially a function of how far the mutation displaces the query
points, saturating almost at once. On rotations about z: 0° -> 0.00, 0.5° -> 9.88, 1° -> 19.28,
2° -> 35.68, 5° -> 57.89, 10° -> 62.43, 20° -> 63.58, 45° -> 63.93, against a ceiling of 2K = 64.
The gate's own rotation is **179.68°**, so 63.97 is the correct value for the rotation the gate
draws, and 40.28 corresponds to an effective displacement of about 2°. The value is invariant to
`retrieval_z_window` (0.05-3.0), `retrieval_z_window_gap_factor`, `retrieval_candidates_per_section`,
`rotation_bias` and `rotation_bias_max_tilt_deg`; and both arms return **zero PAD slots**, because
C1c's per-query gap term makes the pool non-empty by construction — which closes the other route to
a smaller symmetric difference.

*Open, with two untestable candidates, both in pre-import code:* a mutation pose built differently
(a ~2° effective displacement rather than ~180°), or the **pre-C1c** retrieval window, which had no
gap term and so could return a truncated pool. Note that `PROGRESS.md` records the C1c amendment as
leaving "gate numbers unchanged"; if that claim was made without regenerating the report, this
diagnostic is a counter-example to it, and that is the first thing to check if the pre-amendment
code is recoverable from outside this repository.

*Separately, and independent of the discrepancy:* **`G2.1h-c` saturates at 2K for any real
rotation**, so as a graded diagnostic it carries almost no information — it is a binary "the channel
is wired" check wearing a continuous number. Proposed: keep the criterion (it does catch an unwired
channel) but stop reporting its magnitude as if it measured something, or restate it at a fixed
small probe rotation where it is actually graded.

---

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
