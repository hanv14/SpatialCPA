# Open questions and suspected errors in `specs/`

Raised after reading all of `specs/`, `design/v23_design.md`, `design/v23_sectioning_equivariance.md`,
and cross-checking against `reference/learn_spatialcpav20.py` and `benchmark-pbya-v3/`.
Nothing here is a reason to delay T01 except the items in §A, which change interfaces.

Status key: **OPEN** (needs a decision), **PROPOSED** (I will do the stated thing unless told
otherwise), **INFO** (recorded, no action).

---

## A. Contradictions between task files — resolve before the affected task

### A1. `Volume` vs `TrainingVolume` / `HeldOutSections` — **OPEN**
T01 specifies `split_holdout(vol, mode, fold) -> tuple[Volume, list[Section]]` and
`loso_folds(vol) -> Iterator[tuple[Volume, Section]]`. T08 and T09 require the training portion to
be a distinct type `TrainingVolume`, the holdout to be `HeldOutSections`, and passing the wrong one
to be a **`TypeError`** (`test_metric_aware_rejects_heldout`). A `typing.NewType` gives a mypy error,
not a runtime `TypeError`.

*Proposal:* make them real classes in `data/schema.py` — `TrainingVolume(Volume)` and
`HeldOutSections` (a container, deliberately *not* a `Volume`) — have `split_holdout` return
`tuple[TrainingVolume, HeldOutSections]` from the start, and have the loss/calibration entrypoints
runtime-check the type. Costs nothing in T01 and makes T08/T09's guarantee real.

### A2. Per-gene-module length-scale calibration is not implementable as written — **OPEN**
T09 §2 calibrates `ell` "per gene-module (Leiden clusters of gene embedding space, ~10 modules)".
But `ell` parameterises the **latent** GRF (`d_h = 64` channels queried at cell positions), and gene
modules only exist downstream of the decoder. One `ell` per gene module cannot be expressed against
a single latent field: changing `ell` for module *m* would require the field to know which latent
channels that module reads from, which is not a property the decoder is constrained to have.

*Proposal:* calibrate one global `ell = (ℓx, ℓy, ℓz)` (this is also what GATE 1's monotonicity
criterion is defined on), and report per-module Moran's I agreement as a **diagnostic**. If
per-module control turns out to be needed, the cheap version is to partition the `d_h` latent
channels into groups with their own `ell` and add a loss tying gene modules to channel groups —
that is a design change and should be decided explicitly, not improvised at T09. To keep the door
open, I will give `GaussianRandomField` an optional per-channel-group `ell` in T03.

### A3. T10's metric provenance is wrong — **OPEN**
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

*Proposal:* port the six from `bench3/evaluate_paper.py`, assert equivalence against it in
`test_metrics_match_reference_after_fixes`, and keep T10's bug note as a caveat about v20's
*internal* scoring (which is what its own tuning was driven by) rather than about the benchmark.

### A4. `Config` is missing fields that later tasks depend on — **PROPOSED**
Convention 1 forbids constants outside `Config`, but the task files write several inline. Not in the
T01 field list and needed: `field_dim` (`d_f`, used everywhere from T04 on), `ctx_dim` (`d_ctx`,
retrieval output), retrieval attention head count, `expr_pca_dim` (neighbour tokens, Sinkhorn basis),
`metric_knn_k` (the k in Moran's/Geary's graphs — bench3 uses 10), `potts_knn_k` (T05 hardcodes 8),
`layout_max_proposal_factor` (T05's `20*N`) and `layout_envelope_slack` (its `1.1`),
`swd_polish_steps` (T05 hybrid's 200), profile `n_bins` / `field_grid` (T08's 24×24) and
`profile_sigma_frac` (0.75), `loso_every_k_steps` (4) and `loso_max_cells` (4000),
`n_uncertainty_samples` (T09's M=8), `sefl_min_stratum_cells` (T07's 20), `holdout_consecutive_k`
(T01 calls it "configurable" but gives no field), `bisection_max_iter` / `bisection_grid_size` (T09).

I will add these in T01 with the spec's values as defaults and document each. Flagging because it
makes T01's `Config` noticeably larger than the block printed in the spec.

### A5. `Config.validate()` cannot check the example it is given — **PROPOSED**
T01 asks `validate()` to raise on "`fourier_bands_z > 4` with fewer than 8 training sections", but
`Config` is standalone and has no volume. *Proposal:* `validate()` for self-consistency (set
membership, positivity, mutually exclusive gates) plus `validate_against_volume(vol)` for the
data-dependent checks, called by `load_volume`.

### A6. Nothing specifies building the v20 cross-mix — **OPEN**
`expr_mode ∈ {zinb-flow, cross-mix, auto-blend}` is a `Config` gate (T01), the no-regression
guarantee depends on `cross-mix` (T09 §3, `test_selector_can_recover_v20_config`), and T09's
uncertainty-gated anchoring blends "via the v20 Bernoulli cross-mix" (design §5). No task file
specifies implementing it, and the coverage matrix mentions it only as reference material.
*Proposal:* implement it in T06 alongside the decoder (it shares the count-preserving path), ~40
lines, ported from v20 with its behaviour pinned by a test.

---

## B. Acceptance tests that will fail for reasons unrelated to the model

### B1. Bitwise equality across two plane pathways (G1.2, and `test_noise_identical_along_intersection` in T07) — **PROPOSED**
`torch.equal` on the GRF sampled through plane 1's and plane 2's coordinate constructions only holds
if both produce **bit-identical** `xyz`. `origin + u*e1 + v*e2` with different orthonormal bases
rounds differently in float32, so this can fail while the field is perfectly correct — exactly the
misdiagnosis the spec warns about elsewhere.
*Proposal:* route every plane→points construction through one canonical function, assert bitwise
equality of the *field* given identical `xyz` (this is the real property: purity), and assert
`allclose(atol=1e-6)` between the two plane constructions. Both assertions in the test, documented.

### B2. G1.1's monotone-in-M requirement is stochastic — **PROPOSED**
"Error decreases monotonically as M goes 1024 → 2048 → 4096" is a single-draw statement about a
random estimator. *Proposal:* average the covariance MAE over ≥ 5 seeds per M and assert the trend
on the means (plus `err(4096) < err(1024)` outright).

### B3. `test_grf_channels_independent` tolerance is at the noise floor — **PROPOSED**
Finite-M cross-channel correlation is O(1/√M) ≈ 0.016 at M = 4096, before point-sampling noise; the
spec's threshold is 0.02. *Proposal:* orthogonalise the `A` columns at construction (a QR on
`(M, d_h)`, which also stabilises the marginal-variance renormalisation the spec asks for) and keep
the 0.02 threshold; if that is unwanted, relax to 0.03.

### B4. `test_relative_position_only` (T04) is false for the full model — **PROPOSED**
"Translating the whole volume by a constant leaves outputs unchanged" cannot hold end-to-end: the
GRF realisation is a function of absolute position (`ξ(p + t) ≠ ξ(p)` by construction — that is the
point of T03), and the triplane is bbox-relative only if the bbox is translated with the data.
*Proposal:* scope the test to the retrieval branch's neighbour encoding, which is where the
absolute-coordinate leakage the spec is guarding against would actually appear.

### B5. Rotation equivariance vs data-frame Fourier encoding (T04) — **PROPOSED**
The spec asks for both "encoding a fixed cell must be invariant to the augmentation rotation" and "a
full forward pass is equivariant: rotate inputs, inverse-rotate outputs, get the same result". These
are consistent only under a precise statement of which frame each quantity lives in.
*Proposal:* state the contract as `F(R·p | volume rotated by R) == F(p | volume unrotated)` with the
data-frame encoding carried through `RotationContext`, write it in the module docstring, and make
the two tests assert the two halves of it.

### B6. Hard-core radius fights the pair-correlation test (T05) — **OPEN**
`r0` = 5th percentile of real nearest-neighbour distances, and `test_hardcore_respected` forbids any
generated pair closer than `r0`. By construction 5% of *real* pairs are closer than `r0`, so the
generated layout is strictly more regular than the tissue it is imitating, which pushes against
`test_pcf_matches_real` (max |Δg(r)| < 0.15 from `r0` up).
*Proposal:* set `r0` to the 1st percentile (or the observed minimum) and let the fitted `gamma`/`R`
carry the soft repulsion; keep the 5th percentile as a `Config`-selectable alternative and record
which was used.

### B7. The layout sampler is a conditional Strauss, not a Strauss process (T05) — **PROPOSED**
Step 1 draws `N ~ Poisson(N_expected)` and step 3 thins; conditioning on `N` and then thinning is
not the same object as a Strauss process (which produces fewer points than its Poisson envelope).
The `20*N`-proposals escape hatch then silently breaks `test_expected_count_matches`.
*Proposal:* implement and document it as a **conditional-on-N** Strauss sampler (which is what makes
the count-from-intensity claim work), make the proposal cap a `Config` field, and raise rather than
warn under test.

### B8. The Matérn RFF parametrisation will not match `scipy`'s `Matern(length_scale=ell)` — **PROPOSED**
`omega = (z/√g)/ell` with `g ~ Gamma(ν, 1/ν)` yields a Matérn *shape* but with an effective
length-scale off by a √(2ν)-type constant relative to `scipy`'s parametrisation, so G1.1's
MAE < 0.03 could fail purely on convention. The spec anticipates this ("verify this empirically
rather than trusting the derivation"). *Proposal:* fix the constant numerically at construction,
unit-test the realised covariance against `scipy` at several `ν`, and write the convention in the
docstring.

### B9. G1.4's throughput target needs chunking to be possible at all — **PROPOSED**
10⁶ points at M = 4096 is a 16 GB float32 feature matrix if materialised. *Proposal:* chunked
evaluation (fixed chunk size in `Config`), and if 5 s on this CPU proves out of reach after
chunking, report the measured number rather than quietly loosening the gate.

---

## C. Under-specified — I will pick the stated option unless told otherwise

### C1. GATE 2's evaluation set is undefined — **OPEN**
"Reconstruct held-in cells from planes at 0°…90°" — but real cells only exist on the sectioning
planes, so an oblique plane passes through very few of them, and R² at 90° would be computed on a
different (and much smaller) cell set than at 0°. That makes the ratio `R²(θ)/R²(0°)` partly an
artefact of sample size. *Proposal:* evaluation set = cells within `thickness/2` of the query plane,
pooled across all training sections, with `n` reported per angle and a minimum-`n` guard; if `n` at
large angles is too small on the fixture, thicken the fixture's slabs rather than changing the gate.

### C2. `KL(ZINB₁ ‖ ZINB₂)` has no closed form (T07) — **PROPOSED**
Neither NB–NB nor ZINB–ZINB KL is closed-form (both need an infinite sum over counts). *Proposal:*
closed-form Bernoulli KL on `π`, plus an analytic surrogate on the NB component (KL between the
Gaussian approximations in `(log μ, log θ)`), documented as a surrogate; alternatively a fixed-sample
MC estimate. State the choice in the docstring and note it in the paper's methods.

### C3. MedCPT pooling instruction contradicts itself (T02) — **PROPOSED**
The spec says "mean-pool the last hidden state", then says MedCPT-Query-Encoder is trained with CLS
pooling and to use whichever the checkpoint specifies. *Proposal:* use the CLS/first-token
representation (what the MedCPT query encoder is trained for), make it a `Config` field with
mean-pooling as the alternative, and write the justification in a comment as the spec asks.

### C4. The fixture's ground-truth field has nowhere to live (T01) — **PROPOSED**
The spec suggests `vol.uns["gt_field"]`, but the `Volume` dataclass has no `uns`. *Proposal:* return
`(volume, gt_field)` as a tuple; keep `Volume` free of an untyped dict.

### C5. `Volume`'s derived fields need `field(init=False)` (T01) — **PROPOSED**
`median_spacing`, `median_nn_dist`, `bbox` are listed as ordinary fields but described as computed in
`__post_init__`. They will be `field(init=False)` so they cannot be passed in and go stale.

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

## D. In the design docs but missing from `specs/11_COVERAGE_MATRIX.md`

The matrix says an unmapped design component is an omission to be flagged. These are the ones I
found:

| Design location | Component | Where it should land |
|---|---|---|
| `v23_design.md` §7 Baselines | **v14 and v18** are listed as baselines; T10 wires only `run_v20` | T10 — or drop them explicitly and say why |
| `v23_design.md` §7 Datasets | "≥ 1 non-brain (embryo/tumour) and ≥ 1 non-transcriptomic panel (EASI-FISH)" — a stated reviewer defence against brain-only overfitting | T10 (nothing in `specs/` names any dataset requirement) |
| `v23_design.md` §3.5 | "Calibrate `π` **and the mean–variance relation** per gene against the flanking sections" — T09 calibrates `π` only | T09 §2 |
| `v23_design.md` §2.2 / §7 E1 | Zero-shot table must report **both** `r_g = 0` (pure text) and `r_g = ψ(t_g)` (distilled); T06/T10 require only one arm | T10 E1 |
| `v23_design.md` §5, §6 | The v20 **Bernoulli cross-mix** itself (see A6) | T06 |

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
  fix-the-bug criteria, since the latter three are implementation-correctness checks.
