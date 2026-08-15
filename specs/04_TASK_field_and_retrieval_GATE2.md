# T04 — Anatomical field + retrieval cross-attention  ⛔ **GATE 2**

**Goal.** A continuous, rotation-agnostic representation of tissue anatomy `F(x,y,z) -> f`, plus a
retrieval branch that grounds generation in real observed cells. Together these are the conditioning
signal for both heads.

**Files:** `spatialcpav25_gen/model/field.py`, `spatialcpav25_gen/model/retrieval.py`, `tests/test_field.py`,
`tests/test_retrieval.py`, `scripts/gate2_report.py`

**Dependencies:** T01, T03.

---

## 1. Multi-orientation triplane field — `spatialcpav25_gen/model/field.py`

```python
class TriplaneField(nn.Module):
    """F(x,y,z) -> (N, d_f). Continuous, queryable anywhere in the bbox."""
    def __init__(self, cfg: Config, bbox: np.ndarray): ...
    def forward(self, xyz: Tensor) -> Tensor: ...
    def tv_z_penalty(self) -> Tensor: ...
```

Structure:
- `cfg.n_plane_orientations` (default 4) **triplane sets**, each rotated by a fixed
  `R_p ∈ SO(3)`. Use a tetrahedral / maximally-separated set of orientations, computed
  deterministically at init, not random.
- Each set = three feature planes: `XY (res_xy², C)`, `XZ (res_xy × res_z, C)`,
  `YZ (res_xy × res_z, C)`, bilinearly sampled with `grid_sample(align_corners=False)`.
- Query: for each orientation `p`, rotate `xyz` by `R_p`, normalise to `[-1,1]` using the rotated
  bbox, sample three planes, concatenate; then **sum across orientations** (sum, not concat — keeps
  `d_f` fixed and makes the ensemble a smoothing over directions).
- Concatenate the anisotropic Fourier positional encoding (below) and pass through a 3-layer MLP
  with `cfg.field_mlp_hidden` → `d_f`.

**Why the multi-orientation ensemble exists:** a single axis-aligned triplane concentrates
representational capacity on the axis-aligned planes, so oblique sections come out systematically
blurrier than coronal ones. That would gut the paper's central capability claim. This is GATE 2.

### Anisotropic Fourier encoding

```python
def fourier_encode(xyz_data_frame: Tensor, cfg: Config) -> Tensor
```

`cfg.fourier_bands_xy = 8` in-plane, `cfg.fourier_bands_z = 2` along the **sampling axis**.
Justification: with 4–9 sections along z, high-frequency z basis functions are unconstrained and
will overfit to the section positions.

⚠️ **The most likely silent bug in this task.** The low-frequency axis must follow the *tissue's
true sectioning axis*, not the model's current coordinate frame. When rotation augmentation is on,
the encoding must be computed in the **original data frame**, before the augmentation rotation is
applied. Write the transform order explicitly in a comment and assert it in a test
(`test_fourier_axis_follows_data_frame`): encoding a fixed cell must be invariant to the
augmentation rotation.

### Rotation augmentation

```python
def random_rotation(seed: int, bias: str = "uniform") -> np.ndarray  # (3,3)
```

Applied per training step to the **whole volume** — coordinates, plane definitions, retrieval
neighbourhoods, GRF query points — consistently. Provide `RotationContext` as a context manager so
it is impossible to rotate one and forget another. Include a test that a full forward pass is
equivariant: rotating inputs and inverse-rotating outputs returns (approximately) the same result.

### Regularisation

`tv_z_penalty()` — total variation along the z axis of the XZ/YZ planes of the **unrotated**
orientation only (the rotated ones have no meaningful z axis). Weight `cfg.tv_z_weight`.

---

## 2. Retrieval cross-attention — `spatialcpav25_gen/model/retrieval.py`

```python
class RetrievalIndex:
    """Real cells available as conditioning evidence, with gap-aware masking."""
    def __init__(self, vol: Volume, cfg: Config): ...
    def query(self, xyz: np.ndarray, exclude_z: set[float], seed: int
              ) -> tuple[np.ndarray, np.ndarray]:   # (N, K) indices, (N, K) weights

class RetrievalAttention(nn.Module):
    def forward(self, q_feat: Tensor, neigh_tokens: Tensor, neigh_mask: Tensor) -> Tensor
```

Ranking score for candidate real cell `j` against query point `p` — **all three terms required**:

```
score = -( d_inplane(p, j) / median_nn_dist )
        -  w_z * ( |z_p - z_j| / median_spacing )        <-- SpatialZ omits this entirely
        +  w_niche * cos( niche(p), niche(j) )
```

The z-proximity term is the specific, nameable flaw in the competing method: its donor weights use
in-plane distance and niche similarity only, so a cell one-fifth of the way between two sections
draws roughly equally from both. Make `w_z` a config field and ablate it (this is ablation A5).

`niche(·)` — a local cell-type composition vector at 3 spatial scales, computed with a
**density-adaptive** radius (`r = k-th nearest neighbour distance`, not a fixed micrometre radius),
so it transfers across datasets with different cell densities.

**Gap-aware section dropout.** With probability `cfg.section_dropout_p`, drop the nearest section(s)
from the candidate pool during training. This is the wide-gap curriculum: the model must learn to
reconstruct from remoter evidence. Unlike previous versions where the curriculum was inert, here the
retrieval branch is load-bearing at inference, so the curriculum actually changes behaviour.

Attention: standard multi-head cross-attention, query = `[F(p), fourier(p), type_emb, ctx]`,
keys/values = encoded neighbour tokens `[expr_pca, type_emb, region_emb, Δposition, Δz]`. Mask
padded neighbours. Output `ctx ∈ R^{d_ctx}`.

⚠️ Encode neighbours by **relative** position (`p_j − p`), never absolute — absolute coordinates
let the model memorise section identity and it will not generalise to a new z.

---

## ⛔ GATE 2 — acceptance criteria

`scripts/gate2_report.py`, on the synthetic fixture, trains a **lightweight probe**: field +
retrieval → linear head predicting the top-32 expression PCs of held-in cells. (The full generative
heads do not exist yet; this isolates the backbone's representational quality.)

### The evaluation set — read this before writing the probe (settled, SPEC_QUESTIONS C1)

Real cells exist only on the sectioning planes, so an oblique query plane passes through very few of
them. Three rules make the angles comparable; **all three are part of the criterion, and
`reports/gate2.md` must state the contract and the numbers it produced.**

1. **Membership.** The evaluation set at angle θ is every training-section cell within
   `thickness / 2` of the query plane, pooled across all training sections. Report `n` per angle.
2. **Equal `n` across angles.** Subsample every angle's set to the smallest of them, with an
   explicit seed, and evaluate on that. Reporting `n` is not enough: R² is a variance-explained
   ratio and its sampling error, and the mix of tissue it covers, both move with `n`, so an
   unsubsampled comparison partly measures sample size. Report the common `n` and the pre-subsample
   `n` per angle. If the common `n` falls below a floor (a new `Config` field, added by T04, e.g.
   `gate2_min_cells_per_angle`), **thicken the fixture's slabs and re-run — do not lower the floor
   and do not drop an angle.**
3. **Leave-own-section-out retrieval.** For every evaluated cell, its **own source section is
   excluded from the retrieval candidate pool, at every angle** — the same exclusion the model
   would face for a genuinely unseen plane. Without it, a cell in the 90° strip retrieves in-plane
   neighbours a few micrometres away *inside its own section*: the oblique plane becomes trivially
   easy, R²(90°) rises to meet R²(0°), and **GATE 2 passes while hiding exactly the equivariance
   failure it exists to detect**. This needs a `Config` flag (added by T04, e.g.
   `retrieval_exclude_source_section`, default `True`) plumbed through `retrieve()`'s candidate
   filter beside `exclude_z`, and an acceptance test (below) that fails if the exclusion is dropped.

4. **A fixed R² denominator, and both arms depth-matched.** Added after T04 measured the
   consequences of leaving them unstated; see "Why G2.1 is stated this way" below. R² is
   `1 − SSE / (n · V)` with `V` the per-cell target variance over **all** training cells, shared by
   every angle — not each set's own variance. And the 0° arm is the **mean over coronal planes at
   every section**, not a single plane through the volume's centre.

### G2.1 — Oblique parity (**the gate**)

Reconstruct held-in cells from planes at dihedral angles 0°, 15°, 30°, 45°, 60°, 90° to the
sectioning plane, on the evaluation set defined above. Report R² per angle, with `n`.

**G2.1a — required: `min_angle R²_fixed ≥ 0.90 × mean_over_sections R²_fixed(0° at that section)`.**

Both arms are depth-matched. The numerator is the worst oblique angle's fixed-denominator R²; the
denominator is the mean of the coronal arms placed at *each* section in turn, each at the common `n`.

**G2.1b — required, independent check: the same ratio with both arms restricted to the interior
sections.** Drop the first and last section from every angle's membership *and* from the coronal
arms, re-derive the common `n`, and re-measure. This drops the mechanism instead of averaging over
it, so the verdict rests on two matched constructions rather than one. If G2.1a clears 0.90 and
G2.1b does not, the pass is an artefact of the averaging and must not be accepted.

#### Why G2.1 is stated this way (T04, 2026-08-15)

The original form — `min_angle R² ≥ 0.90 × R²(0°)`, each R² about its own set's mean, the 0° arm a
single plane through the centre — has two defects, both measured rather than argued:

1. **The denominator was not comparable across angles.** A 0° strip is one section and carries only
   in-plane target variance; a 90° strip spans the stack and carries the along-z variance too.
   Measured, the per-cell denominators differed by 1.07×, so part of the ratio answered "how much
   variance was there to explain". Fixing the denominator moved the number from 0.941 to 0.886.

2. **An oblique strip necessarily samples the edge sections; a single interior coronal plane never
   does.** This is geometry, not sampling: a plane at any angle other than 0° cuts through the whole
   stack, so ~23% of its cells come from the first and last sections — and those sections have
   training and retrieval evidence on **one side only**. A 0° plane through the centre draws none of
   them. The old criterion therefore compared a depth-representative numerator against a
   depth-privileged baseline, and the gap it reported was largely that comparison.

**The mechanism is EDGE contamination specifically, not general depth heterogeneity**, and the
distinction is what justifies the amendment rather than merely motivating it. Measured coronal arms,
one per section, at the common `n` = 1011:

| section z (µm) | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| R²_fixed | **0.2912** | 0.4234 | 0.4364 | 0.4280 | 0.4567 | 0.4532 | 0.4625 | 0.4715 | **0.3642** |

Overall spread **0.180**. But the **interior-only spread is 0.0481 — just under the 0.05 fixed in
advance as the level that would have *rejected* this account** and left the failing number standing
on its own. The interior sections are homogeneous to within the criterion's own resolution; the
entire spread is the two boundary sections. That is why G2.1b (interior-only, both arms) is a
required check and not a footnote: if the amendment were laundering general heterogeneity, the
interior-only construction would not clear 0.90 either.

**The escalation was run first, and came back null.** The amendment is not a substitute for the
spec's remedies — those were exhausted before it was proposed:

| Escalation step | Result |
|---|---|
| `n_plane_orientations` 4 → 8, criterion unchanged | **+0.00086** (0.8858 → 0.8867). Reverted to 4: 2× the feature-plane memory for nothing |
| Augmentation reaches coords / planes / retrieval / GRF | Verified by **mutation** — each channel left un-rotated in turn changes the result (coords 0.0117, GRF 1.1207, retrieval 40.3 of K = 32 neighbours, plane normals 0.8899). All four wired |
| Full forward pass "approximately equivariant" | **0.78%** of the target spread across 16 random poses |

Both mechanisms the gate exists to catch were ruled out by measurement before the contract was
touched.

**Report all of it.** `reports/gate2.md` states G2.1a and G2.1b as the verdict, and keeps the two
superseded constructions with their values — the single-central-plane fixed denominator (**0.886,
which failed**) and the original per-set denominator (0.941) — plus the escalation table. A gate
whose definition moved must show what it moved from.

If G2.1a or G2.1b fails:
1. First raise `n_plane_orientations` 4 → 8 and re-run.
2. Then verify rotation augmentation is actually applied to *all* of coords/planes/retrieval/GRF —
   by mutation (G2.1h), not by an invariance assertion, which an *unwired* channel passes trivially.
3. Check the draw-noise floor (G2.1i) before concluding anything: at the fixture's `n` the ratio's
   draw-to-draw σ was 0.0168, which is larger than the shortfall the original criterion reported.
   A deficit smaller than the floor is not a deficit, it is an unmeasurable.
4. If still failing, **stop and report** — the backbone needs a steerable/equivariant architecture,
   which is a design change, not a tuning fix. Better to learn this now than in month three.

### G2.2 — z-interpolation, not memorisation
Hold out a middle section; the probe's R² at the held-out z must be ≥ 0.8 × the mean R² at
neighbouring training z. A sharp dip at the held-out z means the triplane has overfit to section
positions → reduce `fourier_bands_z`, raise `tv_z_weight`.

### G2.3 — z-proximity term earns its place
Ablating `w_z = 0` must **worsen** probe R² for query points at fractional positions 0.2 and 0.8
between two sections, while barely affecting 0.5. This confirms the mechanism does what the design
claims (and pre-validates ablation A5).

### G2.4 — Retrieval does not collapse to copying
Attention entropy over the K neighbours must be > 0.5 × log(K) on average. If attention is
one-hot, the model is copying the nearest cell and will fail on wide gaps.

Note the criterion is **one-sided**. T04 passed it at 0.987 × log(K) — near-uniform averaging, the
*opposite* extreme from collapse, and equivalent to a fixed kernel smoother. G2.4 therefore shows
only that the attention has not collapsed, never that it is selective; `specs/06` carries the
requirement that T06 drive the number down while staying above this line.

### G2.1h — Rotation augmentation reaches all four channels (**permanent criterion**)

For each of coords / plane definitions / retrieval neighbourhoods / GRF query points in turn, leave
that channel un-rotated while the other three rotate, and assert the result **changes**. Report the
size of each effect.

This is a mutation test and it must stay one. An invariance assertion cannot do this job: an
*unwired* channel satisfies invariance trivially, so "the forward pass is equivariant" passes just as
happily when a channel is missing as when everything is correct. A partial rotation produces the same
signature as a directional deficit, and T04's oblique shortfall could not have been diagnosed without
separating them. Report the achieved equivariance beside it — the spread of the trained probe's
prediction for one fixed cell across random poses, as a fraction of the target spread — which is the
honest form of "a full forward pass is (approximately) equivariant".

### G2.1i — The criterion's own resolution (**permanent criterion**)

Re-draw the equal-`n` evaluation sets under several independent seeds, with the probe untouched, and
report the standard deviation of the G2.1a ratio and of each angle's R².

**Report before interpreting any shortfall.** At T04's fixture the ratio's draw-to-draw σ was 0.0168
while the shortfall being judged was 0.0029, and 6 of 12 draws straddled the threshold: the criterion
could not resolve the number it was being asked about. The residual variation across oblique angles
(0.021 after stratification) was likewise the same size as the draw noise, and without this floor it
would have been read as evidence of a directional mechanism. A deficit smaller than the floor is not
a deficit.

---

## Other acceptance tests

- `test_field_continuity` — `||F(p) − F(p+δ)||` scales ~linearly with small `δ`; no discontinuities
  at plane boundaries.
- `test_rotation_equivariance` — described above; tolerance 1e-3 relative.
- `test_fourier_axis_follows_data_frame` — described above; this test catches the §1 bug.
- `test_bbox_query_outside` — querying outside the bbox clamps and warns, does not crash.
- `test_retrieval_excludes_holdout` — `exclude_z` is honoured; a test asserts no held-out section
  index is ever returned.
- `test_retrieval_excludes_source_section` — with `retrieval_exclude_source_section=True`, no
  returned neighbour shares the query cell's `section_id`. Pair it with
  `test_source_section_exclusion_changes_oblique_R2`: with the exclusion **off**, R²(90°) rises
  measurably on the fixture. That second test is what stops the exclusion from being quietly
  dropped later — if it ever passes with no difference, the exclusion is not plumbed through.
- `test_niche_density_adaptive` — doubling all coordinates leaves niche vectors unchanged.
- `test_relative_position_only` — translating the whole volume by a constant leaves outputs
  unchanged (catches absolute-coordinate leakage).
- `test_inert_score_warns_when_the_union_is_no_larger_than_k` — the candidate-pool invariant is
  about the **union** (`candidates_per_section × n_admissible_sections`), not the per-section cap.
  `Config.validate` covers a single admissible section; only a runtime warning can cover the gap
  dropout, which shrinks the section count per query *at inference*.

## Definition of done

`reports/gate2.md` with the angle-vs-R² table and every criterion passing — **G2.1a and G2.1b, G2.2,
G2.3, G2.4, plus the permanent G2.1h and G2.1i** — **and the evaluation-set contract stated in
full**: the slab half-thickness, the common `n` after subsampling (for both the full and the
interior-only construction), the pre-subsample `n` per angle, the subsample seed, and confirmation
that own-section retrieval was excluded. The two superseded constructions and the escalation table
are kept in the report, so the record shows what the criterion moved from and what was tried before
it moved.

`PROGRESS.md` updated with the oblique parity ratio — **the depth-matched G2.1a number, named as
such**. Quoting the per-set or single-central-plane variants without saying which is how a
0.941 / 0.886 / 0.955 disagreement gets into a paper.

## Do NOT

- Do not use a dense 3D voxel grid (memory, and it overfits z even harder).
- Do not encode absolute neighbour positions.
- Do not proceed to T05 without G2.1 passing.
- Do not raise `n_plane_orientations` above 4 without a measurement showing a *directional* deficit.
  T04 ran the 4 → 8 escalation: it bought +0.00086 for 2× the feature-plane memory, because the
  deficit it was meant to address was not directional. The spec naming it as a remedy is not on its
  own a reason to pay for it.
- Do not read a G2.1 shortfall smaller than the G2.1i draw-noise floor as a deficit.
- Do not substitute an invariance assertion for G2.1h's mutation test: an unwired channel passes
  invariance trivially, which is exactly the case the criterion exists to catch.
