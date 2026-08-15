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

### G2.1 — Oblique parity (**the gate**)
Reconstruct held-in cells from planes at dihedral angles 0°, 15°, 30°, 45°, 60°, 90° to the
sectioning plane, on the evaluation set defined above. Report R² per angle, with `n`.

**Required: `min_angle R² ≥ 0.90 × R²(0°)`.**

If this fails:
1. First raise `n_plane_orientations` 4 → 8 and re-run.
2. Then verify rotation augmentation is actually applied to *all* of coords/planes/retrieval/GRF.
3. If still failing, **stop and report** — the backbone needs a steerable/equivariant architecture,
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

## Definition of done

`reports/gate2.md` with the angle-vs-R² table and all four criteria passing, **and the evaluation-set
contract stated in full**: the slab half-thickness, the common `n` after subsampling, the
pre-subsample `n` per angle, the subsample seed, and confirmation that own-section retrieval was
excluded. `PROGRESS.md` updated with the oblique parity ratio — that number goes in the paper.

## Do NOT

- Do not use a dense 3D voxel grid (memory, and it overfits z even harder).
- Do not encode absolute neighbour positions.
- Do not proceed to T05 without G2.1 passing.
