# T07 — SEFL: sectioning-equivariant self-supervision

**Goal.** Turn in-silico sectioning itself into a training signal. A section is an *observation
operator* applied to one shared tissue; all valid operators must be mutually consistent. These
losses need **no ground truth**, so they turn 5 real sections into effectively unlimited
constraints — this is the answer to the small-N problem.

**Files:** `spatialcpav25_gen/losses/sefl.py`, `spatialcpav25_gen/infer/planes.py`, `tests/test_sefl.py`,
`tests/test_planes.py`

**Dependencies:** T01–T06.

---

## ⚠️ Read this before implementing: invariant vs. equivariant

Constraining the wrong quantities will actively damage the metrics this project is trying to win.

| **INVARIANT — constrain these** | **EQUIVARIANT — do NOT force to be equal** |
|---|---|
| The field `F(x,y,z)` itself | In-plane Moran's I / Geary's C |
| Cell identity and state at a physical point | Depth / laminar profiles |
| `p(expression \| cell type, region)` | Cells per unit **area** |
| Gene–gene covariance, module scores | Apparent structure size / elongation |
| Cells per unit **volume** | Total cells per section |

The right column genuinely differs with section angle because tissue is anisotropic — a tangential
cut through cortex has no laminar gradient, and that is correct, not an error. Forcing those to
match flattens real structure. Their correct treatment is **prediction** (T10, validation V3), not
a loss.

Encode this as a module-level constant `INVARIANT_QUANTITIES` / `EQUIVARIANT_QUANTITIES` with the
docstring above, so it is impossible to add a loss on the wrong side by accident.

---

## 1. Plane geometry — `spatialcpav25_gen/infer/planes.py`

```python
@dataclass(frozen=True)
class Plane:
    origin: np.ndarray     # (3,)
    normal: np.ndarray     # (3,), unit
    thickness: float       # micrometres
    def sample_points(self, n, bbox, seed) -> np.ndarray      # (n, 3) within slab ∩ bbox
    def basis(self) -> np.ndarray                             # (2, 3) in-plane orthonormal

def intersect(p1: Plane, p2: Plane) -> LineSegment | None
def random_plane_pair(bbox, min_angle_deg, max_angle_deg, seed) -> tuple[Plane, Plane]
def curved_surface(control_points, thickness) -> Surface       # for anatomy-following sections
```

`intersect` returns the line segment where the two mid-planes cross, clipped to the bbox; `None` if
they do not intersect inside it. Test this against hand-computed cases — geometry bugs here would
be misdiagnosed as model failures for weeks.

## 2. `L_cross` — plane-intersection consistency

```python
def loss_cross(model, model_ema, bbox, cfg, gen) -> Tensor
```

```
1. p1, p2 = random_plane_pair(bbox, min_angle=20 deg, max_angle=160 deg)
2. seg = intersect(p1, p2); skip (return 0) if None or shorter than 3 * median_nn_dist
3. Sample n_L = 256 points along seg
4. Branch 1: condition through p1's pathway (student, gradients on)
   Branch 2: condition through p2's pathway (EMA TEACHER, stop-gradient)
5. Loss = KL( ZINB_1 || ZINB_2 ) + ||h_1 - h_2||^2
        + CE(type_logits_1, softmax(type_logits_2)) + ||lambda_1 - lambda_2||^2
```

**Anti-collapse — mandatory.** Branch 2 must be the EMA teacher with `detach()`. A symmetric
consistency loss has the trivial minimiser "constant field" and will find it. Add a runtime alarm:
if the mean per-gene variance of generated expression drops below 25% of the real sections' at any
epoch, log a **COLLAPSE WARNING** with the epoch number.

Because the noise field is continuous in 3D (T03), both branches receive *identical* noise along the
intersection line — so this loss only has to correct the **conditioning** pathway, which is a much
easier optimisation than making two independent stochastic processes agree. Assert this in a test.

**Why it earns its place:** the intersection line crosses the tissue's depth axis at arbitrary
angles, so agreement along it forces every marker's depth gradient to be geometrically coherent in
3D rather than fitted per-plane. It is a direct regulariser on the marker-field and marker-depth
metrics.

## 3. `L_thick` — thickness coarse-graining consistency

A thick section is an aggregate of thin ones:
`S(Pi, 3h) == aggregate[ S(Pi, h, -h), S(Pi, h, 0), S(Pi, h, +h) ]`

```python
def loss_thick(model, model_ema, bbox, cfg, gen) -> Tensor
```

**Aggregate correctly — this is where naive implementations go wrong:**
- Cell **counts add**. Do not force equal counts between thick and thin. Compare
  `N_thick` vs. `sum(N_thin)` with a Poisson-consistent relative loss.
- For single-cell data the thick section is a **union** of cells: compare the *distributions* of
  cell states (entropic Sinkhorn on expression PCs, blur = median NN distance in PC space) and the
  per-type counts. There is **no per-cell correspondence** — do not attempt one.
- For binned/spot data, expression **sums** within a bin: compare binned totals directly.

Payoff beyond regularisation: this is a principled cross-technology harmonisation mechanism (thin
imaging sections vs. thick capture-based sections become consistent observations of one field). It
deserves its own paper subsection, and validation V4 in T10.

## 4. `L_prog` — molecular-program invariance

```python
def loss_prog(model, model_ema, bbox, cfg, gen) -> Tensor
```

```
For two random sampling angles theta1, theta2, for each (cell type c, region r) present in both:
    MMD^2( expr | c,r, theta1 ;  expr | c,r, theta2 )         # multi-bandwidth RBF
  + || Corr_gene(theta1) - Corr_gene(theta2) ||_F^2           # gene-gene covariance
  + sum_m || mean module_score_m(theta1) - mean module_score_m(theta2) ||^2
```

**Conditioning on `(c, r)` is essential.** Unconditional matching would be wrong: different planes
genuinely sample different mixtures of types and regions, and forcing marginals to match would force
the model to hallucinate a homogeneous tissue. Skip strata with fewer than 20 cells on either side.

Modules `m`: Leiden clusters of the gene–gene correlation graph on training data, computed once and
cached.

## 5. Schedule and cost

- Warm up on reconstruction only for `cfg.sefl_warmup_frac` (default 0.2) of training — early on,
  the consistency losses are satisfiable by degenerate solutions.
- Then ramp `w_cross`, `w_thick`, `w_prog` linearly to their configured values.
- Reconstruction terms must **dominate throughout**. Consistency is a regulariser, not the
  objective. Log the ratio `sum(consistency) / sum(reconstruction)`; warn if it exceeds 0.5.
- Cost control: apply SEFL on every 3rd step; sample **patches** of each plane (≈2000 points), not
  full sections; `n_L = 256` intersection points is ample.

## Acceptance tests

- `test_intersect_known_cases` — orthogonal planes through the origin, parallel planes (→ None),
  planes meeting outside the bbox (→ None). Hand-computed expectations.
- `test_noise_identical_along_intersection` — the GRF values along `seg` from both plane pathways
  are bitwise equal (guards the T03 property end-to-end).
- `test_cross_loss_decreases` — 500 training steps on the fixture reduce `L_cross` by ≥ 60%.
- `test_no_collapse` — after training with SEFL on, per-gene variance of generated expression stays
  ≥ 60% of the real sections'. **Run the same test with the EMA teacher disabled and assert it
  fails** — this documents that the asymmetry is load-bearing.
- `test_thick_counts_add` — `N` at thickness 3h is within Poisson tolerance of 3 × `N` at h;
  explicitly assert the loss does *not* penalise this difference.
- `test_prog_conditioning` — an unconditional variant of `L_prog` measurably homogenises the tissue
  (region-level expression differences shrink); the conditional version does not. Documents §4.
- `test_equivariant_not_constrained` — a section at 90° to the sectioning plane retains a *different*
  in-plane Moran's I from a 0° section after training. If these converge, an invariance was
  wrongly applied to an equivariant quantity.
- `test_sefl_cost` — SEFL adds < 60% wall-clock overhead per epoch at the configured sampling rates.

## Definition of done

Training with SEFL on the fixture: no collapse, `L_cross` converged, and held-in reconstruction
quality **not worse** than without SEFL (it should be equal or better). `PROGRESS.md` records the
consistency/reconstruction ratio and the collapse-alarm history.

## Also build: the deliberate negative control

Implement `loss_prog_WRONG` — a variant that (incorrectly) forces in-plane Moran's I to match across
angles. It is **not** used in training. It exists so the paper can show a trained ablation of it
performing *worse*, which makes the invariant/equivariant distinction concrete for reviewers and
preempts "why didn't you just constrain everything". Mark it clearly and exclude it from the default
loss registry.

## Do NOT

- Do not make both consistency branches symmetric.
- Do not constrain anything in the equivariant column.
- Do not attempt per-cell matching in `L_thick`.
- Do not let consistency losses outweigh reconstruction.
