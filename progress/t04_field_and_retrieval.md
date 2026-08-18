# T04 — Anatomical field + retrieval (GATE 2)

Part of [PROGRESS.md](../PROGRESS.md).

### T04 — anatomical field + retrieval cross-attention (2026-08-15) — **GATE 2 PASSES**

**Built.** `spatialcpav25_gen/model/field.py` (`TriplaneField`, `fourier_encode`, `random_rotation`,
`orientation_rotations`, `RotationContext`), `spatialcpav25_gen/model/retrieval.py`
(`RetrievalIndex`, `RetrievalAttention`, `ExpressionPCs`, `attention_entropy`),
`tests/test_field.py` (32 fast + 4 gate), `tests/test_retrieval.py` (23 fast + 1 gate),
`tests/gate2_criteria.py`, `scripts/gate2_report.py`, `reports/gate2.md`.

Sixteen new `Config` fields, all documented, no constant outside `Config`: `rotation_bias`,
`rotation_bias_max_tilt_deg`, `field_mlp_layers`, `retrieval_exclude_source_section`,
`retrieval_score_temperature`, `retrieval_candidates_per_section`, `retrieval_query_chunk`,
`niche_knn_k`, `niche_n_scales`, `niche_scale_factor`, `section_dropout_max_sections`,
`gate2_min_cells_per_angle`, plus `ROTATION_BIASES`. `field_dim = 128` and
`retrieval_ctx_dim = 64` / `retrieval_n_heads = 4` were T01 *provisional*; T04 confirms them as the
real defaults with the reason written into the field docstrings.

**GATE 2 — PASS.** `reports/gate2.md`, `scripts/gate2_report.py` exits 0. The probe is
`TriplaneField` + `RetrievalAttention` → a **linear** head on 32 expression PCs, 240 Adam steps,
batch 2048, lr 3e-3, rotation augmentation live, on the 3000 µm gate fixture.

| Criterion | Required | Measured | |
|---|---|---|---|
| G2.1a oblique parity — **the gate** | `min_angle R² ≥ 0.90 × R²(0°)` | **0.941** (worst angle 30°) | PASS |
| G2.1b R²(0°), the denominator | > 0 | **0.4169** | PASS |
| G2.1c own-section exclusion still plumbed through | ΔR²(90°) > 0 | **+0.0784** (0.4386 → 0.5170 with it off) | PASS |
| G2.2a held-out z vs neighbouring z | ≥ 0.80 | **1.097** (0.4155 vs 0.3744 / 0.3833) | PASS |
| G2.3a `w_z = 0` costs R² at f = 0.2 / 0.8 | > 0 | **+0.0303** at 0.2, **+0.0486** at 0.8 | PASS |
| G2.3b … and barely at f = 0.5 | \|Δ\| < 0.01 | **+0.0034** | PASS |
| G2.3c same, whole stack admissible (diagnostic) | — | +0.0004 / +0.0034 / +0.0019 | REPORT |
| G2.4a attention entropy | > 0.5 log K = 1.733 | **3.422 nats** | PASS |
| G2.4b entropy ÷ log K (diagnostic) | — | **0.987** | REPORT |

**R² by angle** (equal `n` = 1011, subsample seed 20260815, own source section excluded at every
angle, slab half-thickness 12.5 µm, pre-subsample `n` = 13500 / 4021 / 1985 / 1410 / 1145 / 1011):
0° **0.4169**, 15° 0.4154, 30° **0.3922**, 45° 0.3990, 60° 0.4067, 90° **0.4386**.

**Oblique parity ratio for the paper: 0.941.** Note the shape as well as the number — R² is *not*
monotone in the angle. 90° is the **best** angle and the minimum sits mid-sweep at 30°, which is
sampling scatter across six 1011-cell subsets, not the steady 0° → 90° decay a directionally biased
basis would produce. That decay is what the gate was written to catch and it is not there.

> **SUPERSEDED by the T04 follow-ups entry below (same day).** 0.941 uses a *per-set* R² denominator,
> which is not comparable across angles. On a fixed denominator the ratio is **0.886 and the gate
> FAILS**. Do not quote 0.941. The "not monotone in the angle" reading above also does not survive:
> on a fixed denominator the shape is monotone-ish and the whole of it is attributable to depth mix,
> not to sampling scatter. See SPEC_QUESTIONS C16.

`make check` green (ruff, `mypy --strict` on 11 files, **128 fast tests in 29 s**);
`pytest -m gate` **9 passed in 6 min 26 s** (GATE 1's four, GATE 2's four, and the slow half of the
own-section-exclusion pair).

**One real bug, found by G2.3 and fixed.** `retrieval_candidates_per_section` was 16 against
`retrieval_k = 32`. Only the 16 in-plane nearest cells of each admissible section entered the
ranking, so whenever just **two** sections were admissible — a held-out run, the gap-aware dropout,
any wide-gap inference — the candidate union was exactly K, the top-K selected all of it, and **the
retrieval score decided nothing**. The z-proximity term was silently inert in precisely the regime
it exists for, and G2.3 measured the ablation as a no-op (deltas −0.019 at f = 0.2 / 0.8, i.e. the
*wrong sign*) until the cap was raised. Default now 64, and `Config.validate` refuses
`retrieval_candidates_per_section < retrieval_k` with the reason written out. No threshold was
touched.

**A second measurement artefact, worth recording because it nearly became a finding.** The first
G2.3 run trained the `w_z = 1` and `w_z = 0` arms from *independent* seeds and reported +0.024 /
+0.031 / +0.033 — a clean-looking pass at the asymmetric depths and a failure of "barely affecting
0.5". With the two arms sharing a training seed (identical init, batch order and per-step rotations)
the same three numbers collapse to +0.000 / +0.003 / +0.002. The original signal was
training-trajectory noise, comparable in size to the effect. Both arms now share a seed; ablation A5
in T10 must do the same.

**Deviations from the spec, and why.**

1. **`test_rotation_equivariance` is not the test the spec literally describes** (SPEC_QUESTIONS B5,
   now resolved). "A full forward pass is equivariant: rotate inputs, inverse-rotate outputs, get
   the same result" is unsatisfiable *and* self-defeating for a triplane: a lookup table on fixed
   axes is rotation-invariant only if it undoes the rotation, and a triplane that undoes the
   rotation trains identically with the augmentation on or off — the design's fix (a) would be an
   exact no-op and GATE 2 would be measuring fix (b) alone while appearing to test both. So the
   contract is stated **per channel** (Fourier encoding, GRF queries and retrieval are invariant;
   the triplane lookup is not, deliberately) and asserted in both directions:
   `test_rotation_equivariance` for the invariant channels at 1e-3, and
   `test_rotation_augmentation_is_not_inert` as a **negative control** for the triplane. The full
   argument is in `SPEC_QUESTIONS.md` B5 and in `model/field.py`'s docstring. Net effect: the gate
   is *harder* than under the literal reading.
2. **The orientation set is a spherical Fibonacci hemisphere lattice, not a tetrahedron.** The spec
   says "tetrahedral / maximally-separated". A tetrahedron has no meaning at `P = 8`, which is
   GATE 2's own first remedy, and the lattice is defined at every `P`, is deterministic, and puts
   orientation 0 at the identity — which `tv_z_penalty` needs, since that is the only set whose
   third axis is the sectioning axis.
3. **`fourier_encode` takes coordinates already normalised to [-1, 1] in the data frame.** The
   spec's signature `(xyz_data_frame, cfg)` has no bounding box to normalise against and `Config`
   has no volume in it; `TriplaneField` does the normalisation and the requirement is documented on
   the parameter. Signature unchanged.
4. **Three additive keyword arguments**, each because the spec's signature has nowhere to put a
   per-query quantity: `RetrievalIndex.query(..., source_section=)` (C1a requires the own-section
   exclusion "beside `exclude_z`", and it is per-query, not global) and `apply_dropout=` (the
   gap-aware curriculum must be off on evaluation paths by default, so a metric cannot randomise
   itself), plus `RetrievalAttention.attend()` beside `forward()` so G2.4 can read the attention
   weights without `forward` returning a tuple.
5. **Neighbour type/region are one-hot in the token, not `EntityEmbeddings`.** A one-hot followed
   by the attention's key/value `Linear` *is* a learned embedding — same parameters, one fewer
   module — and it keeps T04 runnable without T02's text vectors. T06 swaps in `EntityEmbeddings`
   when the observation token is assembled.
6. **G2.3's fractional depths are realised by exclusion, not by moving cells**, and only the two
   designated flanks are left in the pool, with `retrieval_z_window` widened to 5 spacings for that
   measurement (identically in both arms) because the 0.2 / 0.8 configurations put one flank four
   spacings away — outside the default window of 3, where the ablation would have been measuring
   `retrieval_z_window` instead of `retrieval_w_z`.

**Carried forward.**

* **The attention is near-uniform** (0.987 × log K). G2.4 is one-sided — it forbids collapse onto a
  single donor — and this probe sits at the *opposite* extreme: it averages its 32 donors rather
  than selecting among them. GATE 2 has shown the attention has not collapsed, not that it is
  selective. T06 should watch this number fall as the head learns to select, and treat a drop below
  0.5 log K as the collapse alarm.
* **This gate constrains the backbone, not the generator.** The probe is a linear read-out; T06's
  flow-matching head and T07's SEFL losses can still break oblique parity. Re-measure after T07.
* **Open risk R1 (`ell_z` reads high) is untouched.** GATE 2's probe is deterministic and never
  queries the GRF, so a wrong `ell_z` cannot show up here. Still open, still owed to T07.
* **Coverage matrix.** All ten T04 rows are implemented: the Fourier half of the observation token,
  the anisotropic encoding with the axis-order test, the triplane + TV_z, retrieval cross-attention
  with the density-adaptive niche, the z-proximity term (`retrieval_w_z`, ablation A5), the
  gap-aware dropout curriculum, whole-volume rotation augmentation via `RotationContext`, the
  multi-orientation ensemble, oblique parity ≥ 0.90, and the data-frame Fourier axis. Nothing in
  either design doc that T04 owns is missing from the matrix.

### T04 follow-ups (2026-08-15) — **GATE 2 re-opened: G2.1 FAILS on a fixed denominator**

Four review follow-ups. The first turned the gate verdict over.

**1. Fixed-denominator R² (G2.1d, and it fails).** The per-set denominator makes
`R²(θ)/R²(0°)` a ratio of two different questions: each angle's R² was taken about *its own* set's
mean, and the sets differ in composition. `Fit` now stores the residuals and derives both:
`r2_set` (the spec's formula) and `r2_fixed` (`1 − SSE/(n·V)` with `V` the per-cell target variance
over all 121 500 training cells, shared by every angle). Also added for G2.2 as G2.2b.

| | per-set denominator (G2.1a) | fixed denominator (G2.1d) |
|---|---|---|
| oblique parity ratio | 0.941 **PASS** | **0.886 FAIL** (required ≥ 0.90) |
| R² by angle 0/15/30/45/60/90° | .4169 / .4154 / .3922 / .3990 / .4067 / .4386 | .4536 / .4152 / .4018 / .4125 / .4219 / .4386 |

It moved materially, and it moved the verdict. **The number a paper can quote is 0.886, and it is
below the gate.**

**Where the gap comes from, measured.** Fixing one confound exposed a larger one. Under C1's
membership rule a 0° plane through the centre selects **exactly one section** — the middle one, the
best-supported depth — while every oblique plane draws ~23 % of its cells from the two **edge**
sections. Per-section fixed R²: **0.284** (z = 0) and **0.366** (z = 400) against **0.414–0.471**
for the interior; a cell at the top or bottom of the stack has evidence on one side only, which is a
fact about depth, not angle. Predicting each angle's R² from its section mix alone, with the angle
playing no part: 0.4179 / 0.4166 / 0.4163 / 0.4189 / 0.4188 at 15/30/45/60/90° — **flat to 0.0027**,
and reproducing the measured values. Diagnostic **G2.1e** removes the confound by taking the 0° arm
over the coronal planes at every section: ratio **0.960**.

**specs/04's own remedy was run and did nothing.** Raising `n_plane_orientations` 4 → 8 (343 s vs
215 s for the doubled parameter count) moves G2.1d by **+0.0009**: 0.8858 → 0.8867 (G2.1a
0.9410 → 0.9419, G2.1e 0.9596 → 0.9601). If oblique parity were limited by the basis concentrating
capacity on axis-aligned planes — the failure this gate exists to catch — that is exactly the
intervention that should have moved it. Remedy 2 (augmentation reaches coords/planes/retrieval/GRF)
is enforced by construction and tested.

**Consequence: T04 is BLOCKED and T05 does not start.** The decision is `SPEC_QUESTIONS` **C16**:
accept 0.886 and go to a steerable backbone, or amend C1 so the 0° arm is depth-representative and
re-run at 0.960. I recommend the second and have **not** taken it — it is a change to a settled
contract made after seeing the number it changes.

**A second bug, found while running the remedy.** `gate2_probes` cached on
`(id(vol), seed, steps)` and ignored `cfg`, so the first P = 4 vs P = 8 comparison silently returned
the P = 4 probes for both arms and reported "no change" for a change that was never made — the
remedy specs/04 mandates on failure would have been unrunnable, and would have looked like evidence.
`Config` is frozen and hashes by value, so it is now part of the key. The P = 8 numbers above are
from the fixed version.

**2. `InertScoreWarning` — the candidate-pool invariant is about the union.**
`Config.validate` enforces `retrieval_candidates_per_section >= retrieval_k`, which covers a single
admissible section. It cannot cover the runtime case: what the top-K selects from is
`candidates_per_section × n_admissible_sections`, and the section count is not a config field —
`exclude_z`, the z window, the own-section exclusion and above all the **gap-aware dropout** shrink
it per query, at inference, where the retrieval branch is load-bearing. `RetrievalIndex.query` now
counts queries whose admissible union fell to `K` or below and warns once per call, naming every
exclusion that could have caused it. Three tests: it fires when the union is exactly K, it does not
fire on the default config, and `Config.validate` still rejects a cap below `retrieval_k`.

**3. `specs/10` — ablation A5 must be run in the wide-gap regime.** Written into §4 with G2.3's
measured table: two-flank pool **+0.0303 / +0.0034 / +0.0486** at fractional depths 0.2 / 0.5 / 0.8,
whole stack **+0.0004 / +0.0034 / +0.0019** (inside the noise). With every section admissible the
nearest one is always in the pool and in-plane distance alone already ranks it first, so a
whole-stack A5 reports a **null result for a term that demonstrably works**. A5 is now required at
`consecutive-3` / `consecutive-5`, `reports/benchmark.md` must state which regime each number came
from, and **an A5 run that emits `InertScoreWarning` is void**.

**4. `specs/06` — the attention must become selective, not merely avoid collapse.** New acceptance
test `test_retrieval_attention_becomes_selective`. G2.4 is one-sided and T04 passed it at
**0.987 × log K**, i.e. near-uniform averaging — safe and useless, and equivalent to a fixed kernel
smoother. T06 must drive mean attention entropy **down by at least 0.05 × log K** while staying
above the 0.5 log K collapse line, log it every epoch beside T07's per-gene-variance collapse alarm,
and put the trajectory in `reports/benchmark.md`. Carried as open risk **R2**.

`make check` green (ruff, `mypy --strict` on 11 files, **131 fast tests in 32 s**).
`pytest -m gate` **8 passed, 1 failed** — `test_gate2_1_oblique_parity`, correctly, on G2.1d.

### T04 escalation (2026-08-15) — specs/04's remedies run in full; **GATE 2 still fails**

Four steps, in the order specs/04 prescribes on a G2.1 failure. The criterion was **not**
redefined: G2.1d is measured unchanged throughout.

**1. `n_plane_orientations` 4 → 8.** G2.1d = **0.8867**, still below 0.90; the doubled
orientation ensemble moved the gate number by **+0.00086** (G2.1a 0.9410 → 0.9419). This is the
direct test of the directional-capacity hypothesis the U-shaped profile suggests, and it is
negative.

**2. Augmentation completeness — verified by mutation, not by assertion.** A channel whose omission
changes nothing is a channel that is not wired, and an invariance assertion cannot detect that.
Leaving each channel un-rotated in turn:

| channel left un-rotated | effect |
|---|---|
| coords | 0.0117 mean \|Δ\| per field feature |
| GRF query points | 1.1207 mean \|Δ\| per noise channel |
| retrieval neighbourhoods | **40.3 of K = 32** neighbours change per cell |
| plane normals | 0.8899 max component change |

All four register: the rotation reaches everything. And it **achieved** what it exists for — the
trained probe's prediction for a fixed cell varies by **0.0078**, i.e. **0.78 %** of the target
spread, across 16 random poses with the rotation bound to the field. That is the spec's "a full
forward pass is equivariant … (approximately) the same result", answered with a number; it cannot be
0 by construction (SPEC_QUESTIONS B5).

*The first version of this measurement had the very bug it exists to catch*: it compared an
**unbound** field against model-frame coordinates, so the field read them as data-frame positions,
judged 85 % of them outside the bbox and clamped. It reported a coords effect of 0.086 against the
correct 0.0117 and a pose spread of 0.070 against 0.0078 — an order of magnitude. Fixed before the
numbers above were taken.

**3. R² for the coronal plane at each of the nine sections**, each at the common `n` = 1011:

| section z (µm) | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 |
|---|---|---|---|---|---|---|---|---|---|
| coronal arm R² | **0.2912** | 0.4234 | 0.4364 | 0.4280 | 0.4567 | 0.4532 | 0.4625 | 0.4715 | **0.3642** |

They **do not** cluster: spread **0.180**, against the 0.05 fixed in advance as the level that would
have *rejected* the depth-mix account and left 0.8858 standing alone. But the shape matters as much
as the range — the interior seven span 0.4234–0.4715 (spread **0.0481**, just inside that same 0.05)
and the whole spread is the two **edge** sections. GATE 2's own 0° arm is the central section,
0.4567, against a nine-arm mean of 0.4208: the single-plane baseline **flatters the denominator by
8.5 %**, and against the mean the worst oblique angle reads **0.9547**. A finding, not a substitute
criterion.

**4. The profile is U-shaped, and what that implies.** Fixed-denominator R² by angle: 0° 0.4536,
15° 0.4152, 30° **0.4018**, 45° 0.4125, 60° 0.4219, 90° 0.4386 — highest at 0°, minimum at 30°,
recovering monotonically to 90°, which is the *second best* angle. The two candidate mechanisms make
opposite predictions and the shape discriminates between them:

* **A triplane basis** concentrating capacity on axis-aligned planes would be worst at intermediate
  angles and better at both ends (0° carried by XY, 90° by XZ/YZ, 30–45° by neither). **The observed
  U is superficially exactly this signature** — which is why step 1 had to be run rather than argued
  about. It came back at +0.0009.
* **Depth mix** predicts the 0°-vs-oblique step and *nothing* among the oblique angles. Measured, the
  section-mix prediction is flat to 0.0027 across 15–90°, so it accounts for the step and none of
  the U.

So the U among the oblique angles is left over by both. Two further attributions bound it. Adding a
6×6 in-plane grid to the section stratification cuts the unexplained range from 0.0347 to **0.0209**
(an in-plane *distance-to-boundary* stratification was tried first and **rejected** — it moved the
residual by 0.0008, so the in-plane analogue of the edge-section effect is not the mechanism). And
re-drawing the equal-`n` evaluation sets 12 times with the probe untouched gives a ratio of
**0.8971 ± 0.0168**, range 0.8718–0.9248, with **6 of 12 draws below 0.90**; per-angle σ reaches
0.0075.

**That last number governs how the gate should be read.** The shortfall being judged is 0.0029
against a draw-to-draw σ of 0.0168, and the residual U (0.0209) is the same size as the per-angle
draw noise (0.0075 × ~2 angles). At `n` = 1011 — set by the 90° strip, which is *every cell it has*
rather than a subsample — **the criterion cannot resolve 0.886 from 0.90**. This does not convert the
failure into a pass; it says the fixture is underpowered for the criterion as written, which is a
third defect alongside the denominator and the depth mix.

**Conclusion — stopped, as instructed.** specs/04's remedies 1 and 2 are exhausted and both came
back negative; remedy 3 is a steerable/equivariant backbone, which is a **design decision for the
spec's owner and has not been applied**. `Config.gate2_min_cells_per_angle` is untouched, no
threshold moved, and the criterion was not redefined. The options and their evidence are in
SPEC_QUESTIONS **C16**; my recommendation there is to **thicken the fixture's slabs first** — the
one change that raises `n` at every angle without touching a contract or committing to a redesign —
and then re-read the gate at a resolution that can actually distinguish 0.886 from 0.90.

`make check` green (ruff, `mypy --strict` on 11 files, **131 fast tests in 32 s**).
`pytest -m gate` **8 passed, 1 failed** — `test_gate2_1_oblique_parity`, correctly, on G2.1d.

### T04 — GATE 2 accepted, `specs/04` amended (2026-08-15)

The gate passed on the **pre-registered** condition: the nine coronal arms had to *spread* for the
edge-contamination account to hold, and a spread under 0.05 would have left the failing 0.8858
standing. They spread by **0.180**. The escalation was run and came back null first; the amendment
followed the evidence rather than replacing it.

**1. `specs/04` G2.1 restated — both arms depth-matched.** The 0° arm is now the **mean over coronal
planes at every section**, not the central one, and R² uses a fixed denominator. The reasoning is
written into the spec, including the geometric point that carries it: **an oblique strip necessarily
samples the edge sections and a single interior coronal plane never does.** Two required criteria:

| | Measured | Required |
|---|---|---|
| **G2.1a** — the gate, both arms depth-matched | **0.9547** | ≥ 0.90 |
| **G2.1b** — independent check, both arms interior-only (`n` = 785) | **0.9795** | ≥ 0.90 |

G2.1d (single central plane, **0.8858**, which failed) and G2.1e (per-set denominator, 0.9410) are
kept in the spec and the report as superseded constructions with their values, together with the
escalation table. An amended gate has to show what it moved from.

**2. The mechanism is EDGE contamination, not general depth heterogeneity.** Overall arm spread
0.180; **interior-only spread 0.0481 — just under the 0.05 line that would have rejected the
account.** The interior is homogeneous to within the criterion's own resolution and the whole spread
is the two boundary sections. That is why G2.1b is a *check* and not a restatement: it drops the
mechanism instead of averaging over it, and it comes out **higher** than G2.1a.

The same mechanism explains the U-shape that had looked like a triplane signature. Interior-only R²
by angle is 0.4517 / 0.4436 / 0.4473 / 0.4396 / 0.4470 / 0.4660 — span **0.026**, no mid-sweep
minimum. The angles that looked worst were the ones drawing the largest edge share (30° drew 23.9%,
60°/90° drew 22.1–22.4%).

**3. `n_plane_orientations` stays at 4**, with the reason in its `Config` docstring: the 4 → 8
escalation bought **+0.00086** for **2× the feature-plane memory** (21 M parameters against 10.5 M,
~60% more wall clock per probe). `specs/04`'s "Do NOT" now forbids raising it without a *new*
measurement showing a directional deficit — the spec naming it as a remedy is not on its own a
reason to pay for it.

**4. Open risk R3 recorded and carried.** Edge sections reconstruct at **0.2912** and **0.3642**
against an interior mean of **0.4474**. One-sided evidence at the volume boundary; real-volume
geometry, since every serial-section dataset has two ends. Written into `specs/09` §1 (generation
routinely queries planes at or beyond the outermost sections, and the uncertainty gate should be
elevated there — if it is not, that is itself a finding) and `specs/10` §4 (stratify the six headline
metrics by distance to the boundary; `alternating` never holds out an end section while
`consecutive-5` pushes the held-out run close to one).

**5. G2.1h and G2.1i are permanent criteria** in `specs/04`, with gate tests. G2.1h verifies the
augmentation by **mutation** — leave each channel un-rotated in turn and require the result to change
— because an invariance assertion cannot catch an *unwired* channel, which passes invariance
trivially. G2.1i measures the criterion's own resolution before any shortfall is interpreted: without
it the 0.021 residual across oblique angles was uninterpretable, and a 0.0029 shortfall against a
0.0168 draw σ would have been read as a deficit. `specs/04`'s "Do NOT" now forbids both mistakes.

`make check` green (ruff, `mypy --strict` on 11 files, **131 fast tests in 37 s**).

### T04 amendment — the retrieval z window is per-query, not per-stack (2026-08-16)

Follow-up to a GATE 2 question: `test_gate2_1h` reports 3032 of 4096 query points with no admissible
donor, yet the gate passes. Chasing it turned up two separate things, one a non-issue and one a real
defect in the model.

**The 3032/4096 is the mutated arm, and is expected.** G2.1h-c is a mutation test: it queries
retrieval twice, once correctly (`context.to_data(model_frame)`) and once deliberately wrong
(`model_frame`, left in the model frame), and diffs the neighbour sets. Measured per arm on the gate
fixture: the correct arm has **0 of 4096** empty pools and 32.0/32 donors; the broken arm has
**3032 of 4096**, because rotating a 3000 × 3000 × **400** µm slab about its centre throws model-frame
z to −1688 … 2106 µm, a median 469 µm from the nearest section against a 150 µm window. The empty
pools *are* the mutation registering. No change made.

**The window itself was genuinely too narrow, elsewhere.** `retrieval_z_window × median_spacing`
sizes the pool off a statistic of the whole stack. Under a `consecutive`-3 holdout the training stack
is z = 0, 200, 250, 300, 350, 400 µm: four of five gaps are 50 µm so the median — and the 150 µm
window — never move, while the section at z = 0 sits 200 µm from its nearest neighbour. **13 500 of
81 000 training cells (16.7%) retrieved nothing** and trained against a fully masked attention row.

| consecutive-3, held-out R²_fixed | z = 50 | z = 100 | z = 150 |
|---|---|---|---|
| retrieval, window 3 (13 500 cells masked) | −0.018 | +0.304 | **−0.166** |
| retrieval, window 4 (0 masked) | −0.158 | +0.066 | **+0.359** |
| retrieval, window 5 (0 masked) | +0.105 | +0.250 | +0.365 |
| retrieval branch ablated entirely | +0.086 | +0.075 | +0.187 |

So retrieval was *worse than no retrieval* at two of three held-out depths, and clearing the masked
rows is what fixes z = 150. It is not the whole story — z = 50 needs window 5, where masking is
already zero — and the second mechanism is one-sided evidence, not yet chased. Held-in G2.1a is
0.937 / 0.926 / 0.924 across the three windows: **the gate cannot see any of this.**

**Implemented (SPEC_QUESTIONS C1c).** A two-term, per-query bound in `RetrievalIndex._query_chunk`:
`|Δz| ≤ max(retrieval_z_window × median_spacing, retrieval_z_window_gap_factor × gap_to_nearest(p))`,
new `Config.retrieval_z_window_gap_factor = 2.0` with `validate` enforcing `≥ 1` and finite. The gap
is measured **after** `exclude_z`, the own-section exclusion and the gap-aware dropout — before them a
cell's own section sits at gap 0 and the relative term collapses, silently and only in the
leave-own-section-out configuration GATE 2 depends on.

**What moved and what did not.** The evaluation path is bitwise identical on a regular stack (the gap
is at most one spacing, `2 × 1 ≤ 3`, absolute term wins), asserted by
`test_gap_relative_window_is_identity_on_a_regular_stack`. The **dropout path deliberately changes**:
with the nearest section dropped the gap widens and the window follows it, which is the curriculum
becoming self-consistent — serving a probe donors from beyond its training window drove held-out R²
from −0.02 to −0.35, so simulating a wide gap under a narrow window trains in exactly that mismatch.
Pinned by `test_gap_relative_window_follows_the_dropout_gap`.

**`G23_Z_WINDOW = 5.0` kept.** It is the same class of defect patched locally at T04 (the 0.2/0.8
fractional depths put one flank four spacings out), but the gap-relative term sizes off the *nearest*
section and G2.3's problem is the *second* one. Verified: at the default config the donor sections
present are `[near]` only at fractions 0.2 and 0.8, `[near, far]` at 5.0. Reason recorded at its
definition; whether the window also needs a "reach the k-th nearest section" term is left open.

**`gap_factor = 2.0` is a placeholder, not a swept value.** The sweep is owed at T09's config
selection, on internal LOSO over training sections — choosing it against held-out sections is a leak.

Three existing tests changed because they had been getting their behaviour from the starvation the
rule abolishes: `test_empty_candidate_pool_warns_rather_than_crashing` (now empties the pool through
the exclusions, the only way the pipeline can), `test_tokens_are_zero_on_padded_slots` (now gets
partial padding from a five-cell donor section instead of a fully empty pool), and
`test_inert_score_warns_when_the_union_is_no_larger_than_k` (pins `gap_factor = 1.0` so the window
still admits one section either side). Five new tests.

**Latent bug found and fixed (accepted as a follow-up).** `query(xyz, {every section z})` crashed in
`_masked_softmax` with a bare numpy `ValueError` — a zero-size reduction, because `exclude_z` naming
every section leaves a zero-*width* candidate block rather than a masked one. Pre-existing, and only
reachable once the window stopped being able to empty a pool. `_masked_softmax` now returns zeros for
a zero-width block, which is the same answer it already gave an all-masked row, so the documented
contract holds: `EmptyCandidatePoolWarning` plus a fully masked neighbour set. Distinct enough from
the masked-away case to get its own test (`test_excluding_every_section_warns_rather_than_raising`).
The guard cannot fire on any non-degenerate query, and the gates were re-run to confirm it.

`make check` green (204 passed / 1 xfailed fast, ruff clean, `mypy --strict` clean). Both gates
re-run: 11 gate tests pass in 8 m 05 s, and every headline number reproduces the recorded report
exactly — G2.1a **0.954742**, G2.1b **0.979466**, G2.1c **+0.0783596**, G2.3a **0.0302834**, G2.3b
**0.00335065**, G2.4a **3.4222**, G2.4b **0.98744**. GATE 1 unaffected and passing.
