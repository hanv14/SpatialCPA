# The oblique demonstration — geometry, resolution and preconditions

**Read `reports/oblique_demonstration_preregistration.md` first**, including §2-ter (F2: θ\* excludes an evaluation set of zero measure, and no fill floor below that is derivable), **§2-quater** (F2's second repair: the stratum is tested in **micrometres** against the volume's median nearest-neighbour distance, because `fill > 0` admitted 90° at 3 × 10⁻¹⁷; and the donor slab is offset by the section **spacing**, not the slab thickness), §3-bis (donors are a flanking slab), and `reports/metric_resolution.md`.

`merfish_thick_hypothalamus`, 4 sections at 57.5 µm, slab thickness **28.6 µm**.

> **Thickness provenance:** EXTERNAL: specs/10 §8 — a 200 µm block cut into 7 slabs (200/7 = 28.6)

| θ | fill | **stratum** | truth | strata | donors | types | **G1 margin** | largest | blur | blur/radius | clears |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0° | **0.50** | — | 11104 | 1 | 14585 | 9 | **+3.6** | 4473 | 186 µm | 0.300 | n/a — the coronal control |
| 30° | **0.43** | 49.5 µm | 1906 | 4 | 1991 | 8 | **+2.6** | 724 | 115 µm | 0.259 | **yes** |
| 45° | **0.35** | 28.6 µm | 1311 | 4 | 1456 | 8 | **+2.6** | 417 | 112 µm | 0.264 | **yes** |
| 60° | **0.25** | 16.5 µm | 1011 | 4 | 1198 | 6 | **+0.6** | 328 | 116 µm | 0.255 | **yes** |
| 90° | **0.00** | 0.0 µm | 989 | 4 | 913 | 7 | **+1.6** | 471 | 115 µm | 0.263 | no — **zero measure (F2)** |

## The two bounds, both on the field rather than on this method

**The comb limit** (`reports/the_comb_limit.md`). `fill = t·cos θ / s`. An oblique ground truth from `N` serial sections has `N` samples along depth, so the cells lie in `N` strata, each **`t·cos θ / sin θ`** micrometres wide. **F2 tests that width in micrometres against this volume's median nearest-neighbour distance, 8.0 µm** (§2-quater): a stratum narrower than the spacing between neighbouring cells is a line drawn through a point cloud. Testing `fill > 0` instead admitted 90° at 3 × 10⁻¹⁷, because `cos(π/2)` is 6 × 10⁻¹⁷ in binary.

**The metric's resolution** (`reports/metric_resolution.md`). `celltype_localization` transports under `exp(−d²/(eps·scale))` with `eps = 0.05`, a Gaussian of `radius·√(eps·scale)` µm. Because `scale` is computed on **radius-normalised** coordinates it is dimensionless, so **`blur / radius = √(eps·scale)` is a constant of the metric — 0.25–0.30 here — not a property of this tissue.** The statistic therefore distinguishes roughly **three to four locations along a radius, on any dataset at any magnification**. In micrometres it is 186 µm on the full coronal section — which is the geometry every published score in this literature was computed on.

**residual** is what survives when the comb is convolved with that kernel — measured, not argued. It is negligible at EVERY angle, so the comb does not damage this statistic and pre-registration's F1 rule is withdrawn as unnecessary. That is not a licence: it is because the statistic cannot resolve anything below ~110 µm, which qualifies **every** localisation number in this campaign and is volunteered as such.

## θ\* = **60°**

The largest angle clearing G1 and G2 **whose evaluation set has non-zero measure**. Gate-driven and score-free: no arm has been scored when this angle is chosen.

**G1's margin at θ\* is +0.6 types.** G1 requires 60% of the coronal plane's 9 scorable types, i.e. 5.4. A margin under one whole type means **one cell type crossing the metric's own 20-cell floor moves θ\*** — printed rather than left as arithmetic, because the angle the claim is made at should not rest on a fraction of a type without the reader seeing it.

**Scored at every qualifying angle: 30°, 45°, 60°** — each with its fill printed beside its score. The curve is the result; θ\* is a label on it.

⚠️ **The counts are not monotone in angle.** At the widest angles the largest type *rises* again — the strata concentrate as the plane aligns with the depth axis, so fewer, denser teeth hold more cells of one type than a broader tilted band does. It is a comb artefact, and G1 and G2 cannot see it: they count cells and types, both of which a comb has in abundance (largest type peaks among OBLIQUE angles at 724 at 30°; the coronal row is excluded from this comparison, being a full section rather than a comb).

## Leakage preconditions (L1, L2)

| θ | check | |
|---|---|---|
| 0° | L1 — no donor coordinate coincides with a ground-truth coordinate | ✅ |
| 0° | L2 — the donor slab and the evaluation slab are disjoint by construction | ✅ |
| 0° | and neither set is empty | ✅ |
| 30° | L1 — no donor coordinate coincides with a ground-truth coordinate | ✅ |
| 30° | L2 — the donor slab and the evaluation slab are disjoint by construction | ✅ |
| 30° | and neither set is empty | ✅ |
| 45° | L1 — no donor coordinate coincides with a ground-truth coordinate | ✅ |
| 45° | L2 — the donor slab and the evaluation slab are disjoint by construction | ✅ |
| 45° | and neither set is empty | ✅ |
| 60° | L1 — no donor coordinate coincides with a ground-truth coordinate | ✅ |
| 60° | L2 — the donor slab and the evaluation slab are disjoint by construction | ✅ |
| 60° | and neither set is empty | ✅ |
| 90° | L1 — no donor coordinate coincides with a ground-truth coordinate | ✅ |
| 90° | L2 — the donor slab and the evaluation slab are disjoint by construction | ✅ |
| 90° | and neither set is empty | ✅ |

Asserted on the **returned arrays**, not argued from the call signature. The flanking-slab construction makes them disjoint by construction; these confirm that rather than enforce it.

## On the ordering

**Scoring was not run in this pass** — geometry, resolution and preconditions only; pass `--score`. θ\* and the qualifying angles are fixed by G1, G2 and F2, none of which reads a score, and they are printed above the score table for that reason. The geometry pass is run first and its θ\* published before any arm has a number; that ordering is the whole protection against choosing the angle for its score.
