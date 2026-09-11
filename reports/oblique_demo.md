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

## Scores

All arms are **copy-based and fit-free**, and all draw **only from cells outside the evaluation slab** — so neither side holds any of the answer. The first scored run did not have that property: the baseline emitted the whole section the plane passes through, and roughly a quarter of the ground truth was present in it verbatim (`reports/retractions.md` R11).

- `copy-nearest-z` — the previous method off-axis: the nearest section's face pasted onto the plane, **minus the cells inside the evaluation slab**. Its footprint is the section's, not the plane's. **The baseline.**
- `resample-pd` — the cells the flanking slab contains: the plane's own footprint. **Ours.**
- `null` — `resample-pd`'s positions with types permuted. P2's arm-side floor.

### P2's gate, settled before these scores existed (§5-ter)

The ground truth's types were permuted **among its own cells** and scored against itself — no method, no arm, no donor — so whatever it reports is a property of the statistic at that cell count.

⚠️ **The hypothesis this test was built on is REFUTED.** I predicted the floor was small-`n` noise and would fall as `n` rose. It does not fall at any angle: it is as high at the full sample as at 250 cells, so `G2` constraining only the largest type is not the mechanism (`retractions.md` R17).

**And the finding is larger than the thing it was testing: a section whose cell types have been completely scrambled scores well above zero.** That is a property of `celltype_localization` itself — see `reports/metric_resolution.md`.

| n | 30° | 45° | 60° |
|---|---|---|---|
| 250 | 0.0980 ± 0.0457 | 0.0532 ± 0.0949 | 0.0849 ± 0.0637 |
| 500 | 0.2360 ± 0.1125 | 0.0822 ± 0.0757 | 0.0338 ± 0.1142 |
| 1000 | 0.1070 ± 0.0618 | 0.0897 ± 0.0494 | 0.0891 ± 0.0992 |
| 1011 | — | — | 0.1305 ± 0.0573 |
| 1311 | — | 0.0333 ± 0.0622 | — |
| 1906 | 0.1714 ± 0.1129 | — | — |

Spreads are over three seeds and are comparable to the values themselves, so these medians are not precisely placed. **Each angle's ceiling comes from its own calibration at its own `n`** — the first version resolved one ceiling from the first angle and applied it to all three (`retractions.md` R15).

- **30°** (n = 1906): self-null +0.1714 > 0.05: the metric has a noise floor at this n, so P2's ceiling is replaced by self-null + spread = 0.2843. **The original 0.10 was mis-set by me**, not failed by the arms
- **45°** (n = 1311): self-null +0.0333 ≤ 0.05: the metric has no material noise floor here, so P2's original 0.10 ceiling stands and an arm above it has really failed
- **60°** (n = 1011): self-null +0.1305 > 0.05: the metric has a noise floor at this n, so P2's ceiling is replaced by self-null + spread = 0.1878. **The original 0.10 was mis-set by me**, not failed by the arms

### The footprint: is each arm a section of *this* plane? (§5-quinquies)

Pre-registered **before the measurement was written**, with its bands fixed and the middle band defaulting against us. `u` is the comb axis — the narrow one — and the ground truth *is* the section, so its `u`-range is the plane's own footprint. A cell outside it claims to be a cell of the section where the section does not exist.

| θ | plane's footprint | `copy-nearest-z` | ×  | outside | `resample-pd` | × | outside | verdict |
|---|---|---|---|---|---|---|---|---|
| 30° | 390 µm | 1626 µm | **4.17** | **71%** | 390 µm | 1.00 | 35% | **OUTSIDE** |
| 45° | 270 µm | 1328 µm | **4.92** | **75%** | 270 µm | 1.00 | 22% | **OUTSIDE** |
| 60° | 213 µm | 939 µm | **4.40** | **72%** | 213 µm | 1.00 | 22% | **OUTSIDE** |

- **30°** — the baseline spans 4.2x the plane's own footprint and 71% of its cells lie outside it entirely — it is **not a section at this angle**
- **45°** — the baseline spans 4.9x the plane's own footprint and 75% of its cells lie outside it entirely — it is **not a section at this angle**
- **60°** — the baseline spans 4.4x the plane's own footprint and 72% of its cells lie outside it entirely — it is **not a section at this angle**

### Pose (diagnostic, **not** a gate — §5-quater)

Dropped as a precondition by the project author: `evaluate_paper` aligns every prediction independently, so **every published number in this benchmark is already cross-pose** and P4 would hold this comparison to a standard nothing else meets. I raised the argument and noted that it favours us; the author made the call. Reported here because of what it shows.

| θ | `copy-nearest-z` | `resample-pd` | separation |
|---|---|---|---|
| 30° | -18.00° | 174.00° | 168.00° |
| 45° | 0.00° | 39.19° | 39.19° |
| 60° | -3.00° | -11.89° | 8.89° |

**The third bound on oblique evaluation.** An oblique strip is a **ribbon**; a ribbon maps onto itself under a half-turn, so `align_by_expression` has two near-equivalent optima and picks between them arbitrarily. `resample-pd` aligned at 174° at 30° in two runs whose baselines differed, so it is a property of the arm's shape against the ground truth, not of the comparison. **Expression-based alignment is underdetermined on elongated point clouds** — which is what every oblique evaluation set is. (Separations are wrapped to [0°, 180°]; an earlier run printed 192°, which is 168° the other way.)

### Scores

Intervals are a **leave-one-cell-type-out jackknife**: the point estimate is the full-sample score, **untouched by construction**, which is the property the cell bootstrap lacked — at 45° its median sat 0.118 above the estimate it was meant to bracket (`retractions.md` R14). Generation seeds cannot supply one: both compared arms are deterministic and their across-seed spread is exactly 0.0000 (R12).

| θ | fill | `copy-nearest-z` | `resample-pd` | difference | ± (ours) | `null` |
|---|---|---|---|---|---|---|
| 30° | 0.43 | +0.2485 | **+0.1302** | -0.1183 | ± 0.1180 (9 types) | +0.1424 |
| 45° | 0.35 | +0.3340 | **+0.1167** | -0.2173 | ± 0.2841 (9 types) | +0.0290 |
| 60° | 0.25 | +0.4516 | **+0.4054** | -0.0462 | ± 0.5477 (9 types) | +0.2461 |

### Preconditions — an angle failing any of them is NOT READABLE

| θ | precondition | |
|---|---|---|
| 30° | L1 — `copy-nearest-z` shares no coordinate with the ground truth | ✅ |
| 30° | and `copy-nearest-z` is not empty | ✅ |
| 30° | L1 — `resample-pd` shares no coordinate with the ground truth | ✅ |
| 30° | and `resample-pd` is not empty | ✅ |
| 30° | P2 — the permuted-type null is +0.1424 against a ceiling of 0.2843 | ✅ |
| 45° | L1 — `copy-nearest-z` shares no coordinate with the ground truth | ✅ |
| 45° | and `copy-nearest-z` is not empty | ✅ |
| 45° | L1 — `resample-pd` shares no coordinate with the ground truth | ✅ |
| 45° | and `resample-pd` is not empty | ✅ |
| 45° | P2 — the permuted-type null is +0.0290 against a ceiling of 0.1000 | ✅ |
| 60° | L1 — `copy-nearest-z` shares no coordinate with the ground truth | ✅ |
| 60° | and `copy-nearest-z` is not empty | ✅ |
| 60° | L1 — `resample-pd` shares no coordinate with the ground truth | ✅ |
| 60° | and `resample-pd` is not empty | ✅ |
| 60° | P2 — the permuted-type null is +0.2461 against a ceiling of 0.1878 | ❌ **FAILED** |

### **PARTIAL**

θ\* = 60° fails a precondition above, so **no score at that angle is readable** — the verdict may not outrank a precondition the report has already printed (`retractions.md` R13). The largest angle passing **every** precondition is **45°**, where `resample-pd` scores +0.1167 against the baseline's +0.3340 (-0.2173).

The verdict, the band and the outcomes were fixed in `reports/oblique_demonstration_preregistration.md` §6 before any arm was scored; θ\* was fixed by G1, G2 and F2 before that; and §0-bis records, before the repairs were made, that **a corrected comparison may still show `resample-pd` losing** — in which case §5 becomes a negative section and the two bounds carry the paper.

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

Scoring was run in this pass. θ\* and the qualifying angles are fixed by G1, G2 and F2, none of which reads a score, and they are printed above the score table for that reason. The geometry pass is run first and its θ\* published before any arm has a number; that ordering is the whole protection against choosing the angle for its score.
