# 5. Sections generated and scored off-axis

*Every number in this section is `[PENDING]` until `scripts/oblique_demo.py --score` runs. Nothing
here may be revised on the strength of a score: θ\*, the angles, the arms, the gates, the
preconditions and the four outcomes are fixed in
`reports/oblique_demonstration_preregistration.md`, and θ\* is fixed by G1, G2 and F2 — none of
which reads a score — and printed in the geometry pass before any arm is run.*

## 5.1 The construction

There is no obliquely-cut section in any dataset we know of, so both sides of the comparison are
built from real cells and the construction has to be stated before the numbers.

**The evaluation set** is the real cells within a slab of thickness `t` about the tilted plane. §3.1
says what that can be: `N` strata, `fill = t·cos θ / s` of the plane covered.

**The donors** are a *flanking slab* — the same orientation, origin offset along the normal by one
**section spacing**. This is `flanking_copy`'s construction generalised to an arbitrary orientation:
at 0° it is exactly the adjacent sections, and at every angle it is disjoint from the evaluation set
by construction, so no arm can copy the answer it is about to be scored against.

**Three arms, all copy-based and therefore requiring no fit.** `celltype_localization` touches
generated expression only through pose estimation (§6.3), so the claim this section makes is about
layout and is measured without a model.

| arm | positions | what it tests |
|---|---|---|
| `copy-nearest-z` | the nearest section's **whole face**, pasted on the plane | the previous method off-axis. Its footprint is the section's, not the plane's. **The baseline.** |
| `resample-pd` | the cells the flanking slab **actually contains** | the plane's own footprint. **Ours.** |
| `null` | `resample-pd`'s positions, types permuted | the floor, and the precondition that the metric is responding at all |

**That contrast is the contribution.** `nearest-z` selects a donor by `|Δz|`, a quantity that names
nothing for a plane spanning the stack, and can only paste a coronal face onto an oblique plane. A
continuous field makes the selection well-defined at any orientation, and what it selects is the
cells the plane passes through.

## 5.2 The angle: θ\* = `[PENDING]`, expected 60°

Chosen by three criteria, none of which reads a score:

- **G1** — scorable types ≥ 60% of the coronal plane's. *The 60% is ours and is labelled as ours;
  the 20-cell floor it counts against is the metric's.*
- **G2** — the largest type ≥ `max_n` = 250, the metric's own subsample cap.
- **F2** — the stratum `t·cos θ / sin θ` must be at least the volume's median nearest-neighbour
  distance. A stratum narrower than the spacing between neighbouring cells is a line drawn through a
  point cloud, not a section.

On `merfish_thick_hypothalamus` F2 admits 30°, 45°, 60° and 70°, and excludes 75°, 85° and 90°.
**We report every qualifying angle with its fill beside its score. The curve is the result; θ\* is a
label on it.**

⚠️ **G1's margin at 60° is 0.6 of one cell type** — six scorable types against a threshold of 5.4.
One type crossing the metric's 20-cell floor moves θ\*. It is printed in the table for that reason.

## 5.3 Results

| θ | fill | `copy-nearest-z` | `resample-pd` | difference | across-seed spread | `null` |
|---|---|---|---|---|---|---|
| 30° | 0.43 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 45° | 0.35 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |
| 60° | 0.25 | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` | `[PENDING]` |

Three generation seeds, reported per seed in the underlying record; the median and the full
across-seed spread are shown. **Verdict: `[PENDING]`**, one of the four fixed in §6 of the
pre-registration:

- **DEMONSTRATED** — `resample-pd(θ*) ≥ copy-nearest-z(θ*) − 0.05`.
- **DEMONSTRATED WITH A COST** — below that band. The price is a number in the abstract.
- **PARTIAL** — θ\* fails a precondition; a smaller qualifying angle carries the claim.
- **NOT DEMONSTRATED** — no qualifying angle passes. This section becomes a negative result and §3
  and §4 carry the paper.

## 5.4 What this section does not establish

- **Not a reconstruction claim.** On-axis the method does not reach the copy floor (§6.1), and this
  section does not argue that it does.
- **Not about expression.** The metric is nearly blind to the expression head; this is a layout
  result (§6.3).
- **Bounded by §3 on both sides.** The evaluation set covers `fill` of the plane, and the statistic
  resolves ~0.26–0.30 of the tissue radius. Both bounds are printed in the same table as the scores
  rather than deferred to a limitations paragraph.
- **One specimen.** `merfish_thick_cortex` is the replication candidate and has not been built
  (§7.1).
