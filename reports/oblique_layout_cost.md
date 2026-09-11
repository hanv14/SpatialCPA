# Cost — plane-distance donor selection, and what strip geometry does to it

Costing step 2 before building it, as asked. **The edit is small; the geometry is the real cost**,
and it constrains step 3 more than it constrains step 2.

## 1. The defect, precisely

```python
nearest = min(sections, key=lambda f: (abs(f.z - float(plane.origin[2])), f.section_id))
uv = np.asarray(nearest.coords_uv, dtype=np.float64)
```

Two things, not one:

1. **`plane.origin[2]` is one point's z.** For a plane spanning the stack it names nothing.
2. **It copies one whole section.** Even with a correct distance, a flat section at constant `z`
   intersects an oblique plane in a **line**; the cells within `thickness/2` of the plane form a
   *band* around that line, not a section. So a per-**section** distance still copies the wrong set.

So the fix is not a one-line substitution: it is **per-cell** point-to-plane distance,
`d_i = |(x_i - origin) . normal|`, pooled across **all** sections, keeping `d_i <= thickness/2`.
`flanking_from_section` already projects orthogonally into the plane's frame
(`plane.to_uv(to_xyz(section))`) and its own docstring says the projection is exact only "for the
parallel planes those two modes are meant for" — so the projection machinery exists and is honest
about its scope.

## 2. What it costs

| | |
|---|---|
| the edit | ~30 lines in `_resample_layout`: pool by per-cell distance, project, thin to `n_target`. Not one line |
| config | one gate, `resample_donor_selection: {"nearest-z", "plane-distance"}`, defaulting to **`nearest-z`** so nothing shipped changes |
| tests | four, and the first is the one that matters |
| fits | **zero** — `layout_mode` and its gates are fit-invariant |
| runtime | minutes |

**The no-regression test is the load-bearing one**: at a coronal plane, `plane-distance` must
reproduce `nearest-z` **bitwise**. If it does, the new rule is a strict generalisation and every
existing tier-1 number stands unchanged; if it does not, the two are different methods and every
comparison to `resample`'s 0.7546 becomes cross-construction.

## 3. What could go wrong — strip geometry at the angles STARmap allows

**This is the finding, and it bears on step 3 more than step 2.**

Tier-1 STARmap is a **slab**, not a cube. Section depths are z = 19, 30, 41, 52, 63, 74, 85 µm —
11 µm apart, **66 µm of total depth**, against an in-plane extent of order a millimetre. That is an
aspect ratio around **15:1**. (GATE 2's synthetic fixture was 3000 µm across and 400 µm deep —
**7.5:1**, twice as forgiving, and it is the only geometry oblique parity has ever been measured on.)

A plane tilted by θ from the coronal exits the thin dimension after `D / sin θ`, so the cut region is
`L x min(L, D / sin θ)` and the cells available fall away fast:

| θ | strip width `D / sin θ` | cells, if a coronal face holds ~4 200 |
|---|---|---|
| 5° | 757 µm | ~3 100 |
| 10° | 380 µm | ~1 600 |
| 15° | 255 µm | ~1 070 |
| 30° | 132 µm | ~550 |
| 45° | 93 µm | ~390 |
| 90° | 66 µm | ~280 |

**Three consequences:**

**3a. The angle budget is set by the specimen, not the method.** Past ~15–30° an oblique section is
a few hundred cells in a long thin band. `celltype_localization` needs `min_gt_cells = 20` per type
and weights by frequency, so a 300-cell strip supports only the commonest few types and the score
becomes a different quantity. **The paper's oblique figure must state the angle and the cell count
together, and the angle budget must be measured on the real volume before it is chosen.** The
in-plane extent is not committed anywhere in this repository — the 66 µm depth is, the width is not
— so the table above is a shape argument with one measured side, and the runner should print both.

**3b. An aspect guard is needed and it must be measured, not guessed.** Two bounds, both from
`Config`: a floor on cells retained in the slab, and a floor on the cut region's aspect ratio. A
plane that fails either is refused with the numbers in the message, rather than generating a sliver
that scores as though it were a section.

**3c. There is no ground truth at an oblique angle — but there is a scorable one.** The record
already says it (`progress/t09_inference_and_calibration.md`: *"On real data there is no ground
truth at an oblique angle — no real section to score against"*). The way out is the one GATE 2 used:
**an oblique plane through a real volume passes near real cells, and those cells' own measurements
are the ground truth for that plane.** The same per-cell point-to-plane pooling that step 2 builds
for donor selection **is** that evaluation set.

⚠️ **Which creates a leakage trap that step 2 must close.** If the donors are the real cells near
the plane and the ground truth is the real cells near the plane, `resample` would be copying the
answer. The pooling therefore needs an **exclusion set** — the same mechanism retrieval already uses
for its own section (GATE 2's C1) — and the no-leak test is a precondition, not a nicety.

## 4. Recommendation

Build it, with the exclusion in from the start and the coronal-bitwise test as the gate. It is
cheap, it is defined at any angle by construction, and **it makes step 1's outcome non-fatal**: if
the split says PLACEMENT, `plane-distance` resample is a route that does not depend on the intensity
field at all.

The thing to decide **before** step 3, not during it: **the angle budget**. Measure the in-plane
extent, put the table above on real numbers, and pick the largest angle that still clears the aspect
guard. If that angle turns out to be 10–15°, the paper's demonstration is "oblique within the
specimen's geometry" and should say so plainly — which is still a capability no published method
has, and is better than a 45° figure whose 390 cells cannot be scored.
