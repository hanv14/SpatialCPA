# The comb limit: what serial sections bound about oblique evaluation

*A limit on the field, with the arithmetic. It applies to any method that generates or evaluates an
obliquely-oriented section from a stack of serial sections, ours included, and it is not stated
anywhere in the literature we have read.*

## The claim

**An oblique ground truth drawn from `N` serial sections has only `N` samples along depth.** At 90°
to the sectioning plane it is `N` parallel lines of cells, not a two-dimensional section — whatever
produced it, and however good the method under test is.

That bound is geometric. It holds for the best possible method and for the worst, and it cannot be
lifted by modelling. It can only be lifted by cutting the tissue differently.

## The arithmetic

Take a plane through a stack whose sections lie at spacing `s`, tilted by `θ` from the sectioning
plane, and admit the cells within a slab of thickness `t`. With the plane's normal
`n = (0, sin θ, cos θ)`, its in-plane axes are `e₁ = (1, 0, 0)` and `e₂ = (0, cos θ, −sin θ)`, so the
second in-plane coordinate of a cell is

```
v = (y − y₀)·cos θ − (z − z₀)·sin θ
```

A section sits at one `z`. Inside the slab its cells satisfy `|(y − y₀) sin θ + (z − z₀) cos θ| ≤ t/2`,
so their `y` spans a width `t / sin θ` and therefore

- **each section contributes one stratum**, of width `t·cos θ / sin θ` in `v`;
- **adjacent strata are separated by** `s / sin θ` in `v` — the `y`-centre shift `s·cos²θ / sin θ`
  and the direct `z` term `s·sin θ` summing to `s(cos²θ + sin²θ)/sin θ`.

Their ratio is the **fill**:

```
                 stratum width     t · cos θ
    fill(θ)  =  ───────────────  =  ─────────
                  separation            s
```

No free constant. Three consequences, all immediate:

1. **`fill(90°) = 0` exactly.** `v = −(z − z₀)`, which takes one value per section. The plane
   orthogonal to the sectioning plane meets the data in `N` lines.
2. **`fill` is linear in slab thickness.** Halving `t` halves the fill at every angle. This is why
   the thickness a runner assumes is not a detail (`retractions.md` R5).
3. **`fill ≥ 1` requires `t ≥ s / cos θ`**, i.e. slabs at least as thick as their spacing. Serial
   sectioning with any gap between sections cannot reach it at any oblique angle.

## What it measures, on real specimens

Two built volumes, both leakage-guarded training inputs:

| specimen | sections | `s` | `t` | fill 30° | fill 45° | fill 60° | fill 90° |
|---|---|---|---|---|---|---|---|
| `starmap_visual_cortex` | 4 | 22.0 µm | 22.0 µm | 0.87 | 0.71 | 0.50 | **0** |
| `merfish_thick_hypothalamus` | 4 | 57.5 µm | ~27 µm | 0.41 | 0.33 | 0.23 | **0** |

`merfish_thick_hypothalamus` has the better *aspect ratio* (10.3 : 1 against 21.6 : 1) and therefore
retains far more cells at an angle — but its **fill is worse**, because holding out every other
section doubles the spacing while the slabs stay ~27 µm thick. **Aspect ratio and fill are different
constraints and a specimen can be good at one and bad at the other.** Neither is reported in any
oblique-sectioning result we are aware of.

## Why the gates cannot see it

`paper_celltype_localization` skips a type with fewer than `min_gt_cells` = 20 cells and subsamples
each type to `max_n` = 250. Both are **counts**. A point cloud collapsed onto `N` lines can hold as
many cells and as many types as a filled one — `merfish_thick_hypothalamus` at 90° has 486 cells and
4 scorable types on 4 lines — so the gates pass it and the statistic is computed on a degenerate
cloud without complaint. The bounding-box aspect ratio does not see it either: it measures the
extent of the cloud, not the occupancy inside it.

**This is why the fill has to be reported as its own column.** It is not derivable from anything the
budget already prints.

## What it licenses, and what it forbids

**Comparisons between arms that reproduce real cells remain valid at any fill.** A copy baseline, a
`plane-distance` resample and a permuted-type null all place cells where real cells are, so all
three are combs with the same teeth as the ground truth. Comparing them measures the method.

**A comparison against an arm that generates a continuous fill is not valid at low fill.** A
continuous field places cells between the teeth, where the ground truth has none by construction, and
an optimal-transport metric charges it for that. At `fill ≈ 0` it is being charged for the
specimen's sampling, not for its own error. So an ablation of that kind is reported **beside the
fill, and separately from the headline**, or not at all.

This is a constraint on the experiment, not a finding about which layout is better. Stating it is
what lets the honest comparison be made.

## Our own claim, sitting inside the limit

We do not claim 90°. The comb limit says 90° is not available on a 4-section volume, and no specimen
in the cross-dataset sweep changes that — it is a property of serial sectioning, not of any
particular tissue. Our claim is made at the largest angle that clears the metric's own constants
**with the fill stated beside it**, which on present evidence is **30–45°**.

The weaker headline is the correct one. A 90° figure on a volume whose fill is zero would be a
picture of four lines.

## What would lift it

Nothing in modelling. Three things in preparation, in descending order of practicality:

1. **Thicker slabs relative to spacing.** `fill = t·cos θ / s`, so a block cut into *contiguous*
   slabs (`t = s`) doubles the fill of one cut into slabs with gaps.
2. **Not holding out alternate sections.** `paper_2_4_6` doubles `s` on the training volume for good
   leakage reasons. A holdout design that removes a contiguous run instead would keep `s` at the
   specimen's own pitch — at some cost in how far a held-out plane sits from its donors.
3. **A genuinely isotropic volume.** Light-sheet or expansion protocols that image a block rather
   than a stack have no `s` at all, and the limit does not apply to them. That is the experiment this
   analysis says the field needs, and it is not one we can run.
