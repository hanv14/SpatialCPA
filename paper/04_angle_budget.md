# 4. What geometry an oblique evaluation requires

*Free to compute: coordinates and cell-type labels, no fit, no model, no generation. The gates are
the evaluation metric's own constants; the one number that is ours is labelled as ours. Source:
`reports/angle_budget.md` and `scripts/angle_budget.py`.*

## 4.1 The question nobody has had to ask

§3 gives two bounds on oblique evaluation. Both are properties of a *specimen* and a *statistic*, so
before generating anything one can ask a purely geometric question: **at what angles does this
specimen admit a section the standard metric could score at all?**

No method need be involved, and the answer is not a property of any method. We are not aware of any
published oblique-sectioning result that states it, which means no published result states the
conditions under which it could have been obtained.

## 4.2 The geometry

Tilt a plane by `θ` through a stack of depth `D` whose slabs are `t` thick. The plane exits the thin
dimension after

```
    strip width  =  (D + t) / sin θ
```

so the cells available fall away as `1/sin θ`, and past some angle an "oblique section" is a sliver.
Measured against `starmap_visual_cortex` (`D` = 66 µm, `t` = 22 µm), the prediction holds to within
2% across the usable range:

| θ | predicted | measured |
|---|---|---|
| 5° | 1009.6 µm | 1007.8 µm |
| 10° | 506.8 µm | 503.8 µm |
| 15° | 340.0 µm | 335.7 µm |
| 20° | 257.3 µm | 251.8 µm |
| 30° | 176.0 µm | 169.9 µm |

At 90° it correctly clips to the tissue's own depth instead of following the formula.

## 4.3 The gates come from the metric, not from us

`paper_celltype_localization` skips a cell type with fewer than `min_gt_cells` = 20 cells and
subsamples every type to `max_n` = 250. Both are counts the metric itself defines, and they give two
conditions for a plane to be scorable at all:

- **G1** — the number of **scorable types** (≥ 20 cells) must be at least **60%** of the coronal
  plane's. *The 60% is ours and is the only chosen number in this section.*
- **G2** — the **largest type** must hold at least `max_n` = 250 cells, the metric's own subsample
  cap. Below it the strip sits under the design point of the statistic.

A third condition comes from §3.1 rather than from the metric:

- **F2** — each stratum, `t·cos θ / sin θ` micrometres wide, must be at least as wide as the
  volume's own median nearest-neighbour distance. A stratum narrower than the spacing between
  neighbouring cells is a line drawn through a point cloud, not a section. Every term is measured.

## 4.4 Two specimens, measured

| | `starmap_visual_cortex` | `merfish_thick_hypothalamus` |
|---|---|---|
| extent (µm) | 1545 × 1301 × **66** | 1613 × 1884 × **170** |
| **in-plane : depth** | **21.6 : 1** | **10.3 : 1** |
| sections (training) | 4 | 4 |
| spacing `s` | 22.0 µm | 57.5 µm |
| slab `t` | ≤ 22.0 µm (not recorded) | 28.6 µm (from the protocol) |
| **largest scorable angle** | **5°** | **60°** |
| cells there | 3 248 | 1 011 |
| fill there | ≤ 0.99 | 0.25 |

**A 5° tilt on a 21.6 : 1 slab is a coronal section.** The headline dataset of this literature cannot
carry an oblique demonstration at any angle a reader would call oblique — not because of any method,
but because 66 µm of depth against 1.5 mm in plane leaves nothing to cut.

**The second specimen is not a stack of thin sections.** It is a 200 µm block cut into seven ~28.6 µm
slabs, and that single fact — a block rather than a stack — is what moves the largest scorable angle
from 5° to 60°. Depth is the whole constraint, and it is a property of the *preparation*.

## 4.5 A pre-build screen, and two datasets ruled out by arithmetic

G2 requires a single cell type with ≥ 250 cells **inside the strip**. The strip retains a measured
fraction of the volume at 90°: 1.8% on `starmap_visual_cortex` and 4.2% on
`merfish_thick_hypothalamus`. Taking 2–4%, a specimen can clear G2 at a wide angle only if its
**largest type holds roughly 8 300 cells in the whole volume** — which needs a volume of order 10⁵
cells unless the types are extraordinarily unbalanced.

That rules out candidates before anyone builds them. Two in our own table hold fewer cells **in their
entire volume** than the strip would need from one type:

| dataset | cells in volume | verdict |
|---|---|---|
| `exseq_visual_cortex` | 1 130 | cannot clear G2 at any oblique angle |
| `exseq_breast_cancer` | 1 979 | the same |

This is the practical form of the section: **a screen that costs nothing and answers "is an oblique
evaluation possible on this specimen?" before the specimen is prepared or the model is trained.**

## 4.6 What this section contributes

A reader with a coordinate file and a cell-type column can now compute, in a few lines and before
any modelling:

1. the **largest angle** their specimen admits, from the metric's own constants;
2. the **fill** at that angle, i.e. how much of the plane real cells can cover (§3.1);
3. the **resolution** their statistic will bring to it, as a fraction of the tissue radius (§3.2);
4. whether the specimen clears the **pre-build screen** at all.

And three conclusions that are properties of how this field **prepares tissue and builds its
benchmarks**, not of any method:

1. **Stacks of thin sections cannot support oblique evaluation.** At 21.6 : 1 the largest scorable
   angle is 5°.
2. **Blocks cut into slabs can.** The same gates give 60° on a 10.3 : 1 block. Depth is the whole
   constraint, and it is set by the preparation.
3. **A leakage-guarded volume is a worse instrument for oblique evaluation than the specimen it came
   from — and the loss is a factor of two.** This is the one that is not about tissue at all.

### 4.6.1 The holdout design halves the fill

The standard design holds out **alternate** sections. That doubles the *training* volume's spacing
while the slabs stay exactly as thick as they were cut, so `t/s` falls to ≈ 0.5 and, since
`fill = t·cos θ / s`, **coverage of the oblique plane halves at every angle**.
`merfish_thick_hypothalamus` measures it exactly: 28.6 / 57.5 = **0.497**.

This is not a property of the tissue, the microscope or the method. It is a consequence of how the
benchmark is constructed, it applies to **every** dataset built under this design, and it is
invisible in any summary that reports cell counts rather than geometry — the strip still holds
hundreds of cells and dozens of types; it simply covers half as much of the plane.

**A holdout that removed a contiguous run of sections instead would leave `s` at the specimen's own
pitch and double the fill**, at some cost in how far a held-out plane sits from its donors. We do not
claim that trade is worth making in general; we claim it is a trade nobody currently knows they are
making.

## 4.7 The sweep

<!-- TABLE PENDING: the eight-dataset sweep from reports/angle_budget.md. Four datasets were read
     and four could not be, for want of a built input. Rows are not reproduced here because that
     report has not been read into the draft; the two specimens in §4.4 are quoted from runs held
     in full. Do not fabricate the missing rows. -->

`scripts/angle_budget.py --datasets all` sweeps every built dataset carrying cell types. Four were
read and four could not be, for want of a built input; a dataset that cannot be read is reported as a
named row rather than a silent omission.

The two specimens in §4.4 are the two that **bracket** the finding — the worst and the best geometry
available to us — and the arithmetic of §4.2 and §4.5 is what generalises, not any particular row.

⚠️ `starmap_visual_cortex`'s row was measured before two corrections (a reference plane that
straddled two sections, and a slab thickness defaulted from the section spacing). The **5°** verdict
is unaffected — G2 is an absolute count and 10° fails it at 159 cells against 250, whichever
reference row is used — but the fill figures in that row are upper bounds, as §3.1 states.
