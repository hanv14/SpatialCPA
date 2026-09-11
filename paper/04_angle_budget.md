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

## 4.4 Every built specimen, measured

| dataset | cells | extent (µm) | **in-plane : depth** | `s` | **budget** | cells there | first failure |
|---|---|---|---|---|---|---|---|
| `starmap_visual_cortex` | 16 527 | 1545 × 1301 × **66** | 21.6 : 1 | 22.0 µm | **5°** | 3 211 | 10° — largest type 187 < 250 (G2) |
| `deep_starmap` | 115 830 | 4385 × 4155 × **125** | 34.1 : 1 | 42.0 µm | **5°** | 14 209 | 10° — 59 of 124 types, below 60% (G1) |
| `merfish_thick_cortex` | 17 467 | 2160 × 1950 × **83** | 24.8 : 1 | 27.5 µm | **5°** | 2 791 | 10° — largest type 207 < 250 (G2) |
| **`merfish_thick_hypothalamus`** | 47 189 | 1613 × 1884 × **170** | **10.3 : 1** | 57.5 µm | **90°** ⚠️ | 1 988 | *clears every angle measured* |
| `cosmx_nsclc_3d` | — | — | — | — | *not read* | — | leakage-guarded input not built |
| `exseq_breast_cancer` | — | — | — | — | *not read* | — | leakage-guarded input not built |
| `exseq_visual_cortex` | — | — | — | — | *not read* | — | leakage-guarded input not built |
| `allen_merfish_brain` | — | — | — | — | *not read* | — | leakage-guarded input not built |

A dataset that could not be read is a named row with its reason, not a silent omission. ⚠️ **That 90° is what the sweep's `t = s` default gives; the budget this paper quotes for that specimen is 60°** — see the note at the end of §4.7.

### The result is bimodal, not graded

**Three of the four readable specimens give exactly 5°. The fourth gives 60°.** Nothing in between,
and nothing at 10° — every one of the three fails at the first angle past 5°.

**And the budget does not track the aspect ratio.** The three that stop at 5° span 21.6 : 1, 24.8 : 1
and **34.1 : 1**, in no particular order; `deep_starmap` has seven times the cells of
`starmap_visual_cortex` and the same budget. What separates the fourth specimen is not shape or
size but **depth**: 170 µm against 66, 83 and 125 — and depth is set by the preparation.

`merfish_thick_hypothalamus` is **a 200 µm block cut into seven ~28.6 µm slabs**. The other three are
stacks of thin sections. That single distinction is the whole of the difference between 5° and 60°.

**A 5° tilt on a 21.6 : 1 slab is a coronal section.** The headline dataset of this literature — and
the two next-largest — cannot carry an oblique demonstration at any angle a reader would call
oblique. Not because of any method: because there is nothing to cut.

### The two specimens the rest of this paper uses

| | `starmap_visual_cortex` | `merfish_thick_hypothalamus` |
|---|---|---|
| **in-plane : depth** | 21.6 : 1 | **10.3 : 1** |
| slab `t` | ≤ 22.0 µm (not recorded) | 28.6 µm (from the protocol) |
| **largest scorable angle** | **5°** | **60°** (90° only under the `t = s` default — see the note at §4.7) |
| fill there | ≤ 0.99 | 0.25 |

They bracket the range: the worst geometry in the table and the best.

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

## 4.7 Scope

Four built datasets were read and four could not be, for want of a built input (§4.4). The full
per-angle tables for all four readable specimens are in `reports/angle_budget.md`.

**The replication candidate was measured and does not clear.** `merfish_thick_cortex` is the closest
analogue to the specimen that works — the same holdout design, the same thick-slab preparation at
half the thickness — and it stops at **5°**, failing G2 at 10° with a largest type of 207 against
250. So the wide-angle result rests on **one specimen**, and the most likely candidate to replicate it has
been checked and does not. That is §7.1's first limitation and it is now measured rather than
anticipated.

⚠️ **The sweep behind §4.4 was run before its own correction landed, and one row does not survive
it.** Slab thickness is recorded in none of the four builds, so the runner defaulted `t` to the
section spacing on all four. A slab cannot be thicker than its own spacing, so `t = s` is the **most
generous** geometry available: every cell count, scorable-type count and `fill` in §4.4 is an
**upper bound**.

For the three 5° rows that is conservative — they fail G1 or G2 on the most generous geometry, and a
truer `t` can only make them fail harder. **For `merfish_thick_hypothalamus` it runs the other way.**
The same sweep at `t` = 13.5 µm (`reports/angle_budget_thin.json`) gives that specimen a budget of
**30°**, failing G1 at 45°. Its 90° figure is a property of the default, not of the specimen.
**The budget this paper quotes for it is 60°** — measured by §5's own run at the protocol thickness
of 28.6 µm with the stratum-width gate applied — and 90° is reported only as what the default gives.

This is the **second** time this number has been withdrawn, and the second withdrawal is the more
useful one. `retractions.md` R5 withdrew it once, on the same cause, and recorded the runner as
fixed. §4.4 reports it again because that sweep predates the fix. But a **post-fix** run of the same
specimen is also committed (`reports/angle_budget_true.json`), and it gives the same `t` = 57.5 µm and
the same 90°: the fix takes a measured thickness where the loader recorded one, and this build
records none. What the fix added was the record saying, in its own words, that the fallback
*"OVERSTATES the slab"* on a leakage-guarded input — **and that sentence sat in the artifact while the
number it qualifies was read off as a budget.** Recorded as R18, with the check that now catches it
(`angle_budget.py --audit`).

§4's argument does not rest on that number. It rests on the gap between a stack of thin sections and
a block cut into slabs, and **60° against 5° is the same gap.**
