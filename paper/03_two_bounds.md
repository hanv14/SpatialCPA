# 3. Two bounds on oblique evaluation

*Written before our own oblique result is shown. Both are properties of serial-section data and of
the standard evaluation metric, not of any method, and we have not found either stated.*

## 3.1 The comb limit — what an oblique ground truth can contain

Three-dimensional spatial transcriptomics is published as a stack of serial sections. There is no
obliquely-cut section in such a dataset, so an oblique ground truth must be assembled from the cells
a tilted plane passes near. **That assembly is a comb.**

For sections at spacing `s`, a slab of thickness `t` tilted by `θ`, the plane's second in-plane
coordinate is `v = (y − y₀)cos θ − (z − z₀)sin θ`. A section sits at one `z`, so inside the slab its
cells span `y` over a width `t / sin θ` and therefore occupy a single stratum of width
`t·cos θ / sin θ` in `v`, while adjacent strata are `s / sin θ` apart. The ratio is

> ### fill(θ) = t · cos θ / s

with no free constant. Three consequences:

1. **`fill(90°) = 0` exactly.** `v = −(z − z₀)`, one value per section: a plane orthogonal to the
   sectioning plane meets the data in `N` lines with zero area. **No score computed on a set of
   measure zero is a score on a section**, whatever its cell count.
2. **`fill` is linear in slab thickness.** Halving `t` halves coverage at every angle.
3. **`fill ≥ 1` requires `t ≥ s / cos θ`** — slabs at least as thick as their spacing. Serial
   sectioning with any gap cannot reach it at any oblique angle.

Measured on two built volumes, both leakage-guarded training inputs:

| specimen | sections | `s` | `t` | 30° | 45° | 60° | 90° |
|---|---|---|---|---|---|---|---|
| `starmap_visual_cortex` | 4 | 22.0 µm | 22.0 µm | 0.87 | 0.71 | 0.50 | **0** |
| `merfish_thick_hypothalamus` | 4 | 57.5 µm | 28.6 µm | 0.43 | 0.35 | 0.25 | **0** |

**Aspect ratio and fill are different constraints.** `merfish_thick_hypothalamus` has much the
better in-plane-to-depth ratio (10.3 : 1 against 21.6 : 1) and therefore retains far more cells at
an angle — and the worse fill, because a holdout design that removes alternate sections doubles `s`
while the slabs stay the thickness they were cut at. A specimen can be good at one and bad at the
other, and only one of them is visible in a cell count.

**Nothing in modelling lifts this.** Three things in preparation do: contiguous slabs (`t = s`); a
holdout that removes a contiguous run rather than alternate sections; or an imaging protocol that
captures a block rather than a stack, which has no `s` and to which the limit does not apply.

## 3.2 The resolution limit — what the standard metric can distinguish

`paper_celltype_localization`, the metric this literature scores spatial fidelity with, normalises
coordinates by the tissue radius and transports under an entropic Sinkhorn kernel
`exp(−d²/(eps·scale))` with `eps = 0.05` and `scale` the median squared inter-cell distance. Its
length scale in micrometres is `radius·√(eps·scale)` — but `scale` is computed on coordinates
**already divided by the radius**, so it is a pure number, and the ratio is what is invariant:

> ### blur / radius = √(eps · scale) ≈ 0.26 – 0.30
>
> **a constant of the metric, not a property of any tissue**

**The statistic distinguishes roughly three to four locations along a radius — on any dataset, at any
magnification, in any tissue.** It measures whether a cell type is in the right *region*, not whether
it is in the right place within it.

In micrometres, on the **full coronal sections that every published score in this literature is
computed on**, that is **186 µm against a radius of 621 µm**. (The oblique planes of §5 give
112–116 µm against radii of 425–454 µm — the same ratio on a smaller cloud.)

This qualifies every number scored with this metric, ours and everyone else's. We state it in the
dimensionless form because that is the form a reader can verify against their own data, in five
lines, without ours.

**It also disposes of an objection to §5.** One might expect a comb ground truth to penalise any
method that fills the gaps. Convolving the comb with this kernel, in closed form —
`2|sin(πf)|/(πf) · exp(−2π²σ²/p²)` — leaves a residual density modulation of **1.7 × 10⁻⁴ at 30°,
falling to 3 × 10⁻¹² at 60°** (it falls because the strata crowd together as `p = s/sin θ` shrinks).
**The metric cannot see the comb at any angle.** The two bounds are independent and do not compound:
one limits what the data can contain, the other what the statistic can distinguish, and here the
second is by far the looser.

That is not a licence for 90°. It is the reason 90° must still be refused: at `fill = 0` the only
thing that would make a score look reasonable is the instrument's inability to see that the object
is a set of lines. We take the exclusion on measure, not on the metric's opinion.
