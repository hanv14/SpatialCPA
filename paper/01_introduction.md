# 1. Introduction

Three-dimensional spatial transcriptomics is published as a stack of serial sections: a specimen is
cut along one axis, each slice is imaged, and the volume is reconstructed by registering the slices
back together. Every method that generates a "missing section" therefore generates one **parallel to
the cutting plane**, because that is the only orientation in which a ground truth exists.

This is a real limitation on what such models can be asked for. A volume reconstructed from coronal
slices cannot be queried sagittally; a model that interpolates between sections cannot be asked for a
plane through a structure at the angle an anatomist would choose. The obstacle is not the generative
model — it is that the standard formulation has no answer for a plane that is not one of the slices.
The dominant layout rule selects a donor by `|Δz|`, a quantity that names nothing for a plane
spanning the stack, and then copies that donor's whole face.

**We make section generation well-defined at an arbitrary plane** by carrying the volume as a
continuous field — a correlated 3D noise prior, an anatomical field with retrieval, and heads that
are queried at coordinates rather than at section indices (§2). A plane is then just a set of
coordinates, and the question "what is the section here?" has an answer at any position and any
orientation.

**And then we could not demonstrate that it helps.** That is the substance of this paper, and it is
worth saying at the outset what happened, because the reason is more useful than the result would
have been.

## What we found

Setting out to score an obliquely-oriented generated section against a real one, we found that the
evaluation itself is bounded in three independent ways, none of them previously stated, and all three
computable from published constants and a coordinate file:

1. **The comb limit.** There is no obliquely-cut section in any dataset we know of, so an oblique
   ground truth must be assembled from the cells a tilted plane passes near. That assembly is a set
   of `N` strata covering `fill = t·cos θ / s` of the plane, and at 90° it is `N` lines with zero
   area. **This bounds what any method can be scored against on serial sections** (§3.1).
2. **The metric's resolution.** The standard localisation statistic transports under a Gaussian whose
   width is a fixed fraction of the tissue radius — `√(eps·scale) ≈ 0.26–0.30`, dimensionless, the
   same on any dataset at any magnification. It distinguishes **three to four locations along a
   radius** (§3.2).
3. **The scrambled-section floor.** A section whose cell types are randomly permuted *among its own
   cells* — no method, no donor — scores **0.03–0.24** on that statistic, and this does not fall with
   cell count. **Any difference below roughly 0.2 is inside the range a section carrying no type
   information at all can reach** (§5.6).

They also say that most specimens cannot be evaluated obliquely at all. Across every built dataset we
could read, **three of four stop at 5°** — a tilt that is, on a 21.6 : 1 slab, a coronal section —
and the fourth reaches **60°**. What separates it is not size or shape but **depth**: it is a 200 µm
block cut into slabs, where the others are stacks of thin sections (§4.4).

Together these say that an oblique evaluation on serial-section data is not powered to separate two
plausible methods, and we show it directly: with the leak removed from the baseline and a real
precision bound attached, the difference between our layout and the previous one is **0.39σ at 30°
and 0.52σ at 45°** — not one difference reaching a single standard error (§5.5).

**This is not a success.** The capability was not demonstrated. What we can demonstrate is the gap it
was meant to close: off-axis, the previous method's output spans **four to five times the plane's own
footprint**, with roughly three-quarters of its cells at positions the plane does not pass through
(§5.3). It is a coronal face pasted onto an oblique plane. Our layout returns the cells the plane
actually cuts. The two are different objects; the standard metric cannot tell them apart.

## What this paper contributes

- **A method** that makes section generation well-defined at an arbitrary position and orientation,
  and a shipped configuration that is honest about which of its components earn their place (§2, §6).
- **Three quantified bounds on oblique evaluation** — the comb limit, the metric's resolution, and
  the scrambled-section floor — each a property of the data or the statistic rather than of any
  method (§3, §5.6).
- **A free, pre-build screen** answering *"can an oblique evaluation be done on this specimen at
  all?"* from coordinates alone, with two candidate datasets ruled out by arithmetic before anyone
  builds them (§4).
- **A fact about how this field builds its benchmarks, not about its tissue:** holding out alternate
  sections doubles the training volume's section spacing while the slabs stay as cut, so it **halves
  the oblique coverage of every dataset built that way** — measured at 0.497 on the one specimen
  where the slab thickness is recorded. A leakage-guarded volume is a factor of two worse as an
  instrument for this than the specimen it came from, and the loss is invisible to any summary that
  counts cells rather than geometry (§4.6.1).
- **A negative result reported in full**, with the reconstruction deficit our own method does not
  close (§6) and nineteen retractions of our own claims, each with the evidence that overturned it
  (§7).

## What we do not claim

That our layout is better off-axis, or worse. That the demonstration succeeded. Both would require a
resolution the available instrument does not have, and §5.6 says how much more would be needed.
