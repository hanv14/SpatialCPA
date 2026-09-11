# Extended summary

*Not part of the assembled draft.* `scripts/assemble_draft.py` builds `DRAFT.md` from
`00_abstract.md` and §1–§8; this file is the long-form version of the abstract, kept for venues that
allow a several-hundred-word summary or a significance statement. It carries the same claims and the
same numbers as the abstract — if one is edited, edit both, and the numbers in each are quoted from
the sections rather than from the other.

Three-dimensional spatial transcriptomics is published as stacks of serial sections, so every method
that generates a "missing section" generates one parallel to the cutting plane — the only orientation
in which a ground truth exists. We make section generation well-defined at an **arbitrary position
and orientation** by carrying the volume as a continuous field: a correlated 3D noise prior, an
anatomical field with retrieval, and heads queried at coordinates rather than at section indices. A
plane becomes a set of coordinates, and "what is the section here?" has an answer at any angle.

**We then could not demonstrate that it helps, and the reasons are the contribution.** Setting out to
score an obliquely-oriented generated section against a real one, we found the evaluation itself
bounded in three independent ways, none previously stated and all computable from published constants
and a coordinate file. **The comb limit**: no obliquely-cut section exists in any dataset we know of,
so an oblique ground truth must be assembled from the cells a tilted plane passes near, giving `N`
strata covering `fill = t·cos θ / s` of the plane — exactly **0** at 90°. **The metric's resolution**:
the standard cell-type localization statistic has `blur / radius = √(eps·scale) ≈ 0.26–0.30`, a
dimensionless constant of the statistic rather than of any tissue, distinguishing three to four
locations along a radius on any dataset at any magnification. **The scrambled-section floor**: a
section whose cell types are randomised among its own cells, scored against itself, reaches
**0.03–0.24**, and this does not fall with cell count.

Because these are properties of specimens and statistics, they can be evaluated before any modelling.
Across every built dataset we could read, **three of four admit no scorable oblique angle beyond 5°**
— a tilt that is, on a 21.6 : 1 slab, a coronal section. The fourth reaches 60°, and what separates it
is depth: it is a block cut into slabs where the others are stacks of thin sections. We also show that
**holding out alternate sections halves the oblique coverage of every dataset built that way**, a
property of benchmark construction rather than of tissue, invisible to any summary that counts cells
rather than geometry.

Within those bounds, the standard metric **cannot separate our layout from the previous one at any
angle it can score**: 0.39σ at 30° and 0.52σ at 45°, with no readable difference reaching a single
standard error, against a rule fixed before the numbers existed. We report this as an explained
negative rather than a result, and we report what the demonstration does establish — that the
previous method's off-axis output spans four to five times the target plane's footprint with
three-quarters of its cells outside it, so it is not a section of that plane in the sense the metric
assumes.

Finally we re-score six published and internal methods together on the content-hash-pinned evaluator,
with a model-free copy of the flanking sections beside them. **On four of the five metrics that admit
such a floor, no method reaches it**, ours included; the competitor beats our reconstruction on five
of six, which is what our own volunteered architecture-ceiling analysis predicts; and two released
versions of one method emit **bitwise-identical predictions** under the standard holdout, so the
headline protocol of this literature cannot distinguish them at all.

We report the reconstruction deficit our method does not close, eighteen retractions of our own
claims with the evidence that overturned each, and two fabrications caught in our own drafting — an
invented number and a figure caption claiming more than its panel showed — because they are the two
failures no procedure in this work caught.
