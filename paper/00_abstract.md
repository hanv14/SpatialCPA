# Abstract

Three-dimensional spatial transcriptomics is published as stacks of serial sections, so every method
that generates a "missing section" generates one **parallel to the cutting plane** — the only
orientation in which a ground truth exists. We make section generation well-defined at an **arbitrary
position and orientation** by carrying the volume as a continuous field, with a correlated 3D noise
prior and heads queried at coordinates rather than at section indices.

**We then could not demonstrate that it helps, and the reasons are the contribution.** Oblique
evaluation is bounded in three independent ways, none of which we have found stated, and all
computable from published constants and a coordinate file. An oblique ground truth assembled from
serial sections is `N` strata covering `fill = t·cos θ / s` of the plane — exactly **0** at 90°. The
standard localization statistic resolves `blur / radius = √(eps·scale)` ≈ **0.26–0.30** of the tissue
radius, a dimensionless constant of the statistic rather than of any tissue. And a section whose cell
types are randomised among its own cells scores **0.03–0.24** against itself, without falling with
cell count. Being properties of specimens and statistics, these are computable before any modelling:
**three of four built datasets admit no scorable oblique angle beyond 5°**, and holding out alternate
sections **halves** the oblique coverage of every dataset built that way.

Within those bounds the metric cannot separate our layout from the previous one at any angle it can
score — 0.52σ at 45°, no readable difference reaching one standard error. Re-scoring six methods
against a model-free copy of the flanking sections, **not one reaches it on four of the five metrics
that have such a floor**, ours included.
