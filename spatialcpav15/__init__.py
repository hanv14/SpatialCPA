"""SpatialCPA-v15 — dense 3D atlas reconstruction from sparse 2D sections.

A complete implementation of ``spatialcpa_v15_idea.md``:

* **Phase 1** — QC, a *continuous hierarchical type embedding* built from signed
  marker effect sizes (data-driven pseudobulk backbone blended per type with an
  encoder prior under a sample-size shrinkage weight), a common frame, normalized
  anatomical coordinates and a 3D neighbourhood graph with explicit edge ``dz``.
* **Phase 2** — the structure field ``lambda_c(x, y, z)`` as commensurate density
  channels plus density-weighted mean type-embedding channels; a **flow-free,
  query-based** conditional diffusion interpolator trained on
  ``(I_{a-1}, I_a, I_b, I_{b+1}, t, dz) -> I_t`` tuples; deterministic DDIM
  multi-query with fixed noise per seed; and a machine-checked stop condition
  against linear interpolation.
* **Phase 3** — an NB expression VAE used as a *likelihood*, and a conditional
  latent diffusion over its frozen latent conditioned on the type embedding,
  position, ``dz`` and cross-attention over real bracketing cells.
* **Phase 4** — layout as a marked spatial point process, joint-block generation,
  decoding to counts, and mandatory per-cell provenance.

Nothing here is shared with SpatialCPA-v8: v8 is a training-free entropic-OT
method (barycentric/McCann bridges, smoothed-OT displacement morphs,
velocity-field advection of a single anchor slice, an OT expression fusion and a
niche MRF annotator).  v15 shares none of those components — it is a learned
generative pipeline whose estimand is the completed field, not a transported
slice.
"""

from . import runtime
from .config import (ExprDiffusionConfig, ExprVAEConfig, FieldConfig,
                     FrameConfig, HierarchyConfig, InferenceConfig,
                     LayoutConfig, QCConfig, RuntimeConfig,
                     SpatialCPAv15Config, StructureConfig, TypeConfig)
from .data import Slice, SliceStack, VirtualSlice, build_stack_from_arrays
from .generator import SpatialCPAv15

__version__ = "15.0.0"

__all__ = [
    "SpatialCPAv15", "SpatialCPAv15Config", "Slice", "SliceStack",
    "VirtualSlice", "build_stack_from_arrays",
    "QCConfig", "TypeConfig", "HierarchyConfig", "FrameConfig", "FieldConfig",
    "StructureConfig", "LayoutConfig", "ExprVAEConfig", "ExprDiffusionConfig",
    "InferenceConfig", "RuntimeConfig", "runtime", "__version__",
]
