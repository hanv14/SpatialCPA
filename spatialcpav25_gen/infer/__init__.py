"""Inference-side geometry, generation and calibration.

T05 needed one thing from this package — the plane (and slab) a layout is sampled on. T07
added the rest of the geometry ``specs/07`` §1 asks for: ``intersect``,
``random_plane_pair``, ``curved_surface`` and the segment / surface types the SEFL losses are
stated in. T09 adds :mod:`spatialcpav25_gen.infer.generate` (``generate_section`` and the
three multi-section wrappers) and :mod:`spatialcpav25_gen.infer.calibrate` (the leakage-free
calibrators) beside them.

**Only the geometry is re-exported here**, and that is a constraint rather than an omission:
:mod:`spatialcpav25_gen.model.layout` imports ``infer.planes`` at module scope, so importing
``infer.generate`` from this ``__init__`` would make ``model.layout -> infer.planes ->
infer/__init__ -> infer.generate -> model.layout`` a cycle at import time. Import the two T09
modules by name::

    from spatialcpav25_gen.infer.generate import generate_section
    from spatialcpav25_gen.infer.calibrate import calibrate_lengthscale
"""

from spatialcpav25_gen.infer.planes import (
    LineSegment,
    Plane,
    Surface,
    curved_surface,
    intersect,
    plane_from_normal,
    plane_pose,
    random_plane_pair,
    section_plane,
    uniform_plane_points,
    uniform_slab_points,
)

__all__ = [
    "LineSegment",
    "Plane",
    "Surface",
    "curved_surface",
    "intersect",
    "plane_from_normal",
    "plane_pose",
    "random_plane_pair",
    "section_plane",
    "uniform_plane_points",
    "uniform_slab_points",
]
