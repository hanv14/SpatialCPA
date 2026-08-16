"""Inference-side geometry and generation.

T05 needs one thing from this package — the plane (and slab) a layout is sampled on — so
that is all :mod:`spatialcpav25_gen.infer.planes` holds today. T07 adds ``intersect`` and
``random_plane_pair``, T09 adds ``generate_section`` / ``generate_stack`` /
``generate_curved`` and the calibrators.
"""

from spatialcpav25_gen.infer.planes import (
    Plane,
    plane_from_normal,
    section_plane,
    uniform_plane_points,
    uniform_slab_points,
)

__all__ = [
    "Plane",
    "plane_from_normal",
    "section_plane",
    "uniform_plane_points",
    "uniform_slab_points",
]
