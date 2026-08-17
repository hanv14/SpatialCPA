"""Inference-side geometry and generation.

T05 needed one thing from this package — the plane (and slab) a layout is sampled on. T07
added the rest of the geometry ``specs/07`` §1 asks for: ``intersect``,
``random_plane_pair``, ``curved_surface`` and the segment / surface types the SEFL losses
are stated in. T09 adds ``generate_section`` / ``generate_stack`` / ``generate_curved`` and
the calibrators.
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
