"""A minimal sampling plane, for GATE 1's intersection test only.

T07 owns the real plane geometry (``spatialcpav25_gen/infer/planes.py``: ``intersect``,
``random_plane_pair``, curved surfaces, and hand-computed test cases). T03 needs exactly
one thing from it — two non-parallel planes and the line where they cross — and building
the T07 module early would be skipping ahead. This is that one thing, in the test tree,
and T07 replaces it.

The construction is deliberately *canonical* (SPEC_QUESTIONS B1): a plane's in-plane
basis is a fixed function of its normal, and turning ``(u, v)`` into ``(x, y, z)`` happens
in exactly one place. G1.2 asks two planes to produce bit-identical field values along
their intersection; that can only be meaningful if both go through the same
``origin + u * e1 + v * e2`` code, in float64, and round to float32 at the same moment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SamplingPlane:
    """A plane in R^3 with an orthonormal in-plane frame.

    Attributes
    ----------
    origin
        ``(3,)`` a point on the plane, micrometres.
    e1, e2
        ``(3,)`` orthonormal in-plane basis vectors.
    normal
        ``(3,)`` unit normal, ``e1 x e2``.
    """

    origin: FloatArray
    e1: FloatArray
    e2: FloatArray
    normal: FloatArray

    def to_xyz(self, uv: FloatArray) -> FloatArray:
        """``(N, 2)`` in-plane coordinates -> ``(N, 3)`` physical coordinates, float64."""
        uv = np.asarray(uv, dtype=np.float64)
        return self.origin[None, :] + uv[:, 0:1] * self.e1[None, :] + uv[:, 1:2] * self.e2[None, :]

    def to_uv(self, xyz: FloatArray) -> FloatArray:
        """``(N, 3)`` physical coordinates -> ``(N, 2)`` in-plane coordinates, float64."""
        centred = np.asarray(xyz, dtype=np.float64) - self.origin[None, :]
        return np.stack([centred @ self.e1, centred @ self.e2], axis=1)


def plane_from_normal(normal: npt.ArrayLike, origin: npt.ArrayLike) -> SamplingPlane:
    """Build a plane from a normal and a point on it, with a canonical in-plane basis."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    # The least-aligned axis gives a numerically stable first basis vector.
    axis = np.zeros(3)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    e1 = np.cross(n, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return SamplingPlane(
        origin=np.asarray(origin, dtype=np.float64),
        e1=e1,
        e2=e2,
        normal=np.cross(e1, e2),
    )


def intersection_line(a: SamplingPlane, b: SamplingPlane) -> tuple[FloatArray, FloatArray]:
    """Return ``(point, direction)`` of the line where two planes cross.

    Raises ``ValueError`` if the planes are parallel. ``direction`` is a unit vector;
    ``point`` is the point of the line closest to the origin of coordinates, which makes
    the parametrisation a function of the two planes alone.
    """
    direction = np.cross(a.normal, b.normal)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("intersection_line: the planes are parallel")
    direction = direction / norm
    d_a = float(a.normal @ a.origin)
    d_b = float(b.normal @ b.origin)
    # Solve [n_a; n_b; direction] x = [d_a, d_b, 0]: the unique point on the line whose
    # component along the line is zero.
    matrix = np.stack([a.normal, b.normal, direction], axis=0)
    point = np.linalg.solve(matrix, np.array([d_a, d_b, 0.0]))
    return point, direction
