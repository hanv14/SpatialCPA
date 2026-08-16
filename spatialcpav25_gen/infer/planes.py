"""The plane a section is cut on, and the slab of tissue it represents.

Scope
-----
T05 samples a layout **on a plane** and integrates an intensity **over the slab** that
plane stands for, so it needs a plane type. The full geometry of ``specs/07`` and
``specs/09`` — ``intersect``, ``random_plane_pair``, curved / anatomy-following surfaces —
is *not* here; this module is the minimum T05 uses, written so those can be added beside
it rather than on top of it. ``tests/fixtures/planes.py`` (T03's stand-in, used by GATE 1)
uses the same canonical construction and is superseded by this module for anything outside
that gate.

Why a plane carries a thickness
-------------------------------
A real section is a slab, not a surface. T05's cell count comes from integrating the
per-type intensity over the slab **volume**, not over the plane's area, and T07's
``L_thick`` is defined entirely in terms of that thickness. So thickness is a field of the
plane, not an argument that some call sites remember to pass.

Positions, though, sit on the **mid-plane**: a generated section reports in-plane ``(u, v)``
coordinates, every kNN-graph metric in the benchmark is in-plane, and the repulsion
parameters of T05 are fitted to an in-plane ``g(r)``. Count from the volume, positions on
the plane — see ``spatialcpav25_gen/model/layout.py`` for why that pair is the coherent
choice.

Canonical basis
---------------
A plane's in-plane frame is a fixed function of its normal (SPEC_QUESTIONS B1), and
``(u, v) -> (x, y, z)`` happens in exactly one place. Two planes constructed from the same
normal are therefore the same object down to the last bit, which is what T07's
intersection-consistency test needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spatialcpav25_gen.data.schema import Section

__all__ = [
    "Plane",
    "plane_from_normal",
    "section_plane",
    "uniform_plane_points",
    "uniform_slab_points",
]

FloatArray = npt.NDArray[np.float64]

_MIN_EXTENT_UM = 1e-6
"""Smallest admissible half-extent or thickness, micrometres. A plane flatter than this
has no interior to sample and no slab to integrate over."""


@dataclass(frozen=True)
class Plane:
    """A rectangular window on a plane in R^3, plus the slab thickness it represents.

    Attributes
    ----------
    origin
        ``(3,)`` float64 centre of the window, micrometres.
    e1, e2
        ``(3,)`` float64 orthonormal in-plane basis vectors.
    normal
        ``(3,)`` float64 unit normal, ``e1 x e2``.
    half_extent
        ``(2,)`` float64 half-width of the window along ``e1`` and ``e2``, micrometres.
    thickness
        Slab thickness along ``normal``, micrometres.
    """

    origin: FloatArray
    e1: FloatArray
    e2: FloatArray
    normal: FloatArray
    half_extent: FloatArray
    thickness: float

    def __post_init__(self) -> None:
        """Validate shapes and extents. Raises ``ValueError`` naming the offending field."""
        for name in ("origin", "e1", "e2", "normal"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"Plane.{name} must be (3,), got {value.shape}")
        extent = np.asarray(self.half_extent, dtype=np.float64)
        if extent.shape != (2,):
            raise ValueError(f"Plane.half_extent must be (2,), got {extent.shape}")
        if not np.all(extent > _MIN_EXTENT_UM):
            raise ValueError(
                f"Plane.half_extent must be > {_MIN_EXTENT_UM} um on both axes, got "
                f"{extent.tolist()}"
            )
        if not np.isfinite(self.thickness) or self.thickness <= _MIN_EXTENT_UM:
            raise ValueError(
                f"Plane.thickness must be > {_MIN_EXTENT_UM} um, got {self.thickness!r}; "
                "the slab volume T05 integrates over would otherwise be zero"
            )

    @property
    def area(self) -> float:
        """In-plane area of the window, micrometres^2."""
        extent = np.asarray(self.half_extent, dtype=np.float64)
        return float(4.0 * extent[0] * extent[1])

    @property
    def slab_volume(self) -> float:
        """Volume of the slab this plane represents, micrometres^3: ``area * thickness``."""
        return float(self.area * self.thickness)

    def to_xyz(self, uv: npt.ArrayLike) -> FloatArray:
        """``(N, 2)`` in-plane micrometres -> ``(N, 3)`` physical micrometres, float64."""
        coords = np.asarray(uv, dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Plane.to_xyz expects (N, 2), got {coords.shape}")
        return np.asarray(
            self.origin[None, :]
            + coords[:, 0:1] * self.e1[None, :]
            + coords[:, 1:2] * self.e2[None, :],
            dtype=np.float64,
        )

    def to_uv(self, xyz: npt.ArrayLike) -> FloatArray:
        """``(N, 3)`` physical micrometres -> ``(N, 2)`` in-plane micrometres, float64.

        The orthogonal projection onto the plane: the component along ``normal`` is
        dropped, not refused. A caller projecting cells that are off the plane is
        responsible for knowing how far off they were.
        """
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Plane.to_uv expects (N, 3), got {points.shape}")
        centred = points - self.origin[None, :]
        return np.stack([centred @ self.e1, centred @ self.e2], axis=1)

    def contains_uv(self, uv: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Test membership: ``(N, 2)`` in-plane coordinates -> ``(N,)`` bool, inside the window."""
        coords = np.asarray(uv, dtype=np.float64)
        extent = np.asarray(self.half_extent, dtype=np.float64)
        return np.asarray(np.all(np.abs(coords) <= extent[None, :], axis=1), dtype=np.bool_)


def plane_from_normal(
    normal: npt.ArrayLike,
    origin: npt.ArrayLike,
    half_extent: npt.ArrayLike,
    thickness: float,
) -> Plane:
    """Build a :class:`Plane` from a normal, a centre, a window size and a thickness.

    The in-plane basis is canonical: it is a fixed function of ``normal`` alone, so two
    calls with the same normal give bit-identical frames.
    """
    n = np.asarray(normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(n))
    if not np.isfinite(norm) or norm < _MIN_EXTENT_UM:
        raise ValueError(f"plane_from_normal: normal must be a non-zero vector, got {n.tolist()}")
    n = n / norm
    # The least-aligned axis gives a numerically stable first basis vector.
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(n)))] = 1.0
    e1 = np.cross(n, axis)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    return Plane(
        origin=np.asarray(origin, dtype=np.float64).reshape(3),
        e1=e1,
        e2=e2,
        normal=np.cross(e1, e2),
        half_extent=np.asarray(half_extent, dtype=np.float64).reshape(2),
        thickness=float(thickness),
    )


def section_plane(section: Section, *, margin: float = 0.0) -> Plane:
    """Return a real section's plane: normal ``(0, 0, 1)``, window = its in-plane bbox.

    Parameters
    ----------
    section
        The section. Its ``thickness`` becomes the plane's, so the slab integral T05 takes
        over this plane is the slab the section actually is.
    margin
        Micrometres added to each half-extent. ``0`` (the default) makes the window the
        tight bounding box of the section's cells.

    Notes
    -----
    The window is the *observed* bounding box, which under-states the tissue by about half
    a nearest-neighbour distance on each side — the outermost cell is not the edge of the
    section. That bias is shared by the observed count and by the integral it is compared
    against, so it cancels in the Poisson NLL; it does not cancel in an absolute density,
    which is why ``mean_cell_density`` documents the same caveat.
    """
    xy = np.asarray(section.coords, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(
            f"section_plane: Section.coords for section_id={section.section_id!r} must be "
            f"(N, 2), got {xy.shape}"
        )
    lo = xy.min(axis=0)
    hi = xy.max(axis=0)
    centre = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) + float(margin)
    return plane_from_normal(
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.array([centre[0], centre[1], float(section.z)]),
        half_extent=half,
        thickness=float(section.thickness),
    )


def uniform_plane_points(plane: Plane, n: int, generator: np.random.Generator) -> FloatArray:
    """``n`` uniform points on the plane's mid-plane window. Returns ``(n, 2)`` float64 uv.

    Stochastic, so the generator is explicit (Convention 3).
    """
    if n < 0:
        raise ValueError(f"uniform_plane_points: n must be >= 0, got {n}")
    extent = np.asarray(plane.half_extent, dtype=np.float64)
    return np.asarray(generator.uniform(-extent, extent, size=(int(n), 2)), dtype=np.float64)


def uniform_slab_points(plane: Plane, n: int, generator: np.random.Generator) -> FloatArray:
    """``n`` uniform points in the plane's slab. Returns ``(n, 3)`` float64 ``(x, y, z)``.

    The slab is the window swept ``+-thickness / 2`` along the normal, so the returned
    points are what a Monte-Carlo estimate of ``integral over the slab of sum_c lambda_c``
    is averaged over (T05 §1). Stochastic, so the generator is explicit (Convention 3).
    """
    uv = uniform_plane_points(plane, n, generator)
    half = 0.5 * float(plane.thickness)
    w = generator.uniform(-half, half, size=(int(n), 1))
    return np.asarray(plane.to_xyz(uv) + w * plane.normal[None, :], dtype=np.float64)
