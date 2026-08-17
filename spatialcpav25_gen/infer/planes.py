"""The plane a section is cut on, and the slab of tissue it represents.

Scope
-----
T05 samples a layout **on a plane** and integrates an intensity **over the slab** that
plane stands for, so it needs a plane type. T07 added the rest of ``specs/07`` §1 beside
it — :func:`intersect`, :func:`random_plane_pair`, :func:`curved_surface` and the
:class:`LineSegment` / :class:`Surface` types the SEFL losses are stated in.
``tests/fixtures/planes.py`` (T03's stand-in, used by GATE 1) uses the same canonical
construction and is superseded by this module for anything outside that gate.

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

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section

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

FloatArray = npt.NDArray[np.float64]

_MIN_EXTENT_UM = 1e-6
"""Smallest admissible half-extent or thickness, micrometres. A plane flatter than this
has no interior to sample and no slab to integrate over."""

_RBF_RIDGE = 1e-8
"""Ridge added to the radial-basis matrix of :func:`curved_surface`. Small enough that the
surface still passes through its control points to well under a micrometre, large enough that
two control points a nanometre apart do not make the system singular."""

_PARALLEL_TOL = 1e-9
"""Below this, two unit normals are parallel and their planes never meet. The cross product
of two unit vectors has magnitude ``sin(angle)``, so this is an angle of 6e-8 degrees — far
below any angle :func:`random_plane_pair` can draw and far above float64's noise on a
normalised cross product."""


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

    def basis(self) -> FloatArray:
        """Return the in-plane orthonormal frame ``[e1; e2]``. ``(2, 3)`` float64 (``specs/07``)."""
        return np.stack(
            [
                np.asarray(self.e1, dtype=np.float64),
                np.asarray(self.e2, dtype=np.float64),
            ],
            axis=0,
        )

    def sample_points(
        self, n: int, bbox: npt.ArrayLike, seed: int, *, cfg: Config | None = None
    ) -> FloatArray:
        """``n`` points inside slab ∩ bbox. Returns ``(n, 3)`` float64 ``(x, y, z)``.

        Parameters
        ----------
        n
            How many points to return. Exactly ``n`` come back, or the call raises: a
            consistency loss that silently received 40 points instead of 2000 would look
            like a converged loss rather than a starved one.
        bbox
            ``(2, 3)`` min/max of the volume, micrometres. The slab is a rectangular window
            swept along the normal and an oblique one leaves the tissue at its corners, so
            the returned points are the slab's intersection with the volume.
        seed
            Explicit (Convention 3): rejection sampling is stochastic and two calls with the
            same seed return bitwise identical points.
        cfg
            Supplies ``sefl_rejection_max_rounds``. Keyword-only and optional because
            ``specs/07`` §1 fixes the signature at ``(n, bbox, seed)``; ``None`` reads the
            field's default, the same escape hatch (and the same reason) as
            :func:`~spatialcpav25_gen.model.field.random_rotation`'s.

        Raises
        ------
        ValueError
            If ``sefl_rejection_max_rounds`` rounds of rejection sampling cannot fill ``n``
            points — the plane barely grazes the volume, and a loss evaluated on the handful
            of points in the corner would be noise. The message says how many were found,
            because that number is the diagnosis.
        """
        box = np.asarray(bbox, dtype=np.float64)
        if box.shape != (2, 3) or not np.all(box[1] > box[0]):
            raise ValueError(f"Plane.sample_points: bbox must be a (2, 3) min/max, got {box}")
        if n < 0:
            raise ValueError(f"Plane.sample_points: n must be >= 0, got {n}")
        generator = np.random.default_rng(seed)
        kept: list[FloatArray] = []
        found = 0
        rounds = int((cfg if cfg is not None else Config()).sefl_rejection_max_rounds)
        for _ in range(rounds):
            if found >= n:
                break
            draw = uniform_slab_points(self, max(4 * (n - found), 16), generator)
            inside = np.all((draw >= box[0][None, :]) & (draw <= box[1][None, :]), axis=1)
            kept.append(draw[inside])
            found += int(inside.sum())
        points = np.concatenate(kept, axis=0) if kept else np.zeros((0, 3), dtype=np.float64)
        if points.shape[0] < n:
            raise ValueError(
                f"Plane.sample_points: only {points.shape[0]} of {n} points fell inside the "
                f"bounding box after {rounds} rounds. The slab at origin "
                f"{np.asarray(self.origin).tolist()} with normal "
                f"{np.asarray(self.normal).tolist()} barely intersects the volume; cut closer "
                "to its centre or shrink Plane.half_extent."
            )
        return np.asarray(points[:n], dtype=np.float64)


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


@dataclass(frozen=True)
class LineSegment:
    """The segment two cutting planes share, in physical micrometres.

    Attributes
    ----------
    start, end
        ``(3,)`` float64 endpoints, micrometres.

    Notes
    -----
    This is where T07's ``L_cross`` lives. Two sections of one tissue that cross must agree
    *on the line where they cross*, and because T03's noise field is continuous in 3D both
    pathways receive the identical noise realisation there — so the loss only has to correct
    the conditioning, which is the whole reason ``L_cross`` is trainable at all.
    """

    start: FloatArray
    end: FloatArray

    def __post_init__(self) -> None:
        """Validate both endpoints are ``(3,)`` and finite."""
        for name in ("start", "end"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"LineSegment.{name} must be a finite (3,), got {value}")

    @property
    def length(self) -> float:
        """Euclidean length, micrometres."""
        return float(np.linalg.norm(np.asarray(self.end) - np.asarray(self.start)))

    @property
    def direction(self) -> FloatArray:
        """``(3,)`` unit vector from ``start`` to ``end``; zero for a degenerate segment."""
        delta = np.asarray(self.end, dtype=np.float64) - np.asarray(self.start, dtype=np.float64)
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > _MIN_EXTENT_UM else np.zeros(3, dtype=np.float64)

    def points(self, n: int) -> FloatArray:
        """``n`` evenly spaced points from ``start`` to ``end`` inclusive. ``(n, 3)`` float64.

        Deterministic on purpose, and it takes no seed because it draws nothing: the two
        branches of ``L_cross`` must be evaluated at *the same physical points* for the
        comparison to mean anything, and the cheapest way to guarantee that is for the point
        set to be a pure function of the segment.
        """
        if n < 1:
            raise ValueError(f"LineSegment.points: n must be >= 1, got {n}")
        t = np.linspace(0.0, 1.0, int(n), dtype=np.float64)[:, None]
        start = np.asarray(self.start, dtype=np.float64)[None, :]
        end = np.asarray(self.end, dtype=np.float64)[None, :]
        return np.asarray(start + t * (end - start), dtype=np.float64)


@dataclass(frozen=True)
class Surface:
    """A curved, anatomy-following cutting surface: ``z = f(x, y)`` through control points.

    Attributes
    ----------
    control_points
        ``(P, 3)`` float64 points the surface passes through, micrometres.
    thickness
        Slab thickness measured along the local normal, micrometres.
    width
        Width of the Gaussian radial basis, micrometres. Set by :func:`curved_surface` from
        the control points' own spacing, so a sparse set gives a gentle surface and a dense
        one a detailed surface without a hand-set smoothing constant.

    Notes
    -----
    A **graph** surface, not an arbitrary manifold: real "anatomy-following" sections are
    cut by tilting and bending a block that is still sectioned top-down, so ``z`` is a
    function of ``(x, y)``. That restriction is what makes ``point_at`` a closed form and
    ``normal_at`` an exact gradient rather than a numerical one. T09's ``generate_curved``
    is the consumer; T07 needs the type to exist and to interpolate exactly.
    """

    control_points: FloatArray
    thickness: float
    width: float
    coefficients: FloatArray
    """``(P,)`` float64 radial-basis coefficients, solved by :func:`curved_surface` so the
    surface passes through every control point exactly."""

    def __post_init__(self) -> None:
        """Validate the control points, the thickness and the basis width."""
        points = np.asarray(self.control_points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
            raise ValueError(
                f"Surface.control_points must be (P, 3) with P >= 1, got {points.shape}"
            )
        if np.asarray(self.coefficients).shape != (points.shape[0],):
            raise ValueError(
                f"Surface.coefficients must be ({points.shape[0]},), got "
                f"{np.asarray(self.coefficients).shape}"
            )
        if not np.isfinite(self.thickness) or self.thickness <= _MIN_EXTENT_UM:
            raise ValueError(
                f"Surface.thickness must be > {_MIN_EXTENT_UM} um, got {self.thickness!r}"
            )
        if not np.isfinite(self.width) or self.width <= _MIN_EXTENT_UM:
            raise ValueError(f"Surface.width must be > {_MIN_EXTENT_UM} um, got {self.width!r}")

    @property
    def mean_height(self) -> float:
        """Mean ``z`` of the control points: what the surface relaxes to far from all of them."""
        return float(np.asarray(self.control_points, dtype=np.float64)[:, 2].mean())

    def height(self, xy: npt.ArrayLike) -> FloatArray:
        """``(N, 2)`` in-plane micrometres -> ``(N,)`` surface height ``z``, float64.

        Gaussian radial-basis interpolation of the control heights *about their mean*:
        ``z(x) = mean + sum_i w_i exp(-|x - x_i|^2 / 2 width^2)``. Interpolating (the
        coefficients are solved, not the heights averaged) so an anatomy-following surface
        actually follows the anatomy it was given, and centred on the mean so that far from
        every control point it relaxes to the flat surface rather than to zero.
        """
        query = np.asarray(xy, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 2:
            raise ValueError(f"Surface.height expects (N, 2), got {query.shape}")
        control = np.asarray(self.control_points, dtype=np.float64)
        d2 = ((query[:, None, :] - control[None, :, :2]) ** 2).sum(axis=2)
        kernel = np.exp(-d2 / (2.0 * float(self.width) ** 2))
        return np.asarray(
            self.mean_height + kernel @ np.asarray(self.coefficients, dtype=np.float64),
            dtype=np.float64,
        )

    def point_at(self, xy: npt.ArrayLike) -> FloatArray:
        """``(N, 2)`` -> ``(N, 3)`` points on the surface, float64."""
        query = np.asarray(xy, dtype=np.float64)
        return np.concatenate([query, self.height(query)[:, None]], axis=1)

    def normal_at(self, xy: npt.ArrayLike, *, step: float = 1.0) -> FloatArray:
        """``(N, 2)`` -> ``(N, 3)`` unit normals of the graph surface, float64.

        ``n = normalise([-dz/dx, -dz/dy, 1])`` by central differences at ``step``
        micrometres. A curved surface has no single normal, which is exactly why a
        :class:`Plane` cannot represent one.
        """
        query = np.asarray(xy, dtype=np.float64)
        h = float(step)
        dx = (
            self.height(query + np.array([[h, 0.0]])) - self.height(query - np.array([[h, 0.0]]))
        ) / (2.0 * h)
        dy = (
            self.height(query + np.array([[0.0, h]])) - self.height(query - np.array([[0.0, h]]))
        ) / (2.0 * h)
        normals = np.stack([-dx, -dy, np.ones_like(dx)], axis=1)
        unit = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        return np.asarray(unit, dtype=np.float64)


def curved_surface(control_points: npt.ArrayLike, thickness: float) -> Surface:
    """Build a :class:`Surface` interpolating ``control_points`` (``specs/07`` §1).

    The basis width is the *median* nearest-neighbour spacing of the control points — their
    own scale, so a sparse set gives a gentle surface and a dense one a detailed surface with
    nothing hand-set. The coefficients solve ``K w = z - mean(z)`` with a small ridge
    (``_RBF_RIDGE``) on the diagonal, which is what keeps a near-duplicate pair of control
    points from making ``K`` singular; at that ridge the interpolation error at the controls
    is below a micrometre.
    """
    points = np.asarray(control_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError(
            f"curved_surface: control_points must be (P, 3) with P >= 1, got {points.shape}"
        )
    if points.shape[0] == 1:
        width = 1.0
    else:
        d2 = ((points[:, None, :2] - points[None, :, :2]) ** 2).sum(axis=2)
        np.fill_diagonal(d2, np.inf)
        width = float(max(np.median(np.sqrt(d2.min(axis=1))), _MIN_EXTENT_UM * 10.0))
    d2 = ((points[:, None, :2] - points[None, :, :2]) ** 2).sum(axis=2)
    kernel = np.exp(-d2 / (2.0 * width**2)) + _RBF_RIDGE * np.eye(points.shape[0])
    target = points[:, 2] - float(points[:, 2].mean())
    coefficients = np.linalg.solve(kernel, target)
    return Surface(
        control_points=points,
        thickness=float(thickness),
        width=width,
        coefficients=np.asarray(coefficients, dtype=np.float64),
    )


def plane_pose(plane: Plane) -> FloatArray:
    """Return the rotation presenting ``plane`` as *the* cutting plane. ``(3, 3)`` float64.

    The matrix taking the plane's normal to ``+z``, i.e. the frame in which this plane is
    a coronal section. It is what makes two SEFL branches *different*: the conditioning
    pathway of a section is the model queried in the pose that section was cut in, and the
    triplane feature planes are (deliberately, T04) not rotation-invariant. Everything the
    design lists as data-frame — the GRF, retrieval, the Fourier encoding — is unaffected,
    which is why the two branches see identical noise along their intersection.
    """
    # Imported here, not at module scope: `model.layout` imports this module, so a top-level
    # import of anything under `model` closes the loop and breaks `import
    # spatialcpav25_gen.infer` outright. The rotation maths belongs to T04's field module and
    # is not worth duplicating to avoid one deferred import.
    from spatialcpav25_gen.model.field import minimal_rotation

    return minimal_rotation(
        np.asarray(plane.normal, dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )


def intersect(p1: Plane, p2: Plane) -> LineSegment | None:
    """Return the segment where two mid-planes cross, clipped to both windows, or ``None``.

    Parameters
    ----------
    p1, p2
        The two planes. Only their **mid-planes** take part: a slab-slab intersection is a
        volume, and every quantity ``L_cross`` compares is defined on the mid-plane where a
        section reports its cells.

    Returns
    -------
    LineSegment or None
        ``None`` when the planes are parallel (no line at all) **or** when the line exists
        but leaves both rectangular windows without ever being inside both — the "planes
        meeting outside the bbox" case of ``specs/07``'s acceptance test. The windows are the
        clip region rather than a separate ``bbox`` argument because a :class:`Plane` already
        carries the extent it is a section of, and ``specs/07``'s signature has nowhere to
        put a third argument.

    Notes
    -----
    The line is ``{q + t d}`` with ``d = n1 x n2`` and ``q`` the point on it closest to the
    origin of the two planes' offsets, solved in closed form::

        q = ((c1 (n2.n2) - c2 (n1.n2)) n1 + (c2 (n1.n1) - c1 (n1.n2)) n2) / det
        det = (n1.n1)(n2.n2) - (n1.n2)^2

    with ``ci = ni . origin_i``. Clipping is then one 1-D interval intersection per window
    axis — four half-planes per plane, eight in total.
    """
    n1 = np.asarray(p1.normal, dtype=np.float64)
    n2 = np.asarray(p2.normal, dtype=np.float64)
    direction = np.cross(n1, n2)
    norm = float(np.linalg.norm(direction))
    if norm < _PARALLEL_TOL:
        return None
    direction = direction / norm

    c1 = float(n1 @ np.asarray(p1.origin, dtype=np.float64))
    c2 = float(n2 @ np.asarray(p2.origin, dtype=np.float64))
    n1n1, n2n2, n1n2 = float(n1 @ n1), float(n2 @ n2), float(n1 @ n2)
    det = n1n1 * n2n2 - n1n2 * n1n2
    anchor = ((c1 * n2n2 - c2 * n1n2) * n1 + (c2 * n1n1 - c1 * n1n2) * n2) / det

    lo, hi = -np.inf, np.inf
    for plane in (p1, p2):
        extent = np.asarray(plane.half_extent, dtype=np.float64)
        for axis, basis_vector in enumerate((plane.e1, plane.e2)):
            e = np.asarray(basis_vector, dtype=np.float64)
            offset = float(e @ (anchor - np.asarray(plane.origin, dtype=np.float64)))
            slope = float(e @ direction)
            limit = float(extent[axis])
            if abs(slope) < _PARALLEL_TOL:
                # The line does not move along this axis: it is either inside the slab of the
                # window for all t, or outside it for all t.
                if abs(offset) > limit:
                    return None
                continue
            t_a = (-limit - offset) / slope
            t_b = (limit - offset) / slope
            lo = max(lo, min(t_a, t_b))
            hi = min(hi, max(t_a, t_b))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return LineSegment(start=anchor + lo * direction, end=anchor + hi * direction)


def random_plane_pair(
    bbox: npt.ArrayLike,
    min_angle_deg: float,
    max_angle_deg: float,
    seed: int,
    *,
    thickness: float,
) -> tuple[Plane, Plane]:
    """Two planes through a volume whose dihedral angle lies in ``[min, max]`` degrees.

    Parameters
    ----------
    bbox
        ``(2, 3)`` min/max of the volume, micrometres. Both windows are centred on a point
        drawn from the middle half of it and are wide enough to span it, so the two planes
        cross *inside* the tissue rather than at a corner.
    min_angle_deg, max_angle_deg
        The admissible dihedral angle, ``Config.sefl_min_angle_deg`` /
        ``sefl_max_angle_deg`` at the call sites. Below the minimum the two sections are
        nearly the same cut and the consistency they assert is vacuous; above the maximum
        they are nearly the same cut again with one normal flipped.
    seed
        Explicit (Convention 3).
    thickness
        Slab thickness of both planes, micrometres. Keyword-only and required: ``specs/07``'s
        signature predates T05 making thickness part of a plane's identity, and defaulting it
        would put a hand-set section thickness in the middle of a loss.

    Returns
    -------
    tuple
        ``(p1, p2)``. Their :func:`intersect` may still be ``None`` — a chord of the volume
        can miss a window — which is the case ``loss_cross`` skips.
    """
    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (2, 3) or not np.all(box[1] > box[0]):
        raise ValueError(f"random_plane_pair: bbox must be a (2, 3) min/max, got {box}")
    if not 0.0 <= min_angle_deg < max_angle_deg <= 180.0:
        raise ValueError(
            "random_plane_pair requires 0 <= min_angle_deg < max_angle_deg <= 180, got "
            f"({min_angle_deg}, {max_angle_deg})"
        )
    gen = np.random.default_rng(seed)
    centre = 0.5 * (box[0] + box[1])
    span = box[1] - box[0]
    # A window that spans the bounding box's diagonal, so neither plane's rectangle can be
    # the thing that stops the two from meeting inside the tissue.
    half = float(np.linalg.norm(span)) * 0.5
    origins = centre[None, :] + gen.uniform(-0.25, 0.25, size=(2, 3)) * span[None, :]

    n1 = _random_unit(gen)
    angle = math.radians(float(gen.uniform(min_angle_deg, max_angle_deg)))
    # Rotate n1 by `angle` about an axis perpendicular to it: the dihedral angle between the
    # planes is then exactly the angle between the normals, by construction.
    axis = np.cross(n1, _random_unit(gen))
    while float(np.linalg.norm(axis)) < _PARALLEL_TOL:
        axis = np.cross(n1, _random_unit(gen))
    axis = axis / np.linalg.norm(axis)
    n2 = (
        n1 * math.cos(angle)
        + np.cross(axis, n1) * math.sin(angle)
        + axis * float(axis @ n1) * (1.0 - math.cos(angle))
    )
    extent = np.array([half, half], dtype=np.float64)
    return (
        plane_from_normal(n1, origins[0], extent, thickness),
        plane_from_normal(n2, origins[1], extent, thickness),
    )


def _random_unit(generator: np.random.Generator) -> FloatArray:
    """One uniformly distributed unit vector on the sphere. ``(3,)`` float64."""
    while True:
        vector = generator.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > _PARALLEL_TOL:
            return np.asarray(vector / norm, dtype=np.float64)


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
