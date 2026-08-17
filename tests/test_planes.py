"""T07 acceptance tests for the plane geometry of ``specs/07`` §1.

The spec's warning is the reason this file is long and boring: "Test this against
hand-computed cases — geometry bugs here would be misdiagnosed as model failures for weeks."
Every expectation below is worked out on paper, not read off the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.planes import (
    LineSegment,
    Plane,
    curved_surface,
    intersect,
    plane_from_normal,
    plane_pose,
    random_plane_pair,
    section_plane,
)

X = np.array([1.0, 0.0, 0.0])
Y = np.array([0.0, 1.0, 0.0])
Z = np.array([0.0, 0.0, 1.0])

BBOX = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 400.0]])
THICKNESS = 25.0


def square(normal, origin, half=100.0, thickness=THICKNESS) -> Plane:
    """A square window of side ``2 * half`` on the plane with the given normal."""
    return plane_from_normal(normal, origin, (half, half), thickness)


# --------------------------------------------------------------------------------------
# intersect — the hand-computed cases specs/07 asks for
# --------------------------------------------------------------------------------------


def test_intersect_known_cases():
    """Orthogonal planes through the origin, parallel planes, and windows that miss."""
    # (a) Orthogonal planes through the origin. n1 = z, n2 = x, so the line is along
    # n1 x n2 = z x x = +y, through the origin, and both windows cap |y| at 100.
    seg = intersect(square(Z, np.zeros(3)), square(X, np.zeros(3)))
    assert seg is not None
    assert seg.length == pytest.approx(200.0)
    assert np.abs(seg.direction) == pytest.approx(np.abs(Y))
    assert np.allclose(0.5 * (seg.start + seg.end), np.zeros(3), atol=1e-9)
    # every sampled point lies on both mid-planes
    points = seg.points(16)
    assert np.allclose(points[:, 2], 0.0, atol=1e-9)
    assert np.allclose(points[:, 0], 0.0, atol=1e-9)

    # (b) Orthogonal planes offset from each other: n1 = z at z = 30, n2 = y at y = -40.
    # The line is z x y = -x, at (t, -40, 30); both windows cap |t| at 100.
    seg = intersect(square(Z, np.array([0.0, 0.0, 30.0])), square(Y, np.array([0.0, -40.0, 0.0])))
    assert seg is not None
    assert np.allclose(seg.points(8)[:, 1], -40.0, atol=1e-9)
    assert np.allclose(seg.points(8)[:, 2], 30.0, atol=1e-9)
    assert seg.length == pytest.approx(200.0)

    # (c) Parallel planes: no line at all, whatever their offset.
    assert intersect(square(Z, np.zeros(3)), square(Z, np.array([0.0, 0.0, 50.0]))) is None
    assert intersect(square(Z, np.zeros(3)), square(-Z, np.array([0.0, 0.0, 50.0]))) is None
    assert intersect(square(Z, np.zeros(3)), square(Z, np.zeros(3))) is None

    # (d) The line exists but lies outside the windows: n1 = z at the origin with a 10 um
    # window, n2 = x at x = 1000. The line is x = 1000, z = 0 — 990 um outside p1's window.
    small = square(Z, np.zeros(3), half=10.0)
    assert intersect(small, square(X, np.array([1000.0, 0.0, 0.0]))) is None

    # (e) A 45 degree pair, hand-computed. n1 = z, n2 = (0, 1, 1)/sqrt(2) through the
    # origin: the line is n1 x n2 = (-1/sqrt 2, 0, 0), i.e. the x axis.
    seg = intersect(square(Z, np.zeros(3)), square(np.array([0.0, 1.0, 1.0]), np.zeros(3)))
    assert seg is not None
    assert np.abs(seg.direction) == pytest.approx(np.abs(X), abs=1e-9)
    assert np.allclose(seg.points(5)[:, 1:], 0.0, atol=1e-9)


def test_intersect_is_order_independent():
    """The segment does not depend on which plane is named first."""
    p1 = square(Z, np.array([10.0, -5.0, 0.0]), half=120.0)
    p2 = square(np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 20.0]), half=90.0)
    a = intersect(p1, p2)
    b = intersect(p2, p1)
    assert a is not None
    assert b is not None
    ends_a = np.sort(np.stack([a.start, a.end]), axis=0)
    ends_b = np.sort(np.stack([b.start, b.end]), axis=0)
    assert np.allclose(ends_a, ends_b, atol=1e-8)


def test_intersect_points_lie_on_both_planes():
    """Every point of the segment is on both mid-planes and inside both windows."""
    gen = np.random.default_rng(7)
    for seed in range(20):
        p1, p2 = random_plane_pair(BBOX, 20.0, 160.0, seed, thickness=THICKNESS)
        seg = intersect(p1, p2)
        if seg is None:
            continue
        points = seg.points(32)
        for plane in (p1, p2):
            offset = (points - plane.origin[None, :]) @ plane.normal
            assert np.max(np.abs(offset)) < 1e-6
            # The clip endpoints sit *on* the window edge, so compare with a float64
            # tolerance rather than `contains_uv`'s exact `<=`.
            inside = np.abs(plane.to_uv(points)) <= plane.half_extent[None, :] + 1e-6
            assert bool(inside.all())
        assert gen.random() >= 0.0  # the loop above is deterministic; no RNG needed


# --------------------------------------------------------------------------------------
# LineSegment / Plane helpers
# --------------------------------------------------------------------------------------


def test_line_segment_points_are_evenly_spaced_and_inclusive():
    seg = LineSegment(start=np.array([0.0, 0.0, 0.0]), end=np.array([3.0, 4.0, 0.0]))
    assert seg.length == pytest.approx(5.0)
    points = seg.points(6)
    assert points.shape == (6, 3)
    assert np.allclose(points[0], seg.start)
    assert np.allclose(points[-1], seg.end)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert np.allclose(steps, steps[0])
    with pytest.raises(ValueError):
        seg.points(0)


def test_plane_basis_is_orthonormal_and_matches_the_normal():
    for seed in range(10):
        normal = np.random.default_rng(seed).normal(size=3)
        plane = square(normal, np.zeros(3))
        basis = plane.basis()
        assert basis.shape == (2, 3)
        assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-12)
        assert np.allclose(np.cross(basis[0], basis[1]), plane.normal, atol=1e-12)


def test_plane_sample_points_stay_in_the_slab_and_the_bbox():
    plane = plane_from_normal(np.array([0.3, -0.2, 1.0]), BBOX.mean(axis=0), (300.0, 300.0), 30.0)
    points = plane.sample_points(500, BBOX, seed=3)
    assert points.shape == (500, 3)
    offset = (points - plane.origin[None, :]) @ plane.normal
    assert np.max(np.abs(offset)) <= 0.5 * plane.thickness + 1e-9
    assert np.all(points >= BBOX[0][None, :] - 1e-9)
    assert np.all(points <= BBOX[1][None, :] + 1e-9)
    # Convention 3: same seed, same points, bitwise.
    assert np.array_equal(points, plane.sample_points(500, BBOX, seed=3))
    assert not np.array_equal(points, plane.sample_points(500, BBOX, seed=4))


def test_plane_sample_points_raises_when_the_slab_misses_the_volume():
    outside = plane_from_normal(Z, np.array([0.0, 0.0, 5000.0]), (100.0, 100.0), 10.0)
    with pytest.raises(ValueError, match="fell inside the bounding box"):
        outside.sample_points(64, BBOX, seed=0)


def test_plane_pose_takes_the_normal_to_z():
    for seed in range(8):
        normal = np.random.default_rng(seed + 100).normal(size=3)
        plane = square(normal, np.zeros(3))
        rotation = plane_pose(plane)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert np.allclose(rotation @ plane.normal, Z, atol=1e-9)
    # A coronal section is already in the pose, so its rotation is the identity.
    assert np.allclose(plane_pose(square(Z, np.zeros(3))), np.eye(3))


# --------------------------------------------------------------------------------------
# random_plane_pair
# --------------------------------------------------------------------------------------


def test_random_plane_pair_respects_the_angle_range():
    cfg = Config()
    lo, hi = cfg.sefl_min_angle_deg, cfg.sefl_max_angle_deg
    angles = []
    for seed in range(64):
        p1, p2 = random_plane_pair(BBOX, lo, hi, seed, thickness=THICKNESS)
        cosine = float(np.clip(p1.normal @ p2.normal, -1.0, 1.0))
        angles.append(math.degrees(math.acos(cosine)))
    assert min(angles) >= lo - 1e-6
    assert max(angles) <= hi + 1e-6
    # ...and the draw actually spans the range rather than sitting at one end.
    assert max(angles) - min(angles) > 0.5 * (hi - lo)


def test_random_plane_pair_is_deterministic_and_seed_dependent():
    a1, a2 = random_plane_pair(BBOX, 20.0, 160.0, 11, thickness=THICKNESS)
    b1, b2 = random_plane_pair(BBOX, 20.0, 160.0, 11, thickness=THICKNESS)
    c1, _ = random_plane_pair(BBOX, 20.0, 160.0, 12, thickness=THICKNESS)
    assert np.array_equal(a1.normal, b1.normal)
    assert np.array_equal(a2.origin, b2.origin)
    assert not np.array_equal(a1.normal, c1.normal)


def test_random_plane_pair_rejects_a_bad_range():
    with pytest.raises(ValueError, match="min_angle_deg < max_angle_deg"):
        random_plane_pair(BBOX, 160.0, 20.0, 0, thickness=THICKNESS)


def test_random_plane_pairs_usually_intersect_inside_the_volume():
    """The pair is drawn to cross inside the tissue; `loss_cross` skips the rest."""
    found = sum(
        intersect(*random_plane_pair(BBOX, 20.0, 160.0, seed, thickness=THICKNESS)) is not None
        for seed in range(50)
    )
    assert found >= 45


# --------------------------------------------------------------------------------------
# curved surfaces
# --------------------------------------------------------------------------------------


def test_curved_surface_interpolates_its_control_points():
    controls = np.array(
        [[0.0, 0.0, 100.0], [500.0, 0.0, 150.0], [0.0, 500.0, 60.0], [500.0, 500.0, 120.0]]
    )
    surface = curved_surface(controls, thickness=25.0)
    heights = surface.height(controls[:, :2])
    assert np.allclose(heights, controls[:, 2], atol=1e-6)
    assert surface.thickness == 25.0


def test_curved_surface_through_a_plane_is_that_plane():
    controls = np.array([[0.0, 0.0, 80.0], [400.0, 0.0, 80.0], [0.0, 400.0, 80.0]])
    surface = curved_surface(controls, thickness=10.0)
    query = np.array([[100.0, 100.0], [250.0, 300.0], [-50.0, 20.0]])
    assert np.allclose(surface.height(query), 80.0, atol=1e-9)
    assert np.allclose(surface.normal_at(query), np.array([0.0, 0.0, 1.0])[None, :], atol=1e-6)
    assert np.allclose(surface.point_at(query)[:, :2], query)


def test_curved_surface_bends_and_its_normal_tilts():
    controls = np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 200.0]])
    surface = curved_surface(controls, thickness=10.0)
    mid = surface.height(np.array([[200.0, 0.0]]))[0]
    assert 0.0 < mid < 200.0
    normal = surface.normal_at(np.array([[200.0, 0.0]]))[0]
    assert normal[0] < 0.0  # the surface rises with x, so the normal tilts back
    assert normal[2] > 0.0
    assert np.linalg.norm(normal) == pytest.approx(1.0)


def test_curved_surface_rejects_bad_input():
    with pytest.raises(ValueError, match="control_points"):
        curved_surface(np.zeros((0, 3)), thickness=10.0)
    with pytest.raises(ValueError, match="thickness"):
        curved_surface(np.zeros((2, 3)), thickness=0.0)


# --------------------------------------------------------------------------------------
# the T05 plane still behaves
# --------------------------------------------------------------------------------------


def test_section_plane_still_round_trips(volume):
    """T07 extended the module; T05's own construction is unchanged."""
    plane = section_plane(volume.sections[0])
    uv = plane.to_uv(np.array([[10.0, 20.0, float(volume.sections[0].z)]]))
    xyz = plane.to_xyz(uv)
    assert np.allclose(xyz, [[10.0, 20.0, float(volume.sections[0].z)]])
    assert plane.slab_volume == pytest.approx(plane.area * plane.thickness)
