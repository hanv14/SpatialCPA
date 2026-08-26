"""T04 — the multi-orientation triplane anatomical field, the Fourier encoding and rotations.

The spec's "Other acceptance tests" for `model/field.py`, plus the GATE 2 criteria that
`tests/gate2_criteria.py` measures and `reports/gate2.md` prints. The gate tests are marked
``slow`` and ``gate``; everything else runs in ``make test``.

The rotation contract these tests assert is stated in full in
``spatialcpav25_gen/model/field.py``'s docstring and in ``SPEC_QUESTIONS.md`` B5. Its two
halves, both asserted below:

* the data-frame channels — the Fourier encoding, the GRF query, the retrieval query — are
  **invariant** to the augmentation rotation;
* the triplane lookup is **not**, and ``test_rotation_augmentation_is_not_inert`` is the
  negative control that says so. A triplane that undid the rotation would satisfy the
  spec's literal wording and make the augmentation an exact no-op.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config, ConfigError
from spatialcpav25_gen.data.schema import Volume
from spatialcpav25_gen.model.field import (
    ROTATION_CHANNELS,
    BBoxClampWarning,
    RotationContext,
    RotationError,
    TriplaneField,
    fourier_encode,
    fourier_encode_dim,
    orientation_rotations,
    random_rotation,
)
from spatialcpav25_gen.model.noise import GaussianRandomField

from tests.fixtures.synthetic import make_synthetic_volume
from tests.gate2_criteria import (
    ANGLES_DEG,
    measure_augmentation_completeness,
    measure_g2_1,
    measure_g2_2,
    measure_g2_3,
    measure_g2_4,
    measure_ratio_stability,
)

# A deliberately small field: these tests are about the contract, not about capacity, and
# the fast suite must stay under three minutes on CPU (Convention 7). The default
# 256 x 256 x 32 planes at four orientations are 10.5 M parameters; this is 200 k.
SMALL_FIELD = {
    "triplane_res_xy": 64,
    "triplane_res_z": 16,
    "triplane_channels": 8,
}
PROBE_POINTS = 512
PROBE_SEED = 41


@pytest.fixture(scope="module")
def small_cfg() -> Config:
    """The small-field config every fast test in this module uses."""
    return Config(**SMALL_FIELD)


@pytest.fixture(scope="module")
def field(small_cfg: Config, volume: Volume) -> TriplaneField:
    """A field over the synthetic volume, with feature planes carrying real structure.

    The planes are filled with unit-variance noise rather than left at their small
    initialisation: several of these tests measure whether the field *responds* to
    something, and a field whose planes are all ~1e-2 responds to everything by ~1e-2,
    which would let a broken implementation pass by being uniformly flat.
    """
    model = TriplaneField(small_cfg, volume.bbox)
    generator = torch.Generator().manual_seed(PROBE_SEED)
    with torch.no_grad():
        for planes in (model.plane_xy, model.plane_xz, model.plane_yz):
            planes.normal_(0.0, 1.0, generator=generator)
    return model


@pytest.fixture(scope="module")
def probe_points(volume: Volume) -> np.ndarray:
    """``(PROBE_POINTS, 3)`` points strictly inside the volume's bounding box."""
    lo, hi = volume.bbox[0].astype(np.float64), volume.bbox[1].astype(np.float64)
    inset = lo + 0.1 * (hi - lo)
    extent = 0.8 * (hi - lo)
    gen = np.random.default_rng(PROBE_SEED)
    return (inset[None, :] + gen.random((PROBE_POINTS, 3)) * extent[None, :]).astype(np.float32)


# --------------------------------------------------------------------------------------
# the Fourier encoding
# --------------------------------------------------------------------------------------


def test_fourier_encode_shape_and_range(small_cfg: Config) -> None:
    gen = np.random.default_rng(0)
    xyz = torch.from_numpy(gen.uniform(-1.0, 1.0, size=(64, 3)).astype(np.float32))
    out = fourier_encode(xyz, small_cfg)
    assert out.shape == (64, fourier_encode_dim(small_cfg))
    assert out.shape[1] == 2 * (2 * small_cfg.fourier_bands_xy + small_cfg.fourier_bands_z)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) <= 1.0


def test_fourier_encoding_is_anisotropic(small_cfg: Config) -> None:
    """B_z < B_xy: the z axis gets strictly fewer bands than the in-plane axes.

    This is the whole justification for the encoding being anisotropic - with 4-9 sections
    the high z frequencies are unconstrained - so it is asserted rather than assumed.
    """
    assert small_cfg.fourier_bands_z < small_cfg.fourier_bands_xy
    columns_per_axis = 2 * small_cfg.fourier_bands_xy
    total = fourier_encode_dim(small_cfg)
    assert total == 2 * columns_per_axis + 2 * small_cfg.fourier_bands_z


def test_fourier_axis_follows_data_frame(
    small_cfg: Config, field: TriplaneField, probe_points: np.ndarray, volume: Volume
) -> None:
    """**The §1 silent bug.** A fixed cell's encoding is invariant to the augmentation.

    The low-frequency axis must follow the *tissue's* sectioning axis, not the model's
    current frame. So the encoding is computed on the data-frame position, before the
    augmentation rotation - and a cell that has not moved must therefore encode to exactly
    the same vector however the volume is posed.
    """
    centre = volume.bbox.mean(axis=0).astype(np.float64)
    baseline = field._normalised_data_frame(torch.from_numpy(probe_points))
    encoded = fourier_encode(baseline, small_cfg)

    for seed in (1, 2, 3):
        rotation = random_rotation(seed)
        with RotationContext(
            small_cfg, rotation, centre, requires=("coords",), fields=(field,)
        ) as rot:
            moved = torch.from_numpy(rot.coords(probe_points).astype(np.float32))
            rotated_encoding = fourier_encode(field._normalised_data_frame(moved), small_cfg)
        # 1e-3, not zero: coordinates are float32 (Convention 4), so a rotation and its
        # inverse do not round-trip exactly (~4e-5 um here) and the highest band multiplies
        # the normalised coordinate by 2^7 pi. Measured gap ~9e-5; an encoding that had
        # followed the model frame would differ by O(1).
        assert torch.allclose(rotated_encoding, encoded, atol=1e-3), (
            "the Fourier encoding moved with the augmentation rotation: its low-frequency "
            "axis is following the model frame, not the tissue's sectioning axis"
        )


def test_fourier_encoding_is_not_constant_along_z(small_cfg: Config) -> None:
    """``fourier_bands_z = 2`` is low, not zero: depth still has to be encoded."""
    base = torch.zeros((8, 3))
    base[:, 2] = torch.linspace(-1.0, 1.0, 8)
    encoded = fourier_encode(base, small_cfg)
    assert float(encoded.std(dim=0).max()) > 0.1


# --------------------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("bias", ["uniform", "axial"])
def test_random_rotation_is_a_proper_rotation(bias: str) -> None:
    for seed in range(8):
        rotation = random_rotation(seed, bias)
        assert rotation.shape == (3, 3)
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_random_rotation_is_deterministic() -> None:
    assert np.array_equal(random_rotation(7), random_rotation(7))
    assert not np.array_equal(random_rotation(7), random_rotation(8))


def test_axial_bias_stays_near_the_sectioning_axis() -> None:
    """``axial`` tilts by at most ``rotation_bias_max_tilt_deg``; ``uniform`` does not."""
    cap = math.radians(Config().rotation_bias_max_tilt_deg)
    pole = np.array([0.0, 0.0, 1.0])
    axial = [
        math.acos(np.clip((random_rotation(s, "axial") @ pole) @ pole, -1, 1)) for s in range(64)
    ]
    assert max(axial) <= cap + 1e-9
    uniform = [
        math.acos(np.clip((random_rotation(s, "uniform") @ pole) @ pole, -1, 1)) for s in range(64)
    ]
    assert max(uniform) > cap


def test_random_rotation_rejects_unknown_bias() -> None:
    with pytest.raises(ValueError, match="bias must be one of"):
        random_rotation(0, "tetrahedral")


def test_orientation_set_is_deterministic_and_starts_unrotated() -> None:
    """The ensemble is fixed at init, not random, and orientation 0 is the identity.

    Orientation 0 has to be the unrotated set: it is the only one whose third axis is the
    tissue's sectioning axis, and therefore the only one ``tv_z_penalty`` can meaningfully
    smooth along z.
    """
    for n in (1, 4, 8):
        first = orientation_rotations(n)
        assert first.shape == (n, 3, 3)
        assert np.array_equal(first, orientation_rotations(n))
        assert np.allclose(first[0], np.eye(3))
        for rotation in first:
            assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)


def test_orientation_set_is_spread_over_directions() -> None:
    """Four orientations must actually point in four different directions."""
    pole = np.array([0.0, 0.0, 1.0])
    axes = np.stack([rotation.T @ pole for rotation in orientation_rotations(4)])
    cosines = axes @ axes.T
    off_diagonal = cosines[~np.eye(4, dtype=bool)]
    assert float(off_diagonal.max()) < 0.95, "two orientations are nearly the same direction"


# --------------------------------------------------------------------------------------
# RotationContext
# --------------------------------------------------------------------------------------


def test_rotation_context_round_trips(small_cfg: Config, probe_points: np.ndarray) -> None:
    rotation = random_rotation(11)
    centre = probe_points.mean(axis=0).astype(np.float64)
    context = RotationContext(small_cfg, rotation, centre)
    model = context.to_model(probe_points)
    assert np.abs(context.to_data(model) - probe_points).max() < 1e-6
    # Rotating about the centre, not the origin: the centre must not move.
    assert np.abs(context.to_model(centre[None, :]) - centre[None, :]).max() < 1e-9


def test_rotation_context_refuses_to_forget_a_channel(
    small_cfg: Config, probe_points: np.ndarray
) -> None:
    """Rotating the coordinates and forgetting the rest is the bug the context exists to stop."""
    centre = probe_points.mean(axis=0).astype(np.float64)

    def forget_a_channel() -> None:
        with RotationContext(small_cfg, random_rotation(12), centre) as rot:
            rot.coords(probe_points)

    with pytest.raises(RotationError, match="un-rotated"):
        forget_a_channel()

    # Naming the channels this caller actually has is allowed, and explicit.
    with RotationContext(small_cfg, random_rotation(12), centre, requires=("coords", "grf")) as rot:
        model = rot.coords(probe_points)
        rot.grf(model)


def test_rotation_context_reports_all_four_channels(
    small_cfg: Config, probe_points: np.ndarray
) -> None:
    centre = probe_points.mean(axis=0).astype(np.float64)
    with RotationContext(small_cfg, random_rotation(13), centre) as rot:
        model = rot.coords(probe_points)
        rot.planes(probe_points[:4], np.tile([0.0, 0.0, 1.0], (4, 1)))
        rot.retrieval(model)
        rot.grf(model)
        assert set(rot.used) == set(ROTATION_CHANNELS)


def test_rotation_context_does_not_mask_the_bodys_own_error(
    small_cfg: Config, probe_points: np.ndarray
) -> None:
    """An exception inside the block must survive, not be replaced by "channel missing"."""
    centre = probe_points.mean(axis=0).astype(np.float64)

    def fail_inside() -> None:
        with RotationContext(small_cfg, random_rotation(14), centre) as rot:
            rot.coords(probe_points)
            raise ZeroDivisionError("the body's own failure")

    with pytest.raises(ZeroDivisionError):
        fail_inside()


def test_rotation_context_restores_the_field(
    small_cfg: Config, field: TriplaneField, probe_points: np.ndarray, volume: Volume
) -> None:
    centre = volume.bbox.mean(axis=0).astype(np.float64)
    before = field(probe_points).detach().clone()
    with RotationContext(
        small_cfg, random_rotation(15), centre, requires=("coords",), fields=(field,)
    ) as rot:
        rot.coords(probe_points)
    assert torch.equal(field(probe_points), before)
    assert np.allclose(field.rotation_matrix, np.eye(3))


def test_rotation_context_random_honours_rotation_aug(
    small_cfg: Config, probe_points: np.ndarray
) -> None:
    centre = probe_points.mean(axis=0).astype(np.float64)
    off = RotationContext.random(small_cfg.replace(rotation_aug=False), 3, centre)
    assert off.is_identity
    on = RotationContext.random(small_cfg.replace(rotation_aug=True), 3, centre)
    assert not on.is_identity


def test_rotation_context_rejects_a_reflection(small_cfg: Config) -> None:
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="proper"):
        RotationContext(small_cfg, reflection, np.zeros(3))


# --------------------------------------------------------------------------------------
# the rotation contract (SPEC_QUESTIONS B5)
# --------------------------------------------------------------------------------------


def test_rotation_equivariance(small_cfg: Config, volume: Volume, probe_points: np.ndarray) -> None:
    """Rotate the world, query at the rotated point, and the data-frame channels agree.

    The half of the contract that holds by construction, at the spec's 1e-3 relative
    tolerance: the GRF value, the Fourier encoding and the retrieval query point are all
    functions of the *tissue* position, so a rotation cannot change them. This is the test
    that catches a ``RotationContext`` that transforms one channel the wrong way round -
    the failure mode with no error message.
    """
    from spatialcpav25_gen.model.retrieval import RetrievalIndex

    centre = volume.bbox.mean(axis=0).astype(np.float64)
    grf = GaussianRandomField(small_cfg, (small_cfg.ell_xy, small_cfg.ell_xy, small_cfg.ell_z), 5)
    index = RetrievalIndex(volume, small_cfg)
    model = TriplaneField(small_cfg, volume.bbox)

    reference_noise = grf(probe_points.astype(np.float32)).detach().numpy()
    reference_idx, reference_w = index.query(probe_points.astype(np.float64), set(), seed=0)
    reference_encoding = fourier_encode(
        model._normalised_data_frame(torch.from_numpy(probe_points)), small_cfg
    )

    for seed in (21, 22, 23):
        with RotationContext(small_cfg, random_rotation(seed), centre, fields=(model,)) as rot:
            moved = rot.coords(probe_points)
            noise = grf(rot.grf(moved).astype(np.float32)).detach().numpy()
            idx, weights = index.query(rot.retrieval(moved), set(), seed=0)
            rot.planes(probe_points[:2], np.tile([0.0, 0.0, 1.0], (2, 1)))
            encoding = fourier_encode(
                model._normalised_data_frame(torch.from_numpy(moved.astype(np.float32))),
                small_cfg,
            )

        scale = float(np.abs(reference_noise).mean())
        assert np.abs(noise - reference_noise).max() < 1e-3 * scale + 1e-4
        assert np.array_equal(idx, reference_idx)
        assert np.abs(weights - reference_w).max() < 1e-9
        assert torch.allclose(encoding, reference_encoding, atol=1e-4)


def test_rotation_augmentation_is_not_inert(
    small_cfg: Config, field: TriplaneField, volume: Volume, probe_points: np.ndarray
) -> None:
    """The other half of the contract: the **triplane lookup** must move with the rotation.

    A triplane that undid the augmentation would satisfy the spec's literal wording
    ("a full forward pass is equivariant") and make the augmentation an exact no-op - the
    field would train identically with it on or off, and GATE 2 would be measuring the
    multi-orientation ensemble alone while appearing to test both of the design's two
    fixes. So the negative control is an assertion, not a comment. See SPEC_QUESTIONS B5.
    """
    centre = volume.bbox.mean(axis=0).astype(np.float64)
    baseline = field._sample_planes(torch.from_numpy(probe_points)).detach()
    with RotationContext(
        small_cfg, random_rotation(31), centre, requires=("coords",), fields=(field,)
    ) as rot:
        moved = torch.from_numpy(rot.coords(probe_points).astype(np.float32))
        rotated = field._sample_planes(moved).detach()
    relative = float((rotated - baseline).abs().mean() / baseline.abs().mean())
    assert relative > 0.1, (
        f"the triplane lookup barely moved under a random rotation (relative change "
        f"{relative:.4f}); the augmentation is inert and the field is not being trained to "
        "be orientation-agnostic"
    )


# --------------------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------------------


def test_field_shape_and_determinism(small_cfg: Config, volume: Volume) -> None:
    a = TriplaneField(small_cfg, volume.bbox)
    b = TriplaneField(small_cfg, volume.bbox)
    points = np.array([[100.0, 200.0, 150.0], [400.0, 400.0, 200.0]], dtype=np.float32)
    assert a(points).shape == (2, small_cfg.field_dim)
    assert torch.equal(a(points), b(points)), "two fields with the same cfg.seed must agree bitwise"


def test_field_output_width_is_independent_of_orientation_count(volume: Volume) -> None:
    """Summed, not concatenated: raising ``n_plane_orientations`` 4 -> 8 must not change d_f.

    GATE 2's first remedy is exactly that change, so it has to leave the meaning of every
    downstream width alone.
    """
    points = np.array([[100.0, 200.0, 150.0]], dtype=np.float32)
    widths = set()
    for n in (1, 4, 8):
        cfg = Config(**SMALL_FIELD, n_plane_orientations=n)
        widths.add(int(TriplaneField(cfg, volume.bbox)(points).shape[1]))
    assert widths == {Config().field_dim}


def test_field_continuity(field: TriplaneField, probe_points: np.ndarray) -> None:
    """``||F(p) - F(p + d)||`` scales ~linearly with small ``d``; no plane-boundary jumps.

    Linear rather than merely continuous: bilinear interpolation is piecewise linear, so
    halving the step must halve the response. A discontinuity at a texel or plane boundary
    would show up as a step ratio that does not fall with the step.
    """
    gen = np.random.default_rng(77)
    direction = gen.standard_normal(probe_points.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    base = field(probe_points).detach()
    steps = [0.4, 0.2, 0.1, 0.05]
    responses = []
    for step in steps:
        moved = (probe_points + step * direction).astype(np.float32)
        responses.append(float((field(moved).detach() - base).norm(dim=1).mean()))

    for coarse, fine, step_coarse, step_fine in zip(
        responses[:-1], responses[1:], steps[:-1], steps[1:], strict=True
    ):
        ratio = (coarse / fine) / (step_coarse / step_fine)
        assert 0.8 < ratio < 1.25, (
            f"the response to a step is not scaling linearly (ratio {ratio:.3f}); the field "
            "has a discontinuity"
        )

    # No point anywhere jumps: the largest single-point response to the smallest step is
    # not orders of magnitude above the mean.
    moved = (probe_points + steps[-1] * direction).astype(np.float32)
    jumps = (field(moved).detach() - base).norm(dim=1)
    assert float(jumps.max()) < 20.0 * float(jumps.mean())


def test_bbox_query_outside(field: TriplaneField, volume: Volume) -> None:
    """Querying outside the bbox clamps and warns; it does not crash and is not NaN."""
    hi = volume.bbox[1].astype(np.float64)
    lo = volume.bbox[0].astype(np.float64)
    span = hi - lo
    outside = np.stack([hi + 0.5 * span, lo - 0.5 * span, hi + 5.0 * span]).astype(np.float32)

    with pytest.warns(BBoxClampWarning, match="outside the bounding box"):
        out = field(outside)
    assert out.shape == (3, field.cfg.field_dim)
    assert torch.isfinite(out).all()

    # Clamped, not extrapolated: once a query is outside every orientation's box, going
    # further out changes nothing. (The features are *not* equal to those of the point
    # clamped to the data bbox: each rotated orientation normalises against the bounding
    # box of the *rotated* volume, which is larger, so the data bbox's corner is interior
    # to it. Saturation is the property that says the lookup does not extrapolate.)
    far = (hi + 20.0 * span).astype(np.float32)[None, :]
    further = (hi + 200.0 * span).astype(np.float32)[None, :]
    assert torch.allclose(
        field._sample_planes(torch.from_numpy(far)),
        field._sample_planes(torch.from_numpy(further)),
        atol=1e-5,
    )


def test_inside_queries_do_not_warn(field: TriplaneField, probe_points: np.ndarray) -> None:
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", BBoxClampWarning)
        field(probe_points)


def test_tv_z_penalty_is_positive_and_only_touches_the_unrotated_set(
    small_cfg: Config, volume: Volume
) -> None:
    """TV_z regularises orientation 0's XZ/YZ planes and nothing else.

    The rotated orientations have no meaningful z axis, so penalising variation along their
    third axis would suppress genuine in-plane structure that a rotation happened to put
    there.
    """
    model = TriplaneField(small_cfg, volume.bbox)
    with torch.no_grad():
        model.plane_xz.zero_()
        model.plane_yz.zero_()
    assert float(model.tv_z_penalty()) == 0.0

    with torch.no_grad():
        model.plane_xz[1:] = torch.arange(model.plane_xz.shape[2], dtype=torch.float32)[
            None, None, :, None
        ]
    assert float(model.tv_z_penalty()) == 0.0, "TV_z reached a rotated orientation"

    with torch.no_grad():
        model.plane_xz[0] = torch.arange(model.plane_xz.shape[2], dtype=torch.float32)[
            None, :, None
        ]
    assert float(model.tv_z_penalty()) == pytest.approx(1.0)


def test_tv_z_penalty_is_differentiable(small_cfg: Config, volume: Volume) -> None:
    model = TriplaneField(small_cfg, volume.bbox)
    penalty = model.tv_z_penalty()
    penalty.backward()
    assert model.plane_xz.grad is not None
    assert float(model.plane_xz.grad[0].abs().sum()) > 0.0
    assert float(model.plane_xz.grad[1:].abs().sum()) == 0.0


def test_gradient_flows_to_planes_and_positions(field: TriplaneField) -> None:
    points = torch.tensor([[300.0, 300.0, 200.0]], requires_grad=True)
    field(points).sum().backward()
    assert points.grad is not None
    assert bool(torch.isfinite(points.grad).all())
    assert field.plane_xy.grad is not None
    assert float(field.plane_xy.grad.abs().sum()) > 0.0


def test_field_rejects_degenerate_bbox(small_cfg: Config) -> None:
    flat = np.array([[0.0, 0.0, 100.0], [1000.0, 1000.0, 100.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="positive extent"):
        TriplaneField(small_cfg, flat)
    with pytest.raises(ValueError, match=r"\(2, 3\)"):
        TriplaneField(small_cfg, np.zeros((3, 3), dtype=np.float32))


def test_debug_shapes_assertions_run(volume: Volume) -> None:
    cfg = Config(**SMALL_FIELD, debug_shapes=True)
    model = TriplaneField(cfg, volume.bbox)
    out = model(np.array([[100.0, 100.0, 100.0]], dtype=np.float32))
    assert out.shape == (1, cfg.field_dim)


def test_config_rejects_a_one_layer_field_mlp() -> None:
    with pytest.raises(ConfigError, match="field_mlp_layers"):
        Config().replace(field_mlp_layers=1)


# --------------------------------------------------------------------------------------
# GATE 2
# --------------------------------------------------------------------------------------


def _assert_gate(section) -> None:  # type: ignore[no-untyped-def]
    """Assert every assertable criterion of a gate section, naming the measurement."""
    for criterion in section.criteria:
        if criterion.is_assertable:
            assert criterion.passed, criterion.summary()


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_1_oblique_parity(cfg: Config, gate_volume: Volume) -> None:
    """**The gate.** Depth-matched oblique parity >= 0.90, plus the edge-excluded check.

    Both arms of the ratio are depth-representative (specs/04, amended after T04): the 0 deg
    arm is the mean over coronal planes at *every* section, not the central one. G2.1b is the
    independent construction — both arms restricted to interior sections — so the verdict
    rests on two matched measurements rather than on averaging the edge contamination away.
    """
    _assert_gate(measure_g2_1(cfg, gate_volume, seed=0))


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_1h_augmentation_is_complete(cfg: Config, gate_volume: Volume) -> None:
    """Every rotation channel is wired: mutation, not assertion.

    A permanent criterion, not a one-off diagnostic. A partial rotation — one channel left in
    the wrong frame — produces exactly the signature a directional deficit does, and an
    invariance assertion cannot tell the two apart, because an *unwired* channel satisfies
    invariance trivially. This is what let T04's oblique deficit be diagnosed rather than
    guessed at, so it stays in the gate.
    """
    _assert_gate(measure_augmentation_completeness(cfg, gate_volume, seed=0))


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_1i_draw_noise_floor(cfg: Config, gate_volume: Volume) -> None:
    """The criterion's own resolution, re-measured every time the gate runs.

    A permanent criterion. Without the draw-to-draw floor, T04's 0.021 residual across the
    oblique angles was uninterpretable — it could have been a directional mechanism or it
    could have been the subsample, and no amount of argument distinguishes them. It reports
    rather than asserts: what it must never do is go unmeasured, because then a future
    shortfall is read as a deficit without anyone knowing whether the measurement can see it.
    """
    section = measure_ratio_stability(cfg, gate_volume, seed=0)
    _assert_gate(section)
    ratios = np.asarray(section.artifacts["ratios"])
    assert ratios.size >= 2
    assert np.isfinite(ratios).all()


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_2_z_interpolation(cfg: Config, gate_volume: Volume) -> None:
    _assert_gate(measure_g2_2(cfg, gate_volume, seed=0))


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_3_z_proximity_term_earns_its_place(cfg: Config, gate_volume: Volume) -> None:
    _assert_gate(measure_g2_3(cfg, gate_volume, seed=0))


@pytest.mark.slow
@pytest.mark.gate
def test_gate2_4_retrieval_does_not_collapse_to_copying(cfg: Config, gate_volume: Volume) -> None:
    _assert_gate(measure_g2_4(cfg, gate_volume, seed=0))


def test_gate2_angles_cover_the_specs_sweep() -> None:
    """The dihedral angles the gate reports are exactly the ones the spec names."""
    assert ANGLES_DEG == (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)


def test_the_bbox_spans_the_sections_slabs_not_their_centres():
    """SPEC_QUESTIONS C33, fixed 2026-08-26. The support is the tissue, not the mid-planes.

    ``CTFFlow._layout_targets`` jitters every cell's depth uniformly within its slab
    (``z += U(-thickness/2, +thickness/2)``) and draws the Poisson integral's Monte-Carlo set
    from the same interval. While ``Volume.bbox`` was the box of the section **centres**, the
    first and last sections' own slabs lay half outside it, so half of every boundary
    section's queries were clamped to the bbox face — 2067 of ``section_1``'s 4073 cells on
    tier-1 STARmap, and ~50 % whatever the point set, which is what made it read as a frame
    bug rather than a geometry one. Measured after the fix: **0.05 %**, and the survivors are
    in-plane layout proposals, which is what the warning calls expected.

    Asserted at both ends, and that the in-plane axes are untouched: ``x`` and ``y`` are real
    cell coordinates and a cell on the face is a real cell there.
    """
    vol, _ = make_synthetic_volume(seed=0)
    box = np.asarray(vol.bbox, dtype=np.float64)
    first, last = vol.sections[0], vol.sections[-1]
    half_first = 0.5 * float(first.thickness)
    half_last = 0.5 * float(last.thickness)

    assert box[0, 2] == pytest.approx(first.z - half_first, abs=1e-4)
    assert box[1, 2] == pytest.approx(last.z + half_last, abs=1e-4)

    # Every depth the jitter can reach is inside the support, at both ends.
    gen = np.random.default_rng(0)
    for section, half in ((first, half_first), (last, half_last)):
        jittered = float(section.z) + gen.uniform(-half, half, size=20_000)
        assert jittered.min() >= box[0, 2] - 1e-6
        assert jittered.max() <= box[1, 2] + 1e-6

    # In-plane is still the cells' own extent — not inflated.
    xy = np.concatenate([np.asarray(s.coords, dtype=np.float64) for s in vol.sections])
    assert box[0, 0] == pytest.approx(xy[:, 0].min(), abs=1e-4)
    assert box[1, 1] == pytest.approx(xy[:, 1].max(), abs=1e-4)
