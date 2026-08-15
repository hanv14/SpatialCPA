"""T03 — the 3D Gaussian random field prior, and GATE 1.

Two groups of tests:

* the spec's "Other acceptance tests", which are fast unit tests of the field's
  distributional and reproducibility contract, and run in ``make test``;
* GATE 1's four criteria (``G1.1`` - ``G1.4``), marked ``slow`` and ``gate``, which assert
  exactly the thresholds ``tests/gate1_criteria.py`` measures and ``reports/gate1.md``
  prints. If one of those fails, the gate has failed and the project stops there — that is
  what the marker is for, and why the assertion message carries the measured number.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config, ConfigError
from spatialcpav25_gen.data.schema import HeldOutSections, Volume
from spatialcpav25_gen.model.noise import (
    GaussianRandomField,
    LengthscaleFitWarning,
    VariogramError,
    fit_lengthscale_from_sections,
    matern_correlation,
    scaled_distance,
)

from tests.fixtures.synthetic import (
    DEFAULT_CELL_DENSITY,
    DEFAULT_EXTENT_UM,
    GATE_EXTENT_UM,
    GroundTruthField,
    make_synthetic_volume,
)
from tests.gate1_criteria import (
    measure_g1_1,
    measure_g1_2,
    measure_g1_3,
    measure_g1_4,
)

# The probe domain for the marginal-moment tests. The spec asks for |mean| < 0.02 and
# variance in [0.97, 1.03] over 1e5 points; those are statements about the field's
# *marginal* law, and a correlated field only reveals its marginal law over a domain many
# correlation lengths across. At 120 lengths per side there are ~1.7e6 correlation volumes
# in the box, so the sample mean of a unit-variance field has a standard deviation of
# ~0.003 and the thresholds test the field rather than the size of the box.
PROBE_BOX_ELLS = 120.0
PROBE_POINTS = 100_000
FIT_SEED = 17


@pytest.fixture(scope="module")
def ell(cfg: Config) -> tuple[float, float, float]:
    """The default length-scale triple."""
    return (cfg.ell_xy, cfg.ell_xy, cfg.ell_z)


@pytest.fixture(scope="module")
def field(cfg: Config, ell: tuple[float, float, float]) -> GaussianRandomField:
    """One field realisation, shared by the tests that only read from it."""
    return GaussianRandomField(cfg, ell, seed=3)


@pytest.fixture(scope="module")
def probe_values(
    cfg: Config, ell: tuple[float, float, float], field: GaussianRandomField
) -> np.ndarray:
    """``(PROBE_POINTS, d_h)`` field values over a box 120 correlation lengths wide."""
    box = PROBE_BOX_ELLS * max(ell)
    points = np.random.default_rng(11).uniform(0.0, box, size=(PROBE_POINTS, 3))
    return field.evaluate_numpy(points).astype(np.float64)


# --------------------------------------------------------------------------------------
# the spec's "Other acceptance tests"
# --------------------------------------------------------------------------------------


def test_grf_zero_mean_unit_var(probe_values: np.ndarray) -> None:
    """Per-channel mean |mu| < 0.02 and variance in [0.97, 1.03] over 1e5 points."""
    mean = probe_values.mean(axis=0)
    variance = probe_values.var(axis=0)
    assert np.abs(mean).max() < 0.02, f"max |mean| = {np.abs(mean).max():.4f}"
    assert variance.min() > 0.97, f"min variance = {variance.min():.4f}"
    assert variance.max() < 1.03, f"max variance = {variance.max():.4f}"


def test_grf_marginal_variance_is_exact_by_construction(field: GaussianRandomField) -> None:
    """The space-averaged variance is ``||A_c||^2 / M``, and the amplitudes make it 1 exactly.

    The empirical check above is a finite sample of the same statement; this one has no
    sampling error in it, which is the point of normalising the amplitude columns
    analytically instead of measuring them on a probe sample.
    """
    amplitude = field.amplitude.numpy()
    # amplitude already carries sqrt(2 / M), so ||a_c||^2 = 2 * ||A_c||^2 / M = 2.
    norms = (amplitude**2).sum(axis=0)
    assert np.allclose(norms, 2.0, atol=1e-9), f"amplitude column norms^2 = {norms[:4]}"


def test_grf_channels_independent(probe_values: np.ndarray) -> None:
    """Mean |corr| between channels < 0.02."""
    corr = np.corrcoef(probe_values.T)
    off_diagonal = np.abs(corr[~np.eye(corr.shape[0], dtype=bool)])
    assert off_diagonal.mean() < 0.02, f"mean |corr| = {off_diagonal.mean():.4f}"


def test_grf_channels_are_exactly_orthogonal(field: GaussianRandomField) -> None:
    """Channel independence is algebraic, not statistical: the amplitude columns are orthogonal."""
    amplitude = field.amplitude.numpy()
    gram = amplitude.T @ amplitude
    off_diagonal = np.abs(gram[~np.eye(gram.shape[0], dtype=bool)])
    assert off_diagonal.max() < 1e-9, f"max |off-diagonal| = {off_diagonal.max():.2e}"


def test_with_lengthscale_preserves_seed(
    cfg: Config, ell: tuple[float, float, float], field: GaussianRandomField
) -> None:
    """Halving ``ell`` twice equals quartering it once — same draws, no redraw."""
    halved_twice = field.with_lengthscale(tuple(v / 2 for v in ell)).with_lengthscale(
        tuple(v / 4 for v in ell)
    )
    quartered = field.with_lengthscale(tuple(v / 4 for v in ell))
    assert torch.equal(halved_twice.omega, quartered.omega)
    assert torch.equal(halved_twice.phase, quartered.phase)
    assert torch.equal(halved_twice.amplitude, quartered.amplitude)
    assert torch.equal(halved_twice.directions, field.directions)

    points = np.random.default_rng(4).uniform(0.0, 500.0, size=(256, 3))
    assert torch.equal(halved_twice(points), quartered(points))
    # ... and it is genuinely a different field from the one it was derived from.
    assert not torch.equal(quartered(points), field(points))


def test_torch_numpy_agree(field: GaussianRandomField) -> None:
    """The two evaluation paths agree to < 1e-5 — in fact exactly, on every machine.

    ``evaluate_numpy`` delegates to the torch path rather than re-implementing the
    arithmetic, so the tolerance the spec allows is not needed: a second float32 matmul
    would reassociate differently from torch's, by a few 1e-6 that vary with the BLAS
    vendor, and the difference would have been a portability trap rather than information.
    """
    points = np.random.default_rng(5).uniform(-500.0, 500.0, size=(2048, 3))
    torch_values = field(points).numpy()
    numpy_values = field.evaluate_numpy(points)
    gap = float(np.abs(torch_values - numpy_values).max())
    assert gap < 1e-5, f"max |torch - numpy| = {gap:.3e}"
    assert np.array_equal(torch_values, numpy_values), "the numpy path delegates; expect equality"


def test_fit_lengthscale_recovers_truth(
    volume: Volume, gt_field: GroundTruthField, cfg: Config
) -> None:
    """The fitted ``ell_xy`` is within 30% of the length the fixture was built with."""
    with warnings.catch_warnings():
        # The along-z variogram does not saturate on a 400 um stack; that is asserted
        # separately in test_fit_warns_when_variogram_does_not_saturate.
        warnings.simplefilter("ignore", LengthscaleFitWarning)
        fitted = fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED)
    truth = gt_field.autocorr_length_xy
    error = abs(fitted[0] - truth) / truth
    assert fitted[0] == fitted[1], "the two in-plane axes share one length-scale"
    assert error < 0.30, f"fitted ell_xy = {fitted[0]:.1f} um vs truth {truth:.1f} um ({error:.1%})"


# --------------------------------------------------------------------------------------
# the kernel, and the parametrisation the spec says to verify rather than trust
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("nu", [0.5, 1.5, 2.5, 4.0])
def test_matern_correlation_matches_sklearn(nu: float) -> None:
    """Our analytic Matern is the standard parametrisation, not a rescaled cousin."""
    from sklearn.gaussian_process.kernels import Matern

    radius = np.linspace(0.0, 4.0, 41)
    reference = Matern(length_scale=1.0, nu=nu)(radius[:, None], np.zeros((1, 1)))[:, 0]
    assert np.abs(matern_correlation(radius, nu) - reference).max() < 1e-10


def test_matern_correlation_closed_forms() -> None:
    """nu = 1/2 is the exponential kernel and nu = 3/2 its closed form; check both."""
    radius = np.linspace(0.0, 3.0, 31)
    assert np.abs(matern_correlation(radius, 0.5) - np.exp(-radius)).max() < 1e-12
    root3 = math.sqrt(3.0)
    closed = (1.0 + root3 * radius) * np.exp(-root3 * radius)
    assert np.abs(matern_correlation(radius, 1.5) - closed).max() < 1e-12


def test_realised_covariance_tracks_the_anisotropic_matern(cfg: Config) -> None:
    """The field's own covariance follows ``matern(||p - q||_ell)`` with anisotropic ``ell``.

    The fast, small version of G1.1: it fixes the parametrisation (a length-scale
    convention off by ``sqrt(2 nu)`` would show up as a large, systematic gap) without the
    gate's sweep over M.
    """
    ell = (60.0, 60.0, 300.0)
    gen = np.random.default_rng(6)
    p = gen.uniform(0.0, 1000.0, size=(2000, 3))
    direction = gen.standard_normal((2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = gen.uniform(0.0, 3.0, size=2000)
    q = p + radius[:, None] * direction * np.asarray(ell)[None, :]

    assert np.abs(scaled_distance(p, q, ell) - radius).max() < 1e-9
    grf = GaussianRandomField(cfg, ell, seed=7)
    gap = float(np.abs(grf.covariance(p, q) - matern_correlation(radius, cfg.matern_nu)).mean())
    assert gap < 0.03, f"mean |realised - analytic| = {gap:.4f}"


# --------------------------------------------------------------------------------------
# determinism, purity, and the contract
# --------------------------------------------------------------------------------------


def test_same_seed_is_bitwise_identical(cfg: Config, ell: tuple[float, float, float]) -> None:
    """Two constructions with the same seed produce the same field, bit for bit."""
    points = np.random.default_rng(8).uniform(0.0, 800.0, size=(512, 3))
    first = GaussianRandomField(cfg, ell, seed=42)(points)
    second = GaussianRandomField(cfg, ell, seed=42)(points)
    third = GaussianRandomField(cfg, ell, seed=43)(points)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_field_is_pure_and_depends_only_on_position(field: GaussianRandomField) -> None:
    """Reordering the points reorders the values and changes nothing else.

    The property the "Do NOT" section is about: the noise at a point may not depend on
    which other points were generated with it. A graph-smoothing prior would fail this.
    The query keeps its shape here, so the assertion is bitwise — that is the guarantee
    G1.2 and T07's ``L_cross`` are built on.
    """
    points = np.random.default_rng(9).uniform(0.0, 900.0, size=(400, 3))
    full = field(points)
    shuffled = np.random.default_rng(10).permutation(points.shape[0])
    assert torch.equal(field(points[shuffled]), full[shuffled])


def test_batch_shape_does_not_change_a_single_value(field: GaussianRandomField) -> None:
    """A point's value does not depend on how many other points were asked for. Bitwise.

    ``forward`` evaluates fixed-size chunks and zero-pads the last one, so every matmul in
    every query is ``(grf_chunk_points, M) @ (M, d_h)`` — the same shape, therefore the
    same kernel, blocking and reduction order over the M features. Without the padding a
    one-row query is dispatched to a matrix-*vector* kernel and lands ~1e-5 away, which
    would make "exact by construction" (G1.2, T07's ``L_cross``) depend on the caller
    happening to use matching batch sizes.

    The sizes below straddle the default 1024-point chunk deliberately: 1 (the GEMV case),
    2, 137, 400, and 4000 (multi-chunk, with a padded tail).
    """
    points = np.random.default_rng(9).uniform(0.0, 900.0, size=(4096, 3))
    full = field(points)
    for n in (1, 2, 137, 400, 4000):
        assert torch.equal(field(points[:n]), full[:n]), (
            f"a {n}-point query differs from the same points inside a 4096-point batch by "
            f"{float((field(points[:n]) - full[:n]).abs().max()):.3e}"
        )


def test_chunk_boundaries_do_not_change_a_single_value(
    cfg: Config, ell: tuple[float, float, float]
) -> None:
    """Values are bitwise stable across chunk *boundaries*, and near-stable across chunk sizes.

    Two fields differing only in ``grf_chunk_points`` are two different arithmetic
    programs: the chunk size *is* the matmul's row count, so a different value picks a
    different kernel. That is a property of BLAS, not something padding can remove, and it
    does not touch anything load-bearing — ``grf_chunk_points`` is a frozen ``Config``
    field, so within one run every query goes through the same shape (asserted above).
    Measured gap on the reference box: 0.0 between 97, 512 and 1024, and 2.0e-6 against a
    single 100000-row matmul.
    """
    points = np.random.default_rng(12).uniform(0.0, 900.0, size=(3000, 3))
    reference = GaussianRandomField(cfg, ell, seed=13)(points)
    # Same field, queried in pieces that fall either side of the chunk boundary.
    field = GaussianRandomField(cfg, ell, seed=13)
    for cut in (1, cfg.grf_chunk_points - 1, cfg.grf_chunk_points, cfg.grf_chunk_points + 1):
        assert torch.equal(field(points[:cut]), reference[:cut])
    for chunk in (97, 512, 100_000):
        other = GaussianRandomField(cfg.replace(grf_chunk_points=chunk), ell, seed=13)(points)
        assert float((other - reference).abs().max()) < 1e-5


def test_gradient_flows_to_positions(field: GaussianRandomField) -> None:
    """Differentiable with respect to ``xyz``, and only with respect to ``xyz``."""
    points = torch.tensor(
        np.random.default_rng(14).uniform(0.0, 500.0, size=(64, 3)),
        dtype=torch.float32,
        requires_grad=True,
    )
    field(points).square().sum().backward()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()
    assert float(points.grad.abs().max()) > 0.0
    assert not any(p.requires_grad for p in field.buffers())


def test_shape_and_argument_errors(cfg: Config, ell: tuple[float, float, float]) -> None:
    """Bad inputs raise with a message that names what was wrong (Convention 6)."""
    grf = GaussianRandomField(cfg, ell, seed=15)
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        grf(np.zeros((10, 2)))
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        grf.evaluate_numpy(np.zeros((10,)))
    with pytest.raises(ValueError, match="matching point sets"):
        grf.covariance(np.zeros((10, 3)), np.zeros((11, 3)))
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianRandomField(cfg, (100.0, 0.0, 100.0), seed=1)
    with pytest.raises(ValueError, match="strictly positive"):
        GaussianRandomField(cfg, (100.0, 100.0, float("inf")), seed=1)
    with pytest.raises(ConfigError, match="latent_dim"):
        cfg.replace(latent_dim=64, n_rff=32)


def test_empty_query_returns_empty(field: GaussianRandomField, cfg: Config) -> None:
    """A zero-row query is legal and returns ``(0, d_h)`` rather than raising."""
    assert field(np.zeros((0, 3))).shape == (0, cfg.latent_dim)


def test_debug_shapes_assertions_run(cfg: Config, ell: tuple[float, float, float]) -> None:
    """With ``debug_shapes`` on, the runtime shape assertions execute and pass."""
    debug_cfg = cfg.replace(debug_shapes=True)
    out = GaussianRandomField(debug_cfg, ell, seed=16)(np.zeros((17, 3)))
    assert out.shape == (17, cfg.latent_dim)


# --------------------------------------------------------------------------------------
# fitting ell
# --------------------------------------------------------------------------------------


def test_fit_lengthscale_is_deterministic(volume: Volume, cfg: Config) -> None:
    """Same seed, same triple; a different seed moves it by less than the tolerance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LengthscaleFitWarning)
        first = fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED)
        again = fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED)
        other = fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED + 1)
    assert first == again
    assert abs(other[0] - first[0]) / first[0] < 0.10


def test_fit_lengthscale_rejects_heldout(volume: Volume, cfg: Config) -> None:
    """Fitting on held-out sections is leakage, and the type system says so."""
    heldout = HeldOutSections(
        sections=list(volume.sections[:3]),
        gene_names=list(volume.gene_names),
        celltype_names=list(volume.celltype_names),
        region_names=None if volume.region_names is None else list(volume.region_names),
        specimen_id=volume.specimen_id,
    )
    with pytest.raises(TypeError, match="leakage"):
        fit_lengthscale_from_sections(heldout, cfg, seed=FIT_SEED)  # type: ignore[arg-type]


def test_fit_warns_when_variogram_does_not_saturate(volume: Volume, cfg: Config) -> None:
    """A 400 um stack cannot watch a 200 um correlation decay away, and the fit says so."""
    with pytest.warns(LengthscaleFitWarning, match="along-z"):
        fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED)


def test_fit_refuses_structureless_expression(volume: Volume, cfg: Config) -> None:
    """With the cells shuffled inside each section there is no length-scale to fit."""
    from tests.conftest import copy_section, rebuild_volume

    gen = np.random.default_rng(18)
    shuffled = [
        copy_section(section, coords=section.coords[gen.permutation(section.n_cells)])
        for section in volume.sections
    ]
    with pytest.raises(VariogramError):
        fit_lengthscale_from_sections(rebuild_volume(volume, shuffled), cfg, seed=FIT_SEED)


def test_fit_is_anisotropic(volume: Volume, cfg: Config) -> None:
    """``ell_z`` is fitted separately from ``ell_xy`` — the anisotropy is the point."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LengthscaleFitWarning)
        fitted = fit_lengthscale_from_sections(volume, cfg, seed=FIT_SEED)
    assert fitted[2] > fitted[0], (
        "the fixture's field is built with a longer correlation length along z than "
        f"in-plane; the fit returned {fitted}"
    )


# --------------------------------------------------------------------------------------
# GATE 1
# --------------------------------------------------------------------------------------

GATE_SEED = 0


def _assert_gate(section) -> None:  # type: ignore[no-untyped-def]
    """Assert every assertable criterion, reporting all of them in the message."""
    failures = [c.summary() for c in section.criteria if not c.passed]
    assert not failures, (
        f"{section.key} {section.title} FAILED\n"
        + "\n".join("  " + line for line in failures)
        + "\n  full section:\n"
        + "\n".join("  " + c.summary() for c in section.criteria)
    )


@pytest.mark.slow
@pytest.mark.gate
def test_gate1_1_covariance_correctness(cfg: Config) -> None:
    """G1.1 — realised covariance matches the analytic anisotropic Matern, and improves with M."""
    _assert_gate(measure_g1_1(cfg, seed=GATE_SEED))


@pytest.mark.slow
@pytest.mark.gate
def test_gate1_2_intersection_consistency(cfg: Config, gate_volume: Volume) -> None:
    """G1.2 — two planes crossing inside the bbox get identical noise along the line."""
    _assert_gate(measure_g1_2(cfg, gate_volume, seed=GATE_SEED))


@pytest.mark.slow
@pytest.mark.gate
def test_gate1_3_autocorrelation_transfer(
    cfg: Config, gate_volume: Volume, gate_gt_field: GroundTruthField
) -> None:
    """G1.3 — the decisive test: a correlated prior preserves spatial autocorrelation.

    Measured on the gate fixture (3000 um field of view), not on the fast suite's 1000 um
    one: see ``tests.fixtures.synthetic.GATE_EXTENT_UM``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", LengthscaleFitWarning)
        section = measure_g1_3(cfg, gate_volume, gate_gt_field, seed=GATE_SEED)
    _assert_gate(section)


@pytest.mark.slow
@pytest.mark.gate
def test_gate1_4_determinism_and_scaling(cfg: Config) -> None:
    """G1.4 — same seed in another process, and a million queries inside the budget."""
    _assert_gate(measure_g1_4(cfg, seed=GATE_SEED))


@pytest.mark.slow
def test_gate1_measurements_are_reproducible(cfg: Config) -> None:
    """The gate's own numbers do not move between runs; a gate that jittered would be noise."""
    first = measure_g1_1(cfg, seed=GATE_SEED)
    second = measure_g1_1(cfg, seed=GATE_SEED)
    assert [c.measured for c in first.criteria] == [c.measured for c in second.criteria]


def test_fixture_volume_is_the_one_the_gate_uses(volume: Volume) -> None:
    """The gate report and the tests must be looking at the same synthetic volume."""
    other, _ = make_synthetic_volume(seed=0)
    assert volume.section_ids == other.section_ids
    assert np.array_equal(volume.sections[0].coords, other.sections[0].coords)


def test_default_fixture_is_unchanged_by_the_extent_parameter(volume: Volume) -> None:
    """Widening the field of view is opt-in: the default fixture is what T01/T02 measured.

    ``n_cells_per_section=None`` derives the count from ``cell_density * extent_xy ** 2``,
    so the gate fixture is a wider field of view at the same tissue density rather than a
    sparser tissue — the kNN-graph statistics every gate criterion is defined on would
    otherwise move for a reason that has nothing to do with the field.
    """
    assert volume.sections[0].n_cells == 1500
    assert float(volume.bbox[1, 0] - volume.bbox[0, 0]) < DEFAULT_EXTENT_UM
    density = volume.sections[0].n_cells / DEFAULT_EXTENT_UM**2
    assert abs(density - DEFAULT_CELL_DENSITY) < 1e-9


@pytest.mark.slow
def test_gate_fixture_holds_density_and_widens_the_window(gate_volume: Volume) -> None:
    """The gate fixture is 3x wider with 9x the cells, i.e. the same tissue seen wider."""
    extent = float(gate_volume.bbox[1, 0] - gate_volume.bbox[0, 0])
    assert extent > 0.99 * GATE_EXTENT_UM
    assert gate_volume.sections[0].n_cells == int(DEFAULT_CELL_DENSITY * GATE_EXTENT_UM**2)
    # Nearest-neighbour distance is what every kNN-graph statistic is scaled by; it must
    # not move when the window does.
    assert abs(gate_volume.median_nn_dist - 13.9) < 0.5
