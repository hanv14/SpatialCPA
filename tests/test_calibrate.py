"""T09 acceptance tests for the leakage-free calibrators (``specs/09`` §2).

The three branches of the length-scale search are tested against an **injected** objective —
a planted unimodal curve — rather than against twenty generated sections. That is not a
short-cut: the branches are properties of the *search*, the spec's own criterion for the
unreachable one is stated in terms of the grid maximiser, and GATE 1 already measured that
the real objective has the assumed shape (unimodality violation 0.000, maximiser 2.52x the
fitted ``ell``). ``test_calibration_converges`` runs the real objective end to end and is
``slow``.

Every numeric threshold is the spec's: ``|I_gen - I_flank| < 0.02`` within
``Config.bisection_max_iter`` iterations; the unreachable branch returning the maximiser and
**not** a bracket endpoint; the bracket's upper end being the extent cap on a narrow field of
view; and the fitted ``w(v)`` being non-increasing.

Open risk **R1** is the other thing pinned here: ``ell_z`` is calibrated against the
*observed* between-section correlation (remedy 2) and the variogram fit enters as the
bracket's upper **endpoint** (remedy 3), so a stack that cannot constrain it reports
``target_unreachable`` and the number is an upper bound.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import HeldOutSections
from spatialcpav25_gen.infer.calibrate import (
    CALIBRATION_STATUSES,
    CalibrationError,
    DetectionCalibration,
    IsotonicRegressor,
    UnreachableTargetWarning,
    _require_ell_reaches_output,
    calibrate_anchor_weight,
    calibrate_detection,
    calibrate_ell_z,
    calibrate_lengthscale,
    calibrate_retrieval_window,
    mean_morans_i,
    observed_z_decay,
)
from spatialcpav25_gen.model.field import BBoxClampWarning

from tests.fixtures.synthetic import make_synthetic_volume
from tests.test_generate import E5_STEPS, SEED, build_model, t09_cfg

MORANS_TOL = 0.02
"""``specs/09``: ``test_calibration_converges`` requires ``|I_gen - I_flank| < 0.02``."""


def unimodal(peak: float, height: float, floor: float = 0.0):
    """A planted objective that rises, turns over and falls — GATE 1's measured shape.

    ``height`` is the value at ``peak``; the curve is a Gaussian in ``log ell``, so it is
    strictly increasing below the maximiser and strictly decreasing above it, which is
    exactly the structure the calibrator's "locate the maximum, then bisect below it" relies
    on and the structure a naive bisection breaks on.
    """

    def objective(ell: float) -> float:
        return floor + (height - floor) * math.exp(-0.5 * (math.log(ell / peak) / 0.5) ** 2)

    return objective


class _StubModel:
    """Stands in for a ``CTFFlow`` where the calibrator only needs the search, not a model."""

    def __init__(self, grf_seed: int = 0):
        self.grf = type("_G", (), {"seed": grf_seed})()


@pytest.fixture(scope="module")
def training_volume():
    """The synthetic fixture's training volume — no held-out section can reach a calibrator."""
    vol, _ = make_synthetic_volume(seed=0)
    training, held = split_holdout(vol, "alternating", 0, Config())
    return vol, training, held


# --------------------------------------------------------------------------------------
# the three branches of the length-scale search
# --------------------------------------------------------------------------------------


def test_calibrator_recovers_a_reachable_target(training_volume):
    """A target inside the achievable range converges, and the achieved I_gen matches it."""
    _, training, _ = training_volume
    cfg = t09_cfg()
    fitted_peak = 0.6 * float(cfg.ell_xy)
    objective = unimodal(peak=fitted_peak, height=0.6)
    target = 0.35  # on the rising branch, well inside the range
    result = calibrate_lengthscale(
        _StubModel(),  # type: ignore[arg-type]
        training,
        cfg,
        seed=SEED,
        measure=objective,
        measure_z=unimodal(peak=100.0, height=0.9),
    )
    assert result.status == "converged"
    assert abs(result.i_gen - result.i_target) < float(cfg.calibration_morans_tol) or abs(
        result.i_gen - target
    ) < float(cfg.calibration_morans_tol)
    assert 0 < result.iterations <= int(cfg.bisection_max_iter)
    assert result.ell[0] == result.ell[1]


def test_calibrator_reports_unreachable_target(training_volume):
    """A target above the objective's maximum returns the **maximiser**, not an endpoint.

    The branch T03 measured and the one a naive bisection gets wrong: with no root on the
    bracket, bisection walks into whichever endpoint its comparison sends it to and reports
    it as an answer.
    """
    _, training, _ = training_volume
    cfg = t09_cfg()
    peak = 0.9 * float(cfg.ell_xy)
    objective = unimodal(peak=peak, height=0.2)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = calibrate_lengthscale(
            _StubModel(),  # type: ignore[arg-type]
            training,
            cfg.replace(ell_xy=float(cfg.ell_xy)),
            seed=SEED,
            measure=objective,
            measure_z=unimodal(peak=100.0, height=0.9),
        )
    assert result.status == "target_unreachable"
    assert result.i_target > result.i_gen
    assert any(w.category is UnreachableTargetWarning for w in caught)

    grid_best = float(result.grid_ell[int(np.argmax(result.grid_i))])
    assert result.ell[0] == pytest.approx(grid_best)
    assert result.ell[0] > float(result.grid_ell.min())
    assert result.ell[0] < float(result.grid_ell.max())


def test_calibrator_bracket_respects_the_caps():
    """On a narrow field of view the bracket's upper end is the extent cap.

    ``ell_max = min(calibration_ell_max_extent_frac x extent, calibration_ell_max_fitted_
    multiple x fitted)``. A 300 um-wide volume makes the first term 60 um, well below twice
    any plausible fitted length-scale, so the extent cap must bind — and the grid must stop
    there.
    """
    # Denser than the default fixture so that a 300 um window still fills the in-plane grid
    # the along-z variogram averages on; the cap being tested is about the *window*, not the
    # cell count.
    narrow, _ = make_synthetic_volume(seed=0, extent_xy=300.0, cell_density=0.01)
    cfg = t09_cfg()
    training, _ = split_holdout(narrow, "alternating", 0, cfg)
    # The *observed* in-plane extent: the bounding box is the cells' own, a nearest-neighbour
    # distance shy of the 300 um the fixture was drawn in (``section_plane``'s caveat).
    box = np.asarray(training.bbox, dtype=np.float64)
    extent = float(np.min(box[1, :2] - box[0, :2]))
    extent_cap = float(cfg.calibration_ell_max_extent_frac) * extent
    result = calibrate_lengthscale(
        _StubModel(),  # type: ignore[arg-type]
        training,
        cfg,
        seed=SEED,
        measure=unimodal(peak=0.5 * extent_cap, height=0.5),
        measure_z=unimodal(peak=100.0, height=0.9),
    )
    fitted_cap = float(cfg.calibration_ell_max_fitted_multiple) * float(result.ell_fitted[0])
    assert float(result.grid_ell.max()) == pytest.approx(min(extent_cap, fitted_cap))
    assert float(result.grid_ell.max()) == pytest.approx(extent_cap), (
        f"the extent cap {extent_cap:g} um should bind on a {extent:g} um volume, but the "
        f"fitted cap is {fitted_cap:g} um"
    )


def test_ell_z_enters_as_a_bracket_endpoint_and_can_be_unreachable(training_volume):
    """Open risk R1: the fitted ``ell_z`` bounds the search, and an unreachable target says so.

    Remedy 3's criterion, as inherited by T09: a run whose ``ell_z`` bracket cannot be closed
    against the observed between-section correlation reports ``target_unreachable`` and the
    selected ``ell_z`` is recorded as an upper bound, not a fitted value.
    """
    _, training, _ = training_volume
    cfg = t09_cfg()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnreachableTargetWarning)
        result = calibrate_lengthscale(
            _StubModel(),  # type: ignore[arg-type]
            training,
            cfg,
            seed=SEED,
            measure=unimodal(peak=60.0, height=0.6),
            # An objective that never reaches the real stack's adjacent-section correlation.
            measure_z=unimodal(peak=120.0, height=0.05),
        )
    assert result.ell_z_status == "target_unreachable"
    assert result.ell_z_is_upper_bound
    assert result.z_target > result.z_achieved
    fitted_z = float(result.ell_fitted[2])
    assert result.ell[2] <= float(cfg.calibration_ell_z_max_fitted_multiple) * fitted_z + 1e-6


def test_calibrator_refuses_a_prior_that_ignores_ell(training_volume):
    """Calibrating ``ell`` under ``prior_mode="iid"`` is refused, not silently flat.

    The measurement that put this guard here: on the fixture the selector chose
    ``prior_mode="iid"`` and the calibration that followed reported ``target_unreachable``
    with the maximiser pinned at the bracket's low end — a correct answer to a question that
    could not be asked, because under ``iid`` the prior is white noise and never queries the
    field. An injected objective is still allowed: the search is then being tested against a
    planted curve and no prior is consulted.
    """
    _, training, _ = training_volume
    cfg = t09_cfg(prior_mode="iid")
    with pytest.raises(CalibrationError, match="does not use the GRF"):
        calibrate_lengthscale(_StubModel(), training, cfg, seed=SEED)  # type: ignore[arg-type]
    # ...and the planted-objective path still runs, because it never generates.
    result = calibrate_lengthscale(
        _StubModel(),  # type: ignore[arg-type]
        training,
        cfg,
        seed=SEED,
        measure=unimodal(peak=60.0, height=0.6),
        measure_z=unimodal(peak=120.0, height=0.9),
    )
    assert result.status in CALIBRATION_STATUSES


def test_calibrator_refuses_an_expression_path_that_ignores_ell(training_volume):
    """Calibrating ``ell`` under ``expr_mode="cross-mix"`` is refused for the same reason.

    ``prior_mode="iid"`` is not the only way to sever ``ell`` from the output. ``cross-mix``
    copies every emitted count verbatim from a donor cell and never evaluates the flow, so the
    latent the GRF parameterises never reaches the counts. Measured on the fixture: the
    between-section objective is identical to ten decimal places (0.6727617248) at ``ell_z``
    25, 60, 137, 200, 275 and 364.6 um, while the same sweep under ``zinb-flow`` rises
    monotonically 0.8867 -> 0.9316. A constant objective has no maximiser, so the search
    returns whichever grid point ties first — the bracket endpoint ``specs/09`` §2 says an
    unreachable target must never return.
    """
    _, training, _ = training_volume
    cfg = t09_cfg(prior_mode="correlated", expr_mode="cross-mix")
    for calibrate in (calibrate_lengthscale, calibrate_ell_z):
        with pytest.raises(CalibrationError, match="emits donor counts verbatim"):
            calibrate(_StubModel(), training, cfg, seed=SEED)  # type: ignore[arg-type]
    # The two paths ell *can* act through are not refused.
    for expr_mode in ("zinb-flow", "auto-blend"):
        _require_ell_reaches_output(
            t09_cfg(prior_mode="correlated", expr_mode=expr_mode), "probe", True
        )


def test_observed_z_decay_is_measured_not_fitted(training_volume):
    """The R1 target is the *observed* decay: correlation falls as sections get further apart."""
    _, training, _ = training_volume
    cfg = t09_cfg()
    lags, correlations = observed_z_decay(training, cfg)
    assert lags.size >= 3
    order = np.argsort(lags)
    assert correlations[order][0] > correlations[order][-1]


# --------------------------------------------------------------------------------------
# leakage
# --------------------------------------------------------------------------------------


def test_calibration_no_leakage(training_volume):
    """Held-out sections cannot be passed, and the result does not depend on their existence.

    Both halves of ``specs/09``'s criterion: the signature refuses the type, and a
    calibration run on a training volume split from a *larger* parent object returns exactly
    what it returns when the parent has no held-out section at all.
    """
    vol, training, held = training_volume
    cfg = t09_cfg()
    assert isinstance(held, HeldOutSections)
    for bad in (held, vol):
        with pytest.raises(TypeError, match="TrainingVolume"):
            calibrate_lengthscale(_StubModel(), bad, cfg, seed=SEED)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            calibrate_retrieval_window(bad, cfg)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            observed_z_decay(bad, cfg)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            calibrate_detection(_StubModel(), bad, cfg, seed=SEED)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            calibrate_anchor_weight(_StubModel(), bad, cfg, seed=SEED)  # type: ignore[arg-type]

    # The same training sections, reached through a parent that has no holdout at all.
    from spatialcpav25_gen.data.schema import TrainingVolume

    standalone = TrainingVolume(
        sections=list(training.sections),
        gene_names=list(training.gene_names),
        celltype_names=list(training.celltype_names),
        region_names=None if training.region_names is None else list(training.region_names),
        specimen_id=training.specimen_id,
    )
    objective = unimodal(peak=60.0, height=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnreachableTargetWarning)
        a = calibrate_lengthscale(
            _StubModel(),
            training,
            cfg,
            seed=SEED,
            measure=objective,  # type: ignore[arg-type]
            measure_z=unimodal(peak=120.0, height=0.9),
        )
        b = calibrate_lengthscale(
            _StubModel(),
            standalone,
            cfg,
            seed=SEED,
            measure=objective,  # type: ignore[arg-type]
            measure_z=unimodal(peak=120.0, height=0.9),
        )
    assert a.ell == b.ell
    assert a.status == b.status
    assert a.i_target == b.i_target
    assert calibrate_retrieval_window(training, cfg).window == pytest.approx(
        calibrate_retrieval_window(standalone, cfg).window
    )


# --------------------------------------------------------------------------------------
# the isotonic anchor map
# --------------------------------------------------------------------------------------


def test_isotonic_regressor_is_non_increasing_and_interpolates():
    """PAVA returns a non-increasing fit, and prediction is clamped outside the fitted range."""
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    y = np.array([1.0, 0.2, 0.8, 0.0, 0.1])  # deliberately non-monotone
    fitted = IsotonicRegressor.fit(x, y)
    assert fitted.is_monotone
    assert np.all(np.diff(fitted.y) <= 1e-9)
    assert fitted.predict(np.array([0.0]))[0] == pytest.approx(float(fitted.y[0]))
    assert fitted.predict(np.array([9.9]))[0] == pytest.approx(float(fitted.y[-1]))
    middle = fitted.predict(np.array([0.25]))[0]
    assert min(fitted.y) - 1e-9 <= middle <= max(fitted.y) + 1e-9


def test_isotonic_fit_is_monotone_on_random_input():
    """PAVA is monotone on every input, not only on the tidy ones.

    A randomised check, because the merge step is the kind of index arithmetic that is right
    on short runs and wrong on long ones: the first implementation merged into the wrong
    block (``block[-2] += block.pop()`` resolves ``-2`` *after* the pop) and still passed a
    five-point example.
    """
    generator = np.random.default_rng(0)
    for _ in range(200):
        n = int(generator.integers(2, 12))
        x = np.sort(generator.random(n))
        y = generator.random(n)
        w = generator.random(n) + 0.1
        for increasing in (False, True):
            fitted = IsotonicRegressor.fit(x, y, weights=w, increasing=increasing)
            assert fitted.is_monotone, (x, y, w, increasing, fitted.y)


def test_isotonic_fit_refuses_non_finite_knots():
    """A NaN knot is an error: every comparison against NaN is False, so it would fit silently."""
    with pytest.raises(CalibrationError, match="finite"):
        IsotonicRegressor.fit(np.array([0.0, 1.0]), np.array([np.nan, 0.5]))


def test_anchor_weight_monotone_on_a_planted_map():
    """The map a perfect measurement would produce is recovered as non-increasing."""
    variance = np.linspace(0.0, 1.0, 40)
    weight = np.clip(1.0 - variance + 0.05 * np.sin(20 * variance), 0.0, 1.0)
    fitted = IsotonicRegressor.fit(variance, weight)
    assert fitted.is_monotone
    assert fitted.predict(np.array([0.05]))[0] > fitted.predict(np.array([0.95]))[0]


# --------------------------------------------------------------------------------------
# detection and dispersion
# --------------------------------------------------------------------------------------


def test_detection_calibration_carries_both_corrections():
    """``DetectionCalibration`` moves ``pi`` **and** ``theta``, leaves ``mu``, records sections."""
    cfg = t09_cfg()
    calibration = DetectionCalibration(
        pi_scale=1.0,
        pi_shift=np.array([-1.0, 1.0]),
        theta_scale=1.0,
        theta_shift=np.array([0.5, -0.5]),
        section_ids=("s0", "s1"),
        detection_gen=np.array([0.2, 0.4]),
        detection_real=np.array([0.3, 0.3]),
    )
    mu = torch.full((5, 2), 2.0)
    theta = torch.full((5, 2), 1.0)
    pi = torch.full((5, 2), 0.5)
    mu_out, theta_out, pi_out = calibration.apply(mu, theta, pi, cfg)
    assert torch.equal(mu_out, mu)
    assert not torch.equal(theta_out, theta)
    assert float(pi_out[0, 0]) < 0.5 < float(pi_out[0, 1])
    assert float(theta_out[0, 0]) > 1.0 > float(theta_out[0, 1])
    assert calibration.section_ids == ("s0", "s1")
    assert calibration.n_genes == 2

    with pytest.raises(CalibrationError, match="per gene"):
        calibration.apply(torch.zeros((5, 3)), torch.ones((5, 3)), torch.full((5, 3), 0.5), cfg)


def test_mean_morans_i_needs_matching_shapes():
    """The observable refuses mismatched counts and coordinates rather than broadcasting."""
    cfg = t09_cfg()
    with pytest.raises(CalibrationError, match="rows of counts"):
        mean_morans_i(np.zeros((10, 3)), np.zeros((9, 2)), cfg, seed=0)


# --------------------------------------------------------------------------------------
# the real objective, end to end
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_for_calibration():
    """A trained model and its training volume, for the end-to-end calibrations.

    Fitted at ``E5_STEPS`` rather than at ``tests.test_generate``'s cheaper ``GEN_STEPS``,
    because two of these tests are statements about a *fitted* model rather than about wiring:
    the detection correction's headroom is a comparison between the model's own error and the
    tissue's section-to-section variation, and at 60 steps the first is still an artefact of
    the budget (0.0545 against 0.0217 at 600)."""
    cfg = t09_cfg()
    model, training = build_model(cfg, steps=E5_STEPS)
    return model, training, cfg


@pytest.mark.slow
def test_calibration_converges(trained_for_calibration):
    """The real bisection reaches ``|I_gen - I_flank| < 0.02`` within 8 iterations, or says why.

    ``specs/09``'s criterion, with the spec's own caveat honoured: when the flanking
    sections' Moran's I lies above everything the generated section can reach on the capped
    bracket there **is** no root, and the calibrator must report ``target_unreachable`` with
    the maximiser rather than pretend to converge. Both outcomes are a pass here; silently
    returning a bracket endpoint is not, and that is what the status assertions pin.
    """
    model, training, cfg = trained_for_calibration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        warnings.simplefilter("ignore", UnreachableTargetWarning)
        result = calibrate_lengthscale(model, training, cfg, seed=SEED)
    assert result.status in {"converged", "target_unreachable", "boundary"}
    assert result.iterations <= int(cfg.bisection_max_iter)
    if result.status == "converged":
        assert abs(result.i_gen - result.i_target) < MORANS_TOL
    else:
        assert result.i_target > float(np.max(result.grid_i)) - MORANS_TOL
        assert result.ell[0] == pytest.approx(float(result.grid_ell[int(np.argmax(result.grid_i))]))
    assert result.section_ids  # the flanking training sections it measured the target on


@pytest.mark.slow
def test_anchor_weight_is_fitted_and_monotone(trained_for_calibration):
    """``calibrate_anchor_weight`` returns a non-increasing map fitted on LOSO folds."""
    model, training, cfg = trained_for_calibration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        fitted = calibrate_anchor_weight(model, training, cfg, seed=SEED, n_folds=2)
    assert isinstance(fitted, IsotonicRegressor)
    assert fitted.is_monotone
    predictions = fitted.predict(np.linspace(0.0, float(fitted.x.max()) * 1.5, 16))
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))
    assert np.all(np.diff(predictions) <= 1e-9)


def _detection_rate(mu, theta, pi) -> np.ndarray:
    """Per-gene detection rate implied by a decoder output. ``(G,)`` float64."""
    from spatialcpav25_gen.infer.calibrate import _zinb_detection

    return _zinb_detection(
        mu.numpy().astype(np.float64),
        theta.numpy().astype(np.float64),
        pi.numpy().astype(np.float64),
    ).mean(axis=0)


def _mv_slope(counts: np.ndarray) -> float:
    """Log-log slope of variance against mean over genes — T06's mean-variance relation."""
    mean = counts.mean(axis=0)
    variance = counts.var(axis=0)
    ok = (mean > 0.05) & (variance > 0)
    return float(np.polyfit(np.log(mean[ok]), np.log(variance[ok]), 1)[0])


@pytest.mark.slow
def test_detection_calibration_matches_the_fold_it_was_fitted_on(trained_for_calibration):
    """The solver hits its target: fitted on a fold and applied there, detection error falls.

    This is the machinery, isolated from the question of whether the correction *transfers*
    between sections — which on this fixture it does not, and
    ``test_detection_calibration_does_not_transfer_between_sections`` is where that is
    recorded with its numbers.
    """
    from spatialcpav25_gen.infer.calibrate import (
        DetectionCalibration as _Calibration,
    )
    from spatialcpav25_gen.infer.calibrate import (
        _decode_hidden,
        _fold_statistics,
    )

    model, training, cfg = trained_for_calibration
    section = training.sections[3]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        fold = _decode_hidden(model, section, cfg, seed=SEED + 99)
        stats = _fold_statistics(model, section, cfg, seed=SEED + 99)
    real = (fold.real > 0).mean(axis=0)
    oracle = _Calibration(
        pi_scale=1.0,
        pi_shift=stats["pi_shift"],
        theta_scale=1.0,
        theta_shift=stats["theta_shift"],
        section_ids=(section.section_id,),
        detection_gen=stats["rate_gen"],
        detection_real=stats["rate_real"],
    )
    mu, theta, pi = oracle.apply(fold.mu, fold.theta, fold.pi, cfg)
    before = float(np.mean(np.abs(_detection_rate(fold.mu, fold.theta, fold.pi) - real)))
    after = float(np.mean(np.abs(_detection_rate(mu, theta, pi) - real)))
    assert after < before, f"detection MAE {before:.4f} -> {after:.4f}"
    assert not np.allclose(theta.numpy(), fold.theta.numpy()), "the theta half did nothing"

    # ...and the theta half does the job pi cannot: the mean-variance relation moves toward
    # the real section's. `design/v23_design.md` §3.5 asks for both corrections and `specs/`
    # had only `pi` (SPEC_QUESTIONS D-table); this is the property that distinguishes them,
    # and the one T06's `test_mean_variance_relation` measures at inference.
    from spatialcpav25_gen.model.expression import sample_counts

    real_slope = _mv_slope(fold.real)
    plain = _mv_slope(sample_counts(fold.mu, fold.theta, fold.pi, np.random.default_rng(0)).numpy())
    calibrated = _mv_slope(sample_counts(mu, theta, pi, np.random.default_rng(0)).numpy())
    assert abs(calibrated - real_slope) <= abs(plain - real_slope) + 1e-9, (
        f"mean-variance slope: real {real_slope:.3f}, uncalibrated {plain:.3f}, "
        f"calibrated {calibrated:.3f}"
    )


@pytest.mark.slow
def test_detection_calibration_does_not_transfer_between_sections(trained_for_calibration):
    """**A recorded negative result**: there is no headroom for the correction on this fixture.

    The model's own per-gene detection error is *smaller* than the section-to-section
    variation of the real rates, so a correction fitted on other sections imports more of that
    variation than it removes model error. Measured at 600 steps: model error **0.0217**, real
    between-section variation **0.040**, cross-fold **0.0326**, and the oracle — fitted on the
    very section it is applied to, which no leakage-free procedure can reach — only **0.0191**.

    The assertion is the *diagnosis*, not the outcome: the real rates must vary by at least as
    much as the model errs. That is what makes the correction unable to help here, and it is
    what would change on a dataset where the model is genuinely miscalibrated. The calibration
    is therefore **not applied by default** — ``generate_section``'s ``calibration`` argument
    defaults to ``None`` — and T10 decides on real data.
    """
    from spatialcpav25_gen.infer.calibrate import _decode_hidden

    model, training, cfg = trained_for_calibration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        folds = [
            _decode_hidden(model, section, cfg, seed=SEED + index)
            for index, section in enumerate(training.sections)
        ]
    rates = np.stack([(f.real > 0).mean(axis=0) for f in folds])
    between_sections = float(np.abs(rates - rates.mean(axis=0)).mean())
    model_error = float(
        np.mean(np.abs(_detection_rate(folds[3].mu, folds[3].theta, folds[3].pi) - rates[3]))
    )
    assert between_sections > model_error, (
        f"real between-section variation {between_sections:.4f} no longer exceeds the model's "
        f"own error {model_error:.4f}; the correction now has headroom and the transfer "
        "measurement in progress/t09_inference_and_calibration.md needs re-running"
    )
