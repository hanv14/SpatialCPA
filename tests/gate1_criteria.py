"""GATE 1's four acceptance criteria, measured once and consumed twice.

``tests/test_noise.py`` asserts the thresholds; ``scripts/gate1_report.py`` prints them
into ``reports/gate1.md`` with the plots. Keeping the measurement in one place is what
stops the report and the test from drifting into disagreeing about whether the gate
passed — which, for a criterion that is supposed to stop the project, would be the worst
possible bug.

Every number in the spec's "GATE 1 — acceptance criteria" section appears here as a
:class:`Criterion` with its threshold. Rows whose threshold the spec leaves qualitative
("materially lower") are carried as ``report``-only rows: they are printed, they are not
asserted, and the report says so rather than inventing a number.

The constants below (pair counts, sweep grids, probe sizes) are properties of the
*measurement*, not of the model, so they live here rather than in ``Config``: Convention 1
governs the package, and a gate that read its own sample size out of the config could be
made to pass by editing the config.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, Volume, to_xyz
from spatialcpav25_gen.model.noise import (
    GaussianRandomField,
    fit_lengthscale_details,
    matern_correlation,
)

from tests.fixtures.planes import intersection_line, plane_from_normal
from tests.fixtures.synthetic import GroundTruthField

FloatArray = npt.NDArray[np.float64]

# ---- G1.1 ----------------------------------------------------------------------------
COV_N_PAIRS = 4000
"""Point pairs whose covariance is compared with the analytic Matern (the spec's 4000)."""
COV_M_SWEEP = (1024, 2048, 4096)
"""Feature counts the covariance error must decrease over."""
COV_N_FIELD_SEEDS = 8
"""Independent field realisations per M. The reported error is the mean *of the
per-realisation errors*, not the error of an average field, so this does not make the
criterion easier — it makes "the error decreases with M" a statement about the estimator's
mean rather than about one draw of it (SPEC_QUESTIONS B2)."""
COV_MC_CHANNELS_SEED = 5
"""Seed for the cross-check that samples the covariance over the ``d_h`` channels instead
of computing it exactly; reported, not asserted."""
COV_MAX_SCALED_DISTANCE = 3.0
"""Pairs are drawn with ``||p - q||_ell`` uniform on ``[0, 3]``, i.e. spanning correlation
1.0 down to ~0.01. Pairs drawn uniformly in a box would nearly all sit where the kernel is
already zero, and the criterion would be met by predicting zero everywhere."""
COV_ELL_Z_FACTOR = 2.5
"""The probe length-scale is deliberately anisotropic — an isotropic one would not test
the ``||p - q||_ell`` metric that makes oblique sections quantitatively right."""
COV_MAE_THRESHOLD = 0.03

# ---- G1.2 ----------------------------------------------------------------------------
PLANE_TILT_DEG = 40.0
"""Dihedral angle between the axis-aligned plane and the oblique one."""
PLANE_COORD_TOL_UM = 1e-6
"""Agreement required between the two float64 plane->xyz constructions, micrometres."""

# ---- G1.3 ----------------------------------------------------------------------------
MORAN_ERROR_RATIO_THRESHOLD = 0.5
MORAN_PEARSON_THRESHOLD = 0.7
ELL_SWEEP_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
"""The spec's 0.25x - 4x sweep; median Moran's I must be monotone across it."""
ELL_CROSSING_FACTORS = tuple(float(f) for f in np.geomspace(0.25, 4.0, 13))
"""A finer grid, used only to locate the ``ell`` that best reproduces ``I_real``."""
ELL_BEST_TOLERANCE = 0.25
"""How far the fitted ``ell`` may sit from the best-matching one."""
G13_N_SEEDS = 3
"""Count-draw / noise realisations averaged, so a single lucky draw cannot decide a gate."""

# ---- G1.4 ----------------------------------------------------------------------------
THROUGHPUT_N_POINTS = 1_000_000
THROUGHPUT_SECONDS = 5.0
DETERMINISM_N_POINTS = 4096


@dataclass(frozen=True)
class Criterion:
    """One acceptance criterion: what was measured, against what, and whether it passed."""

    key: str
    description: str
    measured: float
    threshold: float | None
    comparison: str
    """One of ``<``, ``>``, ``<=``, ``>=``, ``==`` — or ``report`` for a row the spec
    states qualitatively, which is printed but never asserted."""
    unit: str = ""
    note: str = ""

    @property
    def is_assertable(self) -> bool:
        """False for rows the spec does not give a number for."""
        return self.comparison != "report"

    @property
    def passed(self) -> bool:
        """True if the measurement satisfies the threshold (report-only rows pass)."""
        if not self.is_assertable or self.threshold is None:
            return True
        checks = {
            "<": self.measured < self.threshold,
            ">": self.measured > self.threshold,
            "<=": self.measured <= self.threshold,
            ">=": self.measured >= self.threshold,
            "==": self.measured == self.threshold,
        }
        return checks[self.comparison]

    @property
    def verdict(self) -> str:
        """``PASS`` / ``FAIL`` / ``REPORT``."""
        if not self.is_assertable:
            return "REPORT"
        return "PASS" if self.passed else "FAIL"

    def summary(self) -> str:
        """One-line rendering, used in assertion messages and in the report."""
        target = "-" if self.threshold is None else f"{self.comparison} {self.threshold:g}"
        return (
            f"{self.key} {self.description}: measured {self.measured:.6g}{self.unit} "
            f"(required {target}) -> {self.verdict}"
        )


@dataclass(frozen=True)
class GateSection:
    """The criteria of one gate section, plus whatever the report wants to plot."""

    key: str
    title: str
    criteria: list[Criterion]
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True when every assertable criterion in the section passed."""
        return all(c.passed for c in self.criteria)


def _seed_for(*parts: int) -> int:
    """Derive an independent seed from a list of integers, reproducibly."""
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


# --------------------------------------------------------------------------------------
# graph statistics
# --------------------------------------------------------------------------------------


def knn_graph(coords: FloatArray, k: int) -> tuple[npt.NDArray[np.intp], FloatArray]:
    """Row-normalised kNN graph. ``(N, 2)`` coords -> ``(N, k)`` indices, ``(N, k)`` weights."""
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    idx = tree.query(coords, k=k + 1)[1][:, 1:]
    return idx.astype(np.intp), np.full(idx.shape, 1.0 / k)


def morans_i(x: FloatArray, idx: npt.NDArray[np.intp], weights: FloatArray) -> FloatArray:
    """Per-column Moran's I of ``(N, G)`` ``x`` on a fixed row-normalised graph. ``(G,)``."""
    centred = x - x.mean(axis=0, keepdims=True)
    denom = (centred**2).sum(axis=0)
    lagged = np.einsum("nk,nkg->ng", weights, centred[idx])
    numer = (centred * lagged).sum(axis=0)
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = (x.shape[0] / weights.sum()) * numer[ok] / denom[ok]
    return out


def gearys_c(x: FloatArray, idx: npt.NDArray[np.intp], weights: FloatArray) -> FloatArray:
    """Per-column Geary's C of ``(N, G)`` ``x`` on a fixed row-normalised graph. ``(G,)``."""
    n = x.shape[0]
    centred = x - x.mean(axis=0, keepdims=True)
    denom = (centred**2).sum(axis=0)
    diff = (x[idx] - x[:, None, :]) ** 2
    numer = np.einsum("nk,nkg->g", weights, diff)
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = (n - 1) * numer[ok] / (2.0 * weights.sum() * denom[ok])
    return out


# --------------------------------------------------------------------------------------
# G1.1 — covariance correctness
# --------------------------------------------------------------------------------------


def measure_g1_1(cfg: Config, *, seed: int) -> GateSection:
    """Realised vs analytic anisotropic Matern covariance, over a sweep of ``M``.

    Pairs are drawn at a *scaled* distance ``||p - q||_ell`` uniform on ``[0, 3]``, with a
    deliberately anisotropic ``ell``, so the comparison exercises the anisotropic metric
    over the whole range where the kernel is non-trivial.
    """
    ell = (cfg.ell_xy, cfg.ell_xy, cfg.ell_xy * COV_ELL_Z_FACTOR)
    gen = np.random.default_rng(_seed_for(seed, 1))
    box = 10.0 * max(ell)
    p = gen.uniform(0.0, box, size=(COV_N_PAIRS, 3))
    direction = gen.standard_normal((COV_N_PAIRS, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = gen.uniform(0.0, COV_MAX_SCALED_DISTANCE, size=COV_N_PAIRS)
    q = p + radius[:, None] * direction * np.asarray(ell)[None, :]
    analytic = matern_correlation(radius, cfg.matern_nu)

    errors: dict[int, float] = {}
    spread: dict[int, float] = {}
    realised_by_m: dict[int, FloatArray] = {}
    for n_rff in COV_M_SWEEP:
        probe_cfg = cfg.replace(n_rff=n_rff)
        per_realisation: list[float] = []
        for realisation in range(COV_N_FIELD_SEEDS):
            grf = GaussianRandomField(probe_cfg, ell, _seed_for(seed, 2, n_rff, realisation))
            realised = grf.covariance(p, q)
            per_realisation.append(float(np.abs(realised - analytic).mean()))
            if realisation == 0:
                realised_by_m[n_rff] = realised
        errors[n_rff] = float(np.mean(per_realisation))
        spread[n_rff] = float(np.std(per_realisation))

    steps = [errors[b] - errors[a] for a, b in itertools.pairwise(COV_M_SWEEP)]

    # Cross-check: the covariance the model's own d_h channels actually realise. Its
    # expected scatter about the exact value is 1/sqrt(d_h) per pair, which is why it is
    # reported rather than asserted (at d_h = 64 that floor, ~0.125, is four times the
    # criterion's 0.03 and would say nothing about the field).
    mc_field = GaussianRandomField(cfg, ell, _seed_for(seed, 2, COV_MC_CHANNELS_SEED))
    sampled = (mc_field.evaluate_numpy(p) * mc_field.evaluate_numpy(q)).mean(axis=1)
    exact = mc_field.covariance(p, q)
    return GateSection(
        key="G1.1",
        title="Covariance correctness",
        criteria=[
            Criterion(
                key="G1.1a",
                description=(
                    f"mean |realised - analytic Matern| over {COV_N_PAIRS} pairs at "
                    f"M = {COV_M_SWEEP[-1]}, averaged over {COV_N_FIELD_SEEDS} realisations"
                ),
                measured=errors[COV_M_SWEEP[-1]],
                threshold=COV_MAE_THRESHOLD,
                comparison="<",
                note=f"per-realisation sd {spread[COV_M_SWEEP[-1]]:.2g}",
            ),
            Criterion(
                key="G1.1b",
                description=(
                    "covariance error decreases at every step of M = "
                    + " -> ".join(str(m) for m in COV_M_SWEEP)
                    + " (worst step shown)"
                ),
                measured=max(steps),
                threshold=0.0,
                comparison="<",
                note=", ".join(f"M={m}: {errors[m]:.5f}" for m in COV_M_SWEEP),
            ),
            Criterion(
                key="G1.1c",
                description=(
                    f"RMS gap between the covariance sampled over the model's {cfg.latent_dim} "
                    f"channels and the exact one (expected 1/sqrt(d_h) = "
                    f"{1.0 / np.sqrt(cfg.latent_dim):.3f})"
                ),
                measured=float(np.sqrt(((sampled - exact) ** 2).mean())),
                threshold=None,
                comparison="report",
            ),
        ],
        artifacts={
            "ell": ell,
            "errors": errors,
            "spread": spread,
            "radius": radius,
            "analytic": analytic,
            "realised": realised_by_m,
            "n_realisations": COV_N_FIELD_SEEDS,
        },
    )


# --------------------------------------------------------------------------------------
# G1.2 — exact intersection consistency
# --------------------------------------------------------------------------------------


def measure_g1_2(cfg: Config, vol: Volume, *, seed: int) -> GateSection:
    """Two non-parallel planes must give bit-identical noise along the line they share."""
    lo = vol.bbox[0].astype(np.float64)
    hi = vol.bbox[1].astype(np.float64)
    centre = 0.5 * (lo + hi)
    tilt = np.deg2rad(PLANE_TILT_DEG)
    plane_a = plane_from_normal([0.0, 0.0, 1.0], centre)
    plane_b = plane_from_normal([0.0, np.sin(tilt), np.cos(tilt)], centre)

    point, direction = intersection_line(plane_a, plane_b)
    half = 0.5 * float(np.min(hi[:2] - lo[:2]))
    t = np.linspace(-half, half, cfg.sefl_n_line_points)
    line = point[None, :] + t[:, None] * direction[None, :]

    xyz_a = plane_a.to_xyz(plane_a.to_uv(line))
    xyz_b = plane_b.to_xyz(plane_b.to_uv(line))
    coord_gap = float(np.abs(xyz_a - xyz_b).max())
    differing_coords = int((xyz_a.astype(np.float32) != xyz_b.astype(np.float32)).sum())

    grf = GaussianRandomField(cfg, (cfg.ell_xy, cfg.ell_xy, cfg.ell_z), _seed_for(seed, 3))
    xi_a = grf(xyz_a)
    xi_b = grf(xyz_b)
    xi_a_again = grf(xyz_a)
    field_gap = float((xi_a - xi_b).abs().max())

    return GateSection(
        key="G1.2",
        title="Exact intersection consistency",
        criteria=[
            Criterion(
                key="G1.2a",
                description=(
                    f"max |xyz| difference between the two plane pathways over "
                    f"{cfg.sefl_n_line_points} points on the intersection line"
                ),
                measured=coord_gap,
                threshold=PLANE_COORD_TOL_UM,
                comparison="<",
                unit=" um",
                note=(
                    "two orthonormal bases reach the same point by different arithmetic, so "
                    "the float64 reconstructions agree to rounding, not exactly "
                    "(SPEC_QUESTIONS B1)"
                ),
            ),
            Criterion(
                key="G1.2b",
                description="coordinates differing after rounding to the float32 query grid",
                measured=float(differing_coords),
                threshold=0.0,
                comparison="==",
            ),
            Criterion(
                key="G1.2c",
                description="max |xi| difference between the two plane pathways (bitwise: 0)",
                measured=field_gap,
                threshold=0.0,
                comparison="==",
            ),
            Criterion(
                key="G1.2d",
                description="the field is pure: same query twice, max difference",
                measured=float((xi_a - xi_a_again).abs().max()),
                threshold=0.0,
                comparison="==",
            ),
        ],
        artifacts={
            "line": line,
            "xi": xi_a.numpy(),
            "coord_gap": coord_gap,
            "angle_deg": PLANE_TILT_DEG,
        },
    )


# --------------------------------------------------------------------------------------
# G1.3 — autocorrelation transfer (the decisive test)
# --------------------------------------------------------------------------------------


def _section_expression(
    gt: GroundTruthField, section: Section, latent: FloatArray, count_seed: int
) -> FloatArray:
    """Push a latent through the fixture's generative map. ``(N, L)`` -> ``(N, G)`` log1p counts."""
    xyz = to_xyz(section).astype(np.float64)
    mu = gt.expression_mu(latent, xyz, section.cell_type)
    return np.log1p(gt.sample_counts(mu, seed=count_seed).astype(np.float64))


def _autocorrelation(
    values: FloatArray, idx: npt.NDArray[np.intp], weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Per-gene Moran's I and Geary's C."""
    return morans_i(values, idx, weights), gearys_c(values, idx, weights)


def measure_g1_3(cfg: Config, vol: Volume, gt: GroundTruthField, *, seed: int) -> GateSection:
    """Does a correlated prior actually preserve spatial autocorrelation downstream?

    The fixture's own generative map is reused with the latent replaced by (a) i.i.d.
    noise and (b) the GRF at the fitted ``ell``. Everything else — positions, cell types,
    depth, the count draw's seed — is held fixed, so the only thing that differs between
    the two arms is the prior.
    """
    section = vol.sections[len(vol.sections) // 2]
    coords = np.asarray(section.coords, dtype=np.float64)
    xyz = to_xyz(section).astype(np.float64)
    idx, weights = knn_graph(coords, cfg.metric_knn_k)

    real = np.log1p(np.asarray(section.counts.toarray(), dtype=np.float64))
    i_real, c_real = _autocorrelation(real, idx, weights)

    fit = fit_lengthscale_details(vol, cfg, seed=_seed_for(seed, 4))
    ell = fit.ell
    base_field = GaussianRandomField(cfg.replace(latent_dim=gt.latent_dim), ell, _seed_for(seed, 5))

    iid_errors: list[float] = []
    grf_errors: list[float] = []
    iid_r: list[float] = []
    grf_r: list[float] = []
    iid_geary: list[float] = []
    grf_geary: list[float] = []
    finite = np.isfinite(i_real) & np.isfinite(c_real)
    for repeat in range(G13_N_SEEDS):
        count_seed = _seed_for(seed, 6, repeat)
        gen = np.random.default_rng(_seed_for(seed, 7, repeat))
        iid_latent = gen.standard_normal((section.n_cells, gt.latent_dim))
        grf_latent = base_field.evaluate_numpy(xyz).astype(np.float64)
        for latent, errors, radii, geary in (
            (iid_latent, iid_errors, iid_r, iid_geary),
            (grf_latent, grf_errors, grf_r, grf_geary),
        ):
            values = _section_expression(gt, section, latent, count_seed)
            i_gen, c_gen = _autocorrelation(values, idx, weights)
            ok = finite & np.isfinite(i_gen) & np.isfinite(c_gen)
            errors.append(float(np.median(np.abs(i_gen[ok] - i_real[ok]))))
            radii.append(float(np.corrcoef(i_gen[ok], i_real[ok])[0, 1]))
            geary.append(float(np.median(np.abs(c_gen[ok] - c_real[ok]))))

    median_i_real = float(np.median(i_real[finite]))
    sweep = _ell_sweep(
        cfg, gt, section, base_field, ell, idx, weights, seed=seed, factors=ELL_SWEEP_FACTORS
    )
    crossing = _ell_sweep(
        cfg, gt, section, base_field, ell, idx, weights, seed=seed, factors=ELL_CROSSING_FACTORS
    )
    best_factor = _best_matching_factor(ELL_CROSSING_FACTORS, crossing, median_i_real)
    steps = [b - a for a, b in itertools.pairwise(sweep)]

    mean_iid = float(np.mean(iid_errors))
    mean_grf = float(np.mean(grf_errors))
    return GateSection(
        key="G1.3",
        title="Autocorrelation transfer",
        criteria=[
            Criterion(
                key="G1.3a",
                description=(
                    "median |I_gen - I_real| with the GRF prior, as a fraction of the same "
                    "quantity with an i.i.d. prior"
                ),
                measured=mean_grf / mean_iid if mean_iid > 0 else float("inf"),
                threshold=MORAN_ERROR_RATIO_THRESHOLD,
                comparison="<",
                note=f"GRF {mean_grf:.4f} vs i.i.d. {mean_iid:.4f}",
            ),
            Criterion(
                key="G1.3b",
                description="Pearson r between per-gene I_gen and I_real, GRF prior",
                measured=float(np.mean(grf_r)),
                threshold=MORAN_PEARSON_THRESHOLD,
                comparison=">",
            ),
            Criterion(
                key="G1.3c",
                description=(
                    "median I_gen is monotone increasing over ell = "
                    + " / ".join(f"{f:g}x" for f in ELL_SWEEP_FACTORS)
                    + " (smallest step shown)"
                ),
                measured=float(min(steps)),
                threshold=0.0,
                comparison=">",
            ),
            Criterion(
                key="G1.3d",
                description="|fitted ell - best-matching ell| / best-matching ell",
                measured=abs(1.0 / best_factor - 1.0) if best_factor > 0 else float("inf"),
                threshold=ELL_BEST_TOLERANCE,
                comparison="<",
                note=(
                    f"fitted ell_xy = {ell[0]:.1f} um, best match at {best_factor:.3g}x = "
                    f"{ell[0] * best_factor:.1f} um"
                ),
            ),
            Criterion(
                key="G1.3e",
                description="Pearson r with the i.i.d. prior (the spec asks only that it be lower)",
                measured=float(np.mean(iid_r)),
                threshold=None,
                comparison="report",
                note=f"GRF r = {float(np.mean(grf_r)):.4f}",
            ),
            Criterion(
                key="G1.3f",
                description="median |C_gen - C_real|, GRF as a fraction of i.i.d. (Geary's C)",
                measured=(
                    float(np.mean(grf_geary) / np.mean(iid_geary))
                    if np.mean(iid_geary) > 0
                    else float("inf")
                ),
                threshold=None,
                comparison="report",
                note=(
                    f"GRF {float(np.mean(grf_geary)):.4f} vs i.i.d. "
                    f"{float(np.mean(iid_geary)):.4f}"
                ),
            ),
        ],
        artifacts={
            "ell": ell,
            "fit": fit,
            "i_real": i_real,
            "median_i_real": median_i_real,
            "sweep_factors": ELL_SWEEP_FACTORS,
            "sweep_median_i": sweep,
            "crossing_factors": ELL_CROSSING_FACTORS,
            "crossing_median_i": crossing,
            "best_factor": best_factor,
            "iid_errors": iid_errors,
            "grf_errors": grf_errors,
            "section_id": section.section_id,
            "n_cells": section.n_cells,
        },
    )


def _ell_sweep(
    cfg: Config,
    gt: GroundTruthField,
    section: Section,
    base_field: GaussianRandomField,
    ell: tuple[float, float, float],
    idx: npt.NDArray[np.intp],
    weights: FloatArray,
    *,
    seed: int,
    factors: tuple[float, ...],
) -> list[float]:
    """Median per-gene Moran's I of a generated section, as ``ell`` is scaled by each factor.

    The same realisation is rescaled with :meth:`GaussianRandomField.with_lengthscale` and
    the count draw uses one fixed seed, so consecutive points differ only in ``ell`` —
    which is the condition under which "monotone in ``ell``" means anything, and is also
    what T09's bisection will rely on.
    """
    xyz = to_xyz(section).astype(np.float64)
    count_seed = _seed_for(seed, 8)
    medians: list[float] = []
    for factor in factors:
        scaled = base_field.with_lengthscale(tuple(v * factor for v in ell))  # type: ignore[arg-type]
        latent = scaled.evaluate_numpy(xyz).astype(np.float64)
        values = _section_expression(gt, section, latent, count_seed)
        moran = morans_i(values, idx, weights)
        medians.append(float(np.median(moran[np.isfinite(moran)])))
    return medians


def _best_matching_factor(factors: tuple[float, ...], medians: list[float], target: float) -> float:
    """The ``ell`` factor at which median ``I_gen`` crosses ``I_real``.

    Linear interpolation in ``log(ell)`` between the two sweep points that bracket the
    target; if nothing brackets it, the closest sweep point, which is the honest answer
    when the target lies outside the range that was swept.
    """
    log_factors = np.log(np.asarray(factors, dtype=np.float64))
    values = np.asarray(medians, dtype=np.float64)
    for i in range(len(values) - 1):
        lo, hi = values[i], values[i + 1]
        if (lo - target) * (hi - target) <= 0.0 and hi != lo:
            weight = (target - lo) / (hi - lo)
            return float(np.exp(log_factors[i] + weight * (log_factors[i + 1] - log_factors[i])))
    return float(factors[int(np.argmin(np.abs(values - target)))])


FOV_DIAGNOSTIC_EXTENT_UM = 3000.0
"""In-plane extent of the diagnostic volume: 3x the fixture's, same cell density."""
FOV_DIAGNOSTIC_CELLS = 13_500
"""Cells per section, chosen to hold the fixture's density at the larger extent."""


def measure_fov_diagnostic(cfg: Config, *, seed: int) -> GateSection:
    """Diagnostic, **not** a gate criterion: the same sweep on a 3x wider section.

    G1.3c asks whether median ``I_gen`` is monotone in ``ell``. Moran's I of *expression*
    is a ratio — spatially structured variance over total variance — and a stationary
    field's variance *within a finite window* falls once its correlation length approaches
    the window. This rebuilds the fixture with a 3000 um field of view instead of 1000 um,
    everything else equal, to show how much of the shape of ``I_gen(ell)`` is the field and
    how much is the window.
    """
    from tests.fixtures.synthetic import make_synthetic_volume

    vol, gt = make_synthetic_volume(
        seed=0, extent_xy=FOV_DIAGNOSTIC_EXTENT_UM, n_cells_per_section=FOV_DIAGNOSTIC_CELLS
    )
    section = vol.sections[len(vol.sections) // 2]
    idx, weights = knn_graph(np.asarray(section.coords, dtype=np.float64), cfg.metric_knn_k)
    real = np.log1p(np.asarray(section.counts.toarray(), dtype=np.float64))
    i_real = morans_i(real, idx, weights)
    median_i_real = float(np.median(i_real[np.isfinite(i_real)]))

    ell = fit_lengthscale_details(vol, cfg, seed=_seed_for(seed, 4)).ell
    base_field = GaussianRandomField(cfg.replace(latent_dim=gt.latent_dim), ell, _seed_for(seed, 5))
    sweep = _ell_sweep(
        cfg, gt, section, base_field, ell, idx, weights, seed=seed, factors=ELL_SWEEP_FACTORS
    )
    steps = [b - a for a, b in itertools.pairwise(sweep)]
    return GateSection(
        key="D1",
        title=(
            "Diagnostic: the same ell sweep at a "
            f"{FOV_DIAGNOSTIC_EXTENT_UM:.0f} um field of view"
        ),
        criteria=[
            Criterion(
                key="D1a",
                description="smallest step of median I_gen over the same 0.25x - 4x sweep",
                measured=float(min(steps)),
                threshold=None,
                comparison="report",
                note=", ".join(
                    f"{f:g}x: {v:.4f}" for f, v in zip(ELL_SWEEP_FACTORS, sweep, strict=False)
                ),
            ),
            Criterion(
                key="D1b",
                description="median I_real on the wider section, for reference",
                measured=median_i_real,
                threshold=None,
                comparison="report",
                note=f"fitted ell_xy = {ell[0]:.1f} um",
            ),
        ],
        artifacts={
            "ell": ell,
            "sweep_median_i": sweep,
            "median_i_real": median_i_real,
            "extent_um": FOV_DIAGNOSTIC_EXTENT_UM,
        },
    )


def measure_latent_sweep(
    cfg: Config, vol: Volume, gt: GroundTruthField, *, seed: int
) -> GateSection:
    """Diagnostic: Moran's I of the **field itself**, not of the expression it drives.

    Separates the two things G1.3c could be measuring. If the field's own autocorrelation
    is monotone in ``ell`` and the expression's is not, the non-monotonicity lives in the
    observation model, not in the prior.
    """
    section = vol.sections[len(vol.sections) // 2]
    xyz = to_xyz(section).astype(np.float64)
    idx, weights = knn_graph(np.asarray(section.coords, dtype=np.float64), cfg.metric_knn_k)
    ell = fit_lengthscale_details(vol, cfg, seed=_seed_for(seed, 4)).ell
    base_field = GaussianRandomField(cfg.replace(latent_dim=gt.latent_dim), ell, _seed_for(seed, 5))
    medians: list[float] = []
    within_sd: list[float] = []
    for factor in ELL_SWEEP_FACTORS:
        latent = (
            base_field.with_lengthscale(
                tuple(v * factor for v in ell)  # type: ignore[arg-type]
            )
            .evaluate_numpy(xyz)
            .astype(np.float64)
        )
        medians.append(float(np.median(morans_i(latent, idx, weights))))
        within_sd.append(float(latent.std(axis=0).mean()))
    steps = [b - a for a, b in itertools.pairwise(medians)]
    return GateSection(
        key="D2",
        title="Diagnostic: Moran's I of the noise field itself",
        criteria=[
            Criterion(
                key="D2a",
                description=(
                    "smallest step of median I of the GRF channels over the 0.25x - 4x sweep"
                ),
                measured=float(min(steps)),
                threshold=None,
                comparison="report",
                note=", ".join(
                    f"{f:g}x: {v:.4f}" for f, v in zip(ELL_SWEEP_FACTORS, medians, strict=False)
                ),
            ),
            Criterion(
                key="D2b",
                description="within-section sd of the unit-variance field at the largest ell",
                measured=within_sd[-1],
                threshold=None,
                comparison="report",
                note=", ".join(
                    f"{f:g}x: {v:.3f}" for f, v in zip(ELL_SWEEP_FACTORS, within_sd, strict=False)
                ),
            ),
        ],
        artifacts={"medians": medians, "within_sd": within_sd, "ell": ell},
    )


# --------------------------------------------------------------------------------------
# G1.4 — determinism and scaling
# --------------------------------------------------------------------------------------


_CHILD_PROGRAM = """
import numpy as np, sys
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.model.noise import GaussianRandomField
cfg = Config()
field = GaussianRandomField(cfg, ({ell_x}, {ell_y}, {ell_z}), {seed})
points = np.random.default_rng({point_seed}).uniform(0.0, 1000.0, size=({n}, 3))
np.save(sys.argv[1], field(points).numpy())
"""


def measure_g1_4(cfg: Config, *, seed: int) -> GateSection:
    """Same seed, two processes, identical bits; and the cost of a million queries."""
    import tempfile
    from pathlib import Path

    ell = (cfg.ell_xy, cfg.ell_xy, cfg.ell_z)
    field_seed = _seed_for(seed, 9)
    point_seed = _seed_for(seed, 10)
    points = np.random.default_rng(point_seed).uniform(0.0, 1000.0, size=(DETERMINISM_N_POINTS, 3))
    here = GaussianRandomField(cfg, ell, field_seed)(points).numpy()

    program = _CHILD_PROGRAM.format(
        ell_x=ell[0],
        ell_y=ell[1],
        ell_z=ell[2],
        seed=field_seed,
        point_seed=point_seed,
        n=DETERMINISM_N_POINTS,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "child.npy"
        subprocess.run(
            [sys.executable, "-c", program, str(out)],
            check=True,
            capture_output=True,
        )
        there = np.load(out)
    differing = int((here != there).sum())

    big = np.random.default_rng(_seed_for(seed, 11)).uniform(
        0.0, 1000.0, size=(THROUGHPUT_N_POINTS, 3)
    )
    field = GaussianRandomField(cfg, ell, field_seed)
    field(big[:1024])  # touch the buffers once so the timing is not a first-call artefact
    start = time.perf_counter()
    with torch.no_grad():
        field(big)
    elapsed = time.perf_counter() - start

    return GateSection(
        key="G1.4",
        title="Determinism and scaling",
        criteria=[
            Criterion(
                key="G1.4a",
                description=(
                    f"values differing between two processes at the same seed "
                    f"({DETERMINISM_N_POINTS} x {cfg.latent_dim})"
                ),
                measured=float(differing),
                threshold=0.0,
                comparison="==",
            ),
            Criterion(
                key="G1.4b",
                description=(
                    f"wall clock to query {THROUGHPUT_N_POINTS:,} points at M = {cfg.n_rff}, "
                    f"d_h = {cfg.latent_dim}"
                ),
                measured=elapsed,
                threshold=THROUGHPUT_SECONDS,
                comparison="<",
                unit=" s",
                note="single CPU, no GPU; hardware-dependent (SPEC_QUESTIONS B9)",
            ),
        ],
        artifacts={"seconds": elapsed, "points_per_second": THROUGHPUT_N_POINTS / elapsed},
    )
