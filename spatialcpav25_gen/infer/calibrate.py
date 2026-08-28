"""Leakage-free calibration of the free statistical parameters (T09 §2).

Four quantities are calibrated here, all of them on **training sections only**:

* ``ell_xy`` — bisected so a *generated* section's mean Moran's I matches the mean Moran's I
  of the **flanking training sections** (:func:`calibrate_lengthscale`);
* ``ell_z`` — matched against the *observed* between-section correlation decay
  (:func:`calibrate_ell_z`), which is open risk **R1**'s adopted remedy 2, with remedy 3 —
  the variogram fit enters as the bracket's **upper endpoint**, never as a value — as the
  guard;
* the per-gene ``pi`` and ``log theta`` corrections (:func:`calibrate_detection`), fitted on
  LOSO reconstructions and applied at generation;
* the anchor weight ``w(v)`` (:func:`calibrate_anchor_weight`), a 1-D isotonic regression
  from latent variance to Bernoulli mixing weight, which is what replaces the previous
  version's hand-tuned gap heuristics entirely.

Leakage is prevented by type, not by care: every entry point runs
:func:`~spatialcpav25_gen.losses.metric_aware.require_training_volume`, so a
:class:`~spatialcpav25_gen.data.schema.HeldOutSections` — and a plain ``Volume``, which may
still contain the evaluation sections — is a ``TypeError`` before anything is measured.

The length-scale objective is unimodal, not monotone
----------------------------------------------------
Measured at T03/GATE 1 and the reason this module is more than a bisection: mean ``I_gen``
rises with ``ell``, turns over, and falls, because Moran's I is structured variance over
*total* variance and a stationary field loses within-window variance once its correlation
length approaches the section. The maximiser sat at 0.086 of the in-plane extent on the
3000 um gate fixture and 0.112 on the 1000 um one — at 2.5x and 0.8x the *fitted* ``ell``
respectively. So the calibrator caps the bracket, **locates the maximum on a log grid**,
bisects only on the monotone branch below it, and returns a status. A bisection that
ignored this would walk into a boundary and return it as though it were an answer
(Convention 6).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume, section_seed
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.losses.metric_aware import (
    knn_weight_graph,
    morans_i,
    normalised_linear,
    require_training_volume,
)
from spatialcpav25_gen.model.expression import draw_mean_variance
from spatialcpav25_gen.model.noise import fit_lengthscale_from_sections

if TYPE_CHECKING:  # pragma: no cover - the model imports this package, not the other way
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "CALIBRATION_STATUSES",
    "CalibrationError",
    "CalibrationNotAppliedWarning",
    "DetectionCalibration",
    "IsotonicRegressor",
    "LengthscaleCalibration",
    "RetrievalWindowCalibration",
    "UnreachableTargetWarning",
    "apply_lengthscale",
    "calibrate_anchor_weight",
    "calibrate_detection",
    "calibrate_ell_z",
    "calibrate_lengthscale",
    "calibrate_retrieval_window",
    "mean_morans_i",
    "observed_z_decay",
    "section_grid_means",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

CALIBRATION_STATUSES: Final[frozenset[str]] = frozenset(
    {"converged", "target_unreachable", "boundary"}
)
"""The three outcomes a calibrator may report (``specs/09`` §2).

``converged``
    The achieved value is within ``Config.calibration_morans_tol`` of the target.
``target_unreachable``
    The target lies **above the objective's maximum** over the whole bracket, so there is no
    root at all. The value returned is the grid maximiser — the closest the objective can
    come — and it is never presented as a converged answer.
``boundary``
    The answer is limited by the bracket rather than by the data: the root lies outside the
    bracket on the low side, or the bisection ran out of iterations without meeting the
    tolerance. A run that reports this has a number to use and a reason to distrust it.
"""

_EPS: Final[float] = 1e-9

_CALIBRATION_ALTERNATIONS: Final[int] = 2
"""Alternations between the ``theta`` and ``pi`` solves in :func:`_fold_statistics`.

Not a ``Config`` field: it is the fixed point of a two-variable solve, not a tunable. Two
passes bring the coupled solution within a thousandth of both targets on the fixture, and a
third changes nothing measurable."""


class CalibrationError(ValueError):
    """Raised when a calibration cannot be run on the volume it was given."""


class UnreachableTargetWarning(UserWarning):
    """A calibration target lies outside what the objective can reach on its bracket.

    Carries both numbers, because "the calibration did not converge" without them is not a
    diagnosis. For ``ell_z`` this is open risk **R1**'s guard firing: the value returned is a
    **bound**, not a fitted length-scale. Which bound depends on where the objective ran out.
    Measured under ``expr_mode="zinb-flow"``, where the objective is live and monotone
    increasing, it terminates at the bracket's **top** and still undershoots — so the data say
    the true ``ell_z`` is at least that large, and the number is a **lower** bound on the
    tissue even though it is the largest value the search was allowed to try.
    """


class CalibrationNotAppliedWarning(UserWarning):
    """A calibrated length-scale was dropped rather than written into the ``Config``.

    Raised by :func:`apply_lengthscale` for an axis whose status is not ``"converged"``.
    Carries the achieved and target values, because "not applied" without them does not say
    whether the search was close or the objective was dead.
    """


# --------------------------------------------------------------------------------------
# the observables the calibrators match
# --------------------------------------------------------------------------------------


def mean_morans_i(
    counts: npt.NDArray[Any], coords: npt.NDArray[Any], cfg: Config, *, seed: int
) -> float:
    """Mean over genes of Moran's I of one section. ``(N, G)``, ``(N, D)`` -> float.

    Library-size normalised on the linear scale (the scale T08's autocorrelation term runs
    on), on the row-normalised kNN graph of ``Config.metric_knn_k`` neighbours — the degree
    the vendored paper metric uses, so this number and T10's are the same statistic.

    Subsampled to ``Config.calibration_max_cells`` cells at ``seed``: one bisection is ~20
    generations and both sides of the comparison are capped the same way, so the cap changes
    the estimate's variance and not its expectation.
    """
    x = np.asarray(counts, dtype=np.float64)
    p = np.asarray(coords, dtype=np.float64)
    if x.shape[0] != p.shape[0]:
        raise CalibrationError(
            f"mean_morans_i: {x.shape[0]} rows of counts against {p.shape[0]} coordinates"
        )
    keep = _subsample(x.shape[0], int(cfg.calibration_max_cells), seed)
    x, p = x[keep], p[keep]
    if x.shape[0] < int(cfg.metric_knn_k) + 1:
        raise CalibrationError(
            f"mean_morans_i: {x.shape[0]} cells cannot support a "
            f"{cfg.metric_knn_k}-nearest-neighbour graph"
        )
    graph = knn_weight_graph(p, cfg)
    values = normalised_linear(torch.from_numpy(x.astype(np.float32)), 1.0, float(cfg.metric_eps))
    per_gene = morans_i(values, graph, eps=float(cfg.metric_eps)).numpy()
    return float(np.nanmean(per_gene))


def _subsample(n: int, cap: int, seed: int) -> IntArray:
    """Sorted indices of at most ``cap`` of ``n`` rows; all of them when ``n <= cap``."""
    if n <= cap:
        return np.arange(n, dtype=np.intp)
    gen = np.random.default_rng(seed)
    return np.sort(gen.choice(n, size=cap, replace=False)).astype(np.intp)


def section_grid_means(
    counts: npt.NDArray[Any],
    coords: npt.NDArray[Any],
    bounds: npt.NDArray[Any],
    cfg: Config,
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """In-plane grid means of one section's expression. ``(G*G, P)`` and ``(G*G,)`` bool.

    Cells in two different sections have no correspondence, so anything compared *between*
    sections has to be compared between spatial averages. The grid is
    ``Config.variogram_z_grid_size`` squared over ``bounds`` (a ``(2, 2)`` in-plane min/max),
    the values are library-size-normalised ``log1p``, and a grid cell counts as filled only
    when it holds at least ``Config.variogram_z_min_cells_per_cell`` cells.

    The same construction as the along-z arm of
    :func:`~spatialcpav25_gen.model.noise.fit_lengthscale_from_sections`, deliberately: the
    ``ell_z`` calibrator has to match the observable that fit models.
    """
    grid = int(cfg.variogram_z_grid_size)
    x = np.asarray(counts, dtype=np.float64)
    totals = np.clip(x.sum(axis=1, keepdims=True), 1.0, None)
    values = np.log1p(x / totals * float(np.median(totals)))
    lo = np.asarray(bounds, dtype=np.float64)[0]
    hi = np.asarray(bounds, dtype=np.float64)[1]
    p = np.asarray(coords, dtype=np.float64)
    index = np.clip(((p - lo) / np.maximum(hi - lo, 1e-9) * grid).astype(int), 0, grid - 1)
    flat = index[:, 0] * grid + index[:, 1]
    occupancy = np.bincount(flat, minlength=grid * grid)
    summed = np.zeros((grid * grid, values.shape[1]), dtype=np.float64)
    np.add.at(summed, flat, values)
    ok = occupancy >= int(cfg.variogram_z_min_cells_per_cell)
    means = np.zeros_like(summed)
    means[ok] = summed[ok] / occupancy[ok, None]
    return means, np.asarray(ok, dtype=np.bool_)


def observed_z_decay(vol: TrainingVolume, cfg: Config) -> tuple[FloatArray, FloatArray]:
    """Between-section correlation of the **real** training sections. ``(P,)``, ``(P,)``.

    ``(dz, r)`` over every ordered pair of training sections: the correlation between their
    grid-averaged log-normalised profiles. This is the observable open risk R1's remedy 2
    calibrates ``ell_z`` against — the *measured* decay rather than a parameter of a model of
    it, which on a nine-section stack is exactly the difference between 200 um and the 561 um
    the variogram extrapolates to.
    """
    require_training_volume(vol, "observed_z_decay")
    box = np.asarray(vol.bbox, dtype=np.float64)[:, :2]
    means = [
        section_grid_means(
            np.asarray(s.counts.todense(), dtype=np.float64),
            np.asarray(s.coords, dtype=np.float64),
            box,
            cfg,
        )
        for s in vol.sections
    ]
    z_values = vol.z_values.astype(np.float64)
    lags: list[float] = []
    correlations: list[float] = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            shared = means[i][1] & means[j][1]
            if int(shared.sum()) < 3:
                continue
            a = means[i][0][shared].ravel()
            b = means[j][0][shared].ravel()
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            lags.append(abs(float(z_values[j] - z_values[i])))
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    if not lags:
        raise CalibrationError(
            f"observed_z_decay: no in-plane grid cell of specimen {vol.specimen_id!r} is "
            f"filled in two sections at Config.variogram_z_grid_size="
            f"{cfg.variogram_z_grid_size}; the between-section correlation is undefined and "
            "ell_z cannot be calibrated against it (open risk R1)"
        )
    return np.asarray(lags, dtype=np.float64), np.asarray(correlations, dtype=np.float64)


def _adjacent_correlation(lags: FloatArray, correlations: FloatArray, spacing: float) -> float:
    """Mean correlation at the shortest lag: the statistic ``ell_z`` is matched on.

    The shortest lag is one section spacing, which is the only lag a short stack measures
    with many pairs, and it is where a wrong ``ell_z`` shows up first: at 561 um against a
    200 um truth the adjacent sections are asked to be nearly identical.
    """
    tolerance = 0.5 * float(spacing)
    shortest = float(np.min(lags))
    near = np.abs(lags - shortest) <= tolerance
    return float(np.mean(correlations[near]))


# --------------------------------------------------------------------------------------
# the search: cap the bracket, find the maximum, bisect below it
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LengthscaleCalibration:
    """What :func:`calibrate_lengthscale` measured, and how far it can be trusted.

    Attributes
    ----------
    ell
        ``(ell_x, ell_y, ell_z)`` micrometres, ``ell_x == ell_y``.
    status
        One of :data:`CALIBRATION_STATUSES`, for the in-plane calibration.
    i_gen, i_target
        The achieved mean Moran's I of a generated section and the target measured on the
        flanking training sections.
    grid_ell, grid_i
        The log grid the maximum was located on, and the objective there. Kept because "the
        target was unreachable" is only actionable beside the curve that could not reach it.
    iterations
        Bisection steps actually taken.
    ell_fitted
        The variogram fit, for reference. Never the returned value on its own.
    ell_z_status
        The ``ell_z`` calibration's own status. ``target_unreachable`` means the returned
        number is a **bound rather than a fit** (open risk R1, remedy 3) — read its direction
        off ``z_achieved`` against ``z_target``: an objective that undershoots at the bracket's
        top makes it a *lower* bound on the tissue's ``ell_z``.
    z_target, z_achieved
        Observed and generated adjacent-section correlation.
    section_ids
        The training sections the target was measured on. Recorded so a report can show that
        no held-out section is among them.
    """

    ell: tuple[float, float, float]
    status: str
    i_gen: float
    i_target: float
    grid_ell: FloatArray
    grid_i: FloatArray
    iterations: int
    ell_fitted: tuple[float, float, float]
    ell_z_status: str = "converged"
    z_target: float = float("nan")
    z_achieved: float = float("nan")
    section_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject a status outside :data:`CALIBRATION_STATUSES` (Convention 6)."""
        for name in ("status", "ell_z_status"):
            value = getattr(self, name)
            if value not in CALIBRATION_STATUSES:
                raise CalibrationError(
                    f"LengthscaleCalibration.{name}={value!r} is not one of "
                    f"{sorted(CALIBRATION_STATUSES)}"
                )

    @property
    def ell_z_is_upper_bound(self) -> bool:
        """Whether ``ell_z`` must be read as a bound rather than as a fitted value."""
        return self.ell_z_status != "converged"


def _bracket(vol: TrainingVolume, cfg: Config, ell_fitted: float) -> tuple[float, float]:
    """Return the capped in-plane search bracket ``(ell_min, ell_max)``, micrometres.

    ``specs/09`` §2.1: ``ell_max = min(calibration_ell_max_extent_frac * extent,
    calibration_ell_max_fitted_multiple * ell_fitted)`` and ``ell_min =
    variogram_ell_min_factor * median_nn_dist``. Neither cap is a guarantee — 0.2 x extent is
    about twice the measured maximiser — so they bound the search; :func:`_locate_and_bisect`
    is what makes it well-posed.
    """
    box = np.asarray(vol.bbox, dtype=np.float64)
    extent = float(np.min(box[1, :2] - box[0, :2]))
    ell_min = float(cfg.variogram_ell_min_factor) * float(vol.median_nn_dist)
    ell_max = min(
        float(cfg.calibration_ell_max_extent_frac) * extent,
        float(cfg.calibration_ell_max_fitted_multiple) * float(ell_fitted),
    )
    if not ell_max > ell_min:
        raise CalibrationError(
            f"calibrate_lengthscale: the bracket is empty — ell_min={ell_min:g} um from "
            f"{cfg.variogram_ell_min_factor} x the median nearest-neighbour distance, "
            f"ell_max={ell_max:g} um from min({cfg.calibration_ell_max_extent_frac} x extent, "
            f"{cfg.calibration_ell_max_fitted_multiple} x the fitted {ell_fitted:g} um). "
            "The fitted length-scale is below one cell spacing; the volume cannot constrain it."
        )
    return ell_min, ell_max


@dataclass
class _SearchResult:
    """The outcome of one 1-D search: value, status, achieved objective, the grid, iterations."""

    value: float
    status: str
    achieved: float
    grid_x: FloatArray = field(default_factory=lambda: np.zeros(0))
    grid_y: FloatArray = field(default_factory=lambda: np.zeros(0))
    iterations: int = 0


def _locate_and_bisect(
    objective: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    cfg: Config,
    *,
    what: str,
) -> _SearchResult:
    """Locate the objective's maximum on ``[lo, hi]``, then bisect below it for ``target``.

    The three-step procedure of ``specs/09`` §2, and the statuses it can return are
    :data:`CALIBRATION_STATUSES`. ``objective`` is assumed to *rise then fall* on the bracket
    (GATE 1's G1.3g measured exactly that, violation 0.000); on the monotone branch below the
    maximiser bisection is well-posed, and above it there is nothing to bisect.

    ``what`` names the quantity in the warning, so a run that reports an unreachable target
    says which one.
    """
    grid_x = np.exp(np.linspace(math.log(lo), math.log(hi), int(cfg.bisection_grid_size)))
    grid_y = np.asarray([objective(float(x)) for x in grid_x], dtype=np.float64)
    best = int(np.argmax(grid_y))
    tol = float(cfg.calibration_morans_tol)

    if target > float(grid_y[best]) + tol:
        warnings.warn(
            f"calibrate: the {what} target {target:.4f} exceeds the objective's maximum "
            f"{float(grid_y[best]):.4f} over the whole bracket "
            f"[{lo:g}, {hi:g}], so there is no root. Returning the grid maximiser "
            f"{float(grid_x[best]):g} with status 'target_unreachable' — it is the closest "
            "the objective can come, and it is not a converged calibration.",
            UnreachableTargetWarning,
            stacklevel=3,
        )
        return _SearchResult(
            value=float(grid_x[best]),
            status="target_unreachable",
            achieved=float(grid_y[best]),
            grid_x=grid_x,
            grid_y=grid_y,
        )

    # Bisect on [lo, argmax], which is monotone increasing by construction.
    left, right = float(lo), float(grid_x[best])
    y_left = float(grid_y[0])
    if target <= y_left:
        status = "converged" if abs(y_left - target) < tol else "boundary"
        return _SearchResult(
            value=left, status=status, achieved=y_left, grid_x=grid_x, grid_y=grid_y
        )

    value, achieved, iterations = left, y_left, 0
    for _ in range(int(cfg.bisection_max_iter)):
        iterations += 1
        value = math.sqrt(left * right)  # geometric midpoint: the grid is logarithmic
        achieved = objective(value)
        if abs(achieved - target) < tol:
            return _SearchResult(
                value=value,
                status="converged",
                achieved=achieved,
                grid_x=grid_x,
                grid_y=grid_y,
                iterations=iterations,
            )
        if achieved < target:
            left = value
        else:
            right = value
    return _SearchResult(
        value=value,
        status="boundary",
        achieved=achieved,
        grid_x=grid_x,
        grid_y=grid_y,
        iterations=iterations,
    )


def _require_ell_reaches_output(cfg: Config, where: str, generates: bool) -> None:
    """Refuse to calibrate ``ell`` when it cannot reach the generated section at all.

    ``ell`` is the **GRF's** length-scale, so it moves the output only if the prior draws from
    the field *and* the expression path reads the latent that prior produces. Two configs break
    that chain, and both make the objective constant rather than merely weak:

    * ``prior_mode = "iid"`` (ablation A1) — the prior is white noise and never queries the
      field;
    * ``expr_mode = "cross-mix"`` (the v20 fallback) — :func:`_cross_mix` copies each emitted
      count **verbatim** from a donor cell and never evaluates the flow, so no latent, and
      therefore no ``ell``, is consulted on the output path.

    Both were measured on the fixture, and both are why this guard exists. Under
    ``prior_mode="iid"`` the first calibration reported ``target_unreachable`` with the
    maximiser pinned at the bracket's low end. Under ``expr_mode="cross-mix"`` the
    between-section objective is **bitwise identical** — 0.6727617248 at every one of
    ``ell_z`` 25, 60, 137, 200, 275 and 364.6 um, a 15x sweep — while the same sweep under
    ``zinb-flow`` rises monotonically 0.8867 -> 0.9316 and the GRF's own between-plane
    correlation rises -0.005 -> 0.921. A flat objective has no maximiser: the grid argmax is
    whichever point ties first, which is the bracket's endpoint, and ``specs/09`` §2 requires
    precisely that an unreachable target *not* return an endpoint. So the answer would look
    like a measurement and be an artefact of tie-breaking.

    ``generates`` is ``False`` when the caller injected its own objective; the search is then
    being tested against a planted curve and no prior is consulted at all.
    """
    if not generates:
        return
    if cfg.prior_mode != "correlated":
        raise CalibrationError(
            f"{where}: Config.prior_mode={cfg.prior_mode!r} does not use the GRF, so ell has "
            "no effect on the generated section and the calibration objective is flat. "
            "Calibrate under prior_mode='correlated' (the shipped default); 'iid' is ablation "
            "A1 and needs no length-scale."
        )
    if cfg.expr_mode == "cross-mix":
        raise CalibrationError(
            f"{where}: Config.expr_mode='cross-mix' emits donor counts verbatim and never "
            "evaluates the flow, so the latent the GRF parameterises never reaches the "
            "output and the calibration objective is constant in ell (measured: identical to "
            "10 decimal places across a 15x ell_z sweep). Calibrate under "
            "expr_mode='zinb-flow' or 'auto-blend'; 'cross-mix' is the v20 fallback and needs "
            "no length-scale."
        )


def apply_lengthscale(cfg: Config, calibration: LengthscaleCalibration) -> Config:
    """Write a calibrated ``ell`` into ``cfg`` so it reaches the prior at generation.

    This is the step that makes a calibration do anything: ``ell`` reaches the GRF only as
    ``Config.ell_xy`` / ``Config.ell_z``, which
    :func:`~spatialcpav25_gen.infer.generate.generate_section` swaps into the field through
    ``with_lengthscale``. Without a writer, ``calibrate_lengthscale`` measures a length-scale
    and generation goes on using the config's own (``specs/09`` §2, added after T09 measured
    that the two were never connected).

    **Only a ``"converged"`` axis is applied.** ``target_unreachable`` and ``boundary`` are the
    two ways the search reports that it did not find a root, and T09 measured what a
    non-converged value is worth: on a flat objective the returned number is whichever grid
    point tied first, so applying it would ship a tie-break as a length-scale. Such an axis is
    **dropped with a** :class:`CalibrationNotAppliedWarning` **naming the achieved and target
    values**, and the config's existing value stands.

    The two axes are decided **separately**, on ``status`` and ``ell_z_status`` respectively,
    because they are separate searches against separate targets: an in-plane calibration that
    converged is not made worthless by a stack too short to constrain ``ell_z``. The cost is
    that a half-applied result has an anisotropy ratio mixing a calibrated axis with a default
    one, which is why each dropped axis says so out loud rather than silently.

    Parameters
    ----------
    cfg
        The config generation will run under.
    calibration
        What :func:`calibrate_lengthscale` returned.

    Returns
    -------
    Config
        ``cfg`` with the converged axes replaced. Returns an equal config when neither axis
        converged — never a partially-mutated one, since ``Config`` is frozen.
    """
    updates: dict[str, float] = {}
    if calibration.status == "converged":
        updates["ell_xy"] = float(calibration.ell[0])
    else:
        warnings.warn(
            f"apply_lengthscale: ell_xy is not applied — status={calibration.status!r}, so "
            f"the search found no root and {calibration.ell[0]:.4g} um is the grid maximiser "
            f"rather than a calibrated value (achieved mean Moran's I {calibration.i_gen:.4f} "
            f"against a target of {calibration.i_target:.4f}). Config.ell_xy stays at "
            f"{float(cfg.ell_xy):.4g} um.",
            CalibrationNotAppliedWarning,
            stacklevel=2,
        )
    if calibration.ell_z_status == "converged":
        updates["ell_z"] = float(calibration.ell[2])
    else:
        warnings.warn(
            f"apply_lengthscale: ell_z is not applied — status={calibration.ell_z_status!r}, "
            f"so {calibration.ell[2]:.4g} um is a bound rather than a calibrated value "
            f"(achieved between-section correlation {calibration.z_achieved:.4f} against an "
            f"observed {calibration.z_target:.4f}; open risk R1). Config.ell_z stays at "
            f"{float(cfg.ell_z):.4g} um.",
            CalibrationNotAppliedWarning,
            stacklevel=2,
        )
    return cfg.replace(**updates) if updates else cfg


def _probe_section(vol: TrainingVolume) -> tuple[Section, list[Section]]:
    """Return the section a calibration generates at, and its flanking training sections.

    The middle of the stack, so both flanks exist and the probe is an interpolation rather
    than the boundary regime of open risk R3.
    """
    require_training_volume(vol, "calibrate_lengthscale")
    if vol.n_sections < 3:
        raise CalibrationError(
            f"calibrate_lengthscale: specimen {vol.specimen_id!r} has {vol.n_sections} "
            "training sections; a flanking pair needs at least 3"
        )
    middle = vol.n_sections // 2
    return vol.sections[middle], [vol.sections[middle - 1], vol.sections[middle + 1]]


def calibrate_lengthscale(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int = 0,
    measure: Callable[[float], float] | None = None,
    measure_z: Callable[[float], float] | None = None,
) -> LengthscaleCalibration:
    """Calibrate ``ell`` against the flanking training sections.

    Returns a :class:`LengthscaleCalibration`.

    ``ell_xy`` is bisected so the **generated** section's mean Moran's I matches the mean
    Moran's I of the **flanking training sections**; ``ell_z`` comes from the observed
    between-section correlation decay (:func:`calibrate_ell_z`). Ground truth is never
    consulted, and there is no parameter through which a held-out section could enter: ``vol``
    is a :class:`~spatialcpav25_gen.data.schema.TrainingVolume`, which only
    :func:`~spatialcpav25_gen.data.loaders.split_holdout` produces.

    Parameters
    ----------
    model, vol, cfg
        The model, its training volume and the generation-time config.
    seed
        Explicit (Convention 3): the generated probe sections and the cell subsample derive
        from it.
    measure
        Additive keyword: the objective ``ell_xy -> mean I_gen``. ``None`` generates a probe
        section per evaluation, which is the production path; a test injects an analytic
        unimodal objective to exercise the search's three branches without 20 generations.
    measure_z
        The same for ``ell_z -> adjacent-section correlation``.
    """
    require_training_volume(vol, "calibrate_lengthscale")
    _require_ell_reaches_output(cfg, "calibrate_lengthscale", measure is None)
    fitted = fit_lengthscale_from_sections(vol, cfg, seed=seed)
    probe, flanks = _probe_section(vol)
    lo, hi = _bracket(vol, cfg, fitted[0])

    target = float(
        np.mean(
            [
                mean_morans_i(
                    np.asarray(s.counts.todense(), dtype=np.float64),
                    np.asarray(s.coords, dtype=np.float64),
                    cfg,
                    seed=seed,
                )
                for s in flanks
            ]
        )
    )
    objective = _morans_objective(model, vol, cfg, probe, seed=seed) if measure is None else measure
    found = _locate_and_bisect(objective, lo, hi, target, cfg, what="Moran's I")

    ell_z, z_status, z_target, z_achieved = calibrate_ell_z(
        model, vol, cfg, seed=seed, measure=measure_z, ell_xy=found.value, fitted=fitted
    )
    return LengthscaleCalibration(
        ell=(found.value, found.value, ell_z),
        status=found.status,
        i_gen=found.achieved,
        i_target=target,
        grid_ell=found.grid_x,
        grid_i=found.grid_y,
        iterations=found.iterations,
        ell_fitted=fitted,
        ell_z_status=z_status,
        z_target=z_target,
        z_achieved=z_achieved,
        section_ids=tuple(s.section_id for s in flanks),
    )


def _morans_objective(
    model: CTFFlow, vol: TrainingVolume, cfg: Config, probe: Section, *, seed: int
) -> Callable[[float], float]:
    """Return ``ell_xy -> mean I_gen`` of a section generated at ``probe``'s own plane.

    The probe section is **excluded from its own retrieval pool**, so the objective measures
    the field's autocorrelation rather than the index's ability to copy the answer.
    """
    plane = section_plane(probe)

    def objective(ell_xy: float) -> float:
        adata = generate_section(
            model,
            plane,
            vol,
            cfg.replace(ell_xy=float(ell_xy)),
            seed,
            exclude_z={float(probe.z)},
        )
        return mean_morans_i(
            emitted_counts(adata),
            np.asarray(adata.obsm[cfg.coord_key], dtype=np.float64),
            cfg,
            seed=seed,
        )

    return objective


def calibrate_ell_z(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int = 0,
    measure: Callable[[float], float] | None = None,
    ell_xy: float | None = None,
    fitted: tuple[float, float, float] | None = None,
) -> tuple[float, str, float, float]:
    """Calibrate ``ell_z`` against the observed between-section correlation. R1's remedy 2.

    Returns ``(ell_z, status, target, achieved)``.

    The target is the **measured** adjacent-section correlation of the real training sections
    (:func:`observed_z_decay`); the objective is the same statistic on a two-section virtual
    stack generated at the same depths. The bracket runs from
    ``Config.calibration_ell_z_min_factor x median_spacing`` to
    ``Config.calibration_ell_z_max_fitted_multiple x`` the variogram's ``ell_z`` — remedy
    **3**: on a nine-section stack the fit extrapolates past its data and reads 561 um
    against a 200 um truth, so it enters here as the bracket's **upper endpoint, never as a
    value**. When even that endpoint cannot reach the observed correlation the status is
    ``target_unreachable`` and the returned number is a **bound, not a fit**, which is the
    criterion T09 inherited from the R1 decision. Measured on the fixture under a live
    objective: the search terminates at the bracket's top (364.6 um) with the generated
    correlation at 0.8706 against an observed 0.9182, so what the data support is
    ``ell_z >= 364.6 um`` — a *lower* bound on the tissue, even though the number is the
    largest the bracket allowed. Either way :func:`apply_lengthscale` drops it and the
    config's own ``ell_z`` stands.
    """
    require_training_volume(vol, "calibrate_ell_z")
    _require_ell_reaches_output(cfg, "calibrate_ell_z", measure is None)
    # ``fitted`` is threaded in by :func:`calibrate_lengthscale`, which has already paid for
    # the variogram: fitting it twice would cost a second pass and — worse — let the two
    # calibrations disagree about the bracket if the subsample ever changed.
    fitted = fit_lengthscale_from_sections(vol, cfg, seed=seed) if fitted is None else fitted
    lo = float(cfg.calibration_ell_z_min_factor) * float(vol.median_spacing)
    hi = float(cfg.calibration_ell_z_max_fitted_multiple) * float(fitted[2])
    if not hi > lo:
        raise CalibrationError(
            f"calibrate_ell_z: the bracket is empty — [{lo:g}, {hi:g}] um. The fitted ell_z "
            f"({fitted[2]:g} um) is below {cfg.calibration_ell_z_min_factor} x the median "
            f"spacing ({vol.median_spacing:g} um), so the stack cannot resolve it at all."
        )
    lags, correlations = observed_z_decay(vol, cfg)
    target = _adjacent_correlation(lags, correlations, float(vol.median_spacing))
    objective = (
        _z_decay_objective(model, vol, cfg, seed=seed, ell_xy=ell_xy)
        if measure is None
        else measure
    )
    found = _locate_and_bisect(objective, lo, hi, target, cfg, what="between-section correlation")
    return found.value, found.status, target, found.achieved


def _z_decay_objective(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int,
    ell_xy: float | None,
) -> Callable[[float], float]:
    """Return ``ell_z -> adjacent-section correlation`` of a generated two-section stack.

    Generated at two adjacent *training* depths with **both** excluded from retrieval, so the
    correlation between them comes from the field and not from a shared donor pool — the
    latter would be correlated whatever ``ell_z`` was, and the objective would be flat.
    """
    from spatialcpav25_gen.infer.generate import plane_at_z

    middle = vol.n_sections // 2
    depths = [float(vol.z_values[middle - 1]), float(vol.z_values[middle])]
    exclude = set(depths)

    def objective(ell_z: float) -> float:
        gen_cfg = cfg.replace(ell_z=float(ell_z))
        if ell_xy is not None:
            gen_cfg = gen_cfg.replace(ell_xy=float(ell_xy))
        means = []
        box = np.asarray(vol.bbox, dtype=np.float64)[:, :2]
        for index, z in enumerate(depths):
            adata = generate_section(
                model,
                plane_at_z(vol, z, gen_cfg),
                vol,
                gen_cfg,
                seed + index,
                grf_seed=int(model.grf.seed),
                exclude_z=exclude,
            )
            means.append(
                section_grid_means(
                    emitted_counts(adata),
                    np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2],
                    box,
                    gen_cfg,
                )
            )
        shared = means[0][1] & means[1][1]
        if int(shared.sum()) < 3:
            raise CalibrationError(
                "calibrate_ell_z: the two generated sections share fewer than three filled "
                "grid cells, so their correlation is undefined. Lower "
                "Config.variogram_z_min_cells_per_cell or generate more cells."
            )
        a = means[0][0][shared].ravel()
        b = means[1][0][shared].ravel()
        if a.std() == 0.0 or b.std() == 0.0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    return objective


# --------------------------------------------------------------------------------------
# detection and dispersion
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionCalibration:
    """Per-gene corrections on the ``pi`` logit **and** on ``log theta`` (T09 §2).

    ``design/v23_design.md`` §3.5 asks for both and ``specs/`` had only ``pi`` (SPEC_QUESTIONS
    D-table): the two are not substitutes. ``pi`` moves the **zeros** — the detection rate —
    and ``theta`` moves the **spread of the non-zeros**, which is what T06's
    ``test_mean_variance_relation`` measures. Correcting only the first makes a section with
    the right sparsity and the wrong dispersion.

    The correction is **affine** on each scale — ``logit(pi') = scale x logit(pi) + shift``,
    ``log theta' = scale x log theta + shift`` — with the **slope fixed at 1** by
    :func:`calibrate_detection` and the per-gene intercept **solved exactly**: the shift is
    the one that makes the fold's predicted per-gene detection rate (or variance) equal the
    real section's, found by bisection on a monotone function. Two reasons the slope is not
    fitted, and both are about identifiability rather than cost. There is **one** target
    statistic per gene, so a per-gene slope is unidentified; and a slope shared across genes
    is applied to *per-cell* logits, where it rescales the within-gene spread that the ZINB
    already models, against no target that constrains it. The slope stays on the object
    because it is what ``apply`` computes and because an ablation may set it.

    The exact per-gene shift is then shrunk by ``n / (n + Config.calibration_detection_ridge)``,
    ``n`` being the number of cells the gene was actually detected in: a gene seen in three
    cells keeps three thirteenths of a correction fitted to three cells.

    Attributes
    ----------
    pi_scale, pi_shift
        Global slope and ``(G,)`` per-gene intercept of ``logit(pi') = scale logit(pi) +
        shift``.
    theta_scale, theta_shift
        The same for ``log theta' = scale log theta + shift``.
    section_ids
        The training sections the correction was fitted on. ``specs/09`` §2 requires the
        object to record this, because a calibration is only leakage-free relative to the
        sections it saw.
    detection_gen, detection_real
        ``(G,)`` per-gene detection rates before the correction, generated and real. The
        report's evidence that the correction was needed.
    """

    pi_scale: float
    pi_shift: FloatArray
    theta_scale: float
    theta_shift: FloatArray
    section_ids: tuple[str, ...]
    detection_gen: FloatArray
    detection_real: FloatArray

    @property
    def n_genes(self) -> int:
        """G: genes the correction covers."""
        return int(self.pi_shift.shape[0])

    def apply(
        self, mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply both corrections to a decoder output. Three ``(N, G)`` -> three ``(N, G)``.

        ``mu`` is returned untouched: the mean is what the flow and the size factor are for,
        and moving it here would make the calibration a second, unsupervised expression head.
        The corrections are clamped to ``Config.calibration_logit_max`` in magnitude, which is
        the whole usable range of a probability the decoder's own guards do not already floor.
        """
        if pi.shape[1] != self.n_genes:
            raise CalibrationError(
                f"DetectionCalibration.apply: fitted on {self.n_genes} genes, applied to "
                f"{pi.shape[1]}. The correction is per gene and the panels must match."
            )
        limit = float(cfg.calibration_logit_max)
        eps = float(cfg.zinb_eps)
        pi_logit = torch.logit(pi.clamp(eps, 1.0 - eps))
        shift = torch.from_numpy(self.pi_shift.astype(np.float32))[None, :]
        pi_new = torch.sigmoid(torch.clamp(float(self.pi_scale) * pi_logit + shift, -limit, limit))
        log_theta = torch.log(theta.clamp(min=float(cfg.zinb_theta_min)))
        theta_shift = torch.from_numpy(self.theta_shift.astype(np.float32))[None, :]
        theta_new = torch.exp(
            torch.clamp(float(self.theta_scale) * log_theta + theta_shift, -limit, limit)
        ).clamp(float(cfg.zinb_theta_min), float(cfg.zinb_theta_max))
        return mu, theta_new, pi_new


def calibrate_detection(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int = 0,
    n_folds: int | None = None,
) -> DetectionCalibration:
    """Fit the ``pi`` and ``log theta`` corrections on LOSO reconstructions. T09 §2.

    Each fold hides one **training** section, runs the generation path at that section's own
    cell positions with the section excluded from its own retrieval pool, and solves for the
    per-gene shift that makes the decoder's detection rate and variance match the section's.
    Nothing held out is touched: ``vol`` is a ``TrainingVolume`` and the folds are its
    sections.

    ``n_folds`` caps the number of folds for cost; ``None`` uses every training section. The
    per-gene shifts are averaged over folds, so one atypical section cannot set a gene's
    correction on its own, and then shrunk toward zero by how well that gene's rate was
    measured.
    """
    require_training_volume(vol, "calibrate_detection")
    sections = list(vol.sections)[: len(vol.sections) if n_folds is None else max(n_folds, 1)]
    folds = [
        _fold_statistics(model, section, cfg, seed=seed + index)
        for index, section in enumerate(sections)
    ]
    stacked = {key: np.mean([f[key] for f in folds], axis=0) for key in folds[0]}
    limit = float(cfg.calibration_logit_max)
    counted = stacked["n_detected"] / (
        stacked["n_detected"] + float(cfg.calibration_detection_ridge)
    )
    pi_weight = counted * _fold_shrinkage([f["pi_shift"] for f in folds])
    theta_weight = counted * _fold_shrinkage([f["theta_shift"] for f in folds])
    return DetectionCalibration(
        pi_scale=1.0,
        pi_shift=np.asarray(np.clip(stacked["pi_shift"] * pi_weight, -limit, limit), np.float64),
        theta_scale=1.0,
        theta_shift=np.asarray(
            np.clip(stacked["theta_shift"] * theta_weight, -limit, limit), np.float64
        ),
        section_ids=tuple(s.section_id for s in sections),
        detection_gen=stacked["rate_gen"],
        detection_real=stacked["rate_real"],
    )


def _fold_shrinkage(shifts: Sequence[FloatArray]) -> FloatArray:
    """Empirical-Bayes weight on a per-gene shift, from its spread across folds. ``(G,)``.

    ``weight = signal / (signal + se^2)``, with ``se^2`` the squared standard error of the
    fold mean and ``signal`` the between-gene variance of that mean less the average ``se^2``
    — the usual decomposition of an observed spread into a real one and a sampling one.

    It is here because the measurement says it is needed: on the fixture a gene's ``pi`` shift
    varies by **1.92** (logit units) between folds against a between-gene spread of **3.41**,
    so roughly a fifth of what a two-fold mean would apply is fold noise, and the real
    per-gene detection rate itself moves by 0.040 between sections against a model error of
    0.058. A correction fitted without shrinkage transfers *worse* than no correction at all.
    With one fold there is nothing to estimate and the weight is 1 — the detection-count ridge
    is then the only shrinkage, which is what ``Config.calibration_detection_ridge`` is for.
    """
    stack = np.stack([np.asarray(s, dtype=np.float64) for s in shifts], axis=0)
    if stack.shape[0] < 2:
        return np.ones(stack.shape[1], dtype=np.float64)
    se2 = stack.var(axis=0, ddof=1) / float(stack.shape[0])
    signal = max(float(np.var(stack.mean(axis=0)) - float(np.mean(se2))), _EPS)
    return np.asarray(signal / (signal + se2), dtype=np.float64)


def _bisect_shift(
    predict: Callable[[FloatArray], FloatArray],
    target: FloatArray,
    limit: float,
    *,
    decreasing: bool,
    iterations: int,
) -> FloatArray:
    """Per-gene shift solving ``predict(shift) = target``, by vectorised bisection. ``(G,)``.

    ``predict`` is monotone in the shift — decreasing for the ``pi`` correction (more dropout
    means fewer detections) and for the ``theta`` one (more dispersion parameter means less
    variance) — so bisection on ``[-limit, limit]`` is well-posed and needs no derivative.
    Genes whose target lies outside what the bracket can reach come back at the endpoint,
    which the caller clamps anyway.
    """
    lo = np.full(target.shape, -limit, dtype=np.float64)
    hi = np.full(target.shape, limit, dtype=np.float64)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        value = predict(mid)
        above = value > target
        if decreasing:
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        else:
            hi = np.where(above, mid, hi)
            lo = np.where(above, lo, mid)
    return 0.5 * (lo + hi)


def _fold_statistics(
    model: CTFFlow, hidden: Section, cfg: Config, *, seed: int
) -> dict[str, FloatArray]:
    """Per-gene calibration shifts for one LOSO fold. Every value is ``(G,)`` float64.

    Both shifts are solved against **closed forms** rather than against a draw: the detection
    rate of a zero-inflated negative binomial is ``(1 - pi)(1 - P_NB(0))`` and its variance is
    ``keep mu (1 + mu/theta) + pi keep mu^2``, so neither target needs a sample and neither
    carries a sample's noise — the same argument that makes T08's profile term take the
    decoder's mean.

    The two are solved **alternately**, and that is not a refinement: the corrections are not
    orthogonal. ``theta`` enters the detection rate through ``P_NB(0) = (theta / (theta +
    mu))^theta`` — a larger dispersion parameter moves the negative binomial toward Poisson
    and *raises* detection at fixed ``mu`` — so a ``pi`` shift solved at the uncorrected
    ``theta`` is wrong by however much ``theta`` then moves. Measured on the fixture: solved
    independently, the two corrections together made the per-gene detection MAE **worse**,
    0.055 -> 0.230, while each alone matched its own target exactly. Two alternations bring
    the coupled solution within a thousandth of the target.
    """
    fold = _decode_hidden(model, hidden, cfg, seed=seed)
    return solve_detection_shifts(
        fold.mu.numpy().astype(np.float64),
        fold.theta.numpy().astype(np.float64),
        fold.pi.numpy().astype(np.float64),
        fold.real,
        cfg,
    )


def solve_detection_shifts(
    mu: FloatArray,
    theta: FloatArray,
    pi: FloatArray,
    real: FloatArray,
    cfg: Config,
) -> dict[str, FloatArray]:
    """Solve both per-gene shifts from arrays alone, with no model and no decode.

    Split out at T10 so the solver can be exercised on constructed inputs. The model path is
    the only way to get a *real* fold, but "does the correction take the headroom when there
    is headroom" is a question about the solve, not about the decode, and answering it needed
    a fit for no reason — and could not be asked at all on a fixture that happens to have no
    headroom (``tests/test_calibrate.py``, ``specs/09`` 5).

    Parameters
    ----------
    mu, theta, pi
        The decoder's ZINB parameters for one section, each ``(N, G)`` float64.
    real
        That section's real counts, ``(N_real, G)``. Only its per-gene detection rate, mean
        and variance are read, so ``N_real`` need not equal ``N``.
    cfg
        Supplies the bracket (``calibration_logit_max``), the bisection budget and the
        ``zinb_theta_*`` clamps.

    Returns
    -------
    dict
        ``pi_shift`` ``(G,)``, ``theta_shift`` ``(G,)``, ``n_detected`` ``(G,)``,
        ``rate_gen`` ``(G,)``, ``rate_real`` ``(G,)``.
    """
    pi_logit = _logit(np.clip(pi, _EPS, 1.0 - _EPS))
    log_theta = np.log(np.clip(theta, float(cfg.zinb_theta_min), float(cfg.zinb_theta_max)))
    limit = float(cfg.calibration_logit_max)
    iterations = int(cfg.bisection_max_iter) * 8

    rate_real = (real > 0).mean(axis=0)
    mean_real = real.mean(axis=0)
    # Method-of-moments overdispersion of the real section, ``phi = (Var - mean) / mean^2``:
    # the *shape* of its mean-variance relation, free of the mean's own scale. Floored at zero
    # because a gene measured as under-dispersed asks for an infinite ``theta``, and the
    # decoder's guard already caps that.
    dispersion_real = np.maximum(
        (real.var(axis=0) - mean_real) / np.maximum(mean_real**2, _EPS), 0.0
    )
    rate_gen = _zinb_detection(mu, theta, pi).mean(axis=0)

    pi_shift = np.zeros(mu.shape[1], dtype=np.float64)
    theta_shift = np.zeros(mu.shape[1], dtype=np.float64)
    for _ in range(_CALIBRATION_ALTERNATIONS):
        keep_now = 1.0 - _sigmoid(pi_logit + pi_shift[None, :])
        expected_now = keep_now * mu
        mean_now = expected_now.mean(axis=0)
        # The target is the **mean-variance relation**, not the absolute variance: match
        # ``Var = mean + phi mean^2`` at the model's *own* mean, ``phi`` being the real
        # section's method-of-moments overdispersion. Two failures this avoids, both measured
        # on the fixture. Matching the absolute variance asks ``theta`` to repair a wrong
        # **mean** — for the dense genes the model sat at 14 where the section's own mean is an
        # order of magnitude higher, and the solve drove ``theta`` to 0.007, the negative
        # binomial's own zeros swamped those genes, and their detection rate collapsed from
        # 0.68 to 0.06 with the ``pi`` correction pinned at its bracket end trying to undo it.
        # And the model's per-cell means already carry variance across cells, so the *draw*
        # only has to supply what is left after the law of total variance (the subtraction
        # below). ``design/v23_design.md`` §3.5 asks for the relation, and the relation is what
        # T06's ``test_mean_variance_relation`` measures.
        within_target = np.maximum(
            mean_now + dispersion_real * mean_now**2 - expected_now.var(axis=0),
            float(cfg.metric_eps),
        )

        def variance(shift: FloatArray, keep: FloatArray = keep_now) -> FloatArray:
            theta_new = np.exp(log_theta + shift[None, :])
            per_cell = keep * mu * (1.0 + mu / theta_new) + (1.0 - keep) * keep * mu**2
            return np.asarray(per_cell.mean(axis=0), dtype=np.float64)

        theta_shift = _bisect_shift(
            variance, within_target, limit, decreasing=True, iterations=iterations
        )

        def detection(shift: FloatArray, theta_shift: FloatArray = theta_shift) -> FloatArray:
            theta_new = np.exp(log_theta + theta_shift[None, :])
            pi_new = _sigmoid(pi_logit + shift[None, :])
            return np.asarray(_zinb_detection(mu, theta_new, pi_new).mean(axis=0), dtype=np.float64)

        pi_shift = _bisect_shift(
            detection, rate_real, limit, decreasing=True, iterations=iterations
        )
    return {
        "pi_shift": pi_shift,
        "theta_shift": theta_shift,
        "n_detected": rate_real * float(real.shape[0]),
        "rate_gen": rate_gen,
        "rate_real": rate_real,
    }


def _zinb_detection(mu: FloatArray, theta: FloatArray, pi: FloatArray) -> FloatArray:
    """Per-cell, per-gene probability of a non-zero draw. Three ``(N, G)`` -> ``(N, G)``."""
    nb_nonzero = 1.0 - np.power(theta / np.maximum(theta + mu, _EPS), theta)
    return np.asarray((1.0 - pi) * nb_nonzero, dtype=np.float64)


def _sigmoid(x: FloatArray) -> FloatArray:
    """Logistic function, elementwise and overflow-safe."""
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * np.clip(x, -60.0, 60.0))), dtype=np.float64)


def _logit(p: FloatArray) -> FloatArray:
    """Log-odds of a probability array, elementwise."""
    return np.asarray(np.log(p / (1.0 - p)), dtype=np.float64)


@dataclass(frozen=True)
class _FoldDecode:
    """One hidden section, decoded down the generation path. Every tensor is ``(N, G)``."""

    mu: Tensor
    theta: Tensor
    pi: Tensor
    real: FloatArray
    variance: FloatArray
    coords: FloatArray
    neighbours: IntArray
    weights: FloatArray
    cell_type: Tensor
    region: Tensor | None


def _decode_hidden(model: CTFFlow, hidden: Section, cfg: Config, *, seed: int) -> _FoldDecode:
    """Run the generation path at a hidden training section's own cell positions.

    The same construction as T08's
    :func:`~spatialcpav25_gen.train.loso.reconstruct_hidden` — same exclusion, same
    ``apply_dropout=False``, same prior-to-decoder path — but it returns the decoder's
    ``(mu, theta, pi)`` over the **whole panel** rather than the loss-side normalised
    tensors. The calibration needs the parameters themselves (it corrects two of them) and
    it needs every gene, because :meth:`DetectionCalibration.apply` runs on the full panel at
    generation.
    """
    from spatialcpav25_gen.data.schema import to_xyz

    # `section_seed`, not the builtin `hash()`: the latter is salted per process, so the
    # same declared seed drew a different stream in every run (Convention 3).
    gen = np.random.default_rng([int(seed), section_seed(hidden.section_id)])
    rows = _subsample(hidden.n_cells, int(cfg.loso_max_cells), int(seed))
    xyz = to_xyz(hidden)[rows].astype(np.float64)
    points = torch.from_numpy(xyz.astype(np.float32))
    cell_type = torch.from_numpy(np.asarray(hidden.cell_type[rows], dtype=np.int64))
    region = (
        None
        if hidden.region is None
        else torch.from_numpy(np.asarray(hidden.region[rows], dtype=np.int64))
    )
    index = model.data.index
    neighbours, weights = index.query(
        xyz, {float(hidden.z)}, seed=int(gen.integers(0, 2**31 - 1)), apply_dropout=False
    )
    with torch.no_grad():
        tokens, mask = index.neighbour_tokens(xyz, neighbours)
        cond, _ = model.conditioning(points, points, cell_type, region, tokens, mask)
        h0 = model.prior_latent(xyz, seed=int(seed))
        h = model.flow.sample(h0, cond, int(cfg.ode_steps))
        gene_emb = model.embeddings.gene(torch.arange(model.stats.n_genes, dtype=torch.long))
        mu, theta, pi = model.decoder(h, gene_emb, model.size_head.size_factor(h))
        expected, variance = draw_mean_variance(mu, theta, pi, cfg)
    real = np.asarray(hidden.counts[rows].todense(), dtype=np.float64)
    del expected
    return _FoldDecode(
        mu=mu,
        theta=theta,
        pi=pi,
        real=real,
        variance=variance.numpy().astype(np.float64),
        coords=xyz,
        neighbours=neighbours,
        weights=weights,
        cell_type=cell_type,
        region=region,
    )


# --------------------------------------------------------------------------------------
# the anchor weight w(v)
# --------------------------------------------------------------------------------------


class IsotonicRegressor:
    """A 1-D isotonic map, fitted by pool-adjacent-violators. Non-increasing by default.

    ``specs/09`` §1's ``w(v)``: the mixing weight that minimised reconstruction error, as a
    function of the latent variance, constrained to be monotone so that "more uncertain" can
    never mean "anchor harder". PAVA rather than ``sklearn.isotonic`` so the fit is a few
    lines of readable arithmetic with no hidden tie-breaking, and so
    ``test_anchor_weight_monotone`` is testing this code.
    """

    def __init__(
        self, x: npt.NDArray[Any], y: npt.NDArray[Any], *, increasing: bool = False
    ) -> None:
        """Store the fitted knots. ``x`` must be sorted ascending and ``y`` already monotone."""
        self.x = np.asarray(x, dtype=np.float64).reshape(-1)
        self.y = np.asarray(y, dtype=np.float64).reshape(-1)
        self.increasing = bool(increasing)
        if self.x.shape != self.y.shape:
            raise CalibrationError(
                f"IsotonicRegressor: {self.x.shape[0]} knots against {self.y.shape[0]} values"
            )
        if self.x.size == 0:
            raise CalibrationError("IsotonicRegressor: no knots; nothing was fitted")

    @classmethod
    def fit(
        cls,
        x: npt.NDArray[Any],
        y: npt.NDArray[Any],
        *,
        weights: npt.NDArray[Any] | None = None,
        increasing: bool = False,
    ) -> IsotonicRegressor:
        """Fit a monotone step function to ``(x, y)`` by pool-adjacent-violators.

        ``increasing=False`` (the default) fits a **non-increasing** map, which is what the
        anchor weight is: high latent variance means trust the generative sample.
        """
        order = np.argsort(np.asarray(x, dtype=np.float64).reshape(-1), kind="stable")
        xs = np.asarray(x, dtype=np.float64).reshape(-1)[order]
        ys = np.asarray(y, dtype=np.float64).reshape(-1)[order]
        w = (
            np.ones_like(ys)
            if weights is None
            else np.asarray(weights, dtype=np.float64).reshape(-1)[order]
        )
        if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
            raise CalibrationError(
                "IsotonicRegressor.fit: x and y must be finite; a NaN knot would make the "
                "fitted map silently non-monotone (every comparison against NaN is False)"
            )
        sign = 1.0 if increasing else -1.0
        values = sign * ys
        # PAVA: merge adjacent blocks that violate the ordering, keeping weighted means. The
        # blocks are (sum of value x weight, sum of weight, how many points) so a merge is an
        # addition and the block's value stays its weighted mean.
        block_value: list[float] = []
        block_weight: list[float] = []
        block_size: list[int] = []
        for value, weight in zip(values.tolist(), w.tolist(), strict=True):
            block_value.append(value * weight)
            block_weight.append(weight)
            block_size.append(1)
            while len(block_value) > 1 and (
                block_value[-2] / max(block_weight[-2], _EPS)
                > block_value[-1] / max(block_weight[-1], _EPS)
            ):
                # Pop first, then add into what is now the last block. Writing this as
                # ``block_value[-2] += block_value.pop()`` resolves ``-2`` *after* the pop and
                # merges into the wrong block — or raises, at two blocks.
                merged_value = block_value.pop()
                merged_weight = block_weight.pop()
                merged_size = block_size.pop()
                block_value[-1] += merged_value
                block_weight[-1] += merged_weight
                block_size[-1] += merged_size
        fitted: list[float] = []
        for value, weight, size in zip(block_value, block_weight, block_size, strict=True):
            fitted.extend([value / max(weight, _EPS)] * size)
        return cls(xs, sign * np.asarray(fitted, dtype=np.float64), increasing=increasing)

    def predict(self, v: npt.NDArray[Any]) -> FloatArray:
        """Evaluate the map. ``(N,)`` -> ``(N,)`` float64, clamped to the fitted range.

        Linear interpolation between knots and constant extrapolation outside them: the fit
        says nothing about variances it never saw, and extrapolating a monotone step function
        would invent a trend.
        """
        query = np.asarray(v, dtype=np.float64).reshape(-1)
        if self.x.size == 1:
            return np.full(query.shape, float(self.y[0]), dtype=np.float64)
        return np.asarray(
            np.interp(query, self.x, self.y, left=float(self.y[0]), right=float(self.y[-1])),
            dtype=np.float64,
        )

    def __call__(self, v: npt.NDArray[Any]) -> FloatArray:
        """Alias of :meth:`predict`, so the map reads like the function it is."""
        return self.predict(v)

    @property
    def is_monotone(self) -> bool:
        """Whether the fitted values respect the requested direction, to within 1e-9."""
        diffs = np.diff(self.y)
        return bool(np.all(diffs >= -_EPS) if self.increasing else np.all(diffs <= _EPS))


def _profile_correlation(a: FloatArray, b: FloatArray) -> float:
    """Mean per-cell Pearson correlation of two ``(N, G)`` profiles on the ``log1p`` scale.

    The reconstruction score the anchor weight is fitted against. Deliberately **not** an
    element-wise error: the expected element-wise error of a Bernoulli mixture is linear in
    the mixing weight, so its minimiser is always 0 or 1 and there would be nothing for an
    isotonic map to say. A per-cell profile correlation is not separable over genes, so a
    mixture of two imperfect profiles can beat both — which is the whole premise of blending.
    """
    x = np.log1p(np.asarray(a, dtype=np.float64))
    y = np.log1p(np.asarray(b, dtype=np.float64))
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    ok = denominator > _EPS
    if not np.any(ok):
        return 0.0
    return float(((x * y).sum(axis=1)[ok] / denominator[ok]).mean())


def calibrate_anchor_weight(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int = 0,
    n_folds: int | None = None,
) -> IsotonicRegressor:
    """Fit ``w(v)``, the uncertainty-gated anchor weight, on LOSO reconstructions. T09 §1.

    For every fold: reconstruct a hidden **training** section down the generation path, take
    the per-cell latent variance over ``Config.n_uncertainty_samples`` GRF realisations, and
    for each variance bin score every candidate mixing weight on the grid by the profile
    correlation of the resulting blend against the section's real counts. The bin's best
    weight is the point the isotonic map is fitted through, constrained **non-increasing**:
    more uncertainty can never mean anchor harder.

    This replaces the previous version's gap heuristics entirely — there is no ``gap_scale``,
    no ``alpha_tol`` and no ``edit_weight`` here or anywhere else (``specs/09``'s "Do NOT").

    ``n_folds`` caps the folds for cost; ``None`` uses every training section.
    """
    from spatialcpav25_gen.infer.generate import anchor_blend, latent_uncertainty

    require_training_volume(vol, "calibrate_anchor_weight")
    sections = list(vol.sections)[: len(vol.sections) if n_folds is None else max(n_folds, 1)]
    grid = np.linspace(0.0, 1.0, int(cfg.anchor_weight_grid_size))
    variances: list[FloatArray] = []
    scores: list[FloatArray] = []

    for index, hidden in enumerate(sections):
        fold_seed = seed + index
        fold = _decode_hidden(model, hidden, cfg, seed=fold_seed)
        uncertainty = latent_uncertainty(
            model,
            fold.coords,
            fold.cell_type,
            fold.region,
            fold.neighbours,
            cfg,
            seed=fold_seed,
        )
        gen = np.random.default_rng([fold_seed, 0xA5C])
        generated = _draw_counts(model, uncertainty.h, cfg, gen)
        anchored = _anchor_profiles(model, fold, cfg, gen)
        real = fold.real
        per_weight = []
        for w in grid:
            blend = anchor_blend(generated, anchored, np.full(real.shape[0], float(w)), gen).numpy()
            per_weight.append(_per_cell_correlation(blend, real))
        variances.append(uncertainty.variance)
        scores.append(np.stack(per_weight, axis=1))

    variance = np.concatenate(variances)
    score = np.concatenate(scores, axis=0)
    edges = np.quantile(variance, np.linspace(0.0, 1.0, int(cfg.anchor_variance_bins) + 1))
    centres: list[float] = []
    best: list[float] = []
    weights: list[float] = []
    for lo, hi in itertools.pairwise(edges):
        inside = (variance >= lo) & (variance <= hi)
        if int(inside.sum()) == 0:
            continue
        centres.append(float(np.mean(variance[inside])))
        best.append(float(grid[int(np.argmax(score[inside].mean(axis=0)))]))
        weights.append(float(inside.sum()))
    if not centres:  # pragma: no cover - a fold always has cells
        raise CalibrationError("calibrate_anchor_weight: no variance bin held any cell")
    return IsotonicRegressor.fit(
        np.asarray(centres), np.asarray(best), weights=np.asarray(weights), increasing=False
    )


def _draw_counts(model: CTFFlow, h: Tensor, cfg: Config, gen: np.random.Generator) -> Tensor:
    """Draw counts from a latent, uncalibrated. ``(N, d_h)`` -> ``(N, G)`` float32."""
    from spatialcpav25_gen.model.expression import sample_counts, sample_zigamma

    with torch.no_grad():
        gene_emb = model.embeddings.gene(torch.arange(model.stats.n_genes, dtype=torch.long))
        mu, theta, pi = model.decoder(h, gene_emb, model.size_head.size_factor(h))
    draw = sample_zigamma if cfg.decoder == "zigamma" else sample_counts
    return draw(mu, theta, pi, gen)


def _anchor_profiles(
    model: CTFFlow, fold: _FoldDecode, cfg: Config, gen: np.random.Generator
) -> Tensor:
    """Return the retrieval-anchored real profile for a fold's cells. ``(N, G)`` float32.

    The v20 cross-mix over the fold's own retrieval donors — real counts, chosen per gene —
    which is the "retrieval-anchored real profile" ``specs/09`` §1 blends toward.
    """
    from spatialcpav25_gen.model.expression import cross_mix_counts

    valid = fold.neighbours >= 0
    w = np.where(valid, fold.weights, 0.0)
    total = w.sum(axis=1, keepdims=True)
    if not np.all(total > 0):
        raise CalibrationError(
            "calibrate_anchor_weight: a cell of the hidden section has no admissible donor, "
            "so there is no anchored profile to blend toward. Widen the retrieval window "
            "(specs/09 §1's derived window) before calibrating."
        )
    w = w / total
    safe = np.where(valid, fold.neighbours, 0)
    donors = np.asarray(model.data.counts[safe.reshape(-1)].todense(), dtype=np.float64)
    donors = donors.reshape(safe.shape[0], safe.shape[1], -1)
    return cross_mix_counts(donors, w, gen, cfg=cfg)


def _per_cell_correlation(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> FloatArray:
    """Per-cell Pearson correlation of two ``(N, G)`` profiles on ``log1p``. ``(N,)`` float64."""
    x = np.log1p(np.asarray(a, dtype=np.float64))
    y = np.log1p(np.asarray(b, dtype=np.float64))
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = np.sqrt((x**2).sum(axis=1) * (y**2).sum(axis=1))
    out = np.zeros(x.shape[0], dtype=np.float64)
    ok = denominator > _EPS
    out[ok] = (x * y).sum(axis=1)[ok] / denominator[ok]
    return out


# --------------------------------------------------------------------------------------
# the retrieval z window, derived from the geometry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalWindowCalibration:
    """The derived ``retrieval_z_window``, in units of the median section spacing.

    ``specs/09`` §1: ``Config.retrieval_z_window = 3.0`` is the right *unit* and the wrong
    *constant* once the holdout is wide. Measured at T06 on the ``consecutive-3`` holdout,
    the first training section is 200 um from the next admissible one, so after the
    own-section exclusion its cells had **no admissible donor at all** inside 3 x 50 um and
    ``EmptyCandidatePoolWarning`` fired on 100-110 of every 512 cells. The floor is therefore
    the largest gap any query has to its nearest admissible section, in spacings, rounded up.

    Attributes
    ----------
    window
        The value to put in ``Config.retrieval_z_window``. Never below the configured one,
        which stays as the fallback and the ablation handle.
    max_gap_um
        The largest section-to-section gap that forced it.
    per_section_gap_um
        ``(S,)`` each training section's gap to its nearest *other* training section — the
        gap that survives the own-section exclusion, which is what makes this the binding
        quantity rather than the median spacing.
    section_ids
        The training sections it was fitted on; a report shows no held-out section is among
        them.
    """

    window: float
    max_gap_um: float
    per_section_gap_um: FloatArray
    section_ids: tuple[str, ...]


def calibrate_retrieval_window(vol: TrainingVolume, cfg: Config) -> RetrievalWindowCalibration:
    """Derive ``retrieval_z_window`` from the training stack's own geometry (T09 §1/§2).

    Leakage-free by the same construction as the length-scale calibrator: the gaps are
    between **training** sections, measured after the own-section exclusion that GATE 2 rests
    on, so the window admits at least the nearest admissible section for every query the
    training loop or an internal LOSO fold can make.
    """
    require_training_volume(vol, "calibrate_retrieval_window")
    z = vol.z_values.astype(np.float64)
    if z.size < 2:  # pragma: no cover - Volume.__post_init__ refuses a one-section stack
        raise CalibrationError("calibrate_retrieval_window: a stack needs at least 2 sections")
    distance = np.abs(z[:, None] - z[None, :])
    np.fill_diagonal(distance, np.inf)
    gaps = distance.min(axis=1)
    spacing = max(float(vol.median_spacing), 1e-6)
    window = max(float(cfg.retrieval_z_window), float(np.ceil(float(gaps.max()) / spacing)))
    return RetrievalWindowCalibration(
        window=window,
        max_gap_um=float(gaps.max()),
        per_section_gap_um=np.asarray(gaps, dtype=np.float64),
        section_ids=tuple(vol.section_ids),
    )
