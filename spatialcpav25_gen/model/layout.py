"""Layout: how many cells there are, where they sit, and what type each one is.

A generated section is a **marked point process**, not a resampled coordinate list. The
three pieces, in the order they run:

1. a per-type intensity field ``lambda_c(x, y, z)`` in cells per micrometre^3, fitted by
   the inhomogeneous-Poisson-process negative log-likelihood (:mod:`.losses.reconstruction`);
2. positions drawn from ``sum_c lambda_c`` by rejection sampling and then thinned by a
   hard-core / soft-repulsion interaction, so no two cells overlap and local spacing looks
   like tissue;
3. marks drawn from ``lambda_c(p) / sum_c lambda_c(p)`` and smoothed by a few rounds of
   Potts updates on the kNN graph, so types form patches instead of salt-and-pepper noise.

What this is competing with, and where it wins
----------------------------------------------
The competing method optimises a sliced-Wasserstein objective over point coordinates and
assigns types by a ``k = 1`` nearest-neighbour vote. That objective constrains the
*marginal* geometry of the cloud and nothing else: it does not constrain local spacing (so
coincident and clumped points survive, and every kNN-graph metric downstream — Moran's I,
Geary's C, neighbourhood enrichment — is computed on a graph that tissue would never
produce), and a ``k = 1`` vote does not constrain spatial type organisation at all, which
is exactly what ``paper_celltype_localization`` measures. Steps 2 and 3 are the answer to
those two, and both parameter sets are **fitted from training sections**, never hand-set:
:func:`fit_repulsion` and :func:`fit_potts_beta`.

Count from the slab volume, positions on the mid-plane
------------------------------------------------------
``N_expected`` is the intensity integrated over the **slab volume** (T05 §1) — never the
area, and never a flanking section's count. Two things depend on that: T07's thickness
consistency is stated in terms of a slab, and a count that emerges from the intensity is
what makes ``L_thick`` coherent instead of circular.

The *positions*, though, are sampled on the mid-plane rather than spread through the slab.
The spec fixes the count's domain and leaves the positions' open, and the mid-plane is the
coherent choice: a generated section reports in-plane coordinates, every benchmark metric
is computed on an in-plane kNN graph, and the repulsion parameters are fitted to an
in-plane ``g(r)`` — sampling positions through the slab would make the generated in-plane
``g(r)`` a projection of a 3-D process and the fitted parameters would no longer describe
it. Recorded in ``PROGRESS.md`` as a T05 decision.

Conditional on N (SPEC_QUESTIONS B7)
------------------------------------
``N ~ Poisson(N_expected)`` is drawn first and the interaction then thins *proposals* until
``N`` points are placed. That is a Strauss process **conditioned on its count**, not a
Strauss process — the latter would return fewer points than its Poisson envelope, and the
count would then no longer be the intensity integral, which is the property T07 needs. The
proposal budget (``Config.layout_max_proposal_factor``) exists so an inconsistent
(intensity, ``r0``) pair cannot spin forever; exhausting it warns
(:class:`ProposalBudgetWarning`) and the returned layout says so in
``Layout.budget_exhausted``. Under pytest the warning is an error, so no test can quietly
measure a truncated sample.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial import cKDTree
from torch import Tensor, nn

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume, to_xyz
from spatialcpav25_gen.infer.planes import (
    Plane,
    section_plane,
    uniform_plane_points,
    uniform_slab_points,
)
from spatialcpav25_gen.model.field import fourier_encode, fourier_encode_dim

__all__ = [
    "FieldFn",
    "FlankingSection",
    "IntensityFn",
    "IntensityHead",
    "Layout",
    "LayoutError",
    "ProposalBudgetWarning",
    "RepulsionParams",
    "expected_count",
    "fit_intensity_head",
    "fit_potts_beta",
    "fit_repulsion",
    "flanking_from_section",
    "fourier_bands_for_lengthscale",
    "intensity_fn_from_head",
    "mean_cell_density",
    "pair_correlation",
    "potts_beta_grid",
    "potts_smooth",
    "sample_layout",
    "type_purity",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int32]

IntensityFn = Callable[[FloatArray], FloatArray]
"""``(M, 3)`` micrometres -> ``(M, C)`` per-type intensity in cells per micrometre^3.

Deliberately a plain numpy callable rather than a module: the sampler is a numpy algorithm,
the tests drive it with analytic intensities, and T09 wraps the trained head with
:func:`intensity_fn_from_head`.
"""

FieldFn = Callable[[Tensor], Tensor]
"""``(N, 3)`` model-frame micrometres -> ``(N, d_f)`` field features (T04's ``TriplaneField``)."""


class LayoutError(ValueError):
    """Raised when a layout cannot be built from the arguments given."""


class ProposalBudgetWarning(UserWarning):
    """The Strauss sampler ran out of proposals before placing ``N`` points.

    ``r0`` is too large for the requested density: the intensity and the repulsion describe
    incompatible tissue. The layout is returned truncated with
    ``Layout.budget_exhausted = True`` rather than silently padded, and
    ``pyproject.toml`` turns this warning into an error under pytest (SPEC_QUESTIONS B7) so
    no acceptance test can measure a truncated sample and call it a pass.
    """


@dataclass(frozen=True)
class RepulsionParams:
    """The fitted interaction of the point process.

    Attributes
    ----------
    r0
        Hard-core radius, micrometres: no two cells are closer than this.
    R
        Interaction range, micrometres: the lag at which the empirical ``g(r)`` first
        reaches 1.
    gamma
        Strauss interaction parameter in ``(0, 1]``. ``1`` is no soft repulsion at all.
    r0_percentile
        Which percentile of the pooled nearest-neighbour distances produced ``r0``. Carried
        on the object because it changes a published point-process number and therefore has
        to be reported (SPEC_QUESTIONS B6).
    density
        Mean in-plane cell density of the sections it was fitted on, cells per
        micrometre^2. Recorded so a run can say what tissue the interaction describes.
    """

    r0: float
    R: float
    gamma: float
    r0_percentile: float
    density: float

    def log_acceptance(self, distances: FloatArray) -> float:
        """``-sum_j phi(d_j)`` for one proposal against the accepted points.

        ``phi(d) = +inf`` for ``d < r0``, ``-log(gamma)`` for ``r0 <= d < R`` and ``0``
        beyond, so this is ``-inf`` inside the hard core and ``n_close * log(gamma)``
        otherwise. Returns a float, not an array.
        """
        d = np.asarray(distances, dtype=np.float64)
        if d.size and float(d.min()) < self.r0:
            return -math.inf
        n_close = int(np.count_nonzero((d >= self.r0) & (d < self.R)))
        return n_close * math.log(self.gamma)


@dataclass(frozen=True)
class FlankingSection:
    """A real section's coordinates in a target plane's frame.

    The evidence ``layout_mode="hybrid"`` polishes toward and ``layout_mode="resample"``
    copies outright. Built by :func:`flanking_from_section`.

    Attributes
    ----------
    coords_uv
        ``(M, 2)`` float64 in-plane coordinates in the target plane's ``(u, v)`` frame.
    cell_type
        ``(M,)`` int32 type codes.
    z
        The source section's depth, micrometres: which flank this is.
    section_id
        The source section's identifier, so a resampled layout can say what it reused.
    """

    coords_uv: FloatArray
    cell_type: IntArray
    z: float
    section_id: str


@dataclass(frozen=True)
class Layout:
    """A generated section's cells: where they are and what they are.

    Attributes
    ----------
    coords_uv
        ``(N, 2)`` float32 in-plane micrometres in the plane's frame.
    coords_xyz
        ``(N, 3)`` float32 physical micrometres ``(x, y, z)``.
    cell_type
        ``(N,)`` int32 codes into ``Volume.celltype_names``.
    n_expected
        The intensity integral over the slab: what ``N`` was drawn against.
    n_proposals
        Proposals the sampler consumed, ``0`` for ``resample``.
    budget_exhausted
        ``True`` when the proposal budget ran out before ``N`` points were placed.
    mode
        The ``Config.layout_mode`` that produced it.
    seed
        The seed it was produced from.
    plane
        The plane it was sampled on.
    repulsion
        The interaction used, or ``None`` under ablation A4 / ``resample``.
    """

    coords_uv: npt.NDArray[np.float32]
    coords_xyz: npt.NDArray[np.float32]
    cell_type: IntArray
    n_expected: float
    n_proposals: int
    budget_exhausted: bool
    mode: str
    seed: int
    plane: Plane
    repulsion: RepulsionParams | None

    @property
    def n_cells(self) -> int:
        """Number of cells in the layout."""
        return int(self.coords_uv.shape[0])


# --------------------------------------------------------------------------------------
# point-pattern statistics
# --------------------------------------------------------------------------------------


def type_purity(coords_uv: npt.ArrayLike, cell_type: npt.ArrayLike, cfg: Config) -> FloatArray:
    """Per-cell neighbourhood type purity. ``(N, 2)``, ``(N,)`` -> ``(N,)`` float64 in [0, 1].

    The fraction of a cell's ``cfg.potts_knn_k`` in-plane nearest neighbours that share its
    type. This is the statistic :func:`fit_potts_beta` matches: it is what "cell types are
    spatially organised in patches" means as a number, and it is the same neighbourhood the
    Potts smoothing acts on, so fitting against it is fitting against the thing being
    changed.
    """
    coords = np.asarray(coords_uv, dtype=np.float64)
    types = np.asarray(cell_type)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise LayoutError(f"type_purity: coords_uv must be (N, 2), got {coords.shape}")
    if types.shape != (coords.shape[0],):
        raise LayoutError(f"type_purity: cell_type must be ({coords.shape[0]},), got {types.shape}")
    k = min(int(cfg.potts_knn_k), coords.shape[0] - 1)
    if k < 1:
        raise LayoutError("type_purity: needs at least 2 cells")
    idx = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]
    if cfg.debug_shapes:
        assert idx.shape == (coords.shape[0], k), idx.shape
    return np.asarray((types[idx] == types[:, None]).mean(axis=1), dtype=np.float64)


def _pcf_pair_counts(
    coords: FloatArray, edges: FloatArray, window: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Border-corrected pair counts per distance bin, and their CSR expectation.

    ``coords`` ``(N, 2)``, ``edges`` ``(B + 1,)``, ``window`` ``(2, 2)`` min/max.
    Returns ``(counts, expected)``, both ``(B,)``, so that several sections can be pooled by
    summing the two arrays before dividing — which is the correct pooling for a ratio
    estimator, and not the same thing as averaging per-section ``g``.

    Only points at least ``edges[-1]`` from the window boundary serve as centres: their
    annuli lie entirely inside the observed window, so no edge correction is needed and none
    is guessed at.
    """
    lo = window[0]
    hi = window[1]
    r_max = float(edges[-1])
    area = float(np.prod(hi - lo))
    density = coords.shape[0] / area
    interior = coords[
        np.all((coords >= lo[None, :] + r_max) & (coords <= hi[None, :] - r_max), axis=1)
    ]
    if interior.shape[0] == 0:
        raise LayoutError(
            f"pair_correlation: no point lies more than r_max={r_max:g} um from the window "
            f"boundary ({(hi - lo).tolist()} um), so g(r) has no unbiased centre. Lower "
            "Config.pcf_max_r_factor or use a wider window."
        )
    cumulative = np.asarray(
        cKDTree(interior).count_neighbors(cKDTree(coords), edges), dtype=np.float64
    )
    counts = np.diff(cumulative)
    annulus = math.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    expected = interior.shape[0] * density * annulus
    return counts, expected


def pair_correlation(
    coords_uv: npt.ArrayLike,
    cfg: Config,
    *,
    r_max: float,
    window: npt.ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Empirical pair-correlation function ``g(r)``.

    Parameters
    ----------
    coords_uv
        ``(N, 2)`` in-plane coordinates, micrometres.
    cfg
        Supplies ``pcf_n_bins`` and ``debug_shapes``.
    r_max
        Largest lag, micrometres. The caller sets it (usually
        ``cfg.pcf_max_r_factor * median_nn_dist``) because ``g(r)`` from two different point
        sets is only comparable on identical bins.
    window
        ``(2, 2)`` float64 min/max of the observation window. ``None`` uses the bounding box
        of ``coords_uv``, which understates the true window by about half a
        nearest-neighbour distance — acceptable when the same convention is used on both
        sides of a comparison, which is why it is a parameter.

    Returns
    -------
    tuple
        ``(r_centres, g)``, both ``(cfg.pcf_n_bins,)`` float64. ``g(r) = 1`` is complete
        spatial randomness; ``g(r) = 0`` inside the hard core; ``g > 1`` is clustering.
    """
    coords = np.asarray(coords_uv, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise LayoutError(f"pair_correlation: coords_uv must be (N, 2), got {coords.shape}")
    if not r_max > 0:
        raise LayoutError(f"pair_correlation: r_max must be > 0, got {r_max!r}")
    box = (
        np.stack([coords.min(axis=0), coords.max(axis=0)], axis=0)
        if window is None
        else np.asarray(window, dtype=np.float64)
    )
    if box.shape != (2, 2):
        raise LayoutError(f"pair_correlation: window must be (2, 2) min/max, got {box.shape}")
    edges = np.linspace(0.0, float(r_max), int(cfg.pcf_n_bins) + 1)
    counts, expected = _pcf_pair_counts(coords, edges, box)
    centres = 0.5 * (edges[:-1] + edges[1:])
    g = np.divide(counts, expected, out=np.zeros_like(counts), where=expected > 0)
    if cfg.debug_shapes:
        assert centres.shape == (int(cfg.pcf_n_bins),), centres.shape
        assert g.shape == centres.shape, g.shape
    return centres, g


def mean_cell_density(sections: Sequence[Section]) -> float:
    """Mean cell density over sections, in cells per micrometre^3.

    Pooled as ``sum(cells) / sum(slab volumes)``, each section's slab volume being its
    observed in-plane bounding box times its thickness. The bounding box understates the
    tissue by about half a nearest-neighbour distance per side (the outermost cell is not
    the edge of the section), so this reads a few percent high; it is used as the
    intensity head's initialisation and as the density of the repulsion fit's simulation
    window, both of which are shape-free uses.
    """
    if not sections:
        raise LayoutError("mean_cell_density: needs at least one section")
    cells = sum(s.n_cells for s in sections)
    volume = sum(section_plane(s).slab_volume for s in sections)
    if volume <= 0:
        raise LayoutError("mean_cell_density: total slab volume is zero")
    return float(cells / volume)


def expected_count(lambda_mc: npt.ArrayLike, slab_volume: float) -> float:
    """Monte-Carlo estimate of ``integral over the slab of sum_c lambda_c``.

    ``lambda_mc`` is ``(M, C)`` intensity evaluated at ``M`` uniform points in the slab;
    the estimate is ``slab_volume * mean_m sum_c lambda_mc[m, c]``. Returns cells.
    """
    values = np.asarray(lambda_mc, dtype=np.float64)
    if values.ndim != 2:
        raise LayoutError(f"expected_count: lambda_mc must be (M, C), got {values.shape}")
    return float(values.sum(axis=1).mean() * float(slab_volume))


# --------------------------------------------------------------------------------------
# fitting the interaction, from training sections only
# --------------------------------------------------------------------------------------


def _require_training_volume(vol: object, caller: str) -> TrainingVolume:
    """Refuse anything but a :class:`TrainingVolume` (the leakage discipline, by type)."""
    if not isinstance(vol, TrainingVolume):
        raise TypeError(
            f"{caller}: expected a TrainingVolume, got {type(vol).__name__}. The repulsion "
            "and the Potts coupling are fitted on training sections only; pass "
            "split_holdout(...)[0]."
        )
    return vol


def fit_repulsion(vol: TrainingVolume, cfg: Config, *, seed: int) -> RepulsionParams:
    """Fit ``(r0, R, gamma)`` to the training sections' pair-correlation function.

    ``r0`` is the ``cfg.repulsion_r0_percentile``-th percentile of the nearest-neighbour
    distances pooled over training sections. The default is the **1st** percentile, not the
    spec's 5th: at the 5th, 5% of *real* pairs are by construction closer than ``r0``, so a
    generated layout that respects ``r0`` is strictly more regular than the tissue it
    imitates, and ``test_hardcore_respected`` and ``test_pcf_matches_real`` then pull in
    opposite directions (SPEC_QUESTIONS B6). ``Config.repulsion_r0_percentile = 5.0``
    remains selectable and is recorded on the returned object.

    ``R`` is the lag at which the pooled empirical ``g(r)`` first reaches 1. ``gamma`` is
    fitted by a 1-D grid search: each candidate simulates a homogeneous realisation
    **through the production sampler** and is scored by the L2 distance between its ``g(r)``
    and the empirical one over ``r`` in ``[r0, R]``.

    Deterministic given ``seed``.
    """
    training = _require_training_volume(vol, "fit_repulsion")
    sections = training.sections

    nn_distances = np.concatenate(
        [
            cKDTree(np.asarray(s.coords, dtype=np.float64)).query(
                np.asarray(s.coords, dtype=np.float64), k=2
            )[0][:, 1]
            for s in sections
        ]
    )
    r0 = float(np.percentile(nn_distances, float(cfg.repulsion_r0_percentile)))

    r_max = float(cfg.pcf_max_r_factor) * training.median_nn_dist
    edges = np.linspace(0.0, r_max, int(cfg.pcf_n_bins) + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts = np.zeros(int(cfg.pcf_n_bins), dtype=np.float64)
    expected = np.zeros_like(counts)
    density_2d = 0.0
    for s in sections:
        coords = np.asarray(s.coords, dtype=np.float64)
        window = np.stack([coords.min(axis=0), coords.max(axis=0)], axis=0)
        c, e = _pcf_pair_counts(coords, edges, window)
        counts += c
        expected += e
        density_2d += s.n_cells / float(np.prod(window[1] - window[0]))
    density_2d /= len(sections)
    g_real = np.divide(counts, expected, out=np.zeros_like(counts), where=expected > 0)

    reached = np.nonzero(g_real >= 1.0)[0]
    if reached.size == 0:
        raise LayoutError(
            "fit_repulsion: the empirical g(r) never reaches 1 within "
            f"Config.pcf_max_r_factor={cfg.pcf_max_r_factor} nearest-neighbour distances "
            f"(max g = {g_real.max():.3f}), so the interaction range R is undefined. Either "
            "the sections are clustered at every observed scale or the range is too short."
        )
    radius = float(centres[int(reached[0])])
    if radius <= r0:
        raise LayoutError(
            f"fit_repulsion: fitted R={radius:g} um is not greater than r0={r0:g} um, so "
            "there is no soft-repulsion range to fit gamma over. Raise Config.pcf_n_bins "
            "so the rise of g(r) is resolved."
        )

    mask = (centres >= r0) & (centres <= radius)
    if not mask.any():
        raise LayoutError(
            f"fit_repulsion: no g(r) bin falls in [r0, R] = [{r0:g}, {radius:g}] um; raise "
            f"Config.pcf_n_bins (currently {cfg.pcf_n_bins})."
        )

    gammas = np.linspace(float(cfg.repulsion_gamma_min), 1.0, int(cfg.repulsion_gamma_grid_size))
    side = math.sqrt(float(cfg.repulsion_fit_cells) / density_2d)
    losses = np.empty(gammas.shape[0], dtype=np.float64)
    for i, gamma in enumerate(gammas):
        candidate = RepulsionParams(
            r0=r0,
            R=radius,
            gamma=float(gamma),
            r0_percentile=float(cfg.repulsion_r0_percentile),
            density=density_2d,
        )
        g_sim = _simulate_g(candidate, density_2d, side, edges, cfg, seed=seed + i, r_max=r_max)
        losses[i] = float(np.sum((g_sim[mask] - g_real[mask]) ** 2))
    best = int(np.argmin(losses))
    return RepulsionParams(
        r0=r0,
        R=radius,
        gamma=float(gammas[best]),
        r0_percentile=float(cfg.repulsion_r0_percentile),
        density=density_2d,
    )


def _simulate_g(
    repulsion: RepulsionParams,
    density_2d: float,
    side: float,
    edges: FloatArray,
    cfg: Config,
    *,
    seed: int,
    r_max: float,
) -> FloatArray:
    """``g(r)`` of one homogeneous realisation at ``density_2d``, on the given bins.

    Runs the production sampler (:func:`sample_layout`) on a square window of side ``side``
    with a single, constant cell type, so the fit and the thing it is fitting are the same
    code. Returns ``(len(edges) - 1,)`` float64.
    """
    thickness = 1.0  # a unit slab: the 3-D intensity below is chosen against it.
    plane = _square_plane(side, thickness)
    lam = density_2d / thickness

    def intensity(points: FloatArray) -> FloatArray:
        return np.full((points.shape[0], 1), lam, dtype=np.float64)

    fit_cfg = cfg.replace(layout_mode="field", repulsion=True, potts_iters=1)
    layout = sample_layout(intensity, plane, fit_cfg, seed, repulsion=repulsion)
    coords = np.asarray(layout.coords_uv, dtype=np.float64)
    window = np.stack(
        [-np.asarray(plane.half_extent, dtype=np.float64), np.asarray(plane.half_extent)], axis=0
    )
    _, g = pair_correlation(coords, cfg, r_max=r_max, window=window)
    return g


def _square_plane(side: float, thickness: float) -> Plane:
    """Return a ``side x side`` window on the ``z = 0`` plane, for the repulsion fit."""
    from spatialcpav25_gen.infer.planes import plane_from_normal

    return plane_from_normal(
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.zeros(3),
        half_extent=np.full(2, 0.5 * side),
        thickness=thickness,
    )


def fit_potts_beta(
    vol: TrainingVolume, intensity_fn: IntensityFn, cfg: Config, *, seed: int
) -> float:
    """Fit the Potts coupling to the training sections' neighbourhood-purity distribution.

    Independent marks are too noisy — real cell types sit in patches, and
    ``paper_celltype_localization`` measures exactly that — but over-smoothing erases the
    rare interspersed types the same benchmark rewards. So ``beta`` is fitted, never
    hand-set (T05 "Do NOT").

    The fit is a 1-D grid search over ``[0, cfg.potts_beta_max]``. For each training
    section, marks are drawn once (seeded) from ``intensity_fn`` **at that section's real
    cell positions** — the sampler's own initial state — and every candidate ``beta``
    smooths the *same* marks, so the comparison is paired and the objective is a clean
    function of ``beta``. The chosen value is the largest that satisfies **both** of T05's
    statements about the coupling:

    1. its mean purity does not **exceed** the training sections' mean purity — matching
       from below, because the failure mode that costs the benchmark is over-smoothing;
    2. every rare type (expected prevalence below ``cfg.potts_rare_prevalence``) keeps at
       least ``cfg.potts_rare_retention`` of its expected count **in every section**. T05
       states this as an acceptance criterion; making it a constraint on the fit is what
       turns "do not smooth types so hard that rare types vanish" into something the code
       guarantees. On the synthetic fixture with a 2% type injected the constraint binds:
       purity alone would choose 0.278, the rare-type floor takes it to 0.075.

    If nothing satisfies both, the answer is ``0`` — no smoothing — rather than the least
    bad violation.

    Why ``intensity_fn`` is an argument (T05 writes ``fit_potts_beta(vol)``)
    ---------------------------------------------------------------------
    ``beta`` is not a property of the tissue alone; it is the coupling that closes the gap
    between the purity of a draw from ``lambda_c`` and the purity of the tissue, and that
    gap depends on how organised ``lambda_c`` already is. Fitting from a structureless
    initial state — an i.i.d. draw from the global composition — asks the coupling to do all
    the organising work and over-estimates it by a long way, which is exactly the
    over-smoothing T05's "Do NOT" forbids and what erases a 2% cell type. So the intensity
    the marks will actually be drawn from is required rather than assumed. Recorded in
    ``PROGRESS.md`` as a deviation from the spec's signature.

    Deterministic given ``seed``. Returns the fitted ``beta``.
    """
    training = _require_training_volume(vol, "fit_potts_beta")
    sections = training.sections
    target = float(
        np.mean(
            np.concatenate(
                [type_purity(s.coords, s.cell_type, cfg) for s in sections],
            )
        )
    )

    gen = np.random.default_rng(seed)
    draws: list[tuple[FloatArray, IntArray, FloatArray, FloatArray]] = []
    for s in sections:
        coords = np.asarray(s.coords, dtype=np.float64)
        lam = _check_intensity(
            intensity_fn(to_xyz(s).astype(np.float64)), s.n_cells, f"section {s.section_id!r}"
        )
        probabilities = lam / lam.sum(axis=1, keepdims=True)
        marks = _categorical(probabilities, gen)
        draws.append((coords, marks, np.log(lam + float(cfg.layout_intensity_eps)), probabilities))

    betas = potts_beta_grid(cfg)
    purities = np.empty(betas.shape[0], dtype=np.float64)
    keeps_rare = np.ones(betas.shape[0], dtype=bool)
    for i, beta in enumerate(betas):
        values: list[FloatArray] = []
        retained: list[float] = []
        for coords, marks, unary, probabilities in draws:
            smoothed = potts_smooth(
                coords,
                marks,
                unary,
                cfg,
                generator=np.random.default_rng(seed + i),
                beta=float(beta),
            )
            values.append(type_purity(coords, smoothed, cfg))
            retained.extend(_rare_retention(smoothed, probabilities, cfg))
        purities[i] = float(np.mean(np.concatenate(values)))
        keeps_rare[i] = not retained or min(retained) >= float(cfg.potts_rare_retention)

    admissible = np.nonzero((purities <= target) & keeps_rare)[0]
    if admissible.size == 0:
        # Either a draw from the intensity is already purer than the tissue — the intensity
        # carries all the organisation there is and any coupling would over-smooth — or the
        # rare-type floor rules out every coupling. No smoothing is then the honest answer.
        return 0.0
    return float(betas[int(admissible[np.argmax(purities[admissible])])])


def potts_beta_grid(cfg: Config) -> FloatArray:
    """Return the candidate couplings: ``0``, then geometric steps to ``potts_beta_max``.

    ``(cfg.potts_beta_grid_size,)`` float64 from ``0`` to ``cfg.potts_beta_max``. Geometric
    rather than linear because ``beta`` enters an exponent — see ``Config.potts_beta_min``.
    """
    return np.concatenate(
        [
            [0.0],
            np.geomspace(
                float(cfg.potts_beta_min),
                float(cfg.potts_beta_max),
                int(cfg.potts_beta_grid_size) - 1,
            ),
        ]
    )


def _rare_retention(smoothed: IntArray, probabilities: FloatArray, cfg: Config) -> list[float]:
    """Share of each rare type's *expected* count that survived, one entry per rare type.

    A type is rare when its expected prevalence — ``mean_i P(c_i = c)`` under the
    intensity, not a realised count — is below ``cfg.potts_rare_prevalence``. Using the
    expectation as the denominator keeps the criterion from inheriting the noise of the
    initial draw, which on a 2% type over 1500 cells is ~20%.
    """
    expected = probabilities.sum(axis=0)
    total = float(expected.sum())
    out: list[float] = []
    for c in range(probabilities.shape[1]):
        if total <= 0.0 or expected[c] / total >= float(cfg.potts_rare_prevalence):
            continue
        if expected[c] <= 0.0:
            continue
        out.append(float(np.count_nonzero(smoothed == c)) / float(expected[c]))
    return out


# --------------------------------------------------------------------------------------
# marks
# --------------------------------------------------------------------------------------


def _categorical(probabilities: FloatArray, generator: np.random.Generator) -> IntArray:
    """One draw per row of ``(N, C)`` probabilities. Returns ``(N,)`` int32.

    Vectorised inverse-CDF: exactly one uniform per row, so the number of draws is a
    function of ``N`` alone and the sampler stays reproducible (Convention 3).
    """
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative /= cumulative[:, -1:]
    u = generator.random(probabilities.shape[0])
    return np.asarray(
        (u[:, None] > cumulative).sum(axis=1).clip(0, probabilities.shape[1] - 1), dtype=np.int32
    )


def potts_smooth(
    coords_uv: npt.ArrayLike,
    cell_type: npt.ArrayLike,
    log_intensity: npt.ArrayLike,
    cfg: Config,
    *,
    generator: np.random.Generator | None = None,
    beta: float | None = None,
    iters: int | None = None,
) -> IntArray:
    """Potts smoothing of the marks on the kNN graph. ``(N,)`` int32 out.

    Parameters
    ----------
    coords_uv
        ``(N, 2)`` in-plane coordinates.
    cell_type
        ``(N,)`` initial marks.
    log_intensity
        ``(N, C)`` unary term ``log lambda_c(p_i)``.
    cfg
        Supplies ``potts_update``, ``potts_knn_k``, ``potts_beta``, ``potts_iters`` and
        ``debug_shapes``.
    generator
        Required under ``potts_update="gibbs"`` (Convention 3: the update is stochastic).
        Unused, and may be ``None``, under ``"icm"``.
    beta, iters
        Override ``cfg.potts_beta`` / ``cfg.potts_iters``. :func:`fit_potts_beta` uses the
        override to sweep; the pipeline passes the fitted value through the config.

    Notes
    -----
    The energy is T05's: ``-log lambda_c(p_i) - beta * #{neighbours with type c}`` on the
    ``cfg.potts_knn_k``-nearest in-plane graph.

    ``potts_update="gibbs"`` **samples** each label from that energy's conditional;
    ``"icm"`` takes its ``argmax``, which is what T05 writes and which is kept as the
    negative control — see ``Config.potts_update`` for the measurement that decided the
    default, and ``test_icm_erases_rare_types`` for the assertion.

    Updates are **synchronous** (every cell is updated from the previous round's labels),
    which makes the result a function of the input rather than of an iteration order. Under
    ``"icm"`` the loop stops early when a round changes nothing; under ``"gibbs"`` it does
    not, because a fixed point of a sampler is not a stopping condition.
    """
    coords = np.asarray(coords_uv, dtype=np.float64)
    types = np.asarray(cell_type, dtype=np.int32).copy()
    unary = np.asarray(log_intensity, dtype=np.float64)
    n = coords.shape[0]
    if unary.ndim != 2 or unary.shape[0] != n:
        raise LayoutError(f"potts_smooth: log_intensity must be ({n}, C), got {unary.shape}")
    if types.shape != (n,):
        raise LayoutError(f"potts_smooth: cell_type must be ({n},), got {types.shape}")
    n_types = int(unary.shape[1])
    coupling = float(cfg.potts_beta if beta is None else beta)
    rounds = int(cfg.potts_iters if iters is None else iters)
    gibbs = cfg.potts_update == "gibbs"
    if gibbs and generator is None:
        raise LayoutError(
            "potts_smooth: Config.potts_update='gibbs' samples the Potts conditional and "
            "needs an explicit generator (Convention 3). Pass generator=..., or select "
            "Config.potts_update='icm' for the deterministic mode-seeking variant."
        )
    k = min(int(cfg.potts_knn_k), n - 1)
    if k < 1 or rounds < 1 or (coupling == 0.0 and not gibbs):
        # Nothing to smooth with: under ICM a zero coupling still moves the labels (to the
        # mode of the unary term), which is not what "no smoothing" means, so it is the one
        # case that returns the marks untouched.
        return np.asarray(types, dtype=np.int32)
    idx = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]
    eye = np.eye(n_types, dtype=np.float64)
    for _ in range(rounds):
        score = unary + coupling * eye[types[idx]].sum(axis=1)
        if gibbs:
            assert generator is not None  # checked above; narrows the type
            weights = np.exp(score - score.max(axis=1, keepdims=True))
            updated = _categorical(weights / weights.sum(axis=1, keepdims=True), generator)
        else:
            updated = np.asarray(np.argmax(score, axis=1), dtype=np.int32)
            if np.array_equal(updated, types):
                break
        types = updated
    if cfg.debug_shapes:
        assert types.shape == (n,), types.shape
    return types


# --------------------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------------------


def flanking_from_section(section: Section, plane: Plane) -> FlankingSection:
    """Project a real section into ``plane``'s in-plane frame.

    Used to build the evidence for ``layout_mode`` ``"hybrid"`` and ``"resample"``, both of
    which are defined against the flanking real sections. The projection is orthogonal, so
    for the parallel planes those two modes are meant for it is exactly the section's own
    ``(x, y)`` in the plane's coordinates.
    """
    return FlankingSection(
        coords_uv=plane.to_uv(to_xyz(section)),
        cell_type=np.asarray(section.cell_type, dtype=np.int32),
        z=float(section.z),
        section_id=section.section_id,
    )


def sample_layout(
    intensity_fn: IntensityFn,
    plane: Plane,
    cfg: Config,
    seed: int,
    *,
    repulsion: RepulsionParams | None = None,
    flanking: Sequence[FlankingSection] | None = None,
) -> Layout:
    """Sample a section's layout on ``plane``. Returns a :class:`Layout` of ``N`` cells.

    Parameters
    ----------
    intensity_fn
        ``(M, 3)`` micrometres -> ``(M, C)`` intensity in cells per micrometre^3.
    plane
        The plane, carrying the slab thickness the count is integrated over.
    cfg
        Supplies ``layout_mode``, ``layout_n_mc``, ``layout_envelope_slack``,
        ``layout_max_proposal_factor``, ``layout_proposal_batch``, ``repulsion``,
        ``potts_*``, ``swd_*`` and ``debug_shapes``.
    seed
        Every draw derives from it; two calls with the same seed are bitwise identical.
    repulsion
        The fitted interaction. Required when ``cfg.repulsion`` is ``True`` — there is no
        default hard core, because a hand-set one is exactly what T05 forbids. Ignored, and
        recorded as ``None``, under ablation A4 (``cfg.repulsion = False``).
    flanking
        The flanking real sections in ``plane``'s frame. Required by ``"hybrid"`` and
        ``"resample"``.

    Notes
    -----
    ``"field"`` draws ``N ~ Poisson(N_expected)`` from the slab integral, places the points
    by rejection sampling against ``sum_c lambda_c`` thinned by the interaction, then draws
    and smooths the marks. ``"hybrid"`` adds a sliced-Wasserstein polish of the positions
    toward the union of the flanking coordinate marginals **before** the marks are drawn, so
    every mark is drawn at the position its cell ends up at. ``"resample"`` reuses the
    nearest flanking section's coordinates and types unchanged: the previous version's
    behaviour, kept as the no-regression fallback, and the one mode whose count does not
    come from the intensity.
    """
    if cfg.layout_mode == "resample":
        return _resample_layout(plane, cfg, seed, flanking)
    if cfg.repulsion and repulsion is None:
        raise LayoutError(
            "sample_layout: Config.repulsion is True but no RepulsionParams were given. "
            "Fit them with fit_repulsion(training_volume, cfg, seed=...); T05 forbids a "
            "hand-set r0/R/gamma. Pass Config.replace(repulsion=False) for ablation A4."
        )
    interaction = repulsion if cfg.repulsion else None

    gen = np.random.default_rng(seed)
    mc_xyz = uniform_slab_points(plane, int(cfg.layout_n_mc), gen)
    lam_mc = _check_intensity(intensity_fn(mc_xyz), mc_xyz.shape[0], "the slab MC sample")
    n_expected = expected_count(lam_mc, plane.slab_volume)
    n_target = int(gen.poisson(n_expected))

    envelope_uv = uniform_plane_points(plane, int(cfg.layout_n_mc), gen)
    lam_envelope = _check_intensity(
        intensity_fn(plane.to_xyz(envelope_uv)), envelope_uv.shape[0], "the mid-plane MC sample"
    )
    envelope = float(lam_envelope.sum(axis=1).max()) * float(cfg.layout_envelope_slack)
    if not envelope > 0:
        raise LayoutError(
            "sample_layout: the total intensity is zero everywhere on the mid-plane MC "
            "sample, so there is nothing to sample from."
        )

    uv, n_proposals, exhausted = _propose_points(
        intensity_fn, plane, cfg, gen, n_target, envelope, interaction
    )
    if exhausted:
        warnings.warn(
            f"sample_layout: placed {uv.shape[0]} of {n_target} points in "
            f"{n_proposals} proposals (Config.layout_max_proposal_factor="
            f"{cfg.layout_max_proposal_factor}). r0="
            f"{'n/a' if interaction is None else format(interaction.r0, '.3g')} um is too "
            f"large for the requested density ({n_target / plane.area:.3g} cells/um^2): the "
            "intensity and the repulsion describe incompatible tissue.",
            ProposalBudgetWarning,
            stacklevel=2,
        )

    if cfg.layout_mode == "hybrid":
        uv = _swd_polish(uv, _flanking_targets(flanking, "hybrid"), cfg, gen)

    xyz = plane.to_xyz(uv)
    lam = _check_intensity(intensity_fn(xyz), uv.shape[0], "the sampled positions")
    marks = _categorical(lam / lam.sum(axis=1, keepdims=True), gen)
    marks = potts_smooth(
        uv, marks, np.log(lam + float(cfg.layout_intensity_eps)), cfg, generator=gen
    )

    return _build_layout(
        uv=uv,
        xyz=xyz,
        marks=marks,
        n_expected=n_expected,
        n_proposals=n_proposals,
        exhausted=exhausted,
        cfg=cfg,
        seed=seed,
        plane=plane,
        repulsion=interaction,
    )


def _build_layout(
    *,
    uv: FloatArray,
    xyz: FloatArray,
    marks: IntArray,
    n_expected: float,
    n_proposals: int,
    exhausted: bool,
    cfg: Config,
    seed: int,
    plane: Plane,
    repulsion: RepulsionParams | None,
) -> Layout:
    """Assemble the :class:`Layout`, casting to the package's storage dtypes."""
    layout = Layout(
        coords_uv=np.asarray(uv, dtype=np.float32),
        coords_xyz=np.asarray(xyz, dtype=np.float32),
        cell_type=np.asarray(marks, dtype=np.int32),
        n_expected=float(n_expected),
        n_proposals=int(n_proposals),
        budget_exhausted=bool(exhausted),
        mode=cfg.layout_mode,
        seed=int(seed),
        plane=plane,
        repulsion=repulsion,
    )
    if cfg.debug_shapes:
        n = layout.n_cells
        assert layout.coords_uv.shape == (n, 2), layout.coords_uv.shape
        assert layout.coords_xyz.shape == (n, 3), layout.coords_xyz.shape
        assert layout.cell_type.shape == (n,), layout.cell_type.shape
    return layout


def _check_intensity(values: npt.ArrayLike, n: int, what: str) -> FloatArray:
    """Validate an ``intensity_fn`` return: ``(n, C)``, finite, non-negative."""
    lam = np.asarray(values, dtype=np.float64)
    if lam.ndim != 2 or lam.shape[0] != n:
        raise LayoutError(
            f"sample_layout: intensity_fn must return (N, C) for N query points; got "
            f"{lam.shape} for {n} points at {what}"
        )
    if not np.all(np.isfinite(lam)) or lam.min() < 0.0:
        raise LayoutError(
            f"sample_layout: intensity_fn returned negative or non-finite values at {what}; "
            "an intensity is a density in cells per um^3"
        )
    return lam


def _propose_points(
    intensity_fn: IntensityFn,
    plane: Plane,
    cfg: Config,
    gen: np.random.Generator,
    n_target: int,
    envelope: float,
    repulsion: RepulsionParams | None,
) -> tuple[FloatArray, int, bool]:
    """Rejection-sample ``n_target`` points, thinned by the interaction.

    Returns ``(uv, n_proposals, budget_exhausted)`` with ``uv`` ``(n, 2)`` float64,
    ``n <= n_target``. Proposals are drawn in vectorised batches (the intensity call is the
    expensive part); the interaction test is then sequential over the survivors of the
    intensity thinning, because whether a point is accepted depends on the points accepted
    before it.
    """
    accepted = np.empty((n_target, 2), dtype=np.float64)
    n_accepted = 0
    n_proposals = 0
    budget = int(cfg.layout_max_proposal_factor) * n_target
    batch_size = int(cfg.layout_proposal_batch)

    while n_accepted < n_target and n_proposals < budget:
        batch = min(batch_size, budget - n_proposals)
        uv = uniform_plane_points(plane, batch, gen)
        lam_total = _check_intensity(intensity_fn(plane.to_xyz(uv)), batch, "a proposal batch").sum(
            axis=1
        )
        u_intensity = gen.random(batch)
        u_repulsion = gen.random(batch)
        n_proposals += batch
        survivors = np.nonzero(u_intensity < lam_total / envelope)[0]
        for i in survivors:
            if repulsion is not None and n_accepted > 0:
                distances = np.linalg.norm(accepted[:n_accepted] - uv[i][None, :], axis=1)
                if math.log(max(float(u_repulsion[i]), 1e-300)) >= repulsion.log_acceptance(
                    distances
                ):
                    continue
            accepted[n_accepted] = uv[i]
            n_accepted += 1
            if n_accepted == n_target:
                break

    return accepted[:n_accepted], n_proposals, n_accepted < n_target


def _flanking_targets(
    flanking: Sequence[FlankingSection] | None, mode: str
) -> Sequence[FlankingSection]:
    """Return the flanking sections, or raise naming the mode that needs them."""
    if not flanking:
        raise LayoutError(
            f"sample_layout: Config.layout_mode={mode!r} needs the flanking real sections; "
            "pass flanking=[flanking_from_section(s, plane) for s in ...]. Only "
            "layout_mode='field' generates without them."
        )
    return flanking


def _swd_polish(
    uv: FloatArray, flanking: Sequence[FlankingSection], cfg: Config, gen: np.random.Generator
) -> FloatArray:
    """Sliced-Wasserstein flow of ``uv`` toward the union of the flanking marginals.

    ``(n, 2)`` -> ``(n, 2)``. ``cfg.swd_polish_steps`` steps; each draws
    ``cfg.swd_n_projections`` random directions, matches the sorted 1-D projections by rank
    (quantile matching, which is the 1-D optimal transport map), averages the resulting
    displacements over directions and applies ``cfg.swd_polish_step`` of it. Borrowing the
    one genuine strength of the competing method — exact marginal geometry matching — while
    keeping the repulsion the field sampler imposed, which a full snap onto the target would
    destroy.
    """
    target = np.concatenate([f.coords_uv for f in flanking], axis=0)
    if target.shape[0] == 0:
        raise LayoutError("sample_layout: the flanking sections carry no cells to polish toward")
    points = np.array(uv, dtype=np.float64, copy=True)
    n = points.shape[0]
    if n == 0:
        return points
    quantiles = (np.arange(n, dtype=np.float64) + 0.5) / n
    for _ in range(int(cfg.swd_polish_steps)):
        angles = gen.uniform(0.0, 2.0 * math.pi, size=int(cfg.swd_n_projections))
        directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (P, 2)
        projected = points @ directions.T  # (n, P)
        target_projected = np.sort(target @ directions.T, axis=0)  # (M, P)
        order = np.argsort(projected, axis=0)
        matched = np.empty_like(projected)
        m = target_projected.shape[0]
        target_quantiles = (np.arange(m, dtype=np.float64) + 0.5) / m
        for p in range(directions.shape[0]):
            matched[order[:, p], p] = np.interp(quantiles, target_quantiles, target_projected[:, p])
        displacement = (matched - projected) @ directions / directions.shape[0]
        points += float(cfg.swd_polish_step) * displacement
    return points


def _resample_layout(
    plane: Plane, cfg: Config, seed: int, flanking: Sequence[FlankingSection] | None
) -> Layout:
    """Reuse the nearest flanking section's coordinates and types (the v20 behaviour).

    The one mode whose count does *not* come from the intensity integral: it is the
    no-regression fallback, and reproducing the previous version means reproducing that
    too. ``n_expected`` is reported as the reused count so the field is never a lie.
    """
    sections = _flanking_targets(flanking, "resample")
    nearest = min(sections, key=lambda f: (abs(f.z - float(plane.origin[2])), f.section_id))
    uv = np.asarray(nearest.coords_uv, dtype=np.float64)
    return _build_layout(
        uv=uv,
        xyz=plane.to_xyz(uv),
        marks=np.asarray(nearest.cell_type, dtype=np.int32),
        n_expected=float(uv.shape[0]),
        n_proposals=0,
        exhausted=False,
        cfg=cfg,
        seed=seed,
        plane=plane,
        repulsion=None,
    )


# --------------------------------------------------------------------------------------
# the intensity head
# --------------------------------------------------------------------------------------


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Seeded Kaiming-uniform initialisation, so construction is reproducible."""
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


def _softplus_inverse(y: float) -> float:
    """``x`` such that ``softplus(x) = y``, for ``y > 0``."""
    if y <= 0.0:
        raise LayoutError(f"_softplus_inverse: needs y > 0, got {y!r}")
    return float(np.log(np.expm1(y))) if y < 20.0 else y


def fourier_bands_for_lengthscale(extent: float, ell: float, cfg: Config) -> int:
    """How many in-plane Fourier bands the intensity head may have. Returns ``>= 1``.

    The answer T06's trainer owes to SPEC_QUESTIONS **B10**, and the reason it is here rather
    than in T06's own files: it is a statement about :class:`IntensityHead`'s basis.

    The Poisson MLE of a flexible intensity overfits the point pattern. At the default
    ``fourier_bands_xy = 8`` the head's finest band is ``2^7 = 128`` cycles across the
    section — structure of a hundredth of the field of view — and given enough steps it fits
    the sampling noise: measured on T05's known-intensity fixture the recovered correlation
    decays from 0.97 at 300 steps to 0.28 at 1200 while the NLL keeps falling. T05 specified
    no regulariser and deliberately did not invent one; its acceptance test lowered the basis
    by hand and said so.

    The principled version of that hand-lowering is a basis **tied to the fitted length-scale**
    (the third of B10's three candidate answers, and the only one that adds no new tunable and
    no extra fitting loop). The intensity is a smoothed functional of the anatomical field, and
    the field's own in-plane correlation length is ``ell_xy``: a basis function whose wavelength
    is well below ``ell`` describes variation the tissue does not have and the point pattern
    cannot constrain. Band ``b`` has wavelength ``extent / 2^b``, so the bands kept are those
    with ``extent / 2^b >= intensity_basis_ell_multiple * ell``, i.e.
    ``B = floor(log2(extent / (m * ell))) + 1``, capped at ``Config.fourier_bands_xy`` and
    floored at 1.

    Parameters
    ----------
    extent
        The in-plane extent the encoding is normalised against, in micrometres — the width of
        ``Volume.bbox``, i.e. what one normalised unit is worth.
    ell
        The fitted in-plane correlation length in micrometres (``Config.ell_xy`` after
        ``fit_lengthscale_from_sections``, never a hand-set value for a published run).
    cfg
        Supplies ``fourier_bands_xy`` (the cap) and ``intensity_basis_ell_multiple``.

    Notes
    -----
    Early stopping — B10's other candidate — was rejected rather than forgotten: the in-sample
    NLL falls monotonically while the fit deteriorates, so the stopping signal has to come
    from a section held out of the fit, which spends training data on a capacity choice that a
    length-scale the pipeline has *already fitted* answers directly. It also leaves the number
    of steps as the real hyperparameter, which is not a statement about the tissue.
    """
    if not extent > 0 or not ell > 0:
        raise LayoutError(
            f"fourier_bands_for_lengthscale: extent and ell must be > 0 um, got "
            f"{extent!r} and {ell!r}"
        )
    resolvable = float(extent) / (float(cfg.intensity_basis_ell_multiple) * float(ell))
    bands = math.floor(math.log2(resolvable)) + 1 if resolvable >= 1.0 else 1
    return max(1, min(int(cfg.fourier_bands_xy), bands))


class IntensityHead(nn.Module):
    """Per-cell-type spatial intensity ``lambda_c(x, y, z)``, in cells per micrometre^3.

    Args:
        cfg: supplies ``field_dim``, ``fourier_bands_xy/_z``, ``ctx_emb_dim``,
             ``layout_mlp_hidden``, ``layout_mlp_layers``, ``seed`` and ``debug_shapes``.
        n_types: C, the number of cell types.
        bbox: ``(2, 3)`` float32 min/max over ``(x, y, z)``, micrometres, in the data frame
              — the box the Fourier encoding is normalised against, normally
              ``Volume.bbox``.
        init_density: the volume's mean cell density in cells per micrometre^3
              (:func:`mean_cell_density`). The output bias starts at
              ``softplus^-1(init_density / C)``, so an untrained head already predicts the
              observed density and the fit only has to learn the *shape*. It is a property
              of the dataset, not a constant, which is why it is an argument and not a
              ``Config`` field.
        n_regions: number of anatomical regions, or ``None`` for a dataset without them.
    Returns from forward: ``(N, n_types)``, strictly positive.

    Structure
    ---------
    ``lambda_c = softplus(MLP_c([field_feat, fourier(xyz), region_emb]))``, the per-type
    heads sharing a trunk (one MLP with ``C`` outputs). The Fourier encoding is the
    anisotropic one of T04, evaluated on ``xyz`` normalised by ``bbox`` — the same encoding
    the field uses, so the head can express structure the field's own output has smoothed
    over.

    The region embedding, and the unknown row
    -----------------------------------------
    A cell has a region label; an arbitrary point in the slab does not. The embedding table
    therefore has ``n_regions + 1`` rows and the last one means *unknown*, which is what a
    Monte-Carlo integration point gets. Without it the integral term of the Poisson NLL and
    the likelihood term would be evaluated under different conditioning, and the fitted
    intensity would absorb the difference (T05 does not say what to do here; recorded in
    ``PROGRESS.md``).
    """

    bbox: Tensor

    def __init__(
        self,
        cfg: Config,
        n_types: int,
        bbox: npt.NDArray[Any],
        *,
        init_density: float,
        n_regions: int | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_types = int(n_types)
        self.n_regions = None if n_regions is None else int(n_regions)
        if self.n_types < 1:
            raise LayoutError(f"IntensityHead: n_types must be >= 1, got {n_types}")
        box = np.asarray(bbox, dtype=np.float64)
        if box.shape != (2, 3) or not np.all(np.isfinite(box)) or not np.all(box[1] > box[0]):
            raise LayoutError(
                f"IntensityHead: bbox must be a finite (2, 3) min/max with positive extent, "
                f"got {box.tolist() if box.size == 6 else box.shape}"
            )
        self.register_buffer("bbox", torch.from_numpy(box.astype(np.float32)))

        generator = torch.Generator().manual_seed(int(cfg.seed))
        width_in = int(cfg.field_dim) + fourier_encode_dim(cfg)
        if self.n_regions is not None:
            self.region_embedding: nn.Embedding | None = nn.Embedding(
                self.n_regions + 1, int(cfg.ctx_emb_dim)
            )
            with torch.no_grad():
                self.region_embedding.weight.normal_(0.0, 0.01, generator=generator)
            width_in += int(cfg.ctx_emb_dim)
        else:
            self.region_embedding = None

        layers: list[nn.Module] = []
        width = width_in
        for _ in range(int(cfg.layout_mlp_layers) - 1):
            linear = nn.Linear(width, int(cfg.layout_mlp_hidden))
            _init_linear(linear, generator)
            layers.extend((linear, nn.GELU()))
            width = int(cfg.layout_mlp_hidden)
        out = nn.Linear(width, self.n_types)
        _init_linear(out, generator)
        with torch.no_grad():
            out.weight.mul_(0.1)
            out.bias.fill_(_softplus_inverse(float(init_density) / self.n_types))
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """``(N, 3)``, ``(N, d_f)`` -> ``(N, n_types)``; see :meth:`forward`."""
        return cast(Tensor, super().__call__(*args, **kwargs))

    def forward(self, xyz: Tensor, field_feat: Tensor, *, region: Tensor | None = None) -> Tensor:
        """``(N, 3)`` micrometres and ``(N, d_f)`` field features -> ``(N, n_types)``.

        The output is strictly positive (softplus) and is an intensity in cells per
        micrometre^3. ``region`` is ``(N,)`` int64 region codes; ``None`` uses the unknown
        row, which is what an integration point gets.
        """
        points = xyz if isinstance(xyz, Tensor) else torch.as_tensor(xyz, dtype=torch.float32)
        points = points.to(torch.float32)
        if self.cfg.debug_shapes:
            assert points.ndim == 2, points.shape
            assert points.shape[1] == 3, points.shape
            assert field_feat.shape == (points.shape[0], int(self.cfg.field_dim)), field_feat.shape
        lo = self.bbox[0][None, :]
        span = torch.clamp(self.bbox[1] - self.bbox[0], min=1e-6)[None, :]
        encoding = fourier_encode(2.0 * (points - lo) / span - 1.0, self.cfg)
        parts = [field_feat.to(torch.float32), encoding]
        if self.region_embedding is not None:
            codes = (
                torch.full((points.shape[0],), int(self.n_regions or 0), dtype=torch.long)
                if region is None
                else region.to(torch.long)
            )
            parts.append(self.region_embedding(codes))
        out = torch.nn.functional.softplus(cast(Tensor, self.mlp(torch.cat(parts, dim=1))))
        if self.cfg.debug_shapes:
            assert out.shape == (points.shape[0], self.n_types), out.shape
        return out


def intensity_fn_from_head(
    head: IntensityHead,
    field_fn: FieldFn,
    *,
    chunk: int | None = None,
) -> IntensityFn:
    """Wrap a trained head and a field as the numpy :data:`IntensityFn` the sampler takes.

    Evaluated under ``torch.no_grad`` in chunks of ``chunk`` (default
    ``head.cfg.grf_chunk_points``) points. The returned callable maps ``(M, 3)`` float64
    micrometres to ``(M, C)`` float64 intensity.
    """
    size = int(head.cfg.grf_chunk_points if chunk is None else chunk)

    def intensity(points: FloatArray) -> FloatArray:
        query = np.asarray(points, dtype=np.float32)
        out = np.empty((query.shape[0], head.n_types), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, query.shape[0], size):
                block = torch.from_numpy(query[start : start + size])
                out[start : start + size] = (
                    head(block, field_fn(block)).detach().numpy().astype(np.float64)
                )
        return out

    return intensity


def fit_intensity_head(
    head: IntensityHead,
    sections: Sequence[Section],
    field_fn: FieldFn,
    cfg: Config,
    *,
    seed: int,
) -> list[float]:
    """Fit ``head`` to ``sections`` by the inhomogeneous-Poisson-process NLL.

    Each section is treated as a slab of thickness ``Section.thickness`` (defaulted to the
    volume's median spacing by the loader, and flagged there as assumed). The integral term
    is a Monte-Carlo estimate over ``cfg.layout_n_mc`` uniform points in the slab **volume**,
    redrawn every step, and the observed cells' depths are **jittered uniformly within their
    slab**, also every step. Both are load-bearing, and both were found by measurement:

    * *Redrawn MC points.* Against a fixed integration set, the intensity can grow spikes
      between the integration points: the likelihood term rises without the integral term
      noticing. Redrawing makes the estimator's gradient unbiased at every step.
    * *Jittered depths.* Every cell of a section is recorded at that section's nominal ``z``,
      while the integral runs over a continuous slab. An intensity that puts all its mass in
      thin sheets at the nominal depths therefore scores arbitrarily well — measured on the
      synthetic fixture, the NLL fell three nats below the value at the **true** intensity
      while the correlation with that truth stayed at zero. Jittering says what the data
      actually says, which is that a cell is *somewhere* in its slab, and it is the same
      statement T07's ``L_thick`` makes.

    Because both change every step, the field features are recomputed each step (under
    ``torch.no_grad``: this is a fit of the head against a *frozen* field, not joint
    training — T06's trainer optimises the two together). Full-batch Adam for
    ``cfg.layout_fit_steps`` steps at ``cfg.layout_fit_lr``.

    A note on capacity. The Poisson MLE of a flexible intensity overfits the point pattern:
    with the default ``fourier_bands_xy = 8`` the head resolves structure down to a
    hundredth of the section and, given enough steps, fits the sampling noise — the NLL keeps
    falling while the recovered intensity walks away from the truth (measured: r 0.97 at 300
    steps, 0.28 at 1200). The spatial basis has to be matched to the scale the intensity
    actually varies on. Recorded in ``SPEC_QUESTIONS`` B10; T05 does not specify a
    regulariser and this function does not invent one.

    Deterministic given ``seed`` (which drives the jitter and the MC points). Returns the
    per-step loss, normalised by the total cell count so the numbers are comparable across
    datasets.
    """
    from spatialcpav25_gen.losses.reconstruction import layout_poisson_nll

    if not sections:
        raise LayoutError("fit_intensity_head: needs at least one section")
    gen = np.random.default_rng(seed)
    planes = [section_plane(s) for s in sections]
    cell_xyz = [to_xyz(s).astype(np.float64) for s in sections]
    codes = [torch.from_numpy(np.asarray(s.cell_type, dtype=np.int64)) for s in sections]
    total_cells = sum(s.n_cells for s in sections)

    optimiser = torch.optim.Adam(head.parameters(), lr=float(cfg.layout_fit_lr))
    history: list[float] = []
    for _ in range(int(cfg.layout_fit_steps)):
        optimiser.zero_grad()
        loss = torch.zeros((), dtype=torch.float32)
        for section, plane, xyz, code in zip(sections, planes, cell_xyz, codes, strict=True):
            half = 0.5 * float(section.thickness)
            jittered = xyz.copy()
            jittered[:, 2] += gen.uniform(-half, half, size=jittered.shape[0])
            cells = torch.from_numpy(jittered.astype(np.float32))
            mc = torch.from_numpy(
                uniform_slab_points(plane, int(cfg.layout_n_mc), gen).astype(np.float32)
            )
            with torch.no_grad():
                cell_feat = field_fn(cells).detach()
                mc_feat = field_fn(mc).detach()
            loss = loss + layout_poisson_nll(
                head(cells, cell_feat), code, head(mc, mc_feat), plane.slab_volume, cfg
            )
        loss = loss / float(total_cells)
        loss.backward()  # type: ignore[no-untyped-call]
        optimiser.step()
        history.append(float(loss.detach()))
    return history
