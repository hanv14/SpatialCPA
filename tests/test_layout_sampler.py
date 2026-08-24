"""The test the layout sampler never had: is the point pattern drawn from lambda?

`reports/r11_envelope.md` found three defects in the field layout, and the third is the one
that matters here: the rejection sampler's envelope is the maximum of `sum_c lambda_c` over a
`layout_n_mc` sample of the mid-plane, i.e. a *sampled* maximum with a 140-853x spread across
sections, and the acceptance ratio `lambda / envelope` is never clamped. Wherever the true
intensity exceeds that sampled maximum the ratio passes 1 and the point is accepted with
certainty, so the realised draw is from `min(lambda, envelope)`: peaks are flattened and the
pattern is biased, not merely short of points.

None of T05's tests could see that. They measure a fitted intensity against a fixture whose
own truth is a smooth field, at a max/mean the defect barely bites, and they compare summary
statistics (count, `g(r)`, purity, localization) rather than the density itself.

`reports/r11_fix_options.md` closes with the load-bearing fact: `sample_layout` takes an
`intensity_fn`, not a model, so a sampler can be validated with **no fit and no data** by
handing it a closed-form intensity whose true density is known. That is what this file does.

The intensity, and why it has a closed form
-------------------------------------------
`SpikyIntensity` is a sum of isotropic Gaussian bumps on a floor. The integral of a Gaussian
over an axis-aligned rectangle is a product of two `erf` differences, so the expected number
of points in **any** rectangle — the whole window, a reference bin, one type's share of a bin
— is exact arithmetic rather than a second Monte-Carlo estimate. Nothing here is fitted,
nothing is loaded, and the file imports no fixture.

Its parameters are chosen to put `max / mean` in the **hundreds** (`test_the_regime_is_the_one
_that_broke_the_sampler` measures it and pins it), because that is the regime `r11` found on
`section_4` and the regime in which the envelope's truncation is the dominant error.

What is asserted
----------------
* `test_grid_total_density_is_correct` — the slab integral, and that every point asked for is
  placed.
* `test_grid_spatial_density_is_correct` — per-bin counts against the exact rectangle
  integrals, on a reference grid the sampler does not share.
* `test_grid_per_type_mix_is_correct` — global and per-bump composition against the exact
  per-type shares.
* `test_grid_converges_with_resolution` — the midpoint rule's `O(h^2)` error, measured over a
  resolution sweep, which is what makes `Config.layout_grid_cells` a tunable with a
  convergence check rather than a magic number.
* `test_rejection_sampler_fails_the_same_criterion` — the **negative control**. The same
  assertions on the same intensity with `layout_sampler="rejection"`, asserted to *fail*: a
  test that both samplers pass measures nothing.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.special import erf
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.planes import Plane, uniform_plane_points
from spatialcpav25_gen.model.layout import (
    FloatArray,
    Layout,
    ProposalBudgetWarning,
    mid_plane_grid,
    sample_layout,
)

SEED = 20260824

# The window. Micrometres, and large relative to the bumps: max/mean of a bump field is
# `Area / (n_bumps * 2 pi s^2)`, so the spread this file needs comes from the ratio of the
# window to the features on it, exactly as it does on a real section.
HALF_EXTENT = 500.0
THICKNESS = 25.0

# The bumps. Three types, three well-separated centres, one shared width. Each bump carries a
# *mixture* of types rather than a single one (70/15/15 below), so the per-type test has
# something to be wrong about: a sampler that placed points correctly but drew marks from the
# wrong conditional would pass a pure-type check trivially.
BUMP_SIGMA = 15.0
BUMP_CENTRES: tuple[tuple[float, float], ...] = ((-250.0, -250.0), (250.0, -250.0), (0.0, 250.0))
BUMP_MAJOR = 0.70
BUMP_MINOR = 0.15

# Cells per micrometre^3. Scaled so the slab holds ~20000 points: enough that a reference bin
# holding a whole bump has a multinomial standard deviation near 1% of its own count, which is
# what lets the tolerances below be tight enough to catch a 2% distortion.
BUMP_PEAK = 0.9
FLOOR = 2.0e-6

# Reference bins per axis for the spatial test. Deliberately **not** a divisor relationship
# with `Config.layout_grid_cells`: a reference grid aligned to the sampler's own grid would
# hide a systematic within-cell error by integrating over exactly the cells that carry it.
# The negative control's intensity. A fifth of `BUMP_SIGMA`, which puts the bumps' *peaks*
# below one point of a default `layout_n_mc` sample — the regime in which the envelope stops
# being a fair estimate of the supremum and starts being a lottery. `test_needle_is_a_needle`
# checks that arithmetic rather than trusting it.
NEEDLE_SIGMA = 3.0
NEEDLE_GRID_CELLS = 512
CONTROL_SEEDS = (SEED, SEED + 1, SEED + 2)

REFERENCE_BINS = 12
MIN_EXPECTED_PER_BIN = 40.0

# Tolerance on a reference bin, as (relative, sigmas). The relative part is the midpoint
# rule's own bias at `layout_grid_cells=128` (h^2 / (12 s^2) = 2.3% at the peak, less
# everywhere else); the sigma part is the multinomial noise of the bin's own count.
BIN_REL_TOL = 0.06
BIN_SIGMAS = 4.0

# Tolerance on the Monte-Carlo slab integral, which is `expected_count`'s and not the
# sampler's. See `test_grid_total_density_is_correct` for where the number comes from.
MC_INTEGRAL_TOL = 0.05


class SpikyIntensity:
    """A closed-form per-type intensity: Gaussian bumps on a floor. Cells per micrometre^3.

    ``__call__`` is the ``IntensityFn`` the sampler consumes: ``(M, 3)`` micrometres ->
    ``(M, C)``. It is independent of ``z``, so the slab integral is the in-plane integral
    times the thickness and :meth:`integral` needs no third dimension.

    :meth:`integral` is the reason this class exists rather than a closure: the integral of
    ``exp(-((u - u0)^2 + (v - v0)^2) / (2 s^2))`` over an axis-aligned rectangle separates into
    a product of two one-dimensional integrals, each of which is
    ``s sqrt(pi / 2) [erf((b - u0) / (s sqrt 2)) - erf((a - u0) / (s sqrt 2))]``. So every
    expected count this file asserts against is exact arithmetic, not a second estimate.
    """

    def __init__(self, sigma: float = BUMP_SIGMA) -> None:
        self.centres = np.asarray(BUMP_CENTRES, dtype=np.float64)  # (K, 2)
        self.sigma = float(sigma)
        self.n_types = self.centres.shape[0]
        # (K, C): bump k is BUMP_MAJOR type k and BUMP_MINOR each of the others.
        weights = np.full((self.n_types, self.n_types), BUMP_MINOR, dtype=np.float64)
        np.fill_diagonal(weights, BUMP_MAJOR)
        self.weights = BUMP_PEAK * weights

    def __call__(self, xyz: FloatArray) -> FloatArray:
        """``(M, 3)`` micrometres -> ``(M, C)`` intensity, cells per micrometre^3."""
        points = np.asarray(xyz, dtype=np.float64)
        offsets = points[:, None, :2] - self.centres[None, :, :]  # (M, K, 2)
        bump = np.exp(-np.sum(offsets**2, axis=2) / (2.0 * self.sigma**2))  # (M, K)
        return np.asarray(FLOOR + bump @ self.weights, dtype=np.float64)

    def integral(self, lo: FloatArray, hi: FloatArray) -> FloatArray:
        """Exact ``(C,)`` in-plane integral over the rectangle ``[lo, hi]``, cells per um.

        ``lo`` and ``hi`` are ``(2,)`` in-plane micrometres. Multiply by the slab thickness
        for an expected count.
        """
        low = np.asarray(lo, dtype=np.float64)
        high = np.asarray(hi, dtype=np.float64)
        area = float(np.prod(high - low))
        scale = self.sigma * np.sqrt(0.5 * np.pi)
        root = self.sigma * np.sqrt(2.0)
        span = scale * (
            erf((high[None, :] - self.centres) / root) - erf((low[None, :] - self.centres) / root)
        )  # (K, 2)
        bump_mass = span[:, 0] * span[:, 1]  # (K,)
        return np.asarray(FLOOR * area + bump_mass @ self.weights, dtype=np.float64)

    def window_integral(self, plane: Plane) -> FloatArray:
        """Exact ``(C,)`` in-plane integral over ``plane``'s whole window."""
        extent = np.asarray(plane.half_extent, dtype=np.float64)
        return self.integral(-extent, extent)


def make_plane() -> Plane:
    """The axis-aligned window the whole file samples on."""
    return Plane(
        origin=np.zeros(3),
        e1=np.array([1.0, 0.0, 0.0]),
        e2=np.array([0.0, 1.0, 0.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        half_extent=np.array([HALF_EXTENT, HALF_EXTENT]),
        thickness=THICKNESS,
    )


def sampler_cfg(**overrides: object) -> Config:
    """The config every test here starts from.

    ``repulsion=False`` because the interaction is not what is under test and a Strauss
    thinning would put a second, correct, deviation between the empirical density and
    ``lambda``. ``potts_beta=0.0`` under Gibbs makes the mark update a redraw from
    ``lambda_c / sum_c lambda_c`` with no spatial coupling, which is what lets the per-type
    assertions compare against the exact per-type shares rather than against a smoothed
    version of them. ``layout_n_mc`` is raised because the slab integral is a Monte-Carlo
    estimate over a spiky function and the default 4096 has a wide enough standard error to
    swamp what the sampler is being asked about.
    """
    defaults: dict[str, object] = {
        "layout_mode": "field",
        "repulsion": False,
        "potts_beta": 0.0,
        "potts_update": "gibbs",
        "layout_n_mc": 200_000,
    }
    return Config().replace(**{**defaults, **overrides})


def draw(cfg: Config, intensity: SpikyIntensity, plane: Plane, *, seed: int = SEED) -> Layout:
    """One layout on the closed-form intensity."""
    return sample_layout(intensity, plane, cfg, seed=seed)


def bin_edges(plane: Plane, n_bins: int) -> FloatArray:
    """``(n_bins + 1, 2)`` reference-bin edges along ``e1`` and ``e2``."""
    extent = np.asarray(plane.half_extent, dtype=np.float64)
    return np.stack(
        [np.linspace(-extent[axis], extent[axis], n_bins + 1) for axis in (0, 1)], axis=1
    )


def expected_bin_counts(
    intensity: SpikyIntensity, plane: Plane, n_bins: int, n_points: int
) -> FloatArray:
    """``(n_bins, n_bins)`` expected counts per reference bin, given ``n_points`` placed.

    Conditional on the realised count, so this is the *multinomial* expectation — the exact
    rectangle integral normalised by the window integral, times the points actually placed.
    Comparing against the unconditional Poisson mean instead would fold the count's own
    fluctuation into a test about position.
    """
    edges = bin_edges(plane, n_bins)
    total = float(intensity.window_integral(plane).sum())
    counts = np.empty((n_bins, n_bins), dtype=np.float64)
    for i in range(n_bins):
        for j in range(n_bins):
            lo = np.array([edges[i, 0], edges[j, 1]])
            hi = np.array([edges[i + 1, 0], edges[j + 1, 1]])
            counts[i, j] = n_points * float(intensity.integral(lo, hi).sum()) / total
    return counts


def observed_bin_counts(layout: Layout, plane: Plane, n_bins: int) -> FloatArray:
    """``(n_bins, n_bins)`` counts of the layout's points per reference bin."""
    edges = bin_edges(plane, n_bins)
    uv = np.asarray(layout.coords_uv, dtype=np.float64)
    counts, _, _ = np.histogram2d(uv[:, 0], uv[:, 1], bins=[edges[:, 0], edges[:, 1]])
    return np.asarray(counts, dtype=np.float64)


def bin_deviations(layout: Layout, intensity: SpikyIntensity, plane: Plane) -> FloatArray:
    """Per-bin ``|observed - expected|`` in units of the bin's own tolerance.

    A value at or below 1 is a bin the sampler placed correctly; the tolerance is
    ``BIN_REL_TOL * expected + BIN_SIGMAS * sqrt(expected)``, i.e. the midpoint rule's bias
    plus the bin count's own multinomial noise. Only bins with at least
    ``MIN_EXPECTED_PER_BIN`` expected points are returned: below that the noise term dominates
    so completely that the bin can say nothing either way.
    """
    expected = expected_bin_counts(intensity, plane, REFERENCE_BINS, layout.n_cells)
    observed = observed_bin_counts(layout, plane, REFERENCE_BINS)
    live = expected >= MIN_EXPECTED_PER_BIN
    tolerance = BIN_REL_TOL * expected[live] + BIN_SIGMAS * np.sqrt(expected[live])
    return np.asarray(np.abs(observed[live] - expected[live]) / tolerance, dtype=np.float64)


# --------------------------------------------------------------------------------------
# the regime
# --------------------------------------------------------------------------------------


def test_the_regime_is_the_one_that_broke_the_sampler():
    """max / mean of the total intensity is in the hundreds, as it was on `section_4`.

    Pinned rather than asserted loosely, because every other test in this file is only
    interesting in this regime: at a max/mean of 5 the envelope's truncation is invisible and
    the biased sampler passes everything below.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    peak = float(intensity(np.array([[*BUMP_CENTRES[0], 0.0]])).sum())
    mean = float(intensity.window_integral(plane).sum()) / plane.area
    assert 100.0 < peak / mean < 1000.0, peak / mean


# --------------------------------------------------------------------------------------
# the grid sampler
# --------------------------------------------------------------------------------------


def test_grid_total_density_is_correct():
    """The count matches the exact slab integral, and every point asked for is placed.

    Three separate statements, kept separate because they can fail for different reasons.

    1. ``layout.n_expected`` is the Monte-Carlo slab integral, and it must land within
       ``MC_INTEGRAL_TOL`` of the exact one. That tolerance is the estimator's, not the
       sampler's: the bumps cover 0.4% of the window, so at ``layout_n_mc`` points only
       ``0.004 * layout_n_mc`` of them see a bump at all and the relative standard error is
       about ``1 / sqrt`` of that — 3.4% at 200000. Tightening it means raising
       ``layout_n_mc``, and the count's precision is a property of ``expected_count``, which
       option D does not change.
    2. ``N`` is a Poisson draw around ``n_expected``, checked at five standard deviations.
    3. The sampler **places all N**. With no interaction there is nothing to reject, so
       ``n_cells == n_proposals`` exactly — the property the envelope could never offer, and
       the one that made ``r11``'s acceptance 0.12%.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    cfg = sampler_cfg()
    exact = float(intensity.window_integral(plane).sum()) * plane.thickness

    layout = draw(cfg, intensity, plane)
    assert abs(layout.n_expected - exact) / exact < MC_INTEGRAL_TOL, (layout.n_expected, exact)
    assert abs(layout.n_cells - layout.n_expected) < 5.0 * np.sqrt(layout.n_expected), (
        layout.n_cells,
        layout.n_expected,
    )
    assert not layout.budget_exhausted
    assert layout.n_cells == layout.n_proposals


def test_grid_spatial_density_is_correct():
    """Per-bin counts match the exact rectangle integrals on an unaligned reference grid."""
    intensity = SpikyIntensity()
    plane = make_plane()
    layout = draw(sampler_cfg(), intensity, plane)
    deviations = bin_deviations(layout, intensity, plane)
    assert deviations.size >= 4, deviations.size
    assert float(deviations.max()) <= 1.0, (
        f"worst bin is {deviations.max():.2f} tolerances out over {deviations.size} live bins"
    )


def test_grid_per_type_mix_is_correct():
    """Global and per-bump compositions match the exact per-type shares.

    The marks are drawn at the sampled positions from ``lambda_c / sum_c lambda_c``, so a
    sampler that put points in the wrong places would get the *global* mix wrong even with a
    perfect conditional — which is why the global check is here and not only in T05.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    layout = draw(sampler_cfg(), intensity, plane)
    types = np.asarray(layout.cell_type, dtype=np.int64)
    n = layout.n_cells

    share = intensity.window_integral(plane)
    share = share / share.sum()
    observed = np.bincount(types, minlength=intensity.n_types) / n
    assert np.max(np.abs(observed - share)) < 0.01, (observed, share)

    # Locally: inside two sigma of bump k the composition is the bump's own mixture, which is
    # BUMP_MAJOR / (BUMP_MAJOR + 2 * BUMP_MINOR) for type k once the floor is negligible.
    uv = np.asarray(layout.coords_uv, dtype=np.float64)
    for k, centre in enumerate(BUMP_CENTRES):
        near = np.linalg.norm(uv - np.asarray(centre)[None, :], axis=1) <= 2.0 * BUMP_SIGMA
        assert int(near.sum()) > 500, (k, int(near.sum()))
        local = intensity(np.array([[*centre, 0.0]]))[0]
        expected = float(local[k] / local.sum())
        got = float(np.mean(types[near] == k))
        assert abs(got - expected) < 0.03, (k, got, expected)


@pytest.mark.parametrize("cells", [32, 64, 128, 256])
def test_grid_converges_with_resolution(cells: int):
    """Every resolution from 32 cells up already places the density inside tolerance.

    The convergence itself is measured in ``test_grid_error_falls_as_h_squared``; this is the
    statement ``Config.layout_grid_cells`` needs on its own, which is that the default is not
    perched on the edge of a cliff.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    layout = draw(sampler_cfg(layout_grid_cells=cells), intensity, plane)
    assert float(bin_deviations(layout, intensity, plane).max()) <= 1.0


def test_grid_error_falls_as_h_squared():
    """The grid's own quadrature error falls with the cell size. The convergence check.

    Measured with no sampling at all: the grid's total mass ``sum_k lambda(centre_k) *
    cell_area`` against the exact window integral. That is the composite midpoint rule, whose
    error is ``O(h^2)``, and it is the *whole* of the grid sampler's approximation — the
    multinomial draw from those weights is exact. Measuring it here rather than through a draw
    is what makes the statement about the resolution rather than about a seed.

    This is what makes ``Config.layout_grid_cells`` a tunable with a convergence check rather
    than a magic number, which is the property ``r11`` contrasts with a sampled maximum.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    exact = float(intensity.window_integral(plane).sum())

    errors = []
    for cells in (32, 64, 128, 256):
        centres, cell = mid_plane_grid(plane, sampler_cfg(layout_grid_cells=cells))
        # The cells tile the window exactly, which is what lets the quadrature be a plain sum.
        assert np.isclose(centres.shape[0] * float(np.prod(cell)), plane.area)
        lam = intensity(plane.to_xyz(centres)).sum(axis=1)
        errors.append(abs(float(lam.sum() * float(np.prod(cell))) - exact) / exact)

    # Halving h at least halves the error. A clean O(h^2) does far better than that, and the
    # weaker inequality is the one that stays true once the error reaches float noise.
    for coarse, fine in itertools.pairwise(errors):
        assert fine < 0.6 * coarse + 1e-9, errors
    assert errors[-1] < 1e-3, errors


def test_grid_is_deterministic():
    """Same seed -> bitwise identical layout; a different seed -> a different one."""
    intensity = SpikyIntensity()
    plane = make_plane()
    cfg = sampler_cfg()
    first = draw(cfg, intensity, plane)
    again = draw(cfg, intensity, plane)
    other = draw(cfg, intensity, plane, seed=SEED + 1)
    assert np.array_equal(first.coords_uv, again.coords_uv)
    assert np.array_equal(first.cell_type, again.cell_type)
    shared = min(first.n_cells, other.n_cells)
    assert not np.array_equal(first.coords_uv[:shared], other.coords_uv[:shared])


def test_grid_records_the_sampler_it_used():
    """The layout's provenance says which mode produced it, for both samplers."""
    intensity = SpikyIntensity()
    plane = make_plane()
    assert draw(sampler_cfg(), intensity, plane).mode == "field"
    assert Config().layout_sampler == "grid"


# --------------------------------------------------------------------------------------
# the negative control: the retained rejection sampler, on the same closed-form intensity
# --------------------------------------------------------------------------------------


def near_bump_fraction(layout: Layout, radius: float) -> float:
    """Fraction of the layout's points within ``radius`` of any bump centre."""
    uv = np.asarray(layout.coords_uv, dtype=np.float64)
    near = np.zeros(uv.shape[0], dtype=bool)
    for centre in BUMP_CENTRES:
        near |= np.linalg.norm(uv - np.asarray(centre)[None, :], axis=1) <= radius
    return float(near.mean())


def exact_near_bump_fraction(intensity: SpikyIntensity, plane: Plane, radius: float) -> float:
    """The same fraction, exactly, from the closed-form integrals.

    The bumps are well separated and their bounding squares disjoint, so the mass within
    ``radius`` is the sum of three exact rectangle integrals minus the floor's contribution
    outside the disc — which is bounded by the floor's share of the window and is negligible
    at the amplitudes here. Asserted rather than assumed: see ``test_needle_is_a_needle``.
    """
    total = float(intensity.window_integral(plane).sum())
    inside = 0.0
    for centre in BUMP_CENTRES:
        lo = np.asarray(centre, dtype=np.float64) - radius
        hi = np.asarray(centre, dtype=np.float64) + radius
        inside += float(intensity.integral(lo, hi).sum())
    return inside / total


def test_needle_is_a_needle():
    """The control intensity is sharp enough that ``layout_n_mc`` points can miss it entirely.

    The envelope is a maximum, so what matters is whether a uniform sample lands near a bump's
    *peak* — not merely inside its support. The three ``NEEDLE_SIGMA`` peaks cover
    ``3 * pi * sigma^2`` of a 10^6 um^2 window, where ``lambda`` is within 40% of its
    supremum, and ``Config``'s default ``layout_n_mc = 4096`` puts under one point there. That
    is the premise of the two tests below, and it is arithmetic rather than luck.
    """
    plane = make_plane()
    covered = 3.0 * np.pi * NEEDLE_SIGMA**2
    assert covered / plane.area * Config().layout_n_mc < 1.0
    intensity = SpikyIntensity(NEEDLE_SIGMA)
    peak = float(intensity(np.array([[*BUMP_CENTRES[0], 0.0]])).sum())
    mean = float(intensity.window_integral(plane).sum()) / plane.area
    assert peak / mean > 1000.0, peak / mean


def test_the_envelope_is_a_sampled_maximum():
    """`r11` defect 3, reproduced with no fit, no data and no sampling.

    The envelope ``sample_layout`` builds under ``layout_sampler="rejection"`` is
    ``max(sum_c lambda_c)`` over a uniform ``layout_n_mc`` sample of the mid-plane, times
    ``layout_envelope_slack``. It is therefore a **random variable**, not a bound: rebuilt at
    a different seed it takes a different value, and on a sharp intensity it lands far below
    the true supremum. Both are asserted, and the second is what makes the acceptance ratio
    ``lambda / envelope`` exceed 1 — where it does, an unclamped test accepts with certainty
    and the realised draw is from ``min(lambda, envelope)``.
    """
    intensity = SpikyIntensity(NEEDLE_SIGMA)
    plane = make_plane()
    cfg = Config()
    supremum = float(intensity(np.array([[*BUMP_CENTRES[0], 0.0]])).sum())
    envelopes = []
    for seed in range(8):
        generator = np.random.default_rng(seed)
        uv = uniform_plane_points(plane, int(cfg.layout_n_mc), generator)
        peak = float(intensity(plane.to_xyz(uv)).sum(axis=1).max())
        envelopes.append(peak * float(cfg.layout_envelope_slack))

    assert max(envelopes) / min(envelopes) > 5.0, envelopes
    assert min(envelopes) < 0.5 * supremum, (min(envelopes), supremum)


def test_grid_sampler_places_the_needle_correctly():
    """The grid sampler puts the right share of its points on the bumps, at every seed."""
    intensity = SpikyIntensity(NEEDLE_SIGMA)
    plane = make_plane()
    cfg = sampler_cfg(layout_grid_cells=NEEDLE_GRID_CELLS)
    radius = 3.0 * NEEDLE_SIGMA
    exact = exact_near_bump_fraction(intensity, plane, radius)
    for seed in CONTROL_SEEDS:
        got = near_bump_fraction(draw(cfg, intensity, plane, seed=seed), radius)
        assert abs(got - exact) < 0.02, (seed, got, exact)


def test_rejection_sampler_fails_the_same_criterion():
    """The control: the retained rejection sampler gets the same intensity wrong.

    A validation both samplers pass measures nothing. Two failures are asserted, and both are
    the ones ``r11`` diagnosed.

    * It **starves**. Acceptance is ``mean / envelope``, which is a fraction of a percent in
      this regime, so ``layout_max_proposal_factor * N`` proposals do not place ``N`` points
      and the layout comes back truncated — loudly, via :class:`ProposalBudgetWarning`, which
      is the one part of the old sampler that was already honest.
    * It **flattens, differently at every seed**. Because the envelope is a sampled maximum
      (:func:`test_the_envelope_is_a_sampled_maximum`), the share of points landing on the
      bumps is a function of that seed's luck rather than of ``lambda``: measured across three
      seeds it spans most of the unit interval, while :func:`test_grid_sampler_places_the_
      needle_correctly` holds the same quantity to 0.02 on the same three seeds.

    If a future change makes the rejection sampler correct, this test fails and says so.
    """
    intensity = SpikyIntensity(NEEDLE_SIGMA)
    plane = make_plane()
    cfg = sampler_cfg(layout_sampler="rejection", layout_n_mc=int(Config().layout_n_mc))
    radius = 3.0 * NEEDLE_SIGMA
    exact = exact_near_bump_fraction(intensity, plane, radius)

    fractions = []
    for seed in CONTROL_SEEDS:
        with pytest.warns(ProposalBudgetWarning):
            layout = draw(cfg, intensity, plane, seed=seed)
        assert layout.budget_exhausted
        assert layout.n_cells < 0.75 * layout.n_expected, (seed, layout.n_cells)
        fractions.append(near_bump_fraction(layout, radius))

    assert max(fractions) - min(fractions) > 0.3, fractions
    assert max(abs(value - exact) for value in fractions) > 0.3, (fractions, exact)


def test_rejection_sampler_starves_on_the_bump_field():
    """Even where the envelope is a fair estimate of the supremum, acceptance is hopeless.

    On the ``BUMP_SIGMA`` field a ``layout_n_mc`` sample does land on the bumps, so the
    envelope is close to the true supremum and the *shape* of the draw is roughly right. The
    sampler still fails, for the other reason: acceptance is ``mean / envelope``, the max/mean
    pinned by :func:`test_the_regime_is_the_one_that_broke_the_sampler` is in the hundreds, and
    ``layout_max_proposal_factor = 20`` cannot cover it. This is ``r11``'s 0.12% acceptance,
    reproduced without a fit.
    """
    intensity = SpikyIntensity()
    plane = make_plane()
    cfg = sampler_cfg(layout_sampler="rejection")
    with pytest.warns(ProposalBudgetWarning):
        layout = draw(cfg, intensity, plane)
    assert layout.budget_exhausted
    assert layout.n_cells < 0.5 * layout.n_expected, (layout.n_cells, layout.n_expected)
    assert layout.n_cells / layout.n_proposals < 0.01, (layout.n_cells, layout.n_proposals)
