"""T05 acceptance tests for the layout head.

Every numeric threshold here is the spec's: Pearson r > 0.9 for the recovered intensity,
5% on the expected count, ``r0`` on the hard core, 0.15 on the pair-correlation match, 50%
retention for a 2% cell type, 10% on cell-type localization.

Two of them carry **negative controls**, and the controls are assertions rather than
comments: a positive result whose control also passes measures nothing.

* ``test_pcf_matches_real`` additionally asserts that a **pure-Poisson** sampler
  (``Config.repulsion=False``, ablation A4) *fails* the same criterion. Running that control
  is what showed the criterion's stated range to be blind — see the test's own docstring —
  so it now measures both ranges and pins the blindness as a fact.
* ``test_rare_types_survive`` additionally asserts that a deliberately **over-smoothed**
  Potts coupling *does* destroy the rare type, so the 50% criterion can detect
  over-smoothing rather than being satisfied by any coupling at all.
* ``test_hardcore_respected`` and ``test_icm_erases_rare_types`` carry controls too: the
  first that switching the interaction off *does* produce pairs inside the hard core, the
  second that the update rule T05 names (ICM) erases a 2% cell type on its own.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import Section, TrainingVolume, Volume, to_xyz
from spatialcpav25_gen.infer.planes import Plane, plane_from_normal, section_plane
from spatialcpav25_gen.losses.reconstruction import layout_poisson_nll
from spatialcpav25_gen.model.layout import (
    FloatArray,
    IntensityFn,
    IntensityHead,
    LayoutError,
    ProposalBudgetWarning,
    RepulsionParams,
    expected_count,
    fit_intensity_head,
    fit_potts_beta,
    fit_repulsion,
    flanking_from_section,
    mean_cell_density,
    pair_correlation,
    potts_beta_grid,
    potts_smooth,
    sample_layout,
    type_purity,
)

from tests.fixtures.synthetic import GroundTruthField, make_synthetic_volume

SEED = 20250816

# The fixture's cell types are drawn as argmax(type_logits + 0.6 * Gumbel), which is a
# draw from softmax(logits / 0.6): the fixture's *own* generative composition.
FIXTURE_TYPE_TEMPERATURE = 0.6

# What the tests hand the layout head instead. A trained intensity head is a smoothed
# estimate of the composition, not the generative one; 1.5 stands in for that gap, so a
# draw from it is measurably less organised than the tissue and the Potts coupling has work
# to do. See `fixture_intensity`.
MODEL_TYPE_TEMPERATURE = 1.5

# The prevalence T05's rare-type criterion is stated at.
RARE_PREVALENCE = 0.02


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def split(volume: Volume, cfg: Config):
    """The fixture volume split into training sections and held-out ones."""
    return split_holdout(volume, "alternating", 0, cfg)


@pytest.fixture(scope="module")
def training(split) -> TrainingVolume:
    """The synthetic volume with its held-out sections removed (leakage discipline)."""
    return split[0]


@pytest.fixture(scope="module")
def held_out(split):
    """The held-out sections. Only the definition-of-done test may look at them."""
    return split[1]


@pytest.fixture(scope="module")
def repulsion(training: TrainingVolume, cfg: Config) -> RepulsionParams:
    """The interaction fitted once on the training sections."""
    return fit_repulsion(training, cfg, seed=SEED)


@pytest.fixture(scope="module")
def beta(training: TrainingVolume, gt_field, cfg: Config) -> float:
    """The Potts coupling fitted once, against the intensity the marks are drawn from."""
    density = mean_cell_density(training.sections)
    return fit_potts_beta(training, fixture_intensity(gt_field, density), cfg, seed=SEED)


def fixture_intensity(
    gt: GroundTruthField, density_3d: float, temperature: float = MODEL_TYPE_TEMPERATURE
) -> IntensityFn:
    """A per-type intensity: constant total, spatially varying composition.

    ``sum_c lambda_c`` is flat, so the count and pair-correlation tests are about the point
    process alone, while the per-type split is the fixture's own generative composition, so
    the mark tests are about the marks alone.

    ``temperature`` above the fixture's own ``FIXTURE_TYPE_TEMPERATURE`` flattens that
    composition. The default is deliberately flatter than the truth: a *fitted* intensity
    head is a smoothed estimate of the tissue's composition, never the generative one, so a
    draw from it is less organised than the tissue and the Potts coupling has something to
    do. Handing the tests the exact generative composition would make the fitted ``beta``
    zero by construction and ``test_potts_improves_purity`` vacuous.
    """

    def intensity(xyz: FloatArray) -> FloatArray:
        logits = gt.type_logits(gt.latent(np.asarray(xyz, dtype=np.float64)))
        logits = logits / temperature
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return density_3d * weights / weights.sum(axis=1, keepdims=True)

    return intensity


def intensity_with_rare_type(
    gt: GroundTruthField, density_3d: float, prevalence: float = RARE_PREVALENCE
) -> IntensityFn:
    """The fixture's composition plus an injected type at exactly ``prevalence``.

    The fixture's own rarest type sits at 6.3%, not the ~2% its docstring claims (pinned by
    ``test_fixture_rarest_type_is_six_percent``), so T05's "a type at 2% prevalence" has to
    be built rather than found. The injected type is a horizontal stripe — its share varies
    smoothly from 0.2% to 3.8% across the section — i.e. a genuinely *interspersed* rare
    niche, which is the case Potts smoothing threatens.
    """
    base = fixture_intensity(gt, density_3d)
    period = 500.0  # micrometres: exactly two stripes across the fixture's 1000 um section

    def intensity(xyz: FloatArray) -> FloatArray:
        points = np.asarray(xyz, dtype=np.float64)
        share = prevalence * (1.0 + 0.9 * np.sin(2.0 * np.pi * points[:, 1] / period))
        common = base(points) * (1.0 - share)[:, None]
        return np.concatenate([common, (density_3d * share)[:, None]], axis=1)

    return intensity


def section_frame(section: Section, thickness: float | None = None) -> Plane:
    """The plane of a real section, optionally with an overridden thickness."""
    plane = section_plane(section)
    if thickness is None:
        return plane
    return plane_from_normal(plane.normal, plane.origin, plane.half_extent, thickness)


def target_plane(training: TrainingVolume) -> Plane:
    """A plane halfway between two training sections: what generation actually asks for."""
    first, second = training.sections[0], training.sections[1]
    plane = section_plane(first)
    origin = np.array([plane.origin[0], plane.origin[1], 0.5 * (first.z + second.z)])
    return plane_from_normal(plane.normal, origin, plane.half_extent, first.thickness)


def min_pair_distance(coords: np.ndarray) -> float:
    """Smallest distance between any two of ``(N, 2)`` points."""
    return float(cKDTree(coords).query(coords, k=2)[0][:, 1].min())


def _sinkhorn(a: np.ndarray, b: np.ndarray, scale: float, eps: float = 0.05, n_iter: int = 200):
    """Entropic-OT transport cost, transcribed from bench3/evaluate_paper.py.

    T10 vendors the real metric, content-hash pinned; this is a local copy so T05 can check
    its definition of done without depending on the benchmark harness.
    """
    C = cdist(a, b, metric="sqeuclidean") / scale
    K = np.exp(-C / eps) + 1e-300
    n, m = C.shape
    weights_a = np.full(n, 1.0 / n)
    weights_b = np.full(m, 1.0 / m)
    u = weights_a / (K @ np.ones(m))
    v = np.ones(m)
    for _ in range(n_iter):
        u = weights_a / (K @ v)
        v = weights_b / (K.T @ u)
    return float(((u[:, None] * K * v[None, :]) * C).sum())


def _sinkhorn_divergence(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    """Debiased Sinkhorn divergence (0 for identical distributions)."""
    return max(
        _sinkhorn(a, b, scale) - 0.5 * (_sinkhorn(a, a, scale) + _sinkhorn(b, b, scale)), 0.0
    )


def celltype_localization(
    pred_xy: np.ndarray,
    pred_types: np.ndarray,
    gt_xy: np.ndarray,
    gt_types: np.ndarray,
    seed: int,
    min_gt_cells: int = 20,
    min_pred_cells: int = 5,
    max_n: int = 250,
) -> float:
    """``paper_celltype_localization``: is each type in the right part of the tissue?

    Transcribed from ``benchmark-pbya-v3/src/bench3/evaluate_paper.py`` (T10 vendors it
    verbatim and pins its hash). 1 = the type's spatial distribution is reproduced,
    0 = no better than scattering it at random.
    """
    rng = np.random.default_rng(seed)
    centre = gt_xy.mean(axis=0)
    radius = float(np.sqrt(((gt_xy - centre) ** 2).sum(axis=1)).mean())
    g_all = (gt_xy - centre) / radius
    p_all = (pred_xy - centre) / radius
    ref = g_all[rng.choice(len(g_all), min(len(g_all), 400), replace=False)]
    scale = float(np.median(cdist(ref, ref, metric="sqeuclidean"))) or 1.0

    def sub(m: np.ndarray) -> np.ndarray:
        return m[rng.choice(len(m), max_n, replace=False)] if len(m) > max_n else m

    scores, weights = [], []
    for t in np.unique(gt_types):
        g_mask = gt_types == t
        if g_mask.sum() < min_gt_cells:
            continue
        w = float(g_mask.mean())
        gc = sub(g_all[g_mask])
        null_draw = sub(g_all[rng.choice(len(g_all), int(g_mask.sum()), replace=False)])
        d_null = _sinkhorn_divergence(null_draw, gc, scale)
        if d_null < 1e-4:
            continue
        p_mask = pred_types == t
        if p_mask.sum() < min_pred_cells:
            scores.append(0.0)
            weights.append(w)
            continue
        d_obs = _sinkhorn_divergence(sub(p_all[p_mask]), gc, scale)
        scores.append(float(np.clip(1.0 - d_obs / d_null, 0.0, 1.0)))
        weights.append(w)
    return float(np.average(scores, weights=np.asarray(weights)))


# --------------------------------------------------------------------------------------
# 1. intensity field and the Poisson NLL
# --------------------------------------------------------------------------------------


def test_poisson_nll_matches_reference():
    """The NLL is the sum of -log lambda at the cells plus the slab-volume integral."""
    cfg = Config()
    gen = torch.Generator().manual_seed(SEED)
    lam_cells = torch.rand((17, 3), generator=gen) + 0.1
    codes = torch.randint(0, 3, (17,), generator=gen)
    lam_mc = torch.rand((64, 3), generator=gen) + 0.1
    volume = 1234.5

    expected = (
        -torch.log(lam_cells[torch.arange(17), codes] + cfg.layout_intensity_eps).sum()
        + volume * lam_mc.sum(dim=1).mean()
    )
    got = layout_poisson_nll(lam_cells, codes, lam_mc, volume, cfg)
    assert torch.allclose(got, expected, atol=1e-5)

    # The integral is over the slab *volume*: doubling the thickness doubles it.
    doubled = layout_poisson_nll(lam_cells, codes, lam_mc, 2.0 * volume, cfg)
    assert math.isclose(
        float(doubled - got), float(volume * lam_mc.sum(dim=1).mean()), rel_tol=1e-5
    )


KNOWN_BASE = np.array([1.1e-4, 1.1e-4])
"""Baseline intensity per type of the known-lambda fixture, cells per micrometre^3."""

KNOWN_EXTENT = 1000.0
KNOWN_THICKNESS = 25.0
KNOWN_SPACING = 100.0
KNOWN_SECTIONS = 3


def _known_intensity() -> IntensityFn:
    """A smooth, strictly positive, strongly modulated intensity with a known shape.

    ``exp(0.9 sin + 0.7 cos)`` varies by a factor of five across the section. The
    modulation has to be strong for the test to mean anything: the Poisson log-likelihood
    of a *weakly* modulated intensity is only hundredths of a nat per cell better than a
    constant one, which is below the Monte-Carlo noise of the integral term, so the fit
    could not be expected to resolve it in any number of steps.
    """

    def intensity(xyz: FloatArray) -> FloatArray:
        p = np.asarray(xyz, dtype=np.float64)
        x = 2.0 * np.pi * p[:, 0] / KNOWN_EXTENT
        y = 2.0 * np.pi * p[:, 1] / KNOWN_EXTENT
        shape = np.stack(
            [0.9 * np.sin(x) + 0.7 * np.cos(y), 0.9 * np.cos(x) + 0.7 * np.sin(y)], axis=1
        )
        return KNOWN_BASE[None, :] * np.exp(shape)

    return intensity


def test_poisson_nll_recovers_intensity():
    """Fitting the Poisson NLL on a pattern drawn from a known lambda recovers it: r > 0.9.

    ``fourier_bands_xy`` is lowered to 2 (one and two cycles across the section — the scale
    this intensity actually varies on). At the default 8 the head resolves structure down to
    a hundredth of the section and the Poisson MLE fits the point pattern's own sampling
    noise: the NLL keeps falling while the correlation with the truth decays from 0.97 to
    0.28. Capacity control, not a tuning knob — see ``fit_intensity_head`` and
    SPEC_QUESTIONS B10.
    """
    cfg = Config().replace(
        repulsion=False,
        potts_iters=1,
        layout_n_mc=1024,
        fourier_bands_xy=2,
        field_dim=8,
        layout_mlp_hidden=64,
    )
    truth = _known_intensity()

    sections = []
    for i in range(KNOWN_SECTIONS):
        plane = plane_from_normal(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, KNOWN_SPACING * i]),
            np.full(2, 0.5 * KNOWN_EXTENT),
            KNOWN_THICKNESS,
        )
        layout = sample_layout(truth, plane, cfg, seed=SEED + i)
        sections.append(
            Section(
                z=float(plane.origin[2]),
                thickness=KNOWN_THICKNESS,
                # coords_xyz, not coords_uv: the plane's own frame is a rotation of the
                # data frame, and the truth above is written in the data frame.
                coords=layout.coords_xyz[:, :2].astype(np.float32),
                cell_type=layout.cell_type,
                region=None,
                counts=sparse.csr_matrix((layout.n_cells, 1), dtype=np.float32),
                section_id=f"known_{i}",
            )
        )

    half = 0.5 * KNOWN_EXTENT
    bbox = np.array(
        [
            [-half, -half, -KNOWN_THICKNESS],
            [half, half, KNOWN_SPACING * (KNOWN_SECTIONS - 1) + KNOWN_THICKNESS],
        ],
        dtype=np.float32,
    )
    head = IntensityHead(
        cfg, n_types=2, bbox=bbox, init_density=mean_cell_density(sections), n_regions=None
    )

    def field_fn(points: torch.Tensor) -> torch.Tensor:
        return torch.zeros((points.shape[0], cfg.field_dim), dtype=torch.float32)

    history = fit_intensity_head(head, sections, field_fn, cfg, seed=SEED)
    assert history[-1] < history[0], (history[0], history[-1])

    side = np.linspace(-0.45 * KNOWN_EXTENT, 0.45 * KNOWN_EXTENT, 24)
    gx, gy = np.meshgrid(side, side, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, KNOWN_SPACING)], axis=1)
    with torch.no_grad():
        query = torch.from_numpy(grid.astype(np.float32))
        predicted = head(query, field_fn(query)).numpy().astype(np.float64)
    actual = truth(grid)

    r_total = float(np.corrcoef(predicted.sum(axis=1), actual.sum(axis=1))[0, 1])
    per_type = [float(np.corrcoef(predicted[:, c], actual[:, c])[0, 1]) for c in range(2)]
    assert r_total > 0.9, r_total
    assert min(per_type) > 0.9, per_type


# --------------------------------------------------------------------------------------
# 2. positions
# --------------------------------------------------------------------------------------


def test_expected_count_matches(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """Sampled N over 50 seeds has mean within 5% of N_expected."""
    density = mean_cell_density(training.sections)
    intensity = fixture_intensity(gt_field, density)
    plane = plane_from_normal(
        np.array([0.0, 0.0, 1.0]),
        np.array([500.0, 500.0, 25.0]),
        np.full(2, 300.0),
        training.sections[0].thickness,
    )

    counts = []
    for i in range(50):
        layout = sample_layout(intensity, plane, cfg, seed=SEED + i, repulsion=repulsion)
        assert not layout.budget_exhausted
        counts.append(layout.n_cells)
    n_expected = sample_layout(intensity, plane, cfg, seed=SEED, repulsion=repulsion).n_expected

    relative = abs(float(np.mean(counts)) - n_expected) / n_expected
    assert relative < 0.05, (float(np.mean(counts)), n_expected, relative)

    # ... and the expected count itself is the slab-volume integral, so it scales with
    # thickness. This is the property T07's L_thick rests on.
    thick = plane_from_normal(plane.normal, plane.origin, plane.half_extent, 2.0 * plane.thickness)
    thick_expected = sample_layout(intensity, thick, cfg, seed=SEED, repulsion=repulsion).n_expected
    assert math.isclose(thick_expected, 2.0 * n_expected, rel_tol=0.05)


def test_hardcore_respected(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """No pair of sampled points is closer than r0."""
    density = mean_cell_density(training.sections)
    layout = sample_layout(
        fixture_intensity(gt_field, density),
        target_plane(training),
        cfg,
        seed=SEED,
        repulsion=repulsion,
    )
    assert not layout.budget_exhausted
    assert min_pair_distance(layout.coords_uv) >= repulsion.r0

    # The control: with the interaction switched off (ablation A4) the same intensity
    # produces pairs inside the hard core, so the assertion above is not vacuous.
    poisson = sample_layout(
        fixture_intensity(gt_field, density),
        target_plane(training),
        cfg.replace(repulsion=False),
        seed=SEED,
    )
    assert min_pair_distance(poisson.coords_uv) < repulsion.r0


PCF_SIM_REPLICATES = 3
"""Layouts averaged into the simulated ``g(r)``.

One 1500-point realisation estimates ``g(r)`` to about +-0.08 per bin, so comparing a
single simulated realisation with a single real section at a 0.15 threshold would be
measuring the *estimator's* noise as much as the process's. Both sides here therefore
estimate the process: the real curve is averaged over all training sections, the simulated
one over this many seeds.
"""


def _mean_pcf(
    coord_sets: list[tuple[np.ndarray, np.ndarray]], cfg: Config, r_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ``g(r)`` over several ``(coords, window)`` pairs, on shared bins."""
    curves = []
    centres = np.zeros(int(cfg.pcf_n_bins))
    for coords, window in coord_sets:
        centres, g = pair_correlation(coords, cfg, r_max=r_max, window=window)
        curves.append(g)
    return centres, np.mean(np.stack(curves, axis=0), axis=0)


def _pcf_gap(
    simulated: list[tuple[np.ndarray, np.ndarray]],
    real: list[tuple[np.ndarray, np.ndarray]],
    cfg: Config,
    rep: RepulsionParams,
    r_max: float,
    *,
    r_lo: float,
) -> float:
    """max |g_sim(r) - g_real(r)| over r in [r_lo, 3R], on shared bins."""
    centres, g_real = _mean_pcf(real, cfg, r_max)
    _, g_sim = _mean_pcf(simulated, cfg, r_max)
    mask = (centres >= r_lo) & (centres <= 3.0 * rep.R)
    assert mask.sum() >= 3, centres[mask]
    return float(np.max(np.abs(g_sim[mask] - g_real[mask])))


def _windowed(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pair coordinates with their own bounding-box window."""
    return coords, np.stack([coords.min(axis=0), coords.max(axis=0)], axis=0)


def test_pcf_matches_real(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """Simulated g(r) matches the real one to < 0.15 — and a pure-Poisson sampler does not.

    Measured over two ranges, and the pair is the point:

    * **the spec's ``[r0, 3R]``.** The generated process matches the tissue there — and so
      does a pure-Poisson one. That is not a pass for Poisson, it is a **defect in the
      range**: a hard-core process differs from Poisson only *inside* the correlation hole,
      and the hole ends at about ``r0`` by construction (``r0`` is a low percentile of the
      nearest-neighbour distances). The spec's range therefore starts exactly where the
      discriminating signal stops. Asserted here as a fact, so it cannot change unnoticed
      (SPEC_QUESTIONS B12).
    * **``[0, 3R]``**, the same statistic over the range that contains the hole. This is
      strictly harder than the spec's criterion — it is a superset of its range — the field
      sampler passes it, and the pure-Poisson control fails it by a factor of six. That is
      the assertion that makes the repulsion load-bearing (ablation A4).
    """
    density = mean_cell_density(training.sections)
    intensity = fixture_intensity(gt_field, density)
    plane = target_plane(training)
    real = [_windowed(np.asarray(s.coords, dtype=np.float64)) for s in training.sections]
    r_max = float(cfg.pcf_max_r_factor) * training.median_nn_dist

    def simulate(config: Config) -> list[tuple[np.ndarray, np.ndarray]]:
        out = []
        for i in range(PCF_SIM_REPLICATES):
            layout = sample_layout(intensity, plane, config, seed=SEED + i, repulsion=repulsion)
            assert not layout.budget_exhausted
            out.append(_windowed(np.asarray(layout.coords_uv, dtype=np.float64)))
        return out

    field = simulate(cfg)
    poisson = simulate(cfg.replace(repulsion=False))

    gap_spec = _pcf_gap(field, real, cfg, repulsion, r_max, r_lo=repulsion.r0)
    gap_full = _pcf_gap(field, real, cfg, repulsion, r_max, r_lo=0.0)
    assert gap_spec < 0.15, gap_spec
    assert gap_full < 0.15, gap_full

    # NEGATIVE CONTROL (ablation A4), on the range that contains the correlation hole.
    poisson_full = _pcf_gap(poisson, real, cfg, repulsion, r_max, r_lo=0.0)
    assert poisson_full >= 0.15, poisson_full
    assert poisson_full > 4.0 * gap_full, (poisson_full, gap_full)

    # ... and the pinned defect: over the spec's own range the control does NOT fail.
    poisson_spec = _pcf_gap(poisson, real, cfg, repulsion, r_max, r_lo=repulsion.r0)
    assert poisson_spec < 0.15, poisson_spec


# --------------------------------------------------------------------------------------
# 3. marks
# --------------------------------------------------------------------------------------


def _marked_sections(
    training: TrainingVolume, intensity: IntensityFn, seed: int
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Per training section: coords, an independent mark draw, log lambda, and P(type).

    The sampler's own initial state, on the real cell positions: what the Potts step is
    handed in the pipeline and what ``fit_potts_beta`` fits against.
    """
    gen = np.random.default_rng(seed)
    out = []
    for section in training.sections:
        coords = np.asarray(section.coords, dtype=np.float64)
        lam = intensity(to_xyz(section).astype(np.float64))
        probabilities = lam / lam.sum(axis=1, keepdims=True)
        cumulative = np.cumsum(probabilities, axis=1)
        u = gen.random(coords.shape[0])
        marks = (u[:, None] > cumulative).sum(axis=1).clip(0, lam.shape[1] - 1).astype(np.int32)
        out.append((coords, marks, np.log(lam), probabilities))
    return out


def _pooled_purity(
    marked: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cfg: Config,
    beta: float | None = None,
    seed: int = SEED,
) -> float:
    """Mean neighbourhood purity over every section, before (beta=None) or after smoothing."""
    values = []
    for i, (coords, marks, log_lam, _) in enumerate(marked):
        types = (
            marks
            if beta is None
            else potts_smooth(
                coords,
                marks,
                log_lam,
                cfg,
                generator=np.random.default_rng(seed + i),
                beta=beta,
            )
        )
        values.append(type_purity(coords, types, cfg))
    return float(np.mean(np.concatenate(values)))


def test_potts_improves_purity(training: TrainingVolume, gt_field, cfg: Config, beta: float):
    """Purity after smoothing is closer to the real sections' than before, and not above it."""
    real = float(
        np.mean(
            np.concatenate([type_purity(s.coords, s.cell_type, cfg) for s in training.sections])
        )
    )
    marked = _marked_sections(
        training, fixture_intensity(gt_field, mean_cell_density(training.sections)), SEED
    )
    before = _pooled_purity(marked, cfg)
    after = _pooled_purity(marked, cfg, beta=beta, seed=SEED + 1000)

    assert beta > 0.0, "the fitted coupling must actually do something on this fixture"
    assert abs(after - real) < abs(before - real), (before, after, real)
    # Over-smoothing check: the fitted coupling must not push purity past the tissue's. The
    # tolerance is three standard errors of the real mean over the pooled training cells.
    purities = np.concatenate([type_purity(s.coords, s.cell_type, cfg) for s in training.sections])
    tolerance = float(np.std(purities)) / math.sqrt(purities.size)
    assert after <= real + 3.0 * tolerance, (after, real, tolerance)


def _rare_retention(
    marked: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    cfg: Config,
    beta: float,
    seed: int,
) -> float:
    """Share of the injected rare type's expected count that survives the smoothing."""
    kept = 0.0
    expected = 0.0
    for i, (coords, marks, log_lam, probabilities) in enumerate(marked):
        rare = log_lam.shape[1] - 1  # the injected type is the last column
        smoothed = potts_smooth(
            coords, marks, log_lam, cfg, generator=np.random.default_rng(seed + i), beta=beta
        )
        kept += float(np.count_nonzero(smoothed == rare))
        expected += float(probabilities[:, rare].sum())
    return kept / expected


def test_rare_types_survive(training: TrainingVolume, gt_field, cfg: Config):
    """A 2%-prevalence type keeps >= 50% of its expected count — and over-smoothing does not."""
    intensity = intensity_with_rare_type(gt_field, mean_cell_density(training.sections))
    rare_beta = fit_potts_beta(training, intensity, cfg, seed=SEED)
    marked = _marked_sections(training, intensity, SEED)

    expected = sum(float(p[:, -1].sum()) for *_, p in marked)
    total = sum(len(m) for _, m, *_ in marked)
    assert abs(expected / total - RARE_PREVALENCE) < 0.002, expected / total

    retained = _rare_retention(marked, cfg, rare_beta, seed=SEED + 2000)
    assert retained >= 0.5, (retained, rare_beta)

    # NEGATIVE CONTROL. The criterion has to be able to fail: a coupling far above the
    # fitted one must erase the rare type. Without this, "the rare type survived" would be
    # a statement about the fixture rather than about the fitted beta.
    crushed = _rare_retention(marked, cfg.replace(potts_iters=8), cfg.potts_beta_max, SEED + 2000)
    assert crushed < 0.5, (crushed, retained)
    assert crushed < retained, (crushed, retained)


def test_icm_erases_rare_types(training: TrainingVolume, gt_field, cfg: Config):
    """T05 says ICM; ICM's mode-seeking sweep wipes out a 2% type before beta does anything.

    The measurement behind ``Config.potts_update``'s default of ``gibbs`` (SPEC_QUESTIONS
    B11), asserted rather than described: with ``potts_update="icm"`` at the **smallest
    coupling the fit can choose** (``Config.potts_beta_min``) the rare type is already gone
    and purity has overshot the tissue's, so there is no beta left to fit and T05's own
    "Do NOT" is violated by the update rule rather than by the coupling.
    """
    intensity = intensity_with_rare_type(gt_field, mean_cell_density(training.sections))
    marked = _marked_sections(training, intensity, SEED)
    icm = cfg.replace(potts_update="icm")

    real = float(
        np.mean(
            np.concatenate([type_purity(s.coords, s.cell_type, cfg) for s in training.sections])
        )
    )
    smallest = cfg.potts_beta_min
    assert _rare_retention(marked, icm, smallest, SEED) < 0.05
    assert _pooled_purity(marked, icm, beta=smallest) > real
    # Gibbs, at the same coupling, leaves the rare type essentially intact.
    assert _rare_retention(marked, cfg, smallest, SEED) > 0.5


def test_fitted_beta_is_not_hand_set(training: TrainingVolume, gt_field, cfg: Config, beta: float):
    """fit_potts_beta returns a grid value fitted from the data, and the fit is reproducible."""
    grid = potts_beta_grid(cfg)
    assert float(np.min(np.abs(grid - beta))) < 1e-12, beta
    intensity = fixture_intensity(gt_field, mean_cell_density(training.sections))
    assert fit_potts_beta(training, intensity, cfg, seed=SEED) == beta


# --------------------------------------------------------------------------------------
# 4. determinism, modes, leakage
# --------------------------------------------------------------------------------------


def test_layout_deterministic(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """Same seed -> identical layout; a different seed -> a different one."""
    density = mean_cell_density(training.sections)
    intensity = fixture_intensity(gt_field, density)
    plane = target_plane(training)
    first = sample_layout(intensity, plane, cfg, seed=SEED, repulsion=repulsion)
    again = sample_layout(intensity, plane, cfg, seed=SEED, repulsion=repulsion)
    other = sample_layout(intensity, plane, cfg, seed=SEED + 1, repulsion=repulsion)

    assert np.array_equal(first.coords_uv, again.coords_uv)
    assert np.array_equal(first.coords_xyz, again.coords_xyz)
    assert np.array_equal(first.cell_type, again.cell_type)
    assert first.n_expected == again.n_expected
    assert not np.array_equal(first.coords_uv, other.coords_uv)

    # The fits are stochastic too (Convention 3): same seed, same parameters.
    assert fit_repulsion(training, cfg, seed=SEED) == repulsion


def test_all_three_modes_run(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """field, hybrid and resample each produce a valid Layout with a plausible N."""
    density = mean_cell_density(training.sections)
    intensity = fixture_intensity(gt_field, density)
    plane = target_plane(training)
    flanking = [flanking_from_section(s, plane) for s in training.sections[:2]]

    layouts = {
        mode: sample_layout(
            intensity,
            plane,
            cfg.replace(layout_mode=mode),
            seed=SEED,
            repulsion=repulsion,
            flanking=flanking,
        )
        for mode in ("field", "hybrid", "resample")
    }
    for mode, layout in layouts.items():
        assert layout.mode == mode
        assert layout.n_cells > 0
        assert layout.coords_uv.shape == (layout.n_cells, 2)
        assert layout.coords_xyz.shape == (layout.n_cells, 3)
        assert layout.cell_type.shape == (layout.n_cells,)
        assert layout.cell_type.min() >= 0
        assert layout.cell_type.max() < len(training.celltype_names)
        assert np.allclose(layout.coords_xyz[:, 2], plane.origin[2], atol=1e-3)

    for mode in ("field", "hybrid"):
        layout = layouts[mode]
        # Poisson: |N - E N| within 5 standard deviations.
        assert abs(layout.n_cells - layout.n_expected) < 5.0 * math.sqrt(layout.n_expected)
    # resample reuses a real section outright, so its count is that section's.
    assert layouts["resample"].n_cells == flanking[0].coords_uv.shape[0]

    # The hybrid polish moves the points toward the flanking marginals without discarding
    # the hard core the field sampler imposed.
    assert not np.array_equal(layouts["hybrid"].coords_uv, layouts["field"].coords_uv)


def test_modes_refuse_missing_evidence(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """hybrid/resample without flanking sections, and field without a fitted interaction, raise."""
    intensity = fixture_intensity(gt_field, mean_cell_density(training.sections))
    plane = target_plane(training)
    for mode in ("hybrid", "resample"):
        with pytest.raises(LayoutError, match="flanking"):
            sample_layout(
                intensity, plane, cfg.replace(layout_mode=mode), SEED, repulsion=repulsion
            )
    with pytest.raises(LayoutError, match="fit_repulsion"):
        sample_layout(intensity, plane, cfg, SEED)


def test_fits_refuse_a_full_volume(volume: Volume, cfg: Config):
    """The interaction and the coupling are fitted on training sections only, by type."""
    with pytest.raises(TypeError, match="TrainingVolume"):
        fit_repulsion(volume, cfg, seed=SEED)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TrainingVolume"):
        fit_potts_beta(volume, lambda p: np.ones((p.shape[0], 2)), cfg, seed=SEED)  # type: ignore[arg-type]


def test_repulsion_records_its_percentile(training: TrainingVolume, cfg: Config, repulsion):
    """Which percentile produced r0 is recorded: it changes a published number (B6)."""
    assert repulsion.r0_percentile == cfg.repulsion_r0_percentile == 1.0
    assert 0.0 < repulsion.r0 < repulsion.R
    assert 0.0 < repulsion.gamma <= 1.0
    alternative = fit_repulsion(training, cfg.replace(repulsion_r0_percentile=5.0), seed=SEED)
    assert alternative.r0_percentile == 5.0
    assert alternative.r0 > repulsion.r0


def test_budget_exhaustion_warns(training: TrainingVolume, gt_field, cfg: Config):
    """An r0 too large for the density warns and reports the truncation (SPEC_QUESTIONS B7)."""
    density = mean_cell_density(training.sections)
    intensity = fixture_intensity(gt_field, density)
    plane = target_plane(training)
    impossible = RepulsionParams(r0=60.0, R=70.0, gamma=0.5, r0_percentile=1.0, density=density)
    with pytest.warns(ProposalBudgetWarning, match="too large for the requested density"):
        layout = sample_layout(intensity, plane, cfg, seed=SEED, repulsion=impossible)
    assert layout.budget_exhausted
    assert layout.n_cells < layout.n_expected
    assert min_pair_distance(layout.coords_uv) >= impossible.r0


# --------------------------------------------------------------------------------------
# 5. definition of done
# --------------------------------------------------------------------------------------


def _localization_arms(
    training: TrainingVolume, held_out, gt_field, cfg: Config, repulsion, beta: float
) -> dict[str, list[float]]:
    """Cell-type localization on **every** held-out section, four arms each.

    ``self``
        The held-out section scored against itself: the metric's ceiling *under its own
        subsampling*, which is not 1 and which varies by section, so it is measured rather
        than assumed.
    ``generated``
        ``field`` mode, from the smoothed stand-in intensity — the layout head as the
        pipeline will run it.
    ``ideal``
        ``field`` mode from the fixture's **own generative composition** (temperature
        ``FIXTURE_TYPE_TEMPERATURE``, at which the fitted coupling is 0): an independent draw
        from the law that produced the data. What a *perfect* intensity head would score.
    ``flanking``
        The nearest real training section, i.e. what ``layout_mode="resample"`` — the
        previous version's behaviour — achieves on the same held-out section.
    """
    density = mean_cell_density(training.sections)
    smoothed = fixture_intensity(gt_field, density)
    truth = fixture_intensity(gt_field, density, FIXTURE_TYPE_TEMPERATURE)
    arms: dict[str, list[float]] = {"self": [], "generated": [], "ideal": [], "flanking": []}
    for held in held_out:
        plane = section_frame(held)
        gt_xy = np.asarray(held.coords, dtype=np.float64)
        gt_types = np.asarray(held.cell_type)

        def score(xy: np.ndarray, types: np.ndarray, gt_xy=gt_xy, gt_types=gt_types) -> float:
            return celltype_localization(xy, types, gt_xy, gt_types, seed=SEED)

        # coords_xyz, not coords_uv: the plane's (u, v) frame is a rotation of the data
        # frame, and the held-out section's coordinates are in the data frame.
        for name, intensity, coupling in (
            ("generated", smoothed, beta),
            ("ideal", truth, 0.0),
        ):
            layout = sample_layout(
                intensity, plane, cfg.replace(potts_beta=coupling), seed=SEED, repulsion=repulsion
            )
            arms[name].append(
                score(
                    np.asarray(layout.coords_xyz[:, :2], dtype=np.float64),
                    np.asarray(layout.cell_type),
                )
            )
        neighbour = min(training.sections, key=lambda s: abs(s.z - held.z))
        arms["flanking"].append(
            score(np.asarray(neighbour.coords, dtype=np.float64), np.asarray(neighbour.cell_type))
        )
        arms["self"].append(score(gt_xy, gt_types))
    return arms


@pytest.fixture(scope="module")
def localization(training: TrainingVolume, held_out, gt_field, cfg: Config, repulsion, beta: float):
    """The four localization arms, measured once."""
    return _localization_arms(training, held_out, gt_field, cfg, repulsion, beta)


def test_localization_beats_the_real_data_baseline(localization):
    """field mode localizes cell types better than the real flanking section does.

    T05's definition of done is "within 10% of the real section's value", and the phrase has
    two readings. This is the **no-regression** one: the real value available on real data is
    what a real neighbouring section achieves, which is exactly what ``layout_mode="resample"``
    (the previous version) produces. Measured per held-out section, generated / flanking =
    1.654 / 1.154 / 1.246, i.e. the layout head beats it everywhere rather than coming within
    10% of it.

    The other reading — the held-out section's *own* score — is
    ``test_localization_within_10_percent_of_heldout_self_score`` below, and it fails.
    """
    for generated, flanking in zip(
        localization["generated"], localization["flanking"], strict=True
    ):
        assert generated >= 0.9 * flanking, (generated, flanking)


def test_localization_matches_an_ideal_intensity(localization):
    """The layout head scores what an independent draw from the true law scores.

    The ``ideal`` arm samples positions and marks from the fixture's **own** generative
    composition, so it is a draw from the process that produced the held-out section — the
    best any intensity head could do. The layout head running on a *smoothed* stand-in
    intensity reaches **99.4%** of it (0.7128 against 0.7178, averaged over the three
    held-out sections). That is what says the remaining gap to the ceiling is the metric
    penalising realisation noise, not the sampler losing localization.
    """
    generated = float(np.mean(localization["generated"]))
    ideal = float(np.mean(localization["ideal"]))
    assert generated >= 0.9 * ideal, (generated, ideal)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "T05's definition of done read as 'within 10% of the held-out section's own score' "
        "FAILS: 0.776 of it on average (0.909 / 0.613 / 0.806 per section), and an "
        "independent draw from the fixture's true generative law reaches only 0.779 — so the "
        "criterion asks the layout head to beat the process that produced the data. An "
        "amendment is proposed in specs/05; the spec's owner decides. xfail is strict, so if "
        "this ever starts passing the suite fails and the record has to be updated."
    ),
)
def test_localization_within_10_percent_of_heldout_self_score(localization):
    """The strict reading of the definition of done, pinned as a known failure."""
    ratios = [
        generated / own
        for generated, own in zip(localization["generated"], localization["self"], strict=True)
    ]
    assert float(np.mean(ratios)) >= 0.9, ratios


def test_pair_correlation_is_one_for_poisson():
    """The g(r) estimator reads 1 for complete spatial randomness (its own calibration)."""
    cfg = Config()
    gen = np.random.default_rng(SEED)
    coords = gen.uniform(0.0, 1000.0, size=(4000, 2))
    window = np.array([[0.0, 0.0], [1000.0, 1000.0]])
    _, g = pair_correlation(coords, cfg, r_max=100.0, window=window)
    assert np.max(np.abs(g - 1.0)) < 0.15, g


def test_expected_count_is_a_volume_integral():
    """expected_count is slab_volume * mean total intensity, with no area shortcut."""
    lam = np.full((128, 3), 2.0)
    assert math.isclose(expected_count(lam, 10.0), 60.0)


def test_layout_module_needs_no_network(training: TrainingVolume, gt_field, cfg: Config, repulsion):
    """Sanity: a whole layout is produced from the synthetic fixture alone (Convention 7)."""
    layout = sample_layout(
        fixture_intensity(gt_field, mean_cell_density(training.sections)),
        target_plane(training),
        cfg,
        seed=SEED,
        repulsion=repulsion,
    )
    assert layout.repulsion is repulsion
    assert layout.seed == SEED
    assert layout.plane is not None


def test_fixture_rarest_type_is_six_percent():
    """T01's fixture claims a ~2% type; it realises 6.3%. Pinned, because T05 needs 2%.

    ``tests/fixtures/synthetic.py`` documents "one rare (~2%) type, which T05's
    ``test_rare_types_survive`` needs" — but ``type_bias = linspace(0.6, -2.4, 6)`` pushed
    through the Gumbel argmax realises 6.3% for the rarest of the six. That is not a bug in
    the fixture (a 6% type is a perfectly good rare type), but it does mean the spec's 2%
    criterion cannot be measured on the fixture's own labels, so
    ``intensity_with_rare_type`` injects a genuine 2% type on top of the fixture's field.
    This test pins the discrepancy so the injection cannot be quietly dropped later.
    """
    vol, _ = make_synthetic_volume(seed=0)
    counts = np.bincount(
        np.concatenate([s.cell_type for s in vol.sections]), minlength=len(vol.celltype_names)
    )
    prevalence = counts / counts.sum()
    assert 0.05 < prevalence.min() < 0.08, prevalence.tolist()
