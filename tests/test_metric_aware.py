"""T08 acceptance tests for the metric-aware losses and the internal LOSO loop.

Every numeric threshold here is ``specs/08``'s: agreement with an independent reference
implementation of Moran's I and Geary's C to **1e-5**, a soft profile within **2%** of hard
binning at ``sigma = 0.1 x`` the bin width, an identical principal axis across epochs, a
``TypeError`` when held-out data is passed, and — the two that carry the task's claims — the
on/off comparison on three metric families and the **R4** covariance criterion.

The R4 test is the reason this task exists. T06 measured a model that gets better at its own
likelihood while its generated section's gene-gene correlation gets worse (Frobenius error
**9.316** against an independent-donor baseline's **7.783** on the default holdout, **17.7**
against **11.3** at ``consecutive-3``), and had no term that could stop it.
``test_metric_losses_close_the_covariance_loss`` asks whether these terms do, on **both**
holdout regimes — both, because the default regime alone is what let T06's withdrawn
decomposition look like a win. The verdict either way is recorded in ``PROGRESS.md``.

Cost. Everything that runs an optimiser loop is ``slow`` (SPEC_QUESTIONS C6); the fast tests
here are the six structural ones plus one reconstruction of an untrained model.

Why this module imports from ``tests.test_expression``. The R4 numbers must be comparable with
T06's, and "comparable" means computed by the *same* code: ``gene_correlation``,
``informative_genes``, ``frobenius_offdiag``, ``donor_baseline`` and the reduced training
config all come from there rather than being restated here, where they could drift by a
normalisation and turn a regression into a headline.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import HeldOutSections, Section, TrainingVolume, Volume, to_xyz
from spatialcpav25_gen.losses.metric_aware import (
    MetricDominanceWarning,
    _type_grid_size,
    gearys_c,
    knn_weight_graph,
    loss_autocorr,
    loss_distribution,
    loss_profile,
    marker_genes,
    morans_i,
    normalised_linear,
    profile_axis,
    soft_depth_profile,
    soft_field_profile,
)
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData
from spatialcpav25_gen.train.loso import LOSOScheduler, metric_aware_terms, reconstruct_hidden

from tests.fixtures.synthetic import GroundTruthField, make_synthetic_volume
from tests.test_expression import (
    build_embeddings,
    donor_baseline,
    expression_cfg,
    frobenius_offdiag,
    gene_correlation,
    generate_counts,
    informative_genes,
    train_model,
)

SEED = 20260818

METRIC_OFF: dict[str, object] = {"w_autocorr": 0.0, "w_profile": 0.0, "w_distribution": 0.0}
"""The arm without T08's terms — T06's model exactly, whatever the shipped defaults become."""

METRIC_ON: dict[str, object] = {"w_autocorr": 0.5, "w_profile": 0.5, "w_distribution": 0.5}
"""``specs/08``'s weights, stated here rather than read from ``Config`` so that the two arms of
the comparison stay pinned even if the shipped default moves after the measurement."""

TRAIN_STEPS = 1200
"""Optimiser steps per arm.

``specs/08`` says 1000 for the on/off comparison; this is T06's own ``TRAIN_STEPS`` instead, so
that the *off* arm is the model whose numbers are already on the record (9.316 against a 7.783
baseline) and the R4 comparison is against a measurement rather than a re-measurement. The
deviation is recorded in PROGRESS.md."""

REFERENCE_TOL = 1e-5
"""``specs/08``: agreement with a NumPy/esda reference at 1e-5."""

HARD_BINNING_TOL = 0.02
"""``specs/08``: relative error < 2% at ``sigma = 0.1 x`` the bin width."""

SIGMA_SMALL = 0.1
"""...and the ``sigma`` the convergence is measured at, in bin widths."""

DONOR_BASELINE_ALTERNATING = 7.783
"""T06's recorded independent-donor Frobenius error on the default holdout. Recomputed in the
test rather than trusted — this is the number the recomputation must reproduce."""

DONOR_BASELINE_CONSECUTIVE = 11.3
"""...and at ``consecutive-3``, where T06 recorded the model at 17.7."""

CEILING_DRAWS = 3
"""Draws from the fixture's known law used to measure the achievable ceiling. Three is what
``scripts/t06_expression_report.py`` used for the 5.601 / 5.513 on the record; the quantity is
stable to ~0.01 across draws, so it is a floor to compare against rather than a fitted value."""


# --------------------------------------------------------------------------------------
# reference implementations — deliberately naive, deliberately not the module's code
# --------------------------------------------------------------------------------------


def dense_weights(graph: torch.Tensor) -> np.ndarray:
    """The sparse weight matrix as a dense NumPy array."""
    return np.asarray(graph.to_dense().numpy(), dtype=np.float64)


def reference_morans(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Moran's I straight from the definition, dense and doubly summed."""
    n = x.shape[0]
    centred = x - x.mean(axis=0, keepdims=True)
    numerator = np.einsum("ij,ig,jg->g", w, centred, centred)
    return (n / w.sum()) * numerator / (centred**2).sum(axis=0)


def reference_gearys(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Geary's C straight from the definition, forming the full pairwise difference."""
    n = x.shape[0]
    centred = x - x.mean(axis=0, keepdims=True)
    difference = x[:, None, :] - x[None, :, :]
    numerator = (w[:, :, None] * difference**2).sum(axis=(0, 1))
    return ((n - 1) / (2.0 * w.sum())) * numerator / (centred**2).sum(axis=0)


def hard_depth_profile(
    x: np.ndarray, projection: np.ndarray, n_bins: int, bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Plain binning: ``(n_bins, G)`` per-bin means and the ``(n_bins,)`` occupancy mask."""
    low, high = bounds
    width = (high - low) / n_bins
    index = np.clip(((projection - low) / width).astype(int), 0, n_bins - 1)
    out = np.zeros((n_bins, x.shape[1]))
    occupied = np.zeros(n_bins, dtype=bool)
    for b in range(n_bins):
        mask = index == b
        occupied[b] = bool(mask.any())
        if occupied[b]:
            out[b] = x[mask].mean(axis=0)
    return out, occupied


def random_field(n: int, n_genes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` positions and ``(N, G)`` spatially structured values, from ``seed``."""
    gen = np.random.default_rng(seed)
    coords = gen.uniform(0.0, 1000.0, size=(n, 2))
    # Structure, not noise: a flat field gives Moran's I ~ 0 for every implementation and the
    # reference comparison would pass on a function that returned zeros.
    phase = gen.uniform(0.0, 2 * np.pi, size=(n_genes,))
    direction = gen.normal(size=(2, n_genes))
    direction /= np.linalg.norm(direction, axis=0, keepdims=True)
    signal = np.sin(coords @ direction / 150.0 + phase)
    return coords, signal + 0.3 * gen.normal(size=(n, n_genes))


# --------------------------------------------------------------------------------------
# 1. differentiable spatial autocorrelation
# --------------------------------------------------------------------------------------


def test_morans_matches_reference(cfg: Config):
    """Moran's I agrees with an independent dense reference to 1e-5."""
    coords, values = random_field(60, 5, SEED)
    graph = knn_weight_graph(coords, cfg, k=6)
    ours = morans_i(torch.from_numpy(values), graph).numpy()
    theirs = reference_morans(values, dense_weights(graph))
    assert np.abs(ours - theirs).max() < REFERENCE_TOL, (ours, theirs)
    # ...and the statistic is not trivially zero, or the agreement would mean nothing.
    assert np.abs(theirs).min() > 0.05, theirs
    # The production path runs in float32; it has to agree too, just not to 1e-5.
    single = morans_i(torch.from_numpy(values.astype(np.float32)), graph).numpy()
    assert np.abs(single - theirs).max() < 1e-4, (single, theirs)


def test_gearys_matches_reference(cfg: Config):
    """Geary's C agrees with an independent dense reference to 1e-5."""
    coords, values = random_field(60, 5, SEED + 1)
    graph = knn_weight_graph(coords, cfg, k=6)
    ours = gearys_c(torch.from_numpy(values), graph).numpy()
    theirs = reference_gearys(values, dense_weights(graph))
    assert np.abs(ours - theirs).max() < REFERENCE_TOL, (ours, theirs)
    # Geary's C is 1 under no autocorrelation and below it under positive autocorrelation;
    # the fixture is positively autocorrelated, so a value pinned at 1 would be a bug.
    assert theirs.max() < 0.95, theirs
    # The two statistics see the same structure: I high where C is low.
    morans = reference_morans(values, dense_weights(graph))
    assert np.corrcoef(morans, theirs)[0, 1] < -0.8, (morans, theirs)


def test_morans_differentiable(cfg: Config):
    """The gradient of Moran's I w.r.t. the expression is finite and non-zero.

    And the graph carries none of it: ``specs/08`` forbids differentiating through kNN
    construction, which here is structural rather than a rule — ``W`` is built by a KD-tree on
    a detached NumPy array and has no ``grad_fn`` at all.
    """
    coords, values = random_field(80, 4, SEED + 2)
    graph = knn_weight_graph(coords, cfg, k=6)
    assert not graph.requires_grad
    assert graph.grad_fn is None

    x = torch.from_numpy(values.astype(np.float32)).requires_grad_(True)
    morans_i(x, graph).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().max()) > 0.0

    y = torch.from_numpy(values.astype(np.float32)).requires_grad_(True)
    gearys_c(y, graph).sum().backward()
    assert y.grad is not None
    assert torch.isfinite(y.grad).all()
    assert float(y.grad.abs().max()) > 0.0


def test_loss_autocorr_is_zero_on_itself_and_positive_otherwise(cfg: Config):
    """The agreement term bottoms out at agreement, and a shuffled field costs more."""
    coords, values = random_field(120, 8, SEED + 3)
    graph = knn_weight_graph(coords, cfg)
    x = torch.from_numpy(values.astype(np.float32))
    assert float(loss_autocorr(x, x, graph, graph, cfg)) < 1e-6

    shuffled = x[torch.from_numpy(np.random.default_rng(SEED).permutation(x.shape[0]))]
    assert float(loss_autocorr(shuffled, x, graph, graph, cfg)) > 1e-3


# --------------------------------------------------------------------------------------
# 2. soft spatial profiles
# --------------------------------------------------------------------------------------


def test_soft_profile_approaches_hard(cfg: Config):
    """As ``sigma -> 0`` the soft depth profile converges to hard binning (< 2% at 0.1 width)."""
    gen = np.random.default_rng(SEED + 4)
    coords = gen.uniform(0.0, 1000.0, size=(400, 3)).astype(np.float32)
    coords[:, 2] = 0.0
    values = gen.uniform(1.0, 2.0, size=(400, 3)).astype(np.float32)
    axis = torch.tensor([1.0, 0.0, 0.0])
    n_bins = int(cfg.profile_n_bins)
    projection = coords @ axis.numpy()
    bounds = (float(projection.min()), float(projection.max()))
    width = (bounds[1] - bounds[0]) / n_bins

    hard, occupied = hard_depth_profile(values.astype(np.float64), projection, n_bins, bounds)
    soft = soft_depth_profile(
        torch.from_numpy(values),
        torch.from_numpy(coords),
        axis,
        n_bins,
        SIGMA_SMALL * width,
        bounds=bounds,
    ).numpy()
    relative = np.abs(soft[occupied] - hard[occupied]) / np.abs(hard[occupied])
    assert relative.max() < HARD_BINNING_TOL, (relative.max(), relative.mean())

    # The convergence has to be a limit, not a coincidence: at the working sigma the profile is
    # visibly smoother than hard binning, which is the whole point of using it.
    wide = soft_depth_profile(
        torch.from_numpy(values),
        torch.from_numpy(coords),
        axis,
        n_bins,
        float(cfg.profile_sigma_frac) * width,
        bounds=bounds,
    ).numpy()
    assert (
        np.abs(wide[occupied] - hard[occupied]).max()
        > np.abs(soft[occupied] - hard[occupied]).max()
    )


def test_soft_profile_is_differentiable_and_shaped(cfg: Config):
    """Both profiles have the shapes ``specs/08`` states and pass a gradient to the values."""
    gen = np.random.default_rng(SEED + 5)
    coords = torch.from_numpy(gen.uniform(0.0, 500.0, size=(200, 3)).astype(np.float32))
    values = torch.from_numpy(gen.uniform(0.0, 3.0, size=(200, 7)).astype(np.float32))
    values.requires_grad_(True)
    axis = torch.tensor([0.0, 1.0, 0.0])

    depth = soft_depth_profile(values, coords, axis, int(cfg.profile_n_bins), 20.0)
    assert depth.shape == (int(cfg.profile_n_bins), 7)

    grid = (int(cfg.profile_grid_size), int(cfg.profile_grid_size))
    field = soft_field_profile(values, coords[:, :2], grid, 30.0)
    assert field.shape == (*grid, 7)

    (depth.sum() + field.sum()).backward()
    assert values.grad is not None
    assert float(values.grad.abs().max()) > 0.0


def test_profile_axis_stable(cfg: Config, volume: Volume):
    """The principal tissue axis is identical across epochs — bitwise, not approximately.

    ``specs/08``: "Do not recompute the principal tissue axis per epoch. A drifting axis makes
    the loss non-stationary and the metric incomparable across epochs." The control is in the
    second half: recomputing it on each LOSO fold's own volume — the mistake the caching
    prevents — *does* move it, so the stability above is a property of where the axis lives and
    not of the number being insensitive.
    """
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    scheduler = LOSOScheduler(training, cfg, seed=SEED)
    axes = [profile_axis(scheduler.volume, cfg).numpy() for _ in range(5)]
    for axis in axes[1:]:
        assert np.array_equal(axis, axes[0]), (axis, axes[0])
    assert np.isclose(np.linalg.norm(axes[0]), 1.0)
    assert axes[0][2] == 0.0, axes[0]

    drift = [
        float(np.abs(np.dot(scheduler.epoch_task(e)[0].principal_axis, axes[0])) - 1.0)
        for e in range(scheduler.n_folds)
    ]
    assert max(np.abs(drift)) > 0.0, drift


def test_anatomical_axis_can_be_supplied_before_use(cfg: Config, volume: Volume):
    """A user-supplied anatomical axis replaces the fitted one — but not once it has been read."""
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    training.set_anatomical_axis([0.0, 2.0, 0.0])
    assert np.array_equal(training.principal_axis, np.array([0.0, 1.0, 0.0]))
    with pytest.raises(Exception, match="already been used"):
        training.set_anatomical_axis([1.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------
# 3. distribution matching
# --------------------------------------------------------------------------------------


def test_distribution_falls_as_the_clouds_meet(cfg: Config):
    """The divergence is near zero between a cloud and itself and grows with separation."""
    gen = np.random.default_rng(SEED + 6)
    real = torch.from_numpy(gen.normal(size=(200, 6)).astype(np.float32))
    same = torch.from_numpy(gen.normal(size=(200, 6)).astype(np.float32))
    shifted = same + 3.0
    near = float(loss_distribution(same, real, cfg))
    far = float(loss_distribution(shifted, real, cfg))
    assert near < far, (near, far)
    assert float(loss_distribution(real, real, cfg)) < near

    moved = shifted.clone().requires_grad_(True)
    loss_distribution(moved, real, cfg).backward()
    assert moved.grad is not None
    assert float(moved.grad.abs().max()) > 0.0

    # The named fallback computes something with the same ordering.
    mmd_cfg = cfg.replace(metric_distribution_kind="mmd")
    assert float(loss_distribution(same, real, mmd_cfg)) < float(
        loss_distribution(shifted, real, mmd_cfg)
    )


# --------------------------------------------------------------------------------------
# 4. the LOSO loop, and the leakage discipline
# --------------------------------------------------------------------------------------


def small_cfg() -> Config:
    """The reduced model of T06, terms on, with the metric block sized for the fast suite.

    ``METRIC_ON`` explicitly, because the shipped defaults are 0 (see ``Config.w_autocorr``):
    these tests are about what the block computes, not about whether it is switched on.
    """
    return expression_cfg().replace(**METRIC_ON, loso_max_cells=256, metric_distribution_points=128)


def build_model(cfg: Config, volume: Volume, training: TrainingVolume) -> CTFFlow:
    """An untrained model over ``training``. Structural tests need no fit."""
    data = TrainingData.build(training, cfg)
    return CTFFlow(cfg, data, build_embeddings(cfg, volume), grf_seed=3)


def test_loso_excludes_from_retrieval(volume: Volume):
    """The hidden section reaches neither the retrieval pool nor the epoch's supervision.

    ``specs/08`` §4's structural requirement: "The hidden section is excluded from the
    retrieval index and from the field's supervision for that epoch — otherwise the model
    reconstructs it by memorisation and the loss is vacuous." All three channels are asserted,
    and each is asserted to be non-vacuous first (there *are* donors, there *are* rows), or an
    empty result would pass every exclusion trivially.
    """
    cfg = small_cfg()
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    model = build_model(cfg, volume, training)
    scheduler = LOSOScheduler(training, cfg, seed=SEED)
    rest, hidden = scheduler.epoch_task(0)

    assert hidden.section_id not in rest.section_ids
    assert set(rest.section_ids) | {hidden.section_id} == set(training.section_ids)

    index = model.data.index
    which = training.section_ids.index(hidden.section_id)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        recon = reconstruct_hidden(model, hidden, cfg, seed=SEED)
    found = recon.neighbours[recon.neighbours >= 0]
    assert found.size > 0, "no donors at all: the exclusion would be vacuous"
    assert not np.any(index.section_index[found] == which), (
        "a cell of the hidden section was retrieved as a donor"
    )

    batch = model.data.sample_batch(cfg, seed=SEED, step=0, hide=hidden.section_id)
    assert batch.n_cells > 0
    assert not np.any(index.section_index[batch.rows] == which), (
        "the hidden section supervised the field in the epoch it was hidden"
    )
    assert not np.any(index.section_index[batch.neighbours[batch.neighbours >= 0]] == which)
    assert batch.layout is not None
    assert batch.layout.section_id != hidden.section_id

    # ...and without `hide` the same call does reach it, so the exclusion is doing the work.
    open_batch = model.data.sample_batch(cfg, seed=SEED, step=0)
    assert np.any(index.section_index[open_batch.rows] == which)


def test_loso_round_robin_is_deterministic(cfg: Config, volume: Volume):
    """Same seed, same schedule; every training section is hidden exactly once per cycle."""
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    first = LOSOScheduler(training, cfg, seed=SEED)
    second = LOSOScheduler(training, cfg, seed=SEED)
    cycle = [first.epoch_task(e)[1].section_id for e in range(first.n_folds)]
    assert cycle == [second.epoch_task(e)[1].section_id for e in range(second.n_folds)]
    assert sorted(cycle) == sorted(training.section_ids)
    # ...and it repeats rather than running out.
    assert first.epoch_task(first.n_folds)[1].section_id == cycle[0]
    assert first.epoch_of(0) == 0
    assert first.epoch_of(int(cfg.loso_epoch_steps)) == 1
    assert first.active(0)
    assert not first.active(1)


def test_metric_aware_rejects_heldout(cfg: Config, volume: Volume):
    """Held-out data is a ``TypeError`` at every door, and a ``Volume`` is too.

    ``specs/08``: "Make the mistake a **type error**, not a code-review question." A plain
    ``Volume`` is refused as well as ``HeldOutSections``, because a ``Volume`` may still hold
    the evaluation sections — only ``split_holdout`` produces the type that cannot.
    """
    training, held = split_holdout(volume, "alternating", 0, cfg)
    assert isinstance(held, HeldOutSections)
    x = torch.zeros((4, 2))
    coords = torch.zeros((4, 3))

    for bad in (held, volume):
        with pytest.raises(TypeError, match="TrainingVolume"):
            loss_profile(x, coords, x, coords, bad, cfg)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            profile_axis(bad, cfg)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="TrainingVolume"):
            LOSOScheduler(bad, cfg, seed=SEED)  # type: ignore[arg-type]

    # The axis is a property of the training volume and of nothing else (SPEC_QUESTIONS C10).
    assert not hasattr(volume, "principal_axis")
    assert not hasattr(held, "principal_axis")
    assert hasattr(training, "principal_axis")


def test_metric_terms_are_finite_and_differentiable(volume: Volume):
    """One block on an untrained model: three named terms, finite, with a gradient behind them."""
    cfg = small_cfg()
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    model = build_model(cfg, volume, training)
    scheduler = LOSOScheduler(training, cfg, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        terms = metric_aware_terms(model, scheduler, cfg, step=0, seed=SEED)
    assert set(terms) == {"autocorr", "profile", "distribution"}
    for name, value in terms.items():
        assert torch.isfinite(value), (name, value)

    torch.stack(list(terms.values())).sum().backward()
    touched = [n for n, p in model.named_parameters() if p.grad is not None]
    # The generation path, and only it: encoder parameters see no gradient from these terms,
    # which is the point — nothing else in the model is a statement about a *generated* section.
    assert any(name.startswith("decoder") for name in touched), touched
    assert any(name.startswith("flow") for name in touched), touched
    assert any(name.startswith("field") for name in touched), touched
    assert any(name.startswith("intensity") for name in touched), touched
    assert not any(name.startswith("encoder") for name in touched), touched


def test_metric_terms_switch_off_by_weight(volume: Volume):
    """A zero weight removes its term rather than adding a zero, matching ``forward_train``."""
    cfg = small_cfg().replace(w_profile=0.0, w_distribution=0.0)
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    model = build_model(cfg, volume, training)
    scheduler = LOSOScheduler(training, cfg, seed=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        terms = metric_aware_terms(model, scheduler, cfg, step=0, seed=SEED)
    assert set(terms) == {"autocorr"}


def test_dominance_warning_fires(cfg: Config):
    """``specs/08``'s "do not weight these losses above reconstruction", made observable."""
    from spatialcpav25_gen.train.loso import check_metric_dominance

    terms = {"recon": torch.tensor(1.0), "autocorr": torch.tensor(10.0)}
    weights = {"recon": 1.0, "autocorr": 0.5}
    with pytest.warns(MetricDominanceWarning, match="above Config"):
        ratio = check_metric_dominance(terms, weights, cfg, step=7)
    assert ratio == pytest.approx(5.0)


# --------------------------------------------------------------------------------------
# 5. the two measurements — the point of the task
# --------------------------------------------------------------------------------------


def reconstruction_scores(model: CTFFlow, section: Section, cfg: Config, *, seed: int):
    """Score a **training** section reconstructed with itself excluded from retrieval.

    The three families ``specs/08`` asks the on/off comparison to report, measured on the
    held-in LOSO protocol so the comparison is leakage-free.

    Everything is computed on the model's **drawn counts** (``Reconstruction.drawn``), not on
    its mean field, and that is the difference between measuring the model and measuring the
    loss. The autocorrelation term deliberately compares a *noise-corrected* mean-field Moran's
    I against the real section's measured one (SPEC_QUESTIONS C24), so a model that satisfies
    it has a mean field whose **uncorrected** I sits above the real section's — scoring that
    would penalise the model for exactly what the loss asks of it, and would report a
    regression where the emitted section had improved. The benchmark scores emitted counts;
    so does this.

    Returns ``morans_mae`` (lower is better), ``morans_r``, ``marker_depth_r`` and
    ``localization_r`` (higher is better).
    """
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        recon = reconstruct_hidden(model, section, cfg, seed=seed)
    coords = recon.coords.numpy()
    graph = knn_weight_graph(coords, cfg)
    emitted = normalised_linear(recon.drawn, 1.0, float(cfg.metric_eps))
    measured = normalised_linear(recon.x_real, 1.0, float(cfg.metric_eps))
    i_gen = morans_i(emitted, graph).numpy()
    i_real = morans_i(measured, graph).numpy()
    markers = marker_genes(measured, graph, cfg)

    axis = profile_axis(model.data.vol, cfg)
    bounds = (float((recon.coords @ axis).min()), float((recon.coords @ axis).max()))
    sigma = float(cfg.profile_sigma_frac) * (bounds[1] - bounds[0]) / int(cfg.profile_n_bins)
    depth_gen = soft_depth_profile(
        emitted.index_select(1, markers),
        recon.coords,
        axis,
        int(cfg.profile_n_bins),
        sigma,
        bounds=bounds,
    ).numpy()
    depth_real = soft_depth_profile(
        measured.index_select(1, markers),
        recon.coords,
        axis,
        int(cfg.profile_n_bins),
        sigma,
        bounds=bounds,
    ).numpy()

    plane = recon.coords[:, :2]
    # The predicted composition lives at uniform slab points, the labels at the cells; both
    # are binned on the same grid, sized from the field's own correlation length.
    type_grid = _type_grid_size(float(plane[:, 0].max() - plane[:, 0].min()), cfg)
    type_sigma = (
        float(cfg.profile_sigma_frac)
        * float(plane[:, 0].max() - plane[:, 0].min())
        / float(type_grid[0])
    )
    bounds = (
        (float(plane[:, 0].min()), float(plane[:, 0].max())),
        (float(plane[:, 1].min()), float(plane[:, 1].max())),
    )
    type_gen = soft_field_profile(
        recon.types_gen, recon.types_coords[:, :2], type_grid, type_sigma, bounds=bounds
    ).numpy()
    type_real = soft_field_profile(
        recon.types_real, plane, type_grid, type_sigma, bounds=bounds
    ).numpy()
    return {
        "morans_mae": float(np.median(np.abs(i_gen - i_real))),
        "morans_r": _safe_r(i_gen, i_real),
        "marker_depth_r": float(
            np.mean([_safe_r(depth_gen[:, g], depth_real[:, g]) for g in range(depth_gen.shape[1])])
        ),
        "localization_r": float(
            np.mean(
                [
                    _safe_r(type_gen[..., c].ravel(), type_real[..., c].ravel())
                    for c in range(type_gen.shape[2])
                ]
            )
        ),
    }


def _safe_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r, returning 0.0 where a constant vector makes it undefined."""
    if a.std() == 0.0 or b.std() == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


@pytest.fixture(scope="module")
def arms() -> dict[str, object]:
    """Three arms on the default holdout, trained once and shared by both measurements.

    ``off``
        T06's model: no metric-aware terms and no internal LOSO.
    ``schedule``
        Internal LOSO **without** the terms — the same section hidden from the same epoch's
        batches, retrieval and layout targets, and nothing charged for it.
    ``on``
        Both.

    ``schedule`` is the control ``specs/08``'s one-line comparison does not name and the result
    needs: hiding a section is part of the method, so ``on`` trains on eight ninths of the stack
    at any moment and ``off`` on all of it. Without the middle arm a difference between ``off``
    and ``on`` cannot be attributed to the losses at all.
    """
    base = expression_cfg()
    off = train_model(holdout_mode="alternating", steps=TRAIN_STEPS, cfg=base.replace(**METRIC_OFF))
    schedule = train_model(
        holdout_mode="alternating",
        steps=TRAIN_STEPS,
        cfg=base.replace(**METRIC_OFF),
        loso=True,
    )
    with warnings.catch_warnings():
        # The block's own dominance warning is a *result*, recorded in PROGRESS.md, not a
        # reason to fail the fixture that measures it.
        warnings.simplefilter("ignore", MetricDominanceWarning)
        on = train_model(
            holdout_mode="alternating", steps=TRAIN_STEPS, cfg=base.replace(**METRIC_ON)
        )
    return {"off": off, "schedule": schedule, "on": on}


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "specs/08's definition of done, kept verbatim and RED. On the fixture at T06's budget "
        "the terms make every metric they are made of slightly worse: Moran's MAE 0.0395 -> "
        "0.0408, marker-depth r 0.9792 -> 0.9657, localization 0.9734 -> 0.9660, at a "
        "reconstruction cost of 1.5901 -> 1.6845 nats/pair. The schedule-only control sits "
        "between them (0.0340 / 0.9719 / 0.9696), so the loss terms and not the hidden section "
        "are what costs. specs/08 says to report this rather than to tune until it passes; the "
        "shipped weights are therefore 0 and PROGRESS.md carries the table. It is a budget "
        "effect and not a dead loss: at 2400 steps the ordering reverses on Moran's MAE (0.0339 "
        "off, 0.0279 on), on marker-depth r (0.983 off, 0.990 on) and on the covariance (9.049 "
        "off, 8.489 on), with localization 0.001 behind. T06's 1200 steps is early stopping "
        "fitted to this fixture, which is open risk R4's own point."
    ),
)
def test_metric_losses_improve_metrics(arms: dict[str, object]):
    """**The point of the whole task**: the terms improve the metrics they are made of.

    ``specs/08``: "Train 1000 steps with the metric-aware terms on vs. off; with them on,
    held-in Moran's agreement, marker-depth r, and localization must each be better. If any is
    not, report it — a loss that does not improve its own metric is either mis-specified or
    mis-weighted." The numbers are in PROGRESS.md as ablation A2.

    Held as a strict xfail so the shortfall is on the record and a later change that closes it
    breaks the suite until the record is updated — T05's precedent, and T06's and T07's.
    """
    scores = {}
    for name in ("off", "schedule", "on"):
        trained = arms[name]
        section = trained.training.sections[trained.training.n_sections // 2]  # type: ignore[attr-defined]
        scores[name] = reconstruction_scores(
            trained.model,  # type: ignore[attr-defined]
            section,
            trained.cfg,  # type: ignore[attr-defined]
            seed=SEED + 9,
        )
    report = " ".join(f"{name}={scores[name]}" for name in ("off", "schedule", "on"))
    print(f"\nA2 metric-aware on/off, held-in LOSO reconstruction:\n  {report}")
    for name in ("off", "schedule", "on"):
        history = arms[name].history  # type: ignore[attr-defined]
        final = {key: round(values[-1], 4) for key, values in history.terms.items()}
        print(f"  {name}: final terms {final}")
        if history.metric_ratio:
            ratios = history.metric_ratio
            print(
                f"  {name}: metric/reconstruction ratio median "
                f"{float(np.median(ratios)):.3f} max {max(ratios):.3f} over {len(ratios)} blocks"
            )
    # `specs/08`'s criterion is stated against the terms being off, which is the `off` arm;
    # the `schedule` arm is reported beside it so a difference can be attributed.
    assert scores["on"]["morans_mae"] < scores["off"]["morans_mae"], report
    assert scores["on"]["marker_depth_r"] > scores["off"]["marker_depth_r"], report
    assert scores["on"]["localization_r"] > scores["off"]["localization_r"], report


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Open risk R4's success criterion, kept verbatim and RED on both regimes. With the "
        "metric-aware terms on: alternating model 11.022 against a 7.732 baseline (ceiling "
        "5.601), consecutive-3 model 13.391 against 11.383 (ceiling 5.513). The consecutive "
        "arm improves on T06's 17.7 and the alternating one does not improve on its 9.316, and "
        "neither reaches its baseline. Per specs/08's own instruction the covariance claim is "
        "therefore DOWNGRADED to a mechanism claim and specs/10 2 frames it that way; the "
        "verdict is recorded in PROGRESS.md."
    ),
)
@pytest.mark.parametrize("holdout_mode", ["alternating", "consecutive"])
def test_metric_losses_close_the_covariance_loss(
    arms: dict[str, object], synthetic: tuple[Volume, GroundTruthField], holdout_mode: str
):
    """**Open risk R4's success criterion**: the covariance error falls below the baseline's.

    ``specs/08``: with the metric-aware terms on, the generated section's gene-gene correlation
    Frobenius error against the held-out section must fall below the independent-donor
    baseline's **7.783** on the default ``alternating`` holdout **and hold at
    ``consecutive-3``** (where T06 measured 17.7 for the model against 11.3 for the baseline).
    Both, not either.

    The achievable ceiling is reported beside the pair and asserted as a floor: the same cells
    with the fixture's true ``mu`` and only a fresh count draw. A model below it has a
    measurement bug, not a result (SPEC_QUESTIONS B16).

    Strict xfail: the criterion is unmet on both regimes and the *decision* it forces —
    downgrade the covariance claim — is recorded rather than drifted into.
    """
    _, gt = synthetic
    if holdout_mode == "alternating":
        trained = arms["on"]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MetricDominanceWarning)
            trained = train_model(
                holdout_mode="consecutive",
                steps=TRAIN_STEPS,
                cfg=expression_cfg().replace(**METRIC_ON),
            )
    counts, xyz = generate_counts(trained, seed=SEED + 5)  # type: ignore[arg-type]
    section = trained.held_out  # type: ignore[attr-defined]
    real = np.asarray(section.counts.todense(), dtype=np.float64)
    keep = informative_genes(real)
    target = gene_correlation(real, keep)

    ours = frobenius_offdiag(gene_correlation(counts, keep), target)
    theirs = frobenius_offdiag(
        gene_correlation(donor_baseline(trained, xyz, seed=SEED + 6), keep),  # type: ignore[arg-type]
        target,
    )
    true_xyz = to_xyz(section).astype(np.float64)
    mu = gt.expression_mu(gt.latent(true_xyz), true_xyz, section.cell_type)
    draws = [
        gene_correlation(gt.sample_counts(mu, seed=600 + s), keep) for s in range(CEILING_DRAWS)
    ]
    ceiling = float(np.mean([frobenius_offdiag(draw, target) for draw in draws]))
    report = f"{holdout_mode}: model={ours:.3f} baseline={theirs:.3f} ceiling={ceiling:.3f}"
    print(f"\nR4 covariance, {report}")
    assert ours > ceiling, f"below the achievable ceiling, i.e. a measurement bug: {report}"
    assert ours < theirs, report


@pytest.mark.slow
def test_metric_block_costs_what_it_claims(volume: Volume):
    """The block's cost control works: it runs every ``loso_every_k_steps`` and no oftener."""
    cfg = small_cfg().replace(loso_every_k_steps=4, loso_epoch_steps=8)
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    model = build_model(cfg, volume, training)
    scheduler = LOSOScheduler(training, cfg, seed=SEED)
    assert [scheduler.active(step) for step in range(8)] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    hidden = [scheduler.hidden_section(step).section_id for step in range(16)]
    assert len(set(hidden[:8])) == 1
    assert len(set(hidden[8:])) == 1
    assert hidden[0] != hidden[8]
    del model


def test_knn_graph_matches_a_kdtree_by_hand(cfg: Config):
    """The graph is the row-normalised kNN graph it says it is, with no self-loops."""
    gen = np.random.default_rng(SEED + 7)
    coords = gen.uniform(0.0, 100.0, size=(30, 2))
    graph = knn_weight_graph(coords, cfg, k=4)
    dense = dense_weights(graph)
    assert np.allclose(dense.sum(axis=1), 1.0)
    assert np.allclose(np.diag(dense), 0.0)
    idx = cKDTree(coords).query(coords, k=5)[1][:, 1:]
    for i in range(coords.shape[0]):
        assert set(np.nonzero(dense[i])[0]) == set(idx[i])


def test_reconstruction_is_deterministic(volume: Volume):
    """Two reconstructions at the same seed are bitwise identical (Convention 3)."""
    cfg = small_cfg()
    training, _ = split_holdout(volume, "alternating", 0, cfg)
    model = build_model(cfg, volume, training)
    hidden = training.sections[2]
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        first = reconstruct_hidden(model, hidden, cfg, seed=SEED)
        second = reconstruct_hidden(model, hidden, cfg, seed=SEED)
    assert torch.equal(first.x_gen, second.x_gen)
    assert np.array_equal(first.gene_idx, second.gene_idx)
    assert np.array_equal(first.neighbours, second.neighbours)


def test_synthetic_volume_is_the_shared_one(volume: Volume):
    """Guard: the fixture these numbers are quoted against is the seed-0 synthetic volume."""
    other, _ = make_synthetic_volume(seed=0)
    assert volume.section_ids == other.section_ids
    assert volume.n_genes == other.n_genes
