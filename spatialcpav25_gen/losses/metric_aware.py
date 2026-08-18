"""Metric-aware losses: the evaluation criteria, made differentiable (T08).

Every other loss in this package is a statement about the training data — the likelihood of
the counts that were observed, the consistency of two virtual sections with each other. None
of them says anything about the *distribution* of a generated section, which is what the
benchmark scores. That gap has a name in ``PROGRESS.md``: open risk **R4**, the likelihood
improving while every distributional statistic of the generated section degrades. These are
the terms that can close it.

Three families, matching ``specs/08`` §1-§3:

``loss_autocorr``
    Per-gene Moran's I and Geary's C agreement on a fixed kNN graph.
``loss_profile``
    Soft depth profiles, a coarse 2-D field, and the per-type spatial histogram that is
    where cell-type localization is optimised.
``loss_distribution``
    An entropic Sinkhorn divergence between the generated and real cells in a fixed PCA
    basis (multi-bandwidth RBF MMD as the named fallback).

Leakage
-------
These losses are computed by hiding one **training** section and reconstructing it from the
others (:mod:`spatialcpav25_gen.train.loso`). The evaluation sections are never touched, and
that is enforced by type rather than by convention: anything here that needs a volume takes a
:class:`~spatialcpav25_gen.data.schema.TrainingVolume`, which only
:func:`~spatialcpav25_gen.data.loaders.split_holdout` produces, and
:class:`~spatialcpav25_gen.data.schema.HeldOutSections` is not one and never becomes one.
:func:`require_training_volume` is the single gate, so the failure is a ``TypeError`` with a
message rather than a silent success on the wrong data.

Scale
-----
``specs/08`` §1 is explicit that both sides must be on the same scale and that the choice has
to be documented. Both sides of every term here are the identical transform of a **count
draw**, library-size-normalised by the same per-cell constant (:func:`normalised_linear`,
:func:`normalised_log`). This is an input-side transform of the kind Convention 5 permits;
nothing here is a decoder target.

The transform is **linear** for the autocorrelation and profile terms, where ``specs/08``
suggests ``log1p``, and the reason is a measurement (SPEC_QUESTIONS C24). The generated side is
a draw carried on a straight-through estimator (see :mod:`spatialcpav25_gen.train.loso`), whose
gradient is the derivative of the transform *at the sampled point*. ``log1p`` is concave and
counts are sparse: at a zero draw its slope is 1, while the true derivative of
``E[log1p(X)]`` with respect to the mean is far smaller, because most draws stay at zero. The
optimiser spends that gap — measured, the largest normalised value climbs 4.5 -> 28 within 28
metric steps and the parameter gradient reaches 1.5e7, on the profile terms alone. On a linear
scale the same estimator is *exact* for a bin mean, which is what a profile is, and the
pathology disappears. Moran's I and Geary's C are invariant to a per-gene affine rescaling, so
the choice does not move them at all; it moves the profiles, which are the terms it is made
for. The distribution term keeps ``log1p`` because its fixed PCA basis was fitted on ``log1p``
expression and a basis has to be applied to the data it was fitted on.

The graph is not differentiated through
---------------------------------------
:func:`knn_weight_graph` builds ``W`` from detached coordinates with a KD-tree and returns a
constant sparse tensor. ``specs/08``'s "Do NOT differentiate through kNN graph construction"
is therefore structural: there is no path from ``W`` back to anything.
"""

from __future__ import annotations

from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import TrainingVolume
from spatialcpav25_gen.losses.sefl import mmd2_rbf, sinkhorn_divergence

__all__ = [
    "METRIC_TERMS",
    "MetricDominanceWarning",
    "gearys_c",
    "knn_weight_graph",
    "loss_autocorr",
    "loss_distribution",
    "loss_profile",
    "marker_genes",
    "metric_ratio",
    "morans_i",
    "normalised_linear",
    "normalised_log",
    "profile_axis",
    "require_training_volume",
    "soft_depth_profile",
    "soft_field_profile",
]

FloatArray = npt.NDArray[np.float64]

SparseTensor: TypeAlias = Tensor
"""A ``torch`` sparse COO tensor.

``specs/08`` writes the weight matrix's type as ``SparseTensor``; ``torch`` has no such class
(sparsity is a layout of :class:`~torch.Tensor`, not a type), so the name is an alias that
keeps the spec's signatures readable and says what the argument has to be.
"""

METRIC_TERMS: tuple[str, ...] = ("autocorr", "profile", "distribution")
"""The named terms the metric-aware block contributes to a training step, in weight order."""


class MetricDominanceWarning(UserWarning):
    """The weighted metric-aware terms exceeded their share of the reconstruction terms.

    ``specs/08``'s "Do not weight these losses above reconstruction". A warning rather than an
    error for the same reason T07's version is: a term that briefly overtakes reconstruction is
    survivable and the trajectory is the diagnosis, but a run that ends with it firing has a
    result to report rather than to use quietly.
    """


# --------------------------------------------------------------------------------------
# the graph, and the two statistics on it
# --------------------------------------------------------------------------------------


def knn_weight_graph(
    coords: npt.NDArray[Any], cfg: Config, *, k: int | None = None
) -> SparseTensor:
    """Row-normalised kNN spatial weights. ``(N, D)`` -> sparse ``(N, N)`` float32.

    Parameters
    ----------
    coords
        ``(N, D)`` cell positions. In-plane ``(N, 2)`` for a section, ``(N, 3)`` for a slab;
        the statistics below do not care which, only that both sides use the same convention.
    cfg
        Supplies ``metric_knn_k`` — 10, the degree the vendored paper metric uses — and
        ``debug_shapes``.
    k
        Overrides ``cfg.metric_knn_k``. Clipped to ``N - 1``.

    Returns
    -------
    SparseTensor
        ``(N, N)`` sparse COO, ``w_ij = 1/k`` for each of ``i``'s ``k`` nearest neighbours and
        zero elsewhere; no self-loops, so ``S0 = N``.

    Notes
    -----
    **Built from detached coordinates, on purpose.** ``specs/08``: "Build ``W`` once per
    section from the generated positions (detached — do not differentiate through graph
    construction; it is not meaningfully differentiable and the attempt introduces
    instability)." The KD-tree query is a NumPy call on a NumPy array, so there is no autograd
    path into the result at all — the prohibition is structural rather than a rule to remember.
    """
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError(f"knn_weight_graph expects (N, D) coordinates, got {points.shape}")
    n = int(points.shape[0])
    degree = int(cfg.metric_knn_k if k is None else k)
    if n < 2:
        raise ValueError(f"knn_weight_graph needs at least 2 points, got {n}")
    degree = max(1, min(degree, n - 1))
    idx = np.atleast_2d(cKDTree(points).query(points, k=degree + 1)[1][:, 1:]).astype(np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), degree)
    indices = torch.from_numpy(np.stack([rows, idx.reshape(-1)], axis=0))
    values = torch.full((n * degree,), 1.0 / float(degree), dtype=torch.float32)
    graph = torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()
    if cfg.debug_shapes:
        assert graph.shape == (n, n), graph.shape
    return graph


def morans_i(
    x: Tensor, W: SparseTensor, *, eps: float = 1e-8, noise_var: Tensor | None = None
) -> Tensor:
    """Moran's I per gene. ``(N, G)`` and sparse ``(N, N)`` -> ``(G,)``.

    ``I = (N / S0) * sum_ij w_ij (x_i - xbar)(x_j - xbar) / sum_i (x_i - xbar)^2``.

    Differentiable in ``x`` for a fixed ``W``. ``eps`` floors the variance denominator at
    ``eps * N`` (``Config.metric_eps`` at the call sites); see :func:`_floor`.

    ``noise_var`` is ``(N, G)`` per-observation sampling variance, added to the denominator.
    Pass it when ``x`` is a **mean field** whose measured counterpart is a *draw*, which is the
    generated side's situation throughout T08: the two are otherwise not the same statistic.
    Sampling noise is independent between cells and ``W`` has no self-loops, so it leaves the
    numerator's expectation alone and inflates the denominator by exactly ``sum_i Var_i`` —
    i.e. it *attenuates* the measured I, by a factor that depends on how deeply the section was
    sampled. Matching a noiseless model statistic to a noisy measured one therefore asks the
    model for a mean field that is *less* autocorrelated than the tissue, and the error is not
    small on sparse data. Adding the term makes the model's side an estimate of the same
    quantity the real side measures (SPEC_QUESTIONS C24).
    """
    values, graph, n, s0 = _check_autocorr_inputs(x, W, "morans_i")
    centred = values - values.mean(dim=0, keepdim=True)
    lagged = torch.sparse.mm(graph, centred)
    numerator = (centred * lagged).sum(dim=0)
    denominator = _with_noise(centred.pow(2).sum(dim=0), noise_var)
    return cast(Tensor, (n / s0) * numerator / _floor(denominator, n, eps))


def gearys_c(
    x: Tensor, W: SparseTensor, *, eps: float = 1e-8, noise_var: Tensor | None = None
) -> Tensor:
    """Geary's C per gene. ``(N, G)`` and sparse ``(N, N)`` -> ``(G,)``.

    ``C = ((N - 1) / (2 S0)) * sum_ij w_ij (x_i - x_j)^2 / sum_i (x_i - xbar)^2``.

    ``eps`` floors the same variance denominator as :func:`morans_i`, and ``noise_var`` plays
    the same role — with one extra term, because a draw's noise enters Geary's *numerator*
    too: ``E[(x_i - x_j)^2] = (m_i - m_j)^2 + Var_i + Var_j``, so the sum picks up
    ``(r_i + c_i) Var_i`` summed over cells, ``r`` and ``c`` being ``W``'s row and column sums.

    The numerator is expanded as ``sum_i r_i x_i^2 + sum_j c_j x_j^2 - 2 sum_ij w_ij x_i x_j``
    with ``r`` and ``c`` the row and column sums of ``W``, so the whole statistic costs two
    sparse matrix products and never forms the ``(N, N)`` difference tensor — which at the
    4000-cell subsample ``specs/08`` §4 allows would be 16 million entries per gene.
    """
    values, graph, n, s0 = _check_autocorr_inputs(x, W, "gearys_c")
    row = torch.sparse.sum(graph, dim=1).to_dense().unsqueeze(1)
    col = torch.sparse.sum(graph, dim=0).to_dense().unsqueeze(1)
    squared = values.pow(2)
    cross = (values * torch.sparse.mm(graph, values)).sum(dim=0)
    numerator = (row * squared).sum(dim=0) + (col * squared).sum(dim=0) - 2.0 * cross
    if noise_var is not None:
        numerator = numerator + ((row + col) * noise_var).sum(dim=0)
    centred = values - values.mean(dim=0, keepdim=True)
    denominator = _with_noise(centred.pow(2).sum(dim=0), noise_var)
    scale = (n - 1.0) / (2.0 * s0)
    return cast(Tensor, scale * numerator / _floor(denominator, n, eps))


def _check_autocorr_inputs(
    x: Tensor, W: SparseTensor, where: str
) -> tuple[Tensor, SparseTensor, float, Tensor]:
    """Validate ``(x, W)`` and return ``(x, W_coalesced, float(N), S0)``."""
    if x.ndim != 2:
        raise ValueError(f"{where} expects (N, G) values, got {tuple(x.shape)}")
    if not W.is_sparse:
        raise ValueError(f"{where} expects a sparse (N, N) weight matrix, got a dense tensor")
    if W.shape != (x.shape[0], x.shape[0]):
        raise ValueError(
            f"{where}: the weight matrix is {tuple(W.shape)} but there are {x.shape[0]} cells"
        )
    graph = W.coalesce().to(x.dtype)
    return x, graph, float(x.shape[0]), graph.values().sum()


def _with_noise(denominator: Tensor, noise_var: Tensor | None) -> Tensor:
    """Add the per-gene total sampling variance to a variance denominator."""
    return denominator if noise_var is None else denominator + noise_var.sum(dim=0)


def _floor(denominator: Tensor, n: float, eps: float) -> Tensor:
    """Clamp a per-gene variance denominator at ``eps * N``.

    **Not the smallest representable float**, which is what a naive guard reaches for. Both
    statistics are a ratio whose denominator is a gene's total squared deviation, so the
    *value* is bounded (a row-normalised ``W`` is a contraction, hence ``|I| <= 1``) but the
    *gradient* carries a ``1 / denominator``. A gene that is constant over the subsample — an
    all-zero gene on a sparse panel — has a denominator of exactly zero: the ratio is
    ``0 / tiny``, which is fine, and its derivative is ``inf``, which becomes ``NaN`` one
    subtraction later and takes the run with it. Flooring at ``finfo.tiny`` (1e-38 in float32)
    leaves that hazard in place; flooring at ``eps * N`` removes it.

    ``eps * N`` is the natural scale because the denominator is a *sum* over ``N`` cells: at
    ``Config.metric_eps = 1e-8`` and 1500 cells it declares a gene constant once its RMS
    deviation falls below ``1e-4`` of a per-cell library, which no gene with even one non-zero
    cell comes close to.
    """
    return torch.clamp(denominator, min=eps * n)


def loss_autocorr(
    x_gen: Tensor,
    x_real: Tensor,
    W_gen: SparseTensor,
    W_real: SparseTensor,
    cfg: Config,
    *,
    noise_var: Tensor | None = None,
) -> Tensor:
    """Spatial-autocorrelation agreement. Scalar ``()``.

    Parameters
    ----------
    x_gen, x_real
        ``(N, G)`` and ``(M, G)`` library-size-normalised expression of the generated and the
        real section, on the same gene axis and the same scale
        (:func:`normalised_linear` in the pipeline).
    W_gen, W_real
        Their kNN graphs, from :func:`knn_weight_graph`. Two graphs and not one because in
        general the two sections' positions differ; when the reconstruction is evaluated at
        the real cells' own positions they are the same matrix, which is correct rather than
        redundant.
    cfg
        Supplies ``metric_huber_delta`` and ``metric_eps``.
    noise_var
        ``(N, G)`` sampling variance of the **generated** side, when ``x_gen`` is a mean field
        (as it is in the pipeline). Additive keyword; see :func:`morans_i`.

    Returns
    -------
    Tensor
        Scalar: the variance-weighted Huber loss between the per-gene Moran's I vectors plus
        the same between the per-gene Geary's C vectors.

    Notes
    -----
    Genes are weighted by their **real** variance, normalised to sum to one, so that a gene
    that is flat in the real section — whose I is then an estimate of nothing — cannot
    dominate the term. The weights are detached: they say which genes matter, and letting a
    gradient reach them would let the model reduce the loss by deciding that fewer do.
    """
    weights = _gene_weights(x_real, cfg)
    delta = float(cfg.metric_huber_delta)
    eps = float(cfg.metric_eps)
    terms = [
        _weighted_huber(
            morans_i(x_gen, W_gen, eps=eps, noise_var=noise_var),
            morans_i(x_real, W_real, eps=eps).detach(),
            weights,
            delta,
        ),
        _weighted_huber(
            gearys_c(x_gen, W_gen, eps=eps, noise_var=noise_var),
            gearys_c(x_real, W_real, eps=eps).detach(),
            weights,
            delta,
        ),
    ]
    return torch.stack(terms).sum()


def _gene_weights(x_real: Tensor, cfg: Config) -> Tensor:
    """``(G,)`` detached per-gene weights: real variance, normalised to sum to one.

    The variance is measured on ``log1p`` of the real values even though the statistic itself
    is computed on the linear scale, and that is deliberate. ``specs/08`` asks for weights that
    keep uninformative genes from dominating; a variance on the linear scale grows with the
    square of a gene's mean, so on a panel whose expression spans four orders of magnitude it
    would put essentially all of its mass on the handful of most-expressed genes — which is a
    different criterion from "informative", and not the one the spec names. The weights are
    detached either way, so this is a choice of *ranking*, not of gradient.
    """
    variance = torch.log1p(torch.clamp(x_real.detach(), min=0.0)).var(dim=0, unbiased=False)
    total = torch.clamp(variance.sum(), min=float(cfg.metric_eps))
    return variance / total


def _weighted_huber(gen: Tensor, real: Tensor, weights: Tensor, delta: float) -> Tensor:
    """Weighted Huber loss between two ``(G,)`` vectors. Scalar ``()``."""
    per_gene = torch.nn.functional.huber_loss(gen, real, reduction="none", delta=delta)
    return (per_gene * weights).sum()


def marker_genes(x_real: Tensor, W_real: SparseTensor, cfg: Config) -> Tensor:
    """Return the most spatially structured genes of the real section. ``(k,)`` int64 indices.

    ``specs/08`` §2's "top-k spatially variable genes from training": the ``k =
    Config.metric_marker_genes`` genes with the highest Moran's I in the section being
    reconstructed, which under internal LOSO is a *training* section. Deterministic given the
    section and the gene subsample — there is nothing stochastic to seed.

    Candidates are restricted to genes detected in at least
    ``Config.metric_marker_min_detection`` of the real cells; see that field for the failure
    the floor prevents. If fewer than ``k`` genes clear it the pool falls back to the whole
    panel, because a loss that silently computed itself over three genes would be worse than
    one computed over noisy ones.
    """
    with torch.no_grad():
        detection = (x_real.detach() > 0).to(x_real.dtype).mean(dim=0)
        score = torch.nan_to_num(morans_i(x_real, W_real, eps=float(cfg.metric_eps)), nan=0.0)
    k = min(int(cfg.metric_marker_genes), int(x_real.shape[1]))
    eligible = detection >= float(cfg.metric_marker_min_detection)
    if int(eligible.sum()) >= k:
        score = torch.where(eligible, score, torch.full_like(score, -float("inf")))
    return torch.topk(score, k).indices.sort().values


# --------------------------------------------------------------------------------------
# soft spatial profiles
# --------------------------------------------------------------------------------------


def soft_depth_profile(
    x: Tensor,
    coords: Tensor,
    axis: Tensor,
    n_bins: int,
    sigma: float,
    *,
    bounds: tuple[float, float] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Gaussian-binned profile along ``axis``. ``(N, G)``, ``(N, D)`` -> ``(n_bins, G)``.

    Parameters
    ----------
    x
        ``(N, G)`` values, on whatever scale the caller has chosen for both sides.
    coords
        ``(N, D)`` positions; ``axis`` must have the same width.
    axis
        ``(D,)`` unit direction — the principal tissue axis
        (:attr:`~spatialcpav25_gen.data.schema.TrainingVolume.principal_axis`).
    n_bins, sigma
        Bins along the projection, and the kernel width in micrometres.
        ``specs/08`` fixes ``sigma`` at ``0.75 x`` the bin width
        (``Config.profile_sigma_frac``); it is an argument because the convergence test drives
        it to zero.
    bounds
        ``(low, high)`` of the projection, so that two profiles that will be compared are
        binned identically. Additive keyword: the spec's signature has nowhere to put it, and
        without it the generated and the real profile would sit on two different rulers
        whenever their extents differ by a cell.
    eps
        Denominator guard for a bin with no cell near it (``Config.metric_eps``).

    Returns
    -------
    Tensor
        ``(n_bins, G)``: the kernel-weighted mean of ``x`` in each bin.

    Notes
    -----
    Two normalisations, in this order, and the order is what makes the limit right. Each
    **cell's** weights over bins are normalised to one (the spec's "row-normalised"), so a cell
    contributes exactly one cell's worth of evidence however many bins are near it; then each
    **bin** divides by the weight it received, which turns the sum into a mean. As ``sigma ->
    0`` the first normalisation makes the weights one-hot and the second makes the bin value the
    plain mean over the cells in it, i.e. exactly hard binning — which is what
    ``test_soft_profile_approaches_hard`` measures.
    """
    if x.ndim != 2 or coords.ndim != 2 or x.shape[0] != coords.shape[0]:
        raise ValueError(
            f"soft_depth_profile expects (N, G) and (N, D) with matching N, got "
            f"{tuple(x.shape)} and {tuple(coords.shape)}"
        )
    if axis.shape != (coords.shape[1],):
        raise ValueError(
            f"soft_depth_profile: axis must be ({coords.shape[1]},) to match the coordinates, "
            f"got {tuple(axis.shape)}"
        )
    if n_bins < 2:
        raise ValueError(f"soft_depth_profile: n_bins must be >= 2, got {n_bins}")
    projection = coords @ axis
    centres = _bin_centres(projection, n_bins, bounds, projection.dtype)
    weights = _kernel_weights(projection.unsqueeze(1), centres.unsqueeze(0), sigma, eps)
    return _weighted_bin_mean(weights, x, eps)


def soft_field_profile(
    x: Tensor,
    coords: Tensor,
    grid_shape: tuple[int, int],
    sigma: float,
    *,
    bounds: tuple[tuple[float, float], tuple[float, float]] | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Gaussian-binned 2-D field. ``(N, G)``, ``(N, 2)`` -> ``(H, W, G)``.

    The separable version of :func:`soft_depth_profile` on the section's two in-plane axes:
    the per-cell weight of grid cell ``(a, b)`` is the product of its weights on each axis,
    normalised over grid cells, and each grid cell then divides by the weight it received.
    ``coords`` are **in-plane** ``(N, 2)`` — for an axis-aligned section that is ``(x, y)``, for
    an oblique one the plane's own frame.

    ``bounds`` fixes the grid so two fields are comparable; see :func:`soft_depth_profile`.
    """
    if x.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2 or x.shape[0] != coords.shape[0]:
        raise ValueError(
            f"soft_field_profile expects (N, G) and (N, 2) with matching N, got "
            f"{tuple(x.shape)} and {tuple(coords.shape)}"
        )
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height < 2 or width < 2:
        raise ValueError(f"soft_field_profile: grid_shape must be >= (2, 2), got {grid_shape}")
    limits = (None, None) if bounds is None else bounds
    rows = _bin_centres(coords[:, 0], height, limits[0], coords.dtype)
    cols = _bin_centres(coords[:, 1], width, limits[1], coords.dtype)
    w_row = _kernel_weights(coords[:, 0:1], rows.unsqueeze(0), sigma, eps, normalise=False)
    w_col = _kernel_weights(coords[:, 1:2], cols.unsqueeze(0), sigma, eps, normalise=False)
    joint = (w_row.unsqueeze(2) * w_col.unsqueeze(1)).reshape(coords.shape[0], height * width)
    joint = joint / torch.clamp(joint.sum(dim=1, keepdim=True), min=eps)
    return _weighted_bin_mean(joint, x, eps).reshape(height, width, x.shape[1])


def _bin_centres(
    projection: Tensor, n_bins: int, bounds: tuple[float, float] | None, dtype: torch.dtype
) -> Tensor:
    """``(n_bins,)`` centres of equal-width bins spanning ``bounds`` (or the data's range)."""
    if bounds is None:
        with torch.no_grad():
            low = float(projection.min())
            high = float(projection.max())
    else:
        low, high = float(bounds[0]), float(bounds[1])
    if not high > low:
        high = low + 1.0
    width = (high - low) / float(n_bins)
    return torch.linspace(low + 0.5 * width, high - 0.5 * width, n_bins, dtype=dtype)


def _kernel_weights(
    projection: Tensor, centres: Tensor, sigma: float, eps: float, *, normalise: bool = True
) -> Tensor:
    """``(N, B)`` Gaussian weights, optionally normalised so each row sums to one."""
    scale = max(float(sigma), eps)
    distance = (projection - centres) / scale
    # Subtracting each cell's own minimum before the exponential: at the small sigma the
    # convergence test uses, every raw exponent underflows to zero and the row normalisation
    # below becomes 0/0. The shift cancels exactly in the normalised weights.
    shifted = distance.pow(2)
    weights = torch.exp(-0.5 * (shifted - shifted.min(dim=1, keepdim=True).values))
    if not normalise:
        return weights
    return weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=eps)


def _weighted_bin_mean(weights: Tensor, x: Tensor, eps: float) -> Tensor:
    """``(B, G)`` weighted mean of ``(N, G)`` ``x`` under ``(N, B)`` per-cell weights."""
    total = weights.sum(dim=0).unsqueeze(1)
    return (weights.t() @ x) / torch.clamp(total, min=eps)


def profile_axis(vol: TrainingVolume, cfg: Config) -> Tensor:
    """Return the axis the depth profile is taken along. ``(3,)`` float32.

    Read from the :class:`~spatialcpav25_gen.data.schema.TrainingVolume` — computed once from
    its own sections and cached there — never recomputed per epoch (``specs/08``: "Do not
    recompute the principal tissue axis per epoch"), and never available on a plain
    :class:`~spatialcpav25_gen.data.schema.Volume`, which would have consulted the held-out
    sections (SPEC_QUESTIONS C10).
    """
    axis = require_training_volume(vol, "profile_axis").principal_axis
    if cfg.debug_shapes:
        assert axis.shape == (3,), axis.shape
    return torch.from_numpy(np.asarray(axis, dtype=np.float32))


def loss_profile(
    x_gen: Tensor,
    coords_gen: Tensor,
    x_real: Tensor,
    coords_real: Tensor,
    vol: TrainingVolume,
    cfg: Config,
    *,
    types_gen: Tensor | None = None,
    types_real: Tensor | None = None,
    types_coords: Tensor | None = None,
) -> Tensor:
    """Depth, field and per-type spatial-profile agreement. Scalar ``()``.

    Parameters
    ----------
    x_gen, x_real
        ``(N, G)`` and ``(M, G)`` normalised ``log1p`` expression, same gene axis.
    coords_gen, coords_real
        ``(N, 3)`` and ``(M, 3)`` physical positions in micrometres.
    vol
        The **training** volume, for its principal axis. A
        :class:`~spatialcpav25_gen.data.schema.Volume` or
        :class:`~spatialcpav25_gen.data.schema.HeldOutSections` is a ``TypeError``.
    cfg
        Supplies ``profile_n_bins``, ``profile_grid_size``, ``profile_sigma_frac``,
        ``metric_marker_genes``, ``metric_knn_k``, ``metric_huber_delta`` and ``metric_eps``.
    types_gen, types_real
        ``(M, C)`` predicted per-type composition — the intensity head's ``lambda_c / sum_c
        lambda_c``, which is differentiable where a drawn label is not — and ``(N, C)`` the
        real one-hot cell types at ``coords_real``. Both or neither. Additive keywords:
        ``specs/08`` §2's third bullet asks for a per-type spatial histogram and its stated
        signature has nowhere to put the labels (SPEC_QUESTIONS C21).
    types_coords
        ``(M, 3)`` where ``types_gen`` was read; defaults to ``coords_gen``. In the pipeline
        these are **uniform points in the slab**, not the cells — see
        :func:`~spatialcpav25_gen.train.loso._uniform_composition` for the degeneracy that
        makes the difference between a loss and a divergent run.

    Returns
    -------
    Tensor
        Scalar: the marker-gene depth-profile term, plus the marker-gene 2-D field term, plus
        the per-type spatial histogram term when the types are given.

    Notes
    -----
    All three terms are Huber losses over profiles binned on a **shared** grid, computed from
    the two coordinate sets together. Sharing the grid is what makes the comparison a
    comparison; binning each side on its own extent would make a section that had drifted by
    one bin width look perfect.
    """
    axis = profile_axis(vol, cfg)
    if (types_gen is None) != (types_real is None):
        raise ValueError(
            "loss_profile: pass both types_gen and types_real or neither; the per-type "
            "histogram compares a predicted composition against the real labels"
        )
    eps = float(cfg.metric_eps)
    delta = float(cfg.metric_huber_delta)
    n_bins = int(cfg.profile_n_bins)
    grid = (int(cfg.profile_grid_size), int(cfg.profile_grid_size))

    graph_real = knn_weight_graph(coords_real.detach().numpy(), cfg)
    markers = marker_genes(x_real, graph_real, cfg)
    # Per-gene *relative* expression: each marker divided by its own mean in the real section,
    # a detached constant. Without it the Huber sees only the few genes expressed at tens of
    # counts, because a profile's units are the gene's own, and "top-k spatially variable" is
    # not the same set as "top-k abundant".
    scale = torch.clamp(x_real.detach().mean(dim=0).index_select(0, markers), min=eps)
    gen = x_gen.index_select(1, markers) / scale
    real = x_real.index_select(1, markers) / scale

    depth_bounds = _shared_bounds(coords_gen @ axis, coords_real @ axis)
    sigma_depth = float(cfg.profile_sigma_frac) * (depth_bounds[1] - depth_bounds[0]) / n_bins
    terms = [
        torch.nn.functional.huber_loss(
            soft_depth_profile(
                gen, coords_gen, axis, n_bins, sigma_depth, bounds=depth_bounds, eps=eps
            ),
            soft_depth_profile(
                real, coords_real, axis, n_bins, sigma_depth, bounds=depth_bounds, eps=eps
            ).detach(),
            delta=delta,
        )
    ]

    plane_gen, plane_real = coords_gen[:, :2], coords_real[:, :2]
    x_bounds = _shared_bounds(plane_gen[:, 0], plane_real[:, 0])
    y_bounds = _shared_bounds(plane_gen[:, 1], plane_real[:, 1])
    span = min(x_bounds[1] - x_bounds[0], y_bounds[1] - y_bounds[0])
    sigma_field = float(cfg.profile_sigma_frac) * span / float(grid[0])
    bounds = (x_bounds, y_bounds)
    terms.append(
        torch.nn.functional.huber_loss(
            soft_field_profile(gen, plane_gen, grid, sigma_field, bounds=bounds, eps=eps),
            soft_field_profile(
                real, plane_real, grid, sigma_field, bounds=bounds, eps=eps
            ).detach(),
            delta=delta,
        )
    )

    if types_gen is not None and types_real is not None:
        type_grid = _type_grid_size(span, cfg)
        type_sigma = float(cfg.profile_sigma_frac) * span / float(type_grid[0])
        type_plane = plane_gen if types_coords is None else types_coords[:, :2]
        terms.append(
            torch.nn.functional.huber_loss(
                soft_field_profile(
                    types_gen, type_plane, type_grid, type_sigma, bounds=bounds, eps=eps
                ),
                soft_field_profile(
                    types_real, plane_real, type_grid, type_sigma, bounds=bounds, eps=eps
                ).detach(),
                delta=delta,
            )
        )
    return torch.stack(terms).sum()


def _type_grid_size(span: float, cfg: Config) -> tuple[int, int]:
    """Grid for the per-type histogram: bins one ``ell_xy`` across, never finer than the field.

    See ``Config.metric_type_grid_ell_multiple`` for the measurement behind this: on a grid fine
    enough that a bin holds two cells, the per-type term's own minimiser is a one-hot intensity
    field, and chasing it destroys the anatomical field the intensity head reads from.
    """
    width = max(float(cfg.metric_type_grid_ell_multiple) * float(cfg.ell_xy), 1e-6)
    side = int(min(max(int(span / width), 2), int(cfg.profile_grid_size)))
    return (side, side)


def _shared_bounds(a: Tensor, b: Tensor) -> tuple[float, float]:
    """``(low, high)`` covering both samples, so two profiles land on the same ruler."""
    with torch.no_grad():
        low = float(torch.minimum(a.min(), b.min()))
        high = float(torch.maximum(a.max(), b.max()))
    return (low, high if high > low else low + 1.0)


# --------------------------------------------------------------------------------------
# distribution matching
# --------------------------------------------------------------------------------------


def loss_distribution(x_gen: Tensor, x_real: Tensor, cfg: Config) -> Tensor:
    """Divergence between the generated and real cell clouds. Scalar ``()``.

    Parameters
    ----------
    x_gen, x_real
        ``(N, P)`` and ``(M, P)`` cell coordinates in a **fixed PCA basis fitted once on
        training data** — in the pipeline
        :class:`~spatialcpav25_gen.model.retrieval.ExpressionPCs`, applied by
        :mod:`spatialcpav25_gen.train.loso`. Projection and subsampling are the caller's,
        which is what lets this function stay deterministic and seedless (Convention 3): it
        has no randomness of its own.
    cfg
        Supplies ``metric_distribution_kind``, ``metric_sinkhorn_blur_multiple``,
        ``metric_eps`` and — through the shared solver — ``sefl_sinkhorn_iters`` or the
        ``sefl_mmd_*`` bandwidths.

    Returns
    -------
    Tensor
        Scalar: the debiased entropic Sinkhorn divergence at ``blur = metric_sinkhorn_blur_
        multiple x`` the median nearest-neighbour distance of the **real** cloud in PC space
        (``specs/08`` §3), or the multi-bandwidth RBF MMD² when
        ``Config.metric_distribution_kind == "mmd"``, **per PC dimension**.

    Notes
    -----
    This is the term that targets embedding mixing, and the honest note ``specs/08`` asks for
    is in ``PROGRESS.md``: mixing is the one metric where the competing method is genuinely
    strong, because its per-gene chimerism maximises cloud overlap almost by construction. A
    tie is an acceptable outcome there.

    The blur is measured on the real cloud under ``no_grad``: an entropic scale that tracked
    the generated cloud could be reduced by shrinking it, which is a way of scoring well by
    generating less.

    **Per PC dimension**, which is not cosmetic. The Sinkhorn divergence has the units of a
    squared distance in PC space, so its magnitude grows with ``Config.expr_pca_dim`` — at 16
    components it lands around 10 while the reconstruction NLL is around 1.6, and a weight of
    0.5 then puts the term at three times reconstruction without anyone having chosen that.
    ``specs/08``'s "do not weight these losses above reconstruction" is a statement about the
    *weighted* term, so the term has to be on a stated scale before a weight can mean anything;
    dividing by ``P`` makes it a mean-per-component and makes ``w_distribution`` comparable
    across panels and PCA widths. The MMD is divided by the same ``P`` for the same reason,
    though it is already O(1).
    """
    if x_gen.ndim != 2 or x_real.ndim != 2 or x_gen.shape[1] != x_real.shape[1]:
        raise ValueError(
            f"loss_distribution expects (N, P) and (M, P), got {tuple(x_gen.shape)} and "
            f"{tuple(x_real.shape)}"
        )
    dimensions = float(x_gen.shape[1])
    if cfg.metric_distribution_kind == "mmd":
        return mmd2_rbf(x_gen, x_real, cfg) / dimensions
    blur = float(cfg.metric_sinkhorn_blur_multiple) * _median_nn_distance(x_real, cfg)
    return sinkhorn_divergence(x_gen, x_real, blur, cfg) / dimensions


def _median_nn_distance(points: Tensor, cfg: Config) -> float:
    """Median nearest-neighbour distance within a point cloud, ignoring self-pairs."""
    with torch.no_grad():
        distances = torch.cdist(points, points)
        distances.fill_diagonal_(float("inf"))
        median = float(torch.median(distances.min(dim=1).values))
    return max(median, float(cfg.metric_eps))


# --------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------


def normalised_linear(
    counts: Tensor, reference_total: float, eps: float, *, totals: Tensor | None = None
) -> Tensor:
    """Library-size normalise to ``reference_total``, no nonlinearity. ``(N, G)`` -> ``(N, G)``.

    The scale the autocorrelation and profile terms run on; see the module docstring for why
    it is not ``log1p``. ``totals`` behaves exactly as in :func:`normalised_log`, and the
    pipeline passes the **real** cells' library sizes for both sides for the same reason.
    """
    denominator = counts.sum(dim=1, keepdim=True) if totals is None else totals
    return counts / torch.clamp(denominator, min=eps) * float(reference_total)


def normalised_log(
    counts: Tensor, reference_total: float, eps: float, *, totals: Tensor | None = None
) -> Tensor:
    """Library-size normalise to ``reference_total`` and ``log1p``. ``(N, G)`` -> ``(N, G)``.

    The one scale every function in this module consumes. ``reference_total`` is a *fixed*
    scalar — the real section's median per-cell total — used for both the generated and the
    real side, so the two are on the same ruler and the generated side cannot lower the loss
    by moving its own reference (``specs/08`` §1's "be consistent and document it").

    ``totals`` is the ``(N, 1)`` per-cell library size to divide by; ``None`` uses each row's
    own sum, which is the ordinary input-side transform. **The pipeline passes the real cells'
    totals for both sides**, and that is load-bearing rather than tidy: with each row divided by
    its *own* generated sum, every gene's normalised value shares one denominator, and the model
    can raise the spatial autocorrelation of the whole panel at once by giving that denominator
    spatial structure — put the library into one gene and every other gene inherits
    ``-log(total)``'s structure for free. T08 watched exactly that happen: the autocorrelation
    term fell 0.33 -> 0.16 while the largest normalised value pinned at ``log1p(reference)``,
    i.e. one gene holding the entire library, and the parameter gradient went from 6 to 1.6e8.
    Dividing both sides by the same fixed per-cell constant removes the shared denominator and
    with it the shortcut; the generated library size is then not something these terms can
    move, which is correct — it belongs to ``size_factor_loss`` (T06) and the detection
    calibration (T09).

    An input-side transform, never a decoder target (Convention 5).
    """
    denominator = counts.sum(dim=1, keepdim=True) if totals is None else totals
    return torch.log1p(counts / torch.clamp(denominator, min=eps) * float(reference_total))


def require_training_volume(vol: Any, where: str) -> TrainingVolume:
    """Return ``vol`` if it is a :class:`TrainingVolume`, else raise ``TypeError``.

    The single gate through which a volume reaches a metric-aware loss.
    ``specs/08``: "Make the mistake a **type error**, not a code-review question." A plain
    :class:`~spatialcpav25_gen.data.schema.Volume` is refused as well as
    :class:`~spatialcpav25_gen.data.schema.HeldOutSections`, because a ``Volume`` may still
    contain the evaluation sections — only
    :func:`~spatialcpav25_gen.data.loaders.split_holdout` produces the type that cannot.
    """
    if not isinstance(vol, TrainingVolume):
        raise TypeError(
            f"{where} requires a TrainingVolume produced by split_holdout, got "
            f"{type(vol).__name__}. The metric-aware losses run on internal leave-one-section-"
            "out over *training* sections; held-out sections are never touched by a loss "
            "(CLAUDE.md, leakage discipline)."
        )
    return vol


def metric_ratio(terms: dict[str, Tensor], weights: dict[str, float]) -> float:
    """Weighted metric-aware terms as a fraction of the weighted reconstruction terms.

    ``specs/08``'s "Do not weight these losses above reconstruction", as a number the trainer
    can log and warn on. Returns ``inf`` when the reconstruction term is absent, which is a
    state that should never occur and is better read as "infinitely dominant" than as zero.

    The denominator is ``w_recon * recon`` and nothing else. The layout NLL is a *different
    head's* likelihood, summed over a point process rather than averaged over (cell, gene)
    pairs, and it is an order of magnitude larger on this fixture; folding it in would let a
    metric term sit at twice the reconstruction loss and still read as 0.46 — which is what it
    did on T08's first arm, while the run degraded.
    """
    metric = sum(
        weights.get(name, 0.0) * float(terms[name].detach())
        for name in METRIC_TERMS
        if name in terms
    )
    base = weights.get("recon", 0.0) * float(terms["recon"].detach()) if "recon" in terms else 0.0
    return float("inf") if base <= 0.0 else abs(metric) / base
