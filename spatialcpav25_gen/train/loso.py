"""Internal leave-one-section-out: the loop the metric-aware losses are computed in (T08 §4).

Each epoch one **training** section is hidden and reconstructed from the others; the
metric-aware terms of :mod:`spatialcpav25_gen.losses.metric_aware` compare that
reconstruction against the section that was hidden. Nothing here ever sees an evaluation
section: :class:`LOSOScheduler` takes a
:class:`~spatialcpav25_gen.data.schema.TrainingVolume` and nothing else, which is a type that
only :func:`~spatialcpav25_gen.data.loaders.split_holdout` produces.

Why a hidden section rather than the batch
------------------------------------------
``specs/08`` is emphatic that the hidden section must be excluded from the retrieval index
*and* from the field's supervision for that epoch, "otherwise the model reconstructs it by
memorisation and the loss is vacuous". Both exclusions are implemented here and asserted by
``test_loso_excludes_from_retrieval``:

* **retrieval** — every query in :func:`reconstruct_hidden` passes the hidden section's depth
  in ``exclude_z``, so no cell of that section can be a donor. The index itself is left alone;
  rebuilding it per epoch would cost a PCA and a niche pass per fold and buy the same
  guarantee.
* **supervision** — the trainer draws that epoch's batches with ``hide=section_id``
  (:meth:`~spatialcpav25_gen.model.spatialcpav25_gen.TrainingData.sample_batch`), so the
  hidden section contributes neither a reconstruction target nor a layout target while it is
  hidden.

What is reconstructed, and what is not
--------------------------------------
The reconstruction runs the **generation** path — GRF prior, flow, decoder — at the hidden
section's own cell positions, and compares expression. It does not resample the layout. Two
reasons, both structural: the layout's positions come out of a discrete point process, so
there is no gradient from a positional statistic back to the intensity head; and comparing a
freshly sampled point pattern against the real one would measure the sampler's draw noise on
top of everything else. The intensity head is still constrained, differentiably, through the
per-type spatial histogram (``specs/08`` §2's third bullet), which reads ``lambda_c`` at
uniform points in the same slab.

Comparing a model against a measurement
---------------------------------------
The real section is a **draw**; the model is a *distribution*. Every term here has to bridge
that, and T08 measured what each of the three ways of doing it costs (SPEC_QUESTIONS C24 has
the whole table; the module docstring of :mod:`spatialcpav25_gen.losses.metric_aware` has the
scale half of it).

* **Profiles** take the decoder's **mean**. A profile is a bin mean, and the bin mean of the
  mean field is an unbiased estimate of the bin mean of a draw — exactly, with no correction
  and no sampling noise in the gradient.
* **Autocorrelation** takes the mean field **plus the draw's analytic variance**
  (:func:`~spatialcpav25_gen.model.expression.draw_mean_variance`), which enters Moran's and
  Geary's denominators. Sampling noise attenuates a measured Moran's I, so an uncorrected mean
  field would be asked to be *less* autocorrelated than the tissue.
* **Distribution** takes a **reparameterised surrogate** for the draw,
  ``clamp(mean + sqrt(variance) * epsilon, 0)`` at a seeded ``epsilon``. A cloud comparison
  needs *dispersed* points on both sides — a mean field is a tighter cloud than any sample of
  it — and this one has the draw's first two moments per (cell, gene) with a pathwise gradient
  that is correct for the surrogate rather than an estimate of a discrete draw's.

Three ways this can go wrong, all measured rather than argued, because each looked reasonable
first. Comparing a mean field against samples *everywhere* made every metric the terms exist
to improve get worse (marker-depth r 0.985 -> 0.642, Frobenius covariance 9.3 -> 26.0). A
straight-through count draw in the **profile** and **autocorrelation** terms let the model
inflate expression instead — mean normalised count 4 -> 551 against a real 6.8 — partly
because a deeper draw carries proportionally less sampling noise and so measures a higher
autocorrelation, which is a way of scoring well by generating more. And a straight-through
draw in the **distribution** term did the same on its own (4 -> 761, its own loss rising
1.43 -> 5.56 while it did it): ``log1p`` is concave, its slope at a zero draw is 1, and the
true derivative of ``E[log1p(X)]`` is far smaller, so every step overstates what raising the
mean buys and the next step asks for more.

Convention 5 is untouched throughout: these are *loss* inputs, never a decoder target and
never something the generation path emits.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import loso_folds
from spatialcpav25_gen.data.schema import Section, TrainingVolume, section_seed, to_xyz
from spatialcpav25_gen.infer.planes import section_plane, uniform_slab_points
from spatialcpav25_gen.losses.metric_aware import (
    METRIC_TERMS,
    MetricDominanceWarning,
    knn_weight_graph,
    loss_autocorr,
    loss_distribution,
    loss_profile,
    metric_ratio,
    normalised_linear,
    normalised_log,
    require_training_volume,
)
from spatialcpav25_gen.model.expression import (
    draw_mean_variance,
    sample_counts,
    sample_zigamma,
)

if TYPE_CHECKING:  # pragma: no cover - the model imports this module, not the other way
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "LOSOScheduler",
    "Reconstruction",
    "check_metric_dominance",
    "metric_aware_terms",
    "reconstruct_hidden",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


class LOSOScheduler:
    """Which training section is hidden in which epoch.

    Args:
        vol: the training volume. A :class:`~spatialcpav25_gen.data.schema.Volume` or
             :class:`~spatialcpav25_gen.data.schema.HeldOutSections` is a ``TypeError`` —
             internal LOSO runs over training sections only.
        cfg: supplies ``loso_epoch_steps`` and ``loso_every_k_steps``.
        seed: explicit (Convention 3). Fixes the round-robin order, which is a *permutation*
             of the sections rather than their stack order: consecutive epochs then hide
             sections that are not neighbours, so the model is not asked to interpolate the
             same neighbourhood twice in a row.

    The folds are built once, in the constructor, by
    :func:`~spatialcpav25_gen.data.loaders.loso_folds` — the same function T01 wrote for this
    purpose and the same type check. Building them lazily would make the first step of each
    epoch pay for a volume construction; building them eagerly costs one ``median_nn_dist``
    pass per section and holds no copies (the folds share the parent's ``Section`` objects).
    """

    def __init__(self, vol: TrainingVolume, cfg: Config, *, seed: int) -> None:
        self.volume = require_training_volume(vol, "LOSOScheduler")
        self.cfg = cfg
        self.seed = int(seed)
        self._folds: list[tuple[TrainingVolume, Section]] = list(loso_folds(self.volume))
        if not self._folds:  # pragma: no cover - a Volume with < 2 sections cannot exist
            raise ValueError("LOSOScheduler: the training volume has no sections to hide")
        self._order: IntArray = np.random.default_rng(self.seed).permutation(len(self._folds))

    @property
    def n_folds(self) -> int:
        """Number of training sections, i.e. the round-robin's period."""
        return len(self._folds)

    @property
    def principal_axis(self) -> npt.NDArray[np.float64]:
        """The parent volume's principal tissue axis, computed once and never per epoch."""
        return self.volume.principal_axis

    def epoch_of(self, step: int) -> int:
        """Return the epoch a step belongs to: ``step // Config.loso_epoch_steps``."""
        return int(step) // max(int(self.cfg.loso_epoch_steps), 1)

    def active(self, step: int) -> bool:
        """Whether the metric-aware block runs at this step (every ``loso_every_k_steps``)."""
        return int(step) % max(int(self.cfg.loso_every_k_steps), 1) == 0

    def epoch_task(self, epoch: int) -> tuple[TrainingVolume, Section]:
        """Return the epoch's fold: the volume that must reconstruct, and the section it hides.

        Round-robin over the training sections in the seeded order, so every section is hidden
        equally often and the schedule is reproducible from ``seed`` alone.
        """
        if epoch < 0:
            raise ValueError(f"LOSOScheduler.epoch_task: epoch must be >= 0, got {epoch}")
        return self._folds[int(self._order[int(epoch) % self.n_folds])]

    def hidden_section(self, step: int) -> Section:
        """Return the section hidden at ``step``, which the trainer passes to ``sample_batch``."""
        return self.epoch_task(self.epoch_of(step))[1]


@dataclass(frozen=True)
class Reconstruction:
    """One hidden section, reconstructed, beside the real thing.

    Every field is on the subsample of cells the block ran on, and the generated and the real
    side share those cells' positions — the reconstruction is evaluated where the section's
    cells actually are (see the module docstring).

    Attributes
    ----------
    x_gen, x_real
        ``(N, G')`` library-size-normalised expression, generated and real, on the linear
        scale the autocorrelation and profile terms take. The generated side is the decoder's
        **mean**; a bin mean of it is an unbiased estimate of the same bin mean of a draw,
        which a profile comparison needs and a draw carried on a straight-through estimator
        does not give (see the module docstring).
    noise_var
        ``(N, G')`` sampling variance of a draw at the same scale, the correction that makes
        the generated Moran's I and Geary's C estimates of the *measured* statistic rather
        than of its noiseless limit.
    coords
        ``(N, 3)`` float32 physical positions, shared by both sides.
    types_gen
        ``(N, C)`` the intensity head's per-type composition ``lambda_c / sum_c lambda_c``.
    types_real
        ``(N, C)`` one-hot real cell types.
    types_coords
        ``(M, 3)`` float32 uniform points in the slab where ``types_gen`` was read.
    drawn
        ``(N, G')`` an actual count draw from the decoder at these cells, detached. **No loss
        reads it** — it is the measurement quantity: the thing the model would emit, and the
        only side of the reconstruction that is comparable with the real section the way the
        benchmark compares them. T08's on/off table is computed on it.
    pcs_gen, pcs_real
        ``(N, P)`` scores in the retrieval index's fixed expression PCA basis.
    gene_idx
        ``(G',)`` the panel columns this block drew.
    neighbours
        ``(N, K)`` retrieval result, kept so the exclusion can be asserted structurally.
    section_id
        The hidden section's identifier.
    """

    x_gen: Tensor
    x_real: Tensor
    noise_var: Tensor
    coords: Tensor
    types_gen: Tensor
    types_real: Tensor
    types_coords: Tensor
    drawn: Tensor
    pcs_gen: Tensor
    pcs_real: Tensor
    gene_idx: IntArray
    neighbours: IntArray
    section_id: str

    @property
    def n_cells(self) -> int:
        """N: cells the block ran on."""
        return int(self.coords.shape[0])


def reconstruct_hidden(
    model: CTFFlow,
    hidden: Section,
    cfg: Config,
    *,
    seed: int,
    gene_pool: IntArray | None = None,
) -> Reconstruction:
    """Reconstruct a hidden training section down the generation path. See :class:`Reconstruction`.

    Parameters
    ----------
    model
        The model being trained. Its retrieval index is queried with the hidden section
        excluded; its parameters are the only things gradients reach.
    hidden
        The section to reconstruct — a *training* section, chosen by :class:`LOSOScheduler`.
    cfg
        Supplies ``loso_max_cells``, ``genes_per_step``, ``ode_steps``, ``expr_pca_dim`` and
        ``metric_eps``.
    seed
        Explicit (Convention 3): fixes the cell subsample, the gene subsample and the
        retrieval's dropout stream. Two calls with the same seed and the same weights return
        bitwise identical tensors.
    gene_pool
        Panel columns the block may draw from; ``None`` is the whole panel. Threaded through
        so that a zero-shot run's held-out genes stay held out here too.

    Notes
    -----
    ``apply_dropout=False`` on the retrieval query. The gap-aware curriculum exists to make
    training harder in a controlled way, and the metric-aware terms are a *measurement* of
    the reconstruction; randomising the evidence between two steps would put that randomness
    straight into the loss.
    """
    # `section_seed`, not the builtin `hash()`: the latter is salted per process, so the
    # same declared seed drew a different stream in every run (Convention 3).
    gen = np.random.default_rng([int(seed), section_seed(hidden.section_id)])
    n_cells = min(int(cfg.loso_max_cells), hidden.n_cells)
    rows = np.sort(gen.choice(hidden.n_cells, size=n_cells, replace=False)).astype(np.intp)
    pool = (
        np.arange(model.stats.n_genes, dtype=np.intp)
        if gene_pool is None
        else np.asarray(gene_pool, dtype=np.intp)
    )
    n_genes = min(int(cfg.genes_per_step), int(pool.size))
    gene_idx = np.sort(gen.choice(pool, size=n_genes, replace=False)).astype(np.intp)

    xyz = to_xyz(hidden)[rows].astype(np.float64)
    xyz_t = torch.from_numpy(xyz.astype(np.float32))
    cell_type = torch.from_numpy(np.asarray(hidden.cell_type[rows], dtype=np.int64))
    region = (
        None
        if hidden.region is None
        else torch.from_numpy(np.asarray(hidden.region[rows], dtype=np.int64))
    )
    index = model.data.index
    neighbours, _ = index.query(
        xyz,
        {float(hidden.z)},
        seed=int(gen.integers(0, 2**31 - 1)),
        apply_dropout=False,
    )
    tokens, mask = index.neighbour_tokens(xyz, neighbours)
    cond, _ = model.conditioning(xyz_t, xyz_t, cell_type, region, tokens, mask)

    real = torch.from_numpy(
        np.asarray(hidden.counts[rows][:, gene_idx].todense(), dtype=np.float32)
    )
    eps = float(cfg.metric_eps)
    totals = torch.clamp(real.sum(dim=1, keepdim=True), min=1.0)
    reference = float(np.median(totals.numpy()))

    # The generation path, at the real positions: prior -> flow -> decoder.
    h0 = model.prior_latent(xyz, seed=int(seed))
    h = model.flow.sample(h0, cond, int(cfg.ode_steps))
    gene_rows = torch.from_numpy(gene_idx.astype(np.int64))
    gene_emb = model.embeddings.gene(gene_rows)
    mu, theta, pi = model.decoder(h, gene_emb, _matched_size_factor(hidden, rows, model), gene_rows)
    expected, variance = draw_mean_variance(mu, theta, pi, cfg)
    dispersed = _reparameterised_draw(expected, variance, seed=int(seed))
    sampler = sample_zigamma if cfg.decoder == "zigamma" else sample_counts
    with torch.no_grad():
        drawn = sampler(mu, theta, pi, np.random.default_rng([int(seed), n_genes]))
    # Both sides divided by the **real** cell's library size, not by their own row sums; see
    # `normalised_log` for the shortcut that a shared generated denominator opens up. Linear
    # for the autocorrelation and profile terms, `log1p` for the PCA basis the distribution
    # term is stated in — the module docstring of `losses.metric_aware` has the measurement.
    x_gen = normalised_linear(expected, reference, eps, totals=totals)
    x_real = normalised_linear(real, reference, eps, totals=totals)
    noise_var = variance * (float(reference) / torch.clamp(totals, min=eps)) ** 2
    log_gen = normalised_log(dispersed, reference, eps, totals=totals)
    log_real = normalised_log(real, reference, eps, totals=totals)

    # The per-type composition is read at **uniform** points in the slab, not at the cells.
    # See `_uniform_composition` — at the cells it is a per-cell classifier and it runs away.
    types_gen, types_xyz = _uniform_composition(model, hidden, n_cells, gen, eps)
    types_real = torch.nn.functional.one_hot(cell_type, num_classes=int(model.stats.n_types)).to(
        torch.float32
    )
    basis = _pca_basis(model, gene_idx)
    return Reconstruction(
        x_gen=x_gen,
        x_real=x_real,
        coords=xyz_t,
        types_gen=types_gen,
        types_real=types_real,
        types_coords=types_xyz,
        noise_var=noise_var,
        drawn=drawn,
        pcs_gen=_project(log_gen, basis),
        pcs_real=_project(log_real, basis),
        gene_idx=gene_idx,
        neighbours=neighbours,
        section_id=hidden.section_id,
    )


def _uniform_composition(
    model: CTFFlow, hidden: Section, n_points: int, gen: np.random.Generator, eps: float
) -> tuple[Tensor, Tensor]:
    """Read the per-type composition at uniform slab points. ``(M, C)`` and ``(M, 3)``.

    **Not at the cells' own positions, and this is the difference between a loss and a
    runaway.** ``specs/08`` §2 asks for per-type spatial *histogram* agreement, and the model's
    side of that histogram is the point process's expected count per bin, ``integral over the
    bin of lambda_c`` — a quantity that knows where cells are *likely*, not where they *are*.

    Read ``lambda_c`` at the cells instead and the term acquires a second minimiser: a field
    that classifies each individual cell correctly averages, over any bin, to exactly the bin's
    label fractions, so the loss pays nothing for unbounded confidence and the intensity head is
    free to sharpen without limit. It does. Measured over 30 metric steps with the composition
    read at the cells: ``max_c lambda_c / sum_c lambda_c`` 0.62 -> 0.9975, ``max lambda`` up
    230x while ``min lambda`` fell by three orders of magnitude, the shared anatomical field's
    magnitude 0.29 -> 1.08, and the gradient reaching the flow through that field 0.4 -> 913,
    ending in a ``NaN``. Coarsening the bins does not fix it, because the degeneracy is per
    *cell*, not per bin. Uniform points remove it: the field cannot know which of them is a
    cell, so the only way to match a bin is to have the bin's composition right.

    The points are drawn in the hidden section's own slab (its true plane and thickness,
    ``specs/08`` §4) from the block's seeded generator, and there are as many of them as there
    are cells in the subsample.
    """
    plane = section_plane(hidden)
    points = uniform_slab_points(plane, n_points, gen)
    xyz = torch.from_numpy(points.astype(np.float32))
    lam = model.intensity(xyz, model.field(xyz))
    return lam / torch.clamp(lam.sum(dim=1, keepdim=True), min=eps), xyz


def _matched_size_factor(hidden: Section, rows: IntArray, model: CTFFlow) -> Tensor:
    """Return the **real** cells' size factors. ``(N,)`` float32, a constant with no gradient.

    Not :meth:`SizeFactorHead.size_factor`, and the reason is both principled and measured.

    *Principled*: every statistic in ``specs/08`` is computed on library-size-normalised
    expression, so the library size is explicitly not what these terms measure — it is
    ``size_factor_loss``'s (T06) and the detection calibrator's (T09). Giving the generated side
    the real cell's library size also matches the *draw noise*: a cell sequenced twice as deeply
    has relatively less Poisson noise, and comparing a shallow draw against a deep real cell
    would charge the loss for a difference in sampling depth.

    *Measured*: with the predicted size factor in the graph the normalisation cancels it in
    value but not in gradient — the straight-through denominator is the drawn row sum, so the
    exact cancellation of ``mu / sum(mu)`` breaks — and the residual is a nearly flat direction
    with a large derivative. It concentrates on ``size_head``'s last layer, whose output goes
    through an ``exp``: T08 watched the gradient there reach 53 while every other parameter sat
    below 5, and two steps later ``mu`` was ``NaN``.
    """
    totals = np.asarray(hidden.counts[rows].sum(axis=1), dtype=np.float32).reshape(-1)
    factors = np.clip(totals, 1.0, None) / float(model.stats.median_total)
    return torch.from_numpy(factors)


def _reparameterised_draw(expected: Tensor, variance: Tensor, *, seed: int) -> Tensor:
    """Return a dispersed, differentiable stand-in for a count draw. ``(N, G')``.

    ``clamp(mean + sqrt(variance) * epsilon, 0)`` at a fixed standard normal ``epsilon``, so it
    carries the draw's first two moments per (cell, gene), its zeros (from the clamp), and a
    **pathwise** gradient through both moments. The one term that needs it is the distribution
    divergence, which compares point clouds: a mean field is a systematically tighter cloud than
    any sample of it, and matching one to the other would be closed by inflating the means.

    Why not the real sampler on a straight-through estimator, which was the first thing tried:
    the value would be right and the gradient would be the derivative of the transform *at the
    sampled point*, and the transform here is ``log1p``. Its slope at a zero draw is 1 while the
    true derivative of ``E[log1p(X)]`` is a fraction of that, so the estimator overstates what
    raising the mean buys — measured, the generated mean normalised count ran from 4 to 761
    against a real 6.8 while the term's own value rose from 1.43 to 5.56.

    ``epsilon`` is drawn from an explicit seeded generator (Convention 3), on the ``torch``
    side because it is inside the autograd graph — the same split as
    :meth:`LatentFlow.cfm_loss`.
    """
    generator = torch.Generator().manual_seed(int(seed) % (2**31))
    noise = torch.randn(expected.shape, generator=generator, dtype=expected.dtype)
    return torch.clamp(expected + torch.sqrt(torch.clamp(variance, min=0.0)) * noise, min=0.0)


@dataclass(frozen=True)
class _PCABasis:
    """The retrieval index's fitted expression PCA, restricted to one gene subsample."""

    mean: Tensor
    """``(1, G')`` mean of the fitted normalised expression on those genes."""
    basis: Tensor
    """``(G', P)`` the fitted loadings' rows for those genes."""
    scale: Tensor
    """``(1, P)`` per-component standard deviation."""


def _pca_basis(model: CTFFlow, gene_idx: IntArray) -> _PCABasis:
    """Return the **fixed** PCA basis of ``specs/08`` §3, restricted to the block's genes.

    :class:`~spatialcpav25_gen.model.retrieval.ExpressionPCs` is fitted once, on the training
    cells, when the retrieval index is built — so it is exactly the "fixed PCA basis fitted
    once on training data" the spec asks for, and it is already leakage-free by the index's
    own construction. Restricting its rows to a gene subsample is a projection of the same
    fixed map, applied identically to the generated and the real side; it is not a refit, and
    with ``genes_per_step`` at the full panel width it is the map itself.
    """
    pcs = model.data.index.expression_pcs
    return _PCABasis(
        mean=torch.from_numpy(np.asarray(pcs.mean[:, gene_idx], dtype=np.float32)),
        basis=torch.from_numpy(np.asarray(pcs.basis[gene_idx, :], dtype=np.float32)),
        scale=torch.from_numpy(np.asarray(pcs.scale, dtype=np.float32)),
    )


def _project(x: Tensor, basis: _PCABasis) -> Tensor:
    """``(N, G')`` normalised expression -> ``(N, P)`` scores in the fixed basis."""
    return ((x - basis.mean) @ basis.basis) / basis.scale


def metric_aware_terms(
    model: CTFFlow,
    scheduler: LOSOScheduler,
    cfg: Config,
    *,
    step: int,
    seed: int,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """Return the step's **unweighted** metric-aware terms, by name. See :data:`METRIC_TERMS`.

    Reconstructs the epoch's hidden section (:func:`reconstruct_hidden`) and evaluates
    ``specs/08`` §1-§3 on it: ``autocorr``, ``profile`` and ``distribution``. Terms whose
    weight is zero are absent rather than zero, matching ``forward_train``'s contract — a
    missing weight is then a ``KeyError`` and not a silent no-op.

    The distribution term's cell subsample is drawn here, seeded, so that
    :func:`~spatialcpav25_gen.losses.metric_aware.loss_distribution` itself stays
    deterministic and needs no generator of its own.
    """
    epoch = scheduler.epoch_of(step)
    _, hidden = scheduler.epoch_task(epoch)
    recon = reconstruct_hidden(model, hidden, cfg, seed=seed + step, gene_pool=gene_pool)
    coords = recon.coords.detach().numpy()
    terms: dict[str, Tensor] = {}
    if cfg.w_autocorr > 0.0:
        graph = knn_weight_graph(coords, cfg)
        terms["autocorr"] = loss_autocorr(
            recon.x_gen, recon.x_real, graph, graph, cfg, noise_var=recon.noise_var
        )
    if cfg.w_profile > 0.0:
        terms["profile"] = loss_profile(
            recon.x_gen,
            recon.coords,
            recon.x_real,
            recon.coords,
            scheduler.volume,
            cfg,
            types_gen=recon.types_gen,
            types_real=recon.types_real,
            types_coords=recon.types_coords,
        )
    if cfg.w_distribution > 0.0:
        keep = _subsample(recon.n_cells, int(cfg.metric_distribution_points), seed + step)
        terms["distribution"] = loss_distribution(
            recon.pcs_gen[keep], recon.pcs_real[keep].detach(), cfg
        )
    return terms


def _subsample(n: int, limit: int, seed: int) -> Tensor:
    """``(min(n, limit),)`` int64 row indices, drawn without replacement from ``seed``."""
    if n <= limit:
        return torch.arange(n, dtype=torch.int64)
    gen = np.random.default_rng(int(seed))
    return torch.from_numpy(np.sort(gen.choice(n, size=limit, replace=False)).astype(np.int64))


def check_metric_dominance(
    terms: dict[str, Tensor], weights: dict[str, float], cfg: Config, *, step: int
) -> float:
    """Return the metric/reconstruction ratio, warning when it exceeds its cap.

    ``specs/08``'s last "Do NOT": the metric-aware terms must not be weighted above
    reconstruction. The cap is ``Config.metric_dominance_ratio_warn`` and the finding is a
    :class:`~spatialcpav25_gen.losses.metric_aware.MetricDominanceWarning`, not an error, for
    the reason T07 gives for its own: the trajectory is the diagnosis, and a run that ends with
    it firing has a result to report rather than one to use quietly.
    """
    ratio = metric_ratio(terms, weights)
    if not any(name in terms for name in METRIC_TERMS):
        return ratio
    if ratio > float(cfg.metric_dominance_ratio_warn):
        warnings.warn(
            f"metric-aware block: at step {step} the weighted metric terms are {ratio:.2f} of "
            f"the weighted reconstruction terms, above Config.metric_dominance_ratio_warn="
            f"{cfg.metric_dominance_ratio_warn}. specs/08: do not weight these losses above "
            "reconstruction.",
            MetricDominanceWarning,
            stacklevel=2,
        )
    return ratio
