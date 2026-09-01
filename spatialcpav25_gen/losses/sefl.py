"""SEFL: sectioning-equivariant self-supervision (T07).

A section is an **observation operator** applied to one shared tissue, and all valid
operators must be mutually consistent. That is the whole of SEFL: the three losses here need
no ground truth, so five real sections become an unlimited supply of constraints, which is
this project's answer to the small-N problem.

What the model's two branches actually differ by
------------------------------------------------
Each branch conditions the *same physical points* through a different **pose** — the frame in
which that branch's plane is the cutting plane
(:func:`~spatialcpav25_gen.infer.planes.plane_pose`). Under T04's contract that changes
exactly one thing, the triplane lookup, and leaves the three data-frame channels alone:

* the **GRF** is queried in the data frame, so both branches receive *bitwise identical*
  noise along the intersection (T03's G1.2, asserted end-to-end by
  ``tests/test_sefl.py::test_noise_identical_along_intersection``);
* **retrieval** is data-frame too, so both branches see the same donor cells — the query is
  therefore made once and shared, and ``test_retrieval_is_shared_between_branches`` asserts
  that the two poses return the same donors;
* the **Fourier encoding** is data-frame (T04's silent-bug test).

So ``L_cross`` has only the conditioning pathway to correct, not two independent stochastic
processes to reconcile — which is what makes it optimisable at all rather than merely
well-posed.

Anti-collapse
-------------
Every loss here is **asymmetric by construction**: branch 2 runs through the
:class:`EMATeacher` under ``no_grad``. A symmetric consistency loss has the trivial minimiser
"constant field" and will find it. ``Config.sefl_ema_teacher = False`` makes both branches the
live student and is the *documented negative control* — the collapse test is run with it off
and asserted to fail.

The invariant / equivariant distinction
---------------------------------------
:data:`INVARIANT_QUANTITIES` and :data:`EQUIVARIANT_QUANTITIES` are module constants because
``specs/07`` asks for them to be: constraining a quantity from the second column actively
damages the metrics this project is trying to win, and a table in the code is something a
reviewer can check a new loss against. :func:`loss_prog_WRONG` is the deliberate violation,
built to be trained and reported as *worse* (ablation A8); it is excluded from
:data:`SEFL_LOSSES` so it can never be picked up by accident.

Two documented departures from ``specs/07``'s formulae
------------------------------------------------------
Both are changes of *coordinates*, not of content, and both are made for the reason the spec
itself gives for matching decoder parameters directly ("scale-stable"):

``lambda``
    ``specs/07`` §2 writes ``||lambda_1 - lambda_2||^2`` on the intensity. An intensity is
    ~1e-5 cells per micrometre^3, so its squared difference is ~1e-10 and the term would
    contribute nothing at any weight anyone would write down. The term is computed on
    ``log lambda``.
``CE(type_logits_1, softmax(type_logits_2))``
    Implemented as ``KL(p_2 || p_1)``, which is that cross-entropy minus the teacher's own
    entropy. The teacher is detached, so the two have *identical gradients*; the difference is
    that the KL is zero at agreement, so the logged number measures disagreement rather than
    disagreement plus an irreducible floor — which matters because ``L_cross`` is required to
    fall by 60 %.
"""

from __future__ import annotations

import copy
import math
import warnings
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.spatial import cKDTree
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.planes import (
    Plane,
    intersect,
    plane_from_normal,
    plane_pose,
    random_plane_pair,
)
from spatialcpav25_gen.model.embeddings import coexpression_modules
from spatialcpav25_gen.model.field import RotationContext

if TYPE_CHECKING:  # pragma: no cover - import cycle: the trainer imports this module
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "EQUIVARIANT_QUANTITIES",
    "INVARIANT_QUANTITIES",
    "SEFL_DEFAULT_TERMS",
    "SEFL_LOSSES",
    "SEFL_TERMS",
    "BranchOutputs",
    "CollapseWarning",
    "ConsistencyDominanceWarning",
    "EMATeacher",
    "check_collapse",
    "consistency_ratio",
    "cross_terms",
    "evaluate_branch",
    "gene_modules",
    "labels_at",
    "loss_cross",
    "loss_prog",
    "loss_prog_WRONG",
    "loss_thick",
    "mmd2_rbf",
    "morans_i_torch",
    "prog_terms",
    "prog_wrong_terms",
    "sefl_active",
    "sefl_ramp",
    "sefl_terms",
    "sinkhorn_divergence",
    "thick_terms",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

INVARIANT_QUANTITIES: Final[tuple[str, ...]] = (
    "field",
    "cell_identity_at_a_point",
    "expression_given_type_and_region",
    "gene_gene_covariance",
    "module_scores",
    "cells_per_unit_volume",
)
"""Quantities a SEFL loss **may** constrain: properties of the tissue, so two sections through
the same point must agree on them.

====================================  ================================================
``field``                             ``F(x, y, z)`` itself.
``cell_identity_at_a_point``          cell identity and state at a physical point.
``expression_given_type_and_region``  ``p(expression | cell type, region)``.
``gene_gene_covariance``              the gene-gene covariance matrix, module scores.
``module_scores``                     mean molecular-program scores.
``cells_per_unit_volume``             density per **volume** — never per area.
====================================  ================================================
"""

EQUIVARIANT_QUANTITIES: Final[tuple[str, ...]] = (
    "in_plane_morans_i",
    "in_plane_gearys_c",
    "depth_profile",
    "cells_per_unit_area",
    "apparent_structure_size",
    "total_cells_per_section",
)
"""Quantities a SEFL loss must **not** force to be equal across section angles.

These genuinely differ with the cutting angle because tissue is anisotropic: a tangential cut
through cortex has no laminar gradient, and that is correct rather than an error. Forcing them
to match flattens real structure. Their correct treatment is *prediction* — T10's validation
V3 — not a loss.

=============================  =======================================================
``in_plane_morans_i``          in-plane Moran's I (:func:`loss_prog_WRONG` constrains
                               it, on purpose, to be reported as worse).
``in_plane_gearys_c``          in-plane Geary's C.
``depth_profile``              depth / laminar profiles.
``cells_per_unit_area``        cells per unit **area**.
``apparent_structure_size``    apparent structure size and elongation.
``total_cells_per_section``    total cells in a section.
=============================  =======================================================
"""

SEFL_TERMS: Final[tuple[str, ...]] = ("cross", "thick", "prog", "prog_wrong")
"""The named terms SEFL can contribute to the trainer's loss dict, in ``specs/07`` order.

``prog_wrong`` is ablation A8's deliberate negative control (``Config.w_prog_wrong``, shipped at
0 so the term is absent). It is listed here because :func:`consistency_ratio` must count it as
*consistency* rather than as reconstruction — an A8 run whose wrong loss silently landed in the
denominator would report a healthy ratio while the term dominated. It is deliberately **not** in
:data:`SEFL_LOSSES` or :data:`SEFL_DEFAULT_TERMS`."""

SEFL_DEFAULT_TERMS: Final[tuple[str, ...]] = ("cross", "thick", "prog")
"""The terms a legitimate SEFL configuration may use — :data:`SEFL_TERMS` minus A8's control."""

_CROSS_TERMS: Final[tuple[str, ...]] = ("params", "h", "type", "lambda")
_THICK_TERMS: Final[tuple[str, ...]] = ("count", "count_by_type", "state")
_PROG_TERMS: Final[tuple[str, ...]] = ("mmd", "corr", "module")


class CollapseWarning(UserWarning):
    """The generated expression's per-gene variance fell below the collapse threshold.

    ``specs/07`` §2's runtime alarm. A warning and not an error because a *transient* dip
    early in the ramp is survivable and the trajectory is the diagnosis; a persistent one
    means the consistency losses have found the constant field and the run is dead.
    """


class ConsistencyDominanceWarning(UserWarning):
    """The weighted consistency terms exceeded their share of the reconstruction terms.

    ``specs/07`` §5: "Reconstruction terms must dominate throughout. Consistency is a
    regulariser, not the objective."
    """


# --------------------------------------------------------------------------------------
# the teacher
# --------------------------------------------------------------------------------------


class EMATeacher:
    """A stop-gradient exponential moving average of the student, as a second model.

    Args:
        model: the student. The teacher is a deep copy of it whose parameters never require
            gradients, sharing the student's ``TrainingData`` (the retrieval index and the
            pooled count matrix are read-only and would otherwise be duplicated).
        cfg:   supplies ``ema_decay`` and ``sefl_ema_teacher``.
        enabled: overrides ``cfg.sefl_ema_teacher``. ``False`` is the **negative control**:
            :meth:`branch` then returns the live student, both branches carry gradients, the
            loss becomes symmetric, and the collapse it is supposed to prevent is expected.

    Notes
    -----
    A separate module rather than
    :class:`~spatialcpav25_gen.model.spatialcpav25_gen.EMA`'s ``swap_in`` / ``restore``:
    swapping copies into the *live* parameters in place, and a parameter mutated between a
    forward pass and its ``backward`` either poisons the gradient or trips autograd's version
    counter. The teacher has to be a different object for the two branches to coexist in one
    graph.
    """

    def __init__(self, model: CTFFlow, cfg: Config, *, enabled: bool | None = None) -> None:
        self.decay = float(cfg.ema_decay)
        self.enabled = bool(cfg.sefl_ema_teacher if enabled is None else enabled)
        self.module = _clone_model(model)
        self.module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: CTFFlow) -> None:
        """Fold the student's current parameters into the average. No-op when disabled."""
        if not self.enabled:
            return
        live = dict(model.named_parameters())
        for name, parameter in self.module.named_parameters():
            if name in live:
                parameter.mul_(self.decay).add_(live[name].detach(), alpha=1.0 - self.decay)

    def branch(self, model: CTFFlow) -> CTFFlow:
        """Return the module branch 2 runs through: the teacher, or the student when disabled."""
        return self.module if self.enabled else model


def _clone_model(model: CTFFlow) -> CTFFlow:
    """Deep-copy a ``CTFFlow``'s parameters while sharing its (read-only) data."""
    data = model.data
    model.data = cast(Any, None)
    try:
        clone = copy.deepcopy(model)
    finally:
        model.data = data
    clone.data = data
    return clone


# --------------------------------------------------------------------------------------
# one branch of a consistency loss
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchOutputs:
    """Everything a consistency loss compares, evaluated at one pose.

    Attributes
    ----------
    h
        ``(N, d_h)`` cell latent after the flow — "cell state at a physical point".
    log_mu, log_theta, pi_logit
        ``(N, G')`` decoder parameters in the scale-stable coordinates ``specs/07`` §2
        specifies: two branches that agree on all three agree on the distribution.
    type_logits
        ``(N, C)`` log per-type intensity, i.e. ``log p(type | point)`` up to a constant —
        "cell identity at a physical point".
    log_lambda
        ``(N,)`` log **total** intensity, cells per micrometre^3 — "cells per unit volume",
        the invariant-column entry. Not per unit area, which is equivariant.
    lambda_by_type
        ``(N, C)`` per-type intensity, cells per micrometre^3, as the counts are integrated
        from.
    """

    h: Tensor
    log_mu: Tensor
    log_theta: Tensor
    pi_logit: Tensor
    type_logits: Tensor
    log_lambda: Tensor
    lambda_by_type: Tensor


def evaluate_branch(
    module: CTFFlow,
    plane: Plane,
    xyz: FloatArray,
    gene_idx: IntArray,
    cfg: Config,
    *,
    seed: int,
    labels: tuple[Tensor, Tensor | None] | None = None,
    neighbours: IntArray | None = None,
) -> BranchOutputs:
    """Condition ``xyz`` through ``plane``'s pathway. Returns :class:`BranchOutputs`.

    Parameters
    ----------
    module
        The student, or the teacher — the caller decides, and wraps a teacher call in
        ``torch.no_grad()``.
    plane
        The section this branch is. Only its **pose** enters (:func:`plane_pose`): the points
        are given, and a plane's contribution to conditioning is the frame it is cut in.
    xyz
        ``(N, 3)`` float64 data-frame points, micrometres.
    gene_idx
        ``(G',)`` genes the decoder parameters are compared on. The same array in both
        branches — "the shared genes at the shared points".
    cfg
        Supplies ``ode_steps`` and the numeric guards.
    seed
        Explicit (Convention 3): the retrieval query's dropout seed and the i.i.d. prior's.
    labels
        Pre-computed ``(cell_type, region)`` for ``xyz``, from :func:`labels_at`. Shared
        between branches: cell identity at a physical point is an *invariant*, so letting each
        branch invent its own would assert consistency against a moving target.
    neighbours
        Pre-computed ``(N, K)`` retrieval result, likewise shared — retrieval is a data-frame
        channel and both branches would compute the same one.

    Notes
    -----
    All four rotation channels are exercised, including ``planes``: SEFL is the one caller in
    the system that genuinely has a plane to rotate (the trainer's own step does not, and
    says so in ``RotationContext(requires=...)`` — GATE 2's G2.1h).
    """
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"evaluate_branch expects (N, 3) points, got {points.shape}")
    cell_type, region = labels_at(module, points, seed=seed) if labels is None else labels
    centre = np.asarray(module.data.vol.bbox, dtype=np.float64).mean(axis=0)

    rotation = RotationContext(cfg, plane_pose(plane), centre, fields=(module.field,))
    with rotation:
        rotation.planes(np.asarray(plane.origin)[None, :], np.asarray(plane.normal)[None, :])
        xyz_model = rotation.coords(points)
        xyz_retrieval = rotation.retrieval(xyz_model)
        xyz_grf = rotation.grf(xyz_model)
        idx = _neighbours(module, xyz_retrieval, seed=seed) if neighbours is None else neighbours
        tokens, mask = module.data.index.neighbour_tokens(xyz_retrieval, idx)
        model_points = torch.from_numpy(xyz_model.astype(np.float32))
        data_points = torch.from_numpy(points.astype(np.float32))
        cond, _ = module.conditioning(model_points, data_points, cell_type, region, tokens, mask)
        h0 = module.prior_latent(xyz_grf, seed=seed)
        h = module.flow.sample(h0, cond, int(cfg.ode_steps))
        size_factor = module.size_head.size_factor(h)
        gene_rows = torch.from_numpy(gene_idx.astype(np.int64))
        gene_emb = module.embeddings.gene(gene_rows)
        mu, theta, pi = module.decoder(h, gene_emb, size_factor, gene_rows)
        lam = module.intensity(data_points, module.field(model_points), region=region)

    eps = float(cfg.zinb_eps)
    lam = torch.clamp(lam, min=eps)
    log_lam = torch.log(lam)
    pi_c = torch.clamp(pi, min=eps, max=1.0 - eps)
    out = BranchOutputs(
        h=h,
        log_mu=torch.log(torch.clamp(mu, min=float(cfg.zinb_mu_min))),
        log_theta=torch.log(torch.clamp(theta, min=float(cfg.zinb_theta_min))),
        pi_logit=torch.log(pi_c) - torch.log1p(-pi_c),
        type_logits=log_lam,
        log_lambda=torch.log(lam.sum(dim=1)),
        lambda_by_type=lam,
    )
    if cfg.debug_shapes:
        n, g = points.shape[0], gene_idx.shape[0]
        assert out.log_mu.shape == (n, g), out.log_mu.shape
        assert out.log_lambda.shape == (n,), out.log_lambda.shape
    return out


def labels_at(model: CTFFlow, xyz: FloatArray, *, seed: int) -> tuple[Tensor, Tensor | None]:
    """Nearest real cell's type and region for each point. ``(N,)`` int64, ``(N,)`` or ``None``.

    The same nearest-neighbour transfer ``CTFFlow._region_of`` and the benchmark's own
    evaluators use. A generated point has no label of its own and the field is not asked to
    invent one — and because the transfer is a pure function of position it is identical in
    both branches, which is what makes the comparison a statement about the pathway.

    Physically nearest, by a KD-tree over the index's cells, rather than the top of the
    *retrieval score* — the score is a learned conditioning ranking that also weighs the niche
    and the z gap, and "which cell is at this point" is a question about distance. It is also
    an order of magnitude cheaper, which matters: this is called for every point set of every
    SEFL step, and ``specs/07`` §5 caps the block's cost. ``seed`` is accepted for the
    signature's sake and unused — nothing here is stochastic.
    """
    index = model.data.index
    nearest = _nearest_cells(index, np.asarray(xyz, dtype=np.float64))
    cell_type = torch.from_numpy(index.cell_type[nearest].astype(np.int64))
    region = (
        None if index.region is None else torch.from_numpy(index.region[nearest].astype(np.int64))
    )
    return cell_type, region


_TREE_CACHE: weakref.WeakKeyDictionary[Any, cKDTree] = weakref.WeakKeyDictionary()
"""One KD-tree per retrieval index, built on first use. Weak, so it dies with the index."""


def _nearest_cells(index: Any, xyz: FloatArray) -> IntArray:
    """``(N, 3)`` -> ``(N,)`` row index of the physically nearest cell of the index."""
    tree = _TREE_CACHE.get(index)
    if tree is None:
        tree = cKDTree(np.asarray(index.coords, dtype=np.float64))
        _TREE_CACHE[index] = tree
    return np.asarray(tree.query(np.asarray(xyz, dtype=np.float64), k=1)[1], dtype=np.intp)


def _neighbours(model: CTFFlow, xyz: FloatArray, *, seed: int) -> IntArray:
    """``(N, K)`` retrieval result for data-frame points, without the training dropout."""
    idx, _ = model.data.index.query(np.asarray(xyz, dtype=np.float64), set(), seed=seed)
    return idx


# --------------------------------------------------------------------------------------
# helpers shared by the three losses
# --------------------------------------------------------------------------------------


def _seed(gen: np.random.Generator) -> int:
    """One reproducible child seed from the loss's own generator (Convention 3)."""
    return int(gen.integers(0, 2**31 - 1))


def _section_thickness(model: CTFFlow) -> float:
    """Return the training sections' median slab thickness, um — ``h`` in ``L_thick``."""
    return float(np.median([float(s.thickness) for s in model.data.vol.sections]))


def _gene_subsample(
    model: CTFFlow, cfg: Config, gen: np.random.Generator, gene_pool: IntArray | None = None
) -> IntArray:
    """``(G',)`` sorted genes the branches are compared on.

    ``Config.sefl_genes_per_step`` of them, redrawn every step — a separate (smaller) budget
    from the reconstruction path's ``genes_per_step``, for the cost reason that field
    documents.

    ``gene_pool`` is the **leakage guard**, and it is not optional in the pipeline: the
    zero-shot experiment holds genes out by never letting a batch draw them, and a consistency
    loss that quietly drew from the whole panel would give a held-out gene's free residual
    ``r_g`` a gradient and turn a zero-shot number into an in-sample one. ``None`` is the whole
    panel. ``tests/test_expression.py::test_trainer_forwards_the_gene_pool`` catches the leak
    by mutation — it caught this one.
    """
    pool = (
        np.arange(int(model.stats.n_genes), dtype=np.intp)
        if gene_pool is None
        else np.asarray(gene_pool, dtype=np.intp)
    )
    size = min(int(cfg.sefl_genes_per_step), int(pool.size))
    return np.sort(gen.choice(pool, size=size, replace=False)).astype(np.intp)


def _random_plane(
    bbox: FloatArray, cfg: Config, gen: np.random.Generator, thickness: float
) -> Plane:
    """One random plane through the volume, at the same construction the pair uses."""
    plane, _ = random_plane_pair(
        bbox,
        float(cfg.sefl_min_angle_deg),
        float(cfg.sefl_max_angle_deg),
        _seed(gen),
        thickness=thickness,
    )
    return plane


def _slab_sample(
    plane: Plane, n: int, bbox: FloatArray, cfg: Config, gen: np.random.Generator
) -> tuple[FloatArray, float]:
    """``n`` points in slab ∩ bbox and a Monte-Carlo estimate of that region's volume.

    Returns ``((n, 3) float64, volume in micrometre^3)``. The volume is the slab's own times
    the fraction of proposals that fell inside the bounding box, because an oblique slab
    leaves the tissue at its corners and integrating over the *window* volume there would
    over-count exactly the cells that are not in the specimen. Both halves of ``L_thick``'s
    count comparison use this estimator, so the thick slab and its three thin ones are
    clipped consistently.
    """
    box = np.asarray(bbox, dtype=np.float64)
    kept: list[FloatArray] = []
    found = 0
    proposed = 0
    for _ in range(int(cfg.sefl_rejection_max_rounds)):
        if found >= n:
            break
        draw = _uniform_slab(plane, max(4 * (n - found), 16), gen)
        proposed += draw.shape[0]
        inside = np.all((draw >= box[0][None, :]) & (draw <= box[1][None, :]), axis=1)
        kept.append(draw[inside])
        found += int(inside.sum())
    points = np.concatenate(kept, axis=0) if kept else np.zeros((0, 3), dtype=np.float64)
    if points.shape[0] < n:
        raise ValueError(
            f"_slab_sample: only {points.shape[0]} of {n} points fell inside the bounding box; "
            "the slab barely intersects the volume"
        )
    volume = float(plane.slab_volume) * float(found) / float(max(proposed, 1))
    return np.asarray(points[:n], dtype=np.float64), volume


def _uniform_slab(plane: Plane, n: int, gen: np.random.Generator) -> FloatArray:
    """``n`` uniform points in the plane's (unclipped) slab. ``(n, 3)`` float64."""
    extent = np.asarray(plane.half_extent, dtype=np.float64)
    uv = gen.uniform(-extent, extent, size=(int(n), 2))
    w = gen.uniform(-0.5 * plane.thickness, 0.5 * plane.thickness, size=(int(n), 1))
    return np.asarray(plane.to_xyz(uv) + w * np.asarray(plane.normal)[None, :], dtype=np.float64)


def _offset_plane(plane: Plane, offset: float, thickness: float) -> Plane:
    """Return the same window, moved ``offset`` um along the normal, at ``thickness``."""
    return plane_from_normal(
        np.asarray(plane.normal, dtype=np.float64),
        np.asarray(plane.origin, dtype=np.float64)
        + offset * np.asarray(plane.normal, dtype=np.float64),
        np.asarray(plane.half_extent, dtype=np.float64),
        thickness,
    )


def _zero_terms(names: tuple[str, ...]) -> dict[str, Tensor]:
    """All-zero terms, for the skip paths. Scalars, so they sum with the live ones."""
    return {name: torch.zeros((), dtype=torch.float32) for name in names}


def _poisson_consistency(observed: Tensor, expected: Tensor) -> Tensor:
    """Poisson-tolerant relative squared error of two counts. Per element; the caller reduces.

    ``relu(z^2 - 1) / expected`` with ``z = (observed - expected) / sqrt(expected)``: a
    Pearson residual, hinged at one standard deviation and expressed per cell.

    Two properties, and ``L_thick`` needs both.

    *Poisson-tolerant.* A difference the size of counting noise (``|z| <= 1``) costs **exactly
    zero**, not "not much" — which is what ``specs/07``'s "compare with a Poisson-consistent
    relative loss ... do not force equal counts" asks for, and what
    ``test_thick_counts_add`` checks.

    *Relative.* The bare Pearson statistic ``z^2`` grows with the section: a systematic
    relative error ``eps`` costs ``eps^2 N``, so on a 10^4-cell slab a 10 % disagreement scores
    100 while the reconstruction term is ~2 nats. Measured on the fixture at the spec's own
    schedule, the un-normalised term ran to **897** and drove the consistency/reconstruction
    ratio to a median of **4.3** against ``specs/07`` §5's ceiling of 0.5 — the consistency
    block became the objective, which is precisely what §5 forbids. Dividing by the expected
    count leaves ``eps^2``, so ``w_thick`` means the same thing on a slab of any size — the
    same argument that makes ``zinb_nll`` a mean.
    """
    expected_c = torch.clamp(expected, min=1.0)
    z2 = (observed - expected) ** 2 / expected_c
    return torch.relu(z2 - 1.0) / expected_c


# --------------------------------------------------------------------------------------
# L_cross
# --------------------------------------------------------------------------------------


def cross_terms(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """Return the named parts of ``L_cross`` (``specs/07`` §2). All scalars ``()``.

    ``params`` (the three decoder terms summed), ``h``, ``type`` and ``lambda``, logged
    separately "so a collapse in one is visible". :func:`loss_cross` is their sum.

    The recipe, from the spec: draw a plane pair at a dihedral angle in
    ``[sefl_min_angle_deg, sefl_max_angle_deg]``; intersect them; **skip** (all terms zero) if
    they do not meet or the segment is shorter than ``sefl_min_segment_nn_multiple`` median
    nearest-neighbour distances; sample ``sefl_n_line_points`` points along it; condition
    through plane 1 (student, gradients) and plane 2 (EMA teacher, stop-gradient).
    """
    box = np.asarray(bbox, dtype=np.float64)
    thickness = _section_thickness(model)
    p1, p2 = random_plane_pair(
        box,
        float(cfg.sefl_min_angle_deg),
        float(cfg.sefl_max_angle_deg),
        _seed(gen),
        thickness=thickness,
    )
    segment = intersect(p1, p2)
    floor = float(cfg.sefl_min_segment_nn_multiple) * float(model.data.index.median_nn_dist)
    if segment is None or segment.length < floor:
        return _zero_terms(_CROSS_TERMS)

    points = segment.points(int(cfg.sefl_n_line_points))
    seed = _seed(gen)
    gene_idx = _gene_subsample(model, cfg, gen, gene_pool)
    labels = labels_at(model, points, seed=seed)
    neighbours = _neighbours(model, points, seed=seed)

    student = evaluate_branch(
        model, p1, points, gene_idx, cfg, seed=seed, labels=labels, neighbours=neighbours
    )
    other = _teacher_branch(
        model,
        model_ema,
        p2,
        points,
        gene_idx,
        cfg,
        seed=seed,
        labels=labels,
        neighbours=neighbours,
    )

    params = (
        torch.mean((student.log_mu - other.log_mu) ** 2)
        + torch.mean((student.log_theta - other.log_theta) ** 2)
        + torch.mean((student.pi_logit - other.pi_logit) ** 2)
    )
    target = torch.softmax(other.type_logits, dim=1)
    log_target = torch.log_softmax(other.type_logits, dim=1)
    log_student = torch.log_softmax(student.type_logits, dim=1)
    return {
        "params": params,
        "h": torch.mean((student.h - other.h) ** 2),
        "type": torch.mean((target * (log_target - log_student)).sum(dim=1)),
        "lambda": torch.mean((student.log_lambda - other.log_lambda) ** 2),
    }


def _teacher_branch(
    model: CTFFlow,
    model_ema: EMATeacher,
    plane: Plane,
    points: FloatArray,
    gene_idx: IntArray,
    cfg: Config,
    *,
    seed: int,
    labels: tuple[Tensor, Tensor | None] | None = None,
    neighbours: IntArray | None = None,
) -> BranchOutputs:
    """Evaluate branch 2: the teacher under ``no_grad``, or the student when disabled.

    The one place the anti-collapse asymmetry is implemented, so there is one place to read
    it. With ``Config.sefl_ema_teacher = False`` this returns a *gradient-carrying* student
    branch and the loss becomes symmetric — the negative control, not a fallback.
    """
    if not model_ema.enabled:
        return evaluate_branch(
            model, plane, points, gene_idx, cfg, seed=seed, labels=labels, neighbours=neighbours
        )
    with torch.no_grad():
        return evaluate_branch(
            model_ema.module,
            plane,
            points,
            gene_idx,
            cfg,
            seed=seed,
            labels=labels,
            neighbours=neighbours,
        )


def loss_cross(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
) -> Tensor:
    """Plane-intersection consistency. Scalar ``()``; zero when the planes do not meet.

    See :func:`cross_terms` for the parts and the construction. **Why it earns its place:**
    the intersection line crosses the tissue's depth axis at arbitrary angles, so agreement
    along it forces every marker's depth gradient to be geometrically coherent in 3D rather
    than fitted per-plane — a direct regulariser on the marker-field and marker-depth metrics.
    """
    return _sum_terms(cross_terms(model, model_ema, bbox, cfg, gen))


def _sum_terms(terms: dict[str, Tensor]) -> Tensor:
    """Sum a term dict into the scalar the trainer weights."""
    return torch.stack(list(terms.values())).sum()


# --------------------------------------------------------------------------------------
# L_thick
# --------------------------------------------------------------------------------------


def thick_terms(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """Return the named parts of ``L_thick`` (``specs/07`` §3). All scalars ``()``.

    ``S(Pi, 3h) == aggregate[S(Pi, h, -h), S(Pi, h, 0), S(Pi, h, +h)]`` with ``3`` =
    ``Config.thickness_ratio``. The three ways to get the aggregation wrong are all guarded:

    ``count``
        Cell counts **add**. The term is the Poisson-consistent
        ``(N_thick - sum N_thin)^2 / sum N_thin``, whose expectation is O(1) rather than O(N)
        when the two agree. Nothing here asks the two for equal counts.
    ``count_by_type``
        The same statistic per cell type, summed — a thick section is not allowed to add its
        cells to the wrong types.
    ``state``
        For single-cell data the thick section is a **union** of cells with no per-cell
        correspondence, so the comparison is between *distributions*: an entropic Sinkhorn
        divergence on expression PCs at a blur of ``sefl_sinkhorn_blur_multiple`` median
        nearest-neighbour distances in PC space. For ``section_granularity = "binned"``
        expression sums within a bin, and binned totals are compared directly.

    Common random numbers, and what that leaves the term measuring
    --------------------------------------------------------------
    The thin slabs' integration points are the thick slab's, **partitioned** by which sub-slab
    each point falls in. That is the whole aggregation identity made manifest: the three thin
    slabs *are* the thick one, so their integration domains partition it and their volumes add
    to its.

    Measured, the alternative is not viable. With independent draws per slab the term is
    dominated by its own Monte-Carlo error — the estimator's discrepancy is percent-scale on a
    count of thousands, which the Poisson normalisation does not tame — and it ranged over
    **0.007 to 250** across a 500-step run against a reconstruction term of 2.1, i.e. the
    consistency block was optimising sampling noise. With common random numbers the estimator
    error cancels exactly and what is left is the quantity of interest: whether the **student**
    and the **teacher** agree on the intensity integrated over the same tissue. That is the
    honest content of thickness consistency in a *continuous-field* model — coarse-graining is
    structural here, guaranteed by T05 integrating over the slab **volume**, and
    ``test_thick_counts_add`` is what checks that structure. What is not structural, and what
    this term supplies, is agreement between the two branches.
    """
    box = np.asarray(bbox, dtype=np.float64)
    thin_h = _section_thickness(model)
    ratio = int(cfg.thickness_ratio)
    thick_h = thin_h * ratio
    base = _random_plane(box, cfg, gen, thick_h)
    thin_planes = [
        _offset_plane(base, (index - (ratio - 1) / 2.0) * thin_h, thin_h) for index in range(ratio)
    ]

    n_mc = int(cfg.layout_n_mc)
    n_state = int(cfg.sefl_max_distribution_points)
    gene_idx = _gene_subsample(model, cfg, gen, gene_pool)
    seed = _seed(gen)

    try:
        thick_mc, thick_volume = _slab_sample(base, n_mc, box, cfg, gen)
        thick_state, _ = _slab_sample(base, n_state, box, cfg, gen)
    except ValueError:
        # The drawn plane grazes the volume; there is no slab to coarse-grain.
        return _zero_terms(_THICK_TERMS)
    thin_mc = _partition_by_slab(base, thin_planes, thick_mc)
    thin_state = _partition_by_slab(base, thin_planes, thick_state)

    thick_lambda = _intensity_at(model, base, thick_mc, cfg, seed=seed)
    thin_by_type = torch.zeros_like(thick_lambda[0])
    for plane, rows in zip(thin_planes, thin_mc, strict=True):
        if rows.size == 0:
            continue
        # Each sub-slab's volume is the thick slab's times its share of the shared points —
        # the Monte-Carlo estimate of the sub-slab's volume from the very same draw.
        share = float(rows.size) / float(thick_mc.shape[0])
        with torch.no_grad():
            values = _intensity_at(
                _teacher_module(model, model_ema), plane, thick_mc[rows], cfg, seed=seed
            )
        thin_by_type = thin_by_type + values.mean(dim=0) * thick_volume * share

    thick_by_type = thick_lambda.mean(dim=0) * thick_volume
    if model_ema.enabled:
        thin_by_type = thin_by_type.detach()

    count = _poisson_consistency(thick_by_type.sum(), thin_by_type.sum())
    count_by_type = _poisson_consistency(thick_by_type, thin_by_type).sum()

    state = _thick_state_term(
        model, model_ema, base, thin_planes, thick_state, thin_state, gene_idx, cfg, seed=seed
    )
    return {"count": count, "count_by_type": count_by_type, "state": state}


def _partition_by_slab(
    thick: Plane, thin_planes: list[Plane], points: FloatArray
) -> list[IntArray]:
    """Split ``points`` between the thin sub-slabs they fall in. One index array per slab.

    The signed offset along the shared normal decides, so the partition is exact and every
    point lands in exactly one sub-slab (the outermost bins absorb the float64 edge cases at
    ``+-thickness/2``). This is what makes ``L_thick``'s count identity free of Monte-Carlo
    error — see :func:`thick_terms`.
    """
    normal = np.asarray(thick.normal, dtype=np.float64)
    offset = (np.asarray(points, dtype=np.float64) - np.asarray(thick.origin)) @ normal
    edges = [
        float(np.asarray(plane.origin - thick.origin) @ normal) + 0.5 * float(plane.thickness)
        for plane in thin_planes[:-1]
    ]
    which = np.digitize(offset, edges)
    return [np.nonzero(which == index)[0].astype(np.intp) for index in range(len(thin_planes))]


def _teacher_module(model: CTFFlow, model_ema: EMATeacher) -> CTFFlow:
    """Branch 2's module: the teacher, or the student under the negative control."""
    return model_ema.branch(model)


def _intensity_at(
    module: CTFFlow, plane: Plane, xyz: FloatArray, cfg: Config, *, seed: int
) -> Tensor:
    """``(N, C)`` per-type intensity at ``xyz`` through ``plane``'s pose. Cells per um^3.

    Only the intensity head is evaluated — the count identity is a statement about the
    layout, and running the flow and the decoder for it would cost an order of magnitude more
    for a number they do not enter.
    """
    points = np.asarray(xyz, dtype=np.float64)
    _, region = labels_at(module, points, seed=seed)
    centre = np.asarray(module.data.vol.bbox, dtype=np.float64).mean(axis=0)
    rotation = RotationContext(cfg, plane_pose(plane), centre, fields=(module.field,))
    with rotation:
        rotation.planes(np.asarray(plane.origin)[None, :], np.asarray(plane.normal)[None, :])
        xyz_model = rotation.coords(points)
        rotation.retrieval(xyz_model)
        rotation.grf(xyz_model)
        data_points = torch.from_numpy(points.astype(np.float32))
        model_points = torch.from_numpy(xyz_model.astype(np.float32))
        return module.intensity(data_points, module.field(model_points), region=region)


def _thick_state_term(
    model: CTFFlow,
    model_ema: EMATeacher,
    thick_plane: Plane,
    thin_planes: list[Plane],
    thick_points: FloatArray,
    thin_points: list[IntArray],
    gene_idx: IntArray,
    cfg: Config,
    *,
    seed: int,
) -> Tensor:
    """Return the distribution half of ``L_thick``. Scalar ``()``.

    Single-cell: a Sinkhorn divergence between the thick section's cell states and the union
    of the thin sections'. **No per-cell correspondence is attempted** — the two point sets
    are different draws of different sizes and matching them cell by cell would be a
    fabrication (``specs/07`` §3, and the "Do NOT" list).
    """
    student = evaluate_branch(
        model,
        thick_plane,
        thick_points,
        gene_idx,
        cfg,
        seed=seed,
        labels=labels_at(model, thick_points, seed=seed),
        neighbours=_neighbours(model, thick_points, seed=seed),
    )
    # The thin sections are evaluated as **one** branch over the union of their points. That
    # is not an approximation: `_offset_plane` keeps the normal, so all three thin planes and
    # the thick one share a pose, and a branch depends on its plane only through that pose.
    # It is also the definition of the aggregate — "a thick section is a union of thin ones" —
    # and it pays a branch's fixed cost once instead of `thickness_ratio` times.
    union = thick_points[np.concatenate(thin_points, axis=0)]
    other = _teacher_branch(
        model,
        model_ema,
        thin_planes[0],
        union,
        gene_idx,
        cfg,
        seed=seed,
        labels=labels_at(model, union, seed=seed),
        neighbours=_neighbours(model, union, seed=seed),
    )
    if cfg.section_granularity == "binned":
        thick_total = _binned_totals(thick_plane, thick_points, student.log_mu.exp(), cfg)
        thin_total = _binned_totals(thick_plane, union, other.log_mu.exp(), cfg)
        scale = torch.clamp(thin_total.abs().mean(), min=1e-6)
        return torch.mean(((thick_total - thin_total) / scale) ** 2)

    thick_pcs = _expression_pcs_torch(model, student.log_mu, gene_idx)
    thin_pcs = _expression_pcs_torch(model, other.log_mu, gene_idx)
    blur = float(cfg.sefl_sinkhorn_blur_multiple) * _median_nn_distance(thin_pcs.detach())
    return sinkhorn_divergence(thick_pcs, thin_pcs, blur, cfg)


def _binned_totals(plane: Plane, xyz: FloatArray, values: Tensor, cfg: Config) -> Tensor:
    """Sum ``values`` into an in-plane grid of ``Config.sefl_thick_grid`` squares. ``(B, G')``.

    For binned or spot data, where "expression sums within a bin" is literally what the assay
    does, so the aggregation is a sum and not a distribution comparison.
    """
    grid = int(cfg.sefl_thick_grid)
    uv = plane.to_uv(np.asarray(xyz, dtype=np.float64))
    extent = np.asarray(plane.half_extent, dtype=np.float64)
    cell = np.clip(((uv + extent) / (2.0 * extent) * grid).astype(np.int64), 0, grid - 1)
    flat = torch.from_numpy((cell[:, 0] * grid + cell[:, 1]).astype(np.int64))
    out = torch.zeros((grid * grid, values.shape[1]), dtype=values.dtype)
    return out.index_add(0, flat, values)


def _expression_pcs_torch(model: CTFFlow, log_mu: Tensor, gene_idx: IntArray) -> Tensor:
    """Project decoded expression onto the training expression PCA. ``(N, G')`` -> ``(N, d)``.

    The **fitted** basis of :class:`~spatialcpav25_gen.model.retrieval.ExpressionPCs`,
    restricted to the genes this step drew, applied after the same input-side transform the
    index uses (library-size normalise, ``log1p``). Restricting a basis is not the same thing
    as refitting one on the subset, and it is the right choice here: the two branches must land
    in *one* space for their distance to mean anything, and refitting per step would move the
    space under the loss.
    """
    pcs = model.data.index.expression_pcs
    mu = torch.exp(log_mu)
    totals = torch.clamp(mu.sum(dim=1, keepdim=True), min=1e-6)
    normalised = torch.log1p(mu / totals * float(np.median(model.data.total_counts)))
    mean = torch.from_numpy(pcs.mean[:, gene_idx].astype(np.float32))
    basis = torch.from_numpy(pcs.basis[gene_idx, :].astype(np.float32))
    scale = torch.from_numpy(pcs.scale.astype(np.float32))
    return (normalised - mean) @ basis / scale


def _median_nn_distance(points: Tensor) -> float:
    """Median nearest-neighbour distance of a point cloud, in its own units."""
    values = points.detach().numpy().astype(np.float64)
    if values.shape[0] < 2:
        return 1.0
    distances = cKDTree(values).query(values, k=2)[0][:, 1]
    return float(max(np.median(distances), 1e-6))


def loss_thick(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
) -> Tensor:
    """Thickness coarse-graining consistency. Scalar ``()``. See :func:`thick_terms`.

    Beyond regularisation this is a cross-technology harmonisation mechanism: thin imaging
    sections and thick capture-based sections become consistent observations of one field
    (validation V4, T10).
    """
    return _sum_terms(thick_terms(model, model_ema, bbox, cfg, gen))


# --------------------------------------------------------------------------------------
# L_prog
# --------------------------------------------------------------------------------------


def prog_terms(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    conditional: bool = True,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """Return the named parts of ``L_prog`` (``specs/07`` §4). All scalars ``()``.

    For two random sampling angles, and for each ``(cell type c, region r)`` present in both
    with at least ``Config.sefl_min_stratum_cells`` cells on either side: ``mmd`` (multi-
    bandwidth RBF MMD^2 between the two strata's expression), ``corr`` (Frobenius error
    between their gene-gene correlation matrices, divided by ``G'^2`` so a weight means the
    same thing on any panel) and ``module`` (squared error between their mean
    molecular-program scores).

    Parameters
    ----------
    conditional
        ``True`` is the loss. ``False`` pools every cell into one stratum and is the **wrong**
        variant ``specs/07`` §4 warns about: different planes genuinely sample different
        mixtures of types and regions, so matching the marginals forces the model to
        hallucinate a homogeneous tissue. It exists to be measured
        (``test_prog_conditioning``) and is not reachable from :data:`SEFL_LOSSES`.
    """
    box = np.asarray(bbox, dtype=np.float64)
    thickness = _section_thickness(model)
    p1, p2 = random_plane_pair(
        box,
        float(cfg.sefl_min_angle_deg),
        float(cfg.sefl_max_angle_deg),
        _seed(gen),
        thickness=thickness,
    )
    n = int(cfg.sefl_patch_cells)
    seed = _seed(gen)
    gene_idx = _gene_subsample(model, cfg, gen, gene_pool)
    try:
        points1 = p1.sample_points(n, box, _seed(gen), cfg=cfg)
        points2 = p2.sample_points(n, box, _seed(gen), cfg=cfg)
    except ValueError:
        return _zero_terms(_PROG_TERMS)

    labels1 = labels_at(model, points1, seed=seed)
    labels2 = labels_at(model, points2, seed=seed)
    neighbours1 = _neighbours(model, points1, seed=seed)
    neighbours2 = _neighbours(model, points2, seed=seed)
    student = evaluate_branch(
        model, p1, points1, gene_idx, cfg, seed=seed, labels=labels1, neighbours=neighbours1
    )
    other = _teacher_branch(
        model,
        model_ema,
        p2,
        points2,
        gene_idx,
        cfg,
        seed=seed,
        labels=labels2,
        neighbours=neighbours2,
    )

    modules = torch.from_numpy(gene_modules(model, cfg, seed=int(cfg.seed))[gene_idx])
    strata = _strata(labels1, labels2, cfg, conditional=conditional)
    if not strata:
        return _zero_terms(_PROG_TERMS)

    totals = {name: torch.zeros((), dtype=torch.float32) for name in _PROG_TERMS}
    for rows1, rows2 in strata:
        expr1 = student.log_mu[rows1]
        expr2 = other.log_mu[rows2]
        totals["mmd"] = totals["mmd"] + mmd2_rbf(_cap(expr1, cfg, gen), _cap(expr2, cfg, gen), cfg)
        totals["corr"] = totals["corr"] + torch.mean(
            (_correlation(expr1) - _correlation(expr2)) ** 2
        )
        totals["module"] = totals["module"] + torch.sum(
            (_module_scores(expr1, modules, cfg) - _module_scores(expr2, modules, cfg)) ** 2
        )
    return {name: value / float(len(strata)) for name, value in totals.items()}


def _strata(
    labels1: tuple[Tensor, Tensor | None],
    labels2: tuple[Tensor, Tensor | None],
    cfg: Config,
    *,
    conditional: bool,
) -> list[tuple[Tensor, Tensor]]:
    """Row indices of every ``(cell type, region)`` stratum present on both sides.

    ``conditional=False`` returns the single pooled stratum — the wrong variant. Strata with
    fewer than ``Config.sefl_min_stratum_cells`` cells on *either* side are skipped: below
    that the MMD and the gene-gene correlation are estimating a distribution from a handful of
    points and the loss is fitting their sampling noise.
    """
    type1, region1 = labels1
    type2, region2 = labels2
    if not conditional:
        return [(torch.arange(type1.shape[0]), torch.arange(type2.shape[0]))]
    minimum = int(cfg.sefl_min_stratum_cells)
    key1 = _stratum_key(type1, region1)
    key2 = _stratum_key(type2, region2)
    out: list[tuple[Tensor, Tensor]] = []
    for key in torch.unique(key1).tolist():
        rows1 = torch.nonzero(key1 == key, as_tuple=False).squeeze(1)
        rows2 = torch.nonzero(key2 == key, as_tuple=False).squeeze(1)
        if int(rows1.numel()) >= minimum and int(rows2.numel()) >= minimum:
            out.append((rows1, rows2))
    return out


def _stratum_key(cell_type: Tensor, region: Tensor | None) -> Tensor:
    """One int64 code per ``(cell type, region)`` pair. ``(N,)`` -> ``(N,)``."""
    if region is None:
        return cell_type
    return cell_type * (int(region.max().item()) + 1) + region


def _cap(values: Tensor, cfg: Config, gen: np.random.Generator) -> Tensor:
    """Subsample rows to ``Config.sefl_max_distribution_points``; MMD is quadratic in them."""
    limit = int(cfg.sefl_max_distribution_points)
    if values.shape[0] <= limit:
        return values
    rows = gen.choice(int(values.shape[0]), size=limit, replace=False)
    return values[torch.from_numpy(np.sort(rows).astype(np.int64))]


def _correlation(values: Tensor) -> Tensor:
    """Gene-gene Pearson correlation of ``(N, G')`` expression. ``(G', G')``."""
    centred = values - values.mean(dim=0, keepdim=True)
    scale = torch.clamp(centred.pow(2).sum(dim=0).sqrt(), min=1e-6)
    normalised = centred / scale[None, :]
    return normalised.T @ normalised


def _module_scores(values: Tensor, modules: Tensor, cfg: Config) -> Tensor:
    """Mean expression within each molecular program. ``(N, G')``, ``(G',)`` -> ``(M,)``.

    The mean over the stratum's cells of the mean over the module's genes, so a program's
    score is on the same scale whatever its size. Programs with fewer than
    ``Config.sefl_module_min_genes`` genes in this step's subsample are dropped (their score
    would be a gene, which the MMD term already sees).
    """
    scores: list[Tensor] = []
    for label in torch.unique(modules).tolist():
        columns = modules == label
        if int(columns.sum()) >= int(cfg.sefl_module_min_genes):
            scores.append(values[:, columns].mean())
    if not scores:
        return torch.zeros((0,), dtype=values.dtype)
    return torch.stack(scores)


def loss_prog(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    conditional: bool = True,
) -> Tensor:
    """Molecular-program invariance across sampling angles. Scalar ``()``."""
    return _sum_terms(prog_terms(model, model_ema, bbox, cfg, gen, conditional=conditional))


def loss_prog_WRONG(  # noqa: N802 - the shouted name is the point
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    gene_pool: IntArray | None = None,
) -> Tensor:
    """**DELIBERATE NEGATIVE CONTROL — never train with this.** Scalar ``()``.

    Forces in-plane Moran's I to agree across two sampling angles, i.e. treats a quantity from
    :data:`EQUIVARIANT_QUANTITIES` as if it were invariant. It is wrong on purpose: a
    tangential cut through a laminar structure *has* a different in-plane autocorrelation from
    a perpendicular one, and that difference is signal.

    It exists so the paper can show a trained ablation (A8) performing worse, which makes the
    invariant/equivariant distinction concrete and pre-empts "why didn't you just constrain
    everything". Excluded from :data:`SEFL_LOSSES`;
    ``test_wrong_loss_is_not_in_the_registry`` asserts that, and the coverage matrix lists it
    as one of the project's two deliberate negative controls. Do not delete it as unused.
    """
    box = np.asarray(bbox, dtype=np.float64)
    p1, p2 = random_plane_pair(
        box,
        float(cfg.sefl_min_angle_deg),
        float(cfg.sefl_max_angle_deg),
        _seed(gen),
        thickness=_section_thickness(model),
    )
    n = min(int(cfg.sefl_patch_cells), int(cfg.sefl_max_distribution_points))
    seed = _seed(gen)
    gene_idx = _gene_subsample(model, cfg, gen, gene_pool)
    try:
        points1 = p1.sample_points(n, box, _seed(gen), cfg=cfg)
        points2 = p2.sample_points(n, box, _seed(gen), cfg=cfg)
    except ValueError:
        return torch.zeros((), dtype=torch.float32)
    student = evaluate_branch(
        model,
        p1,
        points1,
        gene_idx,
        cfg,
        seed=seed,
        neighbours=_neighbours(model, points1, seed=seed),
    )
    other = _teacher_branch(
        model,
        model_ema,
        p2,
        points2,
        gene_idx,
        cfg,
        seed=seed,
        neighbours=_neighbours(model, points2, seed=seed),
    )
    i1 = morans_i_torch(student.log_mu, p1.to_uv(points1), cfg)
    i2 = morans_i_torch(other.log_mu, p2.to_uv(points2), cfg)
    return torch.mean((i1 - i2) ** 2)


def prog_wrong_terms(
    model: CTFFlow,
    model_ema: EMATeacher,
    bbox: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """A8's negative control as a named part, so the trainer can weight it. ``{"morans": ()}``.

    :func:`sefl_terms` needs a :class:`TermsFn`, and :func:`loss_prog_WRONG` returns a bare
    scalar. This is the adapter and nothing more — it adds no arithmetic, so the value the
    trainer weights is exactly what ``loss_prog_WRONG`` computes and the A8 arm is the published
    negative control rather than a variant of it.
    """
    return {
        "morans": loss_prog_WRONG(model, model_ema, bbox, cfg, gen, gene_pool=gene_pool),
    }


LossFn = Callable[["CTFFlow", EMATeacher, FloatArray, Config, np.random.Generator], Tensor]


class TermsFn(Protocol):
    """A ``*_terms`` builder: the spec's five positional arguments plus the leakage guard."""

    def __call__(
        self,
        model: CTFFlow,
        model_ema: EMATeacher,
        bbox: FloatArray,
        cfg: Config,
        gen: np.random.Generator,
        *,
        gene_pool: IntArray | None = ...,
    ) -> dict[str, Tensor]:
        """Return the loss's named parts, all scalars."""
        ...


SEFL_LOSSES: Final[dict[str, LossFn]] = {
    "cross": loss_cross,
    "thick": loss_thick,
    "prog": loss_prog,
}
"""The default loss registry: the three losses training may use, keyed by their weight name
(``Config.w_cross`` and friends). :func:`loss_prog_WRONG` is **not** here, and the
unconditional variant of :func:`loss_prog` is reachable only through its keyword."""


# --------------------------------------------------------------------------------------
# schedule, ratio and alarm
# --------------------------------------------------------------------------------------


def sefl_active(step: int, cfg: Config) -> bool:
    """Whether the SEFL block runs at ``step``: every ``Config.sefl_every_n_steps``-th one."""
    return int(step) % max(int(cfg.sefl_every_n_steps), 1) == 0


def sefl_ramp(step: int, total_steps: int, cfg: Config) -> float:
    """Return the multiplier on every SEFL weight at ``step``. In ``[0, 1]``.

    Zero for the first ``Config.sefl_warmup_frac`` of the run — early on the consistency
    losses are satisfiable by degenerate solutions — then linear to 1 over the following
    ``Config.sefl_ramp_frac``, then 1. A zero-length ramp is a step, not a division by zero.
    """
    total = max(int(total_steps), 1)
    warmup = float(cfg.sefl_warmup_frac) * total
    ramp = float(cfg.sefl_ramp_frac) * total
    if step < warmup:
        return 0.0
    if ramp <= 0.0:
        return 1.0
    return float(min((step - warmup) / ramp, 1.0))


def consistency_ratio(terms: dict[str, Tensor], weights: dict[str, float]) -> float:
    """``sum(weighted consistency) / sum(weighted reconstruction)``. Returns a float.

    The denominator is every non-SEFL, non-diagnostic term, which is what "reconstruction
    terms must dominate" means in a loss dict that also carries ``cfm``, ``size`` and
    ``layout``. Returns ``inf`` when the denominator is zero, so a caller cannot read a
    divide-by-zero as a healthy ratio.
    """
    consistency = 0.0
    reconstruction = 0.0
    for name, value in terms.items():
        if name.startswith("diag_"):
            continue
        weighted = float(weights.get(name, 0.0)) * float(value)
        if name in SEFL_TERMS:
            consistency += weighted
        else:
            reconstruction += weighted
    if reconstruction == 0.0:
        return float("inf")
    return consistency / reconstruction


def check_collapse(generated_variance: float, real_variance: float, step: int, cfg: Config) -> bool:
    """Run ``specs/07`` §2's collapse alarm. Returns ``True`` if it fired.

    Warns :class:`CollapseWarning`, naming the step, when the mean per-gene variance of
    generated expression is below ``Config.sefl_collapse_warn_fraction`` of the real cells'.
    Both arguments must be variances of **drawn counts** — a variance of ``mu`` against a
    variance of counts is not the same quantity and would fire on a healthy model.
    """
    if real_variance <= 0.0:
        return False
    ratio = float(generated_variance) / float(real_variance)
    if ratio >= float(cfg.sefl_collapse_warn_fraction):
        return False
    warnings.warn(
        f"COLLAPSE WARNING at step {step}: the mean per-gene variance of generated expression "
        f"is {ratio:.3f} of the real cells' (alarm at "
        f"{cfg.sefl_collapse_warn_fraction}). The consistency losses may have found the "
        "constant field; check that Config.sefl_ema_teacher is on and that the SEFL weights "
        "have not overtaken reconstruction.",
        CollapseWarning,
        stacklevel=2,
    )
    return True


def sefl_terms(
    model: CTFFlow,
    model_ema: EMATeacher,
    cfg: Config,
    gen: np.random.Generator,
    *,
    gene_pool: IntArray | None = None,
) -> dict[str, Tensor]:
    """Every non-zero SEFL term plus its parts as ``diag_`` entries. Used by the trainer.

    Terms whose weight is zero are **absent**, not zero, so switching a term off costs nothing
    rather than costing a forward pass whose result is multiplied by zero (ablation A7 turns
    all three off).
    """
    bbox = np.asarray(model.data.vol.bbox, dtype=np.float64)
    out: dict[str, Tensor] = {}
    builders: tuple[tuple[str, float, TermsFn], ...] = (
        ("cross", float(cfg.w_cross), cross_terms),
        ("thick", float(cfg.w_thick), thick_terms),
        ("prog", float(cfg.w_prog), prog_terms),
        # A8's negative control. Zero by default, so the builder is skipped and the term is
        # absent rather than a forward pass multiplied by zero (Config.w_prog_wrong).
        ("prog_wrong", float(cfg.w_prog_wrong), prog_wrong_terms),
    )
    for name, weight, builder in builders:
        if weight <= 0.0:
            continue
        parts = builder(model, model_ema, bbox, cfg, gen, gene_pool=gene_pool)
        out[name] = _sum_terms(parts)
        for part, value in parts.items():
            out[f"diag_{name}_{part}"] = value.detach()
    return out


# --------------------------------------------------------------------------------------
# statistics the losses are built from
# --------------------------------------------------------------------------------------


def mmd2_rbf(x: Tensor, y: Tensor, cfg: Config) -> Tensor:
    """Multi-bandwidth RBF MMD^2 between two point sets. ``(N, D)``, ``(M, D)`` -> ``()``.

    Bandwidths are ``Config.sefl_mmd_bandwidth_step ** k`` times the median pairwise distance
    of the pooled sample, for ``sefl_mmd_n_bandwidths`` values of ``k`` centred on zero. The
    biased (V-statistic) estimator, so the value is non-negative and usable as a loss.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError(f"mmd2_rbf expects (N, D) and (M, D), got {x.shape} and {y.shape}")
    pooled = torch.cat([x, y.detach()], dim=0)
    with torch.no_grad():
        median = torch.median(torch.cdist(pooled, pooled))
    scale = torch.clamp(median, min=1e-6)
    d_xx = torch.cdist(x, x) ** 2
    d_yy = torch.cdist(y, y) ** 2
    d_xy = torch.cdist(x, y) ** 2
    n_bandwidths = max(int(cfg.sefl_mmd_n_bandwidths), 1)
    step = float(cfg.sefl_mmd_bandwidth_step)
    powers = [k - (n_bandwidths - 1) / 2.0 for k in range(n_bandwidths)]
    total = torch.zeros((), dtype=x.dtype)
    for power in powers:
        sigma2 = (scale * step**power) ** 2
        total = total + (
            torch.exp(-d_xx / (2.0 * sigma2)).mean()
            + torch.exp(-d_yy / (2.0 * sigma2)).mean()
            - 2.0 * torch.exp(-d_xy / (2.0 * sigma2)).mean()
        )
    return total / float(n_bandwidths)


def sinkhorn_divergence(x: Tensor, y: Tensor, blur: float, cfg: Config) -> Tensor:
    """Debiased entropic Sinkhorn divergence. ``(N, D)``, ``(M, D)`` -> ``()``.

    ``S(x, y) - (S(x, x) + S(y, y)) / 2`` with uniform marginals, squared-Euclidean cost and
    entropic regularisation ``blur^2``, run in the log domain for
    ``Config.sefl_sinkhorn_iters`` iterations. Debiased because the entropic transport cost of
    a sample with *itself* is not zero, so the raw cost has a floor that moves with the
    sample's own spread — and a loss whose floor moves is a loss that can be reduced by
    shrinking the data.
    """
    return (
        _sinkhorn_cost(x, y, blur, cfg)
        - 0.5 * _sinkhorn_cost(x, x, blur, cfg)
        - 0.5 * _sinkhorn_cost(y, y, blur, cfg)
    )


def _sinkhorn_cost(x: Tensor, y: Tensor, blur: float, cfg: Config) -> Tensor:
    r"""Entropic optimal-transport cost between uniform samples. ``(N, D)``, ``(M, D)`` -> ``()``.

    The iteration runs under ``no_grad`` and the **plan is detached**, so the gradient flows
    through the cost matrix alone. That is the envelope theorem, not an approximation of
    convenience: at a converged plan the derivative of the transport cost with respect to the
    point positions is exactly the derivative of ``<plan, cost>`` holding the plan fixed, and
    it is what ``geomloss`` does for the same reason. It also keeps the autograd tape one
    matrix deep instead of ``2 * sefl_sinkhorn_iters`` ``logsumexp``\\ s deep, which on the
    reduced fixture is the difference between 146 ms and 60 ms per ``L_thick``.
    """
    epsilon = max(float(blur), 1e-6) ** 2
    cost = torch.cdist(x, y) ** 2
    n, m = cost.shape
    log_a = torch.full((n,), -math.log(n), dtype=cost.dtype)
    log_b = torch.full((m,), -math.log(m), dtype=cost.dtype)
    with torch.no_grad():
        detached = cost.detach()
        f = torch.zeros(n, dtype=cost.dtype)
        g = torch.zeros(m, dtype=cost.dtype)
        for _ in range(max(int(cfg.sefl_sinkhorn_iters), 1)):
            f = -epsilon * torch.logsumexp(
                (g[None, :] - detached) / epsilon + log_b[None, :], dim=1
            )
            g = -epsilon * torch.logsumexp(
                (f[:, None] - detached) / epsilon + log_a[:, None], dim=0
            )
        plan = torch.exp(
            (f[:, None] + g[None, :] - detached) / epsilon + log_a[:, None] + log_b[None, :]
        )
    return cast(Tensor, (plan * cost).sum())


def morans_i_torch(values: Tensor, coords: FloatArray, cfg: Config) -> Tensor:
    """Differentiable in-plane Moran's I per column. ``(N, G)``, ``(N, 2)`` -> ``(G,)``.

    On the row-normalised ``Config.sefl_morans_k`` nearest-neighbour graph of the *in-plane*
    coordinates. Only :func:`loss_prog_WRONG` uses it — Moran's I is in
    :data:`EQUIVARIANT_QUANTITIES` and constraining it across angles is the negative control.
    T08 owns the metric-aware version that is allowed to appear in a real loss, against real
    data rather than across angles.
    """
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"morans_i_torch expects (N, 2) coordinates, got {points.shape}")
    k = min(int(cfg.sefl_morans_k), points.shape[0] - 1)
    idx = cKDTree(points).query(points, k=k + 1)[1][:, 1:]
    neighbours = torch.from_numpy(np.ascontiguousarray(idx.astype(np.int64)))
    centred = values - values.mean(dim=0, keepdim=True)
    lagged = centred[neighbours].mean(dim=1)
    denominator = torch.clamp(centred.pow(2).sum(dim=0), min=1e-12)
    return (centred * lagged).sum(dim=0) / denominator


_MODULE_CACHE: weakref.WeakKeyDictionary[Any, IntArray] = weakref.WeakKeyDictionary()
"""Molecular programs, cached per model. Weak, so a discarded model does not pin its labels."""


def gene_modules(model: CTFFlow, cfg: Config, *, seed: int) -> IntArray:
    """Molecular programs: Leiden modules of the training gene-gene correlation graph.

    ``(G,)`` int64 labels, one per gene of the model's panel, computed once per model and
    cached (``specs/07`` §4: "computed once and cached"). The graph is T02's — the same
    ``coexpression_modules`` the text diagnostics partition — so "module" means one thing in
    this codebase rather than two.
    """
    cached = _MODULE_CACHE.get(model)
    if cached is not None:
        return cached
    counts = model.data.counts
    values = np.asarray(counts.todense(), dtype=np.float64)
    totals = np.clip(values.sum(axis=1, keepdims=True), 1.0, None)
    normalised = np.log1p(values / totals * float(np.median(totals)))
    coexpression = np.corrcoef(normalised, rowvar=False)
    coexpression = np.nan_to_num(coexpression, nan=0.0)
    labels = coexpression_modules(coexpression, int(cfg.text_diag_knn_k), cfg, seed=seed)
    out = np.asarray(labels, dtype=np.intp)
    _MODULE_CACHE[model] = out
    return out
