"""Retrieval: real observed cells, offered to a query point as conditioning evidence.

The field carries anatomy; this branch carries realism. For a query position it retrieves
``K`` real cells from the training sections and cross-attends to them, so generation is
grounded in cells that were actually measured rather than in whatever the field's MLP
extrapolates.

The z-proximity term, and why it is the point
---------------------------------------------
A candidate cell ``j`` is ranked against a query point ``p`` by::

    score(p, j) = - d_inplane(p, j) / median_nn_dist
                  - w_z * |z_p - z_j| / median_spacing        <-- the competing method omits this
                  + w_niche * cos( niche(p), niche(j) )

All three terms are required. The competing method's donor weights use in-plane distance
and niche similarity only, so **a cell one-fifth of the way between two sections draws
roughly equally from both** — the near section and the far one are indistinguishable to a
score that cannot see z. That is the specific, nameable flaw this design fixes, and it is
ablation A5: ``Config.retrieval_w_z = 0`` reproduces the omission exactly, and GATE 2's
G2.3 measures what it costs at fractional depths 0.2 and 0.8 (where the two sections are
*not* equidistant) against 0.5 (where they are, so the term has nothing to say).

The z window is per-query, not per-stack
----------------------------------------
Ranking decides *which* donors come back; the window decides which are eligible at all.
A candidate is eligible when::

    |z_p - z_j| <= max(retrieval_z_window x median_spacing,
                       retrieval_z_window_gap_factor x gap_to_nearest(p))

The second term is what makes the bound a statement about **this query's evidence** rather
than about the stack's average. ``median_spacing`` is global, so on any irregular stack —
a holdout, a dropped section, a damaged one — it sizes the window off tissue that is not
there, and a section whose real gap exceeds the median retrieves nothing at all. With the
factor at ``>= 1`` the nearest surviving section is admitted at any gap, so the pool is
non-empty by construction and ``EmptyCandidatePoolWarning`` becomes a signal about the
*exclusions* instead of routine noise. ``Config.retrieval_z_window_gap_factor`` carries the
measurements that put it there.

Two consequences worth stating, because both are easy to get wrong later:

* **Order.** The gap is measured after ``exclude_z``, the own-section exclusion and the
  gap-aware dropout have run — a window sized off a section the query may not use is not a
  bound on anything. ``_query_chunk`` applies the filter last for that reason.
* **The window is part of the learned input distribution.** A model trained with donors no
  further than ``W`` and served donors from beyond it is being fed out-of-distribution
  evidence: on the GATE 2 fixture, widening the window at evaluation time only, without
  retraining, drove held-out ``R^2`` at one depth from ``-0.02`` to ``-0.35``. Training,
  calibration and inference must therefore share one rule, which is why it lives here and
  is driven by ``Config`` rather than being a per-call argument.

The niche is density-adaptive on purpose
----------------------------------------
``niche(.)`` is a local cell-type composition vector at ``cfg.niche_n_scales`` spatial
scales. The scales are set by a **rank**, not by a micrometre radius: scale ``s`` is the
composition among the ``niche_knn_k * niche_scale_factor ** s`` nearest cells, i.e. the
radius is the distance to that neighbour. A fixed radius means something different in
cortex than in white matter and something different again in a Xenium panel than in a MERFISH
one; a rank means the same thing in all of them. It is also exactly scale-invariant —
doubling every coordinate leaves the ranks, and therefore the niche, untouched, which is
what ``test_niche_density_adaptive`` asserts.

Relative positions only
-----------------------
Neighbour tokens carry ``p_j - p``, never ``p_j``. Absolute coordinates would let the model
memorise section identity — "cells at z = 250 look like this" — and it would then have
nothing to say at a z no section sits at, which is the entire task. The tokens are built in
the **data frame**, so they are invariant both to the augmentation rotation and to
translating the whole volume (``test_relative_position_only``).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy import sparse
from scipy.spatial import cKDTree
from torch import Tensor, nn

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Volume, to_xyz

__all__ = [
    "EmptyCandidatePoolWarning",
    "ExpressionPCs",
    "InertScoreWarning",
    "RetrievalAttention",
    "RetrievalIndex",
    "attention_entropy",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

PAD_INDEX = -1
"""Marker for a padded neighbour slot in :meth:`RetrievalIndex.query`'s indices. Negative
so that it can never be confused with a real row of the index, and so that
``idx >= 0`` is the mask."""

_Z_MATCH_FRAC = 1e-3
"""``exclude_z`` matches a section when the depths agree to this fraction of the median
spacing. Exact float equality on a depth that has been through a float32 coordinate array
and back would silently fail to exclude the section it names, which is a leak, not a
rounding detail."""


class InertScoreWarning(UserWarning):
    """The candidate pool was no larger than K, so the retrieval score chose nothing.

    The invariant that makes the score meaningful is about the **union** of candidates, not
    the per-section cap: ``retrieval_candidates_per_section`` bounds each admissible
    section's contribution, but what the top-K actually selects from is
    ``candidates_per_section x n_admissible_sections``. When that product falls to ``K`` or
    below, the top-K takes the whole pool and the three-term score — in-plane distance, z
    proximity, niche similarity — has no effect whatsoever on which donors are returned.

    ``Config.validate`` enforces ``retrieval_candidates_per_section >= retrieval_k``, which
    guarantees the invariant for a *single* admissible section. It cannot guarantee it at
    runtime, because the number of admissible sections is not a config field: ``exclude_z``,
    the z window, the own-section exclusion and — especially — the gap-aware section dropout
    all shrink it per query, and dropout does so *at inference*, which is where the retrieval
    branch is load-bearing. Hence a warning rather than a validation rule.
    """


class EmptyCandidatePoolWarning(UserWarning):
    """Some query point had no admissible donor cell at all.

    Every exclusion is legitimate on its own — a held-out section, the query's own section,
    the gap-aware dropout — but together they can leave no section standing, and the
    attention then reads a fully masked neighbour set and returns its bias.

    The z window is **not** one of the causes, and that is a deliberate property rather
    than an accident: ``retrieval_z_window_gap_factor >= 1`` makes the bound relative to
    each query's own gap, so the nearest surviving section is admitted however far away it
    is. Before that the window was a fixed multiple of the stack's *median* spacing and
    could starve a whole section of an irregular stack — 13500 of 81000 training cells on
    the GATE 2 fixture under a ``consecutive``-3 holdout. So this warning now means "the
    exclusions left this query nothing", which is a real signal, and it should be rare.
    """


class RetrievalIndex:
    """Real cells available as conditioning evidence, with gap-aware masking.

    Args:
        vol: the volume whose cells are the donor pool. In the pipeline a
             ``TrainingVolume`` — held-out sections are not evidence.
        cfg: supplies ``retrieval_k`` (K), ``retrieval_z_window``,
             ``retrieval_z_window_gap_factor``, ``retrieval_w_z``,
             ``retrieval_w_niche``, ``retrieval_score_temperature``,
             ``retrieval_candidates_per_section``, ``retrieval_query_chunk``,
             ``retrieval_exclude_source_section``, ``section_dropout_p``,
             ``section_dropout_max_sections``, ``expr_pca_dim``, ``niche_*`` and
             ``debug_shapes``.

    Attributes
    ----------
    coords
        ``(N, 3)`` float64 data-frame cell positions, pooled over sections in stack order.
    section_index
        ``(N,)`` intp index into ``section_z`` / ``section_ids``.
    expr_pca
        ``(N, expr_pca_dim)`` float32 expression PC scores. Library-size normalised and
        ``log1p``-ed first: an *input-side* transform, of the kind Convention 5 permits and
        requires to stay off the decoder's target.
    niche
        ``(N, niche_n_scales * n_types)`` float64 unit-norm local composition vectors.
    expression_pcs
        The fitted :class:`ExpressionPCs`, so the same basis can be applied to cells the
        index does not hold (GATE 2's held-out section, T08's LOSO folds).
    """

    def __init__(self, vol: Volume, cfg: Config) -> None:
        if not isinstance(vol, Volume):
            raise TypeError(
                f"RetrievalIndex expects a Volume (in the pipeline a TrainingVolume), got "
                f"{type(vol).__name__}. Held-out sections are not conditioning evidence."
            )
        self.cfg = cfg
        self.specimen_id = vol.specimen_id
        self.section_ids = list(vol.section_ids)
        self.section_z = vol.z_values.astype(np.float64)
        self.median_nn_dist = float(vol.median_nn_dist)
        self.median_spacing = float(vol.median_spacing)
        self.n_types = len(vol.celltype_names)
        self.n_regions = 0 if vol.region_names is None else len(vol.region_names)

        self.coords = np.concatenate([to_xyz(s).astype(np.float64) for s in vol.sections], axis=0)
        self.section_index = np.concatenate(
            [np.full(s.n_cells, i, dtype=np.intp) for i, s in enumerate(vol.sections)]
        )
        self.cell_type = np.concatenate([s.cell_type.astype(np.intp) for s in vol.sections])
        self.region = (
            None
            if vol.region_names is None
            else np.concatenate(
                [
                    (
                        np.zeros(s.n_cells, dtype=np.intp)
                        if s.region is None
                        else s.region.astype(np.intp)
                    )
                    for s in vol.sections
                ]
            )
        )
        self._offsets = np.concatenate(
            [np.array([0], dtype=np.intp), np.cumsum([s.n_cells for s in vol.sections])]
        ).astype(np.intp)
        self._trees = [cKDTree(np.asarray(s.coords, dtype=np.float64)) for s in vol.sections]

        self.expr_pca, self.expression_pcs = _expression_pcs(vol, cfg)
        self.niche = self._build_niche(cfg)

    # ----------------------------------------------------------------------------------
    # properties
    # ----------------------------------------------------------------------------------

    @property
    def n_cells(self) -> int:
        """N: donor cells in the index."""
        return int(self.coords.shape[0])

    @property
    def n_sections(self) -> int:
        """S: sections the donors come from."""
        return len(self.section_ids)

    @property
    def niche_dim(self) -> int:
        """Width of a niche vector: ``niche_n_scales * n_types``."""
        return int(self.cfg.niche_n_scales) * self.n_types

    @property
    def token_dim(self) -> int:
        """Width of one neighbour token (see :meth:`neighbour_tokens`)."""
        return int(self.cfg.expr_pca_dim) + self.n_types + self.n_regions + 4

    def __repr__(self) -> str:
        return (
            f"RetrievalIndex(specimen={self.specimen_id!r}, cells={self.n_cells}, "
            f"sections={self.n_sections}, K={int(self.cfg.retrieval_k)})"
        )

    # ----------------------------------------------------------------------------------
    # the niche
    # ----------------------------------------------------------------------------------

    def _build_niche(self, cfg: Config) -> FloatArray:
        """``(N, niche_n_scales * n_types)`` per-cell niche, built section by section.

        Each cell's niche is the cell-type composition among its ``k_s`` in-plane nearest
        neighbours *within its own section*, for each scale ``s``. Within a section
        in-plane distance and 3D distance coincide, so this is the same estimator
        :meth:`_query_niche` applies to a query point over its candidate pool — the two are
        comparable by cosine because they are the same statistic.
        """
        scales = _niche_scales(cfg)
        out = np.zeros((self.n_cells, self.niche_dim), dtype=np.float64)
        for s_index, tree in enumerate(self._trees):
            start, stop = int(self._offsets[s_index]), int(self._offsets[s_index + 1])
            points = self.coords[start:stop, :2]
            # k + 1: a point's first neighbour is itself, and a cell is part of its own
            # niche, so the self hit is kept and one extra is not requested.
            k = min(int(scales[-1]), points.shape[0])
            neighbours = tree.query(points, k=k)[1].reshape(points.shape[0], k)
            types = self.cell_type[start:stop][neighbours]
            for scale_index, k_s in enumerate(scales):
                take = types[:, : min(int(k_s), k)]
                block = _composition(take, self.n_types)
                out[start:stop, scale_index * self.n_types : (scale_index + 1) * self.n_types] = (
                    block
                )
        return _unit_rows(out)

    def _query_niche(
        self, xyz: FloatArray, candidate_idx: IntArray, valid: npt.NDArray[np.bool_]
    ) -> FloatArray:
        """Niche of a query point, from its **admissible** candidates only. ``(n, D)``.

        Built from the candidate pool rather than from whatever cells are nearest in the
        volume, so every exclusion (held-out section, own section, gap dropout) reaches the
        niche as well as the ranking. A query niche computed over the query's own section
        would smuggle that section back in through the score's third term after the first
        two had excluded it.
        """
        scales = _niche_scales(self.cfg)
        n = xyz.shape[0]
        out = np.zeros((n, self.niche_dim), dtype=np.float64)
        safe = np.where(valid, candidate_idx, 0)
        delta = self.coords[safe] - xyz[:, None, :]
        distance = np.where(valid, np.sqrt((delta**2).sum(axis=-1)), np.inf)
        order = np.argsort(distance, axis=1, kind="stable")
        ordered_types = np.take_along_axis(self.cell_type[safe], order, axis=1)
        ordered_valid = np.take_along_axis(valid, order, axis=1)
        for scale_index, k_s in enumerate(scales):
            take = min(int(k_s), ordered_types.shape[1])
            block = _composition(
                ordered_types[:, :take], self.n_types, mask=ordered_valid[:, :take]
            )
            out[:, scale_index * self.n_types : (scale_index + 1) * self.n_types] = block
        return _unit_rows(out)

    # ----------------------------------------------------------------------------------
    # the query
    # ----------------------------------------------------------------------------------

    def query(
        self,
        xyz: npt.NDArray[Any],
        exclude_z: set[float],
        seed: int,
        *,
        source_section: npt.NDArray[Any] | None = None,
        apply_dropout: bool = False,
    ) -> tuple[IntArray, FloatArray]:
        """Rank donor cells for each query point. ``(N, 3)`` -> ``(N, K)``, ``(N, K)``.

        Parameters
        ----------
        xyz
            ``(N, 3)`` query positions **in the data frame** (micrometres). Under rotation
            augmentation, pass ``RotationContext.retrieval(xyz_model)``: the index holds
            real cells at their real positions, so the query has to be asked in the tissue's
            own frame.
        exclude_z
            Section depths to remove from the candidate pool entirely — the held-out
            section(s) when reconstructing one, the flanking pair's neighbours when
            calibrating. Matched to within ``1e-3`` of the median spacing rather than by
            exact float equality, because a depth that has round-tripped through a float32
            coordinate array will not compare equal to the one that named it, and a
            silently ineffective exclusion is a leak.
        seed
            Explicit (Convention 3): seeds the gap-aware section dropout. Two calls with
            the same seed and the same arguments return bitwise identical results.
        source_section
            ``(N,)`` index into :attr:`section_ids` (or ``PAD_INDEX`` for a query that is
            not a real cell), naming each query's **own** section. With
            ``Config.retrieval_exclude_source_section`` set — the default — that section is
            removed from the query's pool, which is the exclusion GATE 2 rests on
            (SPEC_QUESTIONS C1a). Additive keyword beside ``exclude_z``, as C1 requires;
            the spec's three-argument signature has nowhere to put a *per-query* exclusion.
        apply_dropout
            Enable the gap-aware curriculum: with probability
            ``Config.section_dropout_p``, drop the query's ``section_dropout_max_sections``
            nearest admissible section(s), forcing reconstruction from remoter evidence.
            Off by default so that an evaluation path cannot accidentally randomise itself;
            the training loop passes ``True``.

        Returns
        -------
        tuple
            ``(N, K)`` intp row indices into the index (``PAD_INDEX`` where fewer than ``K``
            admissible candidates existed) and ``(N, K)`` float64 donor weights, the softmax
            of the score at ``Config.retrieval_score_temperature``, zero on padded slots.
            ``idx >= 0`` is the neighbour mask.
        """
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"RetrievalIndex.query expects (N, 3) coordinates, got {points.shape}")
        n = int(points.shape[0])
        k = int(self.cfg.retrieval_k)
        owner = self._check_source_section(source_section, n)

        section_allowed = self._sections_allowed(exclude_z)
        out_idx = np.full((n, k), PAD_INDEX, dtype=np.intp)
        out_weights = np.zeros((n, k), dtype=np.float64)
        empty = 0
        inert = 0
        chunk = int(self.cfg.retrieval_query_chunk)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            block_idx, block_weights, block_empty, block_inert = self._query_chunk(
                points[start:stop],
                section_allowed,
                owner[start:stop],
                seed=seed,
                offset=start,
                apply_dropout=apply_dropout,
            )
            out_idx[start:stop] = block_idx
            out_weights[start:stop] = block_weights
            empty += block_empty
            inert += block_inert

        if inert:
            warnings.warn(
                f"RetrievalIndex.query: {inert} of {n} query point(s) had a candidate union of "
                f"at most Config.retrieval_k={k} cells, so the top-K returned the whole pool "
                "and the retrieval score decided nothing for them (specimen "
                f"{self.specimen_id!r}). The invariant is about the union — "
                f"retrieval_candidates_per_section={self.cfg.retrieval_candidates_per_section} "
                f"x the number of admissible sections — not the per-section cap, and the "
                f"number of admissible sections shrinks with exclude_z ({len(exclude_z)} "
                f"excluded), the z window (per query, the larger of "
                f"{self.cfg.retrieval_z_window} x {self.median_spacing:g} um and "
                f"{self.cfg.retrieval_z_window_gap_factor} x its gap to the nearest "
                "admissible section), the own-section exclusion "
                f"({self.cfg.retrieval_exclude_source_section}) and the gap-aware dropout "
                f"(applied: {apply_dropout}). Raise retrieval_candidates_per_section or widen "
                "retrieval_z_window: the z-proximity term is inert for these queries.",
                InertScoreWarning,
                stacklevel=2,
            )

        if empty:
            warnings.warn(
                f"RetrievalIndex.query: {empty} of {n} query point(s) had no admissible donor "
                f"cell (specimen {self.specimen_id!r}). Excluded depths {sorted(exclude_z)}, "
                f"own-section exclusion {self.cfg.retrieval_exclude_source_section}. Their "
                "neighbour sets are fully masked and the attention returns its bias. Note "
                "the z window cannot be the cause: it admits the nearest surviving section "
                f"at any gap (retrieval_z_window_gap_factor="
                f"{self.cfg.retrieval_z_window_gap_factor} >= 1), so these queries have no "
                "candidate section left at all — look at the exclusions, not the window.",
                EmptyCandidatePoolWarning,
                stacklevel=2,
            )
        if self.cfg.debug_shapes:
            assert out_idx.shape == (n, k), out_idx.shape
            assert out_weights.shape == (n, k), out_weights.shape
        return out_idx, out_weights

    def _query_chunk(
        self,
        points: FloatArray,
        section_allowed: npt.NDArray[np.bool_],
        owner: IntArray,
        *,
        seed: int,
        offset: int,
        apply_dropout: bool,
    ) -> tuple[IntArray, FloatArray, int, int]:
        """Score and rank one chunk of queries.

        Returns ``(idx, weights, n_empty, n_inert)`` — the last being the number of queries
        whose admissible candidate union was no larger than ``K``, so the top-K selected all
        of it and the score decided nothing (:class:`InertScoreWarning`).
        """
        n = points.shape[0]
        k = int(self.cfg.retrieval_k)
        per_section = int(self.cfg.retrieval_candidates_per_section)

        candidate_idx, in_plane, valid = self._gather_candidates(
            points, section_allowed, per_section
        )
        section_of = self.section_index[np.where(valid, candidate_idx, 0)]
        delta_z = np.abs(self.section_z[section_of] - points[:, 2:3])

        # Order is load-bearing: the z window is sized off the gap to the nearest
        # *usable* section, so every other exclusion has to have run first. Sizing it off
        # a section the query may not retrieve from — its own, one named in exclude_z, one
        # the curriculum just dropped — hands back a bound derived from evidence that is
        # not there, and the non-empty guarantee is void. GATE 2 runs with the own-section
        # exclusion on, so this is the configuration the gate rests on, not a corner case.
        if self.cfg.retrieval_exclude_source_section:
            valid &= section_of != owner[:, None]
        if apply_dropout and self.cfg.section_dropout_p > 0.0:
            valid &= self._dropout_mask(section_of, delta_z, valid, seed=seed, offset=offset, n=n)
        valid &= delta_z <= self._z_window(delta_z, valid)

        niche_q = self._query_niche(points, candidate_idx, valid)
        niche_j = self.niche[np.where(valid, candidate_idx, 0)]
        similarity = np.einsum("nd,nkd->nk", niche_q, niche_j)

        score = (
            -in_plane / max(self.median_nn_dist, 1e-9)
            - self.cfg.retrieval_w_z * delta_z / max(self.median_spacing, 1e-9)
            + self.cfg.retrieval_w_niche * similarity
        )
        score = np.where(valid, score, -np.inf)

        take = min(k, score.shape[1])
        order = np.argsort(-score, axis=1, kind="stable")[:, :take]
        best_score = np.take_along_axis(score, order, axis=1)
        best_idx = np.take_along_axis(candidate_idx, order, axis=1)
        keep = np.isfinite(best_score)

        idx = np.full((n, k), PAD_INDEX, dtype=np.intp)
        idx[:, :take] = np.where(keep, best_idx, PAD_INDEX)
        weights = np.zeros((n, k), dtype=np.float64)
        weights[:, :take] = _masked_softmax(best_score, keep, self.cfg.retrieval_score_temperature)
        pool = valid.sum(axis=1)
        return (
            idx,
            weights,
            int((~keep.any(axis=1)).sum()),
            int(((pool > 0) & (pool <= k)).sum()),
        )

    def _z_window(
        self, delta_z: FloatArray, valid: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.float64]:
        """Per-query admissible ``|z_j - z_p|``, micrometres. ``(n, C)``, ``(n, C)`` -> ``(n, 1)``.

        Two terms, the larger winning (``Config.retrieval_z_window_gap_factor`` documents
        the failure that put the second one here)::

            max(retrieval_z_window x median_spacing,
                retrieval_z_window_gap_factor x gap to the nearest admissible candidate)

        ``valid`` must already carry **every** other exclusion — see the call site. A query
        with no surviving candidate at all gets an infinite bound, which changes nothing
        (there is nothing left to admit) and keeps the arithmetic free of ``nan``.
        """
        # ``initial`` rather than a special case: with every section excluded the candidate
        # block is (n, 0) and the reduction has nothing to fold over.
        nearest = np.min(np.where(valid, delta_z, np.inf), axis=1, keepdims=True, initial=np.inf)
        absolute = float(self.cfg.retrieval_z_window) * self.median_spacing
        relative = float(self.cfg.retrieval_z_window_gap_factor) * nearest
        window: npt.NDArray[np.float64] = np.maximum(absolute, relative)
        if self.cfg.debug_shapes:
            assert window.shape == (delta_z.shape[0], 1), window.shape
        return window

    def _gather_candidates(
        self,
        points: FloatArray,
        section_allowed: npt.NDArray[np.bool_],
        per_section: int,
    ) -> tuple[IntArray, FloatArray, npt.NDArray[np.bool_]]:
        """Return the per-section in-plane nearest candidates. Three ``(n, S' * C)`` arrays.

        Only the ``per_section`` in-plane nearest cells of each admissible section enter the
        ranking. The score is strictly decreasing in in-plane distance at fixed ``z`` and
        fixed niche, so the global top-K is contained in this union unless the niche term
        reorders more than ``per_section`` cells inside one section — bounded, cheap, and
        the alternative (scoring every cell of every section against every query) is
        quadratic in the volume.
        """
        n = points.shape[0]
        blocks_idx: list[IntArray] = []
        blocks_dist: list[FloatArray] = []
        blocks_valid: list[npt.NDArray[np.bool_]] = []
        for s_index, tree in enumerate(self._trees):
            if not section_allowed[s_index]:
                continue
            start = int(self._offsets[s_index])
            size = int(self._offsets[s_index + 1]) - start
            k = min(per_section, size)
            distance, local = tree.query(points[:, :2], k=k)
            distance = np.asarray(distance, dtype=np.float64).reshape(n, k)
            local = np.asarray(local, dtype=np.intp).reshape(n, k)
            # cKDTree pads a short result with index == size and distance == inf.
            ok = local < size
            blocks_idx.append(np.where(ok, local + start, 0))
            blocks_dist.append(np.where(ok, distance, np.inf))
            blocks_valid.append(ok)
        if not blocks_idx:
            zero = (n, 0)
            return (
                np.zeros(zero, dtype=np.intp),
                np.zeros(zero, dtype=np.float64),
                np.zeros(zero, dtype=bool),
            )
        return (
            np.concatenate(blocks_idx, axis=1),
            np.concatenate(blocks_dist, axis=1),
            np.concatenate(blocks_valid, axis=1),
        )

    def _sections_allowed(self, exclude_z: set[float]) -> npt.NDArray[np.bool_]:
        """``(S,)`` bool: which sections survive ``exclude_z``."""
        allowed = np.ones(self.n_sections, dtype=bool)
        tolerance = _Z_MATCH_FRAC * max(self.median_spacing, 1e-9)
        for z in exclude_z:
            hit = np.abs(self.section_z - float(z)) <= tolerance
            if not hit.any():
                raise ValueError(
                    f"RetrievalIndex.query: exclude_z names z={z!r}, which is not a section "
                    f"of specimen {self.specimen_id!r} (depths {self.section_z.tolist()}). "
                    "An exclusion that matches nothing is a leak, not a no-op."
                )
            allowed &= ~hit
        return allowed

    def _dropout_mask(
        self,
        section_of: IntArray,
        delta_z: FloatArray,
        valid: npt.NDArray[np.bool_],
        *,
        seed: int,
        offset: int,
        n: int,
    ) -> npt.NDArray[np.bool_]:
        """Gap-aware curriculum: drop each query's nearest admissible section(s).

        This is the wide-gap curriculum, and unlike previous versions it is *not* inert:
        the retrieval branch is load-bearing at inference, so a model that has only ever
        seen a donor section one spacing away has learned a shortcut it cannot use when the
        gap is three.
        """
        gen = np.random.default_rng([seed, offset])
        fires = gen.random(n) < self.cfg.section_dropout_p
        keep = np.ones(section_of.shape, dtype=bool)
        if not fires.any():
            return keep
        far = np.where(valid, delta_z, np.inf)
        for _ in range(int(self.cfg.section_dropout_max_sections)):
            nearest = np.argmin(far, axis=1)
            dropped_section = section_of[np.arange(section_of.shape[0]), nearest]
            reachable = np.isfinite(far[np.arange(far.shape[0]), nearest])
            hit = fires & reachable
            match = (section_of == dropped_section[:, None]) & hit[:, None]
            keep &= ~match
            far = np.where(match, np.inf, far)
        return keep

    def _check_source_section(self, source_section: npt.NDArray[Any] | None, n: int) -> IntArray:
        """Validate the per-query own-section codes and return them as ``(n,)`` intp."""
        if source_section is None:
            if self.cfg.retrieval_exclude_source_section:
                # Not an error: T05/T09 query points that are not cells and have no source
                # section. It *is* an error to reach GATE 2's evaluation this way, which is
                # why the gate's own measurement passes the array explicitly.
                return np.full(n, PAD_INDEX, dtype=np.intp)
            return np.full(n, PAD_INDEX, dtype=np.intp)
        owner = np.asarray(source_section, dtype=np.intp).reshape(-1)
        if owner.shape[0] != n:
            raise ValueError(
                f"RetrievalIndex.query: source_section has {owner.shape[0]} entries but "
                f"{n} query points were given"
            )
        if int(owner.max(initial=PAD_INDEX)) >= self.n_sections:
            raise ValueError(
                f"RetrievalIndex.query: source_section holds index "
                f"{int(owner.max(initial=PAD_INDEX))} but specimen {self.specimen_id!r} has "
                f"{self.n_sections} sections"
            )
        return owner

    def section_index_of_cells(self) -> IntArray:
        """``(N,)`` the source-section index of every donor cell.

        The array GATE 2 passes back as ``source_section`` when the query points *are* the
        index's own cells, which is the leave-own-section-out contract.
        """
        return self.section_index

    # ----------------------------------------------------------------------------------
    # tokens
    # ----------------------------------------------------------------------------------

    def neighbour_tokens(self, xyz: npt.NDArray[Any], idx: IntArray) -> tuple[Tensor, Tensor]:
        """Encode retrieved neighbours. ``(N, 3)``, ``(N, K)`` -> ``(N, K, d_token)``, ``(N, K)``.

        The token is ``[expr_pca, type, region, delta_position, delta_z]``:

        ``expr_pca``
            ``cfg.expr_pca_dim`` expression PCs of the donor cell.
        ``type``, ``region``
            One-hot codes. A one-hot followed by the attention's ``Linear`` key/value
            projection *is* a learned embedding — the same parameters, one fewer module —
            and it keeps this branch runnable without T02's text vectors. T06 swaps in
            ``EntityEmbeddings`` when the observation token is assembled.
        ``delta_position``
            ``(p_j - p) / median_nn_dist``, **relative**, in the data frame. Never absolute:
            absolute coordinates let the model memorise section identity and it then has
            nothing to say at a z no section sits at (T04 "Do NOT").
        ``delta_z``
            ``(z_j - z_p) / median_spacing``. Redundant with the third component of
            ``delta_position`` up to a scale, and carried anyway: it is the quantity the
            score's z term ranks on, and giving the attention the same axis in the units the
            gap is measured in is what lets it tell "one section away" from "three".

        Returns the tokens and the ``(N, K)`` bool mask (``True`` = a real neighbour).
        Masked slots are zeroed so a padded token cannot leak through a bias term.
        """
        points = np.asarray(xyz, dtype=np.float64)
        neighbours = np.asarray(idx, dtype=np.intp)
        if neighbours.ndim != 2 or neighbours.shape[0] != points.shape[0]:
            raise ValueError(
                f"neighbour_tokens: idx must be (N, K) matching the {points.shape[0]} query "
                f"points, got {neighbours.shape}"
            )
        mask = neighbours >= 0
        safe = np.where(mask, neighbours, 0)

        parts: list[FloatArray] = [self.expr_pca[safe].astype(np.float64)]
        parts.append(_one_hot(self.cell_type[safe], self.n_types))
        if self.region is not None:
            parts.append(_one_hot(self.region[safe], self.n_regions))
        delta = (self.coords[safe] - points[:, None, :]) / max(self.median_nn_dist, 1e-9)
        parts.append(delta)
        delta_z = (self.coords[safe][:, :, 2] - points[:, None, 2]) / max(self.median_spacing, 1e-9)
        parts.append(delta_z[:, :, None])

        tokens = np.concatenate(parts, axis=2) * mask[:, :, None]
        if self.cfg.debug_shapes:
            assert tokens.shape == (points.shape[0], neighbours.shape[1], self.token_dim)
        return (
            torch.from_numpy(tokens.astype(np.float32)),
            torch.from_numpy(mask),
        )


# --------------------------------------------------------------------------------------
# cross-attention
# --------------------------------------------------------------------------------------


class RetrievalAttention(nn.Module):
    """Multi-head cross-attention from a query point to its retrieved real cells.

    Args:
        cfg:       supplies ``retrieval_ctx_dim`` (d_ctx), ``retrieval_n_heads``, ``seed``
                   and ``debug_shapes``.
        q_dim:     width of the query features — in the pipeline
                   ``[F(p), fourier(p), type_emb, ...]`` assembled by the caller.
        token_dim: width of one neighbour token (``RetrievalIndex.token_dim``).
    Returns from forward: ``(N, d_ctx)``.

    Notes
    -----
    Padded neighbours are masked in the logits, not zeroed afterwards: a zeroed value with
    a non-zero softmax weight still shifts the output towards the origin, which reads as
    "this query has weak evidence" when in fact it has fewer donors.

    A query with **no** admissible donor at all gets an all-masked row. Its softmax is
    defined as uniform-over-nothing — the output is exactly the value projection's bias —
    rather than NaN, so one such point at the end of a stack does not poison a whole batch's
    gradient. :class:`EmptyCandidatePoolWarning` is where that condition is reported.
    """

    def __init__(self, cfg: Config, q_dim: int, token_dim: int) -> None:
        super().__init__()
        if q_dim < 1 or token_dim < 1:
            raise ValueError(
                f"RetrievalAttention: q_dim and token_dim must be >= 1, got {q_dim}, {token_dim}"
            )
        self.cfg = cfg
        self.d_ctx = int(cfg.retrieval_ctx_dim)
        self.n_heads = int(cfg.retrieval_n_heads)
        self.head_dim = self.d_ctx // self.n_heads
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.q_proj = nn.Linear(int(q_dim), self.d_ctx)
        self.k_proj = nn.Linear(int(token_dim), self.d_ctx)
        self.v_proj = nn.Linear(int(token_dim), self.d_ctx)
        self.out_proj = nn.Linear(self.d_ctx, self.d_ctx)
        for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            _init_linear(layer, generator)

    def __call__(self, q_feat: Tensor, neigh_tokens: Tensor, neigh_mask: Tensor) -> Tensor:
        """``(N, q_dim)``, ``(N, K, d_token)``, ``(N, K)`` -> ``(N, d_ctx)``."""
        return cast(Tensor, super().__call__(q_feat, neigh_tokens, neigh_mask))

    def forward(self, q_feat: Tensor, neigh_tokens: Tensor, neigh_mask: Tensor) -> Tensor:
        """``(N, q_dim)``, ``(N, K, d_token)``, ``(N, K)`` bool -> ``(N, d_ctx)`` float32."""
        return self.attend(q_feat, neigh_tokens, neigh_mask)[0]

    def attend(
        self, q_feat: Tensor, neigh_tokens: Tensor, neigh_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """As :meth:`forward`, also returning the ``(N, n_heads, K)`` attention weights.

        The weights are what GATE 2's G2.4 measures: if the attention is one-hot the model
        is copying its nearest donor and will fail as soon as the gap is wide enough that
        the nearest donor is not a good answer.
        """
        if q_feat.ndim != 2 or neigh_tokens.ndim != 3 or neigh_mask.ndim != 2:
            raise ValueError(
                "RetrievalAttention expects (N, q_dim), (N, K, d_token), (N, K); got "
                f"{tuple(q_feat.shape)}, {tuple(neigh_tokens.shape)}, {tuple(neigh_mask.shape)}"
            )
        n, k = int(neigh_mask.shape[0]), int(neigh_mask.shape[1])
        if q_feat.shape[0] != n or neigh_tokens.shape[:2] != (n, k):
            raise ValueError(
                "RetrievalAttention: the three inputs disagree about N or K: "
                f"{tuple(q_feat.shape)}, {tuple(neigh_tokens.shape)}, {tuple(neigh_mask.shape)}"
            )

        query = self.q_proj(q_feat).view(n, self.n_heads, self.head_dim)
        keys = self.k_proj(neigh_tokens).view(n, k, self.n_heads, self.head_dim)
        values = self.v_proj(neigh_tokens).view(n, k, self.n_heads, self.head_dim)

        logits = torch.einsum("nhd,nkhd->nhk", query, keys) / math.sqrt(self.head_dim)
        mask = neigh_mask.to(torch.bool)[:, None, :].expand(n, self.n_heads, k)
        logits = logits.masked_fill(~mask, float("-inf"))
        # An all-masked row would softmax to NaN. Give it a uniform (all-zero-logit) row and
        # zero the weights afterwards, so its output is exactly the value bias.
        empty = ~mask.any(dim=2, keepdim=True)
        weights = torch.softmax(torch.where(empty, torch.zeros_like(logits), logits), dim=2)
        weights = torch.where(empty, torch.zeros_like(weights), weights)

        pooled = torch.einsum("nhk,nkhd->nhd", weights, values).reshape(n, self.d_ctx)
        out = cast(Tensor, self.out_proj(pooled))
        if self.cfg.debug_shapes:
            assert out.shape == (n, self.d_ctx), out.shape
            assert weights.shape == (n, self.n_heads, k), weights.shape
        return out, weights


def attention_entropy(weights: Tensor) -> Tensor:
    """Shannon entropy in nats of each attention row. ``(N, H, K)`` -> ``(N, H)``.

    GATE 2's G2.4 requires the mean to exceed ``0.5 * log(K)``. A one-hot row has entropy
    0; a uniform row over ``K`` donors has ``log(K)``. The criterion is not "spread your
    attention" for its own sake — it is that a model whose attention has collapsed onto one
    neighbour has learned to copy, and copying has nothing to say when the nearest real cell
    is three sections away.
    """
    safe = torch.clamp(weights, min=1e-12)
    entropy: Tensor = -(weights * torch.log(safe)).sum(dim=-1)
    return entropy


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpressionPCs:
    """The expression PCA the index was built with, kept so it can be reapplied.

    Fitted on the index's own (training) cells. Anything that needs the *same* coordinates
    for cells the index does not hold — GATE 2's held-out section, T08's LOSO folds — must
    reapply this basis rather than refit one, or the two score sets are not comparable and
    the refit would have seen the held-out cells.
    """

    mean: FloatArray
    """(1, G) mean of the log1p-normalised expression over the fitted cells."""
    basis: FloatArray
    """(G, expr_pca_dim) leading eigenvectors of the gene-gene covariance."""
    scale: FloatArray
    """(1, expr_pca_dim) per-component standard deviation, so scores are unit variance."""

    def project(self, counts: Any) -> npt.NDArray[np.float32]:
        """Apply the fitted basis to a raw count matrix. ``(N, G)`` -> ``(N, expr_pca_dim)``."""
        values = _normalised_expression(counts)
        scores = (values.astype(np.float64) - self.mean) @ self.basis
        return cast("npt.NDArray[np.float32]", (scores / self.scale).astype(np.float32))


def _normalised_expression(counts: Any) -> npt.NDArray[np.float64]:
    """Library-size normalise and ``log1p`` a count matrix. ``(N, G)`` -> ``(N, G)`` float64.

    An *input-side* transform of the kind Convention 5 permits and requires to stay off the
    decoder's target: PCs of raw counts would mostly measure library size.
    """
    dense = np.asarray(counts.toarray() if sparse.issparse(counts) else counts, dtype=np.float64)
    totals = dense.sum(axis=1, keepdims=True)
    totals[totals == 0.0] = 1.0
    return cast("npt.NDArray[np.float64]", np.log1p(dense / totals * float(np.median(totals))))


def _expression_pcs(vol: Volume, cfg: Config) -> tuple[npt.NDArray[np.float32], ExpressionPCs]:
    """Fit the expression PCA over a volume. ``(N, expr_pca_dim)`` scores plus the basis.

    Mean-centred, then projected onto the leading eigenvectors of the gene-gene covariance.
    The eigendecomposition is of the ``(G, G)`` covariance rather than an SVD of the
    ``(N, G)`` matrix: same subspace, and at ``N >> G`` it is the difference between seconds
    and minutes on CPU.

    Deterministic — no randomised solver, so no seed to plumb (Convention 3 is satisfied by
    there being nothing stochastic here). Eigenvector signs are canonicalised so two runs on
    two machines cannot return the same subspace with flipped axes.
    """
    dim = int(cfg.expr_pca_dim)
    values = np.concatenate(
        [_normalised_expression(section.counts) for section in vol.sections], axis=0
    )
    if dim > values.shape[1]:
        raise ValueError(
            f"RetrievalIndex: Config.expr_pca_dim={dim} exceeds the {values.shape[1]} genes "
            f"of specimen {vol.specimen_id!r}"
        )
    mean = values.mean(axis=0, keepdims=True)
    centred = values - mean
    covariance = (centred.T @ centred) / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:dim]
    basis = eigenvectors[:, order]
    # Canonical sign: the largest-magnitude loading of each component is positive.
    signs = np.sign(basis[np.abs(basis).argmax(axis=0), np.arange(basis.shape[1])])
    signs[signs == 0.0] = 1.0
    basis = basis * signs[None, :]
    scores = centred @ basis
    scale = scores.std(axis=0, keepdims=True)
    scale[scale <= 0.0] = 1.0
    pcs = ExpressionPCs(mean=mean, basis=basis, scale=scale)
    return cast("npt.NDArray[np.float32]", (scores / scale).astype(np.float32)), pcs


def _niche_scales(cfg: Config) -> list[int]:
    """Neighbour counts defining the niche's scales: ``k, k*f, k*f^2, ...``, strictly increasing."""
    scales: list[int] = []
    for s in range(int(cfg.niche_n_scales)):
        value = round(int(cfg.niche_knn_k) * float(cfg.niche_scale_factor) ** s)
        scales.append(max(value, (scales[-1] + 1) if scales else 1))
    return scales


def _composition(
    types: IntArray, n_types: int, mask: npt.NDArray[np.bool_] | None = None
) -> FloatArray:
    """Cell-type composition of each row. ``(n, k)`` codes -> ``(n, n_types)`` fractions."""
    n, k = types.shape
    weights = np.ones((n, k), dtype=np.float64) if mask is None else mask.astype(np.float64)
    out = np.zeros((n, n_types), dtype=np.float64)
    rows = np.repeat(np.arange(n), k)
    np.add.at(out, (rows, types.reshape(-1)), weights.reshape(-1))
    total = out.sum(axis=1, keepdims=True)
    total[total == 0.0] = 1.0
    return cast(FloatArray, out / total)


def _unit_rows(values: FloatArray) -> FloatArray:
    """L2-normalise each row, leaving all-zero rows alone (so the cosine term is 0, not NaN)."""
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return cast(FloatArray, values / norm)


def _one_hot(codes: IntArray, n: int) -> FloatArray:
    """One-hot encode ``(a, b)`` integer codes into ``(a, b, n)`` float64."""
    out = np.zeros((*codes.shape, n), dtype=np.float64)
    np.put_along_axis(out, np.clip(codes, 0, n - 1)[..., None], 1.0, axis=-1)
    return out


def _masked_softmax(
    score: FloatArray, keep: npt.NDArray[np.bool_], temperature: float
) -> FloatArray:
    """Row-wise softmax over the kept entries; all-masked rows return zeros, not NaN.

    A **zero-width** block — no candidate columns at all, which is what ``exclude_z`` naming
    every section leaves behind — is the same answer as an all-masked row and is returned
    the same way. Handled here rather than left to the reduction below, because
    ``max`` over an empty axis raises ``ValueError`` and the caller's contract for a query
    with no evidence is :class:`EmptyCandidatePoolWarning`, not a numpy error from three
    frames down.
    """
    if score.shape[1] == 0:
        return np.zeros(score.shape, dtype=np.float64)
    scaled = np.where(keep, score / max(temperature, 1e-9), -np.inf)
    shift = np.where(keep.any(axis=1, keepdims=True), scaled.max(axis=1, keepdims=True), 0.0)
    exponent = np.where(keep, np.exp(scaled - shift), 0.0)
    total = exponent.sum(axis=1, keepdims=True)
    total[total == 0.0] = 1.0
    return cast(FloatArray, exponent / total)


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Re-initialise a linear layer from an explicit generator (Convention 3)."""
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)
