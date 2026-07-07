"""
End-to-end virtual-slice synthesis for SpatialCPA-v6.

Ties the stages together into a single call:

    embedding  ->  optimal-transport placement  ->  annotation (FM prior +
    composition + cell-cell-communication niche)  ->  expression

A design principle that drives the whole method: the correspondence-free metrics
factorize. ``coexpression`` / ``morans`` / ``gene_var`` / ``sinkhorn`` depend only
on **(position, expression)**; ``celltype_composition`` / ``celltype_nhood`` depend
only on **(position, label)**. v6 therefore treats the two channels separately:

* **(position, expression)** are taken from *real* training cells drawn from both
  flanking slices in the z-interpolated ratio, so the real cell-to-cell coupling
  (hence the gene-gene structure and spatial autocorrelation) is preserved and the
  population is the mixture of both flanking distributions — a better estimate of
  the true intermediate than either single slice (which is what SpatialZ / a
  single-slice copy uses). An OT-geodesic placement is available as an option.
* **labels** are re-derived by the foundation-model cell-state classifier,
  constrained to the interpolated composition and refined by the cell-cell
  communication (niche) MRF. Because this touches only the label channel, it can
  improve the two cell-type metrics *without any risk* to the expression metrics.

Contract (identical to the other benchmark-pbya-v2 generators): the method
receives the *training-only, re-registered* flanking slices and a scalar target
``z``; it never sees the held-out slice's ``(x, y)`` or content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import SpatialCPAv6Config
from .data import Slice, SliceStack
from .embedding import build_embedder
from . import transport as tp
from . import annotation as ann


@dataclass
class VirtualSlice:
    coords: np.ndarray                     # (M, 3)
    expression: np.ndarray                 # (M, G)
    cell_type: Optional[np.ndarray] = None  # (M,) string labels
    cell_type_idx: Optional[np.ndarray] = None  # (M,) int


class SpatialCPAv6:
    """Optimal-transport virtual-slice synthesizer.

    Parameters
    ----------
    stack
        Training-only :class:`SliceStack` (held-out slice excluded upstream).
    gene_names, cell_type_names
        Vocabularies for decoding outputs.
    cfg
        :class:`~spatialcpav6.config.SpatialCPAv6Config`.
    """

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv6Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv6Config()

        # Fit the (foundation-model / PCA) embedder on the training union once.
        self.embedder = build_embedder(stack.union_expression(),
                                       self.gene_names, self.cfg.embedding)
        self._slice_embed = {id(s): self.embedder(s.expression) for s in stack.slices}
        self.n_types = stack.n_cell_types()

    def _embed(self, s: Slice) -> np.ndarray:
        return self._slice_embed[id(s)]

    # ------------------------------------------------------------------ #
    # Placement                                                          #
    # ------------------------------------------------------------------ #
    def _place_real(self, lower, upper, t, n_target, rng):
        """Real-cell placement: draw cells from both flanking slices.

        Each synthesized cell copies a *real* training cell's ``(x, y)``,
        expression, type and embedding — from the lower slice with probability
        ``1-t`` and the upper with probability ``t`` — so the real
        position-expression coupling is preserved and the count is the
        z-interpolated flanking count. Returns index arrays into the flanking
        slices plus a source mask (True = upper).
        """
        n_lo, n_up = lower.n_spots, upper.n_spots
        take_up = int(round(t * n_target))
        take_lo = n_target - take_up
        lo_sel = rng.integers(0, n_lo, size=take_lo) if take_lo > 0 and n_lo > 0 else np.zeros(0, int)
        up_sel = rng.integers(0, n_up, size=take_up) if take_up > 0 and n_up > 0 else np.zeros(0, int)
        return lo_sel, up_sel

    # ------------------------------------------------------------------ #
    # Placement helpers. Each returns
    #   (coords_xy, expr, q_embed, init_types, dissimilarity_or_None)
    # ------------------------------------------------------------------ #
    def _place_morph(self, c):
        """Coherent single-sheet barycentric OT morph of the nearest slice."""
        lower, upper, z, t = c["lower"], c["upper"], c["z"], c["t"]
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            anchor, other, anchor_e, other_e, w, a_lab = (
                lower, upper, c["lo_e"], c["up_e"], t, c["lo_lab"])
        else:
            anchor, other, anchor_e, other_e, w, a_lab = (
                upper, lower, c["up_e"], c["lo_e"], 1.0 - t, c["up_lab"])
        coords_xy, a_idx, disp = tp.barycentric_interpolate(
            anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
            anchor_e, other_e, w, self.cfg.transport, seed=self.cfg.synthesis.seed)
        expr = anchor.expression[a_idx].astype(np.float32)
        q_embed = anchor_e[a_idx]
        init_types = a_lab[a_idx] if c["has_ct"] else None
        return coords_xy, expr, q_embed, init_types, disp

    def _place_backbone(self, c):
        """Positions + expression from the single flanking slice nearest in z."""
        lower, upper, z = c["lower"], c["upper"], c["z"]
        back = lower if abs(lower.z_center - z) <= abs(upper.z_center - z) else upper
        back_e = c["lo_e"] if back is lower else c["up_e"]
        init_types = back.cell_type_indices if c["has_ct"] else None
        return (back.coords_xy.astype(np.float64), back.expression.astype(np.float32),
                back_e, init_types, None)

    def _place_interpolate(self, c):
        """Real-cell mixing from both slices in the z-interpolated ratio."""
        lower, upper, t = c["lower"], c["upper"], c["t"]
        lo_xy, up_xy, lo_e, up_e = c["lo_xy"], c["up_xy"], c["lo_e"], c["up_e"]
        lo_sel, up_sel = self._place_real(lower, upper, t, c["n_target"], c["rng"])
        coords_xy = np.concatenate([lo_xy[lo_sel], up_xy[up_sel]], axis=0)
        expr = np.concatenate([lower.expression[lo_sel],
                               upper.expression[up_sel]], axis=0).astype(np.float32)
        q_embed = np.concatenate([lo_e[lo_sel], up_e[up_sel]], axis=0)
        init_types = (np.concatenate([c["lo_lab"][lo_sel], c["up_lab"][up_sel]])
                      if c["has_ct"] else None)
        return coords_xy, expr, q_embed, init_types, None

    def _place_ot_geodesic(self, c):
        """Pair-sampled McCann displacement interpolation (ablation)."""
        lower, upper, t = c["lower"], c["upper"], c["t"]
        lo_e, up_e = c["lo_e"], c["up_e"]
        res = tp.transport_interpolate(c["lo_xy"], c["up_xy"], lo_e, up_e, t,
                                       c["n_target"], self.cfg.transport,
                                       seed=self.cfg.synthesis.seed)
        n = res.coords_xy.shape[0]
        pick_up = c["rng"].random(n) < t
        expr = np.where(pick_up[:, None], upper.expression[res.up_idx],
                        lower.expression[res.lo_idx]).astype(np.float32)
        q_embed = (1.0 - t) * lo_e[res.lo_idx] + t * up_e[res.up_idx]
        init_types = (np.where(pick_up, c["up_lab"][res.up_idx], c["lo_lab"][res.lo_idx])
                      if c["has_ct"] else None)
        return res.coords_xy, expr, q_embed, init_types, None

    # ------------------------------------------------------------------ #
    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.synthesis.seed)
        lower, upper = self.stack.pick_flanking_slices(z)
        if lower.n_spots == 0 or upper.n_spots == 0:
            return VirtualSlice(coords=np.zeros((0, 3), np.float32),
                                expression=np.zeros((0, self.stack.n_genes), np.float32))
        lo_e, up_e = self._embed(lower), self._embed(upper)
        lo_xy = lower.coords_xy.astype(np.float64)
        up_xy = upper.coords_xy.astype(np.float64)

        t = tp.interpolation_fraction(z, lower.z_center, upper.z_center)
        n_target = tp.interpolated_count(lower.n_spots, upper.n_spots, t,
                                         cfg.synthesis.count_mode)

        has_ct = self.stack.has_cell_type and self.n_types and self.n_types >= 2
        lo_lab = lower.cell_type_indices if has_ct else None
        up_lab = upper.cell_type_indices if has_ct else None

        # ---- placement ------------------------------------------------------- #
        ctx = dict(lower=lower, upper=upper, lo_e=lo_e, up_e=up_e, lo_xy=lo_xy,
                   up_xy=up_xy, lo_lab=lo_lab, up_lab=up_lab, t=t, z=z,
                   n_target=n_target, has_ct=has_ct, rng=rng)
        placement = cfg.synthesis.placement
        self._last_placement = placement
        self._last_dissimilarity = None

        if placement == "adaptive":
            # Route by the OT-map displacement between the flanking slices: keep the
            # coherent morph when they are near-identical (small displacement, e.g.
            # thin volumetric z-planes), and switch to both-slice interpolation when
            # they are distinct tissue (large displacement), where the barycentric map
            # contracts and crushes density/composition.
            coords_xy, expr, q_embed, init_types, disp = self._place_morph(ctx)
            self._last_dissimilarity = disp
            if disp is not None and disp > cfg.transport.adaptive_threshold:
                coords_xy, expr, q_embed, init_types, _ = self._place_interpolate(ctx)
                self._last_placement = "interpolate"
            else:
                self._last_placement = "morph"
        elif placement == "morph":
            coords_xy, expr, q_embed, init_types, _ = self._place_morph(ctx)
        elif placement == "backbone":
            coords_xy, expr, q_embed, init_types, _ = self._place_backbone(ctx)
        elif placement == "interpolate":
            coords_xy, expr, q_embed, init_types, _ = self._place_interpolate(ctx)
        elif placement == "ot_geodesic":
            coords_xy, expr, q_embed, init_types, _ = self._place_ot_geodesic(ctx)
        else:
            # Fail loudly rather than silently falling back — a silent fallback to
            # a different placement is exactly how a version mismatch (new wrapper +
            # old package) can go unnoticed.
            raise ValueError(f"Unknown placement '{placement}'")
        n = coords_xy.shape[0]

        # ---- annotation (label channel only) --------------------------------- #
        cell_type_idx = None
        cell_type_labels = None
        if has_ct:
            init_labels = init_types

            if cfg.annotation.enabled:
                src_embed = np.concatenate([lo_e, up_e], axis=0)
                src_labels = np.concatenate([lo_lab, up_lab], axis=0)
                cell_type_idx = ann.annotate(
                    q_embed, coords_xy, src_embed, src_labels,
                    lo_xy, lo_lab, up_xy, up_lab,
                    self.n_types, t, init_labels,
                    cfg.annotation, cfg.communication, seed=cfg.synthesis.seed)
            else:
                cell_type_idx = init_labels

            if self.cell_type_names is not None:
                cell_type_labels = np.array(
                    [self.cell_type_names[i] for i in cell_type_idx])
            else:
                cell_type_labels = cell_type_idx.astype(str)

        # ---- optional expression transfer (independent of the above) --------- #
        # Off by default: copying real profiles already maximizes variance and
        # preserves gene-gene structure. "transfer"/"blend" trade variance for a
        # denoised profile; kept as ablations.
        if cfg.synthesis.expression_mode in ("transfer", "blend") and cell_type_idx is not None:
            src_expr = np.concatenate([lower.expression, upper.expression], axis=0)
            src_embed = np.concatenate([lo_e, up_e], axis=0)
            src_labels = np.concatenate([lower.cell_type_indices,
                                         upper.cell_type_indices], axis=0)
            transferred = _transfer_same_type(q_embed, cell_type_idx, src_expr,
                                              src_embed, src_labels,
                                              k=cfg.synthesis.transfer_k)
            if cfg.synthesis.expression_mode == "transfer":
                expr = transferred
            else:  # blend
                a = cfg.synthesis.transfer_alpha
                expr = (a * transferred + (1.0 - a) * expr).astype(np.float32)

        coords = np.column_stack(
            [coords_xy, np.full(n, float(z), dtype=np.float32)]).astype(np.float32)
        return VirtualSlice(coords=coords, expression=expr,
                            cell_type=cell_type_labels, cell_type_idx=cell_type_idx)


def _transfer_same_type(query_embed, query_labels, src_expr, src_embed,
                        src_labels, k=1):
    """Copy expression from the nearest same-type training cell(s) in embed space."""
    out = np.zeros((query_embed.shape[0], src_expr.shape[1]), dtype=np.float32)
    for c in np.unique(query_labels):
        q = np.where(query_labels == c)[0]
        s = np.where(src_labels == c)[0]
        if s.size == 0:
            s = np.arange(src_expr.shape[0])
        kk = min(k, s.size)
        _, nn = cKDTree(src_embed[s]).query(query_embed[q], k=kk)
        nn = np.atleast_2d(nn)[:, :kk]
        out[q] = src_expr[s][nn].mean(axis=1)
    return out
