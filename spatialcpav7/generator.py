"""
End-to-end virtual-slice synthesis for SpatialCPA-v7.

Ties the stages together into a single call:

    embedding (+ cross-slice anchoring)  ->  fused-GW placement  ->  annotation
    (FM prior + 2D/3D cell-cell communication + composition)  ->  expression

The design principle inherited from v6 — the correspondence-free metrics
factorize — still drives the method, and v7 sharpens each channel:

* **(position, expression)** are taken from *real* training cells, so gene-gene
  structure, spatial autocorrelation and the expression distribution are
  preserved. v7 places them along the **fused Gromov-Wasserstein** map, which
  additionally preserves the intra-slice neighbourhood *graph* across the morph
  (helping Moran's-I and neighbourhood agreement, not just the marginals).
* **labels** are re-derived by the FM cell-state classifier, constrained to the
  interpolated composition and refined by a **2D + 3D cell-cell communication**
  MRF. Because this touches only the label channel, it improves the cell-type
  metrics with no risk to the expression metrics.

Contract (identical to the other benchmark-pbya-v2 generators): the method
receives the *training-only, re-registered* flanking slices and a scalar target
``z``; it never sees the held-out slice's ``(x, y)`` or content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import SpatialCPAv7Config
from .data import Slice, SliceStack
from .embedding import build_embedder, mutual_nn_align
from . import transport as tp
from . import annotation as ann
from .foundation_assets import default_lr_pairs


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


class SpatialCPAv7:
    """Foundation-anchored fused-transport virtual-slice synthesizer."""

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv7Config] = None,
                 lr_pairs=None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv7Config()
        self.lr_pairs = lr_pairs if lr_pairs is not None else default_lr_pairs()

        self.embedder = build_embedder(stack.union_expression(),
                                       self.gene_names, self.cfg.embedding)
        self._slice_embed = {id(s): self.embedder(s.expression) for s in stack.slices}
        self.n_types = stack.n_cell_types()

    def _embed(self, s: Slice) -> np.ndarray:
        return self._slice_embed[id(s)]

    # ------------------------------------------------------------------ #
    # Placement helpers. Each returns
    #   (coords_xy, expr, q_embed, init_types, dissimilarity_or_None)
    # ------------------------------------------------------------------ #
    def _place_fgw_morph(self, c):
        lower, upper, z, t = c["lower"], c["upper"], c["z"], c["t"]
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            anchor, other, anchor_e, other_e, w, a_lab = (
                lower, upper, c["lo_e"], c["up_e"], t, c["lo_lab"])
        else:
            anchor, other, anchor_e, other_e, w, a_lab = (
                upper, lower, c["up_e"], c["lo_e"], 1.0 - t, c["up_lab"])
        coords_xy, a_idx, disp = tp.fgw_morph(
            anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
            anchor_e, other_e, w, self.cfg.transport, seed=self.cfg.synthesis.seed)
        expr = anchor.expression[a_idx].astype(np.float32)
        q_embed = anchor_e[a_idx]
        init_types = a_lab[a_idx] if c["has_ct"] else None
        return coords_xy, expr, q_embed, init_types, disp

    def _place_backbone(self, c):
        lower, upper, z = c["lower"], c["upper"], c["z"]
        back = lower if abs(lower.z_center - z) <= abs(upper.z_center - z) else upper
        back_e = c["lo_e"] if back is lower else c["up_e"]
        init_types = back.cell_type_indices if c["has_ct"] else None
        return (back.coords_xy.astype(np.float64), back.expression.astype(np.float32),
                back_e, init_types, None)

    def _place_interpolate(self, c):
        lower, upper, t = c["lower"], c["upper"], c["t"]
        lo_xy, up_xy, lo_e, up_e = c["lo_xy"], c["up_xy"], c["lo_e"], c["up_e"]
        rng = c["rng"]; n_target = c["n_target"]
        n_lo, n_up = lower.n_spots, upper.n_spots
        take_up = int(round(t * n_target)); take_lo = n_target - take_up
        lo_sel = rng.integers(0, n_lo, size=take_lo) if take_lo > 0 and n_lo > 0 else np.zeros(0, int)
        up_sel = rng.integers(0, n_up, size=take_up) if take_up > 0 and n_up > 0 else np.zeros(0, int)
        coords_xy = np.concatenate([lo_xy[lo_sel], up_xy[up_sel]], axis=0)
        expr = np.concatenate([lower.expression[lo_sel],
                               upper.expression[up_sel]], axis=0).astype(np.float32)
        q_embed = np.concatenate([lo_e[lo_sel], up_e[up_sel]], axis=0)
        init_types = (np.concatenate([c["lo_lab"][lo_sel], c["up_lab"][up_sel]])
                      if c["has_ct"] else None)
        return coords_xy, expr, q_embed, init_types, None

    def _place_fgw_geodesic(self, c):
        lower, upper, t = c["lower"], c["upper"], c["t"]
        lo_e, up_e = c["lo_e"], c["up_e"]
        res = tp.fgw_geodesic(c["lo_xy"], c["up_xy"], lo_e, up_e, t,
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

        # Cross-slice manifold anchoring: remove the batch offset between the two
        # flanking slices so the transport cost and the cell-type prior see one
        # consistent cell-state space. Leakage-safe (aligns training slices only).
        if cfg.embedding.cross_slice_anchor:
            lo_e, up_e = mutual_nn_align(lo_e, up_e, k=cfg.embedding.anchor_k,
                                         strength=cfg.embedding.anchor_strength)

        lo_xy = lower.coords_xy.astype(np.float64)
        up_xy = upper.coords_xy.astype(np.float64)
        t = tp.interpolation_fraction(z, lower.z_center, upper.z_center)
        n_target = tp.interpolated_count(lower.n_spots, upper.n_spots, t,
                                         cfg.synthesis.count_mode)

        has_ct = self.stack.has_cell_type and self.n_types and self.n_types >= 2
        lo_lab = lower.cell_type_indices if has_ct else None
        up_lab = upper.cell_type_indices if has_ct else None

        ctx = dict(lower=lower, upper=upper, lo_e=lo_e, up_e=up_e, lo_xy=lo_xy,
                   up_xy=up_xy, lo_lab=lo_lab, up_lab=up_lab, t=t, z=z,
                   n_target=n_target, has_ct=has_ct, rng=rng)
        placement = cfg.synthesis.placement
        self._last_placement = placement
        self._last_dissimilarity = None

        if placement == "adaptive":
            coords_xy, expr, q_embed, init_types, disp = self._place_fgw_morph(ctx)
            self._last_dissimilarity = disp
            if disp is not None and disp > cfg.transport.adaptive_threshold:
                coords_xy, expr, q_embed, init_types, _ = self._place_interpolate(ctx)
                self._last_placement = "interpolate"
            else:
                self._last_placement = "fgw_morph"
        elif placement == "fgw_morph":
            coords_xy, expr, q_embed, init_types, _ = self._place_fgw_morph(ctx)
        elif placement == "backbone":
            coords_xy, expr, q_embed, init_types, _ = self._place_backbone(ctx)
        elif placement == "interpolate":
            coords_xy, expr, q_embed, init_types, _ = self._place_interpolate(ctx)
        elif placement == "fgw_geodesic":
            coords_xy, expr, q_embed, init_types, _ = self._place_fgw_geodesic(ctx)
        else:
            raise ValueError(f"Unknown placement '{placement}'")
        n = coords_xy.shape[0]

        # ---- annotation (label channel only) --------------------------------- #
        cell_type_idx = None
        cell_type_labels = None
        if has_ct:
            if cfg.annotation.enabled:
                src_embed = np.concatenate([lo_e, up_e], axis=0)
                src_labels = np.concatenate([lo_lab, up_lab], axis=0)
                cell_type_idx = ann.annotate(
                    q_embed, coords_xy, src_embed, src_labels,
                    lo_xy, lo_lab, up_xy, up_lab,
                    lower.expression, upper.expression, self.gene_names, self.lr_pairs,
                    self.n_types, t, init_types,
                    cfg.annotation, cfg.communication, seed=cfg.synthesis.seed)
            else:
                cell_type_idx = (np.asarray(init_types).astype(int)
                                 if init_types is not None else np.zeros(n, int))

            if self.cell_type_names is not None:
                cell_type_labels = np.array(
                    [self.cell_type_names[i] for i in cell_type_idx])
            else:
                cell_type_labels = cell_type_idx.astype(str)

        # ---- optional expression transfer (independent of the above) --------- #
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
            else:
                a = cfg.synthesis.transfer_alpha
                expr = (a * transferred + (1.0 - a) * expr).astype(np.float32)

        coords = np.column_stack(
            [coords_xy, np.full(n, float(z), dtype=np.float32)]).astype(np.float32)
        return VirtualSlice(coords=coords, expression=expr,
                            cell_type=cell_type_labels, cell_type_idx=cell_type_idx)


def _transfer_same_type(query_embed, query_labels, src_expr, src_embed,
                        src_labels, k=1):
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
