"""
End-to-end virtual-slice synthesis for SpatialCPA-v8.

Pipeline (each stage is deliberately separable so optimizing one cannot spoil
another — the four steps of the design brief):

    embedding
      → 1. count           z-interpolated flanking count (emergent)
      → 2. placement       symmetric McCann barycentric bridge (both flanks)
      →    density calib.   resample toward the interpolated density field
      → 3. annotation      OT-source anchor + FM/spatial prior + niche MRF
      → 4. expression      real profile of each cell's source (+ optional transfer)

Why the stages don't fight each other — the correspondence-free metrics
factorize. ``coexpression`` / ``morans`` / ``sinkhorn`` / ``gene_*`` depend only
on **(position, expression)**; ``celltype_composition`` / ``celltype_nhood``
depend only on **(position, label)**; ``field`` / ``density`` / ``dice`` depend
only on **position**. v8 therefore:

* fixes **(position, expression)** by drawing *real* cells from *both* flanking
  slices through one coherent OT bridge — the population is the true mixture (so
  the distribution/co-expression metrics are right) and placement is coherent (so
  the spatial-autocorrelation metrics are right);
* fixes **position** further with a density calibration to the interpolated field
  (the density/field/dice metrics), which only *reselects* real cells, so it
  cannot change the expression distribution structure;
* fixes **label** last with a niche-aware annotator, which touches only the label
  channel, so it cannot hurt any expression metric.

Contract (identical to the other benchmark-pbya-v2 generators): the method
receives the *training-only, re-registered* flanking slices and a scalar target
``z``; it never sees the held-out slice's ``(x, y)`` or content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import SpatialCPAv8Config
from .data import Slice, SliceStack
from .embedding import build_embedder
from . import transport as tp
from . import density as dens
from . import annotation as ann
from . import selection as sel


@dataclass
class VirtualSlice:
    coords: np.ndarray                     # (M, 3)
    expression: np.ndarray                 # (M, G)
    cell_type: Optional[np.ndarray] = None  # (M,) string labels
    cell_type_idx: Optional[np.ndarray] = None  # (M,) int


class SpatialCPAv8:
    """Symmetric optimal-transport bridge virtual-slice synthesizer.

    Parameters
    ----------
    stack
        Training-only :class:`SliceStack` (held-out slice excluded upstream).
    gene_names, cell_type_names
        Vocabularies for decoding outputs.
    cfg
        :class:`~spatialcpav8.config.SpatialCPAv8Config`.
    """

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv8Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv8Config()

        self.embedder = build_embedder(stack.union_expression(),
                                       self.gene_names, self.cfg.embedding)
        self._slice_embed = {id(s): self.embedder(s.expression) for s in stack.slices}
        self.n_types = stack.n_cell_types()
        self._last_placement = None
        self._last_dissimilarity = None

    def _embed(self, s: Slice) -> np.ndarray:
        return self._slice_embed[id(s)]

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

        # ---- placement: resolve mode (internal CV for "adaptive"), then place  #
        mode = cfg.bridge.mode
        self._last_dissimilarity = None
        if mode == "adaptive":
            mode = self._resolve_adaptive_mode(cfg)
        self._last_placement = mode

        coords_xy, expr, q_embed, init_types = self._place_by_mode(
            mode, lower, upper, lo_e, up_e, lo_xy, up_xy, lo_lab, up_lab,
            t, z, n_target, has_ct, rng, cfg)

        # ---- density calibration (position channel only) --------------------- #
        if cfg.density.enabled and coords_xy.shape[0] >= 8:
            sel = dens.calibrate(coords_xy, lo_xy, up_xy, t, cfg.density,
                                 seed=cfg.synthesis.seed)
            coords_xy = coords_xy[sel]
            expr = expr[sel]
            q_embed = q_embed[sel]
            if init_types is not None:
                init_types = init_types[sel]
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
        return VirtualSlice(coords=coords, expression=expr.astype(np.float32),
                            cell_type=cell_type_labels, cell_type_idx=cell_type_idx)

    # ------------------------------------------------------------------ #
    # Mode dispatch                                                       #
    # ------------------------------------------------------------------ #
    def _place_by_mode(self, mode, lower, upper, lo_e, up_e, lo_xy, up_xy,
                       lo_lab, up_lab, t, z, n_target, has_ct, rng, cfg):
        if mode in ("smooth_morph", "diffeo_morph"):
            return self._place_smooth_morph(lower, upper, lo_e, up_e, lo_lab, up_lab,
                                            t, z, has_ct, cfg, svf=(mode == "diffeo_morph"))
        if mode in ("symmetric", "diffeo"):
            bridge_fn = tp.svf_bridge if mode == "diffeo" else tp.symmetric_bridge
            br = bridge_fn(lo_xy, up_xy, lo_e, up_e, t, n_target,
                           cfg.transport, cfg.bridge, seed=cfg.synthesis.seed)
            self._last_dissimilarity = br.dissimilarity
            coords_xy = br.coords_xy.astype(np.float64)
            expr, q_embed, init_types = self._gather_sources(
                br, lower, upper, lo_e, up_e, lo_lab, up_lab, has_ct)
            return coords_xy, expr, q_embed, init_types
        if mode == "coherent_mix":
            return self._place_coherent_mix(lower, upper, lo_e, up_e, lo_lab, up_lab,
                                            t, z, n_target, has_ct, cfg)
        if mode == "morph":
            return self._place_morph(lower, upper, lo_e, up_e, lo_lab, up_lab,
                                     t, z, has_ct, cfg)
        if mode == "backbone":
            return self._place_backbone(lower, upper, lo_e, up_e, lo_lab, up_lab,
                                        z, has_ct)
        if mode == "interpolate":
            return self._place_interpolate(lower, upper, lo_e, up_e, lo_lab, up_lab,
                                           t, n_target, has_ct, rng)
        raise ValueError(f"Unknown bridge mode '{mode}'")

    # ------------------------------------------------------------------ #
    # Adaptive placement via leakage-safe internal cross-validation       #
    # ------------------------------------------------------------------ #
    def _resolve_adaptive_mode(self, cfg):
        """Pick the placement that best reconstructs a held-out *training* slice.

        Cached after the first call. Falls back to the displacement heuristic when
        there are too few training slices for a CV split. Only training slices are
        used, so this introduces no leakage.
        """
        if getattr(self, "_cv_mode", None) is not None:
            return self._cv_mode
        candidates = list(cfg.bridge.adaptive_candidates)
        mid = sel.pick_cv_slice(self.stack)
        if mid is None:
            self._cv_mode = "interpolate"  # safe default with <3 training slices
            self._cv_scores = {}
            return self._cv_mode

        target = self.stack.slices[mid]
        reduced = SliceStack([s for i, s in enumerate(self.stack.slices) if i != mid])
        real_xy = target.coords_xy.astype(np.float64)
        real_X = target.expression.astype(np.float64)
        real_types = target.cell_type_indices if self.stack.has_cell_type else None

        scores = {}
        for cand in candidates:
            try:
                sub_cfg = _clone_cfg_with_mode(cfg, cand)
                sub = SpatialCPAv8(reduced, self.gene_names,
                                   self.cell_type_names, cfg=sub_cfg)
                vs = sub.generate_virtual_slice(z=float(target.z_center))
                if vs.coords.shape[0] < 5:
                    scores[cand] = -np.inf
                    continue
                scores[cand] = sel.score_reconstruction(
                    vs.coords, vs.expression, vs.cell_type_idx,
                    real_xy, real_X, real_types)
            except Exception:
                scores[cand] = -np.inf

        self._cv_scores = scores
        best = max(scores, key=scores.get) if scores else "interpolate"
        if not np.isfinite(scores.get(best, -np.inf)):
            best = "interpolate"
        self._cv_mode = best
        return best

    # ------------------------------------------------------------------ #
    # Source gathering for the symmetric bridge                          #
    # ------------------------------------------------------------------ #
    def _gather_sources(self, br, lower, upper, lo_e, up_e, lo_lab, up_lab, has_ct):
        """Pull real expression / embedding / label for each bridged cell."""
        n = br.coords_xy.shape[0]
        G = lower.expression.shape[1]
        d = lo_e.shape[1]
        expr = np.zeros((n, G), np.float32)
        q_embed = np.zeros((n, d), np.float32)
        init_types = np.zeros(n, np.int64) if has_ct else None
        up = br.from_upper
        lo = ~up
        if lo.any():
            idx = br.lo_src[lo]
            expr[lo] = lower.expression[idx]
            q_embed[lo] = lo_e[idx]
            if has_ct:
                init_types[lo] = lo_lab[idx]
        if up.any():
            idx = br.up_src[up]
            expr[up] = upper.expression[idx]
            q_embed[up] = up_e[idx]
            if has_ct:
                init_types[up] = up_lab[idx]
        return expr, q_embed, init_types

    # ------------------------------------------------------------------ #
    # Ablation placements                                                #
    # ------------------------------------------------------------------ #
    def _place_coherent_mix(self, lower, upper, lo_e, up_e, lo_lab, up_lab, t, z,
                            n_target, has_ct, cfg):
        anchor_is_lower = abs(lower.z_center - z) <= abs(upper.z_center - z)
        br = tp.coherent_mix(
            lower.coords_xy.astype(np.float64), upper.coords_xy.astype(np.float64),
            lo_e, up_e, t, n_target, cfg.transport, cfg.bridge,
            seed=cfg.synthesis.seed, anchor_is_lower=anchor_is_lower)
        self._last_dissimilarity = br.dissimilarity
        expr, q_embed, init_types = self._gather_sources(
            br, lower, upper, lo_e, up_e, lo_lab, up_lab, has_ct)
        return br.coords_xy.astype(np.float64), expr, q_embed, init_types

    def _place_smooth_morph(self, lower, upper, lo_e, up_e, lo_lab, up_lab, t, z,
                            has_ct, cfg, svf=False):
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            anchor, other, ae, oe, w, a_lab = lower, upper, lo_e, up_e, t, lo_lab
        else:
            anchor, other, ae, oe, w, a_lab = upper, lower, up_e, lo_e, 1.0 - t, up_lab
        if svf:
            coords_xy, a_idx, disp = tp.svf_morph(
                anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
                ae, oe, w, cfg.transport, cfg.bridge, seed=cfg.synthesis.seed)
        else:
            coords_xy, a_idx, disp = tp.smooth_morph(
                anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
                ae, oe, w, cfg.transport, cfg.bridge, seed=cfg.synthesis.seed)
        self._last_dissimilarity = disp
        expr = anchor.expression[a_idx].astype(np.float32)
        q_embed = ae[a_idx]
        init_types = a_lab[a_idx] if has_ct else None
        return coords_xy, expr, q_embed, init_types

    def _place_morph(self, lower, upper, lo_e, up_e, lo_lab, up_lab, t, z, has_ct, cfg):
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            anchor, other, ae, oe, w, a_lab = lower, upper, lo_e, up_e, t, lo_lab
        else:
            anchor, other, ae, oe, w, a_lab = upper, lower, up_e, lo_e, 1.0 - t, up_lab
        coords_xy, a_idx, disp = tp.one_sided_morph(
            anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
            ae, oe, w, cfg.transport, seed=cfg.synthesis.seed)
        self._last_dissimilarity = disp
        expr = anchor.expression[a_idx].astype(np.float32)
        q_embed = ae[a_idx]
        init_types = a_lab[a_idx] if has_ct else None
        return coords_xy, expr, q_embed, init_types

    def _place_backbone(self, lower, upper, lo_e, up_e, lo_lab, up_lab, z, has_ct):
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            back, be, blab = lower, lo_e, lo_lab
        else:
            back, be, blab = upper, up_e, up_lab
        init_types = blab if has_ct else None
        return (back.coords_xy.astype(np.float64), back.expression.astype(np.float32),
                be, init_types)

    def _place_interpolate(self, lower, upper, lo_e, up_e, lo_lab, up_lab, t,
                           n_target, has_ct, rng):
        n_lo, n_up = lower.n_spots, upper.n_spots
        take_up = int(round(t * n_target))
        take_lo = n_target - take_up
        lo_sel = rng.integers(0, n_lo, size=take_lo) if take_lo > 0 and n_lo > 0 else np.zeros(0, int)
        up_sel = rng.integers(0, n_up, size=take_up) if take_up > 0 and n_up > 0 else np.zeros(0, int)
        coords_xy = np.concatenate([lower.coords_xy[lo_sel], upper.coords_xy[up_sel]], axis=0).astype(np.float64)
        expr = np.concatenate([lower.expression[lo_sel], upper.expression[up_sel]], axis=0).astype(np.float32)
        q_embed = np.concatenate([lo_e[lo_sel], up_e[up_sel]], axis=0)
        init_types = (np.concatenate([lo_lab[lo_sel], up_lab[up_sel]]) if has_ct else None)
        return coords_xy, expr, q_embed, init_types


def _clone_cfg_with_mode(cfg, mode):
    """Shallow-copy the config with a concrete placement mode for a CV probe.

    The probe reuses the same transport/annotation settings but a fixed placement
    and (to keep CV fast and side-effect-free) density calibration disabled.
    """
    import copy
    sub = copy.deepcopy(cfg)
    sub.bridge.mode = mode
    sub.density.enabled = False
    return sub


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
