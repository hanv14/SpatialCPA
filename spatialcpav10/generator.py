"""
Biologically-constrained virtual-slice synthesis for SpatialCPA-v10.

Pipeline (cell type is the organizing latent, per the design brief):

    diffeomorphic placement (v8 backbone)
      -> 1. cell-type annotation (PRIMARY)   : type field + composition + niche MRF
      -> 2. cell-cell communication          : ligand-receptor coupling (DB / inferred)
      -> 3. biological expression generation  : μ_c(z) + λ·LR_modulation + real residual

Unlike v8 (which copies real expression), v10 *generates* each cell's expression from
an explicit biological model — a z-continuous cell-type program, modulated by the
ligand-receptor signaling of its spatial neighbours — keeping a real residual so
gene-gene structure stays realistic (the "balanced hybrid"). Positions still come from
the v8 diffeomorphic morph (a coherent tissue deformation), because coherent placement
is what the spatial-structure metrics require.

Contract: training-only flanking slices + scalar target z; no held-out (x, y)/content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SpatialCPAv10Config
from .data import Slice, SliceStack
from . import transport as tp
from . import annotation as ann
from . import biology as bio


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


class SpatialCPAv10:
    """Biologically-constrained virtual-slice synthesizer."""

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv10Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv10Config()
        self.n_types = stack.n_cell_types()
        self.embedder = _fit_pca(stack.union_expression(), d=cfg.embedding.n_components if cfg else 32)
        self._slice_embed = {id(s): self.embedder(s.expression) for s in stack.slices}
        self._W = self._build_lr_coupling()

    def _embed(self, s):
        return self._slice_embed[id(s)]

    # ------------------------------------------------------------------ #
    def _build_lr_coupling(self):
        """Ligand-receptor coupling matrix W (curated DB or data-inferred)."""
        b = self.cfg.biology
        if not b.enabled or b.lr_source == "off":
            return None
        W = None
        if b.lr_source in ("auto", "db"):
            import os
            path = b.lr_db_path or os.environ.get("SPATIALCPAV10_LR_DB")
            W = bio.load_lr_coupling(path, self.gene_names)
            if W is not None:
                print(f"[spatialcpav10] LR coupling from curated DB "
                      f"({int((W != 0).sum())} edges).")
        if W is None and b.lr_source in ("auto", "infer"):
            W = bio.infer_lr_coupling(self.stack, self.gene_names, k=b.lr_neighbors,
                                      top_frac=b.lr_infer_top_frac)
            if W is not None:
                print(f"[spatialcpav10] LR coupling inferred from data "
                      f"({int((W != 0).sum())} edges).")
        return W

    # ------------------------------------------------------------------ #
    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.synthesis.seed)
        lower, upper = self.stack.pick_flanking_slices(z)
        if lower.n_spots == 0 or upper.n_spots == 0:
            return VirtualSlice(coords=np.zeros((0, 3), np.float32),
                                expression=np.zeros((0, self.stack.n_genes), np.float32))
        lo_e, up_e = self._embed(lower), self._embed(upper)
        t = tp.interpolation_fraction(z, lower.z_center, upper.z_center)
        n_target = tp.interpolated_count(lower.n_spots, upper.n_spots, t,
                                         cfg.synthesis.count_mode)
        has_ct = self.stack.has_cell_type and self.n_types and self.n_types >= 2
        lo_lab = lower.cell_type_indices if has_ct else None
        up_lab = upper.cell_type_indices if has_ct else None

        # ---- placement: diffeomorphic single-slice backbone (v8) ------------- #
        if abs(lower.z_center - z) <= abs(upper.z_center - z):
            anchor, other, ae, oe, w, a_lab, o_lab = lower, upper, lo_e, up_e, t, lo_lab, up_lab
        else:
            anchor, other, ae, oe, w, a_lab, o_lab = upper, lower, up_e, lo_e, 1.0 - t, up_lab, lo_lab
        coords_xy, a_idx, disp, match_other = tp.svf_morph(
            anchor.coords_xy.astype(np.float64), other.coords_xy.astype(np.float64),
            ae, oe, w, cfg.transport, cfg.bridge, seed=cfg.synthesis.seed)
        self._last_dissimilarity = disp
        q_embed = np.asarray(ae[a_idx], dtype=np.float32)
        anchor_expr = anchor.expression[a_idx].astype(np.float32)      # real residual base
        src_types = a_lab[a_idx] if has_ct else None
        n = coords_xy.shape[0]

        cell_type_idx = None
        cell_type_labels = None
        if has_ct:
            # ---- 1. cell-type annotation (PRIMARY) + niche communication ------ #
            init_types = src_types
            if cfg.annotation.enabled:
                src_embed = np.concatenate([lo_e, up_e], axis=0)
                src_labels = np.concatenate([lo_lab, up_lab], axis=0)
                cell_type_idx = ann.annotate(
                    q_embed, coords_xy, src_embed, src_labels,
                    lower.coords_xy.astype(np.float64), lo_lab,
                    upper.coords_xy.astype(np.float64), up_lab,
                    self.n_types, t, init_types,
                    cfg.annotation, cfg.communication, seed=cfg.synthesis.seed)
            else:
                cell_type_idx = init_types

            # ---- 2-3. biological expression generation ----------------------- #
            if cfg.biology.enabled:
                programs = (bio.z_interpolated_programs(
                                lower.expression, lo_lab, upper.expression, up_lab,
                                self.n_types, t)
                            if cfg.biology.z_continuity
                            else bio.type_means(self.stack.union_expression(),
                                                np.concatenate([lo_lab, up_lab]), self.n_types))
                expr = bio.generate_expression(
                    cell_type_idx, coords_xy, programs, self._W,
                    residual=anchor_expr, residual_type_means=programs,
                    source_type_idx=src_types,
                    lr_lambda=cfg.biology.lr_lambda,
                    program_weight=cfg.biology.program_weight,
                    residual_weight=cfg.biology.residual_weight,
                    k=cfg.biology.lr_neighbors)
            else:
                expr = anchor_expr

            if self.cell_type_names is not None:
                cell_type_labels = np.array([self.cell_type_names[i] for i in cell_type_idx])
            else:
                cell_type_labels = cell_type_idx.astype(str)
        else:
            # No cell types -> cannot use type programs; copy the real profile.
            expr = anchor_expr

        coords = np.column_stack(
            [coords_xy, np.full(n, float(z), dtype=np.float32)]).astype(np.float32)
        return VirtualSlice(coords=coords, expression=expr.astype(np.float32),
                            cell_type=cell_type_labels, cell_type_idx=cell_type_idx)


def _fit_pca(train_expr, d=32):
    X = np.asarray(train_expr, np.float64)
    mean = X.mean(0, keepdims=True); std = X.std(0, keepdims=True); std[std == 0] = 1.0
    Xc = (X - mean) / std
    d = int(min(d, min(Xc.shape) - 1)) if min(Xc.shape) > 1 else 1
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:max(d, 1)]

    def fn(expr):
        return (((np.asarray(expr, np.float64) - mean) / std) @ comps.T).astype(np.float32)
    return fn
