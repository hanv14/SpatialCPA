"""
End-to-end virtual-slice synthesis for SpatialCPA-v9 (neural flow bridge).

Pipeline:

    expression autoencoder  ->  conditional flow-matching bridge (placement +
    expression latent)  ->  expression decoding (learned or nearest-real-cell) ->
    annotation (OT-anchor + prior + cell-cell-communication niche MRF)

The flow model is *trained* on the training slices (all adjacent pairs, OT-coupled
supervision) and then generalizes to any query z. Robustness is first-class: if
PyTorch is unavailable or training raises, the generator falls back to the v8
coherent optimal-transport morph, so a prediction is always produced and v9 never
does worse than the strong training-free baseline.

Contract (identical to the other benchmark-pbya-v2 generators): the method
receives the training-only, re-registered flanking slices and a scalar target z;
it never sees the held-out slice's (x, y) or content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .config import SpatialCPAv9Config
from .data import Slice, SliceStack
from . import transport as tp
from . import annotation as ann


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


class SpatialCPAv9:
    """Neural cross-slice flow-matching virtual-slice synthesizer."""

    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv9Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv9Config()
        self.n_types = stack.n_cell_types()
        self.bridge = None
        self._fit()

    # ------------------------------------------------------------------ #
    def _fit(self):
        cfg = self.cfg
        try:
            import torch  # noqa: F401
        except Exception as e:
            print(f"[spatialcpav9] PyTorch unavailable ({e}); using OT-morph fallback.")
            return
        try:
            from .flow import NeuralBridge
            gene_emb = self._load_gene_embedding()
            self.bridge = NeuralBridge(cfg, self.gene_names).fit(self.stack, gene_emb)
            # Bank of all training cells' latent / expression / type for nearest-real
            # decoding and annotation anchoring.
            lats, exprs, labs = [], [], []
            for s in self.stack.slices:
                lats.append(self.bridge._latent[id(s)])
                exprs.append(s.expression)
                labs.append(s.cell_type_indices if s.cell_type_indices is not None
                            else np.full(s.n_spots, -1, int))
            self._bank_lat = np.concatenate(lats, axis=0).astype(np.float32)
            self._bank_expr = np.concatenate(exprs, axis=0).astype(np.float32)
            self._bank_lab = np.concatenate(labs, axis=0)
            self._bank_tree = cKDTree(self._bank_lat)
            print(f"[spatialcpav9] neural bridge trained "
                  f"(latent_dim={cfg.model.latent_dim}, cells={self._bank_lat.shape[0]}).")
        except Exception as e:
            import traceback
            print(f"[spatialcpav9] training failed ({e}); OT-morph fallback.")
            if not cfg.train.fallback_on_error:
                raise
            traceback.print_exc()
            self.bridge = None

    def _load_gene_embedding(self):
        cfg = self.cfg.embedding
        if cfg.method not in ("fm_gene", "concat"):
            return None
        try:
            from .foundation_hook import load_gene_embedding
            return load_gene_embedding(cfg.fm_gene_embedding_path, self.gene_names)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        cfg = self.cfg
        rng = np.random.default_rng(cfg.synthesis.seed)
        lower, upper = self.stack.pick_flanking_slices(z)
        if lower.n_spots == 0 or upper.n_spots == 0:
            return VirtualSlice(coords=np.zeros((0, 3), np.float32),
                                expression=np.zeros((0, self.stack.n_genes), np.float32))

        t = tp.interpolation_fraction(z, lower.z_center, upper.z_center)
        n_target = tp.interpolated_count(lower.n_spots, upper.n_spots, t,
                                         cfg.synthesis.count_mode)
        has_ct = self.stack.has_cell_type and self.n_types and self.n_types >= 2
        lo_lab = lower.cell_type_indices if has_ct else None
        up_lab = upper.cell_type_indices if has_ct else None

        if self.bridge is not None and self.bridge.ok:
            coords_xy, latent, src_idx = self.bridge.generate(
                lower, upper, t, n_target, seed=cfg.synthesis.seed)
            self._last_mode = "neural_flow"
            mode = cfg.synthesis.expression_mode
            if mode == "source":
                # Real profile of each cell's source (the lower cell it flowed from).
                # Robust and structure-preserving (identical treatment to v8's morph):
                # the learned flow refines POSITIONS only, so a poorly-trained flow can
                # never degrade the expression / co-expression / variance metrics.
                expr = lower.expression[src_idx].astype(np.float32)
                init_types = lo_lab[src_idx] if has_ct else None
                q_embed = self.bridge._latent[id(lower)][src_idx]
            else:
                # Latent-space decoding modes (research / ablation): expression is
                # derived from the flowed latent, so it reflects the learned transport.
                _, nn = self._bank_tree.query(latent, k=1)
                init_types = self._bank_lab[nn] if has_ct else None
                q_embed = latent
                if mode == "decode":
                    expr = self.bridge.decode_expression(latent)
                elif mode == "blend":
                    a = cfg.synthesis.blend_alpha
                    expr = (a * self.bridge.decode_expression(latent)
                            + (1 - a) * self._bank_expr[nn]).astype(np.float32)
                else:  # nearest
                    expr = self._bank_expr[nn]
        else:
            # ---- OT-morph fallback (v8-equivalent) ---------------------- #
            self._last_mode = "ot_morph_fallback"
            lo_e = _pca_embed(self.stack.union_expression(), lower.expression)
            up_e = _pca_embed(self.stack.union_expression(), upper.expression)
            coords_xy, a_idx, _ = tp.smooth_morph(
                lower.coords_xy.astype(np.float64), upper.coords_xy.astype(np.float64),
                lo_e, up_e, t, cfg.transport, cfg.transport, seed=cfg.synthesis.seed)
            expr = lower.expression[a_idx].astype(np.float32)
            q_embed = lo_e[a_idx]
            init_types = lo_lab[a_idx] if has_ct else None

        n = coords_xy.shape[0]

        # ---- annotation (label channel only) --------------------------- #
        cell_type_idx = None
        cell_type_labels = None
        if has_ct:
            if cfg.annotation.enabled:
                src_embed = np.concatenate(
                    [self._embed_slice(lower), self._embed_slice(upper)], axis=0)
                src_labels = np.concatenate([lo_lab, up_lab], axis=0)
                cell_type_idx = ann.annotate(
                    q_embed, coords_xy, src_embed, src_labels,
                    lower.coords_xy.astype(np.float64), lo_lab,
                    upper.coords_xy.astype(np.float64), up_lab,
                    self.n_types, t, init_types,
                    cfg.annotation, cfg.communication, seed=cfg.synthesis.seed)
            else:
                cell_type_idx = init_types
            if self.cell_type_names is not None:
                cell_type_labels = np.array([self.cell_type_names[i] for i in cell_type_idx])
            else:
                cell_type_labels = cell_type_idx.astype(str)

        coords = np.column_stack(
            [coords_xy, np.full(n, float(z), dtype=np.float32)]).astype(np.float32)
        return VirtualSlice(coords=coords, expression=expr.astype(np.float32),
                            cell_type=cell_type_labels, cell_type_idx=cell_type_idx)

    # ------------------------------------------------------------------ #
    def _embed_slice(self, s):
        """Latent embedding of a slice for the annotation prior (bank or PCA)."""
        if self.bridge is not None and self.bridge.ok:
            return self.bridge._latent[id(s)]
        return _pca_embed(self.stack.union_expression(), s.expression)


def _pca_embed(train_expr, expr, d=16):
    """Cheap PCA embedding used only by the fallback path / annotation prior."""
    X = np.asarray(train_expr, np.float64)
    mean = X.mean(0, keepdims=True)
    std = X.std(0, keepdims=True); std[std == 0] = 1.0
    Xc = (X - mean) / std
    d = int(min(d, min(Xc.shape) - 1)) if min(Xc.shape) > 1 else 1
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:max(d, 1)]
    return (((np.asarray(expr, np.float64) - mean) / std) @ comps.T).astype(np.float32)
