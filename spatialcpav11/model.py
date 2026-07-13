"""
SpatialCPA-v11 — two-stage continuous 3D virtual-slice model (orchestration).

Ties together the neighbouring-slice context encoder, the Stage-1 layout field and
the Stage-2 expression field, the foundation-model teacher, and all losses, into a
model that trains per stack (leave-one-slice-out self-supervision + distillation) and
is queried at arbitrary continuous ``z`` (``generate_virtual_slice``). Positions and
types are produced first (Stage 1); expression is produced conditioned on the layout
(Stage 2). If PyTorch is unavailable or training fails, a deterministic OT-morph
fallback keeps the method usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SpatialCPAv11Config
from .data import Slice, SliceStack


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


def _pca(train_expr, d):
    X = np.asarray(train_expr, np.float64)
    mean = X.mean(0, keepdims=True); std = X.std(0, keepdims=True); std[std == 0] = 1.0
    Xc = (X - mean) / std
    d = int(min(d, min(Xc.shape) - 1)) if min(Xc.shape) > 1 else 1
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:max(d, 1)]
    return lambda E: (((np.asarray(E, np.float64) - mean) / std) @ comps.T).astype(np.float32)


class SpatialCPAv11:
    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv11Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv11Config()
        self.n_types = max(stack.n_cell_types() or 1, 1)
        self.n_genes = stack.n_genes
        self.trained = False
        # normalization
        allxy = np.concatenate([s.coords_xy for s in stack.slices], 0).astype(np.float64)
        self._xy_c = allxy.mean(0); self._xy_s = (allxy.max(0) - allxy.min(0)) / 2 + 1e-6
        zc = stack.z_centers()
        self._z_c = float(zc.mean()); self._z_s = float((zc.max() - zc.min()) / 2 + 1e-6)
        self._embed = _pca(stack.union_expression(), self.cfg.context.expr_embed_dim)
        # actual embedding dim (PCA is capped by n_genes / n_cells)
        self._embed_dim = int(self._embed(stack.slices[0].expression).shape[1])
        self._fit()

    # normalization helpers
    def _nxy(self, xy):
        return (np.asarray(xy, np.float64) - self._xy_c) / self._xy_s
    def _dxy(self, nxy):
        return np.asarray(nxy, np.float64) * self._xy_s + self._xy_c
    def _nz(self, z):
        return (float(z) - self._z_c) / self._z_s

    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_oom(e):
        return ("out of memory" in str(e).lower()
                or e.__class__.__name__ in ("OutOfMemoryError", "AcceleratorError"))

    def _fit(self):
        try:
            import torch  # noqa: F401
        except Exception as e:
            print(f"[spatialcpav11] PyTorch unavailable ({e}); OT-morph fallback.")
            return
        from .trainer import train_model
        # Attempt on the selected device; on CUDA out-of-memory retry on CPU (the
        # neural model then still trains, just slower — better than the trivial
        # nearest-slice fallback). Only give up to the OT-morph fallback if CPU fails.
        self._force_cpu = False
        for attempt in ("device", "cpu"):
            try:
                train_model(self)
                self.trained = True
                return
            except Exception as e:
                oom = self._is_oom(e)
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if attempt == "device" and oom and not self._force_cpu:
                    print(f"[spatialcpav11] CUDA out of memory; retrying training on CPU. "
                          f"(free a GPU or pass --device cpu to skip this.)")
                    self._force_cpu = True
                    continue
                import traceback
                print(f"[spatialcpav11] training failed ({e}); OT-morph fallback.")
                if not self.cfg.train.fallback_on_error:
                    raise
                traceback.print_exc()
                self.trained = False
                return

    # ------------------------------------------------------------------ #
    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        if self.trained:
            try:
                from .trainer import infer_slice
                return infer_slice(self, z)
            except Exception as e:
                print(f"[spatialcpav11] inference failed ({e}); OT-morph fallback.")
        return self._fallback(z)

    # ------------------------------------------------------------------ #
    def _fallback(self, z):
        """Dependency-free fallback: copy the nearest real slice at the target z."""
        lower, upper = self.stack.pick_flanking_slices(z)
        anchor = lower if abs(lower.z_center - z) <= abs(upper.z_center - z) else upper
        if anchor.n_spots == 0:
            return VirtualSlice(np.zeros((0, 3), np.float32), np.zeros((0, self.n_genes), np.float32))
        coords = np.column_stack([anchor.coords_xy.astype(np.float64),
                                  np.full(anchor.n_spots, float(z))]).astype(np.float32)
        ct = anchor.cell_type_indices
        labels = (np.array([self.cell_type_names[i] for i in ct]) if (ct is not None and self.cell_type_names)
                  else (ct.astype(str) if ct is not None else None))
        return VirtualSlice(coords, anchor.expression.astype(np.float32), labels, ct)
