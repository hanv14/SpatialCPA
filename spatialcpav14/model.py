"""
SpatialCPA-v14 / H3D-FLA — orchestration.

Ties together Stage 1 (expression latent + pseudo-image channels), Stage 2 (joint
encoder), Stage 3 (3D-attention context + conditional flow matching), Stage 4 (biology
regularizers) and Stage 5 (inference). Fits per stack and is queried at arbitrary
continuous ``z`` via :meth:`generate_virtual_slice`. If PyTorch is unavailable or training
fails, a dependency-free latent-grounded fallback keeps the method usable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SpatialCPAv14Config
from .data import Slice, SliceStack
from .latents import ExpressionLatent, morphology_features


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


class SpatialCPAv14:
    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv14Config] = None) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv14Config()
        self.n_types = max(stack.n_cell_types() or 1, 1)
        self.n_genes = stack.n_genes
        self.n_morph = max(self.n_types, 1) + 1
        self.trained = False

        # Stage 1: fit the expression latent on the union of training expression.
        self.expr_latent = ExpressionLatent(self.cfg.latent.expr_latent_dim, self.cfg.seed)
        union = stack.union_expression()
        self.expr_latent.fit(union)

        # standardization stats for the encoder inputs (e and morphology m)
        e_union = self.expr_latent.encode(union)
        self._e_mean = e_union.mean(0); self._e_std = e_union.std(0) + 1e-6
        m_list = [morphology_features(s.coords_xy, s.cell_type_indices, self.n_types,
                                      k=self.cfg.latent.morph_k,
                                      density_sigma=self.cfg.latent.density_sigma)
                  for s in stack.slices]
        m_union = np.vstack(m_list) if m_list else np.zeros((0, self.n_morph), np.float32)
        self._m_mean = m_union.mean(0); self._m_std = m_union.std(0) + 1e-6

        # spatial normalization for the (x, y, z) query encoder
        allxy = np.concatenate([s.coords_xy for s in stack.slices], 0).astype(np.float64)
        self._xy_c = allxy.mean(0); self._xy_s = (allxy.max(0) - allxy.min(0)) / 2 + 1e-6
        zc = stack.z_centers()
        self._z_c = float(zc.mean()); self._z_s = float((zc.max() - zc.min()) / 2 + 1e-6)
        self._fit()

    # normalization helpers
    def _nxy(self, xy):
        return (np.asarray(xy, np.float64) - self._xy_c) / self._xy_s
    def _dxy(self, nxy):
        return np.asarray(nxy, np.float64) * self._xy_s + self._xy_c
    def _nz(self, z):
        return (float(z) - self._z_c) / self._z_s

    @staticmethod
    def _is_oom(e):
        return ("out of memory" in str(e).lower()
                or e.__class__.__name__ in ("OutOfMemoryError", "AcceleratorError"))

    def _fit(self):
        try:
            import torch  # noqa: F401
        except Exception as e:
            print(f"[spatialcpav14] PyTorch unavailable ({e}); latent-grounded fallback.")
            return
        from .trainer import train_model
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
                    print("[spatialcpav14] CUDA out of memory; retrying training on CPU.")
                    self._force_cpu = True
                    continue
                import traceback
                print(f"[spatialcpav14] training failed ({e}); latent-grounded fallback.")
                if not self.cfg.train.fallback_on_error:
                    raise
                traceback.print_exc()
                self.trained = False
                return

    def generate_virtual_slice(self, z: float) -> VirtualSlice:
        if self.trained:
            try:
                from .trainer import generate_slice
                return generate_slice(self, z)
            except Exception as e:
                print(f"[spatialcpav14] generation failed ({e}); latent-grounded fallback.")
                import traceback; traceback.print_exc()
        return self._fallback(z)

    def _fallback(self, z):
        """Dependency-free fallback: resample the flanking supports at the interpolated
        ratio and ground each cell in the molecularly-nearest local real profile. Uses the
        Stage-1 expression latent (no torch) — a coherent recombination of both flanking
        slices, not a raw copy."""
        from scipy.spatial import cKDTree
        lower, upper = self.stack.pick_flanking_slices(z)
        if lower.n_spots == 0 and upper.n_spots == 0:
            return VirtualSlice(np.zeros((0, 3), np.float32), np.zeros((0, self.n_genes), np.float32))
        zl, zh = lower.z_center, upper.z_center
        t = 0.5 if zh == zl else float(np.clip((z - zl) / (zh - zl), 0, 1))
        n_target = max(int(round((1 - t) * lower.n_spots + t * upper.n_spots)), 1)
        rng = np.random.default_rng(self.cfg.seed)

        lo_xy = self._nxy(lower.coords_xy); hi_xy = self._nxy(upper.coords_xy)
        props = np.vstack([lo_xy, hi_xy])
        w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)),
                            np.full(len(hi_xy), max(t, 1e-3))])
        w = w / w.sum()
        sel = rng.choice(props.shape[0], n_target, replace=True, p=w)
        med = np.median([lower.median_spacing(), upper.median_spacing()])
        anchor = props[sel] + rng.standard_normal((n_target, 2)) * (0.25 * med / self._xy_s.mean())

        pool_xy = props
        pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
        pool_e = self.expr_latent.encode(pool_expr)
        pool_type = np.concatenate([
            lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
            upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
        # target latent per cell = interpolation of the two nearest flanking latents
        K = min(self.cfg.generation.ground_k, pool_xy.shape[0])
        _, cand = cKDTree(pool_xy).query(anchor, k=K)
        if cand.ndim == 1:
            cand = cand[:, None]
        anchor_e = pool_e[cand].mean(1)                     # local mean latent
        expr = np.empty((n_target, self.n_genes), np.float32)
        ct = np.empty(n_target, np.int64)
        for i in range(n_target):
            ci = cand[i]
            d = np.linalg.norm(pool_e[ci] - anchor_e[i], axis=1)
            p = np.exp(-(d - d.min()) / max(self.cfg.generation.ground_temp, 1e-3)); p /= p.sum()
            pick = ci[rng.choice(len(ci), p=p)]
            expr[i] = pool_expr[pick]; ct[i] = pool_type[pick]
        if self.cfg.generation.output_counts:
            expr = np.expm1(np.clip(expr, 0.0, 20.0)).astype(np.float32)
        coords = np.column_stack([self._dxy(anchor), np.full(n_target, float(z))]).astype(np.float32)
        ct_idx = ct if self.n_types >= 2 else None
        labels = (np.array([self.cell_type_names[i] for i in ct_idx])
                  if (ct_idx is not None and self.cell_type_names)
                  else (ct_idx.astype(str) if ct_idx is not None else None))
        return VirtualSlice(coords, expr, labels, ct_idx)
