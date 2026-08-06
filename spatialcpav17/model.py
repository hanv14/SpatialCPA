"""
SpatialCPA-v17 — orchestration for the fully-generative latent flow-matching atlas.

Sets up the joint VAE inputs (library-normalized log expression + morphology channels),
the (x, y, z) normalization, trains the two phases, and generates virtual slices by
**decoding** the flow-generated latent. A dependency-free fallback keeps the method usable
without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .config import SpatialCPAv17Config
from .data import Slice, SliceStack
from .latents import morphology_features


@dataclass
class VirtualSlice:
    coords: np.ndarray
    expression: np.ndarray
    cell_type: Optional[np.ndarray] = None
    cell_type_idx: Optional[np.ndarray] = None


class SpatialCPAv17:
    def __init__(self, stack: SliceStack, gene_names: Sequence[str],
                 cell_type_names: Optional[Sequence[str]] = None,
                 cfg: Optional[SpatialCPAv17Config] = None,
                 is_counts: bool = True) -> None:
        self.stack = stack
        self.gene_names = list(gene_names)
        self.cell_type_names = list(cell_type_names) if cell_type_names is not None else None
        self.cfg = cfg or SpatialCPAv17Config()
        self.n_types = max(stack.n_cell_types() or 1, 1)
        self.n_genes = stack.n_genes
        self.n_morph = max(self.n_types, 1) + 1
        self.is_counts = bool(is_counts)
        lk = self.cfg.vae.likelihood
        self.likelihood = "nb" if (lk == "nb" or (lk == "auto" and self.is_counts)) else "gaussian"
        self.trained = False

        union = np.asarray(stack.union_expression(), dtype=np.float64)
        # encoder input transform (library-normalized log for counts; identity for gaussian)
        if self.likelihood == "nb":
            union = np.clip(union, 0.0, None)
            xin_u = self._logmicro(union)
        else:
            xin_u = union
        self._xin_mean = xin_u.mean(0); self._xin_std = xin_u.std(0) + 1e-6

        # morphology standardization
        m_list = [morphology_features(s.coords_xy, s.cell_type_indices, self.n_types,
                                      k=self.cfg.latent.morph_k,
                                      density_sigma=self.cfg.latent.density_sigma)
                  for s in stack.slices]
        m_u = np.vstack(m_list) if m_list else np.zeros((0, self.n_morph), np.float32)
        self._m_mean = m_u.mean(0); self._m_std = m_u.std(0) + 1e-6

        # spatial normalization
        allxy = np.concatenate([s.coords_xy for s in stack.slices], 0).astype(np.float64)
        self._xy_c = allxy.mean(0); self._xy_s = (allxy.max(0) - allxy.min(0)) / 2 + 1e-6
        zc = stack.z_centers()
        self._z_c = float(zc.mean()); self._z_s = float((zc.max() - zc.min()) / 2 + 1e-6)
        self._fit()

    # ── transforms ────────────────────────────────────────────────────────────
    def _logmicro(self, counts):
        """Library-normalized log for counts: log1p(counts / library * 1e4)."""
        c = np.asarray(counts, np.float64)
        lib = c.sum(1, keepdims=True); lib[lib == 0] = 1.0
        return np.log1p(c / lib * 1e4)

    def xin(self, expr):
        """Encoder input (standardized) for a batch of cells."""
        x = self._logmicro(expr) if self.likelihood == "nb" else np.asarray(expr, np.float64)
        return ((x - self._xin_mean) / self._xin_std).astype(np.float32)

    def library(self, expr):
        if self.likelihood == "nb":
            lib = np.asarray(expr, np.float64).clip(0, None).sum(1)
            lib[lib == 0] = 1.0
            return lib.astype(np.float32)
        return np.ones(np.asarray(expr).shape[0], np.float32)

    def morph(self, s):
        m = morphology_features(s.coords_xy, s.cell_type_indices, self.n_types,
                                k=self.cfg.latent.morph_k, density_sigma=self.cfg.latent.density_sigma)
        return ((m - self._m_mean) / self._m_std).astype(np.float32)

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
            print(f"[spatialcpav17] PyTorch unavailable ({e}); decode fallback.")
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
                    print("[spatialcpav17] CUDA out of memory; retrying on CPU.")
                    self._force_cpu = True
                    continue
                import traceback
                print(f"[spatialcpav17] training failed ({e}); decode fallback.")
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
                print(f"[spatialcpav17] generation failed ({e}); fallback.")
                import traceback; traceback.print_exc()
        return self._fallback(z)

    def _fallback(self, z):
        """Dependency-free fallback: resample the flanking supports and copy the nearest
        real profile (mean of the two bracketing cells) — a coherent recombination."""
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
        w = np.concatenate([np.full(len(lo_xy), max(1 - t, 1e-3)), np.full(len(hi_xy), max(t, 1e-3))])
        w = w / w.sum()
        sel = rng.choice(props.shape[0], n_target, replace=props.shape[0] < n_target, p=w)
        pool_expr = np.vstack([np.asarray(lower.expression), np.asarray(upper.expression)])
        pool_type = np.concatenate([
            lower.cell_type_indices if lower.cell_type_indices is not None else np.zeros(lower.n_spots, int),
            upper.cell_type_indices if upper.cell_type_indices is not None else np.zeros(upper.n_spots, int)])
        expr = pool_expr[sel].astype(np.float32)
        ct = pool_type[sel] if self.n_types >= 2 else None
        anchor = props[sel]
        coords = np.column_stack([self._dxy(anchor), np.full(n_target, float(z))]).astype(np.float32)
        labels = (np.array([self.cell_type_names[i] for i in ct]) if (ct is not None and self.cell_type_names)
                  else (ct.astype(str) if ct is not None else None))
        return VirtualSlice(coords, expr, labels, ct)
