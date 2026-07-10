"""
Frozen multimodal foundation-model *teacher* for Stage-1 layout distillation.

The layout generator is trained primarily by knowledge distillation from a frozen
multimodal FM (OmiCLIP / Path2Space) originally trained on paired H&E + ST. Even
though our serial data has **no images**, such a teacher — through its ST/expression
tower — provides *layout-related* supervision: (i) a per-region **embedding** the
student's layout code is aligned to (feature distillation), and (ii) a **pseudo-layout**
(spatial-domain segmentation) the student's type/region field is distilled toward.

Real teachers are plugged in via :func:`register_teacher` (map name -> builder that
returns an object with ``embed(expr, xy)`` and ``domains(expr, xy)``); point
``TeacherConfig.weights_path`` at the checkpoint. When the FM is unavailable, a
**data-derived proxy** stands in (documented approximation, not the real FM): a
spatially-smoothed expression embedding and a spatial-domain clustering computed from
the training slices — the same *kind* of layout-aware signal an ST foundation model
provides, so distillation runs end-to-end and the real FM can be swapped in for a
publication-grade result.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.spatial import cKDTree

TEACHER_REGISTRY: Dict[str, Callable] = {}


def register_teacher(name: str, builder: Callable) -> None:
    """Register a real FM teacher builder: ``builder(cfg) -> teacher`` with
    ``embed(expr, xy) -> (n, d)`` and ``domains(expr, xy) -> (n,) int``."""
    TEACHER_REGISTRY[name] = builder


def _spatial_smooth(expr, xy, k=12):
    """Mean over spatial kNN — a spatial-context (layout-aware) expression profile."""
    n = expr.shape[0]
    if n < k + 1:
        return expr.copy()
    _, nn = cKDTree(xy).query(xy, k=min(k + 1, n))
    return expr[nn].mean(axis=1)


class ProxyTeacher:
    """Data-derived stand-in for OmiCLIP / Path2Space (documented approximation).

    Fits, on the training slices, (i) a PCA over spatially-smoothed expression whose
    projection is a layout-aware per-spot embedding, and (ii) KMeans over that
    embedding giving pseudo spatial domains. Both are what an ST foundation model's
    layout signal approximates; the real teacher replaces this class.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._pca = None
        self._km = None

    def fit(self, stack):
        feats = []
        for s in stack.slices:
            feats.append(_spatial_smooth(np.asarray(s.expression, np.float64),
                                         np.asarray(s.coords_xy, np.float64)))
        F = np.concatenate(feats, axis=0)
        mean = F.mean(0, keepdims=True); std = F.std(0, keepdims=True); std[std == 0] = 1.0
        self._mean, self._std = mean, std
        Fz = (F - mean) / std
        d = int(min(self.cfg.embed_dim, min(Fz.shape) - 1)) if min(Fz.shape) > 1 else 1
        _, _, Vt = np.linalg.svd(Fz, full_matrices=False)
        self._comps = Vt[:max(d, 1)]
        E = (Fz @ self._comps.T).astype(np.float32)
        try:
            from sklearn.cluster import KMeans
            self._km = KMeans(n_clusters=min(self.cfg.n_pseudo_domains, max(2, E.shape[0] // 20)),
                              n_init=4, random_state=0).fit(E)
        except Exception:
            self._km = None
        return self

    def embed(self, expr, xy):
        Fs = _spatial_smooth(np.asarray(expr, np.float64), np.asarray(xy, np.float64))
        E = ((Fs - self._mean) / self._std) @ self._comps.T
        return np.ascontiguousarray(E, dtype=np.float32)

    def domains(self, expr, xy):
        E = self.embed(expr, xy)
        if self._km is None:
            return np.zeros(E.shape[0], dtype=int)
        return self._km.predict(E).astype(int)


def build_teacher(cfg, stack):
    """Instantiate the teacher (real FM if available, else the proxy stand-in)."""
    kind = cfg.kind
    if kind in ("omiclip", "path2space") or (kind == "auto" and cfg.name in TEACHER_REGISTRY
                                             and cfg.weights_path):
        name = cfg.name if kind == "auto" else kind
        builder = TEACHER_REGISTRY.get(name)
        if builder is not None:
            try:
                t = builder(cfg)
                print(f"[spatialcpav11] teacher: pretrained {name} ({cfg.weights_path}).")
                return t
            except Exception as e:  # pragma: no cover - depends on external asset
                print(f"[spatialcpav11] teacher {name} failed ({e}); using proxy stand-in.")
    print("[spatialcpav11] teacher: data-derived proxy stand-in "
          "(swap in OmiCLIP/Path2Space weights for a publication result).")
    return ProxyTeacher(cfg).fit(stack)
