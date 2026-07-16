"""
Stage 1 of H3D-FLA — preprocessing, expression latents, and pseudo-image channels.

Two products per cell, computed from the *training-only* slices:

* **Expression latent** ``e`` (``LatentConfig.expr_latent_dim``): a compact molecular
  code obtained by standardizing genes and taking a truncated-SVD (PCA) projection. The
  fitted mean/scale/components are stored so any cell (or a flow-decoded latent) can be
  mapped to/from gene space.
* **Pseudo-image / morphological channels** ``m``: the pipeline's multi-channel image
  stack sampled *at each cell* instead of rasterized to a grid — a soft local cell-type
  composition map (one channel per type, à la the tumor / stroma / immune probability
  maps) plus a local cell-density channel. These carry the morphology the joint encoder
  fuses with the molecular latent, and the edge structure the biology-informed
  regularizers read.

Everything here is pure numpy/scipy so Stage 1 runs with or without PyTorch.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class ExpressionLatent:
    """Standardize-then-PCA expression encoder with an (approximate) inverse."""

    def __init__(self, dim: int = 32, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)
        self.mean_ = None
        self.scale_ = None
        self.components_ = None   # (dim, n_genes)

    def fit(self, X: np.ndarray) -> "ExpressionLatent":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(0)
        self.scale_ = X.std(0) + 1e-6
        Xs = (X - self.mean_) / self.scale_
        d = min(self.dim, min(Xs.shape) - 1) if min(Xs.shape) > 1 else 1
        d = max(d, 1)
        try:
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=d, random_state=self.seed)
            svd.fit(Xs)
            comp = svd.components_
        except Exception:
            # numpy SVD fallback
            U, S, Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
            comp = Vt[:d]
        self.components_ = np.ascontiguousarray(comp, dtype=np.float32)   # (d, G)
        self.dim = comp.shape[0]
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return (Xs @ self.components_.T).astype(np.float32)

    def decode(self, E: np.ndarray) -> np.ndarray:
        """Approximate inverse: latent -> expression (non-negative, log-scale)."""
        Xs = np.asarray(E, dtype=np.float64) @ self.components_
        X = Xs * self.scale_ + self.mean_
        return np.clip(X, 0.0, None).astype(np.float32)


def morphology_features(coords_xy: np.ndarray, cell_type_idx, n_types: int,
                        k: int = 12, density_sigma: float = 1.0) -> np.ndarray:
    """Per-cell pseudo-image channels: soft local cell-type map + local density.

    Returns an ``(n, n_types + 1)`` array: the first ``n_types`` columns are the
    kNN-smoothed cell-type composition around each cell (the soft probability maps), the
    last column is a normalized local density. Computed within a single slice.
    """
    xy = np.asarray(coords_xy, dtype=np.float64)
    n = xy.shape[0]
    ncols = max(n_types, 1) + 1
    if n == 0:
        return np.zeros((0, ncols), dtype=np.float32)
    kk = min(k + 1, n)
    tree = cKDTree(xy)
    dist, idx = tree.query(xy, k=kk)
    if dist.ndim == 1:
        dist, idx = dist[:, None], idx[:, None]
    dist, idx = dist[:, 1:], idx[:, 1:]        # drop self
    med = np.median(dist) + 1e-9
    w = np.exp(-(dist / (density_sigma * med)) ** 2)         # (n, k)
    wsum = w.sum(1, keepdims=True) + 1e-9

    feats = np.zeros((n, ncols), dtype=np.float32)
    if cell_type_idx is not None and n_types >= 1:
        t = np.asarray(cell_type_idx, dtype=np.int64)
        neigh_t = t[idx]                                     # (n, k)
        for c in range(n_types):
            feats[:, c] = ((neigh_t == c) * w).sum(1) / wsum[:, 0]
    # local density channel: inverse mean neighbor distance, robustly scaled
    dens = 1.0 / (dist.mean(1) + med)
    feats[:, -1] = (dens / (np.median(dens) + 1e-9)).astype(np.float32)
    return feats
