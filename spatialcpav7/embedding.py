"""
Cell-state embeddings for SpatialCPA-v7.

The embedding is the shared latent space in which (a) the fused-transport cost
between flanking slices is computed and (b) cell types are assigned. It is where
*external* biological knowledge enters the otherwise self-contained
interpolation: a foundation model trained on large spatial-omics (and, when
available, paired H&E) corpora provides a cell-state manifold that is robust to a
single panel's technical noise.

Two things are new relative to v6:

* :func:`mutual_nn_align` — a lightweight, deterministic **cross-slice batch
  anchoring**. Adjacent physical sections are different imaging batches; a raw
  shared embedding can carry a per-slice offset that misleads both the transport
  cost and the cell-type prior. Aligning the flanking slices by their
  mutual-nearest-neighbour anchors removes that offset. Leakage-safe: it aligns
  only the two *training* flanking slices to each other.
* The builder additionally exposes ``coexpr`` and the FM hooks used by the
  label-propagation classifier in :mod:`spatialcpav7.annotation`.

Every method degrades gracefully — no external asset is required to run.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Optional

import numpy as np

# Registry for full foundation-model encoders. A builder maps
#   (train_expression: (N,G), gene_names: list[str], cfg) -> Callable[[X], (N,d)]
EMBEDDER_REGISTRY: Dict[str, Callable] = {}


def register_embedder(name: str, builder: Callable) -> None:
    """Register a foundation-model embedder builder under ``name``."""
    EMBEDDER_REGISTRY[name] = builder


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _select_hvg(X: np.ndarray, max_hvg: int) -> np.ndarray:
    G = X.shape[1]
    if G <= max_hvg:
        return np.arange(G)
    v = X.var(axis=0)
    return np.sort(np.argsort(v)[-max_hvg:])


def _pca_fit(X: np.ndarray, n_components: int, standardize: bool, whiten: bool):
    mean = X.mean(axis=0)
    Xc = X - mean
    if standardize:
        scale = Xc.std(axis=0)
        scale[scale == 0] = 1.0
        Xc = Xc / scale
    else:
        scale = np.ones(X.shape[1], dtype=X.dtype)
    n_components = int(min(n_components, min(Xc.shape) - 1)) if min(Xc.shape) > 1 else 1
    n_components = max(n_components, 1)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]
    sv = S[:n_components]
    denom = np.where(sv > 0, sv, 1.0)
    return mean, scale, comps, denom, whiten


def _pca_apply(X, mean, scale, comps, denom, whiten):
    Z = ((X - mean) / scale) @ comps.T
    if whiten:
        Z = Z / (denom / np.sqrt(max(X.shape[0], 1)) + 1e-8)
    return np.ascontiguousarray(Z, dtype=np.float32)


def _load_gene_embedding(path: Optional[str], gene_names, dim: int):
    path = path or os.environ.get("SPATIALCPAV7_FM_GENE_EMBEDDING") \
        or os.environ.get("SPATIALCPAV6_FM_GENE_EMBEDDING")
    if not path or not os.path.exists(path):
        return None
    genes = [str(g) for g in gene_names]
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        vocab = [str(g) for g in d["genes"]]
        emb = np.asarray(d["embedding"], dtype=np.float32)
    else:
        emb = np.asarray(np.load(path), dtype=np.float32)
        if emb.shape[0] != len(genes):
            return None
        vocab = genes
    if dim and dim < emb.shape[1]:
        emb = emb[:, :dim]
    lut = {g: i for i, g in enumerate(vocab)}
    W = np.zeros((len(genes), emb.shape[1]), dtype=np.float32)
    for j, g in enumerate(genes):
        i = lut.get(g)
        if i is not None:
            W[j] = emb[i]
    if not np.any(W):
        return None
    return W


class Embedder:
    """A fitted cell-state embedder shared across all slices of a stack."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray], name: str):
        self._fn = fn
        self.name = name

    def __call__(self, expression: np.ndarray) -> np.ndarray:
        Z = self._fn(np.asarray(expression, dtype=np.float32))
        return np.ascontiguousarray(Z, dtype=np.float32)


def build_embedder(train_expression: np.ndarray, gene_names, cfg) -> Embedder:
    """Fit the embedder requested by ``cfg`` on the training expression."""
    method = cfg.method
    X = np.asarray(train_expression, dtype=np.float32)

    if method in EMBEDDER_REGISTRY:
        try:
            fn = EMBEDDER_REGISTRY[method](X, list(gene_names), cfg)
            return Embedder(fn, method)
        except Exception as e:  # pragma: no cover - depends on external asset
            print(f"[spatialcpav7.embedding] registered embedder '{method}' failed "
                  f"({e}); falling back to PCA.")
            method = "pca"

    hvg = _select_hvg(X, cfg.max_hvg)
    Xh = X[:, hvg]

    def _make_pca():
        p = _pca_fit(Xh, cfg.n_components, cfg.standardize, cfg.whiten)
        return lambda E: _pca_apply(E[:, hvg], *p)

    def _make_fm_gene():
        W = _load_gene_embedding(cfg.fm_gene_embedding_path, gene_names,
                                 cfg.fm_gene_embedding_dim)
        if W is None:
            return None
        col_mean = X.mean(axis=0, keepdims=True)
        col_std = X.std(axis=0, keepdims=True)
        col_std[col_std == 0] = 1.0

        def fn(E):
            Zc = (np.asarray(E, dtype=np.float32) - col_mean) / col_std
            Z = Zc @ W
            Z = Z - Z.mean(axis=0, keepdims=True)
            s = Z.std(axis=0, keepdims=True)
            s[s == 0] = 1.0
            return (Z / s).astype(np.float32)
        return fn

    def _make_coexpr():
        from .foundation_assets import build_coexpression_embedding
        _, W = build_coexpression_embedding(X, gene_names, cfg.n_components)
        col_mean = X.mean(axis=0, keepdims=True)
        col_std = X.std(axis=0, keepdims=True); col_std[col_std == 0] = 1.0

        def fn(E):
            Z = ((np.asarray(E, dtype=np.float32) - col_mean) / col_std) @ W
            Z = Z - Z.mean(axis=0, keepdims=True)
            s = Z.std(axis=0, keepdims=True); s[s == 0] = 1.0
            return (Z / s).astype(np.float32)
        return fn

    if method == "pca":
        return Embedder(_make_pca(), "pca")
    if method == "coexpr":
        return Embedder(_make_coexpr(), "coexpr")
    if method == "fm_gene":
        fn = _make_fm_gene()
        if fn is None:
            print("[spatialcpav7.embedding] no gene-embedding matrix found; "
                  "falling back to PCA.")
            return Embedder(_make_pca(), "pca(fallback)")
        return Embedder(fn, "fm_gene")
    if method == "concat":
        pca_fn = _make_pca()
        fm_fn = _make_fm_gene()
        if fm_fn is None:
            print("[spatialcpav7.embedding] concat: gene-embedding matrix missing; "
                  "using PCA only.")
            return Embedder(pca_fn, "pca(concat-fallback)")
        return Embedder(lambda E: np.concatenate([pca_fn(E), fm_fn(E)], axis=1),
                        "concat")
    raise ValueError(f"Unknown embedding method '{method}'")


# --------------------------------------------------------------------------- #
# Cross-slice manifold anchoring (mutual nearest neighbours)                    #
# --------------------------------------------------------------------------- #
def mutual_nn_align(lo_e: np.ndarray, up_e: np.ndarray, k: int = 15,
                    strength: float = 1.0):
    """Align two slice embeddings by their mutual-nearest-neighbour anchors.

    Finds pairs ``(i, j)`` that are each other's nearest neighbours across slices
    (robust, correspondence-free MNN anchors à la batch integration), estimates
    the mean embedding offset over those anchors, and shifts the upper slice by
    ``strength`` × that offset onto the lower slice's frame. Returns the aligned
    ``(lo_e, up_e)``. Deterministic, cheap, and leakage-safe (uses only the two
    training flanking slices). Falls back to the identity if too few anchors.
    """
    from scipy.spatial import cKDTree
    lo_e = np.asarray(lo_e, dtype=np.float64)
    up_e = np.asarray(up_e, dtype=np.float64)
    if strength <= 0 or lo_e.shape[0] < k + 1 or up_e.shape[0] < k + 1:
        return lo_e.astype(np.float32), up_e.astype(np.float32)

    kk = min(k, lo_e.shape[0] - 1, up_e.shape[0] - 1)
    tree_up = cKDTree(up_e)
    tree_lo = cKDTree(lo_e)
    # nearest upper for each lower, nearest lower for each upper
    _, lo2up = tree_up.query(lo_e, k=1)
    _, up2lo = tree_lo.query(up_e, k=1)
    lo2up = np.atleast_1d(lo2up)
    up2lo = np.atleast_1d(up2lo)
    # mutual pairs: lower i -> upper j and upper j -> lower i
    mutual = [(i, j) for i, j in enumerate(lo2up) if up2lo[j] == i]
    if len(mutual) < max(3, kk // 2):
        return lo_e.astype(np.float32), up_e.astype(np.float32)
    mi = np.array([m[0] for m in mutual])
    mj = np.array([m[1] for m in mutual])
    offset = (lo_e[mi] - up_e[mj]).mean(axis=0)      # move up onto lo
    up_aligned = up_e + strength * offset[None, :]
    return lo_e.astype(np.float32), up_aligned.astype(np.float32)
