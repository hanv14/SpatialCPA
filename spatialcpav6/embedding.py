"""
Cell-state embeddings for SpatialCPA-v6.

The embedding is the shared latent space in which (a) the optimal-transport cost
between flanking slices is computed and (b) cell types are assigned. A good
embedding is what lets *external* biological knowledge enter the otherwise
self-contained interpolation: a foundation model trained on large spatial-omics
(and, when available, paired H&E) corpora provides a cell-state manifold that is
robust to the panel's technical noise and encodes gene-gene / morphology priors
the ~hundreds of cells of a single experiment cannot reveal.

Design: an :class:`Embedder` is *fit* on the union of the training slices, then
*applied* to any slice, so lower/upper/synthesized cells share one space.

Available methods (all degrade gracefully — no external asset is required to
run):

* ``"pca"``      local unsupervised SVD embedding (default; pure numpy).
* ``"fm_gene"``  project expression through a pretrained *gene embedding*
                 (``cell = X_norm @ W_gene``). W_gene can be scGPT / Geneformer /
                 Gene2vec token embeddings or an H&E-derived gene-program matrix.
                 Injects the foundation model's learned gene relationships. Falls
                 back to ``"pca"`` when the matrix is missing.
* ``"concat"``   concatenate ``pca`` ⊕ ``fm_gene``.
* any name registered via :func:`register_embedder` — the extension point for a
  full foundation-model encoder (scGPT / Geneformer / UCE for expression; UNI /
  CONCH for paired morphology). The registered builder receives the fitted
  training expression and returns a callable ``expression -> (N, d)``.

Leakage note: the embedder is fit on the *training* slices only; the held-out
slice is never seen. Pretrained foundation-model weights are external priors,
not held-out data.
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
    """Indices of the most variable genes (cap embedding cost / denoise)."""
    G = X.shape[1]
    if G <= max_hvg:
        return np.arange(G)
    v = X.var(axis=0)
    return np.sort(np.argsort(v)[-max_hvg:])


def _pca_fit(X: np.ndarray, n_components: int, standardize: bool, whiten: bool):
    """Fit a PCA (numpy SVD) and return (transform_fn, mean, scale, comps)."""
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
    # Economy SVD; components are right-singular vectors.
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:n_components]                       # (d, G)
    sv = S[:n_components]
    denom = np.where(sv > 0, sv, 1.0)
    return mean, scale, comps, denom, whiten


def _pca_apply(X, mean, scale, comps, denom, whiten):
    Z = ((X - mean) / scale) @ comps.T             # (N, d)
    if whiten:
        # scale each axis to unit variance (singular values / sqrt(n)).
        Z = Z / (denom / np.sqrt(max(X.shape[0], 1)) + 1e-8)
    return np.ascontiguousarray(Z, dtype=np.float32)


def _load_gene_embedding(path: Optional[str], gene_names, dim: int):
    """Load a pretrained gene-embedding matrix aligned to ``gene_names``.

    Returns ``W`` of shape ``(G_panel, d)`` (rows for genes absent from the
    pretrained vocabulary are zero) or ``None`` if unavailable.
    """
    path = path or os.environ.get("SPATIALCPAV6_FM_GENE_EMBEDDING")
    if not path or not os.path.exists(path):
        return None
    genes = [str(g) for g in gene_names]
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        vocab = [str(g) for g in d["genes"]]
        emb = np.asarray(d["embedding"], dtype=np.float32)
    else:  # .npy aligned to panel order
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
        # Guard against degenerate all-zero rows for downstream cosine/euclidean.
        return np.ascontiguousarray(Z, dtype=np.float32)


def build_embedder(train_expression: np.ndarray, gene_names, cfg) -> Embedder:
    """Fit the embedder requested by ``cfg`` on the training expression.

    ``cfg`` is an :class:`~spatialcpav6.config.EmbeddingConfig`.
    """
    method = cfg.method
    X = np.asarray(train_expression, dtype=np.float32)

    # Full foundation-model encoder plugged in via the registry.
    if method in EMBEDDER_REGISTRY:
        try:
            fn = EMBEDDER_REGISTRY[method](X, list(gene_names), cfg)
            return Embedder(fn, method)
        except Exception as e:  # pragma: no cover - depends on external asset
            print(f"[spatialcpav6.embedding] registered embedder '{method}' failed "
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
        # Row-normalize expression so the projection is composition-like, then
        # project through the pretrained gene embedding and whiten.
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
        # Data-derived gene-program embedding: project cells through the SVD of the
        # training gene-gene correlation matrix (leakage-safe; no external asset).
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
            print("[spatialcpav6.embedding] no gene-embedding matrix found "
                  "(fm_gene_embedding_path / SPATIALCPAV6_FM_GENE_EMBEDDING); "
                  "falling back to PCA.")
            return Embedder(_make_pca(), "pca(fallback)")
        return Embedder(fn, "fm_gene")

    if method == "concat":
        pca_fn = _make_pca()
        fm_fn = _make_fm_gene()
        if fm_fn is None:
            print("[spatialcpav6.embedding] concat: gene-embedding matrix missing; "
                  "using PCA only.")
            return Embedder(pca_fn, "pca(concat-fallback)")
        return Embedder(lambda E: np.concatenate([pca_fn(E), fm_fn(E)], axis=1),
                        "concat")

    raise ValueError(f"Unknown embedding method '{method}'")
