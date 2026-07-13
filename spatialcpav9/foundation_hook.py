"""
Pretrained gene-embedding loader for SpatialCPA-v9.

Loads a pretrained gene-embedding matrix (scGPT / Geneformer / Gene2vec, converted
to the ``.npz`` ``genes``/``embedding`` format used across SpatialCPA) aligned to
the panel, for warm-starting the autoencoder encoder (an external foundation-model
prior on gene relationships). Returns ``None`` when unavailable so the model simply
trains the encoder from scratch.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np


def load_gene_embedding(path: Optional[str], gene_names: Sequence[str]) -> Optional[np.ndarray]:
    path = path or os.environ.get("SPATIALCPAV9_FM_GENE_EMBEDDING")
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
    lut = {g: i for i, g in enumerate(vocab)}
    W = np.zeros((len(genes), emb.shape[1]), dtype=np.float32)
    for j, g in enumerate(genes):
        i = lut.get(g)
        if i is not None:
            W[j] = emb[i]
    return W if np.any(W) else None
