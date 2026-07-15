"""
Cell-sentence tokenizer for SpatialCPA-v13.

Converts a cell's gene-expression vector into a **cell-sentence**: a rank-ordered
sequence of gene tokens (the genes expressed most, in descending order), the same
representation used by single-cell language models such as Cell2Sentence, Geneformer
and scGPT. The vocabulary is::

    [PAD]=0  [MASK]=1  [CLS]=2   then one token per gene: gene g -> id (3 + g)

A sentence is ``[CLS] g_{π(1)} g_{π(2)} ... g_{π(K)}`` where ``π`` orders genes by
descending (normalized) expression and ``K = top_genes``. De-tokenization turns a
predicted set/ordering of gene tokens back into a rank profile.
"""

from __future__ import annotations

import numpy as np

PAD, MASK, CLS = 0, 1, 2
N_SPECIAL = 3


class CellSentenceTokenizer:
    def __init__(self, n_genes: int, top_genes: int = 32, min_expr: float = 0.0):
        self.n_genes = int(n_genes)
        self.top_genes = int(top_genes)
        self.min_expr = float(min_expr)
        self.vocab_size = N_SPECIAL + self.n_genes
        self.max_len = 1 + self.top_genes           # [CLS] + genes

    def gene_to_token(self, g):
        return N_SPECIAL + np.asarray(g, dtype=np.int64)

    def token_to_gene(self, tok):
        return np.asarray(tok, dtype=np.int64) - N_SPECIAL

    def encode(self, expr: np.ndarray):
        """Encode an (n, G) expression matrix into (n, max_len) token ids + pad mask.

        Returns ``(tokens, pad_mask)`` where ``pad_mask`` is True at padded positions.
        """
        X = np.asarray(expr, dtype=np.float32)
        n, G = X.shape
        K = min(self.top_genes, G)
        toks = np.full((n, self.max_len), PAD, dtype=np.int64)
        pad = np.ones((n, self.max_len), dtype=bool)
        toks[:, 0] = CLS
        pad[:, 0] = False
        # top-K genes by expression (descending), keeping only those above min_expr
        order = np.argsort(-X, axis=1)[:, :K]
        for i in range(n):
            gi = order[i]
            gi = gi[X[i, gi] > self.min_expr]
            L = len(gi)
            if L:
                toks[i, 1:1 + L] = self.gene_to_token(gi)
                pad[i, 1:1 + L] = False
        return toks, pad

    def is_gene_token(self, tok):
        return np.asarray(tok) >= N_SPECIAL
