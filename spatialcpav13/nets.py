"""
Transformer language-model modules for SpatialCPA-v13 (PyTorch).

* :class:`CellTransformer` — a self-attention transformer over cell-sentences (token +
  learned within-sentence positional embeddings -> ``nn.TransformerEncoder``), pooled at
  the ``[CLS]`` position into a **cell embedding**. Heads: a masked gene-language-model
  head (over the gene vocabulary), a cell-type head, and an expression-decode head.
* :class:`SpatialContextAttention` — a cross-attention block: a learned **spatial query**
  (an MLP encoding of the continuous ``(x, y, z)``) attends over the flanking cells'
  embeddings, producing the retrieval-augmented **context embedding** used to generate a
  virtual cell in context.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.GELU, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
    return nn.Sequential(*layers)


class CellTransformer(nn.Module):
    """Self-attention transformer language model over cell-sentences."""

    def __init__(self, vocab_size, n_genes, n_types, cfg):
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.tok_emb = nn.Embedding(vocab_size, d, padding_idx=0)
        self.pos_emb = nn.Embedding(cfg.max_len, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        self.mlm_head = nn.Linear(d, vocab_size)
        self.type_head = nn.Linear(d, max(n_types, 1))
        self.expr_head = nn.Linear(d, n_genes)

    def forward(self, tokens, pad_mask):
        """tokens (B, L) ids; pad_mask (B, L) True at pads. Returns per-token hidden and
        the pooled [CLS] cell embedding."""
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device)[None, :].expand(B, L)
        h = self.tok_emb(tokens) + self.pos_emb(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        cls = h[:, 0]                       # [CLS] pooled cell embedding
        return h, cls

    def mlm_logits(self, hidden):
        return self.mlm_head(hidden)

    def cell_embedding(self, tokens, pad_mask):
        _, cls = self.forward(tokens, pad_mask)
        return cls


class SpatialContextAttention(nn.Module):
    """Cross-attention: a spatial query attends over flanking cells' embeddings (RAG)."""

    def __init__(self, d_model, n_heads, n_genes, n_types, dropout=0.1):
        super().__init__()
        self.spatial_enc = mlp([3, d_model, d_model], last_act=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = mlp([d_model, 2 * d_model, d_model])
        self.type_head = nn.Linear(d_model, max(n_types, 1))
        self.expr_head = nn.Linear(d_model, n_genes)

    def query(self, xyz):
        """Encode continuous (x, y, z) into a spatial query embedding (B, d)."""
        return self.spatial_enc(xyz)

    def forward(self, xyz, neigh_emb, neigh_mask=None):
        """xyz (B, 3); neigh_emb (B, N, d) flanking neighbour cell embeddings.
        Returns the context embedding (B, d)."""
        q = self.query(xyz)[:, None, :]                 # (B, 1, d)
        key_pad = None if neigh_mask is None else neigh_mask
        ctx, _ = self.attn(q, neigh_emb, neigh_emb, key_padding_mask=key_pad)
        ctx = self.norm(ctx[:, 0] + q[:, 0])
        ctx = self.norm(ctx + self.ff(ctx))
        return ctx
