"""
Neural network components for SpatialCPA-v9 (PyTorch).

Three modules:

* :class:`ExpressionAE` — a small MLP autoencoder that maps normalized expression
  to a low-dimensional latent (and back). The flow operates in this latent space,
  which makes generative modelling of high-dimensional expression tractable and
  denoises the panel. The first encoder layer can be warm-started from a pretrained
  gene embedding (foundation-model prior).
* :class:`ContextEncoder` — a permutation-invariant (DeepSets-style) summary of the
  two flanking slices' cell features plus the axial gaps, giving the velocity field
  awareness of *which* tissue it is interpolating and how far apart the slices are.
* :class:`FlowNet` — the conditional velocity field ``v_theta(x, s, context)`` of a
  rectified-flow / flow-matching model, over the joint (position, expression-latent)
  cell feature ``x`` and flow time ``s``.

Imported only when PyTorch is available; the generator guards the import and falls
back to the optimal-transport morph otherwise.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes, act=nn.SiLU, dropout=0.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ExpressionAE(nn.Module):
    """MLP autoencoder: expression (G) <-> latent (d)."""

    def __init__(self, n_genes, latent_dim, hidden, dropout=0.0,
                 gene_embedding: np.ndarray | None = None):
        super().__init__()
        self.encoder = _mlp([n_genes, hidden, hidden, latent_dim], dropout=dropout)
        self.decoder = _mlp([latent_dim, hidden, hidden, n_genes], dropout=dropout)
        if gene_embedding is not None:
            # Warm-start the first encoder layer with the pretrained gene embedding
            # (project genes -> embedding dim -> hidden via the loaded matrix).
            W = torch.tensor(np.asarray(gene_embedding, dtype=np.float32))
            first = self.encoder[0]
            with torch.no_grad():
                d = min(first.weight.shape[0], W.shape[1])
                first.weight[:d, :W.shape[0]] = W[:, :d].T

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class ContextEncoder(nn.Module):
    """Permutation-invariant summary of the two flanking slices + axial gaps."""

    def __init__(self, feat_dim, context_dim, hidden=128):
        super().__init__()
        self.phi = _mlp([feat_dim, hidden, hidden])
        # pooled(lower) + pooled(upper) + [gap_lo, gap_hi, t] -> context
        self.rho = _mlp([2 * hidden + 3, hidden, context_dim])

    def forward(self, lo_feat, up_feat, gaps):
        # lo_feat, up_feat: (N_lo, F), (N_up, F); gaps: (3,) tensor [gap_lo, gap_hi, t]
        pl = self.phi(lo_feat).mean(dim=0)
        pu = self.phi(up_feat).mean(dim=0)
        return self.rho(torch.cat([pl, pu, gaps], dim=-1))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, s):  # s: (B,)
        half = self.dim // 2
        freqs = torch.exp(
            -np.log(10000.0) * torch.arange(half, device=s.device).float() / max(half - 1, 1))
        ang = s[:, None] * freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FlowNet(nn.Module):
    """Conditional velocity field v_theta(x, s, context) for the flow bridge."""

    def __init__(self, feat_dim, context_dim, hidden, layers, time_embed_dim,
                 dropout=0.0):
        super().__init__()
        self.time_embed = TimeEmbedding(time_embed_dim)
        in_dim = feat_dim + time_embed_dim + context_dim
        sizes = [in_dim] + [hidden] * layers + [feat_dim]
        self.net = _mlp(sizes, dropout=dropout)

    def forward(self, x, s, context):
        # x: (B, F); s: (B,); context: (context_dim,) broadcast over batch
        te = self.time_embed(s)
        ctx = context[None, :].expand(x.shape[0], -1)
        return self.net(torch.cat([x, te, ctx], dim=-1))
