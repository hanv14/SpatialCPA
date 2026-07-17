"""
Neural modules for SpatialCPA-v14 / H3D-FLA (PyTorch).

* :class:`FourierEmbed` — Fourier positional encoding of continuous ``(x, y, z)`` (and a
  separate sinusoidal embedding of the flow time ``t``).
* :class:`JointEncoder` — Stage 2 fusion of the expression latent ``e`` and the
  morphological (pseudo-image) channels ``m`` into a unified joint latent ``h``, with
  decoders back to ``e``, to ``m``, to cell type, and a hypoxia scalar head. The decoders
  make Stage 5 (latent -> expression/type/image) simple and support the closed-loop
  consistency loss.
* :class:`ContextAttention` — Stage 3.1 3D positional-attention context: a spatial query
  (Fourier ``(x, y, z)``) cross-attends over local flanking cells' joint latents *and*
  per-slice global summary tokens, yielding a context vector ``C(z)`` capturing local and
  long-range 3D relationships.
* :class:`VectorField` — Stage 3.2 conditional flow-matching velocity network:
  ``v_t(h_t | t, C(z), z)`` (MLP backbone), plus a displacement head that decodes a
  continuous in-plane deformation for the generated sheet.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, act=nn.GELU, last_act=False, dropout=0.0):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FourierEmbed(nn.Module):
    """Fixed Fourier features of a continuous vector (dim -> dim*2*bands)."""

    def __init__(self, in_dim, bands=6):
        super().__init__()
        freqs = 2.0 ** torch.arange(bands).float() * np.pi
        self.register_buffer("freqs", freqs)
        self.out_dim = in_dim * 2 * bands

    def forward(self, x):
        # x (..., in_dim) -> (..., in_dim*2*bands)
        proj = x[..., None] * self.freqs                      # (..., in_dim, bands)
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1)     # (..., in_dim, 2*bands)
        return emb.reshape(*x.shape[:-1], -1)


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of the scalar flow time t in [0, 1]."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer("freqs", torch.exp(
            torch.linspace(0.0, np.log(1000.0), half)))

    def forward(self, t):
        t = t.reshape(-1, 1)
        args = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, emb[:, :1] * 0], dim=1)
        return emb


class JointEncoder(nn.Module):
    """Stage 2 joint molecular-morphological encoder + decoders."""

    def __init__(self, d_e, n_morph, joint_dim, n_types, hidden=128, dropout=0.05):
        super().__init__()
        self.enc = mlp([d_e + n_morph, hidden, joint_dim], last_act=False, dropout=dropout)
        self.dec_e = mlp([joint_dim, hidden, d_e])
        self.dec_m = mlp([joint_dim, hidden, n_morph])
        self.type_head = nn.Linear(joint_dim, max(n_types, 1))
        self.hypoxia_head = nn.Linear(joint_dim, 1)             # TME gradient scalar

    def encode(self, e, m):
        return self.enc(torch.cat([e, m], dim=-1))

    def decode_e(self, h):
        return self.dec_e(h)

    def decode_m(self, h):
        return self.dec_m(h)


class ContextAttention(nn.Module):
    """Stage 3.1 — 3D positional-attention context module."""

    def __init__(self, joint_dim, d_model, n_heads, fourier_bands=6, dropout=0.05):
        super().__init__()
        self.pos = FourierEmbed(3, fourier_bands)              # (x, y, z)
        pd = self.pos.out_dim
        self.query_enc = mlp([pd, d_model, d_model], last_act=True)
        # token = [joint latent h ; Fourier(pos)] projected to d_model
        self.token_enc = mlp([joint_dim + pd, d_model, d_model], last_act=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = mlp([d_model, 2 * d_model, d_model])
        self.out_dim = d_model

    def encode_tokens(self, h, pos):
        return self.token_enc(torch.cat([h, self.pos(pos)], dim=-1))

    def forward(self, query_pos, tokens, key_padding_mask=None):
        """query_pos (B, 3); tokens (B, T, d_model). Returns context (B, d_model)."""
        q = self.query_enc(self.pos(query_pos))[:, None, :]    # (B, 1, d)
        ctx, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        c = self.norm(ctx[:, 0] + q[:, 0])
        c = self.norm(c + self.ff(c))
        return c


class VectorField(nn.Module):
    """Stage 3.2 — conditional flow-matching velocity field v_t(h_t | t, C, z)."""

    def __init__(self, joint_dim, ctx_dim, fourier_bands=6, hidden=192, n_layers=4,
                 time_dim=32):
        super().__init__()
        self.time = TimeEmbed(time_dim)
        self.zpos = FourierEmbed(1, fourier_bands)
        in_dim = joint_dim + ctx_dim + time_dim + self.zpos.out_dim
        sizes = [in_dim] + [hidden] * (n_layers - 1)
        self.backbone = mlp(sizes, last_act=True)
        self.head = nn.Linear(hidden, joint_dim)
        # continuous in-plane displacement decoder (learned deformation field)
        self.disp_head = mlp([hidden, hidden // 2, 2])

    def features(self, h_t, t, ctx, z):
        te = self.time(t)
        ze = self.zpos(z.reshape(-1, 1))
        x = torch.cat([h_t, ctx, te, ze], dim=-1)
        return self.backbone(x)

    def forward(self, h_t, t, ctx, z):
        return self.head(self.features(h_t, t, ctx, z))

    def displacement(self, h1, t, ctx, z):
        """Decode an in-plane displacement from the (clean) latent + context."""
        return self.disp_head(self.features(h1, t, ctx, z))
