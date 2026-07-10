"""
Neural modules for SpatialCPA-v11 (PyTorch): Fourier encodings, the neighbouring-
slice context encoder, and the two implicit coordinate-network fields.

* :class:`FourierFeatures` — sinusoidal positional encoding for the continuous query
  coordinates (separately configurable bands for ``z`` and ``(x, y)``), which is what
  lets the fields represent smooth variation between and beyond the real slices.
* :class:`ContextEncoder` — a permutation-invariant (DeepSets) encoder of the aligned
  neighbouring slices' spots, producing a global context vector; plus a low-res
  *rasterization* of each flanking slice (occupancy + type) that the fields sample
  bilinearly at the query ``(x, y)`` for spatial conditioning.
* :class:`LayoutField` — Stage 1: ``(x, y, z, context) -> occupancy logit, type logits``
  and a layout code handed to Stage 2.
* :class:`ExpressionField` — Stage 2: ``(x, y, z, layout code, context) -> expression``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU, dropout=0.0, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FourierFeatures(nn.Module):
    """Axis-wise sinusoidal features: x -> [x, sin(2^k π f x), cos(...)]."""

    def __init__(self, n_bands, max_freq):
        super().__init__()
        freqs = torch.linspace(1.0, max_freq, n_bands) * np.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_mult(self):
        return 1 + 2 * len(self.freqs)

    def forward(self, x):  # x: (..., d)
        proj = x[..., None] * self.freqs                      # (..., d, B)
        enc = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., d, 2B)
        enc = enc.reshape(*x.shape[:-1], -1)
        return torch.cat([x, enc], dim=-1)


class ContextEncoder(nn.Module):
    """Permutation-invariant encoding of the neighbouring slices' spots."""

    def __init__(self, spot_feat_dim, hidden, context_dim):
        super().__init__()
        self.phi = mlp([spot_feat_dim, hidden, hidden])
        # mean+max pool per slice, concatenated with the slice's normalized z
        self.rho = mlp([2 * hidden + 1, hidden, context_dim])

    def forward(self, slice_feats):
        """slice_feats: list of (n_i, F) spot features (one per context slice) with
        the slice's normalized z appended as the last feature column."""
        parts = []
        for sf in slice_feats:
            z = sf[:, -1:].mean(0)                # slice z (normalized)
            h = self.phi(sf)
            parts.append(self.rho(torch.cat([h.mean(0), h.max(0).values, z], dim=-1)))
        return torch.stack(parts, 0).mean(0)      # (context_dim,)


def sample_raster(raster, xy):
    """Bilinearly sample a (C, G, G) raster at points xy in [-1, 1]^2."""
    grid = xy.view(1, 1, -1, 2)                   # (1,1,N,2), x,y in [-1,1]
    out = F.grid_sample(raster[None], grid, mode="bilinear",
                        align_corners=True, padding_mode="border")
    return out[0, :, 0].t()                       # (N, C)


class LayoutField(nn.Module):
    """Stage 1 — occupancy + type/region field, with a layout code for Stage 2."""

    def __init__(self, in_dim, cfg, n_types):
        super().__init__()
        self.trunk = mlp([in_dim] + [cfg.hidden] * cfg.layers + [cfg.hidden],
                         dropout=cfg.dropout, last_act=True)
        self.code = nn.Linear(cfg.hidden, cfg.layout_feat_dim)
        self.occ = nn.Linear(cfg.hidden, 1)
        self.typ = nn.Linear(cfg.hidden, max(n_types, 1))

    def forward(self, feat):
        h = self.trunk(feat)
        return self.occ(h).squeeze(-1), self.typ(h), self.code(h)


class ExpressionField(nn.Module):
    """Stage 2 — expression field conditioned on the Stage-1 layout code."""

    def __init__(self, in_dim, cfg, n_genes):
        super().__init__()
        self.trunk = mlp([in_dim] + [cfg.hidden] * cfg.layers + [cfg.hidden],
                         dropout=cfg.dropout, last_act=True)
        self.head = nn.Linear(cfg.hidden, n_genes)

    def forward(self, feat):
        return self.head(self.trunk(feat))
