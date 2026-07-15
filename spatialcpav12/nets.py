"""
Neural modules for SpatialCPA-v12 (PyTorch): Fourier encodings, the neighbouring-
slice context encoder, the Stage-1 layout field, and the Stage-2 **generative**
expression decoder.

* :class:`FourierFeatures` — sinusoidal positional encoding for the continuous query
  coordinates (separately configurable bands for ``z`` and ``(x, y)``), which is what
  lets the fields represent smooth variation between and beyond the real slices.
* :class:`ContextEncoder` — a permutation-invariant (DeepSets) encoder of the aligned
  neighbouring slices' spots, producing a global context vector; plus a low-res
  *rasterization* of each flanking slice (occupancy + type) that the fields sample
  bilinearly at the query ``(x, y)`` for spatial conditioning.
* :class:`LayoutField` — Stage 1: ``(x, y, z, context) -> occupancy logit, type logits``
  and a layout code handed to Stage 2.
* :class:`GenerativeExpressionField` — Stage 2: a conditional factor-analysis decoder.
  It outputs a per-cell mean ``mu(x, y, z, layout, context)`` and holds a *shared*
  low-rank loading matrix ``L`` (G × r) plus a per-gene idiosyncratic log-variance
  ``log_psi``. Expression is the Gaussian ``x = mu + L·s`` (``s ~ N(0, I_r)``) with
  covariance ``L Lᵀ + Ψ`` — trained to the real gene-gene covariance by a
  factor-analysis likelihood, so generated cells carry realistic covariance/variance.
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


class GenerativeExpressionField(nn.Module):
    """Stage 2 — conditional factor-analysis decoder (covariance-preserving).

    Outputs the per-cell mean ``mu`` from the coordinate/layout/context features and
    holds a shared low-rank loading matrix ``L`` (G × r) and a per-gene idiosyncratic
    log-variance ``log_psi`` (G). The implied per-cell distribution is the Gaussian
    ``N(mu, L Lᵀ + Ψ)`` with ``Ψ = diag(exp(log_psi))``; sampling ``x = mu + L·s``,
    ``s ~ N(0, I_r)`` yields generated cells whose gene-gene covariance matches the
    real data the loadings were fit to.
    """

    def __init__(self, in_dim, cfg, n_genes):
        super().__init__()
        self.trunk = mlp([in_dim] + [cfg.hidden] * cfg.layers + [cfg.hidden],
                         dropout=cfg.dropout, last_act=True)
        self.head = nn.Linear(cfg.hidden, n_genes)
        self.n_genes = n_genes
        self.n_factors = int(cfg.n_factors)
        self.min_log_psi = float(cfg.min_log_psi)
        self.max_log_psi = float(cfg.max_log_psi)
        # Shared factor loadings and idiosyncratic noise (data-covariance parameters).
        self.loadings = nn.Parameter(torch.zeros(n_genes, self.n_factors))
        self.log_psi = nn.Parameter(torch.zeros(n_genes))

    def init_covariance(self, L0, log_psi0):
        """Warm-start L and log_psi from a data-derived covariance factorization."""
        with torch.no_grad():
            self.loadings.copy_(torch.as_tensor(L0, dtype=self.loadings.dtype,
                                                device=self.loadings.device))
            self.log_psi.copy_(torch.as_tensor(log_psi0, dtype=self.log_psi.dtype,
                                               device=self.log_psi.device))

    def psi(self):
        return torch.exp(self.log_psi.clamp(self.min_log_psi, self.max_log_psi))

    def forward(self, feat):
        return self.head(self.trunk(feat))        # mu (n, G)

    def sample(self, mu, s):
        """Generate expression x = mu + L·s from a factor code s (n, r)."""
        return mu + s @ self.loadings.t()
