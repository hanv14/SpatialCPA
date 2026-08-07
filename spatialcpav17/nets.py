"""
Neural modules for SpatialCPA-v17 (PyTorch) — VAE + NB decoder + flow field.

* :class:`FourierEmbed`, :class:`TimeEmbed` — positional / time encodings.
* :class:`JointVAE` — a variational joint molecular-morphological autoencoder. The encoder
  maps ``[log-normalized expression ⊕ morphology] -> (μ, logσ²)`` and reparameterizes to a
  KL-regularized latent ``h``. The decoder is **negative-binomial** (scVI-style): it maps
  ``h`` to per-gene proportions ``ρ`` (softmax) and, with a per-gene dispersion and the
  cell's library size ``ℓ``, gives the NB mean ``μ = ℓ · ρ``. Auxiliary heads reconstruct
  the morphology channels and predict cell type. A Gaussian likelihood is provided for
  non-count inputs.
* :class:`ContextAttention`, :class:`VectorField` — 3D positional-attention context and the
  conditional flow-matching velocity field (identical to v14).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        proj = x[..., None] * self.freqs
        emb = torch.cat([proj.sin(), proj.cos()], dim=-1)
        return emb.reshape(*x.shape[:-1], -1)


class TimeEmbed(nn.Module):
    """Sinusoidal embedding of the scalar flow time t in [0, 1]."""

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        half = dim // 2
        self.register_buffer("freqs", torch.exp(torch.linspace(0.0, np.log(1000.0), half)))

    def forward(self, t):
        t = t.reshape(-1, 1)
        args = t * self.freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, emb[:, :1] * 0], dim=1)
        return emb


# --------------------------------------------------------------------------- #
# Variational joint autoencoder with a negative-binomial decoder               #
# --------------------------------------------------------------------------- #
def nb_nll(x, mu, theta, eps=1e-8):
    """Negative-binomial negative log-likelihood, summed over genes, mean over cells.

    x     : observed counts (B, G)
    mu    : NB mean (B, G)
    theta : NB inverse-dispersion (G,) or (B, G), > 0
    """
    theta = theta + eps
    mu = mu + eps
    log_theta_mu = torch.log(theta + mu)
    ll = (theta * (torch.log(theta) - log_theta_mu)
          + x * (torch.log(mu) - log_theta_mu)
          + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1.0))
    return -ll.sum(dim=-1).mean()


class JointVAE(nn.Module):
    """Variational encoder + NB (or Gaussian) decoder over a joint expr+morph latent."""

    def __init__(self, n_genes, n_morph, n_types, cfg, likelihood="nb"):
        super().__init__()
        self.n_genes = n_genes
        self.likelihood = likelihood
        d = cfg.latent_dim
        h = cfg.hidden
        enc_in = n_genes + n_morph
        self.enc_body = mlp([enc_in] + [h] * cfg.n_layers, last_act=True, dropout=cfg.dropout)
        self.enc_mu = nn.Linear(h, d)
        self.enc_logvar = nn.Linear(h, d)
        self.dec_body = mlp([d] + [h] * cfg.n_layers, last_act=True, dropout=cfg.dropout)
        self.dec_expr = nn.Linear(h, n_genes)            # gene logits -> softmax proportions (NB)
        self.px_r = nn.Parameter(torch.randn(n_genes) * 0.1)   # NB log dispersion (gene-wise)
        self.px_logsigma = nn.Parameter(torch.zeros(n_genes)) # Gaussian log std (gene-wise)
        self.dec_morph = nn.Linear(h, n_morph)
        self.type_head = nn.Linear(d, max(n_types, 1))
        self.d = d

    def encode(self, x_norm, m):
        hbody = self.enc_body(torch.cat([x_norm, m], dim=-1))
        return self.enc_mu(hbody), self.enc_logvar(hbody)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, h, library):
        """Return decoder outputs. NB: (mu_nb, theta, rho). Gaussian: (mean, sigma, None).

        The second slot carries the per-gene scale used for posterior-predictive sampling:
        NB inverse-dispersion ``theta`` or Gaussian standard deviation ``sigma`` (both gene-wise).
        """
        body = self.dec_body(h)
        if self.likelihood == "nb":
            rho = torch.softmax(self.dec_expr(body), dim=-1)   # gene proportions (sum=1)
            mu_nb = library.reshape(-1, 1) * rho
            theta = torch.exp(self.px_r)
            return mu_nb, theta, rho
        mean = self.dec_expr(body)                             # Gaussian mean (standardized space)
        sigma = torch.exp(self.px_logsigma)                   # Gaussian std (standardized space)
        return mean, sigma, None

    def decode_morph(self, h):
        return self.dec_morph(self.dec_body(h))


class ContextAttention(nn.Module):
    """Stage 3.1 — 3D positional-attention context module (identical to v14)."""

    def __init__(self, joint_dim, d_model, n_heads, fourier_bands=6, dropout=0.05):
        super().__init__()
        self.pos = FourierEmbed(3, fourier_bands)
        pd = self.pos.out_dim
        self.query_enc = mlp([pd, d_model, d_model], last_act=True)
        self.token_enc = mlp([joint_dim + pd, d_model, d_model], last_act=True)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = mlp([d_model, 2 * d_model, d_model])
        self.out_dim = d_model

    def encode_tokens(self, h, pos):
        return self.token_enc(torch.cat([h, self.pos(pos)], dim=-1))

    def forward(self, query_pos, tokens, key_padding_mask=None):
        q = self.query_enc(self.pos(query_pos))[:, None, :]
        ctx, _ = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask)
        c = self.norm(ctx[:, 0] + q[:, 0])
        c = self.norm(c + self.ff(c))
        return c


class VectorField(nn.Module):
    """Stage 3.2 — conditional flow-matching velocity field (identical to v14)."""

    def __init__(self, joint_dim, ctx_dim, fourier_bands=6, hidden=192, n_layers=4, time_dim=32):
        super().__init__()
        self.time = TimeEmbed(time_dim)
        self.zpos = FourierEmbed(1, fourier_bands)
        in_dim = joint_dim + ctx_dim + time_dim + self.zpos.out_dim
        sizes = [in_dim] + [hidden] * (n_layers - 1)
        self.backbone = mlp(sizes, last_act=True)
        self.head = nn.Linear(hidden, joint_dim)

    def forward(self, h_t, t, ctx, z):
        te = self.time(t)
        ze = self.zpos(z.reshape(-1, 1))
        x = torch.cat([h_t, ctx, te, ze], dim=-1)
        return self.head(self.backbone(x))
