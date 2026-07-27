"""Shared neural components for SpatialCPA-v15 (PyTorch).

Two generative backbones live here:

* :class:`QueryUNet` — the Phase 2.3 **query-based interpolator**.  It takes the
  four bracketing structure-field slabs as input channels and is conditioned on
  both the normalized position in the gap ``t`` **and** the absolute gap width
  ``dz``, because the same ``t`` in a wide gap is a harder query than in a narrow
  one and the model must widen its distribution accordingly.  It is deliberately
  **flow-free**: no optical flow, no warping field.  Flow assumes conserved,
  translating content and cannot represent branching, merging or termination —
  which are exactly the events worth generating.  A self-attention block at the
  bottleneck lets the model find soft correspondence instead.

* :class:`ContextDenoiser` — the Phase 3.2 **conditional latent diffusion**
  denoiser: a transformer over the whole gap block (self-attention across cells
  *and across z*, never independent per-slice) with cross-attention onto the
  encoded latents of nearby cells in the real bracketing slices.

Plus the NB/ZINB expression VAE of Phase 3.1 (:class:`ExpressionVAE`), whose
decoder factorizes into a library-free gene *profile* and an explicit library
scalar, and the cosine-schedule diffusion utilities used by both.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── embeddings ────────────────────────────────────────────────────────────────

def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    """Sinusoidal features of a scalar (timestep, t, dz, position, ...)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    a = x.reshape(-1, 1).float() * freqs.reshape(1, -1)
    out = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
    if dim % 2:
        out = F.pad(out, (0, 1))
    return out


class FourierPosition(nn.Module):
    """Fixed random Fourier features of normalized position (Phase 3.2)."""

    def __init__(self, in_dim: int, n_freq: int = 8, scale: float = 4.0, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        B = torch.randn(in_dim, n_freq, generator=g) * scale
        self.register_buffer("B", B)

    @property
    def out_dim(self) -> int:
        return int(self.B.shape[1] * 2 + self.B.shape[0])

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (p @ self.B)
        return torch.cat([p, torch.cos(proj), torch.sin(proj)], dim=-1)


# ── diffusion schedule ────────────────────────────────────────────────────────

class CosineSchedule:
    """Cosine-schedule DDPM with a deterministic (eta=0) DDIM sampler.

    Determinism matters here beyond the usual reasons: Phase 2.4 requires that
    ``t -> I_t`` be a *continuous function* so that independently issued queries
    inside one gap compose into a coherent volume instead of flickering.  With a
    deterministic ODE sampler and the *same* initial noise reused for every query
    in the gap, nearby ``t`` produce nearby outputs by construction.
    """

    def __init__(self, n_steps: int, device="cpu"):
        self.n_steps = n_steps
        s = 0.008
        ts = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64) / n_steps
        f = torch.cos((ts + s) / (1 + s) * math.pi / 2) ** 2
        ab = (f / f[0]).clamp(1e-6, 1.0)
        self.alphas_cumprod = ab[1:].float().to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor) -> torch.Tensor:
        ab = self.alphas_cumprod[t].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def ddim(self, model_eps, shape, device, n_steps: int,
             noise: Optional[torch.Tensor] = None,
             clip_x0: float = 4.0) -> torch.Tensor:
        """Deterministic DDIM (eta = 0).  ``noise`` fixes the trajectory.

        ``clip_x0`` bounds the predicted clean sample at every step.  At the
        noisiest steps ``alpha_bar`` is ~1e-6, so the ``x0`` estimate divides by
        ~1e-3 and any error in the predicted noise is amplified a thousandfold;
        without the bound a single early step can throw the trajectory far
        outside the data manifold and the frozen decoder then turns it into
        nonsense.  Both diffusions here operate on standardized quantities, so
        +-4 standard deviations is a generous bound.
        """
        x = torch.randn(shape, device=device) if noise is None else noise.to(device)
        steps = torch.linspace(self.n_steps - 1, 0, n_steps).long().tolist()
        for i, t in enumerate(steps):
            tt = torch.full((shape[0],), t, device=device, dtype=torch.long)
            eps = model_eps(x, tt)
            ab = self.alphas_cumprod[t]
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
            if clip_x0 is not None:
                x0 = x0.clamp(-clip_x0, clip_x0)
                eps = (x - ab.sqrt() * x0) / (1 - ab).sqrt().clamp_min(1e-6)
            if i + 1 < len(steps):
                ab_prev = self.alphas_cumprod[steps[i + 1]]
                x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
            else:
                x = x0
        return x


# ── Phase 2.3 — the query-based structure interpolator ───────────────────────

class FiLM(nn.Module):
    def __init__(self, cond_dim: int, ch: int):
        super().__init__()
        self.to_scale_shift = nn.Linear(cond_dim, 2 * ch)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        s, b = self.to_scale_shift(c).chunk(2, dim=-1)
        return h * (1 + s[:, :, None, None]) + b[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.n2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = FiLM(cond_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, c):
        h = F.silu(self.n1(self.c1(x)))
        h = self.film(h, c)
        h = F.silu(self.n2(self.c2(h)))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    """Bottleneck attention — the flow-free way to find soft correspondence."""

    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        self.n = nn.GroupNorm(min(8, ch), ch)
        self.qkv = nn.Conv2d(ch, 3 * ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.heads = heads

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.heads, C // self.heads,
                                              H * W).unbind(1)
        a = torch.softmax(q.transpose(-1, -2) @ k / math.sqrt(C // self.heads), dim=-1)
        o = (v @ a.transpose(-1, -2)).reshape(B, C, H, W)
        return x + self.proj(o)


class QueryUNet(nn.Module):
    """Denoiser for the structure field: ``(I_{a-1}, I_a, I_b, I_{b+1}, t, dz) -> I_t``.

    The network predicts the noise on the *residual* of the target field from the
    linear bracket interpolation.  That parameterization makes the linear
    interpolator the model's zero-effort mean, which is precisely the baseline the
    Phase 2.5 stop condition demands the model beat — so the generative capacity
    is spent entirely on the non-linear part of the structure (appearance,
    disappearance, peaking mid-gap) rather than on relearning the ramp.
    """

    def __init__(self, n_ch: int, hidden: int = 64, cond_dim: int = 96):
        super().__init__()
        self.n_ch = n_ch
        self.cond = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU(),
                                  nn.Linear(cond_dim, cond_dim))
        self.cond_dim = cond_dim
        in_ch = n_ch * 5                       # noisy residual + 4 bracket slabs
        self.inp = nn.Conv2d(in_ch, hidden, 3, padding=1)
        self.d1 = ResBlock(hidden, hidden, cond_dim)
        self.d2 = ResBlock(hidden, hidden * 2, cond_dim)
        self.mid = ResBlock(hidden * 2, hidden * 2, cond_dim)
        self.attn = SpatialSelfAttention(hidden * 2)
        self.u2 = ResBlock(hidden * 4, hidden, cond_dim)
        self.u1 = ResBlock(hidden * 2, hidden, cond_dim)
        self.out = nn.Conv2d(hidden, n_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _cond(self, timestep, t, dz):
        d = self.cond_dim // 3
        c = torch.cat([sinusoidal(timestep, d), sinusoidal(t * 10.0, d),
                       sinusoidal(dz, self.cond_dim - 2 * d)], dim=-1)
        return self.cond(c)

    def forward(self, x, brackets, timestep, t, dz):
        c = self._cond(timestep, t, dz)
        h0 = self.inp(torch.cat([x, brackets], dim=1))
        h1 = self.d1(h0, c)
        h2 = self.d2(F.avg_pool2d(h1, 2), c)
        m = self.attn(self.mid(h2, c))
        u2 = self.u2(torch.cat([m, h2], dim=1), c)
        u2 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.u1(torch.cat([u2, h1], dim=1), c)
        return self.out(u1)


# ── Phase 3.1 — the expression VAE (a likelihood, not a generator) ───────────

def _mlp(dims, out_act=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_act:
            layers += [nn.LayerNorm(dims[i + 1]), nn.SiLU()]
    return nn.Sequential(*layers)


class ExpressionVAE(nn.Module):
    """NB / ZINB / Gaussian VAE over single-cell profiles.

    For **count** data (NB / ZINB) the decoder emits a **library-free** profile
    (a softmax over genes): total counts are technical depth, not biology, so the
    library size enters only as an explicit scalar — both as a decoder input and
    as the multiplier that turns the profile into a rate.

    For **continuous** data (Gaussian — protein intensities from IMC, and any
    panel that does not ship counts) the same profile-times-scalar decoder is
    kept: a per-cell shared scale plus a composition is a good description of
    intensity too, and it is a useful structural prior. What changes is the
    *scale the error is measured on*. Squared error on raw intensity is dominated
    entirely by the few brightest channels — with IMC values in the hundreds to
    thousands, dim markers contribute essentially nothing to the gradient — so
    the Gaussian branch scores the residual on the **log1p scale**, which is
    where the data is normalized and where the benchmark's metrics are computed.

    The KL weight is deliberately tiny either way: this stage is a likelihood and
    a coordinate system, and over-regularizing it would throw away exactly the
    resolution the diffusion prior is there to model.
    """

    def __init__(self, n_genes: int, latent_dim: int = 24, hidden: int = 128,
                 n_layers: int = 2, likelihood: str = "nb",
                 per_gene_variance: bool = False):
        super().__init__()
        self.likelihood = likelihood
        self.per_gene_variance = per_gene_variance
        self.n_genes = n_genes
        enc_dims = [n_genes + 1] + [hidden] * n_layers
        self.enc = _mlp(enc_dims, out_act=True)
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_lv = nn.Linear(hidden, latent_dim)
        dec_dims = [latent_dim + 1] + [hidden] * n_layers
        self.dec = _mlp(dec_dims, out_act=True)
        self.to_rate = nn.Linear(hidden, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        # Gaussian: per-gene observation noise, so channels are weighted by how
        # informative they are rather than by how bright they happen to be.
        self.log_sigma = nn.Parameter(torch.zeros(n_genes))
        if likelihood == "zinb":
            self.to_drop = nn.Linear(hidden, n_genes)
        self.latent_dim = latent_dim

    @property
    def is_continuous(self) -> bool:
        return self.likelihood == "gaussian"

    def encode(self, x_log: torch.Tensor, log_lib: torch.Tensor):
        h = self.enc(torch.cat([x_log, log_lib[:, None]], dim=-1))
        return self.to_mu(h), self.to_lv(h).clamp(-8, 4)

    def decode(self, z: torch.Tensor, log_lib: torch.Tensor):
        """Return ``(mu, profile, dropout_logits)``.

        ``mu`` is on the scale the likelihood is evaluated on: counts for NB /
        ZINB, log1p-intensity for Gaussian.  Use :meth:`to_data_scale` to get
        back to the caller's original units.
        """
        h = self.dec(torch.cat([z, log_lib[:, None]], dim=-1))
        rho = torch.softmax(self.to_rate(h), dim=-1)         # library-free profile
        mu = rho * log_lib.exp()[:, None]
        drop = self.to_drop(h) if self.likelihood == "zinb" else None
        return mu, rho, drop

    def to_data_scale(self, mu: torch.Tensor) -> torch.Tensor:
        """Decoder output -> the caller's original units (already there)."""
        return mu

    def nll(self, x_target, mu, drop=None):
        """Negative log-likelihood of ``x_target`` under the decoder output.

        ``x_target`` is in data units for every likelihood; the Gaussian branch
        compares on the log1p scale so bright channels cannot dominate.
        """
        if self.is_continuous:
            r = torch.log1p(x_target.clamp_min(0)) - torch.log1p(mu.clamp_min(0))
            if not self.per_gene_variance:
                # Homoscedastic on purpose: a learned per-gene variance shrinks
                # noisy channels toward their mean, which attenuates exactly the
                # gene-gene correlations the structural metrics measure.  Every
                # channel is weighted equally, as the evaluation weights them.
                return (r ** 2).sum(-1)
            log_sigma = self.log_sigma.clamp(-6, 6)
            return (0.5 * (r ** 2) * torch.exp(-2 * log_sigma) + log_sigma).sum(-1)
        x_counts = x_target
        theta = self.log_theta.exp().clamp(1e-4, 1e6)
        eps = 1e-8
        lt = torch.log(theta + eps)
        lmt = torch.log(theta + mu + eps)
        nb = (torch.lgamma(x_counts + theta) - torch.lgamma(theta)
              - torch.lgamma(x_counts + 1.0)
              + theta * (lt - lmt) + x_counts * (torch.log(mu + eps) - lmt))
        if self.likelihood == "zinb" and drop is not None:
            log_nb_zero = theta * (lt - lmt)
            zero = torch.logaddexp(drop, log_nb_zero) - F.softplus(drop)
            nonzero = nb - F.softplus(drop)
            nb = torch.where(x_counts < 1e-8, zero, nonzero)
        return -nb.sum(-1)

    def forward(self, x_log, x_counts, log_lib, kl_weight: float):
        mu_z, lv = self.encode(x_log, log_lib)
        z = mu_z + torch.randn_like(mu_z) * (0.5 * lv).exp()
        mu, rho, drop = self.decode(z, log_lib)
        rec = self.nll(x_counts, mu, drop).mean()
        kl = (-0.5 * (1 + lv - mu_z ** 2 - lv.exp()).sum(-1)).mean()
        return rec + kl_weight * kl, rec.detach(), kl.detach(), mu_z


# ── Phase 3.2 — the conditional latent-diffusion denoiser ────────────────────

class CrossAttentionBlock(nn.Module):
    """Self-attention across the target block + cross-attention onto context."""

    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d))

    def forward(self, h, ctx):
        x = self.n1(h)
        h = h + self.sa(x, x, x, need_weights=False)[0]
        if ctx is not None:
            q = self.n2(h).unsqueeze(2)                 # (B, N, 1, d)
            B, N, _, d = q.shape
            q = q.reshape(B * N, 1, d)
            k = ctx.reshape(B * N, ctx.shape[2], d)
            h = h + self.ca(q, k, k, need_weights=False)[0].reshape(B, N, d)
        return h + self.ff(self.n3(h))


class ContextDenoiser(nn.Module):
    """Denoises the whole gap block jointly (Phase 3.2).

    Conditioning, in the proposal's order: the **continuous hierarchical type
    embedding** (never a bare discrete label — that is the bottleneck that makes
    a generator emit type-mean + noise), the normalized position, the
    neighbourhood context cross-attended from the *real* bracketing slices, and
    the ``dz`` to the nearest real slice.
    """

    def __init__(self, latent_dim: int, type_dim: int, hidden: int = 96,
                 heads: int = 4, n_blocks: int = 3, seed: int = 0):
        super().__init__()
        self.pos = FourierPosition(3, n_freq=8, seed=seed)
        cond_in = type_dim + self.pos.out_dim + 3       # + dz, log-lib, t-fraction
        self.cond = _mlp([cond_in, hidden, hidden], out_act=True)
        self.time = _mlp([hidden, hidden, hidden], out_act=True)
        self.inp = nn.Linear(latent_dim, hidden)
        self.ctx_in = _mlp([latent_dim + self.pos.out_dim + type_dim + 1,
                            hidden, hidden], out_act=True)
        self.blocks = nn.ModuleList([CrossAttentionBlock(hidden, heads)
                                     for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, latent_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.hidden = hidden

    def forward(self, x, timestep, type_emb, pos, dz, log_lib, tfrac,
                ctx_lat, ctx_pos, ctx_type, ctx_dz):
        """``x``: (B, N, L) noisy latents; ``ctx_*``: (B, N, K, ...) context."""
        p = self.pos(pos)
        c = self.cond(torch.cat([type_emb, p, dz[..., None], log_lib[..., None],
                                 tfrac[..., None]], dim=-1))
        t = self.time(sinusoidal(timestep.reshape(-1), self.hidden)
                      ).reshape(x.shape[0], 1, self.hidden)
        h = self.inp(x) + c + t
        ctx = None
        if ctx_lat is not None and ctx_lat.shape[2] > 0:
            cp = self.pos(ctx_pos.reshape(-1, 3)).reshape(*ctx_pos.shape[:3], -1)
            ctx = self.ctx_in(torch.cat([ctx_lat, cp, ctx_type,
                                         ctx_dz[..., None]], dim=-1))
        for blk in self.blocks:
            h = blk(h, ctx)
        return self.out(h)
