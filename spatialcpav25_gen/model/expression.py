"""Expression: a per-cell latent from flow matching, decoded through a gene-conditioned head.

Two failure modes this shape exists to avoid, both observed in earlier versions of this
project (T06 §1):

*Mean regression over-smooths.* Predicting ``E[expr]`` collapses variance and destroys
spatial autocorrelation, so nothing here ever emits ``mu``: :func:`sample_counts` draws from
the conditional distribution and the generation path asserts the emitted matrix's detection
rate against the training sections' (:func:`assert_detection_rate`).

*Per-gene independent sampling produces chimeras.* The competing method draws each gene
independently from among ``<= 3`` donor cells, which destroys within-cell gene-gene
covariance. Here **all genes of a cell decode from one shared latent** ``h_i``, which is the
quantitative claim ``test_shared_latent_preserves_covariance`` measures against
:mod:`spatialcpav25_gen.eval.baselines`.

The open-vocabulary property
----------------------------
The gene enters the decoder **only** through its embedding ``e_g``, never through a row of a
fixed-width output layer, so one decoder serves an arbitrary gene set — that is what makes
zero-shot genes and cross-panel transfer possible, and it is why ``Config.genes_per_step``
subsamples the panel every training step: the decoder must never see a fixed panel width.

The same argument binds the *encoder*, and T06 §2 does not say so. ``h1 = Enc(log1p(counts /
size_factor))`` reads as a fixed-width input layer, which would tie the data-side latent to
one panel and make ``forward_zero_shot`` decoding meaningless for any volume the encoder was
not built on. :class:`ExpressionEncoder` is therefore a *set* encoder over
``(expression, gene embedding)`` pairs: permutation-invariant in the genes, defined for any
subset of them, and equal to the spec's formula in everything but the width of its input.

Two generators, both explicit, and why one is numpy
---------------------------------------------------
Convention 3 wants an explicit ``seed`` or generator on every stochastic function.
:meth:`LatentFlow.cfm_loss` takes a ``torch.Generator`` because it draws inside the autograd
graph. The two functions on the **output** path — :func:`sample_counts` and
:func:`cross_mix_counts` — take a ``numpy.random.Generator`` instead, for two reasons that
both point the same way: torch has no seedable Gamma sampler (``torch._standard_gamma``
ignores generators), and ``cross_mix_counts`` has to reproduce v20's draw *bit for bit*
(T06 §4b), which is a numpy draw. The rest of the package already passes
``numpy.random.Generator`` explicitly (``potts_smooth``, ``uniform_slab_points``), so this is
the established spelling rather than a new one.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from spatialcpav25_gen.config import Config

__all__ = [
    "DecoderOutput",
    "ExpressionEncoder",
    "ExpressionError",
    "LatentFlow",
    "SizeFactorHead",
    "ZIGammaDecoder",
    "ZINBDecoder",
    "assert_detection_rate",
    "build_decoder",
    "cross_mix_counts",
    "detection_rate",
    "draw_mean_variance",
    "sample_counts",
    "sample_zigamma",
    "time_embedding",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

DecoderOutput = tuple[Tensor, Tensor, Tensor]
"""``(mu, theta, pi)``, each ``(N, G')``: the three parameters every decoder returns.

One triple for both decoders, so ``Config.decoder`` is a real gate and not two code paths.
``theta`` is the negative binomial's dispersion under :class:`ZINBDecoder` and the Gamma's
shape under :class:`ZIGammaDecoder`; in both it is the parameter that sets the
mean-variance relation at fixed ``mu``, which is what T09 calibrates.
"""


class ExpressionError(ValueError):
    """Raised when the expression path is given something it cannot decode or emit."""


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Seeded Kaiming-uniform initialisation, so construction is reproducible."""
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


def _softplus_inverse(y: float) -> float:
    """``x`` such that ``softplus(x) = y``, for ``y > 0``."""
    if y <= 0.0:
        raise ExpressionError(f"_softplus_inverse: needs y > 0, got {y!r}")
    return float(np.log(np.expm1(y))) if y < 20.0 else y


def _mlp(widths: list[int], generator: torch.Generator) -> nn.Sequential:
    """Build a GELU MLP over ``widths`` (input first, output last). ``len(widths) >= 2``."""
    layers: list[nn.Module] = []
    for index in range(len(widths) - 1):
        linear = nn.Linear(widths[index], widths[index + 1])
        _init_linear(linear, generator)
        layers.append(linear)
        if index < len(widths) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------------------
# 1. the data-side latent
# --------------------------------------------------------------------------------------


class ExpressionEncoder(nn.Module):
    """``h1``: the data-side per-cell latent, from observed expression. Returns ``(N, d_h)``.

    Args:
        cfg: supplies ``latent_dim`` (d_h), ``gene_emb_dim``, ``latent_encoder_hidden``,
             ``latent_encoder_layers``, ``seed`` and ``debug_shapes``.

    ``h1 = MLP([pool_i, mean_g x_ig, log s_i])`` with
    ``pool_i = mean_g x_ig * (P e_g)`` and ``x = log1p(counts / size_factor)``.

    The pooling is a mean over the genes *presented*, so the encoder is defined on any gene
    subset and gives the same answer under any permutation of it — the property the
    open-vocabulary decoder has by construction and the spec's fixed-width ``Enc`` would not.
    ``mean_g x`` and ``log s`` enter beside the pooled interaction because the pooling alone
    is scale-free in a way that throws away exactly the two scalars the ZINB head needs back.

    ``log1p`` normalisation is an **input-side** transform (Convention 5). It never reaches
    the decoder's target, which is raw counts.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.latent_encoder_layers < 2:
            raise ExpressionError(
                f"ExpressionEncoder: Config.latent_encoder_layers="
                f"{cfg.latent_encoder_layers} must be >= 2; one layer makes h1 a linear "
                "read-out of the pooled expression"
            )
        self.cfg = cfg
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.gene_proj = nn.Linear(int(cfg.gene_emb_dim), int(cfg.gene_emb_dim), bias=False)
        _init_linear(self.gene_proj, generator)
        widths = [int(cfg.gene_emb_dim) + 2]
        widths += [int(cfg.latent_encoder_hidden)] * (int(cfg.latent_encoder_layers) - 1)
        widths.append(int(cfg.latent_dim))
        self.mlp = _mlp(widths, generator)

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """``(N, G')``, ``(G', d_e)``, ``(N,)`` -> ``(N, d_h)``; see :meth:`forward`."""
        return cast(Tensor, super().__call__(*args, **kwargs))

    def forward(self, counts: Tensor, gene_emb: Tensor, size_factor: Tensor) -> Tensor:
        """``(N, G')`` raw counts, ``(G', d_e)`` embeddings, ``(N,)`` size factors -> ``(N, d_h)``.

        ``size_factor`` is the cell's total over the **whole** panel divided by the volume's
        median total, i.e. dimensionless and about 1 — see :class:`SizeFactorHead`.
        """
        if counts.ndim != 2 or gene_emb.ndim != 2 or size_factor.ndim != 1:
            raise ExpressionError(
                "ExpressionEncoder expects (N, G'), (G', d_e), (N,); got "
                f"{tuple(counts.shape)}, {tuple(gene_emb.shape)}, {tuple(size_factor.shape)}"
            )
        if counts.shape[1] != gene_emb.shape[0] or counts.shape[0] != size_factor.shape[0]:
            raise ExpressionError(
                "ExpressionEncoder: counts, gene_emb and size_factor disagree about N or G': "
                f"{tuple(counts.shape)}, {tuple(gene_emb.shape)}, {tuple(size_factor.shape)}"
            )
        scale = torch.clamp(size_factor, min=float(self.cfg.zinb_eps))[:, None]
        normalised = torch.log1p(counts.to(torch.float32) / scale)
        pooled = normalised @ self.gene_proj(gene_emb) / float(counts.shape[1])
        level = normalised.mean(dim=1, keepdim=True)
        out = cast(Tensor, self.mlp(torch.cat([pooled, level, torch.log(scale)], dim=1)))
        if self.cfg.debug_shapes:
            assert out.shape == (counts.shape[0], int(self.cfg.latent_dim)), out.shape
        return out


class SizeFactorHead(nn.Module):
    """Per-cell library size decoded from ``h``. Returns ``(N,)`` log total counts.

    Args:
        cfg: supplies ``latent_dim``, ``size_factor_hidden``, ``size_factor_min``, ``seed``
             and ``debug_shapes``.
        reference_total: the volume's **median** per-cell total, in counts. The output bias
             starts at its logarithm, so an untrained head already predicts the right library
             size and training only has to learn the deviation. A property of the dataset, not
             a constant, which is why it is an argument and not a ``Config`` field — the same
             shape of argument as ``IntensityHead(init_density=...)``.

    ``size_factor = exp(log_total) / reference_total`` is the dimensionless quantity the
    encoder normalises by and the decoder multiplies ``mu`` by; :meth:`size_factor` performs
    that conversion so no caller has to remember the reference.
    """

    def __init__(self, cfg: Config, *, reference_total: float) -> None:
        super().__init__()
        if not reference_total > 0:
            raise ExpressionError(
                f"SizeFactorHead: reference_total must be > 0 counts, got {reference_total!r}"
            )
        self.cfg = cfg
        self.reference_total = float(reference_total)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.mlp = _mlp(
            [int(cfg.latent_dim), int(cfg.size_factor_hidden), 1],
            generator,
        )
        with torch.no_grad():
            last = cast(nn.Linear, self.mlp[-1])
            last.weight.mul_(0.1)
            last.bias.fill_(math.log(self.reference_total))

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        """``(N, d_h)`` -> ``(N,)`` predicted log total counts; see :meth:`forward`."""
        return cast(Tensor, super().__call__(*args, **kwargs))

    def forward(self, h: Tensor) -> Tensor:
        """``(N, d_h)`` -> ``(N,)`` float32 predicted **log** total counts."""
        if h.ndim != 2:
            raise ExpressionError(f"SizeFactorHead expects (N, d_h), got {tuple(h.shape)}")
        out = cast(Tensor, self.mlp(h)).squeeze(1)
        floor = math.log(float(self.cfg.size_factor_min))
        out = torch.clamp(out, min=floor)
        if self.cfg.debug_shapes:
            assert out.shape == (h.shape[0],), out.shape
        return out

    def size_factor(self, h: Tensor) -> Tensor:
        """``(N, d_h)`` -> ``(N,)`` dimensionless size factor ``exp(log_total) / reference``."""
        return torch.exp(self.forward(h)) / self.reference_total


# --------------------------------------------------------------------------------------
# 2. flow matching in the cell latent
# --------------------------------------------------------------------------------------


def time_embedding(t: Tensor, dim: int) -> Tensor:
    """Sinusoidal embedding of a flow time. ``(N,)`` in ``[0, 1]`` -> ``(N, dim)`` float32.

    ``[sin(2^k pi t), cos(2^k pi t)]`` for ``k = 0 .. dim/2 - 1``: the same octave ladder
    :func:`~spatialcpav25_gen.model.field.fourier_encode` uses for position, so the velocity
    network reads time in the units it reads space in. ``dim`` must be even.
    """
    if dim < 2 or dim % 2 != 0:
        raise ExpressionError(f"time_embedding: dim must be even and >= 2, got {dim}")
    if t.ndim != 1:
        raise ExpressionError(f"time_embedding expects (N,), got {tuple(t.shape)}")
    freqs = (2.0 ** torch.arange(dim // 2, dtype=t.dtype, device=t.device)) * math.pi
    angle = t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=1).to(torch.float32)


class LatentFlow(nn.Module):
    """Conditional flow matching, straight-line path, in ``R^{d_h}``.

    Args:
        cfg:      supplies ``latent_dim`` (d_h), ``flow_hidden``, ``flow_layers``,
                  ``flow_time_dim``, ``cfm_sigma_min``, ``ode_steps``, ``seed`` and
                  ``debug_shapes``.
        cond_dim: width of the conditioning vector
                  ``[F(p), fourier(p), type_emb, region_emb, ctx_retrieval, z_embed]``,
                  assembled by :class:`~spatialcpav25_gen.model.spatialcpav25_gen.CTFFlow`.

    The path is ``h_t = (1-t) h0 + t h1`` with target velocity ``u = h1 - h0`` and
    ``t ~ U(0, 1)``; the loss is ``E_t ||v_theta(h_t, t, cond) - u||^2``.

    ``h0`` is the **GRF** (T03) queried at the cells' physical positions, never an i.i.d.
    Gaussian. It is passed in: this module constructs no noise, so the field stays the one
    object whose continuity in 3D makes two intersecting planes agree.
    """

    def __init__(self, cfg: Config, cond_dim: int) -> None:
        super().__init__()
        if cfg.flow_layers < 2:
            raise ExpressionError(
                f"LatentFlow: Config.flow_layers={cfg.flow_layers} must be >= 2; one layer "
                "makes the velocity a linear function of its inputs, which cannot transport "
                "a Gaussian onto anything else"
            )
        if cond_dim < 1:
            raise ExpressionError(f"LatentFlow: cond_dim must be >= 1, got {cond_dim}")
        self.cfg = cfg
        self.cond_dim = int(cond_dim)
        self.d_h = int(cfg.latent_dim)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        widths = [self.d_h + int(cfg.flow_time_dim) + self.cond_dim]
        widths += [int(cfg.flow_hidden)] * (int(cfg.flow_layers) - 1)
        widths.append(self.d_h)
        self.mlp = _mlp(widths, generator)

    def velocity(self, h_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        """``(N, d_h)``, ``(N,)`` or ``()``, ``(N, cond_dim)`` -> ``(N, d_h)`` float32.

        A scalar ``t`` is broadcast over the batch, which is what the ODE sampler wants.
        """
        if h_t.ndim != 2 or cond.ndim != 2:
            raise ExpressionError(
                f"LatentFlow.velocity expects (N, d_h) and (N, cond_dim), got "
                f"{tuple(h_t.shape)}, {tuple(cond.shape)}"
            )
        if h_t.shape[1] != self.d_h or cond.shape[1] != self.cond_dim:
            raise ExpressionError(
                f"LatentFlow.velocity: expected widths d_h={self.d_h}, "
                f"cond_dim={self.cond_dim}; got {h_t.shape[1]}, {cond.shape[1]}"
            )
        if h_t.shape[0] != cond.shape[0]:
            raise ExpressionError(
                f"LatentFlow.velocity: h_t has {h_t.shape[0]} rows but cond has {cond.shape[0]}"
            )
        times = t.reshape(-1).to(h_t.dtype)
        if times.numel() == 1:
            times = times.expand(h_t.shape[0])
        elif times.numel() != h_t.shape[0]:
            raise ExpressionError(
                f"LatentFlow.velocity: t must be scalar or ({h_t.shape[0]},), got {tuple(t.shape)}"
            )
        features = torch.cat([h_t, time_embedding(times, int(self.cfg.flow_time_dim)), cond], dim=1)
        out = cast(Tensor, self.mlp(features))
        if self.cfg.debug_shapes:
            assert out.shape == h_t.shape, out.shape
        return out

    def cfm_loss(self, h1: Tensor, h0: Tensor, cond: Tensor, gen: torch.Generator) -> Tensor:
        """Conditional flow matching loss. Returns a scalar ``()``.

        ``h1`` is **detached**, as T06 §2 requires: the flow chases a stable target instead of
        co-adapting with the encoder, which is otherwise free to move the target somewhere
        easy to hit. The encoder is trained by the reconstruction term, not by this one.

        ``t`` and the ``cfm_sigma_min`` path noise both come from ``gen``, so two calls with
        the same generator state give the same loss (Convention 3).
        """
        if h1.shape != h0.shape:
            raise ExpressionError(
                f"LatentFlow.cfm_loss: h1 {tuple(h1.shape)} and h0 {tuple(h0.shape)} must match"
            )
        target = h1.detach()
        n = int(target.shape[0])
        t = torch.rand(n, generator=gen, dtype=torch.float32)
        noise = torch.randn(target.shape, generator=gen, dtype=torch.float32)
        h_t = (1.0 - t)[:, None] * h0 + t[:, None] * target
        h_t = h_t + float(self.cfg.cfm_sigma_min) * noise
        velocity = self.velocity(h_t, t, cond)
        out = torch.mean((velocity - (target - h0)) ** 2)
        if self.cfg.debug_shapes:
            assert out.shape == (), out.shape
        return out

    def sample(self, h0: Tensor, cond: Tensor, steps: int) -> Tensor:
        """Integrate the ODE from ``h0`` at ``t = 0`` to ``t = 1``. ``(N, d_h)`` -> ``(N, d_h)``.

        Heun (2nd order, one extra velocity evaluation per step), ``steps`` uniform steps.
        Deterministic: given ``h0`` and ``cond`` this touches no RNG at all, which is what
        makes the generation path reproducible from the GRF's seed alone.
        """
        if steps < 1:
            raise ExpressionError(f"LatentFlow.sample: steps must be >= 1, got {steps}")
        h = h0
        dt = 1.0 / float(steps)
        for index in range(steps):
            t0 = torch.full((1,), index * dt, dtype=torch.float32)
            t1 = torch.full((1,), min((index + 1) * dt, 1.0), dtype=torch.float32)
            v0 = self.velocity(h, t0, cond)
            v1 = self.velocity(h + dt * v0, t1, cond)
            h = h + 0.5 * dt * (v0 + v1)
        if self.cfg.debug_shapes:
            assert h.shape == h0.shape, h.shape
        return h


# --------------------------------------------------------------------------------------
# 3. the gene-conditioned decoder
# --------------------------------------------------------------------------------------


class _GeneConditionedTrunk(nn.Module):
    """The shared ``u_ig -> hidden`` body of both decoders. Returns ``(N, G', hidden)``.

    ``u_ig = [h_i, e_g, h_i * (A e_g)]`` (T06 §3). The first linear layer is applied as
    three separate projections summed by broadcasting — ``W u = W_h h + W_e e_g + W_i (h *
    A e_g)`` — which is the same function as ``Linear(cat(...))`` and never materialises the
    ``(N, G', 2 d_h + d_e)`` concatenation. The interaction term itself is
    ``(N, G', d_h)`` and unavoidable; it is the only place a *per-pair* tensor is required,
    and it is what "bilinear/FiLM interaction" means.
    """

    def __init__(self, cfg: Config, gene_emb_dim: int) -> None:
        super().__init__()
        if cfg.decoder_layers < 2:
            raise ExpressionError(
                f"decoder: Config.decoder_layers={cfg.decoder_layers} must be >= 2; one "
                "layer makes every ZINB parameter a linear read-out of u_ig"
            )
        self.cfg = cfg
        self.gene_emb_dim = int(gene_emb_dim)
        d_h = int(cfg.latent_dim)
        hidden = int(cfg.decoder_hidden)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.A = nn.Linear(self.gene_emb_dim, d_h, bias=False)
        self.in_h = nn.Linear(d_h, hidden)
        self.in_e = nn.Linear(self.gene_emb_dim, hidden, bias=False)
        self.in_i = nn.Linear(d_h, hidden, bias=False)
        for layer in (self.A, self.in_h, self.in_e, self.in_i):
            _init_linear(layer, generator)
        body: list[nn.Module] = [nn.GELU()]
        for _ in range(int(cfg.decoder_layers) - 2):
            linear = nn.Linear(hidden, hidden)
            _init_linear(linear, generator)
            body.extend((linear, nn.GELU()))
        self.body = nn.Sequential(*body)

    def forward(self, h: Tensor, gene_emb: Tensor) -> Tensor:
        """``(N, d_h)``, ``(G', d_e)`` -> ``(N, G', decoder_hidden)`` float32."""
        interaction = h[:, None, :] * self.A(gene_emb)[None, :, :]
        pre = self.in_h(h)[:, None, :] + self.in_e(gene_emb)[None, :, :] + self.in_i(interaction)
        out = cast(Tensor, self.body(pre))
        if self.cfg.debug_shapes:
            assert out.shape == (h.shape[0], gene_emb.shape[0], int(self.cfg.decoder_hidden))
        return out


def _check_decoder_inputs(h: Tensor, gene_emb: Tensor, size_factor: Tensor, name: str) -> None:
    """Raise :class:`ExpressionError` unless the three decoder inputs agree in shape."""
    if h.ndim != 2 or gene_emb.ndim != 2 or size_factor.ndim != 1:
        raise ExpressionError(
            f"{name} expects (N, d_h), (G', d_e), (N,); got {tuple(h.shape)}, "
            f"{tuple(gene_emb.shape)}, {tuple(size_factor.shape)}"
        )
    if h.shape[0] != size_factor.shape[0]:
        raise ExpressionError(
            f"{name}: h has {h.shape[0]} rows but size_factor has {size_factor.shape[0]}"
        )


class ZINBDecoder(nn.Module):
    """Gene-conditioned zero-inflated negative binomial head. Returns ``(mu, theta, pi)``.

        Args:
            cfg: supplies ``latent_dim``, ``decoder_hidden``, ``decoder_layers``, ``zinb_eps``,
                 ``zinb_theta_min/_max``, ``zinb_mu_min/_max``, ``seed`` and ``debug_shapes``.
            gene_emb_dim: width of ``e_g`` (``Config.gene_emb_dim`` in the pipeline; passed
                 explicitly so a zero-shot embedding of a different width is a type error rather
                 than a broadcast).
            init_mean_expression: mean count per (cell, gene) over the training volume. The
                 ``mu`` head's bias starts at ``softplus^-1`` of it, so an untrained decoder
                 already predicts the panel's average count and the fit learns the structure. A
                 dataset property, like ``IntensityHead(init_density=...)``.

    ``mu = link(MLP_mu(u)) * size_factor``, ``theta = softplus(MLP_theta(u)) + eps``,
        ``pi = sigmoid(MLP_pi(u))``, with the guards T06 §3 requires: ``theta`` clamped to
        ``[zinb_theta_min, zinb_theta_max]``, ``mu`` to ``[zinb_mu_min, zinb_mu_max]``.

        ``link`` is ``Config.decoder_mu_link``: **``exp`` by default since T10**, with
        ``softplus`` — ``specs/06`` §3's own choice, and the default up to T10 — selectable.
        **That field's docstring carries both measurements**: T06's on the synthetic fixture,
        where ``softplus`` won, and T10's on real STARmap, where ``exp`` takes the structured
        share of between-cell variance from 15.1 % to 61.4 %. ``theta`` keeps softplus because a
        dispersion spans one order of magnitude, not four.

        ``theta`` and ``pi`` are per *(cell, gene)*, not per gene. A per-gene dispersion cannot
        express that the same gene is over-dispersed in one region and not in another, and T09
        calibrates ``log theta`` per gene *on top of* this — a shift of a distribution the model
        still has to produce.
    """

    def __init__(self, cfg: Config, gene_emb_dim: int, *, init_mean_expression: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = _GeneConditionedTrunk(cfg, gene_emb_dim)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        hidden = int(cfg.decoder_hidden)
        self.head_mu = nn.Linear(hidden, 1)
        self.head_theta = nn.Linear(hidden, 1)
        self.head_pi = nn.Linear(hidden, 1)
        for layer in (self.head_mu, self.head_theta, self.head_pi):
            _init_linear(layer, generator)
        self._log_mu_min = math.log(float(cfg.zinb_mu_min))
        self._log_mu_max = math.log(float(cfg.zinb_mu_max))
        mean = max(float(init_mean_expression), float(cfg.zinb_mu_min))
        with torch.no_grad():
            for layer in (self.head_mu, self.head_theta, self.head_pi):
                layer.weight.mul_(0.1)
            self.head_mu.bias.fill_(
                math.log(mean) if cfg.decoder_mu_link == "exp" else _softplus_inverse(mean)
            )
            self.head_theta.bias.fill_(_softplus_inverse(1.0))
            self.head_pi.bias.zero_()

    @property
    def gene_emb_dim(self) -> int:
        """Width of the gene embedding this decoder was built for."""
        return self.trunk.gene_emb_dim

    def __call__(self, *args: Any, **kwargs: Any) -> DecoderOutput:
        """``(N, d_h)``, ``(G', d_e)``, ``(N,)`` -> three ``(N, G')``; see :meth:`forward`."""
        return cast(DecoderOutput, super().__call__(*args, **kwargs))

    def forward(self, h: Tensor, gene_emb: Tensor, size_factor: Tensor) -> DecoderOutput:
        """``(N, d_h)``, ``(G', d_e)``, ``(N,)`` -> ``mu``, ``theta``, ``pi`` each ``(N, G')``."""
        _check_decoder_inputs(h, gene_emb, size_factor, "ZINBDecoder")
        features = self.trunk(h, gene_emb)
        eps = float(self.cfg.zinb_eps)
        scale = torch.clamp(size_factor, min=eps)[:, None]
        raw = self.head_mu(features).squeeze(-1)
        if self.cfg.decoder_mu_link == "exp":
            # The pre-activation is log-expression, clamped to the guards *before* the
            # exponential so an early-training excursion cannot produce an inf.
            mu = torch.exp(torch.clamp(raw, min=self._log_mu_min, max=self._log_mu_max)) * scale
        else:
            mu = torch.nn.functional.softplus(raw) * scale
        mu = torch.clamp(mu, min=float(self.cfg.zinb_mu_min), max=float(self.cfg.zinb_mu_max))
        theta = torch.nn.functional.softplus(self.head_theta(features).squeeze(-1)) + eps
        theta = torch.clamp(
            theta, min=float(self.cfg.zinb_theta_min), max=float(self.cfg.zinb_theta_max)
        )
        pi = torch.clamp(torch.sigmoid(self.head_pi(features).squeeze(-1)), min=eps, max=1.0 - eps)
        if self.cfg.debug_shapes:
            shape = (h.shape[0], gene_emb.shape[0])
            assert mu.shape == shape, mu.shape
            assert theta.shape == shape, theta.shape
            assert pi.shape == shape, pi.shape
        return mu, theta, pi


class ZIGammaDecoder(nn.Module):
    """Zero-inflated Gamma head for intensity-valued (non-count) assays. Same interface.

    Selected by ``Config.decoder = "zigamma"``. ``mu`` is the Gamma's mean and ``theta`` its
    **shape** (so the variance is ``mu^2 / theta``, and the coefficient of variation is
    ``theta^{-1/2}`` independently of the mean — which is what a fluorescence intensity does
    and a count does not). ``pi`` is the probability of an exact zero, which intensity assays
    still produce.

    Structurally identical to :class:`ZINBDecoder` — same trunk, same guards, same three
    heads — so that ``Config.decoder`` selects between two likelihoods rather than two
    architectures. The likelihoods differ in
    :mod:`spatialcpav25_gen.losses.reconstruction`.
    """

    def __init__(self, cfg: Config, gene_emb_dim: int, *, init_mean_expression: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.zinb = ZINBDecoder(cfg, gene_emb_dim, init_mean_expression=init_mean_expression)

    @property
    def gene_emb_dim(self) -> int:
        """Width of the gene embedding this decoder was built for."""
        return self.zinb.gene_emb_dim

    def __call__(self, *args: Any, **kwargs: Any) -> DecoderOutput:
        """``(N, d_h)``, ``(G', d_e)``, ``(N,)`` -> three ``(N, G')``; see :meth:`forward`."""
        return cast(DecoderOutput, super().__call__(*args, **kwargs))

    def forward(self, h: Tensor, gene_emb: Tensor, size_factor: Tensor) -> DecoderOutput:
        """``(N, d_h)``, ``(G', d_e)``, ``(N,)`` -> ``mu``, ``theta`` (shape), ``pi``."""
        return self.zinb.forward(h, gene_emb, size_factor)


def build_decoder(
    cfg: Config, gene_emb_dim: int, *, init_mean_expression: float
) -> ZINBDecoder | ZIGammaDecoder:
    """Construct the decoder ``Config.decoder`` names.

    ``"gaussian"`` is ablation A6 and belongs to T10, which owns the ablation table; it
    raises here rather than silently decoding counts through the ZINB head (Convention 6).
    """
    if cfg.decoder == "zinb":
        return ZINBDecoder(cfg, gene_emb_dim, init_mean_expression=init_mean_expression)
    if cfg.decoder == "zigamma":
        return ZIGammaDecoder(cfg, gene_emb_dim, init_mean_expression=init_mean_expression)
    raise ExpressionError(
        f"Config.decoder={cfg.decoder!r} is not built yet. 'zinb' and 'zigamma' are T06; "
        "'gaussian' is ablation A6 and is owed by T10, which owns the ablation table."
    )


# --------------------------------------------------------------------------------------
# 4. the output path — counts, never means
# --------------------------------------------------------------------------------------


def _as_numpy(values: Tensor | npt.NDArray[Any], name: str) -> FloatArray:
    """Detach a tensor (or accept an array) as float64, for the numpy samplers."""
    if isinstance(values, Tensor):
        array = cast(FloatArray, values.detach().to(torch.float64).cpu().numpy())
    else:
        array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ExpressionError(f"{name} contains non-finite values")
    return array


def sample_counts(
    mu: Tensor | npt.NDArray[Any],
    theta: Tensor | npt.NDArray[Any],
    pi: Tensor | npt.NDArray[Any],
    gen: np.random.Generator,
) -> Tensor:
    """Draw ZINB counts. Three ``(N, G')`` -> ``(N, G')`` float32, integer-valued.

    ``NB(mu, theta)`` then zeroed with probability ``pi``. **Never returns ``mu``** — that is
    the whole point of the function, and :func:`assert_detection_rate` is the guard in the
    generation path that would catch a regression to means.

    The negative binomial is drawn as a Gamma-Poisson mixture
    (``lambda ~ Gamma(theta, mu/theta)``, ``x ~ Poisson(lambda)``) rather than through
    ``Generator.negative_binomial``: it is the same distribution, it stays finite at the
    ``theta`` guards' extremes (``theta = 1e6`` makes ``p = theta/(theta+mu)`` round to 1 and
    the direct sampler is then drawing from a degenerate geometric mixture), and it is the
    construction the synthetic fixture's own ``sample_counts`` uses.

    ``gen`` is an explicit ``numpy.random.Generator`` (Convention 3; see the module
    docstring for why it is numpy).
    """
    mu_a = _as_numpy(mu, "sample_counts: mu")
    theta_a = _as_numpy(theta, "sample_counts: theta")
    pi_a = _as_numpy(pi, "sample_counts: pi")
    if mu_a.shape != theta_a.shape or mu_a.shape != pi_a.shape:
        raise ExpressionError(
            f"sample_counts: mu {mu_a.shape}, theta {theta_a.shape} and pi {pi_a.shape} "
            "must have the same shape"
        )
    if mu_a.min(initial=0.0) < 0.0 or theta_a.min(initial=1.0) <= 0.0:
        raise ExpressionError("sample_counts: mu must be >= 0 and theta > 0")
    lam = gen.gamma(shape=theta_a, scale=np.maximum(mu_a, 0.0) / theta_a)
    counts = gen.poisson(lam)
    dropout = gen.random(mu_a.shape) < pi_a
    return torch.from_numpy(np.where(dropout, 0.0, counts).astype(np.float32))


def draw_mean_variance(mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config) -> tuple[Tensor, Tensor]:
    """Mean and variance of one draw from the decoder. Three ``(N, G')`` -> two ``(N, G')``.

    Closed forms for the two likelihoods this decoder can carry, both zero-inflated, both of
    the shape ``Var = (1 - pi) * Var_base + pi (1 - pi) mu^2``:

    * ``zinb`` — ``Var_base = mu (1 + mu / theta)``, the negative binomial's;
    * ``zigamma`` — ``Var_base = mu^2 / theta``, the Gamma's, ``theta`` being its shape.

    Differentiable in all three arguments and free of any draw, which is what it is for: T08's
    metric-aware autocorrelation term needs the *expected* value of a statistic of a draw, and
    a statistic that divides by a sample variance needs that variance analytically or not at
    all (see :func:`~spatialcpav25_gen.losses.metric_aware.morans_i`'s ``noise_var``).
    """
    if cfg.decoder == "zinb":
        base = mu * (1.0 + mu / theta)
    elif cfg.decoder == "zigamma":
        base = mu.pow(2) / theta
    else:  # pragma: no cover - build_decoder refuses anything else first
        raise ExpressionError(
            f"draw_mean_variance: no closed form for Config.decoder={cfg.decoder!r}"
        )
    keep = 1.0 - pi
    return keep * mu, keep * base + pi * keep * mu.pow(2)


def sample_zigamma(
    mu: Tensor | npt.NDArray[Any],
    theta: Tensor | npt.NDArray[Any],
    pi: Tensor | npt.NDArray[Any],
    gen: np.random.Generator,
) -> Tensor:
    """Draw zero-inflated Gamma intensities. Three ``(N, G')`` -> ``(N, G')`` float32.

    The non-count twin of :func:`sample_counts`: ``Gamma(shape=theta, scale=mu/theta)`` zeroed
    with probability ``pi``. Not integer-valued, by construction — an intensity assay's values
    are not counts, and pretending otherwise is what ``Config.decoder`` exists to avoid.
    """
    mu_a = _as_numpy(mu, "sample_zigamma: mu")
    theta_a = _as_numpy(theta, "sample_zigamma: theta")
    pi_a = _as_numpy(pi, "sample_zigamma: pi")
    if mu_a.shape != theta_a.shape or mu_a.shape != pi_a.shape:
        raise ExpressionError("sample_zigamma: mu, theta and pi must have the same shape")
    values = gen.gamma(shape=theta_a, scale=np.maximum(mu_a, 0.0) / theta_a)
    dropout = gen.random(mu_a.shape) < pi_a
    return torch.from_numpy(np.where(dropout, 0.0, values).astype(np.float32))


def cross_mix_counts(
    donor_counts: Tensor | npt.NDArray[Any],
    weights: Tensor | npt.NDArray[Any],
    gen: np.random.Generator,
    *,
    cfg: Config | None = None,
) -> Tensor:
    """Apply the v20 Bernoulli cross-mix. ``(N, D, G)``, ``(N, D)`` -> ``(N, G)`` float32.

    Per **cell and gene**, one donor is chosen by a draw on that cell's donor weights and
    **that donor's count is taken verbatim**. Every emitted value is therefore a real
    measurement: no value is ever a blend of two donors, which is neither an integer nor a
    draw from anything, and sparsity, dispersion and count-ness hold by construction. This is
    ``Config.expr_mode = "cross-mix"``, the previous version's expression path, kept as the
    no-regression fallback T09's selector must be able to recover (SPEC_QUESTIONS A6).

    Parameters
    ----------
    donor_counts
        ``(N, D, G)`` real counts of the ``D`` donors available to each of the ``N`` cells.
    weights
        ``(N, D)`` non-negative donor weights, each row summing to 1 to within
        ``Config.cross_mix_weight_tol``. In v20's two-donor case the row is
        ``[1 - alpha w_other, alpha w_other]``.
    gen
        Explicit ``numpy.random.Generator``. **One** ``gen.random((N, G))`` draw is consumed,
        in C order, and the donor is the largest ``j`` with ``u < sum_{k >= j} w_k``. Both
        halves of that sentence are load-bearing: they are what make the two-donor case
        bitwise identical to v20's ``rng.random(expr.shape) < w_other`` followed by
        ``np.where(mask, partner, base)`` — the same uniform draw and the same event. A
        forward cumulative sum would select a *different* set of entries from the same
        numbers, with the same distribution and no bitwise agreement, and
        ``test_cross_mix_matches_v20`` would then have had to be weakened to a
        donor-frequency check.
    cfg
        Additive keyword (as T01's ``split_holdout(..., cfg=None)`` and T02's
        ``build_gene_meta(..., cfg=None)``): T06's signature has nowhere to read
        ``Config.cross_mix_weight_tol`` from, and Convention 1 forbids the tolerance being
        written inline. ``None`` uses the default config's value.
    """
    tol = float((cfg or Config()).cross_mix_weight_tol)
    counts = np.asarray(
        donor_counts.detach().cpu().numpy() if isinstance(donor_counts, Tensor) else donor_counts,
        dtype=np.float64,
    )
    w = _as_numpy(weights, "cross_mix_counts: weights")
    if counts.ndim != 3 or w.ndim != 2:
        raise ExpressionError(
            f"cross_mix_counts expects (N, D, G) and (N, D), got {counts.shape} and {w.shape}"
        )
    if counts.shape[:2] != w.shape:
        raise ExpressionError(
            f"cross_mix_counts: donor_counts {counts.shape} and weights {w.shape} disagree "
            "about N or D"
        )
    if w.min(initial=0.0) < 0.0:
        raise ExpressionError("cross_mix_counts: weights must be non-negative")
    total = w.sum(axis=1)
    if np.any(np.abs(total - 1.0) > tol):
        worst = int(np.argmax(np.abs(total - 1.0)))
        raise ExpressionError(
            f"cross_mix_counts: weights row {worst} sums to {total[worst]!r}, further than "
            f"{tol} from 1. The donor weights are a distribution over donors; a row that "
            "does not sum to 1 would silently bias the choice towards the base donor"
        )
    n, _, g = counts.shape
    u = gen.random((n, g))
    # Suffix sums, so that with D = 2 the event is exactly v20's `u < w_other`.
    suffix = np.cumsum(w[:, ::-1], axis=1)[:, ::-1]
    chosen = (u[:, None, :] < suffix[:, 1:, None]).sum(axis=1)
    out = np.take_along_axis(counts, chosen[:, None, :], axis=1)[:, 0, :]
    if out.shape != (n, g):
        raise ExpressionError(f"cross_mix_counts: internal shape error {out.shape}")
    return torch.from_numpy(out.astype(np.float32))


def detection_rate(counts: Tensor | npt.NDArray[Any]) -> FloatArray:
    """Per-gene fraction of cells with a non-zero value. ``(N, G)`` -> ``(G,)`` float64."""
    array = _as_numpy(counts, "detection_rate: counts")
    if array.ndim != 2:
        raise ExpressionError(f"detection_rate expects (N, G), got {array.shape}")
    if array.shape[0] == 0:
        raise ExpressionError("detection_rate: needs at least one cell")
    return cast(FloatArray, (array > 0).mean(axis=0))


def assert_detection_rate(
    counts: Tensor | npt.NDArray[Any], reference: float, cfg: Config, *, what: str = "generated"
) -> float:
    """Check an emitted matrix's median detection rate against the training sections'.

    The guard T06 §4 asks for in the generation path. Returns the measured median rate;
    raises :class:`ExpressionError` when it is further than ``Config.detection_rate_tol``
    from ``reference``.

    A silent switch to emitting ``mu`` — the failure this exists to catch — puts essentially
    every gene's detection rate at 1.0, and v20's densification bug put the median at 0.909
    against a truth of 0.216. Both are far outside the band.
    """
    measured = float(np.median(detection_rate(counts)))
    tol = float(cfg.detection_rate_tol)
    if abs(measured - float(reference)) > tol:
        raise ExpressionError(
            f"{what} expression has median per-gene detection rate {measured:.4f} against the "
            f"training sections' {float(reference):.4f}, outside the plausible band of "
            f"+-{tol} (Config.detection_rate_tol). A detection rate near 1.0 is what emitting "
            "mu instead of a draw looks like; a rate far above the reference is the "
            "densification that motivated the v20 cross-mix."
        )
    return measured
