"""Likelihood terms: the layout's point-process NLL (T05) and expression's (T06).

The two live together because they are the two halves of "what did the model say about the
data it was shown" — where the cells are, and what they expressed.

Every expression likelihood here is written against **raw counts** (Convention 5). The
``log1p`` normalisation the encoder applies is an input-side transform and must never reach
these functions' ``counts`` argument; a normalised target would make ``mu`` a mean of
logarithms and every downstream count statistic wrong.

Numerics
--------
A numerically sloppy ZINB is a classic source of silent NaNs (T06 §3), so the log-likelihood
is assembled entirely in log space: the ``x = 0`` branch is a ``logaddexp`` of the two ways a
zero can arise, and neither branch ever forms ``mu / (theta + mu)`` outside a logarithm.
``zinb_log_prob`` is asserted against a hand-derived reference to 1e-5 and asserted finite
over the whole guarded parameter range (``mu`` 1e-8 to 1e8, ``theta`` 1e-4 to 1e6, ``pi`` 0
to 1) by ``tests/test_expression.py``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spatialcpav25_gen.config import Config

__all__ = [
    "layout_poisson_nll",
    "size_factor_loss",
    "zigamma_log_prob",
    "zigamma_nll",
    "zinb_log_prob",
    "zinb_nll",
]


def layout_poisson_nll(
    lambda_cells: Tensor,
    cell_type: Tensor,
    lambda_mc: Tensor,
    slab_volume: float,
    cfg: Config,
) -> Tensor:
    """Inhomogeneous Poisson process NLL of one section's layout. Returns a scalar.

    Parameters
    ----------
    lambda_cells
        ``(N, C)`` per-type intensity at the ``N`` observed cell positions, in cells per
        micrometre^3 — the output of :class:`~spatialcpav25_gen.model.layout.IntensityHead`.
    cell_type
        ``(N,)`` int64 type codes of those cells.
    lambda_mc
        ``(M, C)`` per-type intensity at ``M`` points drawn **uniformly in the slab
        volume** (``uniform_slab_points``).
    slab_volume
        The slab's volume in micrometre^3: the section's in-plane window times its
        thickness.
    cfg
        Supplies ``layout_intensity_eps`` and ``debug_shapes``.

    Returns
    -------
    Tensor
        Scalar ``()``, ``-sum_i log lambda_{c_i}(p_i) + integral over the slab of
        sum_c lambda_c``, the second term estimated as
        ``slab_volume * mean_m sum_c lambda_mc[m, c]``.

    Notes
    -----
    **The slab volume, not the area.** T07's thickness-consistency loss is stated in terms
    of a slab, and a count that emerges from a *volume* integral is what makes it coherent:
    doubling a section's thickness must double its expected cell count, and an area integral
    would leave it unchanged.

    The term is a sum over cells, not a mean, because that is what the process likelihood
    is — the two terms have to stay on the same scale for the fitted intensity to have the
    right units. A caller that wants a per-cell number for logging divides afterwards.
    """
    if cfg.debug_shapes:
        assert lambda_cells.ndim == 2, lambda_cells.shape
        assert lambda_mc.ndim == 2, lambda_mc.shape
        assert cell_type.shape == (lambda_cells.shape[0],), cell_type.shape
        assert lambda_mc.shape[1] == lambda_cells.shape[1], lambda_mc.shape
    if lambda_mc.shape[1] != lambda_cells.shape[1]:
        raise ValueError(
            f"layout_poisson_nll: lambda_cells has {lambda_cells.shape[1]} cell types but "
            f"lambda_mc has {lambda_mc.shape[1]}"
        )
    if cell_type.shape[0] != lambda_cells.shape[0]:
        raise ValueError(
            f"layout_poisson_nll: cell_type has {cell_type.shape[0]} entries but "
            f"lambda_cells has {lambda_cells.shape[0]} rows"
        )
    eps = float(cfg.layout_intensity_eps)
    observed = lambda_cells.gather(1, cell_type.to(torch.long)[:, None]).squeeze(1)
    likelihood = -torch.log(observed + eps).sum()
    integral = float(slab_volume) * lambda_mc.sum(dim=1).mean()
    out = likelihood + integral
    if cfg.debug_shapes:
        assert out.shape == (), out.shape
    return out


# --------------------------------------------------------------------------------------
# expression (T06)
# --------------------------------------------------------------------------------------


def _check_expression_shapes(
    counts: Tensor, mu: Tensor, theta: Tensor, pi: Tensor, caller: str
) -> None:
    """Raise ``ValueError`` unless the four expression arguments are the same ``(N, G')``."""
    if counts.shape != mu.shape or counts.shape != theta.shape or counts.shape != pi.shape:
        raise ValueError(
            f"{caller}: counts {tuple(counts.shape)}, mu {tuple(mu.shape)}, theta "
            f"{tuple(theta.shape)} and pi {tuple(pi.shape)} must all be (N, G')"
        )
    if counts.ndim != 2:
        raise ValueError(f"{caller}: expected (N, G') tensors, got {counts.ndim} dimensions")


def zinb_log_prob(counts: Tensor, mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config) -> Tensor:
    """Zero-inflated negative binomial log-likelihood, elementwise. ``(N, G')`` -> ``(N, G')``.

    Parameters
    ----------
    counts
        ``(N, G')`` **raw** counts (Convention 5). Non-integer values are permitted by the
        arithmetic (``lgamma`` is defined between the integers) but mean the caller has
        handed a normalised matrix to a count likelihood; ``validate_volume`` warns about
        that upstream and ``Config.decoder = "zigamma"`` is the intended path for
        intensity-valued assays.
    mu, theta, pi
        ``(N, G')`` mean, dispersion and zero-inflation probability, as returned by
        :class:`~spatialcpav25_gen.model.expression.ZINBDecoder`.
    cfg
        Supplies ``zinb_eps``, ``zinb_mu_min/_max``, ``zinb_theta_min/_max`` and
        ``debug_shapes``. The guards are applied here as well as in the decoder, so that a
        hand-built parameter set — a test, T07's teacher, T09's calibrated ``theta`` — cannot
        reach the logarithms unguarded.

    Returns
    -------
    Tensor
        ``(N, G')`` ``log P(x | mu, theta, pi)``, in nats. The parameterisation is NB2:
        ``E[x] = mu``, ``Var[x] = mu + mu^2 / theta``.

    Notes
    -----
    The two branches, both in log space, with ``L = log(theta + mu)``::

        x = 0 : logaddexp( log pi , log(1 - pi) + theta (log theta - L) )
        x > 0 : log(1 - pi) + lgamma(x + theta) - lgamma(theta) - lgamma(x + 1)
                            + theta (log theta - L) + x (log mu - L)

    A zero can arise two ways — the inflation component, or the NB itself — and the
    ``logaddexp`` is what keeps both paths' gradients finite when one of them dominates.
    """
    _check_expression_shapes(counts, mu, theta, pi, "zinb_log_prob")
    eps = float(cfg.zinb_eps)
    mu_c = torch.clamp(mu, min=float(cfg.zinb_mu_min), max=float(cfg.zinb_mu_max))
    theta_c = torch.clamp(theta, min=float(cfg.zinb_theta_min), max=float(cfg.zinb_theta_max))
    pi_c = torch.clamp(pi, min=eps, max=1.0 - eps)
    x = counts.to(mu_c.dtype)

    log_theta_mu = torch.log(theta_c + mu_c)
    log_theta = torch.log(theta_c)
    log_mu = torch.log(mu_c)
    log_pi = torch.log(pi_c)
    log_not_pi = torch.log1p(-pi_c)

    nb_zero = theta_c * (log_theta - log_theta_mu)
    zero_case = torch.logaddexp(log_pi, log_not_pi + nb_zero)
    positive_case = (
        log_not_pi
        + torch.lgamma(x + theta_c)
        - torch.lgamma(theta_c)
        - torch.lgamma(x + 1.0)
        + nb_zero
        + x * (log_mu - log_theta_mu)
    )
    out = torch.where(x < 1.0, zero_case, positive_case)
    if cfg.debug_shapes:
        assert out.shape == counts.shape, out.shape
    return out


def zinb_nll(counts: Tensor, mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config) -> Tensor:
    """Mean ZINB negative log-likelihood over the ``(cell, gene)`` pairs. Scalar ``()``.

    A **mean**, not a sum, for the same reason T02's distillation loss is one: the loss
    weights in ``Config`` have to mean the same thing on a 200-gene and a 20 000-gene panel,
    and ``Config.genes_per_step`` changes the pair count every step. ``w_recon`` multiplies
    nats per (cell, gene).
    """
    out = -zinb_log_prob(counts, mu, theta, pi, cfg).mean()
    if cfg.debug_shapes:
        assert out.shape == (), out.shape
    return out


def zigamma_log_prob(counts: Tensor, mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config) -> Tensor:
    """Zero-inflated Gamma log-likelihood, elementwise. ``(N, G')`` -> ``(N, G')``.

    The likelihood for intensity-valued assays (``Config.decoder = "zigamma"``), with ``mu``
    the mean and ``theta`` the **shape** — so ``Var = mu^2 / theta`` and the coefficient of
    variation does not depend on the mean, which is what a fluorescence measurement does.
    An exact zero is attributed entirely to the inflation component: a continuous density
    puts no mass on a point, so unlike the ZINB there is no second way to be zero.
    """
    _check_expression_shapes(counts, mu, theta, pi, "zigamma_log_prob")
    eps = float(cfg.zinb_eps)
    mu_c = torch.clamp(mu, min=float(cfg.zinb_mu_min), max=float(cfg.zinb_mu_max))
    theta_c = torch.clamp(theta, min=float(cfg.zinb_theta_min), max=float(cfg.zinb_theta_max))
    pi_c = torch.clamp(pi, min=eps, max=1.0 - eps)
    x = torch.clamp(counts.to(mu_c.dtype), min=0.0)
    rate = theta_c / mu_c
    positive_case = (
        torch.log1p(-pi_c)
        + theta_c * torch.log(rate)
        + (theta_c - 1.0) * torch.log(torch.clamp(x, min=eps))
        - rate * x
        - torch.lgamma(theta_c)
    )
    out = torch.where(x <= 0.0, torch.log(pi_c), positive_case)
    if cfg.debug_shapes:
        assert out.shape == counts.shape, out.shape
    return out


def zigamma_nll(counts: Tensor, mu: Tensor, theta: Tensor, pi: Tensor, cfg: Config) -> Tensor:
    """Mean zero-inflated Gamma negative log-likelihood. Scalar ``()``."""
    out = -zigamma_log_prob(counts, mu, theta, pi, cfg).mean()
    if cfg.debug_shapes:
        assert out.shape == (), out.shape
    return out


def size_factor_loss(predicted_log_total: Tensor, observed_total: Tensor, cfg: Config) -> Tensor:
    """Squared error of the decoded library size, in log counts. Scalar ``()``.

    T06 §3: "``size_factor`` decoded from ``h_i`` by a small head, trained against the
    observed per-cell total". In **log** counts rather than counts, because per-cell totals
    are log-normally distributed over an order of magnitude or more and a squared error in
    counts would be a fit to the largest cells alone.
    """
    if predicted_log_total.ndim != 1 or observed_total.ndim != 1:
        raise ValueError(
            f"size_factor_loss expects (N,) tensors, got {tuple(predicted_log_total.shape)} "
            f"and {tuple(observed_total.shape)}"
        )
    if predicted_log_total.shape != observed_total.shape:
        raise ValueError(
            f"size_factor_loss: predicted {tuple(predicted_log_total.shape)} and observed "
            f"{tuple(observed_total.shape)} must match"
        )
    target = torch.log(
        torch.clamp(observed_total.to(torch.float32), min=float(cfg.size_factor_min))
    )
    out = torch.mean((predicted_log_total - target) ** 2)
    if cfg.debug_shapes:
        assert out.shape == (), out.shape
    return out
