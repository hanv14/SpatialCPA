"""
Loss functions for SpatialCPA-v12.

Extends v11's objectives with a **factor-analysis negative log-likelihood** for the
generative Stage-2 decoder, which trains the shared loadings ``L`` and idiosyncratic
noise ``Ψ`` to the real gene-gene covariance so that generated cells reproduce it.

Groups (see :class:`~spatialcpav12.config.LossConfig`):

* **Layout reconstruction / distillation** — occupancy BCE + type CE on real spots;
  cosine feature-distillation of the student layout code to the teacher embedding;
  CE distillation of the type field to the teacher pseudo-layout (domains).
* **Expression** — mean reconstruction (MSE, anchors ``mu``) + factor-analysis NLL of
  the real spots under ``N(mu, L Lᵀ + Ψ)`` (fits the covariance).
* **Cross-z consistency** — the layout and expression means evaluated at ``z`` and
  ``z ± dz`` should be close (finite-difference smoothness across the continuous axis).
* **Biology-informed constraints** — interface preservation, within-domain expression
  gradient smoothness, and spatial-domain coherence.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def occupancy_bce(occ_logit, target):
    return F.binary_cross_entropy_with_logits(occ_logit, target)


def type_ce(type_logits, target_idx, weight=None):
    return F.cross_entropy(type_logits, target_idx, weight=weight)


def expr_mse(pred, target):
    return F.mse_loss(pred, target)


def factor_analysis_nll(x, mu, loadings, psi):
    """Mean per-cell negative log-likelihood under ``N(mu, L Lᵀ + Ψ)``.

    Uses the Woodbury identity and the matrix-determinant lemma so the G×G
    covariance is never formed: with ``Ψ = diag(psi)`` (G,), ``L`` (G, r),
    residual ``d = x - mu`` (n, G):

        Σ⁻¹ = Ψ⁻¹ − Ψ⁻¹ L (I_r + Lᵀ Ψ⁻¹ L)⁻¹ Lᵀ Ψ⁻¹
        log|Σ| = Σ log psi + log|I_r + Lᵀ Ψ⁻¹ L|

    so the quadratic term is ``dᵀΨ⁻¹d − wᵀ M⁻¹ w`` with ``w = Lᵀ Ψ⁻¹ d`` (r,),
    ``M = I_r + Lᵀ Ψ⁻¹ L`` (r, r) — an r×r solve shared across all cells.
    """
    n, G = x.shape
    r = loadings.shape[1]
    d = x - mu                                   # (n, G)
    inv_psi = 1.0 / psi                          # (G,)
    Lp = loadings * inv_psi[:, None]             # Ψ⁻¹ L  (G, r)
    M = torch.eye(r, device=x.device, dtype=x.dtype) + loadings.t() @ Lp     # (r, r)
    # Cholesky for a stable solve + logdet of the small r×r system.
    jitter = 1e-5 * torch.eye(r, device=x.device, dtype=x.dtype)
    chol = torch.linalg.cholesky(M + jitter)
    w = d @ Lp                                   # (n, r) = (Ψ⁻¹L)ᵀ d per cell
    quad_diag = (d * d * inv_psi[None, :]).sum(dim=1)                 # (n,)
    sol = torch.cholesky_solve(w.t(), chol).t()                      # (n, r) = M⁻¹ w
    quad_low = (w * sol).sum(dim=1)                                   # (n,)
    quad = quad_diag - quad_low                                      # (n,)
    logdet = torch.log(psi).sum() + 2.0 * torch.log(torch.diagonal(chol)).sum()
    const = G * np.log(2.0 * np.pi)
    return 0.5 * (quad + logdet + const).mean()


def distill_embed(student_code, teacher_embed):
    """Cosine feature-distillation (align student layout code to teacher embedding)."""
    s = F.normalize(student_code, dim=-1)
    t = F.normalize(teacher_embed, dim=-1)
    return (1.0 - (s * t).sum(-1)).mean()


def consistency(f_z, f_zp, f_zm):
    """Finite-difference smoothness across z: penalize curvature f(z±dz) − f(z)."""
    return ((f_zp - f_z) ** 2).mean() + ((f_zm - f_z) ** 2).mean()


# --------------------------------------------------------------------------- #
# Biology-informed constraints                                                 #
# --------------------------------------------------------------------------- #
def _knn_idx(xy, k):
    d = torch.cdist(xy, xy)
    d.fill_diagonal_(float("inf"))
    return d.topk(min(k, xy.shape[0] - 1), largest=False).indices     # (n, k)


def domain_coherence(type_prob, xy, k=8):
    """Nearby query points should share a spatial domain: penalize soft-type
    disagreement with spatial neighbours (a differentiable CRF-style smoothness)."""
    if xy.shape[0] < k + 1:
        return xy.new_zeros(())
    nn = _knn_idx(xy, k)
    neigh = type_prob[nn].mean(dim=1)                 # (n, C) mean neighbour type dist
    return ((type_prob - neigh) ** 2).sum(-1).mean()


def interface_preservation(type_prob, xy, target_M, k=10):
    """Match the neighbourhood-enrichment matrix P(neighbour=j | centre=i) of the
    predicted soft types to the flanking slices' interpolated ``target_M`` — so the
    interfaces between spatial domains (who borders whom) are preserved."""
    n, C = type_prob.shape
    if n < k + 1:
        return xy.new_zeros(())
    nn = _knn_idx(xy, k)
    neigh = type_prob[nn].mean(dim=1)                 # (n, C)
    # soft co-occurrence M[i,j] = Σ_cells p_i[i]·neigh[j], row-normalized
    M = type_prob.t() @ neigh                         # (C, C)
    M = M / (M.sum(dim=1, keepdim=True) + 1e-8)
    return ((M - target_M) ** 2).mean()


def grad_smoothness(expr, xy):
    """Within-domain expression gradient smoothness: penalize the spatial gradient of
    predicted expression (native microenvironment varies smoothly inside a domain).
    ``xy`` must require grad; uses autograd for ∂expr/∂xy."""
    g = torch.autograd.grad(expr.sum(), xy, create_graph=True, retain_graph=True)[0]
    return (g ** 2).mean()
