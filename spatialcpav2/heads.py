"""
Prediction heads for SpatialCPA v2.

The central design goal of v2 is to model the *joint* spatial–cell-type–
expression relationship coherently. Two changes make this concrete:

1. **FiLM conditioning.** Instead of concatenating a cell-type embedding to the
   spatial context (v1), the cell type produces per-layer feature-wise affine
   modulations (gamma, beta) of the expression trunk. This lets the cell type
   *reshape* the entire expression computation rather than nudging its input,
   which is a much deeper form of conditioning.

2. **Marginalised expression.** The decoder can be evaluated against the full
   cell-type posterior P(c | x, y, z) so the predicted expression is the
   posterior-weighted mixture over programs, keeping the cell-type head and the
   expression head consistent with one another (a cell type the classifier
   thinks is unlikely cannot dominate the expression it emits).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """Softmax classifier: h(x,y,z) -> logits over labels."""

    def __init__(self, input_dim, n_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, h):
        return self.net(h)

    def predict_proba(self, h):
        return F.softmax(self.forward(h), dim=-1)


class FiLMExpressionDecoder(nn.Module):
    """
    Cell-type-conditioned expression decoder with FiLM modulation (MSE mode).

    Parameters
    ----------
    input_dim : int
        Spatial-context dimensionality.
    n_genes : int
    n_cell_types : int
    cell_type_embed_dim : int
    hidden_dim : int
    n_layers : int
        Number of FiLM-modulated residual blocks in the shared trunk.
    """

    def __init__(self, input_dim, n_genes, n_cell_types, cell_type_embed_dim=64,
                 hidden_dim=512, n_layers=3, dropout=0.05):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.cell_type_embedding = nn.Embedding(n_cell_types, cell_type_embed_dim)
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Per-block FiLM generators: embedding -> (gamma, beta) for each block.
        self.film = nn.ModuleList([
            nn.Linear(cell_type_embed_dim, 2 * hidden_dim) for _ in range(n_layers)
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        # Per-cell-type gene bias captures the mean program of each type
        # (so the trunk only has to learn spatial modulation of it).
        self.type_gene_bias = nn.Embedding(n_cell_types, n_genes)
        nn.init.zeros_(self.type_gene_bias.weight)
        self.out = nn.Linear(hidden_dim, n_genes)

    def _trunk(self, h, ct_embed):
        x = self.in_proj(h)
        for norm, film, block in zip(self.norms, self.film, self.blocks):
            gamma, beta = film(ct_embed).chunk(2, dim=-1)
            xn = norm(x)
            xn = (1.0 + gamma) * xn + beta          # FiLM modulation
            x = x + block(xn)
        return x

    def forward(self, h, cell_type_idx):
        """Hard-conditioned prediction. Returns (N, n_genes)."""
        ct_embed = self.cell_type_embedding(cell_type_idx)
        x = self._trunk(h, ct_embed)
        return self.out(x) + self.type_gene_bias(cell_type_idx)

    def forward_marginal(self, h, ct_probs, top_k=None):
        """
        Posterior-weighted expression: sum_c P(c|x) * decode(h, c).

        Parameters
        ----------
        h : (N, input_dim)
        ct_probs : (N, n_cell_types) posterior over cell types.
        top_k : int or None
            If set, only marginalise over the top-k types per cell (speed).
        """
        N = h.shape[0]
        C = self.n_cell_types
        if top_k is not None and top_k < C:
            topp, topc = ct_probs.topk(top_k, dim=-1)   # (N, k)
            topp = topp / topp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            out = h.new_zeros(N, self.n_genes)
            for j in range(top_k):
                idx = topc[:, j]
                out = out + topp[:, j:j + 1] * self.forward(h, idx)
            return out
        # full marginalisation
        out = h.new_zeros(N, self.n_genes)
        for c in range(C):
            idx = h.new_full((N,), c, dtype=torch.long)
            out = out + ct_probs[:, c:c + 1] * self.forward(h, idx)
        return out


class ZINBExpressionDecoder(nn.Module):
    """FiLM-conditioned ZINB decoder (raw-count mode)."""

    def __init__(self, input_dim, n_genes, n_cell_types, cell_type_embed_dim=64,
                 hidden_dim=512, n_layers=3, dropout=0.05):
        super().__init__()
        self.core = FiLMExpressionDecoder(
            input_dim, hidden_dim, n_cell_types, cell_type_embed_dim,
            hidden_dim, n_layers, dropout)
        # reuse the trunk; separate output heads
        self.n_genes = n_genes
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, n_genes), nn.Softplus())
        self.theta_head = nn.Sequential(nn.Linear(hidden_dim, n_genes), nn.Softplus())
        self.pi_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, h, cell_type_idx):
        ct_embed = self.core.cell_type_embedding(cell_type_idx)
        x = self.core._trunk(h, ct_embed)
        mu = self.mu_head(x) + 1e-6
        theta = self.theta_head(x) + 1e-6
        pi_logits = self.pi_head(x)
        return mu, theta, pi_logits


def zinb_log_prob(x, mu, theta, pi_logits):
    """Log probability of Zero-Inflated Negative Binomial."""
    eps = 1e-8
    log_theta_mu = torch.log(theta + mu + eps)
    nb_log_prob = (
        torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )
    log_pi = F.logsigmoid(pi_logits)
    log_one_minus_pi = F.logsigmoid(-pi_logits)
    nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
    zero_case = torch.logsumexp(
        torch.stack([log_pi, log_one_minus_pi + nb_zero], dim=0), dim=0)
    nonzero_case = log_one_minus_pi + nb_log_prob
    is_zero = (x < 0.5).float()
    return is_zero * zero_case + (1 - is_zero) * nonzero_case
