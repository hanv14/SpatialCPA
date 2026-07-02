"""
Prediction Heads for SpatialCPA.

Three parallel heads that consume the spatial context vector:
  A) Cell-type classifier
  B) Region classifier
  C) Expression decoder — both ZINB and direct MSE variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierHead(nn.Module):
    """
    Softmax classifier head for cell type or region prediction.

    h(x,y,z) → hidden → softmax → P(label | x,y,z)
    """

    def __init__(self, input_dim, n_classes, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, h):
        """Returns logits (N, n_classes)."""
        return self.net(h)

    def predict_proba(self, h):
        """Returns probabilities (N, n_classes)."""
        return F.softmax(self.forward(h), dim=-1)


class DirectExpressionDecoder(nn.Module):
    """
    Direct expression prediction decoder (MSE-based).

    Takes spatial context h(x,y,z) + cell type embedding and directly
    predicts normalized expression values. Suitable for pre-normalized
    continuous expression data (not raw counts).

    Parameters
    ----------
    input_dim : int
        Dimension of spatial context vector.
    n_genes : int
        Number of genes to predict.
    n_cell_types : int
        Number of cell types for conditioning.
    cell_type_embed_dim : int
        Dimension of cell type embedding.
    hidden_dim : int
        Hidden layer width.
    """

    def __init__(self, input_dim, n_genes, n_cell_types, cell_type_embed_dim=64,
                 hidden_dim=512):
        super().__init__()
        self.n_genes = n_genes
        self.cell_type_embedding = nn.Embedding(n_cell_types, cell_type_embed_dim)

        combined_dim = input_dim + cell_type_embed_dim

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.output_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, h, cell_type_idx):
        """
        Parameters
        ----------
        h : (N, input_dim) spatial context.
        cell_type_idx : (N,) integer cell type indices.

        Returns
        -------
        predicted_expr : (N, n_genes) predicted expression values.
        """
        ct_embed = self.cell_type_embedding(cell_type_idx)
        combined = torch.cat([h, ct_embed], dim=1)
        shared = self.shared(combined)
        return self.output_head(shared)


class ZINBExpressionDecoder(nn.Module):
    """
    Expression decoder using Zero-Inflated Negative Binomial distribution.
    Use only when input data consists of raw integer counts.
    """

    def __init__(self, input_dim, n_genes, n_cell_types, cell_type_embed_dim=64,
                 hidden_dim=512):
        super().__init__()
        self.n_genes = n_genes
        self.cell_type_embedding = nn.Embedding(n_cell_types, cell_type_embed_dim)

        combined_dim = input_dim + cell_type_embed_dim

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),
        )
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),
        )
        self.pi_head = nn.Linear(hidden_dim, n_genes)

    def forward(self, h, cell_type_idx):
        ct_embed = self.cell_type_embedding(cell_type_idx)
        combined = torch.cat([h, ct_embed], dim=1)
        shared = self.shared(combined)

        mu = self.mu_head(shared) + 1e-6
        theta = self.theta_head(shared) + 1e-6
        pi_logits = self.pi_head(shared)

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

    pi = torch.sigmoid(pi_logits)
    log_pi = F.logsigmoid(pi_logits)
    log_one_minus_pi = F.logsigmoid(-pi_logits)

    nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
    zero_case = torch.logsumexp(
        torch.stack([log_pi, log_one_minus_pi + nb_zero], dim=0), dim=0
    )
    nonzero_case = log_one_minus_pi + nb_log_prob

    is_zero = (x < 0.5).float()
    log_prob = is_zero * zero_case + (1 - is_zero) * nonzero_case

    return log_prob
