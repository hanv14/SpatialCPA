"""
Prediction Heads for SpatialCPA.

Three parallel heads that consume the spatial context vector:
  A) Cell-type classifier
  B) Region classifier
  C) Expression decoder (ZINB-based)
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


class ZINBExpressionDecoder(nn.Module):
    """
    Expression decoder using Zero-Inflated Negative Binomial distribution.

    Takes spatial context h(x,y,z) + cell type embedding and predicts
    ZINB parameters (mean, dispersion, dropout) for each gene.

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

        # ZINB parameters
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),  # mean must be positive
        )
        self.theta_head = nn.Sequential(
            nn.Linear(hidden_dim, n_genes),
            nn.Softplus(),  # dispersion must be positive
        )
        self.pi_head = nn.Linear(hidden_dim, n_genes)  # logits for dropout

    def forward(self, h, cell_type_idx):
        """
        Parameters
        ----------
        h : (N, input_dim) spatial context.
        cell_type_idx : (N,) integer cell type indices.

        Returns
        -------
        mu : (N, n_genes) mean expression.
        theta : (N, n_genes) dispersion.
        pi_logits : (N, n_genes) dropout logits.
        """
        ct_embed = self.cell_type_embedding(cell_type_idx)
        combined = torch.cat([h, ct_embed], dim=1)
        shared = self.shared(combined)

        mu = self.mu_head(shared) + 1e-6
        theta = self.theta_head(shared) + 1e-6
        pi_logits = self.pi_head(shared)

        return mu, theta, pi_logits

    def sample(self, h, cell_type_idx):
        """
        Sample expression profiles from the predicted ZINB distribution.

        Returns
        -------
        expr : (N, n_genes) sampled counts.
        """
        mu, theta, pi_logits = self.forward(h, cell_type_idx)
        return zinb_sample(mu, theta, pi_logits)

    def predict_mean(self, h, cell_type_idx):
        """Return the mean expression (no sampling)."""
        mu, theta, pi_logits = self.forward(h, cell_type_idx)
        pi = torch.sigmoid(pi_logits)
        return mu * (1 - pi)


def zinb_log_prob(x, mu, theta, pi_logits):
    """
    Log probability of Zero-Inflated Negative Binomial.

    Parameters
    ----------
    x : (N, G) observed counts.
    mu : (N, G) mean.
    theta : (N, G) dispersion (inverse overdispersion).
    pi_logits : (N, G) dropout logits.

    Returns
    -------
    log_prob : (N, G) log probability.
    """
    eps = 1e-8

    # Negative binomial component
    # NB parameterized by mean mu and dispersion theta
    # P(x | mu, theta) = Gamma(x + theta) / (Gamma(theta) * x!) *
    #                     (theta / (theta + mu))^theta * (mu / (theta + mu))^x
    log_theta_mu = torch.log(theta + mu + eps)
    nb_log_prob = (
        torch.lgamma(x + theta + eps)
        - torch.lgamma(theta + eps)
        - torch.lgamma(x + 1)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
    )

    # Zero-inflation
    # P(x=0) = pi + (1-pi) * NB(0)
    # P(x>0) = (1-pi) * NB(x)
    pi = torch.sigmoid(pi_logits)
    log_pi = F.logsigmoid(pi_logits)
    log_one_minus_pi = F.logsigmoid(-pi_logits)

    # For x == 0
    nb_zero = theta * (torch.log(theta + eps) - log_theta_mu)
    zero_case = torch.logsumexp(
        torch.stack([log_pi, log_one_minus_pi + nb_zero], dim=0), dim=0
    )

    # For x > 0
    nonzero_case = log_one_minus_pi + nb_log_prob

    # Select based on x
    is_zero = (x < 0.5).float()
    log_prob = is_zero * zero_case + (1 - is_zero) * nonzero_case

    return log_prob


def zinb_sample(mu, theta, pi_logits):
    """
    Sample from ZINB distribution.

    Parameters
    ----------
    mu, theta, pi_logits : (N, G) ZINB parameters.

    Returns
    -------
    samples : (N, G) sampled counts.
    """
    with torch.no_grad():
        # Sample dropout mask
        pi = torch.sigmoid(pi_logits)
        dropout_mask = torch.bernoulli(pi)

        # Sample from Negative Binomial via Gamma-Poisson mixture
        # rate = Gamma(shape=theta, rate=theta/mu)
        concentration = theta
        rate = theta / (mu + 1e-8)
        gamma_samples = torch.distributions.Gamma(concentration, rate).sample()
        poisson_samples = torch.poisson(gamma_samples)

        # Apply zero-inflation
        samples = (1 - dropout_mask) * poisson_samples

    return samples
