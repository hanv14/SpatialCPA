"""
SpatialCPA — main model class.

Assembles the Fourier encoder, spatial backbone, and prediction heads
into a single end-to-end differentiable model.
"""

import torch
import torch.nn as nn

from spatialcpa.fourier import FourierFeatureEncoder
from spatialcpa.backbone import SpatialBackbone
from spatialcpa.heads import (
    ClassifierHead,
    DirectExpressionDecoder,
    GaussianExpressionDecoder,
    ZINBExpressionDecoder,
)


class SpatialCPA(nn.Module):
    """
    SpatialCPA: Continuous 3D spatial transcriptomics model.

    Parameters
    ----------
    n_genes : int
        Number of genes.
    n_cell_types : int
        Number of cell types.
    n_regions : int or None
        Number of region labels. If None, region head is disabled.
    n_freq_xy : int
        Fourier frequencies for x, y axes.
    n_freq_z : int
        Fourier frequencies for z axis.
    xy_scale : float
        Characteristic xy spatial scale.
    z_scale : float
        Characteristic z spatial scale.
    backbone_hidden : int
        Backbone hidden layer width.
    backbone_output : int
        Backbone output (spatial context) dimension.
    backbone_layers : int
        Number of backbone layers.
    dropout : float
        Dropout rate.
    use_zinb : bool
        Backward-compatible flag. If True, use ZINB decoder (for raw count
        data). If False, use the decoder selected by ``expression_mode``.
    expression_mode : str or None
        Expression decoder / likelihood to use:
          - ``'mse'``      deterministic regression (v1/v2 behaviour).
          - ``'gaussian'`` generative Gaussian head (mean + variance) — enables
                           stochastic expression sampling for normalized data
                           (recommended for v3 virtual-slice generation).
          - ``'zinb'``     zero-inflated negative binomial (raw counts).
        If None, it is derived from ``use_zinb`` for backward compatibility
        (``'zinb'`` when ``use_zinb`` else ``'mse'``). When set explicitly, it
        takes precedence over ``use_zinb``.
    """

    def __init__(
        self,
        n_genes,
        n_cell_types,
        n_regions=None,
        n_freq_xy=48,
        n_freq_z=32,
        xy_scale=10.0,
        z_scale=100.0,
        backbone_hidden=512,
        backbone_output=256,
        backbone_layers=8,
        dropout=0.1,
        use_zinb=False,
        expression_mode=None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_regions = n_regions

        if expression_mode is None:
            expression_mode = 'zinb' if use_zinb else 'mse'
        if expression_mode not in ('mse', 'gaussian', 'zinb'):
            raise ValueError(
                f"expression_mode must be 'mse', 'gaussian' or 'zinb', "
                f"got {expression_mode!r}"
            )
        self.expression_mode = expression_mode
        # Keep the boolean around for code paths that still branch on it.
        self.use_zinb = (expression_mode == 'zinb')

        # Fourier feature encoder
        self.fourier = FourierFeatureEncoder(
            n_freq_xy=n_freq_xy,
            n_freq_z=n_freq_z,
            xy_scale=xy_scale,
            z_scale=z_scale,
        )

        # Spatial backbone
        self.backbone = SpatialBackbone(
            input_dim=self.fourier.output_dim,
            hidden_dim=backbone_hidden,
            output_dim=backbone_output,
            n_layers=backbone_layers,
            dropout=dropout,
        )

        # Head A: Cell-type classifier
        self.cell_type_head = ClassifierHead(
            input_dim=backbone_output,
            n_classes=n_cell_types,
            hidden_dim=backbone_output,
        )

        # Head B: Region classifier (optional)
        self.region_head = None
        if n_regions is not None and n_regions > 0:
            self.region_head = ClassifierHead(
                input_dim=backbone_output,
                n_classes=n_regions,
                hidden_dim=backbone_output,
            )

        # Head C: Expression decoder
        if self.expression_mode == 'zinb':
            self.expression_decoder = ZINBExpressionDecoder(
                input_dim=backbone_output,
                n_genes=n_genes,
                n_cell_types=n_cell_types,
                cell_type_embed_dim=64,
                hidden_dim=backbone_hidden,
            )
        elif self.expression_mode == 'gaussian':
            self.expression_decoder = GaussianExpressionDecoder(
                input_dim=backbone_output,
                n_genes=n_genes,
                n_cell_types=n_cell_types,
                cell_type_embed_dim=64,
                hidden_dim=backbone_hidden,
            )
        else:  # 'mse'
            self.expression_decoder = DirectExpressionDecoder(
                input_dim=backbone_output,
                n_genes=n_genes,
                n_cell_types=n_cell_types,
                cell_type_embed_dim=64,
                hidden_dim=backbone_hidden,
            )

    def encode_coords(self, coords):
        """Encode 3D coordinates → spatial context vector."""
        ff = self.fourier(coords)
        return self.backbone(ff)

    def forward(self, coords, cell_type_idx=None):
        """
        Full forward pass.

        Returns dict with 'h', 'cell_type_logits', 'region_logits',
        and either 'predicted_expr' (MSE mode) or 'mu'/'theta'/'pi_logits' (ZINB mode).
        """
        h = self.encode_coords(coords)
        ct_logits = self.cell_type_head(h)

        region_logits = None
        if self.region_head is not None:
            region_logits = self.region_head(h)

        if cell_type_idx is None:
            cell_type_idx = ct_logits.argmax(dim=-1)

        result = {
            'h': h,
            'cell_type_logits': ct_logits,
            'region_logits': region_logits,
        }

        if self.expression_mode == 'zinb':
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            result['mu'] = mu
            result['theta'] = theta
            result['pi_logits'] = pi_logits
        elif self.expression_mode == 'gaussian':
            expr_mu, expr_logvar = self.expression_decoder(h, cell_type_idx)
            result['expr_mu'] = expr_mu
            result['expr_logvar'] = expr_logvar
            result['predicted_expr'] = expr_mu
        else:  # 'mse'
            predicted_expr = self.expression_decoder(h, cell_type_idx)
            result['predicted_expr'] = predicted_expr

        return result

    def predict_cell_type(self, coords):
        """Return cell type probabilities at given coordinates."""
        h = self.encode_coords(coords)
        return self.cell_type_head.predict_proba(h)

    def predict_expression(self, coords, cell_type_idx):
        """Predict the *mean* gene expression at given coordinates."""
        h = self.encode_coords(coords)
        if self.expression_mode == 'zinb':
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            pi = torch.sigmoid(pi_logits)
            return mu * (1 - pi)
        elif self.expression_mode == 'gaussian':
            mu, _ = self.expression_decoder(h, cell_type_idx)
            return mu
        else:
            return self.expression_decoder(h, cell_type_idx)

    def sample_expression(self, coords, cell_type_idx, temperature=1.0,
                          generator=None):
        """
        Draw a stochastic (generative) expression sample at given coordinates.

        For ``gaussian`` mode this samples from the learned per-gene Normal; for
        ``zinb`` mode it samples from the zero-inflated negative binomial; for
        ``mse`` mode there is no learned noise model, so the deterministic mean
        is returned (equivalent to ``predict_expression``).

        Parameters
        ----------
        temperature : float
            Scales sampling noise. 0.0 → deterministic mean; 1.0 → full learned
            variance.
        generator : torch.Generator or None
            Optional RNG for reproducibility.
        """
        h = self.encode_coords(coords)
        if self.expression_mode == 'gaussian':
            return self.expression_decoder.sample(
                h, cell_type_idx, temperature=temperature, generator=generator)
        elif self.expression_mode == 'zinb':
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            if temperature <= 0.0:
                pi = torch.sigmoid(pi_logits)
                return mu * (1 - pi)
            # Sample NB via Gamma-Poisson mixture, then apply dropout mask.
            theta_c = theta.clamp(min=1e-4)
            gamma = torch.distributions.Gamma(
                concentration=theta_c,
                rate=(theta_c / mu.clamp(min=1e-8))).sample()
            counts = torch.poisson(gamma.clamp(max=1e6))
            pi = torch.sigmoid(pi_logits)
            drop = (torch.rand(pi.shape, device=pi.device) < pi).float()
            return counts * (1.0 - drop)
        else:  # 'mse' — no noise model
            return self.expression_decoder(h, cell_type_idx)
