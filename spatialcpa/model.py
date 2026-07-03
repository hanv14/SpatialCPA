"""
SpatialCPA — main model class.

Assembles the Fourier encoder, spatial backbone, and prediction heads
into a single end-to-end differentiable model.
"""

import torch
import torch.nn as nn

from spatialcpa.fourier import FourierFeatureEncoder
from spatialcpa.backbone import SpatialBackbone
from spatialcpa.heads import ClassifierHead, DirectExpressionDecoder, ZINBExpressionDecoder


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
        If True, use ZINB decoder (for raw count data).
        If False, use direct MSE decoder (for normalized data).
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
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_regions = n_regions
        self.use_zinb = use_zinb

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
        if use_zinb:
            self.expression_decoder = ZINBExpressionDecoder(
                input_dim=backbone_output,
                n_genes=n_genes,
                n_cell_types=n_cell_types,
                cell_type_embed_dim=64,
                hidden_dim=backbone_hidden,
            )
        else:
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

        if self.use_zinb:
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            result['mu'] = mu
            result['theta'] = theta
            result['pi_logits'] = pi_logits
        else:
            predicted_expr = self.expression_decoder(h, cell_type_idx)
            result['predicted_expr'] = predicted_expr

        return result

    def predict_cell_type(self, coords):
        """Return cell type probabilities at given coordinates."""
        h = self.encode_coords(coords)
        return self.cell_type_head.predict_proba(h)

    def predict_expression(self, coords, cell_type_idx):
        """Predict gene expression at given coordinates."""
        h = self.encode_coords(coords)
        if self.use_zinb:
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            pi = torch.sigmoid(pi_logits)
            return mu * (1 - pi)
        else:
            return self.expression_decoder(h, cell_type_idx)
