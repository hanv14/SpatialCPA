"""
SpatialCPA — main model class.

Assembles the Fourier encoder, spatial backbone, and prediction heads
into a single end-to-end differentiable model.
"""

import torch
import torch.nn as nn
import numpy as np

from spatialcpa.fourier import FourierFeatureEncoder
from spatialcpa.backbone import SpatialBackbone
from spatialcpa.heads import ClassifierHead, ZINBExpressionDecoder


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
        Characteristic xy spatial scale (µm).
    z_scale : float
        Characteristic z spatial scale (µm).
    backbone_hidden : int
        Backbone hidden layer width.
    backbone_output : int
        Backbone output (spatial context) dimension.
    backbone_layers : int
        Number of backbone layers.
    dropout : float
        Dropout rate.
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
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_regions = n_regions

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

        # Head C: Expression decoder (ZINB)
        self.expression_decoder = ZINBExpressionDecoder(
            input_dim=backbone_output,
            n_genes=n_genes,
            n_cell_types=n_cell_types,
            cell_type_embed_dim=64,
            hidden_dim=backbone_hidden,
        )

    def encode_coords(self, coords):
        """
        Encode 3D coordinates → spatial context vector.

        Parameters
        ----------
        coords : (N, 3) tensor of (x, y, z).

        Returns
        -------
        h : (N, backbone_output) spatial context.
        """
        ff = self.fourier(coords)
        return self.backbone(ff)

    def forward(self, coords, cell_type_idx=None):
        """
        Full forward pass.

        Parameters
        ----------
        coords : (N, 3) tensor of (x, y, z).
        cell_type_idx : (N,) integer tensor of cell type indices.
                        If None, uses argmax of cell type predictions.

        Returns
        -------
        dict with keys:
            'h' : (N, D) spatial context
            'cell_type_logits' : (N, n_cell_types)
            'region_logits' : (N, n_regions) or None
            'mu' : (N, n_genes) ZINB mean
            'theta' : (N, n_genes) ZINB dispersion
            'pi_logits' : (N, n_genes) ZINB dropout logits
        """
        h = self.encode_coords(coords)

        ct_logits = self.cell_type_head(h)

        region_logits = None
        if self.region_head is not None:
            region_logits = self.region_head(h)

        # Use provided cell types or predict them
        if cell_type_idx is None:
            cell_type_idx = ct_logits.argmax(dim=-1)

        mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)

        return {
            'h': h,
            'cell_type_logits': ct_logits,
            'region_logits': region_logits,
            'mu': mu,
            'theta': theta,
            'pi_logits': pi_logits,
        }

    def predict_cell_type(self, coords):
        """Return cell type probabilities at given coordinates."""
        h = self.encode_coords(coords)
        return self.cell_type_head.predict_proba(h)

    def predict_expression(self, coords, cell_type_idx, sample=True):
        """
        Predict gene expression at given coordinates for given cell types.

        Parameters
        ----------
        coords : (N, 3) coordinates.
        cell_type_idx : (N,) cell type indices.
        sample : bool
            If True, sample from ZINB. If False, return mean.

        Returns
        -------
        expr : (N, n_genes)
        """
        h = self.encode_coords(coords)
        if sample:
            return self.expression_decoder.sample(h, cell_type_idx)
        else:
            return self.expression_decoder.predict_mean(h, cell_type_idx)
