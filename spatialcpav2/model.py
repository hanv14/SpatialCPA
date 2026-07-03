"""
SpatialCPA v2 — main model class.

Assembles the calibrated positional encoder, the gated spatial backbone, and
the FiLM-conditioned heads into a single end-to-end differentiable neural field
that models cell type and gene expression jointly at any (x, y, z).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spatialcpav2.fourier import FourierFeatureEncoder
from spatialcpav2.backbone import SpatialBackbone
from spatialcpav2.heads import ClassifierHead, FiLMExpressionDecoder, ZINBExpressionDecoder


class SpatialCPAv2(nn.Module):
    """
    Continuous 3D spatial-transcriptomics neural field (v2).

    Parameters mirror v1 where possible; new knobs (``n_rff``, ``decoder_layers``,
    ``xy_extent``/``z_extent``) are optional and data-adaptive.
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
        xy_extent=None,
        z_extent=None,
        n_rff=96,
        backbone_hidden=512,
        backbone_output=256,
        backbone_layers=8,
        decoder_layers=3,
        cell_type_embed_dim=64,
        dropout=0.1,
        use_zinb=False,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_cell_types = n_cell_types
        self.n_regions = n_regions
        self.use_zinb = use_zinb

        self.fourier = FourierFeatureEncoder(
            n_freq_xy=n_freq_xy, n_freq_z=n_freq_z,
            xy_scale=xy_scale, z_scale=z_scale,
            xy_extent=xy_extent, z_extent=z_extent, n_rff=n_rff,
        )

        self.backbone = SpatialBackbone(
            input_dim=self.fourier.output_dim,
            hidden_dim=backbone_hidden,
            output_dim=backbone_output,
            n_layers=backbone_layers,
            dropout=dropout,
        )

        self.cell_type_head = ClassifierHead(
            input_dim=backbone_output, n_classes=n_cell_types,
            hidden_dim=backbone_output, dropout=dropout,
        )

        self.region_head = None
        if n_regions is not None and n_regions > 0:
            self.region_head = ClassifierHead(
                input_dim=backbone_output, n_classes=n_regions,
                hidden_dim=backbone_output, dropout=dropout,
            )

        decoder_cls = ZINBExpressionDecoder if use_zinb else FiLMExpressionDecoder
        self.expression_decoder = decoder_cls(
            input_dim=backbone_output, n_genes=n_genes, n_cell_types=n_cell_types,
            cell_type_embed_dim=cell_type_embed_dim, hidden_dim=backbone_hidden,
            n_layers=decoder_layers, dropout=dropout,
        )

    def encode_coords(self, coords):
        return self.backbone(self.fourier(coords))

    def forward(self, coords, cell_type_idx=None):
        h = self.encode_coords(coords)
        ct_logits = self.cell_type_head(h)

        region_logits = self.region_head(h) if self.region_head is not None else None

        if cell_type_idx is None:
            cell_type_idx = ct_logits.argmax(dim=-1)

        result = {"h": h, "cell_type_logits": ct_logits, "region_logits": region_logits}

        if self.use_zinb:
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            result.update(mu=mu, theta=theta, pi_logits=pi_logits)
        else:
            result["predicted_expr"] = self.expression_decoder(h, cell_type_idx)
        return result

    def predict_cell_type(self, coords):
        h = self.encode_coords(coords)
        return self.cell_type_head.predict_proba(h)

    def predict_expression(self, coords, cell_type_idx):
        h = self.encode_coords(coords)
        if self.use_zinb:
            mu, theta, pi_logits = self.expression_decoder(h, cell_type_idx)
            return mu * (1 - torch.sigmoid(pi_logits))
        return self.expression_decoder(h, cell_type_idx)

    @torch.no_grad()
    def predict_expression_marginal(self, coords, ct_probs=None, top_k=3):
        """Posterior-weighted expression prediction (MSE mode only)."""
        h = self.encode_coords(coords)
        if ct_probs is None:
            ct_probs = self.cell_type_head.predict_proba(h)
        if self.use_zinb:
            # marginalisation not defined for ZINB sampling path; fall back to argmax
            idx = ct_probs.argmax(dim=-1)
            mu, theta, pi_logits = self.expression_decoder(h, idx)
            return mu * (1 - torch.sigmoid(pi_logits))
        return self.expression_decoder.forward_marginal(h, ct_probs, top_k=top_k)
