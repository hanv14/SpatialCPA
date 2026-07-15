"""
SpatialCPA-v12 — generative continuous 3D virtual-slice generation for aligned serial
spatial transcriptomics with no paired H&E.

v12 enhances v11's two implicit coordinate-network fields, queryable at arbitrary
continuous (x, y, z):

  Stage 1  LayoutField          : neighbouring real slices + Fourier(z) -> positions + type
  Stage 2  GenerativeExpressionField : (x, y, z) + Stage-1 layout code   -> sampled expression

Stage 1 is trained by knowledge distillation from a frozen multimodal foundation-model
teacher (OmiCLIP / Path2Space) plus self-supervised reconstruction of real slices;
Stage 2 is a conditional **factor-analysis decoder** trained by a mean-reconstruction +
covariance likelihood, so generated cells carry realistic gene-gene covariance and
per-gene variance. A spatially-coherent latent, leakage-safe field calibration, and
prior-corrected composition sharpen the sampled slice. Falls back to a nearest-slice
layout if PyTorch is unavailable.

    from spatialcpav12 import SpatialCPAv12, SpatialCPAv12Config, Slice, SliceStack
    gen = SpatialCPAv12(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)     # continuous z
"""

from .config import (
    SpatialCPAv12Config, FourierConfig, ContextConfig, LayoutConfig,
    ExpressionConfig, TeacherConfig, LossConfig, TrainConfig, InferenceConfig,
)
from .data import Slice, SliceStack
from .model import SpatialCPAv12, VirtualSlice

__version__ = "12.0.0"

__all__ = [
    "SpatialCPAv12", "VirtualSlice", "SpatialCPAv12Config", "FourierConfig",
    "ContextConfig", "LayoutConfig", "ExpressionConfig", "TeacherConfig",
    "LossConfig", "TrainConfig", "InferenceConfig", "Slice", "SliceStack",
]
