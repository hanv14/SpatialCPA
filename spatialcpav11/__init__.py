"""
SpatialCPA-v11 — two-stage continuous 3D virtual-slice generation for aligned serial
spatial transcriptomics with no paired H&E.

Two implicit coordinate-network fields, queryable at arbitrary continuous (x, y, z):

  Stage 1  LayoutField      : neighbouring real slices + Fourier(z) -> positions + type
  Stage 2  ExpressionField  : (x, y, z) + Stage-1 layout code       -> gene expression

Stage 1 is trained by knowledge distillation from a frozen multimodal foundation-model
teacher (OmiCLIP / Path2Space) plus self-supervised reconstruction of real slices;
Stage 2 by expression reconstruction. Cross-z consistency and biology-informed
constraints regularize both. If PyTorch is unavailable it falls back to a nearest-slice
layout.

    from spatialcpav11 import SpatialCPAv11, SpatialCPAv11Config, Slice, SliceStack
    gen = SpatialCPAv11(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)     # continuous z
"""

from .config import (
    SpatialCPAv11Config, FourierConfig, ContextConfig, LayoutConfig,
    ExpressionConfig, TeacherConfig, LossConfig, TrainConfig, InferenceConfig,
)
from .data import Slice, SliceStack
from .model import SpatialCPAv11, VirtualSlice

__version__ = "11.0.0"

__all__ = [
    "SpatialCPAv11", "VirtualSlice", "SpatialCPAv11Config", "FourierConfig",
    "ContextConfig", "LayoutConfig", "ExpressionConfig", "TeacherConfig",
    "LossConfig", "TrainConfig", "InferenceConfig", "Slice", "SliceStack",
]
