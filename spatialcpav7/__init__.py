"""SpatialCPA-v7 — foundation-anchored fused-transport histogenesis.

A training-free (inference-time) generator that synthesizes a held-out tissue
section from its two flanking sections and a scalar target z, by:

1. embedding cells in a foundation-model cell-state space (local PCA / coexpr
   fallback) with cross-slice mutual-NN anchoring, :mod:`spatialcpav7.embedding`;
2. placing cells via **fused Gromov-Wasserstein** displacement interpolation,
   which preserves the intra-slice neighbourhood graph across the morph,
   :mod:`spatialcpav7.transport`;
3. annotating them with an FM cell-state prior constrained to the interpolated
   composition and to **2D + 3D cell-cell communication** (in-plane niche plus a
   cross-slice z-stacking niche and a ligand-receptor flux prior),
   :mod:`spatialcpav7.annotation` / :mod:`spatialcpav7.communication`;
4. transferring real expression profiles (endpoint copy by default).

See ``README.md`` for the method and its benchmark integration.
"""

from .config import (
    SpatialCPAv7Config, EmbeddingConfig, TransportConfig,
    CommunicationConfig, AnnotationConfig, SynthesisConfig,
)
from .data import Slice, SliceStack
from .embedding import build_embedder, register_embedder, Embedder, mutual_nn_align
from .generator import SpatialCPAv7, VirtualSlice

__all__ = [
    "SpatialCPAv7Config", "EmbeddingConfig", "TransportConfig",
    "CommunicationConfig", "AnnotationConfig", "SynthesisConfig",
    "Slice", "SliceStack",
    "build_embedder", "register_embedder", "Embedder", "mutual_nn_align",
    "SpatialCPAv7", "VirtualSlice",
]

__version__ = "7.0.0"   # fused-GW placement + 2D/3D cell-cell-communication annotation
