"""
SpatialCPA-v17 — a fully-generative latent flow-matching atlas.

v17 replaces v14's linear PCA expression code with a learned **variational autoencoder**
(VAE) carrying a **negative-binomial (count-aware) decoder** (scVI-style). The joint
molecular-morphological latent is KL-regularized, so a conditional flow-matching model
trained in that latent space produces samples on the decoder's manifold that can be
**decoded directly into gene expression** — no retrieval (retrieval kept as an ablation).

Two-phase training: (A) pretrain the VAE, freeze; (B) train the flow in the frozen latent.

    from spatialcpav17 import SpatialCPAv17, SpatialCPAv17Config, Slice, SliceStack
    gen = SpatialCPAv17(stack, gene_names=genes, cell_type_names=types, is_counts=True)
    vs = gen.generate_virtual_slice(z=target_z)
"""

from .config import (
    SpatialCPAv17Config, LatentConfig, VAEConfig, AttnConfig, FlowConfig,
    TrainConfig, GenerationConfig,
)
from .data import Slice, SliceStack
from .latents import morphology_features
from .model import SpatialCPAv17, VirtualSlice

__version__ = "17.0.0"

__all__ = [
    "SpatialCPAv17", "VirtualSlice", "SpatialCPAv17Config", "LatentConfig",
    "VAEConfig", "AttnConfig", "FlowConfig", "TrainConfig", "GenerationConfig",
    "Slice", "SliceStack", "morphology_features",
]
