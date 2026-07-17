"""
Configuration for SpatialCPA-v14 — the **H3D-FLA** virtual-slice generator
(Hybrid 3D Flow-matching Latent Atlas).

v14 implements the ``H3D_FLA_Pipeline.md`` proposal end-to-end. The engine is
**conditional flow matching in a joint molecular-morphological latent space**,
conditioned on a **3D positional-attention context** aggregated over the real slices,
trained **gap-aware** with **z-marginalization** and **biology-informed regularizers**
(interface coherence, adaptive smoothness, closed-loop consistency). Virtual slices are
produced by integrating the conditional ODE from noise and decoding the generated joint
latents into expression + cell types, grounded in the real training biology.

This is a paradigm **completely distinct from v8** (no optimal transport / barycentric
morph / OT fusion / niche Markov-random-field — v8 is training-free numpy/scipy) and from
v13 (no cell-sentence tokenizer / gene-language-model / retrieval softmax). The parts
below map one-to-one to the pipeline stages; every default is the intended production
setting so that running the wrapper with no flags reproduces the proposed pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatentConfig:
    """Stage 1 — expression latent + morphological (pseudo-image) channels."""
    expr_latent_dim: int = 32       # d_e: expression latent (PCA on HVGs / normalized)
    morph_k: int = 12               # kNN for per-cell pseudo-image (soft cell-type + density) channels
    density_sigma: float = 1.0      # density kernel bandwidth in units of median spacing


@dataclass
class EncoderConfig:
    """Stage 2 — joint molecular-morphological encoder (fusion MLP)."""
    joint_dim: int = 48             # d: unified joint latent h
    hidden: int = 128
    dropout: float = 0.05


@dataclass
class AttnConfig:
    """Stage 3.1 — 3D positional-attention context module."""
    d_model: int = 96
    n_heads: int = 4
    n_context: int = 16             # local flanking cells attended per query (local context)
    n_global_tokens: int = 1        # per-slice summary tokens (long-range 3D context)
    fourier_bands: int = 6          # Fourier bands for (x, y, z) positional encoding
    dropout: float = 0.05


@dataclass
class FlowConfig:
    """Stage 3.2 — conditional flow-matching vector field (MLP backbone)."""
    hidden: int = 192
    n_layers: int = 4
    time_embed_dim: int = 32
    sigma_min: float = 1e-3         # OT straight-line path floor
    n_ode_steps: int = 12           # Euler steps for the sampling ODE
    n_ensemble: int = 4             # initial noises marginalized per query (uncertainty + denoise)


@dataclass
class BioConfig:
    """Stage 4 — biology-informed regularization (TME focus)."""
    w_interface: float = 0.10       # edge-aware interface coherence (adaptive smoothness)
    w_hypoxia: float = 0.05         # core->periphery gradient directionality (soft margin)
    w_consistency: float = 0.10     # closed-loop encode(decode(h)) cycle consistency
    w_smooth: float = 0.05          # latent total-variation, relaxed at morphological edges
    hypoxia_margin: float = 0.02
    anneal_epochs: int = 40         # ramp bio weights in over this many flow epochs


@dataclass
class TrainConfig:
    """Stage 6 — two-phase training strategy."""
    pretrain_epochs: int = 60       # Phase A: joint encoder + decoders (reconstruction)
    epochs: int = 160               # Phase B: flow field + attention context (CFM + bio)
    batch_cells: int = 256
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 2.0
    gap_dropout: float = 0.35       # prob. of dropping a whole context slice (gap-aware)
    z_sigma: float = 0.15           # z-perturbation for marginalization (units of z-half-range)
    device: str = "auto"            # "auto" | "cpu" | "cuda"
    seed: int = 42
    verbose: bool = True
    fallback_on_error: bool = True


@dataclass
class GenerationConfig:
    """Stage 5 — inference for arbitrary / missing z (virtual slice generation)."""
    ground_expression: bool = True  # emit a real profile (preserve gene-gene covariance)
    ground_mode: str = "anchor"     # "anchor" (resampled cell's own real profile — coherent) | "latent"
    ground_k: int = 8               # candidate real cells per query for latent grounding
    ground_temp: float = 0.25       # softmax temperature over latent similarity (RAG-style)
    ground_blend_flow: float = 0.20  # (anchor mode) prob. a cell is re-grounded to the flow-latent pick
    edit_weight: float = 0.25       # blend toward the flow-decoded profile (0 = pure real exemplar)
    output_counts: bool = True      # emit count-like (expm1) expression for the evaluator
    composition_match: bool = True  # match cell-type composition to the interpolated flanking mix
    # Position layout for the generated sheet. "morph" resamples the flanking supports in
    # the z-interpolated ratio then applies the flow-decoded continuous displacement field
    # (a learned deformation — NOT optimal transport); "flanking" uses the resample only;
    # "nearest" copies the single nearest slice's layout (near-identical planes).
    position_mode: str = "flanking"     # "flanking" | "morph" | "nearest"
    displacement_scale: float = 0.5     # scale on the flow-decoded displacement field
    near_identical_ratio: float = 0.60  # cross/within spacing threshold for auto "nearest"


@dataclass
class SpatialCPAv14Config:
    latent: LatentConfig = field(default_factory=LatentConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    attn: AttnConfig = field(default_factory=AttnConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    bio: BioConfig = field(default_factory=BioConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42
