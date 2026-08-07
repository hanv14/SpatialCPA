"""
Configuration for SpatialCPA-v17 — a fully-generative latent flow-matching atlas.

v17 replaces v14's linear PCA expression code with a **learned variational
autoencoder** (VAE) with a **negative-binomial (count-aware) decoder** (scVI-style).
The joint molecular-morphological latent is KL-regularized toward a standard normal, so
the conditional flow-matching model trained in that latent space produces samples that
lie on the decoder's manifold and can be **decoded directly into gene expression** — no
retrieval required (retrieval is retained as an ablation flag).

Two-phase training (as in v14): (A) pretrain the VAE, freeze it; (B) train the flow in
the frozen latent space. Every default is the intended production setting.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LatentConfig:
    """Stage 1 — morphological (pseudo-image) channels fused into the VAE input."""
    morph_k: int = 12               # kNN for per-cell soft cell-type + density channels
    density_sigma: float = 1.0      # density kernel bandwidth in units of median spacing


@dataclass
class VAEConfig:
    """Stage 2 — variational joint encoder + negative-binomial decoder (scVI-style)."""
    latent_dim: int = 32            # d: joint latent h (KL-regularized -> flow-friendly)
    hidden: int = 256               # encoder/decoder MLP width
    n_layers: int = 2               # hidden layers per MLP
    dropout: float = 0.05
    kl_weight: float = 1.0e-3       # weight on KL(q(h|x) || N(0,I)) (beta-VAE)
    kl_warmup_epochs: int = 20      # linearly ramp the KL weight over this many epochs
    likelihood: str = "auto"        # "auto" -> "nb" for counts, "gaussian" otherwise
    w_morph: float = 0.5            # auxiliary morphology-reconstruction weight
    w_type: float = 0.5             # auxiliary cell-type prediction weight
    eps: float = 1.0e-4


@dataclass
class AttnConfig:
    """Stage 3.1 — 3D positional-attention context module (unchanged from v14)."""
    d_model: int = 96
    n_heads: int = 4
    n_context: int = 16
    n_global_tokens: int = 1
    fourier_bands: int = 6
    dropout: float = 0.05
    context_slices_each_side: int = 1


@dataclass
class FlowConfig:
    """Stage 3.2 — conditional flow-matching vector field (unchanged from v14)."""
    hidden: int = 192
    n_layers: int = 4
    time_embed_dim: int = 32
    sigma_min: float = 1e-3
    n_ode_steps: int = 12
    n_ensemble: int = 4


@dataclass
class TrainConfig:
    """Stage 6 — two-phase training strategy."""
    pretrain_epochs: int = 120      # Phase A: VAE (longer than v14 — a real generative decoder)
    epochs: int = 160               # Phase B: flow field + attention
    batch_cells: int = 256
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 2.0
    gap_dropout: float = 0.35
    z_sigma: float = 0.15
    device: str = "auto"
    seed: int = 42
    verbose: bool = True
    fallback_on_error: bool = True


@dataclass
class GenerationConfig:
    """Stage 5 — inference: decode the flow-generated latent into expression."""
    decode_only: bool = True        # emit decoder(h*) expression (fully generative)
    emit: str = "sample"            # "sample" -> NB posterior-predictive draw; "mean" -> NB mean
    retrieval: bool = False         # ABLATION: switch to v14-style real-profile retrieval
    ground_k: int = 2               # (retrieval mode) candidate real cells per spot
    output_counts: bool = True      # emit count-like expression for the evaluator
    composition_match: bool = True  # match cell-type composition to the interpolated mix
    # Position layout for the generated sheet (same coherent-patch scheme as v14).
    position_mode: str = "flanking"      # "flanking" | "nearest"
    coherent_source: bool = True
    coherent_freq: float = 4.0
    near_identical_ratio: float = 0.60
    library_from_anchor: bool = True  # per-cell library size from its coherent-patch anchor cell


@dataclass
class SpatialCPAv17Config:
    latent: LatentConfig = field(default_factory=LatentConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)
    attn: AttnConfig = field(default_factory=AttnConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42
