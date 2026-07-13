"""
Configuration for SpatialCPA-v9 (neural cross-slice flow-matching bridge).

v9 is the first *learned* SpatialCPA generator: a conditional flow-matching
(rectified-flow) model transports the cell distribution of one flanking slice to
the other in a joint (position, expression-latent) space, conditioned on the axial
gap and a permutation-invariant context summary of the neighbouring slices.
Evaluating the learned transport at the fractional depth of the target z yields the
virtual slice — a *learned* generalization of the optimal-transport displacement
interpolation used by v6/v8.

Everything trains on the training slices only (all adjacent pairs supply
interpolation supervision) and then generalizes to any query z. The model degrades
gracefully: if PyTorch is unavailable or training fails, it falls back to the v8
coherent optimal-transport morph so a prediction is always produced.

The nested configs:

    EmbeddingConfig     optional pretrained gene-embedding warm-start for the AE
    ModelConfig         autoencoder + flow-network architecture
    TrainConfig         optimization schedule (AE + flow)
    FlowConfig          ODE integration at inference
    TransportConfig     OT pairing for flow supervision + the fallback morph
    AnnotationConfig    cell-type assignment (OT-anchor + prior + niche MRF)
    CommunicationConfig cell-cell-communication (niche) label refinement
    SynthesisConfig     count / expression decoding
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Optional pretrained gene embedding used to *warm-start* the AE encoder.

    ``method="pca"`` (default) initializes nothing special; ``"fm_gene"`` loads a
    pretrained gene-embedding matrix (scGPT / Geneformer / Gene2vec) and uses it to
    initialize the first encoder layer, injecting external biology. Falls back to a
    random init when the matrix is unavailable.
    """

    method: str = "pca"
    fm_gene_embedding_path: str | None = None
    max_hvg: int = 2000


@dataclass
class ModelConfig:
    """Autoencoder + flow-network architecture."""

    latent_dim: int = 16          # expression latent dimension the flow operates in
    ae_hidden: int = 256          # autoencoder hidden width
    flow_hidden: int = 256        # velocity-network hidden width
    flow_layers: int = 4          # velocity-network depth
    time_embed_dim: int = 32      # sinusoidal time-embedding width
    context_dim: int = 64         # cross-slice context summary width
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Optimization schedule (training-free at inference; trains per holdout)."""

    ae_epochs: int = 150
    flow_epochs: int = 400
    batch_size: int = 2048
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    ot_pairing: bool = True       # pair adjacent-slice cells by OT for supervision
    max_pairs_per_epoch: int = 20000
    grad_clip: float = 5.0
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    seed: int = 42
    verbose: bool = True
    # If training diverges / errors, fall back to the OT morph (never fail).
    fallback_on_error: bool = True


@dataclass
class FlowConfig:
    """Inference-time ODE integration of the learned velocity field."""

    n_steps: int = 24             # Euler steps to integrate the probability-flow ODE
    # Blend the learned displacement with the coherent OT-morph displacement as a
    # structural prior: 1.0 = pure learned flow, 0.0 = pure OT morph. A small prior
    # weight regularizes the learned field toward a coherent tissue deformation.
    morph_prior: float = 0.5
    stochastic: bool = False      # add small noise per step (SDE sampler)
    noise_scale: float = 0.0


@dataclass
class TransportConfig:
    """Entropic-OT pairing for flow supervision and the fallback coherent morph."""

    max_ot_cells: int = 1000
    epsilon: float = 0.05
    n_iter: int = 200
    embed_weight: float = 0.15
    deshrink: bool = True
    deshrink_strength: float = 1.0
    smooth_k: int = 12            # displacement-field smoothing for the fallback morph
    smooth_iters: int = 3


@dataclass
class CommunicationConfig:
    enabled: bool = True
    k_neighbors: int = 10
    n_sweeps: int = 8
    niche_weight: float = 1.0
    prior_weight: float = 1.0
    composition_weight: float = 1.0
    temperature: float = 1.0
    lr_affinity: float = 0.0


@dataclass
class AnnotationConfig:
    enabled: bool = True
    classifier: str = "spatial"      # "spatial" | "prototype" | "knn"
    knn_k: int = 15
    prototype_temperature: float = 0.5
    anchor_weight: float = 3.0
    fm_weight: float = 1.0
    constrain_composition: bool = True
    composition_sinkhorn_iter: int = 50


@dataclass
class SynthesisConfig:
    count_mode: str = "interpolate"   # "interpolate" | "lower" | "upper" | "mean"
    # Expression decoding for each synthesized cell:
    #   "source"    — real profile of the lower cell each output cell flowed from
    #                 (default; the learned flow refines POSITIONS only, so
    #                 expression/co-expression/variance are never degraded by an
    #                 imperfectly-trained flow — v9 >= the OT-morph baseline).
    #   "nearest"   — snap to the nearest real training cell in latent space.
    #   "decode"    — decode the flowed latent through the AE decoder (learned).
    #   "blend"     — average of decode and nearest.
    expression_mode: str = "source"
    blend_alpha: float = 0.5
    seed: int = 42


@dataclass
class SpatialCPAv9Config:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    seed: int = 42
