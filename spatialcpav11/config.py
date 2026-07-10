"""
Configuration for SpatialCPA-v11 — two-stage continuous 3D virtual-slice model.

v11 is a *continuous implicit-field* generator for aligned serial spatial
transcriptomics (no paired H&E). Two coordinate-networks (implicit neural fields)
are queried at arbitrary continuous ``(x, y, z)``:

    Stage 1  LayoutField(x, y, z | context)      -> occupancy + cell-type/region
    Stage 2  ExpressionField(x, y, z, layout)    -> gene-expression profile

Both are conditioned on a permutation-invariant encoding of the aligned neighbouring
real slices and on a Fourier encoding of the continuous query ``z``, so the model can
be queried *between or beyond* the observed sections. Stage 1 is trained by knowledge
distillation from a frozen multimodal foundation-model teacher (OmiCLIP / Path2Space)
plus self-supervised reconstruction of real slices; Stage 2 by expression
reconstruction. Cross-z consistency and biology-informed constraints regularize both.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FourierConfig:
    """Fourier-feature encodings for continuous coordinates."""
    z_bands: int = 8          # sinusoidal bands for the continuous query z
    xy_bands: int = 10        # sinusoidal bands for (x, y)
    z_max_freq: float = 8.0
    xy_max_freq: float = 16.0


@dataclass
class ContextConfig:
    """Encoder of the aligned neighbouring real slices (features/embeddings)."""
    expr_embed_dim: int = 32      # PCA/FM expression embedding fed per spot
    hidden: int = 128
    context_dim: int = 128        # global permutation-invariant context vector
    raster_grid: int = 32         # low-res rasterization of each flanking slice
    raster_smooth: float = 1.0    # Gaussian smoothing (bins) of the raster
    n_context_slices: int = 2     # neighbouring slices used as context (flanking)


@dataclass
class LayoutConfig:
    """Stage-1 layout field (occupancy + type/region)."""
    hidden: int = 256
    layers: int = 5
    layout_feat_dim: int = 64     # hidden layout code handed to Stage 2
    dropout: float = 0.0


@dataclass
class ExpressionConfig:
    """Stage-2 expression field (conditioned on layout)."""
    hidden: int = 256
    layers: int = 5
    dropout: float = 0.0


@dataclass
class TeacherConfig:
    """Frozen multimodal foundation-model teacher for layout distillation.

    ``kind``:
      * ``"omiclip"`` / ``"path2space"`` — load the named pretrained encoder from
        ``weights_path`` (expression-side tower; images are not required at train
        time). Registered via :func:`spatialcpav11.teacher.register_teacher`.
      * ``"auto"`` (default) — use the named teacher if its weights are available,
        else fall back to the data-derived stand-in.
      * ``"proxy"`` — data-derived stand-in only (documented approximation): a
        spatial-domain embedding + clustering computed from the training slices, so
        distillation runs with no external asset. Swap in the real FM for publication.
    """
    kind: str = "auto"
    name: str = "omiclip"
    weights_path: str | None = None
    embed_dim: int = 64
    n_pseudo_domains: int = 8     # pseudo-layout (spatial-domain) clusters from the teacher


@dataclass
class LossConfig:
    """Weights for every training objective."""
    # Stage 1
    w_layout_occ: float = 1.0         # occupancy reconstruction (BCE)
    w_layout_type: float = 1.0        # cell-type reconstruction (CE) on real spots
    w_distill_embed: float = 1.0      # teacher feature-alignment (cosine/MSE)
    w_distill_pseudo: float = 0.5     # teacher pseudo-layout (domain) distillation (CE)
    # Stage 2
    w_expr_recon: float = 1.0         # expression reconstruction (MSE) on real spots
    # Cross-z consistency
    w_consistency_layout: float = 0.3
    w_consistency_expr: float = 0.3
    consistency_dz: float = 0.15      # finite-difference step in normalized z
    # Biology-informed constraints
    w_interface: float = 0.2          # interface preservation (neighbour-enrichment match)
    w_grad_smooth: float = 0.1        # within-domain expression gradient smoothness
    w_domain_coherence: float = 0.1   # spatial coherence of the type field


@dataclass
class TrainConfig:
    epochs: int = 300
    batch_points: int = 4096      # query points per step
    neg_ratio: float = 1.0        # empty (negative) queries per positive
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    seed: int = 42
    verbose: bool = True
    fallback_on_error: bool = True    # fall back to a deterministic layout if training fails


@dataclass
class InferenceConfig:
    """Continuous querying / decoding a discrete slice from the fields."""
    # z-marginalization / hybrid inference: average field predictions over a small
    # window of z around the query to stabilize between/beyond real slices.
    z_marginalize: int = 3        # samples in the z window (1 = point query)
    z_window: float = 0.1         # half-width of the z window (normalized z)
    occ_grid: int = 64            # grid for sampling the occupancy field
    count_mode: str = "interpolate"   # emergent cell count
    expr_decode: str = "residual"  # "residual" (layout-conditioned real profile; default) | "field" (pure Stage-2)
    residual_weight: float = 0.7  # for expr_decode="residual"


@dataclass
class SpatialCPAv11Config:
    fourier: FourierConfig = field(default_factory=FourierConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42
