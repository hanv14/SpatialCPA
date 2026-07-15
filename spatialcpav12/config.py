"""
Configuration for SpatialCPA-v12 — generative continuous 3D virtual-slice model.

v12 *enhances* v11's two-stage continuous implicit-field design. It keeps the
Stage-1 ``LayoutField`` (occupancy + cell-type/region, distilled from a frozen
foundation-model teacher) and the continuous-``z`` querying, but replaces v11's
mean-regressing Stage-2 expression field with a **generative, covariance-preserving
decoder** and adds leakage-safe field calibration:

    Stage 1  LayoutField(x, y, z | context)          -> occupancy + cell-type/region
    Stage 2  GenerativeExpressionField(x, y, z, ...)  -> N(mu(q), L Lᵀ + Ψ) sampler

The four v12 contributions (all distinct from the training-free OT/copy family):

1. **Conditional factor-analysis expression decoder.** Each cell's expression is a
   sample ``x = mu(q) + L·s`` (``s ~ N(0, I_r)``), a Gaussian with covariance
   ``L Lᵀ + Ψ`` whose low-rank loadings ``L`` are trained to the *real* gene-gene
   covariance by a factor-analysis likelihood. Synthesized cells therefore carry
   realistic gene-gene correlation and per-gene variance — the structure a pure
   MSE field (v11) collapses away — while remaining genuinely generated, not copied.
2. **Spatially-coherent latent field.** The factor code ``s`` is drawn as a smooth
   field over the synthesized cells' kNN graph (plus an idiosyncratic part), so
   spatially-variable genes stay spatially autocorrelated (Moran's I) instead of
   being scattered by i.i.d. per-cell noise.
3. **Leakage-safe field calibration.** The sampling density and the per-gene
   mean/variance are calibrated to the *z-interpolated flanking* fields (training
   slices only), sharpening the density field and pinning per-gene statistics.
4. **Prior-corrected composition.** Predicted cell-type proportions are nudged to
   the z-interpolated flanking composition by soft prior correction (no MRF).
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
    """Stage-2 generative expression field (conditioned on layout)."""
    hidden: int = 256
    layers: int = 5
    dropout: float = 0.0
    # Generative decoder: expression = mu(query) + L @ s, s ~ N(0, I_r).
    n_factors: int = 24           # rank r of the shared factor-loading matrix L
    min_log_psi: float = -6.0     # floor for the per-gene idiosyncratic log-variance
    max_log_psi: float = 4.0      # ceiling (numerical stability of the FA likelihood)


@dataclass
class TeacherConfig:
    """Frozen multimodal foundation-model teacher for layout distillation.

    Identical mechanism to v11: ``omiclip`` (real, gene-sentence -> CLIP/CoCa text
    tower), ``path2space`` / ``gene_embedding`` (pretrained gene-embedding matrix),
    ``auto`` (real if an asset is given, else the data-derived proxy), or ``proxy``.
    """
    kind: str = "auto"
    name: str = "omiclip"
    weights_path: str | None = None
    embed_dim: int = 64
    n_pseudo_domains: int = 8     # pseudo-layout (spatial-domain) clusters from the teacher
    model_arch: str = "coca_ViT-L-14"   # open_clip architecture for OmiCLIP
    top_genes: int = 50                 # genes per spot in the OmiCLIP gene-sentence
    encode_batch: int = 256             # spots per forward pass on the FM
    device: str = "auto"                # teacher inference device (auto|cpu|cuda)
    gene_embedding_path: str | None = None
    symbol_map_path: str | None = None


@dataclass
class LossConfig:
    """Weights for every training objective."""
    # Stage 1
    w_layout_occ: float = 1.0         # occupancy reconstruction (BCE)
    w_layout_type: float = 1.0        # cell-type reconstruction (CE) on real spots
    w_distill_embed: float = 1.0      # teacher feature-alignment (cosine/MSE)
    w_distill_pseudo: float = 0.5     # teacher pseudo-layout (domain) distillation (CE)
    # Stage 2
    w_expr_recon: float = 1.0         # mean reconstruction (MSE) on real spots (anchors mu)
    w_expr_nll: float = 0.5           # factor-analysis NLL (trains L, Ψ to real covariance)
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
    """Continuous querying / generative decoding of a discrete slice from the fields."""
    # z-marginalization / hybrid inference over a small z-window around the query.
    z_marginalize: int = 3        # samples in the z window (1 = point query)
    z_window: float = 0.1         # half-width of the z window (normalized z)
    occ_grid: int = 96            # grid for sampling the occupancy field (finer than v11)
    count_mode: str = "interpolate"   # emergent cell count
    # Position generation: "auto" (default) picks "flanking" for near-identical planes
    # (real positions reproduce the density) and "field" for distinct/drifting tissue
    # (the learned occupancy grid must place cells); "hybrid" reweights real flanking
    # positions by the learned occupancy field.
    position_mode: str = "auto"
    flanking_carry_type: bool = False    # in flanking/hybrid mode, keep each proposal cell's real type
    flanking_single_slice: bool = True   # sample flanking positions from the single nearest slice
    # Expression decode:
    #   "generative" (default) — sample x = mu + L·s with a spatially-coherent latent;
    #   "field"      — deterministic mean mu only (pure Stage-2 mean);
    #   "residual"   — v11-style blend of mu with a real same-type profile (fallback).
    expr_decode: str = "generative"
    latent_coherence: float = 0.9  # fraction of the factor code drawn as a smooth field
    anchor_weight: float = 0.9     # weight of the real same-type mean anchor (0 = pure field mean)
    anchor_expr_weight: float = 0.0  # weight of predicted expression-state in anchor retrieval (0 = position-only)
    noise_scale: float = 0.2       # scale of the additive factor-analysis noise (0 = deterministic mean)
    residual_weight: float = 0.7   # weight of the real profile when expr_decode="residual"
    # Leakage-safe calibration to the z-interpolated flanking fields (training slices).
    calibrate_gene_stats: bool = False  # pin per-gene mean/variance to the interpolated target
    output_counts: bool = True          # emit count-like (expm1) expression for the evaluator
    calibrate_density: bool = False     # blend the learned occupancy with the interpolated density
    density_blend: float = 0.5          # blend of learned occupancy vs interpolated density
    composition_calibrate: bool = True  # prior-correct type proportions to the interpolated mix


@dataclass
class SpatialCPAv12Config:
    fourier: FourierConfig = field(default_factory=FourierConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42
