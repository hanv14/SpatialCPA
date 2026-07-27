"""Configuration for SpatialCPA-v15.

Every default here is the **intended production setting**: constructing
``SpatialCPAv15Config()`` and running the generator reproduces the pipeline
described in ``spatialcpa_v15_idea.md`` end to end, with no flags to set.

The dataclasses map one-to-one onto the phases of the proposal:

======================  ==========================================================
dataclass               proposal section
======================  ==========================================================
``QCConfig``            Phase 1.1 — per-section QC
``TypeConfig``          Phase 1.2 — markers, prior encoder, data backbone, blend
``HierarchyConfig``     Phase 1.3 — class -> subclass -> cluster
``FrameConfig``         Phase 1.4/1.5/1.6 — common frame, normalized coords, graph
``FieldConfig``         Phase 2.1/2.2 — multi-channel rasterization + compression
``StructureConfig``     Phase 2.3/2.4/2.5 — query-based VFI diffusion + gate
``LayoutConfig``        Phase 2.4/4.3 — marked spatial point process
``ExprVAEConfig``       Phase 3.1 — NB/ZINB expression VAE (used as a likelihood)
``ExprDiffusionConfig`` Phase 3.2/3.3 — conditional latent diffusion + gate
``InferenceConfig``     Phase 4 — seeds, provenance, output
``RuntimeConfig``       device selection and GPU-sharing policy
======================  ==========================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Phase 1 ───────────────────────────────────────────────────────────────────

@dataclass
class QCConfig:
    """Phase 1.1 — per-section QC (kept per section so batch structure stays visible)."""

    min_total_expression: float = 0.0
    """Drop cells whose total expression is <= this (empty segmentations)."""

    min_cells_per_section: int = 8
    """Sections below this are reported as low-quality (never silently dropped)."""


@dataclass
class TypeConfig:
    """Phase 1.2 — the continuous type embedding ``e_c``."""

    n_components: int = 16
    """Dimensionality d of ``e_c`` (proposal: d ~ 8-32)."""

    top_k_markers: int = 30
    """Markers kept per type, with signed effect size (proposal: 20-50)."""

    min_cells_for_markers: int = 5
    """Types with fewer cells get an all-zero marker signature."""

    symbol_embedding_dim: int = 64
    """Width of the order-invariant gene-symbol encoder (the external prior)."""

    fm_gene_embedding_path: Optional[str] = None
    """Optional pretrained gene embedding (.npz with genes/embedding, or a
    panel-aligned .npy). When given it replaces the built-in symbol encoder as
    the Phase 1.2.1 encoder — everything downstream is unchanged."""

    shrinkage_kappa: float = 50.0
    """kappa in the sample-size weight ``w_c = n_c / (n_c + kappa)`` (Phase 1.2.2).
    Well-sampled types follow the data backbone; rare types borrow the prior."""

    batch_correct_pseudobulk: bool = True
    """Center pseudobulk per section before pooling, so section-level technical
    offsets do not enter *both* embeddings (the caveat in Phase 1.2.2)."""


@dataclass
class HierarchyConfig:
    """Phase 1.3 — class -> subclass -> cluster over the type embedding."""

    n_classes: Optional[int] = None
    """None -> ceil(sqrt(C)) classes."""

    n_subclasses: Optional[int] = None
    """None -> 2x n_classes (capped at C)."""


@dataclass
class FrameConfig:
    """Phase 1.4/1.5/1.6 — common frame, normalized coordinates, 3D graph."""

    stabilize: str = "auto"
    """Phase 1.4 landmark stabilization: ``auto`` applies a translation-only
    drift correction (from point-derived landmarks: section centroid + tissue
    outline) *only* when the residual drift exceeds ``drift_tolerance``;
    ``always`` / ``never`` force it. Rigid inter-section registration proper is
    the harness's job (it re-registers training-only per holdout); this step
    only removes leftover drift, and is inverted before output."""

    drift_tolerance: float = 0.5
    """Drift threshold in units of median in-plane cell spacing."""

    k_neighbors_3d: int = 12
    """Phase 1.6 — 3D kNN graph degree; every edge stores its own ``dz``."""


# ── Phase 2 ───────────────────────────────────────────────────────────────────

@dataclass
class FieldConfig:
    """Phase 2.1/2.2 — multi-channel structure field."""

    bin_spacings: float = 2.0
    """In-plane voxel size in units of median cell spacing. The proposal asks for
    ~50 um; expressing it in cell spacings makes it technology-independent and
    keeps voxel aspect ratios sane (Phase 2.1 anisotropy note)."""

    min_grid: int = 24
    max_grid: int = 56
    """Clamp on the in-plane grid so 3D convolutions stay well conditioned."""

    kde_bandwidth_bins: float = 1.0
    """Gaussian smoothing kernel K_h, in voxels."""

    max_group_channels: int = 24
    """Per-type density channels are used while the number of types is <= this;
    above it the channels drop to the coarser hierarchy level (Phase 2.1,
    'many types' note) and the embedding field carries within-group identity."""

    embedding_channels: int = 8
    """Number of density-weighted mean type-embedding channels (Phase 2.1(b))."""

    compress_max_channels: int = 40
    """Phase 2.2 — above this many channels the field is compressed with a
    channel-space PCA fitted on the observed sections. At the volume sizes this
    method targets a full 3D VAE is unnecessary; the rule is automatic."""


@dataclass
class StructureConfig:
    """Phase 2.3/2.4/2.5 — query-based interpolation, flow-free and generative."""

    enabled: bool = True

    hidden: int = 64
    """U-Net base width."""

    n_timesteps: int = 200
    """Diffusion steps used in training (cosine schedule)."""

    ddim_steps: int = 20
    """Deterministic DDIM (eta=0) sampling steps — Phase 2.4."""

    epochs: int = 220
    batch_size: int = 16
    lr: float = 2e-3

    crop: int = 32
    """Random in-plane crop used for augmentation (translation invariance)."""

    z_reversal: bool = True
    """Phase 2.3 — z-reversal is an exact symmetry of the problem: free data."""

    mirror: bool = False
    """In-plane mirroring is OFF by default: it is not a safe symmetry when any
    biology is lateralized (Phase 2.3 caution)."""

    asymmetric_brackets: bool = True
    """Allow a missing outer bracket (duplicated inner one) — Phase 2.3."""

    gate_margin: float = 0.0
    """Phase 2.5 stop condition: the learned field must beat linear interpolation
    of the field by this margin (in relative MSE) on internal held-out slices,
    otherwise the linear field is used and the fact is reported."""


@dataclass
class LayoutConfig:
    """Phase 2.4/4.3 — layout as a marked spatial point process."""

    hardcore_beta: float = 0.55
    """Minimum inter-cell distance as a fraction of the observed median
    nearest-neighbour distance. A raw Poisson sample clumps; real tissue is
    regular, so a soft hardcore relaxation is applied."""

    hardcore_iters: int = 12

    intensity_upsample: int = 3
    """Bilinear refinement factor applied to the intensity before sampling.
    The field voxel is sized for the convolutions, not for placing cells."""

    softmax_tau: float = 0.35
    """Temperature for the within-group type posterior from the embedding field."""

    mark_sharpness: float = 1.0
    """Exponent applied to the type posterior before drawing marks. The completed
    field is an interpolation of the flanking type densities and is therefore
    smoother than either real slice; sharpening restores locally coherent marks
    (a cell inside a region dominated by one type should usually get that type)
    without touching the composition, which the repair step pins separately."""

    composition_repair: bool = True
    """Nudge the sampled type composition onto the field-implied composition by
    re-labelling the least-confident cells (no transport plan involved)."""


# ── Phase 3 ───────────────────────────────────────────────────────────────────

@dataclass
class ExprVAEConfig:
    """Phase 3.1 — the expression VAE, used as a *likelihood*, not a generator."""

    latent_dim: int = 24
    hidden: int = 128
    n_layers: int = 2

    likelihood: str = "auto"
    """``auto`` -> NB when the input is count-like, Gaussian otherwise.
    ``nb`` / ``zinb`` / ``gaussian`` force it."""

    kl_weight: float = 1e-3
    """Deliberately low (near-autoencoder): diffusion supplies the prior."""

    epochs: int = 260
    """Minimum optimizer steps. The actual number is
    ``max(epochs, min_passes * ceil(n_cells / batch_size))`` so a larger stack is
    not under-trained just because the step count is a constant."""

    min_passes: int = 15
    batch_size: int = 256
    lr: float = 2e-3

    min_active_units: float = 0.25
    """Posterior-collapse check: fraction of latent dims that must stay active."""


@dataclass
class ExprDiffusionConfig:
    """Phase 3.2/3.3 — conditional latent diffusion in the frozen VAE latent."""

    enabled: bool = True

    hidden: int = 96
    n_heads: int = 4
    n_blocks: int = 3

    n_context: int = 16
    """Neighbourhood context cells cross-attended per target cell, taken from the
    *real* bracketing slices."""

    n_timesteps: int = 200
    ddim_steps: int = 20

    epochs: int = 260
    block_size: int = 256
    """Target cells generated jointly per block (attention runs across the whole
    block, including across z — never independent per-slice)."""

    lr: float = 1.5e-3

    slice_dropout: float = 0.6
    """Probability of holding out a *consecutive run* of context slices during
    training, so the model learns to generate from receded context."""

    max_dropout_run: int = 2

    decoder_readout: str = "auto"
    """How latents become counts: ``mean`` (NB mean), ``sample`` (NB draw), or
    ``auto`` — chosen by the leakage-safe Phase 3.3 internal gate."""


# ── Runtime ───────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    """Where the three trained models run, and how the GPU is shared.

    SpatialCPA-v15 uses the GPU automatically when one is present. The defaults
    are deliberately **co-operative**: the process lets its memory grow and
    shrink with demand and hands cached blocks back between stages, so another
    process can use the same card at the same time.
    """

    device: str = "auto"
    """``auto`` (GPU when available), ``cuda`` or ``cpu``. Applies to all three
    trained models — there is no per-stage device."""

    cuda_index: int = 0
    """Which GPU to use when several are visible."""

    expandable_segments: bool = True
    """Back the CUDA caching allocator with expandable virtual-memory segments,
    so its pools grow and shrink instead of reserving fixed blocks that are never
    returned. This is what keeps the card usable by other processes."""

    release_cache_between_stages: bool = True
    """Return cached GPU blocks to the driver after each training stage, so the
    memory is free during the numpy-only parts of the pipeline and between the
    structure model, the VAE and the latent diffusion."""

    memory_fraction: Optional[float] = None
    """Optional hard cap on this process's share of the card (e.g. ``0.5``).
    ``None`` = uncapped. Set it when the GPU is genuinely shared and a peak must
    never crowd out a co-tenant; leave it off when the run has the card."""


# ── Phase 4 ───────────────────────────────────────────────────────────────────

@dataclass
class InferenceConfig:
    """Phase 4 — inference, uncertainty and provenance."""

    n_structure_seeds: int = 8
    """N deterministic structure completions; voxelwise spread across seeds is
    the structural uncertainty map (proposal suggests N ~ 20; 8 is the default
    for tractability and is the only numeric deviation — raise it freely)."""

    n_expression_samples: int = 1
    """Expression samples kept per cell (>1 returns the spread as uncertainty)."""

    output_counts: bool = True
    """Emit count-scale expression (library x NB rate). False -> log1p."""

    sub_nyquist_flag: bool = True
    """Flag structures whose z-extent is < 2*dz as unresolved (Phase 4.7)."""


# ── Top level ─────────────────────────────────────────────────────────────────

@dataclass
class SpatialCPAv15Config:
    """Complete configuration. Defaults == the proposed pipeline."""

    seed: int = 42
    verbose: bool = True

    qc: QCConfig = field(default_factory=QCConfig)
    types: TypeConfig = field(default_factory=TypeConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    field_: FieldConfig = field(default_factory=FieldConfig)
    structure: StructureConfig = field(default_factory=StructureConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    vae: ExprVAEConfig = field(default_factory=ExprVAEConfig)
    diffusion: ExprDiffusionConfig = field(default_factory=ExprDiffusionConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
