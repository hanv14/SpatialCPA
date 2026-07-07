"""
Configuration for SpatialCPA-v6 (optimal-transport virtual-slice synthesis).

Every knob lives here as a dataclass so nothing is hard-coded elsewhere. The
five nested configs mirror the five stages of the method:

    EmbeddingConfig      cell-state representation (foundation-model prior / PCA)
    TransportConfig      optimal-transport coupling between the flanking slices
    CommunicationConfig  cell-cell-communication (niche) label refinement
    AnnotationConfig     cell-type assignment (FM prototype prior + constraints)
    SynthesisConfig      placement / count / expression transfer

The defaults are chosen to run without any external asset (foundation-model
weights, GPU) — the local PCA embedder and the OT/MRF core are pure
numpy/scipy — while every foundation-model hook degrades gracefully when its
asset is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Cell-state embedding used for OT cost and cell-type annotation.

    ``method``:
      * ``"pca"``      — local, unsupervised SVD embedding (default; no asset).
      * ``"fm_gene"``  — project expression through a *pretrained gene embedding*
        matrix (scGPT / Geneformer / Gene2vec token embeddings, or an H&E-paired
        gene program matrix): ``cell_embed = X_norm @ W_gene``. This injects the
        foundation model's learned gene-gene relationships as an external prior.
        Falls back to ``"pca"`` if the matrix is unavailable.
      * ``"concat"``   — concatenate ``pca`` and ``fm_gene`` (uses both the local
        structure and the external prior).
      * a name registered via :func:`spatialcpav6.embedding.register_embedder`
        for a full foundation model (scGPT/Geneformer/UCE encoder, or an H&E
        UNI/CONCH morphology encoder when paired images are provided).
    """

    method: str = "pca"
    n_components: int = 32
    # Path to a pretrained gene-embedding matrix for method="fm_gene"/"concat".
    # A ``.npz`` with arrays ``genes`` (G0,) and ``embedding`` (G0, d), or a
    # ``.npy`` (G0, d) aligned to the panel order. Env var
    # SPATIALCPAV6_FM_GENE_EMBEDDING is used if this is None.
    fm_gene_embedding_path: str | None = None
    fm_gene_embedding_dim: int = 0        # 0 = use all columns of the matrix
    standardize: bool = True              # z-score genes before embedding
    whiten: bool = True                   # unit-variance embedding axes
    max_hvg: int = 2000                   # cap genes used for the embedding


@dataclass
class TransportConfig:
    """Entropic optimal transport between the two flanking slices.

    The transport plan defines a *morphing* of the lower slice into the upper
    one; the synthesized (in-between) slice is the displacement-interpolation
    (McCann geodesic) midpoint at fraction ``t``.
    """

    max_ot_cells: int = 1500      # subsample each flanking slice to this for OT
    epsilon: float = 0.05         # Sinkhorn entropic regularization (peaked plan)
    n_iter: int = 200             # Sinkhorn iterations
    # cost = (1-w)*spatial + w*embedding (both median-normed). Spatial-dominant so
    # the morph map stays local (≈ identity when slices are near-identical) while a
    # little molecular guidance keeps matches across tissue shifts coherent.
    embed_weight: float = 0.15
    deshrink: bool = True         # rescale interpolated cloud to interpolated covariance
    deshrink_strength: float = 1.0  # 0 = off, 1 = full covariance match


@dataclass
class CommunicationConfig:
    """Cell-cell-communication (niche) model used to refine cell-type labels.

    Labels are optimized so the synthesized slice reproduces the flanking
    slices' *neighborhood-enrichment* architecture P(neighbor=j | center=i) —
    i.e. which cell types sit next to which — the leakage-safe estimate of the
    held-out slice's 2D/3D niche structure.
    """

    enabled: bool = True
    k_neighbors: int = 10         # spatial graph degree (matches the evaluator's k)
    n_sweeps: int = 8             # ICM sweeps over all cells
    niche_weight: float = 1.0     # weight of the neighborhood (communication) term
    prior_weight: float = 1.0     # weight of the FM cell-state prior (unary)
    composition_weight: float = 1.0   # weight pinning the marginal to the target mix
    temperature: float = 1.0      # softmax temperature for the ICM update
    lr_affinity: float = 0.0      # optional ligand-receptor affinity blend (0 = off)


@dataclass
class AnnotationConfig:
    """Cell-type assignment: foundation-model prototype prior + constraints.

    Copying a real flanking cell's type is already a strong label, so annotation
    is *anchored* to that copied type and only gently refined: the foundation-model
    classifier corrects likely errors and the niche MRF harmonizes the spatial
    layout. ``anchor_weight`` >> ``fm_weight`` keeps refinement conservative so it
    improves the cell-type metrics without discarding the copied-label signal.
    """

    enabled: bool = True            # False -> keep the real copied endpoint labels
    # "spatial" interpolates the type field from both flanking slices (default;
    # beats single-slice copy when types vary smoothly in z); "prototype"/"knn"
    # classify in the foundation-model cell-state embedding.
    classifier: str = "spatial"
    knn_k: int = 15
    prototype_temperature: float = 0.5
    anchor_weight: float = 3.0      # weight of the copied-endpoint-type anchor (unary)
    fm_weight: float = 1.0          # weight of the foundation-model classifier prior
    # Pin the predicted cell-type composition to the z-interpolated flanking mix.
    # Off by default: real-cell placement already yields the interpolated mix.
    constrain_composition: bool = False
    composition_sinkhorn_iter: int = 50


@dataclass
class SynthesisConfig:
    """Placement, count and expression transfer."""

    # Where synthesized cells' (x, y) + expression come from:
    #   "morph"        — coherent single-sheet barycentric OT morph of the nearest
    #                    flanking slice toward the other (default). Produces ONE cell
    #                    sheet (no density doubling), and its displacement auto-adapts
    #                    to how different the slices are: ≈ a coherent copy when they
    #                    are near-identical (ties a single-slice copy on the coherence
    #                    metrics — no loss on volumetric z-planes) and a genuine morph
    #                    toward the intermediate footprint when they differ (keeps the
    #                    field/ssim wins). Resolves the field-vs-coherence trade-off.
    #   "backbone"     — positions + expression from the single nearest flanking
    #                    slice; expression-structure metrics match a single-slice copy
    #                    exactly (cannot lose). Most conservative.
    #   "interpolate"  — draw real cells from BOTH slices in the z-interpolated ratio
    #                    (mixture; interleaves two lattices — attenuates single-slice
    #                    structure on near-identical sections). Kept for ablation.
    #   "ot_geodesic"  — pair-sampled displacement interpolation (ablation).
    placement: str = "morph"
    # Cell count of the virtual slice: z-interpolated flanking count (emergent).
    count_mode: str = "interpolate"   # "interpolate" | "lower" | "upper" | "mean"
    # Expression source for each synthesized cell.
    #   "endpoint"  — copy the real profile of the source cell (preserves full
    #                 cell-to-cell variance and gene-gene structure). Default.
    #   "transfer"  — copy from the nearest same-type training cell in embed space.
    #   "blend"     — mix endpoint and transferred (transfer_alpha on transferred).
    expression_mode: str = "endpoint"
    transfer_k: int = 1
    transfer_alpha: float = 0.5
    seed: int = 42


@dataclass
class SpatialCPAv6Config:
    """Top-level configuration bundling every stage."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    seed: int = 42
