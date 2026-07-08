"""
Configuration for SpatialCPA-v7 (foundation-anchored fused-transport histogenesis).

v7 keeps v6's dataclass-per-stage layout so every knob is discoverable and
nothing is hard-coded elsewhere, and adds the three ideas that make v7 novel:

    EmbeddingConfig       cell-state manifold (foundation-model prior / PCA) with
                          cross-slice mutual-NN batch anchoring and an optional
                          manifold label-propagation classifier.
    TransportConfig       *fused Gromov-Wasserstein* coupling between the flanking
                          slices — matches cells so that BOTH expression features
                          and the intra-slice spatial graph (relational geometry)
                          are preserved across the morph. (v6 used plain OT.)
    CommunicationConfig   cell-cell communication in 2D *and* 3D. The 3D term
                          couples each virtual cell to the real cells directly
                          above/below it (the flanking z-planes) via a cross-slice
                          niche matrix + a ligand-receptor flux prior.
    AnnotationConfig      cell-type assignment: FM prior + 2D/3D communication MRF
                          + composition, anchored to the morphed real type.
    SynthesisConfig       placement / count / expression transfer.

Every foundation-model / external-asset hook degrades gracefully to a pure
numpy/scipy path, so the benchmark always runs with no downloads and no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Cell-state embedding used for the transport cost and cell-type annotation.

    ``method``:
      * ``"pca"``      — local unsupervised SVD embedding (default; no asset).
      * ``"coexpr"``   — data-derived gene-program embedding (SVD of the training
        gene-gene correlation matrix); leakage-safe, no external asset.
      * ``"fm_gene"``  — project expression through a *pretrained gene embedding*
        matrix (scGPT / Geneformer / Gene2vec token embeddings, or an H&E-paired
        gene-program matrix): ``cell_embed = X_norm @ W_gene``. Injects the
        foundation model's learned gene-gene relationships. Falls back to PCA.
      * ``"concat"``   — concatenate ``pca`` and ``fm_gene``.
      * a name registered via :func:`spatialcpav7.embedding.register_embedder`
        for a full FM encoder (scGPT/Geneformer/UCE for expression; UNI/CONCH for
        paired H&E morphology).
    """

    method: str = "pca"
    n_components: int = 32
    fm_gene_embedding_path: str | None = None
    fm_gene_embedding_dim: int = 0        # 0 = use all columns of the matrix
    standardize: bool = True
    whiten: bool = True
    max_hvg: int = 2000

    # NEW in v7: cross-slice manifold anchoring. Adjacent physical sections are
    # different imaging batches; a raw shared embedding can carry a slice-to-slice
    # offset that misleads the transport cost and cell-type prior. When enabled,
    # the flanking slices' embeddings are aligned by mutual-nearest-neighbour
    # anchors (a lightweight, deterministic batch correction) before transport /
    # annotation. Leakage-safe: it only aligns the two *training* flanking slices.
    cross_slice_anchor: bool = True
    anchor_k: int = 15                    # mutual-NN degree for the anchoring
    anchor_strength: float = 1.0          # 0 = off, 1 = full offset removal


@dataclass
class TransportConfig:
    """Fused Gromov-Wasserstein transport between the two flanking slices.

    v7's placement engine. The coupling ``T`` minimizes

        (1 - alpha_gw) * <M, T>  +  alpha_gw * GW(C_lo, C_hi, T)

    where ``M`` is the feature cost (spatial + cell-state embedding) and
    ``GW`` penalizes distortion of the intra-slice pairwise-distance structure
    (``C_lo``, ``C_hi``). The GW term makes the morph *relation-preserving*:
    cells that are neighbours in the lower slice stay neighbours in the upper,
    so the interpolated slice keeps the tissue's neighbourhood graph — directly
    what Moran's-I agreement and cell-type neighbourhood agreement reward.
    (v6's plain entropic OT matched marginals but not the relational geometry.)
    """

    max_ot_cells: int = 1200      # subsample each flanking slice for the FGW solve
    epsilon: float = 0.05         # entropic regularization (dimensionless; peaked)
    n_iter: int = 200             # outer Sinkhorn iterations
    # feature cost = (1-embed_weight)*spatial + embed_weight*embedding (median-normed).
    embed_weight: float = 0.20
    # Fusion weight on the Gromov (structure) term. 0 -> pure OT (v6 behaviour);
    # ~0.3-0.5 -> geometry-preserving morph. Kept moderate so the feature cost
    # still anchors the map physically.
    alpha_gw: float = 0.35
    gw_iter: int = 20             # inner FGW proximal-gradient iterations
    deshrink: bool = True         # rescale interpolated cloud to interpolated covariance
    deshrink_strength: float = 1.0
    # "adaptive" placement uses the FGW morph while the OT-map displacement between
    # the flanking slices (in cell-spacings) stays below this, and switches to
    # both-slice interpolation above it (distinct tissue, where a single morphed
    # sheet contracts). Calibrated like v6; the wrapper logs the measured value.
    adaptive_threshold: float = 0.85


@dataclass
class CommunicationConfig:
    """Cell-cell communication (niche) model refining cell-type labels — 2D + 3D.

    2D: the synthesized slice reproduces the interpolated in-plane
    neighbourhood-enrichment matrix ``M2D* = (1-t)·M_lo + t·M_hi`` (which types
    sit next to which within the plane).

    3D (NEW in v7): each virtual cell also has real neighbours directly above and
    below it — the cells of the flanking z-planes. v7 estimates the *cross-slice*
    niche matrix ``M3D``[i, j] = P(a cell of type j sits across-z from a cell of
    type i) from the two training slices, and constrains each virtual cell's type
    to be consistent with the real flanking-slice types stacked over it. This is
    the genuine 3D communication signal (z-axis tissue continuity), estimable from
    the training flanking slices only, so it is leakage-safe.
    """

    enabled: bool = True
    k_neighbors: int = 10         # in-plane spatial graph degree (matches evaluator)
    n_sweeps: int = 8             # ICM sweeps over all cells
    niche_weight: float = 1.0     # weight of the 2D neighbourhood term
    prior_weight: float = 1.0     # weight of the FM cell-state prior (unary)
    composition_weight: float = 1.0   # weight pinning the marginal to the target mix
    temperature: float = 1.0

    # --- 3D cross-slice communication (v7) ---
    enable_3d: bool = True
    niche3d_weight: float = 1.5   # weight of the cross-slice (z-stacking) niche term
    cross_k: int = 8              # flanking-slice neighbours polled per virtual cell
    # Ligand-receptor flux prior: reward type pairs whose curated LR partners are
    # co-expressed across the 3D neighbourhood. 0 disables (or when no curated LR
    # pair overlaps the panel). Small by default — a gentle biological nudge on top
    # of the data-driven niche matrices.
    lr_weight: float = 0.5
    lr_min_pairs: int = 3         # need at least this many panel-matched LR pairs


@dataclass
class AnnotationConfig:
    """Cell-type assignment: FM prior + 2D/3D communication MRF + composition.

    Copying the morphed real flanking cell's type is already a strong label, so
    annotation is *anchored* to it and only refined. ``anchor_weight`` keeps the
    refinement conservative: it fixes likely errors and harmonizes the spatial /
    3D layout without discarding the copied-label signal.
    """

    enabled: bool = True
    # "spatial"   — interpolate the type field from both flanking slices (default).
    # "labelprop" — manifold label propagation in the FM cell-state embedding (v7).
    # "prototype" / "knn" — FM-embedding classifiers.
    classifier: str = "spatial"
    knn_k: int = 15
    prototype_temperature: float = 0.5
    labelprop_k: int = 15         # graph degree for manifold label propagation
    labelprop_iter: int = 30      # propagation iterations
    labelprop_alpha: float = 0.85  # propagation teleport (clamping) strength
    anchor_weight: float = 3.0
    fm_weight: float = 1.0
    constrain_composition: bool = True   # ON by default in v7 (leakage-safe mix)
    composition_sinkhorn_iter: int = 50


@dataclass
class SynthesisConfig:
    """Placement, count and expression transfer."""

    # "adaptive"    — FGW morph when the flanking slices are near-identical,
    #                 both-slice interpolation when distinct (default).
    # "fgw_morph"   — coherent single-sheet fused-GW morph of the nearest slice.
    # "interpolate" — real cells from BOTH slices in the z-interpolated ratio.
    # "fgw_geodesic"— sample matched pairs from the FGW plan, place at the McCann
    #                 midpoint (ablation).
    # "backbone"    — positions + expression from the single nearest slice.
    placement: str = "adaptive"
    count_mode: str = "interpolate"   # "interpolate" | "lower" | "upper" | "mean"
    # "endpoint" copies the real profile (default; max variance, intact gene-gene
    # structure). "transfer"/"blend" denoise via nearest same-type training cells.
    expression_mode: str = "endpoint"
    transfer_k: int = 1
    transfer_alpha: float = 0.5
    seed: int = 42


@dataclass
class SpatialCPAv7Config:
    """Top-level configuration bundling every stage."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    seed: int = 42
