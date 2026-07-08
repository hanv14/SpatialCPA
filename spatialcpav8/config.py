"""
Configuration for SpatialCPA-v8 (symmetric optimal-transport bridge synthesis).

v8 is training-free and pure numpy/scipy/sklearn, so it runs in the same
``bench_spatialcpa`` environment as v4-v6 with **no extra flags** — every default
below is the intended production setting, because the benchmark harness invokes
the wrapper with only ``--input/--target-section/--target-z/--output/--seed``.

Seven nested configs mirror the pipeline stages:

    EmbeddingConfig      cell-state representation (foundation-model prior / PCA)
    TransportConfig      entropic-OT coupling between the flanking slices
    BridgeConfig         how the in-between sheet is built from the OT coupling
    DensityConfig        local density calibration to the interpolated field
    CommunicationConfig  cell-cell-communication (niche) label refinement
    AnnotationConfig     cell-type assignment (OT-anchor + FM prior + niche)
    SynthesisConfig      count / expression fusion

The headline change from v6 is :class:`BridgeConfig`: v6 had to *choose* per
holdout between a coherent single-sheet morph (great spatial field, but only one
flanking population, so mixture-sensitive metrics suffer) and both-slice
interpolation (right mixture, but incoherent placement). v8's default
``mode="symmetric"`` removes that trade-off by projecting **both** flanking
populations through the *same* entropic-OT barycentric map and drawing them in
the z-interpolated ratio — one coherent sheet that is also the correct mixture.
:class:`DensityConfig` then calibrates the local cell density to the leakage-safe
interpolated density field, which is what the field/density/dice metrics read.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EmbeddingConfig:
    """Cell-state embedding used for the OT cost and cell-type annotation.

    ``method``:
      * ``"pca"``      — local, unsupervised SVD embedding (default; no asset).
      * ``"coexpr"``   — data-derived gene-program prior (SVD of the training
        gene-gene correlation matrix); leakage-safe, no external download.
      * ``"fm_gene"``  — project expression through a *pretrained gene embedding*
        matrix (scGPT / Geneformer / Gene2vec token embeddings, or an H&E-paired
        gene-program matrix): ``cell_embed = X_norm @ W_gene``. Injects the
        foundation model's learned gene-gene relationships. Falls back to
        ``"pca"`` if the matrix is unavailable.
      * ``"concat"``   — concatenate ``pca`` and ``fm_gene``.
      * a name registered via :func:`spatialcpav8.embedding.register_embedder`
        for a full foundation-model encoder (scGPT/Geneformer/UCE, or an H&E
        UNI/CONCH morphology encoder when paired images are provided).
    """

    method: str = "pca"
    n_components: int = 32
    fm_gene_embedding_path: str | None = None
    fm_gene_embedding_dim: int = 0
    standardize: bool = True
    whiten: bool = True
    max_hvg: int = 2000


@dataclass
class TransportConfig:
    """Entropic optimal transport between the two flanking slices.

    The plan ``P`` defines a soft correspondence; the barycentric map it induces
    is what the :class:`BridgeConfig` sheet is built on.
    """

    max_ot_cells: int = 1500      # subsample each flanking slice to this for OT
    epsilon: float = 0.05         # Sinkhorn entropic regularization (peaked plan)
    n_iter: int = 200             # Sinkhorn iterations
    # cost = (1-w)*spatial + w*embedding (both median-normed). Spatial-dominant so
    # the barycentric map stays local (≈ identity when slices are near-identical)
    # while a little molecular guidance keeps matches coherent across tissue shifts.
    embed_weight: float = 0.15
    deshrink: bool = True         # rescale interpolated cloud to interpolated covariance
    deshrink_strength: float = 1.0


@dataclass
class BridgeConfig:
    """How the in-between cell sheet is built from the OT coupling.

    ``mode``:
      * ``"adaptive"`` (default) — route per holdout by the measured OT-map
        displacement between the flanking slices (in cell-spacings): use the
        **smooth morph** when the slices are near-identical (small displacement,
        e.g. thin volumetric z-planes) and **symmetric** both-slice mixing when
        they are distinct tissue (large displacement, where a single-slice morph
        would miss the true intermediate population). The two regimes have opposite
        optima; adaptive picks the right one per holdout and logs the value.
      * ``"smooth_morph"`` — copy the nearest slice's real cells and displace them
        by the *spatially smoothed* OT field: a coherent near-isometric deformation
        that keeps the structure metrics at copy-quality while morphing the global
        footprint toward the interpolated shape. Best when slices are near-identical.
      * ``"symmetric"`` — bidirectional McCann barycentric bridge drawing both
        flanking populations in the ``(1-t):t`` ratio. Best when slices differ.
      * ``"morph"`` — one-sided *un-smoothed* barycentric morph (v6 morph; ablation).
      * ``"interpolate"`` — random real-cell mixing from both slices (v6 interpolate).
      * ``"backbone"`` — the single nearest flanking slice (most conservative).

    ``adaptive_threshold`` is the displacement (in cell-spacings) above which the
    router switches from smooth morph to symmetric mixing. ``smooth_k`` /
    ``smooth_iters`` control the displacement-field smoothing (coherence of the
    morph). ``symmetric_min_fraction`` guards the minority side so a near-integer
    ``t`` still contributes a few far-slice cells (keeps the mixture non-degenerate).
    """

    # Default is ``smooth_morph`` — the coherent smoothed-OT deformation of the
    # single nearest clean slice. Against a single-slice-copy baseline (SpatialZ)
    # it inherits that baseline's structure/density fidelity (it *is* one clean
    # slice) while its non-rigid warp adds the interpolated-field and cell-matched
    # accuracy the copy lacks — so it beats the copy on the large majority of
    # metrics in both the near-identical and distinct-tissue regimes (validated in
    # ``validation/``). ``adaptive`` instead selects a placement per dataset by
    # leakage-safe internal cross-validation (see ``selection.py``).
    mode: str = "smooth_morph"
    # Candidate placements the adaptive selector cross-validates on a held-out
    # *training* slice (leakage-safe); the best reconstruction is used for the
    # target. Order is the tie-break preference.
    adaptive_candidates: tuple = ("smooth_morph", "interpolate", "coherent_mix")
    adaptive_threshold: float = 0.85     # fallback displacement heuristic (unused by CV)
    smooth_k: int = 12
    smooth_iters: int = 3
    symmetric_min_fraction: float = 0.05


@dataclass
class DensityConfig:
    """Local density calibration to the z-interpolated flanking density field.

    The field / density / dice metrics compare the *spatial cell density* of the
    prediction (after a rigid alignment) with the ground truth. The leakage-safe
    estimate of the held-out density is the z-interpolation of the two flanking
    kernel-density fields evaluated in the common (registered) frame. After
    placement we resample the synthesized cells with a weight that nudges their
    empirical density toward that target — directly improving the density-driven
    metrics without touching expression or labels.
    """

    # OFF by default: importance-resampling toward the interpolated field
    # duplicates cells, which perturbs the expression distribution more than it
    # helps the density metrics on real tissue (validated in the ablation harness).
    # Kept as an opt-in knob for datasets with strongly non-stationary density.
    enabled: bool = False
    bandwidth_spacings: float = 2.0   # KDE bandwidth in units of median cell spacing
    strength: float = 0.7             # 0 = off, 1 = fully re-weight to the target field
    resample_cap: float = 1.5         # max resample multiplier for any single cell


@dataclass
class CommunicationConfig:
    """Cell-cell-communication (niche) model used to refine cell-type labels.

    Labels are optimized so the synthesized slice reproduces the flanking slices'
    *neighborhood-enrichment* architecture ``P(neighbor=j | center=i)`` — the
    leakage-safe estimate of the held-out slice's 2D/3D niche structure.
    """

    enabled: bool = True
    k_neighbors: int = 10         # spatial graph degree (matches the evaluator's k)
    n_sweeps: int = 8             # ICM sweeps over all cells
    niche_weight: float = 1.0
    prior_weight: float = 1.0
    composition_weight: float = 1.0
    temperature: float = 1.0
    lr_affinity: float = 0.0


@dataclass
class AnnotationConfig:
    """Cell-type assignment: OT-correspondence anchor + FM prior + niche MRF.

    v8 anchors on the label of each synthesized cell's *real* source cell (the OT
    barycentric bridge copies real cells, so this anchor is strong), then refines
    with a soft prior and the niche MRF. ``classifier`` picks the prior:
      * ``"spatial"`` — z-interpolated spatial vote of type from both flanking
        slices (default; beats a single-slice copy when types vary smoothly in z).
      * ``"prototype"`` / ``"knn"`` — classify in the foundation-model embedding.
    """

    enabled: bool = True
    classifier: str = "spatial"
    knn_k: int = 15
    prototype_temperature: float = 0.5
    anchor_weight: float = 3.0
    fm_weight: float = 1.0
    # v8 pins composition to the interpolated flanking mix by default: the density
    # resampling perturbs the raw copied mix slightly, so re-pinning keeps
    # celltype_composition on target.
    constrain_composition: bool = True
    composition_sinkhorn_iter: int = 50


@dataclass
class SynthesisConfig:
    """Count and expression fusion."""

    # Cell count of the virtual slice: z-interpolated flanking count (emergent).
    count_mode: str = "interpolate"   # "interpolate" | "lower" | "upper" | "mean"
    # Expression source for each synthesized cell:
    #   "endpoint"  — copy the real profile of the source cell (max variance,
    #                 preserves gene-gene structure). Default.
    #   "transfer"  — nearest same-type training cell in embed space (denoises).
    #   "blend"     — mix endpoint and transferred (transfer_alpha on transferred).
    expression_mode: str = "endpoint"
    transfer_k: int = 1
    transfer_alpha: float = 0.5
    seed: int = 42


@dataclass
class SpatialCPAv8Config:
    """Top-level configuration bundling every stage."""

    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    transport: TransportConfig = field(default_factory=TransportConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    density: DensityConfig = field(default_factory=DensityConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    seed: int = 42
