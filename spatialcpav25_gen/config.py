"""Single source of truth for every constant in SpatialCPA-v25-Gen.

Convention 1 of ``CLAUDE.md``: there are no magic numbers anywhere else in the package.
A new constant becomes a documented :class:`Config` field with a default; a task that needs
to change one does it here, once, where every other task can see it.

The dataclass is frozen. Derive variants with :meth:`Config.replace`, which validates the
result, rather than mutating an existing instance.

Field groups follow the build order in ``specs/``: general, data, text (T02), noise (T03),
field and retrieval (T04), layout (T05), selection gates (T09/T10), SEFL (T07),
expression (T06), losses, metric-aware losses (T08), inference (T09), training.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "DECODERS",
    "DEVICES",
    "EXPR_MODES",
    "GRANULARITIES",
    "HOLDOUT_MODES",
    "LAYOUT_MODES",
    "METRIC_DISTRIBUTION_KINDS",
    "MU_LINKS",
    "POTTS_UPDATES",
    "PRIOR_MODES",
    "ROTATION_BIASES",
    "SUMMARY_FALLBACKS",
    "TEXT_EMB_MODES",
    "TEXT_POOLINGS",
    "Config",
    "ConfigError",
]

# --------------------------------------------------------------------------------------
# Allowed values for the string-valued gates. These live beside `Config` because they are
# part of its definition; nothing outside this module may hard-code one of these strings.
# --------------------------------------------------------------------------------------

LAYOUT_MODES: Final[frozenset[str]] = frozenset({"field", "hybrid", "resample"})
SUMMARY_FALLBACKS: Final[frozenset[str]] = frozenset({"none", "ortholog"})
MU_LINKS: Final[frozenset[str]] = frozenset({"exp", "softplus"})
ROTATION_BIASES: Final[frozenset[str]] = frozenset({"uniform", "axial"})
PRIOR_MODES: Final[frozenset[str]] = frozenset({"correlated", "iid"})
POTTS_UPDATES: Final[frozenset[str]] = frozenset({"gibbs", "icm"})
EXPR_MODES: Final[frozenset[str]] = frozenset({"zinb-flow", "cross-mix", "auto-blend"})
TEXT_EMB_MODES: Final[frozenset[str]] = frozenset({"medcpt", "lookup"})
DECODERS: Final[frozenset[str]] = frozenset({"zinb", "zigamma", "gaussian"})
DEVICES: Final[frozenset[str]] = frozenset({"auto", "cpu", "cuda"})
TEXT_POOLINGS: Final[frozenset[str]] = frozenset({"cls", "mean"})
HOLDOUT_MODES: Final[frozenset[str]] = frozenset({"alternating", "consecutive"})
GRANULARITIES: Final[frozenset[str]] = frozenset({"single-cell", "binned"})
METRIC_DISTRIBUTION_KINDS: Final[frozenset[str]] = frozenset({"sinkhorn", "mmd"})


class ConfigError(ValueError):
    """Raised when a :class:`Config` is internally contradictory or out of range."""


@dataclass(frozen=True)
class Config:
    """Every tunable constant in the system, in one frozen object.

    Notes
    -----
    Fields marked *provisional* have no value fixed by ``specs/``; the task named in the
    comment sets the real default when it lands. They exist now so that no task has to
    write the constant inline (Convention 1).
    """

    # ----------------------------------------------------------------------------------
    # general
    # ----------------------------------------------------------------------------------
    seed: int = 0
    """Global base seed. Stochastic functions take their own explicit seed; this is the
    default they are derived from at the top level (Convention 3)."""

    device: str = "auto"
    """One of ``DEVICES``. ``auto`` resolves to cuda when available, else cpu."""

    debug_shapes: bool = False
    """Enable the runtime shape assertions that every tensor-returning function guards
    with ``if cfg.debug_shapes:`` (Convention 2). Off by default: they cost time."""

    # ----------------------------------------------------------------------------------
    # data (T01)
    # ----------------------------------------------------------------------------------
    coord_key: str = "spatial"
    """``adata.obsm`` key holding (x, y) or (x, y, z) coordinates in micrometres."""

    z_key: str = "z"
    """``adata.obs`` column holding section depth, used when ``coord_key`` is 2-D."""

    celltype_key: str = "cell_type"
    """``adata.obs`` column holding cell type labels."""

    region_key: str | None = "region"
    """``adata.obs`` column holding anatomical region labels; ``None`` if the dataset has
    none. A named-but-absent column is an error, not a silent skip (Convention 6)."""

    counts_layer: str | None = None
    """``adata.layers`` key holding raw counts; ``None`` means ``adata.X`` holds them.
    When set, the matrix is additionally required to be integer-valued."""

    min_cells_per_section: int = 50
    """Sections with fewer cells than this are rejected by ``validate_volume``."""

    thickness_key: str = "section_thickness"
    """Key for the physical slab thickness in micrometres, looked up in ``adata.uns``
    (scalar or per-section mapping) and then ``adata.obs`` (per-cell column). Absent =>
    thickness defaults to the volume's median spacing and is flagged as assumed (T01)."""

    section_key: str | None = None
    """``adata.obs`` column giving section identifiers. ``None`` groups cells into
    sections by their distinct ``z`` value, which is the definition of a section here."""

    min_sections_per_volume: int = 3
    """A volume with fewer sections cannot support leave-one-section-out or a flanking
    pair, so ``validate_volume`` rejects it."""

    holdout_consecutive_k: int = 3
    """Length of the contiguous held-out run in ``split_holdout(mode="consecutive")``.
    T10's regimes ``consecutive-3`` / ``consecutive-5`` are this field set to 3 / 5."""

    small_volume_n_sections: int = 8
    """Below this many *training* sections a volume counts as small, which constrains the
    z-direction Fourier bands (see ``max_fourier_bands_z_small_volume``)."""

    max_fourier_bands_z_small_volume: int = 4
    """Largest ``fourier_bands_z`` permitted on a small volume. Higher-frequency z basis
    functions are unconstrained by so few sections and overfit the section positions."""

    metric_knn_k: int = 10
    """Neighbours in the kNN graph underlying Moran's I / Geary's C. 10 is what
    ``benchmark-pbya-v3`` uses, and the published numbers must stay comparable."""

    # ----------------------------------------------------------------------------------
    # text embeddings (T02)
    # ----------------------------------------------------------------------------------
    text_model: str = "ncbi/MedCPT-Query-Encoder"
    """Frozen sentence encoder used for gene / cell type / region descriptors."""

    text_dim_in: int = 768
    """Width of the frozen text encoder's output."""

    gene_emb_dim: int = 128
    """Width of the learned gene embedding ``e_g``."""

    ctx_emb_dim: int = 64
    """Width of the learned cell-type and region embeddings."""

    text_cache_dir: str = ".cache/spatialcpav25_gen_text"
    """On-disk cache keyed by sha256(model name + descriptor). A cache hit must avoid
    loading the transformer at all, which is what keeps CPU test runs fast."""

    residual_gate_warmup_frac: float = 0.3
    """Fraction of training over which the free-residual gate gamma anneals 0 -> 1."""

    text_pooling: str = "cls"
    """One of ``TEXT_POOLINGS``. MedCPT's query encoder is trained with CLS pooling, so
    that is the default; ``mean`` is the alternative (see SPEC_QUESTIONS C3)."""

    text_batch_size: int = 32
    """Descriptors encoded per forward pass of the frozen text encoder."""

    text_max_length: int = 512
    """Token budget per descriptor; longer descriptors are truncated."""

    gene_meta_path: str = "resources/gene_meta.parquet"
    """Local gene-metadata table (columns: symbol, full_name, summary, aliases,
    ensembl_id) that ``gene_descriptor`` reads. Built once by ``build_gene_meta``; never
    fetched at train or test time."""

    text_allow_network: bool = False
    """Permit ``build_gene_meta`` to query mygene.info. Off by default: nothing in the
    training or test path may reach the network (T02 "Do NOT"), so going online is an
    explicit, one-off opt-in when the table is built."""

    mygene_species: str = "mouse"
    """The **one** species ``build_gene_meta`` queries, and the one a gene-metadata table describes.

    Was ``"human,mouse"``, which is not a filter a symbol-keyed table can honour: the same symbol
    exists in both organisms, so a two-species request lets whichever hit the API returned first
    win. Measured on a 1138-symbol mouse panel, that build produced 389 mouse rows and 744 from
    ground squirrel, mink, ferret and falcon. ``resolve_species`` now refuses anything but a single
    species, and T10's mouse and human datasets get **one table each** at different
    ``gene_meta_path``s — which is also what lets ``load_gene_meta(path, species=...)`` refuse a
    table of the wrong organism.

    A name from ``spatialcpav25_gen.data.text.SPECIES_TAXID`` or a bare NCBI taxid as digits."""

    gene_summary_fallback: str = "ortholog"
    """One of ``SUMMARY_FALLBACKS``: what to do when a gene has no summary in its own species.

    ``"none"`` leaves it absent. ``"ortholog"`` fetches the orthologue's summary from
    ``gene_summary_ortholog_species`` and stores it **with its provenance recorded** —
    ``summary_source``, ``summary_source_taxid`` and ``summary_source_symbol`` — so a descriptor can
    say whose summary it is and every count can be split by source.

    On by default because the alternative measured worse than useless: mouse NCBI summaries cover
    **148/1138 (13%)** of a real mouse panel (SPEC_QUESTIONS B20, confirmed genuinely sparse — 5
    symbols lose one to hit selection), which leaves 87% of descriptors as bare names and gives the
    open-vocabulary claim almost nothing to work with.

    **What it costs, and why the provenance columns are not optional.** Human coverage on the same
    panel is ~93%, so with this on most descriptors carry *human* text and the open-vocabulary claim
    becomes substantially a claim about human summaries transferred to a mouse model. That is
    defensible — orthologous function is largely conserved, and it is what a human reader would do —
    but it is a different claim from the one ``design/v23_design.md`` states. T10's E1 must report
    **both arms**, and with ``summary_source`` recorded it can: the native-only arm is a filter
    on the column, not a second build."""

    gene_summary_ortholog_species: str = "human"
    """Species whose summary stands in under ``gene_summary_fallback="ortholog"``.

    Human because it is the best-annotated genome and the orthologue relation to mouse is the
    best-curated one. The orthologue is resolved through mygene's ``homologene`` field and must be
    **1:1**; it is never resolved by upper-casing the symbol, which is precisely the mistake that
    silently produced an all-human table for a mouse panel (SPEC_QUESTIONS B19)."""

    distill_hidden: int = 256
    """Hidden width of the text -> free-residual distillation head ``psi``
    (768 -> distill_hidden -> out_dim)."""

    text_diag_knn_k: int = 10
    """Neighbours used by ``text_embedding_diagnostics``: both the k nearest text
    neighbours whose purity is reported and the degree of the gene-gene co-expression
    graph the modules are found on."""

    text_diag_leiden_resolution: float = 1.0
    """Resolution of the Leiden partition of the gene-gene co-expression graph."""

    text_diag_leiden_iterations: int = 2
    """Leiden refinement passes. Fixed rather than -1 (run to convergence) so the
    diagnostic is reproducible."""

    text_diag_max_pairs: int = 200_000
    """Cap on the gene pairs entering the text/co-expression Spearman. Above it a seeded
    subsample of pairs is used; G = 20k genes would otherwise be 2e8 pairs."""

    # ----------------------------------------------------------------------------------
    # noise field (T03)
    # ----------------------------------------------------------------------------------
    matern_nu: float = 1.5
    """Matern smoothness of the 3D Gaussian random field prior."""

    n_rff: int = 4096
    """Number of random Fourier features M approximating the Matern kernel."""

    ell_xy: float = 100.0
    """In-plane correlation length in micrometres; fitted from data and calibrated at
    inference (T09), never hand-set for a published run."""

    ell_z: float = 100.0
    """Along-z correlation length in micrometres."""

    latent_dim: int = 64
    """d_h: width of the per-cell latent, i.e. the number of GRF channels."""

    grf_chunk_points: int = 1024
    """Query points per chunk when evaluating the field. Materialising the full (N, M)
    feature matrix at N = 1e6, M = 4096 would need 16 GB (SPEC_QUESTIONS B9), but the
    binding constraint is cache, not memory: the ``(chunk, M)`` block of cosines is
    written, read, and read again, so a chunk that does not fit in cache makes the query
    memory-bound and about three times slower. 1024 x n_rff floats is 16 MB at the default
    M = 4096; raise it only together with a measurement."""

    variogram_n_pcs: int = 50
    """Expression PCs the empirical semivariogram is computed on when fitting ``ell``
    (T03). Distinct from ``expr_pca_dim`` (32), which is a model input width; this is a
    statistic's resolution and the spec fixes it at 50."""

    variogram_n_bins: int = 20
    """Distance bins in the in-plane semivariogram."""

    variogram_max_lag_frac: float = 0.25
    """Largest lag entering the in-plane variogram fit, as a fraction of the shorter
    in-plane extent. Standard variogram practice: beyond a quarter to a half of the
    domain, the estimator is dominated by a handful of long pairs and by whatever
    large-scale trend the tissue has, which biases the fitted length-scale upwards."""

    variogram_max_cells_per_section: int = 800
    """Cells subsampled per section before forming all pairs. The pair count is quadratic,
    and 800 cells already give ~320k pairs per section."""

    variogram_min_pairs_per_bin: int = 32
    """Bins holding fewer pairs than this are dropped as too noisy to fit against."""

    variogram_n_ell_grid: int = 128
    """Log-spaced candidate length-scales scanned by the variogram fit. The two linear
    parameters (nugget, sill) are solved in closed form at each candidate, so this is the
    whole optimiser: no starting point, no local minima. It is also the fit's resolution -
    128 points over the ~200x search range are 4% apart, comfortably finer than the 25%
    tolerance T03's GATE 1 and T09's calibrator report against; at 48 the 12% quantisation
    was visible as the fitted value jumping between two adjacent grid points."""

    variogram_ell_min_factor: float = 0.5
    """Lower end of the length-scale search, as a multiple of the median nearest-neighbour
    distance (in-plane) or the median section spacing (along z). Below it the data cannot
    resolve the correlation length at all."""

    variogram_ell_max_factor: float = 2.0
    """Upper end of the length-scale search, as a multiple of the largest lag used."""

    variogram_z_grid_size: int = 8
    """Side of the coarse in-plane grid whose per-cell means carry the between-section
    comparison. Cells in different sections have no correspondence, so ``ell_z`` is fitted
    on grid-cell means rather than on cells."""

    variogram_z_min_cells_per_cell: int = 5
    """Grid cells with fewer cells than this are excluded from the along-z variogram."""

    variogram_min_saturation: float = 0.75
    """Fraction of the fitted sill the empirical variogram must reach at its largest lag
    for the fitted length-scale to be an interpolation rather than an extrapolation. Below
    it the fit warns: a stack of nine 50 um sections spans 400 um, which is not enough to
    watch a 200 um correlation decay away along z, and the fitted ``ell_z`` will read
    high."""

    variogram_min_structured_frac: float = 0.05
    """Smallest fitted sill (as a share of total PC variance) that counts as spatial
    structure. Below it, fitting a correlation length is fitting noise, and
    ``fit_lengthscale_from_sections`` raises rather than returning an artefact of the
    search grid (Convention 6)."""

    # ----------------------------------------------------------------------------------
    # anatomical field (T04)
    # ----------------------------------------------------------------------------------
    triplane_res_xy: int = 256
    """Resolution of the in-plane axis of every feature plane."""

    triplane_res_z: int = 32
    """Resolution of the z axis of the XZ / YZ feature planes."""

    triplane_channels: int = 32
    """Channels C per feature plane."""

    n_plane_orientations: int = 4
    """Triplane sets, each at a fixed maximally-separated rotation.

    **Stays at 4, and that is a measurement, not a default nobody revisited.** GATE 2's first
    escalation on an oblique-parity failure is raising this to 8; T04 ran it. Doubling the
    ensemble bought **+0.00086** on the gate ratio (0.8858 -> 0.8867 on the superseded
    single-plane criterion) for **2x the feature-plane memory** — 21 M parameters against
    10.5 M at the default resolutions, and ~60% more wall clock per probe. The deficit it was
    meant to address turned out not to be directional at all (`reports/gate2.md`, escalation
    section), so the memory buys nothing. Raise it only against a *new* measurement showing a
    directional deficit, not on the strength of the spec naming it as a remedy."""

    fourier_bands_xy: int = 8
    """Fourier positional-encoding bands in-plane."""

    fourier_bands_z: int = 2
    """Fourier positional-encoding bands along the sampling axis. Deliberately low: with
    4-9 sections, high z frequencies overfit the section positions."""

    rotation_aug: bool = True
    """Apply the whole-volume rotation augmentation during training."""

    rotation_bias: str = "uniform"
    """One of ``ROTATION_BIASES``. ``uniform`` draws from Haar measure on SO(3);
    ``axial`` is the anatomically plausible bias of ``design/v23_sectioning_equivariance.md``
    §2.1(a) - a free spin about the sectioning axis plus a tilt of at most
    ``rotation_bias_max_tilt_deg``, i.e. the poses a block actually gets mounted in."""

    rotation_bias_max_tilt_deg: float = 30.0
    """Largest tilt away from the sectioning axis under ``rotation_bias="axial"``."""

    field_mlp_hidden: int = 256
    """Hidden width of the field MLP."""

    field_mlp_layers: int = 3
    """Linear layers in the field MLP (the spec's "3-layer MLP"). Must be >= 2: one layer
    would make the field a linear read-out of the triplane features."""

    tv_z_weight: float = 1e-3
    """Weight of the total-variation-along-z penalty on the unrotated orientation."""

    field_dim: int = 128
    """d_f: width of the field feature F(x, y, z). T04 confirms T01's provisional 128 as the
    real default: it is twice ``retrieval_ctx_dim`` and half ``field_mlp_hidden``, so the
    field carries the larger share of the conditioning signal, which is what the design
    intends (the field carries anatomy, retrieval carries realism)."""

    # ----------------------------------------------------------------------------------
    # retrieval (T04)
    # ----------------------------------------------------------------------------------
    retrieval_k: int = 32
    """Neighbouring real cells K retrieved per query point."""

    retrieval_z_window: float = 3.0
    """Candidate pool half-width along z, in units of median section spacing.

    The *absolute* term of the two-term bound; see ``retrieval_z_window_gap_factor`` for
    why one term alone is not enough."""

    retrieval_z_window_gap_factor: float = 2.0
    """Candidate pool half-width along z, in units of **this query's own gap** to its
    nearest admissible section. The *relative* term; the effective bound is

        |z_j - z_p| <= max(retrieval_z_window x median_spacing,
                           retrieval_z_window_gap_factor x gap_to_nearest(p))

    ``median_spacing`` is a statistic of the whole stack, so the absolute term alone sizes
    the window off tissue that may not be there. After a holdout, a dropped section or a
    damaged one, the local gap at some depths far exceeds the median: on the GATE 2 fixture
    under a ``consecutive``-3 holdout the training stack is z = 0, 200, 250, 300, 350, 400
    um, four of five gaps are 50 um so the median stays 50 and the window stays 150 — while
    the section at z = 0 is 200 um from its nearest neighbour. All 13500 of its cells then
    retrieve **nothing**, train against a fully masked attention row, and held-out
    reconstruction at z = 150 um inverts to R^2 = -0.17 (against -0.19 for no retrieval
    at all). Widening the *training* window to cover the real gap takes the same depth to
    +0.36.

    The relative term makes the bound per-query and guarantees a non-empty pool by
    construction: the nearest admissible section is at ``gap_to_nearest`` and the factor is
    ``>= 1``, so it is always admitted. On a regular stack ``gap_to_nearest`` is at most one
    spacing, ``2 x 1 <= 3``, the absolute term wins and behaviour is unchanged — which is
    what ``test_gap_relative_window_is_identity_on_a_regular_stack`` asserts bitwise.

    2.0 is a placeholder chosen so a query one gap from its evidence also reaches roughly
    the next section along, not the whole stack. It has **not** been swept: the sweep
    belongs in T09's config selection, on internal LOSO over training sections, because a
    window chosen against held-out sections would be a leak. See SPEC_QUESTIONS C1c."""

    section_dropout_p: float = 0.3
    """Probability of dropping the nearest section from the candidate pool during
    training - the gap-aware curriculum."""

    retrieval_w_z: float = 1.0
    """Weight of the z-proximity term in the retrieval score. ``0`` is ablation A5, which
    reproduces the competing method's omission."""

    retrieval_w_niche: float = 1.0
    """Weight of the niche-similarity term in the retrieval score."""

    retrieval_ctx_dim: int = 64
    """d_ctx: width of the retrieval cross-attention output. T04 confirms 64 as the real
    default - four heads of width 16, and half the field's width."""

    retrieval_n_heads: int = 4
    """Heads in the retrieval cross-attention. T04 confirms 4 as the real default; the
    score has three named terms (in-plane distance, z proximity, niche similarity) and the
    heads are what let the attention weight them differently per query."""

    retrieval_exclude_source_section: bool = True
    """Exclude a query cell's **own** section from its retrieval candidate pool.

    Load-bearing for GATE 2 (SPEC_QUESTIONS C1a): without it, a cell evaluated on a 90
    degree query plane retrieves in-plane neighbours a few micrometres away *inside its own
    section*, the oblique plane becomes trivially easy, and the gate passes while hiding
    exactly the equivariance failure it exists to detect. ``False`` is the setting
    ``test_source_section_exclusion_changes_oblique_R2`` measures against."""

    retrieval_score_temperature: float = 1.0
    """Temperature of the softmax turning retrieval scores into donor weights. The score's
    leading term is an in-plane distance in units of the median nearest-neighbour distance,
    so a temperature of 1 means "one neighbour spacing costs one nat"."""

    retrieval_candidates_per_section: int = 64
    """In-plane nearest candidates taken from each allowed section before scoring.

    The score is monotone decreasing in in-plane distance at fixed z and niche, so the
    global top-K is contained in the per-section top-``retrieval_candidates_per_section``
    union unless the niche term reorders more than this many cells within one section.

    It must comfortably exceed ``retrieval_k`` (``validate`` enforces ``>=``), and that is
    load-bearing rather than a safety margin: when only two sections are admissible — a
    held-out run, the gap-aware dropout, a wide-gap inference — a cap of ``retrieval_k / 2``
    makes the union exactly ``K`` candidates, the top-K selects all of them, and **the score
    stops choosing anything at all**. The z term is then silently inert in precisely the
    wide-gap regime it exists for. Found by GATE 2's G2.3, which measured the ablation as a
    no-op until the cap was raised."""

    retrieval_query_chunk: int = 4096
    """Query points scored per chunk. The candidate block is
    ``(chunk, n_sections x retrieval_candidates_per_section, niche_dim)``, which at the
    defaults is a few million floats; chunking keeps it off the heap for whole-volume
    queries."""

    niche_knn_k: int = 8
    """Neighbours defining the *first* niche radius. The radius is the distance to the
    k-th in-plane nearest neighbour rather than a fixed micrometre value, so the niche
    transfers across datasets with different cell densities (T04 §2)."""

    niche_n_scales: int = 3
    """Spatial scales the niche composition is computed at (the spec's "3 spatial
    scales")."""

    niche_scale_factor: float = 2.0
    """Ratio between consecutive niche radii: scale ``s`` uses the
    ``niche_knn_k * niche_scale_factor ** s``-th nearest neighbour's distance."""

    section_dropout_max_sections: int = 1
    """Sections dropped from the candidate pool when the gap-aware curriculum fires. One
    is the spec's "the nearest section(s)"; raising it makes the curriculum harsher."""

    expr_pca_dim: int = 32
    """Expression PCs used for neighbour tokens, the GATE 2 probe target, and the
    Sinkhorn basis (T08). GATE 2 specifies the top 32."""

    gate2_min_cells_per_angle: int = 500
    """Floor on the common cell count GATE 2's angles are subsampled to (SPEC_QUESTIONS
    C1b). Below it the fixture's slabs are thickened and the gate re-run - the floor is
    never lowered and no angle is ever dropped, because both would make the oblique-parity
    ratio partly a statement about sample size."""

    # ----------------------------------------------------------------------------------
    # layout (T05)
    # ----------------------------------------------------------------------------------
    layout_mode: str = "field"
    """One of ``LAYOUT_MODES``. ``resample`` reuses real flanking coordinates and is the
    previous version's behaviour, kept as the no-regression fallback."""

    potts_beta: float = 0.5
    """Potts coupling for cell-type mark smoothing. Fitted by ``fit_potts_beta``; this is
    only the starting value."""

    potts_iters: int = 2
    """Rounds of iterated conditional modes in the Potts smoothing."""

    layout_n_mc: int = 4096
    """Monte Carlo points for the slab-volume intensity integral."""

    potts_knn_k: int = 8
    """Neighbours in the kNN graph used by the Potts smoothing."""

    layout_max_proposal_factor: int = 20
    """Proposal budget for the Strauss sampler, as a multiple of the target count N.
    Exhausting it means the intensity and the repulsion are inconsistent."""

    layout_envelope_slack: float = 1.1
    """Safety factor on the rejection-sampling envelope (max intensity over an MC
    sample)."""

    swd_polish_steps: int = 200
    """Sliced-Wasserstein polish steps in ``layout_mode="hybrid"``."""

    repulsion_r0_percentile: float = 1.0
    """Percentile of pooled nearest-neighbour distances defining the hard-core radius r0.
    The spec's 5.0 makes generated layouts strictly more regular than the tissue, which
    fights the pair-correlation acceptance test (SPEC_QUESTIONS B6); 1.0 is the default
    and 5.0 remains selectable."""

    layout_mlp_hidden: int = 128
    """Hidden width of the intensity MLP. Half ``field_mlp_hidden``: the head reads a
    ``field_dim`` feature that already carries the anatomy and only has to turn it into
    ``n_types`` scalars, so it is the smaller network of the two."""

    layout_mlp_layers: int = 3
    """Linear layers in the intensity MLP (T05's ``MLP_c``). Must be >= 2, for the same
    reason as ``field_mlp_layers``: one layer makes the log-intensity a linear read-out of
    the field feature and the Fourier encoding."""

    layout_fit_steps: int = 300
    """Full-batch Adam steps taken by ``fit_intensity_head``. The Poisson NLL is a convex
    function of the log-intensity and the head is small, so this is a fit rather than a
    training run; the model's real intensity head is trained jointly by T06's trainer."""

    layout_fit_lr: float = 0.01
    """Adam learning rate for ``fit_intensity_head``. Larger than ``Config.lr`` because
    this fits a few thousand parameters on a fixed batch rather than a whole model."""

    layout_intensity_eps: float = 1e-12
    """Floor on lambda, in cells per micrometre^3, before taking a logarithm. A cell type
    whose intensity has been driven to exactly zero somewhere would otherwise make the
    Poisson NLL infinite at any observed cell of that type."""

    layout_proposal_batch: int = 256
    """Proposals drawn per vectorised batch of the Strauss sampler. The intensity call is
    vectorised over the batch; the repulsion test is then sequential over the survivors,
    because acceptance depends on the points accepted before it."""

    pcf_n_bins: int = 24
    """Distance bins of the empirical pair-correlation function ``g(r)``."""

    pcf_max_r_factor: float = 6.0
    """Largest lag of ``g(r)``, in multiples of the median nearest-neighbour distance. It
    has to cover ``3R`` — the range T05's ``test_pcf_matches_real`` is stated over — with
    ``R`` (the lag at which the empirical ``g`` first reaches 1) typically one to two
    nearest-neighbour distances."""

    repulsion_gamma_grid_size: int = 9
    """Candidate values in the 1-D search for the Strauss interaction parameter gamma. The
    grid is the whole optimiser: each candidate costs one simulated realisation, and gamma
    enters ``g(r)`` smoothly, so a grid is more robust than a gradient-free minimiser on a
    Monte-Carlo objective."""

    repulsion_gamma_min: float = 0.05
    """Smallest candidate gamma; the grid is ``linspace(repulsion_gamma_min, 1.0)``. Zero
    is excluded because ``phi = -log(gamma)`` is infinite there, which is the hard core
    ``r0`` already provides."""

    repulsion_fit_cells: int = 600
    """Points per simulated realisation inside the gamma fit. Enough for a stable ``g(r)``
    over the fitted range while keeping the fit to a second or two: the sampler is
    sequential in the accepted points, so its cost grows faster than linearly."""

    potts_update: str = "gibbs"
    """One of ``POTTS_UPDATES``. How the Potts mark smoothing updates a label.

    T05 writes "``potts_iters`` rounds of **ICM**". ICM takes the ``argmax``, i.e. it seeks
    the *mode* of the Potts posterior — and measured on the synthetic fixture its very first
    sweep, at ``beta = 0``, already raises neighbourhood purity from 0.49 to 0.77 against a
    tissue value of 0.69 and takes a 2%-prevalence cell type to **exactly zero cells**. That
    is T05's own "Do NOT" (do not smooth types so hard that rare types vanish) happening
    before the coupling is even switched on, and it leaves ``beta`` nothing to fit. ``gibbs``
    *samples* the same conditional instead of maximising it, which keeps the marks a draw
    from the marked point process the section is supposed to be, makes purity a smooth,
    monotone and therefore fittable function of ``beta``, and preserves rare types.
    ``icm`` is kept, selectable, and exercised by a test as the negative control that
    documents this (SPEC_QUESTIONS B11)."""

    potts_beta_grid_size: int = 9
    """Candidate values in the 1-D search for the Potts coupling beta, ``0`` included."""

    potts_beta_min: float = 0.02
    """Smallest non-zero candidate beta. The grid is ``0`` followed by
    ``potts_beta_grid_size - 1`` values spaced **geometrically** from here to
    ``potts_beta_max``: beta enters an exponent, so purity responds to it on a log scale,
    and a linear grid puts all its resolution where the answer has already saturated. The
    rare-type constraint bites between 0.05 and 0.3 on the fixture, which a linear grid of
    the same size steps straight over."""

    potts_beta_max: float = 2.0
    """Largest candidate beta. At the default ``potts_knn_k=8`` a coupling of 2 makes eight
    agreeing neighbours worth 16 nats against the unary log-intensity, which is already far
    into the over-smoothed regime the fit exists to stay out of."""

    potts_rare_prevalence: float = 0.05
    """A cell type below this expected prevalence is *rare*, and the Potts fit is not
    allowed to erase it. 5% is ``RARE_CELLTYPE_FRAC`` in
    ``benchmark-pbya-v3/src/bench3/config.py``, i.e. the benchmark's own definition, so the
    types this protects are exactly the ones ``rare_celltype_localization`` scores."""

    potts_rare_retention: float = 0.5
    """Smallest share of a rare type's *expected* count that must survive the smoothing, in
    **every** training section. T05 states the 0.5 as an acceptance criterion
    (``test_rare_types_survive``); making it a constraint on the fit is what turns "do not
    smooth so hard that rare types vanish" into something the code guarantees rather than
    something a reviewer hopes for. Per-section rather than pooled because a pooled
    constraint is satisfied on average while a single section is wiped out, and because a
    fit that sits exactly on a pooled threshold has a coin-flip's chance of missing it on
    the next draw."""

    swd_n_projections: int = 32
    """Random 1-D projections averaged per sliced-Wasserstein polish step
    (``layout_mode="hybrid"``)."""

    swd_polish_step: float = 0.5
    """Fraction of the sliced-Wasserstein displacement applied per polish step. Below 1 so
    the polish is a flow toward the flanking marginals rather than a single snap onto them,
    which would discard the repulsion the field sampler just imposed."""

    # ----------------------------------------------------------------------------------
    # selection gates (T09 select_config; the ablation switches for T10)
    # ----------------------------------------------------------------------------------
    prior_mode: str = "correlated"
    """One of ``PRIOR_MODES``. ``iid`` is ablation A1."""

    expr_mode: str = "zinb-flow"
    """One of ``EXPR_MODES``. ``cross-mix`` is the previous version's expression path."""

    text_emb_mode: str = "medcpt"
    """One of ``TEXT_EMB_MODES``. ``lookup`` is ablation A3."""

    decoder: str = "zinb"
    """One of ``DECODERS``. ``gaussian`` is ablation A6; ``zigamma`` is for
    intensity-valued (non-count) assays."""

    repulsion: bool = True
    """``False`` gives a pure Poisson layout: ablation A4."""

    ceiling_n_draws: int = 8
    """Independent draws from the synthetic fixture's *known* generative law used to measure
    each metric's **achievable ceiling** (T10 §1, added after T05 measured one).

    Every target metric compares a generated section with a real one, so a perfect model
    still scores below the top of the scale: a different *realisation* of the same law is
    not the same point cloud. Measured for ``celltype_localization``, the held-out section
    scored against itself reaches 0.92 while an independent draw from the true law reaches
    0.72 — so a raw 0.71 is 99% of what is achievable, not a mediocre result. The ceiling is
    a Monte-Carlo quantity and needs enough draws to carry a standard deviation; 8 is the
    same budget as ``n_uncertainty_samples`` and is a floor, not a target."""

    # ----------------------------------------------------------------------------------
    # SEFL (T07)
    # ----------------------------------------------------------------------------------
    sefl_every_n_steps: int = 3
    """Apply the SEFL consistency losses on every n-th step (cost control)."""

    sefl_patch_cells: int = 2000
    """Points sampled per plane patch for the consistency losses."""

    sefl_n_line_points: int = 256
    """Points sampled along a plane-plane intersection segment."""

    sefl_min_angle_deg: float = 20.0
    """Minimum dihedral angle of a random plane pair."""

    sefl_max_angle_deg: float = 160.0
    """Maximum dihedral angle of a random plane pair."""

    thickness_ratio: int = 3
    """A thick section is this many thin ones; the coarse-graining ratio in L_thick."""

    sefl_min_stratum_cells: int = 20
    """Skip a (cell type, region) stratum in L_prog with fewer cells than this on either
    side - the statistic is meaningless below it."""

    sefl_genes_per_step: int = 64
    """G' the consistency losses compare decoder parameters and expression on.

    Separate from ``genes_per_step``, and smaller, purely as cost control - ``specs/07`` 5
    caps the SEFL block's wall-clock overhead at 60 % per epoch and the gene-conditioned
    decoder is where a SEFL step spends its time (it is evaluated at every *(point, gene)*
    pair, on two branches, three times per block). Every term here is a mean or a covariance
    over genes, all of them are estimated fine from a subsample, and the subsample is redrawn
    every step, so over a run every gene is constrained. Measured on the reduced fixture: at
    the whole 200-gene panel the block costs **+62 %** per epoch, at 64 genes **+34 %**."""

    sefl_ema_teacher: bool = True
    """Evaluate the second branch of every consistency loss through the **EMA teacher**,
    with a stop-gradient.

    The anti-collapse mechanism, and the one field in this block that is not a cost control.
    A symmetric consistency loss has the trivial minimiser "constant field" and will find it;
    the teacher breaks the symmetry, because a constant student no longer satisfies a loss
    whose target is a lagging average of its own past. ``False`` makes both branches the live
    student with gradients on **both** sides and is the deliberate negative control of
    ``specs/07``: ``tests/test_sefl.py::test_no_collapse_negative_control_fails`` asserts that
    the no-collapse criterion *fails* with it off, which is what documents the asymmetry as
    load-bearing rather than decorative. Never ship a run with this off."""

    sefl_ramp_frac: float = 0.2
    """Fraction of training over which the SEFL weights ramp linearly from zero to their
    configured values, starting at the end of ``sefl_warmup_frac``.

    ``specs/07`` §5 says "ramp ... linearly to their configured values" without saying over
    what horizon, so this is T07's choice and it is a field rather than a literal for that
    reason. One warm-up length: the ramp is as long as the reconstruction-only phase, so the
    consistency terms reach full weight at 40 % of the run and spend the majority of training
    at their configured strength."""

    sefl_collapse_min_steps: int = 100
    """The collapse alarm stays silent for this many optimiser steps, whatever the ramp says.

    The alarm is about a variance that **drops**, so it needs a model that has had one. Two
    gates are needed and one is not enough: the ramp gate ("the SEFL weights are live") is a
    *fraction* of the run, so on a short run it opens after a handful of steps and the alarm
    fires on an untrained decoder that predicts near the panel mean for every cell — measured
    at **0.008** of the real variance at step 3 of a four-step run, which is initialisation,
    not collapse. An alarm that fires routinely on healthy short runs is one nobody reads on
    the long run where it matters. 100 steps is below the first alarm of every real arm
    measured at T07 (earliest: step 110), so it costs no sensitivity."""

    sefl_collapse_warn_fraction: float = 0.25
    """Collapse alarm: warn when the mean per-gene variance of **generated** expression falls
    below this fraction of the real cells'. ``specs/07`` §2's 25 %, verbatim. The comparison
    is drawn counts against drawn counts, on the same genes and the same cells - a variance of
    ``mu`` against a variance of counts would be below it for a healthy model, because
    ``Var[x] = Var[mu] + E[NB noise]``."""

    sefl_consistency_ratio_warn: float = 0.5
    """Warn when ``sum(weighted consistency terms) / sum(weighted reconstruction terms)``
    exceeds this. ``specs/07`` §5: "Reconstruction terms must dominate throughout"."""

    sefl_min_segment_nn_multiple: float = 3.0
    """Skip an intersection segment shorter than this many median nearest-neighbour
    distances (``specs/07`` §2, step 2). Below it the sampled points are all inside one
    cell's neighbourhood and the agreement they assert is trivially satisfied."""

    sefl_rejection_max_rounds: int = 16
    """Rounds of rejection sampling ``Plane.sample_points`` may use to fill its request
    before it raises. Each round proposes four times the shortfall, so a slab covering a
    tenth of the bounding box fills in one or two."""

    sefl_mmd_n_bandwidths: int = 5
    """Bandwidths in ``L_prog``'s multi-bandwidth RBF kernel. Odd, so the ladder is centred
    on the median heuristic itself: a single bandwidth measures the distributions at one
    scale only, and which scale that is changes as training moves the latent."""

    sefl_mmd_bandwidth_step: float = 2.0
    """Ratio between consecutive bandwidths of that ladder — one octave. The multipliers are
    ``step^k`` for ``k`` symmetric about zero, applied to the median pairwise distance of the
    pooled sample."""

    sefl_sinkhorn_iters: int = 32
    """Sinkhorn iterations in ``L_thick``'s distribution comparison. Enough for the entropic
    plan to converge at the blur used here; the loss is differentiated through the unrolled
    iterations, so this is also a depth."""

    sefl_sinkhorn_blur_multiple: float = 1.0
    """Entropic blur of ``L_thick``'s Sinkhorn divergence, as a multiple of the median
    nearest-neighbour distance **in PC space** (``specs/07`` §3 names exactly that scale).
    A blur below the sampling noise would make the divergence measure the draw rather than
    the distribution."""

    sefl_max_distribution_points: int = 512
    """Cap on the points entering a Sinkhorn or MMD comparison. Both are quadratic in the
    point count, and the statistic they estimate saturates long before a 2000-point patch is
    exhausted; the excess is subsampled with the loss's own generator."""

    sefl_module_min_genes: int = 3
    """Molecular programs (Leiden modules of the gene-gene correlation graph) with fewer
    genes than this are dropped from ``L_prog``'s module-score term. A one-gene module is a
    gene, and its score is already in the MMD term."""

    sefl_thick_grid: int = 8
    """Side of the in-plane grid ``L_thick`` compares **binned** totals on when
    ``section_granularity = "binned"``. Not used on single-cell data, where the comparison is
    distributional (there is no per-cell correspondence to bin against)."""

    sefl_morans_k: int = 8
    """Neighbours of the in-plane kNN graph the *deliberately wrong* ``loss_prog_WRONG``
    computes Moran's I on. It exists so the negative control is reproducible, not because
    anything in the trained model reads it."""

    section_granularity: str = "single-cell"
    """One of ``GRANULARITIES``: what one row of the expression matrix is.

    ``L_thick`` aggregates differently for the two, and ``specs/07`` §3 is explicit about it:
    for single cells a thick section is a **union** of cells, so the comparison is between
    *distributions* of cell state with no per-cell correspondence; for bins or spots the
    expression **sums** within a bin, so binned totals are compared directly. Nothing else in
    the package branches on this yet."""

    # ----------------------------------------------------------------------------------
    # expression (T06)
    # ----------------------------------------------------------------------------------
    cfm_sigma_min: float = 1e-4
    """Minimum path noise in conditional flow matching.

    The straight-line path of T06 §2 is ``h_t = (1-t) h0 + t h1`` exactly; this is the width
    of the isotropic Gaussian the sample is then perturbed by, which is the ``sigma_min`` of
    the conditional-flow-matching construction (the conditional path is
    ``N(h_t, sigma_min^2 I)``, not a Dirac). The regression target ``u = h1 - h0`` is
    unchanged. Small enough to be a regulariser on the velocity field rather than a change to
    the path."""

    ode_steps: int = 24
    """Heun steps in the sampling ODE."""

    genes_per_step: int = 128
    """G': genes subsampled per training step, so the decoder never sees a fixed panel
    width. This is what makes panel width irrelevant."""

    zinb_eps: float = 1e-6
    """Additive epsilon on the ZINB dispersion, and the clamp on the zero-inflation
    probability (``pi`` is confined to ``[zinb_eps, 1 - zinb_eps]`` so that ``log pi`` and
    ``log1p(-pi)`` are finite at the ends of the range)."""

    zinb_theta_min: float = 1e-4
    """Lower clamp on the ZINB dispersion, as T06 §3 specifies. Below it the negative
    binomial's variance ``mu + mu^2/theta`` overflows before the log-likelihood does."""

    zinb_theta_max: float = 1e6
    """Upper clamp on the ZINB dispersion (T06 §3). Above it the NB is a Poisson to float32
    precision and ``lgamma(x + theta) - lgamma(theta)`` cancels catastrophically."""

    zinb_mu_min: float = 1e-8
    """Lower clamp on the ZINB mean (T06 §3)."""

    zinb_mu_max: float = 1e8
    """Upper clamp on the ZINB mean (T06 §3)."""

    decoder_mu_link: str = "softplus"
    """One of ``MU_LINKS``: how the decoder turns its ``mu`` head into a positive mean.

    ``"softplus"`` is T06 §3's own ``mu = softplus(MLP_mu(u)) * size_factor`` and it is the
    default. ``"exp"`` exists because there was a good *a priori* reason to expect it to win —
    a real panel's per-gene mean spans four orders of magnitude, ``softplus(x) ~ x`` for
    ``x >> 0``, so a gene with mean 0.004 needs a pre-activation of -5.5 while one with mean 40
    needs +40, and one shared trunk whose inputs are O(1) is then asked for outputs spanning 45
    units with a gradient dominated by the handful of dense genes. Under ``"exp"`` the
    pre-activation *is* log-expression, which is the scale the quantity lives on.

    **Measured, that reasoning does not pay off, so the spec's choice stands.** On the synthetic
    fixture at 1200 steps (wide-gap holdout, whole panel per step): reconstruction NLL 1.649
    nats/pair under softplus against 1.636 under exp, gene-gene Frobenius error 18.02 against
    17.05, and per-gene mean expression correlation **0.802 under softplus against 0.576 under
    exp** — the argument's own target, moving the wrong way, because an exponential link lets an
    early large pre-activation produce an enormous ``mu`` and the clamp then hides the gradient.
    The field is kept, and kept selectable, because the argument is still the right one to make
    on a panel with a wider dynamic range than this fixture's; changing the default needs that
    measurement, not this one. Both arms are in ``PROGRESS.md``."""

    latent_encoder_hidden: int = 256
    """Hidden width of the expression encoder producing the data-side latent ``h1``."""

    latent_encoder_layers: int = 3
    """Linear layers in the expression encoder. Must be >= 2: one layer would make ``h1`` a
    linear read-out of the gene-embedding-weighted expression pooling."""

    flow_hidden: int = 256
    """Hidden width of the velocity network ``v_theta(h_t, t, cond)``."""

    flow_layers: int = 3
    """Linear layers in the velocity network. Must be >= 2."""

    flow_time_dim: int = 32
    """Width of the sinusoidal embedding of the flow time ``t``. Even, so that the
    ``(sin, cos)`` pairs come out whole."""

    decoder_hidden: int = 128
    """Hidden width of the gene-conditioned decoder's shared trunk.

    Deliberately half ``field_mlp_hidden``: the trunk is evaluated at every *(cell, gene)*
    pair, so its activation is ``(batch_cells, genes_per_step, decoder_hidden)`` — 134 MB at
    the defaults — while the interaction term ``h * (A e_g)`` already carries the
    multiplicative capacity a wider trunk would buy. The three parameter heads (``mu``,
    ``theta``, ``pi``) share it."""

    decoder_layers: int = 3
    """Linear layers in the decoder trunk plus head. Must be >= 2."""

    size_factor_hidden: int = 64
    """Hidden width of the size-factor head decoding the per-cell total from ``h``."""

    size_factor_min: float = 1.0
    """Floor on the decoded per-cell total, in counts. One count per cell: a cell with a
    decoded library size below that would make every ``mu`` vanish."""

    z_embed_dim: int = 16
    """Width of the learned depth embedding ``z_embed`` in the conditioning vector. Separate
    from the z bands of ``fourier(p)``: those are normalised by the bounding box and so say
    *where in this volume*, while ``z_embed`` is fed the depth in units of the median section
    spacing and says *how far from a real section*, which is the quantity generation at an
    arbitrary plane needs."""

    detection_rate_tol: float = 0.10
    """Plausible band on the emitted matrix's median per-gene detection rate, as an absolute
    difference from the training sections' (T06 §4). The generation path asserts it: a silent
    switch to emitting ``mu`` would put every gene's detection rate at ~1.0 and show up here.
    0.10 is wide enough not to fire on realisation noise (measured spread between real
    sections of the synthetic fixture: 0.006) and narrow enough that the 4.2x densification
    that motivated v20's cross-mix could not pass it."""

    independent_donor_k: int = 3
    """Donor cells the independent-donor baseline draws each gene from — the competing
    method's "<= 3 donor cells" (``eval/baselines.py``). A deliberate negative control: it is
    the reference point for T06's gene-gene covariance claim and T10 reuses it."""

    cross_mix_weight_tol: float = 1e-3
    """How far the donor weights of :func:`~spatialcpav25_gen.model.expression.cross_mix_counts`
    may sum away from 1 before it raises. v20's ``alpha_tol``, which snapped a
    machine-epsilon mixing weight to zero so that narrow designs stayed *exactly* the
    previous version."""

    intensity_basis_ell_multiple: float = 1.0
    """Finest wavelength the layout intensity head's Fourier basis may resolve, as a multiple
    of the fitted in-plane correlation length ``ell_xy``.

    The answer T06's trainer owes to SPEC_QUESTIONS B10. The Poisson MLE of a flexible
    intensity overfits the point pattern: at ``fourier_bands_xy = 8`` the head resolves
    structure down to a hundredth of the section and the recovered correlation decays from
    0.97 at 300 steps to 0.28 at 1200 while the NLL keeps falling. The intensity is a
    *smoothed* function of the anatomical field, and the field's own correlation length is
    ``ell_xy``, so structure finer than that is not something the point pattern constrains.
    :func:`~spatialcpav25_gen.model.layout.fourier_bands_for_lengthscale` turns this into a
    band count and the trainer builds its intensity head with it; ``1.0`` means "the finest
    band's wavelength is at least one correlation length"."""

    # ----------------------------------------------------------------------------------
    # losses
    # ----------------------------------------------------------------------------------
    w_recon: float = 1.0
    """Reconstruction (ZINB NLL). Must dominate throughout training."""

    w_cfm: float = 1.0
    """Conditional flow matching (T06 §2). Not in the spec's loss list, which names only the
    terms that trade off *against* reconstruction; the CFM term does not compete with it —
    ``h1`` is detached inside ``cfm_loss``, so the flow's gradient never reaches the encoder
    or the decoder — and 1.0 is therefore the natural weight rather than a tuned one."""

    w_size: float = 0.1
    """Library-size head (T06 §3), a squared error in log counts. Small: the head has one
    scalar output and the reconstruction term already sees the library size through
    ``mu * size_factor``, so this exists to make generation's *decoded* library size right,
    not to shape the latent."""

    w_layout: float = 1.0
    """Poisson-process layout NLL."""

    w_autocorr: float = 0.0
    """Metric-aware Moran's I / Geary's C agreement (T08). **A selection gate; 0 is where the
    search starts, not a decision.**

    Chosen per dataset by ``specs/09`` §3's ``select_config``, **jointly with**
    ``Config.train_steps``, on internal LOSO over training sections. The three metric-aware
    weights move together as one gate (``{off, spec weights}``), because that is how T08 measured
    them and nothing separates them.

    ``specs/08`` fixes no weight and T01 wrote 0.5 as a placeholder. T08 measured the three terms
    at 0.5 each and left them at **0**, for one reason and with one caveat, both of which belong
    together (the full table is in ``progress/t08_metric_aware.md``, ablation A2).

    *The reason.* At T06's own 1200-step budget the terms cost on every statistic they are made
    of — Moran's MAE 0.0287 -> 0.0408, marker-depth r 0.978 -> 0.967, cell-type localization
    0.967 -> 0.962, gene-gene Frobenius 9.00 -> 11.15, reconstruction 1.590 -> 1.684 nats/pair.
    A schedule-only control (internal LOSO hiding a section, no terms charged) sits between the
    two, so it is the terms and not the hidden section that costs. Turning a term on by default
    when it is measured to hurt is what T07 refused to do for SEFL, and the same answer applies.

    *The caveat, and it is why this is "not yet" rather than "no".* The terms are **slower**, not
    worse: they add a constraint, and 1200 steps is where T06 stopped *because the model without
    them starts degrading there* — the early stop is itself R4's symptom. At 2400 steps the
    ordering reverses on four of six statistics, including both of the ones these terms are named
    for and the covariance the task exists to fix:

    | statistic | off@1200 | on@1200 | off@2400 | on@2400 |
    |---|---|---|---|---|
    | reconstruction (nats/pair) | 1.5901 | 1.6843 | **1.5703** | 1.5885 |
    | gene-gene Frobenius | 9.000 | 11.154 | 9.049 | **8.489** |
    | Moran's MAE | 0.0287 | 0.0408 | 0.0339 | **0.0279** |
    | marker-depth r | 0.978 | 0.967 | 0.983 | **0.990** |
    | mean-variance slope (real 1.741) | 1.762 | 1.734 | 1.773 | **1.722** |
    | cell-type localization | **0.967** | 0.962 | 0.958 | 0.957 |

    The arm without the terms gets a better *likelihood* with the longer budget and a worse
    covariance (9.000 -> 9.049 while the NLL falls 1.5901 -> 1.5703), which is open risk **R4**
    in miniature; the arm with them improves on both.

    *Which is why this value is selected rather than shipped.* A 0 chosen at 1200 steps is
    calibrated to an undertrained model, so it is not hardcoded: ``select_config`` scores all four
    cells of ``{1x, 2x train_steps} x {off, spec weights}`` and picks per dataset. T10's **A2**
    reports the six target metrics **at both budgets**, and is an *addition* experiment against
    this starting value, in the shape T07 left A7."""

    w_profile: float = 0.0
    """Metric-aware depth / field / per-type profile agreement (T08). **A selection gate; 0 is the
    search's starting value** — see ``w_autocorr`` for the measurement and the joint gate that
    decides all three."""

    w_distribution: float = 0.0
    """Metric-aware Sinkhorn (or MMD) distribution matching (T08). **A selection gate; 0 is the
    search's starting value** — see ``w_autocorr``.

    ``specs/08`` §3's honest note, recorded here because this is the term it is about: mixing is
    the metric where the competing method is genuinely strong, because its per-gene chimerism
    maximises cloud overlap almost by construction. A tie there is an acceptable outcome."""

    w_cross: float = 0.0
    """SEFL plane-intersection consistency (T07). ``specs/07`` sets 0.3; **T07 measured it and
    set the default to 0 on this architecture.** ``0`` also participates in ablation A7.

    The loss is built, tested and kept — T10's A7 and E5 both need it, and the number below is
    a result. But it must not be on by default here, because in *this* architecture it has
    nothing legitimate left to constrain and its remaining minimiser destroys the anatomical
    field:

    * ``CTFFlow``'s expression pathway conditions at **physical points in the data frame** —
      retrieval, the GRF and the Fourier encoding are all data-frame channels — so two
      sections that cross already produce **bitwise identical** expression where they cross,
      untrained, with no consistency loss applied
      (``test_generation_is_intersection_consistent_by_construction``). That is the property
      ``L_cross`` exists to enforce, and the continuous field of v25 supplies it for free.
    * What is left plane-dependent is the augmentation **pose**, which is what a two-branch
      loss can actually compare — and T04 made the triplane pose-dependent *on purpose*, as
      the capacity mechanism GATE 2 rests on. Minimising ``L_cross`` therefore drives the
      field towards pose-invariance, i.e. towards constant, and a constant field carries no
      anatomy.

    Measured on the fixture, 500 steps, ``specs/07``'s own schedule, four arms that differ only
    in which SEFL terms are live (generated per-gene variance as a fraction of the real
    section's; ``L_cross`` self-consistency in brackets):

    | arm | reconstruction | generated variance |
    |---|---|---|
    | SEFL off | **1.738** | **0.711** (0.022) |
    | ``thick`` + ``prog`` only (this default) | 2.082 | **1.331** (0.024) |
    | + ``w_cross = 0.3`` | 2.024 | **0.065** (0.010) |
    | + ``w_cross = 0.3``, teacher off | 1.914 | 0.344 (0.013) |

    ``L_cross`` itself works — it falls **90 %** over the run — and that is exactly the
    problem. The decision is recorded as open risk **R6** and proposed to the spec's owner in
    SPEC_QUESTIONS C19; the alternative fix is a design change (make the branches differ by
    the *evidence* each plane would have, not by the pose), which is not a tuning fix."""

    w_thick: float = 0.0
    """SEFL thickness coarse-graining consistency (T07). ``specs/07`` implies 0.2; **T07 shipped
    it at 0 with the rest of SEFL** — see ``w_prog`` for the measurement that decided it.

    Unlike ``w_cross`` there is nothing wrong with this term: it is verified against its own
    criterion (counts add at 3.000x with the loss charging 0.00) and it is not the one that
    flattens the field. It is off because **SEFL as a whole has never been shown to help**, and
    because leaving it on costs T06's criteria."""

    w_prog: float = 0.0
    """SEFL molecular-program invariance (T07). ``specs/07`` implies 0.2; **shipped at 0**.

    SEFL is **opt-in until T10's A7 measures it on the six target metrics.** The reason is a
    measurement, not caution: with ``w_thick`` and ``w_prog`` at 0.2 — ``w_cross`` already at 0
    — a model trained at T06's own budget and configuration **fails three of T06's acceptance
    tests**:

    | statistic | T06 recorded (SEFL off) | with `thick` + `prog` on | criterion |
    |---|---|---|---|
    | detection-rate MAD | 0.0191 | **0.0551** | < 0.05 |
    | mean-variance slope, relative error | 0.0084 | **0.2838** | < 0.15 |
    | gene-gene Frobenius covariance error | 9.316 | **20.301** | vs a 7.563 baseline |

    A default that breaks the previous task's acceptance criteria is not a default. Turning the
    two terms on is one config override, which is what A7 becomes: an **addition** experiment
    against this shipped default rather than an ablation away from it (``specs/10`` §4,
    amended). If A7 shows a gain on the target metrics, these defaults move back; the numbers
    above are what that gain has to beat."""

    w_prog_wrong: float = 0.0
    """**Ablation A8's negative control — never raise this outside A8** (``specs/10`` §6).

    Weight on :func:`~spatialcpav25_gen.losses.sefl.loss_prog_WRONG`, which forces in-plane
    Moran's I to agree across two sampling angles — i.e. constrains a quantity from
    ``EQUIVARIANT_QUANTITIES`` as though it were invariant. It is wrong on purpose: a tangential
    cut through a laminar structure *has* a different in-plane autocorrelation from a
    perpendicular one, and that difference is signal.

    T07 built the loss and deliberately left it out of ``SEFL_LOSSES``, which left A8 with no
    gate to override and made it the one ablation that could not be run as a config change. This
    field is that gate, and nothing else: the term is absent at ``0.0`` (``sefl_terms`` skips a
    zero-weight builder, so it costs no forward pass), and the only run that sets it is A8, whose
    stated expectation is that the model gets **worse**. A8 is one of the project's two
    deliberate negative controls; ``loss_prog_WRONG`` stays out of ``SEFL_LOSSES`` and out of
    ``SEFL_DEFAULT_TERMS``, asserted by ``test_wrong_loss_is_not_in_the_registry``."""

    w_distill: float = 0.1
    """Text -> free-residual distillation (T02)."""

    sefl_warmup_frac: float = 0.2
    """Fraction of training spent on reconstruction only before the SEFL terms ramp in."""

    ema_decay: float = 0.999
    """EMA decay for the teacher weights. The teacher plus stop-gradient is what stops
    the symmetric consistency losses collapsing to a constant field.

    The horizon it implies, ``1 / (1 - decay) = 1000`` steps, is longer than a short run — and
    T07 checked whether that alone can explain the collapse an over-driven SEFL arm shows
    (the teacher would then be an anchor to *initialisation* rather than a lagging target).
    **It cannot:** on the 500-step fixture arm the generated per-gene variance ratio is 0.054
    at ``0.999`` and 0.064 at ``0.99``, i.e. unchanged. The cause is the loss's own minimiser,
    not the teacher's horizon — see ``progress/t07_sefl_losses.md``. Left at the spec's value; T09's
    config selector may still want to scale it with the step budget."""

    # ----------------------------------------------------------------------------------
    # metric-aware losses (T08)
    # ----------------------------------------------------------------------------------
    profile_n_bins: int = 24
    """Bins in the soft depth profile."""

    profile_grid_size: int = 24
    """Side length of the coarse 2-D field-profile grid (24 x 24)."""

    profile_sigma_frac: float = 0.75
    """Soft-binning kernel width as a fraction of the bin width."""

    loso_every_k_steps: int = 4
    """Run the metric-aware block every k steps."""

    loso_max_cells: int = 4000
    """Cell subsample size for the metric-aware block."""

    loso_epoch_steps: int = 50
    """Optimiser steps one internal-LOSO *epoch* lasts, i.e. how long a section stays hidden.

    ``specs/08`` states the schedule in epochs ("each epoch, hide one training section") and the
    trainer counts steps (T06 §5), so the two units have to be related somewhere; putting the
    conversion here keeps it out of the loop. It must be several times
    ``loso_every_k_steps`` or a hidden section is measured once or twice before it changes and
    the term is noise rather than a signal."""

    metric_huber_delta: float = 1.0
    """Huber transition point for the metric-aware agreement terms (T08 §1, §2).

    Moran's I, Geary's C and a ``log1p`` profile all live in ranges of order one, so ``1.0`` is
    the value at which the loss is quadratic over the normal disagreement and linear over the
    outliers - which is the entire reason ``specs/08`` asks for a Huber rather than an L2."""

    metric_marker_genes: int = 32
    """Marker genes the profile terms are computed over: the most spatially autocorrelated
    genes of the *real* (training) section being reconstructed.

    ``specs/08`` §2 says "top-k spatially variable genes from training" without fixing k. 32 is
    the width at which the mean over markers is stable on the fixture and the cost of the
    profile terms stays below the autocorrelation term's."""

    metric_marker_min_detection: float = 0.05
    """Smallest real detection rate a gene may have and still be chosen as a marker.

    ``specs/08`` §2 says "top-k spatially variable genes from training" and leaves the pool
    open. Ranking the *whole* panel by Moran's I selects for sparse genes whose handful of
    non-zero cells happen to be neighbours: their I is high and almost entirely an artefact of
    the count, and the profile term — which compares each marker against its own mean — then
    divides by a mean of order 1e-3. Measured: with no floor the profile term goes 1.9 -> 10.6
    -> 1766 over forty training steps while every model parameter stays put, because the
    explosion is in the loss's own arithmetic rather than in the model.

    ``0.05`` is the floor T06's ``informative_genes`` already uses, for the same reason (a
    correlation carried by two cells measures the draw, not the tissue)."""

    metric_distribution_kind: str = "sinkhorn"
    """Which divergence :func:`~spatialcpav25_gen.losses.metric_aware.loss_distribution` uses.

    ``sinkhorn`` is ``specs/08`` §3's entropic Sinkhorn divergence and ``mmd`` its named
    fallback. The implementation is the in-repo one T07 already wrote
    (:func:`~spatialcpav25_gen.losses.sefl.sinkhorn_divergence`), **not** ``geomloss``, even
    when ``geomloss`` is installed: a loss that silently changes implementation with the
    environment makes a paper number a property of the machine it ran on (SPEC_QUESTIONS C20).
    The solver's iteration count is ``sefl_sinkhorn_iters``, shared because it is a property of
    the solver rather than of the caller."""

    metric_distribution_points: int = 512
    """Cells per side in the distribution term. The Sinkhorn cost matrix is quadratic in this,
    and 512 x 512 is the point on the fixture where the divergence's own sampling noise stops
    dominating the difference between two arms."""

    metric_sinkhorn_blur_multiple: float = 1.0
    """Sinkhorn blur as a multiple of the median nearest-neighbour distance in PC space, which
    is ``specs/08`` §3's prescription. Measured on the *real* cloud and detached, so the
    entropic scale cannot be reduced by shrinking the generated one."""

    metric_dominance_ratio_warn: float = 1.0
    """Warn when the weighted metric-aware terms exceed this multiple of the weighted
    reconstruction terms. ``specs/08``'s "do NOT weight these losses above reconstruction",
    made observable at run time the same way T07 made its own version observable
    (``sefl_consistency_ratio_warn``)."""

    metric_type_grid_ell_multiple: float = 1.0
    """Grid-cell width of the **per-type** spatial histogram, in field correlation lengths.

    The one place T08's profile terms do not share a grid with the marker-expression field, and
    the reason is an asymmetry rather than a preference. Both sides of the expression field are
    count draws at the same positions, so binning them finely is noisy but fair. The per-type
    histogram compares a *smooth* predicted composition against *hard* labels, and at
    ``profile_grid_size = 24`` a 1000 um section gives 42 um bins holding two or three cells —
    so the target is a one-hot label map and the loss's own minimiser is an infinitely confident
    intensity field. Measured: ``max_c lambda_c / sum_c lambda_c`` climbs 0.73 -> 0.9995 over 25
    metric steps and the gradient follows it through the shared anatomical field into the flow
    (0.4 -> 838) until ``mu`` is ``NaN``.

    ``1.0`` means a bin is one ``ell_xy`` across, which is the same argument
    ``intensity_basis_ell_multiple`` makes for the intensity head's own basis (B10): structure
    finer than the field's correlation length is not something the data constrains, and asking
    for it is asking to overfit. On the fixture it gives ~8 bins a side and ~23 cells a bin."""

    metric_eps: float = 1e-8
    """Denominator guard shared by the metric-aware losses: the soft-profile weight sums and
    the Moran / Geary variance denominators. A gene with zero variance in a subsample, or a
    profile bin with no cell near it, is a 0/0 rather than an error - the surrounding terms are
    still well defined and dropping the whole step would make the loss depend on the
    subsample."""

    # ----------------------------------------------------------------------------------
    # inference and calibration (T09)
    # ----------------------------------------------------------------------------------
    n_uncertainty_samples: int = 8
    """M: flow samples per cell used to estimate the latent variance that gates
    retrieval anchoring."""

    bisection_max_iter: int = 8
    """Bisection iterations in the length-scale calibrator."""

    calibration_ell_max_fitted_multiple: float = 2.0
    """Upper end of the length-scale calibration bracket, as a multiple of the ``ell``
    ``fit_lengthscale_from_sections`` returns (T09; also the cap GATE 1's G1.3c
    monotonicity criterion is stated over, together with
    ``calibration_ell_max_extent_frac`` - whichever binds first).

    ``I_gen(ell)`` turns over at the point where the generated section's neighbourhood
    correlation overtakes the real one, which GATE 1 measures at 2.5x the fitted ``ell`` on
    the 3000 um gate fixture and 1.6x on the 1000 um one - i.e. near the *tissue's*
    correlation length rather than at a fixed fraction of the window. The spec's own
    0.25x - 4x sweep is therefore wider than the monotone branch whatever the field of
    view, and a bracket has to be capped in these units as well as in the window's."""

    calibration_ell_max_extent_frac: float = 0.2
    """Upper end of the length-scale calibration bracket, as a fraction of the sections'
    in-plane extent (T09; also the cap GATE 1's G1.3c monotonicity criterion is stated
    over).

    Mean Moran's I of *generated* expression is not monotone in ``ell`` over an unbounded
    range: it is a ratio of spatially structured variance to total variance, and a
    stationary unit-variance field loses within-window variance once its correlation
    length approaches the window, so ``I_gen(ell)`` rises, turns over and falls. Bisecting
    across the maximum is ill-posed. Measured on the synthetic fixture the maximiser sits
    near 0.07-0.22 of the in-plane extent depending on the field of view, so this cap is
    at the top of that range rather than safely below it - T09 must find the maximum
    rather than trust the cap (see ``reports/gate1.md``)."""

    bisection_grid_size: int = 12
    """Grid size for the fallback search when bisection fails to bracket."""

    calibration_morans_tol: float = 0.02
    """Acceptance tolerance of the length-scale calibrator, in Moran's I.

    ``specs/09``'s ``test_calibration_converges``: the bisection has converged when
    ``|I_gen - I_flank| < 0.02`` within ``bisection_max_iter`` iterations. It is the
    calibrator's own stopping rule as well as the test's threshold, so the two cannot drift."""

    calibration_max_cells: int = 2000
    """Cells the calibrator measures a generated section's Moran's I on.

    One bisection is ~20 generations and Moran's I needs a kNN graph over the emitted cells;
    capping the measurement keeps the cost linear in the number of iterations rather than in
    the section's size. Subsampling changes the *variance* of the estimate, not its
    expectation, and both sides of the comparison are capped the same way."""

    calibration_ell_z_max_fitted_multiple: float = 1.0
    """Upper end of the ``ell_z`` calibration bracket, as a multiple of the fitted ``ell_z``.

    Open risk **R1**, remedy 3: on a short stack the along-z variogram extrapolates past its
    data and reads **high** (561 um fitted against a 200 um ground truth on the fixture), so
    the fitted value enters T09 as a **bracket endpoint, not a value**. 1.0 makes it exactly
    the upper endpoint; the calibrated ``ell_z`` is whatever matches the *observed*
    between-section correlation decay below it (remedy 2)."""

    calibration_ell_z_min_factor: float = 0.25
    """Lower end of the ``ell_z`` calibration bracket, as a multiple of the median section
    spacing. Below one quarter of a spacing the stack cannot resolve the decay at all: the
    shortest lag it can form *is* one spacing."""

    calibration_detection_ridge: float = 10.0
    """Pseudo-count ridge on the per-gene detection and dispersion corrections.

    The corrections are per gene and fitted on a few thousand cells, so a gene detected in
    three of them would otherwise get a correction fitted to three cells. The ridge shrinks
    each gene's correction toward zero by ``n / (n + ridge)``; 10 cells is the scale at which
    a per-gene rate stops being noise."""

    calibration_logit_max: float = 8.0
    """Clamp on the magnitude of the ``pi``-logit and ``log theta`` corrections.

    ``logit(1e-4) ~ -9``, so eight nats is the whole usable range of a detection rate: past
    it the correction is asking for a probability the decoder's own guards already floor, and
    an unclamped affine fit on a gene detected in no cell at all is infinite."""

    anchor_variance_bins: int = 8
    """Latent-variance bins the anchor weight ``w(v)`` is fitted on.

    The isotonic regression of ``specs/09`` §1 is fitted on bin means rather than on raw
    per-cell pairs: the per-cell reconstruction error of a *draw* is far noisier than the
    per-bin mean, and the map has one shape parameter per bin either way."""

    anchor_weight_grid_size: int = 5
    """Candidate blend weights per variance bin, spanning ``[0, 1]`` inclusive.

    ``w`` is a Bernoulli mixing probability, so a five-point grid resolves it to 0.25 —
    finer than the difference between two adjacent bins' optima measured on the fixture, and
    cheap: the whole grid is scored from **one** pair of count matrices per bin."""

    generation_empty_pool_tol: float = 0.01
    """Fraction of generated cells that may have an empty retrieval pool before the
    generation path **fails**.

    ``specs/09`` §1: once the z window is derived from the geometry rather than fixed,
    ``EmptyCandidatePoolWarning`` firing on more than a negligible fraction of queries means
    the geometry is genuinely impossible and the caller needs to know. T06 measured
    100-110 of every 512 cells (20%) at the fixed window on a ``consecutive-3`` holdout,
    which is what this turns from a warning into an error."""

    boundary_margin_spacings: float = 0.5
    """How far outside the training stack a plane may sit before it is flagged as
    extrapolation, in units of the median section spacing.

    Open risk **R3**: a cell at the stack's end has evidence on one side only, and per-section
    R2 on the gate fixture was 0.2912 / 0.3642 at the two ends against an interior mean of
    0.4474. Half a spacing is the point at which a query plane has no flanking pair left, so
    it is where the regime actually changes. The flag travels on the emitted AnnData's
    ``uns`` rather than being left for the caller to infer from ``z``."""

    # ----------------------------------------------------------------------------------
    # automatic configuration selection (T09 3)
    # ----------------------------------------------------------------------------------
    selection_reduced_epoch_frac: float = 0.25
    """Fraction of the full budget a *non-budget* gate's candidate is fitted at.

    ``specs/09`` 3's "reduced-epoch model (25% of ``cfg.epochs``)". It is a cost control for
    gates whose effect is visible early, and it is **invalid for the budget gate itself** —
    a reduced fit of a ``2x`` candidate is the ``1x`` candidate — which is why
    :func:`~spatialcpav25_gen.train.select.select_config` fits the joint gate's cells at the
    budget each one names."""

    claim_min_seeds: int = 3
    """Seeds any measurement that reaches a paper claim must run (``specs/09`` §3's
    repeated-seed rule, scoped in ``specs/10`` §3).

    Three rather than one because a single-seed number cannot distinguish "wins" from "wins by
    less than the run-to-run variation": T09 refitted one configuration at the same seed in a
    different process and its scores moved by up to 0.0120, while the gap between the two
    ``text_emb_mode`` options was 0.0110. Three rather than more because the rule is scoped to
    claim-bearing measurements and every extra seed is a full fit."""

    claim_tie_break_envelope: float = 0.04
    """Below this separation two gate options count as **indistinguishable**, and the capability
    tie-break decides instead of the rank ordering (``specs/09`` §3).

    Set from the measured across-seed spread, not from a nominal figure:
    ``reports/envelope_synthetic.md`` fits three representative cells at three seeds each and
    the largest spread is **0.0335** (``gearys_pearson``). Rounded **up** to 0.04, because that
    0.0335 is a maximum over nine fits and therefore a *lower bound* on the true spread — the
    same measurement at two fits had said 0.0120, so more samples widened it almost threefold.
    A threshold tighter than the evidence would let a difference the campaign cannot reproduce
    be reported as a win.

    The envelope is **not score-dependent** — 0.0299 at a score level of 0.95, 0.0335 at 0.94
    and 0.0270 at 0.83 — so one threshold serves every cell rather than needing a per-score
    scale."""

    selection_convergence_tol: float = 0.05
    """How far a metric may fall at the reduced budget before the incumbent counts as
    **unconverged** there (``specs/09`` §3's widened training-free-option rule).

    All six target metrics are correlations or scores on a comparable ``[-1, 1]`` scale, so an
    absolute tolerance is meaningful across them. Measured on the fixture, and the reason the
    exact value is not load-bearing: with a ``zinb-flow`` incumbent the reduced budget costs
    0.36 / 0.41 / 0.81 on the three distribution-level metrics, while with a ``cross-mix``
    incumbent it costs at most 0.04 and one metric *improves*. The two regimes separate by an
    order of magnitude, so any threshold between them classifies both correctly."""

    selection_convergence_min_metrics: int = 2
    """How many metrics must fall by more than :attr:`selection_convergence_tol` before the
    incumbent is declared unconverged at the reduced budget.

    Two of six rather than one, so a single noisy metric cannot escalate a whole selection to
    full budget, and rather than a majority, so a gate is not scored on an unconverged
    incumbent merely because the damage is concentrated in the statistics that matter most.
    On the fixture the ``zinb-flow`` incumbent fails three and the ``cross-mix`` incumbent
    fails none."""

    selection_passes: int = 2
    """Coordinate-descent passes over the non-joint gates. ``specs/09`` 3: "2 passes ->
    ~10 fits, not 36"."""

    selection_budget_multiple: float = 2.0
    """The ``2x`` arm of the joint ``train_steps`` x metric-weights gate.

    T08 measured the interaction this gate exists for at 1200 and 2400 steps: the
    metric-aware terms lose at the first budget and win on four of six statistics at the
    second. The multiple is a ``Config`` field because the *pair* of budgets is what is
    scored, and a run has to record which pair it scored."""

    selection_metric_weight: float = 0.5
    """The "spec weights" cell of the joint gate: the value ``w_autocorr``, ``w_profile`` and
    ``w_distribution`` take when the gate's weights are **on**.

    ``specs/08`` 3's weights. They ship at 0 (see ``w_autocorr``) and this is the alternative
    the selector scores them against, per dataset, at both budgets."""

    selection_n_folds: int = 3
    """Internal-LOSO folds each selection candidate is scored on.

    Every fold is one generated section compared against the training section it was
    generated in place of, so the cost of a candidate is linear in this. Three interior folds
    is what makes a ~10-fit coordinate descent fit a CPU budget; the *number* matters less
    than that every candidate is scored on the **same** folds, which
    :func:`~spatialcpav25_gen.train.select.selection_folds` guarantees."""

    selection_mixing_knn_k: int = 15
    """Neighbours in the shared-embedding mixing metric of the selection scorer.

    Larger than ``metric_knn_k`` on purpose: mixing is a property of a *neighbourhood*, and at
    k = 10 a single duplicated profile moves the score by 10%. 15 is what the scoreboard's own
    kNN mixing uses."""

    # ----------------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------------
    epochs: int = 200
    """Training epochs."""

    train_steps: int = 1200
    """Optimiser steps for a full fit — **a selection gate, not a constant** (T08, chosen at T09).

    The value `train_ctfflow` is called with. It is a `Config` field so that a selected budget is
    persisted, hashed into the run and reported like every other gate, rather than living in a
    caller's local variable where it cannot be recorded.

    **Do not read 1200 as a decision.** It is T06's number, and T06 chose it by reading this
    fixture's own degradation curve — the arm without T08's metric-aware terms starts getting
    *worse* at generating sections past roughly this point while its likelihood keeps improving
    (open risk R4). That makes 1200 a symptom, not a reference point, and it will not transfer to a
    dataset with different section counts and panel width.

    T08 measured the interaction that makes this a gate: the metric-aware weights **lose** at this
    budget and **win** at twice it (four of six statistics reverse — the table is in `w_autocorr`).
    So the budget and the weights are selected **jointly**, on internal LOSO over training sections
    only, by `specs/09` §3's `select_config`, which requires all four cells of
    `{1x, 2x} x {off, spec weights}` to be scored together — visiting them as separate
    coordinate-descent gates would pick "weights off" from a 1200-step incumbent and stop.
    """

    batch_cells: int = 2048
    """Cells per training batch."""

    lr: float = 3e-4
    """AdamW learning rate."""

    weight_decay: float = 0.01
    """AdamW decoupled weight decay. Torch's own default for ``AdamW``; T06 §5 names the
    optimiser and not this, and there is no measurement here that would justify moving it."""

    grad_clip: float = 1.0
    """Global gradient-norm clip. T06 §5 fixes the value at 1.0; it is a ``Config`` field
    rather than a literal in the trainer because Convention 1 admits no exceptions, and
    because ablating the clip is the first thing to try when a run diverges."""

    lr_min_frac: float = 0.01
    """Floor of the cosine schedule, as a fraction of ``lr``. The schedule T06 §5 asks for
    decays over the whole run; a floor of exactly zero makes the last few hundred steps do no
    work at all, and 1% of the peak is the usual compromise."""

    log_every: int = 10
    """Steps between entries in the trainer's history — the per-term losses, the retrieval
    attention entropy (T06's carried-over GATE 2 item) and the per-gene variance the T07
    collapse alarm watches. Logging every step doubles the cost of a cheap step for a
    trajectory nobody reads at that resolution."""

    # ----------------------------------------------------------------------------------
    # construction / serialisation
    # ----------------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config from a YAML mapping, validating the result.

        An unknown key is an error rather than a silently ignored typo (Convention 6):
        a misspelled ``ell_xy`` that quietly kept the default would be invisible in a
        paper run. An empty file yields the defaults.
        """
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigError(f"{path}: expected a YAML mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> Config:
        """Build a config from a mapping of field name -> value, validating the result."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ConfigError(
                f"unknown Config field(s): {', '.join(unknown)}. "
                "Add the field to spatialcpav25_gen/config.py rather than passing it through."
            )
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def to_dict(self) -> dict[str, Any]:
        """Return every field as a plain dict, suitable for YAML."""
        return dataclasses.asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        """Write every field to ``path`` as a YAML mapping, keys in declaration order."""
        with Path(path).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)

    def content_hash(self) -> str:
        """Return a stable 16-character hash of every field. Identical configs hash equally.

        The identifier ``specs/09`` 1 requires on a generated section's ``uns``: a run that
        cannot say which config produced it cannot be reproduced, and the full mapping is too
        long to read on an AnnData. sha256 of the YAML serialisation with keys in declaration
        order, truncated — the same construction as T02's descriptor cache key, and stable
        across processes (Python's own ``hash`` is salted per process and would change
        between runs).
        """
        payload = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def replace(self, **kwargs: Any) -> Config:
        """Return a copy with ``kwargs`` overridden, validating the result."""
        known = {f.name for f in dataclasses.fields(self)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ConfigError(f"unknown Config field(s): {', '.join(unknown)}")
        cfg = dataclasses.replace(self, **kwargs)
        cfg.validate()
        return cfg

    # ----------------------------------------------------------------------------------
    # validation
    # ----------------------------------------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ConfigError` if the config is out of range or contradictory.

        Self-consistency only: checks that depend on a dataset (how many sections there
        are, whether the counts are integers) live in
        :func:`spatialcpav25_gen.data.schema.validate_config_against_volume`, because a
        ``Config`` on its own has no volume to check against (SPEC_QUESTIONS A5).
        """
        self._check_choices()
        self._check_positive()
        self._check_fractions()
        self._check_relations()

    def _check_choices(self) -> None:
        """Check every string-valued gate against its allowed set."""
        choices: list[tuple[str, str, frozenset[str]]] = [
            ("device", self.device, DEVICES),
            ("layout_mode", self.layout_mode, LAYOUT_MODES),
            ("prior_mode", self.prior_mode, PRIOR_MODES),
            ("expr_mode", self.expr_mode, EXPR_MODES),
            ("text_emb_mode", self.text_emb_mode, TEXT_EMB_MODES),
            ("decoder", self.decoder, DECODERS),
            ("text_pooling", self.text_pooling, TEXT_POOLINGS),
            ("rotation_bias", self.rotation_bias, ROTATION_BIASES),
            ("potts_update", self.potts_update, POTTS_UPDATES),
            ("decoder_mu_link", self.decoder_mu_link, MU_LINKS),
            ("gene_summary_fallback", self.gene_summary_fallback, SUMMARY_FALLBACKS),
            ("section_granularity", self.section_granularity, GRANULARITIES),
            ("metric_distribution_kind", self.metric_distribution_kind, METRIC_DISTRIBUTION_KINDS),
        ]
        for name, value, allowed in choices:
            if value not in allowed:
                raise ConfigError(f"Config.{name}={value!r} is not one of {sorted(allowed)}")

    def _check_positive(self) -> None:
        """Check that every field that must be strictly positive is."""
        positive: dict[str, float] = {
            "min_cells_per_section": self.min_cells_per_section,
            "min_sections_per_volume": self.min_sections_per_volume,
            "metric_knn_k": self.metric_knn_k,
            "text_dim_in": self.text_dim_in,
            "gene_emb_dim": self.gene_emb_dim,
            "ctx_emb_dim": self.ctx_emb_dim,
            "text_batch_size": self.text_batch_size,
            "text_max_length": self.text_max_length,
            "distill_hidden": self.distill_hidden,
            "text_diag_knn_k": self.text_diag_knn_k,
            "selection_convergence_min_metrics": self.selection_convergence_min_metrics,
            "claim_min_seeds": self.claim_min_seeds,
            "text_diag_leiden_resolution": self.text_diag_leiden_resolution,
            "text_diag_leiden_iterations": self.text_diag_leiden_iterations,
            "text_diag_max_pairs": self.text_diag_max_pairs,
            "matern_nu": self.matern_nu,
            "n_rff": self.n_rff,
            "ell_xy": self.ell_xy,
            "ell_z": self.ell_z,
            "latent_dim": self.latent_dim,
            "grf_chunk_points": self.grf_chunk_points,
            "variogram_n_pcs": self.variogram_n_pcs,
            "variogram_n_bins": self.variogram_n_bins,
            "variogram_max_cells_per_section": self.variogram_max_cells_per_section,
            "variogram_min_pairs_per_bin": self.variogram_min_pairs_per_bin,
            "variogram_n_ell_grid": self.variogram_n_ell_grid,
            "variogram_ell_min_factor": self.variogram_ell_min_factor,
            "variogram_ell_max_factor": self.variogram_ell_max_factor,
            "variogram_z_grid_size": self.variogram_z_grid_size,
            "variogram_z_min_cells_per_cell": self.variogram_z_min_cells_per_cell,
            "variogram_min_structured_frac": self.variogram_min_structured_frac,
            "triplane_res_xy": self.triplane_res_xy,
            "triplane_res_z": self.triplane_res_z,
            "triplane_channels": self.triplane_channels,
            "n_plane_orientations": self.n_plane_orientations,
            "field_mlp_hidden": self.field_mlp_hidden,
            "field_mlp_layers": self.field_mlp_layers,
            "field_dim": self.field_dim,
            "rotation_bias_max_tilt_deg": self.rotation_bias_max_tilt_deg,
            "retrieval_k": self.retrieval_k,
            "retrieval_z_window": self.retrieval_z_window,
            "retrieval_z_window_gap_factor": self.retrieval_z_window_gap_factor,
            "retrieval_ctx_dim": self.retrieval_ctx_dim,
            "retrieval_n_heads": self.retrieval_n_heads,
            "retrieval_score_temperature": self.retrieval_score_temperature,
            "retrieval_candidates_per_section": self.retrieval_candidates_per_section,
            "retrieval_query_chunk": self.retrieval_query_chunk,
            "niche_knn_k": self.niche_knn_k,
            "niche_n_scales": self.niche_n_scales,
            "niche_scale_factor": self.niche_scale_factor,
            "section_dropout_max_sections": self.section_dropout_max_sections,
            "expr_pca_dim": self.expr_pca_dim,
            "gate2_min_cells_per_angle": self.gate2_min_cells_per_angle,
            "potts_iters": self.potts_iters,
            "potts_knn_k": self.potts_knn_k,
            "layout_n_mc": self.layout_n_mc,
            "layout_max_proposal_factor": self.layout_max_proposal_factor,
            "swd_polish_steps": self.swd_polish_steps,
            "layout_mlp_hidden": self.layout_mlp_hidden,
            "layout_mlp_layers": self.layout_mlp_layers,
            "layout_fit_steps": self.layout_fit_steps,
            "layout_fit_lr": self.layout_fit_lr,
            "layout_intensity_eps": self.layout_intensity_eps,
            "layout_proposal_batch": self.layout_proposal_batch,
            "pcf_n_bins": self.pcf_n_bins,
            "pcf_max_r_factor": self.pcf_max_r_factor,
            "repulsion_gamma_grid_size": self.repulsion_gamma_grid_size,
            "repulsion_fit_cells": self.repulsion_fit_cells,
            "potts_beta_grid_size": self.potts_beta_grid_size,
            "potts_beta_min": self.potts_beta_min,
            "potts_beta_max": self.potts_beta_max,
            "swd_n_projections": self.swd_n_projections,
            "sefl_every_n_steps": self.sefl_every_n_steps,
            "sefl_patch_cells": self.sefl_patch_cells,
            "sefl_n_line_points": self.sefl_n_line_points,
            "sefl_min_stratum_cells": self.sefl_min_stratum_cells,
            "sefl_collapse_min_steps": self.sefl_collapse_min_steps,
            "sefl_genes_per_step": self.sefl_genes_per_step,
            "sefl_min_segment_nn_multiple": self.sefl_min_segment_nn_multiple,
            "sefl_rejection_max_rounds": self.sefl_rejection_max_rounds,
            "sefl_mmd_n_bandwidths": self.sefl_mmd_n_bandwidths,
            "sefl_mmd_bandwidth_step": self.sefl_mmd_bandwidth_step,
            "sefl_sinkhorn_iters": self.sefl_sinkhorn_iters,
            "sefl_sinkhorn_blur_multiple": self.sefl_sinkhorn_blur_multiple,
            "sefl_max_distribution_points": self.sefl_max_distribution_points,
            "sefl_module_min_genes": self.sefl_module_min_genes,
            "sefl_thick_grid": self.sefl_thick_grid,
            "sefl_morans_k": self.sefl_morans_k,
            "sefl_consistency_ratio_warn": self.sefl_consistency_ratio_warn,
            "thickness_ratio": self.thickness_ratio,
            "cfm_sigma_min": self.cfm_sigma_min,
            "ode_steps": self.ode_steps,
            "genes_per_step": self.genes_per_step,
            "zinb_eps": self.zinb_eps,
            "zinb_theta_min": self.zinb_theta_min,
            "zinb_theta_max": self.zinb_theta_max,
            "zinb_mu_min": self.zinb_mu_min,
            "zinb_mu_max": self.zinb_mu_max,
            "latent_encoder_hidden": self.latent_encoder_hidden,
            "latent_encoder_layers": self.latent_encoder_layers,
            "flow_hidden": self.flow_hidden,
            "flow_layers": self.flow_layers,
            "flow_time_dim": self.flow_time_dim,
            "decoder_hidden": self.decoder_hidden,
            "decoder_layers": self.decoder_layers,
            "size_factor_hidden": self.size_factor_hidden,
            "size_factor_min": self.size_factor_min,
            "z_embed_dim": self.z_embed_dim,
            "detection_rate_tol": self.detection_rate_tol,
            "independent_donor_k": self.independent_donor_k,
            "cross_mix_weight_tol": self.cross_mix_weight_tol,
            "intensity_basis_ell_multiple": self.intensity_basis_ell_multiple,
            "grad_clip": self.grad_clip,
            "log_every": self.log_every,
            "profile_n_bins": self.profile_n_bins,
            "profile_grid_size": self.profile_grid_size,
            "profile_sigma_frac": self.profile_sigma_frac,
            "loso_every_k_steps": self.loso_every_k_steps,
            "loso_max_cells": self.loso_max_cells,
            "loso_epoch_steps": self.loso_epoch_steps,
            "metric_huber_delta": self.metric_huber_delta,
            "metric_marker_genes": self.metric_marker_genes,
            "metric_marker_min_detection": self.metric_marker_min_detection,
            "metric_distribution_points": self.metric_distribution_points,
            "metric_sinkhorn_blur_multiple": self.metric_sinkhorn_blur_multiple,
            "metric_dominance_ratio_warn": self.metric_dominance_ratio_warn,
            "metric_type_grid_ell_multiple": self.metric_type_grid_ell_multiple,
            "metric_eps": self.metric_eps,
            "ceiling_n_draws": self.ceiling_n_draws,
            "n_uncertainty_samples": self.n_uncertainty_samples,
            "bisection_max_iter": self.bisection_max_iter,
            "bisection_grid_size": self.bisection_grid_size,
            "calibration_ell_max_extent_frac": self.calibration_ell_max_extent_frac,
            "potts_rare_prevalence": self.potts_rare_prevalence,
            "potts_rare_retention": self.potts_rare_retention,
            "calibration_ell_max_fitted_multiple": self.calibration_ell_max_fitted_multiple,
            "calibration_morans_tol": self.calibration_morans_tol,
            "calibration_max_cells": self.calibration_max_cells,
            "calibration_ell_z_max_fitted_multiple": self.calibration_ell_z_max_fitted_multiple,
            "calibration_ell_z_min_factor": self.calibration_ell_z_min_factor,
            "calibration_detection_ridge": self.calibration_detection_ridge,
            "calibration_logit_max": self.calibration_logit_max,
            "anchor_variance_bins": self.anchor_variance_bins,
            "anchor_weight_grid_size": self.anchor_weight_grid_size,
            "boundary_margin_spacings": self.boundary_margin_spacings,
            "selection_reduced_epoch_frac": self.selection_reduced_epoch_frac,
            "claim_tie_break_envelope": self.claim_tie_break_envelope,
            "selection_convergence_tol": self.selection_convergence_tol,
            "selection_passes": self.selection_passes,
            "selection_budget_multiple": self.selection_budget_multiple,
            "selection_metric_weight": self.selection_metric_weight,
            "selection_mixing_knn_k": self.selection_mixing_knn_k,
            "selection_n_folds": self.selection_n_folds,
            "epochs": self.epochs,
            "train_steps": self.train_steps,
            "batch_cells": self.batch_cells,
            "lr": self.lr,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ConfigError(f"Config.{name}={value!r} must be > 0")

        non_negative: dict[str, float] = {
            "seed": self.seed,
            "fourier_bands_xy": self.fourier_bands_xy,
            "fourier_bands_z": self.fourier_bands_z,
            "tv_z_weight": self.tv_z_weight,
            "potts_beta": self.potts_beta,
            "retrieval_w_z": self.retrieval_w_z,
            "retrieval_w_niche": self.retrieval_w_niche,
            "w_recon": self.w_recon,
            "w_cfm": self.w_cfm,
            "w_size": self.w_size,
            "weight_decay": self.weight_decay,
            "w_layout": self.w_layout,
            "w_autocorr": self.w_autocorr,
            "w_profile": self.w_profile,
            "w_distribution": self.w_distribution,
            "w_cross": self.w_cross,
            "w_thick": self.w_thick,
            "w_prog": self.w_prog,
            "w_prog_wrong": self.w_prog_wrong,
            "w_distill": self.w_distill,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ConfigError(f"Config.{name}={value!r} must be >= 0")

    def _check_fractions(self) -> None:
        """Check that every field expressed as a fraction lies in its interval."""
        unit_interval: dict[str, float] = {
            "detection_rate_tol": self.detection_rate_tol,
            "lr_min_frac": self.lr_min_frac,
            "residual_gate_warmup_frac": self.residual_gate_warmup_frac,
            "section_dropout_p": self.section_dropout_p,
            "sefl_warmup_frac": self.sefl_warmup_frac,
            "sefl_ramp_frac": self.sefl_ramp_frac,
            "sefl_collapse_warn_fraction": self.sefl_collapse_warn_fraction,
            "variogram_min_structured_frac": self.variogram_min_structured_frac,
            "variogram_min_saturation": self.variogram_min_saturation,
            "calibration_ell_max_extent_frac": self.calibration_ell_max_extent_frac,
            "calibration_morans_tol": self.calibration_morans_tol,
            "generation_empty_pool_tol": self.generation_empty_pool_tol,
            "selection_reduced_epoch_frac": self.selection_reduced_epoch_frac,
            "selection_convergence_tol": self.selection_convergence_tol,
            "claim_tie_break_envelope": self.claim_tie_break_envelope,
            "potts_rare_prevalence": self.potts_rare_prevalence,
            "potts_rare_retention": self.potts_rare_retention,
        }
        for name, value in unit_interval.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"Config.{name}={value!r} must lie in [0, 1]")
        if not 0.0 < self.variogram_max_lag_frac <= 1.0:
            raise ConfigError(
                f"Config.variogram_max_lag_frac={self.variogram_max_lag_frac!r} must lie in (0, 1]"
            )
        if not 0.0 < self.ema_decay < 1.0:
            raise ConfigError(f"Config.ema_decay={self.ema_decay!r} must lie in (0, 1)")
        if not 0.0 < self.repulsion_r0_percentile < 100.0:
            raise ConfigError(
                f"Config.repulsion_r0_percentile={self.repulsion_r0_percentile!r} "
                "must lie in (0, 100)"
            )
        if not 0.0 < self.repulsion_gamma_min <= 1.0:
            raise ConfigError(
                f"Config.repulsion_gamma_min={self.repulsion_gamma_min!r} must lie in (0, 1]; "
                "phi = -log(gamma) is infinite at gamma = 0, which is what the hard core r0 "
                "already provides"
            )
        if not 0.0 < self.swd_polish_step <= 1.0:
            raise ConfigError(f"Config.swd_polish_step={self.swd_polish_step!r} must lie in (0, 1]")
        if self.layout_envelope_slack < 1.0:
            raise ConfigError(
                f"Config.layout_envelope_slack={self.layout_envelope_slack!r} must be >= 1.0; "
                "an envelope below the true maximum invalidates the rejection sampler"
            )

    def _check_relations(self) -> None:
        """Check the constraints that involve more than one field."""
        if not 0.0 <= self.sefl_min_angle_deg < self.sefl_max_angle_deg <= 180.0:
            raise ConfigError(
                "Config requires 0 <= sefl_min_angle_deg < sefl_max_angle_deg <= 180, got "
                f"({self.sefl_min_angle_deg}, {self.sefl_max_angle_deg})"
            )
        if self.holdout_consecutive_k not in (1, 3, 5):
            raise ConfigError(
                f"Config.holdout_consecutive_k={self.holdout_consecutive_k!r} must be 1, 3 or 5"
            )
        if self.holdout_consecutive_k % 2 == 0:
            raise ConfigError(
                "Config.holdout_consecutive_k must be odd so the held-out run is symmetric "
                "about its centre"
            )
        if self.min_sections_per_volume < 3:
            raise ConfigError(
                "Config.min_sections_per_volume must be >= 3: a volume with fewer sections "
                "supports neither leave-one-section-out nor a flanking pair"
            )
        if self.latent_dim > self.n_rff:
            raise ConfigError(
                f"Config.latent_dim={self.latent_dim} exceeds Config.n_rff={self.n_rff}; the "
                "noise field's amplitude columns could not be made independent"
            )
        if self.variogram_n_bins < 3:
            raise ConfigError(
                f"Config.variogram_n_bins={self.variogram_n_bins} must be >= 3: a nugget, a "
                "sill and a length-scale cannot be fitted to fewer points"
            )
        if self.field_mlp_layers < 2:
            raise ConfigError(
                f"Config.field_mlp_layers={self.field_mlp_layers} must be >= 2; a single "
                "layer makes the field a linear read-out of the triplane features"
            )
        if self.layout_mlp_layers < 2:
            raise ConfigError(
                f"Config.layout_mlp_layers={self.layout_mlp_layers} must be >= 2; a single "
                "layer makes the log-intensity a linear read-out of the field feature"
            )
        if self.pcf_n_bins < 3:
            raise ConfigError(
                f"Config.pcf_n_bins={self.pcf_n_bins} must be >= 3: g(r) is fitted against "
                "over the range [r0, R], which cannot be resolved by fewer bins"
            )
        if self.repulsion_gamma_grid_size < 2:
            raise ConfigError(
                f"Config.repulsion_gamma_grid_size={self.repulsion_gamma_grid_size} must be "
                ">= 2: a one-point grid is a hand-set gamma, which T05 forbids"
            )
        if not 0.0 < self.potts_beta_min < self.potts_beta_max:
            raise ConfigError(
                f"Config requires 0 < potts_beta_min < potts_beta_max, got "
                f"({self.potts_beta_min}, {self.potts_beta_max})"
            )
        if self.potts_beta_grid_size < 2:
            raise ConfigError(
                f"Config.potts_beta_grid_size={self.potts_beta_grid_size} must be >= 2: a "
                "one-point grid is a hand-set beta, which T05 forbids"
            )
        if self.niche_scale_factor <= 1.0:
            raise ConfigError(
                f"Config.niche_scale_factor={self.niche_scale_factor!r} must be > 1: the "
                "niche's scales must be distinct"
            )
        if not 0.0 < self.rotation_bias_max_tilt_deg <= 180.0:
            raise ConfigError(
                f"Config.rotation_bias_max_tilt_deg={self.rotation_bias_max_tilt_deg!r} "
                "must lie in (0, 180]"
            )
        if not math.isfinite(self.retrieval_z_window_gap_factor):
            raise ConfigError(
                f"Config.retrieval_z_window_gap_factor="
                f"{self.retrieval_z_window_gap_factor!r} must be finite"
            )
        if self.retrieval_z_window_gap_factor < 1.0:
            raise ConfigError(
                f"Config.retrieval_z_window_gap_factor="
                f"{self.retrieval_z_window_gap_factor!r} must be >= 1. Below 1 the query's "
                "own nearest admissible section falls outside its own window, the candidate "
                "pool can be empty however much evidence the stack holds, and the guarantee "
                "the relative term exists to provide is gone"
            )
        if self.retrieval_candidates_per_section < self.retrieval_k:
            raise ConfigError(
                f"Config.retrieval_candidates_per_section="
                f"{self.retrieval_candidates_per_section} is below "
                f"Config.retrieval_k={self.retrieval_k}. With two admissible sections the "
                "candidate union would then be at most K, the top-K would select all of it, "
                "and the retrieval score would decide nothing — silently, and exactly in the "
                "wide-gap regime the z-proximity term exists for"
            )
        if self.flow_time_dim % 2 != 0:
            raise ConfigError(
                f"Config.flow_time_dim={self.flow_time_dim} must be even: the embedding is "
                "(sin, cos) pairs over flow_time_dim / 2 frequencies"
            )
        if self.latent_encoder_layers < 2:
            raise ConfigError(
                f"Config.latent_encoder_layers={self.latent_encoder_layers} must be >= 2; a "
                "single layer makes h1 a linear read-out of the pooled expression"
            )
        if self.flow_layers < 2:
            raise ConfigError(
                f"Config.flow_layers={self.flow_layers} must be >= 2; a linear velocity "
                "field cannot transport a Gaussian onto anything else"
            )
        if self.decoder_layers < 2:
            raise ConfigError(
                f"Config.decoder_layers={self.decoder_layers} must be >= 2; a single layer "
                "makes every ZINB parameter a linear read-out of u_ig"
            )
        if not self.zinb_theta_min < self.zinb_theta_max:
            raise ConfigError(
                f"Config requires zinb_theta_min < zinb_theta_max, got "
                f"({self.zinb_theta_min}, {self.zinb_theta_max})"
            )
        if not self.zinb_mu_min < self.zinb_mu_max:
            raise ConfigError(
                f"Config requires zinb_mu_min < zinb_mu_max, got "
                f"({self.zinb_mu_min}, {self.zinb_mu_max})"
            )
        if self.independent_donor_k < 2:
            raise ConfigError(
                f"Config.independent_donor_k={self.independent_donor_k} must be >= 2: with "
                "one donor the baseline is a verbatim copy and has no chimerism to measure"
            )
        if self.selection_budget_multiple <= 1.0:
            raise ConfigError(
                f"Config.selection_budget_multiple={self.selection_budget_multiple!r} must be "
                "> 1: the joint gate's two budget cells would otherwise be the same budget, "
                "which is the null comparison specs/09 3's third requirement exists to forbid"
            )
        if self.anchor_variance_bins < 2:
            raise ConfigError(
                f"Config.anchor_variance_bins={self.anchor_variance_bins} must be >= 2: a "
                "one-bin isotonic map is a hand-set blend weight, which specs/09 1 forbids"
            )
        if self.anchor_weight_grid_size < 2:
            raise ConfigError(
                f"Config.anchor_weight_grid_size={self.anchor_weight_grid_size} must be >= 2: "
                "the grid has to contain both endpoints of [0, 1] to express 'never anchor' "
                "and 'always anchor'"
            )
        if self.selection_passes < 1:
            raise ConfigError(
                f"Config.selection_passes={self.selection_passes} must be >= 1: with no pass "
                "the selector returns the incumbent and selects nothing"
            )
        if self.calibration_ell_z_min_factor >= self.calibration_ell_z_max_fitted_multiple:
            raise ConfigError(
                "Config requires calibration_ell_z_min_factor < "
                "calibration_ell_z_max_fitted_multiple, got "
                f"({self.calibration_ell_z_min_factor}, "
                f"{self.calibration_ell_z_max_fitted_multiple}); the ell_z bracket would be "
                "empty and R1's remedy 3 has nothing to bracket"
            )
        if self.retrieval_ctx_dim % self.retrieval_n_heads != 0:
            raise ConfigError(
                f"Config.retrieval_ctx_dim={self.retrieval_ctx_dim} must be divisible by "
                f"retrieval_n_heads={self.retrieval_n_heads}"
            )
