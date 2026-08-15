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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

__all__ = [
    "DECODERS",
    "DEVICES",
    "EXPR_MODES",
    "HOLDOUT_MODES",
    "LAYOUT_MODES",
    "PRIOR_MODES",
    "ROTATION_BIASES",
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
ROTATION_BIASES: Final[frozenset[str]] = frozenset({"uniform", "axial"})
PRIOR_MODES: Final[frozenset[str]] = frozenset({"correlated", "iid"})
EXPR_MODES: Final[frozenset[str]] = frozenset({"zinb-flow", "cross-mix", "auto-blend"})
TEXT_EMB_MODES: Final[frozenset[str]] = frozenset({"medcpt", "lookup"})
DECODERS: Final[frozenset[str]] = frozenset({"zinb", "zigamma", "gaussian"})
DEVICES: Final[frozenset[str]] = frozenset({"auto", "cpu", "cuda"})
TEXT_POOLINGS: Final[frozenset[str]] = frozenset({"cls", "mean"})
HOLDOUT_MODES: Final[frozenset[str]] = frozenset({"alternating", "consecutive"})


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

    mygene_species: str = "human,mouse"
    """Species queried when ``build_gene_meta`` goes online; the datasets in T10 are
    mouse and human."""

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
    """Triplane sets, each at a fixed maximally-separated rotation. GATE 2's first
    remedy is raising this to 8."""

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
    """Candidate pool half-width along z, in units of median section spacing."""

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

    # ----------------------------------------------------------------------------------
    # expression (T06)
    # ----------------------------------------------------------------------------------
    cfm_sigma_min: float = 1e-4
    """Minimum path noise in conditional flow matching."""

    ode_steps: int = 24
    """Heun steps in the sampling ODE."""

    genes_per_step: int = 128
    """G': genes subsampled per training step, so the decoder never sees a fixed panel
    width. This is what makes panel width irrelevant."""

    zinb_eps: float = 1e-6
    """Additive epsilon on the ZINB dispersion."""

    # ----------------------------------------------------------------------------------
    # losses
    # ----------------------------------------------------------------------------------
    w_recon: float = 1.0
    """Reconstruction (ZINB NLL). Must dominate throughout training."""

    w_layout: float = 1.0
    """Poisson-process layout NLL."""

    w_autocorr: float = 0.5
    """Metric-aware Moran's I / Geary's C agreement (T08)."""

    w_profile: float = 0.5
    """Metric-aware depth / field / per-type profile agreement (T08)."""

    w_distribution: float = 0.5
    """Metric-aware Sinkhorn (or MMD) distribution matching (T08)."""

    w_cross: float = 0.3
    """SEFL plane-intersection consistency (T07). ``0`` participates in ablation A7."""

    w_thick: float = 0.2
    """SEFL thickness coarse-graining consistency (T07)."""

    w_prog: float = 0.2
    """SEFL molecular-program invariance (T07)."""

    w_distill: float = 0.1
    """Text -> free-residual distillation (T02)."""

    sefl_warmup_frac: float = 0.2
    """Fraction of training spent on reconstruction only before the SEFL terms ramp in."""

    ema_decay: float = 0.999
    """EMA decay for the teacher weights. The teacher plus stop-gradient is what stops
    the symmetric consistency losses collapsing to a constant field."""

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

    # ----------------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------------
    epochs: int = 200
    """Training epochs."""

    batch_cells: int = 2048
    """Cells per training batch."""

    lr: float = 3e-4
    """AdamW learning rate."""

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
            "sefl_every_n_steps": self.sefl_every_n_steps,
            "sefl_patch_cells": self.sefl_patch_cells,
            "sefl_n_line_points": self.sefl_n_line_points,
            "sefl_min_stratum_cells": self.sefl_min_stratum_cells,
            "thickness_ratio": self.thickness_ratio,
            "cfm_sigma_min": self.cfm_sigma_min,
            "ode_steps": self.ode_steps,
            "genes_per_step": self.genes_per_step,
            "zinb_eps": self.zinb_eps,
            "profile_n_bins": self.profile_n_bins,
            "profile_grid_size": self.profile_grid_size,
            "profile_sigma_frac": self.profile_sigma_frac,
            "loso_every_k_steps": self.loso_every_k_steps,
            "loso_max_cells": self.loso_max_cells,
            "n_uncertainty_samples": self.n_uncertainty_samples,
            "bisection_max_iter": self.bisection_max_iter,
            "bisection_grid_size": self.bisection_grid_size,
            "calibration_ell_max_extent_frac": self.calibration_ell_max_extent_frac,
            "calibration_ell_max_fitted_multiple": self.calibration_ell_max_fitted_multiple,
            "epochs": self.epochs,
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
            "w_layout": self.w_layout,
            "w_autocorr": self.w_autocorr,
            "w_profile": self.w_profile,
            "w_distribution": self.w_distribution,
            "w_cross": self.w_cross,
            "w_thick": self.w_thick,
            "w_prog": self.w_prog,
            "w_distill": self.w_distill,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ConfigError(f"Config.{name}={value!r} must be >= 0")

    def _check_fractions(self) -> None:
        """Check that every field expressed as a fraction lies in its interval."""
        unit_interval: dict[str, float] = {
            "residual_gate_warmup_frac": self.residual_gate_warmup_frac,
            "section_dropout_p": self.section_dropout_p,
            "sefl_warmup_frac": self.sefl_warmup_frac,
            "variogram_min_structured_frac": self.variogram_min_structured_frac,
            "variogram_min_saturation": self.variogram_min_saturation,
            "calibration_ell_max_extent_frac": self.calibration_ell_max_extent_frac,
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
        if self.retrieval_candidates_per_section < self.retrieval_k:
            raise ConfigError(
                f"Config.retrieval_candidates_per_section="
                f"{self.retrieval_candidates_per_section} is below "
                f"Config.retrieval_k={self.retrieval_k}. With two admissible sections the "
                "candidate union would then be at most K, the top-K would select all of it, "
                "and the retrieval score would decide nothing — silently, and exactly in the "
                "wide-gap regime the z-proximity term exists for"
            )
        if self.retrieval_ctx_dim % self.retrieval_n_heads != 0:
            raise ConfigError(
                f"Config.retrieval_ctx_dim={self.retrieval_ctx_dim} must be divisible by "
                f"retrieval_n_heads={self.retrieval_n_heads}"
            )
