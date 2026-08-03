"""Every knob for SpatialCPA-v16. Defaults are the production settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HarmonicConfig:
    # Number of tissue harmonics modelled. This is the method's one genuinely
    # important hyper-parameter: it sets where "structure the model generates"
    # stops and "dispersion the residual stage restores" begins. 64 puts the
    # cut at a spatial scale of roughly a dozen cells on the sections here —
    # coarse enough that the modelled part is anatomy, fine enough that the
    # residual is genuinely unstructured.
    n_modes: int = 64
    # kNN degree for the tissue graph. Matched to the evaluator's SPATIAL_K so
    # the basis v16 generates in is the basis the metric is computed in; a
    # mismatch would not be wrong, but it would make the method's own Moran
    # prediction disagree with the score for no reason.
    k_neighbors: int = 10
    n_freq_bins: int = 24


@dataclass
class GeometryConfig:
    grid_bins: int = 96
    density_smooth: float = 2.0
    occupancy_quantile: float = 0.55
    jitter: float = 0.6


@dataclass
class FlowConfig:
    gene_components: int = 32          # gene-space PCs the spectrum is modelled in
    width: int = 192
    depth: int = 4
    heads: int = 4
    n_fourier: int = 6
    n_context: int = 2                 # bracketing sections per query
    dropout: float = 0.0
    epochs: int = 600
    batch_size: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-4
    ode_steps: int = 16
    n_ensemble: int = 8                # ODE draws averaged per query
    # Whole context sections dropped at random during training, so the model
    # learns to widen its prediction as context recedes — the same gap-aware
    # idea v14 uses, applied to sections rather than cells.
    context_dropout: float = 0.25
    z_jitter: float = 0.15             # fraction of the local gap


@dataclass
class ExpressionConfig:
    # Fraction of the *residual* variance restored. 1.0 reproduces the training
    # sections' dispersion exactly; the knob exists because a method may be run
    # on a dataset whose held-out sections are noisier than its training ones.
    residual_scale: float = 1.0
    # Draw residuals from cells of the same assigned type. Off falls back to a
    # tissue-wide pool, which keeps gene-gene covariance but loses type-specific
    # dispersion.
    residual_by_type: bool = True
    calibrate: bool = True             # the spectral calibration stage
    readout: str = "auto"              # auto | counts | continuous


@dataclass
class TypeConfig:
    # Cell types are generated as signals on the same graph — one indicator
    # field per type, spectrally — then sampled with a composition constraint.
    # Type fields are low-frequency by nature, which is why this basis suits
    # them and why paper_celltype_localization is the metric it targets.
    n_modes: int = 32
    composition_match: bool = True
    temperature: float = 1.0


@dataclass
class RuntimeConfig:
    """Where the flow trains, and how co-operatively it holds the card.

    See ``runtime.py``. The defaults are the sharing-friendly ones: use a GPU if
    one is visible, let the allocator's pools grow and shrink with demand rather
    than reserve fixed blocks, and hand cached memory back between stages. The
    hard cap is off by default because a cap set too tight turns a slow run into
    a crashed one.
    """
    device: str = "auto"               # auto | cpu | cuda
    cuda_index: int = 0
    expandable_segments: bool = True   # pools grow/shrink instead of reserving
    memory_fraction: float | None = None   # optional hard ceiling, e.g. 0.4
    release_cache: bool = True         # empty_cache between stages


@dataclass
class SpatialCPAv16Config:
    harmonics: HarmonicConfig = field(default_factory=HarmonicConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    types: TypeConfig = field(default_factory=TypeConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    seed: int = 42
