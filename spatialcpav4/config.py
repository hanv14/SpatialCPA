"""
Configuration objects for SpatialCPA-v4 (transformer version).

All hyperparameters live in typed :class:`dataclasses` grouped by concern
(model / data / loss / training / inference).  Nothing about the architecture
or optimisation schedule is hard-coded in the model or trainer — every knob is
read from one of these configs, so experiments are fully described by a single
serialisable object.

The top-level :class:`SpatialCPAv4Config` bundles the sub-configs and provides
``to_dict`` / ``from_dict`` / ``save`` / ``load`` helpers so a configuration can
be round-tripped through JSON (e.g. stored next to a checkpoint).

Design notes
------------
* Keep every default sane but overridable.  A wrapper or a notebook should be
  able to construct the whole thing with no arguments and get a working model.
* ``dataclasses.field(default_factory=...)`` is used for the nested configs so
  each top-level instance owns its own sub-config objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Architecture hyperparameters.

    Parameters
    ----------
    hidden_dim
        Transformer model dimension ``d_model``.  Every token component
        embedding is projected to this width and summed.
    num_layers
        Number of ``TransformerEncoderLayer`` blocks.
    num_heads
        Number of attention heads (must divide ``hidden_dim``).
    dim_feedforward
        Width of the position-wise feed-forward network inside each encoder
        layer.  ``None`` -> ``4 * hidden_dim``.
    dropout
        Dropout probability used throughout (transformer + token embedder +
        heads).
    expression_embed_dim
        Output width of the expression encoder before it is projected to
        ``hidden_dim``.  ``None`` -> ``hidden_dim``.
    coord_hidden_dim
        Hidden width of the relative-coordinate MLP encoder.
    label_head_hidden_dim
        Hidden width of the label classification heads.
    expression_head_hidden_dim
        Hidden width of the expression regression head.  ``None`` ->
        ``2 * hidden_dim``.
    expression_encoder
        Which expression encoder to instantiate.  Currently ``"linear"``; the
        registry in :mod:`spatialcpav4.encoders` makes it trivial to add
        ``"autoencoder"`` / ``"pretrained"`` later without touching the model.
    """

    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dim_feedforward: Optional[int] = None
    dropout: float = 0.1

    expression_embed_dim: Optional[int] = None
    coord_hidden_dim: int = 128
    label_head_hidden_dim: int = 128
    expression_head_hidden_dim: Optional[int] = None

    expression_encoder: str = "linear"

    # Output activation for the expression head. Expression is non-negative, so
    # "softplus" (default) prevents unphysical negative predictions; "none"
    # restores the original linear head.
    expression_activation: str = "softplus"

    def resolved_dim_feedforward(self) -> int:
        return self.dim_feedforward if self.dim_feedforward is not None else 4 * self.hidden_dim

    def resolved_expression_embed_dim(self) -> int:
        return (
            self.expression_embed_dim
            if self.expression_embed_dim is not None
            else self.hidden_dim
        )

    def resolved_expression_head_hidden_dim(self) -> int:
        return (
            self.expression_head_hidden_dim
            if self.expression_head_hidden_dim is not None
            else 2 * self.hidden_dim
        )


# --------------------------------------------------------------------------- #
# Data                                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """Data-construction and neighbor-search hyperparameters.

    Parameters
    ----------
    n_neighbors
        Number of nearest neighbors drawn from *each* flanking slice.  The
        transformer therefore sees ``2 * n_neighbors`` neighbor tokens.
    negative_ratio
        Number of synthetic background (occupancy=0) samples generated per real
        (occupancy=1) target spot.  ``0`` disables occupancy negatives.
    negative_min_dist_factor
        A random background coordinate is accepted only if its distance to the
        nearest real spot exceeds ``negative_min_dist_factor * typical_spacing``
        where ``typical_spacing`` is the median nearest-neighbor distance within
        the slice.  Keeps negatives genuinely "outside tissue".
    coord_scale
        Physical length used to normalise relative coordinates before they enter
        the coordinate encoder.  ``None`` -> estimated from the data (median
        neighbor distance).
    val_fraction
        Fraction of target spots held out for validation.
    cache_dir
        Optional directory to persist computed neighbor indices so repeated runs
        skip the KDTree queries.  ``None`` disables disk caching (in-memory
        caching within a single build always happens).
    """

    n_neighbors: int = 10
    negative_ratio: float = 1.0
    negative_min_dist_factor: float = 2.0
    coord_scale: Optional[float] = None
    val_fraction: float = 0.1
    cache_dir: Optional[str] = None


# --------------------------------------------------------------------------- #
# Loss                                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class LossConfig:
    """Loss term weights.

    ``total = expression_weight * (mse_weight * MSE + pearson_weight * Pearson)
             + label_weight * (cell_type_weight * CE_ct + region_weight * CE_reg)
             + occupancy_weight * BCE``
    """

    # Top-level term weights
    expression_weight: float = 1.0
    label_weight: float = 1.0
    occupancy_weight: float = 1.0

    # Expression sub-weights
    mse_weight: float = 1.0
    pearson_weight: float = 0.5
    # Variance-matching weight: penalizes mismatch between the per-gene standard
    # deviation of predictions and targets across the batch. Counteracts the
    # mean-collapse / over-smoothing that MSE regression induces (which shows up
    # as a near-zero per-gene variance agreement at evaluation).
    variance_weight: float = 0.5

    # Label sub-weights
    cell_type_weight: float = 1.0
    region_weight: float = 1.0


# --------------------------------------------------------------------------- #
# Training                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Optimisation and training-loop hyperparameters.

    Parameters
    ----------
    lr
        Peak learning rate for AdamW.
    weight_decay
        AdamW weight decay.
    batch_size
        Mini-batch size (number of target spots per step).
    epochs
        Maximum number of epochs.
    grad_clip
        Global gradient-norm clip value (``None`` disables).
    scheduler
        ``"cosine"`` | ``"plateau"`` | ``"none"``.
    warmup_epochs
        Linear LR warmup length (cosine scheduler only).
    early_stopping_patience
        Stop after this many epochs without validation improvement.  ``None``
        disables early stopping.
    early_stopping_min_delta
        Minimum decrease in validation loss to count as an improvement.
    mixed_precision
        Enable ``torch.cuda.amp`` autocast + gradient scaling (CUDA only).
    num_workers
        DataLoader worker processes.
    device
        ``"cuda"`` / ``"cpu"`` / ``None`` (auto-select).
    seed
        RNG seed.
    checkpoint_dir
        Where ``best.pt`` / ``last.pt`` are written.
    tensorboard_dir
        TensorBoard log directory (``None`` disables logging).
    log_every
        Log training metrics every N optimiser steps.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    grad_clip: Optional[float] = 1.0

    scheduler: str = "cosine"
    warmup_epochs: int = 5

    early_stopping_patience: Optional[int] = 15
    early_stopping_min_delta: float = 1e-4

    mixed_precision: bool = True
    num_workers: int = 0
    device: Optional[str] = None
    seed: int = 42

    checkpoint_dir: str = "checkpoints/spatialcpav4"
    tensorboard_dir: Optional[str] = None
    log_every: int = 50


# --------------------------------------------------------------------------- #
# Inference                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class InferenceConfig:
    """Prediction / virtual-slice hyperparameters.

    Parameters
    ----------
    occupancy_threshold
        Grid points whose predicted occupancy probability is below this value
        are discarded when generating a virtual slice.
    grid_points
        Default number of grid points for :func:`generate_virtual_slice`.
    batch_size
        Inference batch size.
    grid_type
        ``"regular"`` (deterministic square lattice) or ``"random"`` (uniform
        random sampling inside the bounding box).
    """

    occupancy_threshold: float = 0.5
    grid_points: int = 1000
    batch_size: int = 4096
    grid_type: str = "regular"

    # Expression source at generation:
    #   "regress"  — the model's regressed expression (smooth; collapses variance).
    #   "transfer" — copy real profiles from the nearest training cells (preserves
    #                cell-to-cell variance, as SpatialZ / original SpatialCPA do).
    #   "blend"    — transfer_alpha * regressed + (1 - transfer_alpha) * transfer.
    expression_mode: str = "regress"
    transfer_k: int = 1                    # neighbors to transfer from (1 = copy nearest)
    transfer_alpha: float = 0.0            # blend weight on the regressed expression
    transfer_same_celltype: bool = True    # transfer only from same predicted cell type

    # Where candidate cell positions come from at generation:
    #   "flanking" — the real (x, y) of the flanking slices' cells (realistic
    #                tissue density/morphology; strong placement metrics).
    #   "grid"     — a uniform lattice over the bounding box (uniform density).
    position_source: str = "flanking"


# --------------------------------------------------------------------------- #
# Top-level bundle                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class SpatialCPAv4Config:
    """Complete configuration for a SpatialCPA-v4 experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # ---- serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpatialCPAv4Config":
        """Rebuild a config from a (possibly partial) nested dict."""
        kwargs: Dict[str, Any] = {}
        sub_types = {f.name: f.type for f in fields(cls)}
        type_map = {
            "model": ModelConfig,
            "data": DataConfig,
            "loss": LossConfig,
            "train": TrainConfig,
            "inference": InferenceConfig,
        }
        for name, sub_cls in type_map.items():
            sub = d.get(name, {})
            if is_dataclass(sub):
                sub = asdict(sub)
            # Only pass known fields so future/legacy keys don't crash loading.
            valid = {f.name for f in fields(sub_cls)}
            kwargs[name] = sub_cls(**{k: v for k, v in sub.items() if k in valid})
        # Silence "unused" lint for sub_types (kept for readability of intent).
        _ = sub_types
        return cls(**kwargs)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SpatialCPAv4Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))
