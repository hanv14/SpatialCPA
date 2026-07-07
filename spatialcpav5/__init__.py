"""
SpatialCPA-v5 — a Transformer for 3D spatial-transcriptomics virtual slices.

Learns the mapping ``{Slice(i-1), Slice(i+1)} -> Slice(i)`` directly: for every
target location the model attends over the k nearest neighbors from each
flanking slice (as transformer tokens plus a CLS token) and predicts gene
expression, cell type / region labels, and tissue occupancy.

This is a *separate, self-contained* implementation from the coordinate-field
``spatialcpa`` package; the original is untouched.

Public API
----------
Configuration
    :class:`SpatialCPAv5Config` and its sub-configs (``ModelConfig`` etc.).
Data
    :class:`Slice`, :class:`SliceStack`, :func:`build_triplet_samples`.
Model
    :class:`SpatialCPATransformer`.
Training
    :class:`Trainer`, :func:`load_model`.
Inference
    :class:`Predictor`, :class:`SlicePrediction`.
"""

from .config import (
    DataConfig,
    InferenceConfig,
    LossConfig,
    ModelConfig,
    SpatialCPAv5Config,
    TrainConfig,
)
from .data import Slice, SliceStack, TripletSamples, build_triplet_samples
from .dataset import TripletTokenDataset
from .inference import Predictor, SlicePrediction
from .model import SpatialCPATransformer
from .trainer import Trainer, load_model, set_seed

__all__ = [
    # config
    "SpatialCPAv5Config",
    "ModelConfig",
    "DataConfig",
    "LossConfig",
    "TrainConfig",
    "InferenceConfig",
    # data
    "Slice",
    "SliceStack",
    "TripletSamples",
    "build_triplet_samples",
    "TripletTokenDataset",
    # model
    "SpatialCPATransformer",
    # training
    "Trainer",
    "load_model",
    "set_seed",
    # inference
    "Predictor",
    "SlicePrediction",
]

__version__ = "5.0.0"
