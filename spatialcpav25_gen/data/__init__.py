"""Data contracts and loading: the validated schema everything else is written against."""

from spatialcpav25_gen.data.loaders import load_volume, loso_folds, split_holdout
from spatialcpav25_gen.data.schema import (
    AssumedThicknessWarning,
    HeldOutSections,
    NonIntegerCountsWarning,
    OverlappingSlabsWarning,
    SchemaError,
    Section,
    TrainingVolume,
    Volume,
    to_xyz,
    validate_config_against_volume,
    validate_volume,
)

__all__ = [
    "AssumedThicknessWarning",
    "HeldOutSections",
    "NonIntegerCountsWarning",
    "OverlappingSlabsWarning",
    "SchemaError",
    "Section",
    "TrainingVolume",
    "Volume",
    "load_volume",
    "loso_folds",
    "split_holdout",
    "to_xyz",
    "validate_config_against_volume",
    "validate_volume",
]
