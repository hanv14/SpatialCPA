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
from spatialcpav25_gen.data.text import (
    GeneMeta,
    GeneMetaUnavailableWarning,
    TextEncoder,
    build_gene_meta,
    celltype_descriptor,
    descriptor_key,
    gene_descriptor,
    load_gene_meta,
    region_descriptor,
)

__all__ = [
    "AssumedThicknessWarning",
    "GeneMeta",
    "GeneMetaUnavailableWarning",
    "HeldOutSections",
    "NonIntegerCountsWarning",
    "OverlappingSlabsWarning",
    "SchemaError",
    "Section",
    "TextEncoder",
    "TrainingVolume",
    "Volume",
    "build_gene_meta",
    "celltype_descriptor",
    "descriptor_key",
    "gene_descriptor",
    "load_gene_meta",
    "load_volume",
    "loso_folds",
    "region_descriptor",
    "split_holdout",
    "to_xyz",
    "validate_config_against_volume",
    "validate_volume",
]
