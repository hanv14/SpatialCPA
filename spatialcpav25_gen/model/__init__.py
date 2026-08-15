"""Model components: text-grounded embeddings, the noise and anatomical fields, the heads."""

from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    TextGroundedEmbedding,
    text_embedding_diagnostics,
)
from spatialcpav25_gen.model.noise import (
    GaussianRandomField,
    LengthscaleFit,
    fit_lengthscale_from_sections,
    matern_correlation,
    scaled_distance,
)

__all__ = [
    "EntityEmbeddings",
    "GaussianRandomField",
    "LengthscaleFit",
    "TextGroundedEmbedding",
    "fit_lengthscale_from_sections",
    "matern_correlation",
    "scaled_distance",
    "text_embedding_diagnostics",
]
