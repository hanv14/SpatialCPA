"""Model components: text-grounded embeddings, the noise and anatomical fields, the heads."""

from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    TextGroundedEmbedding,
    text_embedding_diagnostics,
)
from spatialcpav25_gen.model.field import (
    BBoxClampWarning,
    RotationContext,
    RotationError,
    TriplaneField,
    fourier_encode,
    fourier_encode_dim,
    orientation_rotations,
    random_rotation,
)
from spatialcpav25_gen.model.noise import (
    GaussianRandomField,
    LengthscaleFit,
    fit_lengthscale_from_sections,
    matern_correlation,
    scaled_distance,
)
from spatialcpav25_gen.model.retrieval import (
    EmptyCandidatePoolWarning,
    RetrievalAttention,
    RetrievalIndex,
    attention_entropy,
)

__all__ = [
    "BBoxClampWarning",
    "EmptyCandidatePoolWarning",
    "EntityEmbeddings",
    "GaussianRandomField",
    "LengthscaleFit",
    "RetrievalAttention",
    "RetrievalIndex",
    "RotationContext",
    "RotationError",
    "TextGroundedEmbedding",
    "TriplaneField",
    "attention_entropy",
    "fit_lengthscale_from_sections",
    "fourier_encode",
    "fourier_encode_dim",
    "matern_correlation",
    "orientation_rotations",
    "random_rotation",
    "scaled_distance",
    "text_embedding_diagnostics",
]
