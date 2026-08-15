"""Model components: text-grounded embeddings, the noise and anatomical fields, the heads."""

from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    TextGroundedEmbedding,
    text_embedding_diagnostics,
)

__all__ = [
    "EntityEmbeddings",
    "TextGroundedEmbedding",
    "text_embedding_diagnostics",
]
