"""
SpatialCPA-v13 — an LLM-based virtual-slice generator for aligned serial spatial
transcriptomics.

v13 is a **transformer language model over cell-sentences**. Each cell is tokenized into
a rank-ordered sequence of gene tokens (Cell2Sentence / Geneformer / scGPT style); a
self-attention transformer is trained with a masked gene-language-model objective and a
spatial in-context objective; and virtual slices are produced by retrieval-augmented
in-context generation (cross-attend over retrieved flanking cells -> sample a real
exemplar from the LM-similarity distribution -> emit a generated, grounded profile).

This is a paradigm distinct from every other SpatialCPA version — no optimal transport /
morph / fusion / niche-MRF (v8), and no implicit coordinate-field or factor-analysis
decoder (v11/v12).

    from spatialcpav13 import SpatialCPAv13, SpatialCPAv13Config, Slice, SliceStack
    gen = SpatialCPAv13(stack, gene_names=genes, cell_type_names=types)
    vs = gen.generate_virtual_slice(z=target_z)
"""

from .config import (
    SpatialCPAv13Config, TokenizerConfig, ModelConfig, TrainConfig, LossConfig,
    PretrainedLMConfig, GenerationConfig,
)
from .data import Slice, SliceStack
from .model import SpatialCPAv13, VirtualSlice
from .tokenizer import CellSentenceTokenizer

__version__ = "13.0.0"

__all__ = [
    "SpatialCPAv13", "VirtualSlice", "SpatialCPAv13Config", "TokenizerConfig",
    "ModelConfig", "TrainConfig", "LossConfig", "PretrainedLMConfig",
    "GenerationConfig", "Slice", "SliceStack", "CellSentenceTokenizer",
]
