"""
Configuration for SpatialCPA-v13 — an **LLM-based** virtual-slice generator.

v13 is a *transformer language model* over spatial transcriptomics. Each cell is
tokenized into a **cell-sentence** — a rank-ordered sequence of gene tokens
(Cell2Sentence / Geneformer / scGPT style) — and a self-attention transformer is trained
as a language model over these sentences with (i) a **masked gene-language-model**
objective (learn the gene "grammar" / co-occurrence) and (ii) a **spatial in-context**
objective (reconstruct a cell from the flanking slices' neighbouring cells attended in
context). Virtual slices are produced by **retrieval-augmented in-context generation**:
for a query position the model builds a context embedding by cross-attending over the
retrieved flanking cells, samples a real exemplar from the LM-similarity distribution,
predicts the cell type from the [CLS] embedding, and emits a generated expression
profile grounded in the retrieved biology.

This is a paradigm distinct from every other SpatialCPA version: no optimal transport /
morph / fusion / niche-MRF (v8), and no implicit coordinate-network field or Gaussian
factor-analysis decoder (v11/v12). The engine is a tokenizer + a self-attention
transformer + retrieval-augmented decoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenizerConfig:
    """Cell-sentence tokenizer (rank encoding of expression into gene tokens)."""
    top_genes: int = 32           # genes per cell-sentence (top expressed, rank-ordered)
    min_expr: float = 0.0         # only tokens for genes above this (normalized) value
    n_zbins: int = 16             # discretized z tokens (continuous z -> nearest bin token)


@dataclass
class ModelConfig:
    """Self-attention transformer language model over cell-sentences."""
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    max_len: int = 40             # [CLS] + top_genes (+ slack)
    n_context: int = 16           # flanking neighbour cells attended in context (RAG)


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_cells: int = 256        # cells per LM step
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    grad_clip: float = 2.0
    mlm_mask_frac: float = 0.15   # fraction of gene tokens masked for the MLM objective
    device: str = "auto"          # "auto" | "cpu" | "cuda"
    seed: int = 42
    verbose: bool = True
    fallback_on_error: bool = True


@dataclass
class LossConfig:
    w_mlm: float = 1.0            # masked gene-language-model loss
    w_context_expr: float = 1.0  # spatial in-context expression reconstruction (MSE)
    w_context_type: float = 1.0  # spatial in-context cell-type prediction (CE)
    w_align: float = 0.5         # context<->cell embedding alignment (retrieval contrastive)


@dataclass
class PretrainedLMConfig:
    """Optional real pretrained gene-language-model encoder hook.

    A pretrained single-cell language model (Cell2Sentence/GPT-2, scGPT, Geneformer)
    can be plugged in as the cell encoder via ``register_pretrained_lm``. When no
    checkpoint is supplied, v13 trains its own self-contained transformer LM (clearly
    logged), so the method always runs.
    """
    kind: str = "none"           # "none" (train own LM) | registered pretrained-LM name
    weights_path: str | None = None
    device: str = "auto"


@dataclass
class GenerationConfig:
    """Retrieval-augmented in-context generation of a virtual slice."""
    retrieval_temp: float = 0.2   # temperature for sampling the real exemplar from LM-similarity
    retrieval_k: int = 24         # candidate real cells considered per query (RAG neighbourhood)
    edit_weight: float = 0.15     # blend toward the LM-decoded profile (0 = pure exemplar)
    gene_edit: bool = True        # apply the LM gene-sentence modulation to the exemplar
    edit_strength: float = 0.5    # strength of the generated gene-sentence modulation
    output_counts: bool = True    # emit count-like (expm1) expression for the evaluator
    composition_match: bool = True  # match cell-type composition to the interpolated flanking mix
    # Position generation (retrieval layout): "flanking" draws real flanking positions
    # (z-weighted) — a two-slice recombined layout; "nearest" uses the single nearest
    # slice; "auto" switches to "nearest" only for extremely near-identical planes.
    position_mode: str = "flanking"   # "flanking" | "nearest" | "auto"
    near_identical_ratio: float = 0.60  # cross/within-slice spacing threshold for "near-identical"


@dataclass
class SpatialCPAv13Config:
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    pretrained_lm: PretrainedLMConfig = field(default_factory=PretrainedLMConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 42
