"""
Encoders for SpatialCPA-v5 token construction.

Two families of encoders live here:

* **Expression encoders** turn a raw (possibly thousands-of-genes) expression
  vector into a compact latent embedding.  They share the
  :class:`ExpressionEncoder` interface and are created through
  :func:`build_expression_encoder`, so the linear default can later be swapped
  for an autoencoder or a pretrained foundation model (scGPT / Geneformer)
  *without touching the rest of the model* — just register a new builder.

* **Relative-coordinate encoder** maps ``[Δx, Δy, Δz, ‖Δ‖]`` to a dense
  embedding via a small MLP, after normalising by a physical ``coord_scale``.

Design contract for expression encoders
----------------------------------------
``forward(expr) -> (..., output_dim)`` where ``expr`` is ``(..., n_genes)``.
The ``output_dim`` attribute must be set so the token embedder knows how to
project the result to the transformer width.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Expression encoders                                                          #
# --------------------------------------------------------------------------- #
class ExpressionEncoder(nn.Module):
    """Base class defining the expression-encoder interface.

    Subclasses must set ``self.output_dim`` and implement ``forward``.
    """

    output_dim: int

    def forward(self, expr: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class LinearExpressionEncoder(ExpressionEncoder):
    """The default learnable linear projection ``expr -> hidden``.

    A single linear layer (optionally followed by a non-linearity + norm) keeps
    the gene dimension out of the transformer entirely, as required.  It is
    intentionally simple so it can serve as a drop-in baseline against heavier
    encoders.

    Parameters
    ----------
    n_genes
        Number of input genes.
    output_dim
        Embedding width.
    dropout
        Dropout applied to the embedding.
    activation
        If True, apply GELU + LayerNorm after the projection.
    """

    def __init__(
        self,
        n_genes: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: bool = True,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.output_dim = output_dim
        self.proj = nn.Linear(n_genes, output_dim)
        if activation:
            self.post = nn.Sequential(
                nn.GELU(),
                nn.LayerNorm(output_dim),
                nn.Dropout(dropout),
            )
        else:
            self.post = nn.Dropout(dropout)

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.post(self.proj(expr))


class MLPExpressionEncoder(ExpressionEncoder):
    """A slightly deeper 2-layer MLP expression encoder.

    Provided as an example of how alternative encoders plug in.  Not used by
    default but registered so ``expression_encoder="mlp"`` works out of the box.
    """

    def __init__(
        self,
        n_genes: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim * 2
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
        )

    def forward(self, expr: torch.Tensor) -> torch.Tensor:
        return self.net(expr)


# ---- registry --------------------------------------------------------------- #
# Maps a name -> a builder ``(n_genes, output_dim, dropout) -> ExpressionEncoder``.
# Extend this to add autoencoder / pretrained encoders without editing the model.
ExpressionEncoderBuilder = Callable[..., ExpressionEncoder]

EXPRESSION_ENCODER_REGISTRY: Dict[str, ExpressionEncoderBuilder] = {
    "linear": lambda n_genes, output_dim, dropout: LinearExpressionEncoder(
        n_genes, output_dim, dropout
    ),
    "mlp": lambda n_genes, output_dim, dropout: MLPExpressionEncoder(
        n_genes, output_dim, dropout=dropout
    ),
}


def build_expression_encoder(
    name: str, n_genes: int, output_dim: int, dropout: float
) -> ExpressionEncoder:
    """Instantiate an expression encoder by registry name."""
    if name not in EXPRESSION_ENCODER_REGISTRY:
        raise KeyError(
            f"Unknown expression encoder '{name}'. "
            f"Available: {sorted(EXPRESSION_ENCODER_REGISTRY)}"
        )
    return EXPRESSION_ENCODER_REGISTRY[name](n_genes, output_dim, dropout)


def register_expression_encoder(name: str, builder: ExpressionEncoderBuilder) -> None:
    """Register a new expression-encoder builder (e.g. scGPT wrapper)."""
    EXPRESSION_ENCODER_REGISTRY[name] = builder


# --------------------------------------------------------------------------- #
# Relative-coordinate encoder                                                  #
# --------------------------------------------------------------------------- #
class RelativeCoordEncoder(nn.Module):
    """Encode relative coordinates ``[Δx, Δy, Δz, ‖Δ‖]`` into an embedding.

    The raw physical deltas are divided by a ``coord_scale`` buffer (estimated
    from the data) so the MLP always sees O(1) magnitudes, regardless of whether
    coordinates are in microns, pixels, or arbitrary units.  ``coord_scale`` is
    a registered buffer so it travels with the checkpoint.

    Parameters
    ----------
    output_dim
        Embedding width (transformer model dim).
    hidden_dim
        MLP hidden width.
    dropout
        Dropout on the embedding.
    coord_scale
        Initial normalisation scale (updated via :meth:`set_coord_scale`).
    """

    IN_DIM = 4  # Δx, Δy, Δz, distance

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        coord_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.register_buffer("coord_scale", torch.tensor(float(coord_scale)))
        self.net = nn.Sequential(
            nn.Linear(self.IN_DIM, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def set_coord_scale(self, scale: float) -> None:
        self.coord_scale.fill_(float(max(scale, 1e-8)))

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        """``rel`` : (..., 4) raw [Δx, Δy, Δz, distance]."""
        return self.net(rel / self.coord_scale)
