"""Text-grounded entity embeddings: a frozen text prior, a free residual, and a bridge between them.

The embedding a gene enters the model with is

``e_g = LayerNorm(W t_g + gamma * r_g)``

where ``t_g`` is the frozen text vector, ``W`` is learned, and ``r_g`` is a free
per-gene residual initialised to **zeros** and gated by ``gamma``, annealed 0 -> 1 over
``cfg.residual_gate_warmup_frac`` of training.

That ordering is the design point, not a detail. Early training has no residual to use,
so ``W`` has to learn something real from the text; only later can ``r`` add what the
literature cannot express. Reverse it - a residual that is free from step 0 - and ``r``
absorbs everything, the text channel becomes decorative, and the zero-shot claim is
hollow, because an unseen gene has no ``r`` to fall back on.

Unseen entities get ``r_hat = psi(t)`` from the distillation head, trained against the
*detached* residual so the tail never wags the dog.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from spatialcpav25_gen.config import Config

__all__ = [
    "EntityEmbeddings",
    "TextGroundedEmbedding",
    "coexpression_modules",
    "text_embedding_diagnostics",
]


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Re-initialise a linear layer from an explicit generator.

    ``nn.Linear``'s own ``reset_parameters`` draws from the global torch RNG, which
    Convention 3 forbids relying on; this overwrites it with the same distribution -
    uniform on ``+/- 1/sqrt(fan_in)``, which is what ``kaiming_uniform_(a=sqrt(5))``
    reduces to - drawn from a generator seeded with ``cfg.seed``.
    """
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


class TextGroundedEmbedding(nn.Module):
    """Text prior + free residual, with a distillation head for unseen entities.

    Args:
        text_vecs: (V, 768) frozen MedCPT vectors for the V known entities.
        out_dim:   embedding width.
        cfg:       supplies ``text_dim_in``, ``distill_hidden``,
                   ``residual_gate_warmup_frac``, ``seed`` and ``debug_shapes``.
    Returns from forward: (V_query, out_dim)

    Components
    ----------
    ``W``
        ``Linear(768, out_dim, bias=False)`` - the text channel.
    ``r``
        ``nn.Embedding(V, out_dim)``, zeros-initialised - the free residual.
    ``gamma``
        Buffer, annealed 0 -> 1 by :meth:`set_progress`.
    ``distill``
        ``MLP(768 -> cfg.distill_hidden -> out_dim)``, trained by
        :meth:`distillation_loss` against ``stopgrad(r)``.

    Notes
    -----
    Parameters are initialised from a generator seeded with ``cfg.seed``, so two
    constructions with the same config are bitwise identical (Convention 3).
    """

    text_vecs: Tensor
    gamma: Tensor

    def __init__(self, text_vecs: Tensor, out_dim: int, cfg: Config) -> None:
        super().__init__()
        if text_vecs.ndim != 2:
            raise ValueError(
                f"TextGroundedEmbedding: text_vecs must be (V, 768), got shape "
                f"{tuple(text_vecs.shape)}"
            )
        if text_vecs.shape[1] != cfg.text_dim_in:
            raise ValueError(
                f"TextGroundedEmbedding: text_vecs has width {text_vecs.shape[1]}, expected "
                f"Config.text_dim_in={cfg.text_dim_in}"
            )
        if text_vecs.shape[0] < 1:
            raise ValueError("TextGroundedEmbedding: text_vecs must hold at least one entity")
        if out_dim < 1:
            raise ValueError(f"TextGroundedEmbedding: out_dim must be >= 1, got {out_dim}")

        self.cfg = cfg
        self.out_dim = int(out_dim)
        generator = torch.Generator().manual_seed(cfg.seed)

        # Frozen: a buffer, not a parameter. MedCPT is never fine-tuned (T02 "Do NOT").
        self.register_buffer("text_vecs", text_vecs.detach().to(torch.float32).clone())
        self.register_buffer("gamma", torch.zeros((), dtype=torch.float32))

        self.W = nn.Linear(cfg.text_dim_in, self.out_dim, bias=False)
        _init_linear(self.W, generator)

        self.r = nn.Embedding(text_vecs.shape[0], self.out_dim)
        with torch.no_grad():
            self.r.weight.zero_()

        self.norm = nn.LayerNorm(self.out_dim)

        hidden = nn.Linear(cfg.text_dim_in, cfg.distill_hidden)
        out = nn.Linear(cfg.distill_hidden, self.out_dim)
        _init_linear(hidden, generator)
        _init_linear(out, generator)
        self.distill = nn.Sequential(hidden, nn.GELU(), out)

    @property
    def n_entities(self) -> int:
        """V: the number of known entities."""
        return int(self.text_vecs.shape[0])

    def set_progress(self, frac: float) -> None:
        """Set the residual gate from training progress ``frac`` in ``[0, 1]``.

        ``gamma = min(frac / cfg.residual_gate_warmup_frac, 1)``, i.e. 0 at the start of
        training and 1 from ``residual_gate_warmup_frac`` onwards. A warm-up fraction of
        exactly 0 means "no warm-up": ``gamma`` is 1 throughout.
        """
        if not 0.0 <= frac <= 1.0:
            raise ValueError(f"set_progress: frac must lie in [0, 1], got {frac!r}")
        warmup = self.cfg.residual_gate_warmup_frac
        gamma = 1.0 if warmup <= 0.0 else min(frac / warmup, 1.0)
        with torch.no_grad():
            self.gamma.fill_(gamma)

    def forward(self, idx: Tensor) -> Tensor:
        """Embed known entities by index. ``(V_query,)`` int64 -> ``(V_query, out_dim)``."""
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"TextGroundedEmbedding.forward: idx must be integer, got {idx.dtype}")
        out: Tensor = self.norm(self.W(self.text_vecs[idx]) + self.gamma * self.r(idx))
        if self.cfg.debug_shapes:
            assert out.shape == (*idx.shape, self.out_dim)
        return out

    def forward_zero_shot(self, text_vecs_new: Tensor, use_distill: bool = True) -> Tensor:
        """Embed unseen entities from their text vectors alone.

        ``(V_new, 768)`` -> ``(V_new, out_dim)``. With ``use_distill=False`` the residual
        is exactly zero (the paper's pure-text arm); with ``use_distill=True`` it is
        ``psi(t_new)`` (the distilled arm). Both are reported in the zero-shot table.
        """
        if text_vecs_new.ndim != 2 or text_vecs_new.shape[1] != self.cfg.text_dim_in:
            raise ValueError(
                f"forward_zero_shot: text_vecs_new must be (V_new, {self.cfg.text_dim_in}), "
                f"got shape {tuple(text_vecs_new.shape)}"
            )
        text = text_vecs_new.to(self.text_vecs.dtype)
        projected = self.W(text)
        residual = self.distill(text) if use_distill else torch.zeros_like(projected)
        out: Tensor = self.norm(projected + self.gamma * residual)
        if self.cfg.debug_shapes:
            assert out.shape == (text_vecs_new.shape[0], self.out_dim)
        return out

    def distillation_loss(self) -> Tensor:
        """Return ``mean || psi(t) - stopgrad(r) ||^2`` over the known entities. Scalar.

        The residual is detached: ``psi`` chases ``r``, never the reverse. Letting the
        gradient flow the other way would let the model make the residual easy to predict
        instead of making the prediction good.
        """
        predicted = self.distill(self.text_vecs)
        return torch.mean((predicted - self.r.weight.detach()) ** 2)


class EntityEmbeddings(nn.Module):
    """The three text-grounded embeddings an observation token is assembled from.

    Genes at ``cfg.gene_emb_dim``; cell types and regions at ``cfg.ctx_emb_dim``. Regions
    are optional because a dataset may have none (``Config.region_key=None``).
    """

    def __init__(
        self,
        cfg: Config,
        gene_vecs: Tensor,
        celltype_vecs: Tensor,
        region_vecs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.gene = TextGroundedEmbedding(gene_vecs, cfg.gene_emb_dim, cfg)
        self.celltype = TextGroundedEmbedding(celltype_vecs, cfg.ctx_emb_dim, cfg)
        self.region = (
            None
            if region_vecs is None
            else TextGroundedEmbedding(region_vecs, cfg.ctx_emb_dim, cfg)
        )

    def components(self) -> list[TextGroundedEmbedding]:
        """Return the embeddings present on this dataset, gene first."""
        present = [self.gene, self.celltype]
        if self.region is not None:
            present.append(self.region)
        return present

    def set_progress(self, frac: float) -> None:
        """Anneal the residual gate of every component. The trainer calls this each epoch."""
        for component in self.components():
            component.set_progress(frac)

    def distillation_loss(self) -> Tensor:
        """Return the summed distillation loss over the components. Scalar."""
        losses = [component.distillation_loss() for component in self.components()]
        return torch.stack(losses).sum()


# --------------------------------------------------------------------------------------
# diagnostics (ablation A3)
# --------------------------------------------------------------------------------------


def text_embedding_diagnostics(
    emb: TextGroundedEmbedding,
    expr: npt.NDArray[Any],
    *,
    seed: int,
) -> dict[str, float]:
    """Report whether the text channel carries usable signal (ablation A3's diagnostic).

    Parameters
    ----------
    emb
        The gene embedding, whose ``text_vecs`` supply the text-space geometry.
    expr
        ``(N, G)`` training expression, raw counts, ``G == emb.n_entities``. Used as
        ``log1p`` for the co-expression correlation - an input-side transform, which
        Convention 5 permits (nothing here touches a decoder target).
    seed
        Seeds the pair subsample and the Leiden partition (Convention 3). Keyword-only and
        required: the spec's signature has no seed, but two of the three numbers are
        stochastic without one.

    Returns
    -------
    dict
        ``text_coexpr_spearman``
            Spearman correlation between pairwise cosine similarity in text space and
            pairwise co-expression correlation. Expect something modest (0.1-0.3) on real
            data; ~0 on a fixture whose gene names are arbitrary.
        ``residual_norm_ratio``
            ``||gamma * r|| / ||W t||`` - how much the model leaned on the free residual
            rather than the text.
        ``knn_purity``
            Mean fraction of each gene's ``cfg.text_diag_knn_k`` nearest text neighbours
            that fall in its own co-expression module (Leiden on the gene-gene correlation
            graph).
        plus ``n_genes``, ``n_pairs``, ``n_modules``, ``knn_k`` for provenance.
    """
    from scipy.stats import spearmanr

    cfg = emb.cfg
    counts = np.asarray(expr, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"text_embedding_diagnostics: expr must be (N, G), got {counts.shape}")
    if counts.shape[1] != emb.n_entities:
        raise ValueError(
            f"text_embedding_diagnostics: expr has {counts.shape[1]} genes but the embedding "
            f"has n_entities={emb.n_entities}; they index the same panel"
        )

    # Input-side log1p only (Convention 5): this is a diagnostic over the *input* panel,
    # nothing here is a decoder target.
    log_expr = np.log1p(counts)
    keep = log_expr.std(axis=0) > 0.0
    n_kept = int(keep.sum())
    if n_kept < 2:
        raise ValueError(
            f"text_embedding_diagnostics: only {n_kept} gene(s) vary in this expression "
            "matrix; a co-expression correlation needs at least 2"
        )

    coexpr = np.corrcoef(log_expr[:, keep], rowvar=False)
    text = emb.text_vecs.detach().cpu().numpy().astype(np.float64)[keep]
    text = text / np.linalg.norm(text, axis=1, keepdims=True)
    cosine = text @ text.T

    upper = np.triu_indices(n_kept, k=1)
    cos_pairs = cosine[upper]
    expr_pairs = coexpr[upper]
    n_pairs = int(cos_pairs.size)
    if n_pairs > cfg.text_diag_max_pairs:
        chosen = np.random.default_rng(seed).choice(
            n_pairs, size=cfg.text_diag_max_pairs, replace=False
        )
        cos_pairs = cos_pairs[chosen]
        expr_pairs = expr_pairs[chosen]

    spearman = float(spearmanr(cos_pairs, expr_pairs).statistic)

    modules = coexpression_modules(coexpr, cfg.text_diag_knn_k, cfg, seed=seed)
    purity = _knn_purity(cosine, modules, cfg.text_diag_knn_k)

    with torch.no_grad():
        text_part = emb.W(emb.text_vecs)
        residual_part = emb.gamma * emb.r.weight
        ratio = float(torch.linalg.norm(residual_part) / torch.linalg.norm(text_part))

    return {
        "text_coexpr_spearman": spearman,
        "residual_norm_ratio": ratio,
        "knn_purity": purity,
        "n_genes": float(n_kept),
        "n_pairs": float(min(n_pairs, cfg.text_diag_max_pairs)),
        "n_modules": float(len(set(modules.tolist()))),
        "knn_k": float(cfg.text_diag_knn_k),
    }


def coexpression_modules(
    coexpr: npt.NDArray[np.float64], k: int, cfg: Config, *, seed: int
) -> npt.NDArray[np.int64]:
    """Leiden modules of the gene-gene co-expression graph. ``(G, G)`` -> ``(G,)`` labels.

    Public because T07's ``L_prog`` needs the same partition: a "molecular program" has to
    mean one thing in this codebase, and the text diagnostics' modules and the consistency
    loss's modules are the same object computed the same way.

    The graph is the mutual-or-not kNN graph on correlation, edges weighted by the
    correlation clipped at 0 (a negative correlation is not evidence of membership in the
    same module, and Leiden's objective needs non-negative weights).
    """
    import igraph
    import leidenalg

    n_genes = int(coexpr.shape[0])
    neighbours = _knn_indices(coexpr, min(k, n_genes - 1))
    edges: dict[tuple[int, int], float] = {}
    for i in range(n_genes):
        for j in neighbours[i]:
            key = (min(i, int(j)), max(i, int(j)))
            edges[key] = max(edges.get(key, 0.0), float(max(coexpr[i, j], 0.0)))

    graph = igraph.Graph(n=n_genes, edges=list(edges), directed=False)
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=list(edges.values()),
        resolution_parameter=cfg.text_diag_leiden_resolution,
        n_iterations=cfg.text_diag_leiden_iterations,
        seed=seed,
    )
    return np.asarray(partition.membership, dtype=np.int64)


def _knn_indices(similarity: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.int64]:
    """``(G, G)`` similarity -> ``(G, k)`` indices of the k most similar others.

    Ties break by index, so the result is deterministic.
    """
    masked = similarity.copy()
    np.fill_diagonal(masked, -np.inf)
    order = np.argsort(-masked, axis=1, kind="stable")
    return np.asarray(order[:, :k], dtype=np.int64)


def _knn_purity(cosine: npt.NDArray[np.float64], modules: npt.NDArray[np.int64], k: int) -> float:
    """Mean fraction of each gene's k nearest text neighbours sharing its module."""
    k_eff = min(k, cosine.shape[0] - 1)
    neighbours = _knn_indices(cosine, k_eff)
    same = modules[neighbours] == modules[:, None]
    return float(same.mean())
