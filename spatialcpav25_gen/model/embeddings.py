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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from spatialcpav25_gen.config import Config

__all__ = [
    "EntityEmbeddings",
    "TextGroundedEmbedding",
    "build_entity_embeddings",
    "coexpression_modules",
    "describe_entity_descriptors",
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
        if self.cfg.text_emb_mode == "lookup":
            # No text prior to lead the way, so nothing to anneal *from*: under lookup-only
            # the residual is the whole embedding, and a warm-up would leave it exactly zero
            # for the first `residual_gate_warmup_frac` of training.
            gamma = 1.0
        with torch.no_grad():
            self.gamma.fill_(gamma)

    def _text_channel(self, vectors: Tensor) -> Tensor:
        """``W t``, or zeros under ``Config.text_emb_mode = "lookup"``. ``(V, 768) -> (V, D)``.

        The gate T01 declared and nothing consumed until now. ``"lookup"`` is the design
        document's *lookup-only* arm: the embedding is a plain learned table with no text
        prior, which is T10's ablation **A3** ("the text channel's value on seen genes") and
        one of the four gates T09's ``select_config`` chooses between. Implemented here rather
        than by passing different vectors in, because the *zero-shot* path has to lose the
        same channel — an entity with no row in the table has no embedding at all without it,
        which is precisely what A3 is asking about.
        """
        projected: Tensor = self.W(vectors)
        if self.cfg.text_emb_mode == "lookup":
            return torch.zeros_like(projected)
        return projected

    def forward(self, idx: Tensor) -> Tensor:
        """Embed known entities by index. ``(V_query,)`` int64 -> ``(V_query, out_dim)``."""
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"TextGroundedEmbedding.forward: idx must be integer, got {idx.dtype}")
        out: Tensor = self.norm(self._text_channel(self.text_vecs[idx]) + self.gamma * self.r(idx))
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
        projected = self._text_channel(text)
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
# the panel's embeddings: descriptors -> MedCPT -> EntityEmbeddings
# --------------------------------------------------------------------------------------


def build_entity_embeddings(
    cfg: Config,
    gene_names: Sequence[str],
    celltype_names: Sequence[str],
    region_names: Sequence[str] | None = None,
    *,
    gene_meta: Mapping[str, Any] | None = None,
    encoder: Any | None = None,
) -> EntityEmbeddings:
    """Build this panel's :class:`EntityEmbeddings` from T02's descriptors and MedCPT.

    The step every caller had been writing for itself, and writing differently: five
    hand-rolled versions existed across ``scripts/`` and the bench3 wrapper, and **three of
    them passed no gene metadata at all**, so ``medcpt`` encoded a bare symbol
    (``"Slc17a7."``) where the panel's table carries a full name and an NCBI summary. That
    is not the open-vocabulary channel the paper claims; it is MedCPT applied to a token.
    One builder, in the package, so the descriptors a run encodes are a property of the
    config rather than of whichever script invoked it.

    Parameters
    ----------
    cfg
        Supplies ``gene_meta_path``, ``mygene_species``, ``text_model``, ``text_cache_dir``
        and ``text_emb_mode``.
    gene_names, celltype_names, region_names
        The panel's entities, in the volume's own order. ``region_names`` may be ``None``
        (or empty) for a dataset with no regions, which is what ``Config.region_key=None``
        means.
    gene_meta
        Pre-loaded ``symbol -> GeneMeta``. Defaults to
        ``load_gene_meta(cfg.gene_meta_path, species=cfg.mygene_species)``, which **raises**
        when the table is absent or is another organism's.
    encoder
        Pre-built :class:`~spatialcpav25_gen.data.text.TextEncoder`. Passing one lets several
        panels share a cache handle; the default constructs one, and on a warm
        ``cfg.text_cache_dir`` it never loads the transformer.

    Returns
    -------
    EntityEmbeddings
        Genes at ``cfg.gene_emb_dim``, cell types and regions at ``cfg.ctx_emb_dim``.

    Notes
    -----
    **Both arms of the ``text_emb_mode`` gate get the same vectors.** ``"lookup"`` is applied
    *inside* :meth:`TextGroundedEmbedding._text_channel`, which zeroes the projection on the
    seen and the zero-shot path alike; withholding the vectors here as well would make the
    two arms differ in two things at once, and the A3 comparison is only a comparison while
    they differ in one. It also keeps ``distillation_loss`` — which reads ``text_vecs``
    directly — measuring the same quantity in both arms.

    A gene the table does not know degrades to ``gene_descriptor(symbol, None)``, i.e. the
    bare symbol, and the count of such genes is returned in ``uns``-style form by
    :func:`describe_entity_descriptors` rather than being swallowed: a panel that is mostly
    bare symbols is a table that needs rebuilding, not a run that should proceed quietly.
    """
    from spatialcpav25_gen.data.text import (
        TextEncoder,
        celltype_descriptor,
        gene_descriptor,
        load_gene_meta,
        region_descriptor,
    )

    if not gene_names:
        raise ValueError("build_entity_embeddings: gene_names is empty")
    if not celltype_names:
        raise ValueError("build_entity_embeddings: celltype_names is empty")

    meta = (
        load_gene_meta(cfg.gene_meta_path, species=cfg.mygene_species)
        if gene_meta is None
        else gene_meta
    )
    text = TextEncoder(cfg) if encoder is None else encoder

    genes = text.encode([gene_descriptor(str(g), meta.get(str(g))) for g in gene_names])
    types = text.encode([celltype_descriptor(str(t), None) for t in celltype_names])
    regions = (
        text.encode([region_descriptor(str(r), None) for r in region_names])
        if region_names
        else None
    )
    return EntityEmbeddings(
        cfg,
        torch.from_numpy(genes),
        torch.from_numpy(types),
        None if regions is None else torch.from_numpy(regions),
    )


def describe_entity_descriptors(
    cfg: Config, gene_names: Sequence[str], gene_meta: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Say what the text channel is actually being given, for the run's provenance record.

    ``{n_genes, n_with_meta, n_bare, n_with_summary, example}``. A run that reports
    ``text_emb_mode=medcpt`` while ``n_bare == n_genes`` has the gate switched on and nothing
    behind it, and that is exactly the state three of the existing callers were in.
    """
    from spatialcpav25_gen.data.text import gene_descriptor, load_gene_meta

    meta = (
        load_gene_meta(cfg.gene_meta_path, species=cfg.mygene_species)
        if gene_meta is None
        else gene_meta
    )
    symbols = [str(g) for g in gene_names]
    known = [s for s in symbols if s in meta]
    with_summary = [s for s in known if getattr(meta[s], "summary", None)]
    return {
        "n_genes": len(symbols),
        "n_with_meta": len(known),
        "n_bare": len(symbols) - len(known),
        "n_with_summary": len(with_summary),
        "example": gene_descriptor(symbols[0], meta.get(symbols[0]))[:160] if symbols else "",
    }


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
