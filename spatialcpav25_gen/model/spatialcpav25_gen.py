"""The whole model as one object, and the trainer that fits it.

:class:`CTFFlow` is field + retrieval + layout + expression. It exists so that the
conditioning vector is assembled in exactly one place — the coverage matrix's "observation
token ... assembled in ``spatialcpav25_gen.py``" — and so that the three consumers of
geometry (the anatomical field, the retrieval index, the GRF) can be driven by a single
rotation context per step.

``forward_train`` returns **unweighted, named** loss terms and the trainer applies the
weights. That is a requirement rather than a style choice: T07 and T08 add terms to the same
dict, T07's ramp has to compare a consistency term against the reconstruction term at
comparable scale, and diagnosing a collapse means reading each term's own trajectory. Keys
beginning ``diag_`` are diagnostics, not losses, and :func:`train_ctfflow` never weights
them — the attention entropy T06 carries over from GATE 2, and the per-gene variance T07's
collapse alarm will watch, both travel that way.

Frames, in one place
--------------------
Two coordinate frames coexist under rotation augmentation and every module wants a specific
one. The rule, enforced by :class:`~spatialcpav25_gen.model.field.RotationContext`:

* :class:`~spatialcpav25_gen.model.field.TriplaneField` takes **model-frame** points (it is
  the pose the feature planes are looked up in);
* the anisotropic Fourier encoding, and :class:`~spatialcpav25_gen.model.layout.IntensityHead`
  which contains one, take **data-frame** points — ``fourier_bands_z`` is small because the
  z axis is the true sectioning axis, and that argument is false in a rotated frame;
* the retrieval index and the GRF take **data-frame** points, because real cells sit at real
  positions and the noise realisation is a property of the tissue, not of its pose.

A training step therefore has no *plane* channel to rotate: every plane is consumed by
drawing points on it, and those points go through the ``coords`` channel. That omission is
declared in ``RotationContext(requires=...)`` rather than left implicit, which is what GATE
2's G2.1h exists to force.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy import sparse
from torch import Tensor, nn

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import (
    Section,
    TrainingVolume,
    Volume,
    to_xyz,
)
from spatialcpav25_gen.infer.planes import Plane, section_plane, uniform_slab_points
from spatialcpav25_gen.losses.reconstruction import (
    layout_poisson_nll,
    size_factor_loss,
    zigamma_nll,
    zinb_nll,
)
from spatialcpav25_gen.model.embeddings import EntityEmbeddings
from spatialcpav25_gen.model.expression import (
    ExpressionEncoder,
    ExpressionError,
    LatentFlow,
    SizeFactorHead,
    ZIGammaDecoder,
    ZINBDecoder,
    assert_detection_rate,
    build_decoder,
    cross_mix_counts,
    sample_counts,
    sample_zigamma,
)
from spatialcpav25_gen.model.field import (
    RotationContext,
    TriplaneField,
    fourier_encode,
    fourier_encode_dim,
)
from spatialcpav25_gen.model.layout import (
    FlankingSection,
    IntensityHead,
    RepulsionParams,
    flanking_from_section,
    fourier_bands_for_lengthscale,
    intensity_fn_from_head,
    mean_cell_density,
    sample_layout,
)
from spatialcpav25_gen.model.noise import GaussianRandomField
from spatialcpav25_gen.model.retrieval import (
    RetrievalAttention,
    RetrievalIndex,
    attention_entropy,
)

if TYPE_CHECKING:  # pragma: no cover - see the deferred import in `train_ctfflow`
    from spatialcpav25_gen.losses.sefl import EMATeacher

__all__ = [
    "EMA",
    "Batch",
    "CTFFlow",
    "LayoutTargets",
    "TrainHistory",
    "TrainingData",
    "VolumeStats",
    "flanking_sections",
    "loss_weights",
    "train_ctfflow",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

LOSS_TERMS: tuple[str, ...] = (
    "recon",
    "cfm",
    "size",
    "layout",
    "distill",
    "tv_z",
    "cross",
    "thick",
    "prog",
)
"""The named loss terms a training step can carry, in the order the trainer sums them.

``forward_train`` returns the first six; T07's ``cross``, ``thick`` and ``prog`` are added by
the trainer's SEFL block, because they are properties of *two virtual sections* rather than of
the batch. T08 adds ``autocorr``, ``profile`` and ``distribution``.
"""

DIAG_PREFIX = "diag_"
"""Keys with this prefix are diagnostics and are never weighted into the total."""


# --------------------------------------------------------------------------------------
# what a batch is
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutTargets:
    """One section's layout likelihood inputs, for the Poisson NLL of T05.

    Attributes
    ----------
    cells_xyz
        ``(Nc, 3)`` float64 data-frame positions of **every** cell of the section, with the
        depth jittered uniformly within the slab. All of them, not a subsample: the process
        likelihood balances a sum over cells against an integral over the slab, and a
        subsample would silently reweight the two.
    cell_type
        ``(Nc,)`` int64 type codes.
    mc_xyz
        ``(M, 3)`` float64 points drawn uniformly in the slab volume, redrawn every step.
    slab_volume
        The slab's volume in micrometre^3.
    section_id
        Which section this is, for logging.
    """

    cells_xyz: FloatArray
    cell_type: Tensor
    mc_xyz: FloatArray
    slab_volume: float
    section_id: str


@dataclass(frozen=True)
class Batch:
    """One training step's cells, genes and layout section.

    Attributes
    ----------
    rows
        ``(N,)`` row indices into the :class:`RetrievalIndex` — which training cells these
        are.
    xyz
        ``(N, 3)`` float64 **data-frame** positions.
    cell_type, region
        ``(N,)`` int64 codes; ``region`` is ``None`` on a dataset without regions.
    counts
        ``(N, G')`` float32 **raw** counts of the subsampled genes (Convention 5).
    gene_idx
        ``(G',)`` columns of the volume's gene axis this step drew — ``Config.genes_per_step``
        of them, so the decoder never sees a fixed panel width.
    total_counts
        ``(N,)`` float32 per-cell total over the **whole** panel, not just ``gene_idx``: the
        library size is a property of the cell and must not move when the gene subsample does.
    size_factor
        ``(N,)`` float32 ``total_counts / median total``, the dimensionless quantity the
        encoder normalises by and the decoder scales ``mu`` with.
    neighbours, source_section
        ``(N, K)`` retrieval result and ``(N,)`` owning-section index.
    layout
        The section whose layout likelihood this step evaluates, or ``None`` when the layout
        term is switched off.
    """

    rows: IntArray
    xyz: FloatArray
    cell_type: Tensor
    region: Tensor | None
    counts: Tensor
    gene_idx: IntArray
    total_counts: Tensor
    size_factor: Tensor
    neighbours: IntArray
    source_section: IntArray
    layout: LayoutTargets | None

    @property
    def n_cells(self) -> int:
        """N: cells in this batch."""
        return int(self.xyz.shape[0])

    @property
    def n_genes(self) -> int:
        """G': genes in this batch."""
        return int(self.gene_idx.shape[0])


@dataclass(frozen=True)
class VolumeStats:
    """The handful of dataset scalars the heads are initialised and guarded against.

    Every one of them is a property of the *training* volume, which is why they are computed
    once here rather than hidden in a ``Config`` default: a constant that depends on the data
    is not a constant (the same argument as ``IntensityHead(init_density=...)`` in T05).
    """

    median_total: float
    """Median per-cell total counts. The reference the size factor is expressed against."""
    mean_expression: float
    """Mean count per (cell, gene). Initialises the decoder's ``mu`` head."""
    mean_density: float
    """Cells per micrometre^3. Initialises the intensity head."""
    median_detection: float
    """Median per-gene detection rate. What the generation path's assertion compares to."""
    n_types: int
    n_regions: int | None
    n_genes: int


@dataclass(frozen=True)
class TrainingData:
    """Pooled training cells in the shape the batch sampler needs.

    Held separately from :class:`CTFFlow` so that the model carries parameters and the data
    carries data: T08's LOSO folds build a second :class:`TrainingData` over a subset of
    sections without rebuilding the model.
    """

    vol: TrainingVolume
    index: RetrievalIndex
    counts: Any
    """``(N, G)`` CSR of every training cell's raw counts, sections in stack order."""
    total_counts: FloatArray
    """``(N,)`` per-cell total over the whole panel."""
    stats: VolumeStats

    @property
    def n_cells(self) -> int:
        """N: pooled training cells."""
        return int(self.counts.shape[0])

    @classmethod
    def build(
        cls, vol: TrainingVolume, cfg: Config, *, index: RetrievalIndex | None = None
    ) -> TrainingData:
        """Pool ``vol``'s sections and measure the statistics the heads need."""
        if not isinstance(vol, TrainingVolume):
            raise TypeError(
                f"TrainingData.build expects a TrainingVolume, got {type(vol).__name__}. "
                "Held-out sections are never training data (CLAUDE.md, leakage discipline)."
            )
        counts = sparse.vstack([s.counts for s in vol.sections]).tocsr().astype(np.float64)
        totals = np.asarray(counts.sum(axis=1), dtype=np.float64).reshape(-1)
        if not np.all(totals > 0):
            n_empty = int((totals <= 0).sum())
            raise ExpressionError(
                f"TrainingData.build: {n_empty} training cell(s) have zero total counts, so "
                "their size factor is undefined. Filter them upstream (the loader's "
                f"Config.counts_layer={cfg.counts_layer!r} matrix is what they came from)."
            )
        # From the CSR directly: densifying a 20k-gene panel to count non-zeros would be the
        # one place in the pipeline that needs the whole matrix in memory at once.
        detection = np.asarray((counts > 0).sum(axis=0), dtype=np.float64).reshape(-1) / float(
            counts.shape[0]
        )
        stats = VolumeStats(
            median_total=float(np.median(totals)),
            mean_expression=float(counts.mean()),
            mean_density=mean_cell_density(vol.sections),
            median_detection=float(np.median(detection)),
            n_types=len(vol.celltype_names),
            n_regions=None if vol.region_names is None else len(vol.region_names),
            n_genes=vol.n_genes,
        )
        return cls(
            vol=vol,
            index=RetrievalIndex(vol, cfg) if index is None else index,
            counts=counts,
            total_counts=totals,
            stats=stats,
        )

    def sample_batch(
        self,
        cfg: Config,
        *,
        seed: int,
        step: int = 0,
        gene_pool: IntArray | None = None,
        with_layout: bool = True,
    ) -> Batch:
        """Draw one batch. Deterministic given ``seed`` and ``step`` (Convention 3).

        ``Config.batch_cells`` cells without replacement, ``Config.genes_per_step`` genes
        without replacement from ``gene_pool`` (the whole panel when it is ``None``; a subset
        is what the zero-shot experiment holds genes out with), and — when ``with_layout`` —
        one section's layout targets, cycling through the sections by ``step`` so that
        consecutive steps do not evaluate the same slab.
        """
        gen = np.random.default_rng([seed, step])
        n_cells = min(int(cfg.batch_cells), self.n_cells)
        rows = np.sort(gen.choice(self.n_cells, size=n_cells, replace=False)).astype(np.intp)
        pool = (
            np.arange(self.stats.n_genes, dtype=np.intp)
            if gene_pool is None
            else np.asarray(gene_pool, dtype=np.intp)
        )
        if pool.size == 0:
            raise ExpressionError("sample_batch: the gene pool is empty")
        n_genes = min(int(cfg.genes_per_step), int(pool.size))
        gene_idx = np.sort(gen.choice(pool, size=n_genes, replace=False)).astype(np.intp)

        block = np.asarray(self.counts[rows][:, gene_idx].todense(), dtype=np.float32)
        totals = self.total_counts[rows].astype(np.float32)
        xyz = self.index.coords[rows]
        neighbours, _ = self.index.query(
            xyz,
            set(),
            seed=int(gen.integers(0, 2**31 - 1)),
            source_section=self.index.section_index[rows],
            apply_dropout=cfg.section_dropout_p > 0.0,
        )
        region = (
            None
            if self.index.region is None
            else torch.from_numpy(self.index.region[rows].astype(np.int64))
        )
        layout = (
            self._layout_targets(cfg, gen, step=step)
            if with_layout and cfg.w_layout > 0.0
            else None
        )
        return Batch(
            rows=rows,
            xyz=xyz,
            cell_type=torch.from_numpy(self.index.cell_type[rows].astype(np.int64)),
            region=region,
            counts=torch.from_numpy(block),
            gene_idx=gene_idx,
            total_counts=torch.from_numpy(totals),
            size_factor=torch.from_numpy(totals / float(self.stats.median_total)),
            neighbours=neighbours,
            source_section=self.index.section_index[rows],
            layout=layout,
        )

    def _layout_targets(self, cfg: Config, gen: np.random.Generator, *, step: int) -> LayoutTargets:
        """Build one section's layout likelihood inputs, jitter and MC points redrawn."""
        section = self.vol.sections[step % self.vol.n_sections]
        plane = section_plane(section)
        xyz = to_xyz(section).astype(np.float64)
        half = 0.5 * float(section.thickness)
        xyz[:, 2] += gen.uniform(-half, half, size=xyz.shape[0])
        return LayoutTargets(
            cells_xyz=xyz,
            cell_type=torch.from_numpy(np.asarray(section.cell_type, dtype=np.int64)),
            mc_xyz=uniform_slab_points(plane, int(cfg.layout_n_mc), gen),
            slab_volume=plane.slab_volume,
            section_id=section.section_id,
        )


# --------------------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------------------


class CTFFlow(nn.Module):
    """Field + retrieval + layout + expression, one object.

    Args:
        cfg:  the frozen config. ``ell_xy`` / ``ell_z`` are read here, so a calibrated
              length-scale reaches the GRF and the intensity head's basis through the config
              rather than through a second channel.
        data: the pooled :class:`TrainingData`. A ``TrainingVolume`` by type, so held-out
              sections cannot become conditioning evidence or a training target.
        embeddings: T02's :class:`EntityEmbeddings` — genes, cell types and (optionally)
              regions. Passed in rather than built here because the text vectors come from a
              cache the model has no business knowing about, and because the zero-shot
              experiment swaps the gene embedding's *query* without rebuilding the model.
        grf_seed: the noise realisation (T03). One realisation per specimen; T09's
              ``generate_stack`` reuses it so a whole stack is mutually coherent.
        repulsion: the fitted Strauss/hard-core interaction. Required by ``generate`` when
              ``Config.repulsion`` is set — T05 forbids a hand-set ``r0``.

    The conditioning vector, once
    -----------------------------
    ``cond = [F(p), fourier(p), type_emb, region_emb, ctx_retrieval, z_embed]`` (T06 §2).
    :meth:`conditioning` is the only place it is built. The retrieval attention's *query* is
    everything but ``ctx_retrieval`` — i.e. the field feature **and** the type, region and
    depth embeddings — which is what ``RetrievalAttention``'s own docstring anticipates
    ("``[F(p), fourier(p), type_emb, ...]`` assembled by the caller") and is a material
    difference from GATE 2's probe, whose query was the field feature alone. A query that
    does not know its own cell type cannot prefer a donor that shares it, which is one of the
    two reasons T04's attention could only average (open risk R2).
    """

    def __init__(
        self,
        cfg: Config,
        data: TrainingData,
        embeddings: EntityEmbeddings,
        *,
        grf_seed: int = 0,
        repulsion: RepulsionParams | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.data = data
        self.stats = data.stats
        self.repulsion = repulsion
        self.embeddings = embeddings
        if embeddings.gene.out_dim != int(cfg.gene_emb_dim):
            raise ExpressionError(
                f"CTFFlow: the gene embedding is {embeddings.gene.out_dim} wide but "
                f"Config.gene_emb_dim={cfg.gene_emb_dim}"
            )
        if (embeddings.region is None) != (self.stats.n_regions is None):
            raise ExpressionError(
                "CTFFlow: the volume has "
                f"{'no' if self.stats.n_regions is None else self.stats.n_regions} regions but "
                f"the region embedding is {'absent' if embeddings.region is None else 'present'}"
            )

        self.field = TriplaneField(cfg, data.vol.bbox)
        self.grf = GaussianRandomField(cfg, (cfg.ell_xy, cfg.ell_xy, cfg.ell_z), grf_seed)
        self.attention = RetrievalAttention(
            cfg, q_dim=self.query_dim, token_dim=data.index.token_dim
        )

        generator = torch.Generator().manual_seed(int(cfg.seed))
        self.z_embed = nn.Linear(3, int(cfg.z_embed_dim))
        bound = 1.0 / math.sqrt(3)
        with torch.no_grad():
            self.z_embed.weight.uniform_(-bound, bound, generator=generator)
            self.z_embed.bias.zero_()

        # The B10 answer: the intensity head's in-plane basis is tied to the fitted
        # length-scale rather than left at the default that overfits the point pattern.
        extent = float(np.min(data.vol.bbox[1, :2] - data.vol.bbox[0, :2]))
        self.intensity_bands = fourier_bands_for_lengthscale(extent, float(cfg.ell_xy), cfg)
        self.intensity_cfg = cfg.replace(fourier_bands_xy=self.intensity_bands)
        self.intensity = IntensityHead(
            self.intensity_cfg,
            n_types=self.stats.n_types,
            bbox=data.vol.bbox,
            init_density=self.stats.mean_density,
            n_regions=self.stats.n_regions,
        )

        self.encoder = ExpressionEncoder(cfg)
        self.flow = LatentFlow(cfg, self.cond_dim)
        self.decoder: ZINBDecoder | ZIGammaDecoder = build_decoder(
            cfg, int(cfg.gene_emb_dim), init_mean_expression=self.stats.mean_expression
        )
        self.size_head = SizeFactorHead(cfg, reference_total=self.stats.median_total)

    # ----------------------------------------------------------------------------------
    # widths
    # ----------------------------------------------------------------------------------

    @property
    def query_dim(self) -> int:
        """Width of the retrieval attention's query: ``cond`` minus ``ctx_retrieval``."""
        width = int(self.cfg.field_dim) + fourier_encode_dim(self.cfg)
        width += int(self.cfg.ctx_emb_dim)
        if self.stats.n_regions is not None:
            width += int(self.cfg.ctx_emb_dim)
        return width + int(self.cfg.z_embed_dim)

    @property
    def cond_dim(self) -> int:
        """Width of the conditioning vector the flow is conditioned on."""
        return self.query_dim + int(self.cfg.retrieval_ctx_dim)

    # ----------------------------------------------------------------------------------
    # conditioning
    # ----------------------------------------------------------------------------------

    def _normalised(self, xyz_data: Tensor) -> Tensor:
        """``(N, 3)`` data-frame micrometres -> ``(N, 3)`` in ``[-1, 1]`` by the bbox."""
        box = torch.as_tensor(self.data.vol.bbox, dtype=torch.float32)
        span = torch.clamp(box[1] - box[0], min=1e-6)
        return 2.0 * (xyz_data - box[0][None, :]) / span[None, :] - 1.0

    def _z_embedding(self, xyz_data: Tensor) -> Tensor:
        """``(N, 3)`` -> ``(N, z_embed_dim)``: depth in units of the median section spacing.

        ``[z / spacing, sin(pi z / spacing), cos(pi z / spacing)]`` through one linear layer.
        The periodic pair is what distinguishes "on a section" from "half a spacing away from
        one", which is the quantity generation at an arbitrary plane needs and which
        ``fourier(p)`` — normalised by the bounding box — cannot express.
        """
        spacing = max(float(self.data.vol.median_spacing), 1e-6)
        u = xyz_data[:, 2:3] / spacing
        features = torch.cat([u, torch.sin(math.pi * u), torch.cos(math.pi * u)], dim=1)
        return cast(Tensor, self.z_embed(features))

    def query_features(
        self,
        xyz_model: Tensor,
        xyz_data: Tensor,
        cell_type: Tensor,
        region: Tensor | None,
    ) -> Tensor:
        """``[F(p), fourier(p), type_emb, region_emb, z_embed]``. ``(N, query_dim)`` float32.

        ``xyz_model`` goes to the triplane field, ``xyz_data`` to the Fourier encoding and the
        depth embedding — see the module docstring on frames. The two are the same array when
        no rotation is applied.
        """
        parts = [
            self.field(xyz_model),
            fourier_encode(self._normalised(xyz_data), self.cfg),
            self.embeddings.celltype(cell_type),
        ]
        if self.embeddings.region is not None:
            if region is None:
                raise ExpressionError(
                    "CTFFlow.query_features: the volume has regions but none were passed. A "
                    "missing region is an error, not a zero vector (Convention 6)."
                )
            parts.append(self.embeddings.region(region))
        parts.append(self._z_embedding(xyz_data))
        out = torch.cat(parts, dim=1)
        if self.cfg.debug_shapes:
            assert out.shape == (xyz_model.shape[0], self.query_dim), out.shape
        return out

    def conditioning(
        self,
        xyz_model: Tensor,
        xyz_data: Tensor,
        cell_type: Tensor,
        region: Tensor | None,
        neighbour_tokens: Tensor,
        neighbour_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Assemble ``cond`` and return it with the attention weights.

        ``(N, cond_dim)`` and ``(N, n_heads, K)``. The weights are returned rather than hidden
        because their entropy is a first-class diagnostic (GATE 2's G2.4, carried into T06).
        """
        q_feat = self.query_features(xyz_model, xyz_data, cell_type, region)
        ctx, weights = self.attention.attend(q_feat, neighbour_tokens, neighbour_mask)
        cond = torch.cat([q_feat, ctx], dim=1)
        if self.cfg.debug_shapes:
            assert cond.shape == (xyz_model.shape[0], self.cond_dim), cond.shape
        return cond, weights

    def prior_latent(self, xyz_data: npt.NDArray[Any], *, seed: int) -> Tensor:
        """``h0``, the source of the flow. ``(N, 3)`` -> ``(N, d_h)`` float32.

        The GRF of T03 queried at the cells' **physical** positions under
        ``Config.prior_mode = "correlated"``, and an i.i.d. Gaussian under ``"iid"``, which is
        ablation A1. The i.i.d. arm needs a seed and the correlated one does not — the field
        is a pure function of position — so the argument is required either way rather than
        being present only in the branch that uses it.
        """
        if self.cfg.prior_mode == "correlated":
            return self.grf(np.asarray(xyz_data, dtype=np.float32))
        gen = torch.Generator().manual_seed(int(seed))
        return torch.randn(
            (int(np.asarray(xyz_data).shape[0]), int(self.cfg.latent_dim)),
            generator=gen,
            dtype=torch.float32,
        )

    # ----------------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------------

    def forward_train(
        self, batch: Batch, *, rotation: RotationContext | None = None
    ) -> dict[str, Tensor]:
        """One step's **unweighted** loss terms, by name. See :data:`LOSS_TERMS`.

        Parameters
        ----------
        batch
            The step's cells, genes and layout section.
        rotation
            The step's rotation context, already entered, or ``None`` for the un-rotated
            pass. Additive keyword: the spec's ``forward_train(batch)`` has nowhere to put the
            augmentation, and threading it through the batch would let a caller rotate the
            coordinates and forget the GRF — the exact mistake ``RotationContext`` exists to
            make impossible.

        Returns
        -------
        dict
            ``recon`` (the decoder's NLL against raw counts), ``cfm``, ``size``, ``layout``,
            ``distill``, ``tv_z``, plus ``diag_attention_entropy`` (nats) and
            ``diag_gene_variance``. Terms the config has switched off are absent, not zero, so
            a missing weight is a ``KeyError`` rather than a silent no-op.
        """
        xyz_data = batch.xyz
        xyz_model = xyz_data if rotation is None else rotation.coords(xyz_data)
        xyz_retrieval = xyz_data if rotation is None else rotation.retrieval(xyz_model)
        xyz_grf = xyz_data if rotation is None else rotation.grf(xyz_model)

        tokens, mask = self.data.index.neighbour_tokens(xyz_retrieval, batch.neighbours)
        cond, weights = self.conditioning(
            torch.from_numpy(np.asarray(xyz_model, dtype=np.float32)),
            torch.from_numpy(np.asarray(xyz_data, dtype=np.float32)),
            batch.cell_type,
            batch.region,
            tokens,
            mask,
        )

        gene_emb = self.embeddings.gene(torch.from_numpy(batch.gene_idx.astype(np.int64)))
        h1 = self.encoder(batch.counts, gene_emb, batch.size_factor)
        h0 = self.prior_latent(xyz_grf, seed=int(batch.rows[0]))

        mu, theta, pi = self.decoder(h1, gene_emb, batch.size_factor)
        nll = zigamma_nll if self.cfg.decoder == "zigamma" else zinb_nll
        terms: dict[str, Tensor] = {
            "recon": nll(batch.counts, mu, theta, pi, self.cfg),
            "cfm": self.flow.cfm_loss(h1, h0, cond, self._flow_generator(batch)),
            "size": size_factor_loss(self.size_head(h1), batch.total_counts, self.cfg),
        }
        if batch.layout is not None:
            terms["layout"] = self._layout_term(batch.layout, rotation)
        if self.cfg.w_distill > 0.0:
            terms["distill"] = self.embeddings.distillation_loss()
        if self.cfg.tv_z_weight > 0.0:
            terms["tv_z"] = self.field.tv_z_penalty()

        with torch.no_grad():
            terms[f"{DIAG_PREFIX}attention_entropy"] = attention_entropy(weights).mean()
            terms[f"{DIAG_PREFIX}gene_variance"] = mu.var(dim=0).mean()
            drawn = self._generated_counts(h0, cond, gene_emb, batch)
            terms[f"{DIAG_PREFIX}gene_variance_gen"] = drawn.var(dim=0).mean()
            terms[f"{DIAG_PREFIX}gene_variance_real"] = batch.counts.var(dim=0).mean()
        return terms

    def _generated_counts(self, h0: Tensor, cond: Tensor, gene_emb: Tensor, batch: Batch) -> Tensor:
        """Return a draw down the **generation** path, on the batch's own cells. ``(N, G')``.

        Prior latent -> flow -> decoder -> a count draw: :meth:`generate` minus the layout, at
        real positions. T07's collapse alarm watches this and not the reconstruction path's
        ``mu``, and the difference is not cosmetic — it is the whole value of the alarm:

        * the reconstruction path decodes ``h1``, which the **encoder** computes from the
          cell's real counts, and the encoder never queries the anatomical field;
        * the generation path decodes ``h`` integrated from the GRF prior under ``cond``, and
          ``cond`` is where the field enters.

        A consistency loss that flattens the field therefore leaves the reconstruction path's
        variance looking healthy while the generated section becomes uniform. Measured at T07
        on a deliberately over-driven arm: reconstruction-path variance ratio **0.36-0.89**
        (healthy) at the same checkpoint whose *generated* section reads **0.054** (dead). The
        alarm ``specs/07`` §2 asks for is only an alarm if it watches this path.

        Both sides are drawn counts: ``Var[x] = Var[mu] + E[NB noise]``, so a variance of
        ``mu`` against a variance of counts would be a different quantity. Seeded from the
        batch (Convention 3).
        """
        h = self.flow.sample(h0, cond, int(self.cfg.ode_steps))
        mu, theta, pi = self.decoder(h, gene_emb, self.size_head.size_factor(h))
        gen = np.random.default_rng([int(batch.rows[0]), batch.n_genes])
        draw = sample_zigamma if self.cfg.decoder == "zigamma" else sample_counts
        return draw(mu, theta, pi, gen)

    def _flow_generator(self, batch: Batch) -> torch.Generator:
        """Return the CFM draw's generator, derived from the batch: the step is reproducible."""
        return torch.Generator().manual_seed(int(batch.rows[0]) * 1_000_003 + batch.n_genes)

    def _layout_term(self, targets: LayoutTargets, rotation: RotationContext | None) -> Tensor:
        """Return the section's inhomogeneous-Poisson NLL, per cell. Scalar ``()``.

        Divided by the section's cell count so that the term is nats per cell and ``w_layout``
        means the same thing on sections of different size — the same argument as ``zinb_nll``
        being a mean.
        """
        cells_data = torch.from_numpy(targets.cells_xyz.astype(np.float32))
        mc_data = torch.from_numpy(targets.mc_xyz.astype(np.float32))
        cells_model = (
            cells_data
            if rotation is None
            else torch.from_numpy(rotation.coords(targets.cells_xyz).astype(np.float32))
        )
        mc_model = (
            mc_data
            if rotation is None
            else torch.from_numpy(rotation.coords(targets.mc_xyz).astype(np.float32))
        )
        lam_cells = self.intensity(cells_data, self.field(cells_model))
        lam_mc = self.intensity(mc_data, self.field(mc_model))
        total = layout_poisson_nll(
            lam_cells, targets.cell_type, lam_mc, targets.slab_volume, self.cfg
        )
        return total / float(max(targets.cells_xyz.shape[0], 1))

    # ----------------------------------------------------------------------------------
    # generation
    # ----------------------------------------------------------------------------------

    def generate(self, plane: Plane, cfg: Config, seed: int) -> Any:
        """Generate a virtual section on ``plane``. Returns an ``AnnData``.

        Parameters
        ----------
        plane
            Where to cut. Carries the slab thickness the cell count is integrated over.
        cfg
            The generation-time config. T06 §5 gives the signature with a ``cfg`` beside a
            model that already has one, and the reason is real: ``layout_mode``, ``expr_mode``,
            ``ode_steps`` and ``prior_mode`` are generation-time choices T09's selector varies
            without retraining. Fields that change the *architecture* may not differ from the
            model's own config, and the check below says so rather than producing a silently
            mismatched section.
        seed
            Every draw derives from it (Convention 3): the layout, the count draw, and the
            i.i.d. prior under ablation A1. The GRF is *not* reseeded — its realisation is the
            model's, which is what makes two planes of one stack mutually coherent.

        Notes
        -----
        The emitted matrix is **always a draw**, never ``mu``: :func:`sample_counts` under
        ``expr_mode="zinb-flow"`` and :func:`cross_mix_counts` under ``"cross-mix"``, and
        :func:`assert_detection_rate` checks the result against the training sections' median
        detection rate before it is returned.
        """
        self._check_generation_cfg(cfg)
        gen = np.random.default_rng(seed)
        layout = sample_layout(
            intensity_fn_from_head(self.intensity, self.field),
            plane,
            cfg,
            seed,
            repulsion=self.repulsion,
            flanking=self._flanking(plane),
        )
        xyz = layout.coords_xyz.astype(np.float64)
        cell_type = torch.from_numpy(layout.cell_type.astype(np.int64))
        region = self._region_of(xyz)

        neighbours, weights = self.data.index.query(
            xyz, set(), seed=int(gen.integers(0, 2**31 - 1))
        )
        if cfg.expr_mode == "cross-mix":
            counts = self._cross_mix_expression(neighbours, weights, gen, cfg)
        elif cfg.expr_mode == "zinb-flow":
            counts = self._flow_expression(xyz, cell_type, region, neighbours, gen, cfg, seed=seed)
        else:
            raise ExpressionError(
                f"Config.expr_mode={cfg.expr_mode!r} is not a T06 path. 'zinb-flow' and "
                "'cross-mix' are built here; 'auto-blend' is T09's uncertainty-gated "
                "anchoring, which blends between these two and owns the gate."
            )
        assert_detection_rate(counts, self.stats.median_detection, cfg)
        return self._to_anndata(layout, counts, region, cfg)

    def _check_generation_cfg(self, cfg: Config) -> None:
        """Refuse a generation config that disagrees with the model's architecture."""
        architectural = (
            "latent_dim",
            "field_dim",
            "gene_emb_dim",
            "ctx_emb_dim",
            "retrieval_ctx_dim",
            "retrieval_n_heads",
            "fourier_bands_xy",
            "fourier_bands_z",
            "z_embed_dim",
            "decoder",
            "triplane_res_xy",
            "triplane_res_z",
            "triplane_channels",
            "n_plane_orientations",
        )
        differing = [f for f in architectural if getattr(cfg, f) != getattr(self.cfg, f)]
        if differing:
            raise ExpressionError(
                f"CTFFlow.generate: the generation config differs from the model's on "
                f"{differing}, which changes the architecture rather than the generation "
                "policy. Vary layout_mode / expr_mode / ode_steps / prior_mode / ell_* here; "
                "anything else needs a retrained model."
            )

    def _flanking(self, plane: Plane) -> list[FlankingSection]:
        """Return the two training sections nearest ``plane``, in its frame — T05's evidence."""
        depth = float(plane.origin[2])
        order = np.argsort(np.abs(self.data.vol.z_values.astype(np.float64) - depth))
        return [flanking_from_section(self.data.vol.sections[int(i)], plane) for i in order[:2]]

    def _region_of(self, xyz: FloatArray) -> Tensor | None:
        """Nearest real cell's region code for each generated position. ``(N,)`` int64.

        A generated cell has no region label of its own, and the field is not asked to invent
        one: the label comes from the nearest *real* cell, which is the same nearest-neighbour
        transfer the benchmark's own evaluators use. ``None`` when the dataset has no regions.
        """
        if self.data.index.region is None:
            return None
        idx, _ = self.data.index.query(xyz, set(), seed=0)
        nearest = np.where(idx[:, 0] >= 0, idx[:, 0], 0)
        return torch.from_numpy(self.data.index.region[nearest].astype(np.int64))

    def _flow_expression(
        self,
        xyz: FloatArray,
        cell_type: Tensor,
        region: Tensor | None,
        neighbours: IntArray,
        gen: np.random.Generator,
        cfg: Config,
        *,
        seed: int,
    ) -> Tensor:
        """Run the ``zinb-flow`` path: GRF -> flow -> decoder -> a draw. ``(N, G)`` float32."""
        points = torch.from_numpy(xyz.astype(np.float32))
        with torch.no_grad():
            tokens, mask = self.data.index.neighbour_tokens(xyz, neighbours)
            cond, _ = self.conditioning(points, points, cell_type, region, tokens, mask)
            h0 = self.prior_latent(xyz, seed=seed)
            h = self.flow.sample(h0, cond, int(cfg.ode_steps))
            size_factor = self.size_head.size_factor(h)
            gene_emb = self.embeddings.gene(torch.arange(self.stats.n_genes, dtype=torch.long))
            mu, theta, pi = self.decoder(h, gene_emb, size_factor)
        draw = sample_zigamma if cfg.decoder == "zigamma" else sample_counts
        return draw(mu, theta, pi, gen)

    def _cross_mix_expression(
        self,
        neighbours: IntArray,
        weights: FloatArray,
        gen: np.random.Generator,
        cfg: Config,
    ) -> Tensor:
        """Run the ``cross-mix`` path: real donor counts, chosen per gene. ``(N, G)`` float32.

        The retrieval score's donor weights *are* v20's mixing weights, renormalised over the
        admissible donors — v20 read them off the two flanking sections and the fractional
        depth between them, which is what this score's in-plane and z terms compute
        continuously. Padded slots get weight zero, so a cell with fewer than ``K`` donors
        mixes only over the ones it has.
        """
        valid = neighbours >= 0
        w = np.where(valid, weights, 0.0)
        total = w.sum(axis=1, keepdims=True)
        if not np.all(total > 0):
            raise ExpressionError(
                "CTFFlow.generate: some generated cell has no admissible donor at all, so the "
                "cross-mix has nothing real to emit. Widen Config.retrieval_z_window or "
                "generate closer to the training sections."
            )
        w = w / total
        safe = np.where(valid, neighbours, 0)
        donors = np.asarray(self.data.counts[safe.reshape(-1)].todense(), dtype=np.float64)
        donors = donors.reshape(safe.shape[0], safe.shape[1], -1)
        return cross_mix_counts(donors, w, gen, cfg=cfg)

    def _to_anndata(self, layout: Any, counts: Tensor, region: Tensor | None, cfg: Config) -> Any:
        """Package a generated section as an AnnData the loader could read back."""
        import anndata as ad
        import pandas as pd

        values = counts.detach().cpu().numpy()
        obs = pd.DataFrame(
            {
                cfg.z_key: layout.coords_xyz[:, 2].astype(np.float64),
                cfg.celltype_key: pd.Categorical(
                    [self.data.vol.celltype_names[int(c)] for c in layout.cell_type],
                    categories=list(self.data.vol.celltype_names),
                ),
            },
            index=[f"gen{i:06d}" for i in range(values.shape[0])],
        )
        if region is not None and self.data.vol.region_names is not None:
            obs[str(cfg.region_key)] = pd.Categorical(
                [self.data.vol.region_names[int(r)] for r in region.numpy()],
                categories=list(self.data.vol.region_names),
            )
        adata = ad.AnnData(
            X=sparse.csr_matrix(values),
            obs=obs,
            var=pd.DataFrame(index=list(self.data.vol.gene_names)),
        )
        # In-plane (u, v) as the coordinate key, because that is what a section reports and
        # what every in-plane metric is computed on; the physical 3-D positions travel beside
        # it, because an oblique plane's cells do not share a depth.
        adata.obsm[cfg.coord_key] = layout.coords_uv.astype(np.float32)
        adata.obsm["xyz"] = layout.coords_xyz.astype(np.float32)
        adata.uns[cfg.thickness_key] = float(layout.plane.thickness)
        adata.uns["generated"] = {
            "expr_mode": cfg.expr_mode,
            "layout_mode": layout.mode,
            "seed": int(layout.seed),
            "n_expected": float(layout.n_expected),
            "specimen_id": self.data.vol.specimen_id,
        }
        return adata


# --------------------------------------------------------------------------------------
# the trainer
# --------------------------------------------------------------------------------------


class EMA:
    """Exponential moving average of a module's parameters. The T07 teacher.

    ``shadow <- decay * shadow + (1 - decay) * live`` after every optimiser step, at
    ``Config.ema_decay``. Buffers are copied rather than averaged: the GRF's draws and the
    triplane's bounding box are the same in both copies, and averaging a bounding box would be
    meaningless.

    :meth:`swap_in` / :meth:`restore` make the averaged weights active for the duration of an
    evaluation without a second model object, which is how T07's teacher will read the same
    architecture with different weights.
    """

    def __init__(self, module: nn.Module, decay: float) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA: decay must lie in (0, 1), got {decay!r}")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        self._saved: dict[str, Tensor] = {}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        """Fold the module's current parameters into the average."""
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap_in(self, module: nn.Module) -> None:
        """Make the averaged weights active, remembering the live ones."""
        self._saved = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if name in self.shadow
        }
        for name, parameter in module.named_parameters():
            if name in self.shadow:
                parameter.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, module: nn.Module) -> None:
        """Put the live weights back. Raises if :meth:`swap_in` was not called first."""
        if not self._saved:
            raise RuntimeError("EMA.restore: nothing to restore; call swap_in first")
        for name, parameter in module.named_parameters():
            if name in self._saved:
                parameter.copy_(self._saved[name])
        self._saved = {}


@dataclass
class TrainHistory:
    """What a training run recorded, every ``Config.log_every`` steps.

    Attributes
    ----------
    steps
        The step indices the entries were taken at.
    terms
        Per-term **unweighted** loss, by name. Logged separately because a collapse shows up
        as one term diverging while the total looks fine — the reason ``forward_train``
        returns a dict at all.
    total
        The weighted sum actually optimised.
    attention_entropy
        Mean attention entropy over the ``K`` retrieved donors, in nats. GATE 2 measured
        3.422 (0.987 x log K) with a linear probe and could not test selectivity; T06 requires
        this to fall. ``log_k`` is beside it so the ratio needs no arithmetic at the call site.
    gene_variance
        Mean over genes of the variance of ``mu`` across the batch's cells. Not an alarm yet —
        T07 owns the collapse alarm — but the quantity it will watch, logged from the start so
        there is a trajectory to compare against.
    """

    steps: list[int] = dataclass_field(default_factory=list)
    terms: dict[str, list[float]] = dataclass_field(default_factory=dict)
    total: list[float] = dataclass_field(default_factory=list)
    attention_entropy: list[float] = dataclass_field(default_factory=list)
    gene_variance: list[float] = dataclass_field(default_factory=list)
    variance_ratio: list[float] = dataclass_field(default_factory=list)
    """Per-gene variance of a *draw* from the decoder over the batch's real counts'. What
    T07's collapse alarm watches; 1.0 is agreement, and the alarm fires below
    ``Config.sefl_collapse_warn_fraction``."""
    consistency_ratio: list[float] = dataclass_field(default_factory=list)
    """``sum(weighted consistency) / sum(weighted reconstruction)`` at each logged step
    (``specs/07`` §5). Empty on a run with the SEFL weights at zero."""
    collapse_alarms: list[int] = dataclass_field(default_factory=list)
    """Steps at which the collapse alarm fired. ``specs/07``'s "collapse-alarm history"."""
    log_k: float = 0.0

    def record(self, step: int, terms: dict[str, Tensor], total: float) -> None:
        """Append one entry."""
        self.steps.append(int(step))
        self.total.append(float(total))
        for name, value in terms.items():
            if name.startswith(DIAG_PREFIX):
                continue
            self.terms.setdefault(name, []).append(float(value.detach()))
        self.attention_entropy.append(float(terms[f"{DIAG_PREFIX}attention_entropy"].detach()))
        self.gene_variance.append(float(terms[f"{DIAG_PREFIX}gene_variance"].detach()))
        real = float(terms[f"{DIAG_PREFIX}gene_variance_real"].detach())
        generated = float(terms[f"{DIAG_PREFIX}gene_variance_gen"].detach())
        self.variance_ratio.append(generated / real if real > 0.0 else float("nan"))

    @property
    def entropy_fraction(self) -> list[float]:
        """The attention entropy as a fraction of ``log K``, which is how G2.4 reads it."""
        if self.log_k <= 0.0:
            raise ValueError("TrainHistory.entropy_fraction: log_k was never set")
        return [value / self.log_k for value in self.attention_entropy]


def loss_weights(cfg: Config) -> dict[str, float]:
    """Return the weight the trainer applies to each name in :data:`LOSS_TERMS`."""
    return {
        "recon": float(cfg.w_recon),
        "cfm": float(cfg.w_cfm),
        "size": float(cfg.w_size),
        "layout": float(cfg.w_layout),
        "distill": float(cfg.w_distill),
        "tv_z": float(cfg.tv_z_weight),
        "cross": float(cfg.w_cross),
        "thick": float(cfg.w_thick),
        "prog": float(cfg.w_prog),
    }


def train_ctfflow(
    model: CTFFlow,
    cfg: Config,
    *,
    steps: int,
    seed: int,
    gene_pool: IntArray | None = None,
    ema: EMA | None = None,
    teacher: EMATeacher | None = None,
) -> TrainHistory:
    """Fit ``model`` for ``steps`` optimiser steps. Returns the recorded :class:`TrainHistory`.

    AdamW at ``Config.lr`` / ``Config.weight_decay``, a cosine schedule down to
    ``lr_min_frac`` of the peak, gradient-norm clipping at ``Config.grad_clip``, and an
    :class:`EMA` of the weights at ``Config.ema_decay`` — the copy T07 reuses as its teacher.
    ``epochs`` is not the unit here: the loop is stated in optimiser steps, which is what the
    schedule, the residual-gate anneal and T07's SEFL ramp are all expressed in, and a caller
    that thinks in epochs passes ``epochs * cells / batch_cells``.

    Parameters
    ----------
    model, cfg
        The model and the config. ``cfg`` is passed explicitly rather than read off the model
        so that a fit at a different learning rate or step budget does not need a second
        model.
    steps, seed
        The budget and the seed everything derives from: batch composition, gene subsampling,
        the per-step rotation, the retrieval dropout and the CFM time draw (Convention 3).
    gene_pool
        Genes the batches may draw from. ``None`` is the whole panel; the zero-shot
        experiment passes the 80% it is allowed to train on, and nothing else in the run needs
        to know about the holdout.
    ema
        An existing average to continue, or ``None`` to start one.
    teacher
        T07's stop-gradient :class:`~spatialcpav25_gen.losses.sefl.EMATeacher`, or ``None`` to
        build one when any SEFL weight is non-zero. Separate from ``ema``: that one averages
        the weights a caller may want *afterwards*, this one is a second model evaluated
        *during* the step, and the two have different lifetimes.

    SEFL
    ----
    When any of ``w_cross`` / ``w_thick`` / ``w_prog`` is non-zero the loop adds T07's
    consistency block on every ``Config.sefl_every_n_steps``-th step, at
    :func:`~spatialcpav25_gen.losses.sefl.sefl_ramp`'s multiplier — zero through the warm-up,
    then linear. The block is backpropagated *separately* from the batch terms and the
    gradients accumulate: the two graphs are built under different field poses, and one
    ``backward`` per graph keeps the augmentation's in-place rebinding of the field's query
    frames out of the other's backward pass. The consistency/reconstruction ratio is logged
    and warned about, and the collapse alarm runs at every logged step.
    """
    # Deferred, and it has to be: `losses.sefl` evaluates the model it is handed, so importing
    # it at module scope would close the loop `model -> losses.sefl -> model`. The losses need
    # no *type* from here (they annotate `CTFFlow` under `TYPE_CHECKING`), and this module needs
    # them only inside the loop, so one deferred import breaks the cycle in the right place.
    from spatialcpav25_gen.losses.sefl import EMATeacher, sefl_active, sefl_ramp, sefl_terms

    if steps < 1:
        raise ValueError(f"train_ctfflow: steps must be >= 1, got {steps}")
    data = model.data
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=steps, eta_min=float(cfg.lr) * float(cfg.lr_min_frac)
    )
    average = EMA(model, float(cfg.ema_decay)) if ema is None else ema
    weights = loss_weights(cfg)
    history = TrainHistory(log_k=math.log(int(cfg.retrieval_k)))
    centre = data.vol.bbox.mean(axis=0).astype(np.float64)
    sefl_on = max(float(cfg.w_cross), float(cfg.w_thick), float(cfg.w_prog)) > 0.0
    sefl_teacher = (EMATeacher(model, cfg) if teacher is None else teacher) if sefl_on else None
    sefl_gen = np.random.default_rng([seed, 0x5EF1])

    model.train()
    for step in range(steps):
        model.embeddings.set_progress(step / max(steps - 1, 1))
        batch = data.sample_batch(cfg, seed=seed, step=step, gene_pool=gene_pool)
        # No `planes` channel: every plane in a training step is consumed by drawing points on
        # it, and those points are rotated through `coords`. Naming the three used channels is
        # what makes the omission reviewable (GATE 2's G2.1h).
        with RotationContext.random(
            cfg,
            seed + step,
            centre,
            requires=("coords", "retrieval", "grf"),
            fields=(model.field,),
        ) as rotation:
            terms = model.forward_train(batch, rotation=rotation)
            unknown = sorted(
                name for name in terms if not name.startswith(DIAG_PREFIX) and name not in weights
            )
            if unknown:
                raise KeyError(
                    f"train_ctfflow: forward_train returned unweighted term(s) {unknown}. "
                    "Add the weight to Config and to loss_weights()."
                )
            total = torch.stack(
                [
                    weights[name] * value
                    for name, value in terms.items()
                    if not name.startswith(DIAG_PREFIX)
                ]
            ).sum()
            optimiser.zero_grad(set_to_none=True)
            total.backward()  # type: ignore[no-untyped-call]

        running = float(total.detach())
        if sefl_teacher is not None:
            ramp = sefl_ramp(step, steps, cfg)
            if ramp > 0.0 and sefl_active(step, cfg):
                consistency = sefl_terms(model, sefl_teacher, cfg, sefl_gen, gene_pool=gene_pool)
                weighted = torch.stack(
                    [
                        ramp * weights[name] * value
                        for name, value in consistency.items()
                        if not name.startswith(DIAG_PREFIX)
                    ]
                ).sum()
                # A plane pair that misses returns constant zeros, so a step in which every
                # live SEFL term skipped has nothing to differentiate — and calling
                # `backward` on it raises rather than being a no-op.
                if weighted.requires_grad:
                    weighted.backward()  # type: ignore[no-untyped-call]
                running += float(weighted.detach())
                terms.update(consistency)

        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip))
        optimiser.step()
        scheduler.step()
        average.update(model)
        if sefl_teacher is not None:
            sefl_teacher.update(model)
        if step % int(cfg.log_every) == 0 or step == steps - 1:
            history.record(step, terms, running)
            _log_sefl(
                history,
                terms,
                weights,
                cfg,
                step=step,
                alarm=(
                    sefl_teacher is not None
                    and sefl_ramp(step, steps, cfg) > 0.0
                    and step >= int(cfg.sefl_collapse_min_steps)
                ),
            )
    model.eval()
    if not np.isfinite(history.total[-1]):
        warnings.warn(
            f"train_ctfflow: the final total loss is {history.total[-1]}; the run diverged. "
            f"Config.grad_clip={cfg.grad_clip} and Config.lr={cfg.lr} are where to look.",
            RuntimeWarning,
            stacklevel=2,
        )
    return history


def _log_sefl(
    history: TrainHistory,
    terms: dict[str, Tensor],
    weights: dict[str, float],
    cfg: Config,
    *,
    step: int,
    alarm: bool,
) -> None:
    """Record T07's two run-time diagnostics: the dominance ratio and the collapse alarm.

    Both are ``specs/07`` §2 and §5 requirements and both are *warnings*, not errors: a
    consistency term that briefly overtakes reconstruction during the ramp is survivable and
    the trajectory is the diagnosis, while a run that ends with either of them firing has a
    result that must be reported rather than quietly used.

    ``alarm`` gates the collapse check on two conditions, and both are needed. The SEFL block
    must be **live** (past the warm-up, ramp above zero) — the alarm is about a variance that
    *drops*, so it starts watching when the thing that could push it down does. And the run
    must be past ``Config.sefl_collapse_min_steps``, because the ramp gate is a *fraction* of
    the run and opens after a handful of steps on a short one, where an untrained decoder
    predicting near the panel mean reads 0.008 of the real variance and trips it. That is
    initialisation, not collapse, and an alarm that cries wolf on every short run is one
    nobody believes on the long run where it matters.
    """
    from spatialcpav25_gen.losses.sefl import (
        ConsistencyDominanceWarning,
        check_collapse,
        consistency_ratio,
    )

    if any(name in terms for name in ("cross", "thick", "prog")):
        ratio = consistency_ratio(terms, weights)
        history.consistency_ratio.append(ratio)
        if ratio > float(cfg.sefl_consistency_ratio_warn):
            warnings.warn(
                f"train_ctfflow: at step {step} the weighted consistency terms are {ratio:.2f} "
                f"of the reconstruction terms, above Config."
                f"sefl_consistency_ratio_warn={cfg.sefl_consistency_ratio_warn}. Consistency "
                "is a regulariser, not the objective (specs/07 5).",
                ConsistencyDominanceWarning,
                stacklevel=2,
            )
    if alarm and check_collapse(
        float(terms[f"{DIAG_PREFIX}gene_variance_gen"].detach()),
        float(terms[f"{DIAG_PREFIX}gene_variance_real"].detach()),
        step,
        cfg,
    ):
        history.collapse_alarms.append(int(step))


def flanking_sections(vol: Volume, plane: Plane, *, k: int = 2) -> list[Section]:
    """Return the ``k`` sections of ``vol`` nearest ``plane``'s depth, nearest first.

    The evidence the ``resample`` and ``hybrid`` layout modes are defined against, and the
    donor pool the independent-donor baseline draws from in T10.
    """
    order = np.argsort(np.abs(vol.z_values.astype(np.float64) - float(plane.origin[2])))
    return [vol.sections[int(i)] for i in order[:k]]
