"""Generation at an arbitrary plane, stack, oblique series or curved surface (T09 §1).

What this module owns
---------------------
:func:`generate_section` is the single generation entry point; ``CTFFlow.generate``
delegates to it (SPEC_QUESTIONS C9, which flagged the two signatures as a drift). Beside it
sit the three multi-section wrappers ``specs/09`` §1 asks for — :func:`generate_stack`,
:func:`generate_oblique`, :func:`generate_curved` — and the two things T09 adds to T06's
path:

* **uncertainty-gated anchoring** (``Config.expr_mode = "auto-blend"``). ``M =
  Config.n_uncertainty_samples`` flow samples under *different GRF realisations* give a
  per-cell latent variance ``v_i``; a learned, non-increasing ``w(v)`` — the isotonic map
  :func:`~spatialcpav25_gen.infer.calibrate.calibrate_anchor_weight` fits on LOSO
  reconstructions — decides, per cell, how much of a **retrieval-anchored real profile** to
  mix in. The mixing is the v20 Bernoulli cross-mix per gene, so every emitted value is
  still a count. There is no ``gap_scale``, no ``alpha_tol`` and no ``edit_weight``: T09's
  "Do NOT" forbids exposing them and this is what replaces them.
* **a derived retrieval z window and a hard empty-pool failure**. ``specs/09`` §1: the
  window is a function of the geometry — the gap from the query plane to its nearest
  admissible section — and once it is derived, an empty candidate pool means the geometry is
  impossible rather than that the constant was too small. Above
  ``Config.generation_empty_pool_tol`` it is an :class:`EmptyCandidatePoolError`, not a
  warning.

Coherence
---------
One GRF realisation per call. :func:`generate_stack`, :func:`generate_oblique` and
:func:`generate_curved` resolve ``grf_seed`` **once** and pass it to every section they
emit, so the sections are slices of one object rather than independent samples of the same
distribution. Reusing the same ``grf_seed`` across separate calls is what extends that
guarantee between them, and it is a parameter rather than an internal precisely so a caller
can. ``grf_seed=None`` means the model's own realisation, which is the coherent default.

The boundary is a different regime (open risk R3)
-------------------------------------------------
A plane at or beyond the outermost training section is extrapolating: its cells have
training sections and retrieval donors on one side only, and T04 measured 20-35 % worse
reconstruction there than in the interior. Every emitted section therefore carries
``uns["boundary"]``, so a caller cannot assume uniform quality across the stack, and
:func:`latent_uncertainty` is exposed separately so the claim that the uncertainty gate
*notices* the boundary can be measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy import sparse
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume, Volume
from spatialcpav25_gen.infer.planes import Plane, Surface, plane_from_normal
from spatialcpav25_gen.losses.metric_aware import require_training_volume
from spatialcpav25_gen.model.expression import (
    ExpressionError,
    assert_detection_rate,
    cross_mix_counts,
    sample_counts,
    sample_zigamma,
)
from spatialcpav25_gen.model.layout import (
    Layout,
    flanking_from_section,
    intensity_fn_from_head,
    sample_layout,
)
from spatialcpav25_gen.model.noise import GaussianRandomField
from spatialcpav25_gen.model.retrieval import RetrievalIndex

if TYPE_CHECKING:  # pragma: no cover - the model imports this module, not the other way
    from spatialcpav25_gen.infer.calibrate import DetectionCalibration, IsotonicRegressor
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "EmptyCandidatePoolError",
    "GenerationError",
    "LatentUncertainty",
    "anchor_blend",
    "emitted_counts",
    "generate_curved",
    "generate_oblique",
    "generate_section",
    "generate_stack",
    "inscribed_half_extent",
    "is_boundary_plane",
    "latent_uncertainty",
    "plane_at_z",
    "required_z_window",
    "stack_correlation",
    "stack_pair_correlations",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


class GenerationError(RuntimeError):
    """Raised when a section cannot be generated as asked."""


class EmptyCandidatePoolError(GenerationError):
    """More than a negligible fraction of generated cells had no retrieval donor at all.

    ``specs/09`` §1 turns T04's :class:`
    ~spatialcpav25_gen.model.retrieval.EmptyCandidatePoolWarning` into a failure *here*,
    once the z window is derived from the geometry: a warning was the right response to a
    constant that might be too small, and the wrong response to a derived window, where an
    empty pool means the requested plane has no admissible evidence anywhere in the stack
    and every affected cell's attention is returning its bias.
    """


@dataclass(frozen=True)
class LatentUncertainty:
    """Per-cell latent variance across ``M`` GRF realisations, and the sample that was kept.

    Attributes
    ----------
    variance
        ``(N,)`` float64 mean over latent channels of the variance of ``h`` across the ``M``
        samples. The quantity ``w(v)`` is a function of.
    h
        ``(N, d_h)`` float32 the **canonical** sample: the one drawn under the call's own GRF
        realisation, i.e. the one a stack's other sections are coherent with. The extra
        realisations exist to measure spread, not to replace it.
    n_samples
        ``M``, as it was actually run.
    """

    variance: FloatArray
    h: Tensor
    n_samples: int


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------


def plane_at_z(vol: Volume, z: float, cfg: Config, *, thickness: float | None = None) -> Plane:
    """Return the axis-aligned plane at depth ``z`` spanning ``vol``'s in-plane extent.

    ``thickness`` defaults to the median thickness of the volume's own sections, which is
    the slab a virtual section at this depth stands for. The window is the volume's
    bounding box, not one section's, so two planes of a stack share a frame and their cells
    are comparable in ``(u, v)``.
    """
    box = np.asarray(vol.bbox, dtype=np.float64)
    centre = 0.5 * (box[0, :2] + box[1, :2])
    half = 0.5 * (box[1, :2] - box[0, :2])
    slab = float(np.median([s.thickness for s in vol.sections])) if thickness is None else thickness
    if cfg.debug_shapes:
        assert half.shape == (2,), half.shape
    return plane_from_normal(
        normal=np.array([0.0, 0.0, 1.0]),
        origin=np.array([centre[0], centre[1], float(z)]),
        half_extent=half,
        thickness=slab,
    )


def inscribed_half_extent(bbox: npt.ArrayLike, e1: FloatArray, e2: FloatArray) -> FloatArray:
    """Largest centred square window on a plane that stays inside ``bbox``. ``(2,)`` float64.

    The window's corner is ``centre ± t e1 ± t e2``, so the binding constraint per axis is
    ``t (|e1_a| + |e2_a|) <= half_span_a`` and the answer is the smallest such ``t``. An
    oblique plane through a flat block is limited by the block's *depth*, which is what makes
    a naive "half the in-plane extent" window leave the tissue at its corners.
    """
    box = np.asarray(bbox, dtype=np.float64)
    half_span = 0.5 * (box[1] - box[0])
    reach = np.abs(np.asarray(e1, dtype=np.float64)) + np.abs(np.asarray(e2, dtype=np.float64))
    with np.errstate(divide="ignore"):
        limits = np.where(reach > 0.0, half_span / np.where(reach > 0.0, reach, 1.0), np.inf)
    t = float(np.min(limits))
    return np.asarray([t, t], dtype=np.float64)


def is_boundary_plane(vol: Volume, plane: Plane, cfg: Config) -> bool:
    """Whether ``plane`` sits outside the training stack's interior (open risk R3).

    ``True`` when the plane's depth is above the last section, below the first, or within
    ``Config.boundary_margin_spacings`` median spacings of either end — the regime where a
    cell's evidence is one-sided and T04 measured a 20-35 % reconstruction deficit.
    """
    z = float(plane.origin[2])
    depths = vol.z_values.astype(np.float64)
    margin = float(cfg.boundary_margin_spacings) * float(vol.median_spacing)
    return bool(z < float(depths.min()) + margin or z > float(depths.max()) - margin)


def required_z_window(
    vol: Volume, plane: Plane, cfg: Config, *, exclude_z: set[float] | None = None
) -> float:
    """Smallest ``retrieval_z_window`` that admits the nearest admissible section. Float.

    ``specs/09`` §1: the window's floor is ``(gap to the nearest admissible section) /
    median_spacing``, **rounded up**, not the fixed 3 that left a fifth of a batch with no
    donor at all under a ``consecutive-3`` holdout. ``Config.retrieval_z_window`` stays as
    the fallback and the ablation handle, so the value returned is never below it.

    Leakage-free by construction: the gap is measured against ``vol``'s own sections, which
    is a :class:`~spatialcpav25_gen.data.schema.TrainingVolume` everywhere in the pipeline.
    """
    excluded = set() if exclude_z is None else exclude_z
    spacing = max(float(vol.median_spacing), 1e-6)
    tolerance = 1e-3 * spacing
    admissible = [
        float(z)
        for z in vol.z_values.astype(np.float64)
        if not any(abs(float(z) - float(e)) <= tolerance for e in excluded)
    ]
    if not admissible:
        raise GenerationError(
            f"required_z_window: every section of specimen {vol.specimen_id!r} is excluded "
            f"({sorted(excluded)}), so no window admits a donor. The plane at z="
            f"{float(plane.origin[2]):g} um has no evidence to generate from."
        )
    gap = float(np.min(np.abs(np.asarray(admissible) - float(plane.origin[2]))))
    return float(max(float(cfg.retrieval_z_window), np.ceil(gap / spacing)))


# --------------------------------------------------------------------------------------
# context managers: the noise realisation and the retrieval window
# --------------------------------------------------------------------------------------


@contextmanager
def _using_field(model: CTFFlow, cfg: Config, grf_seed: int | None) -> Iterator[None]:
    """Swap in the GRF realisation and length-scale this call generates under.

    Three cases, and the middle one is the reason this exists at all:

    * ``grf_seed is None`` and ``cfg``'s ``ell`` matches the model's — nothing to do, and the
      model's own realisation is used, which is what makes repeated calls coherent;
    * ``cfg``'s ``ell`` differs (a calibrated length-scale, which
      ``CTFFlow.check_generation_cfg`` explicitly permits) — the **same** realisation
      rescaled through
      :meth:`~spatialcpav25_gen.model.noise.GaussianRandomField.with_lengthscale`, no redraw,
      so calibrating ``ell`` does not silently change which noise the stack is a slice of.
      Without this the calibrated value would never reach the prior at all;
    * ``grf_seed`` is given — a fresh realisation at that seed, which is how the uncertainty
      gate's ``M`` samples and a deliberately different stack are drawn.
    """
    ell = (float(cfg.ell_xy), float(cfg.ell_xy), float(cfg.ell_z))
    model_ell = (float(model.cfg.ell_xy), float(model.cfg.ell_xy), float(model.cfg.ell_z))
    if grf_seed is None and ell == model_ell:
        yield
        return
    original = model.grf
    field = (
        original.with_lengthscale(ell)
        if grf_seed is None
        else GaussianRandomField(cfg, ell, int(grf_seed))
    )
    model.grf = field
    try:
        yield
    finally:
        model.grf = original


@contextmanager
def _using_z_window(index: RetrievalIndex, window: float) -> Iterator[None]:
    """Run a query under a derived ``retrieval_z_window``.

    The index captures its config at construction, so a window derived per plane has to be
    swapped in rather than passed down. Only the window changes: rebuilding the index would
    recompute the niche and the expression PCA for a bound that is read at query time.
    """
    original = index.cfg
    if float(original.retrieval_z_window) == float(window):
        yield
        return
    index.cfg = original.replace(retrieval_z_window=float(window))
    try:
        yield
    finally:
        index.cfg = original


# --------------------------------------------------------------------------------------
# the pieces of one section
# --------------------------------------------------------------------------------------


def _flanking(vol: Volume, plane: Plane) -> list[Section]:
    """Return the two sections nearest ``plane``'s depth, nearest first — T05's evidence."""
    order = np.argsort(np.abs(vol.z_values.astype(np.float64) - float(plane.origin[2])))
    return [vol.sections[int(i)] for i in order[:2]]


def _region_of(model: CTFFlow, xyz: FloatArray) -> Tensor | None:
    """Nearest real cell's region code for each generated position. ``(N,)`` int64 or ``None``.

    A generated cell has no region label of its own and the field is not asked to invent one:
    the label comes from the nearest *real* cell, the same nearest-neighbour transfer the
    benchmark's own evaluators use.
    """
    if model.data.index.region is None:
        return None
    idx, _ = model.data.index.query(xyz, set(), seed=0)
    nearest = np.where(idx[:, 0] >= 0, idx[:, 0], 0)
    return torch.from_numpy(model.data.index.region[nearest].astype(np.int64))


def _check_empty_pool(neighbours: IntArray, cfg: Config, plane: Plane) -> float:
    """Return the empty-pool fraction, raising above ``Config.generation_empty_pool_tol``."""
    empty = np.asarray(neighbours < 0).all(axis=1)
    fraction = float(empty.mean()) if empty.size else 0.0
    if fraction > float(cfg.generation_empty_pool_tol):
        raise EmptyCandidatePoolError(
            f"generate_section: {int(empty.sum())} of {int(empty.size)} cells "
            f"({fraction:.1%}) had no admissible retrieval donor at the plane through "
            f"z={float(plane.origin[2]):g} um, above "
            f"Config.generation_empty_pool_tol={cfg.generation_empty_pool_tol}. The z window "
            "is derived from this plane's own gap to the nearest admissible section "
            "(specs/09 §1), so this is not a window that is too narrow: the exclusions have "
            "left these cells no candidate section at all. Their attention would return its "
            "bias and the retrieval branch would be silently absent."
        )
    return fraction


def latent_uncertainty(
    model: CTFFlow,
    xyz: FloatArray,
    cell_type: Tensor,
    region: Tensor | None,
    neighbours: IntArray,
    cfg: Config,
    *,
    seed: int,
    grf_seed: int | None = None,
) -> LatentUncertainty:
    """Per-cell latent variance over ``M`` GRF realisations. See :class:`LatentUncertainty`.

    ``M = Config.n_uncertainty_samples``. Sample 0 is drawn under the call's **own**
    realisation (``grf_seed``, or the model's when that is ``None``) and is the one returned
    in :attr:`LatentUncertainty.h`; samples ``1..M-1`` are drawn under realisations seeded
    from ``seed`` and exist only to measure spread. Doing it the other way round — averaging
    the samples — would break stack coherence, because the average of ``M`` realisations is
    not a realisation of anything.

    Under ``Config.prior_mode = "iid"`` the same construction re-seeds the i.i.d. draw
    instead, which is ablation A1's version of the same quantity.

    Shapes: ``(N, 3)`` positions and ``(N, K)`` neighbours in; ``(N,)`` variance and
    ``(N, d_h)`` latent out.
    """
    points = torch.from_numpy(np.asarray(xyz, dtype=np.float32))
    tokens, mask = model.data.index.neighbour_tokens(xyz, neighbours)
    samples: list[Tensor] = []
    n_samples = max(int(cfg.n_uncertainty_samples), 1)
    gen = np.random.default_rng([int(seed), 0xA0C]).integers(0, 2**31 - 1, size=n_samples)
    with torch.no_grad():
        cond, _ = model.conditioning(points, points, cell_type, region, tokens, mask)
        for m in range(n_samples):
            realisation = grf_seed if m == 0 else int(gen[m])
            with _using_field(model, cfg, realisation):
                h0 = model.prior_latent(xyz, seed=int(gen[m]))
            samples.append(model.flow.sample(h0, cond, int(cfg.ode_steps)))
    stacked = torch.stack(samples, dim=0)
    variance = (
        stacked.var(dim=0, unbiased=True).mean(dim=1).numpy().astype(np.float64)
        if n_samples > 1
        else np.zeros(stacked.shape[1], dtype=np.float64)
    )
    if cfg.debug_shapes:
        assert variance.shape == (int(xyz.shape[0]),), variance.shape
        assert samples[0].shape == (int(xyz.shape[0]), int(cfg.latent_dim)), samples[0].shape
    return LatentUncertainty(variance=variance, h=samples[0], n_samples=n_samples)


def _decode(
    model: CTFFlow,
    h: Tensor,
    cfg: Config,
    calibration: DetectionCalibration | None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode a latent to ``(mu, theta, pi)`` over the whole panel, calibration applied.

    Three ``(N, G)`` tensors. The calibration is T09 §2's per-gene affine correction on the
    ``pi`` logit **and** on ``log theta``; applying it here rather than inside the decoder
    keeps it a property of *generation* — a correction fitted on LOSO reconstructions is not
    a parameter of the model and must not reach a training step.
    """
    gene_emb = model.embeddings.gene(torch.arange(model.stats.n_genes, dtype=torch.long))
    mu, theta, pi = model.decoder(h, gene_emb, model.size_head.size_factor(h))
    if calibration is not None:
        mu, theta, pi = calibration.apply(mu, theta, pi, cfg)
    return mu, theta, pi


def _flow_counts(
    model: CTFFlow,
    h: Tensor,
    gen: np.random.Generator,
    cfg: Config,
    calibration: DetectionCalibration | None,
) -> Tensor:
    """Draw counts from a latent. ``(N, d_h)`` -> ``(N, G)`` float32, integer-valued."""
    with torch.no_grad():
        mu, theta, pi = _decode(model, h, cfg, calibration)
    draw = sample_zigamma if cfg.decoder == "zigamma" else sample_counts
    return draw(mu, theta, pi, gen)


def _cross_mix(
    model: CTFFlow,
    neighbours: IntArray,
    weights: FloatArray,
    gen: np.random.Generator,
    cfg: Config,
) -> Tensor:
    """Run the ``cross-mix`` path: real donor counts, chosen per gene. ``(N, G)`` float32.

    The retrieval score's donor weights *are* v20's mixing weights, renormalised over the
    admissible donors. Padded slots get weight zero, so a cell with fewer than ``K`` donors
    mixes only over the ones it has.
    """
    valid = neighbours >= 0
    w = np.where(valid, weights, 0.0)
    total = w.sum(axis=1, keepdims=True)
    if not np.all(total > 0):
        raise EmptyCandidatePoolError(
            "generate_section: some generated cell has no admissible donor at all, so the "
            "retrieval-anchored profile has nothing real to emit. This is the same condition "
            "as EmptyCandidatePoolError above, reached through the cross-mix rather than "
            "through the neighbour indices."
        )
    w = w / total
    safe = np.where(valid, neighbours, 0)
    donors = np.asarray(model.data.counts[safe.reshape(-1)].todense(), dtype=np.float64)
    donors = donors.reshape(safe.shape[0], safe.shape[1], -1)
    return cross_mix_counts(donors, w, gen, cfg=cfg)


def anchor_blend(
    generated: Tensor, anchored: Tensor, weight: FloatArray, gen: np.random.Generator
) -> Tensor:
    """Bernoulli per-gene mix of a generated and an anchored profile. ``(N, G)`` float32.

    ``weight[i]`` is the probability that cell ``i`` takes the **anchored** (real) value at
    any one gene, drawn independently per gene — the v20 cross-mix's own mechanism, so every
    emitted value is still a real count or a real draw and never an average of two
    (Convention 5). One ``gen.random((N, G))`` draw is consumed, in C order.
    """
    a = np.asarray(generated.detach().cpu().numpy(), dtype=np.float64)
    b = np.asarray(anchored.detach().cpu().numpy(), dtype=np.float64)
    if a.shape != b.shape:
        raise ExpressionError(f"anchor_blend: shapes {a.shape} and {b.shape} disagree")
    w = np.asarray(weight, dtype=np.float64).reshape(-1, 1)
    if w.shape[0] != a.shape[0]:
        raise ExpressionError(
            f"anchor_blend: {w.shape[0]} weights for {a.shape[0]} cells; w(v) is per cell"
        )
    take_anchor = gen.random(a.shape) < w
    return torch.from_numpy(np.where(take_anchor, b, a).astype(np.float32))


# --------------------------------------------------------------------------------------
# one section
# --------------------------------------------------------------------------------------


def generate_section(
    model: CTFFlow,
    plane: Plane,
    vol: TrainingVolume,
    cfg: Config,
    seed: int,
    *,
    grf_seed: int | None = None,
    calibration: DetectionCalibration | None = None,
    anchor: IsotonicRegressor | None = None,
    exclude_z: set[float] | None = None,
    z_window: float | None = None,
) -> Any:
    """Generate a virtual section on ``plane``. Returns an ``AnnData``.

    The eight steps of ``specs/09`` §1, in order: query the field on the slab, sample the
    layout (T05), draw ``h0`` from the continuous 3-D GRF (T03), retrieve ``K`` real
    neighbours excluding any section at the target depth, integrate the flow ODE (T06),
    decode and **sample** counts, gate the anchoring on the latent uncertainty, and emit.

    Parameters
    ----------
    model
        The trained :class:`~spatialcpav25_gen.model.spatialcpav25_gen.CTFFlow`.
    plane
        Where to cut, carrying the slab thickness the cell count is integrated over.
    vol
        The training volume. A :class:`~spatialcpav25_gen.data.schema.TrainingVolume` by
        type: held-out sections are never evidence (``CLAUDE.md``, leakage discipline). It
        must be the model's own — a second volume would be conditioning evidence the
        retrieval index has never seen, and the mismatch is refused rather than half-used.
    cfg
        The **generation-time** config. ``layout_mode``, ``expr_mode``, ``ode_steps``,
        ``prior_mode``, ``repulsion`` and the calibrated ``ell_*`` may differ from the
        model's; anything architectural may not (``CTFFlow.check_generation_cfg``).
    seed
        Every draw derives from it (Convention 3): the layout, the count draw, the anchor's
        Bernoulli mixing and the i.i.d. prior under ablation A1.
    grf_seed
        The noise realisation. ``None`` — the default — is the model's own, which is what
        makes two calls mutually coherent; an explicit seed draws a different realisation of
        the same field. Documented as a parameter because *reusing* it is the coherence
        guarantee (``specs/09`` §1).
    calibration
        T09 §2's per-gene ``pi`` / ``log theta`` correction, or ``None`` to decode
        uncalibrated.
    anchor
        The fitted ``w(v)``. Required by ``expr_mode="auto-blend"`` and unused otherwise:
        there is no hand-set default blend weight, because a hand-set blend weight is exactly
        what T09 removes.
    exclude_z
        Section depths to remove from the retrieval pool. ``None`` excludes any training
        section **at the target depth** — ``specs/09`` §1's step 4 — which is what stops a
        section from being reconstructed by copying itself.
    z_window
        Pin ``retrieval_z_window`` for this call (the ablation handle T10's A5 needs).
        ``None`` derives it from the plane's own gap to its nearest admissible section
        (:func:`required_z_window`).

    Returns
    -------
    AnnData
        ``X`` raw counts (CSR, integer-valued), ``obsm[cfg.coord_key]`` the in-plane
        ``(u, v)``, ``obsm['spatial_3d']`` and ``obsm['xyz']`` the physical ``(x, y, z)``,
        ``obs`` the cell type and region, and ``uns`` carrying the plane, both seeds, the
        config hash, the boundary flag and the retrieval diagnostics.
    """
    require_training_volume(vol, "generate_section")
    if vol is not model.data.vol:
        raise GenerationError(
            f"generate_section: the volume passed (specimen {vol.specimen_id!r}, "
            f"{vol.n_sections} sections) is not the model's own (specimen "
            f"{model.data.vol.specimen_id!r}, {model.data.vol.n_sections} sections). The "
            "retrieval index, the field's bounding box and the size-factor reference all "
            "belong to the model's volume; generating against another one would mix two "
            "specimens silently."
        )
    model.check_generation_cfg(cfg)
    if cfg.expr_mode == "auto-blend" and anchor is None:
        raise GenerationError(
            "generate_section: Config.expr_mode='auto-blend' needs the fitted w(v) from "
            "calibrate_anchor_weight(model, vol, cfg). T09 forbids a hand-set blend weight "
            "('do not expose gap/alpha/edit flags to users'), so there is no default."
        )

    depth = float(plane.origin[2])
    excluded = _default_exclusions(vol, depth) if exclude_z is None else set(exclude_z)
    window = (
        required_z_window(vol, plane, cfg, exclude_z=excluded) if z_window is None else z_window
    )

    gen = np.random.default_rng(seed)
    with _using_field(model, cfg, grf_seed):
        layout = _layout_on(model, plane, vol, cfg, seed)
        xyz = layout.coords_xyz.astype(np.float64)
        cell_type = torch.from_numpy(layout.cell_type.astype(np.int64))
        region = _region_of(model, xyz)
        with _using_z_window(model.data.index, window):
            neighbours, weights = model.data.index.query(
                xyz, excluded, seed=int(gen.integers(0, 2**31 - 1))
            )
        empty_fraction = _check_empty_pool(neighbours, cfg, plane)
        counts, uncertainty = _expression(
            model,
            xyz,
            cell_type,
            region,
            neighbours,
            weights,
            cfg,
            gen,
            seed=seed,
            grf_seed=grf_seed,
            calibration=calibration,
            anchor=anchor,
        )

    assert_detection_rate(counts, model.stats.median_detection, cfg)
    return _to_anndata(
        model,
        layout,
        counts,
        region,
        cfg,
        plane=plane,
        seed=seed,
        grf_seed=grf_seed,
        vol=vol,
        window=window,
        empty_fraction=empty_fraction,
        uncertainty=uncertainty,
        calibration=calibration,
    )


def _default_exclusions(vol: Volume, depth: float) -> set[float]:
    """Depths to drop from the retrieval pool: any training section at the target depth.

    ``specs/09`` §1 step 4. Matched to within a thousandth of the median spacing, the same
    tolerance :meth:`~spatialcpav25_gen.model.retrieval.RetrievalIndex.query` matches with,
    because a depth that has round-tripped through float32 does not compare equal to the one
    that named it and a silently ineffective exclusion is a leak.
    """
    tolerance = 1e-3 * max(float(vol.median_spacing), 1e-6)
    return {float(z) for z in vol.z_values.astype(np.float64) if abs(float(z) - depth) <= tolerance}


def _layout_on(model: CTFFlow, plane: Plane, vol: Volume, cfg: Config, seed: int) -> Layout:
    """Sample the section's cells and their types on ``plane`` (T05)."""
    return sample_layout(
        intensity_fn_from_head(model.intensity, model.field),
        plane,
        cfg,
        seed,
        repulsion=model.repulsion,
        flanking=[flanking_from_section(s, plane) for s in _flanking(vol, plane)],
    )


def _expression(
    model: CTFFlow,
    xyz: FloatArray,
    cell_type: Tensor,
    region: Tensor | None,
    neighbours: IntArray,
    weights: FloatArray,
    cfg: Config,
    gen: np.random.Generator,
    *,
    seed: int,
    grf_seed: int | None,
    calibration: DetectionCalibration | None,
    anchor: IsotonicRegressor | None,
) -> tuple[Tensor, LatentUncertainty | None]:
    """Run the expression path ``cfg.expr_mode`` names. ``(N, G)`` counts and the gate's input.

    ``cross-mix`` is the v20 fallback (real donor counts per gene), ``zinb-flow`` is T06's
    generative path, and ``auto-blend`` is T09's: the flow's draw, mixed toward the
    retrieval-anchored profile wherever the latent variance says the model is *certain*.
    """
    if cfg.expr_mode == "cross-mix":
        return _cross_mix(model, neighbours, weights, gen, cfg), None
    if cfg.expr_mode == "zinb-flow":
        points = torch.from_numpy(xyz.astype(np.float32))
        with torch.no_grad():
            tokens, mask = model.data.index.neighbour_tokens(xyz, neighbours)
            cond, _ = model.conditioning(points, points, cell_type, region, tokens, mask)
            h0 = model.prior_latent(xyz, seed=seed)
            h = model.flow.sample(h0, cond, int(cfg.ode_steps))
        return _flow_counts(model, h, gen, cfg, calibration), None
    if cfg.expr_mode != "auto-blend":  # pragma: no cover - Config validates the gate
        raise ExpressionError(f"Config.expr_mode={cfg.expr_mode!r} is not a generation path")

    uncertainty = latent_uncertainty(
        model, xyz, cell_type, region, neighbours, cfg, seed=seed, grf_seed=grf_seed
    )
    if anchor is None:  # pragma: no cover - generate_section refuses auto-blend without one
        raise GenerationError("_expression: auto-blend needs the fitted w(v)")
    generated = _flow_counts(model, uncertainty.h, gen, cfg, calibration)
    anchored = _cross_mix(model, neighbours, weights, gen, cfg)
    return anchor_blend(generated, anchored, anchor.predict(uncertainty.variance), gen), uncertainty


def _to_anndata(
    model: CTFFlow,
    layout: Layout,
    counts: Tensor,
    region: Tensor | None,
    cfg: Config,
    *,
    plane: Plane,
    seed: int,
    grf_seed: int | None,
    vol: Volume,
    window: float,
    empty_fraction: float,
    uncertainty: LatentUncertainty | None,
    calibration: DetectionCalibration | None,
) -> Any:
    """Package a generated section as an AnnData the loader could read back."""
    import anndata as ad
    import pandas as pd

    values = counts.detach().cpu().numpy()
    obs = pd.DataFrame(
        {
            cfg.z_key: layout.coords_xyz[:, 2].astype(np.float64),
            cfg.celltype_key: pd.Categorical(
                [model.data.vol.celltype_names[int(c)] for c in layout.cell_type],
                categories=list(model.data.vol.celltype_names),
            ),
        },
        index=[f"gen{i:06d}" for i in range(values.shape[0])],
    )
    if region is not None and model.data.vol.region_names is not None:
        obs[str(cfg.region_key)] = pd.Categorical(
            [model.data.vol.region_names[int(r)] for r in region.numpy()],
            categories=list(model.data.vol.region_names),
        )
    adata = ad.AnnData(
        X=sparse.csr_matrix(values),
        obs=obs,
        var=pd.DataFrame(index=list(model.data.vol.gene_names)),
    )
    # In-plane (u, v) as the coordinate key, because that is what a section reports and what
    # every in-plane metric is computed on; the physical 3-D positions travel beside it,
    # because an oblique plane's cells do not share a depth.
    adata.obsm[cfg.coord_key] = layout.coords_uv.astype(np.float32)
    adata.obsm["xyz"] = layout.coords_xyz.astype(np.float32)
    adata.obsm["spatial_3d"] = layout.coords_xyz.astype(np.float32)
    adata.uns[cfg.thickness_key] = float(layout.plane.thickness)
    adata.uns["generated"] = {
        "expr_mode": cfg.expr_mode,
        "layout_mode": layout.mode,
        "seed": int(layout.seed),
        "n_expected": float(layout.n_expected),
        "specimen_id": model.data.vol.specimen_id,
    }
    adata.uns["plane"] = {
        "origin": np.asarray(plane.origin, dtype=np.float64).tolist(),
        "normal": np.asarray(plane.normal, dtype=np.float64).tolist(),
        "e1": np.asarray(plane.e1, dtype=np.float64).tolist(),
        "e2": np.asarray(plane.e2, dtype=np.float64).tolist(),
        "half_extent": np.asarray(plane.half_extent, dtype=np.float64).tolist(),
        "thickness": float(plane.thickness),
    }
    adata.uns["seed"] = int(seed)
    adata.uns["grf_seed"] = int(model.grf.seed if grf_seed is None else grf_seed)
    adata.uns["config_hash"] = cfg.content_hash()
    # Open risk R3: the ends of the stack are a distinct, worse regime, so the flag travels
    # with the data rather than being left for a caller to infer from z.
    adata.uns["boundary"] = {
        "is_boundary": bool(is_boundary_plane(vol, plane, cfg)),
        "z_min": float(vol.z_values.min()),
        "z_max": float(vol.z_values.max()),
        "margin_um": float(cfg.boundary_margin_spacings) * float(vol.median_spacing),
    }
    adata.uns["retrieval"] = {
        "z_window": float(window),
        "empty_pool_fraction": float(empty_fraction),
    }
    adata.uns["uncertainty"] = (
        {}
        if uncertainty is None
        else {
            "n_samples": int(uncertainty.n_samples),
            "mean_latent_variance": float(np.mean(uncertainty.variance)),
            "median_latent_variance": float(np.median(uncertainty.variance)),
        }
    )
    adata.uns["calibration"] = (
        {} if calibration is None else {"detection_sections": list(calibration.section_ids)}
    )
    return adata


# --------------------------------------------------------------------------------------
# many sections, one realisation
# --------------------------------------------------------------------------------------


def _concatenate(sections: Sequence[Any], planes: Sequence[Plane], grf_seed: int) -> Any:
    """Stack per-section AnnDatas into one, keeping each section's own ``uns`` beside it."""
    import anndata as ad

    for index, (adata, plane) in enumerate(zip(sections, planes, strict=True)):
        adata.obs["section"] = f"gen{index:03d}"
        adata.obs["section"] = adata.obs["section"].astype("category")
        adata.uns["section_index"] = index
        adata.uns["plane_origin_z"] = float(plane.origin[2])
    merged = ad.concat(list(sections), axis=0, label=None, index_unique="-", uns_merge="first")
    merged.uns["stack"] = {
        "grf_seed": int(grf_seed),
        "n_sections": len(sections),
        "z_values": [float(p.origin[2]) for p in planes],
        "coherent": True,
    }
    merged.uns["sections"] = [dict(a.uns["plane"]) for a in sections]
    merged.uns["boundary_flags"] = [bool(a.uns["boundary"]["is_boundary"]) for a in sections]
    return merged


def _resolve_grf_seed(model: CTFFlow, grf_seed: int | None) -> int:
    """Return the realisation a multi-section call shares: the model's own unless given."""
    return int(model.grf.seed) if grf_seed is None else int(grf_seed)


def generate_stack(
    model: CTFFlow,
    z_values: Sequence[float] | npt.NDArray[Any],
    vol: TrainingVolume,
    cfg: Config,
    seed: int,
    *,
    grf_seed: int | None = None,
    calibration: DetectionCalibration | None = None,
    anchor: IsotonicRegressor | None = None,
) -> Any:
    """Generate a dense virtual stack at ``z_values``. Returns one concatenated ``AnnData``.

    **One GRF realisation for the whole call** (``specs/09`` §1's "Do NOT draw a fresh GRF
    per section inside a stack"): the sections are slices of one object, so expression
    correlation between two of them decays smoothly with ``|dz|`` instead of dropping to the
    between-realisation floor at every step. Each section still gets its own layout and count
    seed, derived from ``seed``, because two adjacent slices of one tissue do not have the
    same cells.
    """
    realisation = _resolve_grf_seed(model, grf_seed)
    depths = [float(z) for z in np.asarray(z_values, dtype=np.float64).reshape(-1)]
    if not depths:
        raise GenerationError("generate_stack: z_values is empty; there is no stack to generate")
    planes = [plane_at_z(vol, z, cfg) for z in depths]
    sections = [
        generate_section(
            model,
            plane,
            vol,
            cfg,
            seed + index,
            grf_seed=realisation,
            calibration=calibration,
            anchor=anchor,
        )
        for index, plane in enumerate(planes)
    ]
    return _concatenate(sections, planes, realisation)


def generate_oblique(
    model: CTFFlow,
    normal: npt.ArrayLike,
    n_sections: int,
    vol: TrainingVolume,
    cfg: Config,
    seed: int,
    *,
    grf_seed: int | None = None,
    calibration: DetectionCalibration | None = None,
    anchor: IsotonicRegressor | None = None,
    spacing: float | None = None,
) -> Any:
    """Generate ``n_sections`` parallel sections with an arbitrary ``normal``.

    The series is centred on the volume's own centre and spaced ``spacing`` micrometres apart
    along the normal (``None`` = the volume's median section spacing). Each window is the
    largest centred square that stays inside the bounding box for that orientation
    (:func:`inscribed_half_extent`) — an oblique plane through a flat block is limited by the
    block's depth, and a window sized off the in-plane extent alone would leave the tissue at
    its corners.

    One realisation for the whole call, as in :func:`generate_stack`: that is what makes two
    oblique planes agree along their intersection (experiment E5).
    """
    if n_sections < 1:
        raise GenerationError(f"generate_oblique: n_sections must be >= 1, got {n_sections}")
    realisation = _resolve_grf_seed(model, grf_seed)
    unit = np.asarray(normal, dtype=np.float64).reshape(3)
    unit = unit / float(np.linalg.norm(unit))
    centre = np.asarray(vol.bbox, dtype=np.float64).mean(axis=0)
    step = float(vol.median_spacing) if spacing is None else float(spacing)
    thickness = float(np.median([s.thickness for s in vol.sections]))
    offsets = (np.arange(n_sections, dtype=np.float64) - 0.5 * (n_sections - 1)) * step
    planes = []
    for offset in offsets:
        seed_plane = plane_from_normal(
            unit, centre + offset * unit, np.array([1.0, 1.0]), thickness
        )
        planes.append(
            plane_from_normal(
                unit,
                centre + offset * unit,
                inscribed_half_extent(vol.bbox, seed_plane.e1, seed_plane.e2),
                thickness,
            )
        )
    sections = [
        generate_section(
            model,
            plane,
            vol,
            cfg,
            seed + index,
            grf_seed=realisation,
            calibration=calibration,
            anchor=anchor,
        )
        for index, plane in enumerate(planes)
    ]
    return _concatenate(sections, planes, realisation)


def generate_curved(
    model: CTFFlow,
    surface: Surface,
    vol: TrainingVolume,
    cfg: Config,
    seed: int,
    *,
    grf_seed: int | None = None,
    calibration: DetectionCalibration | None = None,
    anchor: IsotonicRegressor | None = None,
) -> Any:
    """Generate one anatomy-following section on a curved :class:`Surface`.

    The layout is sampled on the surface's **mean plane** and every cell is then lifted onto
    the surface, ``z <- surface.height(x, y)``, before any query of the field, the GRF or the
    retrieval index. So the in-plane point process is the one T05 fitted — density,
    hard core and Potts marks all live in-plane — while every *expression* query happens at
    the cell's true curved position, which is where the curvature has to matter. Bending the
    intensity integral itself would need a surface-aware sampler and is not what
    ``specs/07`` §1's graph surface supports.

    Shares one realisation with any other call given the same ``grf_seed``, exactly as
    :func:`generate_stack` does.
    """
    realisation = _resolve_grf_seed(model, grf_seed)
    flat = plane_at_z(vol, surface.mean_height, cfg, thickness=float(surface.thickness))
    lifted = _CurvedPlane(
        origin=flat.origin,
        e1=flat.e1,
        e2=flat.e2,
        normal=flat.normal,
        half_extent=flat.half_extent,
        thickness=flat.thickness,
        surface=surface,
    )
    return generate_section(
        model,
        lifted,
        vol,
        cfg,
        seed,
        grf_seed=realisation,
        calibration=calibration,
        anchor=anchor,
    )


@dataclass(frozen=True)
class _CurvedPlane(Plane):
    """A plane whose ``(u, v) -> (x, y, z)`` map follows a :class:`Surface`.

    Only :meth:`to_xyz` changes, and that is enough: the layout sampler, the intensity
    integral and every downstream query go through it, so the cells come out on the surface
    while the in-plane geometry stays the mean plane's. Private because a caller should ask
    for :func:`generate_curved` rather than build one.
    """

    surface: Surface = None  # type: ignore[assignment]

    def to_xyz(self, uv: npt.ArrayLike) -> FloatArray:
        """``(N, 2)`` in-plane micrometres -> ``(N, 3)`` points **on the surface**, float64."""
        flat = super().to_xyz(uv)
        height = self.surface.height(flat[:, :2])
        return np.asarray(np.concatenate([flat[:, :2], height[:, None]], axis=1), dtype=np.float64)


def stack_pair_correlations(adata: Any, cfg: Config) -> tuple[IntArray, FloatArray, FloatArray]:
    """Between-section correlation of a generated stack, per pair. ``(P, 2)``, ``(P,)``, ``(P,)``.

    Returns ``(pairs, dz, r)``: which two sections, how far apart in micrometres, and the
    Pearson correlation between their **grid-averaged** log-normalised profiles on the same
    in-plane grid the along-z variogram uses (``Config.variogram_z_grid_size``). Two sections
    have no cell in common, so a correlation between them is only defined between spatial
    averages — the same construction as
    :func:`~spatialcpav25_gen.model.noise.fit_lengthscale_from_sections`'s along-z arm, and
    the observable T09 §2 calibrates ``ell_z`` against (open risk R1).

    The pair indices are returned because "does the correlation spike at a real section's
    depth" is a question about *which* pair, and recovering that from a flat list of lags is
    exactly the kind of index arithmetic a test should not be doing.
    """
    from spatialcpav25_gen.infer.calibrate import section_grid_means

    labels = np.asarray(adata.obs["section"].to_numpy())
    z_values = np.asarray(adata.uns["stack"]["z_values"], dtype=np.float64)
    counts = emitted_counts(adata)
    coords = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
    bounds = np.stack([coords.min(axis=0), coords.max(axis=0)], axis=0)
    unique = sorted(set(labels.tolist()))
    means = [
        section_grid_means(counts[labels == name], coords[labels == name], bounds, cfg)
        for name in unique
    ]
    pairs: list[tuple[int, int]] = []
    lags: list[float] = []
    correlations: list[float] = []
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            shared = means[i][1] & means[j][1]
            if int(shared.sum()) < 3:
                continue
            a = means[i][0][shared].ravel()
            b = means[j][0][shared].ravel()
            if a.std() == 0.0 or b.std() == 0.0:
                continue
            pairs.append((i, j))
            lags.append(abs(float(z_values[j] - z_values[i])))
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return (
        np.asarray(pairs, dtype=np.intp).reshape(-1, 2),
        np.asarray(lags, dtype=np.float64),
        np.asarray(correlations, dtype=np.float64),
    )


def stack_correlation(adata: Any, cfg: Config) -> tuple[FloatArray, FloatArray]:
    """``(dz, r)`` of a generated stack — :func:`stack_pair_correlations` without the pairs."""
    _, lags, correlations = stack_pair_correlations(adata, cfg)
    return lags, correlations


def emitted_counts(adata: Any) -> FloatArray:
    """``(N, G)`` float64 dense counts of a generated section. A convenience, not arithmetic."""
    return cast(FloatArray, np.asarray(adata.X.todense(), dtype=np.float64))
