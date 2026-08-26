"""Data contracts: what a section and a volume are, and what makes one valid.

Three types carry the leakage discipline of ``CLAUDE.md`` in the type system rather than
in review comments:

``Volume``
    A stack of sections from one specimen, plus the quantities derived from it.
``TrainingVolume``
    A :class:`Volume` that has had its held-out sections removed, with ``median_spacing``
    recomputed on what is left. Only this type may reach a loss, a calibrator, or the
    config selector.
``HeldOutSections``
    The evaluation sections. Deliberately *not* a :class:`Volume`: it has no
    ``median_spacing``, no retrieval-index-shaped surface, and nothing that a training
    function accepts. Passing one where a ``TrainingVolume`` is expected is a
    ``TypeError`` at runtime, not a code-review question (SPEC_QUESTIONS A1, required by
    T08's ``test_metric_aware_rejects_heldout``).

Coordinate convention (Convention 4, with the exception documented in T01): a
``Section`` stores in-plane ``(N, 2)`` coordinates in micrometres and holds its depth
separately as a scalar ``z``. :func:`to_xyz` is the only sanctioned way to obtain
``(N, 3)`` ``(x, y, z)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.spatial import cKDTree

from spatialcpav25_gen.config import Config

__all__ = [
    "AssumedThicknessWarning",
    "ConfigClampWarning",
    "HeldOutSections",
    "NonIntegerCountsWarning",
    "OverlappingSlabsWarning",
    "SchemaError",
    "Section",
    "TrainingVolume",
    "Volume",
    "clamp_config_to_volume",
    "median_nn_distance",
    "median_section_spacing",
    "to_xyz",
    "validate_config_against_volume",
    "validate_volume",
]


class CoincidentCoordsWarning(UserWarning):
    """Raised once per volume that carries cells sharing a section-mate's ``(x, y)``."""


class SchemaError(ValueError):
    """Raised when a :class:`Section` or :class:`Volume` violates the data contract."""


class AssumedThicknessWarning(UserWarning):
    """Section thickness was defaulted, not measured.

    Anything quantitative about cells per unit *volume* derived from such a volume - the
    T05 intensity integral, T07's ``L_thick`` - rests on the assumption, and any report or
    figure built from it has to carry the caveat.
    """


class NonIntegerCountsWarning(UserWarning):
    """A matrix declared as raw counts is not integer-valued.

    Almost always means log-normalised data was passed where counts were expected, which
    silently breaks the ZINB decoder (Convention 5).
    """


class OverlappingSlabsWarning(UserWarning):
    """A section is thicker than the spacing between sections, so the slabs overlap.

    The thickness-consistency loss in T07 is ill-posed on overlapping slabs: the same
    tissue is observed twice and the coarse-graining identity no longer holds.
    """


@dataclass
class Section:
    """One physical tissue section.

    Attributes
    ----------
    z
        Depth in micrometres.
    thickness
        Slab thickness in micrometres. Load-bearing: T05 integrates cell intensity over
        the slab *volume* and T07's ``L_thick`` is defined entirely in terms of it.
    coords
        ``(N, 2)`` float32, in-plane ``(x, y)`` micrometres.
    cell_type
        ``(N,)`` int32 codes into ``Volume.celltype_names``.
    region
        ``(N,)`` int32 codes into ``Volume.region_names``, or ``None``.
    counts
        ``(N, G)`` CSR matrix of raw counts (or native intensity).
    section_id
        Stable identifier, unique within a volume.
    thickness_is_assumed
        ``True`` when ``thickness`` was defaulted from the volume's median spacing rather
        than read from the data.
    """

    z: float
    thickness: float
    coords: npt.NDArray[np.float32]
    cell_type: npt.NDArray[np.int32]
    region: npt.NDArray[np.int32] | None
    counts: Any
    section_id: str
    thickness_is_assumed: bool = False

    @property
    def n_cells(self) -> int:
        """Number of cells in this section."""
        return int(self.coords.shape[0])

    @property
    def n_genes(self) -> int:
        """Number of genes (columns of ``counts``)."""
        return int(self.counts.shape[1])


def to_xyz(section: Section) -> npt.NDArray[np.float32]:
    """Return the section's cell coordinates as ``(N, 3)`` float32 ``(x, y, z)``.

    The only sanctioned way to turn a section's ``(N, 2)`` in-plane coordinates plus its
    scalar depth into the ``(N, 3)`` physical coordinates the rest of the system uses.
    """
    xy = np.asarray(section.coords, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise SchemaError(
            f"Section.coords for section_id={section.section_id!r} must be (N, 2); "
            f"got {xy.shape}"
        )
    z = np.full((xy.shape[0], 1), section.z, dtype=np.float32)
    return np.concatenate([xy, z], axis=1, dtype=np.float32)


def median_section_spacing(sections: Sequence[Section]) -> float:
    """Return the median ``|z_{i+1} - z_i|`` over consecutive sections."""
    if len(sections) < 2:
        raise SchemaError(
            "median section spacing needs at least 2 sections, "
            f"got {len(sections)}"
        )
    z = np.asarray([s.z for s in sections], dtype=np.float64)
    return float(np.median(np.abs(np.diff(z))))


def median_nn_distance(sections: Sequence[Section]) -> float:
    """Return the median in-plane nearest-neighbour distance, pooled over sections."""
    distances: list[npt.NDArray[np.float64]] = []
    for section in sections:
        if section.n_cells < 2:
            continue
        tree = cKDTree(np.asarray(section.coords, dtype=np.float64))
        # k=2: the first neighbour of a point is itself.
        nn = tree.query(np.asarray(section.coords, dtype=np.float64), k=2)[0][:, 1]
        distances.append(np.asarray(nn, dtype=np.float64))
    if not distances:
        raise SchemaError("median nearest-neighbour distance needs a section with >= 2 cells")
    return float(np.median(np.concatenate(distances)))


def n_coincident(section: Section) -> int:
    """Cells in ``section`` sharing their ``(x, y)`` with another. ``Section -> int``.

    Counted as ``n_cells - n_unique_rows``, i.e. the number of rows that would have to be
    removed to make the section's coordinates unique.
    """
    coords = np.asarray(section.coords, dtype=np.float64)
    if coords.shape[0] == 0:
        return 0
    return int(coords.shape[0] - np.unique(coords, axis=0).shape[0])


@dataclass
class Volume:
    """A stack of sections from one specimen, plus shared metadata.

    Attributes
    ----------
    sections
        Sections sorted by ``z`` ascending (checked by :func:`validate_volume`).
    gene_names
        Length ``G``; the column order of every section's ``counts``.
    celltype_names
        Names indexed by ``Section.cell_type``.
    region_names
        Names indexed by ``Section.region``, or ``None`` when the dataset has no regions.
    specimen_id
        Identifier of the specimen this stack came from.
    flattened_sections
        ``True`` when every section is a **z-flattened slab** rather than a true plane: the
        cells kept their real ``(x, y)`` and took the slab's centre ``z``. Set from the source
        dataset, never inferred. It is the *only* thing that permits exact coordinate ties —
        see :func:`_validate_section_arrays`.
    n_coincident_coords
        Derived: how many cells share their ``(x, y)`` with another cell in the same section,
        summed over sections. Always computed, so a volume that is *allowed* ties still says
        how many it has and no downstream code has to rediscover it.
    median_spacing
        Derived: median ``|z_{i+1} - z_i|``.
    median_nn_dist
        Derived: median in-plane nearest-neighbour distance, pooled over sections.
    bbox
        Derived: ``(2, 3)`` float32, row 0 = min over ``(x, y, z)``, row 1 = max. The ``z``
        axis spans the sections' **slabs** — ``z +/- thickness/2`` — not their centres, so the
        support is the tissue the model is actually queried about (SPEC_QUESTIONS C33).

    Notes
    -----
    The three derived fields are ``init=False`` and recomputed in ``__post_init__`` so
    they cannot be passed in and go stale (SPEC_QUESTIONS C5). Removing sections means
    building a new ``Volume``, which is what makes ``split_holdout``'s recomputation of
    ``median_spacing`` automatic rather than something to remember.
    """

    sections: list[Section]
    gene_names: list[str]
    celltype_names: list[str]
    region_names: list[str] | None
    specimen_id: str
    flattened_sections: bool = False

    median_spacing: float = field(init=False)
    median_nn_dist: float = field(init=False)
    n_coincident_coords: int = field(init=False)
    bbox: npt.NDArray[np.float32] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compute the derived fields. Raises if the stack is too short to define them."""
        if len(self.sections) < 2:
            raise SchemaError(
                f"Volume(specimen_id={self.specimen_id!r}) has {len(self.sections)} "
                "section(s); at least 2 are needed to define median_spacing"
            )
        self.median_spacing = median_section_spacing(self.sections)
        self.median_nn_dist = median_nn_distance(self.sections)
        self.n_coincident_coords = sum(n_coincident(s) for s in self.sections)
        self.bbox = self._compute_bbox()

    def _compute_bbox(self) -> npt.NDArray[np.float32]:
        """Return ``(2, 3)`` float32 min/max over ``(x, y, z)``.

        **The z axis spans the sections' slabs, not their centres** (SPEC_QUESTIONS C33,
        accepted 2026-08-26). A section is a slab of tissue ``thickness`` deep, and the model
        is asked about every depth inside it: ``CTFFlow._layout_targets`` jitters each cell's
        depth uniformly within its own slab, and ``uniform_slab_points`` draws the Monte-Carlo
        set the Poisson integral is evaluated on from the same interval. Taking the box of the
        section *centres* put the first and last sections' own slabs half outside the support,
        so **half of every boundary section's queries were clamped to the bbox face** — 2067 of
        ``section_1``'s 4073 cells on tier-1 STARmap, ~50 % on the fixture, and ~50 % whatever
        the point set, which is what made it look like a frame bug rather than a geometry one.

        The consequence was not cosmetic: at the two boundary sections the intensity head was
        queried at the tissue's surface instead of at the depth the jitter chose, so the layout
        term's integral was evaluated on a support that is half surface-clamped — exactly the
        regime open risks R3 and R11 are about.

        The in-plane axes are unchanged: ``x`` and ``y`` are real cell coordinates and a cell
        on the face is a real cell there, which is the reason
        ``field._BBOX_TOLERANCE_FRAC`` exists.
        """
        xy = np.concatenate(
            [np.asarray(s.coords, dtype=np.float32) for s in self.sections], axis=0
        )
        z = np.asarray([s.z for s in self.sections], dtype=np.float32)
        half = 0.5 * np.asarray([s.thickness for s in self.sections], dtype=np.float32)
        lo = np.array([xy[:, 0].min(), xy[:, 1].min(), (z - half).min()], dtype=np.float32)
        hi = np.array([xy[:, 0].max(), xy[:, 1].max(), (z + half).max()], dtype=np.float32)
        return np.stack([lo, hi], axis=0, dtype=np.float32)

    @property
    def n_sections(self) -> int:
        """Number of sections in the stack."""
        return len(self.sections)

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return len(self.gene_names)

    @property
    def n_cells(self) -> int:
        """Total number of cells over all sections."""
        return int(sum(s.n_cells for s in self.sections))

    @property
    def z_values(self) -> npt.NDArray[np.float32]:
        """``(n_sections,)`` float32 section depths."""
        return np.asarray([s.z for s in self.sections], dtype=np.float32)

    @property
    def section_ids(self) -> list[str]:
        """Section identifiers, in stack order."""
        return [s.section_id for s in self.sections]

    @property
    def has_assumed_thickness(self) -> bool:
        """True if any section's thickness was defaulted rather than measured."""
        return any(s.thickness_is_assumed for s in self.sections)


class TrainingVolume(Volume):
    """A :class:`Volume` with the held-out sections already removed.

    Produced only by :func:`spatialcpav25_gen.data.loaders.split_holdout`. Metric-aware
    losses (T08), calibration and config selection (T09) take this type and nothing else,
    so held-out data cannot reach them by accident.

    It also carries :attr:`principal_axis`, the axis T08's depth profiles are taken along.
    That field lives *here* and not on :class:`Volume` deliberately (SPEC_QUESTIONS C10):
    computing it on a full volume would consult held-out sections, and computing it per
    epoch would make the profile loss non-stationary and the metric incomparable between
    epochs. A cached property of the training volume is leakage-free by construction and
    cannot drift.
    """

    @cached_property
    def principal_axis(self) -> npt.NDArray[np.float64]:
        """The principal tissue axis: ``(3,)`` float64 unit vector, ``(x, y, z)``.

        The first principal component of this volume's own **in-plane** cell coordinates,
        each section centred on its own centroid, returned with a zero ``z`` component.

        Two choices worth stating, because ``specs/08`` §2 says only "first PC of cell
        coordinates":

        * **In-plane.** Every cell of a section shares one ``z``, so the component of the
          axis along ``z`` contributes a constant to the projection and a depth profile
          taken along it would measure nothing within a section. Pooling raw ``(x, y, z)``
          on a stack that is wider than it is deep also just returns whichever in-plane
          direction happens to be widest, with the ``z`` component along for the ride.
        * **Centred per section.** Otherwise a stack whose sections drift laterally with
          depth turns that drift into the leading component, which is a property of the
          mounting rather than of the tissue.

        The sign is canonicalised (the largest-magnitude component is made positive) so the
        axis does not flip between machines, which would flip every depth profile.

        :meth:`set_anatomical_axis` replaces it with a user-supplied anatomical direction;
        that is ``specs/08`` §2's "or a user-supplied anatomical axis".
        """
        xy = np.concatenate(
            [
                np.asarray(s.coords, dtype=np.float64)
                - np.asarray(s.coords, dtype=np.float64).mean(axis=0, keepdims=True)
                for s in self.sections
            ],
            axis=0,
        )
        _, vectors = np.linalg.eigh(xy.T @ xy / float(xy.shape[0]))
        first = np.asarray(vectors[:, -1], dtype=np.float64)  # eigh sorts ascending
        axis = np.array([first[0], first[1], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:  # pragma: no cover - two coincident columns cannot both be zero
            raise SchemaError(
                f"TrainingVolume.principal_axis: specimen {self.specimen_id!r} has no "
                "in-plane spread, so no principal axis is defined"
            )
        axis = axis / norm
        return axis if axis[int(np.argmax(np.abs(axis)))] > 0 else -axis

    def set_anatomical_axis(self, axis: npt.NDArray[np.float64] | Sequence[float]) -> None:
        """Pin a user-supplied anatomical axis instead of the fitted first PC.

        ``axis`` is any ``(3,)`` non-zero direction in ``(x, y, z)`` micrometre space; it is
        normalised here. Raises if :attr:`principal_axis` has already been read, because an
        axis that changes mid-run is exactly the drift ``specs/08`` forbids ("do not
        recompute the principal tissue axis per epoch") - only louder.
        """
        if "principal_axis" in self.__dict__:
            raise SchemaError(
                "TrainingVolume.set_anatomical_axis: the principal axis has already been "
                "used, and changing it now would make the depth profiles before and after "
                "incomparable. Set it before the first metric-aware step."
            )
        vector = np.asarray(axis, dtype=np.float64).reshape(-1)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise SchemaError(
                "TrainingVolume.set_anatomical_axis: axis must be a finite (3,) vector in "
                f"(x, y, z), got {np.asarray(axis).shape}"
            )
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise SchemaError("TrainingVolume.set_anatomical_axis: axis must be non-zero")
        self.__dict__["principal_axis"] = vector / norm


@dataclass
class HeldOutSections(Sequence[Section]):
    """The evaluation sections, in a container that is deliberately not a Volume.

    It exposes the sequence protocol (so ``len``, iteration and indexing read like the
    ``list[Section]`` T01 describes) and the metadata an evaluator needs, but none of the
    derived quantities a model could learn from - in particular not ``median_spacing``,
    which would leak the true spacing of the holdout.
    """

    sections: list[Section]
    gene_names: list[str]
    celltype_names: list[str]
    region_names: list[str] | None
    specimen_id: str

    def __len__(self) -> int:
        return len(self.sections)

    def __iter__(self) -> Iterator[Section]:
        return iter(self.sections)

    def __getitem__(self, index: Any) -> Any:
        return self.sections[index]

    @property
    def section_ids(self) -> list[str]:
        """Identifiers of the held-out sections."""
        return [s.section_id for s in self.sections]

    @property
    def z_values(self) -> npt.NDArray[np.float32]:
        """``(n_heldout,)`` float32 depths of the held-out sections."""
        return np.asarray([s.z for s in self.sections], dtype=np.float32)


# --------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------


def validate_volume(vol: Volume, cfg: Config) -> None:
    """Raise :class:`SchemaError` if ``vol`` violates the data contract.

    Checks, in order: section count and ordering; gene-set consistency; coordinates;
    counts; label codes; per-section cell counts; thickness. Every message names the
    offending field and, where the value came from a file, the config key it was read
    from (Convention 6).

    Warnings, not errors, are issued for: non-integer values in a matrix declared as raw
    counts (:class:`NonIntegerCountsWarning`) and slabs thicker than the section spacing
    (:class:`OverlappingSlabsWarning`).
    """
    _validate_sections_ordered(vol, cfg)
    _validate_gene_axis(vol)
    for section in vol.sections:
        _validate_section_arrays(section, vol, cfg)
    _validate_counts(vol, cfg)
    _validate_labels(vol)
    _validate_thickness(vol)

    if vol.flattened_sections and vol.n_coincident_coords:
        warnings.warn(
            f"Volume(specimen_id={vol.specimen_id!r}) is flagged as z-flattened serial "
            f"sections and contains {vol.n_coincident_coords} cell(s) coincident with a "
            f"section-mate ({100.0 * vol.n_coincident_coords / max(vol.n_cells, 1):.2f}% of "
            f"{vol.n_cells}). Exact coordinate ties are permitted for this volume and the "
            f"count is on Volume.n_coincident_coords. They are real: a slab collapsed to its "
            f"centre z makes two cells at the same (x, y) in different planes coincident, so "
            f"any kNN-graph quantity — Moran's I, Geary's C, the layout's neighbourhood terms "
            f"— sees a zero-distance pair there.",
            CoincidentCoordsWarning,
            stacklevel=2,
        )

def _validate_sections_ordered(vol: Volume, cfg: Config) -> None:
    """Check the section count, unique ids, and strictly increasing z."""
    if vol.n_sections < cfg.min_sections_per_volume:
        raise SchemaError(
            f"Volume.sections: specimen {vol.specimen_id!r} has {vol.n_sections} sections, "
            f"fewer than Config.min_sections_per_volume={cfg.min_sections_per_volume}"
        )
    ids = vol.section_ids
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise SchemaError(f"Section.section_id: duplicate section ids {duplicates}")
    z = vol.z_values.astype(np.float64)
    if not np.all(np.isfinite(z)):
        raise SchemaError(
            f"Section.z: non-finite depth in specimen {vol.specimen_id!r} "
            f"(read from Config.z_key={cfg.z_key!r})"
        )
    bad = np.nonzero(np.diff(z) <= 0)[0]
    if bad.size:
        i = int(bad[0])
        raise SchemaError(
            "Volume.sections must be sorted by Section.z, strictly increasing; "
            f"section {ids[i]!r} (z={z[i]}) is followed by {ids[i + 1]!r} (z={z[i + 1]}). "
            f"Depths were read from Config.z_key={cfg.z_key!r}"
        )


def _validate_gene_axis(vol: Volume) -> None:
    """Check that the gene names are unique and that every section shares them."""
    if len(set(vol.gene_names)) != len(vol.gene_names):
        raise SchemaError("Volume.gene_names contains duplicates; the gene axis must be unique")
    n_genes = len(vol.gene_names)
    for section in vol.sections:
        if section.n_genes != n_genes:
            raise SchemaError(
                f"Section.counts for section_id={section.section_id!r} has "
                f"{section.n_genes} columns but Volume.gene_names has {n_genes}; "
                "the gene set and its order must be identical across sections"
            )


def _validate_section_arrays(section: Section, vol: Volume, cfg: Config) -> None:
    """Check one section's coordinates and the agreement of its array lengths."""
    coords = np.asarray(section.coords)
    sid = section.section_id
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise SchemaError(
            f"Section.coords for section_id={sid!r} must be (N, 2) in-plane micrometres "
            f"(read from Config.coord_key={cfg.coord_key!r}); got shape {coords.shape}"
        )
    if coords.dtype != np.float32:
        raise SchemaError(
            f"Section.coords for section_id={sid!r} must be float32 "
            f"(Config.coord_key={cfg.coord_key!r}); got {coords.dtype}"
        )
    if not np.all(np.isfinite(coords)):
        n_bad = int((~np.isfinite(coords)).any(axis=1).sum())
        raise SchemaError(
            f"Section.coords for section_id={sid!r} has {n_bad} non-finite row(s) "
            f"(Config.coord_key={cfg.coord_key!r})"
        )
    if section.n_cells < cfg.min_cells_per_section:
        raise SchemaError(
            f"Section {sid!r} has {section.n_cells} cells, fewer than "
            f"Config.min_cells_per_section={cfg.min_cells_per_section}"
        )
    unique_rows = np.unique(coords, axis=0)
    if unique_rows.shape[0] != coords.shape[0] and not vol.flattened_sections:
        n_dup = coords.shape[0] - unique_rows.shape[0]
        raise SchemaError(
            f"Section.coords for section_id={sid!r} contains {n_dup} duplicate coordinate "
            f"row(s) (Config.coord_key={cfg.coord_key!r}); coincident cells corrupt every "
            "kNN-graph metric downstream. If these are z-flattened serial sections — a slab "
            "collapsed to its centre z, so two cells at the same (x, y) in different planes "
            "become coincident — set Volume.flattened_sections, which permits exact ties and "
            "records the count in Volume.n_coincident_coords"
        )
    n = section.n_cells
    if section.counts.shape[0] != n:
        raise SchemaError(
            f"Section.counts for section_id={sid!r} has {section.counts.shape[0]} rows but "
            f"Section.coords has {n}"
        )
    if section.cell_type.shape != (n,):
        raise SchemaError(
            f"Section.cell_type for section_id={sid!r} must be ({n},) "
            f"(Config.celltype_key={cfg.celltype_key!r}); got {section.cell_type.shape}"
        )
    if section.region is not None and section.region.shape != (n,):
        raise SchemaError(
            f"Section.region for section_id={sid!r} must be ({n},) "
            f"(Config.region_key={cfg.region_key!r}); got {section.region.shape}"
        )
    if (section.region is None) != (vol.region_names is None):
        raise SchemaError(
            f"Section.region for section_id={sid!r} is "
            f"{'absent' if section.region is None else 'present'} but Volume.region_names is "
            f"{'absent' if vol.region_names is None else 'present'}; they must agree "
            f"(Config.region_key={cfg.region_key!r})"
        )


def _validate_counts(vol: Volume, cfg: Config) -> None:
    """Check that counts are finite and non-negative; warn if they are not integers."""
    non_integer: list[str] = []
    for section in vol.sections:
        data = np.asarray(section.counts.data if sparse.issparse(section.counts) else section.counts)
        sid = section.section_id
        if data.size == 0:
            continue
        if not np.all(np.isfinite(data)):
            raise SchemaError(
                f"Section.counts for section_id={sid!r} contains non-finite values "
                f"(read from Config.counts_layer={cfg.counts_layer!r})"
            )
        if data.min() < 0:
            raise SchemaError(
                f"Section.counts for section_id={sid!r} contains negative values "
                f"(min={data.min()}); expression must be raw counts or native intensity "
                f"(Config.counts_layer={cfg.counts_layer!r})"
            )
        if not np.array_equal(data, np.round(data)):
            non_integer.append(sid)

    if non_integer and cfg.counts_layer is not None:
        warnings.warn(
            f"Config.counts_layer={cfg.counts_layer!r} declares raw counts, but "
            f"{len(non_integer)} section(s) hold non-integer values "
            f"(first: {non_integer[0]!r}). This is what log-normalised data looks like, "
            "and it silently breaks the ZINB decoder: normalised values must never reach "
            "the decoder target.",
            NonIntegerCountsWarning,
            stacklevel=2,
        )


def _validate_labels(vol: Volume) -> None:
    """Check that cell type and region codes index into their name lists."""
    for section in vol.sections:
        sid = section.section_id
        _check_codes(section.cell_type, len(vol.celltype_names), "cell_type", sid)
        if section.region is not None and vol.region_names is not None:
            _check_codes(section.region, len(vol.region_names), "region", sid)


def _check_codes(codes: npt.NDArray[np.int32], n_names: int, name: str, sid: str) -> None:
    """Raise if any code falls outside ``[0, n_names)``."""
    if codes.size == 0:
        return
    lo = int(codes.min())
    hi = int(codes.max())
    if lo < 0 or hi >= n_names:
        raise SchemaError(
            f"Section.{name} for section_id={sid!r} has codes in [{lo}, {hi}] but "
            f"Volume.{name}_names has {n_names} entries"
        )


def _validate_thickness(vol: Volume) -> None:
    """Check that every thickness is positive; warn on overlapping slabs."""
    overlapping: list[str] = []
    for section in vol.sections:
        if not np.isfinite(section.thickness) or section.thickness <= 0:
            raise SchemaError(
                f"Section.thickness for section_id={section.section_id!r} must be > 0, got "
                f"{section.thickness!r}"
            )
        if section.thickness > vol.median_spacing:
            overlapping.append(section.section_id)
    if overlapping:
        warnings.warn(
            f"{len(overlapping)} section(s) of specimen {vol.specimen_id!r} are thicker than "
            f"the median spacing ({vol.median_spacing:g} um), so the slabs overlap "
            f"(first: {overlapping[0]!r}). T07's thickness-consistency loss is ill-posed on "
            "overlapping slabs.",
            OverlappingSlabsWarning,
            stacklevel=2,
        )


class ConfigClampWarning(UserWarning):
    """A config field was narrowed to fit the volume, and the run is not at the default."""


def clamp_config_to_volume(cfg: Config, vol: Volume) -> Config:
    """Narrow the volume-dependent width fields to what this panel can support.

    The owed fix ``specs/10`` §0 records and ``specs/11``'s coverage matrix carries: the
    protocol dataset **cannot be fitted at the shipped default**, because STARmap's panel is
    28 genes wide and ``Config.expr_pca_dim`` is 32, so
    :func:`validate_config_against_volume` refuses every fit. The pilot worked around it with
    a hand-written value; this is the rule the spec asked for instead — *clamp to the panel
    width*.

    Two fields, both hard constraints rather than preferences, both narrowed to the volume's
    own size and never widened:

    * ``expr_pca_dim`` -> ``min(expr_pca_dim, vol.n_genes)``. On a narrow panel the result is
      a full-rank rotation rather than a reduction. That is what "the top 32 PCs" degrades to
      when there are fewer than 32 directions, and it is the honest degradation: dropping
      components would be a *choice* about which of this panel's variance to discard, and the
      spec does not make one. (``reports/pilot.md``'s 16 was such a choice, taken by hand and
      recorded as a stand-in; it is not this rule, and a run that wants to be comparable with
      the pilot's numbers has to set it explicitly.)
    * ``retrieval_k`` -> ``min(retrieval_k, vol.n_cells)``.

    Every narrowing warns with both numbers, because the clamped value lands in
    :meth:`Config.content_hash` and a reader comparing two runs is entitled to know which
    fields the data chose rather than the config.

    Returns ``cfg`` itself when nothing needed narrowing, so the common case adds no hash
    churn.
    """
    changes: dict[str, int] = {}
    if cfg.expr_pca_dim > vol.n_genes:
        changes["expr_pca_dim"] = int(vol.n_genes)
    if cfg.retrieval_k > vol.n_cells:
        changes["retrieval_k"] = int(vol.n_cells)
    if not changes:
        return cfg
    for name, value in changes.items():
        warnings.warn(
            f"clamp_config_to_volume: Config.{name}={getattr(cfg, name)} exceeds what "
            f"specimen {vol.specimen_id!r} supports; narrowed to {value}. This changes the "
            "config's content hash, so two runs on panels of different widths are not the "
            "same configuration.",
            ConfigClampWarning,
            stacklevel=2,
        )
    return cfg.replace(**changes)


def validate_config_against_volume(cfg: Config, vol: Volume) -> None:
    """Raise :class:`ConfigError` for config settings this volume cannot support.

    The data-dependent half of :meth:`Config.validate`, which is standalone and has no
    volume to check against (SPEC_QUESTIONS A5). Called by ``load_volume``.
    """
    from spatialcpav25_gen.config import ConfigError

    if (
        vol.n_sections < cfg.small_volume_n_sections
        and cfg.fourier_bands_z > cfg.max_fourier_bands_z_small_volume
    ):
        raise ConfigError(
            f"Config.fourier_bands_z={cfg.fourier_bands_z} exceeds "
            f"max_fourier_bands_z_small_volume={cfg.max_fourier_bands_z_small_volume} on a "
            f"volume with {vol.n_sections} sections (< small_volume_n_sections="
            f"{cfg.small_volume_n_sections}); those z frequencies are unconstrained by so "
            "few sections and will overfit the section positions"
        )
    if cfg.retrieval_k > vol.n_cells:
        raise ConfigError(
            f"Config.retrieval_k={cfg.retrieval_k} exceeds the {vol.n_cells} cells in "
            f"specimen {vol.specimen_id!r}"
        )
    if cfg.expr_pca_dim > vol.n_genes:
        raise ConfigError(
            f"Config.expr_pca_dim={cfg.expr_pca_dim} exceeds the {vol.n_genes} genes in "
            f"specimen {vol.specimen_id!r}"
        )
