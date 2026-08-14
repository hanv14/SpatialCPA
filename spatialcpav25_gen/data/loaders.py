"""Loading volumes from AnnData, and splitting them without leaking held-out sections.

``split_holdout`` is the only producer of :class:`~spatialcpav25_gen.data.schema.TrainingVolume`
and :class:`~spatialcpav25_gen.data.schema.HeldOutSections`. Everything downstream that
must not see evaluation data takes the former type, so the leakage discipline is carried
by the types rather than by convention.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import (
    AssumedThicknessWarning,
    HeldOutSections,
    SchemaError,
    Section,
    TrainingVolume,
    Volume,
    validate_config_against_volume,
    validate_volume,
)

__all__ = ["load_volume", "loso_folds", "split_holdout"]


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def load_volume(
    h5ad_path: str | Path,
    cfg: Config,
    specimen_key: str | None = None,
) -> Volume | list[Volume]:
    """Load one ``.h5ad`` into a validated :class:`Volume`.

    Parameters
    ----------
    h5ad_path
        Path to the AnnData file.
    cfg
        Supplies every key this function reads: ``coord_key``, ``z_key``,
        ``celltype_key``, ``region_key``, ``counts_layer``, ``thickness_key``,
        ``section_key``.
    specimen_key
        ``adata.obs`` column separating specimens. When given, one :class:`Volume` per
        specimen is returned, in sorted specimen order; otherwise a single
        :class:`Volume`.

    Notes
    -----
    Missing required keys raise :class:`~spatialcpav25_gen.data.schema.SchemaError`
    naming both the dataclass field and the AnnData key it was expected at; nothing is
    silently defaulted (Convention 6). The one defaulted quantity is section thickness,
    which falls back to the volume's median spacing, sets ``thickness_is_assumed`` and
    warns exactly once per volume.
    """
    import anndata as ad  # imported here: loading is the only place AnnData is needed

    path = Path(h5ad_path)
    if not path.exists():
        raise FileNotFoundError(f"load_volume: no such file: {path}")
    adata = ad.read_h5ad(path)

    if specimen_key is None:
        return _build_volume(adata, cfg, specimen_id=path.stem)

    if specimen_key not in adata.obs:
        raise SchemaError(
            f"load_volume(specimen_key={specimen_key!r}): no such column in adata.obs; "
            f"available columns: {sorted(map(str, adata.obs.columns))}"
        )
    labels = np.asarray(adata.obs[specimen_key].to_numpy(), dtype=object)
    volumes: list[Volume] = []
    for specimen in sorted({str(v) for v in labels}):
        mask = np.asarray([str(v) == specimen for v in labels])
        volumes.append(_build_volume(adata[mask], cfg, specimen_id=specimen))
    return volumes


def _build_volume(adata: Any, cfg: Config, specimen_id: str) -> Volume:
    """Assemble and validate one :class:`Volume` from an AnnData (or a view of one)."""
    xy, z_per_cell = _read_coords(adata, cfg)
    cell_type, celltype_names = _read_codes(adata, cfg.celltype_key, "celltype_key")
    region, region_names = _read_optional_region(adata, cfg)
    counts = _read_counts(adata, cfg)
    gene_names = [str(g) for g in adata.var_names]

    section_labels = _section_labels(adata, cfg, z_per_cell)
    order = _section_order(section_labels, z_per_cell)

    thickness_by_section = _read_thickness(adata, cfg, section_labels)

    sections: list[Section] = []
    for label in order:
        mask = section_labels == label
        z_values = np.unique(z_per_cell[mask])
        if z_values.size != 1:
            raise SchemaError(
                f"Section.z: section {label!r} spans {z_values.size} distinct depths "
                f"(Config.z_key={cfg.z_key!r}); a section is one depth by definition"
            )
        thickness = thickness_by_section.get(label)
        sections.append(
            Section(
                z=float(z_values[0]),
                # placeholder; replaced below by median_spacing when it is not measured
                thickness=float("nan") if thickness is None else float(thickness),
                coords=np.ascontiguousarray(xy[mask], dtype=np.float32),
                cell_type=np.ascontiguousarray(cell_type[mask], dtype=np.int32),
                region=None if region is None else np.ascontiguousarray(region[mask], np.int32),
                counts=counts[mask].tocsr(),
                section_id=str(label),
                thickness_is_assumed=thickness is None,
            )
        )

    vol = Volume(
        sections=sections,
        gene_names=gene_names,
        celltype_names=celltype_names,
        region_names=region_names,
        specimen_id=specimen_id,
    )
    _apply_assumed_thickness(vol, cfg)
    validate_volume(vol, cfg)
    validate_config_against_volume(cfg, vol)
    return vol


def _read_coords(adata: Any, cfg: Config) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]]:
    """Return ``((N, 2)`` in-plane coordinates, ``(N,)`` per-cell depth)."""
    if cfg.coord_key not in adata.obsm:
        raise SchemaError(
            f"Section.coords: adata.obsm[{cfg.coord_key!r}] is missing "
            f"(Config.coord_key); available keys: {sorted(map(str, adata.obsm.keys()))}"
        )
    coords = np.asarray(adata.obsm[cfg.coord_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise SchemaError(
            f"Section.coords: adata.obsm[{cfg.coord_key!r}] must be (N, 2) or (N, 3) "
            f"micrometres; got shape {coords.shape}"
        )
    if coords.shape[1] == 3:
        return coords[:, :2].astype(np.float32), coords[:, 2]
    if cfg.z_key not in adata.obs:
        raise SchemaError(
            f"Section.z: adata.obsm[{cfg.coord_key!r}] is 2-D, so depth must come from "
            f"adata.obs[{cfg.z_key!r}] (Config.z_key), which is missing; available "
            f"columns: {sorted(map(str, adata.obs.columns))}"
        )
    z = np.asarray(adata.obs[cfg.z_key].to_numpy(), dtype=np.float64)
    return coords.astype(np.float32), z


def _read_codes(adata: Any, key: str, cfg_field: str) -> tuple[npt.NDArray[np.int32], list[str]]:
    """Return ``((N,)`` int32 codes, names) for a categorical ``adata.obs`` column."""
    if key not in adata.obs:
        raise SchemaError(
            f"adata.obs[{key!r}] is missing (Config.{cfg_field}); available columns: "
            f"{sorted(map(str, adata.obs.columns))}"
        )
    column = adata.obs[key]
    # `.array` on a categorical Series is the pandas Categorical itself; anything else is
    # categorised here, which orders the categories lexically and so deterministically.
    categorical = (
        column.array if isinstance(column.dtype, pd.CategoricalDtype) else pd.Categorical(column)
    )
    codes = np.asarray(categorical.codes, dtype=np.int32)
    names = [str(c) for c in categorical.categories]
    if codes.size and int(codes.min()) < 0:
        raise SchemaError(
            f"adata.obs[{key!r}] (Config.{cfg_field}) contains missing values; every cell "
            "must carry a label"
        )
    return codes, names


def _read_optional_region(
    adata: Any, cfg: Config
) -> tuple[npt.NDArray[np.int32] | None, list[str] | None]:
    """Return region codes and names, or ``(None, None)`` when ``region_key`` is None."""
    if cfg.region_key is None:
        return None, None
    if cfg.region_key not in adata.obs:
        raise SchemaError(
            f"Section.region: adata.obs[{cfg.region_key!r}] is missing "
            "(Config.region_key). Set Config.region_key=None for a dataset without "
            "regions rather than leaving a key that does not resolve"
        )
    codes, names = _read_codes(adata, cfg.region_key, "region_key")
    return codes, names


def _read_counts(adata: Any, cfg: Config) -> Any:
    """Return the ``(N, G)`` expression matrix as CSR, from the configured location."""
    if cfg.counts_layer is None:
        matrix = adata.X
        source = "adata.X"
    else:
        if cfg.counts_layer not in adata.layers:
            raise SchemaError(
                f"Section.counts: adata.layers[{cfg.counts_layer!r}] is missing "
                f"(Config.counts_layer); available layers: "
                f"{sorted(map(str, adata.layers.keys()))}"
            )
        matrix = adata.layers[cfg.counts_layer]
        source = f"adata.layers[{cfg.counts_layer!r}]"
    if matrix is None:
        raise SchemaError(f"Section.counts: {source} is None (Config.counts_layer)")
    return sparse.csr_matrix(matrix)


def _section_labels(adata: Any, cfg: Config, z_per_cell: npt.NDArray[np.float64]) -> npt.NDArray[Any]:
    """Return the ``(N,)`` per-cell section label."""
    if cfg.section_key is None:
        return z_per_cell
    if cfg.section_key not in adata.obs:
        raise SchemaError(
            f"Section.section_id: adata.obs[{cfg.section_key!r}] is missing "
            f"(Config.section_key); available columns: {sorted(map(str, adata.obs.columns))}"
        )
    return np.asarray([str(v) for v in adata.obs[cfg.section_key].to_numpy()], dtype=object)


def _section_order(
    section_labels: npt.NDArray[Any], z_per_cell: npt.NDArray[np.float64]
) -> list[Any]:
    """Return the distinct section labels ordered by ascending depth."""
    labels = list(dict.fromkeys(section_labels.tolist()))
    depths = {label: float(np.min(z_per_cell[section_labels == label])) for label in labels}
    return sorted(labels, key=lambda label: depths[label])


def _read_thickness(
    adata: Any, cfg: Config, section_labels: npt.NDArray[Any]
) -> dict[Any, float]:
    """Return measured thickness per section label; empty when the dataset has none.

    Looked up in ``adata.uns[thickness_key]`` (a scalar, or a mapping from section label
    to value) and then in ``adata.obs[thickness_key]`` (a per-cell column, constant
    within a section).
    """
    key = cfg.thickness_key
    if key in adata.uns:
        value = adata.uns[key]
        if isinstance(value, Mapping):
            return {label: float(value[label]) for label in np.unique(section_labels) if label in value}
        scalar = np.asarray(value).reshape(-1)
        if scalar.size != 1:
            raise SchemaError(
                f"Section.thickness: adata.uns[{key!r}] (Config.thickness_key) must be a "
                f"scalar or a mapping from section label to thickness; got {value!r}"
            )
        return {label: float(scalar[0]) for label in np.unique(section_labels)}

    if key in adata.obs:
        column = np.asarray(adata.obs[key].to_numpy(), dtype=np.float64)
        out: dict[Any, float] = {}
        for label in np.unique(section_labels):
            values = np.unique(column[section_labels == label])
            if values.size != 1:
                raise SchemaError(
                    f"Section.thickness: adata.obs[{key!r}] (Config.thickness_key) holds "
                    f"{values.size} distinct values within section {label!r}; a section has "
                    "one thickness"
                )
            out[label] = float(values[0])
        return out

    return {}


def _apply_assumed_thickness(vol: Volume, cfg: Config) -> None:
    """Fill in missing thicknesses from the median spacing, warning once per volume."""
    assumed = [s for s in vol.sections if s.thickness_is_assumed]
    if not assumed:
        return
    for section in assumed:
        section.thickness = float(vol.median_spacing)
    warnings.warn(
        f"Section.thickness is not recorded for {len(assumed)} of {vol.n_sections} sections "
        f"of specimen {vol.specimen_id!r} (looked for adata.uns[{cfg.thickness_key!r}] and "
        f"adata.obs[{cfg.thickness_key!r}], Config.thickness_key). Defaulting to the median "
        f"spacing, {vol.median_spacing:g} um, and flagging Section.thickness_is_assumed. "
        "Any figure or report of cell density per unit volume built from this volume must "
        "carry that caveat.",
        AssumedThicknessWarning,
        stacklevel=3,
    )


# --------------------------------------------------------------------------------------
# splitting
# --------------------------------------------------------------------------------------


def split_holdout(
    vol: Volume,
    mode: str,
    fold: int,
    cfg: Config | None = None,
) -> tuple[TrainingVolume, HeldOutSections]:
    """Split a volume into a training volume and the sections held out for evaluation.

    Parameters
    ----------
    vol
        The full volume.
    mode
        ``"alternating"`` holds out every other *interior* section, ``fold`` selecting the
        parity, so each held-out section has a real neighbour about one median spacing
        away on each side. ``"consecutive"`` holds out a contiguous run of
        ``cfg.holdout_consecutive_k`` sections, ``fold`` selecting which run, so the gap to
        the nearest real section is up to ``(k + 1) / 2`` spacings.
    fold
        Which split of the chosen mode. Out-of-range folds raise rather than wrapping.
    cfg
        Supplies ``holdout_consecutive_k`` and ``min_sections_per_volume``. Defaults to
        ``Config()``. (Additive keyword; T01 writes the signature without it, and the run
        length has to come from somewhere - see SPEC_QUESTIONS A4.)

    Returns
    -------
    tuple
        The :class:`TrainingVolume` - a new object, so ``median_spacing`` and
        ``median_nn_dist`` are recomputed on the *training* sections and the true spacing
        of the holdout is never visible to the model - and the
        :class:`HeldOutSections`.

    Notes
    -----
    The first and last section are never held out: with nothing beyond them there is no
    flanking pair, and the task becomes extrapolation rather than the interpolation the
    benchmark measures.
    """
    cfg = Config() if cfg is None else cfg
    n = vol.n_sections
    if fold < 0:
        raise ValueError(f"split_holdout: fold must be >= 0, got {fold}")

    if mode == "alternating":
        held_idx = [i for i in range(1, n - 1) if i % 2 == fold % 2]
        if not held_idx:
            raise ValueError(
                f"split_holdout(mode='alternating', fold={fold}): no interior section has "
                f"parity {fold % 2} in a volume of {n} sections"
            )
    elif mode == "consecutive":
        k = cfg.holdout_consecutive_k
        starts = list(range(1, n - k))
        if not starts:
            raise ValueError(
                f"split_holdout(mode='consecutive'): a run of "
                f"Config.holdout_consecutive_k={k} sections plus a flanking section on each "
                f"side does not fit in a volume of {n} sections"
            )
        if fold >= len(starts):
            raise ValueError(
                f"split_holdout(mode='consecutive', fold={fold}): only {len(starts)} folds "
                f"exist for k={k} in a volume of {n} sections"
            )
        held_idx = list(range(starts[fold], starts[fold] + k))
    else:
        raise ValueError(
            f"split_holdout: mode must be 'alternating' or 'consecutive', got {mode!r}"
        )

    held_set = set(held_idx)
    train_sections = [s for i, s in enumerate(vol.sections) if i not in held_set]
    if len(train_sections) < cfg.min_sections_per_volume:
        raise ValueError(
            f"split_holdout(mode={mode!r}, fold={fold}) would leave {len(train_sections)} "
            f"training sections, fewer than Config.min_sections_per_volume="
            f"{cfg.min_sections_per_volume}"
        )

    training = TrainingVolume(
        sections=train_sections,
        gene_names=list(vol.gene_names),
        celltype_names=list(vol.celltype_names),
        region_names=None if vol.region_names is None else list(vol.region_names),
        specimen_id=vol.specimen_id,
    )
    heldout = HeldOutSections(
        sections=[vol.sections[i] for i in held_idx],
        gene_names=list(vol.gene_names),
        celltype_names=list(vol.celltype_names),
        region_names=None if vol.region_names is None else list(vol.region_names),
        specimen_id=vol.specimen_id,
    )
    return training, heldout


def loso_folds(vol: TrainingVolume) -> Iterator[tuple[TrainingVolume, Section]]:
    """Iterate leave-one-section-out folds over the **training** sections only.

    Yields ``(rest, hidden)`` for each section in turn, ``rest`` being a fresh
    :class:`TrainingVolume` whose derived quantities are recomputed without the hidden
    section.

    Raises
    ------
    TypeError
        If ``vol`` is not a :class:`TrainingVolume`. A plain :class:`Volume` may still
        contain evaluation sections, and
        :class:`~spatialcpav25_gen.data.schema.HeldOutSections` is nothing but evaluation
        sections; only ``split_holdout`` produces the type this accepts, so held-out data
        cannot reach the metric-aware losses (T08) or the calibrator (T09) through here.
    """
    if not isinstance(vol, TrainingVolume):
        raise TypeError(
            "loso_folds requires a TrainingVolume produced by split_holdout, got "
            f"{type(vol).__name__}: leave-one-section-out must run over training sections "
            "only"
        )
    n = vol.n_sections
    for i in range(n):
        rest = [s for j, s in enumerate(vol.sections) if j != i]
        yield (
            TrainingVolume(
                sections=rest,
                gene_names=list(vol.gene_names),
                celltype_names=list(vol.celltype_names),
                region_names=None if vol.region_names is None else list(vol.region_names),
                specimen_id=vol.specimen_id,
            ),
            vol.sections[i],
        )
