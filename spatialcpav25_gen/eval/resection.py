"""Re-slice or re-partition a built volume's point cloud along a chosen normal (``specs/10`` §9).

Two operations, one mechanism:

* :func:`repartition` cuts the same block into a different number of slabs along the **same**
  normal — ``specs/10`` §8's V4a, where slab thickness is the only variable.
* :func:`resection` cuts along a **different** normal, producing genuine orthogonal sections with
  real ground truth.

Both emit an ordinary bench3 ``data.h5ad`` under a **new dataset id** and never write to the source
(``specs/10`` §9). The source's SHA-256 is asserted unchanged.

**The precondition that decides whether either is possible at all.** Re-cutting needs the cells'
*own* depth along the normal. A built bench3 volume may not have it: ``load_volume`` raises unless
every section spans exactly **one** distinct depth (``loaders.py``: *"a section is one depth by
definition"*), so a file that loads at all has already quantised depth to one value per section. Cut
such a file into more slabs than it has sections and the extra slabs come out **empty** — a result
that looks like a measurement and is an artefact. :func:`plan_partition` refuses it by counting
distinct source values, and the refusal names the count.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["PartitionPlan", "ResectionError", "plan_partition", "repartition", "resection"]

# bench3's own floor: a section with fewer cells than this is not scored.
MIN_CELLS_PER_SECTION = 50


class ResectionError(RuntimeError):
    """A re-slice or re-partition that would produce an artefact rather than a measurement."""


@dataclass(frozen=True)
class PartitionPlan:
    """How a point cloud divides along one axis. Pure geometry; no AnnData, no I/O.

    Attributes
    ----------
    labels
        ``(N,)`` int64 slab index per cell, in ``[0, n_sections)``.
    edges
        ``(n_sections + 1,)`` float64 bin edges along the projection axis, micrometres.
    counts
        ``(n_sections,)`` int64 cells per slab.
    centres
        ``(n_sections,)`` float64 slab centres — each becomes a ``Section.z``.
    n_distinct_source
        How many distinct projected depths the source had. The number the refusal turns on.
    """

    labels: npt.NDArray[np.int64]
    edges: npt.NDArray[np.float64]
    counts: npt.NDArray[np.int64]
    centres: npt.NDArray[np.float64]
    n_distinct_source: int


def plan_partition(
    xyz: npt.NDArray[Any], normal: npt.NDArray[Any] | tuple[float, float, float], n_sections: int
) -> PartitionPlan:
    """Divide ``(N, 3)`` µm coordinates into ``n_sections`` equal-width slabs along ``normal``.

    Parameters
    ----------
    xyz
        ``(N, 3)`` float coordinates in micrometres.
    normal
        ``(3,)`` slab normal; normalised here, so its magnitude is irrelevant.
    n_sections
        How many slabs to cut. Must be at least 2.

    Returns
    -------
    PartitionPlan
        ``labels`` ``(N,)``, ``edges`` ``(n_sections + 1,)``, ``counts`` and ``centres``
        ``(n_sections,)``.

    Raises
    ------
    ResectionError
        If the source has no more distinct depths than the requested slab count — the flattened
        case, where the extra slabs would be empty — or if any slab would fall under bench3's
        50-cell floor.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ResectionError(f"plan_partition: xyz must be (N, 3) micrometres; got {pts.shape}")
    n = int(n_sections)
    if n < 2:
        raise ResectionError(f"plan_partition: n_sections must be at least 2; got {n}")

    unit = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    if not norm > 0:
        raise ResectionError("plan_partition: normal must be non-zero")
    depth = pts @ (unit / norm)

    distinct = int(np.unique(depth).size)
    if distinct <= n:
        raise ResectionError(
            f"plan_partition: the source has only {distinct} distinct depths along this normal, "
            f"so it cannot be cut into {n} slabs — at least {n - distinct} would be EMPTY and the "
            "rest would reproduce the partition the file already has. This is the flattened case: "
            "a built bench3 volume that `load_volume` accepts has exactly one depth per section "
            "(loaders.py: 'a section is one depth by definition'), so its within-slab depth is "
            "already gone. Re-cut the volume at BUILD time (bench3's partition='z_width' with the "
            "slab count you want), not from the built file."
        )

    lo, hi = float(depth.min()), float(depth.max())
    if not hi > lo:
        raise ResectionError("plan_partition: the cloud has zero extent along this normal")
    edges = np.linspace(lo, hi, n + 1)
    # `np.digitize` with right=False puts the maximum in a phantom bin n; fold it back into n-1.
    labels = np.clip(np.digitize(depth, edges[1:-1], right=False), 0, n - 1).astype(np.int64)
    counts = np.bincount(labels, minlength=n).astype(np.int64)

    thin = [(i, int(c)) for i, c in enumerate(counts) if c < MIN_CELLS_PER_SECTION]
    if thin:
        raise ResectionError(
            f"plan_partition: cutting into {n} slabs leaves "
            + ", ".join(f"slab {i} with {c} cells" for i, c in thin)
            + f" — under bench3's {MIN_CELLS_PER_SECTION}-cell floor, so those sections would not "
            "be scored. Use fewer slabs, or say explicitly that this volume does not support "
            "this partition."
        )
    return PartitionPlan(
        labels=labels,
        edges=edges,
        counts=counts,
        centres=0.5 * (edges[:-1] + edges[1:]),
        n_distinct_source=distinct,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resection(
    source_h5ad: str | Path,
    out_h5ad: str | Path,
    *,
    normal: tuple[float, float, float],
    n_sections: int,
    dataset_id: str,
    seed: int,
    coord_key: str = "spatial",
    section_key: str = "section",
) -> Path:
    """Re-slice ``source_h5ad`` along ``normal`` into ``n_sections`` slabs. Returns ``out_h5ad``.

    The output is an ordinary bench3 ``data.h5ad``: same ``obsm[coord_key]``, same
    ``obs[cell_type]``, same ``X``; only ``obs[section_key]`` is rewritten, plus a
    ``uns['resection']`` provenance block naming the source's digest, the normal, the slab count and
    the seed. **The source is never written**, and its SHA-256 is asserted unchanged (``specs/10``
    §9's ``test_resection_leaves_source_bitwise_identical``).

    ``seed`` is taken and recorded although nothing here is stochastic — the partition is a
    deterministic function of the coordinates. It is in the signature so a caller cannot believe a
    re-slice is unseeded when a later version needs one (Convention 3).
    """
    import anndata as ad

    src, dst = Path(source_h5ad), Path(out_h5ad)
    before = _sha256(src)
    adata = ad.read_h5ad(src)

    coords = np.asarray(adata.obsm[coord_key], dtype=np.float64)
    if coords.shape[1] != 3:
        raise ResectionError(
            f"resection: adata.obsm[{coord_key!r}] is {coords.shape[1]}-D; re-slicing needs the "
            "cells' own 3-D positions"
        )
    plan = plan_partition(coords, normal, n_sections)

    width = float(plan.edges[1] - plan.edges[0])
    adata.obs[section_key] = [f"{dataset_id}_s{i:02d}" for i in plan.labels]
    # Every cell's depth becomes its slab's centre, because `load_volume` requires exactly one
    # depth per section. This is the same quantisation the source build applied; it is recorded in
    # the provenance block rather than left for a reader to infer.
    unit = np.asarray(normal, dtype=np.float64)
    unit = unit / float(np.linalg.norm(unit))
    depth = coords @ unit
    coords = coords + np.outer(plan.centres[plan.labels] - depth, unit)
    adata.obsm[coord_key] = coords

    protocol = dict(adata.uns.get("paper_protocol") or {})
    protocol["flattened_z"] = True
    adata.uns["paper_protocol"] = protocol
    adata.uns["resection"] = {
        "source": str(src),
        "source_sha256": before,
        "normal": [float(v) for v in unit],
        "n_sections": int(n_sections),
        "slab_width_um": width,
        "n_distinct_source_depths": int(plan.n_distinct_source),
        "cells_per_section": plan.counts.tolist(),
        "dataset_id": str(dataset_id),
        "seed": int(seed),
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(dst)

    after = _sha256(src)
    if after != before:
        raise ResectionError(
            f"resection: the SOURCE changed during the re-slice ({before[:12]} -> {after[:12]}). "
            "specs/10 §9 requires it be left byte-identical; refusing to report this output."
        )
    return dst


def repartition(
    source_h5ad: str | Path,
    out_h5ad: str | Path,
    *,
    n_sections: int,
    dataset_id: str,
    seed: int,
    coord_key: str = "spatial",
    section_key: str = "section",
) -> Path:
    """Cut the same block into ``n_sections`` slabs along its **existing** normal.

    ``specs/10`` §8's V4a: same volume, same cells, same panel, **thickness the only variable**.
    This is :func:`resection` with the normal held at ``(0, 0, 1)``, and it inherits the flattened
    refusal — which is what a built bench3 file will hit, because it has one depth per section by
    the time it loads at all.
    """
    return resection(
        source_h5ad,
        out_h5ad,
        normal=(0.0, 0.0, 1.0),
        n_sections=n_sections,
        dataset_id=dataset_id,
        seed=seed,
        coord_key=coord_key,
        section_key=section_key,
    )
