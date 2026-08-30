"""A recorded, seeded, stratified held-out gene split — the zero-shot experiment's subject.

The zero-shot question is whether the text channel can place a gene the model was never fitted
on. That needs a gene split, and an *unstratified* draw would confound the answer: hold out only
rare genes and every arm fails; hold out only abundant, spatially structured ones and the metric
is easy for all of them. Worse, ``marker_genes`` selects the metric's markers by **Moran's I**
with a detection floor, so a split that happened to take low-autocorrelation genes would leave
`marker_depth_r` with no eligible markers among the held-out set and no measurement at all.

So the draw is stratified jointly on the two axes that decide both of those: **mean expression**
and **Moran's I**, each cut into quantile bins, with the same fraction taken from every cell of
the grid. Seeded explicitly (Convention 3) and written out gene-by-gene with the statistics it
was stratified on, so the split can be audited and reproduced rather than trusted.

Moran's I is computed on **one section** — the volume's largest — because the split only needs a
ranking, and a per-section kNN graph over every gene of a 1017-gene panel is the expensive part.
Which section is recorded in the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume
from spatialcpav25_gen.losses.metric_aware import knn_weight_graph, marker_genes, morans_i
from spatialcpav25_gen.train.select import _normalised


@dataclass(frozen=True)
class GeneSplit:
    """Which genes the model may fit on, and which are held out. Indices into the panel.

    Attributes
    ----------
    kept, held_out
        ``(n_kept,)`` and ``(n_held,)`` int64 panel indices, sorted. Disjoint, and together the
        whole panel.
    names
        The panel's gene names, so a written split is readable without the volume.
    stats
        ``{gene_index: {"mean": float, "morans": float, "detection": float, "stratum": str}}``
        — what the draw was stratified on, recorded for audit.
    reference_section
        The section Moran's I was computed on.
    seed, n_bins, frac
        The draw's parameters.
    """

    kept: npt.NDArray[np.int64]
    held_out: npt.NDArray[np.int64]
    names: list[str]
    stats: dict[int, dict[str, Any]]
    reference_section: str
    seed: int
    n_bins: int
    frac: float

    def __post_init__(self) -> None:
        """Refuse a split that is not a partition, or that empties either side."""
        both = set(self.kept.tolist()) & set(self.held_out.tolist())
        if both:
            raise ValueError(f"GeneSplit: {len(both)} gene(s) are both kept and held out")
        total = len(self.kept) + len(self.held_out)
        if total != len(self.names):
            raise ValueError(
                f"GeneSplit: {total} genes across the two sides but the panel has "
                f"{len(self.names)}; a split must be a partition"
            )
        if not len(self.kept) or not len(self.held_out):
            raise ValueError(
                f"GeneSplit: {len(self.kept)} kept and {len(self.held_out)} held out; both "
                "sides must be non-empty or there is no experiment"
            )

    def held_out_names(self) -> list[str]:
        """The held-out genes' symbols, in panel order."""
        return [self.names[int(i)] for i in self.held_out]

    def to_json(self) -> dict[str, Any]:
        """A round-trippable record of the split and the statistics behind it."""
        return {
            "seed": int(self.seed),
            "n_bins": int(self.n_bins),
            "frac": float(self.frac),
            "reference_section": self.reference_section,
            "n_kept": len(self.kept),
            "n_held_out": len(self.held_out),
            "kept": [int(i) for i in self.kept],
            "held_out": [int(i) for i in self.held_out],
            "genes": [
                {
                    "index": int(i),
                    "name": self.names[int(i)],
                    "held_out": bool(i in set(self.held_out.tolist())),
                    **{
                        k: (float(v) if k != "stratum" else v)
                        for k, v in self.stats[int(i)].items()
                    },
                }
                for i in range(len(self.names))
            ],
        }


def _reference_section(vol: TrainingVolume) -> Section:
    """The section Moran's I is ranked on: the largest, ties broken by id."""
    return max(vol.sections, key=lambda s: (s.n_cells, s.section_id))


def stratified_gene_split(
    vol: TrainingVolume, cfg: Config, *, seed: int, frac: float = 0.2, n_bins: int = 5
) -> GeneSplit:
    """Hold out ``frac`` of the panel, stratified on mean expression x Moran's I.

    Both axes are cut into ``n_bins`` **quantile** bins, giving ``n_bins**2`` strata; from each
    stratum a ``frac`` share is drawn without replacement, rounded so the total lands within one
    gene of ``frac * n_genes``. Quantile rather than equal-width bins because both statistics are
    heavily skewed on a real panel and equal-width cutting would leave most strata empty.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"stratified_gene_split: frac must be in (0, 1), got {frac!r}")
    section = _reference_section(vol)
    counts = np.asarray(section.counts.todense(), dtype=np.float64)
    x = _normalised(counts, cfg)
    graph = knn_weight_graph(np.asarray(section.coords, dtype=np.float64), cfg)
    with torch.no_grad():
        morans = np.nan_to_num(morans_i(x, graph, eps=float(cfg.metric_eps)).numpy(), nan=0.0)
    mean = np.asarray(x.mean(dim=0))
    detection = np.asarray((counts > 0).mean(axis=0))

    def bins(values: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
        return np.digitize(values, edges).astype(np.int64)

    strata = list(zip(bins(mean), bins(morans), strict=True))
    rng = np.random.default_rng(seed)
    held: list[int] = []
    for key in sorted(set(strata)):
        members = np.array([i for i, s in enumerate(strata) if s == key], dtype=np.int64)
        take = round(frac * len(members))
        if take:
            held.extend(rng.choice(members, size=take, replace=False).tolist())
    held_out = np.array(sorted(held), dtype=np.int64)
    kept = np.array(sorted(set(range(len(vol.gene_names))) - set(held)), dtype=np.int64)
    stats = {
        i: {
            "mean": float(mean[i]),
            "morans": float(morans[i]),
            "detection": float(detection[i]),
            "stratum": f"{strata[i][0]}x{strata[i][1]}",
        }
        for i in range(len(vol.gene_names))
    }
    return GeneSplit(
        kept=kept,
        held_out=held_out,
        names=list(vol.gene_names),
        stats=stats,
        reference_section=section.section_id,
        seed=int(seed),
        n_bins=int(n_bins),
        frac=float(frac),
    )


def markers_within(
    x_real: torch.Tensor, graph: Any, cfg: Config, pool: npt.NDArray[np.int64]
) -> torch.Tensor:
    """``marker_genes`` restricted to a gene ``pool``. ``(k,)`` int64 **panel** indices.

    A thin adapter over :func:`~spatialcpav25_gen.losses.metric_aware.marker_genes` and nothing
    more. It was a mirrored copy of that rule — Moran's I, the detection floor, the fallback —
    and a copy of a selection rule drifts from it; the rule now takes ``pool`` itself.
    """
    return marker_genes(x_real, graph, cfg, pool=torch.from_numpy(np.asarray(pool, np.int64)))
