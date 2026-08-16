"""Baselines. T06 needs exactly one of them, and it is built to fail.

The independent-donor sampler reimplements the competing method's expression path: each
generated cell is matched to its ``Config.independent_donor_k`` nearest real cells and then
**every gene is drawn independently** from among them. That is the "artificial doublet-like
profile" the competing method's own authors describe, and the reason this project decodes all
of a cell's genes from one shared latent instead.

It is a **deliberate negative control** (``specs/11_COVERAGE_MATRIX.md``, "Deliberate negative
controls"), not dead code. Two tasks consume it: T06's
``test_shared_latent_preserves_covariance``, which requires the shared-latent decoder's
gene-gene correlation error to be less than half of this baseline's, and T10, which reports it
in the headline table. Do not delete it as unused.

Why it shares :func:`~spatialcpav25_gen.model.expression.cross_mix_counts`
-------------------------------------------------------------------------
Structurally, "pick a donor per gene and take its real count" is one operation, and both the
v20 cross-mix and this baseline are instances of it — which is worth saying out loud, because
it is precisely where the two methods differ and the difference is in the *weights*, not the
mechanism:

* v20's cross-mix mixes **two** donors at a weight that is zero at narrow gaps, so a narrow
  design emits verbatim real profiles and has no chimerism at all;
* the competing method mixes ``<= 3`` donors at essentially uniform weights **for every cell
  of every section**, so every emitted profile is a chimera of three cells.

Implementing the mechanism once and parameterising the weights makes that comparison a
one-line difference rather than an argument about two similar-looking functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.spatial import cKDTree
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, to_xyz
from spatialcpav25_gen.model.expression import cross_mix_counts

__all__ = ["IndependentDonorBaseline", "independent_donor_counts"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]


def independent_donor_counts(
    donor_counts: Tensor | npt.NDArray[Any],
    gen: np.random.Generator,
    *,
    cfg: Config | None = None,
) -> Tensor:
    """Draw each gene independently from among the donors. ``(N, D, G)`` -> ``(N, G)`` float32.

    The competing method's sampler. Uniform over the ``D`` donors, independently per *(cell,
    gene)* — which is the whole point: the marginal per-gene distribution is right by
    construction and the within-cell gene-gene covariance is destroyed, because two genes of
    the same emitted cell may come from two different donors.

    ``gen`` is an explicit ``numpy.random.Generator`` (Convention 3).
    """
    counts = np.asarray(
        donor_counts.detach().cpu().numpy() if isinstance(donor_counts, Tensor) else donor_counts,
        dtype=np.float64,
    )
    if counts.ndim != 3:
        raise ValueError(f"independent_donor_counts expects (N, D, G), got {counts.shape}")
    n, d, _ = counts.shape
    weights = np.full((n, d), 1.0 / float(d), dtype=np.float64)
    return cross_mix_counts(counts, weights, gen, cfg=cfg)


@dataclass(frozen=True)
class IndependentDonorBaseline:
    """The competing method's expression path over a fixed donor pool.

    Args:
        donor_coords: ``(M, 3)`` float64 physical positions of the real donor cells.
        donor_counts: ``(M, G)`` CSR of their raw counts.
        n_donors: ``D``, donors matched per generated cell.

    Built by :meth:`from_sections`, which is what both T06 and T10 call: the pool is the real
    sections flanking the plane being generated, exactly as the competing method uses them.
    """

    donor_coords: FloatArray
    donor_counts: Any
    n_donors: int

    def __post_init__(self) -> None:
        """Check the pool is large enough to draw from and consistent in shape."""
        if self.donor_coords.ndim != 2 or self.donor_coords.shape[1] != 3:
            raise ValueError(
                f"IndependentDonorBaseline: donor_coords must be (M, 3), got "
                f"{self.donor_coords.shape}"
            )
        if self.donor_counts.shape[0] != self.donor_coords.shape[0]:
            raise ValueError(
                f"IndependentDonorBaseline: donor_counts has {self.donor_counts.shape[0]} rows "
                f"but donor_coords has {self.donor_coords.shape[0]}"
            )
        if self.n_donors < 2:
            raise ValueError(
                f"IndependentDonorBaseline: n_donors must be >= 2, got {self.n_donors}; with "
                "one donor the baseline is a verbatim copy and has no chimerism to measure"
            )
        if self.donor_coords.shape[0] < self.n_donors:
            raise ValueError(
                f"IndependentDonorBaseline: the pool holds {self.donor_coords.shape[0]} cells, "
                f"fewer than n_donors={self.n_donors}"
            )

    @classmethod
    def from_sections(
        cls, sections: list[Section], cfg: Config, *, n_donors: int | None = None
    ) -> IndependentDonorBaseline:
        """Pool real sections as the donor set.

        ``n_donors`` defaults to ``Config.independent_donor_k``.
        """
        if not sections:
            raise ValueError("IndependentDonorBaseline.from_sections: needs at least one section")
        coords = np.concatenate([to_xyz(s).astype(np.float64) for s in sections], axis=0)
        counts = sparse.vstack([s.counts for s in sections]).tocsr().astype(np.float64)
        return cls(
            donor_coords=coords,
            donor_counts=counts,
            n_donors=int(cfg.independent_donor_k if n_donors is None else n_donors),
        )

    def generate(self, xyz: npt.NDArray[Any], seed: int, *, cfg: Config | None = None) -> Tensor:
        """Emit counts at ``xyz``. ``(N, 3)`` -> ``(N, G)`` float32, integer-valued.

        The ``D`` nearest donors in 3-D, then :func:`independent_donor_counts`. Nearest in 3-D
        rather than in-plane because the pool is a set of *flanking* sections at other depths,
        which is the regime the competing method is used in.
        """
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                f"IndependentDonorBaseline.generate expects (N, 3), got {points.shape}"
            )
        tree = cKDTree(self.donor_coords)
        idx = np.atleast_2d(tree.query(points, k=self.n_donors)[1]).astype(np.intp)
        donors = np.asarray(self.donor_counts[idx.reshape(-1)].todense(), dtype=np.float64)
        donors = donors.reshape(points.shape[0], self.n_donors, -1)
        return independent_donor_counts(donors, np.random.default_rng(seed), cfg=cfg)
