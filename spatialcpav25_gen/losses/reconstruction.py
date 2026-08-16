"""Likelihood terms. T05 contributes the layout's point-process NLL.

T06 adds the ZINB reconstruction term to this module; the two live together because they
are the two halves of "what did the model say about the data it was shown" — where the
cells are, and what they expressed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spatialcpav25_gen.config import Config

__all__ = ["layout_poisson_nll"]


def layout_poisson_nll(
    lambda_cells: Tensor,
    cell_type: Tensor,
    lambda_mc: Tensor,
    slab_volume: float,
    cfg: Config,
) -> Tensor:
    """Inhomogeneous Poisson process NLL of one section's layout. Returns a scalar.

    Parameters
    ----------
    lambda_cells
        ``(N, C)`` per-type intensity at the ``N`` observed cell positions, in cells per
        micrometre^3 — the output of :class:`~spatialcpav25_gen.model.layout.IntensityHead`.
    cell_type
        ``(N,)`` int64 type codes of those cells.
    lambda_mc
        ``(M, C)`` per-type intensity at ``M`` points drawn **uniformly in the slab
        volume** (``uniform_slab_points``).
    slab_volume
        The slab's volume in micrometre^3: the section's in-plane window times its
        thickness.
    cfg
        Supplies ``layout_intensity_eps`` and ``debug_shapes``.

    Returns
    -------
    Tensor
        Scalar ``()``, ``-sum_i log lambda_{c_i}(p_i) + integral over the slab of
        sum_c lambda_c``, the second term estimated as
        ``slab_volume * mean_m sum_c lambda_mc[m, c]``.

    Notes
    -----
    **The slab volume, not the area.** T07's thickness-consistency loss is stated in terms
    of a slab, and a count that emerges from a *volume* integral is what makes it coherent:
    doubling a section's thickness must double its expected cell count, and an area integral
    would leave it unchanged.

    The term is a sum over cells, not a mean, because that is what the process likelihood
    is — the two terms have to stay on the same scale for the fitted intensity to have the
    right units. A caller that wants a per-cell number for logging divides afterwards.
    """
    if cfg.debug_shapes:
        assert lambda_cells.ndim == 2, lambda_cells.shape
        assert lambda_mc.ndim == 2, lambda_mc.shape
        assert cell_type.shape == (lambda_cells.shape[0],), cell_type.shape
        assert lambda_mc.shape[1] == lambda_cells.shape[1], lambda_mc.shape
    if lambda_mc.shape[1] != lambda_cells.shape[1]:
        raise ValueError(
            f"layout_poisson_nll: lambda_cells has {lambda_cells.shape[1]} cell types but "
            f"lambda_mc has {lambda_mc.shape[1]}"
        )
    if cell_type.shape[0] != lambda_cells.shape[0]:
        raise ValueError(
            f"layout_poisson_nll: cell_type has {cell_type.shape[0]} entries but "
            f"lambda_cells has {lambda_cells.shape[0]} rows"
        )
    eps = float(cfg.layout_intensity_eps)
    observed = lambda_cells.gather(1, cell_type.to(torch.long)[:, None]).squeeze(1)
    likelihood = -torch.log(observed + eps).sum()
    integral = float(slab_volume) * lambda_mc.sum(dim=1).mean()
    out = likelihood + integral
    if cfg.debug_shapes:
        assert out.shape == (), out.shape
    return out
