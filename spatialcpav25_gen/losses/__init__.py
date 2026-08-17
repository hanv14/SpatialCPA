"""Training objectives.

``reconstruction`` holds the likelihood terms — T05's layout NLL and T06's ZINB
reconstruction; ``sefl`` (T07) holds the three sectioning-equivariance consistency losses,
their stop-gradient teacher and the invariant/equivariant table they are constrained by;
``metric_aware`` (T08) follows.
"""

from spatialcpav25_gen.losses.reconstruction import layout_poisson_nll
from spatialcpav25_gen.losses.sefl import (
    EQUIVARIANT_QUANTITIES,
    INVARIANT_QUANTITIES,
    SEFL_LOSSES,
    EMATeacher,
    loss_cross,
    loss_prog,
    loss_prog_WRONG,
    loss_thick,
)

__all__ = [
    "EQUIVARIANT_QUANTITIES",
    "INVARIANT_QUANTITIES",
    "SEFL_LOSSES",
    "EMATeacher",
    "layout_poisson_nll",
    "loss_cross",
    "loss_prog",
    "loss_prog_WRONG",
    "loss_thick",
]
