"""Training objectives.

``reconstruction`` holds the likelihood terms — T05's layout NLL and T06's ZINB
reconstruction; ``sefl`` (T07) holds the three sectioning-equivariance consistency losses,
their stop-gradient teacher and the invariant/equivariant table they are constrained by;
``metric_aware`` (T08) turns the evaluation criteria themselves into differentiable terms,
computed under the internal leave-one-section-out loop of :mod:`spatialcpav25_gen.train.loso`.
"""

from spatialcpav25_gen.losses.metric_aware import (
    METRIC_TERMS,
    MetricDominanceWarning,
    gearys_c,
    knn_weight_graph,
    loss_autocorr,
    loss_distribution,
    loss_profile,
    morans_i,
    soft_depth_profile,
    soft_field_profile,
)
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
    "METRIC_TERMS",
    "SEFL_LOSSES",
    "EMATeacher",
    "MetricDominanceWarning",
    "gearys_c",
    "knn_weight_graph",
    "layout_poisson_nll",
    "loss_autocorr",
    "loss_cross",
    "loss_distribution",
    "loss_profile",
    "loss_prog",
    "loss_prog_WRONG",
    "loss_thick",
    "morans_i",
    "soft_depth_profile",
    "soft_field_profile",
]
