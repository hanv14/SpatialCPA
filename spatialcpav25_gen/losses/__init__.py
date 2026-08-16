"""Training objectives.

``reconstruction`` holds the likelihood terms — T05's layout NLL now, T06's ZINB
reconstruction next; ``sefl`` (T07) and ``metric_aware`` (T08) follow.
"""

from spatialcpav25_gen.losses.reconstruction import layout_poisson_nll

__all__ = ["layout_poisson_nll"]
