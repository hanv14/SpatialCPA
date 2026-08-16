"""Evaluation: baselines (T06/T10), the vendored paper metrics and the harness (T10).

T06 contributes only :mod:`spatialcpav25_gen.eval.baselines`, and only the one baseline it
needs: the competing method's per-gene independent donor draw, which is the reference point
for the gene-gene covariance claim. T10 adds the rest of the baselines beside it.
"""

from spatialcpav25_gen.eval.baselines import (
    IndependentDonorBaseline,
    independent_donor_counts,
)

__all__ = [
    "IndependentDonorBaseline",
    "independent_donor_counts",
]
