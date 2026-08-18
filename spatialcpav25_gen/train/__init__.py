"""Training loops that are more than one optimiser step.

T08 contributes the internal leave-one-section-out schedule the metric-aware losses are
computed under; T09 adds the per-dataset config selector beside it.
"""

from spatialcpav25_gen.train.loso import (
    LOSOScheduler,
    Reconstruction,
    metric_aware_terms,
    reconstruct_hidden,
)

__all__ = [
    "LOSOScheduler",
    "Reconstruction",
    "metric_aware_terms",
    "reconstruct_hidden",
]
