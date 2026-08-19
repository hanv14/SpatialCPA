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
from spatialcpav25_gen.train.select import (
    GATES,
    METRIC_NAMES,
    V20_CONFIG,
    Candidate,
    SelectionError,
    SelectionResult,
    calibration_chunks,
    joint_gate_cells,
    module_morans_agreement,
    run_selection,
    section_scores,
    select_config,
    selection_folds,
    selection_scores,
    write_selection_report,
)

__all__ = [
    "GATES",
    "METRIC_NAMES",
    "V20_CONFIG",
    "Candidate",
    "LOSOScheduler",
    "Reconstruction",
    "SelectionError",
    "SelectionResult",
    "calibration_chunks",
    "joint_gate_cells",
    "metric_aware_terms",
    "module_morans_agreement",
    "reconstruct_hidden",
    "run_selection",
    "section_scores",
    "select_config",
    "selection_folds",
    "selection_scores",
    "write_selection_report",
]
