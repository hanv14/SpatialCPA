"""Import the unchanged parts of benchmark-pbya-v2 into v3.

v3 changes the *protocol* (which slices are held out, and what is measured), not
the leakage policy, the prediction format or the method code. Rather than fork
those, this module puts ``benchmark-pbya-v2/src`` on ``sys.path`` and re-exports
the pieces v3 reuses:

``leakage_guard``
    ``split_holdout`` / ``assert_no_leakage`` / ``reregister_training`` /
    ``build_labels_train_only`` / ``align_prediction_to_gt``.
``evaluate``
    the cell-matched reference metrics + ``load_prediction`` / ``load_ground_truth``.
``evaluate_generation``
    v2's correspondence-free, scale-fair generation metrics.
``resource_monitor``
    wall time / peak RSS / GPU memory around each method run.

The v2 package is imported under its own name (``benchmark``); v3's package is
``bench3``, so the two never shadow each other.
"""

from __future__ import annotations

import sys

from .config import V2_SRC


def ensure_v2_importable():
    """Put benchmark-pbya-v2/src on sys.path (idempotent)."""
    if not (V2_SRC / "benchmark" / "__init__.py").exists():
        raise ImportError(
            f"benchmark-pbya-v2 not found at {V2_SRC}. benchmark-pbya-v3 reuses "
            f"v2's leakage guard, evaluators and method wrappers; keep the two "
            f"directories as siblings, or set BENCH_V3_* paths accordingly."
        )
    p = str(V2_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)


ensure_v2_importable()

from benchmark import leakage_guard                                   # noqa: E402
from benchmark.evaluate import (                                      # noqa: E402
    evaluate, load_prediction, load_ground_truth,
)
from benchmark.evaluate_generation import evaluate_generation         # noqa: E402
from benchmark.resource_monitor import ResourceMonitor, write_resources  # noqa: E402

split_holdout = leakage_guard.split_holdout
assert_no_leakage = leakage_guard.assert_no_leakage
reregister_training = leakage_guard.reregister_training
build_labels_train_only = leakage_guard.build_labels_train_only
align_prediction_to_gt = leakage_guard.align_prediction_to_gt

__all__ = [
    "leakage_guard", "split_holdout", "assert_no_leakage",
    "reregister_training", "build_labels_train_only", "align_prediction_to_gt",
    "evaluate", "load_prediction", "load_ground_truth", "evaluate_generation",
    "ResourceMonitor", "write_resources", "ensure_v2_importable",
]
