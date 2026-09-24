"""Import the shared evaluators and leakage guard (``src/benchmark``).

``src/benchmark`` holds the pieces the harness and the scorer share. They are
kept under their original import name, ``benchmark``, because the pinned
``evaluate_paper.py`` imports ``benchmark.evaluate_generation`` directly. This
module puts ``src/`` on ``sys.path`` and re-exports:

``leakage_guard``
    ``split_holdout`` / ``assert_no_leakage`` / ``reregister_training`` /
    ``build_labels_train_only`` / ``align_prediction_to_gt``.
``evaluate``
    the cell-matched reference metrics + ``load_prediction`` / ``load_ground_truth``.
``evaluate_generation``
    the correspondence-free, scale-fair generation metrics.
``resource_monitor``
    wall time / peak RSS / GPU memory around each method run.

The name keeps its history (``_v2bridge``) because ``evaluate_paper.py`` imports
it by that name.
"""

from __future__ import annotations

import sys

from .config import SHARED_SRC


def ensure_v2_importable():
    """Put src/ on sys.path so ``import benchmark`` resolves (idempotent)."""
    if not (SHARED_SRC / "benchmark" / "__init__.py").exists():
        raise ImportError(
            f"the shared evaluators were not found at {SHARED_SRC / 'benchmark'}; "
            f"they ship with this package — is the checkout complete?")
    p = str(SHARED_SRC)
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
