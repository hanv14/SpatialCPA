"""flow-cv-stable's confidence-gated threshold (run_spatialcpav18_flowcv_stable.py):
noise-level evidence never switches, consistent evidence does, and the
threshold switches exactly the folds that carry the gain. Lifted with ``ast``."""
from __future__ import annotations

import ast

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")
SRC = BENCH3 / "methods" / "run_spatialcpav18_flowcv_stable.py"


def _lift():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name == "calibrate_delta_lcb")
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "Z"
                                                  for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


NS = _lift()
INF = float("inf")


def test_noisy_positive_evidence_does_not_switch():
    # mean +3 but se 4: the lower bound is negative
    assert NS["calibrate_delta_lcb"]([-0.7, -0.3], [3.0, 0.0], [4.0, 4.0]) == (INF, 0.0)


def test_consistent_evidence_switches_only_the_folds_that_gain():
    delta, lb = NS["calibrate_delta_lcb"]([-0.7, -0.3], [-20.0, 30.0], [2.0, 2.0])
    assert -0.7 < delta < -0.3 and lb == pytest.approx(30.0 - NS["Z"] * 2.0)


def test_switching_everything_when_every_fold_gains():
    delta, lb = NS["calibrate_delta_lcb"]([-0.7, -0.3], [10.0, 10.0], [1.0, 1.0])
    assert delta < -0.7 and lb > 10.0
