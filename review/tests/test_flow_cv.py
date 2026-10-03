"""flow-cv's decision rule (run_spatialcpav18_flow.py): the threshold calibrated
on training folds, and the per-fold metric gain it is calibrated on.

The two functions and their constants are lifted out of the wrapper with ``ast``
and run on numpy alone, so the test needs neither torch nor anndata.
"""
from __future__ import annotations

import ast
from collections import namedtuple

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")

FLOW_WRAPPER = BENCH3 / "methods" / "run_spatialcpav18_flow.py"
WANTED = {"CV_METRICS", "CV_TIE_TOL", "calibrate_delta", "fold_gain"}


def _lift():
    tree = ast.parse(FLOW_WRAPPER.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED
                                                  for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(FLOW_WRAPPER), "exec"), ns)
    assert WANTED <= set(ns), WANTED - set(ns)
    return ns


NS = _lift()
F = namedtuple("F", "margin gain")
INF = float("inf")


def test_no_folds_or_no_gain_never_switches():
    cal = NS["calibrate_delta"]
    assert cal([]) == (INF, 0)
    assert cal([F(0.3, 0), F(-0.2, 0)]) == (INF, 0)
    assert cal([F(0.3, -2), F(-0.2, -1)]) == (INF, 0)


def test_threshold_switches_exactly_the_folds_that_gained():
    cal = NS["calibrate_delta"]
    # the fold where the flow preferred the other flank by 0.3 gained; switching
    # the -0.2 fold as well would have cost 3
    delta, g = cal([F(0.3, 2), F(-0.2, -3)])
    assert g == 2 and -0.2 < delta < 0.3 and 0.3 > delta
    # a gain only reachable by also switching where the flow preferred the rule
    # flank: delta goes negative (a calibrated bias, not the flow's raw sign)
    delta, g = cal([F(-0.5, 3), F(-0.1, 1)])
    assert g == 4 and delta < -0.5


def test_ties_keep_the_largest_threshold():
    cal = NS["calibrate_delta"]
    delta, g = cal([F(0.4, 1), F(0.1, 0)])     # switching 0.1 too adds nothing
    assert g == 1 and 0.1 <= delta < 0.4


def test_fold_gain_counts_wins_minus_losses_with_direction_and_ties():
    gain = NS["fold_gain"]
    keys = [k for k, _ in NS["CV_METRICS"]]
    rule = {k: 0.5 for k in keys}
    assert gain(rule, dict(rule))[0] == 0
    other = dict(rule, morans_mae=0.4)               # lower MAE is better: a win
    assert gain(rule, other)[0] == 1
    other = dict(rule, morans_pearson=0.4)           # lower r: a loss
    assert gain(rule, other)[0] == -1
    other = dict(rule, celltype_localization=0.5 + NS["CV_TIE_TOL"] / 2)
    assert gain(rule, other)[0] == 0                 # last-bit Sinkhorn noise is a tie
    other = dict(rule, marker_depth_r=float("nan"))
    assert gain(rule, other)[0] == 0                 # missing on one side: no vote
