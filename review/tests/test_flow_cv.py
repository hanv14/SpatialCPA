"""flow-cv's and patch-cv's decision rules (run_spatialcpav18_flow.py): the
threshold / switched fraction calibrated on training folds, the per-fold metric
gain they are calibrated on, and the patch bookkeeping.

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
WANTED = {"CV_METRICS", "CV_TIE_TOL", "calibrate_delta", "fold_gain",
          "PATCH_Q_GRID", "patch_ids", "patch_margins", "patches_to_switch", "calibrate_q"}


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


# ── patch-cv ──────────────────────────────────────────────────────────────────
def test_patch_grid_is_shared_by_both_flanks():
    lo = np.array([[0.0, 0.0], [9.0, 0.0], [10.0, 0.0], [0.0, 25.0]])
    hi = np.array([[1.0, 1.0], [19.0, 1.0]])
    pl, ph = NS["patch_ids"](lo, hi, 10.0)
    assert pl[0] == pl[1] == ph[0]            # same 10x10 square
    assert pl[2] == ph[1] != pl[0]            # next square along x, both flanks
    assert pl[3] not in (pl[0], pl[2])        # a row up


def test_patch_margins_need_both_flanks_and_sign_favours_other():
    pid_rule = np.array([1, 1, 2, 2, 3])
    d_rule = np.array([5.0, 5.0, 1.0, 1.0, 9.0])
    pid_other = np.array([1, 1, 2, 2])
    d_other = np.array([2.0, 2.0, 3.0, 3.0])
    patches, m = NS["patch_margins"](pid_rule, d_rule, pid_other, d_other, 2)
    assert list(patches) == [1, 2]            # patch 3: no other-flank cells
    assert m[0] == pytest.approx(3.0) and m[1] == pytest.approx(-2.0)


def test_switch_takes_top_margins_or_a_seeded_random_draw():
    sw = NS["patches_to_switch"]
    patches, m = np.array([10, 11, 12, 13]), np.array([0.1, 0.9, -0.5, 0.4])
    assert sw(patches, m, 0.0, "flow", 0).size == 0
    assert list(sw(patches, m, 0.5, "flow", 0)) == [11, 13]
    assert sorted(sw(patches, m, 1.0, "flow", 0)) == [10, 11, 12, 13]
    r1, r2 = sw(patches, m, 0.5, "random", 7), sw(patches, m, 0.5, "random", 7)
    assert list(r1) == list(r2) and len(r1) == 2   # deterministic, same count
    with pytest.raises(ValueError):
        sw(patches, m, 0.5, "best", 0)


def test_calibrate_q_needs_a_strict_gain_and_prefers_small_q():
    cal = NS["calibrate_q"]
    grid = NS["PATCH_Q_GRID"]
    assert grid[0] == 0.0 and grid[-1] == 1.0
    assert cal({q: 0 for q in grid}) == (0.0, 0)
    assert cal({0.0: 0, 0.1: -1, 0.25: 3, 0.5: 3, 0.75: 1, 1.0: 2}) == (0.25, 3)
