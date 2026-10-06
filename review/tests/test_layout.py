"""The density-relayout bookkeeping (run_spatialcpav18_layout.py): target
counts keep the total and equal the copy at rho 0; rounding is exact. Lifted with
``ast`` so the test needs neither torch nor anndata."""
from __future__ import annotations

import ast

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")
SRC = BENCH3 / "methods" / "run_spatialcpav18_layout.py"
WANTED = {"LAYOUT_RHO_GRID", "bin_ids", "largest_remainder", "target_counts"}


def _lift():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED
                                                  for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


NS = _lift()


def test_target_counts_are_the_copy_at_rho_zero_and_keep_the_total():
    copied = np.array([10, 0, 5, 7])
    w = np.array([1.0, 3.0, 0.0, 2.0])
    assert np.array_equal(NS["target_counts"](copied, w, 0.0), copied)
    for rho in NS["LAYOUT_RHO_GRID"]:
        assert NS["target_counts"](copied, w, rho).sum() == copied.sum()
    full = NS["target_counts"](copied, w, 1.0)
    assert full[2] == 0 and full[1] > full[3] > 0       # proportional to w


def test_largest_remainder_is_exact_and_proportional():
    n = NS["largest_remainder"]([1.0, 1.0, 1.0], 10)
    assert n.sum() == 10 and n.max() - n.min() <= 1
    assert list(NS["largest_remainder"]([0.0, 0.0], 5)) == [0, 0]


def test_bins_group_points_by_square():
    ids = NS["bin_ids"](np.array([[0.1, 0.1], [0.9, 0.9], [1.1, 0.1], [0.1, 1.1]]),
                        np.array([0.0, 0.0]), 1.0)
    assert ids[0] == ids[1] and len({ids[0], ids[2], ids[3]}) == 3
