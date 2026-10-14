"""SpatialCPA-TF's fidelity model and donor decision (run_spatialcpa_tf.py):
the model recovers its coefficients, no donor gradient means the nearest
section, a donor gradient trades off against the gap exactly as the closed form
says, an unestimable contrast falls back to the nearest section, and the
composite drops unusable measures. Lifted with ``ast``."""
from __future__ import annotations

import ast

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")
SRC = BENCH3 / "methods" / "run_spatialcpa_tf.py"
LIFT = {"fidelity_composite", "_design", "fit_fidelity_model", "choose_donor"}


def _lift():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in LIFT)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) == "FIDELITY_METRICS"
                                                  for t in n.targets))]
    ns = {"np": np}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


NS = _lift()
Z = {"s1": 19.0, "s3": 41.0, "s5": 63.0, "s7": 85.0}
PAIRS = [(i, k) for i in Z for k in Z if i != k]


def _model(a, b, beta, c=1.0, noise=0.0):
    rng = np.random.default_rng(0)
    Y = [c + a * Z[i] + b * Z[k] + beta * abs(Z[i] - Z[k]) + noise * rng.standard_normal()
         for i, k in PAIRS]
    return NS["fit_fidelity_model"](PAIRS, Z, Y)


def test_noiseless_fit_recovers_the_coefficients():
    m = _model(0.2, 0.1, -0.3)
    assert m["coef"] == pytest.approx([1.0, 0.2, 0.1, -0.3], abs=1e-9)
    assert m["dof"] == len(PAIRS) - 4


def test_no_donor_gradient_copies_the_nearest_section():
    m = _model(0.0, 0.1, -0.3, noise=1e-12)
    assert NS["choose_donor"](m, 41.0, 63.0, 45.0)[0] is False      # nearer the lower
    assert NS["choose_donor"](m, 41.0, 63.0, 60.0)[0] is True       # nearer the upper


def test_donor_gradient_against_the_gap():
    # gain = a*22 + beta*(|63-zt| - |41-zt|) = 0.2*22 - 0.3*(2*(52-zt)) -> zero at zt = 44.67
    m = _model(0.2, 0.1, -0.3, noise=1e-9)
    use_upper, info = NS["choose_donor"](m, 41.0, 63.0, 52.0)       # midpoint: quality wins
    assert use_upper and info["gain"] == pytest.approx(0.2 * 22, abs=1e-6)
    assert NS["choose_donor"](m, 41.0, 63.0, 46.0)[0] is True
    assert NS["choose_donor"](m, 41.0, 63.0, 43.0)[0] is False      # close to lower: gap wins


def test_unestimable_contrast_is_the_nearest_section():
    z2 = {"s1": 19.0, "s3": 41.0}
    pairs = [("s1", "s3"), ("s3", "s1")]
    m = NS["fit_fidelity_model"](pairs, z2, [1.0, 2.0])
    use_upper, info = NS["choose_donor"](m, 19.0, 41.0, 35.0)
    assert use_upper is True and info["gain"] is None and "nearest" in info["reason"]


def test_exact_midpoint_tie_goes_to_the_lower_flank():
    m = _model(0.0, 0.0, -0.3)
    m["coef"][1] = 0.0                                              # exactly no gradient
    use_upper, info = NS["choose_donor"](m, 41.0, 63.0, 52.0)
    assert use_upper is False and info["reason"] == "tie: nearest"


def test_composite_signs_scales_and_drops_unusable_measures():
    rows = [{"morans_mae": 0.1, "morans_pearson": 0.9, "marker_depth_r": 0.5},
            {"morans_mae": 0.3, "morans_pearson": 0.7, "marker_depth_r": 0.5},
            {"morans_mae": 0.2, "morans_pearson": 0.8}]
    Y, used = NS["fidelity_composite"](rows)
    assert used == ["morans_pearson", "morans_mae"]                  # constant / missing dropped
    sd = np.std([0.9, 0.7, 0.8])
    assert Y == pytest.approx([0.9 / sd - 0.1 / sd, 0.7 / sd - 0.3 / sd, 0.8 / sd - 0.2 / sd])
