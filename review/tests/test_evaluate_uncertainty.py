"""evaluate_uncertainty.py scores what it claims: a score that tracks the patch
error gets a high rho, an unrelated score gets ~0, and a score that only tracks
patch cell count is caught by the partial rho. Needs numpy/scipy (bench_eval)."""
from __future__ import annotations

import sys

import pytest

from conftest import BENCH_ROOT

np = pytest.importorskip("numpy")
pytest.importorskip("anndata")
if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
from src.bench3 import evaluate_uncertainty as eu  # noqa: E402


def _section(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    R = rng.random((n, 6))
    return xy, R


def test_patch_error_is_zero_for_the_ground_truth():
    xy, R = _section()
    err, tab = eu.patch_table(xy, R, xy, R, {"u": np.zeros(len(xy))}, grid=8)
    assert len(err) > 0 and np.allclose(err, 0)
    assert np.all(tab["sampling"] > 0)


def test_a_score_that_tracks_the_error_ranks_it_and_noise_does_not():
    xy, R = _section()
    bad = xy[:, 0] > 60                                   # the prediction is wrong here
    P = R.copy()
    P[bad] = np.random.default_rng(1).random((bad.sum(), 6)) ** 3
    u_good = xy[:, 0] / 100.0
    u_noise = np.random.default_rng(2).random(len(xy))
    err, tab = eu.patch_table(xy, P, xy, R, {"good": u_good, "noise": u_noise}, grid=8)
    g = eu.score_metrics(err, tab["good"], tab["sampling"])
    z = eu.score_metrics(err, tab["noise"], tab["sampling"])
    assert g["rho"] > 0.6 and g["auroc_worst"] > 0.8
    assert abs(z["rho"]) < 0.3
    q = g["quantile_err"]
    assert q[-1] > q[0]                                   # error rises with the score


def test_partial_rho_discounts_a_pure_cell_count_score():
    rng = np.random.default_rng(3)
    n = rng.integers(5, 60, 200).astype(float)
    sampling = np.sqrt(2.0 / n)
    err = sampling * rng.lognormal(0, 0.1, 200)           # error = cell-count noise only
    m = eu.score_metrics(err, 1.0 / n, sampling)          # a score that just counts cells
    assert m["rho"] > 0.8 and abs(m["rho_partial"]) < 0.3
