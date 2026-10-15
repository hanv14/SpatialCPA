"""SpatialCPA-X (run_spatialcpa_x.py), its raw-scale evaluator (evaluate_depth.py)
and the extrapolation designs (run_extrapolation.py). Pure functions lifted with
``ast``: the donor rule inside and beyond the range, empirical-Bayes shrinkage of
depth slopes, the count-transfer model, the abundance metrics and the end-of-
stack hold-outs."""
from __future__ import annotations

import ast

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")


def _lift(path, names, extra=None):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = {"np": np, **(extra or {})}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


X = _lift(BENCH3 / "methods" / "run_spatialcpa_x.py",
          {"choose_donor_any", "nearest_section", "depth_slopes", "transfer_counts"})
Z = {"s1": 19.0, "s3": 41.0, "s5": 63.0, "s7": 85.0}


# ── donor rule ───────────────────────────────────────────────────────────────

def test_no_donor_gradient_is_the_nearest_section_inside_and_beyond():
    for zt, want in ((45.0, "s3"), (60.0, "s5"), (100.0, "s7"), (5.0, "s1")):
        assert X["choose_donor_any"]((0, 0.0, 0.3, -0.3), Z, zt)[0] == want


def test_midpoint_tie_goes_to_the_shallower_flank():
    assert X["choose_donor_any"]((0, 0.0, 0.0, -0.3), Z, 52.0)[0] == "s3"


def test_gradient_picks_the_better_flank_but_not_past_the_gap():
    coef = (0, 0.2, 0.0, -0.3)                  # |beta| > a: always a flank, never further
    assert X["choose_donor_any"](coef, Z, 52.0)[0] == "s5"
    assert X["choose_donor_any"](coef, Z, 43.0)[0] == "s3"
    assert X["choose_donor_any"](coef, Z, 100.0)[0] == "s7"  # beyond the deep end: the edge
    assert X["choose_donor_any"](coef, Z, 5.0)[0] == "s1"    # beyond the shallow end: the edge


def test_strong_gradient_beyond_the_shallow_end_skips_a_poor_edge():
    coef = (0, 0.5, 0.0, -0.3)                  # a > |beta|: deeper sections win outright
    assert X["choose_donor_any"](coef, Z, 5.0)[0] == "s7"


# ── depth response ───────────────────────────────────────────────────────────

def _means(slopes, z=Z, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return {s: np.exp(5.0 + np.asarray(slopes) * zs + noise * rng.standard_normal(len(slopes)))
            for s, zs in z.items()}


def test_noiseless_slopes_are_recovered():
    true = np.array([-0.03, -0.01, 0.02])
    s, info = X["depth_slopes"](_means(true), Z)
    assert info["mode"] == "empirical Bayes"
    assert s == pytest.approx(true, abs=1e-6)


def test_common_slope_plus_noise_is_shrunk_to_the_common_slope():
    m = _means(np.full(40, -0.02), noise=0.3)
    s, info = X["depth_slopes"](m, Z)
    zz = np.array(list(Z.values()))
    raw = np.polyfit(zz, np.log(np.array([m[k] for k in Z])), 1)[0]   # unshrunk OLS slopes
    assert info["mean_shrinkage"] > 0.9
    assert np.abs(s + 0.02).max() < 0.2 * np.abs(raw + 0.02).max()


def test_undetected_gene_keeps_slope_zero_and_two_sections_use_the_total():
    m = _means(np.array([-0.02, -0.02]))
    m["s3"] = m["s3"] * np.array([1.0, 0.0])
    s, info = X["depth_slopes"](m, Z)
    assert s[1] == 0.0 and info["n_genes_fitted"] == 1
    two = {k: _means(np.array([-0.02, -0.04]))[k] for k in ("s1", "s3")}
    s2, info2 = X["depth_slopes"](two, {"s1": 19.0, "s3": 41.0})
    assert info2["mode"].startswith("global") and s2[0] == s2[1]


def test_transfer_counts_mean_integers_zeros_and_seed():
    rng = np.random.default_rng(0)
    C = rng.poisson(50, size=(4000, 3)).astype(float)
    C[:100, 0] = 0.0
    r = np.array([0.5, 1.0, 1.6])
    a = X["transfer_counts"](C, r, np.random.default_rng(7), True)
    b = X["transfer_counts"](C, r, np.random.default_rng(7), True)
    assert np.array_equal(a, b) and np.all(a == np.rint(a))
    assert np.all(a[:100, 0] == 0) and np.array_equal(a[:, 1], C[:, 1])
    assert a.mean(0) == pytest.approx(C.mean(0) * r, rel=0.02)
    s = X["transfer_counts"](C + 0.5, r, np.random.default_rng(7), False)
    assert np.allclose(s, (C + 0.5) * r)


# ── evaluate_depth ───────────────────────────────────────────────────────────

D = _lift(BENCH3 / "evaluate_depth.py", {"section_metrics"})


def test_abundance_metrics_see_scale_and_profile():
    rng = np.random.default_rng(1)
    gt = rng.poisson(np.array([5.0, 20.0, 80.0]), size=(500, 3)).astype(float)
    same = D["section_metrics"](gt, gt)
    assert same["lib_logratio"] == 0.0 and same["gene_mean_abs_logratio"] == 0.0
    half = D["section_metrics"](gt * 0.5, gt)
    assert half["lib_logratio"] == pytest.approx(np.log(0.5))
    assert half["gene_mean_abs_logratio"] == pytest.approx(np.log(2.0))
    assert half["gene_mean_log_r"] == pytest.approx(1.0)


# ── extrapolation designs ────────────────────────────────────────────────────

def test_end_of_stack_holdouts():
    labels = [f"section_{i}" for i in range(1, 8)]
    zb = {s: 19.0 + 11 * j for j, s in enumerate(labels)}
    E = _lift(BENCH3 / "run_extrapolation.py", {"extrapolation_designs"},
              {"sorted_sections": lambda adata: (labels, zb)})
    cfgs = {c["holdout_id"]: c for c in E["extrapolation_designs"](None, "both", 3)}
    assert sorted(cfgs) == ["extrap_hi_1", "extrap_hi_2", "extrap_hi_3",
                            "extrap_lo_1", "extrap_lo_2", "extrap_lo_3"]
    assert cfgs["extrap_hi_2"]["holdout_sections"] == ["section_6", "section_7"]
    assert cfgs["extrap_lo_3"]["remaining_sections"] == labels[3:]
    with pytest.raises(ValueError):
        E["extrapolation_designs"](None, "both", 6)
