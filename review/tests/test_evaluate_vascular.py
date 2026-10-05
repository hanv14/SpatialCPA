"""evaluate_vascular.py behaves as a metric should, on a synthetic section with
known vessels: the ground truth scores 1, any output scale scores the same, a
spatial scramble scores near 0, and a dataset without declared markers is "not
applicable" rather than silently scored. Needs numpy/scipy/anndata (bench_eval)."""
from __future__ import annotations

import sys

import pytest

from conftest import BENCH_ROOT

np = pytest.importorskip("numpy")
ad = pytest.importorskip("anndata")

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))
from src.bench3 import evaluate_vascular as ev  # noqa: E402

GENES = ["Vmark", "Niche", "Grad", "R1", "R2", "R3"]


def synthetic_section(seed=0, n_side=45):
    """Cells on a jittered grid; 'vessels' are the cells near two curves.
    Vmark is high on vessels, Niche is high next to them, Grad is a gradient."""
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(np.arange(n_side), np.arange(n_side)), -1).reshape(-1, 2)
    xy = (g + rng.normal(0, 0.2, g.shape)) * 10.0
    d1 = np.abs(xy[:, 1] - (200 + 80 * np.sin(xy[:, 0] / 70)))
    d2 = np.abs(xy[:, 0] - (300 + 60 * np.cos(xy[:, 1] / 90)))
    dv = np.minimum(d1, d2)
    vessel = dv < 8
    n = len(xy)
    X = np.column_stack([
        np.where(vessel, rng.lognormal(5, 0.3, n), rng.lognormal(2, 0.5, n)),
        np.exp(-dv / 30) * 50 + rng.lognormal(0, 0.5, n),
        xy[:, 0] / 10 + rng.normal(0, 1, n),
        rng.lognormal(1, 1, n), rng.lognormal(1, 1, n), rng.lognormal(1, 1, n),
    ]).clip(0)
    types = np.where(vessel, "vasc", np.where(xy[:, 0] < 225, "left", "right"))
    return X, xy, types


def _rule(types):
    return {"applicable": True, "markers": ["Vmark"], "vascular_type": "vasc",
            "frac": float((types == "vasc").mean())}


def test_ground_truth_scores_one():
    X, xy, t = synthetic_section()
    m = ev.section_metrics(X, xy, t, X, xy, t, GENES, _rule(t), markers=["Vmark", "Grad"])
    for k in ("vasc_dist", "vasc_nn", "vasc_niche", "vasc_niche_r",
              "vasc_type_frac_ratio", "vasc_type_dist"):
        assert m[k] == pytest.approx(1.0), (k, m[k])
    assert m["vasc_field_r"] > 0.99
    assert m["vasc_dist_w1"] == 0.0 and m["vasc_dist_w1_null"] > 0


def test_output_scale_does_not_matter():
    X, xy, t = synthetic_section()
    a = ev.section_metrics(X, xy, t, X, xy, t, GENES, _rule(t), markers=["Vmark", "Grad"])
    b = ev.section_metrics(np.log1p(X) * 3.0, xy, t, X, xy, t, GENES, _rule(t),
                           markers=["Vmark", "Grad"])
    for k in ("vasc_dist", "vasc_nn", "vasc_niche", "vasc_niche_r", "vasc_field_r"):
        assert a[k] == pytest.approx(b[k]), k


def test_spatial_scramble_scores_near_zero():
    X, xy, t = synthetic_section()
    perm = np.random.default_rng(1).permutation(len(xy))
    m = ev.section_metrics(X, xy[perm], t, X, xy, t, GENES, _rule(t),
                           markers=["Vmark", "Grad"])
    assert abs(m["vasc_dist"]) < 0.3 and abs(m["vasc_nn"]) < 0.3
    assert abs(m["vasc_niche"]) < 0.3


def test_real_neighbour_section_scores_between():
    X, xy, t = synthetic_section(seed=0)
    Xn, xyn, tn = synthetic_section(seed=3)          # same tissue, other cells
    m = ev.section_metrics(Xn, xyn, tn, X, xy, t, GENES, _rule(t), markers=["Vmark", "Grad"])
    assert 0.5 < m["vasc_dist"] < 1.0 and 0.5 < m["vasc_niche"] < 1.0


def _adata(types=True):
    X, xy, t = synthetic_section()
    a = ad.AnnData(X.astype(np.float32))
    a.var_names = GENES
    if types:
        a.obs["cell_type"] = t
    return a


def test_undeclared_dataset_is_not_applicable():
    r = ev.vascular_rule(_adata(), "some_other_dataset")
    assert r["applicable"] is False and "VASCULAR_SPEC" in r["reason"]


def test_markers_missing_from_panel_are_not_applicable(monkeypatch):
    monkeypatch.setitem(ev.VASCULAR_SPEC, "toy", {"markers": ["Pecam1"]})
    r = ev.vascular_rule(_adata(), "toy")
    assert r["applicable"] is False and "Pecam1" in r["reason"]


def test_rule_picks_the_marker_enriched_type_and_its_fraction(monkeypatch):
    monkeypatch.setitem(ev.VASCULAR_SPEC, "toy", {"markers": ["Vmark"]})
    a = _adata()
    r = ev.vascular_rule(a, "toy")
    assert r["applicable"] and r["vascular_type"] == "vasc"
    assert r["frac"] == pytest.approx(float((a.obs["cell_type"] == "vasc").mean()))
    monkeypatch.setitem(ev.VASCULAR_SPEC, "toy_untyped", {"markers": ["Vmark"]})
    r = ev.vascular_rule(_adata(types=False), "toy_untyped")
    assert r["applicable"] is False and "frac" in r["reason"]     # no silent default
