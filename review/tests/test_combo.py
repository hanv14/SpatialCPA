"""The combo method's selection rule (run_spatialcpav18_combo.py): the
signal-to-noise composite respects metric direction and noise scale, and a grid
step is taken only past SNR_MIN, ties staying with flow_cv. Lifted with ``ast``
so the test needs neither torch nor anndata."""
from __future__ import annotations

import ast

import pytest

from conftest import BENCH3

np = pytest.importorskip("numpy")
SRC = BENCH3 / "methods" / "run_spatialcpav18_combo.py"
WANTED = {"SIGMA", "SNR_MIN", "COMBO_W_GRID", "COMBO_BETA_GRID", "snr_score", "choose_step"}
CV_METRICS = (("embedding_mixing_pca", +1), ("morans_pearson", +1), ("morans_mae", -1),
              ("marker_depth_r", +1), ("marker_field_r", +1), ("celltype_localization", +1),
              ("rare_celltype_localization", +1), ("gene_detection_spearman", +1))


class _F:                       # the one name snr_score reads from the flow wrapper
    CV_METRICS = CV_METRICS


def _lift():
    tree = ast.parse(SRC.read_text())
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in WANTED)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in WANTED
                                                  for t in n.targets))]
    ns = {"np": np, "_F": _F}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
    return ns


NS = _lift()


def test_sigma_covers_every_fold_metric_and_corners_are_flow_cv():
    assert set(NS["SIGMA"]) == {k for k, _ in CV_METRICS}
    assert NS["COMBO_W_GRID"][0] == 1.0 and NS["COMBO_BETA_GRID"][0] == 0.0


def test_snr_score_uses_direction_and_noise_scale():
    base = {k: 0.5 for k, _ in CV_METRICS}
    sig = NS["SIGMA"]
    m = dict(base, morans_mae=0.5 - sig["morans_mae"])          # lower MAE: +1 sd
    assert NS["snr_score"](m, base) == pytest.approx(1.0)
    m = dict(base, gene_detection_spearman=0.5 + sig["gene_detection_spearman"])
    assert NS["snr_score"](m, base) == pytest.approx(1.0)       # noisy metric: same 1 sd
    m = dict(base, celltype_localization=None)
    assert NS["snr_score"](m, base) == 0.0


def test_choose_step_needs_more_than_snr_min_and_keeps_flow_cv_on_ties():
    cs = NS["choose_step"]
    assert cs(1.0, {1.0: 0.0, 0.75: NS["SNR_MIN"], 0.5: 0.2}) == 1.0
    assert cs(1.0, {1.0: 0.0, 0.75: 3.0, 0.5: 5.0}) == 0.5
    assert cs(0.0, {0.0: 0.0, 0.25: 2.0, 0.5: 2.0, 1.0: -4.0}) == 0.25
