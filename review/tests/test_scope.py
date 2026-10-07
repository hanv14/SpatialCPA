"""This repository reviews SpatialCPA-v18 against SpatialZ, FEAST and isoST only.
No other SpatialCPA version, and no expression ablation, is present: no wrapper,
registry entry, method file or result. Everything benchmark-related lives under
the one folder ``benchmark/``."""
from __future__ import annotations

import re

from conftest import BENCH3, REVIEW_ROOT, load_bench3_config

OTHER_VERSION = re.compile(r"spatialcpav(?!18)\d+|learn_spatialcpav(?!18)\d+|(?<!spatialcpa)v\d+_[a-z]")
REVIEW_METHODS = {"spatialz", "feast", "isost", "spatialcpav18_gen", "spatialcpav18_gen_flow",
                  "spatialcpav18_gen_flow_cv", "spatialcpav18_gen_flow_patch",
                  "spatialcpav18_gen_flow_patch_random", "spatialcpav18_gen_nearest_noflow",
                  "spatialcpav18_gen_flow_transport", "spatialcpav18_gen_flow_transport_ot",
                  "spatialcpav18_gen_flow_transport_zshuffle",
                  "spatialcpav18_gen_flow_transport_pair", "spatialcpav18_gen_flow_h",
                  "spatialcpav18_gen_flow_h_untrained", "spatialcpav18_gen_flow_h_interp",
                  "spatialcpav18_gen_flow_h_srcdepth", "spatialcpav18_gen_uq",
                  "spatialcpav18_gen_flow_layout", "spatialcpav18_gen_flow_layout_untrained",
                  "spatialcpav18_gen_flow_layout_interp", "spatialcpav18_gen_flow_combo",
                  "spatialcpav18_gen_flow_combo_untrained", "spatialcpav18_gen_flow_combo_noflow",
                  "spatialcpav18_gen_flow_cv_stable"}


def test_registry_is_exactly_the_review_scope():
    c = load_bench3_config()
    assert set(c.METHODS) == REVIEW_METHODS
    assert set(c.METHOD_ORDER) == REVIEW_METHODS


def test_wrappers_are_exactly_the_four_methods():
    wrappers = sorted(p.name for p in (BENCH3 / "methods").glob("run_*.py"))
    assert wrappers == ["run_feast.py", "run_isost.py", "run_spatialcpav18.py",
                        "run_spatialcpav18_combo.py", "run_spatialcpav18_flow.py",
                        "run_spatialcpav18_flowcv_stable.py",
                        "run_spatialcpav18_layout.py",
                        "run_spatialcpav18_uq.py",
                        "run_spatialz.py"]
    stray = [p for p in REVIEW_ROOT.rglob("*.py")
             if re.match(r"(learn_|run_)?spatialcpav(?!18)\d+", p.name) or "_ml" in p.stem]
    assert not stray, stray


def test_one_benchmark_folder():
    assert not any(REVIEW_ROOT.glob("benchmark-pbya*")), "old benchmark folders present"
    assert (REVIEW_ROOT / "benchmark" / "src" / "bench3").is_dir()
    assert (REVIEW_ROOT / "benchmark" / "src" / "benchmark").is_dir()


def test_no_other_version_results_committed():
    for d in ("expected", "benchmark/results"):
        root = REVIEW_ROOT / d
        if root.exists():
            bad = [p for p in root.rglob("*") if OTHER_VERSION.search(str(p.relative_to(root)))]
            assert not bad, bad
