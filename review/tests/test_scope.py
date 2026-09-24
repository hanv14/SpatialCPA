"""This repository reviews SpatialCPA-v18 against SpatialZ, FEAST and isoST only.
No other SpatialCPA version's wrapper, registry entry, method file or result is
present."""
from __future__ import annotations

import re

from conftest import BENCH3, REVIEW_ROOT, V2_BENCH, load_bench3_config

ALLOWED_V18 = re.compile(r"^(spatialcpav18_gen|v18_[a-z]+)$")
OTHER_VERSION = re.compile(r"spatialcpav(?!18)\d+|learn_spatialcpav(?!18)\d+|v(?:1[0-79]|2\d|[4-9])_[a-z]")


def test_registry_is_exactly_the_review_scope():
    c = load_bench3_config()
    names = set(c.METHODS)
    comparators = {"spatialz", "feast", "isost"}
    assert comparators <= names
    assert all(ALLOWED_V18.match(n) for n in names - comparators), sorted(names - comparators)
    assert set(c.METHOD_ORDER) == names


def test_no_other_version_wrappers_or_method_files():
    wrappers = sorted(p.name for p in (BENCH3 / "methods").glob("run_*.py"))
    assert wrappers == ["run_spatialcpav18.py", "run_spatialcpav18_ml.py"]
    v2 = sorted(p.name for p in (V2_BENCH / "methods").glob("run_*.py"))
    assert v2 == ["run_feast.py", "run_isost.py", "run_spatialz.py"]
    stray = [p for p in REVIEW_ROOT.rglob("*.py")
             if re.match(r"(learn_|run_)?spatialcpav(?!18)\d+", p.name)]
    assert not stray, stray


def test_no_other_version_results_committed():
    for d in ("expected", "benchmark-pbya-v3/results"):
        root = REVIEW_ROOT / d
        if root.exists():
            bad = [p for p in root.rglob("*") if OTHER_VERSION.search(str(p.relative_to(root)))]
            assert not bad, bad
