"""Shared paths for the review tests. Stdlib only unless a test says otherwise."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REVIEW_ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = REVIEW_ROOT / "benchmark"
BENCH3 = BENCH_ROOT / "src" / "bench3"
SHARED = BENCH_ROOT / "src" / "benchmark"
V18_FILE = REVIEW_ROOT / "learn_spatialcpav18.py"
V18_WRAPPER = BENCH3 / "methods" / "run_spatialcpav18.py"


def load_bench3_config():
    """bench3/config.py imports only pathlib/os, so it loads without the env."""
    if str(BENCH_ROOT) not in sys.path:
        sys.path.insert(0, str(BENCH_ROOT))
    import src.bench3.config as cfg  # noqa: E402
    return cfg


def load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
