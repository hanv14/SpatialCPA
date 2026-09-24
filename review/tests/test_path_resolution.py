"""v18's wrapper finds the committed learn_spatialcpav18.py from a fresh clone,
with no environment variables, and never anything outside the repository.

Imports the wrapper in a subprocess (it resolves the file at import time), so
this needs the harness env (numpy/scipy/anndata) but not torch.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from conftest import REVIEW_ROOT, V18_FILE, V18_WRAPPER

PROBE = textwrap.dedent("""
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("w", sys.argv[1])
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    print("PATH", w._V18_PATH)
    print("TRIED", "|".join(w._TRIED))
""")


def _probe(env_overrides=None, cwd=None):
    env = {k: v for k, v in os.environ.items()
           if k not in ("SPATIALCPAV18_FILE", "SPATIALCPAV18_ROOT")}
    env.update(env_overrides or {})
    out = subprocess.run([sys.executable, "-c", PROBE, str(V18_WRAPPER)],
                         env=env, cwd=cwd or "/", capture_output=True, text=True,
                         check=True).stdout
    kv = dict(line.split(" ", 1) for line in out.splitlines()
              if line.startswith(("PATH ", "TRIED ")))
    return kv["PATH"], kv["TRIED"].split("|")


def test_resolves_committed_file_with_no_env_vars():
    path, tried = _probe()
    assert path == str(V18_FILE)
    assert tried == [str(V18_FILE)], "must not look anywhere but the repository root"


def test_resolution_independent_of_cwd(tmp_path):
    path, _ = _probe(cwd=str(tmp_path))
    assert path == str(V18_FILE)


def test_file_override_wins_and_does_not_fall_through(tmp_path):
    missing = tmp_path / "nope.py"
    path, tried = _probe({"SPATIALCPAV18_FILE": str(missing)})
    assert path == "None" and tried == [str(missing)]


def test_root_override(tmp_path):
    (tmp_path / "learn_spatialcpav18.py").write_bytes(V18_FILE.read_bytes())
    path, _ = _probe({"SPATIALCPAV18_ROOT": str(tmp_path)})
    assert path == str(tmp_path / "learn_spatialcpav18.py")


def test_v2_sibling_is_inside_the_repo():
    src = V18_WRAPPER.read_text()
    assert 'parents[4]\n             / "benchmark-pbya-v2"' in src
    assert (REVIEW_ROOT / "benchmark-pbya-v2" / "src" / "benchmark" / "methods" / "_v2_io.py").exists()
