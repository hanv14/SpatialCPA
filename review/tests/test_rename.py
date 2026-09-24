"""SpatialCPA-v18 is a standalone method: no code or text in its path names another
SpatialCPA version, and the rename that removed those names changed nothing that
executes (REVIEW_NOTES §6)."""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys

import pytest

from conftest import BENCH3, REVIEW_ROOT, V18_FILE, load_bench3_config

OTHER_VERSION = re.compile(
    r"\b[vV](?:8|9|1[0-79]|2\d)\b"              # v14, V21, v8 … (not "MATLAB v5")
    r"|V14Config|SpatialCPAv14|\[v14\]"
    r"|\bv(?:1[0-79]|2\d)_[a-z]"                 # v14_ridge …
    r"|[Ss]patial[Cc][Pp][Aa]-?v(?!18)\d+"       # SpatialCPA-v14, spatialcpav21 …
    r"|learn_spatialcpav(?!18)\d+|H3D-FLA|inert_mechanisms")

# The two files whose job is to name the old identifiers: the equivalence checker
# and this test (plus test_scope's guard pattern).
ALLOWED = {"scripts/verify_rename.py", "tests/test_rename.py", "tests/test_scope.py"}

RENAMED = [  # (renamed file, original in the parent project)
    "learn_spatialcpav18.py",
    "benchmark-pbya-v3/src/bench3/methods/run_spatialcpav18_ml.py",
    "benchmark-pbya-v3/src/bench3/methods/_ml_learners.py",
    "benchmark-pbya-v3/src/bench3/design.py",
    "benchmark-pbya-v3/src/bench3/assets.py",
    "benchmark-pbya-v2/src/benchmark/evaluate_generation.py",
    "benchmark-pbya-v2/src/benchmark/leakage_guard.py",
]


def _code_and_docs():
    for p in sorted(REVIEW_ROOT.rglob("*")):
        rel = p.relative_to(REVIEW_ROOT).as_posix()
        if (p.is_file() and p.suffix in (".py", ".md", ".sh", ".yml")
                and not rel.startswith(("reproduced/", "benchmark-pbya-v3/results/",
                                        "benchmark-pbya/tools/"))
                and rel not in ALLOWED and rel != "REVIEW_NOTES.md"):
            yield rel, p


def test_no_other_version_named_anywhere():
    hits = [f"{rel}:{i}: {line.strip()[:100]}"
            for rel, p in _code_and_docs()
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
            if OTHER_VERSION.search(line)]
    assert not hits, "\n".join(hits)


def test_review_notes_names_old_identifiers_only_in_the_rename_record():
    text = (REVIEW_ROOT / "REVIEW_NOTES.md").read_text()
    start = text.index("### The standalone rename")
    end = text.index("Byte-identical to the parent")
    outside = text[:start] + text[end:]
    hits = [line for line in outside.splitlines() if OTHER_VERSION.search(line)]
    assert not hits, hits


def test_api_names():
    src = V18_FILE.read_text()
    assert "class SpatialCPAv18" in src and "class V18Config" in src


def test_fallback_markers_can_match():
    """The harness fails a run whose log shows v18's numpy fallback. After the
    rename those markers must be strings v18 actually prints."""
    c = load_bench3_config()
    src = V18_FILE.read_text()
    for name, m in c.METHODS.items():
        for marker in m.get("invalid_log_markers", ()):
            if marker.startswith("[v18]"):
                assert marker in src, (name, marker)
    for needed in ("[v18] torch unavailable", "[v18] training failed", "[v18] generation failed"):
        assert needed in src, needed


def _originals():
    rows = {}
    for line in (REVIEW_ROOT / "MANIFEST.original.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            d, rel = line.split(None, 1)
            rows[rel.strip()] = d
    return rows


@pytest.mark.parametrize("rel", RENAMED)
def test_renamed_file_is_equivalent_to_original(rel):
    """Runs where the parent project's originals are present (a clone of the full
    project); skipped in a review-only checkout — run verify_rename.py by hand
    against the lab's copies there."""
    original = REVIEW_ROOT.parent / rel
    want = _originals()[rel]
    if not original.exists() or hashlib.sha256(original.read_bytes()).hexdigest() != want:
        pytest.skip(f"original {rel} (sha256 {want[:12]}…) not available beside the review tree")
    out = subprocess.run([sys.executable, str(REVIEW_ROOT / "scripts" / "verify_rename.py"),
                          str(original), str(REVIEW_ROOT / rel)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_verify_rename_rejects_a_real_change(tmp_path):
    mutated = tmp_path / "m.py"
    mutated.write_text(V18_FILE.read_text().replace(
        "ground_temp: float = 0.25", "ground_temp: float = 0.26"))
    out = subprocess.run([sys.executable, str(REVIEW_ROOT / "scripts" / "verify_rename.py"),
                          str(V18_FILE), str(mutated)], capture_output=True, text=True)
    assert out.returncode == 1
