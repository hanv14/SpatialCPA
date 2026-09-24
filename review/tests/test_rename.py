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



def _code_and_docs():
    for p in sorted(REVIEW_ROOT.rglob("*")):
        rel = p.relative_to(REVIEW_ROOT).as_posix()
        if (p.is_file() and p.suffix in (".py", ".md", ".sh", ".yml")
                and not rel.startswith(("reproduced/", "benchmark/results/",
                                        "benchmark/tools/", "benchmark/src/data/"))
                and rel not in ALLOWED and rel != "REVIEW_NOTES.md"):
            yield rel, p


def test_no_other_version_named_anywhere():
    hits = [f"{rel}:{i}: {line.strip()[:100]}"
            for rel, p in _code_and_docs()
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1)
            if OTHER_VERSION.search(line)]
    assert not hits, "\n".join(hits)


def test_review_notes_names_old_identifiers_only_in_the_change_record():
    text = (REVIEW_ROOT / "REVIEW_NOTES.md").read_text()
    start = text.index("## 5. What was changed from the published tree")
    end = text.index("## 6. Open provenance items")
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


def _provenance():
    """Rows of MANIFEST.original.sha256: (sha, original path, path here, status)."""
    rows = []
    for line in (REVIEW_ROOT / "MANIFEST.original.sha256").read_text().splitlines():
        if line and not line.startswith("#"):
            rows.append(tuple(line.split()))
    return rows


CHANGED = {  # the only executable changes from the parent project (REVIEW_NOTES §5)
    "benchmark/src/bench3/_v2bridge.py",
    "benchmark/src/bench3/config.py",
    "benchmark/src/bench3/methods/run_isost.py",
    "benchmark/src/bench3/methods/run_spatialcpav18.py",
    "benchmark/src/bench3/methods/run_spatialz.py",
    "benchmark/src/bench3/run_benchmark.py",
    "benchmark/src/bench3/selftest.py",
    "benchmark/src/bench3/survey_datasets.py",
    "benchmark/src/benchmark/config.py",
}


def test_provenance_covers_every_source_file():
    listed = {here for _, _, here, _ in _provenance()}
    on_disk = {p.relative_to(REVIEW_ROOT).as_posix()
               for p in (REVIEW_ROOT / "benchmark" / "src").rglob("*.py")
               if "__pycache__" not in p.parts}
    assert on_disk <= listed, sorted(on_disk - listed)
    assert "learn_spatialcpav18.py" in listed


def test_changed_files_are_exactly_the_documented_ones():
    changed = {here for _, _, here, st in _provenance() if st == "changed"}
    assert changed == CHANGED


@pytest.mark.parametrize("row", [r for r in _provenance() if r[3] != "changed"],
                         ids=lambda r: r[2])
def test_file_matches_its_original(row):
    """identical: same bytes. equivalent: same AST (verify_rename.py). Checked
    against the parent project's original when it sits beside this tree (a clone
    of the full project); skipped in a review-only checkout — run
    verify_rename.py by hand against the lab's copies there."""
    want, orig_rel, here, status = row
    original = REVIEW_ROOT.parent / orig_rel
    if not original.exists() or hashlib.sha256(original.read_bytes()).hexdigest() != want:
        pytest.skip(f"original {orig_rel} (sha256 {want[:12]}…) not available")
    if status == "identical":
        assert original.read_bytes() == (REVIEW_ROOT / here).read_bytes()
        return
    out = subprocess.run([sys.executable, str(REVIEW_ROOT / "scripts" / "verify_rename.py"),
                          str(original), str(REVIEW_ROOT / here)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout


def test_verify_rename_rejects_a_real_change(tmp_path):
    mutated = tmp_path / "m.py"
    mutated.write_text(V18_FILE.read_text().replace(
        "ground_temp: float = 0.25", "ground_temp: float = 0.26"))
    out = subprocess.run([sys.executable, str(REVIEW_ROOT / "scripts" / "verify_rename.py"),
                          str(V18_FILE), str(mutated)], capture_output=True, text=True)
    assert out.returncode == 1
