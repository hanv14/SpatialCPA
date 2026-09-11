"""The evaluator this paper's numbers come from is pinned by content hash — checked, not asserted.

``specs/10`` §0 pins ``bench3/evaluate_paper.py`` by SHA-256 and its acceptance list names
``test_evaluate_paper_sha256_unchanged``. The test did not exist: the hash lived in prose in
``specs/10`` and in a ``sha256sum`` command a person was expected to run. Every published number in
this work, and the whole of the paper's §8.2 comparator table, rests on that file being the one the
pin names, so the pin is worth a test rather than a habit.

No data, no GPU, no network, no fit — it reads one file and hashes it (Convention 7).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# `specs/10` §0. Duplicated here on purpose: a test that reads the expected value out of the same
# document it is checking would pass after an edit to either side.
BENCH3_EVALUATE_PAPER_SHA256 = "7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992"
BENCH3_EVALUATE_PAPER_LINES = 764

REPO = Path(__file__).resolve().parent.parent
EVALUATE_PAPER = REPO / "benchmark-pbya-v3" / "src" / "bench3" / "evaluate_paper.py"
SPEC = REPO / "specs" / "10_TASK_benchmark_and_baselines.md"


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file's bytes, read whole — the file is 34 KB."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.skipif(not EVALUATE_PAPER.exists(), reason="bench3 tree not present in this checkout")
def test_evaluate_paper_sha256_unchanged() -> None:
    """The instrument in this checkout is the one `specs/10` §0 pins.

    A mismatch means the metric implementation moved under numbers already reported. The message
    names the file, because a bare hash comparison tells whoever hits it nothing about what to do.
    """
    actual = sha256_of(EVALUATE_PAPER)
    assert actual == BENCH3_EVALUATE_PAPER_SHA256, (
        f"{EVALUATE_PAPER.relative_to(REPO)} is NOT the pinned instrument.\n"
        f"  expected {BENCH3_EVALUATE_PAPER_SHA256}\n"
        f"  actual   {actual}\n"
        "Every published number in this work was measured with the pinned file. Either restore it "
        "or re-measure everything; do not update this constant to make the test pass."
    )


@pytest.mark.skipif(not EVALUATE_PAPER.exists(), reason="bench3 tree not present in this checkout")
def test_pinned_hash_agrees_with_the_spec_that_pins_it() -> None:
    """...and the constant above is the one `specs/10` actually states.

    The duplication in this file is deliberate; this is the test that keeps the duplicate honest.
    Without it, a hash edited in one place and not the other passes everywhere.
    """
    assert SPEC.exists(), "specs/10 is missing, so the pin cannot be cross-checked"
    assert BENCH3_EVALUATE_PAPER_SHA256 in SPEC.read_text(), (
        "the SHA-256 in this test is not the one specs/10 §0 states — one of the two was edited "
        "alone. The spec is the authority; fix the constant here, never the other way round."
    )


@pytest.mark.skipif(not EVALUATE_PAPER.exists(), reason="bench3 tree not present in this checkout")
def test_line_count_matches_the_pin_so_a_mismatch_is_readable() -> None:
    """A second, human-readable descriptor of the same file (`specs/10` §0: "764 lines").

    A hash says *different* and nothing else. A line count says roughly *how* different, which is
    the first thing anyone hitting the failure above wants to know.
    """
    lines = len(EVALUATE_PAPER.read_text().splitlines())
    assert lines == BENCH3_EVALUATE_PAPER_LINES, (
        f"{EVALUATE_PAPER.name} has {lines} lines against the pin's "
        f"{BENCH3_EVALUATE_PAPER_LINES}"
    )


def test_the_hasher_rejects_a_changed_byte() -> None:
    """Positive control: the check has to FAIL on a file that differs, or a green run means nothing.

    Runs without the bench3 tree, so the control holds in a checkout where the three tests above
    skip — which is exactly the checkout where a silent skip would be mistaken for a pass.
    """
    a = hashlib.sha256(b"def score(x):\n    return x\n").hexdigest()
    b = hashlib.sha256(b"def score(x):\n    return x + 1\n").hexdigest()
    assert a != b
    assert len(a) == 64 and set(a) <= set("0123456789abcdef")
