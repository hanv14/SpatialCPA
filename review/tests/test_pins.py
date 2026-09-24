"""Every file that can move a published number is pinned by sha256.

MANIFEST.sha256 lists them (scorer chain, v2 evaluators, the v18 method and its
wrappers, the STARmap source volume). evaluate_paper.py is additionally pinned
here as a literal, so the README, the manifest and this test all state one hash.
"""
from __future__ import annotations

import hashlib

from conftest import BENCH3, REVIEW_ROOT

EVALUATE_PAPER_SHA256 = "7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992"


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest():
    rows = []
    for line in (REVIEW_ROOT / "MANIFEST.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, path = line.split(None, 1)
            rows.append((digest, path.strip()))
    return rows


def test_evaluate_paper_is_the_pinned_file():
    assert _sha(BENCH3 / "evaluate_paper.py") == EVALUATE_PAPER_SHA256


def test_evaluate_paper_pin_matches_manifest_and_readme():
    m = dict((p, d) for d, p in _manifest())
    assert m["benchmark-pbya-v3/src/bench3/evaluate_paper.py"] == EVALUATE_PAPER_SHA256
    assert EVALUATE_PAPER_SHA256 in (REVIEW_ROOT / "README.md").read_text()


def test_every_manifest_entry_matches():
    bad = [(p, d, _sha(REVIEW_ROOT / p)) for d, p in _manifest()
           if not (REVIEW_ROOT / p).exists() or _sha(REVIEW_ROOT / p) != d]
    assert not bad, "files changed since MANIFEST.sha256 was written:\n" + "\n".join(
        f"  {p}\n    manifest {d}\n    on disk  {s}" for p, d, s in bad)


def test_manifest_covers_the_scoring_chain():
    pinned = {p for _, p in _manifest()}
    required = {
        "benchmark-pbya-v3/src/bench3/evaluate_paper.py",
        "benchmark-pbya-v3/src/bench3/align.py",
        "benchmark-pbya-v3/src/bench3/_v2bridge.py",
        "benchmark-pbya-v3/src/bench3/config.py",
        "benchmark-pbya-v3/src/bench3/design.py",
        "benchmark-pbya-v3/src/bench3/prepare_dataset.py",
        "benchmark-pbya-v3/src/bench3/run_benchmark.py",
        "benchmark-pbya-v2/src/benchmark/evaluate.py",
        "benchmark-pbya-v2/src/benchmark/evaluate_generation.py",
        "benchmark-pbya-v2/src/benchmark/leakage_guard.py",
        "benchmark-pbya-v2/src/benchmark/config.py",
        "benchmark-pbya-v2/src/benchmark/methods/_v2_io.py",
        "learn_spatialcpav18.py",
        "benchmark-pbya-v3/src/bench3/methods/run_spatialcpav18.py",
        "data/starmap/STARmap_Wang2018three_data_3D_data.h5ad",
    }
    assert required <= pinned, sorted(required - pinned)


def test_scoring_constants_unchanged():
    """config.py was edited for the review (METHODS trimmed), so its hash differs
    from the published tree. These are the constants evaluate_paper/align read;
    they must equal the published values."""
    from conftest import load_bench3_config
    c = load_bench3_config()
    assert (c.SPATIAL_K, c.FIELD_GRID, c.DEPTH_BINS, c.EMBED_NEIGHBORS) == (10, 20, 20, 15)
    assert c.RARE_CELLTYPE_FRAC == 0.05
    assert (c.ALIGN_ANGLES, c.ALIGN_ICP_ITERS, c.ALIGN_MAX_POINTS) == (24, 12, 3000)
    assert c.RANDOM_SEED == 42
    assert (c.DROP_Z_LOW, c.DROP_Z_HIGH, c.N_SECTIONS) == ((6, 13), (91, 94), 7)
    assert c.HELD_OUT_SECTION_INDICES == (2, 4, 6)
    assert (c.VOXEL_XY_UM, c.VOXEL_Z_UM) == (0.859, 1.0)
    assert c.MARKER_GENES == ("Flt1", "Pcp4", "Cux2")


def test_v2_scoring_constants_unchanged():
    """Same reasoning for benchmark-pbya-v2/src/benchmark/config.py (METHODS trimmed)."""
    import sys
    from conftest import REVIEW_ROOT
    sys.path.insert(0, str(REVIEW_ROOT / "benchmark-pbya-v2" / "src"))
    from benchmark import config as v2
    assert (v2.NN_MATCH_THRESHOLD_UM, v2.SSIM_GRID_SIZE, v2.SSIM_TOP_GENES) == (50.0, 50, 100)
    assert v2.RANDOM_SEED == 42
    assert set(v2.METHODS) == {"spatialz", "feast", "isost"}
