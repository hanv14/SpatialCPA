"""`specs/10` §9's re-slicing preprocessor. Pure geometry; no torch, no data, no network."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from spatialcpav25_gen.eval.resection import (
    MIN_CELLS_PER_SECTION,
    ResectionError,
    plan_partition,
)

SEED = 20250816


@pytest.fixture
def block() -> np.ndarray:
    """A real 3-D point cloud: what bench3 holds at BUILD time, 200 um deep."""
    rng = np.random.default_rng(SEED)
    return np.column_stack([rng.uniform(0, 1000, (14000, 2)), rng.uniform(0, 200, 14000)])


@pytest.fixture
def already_built() -> np.ndarray:
    """A BUILT bench3 volume: 7 slabs, one depth each. The case V4a actually meets.

    `load_volume` raises unless every section spans exactly one depth, so any file that loads has
    already been quantised this way and its within-slab depth is gone.
    """
    rng = np.random.default_rng(SEED)
    centres = np.linspace(14.3, 185.7, 7)
    return np.column_stack([rng.uniform(0, 1000, (14000, 2)), np.repeat(centres, 2000)])


def test_a_built_volume_CANNOT_be_repartitioned_and_the_refusal_says_why(already_built):
    """**The load-bearing test, and the §9 check's actual answer.**

    Cutting a 7-depth file into 14 slabs leaves 7 of them empty. That looks like a measurement and
    is an artefact, so it is refused — and the message has to name the count, or the next person
    reads the refusal as a bug in the partitioner rather than a fact about the file.
    """
    with pytest.raises(ResectionError) as exc:
        plan_partition(already_built, (0, 0, 1), 14)
    message = str(exc.value)
    assert "only 7 distinct depths" in message
    assert "EMPTY" in message
    assert "build" in message.lower(), "and it must name the remedy: re-cut at BUILD time"


def test_it_refuses_even_the_SAME_slab_count_because_that_is_not_a_re_cut(already_built):
    with pytest.raises(ResectionError, match="only 7 distinct depths"):
        plan_partition(already_built, (0, 0, 1), 7)


def test_a_true_point_cloud_partitions_and_halving_the_count_halves_the_width(block):
    """V4a's manipulation, on the cloud bench3 has before it builds."""
    seven, fourteen = plan_partition(block, (0, 0, 1), 7), plan_partition(block, (0, 0, 1), 14)
    width7 = float(seven.edges[1] - seven.edges[0])
    width14 = float(fourteen.edges[1] - fourteen.edges[0])
    assert width14 == pytest.approx(0.5 * width7, rel=1e-12)
    assert seven.counts.sum() == fourteen.counts.sum() == block.shape[0], "no cell lost or gained"
    assert fourteen.labels.min() == 0 and fourteen.labels.max() == 13, "every slab is populated"


def test_the_deepest_cell_lands_in_the_last_slab_not_a_phantom_one(block):
    """`np.digitize` puts the maximum past the last edge; folding it back is not optional."""
    plan = plan_partition(block, (0, 0, 1), 9)
    deepest = int(np.argmax(block[:, 2]))
    assert plan.labels[deepest] == 8
    assert plan.counts.sum() == block.shape[0]


def test_slabs_under_bench3s_own_floor_are_refused_by_name(block):
    with pytest.raises(ResectionError) as exc:
        plan_partition(block, (0, 0, 1), 400)
    assert str(MIN_CELLS_PER_SECTION) in str(exc.value)
    assert "would not be scored" in str(exc.value)


def test_an_oblique_normal_conserves_every_cell(block):
    plan = plan_partition(block, (0, 1, 1), 5)
    assert plan.counts.sum() == block.shape[0]
    assert np.all(plan.counts > 0)


def test_the_normal_need_not_be_a_unit_vector(block):
    a = plan_partition(block, (0.0, 0.0, 1.0), 7)
    b = plan_partition(block, (0.0, 0.0, 37.5), 7)
    assert np.array_equal(a.labels, b.labels)
    assert np.allclose(a.edges * 37.5, b.edges)


def test_it_is_deterministic(block):
    a, b = plan_partition(block, (0, 0, 1), 14), plan_partition(block, (0, 0, 1), 14)
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.edges, b.edges)


def test_it_refuses_the_shapes_it_cannot_read(block):
    with pytest.raises(ResectionError, match=r"\(N, 3\)"):
        plan_partition(block[:, :2], (0, 0, 1), 7)
    with pytest.raises(ResectionError, match="at least 2"):
        plan_partition(block, (0, 0, 1), 1)
    with pytest.raises(ResectionError, match="non-zero"):
        plan_partition(block, (0, 0, 0), 7)


def test_the_source_digest_helper_reads_the_file_it_is_given(tmp_path):
    """§9 requires the source be left byte-identical; that check is only as good as its digest."""
    from spatialcpav25_gen.eval.resection import _sha256

    path = tmp_path / "x.bin"
    path.write_bytes(b"spatialcpa" * 1000)
    assert _sha256(path) == hashlib.sha256(b"spatialcpa" * 1000).hexdigest()
