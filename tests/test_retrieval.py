"""T04 — the retrieval index, its three-term score, and the cross-attention.

The spec's "Other acceptance tests" for `model/retrieval.py`. The two that matter most are
adversarial rather than descriptive:

* ``test_retrieval_excludes_source_section`` and its partner in ``tests/gate2_criteria.py``
  (``test_source_section_exclusion_changes_oblique_R2``). The second exists because the
  first cannot fail if the exclusion is quietly removed from the *gate's* call site while
  remaining in the API. If it ever passes with no difference, the exclusion is not plumbed
  through (SPEC_QUESTIONS C1a).
* ``test_relative_position_only``. Scoped, per SPEC_QUESTIONS B4, to the retrieval branch's
  neighbour encoding: the property cannot hold end-to-end, because the GRF is a function of
  absolute position by construction (T03) - but this is exactly where absolute-coordinate
  leakage would appear.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import HeldOutSections, Volume
from spatialcpav25_gen.model.retrieval import (
    PAD_INDEX,
    EmptyCandidatePoolWarning,
    RetrievalAttention,
    RetrievalIndex,
    attention_entropy,
)

from tests.conftest import copy_section, rebuild_volume

QUERY_CELLS = 400
QUERY_SEED = 5


@pytest.fixture(scope="module")
def index(cfg: Config, volume: Volume) -> RetrievalIndex:
    """A retrieval index over the whole synthetic volume."""
    return RetrievalIndex(volume, cfg)


@pytest.fixture(scope="module")
def query_cells(index: RetrievalIndex) -> tuple[np.ndarray, np.ndarray]:
    """A seeded subsample of the index's own cells, with their source-section codes."""
    gen = np.random.default_rng(QUERY_SEED)
    pick = gen.choice(index.n_cells, size=QUERY_CELLS, replace=False)
    return index.coords[pick], index.section_index[pick]


# --------------------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------------------


def test_index_shapes(index: RetrievalIndex, cfg: Config, volume: Volume) -> None:
    assert index.n_cells == volume.n_cells
    assert index.n_sections == volume.n_sections
    assert index.coords.shape == (volume.n_cells, 3)
    assert index.expr_pca.shape == (volume.n_cells, cfg.expr_pca_dim)
    assert index.niche.shape == (volume.n_cells, cfg.niche_n_scales * len(volume.celltype_names))
    assert index.token_dim == cfg.expr_pca_dim + len(volume.celltype_names) + 3 + 4


def test_index_refuses_heldout_sections(cfg: Config, volume: Volume) -> None:
    """Held-out sections are not conditioning evidence; the type system says so."""
    heldout = HeldOutSections(
        sections=list(volume.sections[:2]),
        gene_names=list(volume.gene_names),
        celltype_names=list(volume.celltype_names),
        region_names=None if volume.region_names is None else list(volume.region_names),
        specimen_id=volume.specimen_id,
    )
    with pytest.raises(TypeError, match="Volume"):
        RetrievalIndex(heldout, cfg)  # type: ignore[arg-type]


def test_query_is_deterministic(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    xyz, owner = query_cells
    first = index.query(xyz, set(), seed=3, source_section=owner, apply_dropout=True)
    second = index.query(xyz, set(), seed=3, source_section=owner, apply_dropout=True)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    other = index.query(xyz, set(), seed=4, source_section=owner, apply_dropout=True)
    assert not np.array_equal(first[0], other[0])


def test_query_weights_are_a_distribution(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    xyz, owner = query_cells
    idx, weights = index.query(xyz, set(), seed=0, source_section=owner)
    assert idx.shape == weights.shape == (QUERY_CELLS, index.cfg.retrieval_k)
    assert (weights >= 0.0).all()
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert (weights[idx == PAD_INDEX] == 0.0).all()


def test_retrieval_excludes_holdout(index: RetrievalIndex, volume: Volume) -> None:
    """``exclude_z`` is honoured: no cell of a held-out section is ever returned."""
    _, heldout = split_holdout(volume, "alternating", 0)
    held_z = {float(z) for z in heldout.z_values}
    held_sections = {
        i for i, z in enumerate(index.section_z) if any(abs(z - h) < 1e-6 for h in held_z)
    }
    assert held_sections, "the fixture must have held-out sections for this test to mean anything"

    gen = np.random.default_rng(1)
    lo, hi = volume.bbox[0].astype(np.float64), volume.bbox[1].astype(np.float64)
    xyz = lo[None, :] + gen.random((256, 3)) * (hi - lo)[None, :]
    idx, _ = index.query(xyz, held_z, seed=0)
    returned = index.section_index[idx[idx >= 0]]
    assert not set(np.unique(returned)) & held_sections


def test_query_rejects_an_exclusion_that_matches_nothing(index: RetrievalIndex) -> None:
    """An exclusion naming a depth no section sits at is a leak, not a no-op."""
    with pytest.raises(ValueError, match="not a section"):
        index.query(index.coords[:8], {12345.0}, seed=0)


def test_retrieval_excludes_source_section(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """With the flag set, no returned neighbour shares the query cell's ``section_id``."""
    xyz, owner = query_cells
    idx, _ = index.query(xyz, set(), seed=0, source_section=owner)
    mask = idx >= 0
    returned = index.section_index[np.where(mask, idx, 0)]
    assert not bool(((returned == owner[:, None]) & mask).any()), (
        "a neighbour came from the query cell's own section: the leave-own-section-out "
        "contract GATE 2 rests on is not being applied"
    )


def test_source_section_exclusion_can_be_turned_off(
    cfg: Config, volume: Volume, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """The off setting must actually return own-section neighbours - and much nearer ones.

    The fast half of the pair described in this module's docstring; the slow half
    (``test_source_section_exclusion_changes_oblique_R2``) measures what it does to the
    gate's own number.
    """
    xyz, owner = query_cells
    permissive = RetrievalIndex(volume, cfg.replace(retrieval_exclude_source_section=False))
    idx, _ = permissive.query(xyz, set(), seed=0, source_section=owner)
    mask = idx >= 0
    returned = permissive.section_index[np.where(mask, idx, 0)]
    own = ((returned == owner[:, None]) & mask).mean()
    assert own > 0.2, f"only {own:.1%} of neighbours came from the query's own section"

    strict_idx, _ = RetrievalIndex(volume, cfg).query(xyz, set(), seed=0, source_section=owner)
    nearest_permissive = _mean_nearest_distance(permissive, xyz, idx)
    nearest_strict = _mean_nearest_distance(permissive, xyz, strict_idx)
    assert nearest_permissive < 0.5 * nearest_strict, (
        "own-section neighbours are supposed to be the trivially near ones; if they are "
        "not, the exclusion cannot be doing what GATE 2 needs it to do"
    )


def _mean_nearest_distance(index: RetrievalIndex, xyz: np.ndarray, idx: np.ndarray) -> float:
    """Mean 3D distance to the top-ranked returned neighbour."""
    first = idx[:, 0]
    ok = first >= 0
    delta = index.coords[first[ok]] - xyz[ok]
    return float(np.linalg.norm(delta, axis=1).mean())


def test_z_proximity_term_changes_the_ranking(
    cfg: Config, volume: Volume, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """``retrieval_w_z = 0`` (ablation A5) must measurably prefer more distant sections.

    This is the competing method's omission, reproduced exactly: a score that cannot see z
    treats the near section and the far one as interchangeable.
    """
    xyz, owner = query_cells
    with_z = RetrievalIndex(volume, cfg)
    without_z = RetrievalIndex(volume, cfg.replace(retrieval_w_z=0.0))
    gaps = []
    for index in (with_z, without_z):
        idx, _ = index.query(xyz, set(), seed=0, source_section=owner)
        mask = idx >= 0
        z = index.section_z[index.section_index[np.where(mask, idx, 0)]]
        gaps.append(float(np.abs(z - xyz[:, 2:3])[mask].mean()))
    assert gaps[1] > gaps[0], (
        f"dropping the z term did not move the donors further away in z ({gaps[1]:.2f} um "
        f"vs {gaps[0]:.2f} um); the term is not in the score"
    )


def test_section_dropout_widens_the_gap(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """The gap-aware curriculum drops the nearest section, so evidence gets remoter."""
    xyz, owner = query_cells
    baseline, _ = index.query(xyz, set(), seed=0, source_section=owner)
    dropped, _ = index.query(xyz, set(), seed=0, source_section=owner, apply_dropout=True)
    gaps = []
    for idx in (baseline, dropped):
        mask = idx >= 0
        z = index.section_z[index.section_index[np.where(mask, idx, 0)]]
        gaps.append(float(np.abs(z - xyz[:, 2:3])[mask].mean()))
    assert gaps[1] > gaps[0]
    assert not np.array_equal(baseline, dropped)


def test_section_dropout_is_off_by_default(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """An evaluation path must not randomise itself just by having a seed."""
    xyz, owner = query_cells
    a, _ = index.query(xyz, set(), seed=0, source_section=owner)
    b, _ = index.query(xyz, set(), seed=999, source_section=owner)
    assert np.array_equal(a, b)


def test_empty_candidate_pool_warns_rather_than_crashing(cfg: Config, volume: Volume) -> None:
    narrow = RetrievalIndex(volume, cfg.replace(retrieval_z_window=0.01))
    xyz = volume.bbox.mean(axis=0).astype(np.float64)[None, :] + np.array([[0.0, 0.0, 20.0]])
    with pytest.warns(EmptyCandidatePoolWarning, match="no admissible donor"):
        idx, weights = narrow.query(xyz, set(), seed=0)
    assert (idx == PAD_INDEX).all()
    assert (weights == 0.0).all()


# --------------------------------------------------------------------------------------
# the niche
# --------------------------------------------------------------------------------------


def test_niche_density_adaptive(cfg: Config, volume: Volume) -> None:
    """Doubling every coordinate leaves the niche vectors unchanged.

    The radius is the distance to the k-th nearest neighbour, not a fixed micrometre value,
    so the niche is a statement about ranks - and ranks are exactly scale-invariant. This is
    what lets the same niche transfer between datasets whose cell densities differ by an
    order of magnitude.
    """
    doubled = rebuild_volume(
        volume,
        [
            copy_section(s, coords=(s.coords * 2.0).astype(np.float32), z=s.z * 2.0)
            for s in volume.sections
        ],
    )
    original = RetrievalIndex(volume, cfg)
    scaled = RetrievalIndex(doubled, cfg)
    assert np.abs(original.niche - scaled.niche).max() < 1e-12


def test_niche_rows_are_unit_norm(index: RetrievalIndex) -> None:
    norms = np.linalg.norm(index.niche, axis=1)
    assert np.allclose(norms, 1.0)


def test_niche_is_spatially_structured(index: RetrievalIndex, cfg: Config) -> None:
    """Neighbouring cells must have more similar niches than random pairs.

    Otherwise the score's third term is noise, and the whole density-adaptive construction
    is decoration.
    """
    gen = np.random.default_rng(9)
    pick = gen.choice(index.n_cells, size=1000, replace=False)
    idx, _ = index.query(index.coords[pick], set(), seed=0)
    neighbour = idx[:, 0]
    ok = neighbour >= 0
    near = float((index.niche[pick[ok]] * index.niche[neighbour[ok]]).sum(axis=1).mean())
    shuffled = gen.permutation(index.n_cells)[: int(ok.sum())]
    far = float((index.niche[pick[ok]] * index.niche[shuffled]).sum(axis=1).mean())
    assert near > far + 0.05, f"niche cosine: neighbours {near:.3f} vs random {far:.3f}"


# --------------------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------------------


def test_relative_position_only(cfg: Config, volume: Volume) -> None:
    """Translating the whole volume by a constant leaves the neighbour tokens unchanged.

    Scoped to the retrieval branch's neighbour encoding (SPEC_QUESTIONS B4): the property
    cannot hold end-to-end, because the GRF is a function of absolute position by
    construction. This is where absolute-coordinate leakage would appear, and where it
    would do the damage - a model that has memorised "cells at z = 250 look like this" has
    nothing to say at a z no section sits at.
    """
    shift = np.array([1234.5, -678.25, 900.0])
    moved = rebuild_volume(
        volume,
        [
            copy_section(
                s,
                coords=(s.coords + shift[None, :2].astype(np.float32)).astype(np.float32),
                z=s.z + float(shift[2]),
            )
            for s in volume.sections
        ],
    )
    here = RetrievalIndex(volume, cfg)
    there = RetrievalIndex(moved, cfg)

    gen = np.random.default_rng(4)
    pick = gen.choice(here.n_cells, size=256, replace=False)
    xyz = here.coords[pick]
    owner = here.section_index[pick]
    idx_a, w_a = here.query(xyz, set(), seed=0, source_section=owner)
    idx_b, w_b = there.query(xyz + shift[None, :], set(), seed=0, source_section=owner)
    assert np.array_equal(idx_a, idx_b), "translating the volume changed which cells were retrieved"
    # 1e-3, not zero: Section.coords is float32 (Convention 4), so a 1234.5 um translation
    # is not exactly representable and the in-plane distances move by ~1e-4 um. The weights
    # themselves are ~1e-1, so this is three orders below the quantity being compared, while
    # a token carrying an absolute coordinate would differ by ~1e3.
    assert np.abs(w_a - w_b).max() < 1e-3

    tokens_a, mask_a = here.neighbour_tokens(xyz, idx_a)
    tokens_b, mask_b = there.neighbour_tokens(xyz + shift[None, :], idx_b)
    assert torch.equal(mask_a, mask_b)
    assert float((tokens_a - tokens_b).abs().max()) < 1e-3, (
        "the neighbour tokens moved with the volume: they are carrying absolute positions"
    )


def test_tokens_are_zero_on_padded_slots(cfg: Config, volume: Volume) -> None:
    narrow = RetrievalIndex(volume, cfg.replace(retrieval_z_window=0.6))
    gen = np.random.default_rng(6)
    pick = gen.choice(narrow.n_cells, size=128, replace=False)
    idx, _ = narrow.query(
        narrow.coords[pick], set(), seed=0, source_section=narrow.section_index[pick]
    )
    tokens, mask = narrow.neighbour_tokens(narrow.coords[pick], idx)
    assert bool((~mask).any()), "this test needs some padded slots to mean anything"
    assert float(tokens[~mask].abs().max()) == 0.0


def test_token_layout_carries_delta_z_in_spacing_units(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """The last token column is ``(z_j - z_p) / median_spacing`` - integers, for real cells."""
    xyz, owner = query_cells
    idx, _ = index.query(xyz, set(), seed=0, source_section=owner)
    tokens, mask = index.neighbour_tokens(xyz, idx)
    delta_z = tokens[:, :, -1][mask].numpy()
    assert np.abs(delta_z - np.round(delta_z)).max() < 1e-4
    assert np.abs(delta_z).max() >= 1.0, "every donor is in the query's own section"


# --------------------------------------------------------------------------------------
# the attention
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def attention(cfg: Config, index: RetrievalIndex) -> RetrievalAttention:
    return RetrievalAttention(cfg, q_dim=cfg.field_dim, token_dim=index.token_dim)


def test_attention_shapes_and_determinism(
    cfg: Config,
    index: RetrievalIndex,
    attention: RetrievalAttention,
    query_cells: tuple[np.ndarray, np.ndarray],
) -> None:
    xyz, owner = query_cells
    idx, _ = index.query(xyz, set(), seed=0, source_section=owner)
    tokens, mask = index.neighbour_tokens(xyz, idx)
    q = torch.zeros((QUERY_CELLS, cfg.field_dim))
    ctx, weights = attention.attend(q, tokens, mask)
    assert ctx.shape == (QUERY_CELLS, cfg.retrieval_ctx_dim)
    assert weights.shape == (QUERY_CELLS, cfg.retrieval_n_heads, cfg.retrieval_k)
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))

    twin = RetrievalAttention(cfg, q_dim=cfg.field_dim, token_dim=index.token_dim)
    assert torch.equal(twin(q, tokens, mask), ctx)


def test_attention_ignores_masked_neighbours(
    cfg: Config, index: RetrievalIndex, attention: RetrievalAttention
) -> None:
    """A masked slot contributes exactly nothing, whatever is written into it."""
    gen = torch.Generator().manual_seed(2)
    tokens = torch.randn((16, cfg.retrieval_k, index.token_dim), generator=gen)
    mask = torch.ones((16, cfg.retrieval_k), dtype=torch.bool)
    mask[:, -4:] = False
    q = torch.randn((16, cfg.field_dim), generator=gen)
    baseline = attention(q, tokens * mask[:, :, None], mask)

    poisoned = tokens.clone()
    poisoned[:, -4:] = 1e3
    assert torch.allclose(attention(q, poisoned, mask), baseline, atol=1e-5)


def test_attention_survives_a_fully_masked_row(
    cfg: Config, index: RetrievalIndex, attention: RetrievalAttention
) -> None:
    """No donor at all gives the value bias, not NaN - one such point must not poison a batch."""
    tokens = torch.zeros((4, cfg.retrieval_k, index.token_dim))
    mask = torch.zeros((4, cfg.retrieval_k), dtype=torch.bool)
    ctx, weights = attention.attend(torch.zeros((4, cfg.field_dim)), tokens, mask)
    assert torch.isfinite(ctx).all()
    assert float(weights.abs().max()) == 0.0


def test_attention_entropy_bounds(cfg: Config) -> None:
    k = cfg.retrieval_k
    uniform = torch.full((3, 1, k), 1.0 / k)
    assert float(attention_entropy(uniform).mean()) == pytest.approx(np.log(k), abs=1e-6)
    one_hot = torch.zeros((3, 1, k))
    one_hot[:, :, 0] = 1.0
    assert float(attention_entropy(one_hot).mean()) == pytest.approx(0.0, abs=1e-6)


def test_attention_rejects_mismatched_shapes(
    cfg: Config, index: RetrievalIndex, attention: RetrievalAttention
) -> None:
    tokens = torch.zeros((8, cfg.retrieval_k, index.token_dim))
    mask = torch.ones((8, cfg.retrieval_k), dtype=torch.bool)
    with pytest.raises(ValueError, match="disagree about N or K"):
        attention(torch.zeros((7, cfg.field_dim)), tokens, mask)


# --------------------------------------------------------------------------------------
# the slow half of the own-section-exclusion pair (SPEC_QUESTIONS C1a)
# --------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.gate
def test_source_section_exclusion_changes_oblique_R2(  # noqa: N802 - the spec names it
    cfg: Config, gate_volume: Volume
) -> None:
    """With the exclusion **off**, R^2 at 90 degrees rises measurably on the gate fixture.

    This is the test that stops the exclusion from being quietly dropped later. Its partner,
    ``test_retrieval_excludes_source_section``, only checks the API: it would still pass if
    someone removed ``source_section=...`` from the *gate's* call site. This one measures the
    gate's own number both ways, so if it ever passes with no difference, the exclusion is
    not plumbed through the candidate filter and GATE 2 is measuring nothing.
    """
    from tests.gate2_criteria import measure_exclusion_effect

    effect = measure_exclusion_effect(cfg, gate_volume, seed=0)
    assert effect["delta"] > 0.0, (
        f"turning the own-section exclusion off changed R^2(90 deg) by {effect['delta']:+.4f} "
        f"({effect['strict']:.4f} -> {effect['permissive']:.4f}). It should rise: an "
        "own-section neighbour a few micrometres away is a near-copy of the query cell. No "
        "change means the exclusion is not reaching the candidate filter."
    )
