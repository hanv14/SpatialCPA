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
    InertScoreWarning,
    RetrievalAttention,
    RetrievalIndex,
    attention_entropy,
)

from tests.conftest import copy_section, rebuild_volume

QUERY_CELLS = 400
QUERY_SEED = 5


def _volume_with_a_thin_section(volume: Volume, *, keep: int) -> Volume:
    """The fixture with its first section cut down to ``keep`` cells."""
    first = volume.sections[0]
    thin = copy_section(
        first,
        coords=first.coords[:keep].copy(),
        cell_type=first.cell_type[:keep].copy(),
        region=None if first.region is None else first.region[:keep].copy(),
        counts=first.counts[:keep].copy(),
    )
    return rebuild_volume(volume, [thin, *volume.sections[1:]])


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
    """An empty pool must be survivable — but it can no longer be *caused* by the window.

    This test used to starve the pool with ``retrieval_z_window=0.01``. Under the two-term
    bound that is impossible by construction: ``retrieval_z_window_gap_factor >= 1`` admits
    the query's nearest surviving section at any gap, and
    ``test_gap_relative_window_never_starves_an_irregular_stack`` is the test that says so.
    What is left is the exclusions, so this empties the pool the way the pipeline actually
    can: ``exclude_z`` leaves one section standing and the query's own-section exclusion
    removes that one too.
    """
    index = RetrievalIndex(volume, cfg)
    rows = np.nonzero(index.section_index == 0)[0][:16]
    others = {float(s.z) for s in volume.sections[1:]}
    with pytest.warns(EmptyCandidatePoolWarning, match="no admissible donor"):
        idx, weights = index.query(
            index.coords[rows], others, seed=0, source_section=index.section_index[rows]
        )
    assert (idx == PAD_INDEX).all()
    assert (weights == 0.0).all()


def test_a_narrow_absolute_window_no_longer_empties_the_pool(cfg: Config, volume: Volume) -> None:
    """The negative of the test above, kept separate because it is the actual fix.

    ``retrieval_z_window=0.01`` is 0.5 um on this fixture — far narrower than any section
    gap. Before the relative term this returned nothing; now it returns the nearest
    section, because a query's own gap is the floor on its own window.
    """
    narrow = RetrievalIndex(volume, cfg.replace(retrieval_z_window=0.01))
    xyz = volume.bbox.mean(axis=0).astype(np.float64)[None, :] + np.array([[0.0, 0.0, 20.0]])
    idx, _ = narrow.query(xyz, set(), seed=0)
    assert (idx != PAD_INDEX).any(), "the nearest section must be admissible at any gap"


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
    """Padding is produced by a *small* donor section, not by a starved window.

    This test used to get its padded slots from ``retrieval_z_window=0.6``, which on this
    fixture admitted nothing at all — so every slot was padded and the assertion held
    trivially. The two-term window makes that unreachable, and the honest source of partial
    padding is a section with fewer than ``retrieval_k`` cells in it: the union is then
    genuinely smaller than K with real donors in it, which is the case the zeroing is for.
    """
    thin = _volume_with_a_thin_section(volume, keep=5)
    index = RetrievalIndex(thin, cfg)
    # Everything but the thin section is excluded, so it is the only evidence there is.
    exclude = {float(s.z) for s in thin.sections[1:]}
    gen = np.random.default_rng(6)
    rows = np.nonzero(index.section_index == 1)[0]
    pick = gen.choice(rows, size=128, replace=False)
    # A five-cell union is smaller than K by construction, so the inert warning is the
    # correct report here rather than noise to be silenced.
    with pytest.warns(InertScoreWarning):
        idx, _ = index.query(
            index.coords[pick], exclude, seed=0, source_section=index.section_index[pick]
        )
    tokens, mask = index.neighbour_tokens(index.coords[pick], idx)
    assert bool(mask.any()), "the thin section must still supply real donors"
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


def test_inert_score_warns_when_the_union_is_no_larger_than_k(
    cfg: Config, volume: Volume, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """The invariant is about the candidate **union**, not the per-section cap.

    ``Config.validate`` enforces ``candidates_per_section >= retrieval_k``, which covers a
    single admissible section. It cannot cover the runtime case: ``exclude_z``, the z window
    and — the one that bites at inference — the gap-aware dropout all shrink the number of
    admissible sections per query. When the union falls to K, the top-K returns the whole
    pool and the three-term score decides nothing, silently.
    """
    xyz, owner = query_cells
    # One admissible section either side, at exactly retrieval_k / 2 candidates each.
    # ``retrieval_z_window_gap_factor=1.0`` pins the relative term to the query's own gap
    # so it admits the nearest section and nothing beyond it; at the default 2.0 the window
    # would reach the second section either way and the union would be 4 x 32, not 2 x 32.
    narrow = RetrievalIndex(
        volume,
        cfg.replace(
            retrieval_z_window=1.2,
            retrieval_z_window_gap_factor=1.0,
            retrieval_k=32,
            retrieval_candidates_per_section=32,
        ),
    )
    with pytest.warns(InertScoreWarning, match="the retrieval score decided nothing"):
        narrow.query(xyz, set(), seed=0, source_section=owner)


def test_no_inert_warning_when_the_pool_is_large_enough(
    index: RetrievalIndex, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    import warnings as _warnings

    xyz, owner = query_cells
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", InertScoreWarning)
        index.query(xyz, set(), seed=0, source_section=owner)


def test_config_rejects_a_candidate_cap_below_k() -> None:
    from spatialcpav25_gen.config import ConfigError

    with pytest.raises(ConfigError, match="retrieval_candidates_per_section"):
        Config().replace(retrieval_candidates_per_section=16, retrieval_k=32)


# --------------------------------------------------------------------------------------
# the two-term z window
# --------------------------------------------------------------------------------------


def _irregular_volume(volume: Volume) -> Volume:
    """The fixture with a hole: sections 1-3 removed, leaving a 4x gap at the bottom.

    The shape of the failure this window exists for, and the same shape a ``consecutive``-3
    holdout produces on the gate fixture. Four of the five surviving gaps are one spacing,
    so ``median_spacing`` — and with it the absolute term — is completely unchanged, while
    the first section's real gap is four times it.
    """
    return rebuild_volume(volume, [s for i, s in enumerate(volume.sections) if i not in (1, 2, 3)])


def test_gap_relative_window_is_identity_on_a_regular_stack(
    cfg: Config, volume: Volume, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """Bitwise identity on the evaluation path, so nothing downstream moves.

    On a regular stack a query's gap to its nearest admissible section is at most one
    spacing, so ``gap_factor x gap <= 2 < 3 = retrieval_z_window``: the absolute term wins
    for every query and the relative term is inert. Every published number — both gates,
    the benchmark, T05's and T06's — is measured on a stack of this shape and on this path,
    so anything short of bitwise equality here is a re-baselining event.

    Scoped to ``apply_dropout=False`` deliberately. The curriculum *does* move, by design:
    ``test_gap_relative_window_follows_the_dropout_gap`` is where that is pinned, with the
    argument for it.

    Compared against ``gap_factor = 1.0``, the smallest legal value, rather than against a
    recorded array: that arm is the one the relative term cannot widen, so the comparison
    isolates the new term instead of merely restating today's output.
    """
    xyz, owner = query_cells
    default = RetrievalIndex(volume, cfg)
    inert = RetrievalIndex(volume, cfg.replace(retrieval_z_window_gap_factor=1.0))
    for exclude in (set(), {float(volume.sections[0].z)}, {float(volume.sections[4].z)}):
        a_idx, a_w = default.query(xyz, exclude, seed=11, source_section=owner)
        b_idx, b_w = inert.query(xyz, exclude, seed=11, source_section=owner)
        assert np.array_equal(a_idx, b_idx), (
            f"the relative term changed the donors on a regular stack (exclude={exclude}); "
            "it must be inert wherever the absolute term already wins"
        )
        assert np.array_equal(a_w, b_w)


def test_gap_relative_window_follows_the_dropout_gap(
    cfg: Config, volume: Volume, query_cells: tuple[np.ndarray, np.ndarray]
) -> None:
    """The one path the relative term is *not* inert on, asserted rather than discovered.

    The curriculum exists to simulate a wide gap: it drops the query's nearest section so
    the model cannot learn a shortcut it will not have at inference. Measuring the window
    after the drop means the simulated gap carries its own window with it — which is the
    point. Measured before it, the curriculum would hand the model a wide gap with a window
    sized for a narrow one, i.e. it would train in exactly the train/inference mismatch
    that costs the most: widening the window at evaluation time only, without retraining,
    drove held-out R^2 on the GATE 2 fixture from -0.02 to -0.35.

    So on a regular stack, with dropout on, the pool must be *strictly larger* than it is
    with the relative term pinned inert. This is a deliberate change to the training path
    and the reason the identity test above is scoped to the evaluation path.
    """
    xyz, owner = query_cells
    default = RetrievalIndex(volume, cfg)
    inert = RetrievalIndex(volume, cfg.replace(retrieval_z_window_gap_factor=1.0))
    wide, _ = default.query(xyz, set(), seed=11, source_section=owner, apply_dropout=True)
    pinned, _ = inert.query(xyz, set(), seed=11, source_section=owner, apply_dropout=True)

    def furthest(idx: np.ndarray, source: RetrievalIndex) -> np.ndarray:
        mask = idx != PAD_INDEX
        z = source.section_z[source.section_index[np.where(mask, idx, 0)]]
        return np.where(mask, np.abs(z - xyz[:, 2:3]), 0.0).max(axis=1)

    assert not np.array_equal(wide, pinned)
    assert furthest(wide, default).mean() > furthest(pinned, inert).mean(), (
        "with the nearest section dropped, the window must follow the widened gap; if it "
        "does not, the curriculum is training the model on a mismatch it will meet at "
        "inference"
    )


def test_gap_relative_window_never_starves_an_irregular_stack(cfg: Config, volume: Volume) -> None:
    """The bug this rule exists for: a section whose real gap exceeds the median spacing.

    Sized off ``median_spacing`` alone the isolated section retrieves nothing at all and
    every one of its cells trains against a fully masked attention row. The first assertion
    is what makes the rest meaningful: it states that the absolute term alone *would* miss,
    so the pool being full afterwards is the relative term's doing and not the fixture's.
    """
    irregular = _irregular_volume(volume)
    gaps = np.diff([float(s.z) for s in irregular.sections])
    assert gaps[0] > cfg.retrieval_z_window * irregular.median_spacing, (
        f"fixture no longer poses the problem: first gap {gaps[0]:g} um against an absolute "
        f"window of {cfg.retrieval_z_window * irregular.median_spacing:g} um"
    )

    starved = RetrievalIndex(irregular, cfg.replace(retrieval_z_window_gap_factor=1.0))
    # gap_factor = 1 admits the nearest section and nothing else, so to reproduce the old
    # behaviour the query must also have its own section excluded — which is the pipeline's
    # default and the configuration GATE 2 runs in.
    rows = np.nonzero(starved.section_index == 0)[0]
    owner = starved.section_index[rows]
    idx, _ = starved.query(starved.coords[rows], set(), seed=0, source_section=owner)
    reached = (idx != PAD_INDEX).any(axis=1)
    assert reached.all(), "gap_factor >= 1 must reach the nearest section even at a 3x gap"

    default = RetrievalIndex(irregular, cfg)
    idx_default, _ = default.query(default.coords[rows], set(), seed=0, source_section=owner)
    assert (idx_default != PAD_INDEX).all(), (
        "at the default gap factor the isolated section must fill its pool, not merely "
        "reach one donor"
    )


def test_z_window_is_measured_after_the_own_section_exclusion(cfg: Config, volume: Volume) -> None:
    """The ordering constraint, asserted rather than left to the call site's comment.

    A cell's *own* section is at gap zero. If the window were sized before the own-section
    exclusion ran, the relative term would be ``gap_factor x 0 = 0``, the absolute term
    would be all that is left, and on an irregular stack the isolated section would starve
    again — silently, and only in the leave-own-section-out configuration GATE 2 depends on.
    """
    irregular = _irregular_volume(volume)
    index = RetrievalIndex(irregular, cfg.replace(retrieval_z_window_gap_factor=1.0))
    rows = np.nonzero(index.section_index == 0)[0]
    idx, _ = index.query(
        index.coords[rows], set(), seed=0, source_section=index.section_index[rows]
    )
    donors = index.section_index[np.where(idx != PAD_INDEX, idx, 0)]
    assert (idx != PAD_INDEX).any(axis=1).all()
    assert not (donors[idx != PAD_INDEX] == 0).any(), "own section leaked into the pool"


def test_config_rejects_a_gap_factor_below_one() -> None:
    from spatialcpav25_gen.config import ConfigError

    with pytest.raises(ConfigError, match="retrieval_z_window_gap_factor"):
        Config().replace(retrieval_z_window_gap_factor=0.5)
    with pytest.raises(ConfigError, match="retrieval_z_window_gap_factor"):
        Config().replace(retrieval_z_window_gap_factor=float("inf"))
