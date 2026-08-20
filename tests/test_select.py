"""T09 acceptance tests for the automatic configuration selector (``specs/09`` §3).

The three that carry the amendment T08 added are asserted **by construction on a stub
scorer**, which is the only way to state them: they are properties of the *search*, and a
search property has to be provable against a scorer whose answers are known.

* ``test_budget_and_metric_weights_are_selected_jointly`` — the stub scores
  ``(2x, weights on)`` best and every other cell worse. A one-gate-at-a-time selector starting
  from a ``1x`` incumbent scores ``(1x, on)``, loses, and never reaches that cell, so it
  cannot pass this test. That is the point.
* ``test_budget_gate_is_not_scored_at_a_reduced_budget`` — the two budget candidates must be
  *fitted* at different step counts. Without it the gate compares a candidate against itself
  and returns a null result for something that demonstrably moves four of six statistics.
* ``test_selection_never_sees_heldout`` — a ``TypeError`` on anything but a
  ``TrainingVolume``, and a chosen budget that does not depend on whether held-out sections
  exist in the parent object. The budget is the gate most easily fitted to a test set.

``test_selector_runs_and_persists`` runs the **real** scorer — fits, generation, the six
metrics, the report — at toy budgets, so the wiring is tested end to end without a paper-scale
run. The paper-scale run is ``scripts/t09_report.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import HeldOutSections, TrainingVolume
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.train.select import (
    ALL_GATES,
    FULL_BUDGET_GATES,
    GATES,
    METRIC_NAMES,
    TRAINING_FREE_OPTIONS,
    V20_CONFIG,
    ScoreCache,
    SelectionError,
    joint_gate_cells,
    run_selection,
    select_config,
    selection_folds,
    write_selection_report,
)

from tests.fixtures.synthetic import make_synthetic_volume
from tests.test_expression import build_embeddings
from tests.test_generate import SEED, t09_cfg


@pytest.fixture(scope="module")
def split():
    """The synthetic fixture, split once for every test in this module."""
    vol, _ = make_synthetic_volume(seed=0)
    training, held = split_holdout(vol, "alternating", 0, Config())
    return vol, training, held


class RecordingScorer:
    """A scorer with known answers that records every ``(config, steps)`` it was asked for.

    ``prefer`` names the overrides that win. Two reward shapes, and which one a test uses is
    part of what that test asserts:

    ``interaction=True`` (the default)
        **Conjunctive**: only a candidate matching *every* preference scores at all. This is
        the shape of the interaction T08 measured — the metric-aware weights pay off only at
        the larger budget — and it is what makes
        ``test_budget_and_metric_weights_are_selected_jointly`` fail on a selector that
        visits the two gates in turn.
    ``interaction=False``
        **Additive**: each preference pays independently, which is the shape of "the new
        components are individually degraded" that the no-regression test needs — coordinate
        descent is supposed to find that one, one gate at a time.
    """

    def __init__(self, prefer: dict[str, object], *, interaction: bool = True):
        self.prefer = prefer
        self.interaction = interaction
        self.calls: list[tuple[dict[str, object], int]] = []

    def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
        """Score one candidate; the recorded call is the evidence the tests read."""
        seen = {name: getattr(cfg, name) for name in self.prefer}
        self.calls.append(({**seen, "train_steps": cfg.train_steps}, int(steps)))
        matches = [getattr(cfg, name) == value for name, value in self.prefer.items()]
        if not matches:
            base = 0.0
        elif self.interaction:
            base = 1.0 if all(matches) else 0.0
        else:
            base = float(np.mean(matches))
        return dict.fromkeys(METRIC_NAMES, base)


def test_joint_gate_has_four_cells_at_two_budgets():
    """``{1x, 2x} x {off, spec weights}`` — four cells, both budgets, both weight settings."""
    cfg = t09_cfg(train_steps=100)
    cells = joint_gate_cells(cfg)
    assert len(cells) == 4
    budgets = {int(overrides["train_steps"]) for _, overrides in cells}
    assert budgets == {100, round(cfg.selection_budget_multiple * 100)}
    weights = {float(overrides["w_autocorr"]) for _, overrides in cells}
    assert weights == {0.0, float(cfg.selection_metric_weight)}
    assert [label for label, _ in cells] == [
        "1x, weights off",
        "1x, weights on",
        "2x, weights off",
        "2x, weights on",
    ]


def test_budget_and_metric_weights_are_selected_jointly(split):
    """All four cells are scored, and the selector can return ``(2x, weights on)``.

    A coordinate-descent selector that visited the budget and the weights as separate gates
    would score ``(1x, on)`` from a ``1x`` incumbent, lose, and stop — it cannot return this
    cell, so this test fails on the implementation ``specs/09`` §3 forbids.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=100, w_autocorr=0.0, w_profile=0.0, w_distribution=0.0)
    double = round(base.selection_budget_multiple * 100)
    scorer = RecordingScorer(
        {"train_steps": double, "w_autocorr": float(base.selection_metric_weight)}
    )
    result = run_selection(training, base, seed=SEED, scorer=scorer)

    scored = {(c.overrides["train_steps"], c.overrides["w_autocorr"]) for c in result.joint}
    assert scored == {
        (100, 0.0),
        (100, float(base.selection_metric_weight)),
        (double, 0.0),
        (double, float(base.selection_metric_weight)),
    }
    assert len(result.joint) == 4
    assert result.config.train_steps == double
    assert result.config.w_autocorr == float(base.selection_metric_weight)
    assert result.config.w_profile == float(base.selection_metric_weight)
    assert result.config.w_distribution == float(base.selection_metric_weight)


def test_budget_gate_is_not_scored_at_a_reduced_budget(split):
    """The two budget candidates are fitted at *different* step counts (§3's requirement 3)."""
    _, training, _ = split
    base = t09_cfg(train_steps=100)
    double = round(base.selection_budget_multiple * 100)
    scorer = RecordingScorer({"train_steps": double})
    result = run_selection(training, base, seed=SEED, scorer=scorer)

    joint_fits = {
        int(overrides["train_steps"]): steps
        for overrides, steps in result.fits
        if int(overrides["train_steps"]) in {100, double} and steps in {100, double}
    }
    assert joint_fits.get(100) == 100
    assert joint_fits.get(double) == double
    assert joint_fits[100] != joint_fits[double]
    # Every joint cell was fitted at the budget it names, not at a common reduced one.
    for candidate in result.joint:
        assert candidate.steps == int(candidate.overrides["train_steps"])


def test_selector_can_recover_v20_config(split):
    """``layout_mode=resample`` + ``expr_mode=cross-mix`` is reachable and is selected.

    The no-regression guarantee: when the new components are artificially degraded — here, by
    a scorer that prefers the previous version's two gates — the selector switches them off by
    itself, which is what "the user never tunes a flag" has to mean in the bad case.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50, layout_mode="field", expr_mode="zinb-flow")
    scorer = RecordingScorer(dict(V20_CONFIG), interaction=False)
    chosen = select_config(training, base, seed=SEED, scorer=scorer)
    assert chosen.layout_mode == "resample"
    assert chosen.expr_mode == "cross-mix"
    # ...and it is reachable, i.e. both options really are on the gate grid.
    options = dict(ALL_GATES)
    assert "resample" in options["layout_mode"]
    assert "cross-mix" in options["expr_mode"]


def test_selection_never_sees_heldout(split):
    """A holdout is a ``TypeError``, and the chosen budget ignores whether one exists."""
    vol, training, held = split
    base = t09_cfg(train_steps=50)
    assert isinstance(held, HeldOutSections)
    for bad in (held, vol):
        with pytest.raises(TypeError, match="TrainingVolume"):
            select_config(bad, base, seed=SEED, scorer=RecordingScorer({}))  # type: ignore[arg-type]

    standalone = TrainingVolume(
        sections=list(training.sections),
        gene_names=list(training.gene_names),
        celltype_names=list(training.celltype_names),
        region_names=None if training.region_names is None else list(training.region_names),
        specimen_id=training.specimen_id,
    )
    prefer = {"train_steps": round(base.selection_budget_multiple * 50)}
    with_holdout = select_config(training, base, seed=SEED, scorer=RecordingScorer(prefer))
    without = select_config(standalone, base, seed=SEED, scorer=RecordingScorer(prefer))
    assert with_holdout.train_steps == without.train_steps
    assert with_holdout.content_hash() == without.content_hash()


def test_selection_requires_a_scorer_or_embeddings(split):
    """Neither a scorer nor an embeddings factory is an error naming both (Convention 6)."""
    _, training, _ = split
    with pytest.raises(SelectionError, match="scorer or an embeddings factory"):
        select_config(training, t09_cfg(train_steps=10), seed=SEED)


def test_lookup_only_text_mode_drops_the_text_channel():
    """``text_emb_mode`` is a real gate: ``lookup`` removes the text prior, seen and unseen.

    T01 declared the field and nothing consumed it, which would have made the selector score
    four identical candidates and report a decision it had not made. The mode is T10's
    ablation A3 as well, so it is implemented once, in the embedding.
    """
    import torch
    from spatialcpav25_gen.model.embeddings import TextGroundedEmbedding

    from tests.fixtures.text import fake_text_vecs

    cfg = t09_cfg()
    vectors = torch.from_numpy(fake_text_vecs(6, cfg.text_dim_in, 1))
    grounded = TextGroundedEmbedding(vectors, 8, cfg)
    lookup = TextGroundedEmbedding(vectors, 8, cfg.replace(text_emb_mode="lookup"))
    idx = torch.arange(6)
    grounded.set_progress(0.0)
    lookup.set_progress(0.0)

    # The residual is zero-initialised, so at the start of training the grounded embedding is
    # the text channel alone and the lookup-only one has nothing at all.
    assert not torch.allclose(grounded(idx), lookup(idx))
    assert torch.allclose(lookup(idx), lookup.norm(torch.zeros(6, 8)))
    # ...and no warm-up under lookup-only: the residual *is* the embedding.
    assert float(lookup.gamma) == 1.0
    assert float(grounded.gamma) == 0.0
    assert torch.allclose(
        lookup.forward_zero_shot(vectors, use_distill=False),
        lookup.norm(torch.zeros(6, 8)),
    )


def test_selection_folds_are_interior_and_fixed(split):
    """Folds avoid the stack's ends (open risk R3) and are the same for every candidate."""
    _, training, _ = split
    cfg = t09_cfg()
    folds = selection_folds(training, cfg)
    assert 0 < len(folds) <= int(cfg.selection_n_folds)
    assert folds == selection_folds(training, cfg)
    ends = {training.sections[0].section_id, training.sections[-1].section_id}
    assert not {s.section_id for s in folds} & ends


def test_report_is_written_with_every_joint_cell(split, tmp_path: Path):
    """The report carries all four cells of the joint gate, not just the winner."""
    _, training, _ = split
    base = t09_cfg(train_steps=40)
    scorer = RecordingScorer({"train_steps": round(base.selection_budget_multiple * 40)})
    result = run_selection(training, base, seed=SEED, scorer=scorer)
    path = write_selection_report(result, tmp_path / "config_selection_test.md")
    text = path.read_text(encoding="utf-8")
    for label in ("1x, weights off", "1x, weights on", "2x, weights off", "2x, weights on"):
        assert label in text
    for name in METRIC_NAMES:
        assert name in text
    assert "TrainingVolume" in text
    assert result.config.content_hash() in text
    # ...and the config itself is persisted beside it, loadable rather than only readable.
    persisted = Config.from_yaml(path.with_suffix(".yaml"))
    assert persisted.content_hash() == result.config.content_hash()


@pytest.mark.slow
def test_selector_runs_and_persists(split, tmp_path: Path):
    """The real scorer runs end to end on the fixture and writes the report.

    Toy budgets: this asserts the *wiring* — fit, generate, score on the six metrics, rank,
    persist — not the numbers. The numbers are ``scripts/t09_report.py``'s, at the real
    budgets, and land in ``reports/config_selection_synthetic.md``.
    """
    vol, training, _ = split
    base = t09_cfg(train_steps=3, selection_n_folds=1, selection_passes=1)
    report = tmp_path / "config_selection_synthetic.md"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        result = run_selection(
            training,
            base,
            seed=SEED,
            embeddings=lambda cfg: build_embeddings(cfg, vol),
            report_path=report,
        )
    assert report.exists()
    assert set(result.joint[0].scores) == set(METRIC_NAMES)
    assert all(np.isfinite(v) for c in result.candidates for v in c.scores.values())
    assert result.config.train_steps in {3, round(base.selection_budget_multiple * 3)}
    assert len(result.joint) == 4


def test_every_gate_is_classified_by_the_training_free_rule():
    """``specs/09`` §3's rule is only a rule if adding a gate forces the classification.

    Each gate must appear in ``TRAINING_FREE_OPTIONS`` — an empty tuple being the positive
    statement "all options train", not a missing entry — and may only name options it actually
    has. ``_check_gate_classification`` runs at import; this asserts it rejects both mistakes.
    """
    for gate, _ in ALL_GATES:
        assert gate in TRAINING_FREE_OPTIONS, f"{gate} is unclassified"
    for gate, options in ALL_GATES:
        assert set(TRAINING_FREE_OPTIONS[gate]) <= set(options)
    # The two gate sets partition the table on exactly that classification.
    assert dict(FULL_BUDGET_GATES).keys() | dict(GATES).keys() == dict(ALL_GATES).keys()
    assert not dict(FULL_BUDGET_GATES).keys() & dict(GATES).keys()
    for gate, _ in FULL_BUDGET_GATES:
        assert TRAINING_FREE_OPTIONS[gate], f"{gate} is full-budget but has no training-free option"
    for gate, _ in GATES:
        assert not TRAINING_FREE_OPTIONS[gate], f"{gate} is reduced-budget but has one"


def test_gates_with_a_training_free_option_are_scored_at_the_selected_budget(split):
    """The R8 fix: no cell of the merged gate is fitted at the reduced budget.

    Measured cause (``reports/r8_budget_grid.md``): ``cross-mix`` copies donor counts and is
    flat in budget (+0.0088 morans from 600 to 2400) while ``zinb-flow`` gains +0.3432, so a
    quarter-budget comparison of that gate measures the budget. Every option of every
    disqualified gate must therefore be fitted at the budget the joint gate selected.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)
    scorer = RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False)
    result = run_selection(training, base, seed=SEED, scorer=scorer)

    assert len(result.full_budget) == 18, "3 layout x 2 prior x 3 expr"
    selected_steps = int(result.config.train_steps)
    assert {c.steps for c in result.full_budget} == {selected_steps}

    # Every option of every disqualified gate is *compared* only inside the merged gate, and
    # the merged gate is entirely at full budget. A gate's selected value still tags along as
    # part of the incumbent while a later gate is scored, which is not the same as scoring it.
    reduced = round(float(base.selection_reduced_epoch_frac) * selected_steps)
    assert reduced != selected_steps, "the fixture must distinguish the two budgets"
    for gate, options in FULL_BUDGET_GATES:
        assert not [c for c in result.candidates if c.gate == gate], (
            f"{gate} is still coordinate-descended; the rule says it must be merged"
        )
        compared = {c.overrides[gate] for c in result.full_budget}
        assert compared == set(options), f"{gate} did not compare every option"
    assert {c.steps for c in result.candidates if c.gate == "full_budget"} == {selected_steps}
    # ...and the gates the rule leaves eligible do keep the reduced budget.
    for gate, _ in GATES:
        assert {c.steps for c in result.candidates if c.gate == gate} == {reduced}


def test_the_merged_gate_beats_coordinate_descent_on_a_compounding_interaction(split):
    """A scorer where the right answer is only reachable if the three gates are scored together.

    The fixture's failure in miniature: the payoff needs ``correlated`` **and** ``zinb-flow``
    together, and each is worse than its alternative on its own. Coordinate descent from a
    ``cross-mix`` incumbent rejects ``correlated``, then never revisits it; the merged gate
    enumerates the cell and finds it.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50, prior_mode="iid", expr_mode="cross-mix")
    scorer = RecordingScorer(
        {"prior_mode": "correlated", "expr_mode": "zinb-flow"}, interaction=True
    )
    chosen = select_config(training, base, seed=SEED, scorer=scorer)
    assert chosen.prior_mode == "correlated"
    assert chosen.expr_mode == "zinb-flow"


def test_score_cache_checkpoints_each_cell_and_resumes(split, tmp_path):
    """An interrupted selection resumes instead of restarting: 18 full-budget fits demand it.

    The cache is keyed on the candidate's full config hash and its budget, so a resumed run
    reuses a cell only when every field that could affect the score matches.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)
    path = tmp_path / "checkpoint.csv"

    first = RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False)
    result = run_selection(training, base, seed=SEED, scorer=first, checkpoint=ScoreCache(path))
    assert len(first.calls) == len(result.fits) > 0
    assert path.exists()

    # A second run over the same cells issues no fits at all, and decides the same thing.
    second = RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False)
    resumed = run_selection(training, base, seed=SEED, scorer=second, checkpoint=ScoreCache(path))
    assert second.calls == [], "a resumed run must not refit a recorded cell"
    assert resumed.fits == []
    assert resumed.config.content_hash() == result.config.content_hash()

    # An unrelated Config change invalidates the cache rather than reusing a stale score.
    third = RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False)
    run_selection(
        training,
        base.replace(metric_knn_k=int(base.metric_knn_k) + 1),
        seed=SEED,
        scorer=third,
        checkpoint=ScoreCache(path),
    )
    assert third.calls, "a different config must not hit the cache"
