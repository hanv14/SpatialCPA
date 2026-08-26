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
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import HeldOutSections, TrainingVolume
from spatialcpav25_gen.model.expression import ExpressionError
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.train.select import (
    ALL_GATES,
    CAPABILITY_CLAIM,
    FIT_INVARIANT_GATES,
    FULL_BUDGET_GATES,
    GATES,
    METRIC_NAMES,
    TRAINING_FREE_OPTIONS,
    V20_CONFIG,
    Candidate,
    CandidateFailedWarning,
    FitScorer,
    InertGateError,
    InertGateWarning,
    ScoreCache,
    SelectionError,
    capability_tie_break,
    incumbent_is_unconverged,
    inert_gates,
    joint_gate_cells,
    repulsion_is_reachable,
    run_selection,
    select_config,
    selection_folds,
    volume_cache_key,
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

    ``slow`` since 2026-08-26: it trains a loop per candidate *and* now pays the inertness
    probe's untrained model per descent gate (32 s -> 69 s), which the fast suite's 3-minute
    budget cannot carry. ``make test-all`` runs it. What the search *decides* is pinned by the
    stub-scorer tests, which stay fast precisely because they do not fit anything.

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


def test_incumbent_unconverged_escalates_every_remaining_gate(split):
    """Condition (2): a gate is not decided on a model unlike the shipped one (R9).

    ``text_emb_mode`` passes condition (1) — both its options train — but on the fixture it was
    decided at 600 steps under a ``zinb-flow`` incumbent scoring 0.5997 / 0.6523 on
    ``morans_pearson`` against 0.96 at the selected budget, and its winner flipped to
    ``lookup``, which disables the MedCPT channel the open-vocabulary claim rests on. When the
    incumbent's own reduced-budget score falls that far short, no remaining gate may use it.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    class BudgetSensitiveScorer(RecordingScorer):
        """Scores rise steeply with budget, so the incumbent is unconverged when reduced."""

        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            scores = super().__call__(cfg, steps=steps, seed=seed)
            floor = 0.2 if steps < int(base.train_steps) * base.selection_budget_multiple else 1.0
            return {name: value * floor for name, value in scores.items()}

    scorer = BudgetSensitiveScorer({"text_emb_mode": "medcpt"}, interaction=False)
    result = run_selection(training, base, seed=SEED, scorer=scorer)
    assert result.reduced_budget_escalated
    assert len(result.escalating_metrics) >= int(base.selection_convergence_min_metrics)
    selected_steps = int(result.config.train_steps)
    for gate, _ in GATES:
        assert {c.steps for c in result.candidates if c.gate == gate} == {selected_steps}, (
            f"{gate} was scored at the reduced budget despite an unconverged incumbent"
        )


def test_a_converged_incumbent_keeps_the_reduced_budget(split):
    """Condition (2) must not fire on every run, or the reduced budget is dead.

    The counterpart the fixture supplies: with a ``cross-mix`` incumbent the reduced budget
    costs at most 0.04 and one metric *improves*, so it stays a usable proxy and the cheap
    descent survives for gates that deserve it.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)
    scorer = RecordingScorer({"text_emb_mode": "medcpt"}, interaction=False)
    result = run_selection(training, base, seed=SEED, scorer=scorer)
    assert not result.reduced_budget_escalated
    assert result.escalating_metrics == ()
    reduced = round(float(base.selection_reduced_epoch_frac) * int(result.config.train_steps))
    for gate, _ in GATES:
        assert {c.steps for c in result.candidates if c.gate == gate} == {reduced}


def test_convergence_predicate_counts_only_shortfalls():
    """A reduced fit that scores *higher* is not evidence the proxy is broken."""
    cfg = t09_cfg()
    full = dict.fromkeys(METRIC_NAMES, 0.9)
    better = dict.fromkeys(METRIC_NAMES, 0.99)
    assert incumbent_is_unconverged(full, better, cfg) == (False, ())
    worse = {**full, "morans_pearson": 0.1, "gearys_pearson": 0.1}
    fired, metrics = incumbent_is_unconverged(full, worse, cfg)
    assert fired
    assert set(metrics) == {"morans_pearson", "gearys_pearson"}
    # One failing metric is short of the default minimum of two, so it does not escalate.
    assert int(cfg.selection_convergence_min_metrics) == 2
    one = {**full, "morans_pearson": 0.1}
    fired_one, metrics_one = incumbent_is_unconverged(full, one, cfg)
    assert not fired_one
    assert metrics_one == ("morans_pearson",), "the metric is still reported, just not enough"


def _cand(gate: str, option: str, values: list[float], rank: float) -> Candidate:
    """One scored candidate of ``gate``, for the tie-break tests."""
    return Candidate(
        gate=gate,
        label=f"{gate}={option}",
        overrides={gate: option},
        steps=2400,
        scores=dict(zip(METRIC_NAMES, values, strict=True)),
        rank=rank,
    )


def test_every_option_has_a_capability_claim_level():
    """The tie-break's classification is forced when a gate is added, like the budget rule's."""
    for gate, options in ALL_GATES:
        assert gate in CAPABILITY_CLAIM, f"{gate} has no claim levels"
        assert set(CAPABILITY_CLAIM[gate]) == set(options), f"{gate} is partly classified"
        assert min(CAPABILITY_CLAIM[gate].values()) == 0, (
            f"{gate} needs a level-0 option — the one that claims nothing beyond reusing "
            "real data — or 'prefer the capability' has no floor to prefer against"
        )


def test_tie_break_prefers_the_exercised_capability_below_the_envelope():
    """`lookup` outranks `medcpt` by less than the envelope, and disables the text channel.

    R10's measured case. The margin is at most 0.011 against a reproducibility envelope of
    0.02, so the rank ordering is not evidence; `medcpt` keeps the MedCPT channel live, which
    is the open-vocabulary claim, so it wins.
    """
    cfg = t09_cfg()
    lookup = _cand(
        "text_emb_mode", "lookup", [0.9511, 0.9334, 0.9688, -0.0425, 0.0460, -0.0660], 1.2
    )
    medcpt = _cand(
        "text_emb_mode", "medcpt", [0.9535, 0.9288, 0.9624, -0.0469, 0.0570, -0.0660], 1.8
    )
    winner, reason = capability_tie_break([lookup, medcpt], "text_emb_mode", cfg)
    assert winner.overrides["text_emb_mode"] == "medcpt"
    assert "tie-broken on capability" in reason


def test_tie_break_refuses_to_credit_an_inert_capability():
    """An exactly-identical rival proves the extra claim does nothing, whatever the ordering.

    The fixture's `auto-blend`: `w(v)` is 0 at every knot, so the blend passes the flow's draw
    through and the cell is bit-identical to `zinb-flow`. Shipping the richer label would claim
    a mechanism no emitted count depends on. Asserted in both candidate orderings, because
    equal ranks make the ordering arbitrary and an early return on the first element was
    exactly the bug this pins.
    """
    cfg = t09_cfg()
    same = [0.9606, 0.9308, 0.9744, -0.0437, 0.0491, -0.0660]
    for order in (("auto-blend", "zinb-flow"), ("zinb-flow", "auto-blend")):
        cands = [_cand("expr_mode", opt, same, 3.0) for opt in order]
        winner, _ = capability_tie_break(cands, "expr_mode", cfg)
        assert winner.overrides["expr_mode"] == "zinb-flow", (
            f"ordering {order} credited an inert claim"
        )

    # ...but a capability that *does* something is credited.
    live = _cand("expr_mode", "auto-blend", [0.97, 0.94, 0.98, -0.040, 0.052, -0.065], 1.0)
    winner, _ = capability_tie_break(
        [_cand("expr_mode", "zinb-flow", same, 2.0), live], "expr_mode", cfg
    )
    assert winner.overrides["expr_mode"] == "auto-blend"


def test_tie_break_leaves_a_clear_margin_alone():
    """Outside the envelope the measurement decides and capability is not consulted."""
    cfg = t09_cfg()
    best = _cand("expr_mode", "zinb-flow", [0.9606, 0.9308, 0.9744, -0.0437, 0.0491, -0.0660], 1.0)
    far = _cand("expr_mode", "cross-mix", [0.60, 0.60, 0.60, -0.05, 0.01, -0.07], 2.0)
    winner, reason = capability_tie_break([best, far], "expr_mode", cfg)
    assert winner is best
    assert "no rival within the envelope" in reason


# --------------------------------------------------------------------------------------
# the fit-invariant gates, and the cache that rests on them
# --------------------------------------------------------------------------------------


def test_layout_mode_does_not_enter_the_fit():
    """Two fits differing only in ``layout_mode`` are bitwise identical, every tensor.

    This is the invariant :data:`FIT_INVARIANT_GATES` declares and
    :class:`FitScorer`'s cache spends: one model serves all three ``layout_mode`` options, so
    the merged 18-cell gate costs 6 fits rather than 18. ``sample_layout`` is never called
    during training and ``_layout_term`` evaluates the intensity at the **real** cells'
    positions, so nothing in the loop can see the gate — but "nothing can see it" is an
    argument about today's code, and the saving it buys is silent if it stops being true.
    A future change that made training read ``layout_mode`` would make the cache serve a stale
    model to two thirds of the gate, and every score would still look plausible. This test is
    what turns that into a failure.
    """
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

    assert FIT_INVARIANT_GATES == ("layout_mode",), FIT_INVARIANT_GATES
    vol, _ = make_synthetic_volume(seed=0)
    base = t09_cfg(train_steps=2)
    training, _ = split_holdout(vol, "alternating", 0, base)

    def fit(mode: str) -> dict[str, object]:
        cfg = base.replace(layout_mode=mode)
        model = CTFFlow(
            cfg, TrainingData.build(training, cfg), build_embeddings(cfg, vol), grf_seed=11
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train_ctfflow(model, cfg, steps=2, seed=SEED)
        return dict(model.state_dict())

    reference = fit("field")
    assert reference, "the model has no parameters to compare"
    for mode in ("hybrid", "resample"):
        other = fit(mode)
        assert set(other) == set(reference), mode
        for name, tensor in reference.items():
            assert torch.equal(tensor, other[name]), (mode, name)


def test_fit_cache_keys_ignore_only_the_fit_invariant_gates():
    """The cache key collapses ``layout_mode`` and nothing else.

    A key that ignored too much would hand one model to candidates that genuinely need
    separate fits — the failure mode this saving has to be protected against, and one that
    would not show up as an error anywhere.
    """
    vol, _ = make_synthetic_volume(seed=0)
    training, _ = split_holdout(vol, "alternating", 0, Config())
    scorer = FitScorer(training, lambda cfg: build_embeddings(cfg, vol))
    base = t09_cfg()

    keys = {
        mode: scorer._fit_key(base.replace(layout_mode=mode), 10, 1)
        for mode in ("field", "hybrid", "resample")
    }
    assert len(set(keys.values())) == 1, keys

    reference = keys["field"]
    for gate, options in ALL_GATES:
        if gate in FIT_INVARIANT_GATES:
            continue
        for option in options:
            other = scorer._fit_key(base.replace(**{gate: option}), 10, 1)
            if getattr(base, gate) != option:
                assert other != reference, (gate, option)
    assert scorer._fit_key(base, 20, 1) != reference, "the budget must be part of the key"
    assert scorer._fit_key(base, 10, 2) != reference, "the seed must be part of the key"


# --------------------------------------------------------------------------------------
# pinned gates, and the per-gate tie-break review
# --------------------------------------------------------------------------------------


def test_a_pinned_gate_is_excluded_from_both_gate_sets_and_fixed_everywhere(split):
    """Pinning a gate removes it from the search and fixes it in every candidate.

    ``specs/09`` §3 selects per dataset; a gate that a *different* dataset settled — R11's
    ``layout_mode`` on real STARmap — must not be re-opened at one seed inside a search whose
    own margins are envelope-sized. The mechanism has to do three things and this asserts all
    three: shrink the merged gate's product, drop the gate from coordinate descent, and carry
    the pinned value into the selected config.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)
    # Pinned to a *non-default* option, so "the pinned value reached everything" is a real
    # assertion rather than one the default would satisfy on its own.
    assert base.layout_mode != "hybrid"
    scorer = RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False)
    result = run_selection(
        training,
        base,
        seed=SEED,
        scorer=scorer,
        pinned={"layout_mode": "hybrid"},
        pinned_reason="R11 settled this gate on real data.",
    )

    assert result.config.layout_mode == "hybrid"
    assert result.pinned == {"layout_mode": "hybrid"}
    assert len(result.full_budget) == 6, "2 prior x 3 expr; the pinned gate leaves the product"
    # A pinned gate is fixed in the base config, not carried as a per-candidate override —
    # which is what makes it unreachable by the search rather than merely unvisited.
    assert not any("layout_mode" in c.overrides for c in result.candidates)
    assert "layout_mode" not in {c.gate for c in result.candidates}
    pinned_base = base.replace(layout_mode="hybrid")
    for overrides, _steps in result.fits:
        assert "layout_mode" not in overrides
        assert pinned_base.replace(**overrides).layout_mode == "hybrid"

    review = {r.gate: r for r in result.reviews}
    assert review["layout_mode"].pinned is True
    assert review["layout_mode"].winner == "hybrid"
    assert "pinned" in review["layout_mode"].reason


def test_pinning_costs_scorings_and_not_fits(split):
    """The saving is 12 LOSO scorings, not 12 fits — ``layout_mode`` never enters the fit.

    Stated as a test because the opposite belief is the natural one and would misdescribe
    what excluding this gate buys. ``FitScorer`` keys its cache with
    :data:`FIT_INVARIANT_GATES` normalised out, so the 18-cell and the 6-cell merged gates
    issue the *same* fits; only the number of cells scored differs.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    free = run_selection(
        training,
        base,
        seed=SEED,
        scorer=RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False),
    )
    pinned = run_selection(
        training,
        base,
        seed=SEED,
        scorer=RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False),
        pinned={"layout_mode": "resample"},
        pinned_reason="R11.",
    )
    assert len(free.full_budget) == 18
    assert len(pinned.full_budget) == 6

    def fit_keys(result):
        scorer = FitScorer(training, lambda cfg: build_embeddings(cfg, training))
        return {
            scorer._fit_key(base.replace(**overrides), steps, SEED)
            for overrides, steps in result.fits
        }

    assert fit_keys(pinned) <= fit_keys(free)
    assert len(fit_keys(free)) == len(fit_keys(pinned)), (
        "the merged gate's fit count must not depend on layout_mode: it is fit-invariant"
    )


@pytest.mark.parametrize(
    ("pinned", "message"),
    [
        ({"not_a_gate": "x"}, "not a gate"),
        ({"prior_mode": "nonsense"}, "not an option"),
        ({g: o[0] for g, o in ALL_GATES}, "nothing to select"),
    ],
)
def test_pinning_refuses_what_it_cannot_mean(split, pinned, message):
    """An unknown gate, an option that gate does not have, or pinning everything. All raise."""
    _, training, _ = split
    with pytest.raises(SelectionError, match=message):
        run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False),
            pinned=pinned,
            pinned_reason="because",
        )


def test_pinning_requires_a_stated_reason(split):
    """A gate removed from selection without a reason is indistinguishable, in the report,
    from a gate that was never a gate."""
    _, training, _ = split
    with pytest.raises(SelectionError, match="pinned_reason"):
        run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False),
            pinned={"layout_mode": "resample"},
        )


def test_the_search_applies_the_capability_tie_break_and_reports_every_gate(split):
    """``min(rank)`` decided every gate; below the envelope the rule decides instead.

    T09 §13 applied this rule *by hand* to a printed table and recorded the result as the
    shipped config, while nothing in the code did it — so a persisted ``selected.yaml`` would
    have carried the rank winner. This pins the wiring: a gate whose options are separated by
    less than ``claim_tie_break_envelope`` ships the capability-preserving option, and the
    review says so.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    class NearTie:
        """``lookup`` outranks ``medcpt`` by 0.001 — inside any plausible envelope."""

        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            value = 0.500 if cfg.text_emb_mode == "medcpt" else 0.501
            return dict.fromkeys(METRIC_NAMES, value)

    result = run_selection(
        training,
        base.replace(claim_tie_break_envelope=0.04),
        seed=SEED,
        scorer=NearTie(),
        pinned={"layout_mode": "resample"},
        pinned_reason="R11.",
    )
    review = {r.gate: r for r in result.reviews}
    text = review["text_emb_mode"]
    assert text.margin == pytest.approx(0.001, abs=1e-9)
    assert text.inside_envelope is True
    assert text.winner == "medcpt", "the capability-preserving option ships below the envelope"
    assert result.config.text_emb_mode == "medcpt", "the review must reach the returned config"


def test_a_margin_outside_the_envelope_leaves_the_rank_winner_alone(split):
    """The rule is a tie-break, not a thumb on the scale: a real margin decides on its own."""
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    class Clear:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            return dict.fromkeys(METRIC_NAMES, 0.9 if cfg.text_emb_mode == "lookup" else 0.2)

    result = run_selection(
        training,
        base.replace(claim_tie_break_envelope=0.04),
        seed=SEED,
        scorer=Clear(),
        pinned={"layout_mode": "resample"},
        pinned_reason="R11.",
    )
    review = {r.gate: r for r in result.reviews}["text_emb_mode"]
    assert review.inside_envelope is False
    assert review.flipped is False
    assert result.config.text_emb_mode == "lookup"


def test_the_report_names_the_pinned_gate_and_the_review(split, tmp_path):
    """Both new sections reach the markdown a reader actually gets."""
    _, training, _ = split
    result = run_selection(
        training,
        t09_cfg(train_steps=50),
        seed=SEED,
        scorer=RecordingScorer({"expr_mode": "zinb-flow"}, interaction=False),
        pinned={"layout_mode": "resample"},
        pinned_reason="R11 settled it on real data; see reports/r11_starmap_layout_modes.md.",
    )
    path = write_selection_report(result, tmp_path / "selection.md")
    text = path.read_text()
    assert "Gates **not** selected here (pinned)" in text
    assert "R11 settled it on real data" in text
    assert "Per-gate tie-break review" in text
    assert "All 6 cells" in text, "the merged gate's size must follow the pinning"
    assert "`prior_mode` x `expr_mode`" in text


def test_an_unscorable_candidate_ranks_last_instead_of_aborting_the_search(split):
    """The emission guard refusing an under-trained candidate is a ranking fact, not a crash.

    ``assert_detection_rate``'s band is measured against the training sections, and on a dense
    panel — STARmap's median per-gene detection is 0.9999 (``specs/10`` §0) — an under-trained
    candidate misses it. The reduced-budget cells of any selection *are* under-trained, so
    without this a nine-hour search dies at its second cell and the fits already paid for are
    lost. It must be loud: a warning, an entry in ``failures``, and a line in the report.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    class GuardTrips:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.text_emb_mode == "lookup":
                raise ExpressionError("detection rate 0.47 against 1.0000")
            return dict.fromkeys(METRIC_NAMES, 0.7)

    with pytest.warns(CandidateFailedWarning, match="ranked last"):
        result = run_selection(
            training,
            base,
            seed=SEED,
            scorer=GuardTrips(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )
    assert result.config.text_emb_mode == "medcpt", "the failing option must not be selected"
    assert [label for label, _s, _w in result.failures] == ["text_emb_mode=lookup"]
    assert "detection rate 0.47" in result.failures[0][2]


def test_a_gate_that_fails_entirely_is_refused_rather_than_ranked(split):
    """All options failing is not a ranking, it is a broken run, and shipping the first label
    would be a silent fallback of exactly the kind Convention 6 forbids."""
    _, training, _ = split

    class AlwaysTrips:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            raise ExpressionError("detection rate 0.47 against 1.0000")

    with (
        pytest.warns(CandidateFailedWarning),
        pytest.raises(SelectionError, match="every candidate of the joint gate failed"),
    ):
        run_selection(training, t09_cfg(train_steps=50), seed=SEED, scorer=AlwaysTrips())


def test_a_real_bug_still_aborts_the_search(split):
    """:data:`SCORING_FAILURES` is deliberately narrow: only the emission guard is a candidate
    fact. A shape error, a missing field or a leakage refusal must still stop everything."""
    _, training, _ = split

    class Broken:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            raise KeyError("celltype_key")

    with pytest.raises(KeyError):
        run_selection(training, t09_cfg(train_steps=50), seed=SEED, scorer=Broken())


def test_repulsion_is_fitted_only_where_a_layout_mode_can_read_it(split):
    """``resample`` returns from ``sample_layout`` before it looks at the interaction.

    So a search whose ``layout_mode`` is pinned to ``resample`` must not pay ``fit_repulsion``
    — and the reason is not only cost: ``fit_repulsion`` raises ``LayoutError`` on a point
    pattern with no soft-repulsion range, which would abort a whole selection over a quantity
    no candidate uses. Asserted on ``FitScorer._fit`` directly, because that is where the
    decision lands, and because two full selections to observe it cost 100 s.
    """
    _, training, _ = split
    cfg = t09_cfg(train_steps=2)
    assert cfg.repulsion, "the fixture config must have repulsion on for this to mean anything"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        off = FitScorer(
            training, lambda c: build_embeddings(c, training), needs_repulsion=False
        )._fit(cfg, steps=2, seed=SEED)
        on = FitScorer(
            training, lambda c: build_embeddings(c, training), needs_repulsion=True
        )._fit(cfg, steps=2, seed=SEED)
    assert off.repulsion is None, "nothing can read it, so it must not have been fitted"
    assert on.repulsion is not None


def test_the_search_decides_repulsion_from_the_pinning():
    """``run_selection`` reads the rule from :func:`repulsion_is_reachable`, not per candidate."""
    cfg = t09_cfg()
    assert cfg.repulsion
    assert repulsion_is_reachable(cfg, {"layout_mode": "resample"}) is False
    assert repulsion_is_reachable(cfg, {"layout_mode": "hybrid"}) is True
    assert repulsion_is_reachable(cfg, None) is True, "unpinned, the gate still offers field"
    assert repulsion_is_reachable(cfg.replace(repulsion=False), None) is False
    # The config's own layout_mode does not decide it: the gate offers all three.
    assert repulsion_is_reachable(cfg.replace(layout_mode="resample"), None) is True


def test_the_review_never_contradicts_the_selected_config(split):
    """The review describes what ships; it must not name a different winner.

    Median rank is not consistent under subsetting — the merged gate ranks 6 or 18 cells at
    once, and a per-gate slice of it ranks 2 or 3 — so a review that recomputed an argmin
    inside the slice could report ``prior_mode ships correlated`` beside a selected config
    saying ``iid``. It did, before this test. The review is anchored on the shipped option and
    answers only the question the rule actually poses: is that option's margin real.
    """
    _, training, _ = split

    class Uneven:
        """Scores that make the merged gate's argmin differ from a per-gate slice's."""

        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            base = {
                ("correlated", "zinb-flow"): [0.90, 0.10, 0.90, 0.10, 0.90, 0.10],
                ("iid", "zinb-flow"): [0.50, 0.55, 0.52, 0.53, 0.51, 0.54],
            }.get((cfg.prior_mode, cfg.expr_mode), [0.2] * 6)
            return dict(zip(METRIC_NAMES, base, strict=True))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=Uneven(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )
    for review in result.reviews:
        if review.pinned or "not comparable" in review.reason:
            continue
        assert review.winner == str(getattr(result.config, review.gate)), (
            f"{review.gate}: review says {review.winner}, config says "
            f"{getattr(result.config, review.gate)}"
        )


def test_a_failure_recorded_by_an_earlier_process_still_reaches_the_report(split, tmp_path):
    """A ``--prewarm`` shard's failure is in the checkpoint, not in this process's memory.

    Without this, the resumed run reports "every candidate scored" about a table that holds a
    cell nothing could score — the exact reading a reader would most want corrected.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)
    cache = ScoreCache(tmp_path / "scores.csv")

    class Trips:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.text_emb_mode == "lookup":
                raise ExpressionError("detection rate 0.47 against 1.0000")
            return dict.fromkeys(METRIC_NAMES, 0.7)

    common = {
        "seed": SEED,
        "pinned": {"layout_mode": "resample"},
        "pinned_reason": "R11.",
        "checkpoint": cache,
    }
    with pytest.warns(CandidateFailedWarning):
        first = run_selection(training, base, scorer=Trips(), **common)
    assert len(first.failures) == 1

    # Second run: everything is a cache hit, so nothing raises — and the failure must survive.
    class NeverCalled:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            raise AssertionError("every cell should have come from the checkpoint")

    second = run_selection(training, base, scorer=NeverCalled(), **common)
    assert [label for label, _s, _w in second.failures] == ["text_emb_mode=lookup"]
    assert "earlier process" in second.failures[0][2]
    text = write_selection_report(second, tmp_path / "report.md").read_text()
    assert "ranked last" in text
    assert "text_emb_mode=lookup" in text


# --------------------------------------------------------------------------------------
# inert gates — the fourth control-that-cannot-fire
# --------------------------------------------------------------------------------------


def test_inertness_is_derived_by_running_the_path_not_declared(split):
    """``cross-mix`` makes ``prior_mode`` / ``text_emb_mode`` inert, and nothing lists that.

    ``infer/generate.py::_expression`` returns from ``_cross_mix`` before ``prior_latent``,
    ``flow.sample``, the decoder and the gene embeddings are reached, so those gates cannot
    change a single emitted count under it. :func:`inert_gates` establishes that by *running*
    each option and comparing counts bitwise — so a future edit that creates a new inert path
    is caught by the person who introduces it rather than by a reversal months later.

    Asserted in both directions: inert under ``cross-mix``, live under ``zinb-flow``.
    """
    vol, training, _ = split
    scorer = FitScorer(training, lambda cfg: build_embeddings(cfg, vol))
    base = t09_cfg(train_steps=2).replace(layout_mode="resample")
    probe = lambda cfg: scorer.inertness_probe(cfg, seed=SEED)
    gates = [("prior_mode", ("correlated", "iid")), ("text_emb_mode", ("medcpt", "lookup"))]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dead = inert_gates(probe, base.replace(expr_mode="cross-mix"), gates, seed=SEED)
        live = inert_gates(probe, base.replace(expr_mode="zinb-flow"), gates, seed=SEED)

    assert set(dead) == {"prior_mode", "text_emb_mode"}, (
        "cross-mix copies donor counts verbatim; neither the GRF nor the gene embeddings "
        f"can reach the output. Got {sorted(dead)}"
    )
    assert "prior_mode" not in live, "the GRF reaches the output under zinb-flow"
    assert "text_emb_mode" not in live, "the gene embeddings reach the decoder under zinb-flow"


def test_a_gate_inert_under_the_incumbent_is_re_ordered_not_scored_there(split):
    """The defect this exists to stop: a 0.0000 margin reported as a decision.

    On real STARmap the merged gate selected ``expr_mode=cross-mix``, and ``text_emb_mode``
    was then coordinate-descended under it — the one configuration where the open-vocabulary
    channel cannot act. Both options scored identically and the gate reported a margin of
    exactly 0.0000, which is an absence of measurement wearing the clothes of a perfect tie.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50)

    class DeadUnderCrossMix:
        """text_emb_mode changes nothing under cross-mix, and matters under zinb-flow."""

        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.expr_mode == "cross-mix":
                return dict.fromkeys(METRIC_NAMES, 0.90)  # identical for both options
            value = 0.70 if cfg.text_emb_mode == "medcpt" else 0.50
            return dict.fromkeys(METRIC_NAMES, value)

        def inertness_probe(self, cfg: Config, *, seed: int):
            # cross-mix emits the same counts whatever text_emb_mode / prior_mode say.
            if cfg.expr_mode == "cross-mix":
                return np.zeros((4, 3))
            return np.full((4, 3), {"medcpt": 1.0, "lookup": 2.0}[cfg.text_emb_mode])

    with pytest.warns(InertGateWarning, match="re-ordered"):
        result = run_selection(
            training,
            base,
            seed=SEED,
            scorer=DeadUnderCrossMix(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )

    assert result.config.expr_mode == "cross-mix", "the merged gate's own winner is unchanged"
    assert "text_emb_mode" in result.inert_notes
    assert "re-ordered" in result.inert_notes["text_emb_mode"]
    # ...and the gate was actually decided, on evidence, rather than tied at 0.0000.
    assert result.config.text_emb_mode == "medcpt"
    scored = [c for c in result.candidates if c.gate == "text_emb_mode"]
    assert scored, "the gate must still be scored, somewhere it is live"
    assert {c.overrides["expr_mode"] for c in scored} == {"zinb-flow"}


def test_a_gate_inert_everywhere_is_refused(split):
    """No live incumbent means no measurement, and shipping a 0.0000 tie would be a fallback."""
    _, training, _ = split

    class DeadEverywhere:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            return dict.fromkeys(METRIC_NAMES, 0.5)

        def inertness_probe(self, cfg: Config, *, seed: int):
            return np.zeros((4, 3))

    with pytest.raises(InertGateError, match="inert under every configuration"):
        run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=DeadEverywhere(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )


def test_the_report_names_an_inert_gate(split, tmp_path):
    """A re-ordered gate is a fact about the run, so a reader must not have to infer it."""
    _, training, _ = split

    class DeadUnderCrossMix:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.expr_mode == "cross-mix":
                return dict.fromkeys(METRIC_NAMES, 0.90)
            return dict.fromkeys(METRIC_NAMES, 0.70 if cfg.text_emb_mode == "medcpt" else 0.50)

        def inertness_probe(self, cfg: Config, *, seed: int):
            if cfg.expr_mode == "cross-mix":
                return np.zeros((4, 3))
            return np.full((4, 3), {"medcpt": 1.0, "lookup": 2.0}[cfg.text_emb_mode])

    with pytest.warns(InertGateWarning):
        result = run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=DeadUnderCrossMix(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )
    text = write_selection_report(result, tmp_path / "r.md").read_text()
    assert "were **inert** under the incumbent" in text
    assert "0.0000" in text
    assert "evidence from there, not from the shipped cell" in text


def test_selection_folds_are_two_on_a_four_section_stack(split):
    """Tier-1 STARmap's ``paper_2_4_6`` leaves four training sections, so n = 2, not 3.

    ``selection_folds`` takes the *interior* sections — a boundary fold would decide gates on
    the worst regime (open risk R3) — so ``Config.selection_n_folds = 3`` cannot be honoured
    on a four-section stack. Every gate decision in a tier-1 selection is therefore a mean of
    **two** numbers, and :func:`fold_scores` exists so a caller can see which fold moved.
    Pinned as a test because the fold count is invisible in the report's six columns.
    """
    _, training, _ = split
    cfg = t09_cfg()
    assert cfg.selection_n_folds == 3
    four = TrainingVolume(
        specimen_id=training.specimen_id,
        sections=list(training.sections)[:4],
        gene_names=training.gene_names,
        celltype_names=training.celltype_names,
        region_names=training.region_names,
        flattened_sections=training.flattened_sections,
    )
    assert len(selection_folds(four, cfg)) == 2, "interior of 4 sections is 2"


def test_a_gate_decided_elsewhere_is_undetermined_not_shipped(split):
    """SPEC_QUESTIONS C34. ``selected.yaml`` must not carry a value the shipped config cannot
    support.

    When ``text_emb_mode`` is inert under the shipped ``expr_mode=cross-mix``, the search
    measures it under ``zinb-flow`` — but that winner is evidence about a configuration this
    dataset does not ship, and under the one it *does* ship the gate changes no emitted count.
    So the gate is recorded **undetermined**, the winner stays in the report, and the selected
    config does not adopt it.
    """
    _, training, _ = split
    base = t09_cfg(train_steps=50).replace(text_emb_mode="lookup")

    class DeadUnderCrossMix:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.expr_mode == "cross-mix":
                return dict.fromkeys(METRIC_NAMES, 0.90)
            return dict.fromkeys(METRIC_NAMES, 0.70 if cfg.text_emb_mode == "medcpt" else 0.50)

        def inertness_probe(self, cfg: Config, *, seed: int):
            if cfg.expr_mode == "cross-mix":
                return np.zeros((4, 3))
            return np.full((4, 3), {"medcpt": 1.0, "lookup": 2.0}[cfg.text_emb_mode])

    with pytest.warns(InertGateWarning):
        result = run_selection(
            training,
            base,
            seed=SEED,
            scorer=DeadUnderCrossMix(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )

    assert result.config.expr_mode == "cross-mix"
    assert "text_emb_mode" in result.undetermined
    assert result.elsewhere_winner["text_emb_mode"] == "medcpt", "it did win, elsewhere"
    assert result.config.text_emb_mode == "lookup", (
        "the elsewhere winner must NOT be adopted: the shipped config cannot support it"
    )
    assert "not written into" in result.undetermined["text_emb_mode"]

    # ...and the report says so, in both places a reader looks.


def test_the_report_marks_an_undetermined_gate_and_prints_the_fold_count(split, tmp_path):
    """The two things a reader must not have to infer: what was not decided, and on how many
    folds the margins that *were* decided rest."""
    _, training, _ = split

    class DeadUnderCrossMix:
        def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
            if cfg.expr_mode == "cross-mix":
                return dict.fromkeys(METRIC_NAMES, 0.90)
            return dict.fromkeys(METRIC_NAMES, 0.70 if cfg.text_emb_mode == "medcpt" else 0.50)

        def inertness_probe(self, cfg: Config, *, seed: int):
            if cfg.expr_mode == "cross-mix":
                return np.zeros((4, 3))
            return np.full((4, 3), {"medcpt": 1.0, "lookup": 2.0}[cfg.text_emb_mode])

    with pytest.warns(InertGateWarning):
        result = run_selection(
            training,
            t09_cfg(train_steps=50),
            seed=SEED,
            scorer=DeadUnderCrossMix(),
            pinned={"layout_mode": "resample"},
            pinned_reason="R11.",
        )
    text = write_selection_report(result, tmp_path / "r.md").read_text()

    assert "**UNDETERMINED**" in text, "the selected-config table must not print a value"
    assert "Gates **UNDETERMINED** for this dataset" in text
    assert "does not ship" in text
    # the fold count, beside every margin
    n = len(selection_folds(training, t09_cfg()))
    assert f"**n = {n}**" in text
    assert "how many LOSO folds each margin is a mean of" in text
    assert all(r.n_folds == n for r in result.reviews if not r.pinned)


def test_the_score_cache_is_keyed_on_the_volume_as_well_as_the_config(split, tmp_path):
    """A score depends on data the ``Config`` does not describe.

    C33 proved it the hard way: widening ``Volume.bbox`` to span the sections' slabs changed
    every score while leaving every config hash identical, so a cache keyed on the config
    alone served pre-fix numbers straight into a post-fix run — silently, because a cache hit
    looks exactly like a cache hit. A changed volume must **miss**.
    """
    _, training, _ = split
    cfg = t09_cfg(train_steps=50)
    scores = dict.fromkeys(METRIC_NAMES, 0.5)

    cache = ScoreCache(tmp_path / "scores.csv", volume_key=volume_cache_key(training))
    cache.put(cfg, 50, "cell", scores)
    assert cache.get(cfg, 50) == scores

    # Same file, same config, a volume whose geometry differs -> miss, not a stale hit.
    shorter = TrainingVolume(
        specimen_id=training.specimen_id,
        sections=list(training.sections)[:-1],
        gene_names=training.gene_names,
        celltype_names=training.celltype_names,
        region_names=training.region_names,
        flattened_sections=training.flattened_sections,
    )
    assert volume_cache_key(shorter) != volume_cache_key(training)
    assert (
        ScoreCache(tmp_path / "scores.csv", volume_key=volume_cache_key(shorter)).get(cfg, 50)
        is None
    )

    # ...and the C33 case exactly: the same sections at a different thickness are the same
    # cells and the same config, but a different bbox — and so a different key.
    thicker = TrainingVolume(
        specimen_id=training.specimen_id,
        sections=[replace(sec, thickness=float(sec.thickness) * 2.0) for sec in training.sections],
        gene_names=training.gene_names,
        celltype_names=training.celltype_names,
        region_names=training.region_names,
        flattened_sections=training.flattened_sections,
    )
    assert thicker.n_cells == training.n_cells
    assert not np.allclose(np.asarray(thicker.bbox), np.asarray(training.bbox))
    assert volume_cache_key(thicker) != volume_cache_key(training)
