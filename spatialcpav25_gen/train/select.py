"""Automatic per-dataset configuration selection (T09 §3).

The user never tunes a flag: every gate — ``layout_mode``, ``prior_mode``, ``expr_mode``,
``text_emb`` and, jointly, the **training budget** with T08's **metric-aware weights** — is
chosen by internal leave-one-section-out over *training* sections. That is both a usability
claim in the paper and the no-regression guarantee: ``layout_mode="resample"`` +
``expr_mode="cross-mix"`` reproduces the previous version, so if the new machinery does not
help on a dataset it is switched off automatically.

The joint gate, and why it cannot be coordinate-descended
---------------------------------------------------------
T08 measured that the metric-aware terms **lose** at 1200 steps and **win** at 2400 on four
of six statistics:

===========================  =========  ========  =========  ========
statistic                    off@1200   on@1200   off@2400   on@2400
===========================  =========  ========  =========  ========
reconstruction (nats/pair)   1.5901     1.6843    **1.5703** 1.5885
gene-gene Frobenius          9.000      11.154    9.049      **8.489**
Moran's MAE                  0.0287     0.0408    0.0339     **0.0279**
marker-depth r               0.978      0.967     0.983      **0.990**
===========================  =========  ========  =========  ========

Coordinate descent varies one gate at a time from the incumbent, so starting at
``(1200, weights off)`` it would score ``(1200, weights on)``, lose, conclude the weights are
harmful, and **never reach** ``(2400, weights on)`` — the cell that wins. So the budget and
the weights are one gate with four cells, scored together, and each cell is fitted **at the
budget it names**: a reduced-epoch fit of a ``2x`` candidate *is* the ``1x`` candidate, and
the comparison would return "no difference" by construction.

How a candidate is scored
-------------------------
One fold = one **training** section, hidden from retrieval and generated in its place, then
compared against the real thing on the six target metrics. There is no cell correspondence
between a generated section and a real one, and none is needed: every one of the six is a
distribution-level statistic. Scoring by *generation* rather than by T08's
``reconstruct_hidden`` is what makes ``layout_mode`` and ``expr_mode`` mean anything at all —
the reconstruction path always runs the flow at the real cells' positions, so it cannot see
either gate.

.. note::
   The six metrics are computed here with T08's differentiable kernels rather than with the
   vendored ``bench3/evaluate_paper.py`` implementations, because ``eval/metrics.py`` is
   T10's module and vendoring it early would pin the scoreboard twice. The **names** match
   T10's, so the selector's table and the paper's table are about the same six quantities,
   and T10 re-scores the selected config with the vendored code. Recorded in
   ``SPEC_QUESTIONS`` and in the report every run writes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import rankdata
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import (
    Section,
    TrainingVolume,
    clamp_config_to_volume,
)
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.losses.metric_aware import (
    _type_grid_size,
    gearys_c,
    knn_weight_graph,
    marker_genes,
    morans_i,
    normalised_linear,
    profile_axis,
    require_training_volume,
    soft_depth_profile,
    soft_field_profile,
)
from spatialcpav25_gen.model.expression import ExpressionError

if TYPE_CHECKING:  # pragma: no cover
    from spatialcpav25_gen.model.embeddings import EntityEmbeddings
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "ALL_GATES",
    "CAPABILITY_CLAIM",
    "FIT_INVARIANT_GATES",
    "FULL_BUDGET_GATES",
    "GATES",
    "METRIC_NAMES",
    "SCORING_FAILURES",
    "TRAINING_FREE_OPTIONS",
    "V20_CONFIG",
    "Candidate",
    "CandidateFailedWarning",
    "FitScorer",
    "GateReview",
    "InertGateError",
    "InertGateWarning",
    "ScoreCache",
    "SelectionError",
    "SelectionResult",
    "average_folds",
    "calibration_chunks",
    "capability_tie_break",
    "descent_gates",
    "fold_scores",
    "full_budget_gate_cells",
    "full_budget_gates",
    "incumbent_is_unconverged",
    "inert_gates",
    "joint_gate_cells",
    "live_incumbent_for",
    "module_morans_agreement",
    "rank_candidates",
    "repulsion_is_reachable",
    "review_gates",
    "run_selection",
    "section_scores",
    "select_config",
    "selection_folds",
    "selection_scores",
    "volume_cache_key",
    "write_selection_report",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

METRIC_NAMES: Final[tuple[str, ...]] = (
    "morans_pearson",
    "gearys_pearson",
    "umap_mixing",
    "marker_field_r",
    "marker_depth_r",
    "celltype_localization",
)
"""The six target metrics, by ``specs/10``'s names. Higher is better for every one of them,
which is what lets the score be a median **rank**."""

ALL_GATES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("layout_mode", ("field", "hybrid", "resample")),
    ("prior_mode", ("correlated", "iid")),
    ("expr_mode", ("zinb-flow", "cross-mix", "auto-blend")),
    ("text_emb_mode", ("medcpt", "lookup")),
)
"""Every gate and its options (``specs/09`` §3's table).

``text_emb`` / ``medcpt+residual`` / ``lookup-only`` are the design document's spellings of
``text_emb_mode`` / ``medcpt`` / ``lookup`` (SPEC_QUESTIONS C7: the ``Config`` spelling wins).
The budget and the metric-aware weights are **not** here — they are one joint gate, scored by
:func:`joint_gate_cells`. Which of these are coordinate-descended and which are scored jointly
at full budget is decided by :data:`TRAINING_FREE_OPTIONS`, not fixed here."""

TRAINING_FREE_OPTIONS: Final[dict[str, tuple[str, ...]]] = {
    "layout_mode": ("resample",),
    "prior_mode": ("iid",),
    "expr_mode": ("cross-mix",),
    "text_emb_mode": (),
}
"""The **training-free-option rule** (``specs/09`` §3), in machine-readable form.

An option is *training-free* when it reaches its final behaviour without training, because it
copies real data instead of generating it: ``resample`` reuses real cell positions, ``iid``
never queries the fitted field, ``cross-mix`` emits donor counts verbatim. Such an option is
already at full strength at any budget while its rivals are not, so a reduced-budget
comparison of that gate measures the budget rather than the gate.

**Every gate must appear here**, including gates with no training-free option — an empty tuple
is the classification "all options train", not a missing entry, and
:func:`_check_gate_classification` refuses a gate that was added to :data:`ALL_GATES` without
one. That is the point of the rule: a future gate is classified when it is added, rather than
being discovered by a reversal months later.

Measured (``reports/r8_budget_grid.md``, open risk R8): at 25 % of the budget ``cross-mix`` won
the ``expr_mode`` gate under both priors and at full budget it came **last** under both, and
``iid`` won ``prior_mode`` at 25 % on exactly the two expression paths where the prior can act.
From 600 to 2400 steps ``morans_pearson`` gains **+0.3432** for ``zinb-flow`` and **-0.0180**
for ``cross-mix``."""


CAPABILITY_CLAIM: Final[dict[str, dict[str, int]]] = {
    "layout_mode": {"resample": 0, "field": 1, "hybrid": 1},
    "prior_mode": {"iid": 0, "correlated": 1},
    "expr_mode": {"cross-mix": 0, "zinb-flow": 1, "auto-blend": 2},
    "text_emb_mode": {"lookup": 0, "medcpt": 1},
}
"""How much each option **claims**, for ``specs/09`` §3's capability tie-break.

``0`` is the option that claims nothing beyond reusing real data; higher numbers add a
mechanism the paper takes credit for. ``resample`` reuses real positions while ``field`` and
``hybrid`` generate them; ``iid`` ignores the GRF and ``correlated`` uses it; ``cross-mix``
copies donor counts, ``zinb-flow`` generates them, and ``auto-blend`` adds T09's
uncertainty-gated anchoring on top of the flow — hence ``2``; ``lookup`` disables the text
channel and ``medcpt`` keeps it live, which is the open-vocabulary claim.

Every gate and every one of its options must appear, and
:func:`_check_gate_classification` refuses a gate that was added without a full classification.

The levels exist because "prefer the capability" pulls in **opposite directions** in the two
cases T09 measured, and only the claim level separates them — see
:func:`capability_tie_break`."""


def capability_tie_break(
    candidates: Sequence[Candidate], gate: str, cfg: Config
) -> tuple[Candidate, str]:
    """Choose between candidates separated by less than the reproducibility envelope.

    ``specs/09`` §3's tie-break. Returns ``(winner, reason)``; the reason is written into the
    report so a reader can see the choice was made on capability rather than on measurement.

    Below ``Config.claim_tie_break_envelope`` the rank ordering **is not evidence** — T09
    measured a run-to-run envelope as wide as the gap it was being asked to resolve — so
    capability decides instead. Two rules, and they pull in opposite directions, which is why
    :data:`CAPABILITY_CLAIM` is a level rather than a flag:

    1. **An exactly-identical rival proves the extra claim is inert.** Among tied candidates
       whose scores are equal to the last decimal, only the **lowest** claim level survives:
       the higher-claiming ones are the same model under a richer label, and shipping that
       label would claim a mechanism no emitted value depends on.
    2. **Among what survives, the highest claim level wins.** Options that differ *within* the
       envelope differ for some reason; the rank ordering cannot say which is better, so the
       option that exercises the capability is preferred to the one that disables it.

    The fixture supplies one case of each. ``auto-blend`` (claim 2) scored **bit-identically**
    to ``zinb-flow`` (claim 1) because the fitted ``w(v)`` is 0 at every knot, so rule 1 drops
    it and ``zinb-flow`` is the honest label. ``lookup`` (claim 0) outranked ``medcpt``
    (claim 1) by at most 0.011 — inside the envelope but *not* identical — so rule 2 selects
    ``medcpt`` and the open-vocabulary channel is not switched off on sub-noise evidence.

    When no rival is inside the envelope the ordinary rank winner is returned unchanged.
    """
    ranked = sorted(candidates, key=lambda c: c.rank)
    best = ranked[0]
    envelope = float(cfg.claim_tie_break_envelope)
    claims = CAPABILITY_CLAIM[gate]

    def separation(a: Candidate, b: Candidate) -> float:
        return max(
            abs(float(a.scores.get(m, 0.0)) - float(b.scores.get(m, 0.0))) for m in METRIC_NAMES
        )

    def claim(c: Candidate) -> int:
        return claims[str(c.overrides.get(gate))]

    tied = [c for c in ranked if separation(c, best) < envelope]
    if len(tied) < 2:
        return best, "decided on rank; no rival within the envelope"

    # Rule 1: an exactly-identical rival with a lower claim proves the extra claim is inert.
    live = [
        c
        for c in tied
        if not any(separation(c, other) == 0.0 and claim(other) < claim(c) for other in tied)
    ]
    # Rule 2: among what survives, prefer the highest claim level, then rank.
    winner = min(live, key=lambda c: (-claim(c), c.rank))
    if winner is best:
        return best, "the rank winner already exercises the highest live capability"
    if claim(winner) < claim(best):
        return winner, (
            f"within the {envelope:g} envelope, and {best.overrides.get(gate)}'s extra claim is "
            f"**inert** here — scores identical to {winner.overrides.get(gate)} — so the honest "
            "label wins rather than the richer one"
        )
    return winner, (
        f"tie-broken on capability: separated from the rank winner ({best.label}) by "
        f"{separation(winner, best):.4f} < the {envelope:g} envelope, and "
        f"{winner.overrides.get(gate)} exercises a headline capability the rank winner disables"
    )


def _check_gate_classification() -> None:
    """Refuse a gate that :data:`TRAINING_FREE_OPTIONS` or :data:`CAPABILITY_CLAIM` misses.

    The rules are only rules if adding a gate forces both questions — which budget may score
    it, and how a sub-envelope tie between its options is broken. Raised at import time so a
    gate added without a classification fails immediately rather than being scored at whatever
    budget the code happens to default to and tie-broken by whatever rank noise produced.
    """

    def _require_gates(name: str, classified: Sequence[str]) -> None:
        absent = [gate for gate, _ in ALL_GATES if gate not in classified]
        if absent:
            raise SelectionError(
                f"{name} does not classify {absent}. specs/09 §3 requires every gate to be "
                "classified when it is added, because the classification decides the gate's "
                "budget and how a sub-envelope tie is broken."
            )

    _require_gates("TRAINING_FREE_OPTIONS", list(TRAINING_FREE_OPTIONS))
    _require_gates("CAPABILITY_CLAIM", list(CAPABILITY_CLAIM))
    for gate, options in ALL_GATES:
        stray = set(TRAINING_FREE_OPTIONS[gate]) - set(options)
        if stray:
            raise SelectionError(
                f"TRAINING_FREE_OPTIONS[{gate!r}] names {sorted(stray)}, which are not options "
                f"of that gate ({list(options)})."
            )
        unclassified = set(options) - set(CAPABILITY_CLAIM[gate])
        if unclassified:
            raise SelectionError(
                f"CAPABILITY_CLAIM[{gate!r}] does not give a claim level to "
                f"{sorted(unclassified)}. Every option needs one: the level is what separates "
                "'prefer the capability' from 'do not credit an inert one'."
            )
        extra = set(CAPABILITY_CLAIM[gate]) - set(options)
        if extra:
            raise SelectionError(
                f"CAPABILITY_CLAIM[{gate!r}] names {sorted(extra)}, which are not options of "
                f"that gate ({list(options)})."
            )
    for gate, options in ALL_GATES:
        unknown = set(TRAINING_FREE_OPTIONS[gate]) - set(options)
        if unknown:
            raise SelectionError(
                f"TRAINING_FREE_OPTIONS[{gate!r}] names {sorted(unknown)}, which are not "
                f"options of that gate ({list(options)})."
            )


FIT_INVARIANT_GATES: Final[tuple[str, ...]] = ("layout_mode",)
"""Gates that provably do not enter the fit, so one model serves every option of them.

``layout_mode`` is read only at generation time: ``sample_layout`` is never called during
training, and ``_layout_term`` evaluates the intensity at the **real** cells' positions. Fitting
the fixture at ``field`` / ``hybrid`` / ``resample`` with one seed gives **bitwise identical**
weights across all 96 parameter and buffer tensors, which is what
``tests/test_select.py::test_layout_mode_does_not_enter_the_fit`` asserts — the cache in
:class:`FitScorer` is only sound while that test passes, so it fails loudly rather than silently
reusing a stale model if a future change makes training read the gate.

The saving is per dataset and not small: the merged full-budget gate is ``layout_mode`` x
``prior_mode`` x ``expr_mode`` = 18 cells, and 6 fits serve all 18. It also makes the comparison
*better* — the three ``layout_mode`` arms of a cell now differ by nothing but the layout, where
before they were three separate fits that happened to agree.

Scores are unchanged either way: identical weights produce identical scores. This is a cost fix,
not a numerical one, and it must stay that way — anything that changed a number here would mean
the invariant does not hold."""


def _check_pinned(pinned: Mapping[str, str] | None) -> dict[str, str]:
    """Validate a ``gate -> option`` pinning and return it as a plain dict.

    A pinned gate is one this run does **not** select: its value is fixed by evidence from
    somewhere else and the search must not re-open it. The refusals are Convention 6's —
    an unknown gate or an option that gate does not have would otherwise silently pin
    nothing and the report would claim a decision the run never made.
    """
    if not pinned:
        return {}
    options_by_gate = dict(ALL_GATES)
    out: dict[str, str] = {}
    for gate, option in pinned.items():
        if gate not in options_by_gate:
            raise SelectionError(
                f"pinned names gate {gate!r}, which is not a gate. The gates are "
                f"{sorted(options_by_gate)}."
            )
        if option not in options_by_gate[gate]:
            raise SelectionError(
                f"pinned[{gate!r}] = {option!r} is not an option of that gate "
                f"({list(options_by_gate[gate])})."
            )
        out[str(gate)] = str(option)
    if len(out) == len(options_by_gate):
        raise SelectionError(
            "pinned fixes every gate, so there is nothing to select. Pin fewer gates, or "
            "call the fit path directly with the config you already have."
        )
    return out


def full_budget_gates(
    pinned: Mapping[str, str] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the gates scored jointly at the selected budget, minus anything ``pinned``."""
    fixed = set(pinned or ())
    return tuple(
        (gate, options)
        for gate, options in ALL_GATES
        if TRAINING_FREE_OPTIONS[gate] and gate not in fixed
    )


def descent_gates(
    pinned: Mapping[str, str] | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the gates coordinate descent visits, minus anything ``pinned``."""
    fixed = set(pinned or ())
    return tuple(
        (gate, options)
        for gate, options in ALL_GATES
        if not TRAINING_FREE_OPTIONS[gate] and gate not in fixed
    )


FULL_BUDGET_GATES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = tuple(
    (gate, options) for gate, options in ALL_GATES if TRAINING_FREE_OPTIONS[gate]
)
"""The gates the rule disqualifies from reduced-budget scoring, scored **jointly** at the
selected budget by :func:`full_budget_gate_cells`. Jointly rather than one after another
because their errors compound through coordinate descent's ordering: on the fixture, fixing
``prior_mode="iid"`` first dropped ``zinb-flow`` from rank 2.5 to 3.0 *before* the ``expr_mode``
gate was scored."""

GATES: Final[tuple[tuple[str, tuple[str, ...]], ...]] = tuple(
    (gate, options) for gate, options in ALL_GATES if not TRAINING_FREE_OPTIONS[gate]
)
"""The gates that keep coordinate descent at ``Config.selection_reduced_epoch_frac`` — every
option trains, so a reduced fit compares like with like."""

V20_CONFIG: Final[dict[str, str]] = {"layout_mode": "resample", "expr_mode": "cross-mix"}
"""The previous version's behaviour, and the no-regression guarantee: this combination is
reachable by the gates above and is what the selector returns when the new components do not
help (``test_selector_can_recover_v20_config``)."""


SCORING_FAILURES: Final[tuple[type[Exception], ...]] = (ExpressionError,)
"""Exceptions that mean *this candidate is bad*, not *this run is broken*.

Deliberately narrow. ``ExpressionError`` is the emission guard
(``model/expression.py::assert_detection_rate``) refusing counts whose per-gene detection rate
is implausible against the training sections' — which an under-trained candidate genuinely is,
and which the reduced-budget cells of any selection genuinely are. Everything else — a shape
error, a missing field, a leakage refusal — still aborts, because none of those is a statement
about the candidate."""


class InertGateWarning(UserWarning):
    """A gate was inert under the incumbent and was measured under a different one."""


def _describe(overrides: Mapping[str, Any], gate: str) -> str:
    """Return the other gates' values, so a message can say *what* made this one inert."""
    return (
        ", ".join(f"{g}={overrides.get(g)}" for g, _ in ALL_GATES if g != gate and g in overrides)
        or "the base config"
    )


class CandidateFailedWarning(UserWarning):
    """A candidate could not be scored, and was ranked last instead of aborting the search."""


class SelectionError(ValueError):
    """Raised when a selection cannot be run as asked."""


class Scorer(Protocol):
    """Scores one candidate config at one budget. Returns the six metrics by name.

    The seam the acceptance tests use: a stub scorer makes the *search* testable without
    fitting ten models, which is what ``test_budget_and_metric_weights_are_selected_jointly``
    needs in order to fail on a one-gate-at-a-time selector.
    """

    def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
        """Return ``{metric name: value}`` for ``cfg`` fitted for ``steps`` steps."""
        ...


@dataclass(frozen=True)
class Candidate:
    """One scored point of the search.

    Attributes
    ----------
    gate
        Which gate this candidate varies (``"joint"`` for the budget x weights cells).
    label
        Human-readable name, e.g. ``layout_mode=hybrid`` or ``2x, weights on``.
    overrides
        The ``Config`` fields it sets, relative to the base config.
    steps
        The budget it was **actually fitted at**. Recorded per candidate because
        ``specs/09`` §3's third requirement is exactly that the budget gate's cells are not
        fitted at a common reduced budget.
    scores
        The six metrics.
    rank
        Median rank across the six metrics among the candidates it was compared with; lower
        is better. Filled in once its comparison group is complete.
    """

    gate: str
    label: str
    overrides: dict[str, Any]
    steps: int
    scores: dict[str, float]
    rank: float = float("nan")


@dataclass(frozen=True)
class SelectionResult:
    """Everything a selection run decided and everything it measured.

    Attributes
    ----------
    config
        The selected config: the base config with every chosen gate applied, including
        ``train_steps`` and the three metric-aware weights.
    joint
        The four cells of the ``{1x, 2x} x {off, spec weights}`` gate, all of them, scored
        together. Reported in full — the winner alone hides the interaction that makes this
        one gate.
    full_budget
        Every cell of the merged ``layout_mode`` x ``prior_mode`` x ``expr_mode`` gate, all
        scored at the selected budget under ``specs/09`` §3's training-free-option rule.
        Reported in full for the same reason as ``joint``.
    reduced_budget_escalated, escalating_metrics
        Whether condition (2) of ``specs/09`` §3's rule fired — the incumbent being unconverged
        at the reduced budget, so every remaining gate was scored at the selected one — and
        which metrics fired it.
    candidates
        Every candidate scored, in the order they were scored.
    fits
        ``(overrides, steps)`` for every fit the run issued, in order. The evidence for
        ``test_budget_gate_is_not_scored_at_a_reduced_budget``.
    dataset, seed
        Provenance.
    section_ids
        The training sections the folds ran on. No held-out section can appear here: the
        entry point takes a ``TrainingVolume``.
    undetermined
        ``{gate: why}`` for every gate this dataset **cannot decide**: it was inert under the
        incumbent that ships, so it was measured elsewhere and its winner is not written into
        :attr:`config`. ``selected.yaml`` records the gate as undetermined rather than
        carrying a value the shipped configuration cannot support (SPEC_QUESTIONS C34).
    elsewhere_winner
        ``{gate: option}`` — what won *there*. Reported, never shipped.
    inert_notes
        ``{gate: why}`` for every gate that was inert under the incumbent and had to be
        measured under a different one. Non-empty is a fact about the run that the report
        prints: the gate's answer is evidence from where it could be measured, not from the
        shipped cell.
    failures
        ``(label, steps, reason)`` for every candidate that could not be scored and was
        ranked last (:data:`SCORING_FAILURES`). Empty on a clean run; non-empty is a fact
        about the search that the report prints and a reader must see.
    reviews
        One :class:`GateReview` per gate: the rank winner, the margin to its closest rival,
        whether that margin is inside ``Config.claim_tie_break_envelope``, and what the
        capability tie-break shipped. :attr:`config` already carries the reviewed answer.
    pinned, pinned_reason
        Gates this run did **not** select, and the caller's one-sentence justification. A
        pinned gate is excluded from the merged gate and from coordinate descent, is fixed
        in every candidate, and is reported as pinned rather than as selected.
    """

    config: Config
    joint: list[Candidate]
    candidates: list[Candidate]
    fits: list[tuple[dict[str, Any], int]]
    dataset: str
    seed: int
    section_ids: tuple[str, ...] = ()
    full_budget: list[Candidate] = field(default_factory=list)
    reduced_budget_escalated: bool = False
    escalating_metrics: tuple[str, ...] = ()
    reviews: list[GateReview] = field(default_factory=list)
    failures: list[tuple[str, int, str]] = field(default_factory=list)
    inert_notes: dict[str, str] = field(default_factory=dict)
    undetermined: dict[str, str] = field(default_factory=dict)
    elsewhere_winner: dict[str, str] = field(default_factory=dict)
    pinned: dict[str, str] = field(default_factory=dict)
    pinned_reason: str = ""


# --------------------------------------------------------------------------------------
# the six metrics, on one (generated, real) pair
# --------------------------------------------------------------------------------------


def _safe_r(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> float:
    """Pearson r, returning 0.0 where a constant vector makes it undefined."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _normalised(counts: npt.NDArray[Any], cfg: Config) -> Tensor:
    """Library-size normalise to 1 on the linear scale. ``(N, G)`` -> ``(N, G)`` float32."""
    return normalised_linear(
        torch.from_numpy(np.asarray(counts, dtype=np.float32)), 1.0, float(cfg.metric_eps)
    )


def section_scores(
    gen_counts: npt.NDArray[Any],
    gen_coords: npt.NDArray[Any],
    gen_types: npt.NDArray[Any],
    real: Section,
    axis: Tensor,
    cfg: Config,
    *,
    n_types: int,
) -> dict[str, float]:
    """Return the six target metrics for one generated section against one real section.

    ``gen_coords`` and the real section's coordinates are **in-plane** ``(N, 2)``; the two
    sides have different cells and different cell counts, which every one of the six
    tolerates by construction — they are all distribution-level statistics.

    Returns ``{metric: value}`` over :data:`METRIC_NAMES`, higher better throughout.
    """
    gen_x = _normalised(gen_counts, cfg)
    real_counts = np.asarray(real.counts.todense(), dtype=np.float64)
    real_x = _normalised(real_counts, cfg)
    gen_p = np.asarray(gen_coords, dtype=np.float64)[:, :2]
    real_p = np.asarray(real.coords, dtype=np.float64)

    w_gen = knn_weight_graph(gen_p, cfg)
    w_real = knn_weight_graph(real_p, cfg)
    eps = float(cfg.metric_eps)
    morans = _safe_r(
        morans_i(gen_x, w_gen, eps=eps).numpy(), morans_i(real_x, w_real, eps=eps).numpy()
    )
    gearys = _safe_r(
        gearys_c(gen_x, w_gen, eps=eps).numpy(), gearys_c(real_x, w_real, eps=eps).numpy()
    )

    markers = marker_genes(real_x, w_real, cfg)
    bounds_x = (float(real_p[:, 0].min()), float(real_p[:, 0].max()))
    bounds_y = (float(real_p[:, 1].min()), float(real_p[:, 1].max()))
    grid = _type_grid_size(bounds_x[1] - bounds_x[0], cfg)
    sigma = float(cfg.profile_sigma_frac) * (bounds_x[1] - bounds_x[0]) / float(grid[0])
    field_gen = soft_field_profile(
        gen_x.index_select(1, markers),
        torch.from_numpy(gen_p.astype(np.float32)),
        grid,
        sigma,
        bounds=(bounds_x, bounds_y),
    ).numpy()
    field_real = soft_field_profile(
        real_x.index_select(1, markers),
        torch.from_numpy(real_p.astype(np.float32)),
        grid,
        sigma,
        bounds=(bounds_x, bounds_y),
    ).numpy()
    marker_field = float(
        np.mean([_safe_r(field_gen[..., g], field_real[..., g]) for g in range(field_gen.shape[2])])
    )

    gen_xyz = torch.from_numpy(
        np.concatenate([gen_p, np.full((gen_p.shape[0], 1), float(real.z))], axis=1).astype(
            np.float32
        )
    )
    real_xyz = torch.from_numpy(
        np.concatenate([real_p, np.full((real_p.shape[0], 1), float(real.z))], axis=1).astype(
            np.float32
        )
    )
    projected = (real_xyz @ axis).numpy()
    depth_bounds = (float(projected.min()), float(projected.max()))
    depth_sigma = (
        float(cfg.profile_sigma_frac)
        * (depth_bounds[1] - depth_bounds[0])
        / int(cfg.profile_n_bins)
    )
    depth_gen = soft_depth_profile(
        gen_x.index_select(1, markers),
        gen_xyz,
        axis,
        int(cfg.profile_n_bins),
        depth_sigma,
        bounds=depth_bounds,
    ).numpy()
    depth_real = soft_depth_profile(
        real_x.index_select(1, markers),
        real_xyz,
        axis,
        int(cfg.profile_n_bins),
        depth_sigma,
        bounds=depth_bounds,
    ).numpy()
    marker_depth = float(
        np.mean([_safe_r(depth_gen[:, g], depth_real[:, g]) for g in range(depth_gen.shape[1])])
    )

    types_gen = torch.nn.functional.one_hot(
        torch.from_numpy(np.asarray(gen_types, dtype=np.int64)), num_classes=n_types
    ).to(torch.float32)
    types_real = torch.nn.functional.one_hot(
        torch.from_numpy(np.asarray(real.cell_type, dtype=np.int64)), num_classes=n_types
    ).to(torch.float32)
    type_field_gen = soft_field_profile(
        types_gen,
        torch.from_numpy(gen_p.astype(np.float32)),
        grid,
        sigma,
        bounds=(bounds_x, bounds_y),
    ).numpy()
    type_field_real = soft_field_profile(
        types_real,
        torch.from_numpy(real_p.astype(np.float32)),
        grid,
        sigma,
        bounds=(bounds_x, bounds_y),
    ).numpy()
    localization = float(
        np.mean(
            [
                _safe_r(type_field_gen[..., c], type_field_real[..., c])
                for c in range(n_types)
                if float(types_real[:, c].sum()) > 0
            ]
        )
    )
    return {
        "morans_pearson": morans,
        "gearys_pearson": gearys,
        "umap_mixing": _mixing(gen_counts, real_counts, cfg),
        "marker_field_r": marker_field,
        "marker_depth_r": marker_depth,
        "celltype_localization": localization,
    }


def _mixing(gen_counts: npt.NDArray[Any], real_counts: npt.NDArray[Any], cfg: Config) -> float:
    """KNN mixing of generated and real cells in a shared embedding. 1 = indistinguishable.

    The shared embedding is the top-``Config.expr_pca_dim`` PCs of the **pooled**
    log-normalised expression — a linear stand-in for the scoreboard's UMAP, which is what a
    kNN mixing score actually consumes and which avoids a stochastic embedding inside a
    selector that has to be reproducible from a seed alone (Convention 3).

    The score is the mean over cells of the fraction of their
    ``Config.selection_mixing_knn_k`` neighbours drawn from the *other* group, divided by the
    fraction expected if the two clouds were identical. Capped at 1: a generated cloud that
    is *more* mixed than chance is not better than one that matches.
    """
    from scipy.spatial import cKDTree

    a = np.asarray(gen_counts, dtype=np.float64)
    b = np.asarray(real_counts, dtype=np.float64)
    pooled = np.concatenate([a, b], axis=0)
    totals = np.clip(pooled.sum(axis=1, keepdims=True), 1.0, None)
    values = np.log1p(pooled / totals * float(np.median(totals)))
    values = values - values.mean(axis=0, keepdims=True)
    n_components = int(min(cfg.expr_pca_dim, values.shape[0] - 1, values.shape[1]))
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    embedding = values @ vt[:n_components].T
    labels = np.concatenate([np.zeros(a.shape[0], dtype=int), np.ones(b.shape[0], dtype=int)])
    k = int(min(cfg.selection_mixing_knn_k, embedding.shape[0] - 1))
    neighbours = cKDTree(embedding).query(embedding, k=k + 1)[1][:, 1:]
    other = (labels[neighbours] != labels[:, None]).mean(axis=1)
    expected = np.where(labels == 0, b.shape[0], a.shape[0]) / float(labels.size - 1)
    return float(np.clip(np.mean(other / np.clip(expected, 1e-9, None)), 0.0, 1.0))


# --------------------------------------------------------------------------------------
# scoring a config on internal LOSO
# --------------------------------------------------------------------------------------


def selection_folds(vol: TrainingVolume, cfg: Config) -> list[Section]:
    """Return the training sections every candidate is scored on: interior, evenly spread.

    Interior because a boundary section has evidence on one side only (open risk R3) and a
    gate chosen on the boundary regime would be chosen on the worst 20 % of the stack; evenly
    spread and *fixed* because a candidate compared on different folds from its rival is not
    compared at all.
    """
    require_training_volume(vol, "selection_folds")
    interior = list(range(1, vol.n_sections - 1))
    if not interior:
        raise SelectionError(
            f"selection_folds: specimen {vol.specimen_id!r} has {vol.n_sections} training "
            "sections and no interior one; every fold would be the boundary regime"
        )
    n = int(min(max(int(cfg.selection_n_folds), 1), len(interior)))
    picks = np.linspace(0, len(interior) - 1, n).round().astype(int)
    return [vol.sections[interior[int(i)]] for i in dict.fromkeys(picks.tolist())]


def average_folds(per_fold: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average per-fold scores into the six. The one place folds become a single number."""
    return {name: float(np.mean([f[name] for f in per_fold])) for name in METRIC_NAMES}


def fold_scores(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int,
    anchor: Any | None = None,
) -> list[dict[str, float]]:
    """Score a fitted model on internal LOSO, **per fold**, unaveraged.

    :func:`selection_scores` is this averaged. Kept separate because the average is where the
    evidence goes missing: ``selection_folds`` returns the *interior* sections, so a
    four-section training stack — which is what tier-1 STARmap's ``paper_2_4_6`` holdout
    leaves — gives **two** folds however large ``Config.selection_n_folds`` is set. A gate
    decided on a mean of two numbers cannot be told apart from one decided on a single fold
    plus noise, and this is the function that lets a caller check.
    """
    return _fold_scores(model, vol, cfg, seed=seed, anchor=anchor)


def selection_scores(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int,
    anchor: Any | None = None,
) -> dict[str, float]:
    """Score a fitted model on internal LOSO: generate each fold's section and compare.

    Each fold's section is **excluded from its own retrieval pool** and generated in its
    place; the six metrics are then computed against the real section and averaged over
    folds. Leakage-free by type and by construction: ``vol`` is a ``TrainingVolume`` and the
    folds are its own sections.
    """
    return average_folds(_fold_scores(model, vol, cfg, seed=seed, anchor=anchor))


def _fold_scores(
    model: CTFFlow,
    vol: TrainingVolume,
    cfg: Config,
    *,
    seed: int,
    anchor: Any | None = None,
) -> list[dict[str, float]]:
    """Return the per-fold six, in fold order. The implementation both entry points share."""
    require_training_volume(vol, "selection_scores")
    axis = profile_axis(vol, cfg)
    per_fold: list[dict[str, float]] = []
    for index, hidden in enumerate(selection_folds(vol, cfg)):
        adata = generate_section(
            model,
            section_plane(hidden),
            vol,
            cfg,
            seed + index,
            exclude_z={float(hidden.z)},
            anchor=anchor,
        )
        types = np.asarray(
            [vol.celltype_names.index(v) for v in adata.obs[cfg.celltype_key]], dtype=np.int64
        )
        per_fold.append(
            section_scores(
                emitted_counts(adata),
                np.asarray(adata.obsm[cfg.coord_key], dtype=np.float64),
                types,
                hidden,
                axis,
                cfg,
                n_types=len(vol.celltype_names),
            )
        )
    return per_fold


@dataclass
class FitScorer:
    """The production scorer: fit a model at the given budget, then score it on internal LOSO.

    Attributes
    ----------
    vol
        The training volume every candidate is fitted and scored on.
    embeddings
        A factory ``Config -> EntityEmbeddings``. A factory rather than an instance because
        the embeddings carry **learned** parameters and every candidate is a fresh fit;
        reusing one object would let the first candidate's training leak into the rest.
    needs_repulsion
        Whether any candidate will *generate* under a ``layout_mode`` that reads the fitted
        interaction. ``sample_layout`` returns ``_resample_layout`` before it looks at
        ``repulsion``, so a search whose ``layout_mode`` is pinned to ``resample`` pays for a
        ``fit_repulsion`` no candidate can use — and pays a worse price than time, because
        ``fit_repulsion`` raises ``LayoutError`` on a point pattern with no soft-repulsion
        range and would abort the whole search over a quantity it is not using.

        A **property of the search, not of the candidate**: the fit cache is keyed with
        ``layout_mode`` normalised out (:data:`FIT_INVARIANT_GATES`), so one model serves every
        option of that gate. Deciding per candidate would make the shared model depend on which
        cell happened to be fitted first. :func:`run_selection` sets it from the options the
        gate will actually offer.
    """

    vol: TrainingVolume
    embeddings: Callable[[Config], EntityEmbeddings]
    needs_repulsion: bool = True
    _fits: dict[str, CTFFlow] = field(default_factory=dict, repr=False)

    def _fit_key(self, cfg: Config, steps: int, seed: int) -> str:
        """Identify the *fit*: the config with :data:`FIT_INVARIANT_GATES` normalised out.

        Two candidates differing only in a gate the fit never reads share a model.
        """
        canonical = cfg.replace(**{gate: getattr(Config(), gate) for gate in FIT_INVARIANT_GATES})
        return f"{canonical.content_hash()}:{int(steps)}:{int(seed)}"

    def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
        """Fit for ``steps`` optimiser steps at ``seed`` and return the six metrics.

        The fit is reused across candidates that differ only in a gate the fit does not read
        (:data:`FIT_INVARIANT_GATES`) — 6 fits serve the merged gate's 18 cells. Everything
        downstream of the weights is still computed per candidate: the anchor calibration
        reconstructs sections down the generation path, so it depends on ``layout_mode`` and is
        **not** cached, and ``selection_scores`` generates with the candidate's own config.
        """
        from spatialcpav25_gen.infer.calibrate import calibrate_anchor_weight

        model = self._fits.get(self._fit_key(cfg, steps, seed))
        if model is None:
            model = self._fit(cfg, steps=steps, seed=seed)
            self._fits[self._fit_key(cfg, steps, seed)] = model
        anchor = (
            calibrate_anchor_weight(model, self.vol, cfg, seed=seed, n_folds=1)
            if cfg.expr_mode == "auto-blend"
            else None
        )
        return selection_scores(model, self.vol, cfg, seed=seed, anchor=anchor)

    def _fit(self, cfg: Config, *, steps: int, seed: int) -> CTFFlow:
        """One fit. Separated from :meth:`__call__` so the cache has a single writer."""
        from spatialcpav25_gen.model.layout import fit_repulsion
        from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

        data = TrainingData.build(self.vol, cfg)
        model = CTFFlow(cfg, data, self.embeddings(cfg), grf_seed=seed)
        train_ctfflow(model, cfg, steps=steps, seed=seed)
        if cfg.repulsion and self.needs_repulsion:
            model.repulsion = fit_repulsion(self.vol, cfg, seed=seed + 1)
        return model

    def inertness_probe(self, cfg: Config, *, seed: int) -> FloatArray:
        """Emit one fold section's counts under ``cfg``, from an **untrained** model.

        Inertness is a property of which code paths ``_expression`` reaches, not of the
        weights, so no fit is needed: an untrained model routes through exactly the same
        branches. That is what makes :func:`inert_gates` affordable enough to run *before*
        a gate is scored rather than after its fits are already spent.

        The model is built once per ``(architecture, seed)`` and reused across options,
        because two options of a gate must differ in nothing but the gate.
        """
        from spatialcpav25_gen.infer.calibrate import calibrate_anchor_weight
        from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData

        # The probe asks whether the emitted counts *change*, never whether they are good, so
        # ``assert_detection_rate``'s plausibility band does not apply to it — and would stop
        # it outright, because an untrained decoder emits a detection rate nothing like the
        # tissue's. Widened to its maximum legal value (a rate is a fraction, so 1.0 admits
        # everything) identically for every option, so it cannot affect the comparison.
        cfg = cfg.replace(detection_rate_tol=1.0)
        key = f"probe:{self._fit_key(cfg, 0, seed)}"
        model = self._fits.get(key)
        if model is None:
            from spatialcpav25_gen.model.layout import fit_repulsion

            model = CTFFlow(
                cfg, TrainingData.build(self.vol, cfg), self.embeddings(cfg), grf_seed=seed
            )
            model.eval()
            # ``sample_layout`` refuses a drawing layout without the fitted interaction, and
            # ``repulsion_is_reachable``'s rule applies here too: ``resample`` never reads it.
            # The interaction depends on the volume and the config, not on training, so an
            # untrained probe carries the same one a fitted model would.
            if cfg.repulsion and cfg.layout_mode != "resample":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.repulsion = fit_repulsion(self.vol, cfg, seed=seed + 1)
            self._fits[key] = model
        hidden = selection_folds(self.vol, cfg)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Mirror __call__: auto-blend refuses to generate without a fitted w(v), and T09
            # forbids a default one. The probe has to exercise the same path the scorer does,
            # or it is not measuring the configuration the search would score.
            anchor = (
                calibrate_anchor_weight(model, self.vol, cfg, seed=seed, n_folds=1)
                if cfg.expr_mode == "auto-blend"
                else None
            )
            adata = generate_section(
                model,
                section_plane(hidden),
                self.vol,
                cfg,
                seed,
                exclude_z={float(hidden.z)},
                anchor=anchor,
            )
        return emitted_counts(adata)

    def release_fits(self) -> None:
        """Drop the cached models. A fit is ~50 MB and the merged gate holds six."""
        self._fits.clear()


# --------------------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------------------


def full_budget_gate_cells(
    base_cfg: Config, pinned: Mapping[str, str] | None = None
) -> list[tuple[str, dict[str, Any]]]:
    """Return every cell of the merged full-budget gate, as ``(label, overrides)``.

    The cartesian product of :data:`FULL_BUDGET_GATES` — on the current table
    ``layout_mode`` x ``prior_mode`` x ``expr_mode``, 3 x 2 x 3 = **18 cells**. Merged rather
    than descended one gate at a time because ``specs/09`` §3's training-free-option rule
    disqualifies all three from reduced-budget scoring *and* because their errors compound
    through the ordering: a wrong choice on an earlier gate biases every gate scored after it.

    ``base_cfg`` is unused except to keep the signature parallel to :func:`joint_gate_cells`;
    the cells are a property of the gate table, not of the config. A gate named in ``pinned``
    is dropped from the product — it is not being selected here, so offering the search its
    options would be scoring a decision that has already been made elsewhere.
    """
    del base_cfg
    gates = full_budget_gates(pinned)
    cells: list[tuple[str, dict[str, Any]]] = []
    names = [gate for gate, _ in gates]
    for combo in product(*[options for _, options in gates]):
        overrides = dict(zip(names, combo, strict=True))
        cells.append((", ".join(f"{k}={v}" for k, v in overrides.items()), overrides))
    return cells


def incumbent_is_unconverged(
    full: Mapping[str, float], reduced: Mapping[str, float], cfg: Config
) -> tuple[bool, tuple[str, ...]]:
    """Condition (2) of ``specs/09`` §3's rule, measured. Returns ``(fired, metrics)``.

    The same config scored at the selected budget and at the reduced one. When the reduced fit
    falls short by more than ``Config.selection_convergence_tol`` on at least
    ``Config.selection_convergence_min_metrics`` of the six, the reduced budget is not a usable
    proxy for **any** remaining gate: a gate decided there is decided on a model that behaves
    nothing like the shipped one, however fair the comparison between its options.

    Unlike :data:`TRAINING_FREE_OPTIONS` this cannot be declared in advance — it depends on the
    incumbent the search arrived at — so it is measured once per run, for one extra reduced
    fit, and the metrics that fired are returned so the report can name them.

    Only shortfalls count. A reduced fit that scores *higher* is not evidence of convergence
    either way, but it is not evidence that the proxy is broken, and on the fixture the
    training-free paths do exactly that on individual metrics.
    """
    fired = tuple(
        name
        for name in METRIC_NAMES
        if float(full.get(name, 0.0)) - float(reduced.get(name, 0.0))
        > float(cfg.selection_convergence_tol)
    )
    return len(fired) >= int(cfg.selection_convergence_min_metrics), fired


class ScoreCache:
    """Per-cell checkpoint for a selection run, so an interrupted one resumes.

    The full-budget gate is 18 fits at the selected budget — hours of compute — and a run that
    loses all of it to an interrupted session is not usable. Every scored cell is appended to a
    CSV **immediately and flushed**, keyed by the candidate config's
    :meth:`~spatialcpav25_gen.config.Config.content_hash` and its budget, and a re-run of the
    same selection skips what is already recorded.

    Keying on the full config hash rather than on the overrides is deliberate: a cell is only
    reusable if *every* field that could affect the score matches, so changing an unrelated
    ``Config`` field correctly invalidates the cache instead of silently reusing a stale score.

    Parameters
    ----------
    path
        The CSV. Created with a header if absent.
    on_write
        Optional callback run after each row is flushed, given the row's label. The report
        script uses it to commit the checkpoint, which is what makes the run survive losing
        the machine rather than merely the process.
    """

    def __init__(
        self,
        path: str | Path,
        on_write: Callable[[str], None] | None = None,
        *,
        volume_key: str = "",
    ) -> None:
        """Load any existing rows from ``path``; create it with a header if it is absent."""
        self.path = Path(path)
        self.on_write = on_write
        self.volume_key = str(volume_key)
        self._rows: dict[str, dict[str, float]] = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self._rows[row["key"]] = {name: float(row[name]) for name in METRIC_NAMES}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["key", "label", "steps", *METRIC_NAMES])

    def key(self, cfg: Config, steps: int) -> str:
        """Return the cache key: the volume, the config's content hash, and the budget.

        **The volume is part of the key**, and it has to be. A score depends on data the
        ``Config`` does not describe: the training volume's own geometry. C33 proved it —
        widening ``Volume.bbox`` to span the sections' slabs changed every score while leaving
        every config hash identical, so a cache keyed on the config alone silently served
        pre-fix numbers into a post-fix run. An empty ``volume_key`` reproduces the old
        behaviour for callers that have not got one, and is the reason this is a keyword.
        """
        stem = f"{cfg.content_hash()}:{int(steps)}"
        return f"{self.volume_key}:{stem}" if self.volume_key else stem

    def get(self, cfg: Config, steps: int) -> dict[str, float] | None:
        """Return the recorded scores for this cell, or ``None`` if it has not been run."""
        return self._rows.get(self.key(cfg, steps))

    def put(self, cfg: Config, steps: int, label: str, scores: dict[str, float]) -> None:
        """Append one scored cell and flush it, then run ``on_write``."""
        key = self.key(cfg, steps)
        self._rows[key] = dict(scores)
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [key, label, int(steps), *[f"{scores[name]:.6f}" for name in METRIC_NAMES]]
            )
            handle.flush()
        if self.on_write is not None:
            self.on_write(label)

    def __len__(self) -> int:
        """Return the number of cells already recorded."""
        return len(self._rows)


def joint_gate_cells(base_cfg: Config) -> list[tuple[str, dict[str, Any]]]:
    """Return the four cells of the ``{1x, 2x} x {weights off, spec weights}`` gate.

    Returned as ``(label, overrides)``, always all four and always in the same order, because
    ``specs/09`` §3 requires the run to report every cell rather than the winner: the reason
    the gate exists is an interaction, and an interaction is only visible in the full table.
    """
    budgets = [
        ("1x", int(base_cfg.train_steps)),
        (
            f"{base_cfg.selection_budget_multiple:g}x",
            round(float(base_cfg.selection_budget_multiple) * int(base_cfg.train_steps)),
        ),
    ]
    weights = [
        ("weights off", {"w_autocorr": 0.0, "w_profile": 0.0, "w_distribution": 0.0}),
        (
            "weights on",
            {
                "w_autocorr": float(base_cfg.selection_metric_weight),
                "w_profile": float(base_cfg.selection_metric_weight),
                "w_distribution": float(base_cfg.selection_metric_weight),
            },
        ),
    ]
    cells: list[tuple[str, dict[str, Any]]] = []
    for budget_label, steps in budgets:
        for weight_label, weight_overrides in weights:
            cells.append(
                (f"{budget_label}, {weight_label}", {"train_steps": steps, **weight_overrides})
            )
    return cells


def _median_ranks(candidates: Sequence[Candidate]) -> list[float]:
    """Median rank across the six metrics for each candidate. Lower is better.

    Ranks are averaged over ties (``scipy.stats.rankdata``'s ``"average"``), so two
    candidates that score identically on a metric share its rank rather than being ordered by
    their position in the list.
    """
    table = np.asarray(
        [[float(c.scores.get(name, float("nan"))) for name in METRIC_NAMES] for c in candidates],
        dtype=np.float64,
    )
    table = np.nan_to_num(table, nan=-np.inf)
    ranks = np.stack(
        [rankdata(-table[:, j], method="average") for j in range(table.shape[1])], axis=1
    )
    return [float(np.median(row)) for row in ranks]


def _ranked(candidates: list[Candidate]) -> list[Candidate]:
    """Return the candidates with their median rank filled in."""
    ranks = _median_ranks(candidates)
    return [
        Candidate(
            gate=c.gate,
            label=c.label,
            overrides=c.overrides,
            steps=c.steps,
            scores=c.scores,
            rank=r,
        )
        for c, r in zip(candidates, ranks, strict=True)
    ]


@dataclass(frozen=True)
class GateReview:
    """One gate's post-hoc decision record: the rank winner, the margin, and what ships.

    ``specs/09`` §3's capability tie-break is a rule about *gates*, but the search itself only
    ever takes ``min(rank)``. T09 applied the rule by hand to the table it had printed; nothing
    in the code did, so a persisted ``selected.yaml`` carried the rank winner while the record
    said the tie-break had decided two of the four gates. This closes that: every gate is
    reviewed against :func:`capability_tie_break` after the search, and the review is what the
    result reports and applies.

    Attributes
    ----------
    gate, options
        The gate reviewed, and the options it was compared over here.
    winner, runner_up
        Selected option and its closest rival on median rank.
    margin
        The largest absolute difference over the six metrics between winner and runner-up —
        the same statistic ``reports/envelope_synthetic.md`` measures the envelope of, so the
        comparison against ``Config.claim_tie_break_envelope`` is like for like.
    inside_envelope
        ``margin < Config.claim_tie_break_envelope``: the rank ordering is not evidence.
    flipped
        Whether the tie-break overturned the rank winner.
    reason
        The tie-break's own sentence, written into the report so a reader can see whether the
        gate was decided by measurement or by capability.
    pinned
        The gate was not selected here at all; ``winner`` is the value it was pinned to.
    n_folds
        How many LOSO folds the margin is a mean of. Printed beside every margin, not kept
        internally: ``selection_folds`` takes the *interior* sections, so tier-1 STARmap's
        four-section training stack gives **two** however large ``Config.selection_n_folds``
        is — and a margin that is a mean of two numbers reads exactly like one that is a mean
        of thirty unless the count is on the page.
    """

    gate: str
    options: tuple[str, ...]
    winner: str
    runner_up: str | None
    margin: float
    inside_envelope: bool
    flipped: bool
    reason: str
    pinned: bool = False
    n_folds: int = 0


def review_gates(
    candidates: Sequence[Candidate],
    selected: Config,
    cfg: Config,
    *,
    pinned: Mapping[str, str] | None = None,
    n_folds: int = 0,
) -> list[GateReview]:
    """Re-check every gate against the capability tie-break, holding the others selected.

    For each gate, the candidates compared are those that differ from the selected config in
    **that gate alone** — which is what makes the margin a statement about the gate rather
    than about the cell. That is the shape of T09 §13's re-check table, and it is the only
    well-defined way to apply a per-gate rule to a merged multi-gate table.

    Returns one :class:`GateReview` per gate, in :data:`ALL_GATES` order. A pinned gate is
    reported as pinned and never tie-broken.
    """
    fixed = dict(pinned or {})
    reviews: list[GateReview] = []
    for gate, options in ALL_GATES:
        if gate in fixed:
            reviews.append(
                GateReview(
                    gate=gate,
                    options=options,
                    winner=fixed[gate],
                    runner_up=None,
                    margin=float("nan"),
                    inside_envelope=False,
                    flipped=False,
                    reason="pinned: not selected on this dataset",
                    pinned=True,
                    n_folds=n_folds,
                )
            )
            continue
        others = {g for g, _ in ALL_GATES if g != gate}
        group: dict[str, Candidate] = {}
        for c in candidates:
            if gate not in c.overrides:
                continue
            if any(
                c.overrides.get(g, getattr(selected, g)) != getattr(selected, g) for g in others
            ):
                continue
            option = str(c.overrides[gate])
            # Several passes revisit the same option; the last scored one is the one the
            # search itself carried forward.
            group[option] = c
        if len(group) < 2:
            reviews.append(
                GateReview(
                    gate=gate,
                    options=options,
                    winner=str(getattr(selected, gate)),
                    runner_up=None,
                    margin=float("nan"),
                    inside_envelope=False,
                    flipped=False,
                    reason="not comparable here: fewer than two options scored against this "
                    "incumbent",
                    n_folds=n_folds,
                )
            )
            continue
        shipped = str(getattr(selected, gate))
        if shipped not in group:
            reviews.append(
                GateReview(
                    gate=gate,
                    options=tuple(group),
                    winner=shipped,
                    runner_up=None,
                    margin=float("nan"),
                    inside_envelope=False,
                    flipped=False,
                    reason="not comparable here: the selected option was not scored against "
                    "this incumbent",
                    n_folds=n_folds,
                )
            )
            continue

        # The review is anchored on the option the **search** selected, not on a rank
        # recomputed inside this two-option subgroup. Median rank is not consistent under
        # subsetting — a merged 18-cell gate and a 2-cell slice of it can name different
        # winners — so a subgroup argmin would make the review contradict ``result.config``
        # while claiming to describe it. Ranks are rewritten so the shipped option leads and
        # the rivals keep their relative order; ``capability_tie_break`` then answers the only
        # question left, which is whether the shipped option's margin is real.
        ranked_group = {c.overrides[gate]: c for c in _ranked(list(group.values()))}
        order = sorted(
            (opt for opt in ranked_group if opt != shipped),
            key=lambda opt: ranked_group[opt].rank,
        )
        anchored = [replace(ranked_group[shipped], rank=0.0)] + [
            replace(ranked_group[opt], rank=float(i + 1)) for i, opt in enumerate(order)
        ]
        winner, reason = capability_tie_break(anchored, gate, cfg)
        best = anchored[0]

        def _separation(a: Candidate, b: Candidate) -> float:
            return max(
                abs(float(a.scores.get(m, 0.0)) - float(b.scores.get(m, 0.0))) for m in METRIC_NAMES
            )

        rivals = anchored[1:]
        # "Closest rival" is the one with the smallest separation, because separation is the
        # quantity the envelope test is about: if the nearest rival is outside it, all are.
        rival = min(rivals, key=lambda c: _separation(best, c)) if rivals else None
        margin = float("nan") if rival is None else _separation(best, rival)
        reviews.append(
            GateReview(
                gate=gate,
                options=tuple(group),
                winner=str(winner.overrides[gate]),
                runner_up=None if rival is None else str(rival.overrides[gate]),
                margin=margin,
                inside_envelope=bool(
                    margin == margin and margin < float(cfg.claim_tie_break_envelope)
                ),
                flipped=winner is not best,
                reason=reason,
                n_folds=n_folds,
            )
        )
    return reviews


def repulsion_is_reachable(cfg: Config, pinned: Mapping[str, str] | None = None) -> bool:
    """Return whether any candidate can *generate* under a layout that reads the interaction.

    ``sample_layout`` returns ``_resample_layout`` before it looks at ``repulsion``, so the
    answer is no when ``Config.repulsion`` is off, and no when ``layout_mode`` is pinned to
    ``resample``. It is a property of the search rather than of a candidate because the fit
    cache normalises ``layout_mode`` out (:data:`FIT_INVARIANT_GATES`) and one model serves
    every option of that gate — deciding per candidate would make the shared model depend on
    which cell was fitted first.
    """
    if not cfg.repulsion:
        return False
    fixed = dict(pinned or {})
    options = (
        (fixed["layout_mode"],)
        if "layout_mode" in fixed
        else dict(ALL_GATES).get("layout_mode", (cfg.layout_mode,))
    )
    return any(mode != "resample" for mode in options)


class InertGateError(SelectionError):
    """A gate was about to be scored under an incumbent that makes its options inert."""


def inert_gates(
    probe: Callable[[Config], Any],
    cfg: Config,
    gates: Sequence[tuple[str, tuple[str, ...]]],
    *,
    seed: int = 0,
) -> dict[str, tuple[str, ...]]:
    """Return ``{gate: options}`` for every gate whose options this config makes **inert**.

    ``specs/09`` §3's fourth control-that-cannot-fire. A gate is inert under a configuration
    when changing it cannot change a single emitted count — so scoring it there measures
    nothing, and the two options come back separated by exactly 0.0000, which reads like a
    perfect tie and is in fact an absence of measurement.

    **The relation is derived, not declared.** ``expr_mode="cross-mix"`` returns from
    ``infer/generate.py::_expression`` before ``prior_latent``, ``flow.sample``, the decoder
    and the gene embeddings are ever reached, so ``prior_mode``, ``text_emb_mode``,
    ``decoder_mu_link`` and ``ell_*`` are all unreachable under it — but nothing here writes
    that list down. Instead each option is *run*: ``probe`` emits a section under it and the
    counts are compared bitwise. Two options that produce identical counts are inert, whatever
    the reason and whatever a future edit does to the branch structure. A hand-maintained
    table would have to be updated by the person who introduces the next inert path, which is
    precisely the person who does not know they have introduced one.

    Inertness is a property of the **code path**, not of the weights, so ``probe`` may return
    an untrained model's output: no fit is needed and the check costs one generation per
    option.

    Parameters
    ----------
    probe
        ``Config -> array-like`` emitting counts under that config. Anything comparable by
        :func:`numpy.array_equal`.
    cfg
        The incumbent the gates would be scored under.
    gates
        ``(gate, options)`` pairs to test.
    seed
        Passed through by the caller's ``probe``; recorded here so the check is reproducible
        (Convention 3).
    """
    del seed
    inert: dict[str, tuple[str, ...]] = {}
    for gate, options in gates:
        emitted = {option: np.asarray(probe(cfg.replace(**{gate: option}))) for option in options}
        first = options[0]
        same = tuple(
            option
            for option in options
            if option != first and np.array_equal(emitted[option], emitted[first])
        )
        if same:
            inert[gate] = (first, *same)
    return inert


def live_incumbent_for(
    gate: str,
    inert_under: Callable[[Config, str], bool],
    incumbent: Config,
    alternatives: Sequence[Candidate],
    base_cfg: Config,
) -> Candidate | None:
    """Return the best-ranked alternative incumbent under which ``gate`` is **live**.

    ``specs/09`` §3's re-ordering: a gate that cannot be measured where the search happens to
    stand is measured where it can be, rather than being reported as a tie. ``alternatives``
    is searched in rank order, so the gate is decided under the *best* configuration that can
    decide it at all — which keeps the comparison as close to the shipped model as the
    obstruction allows.
    """
    del incumbent
    for candidate in sorted(alternatives, key=lambda c: c.rank):
        cfg = base_cfg.replace(**candidate.overrides)
        if not inert_under(cfg, gate):
            return candidate
    return None


def volume_cache_key(vol: TrainingVolume) -> str:
    """Return a short fingerprint of the training volume, for :class:`ScoreCache`.

    Everything a score depends on that the ``Config`` does not describe: the sections, the cell
    and gene counts, and the **bounding box** — which C33 changed without touching a single
    config field, and which changed every score.
    """
    box = np.asarray(vol.bbox, dtype=np.float64).round(4).tolist()
    payload = json.dumps(
        {
            "specimen": vol.specimen_id,
            "sections": list(vol.section_ids),
            "n_cells": int(vol.n_cells),
            "n_genes": int(vol.n_genes),
            "bbox": box,
            "flattened": bool(vol.flattened_sections),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def rank_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Fill in each candidate's median rank within this comparison group.

    The public name for the ranking :func:`run_selection` uses internally, so a driver that
    pre-warms cells in parallel and then needs to know which one won ranks them with the same
    code the selector does rather than with a second implementation of "median rank".
    """
    return _ranked(list(candidates))


def run_selection(
    vol: TrainingVolume,
    base_cfg: Config,
    *,
    seed: int | None = None,
    scorer: Scorer | None = None,
    embeddings: Callable[[Config], EntityEmbeddings] | None = None,
    dataset: str = "synthetic",
    report_path: str | Path | None = None,
    checkpoint: ScoreCache | None = None,
    pinned: Mapping[str, str] | None = None,
    pinned_reason: str = "",
) -> SelectionResult:
    """Run the whole selection and return everything it measured. See :class:`SelectionResult`.

    Order of operations, and it is load-bearing:

    1. **the joint gate first** — all four cells of ``{1x, 2x} x {off, spec weights}``, each
       fitted at the budget it names (``specs/09`` §3's second and third requirements);
    2. **then the merged full-budget gate** — every cell of :func:`full_budget_gate_cells`,
       all at the *selected* budget, because ``specs/09`` §3's training-free-option rule
       disqualifies those gates from reduced-budget scoring and their errors compound if they
       are visited one at a time;
    3. then ``Config.selection_passes`` coordinate-descent passes over the gates that remain
       (:data:`GATES` — every option trains), each candidate fitted at
       ``Config.selection_reduced_epoch_frac`` of the selected budget.

    Every fit is recorded in :attr:`SelectionResult.fits`, and identical ``(overrides,
    steps)`` pairs are scored once and reused — coordinate descent revisits the incumbent on
    every pass and refitting it would be pure cost.

    Parameters
    ----------
    checkpoint
        Optional :class:`ScoreCache`, **keyed on the volume as well as the config** — see
        :meth:`ScoreCache.key`. The merged gate is scored at full budget, so a run that
        loses it to an interrupted session is not usable; with a checkpoint each scored cell
        is flushed to disk as it completes and a re-run skips what is already there.
    pinned
        ``gate -> option`` for gates this run must **not** select, because they were decided
        elsewhere. A pinned gate is fixed in the base config, dropped from the merged gate's
        product and from coordinate descent, and reported as pinned. ``pinned_reason`` is
        required alongside it: a gate removed from selection without a stated reason is
        indistinguishable in the report from a gate that was never a gate.
    pinned_reason
        One sentence saying why. Refused as empty when ``pinned`` is non-empty.
    """
    require_training_volume(vol, "run_selection")
    # ``specs/10`` §0's owed fix, applied here because ``specs/09`` §3 is where the spec puts
    # it: STARmap's 28-gene panel is narrower than ``Config.expr_pca_dim`` and every fit would
    # otherwise be refused by ``validate_config_against_volume``. Warns with both numbers.
    base_cfg = clamp_config_to_volume(base_cfg, vol)
    fixed = _check_pinned(pinned)
    if fixed and not pinned_reason.strip():
        raise SelectionError(
            "run_selection: pinned gates need a pinned_reason. A gate excluded from selection "
            "is a claim that the evidence for it is elsewhere, and the report has to name it."
        )
    if fixed:
        base_cfg = base_cfg.replace(**fixed)
    if scorer is None and embeddings is None:
        raise SelectionError(
            "run_selection: pass either a scorer or an embeddings factory. The default "
            "scorer fits a model per candidate and cannot build the text-grounded "
            "embeddings itself — they come from T02's cache, which this module has no "
            "business knowing about."
        )
    run_seed = int(base_cfg.seed) if seed is None else int(seed)
    score_fn: Scorer = (
        FitScorer(vol, embeddings, needs_repulsion=repulsion_is_reachable(base_cfg, fixed))  # type: ignore[arg-type]
        if scorer is None
        else scorer
    )

    fits: list[tuple[dict[str, Any], int]] = []
    failures: list[tuple[str, int, str]] = []
    inert_notes: dict[str, str] = {}
    undetermined: dict[str, str] = {}
    elsewhere: dict[str, str] = {}
    cache: dict[tuple[tuple[str, Any], ...], dict[str, float]] = {}

    def score(overrides: dict[str, Any], steps: int, label: str = "") -> dict[str, float]:
        key = (*sorted(overrides.items()), ("__steps__", steps))
        if key not in cache:
            cfg = base_cfg.replace(**overrides)
            recorded = None if checkpoint is None else checkpoint.get(cfg, int(steps))
            if recorded is not None:
                cache[key] = recorded  # resumed from a previous run; no fit issued
                if all(v == float("-inf") for v in recorded.values()):
                    # The cell failed in whichever process scored it — a --prewarm shard, or
                    # an earlier attempt. The report has to say so here too, or a resumed run
                    # claims "every candidate scored" about a table that contains a failure.
                    failures.append(
                        (
                            label or str(overrides),
                            int(steps),
                            "recorded as failed by an earlier process (checkpoint); see that "
                            "run's log for the exception",
                        )
                    )
            else:
                fits.append((dict(overrides), int(steps)))
                try:
                    cache[key] = score_fn(cfg, steps=int(steps), seed=run_seed)
                except SCORING_FAILURES as exc:
                    # A candidate that cannot emit plausible counts is a *ranking* fact, not a
                    # reason to lose a nine-hour search at its second cell. The commonest case
                    # is real and expected: ``assert_detection_rate``'s band is measured
                    # against the training sections, and on a dense panel like STARmap's
                    # (median per-gene detection 0.9999, ``specs/10`` §0) an under-trained
                    # candidate misses it — which is exactly what the reduced-budget cells are.
                    # So it ranks last, and it does so **loudly**: warned here, carried in
                    # ``SelectionResult.failures``, printed in the report, and refused
                    # outright if it takes a whole gate with it.
                    name = label or str(overrides)
                    failures.append((name, int(steps), f"{type(exc).__name__}: {exc}"))
                    warnings.warn(
                        f"selection candidate {name!r} at {int(steps)} steps "
                        f"could not be scored and is ranked last: {type(exc).__name__}: {exc}",
                        CandidateFailedWarning,
                        stacklevel=2,
                    )
                    cache[key] = dict.fromkeys(METRIC_NAMES, float("-inf"))
                if checkpoint is not None:
                    checkpoint.put(cfg, int(steps), label or str(overrides), cache[key])
        return cache[key]

    def _require_a_live_group(group: Sequence[Candidate], what: str) -> None:
        """Refuse a gate every option of which failed: that is not a ranking, it is a break."""
        if group and all(all(v == float("-inf") for v in c.scores.values()) for c in group):
            raise SelectionError(
                f"every candidate of {what} failed to score, so there is nothing to choose "
                "between. The reasons follow; fix the run rather than shipping the first "
                "label.\n  " + "\n  ".join(f"{lab} @ {st}: {why}" for lab, st, why in failures)
            )

    # 1. the joint gate, all four cells, each at its own budget.
    joint = _ranked(
        [
            Candidate(
                gate="joint",
                label=label,
                overrides=overrides,
                steps=int(overrides["train_steps"]),
                scores=score(overrides, int(overrides["train_steps"]), label),
            )
            for label, overrides in joint_gate_cells(base_cfg)
        ]
    )
    _require_a_live_group(joint, "the joint gate")
    best_joint = min(joint, key=lambda c: c.rank)
    incumbent: dict[str, Any] = dict(best_joint.overrides)
    candidates: list[Candidate] = list(joint)

    # 2. the merged full-budget gate: every cell, all at the selected budget.
    selected_steps = int(incumbent["train_steps"])
    full_budget = _ranked(
        [
            Candidate(
                gate="full_budget",
                label=label,
                overrides={**incumbent, **overrides},
                steps=selected_steps,
                scores=score({**incumbent, **overrides}, selected_steps, label),
            )
            for label, overrides in full_budget_gate_cells(base_cfg, fixed)
        ]
    )
    _require_a_live_group(full_budget, "the merged full-budget gate")
    candidates.extend(full_budget)
    incumbent = dict(min(full_budget, key=lambda c: c.rank).overrides)

    # 3. condition (2) of the rule, measured: is the reduced budget a usable proxy at all?
    # One extra fit of the incumbent at the reduced budget, against the selected-budget score
    # the search already has for it.
    reduced = max(
        1, round(float(base_cfg.selection_reduced_epoch_frac) * int(incumbent["train_steps"]))
    )
    escalated = False
    escalating_metrics: tuple[str, ...] = ()
    gates = descent_gates(fixed)
    if gates and reduced != selected_steps:
        escalated, escalating_metrics = incumbent_is_unconverged(
            score(incumbent, selected_steps, "incumbent @ selected"),
            score(incumbent, reduced, "incumbent @ reduced"),
            base_cfg,
        )
    descent_steps = selected_steps if escalated else reduced

    # 4. coordinate descent over the gates every option of which trains.
    #
    # Before a gate is scored, it is checked for **inertness** under the incumbent the search
    # arrived at (``specs/09`` §3). A gate whose options cannot change a single emitted count
    # there measures nothing, and reports a 0.0000 separation that reads like a perfect tie.
    # The relation is derived by running the generation path, not declared; see
    # :func:`inert_gates`.
    probe = getattr(score_fn, "inertness_probe", None)
    inert_cache: dict[tuple[str, str], bool] = {}

    def _is_inert(cfg: Config, gate: str) -> bool:
        if probe is None:
            return False
        key = (cfg.content_hash(), gate)
        if key not in inert_cache:
            options = dict(ALL_GATES)[gate]
            found = inert_gates(
                lambda c: probe(c, seed=run_seed), cfg, [(gate, options)], seed=run_seed
            )
            inert_cache[key] = gate in found
        return inert_cache[key]

    for _ in range(int(base_cfg.selection_passes)):
        for gate, options in gates:
            scoring_cfg = base_cfg.replace(**incumbent)
            measured_under: dict[str, Any] = dict(incumbent)
            note = ""
            if _is_inert(scoring_cfg, gate):
                alternative = live_incumbent_for(
                    gate, _is_inert, scoring_cfg, full_budget, base_cfg
                )
                if alternative is None:
                    raise InertGateError(
                        f"gate {gate!r} is inert under every configuration this search scored: "
                        "changing it cannot change a single emitted count, so there is nothing "
                        "to measure and a 0.0000 separation would be reported as a tie. "
                        "specs/09 §3 forbids scoring a gate under an incumbent that makes its "
                        "options inert. Re-run with an incumbent that exercises it, or pin the "
                        "gate and say so."
                    )
                measured_under = dict(alternative.overrides)
                note = (
                    f"inert under the incumbent ({_describe(incumbent, gate)}); "
                    f"re-ordered and measured under {alternative.label}"
                )
                inert_notes[gate] = note
                warnings.warn(
                    f"gate {gate!r} is inert under the incumbent and was re-ordered: "
                    f"measured under {alternative.label} instead. Its selected value is "
                    "evidence from there, not from the shipped cell.",
                    InertGateWarning,
                    stacklevel=2,
                )
            group = []
            for option in options:
                overrides = {**measured_under, gate: option}
                group.append(
                    Candidate(
                        gate=gate,
                        label=f"{gate}={option}" + (" (re-ordered)" if note else ""),
                        overrides=overrides,
                        steps=descent_steps,
                        scores=score(overrides, descent_steps, f"{gate}={option}"),
                    )
                )
            group = _ranked(group)
            _require_a_live_group(group, f"gate {gate}")
            candidates.extend(group)
            chosen = str(min(group, key=lambda c: c.rank).overrides[gate])
            if note:
                # SPEC_QUESTIONS C34, decided 2026-08-26. The gate was measured somewhere the
                # shipped configuration is not, so the shipped configuration cannot support
                # the answer: under the incumbent that ships, this gate changes no emitted
                # count, and writing the winner into ``selected.yaml`` would claim a decision
                # the shipped model cannot express. It is recorded **UNDETERMINED for this
                # dataset**, and the elsewhere-evidence is kept in the report, labelled.
                undetermined[gate] = (
                    f"{note}. Measured winner there: `{chosen}` — **evidence from a "
                    f"configuration this dataset does not ship**, so it is not written into "
                    f"the selected config."
                )
                elsewhere[gate] = chosen
                continue
            # The gate's *answer* is adopted; the rest of the re-ordered incumbent is not —
            # the search still stands where the merged gate left it.
            incumbent = {**incumbent, gate: chosen}

    # 5. the capability tie-break, per gate, on the table the search just produced
    # (``specs/09`` §3). ``min(rank)`` decided every gate above; below the reproducibility
    # envelope that ordering is not evidence, and the rule — not the ranking — decides.
    reviewed = base_cfg.replace(**incumbent)
    reviews = review_gates(
        candidates, reviewed, base_cfg, pinned=fixed, n_folds=len(selection_folds(vol, base_cfg))
    )
    flips = {r.gate: r.winner for r in reviews if r.flipped}
    if flips:
        incumbent = {**incumbent, **flips}
        reviewed = reviewed.replace(**flips)

    result = SelectionResult(
        config=reviewed,
        joint=joint,
        candidates=candidates,
        fits=fits,
        dataset=dataset,
        seed=run_seed,
        section_ids=tuple(s.section_id for s in selection_folds(vol, base_cfg)),
        full_budget=full_budget,
        reduced_budget_escalated=escalated,
        escalating_metrics=tuple(escalating_metrics),
        reviews=reviews,
        failures=failures,
        inert_notes=dict(inert_notes),
        undetermined=dict(undetermined),
        elsewhere_winner=dict(elsewhere),
        pinned=dict(fixed),
        pinned_reason=pinned_reason,
    )
    if report_path is not None:
        write_selection_report(result, report_path)
    return result


def select_config(
    vol: TrainingVolume,
    base_cfg: Config,
    *,
    seed: int | None = None,
    scorer: Scorer | None = None,
    embeddings: Callable[[Config], EntityEmbeddings] | None = None,
    dataset: str = "synthetic",
    report_path: str | Path | None = None,
    pinned: Mapping[str, str] | None = None,
    pinned_reason: str = "",
) -> Config:
    """Choose the per-dataset configuration by internal LOSO. ``specs/09`` §3's entry point.

    The signature the spec fixes is ``select_config(vol, base_cfg) -> Config``; everything
    else is an additive keyword. ``seed`` defaults to ``base_cfg.seed`` rather than to an
    implicit RNG, so the run is reproducible from the config alone (Convention 3), and
    :func:`run_selection` is the same call returning the whole table instead of only the
    winner.

    Raises ``TypeError`` on a :class:`~spatialcpav25_gen.data.schema.HeldOutSections` or a
    plain ``Volume``: the budget is the gate most easily fitted to a test set, so the type
    check is the guarantee (``test_selection_never_sees_heldout``).
    """
    return run_selection(
        vol,
        base_cfg,
        seed=seed,
        scorer=scorer,
        embeddings=embeddings,
        dataset=dataset,
        report_path=report_path,
        pinned=pinned,
        pinned_reason=pinned_reason,
    ).config


# --------------------------------------------------------------------------------------
# diagnostics and the report
# --------------------------------------------------------------------------------------


def module_morans_agreement(
    model: CTFFlow, vol: TrainingVolume, cfg: Config, *, seed: int
) -> list[dict[str, float | int | str]]:
    """Per-gene-module Moran's I agreement — a **diagnostic**, never a target (A2).

    ``specs/09`` §2, settled in SPEC_QUESTIONS A2: one *global* ``ell`` is calibrated, because
    ``ell`` parameterises the latent field and gene modules only exist downstream of the
    decoder, so "the ``ell`` for module m" is not a well-posed quantity. What can be reported
    is whether the single ``ell`` serves every module equally, and that is this table: per
    module, the mean generated and real Moran's I and their difference, on one interior
    training section generated with itself excluded.

    A poor table is evidence for the escalation the spec names (per-channel-group ``ell`` plus
    a tying loss) — a design change to be decided explicitly, not improvised in the
    calibration loop.
    """
    from spatialcpav25_gen.losses.sefl import gene_modules

    require_training_volume(vol, "module_morans_agreement")
    hidden = selection_folds(vol, cfg)[0]
    adata = generate_section(
        model, section_plane(hidden), vol, cfg, seed, exclude_z={float(hidden.z)}
    )
    gen_x = _normalised(emitted_counts(adata), cfg)
    real_x = _normalised(np.asarray(hidden.counts.todense(), dtype=np.float64), cfg)
    i_gen = morans_i(
        gen_x,
        knn_weight_graph(np.asarray(adata.obsm[cfg.coord_key], dtype=np.float64), cfg),
        eps=float(cfg.metric_eps),
    ).numpy()
    i_real = morans_i(
        real_x,
        knn_weight_graph(np.asarray(hidden.coords, dtype=np.float64), cfg),
        eps=float(cfg.metric_eps),
    ).numpy()
    labels = gene_modules(model, cfg, seed=seed)
    rows: list[dict[str, float | int | str]] = []
    for label in sorted(set(labels.tolist())):
        columns = labels == label
        if int(columns.sum()) < int(cfg.sefl_module_min_genes):
            continue
        rows.append(
            {
                "module": int(label),
                "n_genes": int(columns.sum()),
                "i_gen": float(np.nanmean(i_gen[columns])),
                "i_real": float(np.nanmean(i_real[columns])),
                "abs_diff": float(np.nanmean(np.abs(i_gen[columns] - i_real[columns]))),
            }
        )
    return rows


def _table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    """Render a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def calibration_chunks(
    calibration: Any | None,
    window: Any | None,
    modules: Sequence[dict[str, float | int | str]] | None,
) -> list[str]:
    """Render the calibration, retrieval-window and per-module sections as markdown lines.

    Shared by :func:`write_selection_report` and by the calibration-only report arm, so a
    calibration run that skips selection emits the same table rather than a second rendering
    of the same numbers.

    Returns a list of markdown lines (possibly empty when every argument is ``None``).
    """
    chunks: list[str] = []
    if calibration is not None:
        chunks += [
            "",
            "## Calibration (leakage-free, flanking training sections only)",
            "",
            _table(
                [
                    ["`ell_xy`", f"{calibration.ell[0]:.1f} um", str(calibration.status)],
                    [
                        "`ell_z`",
                        f"{calibration.ell[2]:.1f} um",
                        str(calibration.ell_z_status)
                        + (" (bound, not a fit — R1)" if calibration.ell_z_is_upper_bound else ""),
                    ],
                    [
                        "fitted `ell`",
                        f"{calibration.ell_fitted[0]:.1f} / {calibration.ell_fitted[2]:.1f} um",
                        "variogram",
                    ],
                    [
                        "Moran's I",
                        f"gen {calibration.i_gen:.4f} vs flanking {calibration.i_target:.4f}",
                        f"{calibration.iterations} iterations",
                    ],
                    [
                        "between-section r",
                        f"gen {calibration.z_achieved:.4f} vs observed {calibration.z_target:.4f}",
                        "R1 remedy 2",
                    ],
                ],
                ["quantity", "value", "status"],
            ),
        ]
    if window is not None:
        chunks += [
            "",
            f"Derived `retrieval_z_window` = **{window.window:g}** spacings "
            f"(largest section gap {window.max_gap_um:g} um). `Config.retrieval_z_window` "
            "remains the fallback and the ablation handle.",
        ]
    if modules:
        chunks += [
            "",
            "## Per-module Moran's I agreement — diagnostic only (SPEC_QUESTIONS A2)",
            "",
            "One **global** `ell` is calibrated; this table says whether it serves every gene "
            "module equally. It is not a target, and a poor table is evidence for the "
            "per-channel-group escalation, which is a design change to be decided explicitly.",
            "",
            _table(
                [
                    [
                        str(row["module"]),
                        str(row["n_genes"]),
                        f"{float(row['i_gen']):.4f}",
                        f"{float(row['i_real']):.4f}",
                        f"{float(row['abs_diff']):.4f}",
                    ]
                    for row in modules
                ],
                ["module", "genes", "I_gen", "I_real", "|diff|"],
            ),
        ]
    return chunks


def write_selection_report(
    result: SelectionResult,
    path: str | Path,
    *,
    calibration: Any | None = None,
    window: Any | None = None,
    modules: Sequence[dict[str, float | int | str]] | None = None,
) -> Path:
    """Write ``reports/config_selection_{dataset}.md``. Returns the path.

    Contents, in the order ``specs/09`` asks for them: the selected config, **all four cells**
    of the joint gate, the full coordinate-descent score table, the calibration results when
    they were run, and the per-module Moran's I diagnostic (A2). The leakage statement — which
    training sections the folds ran on — is at the top, because it is the claim a reader is
    most entitled to check.

    The selected config is **also written beside the report as YAML**, at the same stem, and
    linked from it. ``specs/09`` §3 asks for the config to be persisted and not only reported:
    a table of gates is what a reader needs, a loadable ``Config`` is what the next run needs,
    and a paper number whose config exists only as prose is not reproducible.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = result.config
    config_path = out.with_suffix(".yaml")
    cfg.to_yaml(config_path)
    chunks: list[str] = [
        f"# Config selection — {result.dataset}",
        "",
        f"Selected by internal LOSO over **training sections only**, seed {result.seed}. "
        f"Folds: {', '.join(result.section_ids)}. No held-out section is reachable from "
        "`select_config`: it takes a `TrainingVolume`, which only `split_holdout` produces.",
        "",
        "## Selected configuration",
        "",
        _table(
            [
                *[
                    [
                        f"`{gate}`",
                        "**UNDETERMINED** (see below)"
                        if gate in result.undetermined
                        else str(getattr(cfg, gate)),
                    ]
                    for gate, _ in ALL_GATES
                ],
                ["`train_steps`", str(cfg.train_steps)],
                ["`w_autocorr`", f"{cfg.w_autocorr:g}"],
                ["`w_profile`", f"{cfg.w_profile:g}"],
                ["`w_distribution`", f"{cfg.w_distribution:g}"],
                ["config hash", f"`{cfg.content_hash()}`"],
                ["persisted at", f"`{config_path.name}`"],
            ],
            ["gate", "selected"],
        ),
        "",
        "## The joint gate: `train_steps` x metric-aware weights",
        "",
        "All four cells, each **fitted at the budget it names**. Reported in full because the "
        "gate exists for an interaction, and coordinate descent over these two would pick "
        "`weights off` from a 1x incumbent and never reach the cell that wins (`specs/09` §3).",
        "",
        _table(
            [
                [
                    c.label,
                    str(c.steps),
                    *[f"{c.scores.get(name, float('nan')):.4f}" for name in METRIC_NAMES],
                    f"{c.rank:.1f}",
                ]
                for c in result.joint
            ],
            ["cell", "steps", *METRIC_NAMES, "median rank"],
        ),
        "",
        "## The merged full-budget gate: "
        + (
            " x ".join(f"`{g}`" for g, _ in full_budget_gates(result.pinned))
            or "(every gate pinned)"
        ),
        "",
        f"All {len(result.full_budget)} cells, every one fitted at the **selected** budget. "
        "These gates are disqualified from reduced-budget scoring by `specs/09` §3's "
        "training-free-option rule — each has an option that reaches its final behaviour "
        "without training (`resample`, `iid`, `cross-mix`) and is therefore at full strength "
        "at any budget while its rivals are not. They are scored jointly rather than one "
        "after another because their errors compound through coordinate descent's ordering "
        "(open risk R8, `reports/r8_budget_grid.md`).",
        "",
        _table(
            [
                [
                    c.label,
                    str(c.steps),
                    *[f"{c.scores.get(name, float('nan')):.4f}" for name in METRIC_NAMES],
                    f"{c.rank:.1f}",
                ]
                for c in sorted(result.full_budget, key=lambda c: c.rank)
            ],
            ["cell", "steps", *METRIC_NAMES, "median rank"],
        ),
        "",
        "## Coordinate descent (the gates every option of which trains)",
        "",
        (
            "**Condition (2) of the training-free-option rule fired**: the incumbent scored "
            f"more than `selection_convergence_tol` worse at the reduced budget on "
            f"{len(result.escalating_metrics)} metric(s) — "
            f"{', '.join(f'`{m}`' for m in result.escalating_metrics)} — so the reduced budget "
            "is not a usable proxy for any remaining gate and these were scored at the "
            "**selected** budget too."
            if result.reduced_budget_escalated
            else (
                "Condition (2) of the rule did not fire: the incumbent scores within "
                "`selection_convergence_tol` at the reduced budget, so it is a usable proxy "
                "and these gates keep it."
            )
        ),
        "",
        _table(
            [
                [
                    c.gate,
                    c.label,
                    str(c.steps),
                    *[f"{c.scores.get(name, float('nan')):.4f}" for name in METRIC_NAMES],
                    f"{c.rank:.1f}",
                ]
                for c in result.candidates
                if c.gate not in ("joint", "full_budget")
            ],
            ["gate", "candidate", "steps", *METRIC_NAMES, "median rank"],
        ),
        "",
        (
            "**Candidates that could not be scored and were ranked last** "
            f"({len(result.failures)}). The emission guard refusing an under-trained "
            "candidate's counts is a fact about that candidate, not about the run — but it "
            "is printed here because a rank built partly on failures is not the same "
            "evidence as one built on scores:\n\n"
            + "\n".join(
                f"* `{label}` @ {steps} steps — {reason}"
                for label, steps, reason in result.failures
            )
            + "\n"
            if result.failures
            else "Every candidate scored; none was ranked last for failing to emit."
        ),
        "",
        f"Fits issued: {len(result.fits)} "
        f"({', '.join(f'{steps} steps' for _, steps in result.fits)}).",
        "",
        "The six metrics are computed with T08's kernels rather than `bench3`'s vendored "
        "implementations — `eval/metrics.py` is T10's module. The names match, and T10 "
        "re-scores the selected config with the vendored code.",
    ]
    if result.undetermined:
        chunks += [
            "",
            "## ⛔ Gates **UNDETERMINED** for this dataset",
            "",
            "These gates were inert under the configuration that ships — changing them cannot "
            "change a single emitted count — so they were measured under a different one. "
            "**The selected config does not carry their winners**: `selected.yaml` records "
            "them as undetermined rather than claiming a decision the shipped model cannot "
            "express (SPEC_QUESTIONS C34). What won *elsewhere* is below, and it is evidence "
            "about that configuration, not about this dataset's shipped one.",
            "",
            _table(
                [
                    [f"`{gate}`", f"`{result.elsewhere_winner.get(gate, '—')}`", why]
                    for gate, why in sorted(result.undetermined.items())
                ],
                ["gate", "won elsewhere", "why it is undetermined here"],
            ),
        ]
    if result.inert_notes:
        chunks += [
            "",
            "## ⚠️ Gates that were **inert** under the incumbent",
            "",
            "A gate is inert under a configuration when changing it cannot change a single "
            "emitted count — so scoring it there measures nothing, and reports a **0.0000** "
            "separation that reads like a perfect tie. `specs/09` §3 forbids it; the relation "
            "is derived by running the generation path (`inert_gates`), not declared, and the "
            "gate is re-ordered onto the best-ranked cell that can decide it.",
            "",
            "**Its selected value is evidence from there, not from the shipped cell.**",
            "",
            _table(
                [[f"`{gate}`", why] for gate, why in sorted(result.inert_notes.items())],
                ["gate", "what happened"],
            ),
        ]
    if result.pinned:
        chunks += [
            "",
            "## Gates **not** selected here (pinned)",
            "",
            result.pinned_reason,
            "",
            _table(
                [[f"`{gate}`", f"`{option}`"] for gate, option in sorted(result.pinned.items())],
                ["gate", "pinned to"],
            ),
        ]
    if result.reviews:
        chunks += [
            "",
            "## Per-gate tie-break review (`specs/09` §3's capability rule)",
            "",
            "Each gate re-checked against the candidates that differ from the selected config "
            "in **that gate alone**. `margin` is the largest absolute difference over the six "
            "metrics between the rank winner and its closest rival — the same statistic "
            "`reports/envelope_synthetic.md` measures the run-to-run envelope of, so the "
            f"comparison against `claim_tie_break_envelope` = {cfg.claim_tie_break_envelope:g} "
            "is like for like. **A margin inside the envelope means the ranking is not "
            "evidence**, and the rule decides instead. Applied once, on the completed table, "
            "which is where the evidence is.",
            "",
            "**`folds` is how many LOSO folds each margin is a mean of, and it is printed "
            "because it is small.** `selection_folds` takes the *interior* sections, so a "
            "four-section training stack — which is what tier-1 STARmap's `paper_2_4_6` "
            f"holdout leaves — gives **{result.reviews[0].n_folds if result.reviews else 0}** "
            f"however large `Config.selection_n_folds` ({cfg.selection_n_folds}) is set. "
            "A margin that is a mean of two numbers reads exactly like one that is a mean of "
            "thirty unless the count is beside it.",
            "",
            _table(
                [
                    [
                        f"`{r.gate}`",
                        f"`{r.winner}`",
                        "—" if r.runner_up is None else f"`{r.runner_up}`",
                        "—" if r.margin != r.margin else f"{r.margin:.4f}",
                        "—" if r.pinned else f"**n = {r.n_folds}**",
                        "**inside**" if r.inside_envelope else ("—" if r.pinned else "outside"),
                        "**yes**" if r.flipped else "no",
                        r.reason,
                    ]
                    for r in result.reviews
                ],
                [
                    "gate",
                    "ships",
                    "closest rival",
                    "margin",
                    "folds",
                    "vs envelope",
                    "flipped",
                    "why",
                ],
            ),
        ]
    chunks += calibration_chunks(calibration, window, modules)
    out.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return out
