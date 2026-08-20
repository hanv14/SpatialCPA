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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import torch
from scipy.stats import rankdata
from torch import Tensor

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume
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

if TYPE_CHECKING:  # pragma: no cover
    from spatialcpav25_gen.model.embeddings import EntityEmbeddings
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow

__all__ = [
    "ALL_GATES",
    "FULL_BUDGET_GATES",
    "GATES",
    "METRIC_NAMES",
    "TRAINING_FREE_OPTIONS",
    "V20_CONFIG",
    "Candidate",
    "ScoreCache",
    "SelectionError",
    "SelectionResult",
    "calibration_chunks",
    "full_budget_gate_cells",
    "incumbent_is_unconverged",
    "joint_gate_cells",
    "module_morans_agreement",
    "run_selection",
    "section_scores",
    "select_config",
    "selection_folds",
    "selection_scores",
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


def _check_gate_classification() -> None:
    """Refuse a gate in :data:`ALL_GATES` that :data:`TRAINING_FREE_OPTIONS` does not classify.

    The rule is only a rule if adding a gate forces the question. Raised at import time so a
    gate added without a classification fails immediately rather than being scored at whatever
    budget the code happens to default to.
    """
    missing = [gate for gate, _ in ALL_GATES if gate not in TRAINING_FREE_OPTIONS]
    if missing:
        raise SelectionError(
            f"TRAINING_FREE_OPTIONS does not classify {missing}. specs/09 §3's "
            "training-free-option rule requires every gate to declare which of its options "
            "reach final behaviour without training (an empty tuple means 'all options "
            "train'), because that decides whether the gate may be scored at a reduced budget."
        )
    for gate, options in ALL_GATES:
        unknown = set(TRAINING_FREE_OPTIONS[gate]) - set(options)
        if unknown:
            raise SelectionError(
                f"TRAINING_FREE_OPTIONS[{gate!r}] names {sorted(unknown)}, which are not "
                f"options of that gate ({list(options)})."
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
    return {name: float(np.mean([f[name] for f in per_fold])) for name in METRIC_NAMES}


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
    """

    vol: TrainingVolume
    embeddings: Callable[[Config], EntityEmbeddings]

    def __call__(self, cfg: Config, *, steps: int, seed: int) -> dict[str, float]:
        """Fit for ``steps`` optimiser steps at ``seed`` and return the six metrics."""
        from spatialcpav25_gen.infer.calibrate import calibrate_anchor_weight
        from spatialcpav25_gen.model.layout import fit_repulsion
        from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

        data = TrainingData.build(self.vol, cfg)
        model = CTFFlow(cfg, data, self.embeddings(cfg), grf_seed=seed)
        train_ctfflow(model, cfg, steps=steps, seed=seed)
        if cfg.repulsion:
            model.repulsion = fit_repulsion(self.vol, cfg, seed=seed + 1)
        anchor = (
            calibrate_anchor_weight(model, self.vol, cfg, seed=seed, n_folds=1)
            if cfg.expr_mode == "auto-blend"
            else None
        )
        return selection_scores(model, self.vol, cfg, seed=seed, anchor=anchor)


# --------------------------------------------------------------------------------------
# the search
# --------------------------------------------------------------------------------------


def full_budget_gate_cells(base_cfg: Config) -> list[tuple[str, dict[str, Any]]]:
    """Return every cell of the merged full-budget gate, as ``(label, overrides)``.

    The cartesian product of :data:`FULL_BUDGET_GATES` — on the current table
    ``layout_mode`` x ``prior_mode`` x ``expr_mode``, 3 x 2 x 3 = **18 cells**. Merged rather
    than descended one gate at a time because ``specs/09`` §3's training-free-option rule
    disqualifies all three from reduced-budget scoring *and* because their errors compound
    through the ordering: a wrong choice on an earlier gate biases every gate scored after it.

    ``base_cfg`` is unused except to keep the signature parallel to :func:`joint_gate_cells`;
    the cells are a property of the gate table, not of the config.
    """
    del base_cfg
    cells: list[tuple[str, dict[str, Any]]] = []
    names = [gate for gate, _ in FULL_BUDGET_GATES]
    for combo in product(*[options for _, options in FULL_BUDGET_GATES]):
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

    def __init__(self, path: str | Path, on_write: Callable[[str], None] | None = None) -> None:
        """Load any existing rows from ``path``; create it with a header if it is absent."""
        self.path = Path(path)
        self.on_write = on_write
        self._rows: dict[str, dict[str, float]] = {}
        if self.path.exists():
            with self.path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    self._rows[row["key"]] = {name: float(row[name]) for name in METRIC_NAMES}
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow(["key", "label", "steps", *METRIC_NAMES])

    @staticmethod
    def key(cfg: Config, steps: int) -> str:
        """Return the cache key: the config's content hash and the budget it is fitted at."""
        return f"{cfg.content_hash()}:{int(steps)}"

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
        Optional :class:`ScoreCache`. Step 2 is 18 fits at full budget, so a run that loses
        them to an interrupted session is not usable; with a checkpoint each scored cell is
        flushed to disk as it completes and a re-run skips what is already there.
    """
    require_training_volume(vol, "run_selection")
    if scorer is None and embeddings is None:
        raise SelectionError(
            "run_selection: pass either a scorer or an embeddings factory. The default "
            "scorer fits a model per candidate and cannot build the text-grounded "
            "embeddings itself — they come from T02's cache, which this module has no "
            "business knowing about."
        )
    run_seed = int(base_cfg.seed) if seed is None else int(seed)
    score_fn: Scorer = FitScorer(vol, embeddings) if scorer is None else scorer  # type: ignore[arg-type]

    fits: list[tuple[dict[str, Any], int]] = []
    cache: dict[tuple[tuple[str, Any], ...], dict[str, float]] = {}

    def score(overrides: dict[str, Any], steps: int, label: str = "") -> dict[str, float]:
        key = (*sorted(overrides.items()), ("__steps__", steps))
        if key not in cache:
            cfg = base_cfg.replace(**overrides)
            recorded = None if checkpoint is None else checkpoint.get(cfg, int(steps))
            if recorded is not None:
                cache[key] = recorded  # resumed from a previous run; no fit issued
            else:
                fits.append((dict(overrides), int(steps)))
                cache[key] = score_fn(cfg, steps=int(steps), seed=run_seed)
                if checkpoint is not None:
                    checkpoint.put(cfg, int(steps), label or str(overrides), cache[key])
        return cache[key]

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
            for label, overrides in full_budget_gate_cells(base_cfg)
        ]
    )
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
    if GATES and reduced != selected_steps:
        escalated, escalating_metrics = incumbent_is_unconverged(
            score(incumbent, selected_steps, "incumbent @ selected"),
            score(incumbent, reduced, "incumbent @ reduced"),
            base_cfg,
        )
    descent_steps = selected_steps if escalated else reduced

    # 4. coordinate descent over the gates every option of which trains.
    for _ in range(int(base_cfg.selection_passes)):
        for gate, options in GATES:
            group = []
            for option in options:
                overrides = {**incumbent, gate: option}
                group.append(
                    Candidate(
                        gate=gate,
                        label=f"{gate}={option}",
                        overrides=overrides,
                        steps=descent_steps,
                        scores=score(overrides, descent_steps, f"{gate}={option}"),
                    )
                )
            group = _ranked(group)
            candidates.extend(group)
            incumbent = dict(min(group, key=lambda c: c.rank).overrides)

    result = SelectionResult(
        config=base_cfg.replace(**incumbent),
        joint=joint,
        candidates=candidates,
        fits=fits,
        dataset=dataset,
        seed=run_seed,
        section_ids=tuple(s.section_id for s in selection_folds(vol, base_cfg)),
        full_budget=full_budget,
        reduced_budget_escalated=escalated,
        escalating_metrics=tuple(escalating_metrics),
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
                ["`layout_mode`", str(cfg.layout_mode)],
                ["`prior_mode`", str(cfg.prior_mode)],
                ["`expr_mode`", str(cfg.expr_mode)],
                ["`text_emb_mode`", str(cfg.text_emb_mode)],
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
        "## The merged full-budget gate: `layout_mode` x `prior_mode` x `expr_mode`",
        "",
        "All 18 cells, every one fitted at the **selected** budget. These three gates are "
        "disqualified from reduced-budget scoring by `specs/09` §3's training-free-option "
        "rule — each has an option that reaches its final behaviour without training "
        "(`resample`, `iid`, `cross-mix`) and is therefore at full strength at any budget "
        "while its rivals are not. They are scored jointly rather than one after another "
        "because their errors compound through coordinate descent's ordering (open risk R8, "
        "`reports/r8_budget_grid.md`).",
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
        f"Fits issued: {len(result.fits)} "
        f"({', '.join(f'{steps} steps' for _, steps in result.fits)}).",
        "",
        "The six metrics are computed with T08's kernels rather than `bench3`'s vendored "
        "implementations — `eval/metrics.py` is T10's module. The names match, and T10 "
        "re-scores the selected config with the vendored code.",
    ]
    chunks += calibration_chunks(calibration, window, modules)
    out.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return out
