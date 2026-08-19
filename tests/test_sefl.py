"""T07 acceptance tests for the SEFL consistency losses.

Every numeric threshold here is ``specs/07``'s: ``L_cross`` down by **60 %** over 500 steps,
generated per-gene variance at or above **60 %** of the real cells' after training, cell
counts adding within **Poisson tolerance** across thicknesses, and SEFL costing under **60 %**
extra wall-clock per epoch at the configured sampling rates.

Three of them are written to *fail*, and each failure is a result:

* ``test_no_collapse_negative_control_fails`` trains the arm with
  ``Config.sefl_ema_teacher = False`` and asserts the no-collapse criterion **does not hold**.
  That is what documents the stop-gradient teacher as load-bearing rather than decorative.
* ``test_no_collapse_at_the_spec_w_cross`` is a **strict xfail**: at ``specs/07``'s own
  ``w_cross = 0.3`` the anatomical field flattens and the criterion fails at 0.065. The
  measurement, its attribution and the proposed fix are in ``Config.w_cross``, PROGRESS.md
  (open risk R6) and SPEC_QUESTIONS C19; ``test_no_collapse`` above it runs the shipped
  default, where the same criterion passes.
* ``test_prog_conditioning`` optimises the *unconditional* variant of ``L_prog`` and asserts
  it homogenises the tissue relative to the conditional variant from the same weights.

Cost. Anything that runs an optimiser loop is ``slow`` (SPEC_QUESTIONS C6); the three trained
arms are module-scoped fixtures.
"""

from __future__ import annotations

import copy
import itertools
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from scipy.spatial import cKDTree
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import Section, TrainingVolume, Volume
from spatialcpav25_gen.infer.planes import (
    intersect,
    plane_from_normal,
    plane_pose,
    random_plane_pair,
    section_plane,
)
from spatialcpav25_gen.losses.sefl import (
    EQUIVARIANT_QUANTITIES,
    INVARIANT_QUANTITIES,
    SEFL_LOSSES,
    SEFL_TERMS,
    CollapseWarning,
    ConsistencyDominanceWarning,
    EMATeacher,
    check_collapse,
    consistency_ratio,
    cross_terms,
    evaluate_branch,
    gene_modules,
    labels_at,
    loss_cross,
    loss_prog,
    loss_prog_WRONG,
    loss_thick,
    mmd2_rbf,
    morans_i_torch,
    sefl_active,
    sefl_ramp,
    sinkhorn_divergence,
    thick_terms,
)
from spatialcpav25_gen.model.embeddings import EntityEmbeddings
from spatialcpav25_gen.model.field import BBoxClampWarning, RotationContext
from spatialcpav25_gen.model.spatialcpav25_gen import (
    CTFFlow,
    TrainingData,
    train_ctfflow,
)

from tests.fixtures.synthetic import make_synthetic_volume
from tests.fixtures.text import fake_text_vecs

SEED = 20260817

# --------------------------------------------------------------------------------------
# the spec's numbers
# --------------------------------------------------------------------------------------

CROSS_DECREASE_MIN = 0.60
"""``specs/07``: 500 training steps must reduce ``L_cross`` by at least 60 %."""

VARIANCE_FLOOR = 0.60
"""...and per-gene variance of generated expression must stay at or above 60 % of the real
sections'. The collapse *alarm* fires at 25 % (``Config.sefl_collapse_warn_fraction``); this
is the stricter acceptance criterion, so a run that passes here never approached the alarm."""

VARIANCE_CEILING = 1.0 / VARIANCE_FLOOR
"""...and no further above it than the floor is below: the band is a **factor of 1.67 either
way**, not a one-sided floor.

``specs/07`` states only the floor, because the failure it was written for is collapse. A
one-sided criterion is satisfied at the opposite extreme from the failure it was written for —
the same trap GATE 2's attention entropy fell into, where 0.987 log K passed a
"not-collapsed" test by being *maximally* uninformative. Overdispersion is a real failure with
a real name: a generated section whose genes vary more than the tissue's is not more
informative, it is noisier, and every count statistic in the benchmark reads it as signal.
Measured at the shipped weights: **1.127-1.331** across runs, inside the band but on the high
side, and 0.711 with SEFL off. Whether the overshoot costs anything on the six target metrics
is T10's A7 to answer; what T07 measured on the statistics it owns is in PROGRESS.md."""

COST_OVERHEAD_MAX = 0.60
"""...and SEFL must add under 60 % wall-clock per epoch at the configured sampling rates."""

POISSON_Z_MAX = 3.0
"""``N`` at thickness ``3h`` is "within Poisson tolerance" of ``3 x N`` at ``h``: three
standard deviations of a Poisson count, which is the tolerance that phrase names."""

PROG_HOMOGENISATION_MAX_RATIO = 0.80
"""After the same optimisation from the same weights, the **unconditional** variant of
``L_prog`` should leave the tissue with at most this fraction of the between-region expression
structure the conditional variant leaves. ``specs/07`` §4's claim stated as the two-arm
comparison it actually is: "forcing marginals to match would force the model to hallucinate a
homogeneous tissue". A one-arm "the spread shrinks" criterion is not measurable — the spread
also moves because the model is being optimised at all, which is what the second arm controls
for. **Measured 1.27**, i.e. the wrong side of 1; the test is a strict xfail and B22 records
why."""

TRAIN_STEPS = 500
"""``specs/07``'s own budget for ``test_cross_loss_decreases``, reused by every trained arm."""


TEST_MODEL_CFG: dict[str, object] = {
    # The reduced widths of T06's slow suite, so a trained arm fits a CPU budget. Nothing
    # about the architecture, the losses or the output path changes.
    "triplane_res_xy": 48,
    "triplane_res_z": 8,
    "triplane_channels": 8,
    "n_plane_orientations": 1,
    "field_dim": 32,
    "field_mlp_hidden": 64,
    "latent_dim": 24,
    "n_rff": 256,
    "gene_emb_dim": 48,
    "ctx_emb_dim": 16,
    "retrieval_ctx_dim": 16,
    "retrieval_k": 16,
    "retrieval_candidates_per_section": 32,
    "flow_hidden": 160,
    "decoder_hidden": 96,
    "latent_encoder_hidden": 96,
    "expr_pca_dim": 16,
    "batch_cells": 512,
    "genes_per_step": 200,
    "layout_n_mc": 512,
    "ode_steps": 8,
    "lr": 3e-3,
    "rotation_bias": "axial",
    # SEFL's own sampling rates, reduced in proportion to `batch_cells` (512 rather than the
    # default 2048), so the ratio of SEFL work to reconstruction work is the ratio a default
    # run has. `sefl_every_n_steps` and the angle range are the spec's and are untouched.
    "sefl_patch_cells": 500,
    "sefl_n_line_points": 128,
    "sefl_max_distribution_points": 256,
}


def sefl_cfg(**overrides: object) -> Config:
    """The reduced config the SEFL tests use."""
    return Config().replace(**{**TEST_MODEL_CFG, **overrides})


def build_embeddings(cfg: Config, vol: Volume) -> EntityEmbeddings:
    """T02's embeddings over deterministic stand-in text vectors (no network, Convention 7)."""
    return EntityEmbeddings(
        cfg,
        torch.from_numpy(fake_text_vecs(vol.n_genes, cfg.text_dim_in, 1)),
        torch.from_numpy(fake_text_vecs(len(vol.celltype_names), cfg.text_dim_in, 2)),
        None
        if vol.region_names is None
        else torch.from_numpy(fake_text_vecs(len(vol.region_names), cfg.text_dim_in, 3)),
    )


@dataclass(frozen=True)
class Built:
    """A model, the volume it was built on, and the pieces the tests need beside it."""

    model: CTFFlow
    cfg: Config
    volume: Volume
    training: TrainingVolume
    held_out: Section
    history: object = None

    @property
    def bbox(self) -> np.ndarray:
        """The training volume's bounding box — every loss's sampling support."""
        return np.asarray(self.training.bbox, dtype=np.float64)


def build_model(cfg: Config, *, grf_seed: int = 11) -> Built:
    """Build (but do not train) one model on the synthetic fixture."""
    vol, _ = make_synthetic_volume(seed=0)
    training, held = split_holdout(vol, "alternating", 0, cfg)
    data = TrainingData.build(training, cfg)
    model = CTFFlow(cfg, data, build_embeddings(cfg, vol), grf_seed=grf_seed)
    return Built(model=model, cfg=cfg, volume=vol, training=training, held_out=held[len(held) // 2])


def train(built: Built, *, steps: int = TRAIN_STEPS, seed: int = SEED) -> Built:
    """Train a built model in place and return it with its history."""
    with warnings.catch_warnings():
        # T04's field clamps query points outside the bounding box and says so; rotation
        # augmentation and oblique SEFL planes produce them by construction.
        warnings.simplefilter("ignore", BBoxClampWarning)
        history = train_ctfflow(built.model, built.cfg, steps=steps, seed=seed)
    return Built(
        model=built.model,
        cfg=built.cfg,
        volume=built.volume,
        training=built.training,
        held_out=built.held_out,
        history=history,
    )


def warm_field(built: Built, *, seed: int = 0, scale: float = 0.5) -> Built:
    """Give the triplane feature planes a non-trivial seeded draw.

    At initialisation the planes have a standard deviation of 1e-2 (``_PLANE_INIT_SD``), so
    the field is nearly constant, every pose returns nearly the same features, and *every*
    consistency loss is ~1e-9 — a true statement about an untrained model and a useless
    fixture for testing geometry. This is the cheap stand-in for "a model that has learned
    something", and it is seeded, so the structural tests stay deterministic.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for planes in (
            built.model.field.plane_xy,
            built.model.field.plane_xz,
            built.model.field.plane_yz,
        ):
            planes.normal_(0.0, scale, generator=generator)
    return built


@pytest.fixture(scope="module")
def built() -> Built:
    """A model with a non-trivial field but no training. Enough for structural properties."""
    return warm_field(build_model(sefl_cfg()))


SPEC_W_CROSS = 0.3
"""``specs/07``'s own ``w_cross``. The shipped default is **0** — see ``Config.w_cross`` for
the four-arm measurement behind that — so the two arms that exercise ``L_cross`` set it back
explicitly. Everything else about them is the spec's schedule: warm-up 0.2, ramp 0.2, every
third step."""

SEFL_ON: dict[str, object] = {"w_thick": 0.2, "w_prog": 0.2}
"""``specs/07``'s weights for the two terms that are *not* harmful, which also ship at **0**:
SEFL is opt-in until T10's A7 measures it (``Config.w_prog`` carries the three T06 criteria it
costs). Every test here that is about a SEFL loss therefore has to turn it on explicitly, which
is the right shape — a test of a loss should name the loss."""


@pytest.fixture(scope="module")
def trained() -> Built:
    """SEFL as ``specs/07`` weights it, minus ``L_cross``: 500 steps, ``thick`` + ``prog``."""
    return train(build_model(sefl_cfg(**SEFL_ON)))


@pytest.fixture(scope="module")
def trained_cross() -> Built:
    """The arm ``specs/07`` describes in full: the same run with all three losses live."""
    return train(build_model(sefl_cfg(**SEFL_ON, w_cross=SPEC_W_CROSS)))


@pytest.fixture(scope="module")
def trained_cross_no_teacher() -> Built:
    """The negative control: all three losses live, stop-gradient teacher switched off."""
    return train(build_model(sefl_cfg(**SEFL_ON, w_cross=SPEC_W_CROSS, sefl_ema_teacher=False)))


def cross_at(built: Built, *, seed: int) -> float:
    """``L_cross`` of a model against *itself* — a teacher freshly copied from the student.

    The measurement is then purely "does this model's conditioning agree across two poses",
    which is comparable between checkpoints; a teacher that lags the student by an EMA would
    mix that lag into the number.
    """
    teacher = EMATeacher(built.model, built.cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            value = loss_cross(
                built.model, teacher, built.bbox, built.cfg, np.random.default_rng(seed)
            )
    return float(value)


def generated_variance_ratio(built: Built, *, seed: int) -> float:
    """Mean per-gene variance of a generated section over the real section's own.

    Both sides are **drawn counts** on the same genes: ``Var[x] = Var[mu] + E[NB noise]``, so
    comparing a variance of ``mu`` with a variance of counts would be a different quantity and
    would look like a collapse on a healthy model.
    """
    # `repulsion=False` is a *generation-time* choice (T05 forbids a hand-set `r0`, and no
    # `RepulsionParams` were fitted for these arms): it changes where the cells are, which is
    # the layout's business, and not what any of them expresses, which is what is measured
    # here. `CTFFlow.check_generation_cfg` allows it for exactly that reason.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        adata = built.model.generate(
            section_plane(built.held_out), built.cfg.replace(repulsion=False), seed
        )
    generated = np.asarray(adata.X.todense(), dtype=np.float64)
    real = np.asarray(built.held_out.counts.todense(), dtype=np.float64)
    return float(generated.var(axis=0).mean() / real.var(axis=0).mean())


def morans_i_numpy(values: np.ndarray, coords: np.ndarray, k: int) -> np.ndarray:
    """Reference in-plane Moran's I per column, on a row-normalised kNN graph."""
    idx = cKDTree(coords).query(coords, k=k + 1)[1][:, 1:]
    centred = values - values.mean(axis=0, keepdims=True)
    lagged = centred[idx].mean(axis=1)
    denominator = (centred**2).sum(axis=0)
    out = np.zeros(values.shape[1])
    ok = denominator > 0
    out[ok] = (centred * lagged).sum(axis=0)[ok] / denominator[ok]
    return out


# --------------------------------------------------------------------------------------
# the invariant / equivariant table (specs/07's opening requirement)
# --------------------------------------------------------------------------------------


def test_invariant_and_equivariant_tables_are_disjoint_and_populated():
    """The table is a code constant, so a loss cannot be added on the wrong side by accident."""
    assert set(INVARIANT_QUANTITIES).isdisjoint(EQUIVARIANT_QUANTITIES)
    assert "cells_per_unit_volume" in INVARIANT_QUANTITIES
    assert "cells_per_unit_area" in EQUIVARIANT_QUANTITIES
    assert "in_plane_morans_i" in EQUIVARIANT_QUANTITIES
    assert "expression_given_type_and_region" in INVARIANT_QUANTITIES
    assert len(INVARIANT_QUANTITIES) == len(set(INVARIANT_QUANTITIES))
    assert len(EQUIVARIANT_QUANTITIES) == len(set(EQUIVARIANT_QUANTITIES))


def test_wrong_loss_is_not_in_the_registry():
    """``loss_prog_WRONG`` is a deliberate negative control and must stay out of training."""
    assert set(SEFL_LOSSES) == set(SEFL_TERMS) == {"cross", "thick", "prog"}
    assert loss_prog_WRONG not in SEFL_LOSSES.values()
    assert not any("wrong" in name.lower() for name in SEFL_LOSSES)
    # ...and it is still importable and callable, because T10's ablation A8 trains it.
    assert callable(loss_prog_WRONG)


# --------------------------------------------------------------------------------------
# schedule, ratio, alarm
# --------------------------------------------------------------------------------------


def test_sefl_ramp_warms_up_then_ramps_then_holds():
    cfg = Config()  # warmup 0.2, ramp 0.2
    assert sefl_ramp(0, 1000, cfg) == 0.0
    assert sefl_ramp(199, 1000, cfg) == 0.0
    assert sefl_ramp(300, 1000, cfg) == pytest.approx(0.5)
    assert sefl_ramp(400, 1000, cfg) == pytest.approx(1.0)
    assert sefl_ramp(999, 1000, cfg) == pytest.approx(1.0)
    values = [sefl_ramp(s, 1000, cfg) for s in range(1000)]
    assert all(b >= a for a, b in itertools.pairwise(values))
    # A zero-length ramp is a step, not a division by zero.
    assert sefl_ramp(200, 1000, cfg.replace(sefl_ramp_frac=0.0)) == pytest.approx(1.0)


def test_sefl_active_respects_every_n_steps():
    cfg = Config()  # every 3rd
    assert [s for s in range(9) if sefl_active(s, cfg)] == [0, 3, 6]
    assert all(sefl_active(s, cfg.replace(sefl_every_n_steps=1)) for s in range(5))


def test_consistency_ratio_warns_when_consistency_stops_being_a_regulariser():
    weights = {"recon": 1.0, "cfm": 1.0, "cross": 0.3, "thick": 0.2, "prog": 0.2}
    healthy = {
        "recon": torch.tensor(2.0),
        "cfm": torch.tensor(1.0),
        "cross": torch.tensor(0.1),
        "diag_gene_variance": torch.tensor(9.9),
    }
    assert consistency_ratio(healthy, weights) == pytest.approx(0.03 / 3.0)
    dominated = {"recon": torch.tensor(0.1), "cross": torch.tensor(10.0)}
    assert consistency_ratio(dominated, weights) > Config().sefl_consistency_ratio_warn
    assert consistency_ratio({"cross": torch.tensor(1.0)}, weights) == float("inf")


def test_collapse_alarm_is_silent_on_a_run_shorter_than_its_floor(built: Built):
    """A short run must not trip the alarm, however degenerate its untrained variance is.

    The regression this pins: with only the ramp gate, a four-step run opens the gate at step
    3 — a fraction of a very short run — and the alarm fires at 0.008 of the real variance on
    a decoder that has taken three steps. That is initialisation, not collapse, and it fired
    inside the *fast* suite, where it trains reviewers to ignore it. ``sefl_collapse_min_steps``
    is the second gate.
    """
    cfg = built.cfg.replace(**SEFL_ON, sefl_warmup_frac=0.0, sefl_ramp_frac=0.0)
    model = build_model(cfg).model
    with warnings.catch_warnings():
        warnings.simplefilter("error", CollapseWarning)
        warnings.simplefilter("ignore", BBoxClampWarning)
        warnings.simplefilter("ignore", ConsistencyDominanceWarning)
        history = train_ctfflow(model, cfg, steps=4, seed=SEED)
    assert history.collapse_alarms == []
    # ...and the variance it would have complained about really is degenerate, so the silence
    # is the gate working rather than the model being fine.
    assert history.variance_ratio[-1] < Config().sefl_collapse_warn_fraction


def test_collapse_alarm_fires_below_the_configured_fraction():
    cfg = Config()  # 0.25
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_collapse(0.30, 1.0, 7, cfg) is False
    with pytest.warns(CollapseWarning, match="COLLAPSE WARNING.*step 7"):
        assert check_collapse(0.24, 1.0, 7, cfg) is True
    with pytest.warns(CollapseWarning):
        check_collapse(0.0, 1.0, 0, cfg)


# --------------------------------------------------------------------------------------
# the noise is the same along the intersection (T03's G1.2, end to end)
# --------------------------------------------------------------------------------------


def test_noise_identical_along_intersection(built: Built):
    """Both plane pathways receive bitwise identical GRF values along the shared segment.

    This is the property that makes ``L_cross`` a correction of the *conditioning* rather
    than a reconciliation of two independent stochastic processes, and ``specs/07`` §2 says
    to assert it.
    """
    p1, p2 = random_plane_pair(built.bbox, 20.0, 160.0, 3, thickness=25.0)
    segment = intersect(p1, p2)
    assert segment is not None
    points = segment.points(built.cfg.sefl_n_line_points)
    centre = built.bbox.mean(axis=0)

    draws = []
    for plane in (p1, p2):
        rotation = RotationContext(
            built.cfg, plane_pose(plane), centre, fields=(built.model.field,)
        )
        with rotation:
            rotation.planes(plane.origin[None, :], plane.normal[None, :])
            xyz_model = rotation.coords(points)
            rotation.retrieval(xyz_model)
            with torch.no_grad():
                draws.append(built.model.prior_latent(rotation.grf(xyz_model), seed=0))
    assert torch.equal(draws[0], draws[1])
    # ...and it is the *field*, not a constant: the values vary along the line.
    assert float(draws[0].var()) > 0.0


def test_generation_is_intersection_consistent_by_construction(built: Built):
    """Two crossing planes generate **identical** expression along their intersection.

    Not "after training" — *bitwise*, on an untrained model, with no consistency loss applied.
    This is the finding that reframes ``L_cross`` (see PROGRESS.md, T07), so it is asserted
    rather than argued: ``CTFFlow.generate``'s expression pathway conditions at *physical*
    points in the data frame — retrieval, the GRF and the Fourier encoding are all data-frame
    channels — and takes the cutting plane only through the **layout**, i.e. through where the
    cells are. Two sections that cross therefore agree exactly where they cross, by
    construction, which is the property ``L_cross`` exists to enforce.

    What is left plane-dependent is the augmentation *pose*, and that is what
    ``evaluate_branch`` compares. ``test_no_collapse`` records what minimising it costs.
    """
    p1, p2 = random_plane_pair(built.bbox, 20.0, 160.0, 9, thickness=25.0)
    segment = intersect(p1, p2)
    assert segment is not None
    points = segment.points(64)
    genes = np.arange(built.model.stats.n_genes, dtype=np.intp)
    labels = labels_at(built.model, points, seed=0)
    neighbours = built.model.data.index.query(points, set(), seed=0)[0]

    draws = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        for plane in (p1, p2):
            # The generation pathway's pose is the identity for every plane, which is the
            # claim: `_flow_expression` conditions with `(points, points)`.
            with torch.no_grad():
                draws.append(
                    evaluate_branch(
                        built.model,
                        plane_from_normal([0.0, 0.0, 1.0], plane.origin, (1.0, 1.0), 25.0),
                        points,
                        genes,
                        built.cfg,
                        seed=0,
                        labels=labels,
                        neighbours=neighbours,
                    )
                )
    assert torch.equal(draws[0].log_mu, draws[1].log_mu)
    assert torch.equal(draws[0].h, draws[1].h)
    # ...and the two *poses* do disagree, which is what `L_cross` measures instead.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            posed = [
                evaluate_branch(
                    built.model,
                    plane,
                    points,
                    genes,
                    built.cfg,
                    seed=0,
                    labels=labels,
                    neighbours=neighbours,
                )
                for plane in (p1, p2)
            ]
    assert not torch.equal(posed[0].log_mu, posed[1].log_mu)


def test_retrieval_is_shared_between_branches(built: Built):
    """Retrieval is a data-frame channel, so the two poses return the same donors.

    ``evaluate_branch`` relies on this to query once per loss instead of once per branch.
    """
    p1, p2 = random_plane_pair(built.bbox, 20.0, 160.0, 5, thickness=25.0)
    points = p1.sample_points(64, built.bbox, seed=1)
    centre = built.bbox.mean(axis=0)
    results = []
    for plane in (p1, p2):
        rotation = RotationContext(
            built.cfg, plane_pose(plane), centre, fields=(built.model.field,)
        )
        with rotation:
            rotation.planes(plane.origin[None, :], plane.normal[None, :])
            xyz_model = rotation.coords(points)
            rotation.grf(xyz_model)
            idx, weights = built.model.data.index.query(
                rotation.retrieval(xyz_model), set(), seed=0
            )
        results.append((idx, weights))
    assert np.array_equal(results[0][0], results[1][0])
    assert np.allclose(results[0][1], results[1][1])


# --------------------------------------------------------------------------------------
# the teacher is a stop-gradient
# --------------------------------------------------------------------------------------


def test_teacher_is_stop_gradient(built: Built):
    """No gradient reaches the teacher, and the student's does not vanish."""
    teacher = EMATeacher(built.model, built.cfg)
    assert teacher.enabled
    assert teacher.module is not built.model
    assert all(not p.requires_grad for p in teacher.module.parameters())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        value = loss_cross(built.model, teacher, built.bbox, built.cfg, np.random.default_rng(2))
    assert float(value) > 0.0
    value.backward()
    assert all(p.grad is None for p in teacher.module.parameters())
    grads = [p.grad for p in built.model.parameters() if p.grad is not None]
    assert grads, "the student received no gradient at all"
    assert max(float(g.abs().max()) for g in grads) > 0.0
    built.model.zero_grad(set_to_none=True)


def test_teacher_disabled_is_symmetric(built: Built):
    """With the teacher off both branches are the live student — the negative control's arm."""
    disabled = EMATeacher(built.model, built.cfg, enabled=False)
    assert disabled.branch(built.model) is built.model
    enabled = EMATeacher(built.model, built.cfg)
    assert enabled.branch(built.model) is enabled.module


def test_teacher_tracks_the_student_with_the_configured_decay(built: Built):
    """One EMA update moves the teacher ``1 - decay`` of the way to the student."""
    cfg = built.cfg.replace(ema_decay=0.5)
    teacher = EMATeacher(built.model, cfg)
    name, parameter = next(iter(built.model.named_parameters()))
    before = teacher.module.get_parameter(name).detach().clone()
    # Save and restore the exact bits rather than adding and subtracting: `x + 1 - 1` is not
    # `x` in float32, and this fixture is shared with tests whose numbers are chaotic in the
    # last bits (`test_prog_conditioning` optimises from it).
    saved = parameter.detach().clone()
    with torch.no_grad():
        parameter.add_(1.0)
    teacher.update(built.model)
    after = teacher.module.get_parameter(name).detach()
    assert torch.allclose(after, 0.5 * before + 0.5 * parameter.detach(), atol=1e-6)
    with torch.no_grad():
        parameter.copy_(saved)


# --------------------------------------------------------------------------------------
# L_cross
# --------------------------------------------------------------------------------------


def test_cross_terms_are_the_specified_four(built: Built):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        terms = cross_terms(
            built.model,
            EMATeacher(built.model, built.cfg),
            built.bbox,
            built.cfg,
            np.random.default_rng(4),
        )
    assert set(terms) == {"params", "h", "type", "lambda"}
    assert all(value.shape == () for value in terms.values())
    assert all(float(value) >= 0.0 for value in terms.values())


def test_cross_loss_skips_planes_that_do_not_meet(built: Built):
    """A pair that does not intersect inside the windows returns exactly zero."""
    far = plane_from_normal([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], (10.0, 10.0), 25.0)
    parallel = plane_from_normal([0.0, 0.0, 1.0], [0.0, 0.0, 300.0], (10.0, 10.0), 25.0)
    assert intersect(far, parallel) is None
    tiny_bbox = np.array([[-1e4, -1e4, -1e4], [-9e3, -9e3, -9e3]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        value = loss_cross(
            built.model,
            EMATeacher(built.model, built.cfg),
            tiny_bbox,
            built.cfg.replace(sefl_min_segment_nn_multiple=1e9),
            np.random.default_rng(0),
        )
    assert float(value) == 0.0


def test_losses_are_deterministic_given_a_seed(built: Built):
    """Convention 3: same generator seed, same loss, bitwise."""
    teacher = EMATeacher(built.model, built.cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            first = [
                float(fn(built.model, teacher, built.bbox, built.cfg, np.random.default_rng(9)))
                for fn in (loss_cross, loss_thick, loss_prog)
            ]
            again = [
                float(fn(built.model, teacher, built.bbox, built.cfg, np.random.default_rng(9)))
                for fn in (loss_cross, loss_thick, loss_prog)
            ]
            other = float(
                loss_cross(built.model, teacher, built.bbox, built.cfg, np.random.default_rng(10))
            )
    assert first == again
    assert other != first[0]


@pytest.mark.slow
def test_cross_loss_decreases(trained_cross: Built):
    """500 training steps reduce ``L_cross`` by at least 60 % (``specs/07``).

    Measured on the run's own trajectory — the first three logged values of the ``cross``
    term against the last three — rather than against an untrained model. An *untrained*
    ``L_cross`` is not a baseline: the feature planes start at a standard deviation of 1e-2,
    so both poses return nearly the same field and the loss is ~1e-9 by construction. The
    quantity the spec is asking about is the one the optimiser sees, which is this one.
    """
    trajectory = trained_cross.history.terms["cross"]  # type: ignore[attr-defined]
    assert len(trajectory) >= 6, "too few logged SEFL steps to read a trajectory"
    before = float(np.median(trajectory[:3]))
    after = float(np.median(trajectory[-3:]))
    decrease = 1.0 - after / before
    print(f"\nL_cross {before:.4f} -> {after:.4f} ({decrease:.1%} down over {TRAIN_STEPS} steps)")
    assert decrease >= CROSS_DECREASE_MIN, f"L_cross {before:.4f} -> {after:.4f} ({decrease:.1%})"


# --------------------------------------------------------------------------------------
# L_thick
# --------------------------------------------------------------------------------------


def test_thick_counts_add(built: Built):
    """``N`` at ``3h`` is within Poisson tolerance of ``3 x N`` at ``h`` — and the loss says so.

    Two assertions, and the second is the one ``specs/07`` insists on: the expected counts
    *do* differ by a factor of three, and ``L_thick``'s count term does **not** penalise that
    difference. A naive implementation that forced ``N_thick == N_thin`` would fail the second
    while passing the first.
    """
    model, cfg = built.model, built.cfg
    plane = plane_from_normal([0.2, -0.1, 1.0], built.bbox.mean(axis=0), (300.0, 300.0), 25.0)
    thick = plane_from_normal(
        plane.normal, plane.origin, plane.half_extent, 25.0 * cfg.thickness_ratio
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            n_thin = float(_expected_count(model, plane, cfg, seed=1))
            n_thick = float(_expected_count(model, thick, cfg, seed=1))
    ratio = n_thick / n_thin
    assert ratio == pytest.approx(float(cfg.thickness_ratio), rel=0.05), (
        f"counts do not add: {n_thin:.1f} at h, {n_thick:.1f} at {cfg.thickness_ratio}h"
    )
    # Poisson tolerance on the *sum*: |N_thick - 3 N_thin| < 3 sqrt(3 N_thin).
    expected = cfg.thickness_ratio * n_thin
    assert abs(n_thick - expected) < POISSON_Z_MAX * np.sqrt(expected)

    # ...and the loss does not penalise the difference it is supposed to expect.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            terms = thick_terms(
                model, EMATeacher(model, cfg), built.bbox, cfg, np.random.default_rng(6)
            )
    assert set(terms) == {"count", "count_by_type", "state"}
    assert float(terms["count"]) < POISSON_Z_MAX**2, (
        f"the count term charges {float(terms['count']):.2f} for counts that add correctly"
    )
    # The same statistic applied to the *equal-counts* comparison a naive implementation
    # would make: it is enormous, which is what makes the assertion above meaningful.
    naive = (n_thick - n_thin) ** 2 / n_thin
    assert naive > 100.0 * float(terms["count"])


def _expected_count(model: CTFFlow, plane, cfg: Config, *, seed: int) -> torch.Tensor:
    """``E[N]`` on a plane: the per-type intensity integrated over the **slab volume**."""
    points = plane.sample_points(cfg.layout_n_mc, model.data.vol.bbox, seed)
    xyz = torch.from_numpy(points.astype(np.float32))
    lam = model.intensity(xyz, model.field(xyz))
    # The slab clipped to the bounding box is what the points were drawn from, so the volume
    # the mean is multiplied by is that slab's, not the window's.
    return lam.sum(dim=1).mean() * float(plane.slab_volume)


# --------------------------------------------------------------------------------------
# L_prog
# --------------------------------------------------------------------------------------


def test_prog_strata_are_conditioned_on_type_and_region(built: Built):
    """The conditional and unconditional variants are different functions."""
    teacher = EMATeacher(built.model, built.cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        with torch.no_grad():
            conditional = float(
                loss_prog(
                    built.model,
                    teacher,
                    built.bbox,
                    built.cfg,
                    np.random.default_rng(8),
                    conditional=True,
                )
            )
            unconditional = float(
                loss_prog(
                    built.model,
                    teacher,
                    built.bbox,
                    built.cfg,
                    np.random.default_rng(8),
                    conditional=False,
                )
            )
    assert conditional != unconditional


def test_gene_modules_are_cached_and_cover_the_panel(built: Built):
    modules = gene_modules(built.model, built.cfg, seed=0)
    assert modules.shape == (built.model.stats.n_genes,)
    assert modules.min() >= 0
    assert len(set(modules.tolist())) > 1, "a single module makes the module term vacuous"
    assert gene_modules(built.model, built.cfg, seed=0) is modules


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="not demonstrable on this fixture; see B22 and PROGRESS")
def test_prog_conditioning(trained: Built):
    """**Pinned failure.** ``specs/07`` §4's claim does not reproduce on this fixture.

    The claim: "Conditioning on ``(c, r)`` is essential. Unconditional matching would be wrong
    ... forcing marginals to match would force the model to hallucinate a homogeneous tissue."
    The experiment: from one trained model, optimise 60 steps on nothing but ``L_prog``, once
    per variant, and compare the between-region expression spread the two arms leave behind.

    Measured, the effect is absent: mean ratio **0.97** from this fixture's trained arm
    (per-seed 1.60 / 0.59 / 0.71 — the direction is not even consistent), and **1.27** from a
    model trained without SEFL at all (1.30 / 1.27 / 1.23, consistently the *wrong* way). The
    claim predicts well under 1 either way. The likely reason is the experiment rather than the
    claim — the conditional loss
    carries one MMD, one correlation and one module term *per stratum* and so applies several
    times the gradient of the single-stratum unconditional one at the same learning rate, and
    nothing in the design says the two should be compared at equal step size. The composition
    difference the claim needs is present (the two planes' ``(type, region)`` mixtures differ
    by a median total variation of **0.51**), so that is not the explanation.

    Kept as a strict xfail with the numbers, and the conditioning is kept in the loss: the
    argument for it is a priori sound — two planes genuinely sample different mixtures — and
    what this fixture fails to show is the *harm* of dropping it. B22.
    """
    ratios = [
        _homogenisation(trained, conditional=False, gen_seed=seed)
        / _homogenisation(trained, conditional=True, gen_seed=seed)
        for seed in (0, 1, 2)
    ]
    ratio = float(np.mean(ratios))
    print(f"\nbetween-region spread, unconditional / conditional: {ratio:.3f} {ratios}")
    assert ratio <= PROG_HOMOGENISATION_MAX_RATIO, (
        f"the unconditional variant did not homogenise the tissue relative to the conditional "
        f"one: ratio {ratio:.3f}"
    )


def _homogenisation(
    built: Built, *, conditional: bool, steps: int = 60, gen_seed: int = 0
) -> float:
    """Between-region expression spread after optimising ``L_prog`` alone from ``built``."""
    model = copy.deepcopy(built.model)
    model.data = built.model.data
    teacher = EMATeacher(model, built.cfg)
    optimiser = torch.optim.Adam(model.parameters(), lr=float(built.cfg.lr))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        for step in range(steps):
            loss = loss_prog(
                model,
                teacher,
                built.bbox,
                built.cfg,
                np.random.default_rng(1000 * gen_seed + step),
                conditional=conditional,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            teacher.update(model)
        after = _region_spread(model, built, seed=0)
    return float(after)


def _region_spread(model: CTFFlow, built: Built, *, seed: int) -> float:
    """Mean pairwise distance between per-region mean expression vectors on one plane."""
    plane = section_plane(built.training.sections[len(built.training.sections) // 2])
    points = plane.sample_points(400, built.bbox, seed)
    _, region = labels_at(model, points, seed=seed)
    genes = np.arange(model.stats.n_genes, dtype=np.intp)
    with torch.no_grad():
        out = evaluate_branch(model, plane, points, genes, built.cfg, seed=seed)
    assert region is not None
    means = []
    for code in torch.unique(region):
        rows = region == code
        if int(rows.sum()) >= 10:
            means.append(out.log_mu[rows].mean(dim=0))
    assert len(means) >= 2
    stacked = torch.stack(means)
    distances = torch.cdist(stacked, stacked)
    n = stacked.shape[0]
    return float(distances.sum() / (n * (n - 1)))


# --------------------------------------------------------------------------------------
# the equivariant column is not constrained
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_equivariant_not_constrained(trained: Built):
    """A 90 degree section keeps a different in-plane Moran's I from a 0 degree one.

    If these converge, an invariance was applied to a quantity from
    :data:`EQUIVARIANT_QUANTITIES` and real structure has been flattened. The fixture's
    anisotropy is known — 120 um in plane against 200 um along z — so a section cut *along*
    z must carry the **higher** in-plane autocorrelation, and the assertion is on that
    direction as well as on the size of the gap.
    """
    model, cfg = trained.model, trained.cfg
    centre = trained.bbox.mean(axis=0)
    coronal = plane_from_normal([0.0, 0.0, 1.0], centre, (400.0, 400.0), 25.0)
    sagittal = plane_from_normal([1.0, 0.0, 0.0], centre, (400.0, 180.0), 25.0)
    genes = np.arange(model.stats.n_genes, dtype=np.intp)

    values = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        for name, plane in (("coronal", coronal), ("sagittal", sagittal)):
            points = plane.sample_points(600, trained.bbox, seed=17)
            with torch.no_grad():
                out = evaluate_branch(model, plane, points, genes, cfg, seed=17)
            values[name] = morans_i_numpy(
                out.log_mu.numpy(), plane.to_uv(points), cfg.sefl_morans_k
            )

    median_coronal = float(np.median(values["coronal"]))
    median_sagittal = float(np.median(values["sagittal"]))
    print(f"\nin-plane Moran's I: coronal {median_coronal:.4f}, sagittal {median_sagittal:.4f}")
    assert median_sagittal > median_coronal, (
        f"the two angles' in-plane Moran's I converged: {median_coronal:.4f} vs "
        f"{median_sagittal:.4f}"
    )
    assert median_sagittal - median_coronal > 0.10 * abs(median_coronal)


def test_morans_i_torch_matches_the_reference(built: Built):
    """The differentiable Moran's I the negative control uses agrees with a numpy reference."""
    gen = np.random.default_rng(0)
    coords = gen.uniform(0.0, 500.0, size=(200, 2))
    values = gen.normal(size=(200, 5))
    mine = morans_i_torch(torch.from_numpy(values), coords, built.cfg)
    theirs = morans_i_numpy(values, coords, built.cfg.sefl_morans_k)
    assert np.allclose(mine.numpy(), theirs, atol=1e-6)
    # ...and it is differentiable, which the numpy reference is not.
    tensor = torch.from_numpy(values).requires_grad_(True)
    morans_i_torch(tensor, coords, built.cfg).sum().backward()
    assert tensor.grad is not None
    assert float(tensor.grad.abs().sum()) > 0.0


def test_wrong_loss_actually_constrains_morans_i(built: Built):
    """The negative control does what it says: a non-zero, differentiable Moran's I penalty."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        value = loss_prog_WRONG(
            built.model,
            EMATeacher(built.model, built.cfg),
            built.bbox,
            built.cfg,
            np.random.default_rng(12),
        )
    assert float(value) > 0.0
    value.backward()
    assert any(p.grad is not None for p in built.model.parameters())
    built.model.zero_grad(set_to_none=True)


# --------------------------------------------------------------------------------------
# collapse
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_no_collapse(trained: Built):
    """Generated per-gene variance stays inside a factor of 1.67 of the real section's.

    Measured on the ``thick`` + ``prog`` arm — the losses SEFL still proposes to use — which is
    not the shipped default (that is SEFL off entirely; ``Config.w_prog``).

    Both sides of the band, not just ``specs/07``'s floor — see :data:`VARIANCE_CEILING`.
    The SEFL-on arm sits **above** the real variance (1.127-1.331 across runs) where the
    SEFL-off arm sits below it (0.711), so the one-sided criterion the spec states would pass
    this arm without ever looking at the direction it moved.
    """
    ratio = generated_variance_ratio(trained, seed=5)
    print(f"\nper-gene variance of generated / real, shipped weights: {ratio:.3f}")
    assert ratio >= VARIANCE_FLOOR, f"collapsed: per-gene variance ratio {ratio:.3f}"
    assert ratio <= VARIANCE_CEILING, f"overdispersed: per-gene variance ratio {ratio:.3f}"


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="specs/07's w_cross collapses the field; see R6 / C19")
def test_no_collapse_at_the_spec_w_cross(trained_cross: Built):
    """**Pinned failure.** At ``specs/07``'s ``w_cross = 0.3`` the field flattens.

    A strict xfail, in the shape T06 used for its own unsatisfiable criterion (B16): the
    number is the result, and the test exists so that a change which fixes it fails *loudly*
    instead of passing quietly. Measured **0.065** against the 0.60 floor, with ``L_cross``
    itself falling 90 % over the same run — the loss works, and that is the problem. The
    branches can only differ by the augmentation **pose**, because
    ``test_generation_is_intersection_consistent_by_construction`` shows the generation
    pathway is already exactly intersection-consistent; minimising a pose difference drives
    T04's deliberately pose-dependent triplane towards constant.
    """
    ratio = generated_variance_ratio(trained_cross, seed=5)
    print(f"\nper-gene variance of generated / real, w_cross={SPEC_W_CROSS}: {ratio:.3f}")
    assert ratio >= VARIANCE_FLOOR, f"per-gene variance ratio {ratio:.3f}"


@pytest.mark.slow
def test_no_collapse_negative_control_fails(trained_cross_no_teacher: Built):
    """**This test asserts a failure.** Without the stop-gradient teacher, the criterion breaks.

    The arm ``specs/07`` describes (``w_cross = 0.3``) with
    ``Config.sefl_ema_teacher = False``: both branches are the live student, the consistency
    losses are symmetric, and the trivial minimiser — a constant field — is reachable in one
    step rather than pursued over a run. If this ever starts *passing*, the asymmetry has
    stopped being load-bearing.

    Read it beside ``test_no_collapse_at_the_spec_w_cross``, which is the same arm *with* the
    teacher and fails too, and worse (0.065 against this arm's 0.344). The teacher is doing
    what it is specified to do — the symmetric loss is satisfied instantly and stops
    constraining anything, while the lagging target keeps the pressure on — but at this weight
    the pressure is the damage. The asymmetry is real; the loss it protects is the problem.
    """
    ratio = generated_variance_ratio(trained_cross_no_teacher, seed=5)
    print(f"\nper-gene variance of generated / real, teacher OFF: {ratio:.3f}")
    assert ratio < VARIANCE_FLOOR, (
        f"the symmetric arm did not collapse: per-gene variance ratio {ratio:.3f}"
    )


# --------------------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------------------


@pytest.mark.slow
def test_sefl_cost():
    """SEFL adds under 60 % wall-clock per epoch at the configured sampling rates.

    Both arms are timed twice and the **minimum** is taken. A wall-clock measurement on a
    shared machine is a lower bound plus contention noise; the minimum is the estimator of a
    lower bound, and without it this test's verdict is decided by whatever else the box was
    doing (measured spread across repeats: 20-30 %).
    """
    steps = 15
    repeats = 2
    base_cfg = sefl_cfg()  # the shipped default *is* SEFL off
    sefl_on = sefl_cfg(**SEFL_ON, w_cross=SPEC_W_CROSS, sefl_warmup_frac=0.0, sefl_ramp_frac=0.0)
    assert max(sefl_on.w_cross, sefl_on.w_thick, sefl_on.w_prog) > 0.0
    assert max(base_cfg.w_cross, base_cfg.w_thick, base_cfg.w_prog) == 0.0

    timings: dict[str, float] = {}
    for name, cfg in (("base", base_cfg), ("sefl", sefl_on)):
        built = build_model(cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BBoxClampWarning)
            warnings.simplefilter("ignore", ConsistencyDominanceWarning)
            warnings.simplefilter("ignore", CollapseWarning)
            train_ctfflow(built.model, cfg, steps=2, seed=SEED)  # warm the caches
            runs = []
            for repeat in range(repeats):
                start = time.perf_counter()
                train_ctfflow(built.model, cfg, steps=steps, seed=SEED + repeat)
                runs.append(time.perf_counter() - start)
            timings[name] = min(runs)
    overhead = timings["sefl"] / timings["base"] - 1.0
    print(f"\nSEFL overhead {overhead:.1%} ({timings['base']:.2f}s -> {timings['sefl']:.2f}s)")
    assert overhead < COST_OVERHEAD_MAX, (
        f"SEFL overhead {overhead:.1%} (base {timings['base']:.2f}s, "
        f"with SEFL {timings['sefl']:.2f}s over {steps} steps)"
    )


# --------------------------------------------------------------------------------------
# the statistics the losses are built from
# --------------------------------------------------------------------------------------


def test_mmd_is_zero_for_identical_samples_and_positive_otherwise():
    cfg = Config()
    gen = torch.Generator().manual_seed(0)
    x = torch.randn((128, 8), generator=gen)
    assert float(mmd2_rbf(x, x, cfg)) == pytest.approx(0.0, abs=1e-6)
    y = torch.randn((128, 8), generator=gen) + 5.0
    assert float(mmd2_rbf(x, y, cfg)) > 0.1
    near = torch.randn((128, 8), generator=gen)
    assert float(mmd2_rbf(x, near, cfg)) < float(mmd2_rbf(x, y, cfg))


def test_sinkhorn_divergence_is_zero_for_identical_samples_and_grows_with_displacement():
    cfg = Config()
    gen = torch.Generator().manual_seed(1)
    x = torch.randn((96, 4), generator=gen)
    assert abs(float(sinkhorn_divergence(x, x, 0.5, cfg))) < 1e-3
    near = float(sinkhorn_divergence(x, x + 0.5, 0.5, cfg))
    far = float(sinkhorn_divergence(x, x + 4.0, 0.5, cfg))
    assert 0.0 < near < far


def test_sinkhorn_divergence_is_differentiable():
    cfg = Config()
    gen = torch.Generator().manual_seed(2)
    x = torch.randn((48, 3), generator=gen, requires_grad=True)
    y = torch.randn((48, 3), generator=gen) + 1.0
    sinkhorn_divergence(x, y, 0.5, cfg).backward()
    assert x.grad is not None
    assert float(x.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------------------
# open risk R1: does L_cross see the along-z length scale?
# --------------------------------------------------------------------------------------


def test_cross_loss_cannot_see_ell_z(built: Built):
    """R1: ``L_cross`` is blind to the along-z length scale, and T07 owes that decision.

    The fitted ``ell_z`` reads **561 um** against the fixture's **200 um** ground truth, because
    a 9-section 50 um stack spans 400 um and cannot resolve it (PROGRESS.md, open risk R1).
    SEFL's claim to oblique correctness rests on the anisotropy being right, and the decision
    on which remedy to adopt was explicitly deferred to "T07's ``L_cross`` evidence". This is
    that evidence, and it is negative in two forms.

    **Structural, and exact.** T03's noise field is continuous in 3-D, so *both branches
    receive the identical realisation along the intersection whatever ``ell`` is* — asserted
    here bitwise at four values of ``ell_z`` spanning a factor of ten. ``L_cross`` compares
    conditioning pathways given that shared prior; the prior's length scale cancels out of the
    comparison by construction. The same argument applies to ``L_thick`` and ``L_prog``.

    **Measured.** Across the same four values ``L_cross`` itself varies by under 25 % relative.

    So no SEFL term can calibrate ``ell_z``: the hypothesis that T07 would supply the
    instrument is closed, and R1 is decided on that basis (remedy 2 at T09, remedy 3 as the
    guard). If this test ever fails, that decision should be re-opened.
    """
    p1, p2 = random_plane_pair(built.bbox, 20.0, 160.0, 3, thickness=25.0)
    segment = intersect(p1, p2)
    assert segment is not None
    points = segment.points(built.cfg.sefl_n_line_points)
    centre = built.bbox.mean(axis=0)

    values = {}
    for ell_z in (100.0, 200.0, 561.0, 1000.0):
        arm = warm_field(build_model(sefl_cfg(ell_z=ell_z)))
        draws = []
        for plane in (p1, p2):
            rotation = RotationContext(
                arm.cfg, plane_pose(plane), centre, fields=(arm.model.field,)
            )
            with rotation:
                rotation.planes(plane.origin[None, :], plane.normal[None, :])
                xyz_model = rotation.coords(points)
                rotation.retrieval(xyz_model)
                with torch.no_grad():
                    draws.append(arm.model.prior_latent(rotation.grf(xyz_model), seed=0))
        # The structural half: identical noise in both branches, at every length scale.
        assert torch.equal(draws[0], draws[1])
        values[ell_z] = np.mean([cross_at(arm, seed=s) for s in (1, 2, 3)])
    scale = max(abs(float(v)) for v in values.values())
    spread = (max(values.values()) - min(values.values())) / scale if scale > 0 else 0.0
    print(f"\nL_cross vs ell_z: {values} (relative spread {spread:.3f})")
    assert spread < 0.25, (
        "L_cross now discriminates ell_z (relative spread "
        f"{spread:.3f} over 100-1000 um: {values}); re-open the R1 decision, remedy 2 "
        "becomes available as a training-time instrument"
    )
