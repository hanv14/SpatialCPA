"""Fit checkpointing, and the one assertion that makes it worth having.

`reports/durability.md` measured the gap this closes: the campaign driver already resumes at
the unit level and the fit inside a unit did not, so a rebuilt container cost a whole fit —
three times in one day, up to 57 minutes each. At the observed interruption rate that is about
47% of all fit time, and a fit longer than the interval between rebuilds never finishes at all.

The acceptance test is not "it resumes". Convention 3 says two runs with the same seed are
bitwise identical and a test asserts it; a resume that landed *close* to an uninterrupted run
would quietly make that untrue, and every seeded claim in the project would then rest on a
guarantee nothing checks. So `test_resumed_fit_is_bitwise_identical` interrupts a fit between
checkpoints, throws the model away, rebuilds it from nothing, resumes from the last checkpoint
on disk, and asserts **bitwise** equality of every parameter, every buffer, the EMA shadow and
the whole recorded history against a run that was never interrupted.

That assertion covers more than it names. The parameters after the replayed steps depend on
the AdamW moments, the cosine schedule's position and T07's in-place `numpy` generator, so
those are all pinned by it without a separate test each — and the SEFL weights are on here
precisely so the generator is exercised: it is the one piece of loop state that is advanced
rather than derived from `(seed, step)`, and the one a naive checkpoint would omit.

Everything below it is a unit test of the payload itself: what it refuses, and what it does
when it is already complete.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.loaders import split_holdout
from spatialcpav25_gen.data.schema import Volume
from spatialcpav25_gen.model.embeddings import EntityEmbeddings
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.spatialcpav25_gen import (
    CTFFlow,
    TrainHistory,
    TrainingData,
    train_ctfflow,
)
from spatialcpav25_gen.train.checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)

from tests.fixtures.synthetic import make_synthetic_volume
from tests.fixtures.text import fake_text_vecs

SEED = 20260824

# Steps, the checkpoint interval, and where the container "dies". The kill is deliberately
# **not** on a checkpoint boundary: the last good checkpoint is then at step 4 and steps 4 and
# 5 are lost work that the resume has to redo, which is the case a boundary-aligned kill would
# not exercise.
STEPS = 12
CHECKPOINT_EVERY = 4
KILL_AFTER = 6

# The reduced widths of T06's and T07's slow suites, so a fit fits the fast suite's budget.
# Nothing about the architecture, the losses or the output path changes.
TEST_MODEL_CFG: dict[str, object] = {
    "triplane_res_xy": 32,
    "triplane_res_z": 8,
    "triplane_channels": 8,
    "n_plane_orientations": 1,
    "field_dim": 32,
    "field_mlp_hidden": 64,
    "latent_dim": 16,
    "n_rff": 128,
    "gene_emb_dim": 32,
    "ctx_emb_dim": 16,
    "retrieval_ctx_dim": 16,
    "retrieval_k": 8,
    "retrieval_candidates_per_section": 16,
    "flow_hidden": 96,
    "decoder_hidden": 64,
    "latent_encoder_hidden": 64,
    "expr_pca_dim": 8,
    "batch_cells": 128,
    "genes_per_step": 40,
    "layout_n_mc": 256,
    "ode_steps": 4,
    # SEFL on, at reduced sampling rates. Its generator is the reason this file exists.
    "w_cross": 0.1,
    "w_thick": 0.1,
    "sefl_patch_cells": 96,
    "sefl_n_line_points": 32,
    "sefl_max_distribution_points": 64,
    "checkpoint_every_n_steps": CHECKPOINT_EVERY,
}


class KilledError(RuntimeError):
    """Stands in for the container being reclaimed mid-fit."""


def checkpoint_cfg(**overrides: object) -> Config:
    """The reduced config every fit in this file uses."""
    return Config().replace(**{**TEST_MODEL_CFG, **overrides})


def build_model(cfg: Config) -> CTFFlow:
    """Build (but do not train) the fixture model. Deterministic in ``cfg`` alone.

    A resume is only meaningful if a *fresh process* can rebuild the same object, so this is
    what both the uninterrupted run and the resumed run start from — the resumed run builds a
    second one rather than reusing the interrupted model, exactly as a restarted container
    would.
    """
    vol, _ = make_synthetic_volume(seed=0)
    training, _ = split_holdout(vol, "alternating", 0, cfg)
    data = TrainingData.build(training, cfg)
    return CTFFlow(cfg, data, build_embeddings(cfg, vol), grf_seed=11)


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


def fit(model: CTFFlow, cfg: Config, **kwargs: object) -> TrainHistory:
    """Run ``train_ctfflow`` with the fixture's warnings suppressed."""
    with warnings.catch_warnings():
        # T04's field clamps query points outside the bounding box and says so; rotation
        # augmentation and oblique SEFL planes produce them by construction (GATE 2).
        warnings.simplefilter("ignore", BBoxClampWarning)
        return train_ctfflow(model, cfg, steps=STEPS, seed=SEED, **kwargs)  # type: ignore[arg-type]


def kill_after(model: CTFFlow, n_steps: int) -> None:
    """Make ``model``'s fit raise :class:`KilledError` at the start of step ``n_steps``.

    ``train_ctfflow`` calls ``embeddings.set_progress`` once at the top of every step and
    before anything else in it, so counting those calls is counting steps, and raising on the
    ``n_steps``-th one stops the fit with steps ``0 .. n_steps - 1`` complete and no part of
    step ``n_steps`` begun. Wrapping a public method is deliberate: a test that reached into
    the loop to stop it would be testing a private seam rather than the one a killed process
    actually lands on.
    """
    original: Callable[[float], None] = model.embeddings.set_progress
    seen = {"steps": 0}

    def counted(frac: float) -> None:
        if seen["steps"] >= n_steps:
            raise KilledError(f"container reclaimed at step {n_steps}")
        seen["steps"] += 1
        original(frac)

    model.embeddings.set_progress = counted  # type: ignore[method-assign]


def assert_same_model(left: CTFFlow, right: CTFFlow) -> None:
    """Assert every parameter and buffer of two models is bitwise equal."""
    for name, value in left.state_dict().items():
        other = right.state_dict()[name]
        assert torch.equal(value, other), name


def assert_same_history(left: TrainHistory, right: TrainHistory) -> None:
    """Assert two recorded histories are equal entry for entry."""
    assert dataclasses.asdict(left) == dataclasses.asdict(right)


@dataclasses.dataclass(frozen=True)
class Completed:
    """One uninterrupted fit, its model, its history and the checkpoint it left behind."""

    cfg: Config
    model: CTFFlow
    history: TrainHistory
    path: Path


@pytest.fixture(scope="module")
def completed(tmp_path_factory) -> Completed:
    """One uninterrupted, checkpointed fit, shared by every test that only needs a payload.

    Module-scoped on purpose: a fit is seconds, the fast suite has a three-minute budget for
    the whole package, and eight tests each running their own would spend most of it proving
    the same twelve steps. Tests that mutate the file copy it first (:func:`private_copy`).
    """
    cfg = checkpoint_cfg()
    path = tmp_path_factory.mktemp("completed") / "fit.pt"
    model = build_model(cfg)
    history = fit(model, cfg, checkpoint=path)
    return Completed(cfg=cfg, model=model, history=history, path=path)


def private_copy(completed: Completed, tmp_path: Path) -> Path:
    """A writable copy of the shared checkpoint, for a test that changes it."""
    target = tmp_path / "fit.pt"
    target.write_bytes(completed.path.read_bytes())
    return target


# --------------------------------------------------------------------------------------
# the acceptance test
# --------------------------------------------------------------------------------------


def test_resumed_fit_is_bitwise_identical(completed: Completed, tmp_path):
    """A fit killed at step 6 and resumed from step 4 equals an uninterrupted one, bitwise.

    This is the whole point of the module. See the file docstring for why "close" would not do.
    """
    cfg = completed.cfg
    reference_model, reference = completed.model, completed.history

    interrupted = build_model(cfg)
    path = tmp_path / "fit.pt"
    kill_after(interrupted, KILL_AFTER)
    with pytest.raises(KilledError):
        fit(interrupted, cfg, checkpoint=path)

    # The kill is between checkpoints, so what survived is the step-4 payload and steps 4-5
    # are lost work.
    saved = load_checkpoint(path)
    assert saved.step == (KILL_AFTER // CHECKPOINT_EVERY) * CHECKPOINT_EVERY
    assert saved.step < KILL_AFTER

    # A restarted container builds the model again from nothing. `interrupted` is not reused.
    resumed_model = build_model(cfg)
    resumed = fit(resumed_model, cfg, checkpoint=path)

    assert_same_model(reference_model, resumed_model)
    assert_same_history(reference, resumed)


def test_a_completed_fit_resumes_to_a_no_op(completed: Completed):
    """Re-running a finished fit against its own checkpoint returns it without training.

    The final write happens after ``model.eval()``, so a resumed run starts with
    ``step == steps``, takes no optimiser step, and returns the restored history. That is what
    makes the campaign driver's ``--skip-existing`` and this checkpoint compose: a unit whose
    fit finished but whose scoring did not costs nothing to re-enter.
    """
    assert load_checkpoint(completed.path).step == STEPS
    second_model = build_model(completed.cfg)
    second = fit(second_model, completed.cfg, checkpoint=completed.path)
    assert_same_model(completed.model, second_model)
    assert_same_history(completed.history, second)


def test_resume_false_starts_over(completed: Completed, tmp_path):
    """``resume=False`` ignores an existing checkpoint and overwrites it."""
    cfg = completed.cfg
    path = private_copy(completed, tmp_path)
    assert load_checkpoint(path).step == STEPS

    model = build_model(cfg)
    kill_after(model, 2)
    with pytest.raises(KilledError):
        fit(model, cfg, checkpoint=path, resume=False)
    # Step 2 is before the first interval, so the file is untouched by the killed run and the
    # restart is visible only in that the run did not begin at step 12.
    assert load_checkpoint(path).step == STEPS


def test_checkpointing_is_off_by_default(completed: Completed):
    """No path, no file, and the same result as before the argument existed."""
    without = build_model(completed.cfg)
    plain = fit(without, completed.cfg)
    assert_same_model(completed.model, without)
    assert_same_history(completed.history, plain)


# --------------------------------------------------------------------------------------
# what the payload refuses
# --------------------------------------------------------------------------------------


def test_resume_refuses_a_different_config(completed: Completed):
    """A checkpoint from another config raises and names what differs (Convention 6)."""
    other = checkpoint_cfg(lr=1e-3)
    with pytest.raises(CheckpointError, match="config_hash"):
        fit(build_model(other), other, checkpoint=completed.path)


def test_resume_refuses_a_different_seed_or_budget(completed: Completed):
    """Neither the seed nor the step budget may change across a resume."""
    saved = load_checkpoint(completed.path)

    with pytest.raises(CheckpointError, match="seed"):
        saved.require_compatible(steps=STEPS, seed=SEED + 1, config_hash=saved.config_hash)
    with pytest.raises(CheckpointError, match="steps"):
        saved.require_compatible(steps=STEPS + 1, seed=SEED, config_hash=saved.config_hash)


def test_resume_refuses_an_older_format(completed: Completed, tmp_path):
    """A payload written by a build with different fields raises rather than being read."""
    path = private_copy(completed, tmp_path)
    saved = load_checkpoint(path)
    save_checkpoint(path, dataclasses.replace(saved, format=CHECKPOINT_FORMAT + 1))

    with pytest.raises(CheckpointError, match="format"):
        fit(build_model(completed.cfg), completed.cfg, checkpoint=path)


def test_missing_checkpoint_field_raises(completed: Completed, tmp_path):
    """A truncated payload names the field it lost."""
    path = private_copy(completed, tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["sefl_rng"]
    torch.save(payload, path)

    with pytest.raises(CheckpointError, match="sefl_rng"):
        load_checkpoint(path)


def test_load_checkpoint_names_a_missing_file(tmp_path):
    """Reading a checkpoint that is not there says where it looked."""
    with pytest.raises(CheckpointError, match="no checkpoint at"):
        load_checkpoint(tmp_path / "absent.pt")


def test_the_rng_state_is_what_is_carried(completed: Completed):
    """The SEFL generator's state is in the payload, and it is not the one at step 0.

    Stated directly as well as through the bitwise test, because it is the field a naive
    checkpoint omits and the omission is invisible until a resumed run's numbers are compared
    against an uninterrupted one — which is the comparison nobody runs on a real fit.
    """
    saved = load_checkpoint(completed.path)

    fresh = np.random.default_rng([SEED, 0x5EF1])
    assert saved.sefl_rng["bit_generator"] == fresh.bit_generator.state["bit_generator"]
    assert saved.sefl_rng["state"] != fresh.bit_generator.state["state"]

    # And restoring it reproduces the next draw, which is the property the resume relies on.
    restored = np.random.default_rng()
    restored.bit_generator.state = saved.sefl_rng
    again = np.random.default_rng()
    again.bit_generator.state = saved.sefl_rng
    assert restored.random() == again.random()


def test_checkpoint_write_leaves_no_partial_file(completed: Completed):
    """The payload lands by rename, so an interrupted write cannot destroy the previous one.

    Also pins that the directory is created: the campaign names a path under a run directory
    that may not exist yet, and a fit that dies on ``FileNotFoundError`` at its first
    checkpoint would be a durability fix that costs a fit.
    """
    assert completed.path.is_file()
    assert not list(completed.path.parent.glob("*.tmp"))
