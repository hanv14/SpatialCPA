"""Fit checkpointing for :func:`~spatialcpav25_gen.model.spatialcpav25_gen.train_ctfflow`.

Why this exists
---------------
``reports/durability.md`` measured the gap: the campaign driver is already resumable
(``bench3 run_all --skip-existing`` skips a unit whose ``prediction.h5`` exists) and the fit
*inside* a unit was not, so a container rebuilt mid-fit cost the whole fit — up to 57 minutes,
three times in one day. With interruptions arriving at mean interval ``T`` and fits of
duration ``D``, the expected wasted fraction is about ``D / 2T``; at the observed numbers that
is ~47% of all fit time, and when ``D > T`` a unit can never finish at all.

What "resumable" has to mean here
---------------------------------
Convention 3 says two runs with the same seed are bitwise identical and a test asserts it. A
resume that is merely *close* would quietly make that untrue, so the acceptance test for this
module is not "it resumes" but **"a run interrupted at step k and resumed is bitwise identical
to an uninterrupted one"** (``tests/test_checkpoint.py``). That is what fixes the contents of
:class:`FitCheckpoint`: everything the loop carries *across* a step boundary goes in, and
anything derived freshly from ``(seed, step)`` inside a step stays out.

What crosses a step boundary, and what does not
-----------------------------------------------
================================  ==============================================================
carried across steps (saved)      derived inside the step from ``(seed, step)`` (not saved)
================================  ==============================================================
model parameters and buffers      ``data.sample_batch(cfg, seed=seed, step=step, ...)``
AdamW state                       ``RotationContext.random(cfg, seed + step, ...)``
the cosine schedule's position    ``metric_aware_terms(..., step=step, seed=seed)``
the :class:`EMA` shadow           ``LOSOScheduler``'s fold order (a pure function of ``seed``)
T07's ``EMATeacher`` parameters   the flow's per-batch generator (keyed on the batch's rows)
T07's SEFL ``numpy`` generator    ``embeddings.set_progress(step / (steps - 1))``
the recorded ``TrainHistory``
================================  ==============================================================

The SEFL generator is the subtle one and the reason this is not a two-line change: it is a
single ``np.random.Generator`` **advanced in place** across the whole run, so a resumed fit
that rebuilt it from the seed would replay the consistency block's draws from step 0 and
diverge. Its ``bit_generator.state`` is therefore part of the checkpoint.

Derived, non-persistent buffers are deliberately absent: the GRF's float32 copies of its
draws and the triplane's per-orientation query frames are registered
``persistent=False`` and are rebuilt by ``__init__`` from the same ``cfg`` and ``grf_seed``.
:func:`restore_derived_buffers` re-derives the pose-dependent ones after a load, because those
also depend on the ``rotation`` / ``centre`` buffers the checkpoint *does* carry.

What is **not** stored: the ``Config`` itself, the volume, and the embeddings. A checkpoint is
not a portable model — it is a resume point for a fit whose caller rebuilds the same objects.
``Config.content_hash()`` is stored so that resuming into a different config raises instead of
producing a run that is reproducible from nothing (Convention 6).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:  # pragma: no cover - the model imports this module, not the other way
    from spatialcpav25_gen.losses.sefl import EMATeacher
    from spatialcpav25_gen.model.spatialcpav25_gen import EMA, CTFFlow, TrainHistory

__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "FitCheckpoint",
    "capture",
    "load_checkpoint",
    "load_into",
    "restore_derived_buffers",
    "save_checkpoint",
]

CHECKPOINT_FORMAT: int = 1
"""Version of the on-disk payload. Bumped when a field is added or its meaning changes; an
older file then raises rather than being read with a field silently missing (Convention 6)."""


class CheckpointError(RuntimeError):
    """A checkpoint could not be read, or does not belong to the run trying to resume it."""


@dataclass(frozen=True)
class FitCheckpoint:
    """One resume point of a ``train_ctfflow`` run.

    Attributes
    ----------
    step
        The **next** step to run, i.e. the number of optimiser steps already taken. Written
        after the step's ``optimiser.step()``, EMA update and history record, so the whole of
        step ``step - 1`` is in the payload and none of step ``step`` is.
    steps, seed
        The budget and the seed of the run that wrote it. Both are checked on resume: a
        different budget changes the cosine schedule's shape and a different seed changes
        every batch, so continuing across either is not a resume.
    config_hash
        ``Config.content_hash()`` of the writing run.
    model, optimiser, scheduler
        ``state_dict()`` of each.
    ema
        The :class:`~spatialcpav25_gen.model.spatialcpav25_gen.EMA` shadow, by parameter name.
    teacher
        ``state_dict()`` of T07's :class:`~spatialcpav25_gen.losses.sefl.EMATeacher` module,
        or ``None`` on a run with every SEFL weight at zero.
    sefl_rng
        ``bit_generator.state`` of the SEFL ``numpy`` generator (see the module docstring).
    history
        The recorded :class:`~spatialcpav25_gen.model.spatialcpav25_gen.TrainHistory` as a
        plain dict, so the resumed run returns the whole run's log and not just its tail.
    format
        :data:`CHECKPOINT_FORMAT`.
    """

    step: int
    steps: int
    seed: int
    config_hash: str
    model: dict[str, Any]
    optimiser: dict[str, Any]
    scheduler: dict[str, Any]
    ema: dict[str, Tensor]
    teacher: dict[str, Any] | None
    sefl_rng: dict[str, Any]
    history: dict[str, Any]
    format: int = CHECKPOINT_FORMAT

    def require_compatible(self, *, steps: int, seed: int, config_hash: str) -> None:
        """Raise :class:`CheckpointError` unless this checkpoint belongs to the calling run.

        Names the field that differs and both values: a resume that silently continued a fit
        from a different config would produce a run reproducible from no seed at all, which is
        the failure Convention 6 exists to prevent.
        """
        if self.format != CHECKPOINT_FORMAT:
            raise CheckpointError(
                f"checkpoint format {self.format} but this build writes {CHECKPOINT_FORMAT}; "
                "the payload's fields have changed. Delete the file and refit."
            )
        mismatches = [
            (name, mine, theirs)
            for name, mine, theirs in (
                ("steps", self.steps, int(steps)),
                ("seed", self.seed, int(seed)),
                ("config_hash", self.config_hash, config_hash),
            )
            if mine != theirs
        ]
        if mismatches:
            detail = "; ".join(
                f"{name}: checkpoint {mine!r}, run {theirs!r}" for name, mine, theirs in mismatches
            )
            raise CheckpointError(
                f"the checkpoint was written by a different run ({detail}). Resuming across it "
                "would give a fit no seed reproduces; pass a different checkpoint path, or "
                "resume=False to start over and overwrite."
            )
        if not 0 <= self.step <= self.steps:
            raise CheckpointError(
                f"checkpoint step {self.step} is outside [0, {self.steps}]; the file is corrupt."
            )


def save_checkpoint(path: str | Path, checkpoint: FitCheckpoint) -> None:
    """Write ``checkpoint`` to ``path`` atomically.

    Written to ``<path>.tmp`` in the same directory and then ``os.replace``d, which is atomic
    within a filesystem. The whole point of this module is surviving a process that dies at an
    arbitrary instant, and a plain ``torch.save`` interrupted mid-write leaves a truncated file
    where the previous good checkpoint used to be.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    torch.save(dataclasses.asdict(checkpoint), tmp)
    os.replace(tmp, target)


def load_checkpoint(path: str | Path) -> FitCheckpoint:
    """Read a :class:`FitCheckpoint` from ``path``.

    ``weights_only=False`` because the payload is not weights: it carries the ``numpy``
    bit-generator state and the history's plain lists beside the tensors. The file is one this
    package wrote, in a directory the caller named.
    """
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"no checkpoint at {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise CheckpointError(f"{source} does not hold a checkpoint mapping, got {type(payload)}")
    fields = {f.name for f in dataclasses.fields(FitCheckpoint)}
    missing = sorted(fields - set(payload))
    if missing:
        raise CheckpointError(
            f"{source} is missing checkpoint field(s) {missing}; it was written by a different "
            "build. Delete the file and refit."
        )
    return FitCheckpoint(**{name: payload[name] for name in fields})


def capture(
    *,
    step: int,
    steps: int,
    seed: int,
    config_hash: str,
    model: CTFFlow,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    average: EMA,
    teacher: EMATeacher | None,
    sefl_gen: np.random.Generator,
    history: TrainHistory,
) -> FitCheckpoint:
    """Snapshot the training loop's cross-step state. See the module docstring for the list.

    Every tensor is cloned: the loop keeps mutating the ones it came from, and a checkpoint
    holding views of live parameters would record whatever they were when it reached the disk.
    """
    return FitCheckpoint(
        step=int(step),
        steps=int(steps),
        seed=int(seed),
        config_hash=str(config_hash),
        model=_detached(model.state_dict()),
        optimiser=_deepcopy_state(optimiser.state_dict()),
        scheduler=_deepcopy_state(scheduler.state_dict()),
        ema={name: value.detach().clone() for name, value in average.shadow.items()},
        teacher=None if teacher is None else _detached(teacher.module.state_dict()),
        sefl_rng=_deepcopy_state(dict(sefl_gen.bit_generator.state)),
        history=dataclasses.asdict(history),
    )


def restore_derived_buffers(model: CTFFlow) -> None:
    """Re-derive the pose-dependent buffers ``load_state_dict`` cannot restore.

    ``TriplaneField`` registers its per-orientation query matrices and normalisation boxes
    ``persistent=False`` and recomputes them whenever the augmentation pose is rebound. They
    are absent from a ``state_dict`` by construction, while the ``rotation`` and ``centre``
    buffers they are derived *from* are present — so a load can leave the two disagreeing.
    Rebinding the loaded rotation puts them back in step through the module's own public API.

    The GRF's float32 copies of its draws are also non-persistent, and need nothing: they are
    derived from buffers that are never trained, so a checkpoint's values are the ones
    ``__init__`` already computed from the same ``cfg`` and ``grf_seed``.
    """
    model.field.set_rotation(model.field.rotation_matrix)


def _detached(state: dict[str, Any]) -> dict[str, Any]:
    """Clone every tensor in a ``state_dict``, leaving anything else alone."""
    return {
        name: value.detach().clone() if isinstance(value, Tensor) else value
        for name, value in state.items()
    }


def _deepcopy_state(state: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a nested optimiser/scheduler/bit-generator state, cloning its tensors."""
    return {name: _copy_value(value) for name, value in state.items()}


def _copy_value(value: Any) -> Any:
    """Recursive helper for :func:`_deepcopy_state`."""
    if isinstance(value, Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.copy()
    return value


def load_into(
    checkpoint: FitCheckpoint,
    *,
    model: CTFFlow,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    average: EMA,
    teacher: EMATeacher | None,
    sefl_gen: np.random.Generator,
) -> None:
    """Restore ``checkpoint`` into freshly built loop objects.

    ``optimiser`` is loaded **after** ``scheduler`` was constructed: building a
    ``CosineAnnealingLR`` writes ``initial_lr`` and the epoch-0 learning rate into the
    optimiser's parameter groups, and the checkpointed groups are the ones that must win.

    A run whose SEFL weights are all zero has no teacher on either side; a checkpoint and a
    run that disagree about that raise, because the disagreement means the two are not the
    same fit (Convention 6).
    """
    _require_teacher_agreement(checkpoint, teacher)
    model.load_state_dict(checkpoint.model)
    restore_derived_buffers(model)
    optimiser.load_state_dict(checkpoint.optimiser)
    scheduler.load_state_dict(checkpoint.scheduler)
    _load_ema(average, checkpoint.ema)
    if teacher is not None and checkpoint.teacher is not None:
        teacher.module.load_state_dict(checkpoint.teacher)
    sefl_gen.bit_generator.state = checkpoint.sefl_rng


def _require_teacher_agreement(checkpoint: FitCheckpoint, teacher: EMATeacher | None) -> None:
    """Raise unless the checkpoint and the resuming run agree about T07's teacher."""
    if (checkpoint.teacher is None) == (teacher is None):
        return
    raise CheckpointError(
        "the checkpoint "
        f"{'carries' if checkpoint.teacher is not None else 'has'} "
        f"{'a' if checkpoint.teacher is not None else 'no'} SEFL teacher but this run "
        f"{'has one' if teacher is not None else 'has none'}; the SEFL weights "
        "(Config.w_cross / w_thick / w_prog / w_prog_wrong) differ between the two fits."
    )


def _load_ema(average: EMA, shadow: dict[str, Tensor]) -> None:
    """Copy a checkpointed shadow into ``average``, naming any parameter that went missing."""
    missing = sorted(set(average.shadow) - set(shadow))
    unexpected = sorted(set(shadow) - set(average.shadow))
    if missing or unexpected:
        raise CheckpointError(
            f"the checkpointed EMA shadow does not match the model: missing {missing}, "
            f"unexpected {unexpected}. The two runs built different models."
        )
    with torch.no_grad():
        for name, value in shadow.items():
            average.shadow[name].copy_(value)
