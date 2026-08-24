# Durability — fit checkpointing

Part of [PROGRESS.md](../PROGRESS.md). Not a task from `specs/`: this is option 1 of
[reports/durability.md](../reports/durability.md), which costed the gap after three container
rebuilds in one day.

### Fit checkpointing in `train_ctfflow` (2026-08-24)

**The gap it closes.** The campaign driver was already resumable at the unit level
(`bench3 run_all --skip-existing` skips a unit whose `prediction.h5` exists) and the fit inside a
unit was not — `grep` found no `torch.save` and no resume path in `train_ctfflow`. So a rebuild cost
the whole fit, up to 57 minutes. With fits of duration `D` and interruptions at mean interval `T`,
the expected wasted fraction is about `D / 2T`; at the observed numbers that is **~47 % of all fit
time**, and when `D > T` a unit never finishes at all.

**What was built.** `spatialcpav25_gen/train/checkpoint.py` (`FitCheckpoint`, `save_checkpoint`,
`load_checkpoint`, `capture`, `load_into`, `restore_derived_buffers`, `CheckpointError`), two new
keyword arguments on `train_ctfflow` (`checkpoint: str | Path | None`, `resume: bool = True`), one
`Config` field (`checkpoint_every_n_steps = 50`), and `tests/test_checkpoint.py` (11 tests, **15 s**
in the fast suite). Passing no path checkpoints nothing and the loop behaves exactly as before.

**The acceptance test is bitwise identity, not "it resumes".** Convention 3 says two runs with the
same seed are bitwise identical and a test asserts it; a resume that landed merely *close* would
quietly make that untrue, and every seeded claim in the project rests on it. So
`test_resumed_fit_is_bitwise_identical` kills a fit at step 6 (deliberately **not** on a checkpoint
boundary — the last good payload is step 4 and steps 4-5 are lost work to redo), throws the model
away, rebuilds it from nothing as a restarted container would, resumes from the file, and asserts
**bitwise** equality of every parameter, every buffer and the whole recorded history against a run
that was never interrupted. It passes.

That one assertion covers the optimiser moments, the cosine schedule's position and T07's generator
without a test each, because the parameters after the replayed steps depend on all three.

**What goes in the payload, and why that list.** Everything the loop carries *across* a step
boundary; nothing derived inside a step from `(seed, step)`, because that is regenerated
identically.

| carried across steps (saved) | derived inside the step (not saved) |
|---|---|
| model parameters and buffers | `data.sample_batch(cfg, seed=seed, step=step, ...)` |
| AdamW state | `RotationContext.random(cfg, seed + step, ...)` |
| the cosine schedule's position | `metric_aware_terms(..., step=step, seed=seed)` |
| the `EMA` shadow | `LOSOScheduler`'s fold order (a pure function of `seed`) |
| T07's `EMATeacher` parameters | the flow's per-batch generator (keyed on the batch's rows) |
| **T07's SEFL `numpy` generator** | `embeddings.set_progress(step / (steps - 1))` |
| the recorded `TrainHistory` | |

**The SEFL generator is the whole subtlety.** `sefl_gen = np.random.default_rng([seed, 0x5EF1])` is a
single generator **advanced in place** across the run — the one piece of loop state not derived from
`(seed, step)`. A resumed fit that rebuilt it from the seed would replay the consistency block's
draws from step 0 and diverge, invisibly, because nobody compares a resumed real fit against an
uninterrupted one. Its `bit_generator.state` is in the payload, and `test_the_rng_state_is_what_is
_carried` asserts it is present and is *not* the step-0 state. The test config turns the SEFL
weights on precisely so this path is exercised.

**Refusals (Convention 6).** A checkpoint is a resume point, not a portable model: it carries no
`Config`, no volume, no embeddings. `Config.content_hash()`, `seed` and `steps` are stored and
checked, and a mismatch raises naming the field and both values rather than continuing into a fit
no seed reproduces. So does a payload from a different `CHECKPOINT_FORMAT`, a payload missing a
field, and a run whose SEFL weights disagree with the checkpoint's about whether a teacher exists.

**Details worth stating.**

* Written **after** the step's `optimiser.step()`, EMA update and history record, so the payload
  holds whole steps and never half of one.
* Written to `<path>.tmp` and `os.replace`d. The point of the module is surviving a process that
  dies at an arbitrary instant, and a `torch.save` interrupted mid-write would leave a truncated
  file where the previous good checkpoint used to be.
* A final write happens after the loop, so re-entering a finished fit is a no-op that returns the
  restored history — which is what makes this compose with `--skip-existing` for a unit whose fit
  finished but whose scoring did not.
* The triplane's per-orientation query frames are `persistent=False` and are derived from the
  `rotation` / `centre` buffers a `state_dict` *does* carry, so `restore_derived_buffers` rebinds
  the loaded pose through the module's own public API after a load. The GRF's float32 copies need
  nothing: they derive from buffers that are never trained.

**What was deliberately not done**, following `reports/durability.md`: no checkpoints are committed
(49 MB each, no LFS, and Convention 3 makes them regenerable), no second resume protocol is added
to bench3, and the fit budget was not reduced to fit the container's window.
