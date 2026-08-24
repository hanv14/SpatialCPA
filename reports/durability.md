# Making long runs durable, after three container rebuilds in one day

Each rebuild returned the container to an identical snapshot (`git` at `dfadaf7`, `runs/pilot/` as
of 10:53) and destroyed everything written since. Cost so far: two 2400-step coupling checkpoints,
one full re-scoring pass, three attempts at the wide-gap T06 arms, and the in-flight logs that would
have said how far any of them got. **This is a campaign-blocking property, not an annoyance.**

## What actually survives, and why

Two things came back after every rebuild: content inside the snapshot, and content pushed to the
git remote. Nothing else did. `runs/` is not special for being `runs/` — it is unprotected for being
untracked. The durable channel available in this container is **the remote**, and the practical rule
that follows is:

> **Preserve the measurement; regenerate the model.**

That rule is affordable here precisely because of Convention 3: every stochastic function takes an
explicit seed and two runs with the same seed are bitwise identical, with a test asserting it. A
lost checkpoint is therefore *reproducible*, at the cost of its fit time. A lost measurement is
gone. So small artifacts (reports, JSON, tables) get committed the moment they exist, and 49 MB
`.pt` files are never treated as things to keep.

## The gap, precisely

| layer | state | consequence |
|---|---|---|
| campaign driver (`bench3 run_all`) | **already resumable** — `--skip-existing` skips any run whose `prediction.h5` exists | a rebuild costs at most the in-flight unit, not the campaign |
| the fit inside a unit (`train_ctfflow`) | **no checkpointing at all** — grep finds no `torch.save`, no resume path | a rebuild costs the whole fit, up to 57 minutes |

So the outer loop is protected and the inner loop is not. That is the entire problem.

**How much it costs, if the campaign ran here.** With fits of duration `D` and interruptions
arriving at mean interval `T`, the expected wasted fraction of fit time is about `D / 2T` for
uniformly-arriving interruptions. At `D` = 57 min and the observed `T` of roughly an hour, that is
**~47% of all fit time wasted**, and when `D > T` progress can stall entirely — a unit that can
never finish inside a window never finishes at all. A 220 CPU-hour campaign cannot absorb that.

## Options, in the order I would do them

> **Outcome (2026-08-24): option 1 is implemented.** `train_ctfflow(checkpoint=..., resume=...)`
> plus `spatialcpav25_gen/train/checkpoint.py` and `Config.checkpoint_every_n_steps`. The acceptance
> test is the one named below — a run interrupted at step 6 and resumed from step 4 is **bitwise
> identical** to an uninterrupted one, asserted over every parameter, every buffer and the whole
> history (`tests/test_checkpoint.py`). T07's in-place SEFL `numpy` generator is in the payload;
> it is the piece a naive checkpoint omits. The bench3 wrapper writes one beside each unit's
> `prediction.h5` (`--no-fit-checkpoint` opts out), so option 4 holds: no second resume protocol,
> just `--skip-existing` plus this. `progress/durability_checkpointing.md`.

**1. Checkpoint the fit (the missing piece).** `train_ctfflow` writes `{step, state_dict,
optimizer_state, generator_state}` every `N` steps to a configurable path, and takes a resume path
that reloads all four. Cost: moderate, on the order of 100 lines plus a `Config` field.

The subtle part, and the reason this is not a two-line change: **the RNG state must be
checkpointed**, or a resumed run diverges from an unbroken one and Convention 3's determinism test
becomes a lie. The acceptance test for this work is therefore not "it resumes" but "a run
interrupted at step k and resumed is **bitwise identical** to an uninterrupted run" — which is
cheap to assert on a short fixture fit and is the only assertion that actually protects the
convention.

**2. Size in-container runs to the window.** Anything run here should finish inside ~20 minutes and
be committed immediately. The wide-gap T06 arms are ~10 minutes each and should have been run in
parallel and committed singly from the start; the 57-minute coupling fits never belonged here.

**3. Run the campaign on the server, which was always the plan.** `specs/10` §3's environment split
already puts the real data and the campaign there, and 220 CPU-hours was never going to run in this
container. Checkpointing still matters on the server — long jobs get interrupted there too — but
the hourly-rebuild behaviour is container-specific and should not shape the campaign's design.

**4. Do not build a bespoke resume protocol.** bench3 already has one at the right granularity, and
adding a second layer of state to a benchmark harness under a hard additivity constraint is exactly
the kind of change that risks other methods' results. `--skip-existing` plus fit checkpointing
covers it.

## What I would not do

* **Commit checkpoints.** 49 MB each, no LFS, and Convention 3 makes them regenerable. Committing
  them trades a real repository for a hypothetical time saving.
* **Reduce the fit budget to fit the window.** T09 selected 2400 steps on a measured gate. Choosing
  a training budget around an infrastructure limit would be fitting the method to the container.
