# T07 — SEFL consistency losses

Part of [PROGRESS.md](../PROGRESS.md).

### T07 — SEFL: plane geometry, the three consistency losses, and one result that reframes `L_cross` (2026-08-17)

`spatialcpav25_gen/losses/sefl.py` (new), the rest of `specs/07` §1's geometry in
`infer/planes.py` (`LineSegment`, `Surface`, `intersect`, `random_plane_pair`, `curved_surface`,
`plane_pose`, `Plane.basis` / `Plane.sample_points`), the SEFL block, the ramp, the dominance
warning and the collapse alarm in `train_ctfflow`, `tests/test_sefl.py` (32 tests: 25 fast, 7 slow,
2 of them strict xfails) and `tests/test_planes.py` (17 tests). 18 new `Config` fields and one new gate set
(`GRANULARITIES`); no constant outside `Config`. `minimal_rotation` and `coexpression_modules`
promoted to public so the pose and the molecular programs are computed in one place each.

**All of `specs/07`'s acceptance tests, with numbers.** Reduced widths (T06's), 500 steps, the
spec's own schedule (warm-up 0.2, ramp 0.2, every third step), seed 20260817.

| Test | Criterion | Measured |
|---|---|---|
| `test_intersect_known_cases` | hand-computed: orthogonal planes, parallel planes → `None`, windows that miss → `None` | all five cases exact to 1e-9, including a 45° pair and an offset orthogonal pair |
| `test_noise_identical_along_intersection` | GRF values along the segment **bitwise** equal from both pathways | `torch.equal` — exact, and the values are not constant (variance > 0) |
| `test_cross_loss_decreases` | ≥ 60 % over 500 steps | **89.8 %** (14.89 → 1.52, medians of the first and last three logged values) |
| `test_no_collapse` | generated per-gene variance inside `[0.60, 1.67]` of the real section's (**both sides** — the spec states only the floor) | **1.04-1.33** on the `thick` + `prog` arm; 0.711 with SEFL off |
| `test_no_collapse_at_the_spec_w_cross` | the same, at `specs/07`'s `w_cross = 0.3` | **0.067** — **strict xfail**, see below |
| `test_no_collapse_negative_control_fails` | the criterion must **fail** with the teacher off | **0.382** < 0.60 — fails as required |
| `test_thick_counts_add` | `N(3h)` within Poisson tolerance of `3 N(h)`, and the loss must not penalise it | ratio **3.000**, `|z|` **0.02** of the 3σ budget; the count term charges **0.00** where the naive equal-counts comparison charges 4.4e4 |
| `test_prog_conditioning` | the unconditional variant homogenises the tissue, the conditional one does not | **absent: mean ratio 0.97** (per-seed 1.60 / 0.59 / 0.71; **1.27** from a SEFL-free model) — **strict xfail**, B22 |
| `test_equivariant_not_constrained` | a 90° section keeps a different in-plane Moran's I from a 0° one | coronal **0.753**, sagittal **0.891** — the sagittal cut carries the *higher* autocorrelation, which is the fixture's own anisotropy (120 µm in plane, 200 µm along z) |
| `test_sefl_cost` | < 60 % wall-clock per epoch | **39 %** (min of two repeats per arm; 34-47 % across runs, the spread being machine load) |

**The result that reframes `L_cross` — and it is a good result for the method.**
`test_generation_is_intersection_consistent_by_construction`: two crossing planes emit **bitwise
identical** expression along their intersection, on an *untrained* model, with no consistency loss
applied. `CTFFlow`'s expression pathway conditions at physical points in the data frame — retrieval,
the GRF and the Fourier encoding are all data-frame channels — and takes the cutting plane only
through the layout. The continuous 3-D field of v25 supplies for free the property `L_cross` was
invented to train, which is a **stronger** claim than "we trained for it" and belongs in T10's E5.

The corollary is the problem. The only plane-dependent channel left is the augmentation **pose**, so
that is what a two-branch loss can compare — and T04 made the triplane pose-dependent *deliberately*,
as the capacity mechanism GATE 2 rests on. Minimising `L_cross` therefore drives the field towards
pose-invariance, i.e. towards constant. Four arms, differing only in which SEFL terms are live:

| arm | reconstruction (nats/pair) | generated per-gene variance ÷ real | `L_cross` self-consistency |
|---|---|---|---|
| SEFL off | **1.738** | **0.711** | 0.0224 |
| `thick` + `prog` only (**shipped default**) | 2.082 | **1.331** | 0.0243 |
| + `w_cross = 0.3` | 2.024 | **0.065** | 0.0100 |
| + `w_cross = 0.3`, teacher off | 1.914 | 0.344 | 0.0131 |

`L_cross` falls 90 % over the run it damages, so this is not a failure to optimise it, and the damage
is attributable: with `w_cross = 0` nothing collapses. **`Config.w_cross` ships at 0** with that
table in its docstring, `loss_cross` stays built and tested (T10's A7 and E5 need it), and the
failure is pinned by a strict xfail rather than tuned away. Open risk **R6**; the two candidate
fixes — redefine the branch difference as each plane's *evidence*, or accept the by-construction
result and drop the loss — are written up for the spec's owner in SPEC_QUESTIONS **C19**. The second
is my inclination; either way it changes what the paper's SEFL section claims, so it is not a
decision to make silently.

**The collapse alarm needed two gates, not one (second one added on review).** Beyond moving it to
the generation path (below), it now also stays silent for `Config.sefl_collapse_min_steps = 100`
optimiser steps. The ramp gate alone is a *fraction* of the run, so on a four-step run it opens at
step 3 and the alarm fired inside the **fast** suite — `test_trainer_forwards_the_gene_pool`, at
0.008 of the real variance, which is an untrained decoder predicting the panel mean, not a collapse.
An alarm that fires routinely on healthy short runs is one nobody reads at T08. The floor costs no
sensitivity: the earliest alarm in any real arm measured here is step 110.
`test_collapse_alarm_is_silent_on_a_run_shorter_than_its_floor` pins it, and asserts the variance it
declines to complain about really is degenerate, so the silence is the gate working rather than the
model being fine.

**The collapse alarm had to be moved before it could see any of this.** As first written it watched
the reconstruction path's per-gene variance, which decodes the **encoder**'s latent — and the encoder
never queries the anatomical field. At the checkpoint whose *generated* section reads 0.054 that
diagnostic read **0.36–0.89**, i.e. healthy. `forward_train` now runs the generation path (prior
latent → flow → decoder → a draw) on the batch's own cells for the diagnostic, and the alarm fires
where it should: 35 alarms over the `w_cross = 0.3` arm, 10 over the shipped-default arm, none before
the ramp starts (an untrained decoder predicts near the panel mean and would trip it at step 0).

**Two other measurements that changed the implementation.**

1. **`L_thick`'s Monte-Carlo error was the loss.** With independent draws per slab the count term
   ranged over **0.007 to 250** against a reconstruction of 2.1 — the block was optimising sampling
   noise. The thin slabs' points are now the thick slab's, *partitioned* by sub-slab (common random
   numbers), which is the aggregation identity made manifest and cancels the estimator error exactly.
   The statistic is also now per cell — `relu(z² − 1) / N` rather than `z²` — because the bare
   Pearson residual grows with the section: a 10 % disagreement on a 10⁴-cell slab scores 100, which
   drove the consistency/reconstruction ratio to a median of **4.3** against `specs/07` §5's ceiling
   of 0.5. After both fixes the median ratio is **0.078** (max 0.75, warned once).
2. **The SEFL block was leaking the gene holdout.** It drew its gene subsample from the whole panel,
   so a gene held out of training received gradients through the consistency losses and its free
   residual `r_g` moved. T06's `test_trainer_forwards_the_gene_pool` caught it by mutation — the test
   that exists because the same leak once turned a zero-shot number into an in-sample one. `gene_pool`
   now threads through `sefl_terms` into every builder.

**Open risk R1 is decided** (see the section above): remedy 2 (calibrate `ell_z` at inference against
observed between-section correlation) with remedy 3 as the guard, implementation owed to T09. The
evidence T07 owed is negative, in two forms, and both are now in
`test_cross_loss_cannot_see_ell_z`. *Structural, and exact*: T03's field is continuous in 3-D, so
both branches receive the **bitwise identical** noise realisation along the intersection at every
`ell_z` — asserted at 100 / 200 / 561 / 1000 µm — and the prior's length scale therefore cancels out
of a comparison between the two branches by construction. *Measured*: `L_cross` itself varies by
**1.6 %** across that ten-fold range. No SEFL term can calibrate the anisotropy, and the same
argument applies to `L_thick` and `L_prog`; the hypothesis that T07 would supply the instrument is
closed.

**A third acceptance test does not reproduce**, and it is the one for `L_prog`'s conditioning
(B22). From a trained model, 60 steps on `L_prog` alone, the unconditional variant does **not**
homogenise the tissue relative to the conditional one: mean ratio **0.97** (per-seed 1.60 / 0.59 /
0.71) from the shipped SEFL arm and **1.27** (1.30 / 1.27 / 1.23, consistently the wrong way) from a
model trained without SEFL, where `specs/07` §4 predicts well under 1. Not a composition artefact
(the two planes' `(type, region)` mixtures differ by a median total variation of 0.51) and not one
unlucky start (an untrained model gives 2.06 / 0.53 / 0.70). The likely cause is that the
conditional loss carries three terms *per stratum* and so applies several times the gradient at the
same learning rate, which "the same number of steps" does not control for. The conditioning stays in
the loss — the a priori argument for it is not in question — and the failure is pinned as a strict
xfail carrying the numbers.

**Deviations from `specs/07`, all recorded in SPEC_QUESTIONS.** C17: the `lambda` term is computed on
`log lambda` (its squared difference is ~1e-10 in raw intensity units) and the type term as
`KL(p₂ ‖ p₁)` (identical gradients to the spec's CE, without the teacher-entropy floor that would
make a 60 % fall unreachable). C18: five constants the spec leaves open become `Config` fields —
`sefl_ramp_frac`, `sefl_genes_per_step` (the spec's own cost cap and its "compare on the shared
genes" collide: the full panel costs **+62 %** against a **< 60 %** ceiling, 64 genes costs +34 %),
`section_granularity`, `sefl_ema_teacher`, and `intersect`'s clip region (the two planes' windows,
which a `Plane` already carries). B21: `test_cross_loss_decreases` has no untrained baseline —
`L_cross` on an untrained model is **3.9e-9**, because the feature planes start at σ = 1e-2 and both
poses return the same nearly-constant field — so the criterion is measured on the run's own
trajectory.

**What the variance overshoot costs (added on review).** The shipped arm sits *above* the real
per-gene variance (1.04-1.33 across runs) where the SEFL-free arm sits below it (0.711), and the
spec's criterion is one-sided so it never looks at the direction — the same shape as GATE 2's
attention entropy passing at the *uninformative* extreme. `test_no_collapse` now asserts both sides
of a band symmetric in ratio (`[0.60, 1.67]`). The overshoot is **not** free, measured on the
distributional statistics T06 owns (generated section vs the held-out real one, same seed):

| statistic | SEFL off | SEFL on (shipped) | T06's criterion |
|---|---|---|---|
| per-gene variance ÷ real | 0.711 | **1.042** | — |
| detection rate `r` | 0.9929 | 0.9756 | > 0.95, both pass |
| detection rate MAD | 0.0233 | **0.0468** | < 0.05 — on uses 94 % of the budget |
| mean-variance log-log slope, relative error | 0.0036 | **0.2104** | < 0.15 — **on fails** |
| zero fraction (real 0.4808) | 0.4781 | 0.4877 | — |

The mean-variance relation is T06's own acceptance criterion and the SEFL-on arm misses it by 40 %
relative. These are not the six target metrics — those are T10's — so the question was put to T06's
own suite directly, at **its** 1200-step budget and configuration, and the answer is worse:

| T06 acceptance test | criterion | T06 recorded (SEFL off) | with `thick` + `prog` at 0.2 |
|---|---|---|---|
| `test_sparsity_preserved` | detection MAD < 0.05 | 0.0191 | **0.0551 — fails** |
| `test_mean_variance_relation` | slope error < 0.15 | 0.0084 | **0.2838 — fails** |
| gene-gene Frobenius covariance | vs a 7.563 baseline | 9.316 | **20.301** |

**So all three SEFL weights now ship at 0 and SEFL is opt-in.** A default that breaks the previous
task's acceptance criteria is not a default, and none of these losses has yet been shown to buy
anything. `specs/10` restates A7 as an **addition** experiment — the shipped model against
`w_thick = w_prog = 0.2` — reporting all six target metrics plus the numbers above; until it runs,
SEFL's net contribution is unverified and the paper's SEFL section cannot be written. Carried as
open risk **R7**.

**Definition of done: partially met, and the gaps are the findings.** No collapse **holds at the
shipped weights for `thick` + `prog`** (variance **1.127**; 10 alarms during the ramp, none of them
below 0.10) and **fails at `specs/07`'s `w_cross`** (0.067). Held-in reconstruction is **not**
equal or better in any SEFL arm on this fixture — 1.738 with SEFL off against 2.082 for
`thick` + `prog`, which is the second gap and is owed to T08's metric-aware terms and T09's
selector rather than to another weight guessed here. `L_cross` converges as specified (89.8 %). The consistency/reconstruction ratio and the collapse-alarm history are in the table
above and logged by `TrainHistory` (`consistency_ratio`, `collapse_alarms`, `variance_ratio`).
