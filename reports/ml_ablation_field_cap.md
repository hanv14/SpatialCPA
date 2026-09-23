# Why the `*_lgbmfield / lgbmband / lgbmbalance` line stops where it does

*Status: a correction and a bound. Read the correction first — it invalidates
numbers published in this repository, including in `config.py` and in the commit
message of `5cd12e0`.*

## 1. The correction: the fixture leaked

Every fixture table quoted for `v*_gbmfield`, `v*_lgbmfield`, `v*_lgbmrepair`,
`v*_lgbmband` and `v*_lgbmbalance` was produced by a synthetic harness that built
the learner's predicted field like this:

```python
pred      = learner_pred(w["gt_field"], ...)   # the HELD-OUT section's own field
pool_pred = learner_pred(w["tr_field"], ...)
```

`w["gt_field"]` is the ground-truth field of the section being generated. The
real learner never sees it: `build_features` gives it `(x, y, z, cell_type)` and
it is fit on the *flanking* sections only. So the harness handed the regressor an
oracle estimate of the target, and then measured how much that oracle could
correct the donor copies. It could correct them a great deal. That gap is a
property of the harness, not of the method.

It is the same class of error as the earlier random-incumbent flaw: the harness
was generous to the arm under test in a way the real pipeline is not.

**Consequence.** The fixture tables in `config.py`'s `*_lgbmband` and
`*_lgbmbalance` blocks, and in commit `5cd12e0`, overstate the achievable gain.
They are not retracted as *measurements* — the arithmetic was right — but the
quantity they measure is "what an oracle-guided selection would buy", which is
not the quantity anyone cares about. `_field_fixture.py` is the corrected
harness; it is the one to use from here.

## 2. The bound: the learner and the donors drink from the same well

`_field_fixture.py` gives the field a `z` dependence, puts the query section at
`z = 0` and the donor pool on the flanking sections at `z = ±dz`, and lets the
learner predict only what a model fit on those flanks can predict. The z-drift is
then a **shared** error — it is in the prediction *and* in the donors — exactly
as on real tissue.

Ranked as `evaluate_paper.py:670` ranks (per-gene `rankdata`, then `(r-0.5)/n`),
mean of 3 worlds:

| dz | host field_r | `lgbm` emit field_r | **headroom** |
|---:|---:|---:|---:|
| 0.25 | 0.5490 | 0.5829 | **0.0338** |
| 0.50 | 0.4846 | 0.5454 | **0.0607** |
| 1.00 | 0.2318 | 0.2725 | **0.0407** |
| 2.00 | −0.0020 | 0.0016 | **0.0037** |

The headroom column is the entire difference between *copying donors* and
*emitting the regressor's field*. It never exceeds 0.06. Every method in this
line works by moving the emission somewhere between those two poles, so that
column bounds what any of them can win on the binned-field family — and
`paper_marker_field_r`, `paper_marker_depth_r` and `paper_celltype_localization`
are all read off the binned field (the last one only through the pose that
`align_by_expression` picks from it).

The reason is structural and is not a tuning problem. The regressor is fit on the
flanking sections; the host's donors are copied from the flanking sections. The
two carry nearly the same information about the query plane, so editing the
donors toward the prediction can only exploit the part they do **not** share —
the learner's interpolation across `z`, and its pooling across cells. That is
real, and it is small.

## 3. What is measured, in the instrument's own units

Every metric in scope reads `pR = _rank_normalize(pred_X)`
(`evaluate_paper.py:670`): `spatial_autocorrelation_metrics`,
`embedding_continuity` and `marker_metrics` are each handed `pR`, and
`align_by_expression` scores poses on `pR[:, cols]`. Two consequences worth
recording, because both contradict assumptions this line of work was built on:

1. **Per-gene mean and variance are destroyed.** After ranking every gene has
   mean 0.5 and identical spread. The emitted population's *dispersion* — the
   thing the typicality band was introduced to protect, and the `var/GT` column
   that three rounds of tuning were steered by — is invisible to all six metrics.
   The band still earns its place, but for a different reason: it stops the
   selection becoming a monotone function of the predicted field, which is what
   collapses cell-level autocorrelation.
2. **Every gene is weighted equally.** A squared-residual objective in raw space
   is decided by whichever genes have the largest counts; the metric averages a
   per-gene Pearson r. `--select-space rank` measures the objective in the
   evaluator's units instead. Measured (4 worlds, dz = 0.25 / 0.5) it buys
   field_r `+0.018 / +0.024` and **costs Geary's r `−0.079 / −0.031`**. That
   fails the standing requirement that Moran's and Geary's not regress, so it is
   a documented, default-off flag and not a shipped method.

## 4. Where the remaining leverage actually is

`paper_celltype_localization` does not depend on emitted expression at all except
through the pose: the ML variants leave coordinates and cell types untouched, so
the only way expression moves that metric is by changing which pose
`align_by_expression` returns. The same pose then fixes `paper_marker_field_r`
and `paper_marker_depth_r`.

That makes the pose decision, not the field's last few points of accuracy, the
thing with discrete upside: a section whose pose is chosen wrongly (a flip, a
90° error) loses all three metrics at once, and recovering it is worth far more
than any amount of incremental field correction. `align_by_expression` already
records `align_rotation_deg`, `align_score` and `align_runner_up` per section
precisely so "a marginal decision is visible rather than silent".

**This is the diagnostic to run next**, and it needs the campaign's own output,
not a fixture: the distribution of `align_rotation_deg` and the
`align_score − align_runner_up` margin across sections, for the host and for one
ML variant. If margins are wide and rotations agree, this line is done and the
bound in §2 is the answer. If some sections sit near a tie, they are where the
metric is actually being lost.
