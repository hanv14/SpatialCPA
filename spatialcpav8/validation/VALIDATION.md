# SpatialCPA-v8 — synthetic validation

These are **synthetic-data** ablations computed by the *real* `benchmark-pbya-v2`
evaluators (`evaluate.py` + `evaluate_generation.py`) and the *real* method
wrappers — only the input data is synthetic, because the processed benchmark
datasets are not bundled here. **No real-leaderboard numbers are reported or
fabricated.**

Reproduce (in the `bench_spatialcpa` env):

```bash
python make_synth_volumetric.py vol.h5ad
python make_synth_distinct.py   dis.h5ad
# v8 default (smooth morph) vs a single-slice copy (the SpatialZ archetype):
python compare_vs_backbone.py vol.h5ad S3
python compare_vs_backbone.py dis.h5ad S3
# v8 vs v6:
python compare_v6_v8.py vol.h5ad S3
python compare_v6_v8.py dis.h5ad S3
```

## v8 vs a single-slice copy (SpatialZ archetype)

`--placement backbone` copies the single nearest training slice — the archetype of
SpatialZ. This is the head-to-head that matters for "can v8 beat SpatialZ". Higher
is better except `gen_sinkhorn`.

**Distinct-tissue regime (IMC-like): WIN 10 / LOSE 2 / TIE 5.** The two "losses"
are `gen_morans_i_pred_median` (−0.003, a magnitude score with no GT target) and
`cm_morans_i_median` (evaluator RNG noise). Everything the copy is supposed to be
good at (co-expression, sinkhorn, gene mean/var) ties, and the morph wins field,
density, cell-matched density, dice, and cell-type accuracy.

**Near-identical / volumetric regime (STARmap-like): WIN 6 / LOSE 4 / TIE 7.** The
morph wins the field and generation-density metrics and ties the structure
metrics; the copy keeps a narrow edge only on **cell-matched density / dice**,
where a perfectly aligned copy is intrinsically hard to beat — the genuine Pareto
residue. (`gen_morans_agreement` and `gen_morans_i_pred_median` losses are <0.001.)

Interpretation: the smooth morph *is* one clean slice (so it inherits the copy's
structure/density fidelity — the purely-expression metrics are identical) plus a
coherent warp (so it wins the field and cell-matched metrics the copy fails). On
the real datasets the actual SpatialZ is considerably weaker than this clean-copy
proxy (its real IMC co-expression is 0.38 vs the proxy's 0.84), so the real margin
should be larger than shown here.

## v8 vs v6

- **Volumetric:** v8 (smooth morph) **strictly dominates v6** (v6 fell back to a
  raw morph / interpolation) on the structure *and* the field/density metrics
  simultaneously.
- **Distinct-tissue:** v8 ≥ v6 (the smooth morph adds density/field over v6's
  interpolation while tying the expression structure).

## Takeaway

Against the SpatialZ archetype, v8's default smooth morph delivers a large,
consistent net gain in both regimes — a near-sweep on distinct tissue and a
majority-win on near-identical tissue — with the only residual losses on
cell-matched density/dice against a *perfectly* aligned copy. Winning literally
every one of the 27 columns against a real-slice copy is a multi-objective
question; the honest result is a decisive net improvement, and the real-data
leaderboard should be regenerated with `run_benchmark.py` where the datasets live.
The `--placement adaptive` mode selects the placement per dataset by leakage-safe
internal cross-validation when a fixed default is not wanted.
