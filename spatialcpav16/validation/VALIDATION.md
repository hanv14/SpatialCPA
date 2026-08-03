# SpatialCPA-v16 — validation status

**Every number here was printed by a real run of the real `benchmark-pbya-v3`
evaluator on the real STARmap volume. Nothing is fabricated, and the headline
result is that v16 does not yet meet its design goal.**

---

## 1. What was run

* Dataset: `starmap_visual_cortex`, built by `src.bench3.prepare_starmap`
  (28 978 cells, 28 genes, 7 sections of 11 planes — the published partition).
* Design: `paper_2_4_6` — sections 2, 4 and 6 held out **simultaneously**,
  16 527 training cells, 12 451 held out. Leakage guard passed.
* Method input: `train_registered.h5ad` built by `src.bench3.run_benchmark.build_input`,
  the same file every method receives.
* Scoring: `src.bench3.run_benchmark.evaluate_prediction` — the paper evaluator,
  unmodified, UMAP enabled.

Baselines are the harness's own probes from `src.bench3.selftest`:
`flanking_copy` (the nearest *training* section, copied — what a method that
ignores interpolation scores) and `oracle` (the real held-out cells — the
ceiling).

## 2. The five target metrics

| metric | **v16** (M=64) | v16 (M=384) | `flanking_copy` | `oracle` |
|---|---|---|---|---|
| `paper_morans_pearson` | +0.897 | +0.876 | **+0.975** | +1.000 |
| `paper_gearys_pearson` | +0.896 | +0.879 | **+0.976** | +1.000 |
| `paper_umap_mixing` | +0.827 | +0.772 | **+0.963** | +1.000 |
| `paper_marker_depth_r` | +0.754 | +0.692 | **+0.944** | +1.000 |
| `paper_celltype_localization` | +0.459 | +0.511 | **+0.754** | +0.982 |

**v16 loses to a flanking copy on all five.** It therefore cannot be claimed to
beat SpatialZ, and no such claim is made anywhere in this package.

**SpatialZ was not run.** It needs the `bench_spatialz` conda environment and the
SpatialZ distribution from Zenodo, neither of which exists in the environment
this was developed in. `flanking_copy` is a *proxy* — a strong one, and the
right one for a first look, but it is not SpatialZ and no number here says
anything directly about SpatialZ's performance.

## 3. What *is* established

The mathematical core is verified exactly. On synthetic signals spanning three
regimes, v16's own prediction of Moran's I — computed from the spectral
coefficients plus the residual bucket, never from the coordinates — agrees with
the evaluator's measurement to four decimal places:

| signal | predicted | measured |
|---|---|---|
| smooth spatial gradient | +0.9818 | +0.9818 |
| pure noise | −0.0081 | −0.0081 |
| mid-frequency (sin, σ=0.3) | +0.7831 | +0.7831 |

That is the identity the method is built on, and it holds. It also caught a real
bug: bounding the residual's effective eigenvalue to the *modelled* eigenvalue
range rather than the operator's full [−1, 1] predicts **+0.61** for the
pure-noise gene whose true value is **−0.01**.

The pipeline also runs end to end through the unmodified v3 harness — leakage
guard, wrapper contract, prediction format, all three evaluators — and produces
a scored `metrics.json` like any other method.

## 4. Diagnosis — why it loses, honestly

**The benchmark has almost no headroom on two of the five.** The v3 README
already documents that STARmap's `flank_r` is **0.98**: the per-gene Moran's I of
the two sections flanking a held-out one are 98 % correlated. So on the
autocorrelation family a trivial copy is nearly perfect by construction, and the
gap between 0.975 and 1.000 is the entire space a method has to compete in.
Beating a copy there is not a modelling problem so much as a
matching-the-copy-exactly problem.

**A de-novo generator starts behind on `paper_umap_mixing`.** The copy emits
*real cells with real profiles*, so its expression cloud is drawn from the true
distribution by definition. v16 synthesizes profiles, and any imperfection in the
conditional shows up directly as reduced mixing. v16's 0.83 means its cells are
locally distinguishable from real ones — the residual stage restores dispersion
and covariance but evidently not the fine structure of the manifold.

**The two metrics with real headroom are where v16's remaining defects are.**
`paper_celltype_localization` (0.46 against a copy's 0.75, an oracle's 0.98) is
the clearest signal: type assignment is the weak stage. The type fields are
generated from the same low-frequency spectral model as expression, and with 19
types on 64–384 harmonics the finer types are under-resolved; the
composition-constrained greedy assignment then spends its budget badly. Raising
the mode count moved localization the right way (0.46 → 0.51) and moved embedding
mixing and depth the wrong way, which says the current single basis is being
asked to serve two jobs with different resolution requirements.

**The fix that worked, for the record.** The first implementation scored
`paper_marker_depth_r` **0.19** and `paper_celltype_localization` **0.04** —
right amount of spatial structure, arbitrary arrangement. The cause was the
eigenvector sign/ordering ambiguity: mode *m* of one section is not mode *m* of
another, and asking the flow to learn the phase from four sections is asking for
something that is not in the data. Replacing basis-matching with **field
evaluation** (evaluate a bracket's smooth field at the target's positions, then
transform into the target's own basis, so nothing is ever paired) took those to
**0.75** and **0.46**. That is documented in `nets.evaluate_field` because it is
the design decision the method turns on.

## 5. What would have to happen next

In rough order of expected value:

1. **Separate the type model from the expression model.** Types need a different
   basis resolution than expression; sharing one is currently costing both.
2. **Condition the residual draw on local context, not just cell type.** A
   type-conditional i.i.d. draw restores marginals and covariance but no local
   structure, which is the most likely cause of the embedding-mixing gap.
3. **Run against SpatialZ.** Everything above is inference from a proxy baseline.
   The comparison the request actually asks for has not been made.
4. **Check whether the goal is reachable on STARmap at all.** With `flank_r` at
   0.98, a dataset with wider section spacing (the v3 README recommends exactly
   this) may be where a generative method can show an advantage that STARmap
   cannot resolve.

## 6. Reproduce

```bash
cd benchmark-pbya-v3
python -m src.bench3.prepare_starmap
python -m src.bench3.run_all --methods spatialcpav16_gen --dataset starmap_visual_cortex
python -m src.bench3.selftest          # the flanking_copy / oracle probe numbers
```
