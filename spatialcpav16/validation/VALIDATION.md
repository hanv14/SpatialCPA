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

**Full leaderboard, run by the repo owner** on `starmap_visual_cortex`,
`paper_2_4_6`, mean over holdouts — the comparison this package could not make
itself, since SpatialZ needs `bench_spatialz` and the Zenodo distribution:

| method | UMAP mix | Moran | Geary | depth | localization |
|---|---|---|---|---|---|
| **spatialz** | 0.793 | **0.929** | **0.931** | **0.920** | **0.817** |
| feast | 0.770 | 0.774 | 0.775 | 0.769 | 0.000 |
| isost | **0.990** | 0.788 | 0.798 | 0.698 | 0.000 |
| spatialcpav8_gen | 0.863 | 0.912 | 0.914 | 0.894 | 0.698 |
| spatialcpav11_gen | 0.591 | 0.828 | 0.830 | 0.822 | 0.466 |
| spatialcpav14_gen | 0.788 | 0.924 | 0.927 | 0.819 | 0.785 |
| spatialcpav15_gen | 0.828 | 0.903 | 0.903 | 0.914 | 0.759 |
| **spatialcpav16_gen** | 0.817 | 0.893 | 0.893 | 0.703 | 0.437 |

**v16 beats SpatialZ on one of the five** (`paper_umap_mixing`, 0.817 vs 0.793)
and loses the other four. It is also **worse than v14 — the version it is meant
to enhance — on four of five**. The design goal is not met.

The one metric it wins is the one its mechanism targets most directly: whole-cell
residual draws restore single-cell dispersion *and* gene-gene covariance, which
is what embedding mixing reads. The two it loses badly are spatial placement.

### After the type-posterior fix

Cell type was originally carried through the same harmonic basis as expression.
That is the wrong basis for it: at 64 modes over ~4 000 cells a mode spans ~60
cells, coarser than most cell types' spatial domain, so the blurred argmax
collapses to the commonest types and the composition constraint then forces
correct *counts* onto arbitrary cells. Estimating the type posterior at cell
resolution instead (`model._type_posterior`):

| config | Moran | Geary | UMAP mix | depth | localization |
|---|---|---|---|---|---|
| v16 as benchmarked above | 0.893 | 0.893 | 0.817 | 0.703 | 0.437 |
| **+ type-posterior fix** | **0.917** | **0.918** | 0.815 | 0.675 | **0.577** |
| + fix, calibration off | 0.863 | 0.865 | 0.828 | **0.734** | 0.578 |
| + fix, residual x0.5 | 0.908 | 0.909 | 0.687 | 0.680 | 0.581 |
| *SpatialZ* | *0.929* | *0.931* | *0.793* | *0.920* | *0.817* |

Localization +0.14 and Moran/Geary +0.024, depth −0.028. Still short of SpatialZ
on four of five.

### What the ablations establish

**The spectral calibration works, and its cost is now measured.** Turning it off
drops Moran 0.917 -> 0.863 and Geary 0.918 -> 0.865, and raises depth 0.675 ->
0.734. It does exactly what it was built to do — set per-gene spatial
autocorrelation — and it does so partly by suppressing the lowest frequency band,
which is where the laminar gradient `paper_marker_depth_r` measures lives. That
is a genuine mechanism with a genuine trade-off, not a wash.

**The residual is not what breaks depth.** Halving it left depth flat (0.675 ->
0.680) and cost embedding mixing heavily (0.815 -> 0.687). The hypothesis that
unstructured residual variance was blurring the marker field is wrong, and the
depth deficit has to be explained by the smooth field or the pose it is scored
in, not by the noise added on top.

**Best achievable today is still not enough.** Taking the best cell of each
column across all configurations gives 0.917 / 0.918 / 0.828 / 0.734 / 0.581
against SpatialZ's 0.929 / 0.931 / 0.793 / 0.920 / 0.817 — a win on embedding
mixing, near-parity on the two autocorrelation metrics, and a wide loss on depth
and localization.

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

`paper_marker_depth_r` (0.73 best, SpatialZ 0.92) and
`paper_celltype_localization` (0.58 best, SpatialZ 0.82) are the whole gap. In
rough order of expected value:

1. **Find out why depth is low, since the residual has been ruled out.** Two
   candidates remain and they are distinguishable: the *pose*, which
   `align_by_expression` selects by binned marker-field agreement and which a
   weak marker field would get wrong (check `align_rotation_deg` and
   `align_runner_up` in `metrics.json` — a marginal decision is recorded there);
   or the smooth field genuinely not carrying the laminar gradient, which the
   per-marker `marker_<gene>_field_r` breakdown would show directly.
2. **Make the calibration frequency-selective.** It currently costs 0.06 of depth
   to buy 0.05 of Moran because it rescales the lowest band along with the rest.
   Exempting the lowest bin — where the anatomy lives — should keep most of the
   autocorrelation gain without the depth cost.
3. **Condition residual draws on position, not only type.** Halving the residual
   showed how much embedding mixing depends on it (0.815 -> 0.687), so it cannot
   simply be reduced; it has to become locally appropriate instead of smaller.
4. **Reconsider whether the spectral basis should carry expression at all.** The
   type channels had to leave it to work. The same argument — that the basis
   resolves ~60 cells and the signal turns over faster — may apply to the marker
   genes that `depth_r` scores.

## 6. Reproduce

```bash
cd benchmark-pbya-v3
python -m src.bench3.prepare_starmap
python -m src.bench3.run_all --methods spatialcpav16_gen --dataset starmap_visual_cortex
python -m src.bench3.selftest          # the flanking_copy / oracle probe numbers
```
