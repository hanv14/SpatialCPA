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

Four hypotheses for the depth deficit have been tested. **Three are dead.**

| change | Moran | Geary | UMAP mix | depth | localization |
|---|---|---|---|---|---|
| type-posterior fix (current default) | **0.917** | **0.918** | 0.815 | 0.675 | 0.577 |
| residual x0.5 | 0.908 | 0.909 | 0.687 | 0.680 | 0.581 |
| calibration off entirely | 0.863 | 0.865 | 0.828 | **0.734** | 0.578 |
| calibration anchored on the low band | 0.912 | 0.913 | 0.807 | 0.677 | **0.586** |

* **The residual does not cause it.** Halving it left depth flat (0.675 -> 0.680)
  and cost embedding mixing heavily (0.815 -> 0.687).
* **The low-frequency ratio does not cause it.** Calibration amplifies the mid
  bands (1.62x) more than the lowest (1.30x), so after ``scale_to_variance`` the
  anatomy loses relative weight — a clean theory that predicts the right sign.
  Anchoring the gains on the low band moved depth 0.675 -> 0.677. A wash. The
  knob (``protect_low_bins``) is kept and defaults off.
* **Calibration as a whole does cause it**, and the mechanism is now the only one
  left standing: gains are computed **per gene**, so each gene gets its own
  spatial reweighting. That is exactly what ``paper_morans_pearson`` rewards —
  per-gene spectral energy is what it measures — and it breaks the *cross-gene*
  spatial coherence that ``paper_marker_depth_r`` and
  ``paper_celltype_localization`` depend on. Off: depth +0.06, Moran -0.05.

**This is a design tension, not a tuning problem.** The calibration stage is the
method's headline contribution and it optimizes one metric family at the direct
expense of another. No setting found so far gets both.

### Grounding was tried, and it failed

The ranking evidence pointed hard at grounding — every method above v16 on depth
and localization carries real profiles into the virtual slice — so v16 grew a
grounding stage: each virtual cell is given a real training cell's profile,
selected in a joint space of position and generated expression, restricted to
donors of its own assigned type, then blended with the generated field. Position,
cell type and the expression target all still come from the generative side.

It made things worse, twice.

| config | Moran | Geary | UMAP mix | depth | localization |
|---|---|---|---|---|---|
| **ungrounded (the default)** | **0.917** | **0.918** | **0.815** | **0.675** | 0.577 |
| grounded, first implementation | 0.760 | 0.766 | 0.536 | 0.667 | 0.579 |
| grounded, dimension-corrected | 0.810 | 0.814 | 0.617 | 0.661 | 0.575 |

The first version had a real bug: the 27-dimensional expression block outweighed
the 2-dimensional position block about 13:1 whatever the weight was set to, so
the match degenerated into "find the training cell most like this smoothed
profile", which selects near-population-mean cells everywhere. Scaling each block
by the square root of its dimensionality recovered 0.08 of embedding mixing and
changed nothing else.

What remains is not a bug. Nearest-neighbour donor selection draws the same cells
repeatedly, and blending a real profile with a smoothed one shrinks the variance
of both. The result is a duplicated, variance-compressed sample of the real
expression distribution — which is exactly what ``paper_umap_mixing`` is built to
detect, and it detects it. **Grounding does not transfer to v16 as a bolt-on**,
and it notably did not move depth (0.675 -> 0.661) either, which was the reason
for trying it.

The stage is kept behind ``ExpressionConfig.ground`` (default off) and
``--no-ground``/`--edit-weight`/`--ground-space-weight`, because the reasoning
that motivated it still stands even though this implementation does not. v14
grounds successfully, so the difference is in *how* — v14 grounds a per-cell
latent it generated jointly with position, v16 grounds a section-level spectral
field. Making it work here would mean generating a per-cell latent to match
against, which is a redesign, not a parameter.

### The structural problem

v14 scores depth 0.819 and localization 0.785 on STARmap; v16 scores 0.690 and
0.576. The difference is not subtle and it is not spectral. **v14 grounds each
generated cell in a real local training profile; v16 synthesizes expression and
deliberately does not.** Every method that scores well on these two metrics —
SpatialZ 0.920/0.817, v8 0.894/0.698, v15 0.914/0.759, v14 0.819/0.785 — carries
real profiles into the virtual slice in some form. v16 is the only one that does
not, and it is last on both.

The metrics reward grounding. That is worth stating plainly before more effort
goes into the generative side: on this benchmark, synthesizing expression from a
latent is a handicap on three of the five headline numbers, and the one metric
v16 wins (``paper_umap_mixing``, where it beats SpatialZ 0.817 vs 0.793) is the
one that rewards *not* copying.

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
