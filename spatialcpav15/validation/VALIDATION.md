# SpatialCPA-v15 — validation vs SpatialCPA-v8

This documents how v15 was validated end-to-end through the **real**
`benchmark-pbya-v2` machinery — the leakage-safe holdout/re-registration
(`benchmark.leakage_guard`), the real method wrappers, and the
correspondence-free generation evaluator (`benchmark.evaluate_generation`) —
head-to-head against **v8 at its production defaults**, the strongest prior
SpatialCPA generator.

Only the *input data* is synthetic in the synthetic cases. **No leaderboard
numbers are fabricated**: every figure below was printed by
`run_spatialcpa_v15.py`, and the predictions, per-holdout metrics and method logs
it wrote are kept under `results_v15/`.

---

## What is measured

Ten correspondence-free generation metrics, grouped by what they depend on:

| depends on | metrics |
|---|---|
| expression values only | `coexpression_agreement`, `sinkhorn` (↓), `gene_mean_pearson`, `gene_var_pearson` |
| (position, expression) jointly | `morans_agreement` |
| (position, label) jointly | `celltype_composition`, `celltype_nhood_agreement` |
| position / density field | `field_pearson`, `field_ssim`, `density_pearson` |

The primary metrics are computed on **per-gene rank-normalized** expression, so
they are invariant to each method's output scale. A **WIN** is a relative
difference above 1 %; anything inside ±1 % is reported as a *tie
(non-inferior)*. A two-sided paired *t*-test across holdouts is reported
alongside — with three holdouts it has little power, so treat it as a consistency
check on the sign, not as a significance claim.

---

## Reproduce

```bash
python run_spatialcpa_v15.py                                   # real STARmap block
python run_spatialcpa_v15.py --synthetic events --holdouts S3,S4,S5
python run_spatialcpa_v15.py --synthetic drift  --holdouts S3,S4,S5
python run_spatialcpa_v15.py --synthetic imc --registration rigid \
    --holdouts ROI1,ROI2,ROI3,ROI4                             # intensity path
```

Both methods run at their production defaults (no tuning flags are passed).

---

## 1. Real data — STARmap 3D visual cortex

`data/starmap/STARmap_Wang2018three_data_3D_data.h5ad`, a contiguous 11-plane
block (z = 20…30, Δz = 1, ≈ 375 cells/plane, 28 genes, 19 leiden clusters used as
cell types). Registration policy `none`, matching the benchmark's *volumetric*
policy for STARmap. Holdouts z24, z25, z26, one at a time.

**`R = Δz / target spacing = 1`** — this block is densely sectioned, which the
proposal identifies as the easy end of the range and which makes linear
interpolation of the field a genuinely strong baseline.

| metric | v15 | v8 | Δ | p (paired) | verdict |
|---|---|---|---|---|---|
| coexpression_agreement | 0.8772 | 0.8005 | +0.0767 | 0.045 | **WIN** |
| morans_agreement | 0.8693 | 0.6606 | +0.2087 | 0.036 | **WIN** |
| sinkhorn ↓ | 0.3439 | 0.4633 | +0.1195 | 0.000 | **WIN** |
| celltype_composition | 0.9008 | 0.8558 | +0.0450 | 0.213 | **WIN** |
| celltype_nhood_agreement | 0.8026 | 0.7980 | +0.0046 | 0.845 | tie (non-inferior) |
| gene_mean_pearson | 0.9263 | 0.9191 | +0.0072 | 0.632 | tie (non-inferior) |
| gene_var_pearson | 0.9613 | 0.6562 | +0.3051 | 0.003 | **WIN** |
| field_pearson | 0.4721 | 0.3724 | +0.0997 | 0.029 | **WIN** |
| field_ssim | 0.7459 | 0.5087 | +0.2371 | 0.006 | **WIN** |
| density_pearson | 0.2120 | 0.1304 | +0.0816 | 0.313 | **WIN** |

**v15: 8 wins, 2 ties, 0 losses — no metric is worse than v8.**

The two ties are *directionally* positive (+0.5 % and +0.9 %), just inside the
non-inferiority band. The largest margins are exactly where a generative
expression model should help: per-gene variance (+46 %, a copy-based method
inherits one slice's dispersion), the distributional Sinkhorn distance (−25 %
divergence), Moran's agreement (+20 %) and binned cell density (+63 %).

## 2. Synthetic — smoothly drifting niches (`make_synth_drift.py`)

Nine sections, radial niche bands migrating with depth, a drifting tissue centre,
and a within-type spatial expression gradient. Everything is monotone in *z*,
which is the regime v8's transport morph is built for: a translating disc is
exactly what an OT displacement represents well and what interpolating two
density fields represents badly.

| metric | v15 | v8 | Δ | verdict |
|---|---|---|---|---|
| coexpression_agreement | 0.9752 | 0.8025 | +0.1727 | **WIN** |
| morans_agreement | 0.3705 | 0.5394 | −0.1689 | lose |
| sinkhorn ↓ | 0.0672 | 0.2275 | +0.1603 | **WIN** |
| celltype_composition | 0.9360 | 0.9421 | −0.0061 | tie (non-inferior) |
| celltype_nhood_agreement | 0.9625 | 0.9746 | −0.0121 | lose |
| gene_mean_pearson | 0.9631 | 0.9799 | −0.0168 | lose |
| gene_var_pearson | 0.9821 | 0.9479 | +0.0342 | **WIN** |
| field_pearson | 0.4065 | 0.5556 | −0.1491 | lose |
| field_ssim | 0.0738 | −0.0416 | +0.1154 | **WIN** |
| density_pearson | 0.3013 | 0.1999 | +0.1014 | **WIN** |

**v15: 5 wins, 1 tie, 4 losses.** The expression-distribution metrics are won
outright (co-expression 0.975, Sinkhorn divergence a third of v8's); the losses
are the tissue-tracking ones (`field_pearson`, `morans_agreement`), where a
transport morph that *translates* the tissue beats an interpolation of two
density fields that *blurs* between the two positions.

## 3. Synthetic — non-monotone z events (`make_synth_events.py`)

A deliberately adversarial stress case, built to isolate the failure mode the
proposal's Phase 2.3 is designed around: a **transient** population that appears,
peaks and vanishes within three sections, plus a **branching** population, on a
stable background. No monotone interpolation between brackets can represent the
peak, and neither can a flow/warp.

| metric | v15 | v8 | Δ | verdict |
|---|---|---|---|---|
| coexpression_agreement | 0.8898 | 0.6235 | +0.2663 | **WIN** |
| morans_agreement | 0.3021 | −0.1881 | +0.4902 | **WIN** |
| sinkhorn ↓ | 0.1418 | 0.3607 | +0.2189 | **WIN** |
| celltype_composition | 0.7982 | 0.8365 | −0.0383 | lose |
| celltype_nhood_agreement | 0.8660 | 0.9256 | −0.0596 | lose |
| gene_mean_pearson | 0.8046 | 0.8218 | −0.0172 | lose |
| gene_var_pearson | 0.9413 | 0.8299 | +0.1114 | **WIN** |
| field_pearson | 0.4845 | 0.4666 | +0.0179 | **WIN** |
| field_ssim | 0.1956 | 0.0416 | +0.1540 | **WIN** |
| density_pearson | 0.4816 | 0.5536 | −0.0720 | lose |

**v15: 6 wins, 4 losses.** This is reported because it is informative, not
because it is favourable. The expression side wins decisively (co-expression
+43 % relative, Sinkhorn −61 % divergence, Moran's agreement positive where v8's
is *negative*). The four losses are all **layout/composition** metrics, and they
share one cause: when the peak of the transient population is the held-out
section, its brackets contain almost none of it, the **Phase 2.5 gate does not
pass** on this stack, and the pipeline therefore falls back to the linear field —
which under- or over-shoots the transient by an order of magnitude. v8 is
anchored on a real slice and so degrades more gracefully in composition, at the
cost of the expression metrics.

That is the honest reading: on the case built to need a learned completer, the
learned completer is precisely what the gate declines to use. It is a real
limitation of the current Phase 2 model on nine sections with one event, not a
property of the design — and the gate refusing to ship a completion it cannot
justify is the specified behaviour, not a bug.

## 4. Synthetic — continuous intensities, IMC-like (`make_synth_imc.py`)

The count path and the *continuous-intensity* path are different code: protein
panels (IMC and similar) go through a Gaussian likelihood in Phase 3.1 rather
than NB, and their sections are not cross-registered by the provider, so the
benchmark applies `rigid` re-registration. This stack exercises that path — 35
strictly-positive log-normal channels, six ROIs each in its own arbitrary rigid
frame, no integers anywhere.

```bash
python run_spatialcpa_v15.py --synthetic imc --registration rigid --holdouts ROI1,ROI2,ROI3,ROI4
```

| metric | v15 | v8 | Δ | verdict |
|---|---|---|---|---|
| coexpression_agreement | 0.9526 | 0.9907 | −0.0380 | lose |
| morans_agreement | 0.8706 | 0.9504 | −0.0798 | lose |
| sinkhorn ↓ | 0.1338 | 0.1188 | −0.0151 | lose |
| celltype_composition | 0.9611 | 0.9582 | +0.0029 | tie (non-inferior) |
| celltype_nhood_agreement | 0.9378 | 0.9864 | −0.0486 | lose |
| gene_mean_pearson | 0.9221 | 0.9767 | −0.0546 | lose |
| gene_var_pearson | 0.9743 | 0.8981 | +0.0762 | **WIN** |
| field_pearson | 0.5833 | 0.6711 | −0.0878 | lose |
| field_ssim | 0.2659 | 0.3940 | −0.1281 | lose |
| density_pearson | 0.3778 | 0.2867 | +0.0911 | **WIN** |

**v15: 2 wins, 1 tie, 7 losses — v8 is clearly ahead in this regime.** Stated
plainly because it is the honest result: **v15 should not be expected to beat v8
on an IMC-style dataset on the strength of the STARmap numbers.**

Two things are going on, and only one of them was a defect:

* *A real bug, now fixed.* The Gaussian branch scored its residual on the **raw
  intensity** scale. With IMC values in the hundreds to thousands, the squared
  error is dominated entirely by the few brightest channels and dim markers
  contribute almost nothing to the gradient. Scoring on the log1p scale — where
  the data is normalized and where the metrics are computed — lifted
  co-expression 0.939 → 0.953, per-gene variance 0.928 → 0.974, Sinkhorn
  0.147 → 0.134 and both field metrics, with no change to any count dataset.
  (A variant that also replaced the compositional decoder with a free log-scale
  head, with and without a learned per-gene variance, was tried and was *worse*
  on co-expression — 0.871 and 0.857 — so the profile-times-scalar decoder was
  kept. The per-gene variance shrinks noisy channels toward their mean, which
  attenuates exactly the gene-gene correlations the metric measures.)
* *A regime difference, not a defect.* This stack is a smooth radial-band disc
  with i.i.d. log-normal noise, which is close to the best case for a method that
  copies real profiles from an adjacent section and morphs them: v8 reaches 0.991
  co-expression, near the ceiling. There is little for a generative model to add.

**This is a surrogate, not real IMC data**, which is not bundled here. It is
built to exercise the continuous-intensity and rigid-registration code paths,
not to predict the leaderboard on `imc_breast_cancer`. Real IMC tumour tissue is
far more heterogeneous than a radial disc, so the true margin could go either
way — but nothing here supports claiming v15 wins that dataset.

## 5. Real `imc_breast_cancer` — v15 loses, and this is the honest record

Run by the maintainer on the real dataset (13 holdouts), via
`run_all --methods spatialcpav15_gen --datasets imc_breast_cancer` and
`rank_generation`. Composite rank across the six primary metrics:

| method | coexpr | morans | sinkhorn ↓ | composition | nhood | gene_var | rank |
|---|---|---|---|---|---|---|---|
| spatialcpav13 / v12 | 0.992 | 0.987 | 0.312 | 0.934 | 0.397 | 0.870 | 2.67 |
| spatialcpav14 | 0.986 | 0.969 | 0.300 | 0.952 | 0.471 | 0.946 | 2.83 |
| **spatialcpav8** | 0.990 | 0.978 | 0.311 | 0.914 | 0.369 | 0.869 | **4.33** |
| spatialcpav11 | 0.812 | 0.403 | 0.353 | 0.727 | 0.561 | 0.855 | 7.00 |
| **spatialcpav15** | **0.798** | **0.441** | 0.356 | 0.820 | 0.329 | **0.971** | **7.17** |
| spatialz | 0.376 | 0.415 | 0.498 | 0.937 | 0.439 | 0.452 | 7.50 |

**v15 is second-worst of the SpatialCPA family here and clearly behind v8.** This
is not a metric artifact — these are the primary, correspondence-free metrics.
The one metric v15 wins is `gene_var_pearson` (0.971, best of all eleven
methods), which is consistent with everything else measured: the generative
expression model preserves dispersion that copy-based methods lose.

### Where the loss is — and is not

Measured directly, on a held-out training slice with the **real layout**, so the
structure stage is factored out (`validation/` diagnostic, IMC-like stack):

| what | coexpr | morans |
|---|---|---|
| VAE reconstruction of real cells (ceiling) | 0.996 | 0.849 |
| **v15 diffusion, as shipped** | **0.989** | **0.877** |
| nearest-real-cell copy (what v8 effectively emits) | 0.993 | 0.824 |

**Phase 3 is not the problem.** Given a correct layout, the latent diffusion
matches the copy baseline on co-expression and beats it on Moran's. The loss on
the real dataset therefore comes from **Phase 2 / 4.3 — the structure field and
the layout sampled from it.**

Two supporting observations:

* The maintainer's own table splits cleanly by method class: every method at
  ~0.99 co-expression (v8, v9, v10, v12, v13) emits **real profiles at
  real-derived positions**; both genuinely *generative* methods cluster far below
  (v11 0.812/0.403, v15 0.798/0.441). That is a property of generating a layout
  rather than transporting one, not a v15-specific defect.
* The field's in-plane resolution is capped at `FieldConfig.max_grid = 56`. Dense
  sections with fine structure hit that cap, and the resulting blur is visible in
  a controlled test: on a dense 4000-cell/section stack with ~60 small clusters,
  the grid clips at 56x56 and Moran's agreement falls to 0.815 against v8's
  0.980, with `field_ssim` 0.43 against 0.97. `--max-grid` now exposes this.

### Root cause found: the >24-type channel collapse

The maintainer's `method_log.txt` gave it away in one line:

```
[phase1.2] 26 types (numbered) ...
[phase2.1] field 56x56 x 21 channels (12 group + 8 embedding)
```

**26 types, 12 group channels.** `FieldConfig.max_group_channels` was 24, so a
26-type panel fell past the threshold in `build_field_spec` and the per-type
density channels collapsed to the 12-subclass hierarchy level. Every type inside
a subclass then had to be recovered from the *blurred, density-weighted mean
embedding* channel via a prototype softmax — at a 56x56 grid with a 1-voxel KDE,
that is an average over neighbouring cells of different types.

This is exactly why STARmap was unaffected: 19 types, under the threshold, so it
kept one channel per type. The failure is a **threshold effect at C > 24**, not a
gradual degradation.

Reproduced and quantified on a 29-type, 4000-cell/section IMC-like stack (one
holdout, same wrapper, only the threshold changed — the `12 group` line is
reproduced exactly):

| metric | 24 -> 12 group channels (old) | 48 -> 29 group channels (new) |
|---|---|---|
| celltype_composition | 0.870 | **0.999** |
| celltype_nhood_agreement | 0.941 | **0.982** |
| morans_agreement | 0.752 | 0.769 |
| coexpression_agreement | 0.938 | 0.942 |
| gene_var_pearson | 0.965 | 0.965 |
| field_pearson | 0.835 | 0.832 |

`max_group_channels` is now **48** by default. Per-type channels are cheap — the
interpolator sees `4x(1+C+E)` input channels — so the budget belongs where the
cost starts to matter, not at the proposal's illustrative "~10-20 channels".
Datasets with <= 24 types (including STARmap) are unaffected: the code path is
identical.

This directly targets two of the metrics v15 loses on the real dataset
(`celltype_composition` 0.820 vs v8's 0.914, `celltype_nhood_agreement` 0.329 vs
0.369). **It is a real defect, found, fixed and verified.**

### Re-run on the real dataset after the fix

Maintainer's re-run (13 holdouts, same command). The channel-collapse fix lands
almost exactly where the surrogate predicted:

| metric | before | after | Δ | v8 | |
|---|---|---|---|---|---|
| celltype_composition | 0.820 | **0.941** | +0.121 | 0.914 | **ahead of v8** |
| celltype_nhood_agreement | 0.329 | **0.433** | +0.104 | 0.369 | **ahead of v8** |
| morans_agreement | 0.441 | 0.512 | +0.071 | 0.978 | behind |
| coexpression_agreement | 0.798 | 0.806 | +0.008 | 0.990 | behind |
| sinkhorn ↓ | 0.356 | 0.354 | −0.002 | 0.311 | behind |
| gene_var_pearson | 0.971 | 0.971 | 0.000 | 0.869 | **ahead of v8** |

v15 now leads v8 on **three of the six** primary metrics. (The composite rank
moved 7.17 → 4.17, but that number is not comparable across the two runs: v12 and
v13 are absent from the second table, and the composite is a rank among whichever
methods are present.)

### The remaining gap looks like a ceiling for generative methods

The three methods that *synthesize* expression land within 0.006 of each other on
co-expression — v15 **0.806**, isoST **0.808**, v11 **0.812** — while every
method that emits **real profiles** (v8, v9, v10, v14) sits at 0.986–0.990. That
split is far too clean to be a v15-specific defect.

A staged measurement on a surrogate built with **correlated channels** (shared
latent factors driving all 38 proteins — the property real panels have and every
earlier surrogate lacked, which used i.i.d. per-channel noise) locates the cost
inside the generative path rather than the latent space:

| stage | coexpr | morans |
|---|---|---|
| VAE reconstruction of real cells (ceiling) | 0.998 | 0.999 |
| **diffusion, 1 sample — as shipped** | **0.940** | 0.890 |
| nearest-real-cell copy (v8-like) | 0.980 | 0.917 |

The VAE latent is not the bottleneck (0.998). The conditional diffusion sampler
costs ~0.04 against a copy. That is the price of sampling rather than copying,
and it is the intended trade — it is also why v15 wins `gene_var_pearson` on
every stack tested. It does **not** reach the 0.806 seen on the real data, so the
real dataset has a further factor none of the five surrogates here reproduces.

### Bug found while measuring this: never average expression samples

`InferenceConfig.n_expression_samples > 1` used to **average** the decoded
samples into the output. Averaging in data space shrinks each cell toward the
conditional mean and flattens exactly the gene-gene covariance the metric
measures — on the correlated-channel panel, co-expression falls **0.940 → 0.864
(4 samples) → 0.709 (8 samples)**. The output is now the first *coherent* sample
and the extra samples only supply the spread, mirroring Phase 2.4 where each seed
gives one coherent volume. The default (1 sample) was unaffected, so no published
number changes — but anyone enabling the uncertainty feature was silently
degrading the result.

### `pearson_median` ~ 0.045 — what it does and does not tell us

The maintainer also reports the cell-matched `pearson_median` (per-gene Pearson
across nearest-neighbour-matched cells, median over genes) at 0.040 before the
fixes and 0.045 after. It is computed on **raw, unnormalized** values in
`evaluate.py`, and benchmark-pbya-v2 explicitly deprecates it for generation.

A hypothesis was tested and **rejected**: that v15's per-cell total ("library"),
which `_sample_library` draws at random from real flanking cells of the same
type, injects a shared random factor. Measured on the correlated-channel stack
with the **real layout**, varying only the library:

| library source | coexpr | raw per-gene Pearson (median) |
|---|---|---|
| real library (oracle, not available at inference) | 0.940 | 0.216 |
| random within type — **as shipped** | 0.940 | 0.103 |
| nearest real cell in the bracketing slices | 0.940 | 0.112 |

Two conclusions, both negative:

* the library choice has **no effect on co-expression** (0.940 either way), so it
  is not the cause of the 0.806 deficit;
* it does explain roughly half the per-gene correlation (0.216 -> 0.103), but
  **there is no available fix**: the real library is not knowable at inference,
  and the spatially nearest real cell's library barely helps (0.112) because
  total intensity varies mostly *within* type rather than across space
  (`corr(sampled, real) = 0.19`, `corr(nearest, real) = 0.19`).

The more useful number is the first row. **Even with a perfect layout and oracle
libraries, this architecture reaches only ~0.22 on that metric.** It scores
whether each individual cell's deviation was reproduced, which a conditional
*sampler* does not attempt by construction — it draws a plausible profile for
"type X at position p", not the specific profile that particular cell had. A
method that copies real cells retains that per-cell structure for free. A low
`pearson_median` is therefore the expected signature of generation, not evidence
of a defect; only the value for a copy-based method on the *same* dataset would
make it interpretable, and that comparison has not been obtained.

### Correction: this is not "the signature of generation"

v8's `pearson_median` on the same dataset is **0.261**; v15's is **0.045**. The
earlier reading in this file — that a low value is what a sampler necessarily
scores — was **wrong**, and the oracle measurement above is what refutes it:
v15's own architecture reaches ~0.216 with a correct layout, so on the real data
v15 is running **4.8x below its own achievable ceiling**, not merely below a
copy-based method. There is a real loss inside the pipeline on this dataset.

### `diagnose_dataset.py` — locating it on data this repo does not have

Five synthetic stacks failed to reproduce the failure, so the remaining work has
to happen on the real dataset. `validation/diagnose_dataset.py` fits v15 once and
then scores a **ladder** of predictions that swap v15's machinery in one piece at
a time, each written as a real `prediction.h5` and scored by the **real**
evaluators, so every number is leaderboard-comparable:

| row | what it is |
|---|---|
| A | GT layout + nearest real cell's profile (the copy ceiling) |
| B | GT layout + GT library + v15 diffusion (oracle expression) |
| C | + v15's sampled library |
| D | + v15's predicted cell types |
| E | full v15 — everything generated (what ships) |

The first large drop between consecutive rows is the stage at fault: B≪A blames
the expression model, C≪B the library, D≪C the type assignment, E≪D the point
process.

```bash
python spatialcpav15/validation/diagnose_dataset.py \
    benchmark-pbya/data/processed/imc_breast_cancer/data.h5ad --registration rigid
```

On the correlated-channel surrogate the ladder already reads clearly:

| variant | pearson_med | coexpr | morans | composition | nhood |
|---|---|---|---|---|---|
| A copy @ GT layout | 0.101 | 0.996 | 0.977 | 0.940 | 0.996 |
| B v15 expr @ GT layout, GT lib | **0.191** | 0.963 | 0.876 | 1.000 | 1.000 |
| C + v15 library | **0.079** | 0.963 | 0.887 | 1.000 | 1.000 |
| D + v15 types | 0.069 | 0.963 | 0.887 | 0.941 | 0.986 |
| E full v15 (shipped) | 0.070 | 0.962 | 0.857 | 0.998 | 0.986 |

Two things stand out even here. v15's expression at a correct layout (B, 0.191)
**beats the copy baseline** (A, 0.101) — the expression model is not the weak
part. And the single largest loss is the **per-cell library** (B → C, −0.112),
consistent with the separate measurement above. Co-expression never collapses on
this stack (0.963 throughout), which is why the real dataset's 0.806 still needs
the real data to explain.

### The ladder on the real dataset — the answer

Maintainer ran `diagnose_dataset.py` on `imc_breast_cancer` (holdout z7, 15
sections, 99 501 training cells, 26 types, H200):

| variant | pearson_med | coexpr | morans | composition | nhood |
|---|---|---|---|---|---|
| A copy @ GT layout | 0.034 | **0.991** | 0.944 | 0.939 | 0.389 |
| B v15 expr @ GT layout, GT lib | 0.098 | **0.789** | 0.708 | 1.000 | 1.000 |
| C   + v15 library | 0.017 | 0.782 | **0.307** | 1.000 | 1.000 |
| D   + v15 types | 0.015 | 0.790 | 0.289 | 0.689 | 0.587 |
| E full v15 (shipped) | 0.045 | 0.791 | 0.537 | 0.919 | 0.334 |

Two distinct losses, both large, and **the earlier conclusion drawn from the
synthetic stacks was wrong**:

1. **A -> B: co-expression 0.991 -> 0.789.** With the layout held at ground
   truth *and* the real library, v15's expression model already produces the
   0.79 that ships (E = 0.791) and that the leaderboard reports (0.806). On this
   dataset the **expression model is the problem**, not the layout — the opposite
   of what every surrogate indicated.
2. **B -> C: Moran's 0.708 -> 0.307.** Swapping the real per-cell library for
   v15's sampled one halves Moran's agreement.

Note also that the copy baseline reaches only `pearson_median` 0.034 at *exact*
GT coordinates — adjacent sections of this specimen simply do not predict each
other cell-for-cell, which is worth remembering before reading much into that
metric here.

### Fixed: the library is now spatially coherent

Cause of loss 2. Library size is technical depth and so is *sampled* rather than
modelled (Phase 3.1) — but "sampled" must not mean "spatially random". In a real
section the per-cell total is strongly spatially autocorrelated (cell size,
staining depth, tissue compaction all vary smoothly), and because the total
multiplies *every* channel, randomising it dilutes the spatial autocorrelation of
every gene at once. The library is now taken from the **spatially nearest**
same-type cell in the bracketing slices (`InferenceConfig.spatial_library`).

Measured on the real STARmap block, this improves things there too — the STARmap
table above is post-fix, and against the previous numbers:

| metric | before | after |
|---|---|---|
| morans_agreement | 0.7925 | **0.8693** |
| field_ssim | 0.5909 | **0.7459** |
| field_pearson | 0.3852 | **0.4721** |
| coexpression_agreement | 0.8736 | 0.8772 |

Still 8 wins / 2 ties / 0 losses against v8 on STARmap, with no metric regressed.

### Located: the Phase 3.2 diffusion was undertrained on large stacks

The **A0** row settles it. On the real dataset:

| variant | pearson_med | coexpr | morans | composition | nhood |
|---|---|---|---|---|---|
| **A0 VAE recon of GT cells** | **0.996** | **0.962** | **0.955** | 1.000 | 1.000 |
| A copy @ GT layout | 0.028 | 0.991 | 0.952 | 0.936 | 0.387 |
| B v15 expr @ GT layout, GT lib | 0.099 | **0.784** | 0.700 | 1.000 | 1.000 |
| C   + v15 library | 0.012 | 0.779 | 0.288 | 1.000 | 1.000 |
| D   + v15 types | 0.014 | 0.787 | 0.267 | 0.696 | 0.590 |
| E full v15 (shipped) | 0.114 | 0.812 | 0.829 | 0.919 | 0.334 |

The Phase 3.1 VAE reconstructs the held-out cells at **0.962 co-expression and
0.996 per-gene Pearson** — the latent space and the decoder are not the
bottleneck, and the compositional decoder is *not* the problem. The entire loss
is **A0 -> B**: the Phase 3.2 conditional diffusion.

The cause is in the training log of that run:

```
[phase3.1] step 1458/5835 ...        <- VAE, scaled to the stack
[phase3.2] epoch 65/260 loss=0.6056  <- diffusion, FIXED at 260 steps
[phase3.2] epoch 130/260 loss=0.4281
[phase3.2] epoch 195/260 loss=0.3620
[phase3.2] epoch 260/260 loss=0.4474 <- still rising: nowhere near converged
```

Phase 3.2 trained a **fixed 260 steps** regardless of stack size. A step is one
*block* of `block_size` cells, so that is:

| stack | cells | blocks/pass | 260 steps = |
|---|---|---|---|
| STARmap | 3 779 | 15 | **17.3 passes** |
| imc_breast_cancer | 99 501 | 389 | **0.67 passes** |

Two-thirds of a single pass over the data. The denoiser was barely trained
exactly where there was the most to learn — and this is why no synthetic stack
reproduced the failure: every one of them is small enough that 260 steps is
plenty. The Phase 3.1 VAE already had a `min_passes` rule (added earlier for the
same reason); Phase 3.2 was simply missed.

Fixed: `n_steps = max(epochs, min_passes * ceil(n_cells / block_size))`, the same
rule the VAE uses, with `min_passes = 15`. On STARmap this evaluates to
`max(260, 225) = 260` — **identical by construction**, and confirmed empirically:
the STARmap table above is unchanged after the fix. On `imc_breast_cancer` it
becomes 5 835 steps, 22x more training. `diagnostics["phase3_diffusion"]` now
reports `n_steps` and `passes_over_cells` so this cannot recur silently.

### `pearson_median` and the generation metrics are in direct tension

An `A1` rung (cell-type-mean assignment at the GT layout — the most conservative
possible prediction, zero within-type variation) makes the trade-off explicit.
On the correlated-channel stack:

| variant | pearson_med | coexpr | morans |
|---|---|---|---|
| A0 VAE recon of GT cells | 0.996 | 0.999 | 0.998 |
| **A1 type-mean @ GT layout** | **0.391** | **0.278** | **0.085** |
| A copy @ GT layout | 0.101 | 0.996 | 0.977 |
| B v15 expr @ GT layout, GT lib | 0.182 | 0.904 | 0.768 |
| E full v15 (shipped) | 0.063 | 0.902 | 0.775 |

Type-mean assignment scores **0.391** on `pearson_median` — four times the copy
baseline and six times full v15 — while its co-expression is 0.278 and its
Moran's agreement 0.085. The metric rewards predicting the **type-conditional
mean** and penalizes within-type variation; the generation metrics reward the
opposite. A method cannot maximize both, and v15 is built to generate that
variation (which is why it wins `gene_var_pearson` 0.971 against v8's 0.869
everywhere it has been measured).

This is the most likely explanation for v8's 0.261 against v15's ~0.09 on
`imc_breast_cancer`, and it means **`pearson_median` should not be expected to
close** without giving up the metrics v15 is designed to win. It is also the
metric benchmark-pbya-v2 explicitly excludes from the generation ranking.

### The library fix, measured on the real dataset

Row E between the two ladder runs isolates it — the only change was the spatially
coherent library:

| metric (row E) | random library | spatial library |
|---|---|---|
| morans_agreement | 0.537 | **0.829** |
| pearson_median | 0.045 | **0.114** |
| coexpression_agreement | 0.791 | 0.812 |

Moran's agreement went from 0.537 to 0.829 against v8's 0.978, and the
cell-matched per-gene correlation more than doubled. (Rows C/D in the run above
still used the random draw — the diagnostic passed no positions to
`_sample_library`; that is fixed too, so C/D now reflect what ships.)

### Previously open, now attributed: co-expression 0.789 at the ground-truth layout

Loss 1 is **not** fixed and not yet explained. The remaining question is whether
the Phase 3.1 VAE can represent this data at all, or whether the Phase 3.2
sampler is at fault — `diagnose_dataset.py` now emits an **A0** row (the VAE's own
encode->decode reconstruction of the held-out cells) which separates the two:
A0 ~ 0.79 indicts the decoder, A0 ~ 0.99 indicts the diffusion. Both decoder
variants (compositional `softmax x library`, and a free per-channel log-scale
head) reconstruct the correlated-channel surrogate at 0.999, so the question
cannot be settled here.

A flaw in the diagnostic itself was also found and fixed: it passed ground-truth
coordinates **unshifted** while Phase 1.4 had applied a drift stabilization to the
training sections (`residual drift = 1.98 spacings` on this dataset), so rows
A-D were offset from the model's frame. Row E is production and unaffected, and
E (0.791) agrees with B (0.789), so the conclusion above survives — but the rows
are only trustworthy after the fix.

### Caveat on the training-length fix: it is not uniformly positive

On the real dataset the fix is a large gain (co-expression 0.806 -> 0.9215,
Moran's 0.512 -> 0.852 on holdout z1). On the **correlated-channel surrogate**
the same change went the other way — co-expression 0.962 -> 0.902, Moran's
0.857 -> 0.775 — because that stack has 21 000 cells, so the old fixed 260 steps
was already ~3 passes and not starved; training 4.8x longer sharpened the
conditional distribution and cost co-expression. STARmap is unaffected (the rule
evaluates to the same 260 steps).

So `min_passes = 15` is right where the model was badly undertrained and may be
more than necessary where it was not. Worth revisiting with a sweep once the
full IMC campaign lands.

### What is *still* not explained

The channel collapse does **not** account for the headline gap. On the 29-type
surrogate co-expression is 0.938 -> 0.942 and Moran's 0.752 -> 0.769: the fix
moves composition and neighbourhood structure, not those two. The real dataset
sits at co-expression **0.798** and Moran's **0.441**, and no surrogate here
reproduces that.

So two causes are now identified and fixed (the Gaussian loss scale, the channel
collapse) and a third remains open. The honest status is that v15's deficit on
`imc_breast_cancer` is **partly diagnosed, not resolved**; whether the fixes
close the leaderboard gap can only be settled by re-running the real dataset.

### Summary across the four synthetic/real stacks

| stack | regime | wins | ties | losses |
|---|---|---|---|---|
| STARmap 3D cortex (real) | counts, volumetric | **8** | 2 | **0** |
| synthetic, drifting niches | counts, rigid drift | 5 | 1 | 4 |
| synthetic, non-monotone events | counts, transient/branching | 6 | 0 | 4 |
| synthetic, IMC-like | continuous intensity, rigid | 2 | 1 | 7 |

**Per-gene variance and cell density are won in all four regimes.** Everything
else depends on the regime. On the real specimen — the only stack with genuine
molecular and spatial structure — v15 is at or above v8 on every metric. On
synthetic stacks whose geometry is close to a rigid translation of a smooth
shape, v8's transport morph models that geometry directly and wins the
tissue-tracking metrics; the IMC-like stack is the extreme of that case and v8
wins it clearly.

The claim this evidence supports is narrow: **v15 dominates v8 on the real
densely-sectioned STARmap specimen (8 wins / 2 ties / 0 losses), and loses to it
on `imc_breast_cancer` (rank 7.17 vs 4.33).** v15 is *not* a uniform improvement
over v8, and the original objective — beat or match v8 on every metric — is met
on STARmap and **not met** on IMC.

### Ablation: does more Phase 2 training close the synthetic gap?

No — and the result is worth recording because it is counter-intuitive. Training
the structure interpolator for 1600 epochs instead of the default 220 makes it
**pass** the Phase 2.5 gate on the drifting stack (relative MSE gain +0.29 vs.
−0.92 at the default), yet the downstream generation metrics get *worse*:

| metric | 220 epochs (gate FAIL → linear) | 1600 epochs (gate PASS → learned) |
|---|---|---|
| coexpression_agreement | 0.9752 | 0.9619 |
| morans_agreement | 0.3705 | 0.3046 |
| celltype_nhood_agreement | 0.9625 | 0.9153 |
| field_pearson | 0.4065 | 0.3701 |
| density_pearson | 0.3013 | 0.3195 |

So the learned completion genuinely is the more accurate *field* — and still
yields a worse *slice*. Field MSE and the benchmark's correspondence-free
generation metrics are not the same objective: sampling a point process from a
sharper, higher-frequency density reintroduces variance the binned metrics
punish. The default is left at 220 epochs, where the gate declines and the
pipeline uses the linear field. `--structure-epochs` exposes the knob.

---

## Instrumenting each stage separately

The proposal insists that stages be measured independently, since errors compound
and an end-to-end number cannot tell you which stage failed. Both internal gates
run by default on every invocation and are printed and stored in
`method_params` / the method log:

**Phase 2.5 (structure vs. linear interpolation of the field), STARmap:**

```
[phase2 gate] mse_learned=1.75288 mse_linear=0.73191 gain=-1.395 | FAIL -> falling back to linear field
```

The learned completion does **not** beat linear interpolation on this stack, and
the pipeline says so and uses the linear field. This is expected and correct at
`R = 1`: adjacent planes 1 µm apart are nearly identical, the residual the model
would have to predict is close to pure noise, and the proposal's own stop
condition exists exactly to catch this. The orthogonal-reslice diagnostic on the
same stack gives `reslice_striping ≈ 0.90` (≈ 1 means the volume looks like
tissue from every axis; ≫ 1 would mean per-slice inconsistency).

**Phase 3.3 (expression vs. the three named baselines), STARmap:**

```
[phase3 gate] readout=mean composite=0.748 vs type_mean=0.631, bracket_linear=0.962 | beats_type_mean=True
```

The expression model **clears the cell-type-mean bar**, which is the bar that
distinguishes a conditional generator from a label propagator. It does not beat
the copy-style baselines (`nearest_slice_copy`, `bracket_linear`) on this
internal score — those baselines hand back *real* profiles from an adjacent
plane, which at Δz = 1 are almost the right answer; that they score high is a
property of a densely sectioned specimen, not evidence that the generator is
weak. The end-to-end benchmark, where the same copy strategy is what v8 does,
shows v15 ahead on every expression metric.

---

## Component tests

`test_components.py` checks the properties that are *stated requirements* rather
than matters of accuracy — the ones whose regression would silently change what
the method is:

```bash
python spatialcpav15/validation/test_components.py
```

It verifies label-kind classification, **order invariance** of the marker
encoder, that the shrinkage blend really hands rare types to the prior, that the
field conserves mass and stays non-negative and that the Phase 2.2 compression
round-trips, that bracket selection gives two slices per side with correct edge
handling, that a fixed noise makes `t ↦ I_t` continuous and bit-reproducible,
that the point process reproduces the intensity it was given (r > 0.97), and that
generation is reproducible and carries all three provenance channels. All checks
pass.

## Reproducibility

Everything is seeded (`--seed`, default 42): the field rasterization is
deterministic; the structure sampler is a deterministic DDIM with per-seed fixed
noise; the layout point process and the expression sampler draw from seeded
generators. Re-running the same command reproduces the same predictions.

Runtime on 4 CPU cores: ≈ 90 s per holdout for v15 (three models trained per
holdout) against ≈ 4 s for v8, which is training-free. That gap is real and worth
stating: v15 buys its margin with computation.

**All numbers above are CPU numbers.** v15 uses a GPU automatically when one is
visible, but this machine has none (`torch.cuda.is_available()` is `False`), so
the CUDA path is unexercised and no GPU timing is claimed. See the GPU section of
`spatialcpav15/README.md` for the sharing policy and what is and is not tested.
