# benchmark-pbya-v2 — leakage-hardened virtual-slice benchmark

A variant of `benchmark-pbya` that runs the 3D virtual-slice-generation benchmark
on **all datasets with no potential data leakage from the held-out slice**. It
does not modify `benchmark-pbya`; the two coexist.

The processed datasets and downloaded method tools are **shared** with v1 (the
data-processing pipeline is non-leaky and unchanged), so v2 only re-implements
the *benchmarking* layer.

---

## Why v2

Three ways the held-out ("leave-one-out") slice could leak into what a method
sees were identified in v1 (see the discussion in the repo history). v2 closes
all three, centrally, in `src/benchmark/leakage_guard.py`:

| # | Leakage vector | v1 behavior | v2 fix |
|---|----------------|-------------|--------|
| 1 | **Held-out geometry** — methods were handed the held-out cells' real `(x, y)` as query points | coordinate-matched methods predict *at* the answer's coordinates | **Generation-only**: methods receive only a scalar target *z*; they synthesize the slice. Cell count is emergent. |
| 2 | **Upstream registration** — several datasets ship coordinates globally registered by the provider using *all* sections, including the held-out one | inherited as-is | **Per-holdout re-registration**: the training slices are re-registered into a common frame **without** the held-out slice, replacing any upstream registration that used it. |
| 3 | **Global statistics** — label vocabularies / clustering built over all cells incl. holdout | Leiden/vocab over all cells | **Train-only** label vocabulary + clustering; per-cell (leakage-safe) expression normalization. |

Membership is enforced too: `split_holdout` + `assert_no_leakage` +
`guard_no_holdout` guarantee no held-out cell ever reaches a method — the method
input file physically excludes it.

---

## How it works (per method × dataset × holdout)

`src/benchmark/run_benchmark.py` orchestrates each run so a wrapper *cannot* see
the held-out slice:

```
full data.h5ad
     │  split_holdout()                       ← held-out kept aside (eval only)
     ▼
training slices ──► reregister_training()      ← TRAINING-ONLY common frame
     │                                            (rigid ICP / PASTE / none)
     ▼
train_registered.h5ad  +  target z (scalar)    ← the ONLY things a method gets
     │  conda run … run_<method>.py --input … --target-section S --target-z Z
     ▼
method synthesizes a virtual slice at z  ──►  prediction.h5
     │  evaluate() with rigid prediction→GT alignment
     ▼
metrics.json
```

The `train_registered.h5ad` is built **once per holdout** and reused across
methods (fair comparison), cached under `results/_v2_inputs/`.

### Evaluation methodology (generation ≠ coordinate-matched)

De-novo generation does **not** produce a cell-to-cell correspondence with the
GT slice — cells are synthesized, not placed on GT cells. So the v1 primary
metric (`pearson_median`: correlate expression across nearest-neighbor-matched
cells) is the **wrong measurement** for generation: it manufactures a
correspondence that doesn't exist, is acutely sensitive to sub-cell alignment and
grid-vs-real density, and on near-symmetric tissue collapses toward/below zero
even for a faithful generation. (This is why generation-mode `pearson_median`
comes out negative.)

v2 therefore evaluates the generated slice as a **field/distribution**, not a
cell list, in `evaluate_generation.py`:

**Primary — correspondence-free (and mostly alignment-free):**
- `gen_coexpression_agreement` — Pearson r between the gene-gene correlation
  matrices of prediction and GT. Alignment-free. *(Validated: faithful +0.97,
  scrambled +0.04, random −0.01; identical at 0° and 180°.)*
- `gen_morans_agreement` — Pearson r between per-gene Moran's I of prediction and
  GT (each within its own slice). Alignment-free; tests whether the same genes
  are spatially structured. *(faithful +0.69, scrambled/random ≈0.)*
- `gen_gene_mean_pearson` / `gen_gene_var_pearson` — per-gene mean/variance
  agreement. Alignment-free. Expression is normalized identically for both
  slices first, so scale (log-pred vs raw-GT) doesn't distort them.

**Secondary — alignment-dependent (trustworthy only on asymmetric tissue):**
- `gen_field_pearson` / `gen_field_ssim` — binned spatial-field agreement.
- `gen_density_pearson` — bin-wise cell-density agreement.

For the alignment-dependent metrics, the synthesized cloud is aligned onto GT
with an **orientation-robust** ICP (multiple initial rotations + reflection,
lowest-residual fit — a single PCA-seeded ICP silently flips on round tissue).
Alignment is an evaluation-side operation (uses GT, feeds nothing back to the
method), so it is not leakage.

The v1 **cell-matched metrics are still computed and stored for reference**
(`pearson_median`, `matching_rate`, …) but are explicitly *not* the primary
score for generation.

---

## Per-dataset re-registration policy

Set automatically by `config.registration_for(dataset)`; override per-run with
`--registration {rigid,paste,none}`.

| Category | Datasets | Policy | Rationale |
|---|---|---|---|
| pre-aligned | allen_* (CCF), st_mouse_brain_ortiz (WholeBrain), imc_* (SIFT), arrayseq_kidney (Z_aligned), merfish_hypothalamus | `rigid` | Re-register training-only to **discard** the upstream registration that used the held-out slice. |
| not-aligned | cosmx_nsclc_3d, openst_lymph_node, visium_mouse_brain_cell2location | `rigid` | Slices aren't cross-registered at all; training-only registration makes interpolation well-posed. |
| volumetric | starmap_visual_cortex, deep_starmap, easi_fish_*, exseq_*, merfish_thick_tissue | `none` | Single 3-D imaging block; z-planes already co-registered — re-registering would distort. |

`rigid` uses a dependency-light coordinate ICP (numpy/scipy). `paste` uses the
`paste` package (expression-aware) when installed, else falls back to `rigid`.

---

## Methods

Generation-only. `generation_native` methods already synthesize de novo.

| Method | Status | Notes |
|---|---|---|
| `spatialcpav4_gen` | **available** | Transformer + occupancy head; grid over flanking training-slice bbox. |
| `spatialz` | available | Reuses SpatialZ synthesis unchanged; v2 only swaps the I/O contract. |
| `feast` | available | Reuses FEAST/PASTE2 interpolation; v2 only swaps the I/O contract. |
| `isost` | available | Reuses isoST SDE generation; v2 only swaps the I/O contract. |
| `stvgp` | **disabled** | Coordinate-query GP regressor — needs held-out `(x, y)`; incompatible with generation-only. |
| `spateo_gp` | **disabled** | Coordinate-query SVGP — same reason. |

All wrappers share `methods/_v2_io.py` (CLI contract, `guard_no_holdout`, output
writer).

---

## Verification status (honest)

This environment has no conda envs / external method tools / most datasets, so
the leakage-critical core was verified **in-process** and the external-tool
wrappers were **ported by reusing their existing synthesis unchanged** (only the
I/O contract changed) and need a runtime pass in their conda envs.

**Verified here (numpy/scipy/torch/anndata):**
- Registration math: Umeyama exact recovery; ICP recovers a known rigid
  transform to ~0 residual; eval alignment recovers rotated+scaled+shifted
  predictions onto GT.
- `split_holdout` / `assert_no_leakage` / `guard_no_holdout` catch leaks;
  `reregister_training` preserves z, anchors the first slice, records transforms;
  `build_labels_train_only` excludes holdout-only classes.
- **Full spatialcpav4 generation path end-to-end**: build training-only
  re-registered input → wrapper synthesis → `prediction.h5` (emergent count,
  correct labels) → `evaluate` rigid alignment.

**Needs a runtime pass (unchanged synthesis, new I/O only):** `run_spatialz.py`,
`run_feast.py`, `run_isost.py` in their `bench_*` conda envs on a real dataset.

---

## Usage

```bash
# one configuration
python -m src.benchmark.holdout --input <shared>/data.h5ad --strategy leave_one_out --output holdouts.json
python -m src.benchmark.run_benchmark \
    --method spatialcpav4_gen --dataset cosmx_nsclc_3d \
    --holdout-json one_holdout.json            # registration auto-selected

# override registration policy
python -m src.benchmark.run_benchmark --method feast --dataset allen_merfish_brain \
    --holdout-json one_holdout.json --registration paste

# full campaign
python -m src.benchmark.run_all --methods spatialcpav4_gen spatialz feast isost
python -m src.benchmark.aggregate_results
```

Data location is auto-resolved to the shared v1 tree; override with
`BENCH_V2_DATA` / `BENCH_V2_TOOLS`.
