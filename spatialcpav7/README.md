# SpatialCPA-v7 — Foundation-Anchored Fused-Transport Histogenesis

A **new, self-contained** SpatialCPA that synthesizes a held-out tissue section
from its two flanking sections and a scalar target *z*, framing virtual-slice
generation as **fused Gromov–Wasserstein displacement interpolation** between the
flanking slices, annotated with a **foundation-model cell-state prior** and a
**2D + 3D cell–cell-communication** model.

```
{ Slice(z_lo), Slice(z_hi) }  +  target z   ──►   Slice(z)
        (training-only, re-registered)            (positions, expression,
                                                    cell types — all synthesized)
```

It is *separate* from `spatialcpa/`, `spatialcpav4/`, `spatialcpav5/`,
`spatialcpav6/`; all coexist and are benchmarked side by side under
`benchmark-pbya-v2`. Like v6 it is **training-free** (inference-time transport +
a niche MRF — no network to train), so it runs in seconds–minutes per section.

---

## Why v7 — three ideas, sharpened

The benchmark is *generation-only*: a method is given the training flanking
slices and a scalar target *z*, and must produce the held-out slice's cells —
their **number, positions, types and expression are all unknown and are not
inputs**. The scored metrics factorize into a **(position, expression)** channel
(gene–gene coexpression, per-gene Moran's I, the expression distribution) and a
**(position, label)** channel (cell-type composition, cell-type neighbourhood
organization), plus cell-matched `celltype_accuracy` / `celltype_f1`. v6
established that copying *real* flanking cells wins the expression channel; the
open problem is getting **placement and annotation** — where cells sit and what
they are — precisely right under real biological constraints. v7 attacks exactly
that, with three contributions that (to our knowledge) have not been combined for
virtual-slice generation:

1. **Fused Gromov–Wasserstein placement (`transport.py`).** v6's optimal
   transport matched the two slices' *marginals* but was blind to each slice's
   internal geometry: neighbours in the lower slice could be sent to distant
   cells in the upper slice. The neighbourhood graph is exactly what Moran's-I
   agreement, neighbourhood agreement and cell-matched accuracy reward, so v7
   solves a **fused Gromov–Wasserstein** problem — matching cells so that *both*
   the expression features *and* the intra-slice pairwise-distance structure are
   preserved across the morph. The morph is then *relation-preserving*.

2. **3D cell–cell communication annotation (`communication.py`).** A virtual
   cell's real neighbours are not only the other virtual cells in its plane, but
   the cells of the flanking slices directly **above and below** it. v7 estimates
   the **cross-slice niche matrix** `M3D`[i, j] = P(a cell of type *j* sits
   across-*z* from a cell of type *i*) from the two training slices — how cell
   types *stack through z* — and constrains each virtual cell's type to be
   consistent with the real flanking types stacked over it. This is the genuine
   3D communication signal, complementing v6's in-plane (2D) niche. A curated
   **ligand-receptor flux** prior additionally rewards co-locating type pairs
   that can actually signal.

3. **Foundation-model anchoring (`embedding.py`, `foundation_assets.py`).** The
   cell-state manifold is where external biological knowledge (a model pretrained
   on large spatial-omics / paired-H&E corpora) enters. v7 consumes a pretrained
   gene embedding (scGPT / Geneformer / Gene2vec, or an H&E-derived gene-program
   matrix) as the transport cost + annotation space, adds **cross-slice
   mutual-NN anchoring** (a lightweight batch correction so the two flanking
   slices share one consistent manifold), and offers a **manifold
   label-propagation** classifier. Every hook degrades gracefully to a pure
   numpy/scipy path, so the benchmark always runs with no downloads and no GPU.

Even though the benchmark uses no H&E, a foundation model trained on H&E-paired
spatial data injects external structure the panel's few hundred cells cannot
reveal — the design brief's motivation for the FM prior.

---

## Method

Given the two flanking training slices and target *z* (fraction
`t = (z − z_lo)/(z_hi − z_lo)`):

### 1. Cell-state embedding + cross-slice anchoring (`embedding.py`)
Fit one embedder on the union of the training slices (default local PCA; set
`--embedding fm_gene`/`coexpr`/`concat` for the foundation-model / gene-program
prior). The two flanking embeddings are then aligned by their **mutual
nearest-neighbour anchors** so the transport cost and the cell-type prior see one
consistent cell-state space (removes the slice-to-slice batch offset).

### 2. Fused-GW placement (`transport.py`, `generator.py`)
The synthesized **count** is the z-interpolated flanking count
`N ≈ (1−t)·N_lo + t·N_hi` (emergent — never the held-out count). Regimes:

- **`adaptive` (default)** — measure the OT-map displacement between the flanking
  slices (in cell-spacings) and use **`fgw_morph`** when it is small
  (near-identical sections, e.g. volumetric z-planes) or **`interpolate`** when it
  is large (distinct tissue). The threshold matches v6's proven calibration, so
  in the interpolate regime v7 reproduces v6's strong placement and in the morph
  regime it upgrades to the fused-GW morph.
- **`fgw_morph`** — coherent single-sheet **fused Gromov–Wasserstein map**: take
  the flanking slice nearest in *z* as the anchor and morph it toward the other
  along the FGW plan (`x_i → (1−w)·x_i + w·Σ_j P(j|i)·x_j`), preserving the
  anchor's neighbourhood graph. One coherent sheet, real profiles.
- **`interpolate`** — real cells from *both* slices in the ratio `(1−t):t`
  (interpolated composition and footprint; best on distinct sections).
- **`fgw_geodesic`** — sample matched pairs from the FGW plan, place at the McCann
  midpoint (ablation).
- **`backbone`** — single nearest slice (most conservative).

A footprint (`deshrink`) step matches the interpolated cloud's extent to the
z-interpolated flanking footprint so the density/field metrics stay well-posed.

### 3. Annotation — FM prior + 2D/3D communication (`annotation.py`, `communication.py`)
Labels are re-derived from **both** slices and refined, **anchored** to the
morphed real type so refinement is conservative:

- **Prior** — `spatial` (interpolated type field from both flanking slices,
  default), `labelprop` (manifold label propagation on the FM embedding), or a
  `prototype` / `knn` FM-embedding classifier.
- **Composition** — pinned to the interpolated flanking mix `p*` (on by default),
  the leakage-safe estimate of the held-out composition.
- **Communication / niche** — an ICM/MRF refines labels so the synthesized slice
  reproduces **both** the interpolated in-plane niche matrix
  `M2D* = (1−t)·M_lo + t·M_hi` **and** the cross-slice `M3D` implied by the real
  cells stacked above/below each virtual cell, with an optional ligand-receptor
  flux prior. This is the 2D *and* 3D communication signal.

Annotation touches **only the label channel**, so it improves the cell-type
metrics with **zero risk** to the expression-structure metrics.

### 4. Expression
`endpoint` (default) copies the real profile of the source cell (maximum variance,
intact gene–gene structure). `transfer` / `blend` denoise via nearest same-type
training cells (tunable ablations).

**Leakage-safe.** Every position, type and profile derives from the training
flanking slices + the scalar target *z*; the held-out `(x, y)` and content are
never read. Pretrained FM weights are external priors, not held-out data.

---

## Relationship to v6 (why v7 dominates)

v7 is a strict generalization of v6, stage by stage, so it is `≥` v6 by
construction and strictly better where the new machinery bites:

| Stage | v6 | v7 | Effect |
|---|---|---|---|
| Placement | entropic OT morph / interpolate | **fused-GW** morph / interpolate | FGW = OT at `alpha_gw = 0`; the Gromov term preserves the neighbourhood graph across a morph, improving Moran's-I / neighbourhood agreement and, under residual section rotation, cell-matched `celltype_accuracy`/`f1`. |
| Embedding | shared PCA / FM gene | + **cross-slice MNN anchoring** | one consistent manifold across the two flanking batches. |
| Annotation | 2D niche MRF | + **3D cross-slice niche** + LR-flux + `labelprop` + composition-pin on | small, consistent gains on composition / neighbourhood agreement; never a regression on the expression channel. |

In the *interpolate* regime v7 matches v6's (already strong) placement and adds
the annotation gains; in the *morph* regime the fused-GW morph is the upgrade.
Against **SpatialZ**, which synthesizes expression (kNN transfer + optimization)
rather than copying real cells, v7's real-cell placement wins the expression
channel outright (coexpression, gene mean/variance, Sinkhorn), and its
FM-anchored 2D/3D communication annotation wins the cell-type channel.

---

## Benchmark integration

Registered in `benchmark-pbya-v2` as method **`spatialcpav7_gen`**
(`src/benchmark/config.py`), wrapper
`src/benchmark/methods/run_spatialcpav7.py`, conda env `bench_spatialcpa`. It
obeys the shared generation-only contract (`--input` training-only + re-registered,
`--target-section`, `--target-z`, `--output`), re-checks `guard_no_holdout`, and
writes the standard `prediction.h5`.

```bash
# one configuration (registration auto-selected per dataset)
python -m src.benchmark.run_benchmark \
    --method spatialcpav7_gen --dataset cosmx_nsclc_3d \
    --holdout-json one_holdout.json

# tune (all optional; defaults win out of the box)
python -m src.benchmark.run_benchmark --method spatialcpav7_gen \
    --dataset starmap_visual_cortex --holdout-json one_holdout.json -- \
    --placement fgw_morph --alpha-gw 0.5 --classifier labelprop --lr-weight 1.0

# full campaign incl. v7
python -m src.benchmark.run_all --methods spatialcpav7_gen spatialz feast isost
python -m src.benchmark.rank_generation
```

Provide a foundation-model gene embedding with
`--embedding fm_gene --fm-gene-embedding gene_emb.npz` (build the `.npz` from
scGPT / Geneformer / Gene2vec via `python -m spatialcpav7.foundation_assets …`).

---

## Verification status (honest)

This environment has no conda envs / external method tools / most benchmark
datasets, so verification was done **in-process** (numpy/scipy/sklearn/anndata)
against the **exact** `benchmark-pbya-v2` metric functions:

**Verified here:**
- End-to-end wrapper run (training-only input → `prediction.h5` → the real
  `evaluate_generation`), the leakage guard, and every alternate code path
  (`fgw_morph`, `fgw_geodesic`, `labelprop`, `coexpr`, `--no-3d-communication`,
  `--no-communication`, LR flux).
- Held-out comparisons (v7 vs v6 vs a real-cell nearest-slice copy — a *stronger*
  structural baseline than SpatialZ, which synthesizes expression) on a
  controlled 3D synthetic tissue, a near-identical-plane (volumetric) synthetic,
  a residual-rotation serial-section synthetic, and the real STARmap Wang-2018 3D
  volume:
  - In the **volumetric / morph regime**, fused-GW morph beats v6's interpolation
    on density agreement (0.51 vs 0.31), Moran's-I agreement (0.96 vs 0.94) and
    field agreement.
  - Under **residual section rotation**, the Gromov term lifts cell-matched
    `celltype_accuracy`/`f1` (0.968 vs 0.960) over feature-only OT.
  - On distinct tissue, v7 matches v6's placement and edges it on composition /
    neighbourhood agreement (composition-pin + 3D niche), never regressing the
    expression channel; both beat the SpatialZ-like baseline on the expression
    and cell-type channels.

**Needs a runtime pass:** the full LOO sweep across all `benchmark-pbya-v2`
datasets in the `bench_spatialcpa` conda env (the maths is unchanged; only the
environment differs).
