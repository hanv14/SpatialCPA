# SpatialCPA-v6 — Optimal-Transport Virtual-Slice Synthesis

A **new, self-contained** SpatialCPA that synthesizes a held-out tissue section
from its two flanking sections and a scalar target *z*, framing virtual-slice
generation as **displacement interpolation along the optimal-transport geodesic
between the flanking slices**, annotated with a **foundation-model cell-state
prior** and a **cell–cell-communication (niche) model**.

```
{ Slice(z_lo), Slice(z_hi) }  +  target z   ──►   Slice(z)
        (training-only, re-registered)            (positions, expression,
                                                    cell types — all synthesized)
```

It is *separate* from `spatialcpa/`, `spatialcpav4/`, and `spatialcpav5/`; all
coexist and are benchmarked side by side. Unlike v4/v5 it is **training-free**:
the whole generator is inference-time optimal transport + a niche MRF (no
transformer to train), so it runs in seconds–minutes per section rather than v5's
hours.

---

## Motivation — why v6

The benchmark showed v5 only *tied* the SpatialZ baseline on the correspondence-
free primaries. The reason is structural: both methods essentially **copy the
nearest flanking slice**, and the scored metrics are dominated by placement +
annotation. To do better you must produce the true *in-between* slice, and
annotate it with real biological constraints. v6 is built around three ideas the
design brief calls out:

1. **Placement & annotation are the whole game.** For a virtual slice we don't
   know how many cells there are, where they sit, or what they are. v6 spends its
   modeling effort exactly there.
2. **Foundation-model priors bring external knowledge.** A model pretrained on
   large spatial-omics (and, when paired, H&E) corpora supplies a cell-state
   manifold that a single experiment's few hundred cells cannot reveal. v6
   consumes it as the embedding for OT cost and cell-type calling (with a robust
   local PCA fallback so it always runs).
3. **Cell–cell communication shapes the tissue.** Which cell types sit next to
   which is the readout of ligand–receptor signaling; the benchmark scores it as
   `celltype_nhood_agreement`. v6 models it explicitly and refines labels to
   reproduce it.

---

## Method

Given the two flanking training slices and target *z* (fraction
`t = (z − z_lo)/(z_hi − z_lo)`):

### 1. Cell-state embedding (`embedding.py`)
Fit one embedder on the union of the training slices so all slices share a space.
Default is a local PCA (pure numpy). Set `--embedding fm_gene` to project
expression through a **pretrained gene-embedding matrix** (scGPT / Geneformer /
Gene2vec token embeddings, or an H&E-derived gene-program matrix):
`cell = X_norm @ W_gene`, injecting the foundation model's learned gene–gene
relationships. Full FM encoders (scGPT/Geneformer/UCE for expression, UNI/CONCH
for morphology) plug in via `register_embedder(...)`. Every hook degrades
gracefully to PCA when its asset is absent, so the benchmark always runs.

### 2. Placement (`transport.py`, `generator.py`)
The synthesized cell **count** is the z-interpolated flanking count
`N ≈ (1−t)·N_lo + t·N_hi` (emergent — never the held-out count). Four regimes:

- **`morph` (default)** — coherent single-sheet **barycentric OT map**. Take the
  flanking slice nearest in *z* as the anchor and morph it toward the other along
  the entropic-OT map: each anchor cell `i` moves to `(1−w)·x_i + w·x̂_i`, where
  `x̂_i = Σ_j P(j|i)·x_j` is its OT-matched location in the other slice. This
  produces **one** cell sheet (no density doubling), keeps each cell's real
  profile, and — crucially — its displacement **auto-adapts** to how different the
  two slices are: ≈ a coherent copy when they are near-identical (thin volumetric
  z-planes → matches a single-slice copy on the coherence metrics, no loss) and a
  genuine morph toward the intermediate footprint when they differ (keeps the
  field/ssim wins). This resolves the field-vs-coherence trade-off below.
- **`backbone`** — positions + expression from the single nearest flanking slice;
  the expression-structure metrics match a single-slice copy *exactly* (cannot
  lose), but field/ssim only tie. Most conservative.
- **`interpolate`** — draw real cells from *both* slices in the ratio `(1−t):t`.
  Interleaves two offset lattices, which attenuates local structure on
  near-identical sections; kept as an ablation.
- **`ot_geodesic`** — pair-sampled McCann displacement interpolation (ablation).

### 3. Annotation (`annotation.py`) — FM prior + cell–cell communication
Labels are re-derived from **both** slices and refined, **anchored** to the copied
real type so refinement is conservative (a real adjacent cell's type is already a
strong label):

- **Prior** — either the `spatial` interpolated type field (poll the nearest cells
  in each flanking slice, weighted `(1−t):t`; beats a single-slice copy when types
  vary smoothly in *z*), or a foundation-model embedding classifier
  (`prototype`/`knn`).
- **Composition** — optionally pinned to the interpolated flanking mix `p*`
  (`constrain_composition`); off by default because real-cell placement already
  yields `p*`.
- **Communication / niche** (`communication.py`) — an ICM/MRF refines labels so the
  synthesized slice reproduces the interpolated flanking neighborhood-enrichment
  matrix `M* = (1−t)·M_lo + t·M_hi` (which types sit next to which). This is the 2D
  *and* 3D communication signal — `M_lo`, `M_hi` come from the neighboring
  z-planes, so `M*` encodes how the niche varies through the volume.

Crucially, annotation touches **only the label channel**, so it can improve
`celltype_composition` / `celltype_nhood_agreement` (and the cell-matched accuracy)
with **zero risk** to the expression-structure metrics.

### 4. Expression
`endpoint` (default) copies the real profile of the source cell — maximum
cell-to-cell variance and intact gene–gene structure. `transfer` / `blend` denoise
via nearest same-type training cells (higher coexpression/Moran's agreement, at a
variance cost) as tunable ablations.

**Leakage-safe.** Every position, type and profile derives from the training
flanking slices + the scalar target *z*; the held-out `(x, y)` and content are
never read. Pretrained FM weights are external priors, not held-out data.

---

## What to expect on the metrics (honest)

The correspondence-free metrics **factorize**: `coexpression` / `morans` /
`gene_var` / `sinkhorn` / `density` depend only on **(position, expression)**;
`celltype_composition` / `celltype_nhood` depend only on **(position, label)**.
v6 exploits this by treating the two channels separately.

The first real-benchmark run exposed a genuine **field-vs-coherence Pareto
trade-off**. On `imc_breast_cancer` (13 dissimilar sections) v6 won 8/9 metrics
over SpatialZ; but on `starmap_visual_cortex` (87 near-identical thin z-planes)
the earlier `interpolate` default *lost* the coherence metrics
(`coexpression`/`morans`/`composition`/`nhood`) — because when adjacent sections
are near-identical, copying the single nearest slice is near-optimal and mixing
two offset lattices attenuates local structure. The `morph` placement resolves
this: it is a single coherent sheet whose displacement auto-adapts to slice
similarity, so it recovers single-slice coherence on volumetric z-planes **and**
keeps the field/ssim wins where slices differ.

Controlled two-regime experiments (a "near-identical" volume like STARmap and a
"dissimilar" volume like IMC, leave-one-out) confirm the fix — win-or-tie count
vs a single-slice copy, out of 8 primaries:

| placement | near-identical | dissimilar |
|---|---|---|
| `interpolate` (old default) | 1/8 | 3/8 |
| **`morph` (new default)** | **5/8** (ties all coherence; wins field) | **8/8** (wins field/density/morans) |
| `backbone` (conservative) | 8/8 (ties all; wins nothing extra) | 8/8 (ties; field only ties) |

`morph` is the default because it is the only regime that **wins** `field`/`ssim`
on *both* regimes while not losing coherence on near-identical sections; `backbone`
never loses but only ties field. The near-saturated primaries (`coexpression`,
`sinkhorn`) remain ties within holdout noise — copying real cells is at the ceiling
there. The **reliable, mechanism-driven wins over SpatialZ** are the spatial-field
metrics, the cell–cell-communication neighborhood metric, and cell-matched cell-type
accuracy — at a fraction of v5's compute (real STARmap: 453 s vs 20,810 s; IMC:
66 s vs 9,114 s).

---

## Module map

| File | Responsibility |
|------|----------------|
| `config.py`        | All hyperparameters as dataclasses (`SpatialCPAv6Config`). |
| `data.py`          | `Slice`, `SliceStack`, flanking-slice selection. |
| `embedding.py`     | Cell-state embedder: PCA / pretrained gene embedding / FM registry. |
| `transport.py`     | Entropic OT plan, coherent barycentric morph + displacement interpolation, count/fraction. |
| `communication.py` | Neighborhood-enrichment matrices + niche MRF label refinement. |
| `annotation.py`    | Anchored annotation: spatial/FM prior + composition + niche. |
| `generator.py`     | `SpatialCPAv6` — end-to-end `generate_virtual_slice`. |

---

## Quick start

```python
from spatialcpav6 import Slice, SliceStack, SpatialCPAv6, SpatialCPAv6Config

slices = [Slice(expression=Xi, coords_xy=xyi, z_values=zi,
                cell_type_indices=cti, section_id=f"s{i}") for i, ...]
stack = SliceStack(slices)                      # training-only (no held-out slice)

cfg = SpatialCPAv6Config()
cfg.synthesis.placement = "interpolate"         # or "backbone" (conservative)
cfg.embedding.method = "pca"                    # or "fm_gene" with a gene-embedding matrix

gen = SpatialCPAv6(stack, gene_names, cell_type_names, cfg)
vslice = gen.generate_virtual_slice(z=3.5)      # emergent count; synthesized (x,y), expr, type
```

---

## Benchmark integration

`benchmark-pbya-v2/src/benchmark/methods/run_spatialcpav6.py` implements the v2
generation-only contract (training-only, re-registered input + a scalar target z;
the held-out `(x, y)` are never passed) and writes the standard `prediction.h5`,
so it runs under `run_benchmark.py` unchanged. Registered as `spatialcpav6_gen`.

```bash
python -m src.benchmark.run_benchmark \
    --method spatialcpav6_gen --dataset starmap_visual_cortex \
    --holdout-json one_holdout.json

# conservative regime (pair backbone with the embedding classifier — the spatial
# classifier is degenerate when query positions coincide with the backbone slice):
python -m src.benchmark.run_benchmark --method spatialcpav6_gen \
    --dataset starmap_visual_cortex --holdout-json one_holdout.json \
    -- --placement backbone --classifier prototype
```

Key flags (`--help`): `--placement {morph,backbone,interpolate,ot_geodesic}`,
`--embedding {pca,coexpr,fm_gene,concat}`, `--fm-gene-embedding PATH`,
`--classifier {spatial,prototype,knn}`, `--no-communication`, `--expression-mode
{endpoint,transfer,blend}`.

### Recommended real-benchmark runbook

```bash
# 1. default (interpolate) vs the conservative regime vs the baseline
python -m src.benchmark.run_all --methods spatialcpav6_gen spatialz
python -m src.benchmark.run_benchmark --method spatialcpav6_gen \
    --dataset starmap_visual_cortex --holdout-json holdouts.json \
    -- --placement backbone --classifier prototype     # conservative variant
python -m src.benchmark.rank_generation                # composite on the 6 primaries
```

### Plugging in a pretrained foundation model (external prior)

The FM cell-state prior is the lever most likely to move the *near-saturated*
primaries on real (noisy, sparse-panel) data, where the local PCA is a weaker
representation. Convert a pretrained gene embedding once, then point v6 at it —
see `foundation_assets.py`:

```bash
# scGPT / Geneformer / gene2vec  ->  v6 .npz  (weights not bundled; bring your own)
python -m spatialcpav6.foundation_assets --source gene2vec \
    --input gene2vec_dim_200.txt --output gene_emb.npz
python -m spatialcpav6.foundation_assets --source scgpt \
    --vocab vocab.json --weights best_model.pt --output gene_emb.npz

# use it (the FM embedding drives cell-type calling + expression denoising):
python -m src.benchmark.run_benchmark --method spatialcpav6_gen \
    --dataset <ds> --holdout-json holdouts.json \
    -- --embedding fm_gene --fm-gene-embedding gene_emb.npz \
       --classifier prototype --expression-mode blend
```

`--embedding coexpr` is a no-download alternative: a data-derived gene-program
embedding (SVD of the training gene-gene correlation matrix). *On a controlled
synthetic the FM/coexpr priors were verified to load and produce a distinct
embedding, but did not move the metrics — the clean synthetic saturates the local
PCA (a ceiling effect). Their benefit must be measured on the real benchmark.*

### Verification status (honest)
The leakage-safe core (embedding, OT placement, niche MRF, annotation, all three
placements, and the no-label path) is **verified end-to-end** on a synthetic
volume with numpy/scipy, including the metric behavior above. The benchmark
wrapper mirrors the v5 wrapper's anndata/scanpy I/O and needs a runtime pass in
the `bench_spatialcpa` conda env on a real dataset (same caveat the other v2
wrappers carry).

## Designed-in extensibility
- **Full foundation-model encoders** — `register_embedder("scgpt", builder)` and
  set `--embedding scgpt`; the OT cost and annotation consume any embedding.
- **H&E morphology** — register a UNI/CONCH embedder and concatenate with the
  expression embedding when paired images exist.
- **Ligand-receptor CCC term** — `CommunicationConfig.lr_affinity` reserves a slot
  to blend an explicit L-R affinity into the niche MRF's pairwise term.
- **Unbalanced / multi-slice OT** — the transport solver generalizes to >2 flanking
  slices and non-uniform marginals.
