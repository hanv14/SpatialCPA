# Protocol, metrics, pose and leakage

_Carried over from the parent project's benchmark README (sections "The protocol" through "Leakage policy"), with paths and method lists updated to this tree. The code these sections describe is the pinned code in `benchmark/src/`._

## The protocol

Straight from the paper's description of the dataset:

| Step | What the paper says | What the benchmark does |
|---|---|---|
| Source | STARmap mouse visual cortex, 3-D, single-cell resolution | `benchmark/data/raw/starmap_visual_cortex/STARmap_Wang2018three_data_3D_data.h5ad` (32 845 cells × 28 genes, 89 z-planes) |
| Trim | remove uppermost `z = 6–13` and lowermost `z = 91–94` | drops 3 867 cells (11.8 %); 77 planes remain (`z = 14–90`) |
| Partition | divide the remainder into **seven consecutive 2-D sections** | 77 / 7 = **exactly 11 planes per section** — the partition is even, no fudging |
| Split | hold out sections **2, 4, 6**; input sections **1, 3, 5, 7** | one holdout config, `paper_2_4_6` |
| Objective | reconstruct the missing sections as virtual slices | generation-only; the held-out sections are the ground truth |

```
                 z(µm)   planes    cells
  section_1       19     14–24      4073   input
  section_2       30     25–35      4187   HELD OUT
  section_3       41     36–46      4169   input
  section_4       52     47–57      4102   HELD OUT
  section_5       63     58–68      4110   input
  section_6       74     69–79      4162   HELD OUT
  section_7       85     80–90      4175   input
```

Two implementation choices worth stating outright:

* **Sections are flattened to 2-D.** Each partition is an 11-plane slab, but the
  protocol calls it a *section* — the thing a microtome produces and the thing a
  virtual slice has to be. So each slab's cells keep their real `(x, y)` and take
  the slab's centre z. The original plane survives in `obs['z_plane']`;
  `--no-flatten-z` opts out.
* **All three sections are held out at once**, not one at a time. That is the
  paper's design and it is materially harder: half the volume is missing and no
  reconstruction ever sees an adjacent real slice. Each held-out section is still
  bracketed by two input sections (1|3, 3|5, 5|7), so the task stays well-posed
  for every interpolation method. A `--design loo` robustness check exists, but
  it is an *easier* task and its numbers are not comparable.

---

## What is measured

The paper validates a reconstruction five ways. Each becomes a metric group in
`evaluate_paper.py`, and `rank_methods.py` ranks methods within each group and
averages the group ranks — so no criterion dominates just because it happens to
contribute more individual numbers.

| Paper's validation | Metrics | Notes |
|---|---|---|
| **UMAP continuity** between real and reconstructed slices | `paper_umap_mixing` ↑, `paper_umap_centroid_dist` ↓, `paper_embedding_mixing_pca` ↑ | kNN mixing in a shared embedding, normalized by the value expected under perfect mixing. 1 = the two clouds are locally indistinguishable, 0 = disjoint islands. The PCA variant is deterministic and is the number to trust if UMAP's stochasticity is a concern. |
| **Marker-gene spatial patterns** (Flt1, Pcp4, Cux2) | `paper_marker_field_r` ↑, `paper_marker_field_ssim` ↑, `paper_marker_depth_r` ↑, `paper_marker_morans_mae` ↓, plus per-gene breakdowns | Binned 2-D field (Pearson **and** SSIM); profile along the cortical **laminar axis**; and the deviation in each marker's own Moran's I. SSIM adds the contrast/level sensitivity that Pearson r is blind to — r is invariant to an affine rescale of the field, so a marker reproduced with the right *shape* but the wrong *level* is caught by SSIM where r misses it (the SpatialZ audit's spatial-pattern SSIM, per marker). The laminar axis is derived from the ground truth as the spatial gradient of a signed layer score (superficial minus deep markers) — for cortex the depth profile is the honest "did you reproduce the pattern" test, and it is far more robust than the 2-D field to residual in-plane misalignment. |
| **Spatial autocorrelation** — Moran's I *and* Geary's C | `paper_morans_pearson`/`_spearman`/`_mae`, `paper_gearys_*`, plus pred/GT medians | Both computed per gene on a row-standardized kNN graph *within* each slice, so they are alignment-free. The correlations say whether the *ranking* of genes by spatial structure survives; the MAEs say whether the *level* is right — which is what catches over-smoothing (blur inflates Moran's I and deflates Geary's C while leaving the ranking intact). |
| **Preservation of cell spatial localization** (incl. rare types) | `paper_celltype_localization` ↑, `paper_celltype_ot` ↓, `paper_rare_celltype_localization` ↑, `paper_rare_celltype_recall` ↑ | Per cell type, a debiased Sinkhorn (OT) divergence between the predicted and true spatial distributions, calibrated against a within-tissue null: scattering that type anywhere in the tissue. 1 = localization reproduced, 0 = no better than random placement. The main score is GT-frequency-weighted, so it also reports the **rare-cell-type** counterpart (types below `RARE_CELLTYPE_FRAC` = 5%): `rare_celltype_localization` is the *unweighted* localization over rare types (the ranked score — are the rare niches in the right place), and `rare_celltype_recall` is the fraction of rare types produced at all (a pose-independent presence diagnostic, not ranked). |
| **Gene expression similarity** | `paper_gene_mean_spearman` ↑, `paper_gene_var_spearman` ↑, `paper_gene_detection_spearman` ↑ (+ pred/GT median detection) | Per-gene mean/variance agreement on log-normalized expression, plus **detection frequency** — the fraction of cells expressing each gene, i.e. the panel's *sparsity structure*. Detection is computed on the raw emitted expression, not the rank-normalized matrix the spatial metrics use: rank-normalization maps every gene's zeros to the same low rank and so erases exactly this quantity, which is why it is reported separately (and why the SpatialZ audit lists it). It is invariant to any zero-preserving transform (log1p, library-normalization) but correctly penalises a method that emits a *dense* field with no zeros. This is its own ranked group, so gene variance and detection actually count toward the composite. |

Also written: the correspondence-free `gen_*` metrics
(`src/benchmark/evaluate_generation.py`) and the cell-matched metrics
(`src/benchmark/evaluate.py`) as **reference only** — de-novo generation produces no
cell-to-cell correspondence, so those are not a valid score here.

`paper_cell_count_ratio` is reported but deliberately **not** ranked: the number
of cells is emergent in generation-only mode, so it is a diagnostic ("did the
method produce a plausible amount of tissue?"), not a quality score.

### Two properties that make the comparison fair

**Scale fairness.** Methods emit expression on different scales (raw counts,
log1p, arbitrary). Every primary metric is computed on **per-gene
rank-normalized** expression — invariant to any monotonic per-gene transform — so
two methods differing only in output scale get identical scores. The
rank-normalizer is imported from `src/benchmark/evaluate_generation.py` rather
than re-implemented, so the paper metrics and the `gen_*` metrics cannot silently
drift apart.

**Correspondence freedom.** Nothing here correlates prediction against ground
truth cell-by-cell. Generation synthesizes cells; it does not place them on GT
cells, so a manufactured correspondence measures alignment noise, not fidelity.

### Choosing the pose (alignment-dependent metrics only)

Two metrics compare *positions* — `paper_marker_field_r` and
`paper_celltype_localization` — so the prediction has to be rotated into the
ground truth's frame first. Everything else (both autocorrelation families, the
embedding mixing, the distributional metrics, `paper_marker_depth_r` up to the
laminar axis) is computed within each slice and is pose-invariant.

`src/bench3/align.py` picks that pose, on two rules:

* **Expression breaks the symmetry that geometry cannot.** Candidate rotations
  are scored by agreement between the *binned marker fields*, not by how many
  cells overlap. On tissue with a symmetric outline — a bilaterally symmetric
  coronal section, say — identity and 180° cover the ground truth equally well,
  so an occupancy score cannot separate them and the pose falls to noise; a flip
  inverts the marker gradient, so field agreement scores it about −1 and rejects
  it. When a dataset's marker panel resolves empty the aligner falls back to the
  genes with the highest ground-truth Moran's I.
* **Reflections are not candidates.** Physical tissue has fixed handedness, so a
  mirrored pose is wrong by construction however well it fits. Only proper
  rotations (det = +1) are searched.

Ties resolve toward the smallest rotation, so an already-registered dataset keeps
the identity pose. The chosen rotation, scale, score and runner-up score are
written into each section's entry in `metrics.json` (`align_rotation_deg`,
`align_score`, `align_runner_up`, …), with the largest rotation applied surfaced
at the top level as `align_rotation_deg_max` — a marginal decision is visible
rather than silent. The figures use the same aligner, so a marker map is drawn at
the orientation its reported `paper_marker_field_r` was computed at.

Measured on synthetic sections whose correct pose is known: on symmetric geometry
the occupancy aligner picked a wrong pose in 3 of 12 runs (`field_r` ≈ −0.94
where the correct pose gives +0.94); the expression-scored one, 0 of 12. Scored
end to end through `evaluate_paper` on a prediction delivered in an arbitrary
frame, `paper_marker_field_r` holds at ≈ 0.89 for rotations of 0°, 40° and 150°,
against 0.44–0.48 before.

---

## Leakage policy

`src/benchmark/leakage_guard.py` (re-exported by `_v2bridge.py`), and it
matters more here, not less, because three sections are missing at once:

1. **Membership** — the held-out cells are physically absent from the file a
   method receives. `split_holdout` builds it, `assert_no_leakage` checks it, and
   the wrapper re-checks with `guard_no_holdout` before touching the data.
2. **Geometry** — methods get a *scalar target z* per held-out section and
   nothing else. Never the held-out `(x, y)`; the cell count is emergent.
3. **Registration** — the training slices are re-registered into a common frame
   using training slices only. For STARmap the policy is `none`: it is a single
   3-D imaging block whose z-planes are inherently co-registered, so
   re-registering would only introduce distortion. (The same call holds for every
   volumetric dataset.)
4. **Global statistics** — label vocabularies are built from the training input
   only; expression normalization is per-cell.

The evaluation side *is* allowed to read the ground truth — that is what
evaluation means. The prediction→GT rigid alignment used by the binned-field and
localization metrics is an evaluation-side operation that feeds nothing back to
the method. How that pose is chosen is described under
[Choosing the pose](#choosing-the-pose-alignment-dependent-metrics-only).

The `train_registered.h5ad` is built **once per holdout and reused by every
method**, so the comparison is apples-to-apples by construction.

---

