# benchmark-pbya-v3 — datasets

_Carried over from the parent project's `benchmark-pbya-v3/README.md` ("Datasets" through "Section count and hold-out pattern"). Paths like `benchmark-pbya/data/raw/...` are relative to this repository root. For the build commands, sizes and times in this repository, see README.md, "Full benchmark"._

## Datasets

Eighteen volumes. The first seven build from `data/raw/`; the rest are read from
`benchmark-pbya`'s **processed** `data.h5ad` (see [Where each source comes
from](#where-each-source-comes-from)).

| dataset | `kind` | partition | source | status |
|---|---|---|---|---|
| `starmap_visual_cortex` | `paper` | `planes` — trim z 6–13 / 91–94, split 77 planes into 7 × 11 | the paper's own volume | the reproduction |
| `exseq_visual_cortex` | `analogue` | `z_width` — cut the z range into 7 equal-width slabs (0.2 % outlier clip, not a noise trim) | `benchmark-pbya/data/raw/exseq_visual_cortex` (or v1's processed h5ad) | the same protocol, a second volume |
| `imc_breast_cancer` | `analogue` | `sections` — all 15 real serial sections at 10 µm, used as-is | `benchmark-pbya/data/raw/imc_breast_cancer` (15 h5ads, or v1's processed) | protein panel, human tumour |
| `cosmx_nsclc_3d` | `analogue` | `sections` — all 6 real cryosections, **30 µm** apart | `benchmark-pbya/data/raw/cosmx_nsclc_3d` (2 zips, or v1's processed) | widest *uniform* gaps |
| `deep_starmap` | `analogue` | `planes` — 0.70 µm optical planes grouped into 7 slabs, no trim | `benchmark-pbya/data/raw/deep_starmap` (3 CSVs, or v1's processed) | dense volume, mouse brain |
| `merfish_hypothalamus` | `analogue` | `sections` — 12 coronal sections, **50 µm** apart (animal 1) | `benchmark-pbya/data/raw/merfish_hypothalamus` (one CSV, all animals) | new tissue, wide gaps |
| `openst_lymph_node` | `analogue` | `sections` — 19 cryosections | `benchmark-pbya/data/raw/openst_lymph_node` (19 h5ad.gz) | human lymphoid tissue |
| `allen_merfish_brain` | `analogue` | `sections` — 59 sections, centred window | Allen ABC `MERFISH-C57BL6J-638850` | whole mouse brain, most sections |
| `allen_zhuang_abca1` | `analogue` | `sections` — centred window of 15 | Allen ABC `Zhuang-ABCA-1` | whole mouse brain |
| `allen_zhuang_abca2` | `analogue` | `sections` — centred window of 15 | Allen ABC `Zhuang-ABCA-2` | whole mouse brain, 2nd parcellation |
| `merfish_thick_cortex` | `analogue` | `z_width` — 100 µm block into 7 slabs (~14 µm) | Fang et al. 2023, Dryad | thick-tissue volume |
| `merfish_thick_hypothalamus` | `analogue` | `z_width` — 200 µm block into 7 slabs (~29 µm) | Fang et al. 2023, Dryad | thickest slabs in the benchmark |
| `easi_fish_lha1/2/3` | `analogue` | `z_width` — each sample into 7 slabs | Figshare 13749154 | three independent volumes, **tiny panel** |
| `exseq_breast_cancer` | `analogue` | `z_width` — **5** sections (the volume is small) | Alon et al. 2021, Zenodo | human tumour, ~3 100 cells |
| `st_mouse_brain_ortiz` | `analogue`, **spot** | `sections` — centred window of 15 of 75 | GEO GSE147747 | spot array, whole transcriptome |
| `visium_mouse_brain_c2l` | `analogue`, **spot** | `sections` — all 3 (mouse 1) | E-MTAB-11114 | spot array, thinnest experiment |

```bash
python -m src.bench3.prepare_dataset --dataset exseq_visual_cortex
python -m src.bench3.run_all --dataset exseq_visual_cortex
```

### Two flags that are not decoration

**`resolution: "spot"`** — `st_mouse_brain_ortiz` and `visium_mouse_brain_c2l`
are spot arrays, not single cells. Everything mechanical works, but a spot pools
tens of cells, so `paper_celltype_localization` scores *deconvolved composition*
rather than cells and the binned marker field is already binned by the array
geometry before v3 bins it. `rank_methods` prints the flag and says so. Read the
two spot rows against each other, never against a single-cell row — the same
discipline `kind` imposes between the paper dataset and the analogues.

**Size caps.** Three datasets arrive at a scale the earlier seven never reached,
so their specs carry caps that are applied **when the dataset is built** — so the
ground truth and every method's input hold the same cells and the same genes:

* `n_hvg` (ST, Visium: 3 000) — whole-transcriptome sources arrive at 20–35 k
  genes against the 28–1 000 of the targeted panels. Uncapped, the per-gene
  Moran's/Geary's families average over tens of thousands of near-empty genes,
  and any method that materializes a dense cell-by-gene matrix cannot load the
  input at all. The dataset's own marker and layer genes are force-included, so
  the selection can never silence `paper_marker_*`.
* `max_cells_per_section` (the three Allen atlases: 20 000) — those volumes run
  to millions of cells. The subsample is spatially stratified, so it keeps the
  tissue outline and the density gradient the field metrics read.

Both are recorded in `uns['paper_protocol']['gene_selection']` /
`['cell_subsample']` and overridable per run with `--n-hvg` /
`--max-cells-per-section`. `openst_lymph_node` is whole-transcriptome too and is
**not** capped here — see [Known gap](#known-gap-openst-is-still-uncapped).

Both write to `benchmark-pbya-v3/data/processed/<dataset>/data.h5ad` (override with
`$BENCH_V3_DATA`; the build prints the destination). ExSeq needs no arguments: it
resolves `benchmark-pbya/data/raw/exseq_visual_cortex` first, then v1's processed
h5ad — `$BENCH_V3_RAW_EXSEQ` or `--raw` override. The raw form is the spacejam2
cell-by-gene CSV, read directly by `sources.read_exseq_csv`, so v1's processing
pipeline does not have to have been run. Cell types come from `results_adata.h5ad`
beside the CSV when it is present and row-aligned; without it they stay `unknown`
and the `paper_celltype_*` group is unavailable, which the build says out loud.

Results, inputs and figures are keyed by dataset (`results/<method>/<dataset>/…`),
so the two never mix. The summary stage handles every dataset present in one go:

```bash
python -m src.bench3.evaluate_all       # each prediction against ITS own ground truth
python -m src.bench3.aggregate_results  # all_metrics/per_section carry a `dataset` column;
                                        # summary_by_method is per (dataset, method)
python -m src.bench3.rank_methods       # one ranking table per dataset — never pooled
python -m src.bench3.plot_paper_figures # results/summary/figures/<dataset>/…
```

`--dataset` takes a registered **name** or a path to a built `data.h5ad`, on every
stage that has it. Ranks are **within** a dataset: a composite is a position among the methods run on
that volume, so averaging STARmap's and ExSeq's composites would compare places in
two different races. Restrict any stage to one dataset with `--dataset-name`.

### Where each source comes from

**The original seven build from `data/raw/`.** v1's processed files are accepted
as an alternative, but none is required: `sources.py` reads each raw distribution
in its own form — ExSeq's cell-by-gene CSV, IMC's per-section h5ads,
Deep-STARmap's expression/spatial CSVs, and CosMx's two zips (the shipped h5ad
carries STIM coordinates in arbitrary units, so it is joined to the per-section
flat files for physical micrometres). Those readers mirror v1's processors rather
than calling them, so v3 stays self-contained.

**The Allen atlases build from either.** `sources.read_allen_ccf` reads the raw
distribution directly — an expression `.h5ad` beside the cell-metadata CSV that
carries the reconstructed CCF position and the cluster annotation — including the
mm → µm conversion, which is the step that is silently wrong if skipped, since
every other v3 dataset is micrometres. `reader_region` selects which Zhuang
parcellation.

**The remaining six read v1's *processed* `data.h5ad`.** `merfish_thick_*`,
`easi_fish_lha*`, `exseq_breast_cancer`, `st_mouse_brain_ortiz` and
`visium_mouse_brain_c2l` have no raw reader in v3, because their raw forms are a
Dryad archive with an R/Seurat fallback path, a MATLAB `.mat` of transcript
positions, and a per-slide Visium ZIP respectively — re-implementing those
faithfully is a much larger job than the CSV and h5ad readers above, and getting
one subtly wrong is the failure mode this benchmark is least able to detect. So
they require v1's processing pipeline to have been run, and the build says so if
the file is missing. Adding a raw reader later changes nothing else: it is one
entry in `sources.READERS` plus a `reader` key.

### The coordinate trap these datasets introduced

The Allen atlases copy their whole metadata CSV into `obs`, which puts
*section-local* `x`/`y`/`z` right next to the reconstructed volume position in
`obsm['spatial']`. `extract_xyz` prefers `obs['x','y','z']` — correct for
STARmap, catastrophic here: the build would assemble the stack out of unrelated
per-section frames and produce a dataset with the right section count, the right
cell counts, monotone section centres, and geometry that is noise.

Two things stop it. Those specs set `"coords": "obsm"`, which says which array to
believe rather than relying on a priority order. And `prepare_dataset` now checks
the property that actually distinguishes the two for any `partition="sections"`
dataset: in real serial sections every cell in a section shares that section's z,
so the spread *within* a section must be small against the gap *between*
sections. Comparable values mean z is a per-cell quantity and the build stops.

That second check matters because the existing z-monotonicity assertion in
`verify` cannot catch this — for `sections` the section index is *assigned* by
sorting on the section centres, so "centres increase with index" is true by
construction. Verified both ways: with `coords: "obsm"` the eight new datasets
build correctly, and flipping `allen_zhuang_abca1` back to `auto` on the same
source is refused by the new check rather than silently succeeding, which is what
it did before.

### Known gap: Open-ST is still uncapped

`openst_lymph_node` is whole-transcriptome like the two spot datasets, and it is
**not** given an `n_hvg` cap here. v18's wrapper densifies the training matrix
(`run_spatialcpav18.py:352`, `X_raw = _to_dense_f32(adata.X)`), which at ~10⁶ cells ×
~2×10⁴ genes is order 80 GB; a run with a wrapper that densifies the same way was
killed by the OOM killer on this dataset. The v18_* ablations densify too, so every
v18-hosted row on this dataset is expected to fail the same way. Capping it is the same one-line spec change the ST
and Visium entries already carry, but it changes the panel a *previously reported*
dataset was measured on, so it is left as a deliberate decision rather than folded
into this change. Until then, expect that run to fail.

**Why ExSeq.** Same tissue as STARmap — mouse visual cortex — so the marker genes
and the laminar-axis composite carry over unchanged, and it is the only candidate
in `benchmark-pbya` for which none of the metric definitions have to be
reinterpreted. It is an independent technology on independent tissue, which is
what a second dataset is *for*.

**About the IMC dataset.** It is the one case where the metrics change meaning, so
read its rows differently from the other two:

* *Protein, not RNA.* The markers are **panCK** (tumour/epithelial compartment)
  and **CD3** (T-cell infiltrate), following the published analysis of this volume.
  Matching ignores case and punctuation, so `panCK` finds `PanCK` or `pan-CK`; it
  is deliberately *not* a prefix match, because `CD3` would then silently select
  `CD31`. If a marker does not match, the build says so and suggests the closest
  panel names.
* *A compartment axis, not a laminar one.* A tumour has no cortical layers, but
  those same two markers define the axis that matters: the signed score is
  `z(CD3) − z(panCK)`, so its in-plane gradient points from tumour toward immune
  infiltrate. `paper_marker_depth_r` therefore reads as **"profile across the
  tumour–immune axis"** — derived from the ground truth exactly as the cortical
  version is. Empty both layer lists in `config.DATASET_SPECS` to fall back to the
  generic axis (the gradient of whichever channel is most spatially structured).
* *Real serial sections, so registration matters.* These are cut, mounted and
  imaged independently — not one imaging block — so the policy is `rigid`, not
  `none`. The training slices move into a common frame while the held-out ground
  truth stays in the original one, and the evaluation-side prediction→GT alignment
  absorbs the difference. That makes the alignment-dependent metrics
  (`paper_marker_field_r`, `paper_celltype_localization`) noisier here than on the
  two co-registered volumes; the autocorrelation and distributional families are
  unaffected.
* *Half the volume is unused.* The paper design needs 7 consecutive sections and
  the dataset has 15, so the centred window is kept and 8 sections are dropped.

### Sections have to be big enough to run

The build refuses a dataset with a section under 50 cells (`--min-cells-per-section`,
or `--allow-small-sections` to force it), and prints `cells/section: min/median/max`
with an imbalance warning otherwise. This is not a metric threshold — it is what the
*task* needs. A section that thin is useless as a method input (SpatialZ's
flanking-slice PCA asks for 20 components and fails below 20 cells; every
interpolation method needs a neighbourhood) and useless as ground truth (the binned
field, the depth profile and the per-type OT all become noise).

It matters most for `z_width`, where equal-width slabs on a volume whose cell
density falls off toward one end can leave the last slab nearly empty. The error
names the offending sections and suggests concrete `--z-trim-quantile` /
`--n-sections` combinations that would work.

### Trimming

Only STARmap's trim is protocol. Dropping `z = 6–13` and `91–94` comes from the
SpatialZ paper's own analysis of that volume, so it is always applied and
`--no-trim` is refused for it. The analogue datasets have no published trim and v3
does not invent one:

* `exseq_visual_cortex` clips 0.2 % of cells at each end of z — not to remove
  noise, but because equal-width binning takes its edges from `min(z)`/`max(z)`,
  where a few segmentation outliers would skew all seven slab boundaries.
  `--no-trim` uses the raw range; `--z-trim-quantile` sets it.
* `imc_breast_cancer` trims nothing: it uses all 15 of its sections. Pass
  `--n-sections` to take a smaller window, and `--section-trim low|center|high` to
  choose which.

### Section count and hold-out pattern

**Seven sections and the 2/4/6 split are STARmap's, because they are the paper's.**
They are pinned for that dataset and derived for every other one:

| dataset | sections | held out | holdout id |
|---|---|---|---|
| `starmap_visual_cortex` | 7 (pinned — the published design) | 2, 4, 6 | `paper_2_4_6` |
| `exseq_visual_cortex` | 7 (a choice: ≈11 µm slabs, like a cryosection) | 2, 4, 6 | `paper_2_4_6` |
| `imc_breast_cancer` | 15 (all it has) | 2, 4, …, 14 | `paper_alt7of15` |
| `cosmx_nsclc_3d` | 6 (all it has) | 2, 4 | `paper_2_4` |
| `deep_starmap` | 7 (a choice: ~14 µm slabs of 0.70 µm planes) | 2, 4, 6 | `paper_2_4_6` |
| `merfish_hypothalamus` | 12 (all animal 1 has) | 2, 4, …, 10 | `paper_alt5of12` |
| `openst_lymph_node` | 19 (all it has) | 2, 4, …, 18 | `paper_alt9of19` |

What actually carries over from the paper is the **alternating hold-out**, not the
number seven: hold out every even section, keep the first and last as input. At
n = 7 that is exactly 2/4/6; at n = 15 it is 2/4/…/14. Either way every held-out
section is bracketed by two input sections — so the task stays well-posed — and
about half the volume is missing, which is what makes it hard. Set `held_out` to an
explicit tuple in `config.DATASET_SPECS` to pin a different split, and
`--n-sections` to change the count.

**One more caveat.** `kind=analogue` is not decoration: the SpatialZ paper validated
this protocol on STARmap, so an ExSeq row is v3's extension of it. Report the two
separately and never pool their ranks.

