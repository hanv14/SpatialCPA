"""Registry, paths and protocol constants for benchmark-pbya-v3.

benchmark-pbya-v3 is a **paper-faithful reproduction benchmark**: it reproduces
the STARmap visual-cortex evaluation described in the SpatialZ paper (Lin et al.,
2025, Nat Methods) and runs *every* method through that exact protocol, so a
published method's numbers are reproducible and the new SpatialCPA variants are
measured against them on identical terms.

Differences from ``benchmark-pbya-v2``
--------------------------------------
* **One dataset, one protocol.** v2 sweeps 17 datasets with leave-one-out. v3
  runs the single design the SpatialZ paper used: partition the STARmap volume
  into 7 consecutive 2-D sections, hold out sections 2/4/6 *simultaneously*, and
  reconstruct them from sections 1/3/5/7.
* **Paper validation strategy.** In addition to v2's correspondence-free
  generation metrics, v3 computes the quantities the paper actually validated on:
  UMAP continuity between real and reconstructed cells, marker-gene spatial
  patterns (Flt1 / Pcp4 / Cux2), Moran's I *and* Geary's C, and preservation of
  per-cell-type spatial localization. See ``evaluate_paper.py``.
* **Shared method code.** The method wrappers are v2's, invoked unchanged by
  path (see ``METHODS``). Reusing them rather than forking keeps a single source
  of truth: whatever v2 measures, v3 measures with the same synthesis code.

The leakage policy is inherited from v2 wholesale (``_v2bridge``): the held-out
sections are physically absent from the method input, methods receive only a
scalar target z per held-out section, and label vocabularies are training-only.
"""

from pathlib import Path
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # benchmark-pbya-v3/
REPO_ROOT = PROJECT_ROOT.parent                           # SpatialCPA/
V1_ROOT = REPO_ROOT / "benchmark-pbya"                    # shared tools/ + envs/
V2_ROOT = REPO_ROOT / "benchmark-pbya-v2"                 # shared method wrappers
V2_SRC = V2_ROOT / "src"                                  # importable `benchmark` pkg

# Raw STARmap volume (Wang et al. 2018). Different checkouts keep it in
# different places, so resolve against the known locations rather than hard-coding
# one; $BENCH_V3_RAW_STARMAP wins outright, and --raw overrides per invocation.
#
# The v1-processed file is an accepted input too: it is the same cells and the
# same 89 z-planes, just with coordinates already converted to micrometres —
# ``prepare_starmap`` detects that and skips the voxel conversion.
_RAW_STARMAP_NAME = "STARmap_Wang2018three_data_3D_data.h5ad"
RAW_STARMAP_CANDIDATES = (
    REPO_ROOT / "data" / "starmap" / _RAW_STARMAP_NAME,
    V1_ROOT / "data" / "raw" / "starmap_visual_cortex" / _RAW_STARMAP_NAME,
    V1_ROOT / "data" / "processed" / "starmap_visual_cortex" / "data.h5ad",
    V1_ROOT / "data" / "processed" / "starmap_visual_cortex.h5ad",
)


def resolve_raw_starmap():
    """First existing candidate, else the preferred path (for the error message)."""
    env = os.environ.get("BENCH_V3_RAW_STARMAP")
    if env:
        return Path(env)
    for cand in RAW_STARMAP_CANDIDATES:
        if cand.exists():
            return cand
    return RAW_STARMAP_CANDIDATES[0]


RAW_STARMAP = resolve_raw_starmap()

# The v3-built, paper-protocol dataset (produced by ``prepare_starmap.py``).
# Laid out as ``data/processed/<dataset>/data.h5ad``, the same convention v1 and
# v2 use, so the tree is navigable the same way — but under benchmark-pbya-v3,
# because this is a *differently partitioned* view of the STARmap volume (seven
# paper sections, not the raw 89 z-planes) and must not be confused with, or
# written over, v1's processed copy. Which protocol produced a given result stays
# visible in the holdout id (``paper_2_4_6``) and in ``uns['paper_protocol']``.
DATA_DIR = Path(os.environ.get("BENCH_V3_DATA", PROJECT_ROOT / "data" / "processed"))
DATASET_NAME = "starmap_visual_cortex"
DATASET_PATH = DATA_DIR / DATASET_NAME / "data.h5ad"

RESULTS_DIR = Path(os.environ.get("BENCH_V3_RESULTS", PROJECT_ROOT / "results"))
SUMMARY_DIR = RESULTS_DIR / "summary"
FIGURES_DIR = SUMMARY_DIR / "figures"
INPUTS_CACHE = RESULTS_DIR / "_inputs"   # training-only inputs, one per holdout
ASSETS_CACHE = RESULTS_DIR / "_assets"   # method assets v3 prepares (see assets.py)


# ── The SpatialZ-paper STARmap protocol ───────────────────────────────────────
# "The original 3D volume was partitioned into consecutive 2D sections. Uppermost
#  layers (z = 6-13) and lowermost layers (z = 91-94) were removed to reduce
#  technical noise. The remaining tissue was divided into seven consecutive
#  sections. Sections 2, 4 and 6 were held out to simulate missing slices, while
#  sections 1, 3, 5 and 7 were used as input."
#
# The raw volume carries 89 z-planes (z = 6..94). Dropping 6-13 and 91-94 leaves
# 77 planes (z = 14..90), which divides into 7 sections of exactly 11 planes.
DROP_Z_LOW = (6, 13)        # inclusive
DROP_Z_HIGH = (91, 94)      # inclusive
N_SECTIONS = 7
HELD_OUT_SECTION_INDICES = (2, 4, 6)
INPUT_SECTION_INDICES = (1, 3, 5, 7)

def section_label(index):
    """Canonical section label for a 1-based paper section index."""
    return f"section_{int(index)}"

HELD_OUT_SECTIONS = tuple(section_label(i) for i in HELD_OUT_SECTION_INDICES)
INPUT_SECTIONS = tuple(section_label(i) for i in INPUT_SECTION_INDICES)

# Voxel calibration — identical to the v1 processor so v3 coordinates are
# directly comparable with the v1/v2 ``starmap_visual_cortex`` dataset.
# Leica TCS SP5, HC FLUOTAR L 25x/0.95 W at 1024x1024 (Wang et al. 2018);
# lateral confirmed via He et al. 2021 (tissue extent 1545 um = 1800 x 0.859).
VOXEL_XY_UM = 0.859
VOXEL_Z_UM = 1.0

# Marker genes the paper compares spatial expression patterns for (Fig. 2).
MARKER_GENES = ("Flt1", "Pcp4", "Cux2")

# Layer-marker composite used to derive the cortical laminar (depth) axis from
# the ground-truth slice; see ``evaluate_paper.laminar_axis``. Only the genes
# actually present in the panel are used.
LAYER_SUPERFICIAL = ("Cux2", "Rasgrf2", "Nov")
LAYER_DEEP = ("Pcp4", "Ctgf", "Sema3e")

# ── Marker panels for the non-cortex datasets ────────────────────────────────
# ``MARKER_GENES`` and the LAYER_* composites above are mouse *visual cortex*:
# they name the genes the paper compares, and the superficial/deep pair defines
# the laminar axis. Neither transfers to a tissue without cortical layers, and
# the README is explicit that a new dataset needs its own chosen before
# ``paper_marker_*`` means anything. These are those choices, one per tissue.
#
# The rule each pair follows is the one the IMC entry established: pick two
# compartments the tissue genuinely separates in space, put them at opposite
# poles, and let ``evaluate_paper.laminar_axis`` derive the axis from the ground
# truth as the in-plane gradient of z(superficial) - z(deep). What the axis
# *means* changes per tissue; how it is computed does not.

# Whole mouse brain (Allen MERFISH / Zhuang ABCA panels). The axis that survives
# at whole-brain scale is grey vs. white matter: Slc17a7 marks excitatory
# neurons (grey), Mbp/Plp1 the myelinated tracts. So paper_marker_depth_r reads
# as "profile across the grey-white axis".
# Chosen against the *panels*, not from a textbook. The Zhuang ABCA panel carries
# Gad2 but not Gad1, and no Mbp at all, so the original pick (Slc17a7/Gad1/Mbp)
# resolved to a single marker there and the metric group was measuring one gene:
#     WARNING: no channel matched 'Gad1' — closest panel names: Gja1, Gad2, ...
#     WARNING: no channel matched 'Mbp'  — closest panel names: Pmfbp1, Mybpc1
# These three are present in the ABCA panel and in the whole-transcriptome
# datasets, so every brain dataset resolves the same three and their rows stay
# comparable. Extra fallbacks follow the primary in each layer list;
# ``resolve_markers`` keeps whichever the panel actually has.
BRAIN_MARKER_GENES = ("Slc17a7", "Gad2", "Sox10")
BRAIN_LAYER_SUPERFICIAL = ("Slc17a7", "Gad2", "Gad1")   # neuronal / grey pole
BRAIN_LAYER_DEEP = ("Sox10", "Plp1", "Mbp", "Mog")      # myelin / white pole

# Mouse hypothalamus (3-D MERFISH thick tissue, EASI-FISH LHA). Neuropeptidergic
# nuclei are the spatial structure here; the axis contrasts the two best-separated
# populations rather than a laminar depth.
HYPOTHALAMUS_MARKER_GENES = ("Gad1", "Slc17a6", "Nts")
HYPOTHALAMUS_LAYER_SUPERFICIAL = ("Slc17a6", "Nts")   # glutamatergic pole
HYPOTHALAMUS_LAYER_DEEP = ("Gad1", "Gad2", "Slc32a1")  # GABAergic pole

# Human tumour panels (ExSeq breast cancer). Same tumour/immune contrast the
# IMC, CosMx and Open-ST entries use, so the four tumour datasets read alike.
TUMOUR_MARKER_GENES = ("EPCAM", "PTPRC", "COL1A1")
TUMOUR_LAYER_SUPERFICIAL = ("PTPRC",)             # immune pole
TUMOUR_LAYER_DEEP = ("EPCAM", "KRT8", "KRT18")    # epithelial / tumour pole

# STARmap is a single 3-D imaging block: the z-planes are inherently
# co-registered, so the training-only re-registration is the identity. (v2 makes
# the same call for every volumetric dataset — re-registering would only
# introduce distortion.)
REGISTRATION = "none"

RANDOM_SEED = 42


# ── Datasets ──────────────────────────────────────────────────────────────────
# STARmap is the paper's dataset and the reference implementation of the
# protocol. Anything else is the *same protocol applied to another volume* — a
# protocol analogue, not part of the published result — and says so in
# ``kind``. Report the two accordingly.
#
# Two partition modes, because volumes come in two shapes:
#   "planes"  discrete optical planes (STARmap: integer z = 6..94). The trim is
#             stated in plane indices and the planes are split into equal blocks,
#             so section boundaries land exactly where the paper puts them.
#   "z_width" continuous z in um (ExSeq: every cell has its own float z, so
#             "unique planes" is meaningless). The z range is cut into equal-width
#             slabs — the same idea, expressed in the only unit available.
#   "sections" real serial sections, already sections; a consecutive window is kept.
#
# Four optional keys carry the differences the later datasets introduced:
#
#   "coords"        which coordinate array to believe. Default "auto" is
#                   ``prepare_dataset.extract_xyz``'s own priority, which prefers
#                   obs['x','y','z']. That is right for STARmap and wrong for the
#                   Allen atlases, whose metadata carries *both* section-local
#                   obs x/y and the reconstructed CCF in obsm['spatial'] — and
#                   picking the former would silently score a stack of unrelated
#                   in-section frames. Those specs say "obsm".
#   "resolution"    "single_cell" (default) or "spot". Spot arrays are not the
#                   resolution this protocol was designed on: a Visium or ST spot
#                   pools tens of cells, so paper_celltype_localization measures
#                   deconvolved composition rather than cells, and the binned
#                   marker field is already spot-binned before v3 bins it. They
#                   are registered because they are usable and comparable *among
#                   themselves*, and flagged so their rows are read separately —
#                   exactly as ``kind`` separates the paper dataset from analogues.
#   "n_hvg"         cap the gene panel at build time (highly variable genes, with
#                   the dataset's markers force-included). None = keep the panel.
#                   Needed for the whole-transcriptome datasets: at ~20-30k genes
#                   the per-gene Moran's/Geary's families average mostly zeros,
#                   and a method that densifies the matrix cannot run at all.
#   "max_cells_per_section"
#                   cap cells per section by stratified spatial subsample.
#                   None = keep them all. Needed for the Allen atlases, which run
#                   to millions of cells per volume.
#
# Both caps are applied when the dataset is *built*, so the ground truth and
# every method's input are the same cells and the same genes. Capping only the
# method input would bias the evaluation.
DATASET_SPECS = {
    "starmap_visual_cortex": {
        "kind": "paper",
        "technology": "STARmap",
        "species": "mouse",
        "tissue": "visual cortex",
        "source": "Wang et al. 2018 Science; starmapresources.org",
        "env_var": "BENCH_V3_RAW_STARMAP",
        "raw_candidates": RAW_STARMAP_CANDIDATES,
        "partition": "planes",
        "source_units": "auto",     # obs x/y/z are voxel indices; obsm may be um
        "drop_z_low": DROP_Z_LOW,
        "drop_z_high": DROP_Z_HIGH,
        "voxel_xy_um": VOXEL_XY_UM,
        "voxel_z_um": VOXEL_Z_UM,
        # Both pinned: 7 sections and the 2/4/6 split are the published design,
        # not a v3 preference. Only this dataset is constrained that way.
        "n_sections": N_SECTIONS,
        "held_out": HELD_OUT_SECTION_INDICES,
        "registration": "none",
        "marker_genes": MARKER_GENES,
        "layer_superficial": LAYER_SUPERFICIAL,
        "layer_deep": LAYER_DEEP,
        "protocol": "SpatialZ paper (7 consecutive sections, hold out 2/4/6)",
    },
    "exseq_visual_cortex": {
        "kind": "analogue",
        "technology": "ExSeq",
        "species": "mouse",
        "tissue": "visual cortex",
        "source": "Alon et al. 2021 Science (spacejam2); via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_EXSEQ",
        # Either source works. The raw spacejam2 distribution is read directly by
        # ``sources.read_exseq_csv`` (a cell-by-gene CSV in micrometres plus the
        # SpaceTx EDV annotations), so v1's pipeline does not have to have been
        # run; v1's processed h5ad is the same content and is accepted too.
        "reader": "exseq_csv",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "exseq_visual_cortex",
            V1_ROOT / "data" / "raw" / "exseq_visual_cortex" / "spacejam2_cellxgene.csv",
            V1_ROOT / "data" / "processed" / "exseq_visual_cortex" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "exseq_visual_cortex.h5ad",
        ),
        "partition": "z_width",
        # v1's ExSeq processor writes micrometres into obsm['spatial'] but records
        # no coordinate_units, so state it here rather than let detection guess.
        "source_units": "um",
        # NOT the paper's trim. The paper drops STARmap's noisiest planes from its
        # own analysis of that volume; nothing equivalent has been established for
        # ExSeq, so no noise trim is applied here. What remains is a much smaller
        # thing with a different job: equal-width binning takes its edges from
        # min(z) and max(z), so a handful of segmentation outliers at the extremes
        # would stretch the range and skew all seven slab boundaries. Clipping
        # 0.2 % of cells at each end makes the binning robust to that while
        # keeping essentially the whole volume. Set 0, or pass --no-trim, to use
        # the raw min/max.
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,          # already micrometres
        "voxel_z_um": 1.0,
        # A continuous volume, so the slab count is a free choice rather than a
        # given. 7 over this ~76 um extent puts each slab near 11 um — the
        # thickness of a real cryosection, and close to STARmap's 11-plane
        # sections, which keeps the two cortex datasets comparable. It is a
        # judgement call, not a requirement: --n-sections changes it.
        "n_sections": 7,
        "held_out": "alternate",
        # Same tissue as STARmap, so the cortical markers and the laminar-axis
        # composite carry over unchanged — whichever of them the ExSeq panel
        # actually contains. ``prepare_dataset`` records the intersection in
        # uns['paper_protocol'] and the evaluator uses that, so a gene the panel
        # lacks is never silently scored.
        "marker_genes": MARKER_GENES,
        "layer_superficial": LAYER_SUPERFICIAL,
        "layer_deep": LAYER_DEEP,
        # One 3-D imaging volume, like STARmap: the z-planes are inherently
        # co-registered and re-registering would only distort.
        "registration": "none",
        "protocol": "SpatialZ protocol applied to ExSeq (analogue, not the paper)",
    },
    "imc_breast_cancer": {
        "kind": "analogue",
        "technology": "3D IMC",
        "species": "human",
        "tissue": "breast cancer (HER2+)",
        "source": "Kuett et al. 2022, Zenodo 10.5281/zenodo.4752030",
        "env_var": "BENCH_V3_RAW_IMC",
        # Raw = one h5ad per serial section (MainHer2BreastCancerModel_zstep10_*),
        # read by ``sources.read_imc_zstack``; v1's processed file also works.
        "reader": "imc_zstack",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "imc_breast_cancer",
            V1_ROOT / "data" / "processed" / "imc_breast_cancer" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "imc_breast_cancer.h5ad",
        ),
        "source_units": "um",
        # The source block is a known size; the build checks it (see
        # prepare_dataset) so a reader that silently drops sections cannot
        # turn this into a smaller, differently-split experiment.
        "expected_sections": 15,
        # 15 physically cut sections at 10 um, and each one already *is* a
        # section, so all of them are used: ``n_sections: None`` means "however
        # many the data has". There is no reason to discard two thirds of a
        # serial block to match STARmap's count — the design that matters is the
        # alternating hold-out, which scales to any number of sections. Set
        # --n-sections to take a smaller window (then --section-trim picks which).
        "partition": "sections",
        "n_sections": None,
        "section_trim": "center",
        "held_out": "alternate",
        # NOT a single imaging volume: serial sections are cut, mounted and imaged
        # independently, so they are not co-registered and interpolation is
        # ill-posed without aligning them. Training-only rigid ICP, exactly what v2
        # does for this dataset — the first v3 dataset where registration is not
        # the identity.
        "registration": "rigid",
        # A protein panel, so the cortex genes do not apply. panCK (tumour /
        # epithelial compartment) and CD3 (T-cell infiltrate) follow the published
        # analysis of this volume. Name matching ignores case and punctuation
        # (``prepare_dataset.resolve_markers``) so panCK / PanCK / pan-CK all
        # resolve to the panel's own spelling — but it is an exact match after
        # canonicalization, never a prefix, so CD3 cannot capture CD31.
        "marker_genes": ("panCK", "CD3"),
        # A tumour has no laminar axis, but it does have a *compartment* axis, and
        # these two markers define it: the signed score is z(CD3) - z(panCK), so
        # its in-plane gradient points from tumour toward immune infiltrate. That
        # makes paper_marker_depth_r "profile across the tumour-immune axis" —
        # meaningful, and derived from the ground truth like the cortical version.
        # Emptying both lists reverts to the generic fallback (the gradient of
        # whichever channel is most spatially structured).
        "layer_superficial": ("CD3",),      # immune pole
        "layer_deep": ("panCK",),           # tumour pole
        "protocol": "SpatialZ protocol applied to 3D IMC (analogue, not the paper)",
    },
    "cosmx_nsclc_3d": {
        "kind": "analogue",
        "technology": "CosMx SMI",
        "species": "human",
        "tissue": "NSCLC tumour",
        "source": "Pentimalli et al. 2025 Cell Systems; Zenodo 15240431",
        "env_var": "BENCH_V3_RAW_COSMX",
        # The raw distribution is read directly: the shipped h5ad has expression
        # and cell types but STIM coordinates in arbitrary units, so
        # ``sources.read_cosmx_raw`` joins it to the per-section flat files for
        # physical positions. v1's processed h5ad is accepted too.
        "reader": "cosmx_raw",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "cosmx_nsclc_3d",
            V1_ROOT / "data" / "processed" / "cosmx_nsclc_3d" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "cosmx_nsclc_3d.h5ad",
        ),
        "source_units": "um",
        # The source block is a known size; the build checks it (see
        # prepare_dataset) so a reader that silently drops sections cannot
        # turn this into a smaller, differently-split experiment.
        "expected_sections": 6,
        # Six real cryosections, 30 um apart (every 6th 5-um section). All of them
        # are used; "alternate" then holds out 2 and 4, leaving 1/3/5/6 as input.
        # That 30 um gap is the widest in the benchmark and the reason this dataset
        # is worth having: it is a genuinely hard interpolation, where STARmap's
        # 11 um spacing leaves a copy baseline near ceiling.
        "partition": "sections",
        "n_sections": None,
        "section_trim": "center",
        "held_out": "alternate",
        # Serial cryosections, imaged independently and not cross-registered —
        # v2 classifies this dataset "not-aligned" for exactly that reason.
        "registration": "rigid",
        # A 960-plex panel on tumour tissue: the same tumour/immune contrast the
        # IMC dataset uses, in RNA. EPCAM marks the epithelial tumour compartment,
        # PTPRC (CD45) the immune infiltrate, and their signed difference gives a
        # tumour-immune axis for paper_marker_depth_r.
        "marker_genes": ("EPCAM", "PTPRC"),
        "layer_superficial": ("PTPRC",),     # immune pole
        "layer_deep": ("EPCAM",),            # tumour pole
        "protocol": "SpatialZ protocol applied to CosMx serial sections (analogue)",
    },
    "deep_starmap": {
        "kind": "analogue",
        "technology": "Deep-STARmap",
        "species": "mouse",
        "tissue": "brain (thick blocks)",
        "source": "Sui et al. 2025 Nat Methods; Zenodo 10.5281/zenodo.16783354",
        "env_var": "BENCH_V3_RAW_DEEP_STARMAP",
        "reader": "deep_starmap_csv",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "deep_starmap",
            V1_ROOT / "data" / "processed" / "deep_starmap" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "deep_starmap.h5ad",
        ),
        "source_units": "um",
        # A dense optical volume like STARmap, not serial sections: 0.70 um
        # z-steps, so the planes must be grouped into slabs. No published trim
        # exists for it, so none is applied (drop ranges are None) — the section
        # count is the only choice, and 7 is a starting point the cells/section
        # guard will reject if the block is too shallow to support it.
        "partition": "planes",
        "drop_z_low": None,
        "drop_z_high": None,
        "voxel_xy_um": 0.32,
        "voxel_z_um": 0.70,
        "n_sections": 7,
        "held_out": "alternate",
        # One 3-D imaging block, inherently co-registered.
        "registration": "none",
        # Mouse brain, so the cortex panel is the right family to try; whichever
        # of these the FUSEmap panel carries is used, and the build warns about
        # the rest rather than scoring an empty set.
        "marker_genes": MARKER_GENES,
        "layer_superficial": LAYER_SUPERFICIAL,
        "layer_deep": LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to Deep-STARmap (analogue)",
    },
    "merfish_hypothalamus": {
        "kind": "analogue",
        "technology": "MERFISH",
        "species": "mouse",
        "tissue": "hypothalamus (preoptic region)",
        "source": "Moffitt et al. 2018 Science; Dryad",
        "env_var": "BENCH_V3_RAW_MERFISH_HYPO",
        "reader": "merfish_hypothalamus_csv",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "merfish_hypothalamus",
            V1_ROOT / "data" / "processed" / "merfish_hypothalamus" / "animal_1" / "data.h5ad",
        ),
        "source_units": "um",
        # The source block is a known size; the build checks it (see
        # prepare_dataset) so a reader that silently drops sections cannot
        # turn this into a smaller, differently-split experiment.
        "expected_sections": 12,
        # 12 real sections spanning Bregma -0.29..+0.26 mm, 50 um apart — five
        # times STARmap's spacing, so a copy baseline has much further to fall.
        # One animal only: sections from different animals are different tissue.
        "partition": "sections",
        "n_sections": None,
        "section_trim": "center",
        "held_out": "alternate",
        # Independently cut and imaged coronal sections, not one imaging block.
        "registration": "rigid",
        # Hypothalamic preoptic region. Gad1/Slc17a6 separate the inhibitory and
        # excitatory populations, which is this tissue's dominant spatial
        # contrast, and their difference gives the depth axis. Mbp marks the
        # fibre tracts. All are in the 155-gene panel.
        "marker_genes": ("Gad1", "Slc17a6", "Mbp"),
        "layer_superficial": ("Slc17a6",),    # excitatory pole
        "layer_deep": ("Gad1",),              # inhibitory pole
        "protocol": "SpatialZ protocol applied to MERFISH hypothalamus (analogue)",
    },
    "openst_lymph_node": {
        "kind": "analogue",
        "technology": "Open-ST",
        "species": "human",
        "tissue": "metastatic lymph node",
        "source": "Schott et al. 2024 Cell; GEO GSE251926",
        "env_var": "BENCH_V3_RAW_OPENST",
        "reader": "openst_lymph_node",
        "raw_candidates": (
            V1_ROOT / "data" / "raw" / "openst_lymph_node",
            V1_ROOT / "data" / "processed" / "openst_lymph_node" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "openst_lymph_node.h5ad",
        ),
        "source_units": "um",
        # The source block is a known size; the build checks it (see
        # prepare_dataset) so a reader that silently drops sections cannot
        # turn this into a smaller, differently-split experiment.
        "expected_sections": 19,
        # 19 consecutive 10 um cryosections through one metastatic lymph node.
        "partition": "sections",
        "n_sections": None,
        "section_trim": "center",
        "held_out": "alternate",
        "registration": "rigid",
        # Tumour metastasis in lymphoid tissue: EPCAM marks the metastatic
        # epithelial deposit, PTPRC the lymphocyte background — the same
        # tumour/immune contrast used for CosMx and IMC, so the three tumour
        # datasets are read the same way.
        "marker_genes": ("EPCAM", "PTPRC"),
        "layer_superficial": ("PTPRC",),
        "layer_deep": ("EPCAM",),
        "protocol": "SpatialZ protocol applied to Open-ST serial sections (analogue)",
    },

    # ── Allen whole-brain MERFISH atlases ────────────────────────────────────
    # Three registered volumes from the Allen Brain Cell atlas. All three share a
    # shape the earlier datasets did not have: dozens of serial sections, order
    # 10^6 cells, and coordinates the *provider* reconstructed into the Allen CCF
    # using every section — including the ones v3 holds out.
    #
    # That last point is leakage vector 2 from v2's table, and it is why the
    # policy here is "rigid" rather than "none": the training slices are put into
    # a common frame using training slices only, replacing the upstream
    # reconstruction. It costs alignment quality (the CCF fit is better than
    # anything training-only ICP will find) and it is the only correct choice —
    # keeping the provider's frame would hand every method a registration that
    # saw the answer.
    "allen_merfish_brain": {
        "kind": "analogue",
        "technology": "MERFISH",
        "species": "mouse",
        "tissue": "whole brain",
        "source": "Allen Brain Cell Atlas, MERFISH-C57BL6J-638850; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_ALLEN_MERFISH",
        "reader": "allen_ccf",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "allen_merfish_brain" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "allen_merfish_brain.h5ad",
            V1_ROOT / "data" / "raw" / "allen_merfish_brain",
        ),
        # v1 converts the CCF mm to um and writes obsm['spatial']; it records no
        # coordinate_units, so state it rather than let detection guess.
        "source_units": "um",
        # The metadata CSV carries section-local obs x/y alongside the CCF — see
        # the "coords" note above. Believe obsm.
        "coords": "obsm",
        "partition": "sections",
        "n_sections": None,
        "expected_sections": 59,
        "section_trim": "center",
        "held_out": "alternate",
        "registration": "rigid",
        "n_hvg": None,              # ~500-gene panel; nothing to cap
        "max_cells_per_section": 20000,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to Allen MERFISH whole brain (analogue)",
    },
    "allen_zhuang_abca1": {
        "kind": "analogue",
        "technology": "MERFISH",
        "species": "mouse",
        "tissue": "whole brain",
        "source": "Allen Brain Cell Atlas, Zhuang-ABCA-1; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_ZHUANG1",
        "reader": "allen_ccf",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "allen_zhuang_merfish" / "Zhuang-ABCA-1" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "allen_zhuang_merfish" / "Zhuang-ABCA-1.h5ad",
            V1_ROOT / "data" / "raw" / "allen_zhuang_merfish",
        ),
        "reader_region": "Zhuang-ABCA-1",
        "source_units": "um",
        "coords": "obsm",
        "partition": "sections",
        # Unlike the Allen 638850 block this one has no single documented section
        # count in the form v3 checks (the parcellation is delivered per region),
        # so the count comes from the data and there is no expected_sections net.
        "n_sections": 15,
        "section_trim": "center",
        "held_out": "alternate",
        "registration": "rigid",
        "n_hvg": None,              # ~1100-gene panel
        "max_cells_per_section": 20000,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to Zhuang-ABCA-1 (analogue)",
    },
    "allen_zhuang_abca2": {
        "kind": "analogue",
        "technology": "MERFISH",
        "species": "mouse",
        "tissue": "whole brain",
        "source": "Allen Brain Cell Atlas, Zhuang-ABCA-2; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_ZHUANG2",
        "reader": "allen_ccf",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "allen_zhuang_merfish" / "Zhuang-ABCA-2" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "allen_zhuang_merfish" / "Zhuang-ABCA-2.h5ad",
            V1_ROOT / "data" / "raw" / "allen_zhuang_merfish",
        ),
        "reader_region": "Zhuang-ABCA-2",
        "source_units": "um",
        "coords": "obsm",
        "partition": "sections",
        "n_sections": 15,
        "section_trim": "center",
        "held_out": "alternate",
        "registration": "rigid",
        "n_hvg": None,
        "max_cells_per_section": 20000,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to Zhuang-ABCA-2 (analogue)",
    },

    # ── 3-D MERFISH thick tissue (Fang et al. 2023 eLife) ────────────────────
    # Two independent thick-tissue blocks imaged as consecutive optical planes —
    # the same shape as STARmap, so partition by z and register with "none".
    # v1 writes one h5ad per region under data/processed/merfish_thick_tissue/.
    "merfish_thick_cortex": {
        "kind": "analogue",
        "technology": "3D MERFISH",
        "species": "mouse",
        "tissue": "cortex (100 um thick block)",
        "source": "Fang et al. 2023 eLife; Dryad 10.5061/dryad.w0vt4b922",
        "env_var": "BENCH_V3_RAW_MERFISH_THICK_CORTEX",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "merfish_thick_tissue" / "cortex" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "merfish_thick_tissue" / "cortex.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        # Same outlier guard as ExSeq, and for the same reason: equal-width bins
        # take their edges from min/max z, so a few segmentation outliers would
        # skew all the slab boundaries. Not a noise trim.
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        # A ~100 um block into 7 slabs is ~14 um each — a cryosection's thickness,
        # and the same choice made for ExSeq and Deep-STARmap.
        "n_sections": 7,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,              # 242-gene panel
        "max_cells_per_section": None,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to 3D MERFISH cortex (analogue)",
    },
    "merfish_thick_hypothalamus": {
        "kind": "analogue",
        "technology": "3D MERFISH",
        "species": "mouse",
        "tissue": "hypothalamus (200 um thick block)",
        "source": "Fang et al. 2023 eLife; Dryad 10.5061/dryad.w0vt4b922",
        "env_var": "BENCH_V3_RAW_MERFISH_THICK_HYPO",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "merfish_thick_tissue" / "hypothalamus" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "merfish_thick_tissue" / "hypothalamus.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        # 200 um into 7 slabs is ~29 um each — thicker than the others, which is
        # the point: it widens the gap range the benchmark covers.
        "n_sections": 7,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,              # 156-gene panel
        "max_cells_per_section": None,
        "marker_genes": HYPOTHALAMUS_MARKER_GENES,
        "layer_superficial": HYPOTHALAMUS_LAYER_SUPERFICIAL,
        "layer_deep": HYPOTHALAMUS_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to 3D MERFISH hypothalamus (analogue)",
    },

    # ── EASI-FISH lateral hypothalamus ───────────────────────────────────────
    # Three *independent* tissue samples with overlapping z ranges, so each is its
    # own volume and they must never be concatenated. v1 collapses each sample to
    # a single obs['section'] but keeps the real per-cell z, so v3 partitions by
    # z_width and ignores that column.
    #
    # The panel is small (a handful of genes). That is a real limitation rather
    # than a detail: the autocorrelation families correlate Moran's I *across
    # genes*, so with under ~10 genes those correlations rest on very few points
    # and should be read as indicative. The distributional and localization
    # families are unaffected.
    "easi_fish_lha1": {
        "kind": "analogue",
        "technology": "EASI-FISH",
        "species": "mouse",
        "tissue": "lateral hypothalamus (LHA1)",
        "source": "Wang et al. 2021 Cell; Figshare 13749154; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_EASIFISH_LHA1",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA1" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA1.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        "n_sections": 7,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,
        "max_cells_per_section": None,
        "marker_genes": HYPOTHALAMUS_MARKER_GENES,
        "layer_superficial": HYPOTHALAMUS_LAYER_SUPERFICIAL,
        "layer_deep": HYPOTHALAMUS_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to EASI-FISH LHA1 (analogue)",
    },
    "easi_fish_lha2": {
        "kind": "analogue",
        "technology": "EASI-FISH",
        "species": "mouse",
        "tissue": "lateral hypothalamus (LHA2)",
        "source": "Wang et al. 2021 Cell; Figshare 13749154; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_EASIFISH_LHA2",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA2" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA2.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        "n_sections": 7,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,
        "max_cells_per_section": None,
        "marker_genes": HYPOTHALAMUS_MARKER_GENES,
        "layer_superficial": HYPOTHALAMUS_LAYER_SUPERFICIAL,
        "layer_deep": HYPOTHALAMUS_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to EASI-FISH LHA2 (analogue)",
    },
    "easi_fish_lha3": {
        "kind": "analogue",
        "technology": "EASI-FISH",
        "species": "mouse",
        "tissue": "lateral hypothalamus (LHA3)",
        "source": "Wang et al. 2021 Cell; Figshare 13749154; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_EASIFISH_LHA3",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA3" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "easi_fish_hypothalamus" / "LHA3.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        "n_sections": 7,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,
        "max_cells_per_section": None,
        "marker_genes": HYPOTHALAMUS_MARKER_GENES,
        "layer_superficial": HYPOTHALAMUS_LAYER_SUPERFICIAL,
        "layer_deep": HYPOTHALAMUS_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to EASI-FISH LHA3 (analogue)",
    },

    # ── ExSeq breast cancer ──────────────────────────────────────────────────
    # A true 3-D volume (transcript-level globalpos, 297 genes) but a *small* one:
    # ~3100 cells total, so 7 sections leave a few hundred each. That clears the
    # 50-cell floor the build enforces but sits under the 500/section the survey
    # calls comfortable, so its alignment-dependent metrics will be the noisiest
    # in the benchmark. Registered because a small human tumour volume is exactly
    # the breadth the panel lacks; read its rows with that caveat.
    "exseq_breast_cancer": {
        "kind": "analogue",
        "technology": "ExSeq",
        "species": "human",
        "tissue": "breast cancer",
        "source": "Alon et al. 2021 Science; Zenodo; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_EXSEQ_BC",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "exseq_breast_cancer" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "exseq_breast_cancer.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "z_width",
        "z_trim_quantile": 0.002,
        "voxel_xy_um": 1.0,
        "voxel_z_um": 1.0,
        # 5, not 7. The volume is small enough that seven sections would leave
        # ~440 cells each; five keeps them ~620 and the alternating hold-out still
        # brackets every held-out section (2 and 4 held out, 1/3/5 input).
        "n_sections": 5,
        "held_out": "alternate",
        "registration": "none",
        "n_hvg": None,              # 297-gene panel
        "max_cells_per_section": None,
        "marker_genes": TUMOUR_MARKER_GENES,
        "layer_superficial": TUMOUR_LAYER_SUPERFICIAL,
        "layer_deep": TUMOUR_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to ExSeq breast cancer (analogue)",
    },

    # ── Spot-resolution arrays ───────────────────────────────────────────────
    # Registered with resolution="spot". See the "resolution" note at the top of
    # this table for why they are separated rather than pooled: a spot is not a
    # cell, so paper_celltype_localization scores deconvolved composition and the
    # marker field is pre-binned by the array geometry. Everything mechanical
    # works; the metrics simply mean something adjacent.
    "st_mouse_brain_ortiz": {
        "kind": "analogue",
        "resolution": "spot",
        "technology": "ST",
        "species": "mouse",
        "tissue": "whole brain",
        "source": "Ortiz et al. 2020 Sci Adv; GEO GSE147747; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_ORTIZ",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "st_mouse_brain_ortiz" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "st_mouse_brain_ortiz.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "sections",
        # The labels are '01A'/'01B' ... '40A'/'40B': A and B are replicate
        # sections at ONE anterior-posterior coordinate, so after registration
        # into the Allen frame each pair sits at the same z. Two sections at the
        # same depth are not a stack, and the build stops on it. They are
        # physically one plane sampled twice, so they are merged rather than
        # dropped — nothing is lost and the depth axis becomes strictly
        # increasing, which the protocol requires.
        "section_z_tolerance": 1.0,        # um
        # 75 source sections from 3 animals interleaved into one
        # anterior-posterior atlas, ~38 distinct depths after merging. A centred
        # window of 15 keeps the design comparable with the Allen entries instead
        # of producing a hold-out nothing else can be read against.
        "n_sections": 15,
        "section_trim": "center",
        "held_out": "alternate",
        # Authors registered every section to the Allen atlas using all sections,
        # so the same argument as the Allen atlases applies: re-register
        # training-only.
        "registration": "rigid",
        # Whole transcriptome — this is the cap that matters. Without it the
        # built file is ~30k genes and the densifying wrappers cannot load it.
        "n_hvg": 3000,
        "max_cells_per_section": None,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to ST mouse brain, spot resolution (analogue)",
    },
    "visium_mouse_brain_c2l": {
        "kind": "analogue",
        "resolution": "spot",
        "technology": "Visium",
        "species": "mouse",
        "tissue": "brain (coronal)",
        "source": "Kleshchevnikov et al. 2022 Nat Biotech; E-MTAB-11114; via benchmark-pbya",
        "env_var": "BENCH_V3_RAW_VISIUM_C2L",
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "visium_mouse_brain_cell2location" / "mouse_1" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "visium_mouse_brain_cell2location" / "mouse_1.h5ad",
        ),
        "source_units": "um",
        "coords": "obsm",
        "partition": "sections",
        # Mouse 1 contributes 3 serial coronal sections at 210 um. Three is the
        # minimum the alternating design can express: hold out section 2, keep 1
        # and 3 as input. It is one held-out section per run, so its numbers carry
        # no within-dataset spread — the thinnest experiment in the benchmark, and
        # included for the technology rather than for its discriminating power.
        "n_sections": 3,
        "section_trim": "center",
        "held_out": "alternate",
        "registration": "rigid",
        "n_hvg": 3000,              # whole transcriptome
        "max_cells_per_section": None,
        "marker_genes": BRAIN_MARKER_GENES,
        "layer_superficial": BRAIN_LAYER_SUPERFICIAL,
        "layer_deep": BRAIN_LAYER_DEEP,
        "protocol": "SpatialZ protocol applied to Visium mouse brain, spot resolution (analogue)",
    },
}


def held_out_indices(dataset_spec, n_sections):
    """1-based indices of the held-out sections for a dataset of ``n_sections``.

    ``held_out`` is either an explicit tuple — STARmap pins (2, 4, 6) because that
    is the published split — or the rule ``"alternate"``: hold out every even
    section, keeping the first and last as input. At n = 7 the rule reproduces the
    paper exactly (2, 4, 6); at n = 15 it gives 2, 4, ..., 14. Either way every
    held-out section is bracketed by two input sections, which is what makes the
    task well-posed for an interpolation method, and roughly half the volume is
    missing, which is what makes it hard.
    """
    rule = dataset_spec.get("held_out", "alternate")
    if isinstance(rule, str):
        if rule != "alternate":
            raise ValueError(f"unknown held_out rule {rule!r}")
        return tuple(range(2, int(n_sections), 2))
    return tuple(int(i) for i in rule)


def spec(dataset_name=DATASET_NAME):
    """Protocol spec for a dataset."""
    try:
        return DATASET_SPECS[dataset_name]
    except KeyError:
        raise ValueError(f"unknown dataset {dataset_name!r}; known: "
                         f"{sorted(DATASET_SPECS)}") from None


def dataset_path(dataset_name=DATASET_NAME):
    """Where ``prepare_dataset`` writes (and everything else reads) a dataset."""
    return DATA_DIR / dataset_name / "data.h5ad"


def resolve_raw(dataset_name=DATASET_NAME):
    """First existing source for a dataset, else the preferred path."""
    s = spec(dataset_name)
    env = os.environ.get(s["env_var"])
    if env:
        return Path(env)
    for cand in s["raw_candidates"]:
        if Path(cand).exists():
            return Path(cand)
    return Path(s["raw_candidates"][0])


DATASETS = {name: {**s, "path": dataset_path(name)}
            for name, s in DATASET_SPECS.items()}


def resolve_dataset_arg(value, default=None):
    """Accept either a registered dataset NAME or a path to a built ``data.h5ad``.

    ``prepare_dataset --dataset`` takes a name while the run/evaluate/plot stages
    historically took a path, so passing the name to the latter is the obvious
    mistake — and it used to fail with a "not found" that pointed at building a
    dataset that may well already exist. Both spellings now work.
    """
    if value is None:
        return Path(default) if default is not None else DATASET_PATH
    text = str(value)
    if text in DATASET_SPECS:
        return dataset_path(text)
    return Path(text)


def dataset_not_found_message(value, resolved):
    """Actionable text for a --dataset that does not resolve to an existing file."""
    lines = [f"dataset not found: {resolved}"]
    name = str(value) if value is not None else DATASET_NAME
    if name in DATASET_SPECS:
        lines.append(f"  read {name!r} as a registered dataset name")
        lines.append(f"  build it:  python -m src.bench3.prepare_dataset "
                     f"--dataset {name}")
    elif value is not None:
        lines.append(f"  {name!r} is neither a registered dataset name nor an "
                     f"existing file")
    lines.append("")
    lines.append("  registered datasets:")
    for n in DATASET_SPECS:
        p = dataset_path(n)
        lines.append(f"    {n:<24s} {'built' if p.exists() else 'NOT built'}  {p}")
    return "\n".join(lines)


# ── Methods ───────────────────────────────────────────────────────────────────
# Generation-only, exactly as in v2: each method receives a training-only,
# re-registered input plus a scalar target z per held-out section, and
# synthesizes the slice de novo.
#
# ``wrapper`` points into benchmark-pbya-v2 on purpose. The wrappers are plain
# CLI scripts over the stable ``methods/_v2_io.py`` contract, so reusing them
# (rather than forking) guarantees v3 and v2 exercise the *same* method code.
#
# The stable part of that contract is the ``_v2_io`` argument set (``--input``,
# ``--target-section``, ``--target-z``, ``--output``, ``--seed``) — *not* the
# per-wrapper ablation flags, whose defaults track whichever variant v2 was last
# tuning. ``wrapper_args`` therefore lets a method pin the configuration v3 runs
# it in, so a v3 run is reproducible from this file and cannot silently change
# (or break) when a v2 default moves. ``run_benchmark`` appends these after the
# shared arguments and before any user-supplied extras, so an explicit extra on
# the command line still wins.
#
# Two more per-method fields keep v3 self-sufficient, so that making a method run
# correctly here never requires editing benchmark-pbya-v2 or a method package:
#   ``sanitize_weight_args``  — CLI flags whose value is a torch checkpoint. v3
#       checks each one loads under this torch and substitutes a tensors-only copy
#       when it does not (``assets.py``).
#   ``invalid_log_markers``   — strings whose presence in the method log means the
#       run silently degraded to a fallback. v3 fails the run instead of scoring
#       it. Log matching is a blunt instrument, but it is the only signal a method
#       that "succeeds" by design exposes across the subprocess boundary.
def _v2_wrapper(name):
    return V2_ROOT / "src" / "benchmark" / "methods" / name


def _v3_wrapper(name):
    """A wrapper that lives in v3 because v2 does not have the method.

    Every wrapper v2 already has is invoked from v2 (see the README): sharing
    them is what guarantees v3 and v2 exercise the same synthesis code. A method
    v2 has never run has nothing to share, and adding it to v2 would mean editing
    a frozen benchmark, so its wrapper lives here. It speaks the identical
    ``_v2_io`` contract — same CLI, same prediction format, same leakage guard —
    so ``run_benchmark`` cannot tell the difference.
    """
    return PROJECT_ROOT / "src" / "bench3" / "methods" / name


METHODS = {
    "spatialcpav8_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav8.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "training-free symmetric entropic-OT / McCann bridge",
        # Pin v8's production configuration — the defaults of
        # ``SpatialCPAv8Config`` (placement ``smooth_morph``: the coherent
        # smoothed-OT deformation of the nearest clean slice; expression
        # ``endpoint``: the source cell's real profile). The v2 wrapper now
        # defaults to the same pair, so this pin is currently a no-op; it stays
        # so that a v3 run remains reproducible from this file if a v2 default
        # moves again. (It previously defaulted to ``diffeo_morph``/``fusion``,
        # which are v10's modes — the packaged v8 implements neither, so every
        # holdout aborted with ``ValueError: Unknown bridge mode``.)
        "wrapper_args": ["--placement", "smooth_morph",
                         "--expression-mode", "endpoint"],
    },
    "spatialcpav11_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav11.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "teacher-distilled conditional generator",
        # The OmiCLIP teacher loads its checkpoint through torch.load, which since
        # torch 2.6 defaults to weights_only=True and rejects the numpy scalars in
        # a training checkpoint. v3 hands over a tensors-only copy instead (same
        # weights; see assets.py) so the teacher builds.
        "sanitize_weight_args": ("--teacher-weights",),
        # v11 degrades to a deterministic OT-morph layout when training fails —
        # sensible for a library, wrong for a benchmark: the fallback is not v11,
        # yet it still writes a prediction that would be scored and ranked beside
        # the real methods, with numbers plausible enough not to look wrong. If
        # the log says it degraded, v3 fails the run.
        "invalid_log_markers": ("neural fields trained: False", "OT-morph fallback."),
    },
    "spatialcpav14_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav14.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "joint latent flow matching",
    },
    "spatialcpav15_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav15.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "structure-field diffusion + NB expression VAE + marked point process",
    },
    "spatialcpav16_gen": {
        "wrapper": _v3_wrapper("run_spatialcpav16.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "spectral tissue flow — flow matching over graph harmonics, "
                 "with the per-gene autocorrelation profile enforced",
        # v16 degrades to the depth-interpolated spectrum when torch is missing
        # or the flow fails to train. That is a sensible library fallback and an
        # invalid benchmark row: it is not the method under test, so v3 fails
        # the run rather than scoring it.
        "invalid_log_markers": ("spectral flow trained: False",
                                "torch UNAVAILABLE"),
    },
    "spatialcpav17_gen": {
        "wrapper": _v2_wrapper("run_spatialcpav17.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "fully-generative VAE with an NB decoder + latent flow matching "
                 "— the virtual slice is decoded, not retrieved",
        # No ``wrapper_args``: v17's default mode is the one under test (decode;
        # ``--retrieval`` is its ablation, and a store_true flag cannot be pinned
        # off anyway). Which mode actually ran is recorded per prediction in
        # ``method_params['mode']``, so a result is self-describing.
        #
        # v17 falls back three ways — no torch, training raised, generation
        # raised — and each one substitutes a flanking resample-and-copy for the
        # method. That is a reasonable library default and an invalid benchmark
        # row: the copy is close to the ``flanking_copy`` probe the self-test
        # uses as a *baseline*, so a degraded run would not merely be wrong, it
        # would score respectably. v3 fails the run instead.
        "invalid_log_markers": ("model trained: False",
                                "torch UNAVAILABLE",
                                "[spatialcpav17] generation failed"),
    },
    "spatialcpav18_gen": {
        # v18 is the single-file ``learn_spatialcpav18.py`` at the repository
        # root, not a package v2 ever ran, so its wrapper lives here (like v16's)
        # and loads that file directly. It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav18.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v14) + benchmark-driven fixes — raw output, gene-mix "
                 "novelty, kNN type vote, and stable/diverse grounding",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v18 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v18 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate. That is a sensible library
        # default and an invalid benchmark row — the fallback is not the method
        # under test — so v3 fails the run rather than scoring it. The first two
        # markers are the wrapper's own; the last three are printed by
        # learn_spatialcpav18.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav19_gen": {
        # v19 is the single-file ``learn_spatialcpav19.py`` at the repository
        # root (v18 + gap-adaptive generation), not a package v2 ever ran, so its
        # wrapper lives here (like v16's and v18's) and loads that file directly.
        # It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav19.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v18) + gap-adaptive generation — gap-scaled generative "
                 "weight, cross-flank paired interpolation, gap-triggered flow "
                 "blend, and a wide-gap training curriculum",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v19 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v19 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate — the same fallback prints
        # as v18 (the shared v14 core). The fallback is not the method under
        # test, so v3 fails the run rather than scoring it. The first two markers
        # are the wrapper's own; the last three are printed by
        # learn_spatialcpav19.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav20_gen": {
        # v20 is the single-file ``learn_spatialcpav20.py`` at the repository
        # root (v18 + Bernoulli cross-mix, repairing v19's wide-gap collapse),
        # not a package v2 ever ran, so its wrapper lives here (like v16/v18/v19)
        # and loads that file directly. It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav20.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v18) + Bernoulli cross-mix — per-gene real-value "
                 "selection between exemplar and cross-flank partner, keeping "
                 "sparsity/count-ness intact at wide gaps (fixes v19)",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v20 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v20 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate — the same fallback prints
        # as v18/v19 (the shared v14 core). The fallback is not the method under
        # test, so v3 fails the run rather than scoring it. The first two markers
        # are the wrapper's own; the last three are printed by
        # learn_spatialcpav20.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav21_gen": {
        # v21 is the single-file ``learn_spatialcpav21.py`` at the repository
        # root (v20 + coherent mix + field-aligned grounding), not a package v2
        # ever ran, so its wrapper lives here (like v16/v18/v19/v20) and loads
        # that file directly. It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav21.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v20) + coherent mix + field-aligned grounding — smooth "
                 "per-gene mix fields that preserve autocorrelation, gap-adaptive "
                 "re-grounding/layout, and field-aligned exemplar + per-gene repair",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v21 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v21 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate — the same fallback prints
        # as v18/v19/v20 (the shared v14 core). The fallback is not the method
        # under test, so v3 fails the run rather than scoring it. The first two
        # markers are the wrapper's own; the last three are printed by
        # learn_spatialcpav21.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav22_gen": {
        # v22 is the single-file ``learn_spatialcpav22.py`` at the repository
        # root (v21 recalibrated on the real 18-dataset benchmark), not a package
        # v2 ever ran, so its wrapper lives here (like v16/v18/v19/v20/v21) and
        # loads that file directly. It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav22.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v21) recalibrated on the real benchmark — wide-gap "
                 "gene-mix off, capped cross-mix, spacing-relative mix/layout "
                 "granules, and a mild narrow-gap tail repair",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v22 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v22 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate — the same fallback prints
        # as v18-v21 (the shared v14 core). The fallback is not the method under
        # test, so v3 fails the run rather than scoring it. The first two markers
        # are the wrapper's own; the last three are printed by
        # learn_spatialcpav22.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav23_gen": {
        # v23 is the single-file ``learn_spatialcpav23.py`` at the repository
        # root (v20 + module-level chimerism, niche-weighted partners, stratified
        # cross-mix, local composition field, paired position morph), not a
        # package v2 ever ran, so its wrapper lives here (like v16/v18-v22) and
        # loads that file directly. It speaks the identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav23.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "H3D-FLA (v20) + co-expression-module chimerism — the principled "
                 "middle between SpatialZ's per-gene chimerism and whole-profile "
                 "copies; niche-weighted, stratified, local-composition, morph",
        # No ``wrapper_args``: the wrapper's argparse defaults already mirror
        # V14Config's production defaults, so a bare invocation runs v23 as
        # intended; which knobs actually ran is recorded per prediction in
        # ``method_params``, so a result is self-describing.
        #
        # v23 degrades to the numpy latent-grounded fallback when torch is
        # missing or the flow fails to train/generate — the same fallback prints
        # as v18-v22 (the shared v14 core). The fallback is not the method under
        # test, so v3 fails the run rather than scoring it. The first two markers
        # are the wrapper's own; the last three are printed by
        # learn_spatialcpav23.py itself when it drops to the fallback.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v14] torch unavailable",
                                "[v14] training failed",
                                "[v14] generation failed"),
    },
    "spatialcpav24_gen": {
        # v24 is the single-file ``learn_spatialcpav24.py`` at the repository root,
        # implementing the ``v23_design.md`` CTF-Flow design. Its wrapper lives here
        # (like v16/v18-v23) and loads that file directly; identical _v2_io contract.
        "wrapper": _v3_wrapper("run_spatialcpav24.py"),
        "conda_env": "bench_spatialcpa",
        "available": True,
        "family": "spatialcpa",
        "notes": "CTF-Flow — continuous transcriptomic field (v23_design.md). "
                 "Neural tier (triplane/flow/ZINB/MedCPT/metric-aware autocorr) is "
                 "implemented and CPU-smoke-tested; a successful GPU training run is "
                 "scored, a torchless/failed run degrades to the numpy tier and is "
                 "quarantined.",
        # The neural CTF-Flow tier is implemented and trains (smoke-tested on CPU),
        # so on a GPU with torch a successful fit reports trained=True and the run
        # is SCORED — none of the markers below fire in that case. The markers fire
        # only on DEGRADATION: no torch, or training/generation raised and the class
        # fell back to the numpy field tier (a v20 superset, not CTF-Flow). Failing
        # a degraded run rather than scoring it keeps the numpy fallback from being
        # mistaken for the neural method.
        "invalid_log_markers": ("flow-matching model trained: False",
                                "torch UNAVAILABLE",
                                "[v24] torch unavailable",
                                "[v24] neural training failed",
                                "[v24] neural generation failed",
                                "[v24] neural tier disabled"),
    },
    "spatialcpav25_gen": {
        # v25 is the ``spatialcpav25_gen`` package (SpatialCPA-v25-Gen / CTF-Flow), not a
        # single file and not a package v2 ever ran, so its wrapper lives here like
        # v16/v18-v24's and speaks the identical _v2_io contract.
        #
        # Deliberately NOT added to METHOD_ORDER: a method registered here but absent from
        # that list "is never run unless it is named explicitly", so a bare ``run_all``
        # plans exactly the campaign it planned before this entry existed and no existing
        # result is affected. Run it with ``--methods spatialcpav25_gen``.
        #
        # The only SpatialCPA method NOT in ``bench_spatialcpa``, and the reason is a hard
        # incompatibility rather than a preference: that environment is pinned to
        # ``python=3.10`` (``benchmark-pbya/envs/spatialcpa.yml``, measured at 3.10.20 on the
        # campaign machine) and v25 requires ``>=3.11,<3.13`` — pip refuses the install
        # outright. Installing it there anyway would mean either overriding the interpreter
        # check, which runs the method on a version nothing has been tested on, or a
        # full-dependency install, which would rewrite the exactly-pinned numpy/torch under
        # the thirteen other methods sharing the env. Per-method ``conda_env`` exists for
        # exactly this: v25 gets its own (``benchmark-pbya/envs/spatialcpav25.yml``) and
        # every other entry above is untouched.
        "wrapper": _v3_wrapper("run_spatialcpav25_gen.py"),
        "conda_env": "bench_spatialcpav25",
        "available": True,
        "family": "spatialcpa",
        "notes": "CTF-Flow — continuous transcriptomic field with a 3D GRF prior, "
                 "rotation-equivariant triplane, z-proximity retrieval and a "
                 "gene-conditioned ZINB decoder. Configuration is selected internally "
                 "per dataset: the wrapper takes NO tuning flags.",
        # No ``wrapper_args``: a bare invocation must produce the shipped configuration,
        # because "fit takes no method flags" is a claim in the paper. The flags the
        # wrapper does accept are ablation switches (specs/10 §6) and the two selection
        # controls, never tuning.
        #
        # v25 has no silent fallback to quarantine — Convention 6 makes a missing
        # requirement raise rather than degrade — so these markers cover only the cases
        # where a *dependency* is absent and the run would otherwise look like a method
        # result: no torch, and the MedCPT encoder unavailable under the shipped
        # ``text_emb_mode="medcpt"`` (substituting a lookup table would silently turn every
        # run into ablation A3).
        "invalid_log_markers": ("torch UNAVAILABLE",
                                "spatialcpav25_gen is not importable",
                                "MedCPT encoder unavailable"),
    },
    "spatialz": {
        "wrapper": _v2_wrapper("run_spatialz.py"),
        "conda_env": "bench_spatialz",
        "available": True,
        "family": "published",
        "notes": "Lin et al. 2025, Nat Methods — the method this protocol comes from",
    },
    "feast": {
        "wrapper": _v2_wrapper("run_feast.py"),
        "conda_env": "bench_feast",
        "available": True,
        "family": "published",
        "notes": "Chen et al. 2025 — parameter-cloud interpolation over a PASTE2 alignment",
    },
    "isost": {
        "wrapper": _v2_wrapper("run_isost.py"),
        "conda_env": "bench_isost",
        "available": True,
        "family": "published",
        "notes": "Li et al. 2025 — biaxial SDE generation",
    },
}

# Order used in tables and figures: published baselines first, then SpatialCPA,
# oldest variant first. This list is also ``run_all``'s default campaign, so a
# method registered above but missing here is never run unless it is named
# explicitly — which is why v16 and v17 are both here rather than only in
# METHODS.
#
# It is now longer than ``nature_theme.PALETTE`` (seven validated hues, never
# cycled). Tables, rankings and the per-dataset figures are unaffected — they
# order by this list but do not colour by it. ``plot_cross_dataset`` does colour
# by it, and refuses to draw more than seven series; pass ``--method-order`` to
# choose which appear. See the README, "Cross-dataset figures".
METHOD_ORDER = [
    "spatialz", "feast", "isost",
    "spatialcpav8_gen", "spatialcpav11_gen", "spatialcpav14_gen",
    "spatialcpav15_gen", "spatialcpav16_gen", "spatialcpav17_gen",
    "spatialcpav18_gen", "spatialcpav19_gen", "spatialcpav20_gen",
    "spatialcpav21_gen", "spatialcpav22_gen", "spatialcpav23_gen",
    "spatialcpav24_gen",
]


# ── Metric names (canonical column order) ─────────────────────────────────────
# Tier A — the paper's own validation strategy (v3-specific, ``paper_*``).
PAPER_METRIC_NAMES = [
    # UMAP continuity between real and reconstructed cells
    "paper_umap_mixing",
    "paper_umap_centroid_dist",
    "paper_embedding_mixing_pca",
    # spatial autocorrelation: Moran's I and Geary's C
    "paper_morans_pearson",
    "paper_morans_spearman",
    "paper_morans_mae",
    "paper_morans_median_pred",
    "paper_morans_median_gt",
    "paper_gearys_pearson",
    "paper_gearys_spearman",
    "paper_gearys_mae",
    "paper_gearys_median_pred",
    "paper_gearys_median_gt",
    # marker-gene spatial patterns (Flt1 / Pcp4 / Cux2)
    "paper_marker_field_r",
    "paper_marker_field_ssim",
    "paper_marker_depth_r",
    "paper_marker_morans_mae",
    # gene expression similarity
    "paper_gene_mean_spearman",
    "paper_gene_var_spearman",
    # gene detection frequency (sparsity structure; the rank-invariant metrics
    # cannot see this, so it is reported on its own)
    "paper_gene_detection_spearman",
    "paper_gene_detection_median_pred",
    "paper_gene_detection_median_gt",
    # preservation of cell spatial localization (+ rare-cell-type preservation)
    "paper_celltype_localization",
    "paper_celltype_ot",
    "paper_rare_celltype_localization",
    "paper_rare_celltype_recall",
    # bookkeeping
    "paper_cell_count_ratio",
]

# Per-marker breakdowns (populated per gene in MARKER_GENES).
PAPER_MARKER_METRIC_NAMES = [
    f"paper_marker_{g}_{suffix}"
    for g in MARKER_GENES
    for suffix in ("field_r", "field_ssim", "depth_r", "morans_delta")
]

# Tier B — v2's correspondence-free generation metrics, computed identically so
# v3 numbers can be read alongside the v2 sweep.
GEN_METRIC_NAMES = [
    "gen_coexpression_agreement",
    "gen_morans_agreement",
    "gen_sinkhorn",
    "gen_celltype_composition",
    "gen_celltype_nhood_agreement",
    "gen_gene_mean_pearson",
    "gen_gene_var_pearson",
    "gen_field_pearson",
    "gen_field_ssim",
    "gen_density_pearson",
    "gen_morans_i_pred_median",
]

# Tier C — v2's cell-matched metrics. Kept for reference only: de-novo generation
# produces no cell-to-cell correspondence, so these are NOT a valid score here
# (see benchmark-pbya-v2/README.md).
MATCHED_METRIC_NAMES = [
    "pearson_median", "spearman_median", "celltype_accuracy",
    "celltype_f1_macro", "ssim_median", "density_pearson", "matching_rate",
]

METRIC_NAMES = (PAPER_METRIC_NAMES + PAPER_MARKER_METRIC_NAMES
                + GEN_METRIC_NAMES + MATCHED_METRIC_NAMES)


# ── Evaluation defaults ───────────────────────────────────────────────────────
SPATIAL_K = 10          # kNN degree for Moran's I / Geary's C spatial weights
FIELD_GRID = 20         # bins per axis for binned spatial fields
DEPTH_BINS = 20         # bins along the cortical laminar axis
EMBED_NEIGHBORS = 15    # kNN degree for the UMAP/PCA mixing score
RARE_CELLTYPE_FRAC = 0.05   # a GT type below this frequency is "rare" for the
                            # rare-cell-type preservation metric (evaluate_paper)

# ── Pose selection for the alignment-dependent metrics (align.py) ─────────────
# Candidate rotations tried when bringing a prediction into the GT frame. Poses
# are scored by marker-field agreement rather than by cell overlap, because a
# symmetric outline covers itself equally well under a flip; reflections are not
# candidates at all, since physical tissue cannot be mirrored.
ALIGN_ANGLES = 24        # candidate rotations, evenly spaced over 360 degrees
ALIGN_ICP_ITERS = 12     # refinement iterations per candidate (0 disables)
ALIGN_MAX_POINTS = 3000  # cells subsampled per side for the pose search
