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
        # v1's processed file only: the raw distribution is two zips whose
        # coordinates have to be rebuilt from per-section flat files, which is
        # v1's processor's job. Run it once if this file is missing.
        "raw_candidates": (
            V1_ROOT / "data" / "processed" / "cosmx_nsclc_3d" / "data.h5ad",
            V1_ROOT / "data" / "processed" / "cosmx_nsclc_3d.h5ad",
        ),
        "source_units": "um",
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

# Order used in tables and figures: published baselines first, then SpatialCPA.
METHOD_ORDER = [
    "spatialz", "feast", "isost",
    "spatialcpav8_gen", "spatialcpav11_gen", "spatialcpav14_gen", "spatialcpav15_gen",
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
    "paper_marker_depth_r",
    "paper_marker_morans_mae",
    # gene expression similarity
    "paper_gene_mean_spearman",
    "paper_gene_var_spearman",
    # preservation of cell spatial localization
    "paper_celltype_localization",
    "paper_celltype_ot",
    # bookkeeping
    "paper_cell_count_ratio",
]

# Per-marker breakdowns (populated per gene in MARKER_GENES).
PAPER_MARKER_METRIC_NAMES = [
    f"paper_marker_{g}_{suffix}"
    for g in MARKER_GENES
    for suffix in ("field_r", "depth_r", "morans_delta")
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
