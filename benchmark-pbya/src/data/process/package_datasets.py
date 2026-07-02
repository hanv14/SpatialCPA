#!/usr/bin/env python
"""Package processed datasets into self-contained directories with README files.

Creates a standardized directory structure:
  - Single-specimen: {dataset}/README.md + {dataset}/data.h5ad
  - Multi-specimen: {dataset}/README.md + {dataset}/{specimen}/data.h5ad

Also copies section images where available (visium_mouse_brain_cell2location).

Usage:
    python src/data/process/package_datasets.py              # package all
    python src/data/process/package_datasets.py --dry-run    # preview only
    python src/data/process/package_datasets.py --dataset X  # one dataset
    python src/data/process/package_datasets.py --force      # overwrite READMEs
"""

import argparse
import ast
import os
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DOWNLOAD_DIR = PROJECT_ROOT / "src" / "data" / "download"
PROCESS_DIR = PROJECT_ROOT / "src" / "data" / "process"
CLAUDE_MD = PROJECT_ROOT / "CLAUDE.md"
QC_REPORT = PROCESSED_DIR / "qc_report.csv"

# ── Dataset registry ─────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    # Single-specimen (10 datasets, 10 h5ad files)
    "allen_merfish_brain": {"type": "single"},
    "cosmx_nsclc_3d": {"type": "single"},
    "deep_starmap": {"type": "single"},
    "exseq_breast_cancer": {"type": "single"},
    "exseq_visual_cortex": {"type": "single"},
    "imc_breast_cancer": {"type": "single"},
    "merfish_hypothalamus": {
        "type": "multi",
        "specimens": [
            "animal_1", "animal_2", "animal_3", "animal_4", "animal_5",
            "animal_6", "animal_7", "animal_8", "animal_9", "animal_10",
            "animal_11",
        ],
    },
    "openst_lymph_node": {"type": "single"},
    "st_mouse_brain_ortiz": {"type": "single"},
    "starmap_visual_cortex": {"type": "single"},
    # Multi-specimen (5 datasets)
    "allen_zhuang_merfish": {
        "type": "multi",
        "specimens": [
            "Zhuang-ABCA-1", "Zhuang-ABCA-2",
            "Zhuang-ABCA-3", "Zhuang-ABCA-4",
        ],
    },
    "easi_fish_hypothalamus": {
        "type": "multi",
        "specimens": ["LHA1", "LHA2", "LHA3"],
    },
    # REMOVED: "htan_colorectal_3d" — frame is tile index, not z-plane; each CSV is single 2D section
    "merfish_thick_tissue": {
        "type": "multi",
        "specimens": ["cortex", "hypothalamus"],
    },
    "visium_mouse_brain_cell2location": {
        "type": "multi",
        "specimens": ["mouse_1"],
    },
    # Tier 3: paper-specific benchmark datasets
    "arrayseq_kidney": {"type": "single"},
    "visium_spinal_cord_isost": {"type": "single"},
    "visium_dlpfc_stvgp": {"type": "single"},
    "st_breast_cancer_stvgp": {"type": "single"},
    "imc_breast_cancer_kuett": {
        "type": "multi",
        "specimens": ["MainHer2", "SecondHer2"],
    },
    # REMOVED:
    # "st_human_heart_asp" — section spacing unknown, cannot assign physical z
    # "visium_dlpfc_spatialDLPFC" — ant/mid/post are separate tissue blocks, not serial sections
}

# ── Technology spot/cell size info ────────────────────────────────────────────
# Per-technology spot/cell size and spacing for READMEs
TECHNOLOGY_SPATIAL_INFO = {
    "Visium": {
        "spot_diameter_um": 55.0,
        "center_to_center_um": 100.0,
        "description": "55 µm capture spots, 100 µm center-to-center spacing",
    },
    "ST": {
        "spot_diameter_um": 100.0,
        "center_to_center_um": 200.0,
        "description": "100 µm capture spots, 200 µm center-to-center spacing",
    },
    "OpenST": {
        "spot_diameter_um": 0.6,
        "center_to_center_um": 0.6,
        "description": "0.6 µm capture spots (near-single-cell resolution)",
    },
    "MERFISH": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Single-molecule imaging; cell segmentation (~10-20 µm per cell)",
    },
    "Deep-STARmap": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Single-molecule imaging; cell segmentation (~10-15 µm per cell)",
    },
    "STARmap": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Single-molecule imaging; cell segmentation (~10-15 µm per cell)",
    },
    "CosMx": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Single-molecule imaging; cell segmentation (~10-15 µm per cell)",
    },
    "EASI-FISH": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Single-molecule FISH; cell segmentation (~10-20 µm per cell)",
    },
    "ExSeq": {
        "spot_diameter_um": None,
        "center_to_center_um": None,
        "description": "Expansion sequencing; single-cell resolution",
    },
    "IMC": {
        "spot_diameter_um": 1.0,
        "center_to_center_um": 1.0,
        "description": "1 µm pixel resolution (imaging mass cytometry)",
    },
}

# ── Section alignment info ───────────────────────────────────────────────────
# Per-dataset alignment status for READMEs
# Categories: "pre-aligned", "volumetric", "not-aligned"

DATASET_ALIGNMENT_INFO = {
    "allen_merfish_brain": {
        "status": "pre-aligned",
        "summary": (
            "Sections **pre-aligned** to the Allen Common Coordinate Framework v3 "
            "(CCFv3) via global affine + per-section deformable registration "
            "(Allen Institute pipeline). Coordinates are in CCFv3 µm space."
        ),
    },
    "allen_zhuang_merfish": {
        "status": "pre-aligned",
        "summary": (
            "Sections **pre-aligned** to the Allen Common Coordinate Framework v3 "
            "(CCFv3) via global affine + per-section deformable registration "
            "(Allen Institute pipeline). Note: 4 anterior sections in Zhuang-ABCA-1 "
            "share z=0 due to upstream CCF registration placing them at the same "
            "coronal depth. Coordinates are in CCFv3 µm space."
        ),
    },
    "cosmx_nsclc_3d": {
        "status": "not-aligned",
        "summary": (
            "Sections **require alignment** for accurate 3D reconstruction. "
            "The authors originally aligned the 6 serial sections using STIM "
            "(Spatial Transcriptomics as Images; Preibisch et al. 2025, Cell Systems), "
            "which renders gene expression as images and applies affine + non-rigid "
            "registration. The STIM-aligned coordinates are in the preprocessed h5ad "
            "(`stimwrap_files.zip` also available on Zenodo). However, the current "
            "processing pipeline replaces STIM coordinates with per-section-centered "
            "raw pixel positions, so sections are **not cross-registered** in the "
            "packaged data. To use STIM-aligned coordinates, extract them from the "
            "preprocessed h5ad or stimwrap outputs."
        ),
    },
    "deep_starmap": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Data comes from confocal z-stack imaging of a "
            "single hydrogel-embedded tissue block (Deep-STARmap). All z-planes are "
            "inherently co-registered within the imaging volume."
        ),
    },
    "easi_fish_hypothalamus": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Each specimen (LHA1, LHA2, LHA3) is a volumetric "
            "EASI-FISH z-stack of a single ~300 µm thick tissue section. Coordinates "
            "are inherently aligned within each specimen. The three specimens are "
            "independent biological replicates with separate coordinate frames, "
            "not serial sections from one block."
        ),
    },
    "exseq_breast_cancer": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Data comes from volumetric expansion sequencing "
            "(ExSeq) of a single tissue block. All z-planes are inherently co-registered "
            "within the expansion microscopy imaging volume."
        ),
    },
    "exseq_visual_cortex": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Data comes from volumetric expansion sequencing "
            "(ExSeq) of a single tissue block. All z-planes are inherently co-registered "
            "within the expansion microscopy imaging volume."
        ),
    },
    "imc_breast_cancer": {
        "status": "pre-aligned",
        "summary": (
            "Sections **pre-aligned** by the authors using Fiji/ImageJ2 Register "
            "Virtual Stack Slices plugin (SIFT-based affine registration). "
            "Serial 5 µm sections were registered to reconstruct the 3D tumor "
            "microenvironment (Kuett et al. 2022)."
        ),
    },
    "merfish_hypothalamus": {
        "status": "pre-aligned",
        "summary": (
            "Sections **inherently aligned** via consistent MERFISH microscope stage "
            "positioning across serial sections. The z-coordinate is assigned from "
            "known Bregma positions (mm × 1000 → µm). Each specimen is one naive "
            "animal with 5-12 coronal sections at 50 µm spacing (animals 1, 2, 7 "
            "have full 12-section coverage; others cover anterior or posterior half). "
            "No additional cross-section registration was applied."
        ),
    },
    "merfish_thick_tissue": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Each specimen (cortex, hypothalamus) is a "
            "volumetric 3D MERFISH acquisition of a single thick tissue section "
            "(cortex: 100 µm, hypothalamus: 200 µm). All z-planes within each "
            "specimen are inherently co-registered from the imaging process."
        ),
    },
    "openst_lymph_node": {
        "status": "not-aligned",
        "summary": (
            "Sections **require alignment** for accurate 3D reconstruction. "
            "The authors aligned 19 serial sections using STIM (Spatial "
            "Transcriptomics as Images), but the deposited data contains only "
            "section-local coordinates. The current processing pipeline assigns "
            "z from section order; x/y coordinates are **not cross-registered** "
            "between sections."
        ),
    },
    "st_mouse_brain_ortiz": {
        "status": "pre-aligned",
        "summary": (
            "Sections **pre-aligned** to the Allen Mouse Brain Atlas using the "
            "WholeBrain R package. Each section is registered to a known "
            "anterior-posterior (Bregma) position. A/B replicate pairs share the "
            "same coronal depth by design. Sections come from 3 interleaved mice "
            "(A1: 42 sections, A2: 23 sections, A3: 10 sections) that together "
            "tile the full AP axis; `obs['animal_id']` tracks the source animal."
        ),
    },
    "starmap_visual_cortex": {
        "status": "volumetric",
        "summary": (
            "**No alignment needed.** Data comes from confocal z-stack imaging of a "
            "single intact tissue block (STARmap). All z-planes (~1 µm steps) are "
            "inherently co-registered within the imaging volume."
        ),
    },
    "visium_mouse_brain_cell2location": {
        "status": "not-aligned",
        "summary": (
            "Sections **require alignment** for accurate 3D reconstruction. "
            "Each section has independent Visium array coordinates. The 3 serial "
            "coronal sections (210 µm spacing) are **not cross-registered** — "
            "x/y positions reflect capture spot locations on separate Visium slides."
        ),
    },
}

# ── Citations ─────────────────────────────────────────────────────────────────
# Each entry: paper (full citation), doi, publication_url, data_urls (list)

CITATIONS = {
    "allen_merfish_brain": {
        "paper": (
            'Yao Z, van Velthoven CTJ, Kunst M, et al. '
            '"A high-resolution transcriptomic and spatial atlas of cell types '
            'in the whole mouse brain." '
            'Nature 624, 317-332 (2023).'
        ),
        "doi": "10.1038/s41586-023-06812-z",
        "publication_url": "https://www.nature.com/articles/s41586-023-06812-z",
        "data_urls": [
            "https://alleninstitute.github.io/abc_atlas_access/",
        ],
    },
    "allen_zhuang_merfish": {
        "paper": (
            'Zhang M, Pan X, Jung W, et al. '
            '"Molecularly defined and spatially resolved cell atlas of the '
            'whole mouse brain." '
            'Nature 624, 343-354 (2023).'
        ),
        "doi": "10.1038/s41586-023-06808-9",
        "publication_url": "https://www.nature.com/articles/s41586-023-06808-9",
        "data_urls": [
            "https://alleninstitute.github.io/abc_atlas_access/",
        ],
    },
    "cosmx_nsclc_3d": {
        "paper": (
            'Pentimalli TM, Schallenberg S, León-Periñán D, et al. '
            '"Combining spatial transcriptomics and ECM imaging in 3D for '
            'mapping cellular interactions in the tumor microenvironment." '
            'Cell Systems 16(5):101261 (2025).'
        ),
        "doi": "10.1016/j.cels.2025.101261",
        "publication_url": (
            "https://www.cell.com/cell-systems/fulltext/"
            "S2405-4712(25)00094-8"
        ),
        "data_urls": [
            "https://zenodo.org/records/15240431",
            "https://github.com/rajewsky-lab/3D_lung",
        ],
    },
    "deep_starmap": {
        "paper": (
            'Sui X, Lo JA, Luo S, et al. '
            '"Scalable spatial single-cell transcriptomics and translatomics '
            'in 3D thick tissue blocks." '
            'Nature Methods 22, 2574-2584 (2025).'
        ),
        "doi": "10.1038/s41592-025-02867-0",
        "publication_url": (
            "https://www.nature.com/articles/s41592-025-02867-0"
        ),
        "data_urls": [
            "https://zenodo.org/records/16783355",
        ],
    },
    "easi_fish_hypothalamus": {
        "paper": (
            'Wang Y, Eddison M, Fleishman G, et al. '
            '"EASI-FISH for thick tissue defines lateral hypothalamus '
            'spatio-molecular organization." '
            'Cell 184(26), 6361-6377.e24 (2021).'
        ),
        "doi": "10.1016/j.cell.2021.11.024",
        "publication_url": (
            "https://www.cell.com/cell/fulltext/S0092-8674(21)01339-8"
        ),
        "data_urls": [
            "https://doi.org/10.25378/janelia.c.5276708.v1",
        ],
    },
    "exseq_breast_cancer": {
        "paper": (
            'Alon S, Goodwin DR, Sinha A, et al. '
            '"Expansion sequencing: Spatially precise in situ transcriptomics '
            'in intact biological systems." '
            'Science 371(6528):eaax2656 (2021).'
        ),
        "doi": "10.1126/science.aax2656",
        "publication_url": (
            "https://www.science.org/doi/10.1126/science.aax2656"
        ),
        "data_urls": [
            "https://zenodo.org/records/4479018",
        ],
    },
    "exseq_visual_cortex": {
        "paper": (
            'Alon S, Goodwin DR, Sinha A, et al. '
            '"Expansion sequencing: Spatially precise in situ transcriptomics '
            'in intact biological systems." '
            'Science 371(6528):eaax2656 (2021).'
        ),
        "doi": "10.1126/science.aax2656",
        "publication_url": (
            "https://www.science.org/doi/10.1126/science.aax2656"
        ),
        "data_urls": [
            "https://spacetx.github.io/",
        ],
    },
    # htan_colorectal_3d: REMOVED (frame is tile index, not z-plane)
    "imc_breast_cancer": {
        "paper": (
            'Kuett L, Catena R, Özcan A, et al. '
            '"Three-dimensional imaging mass cytometry for highly multiplexed '
            'molecular and cellular mapping of tissues and the tumor '
            'microenvironment." '
            'Nature Cancer 3, 122-133 (2022).'
        ),
        "doi": "10.1038/s43018-021-00301-w",
        "publication_url": (
            "https://www.nature.com/articles/s43018-021-00301-w"
        ),
        "data_urls": [
            "https://doi.org/10.5281/zenodo.4752030",
        ],
    },
    "merfish_hypothalamus": {
        "paper": (
            'Moffitt JR, Bambah-Mukku D, Eichhorn SW, et al. '
            '"Molecular, spatial, and functional single-cell profiling of the '
            'hypothalamic preoptic region." '
            'Science 362(6416):eaau5324 (2018).'
        ),
        "doi": "10.1126/science.aau5324",
        "publication_url": (
            "https://www.science.org/doi/10.1126/science.aau5324"
        ),
        "data_urls": [
            "https://datadryad.org/dataset/doi:10.5061/dryad.8t8s248",
        ],
    },
    "merfish_thick_tissue": {
        "paper": (
            'Fang R, Halpern A, Rahman MM, et al. '
            '"Three-dimensional single-cell transcriptome imaging of thick '
            'tissues." '
            'eLife 12:RP90029 (2024).'
        ),
        "doi": "10.7554/eLife.90029",
        "publication_url": "https://elifesciences.org/articles/90029",
        "data_urls": [
            "https://datadryad.org/dataset/doi:10.5061/dryad.w0vt4b922",
        ],
    },
    "openst_lymph_node": {
        "paper": (
            'Schott M, León-Periñán D, Splendiani E, et al. '
            '"Open-ST: High-resolution spatial transcriptomics in 3D." '
            'Cell 187(15), 3953-3972.e26 (2024).'
        ),
        "doi": "10.1016/j.cell.2024.05.055",
        "publication_url": (
            "https://www.cell.com/cell/fulltext/S0092-8674(24)00636-6"
        ),
        "data_urls": [
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251926",
        ],
    },
    # st_human_heart_asp: REMOVED (unknown section spacing)
    "st_mouse_brain_ortiz": {
        "paper": (
            'Ortiz C, Navarro JF, Jurek A, et al. '
            '"Molecular atlas of the adult mouse brain." '
            'Science Advances 6(26):eabb3446 (2020).'
        ),
        "doi": "10.1126/sciadv.abb3446",
        "publication_url": (
            "https://www.science.org/doi/10.1126/sciadv.abb3446"
        ),
        "data_urls": [
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147747",
        ],
    },
    "starmap_visual_cortex": {
        "paper": (
            'Wang X, Allen WE, Wright MA, et al. '
            '"Three-dimensional intact-tissue sequencing of single-cell '
            'transcriptional states." '
            'Science 361(6400):eaat5691 (2018).'
        ),
        "doi": "10.1126/science.aat5691",
        "publication_url": (
            "https://www.science.org/doi/10.1126/science.aat5691"
        ),
        "data_urls": [
            "https://ndownloader.figshare.com/files/58960009",
        ],
    },
    # visium_dlpfc_spatialDLPFC: REMOVED (not serial sections)
    "visium_mouse_brain_cell2location": {
        "paper": (
            'Kleshchevnikov V, Shmatko A, Dann E, et al. '
            '"Cell2location maps fine-grained cell types in spatial '
            'transcriptomics." '
            'Nature Biotechnology 40, 661-671 (2022).'
        ),
        "doi": "10.1038/s41587-021-01139-4",
        "publication_url": (
            "https://www.nature.com/articles/s41587-021-01139-4"
        ),
        "data_urls": [
            "https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11114",
        ],
    },
}


# ── Metadata extraction ──────────────────────────────────────────────────────

def get_script_docstring(script_path: Path) -> str | None:
    """Extract module docstring from a Python script using AST."""
    if not script_path.exists():
        return None
    try:
        source = script_path.read_text()
        tree = ast.parse(source)
        doc = ast.get_docstring(tree)
        if doc:
            # Fix stale output paths: old flat .h5ad → new dir/data.h5ad
            # Handles both literal names and {variable} patterns
            doc = re.sub(
                r'data/processed/(\w+)\.h5ad',
                r'data/processed/\1/data.h5ad',
                doc,
            )
            doc = re.sub(
                r'data/processed/(\w+)/(\{[^}]+\})\.h5ad',
                r'data/processed/\1/\2/data.h5ad',
                doc,
            )
        return doc
    except Exception:
        return None


def get_claude_md_caveats(dataset_name: str) -> str | None:
    """Extract data format caveats for a dataset from CLAUDE.md."""
    if not CLAUDE_MD.exists():
        return None
    text = CLAUDE_MD.read_text()
    match = re.search(
        r"### Data format caveats.*?\n(.*?)(?=\n###|\Z)", text, re.DOTALL
    )
    if not match:
        return None
    caveats_section = match.group(1)
    lines = []
    for line in caveats_section.strip().split("\n"):
        if dataset_name in line:
            cleaned = re.sub(r"^-\s+\*\*[^*]+\*\*:\s*", "", line.strip())
            lines.append(cleaned)
    return "\n".join(lines) if lines else None


def load_qc_report() -> pd.DataFrame | None:
    """Load the QC report CSV."""
    if not QC_REPORT.exists():
        return None
    return pd.read_csv(QC_REPORT)


def get_qc_rows(qc_df: pd.DataFrame | None, dataset_name: str,
                 info: dict) -> pd.DataFrame:
    """Get QC report rows for a dataset."""
    if qc_df is None:
        return pd.DataFrame()
    if info["type"] == "single":
        mask = qc_df["dataset"] == dataset_name
    else:
        mask = qc_df["dataset"].str.startswith(dataset_name + "/")
    return qc_df[mask]


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_int(n) -> str:
    """Format a number with comma separators, or N/A."""
    if n is None or (isinstance(n, float) and pd.isna(n)):
        return "N/A"
    return f"{int(n):,}"


def na_str(val) -> str:
    """Convert value to string, returning N/A for None/NaN."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return str(val)


def fmt_pct(val) -> str:
    """Format a percentage value."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    return f"{val:.1f}%"


def fmt_spatial_extent(row) -> str:
    """Format spatial extent from a QC row."""
    x_min = row.get("x_min")
    if x_min is None or (isinstance(x_min, float) and pd.isna(x_min)):
        return "N/A"
    return (
        f"x=[{row['x_min']:.2f}, {row['x_max']:.2f}], "
        f"y=[{row['y_min']:.2f}, {row['y_max']:.2f}], "
        f"z=[{row['z_min']:.2f}, {row['z_max']:.2f}]"
    )


def section_table_simple(section_details: str) -> list[str]:
    """Parse section_details string into simple markdown table (fallback)."""
    lines = []
    lines.append("| Section | Cells |")
    lines.append("|---------|-------|")
    for part in section_details.split("; "):
        m = re.match(r"(.+?)\s*\((\d+)\)", part.strip())
        if m:
            lines.append(f"| {m.group(1)} | {int(m.group(2)):,} |")
    return lines


def compute_section_stats(h5ad_path: Path) -> list[dict] | None:
    """Compute per-section spatial and expression stats from an h5ad file."""
    import anndata as ad

    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
    except Exception:
        return None

    if "section" not in adata.obs.columns or "spatial" not in adata.obsm:
        adata.file.close()
        return None

    sections = adata.obs["section"]
    spatial = np.array(adata.obsm["spatial"])
    if spatial.ndim != 2 or spatial.shape[1] < 3:
        adata.file.close()
        return None

    # Load expression matrix
    try:
        X = adata.X[:]
        if sparse.issparse(X):
            X = X.tocsr()
        else:
            X = sparse.csr_matrix(X)
    except Exception:
        X = None

    results = []
    for sec_name in sections.cat.categories if hasattr(sections, "cat") else sorted(sections.unique()):
        mask = (sections == sec_name).values
        n_cells = int(mask.sum())
        sp = spatial[mask]

        xs, ys, zs = sp[:, 0], sp[:, 1], sp[:, 2]
        rec = {
            "section": str(sec_name),
            "cells": n_cells,
            "x_min": float(np.nanmin(xs)),
            "x_max": float(np.nanmax(xs)),
            "y_min": float(np.nanmin(ys)),
            "y_max": float(np.nanmax(ys)),
            "z_min": float(np.nanmin(zs)),
            "z_max": float(np.nanmax(zs)),
            "n_unique_x": len(np.unique(xs[~np.isnan(xs)])),
            "n_unique_y": len(np.unique(ys[~np.isnan(ys)])),
            "n_unique_z": len(np.unique(zs[~np.isnan(zs)])),
        }

        if X is not None:
            sec_X = X[mask]
            if sparse.issparse(sec_X):
                if sec_X.nnz > 0:
                    rec["expr_min"] = float(sec_X.data.min())
                    rec["expr_max"] = float(sec_X.data.max())
                else:
                    rec["expr_min"] = 0.0
                    rec["expr_max"] = 0.0
                # Include zeros for true min
                if sec_X.nnz < sec_X.shape[0] * sec_X.shape[1]:
                    rec["expr_min"] = min(rec["expr_min"], 0.0)
            else:
                rec["expr_min"] = float(np.nanmin(sec_X))
                rec["expr_max"] = float(np.nanmax(sec_X))
        else:
            rec["expr_min"] = None
            rec["expr_max"] = None

        results.append(rec)

    adata.file.close()
    return results


def _fmt_range(lo: float, hi: float) -> str:
    """Format a min/max range compactly."""
    if lo == hi:
        return f"{lo:.2f}"
    return f"[{lo:.2f}, {hi:.2f}]"


def _fmt_expr(lo, hi) -> str:
    """Format expression range."""
    if lo is None or hi is None:
        return "N/A"
    if isinstance(lo, float) and np.isnan(lo):
        return "N/A"
    if isinstance(hi, float) and np.isnan(hi):
        return "N/A"
    # Use int format if values are integer-like
    if float(lo) == int(lo) and float(hi) == int(hi):
        return f"[{int(lo)}, {int(hi)}]"
    return f"[{lo:.2f}, {hi:.2f}]"


def section_table_rich(stats: list[dict]) -> list[str]:
    """Build a rich section breakdown table from per-section stats."""
    lines = []
    lines.append(
        "| Section | Cells | x range | y range | z range "
        "| #x | #y | #z | expr range |"
    )
    lines.append(
        "|---------|------:|--------:|--------:|--------:"
        "|---:|---:|---:|----------:|"
    )
    for s in stats:
        lines.append(
            f"| {s['section']} "
            f"| {s['cells']:,} "
            f"| {_fmt_range(s['x_min'], s['x_max'])} "
            f"| {_fmt_range(s['y_min'], s['y_max'])} "
            f"| {_fmt_range(s['z_min'], s['z_max'])} "
            f"| {s['n_unique_x']:,} "
            f"| {s['n_unique_y']:,} "
            f"| {s['n_unique_z']:,} "
            f"| {_fmt_expr(s['expr_min'], s['expr_max'])} |"
        )
    return lines


def _md_link(text: str, url: str) -> str:
    """Create a markdown link, escaping parentheses in URLs."""
    safe_url = url.replace("(", "%28").replace(")", "%29")
    return f"[{text}]({safe_url})"


def format_citation(dataset_name: str) -> list[str]:
    """Format the citation section for a dataset README."""
    cite = CITATIONS.get(dataset_name)
    if not cite:
        return []
    lines = ["## Citation", ""]
    lines.append(cite["paper"])
    lines.append("")
    doi_url = f"https://doi.org/{cite['doi']}"
    lines.append(f"**DOI:** {_md_link(cite['doi'], doi_url)}")
    lines.append("")
    lines.append(
        f"**Publication:** {_md_link(cite['publication_url'], cite['publication_url'])}"
    )
    lines.append("")
    if cite.get("data_urls"):
        lines.append("**Data:**")
        for url in cite["data_urls"]:
            lines.append(f"- {_md_link(url, url)}")
        lines.append("")
    return lines


# ── README generation ─────────────────────────────────────────────────────────

def generate_readme_single(dataset_name: str, qc_df: pd.DataFrame | None,
                           h5ad_path: Path | None = None) -> str:
    """Generate README.md content for a single-specimen dataset."""
    rows = get_qc_rows(qc_df, dataset_name, {"type": "single"})
    row = rows.iloc[0] if len(rows) > 0 else None

    download_doc = get_script_docstring(DOWNLOAD_DIR / f"download_{dataset_name}.py")
    process_doc = get_script_docstring(PROCESS_DIR / f"process_{dataset_name}.py")
    caveats = get_claude_md_caveats(dataset_name)

    lines = [f"# {dataset_name}", ""]

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    if row is not None:
        tech = na_str(row.get('technology'))
        lines.append(f"| Technology | {tech} |")
        lines.append(f"| Species | {na_str(row.get('species'))} |")
        lines.append(f"| Tissue | {na_str(row.get('tissue'))} |")
        lines.append(f"| Expression type | {na_str(row.get('expression_type'))} |")
        tech_info = TECHNOLOGY_SPATIAL_INFO.get(tech)
        if tech_info:
            lines.append(f"| Spatial resolution | {tech_info['description']} |")
    lines.append("| Coordinate units | micrometers (µm) |")
    lines.append("")

    # Data summary
    lines.append("## Data Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    if row is not None:
        lines.append(f"| Cells/spots | {fmt_int(row.get('n_obs'))} |")
        lines.append(f"| Genes/markers | {fmt_int(row.get('n_vars'))} |")
        lines.append(f"| Sections | {fmt_int(row.get('n_sections'))} |")
        lines.append(f"| Sparsity | {fmt_pct(row.get('sparsity_pct'))} |")
        lines.append(f"| Spatial extent | {fmt_spatial_extent(row)} |")
        lines.append(
            f"| Annotation coverage | {fmt_pct(row.get('annotation_coverage_pct'))} |"
        )
    lines.append("")

    # Section breakdown — compute per-section spatial and expression stats
    section_stats = None
    if h5ad_path and h5ad_path.exists():
        print(f"    Computing per-section stats from {h5ad_path.name}...",
              end="", flush=True)
        section_stats = compute_section_stats(h5ad_path)
        print(" done" if section_stats else " failed")

    if section_stats:
        lines.append("## Section Breakdown")
        lines.append("")
        lines.extend(section_table_rich(section_stats))
        lines.append("")
    elif row is not None:
        sd = row.get("section_details")
        if pd.notna(sd) and str(sd).strip():
            lines.append("## Section Breakdown")
            lines.append("")
            lines.extend(section_table_simple(str(sd)))
            lines.append("")

    # obs columns
    if row is not None:
        obs = row.get("obs_columns")
        if pd.notna(obs) and str(obs).strip():
            lines.append("## obs Columns")
            lines.append("")
            cols = [c.strip() for c in str(obs).split(";")]
            lines.append(", ".join(f"`{c}`" for c in cols))
            lines.append("")

    # Citation
    lines.extend(format_citation(dataset_name))

    # Section Alignment
    alignment = DATASET_ALIGNMENT_INFO.get(dataset_name)
    if alignment:
        lines.append("## Section Alignment")
        lines.append("")
        lines.append(alignment["summary"])
        lines.append("")

    # Provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Download: `src/data/download/download_{dataset_name}.py`")
    lines.append(f"- Processing: `src/data/process/process_{dataset_name}.py`")
    lines.append("")

    if process_doc:
        lines.append("## Processing Notes")
        lines.append("")
        lines.append(process_doc)
        lines.append("")

    if caveats:
        lines.append("## Known Issues")
        lines.append("")
        lines.append(caveats)
        lines.append("")

    if row is not None:
        issues = row.get("validation_issues", "")
        if pd.notna(issues) and str(issues).strip():
            lines.append("## Validation Issues")
            lines.append("")
            lines.append(str(issues))
            lines.append("")

    return "\n".join(lines)


def generate_readme_multi(dataset_name: str, info: dict,
                          qc_df: pd.DataFrame | None,
                          base_dir: Path | None = None) -> str:
    """Generate README.md content for a multi-specimen dataset."""
    specimens = info["specimens"]
    rows = get_qc_rows(qc_df, dataset_name, info)

    download_doc = get_script_docstring(DOWNLOAD_DIR / f"download_{dataset_name}.py")
    process_doc = get_script_docstring(PROCESS_DIR / f"process_{dataset_name}.py")
    caveats = get_claude_md_caveats(dataset_name)

    lines = [f"# {dataset_name}", ""]

    # Overview — use first specimen for shared metadata
    first_row = rows.iloc[0] if len(rows) > 0 else None

    lines.append("## Overview")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    if first_row is not None:
        # For metadata fields, collect unique values across all specimens
        def _unique_field(field):
            if len(rows) == 0:
                return "N/A"
            vals = rows[field].dropna().unique().tolist() if field in rows.columns else []
            vals = [str(v) for v in vals if str(v).strip()]
            return "; ".join(sorted(set(vals))) if vals else "N/A"

        tech = _unique_field('technology')
        lines.append(f"| Technology | {tech} |")
        lines.append(f"| Species | {_unique_field('species')} |")
        lines.append(f"| Tissue | {_unique_field('tissue')} |")
        lines.append(
            f"| Expression type | {_unique_field('expression_type')} |"
        )
        tech_info = TECHNOLOGY_SPATIAL_INFO.get(tech)
        if tech_info:
            lines.append(f"| Spatial resolution | {tech_info['description']} |")
    lines.append(f"| Coordinate units | micrometers (µm) |")
    lines.append(f"| Specimens | {len(specimens)} |")
    lines.append("")

    # Per-specimen summary
    lines.append("## Specimens")
    lines.append("")
    lines.append(
        "| Specimen | Cells/spots | Genes/markers | Sections | Sparsity "
        "| Annotation |"
    )
    lines.append(
        "|----------|-------------|---------------|----------|---------"
        "-|------------|"
    )
    total_cells = 0
    for spec in specimens:
        qc_key = f"{dataset_name}/{spec}"
        spec_rows = (
            rows[rows["dataset"] == qc_key] if len(rows) > 0
            else pd.DataFrame()
        )
        if len(spec_rows) > 0:
            r = spec_rows.iloc[0]
            n_obs = r.get("n_obs", 0)
            total_cells += int(n_obs) if pd.notna(n_obs) else 0
            lines.append(
                f"| {spec} "
                f"| {fmt_int(n_obs)} "
                f"| {fmt_int(r.get('n_vars'))} "
                f"| {fmt_int(r.get('n_sections'))} "
                f"| {fmt_pct(r.get('sparsity_pct'))} "
                f"| {fmt_pct(r.get('annotation_coverage_pct'))} |"
            )
        else:
            lines.append(f"| {spec} | N/A | N/A | N/A | N/A | N/A |")
    lines.append("")
    lines.append(f"**Total cells/spots: {total_cells:,}**")
    lines.append("")

    # Per-specimen section breakdowns (rich stats from h5ad)
    for spec in specimens:
        h5ad_path = None
        if base_dir is not None:
            h5ad_path = base_dir / dataset_name / spec / "data.h5ad"
        if h5ad_path and h5ad_path.exists():
            print(f"    Computing per-section stats for {spec}...",
                  end="", flush=True)
            section_stats = compute_section_stats(h5ad_path)
            print(" done" if section_stats else " failed")
            if section_stats:
                lines.append(f"### {spec} Section Breakdown")
                lines.append("")
                lines.extend(section_table_rich(section_stats))
                lines.append("")
        else:
            # Fallback to QC report section_details
            qc_key = f"{dataset_name}/{spec}"
            spec_rows = (
                rows[rows["dataset"] == qc_key] if len(rows) > 0
                else pd.DataFrame()
            )
            if len(spec_rows) > 0:
                sd = spec_rows.iloc[0].get("section_details")
                if pd.notna(sd) and str(sd).strip():
                    lines.append(f"### {spec} Section Breakdown")
                    lines.append("")
                    lines.extend(section_table_simple(str(sd)))
                    lines.append("")

    # obs columns — use first specimen's columns (typically same across all)
    if first_row is not None:
        obs = first_row.get("obs_columns")
        if pd.notna(obs) and str(obs).strip():
            lines.append("## obs Columns")
            lines.append("")
            cols = [c.strip() for c in str(obs).split(";")]
            lines.append(", ".join(f"`{c}`" for c in cols))
            lines.append("")

    # Citation
    lines.extend(format_citation(dataset_name))

    # Section Alignment
    alignment = DATASET_ALIGNMENT_INFO.get(dataset_name)
    if alignment:
        lines.append("## Section Alignment")
        lines.append("")
        lines.append(alignment["summary"])
        lines.append("")

    # Provenance
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- Download: `src/data/download/download_{dataset_name}.py`")
    lines.append(f"- Processing: `src/data/process/process_{dataset_name}.py`")
    lines.append("")

    if process_doc:
        lines.append("## Processing Notes")
        lines.append("")
        lines.append(process_doc)
        lines.append("")

    if caveats:
        lines.append("## Known Issues")
        lines.append("")
        lines.append(caveats)
        lines.append("")

    # Validation issues across specimens
    if len(rows) > 0:
        issue_lines = []
        for _, r in rows.iterrows():
            issues = r.get("validation_issues", "")
            if pd.notna(issues) and str(issues).strip():
                issue_lines.append(f"- **{r['dataset']}**: {issues}")
        if issue_lines:
            lines.append("## Validation Issues")
            lines.append("")
            lines.extend(issue_lines)
            lines.append("")

    return "\n".join(lines)


# ── File restructuring ────────────────────────────────────────────────────────

def restructure_single(dataset_name: str, dry_run: bool) -> list[str]:
    """Move {name}.h5ad -> {name}/data.h5ad."""
    actions = []
    src = PROCESSED_DIR / f"{dataset_name}.h5ad"
    dst_dir = PROCESSED_DIR / dataset_name
    dst = dst_dir / "data.h5ad"

    if dst.exists():
        actions.append(f"  SKIP {dataset_name}/data.h5ad (already exists)")
        return actions

    if not src.exists():
        actions.append(f"  ERROR {dataset_name}.h5ad not found")
        return actions

    actions.append(f"  MOVE {dataset_name}.h5ad -> {dataset_name}/data.h5ad")
    if not dry_run:
        dst_dir.mkdir(parents=True, exist_ok=True)
        os.rename(src, dst)
    return actions


def restructure_multi(dataset_name: str, info: dict,
                      dry_run: bool) -> list[str]:
    """Move {name}/{specimen}.h5ad -> {name}/{specimen}/data.h5ad."""
    actions = []
    base = PROCESSED_DIR / dataset_name

    for specimen in info["specimens"]:
        src = base / f"{specimen}.h5ad"
        dst_dir = base / specimen
        dst = dst_dir / "data.h5ad"

        if dst.exists():
            actions.append(
                f"  SKIP {dataset_name}/{specimen}/data.h5ad (already exists)"
            )
            continue

        if not src.exists():
            actions.append(f"  ERROR {dataset_name}/{specimen}.h5ad not found")
            continue

        actions.append(
            f"  MOVE {dataset_name}/{specimen}.h5ad -> "
            f"{dataset_name}/{specimen}/data.h5ad"
        )
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
    return actions


# ── Image handling ────────────────────────────────────────────────────────────

def extract_cosmx_images(dry_run: bool) -> list[str]:
    """Extract CellComposite images for cosmx_nsclc_3d from flat_files ZIP."""
    actions = []
    zip_path = RAW_DIR / "cosmx_nsclc_3d" / "cosmx_flat_files.zip"
    if not zip_path.exists():
        actions.append("  SKIP cosmx images (cosmx_flat_files.zip not found)")
        return actions

    out_base = PROCESSED_DIR / "cosmx_nsclc_3d" / "images"
    if out_base.exists() and any(out_base.iterdir()):
        actions.append("  SKIP cosmx images/ (already exists)")
        return actions

    with zipfile.ZipFile(zip_path, "r") as zf:
        # Extract CellComposite images per section
        composites = sorted(
            n for n in zf.namelist()
            if "/CellComposite/" in n and n.endswith(".jpg")
        )
        sections = sorted(set(
            n.split("/")[1] for n in composites if len(n.split("/")) >= 3
        ))
        actions.append(
            f"  EXTRACT {len(composites)} CellComposite images "
            f"({len(sections)} sections) -> cosmx_nsclc_3d/images/"
        )
        if not dry_run:
            for member in composites:
                parts = member.split("/")
                if len(parts) >= 4:
                    section = parts[1]
                    fname = parts[-1]
                    out_dir = out_base / section
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_file = out_dir / fname
                    with zf.open(member) as src, open(out_file, "wb") as dst:
                        shutil.copyfileobj(src, dst)
    return actions


def extract_visium_images(dry_run: bool) -> list[str]:
    """Extract section images for visium_mouse_brain_cell2location from ZIP."""
    actions = []
    zip_path = (
        RAW_DIR / "visium_mouse_brain_cell2location"
        / "mouse_brain_visium_wo_cloupe_data.zip"
    )
    if not zip_path.exists():
        actions.append("  SKIP visium images (ZIP not found)")
        return actions

    sections = ["ST8059048", "ST8059049", "ST8059050", "ST8059051", "ST8059052"]
    out_base = PROCESSED_DIR / "visium_mouse_brain_cell2location" / "images"

    with zipfile.ZipFile(zip_path, "r") as zf:
        for section in sections:
            member = (
                f"mouse_brain_visium_wo_cloupe_data/rawdata/{section}"
                f"/spatial/tissue_hires_image.png"
            )
            if member not in zf.namelist():
                continue
            out_dir = out_base / section
            out_file = out_dir / "tissue_hires_image.png"
            if out_file.exists():
                actions.append(
                    f"  SKIP images/{section}/ (already exists)"
                )
                continue
            actions.append(
                f"  EXTRACT {section}/tissue_hires_image.png -> "
                f"images/{section}/"
            )
            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(out_file, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return actions


# ── Main orchestration ────────────────────────────────────────────────────────

def package_dataset(dataset_name: str, info: dict,
                    qc_df: pd.DataFrame | None,
                    dry_run: bool, force: bool) -> list[str]:
    """Package a single dataset: restructure, generate README, copy images."""
    actions = [f"\n{dataset_name} ({info['type']}):"]

    # Restructure h5ad files
    if info["type"] == "single":
        actions.extend(restructure_single(dataset_name, dry_run))
    else:
        actions.extend(restructure_multi(dataset_name, info, dry_run))

    # Generate README
    readme_path = PROCESSED_DIR / dataset_name / "README.md"
    if readme_path.exists() and not force:
        actions.append("  SKIP README.md (exists, use --force to overwrite)")
    else:
        verb = "OVERWRITE" if readme_path.exists() else "CREATE"
        actions.append(f"  {verb} {dataset_name}/README.md")
        if not dry_run:
            if info["type"] == "single":
                h5ad_path = PROCESSED_DIR / dataset_name / "data.h5ad"
                content = generate_readme_single(
                    dataset_name, qc_df, h5ad_path=h5ad_path
                )
            else:
                content = generate_readme_multi(
                    dataset_name, info, qc_df, base_dir=PROCESSED_DIR
                )
            readme_path.parent.mkdir(parents=True, exist_ok=True)
            readme_path.write_text(content)

    # Copy/extract images (only for datasets with available images)
    if dataset_name == "visium_mouse_brain_cell2location":
        actions.extend(extract_visium_images(dry_run))
    elif dataset_name == "cosmx_nsclc_3d":
        actions.extend(extract_cosmx_images(dry_run))

    return actions


def main():
    parser = argparse.ArgumentParser(
        description="Package processed datasets into self-contained directories"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--dataset", type=str,
        help="Package only this dataset",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing READMEs",
    )
    args = parser.parse_args()

    qc_df = load_qc_report()
    if qc_df is None:
        print("WARNING: QC report not found, READMEs will have limited data")

    if args.dataset:
        if args.dataset not in DATASET_REGISTRY:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            print(f"Available: {', '.join(sorted(DATASET_REGISTRY))}")
            raise SystemExit(1)
        datasets = {args.dataset: DATASET_REGISTRY[args.dataset]}
    else:
        datasets = DATASET_REGISTRY

    if args.dry_run:
        print("=== DRY RUN (no changes will be made) ===\n")

    all_actions = []
    for name, info in datasets.items():
        actions = package_dataset(name, info, qc_df, args.dry_run, args.force)
        all_actions.extend(actions)
        for a in actions:
            print(a)

    # Summary
    n_moves = sum(1 for a in all_actions if "MOVE" in a)
    n_creates = sum(1 for a in all_actions if "CREATE" in a or "OVERWRITE" in a)
    n_copies = sum(1 for a in all_actions if "COPY" in a or "EXTRACT" in a)
    n_errors = sum(1 for a in all_actions if "ERROR" in a)
    print(f"\n{'=' * 60}")
    print(
        f"Summary: {n_moves} moves, {n_creates} READMEs, "
        f"{n_copies} image ops, {n_errors} errors"
    )
    if args.dry_run:
        print("(dry run — no changes made)")

    return 1 if n_errors > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
