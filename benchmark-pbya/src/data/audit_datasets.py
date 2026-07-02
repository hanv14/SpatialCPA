#!/usr/bin/env python3
"""
Audit all datasets in the 3D tissue project.

Produces compatibility-matrix/dataset_audit.csv with 52 rows and 21 columns,
cataloging key properties of every dataset: expression data, gene/cell counts,
slice counts, 3D location, consecutiveness, annotations, etc.

Usage:
    python src/data/audit_datasets.py                    # full audit
    python src/data/audit_datasets.py --skip-h5ad        # skip h5ad inspection
    python src/data/audit_datasets.py -o path.csv        # custom output
    python src/data/audit_datasets.py --verbose           # print progress
"""

import argparse
import ast
import csv
import os
import re
import sys
from pathlib import Path

# Project root (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Paths
INVENTORY_CSV = PROJECT_ROOT / "compatibility-matrix" / "dataset_inventory.csv"
SIMILAR_DATASETS_DIR = PROJECT_ROOT / "similar-datasets"
DOWNLOAD_SCRIPTS_DIR = PROJECT_ROOT / "src" / "data" / "download"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT = PROJECT_ROOT / "compatibility-matrix" / "dataset_audit.csv"

# Hardcoded mapping: inventory filename (without .md) -> DATASET_NAME in download scripts
INVENTORY_TO_DATASET = {
    "3D_IMC_breast_cancer_Kuett2022": "imc_breast_cancer",
    "Stereo-seq_axolotl_brain_regeneration": "stereo_axolotl_brain",
    "Stereo-seq_drosophila_embryo_3D": "stereo_drosophila_embryo",
    "MERFISH_Vizgen_MouseBrain": "vizgen_merfish_brain",
    "OpenST_3D_spatial_transcriptomics": "openst_lymph_node",
    "Stereo-seq_mouse_embryo_3D_MOSTA": "stereo_mouse_embryo_mosta",
    "MERFISH_mouse_colon_IBD_Cadinu2024": "merfish_colon_ibd",
    "SlideseqV2_mouse_embryo_3D": "slideseqv2_mouse_embryo",
    "MERFISHplus_human_developing_heart_2025": "merfish_plus_heart",
    "Stereo-seq_zebrafish_embryo_3D": "stereo_zebrafish_embryo",
    "SlideseqV2_mouse_hippocampus": "slideseqv2_hippocampus",
    "Stereoseq_mouse_brain_3D": "stereo_mouse_brain",
    "AllenInstitute_MERFISH_whole_mouse_brain": "allen_merfish_brain",
    "Slideseq_mouse_cerebellum": "slideseq_cerebellum",
    "MIBI_TOF_reproducibility_archival_tissue": "mibi_tof_archival",
    "Langlieb2023_mouse_brain_cytoarchitecture": "langlieb_mouse_brain",
    "AllenInstitute_mouse_brain_Zhuang_MERFISH": "allen_zhuang_merfish",
    "Spatial_mouse_embryo_seqFISH": "seqfish_mouse_embryo",
    "MERFISH_Allen_WMB_CCFv3": "biccn_merfish_brain",
    "EEL-FISH_mouse_brain_atlas": "eel_fish_mouse_brain",
    "BICCN_mouse_motor_cortex_spatial": "biccn_motor_cortex",
    "MERFISH_Moffitt2018_mouse_hypothalamus": "merfish_hypothalamus",
    "MERFISH_MouseHypothalamus_SerialSections_Eichhorn": None,  # removed — duplicate of allen_merfish_brain
    "CosMx_3D_NSCLC_tumor_serial_sections": "cosmx_nsclc_3d",
    "CyCIF_human_tonsil_serial": "cycif_tonsil",
    "HTAN_colorectal_cancer_3D_CyCIF": "htan_colorectal_3d",
    "HuBMAP_3D_IMC_spleen": "hubmap_imc_spleen",
    "STARmap_mouse_medial_prefrontal_cortex": "starmap_mpfc",
    "STARmap-PLUS_mouse_brain_Alzheimers": "starmap_plus_alzheimers",
    "smFISH_drosophila_embryo_BDTNP": "bdtnp_drosophila_embryo",
    "ExSeq_mouse_visual_cortex": "exseq_visual_cortex",
    "EASI-FISH_drosophila_brain": "molcart_drosophila_brain",
    "STARmap_mouse_brain_activity_cFos": "starmap_cfos_activity",
    "ExSeq_human_brain_tumors": None,  # restricted
    "STARmap_mouse_cortex_hippocampus_wholecortex_Chen2024": "starmap_wholecortex_chen2024",
    "STARmap_mouse_hippocampus_ClusterMap": "starmap_visual_cortex_clustermap",
    "EASI-FISH_mouse_hypothalamus": "easi_fish_hypothalamus",
    "ExSeq_mouse_hippocampus": "exseq_breast_cancer",
    "STARmap_mouse_brain_3D_intact": "starmap_plus_3d_intact",
    "3D_intact_tissue_seq_thick_blocks_Sui2025": "deep_starmap",
    "STARmap_mouse_visual_cortex": "starmap_visual_cortex",
    "MERFISH_3D_thick_tissue_Dulac2024": "merfish_thick_tissue",
    "Visium_human_DLPFC_Maynard2021": None,  # removed — subset of visium_dlpfc_spatialDLPFC
    "ST_mouse_brain_75sections_Ortiz2020": "st_mouse_brain_ortiz",
    "Visium_mouse_spinal_cord_injury_TabulaeParalytica2024": "visium_spinal_cord",
    "ST_Visium_human_prostate_cancer_3D_Erickson2022": "st_visium_prostate_erickson",
    "ST_human_developing_heart_Asp2019": "st_human_heart_asp",
    "Visium_mouse_brain_cell2location_Kleshchevnikov2022": "visium_mouse_brain_cell2location",
    "Visium_mouse_brain_sagittal_10x": "visium_mouse_brain_sagittal",
    "Visium_human_DLPFC_30sections_spatialDLPFC_HuukiMyers2024": "visium_dlpfc_spatialDLPFC",
    "Visium_serial_sections_comprehensive_catalog": None,  # meta-catalog
}

# Secondary script that maps to same inventory row as imc_breast_cancer
SECONDARY_SCRIPTS = {
    "imc_breast_cancer_zenodo": "imc_breast_cancer",
}

OUTPUT_COLUMNS = [
    "dataset_name",
    "technology",
    "resolution_level",
    "species",
    "tissue",
    "has_expression_data",
    "num_genes",
    "num_cells_spots",
    "num_slices",
    "has_3d_location",
    "serial_or_3d",
    "are_locations_consecutive",
    "has_cell_type_annotation",
    "has_domain_annotation",
    "has_histology",
    "has_scrnaseq_companion",
    "file_formats",
    "download_script",
    "is_downloaded",
    "public",
    "accession",
]


def load_inventory(path):
    """Read dataset_inventory.csv and return list of dicts."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _extract_formats_from_script(script_path):
    """Extract file formats from a download script by parsing FILES list."""
    text = script_path.read_text(encoding="utf-8")
    formats = set()

    # Look for file extensions in "name": "..." entries
    for m in re.finditer(r'"name"\s*:\s*"([^"]+)"', text):
        name = m.group(1)
        ext = os.path.splitext(name)[1].lower()
        if ext:
            formats.add(ext.lstrip("."))

    # Also look for extensions in URLs
    for m in re.finditer(r'"url"\s*:\s*"([^"]+)"', text):
        url = m.group(1)
        # Strip query params
        url_path = url.split("?")[0]
        ext = os.path.splitext(url_path)[1].lower()
        if ext and ext.lstrip(".") not in ("com", "org", "net", "html", "htm", "php"):
            formats.add(ext.lstrip("."))

    # Normalize: gz, tar.gz handling
    # Check for .tar.gz patterns
    for m in re.finditer(r'"name"\s*:\s*"([^"]+\.tar\.gz)"', text):
        formats.discard("gz")
        formats.add("tar.gz")
    for m in re.finditer(r'"name"\s*:\s*"([^"]+\.h5ad\.gz)"', text):
        formats.discard("gz")
        formats.add("h5ad.gz")

    return formats


def scan_download_scripts(scripts_dir):
    """Scan download scripts and return dict mapping DATASET_NAME -> info dict."""
    result = {}
    for script_path in sorted(scripts_dir.glob("download_*.py")):
        if script_path.name == "download_all.py":
            continue
        text = script_path.read_text(encoding="utf-8")
        # Extract DATASET_NAME
        m = re.search(r'DATASET_NAME\s*=\s*["\']([^"\']+)["\']', text)
        if not m:
            continue
        ds_name = m.group(1)
        formats = _extract_formats_from_script(script_path)
        result[ds_name] = {
            "script_filename": script_path.name,
            "file_formats": formats,
        }
    return result


def find_markdown(inventory_filename, directories):
    """Find the markdown file given the inventory filename.

    Searches in similar-datasets/ and its subdirectories.
    """
    for d in directories:
        candidate = d / inventory_filename
        if candidate.exists():
            return candidate
    return None


def parse_markdown(filepath):
    """Parse a markdown file and extract metadata using regex."""
    if filepath is None or not filepath.exists():
        return {}

    text = filepath.read_text(encoding="utf-8")
    text_lower = text.lower()
    info = {}

    # --- num_genes ---
    # Check for whole transcriptome first
    if re.search(r"whole[\s-]?transcriptome", text_lower):
        info["num_genes"] = "whole_transcriptome"
    else:
        gene_counts = []
        # "N genes" or "N measured genes"
        for m in re.finditer(r"(\d[\d,]*)\s*(?:measured\s+)?genes", text_lower):
            gene_counts.append(int(m.group(1).replace(",", "")))
        # "N-gene"
        for m in re.finditer(r"(\d[\d,]*)-gene", text_lower):
            gene_counts.append(int(m.group(1).replace(",", "")))
        # "N-plex"
        for m in re.finditer(r"(\d[\d,]*)-plex", text_lower):
            gene_counts.append(int(m.group(1).replace(",", "")))
        if gene_counts:
            info["num_genes"] = str(max(gene_counts))

    # --- num_cells_spots ---
    cell_counts = []
    # "N million cells/spots"
    for m in re.finditer(
        r"~?([\d,.]+)\s*(?:million|M)\s*(?:cells|spots|nuclei|neurons|transcripts)",
        text, re.IGNORECASE,
    ):
        val = float(m.group(1).replace(",", ""))
        cell_counts.append(int(val * 1_000_000))
    # "NK cells/spots"
    for m in re.finditer(
        r"~?([\d,.]+)\s*K\s*(?:cells|spots|nuclei|neurons)",
        text, re.IGNORECASE,
    ):
        val = float(m.group(1).replace(",", ""))
        cell_counts.append(int(val * 1_000))
    # "N,NNN cells" or "~N cells"
    for m in re.finditer(
        r"~?([\d,]+)\s*(?:cells|spots|nuclei|neurons|beads)",
        text, re.IGNORECASE,
    ):
        val_str = m.group(1).replace(",", "")
        val = int(val_str)
        if val >= 50:  # filter noise
            cell_counts.append(val)
    # "N-N spots per section" -> take upper bound
    for m in re.finditer(
        r"~?([\d,]+)-([\d,]+)\s*(?:cells|spots|nuclei)\s*per\s*section",
        text, re.IGNORECASE,
    ):
        cell_counts.append(int(m.group(2).replace(",", "")))

    if cell_counts:
        info["num_cells_spots"] = str(max(cell_counts))

    # --- has_3d_location ---
    if re.search(r"x,?\s*y,?\s*z", text_lower) or \
       re.search(r"3d\s*coordinates", text_lower) or \
       re.search(r"z-axis", text_lower) or \
       re.search(r"\bccf\b", text_lower) or \
       re.search(r"volumetric\s*3d", text_lower) or \
       re.search(r"resolved\s*in\s*3d", text_lower) or \
       re.search(r"true\s*3d\s*spatial", text_lower):
        info["has_3d_location_from_md"] = True

    # --- are_locations_consecutive ---
    if re.search(r"non[\s-]?consecutive", text_lower):
        info["are_locations_consecutive"] = "no"
    elif re.search(r"consecutive|adjacent|evenly[\s-]?spaced|serial\s+sections|contiguous", text_lower):
        info["are_locations_consecutive"] = "yes"

    # --- has_cell_type_annotation ---
    if re.search(
        r"cell[\s-]?type\s*annotation|cell\s*classification|neuronal\s*subtypes|"
        r"cluster\s*assignment|annotated\s*cell\s*types|cell[\s-]?type\s*label|"
        r"cell[\s-]?type\s*annotations?\s*available|identifying\s*cell\s*types|"
        r"cell[\s-]?type\s*identif",
        text_lower,
    ):
        info["has_cell_type_annotation"] = "yes"

    # --- has_domain_annotation ---
    if re.search(
        r"spatial\s*domain|domain\s*annotation|layer\s*annotation|cortical\s*layer|"
        r"region\s*annotation|brain\s*region\s*label|anatomical\s*annotation|"
        r"layer\s*label|cortical\s*layers?\s*l[1-6]",
        text_lower,
    ):
        info["has_domain_annotation"] = "yes"

    return info


def inspect_downloaded(dataset_name, skip_h5ad=False):
    """Check data/raw/<dataset_name>/ and optionally read h5ad files."""
    info = {"is_downloaded": "no"}
    ds_dir = DATA_RAW_DIR / dataset_name
    if not ds_dir.exists():
        return info
    info["is_downloaded"] = "yes"

    if skip_h5ad:
        return info

    # Try to inspect h5ad files
    h5ad_files = list(ds_dir.rglob("*.h5ad"))
    if not h5ad_files:
        return info

    try:
        import anndata
    except ImportError:
        return info

    # Read the largest h5ad file
    h5ad_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    try:
        adata = anndata.read_h5ad(h5ad_files[0], backed="r")
        info["h5ad_genes"] = adata.n_vars
        info["h5ad_cells"] = adata.n_obs
        if hasattr(adata, "obsm") and adata.obsm is not None:
            spatial_keys = [k for k in adata.obsm.keys()
                           if "spatial" in k.lower() or "x_spatial" in k.lower()]
            if spatial_keys:
                info["h5ad_has_spatial"] = True
        if hasattr(adata, "obs") and adata.obs is not None:
            ann_cols = [c for c in adata.obs.columns
                       if "type" in c.lower() or "cluster" in c.lower()]
            if ann_cols:
                info["h5ad_has_cell_type"] = True
        adata.file.close()
    except Exception:
        pass

    return info


def assemble_row(inv_row, script_info, md_info, dl_info):
    """Assemble a single output row from all sources."""
    inv_filename = inv_row["filename"]
    inv_stem = inv_filename.replace(".md", "")
    dataset_name = INVENTORY_TO_DATASET.get(inv_stem)
    serial_or_3d = inv_row.get("serial_or_3d", "")

    row = {}
    row["dataset_name"] = dataset_name if dataset_name else "N/A"
    row["technology"] = inv_row.get("technology", "")
    row["resolution_level"] = inv_row.get("resolution_level", "")
    row["species"] = inv_row.get("species", "")
    row["tissue"] = inv_row.get("tissue", "")

    # has_expression_data: default yes for spatial transcriptomics
    row["has_expression_data"] = "yes"

    # num_genes from markdown or h5ad
    num_genes = md_info.get("num_genes", "")
    if "h5ad_genes" in dl_info and num_genes != "whole_transcriptome":
        num_genes = str(dl_info["h5ad_genes"])
    row["num_genes"] = num_genes

    # num_cells_spots from markdown or h5ad
    num_cells = md_info.get("num_cells_spots", "")
    if "h5ad_cells" in dl_info:
        num_cells = str(dl_info["h5ad_cells"])
    row["num_cells_spots"] = num_cells

    # num_slices from inventory
    row["num_slices"] = inv_row.get("num_sections", "")

    # has_3d_location
    if "intact_3d" in serial_or_3d:
        row["has_3d_location"] = "yes"
    elif md_info.get("has_3d_location_from_md"):
        row["has_3d_location"] = "yes"
    elif dl_info.get("h5ad_has_spatial"):
        row["has_3d_location"] = "yes"
    else:
        row["has_3d_location"] = "no"

    row["serial_or_3d"] = serial_or_3d

    # are_locations_consecutive
    if "intact_3d" in serial_or_3d:
        row["are_locations_consecutive"] = "yes"
    elif "non-consecutive" in serial_or_3d:
        row["are_locations_consecutive"] = "no"
    elif "are_locations_consecutive" in md_info:
        row["are_locations_consecutive"] = md_info["are_locations_consecutive"]
    else:
        row["are_locations_consecutive"] = "unknown"

    # has_cell_type_annotation
    if dl_info.get("h5ad_has_cell_type"):
        row["has_cell_type_annotation"] = "yes"
    else:
        row["has_cell_type_annotation"] = md_info.get("has_cell_type_annotation", "unknown")

    # has_domain_annotation
    row["has_domain_annotation"] = md_info.get("has_domain_annotation", "unknown")

    # has_histology
    row["has_histology"] = inv_row.get("has_histology", "no")

    # has_scrnaseq_companion
    row["has_scrnaseq_companion"] = inv_row.get("has_scrnaseq_companion", "no")

    # file_formats and download_script
    if dataset_name is None:
        row["file_formats"] = "N/A"
        row["download_script"] = "N/A"
    elif dataset_name in script_info:
        si = script_info[dataset_name]
        formats = set(si["file_formats"])
        scripts = [si["script_filename"]]
        # Check for secondary scripts
        for sec_name, primary in SECONDARY_SCRIPTS.items():
            if primary == dataset_name and sec_name in script_info:
                formats |= script_info[sec_name]["file_formats"]
                scripts.append(script_info[sec_name]["script_filename"])
        row["file_formats"] = "; ".join(sorted(formats)) if formats else ""
        row["download_script"] = "; ".join(sorted(scripts))
    else:
        row["file_formats"] = ""
        row["download_script"] = ""

    # is_downloaded
    if dataset_name is None:
        row["is_downloaded"] = "no"
    else:
        row["is_downloaded"] = dl_info.get("is_downloaded", "no")

    # public
    row["public"] = inv_row.get("public", "")

    # accession
    row["accession"] = inv_row.get("accession", "")

    return row


def write_csv(rows, output_path):
    """Write audit rows to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description="Audit all datasets in the 3D tissue project.")
    parser.add_argument("--skip-h5ad", action="store_true", help="Skip h5ad file inspection")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--verbose", action="store_true", help="Print progress")
    opts = parser.parse_args(args)

    output_path = Path(opts.output) if opts.output else DEFAULT_OUTPUT

    if opts.verbose:
        print(f"Loading inventory from {INVENTORY_CSV}")
    inventory = load_inventory(INVENTORY_CSV)
    if opts.verbose:
        print(f"  Found {len(inventory)} entries")

    if opts.verbose:
        print(f"Scanning download scripts in {DOWNLOAD_SCRIPTS_DIR}")
    script_info = scan_download_scripts(DOWNLOAD_SCRIPTS_DIR)
    if opts.verbose:
        print(f"  Found {len(script_info)} scripts")

    # Build list of directories to search for markdown files
    md_dirs = [SIMILAR_DATASETS_DIR]
    for d in SIMILAR_DATASETS_DIR.iterdir():
        if d.is_dir():
            md_dirs.append(d)

    rows = []
    for inv_row in inventory:
        inv_filename = inv_row["filename"]
        inv_stem = inv_filename.replace(".md", "")
        dataset_name = INVENTORY_TO_DATASET.get(inv_stem)

        if opts.verbose:
            print(f"  Processing: {inv_stem} -> {dataset_name}")

        # Find and parse markdown
        md_path = find_markdown(inv_filename, md_dirs)
        md_info = parse_markdown(md_path)

        # Check downloaded data
        if dataset_name:
            dl_info = inspect_downloaded(dataset_name, skip_h5ad=opts.skip_h5ad)
        else:
            dl_info = {"is_downloaded": "no"}

        row = assemble_row(inv_row, script_info, md_info, dl_info)
        rows.append(row)

    write_csv(rows, output_path)
    if opts.verbose:
        print(f"\nWrote {len(rows)} rows to {output_path}")
        # Summary
        downloaded = sum(1 for r in rows if r["is_downloaded"] == "yes")
        has_3d = sum(1 for r in rows if r["has_3d_location"] == "yes")
        print(f"  Downloaded: {downloaded}/{len(rows)}")
        print(f"  Has 3D location: {has_3d}/{len(rows)}")


if __name__ == "__main__":
    main()
