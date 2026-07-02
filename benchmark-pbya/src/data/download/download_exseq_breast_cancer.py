#!/usr/bin/env python
"""Download ExSeq human breast cancer dataset from Zenodo.

NOTE: The mouse hippocampus ExSeq processed data was never publicly deposited.
This script instead downloads the only publicly available processed ExSeq data:
the human metastatic breast cancer biopsy (HTAPP) from Zenodo 4479018.

Dataset: Targeted ExSeq on human metastatic breast cancer (liver metastasis)
- ~1,222 cells, 297 genes, 3D spatial coordinates (in .mat file)
- Technology: Expansion Sequencing (targeted panel)
- Source: Zenodo 4479018 (Figure 6 of Alon et al., Science 2021)
- Paper: Alon et al., Science 2021 (DOI: 10.1126/science.aax2656)
- Specimen: HTAPP (Human Tumor Atlas Pilot Project) breast cancer biopsy

Target: data/raw/exseq_breast_cancer/
"""
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "exseq_breast_cancer"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ZENODO_BASE = "https://zenodo.org/records/4479018/files"

# Key data files from Zenodo 4479018
FILES = [
    {
        "url": f"{ZENODO_BASE}/cell_expression-1222.txt?download=1",
        "name": "cell_expression-1222.txt",
        "desc": "Cell-by-gene expression matrix (raw counts)",
    },
    {
        "url": f"{ZENODO_BASE}/cell_expression_normed-1222.txt?download=1",
        "name": "cell_expression_normed-1222.txt",
        "desc": "Normalized cell-by-gene expression matrix",
    },
    {
        "url": f"{ZENODO_BASE}/CellsIDs-1222.txt?download=1",
        "name": "CellsIDs-1222.txt",
        "desc": "Cell identifiers",
    },
    {
        "url": f"{ZENODO_BASE}/GeneNames-1222.txt?download=1",
        "name": "GeneNames-1222.txt",
        "desc": "Gene names (297 targeted genes)",
    },
    {
        "url": f"{ZENODO_BASE}/HTAPP-20191122-alltranscripts_H1Q1.mat?download=1",
        "name": "HTAPP-20191122-alltranscripts_H1Q1.mat",
        "desc": "MATLAB data with 3D spatial coordinates (4.3 GB)",
    },
    {
        "url": f"{ZENODO_BASE}/HTAPP-transcriptobjects-SeuratClassified-1222.mat?download=1",
        "name": "HTAPP-transcriptobjects-SeuratClassified-1222.mat",
        "desc": "Transcript objects with Seurat cell type classifications",
    },
    {
        "url": f"{ZENODO_BASE}/SeuratCellsIDs_2Dseg_20191222.csv?download=1",
        "name": "SeuratCellsIDs_2Dseg_20191222.csv",
        "desc": "Cell segmentation and Seurat classification data",
    },
    {
        "url": f"{ZENODO_BASE}/Fig6-Seurat.ipynb?download=1",
        "name": "Fig6-Seurat.ipynb",
        "desc": "Seurat analysis notebook",
    },
    {
        "url": f"{ZENODO_BASE}/README.md?download=1",
        "name": "zenodo_README.md",
        "desc": "Dataset documentation",
    },
]


def download_file(url, dest, max_retries=3):
    """Download with resume support, retry with backoff, and progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True

    for attempt in range(1, max_retries + 1):
        try:
            headers = {}
            mode = "wb"
            initial_size = 0
            if dest.exists():
                initial_size = dest.stat().st_size
                headers["Range"] = f"bytes={initial_size}-"
                mode = "ab"

            response = requests.get(url, headers=headers, stream=True, timeout=120,
                                    allow_redirects=True)

            if response.status_code == 200 and initial_size > 0:
                initial_size = 0
                mode = "wb"
            elif response.status_code == 416:
                print(f"  Already downloaded: {dest.name}")
                return True
            elif response.status_code not in (200, 206):
                response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if response.status_code == 206:
                total_size += initial_size

            with open(dest, mode) as f, tqdm(
                total=total_size or None,
                initial=initial_size,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            return True

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Failed to download {dest.name} after {max_retries} attempts")
                return False

    return False


def verify():
    """Verify downloaded files."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    for entry in FILES:
        dest = DATA_DIR / entry["name"]
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"  MISSING: {entry['name']}")
            ok = False
        else:
            print(f"  OK: {entry['name']} ({dest.stat().st_size:,} bytes)")

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print("=" * 60)
    print(f"Downloading: {DATASET_NAME}")
    print(f"Source: Zenodo 4479018 (ExSeq HTAPP breast cancer)")
    print(f"Destination: {DATA_DIR}")
    print(f"Note: Mouse hippocampus ExSeq data not publicly available;")
    print(f"      downloading human breast cancer ExSeq data instead")
    print(f"Estimated size: ~5 GB")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for entry in FILES:
        dest = DATA_DIR / entry["name"]
        print(f"\nDownloading: {entry['name']} ({entry['desc']})")
        if download_file(entry["url"], dest):
            success_count += 1
        else:
            print(f"  FAILED: {entry['name']}")

    print(f"\nDownloaded {success_count}/{len(FILES)} files")
    verify()

    if success_count < len(FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
