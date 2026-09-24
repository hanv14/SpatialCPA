#!/usr/bin/env python
"""Download ExSeq visual cortex processed dataset from SpaceTx.

Dataset: Targeted ExSeq mouse primary visual cortex (VISp)
- 1,154 cells (SpaceJam2) / 1,271 cells (SpaceTx website), 42 genes
- 3D spatial coordinates (x_um, y_um, z_um)
- Technology: Expansion Sequencing (targeted panel)
- Source: SpaceTx consortium (spacetx.github.io)
- Paper: Alon et al., Science 2021 (DOI: 10.1126/science.aax2656)
- SpaceTx paper: Bakken et al., Scientific Reports 2023

Target: data/raw/exseq_visual_cortex/
"""
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "exseq_visual_cortex"
DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

S3_BASE = "https://s3.amazonaws.com/starfish.data.spacetx"

# SpaceJam2 version (original segmentation, includes 3D coords)
SPACEJAM2_FILES = [
    {
        "url": f"{S3_BASE}/spacejam2/ExSeq/cellxgene.csv",
        "name": "spacejam2_cellxgene.csv",
        "desc": "Cell-by-gene matrix + 3D coords (1154 cells x 42 genes)",
    },
    {
        "url": f"{S3_BASE}/spacejam2/ExSeq/spottable_exseq.csv",
        "name": "spacejam2_spottable.csv",
        "desc": "Individual transcript spots (~265K spots)",
    },
]

# SpaceTx website version (newer Baysor segmentation)
SPACETX_FILES = [
    {
        "url": f"{S3_BASE}/spacetx-website/data/ExSeq/s3_cell_by_gene.csv",
        "name": "spacetx_cell_by_gene.csv",
        "desc": "Cell-by-gene matrix (1271 cells x 42 genes)",
    },
    {
        "url": f"{S3_BASE}/spacetx-website/data/ExSeq/s3_mapped_cell_table.csv",
        "name": "spacetx_mapped_cell_table.csv",
        "desc": "Cell metadata (x, y, cluster, area, confidence)",
    },
    {
        "url": f"{S3_BASE}/spacetx-website/data/ExSeq/s3_spot_table.csv",
        "name": "spacetx_spot_table.csv",
        "desc": "Individual transcript spots with cell assignments (~265K spots)",
    },
    {
        "url": f"{S3_BASE}/spacetx-website/data/ExSeq/ExSeq_Readme.txt",
        "name": "ExSeq_Readme.txt",
        "desc": "Dataset documentation",
    },
]

ALL_FILES = SPACEJAM2_FILES + SPACETX_FILES


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

            response = requests.get(url, headers=headers, stream=True, timeout=120)

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

    for entry in ALL_FILES:
        dest = DATA_DIR / entry["name"]
        if not dest.exists() or dest.stat().st_size == 0:
            print(f"  MISSING: {entry['name']}")
            ok = False
        else:
            print(f"  OK: {entry['name']} ({dest.stat().st_size:,} bytes)")

    # Quick content check on the key file
    cellxgene = DATA_DIR / "spacejam2_cellxgene.csv"
    if cellxgene.exists():
        try:
            import pandas as pd
            df = pd.read_csv(cellxgene, nrows=5)
            print(f"  spacejam2_cellxgene.csv: {df.shape[1]} columns")
            if "x_um" in df.columns and "y_um" in df.columns and "z_um" in df.columns:
                print(f"  3D coordinates present (x_um, y_um, z_um)")
            else:
                print(f"  WARNING: 3D coordinate columns not found")
        except ImportError:
            pass
        except Exception as e:
            print(f"  CSV CHECK: {e}")

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print("=" * 60)
    print(f"Downloading: {DATASET_NAME}")
    print(f"Source: SpaceTx consortium (spacetx.github.io)")
    print(f"Destination: {DATA_DIR}")
    print(f"Contents: Targeted ExSeq, 42 genes, ~1154-1271 cells, 3D coords")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for entry in ALL_FILES:
        dest = DATA_DIR / entry["name"]
        print(f"\nDownloading: {entry['name']} ({entry['desc']})")
        if download_file(entry["url"], dest):
            success_count += 1
        else:
            print(f"  FAILED: {entry['name']}")

    print(f"\nDownloaded {success_count}/{len(ALL_FILES)} files")
    verify()

    if success_count < len(ALL_FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
