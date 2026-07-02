#!/usr/bin/env python
"""Download Array-seq mouse kidney 3D dataset.

Source: GEO GSE266244
Target: data/raw/arrayseq_kidney/
Description: 8 serial kidney sections (100 µm spacing) from Array-seq (isoST paper).
Key file: GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad (427 MB) - pre-aligned 3D kidney data.
Also downloads per-section h5ad files (AS_KI_1..4) and barcode mapping.
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "arrayseq_kidney"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

GEO_SERIES_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE266nnn/GSE266244/suppl"
GEO_SAMPLE_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM8243nnn/GSM8243015/suppl"

# Download the 3D kidney h5ad directly (427 MB) instead of the full RAW tar (11 GB)
# which contains many unneeded datasets (human spleen, brain, liver, etc.)
FILES = [
    {
        "url": f"{GEO_SAMPLE_BASE}/GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad",
        "name": "GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad",
        "size": 426776445,
    },
    {
        "url": f"{GEO_SERIES_BASE}/GSE266244_ArraySeq_Barcode_Mapping_n12.csv.gz",
        "name": "GSE266244_ArraySeq_Barcode_Mapping_n12.csv.gz",
        "size": None,
    },
]


def download_file(url, dest, expected_size=None, md5=None, max_retries=3):
    """Download with resume support, retry with backoff, and progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            print(f"  Already downloaded: {dest.name}")
            return True
        elif expected_size and dest.stat().st_size > expected_size:
            print(f"  File larger than expected, re-downloading: {dest.name}")
            dest.unlink()
        elif not expected_size and dest.stat().st_size > 0:
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
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if expected_size and dest.stat().st_size != expected_size:
                print(f"  Size mismatch for {dest.name}: expected {expected_size}, got {dest.stat().st_size}")
                if attempt < max_retries:
                    continue
                return False

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

    key_file = BASE_DIR / "GSM8243015_3D_KI_Raw_Z_aligned_annotated.h5ad"
    if not key_file.exists():
        print(f"  MISSING: {key_file.name}")
        ok = False
    else:
        print(f"  OK: {key_file.name} ({key_file.stat().st_size:,} bytes)")

    barcode_map = BASE_DIR / "GSE266244_ArraySeq_Barcode_Mapping_n12.csv.gz"
    if barcode_map.exists():
        print(f"  OK: {barcode_map.name} ({barcode_map.stat().st_size:,} bytes)")

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print(f"Downloading {DATASET_NAME}...")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    success = True
    for f in FILES:
        if not download_file(f["url"], BASE_DIR / f["name"], f.get("size"), f.get("md5")):
            success = False

    if success:
        verify()
    else:
        print("Some downloads failed. Skipping verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
