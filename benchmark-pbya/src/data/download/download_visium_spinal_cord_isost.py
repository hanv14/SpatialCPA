#!/usr/bin/env python
"""Download Visium spinal cord dataset (isoST paper).

Source: GEO GSE234774
Target: data/raw/visium_spinal_cord_isost/
Description: Mouse mid-thoracic spinal cord Visium spatial transcriptomics.
  34 Visium sections across conditions (uninjured, injury timepoints, drug treatments).
  The spatial_3d files contain 3D-reconstructed data with x,y,z coordinates.
  The spatial_2d files contain per-section 2D data.
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "visium_spinal_cord_isost"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE234nnn/GSE234774/suppl"

# Spatial 3D files: expression matrix + metadata with 3D coordinates
FILES = [
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_3d_barcodes.txt.gz",
        "name": "GSE234774_spatial_3d_barcodes.txt.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_3d_features.txt.gz",
        "name": "GSE234774_spatial_3d_features.txt.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_3d_filtered_spatial_3d.mtx.gz",
        "name": "GSE234774_spatial_3d_filtered_spatial_3d.mtx.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_3d_meta.txt.gz",
        "name": "GSE234774_spatial_3d_meta.txt.gz",
        "size": None,
    },
    # Also download 2D spatial data as fallback
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_2d_barcodes.txt.gz",
        "name": "GSE234774_spatial_2d_barcodes.txt.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_2d_features.txt.gz",
        "name": "GSE234774_spatial_2d_features.txt.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_2d_filtered_spatial_2d.mtx.gz",
        "name": "GSE234774_spatial_2d_filtered_spatial_2d.mtx.gz",
        "size": None,
    },
    {
        "url": f"{GEO_BASE}/GSE234774_spatial_2d_meta.txt.gz",
        "name": "GSE234774_spatial_2d_meta.txt.gz",
        "size": None,
    },
    # RAW tar with per-section spatial images and JSONs
    {
        "url": f"{GEO_BASE}/GSE234774_RAW.tar",
        "name": "GSE234774_RAW.tar",
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


def extract_tar():
    """Extract RAW tar archive."""
    import tarfile

    tar_path = BASE_DIR / "GSE234774_RAW.tar"
    if not tar_path.exists():
        print(f"  Cannot extract: {tar_path.name} not found")
        return False

    # Check if already extracted
    extracted = list(BASE_DIR.glob("GSM*.csv.gz")) + list(BASE_DIR.glob("GSM*.json.gz"))
    if len(extracted) > 5:
        print(f"  Already extracted ({len(extracted)} files)")
        return True

    print(f"\nExtracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getmembers()
            print(f"  Archive contains {len(members)} files")
            for m in members:
                m.name = os.path.basename(m.name)
            tf.extractall(path=BASE_DIR)
        print(f"  Extracted to {BASE_DIR}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
        return False


def verify():
    """Verify downloaded files."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    required = [
        "GSE234774_spatial_3d_barcodes.txt.gz",
        "GSE234774_spatial_3d_features.txt.gz",
        "GSE234774_spatial_3d_filtered_spatial_3d.mtx.gz",
        "GSE234774_spatial_3d_meta.txt.gz",
    ]
    for fname in required:
        path = BASE_DIR / fname
        if not path.exists():
            print(f"  MISSING: {fname}")
            ok = False
        else:
            print(f"  OK: {fname} ({path.stat().st_size:,} bytes)")

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
        extract_tar()
        verify()
    else:
        print("Some downloads failed. Skipping verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
