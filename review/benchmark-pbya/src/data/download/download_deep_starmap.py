#!/usr/bin/env python
"""Download Deep-STARmap dataset (Sui et al. 2025).

Source: Zenodo 16783355
Target: data/raw/deep_starmap/
Requirements: expression data, 3D coordinates, cell-type annotations
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "deep_starmap"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ZENODO_RECORD = "16783355"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

FILES = [
    {
        "url": f"{ZENODO_BASE}/Brain_Deep_STARmap_expression_matrix.csv",
        "name": "Brain_Deep_STARmap_expression_matrix.csv",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/Brain_Deep_STARmap_spatial.csv",
        "name": "Brain_Deep_STARmap_spatial.csv",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/Brain_Deep_STARmap_X_umap.csv",
        "name": "Brain_Deep_STARmap_X_umap.csv",
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

            if md5:
                actual_md5 = hashlib.md5(dest.read_bytes()).hexdigest()
                if actual_md5 != md5:
                    print(f"  MD5 mismatch for {dest.name}: expected {md5}, got {actual_md5}")
                    if attempt < max_retries:
                        dest.unlink()
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
    """Verify downloaded files: sizes + CSV content checks."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    for f in FILES:
        path = BASE_DIR / f["name"]
        if not path.exists():
            print(f"  MISSING: {f['name']}")
            ok = False
            continue
        print(f"  OK: {f['name']} ({path.stat().st_size:,} bytes)")

    # CSV content checks
    try:
        import pandas as pd

        expr_path = BASE_DIR / "Brain_Deep_STARmap_expression_matrix.csv"
        if expr_path.exists():
            expr = pd.read_csv(expr_path, nrows=5)
            print(f"  Expression matrix: {expr.shape[1]} columns")

        spatial_path = BASE_DIR / "Brain_Deep_STARmap_spatial.csv"
        if spatial_path.exists():
            spatial = pd.read_csv(spatial_path, nrows=5)
            print(f"  Spatial: columns = {list(spatial.columns)}")
            coord_cols = [c for c in spatial.columns if any(x in c.lower() for x in ["x", "y", "z"])]
            if len(coord_cols) >= 3:
                print(f"  3D coordinates found: {coord_cols}")
            else:
                print(f"  WARNING: Expected 3D coordinates, found: {coord_cols}")

        umap_path = BASE_DIR / "Brain_Deep_STARmap_X_umap.csv"
        if umap_path.exists():
            umap = pd.read_csv(umap_path, nrows=5)
            print(f"  UMAP: columns = {list(umap.columns)}")

    except ImportError:
        print("  WARNING: pandas not installed, skipping CSV content check")
    except Exception as e:
        print(f"  CSV CHECK FAILED: {e}")
        ok = False

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
