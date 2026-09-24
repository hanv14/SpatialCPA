#!/usr/bin/env python
"""Download ST mouse brain 75 sections dataset (Ortiz et al.).

Source: GEO GSE147747
Target: data/raw/st_mouse_brain_ortiz/
Requirements: expression data, 3D coordinates, cell-type annotations
"""
import os
import sys
import time
import hashlib
import gzip
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "st_mouse_brain_ortiz"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

FILES = [
    {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147747/suppl/GSE147747_expr_raw_counts_table.tsv.gz",
        "name": "GSE147747_expr_raw_counts_table.tsv.gz",
        "size": None,
    },
    {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147747/suppl/GSE147747_meta_table.tsv.gz",
        "name": "GSE147747_meta_table.tsv.gz",
        "size": None,
    },
    {
        "url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE147nnn/GSE147747/soft/GSE147747_family.soft.gz",
        "name": "GSE147747_family.soft.gz",
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
    """Verify downloaded files: sizes + content checks."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    for f in FILES:
        path = BASE_DIR / f["name"]
        if not path.exists():
            print(f"  MISSING: {f['name']}")
            ok = False
            continue
        print(f"  OK: {f['name']} ({path.stat().st_size:,} bytes)")

    # Content checks
    try:
        import pandas as pd

        # Check metadata
        meta_path = BASE_DIR / "GSE147747_meta_table.tsv.gz"
        if meta_path.exists():
            meta = pd.read_csv(meta_path, sep="\t", compression="gzip", nrows=10)
            print(f"  Meta columns: {list(meta.columns[:15])}")
            coord_cols = [c for c in meta.columns if any(x in c.lower() for x in ["x", "y", "z", "coord", "spatial", "pos"])]
            if coord_cols:
                print(f"  Coordinate columns: {coord_cols}")

        # Check expression
        expr_path = BASE_DIR / "GSE147747_expr_raw_counts_table.tsv.gz"
        if expr_path.exists():
            expr = pd.read_csv(expr_path, sep="\t", compression="gzip", nrows=5)
            print(f"  Expression: {expr.shape[1]} columns (first 5: {list(expr.columns[:5])})")

    except ImportError:
        print("  WARNING: pandas not installed, skipping content check")
    except Exception as e:
        print(f"  CONTENT CHECK FAILED: {e}")
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
