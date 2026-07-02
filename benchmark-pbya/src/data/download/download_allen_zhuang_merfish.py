#!/usr/bin/env python
"""Download Allen Brain Cell Atlas MERFISH data (Zhuang lab).

Source: AWS S3 (Allen Brain Cell Atlas)
Target: data/raw/allen_zhuang_merfish/
Contents: 8.4M cells, 1,122 genes across ABCA-1 through ABCA-4 parcellations
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "allen_zhuang_merfish"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

S3_BASE = "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com"

FILES = [
    # Per-region expression matrices (Zhuang-ABCA-1 through 4)
    {
        "url": f"{S3_BASE}/expression_matrices/Zhuang-ABCA-1/20230830/Zhuang-ABCA-1-log2.h5ad",
        "name": "Zhuang-ABCA-1-log2.h5ad",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/expression_matrices/Zhuang-ABCA-2/20230830/Zhuang-ABCA-2-log2.h5ad",
        "name": "Zhuang-ABCA-2-log2.h5ad",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/expression_matrices/Zhuang-ABCA-3/20230830/Zhuang-ABCA-3-log2.h5ad",
        "name": "Zhuang-ABCA-3-log2.h5ad",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/expression_matrices/Zhuang-ABCA-4/20230830/Zhuang-ABCA-4-log2.h5ad",
        "name": "Zhuang-ABCA-4-log2.h5ad",
        "size": None,
    },
    # Per-region metadata CSVs with cluster annotations + CCF coordinates
    {
        "url": f"{S3_BASE}/metadata/Zhuang-ABCA-1/20231215/views/cell_metadata_with_cluster_annotation.csv",
        "name": "Zhuang-ABCA-1-cell_metadata.csv",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/metadata/Zhuang-ABCA-2/20231215/views/cell_metadata_with_cluster_annotation.csv",
        "name": "Zhuang-ABCA-2-cell_metadata.csv",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/metadata/Zhuang-ABCA-3/20231215/views/cell_metadata_with_cluster_annotation.csv",
        "name": "Zhuang-ABCA-3-cell_metadata.csv",
        "size": None,
    },
    {
        "url": f"{S3_BASE}/metadata/Zhuang-ABCA-4/20231215/views/cell_metadata_with_cluster_annotation.csv",
        "name": "Zhuang-ABCA-4-cell_metadata.csv",
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
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if expected_size and dest.stat().st_size != expected_size:
                print(f"  Size mismatch for {dest.name}: expected {expected_size}, got {dest.stat().st_size}")
                if attempt < max_retries:
                    continue
                return False

            if md5:
                h = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                actual_md5 = h.hexdigest()
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

        if f.get("size") and path.stat().st_size != f["size"]:
            print(f"  SIZE MISMATCH: {f['name']} (expected {f['size']}, got {path.stat().st_size})")
            ok = False
            continue

        print(f"  OK: {f['name']} ({path.stat().st_size:,} bytes)")

    # h5ad content checks
    try:
        import anndata
        for f in FILES:
            if not f["name"].endswith(".h5ad"):
                continue
            h5ad_path = BASE_DIR / f["name"]
            if h5ad_path.exists():
                adata = anndata.read_h5ad(h5ad_path, backed="r")
                assert adata.X is not None, f"Missing expression matrix (X) in {f['name']}"
                print(f"  {f['name']}: {adata.n_obs} cells, {adata.n_vars} genes")
                adata.file.close()
    except ImportError:
        print("  WARNING: anndata not installed, skipping h5ad content check")
    except Exception as e:
        print(f"  h5ad CHECK FAILED: {e}")
        ok = False

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print(f"Downloading {DATASET_NAME} (4 regional h5ad + 4 metadata CSVs)...")
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
