#!/usr/bin/env python
"""Download STARmap visual cortex dataset.

Source: Figshare (Wang et al. 2018)
Target: data/raw/starmap_visual_cortex/
Requirements: expression data, 3D coordinates, cell-type annotations
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "starmap_visual_cortex"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

FILES = [
    {
        "url": "https://ndownloader.figshare.com/files/58960009",
        "name": "STARmap_Wang2018three_data_3D_data.h5ad",
        "size": 30475717,
    },
]


def download_file(url, dest, expected_size=None, md5=None, max_retries=3):
    """Download with resume support, retry with backoff, and progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            print(f"  Already downloaded: {dest.name}")
            return True
        elif expected_size and dest.stat().st_size > expected_size:
            print(f"  File larger than expected, re-downloading: {dest.name}")
            dest.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            # Support resume
            headers = {}
            mode = "wb"
            initial_size = 0
            if dest.exists():
                initial_size = dest.stat().st_size
                headers["Range"] = f"bytes={initial_size}-"
                mode = "ab"

            response = requests.get(url, headers=headers, stream=True, timeout=60)

            # If server doesn't support range, start over
            if response.status_code == 200 and initial_size > 0:
                initial_size = 0
                mode = "wb"
            elif response.status_code == 416:
                # Range not satisfiable — file may be complete
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

            # Verify size
            if expected_size and dest.stat().st_size != expected_size:
                print(f"  Size mismatch for {dest.name}: expected {expected_size}, got {dest.stat().st_size}")
                if attempt < max_retries:
                    continue
                return False

            # Verify MD5
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
    """Verify downloaded files: sizes + h5ad content checks."""
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
        adata = anndata.read_h5ad(BASE_DIR / FILES[0]["name"])
        assert adata.X is not None, "Missing expression matrix (X)"
        assert adata.obs is not None and len(adata.obs) > 0, "Missing obs annotations"
        assert "spatial" in adata.obsm or "X_spatial" in adata.obsm or any("3d" in k.lower() or "spatial" in k.lower() for k in adata.obsm.keys()), \
            f"Missing spatial coordinates in obsm. Keys: {list(adata.obsm.keys())}"
        print(f"  h5ad OK: {adata.n_obs} cells, {adata.n_vars} genes")
        print(f"  obsm keys: {list(adata.obsm.keys())}")
        print(f"  obs columns: {list(adata.obs.columns[:10])}")
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
