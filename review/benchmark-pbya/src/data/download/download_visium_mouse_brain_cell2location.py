#!/usr/bin/env python
"""Download Visium mouse brain cell2location dataset.

Source: ArrayExpress E-MTAB-11114
Target: data/raw/visium_mouse_brain_cell2location/
Requirements: snRNA-seq reference and Visium spatial data for cell2location deconvolution

Files also accessible via https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-11114
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "visium_mouse_brain_cell2location"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

# The h5ad files are hosted on the cell2location data portal (Sanger),
# not on ArrayExpress which only has the raw Visium outputs.
# See: https://cell2location.readthedocs.io/en/latest/notebooks/cell2location_estimating_signatures.html
SANGER_BASE = "https://cell2location.cog.sanger.ac.uk/tutorial"

FILES = [
    {
        "url": f"{SANGER_BASE}/mouse_brain_snrna/all_cells_20200625.h5ad",
        "name": "all_cells_20200625.h5ad",
        "size": None,
    },
    {
        "url": f"{SANGER_BASE}/mouse_brain_visium_wo_cloupe_data.zip",
        "name": "mouse_brain_visium_wo_cloupe_data.zip",
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

    # Content checks: try reading h5ad files with anndata
    try:
        import anndata

        for f in FILES:
            if f["name"].endswith(".h5ad"):
                h5ad_path = BASE_DIR / f["name"]
                if h5ad_path.exists():
                    adata = anndata.read_h5ad(h5ad_path, backed="r")
                    print(f"  {f['name']}: {adata.shape[0]} cells x {adata.shape[1]} genes")
                    if adata.obs.columns.any():
                        print(f"    obs columns: {list(adata.obs.columns[:10])}")
                    if hasattr(adata, "obsm") and adata.obsm:
                        print(f"    obsm keys: {list(adata.obsm.keys())}")
                    adata.file.close()
    except ImportError:
        print("  WARNING: anndata not installed, skipping h5ad content check")
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
