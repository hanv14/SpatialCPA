#!/usr/bin/env python
"""Download MERFISH hypothalamus dataset (Moffitt et al. 2018).

Source: Figshare (5 processed sections) + Dryad (full dataset)
Target: data/raw/merfish_hypothalamus/
Requirements: expression data, 3D coordinates, cell-type annotations
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "merfish_hypothalamus"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

# Figshare processed sections
FIGSHARE_SECTION_IDS = {
    "Moffitt2018_MERFISH_hypothalamus_section1.h5ad": 58960003,
    "Moffitt2018_MERFISH_hypothalamus_section2.h5ad": 58960006,
    "Moffitt2018_MERFISH_hypothalamus_section3.h5ad": 58960012,
    "Moffitt2018_MERFISH_hypothalamus_section4.h5ad": 58960015,
    "Moffitt2018_MERFISH_hypothalamus_section5.h5ad": 58960018,
}

FILES = [
    {
        "url": f"https://ndownloader.figshare.com/files/{fid}",
        "name": name,
        "size": None,
    }
    for name, fid in FIGSHARE_SECTION_IDS.items()
] + [
    {
        "url": "https://zenodo.org/records/4953173/files/Moffitt_and_Bambah-Mukku_et_al_merfish_all_cells.csv?download=1",
        "name": "Moffitt2018_MERFISH_hypothalamus_full.csv",
        "size": 1031419408,
        "md5": "25a51abdf981039949cfdaf4db0a9ab3",
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

            response = requests.get(url, headers=headers, stream=True, timeout=60)

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
                print(f"  Verifying MD5 for {dest.name}...")
                h = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                actual_md5 = h.hexdigest()
                if actual_md5 != md5:
                    print(f"  MD5 mismatch for {dest.name}: expected {md5}, got {actual_md5}")
                    if attempt < max_retries:
                        dest.unlink()
                        continue
                    return False
                print(f"  MD5 OK: {actual_md5}")

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

    # h5ad content checks on first section
    try:
        import anndata
        first_section = BASE_DIR / list(FIGSHARE_SECTION_IDS.keys())[0]
        if first_section.exists():
            adata = anndata.read_h5ad(first_section)
            assert adata.X is not None, "Missing expression matrix (X)"
            assert adata.obs is not None and len(adata.obs) > 0, "Missing obs annotations"
            print(f"  h5ad OK (section 1): {adata.n_obs} cells, {adata.n_vars} genes")
            print(f"  obsm keys: {list(adata.obsm.keys())}")
            print(f"  obs columns: {list(adata.obs.columns[:10])}")
    except ImportError:
        print("  WARNING: anndata not installed, skipping h5ad content check")
    except Exception as e:
        print(f"  h5ad CHECK FAILED: {e}")
        ok = False

    # CSV content check on Dryad file
    try:
        import pandas as pd
        csv_path = BASE_DIR / "Moffitt2018_MERFISH_hypothalamus_full.csv"
        if csv_path.exists():
            # Read just first few rows to check structure
            df = pd.read_csv(csv_path, nrows=5)
            print(f"  CSV columns: {list(df.columns[:15])}")
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
