#!/usr/bin/env python
"""Download ST breast cancer data (Stahl et al., 2016, Science) for stVGP paper.

Source: spatialresearch.org (DOI: 10.1126/science.aaf2403)
  - 4 serial cryosections of human breast cancer tissue
  - Count matrices as TSV files (spots x genes)
  - Spot coordinates encoded in row names as "XxY" (array indices)
Target: data/raw/st_breast_cancer_stvgp/
Description: Original Spatial Transcriptomics dataset, 4 consecutive 10 µm sections.
  ~250-264 spots per section, ~14,800-14,900 genes per section.
  Used in stVGP paper (Wang et al., 2026, Advanced Science) as breast cancer benchmark.
"""
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "st_breast_cancer_stvgp"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

SR_BASE = "https://www.spatialresearch.org/wp-content/uploads/2016/07"

FILES = [
    {
        "url": f"{SR_BASE}/Layer1_BC_count_matrix-1.tsv",
        "name": "Layer1_BC_count_matrix-1.tsv",
        "expected_size": 7666023,
    },
    {
        "url": f"{SR_BASE}/Layer2_BC_count_matrix-1.tsv",
        "name": "Layer2_BC_count_matrix-1.tsv",
        "expected_size": None,
    },
    {
        "url": f"{SR_BASE}/Layer3_BC_count_matrix-1.tsv",
        "name": "Layer3_BC_count_matrix-1.tsv",
        "expected_size": None,
    },
    {
        "url": f"{SR_BASE}/Layer4_BC_count_matrix-1.tsv",
        "name": "Layer4_BC_count_matrix-1.tsv",
        "expected_size": None,
    },
]


def download_file(url, dest, expected_size=None, max_retries=3):
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

    for f in FILES:
        fpath = BASE_DIR / f["name"]
        if not fpath.exists():
            print(f"  MISSING: {f['name']}")
            ok = False
        else:
            size = fpath.stat().st_size
            print(f"  OK: {f['name']} ({size:,} bytes)")
            if size < 1000:
                print(f"  WARNING: File suspiciously small")
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
        if not download_file(f["url"], BASE_DIR / f["name"], f.get("expected_size")):
            success = False

    verify()
    if not success:
        print("Some downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
