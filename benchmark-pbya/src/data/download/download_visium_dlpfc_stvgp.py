#!/usr/bin/env python
"""Download DLPFC Visium data for stVGP paper.

Source: spatialLIBD (Maynard et al., 2021, Nature Neuroscience)
  - h5 expression matrices from AWS S3 (spatial-dlpfc bucket)
  - Spatial coordinates + scalefactors from GitHub (LieberInstitute/HumanPilot)
  - Layer annotations from GitHub barcode_level_layer_map.tsv
Target: data/raw/visium_dlpfc_stvgp/
Description: 4 Visium sections (151673-151676) from human DLPFC with manual layer annotations.
  These are 4 sections from the same donor (Br8100), constituting two pairs of
  spatially adjacent serial sections (151673/151674 at position 0, 151675/151676 ~300µm away).
"""
import os
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "visium_dlpfc_stvgp"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

S3_BASE = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com"
GH_BASE = "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/10X"

SAMPLES = ["151673", "151674", "151675", "151676"]

FILES = []

# Expression matrices from S3
for sample in SAMPLES:
    FILES.append({
        "url": f"{S3_BASE}/h5/{sample}_filtered_feature_bc_matrix.h5",
        "name": f"{sample}_filtered_feature_bc_matrix.h5",
        "size": None,
    })

# Spatial data from GitHub for each sample
for sample in SAMPLES:
    FILES.append({
        "url": f"{GH_BASE}/{sample}/tissue_positions_list.txt",
        "name": f"{sample}_tissue_positions_list.csv",
        "size": None,
    })
    FILES.append({
        "url": f"{GH_BASE}/{sample}/scalefactors_json.json",
        "name": f"{sample}_scalefactors_json.json",
        "size": None,
    })
    FILES.append({
        "url": f"{S3_BASE}/images/{sample}_tissue_hires_image.png",
        "name": f"{sample}_tissue_hires_image.png",
        "size": None,
    })
    FILES.append({
        "url": f"{S3_BASE}/images/{sample}_tissue_lowres_image.png",
        "name": f"{sample}_tissue_lowres_image.png",
        "size": None,
    })

# Layer annotations (barcode-level) from GitHub
FILES.append({
    "url": f"{GH_BASE}/barcode_level_layer_map.tsv",
    "name": "barcode_level_layer_map.tsv",
    "size": None,
})


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

    for sample in SAMPLES:
        h5_path = BASE_DIR / f"{sample}_filtered_feature_bc_matrix.h5"
        pos_path = BASE_DIR / f"{sample}_tissue_positions_list.csv"
        sf_path = BASE_DIR / f"{sample}_scalefactors_json.json"

        for p in [h5_path, pos_path, sf_path]:
            if not p.exists():
                print(f"  MISSING: {p.name}")
                ok = False
            else:
                print(f"  OK: {p.name} ({p.stat().st_size:,} bytes)")

    layer_path = BASE_DIR / "barcode_level_layer_map.tsv"
    if not layer_path.exists():
        print(f"  MISSING: {layer_path.name}")
        ok = False
    else:
        print(f"  OK: {layer_path.name} ({layer_path.stat().st_size:,} bytes)")

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

    verify()
    if not success:
        print("Some downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
