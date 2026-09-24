#!/usr/bin/env python
"""Download 3D IMC breast cancer dataset (Kuett et al. 2022, Nature Cancer).

Source: Zenodo record 4752030 (open access, CC-BY)
  DOI: https://doi.org/10.5281/zenodo.4752030
Target: data/raw/imc_breast_cancer_kuett/

Contents: Four 3D IMC models of breast cancer tissue:
  - MainHer2BreastCancerModel: 152 serial sections, HER2+ invasive ductal carcinoma
  - SecondHer2BreastCancerModel: 92 serial sections, HER2+ invasive ductal carcinoma
  - LVIBloodBreastCancerModel: lymphovascular invasion (blood vessel)
  - LVILymphBreastCancerModel: lymphovascular invasion (lymph vessel), 16 sections

Each ZIP contains:
  - *_mean_intensities.csv: single-cell mean marker intensities (28 protein markers)
  - *_labels_area.csv: cell labels and areas
  - *_panel.csv: antibody panel metadata
  - final_3D_stack_order.csv: section ordering
  - *_segmentation_*.tif: 3D segmentation masks (cell labels per voxel)
  - compensationMatrix.csv: signal compensation
  - Aligned TIFF images (nested ZIPs)

Section thickness: 2 µm, IMC lateral resolution: ~1 µm/pixel
28 protein markers including panCK, HER2, CD markers, vimentin, etc.

Paper: Kuett L et al. "Three-dimensional imaging mass cytometry for highly
multiplexed molecular and cellular mapping of tissues and the tumor
microenvironment." Nature Cancer 3:122-133, 2022.
"""
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "imc_breast_cancer_kuett"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ZENODO_API_BASE = "https://zenodo.org/api/records/4752030/files"

FILES = [
    {
        "url": f"{ZENODO_API_BASE}/MainHer2BreastCancerModel.zip/content",
        "name": "MainHer2BreastCancerModel.zip",
        "md5": "886419ffad66692d02da49570a37e4d3",
        "size": 3040543575,
    },
    {
        "url": f"{ZENODO_API_BASE}/SecondHer2BreastCancerModel.zip/content",
        "name": "SecondHer2BreastCancerModel.zip",
        "md5": "5f8663789711e0a4a6662624f6d8466f",
        "size": 2399463574,
    },
    {
        "url": f"{ZENODO_API_BASE}/LVIBloodBreastCancerModel.zip/content",
        "name": "LVIBloodBreastCancerModel.zip",
        "md5": "f0be9133bec99c34b0e7cae4c858f3c2",
        "size": 629712861,
    },
    {
        "url": f"{ZENODO_API_BASE}/LVILymphBreastCancerModel.zip/content",
        "name": "LVILymphBreastCancerModel.zip",
        "md5": "f2d8217c24f2e523deb6e086b58f7b18",
        "size": 588377573,
    },
]


def download_file(url, dest, expected_size=None, md5=None, max_retries=3):
    """Download with resume support, retry with backoff, and progress bar."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_size and dest.stat().st_size == expected_size:
            # Verify MD5 if available
            if md5:
                h = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                if h.hexdigest() == md5:
                    print(f"  Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes, MD5 OK)")
                    return True
                else:
                    print(f"  MD5 mismatch, re-downloading: {dest.name}")
                    dest.unlink()
            else:
                print(f"  Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
                return True
        elif expected_size and dest.stat().st_size > expected_size:
            print(f"  File larger than expected, re-downloading: {dest.name}")
            dest.unlink()
        elif not expected_size and dest.stat().st_size > 0:
            if md5:
                h = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                if h.hexdigest() == md5:
                    print(f"  Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes, MD5 OK)")
                    return True
                else:
                    print(f"  MD5 mismatch, re-downloading: {dest.name}")
                    dest.unlink()
            else:
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
                print(f"  Verifying MD5 for {dest.name}...")
                h = hashlib.md5()
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(65536), b""):
                        h.update(chunk)
                if h.hexdigest() != md5:
                    print(f"  MD5 mismatch for {dest.name}: expected {md5}, got {h.hexdigest()}")
                    if attempt < max_retries:
                        dest.unlink()
                        continue
                    return False
                print(f"  MD5 OK: {dest.name}")

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

    for entry in FILES:
        dest = BASE_DIR / entry["name"]
        if not dest.exists():
            print(f"  MISSING: {dest.name}")
            ok = False
        elif entry.get("size") and dest.stat().st_size != entry["size"]:
            print(f"  SIZE MISMATCH: {dest.name} (expected {entry['size']:,}, got {dest.stat().st_size:,})")
            ok = False
        elif dest.stat().st_size < 1000:
            print(f"  WARNING: {dest.name} is suspiciously small ({dest.stat().st_size:,} bytes)")
            ok = False
        else:
            print(f"  OK: {dest.name} ({dest.stat().st_size:,} bytes)")

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print("=" * 60)
    print(f"Downloading: {DATASET_NAME}")
    print(f"Source: Zenodo 4752030 (Kuett et al. 2022, Nature Cancer)")
    print(f"Destination: {BASE_DIR}")
    print(f"Estimated size: ~6.7 GB (4 ZIP archives)")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for entry in FILES:
        dest = BASE_DIR / entry["name"]
        print(f"\nDownloading: {entry['name']} ({entry.get('size', 0) / 1e9:.1f} GB)")
        if download_file(entry["url"], dest, expected_size=entry.get("size"), md5=entry.get("md5")):
            success_count += 1
        else:
            print(f"  FAILED: {entry['name']}")

    print(f"\nDownloaded {success_count}/{len(FILES)} files")
    verify()

    if success_count < len(FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
