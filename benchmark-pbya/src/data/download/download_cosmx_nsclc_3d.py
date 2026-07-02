#!/usr/bin/env python
"""Download CosMx NSCLC 3D dataset (Pentimalli et al., rajewsky-lab).

Source: Zenodo record 15240431 (open access)
Target: data/raw/cosmx_nsclc_3d/
Contents: ~340K cells, 960 genes, 6 serial sections from NSCLC tumor

Paper: Pentimalli TM et al. "Combining spatial transcriptomics and ECM imaging
in 3D for mapping cellular interactions in the tumor microenvironment."
Cell Systems 16(5):101261, May 2025.

GitHub: https://github.com/rajewsky-lab/3D_lung
Interactive 3D browser: https://lung-3d-browser.mdc-berlin.de
"""
import sys
import time
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "cosmx_nsclc_3d"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ZENODO_BASE = "https://zenodo.org/records/15240431/files"

FILES = [
    {
        "url": f"{ZENODO_BASE}/cosmx_flat_files.zip?download=1",
        "name": "cosmx_flat_files.zip",
        "md5": "37a2070e8f7d0398c243e2f1ccad6683",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/preprocessed_objects.zip?download=1",
        "name": "preprocessed_objects.zip",
        "md5": "36bf5a2283af96b147c67d382bac177c",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/SHG.zip?download=1",
        "name": "SHG.zip",
        "md5": "92335b19c8acf07468de7aa650cf9d14",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/stimwrap_files.zip?download=1",
        "name": "stimwrap_files.zip",
        "md5": "6b6d4218c9b47a69af2f25aeb93a86af",
        "size": None,
    },
    {
        "url": f"{ZENODO_BASE}/analysis_notebooks.zip?download=1",
        "name": "analysis_notebooks.zip",
        "md5": "50bb499e3d3f716a112c17f7e80ef37a",
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
                if h.hexdigest() != md5:
                    print(f"  MD5 mismatch for {dest.name}: expected {md5}, got {h.hexdigest()}")
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
    """Verify downloaded files."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    for entry in FILES:
        dest = BASE_DIR / entry["name"]
        if not dest.exists():
            print(f"  MISSING: {dest.name}")
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
    print(f"Source: Zenodo 15240431 (open access)")
    print(f"Destination: {BASE_DIR}")
    print(f"Estimated size: ~9 GB")
    print("=" * 60)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for entry in FILES:
        dest = BASE_DIR / entry["name"]
        print(f"\nDownloading: {entry['name']}")
        if download_file(entry["url"], dest, md5=entry.get("md5")):
            success_count += 1
        else:
            print(f"  FAILED: {entry['name']}")

    print(f"\nDownloaded {success_count}/{len(FILES)} files")
    verify()

    if success_count < len(FILES):
        sys.exit(1)


if __name__ == "__main__":
    main()
