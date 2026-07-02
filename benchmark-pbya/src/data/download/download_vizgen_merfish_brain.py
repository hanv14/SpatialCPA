#!/usr/bin/env python
"""Download Vizgen MERFISH mouse brain dataset.

Source: info.vizgen.com (requires free registration)
Target: data/raw/vizgen_merfish_brain/
Contents: 734K cells, 483 genes, 9 sections — original Vizgen MERFISH receptor map

NOTE: This dataset requires free registration at Vizgen's website. It is NOT
available from any public mirror. The Allen Brain Cell Atlas MERFISH data is a
different dataset (allen_merfish_brain) with different cells/genes/sections.
"""
import os
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "vizgen_merfish_brain"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ENV_VAR = "VIZGEN_TOKEN"
REGISTER_URL = "https://info.vizgen.com/mouse-brain-data"

# Known files in the Vizgen MERFISH mouse brain release (requires registration)
VIZGEN_EXPECTED_FILES = [
    "cell_by_gene.csv",
    "cell_metadata.csv",
    "detected_transcripts.csv",
    "images/mosaic_DAPI_z0.tif",
    "images/mosaic_PolyT_z0.tif",
]


def check_credentials():
    """Check for Vizgen download credentials.

    Returns the download base URL (token) or None if not set.
    """
    token = os.environ.get(ENV_VAR)
    if not token:
        return None
    return token


def download_file(url, dest, expected_size=None, md5=None, max_retries=3, headers=None):
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
    extra_headers = headers or {}
    for attempt in range(1, max_retries + 1):
        try:
            req_headers = dict(extra_headers)
            mode = "wb"
            initial_size = 0
            if dest.exists():
                initial_size = dest.stat().st_size
                req_headers["Range"] = f"bytes={initial_size}-"
                mode = "ab"
            response = requests.get(url, headers=req_headers, stream=True, timeout=120)
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
                total=total_size or None, initial=initial_size,
                unit="B", unit_scale=True, desc=dest.name,
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


def discover_vizgen_files(base_url):
    """Build file list from the Vizgen download base URL."""
    base = base_url.rstrip("/")
    files = []
    for rel_path in VIZGEN_EXPECTED_FILES:
        files.append({
            "url": f"{base}/{rel_path}",
            "name": rel_path.replace("/", "_") if "/" in rel_path else rel_path,
            "dest_path": rel_path,
            "size": None,
        })
    return files


def verify():
    """Verify downloaded files."""
    print(f"\nVerifying {DATASET_NAME} dataset...")
    if not BASE_DIR.exists():
        print(f"  FAIL: Directory not found: {BASE_DIR}")
        return False

    data_files = [f for f in BASE_DIR.rglob("*") if f.is_file()]
    if not data_files:
        print(f"  FAIL: No files found in {BASE_DIR}")
        return False

    has_gene_data = (BASE_DIR / "cell_by_gene.csv").exists()
    has_metadata = (BASE_DIR / "cell_metadata.csv").exists()

    total_size = sum(f.stat().st_size for f in data_files)
    print(f"  Found {len(data_files)} files, total size: {total_size / 1e9:.2f} GB")

    if has_gene_data:
        print("  OK: Found cell_by_gene.csv")
    else:
        print("  WARNING: Missing cell_by_gene.csv")

    if has_metadata:
        print("  OK: Found cell_metadata.csv")
    else:
        print("  WARNING: Missing cell_metadata.csv")

    print("  Verification complete.")
    return True


def main():
    """Download the Vizgen MERFISH mouse brain dataset."""
    print(f"=== {DATASET_NAME} ===")
    print(f"Target: {BASE_DIR}")
    print()

    token = check_credentials()
    if token is None:
        print("This dataset requires free registration at Vizgen's website.")
        print("No public mirror is available.")
        print()
        print("Expected data: 734K cells, 483 genes, 9 sections")
        print("  - Original Vizgen MERFISH mouse brain receptor map")
        print("  - Distinct from Allen Brain Cell Atlas MERFISH (allen_merfish_brain)")
        print()
        print("To download:")
        print(f"  1. Register for free at: {REGISTER_URL}")
        print(f"  2. After registration, you will receive a download URL")
        print(f"  3. Set: export {ENV_VAR}=<your-download-base-url>")
        print(f"  4. Re-run this script")
        print()
        print("Skipping download (no credentials).")
        sys.exit(0)

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Discover files from the download URL
    files = discover_vizgen_files(token)
    print(f"Downloading {len(files)} files from Vizgen...\n")

    success_count = 0
    fail_count = 0

    for file_info in files:
        dest = BASE_DIR / file_info["dest_path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = download_file(
            url=file_info["url"],
            dest=dest,
            expected_size=file_info.get("size"),
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    print(f"\nDownloaded {success_count}/{len(files)} files.")
    if fail_count:
        print(f"  {fail_count} files failed.")

    verify()


if __name__ == "__main__":
    main()
