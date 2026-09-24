#!/usr/bin/env python
"""Download Open-ST lymph node dataset.

Source: GEO GSE251926
Target: data/raw/openst_lymph_node/
Requirements: spatial transcriptomics data from Open-ST lymph node samples
"""
import os
import sys
import time
import hashlib
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "openst_lymph_node"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE251nnn/GSE251926/suppl"

FILES = [
    {
        "url": f"{GEO_BASE}/GSE251926_RAW.tar",
        "name": "GSE251926_RAW.tar",
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


def extract_tar(tar_path):
    """Extract tar archive and list contents."""
    tar_path = Path(tar_path)
    if not tar_path.exists():
        print(f"  Cannot extract: {tar_path.name} not found")
        return False

    print(f"\nExtracting {tar_path.name}...")
    try:
        with tarfile.open(tar_path, "r") as tf:
            members = tf.getnames()
            print(f"  Archive contains {len(members)} files")
            tf.extractall(path=tar_path.parent)
        print(f"  Extracted to {tar_path.parent}")
        return True
    except Exception as e:
        print(f"  Extraction failed: {e}")
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

    # Content checks: list tar contents
    try:
        tar_path = BASE_DIR / "GSE251926_RAW.tar"
        if tar_path.exists():
            with tarfile.open(tar_path, "r") as tf:
                members = tf.getnames()
                print(f"  TAR contents ({len(members)} files):")
                for m in members[:10]:
                    print(f"    {m}")
                if len(members) > 10:
                    print(f"    ... and {len(members) - 10} more")
    except Exception as e:
        print(f"  CONTENT CHECK FAILED: {e}")
        ok = False

    # Check for extracted files
    extracted = [p for p in BASE_DIR.iterdir() if p.name != "GSE251926_RAW.tar"]
    if extracted:
        print(f"  Extracted files: {len(extracted)}")
        for p in sorted(extracted)[:10]:
            print(f"    {p.name} ({p.stat().st_size:,} bytes)")

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
        # Extract tar archive
        tar_path = BASE_DIR / "GSE251926_RAW.tar"
        extract_tar(tar_path)

        verify()
    else:
        print("Some downloads failed. Skipping verification.")
        sys.exit(1)


if __name__ == "__main__":
    main()
