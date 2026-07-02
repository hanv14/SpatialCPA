#!/usr/bin/env python
"""Download HuBMAP IMC spleen dataset.

Source: https://portal.hubmapconsortium.org/
Target: data/raw/hubmap_imc_spleen/
Contents: 52 sections, 39 protein channels, ~5 GB
Note: Tries public API access first. Set HUBMAP_TOKEN for full access.

Known IMC spleen dataset UUIDs on HuBMAP portal:
  - d3130f4a89946cc6b300b115a3120b7a (3D IMC, spleen, 18yo male)
"""
import os
import sys
import time
import json
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_NAME = "hubmap_imc_spleen"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

ENV_VAR = "HUBMAP_TOKEN"
REGISTER_URL = "https://portal.hubmapconsortium.org/"

HUBMAP_API = "https://portal.hubmapconsortium.org/api"
HUBMAP_SEARCH_API = "https://search.api.hubmapconsortium.org/v3"
HUBMAP_ENTITY_API = "https://entity.api.hubmapconsortium.org"
HUBMAP_ASSETS_URL = "https://assets.hubmapconsortium.org"

# Known IMC spleen dataset UUIDs (publicly listed on HuBMAP portal)
KNOWN_DATASET_UUIDS = [
    "d3130f4a89946cc6b300b115a3120b7a",
]

# Search for IMC spleen datasets
SEARCH_KEYWORDS = ["IMC", "spleen", "imaging mass cytometry"]


def check_credentials():
    """Check for HuBMAP credentials.

    Returns the Globus token or None (will try public access).
    """
    token = os.environ.get(ENV_VAR)
    if not token:
        print(f"INFO: {ENV_VAR} not set. Trying public API access...")
        print(f"  Published HuBMAP datasets should be accessible without login.")
        print(f"  If downloads fail, you may need to authenticate via Globus.")
        print(f"  For full access, register at: {REGISTER_URL}")
        print(f"  Set: export {ENV_VAR}=<your-globus-token>")
        print()
        return None  # Will try public access
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


def search_datasets(token=None):
    """Search HuBMAP for IMC spleen datasets using the search API."""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Elasticsearch query for IMC spleen datasets
    query = {
        "query": {
            "bool": {
                "must": [
                    {"match": {"data_types": "IMC"}},
                    {"match": {"origin_samples.organ": "spleen"}},
                ]
            }
        },
        "size": 100,
        "_source": [
            "uuid", "hubmap_id", "data_types", "status",
            "origin_samples.organ", "files",
        ],
    }

    try:
        response = requests.post(
            f"{HUBMAP_SEARCH_API}/search",
            headers=headers, json=query, timeout=60,
        )
        response.raise_for_status()
        results = response.json()
        hits = results.get("hits", {}).get("hits", [])
        return [h["_source"] for h in hits]
    except requests.exceptions.RequestException as e:
        print(f"  Error searching datasets: {e}")
        return None


def get_dataset_files(dataset_uuid, token=None):
    """Get file listing for a specific dataset.

    The HuBMAP Entity API only includes a ``files`` field for processed
    (derived) datasets.  Raw / primary datasets do **not** have this field
    -- their data is accessible exclusively via Globus transfer.

    When the primary dataset has no files we also inspect its *descendant*
    datasets (e.g. image-pyramid derivatives) which often do expose a
    ``files`` list that can be downloaded over HTTPS from the HuBMAP assets
    server.
    """
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        response = requests.get(
            f"{HUBMAP_ENTITY_API}/entities/{dataset_uuid}",
            headers=headers, timeout=60,
        )
        response.raise_for_status()
        entity = response.json()

        files = entity.get("files", [])
        if files:
            return dataset_uuid, files

        # No files on primary dataset -- try descendant (derived) datasets
        print(f"  Primary dataset has no 'files' field (raw data uses Globus).")
        print(f"  Checking descendant datasets for downloadable derivatives...")
        descendants = entity.get("direct_descendants") or []
        if not descendants:
            # Fetch descendants from the portal JSON endpoint
            try:
                portal_resp = requests.get(
                    f"https://portal.hubmapconsortium.org/browse/dataset/{dataset_uuid}.json",
                    timeout=60,
                )
                if portal_resp.status_code == 200:
                    portal_data = portal_resp.json()
                    descendants = portal_data.get("descendants", portal_data.get("immediate_descendants", []))
            except Exception:
                pass

        for desc in descendants:
            desc_uuid = desc.get("uuid", "")
            if not desc_uuid:
                continue
            try:
                desc_resp = requests.get(
                    f"{HUBMAP_ENTITY_API}/entities/{desc_uuid}",
                    headers=headers, timeout=60,
                )
                if desc_resp.status_code == 200:
                    desc_entity = desc_resp.json()
                    desc_files = desc_entity.get("files", [])
                    if desc_files:
                        desc_type = desc_entity.get("dataset_type", "unknown")
                        desc_id = desc_entity.get("hubmap_id", desc_uuid[:12])
                        print(f"  Found {len(desc_files)} files in descendant {desc_id} ({desc_type})")
                        return desc_uuid, desc_files
            except Exception:
                pass

        return dataset_uuid, []
    except requests.exceptions.RequestException as e:
        print(f"  Error getting files for {dataset_uuid}: {e}")
        return dataset_uuid, None


def build_file_url(dataset_uuid, rel_path):
    """Build the download URL for a file in a HuBMAP dataset."""
    return f"{HUBMAP_ASSETS_URL}/{dataset_uuid}/{rel_path}"


def download_dataset_files(dataset_uuid, dataset_id, files, token=None):
    """Download all files from a HuBMAP dataset."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dataset_dir = BASE_DIR / dataset_id
    dataset_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    fail_count = 0

    for file_info in files:
        rel_path = file_info.get("rel_path", "")
        file_size = file_info.get("size")
        if not rel_path:
            continue

        url = build_file_url(dataset_uuid, rel_path)
        dest = dataset_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)

        ok = download_file(
            url=url,
            dest=dest,
            expected_size=file_size,
            headers=headers if token else None,
        )
        if ok:
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def verify():
    """Verify downloaded files."""
    print(f"\nVerifying {DATASET_NAME} dataset...")
    if not BASE_DIR.exists():
        print(f"  FAIL: Directory not found: {BASE_DIR}")
        return False

    found_files = list(BASE_DIR.rglob("*"))
    data_files = [f for f in found_files if f.is_file()]

    if not data_files:
        print(f"  FAIL: No files found in {BASE_DIR}")
        return False

    total_size = sum(f.stat().st_size for f in data_files)
    print(f"  Found {len(data_files)} files, total size: {total_size / 1e9:.2f} GB")

    # Check for expected IMC data types
    has_images = any(
        f.suffix in (".tif", ".tiff", ".mcd", ".ome.tiff")
        for f in data_files
    )
    has_masks = any(
        "mask" in f.name.lower() or "segmentation" in f.name.lower()
        for f in data_files
    )
    has_quantification = any(
        f.suffix in (".csv", ".tsv", ".h5ad", ".txt")
        for f in data_files
    )

    if has_images:
        print("  OK: Found image files")
    else:
        print("  WARNING: No image files (.tif/.mcd) found")

    if has_masks:
        print("  OK: Found segmentation/mask files")
    else:
        print("  INFO: No segmentation mask files found")

    if has_quantification:
        print("  OK: Found quantification/data files")
    else:
        print("  INFO: No quantification files found")

    # Count dataset subdirectories
    subdirs = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    if subdirs:
        print(f"  Found {len(subdirs)} dataset subdirectories")

    for f in data_files:
        if f.stat().st_size == 0:
            print(f"  WARNING: Empty file: {f.name}")

    print("  Verification complete.")
    return True


def main():
    """Download the HuBMAP IMC spleen dataset."""
    print(f"=== Downloading {DATASET_NAME} ===")
    print(f"Target: {BASE_DIR}")
    print(f"Source: HuBMAP Consortium Portal")
    print(f"Description: 52 sections, 39 protein channels, ~5 GB")
    print()

    # Check credentials (not required -- will try public access)
    token = check_credentials()

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # Search for IMC spleen datasets
    print("Searching HuBMAP for IMC spleen datasets...")
    datasets = search_datasets(token)

    if datasets is None or len(datasets) == 0:
        print("No datasets found via search API. Trying known dataset UUIDs...")
        # Fall back to known dataset UUIDs
        datasets = []
        for uuid in KNOWN_DATASET_UUIDS:
            try:
                headers = {}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                response = requests.get(
                    f"{HUBMAP_ENTITY_API}/entities/{uuid}",
                    headers=headers, timeout=60,
                )
                if response.status_code == 200:
                    entity = response.json()
                    datasets.append(entity)
                    print(f"  Found known dataset: {entity.get('hubmap_id', uuid)}")
                else:
                    print(f"  Could not access dataset {uuid} (HTTP {response.status_code})")
            except requests.exceptions.RequestException as e:
                print(f"  Error accessing dataset {uuid}: {e}")

    if not datasets:
        print()
        print("ERROR: Could not find or access any IMC spleen datasets.")
        print()
        print("HuBMAP published datasets should be publicly accessible, but")
        print("some datasets may require Globus authentication for file download.")
        print()
        if token is None:
            print("To authenticate:")
            print(f"  1. Register at: {REGISTER_URL}")
            print(f"  2. Log in with Globus (institutional or eRACommons credentials)")
            print(f"  3. Set: export {ENV_VAR}=<your-globus-token>")
        else:
            print("Your token may be expired. Try getting a fresh Globus token.")
        sys.exit(1)

    print(f"Found {len(datasets)} IMC spleen datasets.\n")

    total_success = 0
    total_fail = 0

    for dataset in datasets:
        ds_uuid = dataset.get("uuid", "")
        ds_id = dataset.get("hubmap_id", ds_uuid[:12])
        ds_status = dataset.get("status", "unknown")

        print(f"--- Dataset: {ds_id} (status: {ds_status}) ---")

        # Get file listing
        files = dataset.get("files")
        file_uuid = ds_uuid  # UUID whose files we're downloading
        if files:
            file_uuid = ds_uuid
        else:
            file_uuid, files = get_dataset_files(ds_uuid, token)

        if not files:
            print(f"  No downloadable files found for {ds_id}.")
            print(f"  This dataset's raw data requires Globus transfer.")
            print(f"  Install HuBMAP CLT: pip install hubmap-clt")
            print(f"  Then run:  hubmap-clt {ds_uuid}")
            print(f"  Or visit: https://portal.hubmapconsortium.org/browse/dataset/{ds_uuid}")
            print()
            continue

        print(f"  {len(files)} files to download")
        s, f = download_dataset_files(file_uuid, ds_id, files, token)
        total_success += s
        total_fail += f
        print(f"  Downloaded {s} files, {f} failed.\n")

    print(f"\nTotal: {total_success} files downloaded, {total_fail} failed.")

    # Verify
    verify()


if __name__ == "__main__":
    main()
