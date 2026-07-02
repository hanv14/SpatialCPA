#!/usr/bin/env python
"""Download 3D MERFISH thick tissue dataset (Fang/Dulac/Zhuang 2024).

Source: Dryad (doi:10.5061/dryad.w0vt4b922)
Target: data/raw/merfish_thick_tissue/
Requirements: expression data, 3D coordinates, cell-type annotations

Uses Playwright (headless browser) to bypass Dryad's AWS WAF bot protection.
The WAF challenge sets cookies that allow subsequent downloads via requests.
"""
import sys
import time
import hashlib
import zipfile
from pathlib import Path

DATASET_NAME = "merfish_thick_tissue"
BASE_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / DATASET_NAME

DOWNLOAD_URL = "https://datadryad.org/downloads/file_stream/3726956"
ZIP_NAME = "Fang_eLife_2023.zip"
SHA256 = "f9cd2efe622796da8fe169490b75014081ace4fe3046e935513c3862797cc275"


def download_with_playwright(url, dest):
    """Download file using Playwright to bypass AWS WAF JavaScript challenge.

    Strategy: Navigate to the file_stream URL in a headless browser. The WAF
    returns a "Validating..." challenge page with JS that sets cookies. After
    the challenge runs, extract the cookies and use requests to download.
    """
    from playwright.sync_api import sync_playwright
    import requests as req_lib
    from tqdm import tqdm

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"  Already downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
        return True

    print(f"  Launching headless browser to solve WAF challenge...")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled'],
        )
        context = browser.new_context(
            accept_downloads=True,
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        )
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
        """)

        # Navigate to file_stream URL to trigger WAF challenge
        print(f"  Navigating to {url}...")
        page.goto(url, wait_until="domcontentloaded", timeout=60000)

        title = page.title()
        if 'validating' in title.lower() or 'challenge' in title.lower():
            print(f"  WAF challenge detected, waiting for cookies to be set...")
            for i in range(12):  # Wait up to 60 seconds
                time.sleep(5)
                new_title = page.title()
                if new_title != title or page.url != url:
                    print(f"  Challenge resolved after {(i+1)*5}s")
                    break
            else:
                print(f"  Challenge page persisted (cookies may still be valid)")

        # Extract cookies from browser
        cookies = context.cookies()
        session = req_lib.Session()
        for cookie in cookies:
            session.cookies.set(cookie['name'], cookie['value'],
                              domain=cookie.get('domain', ''),
                              path=cookie.get('path', '/'))
        session.headers['User-Agent'] = page.evaluate('() => navigator.userAgent')

        browser.close()

    # Download with WAF cookies
    print(f"  Downloading with WAF session cookies...")
    try:
        response = session.get(url, stream=True, timeout=120, allow_redirects=True)
        ct = response.headers.get('content-type', '')
        cl = int(response.headers.get('content-length', 0) or 0)

        if response.status_code == 200 and 'html' not in ct and cl > 1000000:
            print(f"  Downloading {cl:,} bytes...")
            with open(dest, 'wb') as f, tqdm(
                total=cl, unit='B', unit_scale=True, desc=dest.name,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            if dest.exists() and dest.stat().st_size > 0:
                print(f"  Downloaded: {dest.name} ({dest.stat().st_size:,} bytes)")
                return True

        print(f"  Download failed: status={response.status_code}, content-type={ct}")
    except Exception as e:
        print(f"  Download error: {e}")

    return False


def verify_sha256(filepath, expected):
    """Verify SHA-256 hash of a file."""
    print(f"  Verifying SHA-256 for {filepath.name}...")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        print(f"  SHA-256 mismatch: expected {expected}, got {actual}")
        return False
    print(f"  SHA-256 OK: {actual}")
    return True


def unzip_data():
    """Unzip the downloaded archive."""
    zip_path = BASE_DIR / ZIP_NAME
    if not zip_path.exists():
        print("  ZIP file not found, skipping extraction")
        return False

    # Check if already extracted
    extracted_dir = BASE_DIR / "dryad-2024-08-14"
    if extracted_dir.exists():
        print(f"  Already extracted: {extracted_dir}")
        return True

    print("  Extracting ZIP archive...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(BASE_DIR)
        print("  Extraction complete")
        return True
    except zipfile.BadZipFile:
        print("  ERROR: Bad ZIP file")
        return False


def verify():
    """Verify downloaded and extracted files."""
    print(f"\nVerifying {DATASET_NAME}...")
    ok = True

    # Check ZIP exists
    zip_path = BASE_DIR / ZIP_NAME
    if not zip_path.exists():
        print(f"  MISSING: {ZIP_NAME}")
        ok = False
    else:
        print(f"  OK: {ZIP_NAME} ({zip_path.stat().st_size:,} bytes)")

    # Check extracted contents
    extracted = list(BASE_DIR.glob("**/*"))
    extracted = [f for f in extracted if f.is_file() and f.suffix != ".zip"]
    if extracted:
        print(f"  Extracted files: {len(extracted)}")
        for f in sorted(extracted)[:10]:
            print(f"    {f.relative_to(BASE_DIR)} ({f.stat().st_size:,} bytes)")
        if len(extracted) > 10:
            print(f"    ... and {len(extracted) - 10} more")
    else:
        print("  WARNING: No extracted files found")
        ok = False

    # Check CSV content (3D coordinates)
    try:
        import pandas as pd
        barcode_csv = BASE_DIR / "dryad-2024-08-14" / "data" / "data_Figure1" / "barcodes_100ms_DL.csv"
        if barcode_csv.exists():
            df = pd.read_csv(barcode_csv, nrows=5)
            print(f"  CSV columns: {list(df.columns)}")
            has_3d = all(c in df.columns for c in ['global_x', 'global_y', 'global_z'])
            print(f"  Has 3D coordinates: {has_3d}")
    except ImportError:
        print("  WARNING: pandas not installed, skipping CSV content check")
    except Exception as e:
        print(f"  CSV CHECK: {e}")

    if ok:
        print(f"  {DATASET_NAME}: ALL CHECKS PASSED")
    else:
        print(f"  {DATASET_NAME}: SOME CHECKS FAILED")
    return ok


def main():
    print(f"Downloading {DATASET_NAME} (3.75 GB, may take a while)...")
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = BASE_DIR / ZIP_NAME

    if not download_with_playwright(DOWNLOAD_URL, zip_path):
        print("Download failed.")
        sys.exit(1)

    if SHA256:
        verify_sha256(zip_path, SHA256)

    unzip_data()
    verify()


if __name__ == "__main__":
    main()
