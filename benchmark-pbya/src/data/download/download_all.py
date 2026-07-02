#!/usr/bin/env python
"""Download all spatial transcriptomics datasets.

Runs individual download scripts in order (Tier 1 → 3).
Only downloads datasets that are ACTIVE in the processing pipeline.
Usage:
    python src/data/download/download_all.py             # all tiers
    python src/data/download/download_all.py --tier 1     # tier 1 only
    python src/data/download/download_all.py --tier 1 2 3 # tiers 1-3 only
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Only ACTIVE datasets (those with processing scripts and DATASET_REGISTRY entries)
TIERS = {
    1: [  # Small / fast downloads
        "download_starmap_visual_cortex.py",              # 29 MB
        "download_imc_breast_cancer.py",                  # 89 MB
        "download_easi_fish_hypothalamus.py",             # 9 MB
        "download_exseq_breast_cancer.py",                # ~100 MB
        "download_exseq_visual_cortex.py",                # ~100 MB
        "download_st_mouse_brain_ortiz.py",               # 157 MB
        "download_deep_starmap.py",                       # 426 MB
    ],
    2: [  # Medium downloads
        "download_merfish_hypothalamus.py",               # 1.05 GB
        "download_cosmx_nsclc_3d.py",                     # ~1 GB
        "download_merfish_thick_tissue.py",               # 3.75 GB
        "download_visium_mouse_brain_cell2location.py",   # ~2 GB
        "download_openst_lymph_node.py",                  # ~5 GB
    ],
    3: [  # Large downloads (Allen Brain)
        "download_allen_merfish_brain.py",                # 8.16 GB
        "download_allen_zhuang_merfish.py",               # ~5 GB
    ],
}

ALL_TIERS = sorted(TIERS.keys())


def main():
    parser = argparse.ArgumentParser(description="Download all spatial transcriptomics datasets")
    parser.add_argument("--tier", type=int, nargs="+", choices=ALL_TIERS,
                        help="Which tiers to download (default: all)")
    args = parser.parse_args()

    tiers_to_run = args.tier if args.tier else ALL_TIERS

    results = {}
    for tier in sorted(tiers_to_run):
        print(f"\n{'='*60}")
        print(f"  TIER {tier} ({len(TIERS[tier])} scripts)")
        print(f"{'='*60}")

        for script_name in TIERS[tier]:
            script_path = SCRIPT_DIR / script_name
            print(f"\n--- Running {script_name} ---")

            if not script_path.exists():
                print(f"  ERROR: Script not found: {script_path}")
                results[script_name] = False
                continue

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    check=False,
                )
                results[script_name] = result.returncode == 0
                if result.returncode != 0:
                    print(f"  WARNING: {script_name} exited with code {result.returncode}")
            except Exception as e:
                print(f"  ERROR running {script_name}: {e}")
                results[script_name] = False

    # Summary
    print(f"\n{'='*60}")
    print("  DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {script}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} download(s) failed. Re-run individual scripts to retry.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} downloads completed successfully!")


if __name__ == "__main__":
    main()
