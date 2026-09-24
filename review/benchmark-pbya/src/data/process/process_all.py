#!/usr/bin/env python
"""Process all spatial transcriptomics datasets to standardized h5ad.

Runs individual processing scripts in order (Tier 1 -> 3).
Only processes ACTIVE datasets in the pipeline.
Usage:
    python src/data/process/process_all.py             # all tiers
    python src/data/process/process_all.py --tier 1     # tier 1 only
    python src/data/process/process_all.py --tier 1 2 3 # tiers 1-3 only
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

TIERS = {
    1: [  # Small / fast datasets
        "process_starmap_visual_cortex.py",
        "process_imc_breast_cancer.py",
        "process_easi_fish_hypothalamus.py",
        "process_exseq_visual_cortex.py",
        "process_exseq_breast_cancer.py",
    ],
    2: [  # Medium datasets
        "process_merfish_hypothalamus.py",
        "process_deep_starmap.py",
        "process_st_mouse_brain_ortiz.py",
        "process_merfish_thick_tissue.py",
        "process_openst_lymph_node.py",
        "process_visium_mouse_brain_cell2location.py",
        "process_cosmx_nsclc_3d.py",
    ],
    3: [  # Large datasets (Allen Brain)
        "process_allen_merfish_brain.py",
        "process_allen_zhuang_merfish.py",
    ],
}

ALL_TIERS = sorted(TIERS.keys())


def main():
    parser = argparse.ArgumentParser(description="Process all spatial transcriptomics datasets")
    parser.add_argument("--tier", type=int, nargs="+", choices=ALL_TIERS,
                        help="Which tiers to process (default: all)")
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
    print("  PROCESSING SUMMARY")
    print(f"{'='*60}")
    for script, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  [{status}] {script}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n  {len(failed)} processing script(s) failed.")
        sys.exit(1)
    else:
        print(f"\n  All {len(results)} datasets processed successfully!")


if __name__ == "__main__":
    main()
