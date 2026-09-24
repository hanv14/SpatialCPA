#!/usr/bin/env python
"""Build the SpatialZ-paper STARmap visual-cortex dataset (7 consecutive sections).

The protocol, and the implementation, now live in :mod:`prepare_dataset`, which
runs the same construction for any registered dataset. This module stays as the
STARmap entry point because it is the one the README documents and the one the
paper defines:

    python -m src.bench3.prepare_starmap            # identical to before
    python -m src.bench3.prepare_dataset --dataset starmap_visual_cortex

Protocol (SpatialZ, Lin et al. 2025 — STARmap visual cortex benchmark)
----------------------------------------------------------------------
1. Raw 3-D STARmap volume (Wang et al. 2018): 89 consecutive z-planes (z = 6..94).
2. Drop the uppermost (z = 6-13) and lowermost (z = 91-94) planes to reduce
   technical noise. 77 planes remain (z = 14..90).
3. Partition into **seven consecutive 2-D sections** — 77 planes divide into
   exactly 11 per section, so the partition is even and needs no fudging:

       section_1: z 14-24    section_5: z 58-68
       section_2: z 25-35    section_6: z 69-79
       section_3: z 36-46    section_7: z 80-90
       section_4: z 47-57

4. Sections 2, 4 and 6 are held out; 1, 3, 5 and 7 are the input. That split
   lives in ``design.py`` — this script only builds the dataset.
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import dataset_path, resolve_raw
from .prepare_dataset import build, extract_xyz, partition_planes, verify  # noqa: F401

STARMAP = "starmap_visual_cortex"


def main():
    ap = argparse.ArgumentParser(
        description="Build the SpatialZ-paper STARmap 7-section dataset")
    ap.add_argument("--raw", default=str(resolve_raw(STARMAP)),
                    help="raw STARmap h5ad (or a v1-processed data.h5ad)")
    ap.add_argument("--output", default=str(dataset_path(STARMAP)),
                    help="output h5ad")
    ap.add_argument("--n-sections", type=int, default=None)
    ap.add_argument("--no-flatten-z", action="store_true",
                    help="keep each cell's original z instead of collapsing each "
                         "section onto its centre plane (not the paper protocol)")
    ap.add_argument("--print-protocol", action="store_true",
                    help="print the resolved protocol as JSON and exit")
    args = ap.parse_args()

    adata = build(dataset=STARMAP, raw_path=args.raw, output_path=args.output,
                  flatten_z=not args.no_flatten_z, n_sections=args.n_sections)
    if args.print_protocol:
        print(json.dumps(adata.uns["paper_protocol"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
