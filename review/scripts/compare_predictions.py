#!/usr/bin/env python3
"""Compare two prediction.h5 files array by array.

    python scripts/compare_predictions.py EXPECTED.h5 ACTUAL.h5

Every dataset in EXPECTED must exist in ACTUAL with the same shape and dtype and
identical values. ``uns/wall_time_seconds`` is the one field allowed to differ.
This is the strongest check available: if the predictions are bitwise identical,
every metric computed from them is too, whatever the scorer's own tolerance.

Exit 0 if identical, 1 otherwise (with the first differing fields listed).
"""

from __future__ import annotations

import sys

import h5py
import numpy as np

IGNORED = {"uns/wall_time_seconds"}


def datasets(group, prefix=""):
    for key in group:
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            yield prefix + key, obj
        else:
            yield from datasets(obj, prefix + key + "/")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    diffs = []
    with h5py.File(argv[0], "r") as exp, h5py.File(argv[1], "r") as act:
        names = [n for n, _ in datasets(exp)]
        for name, d in datasets(exp):
            if name in IGNORED:
                continue
            if name not in act:
                diffs.append(f"{name}: missing")
                continue
            x, y = d[()], act[name][()]
            xa, ya = np.asarray(x), np.asarray(y)
            if xa.shape != ya.shape:
                diffs.append(f"{name}: shape {xa.shape} vs {ya.shape}")
            elif not np.array_equal(xa, ya):
                if xa.dtype.kind in "fiu":
                    diffs.append(f"{name}: max|diff| = "
                                 f"{np.max(np.abs(xa.astype(float) - ya.astype(float))):.3e}")
                else:
                    diffs.append(f"{name}: differs")
    if diffs:
        print("predictions DIFFER:")
        for d in diffs[:20]:
            print("  " + d)
        return 1
    print(f"predictions identical ({len(names) - len(IGNORED & set(names))} fields; "
          f"ignored: {', '.join(sorted(IGNORED))})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
