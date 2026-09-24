#!/usr/bin/env python3
"""Compare a freshly produced metrics.json against a committed one.

    python scripts/compare_metrics.py EXPECTED.json ACTUAL.json [--atol X] [--umap-atol Y]

Every scalar ``paper_*`` and ``gen_*`` value in EXPECTED must be present in ACTUAL
and agree within tolerance. Two tolerances, because the metrics are not equally
reproducible:

* ``--atol`` for everything deterministic given the prediction (Moran's I,
  Geary's C, marker fields/SSIM/depth, localization OT, gene mean/var/detection,
  the PCA mixing score, the gen_* block).
* ``--umap-atol`` for ``paper_umap_mixing`` / ``paper_umap_centroid_dist``. UMAP is
  stochastic across umap-learn / numba / BLAS versions even at a fixed seed, so
  those two columns get a separate, looser bound. ``paper_embedding_mixing_pca``
  measures the same thing deterministically and is held to ``--atol``.

Exit 0 when everything is within tolerance, 1 otherwise. The table is printed
either way, so a failure says which metric moved and by how much.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

UMAP_KEYS = ("paper_umap_mixing", "paper_umap_centroid_dist")


def scalar_metrics(d: dict) -> dict:
    return {k: float(v) for k, v in d.items()
            if k.startswith(("paper_", "gen_"))
            and isinstance(v, (int, float)) and not isinstance(v, bool)}


def compare(expected: dict, actual: dict, atol: float, umap_atol: float):
    exp, act = scalar_metrics(expected), scalar_metrics(actual)
    rows, ok = [], True
    for k in sorted(exp):
        tol = umap_atol if k in UMAP_KEYS else atol
        e = exp[k]
        a = act.get(k)
        if a is None:
            rows.append((k, e, None, None, tol, "MISSING"))
            ok = False
            continue
        if math.isnan(e) and math.isnan(a):
            rows.append((k, e, a, 0.0, tol, "ok"))
            continue
        d = abs(a - e)
        good = d <= tol
        ok &= good
        rows.append((k, e, a, d, tol, "ok" if good else "DIFF"))
    return ok, rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("expected", type=Path)
    ap.add_argument("actual", type=Path)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--umap-atol", type=float, default=0.02)
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args(argv)

    ok, rows = compare(json.loads(args.expected.read_text()),
                       json.loads(args.actual.read_text()),
                       args.atol, args.umap_atol)
    print(f"{'metric':38s} {'expected':>12s} {'actual':>12s} {'|diff|':>10s} {'tol':>8s}")
    for k, e, a, d, tol, status in rows:
        if args.quiet and status == "ok":
            continue
        fa = "—" if a is None else f"{a:12.6f}"
        fd = "—" if d is None else f"{d:10.2e}"
        print(f"{k:38s} {e:12.6f} {fa:>12s} {fd:>10s} {tol:8.0e}  {status}")
    n_bad = sum(r[-1] != "ok" for r in rows)
    print(f"\n{len(rows) - n_bad}/{len(rows)} metrics within tolerance "
          f"(atol={args.atol:g}, umap_atol={args.umap_atol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
