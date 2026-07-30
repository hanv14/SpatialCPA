"""Screen benchmark-pbya's processed datasets for fitness under the v3 protocol.

v3 runs one protocol: cut a 3-D volume into ``N`` consecutive 2-D sections, hold
out every other one, and reconstruct them. Most of v1's datasets cannot support
that — they are serial sections with large gaps, or spot-resolution, or 2-D. This
script reads each processed ``data.h5ad`` and reports the facts that decide it,
so the choice of what to add is made from the data rather than from memory.

What it measures, and why each matters
--------------------------------------
``planes``        distinct z values. Fewer than ``--n-sections`` cannot be cut at
                  all; only a few times that means each section is one or two
                  planes thick, which is not a section.
``cells/section`` under an even split of the planes. A section with a few hundred
                  cells is too sparse for the localization and field metrics.
``spacing``       median nearest-neighbour distance within a flattened section.
                  Comparable to a cell diameter = a realistic section; far larger
                  = a point cloud that no method can be scored fairly on.
``markers``       how many of v3's cortex marker/layer genes the panel carries.
                  Zero means ``evaluate_paper``'s marker and laminar-axis metrics
                  need genes chosen for that tissue before the dataset is usable.
``flank r``       **the discriminability probe**, and the number to read first.
                  Per-gene Moran's I is computed on the two sections flanking a
                  held-out one and correlated. That is what a method scores by
                  ignoring the interpolation and copying a neighbour, i.e. the
                  benchmark's floor. At 0.99 there is no room between a trivial
                  copy and a perfect reconstruction, and the dataset cannot rank
                  methods however good it looks otherwise.

The probe uses v2's own ``_morans_i`` and ``_rank_normalize`` (via
``evaluate_paper``), so it is the same definition ``paper_morans_pearson`` uses.

Usage:
    python -m src.bench3.survey_datasets
    python -m src.bench3.survey_datasets --root /path/to/benchmark-pbya
    python -m src.bench3.survey_datasets --n-sections 7 --csv survey.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np

from .config import (
    LAYER_DEEP, LAYER_SUPERFICIAL, MARKER_GENES, N_SECTIONS, RANDOM_SEED,
    SPATIAL_K, V1_ROOT,
)

# Fitness thresholds. Deliberately generous — this screens out the impossible,
# it does not certify the rest.
MIN_PLANES_PER_SECTION = 3      # for a volume: thinner than this is a plane, not a section
MIN_CELLS_PER_SECTION = 500     # below this the field/localization metrics are noise
SATURATED_FLANK_R = 0.97        # above this a copy baseline is already near-perfect
PROBE_CELLS = 2000              # cells sampled per section for the Moran's probe

# Two regimes need different rules, and the spacing between adjacent z values
# separates them. Optical planes inside one imaging block sit ~1 um apart and must
# be grouped into slabs to make a section (STARmap: 89 planes -> 7 x 11). Serial
# sections sit tens of um apart and already *are* sections, so one plane per
# section is correct there and must not be penalized.
SECTION_LIKE_GAP_UM = 5.0


def iter_datasets(root):
    """Yield (name, path) for every processed dataset under ``root``."""
    processed = Path(root) / "data" / "processed"
    if not processed.is_dir():
        return
    seen = set()
    for path in sorted(processed.rglob("data.h5ad")):
        name = str(path.parent.relative_to(processed))
        seen.add(name)
        yield name, path
    for path in sorted(processed.glob("*.h5ad")):        # pre-packaging layout
        if path.stem not in seen:
            yield path.stem, path


def _coords(adata):
    """(n, 3) coordinates from the standard obsm, else obs x/y/z. None if 2-D."""
    for key in ("spatial", "spatial3d"):
        if key in adata.obsm:
            c = np.asarray(adata.obsm[key], dtype=np.float64)
            if c.shape[1] >= 3:
                return c[:, :3]
    cols = adata.obs.columns
    for xc, yc, zc in (("x", "y", "z"), ("X", "Y", "Z")):
        if all(c in cols for c in (xc, yc, zc)):
            return np.column_stack([adata.obs[xc].values.astype(np.float64),
                                    adata.obs[yc].values.astype(np.float64),
                                    adata.obs[zc].values.astype(np.float64)])
    return None


def _median_nn(xy, rng, sample=4000):
    from scipy.spatial import cKDTree
    if len(xy) < 3:
        return float("nan")
    idx = rng.choice(len(xy), size=min(sample, len(xy)), replace=False)
    d, _ = cKDTree(xy).query(xy[idx], k=2)
    return float(np.median(d[:, 1]))


def _flank_morans_r(adata, sec_of, sections, rng):
    """Per-gene Moran's I correlation between the two sections flanking one.

    The score a method gets for copying a flanking slice — the benchmark's floor.
    """
    from scipy.stats import pearsonr
    from .evaluate_paper import _morans_i, _normalize_counts, _rank_normalize

    if len(sections) < 3:
        return float("nan")
    mid = len(sections) // 2
    lo, hi = sections[mid - 1], sections[mid + 1]

    vecs = []
    for sec in (lo, hi):
        rows = np.where(sec_of == sec)[0]
        if len(rows) < 50:
            return float("nan")
        if len(rows) > PROBE_CELLS:
            rows = np.sort(rng.choice(rows, size=PROBE_CELLS, replace=False))
        sub = adata[rows]
        X = sub.to_memory().X if adata.isbacked else sub.X
        X = np.asarray(X.todense() if hasattr(X, "todense") else X, dtype=np.float64)
        xy = _coords(adata)[rows][:, :2]
        R = _rank_normalize(_normalize_counts(X))
        vecs.append(_morans_i(xy, R, k=SPATIAL_K))

    a, b = vecs
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok])[0])


def profile(name, path, n_sections=N_SECTIONS, probe=True, rng=None):
    """Facts + verdict for one dataset."""
    import anndata as ad

    rng = rng or np.random.default_rng(RANDOM_SEED)
    row = {"dataset": name, "path": str(path)}
    try:
        adata = ad.read_h5ad(str(path), backed="r")
    except Exception as e:                       # unreadable is a fact too
        return {**row, "verdict": "unreadable", "why": str(e)[:80]}

    try:
        row["cells"] = int(adata.n_obs)
        row["genes"] = int(adata.n_vars)
        meta = adata.uns.get("spatial_metadata") or {}
        row["tech"] = str(meta.get("technology", adata.uns.get("dataset_name", "?")))
        row["tissue"] = str(meta.get("tissue", "?"))
        row["expr"] = str(adata.uns.get("expression_type", "?"))

        coords = _coords(adata)
        if coords is None:
            return {**row, "verdict": "NO — 2-D", "why": "no 3-D coordinates"}

        z = coords[:, 2]
        planes = np.unique(z)
        row["planes"] = int(len(planes))
        row["z_um"] = round(float(planes.max() - planes.min()), 1)
        gap = float(np.median(np.diff(planes))) if len(planes) > 1 else float("nan")
        row["gap_um"] = round(gap, 2)
        # A volume must be slabbed into sections; serial sections already are ones.
        row["kind"] = "serial" if gap >= SECTION_LIKE_GAP_UM else "volume"

        ct = adata.obs["cell_type"].astype(str).values if "cell_type" in adata.obs else None
        row["cell_types"] = int(len(np.unique(ct))) if ct is not None else 0

        panel = {str(g).lower() for g in adata.var_names}
        wanted = set(MARKER_GENES) | set(LAYER_SUPERFICIAL) | set(LAYER_DEEP)
        row["markers"] = f"{sum(g.lower() in panel for g in wanted)}/{len(wanted)}"

        if len(planes) < n_sections:
            return {**row, "verdict": "NO — too few planes",
                    "why": f"{len(planes)} distinct z < {n_sections} sections"}

        # The v3 partition: even split of the *planes*, then flatten.
        blocks = np.array_split(planes, n_sections)
        sec_of = np.zeros(len(z), dtype=int)
        for i, block in enumerate(blocks, start=1):
            sec_of[np.isin(z, block)] = i
        counts = np.array([(sec_of == i).sum() for i in range(1, n_sections + 1)])
        row["planes/sec"] = f"{min(len(b) for b in blocks)}-{max(len(b) for b in blocks)}"
        row["cells/sec"] = int(np.median(counts))

        mid = n_sections // 2 + 1
        row["spacing_um"] = round(_median_nn(coords[sec_of == mid][:, :2], rng), 1)

        row["flank_r"] = (round(_flank_morans_r(adata, sec_of,
                                                list(range(1, n_sections + 1)), rng), 3)
                          if probe else None)

        why = []
        thinnest = min(len(b) for b in blocks)
        if row["kind"] == "volume" and thinnest < MIN_PLANES_PER_SECTION:
            why.append(f"sections only {thinnest} optical plane(s) thick")
        if row["cells/sec"] < MIN_CELLS_PER_SECTION:
            why.append(f"{row['cells/sec']} cells/section")
        if row["cell_types"] < 2:
            why.append("no cell-type labels")
        fr = row.get("flank_r")
        if fr is not None and np.isfinite(fr) and fr >= SATURATED_FLANK_R:
            why.append(f"flanking copy already scores {fr:.3f}")

        row["verdict"] = "candidate" if not why else "marginal"
        if any(w.startswith(("sections only", "no cell-type")) for w in why):
            row["verdict"] = "NO"
        row["why"] = "; ".join(why)
        return row
    finally:
        try:
            adata.file.close()
        except Exception:
            pass


COLUMNS = ["dataset", "tech", "tissue", "kind", "cells", "genes", "planes",
           "gap_um", "z_um", "planes/sec", "cells/sec", "spacing_um", "cell_types",
           "markers", "flank_r", "verdict", "why"]


def main():
    ap = argparse.ArgumentParser(
        description="Screen benchmark-pbya datasets for the v3 section protocol")
    ap.add_argument("--root", default=str(V1_ROOT),
                    help="benchmark-pbya checkout holding data/processed")
    ap.add_argument("--n-sections", type=int, default=N_SECTIONS)
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the flanking-copy Moran's probe (faster)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--only", nargs="*", help="limit to these dataset names")
    args = ap.parse_args()

    datasets = list(iter_datasets(args.root))
    if args.only:
        datasets = [(n, p) for n, p in datasets if n in set(args.only)]
    if not datasets:
        print(f"No processed datasets under {args.root}/data/processed", file=sys.stderr)
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for name, path in datasets:
        print(f"  scanning {name} ...", file=sys.stderr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows.append(profile(name, path, n_sections=args.n_sections,
                                probe=not args.no_probe, rng=rng))

    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in COLUMNS}
    widths["why"] = min(widths["why"], 46)
    print()
    print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
    print("  ".join("-" * widths[c] for c in COLUMNS))
    order = {"candidate": 0, "marginal": 1}
    for r in sorted(rows, key=lambda r: (order.get(r.get("verdict"), 2),
                                         -(r.get("cells") or 0))):
        print("  ".join(str(r.get(c, ""))[:widths[c]].ljust(widths[c]) for c in COLUMNS))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS + ["path"], extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
