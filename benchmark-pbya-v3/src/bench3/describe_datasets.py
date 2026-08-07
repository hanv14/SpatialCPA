"""One row per *built* v3 dataset: geometry, size, annotation, processing.

``survey_datasets`` screens candidate volumes in ``benchmark-pbya`` for protocol
fitness. This is its counterpart on the other side of the build: it describes the
datasets that actually exist under ``data/processed/<dataset>/data.h5ad`` — the
table a methods section needs, and the one to read before deciding which datasets
to report.

Almost nothing here is re-derived. ``prepare_dataset`` writes the resolved
protocol into ``uns['paper_protocol']`` precisely so that a dataset is
self-describing, so the section centres, per-section z ranges, held-out split,
resolved marker panel and any applied caps are *read back* rather than recomputed
from the cells. Only the summary statistics over those records (gaps between
consecutive centres, cells per section, cell-type counts) are computed here.

    python -m src.bench3.describe_datasets
    python -m src.bench3.describe_datasets --out datasets.csv
    python -m src.bench3.describe_datasets --datasets starmap_visual_cortex imc_breast_cancer
    python -m src.bench3.describe_datasets --flank-r      # adds the discrimination probe

A note on two columns that are easy to misread:

``section_z_extent_um_median``
    The measured spread of cell z *within* a section. For ``partition="planes"``
    and ``"z_width"`` that is the slab thickness — the quantity you want. For
    ``partition="sections"`` the cells of a real serial section share one z, so
    this is ~0 and the physical section thickness is a property of the microtome
    that the data does not record. Read it together with ``partition``.

``gap_um_median``
    Spacing between consecutive section *centres*. This is the number that says
    how hard the interpolation is, and it is meaningful for every partition mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATASET_SPECS, RANDOM_SEED, dataset_path, spec
from .design import holdout_id_for

# Labels the pipeline uses for "no annotation"; both count as unannotated.
UNANNOTATED = ("unknown", "unannotated", "nan", "")


def _protocol(adata):
    """``uns['paper_protocol']`` as a plain dict (h5ad round-trips it as a mapping)."""
    pp = adata.uns.get("paper_protocol")
    return dict(pp) if pp is not None else {}


def _as_float_map(obj):
    """``{section: value}`` from a protocol record, tolerant of h5ad round-tripping."""
    if obj is None:
        return {}
    try:
        return {str(k): float(v) for k, v in dict(obj).items()}
    except (TypeError, ValueError):
        return {}


def _as_range_map(obj):
    """``{section: (lo, hi)}`` from a protocol record."""
    out = {}
    for k, v in (dict(obj).items() if obj is not None else ()):
        arr = np.asarray(v, dtype=np.float64).ravel()
        if arr.size >= 2:
            out[str(k)] = (float(arr[0]), float(arr[-1]))
    return out


def _seq(obj):
    """A protocol record's list value, as a plain list of strings.

    h5ad round-trips a list of strings as an ndarray, and ``obj or []`` on one
    raises "truth value of an array ... is ambiguous" — the same trap
    ``design.held_out_for`` documents. Test for None explicitly instead.
    """
    if obj is None:
        return []
    return [str(v) for v in np.asarray(obj).ravel()]


def _join(values):
    return "|".join(str(v) for v in values) if len(values) else ""


def _stats(values, prefix, ndigits=1):
    """min/median/max of ``values`` under ``prefix``; NaN when empty."""
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_min": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_max": np.nan}
    return {f"{prefix}_min": round(float(a.min()), ndigits),
            f"{prefix}_median": round(float(np.median(a)), ndigits),
            f"{prefix}_max": round(float(a.max()), ndigits)}


def describe(dataset, path=None, flank_r=False):
    """Summary row for one built dataset. Returns a dict, or None if not built."""
    import anndata as ad

    path = Path(path or dataset_path(dataset))
    if not path.exists():
        return None

    s = spec(dataset) if dataset in DATASET_SPECS else {}
    # Always backed. The flank probe reads X, but it subsamples to
    # ``survey_datasets.PROBE_CELLS`` per section and pulls only those rows into
    # memory (``_flank_morans_r`` handles a backed object explicitly), so reading
    # the whole matrix would cost gigabytes to use a few thousand rows — openst is
    # 1.5e6 x 3002.
    adata = ad.read_h5ad(str(path), backed="r")
    try:
        pp = _protocol(adata)
        sections = adata.obs["section"].values.astype(str)
        counts = pd.Series(sections).value_counts()

        centres = _as_float_map(pp.get("section_z_um"))
        ranges = _as_range_map(pp.get("z_ranges_um"))
        ordered = sorted(centres, key=centres.get)
        gaps = np.diff([centres[k] for k in ordered]) if len(ordered) > 1 else []
        thick = [hi - lo for lo, hi in ranges.values()]

        held = _seq(pp.get("held_out_sections"))
        n_sections = int(pp.get("n_sections") or counts.size)

        xyz = np.asarray(adata.obsm["spatial"], dtype=np.float64)
        z = xyz[:, 2]

        ct = adata.obs["cell_type"].astype(str).values if "cell_type" in adata.obs else None
        if ct is not None:
            lowered = np.char.lower(ct.astype(str))
            unann = np.isin(lowered, UNANNOTATED)
            n_types = int(len({c for c, u in zip(ct, unann) if not u}))
            pct_unann = round(100.0 * float(unann.mean()), 1)
        else:
            n_types, pct_unann = 0, 100.0

        hvg = dict(pp.get("gene_selection") or {})
        sub = dict(pp.get("cell_subsample") or {})
        markers = _seq(pp.get("marker_genes"))
        requested = _seq(pp.get("marker_genes_requested"))

        row = {
            # identity
            "dataset": dataset,
            "kind": pp.get("kind", s.get("kind")),
            "resolution": pp.get("resolution", s.get("resolution", "single_cell")),
            "technology": s.get("technology"),
            "species": s.get("species"),
            "tissue": s.get("tissue"),
            # size
            "n_cells": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "n_sections": n_sections,
            "n_held_out": len(held),
            "held_out_sections": _join(held),
            "holdout_id": holdout_id_for(held, n_sections) if held else "",
            # geometry
            "partition": pp.get("partition", s.get("partition")),
            "z_min_um": round(float(z.min()), 1),
            "z_max_um": round(float(z.max()), 1),
            "z_span_um": round(float(z.max() - z.min()), 1),
            "registration": s.get("registration"),
            "flattened_z": bool(pp.get("flattened_z", False)),
            # annotation
            "cell_type_source": pp.get("cell_type_source"),
            "n_cell_types": n_types,
            "pct_unannotated": pct_unann,
            "marker_genes": _join(markers),
            "n_markers_resolved": len(markers),
            "n_markers_requested": len(requested),
            "layer_superficial": _join(_seq(pp.get("layer_superficial"))),
            "layer_deep": _join(_seq(pp.get("layer_deep"))),
            # processing
            "trim": pp.get("trim"),
            "expression_type": str(adata.uns.get("expression_type", "")),
            "n_hvg": hvg.get("n_hvg"),
            "genes_before_hvg": hvg.get("genes_before"),
            "max_cells_per_section": sub.get("max_cells_per_section"),
        }
        row.update(_stats(counts.values, "cells_per_section", ndigits=0))
        row.update(_stats(thick, "section_z_extent_um"))
        row.update(_stats(gaps, "gap_um"))

        if flank_r:
            from .survey_datasets import _flank_morans_r
            rng = np.random.default_rng(RANDOM_SEED)
            row["flank_r"] = round(
                _flank_morans_r(adata, sections, ordered or sorted(set(sections)), rng), 3)
        return row
    finally:
        if adata.isbacked:
            adata.file.close()


# Column order for the CSV: identity, size, geometry, annotation, processing.
COLUMNS = [
    "dataset", "kind", "resolution", "technology", "species", "tissue",
    "n_cells", "n_genes", "n_sections", "n_held_out", "held_out_sections",
    "holdout_id",
    "cells_per_section_min", "cells_per_section_median", "cells_per_section_max",
    "partition", "z_min_um", "z_max_um", "z_span_um",
    "section_z_extent_um_min", "section_z_extent_um_median",
    "section_z_extent_um_max",
    "gap_um_min", "gap_um_median", "gap_um_max",
    "registration", "flattened_z",
    "cell_type_source", "n_cell_types", "pct_unannotated",
    "marker_genes", "n_markers_resolved", "n_markers_requested",
    "layer_superficial", "layer_deep",
    "trim", "expression_type", "n_hvg", "genes_before_hvg",
    "max_cells_per_section", "flank_r",
]


def main():
    ap = argparse.ArgumentParser(
        description="Summarize the built benchmark-pbya-v3 datasets as one CSV row each")
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="dataset names (default: every registered one that is built)")
    ap.add_argument("--out", default=None,
                    help="CSV path (default: results/summary/datasets.csv)")
    ap.add_argument("--flank-r", action="store_true",
                    help="also compute the flanking-slice Moran's correlation — the "
                         "score a method gets by copying a neighbour, i.e. how much "
                         "room the dataset leaves for a method to win. Reads X, so "
                         "it is much slower on the large volumes.")
    args = ap.parse_args()

    from .config import SUMMARY_DIR

    names = args.datasets or list(DATASET_SPECS)
    rows, missing = [], []
    for name in names:
        try:
            row = describe(name, flank_r=args.flank_r)
        except Exception as e:                       # one bad file must not hide the rest
            print(f"  {name}: FAILED — {type(e).__name__}: {e}")
            continue
        if row is None:
            missing.append(name)
        else:
            rows.append(row)
            print(f"  {name}: {row['n_cells']} cells x {row['n_genes']} genes, "
                  f"{row['n_sections']} sections")

    if missing:
        print(f"\nNot built ({len(missing)}): {', '.join(missing)}\n"
              f"  build with: python -m src.bench3.prepare_dataset --dataset <name>")
    if not rows:
        raise SystemExit("No built datasets found.")

    df = pd.DataFrame(rows)
    df = df[[c for c in COLUMNS if c in df.columns]]
    out = Path(args.out or (Path(SUMMARY_DIR) / "datasets.csv"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {len(df)} rows to {out}")

    compact = ["dataset", "kind", "resolution", "n_cells", "n_genes", "n_sections",
               "n_held_out", "gap_um_median", "section_z_extent_um_median",
               "n_cell_types", "pct_unannotated"]
    if "flank_r" in df.columns:
        compact.append("flank_r")
    print()
    print(df[[c for c in compact if c in df.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
