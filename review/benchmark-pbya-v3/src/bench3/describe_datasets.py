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
    python -m src.bench3.describe_datasets --designs wide --holdout-block 5
    python -m src.bench3.describe_datasets --designs      # main table only, no scenarios

The per-dataset row reports the *built* (paper) split. In addition, the design
scenarios block expands every holdout design ``run_all``/``run_benchmark`` accepts
— ``paper``, ``loo`` and ``wide`` — so the same datasets can be read under each
``--design``: which sections each holds out, how many runs it is, and
``max_holdout_gap_um``, the hardest held-out -> nearest-input gap (one spacing for
``paper``/``loo``, several for ``wide``). Detailed per-run rows are written to
``dataset_designs.csv``; a compact per-(dataset, design) summary is printed.

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
from .design import (
    consecutive_design, holdout_id_for, loo_design, paper_design, sorted_sections,
)

# Holdout designs described per dataset by ``--designs`` (see design.py). The
# main per-dataset row reports the built (paper) split; this is the same set of
# designs run_all / run_benchmark accept, so the table shows what each `--design`
# would actually hold out on every dataset — not just the paper one.
DESIGN_NAMES = ("paper", "loo", "wide")

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


def describe(dataset, path=None, flank_r=False, n_examples=5):
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
            real = pd.Series(ct[~unann])
            n_types = int(real.nunique())
            pct_unann = round(100.0 * float(unann.mean()), 1)
            # Most frequent first, and the placeholder labels excluded, so the
            # column always shows what the annotation actually *says*. Reading the
            # labels is how a wrong one is caught: cluster ids where subclass names
            # were intended look nothing alike, and no count column shows that.
            examples = list(real.value_counts().index[:int(n_examples)])
        else:
            n_types, pct_unann, examples = 0, 100.0, []

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
            "cell_type_examples": _join(examples),
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


def describe_designs(dataset, path=None, designs=DESIGN_NAMES, wide_block=None):
    """Per-design holdout plan for one built dataset: one row per (design, run).

    Complements ``describe``: where that row reports the *built* (paper) split,
    this expands every design ``run_all``/``run_benchmark`` accepts, so the same
    dataset can be read under each ``--design``. ``loo`` yields one run per
    interior section; ``paper`` and ``wide`` yield a single multi-section run.

    ``max_holdout_gap_um`` is the difficulty summary that actually separates the
    designs: for each run, the largest distance from a held-out section centre to
    the nearest *input* section centre — how far the method must reach. It is one
    spacing for ``paper``/``loo`` and several for ``wide``. Returns a list of row
    dicts (a per-design ``error`` row if a design does not apply, e.g. ``wide`` on
    a volume with fewer than three sections), or None if the dataset is not built.
    """
    import anndata as ad

    path = Path(path or dataset_path(dataset))
    if not path.exists():
        return None

    adata = ad.read_h5ad(str(path), backed="r")
    try:
        _labels, z_by = sorted_sections(adata)
        n_sec = len(_labels)

        def _max_gap(held, keep):
            if not held or not keep:
                return np.nan
            kv = np.array([z_by[k] for k in keep if k in z_by], dtype=np.float64)
            hv = [z_by[h] for h in held if h in z_by]
            if kv.size == 0 or not hv:
                return np.nan
            return round(float(max(np.min(np.abs(kv - h)) for h in hv)), 1)

        rows = []
        for d in designs:
            try:
                if d == "paper":
                    configs = paper_design(adata)
                elif d == "loo":
                    configs = loo_design(adata)
                elif d == "wide":
                    configs = consecutive_design(adata, block=wide_block)
                else:
                    rows.append({"dataset": dataset, "design": d,
                                 "error": "unknown design"})
                    continue
            except Exception as e:                    # design inapplicable here
                rows.append({"dataset": dataset, "design": d,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            if not configs:                           # e.g. loo with no interior
                rows.append({"dataset": dataset, "design": d,
                             "error": f"no runs (n_sections={n_sec})"})
                continue
            for cfg in configs:
                held = list(cfg["holdout_sections"])
                keep = list(cfg["remaining_sections"])
                rows.append({
                    "dataset": dataset,
                    "design": d,
                    "holdout_id": cfg["holdout_id"],
                    "n_held_out": len(held),
                    "held_out_sections": _join(held),
                    "n_input": len(keep),
                    "max_holdout_gap_um": _max_gap(held, keep),
                })
        return rows
    finally:
        if adata.isbacked:
            adata.file.close()


DESIGN_COLUMNS = [
    "dataset", "design", "holdout_id", "n_held_out", "held_out_sections",
    "n_input", "max_holdout_gap_um", "error",
]

# Print order for the design-scenario summary.
_DESIGN_ORDER = {"paper": 0, "loo": 1, "wide": 2}


def _print_design_summary(ddf):
    """Compact one-line-per-(dataset, design) view of the holdout scenarios."""
    has_err = "error" in ddf.columns
    print("\ndesign scenarios (holdout per design; "
          "max_gap = hardest held-out -> nearest-input gap)")
    print(f"  {'dataset':28s} {'design':6s} {'runs':>4s} {'held/run':>8s} "
          f"{'input':>6s} {'max_gap_um':>10s}  holdout_id(s)")
    for dataset, g in ddf.groupby("dataset", sort=False):
        for design, gg in sorted(g.groupby("design"),
                                 key=lambda kv: _DESIGN_ORDER.get(kv[0], 9)):
            real = gg[gg["error"].isna()] if has_err else gg
            if real.empty:
                err = gg["error"].dropna().iloc[0] if has_err else "n/a"
                print(f"  {dataset:28s} {design:6s}   n/a  ({err})")
                continue
            runs = len(real)
            held = int(real["n_held_out"].iloc[0])
            n_in = int(real["n_input"].iloc[0])
            mg = real["max_holdout_gap_um"].astype(float).max()
            mgs = "nan" if not np.isfinite(mg) else f"{mg:.1f}"
            ids = list(real["holdout_id"])
            id_str = ids[0] if runs == 1 else f"{ids[0]} … {ids[-1]} ({runs} runs)"
            print(f"  {dataset:28s} {design:6s} {runs:>4d} {held:>8d} {n_in:>6d} "
                  f"{mgs:>10s}  {id_str}")


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
    "cell_type_source", "n_cell_types", "pct_unannotated", "cell_type_examples",
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
    ap.add_argument("--n-examples", type=int, default=5,
                    help="how many of the most frequent cell-type labels to show "
                         "in cell_type_examples (default 5)")
    ap.add_argument("--flank-r", action="store_true",
                    help="also compute the flanking-slice Moran's correlation — the "
                         "score a method gets by copying a neighbour, i.e. how much "
                         "room the dataset leaves for a method to win. Reads X, so "
                         "it is much slower on the large volumes.")
    ap.add_argument("--designs", nargs="*", choices=list(DESIGN_NAMES),
                    default=list(DESIGN_NAMES),
                    help="also describe the holdout under each of these designs "
                         "(default: all). Pass with no names to skip the design "
                         "scenarios entirely.")
    ap.add_argument("--holdout-block", type=int, default=None,
                    help="consecutive block width for the 'wide' design scenario "
                         "(default 3; clamped to n-2 per dataset)")
    ap.add_argument("--designs-out", default=None,
                    help="CSV path for the per-(design, run) rows "
                         "(default: results/summary/dataset_designs.csv)")
    args = ap.parse_args()

    from .config import SUMMARY_DIR

    names = args.datasets or list(DATASET_SPECS)
    rows, missing = [], []
    for name in names:
        try:
            row = describe(name, flank_r=args.flank_r, n_examples=args.n_examples)
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

    # Kept out of the table above, which is already wide: printed as its own
    # block so the labels stay readable at full length.
    print("\ncell-type labels (most frequent first)")
    for _, r in df.iterrows():
        print(f"  {r['dataset']:28s} {r['cell_type_examples'] or '(none — unannotated)'}")

    # ── design scenarios ──────────────────────────────────────────────────────
    # The per-dataset table above reports the built (paper) split; this expands
    # every design run_all/run_benchmark accepts, so the same datasets can be read
    # under paper, loo and wide. Skipped when --designs is given with no names.
    if args.designs:
        design_rows = []
        for name in names:
            if not dataset_path(name).exists():
                continue
            try:
                dr = describe_designs(name, designs=args.designs,
                                      wide_block=args.holdout_block)
            except Exception as e:
                print(f"  {name}: design scenarios FAILED — {type(e).__name__}: {e}")
                continue
            if dr:
                design_rows.extend(dr)
        if design_rows:
            ddf = pd.DataFrame(design_rows)
            if "error" not in ddf.columns:
                ddf["error"] = np.nan
            ddf = ddf[[c for c in DESIGN_COLUMNS if c in ddf.columns]]
            dout = Path(args.designs_out or (Path(SUMMARY_DIR) / "dataset_designs.csv"))
            dout.parent.mkdir(parents=True, exist_ok=True)
            ddf.to_csv(dout, index=False)
            print(f"\nWrote {len(ddf)} design-scenario rows to {dout}")
            _print_design_summary(ddf)


if __name__ == "__main__":
    main()
