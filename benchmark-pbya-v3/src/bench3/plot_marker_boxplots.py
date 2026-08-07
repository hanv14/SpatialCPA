"""Per-marker, per-section boxplots of the marker-pattern metrics.

``paper_marker_field_r`` and ``paper_marker_depth_r`` are means over the marker
panel (``evaluate_paper.marker_metrics``), and the pooled top-level values are a
second mean over held-out sections. Two levels of averaging leave few points to
plot: STARmap has three held-out sections, so a box over sections alone is drawn
from three numbers.

The per-marker values survive the first average — ``marker_metrics`` writes
``marker_<gene>_field_r`` / ``_depth_r`` / ``_morans_delta`` beside the means, and
``aggregate_results`` carries them into ``per_section_metrics.csv``. This module
plots *those*: one point per (held-out section x marker gene), so STARmap's three
sections and three markers give nine points per method instead of one.

    python -m src.bench3.plot_marker_boxplots
    python -m src.bench3.plot_marker_boxplots --datasets starmap_visual_cortex
    python -m src.bench3.plot_marker_boxplots --methods spatialz feast spatialcpav14_gen

One figure per dataset, because a marker panel belongs to its dataset — STARmap's
Flt1/Pcp4/Cux2 and IMC's panCK/CD3 are not the same quantity and must not share an
axis. Each run also writes the long-form table it drew, so the numbers behind the
figure are inspectable.

Marker identity is carried by point *shape* as well as position, so a reader can
see whether one gene drives a method's spread rather than only that the spread
exists — which is the whole reason for not averaging.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import nature_theme as nt
from .aggregate_results import aggregate
from .config import METHOD_ORDER, RESULTS_DIR, SUMMARY_DIR

# The per-gene statistics ``marker_metrics`` emits, and how to label them.
STATS = {
    "field_r": ("binned-field r", "higher is better"),
    "depth_r": ("depth-profile r", "higher is better"),
    "morans_delta": ("|ΔMoran's I|", "lower is better"),
}

# ``paper_marker_<gene>_<stat>``. The gene group is greedy and the stat suffix is
# anchored, so a gene whose name contains an underscore still parses. The panel
# means (``paper_marker_field_r``, ``paper_marker_morans_mae``) cannot match:
# they have no gene segment before the suffix.
_COL = re.compile(r"^paper_marker_(?P<gene>.+)_(?P<stat>%s)$" % "|".join(STATS))


def to_long(sdf):
    """Melt the per-marker columns of ``per_section_metrics`` into tidy rows.

    Returns columns: method, dataset, holdout_id, section, marker, stat, value.
    """
    keys = [c for c in ("method", "dataset", "holdout_id", "section") if c in sdf.columns]
    rows = []
    for col in sdf.columns:
        m = _COL.match(col)
        if not m:
            continue
        part = sdf[keys].copy()
        part["marker"] = m.group("gene")
        part["stat"] = m.group("stat")
        part["value"] = pd.to_numeric(sdf[col], errors="coerce")
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=keys + ["marker", "stat", "value"])
    out = pd.concat(rows, ignore_index=True)
    return out.dropna(subset=["value"]).reset_index(drop=True)


def _method_order(long_df, requested):
    present = list(dict.fromkeys(long_df["method"]))
    if requested:
        missing = [m for m in requested if m not in present]
        if missing:
            raise SystemExit(
                f"no marker values for: {', '.join(missing)}\n"
                f"  available: {', '.join(present)}")
        return list(requested)
    ordered = [m for m in METHOD_ORDER if m in present]
    return ordered + [m for m in present if m not in ordered]


def plot_dataset(long_df, dataset, methods, out_stem, width_mm=nt.WIDTH_DOUBLE_MM):
    """One figure for one dataset: a panel per statistic, a box per method."""
    import matplotlib.pyplot as plt

    sub = long_df[long_df["dataset"] == dataset]
    stats = [s for s in STATS if (sub["stat"] == s).any()]
    if not stats:
        return None
    markers = sorted(sub["marker"].unique())

    fig, axes = plt.subplots(
        1, len(stats), figsize=nt.figsize(width_mm, 70.0), squeeze=False)
    axes = axes[0]

    for ax, stat, letter in zip(axes, stats, "abcdefg"):
        s = sub[sub["stat"] == stat]
        data, positions, colors = [], [], []
        for i, method in enumerate(methods):
            vals = s.loc[s["method"] == method, "value"].to_numpy(dtype=float)
            data.append(vals if vals.size else np.array([np.nan]))
            positions.append(i)
            colors.append(nt.series_color(i))

        bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                        showfliers=False, medianprops={"color": "black", "lw": 0.8},
                        whiskerprops={"lw": 0.5}, capprops={"lw": 0.5},
                        boxprops={"lw": 0.5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
            patch.set_edgecolor(color)

        # Every underlying point, shaped by marker gene: the reason for the figure.
        rng = np.random.default_rng(0)
        for i, method in enumerate(methods):
            ms = s[s["method"] == method]
            for j, gene in enumerate(markers):
                v = ms.loc[ms["marker"] == gene, "value"].to_numpy(dtype=float)
                if not v.size:
                    continue
                x = i + rng.uniform(-0.18, 0.18, size=v.size)
                ax.plot(x, v, linestyle="none", marker=nt.series_marker(j),
                        markersize=2.6, markerfacecolor="none",
                        markeredgecolor=nt.series_color(i), markeredgewidth=0.6)

        label, direction = STATS[stat]
        ax.set_title(f"{label}  ({direction})")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.axhline(0.0, color="0.8", lw=0.4, zorder=0)
        nt.panel_letter(ax, letter)

    handles = [plt.Line2D([], [], linestyle="none", marker=nt.series_marker(j),
                          markersize=3.0, markerfacecolor="none",
                          markeredgecolor="0.25", markeredgewidth=0.6, label=g)
               for j, g in enumerate(markers)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(markers), 6),
               frameon=False, title="marker gene")
    fig.suptitle(f"{dataset} — per marker x held-out section")
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    written = nt.save(fig, out_stem)
    plt.close(fig)
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Per-marker/per-section boxplots of the marker-pattern metrics")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir", default=None,
                    help="default: results/summary/figures/<dataset>/")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None,
                    help="methods to plot, in order (at most 7 — the palette depth)")
    ap.add_argument("--width", type=float, default=nt.WIDTH_DOUBLE_MM,
                    help="figure width in mm")
    args = ap.parse_args()

    nt.activate()
    _, sdf = aggregate(args.results_dir)
    if not len(sdf):
        raise SystemExit(f"no per-section metrics under {args.results_dir}. "
                         f"Run the campaign, then aggregate_results.")

    long_df = to_long(sdf)
    if not len(long_df):
        raise SystemExit(
            "no per-marker columns found. They appear as paper_marker_<gene>_field_r "
            "in per_section_metrics.csv; a dataset whose marker panel resolved empty "
            "has none.")

    datasets = args.datasets or sorted(long_df["dataset"].unique())
    methods = _method_order(long_df, args.methods)
    if len(methods) > len(nt.PALETTE):
        raise SystemExit(
            f"{len(methods)} methods but the validated palette has "
            f"{len(nt.PALETTE)} hues, which are never cycled. Narrow with "
            f"--methods.")

    table = Path(args.out_dir or SUMMARY_DIR) / "marker_values_long.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(table, index=False)
    print(f"Wrote {len(long_df)} rows to {table}")

    for dataset in datasets:
        n = int((long_df["dataset"] == dataset).sum())
        if not n:
            print(f"  {dataset}: no marker values, skipped")
            continue
        stem = (Path(args.out_dir) / f"fig_marker_box_{dataset}" if args.out_dir
                else Path(SUMMARY_DIR) / "figures" / dataset / "fig_marker_box")
        written = plot_dataset(long_df, dataset, methods, stem, width_mm=args.width)
        if written:
            per = long_df[long_df["dataset"] == dataset]
            print(f"  {dataset}: {n} points "
                  f"({per['section'].nunique()} sections x {per['marker'].nunique()} "
                  f"markers x {per['stat'].nunique()} stats) -> {written[0]}")


if __name__ == "__main__":
    main()
