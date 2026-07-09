"""Plot the disaggregated (per-gene / per-cell) evaluation results.

Companion to ``evaluate_disaggregated.py``. That script emits the *full*
distribution behind every reduced metric; this one draws it. For each metric it
produces **one figure with a grid of panels, one panel per dataset**, and within
each panel the distribution of that metric across methods (violin + inner box) —
so you see the spread over genes / cells that the median in ``metrics.json``
throws away, not just a point estimate.

Styling follows the **Springer Nature** figure convention: Helvetica/Arial sans
text, thin (0.5 pt) axes, no top/right spines, outward ticks, editable
(TrueType) PDF text, and the ``ggsci`` *npg* ("Nature Publishing Group") colour
palette for methods (kept consistent across every panel and figure).

Inputs are the concatenated master CSVs written by
``evaluate_disaggregated.py --all`` (``per_gene_matched.csv``,
``per_gene_generation.csv``, ``per_cell.csv``); by default they are read from
``results/summary/disaggregated/``.

Usage
-----
    # after: python -m src.benchmark.evaluate_disaggregated --all
    python -m src.benchmark.plot_disaggregated
    python -m src.benchmark.plot_disaggregated --metrics pearson field_pearson
    python -m src.benchmark.plot_disaggregated --kind box --input-dir <dir> --output-dir <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .config import SUMMARY_DIR, FIGURES_DIR

# ── Springer Nature ("ggsci" npg / NPG) colour palette ────────────────────────
NPG_PALETTE = [
    "#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F",
    "#8491B4", "#91D1C2", "#DC0000", "#7E6148", "#B09C85",
]


def set_nature_style():
    """Apply Springer-Nature-style Matplotlib rcParams (returns the palette)."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "pdf.fonttype": 42,   # editable text in Illustrator, per Nature guidelines
        "ps.fonttype": 42,
    })
    return NPG_PALETTE


# ── Metric registry: which disaggregated columns to plot ──────────────────────
# table: master CSV basename; column: value column; label: axis/figure title;
# ylim: fixed y-range or None (auto); zero_line: draw a y=0 reference.
METRICS = [
    # per-gene, cell-matched (correspondence-dependent; behind pearson_median etc.)
    dict(table="per_gene_matched", column="pearson",
         label="Per-gene Pearson r (cell-matched)", ylim=(-1, 1), zero_line=True),
    dict(table="per_gene_matched", column="spearman",
         label="Per-gene Spearman ρ (cell-matched)", ylim=(-1, 1), zero_line=True),
    dict(table="per_gene_matched", column="rmse",
         label="Per-gene RMSE (cell-matched)", ylim=(0, None), zero_line=False),
    dict(table="per_gene_matched", column="mae",
         label="Per-gene MAE (cell-matched)", ylim=(0, None), zero_line=False),
    # per-gene, correspondence-free generation primaries
    dict(table="per_gene_generation", column="field_pearson",
         label="Per-gene spatial-field Pearson r", ylim=(-1, 1), zero_line=True),
    dict(table="per_gene_generation", column="morans_i_pred",
         label="Per-gene Moran's I (prediction)", ylim=None, zero_line=True),
    # per-cell
    dict(table="per_cell", column="cell_pearson",
         label="Per-cell expression Pearson r (matched)", ylim=(-1, 1), zero_line=True),
    dict(table="per_cell", column="cell_mae",
         label="Per-cell expression MAE (matched)", ylim=(0, None), zero_line=False),
    dict(table="per_cell", column="nn_dist_um",
         label="Per-cell nearest-prediction distance (µm)", ylim=(0, None), zero_line=False),
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_tables(input_dir):
    """Load whichever master CSVs are present. Returns {table_name: DataFrame}."""
    input_dir = Path(input_dir)
    tables = {}
    for name in ("per_gene_matched", "per_gene_generation", "per_cell"):
        path = input_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if not df.empty:
                tables[name] = df
    return tables


def _method_color_map(tables):
    """Stable method -> colour map (NPG palette) across every table/figure."""
    methods = set()
    for df in tables.values():
        if "method" in df.columns:
            methods |= set(df["method"].dropna().unique())
    methods = sorted(methods)
    return {m: NPG_PALETTE[i % len(NPG_PALETTE)] for i, m in enumerate(methods)}


# ── One panel (one dataset): distribution across methods ──────────────────────
def _draw_panel(ax, sub, column, methods, color_map, kind, ylim, zero_line):
    """Draw the per-method distribution of ``column`` for one dataset."""
    positions, data, colors, present = [], [], [], []
    for i, m in enumerate(methods):
        vals = pd.to_numeric(sub.loc[sub["method"] == m, column],
                             errors="coerce").to_numpy()
        vals = vals[np.isfinite(vals)]
        positions.append(i)
        data.append(vals)
        colors.append(color_map.get(m, "#777777"))
        present.append(len(vals) > 0)

    if zero_line:
        ax.axhline(0, color="0.7", lw=0.5, ls="--", zorder=0)

    for i, (vals, c) in enumerate(zip(data, colors)):
        if len(vals) == 0:
            continue
        if kind in ("violin", "box") and len(vals) >= 2:
            if kind == "violin":
                vp = ax.violinplot([vals], positions=[i], widths=0.8,
                                   showextrema=False, showmedians=False)
                for body in vp["bodies"]:
                    body.set_facecolor(c)
                    body.set_edgecolor(c)
                    body.set_alpha(0.45)
                    body.set_linewidth(0.4)
                # slim box overlay for median + IQR
                bp = ax.boxplot([vals], positions=[i], widths=0.14,
                                showfliers=False, patch_artist=True,
                                medianprops=dict(color="black", lw=0.8),
                                boxprops=dict(facecolor="white", edgecolor=c, lw=0.6),
                                whiskerprops=dict(color=c, lw=0.6),
                                capprops=dict(color=c, lw=0.6))
            else:  # box
                ax.boxplot([vals], positions=[i], widths=0.55,
                           showfliers=False, patch_artist=True,
                           medianprops=dict(color="black", lw=0.8),
                           boxprops=dict(facecolor=c, edgecolor=c, alpha=0.6, lw=0.6),
                           whiskerprops=dict(color=c, lw=0.6),
                           capprops=dict(color=c, lw=0.6))
        else:  # strip, or a single point
            jitter = (np.random.default_rng(0).uniform(-0.12, 0.12, size=len(vals))
                      if len(vals) > 1 else np.zeros(len(vals)))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=3,
                       color=c, alpha=0.6, linewidths=0)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_xlim(-0.6, len(methods) - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0)


# ── One figure per metric: grid of dataset panels ─────────────────────────────
def plot_metric(df, metric, methods, color_map, output_dir, kind="violin"):
    column = metric["column"]
    if column not in df.columns:
        return False
    df = df[np.isfinite(pd.to_numeric(df[column], errors="coerce"))]
    if df.empty:
        return False

    datasets = sorted(df["dataset"].dropna().unique())
    n = len(datasets)
    if n == 0:
        return False

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(max(2.1 * ncols, 3.2), max(2.0 * nrows, 2.2)))

    methods_present = [m for m in methods
                       if m in set(df["method"].dropna().unique())]

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx >= n:
            ax.axis("off")
            continue
        ds = datasets[idx]
        _draw_panel(ax, df[df["dataset"] == ds], column, methods_present,
                    color_map, kind, metric["ylim"], metric["zero_line"])
        ax.set_title(ds)
        if c == 0:
            ax.set_ylabel(metric["label"])
        ax.set_xlabel("")

    # single shared method legend (colour key) as a horizontal strip at the
    # bottom — avoids colliding with (possibly long) per-panel dataset titles.
    handles = [Patch(facecolor=color_map[m], edgecolor="none", alpha=0.7, label=m)
               for m in methods_present]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 6), title="method", title_fontsize=6)

    fig.suptitle(metric["label"], y=0.99, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 if handles else 0.0, 1.0, 0.95))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric['table']}__{column}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}")
    plt.close(fig)
    return True


def generate_all(input_dir=None, output_dir=None, metrics=None, kind="violin"):
    set_nature_style()
    input_dir = Path(input_dir) if input_dir else (SUMMARY_DIR / "disaggregated")
    output_dir = Path(output_dir) if output_dir else (FIGURES_DIR / "disaggregated")

    tables = load_tables(input_dir)
    if not tables:
        print(f"No disaggregated master CSVs found in {input_dir}.\n"
              f"Run:  python -m src.benchmark.evaluate_disaggregated --all")
        return

    color_map = _method_color_map(tables)
    methods = list(color_map.keys())

    wanted = set(metrics) if metrics else None
    made = 0
    for metric in METRICS:
        if wanted and metric["column"] not in wanted and metric["label"] not in wanted:
            continue
        df = tables.get(metric["table"])
        if df is None:
            continue
        if plot_metric(df, metric, methods, color_map, output_dir, kind=kind):
            print(f"  wrote {metric['table']}__{metric['column']}.png/.pdf")
            made += 1
    print(f"\n{made} metric figure(s) written to {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot disaggregated per-gene/per-cell results "
                    "(grid of dataset panels per metric; Springer Nature theme)")
    ap.add_argument("--input-dir",
                    help="dir with the disaggregated master CSVs "
                         "(default: results/summary/disaggregated)")
    ap.add_argument("--output-dir",
                    help="dir for figures (default: results/summary/figures/disaggregated)")
    ap.add_argument("--metrics", nargs="+",
                    help="only plot these metric columns (e.g. pearson field_pearson)")
    ap.add_argument("--kind", choices=["violin", "box", "strip"], default="violin",
                    help="per-method distribution style (default: violin+box)")
    args = ap.parse_args()
    generate_all(args.input_dir, args.output_dir, args.metrics, args.kind)


if __name__ == "__main__":
    main()
