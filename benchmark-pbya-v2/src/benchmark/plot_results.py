"""Generate publication-ready benchmark figures from aggregated results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

from .config import SUMMARY_DIR, FIGURES_DIR

# Heatmap ramp: neutral light grey (low) → deep red (high). Sequential data, so
# one light→dark ramp rather than a red/green diverging scale — the previous
# RdYlGn implied a midpoint that "median Pearson r" does not have, and its
# red/green poles are the worst case for red-green color blindness.
HEATMAP_COLOR_LOW = "#E8E8E8"
HEATMAP_COLOR_HIGH = "#B2182B"


def heatmap_cmap(color_low=HEATMAP_COLOR_LOW, color_high=HEATMAP_COLOR_HIGH):
    """Continuous colormap interpolating ``color_low`` → ``color_high``."""
    return LinearSegmentedColormap.from_list(
        "benchmark_sequential", [color_low, color_high])


def _annotation_ink(rgba):
    """Ink color for a cell annotation: whichever of black/white contrasts more.

    seaborn's own choice is a fixed threshold on the colormap range, which flips
    to white halfway up a light ramp and leaves mid cells at ~2.5:1. This picks
    per cell by WCAG relative luminance instead.
    """
    def channel(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgba[:3])
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    on_black = (luminance + 0.05) / 0.05
    on_white = 1.05 / (luminance + 0.05)
    return "#1A1A1A" if on_black >= on_white else "#FFFFFF"


def load_results(csv_path=None):
    if csv_path is None:
        csv_path = SUMMARY_DIR / "all_metrics.csv"
    return pd.read_csv(csv_path)


# ── Method selection / display naming ────────────────────────────────────────

def parse_method_names(pairs):
    """Parse ``METHOD=LABEL`` CLI pairs into a {method: label} dict.

    Split on the first ``=`` only, so labels may contain ``=``. Quoting is the
    shell's job: ``spatialcpav8_gen='Our Prototype'`` arrives as one argument.
    """
    if not pairs:
        return {}

    mapping = {}
    for item in pairs:
        key, sep, label = item.partition("=")
        if not sep or not key or not label:
            raise SystemExit(
                f"--method-names expects METHOD=LABEL pairs, got {item!r}")
        mapping[key] = label
    return mapping


def apply_method_selection(df, method_order=None, method_names=None):
    """Filter, order and relabel the ``method`` column for plotting.

    ``method_order`` restricts the frame to those methods and fixes the order
    they appear in across every figure; anything not listed is dropped.
    ``method_names`` maps method ids to display labels. The returned ``method``
    column is an ordered Categorical, so both pandas groupby and seaborn honor
    the requested order without each plot re-sorting.
    """
    df = df.copy()
    method_names = method_names or {}
    present = set(df["method"].unique())

    if method_order:
        missing = [m for m in method_order if m not in present]
        if missing:
            print(f"WARNING: no results for {', '.join(missing)} — skipping")
        order = [m for m in method_order if m in present]
        if not order:
            raise SystemExit(
                "--method-order matched no methods in the results "
                f"(available: {', '.join(sorted(present))})")
        dropped = sorted(present - set(order))
        if dropped:
            print(f"Excluding {len(dropped)} method(s) not in --method-order: "
                  f"{', '.join(dropped)}")
        df = df[df["method"].isin(order)]
    else:
        order = sorted(present)

    unknown = sorted(k for k in method_names if k not in present)
    if unknown:
        print(f"WARNING: --method-names given for absent method(s): "
              f"{', '.join(unknown)}")

    labels = [method_names.get(m, m) for m in order]
    duplicated = sorted({l for l in labels if labels.count(l) > 1})
    if duplicated:
        raise SystemExit("--method-names maps several methods to the same "
                         f"label: {', '.join(duplicated)}")

    df["method"] = pd.Categorical([method_names.get(m, m) for m in df["method"]],
                                  categories=labels, ordered=True)
    return df


def method_list(df):
    """Methods present in ``df``, in display order."""
    col = df["method"]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return [m for m in col.cat.categories if (col == m).any()]
    return sorted(col.unique())


def plot_heatmap(df, output_dir=None, color_low=HEATMAP_COLOR_LOW,
                 color_high=HEATMAP_COLOR_HIGH):
    """Heatmap: methods x datasets, cell = median Pearson r."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = df.groupby(["method", "dataset"], observed=True)[
        "pearson_median"].median().unstack()
    if pivot.empty:
        return
    pivot = pivot.reindex(index=method_list(df))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                     max(4, len(pivot.index) * 0.8)))
    cmap = heatmap_cmap(color_low, color_high)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, vmin=0, vmax=1,
                ax=ax, linewidths=0.5)
    # Re-ink the annotations per cell (vmin/vmax are 0/1, so value == cmap position).
    for label in ax.texts:
        try:
            value = float(label.get_text())
        except ValueError:
            continue
        label.set_color(_annotation_ink(cmap(np.clip(value, 0.0, 1.0))))
    ax.set_title("Median Pearson r (per-gene) — Methods vs Datasets")
    ax.set_ylabel("Method")
    ax.set_xlabel("Dataset")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"heatmap_pearson.{ext}", dpi=200)
    plt.close(fig)


def plot_correlation_boxes(df, output_dir=None):
    """Box plots: per-gene correlation distributions, one panel per dataset."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = df["dataset"].unique()
    n = len(datasets)
    if n == 0:
        return

    order = method_list(df)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, ds in enumerate(sorted(datasets)):
        ax = axes[0, i]
        sub = df[df["dataset"] == ds]
        sns.boxplot(data=sub, x="method", y="pearson_median", order=order, ax=ax)
        ax.set_title(ds)
        ax.set_ylim(-0.2, 1.0)
        ax.set_ylabel("Pearson r (median per-gene)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"correlation_boxes.{ext}", dpi=200)
    plt.close(fig)


def plot_runtime_memory(df, output_dir=None):
    """Bar charts: runtime and peak memory per method per dataset."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "wall_time_s" not in df.columns:
        return

    summary = df.groupby(["method", "dataset"], observed=True).agg(
        wall_time_s=("wall_time_s", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
    ).reset_index()

    order = method_list(df)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if not summary.empty:
        sns.barplot(data=summary, x="dataset", y="wall_time_s", hue="method",
                    hue_order=order, ax=ax1)
        ax1.set_title("Wall Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        sns.barplot(data=summary, x="dataset", y="peak_rss_mb", hue="method",
                    hue_order=order, ax=ax2)
        ax2.set_title("Peak RSS (MB)")
        ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"runtime_memory.{ext}", dpi=200)
    plt.close(fig)


def plot_section_accuracy(df, output_dir=None):
    """Per-section accuracy: correlation vs holdout position."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only LOO runs (single holdout section)
    loo = df[df["holdout_id"].str.startswith("loo_")].copy()
    if loo.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for method in method_list(loo):
        sub = loo[loo["method"] == method].sort_values("holdout_id")
        ax.plot(range(len(sub)), sub["pearson_median"].values, "o-",
                label=method, markersize=4)
    ax.set_xlabel("Holdout section (sorted by z)")
    ax.set_ylabel("Pearson r (median per-gene)")
    ax.set_title("Per-section interpolation accuracy (LOO)")
    ax.legend()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"section_accuracy.{ext}", dpi=200)
    plt.close(fig)


def plot_leave_k_degradation(df, output_dir=None):
    """Correlation vs number of held-out sections."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    # Parse k from holdout_id
    def get_k(hid):
        if hid.startswith("loo_"):
            return 1
        if hid.startswith("lko"):
            # lko2_secA_secB → k=2
            try:
                return int(hid.split("_")[0].replace("lko", ""))
            except ValueError:
                return None
        return None

    df["k"] = df["holdout_id"].apply(get_k)
    df = df.dropna(subset=["k", "pearson_median"])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in method_list(df):
        sub = df[df["method"] == method].groupby("k")["pearson_median"].agg(
            ["mean", "std"]).reset_index()
        ax.errorbar(sub["k"], sub["mean"], yerr=sub["std"], fmt="o-",
                     label=method, capsize=3)
    ax.set_xlabel("Number of held-out sections")
    ax.set_ylabel("Pearson r (median per-gene)")
    ax.set_title("Interpolation accuracy vs gap size")
    ax.legend()
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output_dir / f"leave_k_degradation.{ext}", dpi=200)
    plt.close(fig)


def generate_all_plots(csv_path=None, output_dir=None, method_order=None,
                       method_names=None):
    """Generate all standard benchmark figures.

    ``method_order`` / ``method_names`` are applied once here, so every figure
    shows the same methods, in the same order, under the same labels.
    """
    df = load_results(csv_path)
    if df.empty:
        print("No results to plot.")
        return

    df = apply_method_selection(df, method_order, method_names)
    print(f"Plotting {len(method_list(df))} method(s): "
          f"{', '.join(method_list(df))}")

    plot_heatmap(df, output_dir)
    plot_correlation_boxes(df, output_dir)
    plot_runtime_memory(df, output_dir)
    plot_section_accuracy(df, output_dir)
    plot_leave_k_degradation(df, output_dir)
    print(f"Figures saved to {output_dir or FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark figures",
        epilog="example: --method-order spatialcpav8_gen spatialz feast isost "
               "--method-names spatialcpav8_gen='Our Prototype' spatialz=SpatialZ")
    parser.add_argument("--csv", help="Path to all_metrics.csv")
    parser.add_argument("--output-dir", help="Output directory for figures")
    parser.add_argument("--method-order", nargs="+", metavar="METHOD",
                        help="Methods to plot, in the order they should appear "
                             "in every figure. Methods not listed are excluded.")
    parser.add_argument("--method-names", nargs="+", metavar="METHOD=LABEL",
                        help="Display label per method, e.g. "
                             "spatialcpav8_gen='Our Prototype' feast=FEAST. "
                             "Unlisted methods keep their id.")
    args = parser.parse_args()

    generate_all_plots(args.csv, args.output_dir,
                       method_order=args.method_order,
                       method_names=parse_method_names(args.method_names))


if __name__ == "__main__":
    main()
