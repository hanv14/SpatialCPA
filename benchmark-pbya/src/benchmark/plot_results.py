"""Generate publication-ready benchmark figures from aggregated results."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import SUMMARY_DIR, FIGURES_DIR


def load_results(csv_path=None):
    if csv_path is None:
        csv_path = SUMMARY_DIR / "all_metrics.csv"
    return pd.read_csv(csv_path)


def plot_heatmap(df, output_dir=None):
    """Heatmap: methods x datasets, cell = median Pearson r."""
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pivot = df.groupby(["method", "dataset"])["pearson_median"].median().unstack()
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2),
                                     max(4, len(pivot.index) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0, vmax=1,
                ax=ax, linewidths=0.5)
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

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for i, ds in enumerate(sorted(datasets)):
        ax = axes[0, i]
        sub = df[df["dataset"] == ds]
        sns.boxplot(data=sub, x="method", y="pearson_median", ax=ax)
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

    summary = df.groupby(["method", "dataset"]).agg(
        wall_time_s=("wall_time_s", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if not summary.empty:
        sns.barplot(data=summary, x="dataset", y="wall_time_s", hue="method", ax=ax1)
        ax1.set_title("Wall Time (seconds)")
        ax1.tick_params(axis="x", rotation=45)

        sns.barplot(data=summary, x="dataset", y="peak_rss_mb", hue="method", ax=ax2)
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
    for method in sorted(loo["method"].unique()):
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
    for method in sorted(df["method"].unique()):
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


def generate_all_plots(csv_path=None, output_dir=None):
    """Generate all standard benchmark figures."""
    df = load_results(csv_path)
    if df.empty:
        print("No results to plot.")
        return

    plot_heatmap(df, output_dir)
    plot_correlation_boxes(df, output_dir)
    plot_runtime_memory(df, output_dir)
    plot_section_accuracy(df, output_dir)
    plot_leave_k_degradation(df, output_dir)
    print(f"Figures saved to {output_dir or FIGURES_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures")
    parser.add_argument("--csv", help="Path to all_metrics.csv")
    parser.add_argument("--output-dir", help="Output directory for figures")
    args = parser.parse_args()
    generate_all_plots(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
