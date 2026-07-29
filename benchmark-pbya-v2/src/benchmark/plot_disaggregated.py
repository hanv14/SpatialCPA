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

    # only plot chosen datasets, in the given panel order
    python -m src.benchmark.plot_disaggregated --datasets cosmx_nsclc_3d imc_breast_cancer

    # control method order (left-to-right) and display names on the plots
    python -m src.benchmark.plot_disaggregated \
        --method-order spatialcpav8_gen spatialz feast isost \
        --method-names spatialcpav8_gen='SpatialCPA v8' spatialz=SpatialZ \
                       feast=FEAST isost=isoST
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
# ``tier`` groups metrics by what they can support (select with --tier):
#   paper      — scale-fair and/or directly interpretable; safe to publish
#   diagnostic — informative but conditioned/one-sided; read alongside another
#                panel, don't headline
#   reference  — the v1 cell-matched family, kept for continuity
# ``caveat`` is rendered under the figure title so a panel can't be read out of
# context once it leaves this repo.
METRICS = [
    # ── paper ────────────────────────────────────────────────────────────────
    dict(table="per_gene_generation", column="field_pearson",
         label="Per-gene spatial-field Pearson r", ylim=(-1, 1), zero_line=True,
         tier="paper",
         caveat="rank-normalized (scale-fair); alignment-dependent — "
                "trustworthy on asymmetric tissue"),
    dict(table="per_gene_generation", column="morans_i_delta",
         label="Per-gene Moran's I, prediction − GT", ylim=None, zero_line=True,
         tier="paper",
         caveat="0 = prediction matches GT spatial autocorrelation; "
                ">0 over-structured, <0 under-structured"),
    dict(table="per_cell", column="nn_dist_um",
         label="Per-cell nearest-prediction distance (µm)", ylim=(0, None),
         zero_line=False, tier="paper",
         caveat="one-directional (GT → prediction): over-generating cells "
                "shortens it — read with the predicted cell count"),
    # ── diagnostic ───────────────────────────────────────────────────────────
    dict(table="per_cell", column="cell_pearson",
         label="Per-cell expression Pearson r (matched)", ylim=(-1, 1),
         zero_line=True, tier="diagnostic",
         caveat="matched cells only (see % per violin) — a method that places "
                "few cells is scored on its best ones"),
    dict(table="per_cell", column="cell_mae",
         label="Per-cell expression MAE (matched)", ylim=(0, None),
         zero_line=False, tier="diagnostic",
         caveat="matched cells only (see % per violin); log-normalized"),
    # ── reference ────────────────────────────────────────────────────────────
    dict(table="per_gene_matched", column="pearson",
         label="Per-gene Pearson r (cell-matched)", ylim=(-1, 1), zero_line=True,
         tier="reference",
         caveat="cell-matched: manufactures a correspondence generation does "
                "not produce — not the primary measure for generation"),
    dict(table="per_gene_matched", column="spearman",
         label="Per-gene Spearman ρ (cell-matched)", ylim=(-1, 1), zero_line=True,
         tier="reference",
         caveat="cell-matched: manufactures a correspondence generation does "
                "not produce — not the primary measure for generation"),
    dict(table="per_gene_matched", column="rmse",
         label="Per-gene RMSE (cell-matched)", ylim=(0, None), zero_line=False,
         tier="reference",
         caveat="computed on RAW output — compares output scales as much as "
                "accuracy; not comparable across methods"),
    dict(table="per_gene_matched", column="mae",
         label="Per-gene MAE (cell-matched)", ylim=(0, None), zero_line=False,
         tier="reference",
         caveat="computed on RAW output — compares output scales as much as "
                "accuracy; not comparable across methods"),
    dict(table="per_gene_generation", column="morans_i_pred",
         label="Per-gene Moran's I (prediction)", ylim=None, zero_line=True,
         tier="reference",
         caveat="prediction only — not an agreement with GT; "
                "prefer morans_i_delta"),
]

TIERS = ("paper", "diagnostic", "reference")


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


def _derive_columns(tables):
    """Add columns computed from the stored ones.

    ``morans_i_delta`` = prediction − GT Moran's I, per gene. The stored
    ``morans_i_pred`` alone says only that the prediction is spatially
    structured, not that it is structured *like GT* — a method that overshoots
    GT's autocorrelation scores higher on it. The signed difference is the
    agreement statement, and both inputs are already in the table.
    """
    gen = tables.get("per_gene_generation")
    if gen is not None and {"morans_i_pred", "morans_i_gt"} <= set(gen.columns):
        gen = gen.copy()
        gen["morans_i_delta"] = (pd.to_numeric(gen["morans_i_pred"], errors="coerce")
                                 - pd.to_numeric(gen["morans_i_gt"], errors="coerce"))
        tables["per_gene_generation"] = gen
    return tables


def _methods_in_tables(tables):
    """Every method id appearing in any loaded table."""
    methods = set()
    for df in tables.values():
        if "method" in df.columns:
            methods |= set(df["method"].dropna().unique())
    return methods


def _method_color_map(tables, methods=None):
    """Method -> colour map (NPG palette), consistent across table/figure.

    Colours are assigned over ``methods`` when given (i.e. over the selection
    being plotted) rather than over everything in the tables — the palette has
    only 10 entries, so assigning across a dozen-plus method ids wraps and hands
    two plotted methods the same colour.
    """
    if methods is None:
        methods = sorted(_methods_in_tables(tables))
    if len(methods) > len(NPG_PALETTE):
        print(f"  note: {len(methods)} methods but {len(NPG_PALETTE)} palette "
              f"colours — some colours repeat; narrow --method-order to avoid it")
    return {m: NPG_PALETTE[i % len(NPG_PALETTE)] for i, m in enumerate(methods)}


# ── One panel (one dataset): distribution across methods ──────────────────────
def _draw_panel(ax, sub, column, methods, color_map, kind, ylim, zero_line,
                labels=None):
    """Draw the per-method distribution of ``column`` for one dataset."""
    positions, data, colors, present, coverage = [], [], [], [], []
    for i, m in enumerate(methods):
        col = pd.to_numeric(sub.loc[sub["method"] == m, column],
                            errors="coerce").to_numpy()
        vals = col[np.isfinite(col)]
        positions.append(i)
        data.append(vals)
        colors.append(color_map.get(m, "#777777"))
        present.append(len(vals) > 0)
        # Share of this method's rows the violin actually rests on. For the
        # per-cell metrics the dropped rows are the unmatched cells, so this is
        # the match rate — the survivorship the distribution is conditioned on.
        coverage.append(len(vals) / len(col) if len(col) else 0.0)

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

    # Flag any violin resting on less than all of its rows, so a distribution
    # conditioned on a small matched subset can't be misread as the whole slice.
    for i, cov in enumerate(coverage):
        if present[i] and cov < 0.995:
            ax.annotate(f"{cov * 100:.0f}%", xy=(i, 1.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, -1), textcoords="offset points",
                        ha="center", va="top", fontsize=4.5, color="0.35")

    labels = labels or {}
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([labels.get(m, m) for m in methods], rotation=45, ha="right")
    ax.set_xlim(-0.6, len(methods) - 0.4)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.margins(x=0)


# ── One figure per metric: grid of dataset panels ─────────────────────────────
def plot_metric(df, metric, methods, color_map, output_dir, kind="violin",
                method_labels=None, dataset_order=None):
    column = metric["column"]
    if column not in df.columns:
        return False
    finite = np.isfinite(pd.to_numeric(df[column], errors="coerce"))
    if not finite.any():
        return False

    # Non-finite rows are kept: _draw_panel needs them to report the share of
    # rows each violin rests on. Presence checks use the finite rows only.
    present = set(df.loc[finite, "dataset"].dropna().unique())
    if dataset_order:
        # keep the user's chosen datasets, in the given order
        datasets = [d for d in dataset_order if d in present]
    else:
        datasets = sorted(present)
    n = len(datasets)
    if n == 0:
        return False

    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    # The title (+ caveat) gets its own strip added to the figure height rather
    # than carved out of the panels, so panel size doesn't depend on how long
    # the metric's caveat is.
    caveat = metric.get("caveat")
    header_in = 0.52 if caveat else 0.26
    fig_h = max(2.0 * nrows, 2.2) + header_in
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(max(2.1 * ncols, 3.2), fig_h))

    methods_present = [m for m in methods
                       if m in set(df.loc[finite, "method"].dropna().unique())]

    for idx in range(nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if idx >= n:
            ax.axis("off")
            continue
        ds = datasets[idx]
        _draw_panel(ax, df[df["dataset"] == ds], column, methods_present,
                    color_map, kind, metric["ylim"], metric["zero_line"],
                    labels=method_labels)
        ax.set_title(ds)
        if c == 0:
            ax.set_ylabel(metric["label"])
        ax.set_xlabel("")

    # single shared method legend (colour key) as a horizontal strip at the
    # bottom — avoids colliding with (possibly long) per-panel dataset titles.
    method_labels = method_labels or {}
    handles = [Patch(facecolor=color_map[m], edgecolor="none", alpha=0.7,
                     label=method_labels.get(m, m))
               for m in methods_present]
    if handles:
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
                   ncol=min(len(handles), 6), title="method", title_fontsize=6)

    # Header offsets in inches, converted to figure fractions: the figure height
    # varies with the panel-grid rows, so a fixed fraction would put the caveat
    # on top of the title on short figures.
    fig.suptitle(metric["label"], y=1 - 0.13 / fig_h, fontsize=9, va="top")
    if caveat:
        fig.text(0.5, 1 - 0.32 / fig_h, caveat, ha="center", va="top",
                 fontsize=5, color="0.35", wrap=True)
    fig.tight_layout(rect=(0, 0.06 if handles else 0.0, 1.0,
                           1 - header_in / fig_h))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{metric['table']}__{column}"
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{ext}")
    plt.close(fig)
    return True


def _order_methods(present, method_order, include_unlisted=False):
    """Left-to-right method order, and which methods appear at all.

    ``present`` is the set of method ids in the data. Anything in
    ``method_order`` that isn't present is dropped (with a note). By default
    ``--method-order`` is also a *selection*: methods present but unlisted are
    excluded, so the figures show exactly what was asked for (same semantics as
    ``plot_results``). Pass ``include_unlisted`` to append them instead.
    """
    present = set(present)
    if not method_order:
        return sorted(present)
    missing = [m for m in method_order if m not in present]
    if missing:
        print(f"  note: --method-order names not in data (ignored): {missing}")
    ordered = [m for m in method_order if m in present]
    if not ordered:
        raise SystemExit(
            "--method-order matched no methods in the data "
            f"(available: {', '.join(sorted(present))})")
    rest = sorted(present - set(ordered))
    if rest and include_unlisted:
        print(f"  note: methods present but not in --method-order "
              f"(appended after): {rest}")
        return ordered + rest
    if rest:
        print(f"  note: excluding {len(rest)} method(s) not in --method-order: "
              f"{', '.join(rest)}")
    return ordered


def generate_all(input_dir=None, output_dir=None, metrics=None, kind="violin",
                 method_order=None, method_labels=None, datasets=None,
                 include_unlisted=False, tiers=None):
    set_nature_style()
    input_dir = Path(input_dir) if input_dir else (SUMMARY_DIR / "disaggregated")
    output_dir = Path(output_dir) if output_dir else (FIGURES_DIR / "disaggregated")

    tables = _derive_columns(load_tables(input_dir))
    if not tables:
        print(f"No disaggregated master CSVs found in {input_dir}.\n"
              f"Run:  python -m src.benchmark.evaluate_disaggregated --all")
        return

    present = _methods_in_tables(tables)
    methods = _order_methods(present, method_order,
                             include_unlisted=include_unlisted)
    color_map = _method_color_map(tables, methods)  # colours keyed by method id

    method_labels = method_labels or {}
    unknown = [k for k in method_labels if k not in present]
    if unknown:
        print(f"  note: --method-names for methods not in data (ignored): {unknown}")

    if datasets:
        present = set(d for df in tables.values() for d in df["dataset"].dropna().unique())
        missing = [d for d in datasets if d not in present]
        if missing:
            print(f"  note: --datasets not found in data (ignored): {missing}")
        if not any(d in present for d in datasets):
            print("  no requested datasets present; nothing to plot.")
            return

    wanted = set(metrics) if metrics else None
    wanted_tiers = set(tiers) if tiers else None
    made = 0
    for metric in METRICS:
        if wanted and metric["column"] not in wanted and metric["label"] not in wanted:
            continue
        if wanted_tiers and metric.get("tier") not in wanted_tiers:
            continue
        df = tables.get(metric["table"])
        if df is None:
            continue
        if plot_metric(df, metric, methods, color_map, output_dir, kind=kind,
                       method_labels=method_labels, dataset_order=datasets):
            print(f"  wrote {metric['table']}__{metric['column']}.png/.pdf")
            made += 1
    print(f"\n{made} metric figure(s) written to {output_dir}")


def _parse_method_names(pairs):
    """Parse ``key=Display Name`` pairs into a {method_id: label} dict."""
    if not pairs:
        return {}
    mapping = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(
                f"--method-names entry '{p}' must be of the form key=Label "
                f"(e.g. spatialz='SpatialZ')")
        key, label = p.split("=", 1)
        mapping[key.strip()] = label.strip()
    return mapping


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
    ap.add_argument("--tier", nargs="+", choices=TIERS, metavar="TIER",
                    help="only plot metrics in these tiers: "
                         "paper (scale-fair / directly interpretable), "
                         "diagnostic (conditioned or one-sided — read alongside "
                         "another panel), reference (v1 cell-matched family). "
                         "Default: every tier")
    ap.add_argument("--datasets", nargs="+", metavar="DATASET",
                    help="only plot these datasets, in this order "
                         "(e.g. cosmx_nsclc_3d imc_breast_cancer); "
                         "default plots every dataset present, sorted")
    ap.add_argument("--kind", choices=["violin", "box", "strip"], default="violin",
                    help="per-method distribution style (default: violin+box)")
    ap.add_argument("--method-order", nargs="+", metavar="METHOD",
                    help="methods to plot, left-to-right, e.g. "
                         "--method-order spatialcpav8_gen spatialz feast isost "
                         "(methods present but unlisted are EXCLUDED; pass "
                         "--include-unlisted to append them instead)")
    ap.add_argument("--include-unlisted", action="store_true",
                    help="with --method-order, keep methods that aren't listed "
                         "and append them (sorted) after the listed ones")
    ap.add_argument("--method-names", nargs="+", metavar="KEY=LABEL",
                    help="display names for methods, e.g. --method-names "
                         "spatialz=SpatialZ feast=FEAST spatialcpav8_gen='SpatialCPA v8' "
                         "(ordering/colours still use the underlying method id)")
    args = ap.parse_args()
    generate_all(args.input_dir, args.output_dir, args.metrics, args.kind,
                 method_order=args.method_order,
                 method_labels=_parse_method_names(args.method_names),
                 datasets=args.datasets,
                 include_unlisted=args.include_unlisted,
                 tiers=args.tier)


if __name__ == "__main__":
    main()
