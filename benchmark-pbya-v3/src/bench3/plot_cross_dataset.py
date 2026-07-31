"""Cross-dataset figures for benchmark-pbya-v3, in the Nature/Springer theme.

``plot_paper_figures`` draws one dataset at a time — UMAPs, marker maps and
autocorrelation scatters live inside a single volume's ground truth. This module
draws the other view: **every method across every dataset**, which is the figure
a paper actually reports.

Three figures, from the aggregated ``metrics.json`` tree:

``fig_cross_dataset_bars``
    One panel per metric; within a panel, datasets along x and a bar per method.
    Each held-out section's own value is overlaid as a dot, so a bar that rests
    on a wide spread is visibly different from one that does not.
``fig_cross_dataset_heatmap``
    One panel per metric, methods x datasets, on a diverging ramp centred at
    zero — the compact overview, and the panel that carries the numbers.
``fig_cross_dataset_ranks``
    Each method's composite rank within each dataset. Ranks are *within* a
    dataset by construction (see ``rank_methods``); this draws them side by side
    to show whether the ordering survives a change of volume, and deliberately
    reports no average across datasets.

Every figure writes the numbers it drew to a CSV beside itself. That is useful
on its own, and it is also what discharges the palette's contrast obligation
(two of the seven hues sit below 3:1 on white — see ``nature_theme``).

**STARmap is the published protocol's own dataset; the others are analogues.**
The figures mark it in bold and never pool it with the rest, following
``config.DATASET_SPECS['kind']``.

Usage:
    python -m src.bench3.plot_cross_dataset

    python -m src.bench3.plot_cross_dataset \\
      --method-order spatialz feast isost spatialcpav15_gen \\
      --method-names spatialz=SpatialZ feast=FEAST isost=isoST \\
                     spatialcpav15_gen="SpatialCPA v15" \\
      --dataset-names starmap_visual_cortex="STARmap V1" \\
                      exseq_visual_cortex="ExSeq V1" \\
                      cosmx_nsclc_3d="CosMx NSCLC"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import nature_theme as nt
from .aggregate_results import aggregate
from .config import DATASET_SPECS, METHOD_ORDER, RESULTS_DIR, SUMMARY_DIR
from .rank_methods import rank as rank_one_dataset

# ── Metrics ───────────────────────────────────────────────────────────────────
# (column, axis label, +1 if higher is better). The default set is the headline
# panel: one metric from each of the paper's four validation criteria, plus the
# two that most often separate methods. All are ``higher is better`` and share a
# real zero, which is what lets one diverging ramp serve every heatmap panel.
METRIC_SPECS = {
    "paper_umap_mixing":           ("UMAP mixing", +1),
    "paper_embedding_mixing_pca":  ("PCA mixing", +1),
    "paper_morans_pearson":        ("Moran's $I$ agreement", +1),
    "paper_gearys_pearson":        ("Geary's $C$ agreement", +1),
    "paper_marker_depth_r":        ("Marker depth profile", +1),
    "paper_marker_field_r":        ("Marker spatial field", +1),
    "paper_celltype_localization": ("Cell-type localization", +1),
    "paper_gene_mean_spearman":    ("Gene-mean similarity", +1),
    "paper_gene_var_spearman":     ("Gene-variance similarity", +1),
    "paper_morans_mae":            ("Moran's $I$ MAE", -1),
    "paper_gearys_mae":            ("Geary's $C$ MAE", -1),
    "paper_marker_morans_mae":     ("Marker Moran's $I$ MAE", -1),
    "paper_celltype_ot":           ("Cell-type OT divergence", -1),
    "paper_cell_count_ratio":      ("Cell-count ratio", 0),   # diagnostic, not scored
}

DEFAULT_METRICS = [
    "paper_umap_mixing",
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_marker_depth_r",
    "paper_marker_field_r",
    "paper_celltype_localization",
]


def metric_label(metric, overrides=None):
    """Display label for a metric, with the better-direction arrow appended."""
    if overrides and metric in overrides:
        return overrides[metric]
    label, direction = METRIC_SPECS.get(metric, (metric, 0))
    arrow = {1: " ↑", -1: " ↓"}.get(direction, "")
    return f"{label}{arrow}"


# ── --method-order / --method-names / --dataset-names ─────────────────────────

def parse_labeled(values):
    """``['a', 'b=Long name']`` -> ``(['a', 'b'], {'b': 'Long name'})``.

    Split on the FIRST ``=`` only, so a label may itself contain one.
    """
    keys, labels = [], {}
    for raw in values or []:
        text = str(raw)
        key, sep, label = text.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"cannot parse {text!r}: expected NAME or NAME=Label")
        keys.append(key)
        if sep and label.strip():
            labels[key] = label.strip()
    return keys, labels


def resolve_axis(what, present, order_arg, names_arg, fallback_order=()):
    """Decide which keys to plot, in what order, under what labels.

    One rule for both axes, and for both flags: **the keys of whichever flag you
    pass select and order the axis.** So ``--dataset-names a=A b=B`` plots
    exactly a and b, in that order, labelled A and B — the single flag does both
    jobs, which is why no separate ``--dataset-order`` is needed for the common
    case. When both flags are given, the ``--*-order`` one decides membership and
    order and ``--*-names`` supplies labels only. An inline label on the order
    flag wins over the names flag.

    With neither flag, everything found in the results tree is plotted, in
    ``fallback_order`` where that applies (so published baselines keep coming
    before the SpatialCPA variants) and alphabetically after.
    """
    keys_o, labels_o = parse_labeled(order_arg)
    keys_n, labels_n = parse_labeled(names_arg)
    labels = {**labels_n, **labels_o}

    if keys_o:
        chosen = keys_o
    elif keys_n:
        chosen = keys_n
    else:
        known = [k for k in fallback_order if k in present]
        chosen = known + sorted(k for k in present if k not in set(fallback_order))

    missing = [k for k in chosen if k not in present]
    if missing:
        raise SystemExit(
            f"no results for {what} {missing}.\n"
            f"  present in the results tree: {sorted(present)}\n"
            f"  (names must match the results directory exactly; label them with "
            f"NAME=Label)")
    if not chosen:
        raise SystemExit(f"no {what} to plot")

    seen, ordered = set(), []
    for k in chosen:                       # de-duplicate, keep first position
        if k not in seen:
            seen.add(k)
            ordered.append(k)
    return ordered, labels


def display(key, labels):
    return labels.get(key, key)


def is_paper_dataset(name):
    """True for the volume the published protocol was defined on."""
    return DATASET_SPECS.get(name, {}).get("kind") == "paper"


# ── Data ──────────────────────────────────────────────────────────────────────

def load_frames(results_dir, design=None):
    """``(per-run, per-section)`` frames, optionally restricted to one design."""
    df, sdf = aggregate(results_dir)
    if len(df) == 0:
        raise SystemExit(
            f"no metrics.json found under {results_dir}. Run "
            f"`python -m src.bench3.run_all` (or `evaluate_all`) first.")
    if design:
        for frame_name, frame in (("run", df), ("section", sdf)):
            if len(frame):
                keep = frame["holdout_id"].astype(str).str.startswith(design)
                if frame_name == "run":
                    df = frame[keep]
                else:
                    sdf = frame[keep]
        if len(df) == 0:
            raise SystemExit(f"no runs for design={design!r}")
    return df, sdf


def cell_table(df, metrics, methods, datasets):
    """``{(dataset, method, metric): value}`` — the reported, pooled number.

    The bar height is the value in ``metrics.json``, which ``evaluate_paper``
    pools over held-out sections weighted by ground-truth cell count. It is NOT
    the unweighted mean of the per-section dots drawn on top of it, and the two
    differ when sections differ in size — the dots show spread, the bar is the
    quantity the tables quote.
    """
    cols = [m for m in metrics if m in df.columns]
    if not cols:
        raise SystemExit(
            f"none of the requested metrics are present: {metrics}\n"
            f"  available: {sorted(c for c in df.columns if c.startswith('paper_'))}")
    sub = df[df["method"].isin(methods) & df["dataset"].isin(datasets)]
    grouped = sub.groupby(["dataset", "method"])[cols].mean()
    out = {}
    for (ds, me), row in grouped.iterrows():
        for metric in cols:
            v = row[metric]
            out[(ds, me, metric)] = float(v) if pd.notna(v) else np.nan
    return out, cols


def section_points(sdf, metric, dataset, method):
    """Per-held-out-section values behind one bar (may be empty)."""
    if sdf is None or len(sdf) == 0 or metric not in sdf.columns:
        return np.array([])
    m = ((sdf["dataset"] == dataset) & (sdf["method"] == method))
    vals = pd.to_numeric(sdf.loc[m, metric], errors="coerce").to_numpy(dtype=float)
    return vals[np.isfinite(vals)]


def _limits(values, direction):
    """Axis limits that always include zero and never crop a bar."""
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(vals):
        return (0.0, 1.0)
    lo = min(0.0, float(vals.min()))
    hi = max(0.0, float(vals.max()))
    if hi <= lo:
        hi = lo + 1.0
    pad = 0.06 * (hi - lo)
    # Correlation-like metrics are capped at 1; leave a little air but do not
    # imply headroom that cannot exist.
    hi = min(hi + pad, 1.0) if hi <= 1.0 and direction >= 0 else hi + pad
    return (lo - pad if lo < 0 else 0.0, hi)


# ── Figure 1: grouped bars ────────────────────────────────────────────────────

def figure_bars(table, sdf, metrics, methods, datasets, method_labels,
                dataset_labels, metric_labels, out_stem, width_mm,
                panel_height_mm=30.0, formats=("pdf", "png")):
    plt = nt.activate()

    n_rows, n_ds, n_me = len(metrics), len(datasets), len(methods)

    # Reserve every margin explicitly — there is no tight bounding box to rescue
    # a collision, and the saved width has to stay exactly ``width_mm``.
    left_mm, right_mm, top_mm = 20.0, 3.0, 5.0
    tick_labels = [display(d, dataset_labels) for d in datasets]
    plot_w_mm = width_mm - left_mm - right_mm
    rot, ha, xlab_mm = nt.tick_layout(tick_labels, plot_w_mm / n_ds)

    legend_ncol = min(n_me, 4)
    legend_rows = int(np.ceil(n_me / legend_ncol))
    legend_mm = legend_rows * nt.text_height_mm(nt.FONT_TICK) + 3.0
    footnote_mm = 2 * nt.text_height_mm(nt.FONT_TICK - 0.5) + 2.0
    bottom_mm = xlab_mm + legend_mm + footnote_mm + 2.0

    height = panel_height_mm * n_rows + top_mm + bottom_mm
    fig, axes = plt.subplots(n_rows, 1, squeeze=False,
                             figsize=nt.figsize(width_mm, height))
    axes = axes.ravel()
    fig.subplots_adjust(left=left_mm / width_mm, right=1 - right_mm / width_mm,
                        top=1 - top_mm / height, bottom=bottom_mm / height,
                        hspace=0.28)

    x = np.arange(n_ds, dtype=float)
    group_w = 0.82
    bar_w = group_w / n_me

    for r, (metric, ax) in enumerate(zip(metrics, axes)):
        direction = METRIC_SPECS.get(metric, (metric, 0))[1]
        seen_values = []

        for i, method in enumerate(methods):
            offs = -group_w / 2 + bar_w * (i + 0.5)
            heights = [table.get((ds, method, metric), np.nan) for ds in datasets]
            seen_values.extend(h for h in heights if np.isfinite(h))
            # A hairline white edge is the print equivalent of the 2 px surface
            # gap: adjacent fills never touch, so a group reads as bars rather
            # than as one striped block.
            ax.bar(x + offs, heights, width=bar_w * 0.92,
                   color=nt.series_color(i), edgecolor=nt.SURFACE,
                   linewidth=0.3, zorder=2,
                   label=display(method, method_labels) if r == 0 else None)

            for j, ds in enumerate(datasets):
                pts = section_points(sdf, metric, ds, method)
                seen_values.extend(pts.tolist())
                if not len(pts):
                    continue
                # Per-section dots, in ink rather than the series colour: they
                # are a spread annotation on the bar, not a second series.
                ax.scatter(np.full(len(pts), x[j] + offs), pts,
                           s=1.6, c=nt.INK, linewidths=0.25,
                           edgecolors=nt.SURFACE, zorder=3, clip_on=False)

        lo, hi = _limits(seen_values, direction)
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5, n_ds - 0.5)
        if lo < 0:
            ax.axhline(0.0, color=nt.AXIS, linewidth=nt.LINE_HAIRLINE, zorder=1)
        ax.set_ylabel(metric_label(metric, metric_labels))
        nt.panel_letter(ax, "abcdefghijklmnop"[r])

        ax.set_xticks(x)
        if r == n_rows - 1:
            ax.set_xticklabels(tick_labels, rotation=rot, ha=ha,
                               rotation_mode="anchor" if rot else None)
            for tick, ds in zip(ax.get_xticklabels(), datasets):
                if is_paper_dataset(ds):
                    tick.set_fontweight("bold")
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)

    handles, labels = axes[0].get_legend_handles_labels()
    handles, labels = nt.legend_row_major(handles, labels, legend_ncol)
    # Anchored from the top of its own band, so the footnote below it has a
    # known amount of room and the two cannot land on each other.
    fig.legend(handles, labels, loc="upper center", ncol=legend_ncol,
               bbox_to_anchor=(0.5, (footnote_mm + legend_mm) / height))
    fig.align_ylabels(axes)
    fig.text(0.5, 0.6 * footnote_mm / height,
             "Bold dataset = the published protocol's own volume; the rest are "
             "protocol analogues.\nDots are individual held-out sections; the bar "
             "is the cell-count-weighted value reported in metrics.json.",
             ha="center", va="center", linespacing=1.4,
             fontsize=nt.FONT_TICK - 0.5, color=nt.INK_MUTED)
    written = nt.save(fig, out_stem, formats=formats)
    plt.close(fig)
    return written


# ── Figure 2: heatmap ─────────────────────────────────────────────────────────

def figure_heatmap(table, metrics, methods, datasets, method_labels,
                   dataset_labels, metric_labels, out_stem, width_mm,
                   ncol=3, scale=None, formats=("pdf", "png")):
    plt = nt.activate()

    cmap, vmin, vmax = nt.value_ramp(list(table.values()), force=scale)

    n = len(metrics)
    ncol = max(1, min(ncol, n))
    nrow = int(np.ceil(n / ncol))

    # Every panel carries its own row of method names on the left and its own
    # rotated dataset names underneath, so the gaps between panels are not
    # decoration — they are exactly the space those labels occupy. Solve for
    # them in millimetres, then convert to the fractions subplots_adjust wants.
    ylab_mm = max(nt.text_width_mm(display(m, method_labels), nt.FONT_TICK)
                  for m in methods) + 4.0
    x_labels = [display(d, dataset_labels) for d in datasets]
    letter_w_mm, right_mm = 4.0, 3.0

    # The gutter on the left of every panel holds that panel's method names AND
    # its bold letter, so it is reserved once and used for both the outer margin
    # and the inter-column gap.
    left_mm = gap_w_mm = ylab_mm + letter_w_mm
    panel_w_mm = (width_mm - left_mm - right_mm - (ncol - 1) * gap_w_mm) / ncol
    if panel_w_mm <= 5.0:
        raise SystemExit(
            f"{ncol} heatmap columns do not fit in {width_mm:.0f} mm once the "
            f"method names are allowed for. Use --heatmap-cols {max(1, ncol - 1)}, "
            f"shorten labels with --method-names, or widen with --width.")
    _, _, xlab_mm = nt.tick_layout(x_labels, panel_w_mm / len(datasets))

    # A centred title wider than its panel overhangs into the neighbour's
    # gutter and lands on the neighbour's letter, so wrap it to the block it
    # owns rather than letting it run.
    # A title is centred on its panel, and the next panel's letter sits only
    # ~2 mm past this panel's right edge, so the honest symmetric budget is the
    # panel plus a couple of millimetres either side — not the whole gutter.
    titles = {m: nt.wrap_to_mm(metric_label(m, metric_labels),
                               panel_w_mm + 4.0, nt.FONT_BASE)
              for m in metrics}
    title_lines = max(t.count("\n") + 1 for t in titles.values())
    title_mm = title_lines * nt.text_height_mm(nt.FONT_BASE) + 1.5
    letter_mm = nt.text_height_mm(nt.FONT_PANEL_LETTER) + 1.0
    top_mm = title_mm + letter_mm

    cell_mm = 4.6
    panel_h_mm = cell_mm * len(methods)
    gap_h_mm = xlab_mm + title_mm + letter_mm
    cbar_mm = 14.0
    bottom_mm = xlab_mm + cbar_mm
    height = top_mm + nrow * panel_h_mm + (nrow - 1) * gap_h_mm + bottom_mm

    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=nt.figsize(width_mm, height))
    fig.subplots_adjust(left=left_mm / width_mm, right=1 - right_mm / width_mm,
                        top=1 - top_mm / height, bottom=bottom_mm / height,
                        hspace=gap_h_mm / panel_h_mm,
                        wspace=gap_w_mm / panel_w_mm)

    im = None
    for idx, metric in enumerate(metrics):
        ax = axes[idx // ncol][idx % ncol]
        M = np.array([[table.get((ds, me, metric), np.nan) for ds in datasets]
                      for me in methods], dtype=float)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto")

        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right",
                           rotation_mode="anchor")
        for tick, ds in zip(ax.get_xticklabels(), datasets):
            if is_paper_dataset(ds):
                tick.set_fontweight("bold")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([display(m, method_labels) for m in methods])
        ax.set_title(titles[metric], pad=3, linespacing=1.25)
        nt.panel_letter(ax, "abcdefghijklmnop"[idx],
                        dx=-(ylab_mm + 2.0) / panel_w_mm,
                        dy=1.0 + (title_mm + 0.5) / panel_h_mm)

        # A 2 px surface gap between cells, drawn as white minor gridlines.
        ax.set_xticks(np.arange(-0.5, len(datasets)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(methods)), minor=True)
        ax.grid(which="minor", color=nt.SURFACE, linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                if not np.isfinite(v):
                    ax.text(j, i, "n/a", ha="center", va="center",
                            fontsize=nt.FONT_TICK - 1, color=nt.INK_MUTED)
                    continue
                # Text wears ink, except where the fill is too dark to sit on.
                dark = (v - vmin) / (vmax - vmin) > 0.72
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=nt.FONT_TICK - 1,
                        color=nt.SURFACE if dark else nt.INK)

    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].set_axis_off()

    if im is not None:
        cax = fig.add_axes([0.34, 9.0 / height, 0.32, 1.6 / height])
        ticks = [vmin, (vmin + vmax) / 2, vmax]
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
        cbar.set_label("Agreement with ground truth", color=nt.INK)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=2, width=nt.LINE_HAIRLINE)
    written = nt.save(fig, out_stem, formats=formats)
    plt.close(fig)
    return written


# ── Figure 3: rank profile ────────────────────────────────────────────────────

def rank_profile(df, methods, datasets, include_gen=False):
    """Composite rank per (dataset, method), computed *within* each dataset.

    Ranks are recomputed among exactly the methods being plotted — a rank is a
    position in a field, so restricting the field changes it, and silently
    reusing a seven-method rank on a four-method figure would be wrong.
    """
    rows = {}
    for ds in datasets:
        sub = df[(df["dataset"] == ds) & (df["method"].isin(methods))]
        if not len(sub):
            continue
        ranked = rank_one_dataset(sub, include_gen=include_gen)
        if not len(ranked) or "composite_rank" not in ranked.columns:
            continue
        for _, r in ranked.iterrows():
            rows[(ds, str(r["method"]))] = float(r["composite_rank"])
    return rows


def figure_ranks(ranks, methods, datasets, method_labels, dataset_labels,
                 out_stem, width_mm, panel_height_mm=52.0, legend=False,
                 formats=("pdf", "png")):
    """Rank profile with the series named at the end of each line.

    Direct labels rather than a legend: every line ends in its own name, so
    identity never rests on hue — which is what the palette needs past four
    slots, and it spares the reader the round trip to a key. ``legend=True``
    adds one back for a figure that will be reproduced very small.
    """
    plt = nt.activate()

    # The end-of-line labels live outside the axes, so their width is part of
    # the layout rather than something a tight bounding box absorbs afterwards.
    label_mm = max(nt.text_width_mm(display(m, method_labels), nt.FONT_TICK)
                   for m in methods) + 3.0
    left_mm, top_mm = 16.0, 4.0
    x_labels = [display(d, dataset_labels) for d in datasets]
    plot_w_mm = width_mm - left_mm - label_mm
    rot, ha, xlab_mm = nt.tick_layout(x_labels, plot_w_mm / len(datasets))

    legend_ncol = min(len(methods), 4)
    legend_mm = (int(np.ceil(len(methods) / legend_ncol))
                 * nt.text_height_mm(nt.FONT_TICK) + 4.0) if legend else 0.0
    footnote_mm = 2 * nt.text_height_mm(nt.FONT_TICK - 0.5) + 2.0
    bottom_mm = xlab_mm + legend_mm + footnote_mm + 2.0
    height_mm = panel_height_mm + top_mm + bottom_mm

    fig, ax = plt.subplots(figsize=nt.figsize(width_mm, height_mm))
    fig.subplots_adjust(left=left_mm / width_mm,
                        right=1 - label_mm / width_mm,
                        top=1 - top_mm / height_mm,
                        bottom=bottom_mm / height_mm)
    x = np.arange(len(datasets), dtype=float)

    for i, method in enumerate(methods):
        y = np.array([ranks.get((ds, method), np.nan) for ds in datasets])
        ok = np.isfinite(y)
        if not ok.any():
            continue
        # Past four slots the palette no longer separates every pair on an
        # all-pairs form, so identity also rides on marker shape and on the
        # direct label at the right edge — never on hue alone.
        ax.plot(x[ok], y[ok], color=nt.series_color(i), linewidth=1.0,
                marker=nt.series_marker(i), markersize=3.2,
                markeredgecolor=nt.SURFACE, markeredgewidth=0.4,
                label=display(method, method_labels), zorder=2, clip_on=False)
        last = np.flatnonzero(ok)[-1]
        ax.annotate(display(method, method_labels),
                    xy=(x[last], y[last]), xytext=(4, 0),
                    textcoords="offset points", va="center", ha="left",
                    fontsize=nt.FONT_TICK, color=nt.INK)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=rot, ha=ha,
                       rotation_mode="anchor" if rot else None)
    for tick, ds in zip(ax.get_xticklabels(), datasets):
        if is_paper_dataset(ds):
            tick.set_fontweight("bold")

    finite = [v for v in ranks.values() if np.isfinite(v)]
    top = min(finite) if finite else 1.0
    bottom = max(finite) if finite else float(len(methods))
    ax.set_ylim(bottom + 0.4, top - 0.4)          # inverted: rank 1 on top
    ax.set_yticks([t for t in range(1, len(methods) + 1)
                   if top - 0.4 <= t <= bottom + 0.4])
    ax.set_ylabel("Composite rank within dataset (1 = best)")
    ax.set_xlim(-0.25, len(datasets) - 0.75)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = nt.legend_row_major(handles, labels, legend_ncol)
        fig.legend(handles, labels, loc="upper center", ncol=legend_ncol,
                   bbox_to_anchor=(0.5, (footnote_mm + legend_mm) / height_mm))

    fig.text(0.5, 0.6 * footnote_mm / height_mm,
             "Ranks are computed within each dataset and are never averaged "
             "across them:\na composite is a position among the methods run on "
             "that volume.",
             ha="center", va="center", linespacing=1.4,
             fontsize=nt.FONT_TICK - 0.5, color=nt.INK_MUTED)
    written = nt.save(fig, out_stem, formats=formats)
    plt.close(fig)
    return written


# ── The table view ────────────────────────────────────────────────────────────

def write_table(table, metrics, methods, datasets, method_labels,
                dataset_labels, out_csv):
    """Write exactly the numbers the figures drew.

    Not a convenience: two palette hues sit below 3:1 on white, and the rule for
    that is relief — a visible label or a table view. This is the table view,
    and it also makes a figure auditable against the source metrics.
    """
    rows = []
    for ds in datasets:
        for me in methods:
            row = {
                "dataset": ds,
                "dataset_label": display(ds, dataset_labels),
                "kind": DATASET_SPECS.get(ds, {}).get("kind", "unknown"),
                "method": me,
                "method_label": display(me, method_labels),
            }
            for metric in metrics:
                row[metric] = table.get((ds, me, metric), np.nan)
            rows.append(row)
    out = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Cross-dataset figures for v3, in the Nature/Springer theme",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Selection and labelling:\n"
            "  --method-order / --dataset-order  select AND order an axis\n"
            "  --method-names / --dataset-names  label it — and, when the\n"
            "                                    matching *-order flag is absent,\n"
            "                                    select and order it too\n"
            "  Either flag takes NAME or NAME=Label:\n"
            "    --method-order spatialz feast\n"
            "    --method-names spatialz=SpatialZ 'spatialcpav15_gen=SpatialCPA v15'\n"
            "    --dataset-names starmap_visual_cortex='STARmap V1'\n"))
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--out-dir", default=str(Path(SUMMARY_DIR) / "figures" / "cross_dataset"))
    ap.add_argument("--design", default="paper", choices=["paper", "loo", "all"],
                    help="restrict to holdouts of this design (default: paper)")

    ap.add_argument("--method-order", nargs="+", default=None, metavar="NAME[=Label]",
                    help="methods to plot, in order")
    ap.add_argument("--method-names", nargs="+", default=None, metavar="NAME=Label",
                    help="method labels; also selects/orders if --method-order is absent")
    ap.add_argument("--dataset-order", nargs="+", default=None, metavar="NAME[=Label]",
                    help="datasets to plot, in order")
    ap.add_argument("--dataset-names", nargs="+", default=None, metavar="NAME=Label",
                    help="dataset labels; also selects/orders if --dataset-order is absent")

    ap.add_argument("--metrics", nargs="+", default=None,
                    help=f"metric columns to plot (default: {' '.join(DEFAULT_METRICS)})")
    ap.add_argument("--metric-names", nargs="+", default=None, metavar="METRIC=Label",
                    help="override a metric's axis label")

    ap.add_argument("--width", type=float, default=nt.WIDTH_DOUBLE_MM,
                    metavar="MM",
                    help=f"figure width in mm (single column {nt.WIDTH_SINGLE_MM:.0f}, "
                         f"1.5-column {nt.WIDTH_HALF_MM:.0f}, double "
                         f"{nt.WIDTH_DOUBLE_MM:.0f}; default: double)")
    ap.add_argument("--formats", nargs="+", default=["pdf", "png"],
                    help="output formats (default: pdf png)")
    ap.add_argument("--heatmap-cols", type=int, default=3,
                    help="panels per row in the heatmap figure")
    ap.add_argument("--scale", default=None, choices=["sequential", "diverging"],
                    help="heatmap ramp (default: diverging on -1..1 when any "
                         "value is negative, else sequential on 0..1)")
    ap.add_argument("--rank-legend", action="store_true",
                    help="add a legend to the rank figure (lines are already "
                         "labelled at their right-hand end)")
    ap.add_argument("--include-gen", action="store_true",
                    help="add v2's generation metrics as a fifth ranking criterion")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["bars", "heatmap", "ranks"],
                    help="draw only these figures")
    args = ap.parse_args()

    design = None if args.design == "all" else args.design
    df, sdf = load_frames(args.results_dir, design=design)

    methods, method_labels = resolve_axis(
        "method", set(df["method"].astype(str)),
        args.method_order, args.method_names, fallback_order=METHOD_ORDER)
    datasets, dataset_labels = resolve_axis(
        "dataset", set(df["dataset"].astype(str)),
        args.dataset_order, args.dataset_names,
        fallback_order=list(DATASET_SPECS))
    _, metric_labels = parse_labeled(args.metric_names)

    if len(methods) > len(nt.PALETTE):
        raise SystemExit(
            f"{len(methods)} methods requested but the validated palette has "
            f"{len(nt.PALETTE)} hues, and hues are never cycled — two methods "
            f"would share a colour. Narrow --method-order, or split the figure.")

    requested = args.metrics or DEFAULT_METRICS
    table, metrics = cell_table(df, requested, methods, datasets)
    dropped = [m for m in requested if m not in metrics]
    if dropped:
        print(f"  note: not in the results, skipped: {', '.join(dropped)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"bars", "heatmap", "ranks"}
    fmts = tuple(args.formats)

    print(f"benchmark-pbya-v3 — cross-dataset figures (design={args.design})")
    print(f"  methods  ({len(methods)}): "
          f"{', '.join(display(m, method_labels) for m in methods)}")
    print(f"  datasets ({len(datasets)}): "
          f"{', '.join(display(d, dataset_labels) for d in datasets)}")
    print(f"  metrics  ({len(metrics)}): {', '.join(metrics)}")
    print(f"  width: {args.width:.0f} mm -> {out_dir}")

    csv = write_table(table, metrics, methods, datasets, method_labels,
                      dataset_labels, out_dir / "cross_dataset_values.csv")
    print(f"  wrote {csv}")

    if "bars" in want:
        for p in figure_bars(table, sdf, metrics, methods, datasets,
                             method_labels, dataset_labels, metric_labels,
                             out_dir / "fig_cross_dataset_bars", args.width,
                             formats=fmts):
            print(f"  wrote {p}")
    if "heatmap" in want:
        for p in figure_heatmap(table, metrics, methods, datasets,
                                method_labels, dataset_labels, metric_labels,
                                out_dir / "fig_cross_dataset_heatmap",
                                args.width, ncol=args.heatmap_cols,
                                scale=args.scale, formats=fmts):
            print(f"  wrote {p}")
    if "ranks" in want:
        ranks = rank_profile(df, methods, datasets, include_gen=args.include_gen)
        if not ranks:
            print("  (no rankable metrics — skipped the rank figure)")
        else:
            for p in figure_ranks(ranks, methods, datasets, method_labels,
                                  dataset_labels,
                                  out_dir / "fig_cross_dataset_ranks",
                                  args.width, legend=args.rank_legend,
                                  formats=fmts):
                print(f"  wrote {p}")


if __name__ == "__main__":
    main()
