"""Plot the held-out 2D slice: real vs generated (spatial maps).

For a held-out section, this draws the **real** slice next to the **generated**
one(s) as 2D spatial scatter maps of cells at their (x, y) positions — so you can
eyeball whether a method reproduces the tissue's shape, cell-type layout, and a
gene's spatial pattern, not just the summary numbers.

The predicted cloud is moved onto the real slice's frame with the *same*
orientation-robust alignment the evaluator uses (``leakage_guard`` /
``evaluate.align_prediction_to_gt``), so the panels are directly comparable and
share axis limits. Alignment is evaluation-side only (never fed to the method).

Colouring (``--color``):
* ``celltype`` (default when labels exist) — categorical, one colour per type,
  shared legend; the same colour map across every panel.
* ``gene`` (with ``--gene NAME``) — that gene's log-normalized expression on a
  shared colour scale + colourbar.
* ``density`` — hexbin cell-density map (shared scale).

Two ways to run:
* one prediction ­→ ``Real | Generated`` (``--prediction`` + ``--ground-truth``);
* the whole results tree (``--all``) grouped so each figure is
  ``Real | method A | method B | …`` for one dataset × held-out slice.

Usage
-----
    # one prediction, coloured by cell type
    python -m src.benchmark.plot_slices \
        --prediction results/spatialz/cosmx_nsclc_3d/loo_S3/prediction.h5 \
        --ground-truth <data.h5ad>

    # one prediction, coloured by a gene
    python -m src.benchmark.plot_slices --prediction … --ground-truth … \
        --color gene --gene EPCAM

    # every dataset/holdout: Real vs each method side by side
    python -m src.benchmark.plot_slices --all
    python -m src.benchmark.plot_slices --all --datasets cosmx_nsclc_3d \
        --method-order spatialcpav8_gen spatialz feast isost \
        --method-names spatialcpav8_gen='SpatialCPA v8' spatialz=SpatialZ
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scipy.sparse as sp

from .config import FIGURES_DIR
from .evaluate import load_prediction, load_ground_truth
from .evaluate_disaggregated import _iter_predictions
from .leakage_guard import align_prediction_to_gt
from .plot_disaggregated import (
    NPG_PALETTE, set_nature_style, _order_methods, _parse_method_names,
)


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def _gene_lognorm(X_full, col):
    """log1p( count / cell_total * 1e4 ) for one gene column of a cell x gene X."""
    X_full = X_full.tocsr() if sp.issparse(X_full) else np.asarray(X_full)
    totals = np.asarray(X_full.sum(axis=1)).ravel().astype(float)
    totals[totals == 0] = 1.0
    if sp.issparse(X_full):
        x = np.asarray(X_full[:, col].todense()).ravel().astype(float)
    else:
        x = np.asarray(X_full[:, col]).ravel().astype(float)
    x = np.clip(x, 0.0, None)
    return np.log1p(x / totals * 1e4)


def _subsample(n, max_points, seed=0):
    if max_points and n > max_points:
        return np.random.default_rng(seed).choice(n, max_points, replace=False)
    return np.arange(n)


# --------------------------------------------------------------------------- #
# Build the per-figure data: real slice + one or more generated slices         #
# --------------------------------------------------------------------------- #
def _collect(gt_path, named_predictions, color_by, gene, max_points):
    """Return (sections, panels_by_section, celltypes, gene_range).

    ``named_predictions`` : list of (label, prediction_path).
    ``panels_by_section[sec]`` is a list of dicts (one per column):
        {label, xy, ct, val}  — ct set for celltype mode, val for gene mode.
    Column 0 is always the real (GT) slice.
    """
    # Union of held-out sections across the given predictions.
    preds = [(lbl, load_prediction(p)) for lbl, p in named_predictions]
    holdout = sorted({str(s) for _, pr in preds for s in pr["holdout_sections"]})
    gt = load_ground_truth(gt_path, holdout)
    gt_secs = gt.obs["section"].values.astype(str)
    gt_xy_all = gt.obsm["spatial"][:, :2]
    has_ct = "cell_type" in gt.obs.columns

    # Gene column indices (GT + each prediction), if colouring by gene.
    gt_gcol = None
    pred_gcol = {}
    if color_by == "gene":
        if gene is None:
            raise SystemExit("--color gene requires --gene NAME")
        gmatch = np.where(gt.var_names.values == gene)[0]
        if len(gmatch) == 0:
            raise SystemExit(f"gene '{gene}' not found in ground truth")
        gt_gcol = int(gmatch[0])
        for lbl, pr in preds:
            gm = np.where(pr["gene_names"] == gene)[0]
            pred_gcol[lbl] = int(gm[0]) if len(gm) else None

    celltypes = set()
    gvals_all = []
    panels_by_section = {}
    for sec in holdout:
        gmask = gt_secs == sec
        panels = []

        # column 0: real slice
        g_xy = gt_xy_all[gmask]
        sub = _subsample(len(g_xy), max_points)
        panel = {"label": "Real (held-out)", "xy": g_xy[sub]}
        if color_by == "celltype" and has_ct:
            ct = gt.obs["cell_type"].values[gmask].astype(str)[sub]
            panel["ct"] = ct
            celltypes |= set(ct.tolist())
        elif color_by == "gene":
            v = _gene_lognorm(gt.X[gmask], gt_gcol)[sub]
            panel["val"] = v
            gvals_all.append(v)
        panels.append(panel)

        # columns 1..: generated slices (aligned to the real frame)
        for lbl, pr in preds:
            pmask = pr["section"] == sec
            if pmask.sum() == 0:
                panels.append({"label": lbl, "xy": np.empty((0, 2))})
                continue
            p_xy = np.column_stack([pr["x"][pmask], pr["y"][pmask]])
            if gmask.sum() >= 3 and pmask.sum() >= 3:
                p_xy = align_prediction_to_gt(p_xy, g_xy, with_scale=True)
            sub = _subsample(len(p_xy), max_points)
            panel = {"label": lbl, "xy": p_xy[sub]}
            if color_by == "celltype":
                ct = pr["cell_type"][pmask].astype(str)[sub]
                panel["ct"] = ct
                celltypes |= set(ct.tolist())
            elif color_by == "gene":
                col = pred_gcol.get(lbl)
                if col is not None:
                    v = _gene_lognorm(pr["X"][pmask], col)[sub]
                    panel["val"] = v
                    gvals_all.append(v)
            panels.append(panel)
        panels_by_section[sec] = panels

    gene_range = None
    if gvals_all:
        allv = np.concatenate(gvals_all)
        if allv.size:
            gene_range = (float(np.percentile(allv, 1)), float(np.percentile(allv, 99)))
    return holdout, panels_by_section, sorted(celltypes), gene_range


# --------------------------------------------------------------------------- #
# Draw one figure (rows = sections, cols = real + methods)                     #
# --------------------------------------------------------------------------- #
def _draw_figure(sections, panels_by_section, color_by, ct_colors, gene_range,
                 gene, title, output_path):
    nrows = len(sections)
    ncols = max(len(panels_by_section[s]) for s in sections)
    if nrows == 0 or ncols == 0:
        return False

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(2.2 * ncols, 2.35 * nrows))
    mappable = None
    for r, sec in enumerate(sections):
        panels = panels_by_section[sec]
        # shared, equal-aspect limits across the row so shapes are comparable
        pts = [p["xy"] for p in panels if len(p["xy"])]
        if pts:
            allxy = np.vstack(pts)
            pad_x = 0.03 * (np.ptp(allxy[:, 0]) or 1)
            pad_y = 0.03 * (np.ptp(allxy[:, 1]) or 1)
            xlim = (allxy[:, 0].min() - pad_x, allxy[:, 0].max() + pad_x)
            ylim = (allxy[:, 1].min() - pad_y, allxy[:, 1].max() + pad_y)
        else:
            xlim = ylim = (0, 1)

        for c in range(ncols):
            ax = axes[r][c]
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            if c >= len(panels):
                ax.axis("off")
                continue
            p = panels[c]
            xy = p["xy"]
            n = len(xy)
            s = float(np.clip(4000.0 / max(n, 1), 0.5, 8.0))
            if n:
                if color_by == "celltype" and "ct" in p:
                    for t in sorted(set(p["ct"].tolist())):
                        m = p["ct"] == t
                        ax.scatter(xy[m, 0], xy[m, 1], s=s,
                                   color=ct_colors.get(t, "#999999"),
                                   linewidths=0, rasterized=True)
                elif color_by == "gene" and "val" in p:
                    vmin, vmax = gene_range if gene_range else (None, None)
                    mappable = ax.scatter(xy[:, 0], xy[:, 1], s=s, c=p["val"],
                                          cmap="viridis", vmin=vmin, vmax=vmax,
                                          linewidths=0, rasterized=True)
                elif color_by == "density":
                    hb = ax.hexbin(xy[:, 0], xy[:, 1], gridsize=40,
                                   cmap="magma", mincnt=1, linewidths=0)
                    mappable = hb
                else:
                    ax.scatter(xy[:, 0], xy[:, 1], s=s, color="#3C5488",
                               linewidths=0, rasterized=True)
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(p["label"])
            if c == 0:
                ax.set_ylabel(f"section {sec}", rotation=90, labelpad=6)

    # Reserve a right margin for the legend/colourbar, lay out the grid, THEN
    # add the shared artist there (adding it before tight_layout would make
    # tight_layout complain about an incompatible manual axes).
    want_legend = color_by == "celltype" and bool(ct_colors)
    want_cbar = color_by in ("gene", "density") and mappable is not None
    right = 0.86 if want_legend else (0.88 if want_cbar else 0.98)

    if title:
        fig.suptitle(title, y=1.0, fontsize=9)
    fig.tight_layout(rect=(0, 0, right, 0.97))

    if want_legend:
        handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=4,
                          markerfacecolor=col, markeredgecolor="none", label=t)
                   for t, col in ct_colors.items()]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.87, 0.5),
                   title="cell type", title_fontsize=6, ncol=1)
    elif want_cbar:
        cax = fig.add_axes([0.9, 0.25, 0.015, 0.5])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label(f"{gene} (log-norm)" if color_by == "gene" else "cell count",
                     fontsize=6)
        cb.ax.tick_params(labelsize=5)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_path.with_suffix(f".{ext}"), dpi=200)
    plt.close(fig)
    return True


def _celltype_colors(celltypes):
    return {t: NPG_PALETTE[i % len(NPG_PALETTE)] for i, t in enumerate(celltypes)}


# --------------------------------------------------------------------------- #
# Public entry points                                                          #
# --------------------------------------------------------------------------- #
def plot_one(prediction_path, gt_path, color_by="celltype", gene=None,
             output_dir=None, max_points=20000):
    """Real | Generated for a single prediction (one figure per section grid)."""
    set_nature_style()
    pred = load_prediction(prediction_path)
    method = pred["method_name"]
    sections, panels, celltypes, grange = _collect(
        gt_path, [(method, prediction_path)], color_by, gene, max_points)
    if not sections:
        print("No held-out sections found.")
        return
    out_dir = Path(output_dir) if output_dir else Path(prediction_path).parent
    stem = f"slice_{method}" + (f"_{gene}" if color_by == "gene" else f"_{color_by}")
    ok = _draw_figure(sections, panels, color_by, _celltype_colors(celltypes),
                      grange, gene, title=f"{method} — real vs generated",
                      output_path=out_dir / stem)
    if ok:
        print(f"  wrote {out_dir / stem}.png/.pdf")


def plot_all(results_dir=None, datasets=None, method_order=None, method_labels=None,
             color_by="celltype", gene=None, output_dir=None, max_points=20000):
    """One figure per dataset x holdout: Real | method A | method B | …."""
    from .config import RESULTS_DIR
    set_nature_style()
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    output_dir = Path(output_dir) if output_dir else (FIGURES_DIR / "slices")
    method_labels = method_labels or {}

    # group predictions by (dataset, holdout_id)
    groups = {}
    for pred_path, gt_path, _out_dir, ids in _iter_predictions(
            results_dir, datasets=datasets):
        key = (ids["dataset"], ids["holdout_id"])
        groups.setdefault(key, {"gt": gt_path, "preds": {}})
        groups[key]["preds"][ids["method"]] = str(pred_path)
    if not groups:
        print(f"No predictions found under {results_dir}.")
        return

    made = 0
    for (dataset, holdout_id), g in sorted(groups.items()):
        methods = _order_methods(g["preds"].keys(), method_order)
        named = [(method_labels.get(m, m), g["preds"][m]) for m in methods]
        try:
            sections, panels, celltypes, grange = _collect(
                g["gt"], named, color_by, gene, max_points)
        except SystemExit as e:
            print(f"  skip {dataset}/{holdout_id}: {e}")
            continue
        if not sections:
            continue
        safe_ds = dataset.replace("/", "__")
        stem = (f"slice_{safe_ds}_{holdout_id}"
                + (f"_{gene}" if color_by == "gene" else f"_{color_by}"))
        ok = _draw_figure(sections, panels, color_by,
                          _celltype_colors(celltypes), grange, gene,
                          title=f"{dataset} · {holdout_id} — real vs generated",
                          output_path=output_dir / stem)
        if ok:
            print(f"  wrote {output_dir / stem}.png/.pdf")
            made += 1
    print(f"\n{made} slice figure(s) written to {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot the held-out 2D slice: real vs generated "
                    "(aligned spatial maps; Springer Nature theme)")
    ap.add_argument("--prediction", help="single prediction.h5")
    ap.add_argument("--ground-truth", help="dataset data.h5ad (single mode)")
    ap.add_argument("--all", action="store_true",
                    help="walk the results tree; Real vs each method per dataset/holdout")
    ap.add_argument("--results-dir")
    ap.add_argument("--datasets", nargs="+", help="(--all) restrict to these datasets")
    ap.add_argument("--color", choices=["celltype", "gene", "density"],
                    default="celltype", help="how to colour the cells")
    ap.add_argument("--gene", help="gene name (required for --color gene)")
    ap.add_argument("--output-dir")
    ap.add_argument("--max-points", type=int, default=20000,
                    help="subsample cells per panel for plotting (0 = no cap)")
    ap.add_argument("--method-order", nargs="+", metavar="METHOD",
                    help="(--all) left-to-right method order")
    ap.add_argument("--method-names", nargs="+", metavar="KEY=LABEL",
                    help="(--all) display names, e.g. spatialz=SpatialZ")
    args = ap.parse_args()

    if args.all:
        plot_all(results_dir=args.results_dir, datasets=args.datasets,
                 method_order=args.method_order,
                 method_labels=_parse_method_names(args.method_names),
                 color_by=args.color, gene=args.gene, output_dir=args.output_dir,
                 max_points=args.max_points)
        return

    if not args.prediction or not args.ground_truth:
        ap.error("single mode needs --prediction and --ground-truth (or use --all)")
    plot_one(args.prediction, args.ground_truth, color_by=args.color,
             gene=args.gene, output_dir=args.output_dir, max_points=args.max_points)


if __name__ == "__main__":
    main()
