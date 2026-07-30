"""Figures for benchmark-pbya-v3 — the SpatialZ paper's validation, drawn.

Four figures, one per thing the paper validated:

``fig_umap_continuity.png``
    Small multiples, one panel per method: a shared UMAP of the real held-out
    cells and that method's reconstructed cells. A faithful reconstruction
    overlays the real cloud; a poor one forms its own island.
``fig_markers_<section>.png``
    Rows = marker genes (Flt1, Pcp4, Cux2), columns = ground truth then each
    method. Spatial expression maps at matched scale.
``fig_spatial_autocorrelation.png``
    Per-gene Moran's I and Geary's C, ground truth (x) vs reconstruction (y),
    one panel per method with the identity line. Points on the diagonal mean the
    reconstruction carries the right amount of spatial structure.
``fig_summary_heatmap.png``
    Methods × headline agreement metrics.

Colors come from the validated reference palette (blue / orange categorical, the
blue sequential ramp for magnitude, blue↔gray↔red diverging for the agreement
heatmap, whose natural midpoint is zero agreement).

Usage:
    python -m src.bench3.plot_paper_figures
    python -m src.bench3.plot_paper_figures --methods spatialz spatialcpav15_gen
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ._v2bridge import align_prediction_to_gt, load_ground_truth, load_prediction
from .config import (
    DATASET_PATH, EMBED_NEIGHBORS, FIGURES_DIR, MARKER_GENES, METHOD_ORDER,
    RANDOM_SEED, RESULTS_DIR, SPATIAL_K,
)
from .evaluate_paper import gearys_c
from benchmark.evaluate_generation import _morans_i, _rank_normalize  # noqa: E402

# ── Validated reference palette ───────────────────────────────────────────────
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # categorical slots 1-3
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
# Diverging red↔gray↔blue: blue is the positive pole (high agreement) because red
# carries warning semantics in the status palette, and here the negative pole is
# genuinely the bad one — anti-correlation with the ground truth.
DIVERGING = ["#8f1f1e", "#e34948", "#f0a3a2", "#f0efec", "#86b6ef", "#256abf", "#0d366b"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": INK_MUTED,
        "text.color": INK, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.grid": False, "font.size": 9, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    cmaps = {
        "seq": LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE),
        "div": LinearSegmentedColormap.from_list("div_bluered", DIVERGING),
    }
    return plt, cmaps


# ── Loading ───────────────────────────────────────────────────────────────────

def find_runs(results_dir=RESULTS_DIR, methods=None, holdout_id=None):
    """Locate ``prediction.h5`` files: {method: {holdout_id: path}}."""
    results_dir = Path(results_dir)
    found = {}
    for p in sorted(results_dir.rglob("prediction.h5")):
        parts = p.relative_to(results_dir).parts
        if len(parts) < 3 or parts[0].startswith("_") or parts[0] == "summary":
            continue
        method, hid = parts[0], parts[-2]
        if methods and method not in methods:
            continue
        if holdout_id and hid != holdout_id:
            continue
        found.setdefault(method, {})[hid] = p
    return found


def _ordered(methods):
    return sorted(methods, key=lambda m: (METHOD_ORDER.index(m)
                                          if m in METHOD_ORDER else 99, m))


def load_section(pred_path, gt_adata, section):
    """Return per-section aligned prediction + GT arrays on a common gene space."""
    pred = load_prediction(str(pred_path))
    gt_sections = gt_adata.obs["section"].values.astype(str)
    gm = gt_sections == section
    pm = pred["section"] == section
    if pm.sum() < 5 or gm.sum() < 5:
        return None

    common = np.intersect1d(pred["gene_names"], gt_adata.var_names.values)
    if len(common) == 0:
        return None
    pgi = np.array([int(np.where(pred["gene_names"] == g)[0][0]) for g in common])
    ggi = np.array([int(np.where(gt_adata.var_names.values == g)[0][0]) for g in common])

    pX = pred["X"][pm][:, pgi]
    pX = pX.toarray() if sp.issparse(pX) else np.asarray(pX)
    gX = gt_adata.X[gm][:, ggi]
    gX = gX.toarray() if sp.issparse(gX) else np.asarray(gX)

    gt_xy = np.asarray(gt_adata.obsm["spatial"])[gm, :2]
    pred_xy = np.column_stack([pred["x"][pm], pred["y"][pm]])
    return {
        "genes": [str(g) for g in common],
        "pred_R": _rank_normalize(pX), "gt_R": _rank_normalize(gX),
        "pred_xy": align_prediction_to_gt(pred_xy, gt_xy, with_scale=True),
        "gt_xy": gt_xy,
    }


# ── Figure 1: UMAP continuity ─────────────────────────────────────────────────

def _embed(pred_R, gt_R, seed=RANDOM_SEED, max_cells=2500):
    from sklearn.decomposition import PCA
    rng = np.random.default_rng(seed)

    def sub(M):
        return M[rng.choice(len(M), max_cells, replace=False)] if len(M) > max_cells else M

    P, G = sub(pred_R), sub(gt_R)
    Z = np.vstack([P, G])
    labels = np.r_[np.zeros(len(P), int), np.ones(len(G), int)]
    pcs = PCA(n_components=int(min(20, Z.shape[1], Z.shape[0] - 1)),
              random_state=seed).fit_transform(Z)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import umap
            emb = umap.UMAP(n_components=2, n_neighbors=EMBED_NEIGHBORS,
                            random_state=seed).fit_transform(pcs)
        return emb, labels, "UMAP"
    except Exception:
        return pcs[:, :2], labels, "PC"


def figure_umap(runs, gt_adata, section, out_path, seed=RANDOM_SEED):
    plt, _ = _mpl()
    methods = _ordered(runs)
    ncol = min(4, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.3 * nrow),
                             squeeze=False)

    basis = "embedding"
    for ax, method in zip(axes.ravel(), methods):
        data = load_section(runs[method], gt_adata, section)
        if data is None:
            ax.set_axis_off()
            ax.set_title(f"{method}\n(no cells)", color=INK_MUTED)
            continue
        emb, labels, basis = _embed(data["pred_R"], data["gt_R"], seed=seed)
        # Real cells behind, reconstructed on top, both semi-transparent so
        # overlap is legible rather than whichever is drawn last winning.
        ax.scatter(*emb[labels == 1].T, s=4, c=SERIES[0], alpha=0.45,
                   linewidths=0, label="real (held out)")
        ax.scatter(*emb[labels == 0].T, s=4, c=SERIES[1], alpha=0.45,
                   linewidths=0, label="reconstructed")
        ax.set_title(method)
        ax.set_xticks([]), ax.set_yticks([])

    for ax in axes.ravel()[len(methods):]:
        ax.set_axis_off()

    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{basis} continuity — real vs reconstructed cells, {section}",
                 y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    if handles:
        fig.legend(handles, lbls, loc="lower center", ncol=2, frameon=False,
                   markerscale=3, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Figure 2: marker-gene spatial maps ────────────────────────────────────────

def dataset_markers(gt_adata):
    """This dataset's marker genes, from the file rather than STARmap's globals."""
    pp = gt_adata.uns.get("paper_protocol")
    raw = pp.get("marker_genes") if pp is not None else None
    genes = [str(g) for g in raw] if raw is not None and len(raw) else []
    return genes or list(MARKER_GENES)


def figure_markers(runs, gt_adata, section, out_path, markers=None):
    plt, cmaps = _mpl()
    methods = _ordered(runs)

    loaded = {m: load_section(runs[m], gt_adata, section) for m in methods}
    loaded = {m: d for m, d in loaded.items() if d is not None}
    if not loaded:
        return None
    genes = next(iter(loaded.values()))["genes"]
    present = [g for g in (markers or MARKER_GENES) if g in genes]
    if not present:
        return None

    ncol = 1 + len(loaded)
    fig, axes = plt.subplots(len(present), ncol,
                             figsize=(2.5 * ncol, 2.6 * len(present)),
                             squeeze=False)
    ref = next(iter(loaded.values()))

    for r, gene in enumerate(present):
        col = genes.index(gene)
        ax = axes[r][0]
        ax.scatter(ref["gt_xy"][:, 0], ref["gt_xy"][:, 1], s=2,
                   c=ref["gt_R"][:, col], cmap=cmaps["seq"], vmin=0, vmax=1,
                   linewidths=0)
        ax.set_title("ground truth" if r == 0 else "")
        ax.set_ylabel(gene, fontsize=11, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_aspect("equal")

        for c, (method, d) in enumerate(loaded.items(), start=1):
            ax = axes[r][c]
            im = ax.scatter(d["pred_xy"][:, 0], d["pred_xy"][:, 1], s=2,
                            c=d["pred_R"][:, col], cmap=cmaps["seq"],
                            vmin=0, vmax=1, linewidths=0)
            if r == 0:
                ax.set_title(method, fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_aspect("equal")

    cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01,
                        ticks=[0, 0.5, 1.0])
    cbar.set_label("expression (per-gene rank)", color=INK_MUTED)
    cbar.outline.set_visible(False)
    fig.suptitle(f"Marker-gene spatial patterns — {section}", y=1.0)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Figure 3: Moran's I / Geary's C ───────────────────────────────────────────

def figure_autocorrelation(runs, gt_adata, section, out_path, k=SPATIAL_K):
    plt, _ = _mpl()
    methods = _ordered(runs)
    fig, axes = plt.subplots(2, len(methods),
                             figsize=(2.6 * len(methods), 5.6), squeeze=False)

    for c, method in enumerate(methods):
        d = load_section(runs[method], gt_adata, section)
        for r, (stat, fn, lo, hi) in enumerate((
                ("Moran's I", _morans_i, -0.1, 1.0),
                ("Geary's C", gearys_c, 0.0, 1.6))):
            ax = axes[r][c]
            if d is None:
                ax.set_axis_off()
                continue
            gp = fn(d["gt_xy"], d["gt_R"], k)
            pp = fn(d["pred_xy"], d["pred_R"], k)
            ok = np.isfinite(gp) & np.isfinite(pp)
            lo = float(min(lo, np.nanmin([gp[ok].min(), pp[ok].min()])))
            hi = float(max(hi, np.nanmax([gp[ok].max(), pp[ok].max()])))
            ax.plot([lo, hi], [lo, hi], color=GRID, lw=1.5, zorder=1)
            # Single series per panel — the panel title carries identity, so no
            # legend box is needed (and no hue is spent on method identity).
            ax.scatter(gp[ok], pp[ok], s=18, c=SERIES[0], alpha=0.8,
                       linewidths=0.5, edgecolors="white", zorder=2)
            ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(method, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{stat} — reconstructed")
            ax.set_xlabel(f"{stat} — real")

    fig.suptitle(f"Spatial autocorrelation per gene — {section} "
                 f"(points on the diagonal = right amount of spatial structure)",
                 y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Figure 4: summary heatmap ─────────────────────────────────────────────────

HEATMAP_METRICS = [
    ("paper_umap_mixing", "UMAP\nmixing"),
    ("paper_morans_pearson", "Moran's I\nagreement"),
    ("paper_gearys_pearson", "Geary's C\nagreement"),
    ("paper_marker_depth_r", "marker\ndepth profile"),
    ("paper_marker_field_r", "marker\nspatial field"),
    ("paper_gene_mean_spearman", "gene mean\nsimilarity"),
    ("paper_celltype_localization", "cell-type\nlocalization"),
]


def figure_summary_heatmap(out_path, results_dir=RESULTS_DIR, methods=None):
    """Methods × headline agreement metrics. All are ↑ better with a natural
    zero (no agreement), so a diverging ramp centered on zero is the honest
    encoding — a negative cell reads as anti-correlated, not merely small."""
    from .aggregate_results import aggregate

    plt, cmaps = _mpl()
    df, _ = aggregate(results_dir)
    if len(df) == 0:
        return None
    if methods:
        df = df[df["method"].isin(methods)]
    cols = [c for c, _ in HEATMAP_METRICS if c in df.columns]
    if not cols:
        return None

    piv = df.groupby("method")[cols].mean()
    piv = piv.reindex([m for m in _ordered(piv.index)])
    labels = [lbl for c, lbl in HEATMAP_METRICS if c in cols]

    fig, ax = plt.subplots(figsize=(1.35 * len(cols) + 3.0,
                                    0.55 * len(piv) + 2.0))
    M = piv.values.astype(float)
    im = ax.imshow(M, cmap=cmaps["div"], vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), labels, fontsize=8)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=9)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(piv)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)   # 2px surface gap
    ax.tick_params(which="minor", length=0)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "NA", ha="center", va="center",
                        color=INK_MUTED, fontsize=8)
                continue
            # Text wears ink tokens; white only where the fill is too dark.
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.62 else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, ticks=[-1, 0, 1])
    cbar.set_label("agreement with ground truth", color=INK_MUTED)
    cbar.outline.set_visible(False)
    ax.set_title("Paper validation metrics (mean over holdouts) — higher is better")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Draw the v3 paper figures")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset", default=str(DATASET_PATH))
    ap.add_argument("--out-dir", default=str(FIGURES_DIR))
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--holdout-id", default="paper_2_4_6")
    ap.add_argument("--sections", nargs="+", default=None,
                    help="held-out sections to draw (default: all in the run)")
    ap.add_argument("--only", nargs="+", default=None,
                    choices=["umap", "markers", "autocorr", "heatmap"])
    args = ap.parse_args()

    runs = find_runs(args.results_dir, methods=args.methods,
                     holdout_id=args.holdout_id)
    runs = {m: paths[args.holdout_id] for m, paths in runs.items()
            if args.holdout_id in paths}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(args.only) if args.only else {"umap", "markers", "autocorr", "heatmap"}

    if not runs:
        print(f"No predictions found for holdout {args.holdout_id!r} under "
              f"{args.results_dir}.")
    else:
        print(f"Methods: {', '.join(_ordered(runs))}")
        sections = args.sections
        if sections is None:
            pred = load_prediction(str(next(iter(runs.values()))))
            sections = [str(s) for s in pred["holdout_sections"]]
        gt = load_ground_truth(args.dataset, sections)

        for section in sections:
            if "umap" in want:
                p = figure_umap(runs, gt, section,
                                out_dir / f"fig_umap_continuity_{section}.png")
                print(f"  wrote {p}")
            if "markers" in want:
                p = figure_markers(runs, gt, section,
                                   out_dir / f"fig_markers_{section}.png",
                                   markers=dataset_markers(gt))
                print(f"  wrote {p}")
            if "autocorr" in want:
                p = figure_autocorrelation(
                    runs, gt, section,
                    out_dir / f"fig_spatial_autocorrelation_{section}.png")
                print(f"  wrote {p}")

    if "heatmap" in want:
        p = figure_summary_heatmap(out_dir / "fig_summary_heatmap.png",
                                   results_dir=args.results_dir,
                                   methods=args.methods)
        print(f"  wrote {p}" if p else "  (no metrics.json — skipped heatmap)")


if __name__ == "__main__":
    main()
