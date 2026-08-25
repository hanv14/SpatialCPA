"""Re-score a saved model on the pinned bench3 instrument — no refit.

Two questions, one pass over an already-fitted model:

* **T10 item 2** — the ``exp``-link arm's full six target metrics, at ground-truth-matched cell
  density, so ``reports/pilot.md``'s smoke table (softplus, 1200 steps, uncalibrated) can be
  replaced by a current one. ``paper_marker_field_r`` in particular: it was 0.1611 against a
  0.8857 copy floor in the smoke run, it was v25's single loss at T09, and it is the pooled loss
  for v20 and v21 against SpatialZ.
* **R11 / item 1** — ``layout_mode`` field vs hybrid vs resample on real STARmap. ``layout_mode``
  is a **generation-time** gate (``CTFFlow.check_generation_cfg`` lets it differ from the model's),
  so all three come from the same saved weights and none needs a fit.

``layout_sampler`` is a generation-time gate for the same reason — the fit never draws
positions — so ``--layout-sampler rejection`` re-measures an arm on the **biased** sampler
(``reports/r11_envelope.md``) from the very same weights. That is the control that separates *the
sampler changed* from *the weights are not the pilot's*: every layout number recorded before the
grid sampler existed came from a checkpoint that no longer exists, so a bare comparison against
``reports/pilot.md`` §13 confounds the two.

**The density control.** The layout over-produces cells, and a denser point set puts kNN
neighbours closer together, which inflates every graph-based metric. Each arm is therefore scored
twice: as emitted, and subsampled per section to the ground truth's own cell count. The matched
rows are the comparable ones. ``paper_cell_count_ratio`` is 1.0 by construction in the matched
rows and is only meaningful in the raw ones — it is reported from the raw pass and never from the
matched one.

One arm per process. The arms are independent and each is single-threaded under
``OMP_NUM_THREADS=1``, so the three ``layout_mode``s run concurrently rather than one after
another; nothing is shared but the read-only input and the saved weights.

Usage::

    python scripts/t10_rescore_saved.py --model runs/pilot/model_exp_2400.pt --modes field
    python scripts/t10_rescore_saved.py --model ... --modes field --layout-sampler rejection
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.generate import generate_section, plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads

TARGETS = (("section_2", 30.0), ("section_4", 52.0), ("section_6", 74.0))

METRICS = (
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_umap_mixing",
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_celltype_localization",
    "paper_gene_mean_spearman",
)


def gt_counts(ground_truth: Path) -> dict[str, int]:
    import anndata as ad

    gt = ad.read_h5ad(ground_truth, backed="r")
    try:
        sections = gt.obs["section"].values.astype(str)
        return {s: int((sections == s).sum()) for s, _z in TARGETS}
    finally:
        gt.file.close()


def write_prediction(per_section: dict, gene_names: list[str], path: str, seed: int) -> None:
    """Emit through the wrappers' own ``_v2_io`` writer, so the evaluator sees a real prediction."""
    import _v2_io

    _v2_io.write_prediction_h5(
        per_section, gene_names, list(per_section), {"seed": seed}, 0.0, path, "spatialcpav25_gen"
    )


def score(path: str, ground_truth: Path, use_umap: bool) -> dict:
    from bench3.evaluate_paper import evaluate_paper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_paper(path, str(ground_truth), use_umap=use_umap)


def median_of(result: dict, metric: str) -> float:
    """Median over held-out sections — ``specs/10`` §4.6's estimator, never a mean."""
    values = [v for v in per_section_of(result, metric).values() if v is not None]
    return float(np.median(values)) if values else float("nan")


def per_section_of(result: dict, metric: str) -> dict[str, float | None]:
    """``{section_id: value}`` for one metric, in ``TARGETS`` order.

    The per-section split is not a refinement here. ``reports/pilot.md`` §3 measured the
    model-free copy floor scoring ``section_2`` worst on 6 of 6 metrics — the stack's first
    section has flanking evidence on one side only — and R11 records ``cell_count_ratio``
    spanning 500x across these three sections, where the median picks the one sane value. A
    pooled layout number that does not carry its sections misdescribes the layout.
    """
    key = metric.replace("paper_", "")
    ps = result.get("per_section", {})
    out: dict[str, float | None] = {}
    for sid, _z in TARGETS:
        sec = ps.get(sid)
        v = sec.get(key) if isinstance(sec, dict) else None
        out[sid] = None if v is None else float(v)
    return out


def _fmt(v: float | None, prec: int = 4, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/pilot/model_exp_2400.pt")
    ap.add_argument("--modes", nargs="+", default=["field", "hybrid", "resample"])
    ap.add_argument(
        "--layout-sampler",
        default=None,
        choices=["grid", "rejection"],
        help="override Config.layout_sampler. Default: the saved config's (grid, unless the "
        "checkpoint predates the field, in which case Config's own default applies).",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None, help="markdown report (default: reports/r11_<arm>.md)")
    ap.add_argument(
        "--workdir", default="runs/pilot", help="where the arm's prediction .h5 files go"
    )
    ap.add_argument(
        "--no-umap",
        action="store_true",
        help="skip paper_umap_mixing (needs umap-learn). The other six metrics are unaffected: "
        "use_umap gates embedding_continuity and nothing else.",
    )
    ap.add_argument(
        "--score-only",
        action="store_true",
        help="re-score the predictions already in --workdir instead of regenerating them. Use "
        "this after adding a metric to METRICS: generation of the field arm's tens of "
        "thousands of cells costs ~10 min and the predictions do not change.",
    )
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="resolve every path, import the evaluator and the prediction writer, read the "
        "checkpoint's config and the ground truth's cell counts, then exit. Costs seconds and "
        "catches a wrong --bench3 before an hour of fitting does.",
    )
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args, need_input=not args.score_only)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    from t10_chain_diagnostic import build_embeddings, load_training_volume

    truth = gt_counts(paths.ground_truth)
    print(f"  ground-truth cells: {truth}")

    if args.preflight:
        import _v2_io  # noqa: F401
        from bench3.evaluate_paper import evaluate_paper  # noqa: F401

        ckpt = Path(args.model)
        if ckpt.exists():
            saved = Config(**torch.load(ckpt, map_location="cpu")["config"])
            print(
                f"  checkpoint {ckpt}: decoder_mu_link={saved.decoder_mu_link}, "
                f"train_steps={saved.train_steps}, layout_sampler={saved.layout_sampler}, "
                f"text_emb_mode={saved.text_emb_mode}, expr_pca_dim={saved.expr_pca_dim}"
            )
        else:
            print(f"  checkpoint {ckpt}: NOT YET PRESENT (fit it first)")
        print("preflight OK")
        return 0

    checkpoint = torch.load(args.model, map_location="cpu")
    base = Config(**checkpoint["config"])
    if args.layout_sampler is not None:
        base = base.replace(layout_sampler=args.layout_sampler)
    volume = load_training_volume(base, paths.input)
    model = CTFFlow(
        base, TrainingData.build(volume, base), build_embeddings(base, volume), grf_seed=args.seed
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.repulsion = fit_repulsion(volume, base, seed=args.seed + 1)
    print(
        f"loaded {args.model}: decoder_mu_link={base.decoder_mu_link}, "
        f"train_steps={base.train_steps}, layout_sampler={base.layout_sampler}"
    )

    rng = np.random.default_rng(0)
    workdir = Path(args.workdir)
    rows = []
    for mode in args.modes:
        arm = f"{mode}-{base.layout_sampler}"
        out_raw = str(workdir / f"rescore_{arm}_raw.h5")
        out_matched = str(workdir / f"rescore_{arm}_matched.h5")
        if not args.score_only:
            cfg = base.replace(layout_mode=mode)
            raw: dict = {}
            matched: dict = {}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BBoxClampWarning)
                warnings.simplefilter("ignore")
                for name, z in TARGETS:
                    emitted = generate_section(
                        model, plane_at_z(volume, z, cfg), volume, cfg, args.seed
                    )
                    n = int(emitted.n_obs)
                    print(f"  {arm} {name}: {n} cells / {truth[name]} ground truth", flush=True)
                    x = emitted.X
                    x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)
                    xyz = np.asarray(emitted.obsm["xyz"], dtype=np.float64)
                    ct = np.asarray(emitted.obs["cell_type"].values, dtype=str)
                    raw[name] = {"X": sp.csr_matrix(x), "coords": xyz, "cell_type": ct}
                    keep = (
                        np.arange(n)
                        if n <= truth[name]
                        else rng.choice(n, truth[name], replace=False)
                    )
                    matched[name] = {
                        "X": sp.csr_matrix(x[keep]),
                        "coords": xyz[keep],
                        "cell_type": ct[keep],
                    }
            workdir.mkdir(parents=True, exist_ok=True)
            genes = list(volume.gene_names)
            write_prediction(raw, genes, out_raw, args.seed)
            write_prediction(matched, genes, out_matched, args.seed)

        r_raw = score(out_raw, paths.ground_truth, not args.no_umap)
        r_matched = score(out_matched, paths.ground_truth, not args.no_umap)
        # Emitted counts come back from the scorer rather than from a constant: a hard-coded
        # count is a claim about a run that has since been replaced, and R11 already carries one.
        n_pred = per_section_of(r_raw, "paper_n_pred_cells")
        rows.append(
            {
                "arm": arm,
                "mode": mode,
                "layout_sampler": base.layout_sampler,
                "model": str(args.model),
                "decoder_mu_link": base.decoder_mu_link,
                "train_steps": int(base.train_steps),
                "seed": int(args.seed),
                "n_pred": n_pred,
                "n_gt": {s: truth[s] for s, _z in TARGETS},
                "cell_count_ratio_raw": median_of(r_raw, "paper_cell_count_ratio"),
                "cell_count_ratio_per_section": per_section_of(r_raw, "paper_cell_count_ratio"),
                "raw": {m: median_of(r_raw, m) for m in METRICS},
                "matched": {m: median_of(r_matched, m) for m in METRICS},
                "matched_per_section": {m: per_section_of(r_matched, m) for m in METRICS},
                "raw_per_section": {m: per_section_of(r_raw, m) for m in METRICS},
                "pooled_weighted_mean_matched": {m: r_matched.get(m) for m in METRICS},
            }
        )
        counts = ", ".join(f"{s}={_fmt(n_pred[s], 0, False)}/{truth[s]}" for s, _z in TARGETS)
        print(f"  {arm}: counts {counts}", flush=True)
        for m in METRICS:
            print(
                f"    {m:<32} raw {_fmt(rows[-1]['raw'][m])}   "
                f"matched {_fmt(rows[-1]['matched'][m])}",
                flush=True,
            )

    header = "| metric | " + " | ".join(f"`{r['arm']}`" for r in rows) + " |"
    lines = [
        f"# Re-scored on the pinned instrument — `{Path(args.model).name}`, no refit",
        "",
        f"`decoder_mu_link={rows[0]['decoder_mu_link']}`, {rows[0]['train_steps']} steps, "
        f"`layout_sampler={rows[0]['layout_sampler']}`, seed {args.seed}. `layout_mode` and",
        "`layout_sampler` are both generation-time gates, so every arm shares one set of weights.",
        "",
        "**Ground-truth-matched density** — each section subsampled to its own ground-truth cell",
        "count, because a denser point set puts kNN neighbours closer and inflates every",
        "graph-based metric. `paper_cell_count_ratio` is 1.0 by construction here and is reported",
        "from the raw pass instead.",
        "",
        header,
        "|---" * (len(rows) + 1) + "|",
    ]
    for m in METRICS:
        lines.append(f"| `{m}` | " + " | ".join(_fmt(r["matched"][m]) for r in rows) + " |")
    lines.append(
        "| `paper_cell_count_ratio` (raw pass) | "
        + " | ".join(_fmt(r["cell_count_ratio_raw"], 3, False) for r in rows)
        + " |"
    )
    lines += [
        "",
        "Per section, on the two metrics R11 turns on — `celltype_localization` matched,",
        "`cell_count_ratio` raw:",
        "",
        "| arm | " + " | ".join(s for s, _z in TARGETS) + " | median |",
        "|---" * (len(TARGETS) + 2) + "|",
    ]
    for r in rows:
        loc = r["matched_per_section"]["paper_celltype_localization"]
        lines.append(
            f"| `{r['arm']}` localization | "
            + " | ".join(_fmt(loc[s]) for s, _z in TARGETS)
            + f" | {_fmt(r['matched']['paper_celltype_localization'])} |"
        )
        ratio = r["cell_count_ratio_per_section"]
        lines.append(
            f"| `{r['arm']}` count ratio | "
            + " | ".join(_fmt(ratio[s], 3, False) for s, _z in TARGETS)
            + f" | {_fmt(r['cell_count_ratio_raw'], 3, False)} |"
        )
    lines += [
        "",
        "As emitted, without the density control:",
        "",
        header,
        "|---" * (len(rows) + 1) + "|",
    ]
    for m in METRICS:
        lines.append(f"| `{m}` | " + " | ".join(_fmt(r["raw"][m]) for r in rows) + " |")
    lines += ["", "Emitted cell counts (generated/ground truth):", ""]
    for r in rows:
        lines.append(
            f"* `{r['arm']}`: "
            + ", ".join(f"{s}={_fmt(r['n_pred'][s], 0, False)}/{r['n_gt'][s]}" for s, _z in TARGETS)
        )

    text = "\n".join(lines)
    print()
    print(text)
    out = Path(args.out or f"reports/r11_{'_'.join(r['arm'] for r in rows)}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
