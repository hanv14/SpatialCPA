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

**The density control.** The layout over-produces cells, and a denser point set puts kNN
neighbours closer together, which inflates every graph-based metric. Each arm is therefore scored
twice: as emitted, and subsampled per section to the ground truth's own cell count. The matched
rows are the comparable ones. ``paper_cell_count_ratio`` is 1.0 by construction in the matched
rows and is only meaningful in the raw ones — it is reported from the raw pass and never from the
matched one.

Usage::

    python scripts/t10_rescore_saved.py --model runs/pilot/model_exp_2400.pt
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmark-pbya-v3" / "src"))
from t10_chain_diagnostic import build_embeddings, load_training_volume

SEED = 1
GROUND_TRUTH = "benchmark-pbya-v3/data/processed/starmap_visual_cortex/data.h5ad"
TARGETS = (("section_2", 30.0), ("section_4", 52.0), ("section_6", 74.0))
METRICS = (
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_umap_mixing",
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_celltype_localization",
)


def gt_counts() -> dict[str, int]:
    import anndata as ad

    gt = ad.read_h5ad(GROUND_TRUTH, backed="r")
    try:
        sections = gt.obs["section"].values.astype(str)
        return {s: int((sections == s).sum()) for s, _z in TARGETS}
    finally:
        gt.file.close()


def write_prediction(per_section: dict, gene_names: list[str], path: str) -> None:
    """Emit through the wrappers' own ``_v2_io`` writer, so the evaluator sees a real prediction."""
    v2 = Path(__file__).resolve().parents[1] / "benchmark-pbya-v2" / "src" / "benchmark" / "methods"
    sys.path.insert(0, str(v2))
    import _v2_io

    _v2_io.write_prediction_h5(
        per_section, gene_names, list(per_section), {"seed": SEED}, 0.0, path, "spatialcpav25_gen"
    )


def score(path: str) -> dict:
    from bench3.evaluate_paper import evaluate_paper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_paper(path, GROUND_TRUTH, use_umap=True)


def median_of(result: dict, metric: str) -> float:
    """Median over held-out sections — ``specs/10`` §4.6's estimator, never a mean."""
    key = metric.replace("paper_", "")
    per_section = result.get("per_section", {})
    values = [
        s[key] for s in per_section.values() if isinstance(s, dict) and s.get(key) is not None
    ]
    return float(np.median(values)) if values else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="runs/pilot/model_exp_2400.pt")
    ap.add_argument("--modes", nargs="+", default=["field", "hybrid", "resample"])
    ap.add_argument("--out", default="reports/t10_rescore_exp.md")
    args = ap.parse_args(argv)

    checkpoint = torch.load(args.model, map_location="cpu")
    base = Config(**checkpoint["config"])
    volume = load_training_volume(base)
    model = CTFFlow(
        base, TrainingData.build(volume, base), build_embeddings(base, volume), grf_seed=SEED
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.repulsion = fit_repulsion(volume, base, seed=SEED + 1)
    print(
        f"loaded {args.model}: decoder_mu_link={base.decoder_mu_link}, "
        f"train_steps={base.train_steps}"
    )

    truth = gt_counts()
    rng = np.random.default_rng(0)
    rows = []
    for mode in args.modes:
        cfg = base.replace(layout_mode=mode)
        raw: dict = {}
        matched: dict = {}
        counts_note = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BBoxClampWarning)
            warnings.simplefilter("ignore")
            for name, z in TARGETS:
                emitted = generate_section(model, plane_at_z(volume, z, cfg), volume, cfg, SEED)
                n = int(emitted.n_obs)
                counts_note.append(f"{name}={n}/{truth[name]}")
                x = emitted.X
                x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)
                xyz = np.asarray(emitted.obsm["xyz"], dtype=np.float64)
                ct = np.asarray(emitted.obs["cell_type"].values, dtype=str)
                raw[name] = {"X": sp.csr_matrix(x), "coords": xyz, "cell_type": ct}
                keep = (
                    np.arange(n) if n <= truth[name] else rng.choice(n, truth[name], replace=False)
                )
                matched[name] = {
                    "X": sp.csr_matrix(x[keep]),
                    "coords": xyz[keep],
                    "cell_type": ct[keep],
                }
        genes = list(volume.gene_names)
        out_raw = f"runs/pilot/rescore_{mode}_raw.h5"
        out_matched = f"runs/pilot/rescore_{mode}_matched.h5"
        Path("runs/pilot").mkdir(parents=True, exist_ok=True)
        write_prediction(raw, genes, out_raw)
        write_prediction(matched, genes, out_matched)
        r_raw, r_matched = score(out_raw), score(out_matched)
        rows.append(
            {
                "mode": mode,
                "counts": ", ".join(counts_note),
                "cell_count_ratio_raw": median_of(r_raw, "paper_cell_count_ratio"),
                "raw": {m: median_of(r_raw, m) for m in METRICS},
                "matched": {m: median_of(r_matched, m) for m in METRICS},
            }
        )
        print(f"  {mode}: counts {rows[-1]['counts']}", flush=True)
        for m in METRICS:
            print(
                f"    {m:32s} raw {rows[-1]['raw'][m]:+.4f}   matched "
                f"{rows[-1]['matched'][m]:+.4f}",
                flush=True,
            )

    header = "| metric | " + " | ".join(f"`{r['mode']}`" for r in rows) + " |"
    lines = [
        f"# Re-scored on the pinned instrument — `{Path(args.model).name}`, no refit",
        "",
        f"`decoder_mu_link={base.decoder_mu_link}`, {base.train_steps} steps. `layout_mode` is a",
        "generation-time gate, so all arms share one set of weights.",
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
        lines.append(f"| `{m}` | " + " | ".join(f"{r['matched'][m]:+.4f}" for r in rows) + " |")
    lines.append(
        "| `paper_cell_count_ratio` (raw pass) | "
        + " | ".join(f"{r['cell_count_ratio_raw']:.3f}" for r in rows)
        + " |"
    )
    lines += [
        "",
        "As emitted, without the density control:",
        "",
        header,
        "|---" * (len(rows) + 1) + "|",
    ]
    for m in METRICS:
        lines.append(f"| `{m}` | " + " | ".join(f"{r['raw'][m]:+.4f}" for r in rows) + " |")
    lines += ["", "Emitted cell counts (generated/ground truth):", ""]
    for r in rows:
        lines.append(f"* `{r['mode']}`: {r['counts']}")

    text = "\n".join(lines)
    print()
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
