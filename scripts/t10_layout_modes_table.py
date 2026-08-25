"""R11's layout-mode table: the arms, the copy floor and the oracle ceiling, on one instrument.

``t10_rescore_saved.py`` measures **one arm per process** so the three ``layout_mode``s can run
concurrently. This collects their JSON outputs, scores bench3's two model-free probes for the
floor and the ceiling, and writes the table in the shape ``reports/pilot.md`` §13 used — so the
grid-sampler numbers line up column for column with the biased-sampler ones they replace.

The probes come from ``bench3.selftest --probes oracle flanking_copy --keep <dir>``, which writes
each probe's ``prediction.h5`` through the same writer every method wrapper uses. They are
method-free and sampler-free, so they are the same referents whatever the layout does; they are
re-scored here rather than quoted from the pilot because a referent measured on another machine
is not a referent.

Usage::

    python scripts/t10_layout_modes_table.py \\
        --arms reports/r11_field-grid.json reports/r11_hybrid-grid.json ... \\
        --probes runs/pilot/probes --bench3 /path/to/benchmark-pbya-v3
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve

TARGETS = ("section_2", "section_4", "section_6")
HEADLINE = ("paper_celltype_localization", "paper_cell_count_ratio")
SIX = (
    "paper_celltype_localization",
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_gene_mean_spearman",
)


def _fmt(v: float | None, prec: int = 4, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"


def score_probe(path: Path, ground_truth: Path, use_umap: bool) -> dict:
    from bench3.evaluate_paper import evaluate_paper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_paper(str(path), str(ground_truth), use_umap=use_umap)


def per_section(result: dict, metric: str) -> dict[str, float | None]:
    key = metric.replace("paper_", "")
    ps = result.get("per_section", {})
    out: dict[str, float | None] = {}
    for sid in TARGETS:
        sec = ps.get(sid)
        v = sec.get(key) if isinstance(sec, dict) else None
        out[sid] = None if v is None else float(v)
    return out


def median(values: dict[str, float | None]) -> float:
    vals = [v for v in values.values() if v is not None]
    return float(np.median(vals)) if vals else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", required=True, help="the per-arm JSONs to collect")
    ap.add_argument("--probes", default=None, help="selftest --keep dir (oracle, flanking_copy)")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--out", default="reports/r11_starmap_layout_modes.md")
    add_path_args(ap)
    args = ap.parse_args(argv)
    paths = resolve(args, need_input=False)

    arms: list[dict] = []
    for p in args.arms:
        loaded = json.loads(Path(p).read_text())
        arms.extend(loaded if isinstance(loaded, list) else [loaded])

    referents: dict[str, dict] = {}
    if args.probes:
        for probe in ("flanking_copy", "oracle"):
            pred = Path(args.probes) / probe / "prediction.h5"
            if not pred.exists():
                raise SystemExit(
                    f"{probe}: no prediction at {pred}. Produce the referents with\n"
                    f"  python -m bench3.selftest --dataset starmap_visual_cortex "
                    f"--probes oracle flanking_copy --keep {args.probes}"
                )
            r = score_probe(pred, paths.ground_truth, not args.no_umap)
            referents[probe] = {m: per_section(r, m) for m in (*SIX, "paper_cell_count_ratio")}
            print(f"scored {probe}", flush=True)

    seeds = {a["seed"] for a in arms}
    models = {a["model"] for a in arms}
    lines = [
        "# R11 re-measured — `layout_mode` on STARmap tier 1, grid sampler",
        "",
        f"One set of weights (`{sorted(models)[0]}`), "
        f"`decoder_mu_link={arms[0]['decoder_mu_link']}`, {arms[0]['train_steps']} steps, "
        f"seed {sorted(seeds)[0]}. `layout_mode` and `layout_sampler` are generation-time gates",
        "(`CTFFlow.check_generation_cfg`), so no arm below needed a fit and nothing but the",
        "layout varies between them. Same instrument, same three held-out sections and the same",
        "ground-truth-matched density as `reports/pilot.md` §13, so the columns are comparable.",
        "",
        "`celltype_localization` is scored at ground-truth-matched density — each section",
        "subsampled to its own true cell count, because a denser point set puts kNN neighbours",
        "closer and inflates every graph-based metric. `cell_count_ratio` is from the raw pass,",
        "where it means something. Medians are over the three held-out sections",
        "(`specs/10` §4.6), never means.",
        "",
        "## Headline — the two metrics R11 turns on",
        "",
        "| arm | " + " | ".join(TARGETS) + " | **median** |",
        "|---" * (len(TARGETS) + 2) + "|",
    ]

    def referent_rows(metric: str, prec: int = 4, sign: bool = True) -> list[str]:
        out = []
        for probe, label in (
            ("oracle", "`oracle` — ceiling"),
            ("flanking_copy", "`flanking_copy` — copy floor"),
        ):
            if probe in referents:
                ps = referents[probe][metric]
                out.append(
                    f"| {label} | "
                    + " | ".join(_fmt(ps[s], prec, sign) for s in TARGETS)
                    + f" | **{_fmt(median(ps), prec, sign)}** |"
                )
        return out

    lines.append("| **celltype_localization** | | | | |")
    lines.extend(referent_rows("paper_celltype_localization"))
    for a in arms:
        ps = a["matched_per_section"]["paper_celltype_localization"]
        lines.append(
            f"| `layout_mode={a['mode']}` (`{a['layout_sampler']}`) | "
            + " | ".join(_fmt(ps[s]) for s in TARGETS)
            + f" | **{_fmt(a['matched']['paper_celltype_localization'])}** |"
        )
    lines.append("| **cell_count_ratio** (raw pass) | | | | |")
    lines.extend(referent_rows("paper_cell_count_ratio", 3, False))
    for a in arms:
        ps = a["cell_count_ratio_per_section"]
        lines.append(
            f"| `layout_mode={a['mode']}` (`{a['layout_sampler']}`) | "
            + " | ".join(_fmt(ps[s], 3, False) for s in TARGETS)
            + f" | **{_fmt(a['cell_count_ratio_raw'], 3, False)}** |"
        )

    lines += [
        "",
        "Emitted cell counts (generated/ground truth):",
        "",
    ]
    for a in arms:
        lines.append(
            f"* `{a['arm']}`: "
            + ", ".join(f"{s}={_fmt(a['n_pred'][s], 0, False)}/{a['n_gt'][s]}" for s in TARGETS)
        )

    lines += [
        "",
        "## The six target metrics, medians over sections",
        "",
        "| metric | "
        + " | ".join(f"`{a['arm']}`" for a in arms)
        + " | `flanking_copy` | `oracle` |",
        "|---" * (len(arms) + 3) + "|",
    ]
    for m in SIX:
        row = [f"| `{m}` "] + [f"| {_fmt(a['matched'][m])} " for a in arms]
        for probe in ("flanking_copy", "oracle"):
            row.append(f"| {_fmt(median(referents[probe][m])) if probe in referents else '—'} ")
        lines.append("".join(row) + "|")
    ratio_cells = [f"| {_fmt(a['cell_count_ratio_raw'], 3, False)} " for a in arms]
    for probe in ("flanking_copy", "oracle"):
        v = median(referents[probe]["paper_cell_count_ratio"]) if probe in referents else None
        ratio_cells.append(f"| {_fmt(v, 3, False)} ")
    lines.append("| `paper_cell_count_ratio` (raw) " + "".join(ratio_cells) + "|")

    lines += [
        "",
        "## What varies between the arms",
        "",
        "| arm | `layout_mode` | `layout_sampler` | weights | seed |",
        "|---|---|---|---|---|",
    ]
    for a in arms:
        lines.append(
            f"| `{a['arm']}` | {a['mode']} | {a['layout_sampler']} | "
            f"`{Path(a['model']).name}` | {a['seed']} |"
        )
    lines += [
        "",
        "One seed. `reports/envelope_synthetic.md` measured the across-seed envelope at "
        "**0.0335** on the",
        "synthetic fixture, and `specs/10` §3's repeated-seed rule wants "
        "`claim_min_seeds` = 3 before a",
        "claim rests on any of this: a difference below that envelope is not a difference.",
    ]

    text = "\n".join(lines)
    print()
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(
        json.dumps({"arms": arms, "referents": referents}, indent=2)
    )
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
