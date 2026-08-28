"""Is the ceiling headroom stable enough to carry a strategic claim? — a bootstrap.

`reports/t09_depth_ceiling_*.md` reports one number per section: how much a *perfect noiseless*
method could beat the best available copy of another real section. The two datasets came out an
order of magnitude apart — tier-1 STARmap median **+0.175** (5.2x the 0.0335 envelope) against
`deep_starmap`'s **+0.016** (0.5x) — and that gap is the whole basis for calling tier-1 the
informative reconstruction benchmark and `deep_starmap` saturated.

**That claim rests on a reliability estimated from 28 genes at ~4 100 cells per section**, and a
point estimate at that scale is not self-evidently stable. This resamples it.

Each replicate perturbs the two things the estimate depends on:

* **cells** — an independent subsample (without replacement, ``--cell-frac``) of the target's and
  the donor's cells, so the profile carries a different draw of the same tissue;
* **marker genes** — a resample *with replacement* of the metric's own selected marker set, since
  ``marker_depth_r`` is a mean over those genes and the mean's uncertainty is dominated by which
  genes are in it.

and recomputes the whole chain: split-half reliability R (Spearman-Brown corrected), the noiseless
ceiling ``sqrt(R)``, the copy correlation, and ``headroom = sqrt(R) - copy``.

Two copy referents, both reported, because they answer different questions:

``operational``
    the section ``_resample_layout`` actually picks — nearest by ``|dz|``, ties broken by
    ``section_id``. What `cross-mix` is really competing with.
``best``
    the best donor available. The stronger, oracle-copier bound the headroom claim was stated
    against.

**Two biases, both stated because both cut against this file's own conclusion.** R is estimated by
splitting the *subsample*, so Spearman-Brown corrects to the subsample's size rather than the
section's — R, and therefore the ceiling and the headroom, come out slightly **low**. And the gene
resample captures the spread of the mean over the *selected* markers, not the variance of the
selection itself, which on `deep_starmap` (32 chosen from 1017) is real and is **not** included;
on tier-1 there is no selection at all — all 28 genes are markers — so tier-1's interval is the
more complete of the two. Both biases understate tier-1's advantage rather than manufacture it.

Usage::

    python scripts/t09_ceiling_bootstrap.py --dataset starmap_visual_cortex \\
        --bench3 /path/to/benchmark-pbya-v3 --out reports/t09_ceiling_bootstrap_starmap.md
    python scripts/t09_ceiling_bootstrap.py --dataset deep_starmap ... \\
        --compare reports/t09_ceiling_bootstrap_starmap.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.data.schema import section_seed
from spatialcpav25_gen.losses.metric_aware import knn_weight_graph, marker_genes, profile_axis
from spatialcpav25_gen.train.select import _normalised, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import base_config, clamp_config_to_input, load_training_volume
from t09_depth_ceiling import profile, ruler, score

ENVELOPE = 0.0335


def spearman_brown(r: float) -> float:
    """Lift a half-vs-half correlation to the reliability of a full-size measurement."""
    return 2.0 * r / (1.0 + r) if r > -1.0 else r


def replicate(target, donor, markers, axis, cfg, rng, cell_frac: float):
    """One bootstrap draw: ``(sqrt(R), copy_r)`` under resampled cells and resampled genes."""
    genes = markers[torch.from_numpy(rng.integers(0, len(markers), size=len(markers)))]

    t_counts = np.asarray(target.counts.todense(), dtype=np.float64)
    t_xy = np.asarray(target.coords, dtype=np.float64)
    keep = rng.choice(
        t_counts.shape[0], size=max(4, int(cell_frac * t_counts.shape[0])), replace=False
    )
    t_counts, t_xy = t_counts[keep], t_xy[keep]
    bounds, sigma = ruler(target, axis, cfg)
    p_target = profile(t_counts, t_xy, target.z, axis, genes, cfg, bounds, sigma)

    order = rng.permutation(t_counts.shape[0])
    a, b = order[: order.size // 2], order[order.size // 2 :]
    half = score(
        profile(t_counts[a], t_xy[a], target.z, axis, genes, cfg, bounds, sigma),
        profile(t_counts[b], t_xy[b], target.z, axis, genes, cfg, bounds, sigma),
    )
    reliability = spearman_brown(half)

    d_counts = np.asarray(donor.counts.todense(), dtype=np.float64)
    d_xy = np.asarray(donor.coords, dtype=np.float64)
    dkeep = rng.choice(
        d_counts.shape[0], size=max(4, int(cell_frac * d_counts.shape[0])), replace=False
    )
    copy_r = score(
        profile(d_counts[dkeep], d_xy[dkeep], target.z, axis, genes, cfg, bounds, sigma), p_target
    )
    return (float(np.sqrt(reliability)) if reliability > 0 else float("nan")), copy_r


def pick_donors(target, volume):
    """``(operational, best_candidates)`` — the section resample picks, and all other sections."""
    others = [s for s in volume.sections if s.section_id != target.section_id]
    operational = min(others, key=lambda s: (abs(float(s.z) - float(target.z)), s.section_id))
    return operational, others


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--replicates", type=int, default=400)
    ap.add_argument("--cell-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--compare", default=None, help="another run's .json, for the overlap test")
    ap.add_argument("--out", default="reports/t09_ceiling_bootstrap.md")
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    overrides: dict = {}
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    cfg = clamp_config_to_input(base_config(args.seed, **overrides), paths.input)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    axis = profile_axis(volume, cfg)
    targets = selection_folds(volume, cfg)
    print(
        f"  {volume.n_cells} cells x {volume.n_genes} genes; targets "
        f"{[s.section_id for s in targets]}; {args.replicates} replicates at "
        f"{100 * args.cell_frac:.0f}% of cells, marker genes resampled with replacement\n"
    )

    rows, draws_op, draws_best = [], [], []
    for target in targets:
        counts = np.asarray(target.counts.todense(), dtype=np.float64)
        markers = marker_genes(
            _normalised(counts, cfg),
            knn_weight_graph(np.asarray(target.coords, dtype=np.float64), cfg),
            cfg,
        )
        operational, others = pick_donors(target, volume)
        # `section_seed`, not `hash()`: the builtin is salted per process, so two runs
        # with the same --seed would not be comparable (Convention 3).
        rng = np.random.default_rng([int(args.seed), section_seed(target.section_id)])
        head_op, head_best = [], []
        for _ in range(int(args.replicates)):
            state = rng.bit_generator.state
            ceiling, r_op = replicate(target, operational, markers, axis, cfg, rng, args.cell_frac)
            best = r_op
            for donor in others:
                if donor.section_id == operational.section_id:
                    continue
                rng.bit_generator.state = state  # same cells and genes for every donor
                _, r = replicate(target, donor, markers, axis, cfg, rng, args.cell_frac)
                best = max(best, r)
            if ceiling == ceiling:
                head_op.append(ceiling - r_op)
                head_best.append(ceiling - best)
        draws_op.append(head_op)
        draws_best.append(head_best)
        q = lambda v: (
            float(np.percentile(v, 2.5)),
            float(np.median(v)),
            float(np.percentile(v, 97.5)),
        )
        lo_o, md_o, hi_o = q(head_op)
        lo_b, md_b, hi_b = q(head_best)
        rows.append(
            {
                "target": target.section_id,
                "dataset": paths.dataset,
                "holdout": paths.holdout,
                "n_cells": int(counts.shape[0]),
                "n_markers": len(markers),
                "operational_donor": operational.section_id,
                "n_replicates": len(head_op),
                "headroom_operational": {"p2.5": lo_o, "median": md_o, "p97.5": hi_o},
                "headroom_best": {"p2.5": lo_b, "median": md_b, "p97.5": hi_b},
            }
        )
        print(
            f"  {target.section_id}: headroom vs the operational copy "
            f"{md_o:+.4f} [{lo_o:+.4f}, {hi_o:+.4f}] | vs the best copy "
            f"{md_b:+.4f} [{lo_b:+.4f}, {hi_b:+.4f}]",
            flush=True,
        )

    # One dataset-level draw per replicate: the median over targets, so the interval below is
    # the interval on the number the strategic claim actually quotes.
    n = min(len(d) for d in draws_best)
    level_best = np.median(np.array([d[:n] for d in draws_best]), axis=0)
    level_op = np.median(np.array([d[:n] for d in draws_op]), axis=0)
    summary = {
        "dataset": paths.dataset,
        "holdout": paths.holdout,
        "replicates": int(n),
        "cell_frac": float(args.cell_frac),
        "seed": int(args.seed),
        "median_headroom_best": [
            float(np.percentile(level_best, 2.5)),
            float(np.median(level_best)),
            float(np.percentile(level_best, 97.5)),
        ],
        "median_headroom_operational": [
            float(np.percentile(level_op, 2.5)),
            float(np.median(level_op)),
            float(np.percentile(level_op, 97.5)),
        ],
        "draws_best": [float(v) for v in level_best],
    }
    lo, md, hi = summary["median_headroom_best"]
    print(
        f"\n  dataset-level median headroom over the best copy: {md:+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}] = {md / ENVELOPE:.1f}x the {ENVELOPE} envelope "
        f"[{lo / ENVELOPE:.1f}x, {hi / ENVELOPE:.1f}x]"
    )

    overlap = None
    if args.compare:
        other = json.loads(Path(args.compare).read_text())["summary"]
        a = np.asarray(summary["draws_best"])
        b = np.asarray(other["draws_best"])
        m = min(a.size, b.size)
        olo, omd, ohi = other["median_headroom_best"]
        overlap = {
            "other_dataset": other["dataset"],
            "intervals_overlap": bool(lo <= ohi and olo <= hi),
            "p_this_greater": float((a[:m] > b[:m]).mean()),
            "difference": [
                float(np.percentile(a[:m] - b[:m], 2.5)),
                float(np.median(a[:m] - b[:m])),
                float(np.percentile(a[:m] - b[:m], 97.5)),
            ],
        }
        d = overlap["difference"]
        print(f"\n  vs {other['dataset']}: {omd:+.4f} [{olo:+.4f}, {ohi:+.4f}]")
        print(
            f"  intervals overlap: {overlap['intervals_overlap']}  |  "
            f"P(this > other) = {overlap['p_this_greater']:.3f}  |  "
            f"difference {d[1]:+.4f} [{d[0]:+.4f}, {d[2]:+.4f}]"
        )
        print(
            "  ->  "
            + (
                "REVERSAL NOT ESTABLISHED — the intervals overlap"
                if overlap["intervals_overlap"]
                else "reversal holds — the intervals are disjoint"
            )
        )

    text = _report(rows, summary, overlap, cfg, volume, paths, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(
        json.dumps(
            {"per_target": rows, "summary": summary, "overlap": overlap}, indent=2, default=str
        )
    )
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def _report(rows, summary, overlap, cfg, volume, paths, args) -> str:
    lo, md, hi = summary["median_headroom_best"]
    lines = [
        f"# Is the ceiling headroom stable? — bootstrap on `{paths.dataset}`",
        "",
        f"Holdout **`{paths.holdout}`** — {volume.n_cells} cells x {volume.n_genes} genes. "
        f"{summary['replicates']} replicates, each resampling **cells** "
        f"({100 * args.cell_frac:.0f}%, without replacement) and **marker genes** (with "
        f"replacement), recomputing split-half reliability, the ceiling `sqrt(R)`, the copy "
        f"correlation and the headroom. Seed {args.seed}. No model, no fit.",
        "",
        f"**Dataset-level median headroom over the best copy: {md:+.4f} "
        f"[{lo:+.4f}, {hi:+.4f}]** = {md / ENVELOPE:.1f}x the {ENVELOPE} envelope "
        f"[{lo / ENVELOPE:.1f}x, {hi / ENVELOPE:.1f}x].",
        "",
        "| target | cells | markers | operational donor | headroom vs operational copy | "
        "headroom vs best copy |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        o, b = r["headroom_operational"], r["headroom_best"]
        lines.append(
            f"| `{r['target']}` | {r['n_cells']} | {r['n_markers']} | `{r['operational_donor']}` "
            f"| {o['median']:+.4f} [{o['p2.5']:+.4f}, {o['p97.5']:+.4f}] "
            f"| {b['median']:+.4f} [{b['p2.5']:+.4f}, {b['p97.5']:+.4f}] |"
        )
    if overlap:
        d = overlap["difference"]
        lines += [
            "",
            f"### Against `{overlap['other_dataset']}`",
            "",
            f"Difference in dataset-level median headroom: **{d[1]:+.4f} "
            f"[{d[0]:+.4f}, {d[2]:+.4f}]**, P(this > other) = **{overlap['p_this_greater']:.3f}**.",
            "",
            (
                "⚠️ **The intervals overlap — the reversal is not established** and must not be "
                "written as a strategic claim."
                if overlap["intervals_overlap"]
                else "**The intervals are disjoint — the reversal holds.**"
            ),
        ]
    lines += [
        "",
        "**Two biases, both against this file's own conclusion.** R is estimated by splitting the "
        "*subsample*, so Spearman-Brown corrects to the subsample's size rather than the "
        "section's and the ceiling comes out slightly low. And the gene resample captures the "
        "spread of the mean over the *selected* markers, not the variance of the selection — "
        "real on `deep_starmap` (32 chosen from 1017) and absent on tier-1, where all 28 genes "
        "are markers, so tier-1's interval is the more complete of the two. Both understate "
        "tier-1's advantage rather than manufacture it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
