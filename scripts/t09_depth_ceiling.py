"""What is `marker_depth_r` actually capable of? — the metric's ceiling and floor, from data alone.

The `deep_starmap` audits put `zinb-flow` at +0.2745 and `cross-mix` at +0.0147 on
``marker_depth_r``, and nothing in T09 says whether +0.27 is close to what is achievable or
close to nothing. Without that number the gate cannot be interpreted: at a ceiling of +0.85 both
arms fail badly and the margin is a crumb; at a ceiling of +0.30 ``zinb-flow`` is essentially at
ceiling and the whole gate reduces to "``cross-mix`` is broken".

**No model, no fit, no generation** — every reference point below is computed from the built
input alone, with the metric's own kernels (``marker_genes``, ``profile_axis``,
``soft_depth_profile``, ``_normalised``) and the metric's own conventions: markers and depth
bounds always come from the **target** section, and both sides are placed at the target's ``z``,
exactly as ``section_scores`` does it for a generated section.

Four reference points per target section:

``self``
    The target scored against itself. Must be **exactly 1.0** — a correctness check on this
    file, not a result. If it is not 1.0 nothing else here means anything.
``split_half``
    The target's cells split at random into two halves, each half profiled, the halves
    correlated (Spearman-Brown corrected to whole-section size). This is the **reliability** R
    of a whole-section profile — what a method would score if it reproduced the biology
    perfectly *and* carried the same sampling noise the target does.
``noiseless_ceiling``
    ``sqrt(R)``. Correction for attenuation: a method with **no** noise of its own still cannot
    correlate with the observed target above ``sqrt(R)``, because the target itself is a noisy
    measurement. This is the hard bound; ``split_half`` is the bound for a method as noisy as
    the data.
``other_section``
    Another **real** section of the same specimen, profiled on the target's ruler. This is the
    **copying ceiling** — the best a donor-copying method could do if it copied that whole
    section, which is the thing ``cross-mix`` approximates donor by donor. Reported against
    ``|dz|`` because a neighbouring section should copy better than a distant one.
``shuffled``
    The target's own counts with the cell-to-position assignment permuted. The **floor**:
    depth structure destroyed, biology otherwise identical. Should sit at ~0.

Usage::

    python scripts/t09_depth_ceiling.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --bench3 /path/to/benchmark-pbya-v3 --out reports/t09_depth_ceiling_deep.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.losses.metric_aware import (
    knn_weight_graph,
    marker_genes,
    profile_axis,
    soft_depth_profile,
)
from spatialcpav25_gen.train.select import _normalised, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import base_config, clamp_config_to_input, load_training_volume


def ruler(target, axis, cfg):
    """The target section's own bounds and kernel width — the ruler both sides are binned on."""
    xy = np.asarray(target.coords, dtype=np.float64)
    xyz = torch.from_numpy(
        np.concatenate([xy, np.full((xy.shape[0], 1), float(target.z))], axis=1).astype(np.float32)
    )
    projected = (xyz @ axis).numpy()
    bounds = (float(projected.min()), float(projected.max()))
    sigma = float(cfg.profile_sigma_frac) * (bounds[1] - bounds[0]) / int(cfg.profile_n_bins)
    return bounds, sigma


def profile(counts, coords_xy, z_value, axis, markers, cfg, bounds, sigma):
    """``(n_bins, n_markers)`` on the target's ruler. Mirrors ``section_scores`` exactly."""
    x = _normalised(np.asarray(counts, dtype=np.float64), cfg)
    xy = np.asarray(coords_xy, dtype=np.float64)[:, :2]
    xyz = torch.from_numpy(
        np.concatenate([xy, np.full((xy.shape[0], 1), float(z_value))], axis=1).astype(np.float32)
    )
    return soft_depth_profile(
        x.index_select(1, markers), xyz, axis, int(cfg.profile_n_bins), sigma, bounds=bounds
    ).numpy()


def safe_r(a, b) -> float:
    """Pearson r, 0.0 on a degenerate side. Mirrors ``select._safe_r``."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size < 2 or x.std() == 0.0 or y.std() == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score(p_other, p_target) -> float:
    """``marker_depth_r``: the mean per-gene profile correlation. The metric, unmodified."""
    return float(np.mean([safe_r(p_other[:, g], p_target[:, g]) for g in range(p_target.shape[1])]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=1, help="seeds the split-half and the shuffle")
    ap.add_argument("--splits", type=int, default=20, help="random split-half repeats")
    ap.add_argument("--shuffles", type=int, default=20, help="random cell-position shuffles")
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument(
        "--all-sections",
        action="store_true",
        help="target every section, not just the audits' LOSO folds",
    )
    ap.add_argument("--out", default="reports/t09_depth_ceiling.md")
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
    targets = volume.sections if args.all_sections else selection_folds(volume, cfg)
    print(
        f"  {volume.n_cells} cells x {volume.n_genes} genes, {volume.n_sections} sections; "
        f"targets {[s.section_id for s in targets]}; {cfg.metric_marker_genes} marker genes, "
        f"{cfg.profile_n_bins} depth bins"
    )
    print("  no model, no fit, no generation — this reads the built input only\n")

    rows: list[dict] = []
    for target in targets:
        counts = np.asarray(target.counts.todense(), dtype=np.float64)
        xy = np.asarray(target.coords, dtype=np.float64)
        markers = marker_genes(_normalised(counts, cfg), knn_weight_graph(xy, cfg), cfg)
        bounds, sigma = ruler(target, axis, cfg)
        p_target = profile(counts, xy, target.z, axis, markers, cfg, bounds, sigma)
        n_markers = int(p_target.shape[1])

        rng = np.random.default_rng(args.seed)
        self_r = score(p_target, p_target)

        # Noise ceiling. Each half has half the cells, so its profile is noisier than a
        # whole-section one; Spearman-Brown lifts the half-half correlation to the reliability
        # of a full-size measurement, which is the bound a method is actually competing against.
        halves = []
        for _ in range(int(args.splits)):
            order = rng.permutation(counts.shape[0])
            a, b = order[: order.size // 2], order[order.size // 2 :]
            ra = score(
                profile(counts[a], xy[a], target.z, axis, markers, cfg, bounds, sigma),
                profile(counts[b], xy[b], target.z, axis, markers, cfg, bounds, sigma),
            )
            halves.append(2.0 * ra / (1.0 + ra) if ra > -1.0 else ra)
        split_half = float(np.mean(halves))
        noiseless = float(np.sqrt(split_half)) if split_half > 0.0 else float("nan")

        # Floor: same cells, same counts, positions permuted. Depth structure destroyed.
        shuffled = float(
            np.mean(
                [
                    score(
                        profile(
                            counts,
                            xy[rng.permutation(xy.shape[0])],
                            target.z,
                            axis,
                            markers,
                            cfg,
                            bounds,
                            sigma,
                        ),
                        p_target,
                    )
                    for _ in range(int(args.shuffles))
                ]
            )
        )

        others = []
        for donor in volume.sections:
            if donor.section_id == target.section_id:
                continue
            d_counts = np.asarray(donor.counts.todense(), dtype=np.float64)
            r = score(
                profile(
                    d_counts,
                    np.asarray(donor.coords, dtype=np.float64),
                    target.z,
                    axis,
                    markers,
                    cfg,
                    bounds,
                    sigma,
                ),
                p_target,
            )
            others.append(
                {"donor": donor.section_id, "dz": abs(float(donor.z) - float(target.z)), "r": r}
            )
        others.sort(key=lambda o: o["dz"])

        rows.append(
            {
                "target": target.section_id,
                "dataset": paths.dataset,
                "holdout": paths.holdout,
                "n_cells": int(counts.shape[0]),
                "n_markers": n_markers,
                "self": self_r,
                "split_half": split_half,
                "noiseless_ceiling": noiseless,
                "shuffled": shuffled,
                "nearest_other": others[0]["r"] if others else None,
                "best_other": max((o["r"] for o in others), default=None),
                "mean_other": float(np.mean([o["r"] for o in others])) if others else None,
                "others": others,
            }
        )
        near = f"{others[0]['r']:+.4f} (dz {others[0]['dz']:.0f} um)" if others else "n/a"
        print(
            f"  {target.section_id}: self {self_r:+.6f} | split-half {split_half:+.4f} "
            f"(noiseless {noiseless:.4f}) | "
            f"nearest other {near} | best other "
            f"{max((o['r'] for o in others), default=float('nan')):+.4f} | shuffled "
            f"{shuffled:+.4f}",
            flush=True,
        )
        if abs(self_r - 1.0) > 1e-9:
            raise SystemExit(
                f"self-correlation on {target.section_id} is {self_r!r}, not 1.0. The profile "
                "code in this file does not reproduce the metric; no number here is usable."
            )

    text = _report(rows, cfg, volume, paths, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + text)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


# Where each dataset's `expr_mode` audit was written. Only the *path* is recorded here — the
# numbers are read out of the JSON, because this project has twice shipped a report carrying a
# hand-copied number for the wrong dataset.
AUDIT_JSON = {
    "deep_starmap": "reports/t09_audit_deep_expr_mode.json",
    "starmap_visual_cortex": "reports/t09_audit_expr_mode.json",
}


def _audit_reference(dataset):
    """``(path, {arm: marker_depth_r})`` from the dataset's own audit JSON, or ``None``."""
    name = AUDIT_JSON.get(str(dataset))
    if name is None or not Path(name).exists():
        return None
    try:
        rows = json.loads(Path(name).read_text())
        arms = {str(r["option"]): float(r["mean"]["marker_depth_r"]) for r in rows}
    except (KeyError, TypeError, ValueError):
        return None
    return (name, arms) if arms else None


def _report(rows, cfg, volume, paths, args) -> str:
    lines = [
        "# `marker_depth_r` — the metric's ceiling and floor, measured from data alone",
        "",
        f"Dataset **`{paths.dataset}`**, holdout **`{paths.holdout}`** — {volume.n_cells} cells x "
        f"{volume.n_genes} genes over {volume.n_sections} training sections. "
        f"{cfg.metric_marker_genes} marker genes per target, {cfg.profile_n_bins} depth bins, "
        f"{args.splits} split-half repeats, {args.shuffles} shuffles, seed {args.seed}. "
        "**No model, no fit, no generation** — the metric's own kernels applied to the built "
        "input.",
        "",
        "| target | cells | genes | self | split-half R | **noiseless ceiling sqrt(R)** | "
        "nearest other section | best other section | mean other | shuffled (floor) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['target']}` | {r['n_cells']} | {r['n_markers']} | {r['self']:.6f} | "
            f"{r['split_half']:+.4f} | **{r['noiseless_ceiling']:.4f}** | "
            f"{r['nearest_other']:+.4f} | {r['best_other']:+.4f} | "
            f"{r['mean_other']:+.4f} | {r['shuffled']:+.4f} |"
        )
    lines += [
        "",
        "`self` must be exactly 1.0; it is a correctness check on the profile code in "
        "`scripts/t09_depth_ceiling.py`, not a result. The run aborts if it is not.",
        "",
        "### Every donor section, by distance",
        "",
        "| target | donor | \\|dz\\| (um) | `marker_depth_r` |",
        "|---|---|---|---|",
    ]
    for r in rows:
        for o in r["others"]:
            lines.append(f"| `{r['target']}` | `{o['donor']}` | {o['dz']:.0f} | {o['r']:+.4f} |")
    audit = _audit_reference(paths.dataset)
    lines += [
        "",
        "### How to read this against the audits",
        "",
    ]
    if audit is None:
        lines.append(
            f"No `expr_mode` audit JSON was found for `{paths.dataset}`, so the comparison "
            "below has to be made by hand against whatever that dataset's audit reported."
        )
    else:
        arms = "  ".join(f"`{k}` at **{v:+.4f}**" for k, v in audit[1].items())
        lines.append(
            f"`{paths.dataset}`'s `expr_mode` audit (`{audit[0]}`) put {arms} on this metric. "
            "Place them against the columns above:"
        )
    lines += [
        "",
        "* **noiseless ceiling `sqrt(R)`** is the hard bound: even a method with no noise of "
        "its own cannot correlate with the observed target above this, because the target is "
        "itself a finite-sample measurement. **split-half R** is the softer bound, for a method "
        "as noisy as the data.",
        "* **best other section** is the copying ceiling: the most a donor-copying method could "
        "score if it copied a whole real section. `cross-mix` copies donor by donor, so it is "
        "competing against this number, not against 1.0.",
        "* **shuffled** is the floor. An arm at the floor is not modelling depth at all.",
        "",
        "If the copying ceiling is near zero, `cross-mix`'s score is not a model failure — no "
        "donor-copying method could have done better, and the `expr_mode` margin on this metric "
        "says nothing about the flow head. If the copying ceiling is high and `cross-mix` is at "
        "the floor, the deficit is `cross-mix`'s donor retrieval, still not the flow head's "
        "modelling. Only a high ceiling *with* `zinb-flow` well above the copying ceiling makes "
        "the margin a statement about the generative path.",
        "",
        "**One caveat on the split-half.** Each half carries half the cells, and Spearman-Brown "
        "corrects for that only under assumptions this profile does not exactly meet. On a "
        "section with few cells the correction under-estimates R, which is why the per-target "
        "cell count is printed beside it.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
