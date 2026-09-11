"""Measure the angle budget — how far off-axis this specimen's geometry allows.

**Free: no fit, no model, no generation.** It reads the volume's real cells and asks, for each
angle, what a plane tilted that far actually cuts through.

Why it must be measured before the oblique figure is designed. Tier-1 STARmap is a **slab**:
sections at z = 19...85 um, so **66 um of depth** against an in-plane extent this script measures
(it is not committed anywhere in the repository, which is why the strip table in
``reports/oblique_layout_cost.md`` has one measured side and one assumed). A plane tilted by theta
exits the thin dimension after ``D / sin theta``, so the cells available fall away fast and past
some angle an "oblique section" is a sliver that cannot be scored as a section.

The gates below are derived from ``bench3/evaluate_paper.py::celltype_localization``'s own
constants rather than chosen:

* ``min_gt_cells = 20`` — a type with fewer cells is skipped, so the count of **scorable types** is
  what the metric can actually see;
* ``max_n = 250`` — every type is subsampled to at most this, so a strip whose largest type is
  below it sits under the metric's own design point.

The one number that is mine is the 0.6 in G1, and it is stated as mine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench3_paths import add_path_args, resolve  # noqa: E402

# from `celltype_localization`'s signature, read from source
METRIC_MIN_GT_CELLS = 20
METRIC_MAX_N = 250
# mine: how much of the coronal plane's scorable-type count an oblique strip must keep
SCORABLE_TYPE_FRACTION = 0.6

ANGLES = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0)


def scorable_types(codes: np.ndarray) -> int:
    """Types with at least the metric's own ``min_gt_cells``. Everything else it skips."""
    if codes.size == 0:
        return 0
    _u, counts = np.unique(codes, return_counts=True)
    return int((counts >= METRIC_MIN_GT_CELLS).sum())


def largest_type(codes: np.ndarray) -> int:
    if codes.size == 0:
        return 0
    return int(np.unique(codes, return_counts=True)[1].max())


def gates(row: dict, coronal: dict) -> tuple[bool, str]:
    """G1 and G2. Returns (clears, why-not)."""
    need = SCORABLE_TYPE_FRACTION * coronal["scorable_types"]
    if row["n_cells"] == 0:
        return False, "the slab misses the tissue entirely"
    if row["scorable_types"] < need:
        return False, (
            f"{row['scorable_types']} scorable types against the coronal plane's "
            f"{coronal['scorable_types']} — below {SCORABLE_TYPE_FRACTION:.0%} (G1)"
        )
    if row["largest_type"] < METRIC_MAX_N:
        return False, (
            f"the largest type has {row['largest_type']} cells, under the metric's own "
            f"max_n = {METRIC_MAX_N} subsample cap (G2)"
        )
    return True, ""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thickness", type=float, default=None,
                    help="slab thickness in um. Default: the volume's own median section spacing, "
                         "which is what a real section represents")
    ap.add_argument("--angles", type=float, nargs="+", default=list(ANGLES))
    ap.add_argument("--out", default="reports/angle_budget.md")
    ap.add_argument("--self-check", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()
    paths = resolve(args)

    from spatialcpav25_gen.config import Config
    from spatialcpav25_gen.data.schema import to_xyz
    from spatialcpav25_gen.infer.planes import plane_from_normal
    from spatialcpav25_gen.model.layout import cells_near_plane

    from _starmap_run import load_training_volume

    cfg = Config()
    vol = load_training_volume(cfg, paths.input)
    xyz = np.concatenate([np.asarray(to_xyz(s), dtype=np.float64) for s in vol.sections], axis=0)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = hi - lo
    zs = sorted(float(s.z) for s in vol.sections)
    spacing = float(np.median(np.diff(zs))) if len(zs) > 1 else float(extent[2])
    thickness = float(args.thickness) if args.thickness else spacing
    centre = 0.5 * (lo + hi)
    in_plane = float(np.mean(extent[:2]))
    aspect = in_plane / float(extent[2]) if extent[2] > 0 else float("inf")

    print(f"  volume: {xyz.shape[0]} cells, {len(vol.sections)} sections")
    print(f"  extent x/y/z = {extent[0]:.1f} / {extent[1]:.1f} / {extent[2]:.1f} um")
    print(f"  section spacing {spacing:.2f} um, slab thickness {thickness:.2f} um")
    print(f"  IN-PLANE : DEPTH = {aspect:.1f} : 1", flush=True)

    rows = []
    for deg in args.angles:
        t = np.deg2rad(float(deg))
        plane = plane_from_normal(
            [0.0, np.sin(t), np.cos(t)], centre, (extent[0], max(extent[1], extent[2])), thickness
        )
        near = cells_near_plane(vol.sections, plane, exclude=())
        uv = np.asarray(near.coords_uv, dtype=np.float64)
        if uv.shape[0]:
            span = uv.max(axis=0) - uv.min(axis=0)
            ar = float(min(span) / max(span)) if max(span) > 0 else 0.0
        else:
            span, ar = np.zeros(2), 0.0
        rows.append({
            "angle_deg": float(deg),
            "n_cells": int(uv.shape[0]),
            "n_sections": int(len(set(near.section_id.tolist()))),
            "scorable_types": scorable_types(np.asarray(near.cell_type)),
            "largest_type": largest_type(np.asarray(near.cell_type)),
            "extent_u": float(span[0]), "extent_v": float(span[1]),
            "aspect": ar,
        })
        print(f"    {deg:5.1f}°: {rows[-1]['n_cells']:6d} cells from "
              f"{rows[-1]['n_sections']} sections, {rows[-1]['scorable_types']} scorable types, "
              f"largest {rows[-1]['largest_type']}, aspect {ar:.3f}", flush=True)

    coronal = rows[0]
    for r in rows:
        r["clears"], r["why_not"] = gates(r, coronal)
    clean = [r for r in rows if r["clears"]]
    budget = max((r["angle_deg"] for r in clean), default=0.0)

    lines = render(rows, budget, extent, spacing, thickness, aspect, coronal)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"extent_um": extent.tolist(), "section_spacing_um": spacing,
         "slab_thickness_um": thickness, "in_plane_to_depth": aspect,
         "angles": rows, "angle_budget_deg": budget}, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


def render(rows, budget, extent, spacing, thickness, aspect, coronal) -> list[str]:
    out = [
        "# The angle budget — how far off-axis this specimen allows",
        "",
        "**Free: no fit, no model, no generation.** Real cells only, asking what a plane tilted by",
        "each angle actually cuts through.",
        "",
        f"Volume extent **{extent[0]:.0f} x {extent[1]:.0f} x {extent[2]:.0f} um**, section spacing",
        f"**{spacing:.1f} um**, slab thickness **{thickness:.1f} um**.",
        "",
        f"## IN-PLANE : DEPTH = **{aspect:.1f} : 1**",
        "",
        "That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 um across and",
        "400 um deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been",
        "measured on. A plane tilted by θ exits the thin dimension after `D / sin θ`.",
        "",
        "| θ | cells | sections drawn from | scorable types | largest type | aspect | clears |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['angle_deg']:.0f}° | {r['n_cells']} | {r['n_sections']} | "
            f"{r['scorable_types']} | {r['largest_type']} | {r['aspect']:.3f} | "
            f"{'**yes**' if r['clears'] else 'no'} |"
        )
    out += [
        "",
        "**The gates, derived from `celltype_localization`'s own constants rather than chosen:**",
        "",
        f"- **G1** — scorable types (≥ `min_gt_cells` = {METRIC_MIN_GT_CELLS} cells) must be at",
        f"  least **{SCORABLE_TYPE_FRACTION:.0%}** of the coronal plane's {coronal['scorable_types']}."
        "  *The fraction is mine; the cell count is the metric's.*",
        f"- **G2** — the largest type must have ≥ `max_n` = {METRIC_MAX_N} cells, the metric's own",
        "  subsample cap. Below it the strip sits under the design point of the statistic.",
        "",
        f"## The budget: **{budget:.0f}°**",
        "",
    ]
    failed = [r for r in rows if not r["clears"] and r["angle_deg"] > budget]
    if failed:
        out += ["The first angle that fails, and why:", "",
                f"- **{failed[0]['angle_deg']:.0f}°** — {failed[0]['why_not']}", ""]
    out += [
        "**What this decides.** The oblique demonstration runs at the largest angle that clears,",
        "and the figure states the angle and the cell count together. If the budget is 10–15°, the",
        "paper's claim is *oblique within the specimen's geometry* and says so — which is still a",
        "capability no published method has, and is better than a 45° figure whose few hundred",
        "cells cannot be scored as a section.",
        "",
        "⚠️ **A budget is not a result.** It says what this specimen permits, not what the method",
        "achieves at that angle. `reports/oblique_layout_cost.md` §3c: there is no real oblique",
        "section to score against, so the evaluation set is the real cells near the plane — which",
        "are also the donors, and must be excluded from them.",
    ]
    return out


def _self_check() -> int:
    """The gates and the geometry, on synthetic slabs. No data, seconds."""
    coronal = {"scorable_types": 10, "n_cells": 4000, "largest_type": 1200}
    cases = [
        ("a full coronal plane clears", {"n_cells": 4000, "scorable_types": 10, "largest_type": 1200}, True),
        ("an empty slab is refused", {"n_cells": 0, "scorable_types": 0, "largest_type": 0}, False),
        ("too few scorable types (G1)", {"n_cells": 900, "scorable_types": 5, "largest_type": 400}, False),
        (f"largest type under max_n={METRIC_MAX_N} (G2)",
         {"n_cells": 900, "scorable_types": 8, "largest_type": 200}, False),
        ("exactly at both bounds clears",
         {"n_cells": 900, "scorable_types": 6, "largest_type": METRIC_MAX_N}, True),
    ]
    checks = [(f"{lab} -> {gates(row, coronal)[0]}", gates(row, coronal)[0] == want)
              for lab, row, want in cases]
    codes = np.array([0] * 30 + [1] * 19 + [2] * 300)
    checks += [
        (f"scorable_types counts only types with >= {METRIC_MIN_GT_CELLS} cells",
         scorable_types(codes) == 2),
        ("largest_type is the biggest", largest_type(codes) == 300),
        ("both are 0 on an empty strip",
         scorable_types(np.array([], dtype=int)) == 0 and largest_type(np.array([], dtype=int)) == 0),
        ("a refusal always says which gate", all(gates(r, coronal)[1] for _l, r, w in cases if not w)),
    ]
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    failed = sum(1 for _l, ok in checks if not ok)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
