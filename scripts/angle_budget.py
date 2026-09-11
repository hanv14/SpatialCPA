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


# Every built dataset that carries cell types, from `specs/10` §5.4's table. The angle budget is
# FREE -- no fit, no model, no generation -- so §5.4's cost exclusions (which are about fitting)
# do not apply to it. `allen_merfish_brain` is excluded from the CAMPAIGN at 1.17 M cells and 59
# sections; reading its geometry costs one h5ad load, and its 59 sections are the single most
# likely thing in the table to clear a scorable oblique angle.
ALL_DATASETS = (
    "starmap_visual_cortex",
    "deep_starmap",
    "merfish_thick_cortex",
    "merfish_thick_hypothalamus",
    "cosmx_nsclc_3d",
    "exseq_breast_cancer",
    "exseq_visual_cortex",
    "allen_merfish_brain",
)


def measure_one(paths, args) -> dict:
    """The whole budget for one dataset, as a record. No fit, no model, no generation."""
    from spatialcpav25_gen.data.schema import to_xyz
    from spatialcpav25_gen.infer.planes import plane_from_normal
    from spatialcpav25_gen.model.layout import cells_near_plane

    from _starmap_run import load_training_volume, prepare_config

    # The whole canonical form, not half of it. Two of Config's defaults are wrong for a bench3
    # build and BOTH abort the load before any geometry is read: `region_key="region"` names an
    # obs column these files do not carry, and `expr_pca_dim=32` exceeds tier-1's 28-gene panel.
    # This script reads neither field -- see reports/angle_budget_config_note.md on why the load
    # validates them anyway.
    cfg = prepare_config(seed=0, input_path=paths.input)
    vol = load_training_volume(cfg, paths.input)
    xyz = np.concatenate([np.asarray(to_xyz(s), dtype=np.float64) for s in vol.sections], axis=0)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = hi - lo
    zs = sorted(float(s.z) for s in vol.sections)
    spacing = float(np.median(np.diff(zs))) if len(zs) > 1 else float(extent[2])
    thickness = float(args.thickness) if args.thickness else spacing
    in_plane = float(np.mean(extent[:2]))
    aspect = in_plane / float(extent[2]) if extent[2] > 0 else float("inf")

    # RETRACTED (v1): the origin was 0.5 * (lo + hi), the volume's z-MIDPOINT. On tier-1 that is
    # 52.0 um, exactly midway between the sections at 41 and 63, so the +-11 um band admitted BOTH
    # on an exact floating-point tie and the coronal reference row was a DOUBLE-thickness plane:
    # 8279 cells against ~4165 per real section, and G1's 19-type denominator measured on it.
    # See reports/retractions.md R1. The origin is now the median section's own z, so the 0 deg
    # row is one section -- which is what the pipeline generates and what every other row is
    # compared against.
    z_ref = float(np.median(zs))
    z_ref = min(zs, key=lambda z: (abs(z - z_ref), z))
    centre = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), z_ref])

    print(f"  volume: {xyz.shape[0]} cells, {len(vol.sections)} sections")
    print(f"  extent x/y/z = {extent[0]:.1f} / {extent[1]:.1f} / {extent[2]:.1f} um")
    print(f"  section spacing {spacing:.2f} um, slab thickness {thickness:.2f} um")
    print(f"  reference plane centred on the section at z = {z_ref:.2f} um")
    print(f"  IN-PLANE : DEPTH = {aspect:.1f} : 1", flush=True)

    rows = []
    for deg in args.angles:
        t = np.deg2rad(float(deg))
        plane = plane_from_normal(
            [0.0, np.sin(t), np.cos(t)], centre, (extent[0], max(extent[1], extent[2])), thickness
        )
        # Two different questions, and the flag is which (see `cells_near_plane`):
        #   threshold -> what is available to EVALUATE near this plane. These gates' question.
        #   expanded  -> which real cells the LAYOUT would reuse. Differs only when the slab is
        #                empty, which is exactly the generation setting and exactly the case the
        #                first version of the rule got wrong.
        near = cells_near_plane(vol.sections, plane, exclude=())
        donors = cells_near_plane(vol.sections, plane, exclude=(), expand_to_nearest=True)
        uv = np.asarray(near.coords_uv, dtype=np.float64)
        if uv.shape[0]:
            span = uv.max(axis=0) - uv.min(axis=0)
            ar = float(min(span) / max(span)) if max(span) > 0 else 0.0
        else:
            span, ar = np.zeros(2), 0.0
        rows.append({
            "angle_deg": float(deg),
            "n_cells": int(uv.shape[0]),
            "n_donors": int(donors.coords_uv.shape[0]),
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
    at_budget = next(r for r in rows if r["angle_deg"] == budget)
    failed = [r for r in rows if not r["clears"] and r["angle_deg"] > budget]

    return {
        "dataset": paths.dataset,
        "holdout": paths.holdout,
        "n_cells": int(xyz.shape[0]),
        "n_sections": int(len(vol.sections)),
        "n_types": int(len(vol.celltype_names)),
        "extent_um": extent.tolist(),
        "section_spacing_um": spacing,
        "slab_thickness_um": thickness,
        "reference_plane_z_um": z_ref,
        "in_plane_to_depth": aspect,
        "angles": rows,
        "angle_budget_deg": budget,
        "cells_at_budget": int(at_budget["n_cells"]),
        "first_failure": (failed[0]["angle_deg"], failed[0]["why_not"]) if failed else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thickness", type=float, default=None,
                    help="slab thickness in um. Default: the volume's own median section spacing, "
                         "which is what a real section represents")
    ap.add_argument("--angles", type=float, nargs="+", default=list(ANGLES))
    ap.add_argument("--datasets", nargs="+", default=None,
                    help=f"measure several built datasets and write the cross-dataset table. "
                         f"`all` expands to {', '.join(ALL_DATASETS)}. Default: just --dataset. "
                         "Free for every one of them: no fit, no model, no generation.")
    ap.add_argument("--out", default="reports/angle_budget.md")
    ap.add_argument("--self-check", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()

    names = args.datasets or [args.dataset]
    if names == ["all"] or "all" in names:
        names = list(ALL_DATASETS)

    records: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for name in names:
        print(f"\n=== {name}", flush=True)
        try:
            paths = resolve(argparse.Namespace(**{**vars(args), "dataset": name}))
            records.append(measure_one(paths, args))
        except (SystemExit, OSError, KeyError, ValueError) as exc:
            # Named and carried into the report, never dropped (Convention 6). A dataset that is
            # not built on this machine is a fact about the machine, and the table says so rather
            # than quietly showing one fewer row.
            reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            print(f"  SKIPPED: {reason}", flush=True)
            skipped.append((name, reason))

    if not records:
        raise SystemExit(
            "angle_budget: no dataset could be read. Tried: "
            + "; ".join(f"{n} ({why})" for n, why in skipped)
        )

    lines = (render_sweep(records, skipped) if len(names) > 1 else []) + [
        line for rec in records for line in ([""] + render(rec) if len(names) > 1 else render(rec))
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"datasets": records, "skipped": [{"dataset": n, "why": w} for n, w in skipped]},
        indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


def render_sweep(records: list[dict], skipped: list[tuple[str, str]]) -> list[str]:
    """The cross-dataset table: aspect ratio beside the budget, which is the whole story."""
    scored = [r for r in records if r["angle_budget_deg"] >= 30.0]
    out = [
        "# The angle budget across every built specimen",
        "",
        "**Free: no fit, no model, no generation.** For each dataset, real cells only, asking what",
        "a plane tilted by each angle actually cuts through — and whether what it cuts is enough",
        "for `celltype_localization` to score.",
        "",
        "Gates are the metric's **own** constants: **G1** scorable types (≥ `min_gt_cells` = "
        f"{METRIC_MIN_GT_CELLS}) ≥ {SCORABLE_TYPE_FRACTION:.0%} of the coronal plane's; "
        f"**G2** largest type ≥ `max_n` = {METRIC_MAX_N}, its subsample cap. The "
        f"{SCORABLE_TYPE_FRACTION:.0%} is the one number that is mine.",
        "",
        "| dataset | cells | sections | extent x/y/z µm | **in-plane : depth** | **budget** | "
        "cells there | first failure |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        e = r["extent_um"]
        fail = r["first_failure"]
        why = f"{fail[0]:.0f}° — {fail[1]}" if fail else "clears every angle measured"
        out.append(
            f"| `{r['dataset']}` | {r['n_cells']} | {r['n_sections']} | "
            f"{e[0]:.0f} × {e[1]:.0f} × {e[2]:.0f} | **{r['in_plane_to_depth']:.1f} : 1** | "
            f"**{r['angle_budget_deg']:.0f}°** | {r['cells_at_budget']} | {md_cell(why)} |"
        )
    for name, why in skipped:
        out.append(f"| `{name}` | — | — | — | — | *not read* | — | {md_cell(why)} |")

    out += ["", "## What this decides", ""]
    if scored:
        best = max(scored, key=lambda r: r["angle_budget_deg"])
        out += [
            f"**`{best['dataset']}` clears {best['angle_budget_deg']:.0f}°** with "
            f"{best['cells_at_budget']} cells, at an in-plane : depth ratio of "
            f"{best['in_plane_to_depth']:.1f} : 1. The oblique demonstration is **scored** rather",
            "than shown, and the paper's claim is a measured one.",
        ]
    else:
        out += [
            "**No built specimen clears 30°.** Every one of them is a slab: a few tens of "
            "micrometres",
            "of depth against a millimetre or more in plane, so a plane tilted past a few degrees "
            "exits",
            "the thin dimension after `(D + t) / sin θ` and returns a sliver the metric's own "
            "constants",
            "cannot score.",
            "",
            "**This is a finding about the field's data, not a failure of the method.** 3D "
            "spatial",
            "transcriptomics is published as stacks of thin sections, and a stack of thin "
            "sections does",
            "not contain an obliquely-cut section to score against at any useful angle. The",
            "well-definedness of the method off-axis is a property of the method; the "
            "*evaluability*",
            "of an off-axis section is a property of the specimen, and no published work states "
            "what",
            "geometry it needs. This table is that statement.",
            "",
            "So the claim splits, and both halves are honest:",
            "",
            "- **well-definedness** — shown at 45°: the method produces a coherent section where "
            "no",
            "  layout mode previously had a definition. Shown, not scored, and labelled so.",
            "- **evaluability** — scored at the largest angle that clears, with the cell count and "
            "the",
            "  angle stated together, and this table as the reason it is not larger.",
        ]
    out += [
        "",
        "⚠️ **A budget is not a result.** It says what a specimen permits, not what the method",
        "achieves at that angle. `reports/oblique_layout_cost.md` §3c: there is no real oblique",
        "section to score against, so the evaluation set is the real cells near the plane — which",
        "are also the donors, and must be excluded from them.",
        "",
        "---",
    ]
    return out


def render(rec: dict) -> list[str]:
    """One dataset's own table."""
    rows, extent = rec["angles"], rec["extent_um"]
    coronal, budget = rows[0], rec["angle_budget_deg"]
    out = [
        f"## `{rec['dataset']}` — the angle budget",
        "",
        f"{rec['n_cells']} cells, {rec['n_sections']} sections, {rec['n_types']} types. "
        f"Extent **{extent[0]:.0f} × {extent[1]:.0f} × {extent[2]:.0f} µm**, section spacing "
        f"**{rec['section_spacing_um']:.1f} µm**, slab thickness "
        f"**{rec['slab_thickness_um']:.1f} µm**.",
        "",
        f"The reference plane is centred on the **real section at z = "
        f"{rec['reference_plane_z_um']:.1f} µm**, not on the volume's z-midpoint — see "
        "`reports/retractions.md` R1 for why that distinction cost a published reference row.",
        "",
        f"### IN-PLANE : DEPTH = **{rec['in_plane_to_depth']:.1f} : 1**",
        "",
        "That ratio is the whole constraint. GATE 2's synthetic fixture was 3000 µm across and",
        "400 µm deep — **7.5 : 1** — and it is the only geometry oblique parity has ever been",
        "measured on. A plane tilted by θ exits the thin dimension after `(D + t) / sin θ`.",
        "",
        "| θ | cells in slab | donors | sections | scorable types | largest type | strip µm | "
        "aspect | clears |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['angle_deg']:.0f}° | {r['n_cells']} | {r['n_donors']} | {r['n_sections']} | "
            f"{r['scorable_types']} | {r['largest_type']} | {r['extent_u']:.0f} | "
            f"{r['aspect']:.3f} | {'**yes**' if r['clears'] else 'no'} |"
        )
    out += [
        "",
        f"- **G1** — scorable types (≥ `min_gt_cells` = {METRIC_MIN_GT_CELLS} cells) must be at "
        f"least **{SCORABLE_TYPE_FRACTION:.0%}** of the coronal plane's "
        f"{coronal['scorable_types']}. *The fraction is mine; the cell count is the metric's.*",
        f"- **G2** — the largest type must have ≥ `max_n` = {METRIC_MAX_N} cells, the metric's own "
        "subsample cap. Below it the strip sits under the design point of the statistic.",
        "- **cells in slab** is what is available to *evaluate*; **donors** is what the layout "
        "would *reuse*. They differ only when the slab is empty — the generation setting, and the "
        "case the first version of the selection rule got wrong.",
        "",
        f"**Budget: {budget:.0f}°.**",
    ]
    if rec["first_failure"]:
        deg, why = rec["first_failure"]
        out += ["", f"First angle that fails: **{deg:.0f}°** — {md_cell(why)}"]
    return out


def md_cell(text: str) -> str:
    """Escape a pipe so a rendered row cannot shift a value into the wrong column (§4.2m)."""
    return str(text).replace("|", "\\|")


def _self_check() -> int:
    """The gates and the geometry, on synthetic slabs. No data, seconds."""
    coronal = {"scorable_types": 10, "n_cells": 4000, "largest_type": 1200}
    cases = [
        ("a full coronal plane clears",
         {"n_cells": 4000, "scorable_types": 10, "largest_type": 1200}, True),
        ("an empty slab is refused", {"n_cells": 0, "scorable_types": 0, "largest_type": 0}, False),
        ("too few scorable types (G1)",
         {"n_cells": 900, "scorable_types": 5, "largest_type": 400}, False),
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
         scorable_types(np.array([], dtype=int)) == 0
         and largest_type(np.array([], dtype=int)) == 0),
        ("a refusal always says which gate",
         all(gates(r, coronal)[1] for _l, r, w in cases if not w)),
    ]
    # --- the sweep's ASSEMBLY, not just its scoring (specs/10 §4.2p). The sidecar crashed twice
    # after a measurement was paid for because the self-check exercised every constructor and
    # never the assembly; a cross-dataset sweep has exactly that shape, so it is built here.
    def rec(name, budget, aspect, n_sec, fail):
        rows = [dict(angle_deg=a, n_cells=100, n_donors=100, n_sections=1, scorable_types=9,
                     largest_type=300, extent_u=500.0, extent_v=900.0, aspect=0.5,
                     clears=a <= budget, why_not="" if a <= budget else "G2 | with a pipe in it")
                for a in (0.0, 5.0, 45.0)]
        return {"dataset": name, "holdout": "h", "n_cells": 9, "n_sections": n_sec, "n_types": 4,
                "extent_um": [1000.0, 900.0, 66.0], "section_spacing_um": 22.0,
                "slab_thickness_um": 22.0, "reference_plane_z_um": 41.0,
                "in_plane_to_depth": aspect, "angles": rows, "angle_budget_deg": budget,
                "cells_at_budget": 100, "first_failure": fail}

    slabs = [rec("a", 5.0, 21.6, 4, (10.0, "too few | types")), rec("b", 5.0, 30.0, 7, None)]
    thick = [rec("c", 45.0, 3.0, 59, None)]
    sweep_slab = "\n".join(render_sweep(slabs, [("d", "not built on this machine")]))
    sweep_thick = "\n".join(render_sweep(thick, []))
    one = "\n".join(render(slabs[0]))

    def row_widths(md: str) -> list[bool]:
        """Each rendered row against its own header's cell count. Walks the RENDERED text (§4.2m).

        Counts what a markdown renderer counts: an **escaped** pipe is data, not a column break,
        so `\\|` is removed before counting. The first version of this counter did not, and
        reported a correctly-escaped row as a defect -- a checker miscounting is still a checker
        that has to be fixed at the counter, never at the render it is judging.
        """
        ok, header = [], None
        for line in md.split("\n"):
            if not line.startswith("|"):
                header = None
                continue
            n = line.replace("\\|", "").count("|")
            if header is None:
                header = n
            else:
                ok.append(n == header)
        return ok

    checks += [
        ("the sweep renders and names every dataset it was given",
         all(f"`{r['dataset']}`" in sweep_slab for r in slabs)),
        ("a dataset it could NOT read is still a row, never a missing one (Convention 6)",
         "`d`" in sweep_slab and "not read" in sweep_slab),
        ("every rendered row is square with its header, pipes in the data included (§4.2m)",
         all(row_widths(sweep_slab)) and all(row_widths(sweep_thick)) and all(row_widths(one))),
        ("a pipe inside a reason is escaped rather than shifting a column (§4.2m)",
         "\\|" in sweep_slab),
        ("all-slabs reads as a finding about the DATA, not a failure of the method",
         "finding about the field's data" in sweep_slab and "45" in sweep_slab),
        ("and it splits the claim into well-definedness and evaluability",
         "well-definedness" in sweep_slab and "evaluability" in sweep_slab),
        ("a specimen that clears 30 deg flips the conclusion to SCORED",
         "scored" in sweep_thick.lower() and "finding about the field's data" not in sweep_thick),
        ("the per-dataset table separates what is EVALUABLE from what the layout REUSES",
         "cells in slab" in one and "donors" in one),
        ("and it says the reference plane sits on a real section, not the z-midpoint (R1)",
         "centred on the **real section" in one),
        ("`all` expands to more than one dataset, so the sweep is not vacuous",
         len(ALL_DATASETS) > 1 and "starmap_visual_cortex" in ALL_DATASETS
         and "allen_merfish_brain" in ALL_DATASETS),
    ]

    # The wiring, not the scoring (specs/10 §4.2p, §4.2q). This runner has no checkpoint to
    # restore a Config from, so `base_config` is the only sanctioned source; reaching for a bare
    # `Config()` is what took its first real run down inside `loaders.py`.
    from _contract import (
        bench3_clamp_discipline,
        bench3_config_discipline,
        uses_shared_base_config,
    )

    checks += uses_shared_base_config("angle_budget.py")
    checks += bench3_config_discipline()
    checks += bench3_clamp_discipline()
    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    failed = sum(1 for _l, ok in checks if not ok)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
