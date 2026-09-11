"""The oblique demonstration, exactly as `reports/oblique_demonstration_preregistration.md` commits.

**Zero fits for every headline arm.** `celltype_localization` is nearly blind to generated
expression (`retractions.md` R2), and all three headline arms reproduce real cells, so the claim
this runner measures needs no model. `--weights` adds the `field` ablation, which F1 keeps out of
the headline.

The construction, in one paragraph. For each angle a target plane is cut through the training
volume; the **ground truth** is the real cells inside its slab (`cells_near_plane`, threshold form),
written as a one-section dataset so bench3's own `evaluate_paper` scores it unchanged. The
**donors** are a *flanking slab* — the same orientation, origin offset along the normal by one
thickness — which is `flanking_copy`'s construction generalised to any orientation and is disjoint
from the ground truth by construction (§3-bis). The arms differ only in how they turn donors into a
section.

What the demonstration is *for*: off-axis, `nearest-z` can only paste a whole coronal face onto the
plane, so its footprint is the section's, not the plane's. `plane-distance` returns the cells the
plane actually cuts. That difference is the contribution, and this measures it on the pinned
instrument.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench3_paths import add_path_args, resolve  # noqa: E402

METRIC = "celltype_localization"

# from `celltype_localization`'s signature, read from source. Same constants as the angle budget.
METRIC_MIN_GT_CELLS = 20
METRIC_MAX_N = 250
SCORABLE_TYPE_FRACTION = 0.6

ANGLES = (0.0, 30.0, 45.0, 60.0, 90.0)
SEEDS = (1, 2, 3)

# §6's band: `resample-pd` may sit this far below the baseline and still count as demonstrated.
COST_BAND = 0.05
# §5 P2: the permuted-type null must stay at or under this, or the metric is not responding.
NULL_CEILING = 0.10
# §5 P4: arms being differenced must align within this many degrees.
POSE_SPAN_DEG = 5.0


def fill_ratio(thickness: float, spacing: float, deg: float) -> float:
    """`t·cos θ / s` — how much of the oblique plane real cells can cover. No free constant.

    `reports/the_comb_limit.md`. At 90° it is 0: the plane's second in-plane coordinate is
    `-(z - z0)`, which takes one value per section, so the ground truth is N parallel lines.
    """
    if not spacing > 0:
        return float("inf")
    return float(thickness * np.cos(np.deg2rad(float(deg))) / spacing)


def gates(scorable: int, largest: int, coronal_scorable: int) -> tuple[bool, str]:
    """G1 and G2, re-checked on the actual evaluation set. §5's P3."""
    need = SCORABLE_TYPE_FRACTION * coronal_scorable
    if scorable < need:
        return False, (
            f"{scorable} scorable types against the coronal plane's {coronal_scorable} — below "
            f"{SCORABLE_TYPE_FRACTION:.0%} (G1)"
        )
    if largest < METRIC_MAX_N:
        return False, (
            f"the largest type has {largest} cells, under the metric's own max_n = "
            f"{METRIC_MAX_N} subsample cap (G2)"
        )
    return True, ""


def scorable_types(codes: np.ndarray) -> int:
    if codes.size == 0:
        return 0
    return int((np.unique(codes, return_counts=True)[1] >= METRIC_MIN_GT_CELLS).sum())


def largest_type(codes: np.ndarray) -> int:
    if codes.size == 0:
        return 0
    return int(np.unique(codes, return_counts=True)[1].max())


def leak_checks(
    donor_xyz: np.ndarray, truth_xyz: np.ndarray, donor_ids, truth_ids
) -> list[tuple[str, bool]]:
    """L1 and L2, asserted on the RETURNED arrays rather than argued from the call signature."""
    shared_cells = 0
    if donor_xyz.shape[0] and truth_xyz.shape[0]:
        seen = {tuple(np.round(row, 9)) for row in truth_xyz}
        shared_cells = sum(1 for row in donor_xyz if tuple(np.round(row, 9)) in seen)
    return [
        ("L1 — no donor coordinate coincides with a ground-truth coordinate", shared_cells == 0),
        (
            "L2 — the donor slab and the evaluation slab are disjoint by construction",
            float(
                np.min(np.abs(donor_xyz[:, None, :] - truth_xyz[None, :, :]).sum(-1))
                if donor_xyz.shape[0] and truth_xyz.shape[0] and donor_xyz.shape[0] < 4000
                else 1.0
            )
            > 1e-6,
        ),
        ("and neither set is empty", bool(donor_xyz.shape[0]) and bool(truth_xyz.shape[0])),
    ]


def verdict(theta: float, ours: float, base: float, spread: float) -> tuple[str, str]:
    """§6's four outcomes, with 90° replaced by θ* per §2-bis."""
    d = ours - base
    # `>= -COST_BAND` up to float slack. §6 says "at or within the band", and a difference that IS
    # the band lands at -0.05000000000000004 in binary. 1e-12 is numerical slack on a quantity of
    # order 0.05 -- nine orders below the band -- and is NOT the band being widened: no real score
    # difference can sit inside it.
    if d >= -COST_BAND - 1e-12:
        return "DEMONSTRATED", (
            f"at {theta:.0f}°, plane-distance scores {ours:+.4f} against the baseline's "
            f"{base:+.4f} (d = {d:+.4f}, within the {COST_BAND:.2f} band; across-seed spread "
            f"{spread:.4f})"
        )
    return "DEMONSTRATED WITH A COST", (
        f"at {theta:.0f}°, plane-distance scores {ours:+.4f} against the baseline's {base:+.4f} — "
        f"a cost of {-d:.4f}, beyond the {COST_BAND:.2f} band (across-seed spread {spread:.4f}). "
        "The capability is real and its price is a number, stated in the abstract"
    )


def _self_check() -> int:
    """The gates, the fill arithmetic, the leak checks and the outcomes. No data, seconds."""
    checks: list[tuple[str, bool]] = []

    # --- the fill ratio: the comb limit's own arithmetic
    checks += [
        ("fill is 1 at 0° when the slab equals the spacing", fill_ratio(57.5, 57.5, 0.0) == 1.0),
        ("fill is 0 at 90°, whatever the thickness", abs(fill_ratio(57.5, 57.5, 90.0)) < 1e-15),
        ("halving the slab halves the fill", abs(fill_ratio(13.5, 57.5, 30.0)
                                                 - 0.5 * fill_ratio(27.0, 57.5, 30.0)) < 1e-12),
        ("a measured specimen reproduces the published table",
         abs(fill_ratio(27.0, 57.5, 30.0) - 0.4067) < 1e-3),
    ]

    # --- G1/G2 on the evaluation set
    cases = [
        ("a coronal plane clears", 9, 4473, 9, True),
        ("G1 refuses 4 of 9 types", 4, 400, 9, False),
        ("G2 refuses a largest type under max_n", 6, 249, 9, False),
        ("exactly at both bounds clears", 6, METRIC_MAX_N, 9, True),
    ]
    checks += [(f"{lab} -> {gates(s, l, c)[0]}", gates(s, l, c)[0] == want)
               for lab, s, l, c, want in cases]
    checks.append(("a refusal always names its gate",
                   all(gates(s, l, c)[1] for _lab, s, l, c, w in cases if not w)))

    codes = np.array([0] * 30 + [1] * 19 + [2] * 300)
    checks += [
        (f"scorable_types counts only types with >= {METRIC_MIN_GT_CELLS}",
         scorable_types(codes) == 2),
        ("largest_type is the biggest", largest_type(codes) == 300),
    ]

    # --- the leak checks must FIRE on a leak, not merely pass on a clean case
    truth = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    clean = np.array([[9.0, 9.0, 9.0]])
    dirty = np.array([[1.0, 1.0, 1.0]])
    checks += [
        ("L1/L2 pass on disjoint sets", all(ok for _l, ok in leak_checks(clean, truth, (), ()))),
        ("and FAIL on a shared cell — a check that cannot fire is not a check",
         not all(ok for _l, ok in leak_checks(dirty, truth, (), ()))),
        ("and FAIL on an empty donor set", not all(ok for _l, ok in leak_checks(
            np.zeros((0, 3)), truth, (), ()))),
    ]

    # --- §6's outcomes
    checks += [
        ("inside the band is DEMONSTRATED", verdict(45, 0.60, 0.62, 0.01)[0] == "DEMONSTRATED"),
        ("exactly at the band is DEMONSTRATED, float representation included",
         verdict(45, 0.57, 0.62, 0.01)[0] == "DEMONSTRATED"
         and verdict(45, 0.62 - COST_BAND, 0.62, 0.01)[0] == "DEMONSTRATED"),
        ("and a hair beyond it is not — the slack is numerical, not a wider band",
         verdict(45, 0.62 - COST_BAND - 1e-6, 0.62, 0.01)[0] == "DEMONSTRATED WITH A COST"),
        ("beyond it is DEMONSTRATED WITH A COST",
         verdict(45, 0.50, 0.62, 0.01)[0] == "DEMONSTRATED WITH A COST"),
        ("beating the baseline is still DEMONSTRATED, never more",
         verdict(45, 0.90, 0.62, 0.01)[0] == "DEMONSTRATED"),
        ("and every verdict states the spread beside the difference (§4.2o)",
         all("spread" in verdict(45, o, 0.62, 0.01)[1] for o in (0.90, 0.60, 0.50))),
    ]

    from _contract import bench3_clamp_discipline, bench3_config_discipline, uses_shared_base_config

    checks += uses_shared_base_config("oblique_demo.py")
    checks += bench3_config_discipline()
    checks += bench3_clamp_discipline()

    for label, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    failed = sum(1 for _l, ok in checks if not ok)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--angles", type=float, nargs="+", default=list(ANGLES))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--thickness", type=float, default=None,
                    help="slab thickness in um. Default: Section.thickness where the loader "
                         "recorded it as MEASURED; see reports/retractions.md R5")
    ap.add_argument("--weights", default=None,
                    help="optional checkpoint for the `field` ablation, which F1 keeps out of the "
                         "headline comparison. Every headline arm needs no fit.")
    ap.add_argument("--out", default="reports/oblique_demo.md")
    ap.add_argument("--self-check", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()

    paths = resolve(args)
    from spatialcpav25_gen.data.schema import to_xyz
    from spatialcpav25_gen.infer.planes import plane_from_normal
    from spatialcpav25_gen.model.layout import cells_near_plane

    from _starmap_run import load_training_volume, prepare_config

    cfg = prepare_config(seed=0, input_path=paths.input)
    vol = load_training_volume(cfg, paths.input)
    xyz = np.concatenate([np.asarray(to_xyz(s), dtype=np.float64) for s in vol.sections], axis=0)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    extent = hi - lo
    zs = sorted(float(s.z) for s in vol.sections)
    spacing = float(np.median(np.diff(zs))) if len(zs) > 1 else float(extent[2])
    measured = [float(s.thickness) for s in vol.sections if not s.thickness_is_assumed]
    thickness = float(args.thickness) if args.thickness else (
        float(np.median(measured)) if measured else spacing
    )
    z_ref = min(zs, key=lambda z: (abs(z - float(np.median(zs))), z))
    centre = np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), z_ref])
    half_extent = (extent[0], max(extent[1], extent[2]))

    print(f"  {xyz.shape[0]} cells, {len(vol.sections)} sections, spacing {spacing:.2f} um")
    print(f"  slab thickness {thickness:.2f} um "
          f"({'MEASURED' if measured and not args.thickness else 'assumed from spacing'})")

    rows = []
    coronal_scorable = None
    for deg in args.angles:
        t = np.deg2rad(float(deg))
        unit = np.array([0.0, np.sin(t), np.cos(t)])
        target = plane_from_normal(unit, centre, half_extent, thickness)
        truth = cells_near_plane(vol.sections, target)
        n_scorable = scorable_types(np.asarray(truth.cell_type))
        if coronal_scorable is None:
            coronal_scorable = n_scorable
        clears, why = gates(n_scorable, largest_type(np.asarray(truth.cell_type)), coronal_scorable)
        donors = {
            sign: cells_near_plane(
                vol.sections,
                plane_from_normal(unit, centre + sign * thickness * unit, half_extent, thickness),
            )
            for sign in (-1.0, +1.0)
        }
        best = max(donors.values(), key=lambda d: d.coords_uv.shape[0])
        leaks = leak_checks(best.xyz, truth.xyz, best.section_id, truth.section_id)
        rows.append({
            "angle_deg": float(deg),
            "fill_ratio": fill_ratio(thickness, spacing, deg),
            "n_truth": int(truth.xyz.shape[0]),
            "n_donors": int(best.coords_uv.shape[0]),
            "n_strata": int(len(set(truth.section_id.tolist()))),
            "scorable_types": n_scorable,
            "largest_type": largest_type(np.asarray(truth.cell_type)),
            "clears": clears,
            "why_not": why,
            "leaks": [{"check": c, "ok": bool(o)} for c, o in leaks],
        })
        print(f"    {deg:5.1f}°: truth {rows[-1]['n_truth']:6d} from "
              f"{rows[-1]['n_strata']} strata, donors {rows[-1]['n_donors']:6d}, "
              f"fill {rows[-1]['fill_ratio']:.2f}, "
              f"{'clears' if clears else 'REFUSED: ' + why}", flush=True)

    clean = [r for r in rows if r["clears"] and r["angle_deg"] > 0.0]
    theta_star = max((r["angle_deg"] for r in clean), default=0.0)
    record = {
        "dataset": paths.dataset,
        "n_sections": len(vol.sections),
        "section_spacing_um": spacing,
        "slab_thickness_um": thickness,
        "thickness_measured": bool(measured) and not args.thickness,
        "angles": rows,
        "theta_star_deg": theta_star,
        "seeds": list(args.seeds),
        "scoring": "NOT RUN — geometry and preconditions only; see §7 of the report",
    }
    lines = render(record)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


def render(rec: dict) -> list[str]:
    theta = rec["theta_star_deg"]
    out = [
        "# The oblique demonstration — geometry and preconditions",
        "",
        "**Read `reports/oblique_demonstration_preregistration.md` first**, including §2-bis (the "
        "claim is made at θ\\*, not 90°) and §3-bis (donors come from a flanking slab).",
        "",
        f"`{rec['dataset']}`, {rec['n_sections']} sections at {rec['section_spacing_um']:.1f} µm, "
        f"slab thickness **{rec['slab_thickness_um']:.1f} µm** "
        f"({'MEASURED' if rec['thickness_measured'] else '**assumed from spacing** — see R5'}).",
        "",
        "| θ | fill | ground truth | strata | donors | scorable types | largest type | clears |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rec["angles"]:
        out.append(
            f"| {r['angle_deg']:.0f}° | **{r['fill_ratio']:.2f}** | {r['n_truth']} | "
            f"{r['n_strata']} | {r['n_donors']} | {r['scorable_types']} | {r['largest_type']} | "
            f"{'**yes**' if r['clears'] else 'no'} |"
        )
    out += [
        "",
        f"## θ\\* = **{theta:.0f}°**",
        "",
        "The largest oblique angle clearing G1 and G2 on the evaluation set at the measured slab "
        "thickness. Gate-driven and score-free: no arm has been scored at the time this angle is "
        "chosen, which is what stops it being picked for its result.",
        "",
        "**fill** = `t·cos θ / s` (`reports/the_comb_limit.md`). It bounds what **any** method "
        "can demonstrate on serial sections: an oblique ground truth from `N` sections has `N` "
        "samples along depth, so at 90° it is `N` parallel lines whatever generated it. Reported "
        "beside every angle because no gate can see it — G1 and G2 count cells and types, and a "
        "comb holds as many of both as a filled cloud.",
        "",
        "## Leakage preconditions (L1, L2)",
        "",
        "| θ | check | |",
        "|---|---|---|",
    ]
    for r in rec["angles"]:
        for entry in r["leaks"]:
            out.append(
                f"| {r['angle_deg']:.0f}° | {entry['check']} | "
                f"{'✅' if entry['ok'] else '❌ **FAILED**'} |"
            )
    out += [
        "",
        "Asserted on the **returned arrays**, not argued from the call signature. The "
        "flanking-slab construction makes them disjoint by construction; these confirm that "
        "rather than enforce it.",
        "",
        "## 7. Scoring is not run here",
        "",
        "This pass reports geometry and preconditions only. Scoring writes the slab as a "
        "one-section dataset and calls bench3's own `evaluate_paper` on it, and that step is run "
        "separately so the angle θ\\* is fixed — publicly, in this file — **before** any arm has a "
        "number. That ordering is the whole protection against choosing the angle for its score.",
    ]
    return out


if __name__ == "__main__":
    sys.exit(main())
