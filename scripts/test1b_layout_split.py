"""Step 1 — split the layout deficit into PLACEMENT and TYPING.

**Read ``reports/layout_split_preregistration.md`` first.** The arms, the bands, the two
preconditions and four predictions were committed before this script existed.

`paper_celltype_localization` on `layout_mode=field` with a shippable count is **0.5371** against
the model-free copy floor of **0.7765**. That deficit is two mechanisms with different fixes -- the
point process (where cells go) and the mark model (which type is on them) -- and nobody has
separated them.

Two things read out of ``bench3/evaluate_paper.py`` BEFORE the arms were designed, and both
constrain the construction:

* the metric compares per-type point **clouds** by Sinkhorn divergence with **no cell-to-cell
  correspondence**, so a label field moves between point sets by nearest neighbour;
* the pose is aligned **by expression** (``align_by_expression``), so an arm that fabricated or
  dropped expression would be scored at a different pose from the arm it is compared with. Every
  arm therefore carries a coherent expression field, and arms sharing positions share a pose
  exactly. ``align_rotation_deg`` is reported per arm and is a precondition.

Zero fits: ``layout_mode`` is fit-invariant and the r11 arms are provably one fit.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bench3_paths import add_path_args, resolve, set_torch_threads  # noqa: E402

TARGETS = (("section_2", 30.0), ("section_4", 52.0), ("section_6", 74.0))
METRIC = "celltype_localization"

# `reports/r11_probes_recheck.md`, medians over the three held-out sections.
COPY_FLOOR = 0.7765
RESAMPLE_SHIPS = 0.7546
ORACLE = 0.9808

# `reports/layout_split_preregistration.md` §5-§6.
RECOVERED_HIGH = 0.60
RECOVERED_LOW = 0.30
ORACLE_CONTROL_SLACK = 0.05
NULL_CONTROL_MAX = 0.10
MIN_DENOMINATOR = 0.10
MAX_POSE_DIFF_DEG = 5.0


def transfer(src_xyz, dst_xyz, *arrays):
    """Nearest-neighbour transfer of per-cell arrays from one point set to another.

    A cell-type field is a piecewise-constant function of position, so NN is its transfer. The
    metric compares clouds and uses no cell correspondence, so nothing here invents one -- this
    only answers "what type does the tissue have at this location".
    """
    from scipy.spatial import cKDTree

    idx = cKDTree(np.asarray(src_xyz, dtype=np.float64)).query(
        np.asarray(dst_xyz, dtype=np.float64), k=1
    )[1]
    return [np.asarray(a)[idx] for a in arrays]


def verdict(rec_types: float, rec_pos: float) -> tuple[str, str]:
    """The outcome table of ``layout_split_preregistration.md`` §5, in its stated order."""
    hi_t, hi_p = rec_types >= RECOVERED_HIGH, rec_pos >= RECOVERED_HIGH
    lo_t, lo_p = rec_types <= RECOVERED_LOW, rec_pos <= RECOVERED_LOW
    if hi_t and hi_p:
        return (
            "EITHER SUFFICES",
            "each fix alone nearly closes the deficit, so it lives in the joint assignment and "
            "both routes work. A good outcome, not an ambiguous one",
        )
    if hi_t and lo_p:
        return (
            "TYPING",
            "the cells are in the right places and the wrong types are on them. The fix is the "
            "**mark model** -- the per-position categorical -- not the point process",
        )
    if hi_p and lo_t:
        return (
            "PLACEMENT",
            "the typing is sound and the cells are in the wrong places. The fix is the "
            "**intensity field and the sampler**",
        )
    if lo_t and lo_p:
        return (
            "NEITHER SUFFICES",
            "the deficit is genuinely joint and this decomposition does not separate it. The "
            "layout section reports it undivided",
        )
    return (
        "MIXED",
        "neither mechanism dominates; both shares are reported and no single-mechanism claim "
        "may be made",
    )


def preconditions(both_oracle: float, null_types: float, denom: float, poses: dict) -> list[str]:
    """§6. Any failure and the split is NOT READABLE -- returned as the reasons, in order."""
    bad: list[str] = []
    if not np.isfinite(both_oracle) or both_oracle < COPY_FLOOR - ORACLE_CONTROL_SLACK:
        bad.append(
            f"the both-oracle control is {both_oracle:+.4f}, below the copy floor "
            f"{COPY_FLOOR:.4f} by more than {ORACLE_CONTROL_SLACK:.2f}: the nearest-neighbour "
            "transfer is lossy and every arm is attenuated by an unknown amount"
        )
    if not np.isfinite(null_types) or null_types > NULL_CONTROL_MAX:
        bad.append(
            f"the permuted-types null is {null_types:+.4f}, above {NULL_CONTROL_MAX:.2f}: the "
            "metric is not responding to the type-position association this split assumes"
        )
    if not np.isfinite(denom) or denom < MIN_DENOMINATOR:
        bad.append(
            f"the denominator (floor - base) is {denom:+.4f}, below {MIN_DENOMINATOR:.2f}, so "
            "every recovered fraction is unstable"
        )
    for a, b in (("base", "fix_types"), ("fix_positions", "both_oracle")):
        pa, pb = poses.get(a), poses.get(b)
        if pa is not None and pb is not None and abs(pa - pb) > 1e-9:
            bad.append(
                f"{a} and {b} share positions but were aligned at {pa:.3f}° and {pb:.3f}°; they "
                "must be identical or they are not the same pose"
            )
    vals = [p for p in poses.values() if p is not None and np.isfinite(p)]
    if vals and (max(vals) - min(vals)) > MAX_POSE_DIFF_DEG:
        bad.append(
            f"the arms' alignments span {max(vals) - min(vals):.2f}°, above "
            f"{MAX_POSE_DIFF_DEG:.1f}°: the comparison is across poses"
        )
    return bad


def _self_check() -> int:
    """The pre-registered bands and the construction contract. No torch, no data, seconds."""
    cases = [
        ("types recover, positions do not", 0.80, 0.10, "TYPING"),
        ("positions recover, types do not", 0.10, 0.80, "PLACEMENT"),
        ("both recover", 0.80, 0.75, "EITHER SUFFICES"),
        ("neither recovers", 0.10, 0.20, "NEITHER SUFFICES"),
        ("one high, one middling", 0.80, 0.45, "MIXED"),
        ("both middling", 0.45, 0.50, "MIXED"),
        (f"exactly at the high bound {RECOVERED_HIGH}", RECOVERED_HIGH, 0.0, "TYPING"),
        (f"exactly at the low bound {RECOVERED_LOW}", 0.9, RECOVERED_LOW, "TYPING"),
    ]
    results = [(f"{lab} -> {verdict(a, b)[0]}", verdict(a, b)[0] == want) for lab, a, b, want in cases]

    good_poses = {"base": 0.0, "fix_types": 0.0, "fix_positions": 1.0, "both_oracle": 1.0}
    results += [
        ("clean preconditions pass", preconditions(0.98, 0.02, 0.24, good_poses) == []),
        (
            "a lossy NN transfer is caught",
            any("lossy" in r for r in preconditions(0.60, 0.02, 0.24, good_poses)),
        ),
        (
            "a non-null null is caught",
            any("not responding" in r for r in preconditions(0.98, 0.40, 0.24, good_poses)),
        ),
        (
            "a small denominator is caught",
            any("unstable" in r for r in preconditions(0.98, 0.02, 0.05, good_poses)),
        ),
        (
            "arms that share positions but not poses are caught",
            any(
                "same pose" in r
                for r in preconditions(
                    0.98, 0.02, 0.24, {**good_poses, "fix_types": 0.5}
                )
            ),
        ),
        (
            "a wide pose spread across arms is caught",
            any(
                "across poses" in r
                for r in preconditions(
                    0.98, 0.02, 0.24, {**good_poses, "fix_positions": 30.0, "both_oracle": 30.0}
                )
            ),
        ),
    ]
    # The design risk: could `fix_types` be trivially high? Painting the truth's type field onto
    # the model's points would flatter placement if it worked wherever the points are. It does not
    # -- a point set that misses part of the tissue receives only the types found where it IS, and
    # every absent type scores 0. So `fix_types` remains a real test of placement.
    rng_t = np.random.default_rng(0)
    g = rng_t.uniform(0, 1000, size=(3000, 3))
    g_ct = np.where(g[:, 0] < 500, "A", "B")
    m = rng_t.uniform(0, 1000, size=(2000, 3))
    (on_m,) = transfer(g, m, g_ct)
    (on_corner,) = transfer(g, rng_t.uniform(0, 100, size=(500, 3)), g_ct)
    results += [
        (
            f"NN transfer recovers a known spatial type rule "
            f"({np.mean(on_m == np.where(m[:, 0] < 500, 'A', 'B')):.1%} of cells)",
            np.mean(on_m == np.where(m[:, 0] < 500, "A", "B")) > 0.95,
        ),
        (
            "it carries the source's composition rather than inventing one",
            abs(float((on_m == "A").mean()) - float((g_ct == "A").mean())) < 0.05,
        ),
        (
            "and a point set covering only part of the tissue receives only the types found "
            "there — so oracle typing CANNOT hide bad placement, which is the design risk",
            set(on_corner.tolist()) == {"A"},
        ),
    ]
    results += _contract_check()
    for label, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")
    failed = sum(1 for _l, ok in results if not ok)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    return 1 if failed else 0


def _contract_check() -> list[tuple[str, bool]]:
    """Does the pinned evaluator still emit what this runner reads? ``specs/10`` §4.2p.

    Read from the real source with ``ast``/text, so it needs no torch, no data and no fit -- the
    check runs wherever the code is edited rather than only where the data lives.
    """
    from _contract import bench3_clamp_discipline, bench3_config_discipline

    root = Path(__file__).resolve().parent.parent
    ev = (root / "benchmark-pbya-v3/src/bench3/evaluate_paper.py").read_text()
    al = (root / "benchmark-pbya-v3/src/bench3/align.py").read_text()
    return list(bench3_config_discipline()) + list(bench3_clamp_discipline()) + [
        (
            "evaluate_paper still emits `celltype_localization` per section",
            '"celltype_localization"' in ev,
        ),
        (
            "it still aligns BY EXPRESSION, which is why every arm carries one",
            "align_by_expression(\n" in ev or "align_by_expression(pred_xy" in ev,
        ),
        ("the pose is still reported as `align_rotation_deg`", '"align_rotation_deg"' in al),
        (
            "a type with too few predicted cells still scores 0, so composition still counts",
            "min_pred_cells" in ev and "scores.append(0.0)" in ev,
        ),
        (
            "the metric still uses no cell correspondence, so NN transfer is still its vehicle",
            "_sinkhorn_divergence" in ev,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", help="the r11 checkpoint; no fit is run")
    ap.add_argument("--layout-mode", default="field", choices=("field", "hybrid", "resample"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="repeat the WHOLE split at each generation seed. Gives a WITHIN-section "
        "across-seed spread to set against the across-SECTION spread, which is what "
        "the positional-instability observation needs and what retires the one-seed "
        "caveat standing on every field-layout number since R11.",
    )
    ap.add_argument("--out", default="reports/test1b_layout_split.md")
    ap.add_argument("--self-check", action="store_true")
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not args.weights:
        ap.error("--weights is required: this re-generates on an existing fit, never fits")
    paths = resolve(args)
    set_torch_threads()

    import anndata as ad
    import torch
    from spatialcpav25_gen.config import Config
    from spatialcpav25_gen.infer.generate import generate_section, plane_at_z
    from spatialcpav25_gen.model.field import BBoxClampWarning

    from _starmap_run import load_training_volume
    from test1_field_count import build_model, flanking_density_count, score, write_prediction

    checkpoint = torch.load(args.weights, map_location="cpu")
    cfg = Config(**checkpoint["config"]).replace(
        layout_mode=args.layout_mode, layout_sampler="grid"
    )
    vol = load_training_volume(cfg, paths.input)
    model = build_model(checkpoint, cfg, vol, args.seed)
    genes = [str(g) for g in vol.gene_names]
    types = [str(t) for t in vol.celltype_names]

    gt = ad.read_h5ad(paths.ground_truth)
    gt_sections = gt.obs["section"].values.astype(str)
    gt_spatial = np.asarray(gt.obsm["spatial"], dtype=np.float64)

    seeds = [int(x) for x in (args.seeds or [args.seed])]
    per_seed: dict[int, dict] = {}
    for gen_seed in seeds:
      arms: dict[str, dict] = {k: {} for k in
                             ("base", "fix_types", "fix_positions", "both_oracle", "null_types")}
      for name, z in TARGETS:
          plane = plane_at_z(vol, z, cfg)
          n_target = flanking_density_count(vol, plane, z)
          with warnings.catch_warnings():
              warnings.simplefilter("ignore", BBoxClampWarning)
              warnings.simplefilter("ignore")
              with torch.no_grad():
                  emitted = generate_section(model, plane, vol, cfg, gen_seed, n_target=n_target)
          mx = emitted.X
          m_X = np.asarray(mx.toarray() if sp.issparse(mx) else mx, dtype=np.float32)
          m_xyz = np.asarray(emitted.obsm["xyz"], dtype=np.float64)
          m_ct = np.asarray(emitted.obs[cfg.celltype_key].values, dtype=str)

          gm = gt_sections == name
          g_xyz = gt_spatial[gm]
          g_ct = gt.obs["cell_type"].values[gm].astype(str)
          gx = gt.X[gm]
          g_X = np.asarray(gx.toarray() if sp.issparse(gx) else gx, dtype=np.float32)

          # `flanking_copy`'s own donor: the nearest TRAINING section, emitted verbatim.
          src = min(vol.sections, key=lambda s: (abs(float(s.z) - z), str(s.section_id)))
          c_xyz = np.asarray(
              np.column_stack([np.asarray(src.coords, dtype=np.float64)[:, :2],
                               np.full(len(src.coords), float(src.z))])
          )
          cx = src.counts
          c_X = np.asarray(cx.toarray() if sp.issparse(cx) else cx, dtype=np.float32)
          c_ct = np.asarray([types[int(i)] for i in np.asarray(src.cell_type)], dtype=str)

          (gt_on_model,) = transfer(g_xyz, m_xyz, g_ct)
          (model_on_copy,) = transfer(m_xyz, c_xyz, m_ct)
          (gt_on_copy,) = transfer(g_xyz, c_xyz, g_ct)
          permuted = m_ct[np.random.default_rng(gen_seed).permutation(len(m_ct))]

          for key, (X, xyz, ct) in {
              "base": (m_X, m_xyz, m_ct),
              "fix_types": (m_X, m_xyz, gt_on_model),
              "fix_positions": (c_X, c_xyz, model_on_copy),
              "both_oracle": (c_X, c_xyz, gt_on_copy),
              "null_types": (m_X, m_xyz, permuted),
          }.items():
              arms[key][name] = {
                  "X": sp.csr_matrix(X), "coords": xyz, "cell_type": ct
              }
          print(f"  {name}: model {len(m_ct)} cells, copy {len(c_ct)} ({src.section_id}), "
                f"gt {len(g_ct)}", flush=True)

      out: dict[str, dict] = {}
      for key, per_section in arms.items():
          tmp = Path(args.out).with_suffix(f".{key}.pred.h5ad")
          write_prediction(per_section, genes, str(tmp), gen_seed)
          scored = score(str(tmp), paths.ground_truth)
          vals, poses = {}, {}
          for name, _z in TARGETS:
              sec = scored["per_section"][name]
              vals[name] = float(sec[METRIC])
              poses[name] = float(sec.get("align_rotation_deg", float("nan")))
          out[key] = {
              "per_section": vals,
              "median": float(np.median(list(vals.values()))),
              "pose_deg": float(np.median(list(poses.values()))),
              "n_cells": {n: int(arms[key][n]["coords"].shape[0]) for n, _z in TARGETS},
          }
          print(f"  {key}: median {out[key]['median']:+.4f}  pose {out[key]['pose_deg']:.3f}°",
                flush=True)

      per_seed[gen_seed] = out
      print(f"  --- seed {gen_seed} done ---", flush=True)

    return _finish(per_seed, seeds, args)


MODEL_POSITION_ARMS = ("base", "fix_types", "null_types")
COPY_POSITION_ARMS = ("fix_positions", "both_oracle")


def spread_table(per_seed: dict, seeds: list[int]) -> dict:
    """Across-SECTION spread against across-SEED spread, per arm.

    The observation this exists to test, stated in `reports/layout_split_preregistration.md`'s
    review: every arm using the MODEL's positions swings ~0.45 across sections, every arm using the
    COPY's is flat to ~0.02. If that is positional instability rather than section-to-section
    biology, the across-seed spread of the model-position arms will be **small** beside it — the
    field is unstable *per plane*, not noisy per draw.

    Reported, never verdicted: the observation was read off a table this project had already seen,
    so nothing here claims it. It is the number a pre-registered test would be built on.
    """
    out: dict[str, dict] = {}
    for arm in MODEL_POSITION_ARMS + COPY_POSITION_ARMS:
        by_section = {n: [per_seed[s][arm]["per_section"][n] for s in seeds]
                      for n, _z in TARGETS}
        across_section = [float(np.median(v)) for v in by_section.values()]
        out[arm] = {
            "positions": "model" if arm in MODEL_POSITION_ARMS else "copy",
            "across_section_spread": float(max(across_section) - min(across_section)),
            "across_seed_spread": {n: float(max(v) - min(v)) for n, v in by_section.items()},
            "max_across_seed_spread": float(max(max(v) - min(v) for v in by_section.values())),
        }
    return out


def _finish(per_seed: dict, seeds: list[int], args) -> int:
    """Collapse the seeds, apply the pre-registered bands, write the report."""
    out: dict[str, dict] = {}
    for arm in per_seed[seeds[0]]:
        vals = {n: float(np.median([per_seed[s][arm]["per_section"][n] for s in seeds]))
                for n, _z in TARGETS}
        out[arm] = {
            "per_section": vals,
            "median": float(np.median(list(vals.values()))),
            "pose_deg": float(np.median([per_seed[s][arm]["pose_deg"] for s in seeds])),
            "n_cells": per_seed[seeds[0]][arm]["n_cells"],
        }

    base = out["base"]["median"]
    denom = COPY_FLOOR - base
    rec_t = (out["fix_types"]["median"] - base) / denom if denom else float("nan")
    rec_p = (out["fix_positions"]["median"] - base) / denom if denom else float("nan")
    rec_b = (out["both_oracle"]["median"] - base) / denom if denom else float("nan")
    bad = preconditions(
        out["both_oracle"]["median"], out["null_types"]["median"], denom,
        {k: v["pose_deg"] for k, v in out.items()},
    )
    name, why = ("NOT READABLE", "; ".join(bad)) if bad else verdict(rec_t, rec_p)
    spreads = spread_table(per_seed, seeds) if len(seeds) > 1 else {}

    # Per section, because at n = 3 the median IS a section and the three can disagree in SIGN.
    per_sec = {}
    for n, _z in TARGETS:
        b = out["base"]["per_section"][n]
        d = COPY_FLOOR - b
        per_sec[n] = {
            "base": b, "deficit": d,
            "recovered_types": (out["fix_types"]["per_section"][n] - b) / d if d else float("nan"),
            "recovered_positions": (out["fix_positions"]["per_section"][n] - b) / d if d else float("nan"),
            "recovered_both": (out["both_oracle"]["per_section"][n] - b) / d if d else float("nan"),
        }
        per_sec[n]["verdict"] = verdict(
            per_sec[n]["recovered_types"], per_sec[n]["recovered_positions"]
        )[0]

    lines = render(out, base, denom, rec_t, rec_p, rec_b, per_sec, spreads, seeds, name, why, bad, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"arms": out, "per_section": per_sec, "spreads": spreads, "seeds": seeds,
         "recovered_types": rec_t, "recovered_positions": rec_p, "recovered_both": rec_b,
         "verdict": name, "why": why, "preconditions_failed": bad}, indent=2, default=float))
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


def render(out, base, denom, rec_t, rec_p, rec_b, per_sec, spreads, seeds, name, why, bad, args) -> list[str]:
    lines = [
        "# Step 1 — splitting the layout deficit into placement and typing",
        "",
        "**Read `reports/layout_split_preregistration.md` first.** The arms, bands, preconditions",
        "and four predictions were committed before this ran.",
        "",
        f"`layout_mode={args.layout_mode}`, grid sampler, shippable cell count, seed {args.seed},",
        f"weights `{args.weights}`. **Zero fits.** Scored through the pinned `evaluate_paper`.",
        "",
        "Every arm carries a coherent expression field matched to its own positions, because the",
        "pose is aligned **by expression** — arms that share positions share a pose exactly, and",
        "`align_rotation_deg` is a precondition rather than a footnote.",
        "",
        "| arm | positions | types | section_2 | section_4 | section_6 | **median** | pose |",
        "|---|---|---|---|---|---|---|---|",
    ]
    labels = {
        "base": ("model", "model"),
        "fix_types": ("model", "**ground truth**"),
        "fix_positions": ("**the copy's**", "model"),
        "both_oracle": ("the copy's", "ground truth"),
        "null_types": ("model", "*permuted*"),
    }
    for key, (pos, ty) in labels.items():
        r = out[key]
        ps = r["per_section"]
        lines.append(
            f"| `{key}` | {pos} | {ty} | {ps['section_2']:+.4f} | {ps['section_4']:+.4f} "
            f"| {ps['section_6']:+.4f} | **{r['median']:+.4f}** | {r['pose_deg']:.3f}° |"
        )
    lines += [
        "",
        f"Reference: `flanking_copy` **{COPY_FLOOR:.4f}**, `resample` {RESAMPLE_SHIPS:.4f}, "
        f"`oracle` {ORACLE:.4f}.",
        "",
        "## The split",
        "",
        f"`base` = **{base:+.4f}**, denominator `floor - base` = **{denom:+.4f}**.",
        "",
        "| | recovered fraction of the deficit |",
        "|---|---|",
        f"| fixing the **types** (`fix_types`) | **{rec_t:.1%}** |",
        f"| fixing the **positions** (`fix_positions`) | **{rec_p:.1%}** |",
        f"| sum — its distance from 100% is the **interaction** | {rec_t + rec_p:.1%} |",
        f"| fixing **both** (`both_oracle`) | **{rec_b:.1%}** |",
        "",
        "⚠️ **The median at n = 3 IS a section**, and the three can disagree in sign. The verdict",
        "above is the pre-registered one; the table below is why it is not the whole story.",
        "",
        "| section | base | deficit | types | positions | both | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, _z in TARGETS:
        r = per_sec[n]
        lines.append(
            f"| {n} | {r['base']:+.4f} | {r['deficit']:.4f} | {r['recovered_types']:.1%} "
            f"| {r['recovered_positions']:.1%} | {r['recovered_both']:.1%} | {r['verdict']} |"
        )
    lines += [""]
    if spreads:
        lines += [
            "## Across sections against across seeds",
            "",
            f"Seeds: {seeds}. **Reported, not verdicted** — the observation this tests was read off",
            "a table already seen, so nothing here claims it.",
            "",
            "| arm | positions | across-section spread | worst across-seed spread | ratio |",
            "|---|---|---|---|---|",
        ]
        for arm, r in spreads.items():
            ratio = (r["across_section_spread"] / r["max_across_seed_spread"]
                     if r["max_across_seed_spread"] > 0 else float("inf"))
            lines.append(
                f"| `{arm}` | {r['positions']} | {r['across_section_spread']:.4f} "
                f"| {r['max_across_seed_spread']:.4f} | {ratio:.1f}x |"
            )
        lines += [""]
    if bad:
        lines += ["## **NOT READABLE**", "", "Preconditions failed (§6):", ""]
        lines += [f"- {b}" for b in bad]
        lines += ["", "The split is not reported. The preconditions were committed before the run."]
    else:
        lines += [f"## **{name}**", "", why + "."]
    lines += [
        "",
        (
            f"✅ **{len(seeds)} generation seeds** — the one-seed caveat that stood on every "
            "field-layout number since R11 is retired for this measurement."
            if len(seeds) > 1
            else "⚠️ **One seed, one fit.** No across-seed spread exists for this metric on the "
            "field arms (`envelope_correction.md` §3). Raw deficits only; **no "
            "multiple-of-envelope may be quoted.** Pass `--seeds 1 2 3` to retire this."
        ),
        "",
        "⚠️ This splits **one metric at axis-aligned planes**. It says nothing about oblique planes",
        "— see `reports/oblique_layout_cost.md` for why those need a different evaluation set.",
    ]
    return lines


if __name__ == "__main__":
    sys.exit(main())
