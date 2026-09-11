"""Test 1 — the field layout with the cell count supplied from outside the intensity integral.

**Read ``reports/reframing_tests_preregistration.md`` §1 before reading any number this emits.**
The outcome bands, the two count sources and which of them governs were committed before this
script existed.

What it is for, in one paragraph. The copy-based-field framing's novelty claim is **arbitrary
orientations**, and the shipped layout cannot produce one: ``_resample_layout`` picks its donor by
``abs(f.z - plane.origin[2])`` -- a *z*-distance, meaningless for a plane spanning the stack -- and
pastes that section's in-plane coordinates onto the new plane. At an oblique angle it does not
crash; it emits a coronal point pattern relabelled as oblique. So the field layout is the only
candidate, R11 measured it **below the model-free copy floor** on the metric it exists to win, and
R11 also localised its defect to the intensity integral's *scale* rather than its *pattern*. This
separates the two. **It is the existence test for the framing, not an ablation rescue.**

Zero fits. ``layout_mode`` is in ``FIT_INVARIANT_GATES`` and the r11 checkpoint recovery proved all
five r11 arms are one fit differing only in generation-time gates
(``test_layout_mode_does_not_enter_the_fit``, bitwise across all 96 tensors), so this re-generates
and re-scores on weights that already exist.
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

# `reports/reframing_tests_preregistration.md` §1, and `reports/r11_probes_recheck.md` for the
# reference points. Quoted here so the bands cannot drift from the file that fixed them.
METRIC = "paper_celltype_localization"
FIELD_BROKEN_INTEGRAL = 0.6607
RESAMPLE_SHIPS = 0.7546
COPY_FLOOR = 0.7765
ORACLE = 0.9808


def band(value: float, spread: float, oracle_value: float | None) -> tuple[str, str]:
    """The pre-registered outcome. ``value`` is the SHIPPABLE arm's -- §1 says 1b governs."""
    if oracle_value is not None and np.isfinite(oracle_value) and np.isfinite(spread):
        if abs(oracle_value - value) > spread:
            return (
                "UNINFORMATIVE",
                f"the ground-truth-count arm ({oracle_value:+.4f}) and the flanking-density arm "
                f"({value:+.4f}) differ by {abs(oracle_value - value):.4f}, more than the "
                f"across-section spread of {spread:.4f}. The count source is doing the work "
                "rather than the layout, so there is no reading",
            )
    if value >= COPY_FLOOR:
        return (
            "CONFIRMED",
            f"{value:+.4f} clears the model-free copy floor of {COPY_FLOOR:.4f}. The diagnosis "
            "holds, the framing has a mechanism for arbitrary orientations, and claim 2 becomes "
            "makeable",
        )
    if value >= RESAMPLE_SHIPS:
        return (
            "PARTIAL",
            f"{value:+.4f} matches the shipped layout ({RESAMPLE_SHIPS:.4f}) but does not clear "
            f"the copy floor ({COPY_FLOOR:.4f}). Claim 2 may be made ONLY in the weak form: "
            "oblique generation is possible at the quality of axis-aligned generation, which is "
            "itself below a copy",
        )
    return (
        "REFUTED",
        f"{value:+.4f} is below the shipped layout's {RESAMPLE_SHIPS:.4f}. The count was not the "
        "defect. **No configuration generates a meaningful oblique section, and claim 2 cannot be "
        "made at all** -- the framing loses its novelty claim and the paper returns to the "
        "negative result of reports/diagnostic_programme_closed.md",
    )



def _self_check() -> int:
    """The pre-registered bands, asserted against `reframing_tests_preregistration.md` §1.

    `band()` IS the pre-registration in code, so it gets the same treatment every other criterion
    in this campaign gets: cases fixed by the document, including the boundaries and the
    UNINFORMATIVE rule that stops the oracle-fed arm standing in for the shippable one.
    """
    cases = [
        ("clears the copy floor", 0.80, 0.02, None, "CONFIRMED"),
        ("matches resample, below the floor", 0.76, 0.02, None, "PARTIAL"),
        ("below resample", 0.70, 0.02, None, "REFUTED"),
        (f"exactly at the floor {COPY_FLOOR}", COPY_FLOOR, 0.02, None, "CONFIRMED"),
        (f"exactly at resample {RESAMPLE_SHIPS}", RESAMPLE_SHIPS, 0.02, None, "PARTIAL"),
        ("the arms disagree by more than the spread", 0.70, 0.02, 0.80, "UNINFORMATIVE"),
        ("the arms agree within the spread", 0.70, 0.05, 0.72, "REFUTED"),
        ("an oracle arm alone cannot CONFIRM", 0.70, 0.01, 0.90, "UNINFORMATIVE"),
        ("the broken integral's own value still refutes", FIELD_BROKEN_INTEGRAL, 0.02, None,
         "REFUTED"),
    ]
    failed = 0
    for label, value, spread, oracle_value, want in cases:
        got, why = band(value, spread, oracle_value)
        ok = got == want
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {label:44s} -> {got}")
        if not ok:
            print(f"       wanted {want}; reason given: {why}")
    print(f"\n{len(cases) - failed}/{len(cases)} band cases correct")
    return 1 if failed else 0


def flanking_density_count(volume, plane, z: float) -> int:
    """Cells the two flanking sections' measured density implies for this plane. **Shippable.**

    The mean of the flanking sections' in-plane densities times the plane's area. It uses no
    information about the target section, which is what makes it the arm the bands are read on.
    """
    zs = sorted({float(np.median(np.asarray(s.coords)[:, 1] * 0 + s.z)) for s in volume.sections})
    near = sorted(volume.sections, key=lambda s: abs(float(s.z) - z))[:2]
    if not near:
        raise SystemExit("flanking_density_count: the training volume has no sections")
    densities = []
    for sec in near:
        uv = np.asarray(sec.coords, dtype=np.float64)[:, :2]
        span = uv.max(axis=0) - uv.min(axis=0)
        area = float(span[0] * span[1])
        if not area > 0:
            raise SystemExit(f"flanking_density_count: section {sec.section_id!r} has zero area")
        densities.append(uv.shape[0] / area)
    del zs
    return max(1, int(round(float(np.mean(densities)) * float(plane.area))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", help="the r11 checkpoint; no fit is run")
    ap.add_argument(
        "--count-from",
        choices=("flanking-density", "ground-truth", "both"),
        default="both",
        help="flanking-density is the SHIPPABLE arm and governs every band; ground-truth is an "
        "ORACLE input, reported as an upper bound and never quotable as the method's score",
    )
    ap.add_argument("--layout-mode", default="field", choices=("field", "hybrid", "resample"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="reports/test1_field_count.md")
    ap.add_argument(
        "--self-check",
        action="store_true",
        help="assert the pre-registered bands on synthetic values. No data, no fit, seconds.",
    )
    add_path_args(ap)
    args = ap.parse_args(argv)
    if args.self_check:
        return _self_check()
    if not args.weights:
        ap.error("--weights is required: this script re-generates on an existing fit, never fits")
    paths = resolve(args)
    set_torch_threads()

    import torch
    from spatialcpav25_gen.config import Config
    from spatialcpav25_gen.infer.generate import generate_section, plane_at_z
    from spatialcpav25_gen.model.field import BBoxClampWarning
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData

    from _starmap_run import load_training_volume

    # The same load the chain diagnostic's --load-model does: the checkpoint carries its own
    # Config, and only the two generation-time gates are overridden. A fit is never run.
    checkpoint = torch.load(args.weights, map_location="cpu")
    cfg = Config(**checkpoint["config"]).replace(
        layout_mode=args.layout_mode, layout_sampler="grid"
    )
    vol = load_training_volume(cfg, paths.input)
    model = build_model(checkpoint, cfg, vol, CTFFlow, TrainingData)

    truth = ground_truth_counts(paths.ground_truth)
    arms = (
        ("flanking-density", "1b — SHIPPABLE, governs"),
        ("ground-truth", "1a — ORACLE input, upper bound only"),
    )
    wanted = [a for a in arms if args.count_from in (a[0], "both")]

    results: dict[str, dict] = {}
    for source, label in wanted:
        per_section: dict[str, float] = {}
        counts: dict[str, tuple[int, int]] = {}
        preds: dict[str, dict] = {}
        for name, z in TARGETS:
            plane = plane_at_z(vol, z, cfg)
            n_target = (
                truth[name] if source == "ground-truth" else flanking_density_count(vol, plane, z)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", BBoxClampWarning)
                warnings.simplefilter("ignore")
                with torch.no_grad():
                    emitted = generate_section(
                        model, plane, vol, cfg, args.seed, n_target=n_target
                    )
            x = emitted.X
            x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)
            preds[name] = {
                "X": sp.csr_matrix(x),
                "coords": np.asarray(emitted.obsm["xyz"], dtype=np.float64),
                "cell_type": np.asarray(emitted.obs[cfg.celltype_key].values, dtype=str),
            }
            counts[name] = (int(emitted.n_obs), truth[name])
            print(f"  {source} {name}: asked {n_target}, placed {emitted.n_obs}, "
                  f"ground truth {truth[name]}", flush=True)
        tmp = Path(args.out).with_suffix(f".{source}.pred.h5ad")
        write_prediction(preds, [str(g) for g in vol.gene_names], str(tmp), args.seed)
        scored = score(str(tmp), paths.ground_truth)
        key = METRIC.replace("paper_", "")
        for name, _z in TARGETS:
            sec = scored.get("per_section", {}).get(name)
            if not isinstance(sec, dict) or key not in sec:
                raise SystemExit(
                    f"evaluate_paper returned no {key!r} for {name}; it emitted "
                    f"{sorted(sec) if isinstance(sec, dict) else type(sec).__name__}"
                )
            per_section[name] = float(sec[key])
        vals = [per_section[n] for n, _ in TARGETS]
        results[source] = {
            "label": label,
            "per_section": per_section,
            "median": float(np.median(vals)),
            "spread": float(max(vals) - min(vals)),
            "counts": counts,
        }

    ship = results.get("flanking-density")
    oracle_arm = results.get("ground-truth")
    if ship is None:
        verdict, why = ("NOT READ", "the shippable arm was not run, and §1 says it governs")
    else:
        verdict, why = band(
            ship["median"],
            ship["spread"],
            None if oracle_arm is None else oracle_arm["median"],
        )

    lines = render(results, verdict, why, args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"results": results, "verdict": verdict, "why": why}, indent=2, default=float)
    )
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0



def build_model(checkpoint, cfg, vol, CTFFlow, TrainingData):
    """Rebuild the fitted model from the checkpoint. No fit, no re-derivation of embeddings."""
    # `build_embeddings` lives in the chain diagnostic; four other scripts already
    # import it from there for exactly this reason.
    from t10_chain_diagnostic import build_embeddings

    data = TrainingData(
        specimen_id=vol.specimen_id,
        sections=vol.sections,
        gene_names=vol.gene_names,
        celltype_names=vol.celltype_names,
        region_names=vol.region_names,
        flattened_sections=vol.flattened_sections,
    )
    model = CTFFlow(cfg, data, build_embeddings(cfg, vol, for_checkpoint=True), grf_seed=1)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def write_prediction(per_section: dict, gene_names: list[str], path: str, seed: int) -> None:
    """Through the wrappers' own ``_v2_io`` writer, so the pinned evaluator sees a real prediction."""
    import _v2_io

    _v2_io.write_prediction_h5(
        per_section, gene_names, list(per_section), {"seed": seed}, 0.0, path, "spatialcpav25_gen"
    )


def score(path: str, ground_truth) -> dict:
    """``bench3.evaluate_paper`` -- the pinned instrument, the one every comparable number uses."""
    from bench3.evaluate_paper import evaluate_paper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_paper(str(path), str(ground_truth), use_umap=False)


def render(results: dict, verdict: str, why: str, args) -> list[str]:
    out = [
        "# Test 1 — the field layout with the cell count supplied externally",
        "",
        "**Read `reports/reframing_tests_preregistration.md` §1 first.** The bands, the two count",
        "sources and the rule that the shippable one governs were committed before this ran.",
        "",
        "This is the **existence test** for the copy-based-field framing's novelty claim, not an",
        "ablation rescue: `_resample_layout` cannot produce a meaningful oblique section (it pastes",
        "a coronal point pattern onto the new plane), so if the field layout cannot be made to work",
        "then no configuration generates one and claim 2 cannot be made at all.",
        "",
        f"`layout_mode={args.layout_mode}`, grid sampler, seed {args.seed}, weights "
        f"`{args.weights}`. **Zero fits** — `layout_mode` is fit-invariant.",
        "",
        "| reference | `paper_celltype_localization` |",
        "|---|---|",
        f"| `field`, count from the broken integral | {FIELD_BROKEN_INTEGRAL:.4f} |",
        f"| `resample` (ships) | {RESAMPLE_SHIPS:.4f} |",
        f"| **`flanking_copy` — the model-free floor** | **{COPY_FLOOR:.4f}** |",
        f"| `oracle` | {ORACLE:.4f} |",
        "",
        "| arm | section_2 | section_4 | section_6 | **median** | spread |",
        "|---|---|---|---|---|---|",
    ]
    for source, r in results.items():
        ps = r["per_section"]
        out.append(
            f"| {r['label']} | {ps['section_2']:+.4f} | {ps['section_4']:+.4f} "
            f"| {ps['section_6']:+.4f} | **{r['median']:+.4f}** | {r['spread']:.4f} |"
        )
    out += [
        "",
        "| arm | cells placed / ground truth |",
        "|---|---|",
    ]
    for source, r in results.items():
        cells = ", ".join(f"{n}: {a}/{b}" for n, (a, b) in r["counts"].items())
        out.append(f"| {r['label']} | {cells} |")
    out += [
        "",
        "🚩 **The ground-truth-count arm is an ORACLE input.** It is an upper bound and may not be",
        "quoted as the method's score; §1 fixes that the flanking-density arm governs every band.",
        "",
        f"## **{verdict}**",
        "",
        why + ".",
        "",
        "⚠️ **One seed, and no across-seed spread exists for this metric on the field arms** —",
        "the r11 arms are one fit and one seed (`envelope_correction.md` §3). Bands are read on raw",
        "deficits and **no multiple-of-envelope may be quoted**, exactly as the A4 row was",
        "corrected to do.",
    ]
    return out


def ground_truth_counts(ground_truth) -> dict[str, int]:
    import anndata as ad

    gt = ad.read_h5ad(ground_truth, backed="r")
    try:
        sections = gt.obs["section"].values.astype(str)
        return {s: int((sections == s).sum()) for s, _z in TARGETS}
    finally:
        gt.file.close()


if __name__ == "__main__":
    sys.exit(main())
