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




def _contract_check() -> list[tuple[str, bool]]:
    """Do the constructors this runner calls actually have the signatures it uses?

    **This is the check that was missing.** ``--self-check`` was 9/9 and exercised ``band()`` only,
    so the model-construction path had no coverage and the run died on
    ``TrainingData(specimen_id=...)`` — an argument that constructor has never taken. Same shape as
    the sidecar crashing twice after every measurement was paid for: a self-check that verifies the
    code path it exercises rather than the one that fails.

    It reads the **real definitions** out of the source with ``ast``, so it needs no torch, no data
    and no fit, and it runs in this container as well as on the campaign machine. That matters: a
    smoke test that only runs where the data lives would not have caught this before the run.
    """
    import ast

    from _contract import bench3_clamp_discipline, bench3_config_discipline

    root = Path(__file__).resolve().parent.parent
    # A signature check cannot catch a config field left at a default the dataset cannot satisfy;
    # this one asserts every bench3 runner prepares its Config the way the working ones do.
    checks: list[tuple[str, bool]] = list(bench3_config_discipline())
    checks += bench3_clamp_discipline()

    def parse(rel: str) -> ast.Module:
        return ast.parse((root / rel).read_text())

    core = parse("spatialcpav25_gen/model/spatialcpav25_gen.py")
    classes = {n.name: n for n in ast.walk(core) if isinstance(n, ast.ClassDef)}

    td = classes.get("TrainingData")
    td_methods = {n.name: n for n in (td.body if td else []) if isinstance(n, ast.FunctionDef)}
    td_fields = [
        n.target.id
        for n in (td.body if td else [])
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    ]
    build = td_methods.get("build")
    build_args = [a.arg for a in build.args.args] if build else []
    checks += [
        ("TrainingData exists", td is not None),
        ("it is built by the `build` CLASSMETHOD, not by keyword", build is not None),
        (
            "`build` takes (cls, vol, cfg) — the call this runner makes",
            build_args[:3] == ["cls", "vol", "cfg"],
        ),
        (
            "and it does NOT take `specimen_id` — the argument the first version passed, which "
            "is TrainingVolume's field list, not TrainingData's",
            "specimen_id" not in td_fields and "specimen_id" not in build_args,
        ),
    ]

    ctf = classes.get("CTFFlow")
    init = next(
        (n for n in (ctf.body if ctf else []) if isinstance(n, ast.FunctionDef)
         and n.name == "__init__"),
        None,
    )
    pos = [a.arg for a in init.args.args] if init else []
    kw = [a.arg for a in init.args.kwonlyargs] if init else []
    checks += [
        ("CTFFlow.__init__ takes (cfg, data, embeddings) positionally", pos[:4] == ["self", "cfg", "data", "embeddings"]),
        ("and `grf_seed` as a keyword", "grf_seed" in kw),
    ]

    chain = parse("scripts/t10_chain_diagnostic.py")
    be = next(
        (n for n in ast.walk(chain) if isinstance(n, ast.FunctionDef)
         and n.name == "build_embeddings"),
        None,
    )
    be_kw = [a.arg for a in be.args.kwonlyargs] if be else []
    checks += [
        ("build_embeddings takes (cfg, vol)", [a.arg for a in be.args.args][:2] == ["cfg", "vol"]),
        ("and `for_checkpoint`, which is the branch this runner needs", "for_checkpoint" in be_kw),
    ]

    # The checkpoint key, read from every site that WRITES one rather than assumed.
    written: set[str] = set()
    for rel in ("scripts/t10_chain_diagnostic.py", "scripts/t09_ship_starmap.py"):
        for node in ast.walk(parse(rel)):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "save"
                and node.args
                and isinstance(node.args[0], ast.Dict)
            ):
                written |= {
                    k.value for k in node.args[0].keys if isinstance(k, ast.Constant)
                }
    checks += [
        ("every checkpoint writer uses the key `state_dict`", "state_dict" in written),
        ("and none writes `model`, which the first version read", "model" not in written),
    ]

    layout = parse("spatialcpav25_gen/model/layout.py")
    sl = next(
        (n for n in ast.walk(layout) if isinstance(n, ast.FunctionDef) and n.name == "sample_layout"),
        None,
    )
    sl_kw = [a.arg for a in sl.args.kwonlyargs] if sl else []
    src_sl = ast.get_source_segment((root / "spatialcpav25_gen/model/layout.py").read_text(), sl) or ""
    checks += [
        ("sample_layout takes `n_target` — this whole test depends on it", "n_target" in sl_kw),
        (
            "and it RAISES without fitted repulsion, which is why build_model must fit it",
            "fit_repulsion" in src_sl,
        ),
    ]

    gen = parse("spatialcpav25_gen/infer/generate.py")
    gs = next(
        (n for n in ast.walk(gen) if isinstance(n, ast.FunctionDef) and n.name == "generate_section"),
        None,
    )
    checks.append(
        ("generate_section passes `n_target` through", "n_target" in [a.arg for a in gs.args.kwonlyargs])
    )
    return checks


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
    results: list[tuple[str, bool]] = []
    for label, value, spread, oracle_value, want in cases:
        got, why = band(value, spread, oracle_value)
        results.append((f"{label} -> {got}", got == want))
        if got != want:
            print(f"       wanted {want}; reason given: {why}")

    print("the pre-registered bands:")
    for label, ok in results:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")

    print("\nthe construction contract (read from the real definitions, no torch, no data):")
    contract = _contract_check()
    for label, ok in contract:
        print(f"  {'ok  ' if ok else 'FAIL'} {label}")

    everything = results + contract
    failed = sum(1 for _l, ok in everything if not ok)
    print(f"\n{len(everything) - failed}/{len(everything)} checks passed")
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
    from _starmap_run import load_training_volume

    # The same load the chain diagnostic's --load-model does: the checkpoint carries its own
    # Config, and only the two generation-time gates are overridden. A fit is never run.
    checkpoint = torch.load(args.weights, map_location="cpu")
    cfg = Config(**checkpoint["config"]).replace(
        layout_mode=args.layout_mode, layout_sampler="grid"
    )
    vol = load_training_volume(cfg, paths.input)
    model = build_model(checkpoint, cfg, vol, args.seed)

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



def build_model(checkpoint, cfg, vol, seed: int):
    """Rebuild the fitted model from the checkpoint. No fit.

    Mirrors ``t10_chain_diagnostic``'s ``--load-model`` branch exactly, which is the only path in
    this project that reconstructs a model from a checkpoint. **Four things here were wrong in the
    first version** and every one of them is a real constructor detail, not a style choice:

    * ``TrainingData`` is a dataclass of ``(vol, index, counts, total_counts, stats)`` built by the
      ``build`` **classmethod**. The first version called it with ``specimen_id=``/``sections=``,
      which is ``TrainingVolume``'s field list.
    * the checkpoint key is ``"state_dict"``, not ``"model"``.
    * ``build_embeddings(..., for_checkpoint=True)`` supplies **zero** text vectors on purpose:
      ``text_vecs`` is a registered buffer, so ``load_state_dict`` restores the fitted MedCPT
      vectors and this path needs no encoder and no network.
    * **the repulsion must be fitted.** ``sample_layout`` raises when ``cfg.repulsion`` is on and
      no ``RepulsionParams`` were given, and ``layout_mode="field"`` is exactly the path that reads
      it — so without this the run would still have crashed after the first three were fixed.
    """
    import warnings as _warnings

    import torch
    from spatialcpav25_gen.model.field import BBoxClampWarning
    from spatialcpav25_gen.model.layout import fit_repulsion
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData

    # `build_embeddings` lives in the chain diagnostic; four other scripts already import it from
    # there for exactly this reason.
    from t10_chain_diagnostic import build_embeddings

    data = TrainingData.build(vol, cfg)
    model = CTFFlow(cfg, data, build_embeddings(cfg, vol, for_checkpoint=True), grf_seed=seed)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", BBoxClampWarning)
        if cfg.repulsion:
            model.repulsion = fit_repulsion(vol, cfg, seed=seed + 1)
    del torch
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
