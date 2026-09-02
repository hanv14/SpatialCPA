"""One fit under the selected configuration, the full T09 calibration chain, and the score.

The third leg of a per-dataset T09 run, after ``scripts/t09_select_starmap.py``. It does on
real tissue what ``scripts/t09_report.py`` §§4-6 do on the fixture, and then scores the result
on the **pinned instrument** — ``bench3.evaluate_paper``, the same code every comparable number
came out of — against the two model-free referents that bound the problem:

* ``oracle`` — the held-out cells themselves. The ceiling; nothing can beat it.
* ``flanking_copy`` — the neighbouring real section, copied. The floor a generative method has
  to clear to have earned anything.

Neither is a method, so neither depends on the config, the sampler or the seed; they are
re-scored here rather than quoted, because a referent measured on another machine is not a
referent.

**The chain, in the order ``specs/09`` §2 fixes it.**

1. ``calibrate_retrieval_window`` — the z window derived from this stack's own largest gap,
   not the constant. Passed to every generation call.
2. ``calibrate_lengthscale`` — ``ell_xy`` against the flanking **training** sections' mean
   Moran's I, and ``ell_z`` against their observed between-section correlation (open risk R1).
   Both refuse a configuration they cannot act through (``prior_mode="iid"``,
   ``expr_mode="cross-mix"``), which is a real possibility here and is reported rather than
   worked around.
3. ``apply_lengthscale`` — the only sanctioned writer, and it applies **only a converged
   axis**. An unreachable one is dropped with a ``CalibrationNotAppliedWarning`` naming both
   numbers, and the config's own value stands. The two axes are decided separately.
4. ``calibrate_detection`` — the ``pi``/``log theta`` pair. Measured always, **applied only
   under ``--apply-detection``**: T09 found it had no headroom on the fixture (the model was
   already inside the tissue's own section-to-section rate variation) and left it off by
   default. Whether real tissue gives it something to do is exactly what the two rows here
   answer, so both are scored.
5. ``calibrate_anchor_weight`` — only under ``expr_mode="auto-blend"``, which is not the
   shipped path; skipped with a line saying so rather than silently.

Everything is leakage-free by type: every calibrator takes the ``TrainingVolume`` and the
held-out sections are not in the file at all.

**Density control.** Each target section is scored twice — as emitted, and subsampled to the
ground truth's own cell count — because a denser point set puts kNN neighbours closer together
and inflates every graph-based metric. The matched rows are the comparable ones;
``paper_cell_count_ratio`` is meaningful only in the raw pass and is reported from there.

Usage::

    python scripts/t09_ship_starmap.py --preflight --bench3 /path/to/benchmark-pbya-v3
    python scripts/t09_ship_starmap.py \\
        --selected runs/select/starmap_visual_cortex/selected.yaml \\
        --probes runs/probes --bench3 /path/to/benchmark-pbya-v3

An interrupted fit resumes: ``--fit-checkpoint`` is written every
``Config.checkpoint_every_n_steps`` and read back only if it matches this config, seed and
budget. Re-running the identical command after the fit finished re-uses the saved model and
goes straight to calibration (``--model`` / ``--reuse-model``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import yaml
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.infer.calibrate import (
    CalibrationError,
    CalibrationNotAppliedWarning,
    apply_lengthscale,
    calibrate_anchor_weight,
    calibrate_detection,
    calibrate_lengthscale,
    calibrate_retrieval_window,
)
from spatialcpav25_gen.infer.generate import generate_section, plane_at_z
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.layout import fit_repulsion
from spatialcpav25_gen.model.noise import VariogramError
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow
from spatialcpav25_gen.train.select import calibration_chunks, module_morans_agreement

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import (
    base_config,
    clamp_config_to_input,
    describe_text_channel,
    embeddings_factory,
    load_training_volume,
    preflight_text_encoder,
)

TARGETS = (("section_2", 30.0), ("section_4", 52.0), ("section_6", 74.0))

# ``specs/09`` §3 / ``select.METRIC_NAMES``, under ``specs/10``'s ``paper_`` spellings, plus the
# two controls a layout claim is read against.
SIX = (
    "paper_morans_pearson",
    "paper_gearys_pearson",
    "paper_umap_mixing",
    "paper_marker_field_r",
    "paper_marker_depth_r",
    "paper_celltype_localization",
)
EXTRA = ("paper_gene_mean_spearman", "paper_cell_count_ratio")


def _fmt(v, prec: int = 4, sign: bool = True) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{prec}f}" if sign else f"{v:.{prec}f}"


def gt_counts(ground_truth: Path) -> dict[str, int]:
    import anndata as ad

    gt = ad.read_h5ad(ground_truth, backed="r")
    try:
        sections = gt.obs["section"].values.astype(str)
        return {s: int((sections == s).sum()) for s, _z in TARGETS}
    finally:
        gt.file.close()


def score(path: str, ground_truth: Path, use_umap: bool) -> dict:
    from bench3.evaluate_paper import evaluate_paper

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return evaluate_paper(str(path), str(ground_truth), use_umap=use_umap)


def per_section_of(result: dict, metric: str) -> dict[str, float | None]:
    key = metric.replace("paper_", "")
    ps = result.get("per_section", {})
    out: dict[str, float | None] = {}
    for sid, _z in TARGETS:
        sec = ps.get(sid)
        v = sec.get(key) if isinstance(sec, dict) else None
        out[sid] = None if v is None else float(v)
    return out


def median_of(result: dict, metric: str) -> float:
    """Median over held-out sections — ``specs/10`` §4.6's estimator, never a mean.

    A mean over n = 3 launders a fixed structural penalty on one section into the headline
    number, and `section_2` carries exactly such a penalty: it is bracketed by the stack's
    first section, so its evidence is one-sided (``specs/10`` §1).
    """
    values = [v for v in per_section_of(result, metric).values() if v is not None]
    return float(np.median(values)) if values else float("nan")


def write_prediction(per_section: dict, gene_names: list[str], path: str, seed: int) -> None:
    """Emit through the wrappers' own ``_v2_io`` writer, so the evaluator sees a real prediction."""
    import _v2_io

    _v2_io.write_prediction_h5(
        per_section, gene_names, list(per_section), {"seed": seed}, 0.0, path, "spatialcpav25_gen"
    )


def load_selected(path: Path | None, seed: int, overrides: dict) -> tuple[Config, dict]:
    """The selected configuration, or ``Config``'s shipped defaults with a stated fallback.

    A campaign run must not silently invent a configuration, so the absence of a selection is
    reported in the provenance rather than hidden: ``source="defaults"`` is a fact about the
    number, not a detail.
    """
    if path is None:
        return base_config(seed, **overrides), {"source": "defaults", "selection_path": None}
    if not Path(path).exists():
        raise SystemExit(
            f"--selected {path} does not exist. It is written by\n"
            "  python scripts/t09_select_starmap.py --run ...\n"
            "Omit --selected to fit at Config's shipped defaults instead; the report then "
            "records source='defaults', because a run must say which of the two it was."
        )
    payload = yaml.safe_load(Path(path).read_text())
    cfg = Config(**payload["config"]).replace(seed=int(seed), **overrides)
    return cfg, {
        "source": "selected.yaml",
        "selection_path": str(path),
        "selection_config_hash": payload.get("config_hash"),
        "pinned": payload.get("pinned", {}),
        "selection_seed": payload.get("selection_seed"),
    }


def run_calibration(model, volume, cfg: Config, seed: int, args) -> tuple[Config, dict, object]:
    """The whole T09 §2 chain. Returns ``(generation_cfg, record, detection_or_None)``."""
    record: dict = {}

    window = calibrate_retrieval_window(volume, cfg)
    record["retrieval_window"] = {
        "window": float(window.window),
        "max_gap_um": float(window.max_gap_um),
        "configured": float(cfg.retrieval_z_window),
    }
    print(
        f"  retrieval_z_window: derived {window.window:g} spacings "
        f"(largest gap {window.max_gap_um:g} um; Config's fallback {cfg.retrieval_z_window:g})"
    )

    lengthscale = None
    gen_cfg = cfg
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lengthscale = calibrate_lengthscale(model, volume, cfg, seed=seed)
        for w in caught:
            print(f"    warning: {w.message}")
    except (CalibrationError, VariogramError) as exc:
        # Not a failure of the run, and it must not cost the fit that has already been paid
        # for. Two refusals reach here and both are the codebase working:
        #
        # * ``CalibrationError`` — the calibrator declining a configuration ``ell`` cannot act
        #   through (``prior_mode="iid"``, ``expr_mode="cross-mix"``), the guard T09 added
        #   after a flat objective returned a bracket endpoint dressed as a fit;
        # * ``VariogramError`` — the along-z variogram declining to fit at all. Tier-1
        #   STARmap's training stack is **four** sections, giving exactly three distinct z
        #   lags, which is the bare minimum for a nugget, a sill and a length-scale; a stack
        #   that thin can legitimately fail to show enough structured variance.
        #
        # Recorded verbatim, because "why is there no ell here" is the first question a reader
        # of the calibration table will have, and the answer is not "it converged to nothing".
        record["lengthscale"] = {"status": "refused", "reason": str(exc)}
        print(f"  ell: REFUSED — {exc}")

    if lengthscale is not None:
        record["lengthscale"] = {
            "ell_xy": float(lengthscale.ell[0]),
            "ell_z": float(lengthscale.ell[2]),
            "status": str(lengthscale.status),
            "ell_z_status": str(lengthscale.ell_z_status),
            "ell_fitted_xy": float(lengthscale.ell_fitted[0]),
            "ell_fitted_z": float(lengthscale.ell_fitted[2]),
            "i_gen": float(lengthscale.i_gen),
            "i_target": float(lengthscale.i_target),
            "z_achieved": float(lengthscale.z_achieved),
            "z_target": float(lengthscale.z_target),
            "iterations": int(lengthscale.iterations),
        }
        print(
            f"  ell_xy = {lengthscale.ell[0]:.1f} um [{lengthscale.status}], "
            f"ell_z = {lengthscale.ell[2]:.1f} um [{lengthscale.ell_z_status}]; "
            f"I_gen {lengthscale.i_gen:.4f} vs flanking {lengthscale.i_target:.4f}, "
            f"between-section r {lengthscale.z_achieved:.4f} vs {lengthscale.z_target:.4f}"
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", CalibrationNotAppliedWarning)
            gen_cfg = apply_lengthscale(cfg, lengthscale)
        for w in caught:
            print(f"    not applied: {w.message}")
        record["lengthscale"]["applied_ell_xy"] = float(gen_cfg.ell_xy)
        record["lengthscale"]["applied_ell_z"] = float(gen_cfg.ell_z)
        print(
            f"  applied: ell_xy {cfg.ell_xy:g} -> {gen_cfg.ell_xy:g}, "
            f"ell_z {cfg.ell_z:g} -> {gen_cfg.ell_z:g}"
        )

    detection = None
    if not args.no_detection:
        t0 = time.time()
        detection = calibrate_detection(model, volume, gen_cfg, seed=seed)
        record["detection"] = {
            "measured": True,
            "shipped": bool(args.apply_detection),
            "fold_sections": list(detection.section_ids),
            "detection_mae_gen_vs_real": float(
                np.mean(np.abs(detection.detection_gen - detection.detection_real))
            ),
            "seconds": round(time.time() - t0, 1),
        }
        print(
            f"  detection/dispersion calibrated in {time.time() - t0:.0f}s on folds "
            f"{list(detection.section_ids)}; per-gene detection MAE "
            f"{record['detection']['detection_mae_gen_vs_real']:.4f}. "
            f"shipped={bool(args.apply_detection)} — T09's default is off (no headroom on the "
            "fixture); both arms are scored below, which is what decides it here"
        )
    else:
        record["detection"] = {"measured": False, "applied": False}

    anchor = None
    if gen_cfg.expr_mode == "auto-blend":
        anchor = calibrate_anchor_weight(model, volume, gen_cfg, seed=seed)
        record["anchor"] = {"fitted": True}
        print("  anchor w(v) fitted (expr_mode=auto-blend)")
    else:
        record["anchor"] = {
            "fitted": False,
            "why": f"expr_mode={gen_cfg.expr_mode}: w(v) is only read under auto-blend",
        }
        print(f"  anchor w(v): not fitted — expr_mode={gen_cfg.expr_mode} never reads it")

    return gen_cfg, record, (lengthscale, detection, anchor, window)


def generate_targets(model, volume, cfg, seed, window, detection, anchor, truth, workdir, tag):
    """Generate the three target sections and write the raw and density-matched predictions."""
    rng = np.random.default_rng(0)
    raw: dict = {}
    matched: dict = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        warnings.simplefilter("ignore")
        for name, z in TARGETS:
            emitted = generate_section(
                model,
                plane_at_z(volume, z, cfg),
                volume,
                cfg,
                seed,
                calibration=detection,
                anchor=anchor,
                z_window=float(window.window),
            )
            n = int(emitted.n_obs)
            print(f"    {tag} {name}: {n} cells / {truth[name]} ground truth", flush=True)
            x = emitted.X
            x = np.asarray(x.toarray() if sp.issparse(x) else x, dtype=np.float32)
            xyz = np.asarray(emitted.obsm["xyz"], dtype=np.float64)
            ct = np.asarray(emitted.obs[cfg.celltype_key].values, dtype=str)
            raw[name] = {"X": sp.csr_matrix(x), "coords": xyz, "cell_type": ct}
            keep = np.arange(n) if n <= truth[name] else rng.choice(n, truth[name], replace=False)
            matched[name] = {
                "X": sp.csr_matrix(x[keep]),
                "coords": xyz[keep],
                "cell_type": ct[keep],
            }
    workdir.mkdir(parents=True, exist_ok=True)
    genes = list(volume.gene_names)
    out_raw = str(workdir / f"{tag}_raw.h5")
    out_matched = str(workdir / f"{tag}_matched.h5")
    write_prediction(raw, genes, out_raw, seed)
    write_prediction(matched, genes, out_matched, seed)
    return out_raw, out_matched


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--selected",
        default=None,
        help="selected.yaml from t09_select_starmap.py. Omitted: Config's defaults, "
        "recorded as such in the provenance",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--train-steps",
        type=int,
        default=None,
        help="override the selected budget. The selection chose one; overriding it makes this "
        "an ablation arm rather than the shipped fit, and the override lands in the config "
        "hash the report prints",
    )
    ap.add_argument("--workdir", default="runs/ship", help="predictions, model and checkpoint")
    ap.add_argument("--out", default="reports/t09_starmap_shipped.md")
    ap.add_argument(
        "--probes",
        default=None,
        help="a `bench3.selftest --keep` dir holding oracle/ and flanking_copy/",
    )
    ap.add_argument("--model", default=None, help="reuse a saved fit instead of training")
    ap.add_argument(
        "--reuse-model", action="store_true", help="reuse <workdir>/model.pt if it is already there"
    )
    ap.add_argument(
        "--apply-detection",
        action="store_true",
        help="apply the pi / log-theta calibration to the emitted counts as well as "
        "measuring it; both rows are scored either way",
    )
    ap.add_argument(
        "--no-detection",
        action="store_true",
        help="skip the detection calibration entirely (it costs a LOSO decode pass)",
    )
    ap.add_argument(
        "--no-umap",
        action="store_true",
        help="skip paper_umap_mixing; the other six metrics are unaffected",
    )
    ap.add_argument(
        "--no-modules", action="store_true", help="skip the per-module Moran's I diagnostic (A2)"
    )
    ap.add_argument(
        "--w-thick",
        type=float,
        default=None,
        help="override Config.w_thick. With --w-prog this is ablation A7 (specs/10 section 6): "
        "SEFL ships with all three weights at 0, so A7 is an ADDITION experiment and the "
        "spec's value is 0.2. The override lands in the config hash the report prints. Run "
        "scripts/t10_a7_thick_binding.py on this dataset first: L_thick's Poisson hinge makes "
        "it charge exactly zero below a relative count error of ~1/sqrt(N), and on a volume "
        "with small slabs that can be most of training",
    )
    ap.add_argument(
        "--w-prog",
        type=float,
        default=None,
        help="override Config.w_prog; the other half of A7. w_cross stays at 0 in both arms — "
        "it is redundant by construction in v25 and harmful when trained (open risk R6), so "
        "A7 tests TWO losses, not three, and the write-up has to say so",
    )
    ap.add_argument("--text-cache", default=None)
    ap.add_argument("--gene-meta", default=None)
    ap.add_argument(
        "--expr-pca-dim",
        type=int,
        default=None,
        help="override Config.expr_pca_dim. The default is clamped to the panel width "
        "(specs/10 section 0's owed fix), which on STARmap's 28 genes is 28. "
        "reports/pilot.md's recorded numbers used a hand-picked 16, so a run meant to be "
        "compared with those has to say 16 here",
    )
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    ap.add_argument(
        "--preflight",
        action="store_true",
        help="resolve paths, load the volume, encode the panel, read the selection, "
        "check the probes, then exit without fitting",
    )
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    overrides: dict = {}
    if args.text_cache:
        overrides["text_cache_dir"] = args.text_cache
    if args.gene_meta:
        overrides["gene_meta_path"] = args.gene_meta
    if args.train_steps is not None:
        overrides["train_steps"] = int(args.train_steps)
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    # A7. Each overrides exactly one Config field after the selected config is loaded, so the
    # arm differs from the shipped fit in the weight and nothing else (specs/10 section 6).
    if args.w_thick is not None:
        overrides["w_thick"] = float(args.w_thick)
    if args.w_prog is not None:
        overrides["w_prog"] = float(args.w_prog)
    cfg, provenance = load_selected(
        Path(args.selected) if args.selected else None, args.seed, overrides
    )
    cfg = clamp_config_to_input(cfg, paths.input)
    print(
        f"  config {cfg.content_hash()} from {provenance['source']}: "
        f"layout_mode={cfg.layout_mode} prior_mode={cfg.prior_mode} expr_mode={cfg.expr_mode} "
        f"text_emb_mode={cfg.text_emb_mode} train_steps={cfg.train_steps} "
        f"weights={cfg.w_autocorr:g}/{cfg.w_profile:g}/{cfg.w_distribution:g}"
    )
    sefl_on = cfg.w_thick > 0.0 or cfg.w_prog > 0.0 or cfg.w_cross > 0.0
    print(
        f"  SEFL: w_cross={cfg.w_cross:g} w_thick={cfg.w_thick:g} w_prog={cfg.w_prog:g}"
        + ("   <- A7 arm (SEFL ON)" if sefl_on else "   (shipped: SEFL off)")
    )

    volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    truth = gt_counts(paths.ground_truth)
    print(
        f"  volume: {sum(s.n_cells for s in volume.sections)} cells x {volume.n_genes} genes, "
        f"{volume.n_sections} sections; ground truth {truth}"
    )

    workdir = Path(args.workdir)
    if args.preflight:
        described = preflight_text_encoder(cfg, volume)
        print("\n── text channel ──")
        for key, value in described.items():
            print(f"  {key:<22} {value}")
        import _v2_io  # noqa: F401
        from bench3.evaluate_paper import evaluate_paper  # noqa: F401

        if args.probes:
            for probe in ("oracle", "flanking_copy"):
                pred = Path(args.probes) / probe / "prediction.h5"
                print(f"  probe {probe:<14} {'present' if pred.exists() else 'MISSING'}  {pred}")
        saved = workdir / "model.pt"
        print(f"  saved model            {'present' if saved.exists() else 'absent'}  {saved}")
        print("\npreflight OK — nothing was fitted.")
        return 0

    embeddings = embeddings_factory(volume)
    print("  text channel: " + json.dumps(describe_text_channel(cfg, volume), default=str))

    workdir.mkdir(parents=True, exist_ok=True)
    model_path = Path(args.model) if args.model else workdir / "model.pt"
    data = TrainingData.build(volume, cfg)
    model = CTFFlow(cfg, data, embeddings(cfg), grf_seed=args.seed)

    if (args.model or args.reuse_model) and model_path.exists():
        checkpoint = torch.load(model_path, map_location="cpu")
        saved_hash = Config(**checkpoint["config"]).content_hash()
        if saved_hash != cfg.content_hash():
            raise SystemExit(
                f"{model_path} was fitted under config {saved_hash}, not {cfg.content_hash()}. "
                "A fit is not portable across configs; delete it or point --model elsewhere."
            )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        fit_seconds = None
        print(f"  reused the fit at {model_path}")
    else:
        fit_ckpt = workdir / f"fit_seed{args.seed}.pt"
        print(
            f"  fitting {cfg.train_steps} steps; resumable checkpoint {fit_ckpt} "
            f"(every {cfg.checkpoint_every_n_steps} steps)"
        )
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BBoxClampWarning)
            train_ctfflow(
                model, cfg, steps=int(cfg.train_steps), seed=args.seed, checkpoint=str(fit_ckpt)
            )
        fit_seconds = time.time() - t0
        print(f"  fit: {cfg.train_steps} steps in {fit_seconds:.0f}s ({fit_seconds / 3600:.2f} h)")
        torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict()}, model_path)
        print(f"  saved {model_path}")

    # ``sample_layout`` returns ``_resample_layout`` before it ever looks at ``repulsion``,
    # so under the shipped ``layout_mode="resample"`` the fit is pure cost — and a real crash
    # point: ``fit_repulsion`` raises ``LayoutError`` on a point pattern with no soft-repulsion
    # range, which would throw away the fit that has already been paid for. Fitted only where
    # generation reads it.
    if cfg.repulsion and cfg.layout_mode != "resample":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.repulsion = fit_repulsion(volume, cfg, seed=args.seed + 1)
        print(f"  repulsion fitted (layout_mode={cfg.layout_mode} draws positions)")
    else:
        print(
            f"  repulsion: not fitted — layout_mode={cfg.layout_mode} copies real positions "
            "and never reads it"
            if cfg.repulsion
            else "  repulsion: off (Config.repulsion=False, ablation A4c)"
        )

    print("\n── calibration (leakage-free; flanking TRAINING sections only) ──")
    gen_cfg, cal_record, objects = run_calibration(model, volume, cfg, args.seed, args)
    lengthscale, detection, anchor, window = objects

    modules = None
    if not args.no_modules:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modules = module_morans_agreement(model, volume, gen_cfg, seed=args.seed)
        print(
            "  per-module Moran's I agreement (A2, diagnostic only): "
            + ", ".join(f"{r['module']}:{float(r['abs_diff']):.4f}" for r in modules)
        )

    print("\n── generation ──")
    # Both arms whenever the correction was measured: "does it transfer on real tissue" is a
    # question this run can answer for free, and T09 left it open on the fixture. Which arm is
    # the *shipped* one is --apply-detection's business, not the table's.
    arms: list[tuple[str, object]] = [("uncalibrated", None)]
    if detection is not None:
        arms.append(("detection-calibrated", detection))

    rows = []
    for tag, det in arms:
        out_raw, out_matched = generate_targets(
            model, volume, gen_cfg, args.seed, window, det, anchor, truth, workdir, tag
        )
        r_raw = score(out_raw, paths.ground_truth, not args.no_umap)
        r_matched = score(out_matched, paths.ground_truth, not args.no_umap)
        rows.append(
            {
                "arm": tag,
                "detection_applied": det is not None,
                "shipped": (det is not None) == bool(args.apply_detection),
                "config_hash": gen_cfg.content_hash(),
                "seed": int(args.seed),
                "n_pred": per_section_of(r_raw, "paper_n_pred_cells"),
                "n_gt": dict(truth),
                "matched": {m: median_of(r_matched, m) for m in (*SIX, *EXTRA)},
                "raw": {m: median_of(r_raw, m) for m in (*SIX, *EXTRA)},
                "matched_per_section": {m: per_section_of(r_matched, m) for m in SIX},
                "cell_count_ratio_raw": median_of(r_raw, "paper_cell_count_ratio"),
                "cell_count_ratio_per_section": per_section_of(r_raw, "paper_cell_count_ratio"),
            }
        )
        print(
            f"  {tag}: "
            + "  ".join(f"{m.replace('paper_', '')}={_fmt(rows[-1]['matched'][m])}" for m in SIX),
            flush=True,
        )

    referents: dict[str, dict] = {}
    if args.probes:
        for probe in ("oracle", "flanking_copy"):
            pred = Path(args.probes) / probe / "prediction.h5"
            if not pred.exists():
                raise SystemExit(
                    f"{probe}: no prediction at {pred}. Produce both referents once with\n"
                    f"  python -m bench3.selftest --dataset starmap_visual_cortex "
                    f"--probes oracle flanking_copy --keep {args.probes}"
                )
            r = score(str(pred), paths.ground_truth, not args.no_umap)
            referents[probe] = {
                "median": {m: median_of(r, m) for m in (*SIX, *EXTRA)},
                "per_section": {m: per_section_of(r, m) for m in SIX},
            }
            print(f"  scored referent {probe}", flush=True)

    lines = _report(
        rows,
        referents,
        cfg,
        gen_cfg,
        cal_record,
        provenance,
        lengthscale,
        window,
        modules,
        volume,
        args,
    )
    text = "\n".join(lines)
    print()
    print(text)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "arms": rows,
                "referents": referents,
                "calibration": cal_record,
                # The console has printed this since T09; persisting it is what makes a
                # timing gate ("one fit before the other five") checkable from the artifact
                # rather than from a scrollback. `null` means the fit was reused, not free.
                "fit_seconds": fit_seconds,
                "provenance": provenance,
                "config": gen_cfg.to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def _report(
    rows,
    referents,
    cfg,
    gen_cfg,
    cal_record,
    provenance,
    lengthscale,
    window,
    modules,
    volume,
    args,
) -> list[str]:
    arm_names = [r["arm"] for r in rows]
    header = "| metric | " + " | ".join(f"`{a}`" for a in arm_names)
    for probe in ("flanking_copy", "oracle"):
        if probe in referents:
            header += f" | `{probe}`"
    header += " |"
    ncols = len(arm_names) + sum(p in referents for p in ("flanking_copy", "oracle")) + 1

    lines = [
        "# T09 on real data — shipped config, full calibration, tier-1 STARmap",
        "",
        f"Dataset `starmap_visual_cortex`, holdout `paper_2_4_6` (**tier 1**, `specs/10` §1). "
        f"Config `{gen_cfg.content_hash()}` from {provenance['source']}"
        + (f" (`{provenance.get('selection_path')}`)" if provenance.get("selection_path") else "")
        + f", seed {rows[0]['seed']}.",
        "",
        "| gate | value |",
        "|---|---|",
        f"| `layout_mode` | `{gen_cfg.layout_mode}` |",
        f"| `prior_mode` | `{gen_cfg.prior_mode}` |",
        f"| `expr_mode` | `{gen_cfg.expr_mode}` |",
        f"| `text_emb_mode` | `{gen_cfg.text_emb_mode}` |",
        f"| `train_steps` | {gen_cfg.train_steps} |",
        f"| weights (autocorr / profile / distribution) | "
        f"{gen_cfg.w_autocorr:g} / {gen_cfg.w_profile:g} / {gen_cfg.w_distribution:g} |",
        "",
        "**The text channel is live for the first time on this dataset.** Every prior STARmap "
        "number was produced with embeddings built from zeros or from a bare symbol, so "
        "`text_emb_mode=medcpt` was `lookup` in all but name (ablation A3). Descriptors here "
        f"come from `{gen_cfg.gene_meta_path}` through `model.build_entity_embeddings`.",
        "",
        "## The six target metrics, medians over the three held-out sections",
        "",
        "At ground-truth-matched density: each section subsampled to its own true cell count, "
        "because a denser point set puts kNN neighbours closer and inflates every graph-based "
        "metric. Medians, never means (`specs/10` §4.6) — `section_2` carries a fixed "
        "one-sided-evidence penalty that a mean over n = 3 would launder into the headline.",
        "",
        header,
        "|---" * ncols + "|",
    ]
    for m in SIX:
        cells = [f"| {_fmt(r['matched'][m])} " for r in rows]
        for probe in ("flanking_copy", "oracle"):
            if probe in referents:
                cells.append(f"| {_fmt(referents[probe]['median'][m])} ")
        lines.append(f"| `{m}` " + "".join(cells) + "|")
    cells = [f"| {_fmt(r['matched']['paper_gene_mean_spearman'])} " for r in rows]
    for probe in ("flanking_copy", "oracle"):
        if probe in referents:
            cells.append(f"| {_fmt(referents[probe]['median']['paper_gene_mean_spearman'])} ")
    lines.append("| `paper_gene_mean_spearman` " + "".join(cells) + "|")
    cells = [f"| {_fmt(r['cell_count_ratio_raw'], 3, False)} " for r in rows]
    for probe in ("flanking_copy", "oracle"):
        if probe in referents:
            cells.append(
                f"| {_fmt(referents[probe]['median']['paper_cell_count_ratio'], 3, False)} "
            )
    lines.append("| `paper_cell_count_ratio` (raw pass) " + "".join(cells) + "|")

    lines += [
        "",
        "Per section, matched density:",
        "",
        "| arm | metric | " + " | ".join(s for s, _z in TARGETS) + " | median |",
        "|---" * (len(TARGETS) + 3) + "|",
    ]
    for r in rows:
        for m in SIX:
            ps = r["matched_per_section"][m]
            lines.append(
                f"| `{r['arm']}` | `{m.replace('paper_', '')}` | "
                + " | ".join(_fmt(ps[s]) for s, _z in TARGETS)
                + f" | {_fmt(r['matched'][m])} |"
            )
    lines += ["", "Emitted cell counts (generated / ground truth):", ""]
    for r in rows:
        lines.append(
            f"* `{r['arm']}`: "
            + ", ".join(f"{s}={_fmt(r['n_pred'][s], 0, False)}/{r['n_gt'][s]}" for s, _z in TARGETS)
        )

    lines += [
        "",
        "## Calibration statuses",
        "",
        "```",
        json.dumps(cal_record, indent=2, default=str),
        "```",
    ]
    lines += calibration_chunks(lengthscale, window, modules)
    lines += [
        "",
        "## Provenance",
        "",
        "```",
        json.dumps(provenance, indent=2, default=str),
        "```",
        "",
        f"Training volume: {sum(s.n_cells for s in volume.sections)} cells x "
        f"{volume.n_genes} genes over {volume.n_sections} sections, "
        f"flattened={volume.flattened_sections}. "
        f"`use_umap={not args.no_umap}`.",
        "",
        "**One seed.** `specs/09` §3's repeated-seed rule requires "
        f"`claim_min_seeds` = {cfg.claim_min_seeds} for any measurement that reaches a paper "
        "claim, and the across-seed envelope measured on the fixture is 0.0335 "
        "(`reports/envelope_synthetic.md`). Any difference here smaller than that is a tie, "
        "and this table is a single-seed measurement — admissible as a diagnostic, not as a "
        "headline.",
    ]
    return lines


if __name__ == "__main__":
    sys.exit(main())
