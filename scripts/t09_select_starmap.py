"""T09 §3's per-dataset selection, on a real bench3 dataset (tier-1 STARmap by default).

``scripts/t09_report.py`` runs the same search on the synthetic fixture. This runs it on real
tissue, which is what ``specs/09`` §3 asks for — *"selects per dataset"* — and what the
fixture has been shown to be unable to substitute for on at least one gate
(``progress/fixture_limitations.md`` §2).

Three things it does that no existing driver did.

**1. The text channel is real.** Every prior STARmap measurement built embeddings either from
zeros (``t10_chain_diagnostic``, and ``t10_rescore_saved`` through it) or from
``gene_descriptor(symbol, None)`` (the bench3 wrapper) — a bare ``"Slc17a7."``. So
``text_emb_mode="medcpt"`` has never been exercised on real data: every number so far is
ablation A3's ``lookup`` arm under another name. Here the descriptors come from the panel's own
``resources/gene_meta.parquet`` (28/28 of the STARmap panel, all with summaries) through
``model.build_entity_embeddings``, and ``--preflight`` proves the encoder is reachable before
anything is fitted.

**2. Gates may be pinned.** ``--pin layout_mode=resample`` (the default) excludes that gate from
the merged full-budget gate and from coordinate descent, and the report says it was pinned and
why. R11 settled ``layout_mode`` on real data — ``resample`` 0.7546 against ``hybrid`` 0.6692
and a copy floor of 0.7765, 3.2x the across-seed envelope — and re-opening a settled gate at one
seed can only lose to noise.

> **The saving is scoring, not fitting.** ``layout_mode`` does not enter the fit
> (``select.FIT_INVARIANT_GATES``), so 6 fits already served the merged gate's 18 cells;
> pinning it drops the gate to 6 cells and removes **12 LOSO scorings**, not 12 fits. Stated
> because the opposite belief would make this flag look like it saves a day of compute.

**3. It parallelises.** ``run_selection`` is sequential — the merged gate needs the joint gate's
budget, and coordinate descent needs the merged gate's incumbent — but the cells *within* a
stage are independent, and :class:`~spatialcpav25_gen.train.select.ScoreCache` is the seam.
``--prewarm`` scores one shard of one stage and flushes it to the shared CSV; the final
``--run`` finds them all cached and issues only the two fits that genuinely depend on a prior
decision. On a wide box that is ~3 h wall instead of ~9 h serial.

Usage::

    # 0. seconds. Resolves paths, loads the volume, encodes the panel, prints the plan.
    python scripts/t09_select_starmap.py --preflight --bench3 /path/to/benchmark-pbya-v3

    # 1. the joint gate: 4 independent cells
    python scripts/t09_select_starmap.py --prewarm joint --index K --of 4 ...

    # 2. the merged full-budget gate: 6 independent cells at the selected budget
    python scripts/t09_select_starmap.py --prewarm full-budget --index K --of 6 ...

    # 3. the search itself; everything above is a cache hit
    python scripts/t09_select_starmap.py --run ...

Every stage is resumable: a killed process loses at most the cell it was in, and re-running the
identical command skips what is already in ``scores.csv``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import yaml
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.train.select import (
    ALL_GATES,
    METRIC_NAMES,
    SCORING_FAILURES,
    FitScorer,
    ScoreCache,
    SelectionError,
    full_budget_gate_cells,
    joint_gate_cells,
    rank_candidates,
    repulsion_is_reachable,
    run_selection,
    selection_folds,
    volume_cache_key,
)
from spatialcpav25_gen.train.select import Candidate as _Candidate

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

DEFAULT_PIN = "layout_mode=resample"
PIN_REASON = (
    "`layout_mode` is **pinned**, not selected. R11 settled it on this dataset "
    "(`reports/r11_starmap_layout_modes.md`): `resample` 0.7546 against `hybrid` 0.6692 and "
    "`field` 0.6607 on `celltype_localization`, with the model-free `flanking_copy` floor at "
    "0.7765 — a separation 3.2x the across-seed envelope, where the fixture tie-break that "
    "originally shipped `hybrid` was 1.03x it. `specs/05` §4a records it as the shipped "
    "default. Re-opening a gate that real data has settled, at one seed and inside a search "
    "whose own margins are envelope-sized, can only lose to noise. Note the cost this does "
    "*not* save: `layout_mode` does not enter the fit (`select.FIT_INVARIANT_GATES`), so the "
    "merged gate already fitted 6 models for 18 cells — pinning removes 12 LOSO **scorings**, "
    "not 12 fits."
)


def parse_pin(values: list[str] | None) -> dict[str, str]:
    """``["layout_mode=resample"] -> {"layout_mode": "resample"}``, with the shape checked here."""
    out: dict[str, str] = {}
    for item in values or []:
        if item.lower() in ("", "none"):
            continue
        if "=" not in item:
            raise SystemExit(f"--pin expects gate=option, got {item!r}")
        gate, option = item.split("=", 1)
        out[gate.strip()] = option.strip()
    return out


def dataset_id(input_path: Path) -> str:
    """Return the per-dataset key the bench3 wrapper uses.

    Computed the same way so a selection persisted here is the one
    ``run_spatialcpav25_gen.py --require-config`` loads.
    """
    return input_path.resolve().parents[1].name


def volume_fingerprint(input_path: Path) -> dict:
    """Mirror of the wrapper's ``volume_fingerprint``: what this selection was made against.

    A configuration chosen on a different build of the same dataset is not this dataset's
    configuration, and the wrapper refuses a ``selected.yaml`` whose fingerprint does not
    match the input it is handed. Computed identically here so the file this driver writes is
    the file that wrapper accepts.
    """
    import anndata as ad

    adata = ad.read_h5ad(input_path, backed="r")
    try:
        sections = sorted({str(s) for s in adata.obs["section"].values})
        n_obs, n_vars = int(adata.n_obs), int(adata.n_vars)
    finally:
        adata.file.close()
    return {
        "n_cells": n_obs,
        "n_genes": n_vars,
        "sections": sections,
        "input_mtime": round(input_path.stat().st_mtime, 3),
    }


def joint_winner(cfg: Config, cache: ScoreCache) -> dict:
    """The joint gate's winning overrides, read back out of the checkpoint.

    ``--prewarm full-budget`` needs the selected budget, which is the joint gate's answer. It
    is read from the cache rather than recomputed, and ranked with the selector's own
    :func:`~spatialcpav25_gen.train.select.rank_candidates` rather than a second
    implementation of "median rank".
    """
    cells = joint_gate_cells(cfg)
    scored = []
    for label, overrides in cells:
        recorded = cache.get(cfg.replace(**overrides), int(overrides["train_steps"]))
        if recorded is None:
            raise SystemExit(
                f"the joint gate is not complete in {cache.path}: {label!r} is missing.\n"
                f"  Run stage 1 first:  --prewarm joint --index K --of {len(cells)}"
            )
        scored.append(
            _Candidate(
                gate="joint",
                label=label,
                overrides=overrides,
                steps=int(overrides["train_steps"]),
                scores=recorded,
            )
        )
    return dict(min(rank_candidates(scored), key=lambda c: c.rank).overrides)


def stage_cells(stage: str, cfg: Config, pinned: dict[str, str], cache: ScoreCache) -> list[tuple]:
    """``[(label, overrides, steps), ...]`` for one prewarmable stage."""
    if stage == "joint":
        return [(label, ov, int(ov["train_steps"])) for label, ov in joint_gate_cells(cfg)]
    incumbent = joint_winner(cfg, cache)
    steps = int(incumbent["train_steps"])
    return [
        (label, {**incumbent, **ov}, steps) for label, ov in full_budget_gate_cells(cfg, pinned)
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out",
        default="runs/select/starmap_visual_cortex",
        help="where scores.csv, selection_report.md and selected.yaml go",
    )
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument(
        "--base-train-steps",
        type=int,
        default=None,
        help="the 1x budget the joint gate's {1x, 2x} cells are built around. Default: "
        "Config.train_steps. Not a tuning knob for the answer — the gate still selects "
        "between the two — but it is what a smoke run turns down to make the search finish "
        "in seconds, so it is recorded in the report's config hash like every other field",
    )
    ap.add_argument(
        "--pin",
        nargs="*",
        default=[DEFAULT_PIN],
        help=f"gate=option to exclude from selection (default: {DEFAULT_PIN}; "
        "pass --pin none to select every gate)",
    )
    ap.add_argument("--text-cache", default=None, help="override Config.text_cache_dir")
    ap.add_argument("--gene-meta", default=None, help="override Config.gene_meta_path")
    ap.add_argument(
        "--expr-pca-dim",
        type=int,
        default=None,
        help="override Config.expr_pca_dim. The default is clamped to the panel width "
        "(specs/10 section 0's owed fix), which on STARmap's 28 genes is 28. "
        "reports/pilot.md's recorded numbers used a hand-picked 16, so a run meant to be "
        "compared with those has to say 16 here",
    )
    ap.add_argument(
        "--flattened",
        dest="flattened",
        action="store_true",
        default=None,
        help="force flattened_sections (default: read it from the input's uns)",
    )
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="resolve paths, load the volume, encode the panel, print the plan, exit",
    )
    mode.add_argument(
        "--prewarm",
        choices=["joint", "full-budget"],
        help="score one shard of one stage into the shared checkpoint and exit",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="run the whole selection, resuming every cell already checkpointed",
    )
    ap.add_argument("--index", type=int, default=0, help="this shard, 0-based (--prewarm)")
    ap.add_argument("--of", type=int, default=1, help="how many shards (--prewarm)")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    pinned = parse_pin(args.pin)
    overrides: dict[str, object] = {}
    if args.text_cache:
        overrides["text_cache_dir"] = args.text_cache
    if args.gene_meta:
        overrides["gene_meta_path"] = args.gene_meta
    if args.base_train_steps is not None:
        overrides["train_steps"] = int(args.base_train_steps)
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    cfg = clamp_config_to_input(base_config(args.seed, **overrides), paths.input)
    if pinned:
        cfg = cfg.replace(**pinned)

    volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    print(
        f"  volume: {sum(s.n_cells for s in volume.sections)} cells x {volume.n_genes} genes, "
        f"{volume.n_sections} sections, flattened={volume.flattened_sections}"
    )
    print(f"  LOSO folds: {[s.section_id for s in selection_folds(volume, cfg)]}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keyed on the volume as well as the config: C33 changed every score while leaving
    # every config hash identical, so a config-only key served pre-fix numbers into a
    # post-fix run. A changed volume now simply misses rather than silently hitting.
    cache = ScoreCache(out_dir / "scores.csv", volume_key=volume_cache_key(volume))
    print(f"  checkpoint: {cache.path} ({len(cache)} cells already scored)")

    if args.preflight:
        described = preflight_text_encoder(cfg, volume)
        print("\n── text channel ──")
        for key, value in described.items():
            print(f"  {key:<22} {value}")
        if cfg.text_emb_mode == "medcpt" and described["n_bare"] == described["n_genes"]:
            print(
                "\n  ⚠ every gene is a bare symbol: medcpt is on with nothing behind it. "
                "Point --gene-meta at a table that covers this panel."
            )
        print("\n── the plan ──")
        joint = joint_gate_cells(cfg)
        merged = full_budget_gate_cells(cfg, pinned)
        descent = [g for g, _ in ALL_GATES if g not in pinned and not _is_full_budget(g)]
        print(
            f"  stage 1  joint gate            {len(joint)} cells "
            f"({', '.join(label for label, _ in joint)})"
        )
        print(f"  stage 2  merged full-budget    {len(merged)} cells at the selected budget")
        print(f"  stage 3  coordinate descent    gates {descent or '(none)'}")
        print(f"  pinned                         {pinned or '(nothing)'}")
        print("\npreflight OK — nothing was fitted.")
        return 0

    embeddings = embeddings_factory(volume)
    print("  text channel: " + json.dumps(describe_text_channel(cfg, volume), default=str))

    if args.prewarm:
        cells = stage_cells(args.prewarm, cfg, pinned, cache)
        mine = [c for i, c in enumerate(cells) if i % max(1, args.of) == args.index]
        print(
            f"\nstage {args.prewarm}: {len(cells)} cells, shard {args.index}/{args.of} "
            f"-> {[c[0] for c in mine]}"
        )
        # Same rule run_selection applies when it builds its own scorer: with layout_mode
        # pinned to resample nothing can read the fitted interaction, so it is not fitted.
        scorer = FitScorer(volume, embeddings, needs_repulsion=repulsion_is_reachable(cfg, pinned))
        for label, cell_overrides, steps in mine:
            cell_cfg = cfg.replace(**cell_overrides)
            if cache.get(cell_cfg, steps) is not None:
                print(f"  {label}: already checkpointed, skipping")
                continue
            t0 = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", BBoxClampWarning)
                    scores = scorer(cell_cfg, steps=steps, seed=args.seed)
            except SCORING_FAILURES as exc:
                # Same rule as run_selection's: the emission guard refusing an under-trained
                # candidate ranks it last rather than killing the stage. Written to the cache
                # so the final --run sees the same verdict and does not refit it.
                scores = dict.fromkeys(METRIC_NAMES, float("-inf"))
                print(
                    f"  {label} @ {steps} steps FAILED, ranked last: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            cache.put(cell_cfg, steps, label, scores)
            scorer.release_fits()
            print(
                f"  {label} @ {steps} steps in {time.time() - t0:.0f}s: "
                + "  ".join(f"{m}={scores[m]:+.4f}" for m in METRIC_NAMES),
                flush=True,
            )
        print(f"\nshard done; {len(cache)} cells in {cache.path}")
        return 0

    t0 = time.time()
    result = run_selection(
        volume,
        cfg,
        seed=args.seed,
        embeddings=embeddings,
        dataset=dataset_id(Path(paths.input)),
        report_path=out_dir / "selection_report.md",
        checkpoint=cache,
        pinned=pinned,
        pinned_reason=PIN_REASON if pinned else "",
    )
    wall = time.time() - t0

    selected = out_dir / "selected.yaml"
    selected.write_text(
        yaml.safe_dump(
            {
                "dataset_id": result.dataset,
                "volume_fingerprint": volume_fingerprint(Path(paths.input)),
                "selection_seed": args.seed,
                "config_hash": result.config.content_hash(),
                "pinned": result.pinned,
                # SPEC_QUESTIONS C34: a gate this dataset cannot decide is recorded as
                # undetermined rather than carrying a value the shipped config cannot
                # support. The field still holds *something* — Config is a total record —
                # but a consumer that reads this file is told not to treat it as selected.
                "undetermined": sorted(result.undetermined),
                "undetermined_won_elsewhere": result.elsewhere_winner,
                "n_loso_folds": len(selection_folds(volume, cfg)),
                "config": result.config.to_dict(),
            },
            sort_keys=False,
        )
    )
    print(f"\nselection done in {wall:.0f}s, {len(result.fits)} fits issued this process")
    print(f"  selected config {result.config.content_hash()} -> {selected}")
    for gate, _ in ALL_GATES:
        mark = "  ⛔ UNDETERMINED for this dataset" if gate in result.undetermined else ""
        print(f"    {gate:<16} {getattr(result.config, gate)}{mark}")
    if result.undetermined:
        print(
            "\n  ⛔ "
            + ", ".join(sorted(result.undetermined))
            + " could not be decided on this dataset: inert under the shipped incumbent. "
            "The winner measured elsewhere is in the report and is NOT in selected.yaml."
        )
    print(f"    {'train_steps':<16} {result.config.train_steps}")
    print(
        f"    {'weights':<16} {result.config.w_autocorr:g} / {result.config.w_profile:g} / "
        f"{result.config.w_distribution:g}"
    )
    envelope = cfg.claim_tie_break_envelope
    print(f"\n── per-gate tie-break review (margin vs claim_tie_break_envelope={envelope:g}) ──")
    for r in result.reviews:
        margin = "—" if r.margin != r.margin else f"{r.margin:.4f} (n={r.n_folds} folds)"
        print(
            f"  {r.gate:<16} ships {r.winner:<10} rival {r.runner_up!s:<10} "
            f"margin {margin:<8} {'INSIDE' if r.inside_envelope else ''} "
            f"{'FLIPPED' if r.flipped else ''}  {r.reason}"
        )
    print(f"\nreport: {out_dir / 'selection_report.md'}")
    return 0


def _is_full_budget(gate: str) -> bool:
    from spatialcpav25_gen.train.select import TRAINING_FREE_OPTIONS

    return bool(TRAINING_FREE_OPTIONS[gate])


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SelectionError as exc:
        raise SystemExit(str(exc)) from exc
