"""SpatialCPA-v25-Gen (CTF-Flow) — benchmark-pbya-v3 wrapper.

Like the v16/v18-v24 wrappers this lives in ``benchmark-pbya-v3`` rather than in the
frozen v2 tree, and speaks the identical ``_v2_io`` contract, so ``run_benchmark``
invokes it like every other method and ``evaluate_paper`` reads its output unchanged.

**No tuning flags.** v20's wrapper took eleven (``--edit-weight``, ``--gap-scale``,
``--alpha-tol`` and the rest). This one takes none: the configuration is selected
internally per dataset (``specs/09`` §3), and *"fit takes no method flags"* is a claim in
the paper. Everything after the shared ``_v2_io`` arguments is either the seed or an
**ablation** switch from ``specs/10`` §6, each of which overrides exactly one ``Config``
field *after* the selected configuration is loaded, and each of which is recorded in
``method_params`` so a result says what produced it.

Per-dataset selection is **not** run on every invocation — it is ~23 fits, which would make
a bare run cost more than the campaign it belongs to. It runs once per dataset, is persisted
under ``$SPATIALCPAV25_SELECT_DIR/<dataset_id>/``, and is shared by every seed, arm and
tier (``specs/10`` §10.1). ``--select-only`` pre-warms a dataset; ``--require-config``
refuses to select, which is what campaign runs use.

Generation-only: the input file physically excludes the held-out sections and the method
receives one scalar target z per section.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

_V2_BENCH = Path(__file__).resolve().parents[4] / "benchmark-pbya-v2" / "src" / "benchmark"
sys.path.insert(0, str(_V2_BENCH / "methods"))   # _v2_io
sys.path.insert(0, str(_V2_BENCH))               # leakage_guard
import _v2_io  # noqa: E402
import leakage_guard  # noqa: E402

SELECT_DIR_ENV = "SPATIALCPAV25_SELECT_DIR"
DEFAULT_SELECT_DIR = "runs/select"


def check_environment() -> bool:
    """Report the package and torch, exactly as the sibling wrappers do."""
    try:
        import spatialcpav25_gen
        import torch
    except Exception as exc:                       # pragma: no cover - env probe
        print(f"ERROR: spatialcpav25_gen is not importable: {exc}", file=sys.stderr)
        print("  install it with `make install` in the repository root.", file=sys.stderr)
        return False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"spatialcpav25_gen {spatialcpav25_gen.__version__ if hasattr(spatialcpav25_gen, '__version__') else ''} "
          f"(CTF-Flow); torch {torch.__version__} ({device})")
    return True


# ── the volume fingerprint that makes a persisted selection self-invalidating ──

def volume_fingerprint(adata, input_path: str) -> dict:
    """Identify the training volume a selection was made against.

    A configuration chosen on a different build of the same dataset is not this dataset's
    configuration. bench3 already learned this lesson for its own ``_inputs/`` cache, where
    a stale copy made a build fix apply twice; the selection needs the same guard.
    """
    sections = sorted({str(s) for s in adata.obs["section"].values})
    return {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "sections": sections,
        "input_mtime": round(Path(input_path).stat().st_mtime, 3),
    }


def dataset_id(adata, input_path: str) -> str:
    """Stable per-dataset key. Derived datasets get their own, never the source's."""
    name = adata.uns.get("dataset_name")
    if name is not None and str(name).strip():
        return str(name)
    return Path(input_path).resolve().parent.name


def select_dir(dsid: str) -> Path:
    import os

    root = Path(os.environ.get(SELECT_DIR_ENV, DEFAULT_SELECT_DIR))
    return root / dsid


# ── configuration: selected once per dataset, then overridden only by ablations ──

def resolve_config(adata, args, fingerprint: dict, dsid: str):
    """Return ``(cfg, provenance)``: the shipped config for this dataset, or A-arm of it."""
    from spatialcpav25_gen.config import Config

    out_dir = select_dir(dsid)
    selected = out_dir / "selected.yaml"
    provenance: dict = {"dataset_id": dsid, "selection_path": str(selected)}

    if selected.exists():
        import yaml

        payload = yaml.safe_load(selected.read_text())
        stored = payload.get("volume_fingerprint")
        if stored != fingerprint:
            raise SystemExit(
                f"stale selection for dataset {dsid!r}:\n"
                f"  {selected}\n"
                f"  selected against: {stored}\n"
                f"  this input      : {fingerprint}\n"
                f"  A configuration chosen on a different build is not this build's "
                f"configuration. Re-select with --select-only, or point "
                f"${SELECT_DIR_ENV} at the right tree."
            )
        cfg = Config(**payload["config"])
        provenance["selection"] = "loaded"
    elif args.require_config:
        raise SystemExit(
            f"no selected configuration for dataset {dsid!r} at {selected}, and "
            f"--require-config forbids selecting one here.\n"
            f"  Pre-warm it once with:  --select-only\n"
            f"  (selection is ~23 fits; a campaign run must not start one silently.)"
        )
    else:
        cfg = run_selection_for(adata, args, fingerprint, dsid)
        provenance["selection"] = "ran"

    overrides = applied_overrides(args)
    if overrides:
        cfg = cfg.replace(**overrides)
    provenance["overrides"] = overrides
    provenance["config_hash"] = cfg.content_hash()
    return cfg, provenance


def applied_overrides(args) -> dict:
    """The ablation switches actually set. Empty for a bare run — that is the paper claim."""
    mapping = {
        "prior_mode": args.prior_mode,              # A1
        "w_autocorr": args.w_autocorr,              # A2
        "w_profile": args.w_profile,                # A2
        "w_distribution": args.w_distribution,      # A2
        "text_emb_mode": args.text_emb_mode,        # A3
        "retrieval_w_z": args.retrieval_w_z,        # A5
        "decoder": args.decoder,                    # A6
        "w_thick": args.w_thick,                    # A7
        "w_prog": args.w_prog,                      # A7
        "w_prog_wrong": args.w_prog_wrong,          # A8
        "layout_mode": args.layout_mode,
        "expr_mode": args.expr_mode,
        "train_steps": args.train_steps,
    }
    out = {k: v for k, v in mapping.items() if v is not None}
    if args.no_repulsion:                           # A4
        out["repulsion"] = False
    return out


def run_selection_for(adata, args, fingerprint: dict, dsid: str):
    """Run ``specs/09`` §3's selection once for this dataset and persist it."""
    import yaml
    from spatialcpav25_gen.config import Config
    from spatialcpav25_gen.train.select import ScoreCache, run_selection

    out_dir = select_dir(dsid)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  no selected config for {dsid!r}; running selection (~23 fits at this "
          f"dataset's scale). Checkpointed to {out_dir / 'scores.csv'} — an interrupted "
          f"run resumes rather than restarts.")
    volume = load_training_volume(adata, args)
    # run_selection refuses to invent embeddings — they come from T02's cache, which
    # train/select.py has no business knowing about — so the factory is passed here. Without
    # it every --select-only invocation raised SelectionError at the first line of the search,
    # which is why no selection had ever run through this wrapper.
    base = Config(seed=args.seed).replace(
        section_key="section", coord_key="spatial", celltype_key="cell_type", region_key=None,
    )
    pinned, reason = selection_pins(args)
    result = run_selection(
        volume,
        base,
        seed=args.seed,
        embeddings=lambda cfg: build_embeddings(cfg, volume),
        dataset=dsid,
        report_path=out_dir / "selection_report.md",
        checkpoint=ScoreCache(out_dir / "scores.csv"),
        pinned=pinned,
        pinned_reason=reason,
    )
    (out_dir / "selected.yaml").write_text(
        yaml.safe_dump(
            {
                "dataset_id": dsid,
                "volume_fingerprint": fingerprint,
                "selection_seed": args.seed,
                "config_hash": result.config.content_hash(),
                "pinned": result.pinned,
                "undetermined": sorted(result.undetermined),
                "undetermined_won_elsewhere": result.elsewhere_winner,
                "config": result.config.to_dict(),
            },
            sort_keys=False,
        )
    )
    print(f"  selected config {result.config.content_hash()} -> {out_dir / 'selected.yaml'}")
    return result.config


# ── data ──────────────────────────────────────────────────────────────────────

def load_training_volume(adata, args, cfg=None):
    """bench3's training-only input as a ``TrainingVolume``.

    Everything in the file is training data — ``run_benchmark`` removed the held-out
    sections before the wrapper ever saw it — so the whole volume is the training volume,
    and the type carries that guarantee downstream.
    """
    from spatialcpav25_gen.config import Config
    from spatialcpav25_gen.data.loaders import load_volume
    from spatialcpav25_gen.data.schema import TrainingVolume

    cfg = cfg or Config(seed=args.seed)
    cfg = cfg.replace(section_key="section", coord_key="spatial", celltype_key="cell_type",
                      region_key=None)
    # bench3 records the flattening under its own key, so pass it explicitly rather than
    # letting load_volume look for Config.flattened_sections_key: every v3 dataset collapses
    # a multi-plane slab to its centre z, which makes two cells at the same (x, y) in
    # different planes exactly coincident. That is real geometry and the only condition
    # under which the schema permits coordinate ties.
    protocol = dict(adata.uns.get("paper_protocol") or {})
    flattened = bool(protocol.get("flattened_z", False))
    tmp = Path(args.input).with_suffix(".v25input.h5ad")
    adata.write_h5ad(tmp)
    try:
        volume = load_volume(tmp, cfg, flattened_sections=flattened)
    finally:
        tmp.unlink(missing_ok=True)
    return TrainingVolume(
        specimen_id=volume.specimen_id,
        sections=volume.sections,
        gene_names=volume.gene_names,
        celltype_names=volume.celltype_names,
        region_names=volume.region_names,
        flattened_sections=volume.flattened_sections,
    )


def build_embeddings(cfg, volume):
    """T02's entity embeddings for this panel, from the panel's own gene-metadata table.

    Delegates to ``model.build_entity_embeddings``. Two defects it replaces, both of which
    made ``text_emb_mode`` a gate with nothing behind it:

    * the descriptors were ``gene_descriptor(symbol, None)`` — a bare ``"Slc17a7."`` — so the
      full names and NCBI summaries in ``Config.gene_meta_path`` never reached MedCPT, and
      every STARmap number produced through this wrapper is ablation A3's ``lookup`` arm under
      another name;
    * the ``lookup`` arm was additionally handed a **zero** matrix, so the two arms of A3
      differed in two things at once. ``"lookup"`` is applied inside
      ``TextGroundedEmbedding._text_channel``; both arms now get the same vectors and differ
      only in the gate.

    The encoder needs ``transformers`` and the MedCPT weights. If they are unavailable this
    **raises** rather than substituting a lookup table: a silent downgrade would make the two
    arms indistinguishable and the shipped configuration unreproducible (Convention 6).
    """
    from spatialcpav25_gen.model.embeddings import build_entity_embeddings

    return build_entity_embeddings(cfg, volume.gene_names, volume.celltype_names, None)


def selection_pins(args):
    """``(pinned, reason)`` for gates this dataset must not re-select.

    ``--pin-gate layout_mode=resample`` excludes a gate from the merged full-budget gate and
    from coordinate descent. It is not a tuning flag: it names a gate whose evidence is
    *elsewhere*, and the selection report records it as pinned rather than as chosen.

    The saving is scoring, not fitting — ``layout_mode`` does not enter the fit
    (``select.FIT_INVARIANT_GATES``), so 6 fits already served the merged gate's 18 cells and
    pinning it removes 12 LOSO scorings.
    """
    pinned = {}
    for item in getattr(args, "pin_gate", None) or []:
        if "=" not in item:
            raise SystemExit(f"--pin-gate expects gate=option, got {item!r}")
        gate, option = item.split("=", 1)
        pinned[gate.strip()] = option.strip()
    if not pinned:
        return {}, ""
    return pinned, (
        "Pinned by `--pin-gate` on this invocation: "
        + ", ".join(f"`{g}` = `{o}`" for g, o in sorted(pinned.items()))
        + ". A pinned gate is one whose evidence is elsewhere; `specs/09` §3 selects per "
        "dataset, and re-opening a gate that real data has already settled, at one seed and "
        "inside a search whose own margins are envelope-sized, can only lose to noise."
    )


# ── run ───────────────────────────────────────────────────────────────────────

def fit_checkpoint_path(args):
    """Where this unit's fit checkpoint lives, or ``None`` if checkpointing is off.

    Beside the unit's own ``prediction.h5``, named for the seed, so it is scoped exactly the
    way ``run_all --skip-existing`` scopes a unit and two arms of the same dataset cannot
    collide. ``reports/durability.md``: the campaign driver already resumes at the unit level
    and the fit inside a unit did not, which is the whole of the gap this closes.

    Never committed and never inherited: a stale file from a different config, seed or step
    budget raises inside ``train_ctfflow`` rather than being continued.
    """
    if getattr(args, "no_fit_checkpoint", False):
        return None
    return os.path.join(os.path.dirname(os.path.abspath(args.output)) or ".",
                        f"fit_seed{args.seed}.pt")


def run_method(adata, targets, args, cfg, volume):
    from spatialcpav25_gen.infer.generate import generate_section, plane_at_z
    from spatialcpav25_gen.model.layout import fit_repulsion
    from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

    data = TrainingData.build(volume, cfg)
    model = CTFFlow(cfg, data, build_embeddings(cfg, volume), grf_seed=args.seed)
    checkpoint = fit_checkpoint_path(args)
    if checkpoint is not None:
        print(f"  fit checkpoint: {checkpoint} "
              f"(every {int(cfg.checkpoint_every_n_steps)} steps)")
    t0 = time.time()
    train_ctfflow(model, cfg, steps=int(cfg.train_steps), seed=args.seed,
                  checkpoint=checkpoint)
    if cfg.repulsion:
        model.repulsion = fit_repulsion(volume, cfg, seed=args.seed + 1)
    print(f"  fit: {int(cfg.train_steps)} steps in {time.time() - t0:.1f}s")

    results = {}
    for sec, z in targets:
        plane = plane_at_z(volume, float(z), cfg)
        print(f"  {sec}: field query at z={float(z):.2f} ...")
        emitted = generate_section(model, plane, volume, cfg, seed=args.seed)
        n = int(emitted.n_obs)
        if n == 0:
            continue
        print(f"    -> {n} cells synthesized")
        counts = emitted.X
        # _v2_io wants physical (N, 3). ``obsm[coord_key]`` is the in-plane (u, v) a section
        # reports — T01's documented 2-D exception — so take the 3-D positions that travel
        # beside it, which are correct for an oblique plane too, where cells share no depth.
        results[sec] = {
            "X": sp.csr_matrix(np.asarray(counts.toarray() if sp.issparse(counts) else counts,
                                          dtype=np.float32)),
            "coords": np.asarray(emitted.obsm["xyz"], dtype=np.float64),
            "cell_type": np.asarray(emitted.obs["cell_type"].values, dtype=str),
        }
    return results


def main() -> int:
    p = argparse.ArgumentParser(
        description="SpatialCPA-v25-Gen wrapper (CTF-Flow). No tuning flags — "
                    "configuration is selected internally per dataset.")
    _v2_io.add_v2_args(p)
    # selection control (not tuning: these choose *when* selection runs, never what it picks)
    p.add_argument("--select-only", action="store_true",
                   help="run per-dataset selection, persist it, write no prediction")
    p.add_argument("--require-config", action="store_true",
                   help="refuse to select; raise if no selected config exists for this dataset")
    p.add_argument("--pin-gate", action="append", default=None, metavar="GATE=OPTION",
                   help="exclude a gate from selection, fixing it to OPTION and recording it "
                        "as pinned in the report (repeatable). Not a tuning flag: it applies "
                        "only during --select-only and never overrides a selected config")
    # ablation switches (specs/10 §6). Each overrides ONE Config field after selection.
    p.add_argument("--prior-mode", default=None, choices=["correlated", "iid"], help="A1")
    p.add_argument("--w-autocorr", type=float, default=None, help="A2")
    p.add_argument("--w-profile", type=float, default=None, help="A2")
    p.add_argument("--w-distribution", type=float, default=None, help="A2")
    p.add_argument("--text-emb-mode", default=None, choices=["medcpt", "lookup"], help="A3")
    p.add_argument("--no-repulsion", action="store_true",
                   help="A4c — Poisson layout. Only meaningful together with "
                        "--layout-mode field/hybrid: resample never draws positions, so under "
                        "the shipped default this flag changes nothing")
    p.add_argument("--retrieval-w-z", type=float, default=None, help="A5")
    p.add_argument("--decoder", default=None, choices=["zinb", "zigamma", "gaussian"], help="A6")
    p.add_argument("--w-thick", type=float, default=None, help="A7")
    p.add_argument("--w-prog", type=float, default=None, help="A7")
    p.add_argument("--w-prog-wrong", type=float, default=None, help="A8 negative control")
    p.add_argument("--layout-mode", default=None, choices=["field", "hybrid", "resample"],
                   help="A4a/A4b — the generative layout, which ships off. The default is "
                        "resample (specs/05 section 4a): on real tissue the intensity-field "
                        "layout scores below a model-free copy of the flanking section")
    p.add_argument("--expr-mode", default=None, choices=["zinb-flow", "cross-mix", "auto-blend"])
    p.add_argument("--train-steps", type=int, default=None)
    # Not a tuning flag: it selects whether the fit is resumable, never what it computes.
    p.add_argument("--no-fit-checkpoint", action="store_true",
                   help="do not write a resumable fit checkpoint beside the prediction")
    args = p.parse_args()

    if not check_environment():
        return 1

    targets = _v2_io.load_targets(args)
    target_sections = [s for s, _ in targets]
    print(f"Loading training-only input {args.input} ...")
    adata = ad.read_h5ad(args.input)
    _v2_io.guard_no_holdout(adata, target_sections)
    print(f"  input: {adata.n_obs} cells x {adata.n_vars} genes, "
          f"{adata.obs['section'].nunique()} sections")

    dsid = dataset_id(adata, args.input)
    fingerprint = volume_fingerprint(adata, args.input)
    cfg, provenance = resolve_config(adata, args, fingerprint, dsid)
    print(f"  dataset_id={dsid} config={provenance['config_hash']} "
          f"selection={provenance.get('selection')} overrides={provenance['overrides'] or 'none'}")
    if args.select_only:
        print("  --select-only: selection persisted, no prediction written.")
        return 0

    volume = load_training_volume(adata, args, cfg)
    _ = leakage_guard  # the held-out sections are absent from the file by construction

    print(f"Running SpatialCPA-v25-Gen for targets "
          f"{[(s, round(float(z), 2)) for s, z in targets]} ...")
    t0 = time.time()
    results = run_method(adata, targets, args, cfg, volume)
    wall = time.time() - t0
    if not results:
        print("No sections synthesized.")
        return 1

    method_params = {
        "seed": args.seed,
        "design": "CTF-Flow (SpatialCPA-v25-Gen)",
        "generation_only": True,
        **{f"provenance_{k}": (json.dumps(v) if isinstance(v, dict) else v)
           for k, v in provenance.items()},
    }
    _v2_io.write_prediction_h5(
        results, list(adata.var_names), target_sections, method_params, wall,
        args.output, "spatialcpav25_gen")
    return 0


if __name__ == "__main__":
    sys.exit(main())
