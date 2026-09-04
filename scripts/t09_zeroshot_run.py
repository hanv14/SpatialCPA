"""Fit the zero-shot arms with a gene split held out of *every* channel that could leak it.

Two fits per seed, four evaluation arms: ``use_distill`` is a parameter of
:meth:`~spatialcpav25_gen.model.embeddings.TextGroundedEmbedding.forward_zero_shot`, not a
training setting, so ``medcpt`` serves arms A1/A2 and ``lookup`` serves A3/A4.

**The split is applied in three places, and all three are needed.** The pre-flight found that
holding genes out of training alone leaves two other routes into the model:

1. ``gene_pool`` in :func:`~spatialcpav25_gen.model.spatialcpav25_gen.train_ctfflow` — the batches
   never draw a held-out gene, and the same pool reaches ``reconstruct_hidden`` and the SEFL
   consistency losses.
2. ``gene_pool`` on ``TrainingData.build`` — restricts the **retrieval PCA** *and its size
   factor*. The ``zinb-flow`` path conditions on the retrieved neighbours' PCs, so a PCA fitted
   over the whole panel carries the held-out genes in as conditioning. Zeroing the excluded rows
   of the basis is necessary and **not sufficient**: the library size is a sum over every gene, so
   an excluded gene rescales every kept gene's normalised value. Both are closed, and the
   retrieval suite asserts the invariance rather than the mechanism.
3. ``expr_mode`` is pinned to ``zinb-flow`` and **``cross-mix`` is refused outright** — it reads
   ``model.data.counts``, the full training matrix, filtered by cell and never by gene, so it
   would emit the held-out genes verbatim. It is not a zero-shot method but a lookup of the
   answer, and a run that names it exits rather than producing a number.

Usage::

    python scripts/t09_zeroshot_run.py --dataset deep_starmap --seed 2 \\
        --split reports/t09_gene_split_deep.json --arm medcpt \\
        --workdir runs/zeroshot_s2 --fit-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from spatialcpav25_gen.model.field import BBoxClampWarning
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData, train_ctfflow

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import (
    base_config,
    clamp_config_to_input,
    describe_text_channel,
    embeddings_factory,
    load_training_volume,
)

TIMED_FIT_RECORD = "fit_timing.json"
"""Where the first fit writes its measured duration, inside ``--workdir``'s parent.

**A gate you can skip by accident is not a gate.** The "one timed fit before the other five"
rule has now been bypassed three times in this campaign — twice because runs went out before a
duration was reported, once because the loop was started early — so it is enforced in code
rather than in a message: the first fit writes this file, and every later fit refuses to start
until it exists."""


def timing_gate(root: Path, seed: int, arm: str, *, first: tuple[int, str]) -> None:
    """Refuse to start a non-first fit until the first one's duration is on disk.

    ``first`` is the ``(seed, arm)`` the gate designates as fit 1. That fit runs unconditionally
    and records its own time; anything else raises unless the record is present and carries a
    positive ``fit_seconds``.
    """
    record = root / TIMED_FIT_RECORD
    if (seed, arm) == first:
        return
    if not record.is_file():
        raise SystemExit(
            f"TIMING GATE: {record} does not exist, so fit 1 (seed {first[0]}, arm "
            f"{first[1]!r}) has not reported a measured duration. Run it first and let it "
            "finish; it writes the record itself. This gate exists because the same rule has "
            "been bypassed three times by hand — the cost of six fits is decided by the first "
            "one, and an estimate is not a measurement."
        )
    try:
        seconds = float(json.loads(record.read_text())["fit_seconds"])
    except (KeyError, ValueError, TypeError) as failure:
        raise SystemExit(
            f"TIMING GATE: {record} exists but carries no readable fit_seconds ({failure}). "
            "A record without a duration is what the gate is for; delete it and re-run fit 1."
        ) from failure
    if not seconds > 0.0:
        raise SystemExit(
            f"TIMING GATE: {record} reports fit_seconds={seconds}, which is not a "
            "measured duration."
        )
    print(f"  timing gate: fit 1 took {seconds / 3600.0:.2f} h ({seconds:.0f} s) — proceeding")


def write_timing(root: Path, seconds: float, seed: int, arm: str) -> None:
    """Record fit 1's measured duration where :func:`timing_gate` will look for it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / TIMED_FIT_RECORD).write_text(
        json.dumps(
            {"fit_seconds": float(seconds), "seed": int(seed), "arm": arm, "n_fits_planned": 6},
            indent=2,
        )
    )
    print(f"  wrote the timing record: {root / TIMED_FIT_RECORD}")


MIN_PANEL_WITH_META = 0.5
"""Fraction of the panel that must carry metadata before a fit is allowed to start.

**A structural-mismatch detector, not a quality bar.** A panel paired with its own table
resolves near-completely (`deep_starmap` 1017/1017; the `cosmx` human table 960/960). The
failure it exists to catch is a panel paired with **another organism's** table, which is silent:
`load_gene_meta` checks the table's rows against the requested species and both agree, so a
human panel read against the mouse table raises nothing and resolves **121 of 960** by
case-insensitive accident, leaving the rest bare.

The coverage gate cannot catch it either — `t09_zeroshot_text_coverage.py` is pointed at a table
by `--gene-meta` while the fit reads `Config.gene_meta_path`, so the gate can pass on one table
while the experiment runs on another. Half separates 12.6 % from 100 % with room on both sides
and cannot fire on a real pairing."""


def check_text_channel(described: dict, cfg, *, arm: str) -> None:
    """Refuse to fit when the panel and the metadata table are not the same organism's.

    ``described`` is :func:`describe_text_channel`'s record. The check runs on **both** arms:
    ``lookup`` applies its gate inside the embedding and is built from the same descriptors, so
    a ``lookup`` arm fitted against the wrong table is wrong in the same way — and A4's
    ``norm(0)`` void condition only means what it should if the text channel is what we think.
    """
    n_genes = int(described.get("n_genes", 0))
    n_meta = int(described.get("n_with_meta", 0))
    if n_genes <= 0:
        return
    fraction = n_meta / n_genes
    if fraction >= MIN_PANEL_WITH_META:
        return
    raise SystemExit(
        f"TEXT CHANNEL: only {n_meta}/{n_genes} ({fraction:.1%}) of the panel carries metadata "
        f"from {cfg.gene_meta_path} at species {cfg.mygene_species!r}, on arm {arm!r}. A panel "
        "paired with its own table resolves near-completely; this is what a panel read against "
        "ANOTHER ORGANISM's table looks like, and nothing upstream raises because that table is "
        "internally consistent. Pass --gene-meta and --species for this panel's organism. "
        "(The coverage check cannot catch this: it reads the table you point IT at, and the fit "
        "reads Config.gene_meta_path.)"
    )


def load_split(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the recorded gene split. ``(kept, held_out, record)``, both int64 panel indices."""
    record = json.loads(Path(path).read_text())
    return (
        np.asarray(record["kept"], dtype=np.int64),
        np.asarray(record["held_out"], dtype=np.int64),
        record,
    )


def arm_config(arm: str, seed: int, input_path: Path, *, train_steps: int, expr_pca_dim=None):
    """The config one arm is fitted under. ``expr_mode`` and ``prior_mode`` are pinned here.

    Shared with the scorer rather than reconstructed there: the checkpoint's ``config_hash``
    has to match or the resume refuses, and two copies of this drift.
    """
    overrides: dict = {"train_steps": int(train_steps), "text_emb_mode": arm}
    if expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(expr_pca_dim)
    cfg = clamp_config_to_input(base_config(seed, **overrides), input_path)
    return cfg.replace(expr_mode="zinb-flow", prior_mode="correlated")


def build_and_fit(cfg, volume, embeddings, kept, held, *, seed: int, checkpoint: Path) -> CTFFlow:
    """Build the model with the split applied and fit it, or resume a finished fit.

    The scorer calls this too, so the model it scores is built through the same path that
    fitted it — including ``gene_pool`` on ``TrainingData.build``. Rebuilding it without the
    pool would refit the retrieval PCA over the whole panel at *scoring* time and put the
    held-out genes straight back into the conditioning, with the checkpoint none the wiser.
    """
    data = TrainingData.build(volume, cfg, gene_pool=kept)
    if not np.all(data.index.expression_pcs.basis[held, :] == 0.0):
        raise SystemExit(
            "the retrieval PCA basis is non-zero on held-out genes after TrainingData.build "
            "was given the kept pool. The exclusion did not take; do not use this fit."
        )
    model = CTFFlow(cfg, data, embeddings(cfg), grf_seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BBoxClampWarning)
        train_ctfflow(
            model,
            cfg,
            steps=int(cfg.train_steps),
            seed=seed,
            gene_pool=kept,
            checkpoint=str(checkpoint),
        )
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--arm", required=True, choices=["medcpt", "lookup"])
    ap.add_argument("--split", required=True, help="the .json written by t09_zeroshot_ceiling.py")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-mode", default="zinb-flow")
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument(
        "--gene-meta",
        default=None,
        help="the panel's gene-metadata table (default: Config.gene_meta_path, which is the "
        "MOUSE table). A human panel fitted against it resolves a handful of symbols by "
        "case-insensitive accident and leaves the rest bare — silently, because the table is "
        "internally consistent. Required with --species",
    )
    ap.add_argument(
        "--species",
        default=None,
        help="the table's organism (default: Config.mygene_species = 'mouse'). Required "
        "whenever --gene-meta is given: the path and the species are one statement",
    )
    ap.add_argument("--fit-only", action="store_true")
    ap.add_argument(
        "--timing-root",
        default=None,
        help="directory holding the campaign's timing record (default: --workdir's parent). "
        "The first fit writes it; every later fit refuses to start without it",
    )
    ap.add_argument(
        "--first-fit",
        default="2:medcpt",
        help="the 'seed:arm' designated as fit 1 — the one fit the timing gate lets run "
        "unconditionally, and the one that records the duration the other five are gated on",
    )
    ap.add_argument(
        "--no-timing-gate",
        action="store_true",
        help="skip the gate. Deliberate only: it prints a refusal to the log naming the runs "
        "whose cost was never measured, so a skipped gate is visible in the artifact",
    )
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    if args.expr_mode != "zinb-flow":
        raise SystemExit(
            f"--expr-mode={args.expr_mode!r} is refused. `cross-mix` reads the full training "
            "count matrix, filtered by cell and never by gene, so it emits the held-out genes "
            "verbatim: it is a lookup of the answer, not a zero-shot method. The zero-shot "
            "experiment runs under zinb-flow only."
        )

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")

    timing_root = Path(args.timing_root) if args.timing_root else Path(args.workdir).parent
    first_seed, _, first_arm = str(args.first_fit).partition(":")
    if args.no_timing_gate:
        print(
            "  ⚠️ TIMING GATE SKIPPED by --no-timing-gate. The cost of the remaining fits is "
            "an extrapolation, not a measurement, and this line is the record of that."
        )
    else:
        timing_gate(timing_root, int(args.seed), args.arm, first=(int(first_seed), first_arm))

    if (args.gene_meta is None) != (args.species is None):
        raise SystemExit(
            "--gene-meta and --species are one statement: pass both or neither. A table is "
            "keyed by symbol and describes exactly one organism, and inferring the second from "
            "the first is the silent fallback Convention 6 forbids."
        )

    kept, held, split = load_split(args.split)
    cfg = arm_config(
        args.arm,
        args.seed,
        paths.input,
        train_steps=args.train_steps,
        expr_pca_dim=args.expr_pca_dim,
    )
    if args.gene_meta is not None:
        cfg = cfg.replace(gene_meta_path=args.gene_meta, mygene_species=args.species)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    if len(kept) + len(held) != volume.n_genes:
        raise SystemExit(
            f"{args.split} splits {len(kept) + len(held)} genes but the volume has "
            f"{volume.n_genes}. The split was written for a different panel."
        )
    if int(cfg.expr_pca_dim) > len(kept):
        raise SystemExit(
            f"Config.expr_pca_dim={cfg.expr_pca_dim} exceeds the {len(kept)} kept genes the "
            "retrieval PCA may use. Lower it with --expr-pca-dim or hold out fewer genes."
        )

    embeddings = embeddings_factory(volume)
    print(
        f"  arm {args.arm}, seed {args.seed}, {cfg.train_steps} steps, config "
        f"{cfg.content_hash()}; split "
        f"{len(kept)} kept / {len(held)} held out (seed {split['seed']}, "
        f"{split['n_bins']}x{split['n_bins']} strata on {split['reference_section']})"
    )
    described = describe_text_channel(cfg, volume)
    print("  text channel: " + json.dumps(described, default=str))
    check_text_channel(described, cfg, arm=args.arm)
    print(
        "  held out of: training batches (gene_pool), the retrieval PCA and its size factor "
        "(TrainingData.build gene_pool); cross-mix refused"
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    checkpoint = workdir / f"fit_zeroshot_{args.arm}_seed{args.seed}.pt"

    t0 = time.time()
    build_and_fit(cfg, volume, embeddings, kept, held, seed=args.seed, checkpoint=checkpoint)
    fit_seconds = time.time() - t0
    print(f"\n── zeroshot {args.arm} seed {args.seed} ({cfg.content_hash()}) ──")
    print(f"  fitted in {fit_seconds:.0f}s ({fit_seconds / 3600.0:.2f} h) -> {checkpoint}")
    if (int(args.seed), args.arm) == (int(first_seed), first_arm):
        write_timing(timing_root, fit_seconds, int(args.seed), args.arm)
        print(
            f"  This was fit 1. Six fits project to {6 * fit_seconds / 3600.0:.1f} core-hours; "
            "the other five are now unblocked. Report this number before starting them."
        )
    if args.fit_only:
        print("  --fit-only; scoring is a separate pass over these checkpoints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
