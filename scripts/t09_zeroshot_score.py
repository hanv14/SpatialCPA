"""Score the four zero-shot arms on one seed's two fits, over the held-out genes.

``use_distill`` is a parameter of :meth:`TextGroundedEmbedding.forward_zero_shot`, not a
training setting, so two fits carry four arms:

===  =================  ==============  ==============================================
arm  ``text_emb_mode``  ``use_distill``  an unseen gene's embedding
===  =================  ==============  ==============================================
A1   ``medcpt``         ``True``        ``norm(W t + gamma psi(t))`` — the full claim
A2   ``medcpt``         ``False``       ``norm(W t)`` — the designed channel alone
A3   ``lookup``         ``True``        ``norm(gamma psi(t))`` — **the real competitor**
A4   ``lookup``         ``False``       ``norm(0)`` — one vector per gene; the void condition
===  =================  ==============  ==============================================

**Every arm is generated through** :class:`~spatialcpav25_gen.model.embeddings.ZeroShotView`,
including the two that would not need it. A2 and A4 are algebraically plain generation — a gene
whose residual is still its zeros init has ``forward`` equal to ``forward_zero_shot(t, False)``
— but not bitwise: the view runs ``W`` over the unseen rows alone and a float32 GEMM at two
batch sizes rounds differently, 3e-8 on ``W t`` and 1.3e-6 after the ``LayerNorm``
(``tests/test_text.py`` measures it). Routing two arms through one path and two through another
would put that difference *inside* the comparison, so all four take the same path and the only
things that vary are ``use_distill`` and the fit.

**The referents are computed here, on these folds, on these markers.** The pre-flight's ceiling
report measured a constant-field band too, but on its own marker selection; a band from one
script and scores from another are not comparable, and the band is what A1 has to clear and A4
has to stay inside. Both referents reuse an arm's own generated layout with its counts replaced
— constant field: every cell gets that gene's training-volume mean; shuffled: the cells keep
their counts and lose their positions — so what they isolate is the expression, not the layout.

The layout must be **identical across the four arms** or the comparison is not about the text
channel at all. Under ``layout_mode=resample`` it is, by construction, and this script asserts
it rather than assuming it.

**This script reads the gene-metadata table, and it must be the same one the fits used.**
"It only scores existing checkpoints" is the reasoning that would let the flags be skipped, and
it is wrong: ``build_and_fit`` calls ``embeddings_factory(volume)(cfg)``, which is
:func:`~spatialcpav25_gen.model.embeddings.build_entity_embeddings` reading
``cfg.gene_meta_path`` and ``cfg.mygene_species`` and encoding this panel's descriptors through
MedCPT — *before* the checkpoint is loaded over the top. And the held-out genes' text is what
the whole experiment measures: ``ZeroShotView`` embeds an unseen gene as
``forward_zero_shot(base.text_vecs[g])``, so ``text_vecs`` **is** the zero-shot input, not an
incidental buffer.

Two guards, and they catch different things. ``gene_meta_path`` and ``mygene_species`` are both
in ``Config.content_hash``, so a scorer pointed at the wrong table fails the checkpoint's
``require_compatible`` — but it fails *after* encoding a panel of wrong descriptors, with a
message that names a hash and no field. ``check_text_channel`` runs first and names the table,
the species and the resolved fraction, which is the message a reader can act on. The run script
had this same hole and it cost a fit (``progress/t09_inference_and_calibration.md``,
2026-09-01); there the failure was silent, here it is merely unreadable, and neither is
acceptable when one flag fixes both.

Usage::

    python scripts/t09_zeroshot_score.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --split reports/t09_gene_split_deep.json --seed 2 --workdir runs/zeroshot_s2 \\
        --out reports/t09_zeroshot_deep_seed2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, TrainingVolume
from spatialcpav25_gen.infer.generate import emitted_counts, generate_section
from spatialcpav25_gen.infer.planes import section_plane
from spatialcpav25_gen.losses.metric_aware import profile_axis
from spatialcpav25_gen.model.embeddings import ZeroShotView
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow
from spatialcpav25_gen.train.select import METRIC_NAMES, section_scores, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import describe_text_channel, embeddings_factory, load_training_volume
from t09_zeroshot_run import arm_config, build_and_fit, check_text_channel, load_split

ARMS: dict[str, tuple[str, bool]] = {
    "A1": ("medcpt", True),
    "A2": ("medcpt", False),
    "A3": ("lookup", True),
    "A4": ("lookup", False),
}
"""arm -> (``text_emb_mode`` of the fit, ``use_distill`` at generation)."""

SIDES = ("held_out", "kept")
"""Both gene pools are scored. ``kept`` is the in-sample control: an arm that is worse there
too has a fit problem, not a zero-shot result."""


def generated(model: CTFFlow, hidden: Section, vol: TrainingVolume, cfg: Config, seed: int) -> Any:
    """Generate ``hidden``'s plane with the section itself excluded from the retrieval pool."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return generate_section(
            model, section_plane(hidden), vol, cfg, seed, exclude_z={float(hidden.z)}
        )


def layout_of(adata: Any, vol: TrainingVolume, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """``(N, 2)`` physical coordinates and ``(N,)`` cell-type codes of a generated section.

    ``obsm["xyz"]``, never ``obsm[cfg.coord_key]``: the latter is the plane-local ``(u, v)``,
    and scoring it against a real section's physical frame is the defect ``assert_same_frame``
    exists to refuse.
    """
    coords = np.asarray(adata.obsm["xyz"], dtype=np.float64)[:, :2]
    types = np.asarray(
        [vol.celltype_names.index(v) for v in adata.obs[cfg.celltype_key]], dtype=np.int64
    )
    return coords, types


def referents(
    counts: np.ndarray,
    coords: np.ndarray,
    types: np.ndarray,
    hidden: Section,
    vol: TrainingVolume,
    axis: Any,
    cfg: Config,
    pool: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """The two model-free bands the arms are read against, on this fold's own layout.

    ``constant_field``
        Every cell gets each gene's mean over the whole training volume — a section with no
        spatial expression information at all. It is **not** a zero score: ``soft_depth_profile``
        normalises each bin by the weight it received, so a constant field's profile tracks cell
        density along the depth axis, and density is laminar. Measured, never assumed.
    ``shuffled``
        The same cells and the same counts with the positions permuted, which destroys the
        pairing while keeping both marginals.
    """
    volume_mean = np.concatenate(
        [np.asarray(s.counts.todense(), dtype=np.float64) for s in vol.sections], axis=0
    ).mean(axis=0)
    constant = np.broadcast_to(volume_mean, counts.shape).copy()
    order = np.random.default_rng(seed).permutation(coords.shape[0])
    common = {"n_types": len(vol.celltype_names), "gene_pool": pool}
    return {
        "constant_field": section_scores(constant, coords, types, hidden, axis, cfg, **common),
        "shuffled": section_scores(
            counts, coords[order], types[order], hidden, axis, cfg, **common
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--workdir", required=True, help="where t09_zeroshot_run.py wrote the fits")
    ap.add_argument("--out", required=True, help="destination .json")
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument(
        "--gene-meta",
        default=None,
        help="the panel's gene-metadata table, and it MUST be the one the fits were run "
        "under: this script rebuilds the embeddings from it before loading the checkpoint. "
        "Default Config.gene_meta_path, which is the MOUSE table. Required with --species",
    )
    ap.add_argument(
        "--species",
        default=None,
        help="the table's organism (default: Config.mygene_species = 'mouse'). Required "
        "whenever --gene-meta is given: the path and the species are one statement",
    )
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    if (args.gene_meta is None) != (args.species is None):
        raise SystemExit(
            "--gene-meta and --species are one statement and must be given together. The "
            "path alone leaves Config.mygene_species at 'mouse', which makes load_gene_meta's "
            "organism check agree with itself while the panel is another organism's."
        )

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    kept, held, split = load_split(args.split)
    pools = {"held_out": held, "kept": kept}

    fits = {}
    for mode in ("medcpt", "lookup"):
        cfg = arm_config(
            mode,
            args.seed,
            paths.input,
            train_steps=args.train_steps,
            expr_pca_dim=args.expr_pca_dim,
        )
        if args.gene_meta is not None:
            cfg = cfg.replace(gene_meta_path=args.gene_meta, mygene_species=args.species)
        checkpoint = Path(args.workdir) / f"fit_zeroshot_{mode}_seed{args.seed}.pt"
        if not checkpoint.is_file():
            raise SystemExit(
                f"no fit at {checkpoint}. Run scripts/t09_zeroshot_run.py --arm {mode} "
                f"--seed {args.seed} --workdir {args.workdir} first; this script only scores."
            )
        fits[mode] = (cfg, checkpoint)

    # The volume, the folds and the depth axis do not depend on `text_emb_mode`; one of the
    # two configs is named explicitly rather than left to whichever the loop above ended on.
    shared = fits["medcpt"][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(shared, paths.input, flattened=args.flattened)
    if len(kept) + len(held) != volume.n_genes:
        raise SystemExit(
            f"{args.split} splits {len(kept) + len(held)} genes but the volume has "
            f"{volume.n_genes}. The split was written for a different panel."
        )
    described = describe_text_channel(shared, volume)
    print("  text channel: " + json.dumps(described, default=str))
    check_text_channel(described, shared, arm="score")

    embeddings = embeddings_factory(volume)
    folds = selection_folds(volume, shared)
    axis = profile_axis(volume, shared)
    print(
        f"  seed {args.seed}; split {len(kept)} kept / {len(held)} held out "
        f"(seed {split['seed']}); folds {[s.section_id for s in folds]}"
    )

    models: dict[str, CTFFlow] = {}
    for mode, (mode_cfg, checkpoint) in fits.items():
        t0 = time.time()
        models[mode] = build_and_fit(
            mode_cfg, volume, embeddings, kept, held, seed=args.seed, checkpoint=checkpoint
        )
        print(f"  loaded {mode} ({mode_cfg.content_hash()}) in {time.time() - t0:.0f}s")

    rows: list[dict[str, Any]] = []
    for index, hidden in enumerate(folds):
        seed = args.seed + index
        layouts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        emitted: dict[str, np.ndarray] = {}
        for arm, (mode, use_distill) in ARMS.items():
            model = models[mode]
            base = model.embeddings.gene
            model.embeddings.gene = ZeroShotView(
                base, torch.from_numpy(held), use_distill=use_distill
            )
            try:
                adata = generated(model, hidden, volume, fits[mode][0], seed)
            finally:
                model.embeddings.gene = base
            emitted[arm] = emitted_counts(adata)
            layouts[arm] = layout_of(adata, volume, fits[mode][0])

        reference = layouts["A1"]
        for arm, (coords, types) in layouts.items():
            if not (np.array_equal(coords, reference[0]) and np.array_equal(types, reference[1])):
                raise SystemExit(
                    f"fold {hidden.section_id}: arm {arm}'s layout differs from A1's. The four "
                    "arms differ only in the gene embedding, which the layout head does not "
                    "read, so identical layouts are the precondition for reading the gap as a "
                    "statement about the text channel. Under layout_mode=resample they are "
                    "identical by construction; this run's layout_mode is "
                    f"{shared.layout_mode!r}."
                )

        coords, types = reference
        for side in SIDES:
            pool = pools[side]
            for arm in ARMS:
                scores = section_scores(
                    emitted[arm],
                    coords,
                    types,
                    hidden,
                    axis,
                    fits[ARMS[arm][0]][0],
                    n_types=len(volume.celltype_names),
                    gene_pool=pool,
                )
                rows.append(
                    {
                        "fold": hidden.section_id,
                        "side": side,
                        "arm": arm,
                        "text_emb_mode": ARMS[arm][0],
                        "use_distill": ARMS[arm][1],
                        "seed": int(args.seed),
                        **scores,
                    }
                )
                print(
                    f"  {hidden.section_id} {side:9s} {arm}  "
                    + "  ".join(f"{m}={scores[m]:+.4f}" for m in METRIC_NAMES)
                )
            for name, scores in referents(
                emitted["A1"], coords, types, hidden, volume, axis, shared, pool, seed
            ).items():
                rows.append(
                    {
                        "fold": hidden.section_id,
                        "side": side,
                        "arm": name,
                        "text_emb_mode": None,
                        "use_distill": None,
                        "seed": int(args.seed),
                        **scores,
                    }
                )
                print(f"  {hidden.section_id} {side:9s} {name:14s} " + f"{scores}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
