"""Uninformative condition (d): is the scorer's ``constant_field`` degenerate on THIS dataset?

The replication's pre-registration excludes ``constant_field`` as a floor on the authority of
``specs/10`` §4.2c, and §4.2c was established on ``deep_starmap`` and the synthetic fixture. That
exclusion is a claim about the referent's **input**, and the pre-registration makes it a
checkable condition rather than an inheritance: uninformative **(d)** fires if *"the constant
field's normalised input is not bitwise row-identical on this dataset"*.

**Inferring it from the scores is not the check.** The constant field is a deterministic function
of the training volume, so its score is identical across seeds whatever its input looks like; a
spread of exactly 0.0000 in the seed files is consistent with degeneracy and also with a
perfectly reproducible non-degenerate referent. The condition is about the rows, so this reads
the rows.

**Two statistics, and the second is the one §4.2c actually settled on.**

*Bitwise row identity* is the condition as written: every cell's normalised vector equal to every
other's, exactly. It is the strict form and it is what (d) names.

*Across-cell coefficient of variation, per gene* is the measured form §4.2c uses, after a
precision-drift threshold was tried and **failed** on real data (``deep_starmap``'s eight
constant-field rows drifted 0.0042-0.1905, a continuum with no gap, so the cut fell between two
rows of identical construction). On ``deep_starmap`` the CV separated by eight orders of
magnitude: **2.6e-07** for the constant field against **16.1** for a real-counts referent on the
same rows. Reported here for the same comparison so the two datasets can be read against each
other.

**Model-free, and deliberately so.** This instrument builds no model and loads no checkpoint: the
constant field is ``volume_mean`` broadcast over the fold's cells, exactly as
``t09_zeroshot_score.referents`` builds it, and the normalisation is imported from
``train.select`` rather than reimplemented — a reimplementation would answer a question about
this script instead of about the scorer. Nothing here constructs an
:class:`~spatialcpav25_gen.model.embeddings.EntityEmbeddings`, so ``Config.gene_meta_path`` is
never read and this script takes no ``--gene-meta`` flag. That is why it could have been run
before the fits, and why it can be re-run now for nothing.

Usage::

    python scripts/t09_zeroshot_degeneracy.py --dataset cosmx --holdout paper_2_4 \\
        --split reports/t09_gene_split_cosmx.json \\
        --out reports/t09_degeneracy_cosmx.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from spatialcpav25_gen.data.schema import TrainingVolume
from spatialcpav25_gen.train.select import _normalised, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import load_training_volume
from t09_zeroshot_run import arm_config, load_split

DEEP_STARMAP_CV = (2.6e-07, 16.1)
"""(constant field, real counts) across-cell CV on ``deep_starmap``, from ``specs/10`` §4.2c."""


def volume_mean(vol: TrainingVolume) -> np.ndarray:
    """``(G,)`` mean count per gene over every cell in the training volume.

    The same expression ``t09_zeroshot_score.referents`` uses, so the referent measured here is
    the referent the seed files were scored against and not a lookalike.
    """
    stacked = np.concatenate(
        [np.asarray(s.counts.todense(), dtype=np.float64) for s in vol.sections], axis=0
    )
    return np.asarray(stacked.mean(axis=0), dtype=np.float64)


def row_statistics(counts: np.ndarray, cfg: Any, pool: np.ndarray) -> dict[str, Any]:
    """Degeneracy statistics for one ``(N, G)`` count matrix, on one gene pool.

    Returns the strict condition (``bitwise_row_identical``) and the measured one
    (``cv_median`` / ``cv_max``, across cells, per gene, over the pool's genes only — the pool is
    what the metric was scored on and a gene outside it has no bearing on the referent's
    validity there).
    """
    x = _normalised(counts, cfg, pool).numpy()
    scored = x[:, np.asarray(pool, dtype=np.int64)]
    deviation = np.abs(scored - scored[0:1, :])
    mean = scored.mean(axis=0)
    std = scored.std(axis=0)
    cv = np.divide(std, np.abs(mean), out=np.zeros_like(std), where=np.abs(mean) > 0)
    return {
        "n_cells": int(scored.shape[0]),
        "n_genes_scored": int(scored.shape[1]),
        "bitwise_row_identical": bool(np.all(deviation == 0.0)),
        "max_abs_row_deviation": float(deviation.max()),
        "cv_median": float(np.median(cv)),
        "cv_max": float(cv.max()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True, help="the .json written by t09_zeroshot_ceiling.py")
    ap.add_argument("--out", default=None, help="destination .json")
    ap.add_argument("--seed", type=int, default=2, help="only selects the config; nothing is fit")
    ap.add_argument("--train-steps", type=int, default=2400)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    kept, held, _ = load_split(args.split)
    cfg = arm_config(
        "medcpt",
        args.seed,
        paths.input,
        train_steps=args.train_steps,
        expr_pca_dim=args.expr_pca_dim,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    if len(kept) + len(held) != volume.n_genes:
        raise SystemExit(
            f"{args.split} splits {len(kept) + len(held)} genes but the volume has "
            f"{volume.n_genes}. The split was written for a different panel."
        )

    folds = selection_folds(volume, cfg)
    means = volume_mean(volume)
    print(f"  folds {[s.section_id for s in folds]}; {len(kept)} kept / {len(held)} held out")

    report: list[dict[str, Any]] = []
    for fold in folds:
        real = np.asarray(fold.counts.todense(), dtype=np.float64)
        constant = np.broadcast_to(means, real.shape).copy()
        for name, pool in (("held_out", held), ("kept", kept)):
            row = {
                "fold": str(fold.section_id),
                "side": name,
                "constant_field": row_statistics(constant, cfg, pool),
                "real_counts": row_statistics(real, cfg, pool),
            }
            report.append(row)
            c, r = row["constant_field"], row["real_counts"]
            print(
                f"  {fold.section_id} {name:9s} constant: identical={c['bitwise_row_identical']} "
                f"max_dev={c['max_abs_row_deviation']:.3e} cv_med={c['cv_median']:.3e} "
                f"| real: cv_med={r['cv_median']:.3f}"
            )

    fired = [r for r in report if not r["constant_field"]["bitwise_row_identical"]]
    verdict = "uninformative_d_fires" if fired else "constant_field_is_degenerate"
    print()
    if fired:
        print(
            "🚨 UNINFORMATIVE (d) FIRES: the constant field's normalised input is NOT bitwise "
            f"row-identical on {len(fired)} of {len(report)} (fold, pool) cells. §4.2c's "
            "exclusion was established on another dataset and does not transfer here; the "
            "referent analysis the pre-registration rests on does not hold, and no verdict may "
            "be read from these fits until that is resolved."
        )
    else:
        print(
            "✅ The constant field is degenerate on this dataset: every cell's normalised vector "
            "is bitwise identical on every fold and both gene pools. §4.2c's exclusion transfers, "
            "uninformative (d) does not fire on this condition, and `shuffled` is the floor."
        )
    cv_const = max(r["constant_field"]["cv_median"] for r in report)
    cv_real = min(r["real_counts"]["cv_median"] for r in report)
    print(
        f"  Across-cell CV, worst case: constant {cv_const:.3e} against real counts {cv_real:.3f} "
        f"— {cv_real / cv_const:.1e}x. `deep_starmap` measured "
        f"{DEEP_STARMAP_CV[0]:.1e} against {DEEP_STARMAP_CV[1]}."
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(
                {"verdict": verdict, "deep_starmap_cv": DEEP_STARMAP_CV, "cells": report}, indent=2
            )
        )
        print(f"\nwrote {args.out}")
    return 1 if fired else 0


if __name__ == "__main__":
    sys.exit(main())
