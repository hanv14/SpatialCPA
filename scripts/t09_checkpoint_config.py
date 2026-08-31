"""Which `decoder_mu_link` was a campaign fit actually trained under? Read it, do not infer it.

`Config.decoder_mu_link` has defaulted to **`exp`** since 2026-08-21 (commit
`decoder_mu_link defaults to exp: T06's own revisit condition, met by T10`), and every real-data
audit in this project is dated 2026-08-25 or later. So the negatives were fitted under `exp` and
not under `softplus`. That is an argument from dates and from `scripts/_starmap_run.BENCH3_KEYS`
not overriding the field — good enough to raise a doubt, **not** good enough to settle whether a
particular set of weights on a particular machine used it.

This settles it. `FitCheckpoint` stores the writing run's `config_hash`, and `decoder_mu_link` is
inside `Config.content_hash()` (`b6fb1c71844ffe7f` against `079785968f93ec11` on an otherwise
identical config), so the stored hash **identifies the link** given the rest of the config. The
script rebuilds the campaign config both ways and reports which one the checkpoint matches — or
neither, which would mean the fit differs from this config in some other field and nothing may be
concluded about the link from it.

Usage::

    python scripts/t09_checkpoint_config.py --dataset deep_starmap --holdout paper_2_4_6 \\
        runs/zeroshot_s2/fit_zeroshot_medcpt_seed2.pt runs/zeroshot_s3/*.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from spatialcpav25_gen.train.checkpoint import load_checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve
from t09_zeroshot_run import arm_config

NAME = re.compile(r"fit_zeroshot_(?P<arm>medcpt|lookup)_seed(?P<seed>\d+)\.pt$")
LINKS = ("exp", "softplus")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--train-steps", type=int, default=2400)
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    verdicts: list[str] = []
    for path in args.checkpoints:
        name = NAME.search(Path(path).name)
        if name is None:
            print(f"  {path}: not a zero-shot fit checkpoint name; skipped")
            continue
        arm, seed = name["arm"], int(name["seed"])
        saved = load_checkpoint(path)
        matches = [
            link
            for link in LINKS
            if arm_config(arm, seed, paths.input, train_steps=args.train_steps)
            .replace(decoder_mu_link=link)
            .content_hash()
            == saved.config_hash
        ]
        verdict = matches[0] if len(matches) == 1 else "NEITHER"
        verdicts.append(verdict)
        print(
            f"  {Path(path).name}: stored {saved.config_hash}, step {saved.step}/{saved.steps}"
            f"  ->  decoder_mu_link = {verdict}"
        )
    unique = set(verdicts)
    print()
    if unique == {"exp"}:
        print("All checkpoints were fitted under decoder_mu_link='exp'. The negatives measured")
        print("from them are not attributable to a softplus decoder, and no refit is owed.")
    elif "NEITHER" in unique:
        print("At least one checkpoint matches neither link. Its config differs from this one in")
        print("some other field too, so the link cannot be read off the hash — say so rather than")
        print("guessing, and find the run's own log line, which prints the config hash.")
    else:
        print(f"Mixed or softplus: {sorted(unique)}. Anything measured from a softplus fit is")
        print("owed a re-run under the shipped default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
