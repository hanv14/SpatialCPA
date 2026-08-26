"""Where bench3, its v2 wrappers and the built STARmap tier-1 files live.

The T10 pilot scripts resolved these relative to *this* checkout, because in the pilot
container the benchmark trees and the built data sat inside it. On a campaign machine bench3 is
its own tree with the built dataset in it, so every path here is a flag with an
environment-variable default, and nothing falls back silently: a missing file names itself, the
flag that sets it and the variable that would have set it (Convention 6).

One flag is normally enough. ``--bench3 <dir>`` fixes the rest by bench3's own layout
conventions:

* the importable package at ``<dir>/src``,
* v2's shared method wrappers (``_v2_io``, the prediction writer every wrapper speaks) at
  ``<dir>/../benchmark-pbya-v2/src/benchmark/methods`` — bench3's ``config.py`` makes exactly
  the same assumption (``REPO_ROOT = PROJECT_ROOT.parent``),
* the built dataset under ``$BENCH_V3_DATA`` or ``<dir>/data/processed``,
* the leakage-guarded training input under ``$BENCH_V3_RESULTS`` or ``<dir>/results``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = "starmap_visual_cortex"
"""Default dataset. Tier 1 (``specs/10`` §1); ``--dataset`` selects another built one."""

HOLDOUT = "paper_2_4_6"
"""Default holdout id. ``deep_starmap`` runs the same design, which is why it is the analogue
``specs/10`` §5.4 picks for E1."""


def _default_bench3() -> Path:
    env = os.environ.get("SPATIALCPA_BENCH3")
    return Path(env) if env else REPO_ROOT / "benchmark-pbya-v3"


@dataclass(frozen=True)
class Bench3Paths:
    """Resolved, existence-checked locations. ``sys.path`` is already prepared."""

    bench3: Path
    src: Path
    v2_methods: Path
    input: Path
    ground_truth: Path
    dataset: str = DATASET
    holdout: str = HOLDOUT

    def describe(self) -> str:
        return (
            f"bench3      {self.bench3}\n"
            f"  src       {self.src}\n"
            f"  v2 methods{self.v2_methods}\n"
            f"  input     {self.input}\n"
            f"  truth     {self.ground_truth}\n"
            f"  dataset   {self.dataset}  holdout {self.holdout}"
        )


def add_path_args(ap: argparse.ArgumentParser) -> None:
    """Add ``--bench3`` and the three per-file overrides to ``ap``."""
    g = ap.add_argument_group("bench3 paths")
    g.add_argument(
        "--bench3",
        default=None,
        help="benchmark-pbya-v3 root (env SPATIALCPA_BENCH3; default: this repo's copy). "
        "The other three default from it.",
    )
    g.add_argument(
        "--dataset",
        default=DATASET,
        help=f"built dataset name under $BENCH_V3_DATA (default: {DATASET}). Selects both the "
        "ground truth and the leakage-guarded input, so one flag moves a run to another "
        "dataset; --input / --ground-truth still override it individually",
    )
    g.add_argument(
        "--holdout",
        default=HOLDOUT,
        help=f"holdout id under the dataset's _inputs/ (default: {HOLDOUT})",
    )
    g.add_argument("--v2-methods", default=None, help="dir holding _v2_io.py (v2's methods/)")
    g.add_argument("--input", default=None, help="training-only input h5ad (train_registered)")
    g.add_argument("--ground-truth", default=None, help="built dataset h5ad scored against")


def _require(path: Path, what: str, flag: str, env: str | None = None) -> Path:
    if not path.exists():
        hint = f" or {env}" if env else ""
        raise SystemExit(
            f"{what} not found at {path}. Set it with {flag}{hint}. "
            "Nothing is guessed: a scored number has to say which tree it came from."
        )
    return path


def resolve(args: argparse.Namespace, *, need_input: bool = True) -> Bench3Paths:
    """Resolve the paths, check they exist, and put bench3 + ``_v2_io`` on ``sys.path``.

    ``need_input`` is False for a caller that only scores predictions already on disk and never
    loads the training volume.
    """
    bench3 = Path(args.bench3).resolve() if args.bench3 else _default_bench3().resolve()
    _require(bench3, "benchmark-pbya-v3", "--bench3", "SPATIALCPA_BENCH3")
    src = _require(bench3 / "src", "the bench3 package", "--bench3")

    v2 = (
        Path(args.v2_methods)
        if getattr(args, "v2_methods", None)
        else bench3.parent / "benchmark-pbya-v2" / "src" / "benchmark" / "methods"
    )
    _require(v2, "v2's method wrappers (_v2_io.py)", "--v2-methods")

    data_dir = Path(os.environ.get("BENCH_V3_DATA", bench3 / "data" / "processed"))
    results_dir = Path(os.environ.get("BENCH_V3_RESULTS", bench3 / "results"))
    dataset = str(getattr(args, "dataset", None) or DATASET)
    holdout = str(getattr(args, "holdout", None) or HOLDOUT)
    gt = (
        Path(args.ground_truth)
        if getattr(args, "ground_truth", None)
        else data_dir / dataset / "data.h5ad"
    )
    _require(gt, "the built dataset", "--ground-truth", "BENCH_V3_DATA")
    inp = (
        Path(args.input)
        if getattr(args, "input", None)
        else results_dir / "_inputs" / dataset / holdout / "train_registered.h5ad"
    )
    if need_input:
        _require(
            inp,
            "the leakage-guarded training input",
            "--input",
            "BENCH_V3_RESULTS (build it with `python -m bench3.selftest`)",
        )

    for p in (str(src), str(v2)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return Bench3Paths(
        bench3=bench3,
        src=src,
        v2_methods=v2,
        input=inp,
        ground_truth=gt,
        dataset=dataset,
        holdout=holdout,
    )


def set_torch_threads() -> int:
    """Pin torch to ``$OMP_NUM_THREADS`` when it is set. Returns the thread count in force.

    The campaign box is shared and wide: a fit that grabs every core for BLAS is the
    antisocial case, and these scripts are meant to run several to a box, one thread each.
    """
    import torch

    n = os.environ.get("OMP_NUM_THREADS")
    if n:
        torch.set_num_threads(int(n))
    return int(torch.get_num_threads())
