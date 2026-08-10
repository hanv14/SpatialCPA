"""Holdout designs for benchmark-pbya-v3.

The **paper design** is the default and the one the whole v3 folder exists for:

    input     : section_1, section_3, section_5, section_7
    held out  : section_2, section_4, section_6   (all three at once)

Holding all three out *simultaneously* — rather than one at a time — is what the
SpatialZ paper did, and it is a materially harder and more honest task: the
method reconstructs three missing slices from a volume that is half absent, and
never sees a neighbouring real slice closer than the flanking pair. Each held-out
section is still bracketed by two input sections (1|3, 3|5, 5|7), so the problem
stays well-posed for every interpolation method.

``loo`` is available as a robustness check: hold out one interior section at a
time, keeping the other six. It is an *easier* task (the flanking slices are
adjacent), so its numbers are not comparable with the paper design — they are
useful only for asking "does the ranking survive a change of difficulty?".

``wide`` is the opposite end: hold out a *consecutive* interior block of sections
at once (default three), keeping the first and last as input. Where the paper's
alternating hold-out keeps every held-out slice adjacent to a real one, a
consecutive block leaves its middle slices far from any input — the wide-gap
regime where flank-level copies carry an irreducible error and interpolation
quality actually separates methods (this is what SpatialCPA-v19 targets). The
block is centred and the two boundary sections are always kept as input, so every
held-out slice stays bracketed on both sides (well-posed interpolation, never
extrapolation). Its numbers are not comparable with the paper design either — a
different, deliberately harder difficulty — and its results live under distinct
``wide_*`` holdout ids so they never mix with ``paper_*``.

A design is a JSON-serializable dict with the same keys v2's ``run_benchmark``
expects, so the leakage machinery is reused verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import DATASET_PATH, HELD_OUT_SECTIONS, resolve_dataset_arg


def sorted_sections(adata):
    """Section labels sorted by median z, plus the label -> z map."""
    labels = [str(s) for s in np.unique(adata.obs["section"].values.astype(str))]
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    secs = adata.obs["section"].values.astype(str)
    z_by = {s: float(np.median(coords[secs == s, 2])) for s in labels}
    return sorted(labels, key=lambda s: z_by[s]), z_by


def held_out_for(adata):
    """The held-out sections this dataset was built with.

    Read from the file rather than from config, so a dataset built with a
    different section count or split cannot be silently paired with STARmap's.
    """
    # h5ad round-trips a list of strings as a numpy array, so test it explicitly
    # rather than with `or` — an ndarray has no unambiguous truth value.
    pp = adata.uns.get("paper_protocol")
    raw = pp.get("held_out_sections") if pp is not None else None
    held = [str(s) for s in raw] if raw is not None and len(raw) else []
    return tuple(held) if held else HELD_OUT_SECTIONS


def holdout_id_for(holdout_sections, n_sections):
    """Stable directory name for a hold-out set.

    Spelling the indices out keeps STARmap's id exactly ``paper_2_4_6`` — the
    published split, and the name existing results already live under. A dataset
    that holds out more sections than that would make an unwieldy directory name,
    so it gets a summary form instead.
    """
    idx = []
    for sec in holdout_sections:
        try:
            idx.append(int(str(sec).rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            idx = []
            break
    if idx and len(idx) <= 3:
        return "paper_" + "_".join(str(i) for i in sorted(idx))
    return f"paper_alt{len(holdout_sections)}of{n_sections}"


def paper_design(adata):
    """The SpatialZ-paper design: hold out sections 2, 4 and 6 together."""
    labels, z_by = sorted_sections(adata)
    HELD_OUT_SECTIONS = held_out_for(adata)
    missing = [s for s in HELD_OUT_SECTIONS if s not in labels]
    if missing:
        raise ValueError(
            f"dataset is missing held-out sections {missing}; rebuild it with "
            f"`python -m src.bench3.prepare_dataset --dataset <name>`"
        )
    holdout = [s for s in labels if s in HELD_OUT_SECTIONS]
    remaining = [s for s in labels if s not in HELD_OUT_SECTIONS]
    return [{
        "holdout_id": holdout_id_for(holdout, len(labels)),
        "design": "paper",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by[s] for s in holdout},
    }]


def loo_design(adata, exclude_boundary=True):
    """Robustness check: hold out one interior section at a time."""
    labels, z_by = sorted_sections(adata)
    start = 1 if exclude_boundary else 0
    end = len(labels) - 1 if exclude_boundary else len(labels)
    out = []
    for sec in labels[start:end]:
        out.append({
            "holdout_id": f"loo_{sec}",
            "design": "loo",
            "holdout_sections": [sec],
            "remaining_sections": [s for s in labels if s != sec],
            "holdout_z": {sec: z_by[sec]},
        })
    return out


# Default width of the consecutive hold-out block in the ``wide`` design. Three
# matches SpatialCPA-v19's wide-gap validation and the paper design's own count
# of held-out sections, so ``paper`` and ``wide`` remove the same number of slices
# and differ only in whether they are adjacent — isolating the gap width itself.
DEFAULT_WIDE_BLOCK = 3


def _wide_id(holdout_sections, n_sections):
    """Stable directory name for a consecutive hold-out block, e.g. ``wide_3_4_5``.

    Distinct from ``paper_*`` and ``loo_*`` on purpose, so a wide-gap run never
    writes into (or is aggregated with) a paper-design result.
    """
    idx = []
    for sec in holdout_sections:
        try:
            idx.append(int(str(sec).rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            idx = []
            break
    if idx:
        return "wide_" + "_".join(str(i) for i in sorted(idx))
    return f"wide_{len(holdout_sections)}of{n_sections}"


def consecutive_design(adata, block=None):
    """Wide-gap design: hold out a consecutive INTERIOR block of sections at once.

    ``block`` is the number of consecutive sections to remove (default
    ``DEFAULT_WIDE_BLOCK``). The block is centred in the volume and clamped to at
    most ``n - 2`` so the first and last sections always remain as input — every
    held-out slice is therefore bracketed by input slices on both sides, keeping
    the task interpolation rather than extrapolation, only with a much wider gap
    than the paper design.
    """
    labels, z_by = sorted_sections(adata)
    n = len(labels)
    if n < 3:
        raise ValueError(
            f"consecutive (wide) design needs >= 3 sections to keep both "
            f"boundaries as input; this dataset has {n}"
        )
    block = DEFAULT_WIDE_BLOCK if block is None else int(block)
    if block < 1:
        raise ValueError(f"holdout block must be >= 1 (got {block})")
    block = min(block, n - 2)                       # keep first & last as input
    start = (n - block) // 2                        # centre the block
    start = max(1, min(start, n - 1 - block))       # stay strictly interior
    holdout = labels[start:start + block]
    remaining = [s for s in labels if s not in holdout]
    return [{
        "holdout_id": _wide_id(holdout, n),
        "design": "wide",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by[s] for s in holdout},
    }]


DESIGNS = {"paper": paper_design, "loo": loo_design, "wide": consecutive_design}


def build_designs(h5ad_path=DATASET_PATH, design="paper", holdout_block=None):
    """Load the dataset and return the holdout configs for ``design``.

    ``holdout_block`` sets the consecutive block width for ``design='wide'`` and
    is ignored by the other designs.
    """
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {h5ad_path}\n"
            f"Build it first: python -m src.bench3.prepare_dataset --dataset <name>"
        )
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {sorted(DESIGNS)}")
    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    try:
        if design == "wide":
            return consecutive_design(adata, block=holdout_block)
        return DESIGNS[design](adata)
    finally:
        adata.file.close()


def describe(configs):
    lines = []
    for c in configs:
        held = ", ".join(c["holdout_sections"])
        keep = ", ".join(c["remaining_sections"])
        # order by depth, not by label: section_10 sorts before section_2 as text
        zs = ", ".join(f"{s}@z={z:.1f}"
                       for s, z in sorted(c["holdout_z"].items(), key=lambda kv: kv[1]))
        lines.append(f"{c['holdout_id']:>14s}  held out: [{held}]\n"
                     f"{'':>14s}  input   : [{keep}]\n"
                     f"{'':>14s}  targets : {zs}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Emit v3 holdout designs")
    ap.add_argument("--input", default=None,
                    help="dataset NAME or a path to a built data.h5ad")
    ap.add_argument("--design", default="paper", choices=sorted(DESIGNS))
    ap.add_argument("--holdout-block", type=int, default=None,
                    help="consecutive sections to hold out for --design wide "
                         f"(default {DEFAULT_WIDE_BLOCK}; clamped to n-2 so the "
                         f"first and last sections stay as input)")
    ap.add_argument("--output", help="write JSON here (default: print)")
    args = ap.parse_args()

    configs = build_designs(resolve_dataset_arg(args.input), design=args.design,
                            holdout_block=args.holdout_block)
    print(describe(configs))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(configs, indent=2))
        print(f"\nWrote {len(configs)} config(s) to {args.output}")


if __name__ == "__main__":
    main()
