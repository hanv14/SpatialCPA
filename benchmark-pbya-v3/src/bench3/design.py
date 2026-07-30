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

A design is a JSON-serializable dict with the same keys v2's ``run_benchmark``
expects, so the leakage machinery is reused verbatim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import DATASET_PATH, HELD_OUT_SECTIONS


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
        "holdout_id": "paper_2_4_6",
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


DESIGNS = {"paper": paper_design, "loo": loo_design}


def build_designs(h5ad_path=DATASET_PATH, design="paper"):
    """Load the dataset and return the holdout configs for ``design``."""
    import anndata as ad

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"dataset not found: {h5ad_path}\n"
            f"Build it first: python -m src.bench3.prepare_starmap"
        )
    if design not in DESIGNS:
        raise ValueError(f"unknown design {design!r}; choose from {sorted(DESIGNS)}")
    adata = ad.read_h5ad(str(h5ad_path), backed="r")
    try:
        return DESIGNS[design](adata)
    finally:
        adata.file.close()


def describe(configs):
    lines = []
    for c in configs:
        held = ", ".join(c["holdout_sections"])
        keep = ", ".join(c["remaining_sections"])
        zs = ", ".join(f"{s}@z={z:.1f}" for s, z in sorted(c["holdout_z"].items()))
        lines.append(f"{c['holdout_id']:>14s}  held out: [{held}]\n"
                     f"{'':>14s}  input   : [{keep}]\n"
                     f"{'':>14s}  targets : {zs}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Emit v3 holdout designs")
    ap.add_argument("--input", default=str(DATASET_PATH))
    ap.add_argument("--design", default="paper", choices=sorted(DESIGNS))
    ap.add_argument("--output", help="write JSON here (default: print)")
    args = ap.parse_args()

    configs = build_designs(args.input, design=args.design)
    print(describe(configs))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(configs, indent=2))
        print(f"\nWrote {len(configs)} config(s) to {args.output}")


if __name__ == "__main__":
    main()
