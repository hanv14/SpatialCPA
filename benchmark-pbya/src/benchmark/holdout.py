"""Generate holdout configurations for leave-one-out and leave-k-out evaluation."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np


def get_sorted_sections(adata):
    """Return section labels sorted by median z-coordinate."""
    sections = adata.obs["section"].unique().tolist()
    z_by_section = {}
    for sec in sections:
        mask = adata.obs["section"] == sec
        z_by_section[sec] = float(np.median(adata.obsm["spatial"][mask, 2]))
    return sorted(sections, key=lambda s: z_by_section[s]), z_by_section


def leave_one_out(adata, exclude_boundary=True):
    """Generate LOO holdout configs — one per interior section.

    Parameters
    ----------
    adata : AnnData
        Must have obs['section'] and obsm['spatial'] with z in column 2.
    exclude_boundary : bool
        If True, skip the first and last section (cannot interpolate beyond edges).

    Yields
    ------
    dict with keys: holdout_id, holdout_sections, remaining_sections, holdout_z
    """
    sections, z_by_section = get_sorted_sections(adata)
    start = 1 if exclude_boundary else 0
    end = len(sections) - 1 if exclude_boundary else len(sections)
    for i in range(start, end):
        sec = sections[i]
        remaining = [s for s in sections if s != sec]
        yield {
            "holdout_id": f"loo_{sec}",
            "holdout_sections": [sec],
            "remaining_sections": remaining,
            "holdout_z": {sec: z_by_section[sec]},
        }


def leave_k_out(adata, k=2, stride=1, exclude_boundary=True):
    """Generate leave-k-out holdout configs — consecutive blocks of k sections.

    Parameters
    ----------
    adata : AnnData
    k : int
        Number of consecutive sections to hold out.
    stride : int
        Step between starting positions of holdout blocks.
    exclude_boundary : bool
        If True, require at least one section on each side of the block.

    Yields
    ------
    dict with keys: holdout_id, holdout_sections, remaining_sections, holdout_z
    """
    sections, z_by_section = get_sorted_sections(adata)
    n = len(sections)
    start = 1 if exclude_boundary else 0
    end = n - k - (1 if exclude_boundary else 0) + 1
    for i in range(start, end, stride):
        block = sections[i : i + k]
        remaining = [s for s in sections if s not in block]
        yield {
            "holdout_id": f"lko{k}_{'_'.join(block)}",
            "holdout_sections": block,
            "remaining_sections": remaining,
            "holdout_z": {s: z_by_section[s] for s in block},
        }


def subset_sections(adata, section_names):
    """Generate a single holdout config from a specific section subset.

    Given a list of section names (sorted by z), holds out all interior sections
    and keeps the first and last as training anchors.

    Parameters
    ----------
    adata : AnnData
    section_names : list[str]
        Ordered section names. First and last are kept; middle are held out.

    Yields
    ------
    dict with holdout config
    """
    _, z_by_section = get_sorted_sections(adata)
    holdout = section_names[1:-1]
    remaining = [section_names[0], section_names[-1]]
    yield {
        "holdout_id": f"subset_{'_'.join(holdout)}",
        "holdout_sections": holdout,
        "remaining_sections": remaining,
        "holdout_z": {s: z_by_section[s] for s in holdout},
    }


def alternating_holdout(adata, exclude_boundary=True):
    """Hold out every other section (for isoST-style alternating evaluation).

    Yields a single holdout config where odd-indexed interior sections are held out.

    Yields
    ------
    dict with holdout config
    """
    sections, z_by_section = get_sorted_sections(adata)
    holdout = []
    remaining = []
    for i, sec in enumerate(sections):
        if exclude_boundary and (i == 0 or i == len(sections) - 1):
            remaining.append(sec)
        elif i % 2 == 1:  # odd-indexed sections held out
            holdout.append(sec)
        else:
            remaining.append(sec)
    if holdout:
        yield {
            "holdout_id": f"alternating_{len(holdout)}sec",
            "holdout_sections": holdout,
            "remaining_sections": remaining,
            "holdout_z": {s: z_by_section[s] for s in holdout},
        }


def generate_holdouts(h5ad_path, strategy="leave_one_out", k=2, stride=1,
                      exclude_boundary=True):
    """Load dataset and generate holdout configs.

    Returns list of holdout dicts.
    """
    adata = ad.read_h5ad(h5ad_path, backed="r")
    if strategy == "leave_one_out":
        configs = list(leave_one_out(adata, exclude_boundary=exclude_boundary))
    elif strategy == "leave_k_out":
        configs = list(leave_k_out(adata, k=k, stride=stride,
                                   exclude_boundary=exclude_boundary))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    adata.file.close()
    return configs


def main():
    parser = argparse.ArgumentParser(description="Generate holdout configurations")
    parser.add_argument("--input", required=True, help="Path to processed data.h5ad")
    parser.add_argument("--strategy", default="leave_one_out",
                        choices=["leave_one_out", "leave_k_out"])
    parser.add_argument("--k", type=int, default=2, help="Sections to hold out (leave_k_out)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--include-boundary", action="store_true")
    parser.add_argument("--output", help="Write JSON to file (default: stdout)")
    args = parser.parse_args()

    configs = generate_holdouts(
        args.input,
        strategy=args.strategy,
        k=args.k,
        stride=args.stride,
        exclude_boundary=not args.include_boundary,
    )

    out = json.dumps(configs, indent=2)
    if args.output:
        Path(args.output).write_text(out)
        print(f"Wrote {len(configs)} holdout configs to {args.output}")
    else:
        print(out)


if __name__ == "__main__":
    main()
