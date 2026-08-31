"""How much text does each side of the gene split actually carry?

A2 embeds an unseen gene as ``norm(W t)`` and nothing else, so what that gene's descriptor
*says* is the whole of its evidence. The split was stratified on mean expression and Moran's I
— the two axes that decide whether ``marker_depth_r`` can measure anything — and on nothing
about the metadata. If the draw happened to take a disproportionate share of the panel's
summary-less genes, the held-out side would be handicapped in a way that has nothing to do with
whether the text channel works, and A1/A2 would lose for the wrong reason.

Model-free, seconds, and **read before the results**: a coverage gap found afterwards is
indistinguishable from an excuse. It reports, for each side of the split, how many genes carry a
full name, a summary, an *orthologue's* summary rather than their own, and how long their
descriptors are — plus a two-proportion check on the summary rate, since that is the field that
carries almost all the text.

Usage::

    python scripts/t09_zeroshot_text_coverage.py --split reports/t09_gene_split_deep.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.text import gene_descriptor, load_gene_meta


def side_coverage(names: list[str], meta: dict[str, Any]) -> dict[str, Any]:
    """Descriptor statistics for one side of the split."""
    rows = [meta.get(name) for name in names]
    descriptors = [gene_descriptor(name, row) for name, row in zip(names, rows, strict=True)]
    with_summary = [r for r in rows if r is not None and r.summary]
    orthologue = [r for r in with_summary if r.summary_source == "ortholog"]
    lengths = [len(d) for d in descriptors]
    return {
        "n": len(names),
        "n_in_table": sum(1 for r in rows if r is not None),
        "n_full_name": sum(1 for r in rows if r is not None and r.full_name),
        "n_summary": len(with_summary),
        "summary_rate": len(with_summary) / max(len(names), 1),
        "n_orthologue_summary": len(orthologue),
        # Definitional, not a heuristic on punctuation: a descriptor identical to the one
        # the symbol alone produces carries no metadata at all.
        "n_bare_symbol_only": sum(
            1
            for name, d in zip(names, descriptors, strict=True)
            if d == gene_descriptor(name, None)
        ),
        "descriptor_chars_median": int(statistics.median(lengths)) if lengths else 0,
        "descriptor_chars_min": min(lengths, default=0),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split", required=True)
    ap.add_argument("--gene-meta", default=None, help="default: Config.gene_meta_path")
    ap.add_argument("--out", default=None, help="optional .json destination")
    args = ap.parse_args(argv)

    cfg = Config()
    split = json.loads(Path(args.split).read_text())
    meta = load_gene_meta(args.gene_meta or cfg.gene_meta_path, species=cfg.mygene_species)
    names = {"held_out": [], "kept": []}
    for gene in split["genes"]:
        names["held_out" if gene["held_out"] else "kept"].append(str(gene["name"]))

    report = {side: side_coverage(members, meta) for side, members in names.items()}
    held, kept = report["held_out"], report["kept"]
    report["summary_rate_gap"] = held["summary_rate"] - kept["summary_rate"]

    print(f"gene metadata coverage for {args.split}")
    for side in ("held_out", "kept"):
        row = report[side]
        print(
            f"  {side:9s} n={row['n']:4d}  in table {row['n_in_table']:4d}  full name "
            f"{row['n_full_name']:4d}  summary {row['n_summary']:4d} "
            f"({row['summary_rate']:.1%}, {row['n_orthologue_summary']} orthologue)  "
            f"symbol-only {row['n_bare_symbol_only']:3d}  descriptor chars median "
            f"{row['descriptor_chars_median']}, min {row['descriptor_chars_min']}"
        )
    gap = report["summary_rate_gap"]
    print(
        f"\n  summary-rate gap held_out - kept = {gap:+.1%}. The text arms' evidence is the "
        "descriptor, so a large negative gap would handicap A1/A2 for a reason unrelated to "
        "whether the text channel works."
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
