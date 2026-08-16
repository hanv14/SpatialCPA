#!/usr/bin/env python3
"""Build `resources/gene_meta.parquet` once, online, from a panel's gene symbols.

The only sanctioned way this project touches the network. Run it by hand, on a machine
with outbound access, before training; training and tests read the cached table and never
go online (T02 "Do NOT").

    python scripts/build_gene_meta.py --species mouse --symbols-from panel.h5ad
    python scripts/build_gene_meta.py --species human Gad1 Slc17a7 Pvalb
    python scripts/build_gene_meta.py --offline Gad1        # symbol-only rows, no network
    python scripts/build_gene_meta.py --species mouse --merge extra_panel.txt   # extend a table

**One organism per table.** `--species` takes exactly one species; a table is keyed by symbol, so a
two-species request lets the same symbol resolve to two different genes. The table records both the
requested species and the resolved taxid in every row, the writer **replaces** by default, and the
printed counts are properties of the file on disk.

Symbols that mygene.info does not resolve degrade to symbol-only rows and are reported.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.text import (
    build_gene_meta,
    gene_descriptor,
    gene_meta_summary,
    load_gene_meta,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", help="gene symbols to look up")
    parser.add_argument(
        "--symbols-from",
        type=Path,
        default=None,
        help="an .h5ad whose var_names are the symbols, or a text file with one symbol per line",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"output table (default: Config.gene_meta_path = {Config().gene_meta_path})",
    )
    parser.add_argument(
        "--species",
        default=Config().mygene_species,
        help="mygene.info species filter",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not query mygene.info; write symbol-only rows",
    )
    parser.add_argument(
        "--dump-raw",
        metavar="SYMBOL",
        default=None,
        help=(
            "print mygene.info's RAW response for one symbol and exit, without writing anything. "
            "Use it to see the actual shape of the `ensembl` field rather than reasoning about it: "
            "`--species mouse --dump-raw 1700057H15Rik`. Needs network access"
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "keep rows already in the table and reuse cached rows for the requested symbols. "
            "OFF by default: merging is what made a corrected --species re-run a no-op, because "
            "every symbol was already cached and no query was issued"
        ),
    )
    return parser.parse_args(argv)


def read_symbols(args: argparse.Namespace) -> list[str]:
    symbols = list(args.symbols)
    source = args.symbols_from
    if source is not None:
        if source.suffix == ".h5ad":
            import anndata as ad

            symbols += [str(name) for name in ad.read_h5ad(source).var_names]
        else:
            # Blank lines and `#` comments are skipped: a committed symbol list wants a header
            # saying where the panel came from, and without this the header lines are looked up as
            # gene symbols. Found at T06 while writing resources/starmap_panel_symbols.txt.
            symbols += [
                stripped
                for line in source.read_text().splitlines()
                if (stripped := line.strip()) and not stripped.startswith("#")
            ]
    return list(dict.fromkeys(symbols))


def dump_raw(symbol: str, species: str) -> int:
    """Print mygene.info's raw response for one symbol, exactly as the client returns it.

    Deliberately does not write, filter or select anything: the point is to see the structure the
    selection logic is choosing from. Written because a build whose ``taxid`` filter demonstrably
    worked still emitted four other mammals' Ensembl ids, and the shape of the ``ensembl`` field
    that made that possible could only be guessed at from outside.
    """
    from spatialcpav25_gen.data.text import load_mygene_client, resolve_species

    name, taxid = resolve_species(species)
    client = load_mygene_client(Config().replace(mygene_species=name))
    hits = client.querymany(
        [symbol],
        scopes="symbol,alias",
        fields="symbol,name,summary,alias,ensembl.gene,ensembl,taxid,entrezgene,_score",
        species=name,
        verbose=False,
    )
    print(f"# mygene.info raw response for {symbol!r}, species={name} (taxid {taxid})")
    print(f"# {len(hits)} hit(s)\n")
    print(json.dumps(hits, indent=2, sort_keys=True, default=str))
    print(f"\n# taxids present: {sorted({h.get('taxid') for h in hits if isinstance(h, dict)})}")
    for index, hit in enumerate(hits):
        if not isinstance(hit, dict):
            continue
        print(
            f"# hit {index}: taxid={hit.get('taxid')} symbol={hit.get('symbol')!r} "
            f"score={hit.get('_score')} summary={'yes' if hit.get('summary') else 'no'} "
            f"ensembl={hit.get('ensembl')!r}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dump_raw:
        return dump_raw(args.dump_raw, args.species)
    symbols = read_symbols(args)
    if not symbols:
        print("no symbols given; pass some, or --symbols-from", file=sys.stderr)
        return 2

    cfg = Config().replace(
        text_allow_network=not args.offline,
        mygene_species=args.species,
        **({} if args.out is None else {"gene_meta_path": str(args.out)}),
    )
    build_gene_meta(symbols, cfg, merge=args.merge)

    # Report the file that now exists, not the request. The previous version printed how many
    # *requested* symbols carried a full name ("1122/1122") while the file held 1138 rows of
    # mixed-species data, so the number could not detect the thing that was wrong.
    summary = gene_meta_summary(cfg.gene_meta_path)
    print(f"wrote {summary['path']}")
    print(f"  rows on disk        {summary['rows']}   (requested {len(symbols)})")
    print(f"  species requested   {summary['species_requested']}")
    print(f"  species resolved    {summary['species_resolved']} (taxid)")
    print(f"  with full_name      {summary['with_full_name']}/{summary['rows']}")
    print(f"  with summary        {summary['with_summary']}/{summary['rows']}")
    print(f"  with ensembl_id     {summary['with_ensembl_id']}/{summary['rows']}")
    print(f"  ensembl prefixes    {summary['ensembl_prefixes']}")
    print(f"  expected prefix     {summary['expected_ensembl_prefix']}")
    if summary["rows"] != len(symbols) and not args.merge:
        print(
            f"  !! {summary['rows']} rows for {len(symbols)} requested symbols without --merge; "
            "that should not happen",
            file=sys.stderr,
        )
        return 1

    meta = load_gene_meta(cfg.gene_meta_path, species=None if args.offline else args.species)
    for symbol in symbols[:3]:
        print(f"  {gene_descriptor(symbol, meta.get(symbol))[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
