#!/usr/bin/env python3
"""Build `resources/gene_meta.parquet` once, online, from a panel's gene symbols.

The only sanctioned way this project touches the network. Run it by hand, on a machine
with outbound access, before training; training and tests read the cached table and never
go online (T02 "Do NOT").

    python scripts/build_gene_meta.py --symbols-from panel.h5ad
    python scripts/build_gene_meta.py Gad1 Slc17a7 Pvalb
    python scripts/build_gene_meta.py --offline Gad1        # symbol-only rows, no network

Symbols that mygene.info does not resolve degrade to symbol-only rows and are reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.text import build_gene_meta, gene_descriptor, load_gene_meta


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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    symbols = read_symbols(args)
    if not symbols:
        print("no symbols given; pass some, or --symbols-from", file=sys.stderr)
        return 2

    cfg = Config().replace(
        text_allow_network=not args.offline,
        mygene_species=args.species,
        **({} if args.out is None else {"gene_meta_path": str(args.out)}),
    )
    build_gene_meta(symbols, cfg)

    meta = load_gene_meta(cfg.gene_meta_path)
    resolved = [s for s in symbols if s in meta and meta[s].full_name is not None]
    print(f"wrote {cfg.gene_meta_path}: {len(resolved)}/{len(symbols)} symbols carry metadata")
    for symbol in symbols[:3]:
        print(f"  {gene_descriptor(symbol, meta.get(symbol))[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
