# resources/

## `gene_meta.human_orthologs.parquet` — **not** the STARmap panel's table. Do not point `Config.gene_meta_path` at it.

Committed as `resources/gene_meta.parquet` in `b68712d` ("C14: gene metadata for the STARmap panel"),
moved here at the T02 repair (SPEC_QUESTIONS **B19**) because an audit of its contents says it is
**human**, and the STARmap Wang2018 panel in `data/starmap/` is **mouse**:

| | |
|---|---|
| rows | 28 (the right symbols, mouse-cased: `Slc17a7`, `Gad1`, …) |
| Ensembl-id prefixes | **`{'ENSG': 28}`** — every row human. A mouse table's are `ENSMUSG` |
| full names / summaries / Ensembl ids | 28 / 28 / 28 |

It is the same defect the repair was about, in its most deceptive form: `Config.mygene_species` was
`"human,mouse"` when this was built, mygene matched the mouse-cased symbols case-insensitively, and the
human hit outranks the mouse one — so a mouse panel silently got human annotations. **Coverage looks
perfect precisely because it is wrong**: human gene records are the best-annotated in NCBI, so an
accidental human resolution maximises summary coverage. 28/28 summaries is not evidence of a good
table, which is why correctness is now checked on `species_resolved` and the Ensembl-prefix histogram
rather than on coverage.

Kept rather than deleted: the bytes are real human orthologs and are the right table for a *human*
dataset, which T10 needs one of. Nothing points at it. Rebuild it with the species recorded before
using it:

```
pip install -e ".[extra]"                                   # mygene is in the `extra` group
python scripts/build_gene_meta.py --species human \
    --symbols-from resources/starmap_panel_symbols.txt \
    --out resources/gene_meta.human.parquet
```

**`Config.gene_meta_path` (`resources/gene_meta.parquet`) is deliberately absent**, so
`load_gene_meta` raises `FileNotFoundError` telling you to build it, rather than loading a table of
the wrong organism. **C14 is still open**: the mouse table has not been built. On a machine with
outbound access to mygene.info:

```
python scripts/build_gene_meta.py --species mouse \
    --symbols-from resources/starmap_panel_symbols.txt
```

and report the `with summary N/rows` line it prints — that is the mouse-only summary coverage nobody
has measured yet.

## `starmap_panel_symbols.txt`

The 28 real gene symbols of `data/starmap/STARmap_Wang2018three_data_3D_data.h5ad`, committed so that
building the table needs no guessing about which symbols to look up. Header lines are comments and are
skipped by `scripts/build_gene_meta.py`.
