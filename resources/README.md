# resources/

## `gene_meta.parquet` — the live table. Mouse, 1138 symbols, species-correct.

Built on a networked machine at `6f3cdfa` from `mouse_panels_symbols.txt`, and the first committed
table that `load_gene_meta(..., species="mouse")` accepts:

| | |
|---|---|
| rows | 1138 |
| `species_requested` / `species_resolved` | `mouse` / `10090`, uniform |
| Ensembl-id prefixes | `{'ENSMUSG': 1137, 'None': 1}` — the id defect of B19a is gone |
| `full_name` | 1138/1138 |
| summaries, **split by source** | **native 148, ortholog 0, none 990** |

The split is the number to quote, not `with_summary` (SPEC_QUESTIONS **B20**). `ortholog 0` is not a
measurement of the fallback: this table was built **before** `Config.gene_summary_fallback` existed,
so its 990 bare rows have never been offered an orthologue. `_read_gene_meta_table` back-fills the
three `summary_source*` columns for tables of this vintage as `native` — the only thing they can be —
which is why it still loads.

**One run gets the fallback's numbers**, and it needs mygene.info (403 from this container, C14):

```
pip install -e ".[extra]"
python scripts/build_gene_meta.py --species mouse --symbols-from resources/mouse_panels_symbols.txt
```

The report prints `native / ortholog / none` directly. 148 native is confirmed genuine mouse RefSeq
sparsity, not a query defect (B20).

## `gene_meta.human_orthologs.parquet` — **not** the STARmap panel's table. Do not point `Config.gene_meta_path` at it.

**The bytes under this name changed after the name was chosen, and the name is now wrong about
them.** `1c515f3` overwrote the 28-row human table this section was written about (17 kB → 110 kB)
with the **1138-symbol mixed-species** build — `{'ENSMUSG': 389, 'ENSMSIG': 324, 'ENSNVIG': 234,
'ENSMPUG': 111, 'ENSFALG': 73, 'FBgn': 2, None: 5}`, 144/1138 summaries: the file the B19 report
described, four other mammals and a fly. It has no species columns, so `load_gene_meta` refuses it
with "predates the species columns", which is the right answer for both contents.

The 28-row all-human table described below is recoverable, and only, from git:
`git show b68712d:resources/gene_meta.parquet` (28 rows, `{'ENSG': 28}`, 28/28 summaries).

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

**`Config.gene_meta_path` now exists** (see the top of this file) and is species-correct, so the
"deliberately absent" state this section used to describe is over. What is still outstanding is the
STARmap panel's own 28-symbol table, which is a different panel from the 1138-symbol one:

```
python scripts/build_gene_meta.py --species mouse \
    --symbols-from resources/starmap_panel_symbols.txt \
    --out resources/gene_meta.starmap.parquet
```

## `gene_meta.mouse_prefix_bug.parquet` — the mouse build, correct except for `ensembl_id`

Committed as `resources/gene_meta.parquet` in `688820c`; moved here because
`load_gene_meta(..., species="mouse")` refuses it, correctly:

```
ensembl_id values are not mouse ids (expected the 'ENSMUSG' prefix):
{'ENSMSIG': 321, 'ENSNVIG': 241, 'ENSMPUG': 111, 'ENSFALG': 73, 'FBgn': 1} by prefix
```

It is the output of the **first** repair (SPEC_QUESTIONS B19) and it is right about everything the
taxid filter governs — `species_requested` mouse ×1138, `species_resolved` 10090 ×1138, a mouse
record found for all 1138 symbols — and wrong about `ensembl_id`, which came from a non-mouse element
of the mouse hit's own `ensembl` list (**B19a**). Kept as the evidence for that finding. Rebuild with
the current code and the ids will be selected by prefix:

```
python scripts/build_gene_meta.py --species mouse --symbols-from resources/mouse_panels_symbols.txt
```

**It is also the evidence that mouse summary sparsity is real and separate** (B20). Summary presence
is essentially flat across the wrong-prefix groups — ENSMUSG 11.5%, ENSMSIG 10.9%, ENSNVIG 17.0%,
ENSMPUG 17.1%, ENSFALG 11.0% — so whatever the prefix defect was doing, it was not what cost the
summaries; and `full_name` is present for 1138/1138, so a mouse record *was* found and read for every
symbol. 148/1138 is a property of those mouse records.

## `starmap_panel_symbols.txt`

The 28 real gene symbols of `data/starmap/STARmap_Wang2018three_data_3D_data.h5ad`, committed so that
building the table needs no guessing about which symbols to look up. Header lines are comments and are
skipped by `scripts/build_gene_meta.py`.
