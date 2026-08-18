# T02 — Text-grounded embeddings

Part of [PROGRESS.md](../PROGRESS.md).

### T02 — Text-grounded gene and context embeddings (2026-08-15)

**Built.** `spatialcpav25_gen/data/text.py` — `GeneMeta`, `gene_descriptor`,
`celltype_descriptor`, `region_descriptor`, `descriptor_key`, `build_gene_meta`, `load_gene_meta`,
`TextEncoder` (disk cache keyed by `sha256(text_model + descriptor)`, one `.npy` per descriptor,
atomic writes, lazy model load), `TransformerBackend` + the `load_transformer_backend` seam,
`GeneMetaUnavailableWarning`. `spatialcpav25_gen/model/embeddings.py` — `TextGroundedEmbedding`
(`W`, zeros-init `r`, `gamma` buffer, `distill` MLP, `set_progress`, `forward`,
`forward_zero_shot`, `distillation_loss`), `EntityEmbeddings` (the three instances: genes at
`gene_emb_dim`, cell types and regions at `ctx_emb_dim`), `text_embedding_diagnostics` (ablation
A3). `scripts/build_gene_meta.py` (the one-off online build). `tests/test_text.py` (20 tests),
`tests/fixtures/text.py` (a deterministic hash backend — MedCPT cannot be downloaded in the test
environment). 12 new `Config` fields, no constants outside it.

**Test numbers.** `make check` green: ruff clean, `mypy --strict` clean on 8 source files,
**45 tests pass in 9.1 s** on CPU (20 of them T02's; budget 3 min). Nothing marked `slow`.

| Acceptance test | Required | Measured |
|---|---|---|
| `test_descriptor_stability` | alias order irrelevant | exact string equality across 3 alias orderings, incl. duplicates and self-alias |
| `test_cache_hit_avoids_model_load` | second call from cache | `n_model_calls == 0`, vectors bitwise equal; backend constructor asserts if touched |
| `test_missing_gene_meta_degrades` | `"{symbol}."` and encodes | `"Xkr4."`, `(1, 768)` float32, ‖v‖ = 1 |
| `test_zero_shot_shapes` | `(10, out_dim)` | `(10, 128)` both distilled and pure-text arms |
| `test_gamma_anneal` | γ = 0 → output *is* `LayerNorm(W t)`; γ = 1 at end | `torch.equal` (bitwise) with a **non-zero** residual planted, so the gate is what is tested; γ(0.15) = 0.5 |
| `test_distillation_reduces_error` | MSE drops ≥ 50 % in 200 steps | **1.0155 → 1.29e-07 (ratio 1.3e-07)** in 0.4 s |
| `test_offline` | symbol-only rows, no raise | 3/3 rows symbol-only, one `GeneMetaUnavailableWarning`, table round-trips through `load_gene_meta` |

Diagnostics on the synthetic fixture (200 genes × 13 500 cells, descriptors = bare `GeneNNNN.`,
deterministic hash backend standing in for MedCPT):

| Quantity | Value | Reading |
|---|---|---|
| text/co-expression Spearman | **+0.0055** over 19 900 gene pairs | ≈ 0, exactly as the spec predicts for arbitrary gene names |
| kNN purity (k = 10) | **0.2315** vs chance **0.2259** (5 Leiden modules, sizes 57/53/40/33/17) | no module signal, same reading |
| `residual_norm_ratio` | 0.0 at init (zeros residual); 47.7 with a planted N(0, 1) residual | the ratio responds |

That the diagnostic can *detect* signal is tested separately (`test_diagnostics_sees_planted_signal`,
4 planted co-expression modules with text vectors aligned to them): Spearman **0.506**, kNN purity
**0.700**, which is the maximum reachable with 8-gene modules and k = 10, and 4 modules recovered.
So the ≈ 0 on the fixture is a fact about the fixture's gene names, not a broken statistic. **The
number that matters is the real-data one, and it cannot be computed until `resources/gene_meta.parquet`
exists** (SPEC_QUESTIONS C14).

**Deviations from the spec, and why.**

1. **Pooling is CLS, not mean** (SPEC_QUESTIONS C3, settled in T01 as `Config.text_pooling="cls"`).
   The spec says "mean-pool the last hidden state" and then says to use whatever the checkpoint
   specifies; MedCPT-Query-Encoder is trained contrastively on the first-token state. The choice is
   a `Config` field with the justification in a comment at the pooling site, as the spec asks.
2. `build_gene_meta(symbols, cfg=None)` takes the config as an additive keyword — the spec's
   `build_gene_meta(symbols)` has nowhere to read `gene_meta_path`, `text_allow_network` or
   `mygene_species` from. Same shape of deviation as T01's `split_holdout(..., cfg=None)`.
3. `text_embedding_diagnostics(emb, expr, *, seed)` gains a required keyword-only seed
   (SPEC_QUESTIONS C13): the Leiden partition and the gene-pair subsample are stochastic, and
   Convention 3 forbids leaving that to a global RNG. Two calls with the same seed return an equal
   dict, asserted.
4. **Network is opt-in, not "if available".** The spec says to assemble from mygene.info "if network
   is available"; probing availability from inside a test run is exactly what the "Do NOT" section
   forbids. `Config.text_allow_network` defaults to `False`, so the default path is offline and the
   online path is an explicit one-off (`scripts/build_gene_meta.py`). Degradation warns
   (`GeneMetaUnavailableWarning`) rather than being silent (Convention 6).
5. **Descriptor grammar pinned beyond the spec's format string**, because it is the cache key: every
   field is stripped of surrounding whitespace and trailing periods before joining, so a summary
   that already ends in "." does not produce ".." and a different hash. Aliases are sorted,
   de-duplicated, and an alias equal to the symbol is dropped. A region's ancestor path is *not*
   sorted — its order is the hierarchy.
6. `distillation_loss` is a **mean** over entities and components, not a sum (SPEC_QUESTIONS C15), so
   `w_distill` means the same thing on a 200-gene and a 20 000-gene panel.
7. `TextGroundedEmbedding.__init__` initialises `W` and `distill` from a generator seeded with
   `cfg.seed` instead of the global torch RNG (Convention 3); the distribution is the one
   `nn.Linear` would have used. `test_construction_is_deterministic` asserts bitwise equality across
   two constructions and inequality across seeds.
8. `test_distillation_reduces_error` is **not** marked `slow`, against the proposal in
   SPEC_QUESTIONS C6: at this size the 200-step loop is 0.4 s, and the 3-minute budget is nowhere
   near threatened. C6's rule still stands for the loops that actually cost something (T06–T08).
9. `tests/test_schema.py` was reformatted by `ruff format` (pinned 0.4.4). The version in
   `main` used the newer parenthesised-assert style, which the pinned formatter rejects, so
   `make lint` was already failing before this task. Formatting only, no behaviour change.

**Flagged for later.**

* **`resources/gene_meta.parquet` does not exist and cannot be built here** (SPEC_QUESTIONS C14).
  Every descriptor is currently the bare symbol. Someone has to run
  `python scripts/build_gene_meta.py --symbols-from <panel>.h5ad` on a networked machine and commit
  the table before the first real run, or the paper's text channel is symbols only and the A3
  diagnostic has nothing to report.
* Cell-type ontology records and region hierarchies are *accepted* (`celltype_descriptor`,
  `region_descriptor`) but nothing resolves them yet: T04/T06 pass what the dataset carries, and a
  dataset without an ontology gets the raw label, which is what the spec asks for.
* The zero-shot table needs both arms (design §2.2 / §7 E1): `forward_zero_shot(use_distill=False)`
  is the pure-text arm and `use_distill=True` the distilled one. Both exist and are shape-tested
  here; T10 E1 reports them.
* Ablation A3 (`Config.text_emb_mode="lookup"`) is mapped to T10 in the coverage matrix and is not
  wired here; `TextGroundedEmbedding` is unconditionally text-grounded.

### T02 (repaired at T06) — `build_gene_meta` returned four species' genes for `--species mouse` (2026-08-16)

Reported from a networked run: a **1138-symbol mouse panel** written with ENSMUSG 389, ENSMSIG 324
(ground squirrel), ENSNVIG 234 (mink), ENSMPUG 111 (ferret), ENSFALG 73 (falcon), FBgn 2 (fruit fly),
5 with no id, and summaries down to 144/1138. Nothing raised.

**The species parameter was being sent.** I installed the pinned `mygene` client and read its source:
`querymany` puts `species` straight into the POST body, so `species="mouse"` did reach the endpoint.
The mixing came from four defects *around* the query — three of them certain from our own code, and
the fourth is why a wrong hit could win even when the right one was present. Full write-up in
SPEC_QUESTIONS **B19**; in short:

1. **the cache short-circuited every re-run** (`missing = [s for s in symbols if s not in cached]`), so
   a corrected `--species` run issued **no query at all** — the argument could not filter because
   nothing was asked. This is most of "the species argument is not filtering";
2. **the writer merged and never replaced**, leaving 16 stale rows in a 1138-row file for a
   1122-symbol request;
3. **the printed count was not a property of the file** — "1122/1122" counted requested symbols with a
   full name, so it could not detect a wrong file;
4. **among several hits per symbol the first won**, with no species check, no exact-match preference
   and no `_score`; `taxid` was not even in `fields`, and there was no species column, so the result
   could not be audited afterwards.

**And the network path had no test at all**, which is how this survived T02. It has one now: a
`MyGeneClient` protocol and a `load_mygene_client` seam (the shape T02 already used for the
transformer), so a fake client reproduces the reported response offline.

**All six fixes, replayed against the reported response** (a fake client returning exactly that
mix, non-mouse hits first):

| | before | after |
|---|---|---|
| rows on disk for 1138 requested | 1138, mixed | **1138** |
| Ensembl prefixes | ENSMUSG 389 + 744 others + 5 none | **`{'ENSMUSG': 1133, 'None': 5}`** |
| resolved species in the table | not recorded | **`['10090']`** |
| wrong-species hits | written | **749 dropped, warned** |
| re-run with 1122 symbols | 1138 rows, 0 queries | **1122 rows, query issued** |
| `load_gene_meta(..., species="human")` | loaded happily | **raises** |

Also: `Config.mygene_species` default `"human,mouse"` → **`"mouse"`**, because a symbol-keyed table
describes one organism and a two-species request lets whichever hit arrived first win;
`resolve_species` refuses anything else. `specs/10` now requires E1 to pass the species to
`load_gene_meta` and to quote the table's own coverage. One new fast test file section, 7 tests,
`make check` green.

**The table that arrived mid-repair is the same bug again.** `b68712d` ("C14: gene metadata for the
STARmap panel") landed on the branch while I was working. Audited with the tooling this repair added:
28 rows, the right mouse-cased symbols, **Ensembl prefixes `{'ENSG': 28}` — every row human**, and
28/28 full names, summaries and Ensembl ids. It is the two-species coin toss: `mygene_species` was
`"human,mouse"`, the symbols matched case-insensitively, and the human hit outranks the mouse one. Note
what makes it dangerous — **its coverage is perfect *because* it is wrong.** Human gene records are
the best-annotated in NCBI, so resolving to human by accident maximises summary coverage; a reviewer
checking "28/28 have summaries" would have signed it off. Correctness is now checked on
`species_resolved` and the Ensembl-prefix histogram, never on coverage.

Handled without deleting anyone's work: moved to `resources/gene_meta.human_orthologs.parquet` (real
human orthologs are the right table for the *human* dataset T10 needs), `Config.gene_meta_path` left
absent so `load_gene_meta` raises "build it" rather than loading the wrong organism, the audit written
into `resources/README.md`, and `test_committed_gene_meta_tables_are_species_checkable` pinning it. The
schema check also got its own message for this exact case: a table predating the species columns says
"rebuild it, do not add the columns by hand, because the value that would go in them is the thing that
is unknown — check the Ensembl prefixes first".

**C14 is therefore still open, and the number I was asked for still cannot be produced here: the
mouse-only summary coverage.** mygene.info is
403'd in this container (C14), so the only summaries I can count are my fake's. What the old 144/1138
*was* is now explained — summaries were lost on precisely the 744 rows that resolved to non-reference
species, whose gene records carry no NCBI summary — and the corrected query keeps the mouse hit, so
coverage should rise to whatever share of these 1138 mouse genes have an NCBI summary.
`scripts/build_gene_meta.py` prints it as `with summary N/rows`, computed from the file on disk; it
needs one run on a networked machine:

```
pip install -e ".[extra]"
python scripts/build_gene_meta.py --species mouse --symbols-from <panel>.txt
```

### T02 (repaired, second pass) — `ensembl_id` came from the wrong element of the right hit (2026-08-16)

**The report was not self-contradictory, and finding out why exposed a second defect of mine.**
`species_resolved` uniformly `['10090']` alongside prefixes ENSMUSG 390 / ENSMSIG 321 / ENSNVIG 241 /
ENSMPUG 111 / ENSFALG 73 / FBgn 1 is consistent: every other field of those rows came from the mouse
hit, so the ids can only have come from a **non-mouse element of the mouse hit's own `ensembl` list**.
mygene's `ensembl` field carries cross-species mappings and `_ensembl_id` took `value[0]` of it. A
hit's `taxid` says nothing about which element of its `ensembl` field you then read.

The honesty defect: **`species_resolved` was written as `str(taxid)` — the *requested* value.** It
could never disagree with the argument, so "uniformly 10090" was true by construction and read as
evidence when it was not. It now comes from `best["taxid"]`.

**Fixes** (SPEC_QUESTIONS **B19a**), replayed on the reported response — 1138 rows, prefixes
`{'None': 748, 'ENSMUSG': 390}`, 747 wrong-prefix hits reported, table accepted by the species gate:

| fix | |
|---|---|
| `_ensembl_id(value, prefix)` | selects the id whose prefix is the requested species' (`SPECIES_ENSEMBL_PREFIX`, mouse `ENSMUSG`), wherever it sits in the list. No mouse id in the field → **`None`**, an absent id the descriptor never reads, rather than another organism's, which every join does |
| `_check_ensembl_prefix` | asserts the **stored** value at build *and* load, separately from the taxid, because `ensembl_id` is what everything downstream reads and must not inherit the hit's credibility |
| `species_resolved` | read from the hit, so the column is evidence rather than an echo of the request |
| `gene_meta_summary` | also reports `expected_ensembl_prefix`, so the histogram is self-checking |

**I could not print the raw response, and that is a container limit, not a choice.** mygene.info is
403'd here (C14) — verified again: `--dump-raw` reaches the network layer and dies on
`httpx.ProxyError: 403 Forbidden`. So `scripts/build_gene_meta.py --dump-raw SYMBOL` now exists to
print it unfiltered on a machine that can reach the API:

```
python scripts/build_gene_meta.py --species mouse --dump-raw 1700057H15Rik
```

It dumps every hit as JSON plus a per-hit line (`taxid`, `symbol`, `_score`, whether a summary is
present, the whole `ensembl` field). **This determines the fix's outcome on your data:** if each list
also contains the mouse id further down, those 748 become real ENSMUSG ids; if it does not, they stay
`None` and the mouse Ensembl ids simply are not in that response. My replay models the pessimistic
case; the raw dump settles it.

**Summary coverage 148/1138 — CONFIRMED genuinely sparse, from your own committed table, no network
needed** (SPEC_QUESTIONS **B20**). `688820c`'s `gene_meta.parquet` is auditable offline and it answers
the question three ways:

| finding | number |
|---|---|
| summary presence **flat across the wrong-prefix groups** | ENSMUSG **11.5%**, ENSMSIG 10.9%, ENSNVIG 17.0%, ENSMPUG 17.1%, ENSFALG 11.0% |
| `full_name` present | **1138/1138** — a mouse record was found and read for every symbol |
| by symbol class | conventional named genes **148/1105 (13.4%)**, clone/predicted **0/33** |

The first says the `ensembl_id` defect and the summary sparsity are **independent**: had the id defect
been costing summaries, the ENSMUSG rows would carry them at a higher rate, and they do not. The second
says the mouse record was reached every time, so the 990 absences belong to *those records*. The third
says what the sparsity looks like — the genes that do have summaries are the well-studied ones
(`Abcc9`, `Acta2`, `Adam12`, `Adcyap1`, `Adra1a`, …), which is the shape of NCBI mouse curation, and
13.4% is consistent with it. The panel is not the explanation either: only **33/1138 (2.9%)** are RIKEN
clones or predicted genes.

One narrow door remains — a *different mouse hit for the same symbol* carrying a summary the selected
one lacks — and it is now counted rather than argued about: `_query_mygene` warns with that count on
every build.

**So the fallback is the live question, and it is proposed, not implemented**, as asked — human orthologue summary, explicitly
labelled, with four constraints written into B20: a recorded `summary_taxid` provenance column rather
than a substituted value; the orthologue resolved through `homologene` with a required 1:1 mapping and
**never** by uppercasing the symbol (that is exactly the mistake that produced the all-human table);
`gene_descriptor` rendering the provenance in the text ("Human orthologue PVALB: …") so neither the
encoder nor a reader can mistake it; and T10's E1 reporting **both arms**, because importing human gene
descriptions into a mouse model's text channel changes what the open-vocabulary claim is about. Gated
by `Config.gene_summary_fallback`, default `"none"`, never overwriting a native summary.

**One number to weigh before you say yes.** Human coverage on this panel was ~93% against mouse's
13.4%, so with the fallback on roughly **85% of descriptors would carry human text** — the
open-vocabulary claim would then be substantially a claim about *human* gene summaries transferred to a
mouse model. That may be the right scientific call, but it is a different claim from the one the design
states, which is why "E1 reports both arms" is a constraint and not a nicety, and why coverage must
always be quoted split by `summary_taxid`.

**Both tables that arrived are now quarantined with the reason.** `688820c`'s
`gene_meta.parquet` → `resources/gene_meta.mouse_prefix_bug.parquet`: right about everything the taxid
filter governs, wrong about `ensembl_id`, refused at load by the new prefix assertion, and kept because
it *is* the evidence for B19a and B20. `Config.gene_meta_path` stays absent so `load_gene_meta` says
"build it" rather than loading something unusable. Rebuilding with the current code is one command and
fixes the ids.

Five new tests (33 in `tests/test_text.py`), `make check` green.

### 2026-08-16 — B20 fallback implemented: a summary-less mouse gene borrows its human orthologue's, labelled

Approved and built, default **on**. `Config.gene_summary_fallback="ortholog"` (with
`gene_summary_ortholog_species="human"`): where a mouse gene has no NCBI summary of its own, its
**1:1 HomoloGene orthologue's** summary is used, and it is labelled as borrowed in two places — the
descriptor text the frozen encoder reads (`"Slc17a7. Slc17a7 full name. Human orthologue SLC17A7:
Mediates glutamate uptake into synaptic vesicles."`) and three new table columns.

**The columns, and the one deviation from what B20 proposed.** B20 proposed a single `summary_taxid`;
built as **three** — `summary_source` (`native` / `ortholog` / null), `summary_source_taxid`,
`summary_source_symbol` — because the label needs the orthologue's *symbol* and a taxid cannot supply
it, and because a diagnostic filtering on `== "native"` reads better than one on `== "10090"`. B20
also proposed defaulting to `"none"` until T10 measured both arms; built as `"ortholog"`, because
E1's native-only arm is a **filter on `summary_source`**, not a second build, so one table serves
both arms and the default costs nothing.

**Mechanics, each of which is a way this could have gone wrong.** `homologene` is requested on the
primary query, so the fallback costs no extra round trip per gene. The **1:1 requirement is enforced
before** the orthologue query, so a 1:many gene is never fetched, let alone resolved by `_score` —
paralogous families are exactly where a borrowed summary is most plausible and most wrong. The
orthologue query (`scopes="entrezgene"`, `species="human"`) has its `taxid` **verified** like the
primary one, or a third species' summary would be labelled "Human orthologue" in the text. A native
summary is never overwritten. `_read_gene_meta_table` back-fills the three columns as `native` for
tables written before them — the one migration that *is* derivable, unlike the species columns, since
before this fallback there was nowhere else a summary could have come from.

**`resources/gene_meta.parquet` (`6f3cdfa`) audited: the species fix holds.** 1138 rows,
`species_requested` `mouse` / `species_resolved` `10090` uniform, `{'ENSMUSG': 1137, 'None': 1}` by
prefix, `full_name` 1138/1138, and `load_gene_meta(..., species="mouse")` **accepts** it. First
committed table that does. C14 downgraded from blocking: E1 can run on real gene text.

**The split, reported as asked — and what it is not.** The table on disk:

| source | count |
|---|---|
| native | **148** |
| ortholog | **0** |
| none | **990** |

`ortholog 0` is **not a measurement of the fallback**: that table was built before the fallback
existed, so its 990 bare rows have never been offered an orthologue. **The fallback's own
native/ortholog/none numbers cannot be produced in this container** — mygene.info answers 403 to
CONNECT through the agent proxy, re-verified this session (`recentRelayFailures` names
`mygene.info:443`), as do every other gene-annotation host. It is one run on a networked machine:

```
pip install -e ".[extra]"
python scripts/build_gene_meta.py --species mouse --symbols-from resources/mouse_panels_symbols.txt
#   with summary        N/1138
#     native            148   ortholog M   none K
```

`scripts/build_gene_meta.py` now prints that split on every build instead of a bare coverage number,
and `--native-summaries-only` runs the fallback off for comparison. **I have not put an expected
number in the docs.** The reported ~93% human rate makes a high value plausible, but the binding
quantity is 1:1 HomoloGene coverage over these 1138 symbols, and nothing offline predicts it; a
guessed figure in a methods section is worse than an absent one.

**`resources/README.md` corrected on a point it was wrong about.** It describes
`gene_meta.human_orthologs.parquet` as the 28-row all-human table; commit `1c515f3` overwrote those
bytes (17 kB → 110 kB) with the **1138-symbol mixed-species** build — `ENSMUSG 389, ENSMSIG 324,
ENSNVIG 234, ENSMPUG 111, ENSFALG 73, FBgn 2`, 144/1138 summaries. The README was labelling the wrong
exhibit. The 28-row table is recoverable only from `git show b68712d:resources/gene_meta.parquet`.

Six new tests (38 in `tests/test_text.py`) covering the four cases in one build (native kept, 1:1
borrowed and labelled, 1:many skipped, no-orthologue left bare), the fallback switched off making no
second query at all, a third species' orthologue hit being dropped, `_homologene_gene_ids`, and the
pre-`summary_source` table reading as native. Fast suite 198 passed / 1 xfailed in 86 s; `make check`
green.
