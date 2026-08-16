"""Descriptors for genes, cell types and regions, and the frozen text encoder that embeds them.

This is the open-vocabulary mechanism: a gene enters the model through its *text*, not
through a fixed panel index, so a trained model can be asked for a gene it never saw.

Three rules shape this module.

Stability
    A descriptor string is the cache key (via :func:`descriptor_key`), so the formatting
    is a contract, not a detail: fields are stripped of surrounding whitespace and
    trailing sentence punctuation, joined by ``". "``, and the whole string ends in one
    ``"."``. Gene aliases are sorted and de-duplicated, so a table that lists them in a
    different order produces the same string and hits the same cache entry. A region's
    ancestor path is *not* sorted: its order is the hierarchy, from the nearest parent
    outwards.

Offline
    Nothing here reaches the network at train or test time (T02 "Do NOT").
    :func:`build_gene_meta` is the only function that can go online at all, and only when
    ``cfg.text_allow_network`` is set; it is meant to be run once, and its output cached
    at ``cfg.gene_meta_path``. Unknown symbols degrade to symbol-only rows and warn
    (:class:`GeneMetaUnavailableWarning`) rather than raising - the degradation is the
    documented behaviour, the warning is what keeps it from being silent (Convention 6).

Never expression
    Descriptors are built from literature metadata only. Using expression to build them
    would leak the target and destroy the zero-shot claim.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from spatialcpav25_gen.config import Config

__all__ = [
    "GENE_META_COLUMNS",
    "SPECIES_TAXID",
    "EncoderBackend",
    "GeneMeta",
    "GeneMetaError",
    "GeneMetaUnavailableWarning",
    "MyGeneClient",
    "TextEncoder",
    "TransformerBackend",
    "build_gene_meta",
    "celltype_descriptor",
    "descriptor_key",
    "gene_descriptor",
    "gene_meta_summary",
    "load_gene_meta",
    "load_mygene_client",
    "load_transformer_backend",
    "region_descriptor",
]

GENE_META_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "full_name",
    "summary",
    "aliases",
    "ensembl_id",
    "species_requested",
    "species_resolved",
)
"""Column order of ``resources/gene_meta.parquet``.

The two species columns were added after a build of a 1138-symbol **mouse** panel came back with 389
ENSMUSG ids and 744 from ground squirrel, mink, ferret and falcon, two from fruit fly, and no way to
tell from the table that anything was wrong. A table that cannot say which organism it describes
cannot be checked. ``species_resolved`` is the **taxid** as a string, because that is what the API
returns and what an assertion can be written against; ``species_requested`` is the name the caller
used."""

SPECIES_TAXID: Final[dict[str, int]] = {
    "human": 9606,
    "mouse": 10090,
    "rat": 10116,
    "fruitfly": 7227,
    "nematode": 6239,
    "zebrafish": 7955,
    "frog": 8364,
    "pig": 9823,
    "chicken": 9031,
    "macaque": 9544,
}
"""NCBI taxonomy ids of the species mygene.info names, for the ones this project might use.

Not a ``Config`` field and not a magic number: these are facts about the NCBI taxonomy, in the same
category as ``LAYOUT_MODES`` — a fixed set naming what a string-valued argument may be, not a
tunable. A caller wanting a species outside this table passes the bare taxid instead (``"10090"``),
which is checked to be digits and used directly."""


class GeneMetaError(ValueError):
    """The gene-metadata table, or the query that built it, is not what was asked for.

    Raised rather than warned when the *identity* of the data is wrong — a table of the wrong
    organism, a query that came back with a species nobody asked for. A missing summary degrades
    (that is :class:`GeneMetaUnavailableWarning`); a mouse panel annotated from falcon genes does
    not
    degrade, it silently corrupts the text channel the whole open-vocabulary claim rests on, and
    every stage downstream of it succeeds.
    """


class GeneMetaUnavailableWarning(UserWarning):
    """Metadata for one or more symbols could not be obtained, so their descriptors are
    the bare symbol.

    Expected and supported - a gene absent from mygene.info still has to be embeddable -
    but never silent: a run whose text channel is mostly bare symbols carries much less
    signal than one whose descriptors have summaries, and the difference has to be
    visible in the log.
    """


@dataclass(frozen=True)
class GeneMeta:
    """Literature metadata for one gene, as stored in ``resources/gene_meta.parquet``.

    Attributes
    ----------
    symbol
        Canonical gene symbol.
    full_name
        Full gene name, or ``None`` when unknown.
    summary
        NCBI Gene summary, or ``None`` when unknown.
    aliases
        Alternative symbols. Order is irrelevant: :func:`gene_descriptor` sorts them.
    ensembl_id
        Ensembl gene id, or ``None``. Carried for joining panels across datasets, not
        used in the descriptor.
    """

    symbol: str
    full_name: str | None = None
    summary: str | None = None
    aliases: tuple[str, ...] = ()
    ensembl_id: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GeneMeta:
        """Build a :class:`GeneMeta` from one row of a gene-metadata table."""
        return cls(
            symbol=str(row["symbol"]),
            full_name=_clean(row.get("full_name")),
            summary=_clean(row.get("summary")),
            aliases=_coerce_aliases(row.get("aliases")),
            ensembl_id=_clean(row.get("ensembl_id")),
        )


def _clean(value: Any) -> str | None:
    """Normalise one metadata field: strip, drop trailing sentence punctuation, empty -> None.

    Trailing periods are removed because the descriptor grammar supplies its own: without
    this, a summary that already ends in ``"."`` would produce ``".."`` and a different
    cache key for what is the same text.
    """
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip().rstrip(".").strip()
    return text or None


def _coerce_aliases(value: Any) -> tuple[str, ...]:
    """Normalise the alias field of a table row to a tuple of non-empty strings."""
    if value is None:
        return ()
    if isinstance(value, float) and np.isnan(value):
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = [value]
    elif isinstance(value, np.ndarray | list | tuple | pd.Series):
        items = list(value)
    else:
        raise TypeError(
            f"GeneMeta.aliases must be a sequence of strings or None; got {type(value).__name__}"
        )
    cleaned = [_clean(item) for item in items]
    return tuple(item for item in cleaned if item is not None)


# --------------------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------------------


def gene_descriptor(symbol: str, meta: GeneMeta | None) -> str:
    """Return the descriptor string for one gene.

    ``"{symbol}. {full_name}. {summary}. Aliases: {a1}, {a2}."``, with any unknown field
    dropped; ``meta=None`` (or a meta carrying nothing) gives ``"{symbol}."``. Aliases are
    sorted and de-duplicated, and an alias equal to the symbol is dropped, so the string
    does not depend on the order a table happened to list them in.

    The passed ``symbol`` is authoritative - it is the panel's spelling, which is what the
    rest of the pipeline indexes by - even when ``meta.symbol`` differs.
    """
    name = _clean(symbol)
    if name is None:
        raise ValueError("gene_descriptor: symbol must be a non-empty string")
    parts = [name]
    if meta is not None:
        full_name = _clean(meta.full_name)
        if full_name is not None:
            parts.append(full_name)
        summary = _clean(meta.summary)
        if summary is not None:
            parts.append(summary)
        aliases = sorted({a for a in _coerce_aliases(meta.aliases) if a != name})
        if aliases:
            parts.append("Aliases: " + ", ".join(aliases))
    return ". ".join(parts) + "."


def celltype_descriptor(name: str, ontology: dict[str, Any] | None) -> str:
    """Return the descriptor string for one cell type.

    ``ontology`` is a Cell Ontology record with optional ``"label"`` and ``"definition"``
    keys; the result is ``"{label}. {definition}."``, falling back to the raw ``name``
    when the label is absent and to ``"{name}."`` when ``ontology`` is ``None`` or empty.
    A non-empty ``ontology`` carrying neither key is a wrong-shaped record and raises
    rather than quietly degrading to the raw label (Convention 6).
    """
    label = _clean(name)
    if label is None:
        raise ValueError("celltype_descriptor: name must be a non-empty string")
    if not ontology:
        return f"{label}."
    if "label" not in ontology and "definition" not in ontology:
        raise ValueError(
            f"celltype_descriptor({name!r}): the ontology record has neither a 'label' nor a "
            f"'definition' key; got keys {sorted(map(str, ontology))}. Pass None for a cell "
            "type that does not resolve to an ontology term."
        )
    parts = [_clean(ontology.get("label")) or label]
    definition = _clean(ontology.get("definition"))
    if definition is not None:
        parts.append(definition)
    return ". ".join(parts) + "."


def region_descriptor(name: str, hierarchy: list[str] | None) -> str:
    """Return the descriptor string for one anatomical region.

    ``"{name}. Part of: {parent}, {grandparent}, ...."`` - e.g. ``"Primary somatosensory
    area, layer 4. Part of: Isocortex, Cerebral cortex, Brain."`` The ancestor path is
    kept in the given order (nearest parent first) because that order *is* the hierarchy;
    it is what lets an unseen region inherit meaning from its parents.
    """
    label = _clean(name)
    if label is None:
        raise ValueError("region_descriptor: name must be a non-empty string")
    ancestors = [a for a in (_clean(h) for h in hierarchy or []) if a is not None]
    if not ancestors:
        return f"{label}."
    return f"{label}. Part of: {', '.join(ancestors)}."


def descriptor_key(model_name: str, descriptor: str) -> str:
    """Return the cache key for one descriptor: ``sha256(model_name + descriptor)`` hex.

    The model name is part of the key so that two checkpoints never share a cache entry.
    """
    return hashlib.sha256((model_name + descriptor).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# gene metadata table
# --------------------------------------------------------------------------------------


def build_gene_meta(
    symbols: Sequence[str], cfg: Config | None = None, *, merge: bool = False
) -> pd.DataFrame:
    """Assemble the gene-metadata table for ``symbols``, caching it at ``cfg.gene_meta_path``.

    Symbols are looked up on mygene.info **only** when ``cfg.text_allow_network`` is set;
    otherwise - and whenever the lookup fails or returns nothing for a symbol - they degrade to
    symbol-only rows and a :class:`GeneMetaUnavailableWarning` says how many did.

    Parameters
    ----------
    symbols
        Gene symbols, in panel order. Duplicates collapse; order is preserved.
    cfg
        Supplies ``gene_meta_path``, ``text_allow_network`` and ``mygene_species``.
        Defaults to ``Config()``. (Additive keyword: the spec writes
        ``build_gene_meta(symbols)``, which has nowhere to read those from.)
    merge
        Keep rows already in the table that this call did not ask for, and reuse cached rows for
        symbols it did. **Off by default, and that is a bug fix rather than a preference.**

    Returns
    -------
    pandas.DataFrame
        One row per requested symbol, columns :data:`GENE_META_COLUMNS`, in the order the
        symbols were given. The table written to disk holds exactly these rows unless ``merge``.

    Replace, do not merge
    ---------------------
    The first version did the opposite, and it made ``Config.mygene_species`` unusable. It computed
    ``missing = [s for s in symbols if s not in cached]`` and concatenated only the new rows, so:

    * a **corrected re-run issued no queries at all.** After one bad build every symbol was cached,
      so re-running with the right species against a table full of ground squirrel genes changed
      nothing and reported success. The species parameter was never sent, because no request was
      made. This is most of why the species argument looked like it was not filtering.
    * **stale rows were never removed.** A 1122-symbol request against a table built from 1138
      symbols left the 16 extras in place, of unknown provenance and unknown organism.

    So by default this **replaces** the table with exactly the requested symbols, every one looked
    up
    afresh. ``merge=True`` restores accumulate-and-reuse for the case it was meant for — extending a
    table with another panel of the same organism — and then the existing rows' species is checked
    against the request, because merging two organisms into one symbol-keyed table is the other half
    of the same bug.
    """
    cfg = Config() if cfg is None else cfg
    wanted = list(dict.fromkeys(str(s) for s in symbols))
    if not wanted:
        raise ValueError("build_gene_meta: symbols is empty")

    path = Path(cfg.gene_meta_path)
    requested_name = resolve_species(cfg.mygene_species)[0] if cfg.text_allow_network else None
    cached: pd.DataFrame = _empty_gene_meta_table()
    known: dict[str, dict[str, Any]] = {}
    if merge and path.exists():
        cached = _read_gene_meta_table(path)
        known = {str(row["symbol"]): row for row in _records(cached)}
        if requested_name is not None:
            _check_table_species(known.values(), requested_name, path)

    missing = [s for s in wanted if s not in known]
    fetched: dict[str, dict[str, Any]] = {}
    if missing and cfg.text_allow_network:
        try:
            fetched = _query_mygene(missing, cfg)
        except GeneMetaError:
            # The identity of the data is wrong, not merely absent. Do not degrade: a table of the
            # wrong organism is worse than no table, because everything downstream succeeds.
            raise
        except Exception as exc:  # any transport failure degrades, loudly
            warnings.warn(
                f"build_gene_meta: mygene.info lookup for {len(missing)} symbol(s) failed "
                f"({type(exc).__name__}: {exc}); falling back to symbol-only rows.",
                GeneMetaUnavailableWarning,
                stacklevel=2,
            )

    rows: list[dict[str, Any]] = []
    degraded: list[str] = []
    for symbol in wanted:
        record = known.get(symbol) or fetched.get(symbol)
        if record is None:
            record = _symbol_only_row(symbol, requested_name)
            degraded.append(symbol)
        rows.append(record)

    if degraded:
        reason = (
            "Config.text_allow_network is False, so no lookup was attempted"
            if not cfg.text_allow_network
            else "mygene.info returned nothing for them"
        )
        warnings.warn(
            f"build_gene_meta: {len(degraded)} of {len(wanted)} symbol(s) have no metadata "
            f"({reason}); first: {degraded[0]!r}. Their descriptors will be the bare symbol, "
            "which carries much less signal than a name plus summary.",
            GeneMetaUnavailableWarning,
            stacklevel=2,
        )

    table = pd.DataFrame(rows, columns=list(GENE_META_COLUMNS)).reset_index(drop=True)
    written = table
    if merge and not cached.empty:
        extra = cached[~cached["symbol"].astype(str).isin(set(wanted))]
        written = pd.concat([table, extra], ignore_index=True)
    _write_gene_meta_table(written, path)
    return table


def gene_meta_summary(path: str | Path) -> dict[str, Any]:
    """Describe the table **on disk**: rows, species, coverage, and the Ensembl-prefix histogram.

    What a build should report, and what the script got wrong: it printed the number of *requested
    symbols carrying a full name* out of the number requested ("1122/1122") while the file held 1138
    rows of mixed-species data. A count that is not a property of the file cannot detect a file that
    is wrong. The prefix histogram is included because it is how the falcon genes were spotted in
    the
    first place — that should not have needed someone to think of it.
    """
    table = _read_gene_meta_table(Path(path))
    records = _records(table)
    return {
        "path": str(path),
        "rows": len(records),
        "with_full_name": sum(1 for r in records if _clean(r["full_name"])),
        "with_summary": sum(1 for r in records if _clean(r["summary"])),
        "with_ensembl_id": sum(1 for r in records if _clean(r["ensembl_id"])),
        "species_requested": sorted(
            {str(r["species_requested"]) for r in records if r["species_requested"]}
        ),
        "species_resolved": sorted(
            {str(r["species_resolved"]) for r in records if r["species_resolved"]}
        ),
        "ensembl_prefixes": _ensembl_prefix_counts(records),
    }


def _ensembl_prefix_counts(records: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Count Ensembl id prefixes (``ENSMUSG``, ``ENSFALG``, ...): the species-mixing tell."""
    counts: dict[str, int] = {}
    for record in records:
        value = _clean(record.get("ensembl_id"))
        if value is None:
            key = "None"
        else:
            digits = next((i for i, ch in enumerate(value) if ch.isdigit()), len(value))
            key = value[:digits] or value
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _check_table_species(
    records: Iterable[dict[str, Any]], requested_name: str, path: Path
) -> None:
    """Raise unless every *resolved* row of an existing table is the requested species.

    The distinction that makes this usable on a real panel: a row with **no** species resolved is
    either a symbol mygene.info had nothing for — legitimate, and every real panel has some — or a
    row from a table written before the species columns existed. The two are told apart by whether
    the row carries metadata. A row with a full name or an Ensembl id but no species is from the old
    schema and is exactly the kind of row that turned out to be another organism's, so it is
    refused;
    a row with neither is simply absent, and is allowed.
    """
    taxid = str(resolve_species(requested_name)[1])
    wrong: set[str] = set()
    unlabelled = 0
    for row in records:
        resolved = row.get("species_resolved")
        value = str(resolved) if resolved else ""
        if value:
            if value != taxid:
                wrong.add(value)
        elif _clean(row.get("full_name")) or _clean(row.get("ensembl_id")):
            unlabelled += 1
    if wrong:
        raise GeneMetaError(
            f"{path} holds rows resolved to species {sorted(wrong)} but {requested_name!r} "
            f"(taxid {taxid}) was requested. One organism per table: build the second organism at "
            "a different Config.gene_meta_path, or delete this file and rebuild it."
        )
    if unlabelled:
        raise GeneMetaError(
            f"{path} has {unlabelled} row(s) carrying metadata with no species recorded, so it "
            f"cannot be checked against the requested {requested_name!r}. That is what a table "
            "written before the species columns existed looks like, and such a table is exactly "
            "the one that turned out to hold four organisms' genes. Rebuild it with "
            "build_gene_meta "
            "(Config.text_allow_network=True)."
        )


def load_gene_meta(path: str | Path, *, species: str | None = None) -> dict[str, GeneMeta]:
    """Load a gene-metadata table into ``symbol -> GeneMeta``.

    Parameters
    ----------
    path
        The table, normally ``Config.gene_meta_path``.
    species
        The organism the caller's data is from — in the pipeline ``Config.mygene_species``. When
        given, the table's own ``species_resolved`` must match it, and a mismatch **raises**.

    Raises
    ------
    FileNotFoundError
        If the table does not exist. Building it is an explicit, network-touching step
        (:func:`build_gene_meta`); silently returning an empty mapping here would turn a
        missing table into a whole panel of bare-symbol descriptors that nobody notices
        (Convention 6).
    GeneMetaError
        If ``species`` is given and the table is not that organism's, or cannot say. A **refusal,
        not a warning**: descriptors built from another organism's gene summaries are not degraded,
        they are wrong, and everything downstream of them succeeds — the encoder produces vectors,
        the model trains, the numbers look plausible, and the text channel is describing falcon
        genes. There is no version of that which should be recoverable by ignoring a warning.
    """
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"load_gene_meta: no gene-metadata table at {table_path} "
            "(Config.gene_meta_path). Build it once with build_gene_meta(symbols, cfg) - "
            "with Config.text_allow_network=True if the summaries are wanted."
        )
    table = _read_gene_meta_table(table_path)
    records = _records(table)
    if species is not None:
        _check_table_species(records, resolve_species(species)[0], table_path)
    return {str(row["symbol"]): GeneMeta.from_row(row) for row in records}


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the table's rows as plain dicts with ``str`` keys."""
    return [{str(k): v for k, v in record.items()} for record in table.to_dict(orient="records")]


def _empty_gene_meta_table() -> pd.DataFrame:
    """Return an empty table with the canonical columns."""
    return pd.DataFrame({column: pd.Series(dtype=object) for column in GENE_META_COLUMNS})


def _symbol_only_row(symbol: str, requested_name: str | None) -> dict[str, Any]:
    """Return the degraded row for a symbol with no metadata.

    ``species_resolved`` is ``None`` — nothing resolved. ``species_requested`` still records what
    was
    asked for, so a table of degraded rows says which organism it was *meant* to describe, and
    :func:`_check_table_species` can tell "mygene knew nothing about this symbol" (fine) from "this
    row has metadata but no species" (a pre-species-column table, refused).
    """
    return {
        "symbol": symbol,
        "full_name": None,
        "summary": None,
        "aliases": [],
        "ensembl_id": None,
        "species_requested": requested_name,
        "species_resolved": None,
    }


def _read_gene_meta_table(path: Path) -> pd.DataFrame:
    """Read the parquet table, checking that it has the columns the descriptors need."""
    table = pd.read_parquet(path)
    missing = [column for column in GENE_META_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"{path} (Config.gene_meta_path) is missing column(s) {missing}; the table must "
            f"have {list(GENE_META_COLUMNS)}"
        )
    return table[list(GENE_META_COLUMNS)]


def _write_gene_meta_table(table: pd.DataFrame, path: Path) -> None:
    """Write the table to ``path``, atomically, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    table.to_parquet(tmp, index=False)
    os.replace(tmp, path)


class MyGeneClient(Protocol):
    """The one method :func:`_query_mygene` needs from ``mygene.MyGeneInfo``.

    A seam, for the same reason T02 put one in front of the transformer
    (:func:`load_transformer_backend`): this path had **no test at all**, which is how a query that
    returned falcon genes for a mouse panel survived. A fake client makes the whole of
    :func:`_query_mygene` — species filtering, hit selection, the assertions — testable offline.
    """

    def querymany(self, qterms: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Return one dict per (query term, hit) pair; see mygene.info's POST /query."""
        ...  # pragma: no cover - protocol


def load_mygene_client(cfg: Config) -> MyGeneClient:
    """Construct the mygene.info client. The only place this package imports ``mygene``.

    ``mygene`` is in the ``extra`` dependency group and is not installed by default, because nothing
    in training or testing may reach the network (T02 "Do NOT"). Tests replace this function.
    """
    import mygene

    del cfg
    return cast(MyGeneClient, mygene.MyGeneInfo())


def resolve_species(requested: str) -> tuple[str, int]:
    """Resolve a species request to ``(name, taxid)``. Exactly one species, or raise.

    ``Config.mygene_species`` used to default to ``"human,mouse"``, which a symbol-keyed table
    cannot
    honour: the same symbol exists in both organisms, so a two-species request lets whichever hit
    the
    API returned first win. That is not a filter, it is a coin toss, and it is one half of why a
    mouse
    panel came back mixed. One organism per table; a second organism is a second
    ``Config.gene_meta_path``.
    """
    names = [part.strip() for part in str(requested).split(",") if part.strip()]
    if len(names) != 1:
        raise GeneMetaError(
            f"Config.mygene_species={requested!r} names {len(names)} species. A gene-metadata "
            "table is keyed by symbol and therefore describes exactly one organism: a "
            "multi-species request lets the same symbol resolve to two different genes, with "
            "whichever the API returned first winning. Build one table per organism, each at its "
            "own Config.gene_meta_path."
        )
    name = names[0]
    if name in SPECIES_TAXID:
        return name, SPECIES_TAXID[name]
    if name.isdigit():
        return name, int(name)
    raise GeneMetaError(
        f"Config.mygene_species={name!r} is not one of {sorted(SPECIES_TAXID)} and is not a bare "
        "NCBI taxid. Pass a name from that list, or the taxid as digits (mouse is '10090')."
    )


def _is_exact(hit: dict[str, Any], query: str) -> bool:
    """Return ``True`` when the hit's own symbol equals the query, case-insensitively."""
    return str(hit.get("symbol", "")).casefold() == query.casefold()


def _hit_rank(hit: dict[str, Any], query: str) -> tuple[int, float]:
    """Sort key for choosing among several hits for one symbol: exact match first, then score."""
    try:
        score = float(hit.get("_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return (1 if _is_exact(hit, query) else 0, score)


def _query_mygene(symbols: Sequence[str], cfg: Config) -> dict[str, dict[str, Any]]:
    """Query mygene.info for ``symbols``. Network path: never reached by training.

    Returns ``symbol -> row``, containing **only** hits whose ``taxid`` is the requested species'.

    Three things this does that the first version did not, each the direct cause of a measured
    defect
    in a 1138-symbol mouse panel (389 mouse rows, 744 from four other mammals and a fly):

    ``taxid`` is requested and **verified**
        ``species`` is passed to the API *and* every hit's ``taxid`` is checked against the request.
        The first version asked for ``species="mouse"``, never looked at what came back, and wrote
        whatever it got. Whether the endpoint honours the parameter is now irrelevant: a hit from
        the
        wrong species is dropped here, and if *nothing* of the right species came back the query
        raises rather than degrading: a systematic species failure is not a per-gene absence.

    the best hit is chosen, not the first
        ``querymany`` returns one entry per (query, hit) pair and a symbol routinely matches several
        genes — the more so under ``scopes="symbol,alias"``, where another gene's *alias* can arrive
        ahead of the exact symbol match. The first version kept whichever came first. Hits are now
        ranked: right species, then an exact (case-insensitive) symbol match ahead of an alias
        match,
        then mygene's own ``_score``.

    ambiguity is counted and reported
        A symbol still matching two same-species genes exactly is a real paralog ambiguity, not a
        bug; it is resolved by ``_score``, counted, and warned about, so a panel full of them is
        visible rather than silent.
    """
    name, taxid = resolve_species(cfg.mygene_species)
    client = load_mygene_client(cfg)
    hits = client.querymany(
        list(symbols),
        scopes="symbol,alias",
        # taxid is not optional: it is what the species check below reads.
        fields="symbol,name,summary,alias,ensembl.gene,taxid",
        species=name,
        verbose=False,
    )

    by_symbol: dict[str, list[dict[str, Any]]] = {}
    wrong_species: dict[str, int] = {}
    for hit in hits:
        if not isinstance(hit, dict) or hit.get("notfound"):
            continue
        query = str(hit.get("query", ""))
        if not query:
            continue
        hit_taxid = hit.get("taxid")
        if hit_taxid is None or int(hit_taxid) != taxid:
            key = "missing" if hit_taxid is None else str(int(hit_taxid))
            wrong_species[key] = wrong_species.get(key, 0) + 1
            continue
        by_symbol.setdefault(query, []).append(hit)

    if not by_symbol and wrong_species:
        raise GeneMetaError(
            f"_query_mygene: not one of {len(symbols)} symbol(s) resolved to species {name!r} "
            f"(taxid {taxid}); what came back was {dict(sorted(wrong_species.items()))} by taxid. "
            "The species filter is not being applied by the endpoint - do not write this table."
        )
    if wrong_species:
        warnings.warn(
            f"_query_mygene: dropped {sum(wrong_species.values())} hit(s) from other species while "
            f"resolving {name!r} (taxid {taxid}); by taxid: {dict(sorted(wrong_species.items()))}. "
            "Dropped, not written - but a large count means the endpoint is ignoring the species "
            "parameter and this request is doing the filtering by itself.",
            GeneMetaUnavailableWarning,
            stacklevel=2,
        )

    out: dict[str, dict[str, Any]] = {}
    ambiguous: list[str] = []
    for query, candidates in by_symbol.items():
        ranked = sorted(candidates, key=lambda hit: _hit_rank(hit, query), reverse=True)
        best = ranked[0]
        if len(ranked) > 1 and _is_exact(best, query) and _is_exact(ranked[1], query):
            ambiguous.append(query)
        out[query] = {
            "symbol": query,
            "full_name": _clean(best.get("name")),
            "summary": _clean(best.get("summary")),
            "aliases": list(_coerce_aliases(best.get("alias"))),
            "ensembl_id": _ensembl_id(best.get("ensembl")),
            "species_requested": name,
            "species_resolved": str(taxid),
        }
    if ambiguous:
        warnings.warn(
            f"_query_mygene: {len(ambiguous)} symbol(s) matched more than one {name} gene exactly "
            f"(first: {ambiguous[0]!r}); kept the highest-scoring hit. Paralogous symbols do this "
            "legitimately, but a large count means the panel's symbols are not unique in this "
            "organism and the join should be on ensembl_id instead.",
            GeneMetaUnavailableWarning,
            stacklevel=2,
        )
    return out


def _ensembl_id(value: Any) -> str | None:
    """Pull a single Ensembl gene id out of mygene's dict-or-list-of-dicts field."""
    if isinstance(value, dict):
        return _clean(value.get("gene"))
    if isinstance(value, list) and value:
        return _ensembl_id(value[0])
    return _clean(value)


# --------------------------------------------------------------------------------------
# encoder
# --------------------------------------------------------------------------------------


class EncoderBackend(Protocol):
    """Anything that turns descriptors into pooled sentence vectors.

    The seam :class:`TextEncoder` loads lazily, so that a fully cached run never
    constructs one - and so that a test can substitute a deterministic stand-in for a
    checkpoint it cannot download.
    """

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return ``(B, D)`` float32 pooled, *unnormalised* vectors for ``B`` texts."""
        ...


class TransformerBackend:
    """The real backend: ``cfg.text_model`` under ``transformers``, frozen and in eval mode."""

    def __init__(self, cfg: Config) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.text_model)
        self.model = AutoModel.from_pretrained(cfg.text_model)
        # Frozen, and it stays frozen: the adaptation lives in W and r (T02 "Do NOT").
        self.model.eval()
        self.model.requires_grad_(False)

    def encode_batch(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Return ``(B, cfg.text_dim_in)`` float32 pooled vectors for ``B`` texts."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.text_max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state  # (B, L, D)
        if self.cfg.text_pooling == "cls":
            # MedCPT-Query-Encoder's model card takes the first-token ([CLS]) state: the
            # checkpoint is trained contrastively on *that* representation, so mean-pooling
            # would read out something it was never optimised for (SPEC_QUESTIONS C3).
            pooled = hidden[:, 0]
        else:
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # (B, L, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return np.asarray(pooled.to(torch.float32).cpu().numpy(), dtype=np.float32)


def load_transformer_backend(cfg: Config) -> EncoderBackend:
    """Construct the frozen transformer backend.

    A module-level function rather than a method so that the model load is a single,
    monkeypatchable seam: ``test_cache_hit_avoids_model_load`` replaces it with something
    that raises and asserts the cached path never calls it.
    """
    return TransformerBackend(cfg)


class TextEncoder:
    """Encode descriptors with the frozen text model, caching every vector on disk.

    The cache is keyed by ``sha256(cfg.text_model + descriptor)``, one ``.npy`` per
    descriptor under ``cfg.text_cache_dir``. On a full cache hit the transformer is never
    loaded, which is what keeps CPU-only test runs fast.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache_dir = Path(cfg.text_cache_dir)
        self._backend: EncoderBackend | None = None
        self._n_model_calls = 0

    @property
    def n_model_calls(self) -> int:
        """How many batches went through the model; ``0`` means everything came from cache."""
        return self._n_model_calls

    def encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode descriptors. ``list[str]`` of length ``T`` -> ``(T, 768)`` float32.

        Rows are L2-normalised, so a dot product between two of them is a cosine
        similarity. Repeated descriptors are encoded once and broadcast back to every
        position they occupy.
        """
        for i, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"TextEncoder.encode: texts[{i}] is not a non-empty string")
        keys = [descriptor_key(self.cfg.text_model, text) for text in texts]
        if not keys:
            return np.zeros((0, self.cfg.text_dim_in), dtype=np.float32)

        unique: dict[str, str] = dict(zip(keys, texts, strict=True))
        vectors: dict[str, npt.NDArray[np.float32]] = {}
        pending: list[tuple[str, str]] = []
        for key, text in unique.items():
            cached = self._load_cached(key)
            if cached is None:
                pending.append((key, text))
            else:
                vectors[key] = cached

        for start in range(0, len(pending), self.cfg.text_batch_size):
            batch = pending[start : start + self.cfg.text_batch_size]
            raw = np.asarray(
                self._get_backend().encode_batch([text for _, text in batch]), dtype=np.float32
            )
            self._n_model_calls += 1
            if raw.shape != (len(batch), self.cfg.text_dim_in):
                raise ValueError(
                    f"TextEncoder: {self.cfg.text_model} returned {raw.shape}, expected "
                    f"({len(batch)}, {self.cfg.text_dim_in}) (Config.text_dim_in)"
                )
            for (key, _), vector in zip(batch, _l2_normalise(raw), strict=True):
                vectors[key] = vector
                self._store(key, vector)

        out = np.stack([vectors[key] for key in keys]).astype(np.float32)
        if self.cfg.debug_shapes:
            assert out.shape == (len(texts), self.cfg.text_dim_in)
        return out

    def _get_backend(self) -> EncoderBackend:
        """Load the text model on first use, and only on first use."""
        if self._backend is None:
            self._backend = load_transformer_backend(self.cfg)
        return self._backend

    def _cache_path(self, key: str) -> Path:
        """Path of one cached vector."""
        return self.cache_dir / f"{key}.npy"

    def _load_cached(self, key: str) -> npt.NDArray[np.float32] | None:
        """Return the cached vector for ``key``, or ``None`` when it is not cached."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            vector = np.load(path)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"TextEncoder: cached vector {path} is unreadable ({exc}); delete it and "
                "re-encode rather than training on a truncated embedding"
            ) from exc
        if vector.shape != (self.cfg.text_dim_in,):
            raise ValueError(
                f"TextEncoder: cached vector {path} has shape {vector.shape}, expected "
                f"({self.cfg.text_dim_in},) (Config.text_dim_in)"
            )
        return np.asarray(vector, dtype=np.float32)

    def _store(self, key: str, vector: npt.NDArray[np.float32]) -> None:
        """Write one vector to the cache, atomically."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(tmp, vector)
        os.replace(tmp, path)


def _l2_normalise(vectors: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """L2-normalise the rows of ``(B, D)``; a zero row is a broken encoder, not a warning."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or float(norms.min()) == 0.0:
        raise ValueError(
            "TextEncoder: the text model returned a zero or non-finite vector; refusing to "
            "normalise it into arbitrary directions"
        )
    return np.asarray(vectors / norms, dtype=np.float32)
