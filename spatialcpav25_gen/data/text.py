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
from typing import Any, Final, Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from spatialcpav25_gen.config import Config

__all__ = [
    "GENE_META_COLUMNS",
    "EncoderBackend",
    "GeneMeta",
    "GeneMetaUnavailableWarning",
    "TextEncoder",
    "TransformerBackend",
    "build_gene_meta",
    "celltype_descriptor",
    "descriptor_key",
    "gene_descriptor",
    "load_gene_meta",
    "load_transformer_backend",
    "region_descriptor",
]

GENE_META_COLUMNS: Final[tuple[str, ...]] = (
    "symbol",
    "full_name",
    "summary",
    "aliases",
    "ensembl_id",
)
"""Column order of ``resources/gene_meta.parquet``."""


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


def build_gene_meta(symbols: Sequence[str], cfg: Config | None = None) -> pd.DataFrame:
    """Assemble the gene-metadata table for ``symbols``, caching it at ``cfg.gene_meta_path``.

    Rows already present in the cached table are reused. Symbols still missing are looked
    up on mygene.info **only** when ``cfg.text_allow_network`` is set; otherwise - and
    whenever the lookup fails or returns nothing - they degrade to symbol-only rows and a
    :class:`GeneMetaUnavailableWarning` says how many did.

    Parameters
    ----------
    symbols
        Gene symbols, in panel order. Duplicates collapse; order is preserved.
    cfg
        Supplies ``gene_meta_path``, ``text_allow_network`` and ``mygene_species``.
        Defaults to ``Config()``. (Additive keyword: the spec writes
        ``build_gene_meta(symbols)``, which has nowhere to read those from.)

    Returns
    -------
    pandas.DataFrame
        One row per requested symbol, columns :data:`GENE_META_COLUMNS`, in the order the
        symbols were given.
    """
    cfg = Config() if cfg is None else cfg
    wanted = list(dict.fromkeys(str(s) for s in symbols))
    if not wanted:
        raise ValueError("build_gene_meta: symbols is empty")

    path = Path(cfg.gene_meta_path)
    cached = _read_gene_meta_table(path) if path.exists() else _empty_gene_meta_table()
    known = {str(row["symbol"]): row for row in _records(cached)}

    missing = [s for s in wanted if s not in known]
    fetched: dict[str, dict[str, Any]] = {}
    if missing and cfg.text_allow_network:
        try:
            fetched = _query_mygene(missing, cfg)
        except Exception as exc:  # noqa: BLE001 - any transport failure degrades, loudly
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
            record = _symbol_only_row(symbol)
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

    new_rows = [row for symbol, row in zip(wanted, rows, strict=True) if symbol not in known]
    if new_rows:
        merged = pd.concat(
            [cached, pd.DataFrame(new_rows, columns=list(GENE_META_COLUMNS))],
            ignore_index=True,
        )
        _write_gene_meta_table(merged, path)

    return pd.DataFrame(rows, columns=list(GENE_META_COLUMNS)).reset_index(drop=True)


def load_gene_meta(path: str | Path) -> dict[str, GeneMeta]:
    """Load a gene-metadata table into ``symbol -> GeneMeta``.

    Raises
    ------
    FileNotFoundError
        If the table does not exist. Building it is an explicit, network-touching step
        (:func:`build_gene_meta`); silently returning an empty mapping here would turn a
        missing table into a whole panel of bare-symbol descriptors that nobody notices
        (Convention 6).
    """
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(
            f"load_gene_meta: no gene-metadata table at {table_path} "
            "(Config.gene_meta_path). Build it once with build_gene_meta(symbols, cfg) - "
            "with Config.text_allow_network=True if the summaries are wanted."
        )
    table = _read_gene_meta_table(table_path)
    return {str(row["symbol"]): GeneMeta.from_row(row) for row in _records(table)}


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the table's rows as plain dicts with ``str`` keys."""
    return [{str(k): v for k, v in record.items()} for record in table.to_dict(orient="records")]


def _empty_gene_meta_table() -> pd.DataFrame:
    """Return an empty table with the canonical columns."""
    return pd.DataFrame({column: pd.Series(dtype=object) for column in GENE_META_COLUMNS})


def _symbol_only_row(symbol: str) -> dict[str, Any]:
    """Return the degraded row for a symbol with no metadata."""
    return {
        "symbol": symbol,
        "full_name": None,
        "summary": None,
        "aliases": [],
        "ensembl_id": None,
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


def _query_mygene(symbols: Sequence[str], cfg: Config) -> dict[str, dict[str, Any]]:
    """Query mygene.info for ``symbols``. Network path: never reached by training or tests.

    Any failure propagates to :func:`build_gene_meta`, which degrades and warns.
    """
    import mygene

    client = mygene.MyGeneInfo()
    hits = client.querymany(
        list(symbols),
        scopes="symbol,alias",
        fields="symbol,name,summary,alias,ensembl.gene",
        species=cfg.mygene_species,
        verbose=False,
    )
    out: dict[str, dict[str, Any]] = {}
    for hit in hits:
        if not isinstance(hit, dict) or hit.get("notfound"):
            continue
        query = str(hit.get("query", ""))
        if not query or query in out:  # querymany can return several hits; the first wins
            continue
        out[query] = {
            "symbol": query,
            "full_name": _clean(hit.get("name")),
            "summary": _clean(hit.get("summary")),
            "aliases": list(_coerce_aliases(hit.get("alias"))),
            "ensembl_id": _ensembl_id(hit.get("ensembl")),
        }
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
