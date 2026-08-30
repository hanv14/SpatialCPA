"""T02 acceptance tests: descriptors, the cached frozen encoder, and the learned embedding.

Every numeric threshold here comes from `specs/02_TASK_text_embeddings.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.text import (
    GENE_META_COLUMNS,
    GeneMeta,
    GeneMetaError,
    GeneMetaUnavailableWarning,
    TextEncoder,
    build_gene_meta,
    celltype_descriptor,
    descriptor_key,
    gene_descriptor,
    gene_meta_summary,
    load_gene_meta,
    region_descriptor,
    resolve_species,
)
from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    TextGroundedEmbedding,
    ZeroShotView,
    build_entity_embeddings,
    describe_entity_descriptors,
    text_embedding_diagnostics,
)

from tests.fixtures.text import (
    fake_text_vecs,
    install_fake_backend,
    install_raising_backend,
)

# Spec thresholds.
DISTILL_STEPS = 200
DISTILL_MIN_DROP = 0.5  # "distillation MSE drops >= 50%"
ZERO_SHOT_N = 10

# Test-local optimiser settings for the distillation loop. Not Config fields: the trainer's
# schedule (T06) is not what this test measures, only that the head can fit the residual.
DISTILL_LR = 1e-3

# The fixture's gene names are arbitrary strings, so text similarity cannot track
# co-expression: the spec expects ~0 there. Loose enough for pair-correlation slack.
FIXTURE_SPEARMAN_TOL = 0.1

META = GeneMeta(
    symbol="Gad1",
    full_name="glutamate decarboxylase 1",
    summary="Catalyses the production of GABA from glutamate.",
    aliases=("Gad67", "Gad-1"),
    ensembl_id="ENSMUSG00000070880",
)


@pytest.fixture
def cache_cfg(tmp_path) -> Config:
    """A config whose text cache and gene-meta table live in a temporary directory."""
    return Config().replace(
        text_cache_dir=str(tmp_path / "text_cache"),
        gene_meta_path=str(tmp_path / "resources" / "gene_meta.parquet"),
    )


# --------------------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------------------


def test_descriptor_stability():
    """Descriptors are deterministic, and alias order does not change the string."""
    expected = (
        "Gad1. glutamate decarboxylase 1. "
        "Catalyses the production of GABA from glutamate. Aliases: Gad-1, Gad67."
    )
    assert gene_descriptor("Gad1", META) == expected
    assert gene_descriptor("Gad1", META) == gene_descriptor("Gad1", META)

    shuffled = GeneMeta(
        symbol=META.symbol,
        full_name=META.full_name,
        summary=META.summary,
        aliases=("Gad-1", "Gad67"),
        ensembl_id=META.ensembl_id,
    )
    assert gene_descriptor("Gad1", shuffled) == expected

    # Duplicates and an alias equal to the symbol do not change it either.
    noisy = GeneMeta(
        symbol=META.symbol,
        full_name=META.full_name,
        summary=META.summary,
        aliases=("Gad67", "Gad1", "Gad-1", "Gad67"),
        ensembl_id=META.ensembl_id,
    )
    assert gene_descriptor("Gad1", noisy) == expected

    # Trailing punctuation in the source fields does not double up.
    trailing = GeneMeta(symbol="Gad1", full_name="glutamate decarboxylase 1.", summary=None)
    assert gene_descriptor("Gad1", trailing) == "Gad1. glutamate decarboxylase 1."

    assert (
        region_descriptor(
            "Primary somatosensory area, layer 4",
            ["Isocortex", "Cerebral cortex", "Brain"],
        )
        == "Primary somatosensory area, layer 4. Part of: Isocortex, Cerebral cortex, Brain."
    )
    # The ancestor path is a hierarchy: its order is meaning, so it is not sorted.
    assert region_descriptor("R", ["B", "A"]) != region_descriptor("R", ["A", "B"])
    assert region_descriptor("R", None) == "R."
    assert region_descriptor("R", []) == "R."

    assert (
        celltype_descriptor(
            "Pvalb interneuron",
            {
                "label": "PV GABAergic cortical interneuron",
                "definition": "An interneuron expressing Pvalb",
            },
        )
        == "PV GABAergic cortical interneuron. An interneuron expressing Pvalb."
    )
    assert celltype_descriptor("Pvalb interneuron", None) == "Pvalb interneuron."
    assert celltype_descriptor("Pvalb interneuron", {}) == "Pvalb interneuron."
    with pytest.raises(ValueError, match="label"):
        celltype_descriptor("Pvalb interneuron", {"id": "CL:0000617"})


def test_descriptor_key_depends_on_model_and_text():
    a = descriptor_key("ncbi/MedCPT-Query-Encoder", "Gad1.")
    b = descriptor_key("ncbi/MedCPT-Article-Encoder", "Gad1.")
    c = descriptor_key("ncbi/MedCPT-Query-Encoder", "Gad2.")
    assert len({a, b, c}) == 3
    assert a == descriptor_key("ncbi/MedCPT-Query-Encoder", "Gad1.")


# --------------------------------------------------------------------------------------
# encoder + cache
# --------------------------------------------------------------------------------------


def test_cache_hit_avoids_model_load(monkeypatch, cache_cfg):
    """A second encoder over the same descriptors must never touch the transformer."""
    texts = [gene_descriptor("Gad1", META), gene_descriptor("Slc17a7", None)]

    backend = install_fake_backend(monkeypatch, cache_cfg)
    first = TextEncoder(cache_cfg).encode(texts)
    assert backend.calls >= 1
    assert first.shape == (2, cache_cfg.text_dim_in)

    install_raising_backend(monkeypatch)
    encoder = TextEncoder(cache_cfg)
    second = encoder.encode(texts)
    assert encoder.n_model_calls == 0
    assert np.array_equal(first, second)


def test_encode_is_normalised_and_deterministic(monkeypatch, cache_cfg, tmp_path):
    texts = [f"Gene{i:03d}." for i in range(5)] + ["Gene000."]
    install_fake_backend(monkeypatch, cache_cfg)
    vecs = TextEncoder(cache_cfg).encode(texts)

    assert vecs.shape == (6, cache_cfg.text_dim_in)
    assert vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    # A repeated descriptor is encoded once and broadcast.
    assert np.array_equal(vecs[0], vecs[5])

    other = cache_cfg.replace(text_cache_dir=str(tmp_path / "second_cache"))
    install_fake_backend(monkeypatch, other)
    assert np.array_equal(vecs, TextEncoder(other).encode(texts))


def test_missing_gene_meta_degrades(monkeypatch, cache_cfg):
    """An unknown symbol yields "{symbol}." and encodes fine."""
    descriptor = gene_descriptor("Xkr4", None)
    assert descriptor == "Xkr4."

    install_fake_backend(monkeypatch, cache_cfg)
    vecs = TextEncoder(cache_cfg).encode([descriptor])
    assert vecs.shape == (1, cache_cfg.text_dim_in)
    assert np.all(np.isfinite(vecs))
    assert np.isclose(np.linalg.norm(vecs[0]), 1.0, atol=1e-5)


def test_offline(cache_cfg):
    """With no network and no cache, build_gene_meta returns symbol-only rows and does not raise."""
    symbols = ["Gad1", "Slc17a7", "Xkr4"]
    assert cache_cfg.text_allow_network is False

    with pytest.warns(GeneMetaUnavailableWarning):
        table = build_gene_meta(symbols, cache_cfg)

    assert list(table.columns) == list(GENE_META_COLUMNS)
    assert list(table["symbol"]) == symbols
    assert table["full_name"].isna().all()
    assert table["summary"].isna().all()
    assert all(len(a) == 0 for a in table["aliases"])

    # Descriptors built from it are the bare symbols, and the table round-trips.
    meta = load_gene_meta(cache_cfg.gene_meta_path)
    assert [gene_descriptor(s, meta.get(s)) for s in symbols] == ["Gad1.", "Slc17a7.", "Xkr4."]


def test_offline_survives_a_failed_lookup(monkeypatch, cache_cfg):
    """Network allowed but unreachable: degrade to symbol-only rows, warn, do not raise."""
    from spatialcpav25_gen.data import text as text_mod

    def _boom(symbols, cfg):
        raise ConnectionError("no route to host")

    monkeypatch.setattr(text_mod, "_query_mygene", _boom)
    online = cache_cfg.replace(text_allow_network=True)
    with pytest.warns(GeneMetaUnavailableWarning):
        table = build_gene_meta(["Gad1"], online)
    assert list(table["symbol"]) == ["Gad1"]


def test_gene_meta_table_accumulates_offline(cache_cfg):
    """Offline, a cached row survives a later build of a different symbol.

    The cache is a *fallback* for a symbol the lookup could not reach — which offline is all of
    them — so an offline re-run keeps good rows rather than degrading them to symbol-only ones.
    """
    with pytest.warns(GeneMetaUnavailableWarning):
        build_gene_meta(["Gad1"], cache_cfg)

    import pandas as pd

    table = pd.read_parquet(cache_cfg.gene_meta_path)
    table.loc[table["symbol"] == "Gad1", "full_name"] = "glutamate decarboxylase 1"
    table.to_parquet(cache_cfg.gene_meta_path, index=False)

    with pytest.warns(GeneMetaUnavailableWarning):
        out = build_gene_meta(["Slc17a7"], cache_cfg)
    assert list(out["symbol"]) == ["Slc17a7"]

    meta = load_gene_meta(cache_cfg.gene_meta_path)
    assert set(meta) == {"Gad1", "Slc17a7"}
    assert gene_descriptor("Gad1", meta["Gad1"]) == "Gad1. glutamate decarboxylase 1."


def test_load_gene_meta_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="gene_meta_path"):
        load_gene_meta(tmp_path / "nope.parquet")


# --------------------------------------------------------------------------------------
# learned embedding
# --------------------------------------------------------------------------------------


def _embedding(cfg: Config, n: int = 32, out_dim: int | None = None) -> TextGroundedEmbedding:
    vecs = torch.from_numpy(fake_text_vecs(n, cfg.text_dim_in, seed=cfg.seed + 1))
    return TextGroundedEmbedding(vecs, cfg.gene_emb_dim if out_dim is None else out_dim, cfg)


def test_zero_shot_shapes(cfg):
    emb = _embedding(cfg)
    unseen = torch.from_numpy(fake_text_vecs(ZERO_SHOT_N, cfg.text_dim_in, seed=99))

    assert emb.forward_zero_shot(unseen).shape == (ZERO_SHOT_N, cfg.gene_emb_dim)
    assert emb.forward_zero_shot(unseen, use_distill=False).shape == (
        ZERO_SHOT_N,
        cfg.gene_emb_dim,
    )
    assert emb(torch.arange(4)).shape == (4, cfg.gene_emb_dim)


def test_gamma_anneal(cfg):
    """gamma is 0 at progress 0 - where the output is exactly LayerNorm(W t) - and 1 at 1."""
    emb = _embedding(cfg)
    idx = torch.arange(emb.n_entities)
    # A non-zero residual, so "equals LayerNorm(W t)" tests the gate, not the zeros init.
    with torch.no_grad():
        emb.r.weight.normal_(generator=torch.Generator().manual_seed(cfg.seed))

    emb.set_progress(0.0)
    assert float(emb.gamma) == 0.0
    expected = emb.norm(emb.W(emb.text_vecs[idx]))
    assert torch.equal(emb(idx), expected)

    emb.set_progress(1.0)
    assert float(emb.gamma) == 1.0
    assert not torch.equal(emb(idx), expected)

    emb.set_progress(cfg.residual_gate_warmup_frac / 2.0)
    assert float(emb.gamma) == pytest.approx(0.5)

    with pytest.raises(ValueError, match="frac"):
        emb.set_progress(1.5)


def test_residual_is_zero_initialised(cfg):
    """Zeros init plus the anneal is what forces W to learn before r can shortcut it."""
    emb = _embedding(cfg)
    assert torch.equal(emb.r.weight, torch.zeros_like(emb.r.weight))
    assert float(emb.gamma) == 0.0


def test_distillation_reduces_error(cfg):
    """200 steps against a random residual: the distillation MSE drops by at least 50%."""
    emb = _embedding(cfg)
    with torch.no_grad():
        emb.r.weight.normal_(generator=torch.Generator().manual_seed(cfg.seed + 7))

    opt = torch.optim.Adam(emb.distill.parameters(), lr=DISTILL_LR)
    before = float(emb.distillation_loss())
    for _ in range(DISTILL_STEPS):
        opt.zero_grad()
        loss = emb.distillation_loss()
        loss.backward()
        opt.step()
    after = float(emb.distillation_loss())

    assert after <= (1.0 - DISTILL_MIN_DROP) * before, f"{before:.4f} -> {after:.4f}"


def test_distillation_does_not_train_the_residual(cfg):
    """psi chases r, not the other way round: the residual is detached in the loss."""
    emb = _embedding(cfg)
    with torch.no_grad():
        emb.r.weight.normal_(generator=torch.Generator().manual_seed(cfg.seed + 7))
    emb.distillation_loss().backward()
    assert emb.r.weight.grad is None or torch.equal(
        emb.r.weight.grad, torch.zeros_like(emb.r.weight)
    )
    assert emb.W.weight.grad is None


def test_construction_is_deterministic(cfg):
    """Two constructions with the same config are bitwise identical (Convention 3)."""
    a, b = _embedding(cfg), _embedding(cfg)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), name

    other = _embedding(cfg.replace(seed=cfg.seed + 1))
    assert not torch.equal(a.W.weight, other.W.weight)


def test_entity_embeddings_widths(cfg):
    genes = torch.from_numpy(fake_text_vecs(12, cfg.text_dim_in, seed=1))
    types = torch.from_numpy(fake_text_vecs(6, cfg.text_dim_in, seed=2))
    regions = torch.from_numpy(fake_text_vecs(3, cfg.text_dim_in, seed=3))
    ent = EntityEmbeddings(cfg, genes, types, regions)

    assert ent.gene(torch.arange(12)).shape == (12, cfg.gene_emb_dim)
    assert ent.celltype(torch.arange(6)).shape == (6, cfg.ctx_emb_dim)
    assert ent.region is not None
    assert ent.region(torch.arange(3)).shape == (3, cfg.ctx_emb_dim)

    ent.set_progress(1.0)
    assert float(ent.gene.gamma) == 1.0
    assert float(ent.celltype.gamma) == 1.0
    assert float(ent.region.gamma) == 1.0
    assert float(ent.distillation_loss()) >= 0.0

    without_regions = EntityEmbeddings(cfg, genes, types, None)
    assert without_regions.region is None


def test_bad_text_vecs_raise(cfg):
    with pytest.raises(ValueError, match="text_dim_in"):
        TextGroundedEmbedding(torch.zeros(4, cfg.text_dim_in + 1), cfg.gene_emb_dim, cfg)
    with pytest.raises(ValueError, match=r"\(V, 768\)"):
        TextGroundedEmbedding(torch.zeros(cfg.text_dim_in), cfg.gene_emb_dim, cfg)


# --------------------------------------------------------------------------------------
# diagnostics (ablation A3)
# --------------------------------------------------------------------------------------


def test_diagnostics_on_synthetic(monkeypatch, cache_cfg, volume):
    """The A3 diagnostic runs on the fixture; text similarity there is ~0 by construction."""
    install_fake_backend(monkeypatch, cache_cfg)
    descriptors = [gene_descriptor(g, None) for g in volume.gene_names]
    vecs = torch.from_numpy(TextEncoder(cache_cfg).encode(descriptors))
    emb = TextGroundedEmbedding(vecs, cache_cfg.gene_emb_dim, cache_cfg)
    emb.set_progress(1.0)

    expr = np.vstack([s.counts.toarray() for s in volume.sections])
    diag = text_embedding_diagnostics(emb, expr, seed=cache_cfg.seed)

    assert set(diag) >= {
        "text_coexpr_spearman",
        "residual_norm_ratio",
        "knn_purity",
        "n_genes",
        "n_pairs",
        "n_modules",
        "knn_k",
    }
    assert all(np.isfinite(v) for v in diag.values())
    assert abs(diag["text_coexpr_spearman"]) < FIXTURE_SPEARMAN_TOL
    assert 0.0 <= diag["knn_purity"] <= 1.0
    # Zeros-initialised residual: nothing has been learned yet, so the ratio is exactly 0.
    assert diag["residual_norm_ratio"] == 0.0
    assert diag["n_modules"] >= 1
    assert diag["knn_k"] == cache_cfg.text_diag_knn_k
    assert diag["n_genes"] <= len(volume.gene_names)

    same = text_embedding_diagnostics(emb, expr, seed=cache_cfg.seed)
    assert same == diag


def test_diagnostics_sees_planted_signal(cfg):
    """A text space aligned with co-expression scores far above the arbitrary-name case."""
    gen = np.random.default_rng(cfg.seed)
    n_modules, per_module, n_cells = 4, 8, 400
    n_genes = n_modules * per_module

    module_of = np.repeat(np.arange(n_modules), per_module)
    module_activity = gen.standard_normal((n_cells, n_modules))
    jitter = 0.1 * gen.standard_normal((n_cells, n_genes))
    rate = np.exp(1.5 + module_activity[:, module_of] + jitter)
    expr = gen.poisson(rate).astype(np.float64)

    # Text vectors that agree with the modules: same module -> nearby vectors.
    centres = gen.standard_normal((n_modules, cfg.text_dim_in))
    vecs = centres[module_of] + 0.1 * gen.standard_normal((n_genes, cfg.text_dim_in))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    emb = TextGroundedEmbedding(torch.from_numpy(vecs.astype(np.float32)), cfg.gene_emb_dim, cfg)

    diag = text_embedding_diagnostics(emb, expr, seed=cfg.seed)
    assert diag["text_coexpr_spearman"] > FIXTURE_SPEARMAN_TOL
    assert diag["knn_purity"] > 0.5


def test_diagnostics_rejects_mismatched_expression(cfg):
    emb = _embedding(cfg, n=5)
    with pytest.raises(ValueError, match="n_entities"):
        text_embedding_diagnostics(emb, np.zeros((10, 4)), seed=0)


# --------------------------------------------------------------------------------------
# the mygene query — regression tests for a measured multi-species build
# --------------------------------------------------------------------------------------
#
# A 1138-symbol MOUSE panel came back with 389 ENSMUSG ids and 744 from ground squirrel (ENSMSIG),
# mink (ENSNVIG), ferret (ENSMPUG) and falcon (ENSFALG), 2 from fruit fly, and 5 with no id at all;
# summaries fell to 144/1138. Nothing raised. The network path had no test at all, which is how it
# survived; it has one now — a fake client reproducing exactly that response shape.

TAXID_MOUSE = 10090
TAXID_GROUND_SQUIRREL = 43179
TAXID_FALCON = 8954


class FakeMyGene:
    """A mygene client that ignores the ``species`` parameter, as the real one appeared to.

    Returns, per symbol: an alias hit from another species *first* (so "the first wins" picks the
    wrong one), then the exact mouse hit. That is the response the old code turned into falcon rows.
    """

    def __init__(self, *, honour_species: bool = False, mouse_only: bool = False) -> None:
        self.honour_species = honour_species
        self.mouse_only = mouse_only
        self.calls: list[dict[str, object]] = []

    def querymany(self, qterms, **kwargs):
        self.calls.append({"n": len(qterms), **kwargs})
        hits = []
        for symbol in qterms:
            if not (self.honour_species or self.mouse_only):
                hits.append(
                    {
                        "query": symbol,
                        "symbol": f"{symbol}-like",  # an ALIAS match, not the symbol
                        "name": f"{symbol} ortholog (falcon)",
                        "taxid": TAXID_FALCON,
                        "ensembl": {"gene": "ENSFALG00000012345"},
                        "_score": 90.0,
                    }
                )
                hits.append(
                    {
                        "query": symbol,
                        "symbol": symbol,
                        "name": f"{symbol} ortholog (ground squirrel)",
                        "taxid": TAXID_GROUND_SQUIRREL,
                        "ensembl": {"gene": "ENSMSIG00000054321"},
                        "_score": 80.0,
                    }
                )
            hits.append(
                {
                    "query": symbol,
                    "symbol": symbol,
                    "name": f"{symbol} full name",
                    "summary": f"{symbol} does something in mouse.",
                    "alias": [f"{symbol}a"],
                    "taxid": TAXID_MOUSE,
                    "ensembl": {"gene": "ENSMUSG00000000001"},
                    "_score": 70.0,
                }
            )
        return hits


class FakeMyGeneWrongSpeciesOnly:
    """Every hit is from another species: a systematic filter failure, not a per-gene absence."""

    def querymany(self, qterms, **kwargs):
        return [
            {
                "query": s,
                "symbol": s,
                "name": "falcon thing",
                "taxid": TAXID_FALCON,
                "ensembl": {"gene": "ENSFALG00000000001"},
            }
            for s in qterms
        ]


def install_fake_mygene(monkeypatch, client):
    """Point ``load_mygene_client`` at ``client`` and return it."""
    from spatialcpav25_gen.data import text as text_mod

    monkeypatch.setattr(text_mod, "load_mygene_client", lambda _cfg: client)
    return client


def test_resolve_species_requires_exactly_one():
    """A gene table describes one organism; ``"human,mouse"`` is refused, taxids are accepted."""
    assert resolve_species("mouse") == ("mouse", 10090)
    assert resolve_species(" human ") == ("human", 9606)
    assert resolve_species("10090") == ("10090", 10090)
    for bad in ("human,mouse", "", "wombat", "mouse,"):
        if bad == "mouse,":
            assert resolve_species(bad) == ("mouse", 10090)  # a trailing comma is not two species
            continue
        with pytest.raises(GeneMetaError):
            resolve_species(bad)


def test_query_keeps_only_the_requested_species(monkeypatch, cache_cfg):
    """The species filter is enforced **after** the query, not trusted.

    The fake client ignores ``species`` and returns a falcon alias hit first, a ground-squirrel hit
    second and the mouse hit last — the response that produced 744 wrong rows. Every emitted row
    must be mouse, and must be the *exact* mouse hit rather than the first thing that arrived.
    """
    client = install_fake_mygene(monkeypatch, FakeMyGene())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.warns(GeneMetaUnavailableWarning, match="other species"):
        table = build_gene_meta(["Gad1", "Slc17a7"], online)

    assert list(table["symbol"]) == ["Gad1", "Slc17a7"]
    assert list(table["species_requested"]) == ["mouse", "mouse"]
    assert list(table["species_resolved"]) == ["10090", "10090"]
    assert all(str(v).startswith("ENSMUSG") for v in table["ensembl_id"])
    assert list(table["full_name"]) == ["Gad1 full name", "Slc17a7 full name"]
    # `_clean` strips the trailing period (T02's descriptor grammar), so match without it.
    assert all(str(v).endswith("in mouse") for v in table["summary"])
    # and the request did carry the species, whatever the endpoint then did with it
    assert client.calls[0]["species"] == "mouse"
    assert "taxid" in str(client.calls[0]["fields"])


def test_query_raises_when_nothing_is_the_requested_species(monkeypatch, cache_cfg):
    """A systematically wrong species is an error, not a degradation.

    Degrading here is what wrote the table the bug report is about: every row present, every row
    wrong, every downstream stage succeeding.
    """
    install_fake_mygene(monkeypatch, FakeMyGeneWrongSpeciesOnly())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.raises(GeneMetaError, match="not one of"):
        build_gene_meta(["Gad1", "Slc17a7"], online)
    # nothing was written
    assert not Path(online.gene_meta_path).exists()


def test_build_merges_but_always_requeries(monkeypatch, cache_cfg):
    """The default keeps other panels' rows AND re-queries every requested symbol.

    Three failures have to be avoided at once, and this pins all three:

    * building one panel must not destroy another's rows — the second occurrence of that loss
      wiped the STARmap and Zhuang panels during a ``deep_starmap`` build;
    * a corrected ``--species`` re-run must still issue its queries, which is what the old
      accumulate-and-reuse merge broke (SPEC_QUESTIONS B19a);
    * ``--overwrite`` must still be able to say "this table is exactly this panel".
    """
    client = install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1", "Slc17a7", "Pvalb"], online)
    assert gene_meta_summary(online.gene_meta_path)["rows"] == 3

    # A smaller request KEEPS the other panel's row — and re-queries the two it asked for, so a
    # corrected re-run is a real re-run rather than a cache hit reported as success.
    before = len(client.calls)
    build_gene_meta(["Gad1", "Slc17a7"], online)
    summary = gene_meta_summary(online.gene_meta_path)
    assert summary["rows"] == 3, "a merge must not drop the unrequested row"
    assert set(load_gene_meta(online.gene_meta_path)) == {"Gad1", "Slc17a7", "Pvalb"}
    assert client.calls[before]["n"] == 2, "a re-run must re-query, not short-circuit on the cache"

    # A disjoint panel accumulates rather than replacing — the deep_starmap case.
    build_gene_meta(["Sox10"], online)
    assert set(load_gene_meta(online.gene_meta_path)) == {"Gad1", "Slc17a7", "Pvalb", "Sox10"}

    # overwrite=True is the explicit way to make the table exactly one panel.
    build_gene_meta(["Gad1"], online, overwrite=True)
    assert gene_meta_summary(online.gene_meta_path)["rows"] == 1
    assert set(load_gene_meta(online.gene_meta_path)) == {"Gad1"}


def test_gene_meta_summary_describes_the_file(monkeypatch, cache_cfg):
    """The reported counts are properties of the file, including the Ensembl-prefix histogram."""
    install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1", "Slc17a7"], online)
    summary = gene_meta_summary(online.gene_meta_path)
    assert summary["rows"] == 2
    assert summary["with_summary"] == 2
    assert summary["species_resolved"] == ["10090"]
    assert summary["ensembl_prefixes"] == {"ENSMUSG": 2}


def test_load_gene_meta_refuses_a_species_mismatch(monkeypatch, cache_cfg):
    """``load_gene_meta`` **raises** on the wrong organism, and on a table with no species."""
    install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1"], online)

    assert set(load_gene_meta(online.gene_meta_path, species="mouse")) == {"Gad1"}
    with pytest.raises(GeneMetaError, match="One organism per table"):
        load_gene_meta(online.gene_meta_path, species="human")

    # A table written before the species columns existed reads as unlabelled, and is refused: its
    # rows carry metadata but cannot be checked. A *degraded* row (no metadata, no species) is a
    # different thing and must still load — every real panel has symbols mygene.info knows nothing
    # about, and refusing those would make the gate unusable.
    import pandas as pd

    table = pd.read_parquet(online.gene_meta_path)
    table["species_resolved"] = None
    table.to_parquet(online.gene_meta_path, index=False)
    with pytest.raises(GeneMetaError, match="no species recorded"):
        load_gene_meta(online.gene_meta_path, species="mouse")


def test_species_gate_allows_unresolvable_symbols(monkeypatch, cache_cfg):
    """A panel with symbols mygene.info has nothing for still loads under the species gate."""
    from spatialcpav25_gen.data import text as text_mod

    class PartiallyKnown:
        def querymany(self, qterms, **kwargs):
            return [
                {
                    "query": s,
                    "symbol": s,
                    "name": f"{s} name",
                    "taxid": TAXID_MOUSE,
                    "ensembl": {"gene": "ENSMUSG00000000001"},
                }
                for s in qterms
                if s != "Nothingness1"
            ]

    monkeypatch.setattr(text_mod, "load_mygene_client", lambda _cfg: PartiallyKnown())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.warns(GeneMetaUnavailableWarning, match="no metadata"):
        build_gene_meta(["Gad1", "Nothingness1"], online)

    meta = load_gene_meta(online.gene_meta_path, species="mouse")
    assert set(meta) == {"Gad1", "Nothingness1"}
    assert meta["Nothingness1"].full_name is None


def test_merge_refuses_to_mix_organisms(monkeypatch, cache_cfg):
    """Extending a mouse table with a human request raises rather than mixing them."""
    install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1"], online)
    with pytest.raises(GeneMetaError, match="One organism per table"):
        build_gene_meta(["GAD1"], online.replace(mygene_species="human"))


def test_committed_gene_meta_tables_are_species_checkable():
    """No table at ``Config.gene_meta_path`` may be usable without a recorded species.

    Guards the state B19 left the repository in: the one real table committed so far is human (28/28
    ``ENSG``) for a mouse panel, it is parked under a species-explicit name, and nothing points at
    it. If a future commit drops a table at the default path it has to carry the species columns or
    this fails — the point being that both real tables built so far were the wrong organism and both
    looked fine.
    """
    from spatialcpav25_gen.config import Config

    default = Path(Config().gene_meta_path)
    if default.exists():
        # Present is fine; unlabelled is not.
        import pandas as pd

        columns = set(pd.read_parquet(default).columns)
        assert {"species_requested", "species_resolved"} <= columns, (
            f"{default} has no species columns, so its organism cannot be checked (B19)"
        )
        load_gene_meta(default, species=Config().mygene_species)
        # And its summary coverage must be reportable by source, whatever its vintage: a table
        # written before the summary_source columns reads as all-native (B20).
        report = gene_meta_summary(default)
        sources = report["summary_sources"]
        assert sum(sources.values()) == report["rows"]
        assert sources["native"] + sources["ortholog"] == report["with_summary"]

    parked = Path("resources/gene_meta.human_orthologs.parquet")
    if parked.exists():
        with pytest.raises(GeneMetaError, match="predates the species columns"):
            load_gene_meta(parked, species="mouse")

    # The mouse build that is right about species and wrong about ensembl_id (B19a), kept as the
    # evidence for that finding. It must stay refused: `ensembl_id` is what downstream joins read.
    prefix_bug = Path("resources/gene_meta.mouse_prefix_bug.parquet")
    if prefix_bug.exists():
        with pytest.raises(GeneMetaError, match="not mouse ids"):
            load_gene_meta(prefix_bug, species="mouse")


# --------------------------------------------------------------------------------------
# the ensembl_id, which the taxid filter does not protect
# --------------------------------------------------------------------------------------
#
# A build whose taxid filter demonstrably worked — species_resolved uniformly 10090 — still wrote
# ENSMUSG 390, ENSMSIG 321 (ground squirrel), ENSNVIG 241 (mink), ENSMPUG 111 (ferret), ENSFALG 73
# (falcon), FBgn 1. Every other field of those rows came from the mouse hit, so the ids can only
# have come from a non-mouse element of the mouse hit's OWN `ensembl` list, taken by position.


class FakeMyGeneOrthologueIds:
    """One mouse hit whose ``ensembl`` field is a list of cross-species mappings.

    The reported structure: element zero is not the mouse id. ``Orphan1`` has no mouse id at all,
    which is the case that must become ``None`` rather than another organism's id.
    """

    def querymany(self, qterms, **kwargs):
        hits = []
        for symbol in qterms:
            ensembl = [
                {"gene": "ENSMSIG00000011111"},
                {"gene": "ENSNVIG00000022222"},
                {"gene": "ENSMUSG00000033333"},
            ]
            if symbol == "Orphan1":
                ensembl = [{"gene": "ENSFALG00000044444"}, {"gene": "FBgn0011111"}]
            hits.append(
                {
                    "query": symbol,
                    "symbol": symbol,
                    "name": f"{symbol} full name",
                    "summary": f"{symbol} does something",
                    "taxid": TAXID_MOUSE,
                    "ensembl": ensembl,
                    "_score": 60.0,
                }
            )
        return hits


def test_ensembl_id_is_chosen_by_species_prefix(monkeypatch, cache_cfg):
    """The stored id is the mouse one wherever the list puts it, and ``None`` when there is none."""
    install_fake_mygene(monkeypatch, FakeMyGeneOrthologueIds())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.warns(GeneMetaUnavailableWarning, match="none with the 'ENSMUSG' prefix"):
        table = build_gene_meta(["Gad1", "Orphan1"], online)

    row = {r["symbol"]: r for r in table.to_dict(orient="records")}
    assert row["Gad1"]["ensembl_id"] == "ENSMUSG00000033333", "not element zero, the mouse one"
    assert row["Orphan1"]["ensembl_id"] is None, "a foreign id must not be stored as the gene's own"
    # The taxid filter alone passes both of these, which is why the prefix is checked as well.
    assert list(table["species_resolved"]) == [str(TAXID_MOUSE)] * 2

    summary = gene_meta_summary(online.gene_meta_path)
    assert summary["ensembl_prefixes"] == {"ENSMUSG": 1, "None": 1}
    assert summary["expected_ensembl_prefix"] == ["ENSMUSG"]


def test_foreign_ensembl_prefix_is_refused(monkeypatch, cache_cfg):
    """A table holding another organism's ids raises, at build time and at load time."""
    from spatialcpav25_gen.data.text import _check_ensembl_prefix

    rows = [{"symbol": "Gad1", "ensembl_id": "ENSFALG00000000001"}]
    with pytest.raises(GeneMetaError, match="not mouse ids"):
        _check_ensembl_prefix(rows, "mouse", "ENSMUSG")

    install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1"], online)

    import pandas as pd

    table = pd.read_parquet(online.gene_meta_path)
    table["ensembl_id"] = "ENSMSIG00000099999"
    table.to_parquet(online.gene_meta_path, index=False)
    with pytest.raises(GeneMetaError, match="ENSMUSG"):
        load_gene_meta(online.gene_meta_path, species="mouse")


def test_species_resolved_is_read_from_the_hit(monkeypatch, cache_cfg):
    """``species_resolved`` comes from the response, so the column is evidence not an echo.

    It used to be written as ``str(taxid)`` — the *requested* value — which could never disagree
    with the request, so "species_resolved is uniformly 10090" was true by construction and read as
    evidence when it was not.
    """
    from spatialcpav25_gen.data import text as text_mod

    class MislabellingClient:
        """Returns a hit whose taxid passes the filter; the column must quote the hit."""

        def querymany(self, qterms, **kwargs):
            return [
                {
                    "query": s,
                    "symbol": s,
                    "name": "n",
                    "taxid": TAXID_MOUSE,
                    "ensembl": {"gene": "ENSMUSG00000000001"},
                }
                for s in qterms
            ]

    monkeypatch.setattr(text_mod, "load_mygene_client", lambda _cfg: MislabellingClient())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="10090")
    table = build_gene_meta(["Gad1"], online)
    assert list(table["species_requested"]) == ["10090"]
    assert list(table["species_resolved"]) == ["10090"]


def test_summary_lost_to_selection_is_reported(monkeypatch, cache_cfg):
    """When the chosen hit lacks a summary that a sibling hit has, say so.

    The diagnostic that separates "this organism's summaries are sparse" from "hit selection is
    dropping them" — the question raised when mouse coverage came back 148/1138 against the old
    human-leaning query's 1054/1138.
    """
    from spatialcpav25_gen.data import text as text_mod

    class TwoMouseHits:
        def querymany(self, qterms, **kwargs):
            hits = []
            for s in qterms:
                # the higher-scoring hit has no summary; a lower-scoring same-species one does
                hits.append(
                    {
                        "query": s,
                        "symbol": s,
                        "name": f"{s} a",
                        "taxid": TAXID_MOUSE,
                        "ensembl": {"gene": "ENSMUSG00000000001"},
                        "_score": 99.0,
                    }
                )
                hits.append(
                    {
                        "query": s,
                        "symbol": s,
                        "name": f"{s} b",
                        "summary": f"{s} is described here",
                        "taxid": TAXID_MOUSE,
                        "ensembl": {"gene": "ENSMUSG00000000002"},
                        "_score": 10.0,
                    }
                )
            return hits

    monkeypatch.setattr(text_mod, "load_mygene_client", lambda _cfg: TwoMouseHits())
    online = cache_cfg.replace(
        text_allow_network=True, mygene_species="mouse", gene_summary_fallback="none"
    )
    with pytest.warns(GeneMetaUnavailableWarning, match="selected hit has no summary"):
        table = build_gene_meta(["Gad1"], online)
    # Reported, not silently patched: choosing the sibling's summary would be merging two records.
    assert table["summary"].isna().all()


# --------------------------------------------------------------------------------------
# the orthologue summary fallback
# --------------------------------------------------------------------------------------
#
# Mouse NCBI summaries cover 148/1138 of a real panel and that sparsity is genuine (B20), so 87% of
# descriptors would be "{symbol}. {full_name}." — two names and no biology. The fallback borrows the
# human orthologue's summary, resolved through HomoloGene and labelled in the text.

TAXID_HUMAN = 9606


class FakeMyGeneOrthologSummaries:
    """Both calls :func:`_query_mygene` makes: mouse symbols, then human Entrez ids.

    Only ``Gad1`` has a summary of its own. ``Slc17a7`` has a 1:1 human orthologue that does;
    ``Rik2`` has two human orthologues (1:many, must be skipped rather than picked by score);
    ``Orphan3`` has no orthologue at all.
    """

    HOMOLOGENE: ClassVar[dict[str, dict]] = {
        "Gad1": {"id": 20, "genes": [[TAXID_MOUSE, 14415], [TAXID_HUMAN, 2571]]},
        "Slc17a7": {"id": 21, "genes": [[TAXID_MOUSE, 72961], [TAXID_HUMAN, 57030]]},
        "Rik2": {"id": 22, "genes": [[TAXID_MOUSE, 99], [TAXID_HUMAN, 401], [TAXID_HUMAN, 402]]},
    }
    HUMAN: ClassVar[dict[str, tuple[str, str]]] = {
        "57030": ("SLC17A7", "Mediates glutamate uptake into synaptic vesicles"),
        "2571": ("GAD1", "THIS MUST NOT BE USED: Gad1 has a summary of its own"),
        "401": ("ORTH1", "one of two paralogues"),
        "402": ("ORTH2", "the other of two paralogues"),
    }

    def __init__(self, *, human_taxid: int = TAXID_HUMAN) -> None:
        self.human_taxid = human_taxid
        self.calls: list[dict[str, object]] = []

    def querymany(self, qterms, **kwargs):
        self.calls.append({"qterms": list(qterms), **kwargs})
        if kwargs.get("scopes") == "entrezgene":
            return [
                {
                    "query": str(q),
                    "symbol": self.HUMAN[str(q)][0],
                    "summary": f"{self.HUMAN[str(q)][1]}.",
                    "taxid": self.human_taxid,
                    "_score": 50.0,
                }
                for q in qterms
                if str(q) in self.HUMAN
            ]
        return [
            {
                "query": s,
                "symbol": s,
                "name": f"{s} full name",
                "summary": "Catalyses the formation of GABA." if s == "Gad1" else None,
                "taxid": TAXID_MOUSE,
                "ensembl": {"gene": "ENSMUSG00000000001"},
                "homologene": self.HOMOLOGENE.get(s),
                "_score": 70.0,
            }
            for s in qterms
        ]


def test_ortholog_summary_is_borrowed_and_labelled(monkeypatch, cache_cfg):
    """A summary-less mouse gene takes its 1:1 human orthologue's summary, marked as borrowed.

    The four cases in one build: native kept, orthologue borrowed, 1:many skipped, no orthologue
    left bare. The label is in the descriptor text as well as the column, because the frozen
    encoder only ever sees the text.
    """
    panel = ["Gad1", "Slc17a7", "Rik2", "Orphan3"]
    client = install_fake_mygene(monkeypatch, FakeMyGeneOrthologSummaries())
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.warns(GeneMetaUnavailableWarning, match="borrowed a human orthologue's summary"):
        build_gene_meta(panel, online)
    meta = load_gene_meta(online.gene_meta_path, species="mouse")

    assert meta["Gad1"].summary_source == "native", "a native summary is never overwritten"
    assert meta["Gad1"].summary == "Catalyses the formation of GABA"
    assert meta["Gad1"].summary_source_taxid == str(TAXID_MOUSE)
    assert meta["Gad1"].summary_source_symbol is None

    assert meta["Slc17a7"].summary_source == "ortholog"
    assert meta["Slc17a7"].summary == "Mediates glutamate uptake into synaptic vesicles"
    assert meta["Slc17a7"].summary_source_taxid == str(TAXID_HUMAN)
    assert meta["Slc17a7"].summary_source_symbol == "SLC17A7"

    for bare in ("Rik2", "Orphan3"):
        assert meta[bare].summary is None, f"{bare} must not borrow a summary"
        assert meta[bare].summary_source is None

    # The label, in the text the encoder embeds. The mouse symbol stays the panel's spelling.
    assert gene_descriptor("Slc17a7", meta["Slc17a7"]) == (
        "Slc17a7. Slc17a7 full name. Human orthologue SLC17A7: Mediates glutamate uptake into "
        "synaptic vesicles."
    )
    assert "orthologue" not in gene_descriptor("Gad1", meta["Gad1"])

    # Only the summary-less genes' orthologues are looked up, and only in the target species.
    second = client.calls[1]
    assert second["scopes"] == "entrezgene"
    assert second["species"] == "human"
    # Not Gad1's 2571 (it has its own summary), and not Rik2's 401/402 (1:many, skipped before
    # the query rather than fetched and then discarded).
    assert sorted(second["qterms"]) == ["57030"]

    summary = gene_meta_summary(online.gene_meta_path)
    assert summary["summary_sources"] == {"native": 1, "ortholog": 1, "none": 2}
    assert summary["with_summary"] == 2, "native + ortholog, which is why the split is reported"


def test_ortholog_fallback_off_leaves_the_summary_absent(monkeypatch, cache_cfg):
    """``gene_summary_fallback="none"`` makes no second query and borrows nothing."""
    client = install_fake_mygene(monkeypatch, FakeMyGeneOrthologSummaries())
    online = cache_cfg.replace(
        text_allow_network=True, mygene_species="mouse", gene_summary_fallback="none"
    )
    build_gene_meta(["Gad1", "Slc17a7"], online)
    meta = load_gene_meta(online.gene_meta_path, species="mouse")
    assert meta["Slc17a7"].summary is None
    assert meta["Slc17a7"].summary_source is None
    assert len(client.calls) == 1, "the orthologue lookup must not be issued at all"
    assert gene_meta_summary(online.gene_meta_path)["summary_sources"] == {
        "native": 1,
        "ortholog": 0,
        "none": 1,
    }


def test_ortholog_summary_from_another_species_is_dropped(monkeypatch, cache_cfg):
    """The orthologue query's taxid is verified too, or the label would name the wrong species."""
    install_fake_mygene(monkeypatch, FakeMyGeneOrthologSummaries(human_taxid=TAXID_FALCON))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    with pytest.warns(GeneMetaUnavailableWarning, match="came back from another species"):
        build_gene_meta(["Slc17a7"], online)
    meta = load_gene_meta(online.gene_meta_path, species="mouse")
    assert meta["Slc17a7"].summary is None
    assert meta["Slc17a7"].summary_source is None


def test_homologene_requires_a_one_to_one_orthologue():
    """The 1:many case returns both ids, so the caller can skip it rather than guess."""
    from spatialcpav25_gen.data.text import _homologene_gene_ids

    group = FakeMyGeneOrthologSummaries.HOMOLOGENE
    assert _homologene_gene_ids(group["Slc17a7"], TAXID_HUMAN) == ["57030"]
    assert _homologene_gene_ids(group["Rik2"], TAXID_HUMAN) == ["401", "402"]
    assert _homologene_gene_ids(group["Slc17a7"], TAXID_MOUSE) == ["72961"]
    for absent in (None, {}, {"id": 3}, {"genes": "nonsense"}, {"genes": [[1], "x"]}):
        assert _homologene_gene_ids(absent, TAXID_HUMAN) == []


def test_table_written_before_the_summary_source_columns_reads_as_native(cache_cfg, monkeypatch):
    """An older table's summaries are back-filled as ``native`` — the only thing they can be.

    Unlike the species columns, where the missing value is exactly what is unknown and the table is
    refused, there was no other place a summary could have come from before this fallback existed.
    """
    import pandas as pd

    install_fake_mygene(monkeypatch, FakeMyGene(mouse_only=True))
    online = cache_cfg.replace(text_allow_network=True, mygene_species="mouse")
    build_gene_meta(["Gad1", "Nothing2"], online)

    table = pd.read_parquet(online.gene_meta_path)
    table.loc[table["symbol"] == "Nothing2", "summary"] = None
    table = table.drop(columns=["summary_source", "summary_source_taxid", "summary_source_symbol"])
    table.to_parquet(online.gene_meta_path, index=False)

    meta = load_gene_meta(online.gene_meta_path, species="mouse")
    assert meta["Gad1"].summary_source == "native"
    assert meta["Gad1"].summary_source_taxid == str(TAXID_MOUSE)
    assert meta["Nothing2"].summary_source is None
    assert gene_meta_summary(online.gene_meta_path)["summary_sources"] == {
        "native": 1,
        "ortholog": 0,
        "none": 1,
    }


# --------------------------------------------------------------------------------------
# the panel's embeddings: descriptors -> encoder -> EntityEmbeddings
# --------------------------------------------------------------------------------------


def test_build_entity_embeddings_encodes_the_table_not_the_bare_symbol(monkeypatch, cache_cfg):
    """The defect this builder exists to remove.

    Every real-data caller had been passing either zeros or ``gene_descriptor(symbol, None)``,
    so ``text_emb_mode="medcpt"`` encoded ``"Slc17a7."`` — MedCPT applied to a token, with the
    panel's own full names and NCBI summaries sitting unread in
    ``Config.gene_meta_path``. The assertion is therefore not "it returns something": it is
    that the vector differs from the bare-symbol one, which is the only thing that
    distinguishes a live text channel from a dead one.
    """
    with pytest.warns(GeneMetaUnavailableWarning):
        build_gene_meta(["Gad1", "Slc17a7"], cache_cfg)
    install_fake_backend(monkeypatch, cache_cfg)

    emb = build_entity_embeddings(cache_cfg, ["Gad1", "Slc17a7"], ["Excitatory", "Inhibitory"])
    assert emb.gene.text_vecs.shape == (2, cache_cfg.text_dim_in)
    assert emb.celltype.text_vecs.shape == (2, cache_cfg.text_dim_in)
    assert emb.region is None

    meta = load_gene_meta(cache_cfg.gene_meta_path)
    bare = TextEncoder(cache_cfg).encode(["Gad1.", "Slc17a7."])
    described = TextEncoder(cache_cfg).encode(
        [gene_descriptor(s, meta.get(s)) for s in ("Gad1", "Slc17a7")]
    )
    assert np.allclose(emb.gene.text_vecs.numpy(), described)
    # Offline, build_gene_meta writes symbol-only rows, so the two coincide here. The point
    # of the test is that the builder reads the table at all; on a real table they differ,
    # which the next assertion pins with a table that carries a summary.
    assert np.allclose(bare, described)


def test_build_entity_embeddings_uses_a_summary_when_the_table_has_one(monkeypatch, cache_cfg):
    """With a real (non-degraded) row, the encoded descriptor is not the bare symbol."""
    from spatialcpav25_gen.data.text import _write_gene_meta_table

    table = pd.DataFrame(
        [
            {
                "symbol": "Gad1",
                "full_name": "glutamate decarboxylase 1",
                "summary": "Catalyses the production of GABA from L-glutamic acid.",
                "summary_source": "native",
                "aliases": "",
                "ensembl_id": "ENSMUSG00000070880",
                "species_requested": "mouse",
                "species_resolved": "10090",
            }
        ]
    )
    path = Path(cache_cfg.gene_meta_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_gene_meta_table(table, path)
    install_fake_backend(monkeypatch, cache_cfg)

    described = describe_entity_descriptors(cache_cfg, ["Gad1"])
    assert described == {
        "n_genes": 1,
        "n_with_meta": 1,
        "n_bare": 0,
        "n_with_summary": 1,
        "example": "Gad1. glutamate decarboxylase 1. Catalyses the production of GABA from "
        "L-glutamic acid.",
    }

    emb = build_entity_embeddings(cache_cfg, ["Gad1"], ["Excitatory"])
    bare = TextEncoder(cache_cfg).encode(["Gad1."])
    assert not np.allclose(emb.gene.text_vecs.numpy(), bare), (
        "the summary never reached the encoder: this is the dead-channel state"
    )


def test_both_text_modes_get_the_same_vectors(monkeypatch, cache_cfg):
    """A3 is a comparison only while ``text_emb_mode`` is the single thing that differs.

    ``"lookup"`` is applied inside ``TextGroundedEmbedding._text_channel``, which zeroes the
    projection on the seen and the zero-shot path alike. Withholding the vectors as well —
    which every prior caller did, by handing the lookup arm a zero matrix — would make the two
    arms differ in two things at once, and would change ``distillation_loss``, which reads
    ``text_vecs`` directly.
    """
    with pytest.warns(GeneMetaUnavailableWarning):
        build_gene_meta(["Gad1", "Slc17a7"], cache_cfg)
    install_fake_backend(monkeypatch, cache_cfg)

    medcpt = build_entity_embeddings(
        cache_cfg.replace(text_emb_mode="medcpt"), ["Gad1", "Slc17a7"], ["Excitatory"]
    )
    lookup = build_entity_embeddings(
        cache_cfg.replace(text_emb_mode="lookup"), ["Gad1", "Slc17a7"], ["Excitatory"]
    )
    assert torch.equal(medcpt.gene.text_vecs, lookup.gene.text_vecs)
    assert torch.equal(medcpt.celltype.text_vecs, lookup.celltype.text_vecs)

    # ...and the gate still does its job: the text channel is gone on the lookup arm.
    idx = torch.arange(2)
    assert not torch.equal(medcpt.gene(idx), lookup.gene(idx))


def test_build_entity_embeddings_refuses_an_empty_panel(cache_cfg):
    with pytest.raises(ValueError, match="gene_names is empty"):
        build_entity_embeddings(cache_cfg, [], ["Excitatory"])
    with pytest.raises(ValueError, match="celltype_names is empty"):
        build_entity_embeddings(cache_cfg, ["Gad1"], [])


def test_build_entity_embeddings_refuses_a_missing_table(cache_cfg):
    """No table is a refusal, not a whole panel of bare symbols nobody notices."""
    with pytest.raises(FileNotFoundError, match="gene_meta_path"):
        build_entity_embeddings(cache_cfg, ["Gad1"], ["Excitatory"])


# --------------------------------------------------------------------------------------
# the zero-shot view (T09's four-arm experiment)
# --------------------------------------------------------------------------------------


def test_zero_shot_view_is_a_no_op_on_seen_entities(cfg):
    """Rows outside ``unseen`` come back bitwise identical to plain ``forward``.

    The view is the apparatus the paper's zero-shot table is computed with, and its whole
    claim is that it changes the held-out genes and nothing else. Asserted on both settings
    of ``use_distill``, because a view that quietly moved the seen genes would make the arms
    incomparable to every other number measured on this fit.
    """
    emb = _embedding(cfg, n=32)
    emb.set_progress(1.0)
    with torch.no_grad():
        emb.r.weight.normal_(generator=torch.Generator().manual_seed(3))
    unseen = torch.tensor([2, 5, 11], dtype=torch.int64)
    seen = torch.tensor([i for i in range(32) if i not in {2, 5, 11}], dtype=torch.int64)

    baseline = emb(seen)
    for use_distill in (True, False):
        view = ZeroShotView(emb, unseen, use_distill=use_distill)
        assert torch.equal(view(seen), baseline)


def test_zero_shot_view_without_distill_agrees_with_plain_generation(cfg):
    """The pure-text arms are plain generation — to 1e-5, and **not** bitwise.

    An entity whose residual is still its zeros init has ``forward`` equal to
    ``forward_zero_shot(t, use_distill=False)`` *algebraically*, which is exactly the state a
    fit given ``gene_pool`` leaves the held-out genes in. So A2 and A4 need no view, and if the
    view were wrong the two would disagree — the cheapest available check on the apparatus.

    They do not agree bitwise, and the reason matters for how the arms are scored. The view
    runs ``W`` over the unseen rows alone while ``forward`` runs it over the whole query, and a
    float32 GEMM at two batch sizes rounds differently: 3e-8 on ``W t``, amplified to **1.3e-6**
    by the ``LayerNorm``. The seen rows, which both paths compute the same way, are bitwise
    identical (asserted above). The consequence is a rule for the scorer, not a defect here:
    **every arm is computed through the view**, so the only difference between two arms is
    ``use_distill`` and the fit — not which code path produced the embedding.
    """
    emb = _embedding(cfg, n=16)
    emb.set_progress(1.0)
    unseen = torch.tensor([1, 4, 9], dtype=torch.int64)
    with torch.no_grad():
        # Every *seen* residual trained; the unseen ones never received a gradient.
        emb.r.weight.normal_(generator=torch.Generator().manual_seed(7))
        emb.r.weight[unseen] = 0.0
    idx = torch.arange(16, dtype=torch.int64)

    plain = emb(idx)
    pure_text = ZeroShotView(emb, unseen, use_distill=False)(idx)
    assert torch.allclose(pure_text, plain, atol=1e-5, rtol=0.0), (
        f"max |delta| {(pure_text - plain).abs().max().item():.3g}"
    )
    seen = torch.tensor([i for i in range(16) if i not in {1, 4, 9}], dtype=torch.int64)
    assert torch.equal(pure_text[seen], plain[seen])
    # ...and with the distillation head it must *not* be a no-op, or the arm is not an arm.
    distilled = ZeroShotView(emb, unseen, use_distill=True)(idx)
    assert float((distilled[unseen] - plain[unseen]).abs().max()) > 1e-3


def test_zero_shot_view_under_lookup_is_one_vector_for_every_gene(cfg):
    """A4 is degenerate by construction, which is what makes it the leak detector.

    Under ``lookup`` the text channel is zeroed and a never-trained residual is zero, so every
    held-out gene gets ``norm(0)`` — the *same* vector. An arm that cannot tell two genes apart
    cannot score above the constant-field referent unless something else is feeding it the
    answer, so a score above the band is a leak rather than a result.
    """
    emb = _embedding(cfg.replace(text_emb_mode="lookup"), n=12)
    emb.set_progress(1.0)
    unseen = torch.tensor([0, 3, 7], dtype=torch.int64)
    rows = ZeroShotView(emb, unseen, use_distill=False)(unseen)
    assert torch.equal(rows[0], rows[1])
    assert torch.equal(rows[1], rows[2])


def test_zero_shot_view_refuses_an_index_off_the_table(cfg):
    emb = _embedding(cfg, n=8)
    with pytest.raises(ValueError, match="outside the 8-entity table"):
        ZeroShotView(emb, torch.tensor([8], dtype=torch.int64))
