"""T02 acceptance tests: descriptors, the cached frozen encoder, and the learned embedding.

Every numeric threshold here comes from `specs/02_TASK_text_embeddings.md`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.text import (
    GENE_META_COLUMNS,
    GeneMeta,
    GeneMetaUnavailableWarning,
    TextEncoder,
    build_gene_meta,
    celltype_descriptor,
    descriptor_key,
    gene_descriptor,
    load_gene_meta,
    region_descriptor,
)
from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    TextGroundedEmbedding,
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


def test_gene_meta_table_is_reused(cache_cfg):
    """A cached row is reused rather than re-derived, and the table keeps other symbols."""
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
