"""The real-data runtime the T09 STARmap drivers share: volume, config, embeddings.

Three things every driver needs and that had been written five different ways across
``scripts/`` and the bench3 wrapper — with the differences silently changing what was
measured:

* **the training volume**, loaded with bench3's own obs keys and with ``flattened_sections``
  taken from the built file rather than guessed. ``t10_chain_diagnostic`` hard-codes it True
  and the wrapper defaults it False; on the tier-1 build the answer is True (143 of 28 978
  cells are exactly coincident, ``reports/pilot.md`` §4) and the two spellings disagree.
* **the base config**, carrying those keys, so a config persisted by a selection run is
  loadable by the thing that fits under it.
* **the entity embeddings**, from ``model.build_entity_embeddings`` — i.e. from the panel's
  own gene-metadata table. Every prior real-data caller passed either zeros or
  ``gene_descriptor(symbol, None)``, so ``text_emb_mode="medcpt"`` encoded the bare string
  ``"Slc17a7."`` and every STARmap number measured so far is ablation A3's ``lookup`` arm in
  all but name. That is the defect this module exists to remove.

Nothing here is STARmap-specific except the module name; it works for any bench3 dataset.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import TrainingVolume
from spatialcpav25_gen.model.embeddings import (
    EntityEmbeddings,
    build_entity_embeddings,
    describe_entity_descriptors,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

# bench3's obs keys. The wrapper sets these on its own local config; a config *persisted* by a
# selection run has to carry them, or the next process loads the volume under Config's defaults
# and reads a column that is not there.
BENCH3_KEYS: dict[str, object] = {
    "section_key": "section",
    "coord_key": "spatial",
    "celltype_key": "cell_type",
    "region_key": None,
}


def base_config(seed: int, **overrides: object) -> Config:
    """The base ``Config`` a real bench3 dataset is fitted under."""
    return Config(seed=int(seed), **BENCH3_KEYS, **overrides)  # type: ignore[arg-type]


def flattened_from_input(input_path: str | Path) -> bool:
    """Read ``uns['paper_protocol']['flattened_z']`` — and refuse to guess it.

    Whether sections are flattened decides whether exactly coincident cells are legal
    (``Volume.flattened_sections``, ``reports/pilot.md`` §4 option C). Defaulting it False on
    a flattened build aborts the fit in ``load_volume``; defaulting it True on an unflattened
    one disables a real invariant. Neither is a default anyone should get by accident, so a
    file that does not say raises and names the key (Convention 6).
    """
    import anndata as ad

    adata = ad.read_h5ad(Path(input_path), backed="r")
    try:
        protocol = dict(adata.uns.get("paper_protocol") or {})
    finally:
        adata.file.close()
    if "flattened_z" not in protocol:
        raise SystemExit(
            f"{input_path} carries no uns['paper_protocol']['flattened_z']. That flag decides "
            "whether exactly coincident cells are legal in this volume (reports/pilot.md §4), "
            "and it is not guessable from the coordinates. Rebuild the input with a current "
            "bench3, or pass --flattened / --no-flattened to say so explicitly."
        )
    return bool(protocol["flattened_z"])


def clamp_config_to_input(cfg: Config, input_path: str | Path) -> Config:
    """Apply ``specs/10`` §0's clamp from the input file's header, before the volume is built.

    ``clamp_config_to_volume`` needs a ``Volume``, and building one runs
    ``validate_config_against_volume``, which is the very check the clamp exists to satisfy —
    so on a narrow panel the volume can never be built to clamp against. The two numbers the
    clamp reads (`n_genes`, `n_cells`) are in the file's header, so they are read there
    instead. The volume-level clamp inside ``run_selection`` then finds nothing left to do.

    On tier-1 STARmap this is what makes the dataset fittable at all: the panel is 28 genes
    wide against ``Config.expr_pca_dim = 32``.
    """
    import anndata as ad
    from spatialcpav25_gen.data.schema import ConfigClampWarning

    adata = ad.read_h5ad(Path(input_path), backed="r")
    try:
        n_genes, n_cells = int(adata.n_vars), int(adata.n_obs)
    finally:
        adata.file.close()

    changes: dict[str, int] = {}
    if cfg.expr_pca_dim > n_genes:
        changes["expr_pca_dim"] = n_genes
    if cfg.retrieval_k > n_cells:
        changes["retrieval_k"] = n_cells
    if not changes:
        return cfg
    for name, value in changes.items():
        warnings.warn(
            f"clamp_config_to_input: Config.{name}={getattr(cfg, name)} exceeds what "
            f"{Path(input_path).name} supports ({n_genes} genes, {n_cells} cells); narrowed "
            f"to {value}. specs/10 §0's owed fix. This changes the config's content hash.",
            ConfigClampWarning,
            stacklevel=2,
        )
    return cfg.replace(**changes)


def load_training_volume(
    cfg: Config, input_path: str | Path, *, flattened: bool | None = None
) -> TrainingVolume:
    """bench3's leakage-guarded training-only input, as a ``TrainingVolume``.

    Everything in the file is training data — ``run_benchmark`` removed the held-out sections
    before any wrapper saw it — so the whole volume is the training volume and the type
    carries that guarantee to every calibrator and to the selector.

    The round trip through a temporary file goes to a **per-process** directory: several cells
    of a selection are meant to run concurrently, and a shared temp name beside the input would
    have them overwrite and unlink each other's copy.
    """
    import anndata as ad
    from spatialcpav25_gen.data.loaders import load_volume

    src = Path(input_path)
    is_flat = flattened_from_input(src) if flattened is None else bool(flattened)
    adata = ad.read_h5ad(src)
    with tempfile.TemporaryDirectory(prefix="ctfflow_input_") as tmpdir:
        tmp = Path(tmpdir) / "train_registered.h5ad"
        adata.write_h5ad(tmp)
        vol = load_volume(tmp, cfg, flattened_sections=is_flat)
    return TrainingVolume(
        specimen_id=vol.specimen_id,
        sections=vol.sections,
        gene_names=vol.gene_names,
        celltype_names=vol.celltype_names,
        region_names=vol.region_names,
        flattened_sections=vol.flattened_sections,
    )


def embeddings_factory(volume: TrainingVolume) -> Callable[[Config], EntityEmbeddings]:
    """A ``Config -> EntityEmbeddings`` factory over this panel, for ``FitScorer``.

    A factory rather than an instance because the embeddings carry **learned** parameters and
    every selection candidate is a fresh fit; reusing one object would let the first
    candidate's training leak into the rest.

    Both arms of the ``text_emb_mode`` gate get the same MedCPT vectors — ``"lookup"`` is
    applied inside ``TextGroundedEmbedding._text_channel``, and the gate is only a comparison
    while it is the one thing that differs.
    """

    def build(cfg: Config) -> EntityEmbeddings:
        return build_entity_embeddings(cfg, volume.gene_names, volume.celltype_names, None)

    return build


def describe_text_channel(cfg: Config, volume: TrainingVolume) -> dict[str, object]:
    """What the text channel is being given, for the run's provenance line.

    ``n_bare == n_genes`` under ``text_emb_mode="medcpt"`` means the gate is on and there is
    nothing behind it — which is the state every prior STARmap measurement was in.
    """
    out = dict(describe_entity_descriptors(cfg, volume.gene_names))
    out["text_emb_mode"] = cfg.text_emb_mode
    out["text_model"] = cfg.text_model
    out["gene_meta_path"] = cfg.gene_meta_path
    return out


def preflight_text_encoder(cfg: Config, volume: TrainingVolume) -> dict[str, object]:
    """Encode the panel once and report the cache state. The check to run before a campaign.

    Costs one model load on a cold cache and nothing on a warm one, and it fails *here* —
    in seconds — rather than nine hours into a selection, if the MedCPT weights are not
    reachable on this machine.
    """
    from spatialcpav25_gen.data.text import TextEncoder

    encoder = TextEncoder(cfg)
    emb = build_entity_embeddings(
        cfg, volume.gene_names, volume.celltype_names, None, encoder=encoder
    )
    described = describe_text_channel(cfg, volume)
    described["n_model_batches"] = encoder.n_model_calls
    described["cache_dir"] = str(cfg.text_cache_dir)
    described["gene_text_vecs"] = tuple(emb.gene.text_vecs.shape)
    described["celltype_text_vecs"] = tuple(emb.celltype.text_vecs.shape)
    return described
