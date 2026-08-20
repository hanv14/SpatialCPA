"""T01 acceptance tests for the data contracts: schema, loaders, and the fixture.

Numeric thresholds here are the ones written in ``specs/01_TASK_config_and_data.md``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse
from spatialcpav25_gen.config import ConfigError
from spatialcpav25_gen.data.loaders import load_volume, loso_folds, split_holdout
from spatialcpav25_gen.data.schema import (
    AssumedThicknessWarning,
    CoincidentCoordsWarning,
    HeldOutSections,
    NonIntegerCountsWarning,
    OverlappingSlabsWarning,
    SchemaError,
    Section,
    TrainingVolume,
    Volume,
    to_xyz,
    validate_volume,
)

from tests.conftest import copy_section, knn_weights, morans_i, rebuild_volume
from tests.fixtures.synthetic import make_synthetic_volume, volume_to_anndata

# --- thresholds from the spec's "Acceptance tests" section ------------------------------
MIN_MEAN_MORANS_I = 0.25
DETECTION_SPAN = (0.05, 0.95)
MIN_Z_GRADIENT_GENES = 15
MIN_Z_GRADIENT_CORR = 0.5
MIN_PURITY_SIGMA = 3.0

# --- parameters of the test statistics themselves (not model constants) -----------------
N_PURITY_PERMUTATIONS = 40
PERMUTATION_SEED = 12345
# A deliberately small volume for tests that need to build several of their own.
SMALL = {"n_sections": 3, "n_cells_per_section": 200, "n_genes": 40}


# ========================================================================================
# schema
# ========================================================================================


def test_to_xyz_shape_and_dtype(volume):
    """to_xyz is the only sanctioned (N, 2) + z -> (N, 3) conversion."""
    section = volume.sections[2]
    xyz = to_xyz(section)
    assert xyz.shape == (section.n_cells, 3)
    assert xyz.dtype == np.float32
    np.testing.assert_array_equal(xyz[:, :2], section.coords)
    assert np.all(xyz[:, 2] == np.float32(section.z))


def test_volume_derived_fields(volume):
    """median_spacing / median_nn_dist / bbox are derived, not passed in."""
    assert volume.median_spacing == pytest.approx(50.0)
    assert 5.0 < volume.median_nn_dist < 30.0
    assert volume.bbox.shape == (2, 3)
    assert np.all(volume.bbox[0] <= volume.bbox[1])
    assert volume.bbox[0, 2] == pytest.approx(0.0)
    assert volume.bbox[1, 2] == pytest.approx(400.0)
    with pytest.raises(TypeError):
        Volume(  # derived fields are init=False and cannot go stale
            sections=volume.sections,
            gene_names=volume.gene_names,
            celltype_names=volume.celltype_names,
            region_names=volume.region_names,
            specimen_id="x",
            median_spacing=1.0,
        )


def test_valid_volume_passes(volume, cfg):
    """The fixture itself satisfies the contract, with no warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_volume(volume, cfg)


def test_schema_rejects_lognormalised(volume, cfg):
    """log1p data behind counts_layer warns loudly; raw counts do not."""
    logged = []
    for section in volume.sections:
        counts = section.counts.astype(np.float32).copy()
        counts.data = np.log1p(counts.data)
        logged.append(copy_section(section, counts=counts))
    vol = rebuild_volume(volume, logged)

    counts_cfg = cfg.replace(counts_layer="counts")
    with pytest.warns(NonIntegerCountsWarning, match="ZINB"):
        validate_volume(vol, counts_cfg)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        validate_volume(volume, counts_cfg)


def test_schema_rejects_unsorted_or_duplicate(volume, cfg):
    """Unsorted z and duplicate coordinates both raise, naming the field."""
    swapped = list(volume.sections)
    swapped[1], swapped[2] = swapped[2], swapped[1]
    with pytest.raises(SchemaError, match=r"Section\.z"):
        validate_volume(rebuild_volume(volume, swapped), cfg)

    equal_z = [copy_section(volume.sections[0], z=0.0, section_id="a"), *volume.sections[1:]]
    equal_z[1] = copy_section(equal_z[1], z=0.0, section_id="b")
    with pytest.raises(SchemaError, match="strictly increasing"):
        validate_volume(rebuild_volume(volume, equal_z), cfg)

    duped_coords = volume.sections[0].coords.copy()
    duped_coords[3] = duped_coords[0]
    duped = [copy_section(volume.sections[0], coords=duped_coords), *volume.sections[1:]]
    with pytest.raises(SchemaError, match=r"Section\.coords.*duplicate"):
        validate_volume(rebuild_volume(volume, duped), cfg)


def test_duplicate_coords_are_permitted_only_for_flattened_sections(volume, cfg):
    """The exemption is narrow: only a volume flagged as z-flattened slabs may tie.

    bench3 collapses each multi-plane slab to its centre z, so two cells at the same
    ``(x, y)`` in different planes become exactly coincident — 0.49% of the STARmap protocol
    build. That is real geometry, not corruption, so ``flattened_sections`` permits it. Every
    other volume keeps the hard check, because for a true plane a coincident pair is a bug and
    it silently corrupts every kNN-graph metric downstream.
    """
    duped_coords = volume.sections[0].coords.copy()
    duped_coords[3] = duped_coords[0]
    duped_coords[7] = duped_coords[1]
    sections = [copy_section(volume.sections[0], coords=duped_coords), *volume.sections[1:]]

    # UNflattened: still rejected. This is the half that must not regress.
    strict = rebuild_volume(volume, sections, flattened_sections=False)
    assert strict.n_coincident_coords == 2
    with pytest.raises(SchemaError, match=r"Section\.coords.*duplicate"):
        validate_volume(strict, cfg)

    # Flattened: permitted, counted, and warned about exactly once with the count.
    flat = rebuild_volume(volume, sections, flattened_sections=True)
    assert flat.n_coincident_coords == 2
    with pytest.warns(CoincidentCoordsWarning, match="2 cell"):
        validate_volume(flat, cfg)

    # A flattened volume with no ties is silent — the warning reports a fact, not a mode.
    clean = rebuild_volume(volume, list(volume.sections), flattened_sections=True)
    assert clean.n_coincident_coords == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", CoincidentCoordsWarning)
        validate_volume(clean, cfg)


def test_flattened_flag_propagates_to_every_derived_volume(volume, cfg):
    """A split or a LOSO fold of a flattened volume is still flattened.

    Otherwise the exemption would hold for the loaded volume and vanish the moment
    ``split_holdout`` produced the ``TrainingVolume`` the model actually trains on.
    """
    flat = rebuild_volume(volume, list(volume.sections), flattened_sections=True)
    training, _held = split_holdout(flat, "alternating", 0, cfg)
    assert training.flattened_sections
    assert all(fold.flattened_sections for fold, _section in loso_folds(training))


def test_schema_rejects_bad_fields(volume, cfg):
    """Each remaining contract violation raises with the field named."""
    sections = list(volume.sections)

    negative = sections[0].counts.copy().astype(np.float32)
    negative.data[0] = -1.0
    with pytest.raises(SchemaError, match=r"Section\.counts.*negative"):
        validate_volume(
            rebuild_volume(volume, [copy_section(sections[0], counts=negative), *sections[1:]]),
            cfg,
        )

    bad_codes = sections[0].cell_type.copy()
    bad_codes[0] = len(volume.celltype_names)
    with pytest.raises(SchemaError, match=r"Section\.cell_type"):
        validate_volume(
            rebuild_volume(volume, [copy_section(sections[0], cell_type=bad_codes), *sections[1:]]),
            cfg,
        )

    with pytest.raises(SchemaError, match="min_cells_per_section"):
        validate_volume(
            rebuild_volume(
                volume,
                [
                    copy_section(
                        sections[0],
                        coords=sections[0].coords[:10],
                        cell_type=sections[0].cell_type[:10],
                        region=sections[0].region[:10],
                        counts=sections[0].counts[:10],
                    ),
                    *sections[1:],
                ],
            ),
            cfg,
        )

    with pytest.raises(SchemaError, match=r"Section\.thickness"):
        validate_volume(
            rebuild_volume(volume, [copy_section(sections[0], thickness=0.0), *sections[1:]]), cfg
        )

    with pytest.raises(SchemaError, match="min_sections_per_volume"):
        validate_volume(rebuild_volume(volume, sections[:2]), cfg)

    float64 = sections[0].coords.astype(np.float64)
    with pytest.raises(SchemaError, match="float32"):
        validate_volume(
            rebuild_volume(volume, [copy_section(sections[0], coords=float64), *sections[1:]]), cfg
        )


def test_schema_warns_on_overlapping_slabs(volume, cfg):
    """Slabs thicker than the spacing make L_thick ill-posed, so they warn."""
    thick = [copy_section(s, thickness=volume.median_spacing * 1.5) for s in volume.sections]
    with pytest.warns(OverlappingSlabsWarning, match="overlap"):
        validate_volume(rebuild_volume(volume, thick), cfg)


def test_config_against_volume_rejects_z_bands(volume, cfg):
    """The data-dependent half of config validation: z bands vs. section count."""
    from spatialcpav25_gen.data.schema import validate_config_against_volume

    validate_config_against_volume(cfg, volume)
    validate_config_against_volume(cfg.replace(fourier_bands_z=6), volume)  # 9 sections is enough

    small = rebuild_volume(volume, list(volume.sections[:4]))
    validate_config_against_volume(cfg, small)
    with pytest.raises(ConfigError, match="fourier_bands_z"):
        validate_config_against_volume(cfg.replace(fourier_bands_z=6), small)
    with pytest.raises(ConfigError, match="expr_pca_dim"):
        validate_config_against_volume(cfg.replace(expr_pca_dim=volume.n_genes + 1), volume)


# ========================================================================================
# the synthetic fixture
# ========================================================================================


def test_synthetic_has_structure(volume, cfg):
    """The fixture carries the structure later tasks are asked to recover."""
    middle = volume.sections[volume.n_sections // 2]
    dense = np.asarray(middle.counts.todense(), dtype=np.float64)
    idx, weights = knn_weights(middle.coords.astype(np.float64), cfg.metric_knn_k)

    moran = morans_i(np.log1p(dense), idx, weights)
    finite = moran[np.isfinite(moran)]
    assert finite.size >= 0.95 * volume.n_genes
    assert finite.mean() > MIN_MEAN_MORANS_I, f"mean Moran's I = {finite.mean():.3f}"

    all_counts = sparse.vstack([s.counts for s in volume.sections]).tocsr()
    detection = np.asarray((all_counts > 0).mean(axis=0)).ravel()
    assert detection.min() < DETECTION_SPAN[0], f"min detection = {detection.min():.3f}"
    assert detection.max() > DETECTION_SPAN[1], f"max detection = {detection.max():.3f}"

    per_section_mean = np.stack(
        [np.asarray(s.counts.mean(axis=0)).ravel() for s in volume.sections]
    )
    z = volume.z_values.astype(np.float64)
    zc = z - z.mean()
    xc = per_section_mean - per_section_mean.mean(axis=0, keepdims=True)
    denom = np.sqrt((zc**2).sum() * (xc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, (zc[:, None] * xc).sum(axis=0) / denom, 0.0)
    n_gradient = int((np.abs(corr) > MIN_Z_GRADIENT_CORR).sum())
    assert n_gradient >= MIN_Z_GRADIENT_GENES, f"{n_gradient} genes with |corr(mean, z)| > 0.5"

    purity, sigma = _type_purity_sigma(middle, idx, len(volume.celltype_names))
    assert sigma >= MIN_PURITY_SIGMA, (
        f"type purity {purity:.3f} is only {sigma:.1f} sigma above chance"
    )


def _type_purity_sigma(section: Section, idx: np.ndarray, n_types: int) -> tuple[float, float]:
    """Neighbourhood type purity and how many sigma above the label-permuted null it is."""
    labels = section.cell_type
    purity = float((labels[idx] == labels[:, None]).mean())
    gen = np.random.default_rng(PERMUTATION_SEED)
    null = np.empty(N_PURITY_PERMUTATIONS)
    for i in range(N_PURITY_PERMUTATIONS):
        shuffled = gen.permutation(labels)
        null[i] = float((shuffled[idx] == shuffled[:, None]).mean())
    return purity, float((purity - null.mean()) / null.std(ddof=1))


def test_synthetic_hardcore_spacing(volume):
    """Cells respect a minimum spacing, so nearest-neighbour statistics are tissue-like."""
    from scipy.spatial import cKDTree

    for section in volume.sections:
        tree = cKDTree(section.coords.astype(np.float64))
        nn = tree.query(section.coords.astype(np.float64), k=2)[0][:, 1]
        # a Poisson field of the same density would put ~10% of points below 0.3 / sqrt(rho)
        assert nn.min() > 0.5 * volume.median_nn_dist


def test_synthetic_is_deterministic():
    """Convention 3: same seed -> bitwise identical volume."""
    a, gt_a = make_synthetic_volume(seed=7, **SMALL)
    b, gt_b = make_synthetic_volume(seed=7, **SMALL)
    c, _ = make_synthetic_volume(seed=8, **SMALL)

    for sa, sb in zip(a.sections, b.sections, strict=True):
        np.testing.assert_array_equal(sa.coords, sb.coords)
        np.testing.assert_array_equal(sa.cell_type, sb.cell_type)
        np.testing.assert_array_equal(sa.counts.toarray(), sb.counts.toarray())
    np.testing.assert_array_equal(gt_a.omega, gt_b.omega)

    assert not np.array_equal(a.sections[0].coords, c.sections[0].coords)

    points = np.array([[10.0, 20.0, 30.0], [400.0, 500.0, 100.0]])
    np.testing.assert_array_equal(gt_a.latent(points), gt_b.latent(points))


def test_synthetic_field_is_anisotropic():
    """The latent field decorrelates more slowly along z than in-plane (T04/T07 need this)."""
    vol, gt = make_synthetic_volume(seed=1, **SMALL)
    gen = np.random.default_rng(0)
    base = np.column_stack(
        [gen.uniform(0, 1000, 400), gen.uniform(0, 1000, 400), gen.uniform(0, 400, 400)]
    )
    lag = gt.autocorr_length_xy
    shifted_xy = base + np.array([lag, 0.0, 0.0])
    shifted_z = base + np.array([0.0, 0.0, lag])

    f0 = gt.latent(base)
    corr_xy = float(
        np.mean(
            [np.corrcoef(f0[:, c], gt.latent(shifted_xy)[:, c])[0, 1] for c in range(f0.shape[1])]
        )
    )
    corr_z = float(
        np.mean(
            [np.corrcoef(f0[:, c], gt.latent(shifted_z)[:, c])[0, 1] for c in range(f0.shape[1])]
        )
    )
    assert corr_z > corr_xy, f"corr along z {corr_z:.3f} should exceed in-plane {corr_xy:.3f}"
    assert vol.n_sections == SMALL["n_sections"]


# ========================================================================================
# loaders
# ========================================================================================


def _write(tmp_path, adata, name="vol.h5ad"):
    path = tmp_path / name
    adata.write_h5ad(path)
    return path


def test_load_volume_roundtrip(tmp_path, volume, cfg):
    """A volume written to h5ad and loaded back is the same volume."""
    path = _write(tmp_path, volume_to_anndata(volume, cfg))
    loaded = load_volume(path, cfg)
    assert isinstance(loaded, Volume)
    assert loaded.n_sections == volume.n_sections
    assert loaded.gene_names == volume.gene_names
    assert loaded.celltype_names == volume.celltype_names
    assert loaded.median_spacing == pytest.approx(volume.median_spacing)
    for got, want in zip(loaded.sections, volume.sections, strict=True):
        assert got.z == pytest.approx(want.z)
        assert got.thickness == pytest.approx(want.thickness)
        assert not got.thickness_is_assumed
        np.testing.assert_allclose(got.coords, want.coords)
        np.testing.assert_array_equal(got.cell_type, want.cell_type)
        np.testing.assert_array_equal(got.counts.toarray(), want.counts.toarray())


def test_load_volume_missing_keys_raise(tmp_path, volume, cfg):
    """Every missing required key raises, naming the field and the AnnData key."""
    adata = volume_to_anndata(volume, cfg)
    path = _write(tmp_path, adata)

    with pytest.raises(SchemaError, match=r"obsm\['nope'\]|obsm\[.nope.\]"):
        load_volume(path, cfg.replace(coord_key="nope"))
    with pytest.raises(SchemaError, match="celltype_key"):
        load_volume(path, cfg.replace(celltype_key="nope"))
    with pytest.raises(SchemaError, match="counts_layer"):
        load_volume(path, cfg.replace(counts_layer="raw"))

    no_region = _write(tmp_path, volume_to_anndata(volume, cfg, include_region=False), "nr.h5ad")
    with pytest.raises(SchemaError, match="region_key"):
        load_volume(no_region, cfg)
    loaded = load_volume(no_region, cfg.replace(region_key=None))
    assert isinstance(loaded, Volume)
    assert loaded.region_names is None
    assert all(s.region is None for s in loaded.sections)


def test_load_volume_lognormalised_warns(tmp_path, volume, cfg):
    """The end-to-end version of the log-normalised trap, through a layer."""
    adata = volume_to_anndata(volume, cfg, counts_transform="log1p")
    adata.layers["counts"] = adata.X
    path = _write(tmp_path, adata, "log.h5ad")
    with pytest.warns(NonIntegerCountsWarning):
        load_volume(path, cfg.replace(counts_layer="counts"))


def test_thickness_defaults_and_flags(tmp_path, volume, cfg):
    """No thickness metadata -> median_spacing, flagged, and exactly one warning."""
    path = _write(tmp_path, volume_to_anndata(volume, cfg, include_thickness=False), "nt.h5ad")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_volume(path, cfg)
    assert isinstance(loaded, Volume)

    assumed = [w for w in caught if issubclass(w.category, AssumedThicknessWarning)]
    assert len(assumed) == 1, [str(w.message) for w in caught]
    assert "thickness_is_assumed" in str(assumed[0].message)

    assert loaded.has_assumed_thickness
    for section in loaded.sections:
        assert section.thickness_is_assumed is True
        assert section.thickness == pytest.approx(loaded.median_spacing)

    # a volume that does carry the metadata neither defaults nor warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with_meta = load_volume(_write(tmp_path, volume_to_anndata(volume, cfg)), cfg)
    assert isinstance(with_meta, Volume)
    assert not with_meta.has_assumed_thickness


def test_load_volume_splits_specimens(tmp_path, cfg):
    """specimen_key yields one volume per specimen."""
    import pandas as pd

    vol_a, _ = make_synthetic_volume(seed=1, specimen_id="A", **SMALL)
    vol_b, _ = make_synthetic_volume(seed=2, specimen_id="B", **SMALL)
    a = volume_to_anndata(vol_a, cfg)
    b = volume_to_anndata(vol_b, cfg)
    a.obs["specimen"] = pd.Categorical(["A"] * a.n_obs)
    b.obs["specimen"] = pd.Categorical(["B"] * b.n_obs)
    import anndata as ad

    merged = ad.concat([a, b], index_unique="-")
    merged.uns[cfg.thickness_key] = a.uns[cfg.thickness_key]
    path = _write(tmp_path, merged, "two.h5ad")

    volumes = load_volume(path, cfg, specimen_key="specimen")
    assert isinstance(volumes, list)
    assert [v.specimen_id for v in volumes] == ["A", "B"]
    assert all(v.n_sections == SMALL["n_sections"] for v in volumes)


# ========================================================================================
# splitting: leakage discipline
# ========================================================================================


def test_split_holdout_no_leakage(volume, cfg):
    """Held-out ids never appear in training, and median_spacing is recomputed."""
    for mode, fold in [("alternating", 0), ("alternating", 1), ("consecutive", 0)]:
        train, held = split_holdout(volume, mode, fold, cfg)
        assert isinstance(train, TrainingVolume)
        assert isinstance(held, HeldOutSections)
        assert len(held) > 0

        train_ids = set(train.section_ids)
        held_ids = set(held.section_ids)
        assert train_ids & held_ids == set()
        assert train_ids | held_ids == set(volume.section_ids)

        # no held-out section object reachable from the training volume
        held_objects = {id(s) for s in held}
        assert not any(id(s) in held_objects for s in train.sections)

        # recomputed on the training sections, not inherited from the full stack
        assert train.median_spacing == pytest.approx(
            float(np.median(np.abs(np.diff(train.z_values.astype(np.float64)))))
        )
        if mode == "alternating":
            # removing every other section doubles the spacing the model can see
            assert train.median_spacing > volume.median_spacing

        # endpoints stay in training: a held-out section always has real flanking data
        assert volume.section_ids[0] in train_ids
        assert volume.section_ids[-1] in train_ids
        assert not hasattr(held, "median_spacing")


def test_split_holdout_consecutive_run(volume, cfg):
    """The consecutive regime holds out a contiguous run of the configured length."""
    for k in (1, 3, 5):
        train, held = split_holdout(volume, "consecutive", 0, cfg.replace(holdout_consecutive_k=k))
        assert len(held) == k
        positions = [volume.section_ids.index(sid) for sid in held.section_ids]
        assert positions == list(range(positions[0], positions[0] + k))
        assert train.n_sections == volume.n_sections - k

    with pytest.raises(ValueError, match="only 5 folds"):
        split_holdout(volume, "consecutive", 99, cfg)
    with pytest.raises(ValueError, match="mode"):
        split_holdout(volume, "every-third", 0, cfg)
    with pytest.raises(ValueError, match="min_sections_per_volume"):
        split_holdout(volume, "alternating", 1, cfg.replace(min_sections_per_volume=6))


def test_loso_folds_excludes_holdout(volume, cfg):
    """loso_folds accepts only a TrainingVolume, so held-out data cannot reach it."""
    train, held = split_holdout(volume, "alternating", 0, cfg)

    with pytest.raises(TypeError, match="TrainingVolume"):
        list(loso_folds(volume))  # a plain Volume may still contain evaluation sections
    with pytest.raises(TypeError, match="TrainingVolume"):
        list(loso_folds(held))  # type: ignore[arg-type]

    held_ids = set(held.section_ids)
    folds = list(loso_folds(train))
    assert len(folds) == train.n_sections
    assert [hidden.section_id for _, hidden in folds] == train.section_ids
    for rest, hidden in folds:
        assert isinstance(rest, TrainingVolume)
        assert hidden.section_id not in rest.section_ids
        assert set(rest.section_ids) & held_ids == set()
        assert rest.n_sections == train.n_sections - 1
        # derived quantities are recomputed for the fold, not inherited
        assert rest.median_spacing == pytest.approx(
            float(np.median(np.abs(np.diff(rest.z_values.astype(np.float64)))))
        )
