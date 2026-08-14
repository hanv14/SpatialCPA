"""Shared fixtures. The synthetic volume is session-scoped: building it costs a second
or two and every task's tests want the same one."""

from __future__ import annotations

import numpy as np
import pytest
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, Volume

from tests.fixtures.synthetic import GroundTruthField, make_synthetic_volume

# Seed used by the shared fixture. Anything asserting bitwise reproducibility should build
# its own volume rather than relying on this one.
FIXTURE_SEED = 0


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The default config, with the region key the synthetic fixture provides."""
    return Config()


@pytest.fixture(scope="session")
def synthetic() -> tuple[Volume, GroundTruthField]:
    """The synthetic volume and the ground-truth field it was drawn from."""
    return make_synthetic_volume(seed=FIXTURE_SEED)


@pytest.fixture(scope="session")
def volume(synthetic: tuple[Volume, GroundTruthField]) -> Volume:
    """The synthetic volume alone."""
    return synthetic[0]


@pytest.fixture(scope="session")
def gt_field(synthetic: tuple[Volume, GroundTruthField]) -> GroundTruthField:
    """The ground-truth generative map alone."""
    return synthetic[1]


def copy_section(section: Section, **overrides: object) -> Section:
    """Shallow-copy a section with fields overridden; used to build invalid volumes."""
    values = {
        "z": section.z,
        "thickness": section.thickness,
        "coords": section.coords.copy(),
        "cell_type": section.cell_type.copy(),
        "region": None if section.region is None else section.region.copy(),
        "counts": section.counts.copy(),
        "section_id": section.section_id,
        "thickness_is_assumed": section.thickness_is_assumed,
    }
    values.update(overrides)
    return Section(**values)  # type: ignore[arg-type]


def rebuild_volume(vol: Volume, sections: list[Section]) -> Volume:
    """Build a new volume with the same metadata and different sections."""
    return Volume(
        sections=sections,
        gene_names=list(vol.gene_names),
        celltype_names=list(vol.celltype_names),
        region_names=None if vol.region_names is None else list(vol.region_names),
        specimen_id=vol.specimen_id,
    )


def knn_weights(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised kNN weights: ``(N, k)`` neighbour indices and ``(N, k)`` weights."""
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    idx = tree.query(coords, k=k + 1)[1][:, 1:]
    weights = np.full(idx.shape, 1.0 / k)
    return idx, weights


def morans_i(x: np.ndarray, idx: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Moran's I per column of ``(N, G)`` ``x`` on a fixed row-normalised kNN graph.

    A small reference implementation for T01's structure test; the differentiable version
    (T08) and the scoreboard version (T10) come later and are asserted against each other
    there.
    """
    xc = x - x.mean(axis=0, keepdims=True)
    denom = (xc**2).sum(axis=0)
    lagged = np.einsum("nk,nkg->ng", weights, xc[idx])
    numer = (xc * lagged).sum(axis=0)
    s0 = weights.sum()
    out = np.full(x.shape[1], np.nan)
    ok = denom > 0
    out[ok] = (x.shape[0] / s0) * numer[ok] / denom[ok]
    return out
