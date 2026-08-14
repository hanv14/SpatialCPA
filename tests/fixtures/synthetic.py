"""Synthetic volume with known ground-truth structure.

Every subsequent task develops against this fixture: tests must run without a GPU,
without real data and without network (Convention 7). The construction is deliberately
*not* built on any model code - the ground-truth field here has its own small random
Fourier feature implementation, so that when T03's ``GaussianRandomField`` is tested
against it, agreement means something.

What is guaranteed to be recoverable
------------------------------------
* spatial autocorrelation at a known in-plane length (``autocorr_length``) and a longer
  one along z (``autocorr_length_z``) - the anisotropy T04/T07 need for oblique tests;
* genuine spatial cell-type localisation (types are the argmax of field-driven
  intensities, not an i.i.d. draw);
* realistic sparsity: ZINB counts with per-gene dispersion and dropout, detection rates
  spanning near-0 to near-1;
* ``z_gradient_genes`` genes whose mean varies monotonically with depth, so marker depth
  profiles have a ground truth;
* hard-core minimum spacing between cells, so nearest-neighbour statistics and
  pair-correlation functions are tissue-like rather than Poisson-clumped.

All randomness comes from an explicitly seeded ``numpy.random.Generator`` (Convention 3):
two calls with the same seed produce bitwise-identical volumes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.spatial import cKDTree
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, Volume

__all__ = ["GroundTruthField", "make_synthetic_volume", "volume_to_anndata"]


@dataclass(frozen=True)
class GroundTruthField:
    """The generative map the fixture was drawn from, as a callable object.

    ``latent`` is the continuous 3D field; ``expression_mu`` is the map
    ``(latent, position, cell type) -> ZINB mean``. T03's GATE 1 substitutes its own
    noise for ``latent`` and pushes it through ``expression_mu``, which is why the two
    are separate entry points rather than one end-to-end function.
    """

    omega: npt.NDArray[np.float64]
    """(M, 3) random Fourier frequencies of the Matern component."""
    phase: npt.NDArray[np.float64]
    """(M,) random Fourier phases."""
    amp: npt.NDArray[np.float64]
    """(M, L) random Fourier amplitudes, scaled to unit marginal variance."""
    sin_freq: npt.NDArray[np.float64]
    """(K, 3) low-frequency sinusoid wave vectors (cycles per micrometre)."""
    sin_phase: npt.NDArray[np.float64]
    """(K,) sinusoid phases."""
    sin_amp: npt.NDArray[np.float64]
    """(K, L) sinusoid amplitudes."""
    scale: npt.NDArray[np.float64]
    """(L,) per-channel rescaling, measured at construction so each channel has unit
    marginal variance over the volume."""
    type_weights: npt.NDArray[np.float64]
    """(L, T) latent -> cell-type logit weights."""
    type_bias: npt.NDArray[np.float64]
    """(T,) cell-type logit biases; sets the type prevalences, including a rare type."""
    gene_base: npt.NDArray[np.float64]
    """(G,) log baseline expression per gene."""
    gene_loading: npt.NDArray[np.float64]
    """(L, G) latent -> log expression loadings."""
    gene_type_effect: npt.NDArray[np.float64]
    """(T, G) additive log expression effect of each cell type."""
    gene_z_coef: npt.NDArray[np.float64]
    """(G,) coefficient on normalised depth; non-zero for the z-gradient genes."""
    gene_theta: npt.NDArray[np.float64]
    """(G,) negative-binomial dispersion."""
    gene_pi: npt.NDArray[np.float64]
    """(G,) zero-inflation probability, concentrated on low-expression genes."""
    z_center: float
    """Depth at the middle of the stack, in micrometres."""
    z_scale: float
    """Half-range of depth, in micrometres; the z gradient is expressed in these units."""
    autocorr_length_xy: float
    """Ground-truth in-plane correlation length in micrometres."""
    autocorr_length_z: float
    """Ground-truth along-z correlation length in micrometres."""
    matern_nu: float
    """Ground-truth Matern smoothness."""
    chunk_points: int = 8192
    """Query points evaluated per chunk, so the (N, M) feature matrix stays small."""

    @property
    def latent_dim(self) -> int:
        """L: number of latent channels."""
        return int(self.amp.shape[1])

    @property
    def n_types(self) -> int:
        """T: number of cell types."""
        return int(self.type_weights.shape[1])

    @property
    def n_genes(self) -> int:
        """G: number of genes."""
        return int(self.gene_base.shape[0])

    def latent(self, xyz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate the latent field. ``(N, 3)`` micrometres -> ``(N, L)``, unit variance."""
        points = np.asarray(xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"GroundTruthField.latent expects (N, 3), got {points.shape}")
        out = np.empty((points.shape[0], self.latent_dim), dtype=np.float64)
        for start in range(0, points.shape[0], self.chunk_points):
            block = points[start : start + self.chunk_points]
            grf = np.cos(block @ self.omega.T + self.phase) @ self.amp
            wave = np.sin(2.0 * np.pi * (block @ self.sin_freq.T) + self.sin_phase) @ self.sin_amp
            out[start : start + self.chunk_points] = (grf + wave) * self.scale
        return out

    def type_logits(self, latent: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """``(N, L)`` latent -> ``(N, T)`` cell-type logits."""
        return latent @ self.type_weights + self.type_bias

    def expression_mu(
        self,
        latent: npt.NDArray[np.float64],
        xyz: npt.NDArray[np.float64],
        cell_type: npt.NDArray[np.int32],
    ) -> npt.NDArray[np.float64]:
        """``(N, L)``, ``(N, 3)``, ``(N,)`` -> ``(N, G)`` ZINB means.

        The generative map T03 substitutes its own latent into. Deterministic: all the
        randomness of the fixture is in ``latent`` and in the count draw.
        """
        z_norm = (np.asarray(xyz, dtype=np.float64)[:, 2] - self.z_center) / self.z_scale
        log_mu = (
            self.gene_base[None, :]
            + latent @ self.gene_loading
            + self.gene_type_effect[cell_type]
            + z_norm[:, None] * self.gene_z_coef[None, :]
        )
        return np.exp(np.clip(log_mu, -12.0, 8.0))

    def sample_counts(self, mu: npt.NDArray[np.float64], seed: int) -> npt.NDArray[np.int32]:
        """Draw ZINB counts from ``(N, G)`` means. Returns ``(N, G)`` int32."""
        gen = np.random.default_rng(seed)
        theta = self.gene_theta[None, :]
        p = theta / (theta + mu)
        counts = gen.negative_binomial(n=np.broadcast_to(theta, mu.shape), p=p)
        dropout = gen.random(mu.shape) < self.gene_pi[None, :]
        return np.where(dropout, 0, counts).astype(np.int32)


def make_synthetic_volume(
    n_sections: int = 9,
    spacing: float = 50.0,
    n_cells_per_section: int = 1500,
    n_genes: int = 200,
    n_types: int = 6,
    autocorr_length: float = 120.0,
    z_gradient_genes: int = 20,
    seed: int = 0,
    *,
    autocorr_length_z: float = 200.0,
    extent_xy: float = 1000.0,
    thickness: float | None = None,
    latent_dim: int = 8,
    n_rff: int = 1024,
    matern_nu: float = 1.5,
    hardcore_radius: float | None = None,
    sparse_genes: int = 12,
    dense_genes: int = 8,
    base_expression_range: tuple[float, float] = (0.15, 6.0),
    sparse_expression_range: tuple[float, float] = (0.004, 0.03),
    dense_expression_range: tuple[float, float] = (15.0, 40.0),
    specimen_id: str = "synthetic",
) -> tuple[Volume, GroundTruthField]:
    """Build a synthetic volume with known ground-truth structure.

    Parameters
    ----------
    n_sections, spacing
        Stack geometry: ``n_sections`` planes ``spacing`` micrometres apart.
    n_cells_per_section, extent_xy
        Cells per section, placed by a hard-core point process in a
        ``extent_xy`` x ``extent_xy`` micrometre square.
    n_genes, n_types
        Panel width and number of cell types. Type prevalences are deliberately uneven
        and include one rare (~2%) type, which T05's ``test_rare_types_survive`` needs.
    autocorr_length, autocorr_length_z
        Ground-truth correlation lengths of the latent field. They differ, so the field
        is anisotropic: this is what T04/T07's oblique tests are measuring against.
    z_gradient_genes
        Genes given a monotone dependence on depth.
    seed
        Everything stochastic derives from this (Convention 3).
    thickness
        Slab thickness in micrometres; ``None`` gives half the spacing, which keeps the
        slabs non-overlapping.
    latent_dim, n_rff, matern_nu
        Latent field size and the accuracy of its Matern approximation.
    hardcore_radius
        Minimum inter-cell distance; ``None`` derives it from the mean density.
    sparse_genes, dense_genes
        Genes forced to very low and very high expression respectively, so the per-gene
        detection-rate distribution reaches below 0.05 and above 0.95 as the spec's
        structure test requires. The rest of the panel sits in between.
    base_expression_range, sparse_expression_range, dense_expression_range
        Baseline ZINB mean per gene (before the spatial, type and depth terms), drawn
        log-uniformly in these ranges. They set the fixture's overall sparsity, which is
        kept in the range real spatial panels show rather than at a level that would make
        every downstream test easier than reality.
    specimen_id
        Identifier stored on the volume.

    Returns
    -------
    tuple
        The :class:`~spatialcpav25_gen.data.schema.Volume` and the
        :class:`GroundTruthField` it was drawn from. Returned as a pair rather than
        stashed in an untyped ``uns`` dict (SPEC_QUESTIONS C4).
    """
    if n_sections < 3:
        raise ValueError(f"make_synthetic_volume: n_sections must be >= 3, got {n_sections}")
    if z_gradient_genes > n_genes or sparse_genes + dense_genes > n_genes:
        raise ValueError("make_synthetic_volume: more special genes than genes")

    seeds = np.random.SeedSequence(seed).spawn(4)
    gt = _build_ground_truth(
        gen=np.random.default_rng(seeds[0]),
        latent_dim=latent_dim,
        n_genes=n_genes,
        n_types=n_types,
        z_gradient_genes=z_gradient_genes,
        sparse_genes=sparse_genes,
        dense_genes=dense_genes,
        n_rff=n_rff,
        matern_nu=matern_nu,
        autocorr_length=autocorr_length,
        autocorr_length_z=autocorr_length_z,
        n_sections=n_sections,
        spacing=spacing,
        extent_xy=extent_xy,
        base_range=base_expression_range,
        sparse_range=sparse_expression_range,
        dense_range=dense_expression_range,
    )

    if hardcore_radius is None:
        radius = _default_hardcore_radius(n_cells_per_section, extent_xy)
    else:
        radius = float(hardcore_radius)
    slab = float(spacing / 2.0) if thickness is None else float(thickness)

    point_gen = np.random.default_rng(seeds[1])
    type_gen = np.random.default_rng(seeds[2])
    count_seeds = seeds[3].spawn(n_sections)

    sections: list[Section] = []
    for i in range(n_sections):
        z = float(i * spacing)
        xy = _hardcore_points(n_cells_per_section, extent_xy, radius, point_gen)
        xyz = np.concatenate([xy, np.full((xy.shape[0], 1), z)], axis=1)

        latent = gt.latent(xyz)
        logits = gt.type_logits(latent)
        # Gumbel noise keeps type assignment from being a noiseless argmax while leaving
        # the spatial organisation intact.
        gumbel = -np.log(-np.log(type_gen.random(logits.shape)))
        cell_type = np.argmax(logits + 0.6 * gumbel, axis=1).astype(np.int32)

        mu = gt.expression_mu(latent, xyz, cell_type)
        counts = gt.sample_counts(mu, seed=int(count_seeds[i].generate_state(1)[0]))

        sections.append(
            Section(
                z=z,
                thickness=slab,
                coords=xy.astype(np.float32),
                cell_type=cell_type,
                region=_region_codes(xy, extent_xy),
                counts=sparse.csr_matrix(counts),
                section_id=f"{specimen_id}_s{i:02d}",
                thickness_is_assumed=False,
            )
        )

    vol = Volume(
        sections=sections,
        gene_names=[f"Gene{j:04d}" for j in range(n_genes)],
        celltype_names=[f"Type{t}" for t in range(n_types)],
        region_names=["RegionA", "RegionB", "RegionC"],
        specimen_id=specimen_id,
    )
    return vol, gt


def _build_ground_truth(
    *,
    gen: np.random.Generator,
    latent_dim: int,
    n_genes: int,
    n_types: int,
    z_gradient_genes: int,
    sparse_genes: int,
    dense_genes: int,
    n_rff: int,
    matern_nu: float,
    autocorr_length: float,
    autocorr_length_z: float,
    n_sections: int,
    spacing: float,
    extent_xy: float,
    base_range: tuple[float, float],
    sparse_range: tuple[float, float],
    dense_range: tuple[float, float],
    n_probe: int = 4096,
) -> GroundTruthField:
    """Draw the parameters of the generative map."""
    ell = np.array([autocorr_length, autocorr_length, autocorr_length_z], dtype=np.float64)

    # Matern spectral density as a Gamma scale mixture of Gaussians (the same
    # construction T03 uses; reproduced here so the fixture does not depend on T03).
    z_m = gen.standard_normal((n_rff, 3))
    g_m = gen.gamma(shape=matern_nu, scale=1.0 / matern_nu, size=(n_rff, 1))
    omega = (z_m / np.sqrt(g_m)) / ell[None, :]
    phase = gen.uniform(0.0, 2.0 * np.pi, size=n_rff)
    # cos(omega p + b) has variance 1/2, so sqrt(2/M) * A with unit-variance A gives a
    # unit-variance field. 0.7 leaves room for the low-frequency sinusoids below.
    amp = gen.standard_normal((n_rff, latent_dim)) * np.sqrt(2.0 / n_rff) * 0.7

    n_sin = 3
    # Wavelengths of a few times the correlation length: large, smooth anatomy. Scaled
    # per axis by `ell`, so this component is anisotropic in the same direction as the
    # Matern one - otherwise three isotropic sinusoids carrying half the variance would
    # swamp the anisotropy the fixture exists to provide.
    sin_freq = gen.uniform(-1.0, 1.0, size=(n_sin, 3)) / (4.0 * ell[None, :])
    sin_phase = gen.uniform(0.0, 2.0 * np.pi, size=n_sin)
    sin_amp = gen.standard_normal((n_sin, latent_dim)) * np.sqrt(2.0 / n_sin) * 0.7

    type_weights = gen.standard_normal((latent_dim, n_types)) * 1.4
    type_bias = np.linspace(0.6, -2.4, n_types)  # uneven prevalences incl. a rare type

    # Log baselines spanning three orders of magnitude, plus a handful of very sparse
    # genes and a handful of housekeeping-like ones, so per-gene detection rates cover
    # (0.05, 0.95) from both ends.
    gene_base = gen.uniform(np.log(base_range[0]), np.log(base_range[1]), size=n_genes)
    gene_base[:sparse_genes] = gen.uniform(
        np.log(sparse_range[0]), np.log(sparse_range[1]), size=sparse_genes
    )
    gene_base[sparse_genes : sparse_genes + dense_genes] = gen.uniform(
        np.log(dense_range[0]), np.log(dense_range[1]), size=dense_genes
    )

    loading_dirs = gen.standard_normal((latent_dim, n_genes))
    loading_dirs /= np.linalg.norm(loading_dirs, axis=0, keepdims=True)
    gene_loading = loading_dirs * gen.uniform(0.9, 1.7, size=n_genes)[None, :]

    gene_type_effect = gen.standard_normal((n_types, n_genes)) * 0.8

    gene_z_coef = np.zeros(n_genes)
    z_idx = gen.choice(n_genes, size=z_gradient_genes, replace=False)
    signs = gen.choice(np.array([-1.0, 1.0]), size=z_gradient_genes)
    gene_z_coef[z_idx] = signs * gen.uniform(1.0, 1.8, size=z_gradient_genes)

    gene_theta = gen.uniform(2.0, 10.0, size=n_genes)
    # Zero-inflation on top of the NB zeros, concentrated on low-expression genes: a
    # highly expressed gene detected in 40% of cells is not something real data does.
    gene_pi = gen.uniform(0.0, 0.4, size=n_genes) * np.exp(-np.exp(gene_base) / 2.0)

    z_values = np.arange(n_sections, dtype=np.float64) * spacing
    z_center = float(z_values.mean())
    z_scale = float(max(z_values.max() - z_center, 1.0))

    field = GroundTruthField(
        omega=omega,
        phase=phase,
        amp=amp,
        sin_freq=sin_freq,
        sin_phase=sin_phase,
        sin_amp=sin_amp,
        scale=np.ones(latent_dim),
        type_weights=type_weights,
        type_bias=type_bias,
        gene_base=gene_base,
        gene_loading=gene_loading,
        gene_type_effect=gene_type_effect,
        gene_z_coef=gene_z_coef,
        gene_theta=gene_theta,
        gene_pi=gene_pi,
        z_center=z_center,
        z_scale=z_scale,
        autocorr_length_xy=autocorr_length,
        autocorr_length_z=autocorr_length_z,
        matern_nu=matern_nu,
    )

    # Measure the realised per-channel standard deviation over the volume and divide it
    # out, so "unit marginal variance" is a fact about the fixture rather than an
    # approximation of the construction.
    probe = np.column_stack(
        [
            gen.uniform(0.0, extent_xy, n_probe),
            gen.uniform(0.0, extent_xy, n_probe),
            gen.uniform(float(z_values.min()), float(z_values.max()), n_probe),
        ]
    )
    return replace(field, scale=1.0 / field.latent(probe).std(axis=0))


def _default_hardcore_radius(n_cells: int, extent_xy: float) -> float:
    """Hard-core radius at 60% of the mean nearest-neighbour distance of a Poisson field."""
    density = n_cells / (extent_xy**2)
    return float(0.6 * 0.5 / np.sqrt(density))


def _hardcore_points(
    n: int,
    extent_xy: float,
    radius: float,
    gen: np.random.Generator,
    batch: int = 512,
    max_batches: int = 400,
) -> npt.NDArray[np.float64]:
    """Sample ``n`` points at least ``radius`` apart by dart throwing. ``(n, 2)`` float64."""
    accepted = np.empty((n, 2), dtype=np.float64)
    n_accepted = 0
    for _ in range(max_batches):
        candidates = gen.uniform(0.0, extent_xy, size=(batch, 2))
        if n_accepted:
            tree = cKDTree(accepted[:n_accepted])
            candidates = candidates[tree.query(candidates, k=1)[0] >= radius]
        for point in candidates:
            if n_accepted and np.linalg.norm(accepted[:n_accepted] - point, axis=1).min() < radius:
                continue
            accepted[n_accepted] = point
            n_accepted += 1
            if n_accepted == n:
                return accepted
    raise RuntimeError(
        f"_hardcore_points: placed only {n_accepted} of {n} points at radius {radius:g} um "
        f"in a {extent_xy:g} um square; the requested density exceeds what a hard-core "
        "process can reach"
    )


def _region_codes(xy: npt.NDArray[np.float64], extent_xy: float) -> npt.NDArray[np.int32]:
    """Three horizontal anatomical bands, so regions are contiguous and non-trivial."""
    return np.clip((xy[:, 1] / extent_xy * 3.0).astype(np.int32), 0, 2)


def volume_to_anndata(
    vol: Volume,
    cfg: Config,
    *,
    include_thickness: bool = True,
    include_region: bool = True,
    counts_transform: str = "raw",
) -> Any:
    """Write a :class:`Volume` back out as an AnnData, for exercising ``load_volume``.

    Parameters
    ----------
    vol, cfg
        The volume, and the config whose keys the AnnData is written against.
    include_thickness
        When ``False``, omit the thickness metadata so ``load_volume`` has to default it.
    include_region
        When ``False``, omit the region column.
    counts_transform
        ``"raw"`` or ``"log1p"``; the latter produces exactly the mistake
        ``NonIntegerCountsWarning`` exists to catch.
    """
    import anndata as ad
    import pandas as pd

    coords = np.concatenate([s.coords for s in vol.sections], axis=0).astype(np.float32)
    z = np.concatenate([np.full(s.n_cells, s.z, dtype=np.float64) for s in vol.sections])
    counts = sparse.vstack([s.counts for s in vol.sections]).tocsr()
    if counts_transform == "log1p":
        counts = counts.astype(np.float32)
        counts.data = np.log1p(counts.data)
    elif counts_transform != "raw":
        raise ValueError(f"volume_to_anndata: unknown counts_transform {counts_transform!r}")

    obs = pd.DataFrame(
        {
            cfg.z_key: z,
            cfg.celltype_key: pd.Categorical(
                [vol.celltype_names[c] for s in vol.sections for c in s.cell_type],
                categories=vol.celltype_names,
            ),
        },
        index=[f"cell{i:06d}" for i in range(int(counts.shape[0]))],
    )
    if include_region and vol.region_names is not None:
        obs[str(cfg.region_key)] = pd.Categorical(
            [vol.region_names[r] for s in vol.sections if s.region is not None for r in s.region],
            categories=vol.region_names,
        )

    adata = ad.AnnData(
        X=counts,
        obs=obs,
        var=pd.DataFrame(index=list(vol.gene_names)),
    )
    adata.obsm[cfg.coord_key] = coords
    if include_thickness:
        adata.uns[cfg.thickness_key] = float(vol.sections[0].thickness)
    return adata
