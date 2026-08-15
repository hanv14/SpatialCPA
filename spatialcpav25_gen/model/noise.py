"""The 3D Gaussian random field prior — the mechanism the whole method rests on.

Two earlier versions of this project failed because generated sections lost spatial
autocorrelation. The fix, and the single load-bearing idea of SEFL, is that the
flow-matching prior is not i.i.d. noise attached to a generated point set but a
**continuous random field over physical space**::

    xi : R^3 -> R^{d_h},    xi ~ GRF( Matern(nu, ell) ),   ell = (ell_x, ell_y, ell_z)

queried at cell positions, ``h_0^i = xi(p_i)``. Because ``xi`` is a function of position
and nothing else, two virtual sections cut at different angles receive *identical* noise
where they intersect: mutual coherence is exact by construction rather than something a
loss has to negotiate. A kNN graph filter over the generated cells would be cheaper and
would destroy exactly that property (T03 §1); it is not implemented here on purpose.

Construction — random Fourier features
--------------------------------------
The Matern spectral density in ``R^3`` is a multivariate Student-t density with ``2 nu``
degrees of freedom, so a frequency drawn as a Gaussian scale mixture,

    z_m ~ N(0, I_3),  g_m ~ Gamma(shape=nu, scale=1/nu),  omega_m = (z_m / sqrt(g_m)) / ell

has exactly the Matern spectral density, and

    xi(p) = sqrt(2 / M) * cos(omega @ p + b) @ A,   b ~ U(0, 2 pi),  A ~ N(0, 1)^(M, d_h)

has covariance ``Cov(xi_c(p), xi_c(q)) -> matern_nu(||p - q||_ell)`` as ``M -> inf``,
with ``||h||_ell = sqrt(sum_i (h_i / ell_i)^2)``. That is the *standard* parametrisation,

    k(r) = 2^(1-nu) / Gamma(nu) * (sqrt(2 nu) r)^nu * K_nu(sqrt(2 nu) r),

i.e. the one ``sklearn.gaussian_process.kernels.Matern(length_scale=ell, nu=nu)`` uses —
no convention constant is needed (SPEC_QUESTIONS B8 worried there would be; there is not,
and :func:`matern_correlation` plus GATE 1's G1.1 check it numerically rather than on
trust).

Numerical conventions
---------------------
* The draws are stored in float64 — ``with_lengthscale`` divides them by a new ``ell``
  and must not accumulate rounding across a calibration loop — but a query is evaluated
  in float32. Coordinates are micrometres, so the phase ``omega . p`` reaches a few
  hundred radians and its float32 representation carries ~2e-5 rad of error; that is the
  same order as the float32 quantisation of the coordinate itself (Convention 4), and it
  moves ``xi`` by ~2e-5 on a unit-variance field. Evaluating the phase in float64 instead
  would triple the cost of every query — the ``(chunk, M)`` feature block stops fitting in
  cache — to chase precision the inputs do not have.
* ``A``'s columns are orthogonalised and scaled to norm ``sqrt(M)``. The space-averaged
  marginal variance of channel ``c`` is exactly ``||A_c||^2 / M``, so this makes unit
  marginal variance and zero cross-channel covariance *exact* rather than a Monte Carlo
  approximation of them (a deviation from the spec's "measure on a random sample and
  rescale", in the direction of being exactly what that measurement is trying to achieve).
* Every draw comes from ``numpy.random.default_rng(seed)`` (Convention 3), so two
  processes on any platform build the same field bitwise.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy import special
from torch import Tensor, nn

from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Volume

__all__ = [
    "GaussianRandomField",
    "LengthscaleFit",
    "fit_lengthscale_from_sections",
    "matern_correlation",
    "scaled_distance",
]

_TWO_PI = 2.0 * math.pi

_VARIOGRAM_WEIGHT_FLOOR = 1e-9
"""Guard on Cressie's ``N(h) / gamma(h)^2`` weights: a bin with zero semivariance would
otherwise carry infinite weight. Not a tunable — it exists so a division cannot explode.
"""

FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------------------
# the analytic kernel
# --------------------------------------------------------------------------------------


def matern_correlation(r: npt.ArrayLike, nu: float) -> FloatArray:
    """Analytic Matern correlation at scaled distance ``r``.

    ``r`` is ``||p - q||_ell`` (see :func:`scaled_distance`), i.e. the length-scale is
    already divided out. Any shape in, the same shape out.

    ``k(r) = 2^(1-nu)/Gamma(nu) * (sqrt(2 nu) r)^nu * K_nu(sqrt(2 nu) r)``, with
    ``k(0) = 1``; this is the parametrisation ``sklearn``'s ``Matern`` uses, and the one
    :class:`GaussianRandomField` realises.
    """
    if nu <= 0.0:
        raise ValueError(f"matern_correlation: nu must be > 0, got {nu!r}")
    radius = np.asarray(r, dtype=np.float64)
    out = np.ones(radius.shape, dtype=np.float64)
    positive = radius > 0.0
    arg = math.sqrt(2.0 * nu) * radius[positive]
    log_prefactor = (1.0 - nu) * math.log(2.0) - special.gammaln(nu)
    out[positive] = np.exp(log_prefactor + nu * np.log(arg)) * special.kv(nu, arg)
    return out


def scaled_distance(p: FloatArray, q: FloatArray, ell: tuple[float, float, float]) -> FloatArray:
    """``||p - q||_ell`` for ``(N, 3)`` point sets. Returns ``(N,)`` float64."""
    lengths = np.asarray(ell, dtype=np.float64)
    diff = (np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)) / lengths
    return cast(FloatArray, np.sqrt((diff**2).sum(axis=-1)))


# --------------------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------------------


class _FieldDraws(NamedTuple):
    """The seeded random draws that define one field realisation.

    ``directions`` is ``z_m / sqrt(g_m)``, i.e. the frequencies *before* the length-scale
    is divided out. Keeping it — rather than ``omega`` — is what lets
    :meth:`GaussianRandomField.with_lengthscale` rescale without redrawing.
    """

    directions: FloatArray
    """(M, 3) unscaled frequencies."""
    phase: FloatArray
    """(M,) phases in [0, 2 pi)."""
    amplitude: FloatArray
    """(M, d_h) amplitudes, already carrying the sqrt(2 / M) factor."""


def _draw(cfg: Config, seed: int) -> _FieldDraws:
    """Draw the frequencies, phases and amplitudes of one field realisation."""
    n_rff = int(cfg.n_rff)
    latent_dim = int(cfg.latent_dim)
    if latent_dim > n_rff:
        raise ValueError(
            f"Config.latent_dim={latent_dim} exceeds Config.n_rff={n_rff}; the amplitude "
            "columns cannot be orthogonalised, so channels would not be independent"
        )
    gen = np.random.default_rng(seed)
    normal = gen.standard_normal((n_rff, 3))
    gamma = gen.gamma(shape=cfg.matern_nu, scale=1.0 / cfg.matern_nu, size=(n_rff, 1))
    directions = normal / np.sqrt(gamma)
    phase = gen.uniform(0.0, _TWO_PI, size=n_rff)
    amplitude = _orthonormal_amplitudes(gen.standard_normal((n_rff, latent_dim)))
    return _FieldDraws(
        directions=directions,
        phase=phase,
        amplitude=amplitude * math.sqrt(2.0 / n_rff),
    )


def _orthonormal_amplitudes(raw: FloatArray) -> FloatArray:
    """Orthogonalise ``(M, d_h)`` amplitudes and scale every column to norm ``sqrt(M)``.

    The space-averaged marginal variance of channel ``c`` is ``||A_c||^2 / M`` and the
    space-averaged covariance between channels ``c`` and ``c'`` is ``A_c . A_c' / M``, so
    orthonormal columns give exactly unit variance and exactly independent channels.
    ``E[A_c A_c^T] = I`` still holds (a random direction ``u`` in ``R^M`` has
    ``E[u u^T] = I / M``), so the field's covariance function is unchanged.

    The signs of ``R``'s diagonal are normalised, which makes the factorisation canonical
    and therefore reproducible across LAPACK builds.
    """
    q, r = np.linalg.qr(raw)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return cast(FloatArray, q * signs[None, :] * math.sqrt(raw.shape[0]))


class GaussianRandomField(nn.Module):
    """A continuous, anisotropic Matern GRF over R^3 with ``d_h`` output channels.

    ``xi(p)`` has zero mean, unit marginal variance per channel, and
    ``Cov(xi_c(p), xi_c(q)) ~= matern_nu( ||p - q||_ell )``, channels independent.

    Args:
        cfg:  supplies ``n_rff`` (M), ``latent_dim`` (d_h), ``matern_nu``,
              ``grf_chunk_points`` and ``debug_shapes``.
        ell:  ``(ell_x, ell_y, ell_z)`` correlation lengths in micrometres.
        seed: the realisation. Two fields with the same ``cfg`` and ``seed`` are bitwise
              identical, in this process or any other (Convention 3).

    Buffers
    -------
    ``directions``
        ``(M, 3)`` float64 ``z_m / sqrt(g_m)``, the frequencies before ``ell``.
    ``omega``
        ``(M, 3)`` float64 ``directions / ell``.
    ``phase``
        ``(M,)`` float64.
    ``amplitude``
        ``(M, d_h)`` float64, including the ``sqrt(2 / M)`` factor.

    Notes
    -----
    ``__call__`` is pure: it touches no RNG and mutates nothing, so it may be called in
    any order, from any number of threads, with any point set. **The field depends on
    physical position only** — never on which points happen to have been generated
    (T03 "Do NOT"), which is what makes two intersecting sections agree.

    The exact contract on "identical", which is what G1.2 and T07's ``L_cross`` rest on:

    * **Identical points in an identically shaped batch give bitwise identical values**,
      in this process, another process, or another machine. This is the property the
      intersection-consistency claim needs — both branches sample the same number of
      points along the shared line — and it is asserted with ``torch.equal``.
    * Change the *shape* of the batch (query a point alone, or with a different
      ``grf_chunk_points``) and float32 matmul is free to reassociate its sum over the M
      features: torch dispatches a matrix-*vector* kernel for a one-row batch, and BLAS
      vendors block the reduction differently. The values then agree to a few 1e-6 rather
      than exactly. That is a property of float32 GEMM, not of the field, and it is
      asserted with a tolerance rather than papered over with padding that would only
      appear to guarantee more.
    """

    directions: Tensor
    omega: Tensor
    phase: Tensor
    amplitude: Tensor
    omega32: Tensor
    phase32: Tensor
    amplitude32: Tensor

    def __init__(
        self,
        cfg: Config,
        ell: tuple[float, float, float],
        seed: int,
        *,
        draws: _FieldDraws | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.ell = _check_ell(ell)
        self.seed = int(seed)
        realisation = _draw(cfg, self.seed) if draws is None else draws
        lengths = np.asarray(self.ell, dtype=np.float64)
        self.register_buffer("directions", torch.from_numpy(realisation.directions.copy()))
        self.register_buffer("omega", torch.from_numpy(realisation.directions / lengths[None, :]))
        self.register_buffer("phase", torch.from_numpy(realisation.phase.copy()))
        self.register_buffer("amplitude", torch.from_numpy(realisation.amplitude.copy()))
        # The query path is float32 (see the module docstring). These are derived, so they
        # are not persisted: a checkpoint carries the float64 draws and nothing else.
        self.register_buffer("omega32", self.omega.to(torch.float32), persistent=False)
        self.register_buffer("phase32", self.phase.to(torch.float32), persistent=False)
        self.register_buffer("amplitude32", self.amplitude.to(torch.float32), persistent=False)

    # ----------------------------------------------------------------------------------
    # properties
    # ----------------------------------------------------------------------------------

    @property
    def n_rff(self) -> int:
        """M: number of random Fourier features."""
        return int(self.omega.shape[0])

    @property
    def latent_dim(self) -> int:
        """d_h: number of output channels."""
        return int(self.amplitude.shape[1])

    def extra_repr(self) -> str:
        """Show the length-scale, the feature count and the seed."""
        return f"ell={self.ell}, M={self.n_rff}, d_h={self.latent_dim}, seed={self.seed}"

    # ----------------------------------------------------------------------------------
    # evaluation
    # ----------------------------------------------------------------------------------

    def __call__(self, xyz: npt.NDArray[Any] | Tensor) -> Tensor:
        """``xyz``: ``(N, 3)`` physical coords -> ``(N, d_h)`` noise, deterministic given seed."""
        return cast(Tensor, super().__call__(xyz))

    def forward(self, xyz: npt.NDArray[Any] | Tensor) -> Tensor:
        """``(N, 3)`` micrometres -> ``(N, d_h)`` float32.

        Differentiable with respect to ``xyz``; the random draws are buffers and carry no
        gradient. Evaluated in chunks of ``cfg.grf_chunk_points`` so that the ``(N, M)``
        feature matrix never has to exist in full — at ``N = 1e6``, ``M = 4096`` it would
        be 16 GB — and, more to the point, so that each ``(chunk, M)`` block stays in
        cache: the same query is three times slower with a chunk that has to round-trip
        through main memory.
        """
        points = _as_points(xyz)
        if self.cfg.debug_shapes:
            assert points.ndim == 2, points.shape
            assert points.shape[1] == 3, points.shape
        chunk = int(self.cfg.grf_chunk_points)
        blocks: list[Tensor] = []
        for start in range(0, int(points.shape[0]), chunk):
            phase = torch.addmm(self.phase32, points[start : start + chunk], self.omega32.T)
            blocks.append(torch.cos(phase) @ self.amplitude32)
        out = (
            torch.cat(blocks, dim=0)
            if blocks
            else torch.zeros((0, self.latent_dim), dtype=torch.float32)
        )
        if self.cfg.debug_shapes:
            assert out.shape == (points.shape[0], self.latent_dim), out.shape
        return out

    def evaluate_numpy(self, xyz: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
        """Evaluate the field through numpy. ``(N, 3)`` micrometres -> ``(N, d_h)`` float32.

        The numpy-facing path: numpy in, numpy out, no tensor to carry around — used by
        the variogram fit, the gate report and anything else doing analysis rather than
        training.

        It *delegates* to :meth:`forward` rather than re-implementing the arithmetic in
        numpy. A second implementation would be a second thing to keep in step, and its
        float32 matmul would reassociate differently from torch's — a divergence of a few
        1e-6 that varies by BLAS vendor and platform, so the two paths could only ever
        have been compared with a tolerance. Delegating makes them equal by construction,
        on every machine.
        """
        values = self.forward(xyz).detach().numpy()
        return cast("npt.NDArray[np.float32]", values.astype(np.float32, copy=False))

    def covariance(
        self, p: npt.NDArray[Any] | Tensor, q: npt.NDArray[Any] | Tensor
    ) -> npt.NDArray[np.float64]:
        """Covariance of this realisation between paired points. Two ``(N, 3)`` -> ``(N,)``.

        ``Cov(xi_c(p_i), xi_c(q_i))`` over the amplitude draw, with the frequencies held
        at the values this seed produced: ``(2 / M) * sum_m cos(omega_m . p + b_m)
        cos(omega_m . q + b_m)``, evaluated in float64. It converges to
        ``matern_nu(||p - q||_ell)`` as ``M`` grows, and the gap is the field's whole
        approximation error — which is why GATE 1 measures *this* rather than a Monte
        Carlo estimate over the ``d_h`` channels: the latter adds ``O(1/sqrt(d_h))`` of
        amplitude-draw noise, which is a property of how many channels were sampled and
        not of how well the field approximates a Matern.
        """
        points_p = _as_points(p, dtype=torch.float64)
        points_q = _as_points(q, dtype=torch.float64)
        if points_p.shape != points_q.shape:
            raise ValueError(
                f"GaussianRandomField.covariance expects matching point sets, got "
                f"{tuple(points_p.shape)} and {tuple(points_q.shape)}"
            )
        chunk = int(self.cfg.grf_chunk_points)
        out = torch.empty(points_p.shape[0], dtype=torch.float64)
        for start in range(0, int(points_p.shape[0]), chunk):
            stop = start + chunk
            cos_p = torch.cos(points_p[start:stop] @ self.omega.T + self.phase)
            cos_q = torch.cos(points_q[start:stop] @ self.omega.T + self.phase)
            out[start:stop] = (cos_p * cos_q).sum(dim=1) * (2.0 / self.n_rff)
        return cast("npt.NDArray[np.float64]", out.numpy())

    # ----------------------------------------------------------------------------------
    # rescaling
    # ----------------------------------------------------------------------------------

    def with_lengthscale(self, ell: tuple[float, float, float]) -> GaussianRandomField:
        """Rescale the same seed and frequencies — used by the inference-time calibrator.

        The draws are carried over rather than redrawn. T09's calibration loop compares
        ``I_gen`` across a sequence of length-scales and bisects; if each step saw a fresh
        realisation the objective would jitter by more than the effect being measured and
        the bisection would not converge.
        """
        return GaussianRandomField(
            self.cfg,
            ell,
            self.seed,
            draws=_FieldDraws(
                directions=self.directions.detach().numpy(),
                phase=self.phase.detach().numpy(),
                amplitude=self.amplitude.detach().numpy(),
            ),
        )


def _check_ell(ell: tuple[float, float, float]) -> tuple[float, float, float]:
    """Validate a length-scale triple and return it as floats."""
    values = tuple(float(v) for v in ell)
    if len(values) != 3:
        raise ValueError(f"ell must be (ell_x, ell_y, ell_z), got {ell!r}")
    if not all(math.isfinite(v) and v > 0.0 for v in values):
        raise ValueError(f"ell must be finite and strictly positive micrometres, got {ell!r}")
    return cast("tuple[float, float, float]", values)


def _as_points(xyz: npt.NDArray[Any] | Tensor, dtype: torch.dtype = torch.float32) -> Tensor:
    """Coerce ``(N, 3)`` coordinates to a contiguous tensor, preserving any gradient."""
    points = xyz if isinstance(xyz, Tensor) else torch.from_numpy(np.asarray(xyz))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"GaussianRandomField expects (N, 3) coordinates, got {tuple(points.shape)}"
        )
    return points.to(dtype).contiguous()


# --------------------------------------------------------------------------------------
# fitting ell from the training sections
# --------------------------------------------------------------------------------------


class LengthscaleFitWarning(UserWarning):
    """A fitted length-scale sits on the edge of its search grid.

    The returned value is then a bound, not an optimum: the data is telling you the
    correlation length is outside the range the volume can resolve (shorter than the
    cell spacing, or longer than the domain), and calibrating against it (T09) will
    push against the same wall.
    """


class VariogramError(ValueError):
    """The empirical variogram carries too little spatial structure to fit a Matern to."""


class LengthscaleFit(NamedTuple):
    """Everything the variogram fit measured, for the report and for T09's diagnostics."""

    ell: tuple[float, float, float]
    """(ell_x, ell_y, ell_z) in micrometres; ``ell_x == ell_y`` by construction."""
    lags_xy: FloatArray
    """(B_xy,) mean in-plane lag per bin, micrometres."""
    gamma_xy: FloatArray
    """(B_xy,) empirical in-plane semivariance, normalised so the total PC variance is 1."""
    counts_xy: FloatArray
    """(B_xy,) number of point pairs per in-plane bin."""
    nugget_xy: float
    """Fitted in-plane nugget: the share of variance that is not spatially structured."""
    sill_xy: float
    """Fitted in-plane sill (structured share)."""
    rmse_xy: float
    """Weighted RMSE of the in-plane fit."""
    lags_z: FloatArray
    """(B_z,) section-to-section lags, micrometres."""
    gamma_z: FloatArray
    """(B_z,) empirical semivariance between grid-averaged sections."""
    counts_z: FloatArray
    """(B_z,) number of (section pair, grid cell) terms per lag."""
    nugget_z: float
    """Fitted along-z nugget."""
    sill_z: float
    """Fitted along-z sill."""
    rmse_z: float
    """Weighted RMSE of the along-z fit."""
    n_pcs: int
    """Number of expression PCs the variograms were computed on."""


def fit_lengthscale_from_sections(
    vol: Volume, cfg: Config, *, seed: int
) -> tuple[float, float, float]:
    """Fit ``ell = (ell_x, ell_y, ell_z)`` from a volume's expression autocorrelation.

    ``ell_xy`` (used for both in-plane axes) comes from the empirical semivariogram of the
    top ``cfg.variogram_n_pcs`` expression PCs *within* sections; ``ell_z`` from the decay
    of the same PCs *between* sections. The anisotropy matters: with a single isotropic
    length-scale, a section cut at 45 degrees gets an in-plane correlation length wrong by
    a factor that depends on the angle, which is exactly the error SEFL exists to remove.

    Parameters
    ----------
    vol
        A :class:`~spatialcpav25_gen.data.schema.Volume` — in the pipeline a
        ``TrainingVolume``. ``HeldOutSections`` is refused: fitting ``ell`` on held-out
        sections is leakage (``CLAUDE.md``, "Leakage discipline").
    cfg
        Supplies every constant: ``matern_nu``, ``variogram_*``.
    seed
        Seeds the per-section cell subsample. Required rather than defaulted, because
        Convention 3 forbids leaving a stochastic step to an implicit RNG; two calls with
        the same seed return exactly the same triple.

    Returns
    -------
    tuple
        ``(ell_x, ell_y, ell_z)`` in micrometres, with ``ell_x == ell_y``.
    """
    return fit_lengthscale_details(vol, cfg, seed=seed).ell


def fit_lengthscale_details(vol: Volume, cfg: Config, *, seed: int) -> LengthscaleFit:
    """As :func:`fit_lengthscale_from_sections`, keeping the curves that were fitted.

    Same computation; the extra return surface is what the GATE 1 report plots and what a
    human looks at when a fitted length-scale is surprising.
    """
    if not isinstance(vol, Volume):
        raise TypeError(
            "fit_lengthscale_from_sections expects a Volume (in the pipeline a "
            f"TrainingVolume), got {type(vol).__name__}. Fitting the noise field's "
            "length-scale on held-out sections is leakage."
        )
    scores = _expression_pc_scores(vol, cfg)
    lags_xy, gamma_xy, counts_xy = _inplane_variogram(vol, scores, cfg, seed=seed)
    max_lag_xy = float(lags_xy.max())
    ell_xy, nugget_xy, sill_xy, rmse_xy = _fit_matern_variogram(
        lags_xy,
        gamma_xy,
        counts_xy,
        cfg,
        lo=cfg.variogram_ell_min_factor * vol.median_nn_dist,
        hi=cfg.variogram_ell_max_factor * max_lag_xy,
        axis="in-plane",
    )
    lags_z, gamma_z, counts_z = _along_z_variogram(vol, scores, cfg)
    ell_z, nugget_z, sill_z, rmse_z = _fit_matern_variogram(
        lags_z,
        gamma_z,
        counts_z,
        cfg,
        lo=cfg.variogram_ell_min_factor * vol.median_spacing,
        hi=cfg.variogram_ell_max_factor * float(lags_z.max()),
        axis="along-z",
    )
    return LengthscaleFit(
        ell=(ell_xy, ell_xy, ell_z),
        lags_xy=lags_xy,
        gamma_xy=gamma_xy,
        counts_xy=counts_xy,
        nugget_xy=nugget_xy,
        sill_xy=sill_xy,
        rmse_xy=rmse_xy,
        lags_z=lags_z,
        gamma_z=gamma_z,
        counts_z=counts_z,
        nugget_z=nugget_z,
        sill_z=sill_z,
        rmse_z=rmse_z,
        n_pcs=int(scores[0].shape[1]),
    )


def _expression_pc_scores(vol: Volume, cfg: Config) -> list[FloatArray]:
    """Top-``variogram_n_pcs`` expression PC scores per section.

    Returns one ``(N_i, P)`` float64 array per section, in stack order, scaled so that the
    *pooled* variance summed over the retained PCs is 1. A semivariogram averaged over
    them then has a sill of about 1, which makes the fitted nugget and sill read directly
    as the unstructured and structured shares of expression variance.

    The counts are library-size normalised and ``log1p``-ed first. That is an *input-side*
    transform of the kind Convention 5 permits (and requires to stay off the decoder's
    target): a variogram of raw counts would mostly measure library size.
    """
    from scipy import sparse

    blocks: list[FloatArray] = []
    for section in vol.sections:
        counts = section.counts
        dense = np.asarray(
            counts.toarray() if sparse.issparse(counts) else counts, dtype=np.float64
        )
        totals = dense.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1.0
        blocks.append(np.log1p(dense / totals * float(np.median(totals))))
    pooled = np.concatenate(blocks, axis=0)
    pooled -= pooled.mean(axis=0, keepdims=True)

    n_pcs = int(min(cfg.variogram_n_pcs, pooled.shape[0], pooled.shape[1]))
    # Economy SVD: deterministic, and at panel sizes (<= a few thousand genes) cheaper
    # than an iterative solver that would need its own seed.
    _, singular, right = np.linalg.svd(pooled, full_matrices=False)
    basis = right[:n_pcs].T
    total_variance = float((singular[:n_pcs] ** 2).sum() / max(pooled.shape[0] - 1, 1))
    if total_variance <= 0.0:
        raise VariogramError(
            f"specimen {vol.specimen_id!r}: the expression matrix has no variance, so no "
            "correlation length can be fitted"
        )
    scale = 1.0 / math.sqrt(total_variance)
    offset = 0
    scores: list[FloatArray] = []
    for section in vol.sections:
        block = pooled[offset : offset + section.n_cells]
        scores.append((block @ basis) * scale)
        offset += section.n_cells
    return scores


def _inplane_variogram(
    vol: Volume, scores: list[FloatArray], cfg: Config, *, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Binned in-plane semivariogram pooled over sections.

    Returns ``(B,)`` mean lag, ``(B,)`` semivariance and ``(B,)`` pair count, for the bins
    that hold at least ``cfg.variogram_min_pairs_per_bin`` pairs.
    """
    extent = vol.bbox[1, :2] - vol.bbox[0, :2]
    max_lag = float(cfg.variogram_max_lag_frac * float(np.min(extent)))
    n_bins = int(cfg.variogram_n_bins)
    lag_sum = np.zeros(n_bins)
    gamma_sum = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    gen = np.random.default_rng(seed)
    for section, block in zip(vol.sections, scores, strict=True):
        keep = _subsample(section.n_cells, cfg.variogram_max_cells_per_section, gen)
        coords = np.asarray(section.coords, dtype=np.float64)[keep]
        values = block[keep]
        distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        semivariance = 0.5 * ((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=-1)
        upper = np.triu_indices(coords.shape[0], k=1)
        _accumulate_bins(
            distance[upper], semivariance[upper], max_lag, n_bins, lag_sum, gamma_sum, counts
        )
    return _finish_bins(lag_sum, gamma_sum, counts, cfg, axis="in-plane")


def _along_z_variogram(
    vol: Volume, scores: list[FloatArray], cfg: Config
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Semivariogram between sections, on a coarse in-plane grid.

    Cells in different sections have no correspondence, so the comparison is made between
    *grid-cell means* of the PC scores: a ``G x G`` in-plane grid (``G =
    cfg.variogram_z_grid_size``) is averaged within each section, and the semivariance
    between two sections is taken over the grid cells both of them fill. In-plane
    averaging shrinks the sill and inflates the nugget by a constant factor, both of which
    the fit's free ``(nugget, sill)`` absorb, leaving ``ell_z`` where it should be.
    """
    grid = int(cfg.variogram_z_grid_size)
    lo = vol.bbox[0, :2].astype(np.float64)
    hi = vol.bbox[1, :2].astype(np.float64)
    means: list[FloatArray] = []
    valid: list[npt.NDArray[np.bool_]] = []
    for section, block in zip(vol.sections, scores, strict=True):
        coords = np.asarray(section.coords, dtype=np.float64)
        index = np.clip(((coords - lo) / np.maximum(hi - lo, 1e-9) * grid).astype(int), 0, grid - 1)
        flat = index[:, 0] * grid + index[:, 1]
        occupancy = np.bincount(flat, minlength=grid * grid)
        summed = np.zeros((grid * grid, block.shape[1]))
        np.add.at(summed, flat, block)
        ok = occupancy >= cfg.variogram_z_min_cells_per_cell
        cell_mean = np.zeros_like(summed)
        cell_mean[ok] = summed[ok] / occupancy[ok, None]
        means.append(cell_mean)
        valid.append(ok)

    z_values = vol.z_values.astype(np.float64)
    lags: dict[float, tuple[float, float]] = {}
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            shared = valid[i] & valid[j]
            if not shared.any():
                continue
            lag = round(float(z_values[j] - z_values[i]), 6)
            semivariance = 0.5 * ((means[i][shared] - means[j][shared]) ** 2).sum(axis=1)
            total, weight = lags.get(lag, (0.0, 0.0))
            lags[lag] = (total + float(semivariance.sum()), weight + float(shared.sum()))
    if not lags:
        raise VariogramError(
            f"specimen {vol.specimen_id!r}: no in-plane grid cell is filled in two sections "
            f"at Config.variogram_z_grid_size={grid}; ell_z cannot be fitted"
        )
    ordered = sorted(lags)
    counts = np.array([lags[lag][1] for lag in ordered])
    gamma = np.array([lags[lag][0] / lags[lag][1] for lag in ordered])
    keep = counts >= cfg.variogram_min_pairs_per_bin
    if keep.sum() < 3:
        raise VariogramError(
            f"specimen {vol.specimen_id!r}: only {int(keep.sum())} usable section lags; "
            "at least 3 are needed to fit a nugget, a sill and a length-scale along z"
        )
    return np.array(ordered)[keep], gamma[keep], counts[keep]


def _subsample(n: int, cap: int, gen: np.random.Generator) -> npt.NDArray[np.intp]:
    """Return sorted indices of at most ``cap`` cells; all of them when ``n <= cap``."""
    if n <= cap:
        return np.arange(n, dtype=np.intp)
    return np.sort(gen.choice(n, size=cap, replace=False)).astype(np.intp)


def _accumulate_bins(
    distance: FloatArray,
    semivariance: FloatArray,
    max_lag: float,
    n_bins: int,
    lag_sum: FloatArray,
    gamma_sum: FloatArray,
    counts: FloatArray,
) -> None:
    """Add one section's pairs into the shared distance bins, in place."""
    inside = (distance > 0.0) & (distance <= max_lag)
    if not inside.any():
        return
    index = np.minimum((distance[inside] / max_lag * n_bins).astype(int), n_bins - 1)
    np.add.at(lag_sum, index, distance[inside])
    np.add.at(gamma_sum, index, semivariance[inside])
    np.add.at(counts, index, 1.0)


def _finish_bins(
    lag_sum: FloatArray,
    gamma_sum: FloatArray,
    counts: FloatArray,
    cfg: Config,
    *,
    axis: str,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Turn the accumulators into (lag, semivariance, weight), dropping thin bins."""
    keep = counts >= cfg.variogram_min_pairs_per_bin
    if keep.sum() < 3:
        raise VariogramError(
            f"{axis} variogram: only {int(keep.sum())} of {counts.size} distance bins hold "
            f"Config.variogram_min_pairs_per_bin={cfg.variogram_min_pairs_per_bin} pairs; "
            "raise Config.variogram_max_cells_per_section or lower the bin count"
        )
    return lag_sum[keep] / counts[keep], gamma_sum[keep] / counts[keep], counts[keep]


def _fit_matern_variogram(
    lags: FloatArray,
    gamma: FloatArray,
    counts: FloatArray,
    cfg: Config,
    *,
    lo: float,
    hi: float,
    axis: str,
) -> tuple[float, float, float, float]:
    """Least-squares fit of ``gamma(h) = nugget + sill * (1 - matern(h / ell))``.

    The two linear parameters have a closed form given ``ell``, so the fit is a scan over
    a log-spaced grid of ``cfg.variogram_n_ell_grid`` length-scales with a weighted 2x2
    solve inside — boring, has no starting point to get wrong, and cannot land in a local
    minimum.

    Bins are weighted by Cressie's ``N(h) / gamma(h)^2`` rather than by ``N(h)`` alone.
    The semivariogram estimator's variance grows roughly with ``gamma(h)^2``, so pair
    counts on their own make the fit a fit to the *large* lags — where the pairs are, and
    where a real tissue's large-scale trend lives — and bias the length-scale. Measured on
    the synthetic fixture, whose ground-truth in-plane length is 120 um: pair-count weights
    give 137.6 um at a 1000 um field of view and 95.2 um at 3000 um, Cressie's give
    137.6 um and 106.6 um. Better at the wider field of view, neutral at the narrower one.

    Returns ``(ell, nugget, sill, rmse)``.
    """
    if not 0.0 < lo < hi:
        raise VariogramError(f"{axis} variogram: empty length-scale search range [{lo}, {hi}]")
    weights = counts / np.maximum(gamma, _VARIOGRAM_WEIGHT_FLOOR) ** 2
    grid = np.geomspace(lo, hi, int(cfg.variogram_n_ell_grid))
    best: tuple[float, float, float, float] | None = None
    best_sse = math.inf
    for ell in grid:
        shape = 1.0 - matern_correlation(lags / ell, cfg.matern_nu)
        nugget, sill = _weighted_nnls2(shape, gamma, weights)
        residual = gamma - (nugget + sill * shape)
        sse = float((weights * residual**2).sum())
        if sse < best_sse:
            best_sse = sse
            best = (float(ell), nugget, sill, math.sqrt(sse / float(weights.sum())))
    assert best is not None  # the grid is non-empty, so some candidate always wins
    if best[2] < cfg.variogram_min_structured_frac:
        raise VariogramError(
            f"{axis} variogram: the fitted sill is {best[2]:.4f}, below "
            f"Config.variogram_min_structured_frac={cfg.variogram_min_structured_frac}. "
            "There is no spatial structure here to fit a correlation length to; a "
            "length-scale returned anyway would be an artefact of the grid."
        )
    import warnings

    if best[0] in (grid[0], grid[-1]):
        warnings.warn(
            f"{axis} length-scale fit landed on the edge of its search grid "
            f"[{lo:.3g}, {hi:.3g}] um at {best[0]:.3g} um: the value is a bound, not an "
            "optimum. Widen Config.variogram_ell_min_factor / variogram_ell_max_factor, "
            "or accept that this volume cannot resolve the correlation length.",
            LengthscaleFitWarning,
            stacklevel=2,
        )
    saturation = float(gamma.max()) / max(best[1] + best[2], 1e-12)
    if saturation < cfg.variogram_min_saturation:
        warnings.warn(
            f"{axis} variogram reaches only {saturation:.0%} of its fitted sill at the "
            f"largest lag ({lags.max():.3g} um), below "
            f"Config.variogram_min_saturation={cfg.variogram_min_saturation:.0%}. The "
            f"length-scale ({best[0]:.3g} um) is then extrapolated past the range the data "
            "covers and is a lower-confidence number: the volume is not big enough, in this "
            "direction, to see the correlation decay away.",
            LengthscaleFitWarning,
            stacklevel=2,
        )
    return best


def _weighted_nnls2(
    shape: FloatArray, gamma: FloatArray, weights: FloatArray
) -> tuple[float, float]:
    """Weighted least squares for ``gamma ~ nugget + sill * shape``, both constrained >= 0.

    Two parameters, so the projection onto the non-negative quadrant is exhaustive rather
    than iterative: solve unconstrained, and if a coefficient comes out negative, pin it
    at zero and solve the remaining one.
    """
    ones = np.ones_like(shape)
    a11 = float((weights * ones * ones).sum())
    a12 = float((weights * ones * shape).sum())
    a22 = float((weights * shape * shape).sum())
    b1 = float((weights * ones * gamma).sum())
    b2 = float((weights * shape * gamma).sum())
    determinant = a11 * a22 - a12 * a12
    if determinant > 0.0:
        nugget = (a22 * b1 - a12 * b2) / determinant
        sill = (a11 * b2 - a12 * b1) / determinant
        if nugget >= 0.0 and sill >= 0.0:
            return nugget, sill
    if a22 > 0.0:
        sill_only = max(b2 / a22, 0.0)
        if a11 > 0.0:
            nugget_only = max(b1 / a11, 0.0)
            shape_sse = float((weights * (gamma - sill_only * shape) ** 2).sum())
            flat_sse = float((weights * (gamma - nugget_only) ** 2).sum())
            return (0.0, sill_only) if shape_sse <= flat_sse else (nugget_only, 0.0)
        return 0.0, sill_only
    return max(b1 / a11, 0.0) if a11 > 0.0 else 0.0, 0.0
