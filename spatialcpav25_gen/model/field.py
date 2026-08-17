"""The anatomical field: a continuous ``F(x, y, z) -> R^{d_f}`` over the tissue volume.

The field carries *anatomy* — laminar structure, region boundaries, density gradients —
and is the conditioning signal both heads read. It is queryable at any point of the
bounding box, not only where cells were observed, which is what makes a virtual section a
query rather than an interpolation between point clouds.

Two frames, and the transform order between them
------------------------------------------------
Everything in this module is one of two frames, and confusing them is the silent bug T04
warns about:

``data frame``
    The tissue's own coordinates: the frame ``Volume.bbox`` is expressed in, and — this is
    the load-bearing part — **the frame whose z axis is the physical sectioning axis**.
``model frame``
    The frame the learned feature planes live in. Under rotation augmentation the volume is
    presented to the model in a random pose, so ``model = R (data - centre) + centre``.

The transform order, written out because a test asserts it
(``test_fourier_axis_follows_data_frame``)::

    p_data  --(R, applied by RotationContext)-->  p_model  --> triplane lookup
       |
       +--> fourier_encode(p_data)                            <-- NEVER the model frame

The anisotropic Fourier encoding spends 8 bands in-plane and only 2 along z because with
4-9 sections the high z frequencies are unconstrained. That justification is a statement
about **the sectioning axis**, so the encoding is computed in the data frame, *before* the
augmentation rotation. Computing it after would attach the low-frequency axis to whatever
direction the random rotation happened to pick, which is a bug that costs a little accuracy
every step and shows up in no error message.

What is and is not rotation-invariant, by construction
------------------------------------------------------
``RotationContext`` carries one rotation ``R`` across all four consumers of geometry, and
the four behave differently on purpose (SPEC_QUESTIONS B5, decided at T04):

===========================  ==========================================================
Channel                      Behaviour under the augmentation rotation
===========================  ==========================================================
``fourier_encode``           **Invariant** — evaluated in the data frame.
GRF query points             **Invariant** — the noise realisation is attached to the
                             tissue, so the field is queried in the data frame. Querying
                             it in the model frame would redraw the noise every step and
                             destroy the exact intersection consistency T03/T07 rest on.
retrieval neighbourhoods     **Invariant** — neighbour identity, weights and the
                             *relative* positions in the tokens are all data-frame
                             quantities, so a rotation cannot change them.
triplane lookup              **Not invariant, deliberately.** The feature planes are fixed
                             arrays in the model frame; rotating the volume writes the same
                             anatomy onto different texels each step, which is the entire
                             mechanism of ``design/v23_sectioning_equivariance.md`` §2.1(a).
===========================  ==========================================================

The spec asks for "a full forward pass is equivariant: rotating inputs and inverse-rotating
outputs returns (approximately) the same result". Taken literally that is unsatisfiable
*and* self-defeating: a lookup table indexed by fixed axes can only be invariant to a
continuous rotation if it undoes the rotation, and a triplane that undoes the rotation makes
the augmentation an exact no-op — the field would then be trained no differently with the
augmentation on or off, and GATE 2 would measure the multi-orientation ensemble alone while
appearing to test both fixes. ``test_rotation_equivariance`` therefore asserts the contract
above in both directions: the three invariant channels agree to 1e-3 relative, and the
triplane channel is asserted **not** to agree, as a negative control that the augmentation
is live. See ``SPEC_QUESTIONS.md`` B5.

Why the multi-orientation ensemble exists
-----------------------------------------
A single axis-aligned triplane set concentrates representational capacity on the
axis-aligned planes: the XY plane is ``res_xy^2`` and the XZ / YZ planes are
``res_xy x res_z`` with ``res_z`` an order of magnitude smaller, so structure that varies
along z is resolved an order of magnitude more coarsely than structure that varies in
plane. Oblique sections mix the two and come out systematically blurrier than coronal ones.
``cfg.n_plane_orientations`` triplane sets at fixed, maximally separated rotations, **summed**
(not concatenated — the sum keeps ``d_f`` fixed and makes the ensemble a smoothing over
directions), spread that capacity over the sphere. This is GATE 2.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from types import TracebackType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from spatialcpav25_gen.config import ROTATION_BIASES, Config

__all__ = [
    "BBoxClampWarning",
    "RotationContext",
    "RotationError",
    "TriplaneField",
    "fourier_encode",
    "fourier_encode_dim",
    "minimal_rotation",
    "orientation_rotations",
    "random_rotation",
]

FloatArray = npt.NDArray[np.float64]

ROTATION_CHANNELS: Final[tuple[str, ...]] = ("coords", "planes", "retrieval", "grf")
"""The four consumers of geometry the augmentation has to reach, in the order T04 lists
them. ``RotationContext`` refuses to exit if a channel it was told to expect was never
transformed."""

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
"""Angular step of the spherical Fibonacci lattice. Not a tunable: it is the irrational
number that makes the lattice maximally separated at every count."""

_DEGENERATE_AXIS_TOL = 1e-12
"""Below this, two unit vectors are parallel (or antiparallel) and the minimal rotation
between them needs the explicit branch rather than the cross product."""


class BBoxClampWarning(UserWarning):
    """A field query fell outside the bounding box and was clamped to its face.

    Not an error: T05 samples candidate positions from a slightly inflated support and T07
    cuts planes that graze the boundary, so the occasional outside point is expected. It is
    a warning because a *systematic* stream of them means the bbox the field was built with
    is not the bbox the caller is sampling in, and every clamped query returns the value at
    the tissue's surface instead.
    """


class RotationError(RuntimeError):
    """A :class:`RotationContext` left one of its channels un-rotated.

    Rotating the coordinates but forgetting the retrieval neighbourhoods (or the GRF query
    points, or the sampling planes) puts two halves of one training step in two different
    frames. Nothing raises, nothing is NaN, and the model quietly learns from a
    contradiction — which is why the context refuses to exit rather than trusting review.
    """


# --------------------------------------------------------------------------------------
# anisotropic Fourier encoding
# --------------------------------------------------------------------------------------


def fourier_encode_dim(cfg: Config) -> int:
    """Width of :func:`fourier_encode`'s output: ``2 * (2 * B_xy + B_z)``."""
    return 2 * (2 * int(cfg.fourier_bands_xy) + int(cfg.fourier_bands_z))


def fourier_encode(xyz_data_frame: Tensor, cfg: Config) -> Tensor:
    """Anisotropic Fourier positional encoding. ``(N, 3)`` -> ``(N, 2 * (2 B_xy + B_z))``.

    Parameters
    ----------
    xyz_data_frame
        ``(N, 3)`` coordinates **in the data frame, normalised to [-1, 1]** by the volume's
        bounding box. Both halves of that sentence matter.

        *Data frame*: the z axis of the input must be the tissue's true sectioning axis.
        ``cfg.fourier_bands_z`` is small because a stack of 4-9 sections cannot constrain
        high frequencies **along the sampling axis**; applied to a rotated frame the
        argument is simply false, and the encoding would spend its low-frequency budget on
        an arbitrary direction. Under rotation augmentation, call this with
        ``RotationContext.to_data(...)`` output — never with the model-frame coordinates
        the triplane lookup uses.

        *Normalised*: the bands are ``2^b * pi`` cycles over ``[-1, 1]``, so the caller
        divides out the bounding box (which is where the bbox lives — this function has no
        volume to ask). :class:`TriplaneField` does it; a direct caller must too.
    cfg
        Supplies ``fourier_bands_xy``, ``fourier_bands_z`` and ``debug_shapes``.

    Returns
    -------
    Tensor
        ``(N, 2 * (2 B_xy + B_z))`` float32, ordered
        ``[sin(f x), cos(f x)] for f in x-bands`` then the same for ``y`` then ``z``.
    """
    points = _as_points(xyz_data_frame)
    if cfg.debug_shapes:
        assert points.ndim == 2, points.shape
        assert points.shape[1] == 3, points.shape
    bands_xy = int(cfg.fourier_bands_xy)
    bands_z = int(cfg.fourier_bands_z)

    parts: list[Tensor] = []
    for axis, n_bands in ((0, bands_xy), (1, bands_xy), (2, bands_z)):
        if n_bands == 0:
            continue
        # 2^b pi: one, two, four ... cycles over the normalised extent. The standard
        # octave ladder, so band b resolves structure of size extent / 2^b.
        freqs = (2.0 ** torch.arange(n_bands, dtype=points.dtype, device=points.device)) * math.pi
        angle = points[:, axis : axis + 1] * freqs[None, :]
        parts.extend((torch.sin(angle), torch.cos(angle)))

    out = torch.cat(parts, dim=1) if parts else points.new_zeros((points.shape[0], 0))
    if cfg.debug_shapes:
        assert out.shape == (points.shape[0], fourier_encode_dim(cfg)), out.shape
    return out


# --------------------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------------------


def random_rotation(seed: int, bias: str = "uniform") -> FloatArray:
    """Draw one rotation matrix. Returns ``(3, 3)`` float64 with ``det = +1``.

    Parameters
    ----------
    seed
        Explicit, as Convention 3 requires: two calls with the same seed return bitwise
        identical matrices, in this process or any other.
    bias
        ``"uniform"`` draws from the Haar measure on SO(3) — the sign-corrected QR
        factorisation of a Gaussian matrix, which is exactly uniform and (unlike a random
        quaternion) needs no rejection step.

        ``"axial"`` is the anatomically plausible bias of
        ``design/v23_sectioning_equivariance.md`` §2.1(a): a free spin about the sectioning
        axis composed with a tilt of at most ``Config.rotation_bias_max_tilt_deg`` away from
        it. A block is mounted at an arbitrary rotation in the cutting plane but only
        slightly off-axis, so this covers the poses real data occupies rather than the whole
        sphere. It takes its cap from ``Config``; ``uniform`` takes nothing, which is why
        this function keeps the spec's ``cfg``-free signature.

    Notes
    -----
    ``"axial"`` reads ``Config.rotation_bias_max_tilt_deg`` from a default ``Config()``.
    That is the one place in the package where a constant is read from a config the caller
    did not pass, and it is forced by the spec's signature; :meth:`RotationContext.random`
    takes the ``cfg`` the caller is actually using and is the sanctioned entry point.
    """
    if bias not in ROTATION_BIASES:
        raise ValueError(
            f"random_rotation: bias must be one of {sorted(ROTATION_BIASES)}, got {bias!r}"
        )
    gen = np.random.default_rng(seed)
    if bias == "uniform":
        return _haar_rotation(gen)

    max_tilt = math.radians(Config().rotation_bias_max_tilt_deg)
    return _axial_rotation(gen, max_tilt)


def _haar_rotation(gen: np.random.Generator) -> FloatArray:
    """Uniform (Haar) rotation from the sign-corrected QR of a Gaussian matrix."""
    q, r = np.linalg.qr(gen.standard_normal((3, 3)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    rotation = q * signs[None, :]
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] = -rotation[:, 0]
    return cast(FloatArray, rotation)


def _axial_rotation(gen: np.random.Generator, max_tilt: float) -> FloatArray:
    """Free spin about +z composed with a tilt of at most ``max_tilt`` away from it."""
    spin = gen.uniform(0.0, 2.0 * math.pi)
    azimuth = gen.uniform(0.0, 2.0 * math.pi)
    # cos(tilt) uniform on [cos(max_tilt), 1] makes the tilted axis uniform on the
    # spherical cap rather than clustered at the pole.
    cos_tilt = gen.uniform(math.cos(max_tilt), 1.0)
    sin_tilt = math.sqrt(max(1.0 - cos_tilt**2, 0.0))
    target = np.array(
        [sin_tilt * math.cos(azimuth), sin_tilt * math.sin(azimuth), cos_tilt], dtype=np.float64
    )
    spin_matrix = np.array(
        [
            [math.cos(spin), -math.sin(spin), 0.0],
            [math.sin(spin), math.cos(spin), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return cast(FloatArray, minimal_rotation(np.array([0.0, 0.0, 1.0]), target) @ spin_matrix)


def minimal_rotation(source: FloatArray, target: FloatArray) -> FloatArray:
    """Return the smallest-angle rotation taking unit vector ``source`` to ``target``.

    ``(3,)``, ``(3,)`` -> ``(3, 3)`` float64 with ``det = +1``. Public because T07 builds a
    section's **pose** with it: the frame in which a plane is the cutting plane is the frame
    that takes the plane's normal to ``+z``, and ``infer.planes.plane_pose`` is that one
    line. Deterministic and involution-free — the antiparallel case picks the half turn about
    the least-aligned coordinate axis rather than an arbitrary one.
    """
    axis = np.cross(source, target)
    sin_angle = float(np.linalg.norm(axis))
    cos_angle = float(source @ target)
    if sin_angle < _DEGENERATE_AXIS_TOL:
        if cos_angle > 0.0:
            return np.eye(3, dtype=np.float64)
        # Antiparallel: a half turn about any perpendicular axis; take the least-aligned
        # coordinate axis, which is the numerically stable choice.
        perpendicular = np.zeros(3)
        perpendicular[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, perpendicular)
        axis /= np.linalg.norm(axis)
        return cast(FloatArray, 2.0 * np.outer(axis, axis) - np.eye(3))
    axis = axis / sin_angle
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    angle = math.atan2(sin_angle, cos_angle)
    return cast(
        FloatArray,
        np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross),
    )


def orientation_rotations(n: int) -> FloatArray:
    """``(P, 3, 3)`` fixed, maximally separated triplane orientations. Deterministic.

    Orientation 0 is the identity — the *unrotated* set, which is the one
    :meth:`TriplaneField.tv_z_penalty` regularises, since it is the only one whose third
    axis is the tissue's sectioning axis.

    The remaining ``P - 1`` come from a spherical Fibonacci lattice on the upper
    hemisphere: the minimal rotation taking each lattice direction to ``+z``. The lattice
    is used rather than the spec's tetrahedron because it is defined at *every* ``P`` — and
    GATE 2's first remedy is raising ``n_plane_orientations`` from 4 to 8, where "the
    tetrahedral set" has no meaning. At ``P = 4`` the two constructions agree to within a
    few degrees of separation. Antipodal directions are redundant (a triplane set and its
    point reflection sample the same three planes), so the lattice covers a hemisphere
    rather than the sphere.
    """
    if n < 1:
        raise ValueError(f"orientation_rotations: n must be >= 1, got {n}")
    index = np.arange(n, dtype=np.float64)
    z = 1.0 - index / float(n)
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    theta = _GOLDEN_ANGLE * index
    directions = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
    pole = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return np.stack([minimal_rotation(d, pole) for d in directions], axis=0)


class RotationContext:
    """One rotation, applied to the whole volume, with the channels it must reach.

    The augmentation is only correct if **every** consumer of geometry sees the same
    rotation. Rotating the coordinates and forgetting the retrieval neighbourhoods does not
    raise, does not produce a NaN, and leaves the model learning from two halves of a step
    that disagree about where the tissue is. So the context tracks which channels were used
    and refuses to exit with any of ``requires`` untouched.

    Usage::

        with RotationContext.random(cfg, seed=step, centre=vol.bbox.mean(0)) as rot:
            xyz_model = rot.coords(xyz_data)  # data -> model
            xi = grf(rot.grf(xyz_model))  # model -> data, then query
            idx, w = index.query(rot.retrieval(xyz_model), ...)
            origins, normals = rot.planes(origins, normals)
            f = field(xyz_model)  # the one model-frame consumer

    Args:
        cfg:      supplies ``debug_shapes``; ``rotation_bias`` and
                  ``rotation_bias_max_tilt_deg`` when built by :meth:`random`.
        rotation: ``(3, 3)`` the rotation taking data-frame vectors to model-frame ones.
        centre:   ``(3,)`` the point held fixed, normally the volume's bbox centre. A
                  rotation about the origin of coordinates would translate a tissue that
                  does not straddle the origin clean out of its own bounding box.
        requires: the channels that must be used before the block ends. Defaults to all
                  four; a caller that genuinely has no sampling planes (the GATE 2 probe,
                  which reconstructs cells rather than cutting sections) names the three it
                  does use, and that omission is then explicit and reviewable rather than
                  accidental.
        fields:   :class:`TriplaneField` instances to bind for the duration of the block.
                  Binding is what lets ``field.forward`` recover the data frame for its
                  Fourier encoding; the previous rotation is restored on exit even if the
                  body raises.
    """

    def __init__(
        self,
        cfg: Config,
        rotation: npt.ArrayLike,
        centre: npt.ArrayLike,
        *,
        requires: Sequence[str] = ROTATION_CHANNELS,
        fields: Sequence[TriplaneField] = (),
    ) -> None:
        self.cfg = cfg
        self.matrix = _check_rotation(rotation)
        self.centre = np.asarray(centre, dtype=np.float64).reshape(3)
        unknown = sorted(set(requires) - set(ROTATION_CHANNELS))
        if unknown:
            raise ValueError(
                f"RotationContext: unknown channel(s) {unknown}; expected a subset of "
                f"{list(ROTATION_CHANNELS)}"
            )
        self.requires = tuple(requires)
        self.fields = tuple(fields)
        self.used: set[str] = set()
        self._entered = False
        self._saved: list[FloatArray] = []

    # ----------------------------------------------------------------------------------
    # construction
    # ----------------------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        cfg: Config,
        seed: int,
        centre: npt.ArrayLike,
        *,
        requires: Sequence[str] = ROTATION_CHANNELS,
        fields: Sequence[TriplaneField] = (),
    ) -> RotationContext:
        """Draw the step's rotation from ``cfg.rotation_bias``, or identity if augmentation is off.

        The sanctioned entry point: unlike :func:`random_rotation` it reads the bias and its
        cap from the ``cfg`` the caller is actually using, and it honours
        ``cfg.rotation_aug``, so switching the augmentation off is one config field rather
        than a branch at every call site.
        """
        rotation = (
            _rotation_from_cfg(cfg, seed) if cfg.rotation_aug else np.eye(3, dtype=np.float64)
        )
        return cls(cfg, rotation, centre, requires=requires, fields=fields)

    @classmethod
    def identity(
        cls,
        cfg: Config,
        centre: npt.ArrayLike = (0.0, 0.0, 0.0),
        *,
        requires: Sequence[str] = ROTATION_CHANNELS,
        fields: Sequence[TriplaneField] = (),
    ) -> RotationContext:
        """Build the un-rotated context: the reference arm of every equivariance comparison."""
        return cls(cfg, np.eye(3, dtype=np.float64), centre, requires=requires, fields=fields)

    @property
    def is_identity(self) -> bool:
        """True when this context applies no rotation at all."""
        return bool(np.array_equal(self.matrix, np.eye(3)))

    # ----------------------------------------------------------------------------------
    # the context manager
    # ----------------------------------------------------------------------------------

    def __enter__(self) -> RotationContext:
        """Bind the rotation to every registered field and start tracking channel use."""
        self._entered = True
        self.used.clear()
        self._saved = [np.array(f.rotation_matrix, dtype=np.float64) for f in self.fields]
        for field in self.fields:
            field.set_rotation(self.matrix, self.centre)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the fields' previous rotation, then check no required channel was missed."""
        for field, saved in zip(self.fields, self._saved, strict=True):
            field.set_rotation(saved, self.centre)
        self._entered = False
        if exc_type is not None:
            # An exception in the body is the reason a channel is missing; reporting the
            # missing channel here would replace the real error with a symptom of it.
            return
        missing = [c for c in self.requires if c not in self.used]
        if missing:
            raise RotationError(
                f"RotationContext left {missing} un-rotated (used: {sorted(self.used)}). "
                "Every consumer of geometry must see the same rotation: coordinates, "
                "sampling planes, retrieval neighbourhoods and GRF query points. If this "
                "caller genuinely has no such channel, name the ones it does have in "
                "RotationContext(requires=...) so the omission is explicit."
            )

    # ----------------------------------------------------------------------------------
    # the four channels
    # ----------------------------------------------------------------------------------

    def coords(self, xyz_data: npt.ArrayLike) -> FloatArray:
        """Cell coordinates, data frame -> model frame. ``(N, 3)`` -> ``(N, 3)`` float64."""
        self._mark("coords")
        return self.to_model(xyz_data)

    def planes(
        self, origins: npt.ArrayLike, normals: npt.ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        """Map sampling planes, data frame -> model frame. ``(P, 3)``, ``(P, 3)`` -> the same.

        Origins are points and rotate about the centre; normals are directions and only
        rotate. Handing both to one method is the point: a plane whose origin moved and
        whose normal did not is a different plane, and the mistake is invisible downstream.
        """
        self._mark("planes")
        rotated_origins = self.to_model(origins)
        directions = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
        return rotated_origins, cast(FloatArray, directions @ self.matrix.T)

    def retrieval(self, xyz_model: npt.ArrayLike) -> FloatArray:
        """Map retrieval query points, model frame -> data frame. ``(N, 3)`` -> ``(N, 3)``.

        The retrieval index holds *real cells at their real positions*, so a query has to
        be asked in the tissue's own frame. Neighbour identity, the score's three terms and
        the tokens' relative positions are then all invariant to the augmentation, which is
        what the design wants: rotating the volume must not change which real cells are
        near a point.
        """
        self._mark("retrieval")
        return self.to_data(xyz_model)

    def grf(self, xyz_model: npt.ArrayLike) -> FloatArray:
        """GRF query points, model frame -> data frame. ``(N, 3)`` -> ``(N, 3)``.

        The noise realisation is a property of the tissue, not of the pose it is presented
        in. Querying the GRF in the model frame would hand it a rotated point every step,
        i.e. a fresh realisation each time, and the exact intersection consistency T03
        establishes (and T07's ``L_cross`` relies on) would hold only within a step.
        """
        self._mark("grf")
        return self.to_data(xyz_model)

    # ----------------------------------------------------------------------------------
    # the raw transforms (no channel bookkeeping)
    # ----------------------------------------------------------------------------------

    def to_model(self, xyz_data: npt.ArrayLike) -> FloatArray:
        """``R (p - centre) + centre``. ``(N, 3)`` -> ``(N, 3)`` float64."""
        points = np.asarray(xyz_data, dtype=np.float64).reshape(-1, 3)
        return cast(FloatArray, (points - self.centre) @ self.matrix.T + self.centre)

    def to_data(self, xyz_model: npt.ArrayLike) -> FloatArray:
        """``R^T (q - centre) + centre``, the inverse of :meth:`to_model`."""
        points = np.asarray(xyz_model, dtype=np.float64).reshape(-1, 3)
        return cast(FloatArray, (points - self.centre) @ self.matrix + self.centre)

    def _mark(self, channel: str) -> None:
        """Record that ``channel`` was transformed inside this context."""
        self.used.add(channel)


def _rotation_from_cfg(cfg: Config, seed: int) -> FloatArray:
    """Draw a rotation with the bias and cap ``cfg`` specifies."""
    gen = np.random.default_rng(seed)
    if cfg.rotation_bias == "uniform":
        return _haar_rotation(gen)
    return _axial_rotation(gen, math.radians(cfg.rotation_bias_max_tilt_deg))


def _check_rotation(rotation: npt.ArrayLike) -> FloatArray:
    """Validate a ``(3, 3)`` proper rotation and return it as float64."""
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {matrix.shape}")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthogonal (R R^T = I)")
    if float(np.linalg.det(matrix)) < 0.0:
        raise ValueError("rotation must be proper (det = +1); a reflection is not a pose")
    return matrix


# --------------------------------------------------------------------------------------
# the field
# --------------------------------------------------------------------------------------


class TriplaneField(nn.Module):
    """``F(x, y, z) -> (N, d_f)``. Continuous, queryable anywhere in the bbox.

    Args:
        cfg:  supplies ``n_plane_orientations`` (P), ``triplane_res_xy``,
              ``triplane_res_z``, ``triplane_channels`` (C), ``fourier_bands_xy/_z``,
              ``field_mlp_hidden``, ``field_mlp_layers``, ``field_dim`` (d_f), ``seed``
              and ``debug_shapes``.
        bbox: ``(2, 3)`` float32 min/max over ``(x, y, z)`` in micrometres, in the **data
              frame** — normally ``Volume.bbox``.
    Returns from forward: ``(N, d_f)``.

    Structure
    ---------
    ``P = cfg.n_plane_orientations`` triplane sets, each at a fixed rotation from
    :func:`orientation_rotations`. Each set is three feature planes — ``XY`` at
    ``(res_xy, res_xy)``, ``XZ`` and ``YZ`` at ``(res_z, res_xy)`` — with ``C`` channels,
    read by ``grid_sample(align_corners=False)``. Per orientation the query is rotated,
    normalised to ``[-1, 1]`` against *that orientation's* bounding box, and the three
    planes are sampled and concatenated to ``3C``; the orientations are then **summed**.
    The anisotropic Fourier encoding of the data-frame position is concatenated and the
    result goes through a ``cfg.field_mlp_layers``-layer MLP of width
    ``cfg.field_mlp_hidden`` to ``d_f``.

    Frames
    ------
    :meth:`forward` takes **model-frame** coordinates (see the module docstring): under a
    :class:`RotationContext` those are ``R (p_data - centre) + centre``. The triplane lookup
    uses them as given — that is the augmentation. The Fourier encoding is computed on
    ``to_data(...)`` of the same points, so it follows the tissue's sectioning axis whatever
    the pose. With no context bound, the two frames coincide.

    Notes
    -----
    Parameters are initialised from a generator seeded with ``cfg.seed``, so two
    constructions with the same config are bitwise identical (Convention 3). Feature planes
    start at small noise rather than zero: an exactly zero plane has an exactly zero
    gradient through ``grid_sample`` for every query that lands on it, so the ensemble would
    never break its own symmetry and all P orientations would stay identical forever.
    """

    bbox: Tensor
    corners: Tensor
    orientations: Tensor
    rotation: Tensor
    centre: Tensor
    query_matrices: Tensor
    query_lo: Tensor
    query_span: Tensor

    def __init__(self, cfg: Config, bbox: npt.NDArray[Any]) -> None:
        super().__init__()
        self.cfg = cfg
        box = np.asarray(bbox, dtype=np.float64)
        if box.shape != (2, 3):
            raise ValueError(f"TriplaneField: bbox must be (2, 3) min/max, got {box.shape}")
        if not np.all(np.isfinite(box)):
            raise ValueError("TriplaneField: bbox must be finite")
        if not np.all(box[1] > box[0]):
            raise ValueError(
                f"TriplaneField: bbox must have positive extent on every axis, got "
                f"lo={box[0].tolist()} hi={box[1].tolist()}. A volume flat along one axis "
                "cannot be normalised to [-1, 1] on it."
            )

        self.register_buffer("bbox", torch.from_numpy(box.astype(np.float32)))
        self.register_buffer("corners", torch.from_numpy(_bbox_corners(box).astype(np.float32)))
        orientations = orientation_rotations(int(cfg.n_plane_orientations))
        self.register_buffer("orientations", torch.from_numpy(orientations.astype(np.float32)))
        self.register_buffer("rotation", torch.eye(3, dtype=torch.float32))
        self.register_buffer("centre", torch.from_numpy(box.mean(axis=0).astype(np.float32)))

        generator = torch.Generator().manual_seed(int(cfg.seed))
        res_xy = int(cfg.triplane_res_xy)
        res_z = int(cfg.triplane_res_z)
        channels = int(cfg.triplane_channels)
        # (P, C, H, W) per plane family. H is the second query axis, W the first, which is
        # the order grid_sample's (x, y) grid indexes: grid[..., 0] selects W.
        self.plane_xy = self._make_planes(cfg, (channels, res_xy, res_xy), generator)
        self.plane_xz = self._make_planes(cfg, (channels, res_z, res_xy), generator)
        self.plane_yz = self._make_planes(cfg, (channels, res_z, res_xy), generator)

        mlp_in = 3 * channels + fourier_encode_dim(cfg)
        layers: list[nn.Module] = []
        width = mlp_in
        for _ in range(int(cfg.field_mlp_layers) - 1):
            linear = nn.Linear(width, int(cfg.field_mlp_hidden))
            _init_linear(linear, generator)
            layers.extend((linear, nn.GELU()))
            width = int(cfg.field_mlp_hidden)
        out = nn.Linear(width, int(cfg.field_dim))
        _init_linear(out, generator)
        layers.append(out)
        self.mlp = nn.Sequential(*layers)

        self._refresh_query_frames()

    @staticmethod
    def _make_planes(
        cfg: Config, shape: tuple[int, int, int], generator: torch.Generator
    ) -> nn.Parameter:
        """``(P, C, H, W)`` feature planes, small-noise initialised (see the class docstring)."""
        n = int(cfg.n_plane_orientations)
        values = torch.empty((n, *shape))
        values.normal_(0.0, _PLANE_INIT_SD, generator=generator)
        return nn.Parameter(values)

    # ----------------------------------------------------------------------------------
    # frames
    # ----------------------------------------------------------------------------------

    @property
    def rotation_matrix(self) -> FloatArray:
        """``(3, 3)`` float64 copy of the currently bound augmentation rotation."""
        return cast(FloatArray, self.rotation.detach().numpy().astype(np.float64))

    def set_rotation(self, rotation: npt.ArrayLike, centre: npt.ArrayLike | None = None) -> None:
        """Bind the augmentation rotation. Called by :class:`RotationContext`, not by hand.

        Rebinding recomputes each orientation's query frame — the composed matrix
        ``R_p R`` and the bounding box of the data bbox under it — so a query never has to
        rotate eight corners itself.
        """
        matrix = _check_rotation(rotation)
        with torch.no_grad():
            self.rotation.copy_(torch.from_numpy(matrix.astype(np.float32)))
            if centre is not None:
                self.centre.copy_(torch.from_numpy(np.asarray(centre, dtype=np.float32).reshape(3)))
        self._refresh_query_frames()

    def _refresh_query_frames(self) -> None:
        """Recompute the per-orientation query matrices and their normalisation boxes.

        Orientation ``p`` samples at ``R_p q`` where ``q`` is the model-frame query, i.e. at
        ``R_p R (p_data - c) + ...``; the composed matrix is precomputed here. The
        normalisation box is the axis-aligned bounding box of the *data* bbox's eight
        corners under the same map, so the whole tissue always lands inside ``[-1, 1]``
        whatever the pose — a fixed box would let a rotated volume stick out of its own
        planes and clamp at the corners.
        """
        orientations = self.orientations.to(torch.float32)
        rotation = self.rotation.to(torch.float32)
        centre = self.centre.to(torch.float32)
        # A query arrives already in the model frame, so orientation p applies R_p alone.
        # The *box* is the data bbox seen through the whole composition R_p R, because the
        # corners this module stores are data-frame corners.
        composed = torch.einsum("pij,jk->pik", orientations, rotation)
        centred_corners = self.corners.to(torch.float32) - centre[None, :]
        mapped = torch.einsum("pij,nj->pni", composed, centred_corners)
        lo = mapped.min(dim=1).values
        hi = mapped.max(dim=1).values
        span = torch.clamp(hi - lo, min=_MIN_SPAN_UM)
        self.register_buffer("query_matrices", orientations.clone(), persistent=False)
        self.register_buffer("query_lo", lo, persistent=False)
        self.register_buffer("query_span", span, persistent=False)

    # ----------------------------------------------------------------------------------
    # evaluation
    # ----------------------------------------------------------------------------------

    def __call__(self, xyz: npt.NDArray[Any] | Tensor) -> Tensor:
        """``xyz``: ``(N, 3)`` model-frame micrometres -> ``(N, d_f)``."""
        return cast(Tensor, super().__call__(xyz))

    def forward(self, xyz: npt.NDArray[Any] | Tensor) -> Tensor:
        """``(N, 3)`` model-frame micrometres -> ``(N, d_f)`` float32.

        Differentiable with respect to ``xyz`` and to the feature planes. Queries outside
        the bounding box are clamped to its faces (``grid_sample`` with
        ``padding_mode="border"``) and warned about once per call
        (:class:`BBoxClampWarning`) rather than returning zeros or raising: T05 proposes
        positions from a slightly inflated support and T07 cuts planes that graze the
        boundary, so an occasional outside point is expected, while a stream of them means
        the caller's support and the field's bbox disagree.
        """
        points = _as_points(xyz)
        if self.cfg.debug_shapes:
            assert points.ndim == 2, points.shape
            assert points.shape[1] == 3, points.shape
        n_points = int(points.shape[0])
        self._warn_if_outside(points)

        features = self._sample_planes(points)
        encoding = fourier_encode(self._normalised_data_frame(points), self.cfg)
        out = cast(Tensor, self.mlp(torch.cat([features, encoding], dim=1)))
        if self.cfg.debug_shapes:
            assert out.shape == (n_points, int(self.cfg.field_dim)), out.shape
        return out

    def _sample_planes(self, points: Tensor) -> Tensor:
        """Sample and sum the ``P`` triplane sets. ``(N, 3)`` -> ``(N, 3C)``.

        Summed, not concatenated: the sum keeps ``d_f`` independent of ``P`` (so raising
        ``n_plane_orientations`` 4 -> 8, GATE 2's first remedy, does not change the MLP's
        input width and therefore does not change what the number means) and makes the
        ensemble a smoothing over directions rather than a wider basis.
        """
        centred = points - self.centre[None, :]
        # (P, N, 3): the query in each orientation's own frame.
        rotated = torch.einsum("pij,nj->pni", self.query_matrices, centred)
        normalised = 2.0 * (rotated - self.query_lo[:, None, :]) / self.query_span[:, None, :] - 1.0
        total: Tensor | None = None
        for p in range(normalised.shape[0]):
            q = normalised[p]
            sampled = torch.cat(
                [
                    _sample_plane(self.plane_xy[p], q[:, 0], q[:, 1]),
                    _sample_plane(self.plane_xz[p], q[:, 0], q[:, 2]),
                    _sample_plane(self.plane_yz[p], q[:, 1], q[:, 2]),
                ],
                dim=1,
            )
            total = sampled if total is None else total + sampled
        assert total is not None  # n_plane_orientations >= 1 is validated by Config
        if self.cfg.debug_shapes:
            assert total.shape == (points.shape[0], 3 * int(self.cfg.triplane_channels))
        return total

    def _normalised_data_frame(self, points: Tensor) -> Tensor:
        """Model-frame points -> data-frame points normalised to ``[-1, 1]``. ``(N, 3)``.

        **This is the transform order the module docstring writes out, and the one
        ``test_fourier_axis_follows_data_frame`` asserts.** The rotation is undone first,
        then the *data* bbox normalises — so the third component of what
        :func:`fourier_encode` sees is always depth along the sectioning axis, never
        whatever axis the augmentation happened to point at.
        """
        centred = points - self.centre[None, :]
        # R^T: undo the augmentation. Right-multiplying by R is (R^T x)^T for row vectors.
        data = centred @ self.rotation + self.centre[None, :]
        lo = self.bbox[0][None, :]
        span = torch.clamp(self.bbox[1] - self.bbox[0], min=_MIN_SPAN_UM)[None, :]
        return 2.0 * (data - lo) / span - 1.0

    def _warn_if_outside(self, points: Tensor) -> None:
        """Warn once per call if any query lies outside the (possibly rotated) bbox."""
        with torch.no_grad():
            centred = points - self.centre[None, :]
            data = centred @ self.rotation + self.centre[None, :]
            slack = _BBOX_TOLERANCE_FRAC * (self.bbox[1] - self.bbox[0])
            outside = (data < (self.bbox[0] - slack)[None, :]) | (
                data > (self.bbox[1] + slack)[None, :]
            )
            n_outside = int(outside.any(dim=1).sum())
        if n_outside:
            warnings.warn(
                f"TriplaneField: {n_outside} of {points.shape[0]} query point(s) lie outside "
                f"the bounding box lo={self.bbox[0].tolist()} hi={self.bbox[1].tolist()} and "
                "were clamped to its faces. Occasional outside points are expected (layout "
                "proposals, planes grazing the boundary); a systematic stream of them means "
                "the field's bbox is not the support the caller is sampling in.",
                BBoxClampWarning,
                stacklevel=3,
            )

    # ----------------------------------------------------------------------------------
    # regularisation
    # ----------------------------------------------------------------------------------

    def tv_z_penalty(self) -> Tensor:
        """Total variation along z of the XZ / YZ planes of the **unrotated** orientation.

        Scalar, to be weighted by ``cfg.tv_z_weight``. Only orientation 0 is penalised, and
        that is not a shortcut: it is the only set whose third axis *is* the tissue's
        sectioning axis, so it is the only one where "smooth along z" means "do not carve a
        step at each section's depth". The rotated sets have no meaningful z axis, and
        penalising variation along their third axis would suppress genuine in-plane
        structure that happens to have been rotated into it.
        """
        total = torch.zeros((), dtype=self.plane_xz.dtype)
        for planes in (self.plane_xz, self.plane_yz):
            # (P, C, res_z, res_xy): axis 2 is z.
            unrotated = planes[0]
            total = total + torch.abs(unrotated[:, 1:, :] - unrotated[:, :-1, :]).mean()
        return total

    def extra_repr(self) -> str:
        """Show the orientation count, plane sizes and output width."""
        return (
            f"P={int(self.cfg.n_plane_orientations)}, "
            f"res_xy={int(self.cfg.triplane_res_xy)}, res_z={int(self.cfg.triplane_res_z)}, "
            f"C={int(self.cfg.triplane_channels)}, d_f={int(self.cfg.field_dim)}"
        )


_PLANE_INIT_SD = 1e-2
"""Standard deviation of the feature planes at initialisation. Not a tunable: it only has
to be non-zero (see :class:`TriplaneField`'s notes) and small enough that the MLP starts in
its linear regime."""

_MIN_SPAN_UM = 1e-6
"""Floor on a bounding-box extent before dividing by it. Guards the division, not a knob."""

_BBOX_TOLERANCE_FRAC = 1e-5
"""How far outside the bbox a query may sit before it is reported, as a fraction of the
box's extent. Not a tunable: the bbox's faces *are* real cell coordinates, and a cell there
round-trips through the float32 rotation about 5e-5 um outside itself (Convention 4). Without
the tolerance every query containing an edge section would warn about rounding noise, and a
warning that fires on every call reports nothing."""


def _sample_plane(plane: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """Bilinear lookup in one feature plane. ``(C, H, W)``, two ``(N,)`` -> ``(N, C)``.

    ``u`` indexes the plane's width, ``v`` its height, both already normalised to
    ``[-1, 1]``; that is ``grid_sample``'s convention, where ``grid[..., 0]`` selects the
    last input axis. ``align_corners=False`` puts the ``[-1, 1]`` range at the outer edges
    of the corner texels rather than at their centres, which is what makes the lookup a
    partition of the box: with ``align_corners=True`` a query at the exact boundary lands
    half a texel outside.
    """
    grid = torch.stack([u, v], dim=1)[None, :, None, :]  # (1, N, 1, 2)
    sampled = torch.nn.functional.grid_sample(
        plane[None], grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return sampled[0, :, :, 0].transpose(0, 1)


def _bbox_corners(box: FloatArray) -> FloatArray:
    """Return the eight corners of a ``(2, 3)`` min/max box. ``(8, 3)`` float64."""
    return np.stack(
        [
            np.array([box[i, 0], box[j, 1], box[k, 2]])
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        axis=0,
    )


def _init_linear(layer: nn.Linear, generator: torch.Generator) -> None:
    """Re-initialise a linear layer from an explicit generator (Convention 3).

    The same construction ``model/embeddings.py`` uses: uniform on ``+/- 1/sqrt(fan_in)``,
    which is what ``nn.Linear.reset_parameters`` produces, but drawn from a seeded generator
    rather than from the global torch RNG.
    """
    bound = 1.0 / math.sqrt(layer.in_features)
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound, generator=generator)
        if layer.bias is not None:
            layer.bias.uniform_(-bound, bound, generator=generator)


def _as_points(xyz: npt.NDArray[Any] | Tensor) -> Tensor:
    """Coerce ``(N, 3)`` coordinates to a float32 tensor, preserving any gradient."""
    points = xyz if isinstance(xyz, Tensor) else torch.from_numpy(np.asarray(xyz))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got {tuple(points.shape)}")
    return points.to(torch.float32)
