"""GATE 2's four acceptance criteria, measured once and consumed twice.

``tests/test_field.py`` asserts the thresholds; ``scripts/gate2_report.py`` prints them into
``reports/gate2.md``. Keeping the measurement in one place is what stops the report and the
test from disagreeing about whether the gate passed — which, for a criterion that is
supposed to stop the project, would be the worst possible bug. The pattern (and the
``Criterion`` / ``GateSection`` types) is T03's, reused rather than re-implemented.

The probe
---------
The full generative heads do not exist yet, so GATE 2 trains a **lightweight probe** —
field + retrieval -> a linear head predicting the top-``expr_pca_dim`` expression PCs of
held-in cells — which isolates the backbone's representational quality from everything T05
and T06 will later add.

The evaluation set (settled: SPEC_QUESTIONS C1)
-----------------------------------------------
Real cells exist only on the sectioning planes, so an oblique query plane passes through
very few of them. Three rules make the angles comparable, and all three are part of the
criterion:

1. **Membership.** At angle ``theta`` the evaluation set is every training-section cell
   within ``thickness / 2`` of the query plane, pooled across all training sections.
2. **Equal ``n``.** Every angle's set is subsampled to the smallest of them with an explicit
   seed (:data:`SUBSAMPLE_SEED`). R^2 is a variance-explained ratio; both its sampling error
   and the mix of tissue it covers move with ``n``, so an unsubsampled comparison partly
   measures sample size. If the common ``n`` falls below
   ``Config.gate2_min_cells_per_angle``, this module **raises** — the fixture's slabs get
   thickened and the gate is re-run; the floor is never lowered and no angle is dropped.
3. **Leave-own-section-out retrieval.** Every evaluated cell's own source section is
   excluded from the retrieval candidate pool at every angle. Without it a cell in the 90
   degree strip retrieves in-plane neighbours a few micrometres away *inside its own
   section*, the oblique plane becomes trivially easy, and the gate passes while hiding
   exactly the equivariance failure it exists to detect.
   :func:`measure_exclusion_effect` is the standing check that the exclusion is still
   plumbed through.

What the angle does and does not change
---------------------------------------
Worth stating plainly, because it governs how the numbers should be read. The probe is a
function of *position*: the query plane is not an input to it. So the angle enters this
measurement through **membership only** — which cells are evaluated. That is not a
weakness of the contract, it is what makes the criterion a statement about the *field*:
R^2 is variance explained, and a 0 degree strip is one section (all of its target variance
is in-plane) while a 90 degree strip spans the whole stack (much of its target variance is
along z, the axis the triplane resolves at ``res_z`` rather than ``res_xy``). A backbone
whose z resolution lags its in-plane resolution therefore explains less of the 90 degree
strip's variance, and the ratio falls. The report says this in as many words.

The constants below (probe size, step count, sample sizes, seeds) are properties of the
*measurement*, not of the model, so they live here rather than in ``Config`` — a gate that
read its own sample size out of the config could be made to pass by editing the config. The
two exceptions are the ones the spec explicitly makes ``Config`` fields:
``gate2_min_cells_per_angle`` and ``retrieval_exclude_source_section``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from spatialcpav25_gen.config import Config
from spatialcpav25_gen.data.schema import Section, Volume
from spatialcpav25_gen.model.field import RotationContext, TriplaneField
from spatialcpav25_gen.model.retrieval import (
    RetrievalAttention,
    RetrievalIndex,
    attention_entropy,
)
from torch import Tensor, nn

from tests.gate1_criteria import Criterion, GateSection

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.intp]

# ---- the evaluation set ----------------------------------------------------------------
ANGLES_DEG: tuple[float, ...] = (0.0, 15.0, 30.0, 45.0, 60.0, 90.0)
"""The dihedral angles to the sectioning plane the spec names, verbatim."""

SUBSAMPLE_SEED = 20260815
"""Seed of the equal-``n`` subsample. Reported in ``reports/gate2.md`` as C1b requires: a
subsample without a stated seed is not a reproducible evaluation set."""

# ---- the probe -------------------------------------------------------------------------
PROBE_STEPS = 240
"""Optimiser steps per probe. Enough that the probe's R^2 has stopped moving materially
(the gate is a *ratio* between angles, and both arms see the same budget), and small enough
that four probes fit in a few minutes on four CPU cores."""
PROBE_BATCH = 2048
PROBE_LR = 3e-3
PROBE_SEED = 7
"""Base seed of the probe: parameter init, batch sampling and the per-step rotation all
derive from it (Convention 3)."""
PROBE_EVAL_CHUNK = 8192
"""Cells per forward pass at evaluation time. Only a memory knob."""

# ---- thresholds, verbatim from the spec ------------------------------------------------
OBLIQUE_PARITY_THRESHOLD = 0.90
"""G2.1: ``min_angle R^2 >= 0.90 x R^2(0 deg)``. **The gate.**"""
Z_INTERPOLATION_THRESHOLD = 0.80
"""G2.2: R^2 at the held-out z >= 0.8 x the mean R^2 at the neighbouring training z."""
ENTROPY_FRACTION_THRESHOLD = 0.5
"""G2.4: mean attention entropy > 0.5 log K."""

# ---- G2.3's fractional depths ----------------------------------------------------------
FRACTIONAL_DEPTHS: tuple[float, ...] = (0.2, 0.5, 0.8)
"""Where the evaluated section sits between the two sections left in the candidate pool.

Realised by exclusion rather than by moving cells: **every section but the two designated
flanks is excluded**, so the evaluated cells, their targets and the probe are identical
across the three fractions and only the *asymmetry of the evidence* changes. For section
``k``: 0.2 keeps ``k-1`` and ``k+4``, 0.8 keeps ``k-4`` and ``k+1``, 0.5 keeps ``k-1`` and
``k+1``.

Leaving only two sections in the pool is what makes the criterion test the mechanism it
names — a query genuinely one-fifth of the way between two sections, which is the situation
the competing method's score cannot see. With the whole stack available the term has far
less to do, because the *nearest* section is admissible at every fraction and in-plane
distance alone already ranks it first; measured that way the three deltas are +0.0004,
+0.0034 and +0.0019, i.e. inside the noise. That measurement is kept as diagnostic G2.3c,
because it is the honest bound on what the term buys when evidence is dense, and it says the
term earns its place in the *wide-gap* regime specifically."""

G23_Z_WINDOW = 5.0
"""``retrieval_z_window`` used for G2.3's evaluation only, in units of median spacing.

The 0.2 and 0.8 configurations put one flank four spacings away, outside the default window
of 3. Left at 3 the far flank would be filtered out before the score ever saw it, and the
ablation would be measuring ``retrieval_z_window`` rather than ``retrieval_w_z``. Applied
identically to both arms."""

G23_TIE_TOLERANCE = 0.01
"""How much R^2 at the symmetric depth 0.5 may move before "barely affecting 0.5" is false.
Absolute R^2 units."""


def _seed_for(*parts: int) -> int:
    """Derive an independent seed from a list of integers, reproducibly."""
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


# --------------------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------------------


class Probe(nn.Module):
    """Field + retrieval cross-attention -> a linear read-out of expression PCs.

    Deliberately the smallest thing that exercises both branches: any nonlinearity after
    the concatenation would let the head compensate for a directionally biased field, and
    GATE 2 would then measure the head.
    """

    def __init__(
        self, cfg: Config, bbox: npt.NDArray[Any], token_dim: int, target_dim: int
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = TriplaneField(cfg, bbox)
        self.attention = RetrievalAttention(cfg, q_dim=int(cfg.field_dim), token_dim=token_dim)
        self.head = nn.Linear(int(cfg.field_dim) + int(cfg.retrieval_ctx_dim), target_dim)
        generator = torch.Generator().manual_seed(int(cfg.seed))
        bound = 1.0 / math.sqrt(self.head.in_features)
        with torch.no_grad():
            self.head.weight.uniform_(-bound, bound, generator=generator)
            self.head.bias.zero_()

    def forward(self, xyz_model: Tensor, tokens: Tensor, mask: Tensor) -> Tensor:
        """``(N, 3)``, ``(N, K, d_token)``, ``(N, K)`` -> ``(N, target_dim)``."""
        features = self.field(xyz_model)
        ctx = self.attention(features, tokens, mask)
        out: Tensor = self.head(torch.cat([features, ctx], dim=1))
        return out


@dataclass(frozen=True)
class TrainedProbe:
    """A trained probe together with the index and targets it was trained against."""

    probe: Probe
    index: RetrievalIndex
    targets: npt.NDArray[np.float32]
    """``(N, expr_pca_dim)`` PC scores of the index's own cells."""
    neighbours: IntArray
    """``(N, K)`` retrieval result for every cell of the index, own section excluded."""
    losses: list[float]
    """Training loss every tenth step, for the report's convergence note."""


def _retrieve_all(index: RetrievalIndex, *, seed: int) -> IntArray:
    """Retrieve neighbours for every cell of the index, with its own section excluded."""
    idx, _ = index.query(
        index.coords,
        set(),
        seed=seed,
        source_section=index.section_index_of_cells(),
    )
    return idx


def train_probe(
    cfg: Config,
    vol: Volume,
    *,
    seed: int,
    steps: int = PROBE_STEPS,
) -> TrainedProbe:
    """Train the GATE 2 probe on every cell of ``vol``.

    Rotation augmentation is live: each step draws a rotation and the coordinates are
    presented to the field in that pose. The retrieval tokens are *not* recomputed per step
    because they are rotation-invariant by construction (they are data-frame quantities), so
    the context is asked for the one channel this caller transforms and the omission of the
    others is explicit rather than accidental — which is exactly what
    ``RotationContext(requires=...)`` is for.
    """
    index = RetrievalIndex(vol, cfg)
    neighbours = _retrieve_all(index, seed=_seed_for(seed, 1))
    targets = index.expr_pca
    probe = Probe(cfg, vol.bbox, index.token_dim, targets.shape[1])
    optimiser = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
    target_tensor = torch.from_numpy(targets)
    centre = vol.bbox.mean(axis=0).astype(np.float64)
    gen = np.random.default_rng(_seed_for(seed, 2))

    losses: list[float] = []
    for step in range(steps):
        pick = gen.choice(index.n_cells, size=min(PROBE_BATCH, index.n_cells), replace=False)
        xyz = index.coords[pick]
        tokens, mask = index.neighbour_tokens(xyz, neighbours[pick])
        with RotationContext.random(
            cfg,
            _seed_for(seed, 3, step),
            centre,
            requires=("coords",),
            fields=(probe.field,),
        ) as rot:
            xyz_model = torch.from_numpy(rot.coords(xyz).astype(np.float32))
            prediction = probe(xyz_model, tokens, mask)
            loss = torch.mean((prediction - target_tensor[pick]) ** 2)
            loss = loss + cfg.tv_z_weight * probe.field.tv_z_penalty()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        if step % 10 == 0:
            losses.append(float(loss.detach()))
    return TrainedProbe(
        probe=probe, index=index, targets=targets, neighbours=neighbours, losses=losses
    )


@torch.no_grad()
def predict(
    trained: TrainedProbe,
    xyz: FloatArray,
    neighbours: IntArray,
    *,
    index: RetrievalIndex | None = None,
) -> npt.NDArray[np.float32]:
    """Run the probe at ``xyz`` with the given neighbours. ``(n, 3)`` -> ``(n, target_dim)``.

    Evaluated with **no rotation**: the question is how well the model reconstructs this
    cell, and a random pose would add variance that has nothing to do with the criterion.
    """
    source = trained.index if index is None else index
    out: list[npt.NDArray[np.float32]] = []
    for start in range(0, xyz.shape[0], PROBE_EVAL_CHUNK):
        block = slice(start, start + PROBE_EVAL_CHUNK)
        tokens, mask = source.neighbour_tokens(xyz[block], neighbours[block])
        prediction = trained.probe(torch.from_numpy(xyz[block].astype(np.float32)), tokens, mask)
        out.append(prediction.numpy().astype(np.float32))
    return (
        np.concatenate(out, axis=0) if out else np.zeros((0, trained.targets.shape[1]), np.float32)
    )


@dataclass(frozen=True)
class Fit:
    """One evaluation set's residuals, and the two ``R^2`` they support.

    Two denominators, because they answer different questions and the difference is not
    cosmetic when the sets differ in composition:

    ``r2_set``
        ``1 - SSE / SST`` with ``SST`` about the **evaluation set's own** mean. The natural
        "how much of *this* set's variance is explained", and what the spec's
        ``R^2(theta) / R^2(0 deg)`` reads as. But it is *not* comparable across angles: a
        0 degree strip is a single section, whose target variance is entirely in-plane,
        while a 90 degree strip spans the stack and carries the along-z variance as well.
        The denominator moves with the angle, so a ratio of two of them is a ratio of two
        different questions.
    ``r2_fixed``
        ``1 - SSE / (n * V)`` with ``V`` the per-cell target variance over **all** training
        cells, fixed once and shared by every angle. The denominator is then a property of
        the volume, not of the strip, so the ratio compares like with like: it is a
        statement about the model's error per cell and nothing else.

    Both are reported. ``r2_fixed`` is the one that belongs in the paper.
    """

    sse: float
    """Residual sum of squares over the set, summed across all target dimensions."""
    n: int
    """Cells in the set."""
    sst_set: float
    """Total sum of squares about the set's own mean."""
    fixed_variance: float
    """``V``: per-cell target variance over all training cells."""

    @property
    def r2_set(self) -> float:
        """``1 - SSE / SST``, denominator from this set. Not comparable across sets."""
        return 1.0 - self.sse / self.sst_set if self.sst_set > 0.0 else float("nan")

    @property
    def r2_fixed(self) -> float:
        """``1 - SSE / (n * V)``, denominator fixed over the whole volume. Comparable."""
        total = self.n * self.fixed_variance
        return 1.0 - self.sse / total if total > 0.0 else float("nan")


def fit_residuals(
    prediction: npt.NDArray[np.float32],
    target: npt.NDArray[np.float32],
    fixed_variance: float,
) -> Fit:
    """Store one evaluation set's residuals so both ``R^2`` can be derived from them.

    Pooled over the PC dimensions rather than averaged per component: a per-component mean
    would let the many low-variance components, whose R^2 is noise, dominate the headline
    number.
    """
    if prediction.shape != target.shape:
        raise ValueError(f"fit_residuals: shape mismatch {prediction.shape} vs {target.shape}")
    if prediction.shape[0] < 2:
        raise ValueError("fit_residuals: an evaluation set needs at least 2 cells")
    return Fit(
        sse=float(((prediction - target) ** 2).sum()),
        n=int(prediction.shape[0]),
        sst_set=float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum()),
        fixed_variance=float(fixed_variance),
    )


def per_cell_variance(targets: npt.NDArray[np.float32]) -> float:
    """``V``: the target variance per cell, summed over the PC dimensions, over all cells.

    The fixed denominator every angle shares. Computed rather than assumed to be
    ``expr_pca_dim``: the PC scores *are* standardised to unit variance by
    ``ExpressionPCs``, so ``V`` comes out at the dimension count today, but a change to that
    normalisation must move this number rather than silently invalidate the comparison.
    """
    return float(targets.var(axis=0).sum())


def r_squared(prediction: npt.NDArray[np.float32], target: npt.NDArray[np.float32]) -> float:
    """``R^2`` with the set's own mean as the denominator. ``Fit.r2_set``, standalone.

    Kept for the within-set comparisons (G2.3's ablation arms, which are evaluated on
    *identical* cells and so share a denominator by construction).
    """
    return fit_residuals(prediction, target, fixed_variance=1.0).r2_set


# --------------------------------------------------------------------------------------
# the evaluation set
# --------------------------------------------------------------------------------------


def plane_normal(angle_deg: float) -> FloatArray:
    """Unit normal of a query plane at ``angle_deg`` to the sectioning plane. ``(3,)``.

    The sectioning plane's normal is ``+z``, so a dihedral angle of ``theta`` is the normal
    tilted by ``theta`` in the y-z plane. 0 degrees is coronal (parallel to the sections);
    90 degrees cuts straight through the stack.
    """
    angle = math.radians(float(angle_deg))
    return np.array([0.0, math.sin(angle), math.cos(angle)], dtype=np.float64)


def slab_half_thickness(vol: Volume) -> float:
    """``thickness / 2`` in micrometres — the evaluation set's membership radius."""
    return 0.5 * float(np.median([s.thickness for s in vol.sections]))


@dataclass(frozen=True)
class EvaluationSets:
    """The equal-``n`` evaluation sets, plus everything ``reports/gate2.md`` must state."""

    rows: dict[float, IntArray]
    """angle -> the subsampled row indices into the index."""
    pre_subsample_n: dict[float, int]
    """angle -> how many cells were within ``thickness / 2`` before subsampling."""
    common_n: int
    half_thickness: float
    seed: int


def evaluation_sets(
    index: RetrievalIndex, vol: Volume, cfg: Config, *, seed: int = SUBSAMPLE_SEED
) -> EvaluationSets:
    """Build the per-angle evaluation sets under the C1 contract.

    Raises
    ------
    ValueError
        If the common ``n`` falls below ``Config.gate2_min_cells_per_angle``. The remedy is
        to thicken the fixture's slabs and re-run — **not** to lower the floor and **not**
        to drop an angle, either of which would make the oblique-parity ratio partly a
        statement about sample size.
    """
    half = slab_half_thickness(vol)
    centre = vol.bbox.mean(axis=0).astype(np.float64)
    members: dict[float, IntArray] = {}
    for angle in ANGLES_DEG:
        offset = (index.coords - centre[None, :]) @ plane_normal(angle)
        members[angle] = np.nonzero(np.abs(offset) <= half)[0].astype(np.intp)

    pre = {angle: int(rows.size) for angle, rows in members.items()}
    common = min(pre.values())
    if common < int(cfg.gate2_min_cells_per_angle):
        thinnest = min(pre, key=lambda a: pre[a])
        raise ValueError(
            f"GATE 2 evaluation set: the common n is {common} cells (thinnest angle "
            f"{thinnest:g} deg), below Config.gate2_min_cells_per_angle="
            f"{cfg.gate2_min_cells_per_angle}. Thicken the fixture's slabs and re-run. Do "
            "not lower the floor and do not drop an angle: both make the oblique-parity "
            "ratio partly a statement about sample size (SPEC_QUESTIONS C1b)."
        )

    gen = np.random.default_rng(seed)
    rows = {
        angle: np.sort(gen.choice(member, size=common, replace=False)).astype(np.intp)
        for angle, member in members.items()
    }
    return EvaluationSets(
        rows=rows, pre_subsample_n=pre, common_n=common, half_thickness=half, seed=seed
    )


# --------------------------------------------------------------------------------------
# shared probe cache
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate2Probes:
    """The trained probes the four measurements share.

    Training a 10.5 M-parameter triplane four times over would put the gate report into the
    tens of minutes for no extra information: G2.1, G2.3 and G2.4 all read the *same*
    trained backbone and differ only in what they evaluate it on.
    """

    main: TrainedProbe
    """Trained on the whole volume at the default ``retrieval_w_z``. G2.1, G2.3, G2.4."""
    without_z: TrainedProbe
    """The same, with ``retrieval_w_z = 0`` — ablation A5. G2.3's second arm."""
    holdout: TrainedProbe
    """Trained with the middle section removed. G2.2."""
    holdout_section: Section
    """The section ``holdout`` never saw."""
    holdout_neighbour_z: tuple[float, float]
    """The depths of the two training sections flanking it."""


_PROBE_CACHE: dict[tuple[int, int, int, Config], Gate2Probes] = {}


def gate2_probes(cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS) -> Gate2Probes:
    """Train (or return the cached) probes for this volume, seed, step budget **and config**.

    ``cfg`` is part of the key, and that is not defensive: without it, comparing two configs
    in one process — which is exactly what GATE 2's own failure remedy asks for, raising
    ``n_plane_orientations`` 4 -> 8 and re-running — silently returns the first config's
    probes for the second, and the comparison reports "no change" for a change that was never
    made. ``Config`` is a frozen dataclass, so it hashes by value and this is exact.
    """
    key = (id(vol), int(seed), int(steps), cfg)
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached

    # The same training seed for both arms: identical initialisation, identical batch
    # order, identical per-step rotations. The only difference between the two probes is
    # then the retrieval score, which is what the ablation is supposed to be measuring —
    # with independent seeds the training-trajectory noise (~0.01 R^2 here) is comparable to
    # the effect.
    main = train_probe(cfg, vol, seed=_seed_for(seed, 10), steps=steps)
    without_z = train_probe(
        cfg.replace(retrieval_w_z=0.0), vol, seed=_seed_for(seed, 10), steps=steps
    )

    middle = len(vol.sections) // 2
    training = _volume_without(vol, middle)
    holdout = train_probe(cfg, training, seed=_seed_for(seed, 12), steps=steps)

    probes = Gate2Probes(
        main=main,
        without_z=without_z,
        holdout=holdout,
        holdout_section=vol.sections[middle],
        holdout_neighbour_z=(vol.sections[middle - 1].z, vol.sections[middle + 1].z),
    )
    _PROBE_CACHE[key] = probes
    return probes


def _volume_without(vol: Volume, index: int) -> Volume:
    """A copy of ``vol`` with section ``index`` removed, derived quantities recomputed."""
    return Volume(
        sections=[s for i, s in enumerate(vol.sections) if i != index],
        gene_names=list(vol.gene_names),
        celltype_names=list(vol.celltype_names),
        region_names=None if vol.region_names is None else list(vol.region_names),
        specimen_id=vol.specimen_id,
    )


# --------------------------------------------------------------------------------------
# G2.1 — oblique parity (the gate)
# --------------------------------------------------------------------------------------


def measure_g2_1(cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS) -> GateSection:
    """Reconstruction R^2 per dihedral angle, on the equal-``n`` evaluation set."""
    probes = gate2_probes(cfg, vol, seed=seed, steps=steps)
    trained = probes.main
    sets = evaluation_sets(trained.index, vol, cfg)
    variance = per_cell_variance(trained.targets)

    fits: dict[float, Fit] = {}
    for angle, rows in sets.rows.items():
        prediction = predict(trained, trained.index.coords[rows], trained.neighbours[rows])
        fits[angle] = fit_residuals(prediction, trained.targets[rows], variance)

    scores = {a: f.r2_set for a, f in fits.items()}
    fixed = {a: f.r2_fixed for a, f in fits.items()}
    ratio, worst_angle = _parity(scores)
    fixed_ratio, fixed_worst = _parity(fixed)
    baseline = scores[0.0]
    # How far apart the per-set denominators of G2.1a actually are: the reason G2.1d exists.
    per_cell_sst = [f.sst_set / f.n for f in fits.values()]
    spread = max(per_cell_sst) / min(per_cell_sst)

    exclusion = measure_exclusion_effect(cfg, vol, seed=seed, steps=steps)
    depth = measure_depth_mix(trained, sets, variance)
    mean_edge_share = float(np.mean([v for a, v in depth.edge_share.items() if a != 0.0]))
    return GateSection(
        key="G2.1",
        title="Oblique parity (the gate)",
        criteria=[
            Criterion(
                key="G2.1a",
                description=(
                    "min over oblique angles of R^2, as a fraction of R^2 at 0 deg "
                    f"(worst angle {worst_angle:g} deg), on {sets.common_n} cells per angle. "
                    "Per-set denominator, i.e. the spec's ratio as literally written"
                ),
                measured=ratio,
                threshold=OBLIQUE_PARITY_THRESHOLD,
                comparison=">=",
                note=(
                    "R^2 by angle: "
                    + ", ".join(f"{a:g} deg {scores[a]:.4f}" for a in ANGLES_DEG)
                    + "; pre-subsample n: "
                    + ", ".join(f"{a:g} deg {sets.pre_subsample_n[a]}" for a in ANGLES_DEG)
                    + ". Each angle's denominator is its own set's variance, which is not the "
                    "same quantity across angles — see G2.1d, which is the number to quote"
                ),
            ),
            Criterion(
                key="G2.1d",
                description=(
                    "the same ratio with a **fixed** denominator: the per-cell target "
                    f"variance over all {trained.index.n_cells} training cells "
                    f"(V = {variance:.4f}), shared by every angle "
                    f"(worst angle {fixed_worst:g} deg). **This is the number for the paper**"
                ),
                measured=fixed_ratio,
                threshold=OBLIQUE_PARITY_THRESHOLD,
                comparison=">=",
                note=(
                    "R^2 by angle: "
                    + ", ".join(f"{a:g} deg {fixed[a]:.4f}" for a in ANGLES_DEG)
                    + ". A 0 deg strip is one section and carries only in-plane target "
                    "variance; a 90 deg strip spans the stack and carries the along-z "
                    "variance too, so the per-set denominators of G2.1a differ by "
                    f"{spread:.2f}x "
                    "across angles and the ratio of two of them is a ratio of two different "
                    "questions. Fixing the denominator makes the comparison a statement "
                    "about the model's error per cell and nothing else"
                ),
            ),
            Criterion(
                key="G2.1b",
                description="R^2 at 0 deg (axis-aligned), the denominator of the ratio",
                measured=baseline,
                threshold=0.0,
                comparison=">",
                note=(
                    f"slab half-thickness {sets.half_thickness:g} um, common n "
                    f"{sets.common_n}, subsample seed {sets.seed}, own source section "
                    f"excluded from retrieval at every angle "
                    f"(Config.retrieval_exclude_source_section="
                    f"{cfg.retrieval_exclude_source_section})"
                ),
            ),
            Criterion(
                key="G2.1e",
                description=(
                    "diagnostic: the same fixed-denominator ratio with the **depth-mix "
                    "confound removed** — the 0 deg arm taken over the coronal planes at "
                    "every section rather than the single central one, so both sides of the "
                    "ratio are depth-representative"
                ),
                measured=(
                    min(v for a, v in fixed.items() if a != 0.0) / depth.coronal_pooled
                    if depth.coronal_pooled > 0
                    else float("nan")
                ),
                threshold=None,
                comparison="report",
                note=(
                    "per-section R^2: "
                    + ", ".join(
                        f"z={depth.section_z[s]:.0f} um {depth.per_section[s]:.4f}"
                        for s in sorted(depth.per_section)
                    )
                    + ". The two EDGE sections are the outliers, and every oblique angle "
                    f"draws ~{100 * mean_edge_share:.0f}% "
                    "of its cells from them while the 0 deg set draws none — it is one "
                    "interior section. R^2 predicted from section mix alone, ignoring the "
                    "angle entirely: "
                    + ", ".join(f"{a:g} deg {depth.predicted[a]:.4f}" for a in ANGLES_DEG)
                    + " — flat across every oblique angle and reproducing the measured values, "
                    "which is what says the angle dependence is depth mix and not direction. "
                    f"Depth-representative 0 deg arm: {depth.coronal_pooled:.4f}"
                ),
            ),
            Criterion(
                key="G2.1c",
                description=(
                    "turning the own-section exclusion OFF raises R^2 at 90 deg by this much "
                    "(the standing check that the exclusion is still plumbed through)"
                ),
                measured=exclusion["delta"],
                threshold=0.0,
                comparison=">",
                note=(
                    f"R^2(90 deg) = {exclusion['strict']:.4f} with the exclusion, "
                    f"{exclusion['permissive']:.4f} without it. If this ever reaches 0 the "
                    "exclusion is not reaching the gate's candidate filter and the gate can "
                    "pass while hiding the failure it exists to detect (SPEC_QUESTIONS C1a)."
                ),
            ),
        ],
        artifacts={
            "r2": scores,
            "r2_fixed": fixed,
            "fits": fits,
            "fixed_variance": variance,
            "n_train_cells": trained.index.n_cells,
            "sets": sets,
            "worst_angle": worst_angle,
            "ratio": ratio,
            "fixed_worst_angle": fixed_worst,
            "fixed_ratio": fixed_ratio,
            "exclusion": exclusion,
            "depth": depth,
            "losses": trained.losses,
        },
    )


@dataclass(frozen=True)
class DepthMix:
    """How much of the angle dependence is depth mix rather than angle.

    The confound this measures is structural, not statistical, and it was invisible until
    G2.1's denominator was fixed. Under the C1 membership rule a 0 degree query plane through
    the volume's centre selects **exactly one section** — the middle one — while every
    oblique plane draws roughly a fifth of its cells from the two *edge* sections, which have
    training and retrieval evidence on one side only and are reconstructed markedly worse for
    that reason alone. So `R^2(theta) / R^2(0 deg)` compares "the best-supported depth in the
    stack" against "a depth-representative sample", and part of what it reports is the depth
    mix rather than the angle.
    """

    per_section: dict[int, float]
    """section index -> fixed-denominator R^2 over all of that section's cells."""
    section_z: dict[int, float]
    """section index -> depth in micrometres."""
    edge_share: dict[float, float]
    """angle -> fraction of its evaluation set drawn from the two edge sections."""
    predicted: dict[float, float]
    """angle -> R^2 predicted from its section mix alone, ignoring the angle entirely."""
    coronal_pooled: float
    """R^2 of a depth-representative 0 degree arm: the coronal planes at *every* section,
    pooled with equal counts, instead of the single central one."""


def measure_depth_mix(trained: TrainedProbe, sets: EvaluationSets, variance: float) -> DepthMix:
    """Attribute the angle dependence to section mix. Costs one pass over the sections."""
    index = trained.index
    per_section: dict[int, float] = {}
    for s in range(index.n_sections):
        rows = np.nonzero(index.section_index == s)[0].astype(np.intp)
        per_section[s] = fit_residuals(
            predict(trained, index.coords[rows], trained.neighbours[rows]),
            trained.targets[rows],
            variance,
        ).r2_fixed

    edge = {0, index.n_sections - 1}
    edge_share: dict[float, float] = {}
    predicted: dict[float, float] = {}
    for angle, rows in sets.rows.items():
        counts = np.bincount(index.section_index[rows], minlength=index.n_sections)
        share = counts / max(counts.sum(), 1)
        edge_share[angle] = float(sum(share[s] for s in edge))
        predicted[angle] = float(sum(share[s] * per_section[s] for s in range(index.n_sections)))

    # A depth-representative 0 degree arm: every section contributes equally, which is what
    # the oblique arms already do. R^2_fixed is linear in the residual sum, so pooling
    # equal-sized per-section draws is exactly the mean of the per-section values.
    coronal_pooled = float(np.mean(list(per_section.values())))
    return DepthMix(
        per_section=per_section,
        section_z={s: float(index.section_z[s]) for s in range(index.n_sections)},
        edge_share=edge_share,
        predicted=predicted,
        coronal_pooled=coronal_pooled,
    )


def _parity(scores: dict[float, float]) -> tuple[float, float]:
    """``(min over oblique angles / value at 0 deg, the worst angle)``."""
    baseline = scores[0.0]
    oblique = {a: s for a, s in scores.items() if a != 0.0}
    worst = min(oblique, key=lambda a: oblique[a])
    return (oblique[worst] / baseline if baseline > 0 else float("nan")), worst


def measure_exclusion_effect(
    cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS
) -> dict[str, float]:
    """R^2 at 90 degrees with and without the own-section exclusion.

    The same trained probe both times: only the *evidence* changes. With the exclusion off,
    a cell in the 90 degree strip can retrieve in-plane neighbours a few micrometres away
    inside its own section, which is a near-copy of itself — so R^2 must rise. If it does
    not, the exclusion is not plumbed through the candidate filter and
    ``test_source_section_exclusion_changes_oblique_R2`` is the test that says so.
    """
    probes = gate2_probes(cfg, vol, seed=seed, steps=steps)
    trained = probes.main
    sets = evaluation_sets(trained.index, vol, cfg)
    rows = sets.rows[90.0]

    strict = r_squared(
        predict(trained, trained.index.coords[rows], trained.neighbours[rows]),
        trained.targets[rows],
    )
    permissive_index = RetrievalIndex(vol, cfg.replace(retrieval_exclude_source_section=False))
    permissive_idx, _ = permissive_index.query(
        trained.index.coords[rows],
        set(),
        seed=_seed_for(seed, 20),
        source_section=trained.index.section_index[rows],
    )
    permissive = r_squared(
        predict(trained, trained.index.coords[rows], permissive_idx, index=permissive_index),
        trained.targets[rows],
    )
    return {"strict": strict, "permissive": permissive, "delta": permissive - strict}


# --------------------------------------------------------------------------------------
# G2.2 — z-interpolation, not memorisation
# --------------------------------------------------------------------------------------


def measure_g2_2(cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS) -> GateSection:
    """R^2 at a genuinely held-out z against R^2 at the neighbouring training z.

    A sharp dip at the held-out z means the triplane has overfitted to the section
    positions — the remedy the spec names is a lower ``fourier_bands_z`` and a higher
    ``tv_z_weight``, which is why both are ``Config`` fields.
    """
    probes = gate2_probes(cfg, vol, seed=seed, steps=steps)
    trained = probes.holdout
    held = probes.holdout_section

    held_xyz = np.concatenate(
        [np.asarray(held.coords, dtype=np.float64), np.full((held.n_cells, 1), held.z)], axis=1
    )
    variance = per_cell_variance(trained.targets)
    held_targets = trained.index.expression_pcs.project(held.counts)
    held_idx, _ = trained.index.query(held_xyz, set(), seed=_seed_for(seed, 30))
    held_fit = fit_residuals(predict(trained, held_xyz, held_idx), held_targets, variance)

    neighbour_fits: dict[float, Fit] = {}
    for z in probes.holdout_neighbour_z:
        rows = np.nonzero(np.abs(trained.index.section_z[trained.index.section_index] - z) < 1e-6)[
            0
        ].astype(np.intp)
        neighbour_fits[float(z)] = fit_residuals(
            predict(trained, trained.index.coords[rows], trained.neighbours[rows]),
            trained.targets[rows],
            variance,
        )
    held_r2 = held_fit.r2_set
    neighbour_r2 = {z: f.r2_set for z, f in neighbour_fits.items()}
    mean_neighbour = float(np.mean(list(neighbour_r2.values())))
    ratio = held_r2 / mean_neighbour if mean_neighbour > 0 else float("nan")

    mean_neighbour_fixed = float(np.mean([f.r2_fixed for f in neighbour_fits.values()]))
    fixed_ratio = (
        held_fit.r2_fixed / mean_neighbour_fixed if mean_neighbour_fixed > 0 else float("nan")
    )

    return GateSection(
        key="G2.2",
        title="z-interpolation, not memorisation",
        criteria=[
            Criterion(
                key="G2.2a",
                description=(
                    f"R^2 at the held-out z = {held.z:g} um as a fraction of the mean R^2 at "
                    "the two neighbouring training z"
                ),
                measured=ratio,
                threshold=Z_INTERPOLATION_THRESHOLD,
                comparison=">=",
                note=(
                    f"held-out {held_r2:.4f} ({held.n_cells} cells) vs "
                    + ", ".join(f"z={z:g} um {r:.4f}" for z, r in neighbour_r2.items())
                    + f"; the probe was retrained without section {held.section_id!r}, and "
                    "its own section is excluded from retrieval for the neighbouring z too, "
                    "so both sides face the same evidence gap"
                ),
            ),
            Criterion(
                key="G2.2b",
                description=(
                    "the same ratio with the **fixed** denominator (per-cell target variance "
                    f"over all training cells, V = {variance:.4f})"
                ),
                measured=fixed_ratio,
                threshold=Z_INTERPOLATION_THRESHOLD,
                comparison=">=",
                note=(
                    f"held-out {held_fit.r2_fixed:.4f} vs "
                    + ", ".join(f"z={z:g} um {f.r2_fixed:.4f}" for z, f in neighbour_fits.items())
                    + ". The three sets here are whole sections of one volume, so their own "
                    "variances are close and the two denominators barely differ — unlike "
                    "G2.1, where the strips have genuinely different composition"
                ),
            ),
        ],
        artifacts={
            "held_r2": held_r2,
            "held_fit": held_fit,
            "neighbour_r2": neighbour_r2,
            "neighbour_fits": neighbour_fits,
            "held_z": float(held.z),
            "ratio": ratio,
            "fixed_ratio": fixed_ratio,
            "fixed_variance": variance,
        },
    )


# --------------------------------------------------------------------------------------
# G2.3 — the z-proximity term earns its place
# --------------------------------------------------------------------------------------


def measure_g2_3(cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS) -> GateSection:
    """Ablating ``w_z`` must cost R^2 at asymmetric depths and not at the symmetric one."""
    probes = gate2_probes(cfg, vol, seed=seed, steps=steps)
    middle = len(vol.sections) // 2
    wide = cfg.replace(retrieval_z_window=G23_Z_WINDOW)
    arms = {
        w_z: RetrievalIndex(vol, wide.replace(retrieval_w_z=w_z))
        for w_z in (cfg.retrieval_w_z, 0.0)
    }
    with_z = {
        f: _fractional_depth_r2(probes.main, arms[cfg.retrieval_w_z], vol, middle, f, seed=seed)
        for f in FRACTIONAL_DEPTHS
    }
    without_z = {
        f: _fractional_depth_r2(probes.without_z, arms[0.0], vol, middle, f, seed=seed)
        for f in FRACTIONAL_DEPTHS
    }
    deltas = {f: with_z[f] - without_z[f] for f in FRACTIONAL_DEPTHS}
    asymmetric = min(deltas[0.2], deltas[0.8])
    full_stack = {
        f: (
            _fractional_depth_r2(probes.main, probes.main.index, vol, middle, f, seed=seed)
            - _fractional_depth_r2(
                probes.without_z, probes.without_z.index, vol, middle, f, seed=seed
            )
        )
        for f in FRACTIONAL_DEPTHS
    }

    return GateSection(
        key="G2.3",
        title="The z-proximity term earns its place (pre-validates ablation A5)",
        criteria=[
            Criterion(
                key="G2.3a",
                description=(
                    "R^2 lost by setting retrieval_w_z = 0 at the asymmetric fractional "
                    "depths 0.2 and 0.8 (the smaller of the two shown)"
                ),
                measured=asymmetric,
                threshold=0.0,
                comparison=">",
                note=", ".join(
                    f"f={f:g}: {with_z[f]:.4f} with w_z, {without_z[f]:.4f} without "
                    f"(delta {deltas[f]:+.4f})"
                    for f in FRACTIONAL_DEPTHS
                ),
            ),
            Criterion(
                key="G2.3b",
                description=(
                    "and barely affects the symmetric depth 0.5, where the two flanking "
                    "sections are equidistant and the term has nothing to say (|delta|)"
                ),
                measured=abs(deltas[0.5]),
                threshold=G23_TIE_TOLERANCE,
                comparison="<",
                note=(
                    f"delta at 0.5 is {deltas[0.5]:+.4f}, against {asymmetric:+.4f} at the "
                    "asymmetric depths"
                ),
            ),
            Criterion(
                key="G2.3c",
                description=(
                    "diagnostic: the same three deltas with the **whole stack** in the pool "
                    "instead of two flanks (delta at 0.5 shown)"
                ),
                measured=full_stack[0.5],
                threshold=None,
                comparison="report",
                note=(
                    ", ".join(f"f={f:g}: {full_stack[f]:+.4f}" for f in FRACTIONAL_DEPTHS)
                    + ". With every section admissible the nearest one is always in the pool "
                    "and in-plane distance alone already ranks it first, so the z term has "
                    "little left to do and all three deltas sit inside the noise. The honest "
                    "reading of G2.3 is therefore that the term earns its place in the "
                    "wide-gap regime specifically — which is the regime the method exists "
                    "for, and the one ablation A5 must be run in."
                ),
            ),
        ],
        artifacts={
            "with_z": with_z,
            "without_z": without_z,
            "deltas": deltas,
            "full_stack_deltas": full_stack,
        },
    )


def _fractional_depth_r2(
    trained: TrainedProbe,
    index: RetrievalIndex,
    vol: Volume,
    section: int,
    fraction: float,
    *,
    seed: int,
) -> float:
    """R^2 on section ``section``'s cells with the evidence pushed to a fractional depth.

    ``index`` supplies the donors — normally the widened-window index built for this
    measurement, with every section but the two flanks excluded. The probe, the evaluated
    cells and their targets are identical across fractions and across arms; only the
    evidence moves.
    """
    lower, upper = _flank_pair(vol, section, fraction)
    excluded = {float(s.z) for i, s in enumerate(vol.sections) if i not in (lower, upper)}
    rows = np.nonzero(trained.index.section_index == section)[0].astype(np.intp)
    xyz = trained.index.coords[rows]
    idx, _ = index.query(
        xyz,
        excluded,
        seed=_seed_for(seed, 40, int(fraction * 100)),
        source_section=index.section_index[rows],
    )
    return r_squared(predict(trained, xyz, idx, index=index), trained.targets[rows])


def _flank_pair(vol: Volume, section: int, fraction: float) -> tuple[int, int]:
    """The two sections to leave in the pool so ``section`` sits at ``fraction`` between them."""
    offsets = {0.5: (-1, 1), 0.2: (-1, 4), 0.8: (-4, 1)}
    if fraction not in offsets:
        raise ValueError(f"_flank_pair: unsupported fraction {fraction!r}")
    below, above = offsets[fraction]
    lower, upper = section + below, section + above
    if lower < 0 or upper >= len(vol.sections):
        raise ValueError(
            f"_flank_pair: a fractional depth of {fraction} around section {section} needs "
            f"flanks at {lower} and {upper}, but the volume has {len(vol.sections)} sections"
        )
    return lower, upper


# --------------------------------------------------------------------------------------
# G2.4 — retrieval does not collapse to copying
# --------------------------------------------------------------------------------------


def measure_g2_4(cfg: Config, vol: Volume, *, seed: int, steps: int = PROBE_STEPS) -> GateSection:
    """Mean attention entropy over the K neighbours must exceed ``0.5 log K``."""
    probes = gate2_probes(cfg, vol, seed=seed, steps=steps)
    trained = probes.main
    sets = evaluation_sets(trained.index, vol, cfg)
    rows = np.unique(np.concatenate(list(sets.rows.values()))).astype(np.intp)

    entropies: list[float] = []
    valid_counts: list[float] = []
    with torch.no_grad():
        for start in range(0, rows.size, PROBE_EVAL_CHUNK):
            block = rows[start : start + PROBE_EVAL_CHUNK]
            xyz = trained.index.coords[block]
            tokens, mask = trained.index.neighbour_tokens(xyz, trained.neighbours[block])
            features = trained.probe.field(torch.from_numpy(xyz.astype(np.float32)))
            _, weights = trained.probe.attention.attend(features, tokens, mask)
            entropies.append(float(attention_entropy(weights).mean()))
            valid_counts.append(float(mask.sum(dim=1).to(torch.float32).mean()))

    entropy = float(np.mean(entropies))
    mean_valid = float(np.mean(valid_counts))
    limit = math.log(int(cfg.retrieval_k))
    return GateSection(
        key="G2.4",
        title="Retrieval does not collapse to copying",
        criteria=[
            Criterion(
                key="G2.4a",
                description=(
                    f"mean attention entropy over the K = {int(cfg.retrieval_k)} neighbours, "
                    f"in nats, against 0.5 log K = {ENTROPY_FRACTION_THRESHOLD * limit:.4f}"
                ),
                measured=entropy,
                threshold=ENTROPY_FRACTION_THRESHOLD * limit,
                comparison=">",
                unit=" nats",
                note=(
                    f"uniform over K would be {limit:.4f}, one-hot 0; the evaluated cells "
                    f"had {mean_valid:.1f} admissible donors on average"
                ),
            ),
            Criterion(
                key="G2.4b",
                description="mean attention entropy as a fraction of log K",
                measured=entropy / limit,
                threshold=None,
                comparison="report",
                note=(
                    "Read this beside G2.4a rather than as a second pass mark. The criterion "
                    "is one-sided — it forbids collapse onto a single donor — and this probe "
                    "sits at the *other* extreme, near-uniform attention, i.e. it is "
                    "averaging its K donors rather than selecting among them. That is safe "
                    "for the gate and expected of a linear probe trained for a few hundred "
                    "steps, but it means GATE 2 has not shown the attention to be selective, "
                    "only that it has not collapsed. T06 owns the selective version"
                ),
            ),
        ],
        artifacts={"entropy": entropy, "log_k": limit, "mean_valid_neighbours": mean_valid},
    )
