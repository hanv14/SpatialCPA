"""Can `marker_depth_r` measure anything on the held-out genes? — before any zero-shot fit.

The zero-shot experiment asks whether the text channel can place a gene the model never fitted
on. Before paying for four fits, this asks whether the **metric can tell the arms apart on those
genes at all** — the same question `scripts/t09_depth_ceiling.py` asked of the reconstruction
task, and which found `deep_starmap` saturated against an oracle copier.

**The saturation question has a different shape here, and getting it right matters.** In the
reconstruction task the competitor was a *copy of another real section*, so the question was "how
much room is there above copying". A zero-shot method cannot copy: `cross-mix` reads
``model.data.counts`` — the full training matrix, filtered by cell and never by gene — so it
would emit the held-out genes' real counts verbatim. It is not a zero-shot method, it is a lookup
of the answer, and it is excluded from the experiment rather than handicapped. The competitor is
therefore the **constant field**: predict each held-out gene's own global mean everywhere.

⚠️ **That referent is not zero, and it was asserted to be before it was measured.** The intuition
— a flat field has a flat profile, so ``_safe_r`` sees zero variance and returns 0 — is wrong,
because ``soft_depth_profile`` divides each bin by the kernel weight it received and an
under-populated bin is guarded by ``eps`` rather than filled. So the profile of a constant field
is *not* constant: it tracks **where the cells are** along the depth axis, and cell density is
itself laminar. Measured at **+0.013 to +0.049** on the synthetic fixture. Small, but it is the
number every arm must actually beat, and it is reported per side and per section rather than
assumed.

So the question this file answers is:

    is ``sqrt(R)`` on the held-out genes well clear of zero, and comparable to the kept genes'?

* **well clear of zero** — otherwise the held-out genes carry no reproducible depth structure and
  no arm can demonstrate anything on them, whatever it does;
* **comparable to the kept genes'** — otherwise the split is not representative and a result on
  it would not generalise to the panel.

Reported for both sides of the split, model-free: split-half reliability, the ceiling ``sqrt(R)``,
a shuffled floor, the constant-field referent (verified to be 0, not assumed), and — as context
only, since no arm may use it — what copying another real section scores.

Usage::

    python scripts/t09_zeroshot_ceiling.py --dataset deep_starmap \\
        --bench3 /path/to/benchmark-pbya-v3 --out reports/t09_zeroshot_ceiling.md
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.losses.metric_aware import (
    knn_weight_graph,
    morans_i,
    profile_axis,
    soft_depth_profile,
)
from spatialcpav25_gen.train.select import _normalised, selection_folds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _gene_split import markers_within, stratified_gene_split
from _starmap_run import base_config, clamp_config_to_input, load_training_volume
from t09_depth_ceiling import profile, ruler, safe_r, score


def _profile_f64(counts, coords_xy, z_value, axis, markers, cfg, bounds, sigma, pool):
    """:func:`~t09_depth_ceiling.profile` end to end in float64. The precision arm of the probe.

    A genuine recomputation, not a cast: ``_normalised`` returns float32 and casting its output
    afterwards would compare a number with itself and call the referent stable whatever it is.
    The whole chain — the size factor, the coordinates, the axis and the Gaussian binning — runs
    at double precision here.
    """
    dense = np.asarray(counts, dtype=np.float64)
    sized = dense[:, np.asarray(pool, dtype=np.int64)]
    totals = np.clip(sized.sum(axis=1, keepdims=True), float(cfg.metric_eps), None)
    x = torch.from_numpy(dense / totals)
    xy = np.asarray(coords_xy, dtype=np.float64)[:, :2]
    xyz = torch.from_numpy(np.concatenate([xy, np.full((xy.shape[0], 1), float(z_value))], axis=1))
    return soft_depth_profile(
        x.index_select(1, markers),
        xyz,
        axis.to(torch.float64),
        int(cfg.profile_n_bins),
        sigma,
        bounds=bounds,
    ).numpy()


def side_ceiling(target, donors, pool, axis, cfg, rng, splits: int, shuffles: int) -> dict:
    """Every reference point for one gene ``pool`` on one target section. Model-free."""
    counts = np.asarray(target.counts.todense(), dtype=np.float64)
    xy = np.asarray(target.coords, dtype=np.float64)
    graph = knn_weight_graph(xy, cfg)
    markers = markers_within(_normalised(counts, cfg, pool), graph, cfg, pool)
    bounds, sigma = ruler(target, axis, cfg)
    p_target = profile(counts, xy, target.z, axis, markers, cfg, bounds, sigma, pool)

    halves = []
    for _ in range(splits):
        order = rng.permutation(counts.shape[0])
        a, b = order[: order.size // 2], order[order.size // 2 :]
        r = score(
            profile(counts[a], xy[a], target.z, axis, markers, cfg, bounds, sigma, pool),
            profile(counts[b], xy[b], target.z, axis, markers, cfg, bounds, sigma, pool),
        )
        halves.append(2.0 * r / (1.0 + r) if r > -1.0 else r)
    reliability = float(np.mean(halves))

    shuffled = float(
        np.mean(
            [
                score(
                    profile(
                        counts,
                        xy[rng.permutation(xy.shape[0])],
                        target.z,
                        axis,
                        markers,
                        cfg,
                        bounds,
                        sigma,
                        pool,
                    ),
                    p_target,
                )
                for _ in range(shuffles)
            ]
        )
    )
    # The constant field: every cell gets that gene's own global mean. NOT zero — the bin
    # normalisation makes a constant field's profile track cell density along the depth axis,
    # and density is laminar. Computed rather than asserted, which is how that was found.
    #
    # The same precision probe the autocorrelation branch runs, for the same reason: a referent
    # whose value survives a change of precision is a function of the data, and one that does not
    # is a function of summation order. This branch is *expected* to pass — the density signal is
    # real arithmetic on a real quantity — and it is measured rather than assumed, because
    # "expected to pass" is how the hand-written flag got into the other branch.
    flat = np.broadcast_to(counts.mean(axis=0), counts.shape).copy()
    constant = score(profile(flat, xy, target.z, axis, markers, cfg, bounds, sigma, pool), p_target)
    constant_f64 = score(
        _profile_f64(flat, xy, target.z, axis, markers, cfg, bounds, sigma, pool),
        _profile_f64(counts, xy, target.z, axis, markers, cfg, bounds, sigma, pool),
    )
    others = []
    for donor in donors:
        d_counts = np.asarray(donor.counts.todense(), dtype=np.float64)
        others.append(
            {
                "donor": donor.section_id,
                "dz": abs(float(donor.z) - float(target.z)),
                "r": score(
                    profile(
                        d_counts,
                        np.asarray(donor.coords, dtype=np.float64),
                        target.z,
                        axis,
                        markers,
                        cfg,
                        bounds,
                        sigma,
                        pool,
                    ),
                    p_target,
                ),
            }
        )
    return {
        "n_pool": len(pool),
        "n_markers": int(markers.numel()),
        "marker_names_from_pool": True,
        "self": score(p_target, p_target),
        "split_half": reliability,
        "noiseless_ceiling": float(np.sqrt(reliability)) if reliability > 0 else float("nan"),
        "shuffled_floor": shuffled,
        "constant_field": constant,
        "constant_field_float64": constant_f64,
        "constant_field_precision_drift": abs(constant - constant_f64),
        **input_information(flat, cfg, pool, "constant_field"),
        # The stable-referent control, on these rows rather than on another dataset: `shuffled`
        # keeps the real counts and only moves the positions, so its input carries the panel's
        # full variation and it is what "not degenerate" looks like here.
        **input_information(counts, cfg, pool, "shuffled"),
        "best_other_section": max((o["r"] for o in others), default=float("nan")),
        "others": others,
    }


def autocorr_vector(counts, coords_xy, cfg, pool) -> np.ndarray:
    """Per-gene Moran's I over ``pool``. ``(N, G)``, ``(N, 2)`` -> ``(n_pool,)``.

    The quantity `morans_pearson` correlates, computed exactly as ``section_scores`` computes it:
    the same pool-restricted normalisation, the same kNN graph on the cells actually present, the
    same ``metric_eps``. Nothing here is a re-derivation of the metric — a ceiling computed with a
    different estimator is a ceiling for a different metric.
    """
    x = _normalised(np.asarray(counts, dtype=np.float64), cfg, pool)
    graph = knn_weight_graph(np.asarray(coords_xy, dtype=np.float64)[:, :2], cfg)
    values = morans_i(x, graph, eps=float(cfg.metric_eps)).numpy()
    return np.asarray(values[np.asarray(pool, dtype=np.int64)], dtype=np.float64)


INFORMATION_TOL = 1e-6
"""Largest per-gene coefficient of variation an input may have and still carry no information.

**This, and not the precision drift, is what decides degeneracy.** A referent answers "what does
this metric return when there is nothing to find?" only if its input really contains nothing to
find. The constant field gives every cell the same value for a gene, so its across-cell CV should
be exactly 0 and in practice sits at the working precision's epsilon (~1e-7 for float32). An input
whose relative variation is at the level of the arithmetic itself holds no spatial signal, so
whatever the metric returns for it is the metric's behaviour on a degenerate input — **at any
magnitude, and whatever the precision drift happens to be**. The tolerance sits an order above
float32 epsilon and five orders below anything real.

⚠️ **A drift threshold was tried first and was wrong.** The synthetic fixture separated cleanly —
constant-field drift ~1e-2 against ~1e-8 for referents with real variance — and a 0.01 cut looked
principled. On `deep_starmap` the eight constant-field rows drift 0.0042, 0.0092, 0.0092, 0.0111,
0.0438, 0.0545, 0.0738, 0.1905: a **continuum with no gap**, so the cut fell between two rows of
identical construction and called one stable and the other degenerate. A threshold that separates
nothing is a defect in the instrument, not a finding about the data.

**Why a large float64 value is still round-off.** At double precision `section_5`'s held-out
constant field reads +0.3875 rather than +0.5780 — smaller, but nowhere near zero, and a reader is
right to ask why. Round-off in the centring step is proportional to each value's own magnitude
(one ulp of it), so its *pattern across genes* tracks expression level at **every** precision,
and expression level is what real Moran's I correlates with. Doubling the mantissa changes the
number without changing what it is. That is exactly why the input test decides and the drift is
only corroboration.
"""


def _float64_graph(coords_xy, cfg):
    """``knn_weight_graph`` with its values promoted to float64. Same graph, same neighbours."""
    graph = knn_weight_graph(np.asarray(coords_xy, dtype=np.float64)[:, :2], cfg)
    return torch.sparse_coo_tensor(
        graph.indices(), graph.values().to(torch.float64), graph.shape
    ).coalesce()


def _autocorr_vector_f64(counts, coords_xy, cfg, pool) -> np.ndarray:
    """:func:`autocorr_vector` end to end in float64. The precision arm of the probe.

    Deliberately a second implementation of the same arithmetic rather than a parameter on the
    first: the point is to run the computation in a different precision, and a shared code path
    that quietly upcast would answer a different question.
    """
    dense = np.asarray(counts, dtype=np.float64)
    sized = dense[:, np.asarray(pool, dtype=np.int64)]
    totals = np.clip(sized.sum(axis=1, keepdims=True), float(cfg.metric_eps), None)
    x = torch.from_numpy(dense / totals)
    values = morans_i(x, _float64_graph(coords_xy, cfg), eps=float(cfg.metric_eps)).numpy()
    return np.asarray(values[np.asarray(pool, dtype=np.int64)], dtype=np.float64)


def constant_field_probe(counts, coords_xy, cfg, pool, v_target) -> dict:
    """Is the constant-field referent information, or is it arithmetic? Measured, not asserted.

    A constant field has **exactly zero** per-gene variance after normalisation — it is the same
    number in every cell — so it carries no spatial information whatever. That alone does not
    make a non-zero referent invalid: `marker_depth_r` maps a constant input to a real, stable
    function of **cell density**, because ``soft_depth_profile`` divides each bin by the weight it
    received. `morans_pearson` maps it to ``0/0``, which the hardware resolves with round-off.

    The two cases are told apart by **precision stability**, which is a measurement:

    * ``input_std_max`` — the largest per-gene std of the normalised constant field. Zero confirms
      the input really is constant, so anything downstream is the metric's own behaviour.
    * ``float32`` / ``float64`` — the referent computed both ways. A stable value is a function of
      the data; a value that moves by O(0.1-1) is a function of summation order.

    ``is_degenerate`` is the verdict of that comparison against :data:`DEGENERACY_TOL`, and it
    replaces a hand-written ``True``. The earlier version of this file asserted the flag as a
    literal on the strength of a fixture reproduction; an assertion in the output of an instrument
    is indistinguishable from a measurement to anyone reading the JSON, which is exactly the
    failure this function exists to remove.
    """
    constant = np.broadcast_to(counts.mean(axis=0), counts.shape).copy()
    f32 = safe_r(autocorr_vector(constant, coords_xy, cfg, pool), v_target)
    f64 = safe_r(
        _autocorr_vector_f64(constant, coords_xy, cfg, pool),
        _autocorr_vector_f64(counts, coords_xy, cfg, pool),
    )
    return {
        "constant_field": f32,
        "constant_field_float64": f64,
        "constant_field_precision_drift": abs(f32 - f64),
        **input_information(constant, cfg, pool, "constant_field"),
    }


def input_information(values, cfg, pool, name: str) -> dict:
    """Does this referent's input carry anything to find? ``{name}_input_cv_max``, and the verdict.

    The largest per-gene coefficient of variation across cells, over genes with a non-zero mean —
    scale-free, so it means the same thing on a fixture and on a 1017-gene panel, which an
    absolute standard deviation does not. Genes with a zero mean are excluded: their CV is 0/0 and
    they carry no information either way.
    """
    normalised = _normalised(np.asarray(values, dtype=np.float64), cfg, pool).numpy()
    columns = normalised[:, np.asarray(pool, dtype=np.int64)]
    mean = np.abs(columns.mean(axis=0))
    live = mean > 0.0
    cv = np.zeros_like(mean)
    cv[live] = columns.std(axis=0)[live] / mean[live]
    worst = float(cv.max()) if cv.size else 0.0
    return {
        f"{name}_input_cv_max": worst,
        f"{name}_is_degenerate": bool(worst <= INFORMATION_TOL),
    }


def side_ceiling_autocorr(target, donors, pool, cfg, rng, splits: int, shuffles: int) -> dict:
    """Every reference point for `morans_pearson` on one gene ``pool``, one section. Model-free.

    The same design as :func:`side_ceiling`, with the per-gene Moran's I vector in place of the
    depth profile: split-half reliability of that vector, Spearman-Brown corrected, and its square
    root as the correlation a perfect independent replicate could reach.

    **The half-section caveat, stated because it decides the direction of the error.** A split
    half has half the cells, so its kNN graph reaches physically further and its Moran's I is
    both coarser and noisier than a full section's. Spearman-Brown corrects for length, not for a
    changed estimator, so the reliability here is if anything **under**-estimated and the ceiling
    with it. A ceiling that errs low is the safe direction: it cannot manufacture headroom that
    is not there.

    ``constant_field`` is reported and **labelled degenerate**. A constant field has exactly zero
    per-gene variance after normalisation, so Moran's I is ``0/0`` and what comes back is float32
    round-off that scales with the gene's magnitude — measured on the fixture at std 3.5e-8 giving
    `morans_pearson` +0.22, and on `deep_starmap` +0.53. It is not a floor and must not be quoted
    as one; the floor for this metric is the shuffled referent (`specs/10` §4.2b's companion note
    in `progress/`). Reported anyway so the number is on the record beside its status.
    """
    counts = np.asarray(target.counts.todense(), dtype=np.float64)
    xy = np.asarray(target.coords, dtype=np.float64)
    v_target = autocorr_vector(counts, xy, cfg, pool)

    halves = []
    for _ in range(splits):
        order = rng.permutation(counts.shape[0])
        a, b = order[: order.size // 2], order[order.size // 2 :]
        r = safe_r(
            autocorr_vector(counts[a], xy[a], cfg, pool),
            autocorr_vector(counts[b], xy[b], cfg, pool),
        )
        halves.append(2.0 * r / (1.0 + r) if r > -1.0 else r)
    reliability = float(np.mean(halves))

    shuffled = float(
        np.mean(
            [
                safe_r(
                    autocorr_vector(counts, xy[rng.permutation(xy.shape[0])], cfg, pool), v_target
                )
                for _ in range(shuffles)
            ]
        )
    )
    probe = constant_field_probe(counts, xy, cfg, pool, v_target)
    others = [
        {
            "donor": donor.section_id,
            "dz": abs(float(donor.z) - float(target.z)),
            "r": safe_r(
                autocorr_vector(
                    np.asarray(donor.counts.todense(), dtype=np.float64),
                    np.asarray(donor.coords, dtype=np.float64),
                    cfg,
                    pool,
                ),
                v_target,
            ),
        }
        for donor in donors
    ]
    return {
        "n_pool": len(pool),
        "n_markers": len(pool),
        "marker_names_from_pool": True,
        "self": safe_r(v_target, v_target),
        "split_half": reliability,
        "noiseless_ceiling": float(np.sqrt(reliability)) if reliability > 0 else float("nan"),
        "shuffled_floor": shuffled,
        **probe,
        # The stable-referent control on these same rows: `shuffled` keeps the real counts and
        # moves only the positions, so its input carries the panel's full variation.
        **input_information(counts, cfg, pool, "shuffled"),
        "best_other_section": max((o["r"] for o in others), default=float("nan")),
        "others": others,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--metric",
        default="marker_depth_r",
        choices=["marker_depth_r", "morans_pearson"],
        help="which metric's ceiling to measure on the held-out genes. `marker_depth_r` is the "
        "pre-registered primary and the default; `morans_pearson` is the metric the four-arm "
        "run's only positive landed on, whose ceiling that run could not be read against",
    )
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--strata-bins", type=int, default=5)
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--shuffles", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expr-pca-dim", type=int, default=None)
    ap.add_argument("--out", default="reports/t09_zeroshot_ceiling.md")
    ap.add_argument("--split-out", default="reports/t09_gene_split.json")
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    overrides: dict = {}
    if args.expr_pca_dim is not None:
        overrides["expr_pca_dim"] = int(args.expr_pca_dim)
    cfg = clamp_config_to_input(base_config(args.seed, **overrides), paths.input)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    axis = profile_axis(volume, cfg)

    split = stratified_gene_split(
        volume, cfg, seed=args.split_seed, frac=args.held_frac, n_bins=args.strata_bins
    )
    Path(args.split_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.split_out).write_text(json.dumps(split.to_json(), indent=2))
    print(
        f"  split: {len(split.kept)} kept / {len(split.held_out)} held out of "
        f"{len(split.names)}, stratified {args.strata_bins}x{args.strata_bins} on mean x "
        f"Moran's I (ranked on {split.reference_section}), seed {args.split_seed}"
    )
    print(f"  wrote {args.split_out}")

    rows = []
    for target in selection_folds(volume, cfg):
        donors = [s for s in volume.sections if s.section_id != target.section_id]
        rng = np.random.default_rng(args.seed)
        row = {"target": target.section_id, "dataset": paths.dataset, "holdout": paths.holdout}
        for name, pool in (("held_out", split.held_out), ("kept", split.kept)):
            row[name] = (
                side_ceiling(target, donors, pool, axis, cfg, rng, args.splits, args.shuffles)
                if args.metric == "marker_depth_r"
                else side_ceiling_autocorr(
                    target, donors, pool, cfg, rng, args.splits, args.shuffles
                )
            )
            c = row[name]
            print(
                f"  {target.section_id} [{name:<8}] self {c['self']:+.6f} | R "
                f"{c['split_half']:+.4f} | ceiling {c['noiseless_ceiling']:.4f} | constant "
                f"{c['constant_field']:+.6f}"
                + (" (degenerate)" if c.get("constant_field_is_degenerate") else "")
                + f" | shuffled {c['shuffled_floor']:+.4f} | best copy "
                f"{c['best_other_section']:+.4f}",
                flush=True,
            )
            if abs(c["self"] - 1.0) > 1e-9:
                raise SystemExit(
                    f"self-correlation on {target.section_id} [{name}] is {c['self']!r}, not "
                    "1.0. The profile code does not reproduce the metric; nothing here is usable."
                )
        rows.append(row)

    text = _report(rows, split, cfg, volume, paths, args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n")
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2, default=str))
    print("\n" + text)
    print(f"\nwrote {out} and {out.with_suffix('.json')}")
    return 0


def _report(rows, split, cfg, volume, paths, args) -> str:
    lines = [
        f"# Can `{args.metric}` measure anything on the held-out genes?",
        "",
        f"Dataset **`{paths.dataset}`**, holdout **`{paths.holdout}`** — {volume.n_cells} cells x "
        f"{volume.n_genes} genes. Split: **{len(split.kept)} kept / {len(split.held_out)} held "
        f"out**, stratified {split.n_bins}x{split.n_bins} on mean expression x Moran's I "
        f"(ranked on `{split.reference_section}`), seed {split.seed}. "
        f"{args.splits} split-half repeats, {args.shuffles} shuffles. **No model, no fit.**",
        "",
        "| target | side | genes | markers | split-half R | **ceiling √R** | constant field | "
        "shuffled | best copy (context only) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        for side in ("held_out", "kept"):
            c = r[side]
            lines.append(
                f"| `{r['target']}` | `{side}` | {c['n_pool']} | {c['n_markers']} | "
                f"{c['split_half']:+.4f} | **{c['noiseless_ceiling']:.4f}** | "
                f"{c['constant_field']:+.6f} | {c['shuffled_floor']:+.4f} | "
                f"{c['best_other_section']:+.4f} |"
            )
    held = [r["held_out"]["noiseless_ceiling"] for r in rows]
    kept = [r["kept"]["noiseless_ceiling"] for r in rows]
    ratio = float(np.median(held) / np.median(kept)) if np.median(kept) else float("nan")
    worst_const = max(abs(r["held_out"]["constant_field"]) for r in rows)
    worst_shuf = max(abs(r["held_out"]["shuffled_floor"]) for r in rows)
    # The floor is whichever referent is *usable* for this metric. For `marker_depth_r` the
    # constant field is a real (if small) no-information reference; for `morans_pearson` it is
    # float32 round-off on a zero-variance input and only the shuffled referent is a floor.
    degenerate = any(r["held_out"].get("constant_field_is_degenerate") for r in rows)
    floor = worst_shuf if degenerate else max(worst_const, worst_shuf)
    room = float(np.median(held)) - floor
    lines += [
        "",
        "`self` is 1.000000 on every row or the run aborts — a check on this file, not a result.",
        "",
        "### The two questions this decides",
        "",
        f"**1. Is the ceiling clear of the floor on the held-out genes?** Median √R "
        f"**{np.median(held):.4f}**; the largest **constant-field** referent is "
        f"{worst_const:.4f} and the largest shuffled floor {worst_shuf:.4f}, so the room "
        f"available above the **usable** floor is **{room:.4f}**. "
        "A zero-shot arm cannot copy — `cross-mix` reads the full count matrix and would emit "
        "the held-out genes verbatim, so it is excluded from the experiment rather than "
        "handicapped.",
        "",
        (
            "⚠️ For this metric the **constant field is degenerate and is not the floor.** It "
            "has exactly zero per-gene variance after normalisation, so Moran's I is `0/0` and "
            "what comes back is float32 round-off that scales with the gene's own magnitude — "
            "which is why it correlates with the real statistic at all. The floor here is the "
            "**shuffled** referent, and the room above is quoted against that."
            if degenerate
            else "⚠️ The constant-field referent is **not zero**: `soft_depth_profile` normalises "
            "each bin by the weight it received, so a constant field's profile tracks cell "
            "density along the depth axis and density is laminar. It is measured, not assumed, "
            "and it is the thing every arm must beat."
        ),
        "",
        f"**2. Is the split representative?** Held-out ceiling is **{100 * ratio:.0f}%** of the "
        "kept genes' on the same sections. Far from 100% would mean the stratified draw still "
        "took systematically easier or harder genes, and a result on the held-out set would not "
        "generalise to the panel.",
        "",
        "**`best copy` is context only.** It is what copying a whole real section scores on these "
        "genes, and it is reported so the numbers can be placed beside the reconstruction "
        "ceilings — **no arm in the zero-shot experiment may use it.**",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
