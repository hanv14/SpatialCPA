"""Does `L_thick` charge a non-zero loss on this dataset? The free check before A7's fits.

A7 is an **addition** experiment: the shipped model (all three SEFL weights at 0) against the
same model with `w_thick = w_prog = 0.2`. Costed at ~6.2 core-hours on tier-1
(`starmap_visual_cortex`, 62 min/fit, 4.6x reconstruction headroom) against ~23 on
`deep_starmap` (3.74-4.09 h/fit, saturated at 0.5x). Tier-1 is the right dataset by this
project's ceiling-first rule -- **if `L_thick` binds there**.

**Why it might not.** `L_thick` compares one slab of thickness `3h` against three of thickness
`h`, where `h` is the training sections' median `Section.thickness` and `3` is
`Config.thickness_ratio`. Three ways that becomes vacuous, and each is checked here:

1. **The thick slab does not fit.** `_slab_sample` raises when a drawn plane grazes the volume,
   and `thick_terms` then returns exact zeros. If `3h` is a large fraction of the volume's
   z-extent, most draws graze and the term is zero most of the time.
2. **The slabs overlap.** `thickness > spacing` makes the coarse-graining identity false --
   the same tissue observed twice -- and `OverlappingSlabsWarning` says so. `L_thick` is then
   ill-posed rather than merely small.
3. **The thickness is assumed, not measured.** `Section.thickness_is_assumed` marks a value
   defaulted from median spacing. `L_thick` still binds, but every number it produces rests on
   that assumption and the A7 result has to carry the caveat.

**This does not fit a model or take a training step**, and the first version of it was wrong for
that reason. `L_thick` compares the **student** on the thick slab against the **teacher** on the
three thin ones, over common random numbers whose estimator error cancels exactly. At
initialisation the EMA teacher is a deep copy of the student, so the two branches compute the same
intensity from the same points and the term is **exactly zero by construction** — measured on the
synthetic fixture at 0.0, 0.0 and -1.1e-13 for `count`, `count_by_type` and `state`, on a geometry
where the thick slab occupies only 18 % of the z-extent and nothing is grazing. An untrained model
therefore returns zero *whether or not the term binds*, and the check as first written could not
tell the two apart.

So the script reports **two** passes over the same `n_draws` planes, and only the second decides:

* **agreement** — student and teacher identical. Near-zero is the *correct* answer and is a
  validity check that the partition identity holds on this geometry. A large value here would mean
  the common-random-numbers machinery is broken on this dataset.
* **disagreement** — the student's intensity head perturbed by a seeded relative amount, the
  teacher left as the clone. This is the question: **if the two branches disagree, does this
  geometry let the term notice?** A term that stays at zero here is vacuous.

Grazing is counted separately and is distinguishable: `_zero_terms` returns *exact* 0.0 on all
three parts, where agreement leaves float residue around 1e-13.

⚠️ **A non-zero loss is necessary, not sufficient.** It says the geometry admits the term. It
does not say `L_thick` will *help*, which is what A7 measures and what nothing short of the six
fits can say.

**Verdict is descriptive.** `BINDS` / `DOES NOT BIND` / `ILL-POSED` are reported with the
numbers behind them, and the decision -- run A7 here, move it to `deep_starmap`, or run it on
`L_prog` alone as a different experiment -- is the user's.

Usage::

    python scripts/t10_a7_thick_binding.py --dataset starmap_visual_cortex --holdout paper_2_4_6 \\
        --out reports/t10_a7_thick_binding_tier1.json

    # the comparison arm, on the dataset A7 would otherwise run on
    python scripts/t10_a7_thick_binding.py --dataset deep_starmap --holdout paper_2_4_6 \\
        --out reports/t10_a7_thick_binding_deep.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from spatialcpav25_gen.losses.sefl import EMATeacher, thick_terms
from spatialcpav25_gen.model.spatialcpav25_gen import CTFFlow, TrainingData

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bench3_paths import add_path_args, resolve, set_torch_threads
from _starmap_run import embeddings_factory, load_training_volume
from t09_zeroshot_run import arm_config

ZERO = 1e-9
"""Below this a term is counted as charging nothing. Placed above the float residue an
*agreeing* student and teacher leave (measured at 1e-13 on the fixture) and far below any real
disagreement, so it separates "the branches agree" from "the branches disagree and the term
noticed". The report prints the raw distributions beside the verdict either way."""

FACTORS = (1.0, 1.001, 1.003, 1.01, 1.03, 1.1, 1.3)
"""Student-side scale factors swept, i.e. relative count errors of 0 to 30 %. `1.0` is the
validity check (student == teacher; near-zero is the correct answer). The rest bracket the
Poisson hinge on any plausible slab count: `1/sqrt(N)` is 3 % at N = 1000 and 0.3 % at N = 10^5."""

BINDS_FRACTION = 0.5
"""Fraction of *disagreement* draws that must charge a non-zero loss for the verdict to read
BINDS. Set from the mechanism rather than from an expected answer: `thick_terms` zeroes a draw
only when the slab does not fit in the volume, so this asks whether the geometry admits the term
more often than not. Between this and zero reads MARGINAL and is the user's call."""


def geometry(volume, cfg) -> dict:
    """Slab geometry, before any model: does a `3h` slab fit, and do the sections overlap?"""
    thickness = np.array([float(s.thickness) for s in volume.sections], dtype=np.float64)
    z = np.array([float(s.z) for s in volume.sections], dtype=np.float64)
    order = np.argsort(z)
    spacing = np.diff(z[order])
    h = float(np.median(thickness))
    ratio = int(cfg.thickness_ratio)
    bbox = np.asarray(volume.bbox, dtype=np.float64)
    extent = bbox[1] - bbox[0]
    return {
        "n_sections": len(volume.sections),
        "thickness_median_um": h,
        "thickness_min_um": float(thickness.min()),
        "thickness_max_um": float(thickness.max()),
        "thickness_is_assumed": bool(any(bool(s.thickness_is_assumed) for s in volume.sections)),
        "spacing_median_um": float(np.median(spacing)) if spacing.size else float("nan"),
        "spacing_min_um": float(spacing.min()) if spacing.size else float("nan"),
        "slabs_overlap": bool(spacing.size and h > float(spacing.min())),
        "thickness_ratio": ratio,
        "thick_slab_um": h * ratio,
        "bbox_extent_um": [float(v) for v in extent],
        # The fraction of the volume's z-extent one thick slab occupies. Near 1.0 means almost
        # every plane's slab runs out of volume; that is failure mode 1.
        "thick_slab_over_z_extent": float(h * ratio / extent[2]) if extent[2] > 0 else float("inf"),
    }


def draw_terms(model, teacher, cfg, n_draws: int, seed: int) -> dict:
    """Run `thick_terms` on `n_draws` random planes. Returns the per-part distributions.

    ⚠️ `fraction_grazed` counts draws where every part is **exactly** 0.0 — `_zero_terms`'
    signature, meaning the thick slab did not fit in the volume. **It is only a grazing
    estimate when the student-teacher disagreement is large enough to escape the Poisson
    hinge**; below that, `_poisson_consistency` returns exact zeros too and the two causes are
    indistinguishable. On the fixture it reads 0.375 at a 1 % error and 0.000 at 3 %, and the
    3 % figure is the true one. Read it off the largest factor in the sweep, which is what the
    verdict does.
    """
    bbox = np.asarray(model.data.vol.bbox, dtype=np.float64)
    gen = np.random.default_rng(seed)
    parts: dict[str, list[float]] = {}
    grazed = 0
    with torch.no_grad():
        for _ in range(int(n_draws)):
            terms = thick_terms(model, teacher, bbox, cfg, gen)
            if all(float(v) == 0.0 for v in terms.values()):
                grazed += 1
            for name, value in terms.items():
                parts.setdefault(name, []).append(float(value))
    arrays = {name: np.asarray(v, dtype=np.float64) for name, v in parts.items()}
    total = np.sum(np.stack(list(arrays.values())), axis=0)
    out: dict = {"n_draws": int(n_draws), "fraction_grazed": grazed / float(n_draws)}
    for name, values in sorted(arrays.items()):
        out[name] = {
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "fraction_nonzero": float(np.mean(np.abs(values) > ZERO)),
        }
    out["total"] = {
        "median": float(np.median(total)),
        "min": float(total.min()),
        "max": float(total.max()),
        "fraction_nonzero": float(np.mean(np.abs(total) > ZERO)),
    }
    return out


def _scaled_intensity(model, factor: float):
    """Patch `sefl._intensity_at` so the **student's** branch returns `factor` x its intensity.

    Perturbing parameters was the first version and it could not be read: a 5 % move on 26 182
    intensity-head weights changed the integrated count by far less than one Poisson standard
    deviation, so `_poisson_consistency`'s hinge returned exact zero and the check reported
    "does not bind" on a geometry where it plainly does. Scaling the branch's **output** by a
    known factor makes the disagreement the quantity the term actually compares, so the sweep
    below reads in units of relative count error.

    Patched for the duration of one context and restored in a `finally`, the same discipline
    `t09_retention_mechanism` applies to `_flow_counts`. Only the student is scaled: `module is
    model` identifies it, and the teacher is a different object (`_teacher_module` returns the
    EMA clone).
    """
    import contextlib

    from spatialcpav25_gen.losses import sefl as sefl_mod

    @contextlib.contextmanager
    def patched():
        original = sefl_mod._intensity_at

        def scaled(module, plane, xyz, cfg, *, seed):
            out = original(module, plane, xyz, cfg, seed=seed)
            return out * factor if module is model else out

        sefl_mod._intensity_at = scaled
        try:
            yield
        finally:
            sefl_mod._intensity_at = original

    return patched()


def binding_sweep(model, teacher, cfg, n_draws: int, seed: int, factors) -> dict:
    """Charge `L_thick` at a range of student-side relative errors. The binding check.

    Returns, per factor, the fraction of draws whose total charges above `ZERO`, and the
    median charge. The number that matters is the **smallest relative error the term
    notices** — below the Poisson hinge it is exactly zero however wrong the student is.
    """
    rows = {}
    for factor in factors:
        error = abs(factor - 1.0)
        with _scaled_intensity(model, float(factor)):
            drawn = draw_terms(model, teacher, cfg, n_draws, seed)
        rows[f"{error:.4g}"] = {
            "relative_error": float(error),
            "fraction_nonzero": float(drawn["total"]["fraction_nonzero"]),
            "median_total": float(drawn["total"]["median"]),
            "max_total": float(drawn["total"]["max"]),
            "fraction_grazed": float(drawn["fraction_grazed"]),
            "count": float(drawn["count"]["median"]),
            "count_by_type": float(drawn["count_by_type"]["median"]),
            "state": float(drawn["state"]["median"]),
        }
    charging = [r for r in rows.values() if r["fraction_nonzero"] >= BINDS_FRACTION]
    return {
        "by_relative_error": rows,
        "smallest_error_noticed": min((r["relative_error"] for r in charging), default=None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--draws", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--flattened", dest="flattened", action="store_true", default=None)
    ap.add_argument("--no-flattened", dest="flattened", action="store_false")
    add_path_args(ap)
    args = ap.parse_args(argv)

    paths = resolve(args)
    print(paths.describe())
    print(f"  torch threads = {set_torch_threads()}")
    cfg = arm_config("lookup", args.seed, paths.input, train_steps=1).replace(
        w_thick=0.2, w_prog=0.0, w_cross=0.0
    )
    # Warnings are the point here, not noise: AssumedThicknessWarning and
    # OverlappingSlabsWarning are two of the three failure modes this script exists to find.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        volume = load_training_volume(cfg, paths.input, flattened=args.flattened)
    names = sorted({w.category.__name__ for w in caught})

    geo = geometry(volume, cfg)
    print(f"\n  {cfg.content_hash()}  thickness_ratio = {cfg.thickness_ratio}")
    for key, value in geo.items():
        print(f"    {key:<28}{value}")
    if names:
        print(f"    {'load warnings':<28}{', '.join(names)}")

    data = TrainingData.build(volume, cfg)
    model = CTFFlow(cfg, data, embeddings_factory(volume)(cfg), grf_seed=args.seed)
    teacher = EMATeacher(model, cfg)

    print(f"\n    {'mean_density (cells/um^3)':<28}{data.stats.mean_density:.6g}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sweep = binding_sweep(model, teacher, cfg, args.draws, args.seed, FACTORS)

    print(f"\n  L_thick over {args.draws} random planes, student scaled by 1+e:")
    print(f"    {'rel. error e':>14}{'non-zero':>10}{'median':>14}{'max':>14}{'grazed':>9}")
    for row in sweep["by_relative_error"].values():
        print(
            f"    {row['relative_error']:>14.4g}{row['fraction_nonzero']:>10.3f}"
            f"{row['median_total']:>14.6g}{row['max_total']:>14.6g}"
            f"{row['fraction_grazed']:>9.3f}"
        )

    zero_row = sweep["by_relative_error"]["0"]
    if zero_row["fraction_nonzero"] > 0.0:
        print(
            "\n  ⚠️ The term is non-zero with the two branches IDENTICAL. The common-random-"
            "numbers\n     cancellation is not holding here, so the rows below it are measuring"
            "\n     estimator noise rather than disagreement. Do not read them."
        )
    noticed = sweep["smallest_error_noticed"]
    charging = max(
        (r["fraction_nonzero"] for r in sweep["by_relative_error"].values() if r["relative_error"]),
        default=0.0,
    )
    # From the LARGEST disagreement, where a slab that fits must charge: below the Poisson
    # hinge an exact zero means "hinged", not "did not fit", and the two are indistinguishable.
    grazing = float(
        max(sweep["by_relative_error"].values(), key=lambda r: r["relative_error"])[
            "fraction_grazed"
        ]
    )
    verdict = (
        "ILL-POSED"
        if geo["slabs_overlap"]
        else "BINDS"
        if charging >= BINDS_FRACTION
        else "DOES NOT BIND"
        if charging <= 0.0
        else "MARGINAL"
    )
    print(
        f"\n  -> **{verdict}**  smallest relative count error the term notices: "
        + ("never, up to 30%" if noticed is None else f"{noticed:.2%}")
        + f"   (grazed draws {grazing:.3f})"
    )
    if verdict == "BINDS":
        print("     The geometry admits the term, and the sweep says how wrong the student has")
        print("     to be before it charges at all — `_poisson_consistency` hinges at one")
        print("     Poisson sd, so any relative error below ~1/sqrt(N) costs exactly zero.")
        print("     A7 can run here. Whether L_thick HELPS is what the six fits measure.")
    elif verdict == "DOES NOT BIND":
        print("     A 30% student-side error charges nothing, so L_thick has no gradient on this")
        print("     geometry and A7 here would test L_prog alone. That is a different")
        print("     experiment: stop and let the user choose it rather than have it happen.")
        print(f"     Grazed draws {grazing:.3f} — near 1.0 means the thick slab does not fit in")
        print("     the volume's z-extent; near 0.0 means the Poisson hinge is swallowing it.")
    elif verdict == "ILL-POSED":
        print("     Sections are thicker than the spacing between them, so the same tissue is")
        print("     observed twice and the coarse-graining identity L_thick rests on is false.")
        print("     Not a matter of degree; do not run A7 on this dataset.")
    else:
        print("     Between the bounds. Report and let the user decide.")
    if geo["thickness_is_assumed"]:
        print(
            "\n  ⚠️ Section.thickness was DEFAULTED from median spacing, not measured. L_thick "
            "still\n     binds, but every A7 number rests on that assumption and must say so."
        )

    record = {
        "dataset": paths.dataset,
        "holdout": paths.holdout,
        "config_hash": cfg.content_hash(),
        "seed": int(args.seed),
        "geometry": geo,
        "load_warnings": names,
        "mean_density": float(data.stats.mean_density),
        "sweep": sweep,
        "verdict": verdict,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(record, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
