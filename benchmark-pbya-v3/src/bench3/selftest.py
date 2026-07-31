"""Validate the v3 harness with synthetic reconstructions of known quality.

The methods themselves need their conda environments, but the *harness* — the
holdout split, the leakage guard, the shared input, the prediction format and
above all the paper metrics — can be exercised without them, by feeding the
evaluator reconstructions whose quality is known in advance:

``oracle``
    the real held-out cells, verbatim. The upper bound: every agreement metric
    should be at or near its maximum.
``flanking_copy``
    the nearest *training* slice, copied. A real, non-trivial baseline — it is
    essentially what a method that ignores interpolation does — so it should
    score well but below the oracle.
``spatial_scramble``
    the real held-out cells with the coordinates permuted. Every marginal is
    perfect, every spatial relationship is destroyed: the metrics that measure
    *spatial* fidelity must collapse while the distributional ones stay at 1.0.
    This is the discriminating case, and it is what separates the two families of
    metric — a method cannot pass the whole panel by matching distributions alone.
``random``
    expression drawn from the pooled value distribution, cells scattered
    uniformly, types assigned at random. The floor for *every* metric.

Any ordering violation is a bug in the metric, not in the reconstruction — so
this doubles as a regression test.

Usage:
    python -m src.bench3.selftest                 # paper design, all four probes
    python -m src.bench3.selftest --no-umap       # skip the (slow) UMAP
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .config import DATASET_PATH, RANDOM_SEED, SUMMARY_DIR, resolve_dataset_arg
from .design import build_designs
from .evaluate_paper import evaluate_paper
from .run_benchmark import build_input, dataset_meta

# Reuse the exact writer every real method wrapper uses, so the self-test
# exercises the real prediction contract rather than a lookalike.
import sys
from .config import V2_ROOT
sys.path.insert(0, str(V2_ROOT / "src" / "benchmark" / "methods"))
import _v2_io  # noqa: E402


PROBES = ("oracle", "flanking_copy", "spatial_scramble", "random")


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def make_probe(kind, full_adata, train_adata, targets, seed=RANDOM_SEED):
    """Build a synthetic ``results`` dict in the wrapper output format."""
    rng = np.random.default_rng(seed)
    secs_full = full_adata.obs["section"].values.astype(str)
    coords_full = np.asarray(full_adata.obsm["spatial"], dtype=np.float64)
    secs_train = train_adata.obs["section"].values.astype(str)
    coords_train = np.asarray(train_adata.obsm["spatial"], dtype=np.float64)

    results = {}
    for sec, z in targets:
        gm = secs_full == sec
        gt_X = _dense(full_adata.X[gm])
        gt_xy = coords_full[gm, :2]
        gt_ct = full_adata.obs["cell_type"].values[gm].astype(str)

        if kind == "oracle":
            X, xy, ct = gt_X, gt_xy, gt_ct

        elif kind == "flanking_copy":
            # Nearest training section by z — what a method that just copies the
            # closest real slice would emit.
            train_secs = np.unique(secs_train)
            zs = {s: float(np.median(coords_train[secs_train == s, 2]))
                  for s in train_secs}
            src = min(train_secs, key=lambda s: abs(zs[s] - z))
            m = secs_train == src
            X = _dense(train_adata.X[m])
            xy = coords_train[m, :2]
            ct = train_adata.obs["cell_type"].values[m].astype(str)

        elif kind == "spatial_scramble":
            # Permute the coordinates only: expression and cell type keep their
            # original pairing with each other but are detached from position,
            # so every marginal survives and every spatial relationship dies.
            # (Permuting xy and cell type *together* would leave cell-type
            # localization intact and would not exercise that metric.)
            X, xy, ct = gt_X, gt_xy[rng.permutation(len(gt_xy))], gt_ct

        elif kind == "random":
            # The true floor: expression drawn from the pooled value distribution
            # (so per-gene marginals are destroyed too, not just their spatial
            # arrangement), cells scattered uniformly, types assigned at random.
            n = len(gt_xy)
            X = rng.choice(gt_X.ravel(), size=gt_X.shape, replace=True)
            lo, hi = gt_xy.min(axis=0), gt_xy.max(axis=0)
            xy = rng.uniform(lo, hi, size=(n, 2))
            ct = rng.permutation(gt_ct)
        else:
            raise ValueError(f"unknown probe {kind!r}")

        coords = np.column_stack([xy, np.full(len(xy), float(z))])
        results[sec] = {"X": sp.csr_matrix(X.astype(np.float32)),
                        "coords": coords, "cell_type": np.asarray(ct, dtype=str)}
    return results


# Metric -> direction (+1 higher better). Checked for a sane ordering across the
# probes; ``spatial`` marks the metrics that MUST collapse under scrambling.
CHECKS = [
    ("paper_umap_mixing", +1, False),
    ("paper_morans_pearson", +1, True),
    ("paper_gearys_pearson", +1, True),
    ("paper_marker_field_r", +1, True),
    ("paper_marker_depth_r", +1, True),
    ("paper_gene_mean_spearman", +1, False),
    ("paper_celltype_localization", +1, True),
]


def main():
    ap = argparse.ArgumentParser(description="Validate the v3 harness end to end")
    ap.add_argument("--dataset", default=None,
                    help="dataset NAME or a path to a built data.h5ad")
    ap.add_argument("--design", default="paper", choices=["paper", "loo"])
    ap.add_argument("--probes", nargs="+", default=list(PROBES), choices=list(PROBES))
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--keep", help="write the probe predictions here instead of a tempdir")
    args = ap.parse_args()

    import anndata as ad

    args.dataset = str(resolve_dataset_arg(args.dataset))
    cfg = build_designs(args.dataset, design=args.design)[0]
    # The input cache is keyed by dataset name; without passing it, every dataset
    # would self-test against whatever is cached under the STARmap default — a
    # different gene panel and a different set of cells.
    ds_name, registration = dataset_meta(args.dataset)
    print(f"dataset={ds_name} design={args.design} holdout={cfg['holdout_id']} "
          f"held out={cfg['holdout_sections']}")

    info = build_input(cfg, dataset_path=args.dataset, registration=registration,
                       dataset_name=ds_name, verbose=True)
    targets = [(s, float(z)) for s, z in info["targets"]]
    full = ad.read_h5ad(args.dataset)
    train = ad.read_h5ad(info["input_path"])

    # The guard the wrappers run: no held-out cell may be in the method input.
    _v2_io.guard_no_holdout(train, [s for s, _ in targets])
    print(f"leakage guard passed: input has "
          f"{sorted(set(train.obs['section'].astype(str)))}, "
          f"{train.n_obs} cells (full dataset {full.n_obs})")

    workdir = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="bench3_selftest_"))
    workdir.mkdir(parents=True, exist_ok=True)
    genes = full.var_names.tolist()

    table = {}
    for probe in args.probes:
        out = workdir / probe / "prediction.h5"
        results = make_probe(probe, full, train, targets)
        _v2_io.write_prediction_h5(results, genes, [s for s, _ in targets],
                                   {"probe": probe}, 0.0, str(out),
                                   method_name=f"selftest_{probe}")
        print(f"\n── {probe} ──")
        m = evaluate_paper(str(out), args.dataset, use_umap=not args.no_umap)
        table[probe] = m
        for key, _, _ in CHECKS:
            v = m.get(key)
            print(f"  {key:<32s} {'NA' if v is None else f'{v:+.3f}'}")

    # ── Ordering assertions ───────────────────────────────────────────────────
    print("\n══ checks ══")
    failures = []

    def val(probe, key):
        v = table.get(probe, {}).get(key)
        return None if v is None else float(v)

    for key, direction, spatial in CHECKS:
        o, s, r = val("oracle", key), val("spatial_scramble", key), val("random", key)
        if o is None:
            print(f"  SKIP {key}: not computed")
            continue
        ok = True
        detail = f"oracle={o:+.3f}"
        if spatial and s is not None:
            detail += f" scramble={s:+.3f}"
            ok &= (o - s) * direction > 0.05
        if r is not None:
            detail += f" random={r:+.3f}"
            ok &= (o - r) * direction > 0.05
        print(f"  {'PASS' if ok else 'FAIL'} {key:<32s} {detail}")
        if not ok:
            failures.append(key)

    # The oracle is the same cells as the ground truth, so agreement metrics
    # should sit at their ceiling.
    for key in ("paper_morans_pearson", "paper_gearys_pearson",
                "paper_marker_field_r", "paper_marker_depth_r"):
        v = val("oracle", key)
        if v is not None and v < 0.95:
            print(f"  FAIL {key}: oracle should be ~1.0, got {v:+.3f}")
            failures.append(key)

    ratio = val("oracle", "paper_cell_count_ratio")
    if ratio is not None:
        print(f"  {'PASS' if abs(ratio - 1) < 1e-6 else 'FAIL'} "
              f"paper_cell_count_ratio  oracle={ratio:.4f} (expect 1.0)")

    summary_path = Path(SUMMARY_DIR) / "selftest_metrics.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(
        {p: {k: v for k, v in m.items() if k != "per_section"}
         for p, m in table.items()}, indent=2))
    print(f"\nMetrics written to {summary_path}")

    if not args.keep:
        shutil.rmtree(workdir, ignore_errors=True)
        print("(temporary probe predictions removed; use --keep DIR to retain them)")

    if failures:
        print(f"\n{len(set(failures))} metric(s) failed their ordering check: "
              f"{sorted(set(failures))}")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
