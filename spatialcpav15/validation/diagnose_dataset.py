"""Localize where SpatialCPA-v15 loses accuracy **on a dataset you have**.

The maintainer-visible symptom on `imc_breast_cancer` is that v15 scores far
below both v8 *and its own achievable ceiling*: cell-matched `pearson_median`
0.045 against v8's 0.261, while the same architecture reaches ~0.22 with a
correct layout on synthetic data.  None of the synthetic stacks in this
directory reproduces that, so the cause has to be found on the real data.

This script does exactly that.  It fits v15 once, then scores a **ladder** of
predictions that swap v15's machinery in one piece at a time, each written as a
real `prediction.h5` and scored by the **real** benchmark evaluators — so every
number is directly comparable to the leaderboard:

    A  copy baseline      GT layout + nearest real cell's profile   (v8-like ceiling)
    B  oracle expression  GT layout + GT library + v15 diffusion
    C  sampled library    GT layout + v15 library  + v15 diffusion
    D  predicted types    GT positions + v15 types + v15 library + v15 diffusion
    E  full v15           everything generated                     (what ships)

Reading the ladder:

* **A high, B much lower**  -> the expression model is the problem.
* **B ~ A, C much lower**   -> the per-cell library assignment is the problem.
* **C ~ B, D much lower**   -> cell-type assignment from the field is the problem.
* **D ~ C, E much lower**   -> cell *placement* (the point process) is the problem.

The held-out slice is used only to *score* these variants; nothing is fed back
into a benchmark submission.  Rows B-D deliberately use ground truth, so they are
diagnostics, not results.

Usage
-----
    python spatialcpav15/validation/diagnose_dataset.py \\
        <processed/data.h5ad> [--holdout SECTION] [--registration rigid|none|paste]

For the benchmark's own copy of a dataset:

    python spatialcpav15/validation/diagnose_dataset.py \\
        benchmark-pbya/data/processed/imc_breast_cancer/data.h5ad --registration rigid

The `spatialcpav15` package and `benchmark-pbya-v2` are found by walking up from
this file, so it works whether the package sits at `<root>/spatialcpav15` or
`<root>/src/spatialcpav15`.  Override with $SPATIALCPAV15_ROOT / $BENCH_V2_ROOT
if they live somewhere unrelated.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _locate(relative: str, env: str) -> Path:
    """Find a directory containing ``relative`` by walking up from this file.

    The package is not always a direct child of the repository root — it may sit
    under ``src/``, or be installed elsewhere entirely — so nothing here assumes
    a fixed depth.  ``$<env>`` overrides the search.
    """
    override = os.environ.get(env)
    candidates = ([Path(override)] if override else []) + \
        list(Path(__file__).resolve().parents) + [Path.cwd()] + list(Path.cwd().parents)
    seen = []
    for cand in candidates:
        seen.append(str(cand))
        if (cand / relative).exists():
            return cand
    raise SystemExit(
        f"Could not find '{relative}' above {Path(__file__).resolve()}.\n"
        f"Set ${env} to the directory that contains it, e.g.\n"
        f"    {env}=/path/to/repo python {Path(__file__).name} <data.h5ad>\n"
        f"Looked in:\n  " + "\n  ".join(dict.fromkeys(seen)))


# spatialcpav15 may live at <root>/spatialcpav15 or <root>/src/spatialcpav15.
PKG_ROOT = _locate("spatialcpav15/__init__.py", "SPATIALCPAV15_ROOT")
BENCH_ROOT = _locate("benchmark-pbya-v2/src/benchmark/__init__.py", "BENCH_V2_ROOT")
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(BENCH_ROOT / "benchmark-pbya-v2" / "src"))

import numpy as np                                     # noqa: E402
import anndata as ad                                   # noqa: E402
import scipy.sparse as sp                              # noqa: E402
from scipy.spatial import cKDTree                      # noqa: E402

from benchmark import leakage_guard as lg              # noqa: E402
from benchmark.evaluate import evaluate                # noqa: E402
from benchmark.evaluate_generation import evaluate_generation   # noqa: E402
from benchmark.methods import _v2_io                   # noqa: E402

from spatialcpav15 import (SpatialCPAv15, SpatialCPAv15Config,   # noqa: E402
                           build_stack_from_arrays)
from spatialcpav15.phase3_expression import (decode_latents,     # noqa: E402
                                             sample_expression_latents)

METRICS = ["pearson_median", "gen_coexpression_agreement", "gen_morans_agreement",
           "gen_celltype_composition", "gen_celltype_nhood_agreement",
           "gen_gene_var_pearson", "matching_rate"]


def _dense(X):
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float32)


def _prepare(adata):
    """Mirror the wrapper's normalization exactly."""
    import scanpy as sc
    et = adata.uns.get("expression_type", "raw_counts")
    raw = _dense(adata.X).copy() if et == "raw_counts" else None
    if et == "raw_counts":
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    elif et in ("log1p_normalized", "log2_normalized", "normalized"):
        pass
    elif et in ("fluorescence_intensity", "mean_intensity"):
        sc.pp.log1p(adata)
    else:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    return _dense(adata.X), raw, et


def _score(coords, X, types, section, genes, gt_path, wd, tag):
    """Write a prediction.h5 and score it with the real evaluators."""
    path = wd / f"pred_{tag}.h5"
    _v2_io.write_prediction_h5(
        {section: {"X": sp.csr_matrix(np.asarray(X, dtype=np.float32)),
                   "coords": np.asarray(coords, dtype=np.float64),
                   "cell_type": np.asarray(types).astype(str)}},
        genes, [section], {"diagnostic": tag}, 0.0, str(path), f"v15_diag_{tag}")
    out = evaluate(str(path), gt_path, output_path=None)
    gen = evaluate_generation(str(path), gt_path, output_path=None)
    for k, v in gen.items():
        if isinstance(v, (int, float)):
            out[f"gen_{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data")
    ap.add_argument("--holdout", default=None,
                    help="section to hold out (default: a middle interior one)")
    ap.add_argument("--registration", default="rigid",
                    choices=["rigid", "none", "paste"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    wd = Path(tempfile.mkdtemp(prefix="v15diag_"))
    full = ad.read_h5ad(args.data)
    secs_all = full.obs["section"].values.astype(str)
    order = sorted(np.unique(secs_all),
                   key=lambda s: float(np.median(full.obsm["spatial"][secs_all == s, 2])))
    holdout = args.holdout or order[len(order) // 2]
    print(f"dataset  : {args.data}")
    print(f"sections : {len(order)}   holdout: {holdout}   registration: {args.registration}")

    # ── training-only input, exactly as the harness builds it ────────────────
    train, _held = lg.split_holdout(full, [holdout])
    train = lg.reregister_training(train, method=args.registration)
    lg.assert_no_leakage(train, [holdout])

    X_log, X_raw, et = _prepare(train)
    print(f"expression_type = {et}   counts available = {X_raw is not None}")
    ct_all, names = lg.build_labels_train_only(
        train, "cell_type", np.ones(train.n_obs, dtype=bool), seed=args.seed)
    stack = build_stack_from_arrays(
        X_log, np.asarray(train.obsm["spatial"], dtype=np.float64),
        train.obs["section"].values.astype(str), ct_all, X_raw)

    cfg = SpatialCPAv15Config()
    cfg.seed = args.seed
    gen = SpatialCPAv15(stack, gene_names=train.var_names.tolist(),
                        cell_type_names=names, cfg=cfg)

    # ── ground truth for the held-out slice ──────────────────────────────────
    gt = full[secs_all == holdout]
    gt_xyz = np.asarray(gt.obsm["spatial"], dtype=np.float64)
    z_q = float(np.median(gt_xyz[:, 2]))
    genes = train.var_names.tolist()
    gt_types = (gt.obs["cell_type"].values.astype(str)
                if "cell_type" in gt.obs.columns
                else np.array(["NA"] * gt.n_obs))
    gt_raw = _dense(gt.X)
    lut = {n: i for i, n in enumerate(gen.cell_type_names)}
    gt_type_idx = np.array([lut.get(t, 0) for t in gt_types], dtype=np.int64)

    # v15's normalized coordinates for the GT positions
    u_gt, _ = gen.frame.to_normalized(gt_xyz[:, :2], gt_xyz[:, 2], None)
    z_norm = np.full(gt.n_obs, gen.frame.z_to_normalized(z_q))
    idx = gen.index
    a_out, a, b, b_out = gen.stack.brackets(z_q)
    allowed = np.array(sorted({a_out, a, b, b_out}))
    rng = np.random.default_rng(args.seed)
    lib_mean = float(idx.log_lib.mean())

    # per-cell library: GT (oracle) vs v15's sampler
    gt_lib = np.log(np.maximum(gt_raw.sum(axis=1), 1.0))
    v15_lib = gen._sample_library(gt_type_idx, allowed, np.random.default_rng(args.seed))

    def diffuse(u, types, log_lib):
        if not gen.expr.trained:
            return gen._latent_fallback(type("L", (), {"u": u, "types": types})(),
                                        allowed, rng)
        return sample_expression_latents(
            gen.expr, idx, gen.type_rep, u, np.full(u.shape[0], z_norm[0]),
            types, log_lib, allowed, z_q, gen.dz_unit, lib_mean, seed=args.seed)

    rows = {}

    # A — copy baseline at the GT layout (what a transport method effectively emits)
    pool = np.where(np.isin(idx.section, allowed))[0]
    _, nn = cKDTree(idx.u[pool]).query(u_gt, k=1)
    src = pool[np.atleast_1d(nn)]
    train_raw = X_raw if X_raw is not None else np.expm1(np.clip(X_log, 0, 30))
    rows["A copy @ GT layout"] = _score(
        gt_xyz, train_raw[src], np.array(gen.cell_type_names)[idx.types[src]],
        holdout, genes, args.data, wd, "A")

    # B — v15 expression, GT layout + GT library (oracle)
    Zb = diffuse(u_gt, gt_type_idx, gt_lib)
    rows["B v15 expr @ GT layout, GT lib"] = _score(
        gt_xyz, decode_latents(gen.space, Zb, gt_lib, gen.expr.readout, rng),
        gt_types, holdout, genes, args.data, wd, "B")

    # C — same, but v15's sampled library
    Zc = diffuse(u_gt, gt_type_idx, v15_lib)
    rows["C   + v15 library"] = _score(
        gt_xyz, decode_latents(gen.space, Zc, v15_lib, gen.expr.readout, rng),
        gt_types, holdout, genes, args.data, wd, "C")

    # D — GT positions, but types predicted by v15's field
    from spatialcpav15.phase2_layout import _bilinear
    from spatialcpav15.phase2_structure import complete_with_uncertainty
    field, _spread = complete_with_uncertainty(gen.structure, z_q,
                                               cfg.inference.n_structure_seeds)
    G = np.stack([_bilinear(np.clip(field[1 + g], 0.0, None), u_gt, gen.spec)
                  for g in range(gen.spec.n_groups)], axis=1)
    grp = G.argmax(axis=1)
    pred_types = np.array([np.where(gen.spec.group_of_type == g)[0][0] for g in grp])
    Zd = diffuse(u_gt, pred_types, v15_lib)
    rows["D   + v15 types"] = _score(
        gt_xyz, decode_latents(gen.space, Zd, v15_lib, gen.expr.readout, rng),
        np.array(gen.cell_type_names)[pred_types], holdout, genes, args.data, wd, "D")

    # E — the full shipped pipeline
    vs = gen.generate_virtual_slice(z=z_q)
    rows["E full v15 (shipped)"] = _score(
        vs.coords, vs.expression, vs.cell_type, holdout, genes, args.data, wd, "E")

    # ── report ───────────────────────────────────────────────────────────────
    hdr = ["pearson_median", "gen_coexpression_agreement", "gen_morans_agreement",
           "gen_celltype_composition", "gen_celltype_nhood_agreement"]
    short = ["pearson_med", "coexpr", "morans", "composition", "nhood"]
    print(f"\n{'variant':34s}" + "".join(f"{s:>14s}" for s in short))
    print("-" * (34 + 14 * len(short)))
    for k, m in rows.items():
        print(f"{k:34s}" + "".join(
            f"{(m.get(h) if m.get(h) is not None else float('nan')):14.3f}" for h in hdr))
    print(f"\nn_cells: GT={gt.n_obs}  shipped={vs.n_cells}")
    print("Read consecutive rows: the first large drop is the stage at fault.")
    print(f"(prediction files kept in {wd})")


if __name__ == "__main__":
    main()
