"""Vascular-architecture metrics for generated sections (benchmark-pbya-v3, review).

An evaluation-side add-on: it reads each run's existing ``prediction.h5`` and the
dataset file the run was scored against, and writes ``vascular_metrics.json``
beside ``metrics.json``. No method and no ``prepare_dataset`` is rerun, and the
pinned scorer (``evaluate_paper.py``, ``metrics.json``) is not touched.

What is measured — the *organization* of vascular cells, not vascular function
(expression cannot show perfusion):

``vasc_dist``    distance from every non-vascular cell to its nearest vascular
                 cell: are cells as close to vessels as in the real section?
``vasc_nn``      distance from every vascular cell to its nearest other vascular
                 cell: are vascular cells as clustered / chained as real ones
                 (the 2-D proxy for vessel continuity)?
``vasc_niche``   the expression enrichment around vascular cells: the mean
                 rank-expression of each vascular cell's k nearest NON-vascular
                 cells, minus the non-vascular mean, over the non-marker genes.
                 Scored as ``1 - |e_pred - e_gt| / |e_null - e_gt|`` (L2 over
                 genes; null as below), with ``vasc_niche_r`` = Pearson r of the
                 two enrichment vectors alongside. Non-vascular neighbours only:
                 vascular cells chain along vessels, so their nearest cells are
                 mostly other vascular cells, and an all-cell neighbourhood
                 measures the vascular profile itself, not its niche.
``vasc_field_r`` Pearson r between the binned 2-D density of vascular cells in
                 the prediction (posed with ``align_by_expression``, as the paper
                 metrics are) and in the GT.

With a cell-type label space shared by the prediction and the GT, also:

``vasc_type_frac_ratio``  predicted vascular-type fraction / GT fraction.
``vasc_type_dist``        ``vasc_dist`` per non-vascular cell type, averaged.

Every distance is divided by the section's own median nearest-neighbour spacing,
so it is unit- and scale-free. Each distance metric is reported raw (``*_w1``,
1-Wasserstein between prediction and GT distributions) and calibrated as
``1 - W1 / W1(null)``: the null is the GT section with its vascular labels
reassigned to random cells (``NULL_REPS`` draws), i.e. vessels scattered
anywhere. 1 = the real organization, 0 = no better than scattered, < 0 worse.

Which cells are vascular — one rule, applied identically to prediction and GT,
fixed by the dataset alone:

* ``VASCULAR_SPEC[dataset]["markers"]``: the vascular marker genes.
* The vascular *fraction* f: the fraction of the dataset's cells in its most
  marker-enriched cell type (all sections of the dataset file — evaluation-side,
  ground truth only), or ``VASCULAR_SPEC[dataset]["frac"]`` when the dataset has
  no cell types.
* In each section — prediction or GT — the vascular cells are the top f fraction
  by mean per-gene rank of the markers. Rank-based, so every method is scored the
  same whatever its output scale; detection cannot be used (STARmap's Flt1 is
  nonzero in ~100% of cells). Under this rule the vascular fraction is equal by
  construction: placement is scored, density is not. Density is scored only by
  the type rule (``vasc_type_frac_ratio``).

A dataset absent from ``VASCULAR_SPEC``, or whose markers are not in the panel,
gets ``"applicable": false`` and a reason — never a fallback.

Reference row: for every held-out section, the nearest real *input* section (by
z; ties to the lower) is scored as if it were the prediction (``ref_neighbour_*``)
— what copying the nearest real section achieves.

Usage:
    python -m src.bench3.evaluate_vascular                  # what's missing
    python -m src.bench3.evaluate_vascular --force
    python -m src.bench3.evaluate_vascular --methods spatialz spatialcpav18_gen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from ._v2bridge import load_prediction  # noqa: F401  (also puts src/benchmark on sys.path)
from .align import align_by_expression, binned_gene_field, scoring_columns
from .config import (DATASET_NAME, FIELD_GRID, MARKER_GENES, RANDOM_SEED, RESULTS_DIR,
                     SPATIAL_K, dataset_path, resolve_dataset_arg)

from benchmark.evaluate_generation import _rank_normalize  # noqa: E402

# Vascular marker genes per dataset (the GT decides everything else).
VASCULAR_SPEC = {
    "starmap_visual_cortex": {"markers": ["Flt1"]},
}
# Random vascular relabelings of the GT averaged into the null.
NULL_REPS = 5
# A cell type enters vasc_type_dist with at least this many cells on both sides.
TYPE_MIN_CELLS = 20
# The type rule needs this fraction of predicted labels inside the GT label set.
LABEL_SPACE_MIN = 0.95
# Sections with fewer cells than this on either side are not scored.
MIN_CELLS = 30

OUT_NAME = "vascular_metrics.json"
POOLED = ("vasc_dist_w1", "vasc_dist", "vasc_nn_w1", "vasc_nn", "vasc_niche", "vasc_niche_r",
          "vasc_field_r", "vasc_type_frac_ratio", "vasc_type_dist")


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dense(X):
    return X.toarray() if sp.issparse(X) else np.asarray(X)


def median_spacing(xy):
    """Median nearest-neighbour distance of a section (its length unit)."""
    if len(xy) < 2:
        return 1.0
    d, _ = cKDTree(xy).query(xy, k=2)
    s = float(np.median(d[:, 1]))
    return s if s > 0 else 1.0


def marker_vascular(X_markers, frac):
    """Vascular mask: the top ``frac`` of cells by mean per-gene rank of the
    marker columns ``X_markers`` (n, m). At least one cell."""
    n = X_markers.shape[0]
    score = _rank_normalize(X_markers).mean(axis=1)
    k = min(n, max(1, int(round(frac * n))))
    mask = np.zeros(n, dtype=bool)
    mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask


def dist_to_vascular(xy, vasc):
    """Each non-vascular cell's distance to its nearest vascular cell."""
    if vasc.sum() == 0 or (~vasc).sum() == 0:
        return np.zeros(0)
    d, _ = cKDTree(xy[vasc]).query(xy[~vasc], k=1)
    return np.asarray(d, dtype=np.float64)


def vascular_nn(xy, vasc):
    """Each vascular cell's distance to its nearest other vascular cell."""
    if vasc.sum() < 2:
        return np.zeros(0)
    d, _ = cKDTree(xy[vasc]).query(xy[vasc], k=2)
    return np.asarray(d[:, 1], dtype=np.float64)


def niche_enrichment(xy, R, vasc, cols, k=SPATIAL_K):
    """Mean rank-expression of the ``k`` nearest NON-vascular cells of each
    vascular cell, minus the mean over all non-vascular cells, over gene columns
    ``cols``. Returns (len(cols),)."""
    other = ~vasc
    if vasc.sum() == 0 or other.sum() <= k:
        return np.full(len(cols), np.nan)
    R_o = R[other][:, cols]
    _, nb = cKDTree(xy[other]).query(xy[vasc], k=k)
    return R_o[np.asarray(nb).ravel()].mean(axis=0) - R_o.mean(axis=0)


def _l2(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.linalg.norm(a[ok] - b[ok])) if ok.any() else np.nan


def _pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _w1(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(wasserstein_distance(a, b))


def _calibrated(w, w_null):
    if not np.isfinite(w) or not np.isfinite(w_null) or w_null <= 0:
        return np.nan
    return 1.0 - w / w_null


def _null_masks(n, k, seed):
    rng = np.random.default_rng(seed)
    for _ in range(NULL_REPS):
        m = np.zeros(n, dtype=bool)
        m[rng.choice(n, k, replace=False)] = True
        yield m


def vascular_field_r(pred_xy, pred_vasc, gt_xy, gt_vasc, grid=FIELD_GRID):
    """Pearson r between binned vascular-fraction fields on one lattice."""
    allxy = np.vstack([pred_xy, gt_xy])
    xe = np.linspace(allxy[:, 0].min(), allxy[:, 0].max(), grid + 1)
    ye = np.linspace(allxy[:, 1].min(), allxy[:, 1].max(), grid + 1)
    pm, po = binned_gene_field(pred_xy, pred_vasc.astype(float), xe, ye, grid)
    gm, go = binned_gene_field(gt_xy, gt_vasc.astype(float), xe, ye, grid)
    both = po & go
    return _pearson(pm[both], gm[both]) if both.sum() >= 4 else np.nan


# --------------------------------------------------------------------------- #
# Dataset-level rule                                                           #
# --------------------------------------------------------------------------- #
def vascular_rule(adata, dataset):
    """The vascular rule for a dataset, from its ground truth only:
    ``{"applicable", "markers", "frac", "vascular_type", ...}``."""
    spec = VASCULAR_SPEC.get(dataset)
    if spec is None:
        return {"applicable": False,
                "reason": f"no vascular markers declared for dataset {dataset!r} "
                          f"(VASCULAR_SPEC in evaluate_vascular.py)"}
    genes = [str(g) for g in adata.var_names]
    markers = [g for g in spec["markers"] if g in genes]
    if not markers:
        return {"applicable": False,
                "reason": f"none of the vascular markers {spec['markers']} is in "
                          f"the {dataset!r} panel"}
    rule = {"applicable": True, "markers": markers,
            "markers_requested": list(spec["markers"])}
    if "cell_type" in adata.obs.columns:
        mk = [genes.index(g) for g in markers]
        score = _rank_normalize(_dense(adata.X[:, mk])).mean(axis=1)
        types = adata.obs["cell_type"].astype(str).values
        labels = sorted(np.unique(types))
        means = {c: float(score[types == c].mean()) for c in labels}
        vt = max(labels, key=lambda c: (means[c], c))
        others = [means[c] for c in labels if c != vt]
        rule.update(vascular_type=vt, frac=float((types == vt).mean()),
                    frac_source=f"fraction of cell type {vt!r} (highest mean marker rank)",
                    type_score=means[vt],
                    next_type_score=max(others) if others else None)
    elif "frac" in spec:
        rule.update(vascular_type=None, frac=float(spec["frac"]),
                    frac_source="VASCULAR_SPEC frac")
    else:
        return {"applicable": False,
                "reason": f"{dataset!r} has no cell_type and VASCULAR_SPEC gives no frac"}
    return rule


# --------------------------------------------------------------------------- #
# One section                                                                  #
# --------------------------------------------------------------------------- #
def section_metrics(pred_X, pred_xy, pred_types, gt_X, gt_xy, gt_types, genes, rule,
                    seed=RANDOM_SEED, markers=MARKER_GENES):
    """Vascular metrics of one predicted section against its GT section.

    ``pred_X`` / ``gt_X``: expression (any scale) with columns ``genes``;
    ``*_xy``: (n, 2) coordinates; ``*_types``: string labels or None.
    """
    mk = [genes.index(g) for g in rule["markers"]]
    rest = [j for j in range(len(genes)) if j not in mk]
    pR, gR = _rank_normalize(pred_X), _rank_normalize(gt_X)
    frac = rule["frac"]
    p_v, g_v = marker_vascular(pred_X[:, mk], frac), marker_vascular(gt_X[:, mk], frac)
    p_xy = pred_xy / median_spacing(pred_xy)
    g_xy = gt_xy / median_spacing(gt_xy)

    m = {"n_pred_vascular": int(p_v.sum()), "n_gt_vascular": int(g_v.sum())}
    p_d, g_d = dist_to_vascular(p_xy, p_v), dist_to_vascular(g_xy, g_v)
    p_n, g_n = vascular_nn(p_xy, p_v), vascular_nn(g_xy, g_v)
    nulls = list(_null_masks(len(g_xy), int(g_v.sum()), seed))
    w_d, w_n = _w1(p_d, g_d), _w1(p_n, g_n)
    w_d0 = float(np.mean([_w1(dist_to_vascular(g_xy, z), g_d) for z in nulls]))
    w_n0 = float(np.mean([_w1(vascular_nn(g_xy, z), g_n) for z in nulls]))
    m.update(vasc_dist_w1=w_d, vasc_dist_w1_null=w_d0, vasc_dist=_calibrated(w_d, w_d0),
             vasc_nn_w1=w_n, vasc_nn_w1_null=w_n0, vasc_nn=_calibrated(w_n, w_n0))

    g_e = niche_enrichment(g_xy, gR, g_v, rest)
    p_e = niche_enrichment(p_xy, pR, p_v, rest)
    null_e = [niche_enrichment(g_xy, gR, z, rest) for z in nulls]
    m["vasc_niche_l2"] = _l2(p_e, g_e)
    m["vasc_niche_l2_null"] = float(np.nanmean([_l2(e, g_e) for e in null_e]))
    m["vasc_niche"] = _calibrated(m["vasc_niche_l2"], m["vasc_niche_l2_null"])
    m["vasc_niche_r"] = _pearson(p_e, g_e)

    cols, _basis = scoring_columns(genes, markers, gt_xy, gR, spatial_k=SPATIAL_K)
    pred_al, _info = align_by_expression(pred_xy, pR[:, cols], gt_xy, gR[:, cols],
                                         grid=FIELD_GRID, seed=seed)
    m["vasc_field_r"] = vascular_field_r(pred_al, p_v, gt_xy, g_v)
    m["vasc_field_r_null"] = float(np.nanmean(
        [vascular_field_r(gt_xy, z, gt_xy, g_v) for z in nulls]))

    vt = rule.get("vascular_type")
    shared = (vt is not None and pred_types is not None and gt_types is not None
              and np.isin(pred_types, np.unique(gt_types)).mean() >= LABEL_SPACE_MIN)
    m["type_rule"] = bool(shared)
    if shared:
        pt_v, gt_v = pred_types == vt, gt_types == vt
        m["vasc_type_frac_ratio"] = (float(pt_v.mean() / gt_v.mean())
                                     if gt_v.mean() > 0 else np.nan)
        scores = []
        if pt_v.any() and gt_v.any():
            gt_null = list(_null_masks(len(g_xy), int(gt_v.sum()), seed + 1))
            pd_all, gd_all = dist_to_vascular(p_xy, pt_v), dist_to_vascular(g_xy, gt_v)
            pt_rest, gt_rest = pred_types[~pt_v], gt_types[~gt_v]
            for c in np.unique(gt_rest):
                if (gt_rest == c).sum() < TYPE_MIN_CELLS or (pt_rest == c).sum() < TYPE_MIN_CELLS:
                    continue
                w = _w1(pd_all[pt_rest == c], gd_all[gt_rest == c])
                w0 = []
                for z in gt_null:
                    dz = dist_to_vascular(g_xy, z)
                    w0.append(_w1(dz[gt_types[~z] == c], gd_all[gt_rest == c]))
                scores.append(_calibrated(w, float(np.nanmean(w0))))
        m["vasc_type_dist"] = float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan
    return m


# --------------------------------------------------------------------------- #
# One prediction                                                               #
# --------------------------------------------------------------------------- #
def _nearest_input_section(sections, z_by, target, holdout):
    inputs = [s for s in sections if s not in holdout]
    if not inputs:
        return None
    zt = z_by[target]
    return min(inputs, key=lambda s: (abs(z_by[s] - zt), z_by[s]))


def evaluate_vascular(prediction_path, h5ad_path, output_path=None, with_reference=True):
    """Vascular metrics of one ``prediction.h5``, per held-out section and pooled
    (weighted by GT cell count, as ``evaluate_paper``)."""
    import anndata as ad
    pred = load_prediction(prediction_path)
    holdout = [str(s) for s in pred["holdout_sections"]]
    adata = ad.read_h5ad(h5ad_path)
    dataset = str(adata.uns.get("dataset_name", DATASET_NAME))
    out = {"method": pred["method_name"], "dataset": dataset,
           "dataset_file": str(h5ad_path), "dataset_sha256": _sha256(h5ad_path),
           "holdout_sections": holdout,
           "eval": "vascular architecture (evaluate_vascular.py)"}
    rule = vascular_rule(adata, dataset)
    out["rule"] = rule
    if not rule["applicable"]:
        out["applicable"] = False
        return _finish(out, output_path)
    out["applicable"] = True

    genes = sorted(set(map(str, pred["gene_names"])) & set(map(str, adata.var_names)))
    pgi = [list(map(str, pred["gene_names"])).index(g) for g in genes]
    ggi = [list(map(str, adata.var_names)).index(g) for g in genes]
    if any(g not in genes for g in rule["markers"]):
        out.update(applicable=False, reason="the prediction lacks a vascular marker gene")
        return _finish(out, output_path)
    pred_X_all = _dense(pred["X"][:, pgi])
    # the pose for vasc_field_r is chosen on the dataset's own marker panel, as
    # evaluate_paper chooses it
    pp = dict(adata.uns.get("paper_protocol") or {})
    align_markers = [str(g) for g in pp.get("marker_genes", [])] or list(MARKER_GENES)

    secs = adata.obs["section"].astype(str).values
    xyz = np.asarray(adata.obsm["spatial"], dtype=np.float64)
    labels = sorted(np.unique(secs))
    z_by = {s: float(np.median(xyz[secs == s, 2])) for s in labels}
    has_types = "cell_type" in adata.obs.columns
    gt_types_all = adata.obs["cell_type"].astype(str).values if has_types else None

    def gt_part(sec):
        g = secs == sec
        return (_dense(adata.X[g][:, ggi]).astype(np.float64), xyz[g, :2],
                None if gt_types_all is None else gt_types_all[g])

    per, ref, weights = {}, {}, []
    for sec in holdout:
        pm = pred["section"] == sec
        gX, gxy, gtt = gt_part(sec)
        if pm.sum() < MIN_CELLS or len(gxy) < MIN_CELLS:
            per[sec] = {"error": f"too few cells (pred={int(pm.sum())}, gt={len(gxy)})"}
            continue
        pxy = np.column_stack([pred["x"][pm], pred["y"][pm]]).astype(np.float64)
        per[sec] = section_metrics(pred_X_all[pm].astype(np.float64), pxy,
                                   pred["cell_type"][pm].astype(str), gX, gxy, gtt,
                                   genes, rule, markers=align_markers)
        weights.append(len(gxy))
        if with_reference:
            nb = _nearest_input_section(labels, z_by, sec, set(holdout))
            if nb is not None:
                rX, rxy, rtt = gt_part(nb)
                ref[sec] = dict(section_metrics(rX, rxy, rtt, gX, gxy, gtt, genes, rule,
                                                markers=align_markers),
                                source_section=nb)

    usable = [s for s in per if "error" not in per[s]]
    w = np.asarray(weights, dtype=float)
    for key in POOLED:
        for prefix, table in (("", per), ("ref_neighbour_", ref)):
            vals = np.array([table.get(s, {}).get(key, np.nan) for s in usable], dtype=float)
            ok = np.isfinite(vals)
            out[prefix + key] = (float(np.average(vals[ok], weights=w[ok]))
                                 if ok.any() else None)
    out["per_section"] = per
    out["ref_neighbour_per_section"] = ref
    return _finish(out, output_path)


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _finish(out, output_path):
    out = _clean(out)
    if output_path:
        Path(output_path).write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# CLI: every prediction under a results tree                                   #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Vascular metrics for every prediction")
    ap.add_argument("--results-dir", default=str(RESULTS_DIR))
    ap.add_argument("--dataset", default=None,
                    help="ground-truth h5ad (name or path); default: resolved per "
                         "prediction from its results path, as evaluate_all does")
    ap.add_argument("--dataset-name", default=None,
                    help="restrict to predictions of this dataset")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--force", action="store_true",
                    help=f"recompute even when {OUT_NAME} exists")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the nearest-input-section reference row")
    args = ap.parse_args()

    override = resolve_dataset_arg(args.dataset) if args.dataset else None
    root = Path(args.results_dir)
    done = skipped = failed = 0
    for pred_path in sorted(root.rglob("prediction.h5")):
        rel = pred_path.relative_to(root).parts
        method = rel[0]
        if method.startswith("_") or method == "summary":
            continue
        if args.methods and method not in args.methods:
            continue
        ds_name = "/".join(rel[1:-2]) or DATASET_NAME
        if args.dataset_name and ds_name != args.dataset_name:
            continue
        gt_path = override if override is not None else dataset_path(ds_name)
        out_path = pred_path.parent / OUT_NAME
        if not Path(gt_path).exists():
            print(f"SKIP {'/'.join(rel[:-1])}: no ground truth at {gt_path}")
            skipped += 1
            continue
        if out_path.exists() and not args.force:
            skipped += 1
            continue
        print(f"Vascular metrics for {'/'.join(rel[:-1])} ...")
        try:
            m = evaluate_vascular(pred_path, gt_path, out_path,
                                  with_reference=not args.no_reference)
            if m.get("applicable"):
                print(f"  dist={m.get('vasc_dist')} nn={m.get('vasc_nn')} "
                      f"niche={m.get('vasc_niche')} field_r={m.get('vasc_field_r')}"
                      f"  (copy-nearest reference dist={m.get('ref_neighbour_vasc_dist')})")
            else:
                print(f"  not applicable: {m['rule'].get('reason') or m.get('reason')}")
            done += 1
        except Exception as e:  # report and keep going, like evaluate_all
            print(f"  FAILED: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\nScored {done}, skipped {skipped}, failed {failed}")


if __name__ == "__main__":
    main()
