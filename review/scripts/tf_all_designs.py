"""SpatialCPA-TF's donor decision on every hold-out configuration (REVIEW_NOTES §15).

    python scripts/tf_all_designs.py [out.md]

A copy's fidelity depends only on the (donor, target) pair, so one 7 × 7 matrix
of section-copied-onto-section metrics (the scorer's per-section functions,
``run_spatialcpa_tf.section_metrics``) scores every design. For each design that
holds out 1–3 interior STARmap sections, the TF decision is made from that
design's *training* pairs only (``fidelity_composite``, ``fit_fidelity_model``,
``choose_donor``, as the wrapper does) and compared with the nearest section,
the always-deeper flank and the oracle flank. Gains are the per-section SNR
composite (``run_spatialcpav18_combo.SIGMA`` units, summed over the design's
targets) against the nearest section. Writes ``out.md``
(default ``reproduced/tf_all_designs.md``) and the matrix beside it.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark" / "src" / "bench3" / "methods"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_spatialcpa_tf as TF  # noqa: E402
from summarize_combo import SIGMA  # noqa: E402

DATA = ROOT / "benchmark" / "data" / "sections" / "starmap_visual_cortex" / "data.h5ad"


def pair_matrix(path):
    a = ad.read_h5ad(path)
    ep = TF._load_scorer()
    pp = dict(a.uns.get("paper_protocol") or {})
    markers = [str(g) for g in pp.get("marker_genes", [])] or ep.MARKER_GENES
    layers = ([str(g) for g in pp.get("layer_superficial", [])] or ep.LAYER_SUPERFICIAL,
              [str(g) for g in pp.get("layer_deep", [])] or ep.LAYER_DEEP)
    genes = np.asarray(a.var_names, dtype=str)
    order = np.argsort(genes, kind="stable")
    X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
    X = np.asarray(X, dtype=np.float32)[:, order]
    secs = a.obs["section"].values.astype(str)
    xyz = np.asarray(a.obsm["spatial"], dtype=np.float64)
    ct = a.obs["cell_type"].values.astype(str)
    ids = sorted(set(secs.tolist()), key=lambda s: float(np.median(xyz[secs == s, 2])))
    z = {s: float(np.median(xyz[secs == s, 2])) for s in ids}
    P = {}
    for i in ids:
        for k in ids:
            if i != k:
                mi, mk = secs == i, secs == k
                P[f"{i}->{k}"] = {key: float(v) for key, v in TF.section_metrics(
                    ep, X[mi], xyz[mi, :2], ct[mi], X[mk], xyz[mk, :2], ct[mk],
                    [str(g) for g in genes[order]], markers, layers).items() if np.isscalar(v)}
    return ids, z, P


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else ROOT / "reproduced" / "tf_all_designs.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    ids, z, P = pair_matrix(DATA)
    out.with_suffix(".json").write_text(json.dumps({"z": z, "pairs": P}, indent=1))

    def truth(d, h):
        return sum(s * P[f"{d}->{h}"][m] / SIGMA[m] for m, s in TF.FIDELITY_METRICS)

    rows, tot = [], {"tf": [], "deeper": [], "oracle": []}
    for r in (1, 2, 3):
        for H in itertools.combinations(ids[1:-1], r):
            T = [s for s in ids if s not in H]
            pairs = [(i, k) for i in T for k in T if i != k]
            Y, _used = TF.fidelity_composite([P[f"{i}->{k}"] for i, k in pairs])
            model = TF.fit_fidelity_model(pairs, z, Y)
            g = {"tf": 0.0, "deeper": 0.0, "oracle": 0.0}
            picks = []
            for h in H:
                lo = max((t for t in T if z[t] < z[h]), key=z.get)
                hi = min((t for t in T if z[t] > z[h]), key=z.get)
                near = hi if (z[h] - z[lo]) > (z[hi] - z[h]) else lo
                up, _info = TF.choose_donor(model, z[lo], z[hi], z[h])
                tf = hi if up else lo
                base = truth(near, h)
                g["tf"] += truth(tf, h) - base
                g["deeper"] += truth(hi, h) - base
                g["oracle"] += max(truth(lo, h), truth(hi, h)) - base
                picks.append(f"{h.split('_')[-1]}←{tf.split('_')[-1]}")
            for k in tot:
                tot[k].append(g[k])
            rows.append(f"| {', '.join(s.split('_')[-1] for s in H)} | {' '.join(picks)} | "
                        f"{g['tf']:+.1f} | {g['deeper']:+.1f} | {g['oracle']:+.1f} |")
    lines = ["## SpatialCPA-TF on every 1–3-section hold-out of STARmap", "",
             "SNR gain over copying the nearest section (summed over the held-out sections).", "",
             "| held out | TF donors | TF | always deeper | oracle flank |", "|---|---|---|---|---|",
             *rows, ""]
    for k, v in tot.items():
        v = np.array(v)
        lines.append(f"- {k}: total {v.sum():+.1f}, mean {v.mean():+.1f}; designs gaining "
                     f"{int((v > 0).sum())}, losing {int((v < 0).sum())} of {len(v)}")
    text = "\n".join(lines) + "\n"
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
