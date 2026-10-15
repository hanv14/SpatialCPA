"""Summarize SpatialCPA-X (REVIEW_NOTES §16).

    python scripts/summarize_x.py reproduced

Interpolation (paper, block 1, block 3): reproduced/seed_x against seed_tf,
seed_stable and seed_methods. Extrapolation (end-of-stack hold-outs from
``run_extrapolation``): reproduced/extrap/seed<s>/<method>. For each design and
method: the validation metrics, the raw-scale abundance metrics of
``evaluate_depth`` (``depth_metrics.json``; run ``python -m src.bench3.evaluate_depth
--results-dir <root>`` first), and the SNR composite / wins − losses of
spatialcpa_x against each method seed by seed. Writes ``<root>/x_summary.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_combo import GAP, HIDS, KEYS, SEEDS, metrics, snr, wl  # noqa: E402

DS = "starmap_visual_cortex"
X = "spatialcpa_x"
INTERP = {X: "seed_x", "spatialcpa_tf": "seed_tf",
          "spatialcpav18_gen_flow_cv_stable": "seed_stable",
          "spatialcpav18_gen_nearest_noflow": "seed_methods", "spatialcpav18_gen": "seed_methods"}
EXTRAP_METHODS = (X, "spatialcpa_tf", "spatialcpav18_gen_flow_cv_stable",
                  "spatialcpav18_gen_nearest_noflow", "spatialcpav18_gen")
EXTRAP = [f"extrap_{s}_{k}" for s in ("hi", "lo") for k in (1, 2, 3)]
SHORT = {"spatialcpa_x": "X", "spatialcpa_tf": "TF",
         "spatialcpav18_gen_flow_cv_stable": "flow_cv_stable",
         "spatialcpav18_gen_nearest_noflow": "nearest + no flow",
         "spatialcpav18_gen": "published v18"}
COLS = [k for k, _ in KEYS] + ["umap_mixing"]
CN = ["emb", "moranR", "moranMAE", "depth", "field", "loc", "rare", "detect", "UMAP"]
DEPTH = ("lib_abs_logratio", "gene_mean_abs_logratio", "gene_mean_log_r")


def _dir(root, sweep, method, seed, hid):
    return root / sweep / f"seed{seed}" / method / DS / hid


def _depth(root, sweep, method, seed, hid):
    f = _dir(root, sweep, method, seed, hid) / "depth_metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def _table(root, hids, where, label):
    out = [f"### {label}", "", "| design | method | seeds | " + " | ".join(CN)
           + " | lib |log| | gene-mean |log| | gene-mean r |", "|---|---|---|" + "---|" * (len(CN) + 3)]
    for hid in hids:
        for m in where:
            sw = where[m]
            rows = [r for sd in SEEDS if (r := metrics(root, sw, m, sd, hid)) is not None]
            deps = [d for sd in SEEDS if (d := _depth(root, sw, m, sd, hid)) is not None]
            if not rows:
                continue
            v = " | ".join(f"{np.mean([r['paper_' + k] for r in rows]):.3f}" for k in COLS)
            dv = " | ".join(f"{np.mean([d[k] for d in deps]):.3f}" if deps else "—" for k in DEPTH)
            out.append(f"| {GAP.get(hid, hid)} | {SHORT[m]} | {len(rows)} | {v} | {dv} |")
    return out + [""]


def _versus(root, hids, where, label):
    out = [f"### {label}: spatialcpa_x against each seed of the others", "",
           "| design | vs | SNR per seed | mean ± sd | wins−losses per seed |", "|---|---|---|---|---|"]
    for hid in hids:
        xs = [r for sd in SEEDS if (r := metrics(root, where[X], X, sd, hid)) is not None]
        if not xs:
            continue
        for m in where:
            if m == X:
                continue
            s_, w_ = [], []
            for sd in SEEDS:
                b = metrics(root, where[m], m, sd, hid)
                a = metrics(root, where[X], X, sd, hid) or xs[0]
                if b is not None:
                    s_.append(snr(a, b))
                    w_.append(wl(a, b))
            if s_:
                s = np.array(s_)
                out.append(f"| {GAP.get(hid, hid)} | {SHORT[m]} | {', '.join(f'{x:+.1f}' for x in s)} | "
                           f"{s.mean():+.1f} ± {s.std(ddof=1) if len(s) > 1 else 0:.1f} | "
                           f"{', '.join(f'{x:+d}' for x in w_)} |")
    return out + [""]


def _decisions(root, sweep, hids):
    out = ["### spatialcpa_x decisions (seed 1)", ""]
    for hid in hids:
        f = _dir(root, sweep, X, "1", hid) / "prediction.h5"
        if not f.exists():
            continue
        import h5py
        with h5py.File(f, "r") as h:
            raw = h["uns/method_params"][()]
        rec = json.loads(raw if isinstance(raw, str) else raw.decode())["spatialcpa_x"]
        mdl, dr = rec.get("model") or {}, rec["depth_response"]
        dec = "; ".join(f"{d['section'].split('_')[-1]}←{d['donor'].split('_')[-1]}"
                        f"{'' if d['donor'] == d['nearest'] else ' (not nearest)'} "
                        f"×{d['median_capture_ratio']:.2f}" for d in rec["decisions"])
        coef = mdl.get("coef", {})
        out.append(f"- {GAP.get(hid, hid)}: a = {coef.get('a', float('nan')):+.3f}, β = "
                   f"{coef.get('beta', float('nan')):+.3f}; depth response {dr['mode']}, mean "
                   f"slope {dr.get('mean_slope', float('nan')):+.4f}/µm, mean shrinkage "
                   f"{dr.get('mean_shrinkage', float('nan')):.2f}; donors (×median capture ratio): {dec}")
    return out + [""]


def main(argv):
    root = Path(argv[1])
    for k in EXTRAP:
        GAP.setdefault(k, k.replace("extrap_hi_", "deep end, ").replace("extrap_lo_", "shallow end, ")
                       + " held out")
    ex = {m: "extrap" for m in EXTRAP_METHODS}
    out = ["## SpatialCPA-X", ""]
    out += _table(root, HIDS, INTERP, "Interpolation (seed means)")
    out += _versus(root, HIDS, INTERP, "Interpolation")
    out += _decisions(root, "seed_x", HIDS)
    out += _table(root, EXTRAP, ex, "Extrapolation (seed means)")
    out += _versus(root, EXTRAP, ex, "Extrapolation")
    out += _decisions(root, "extrap", EXTRAP)
    text = "\n".join(out) + "\n"
    (root / "x_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
