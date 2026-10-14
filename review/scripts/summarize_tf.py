"""Summarize SpatialCPA-TF against flow_cv_stable and the v18 baselines (REVIEW_NOTES §15).

    python scripts/summarize_tf.py reproduced

Reads reproduced/seed_tf (spatialcpa_tf), seed_stable (flow_cv_stable) and
seed_methods (nearest + no flow, published v18). Per design: the eight
validation metrics plus UMAP mixing (mean over the seeds present), each
method's donor choices, wall time, the SNR composite and wins − losses of
spatialcpa_tf against every seed of the others, and whether spatialcpa_tf's
predictions are identical across its seeds. Writes ``<root>/tf_summary.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_combo import GAP, HIDS, KEYS, SEEDS, metrics, snr, wl  # noqa: E402

TF = ("seed_tf", "spatialcpa_tf")
OTHERS = (("flow_cv_stable", ("seed_stable", "spatialcpav18_gen_flow_cv_stable")),
          ("nearest + no flow", ("seed_methods", "spatialcpav18_gen_nearest_noflow")),
          ("published v18", ("seed_methods", "spatialcpav18_gen")))
SHOW = [k for k, _ in KEYS] + ["umap_mixing"]
SHORT = ["emb", "moranR", "moranMAE", "depth", "field", "loc", "rare", "detect", "UMAP"]


def _pred(root, sweep, method, seed, hid):
    return root / sweep / f"seed{seed}" / method / "starmap_visual_cortex" / hid / "prediction.h5"


def _uns(path):
    with h5py.File(path, "r") as h:
        raw = h["uns/method_params"][()]
        wall = float(h["uns/wall_time_seconds"][()])
    return json.loads(raw if isinstance(raw, str) else raw.decode()), wall


def _same(a, b):
    with h5py.File(a, "r") as f, h5py.File(b, "r") as g:
        return all(np.array_equal(f[k][()], g[k][()]) for k in
                   ("X/data", "X/indices", "X/indptr", "obs/x", "obs/y", "obs/z", "obs/cell_type"))


def main(argv):
    root = Path(argv[1])
    out = ["## SpatialCPA-TF vs flow_cv_stable and v18", ""]
    out += ["| design | method | seeds | " + " | ".join(SHORT) + " | wall s |",
            "|---|---|---|" + "---|" * (len(SHORT) + 1)]
    for hid in HIDS:
        for name, (sw, m) in (("spatialcpa_tf", TF),) + OTHERS:
            rows, walls = [], []
            for sd in SEEDS:
                r = metrics(root, sw, m, sd, hid)
                if r is not None:
                    rows.append(r)
                    walls.append(_uns(_pred(root, sw, m, sd, hid))[1])
            if not rows:
                continue
            vals = " | ".join(f"{np.mean([r['paper_' + k] for r in rows]):.3f}" for k in SHOW)
            out.append(f"| {GAP[hid]} | {name} | {len(rows)} | {vals} | {np.mean(walls):.0f} |")
    out += ["", "### spatialcpa_tf against each seed of the others", "",
            "| design | vs | SNR composite per seed | mean ± sd | wins−losses per seed |",
            "|---|---|---|---|---|"]
    for hid in HIDS:
        tf = metrics(root, *TF, "1", hid)
        if tf is None:
            continue
        for name, (sw, m) in OTHERS:
            s_, w_ = [], []
            for sd in SEEDS:
                b = metrics(root, sw, m, sd, hid)
                if b is not None:
                    s_.append(snr(tf, b))
                    w_.append(wl(tf, b))
            if s_:
                s = np.array(s_)
                out.append(f"| {GAP[hid]} | {name} | {', '.join(f'{x:+.1f}' for x in s)} | "
                           f"{s.mean():+.1f} ± {s.std(ddof=1) if len(s) > 1 else 0:.1f} | "
                           f"{', '.join(f'{x:+d}' for x in w_)} |")
    out += ["", "### spatialcpa_tf decisions and determinism", ""]
    for hid in HIDS:
        p1 = _pred(root, *TF, "1", hid)
        if not p1.exists():
            continue
        rec = _uns(p1)[0]["transfer_fidelity"]
        co, se = rec["model"]["coef"], rec["model"]["se"]
        dec = "; ".join(f"{d['section']} → {d['donor']} (gain {d['gain']:+.2f} ± {d['se']:.2f})"
                        if d["gain"] is not None else f"{d['section']} → {d['donor']} ({d['reason']})"
                        for d in rec["decisions"])
        others = [s for s in SEEDS[1:] if _pred(root, *TF, s, hid).exists()]
        same = all(_same(p1, _pred(root, *TF, s, hid)) for s in others)
        out.append(f"- {GAP[hid]}: a = {co['a']:+.3f} ± {se['a']:.3f}, b = {co['b']:+.3f} ± "
                   f"{se['b']:.3f}, β = {co['beta']:+.3f} ± {se['beta']:.3f} /µm "
                   f"(dof {rec['model']['dof']}, R² {rec['model']['r2']:.2f}); {dec}; "
                   f"seeds 1{''.join(', ' + s for s in others)} identical: {same}")
    text = "\n".join(out) + "\n"
    (root / "tf_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
