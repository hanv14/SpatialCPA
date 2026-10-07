"""Summarize the flow_cv replication and the combo sweep (REVIEW_NOTES §13).

    python scripts/summarize_combo.py reproduced

Reads reproduced/seed_methods (registry methods), seed_combo (fold-calibrated
combo arms), seed_combo_base (combo at w=1, beta=0: the paired baseline) and
seed_combo_corner (combo at w=0.5, beta=1). For each comparison, per seed and
design: the signal-to-noise composite (the fold selection score, pooled
held-out metrics) and wins − losses on the 8 metrics; then mean ± sd across seeds
and the count of seeds above zero. Writes ``<root>/combo_summary.md``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

KEYS = (("embedding_mixing_pca", +1), ("morans_pearson", +1), ("morans_mae", -1),
        ("marker_depth_r", +1), ("marker_field_r", +1), ("celltype_localization", +1),
        ("rare_celltype_localization", +1), ("gene_detection_spearman", +1))
SIGMA = {  # = run_spatialcpav18_combo.SIGMA
    "embedding_mixing_pca": 0.001616, "morans_pearson": 0.001116, "morans_mae": 0.0009241,
    "marker_depth_r": 0.004012, "marker_field_r": 0.001512, "celltype_localization": 0.003078,
    "rare_celltype_localization": 0.006684, "gene_detection_spearman": 0.0488}
HIDS = ("paper_2_4_6", "wide_4", "wide_3_4_5")
GAP = {"paper_2_4_6": "11 µm paper", "wide_4": "11 µm block 1", "wide_3_4_5": "22 µm block 3"}
SEEDS = ("1", "2", "3", "4", "5")


def metrics(root, sweep, seed, method, hid):
    f = root / sweep / f"seed{seed}" / method / "starmap_visual_cortex" / hid / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def choice(root, sweep, seed, method, hid):
    f = root / sweep / f"seed{seed}" / method / "starmap_visual_cortex" / hid / "prediction.h5"
    if not f.exists():
        return None
    with h5py.File(f, "r") as h:
        raw = h["uns/method_params"][()]
    c = json.loads(raw if isinstance(raw, str) else raw.decode()).get("combo") or {}
    return c.get("w"), c.get("beta")


def snr(a, b):
    return sum(s * (a["paper_" + k] - b["paper_" + k]) / SIGMA[k] for k, s in KEYS
               if a.get("paper_" + k) is not None and b.get("paper_" + k) is not None)


def wl(a, b):
    g = 0
    for k, s in KEYS:
        x, y = a.get("paper_" + k), b.get("paper_" + k)
        if x is None or y is None:
            continue
        d = s * (x - y)
        g += int(d > 1e-9) - int(d < -1e-9)
    return g


def compare(root, label, A, B, with_choice=None):
    """A, B: (sweep, method). Rows per design: per-seed (snr, wl), mean ± sd."""
    lines = [f"### {label}", "", "| design | per-seed SNR composite | mean ± sd | seeds > 0 | "
             "per-seed wins−losses | (w, β) chosen |", "|---|---|---|---|---|---|"]
    for hid in HIDS:
        s_, w_, ch = [], [], []
        for sd in SEEDS:
            a, b = metrics(root, *A, sd, hid), metrics(root, *B, sd, hid)
            if a is None or b is None:
                continue
            s_.append(snr(a, b))
            w_.append(wl(a, b))
            if with_choice:
                ch.append(choice(root, *with_choice, sd, hid))
        if not s_:
            continue
        s = np.array(s_)
        sdv = s.std(ddof=1) if len(s) > 1 else 0.0
        lines.append(f"| {GAP[hid]} | {', '.join(f'{x:+.1f}' for x in s)} | {s.mean():+.1f} ± "
                     f"{sdv:.1f} | {int((s > 0).sum())}/{len(s)} | "
                     f"{', '.join(f'{x:+d}' for x in w_)} | "
                     f"{', '.join(str(c) for c in ch) if ch else '—'} |")
    return lines + [""]


def main(argv):
    root = Path(argv[1])
    near = ("seed_methods", "spatialcpav18_gen_nearest_noflow")
    out = ["## flow_cv replication and the combo sweep", ""]
    out += compare(root, "flow_cv vs nearest + no flow (not paired: same seed)",
                   ("seed_methods", "spatialcpav18_gen_flow_cv"), near)
    out += compare(root, "published v18 vs nearest + no flow (same seed)",
                   ("seed_methods", "spatialcpav18_gen"), near)
    base = ("seed_combo_base", "spatialcpav18_gen_flow_combo")
    for arm, m in (("combo (flow)", "spatialcpav18_gen_flow_combo"),
                   ("combo untrained (control)", "spatialcpav18_gen_flow_combo_untrained"),
                   ("combo noflow (control)", "spatialcpav18_gen_flow_combo_noflow")):
        out += compare(root, f"{arm} vs combo at (1, 0) — paired", ("seed_combo", m), base,
                       with_choice=("seed_combo", m))
    out += compare(root, "corner (0.5, 1) vs combo at (1, 0) — paired",
                   ("seed_combo_corner", "spatialcpav18_gen_flow_combo"), base)
    out += compare(root, "combo at (1, 0) vs registry flow_cv (same seed; the stream change)",
                   base, ("seed_methods", "spatialcpav18_gen_flow_cv"))
    text = "\n".join(out)
    (root / "combo_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
