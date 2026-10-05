"""Summarize the STARmap gap sweep (scripts/gap_sweep_starmap.sh).

    python scripts/summarize_gap_sweep.py reproduced/gap_sweep reproduced/gap_sweep_forced

For each hold-out design and method: the fold-chosen lambda, the pooled paper
metrics, and the composite against "nearest + no flow" on the same design —
wins minus losses over the 8 metrics flow-cv calibrates on (here with UMAP
mixing in place of the PCA mixing used on the folds). The forced lambda = 1 rows
are diagnostics. Writes ``<sweep>/summary.md`` and prints it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

BASE = "spatialcpav18_gen_nearest_noflow"
METRICS = (  # (paper_* key, +1 higher is better / -1 lower)
    ("paper_umap_mixing", +1), ("paper_morans_pearson", +1), ("paper_morans_mae", -1),
    ("paper_marker_depth_r", +1), ("paper_marker_field_r", +1),
    ("paper_celltype_localization", +1), ("paper_rare_celltype_localization", +1),
    ("paper_gene_detection_spearman", +1),
)
SHORT = ("UMAP mix", "Moran r", "Moran MAE", "depth r", "field r", "loc", "rare loc", "detect")
TIE = 1e-9
ORDER = ("paper_2_4_6", "wide_4", "wide_3_4_5", "wide_2_3_4_5_6")
GAP = {"paper_2_4_6": "11 µm (alternate)", "wide_4": "11 µm (block 1)",
       "wide_3_4_5": "22 µm (block 3)", "wide_2_3_4_5_6": "33 µm (block 5)"}


def rows(root: Path):
    """{(holdout_id, method): (metrics, lambda or None)}"""
    out = {}
    for mf in sorted(root.glob("*/starmap_visual_cortex/*/metrics.json")):
        method, _, hid = mf.parts[-4:-1]
        lam = None
        pred = mf.parent / "prediction.h5"
        if pred.exists():
            with h5py.File(pred, "r") as f:
                raw = f["uns/method_params"][()]
            mp = json.loads(raw if isinstance(raw, str) else raw.decode())
            cv = mp.get("transport_cv")
            lam = None if cv is None else cv.get("lambda")
        out[(hid, method)] = (json.loads(mf.read_text()), lam)
    return out


def composite(m, base):
    g = 0
    for key, sign in METRICS:
        a, b = m.get(key), base.get(key)
        if a is None or b is None:
            continue
        d = sign * (a - b)
        g += int(d > TIE) - int(d < -TIE)
    return g


def table(found, label):
    lines = [f"### {label}", "",
             "| gap | method | λ | " + " | ".join(SHORT) + " | vs nearest |",
             "|---|---|---|" + "---|" * len(SHORT) + "---|"]
    hids = [h for h in ORDER if any(k[0] == h for k in found)]
    for hid in hids:
        base = next((m for (h, meth), (m, _) in found.items() if h == hid and meth == BASE), None)
        for (h, meth), (m, lam) in sorted(found.items()):
            if h != hid:
                continue
            vals = " | ".join("—" if m.get(k) is None else f"{m[k]:.3f}" for k, _ in METRICS)
            comp = "" if base is None or meth == BASE else f"{composite(m, base):+d}"
            name = meth.replace("spatialcpav18_gen_", "")
            lines.append(f"| {GAP.get(hid, hid)} | {name} | "
                         f"{'—' if lam is None else lam} | {vals} | {comp} |")
    return lines


def main(argv):
    sweep = Path(argv[1])
    found = rows(sweep)
    out = table(found, "Fold-calibrated (λ chosen on training folds; λ = 0 is plain copying)")
    if len(argv) > 2:
        base = {k: v for k, v in found.items() if k[1] == BASE}
        for tr_dir in sorted(Path(argv[2]).glob("*")):
            forced = {(h, f"forced λ=1 {tr_dir.name}"): v for (h, _), v in rows(tr_dir).items()}
            if forced:
                out += [""] + table({**base, **forced},
                                    f"Diagnostic: λ forced to 1 ({tr_dir.name})")
    text = "\n".join(out) + "\n"
    (sweep / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
