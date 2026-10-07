"""Summarize scripts/seed_sweep_layout_interp.sh.

    python scripts/summarize_seed_sweep.py reproduced/seed_layout

Per seed and design: the fold-chosen rho, and the paired 8-metric composite
(wins − losses, as in summarize_gap_sweep.py) of the fold-calibrated run and of
the forced rho = 1 run against the same seed's rho = 0 baseline. Across seeds:
mean ± sd of the composite, how many seeds are positive, and per-metric mean
paired differences. Writes ``<root>/summary.md`` and prints it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_gap_sweep import GAP, METRICS, ORDER, SHORT, composite  # noqa: E402

METHOD = "spatialcpav18_gen_flow_layout_interp"
PRIMARY = "wide_3_4_5"


def load(root, arm):
    """{holdout_id: (metrics, rho)} for one arm of one seed."""
    out = {}
    for mf in sorted((root / arm / METHOD / "starmap_visual_cortex").glob("*/metrics.json")):
        rho = None
        with h5py.File(mf.with_name("prediction.h5"), "r") as f:
            raw = f["uns/method_params"][()]
        cv = json.loads(raw if isinstance(raw, str) else raw.decode()).get("layout_cv") or {}
        rho = cv.get("rho")
        out[mf.parent.name] = (json.loads(mf.read_text()), rho)
    return out


def main(argv):
    root = Path(argv[1])
    seeds = sorted((p for p in root.glob("seed*") if p.is_dir()),
                   key=lambda p: int(p.name[4:]))
    per = {}                     # (hid, seed) -> dict
    for sd in seeds:
        base, cv, forced = load(sd, "base"), load(sd, "cv"), load(sd, "forced")
        for hid in ORDER:
            if hid not in base:
                continue
            b = base[hid][0]
            row = {"seed": sd.name[4:]}
            if hid in cv:
                row.update(rho=cv[hid][1], comp=composite(cv[hid][0], b),
                           diff={k: cv[hid][0].get(k, np.nan) - b.get(k, np.nan)
                                 for k, _ in METRICS})
            if hid in forced:
                row["forced"] = composite(forced[hid][0], b)
            per[(hid, sd.name)] = row

    lines = ["### Interpolated layout, seed replication (paired against rho = 0 per seed)",
             "", f"Primary endpoint: {GAP.get(PRIMARY, PRIMARY)}.", "",
             "| gap | seed | rho chosen | composite (cv − base) | forced rho=1 − base |",
             "|---|---|---|---|---|"]
    for hid in ORDER:
        for sd in seeds:
            r = per.get((hid, sd.name))
            if r:
                lines.append(f"| {GAP.get(hid, hid)} | {r['seed']} | {r.get('rho')} | "
                             f"{r.get('comp', '—'):+} | {r.get('forced', '—'):+} |")
    lines += ["", "| gap | composite mean ± sd | seeds > 0 / < 0 | forced mean ± sd | "
              + " | ".join(f"Δ{s}" for s in SHORT) + " |",
              "|---|---|---|---|" + "---|" * len(SHORT)]
    for hid in ORDER:
        rows = [per[(hid, sd.name)] for sd in seeds if (hid, sd.name) in per]
        if not rows:
            continue
        c = np.array([r["comp"] for r in rows if "comp" in r], dtype=float)
        f = np.array([r["forced"] for r in rows if "forced" in r], dtype=float)
        d = {k: np.nanmean([r["diff"][k] for r in rows if "diff" in r]) for k, _ in METRICS}
        lines.append(f"| {GAP.get(hid, hid)} | {c.mean():+.1f} ± {c.std(ddof=1) if len(c) > 1 else 0:.1f} | "
                     f"{int((c > 0).sum())} / {int((c < 0).sum())} | "
                     f"{f.mean():+.1f} ± {f.std(ddof=1) if len(f) > 1 else 0:.1f} | "
                     + " | ".join(f"{d[k]:+.3f}" for k, _ in METRICS) + " |")
    text = "\n".join(lines) + "\n"
    (root / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv)
