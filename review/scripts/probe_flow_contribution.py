#!/usr/bin/env python3
"""How much does v18's trained flow change what is emitted?

Under the published flags (``--edit-weight 0.0``) no decoded expression reaches
the output: every emitted value is a real training measurement. The flow acts
only through *which* real cell each generated cell copies. The layout pass
assigns each cell an inherited source; then

  1. ``_ground``          re-grounds a cell to the flow latent's preferred local
                          candidate (``--ground-blend-flow 1.0`` = every cell is
                          considered, but ``--ground-keep-margin 1.0`` keeps the
                          inherited source unless the flow prefers another by ≥ 1.0);
  2. ``_vote_types``      re-grounds cells whose kNN-voted type disagrees;
  3. ``_match_composition`` re-grounds to match the interpolated type composition;
  4. ``_gene_mix``        swaps ~15 % of each cell's genes for a same-type partner's.

This runs the unmodified wrapper in-process with those four methods wrapped (not
changed) to count, per target section, how many cells each stage re-picked. It
prints a table and writes nothing but the prediction you point it at.

    python scripts/probe_flow_contribution.py --input <train_registered.h5ad> \\
        --target-section section_2 section_4 section_6 --target-z 30 52 74 \\
        --output /tmp/probe.h5

Extra arguments go to the wrapper; the pinned V18_ARGS are applied first, exactly
as run_benchmark does. Needs the bench_spatialcpa environment.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "benchmark-pbya-v3" / "src" / "bench3" / "methods" / "run_spatialcpav18.py"
sys.path.insert(0, str(ROOT / "benchmark-pbya-v3"))
from src.bench3.config import V18_ARGS  # noqa: E402

spec = importlib.util.spec_from_file_location("run_spatialcpav18", WRAPPER)
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)
M = W._V18.SpatialCPAv14

stats: list[dict] = []
_orig = {n: getattr(M, n) for n in ("_ground", "_vote_types", "_match_composition", "_gene_mix")}


def _ground(self, anchor, anchor_src, *a, **k):
    out = _orig["_ground"](self, anchor, anchor_src, *a, **k)
    stats.append({"n": len(anchor_src), "src": anchor_src.copy(), "after_ground": out[2].copy()})
    return out


def _vote_types(self, anchor, pick, *a, **k):
    before = pick.copy()
    out = _orig["_vote_types"](self, anchor, pick, *a, **k)
    stats[-1]["vote_changed"] = int(np.sum(out[1] != before))
    return out


def _match_composition(self, *a, **k):
    before = a[-1].copy()                      # ``pick`` is the last positional arg
    out = _orig["_match_composition"](self, *a, **k)
    stats[-1]["comp_changed"] = int(np.sum(out[2] != before))
    stats[-1]["final_pick"] = out[2].copy()
    return out


def _gene_mix(self, *a, **k):
    partner, gmask = _orig["_gene_mix"](self, *a, **k)
    stats[-1]["gene_mix_frac"] = float(np.mean(gmask))
    return partner, gmask


M._ground, M._vote_types = _ground, _vote_types
M._match_composition, M._gene_mix = _match_composition, _gene_mix

sys.argv = [str(WRAPPER), *sys.argv[1:], *V18_ARGS] if "--edit-weight" not in sys.argv \
    else [str(WRAPPER), *sys.argv[1:]]
rc = W.main()

print("\nper target section (cells):")
print(f"  {'n':>6s} {'flow re-grounded':>18s} {'type vote':>10s} {'composition':>12s} "
      f"{'final != inherited':>19s} {'genes mixed':>12s}")
for s in stats:
    n = s["n"]
    flow = int(np.sum(s["after_ground"] != s["src"]))
    final = s.get("final_pick", s["after_ground"])
    print(f"  {n:6d} {flow:8d} ({flow / n:6.1%}) {s.get('vote_changed', 0):10d} "
          f"{s.get('comp_changed', 0):12d} {int(np.sum(final != s['src'])):9d} "
          f"({np.mean(final != s['src']):6.1%}) {s.get('gene_mix_frac', 0.0):12.3f}")
sys.exit(rc)
