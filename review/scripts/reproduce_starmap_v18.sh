#!/usr/bin/env bash
# Reproduce the SpatialCPA-v18 STARmap paper row (paper_2_4_6) from scratch and
# check it against the committed row. `make starmap-row` runs this.
#
#   scripts/reproduce_starmap_v18.sh                 # compare against expected/published/ if
#                                                    # present, else expected/cpu-verified/
#   scripts/reproduce_starmap_v18.sh cpu-verified    # force one of the two
#   scripts/reproduce_starmap_v18.sh published
#
# Environment:
#   PYTHON           harness interpreter (bench_eval env). Default: python
#   BENCH_V3_PYTHON  if set, run the v18 wrapper with this interpreter instead of
#                    `conda run -n bench_spatialcpa` (one-venv CPU lock; see README)
#
# Writes only under reproduced/ (git-ignored). Exit 0 = reproduced.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
BENCH="$ROOT/benchmark"
OUT="$ROOT/reproduced"
ROW="spatialcpav18_gen/starmap_visual_cortex/paper_2_4_6"

against="${1:-auto}"
if [ "$against" = auto ]; then
  if [ -f "$ROOT/expected/published/$ROW/metrics.json" ]; then against=published; else against=cpu-verified; fi
fi
EXP="$ROOT/expected/$against/$ROW"
[ -f "$EXP/metrics.json" ] || { echo "no committed row at $EXP/metrics.json" >&2; exit 2; }

step() { printf '\n== [%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
t0=$(date +%s)

step "1/6  pinned files (MANIFEST.sha256, evaluate_paper.py)"
( cd "$ROOT" && "$PY" -m pytest -q tests/test_pins.py )

step "2/6  build the paper-protocol STARmap dataset from benchmark/data/raw/ (~5 s)"
( cd "$BENCH" && "$PY" -m src.bench3.prepare_dataset --dataset starmap_visual_cortex )

step "3/6  run spatialcpav18_gen under the pinned flags, then score (~3 min on 4 CPU cores)"
rm -rf "$OUT/spatialcpav18_gen/starmap_visual_cortex" "$OUT/_inputs/starmap_visual_cortex"
mkdir -p "$OUT"
( cd "$BENCH" && env -u SPATIALCPAV18_FILE -u SPATIALCPAV18_ROOT BENCH_V3_RESULTS="$OUT" \
    "$PY" -m src.bench3.run_all --methods spatialcpav18_gen --dataset starmap_visual_cortex ) \
  2>&1 | tee "$OUT/run_all.log"
[ -f "$OUT/$ROW/metrics.json" ] || { echo "FAIL: no metrics.json — see $OUT/$ROW/method_log.txt" >&2; exit 1; }

step "4/6  the run loaded the committed learn_spatialcpav18.py and nothing else"
want=$(awk '$2=="learn_spatialcpav18.py"{print $1}' "$ROOT/MANIFEST.sha256")
grep -qF "from $ROOT/learn_spatialcpav18.py;" "$OUT/$ROW/method_log.txt" \
  || { echo "FAIL: method log does not show $ROOT/learn_spatialcpav18.py" >&2; exit 1; }
grep -qF "learn_spatialcpav18.py sha256 $want" "$OUT/$ROW/method_log.txt" \
  || { echo "FAIL: loaded v18 sha256 is not $want" >&2; exit 1; }
flags="--edit-weight 0.0 --ground-blend-flow 1.0 --ground-k 8 --ground-temp 0.25 --ground-keep-margin 1.0 --type-mode vote --type-vote-k 12 --gene-mix-frac 0.15"
grep -F "  cmd: " "$OUT/run_all.log" | grep -qF -- "--seed 42 $flags" \
  || { echo "FAIL: the logged command does not carry the pinned v18 flags: $flags" >&2; exit 1; }
echo "ok: $ROOT/learn_spatialcpav18.py ($want)"; echo "ok: --seed 42 $flags"

rc=0
step "5/6  prediction vs expected/$against"
if [ -f "$EXP/prediction.h5" ]; then
  "$PY" "$ROOT/scripts/compare_predictions.py" "$EXP/prediction.h5" "$OUT/$ROW/prediction.h5" || rc=1
else
  echo "(no committed prediction.h5 for $against — metrics only)"
fi

step "6/6  metrics vs expected/$against/$ROW/metrics.json"
"$PY" "$ROOT/scripts/compare_metrics.py" "$EXP/metrics.json" "$OUT/$ROW/metrics.json" || rc=1

printf '\n%s in %ss — compared against expected/%s\n' \
  "$([ $rc = 0 ] && echo REPRODUCED || echo 'NOT REPRODUCED')" "$(( $(date +%s) - t0 ))" "$against"
if [ "$against" = cpu-verified ]; then
  echo "NOTE: expected/cpu-verified is this repo's own CPU run, not the published row."
  echo "      See README.md, 'What \"reproduced\" means here'."
fi
exit $rc
