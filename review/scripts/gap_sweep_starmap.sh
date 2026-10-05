#!/usr/bin/env bash
# STARmap gap sweep for the transport variants (REVIEW_NOTES §7c), on the
# harness's existing hold-out designs — no change to prepare_dataset:
#
#   paper          hold out 2,4,6    every target 11 um from its nearest section
#   wide block 1   hold out 4        11 um (bracketing sections adjacent)
#   wide block 3   hold out 3,4,5    middle target 22 um from the nearest section
#   wide block 5   hold out 2..6     middle target 33 um (only 1 and 7 remain:
#                                    no interior training section, so no folds)
#
# Every method row is fold-calibrated (lambda = 0, plain copying, wins ties).
# A second pass forces lambda = 1 for the flow and the OT control into
# reproduced/gap_sweep_forced/<transport>/ — raw transport quality per gap, a
# diagnostic, not a method.
#
#   scripts/gap_sweep_starmap.sh           # PYTHON / BENCH_V3_PYTHON as in reproduce_starmap_v18.sh
#   JOBS=2 scripts/gap_sweep_starmap.sh    # two runs at a time
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/reproduced/gap_sweep"
FORCED="$ROOT/reproduced/gap_sweep_forced"
JOBS="${JOBS:-1}"
METHODS=(spatialcpav18_gen_nearest_noflow spatialcpav18_gen_flow_transport
         spatialcpav18_gen_flow_transport_ot spatialcpav18_gen_flow_transport_zshuffle
         spatialcpav18_gen_flow_transport_pair)
DESIGNS=("paper" "wide:1" "wide:3" "wide:5")
mkdir -p "$OUT"

( cd "$ROOT/benchmark" && "$PY" -m src.bench3.prepare_dataset --dataset starmap_visual_cortex )

run() {  # run <results root> <method> <design[:block]> [wrapper extra args...]
  local res=$1 m=$2 spec=$3; shift 3
  local d=${spec%%:*} args=(--methods "$m" --dataset starmap_visual_cortex --design "${spec%%:*}")
  [ "$spec" != "$d" ] && args+=(--holdout-block "${spec#*:}")
  [ $# -gt 0 ] && args+=(-- "$@")
  local log="$res/${m}_${spec/:/_}.log"
  mkdir -p "$res"
  if ( cd "$ROOT/benchmark" && BENCH_V3_RESULTS="$res" "$PY" -m src.bench3.run_all "${args[@]}" ) \
       > "$log" 2>&1; then echo "done  $m $spec"; else echo "FAIL  $m $spec (see $log)"; fi
}

n=0
throttle() { n=$((n + 1)); if [ $((n % JOBS)) -eq 0 ]; then wait; fi; }

for m in "${METHODS[@]}"; do
  for d in "${DESIGNS[@]}"; do run "$OUT" "$m" "$d" & throttle; done
done
for tr in flow ot; do
  for d in "${DESIGNS[@]}"; do
    run "$FORCED/$tr" spatialcpav18_gen_flow_transport "$d" --transport "$tr" --transport-lambda 1 &
    throttle
  done
done
wait
"$PY" "$ROOT/scripts/summarize_gap_sweep.py" "$OUT" "$FORCED"
