#!/usr/bin/env bash
# STARmap gap sweep for the transport variants (REVIEW_NOTES §7c) or, with
# SWEEP=h, the h-space grounding arms (§9), on the
# harness's existing hold-out designs — no change to prepare_dataset:
#
#   paper          hold out 2,4,6    every target 11 um from its nearest section
#   wide block 1   hold out 4        11 um (bracketing sections adjacent)
#   wide block 3   hold out 3,4,5    middle target 22 um from the nearest section
#   wide block 5   hold out 2..6     middle target 33 um (only 1 and 7 remain:
#                                    no interior training section, so no folds)
#
# Every method row is fold-calibrated (strength 0, plain copying, wins ties).
# A second pass forces the strength to 1 for each arm into
# reproduced/gap_sweep[_h]_forced/<arm>/ — a diagnostic, not a method. For
# SWEEP=h, every row is compared with h-cv at gamma 0 (reproduced/gap_sweep_h_base/),
# which draws the same random numbers as every arm; the registry
# "nearest + no flow" row is shown too, and its gap to that baseline is the
# run-to-run noise floor.
#
#   scripts/gap_sweep_starmap.sh           # PYTHON / BENCH_V3_PYTHON as in reproduce_starmap_v18.sh
#   JOBS=2 scripts/gap_sweep_starmap.sh    # two runs at a time
#   SWEEP=h scripts/gap_sweep_starmap.sh   # the h-cv arms -> reproduced/gap_sweep_h/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
SWEEP="${SWEEP:-transport}"          # transport (REVIEW_NOTES §7c) | h (§9)
OUT="$ROOT/reproduced/gap_sweep${SWEEP/transport/}"
OUT="${OUT/%gap_sweeph/gap_sweep_h}"
FORCED="${OUT}_forced"
JOBS="${JOBS:-1}"
if [ "$SWEEP" = h ]; then
  METHODS=(spatialcpav18_gen_nearest_noflow spatialcpav18_gen_flow_h
           spatialcpav18_gen_flow_h_untrained spatialcpav18_gen_flow_h_interp
           spatialcpav18_gen_flow_h_srcdepth)
  FORCED_METHOD=spatialcpav18_gen_flow_h
  FORCED_ARMS=(flow untrained interp srcdepth)
  forced_args() { echo --h-source "$1" --h-gamma 1; }
  BASE="${OUT}_base"   # paired baseline: h-cv at gamma 0 (same random streams as every arm)
else
  METHODS=(spatialcpav18_gen_nearest_noflow spatialcpav18_gen_flow_transport
           spatialcpav18_gen_flow_transport_ot spatialcpav18_gen_flow_transport_zshuffle
           spatialcpav18_gen_flow_transport_pair)
  FORCED_METHOD=spatialcpav18_gen_flow_transport
  FORCED_ARMS=(flow ot)
  forced_args() { echo --transport "$1" --transport-lambda 1; }
  BASE=""
fi
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
for arm in "${FORCED_ARMS[@]}"; do
  for d in "${DESIGNS[@]}"; do
    # shellcheck disable=SC2046
    run "$FORCED/$arm" "$FORCED_METHOD" "$d" $(forced_args "$arm") &
    throttle
  done
done
if [ -n "$BASE" ]; then
  for d in "${DESIGNS[@]}"; do
    run "$BASE" "$FORCED_METHOD" "$d" --h-source flow --h-gamma 0 & throttle
  done
fi
wait
"$PY" "$ROOT/scripts/summarize_gap_sweep.py" "$OUT" "$FORCED" $BASE
