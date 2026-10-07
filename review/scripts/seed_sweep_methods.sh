#!/usr/bin/env bash
# Run registry methods over several seeds and hold-out designs, one results root
# per seed (reproduced/<NAME>/seed<s>/). Used for the flow_cv replication and the
# combined-method sweep (REVIEW_NOTES §13).
#
#   METHODS="spatialcpav18_gen spatialcpav18_gen_flow_cv" SEEDS="1 2 3 4 5" \
#   DESIGNS="paper wide:1 wide:3" NAME=seed_methods JOBS=3 scripts/seed_sweep_methods.sh
#
# Extra wrapper arguments for every run: EXTRA="--flag value".
# SKIP_EXISTING=1 resumes an interrupted sweep (run_all --skip-existing).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
NAME="${NAME:-seed_methods}"
OUT="$ROOT/reproduced/$NAME"
JOBS="${JOBS:-1}"
read -r -a METHODS <<< "${METHODS:?set METHODS}"
read -r -a SEEDS <<< "${SEEDS:-1 2 3 4 5}"
read -r -a DESIGNS <<< "${DESIGNS:-paper wide:1 wide:3}"
read -r -a EXTRA <<< "${EXTRA:-}"
mkdir -p "$OUT"

( cd "$ROOT/benchmark" && "$PY" -m src.bench3.prepare_dataset --dataset starmap_visual_cortex )

run() {  # run <seed> <method> <design[:block]>
  local seed=$1 m=$2 spec=$3 res="$OUT/seed$1"
  local d=${spec%%:*} args=(--methods "$m" --dataset starmap_visual_cortex --design "${spec%%:*}"
                            --seed "$seed")
  [ "$spec" != "$d" ] && args+=(--holdout-block "${spec#*:}")
  [ -n "${SKIP_EXISTING:-}" ] && args+=(--skip-existing)
  [ ${#EXTRA[@]} -gt 0 ] && args+=(-- "${EXTRA[@]}")
  mkdir -p "$res"
  local log="$res/${m}_${spec/:/_}.log"
  # Runs of one seed and design share one training-input file, which run_all
  # builds on first use: serialize them on a lock so two never write it at once.
  if ( cd "$ROOT/benchmark" && flock "$res/.lock_${spec/:/_}" env BENCH_V3_RESULTS="$res" \
         "$PY" -m src.bench3.run_all "${args[@]}" ) > "$log" 2>&1; then
    echo "done  seed $seed $m $spec"
  else echo "FAIL  seed $seed $m $spec (see $log)"; fi
}

n=0
throttle() { n=$((n + 1)); if [ $((n % JOBS)) -eq 0 ]; then wait; fi; }
for m in "${METHODS[@]}"; do          # method-major: concurrent runs differ in seed
  for s in "${SEEDS[@]}"; do            # or design, so the lock rarely waits
    for d in "${DESIGNS[@]}"; do run "$s" "$m" "$d" & throttle; done
  done
done
wait
