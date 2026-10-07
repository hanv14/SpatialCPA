#!/usr/bin/env bash
# Seed replication of the interpolated-layout lead (REVIEW_NOTES §11 → §12).
#
# Seed 42 found it (+4 at the 22 µm block-3 design), so the confirmation runs on
# fresh seeds only: SEEDS (default 1..5). Per seed and design, three runs that
# share their random streams:
#   cv      spatialcpav18_gen_flow_layout_interp   (rho chosen by training folds)
#   base    the same method at rho 0                (the paired baseline)
#   forced  the same method at rho 1                (diagnostic)
# Pre-registered primary endpoint: wide block 3 (22 µm), the paired 8-metric
# composite cv − base. Summarized by scripts/summarize_seed_sweep.py.
#
#   JOBS=4 scripts/seed_sweep_layout_interp.sh        # PYTHON / BENCH_V3_PYTHON as usual
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
OUT="$ROOT/reproduced/seed_layout"
JOBS="${JOBS:-1}"
SEEDS=(${SEEDS:-1 2 3 4 5})
DESIGNS=("paper" "wide:1" "wide:3" "wide:5")
M=spatialcpav18_gen_flow_layout_interp
mkdir -p "$OUT"

( cd "$ROOT/benchmark" && "$PY" -m src.bench3.prepare_dataset --dataset starmap_visual_cortex )

run() {  # run <results root> <seed> <design[:block]> [wrapper extra args...]
  local res=$1 seed=$2 spec=$3; shift 3
  local d=${spec%%:*} args=(--methods "$M" --dataset starmap_visual_cortex --design "${spec%%:*}"
                            --seed "$seed")
  [ "$spec" != "$d" ] && args+=(--holdout-block "${spec#*:}")
  [ $# -gt 0 ] && args+=(-- "$@")
  mkdir -p "$res"
  local log="$res/${spec/:/_}.log"
  if ( cd "$ROOT/benchmark" && BENCH_V3_RESULTS="$res" "$PY" -m src.bench3.run_all "${args[@]}" ) \
       > "$log" 2>&1; then echo "done  seed $seed $spec $(basename "$res")"
  else echo "FAIL  seed $seed $spec $(basename "$res") (see $log)"; fi
}

n=0
throttle() { n=$((n + 1)); if [ $((n % JOBS)) -eq 0 ]; then wait; fi; }
for s in "${SEEDS[@]}"; do
  for d in "${DESIGNS[@]}"; do
    run "$OUT/seed$s/cv" "$s" "$d" & throttle
    run "$OUT/seed$s/base" "$s" "$d" --layout-rho 0 & throttle
    run "$OUT/seed$s/forced" "$s" "$d" --layout-rho 1 & throttle
  done
done
wait
"$PY" "$ROOT/scripts/summarize_seed_sweep.py" "$OUT"
