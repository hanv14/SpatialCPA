#!/usr/bin/env bash
# Run ONCE on the machine that produced the published rows. Writes the exact
# environments into envs/lock/, and the provenance hashes the README asks for into
# envs/lock/provenance.txt. Commit the result; `make verify` then prefers them.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p envs/lock
for env in bench_spatialcpa bench_spatialz bench_feast bench_isost bench_eval; do
  if conda env list | awk '{print $1}' | grep -qx "$env"; then
    conda env export -n "$env" --no-builds > "envs/lock/$env.lab.yml"
    conda run -n "$env" python -m pip freeze --all > "envs/lock/$env.lab.pip.txt"
    conda run -n "$env" python -c 'import sys; print(sys.version)' > "envs/lock/$env.lab.python.txt"
    echo "exported $env"
  else
    echo "not present on this machine: $env" >&2
  fi
done
{
  echo "# host: $(hostname)  date: $(date -u +%FT%TZ)"
  echo "# The files the published rows actually ran. Paths are the lab layout;"
  echo "# override with V18=... V3=... TOOLS=... if yours differ (V3 = the lab's benchmark-pbya-v3)."
  V18=${V18:-/data/han/projects/Spatial3D/src/learn_spatialcpav18.py}
  V3=${V3:-$HOME/benchmark-pbya-v3}
  TOOLS=${TOOLS:-$V3/../benchmark-pbya/tools}
  sha256sum "$V18" "$V3/src/bench3/evaluate_paper.py" "$V3/src/bench3/align.py" \
            "$V3/src/bench3/methods/run_spatialcpav18.py" 2>&1 || true
  sha256sum "$TOOLS"/spatialz/SpatialZ_code/*.py 2>&1 || true
  echo "isost_commit $(git -C "$TOOLS/isost" rev-parse HEAD 2>&1 || true)"
} > envs/lock/provenance.txt
echo "wrote envs/lock/provenance.txt — compare it with MANIFEST.sha256"
