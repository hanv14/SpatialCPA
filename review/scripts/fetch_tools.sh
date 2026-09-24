#!/usr/bin/env bash
# Fetch the comparator method code that is not pip-installable into the paths
# the v2 wrappers expect (run_spatialz.py / run_isost.py: parents[4]/benchmark-pbya/tools).
#
#   benchmark-pbya/tools/spatialz/SpatialZ_code/   <- Zenodo 10.5281/zenodo.17416727
#   benchmark-pbya/tools/isost/                    <- github.com/deng-ai-lab/isoST @ ISOST_COMMIT
#
# FEAST needs nothing here (pip: FEAST-py, paste2 — see envs/bench_feast.yml).
# SpatialCPA-v18 needs nothing here (learn_spatialcpav18.py is committed at the repo root).
set -euo pipefail
cd "$(dirname "$0")/.."
TOOLS=benchmark-pbya/tools

# isoST: HEAD of deng-ai-lab/isoST as of 2026-09-24. REPLACE with the commit the
# published isost rows ran on (`git -C benchmark-pbya/tools/isost rev-parse HEAD`
# on the lab machine) — see REVIEW_NOTES.md, "Open provenance items".
ISOST_COMMIT="${ISOST_COMMIT:-805981c3fad66e3c9f5409816f95c17332906600}"

# SpatialZ: the Zenodo record's code archive. Its checksum could not be recorded
# from the environment this repo was built in (Zenodo unreachable); set
# SPATIALZ_MD5 to the lab machine's value to have it verified.
SPATIALZ_RECORD=17416727
SPATIALZ_MD5="${SPATIALZ_MD5:-}"

mkdir -p "$TOOLS"

if [ -d "$TOOLS/isost/.git" ]; then
  echo "isoST: present at $(git -C "$TOOLS/isost" rev-parse HEAD)"
else
  git clone https://github.com/deng-ai-lab/isoST "$TOOLS/isost"
fi
git -C "$TOOLS/isost" fetch --quiet origin "$ISOST_COMMIT" 2>/dev/null || true
git -C "$TOOLS/isost" checkout --quiet "$ISOST_COMMIT"
echo "isoST: checked out $ISOST_COMMIT"

if [ -f "$TOOLS/spatialz/SpatialZ_code/SpatialZ.py" ]; then
  echo "SpatialZ: present"
else
  mkdir -p "$TOOLS/spatialz/_download"
  api="https://zenodo.org/api/records/$SPATIALZ_RECORD"
  echo "SpatialZ: listing $api"
  python3 - "$api" "$TOOLS/spatialz/_download" <<'PY'
import json, sys, urllib.request, pathlib
api, out = sys.argv[1], pathlib.Path(sys.argv[2])
rec = json.load(urllib.request.urlopen(api))
for f in rec["files"]:
    dst = out / f["key"]
    print(f"  {f['key']}  {f['size']/1e6:.1f} MB  {f['checksum']}")
    urllib.request.urlretrieve(f["links"]["self"], dst)
PY
  if [ -n "$SPATIALZ_MD5" ]; then
    echo "$SPATIALZ_MD5  $(ls "$TOOLS"/spatialz/_download/*.zip | head -1)" | md5sum -c -
  fi
  unzip -q -o "$TOOLS"/spatialz/_download/*.zip -d "$TOOLS/spatialz/_unz"
  src=$(dirname "$(find "$TOOLS/spatialz/_unz" -name SpatialZ.py | head -1)")
  [ -n "$src" ] || { echo "SpatialZ.py not found in the Zenodo archive" >&2; exit 1; }
  mv "$src" "$TOOLS/spatialz/SpatialZ_code"
  rm -rf "$TOOLS/spatialz/_unz"
fi

# MANIFEST.tools.sha256 records the SpatialZ.py / Synthesize.py kept in the
# parent project's reference/ directory. Whether those are byte-identical to the
# Zenodo archive has NOT been verified (Zenodo was unreachable when this repo was
# built), so a mismatch is a warning, not a failure — and a reason to ask.
( cd "$TOOLS/spatialz" && sha256sum -c ../../../MANIFEST.tools.sha256 ) \
  || echo "WARNING: SpatialZ code differs from MANIFEST.tools.sha256 (see above)" >&2
