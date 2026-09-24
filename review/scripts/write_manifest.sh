#!/usr/bin/env bash
# Rewrite MANIFEST.sha256: every tracked source file that can move a number, the
# STARmap source volume, and the committed expected rows. Run after an
# intentional change, review the diff, commit it.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# sha256 of every file that can move a published number. Verified by"
  echo "# tests/test_pins.py (make verify) and by step 1 of make starmap-row."
  echo "# Regenerate with scripts/write_manifest.sh after an intentional change."
  find learn_spatialcpav18.py benchmark/src \
       -name '*.py' -not -path '*/__pycache__/*' | LC_ALL=C sort
  echo benchmark/data/raw/starmap_visual_cortex/STARmap_Wang2018three_data_3D_data.h5ad
  find expected -type f \( -name '*.json' -o -name '*.h5' -o -name '*.txt' \) 2>/dev/null | LC_ALL=C sort
} | while read -r f; do
  case "$f" in \#*) echo "$f" ;; *) sha256sum "$f" ;; esac
done > MANIFEST.sha256
echo "wrote MANIFEST.sha256 ($(grep -vc '^#' MANIFEST.sha256) files)"
