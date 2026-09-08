#!/usr/bin/env bash
# Pre-campaign screen — reports/v25_campaign_costing.md §6.
#
# NO GIT. NO NETWORK. Steps 0-1 have no Python dependencies at all.
# Steps 2-3 involve no fits. Step 4 is one short probe fit and is printed, not run.
#
#   bash campaign_screen.sh 2>&1 | tee screen.log
#   tar czf screen.tgz screen.log reports/screen/
#
# Paths follow bench3's own conventions (config.py: REPO_ROOT = bench3's parent), each
# overridable. Nothing falls back silently — a missing path names itself and its flag.
#
#   SPATIALCPA_BENCH3   benchmark-pbya-v3           (default: sibling of this repo)
#   BENCH_V3_DATA       built datasets              (default: $BENCH3/data/processed)
#   BENCH_V3_RESULTS    results + _inputs           (default: $BENCH3/results)
#   V1_ROOT             benchmark-pbya  (v1)        (default: $BENCH3/../benchmark-pbya)
#   V2_ROOT             benchmark-pbya-v2           (default: $BENCH3/../benchmark-pbya-v2)
#   REPO                the v25 checkout            (default: this script's parent)

set -uo pipefail   # deliberately not -e: one dataset failing must not abort the screen

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# bench3 lives inside the repo in some checkouts and beside it in others, so try the
# known layouts rather than guessing one. $SPATIALCPA_BENCH3 wins outright.
BENCH3="${SPATIALCPA_BENCH3:-}"
if [ -z "$BENCH3" ]; then
  for c in "$REPO" "$(dirname "$REPO")" "$(dirname "$(dirname "$REPO")")"; do
    [ -d "$c/benchmark-pbya-v3/src/bench3" ] && { BENCH3="$c/benchmark-pbya-v3"; break; }
  done
  [ -n "$BENCH3" ] || { BENCH3="$REPO/benchmark-pbya-v3"
    echo "WARNING: no benchmark-pbya-v3 found near $REPO — set SPATIALCPA_BENCH3." >&2; }
fi
B3PARENT="$(dirname "$BENCH3")"
V1_ROOT="${V1_ROOT:-$B3PARENT/benchmark-pbya}"
V2_ROOT="${V2_ROOT:-$B3PARENT/benchmark-pbya-v2}"
B3DATA="${BENCH_V3_DATA:-$BENCH3/data/processed}"
B3RES="${BENCH_V3_RESULTS:-$BENCH3/results}"
PY="${PY:-python}"
OUT="$REPO/reports/screen"; mkdir -p "$OUT"

# dataset : holdout — the five §1c survivors whose headroom is unmeasured.
DATASETS="cosmx_nsclc_3d:paper_2_4
merfish_thick_hypothalamus:paper_2_4_6
merfish_thick_cortex:paper_2_4_6
exseq_breast_cancer:paper_2_4_6
exseq_visual_cortex:paper_2_4_6"

ok () { [ -e "$1" ] && echo yes || echo "**NO**"; }

# =====================================================================================
# STEP 0 — preflight. Pure shell, seconds, no Python. Answers §1c's open prerequisite:
# which datasets are actually BUILT. The ceiling scripts need three things per dataset
# (scripts/_bench3_paths.py::resolve), and a bare "SKIP" does not say which is missing.
# =====================================================================================
echo "=============== STEP 0  preflight ==============="
{
  echo "# Step 0 — preflight"; echo; echo "Generated $(date -Is) on $(hostname)"; echo
  echo '## Trees'; echo
  echo '| what | path | present |'; echo '|---|---|---|'
  echo "| v25 repo | \`$REPO\` | $(ok "$REPO") |"
  echo "| bench3 | \`$BENCH3\` | $(ok "$BENCH3") |"
  echo "| bench3 package | \`$BENCH3/src/bench3\` | $(ok "$BENCH3/src/bench3") |"
  echo "| v2 wrappers (\`_v2_io.py\`) | \`$V2_ROOT/src/benchmark/methods/_v2_io.py\` | $(ok "$V2_ROOT/src/benchmark/methods/_v2_io.py") |"
  echo "| v1 processed tree | \`$V1_ROOT/data/processed\` | $(ok "$V1_ROOT/data/processed") |"
  echo "| built datasets | \`$B3DATA\` | $(ok "$B3DATA") |"
  echo "| results / _inputs | \`$B3RES\` | $(ok "$B3RES") |"
  echo
  echo '## Environment'; echo
  echo '```'
  echo "python   $("$PY" -V 2>&1)"
  echo "torch    $("$PY" -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"
  # from /tmp, so the repo on cwd cannot stand in for an actual install in the env
  echo "v25 pkg  $(cd /tmp && "$PY" -c 'import spatialcpav25_gen as m;print(m.__file__)' 2>&1 | tail -1)"
  echo "threads  OMP_NUM_THREADS=${OMP_NUM_THREADS:-<unset>}"
  echo '```'
  echo
  echo '## Per-dataset build state — what step 3 needs'; echo
  echo '| dataset | holdout | built `data.h5ad` | `train_registered.h5ad` | raw source |'
  echo '|---|---|---|---|---|'
  for e in $DATASETS starmap_visual_cortex:paper_2_4_6 deep_starmap:paper_2_4_6; do
    d=${e%%:*}; h=${e##*:}
    echo "| \`$d\` | $h | $(ok "$B3DATA/$d/data.h5ad") | $(ok "$B3RES/_inputs/$d/$h/train_registered.h5ad") | $(ok "$V1_ROOT/data/raw/$d") |"
  done
  echo
  echo 'Missing a built `data.h5ad`:'
  echo '`cd '"$BENCH3"' && python -m src.bench3.prepare_dataset --dataset <name>`'
  echo
  echo 'Built, but missing `train_registered.h5ad`: it is created by the first method run or by'
  echo '`cd '"$BENCH3"' && python -m src.bench3.selftest --dataset <name>`. Both are cheap next to a fit,'
  echo 'but neither is free and neither was costed in §4 — report what is missing rather than building it.'
} | tee "$OUT/step0_preflight.md"
echo

# =====================================================================================
# STEP 1 — does a predictions tree exist? Forks the comparator bill ~10x.
# `evaluate_all --force` re-scores by globbing prediction.h5 (evaluate_all.py:43). A
# metrics.json with no prediction.h5 beside it CANNOT be re-scored — the method re-runs.
# =====================================================================================
echo "=============== STEP 1  comparator predictions inventory ==============="
{
  echo "# Step 1 — comparator predictions inventory"; echo; echo "Generated $(date -Is)"; echo
  echo '## Evaluator pin'; echo
  ev="$BENCH3/src/bench3/evaluate_paper.py"
  if [ -f "$ev" ]; then
    got=$(sha256sum "$ev" | cut -d' ' -f1)
    want=7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992
    echo '```'; echo "measured $got"; echo "pinned   $want"; echo '```'
    [ "$got" = "$want" ] && echo "**MATCH** ($(wc -l < "$ev") lines)" \
                         || echo '🚨 **MISMATCH — stop. The instrument moved.**'
  else echo "🚨 not found at \`$ev\`"; fi
  echo
  ROOTS="${BENCH_V3_RESULTS:-} $BENCH3/results $V2_ROOT/results $V1_ROOT/results $REPO/results $REPO/runs"
  echo '## `prediction.h5` found'; echo
  echo '| results root | method | dataset | holdout | size |'; echo '|---|---|---|---|---|'
  n=0
  for root in $ROOTS; do
    [ -d "$root" ] || continue
    while IFS= read -r p; do
      rel=${p#"$root"/}; rel=${rel%/prediction.h5}
      echo "| \`$root\` | $(echo "$rel"|cut -d/ -f1) | $(echo "$rel"|cut -d/ -f2) | $(echo "$rel"|cut -d/ -f3) | $(du -h "$p"|cut -f1) |"
      n=$((n+1))
    done < <(find "$root" -name prediction.h5 2>/dev/null | sort)
  done
  [ "$n" = 0 ] && echo '| — | — | — | — | **none** |'
  echo; echo "**total: $n**"; echo
  echo '## Quarantined `.h5.degraded` — NOT re-scorable'; echo
  q=0; for root in $ROOTS; do [ -d "$root" ] || continue
    while IFS= read -r p; do echo "- \`$p\`"; q=$((q+1)); done \
      < <(find "$root" -name 'prediction.h5.degraded' 2>/dev/null | sort); done
  echo; echo "**total: $q**"; echo
  echo '## `metrics.json` with no `prediction.h5` beside it — re-scoring impossible'; echo
  o=0; for root in $ROOTS; do [ -d "$root" ] || continue
    while IFS= read -r p; do [ -f "$(dirname "$p")/prediction.h5" ] || { echo "- \`$p\`"; o=$((o+1)); }; done \
      < <(find "$root" -name metrics.json 2>/dev/null | sort); done
  echo; echo "**total: $o**"; echo
  echo '## Verdict'; echo
  if [ "$n" -gt 0 ]; then
    echo "Fork **A1** — $n predictions exist. Re-score with \`python -m src.bench3.evaluate_all --force\`;"
    echo 'no method is re-run. Check the table covers every (method, dataset, holdout, seed) the'
    echo 'campaign needs — anything missing still falls to A2.'
  else
    echo 'Fork **A2** — no predictions. All four comparators re-run from scratch. Their runtime is'
    echo 'unmeasured in this repo: run 4 methods x the chosen datasets, ONE seed each, timed, and'
    echo 're-derive the line item before committing the rest.'
  fi
} | tee "$OUT/step1_predictions.md"
echo

# =====================================================================================
# STEP 2 — flank_r, bench3's own discriminability probe. NOTE: survey_datasets screens
# V1's processed tree (`--root <v1>/data/processed`), NOT bench3's built datasets. They
# are different trees; step 0 reports both.
# =====================================================================================
echo "=============== STEP 2  survey_datasets (flank_r) ==============="
if [ -d "$V1_ROOT/data/processed" ]; then
  ( cd "$BENCH3" && "$PY" -m src.bench3.survey_datasets --root "$V1_ROOT" --csv "$OUT/step2_survey.csv" ) \
    | tee "$OUT/step2_survey.txt"
  echo; echo "-> read flank_r first: at 0.98+ (STARmap's value) there is no room between a trivial"
  echo "   copy and a perfect reconstruction, and the dataset cannot rank methods."
else
  echo "SKIP — no v1 processed tree at $V1_ROOT/data/processed (set V1_ROOT)." | tee "$OUT/step2_survey.txt"
fi
echo

# =====================================================================================
# STEP 3 — model-free ceiling + bootstrap on the five unmeasured survivors. No model,
# no fit, no generation. starmap and deep_starmap are already measured and not repeated.
#
# Both scripts are marker_depth_r-only, so this covers ONE of the six headline metrics.
# Extending eval/ceiling.py to all six is the 1-2 day item in §7b.
# =====================================================================================
echo "=============== STEP 3  ceiling + bootstrap ==============="
for e in $DATASETS; do
  d=${e%%:*}; h=${e##*:}
  echo "--- $d ($h) ---"
  [ -f "$B3DATA/$d/data.h5ad" ] || { echo "SKIP $d — not built (see step 0)"; continue; }
  [ -f "$B3RES/_inputs/$d/$h/train_registered.h5ad" ] || { echo "SKIP $d — no train_registered.h5ad for $h (see step 0)"; continue; }
  "$PY" "$REPO/scripts/t09_depth_ceiling.py" --dataset "$d" --holdout "$h" \
      --bench3 "$BENCH3" --v2-methods "$V2_ROOT/src/benchmark/methods" \
      --out "$OUT/ceiling_$d.md" || { echo "FAIL $d — ceiling"; continue; }
  "$PY" "$REPO/scripts/t09_ceiling_bootstrap.py" --dataset "$d" --holdout "$h" \
      --bench3 "$BENCH3" --v2-methods "$V2_ROOT/src/benchmark/methods" \
      --compare "$REPO/reports/t09_ceiling_bootstrap_starmap.json" \
      --out "$OUT/bootstrap_$d.md" || echo "FAIL $d — bootstrap"
done
echo

cat <<STEP4
=============== STEP 4  deep_starmap peak RSS — ONE SHORT FIT, run by hand ===============

Sets concurrency, and so the campaign's wall clock. Peak RSS is reached during load, cKDTree
construction and the first steps, so 100 steps gives the same number as 2400 at ~1/24 the cost
— roughly 10 minutes, not ~4 hours.

  cd $REPO
  OMP_NUM_THREADS=1 SPATIALCPA_BENCH3=$BENCH3 /usr/bin/time -v \\
    $PY scripts/t09_zeroshot_run.py \\
      --arm lookup --seed 1 --train-steps 100 --fit-only \\
      --split reports/t09_gene_split_deep.json \\
      --workdir /tmp/rss_probe --dataset deep_starmap \\
      2>&1 | tee $OUT/step4_rss.txt

Read "Maximum resident set size (kbytes)". Concurrency = usable RAM / that; if it lands high,
the 24-way schedule in §7e does not hold.
STEP4

echo
echo "Artifacts in $OUT"
echo "  cd $REPO && tar czf screen.tgz screen.log reports/screen/"
