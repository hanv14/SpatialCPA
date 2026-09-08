#!/usr/bin/env bash
# Steps 1-4 of the pre-campaign screen (reports/v25_campaign_costing.md §6).
#
# Runs on the campaign machine. Steps 1-3 involve NO fits. Step 4 is one short
# probe fit and is NOT run by this script -- it is printed at the end.
#
#   bash scripts/campaign_screen.sh 2>&1 | tee screen.log
#   tar czf screen.tgz screen.log reports/screen/
#
# Everything lands under reports/screen/ so the whole thing patches back as one tarball.

set -uo pipefail   # deliberately NOT -e: one dataset failing must not abort the screen

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
BENCH3="${SPATIALCPA_BENCH3:-$REPO/benchmark-pbya-v3}"
OUT="$REPO/reports/screen"
mkdir -p "$OUT"

echo "repo    $REPO"
echo "bench3  $BENCH3"
echo "date    $(date -Is)"
echo

# ----------------------------------------------------------------------------------
# STEP 1 -- does a predictions tree exist?  (forks the comparator bill ~10x)
#
# `evaluate_all --force` re-scores by globbing `prediction.h5` under the results root
# (evaluate_all.py:43).  A metrics.json without its prediction.h5 CANNOT be re-scored:
# the method has to be re-run.  So the thing to count is prediction.h5, per
# (method, dataset, holdout), and separately the `.h5.degraded` quarantined ones,
# which do not count.
# ----------------------------------------------------------------------------------
echo "=============== STEP 1  comparator predictions inventory ==============="
{
  echo "# Step 1 -- comparator predictions inventory"
  echo
  echo "Generated $(date -Is)"
  echo
  echo '## Evaluator pin'
  ev="$BENCH3/src/bench3/evaluate_paper.py"
  if [ -f "$ev" ]; then
    got=$(sha256sum "$ev" | cut -d' ' -f1)
    want=7362669200bbd2be905adf1715c4c6d44842ef1652edb2f4aba697c039538992
    echo "measured  $got"
    echo "pinned    $want"
    [ "$got" = "$want" ] && echo "**MATCH**" || echo "🚨 **MISMATCH -- stop; the instrument moved**"
    echo "lines     $(wc -l < "$ev")"
  else
    echo "🚨 evaluate_paper.py not found at $ev"
  fi
  echo
  echo '## prediction.h5 found'
  echo
  echo '| results root | method | dataset | holdout | size |'
  echo '|---|---|---|---|---|'
  n=0
  for root in "${BENCH_V3_RESULTS:-}" "$BENCH3/results" "$REPO/benchmark-pbya-v2/results" \
              "$REPO/benchmark-pbya/results" "$REPO/results"; do
    [ -n "$root" ] && [ -d "$root" ] || continue
    while IFS= read -r p; do
      rel=${p#"$root"/}; rel=${rel%/prediction.h5}
      m=$(echo "$rel" | cut -d/ -f1); d=$(echo "$rel" | cut -d/ -f2); h=$(echo "$rel" | cut -d/ -f3)
      echo "| \`$root\` | $m | $d | $h | $(du -h "$p" | cut -f1) |"
      n=$((n+1))
    done < <(find "$root" -name prediction.h5 2>/dev/null | sort)
  done
  [ "$n" = 0 ] && echo '| — | — | — | — | **none found** |'
  echo
  echo "**total prediction.h5: $n**"
  echo
  echo '## quarantined (.h5.degraded -- these do NOT count as re-scorable)'
  q=0
  for root in "${BENCH_V3_RESULTS:-}" "$BENCH3/results" "$REPO/benchmark-pbya-v2/results" \
              "$REPO/benchmark-pbya/results" "$REPO/results"; do
    [ -n "$root" ] && [ -d "$root" ] || continue
    while IFS= read -r p; do echo "- \`${p#"$root"/}\` (in $root)"; q=$((q+1))
    done < <(find "$root" -name 'prediction.h5.degraded' 2>/dev/null | sort)
  done
  echo; echo "**total quarantined: $q**"
  echo
  echo '## metrics.json without a prediction.h5 beside it (re-scoring impossible)'
  o=0
  for root in "${BENCH_V3_RESULTS:-}" "$BENCH3/results" "$REPO/benchmark-pbya-v2/results" \
              "$REPO/benchmark-pbya/results" "$REPO/results"; do
    [ -n "$root" ] && [ -d "$root" ] || continue
    while IFS= read -r p; do
      [ -f "$(dirname "$p")/prediction.h5" ] || { echo "- \`${p#"$root"/}\`"; o=$((o+1)); }
    done < <(find "$root" -name metrics.json 2>/dev/null | sort)
  done
  echo; echo "**total orphaned metrics: $o**"
  echo
  echo '## verdict'
  if [ "$n" -gt 0 ]; then
    echo "Fork **A1**: $n predictions exist. Re-score with"
    echo '`python -m src.bench3.evaluate_all --force` -- no method is re-run.'
    echo 'Check the table above covers every (method, dataset, holdout, seed) the campaign needs;'
    echo 'anything missing still falls to fork A2.'
  else
    echo 'Fork **A2**: no predictions. All four comparators re-run from scratch.'
    echo 'Their runtime is unmeasured in this repo -- run 4 methods x 4 datasets ONE seed each,'
    echo 'timed, and re-derive the line item before committing the rest.'
  fi
} | tee "$OUT/step1_predictions.md"
echo

# ----------------------------------------------------------------------------------
# STEP 2 -- flank_r across every built dataset.  bench3's own discriminability probe,
# same _morans_i definition paper_morans_pearson uses.  STARmap reads 0.98.
# ----------------------------------------------------------------------------------
echo "=============== STEP 2  survey_datasets (flank_r) ==============="
( cd "$BENCH3" && python -m src.bench3.survey_datasets --csv "$OUT/step2_survey.csv" ) \
  | tee "$OUT/step2_survey.txt"
echo
echo "-> read the flank_r column first: at 0.98+ there is no room between a trivial copy"
echo "   and a perfect reconstruction, and the dataset cannot rank methods."
echo

# ----------------------------------------------------------------------------------
# STEP 3 -- model-free ceiling + bootstrap on the five unmeasured survivors.
# No model, no fit, no generation.  starmap and deep_starmap are already measured
# (reports/t09_{depth_ceiling,ceiling_bootstrap}_*.md) and are not repeated.
#
# NOTE: both scripts are marker_depth_r-only.  This screen therefore covers 1 of the
# 6 headline metrics (2 with t09_zeroshot_ceiling's morans_pearson).  Extending
# eval/ceiling.py to all six is the 1-2 day item in the costing report.
# ----------------------------------------------------------------------------------
echo "=============== STEP 3  ceiling + bootstrap, 5 unmeasured survivors ==============="
run_ceiling () {  # $1 dataset  $2 holdout
  local ds=$1 ho=$2
  echo "--- $ds ($ho) ---"
  python "$REPO/scripts/t09_depth_ceiling.py" --dataset "$ds" --holdout "$ho" \
      --bench3 "$BENCH3" --out "$OUT/ceiling_${ds}.md" \
    || { echo "SKIP $ds -- ceiling failed (not built? see message above)"; return; }
  python "$REPO/scripts/t09_ceiling_bootstrap.py" --dataset "$ds" --holdout "$ho" \
      --bench3 "$BENCH3" --out "$OUT/bootstrap_${ds}.md" \
      --compare "$REPO/reports/t09_ceiling_bootstrap_starmap.json" \
    || echo "SKIP $ds -- bootstrap failed"
}
run_ceiling cosmx_nsclc_3d              paper_2_4
run_ceiling merfish_thick_hypothalamus  paper_2_4_6
run_ceiling merfish_thick_cortex        paper_2_4_6
run_ceiling exseq_breast_cancer         paper_2_4_6
run_ceiling exseq_visual_cortex         paper_2_4_6
echo

# ----------------------------------------------------------------------------------
# STEP 4 -- peak RSS on deep_starmap.  This IS a fit and is not run automatically.
# ----------------------------------------------------------------------------------
cat <<'STEP4'
=============== STEP 4  deep_starmap peak RSS (ONE SHORT FIT -- run by hand) ===============

Sets the campaign's concurrency, and so its wall clock.  Peak RSS is reached during load,
cKDTree construction and the first steps, so 100 steps gives the same number as 2400 at
~1/24 the cost -- roughly 10 minutes rather than ~4 hours.

  OMP_NUM_THREADS=1 /usr/bin/time -v \
    python scripts/t09_zeroshot_run.py \
      --arm lookup --seed 1 --train-steps 100 --fit-only \
      --split reports/t09_gene_split_deep.json \
      --workdir /tmp/rss_probe --dataset deep_starmap \
      2>&1 | tee reports/screen/step4_rss.txt

Read "Maximum resident set size (kbytes)".  Concurrency = usable RAM / that, and if it
lands high the 24-way schedule in the costing report does not hold.
STEP4

echo
echo "Artifacts in $OUT -- tar czf screen.tgz screen.log reports/screen/"
