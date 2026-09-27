#!/bin/bash
# HB-Cls scoring for recipe-robustness arms with seed42_grid/hb_cls_score.py (same classifier and
# decoding as the controlled grid). Only complete files (jbb=100, harmbench=400) are scored;
# existing outputs are skipped. Run inside a GPU job with >= 40 GB.
# Usage: ./score_arm.sh <arm> <family1> [family2 ...]   e.g. ./score_arm.sh dolly1ep llama31 qwen3
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
ARM=$1; shift; FAMS="$*"
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
source $REPO/evaluation/slurm/env.sh
HBENV=$VENV_DIR
CLS=cais/HarmBench-Llama-2-13b-cls
DI=$REPO/toolkits/HarmBench/results_track2/deepinception
AP=$REPO/toolkits/HarmBench/results_track2/artprompt
OUT=$REPO/toolkits/HarmBench/results_track2/scored
mkdir -p "$OUT"; cd "$REPO"
expected_n () { case "$1" in jbb) echo 100;; harmbench) echo 400;; *) echo -1;; esac; }
echo "ARM_SCORING_START $ARM [$FAMS]"
for fam in $FAMS; do for method in lora fft; do
  cell="${fam}_${method}_${ARM}"
  for bench in jbb harmbench; do
    for pair in "$DI|_hbcls" "$AP|_artprompt_hbcls"; do
      IFS='|' read -r dir suf <<< "$pair"
      inp="$dir/${cell}_${bench}.jsonl"; out="$OUT/${cell}_${bench}${suf}.jsonl"
      exp=$(expected_n "$bench")
      [ ! -f "$inp" ] && { echo "SKIP_MISSING $inp"; continue; }
      n=$(wc -l < "$inp")
      [ "$n" -ne "$exp" ] && { echo "SKIP_INCOMPLETE $inp (n=$n exp=$exp)"; continue; }
      [ -f "$out" ] && { echo "SKIP_DONE $out"; continue; }
      echo "=== scoring $cell $bench $(basename $dir) ($n rows) ==="
      "$HBENV/bin/python" -u "$REPO/evaluation/seed42_grid/hb_cls_score.py" --completions "$inp" --out "$out" --cls "$CLS"
    done
  done
done; done
echo "ARM_SCORING_DONE $ARM"
