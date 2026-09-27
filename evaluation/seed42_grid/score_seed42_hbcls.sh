#!/bin/bash
# Controlled-grid (seed-42) HB-Cls scoring: every DeepInception and ArtPrompt completion file
# (15 cells x {jbb, harmbench}) is scored with hb_cls_score.py
# (cais/HarmBench-Llama-2-13b-cls, greedy single token). Run inside a GPU job with >= 40 GB
# (e.g. via score_seed42_hbcls.slurm).
#
# A file is scored only if it has the full row count (jbb=100, harmbench=400). Existing outputs
# are skipped, so the script can be re-run to fill in missing cells.
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
DI=$PROJECT_ROOT/toolkits/HarmBench/results_phase5/deepinception
AP=$PROJECT_ROOT/toolkits/HarmBench/results_phase5/artprompt
OUT=$PROJECT_ROOT/toolkits/HarmBench/results_phase5/scored
mkdir -p "$OUT"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1

CELLS="llama31_lora llama31_qlora llama31_fft gemma2_lora gemma2_qlora gemma2_fft \
qwen3_lora qwen3_qlora qwen3_fft phi4_lora phi4_qlora phi4_fft qwen25_lora qwen25_qlora qwen25_fft"

expected_n () { case "$1" in jbb) echo 100;; harmbench) echo 400;; *) echo -1;; esac; }

for cell in $CELLS; do
  for bench in jbb harmbench; do
    for pair in "$DI|_hbcls" "$AP|_artprompt_hbcls"; do
      IFS='|' read -r dir suf <<< "$pair"
      base="${cell}_seed42_${bench}"
      inp="$dir/${base}.jsonl"
      out="$OUT/${base}${suf}.jsonl"
      exp=$(expected_n "$bench")
      if [ ! -f "$inp" ]; then echo "SKIP_MISSING $inp"; continue; fi
      n=$(wc -l < "$inp")
      if [ "$n" -ne "$exp" ]; then echo "SKIP_INCOMPLETE $inp (n=$n expected=$exp)"; continue; fi
      if [ -f "$out" ]; then echo "SKIP_DONE $out"; continue; fi
      echo "=== scoring $base $(basename "$dir") ($n rows) ==="
      "$VENV_DIR/bin/python" -u "$PROJECT_ROOT/evaluation/seed42_grid/hb_cls_score.py" \
          --completions "$inp" --out "$out"
    done
  done
done
echo "SEED42_HBCLS_SCORING_DONE"
