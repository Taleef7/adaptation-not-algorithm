#!/bin/bash
# Build secondary-evaluator (GPT-4o-mini, LlamaGuard-3) scoring manifests for the seed-42
# controlled-grid completions. One "IN_PATH OUT_PATH" line per (cell x attack x benchmark);
# the nine <=9B cells first, then the six 14B cells. Inputs that do not exist are skipped.
# Consumed by secondary_eval.slurm.
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$PROJECT_ROOT"
RP=toolkits/HarmBench/results_phase5
MDIR=evaluation/seed42_grid/manifests
mkdir -p "$MDIR" logs

STD="llama31_lora gemma2_lora qwen3_lora llama31_qlora gemma2_qlora qwen3_qlora llama31_fft gemma2_fft qwen3_fft"
B14="phi4_lora phi4_qlora qwen25_lora qwen25_qlora phi4_fft qwen25_fft"

for EVAL in gpt4omini llamaguard; do
  MF="$MDIR/manifest_${EVAL}.txt"
  : > "$MF"
  for cell in $STD $B14; do
    for bench in jbb harmbench; do
      di_in="$RP/deepinception/${cell}_seed42_${bench}.jsonl"
      ap_in="$RP/artprompt/${cell}_seed42_${bench}.jsonl"
      di_out="$RP/scored/${cell}_seed42_${bench}_${EVAL}.jsonl"
      ap_out="$RP/scored/${cell}_seed42_${bench}_artprompt_${EVAL}.jsonl"
      [ -f "$di_in" ] && echo "$di_in $di_out" >> "$MF"
      [ -f "$ap_in" ] && echo "$ap_in $ap_out" >> "$MF"
    done
  done
  echo "$EVAL manifest -> $MF  ($(wc -l < "$MF") files)"
done
