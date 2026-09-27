#!/bin/bash
# GPT-4o-mini scoring (HarmBench evaluate_attacks_api.py) for the 15 GCG cells on JBB-100:
# responses clipped to 512 gpt2 tokens, output in results_api/results.json. Cells that already
# have results are skipped; evaluate_attacks_api.py resumes from its own checkpoint.
# Required env: PROJECT_ROOT, VENV_DIR, OPENAI_API_KEY
set -u
ROOT="${PROJECT_ROOT}"
HB="${ROOT}/toolkits/HarmBench"
PY="${VENV_DIR}/bin/python"
BEHAVIORS="${HB}/data/behavior_datasets/jbb_behaviors.csv"

CELLS=(
  llama31_base_fp16_custom llama31_lora_custom llama31_fft_custom
  qwen25_14b_base_fp16_custom qwen25_14b_lora_custom qwen25_14b_fft_custom
  gemma2_base_fp16_custom gemma2_lora_custom gemma2_fft_custom
  phi4_base_fp16_custom   phi4_lora_custom   phi4_fft_custom
  qwen3_base_fp16_custom  qwen3_lora_custom  qwen3_fft_custom
)

cd "$HB" || exit 1
for cell in "${CELLS[@]}"; do
  COMP="${HB}/results_jbb/GCG/${cell}/completions/${cell}.json"
  SAVE="${HB}/results_jbb/GCG/${cell}/results_api/results.json"
  if [ ! -f "$COMP" ]; then echo "SKIP_MISSING $cell"; continue; fi
  if [ -f "$SAVE" ];  then echo "SKIP_DONE $cell"; continue; fi
  mkdir -p "$(dirname "$SAVE")"
  echo "=== scoring $cell ==="
  "$PY" evaluate_attacks_api.py \
      --behaviors_path "$BEHAVIORS" \
      --completions_path "$COMP" \
      --save_path "$SAVE" \
      --num_tokens 512 || echo "FAILED $cell"
done
echo "ALL DONE"
