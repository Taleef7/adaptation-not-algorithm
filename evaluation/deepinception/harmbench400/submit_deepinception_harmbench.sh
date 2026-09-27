#!/bin/bash
# Original-grid DeepInception on HarmBench-400: one SLURM job per configuration (25 configurations),
# each running deepinception_harmbench.py. Configurations whose output exists are skipped by the script.
# Usage: bash submit_deepinception_harmbench.sh [model_name ...]   (default: all 25)
# Required env: PROJECT_ROOT
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$PROJECT_ROOT"
M=$PROJECT_ROOT/models
HB=$PROJECT_ROOT/toolkits/HarmBench
CSV=$HB/data/behavior_datasets/harmbench_behaviors_text_all.csv
LOG=$PROJECT_ROOT/logs/deepinception_harmbench; mkdir -p "$LOG"

# model_name -> "model_path|adapter_path|load_in_4bit"
declare -A CFG=(
  [gemma2_base_fp16_custom]="$M/gemma2-9b/gemma2-9b-it-base||0"
  [gemma2_base_4bit_custom]="$M/gemma2-9b/gemma2-9b-it-base||1"
  [gemma2_lora_custom]="$M/gemma2-9b/gemma2-9b-it-base|$M/gemma2-9b/gemma2-9b-it-lora-alpaca|0"
  [gemma2_qlora_custom]="$M/gemma2-9b/gemma2-9b-it-base|$M/gemma2-9b/gemma2-9b-it-qlora-alpaca|1"
  [gemma2_fft_custom]="$M/gemma2-9b/gemma2-9b-it-fft-alpaca||0"
  [llama31_base_fp16_custom]="$M/llama3.1/llama-3.1-8b-instruct-base||0"
  [llama31_base_4bit_custom]="$M/llama3.1/llama-3.1-8b-instruct-base||1"
  [llama31_lora_custom]="$M/llama3.1/llama-3.1-8b-instruct-base|$M/llama3.1/llama-3.1-8b-lora-alpaca|0"
  [llama31_qlora_custom]="$M/llama3.1/llama-3.1-8b-instruct-base|$M/llama3.1/llama-3.1-8b-qlora-alpaca-hf|1"
  [llama31_fft_custom]="$M/llama3.1/llama-3.1-8b-fft-alpaca-hf-final||0"
  [qwen3_base_fp16_custom]="$M/qwen3-4b/qwen3-4b-instruct-base||0"
  [qwen3_base_4bit_custom]="$M/qwen3-4b/qwen3-4b-instruct-base||1"
  [qwen3_lora_custom]="$M/qwen3-4b/qwen3-4b-instruct-base|$M/qwen3-4b/qwen3-4b-lora-alpaca|0"
  [qwen3_qlora_custom]="$M/qwen3-4b/qwen3-4b-instruct-base|$M/qwen3-4b/qwen3-4b-qlora-alpaca|1"
  [qwen3_fft_custom]="$M/qwen3-4b/qwen3-4b-fft-alpaca||0"
  [phi4_base_fp16_custom]="$M/phi-4/phi-4-base||0"
  [phi4_base_4bit_custom]="$M/phi-4/phi-4-base||1"
  [phi4_lora_custom]="$M/phi-4/phi-4-base|$M/phi-4/phi-4-lora-alpaca|0"
  [phi4_qlora_custom]="$M/phi-4/phi-4-base|$M/phi-4/phi-4-qlora-alpaca|1"
  [phi4_fft_custom]="$M/phi-4/fft_alpaca||0"
  [qwen25_14b_base_fp16_custom]="$M/qwen2.5-14b/qwen2.5-14b-instruct-base||0"
  [qwen25_14b_base_4bit_custom]="$M/qwen2.5-14b/qwen2.5-14b-instruct-base||1"
  [qwen25_14b_lora_custom]="$M/qwen2.5-14b/qwen2.5-14b-instruct-base|$M/qwen2.5-14b/qwen2.5-14b-lora-alpaca|0"
  [qwen25_14b_qlora_custom]="$M/qwen2.5-14b/qwen2.5-14b-instruct-base|$M/qwen2.5-14b/qwen2.5-14b-qlora-alpaca|1"
  [qwen25_14b_fft_custom]="$M/qwen2.5-14b/qwen2.5-14b-fft-alpaca||0"
)
MODELS="${*:-${!CFG[*]}}"

for model in $MODELS; do
  IFS='|' read -r mpath adapter q4 <<< "${CFG[$model]}"
  args="--model_name $model --model_path $mpath --behaviors_csv $CSV \
--save_path $HB/results/DeepInception_HarmBench/$model/completions/$model.json"
  [ -n "$adapter" ] && args="$args --adapter_path $adapter"
  [ "$q4" = 1 ] && args="$args --load_in_4bit"
  jid=$(sbatch --parsable --account=YOUR_ACCOUNT --partition=a100-80gb --qos=standby \
    --gres=gpu:1 --mem=80G --time=12:00:00 --job-name="di_hb_${model}" \
    --output="$LOG/${model}_%j.log" \
    --wrap="module load external anaconda && conda activate harmbench_env && export PYTHONNOUSERSITE=1 && \
cd $PROJECT_ROOT && python -u evaluation/deepinception/harmbench400/deepinception_harmbench.py $args")
  echo "submitted $model: $jid"
done
