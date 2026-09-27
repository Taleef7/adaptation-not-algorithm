#!/bin/bash
# LlamaGuard-3 scoring (HarmBench evaluate_completions_llamaguard.py, --num_tokens 512) for the
# 15 GCG cells (five families x {base_fp16, lora, fft}, JBB-100). Run from any directory.
# GCG writes completions to completions/<model>.json. Existing outputs are skipped.
# Required env: PROJECT_ROOT (conda env "harmbench_env" is activated inside each job)

PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv"
MODEL_PATH="${PROJECT_ROOT}/models/llama-guard-3-8b"

cd "$HARMBENCH_ROOT"

MODELS=(
    "llama31_base_fp16_custom" "llama31_lora_custom" "llama31_fft_custom"
    "qwen25_14b_base_fp16_custom" "qwen25_14b_lora_custom" "qwen25_14b_fft_custom"
    "gemma2_base_fp16_custom" "gemma2_lora_custom" "gemma2_fft_custom"
    "phi4_base_fp16_custom" "phi4_lora_custom" "phi4_fft_custom"
    "qwen3_base_fp16_custom" "qwen3_lora_custom" "qwen3_fft_custom"
)

ATTACK="GCG"
submitted=0

echo "Submitting Llama Guard evals for JBB ${ATTACK} (${#MODELS[@]} models)"

for model in "${MODELS[@]}"; do
    COMPLETIONS="${HARMBENCH_ROOT}/results_jbb/${ATTACK}/${model}/completions/${model}.json"
    SAVE_PATH="${HARMBENCH_ROOT}/results_jbb/${ATTACK}/${model}/results_llamaguard/${model}.json"
    LOG_DIR="${HARMBENCH_ROOT}/slurm_logs_jbb/evaluate_completions_llamaguard/${ATTACK}/${model}"

    if [ ! -f "$COMPLETIONS" ]; then echo "SKIP_MISSING $COMPLETIONS"; continue; fi
    if [ -f "$SAVE_PATH" ]; then echo "SKIP_DONE $SAVE_PATH"; continue; fi
    mkdir -p "$LOG_DIR" "$(dirname $SAVE_PATH)"

    WRAP_CMD="module load external anaconda && conda activate harmbench_env && python evaluate_completions_llamaguard.py --behaviors_path '$JBB_BEHAVIORS' --completions_path '$COMPLETIONS' --save_path '$SAVE_PATH' --model_path '$MODEL_PATH' --num_tokens 512"

    # a30 (24GB) holds LlamaGuard-3-8B in bf16 with room for a 512-token context.
    output=$(sbatch --partition=a30 --account=YOUR_ACCOUNT --qos=standby --nodes=1 --gres=gpu:1 --mem=32G --time=2:00:00 --job-name="LG-GC-${model:0:8}" --output="${LOG_DIR}/${model}.log" --error="${LOG_DIR}/${model}.log" --wrap="$WRAP_CMD" 2>&1)

    if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        : $((submitted++))
        echo "[$submitted/${#MODELS[@]}] ${model} (ID: ${BASH_REMATCH[1]})"
    else
        echo "Error: $output"
    fi
done

echo "Done! $submitted jobs submitted."
