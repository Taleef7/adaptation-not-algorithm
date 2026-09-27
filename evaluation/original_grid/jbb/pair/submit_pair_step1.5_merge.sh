#!/bin/bash

# PAIR JBB Step 1.5: Merge Test Cases
# Merges individual behavior test cases into single test_cases.json per model

HARMBENCH_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
BASE_SAVE_DIR="$HARMBENCH_DIR/results_jbb"
BASE_LOG_DIR="$HARMBENCH_DIR/slurm_logs_jbb"

# SLURM parameters (quick jobs)
PARTITION="a30"
ACCOUNT="YOUR_ACCOUNT"
QOS="standby"
MEM="16G"
TIME="00:30:00"

# All 20 models
MODELS=(
    "gemma2_base_fp16_custom"
    "gemma2_base_4bit_custom"
    "gemma2_lora_custom"
    "gemma2_qlora_custom"
    "gemma2_fft_custom"
    "llama31_base_fp16_custom"
    "llama31_base_4bit_custom"
    "llama31_lora_custom"
    "llama31_qlora_custom"
    "llama31_fft_custom"
    "qwen3_base_fp16_custom"
    "qwen3_base_4bit_custom"
    "qwen3_lora_custom"
    "qwen3_qlora_custom"
    "qwen3_fft_custom"
    "phi4_base_fp16_custom"
    "phi4_base_4bit_custom"
    "phi4_lora_custom"
    "phi4_qlora_custom"
    "phi4_fft_custom"
)

cd "$HARMBENCH_DIR"

echo "========================================"
echo "PAIR Step 1.5 - JBB Dataset (Merge Test Cases)"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "========================================"
echo ""

SUBMITTED=0

for MODEL in "${MODELS[@]}"; do
    EXPERIMENT_NAME="${MODEL}_pair_experiment"
    SAVE_DIR="$BASE_SAVE_DIR/PAIR/$EXPERIMENT_NAME/test_cases"
    LOG_DIR="$BASE_LOG_DIR/merge_test_cases/PAIR/$EXPERIMENT_NAME"
    
    mkdir -p "$LOG_DIR"
    
    JOB_NAME="pair_merge_${MODEL:0:8}"
    LOG_FILE="$LOG_DIR/merge.log"
    
    sbatch \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --mem="$MEM" \
        --job-name="$JOB_NAME" \
        --qos="$QOS" \
        --nodes=1 \
        --cpus-per-task=1 \
        --gres=gpu:1 \
        --time="$TIME" \
        --output="$LOG_FILE" \
        --wrap="module purge && module load external anaconda && source activate harmbench_env && cd $HARMBENCH_DIR && python -u merge_test_cases.py --method_name PAIR --save_dir $SAVE_DIR"
    
    echo "  ✓ Submitted merge for $MODEL"
    ((SUBMITTED++))
done

echo ""
echo "========================================"
echo "✓ Submitted $SUBMITTED PAIR Step 1.5 (merge) jobs"
echo "========================================"
echo ""
echo "Monitor progress with:"
echo "  watch -n 5 'squeue -u \$USER | grep pair_merge'"
echo ""
echo "After completion, verify with:"
echo "  ls -lh $BASE_SAVE_DIR/PAIR/*/test_cases/test_cases.json"

