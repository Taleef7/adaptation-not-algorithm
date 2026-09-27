#!/bin/bash

# AutoDAN JBB Step 3: Evaluate Completions with Llama-2 Classifier
# Direct submission bypassing run_pipeline.py
# This script evaluates the AutoDAN-generated completions using the HarmBench Llama-2 classifier

set -e

# Configuration
HARMBENCH_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="data/behavior_datasets/jbb_behaviors.csv"
CLASSIFIER="cais/HarmBench-Llama-2-13b-cls"

# SLURM Configuration
PARTITION="training"
ACCOUNT="YOUR_ACCOUNT"
MEM="90G"
QOS="training"
NODES="1"
GPUS="1"
TIME="4:00:00"

# Model configurations
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

echo "========================================"
echo "AutoDAN JBB Step 3: Llama-2 Evaluation"
echo "========================================"
echo "Classifier: $CLASSIFIER"
echo "Behaviors: $JBB_BEHAVIORS"
echo "Total models: ${#MODELS[@]}"
echo ""

# Create log directories
for MODEL in "${MODELS[@]}"; do
    mkdir -p "$HARMBENCH_DIR/slurm_logs_jbb/evaluate_completions/AutoDAN/$MODEL"
done

# Submit evaluation jobs
for MODEL in "${MODELS[@]}"; do
    COMPLETIONS_PATH="results_jbb/AutoDAN/$MODEL/completions/completions.json"
    SAVE_PATH="results_jbb/AutoDAN/$MODEL/results/results.json"
    LOG_FILE="$HARMBENCH_DIR/slurm_logs_jbb/evaluate_completions/AutoDAN/$MODEL/evaluation.log"
    JOB_NAME="eval_${MODEL:0:8}"
    
    # Verify completions file exists
    if [ ! -f "$HARMBENCH_DIR/$COMPLETIONS_PATH" ]; then
        echo "❌ Skipping $MODEL: completions file not found"
        continue
    fi
    
    # Create results directory
    mkdir -p "$HARMBENCH_DIR/$(dirname $SAVE_PATH)"
    
    # Submit evaluation job
    sbatch \
        --partition="$PARTITION" \
        --account="$ACCOUNT" \
        --mem="$MEM" \
        --job-name="$JOB_NAME" \
        --qos="$QOS" \
        --nodes="$NODES" \
        --gpus-per-node="$GPUS" \
        --nodelist=gilbreth-j[000-001] \
        --time="$TIME" \
        --output="$LOG_FILE" \
        --wrap="module purge && module load external anaconda && source activate harmbench_env && cd $HARMBENCH_DIR && python -u evaluate_completions.py --cls_path $CLASSIFIER --behaviors_path $JBB_BEHAVIORS --completions_path $COMPLETIONS_PATH --save_path $SAVE_PATH"
    
    echo "→ Submitted $MODEL"
done

echo ""
echo "All evaluation jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "Check logs in: $HARMBENCH_DIR/slurm_logs_jbb/evaluate_completions/AutoDAN/"
