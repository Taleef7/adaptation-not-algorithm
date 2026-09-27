#!/bin/bash

# Direct AutoDAN Step 2 submission for JBB dataset - Generate Completions

HARMBENCH_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="$HARMBENCH_DIR/data/behavior_datasets/jbb_behaviors.csv"
BASE_SAVE_DIR="$HARMBENCH_DIR/results_jbb"
BASE_LOG_DIR="$HARMBENCH_DIR/slurm_logs_jbb"

# Step 2 parameters
ATTACK_METHOD="AutoDAN"
PARTITION="training"
ACCOUNT="YOUR_ACCOUNT"
QOS="training"
NODES=1
GPUS=1
MEM="90G"
TIME="4:00:00"

# Models to evaluate
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
echo "AutoDAN Step 2 - JBB Dataset (Generate Completions)"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "========================================"
echo ""

SUBMITTED=0

for MODEL in "${MODELS[@]}"; do
    echo "→ Processing $MODEL"
    
    TEST_CASES_PATH="$BASE_SAVE_DIR/$ATTACK_METHOD/$MODEL/test_cases/test_cases.json"
    SAVE_PATH="$BASE_SAVE_DIR/$ATTACK_METHOD/$MODEL/completions/completions.json"
    LOG_DIR="$BASE_LOG_DIR/generate_completions/$ATTACK_METHOD/$MODEL"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Job name
    JOB_NAME="gen_completions_${MODEL:0:8}"
    LOG_FILE="$LOG_DIR/completions.log"
    
    # Submit job with max_new_tokens=512
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
        --wrap="module purge && module load external anaconda && source activate harmbench_env && cd $HARMBENCH_DIR && python -u generate_completions.py --model_name $MODEL --models_config_file configs/model_configs/models.yaml --behaviors_path $JBB_BEHAVIORS --test_cases_path $TEST_CASES_PATH --save_path $SAVE_PATH --max_new_tokens 512"
    
    echo "  ✓ Submitted $MODEL"
    ((SUBMITTED++))
done

echo ""
echo "========================================"
echo "✓ Submitted $SUBMITTED AutoDAN Step 2 jobs"
echo "========================================"
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'squeue -u \$USER | head -20'"
echo ""
echo "Check logs:"
echo "  tail -f $BASE_LOG_DIR/generate_completions/$ATTACK_METHOD/*/completions.log"
