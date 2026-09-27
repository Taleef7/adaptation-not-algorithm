#!/bin/bash

# PAIR JBB Step 1: Generate Test Cases
# Direct submission for JBB dataset

HARMBENCH_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_CONFIG="$HARMBENCH_DIR/configs/method_configs/PAIR_config_jbb.yaml"
JBB_BEHAVIORS="$HARMBENCH_DIR/data/behavior_datasets/jbb_behaviors.csv"
BASE_SAVE_DIR="$HARMBENCH_DIR/results_jbb"
BASE_LOG_DIR="$HARMBENCH_DIR/slurm_logs_jbb"

# PAIR parameters
ATTACK_METHOD="PAIR"
PARTITION="training"
ACCOUNT="YOUR_ACCOUNT"
QOS="training"
NODES=1
GPUS=2
MEM="90G"
TIME="12:00:00"

# Models - PAIR uses _pair_experiment suffix
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

# Number of behaviors per chunk (JBB has 100 behaviors)
BEHAVIORS_PER_CHUNK=5
TOTAL_BEHAVIORS=100
NUM_CHUNKS=$((TOTAL_BEHAVIORS / BEHAVIORS_PER_CHUNK))

cd "$HARMBENCH_DIR"

echo "========================================"
echo "PAIR Step 1 - JBB Dataset (Generate Test Cases)"
echo "========================================"
echo "Models: ${#MODELS[@]}"
echo "Behaviors per chunk: $BEHAVIORS_PER_CHUNK"
echo "Total chunks per model: $NUM_CHUNKS"
echo "========================================"
echo ""

SUBMITTED=0

for MODEL in "${MODELS[@]}"; do
    # PAIR uses <model>_pair_experiment naming
    EXPERIMENT_NAME="${MODEL}_pair_experiment"
    
    echo "→ Processing $MODEL (as $EXPERIMENT_NAME)"
    
    SAVE_DIR="$BASE_SAVE_DIR/$ATTACK_METHOD/$EXPERIMENT_NAME/test_cases"
    LOG_DIR="$BASE_LOG_DIR/generate_test_cases/$ATTACK_METHOD/$EXPERIMENT_NAME"
    
    # Create directories
    mkdir -p "$SAVE_DIR"
    mkdir -p "$LOG_DIR"
    
    # Submit chunked jobs (20 chunks of 5 behaviors each)
    for ((CHUNK=0; CHUNK<NUM_CHUNKS; CHUNK++)); do
        START_IDX=$((CHUNK * BEHAVIORS_PER_CHUNK))
        END_IDX=$(((CHUNK + 1) * BEHAVIORS_PER_CHUNK))
        
        JOB_NAME="pair_step1_${MODEL:0:6}_${CHUNK}"
        LOG_FILE="$LOG_DIR/chunk_${START_IDX}_${END_IDX}.log"
        
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
            --wrap="module purge && module load external anaconda && source activate harmbench_env && cd $HARMBENCH_DIR && python -u generate_test_cases.py --method_name $ATTACK_METHOD --experiment_name $EXPERIMENT_NAME --method_config_file $JBB_CONFIG --behaviors_path $JBB_BEHAVIORS --save_dir $SAVE_DIR --behavior_start_idx $START_IDX --behavior_end_idx $END_IDX --overwrite" > /dev/null
        
        ((SUBMITTED++))
    done
    
    echo "  ✓ Submitted $NUM_CHUNKS chunks for $MODEL"
done

echo ""
echo "========================================"
echo "✓ Submitted $SUBMITTED PAIR Step 1 jobs"
echo "========================================"
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'squeue -u \$USER | head -30'"
echo ""
echo "Check logs:"
echo "  tail -f $BASE_LOG_DIR/generate_test_cases/$ATTACK_METHOD/*/chunk_*.log"
