#!/bin/bash

# Direct AutoDAN Step 1 submission for JBB dataset
# This bypasses run_pipeline.py to explicitly specify the JBB config file

set -e

HARMBENCH_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_CONFIG="$HARMBENCH_DIR/configs/method_configs/AutoDAN_config_jbb.yaml"
JBB_BEHAVIORS="$HARMBENCH_DIR/data/behavior_datasets/jbb_behaviors.csv"
BASE_SAVE_DIR="$HARMBENCH_DIR/results_jbb"
BASE_LOG_DIR="$HARMBENCH_DIR/slurm_logs_jbb"

# AutoDAN parameters
ATTACK_METHOD="AutoDAN"
PARTITION="training"
ACCOUNT="YOUR_ACCOUNT"
QOS="training"
NODES=1
GPUS=2
MEM="80G"
TIME="4:00:00"
BEHAVIOR_CHUNK_SIZE=5

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
echo "AutoDAN Step 1 - JBB Dataset"
echo "========================================"
echo "Config: $JBB_CONFIG"
echo "Behaviors: $JBB_BEHAVIORS"
echo "Models: ${#MODELS[@]}"
echo "========================================"

# Get total number of behaviors
TOTAL_BEHAVIORS=$(tail -n +2 "$JBB_BEHAVIORS" | wc -l)
echo "Total behaviors: $TOTAL_BEHAVIORS"

# Calculate number of chunks
NUM_CHUNKS=$(( ($TOTAL_BEHAVIORS + $BEHAVIOR_CHUNK_SIZE - 1) / $BEHAVIOR_CHUNK_SIZE ))
echo "Behavior chunks: $NUM_CHUNKS (size=$BEHAVIOR_CHUNK_SIZE)"

JOBS_SUBMITTED=0

# Submit jobs for each model
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "→ Processing $MODEL"
    
    SAVE_DIR="$BASE_SAVE_DIR/$ATTACK_METHOD/$MODEL/test_cases"
    LOG_DIR="$BASE_LOG_DIR/generate_test_cases/$ATTACK_METHOD/$MODEL"
    
    mkdir -p "$SAVE_DIR"
    mkdir -p "$LOG_DIR"
    
    # Submit jobs for behavior chunks
    for ((chunk=0; chunk<$NUM_CHUNKS; chunk++)); do
        START_IDX=$((chunk * BEHAVIOR_CHUNK_SIZE))
        END_IDX=$((START_IDX + BEHAVIOR_CHUNK_SIZE))
        
        # Don't exceed total behaviors
        if [ $END_IDX -gt $TOTAL_BEHAVIORS ]; then
            END_IDX=$TOTAL_BEHAVIORS
        fi
        
        JOB_NAME="generate_test_cases_AutoDAN_${MODEL}_${START_IDX}_${END_IDX}"
        LOG_FILE="$LOG_DIR/start_idx_${START_IDX}.log"
        
        # Submit SLURM job with explicit config file and full environment setup
        sbatch --partition="$PARTITION" --account="$ACCOUNT" --mem="$MEM" --job-name="$JOB_NAME" --qos="$QOS" --nodes="$NODES" --gpus-per-node="$GPUS" --nodelist=gilbreth-j[000-001] --time="$TIME" --output="$LOG_FILE" --wrap="module purge && module load external && module load anaconda && source activate harmbench_env && cd $HARMBENCH_DIR && python -u generate_test_cases.py --method_name $ATTACK_METHOD --experiment_name $MODEL --method_config_file $JBB_CONFIG --behaviors_path $JBB_BEHAVIORS --save_dir $SAVE_DIR --behavior_start_idx $START_IDX --behavior_end_idx $END_IDX --overwrite" > /dev/null
        
        JOBS_SUBMITTED=$((JOBS_SUBMITTED + 1))
    done
done

echo ""
echo "========================================"
echo "✓ Submitted $JOBS_SUBMITTED AutoDAN jobs"
echo "========================================"
echo ""
echo "Monitor progress with:"
echo "  watch -n 10 'squeue -u \$USER | head -20'"
echo ""
echo "Check logs:"
echo "  tail -f $BASE_LOG_DIR/generate_test_cases/$ATTACK_METHOD/*/start_idx_0.log"
