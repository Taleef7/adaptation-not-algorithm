#!/bin/bash
#===============================================================================
# AutoDAN attack on the JBB-100 behaviors (original grid).
#
# Step 1 runs with AutoDAN_config_jbb.yaml (JBB target strings, data/optimizer_targets/jbb_targets_text.json)
# by temporarily copying it over AutoDAN_config.yaml; the original config is restored afterwards.
# Steps 1.5-3 follow the standard HarmBench pipeline (scripts/run_pipeline.py).
#===============================================================================

set -e

STEP_ARG=${1:-all}

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv"

# Attack-specific settings
ATTACK_METHOD="AutoDAN"
BASE_SAVE_DIR="${HARMBENCH_ROOT}/results_jbb"
BASE_LOG_DIR="${HARMBENCH_ROOT}/slurm_logs_jbb"

# Config files
ORIGINAL_CONFIG="${HARMBENCH_ROOT}/configs/method_configs/AutoDAN_config.yaml"
JBB_CONFIG="${HARMBENCH_ROOT}/configs/method_configs/AutoDAN_config_jbb.yaml"
BACKUP_CONFIG="${HARMBENCH_ROOT}/configs/method_configs/AutoDAN_config.yaml.backup"

# Partitions
PARTITION_STEP1="training"
PARTITION="a30"

# All 20 model configurations
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

echo "========================================================================"
echo "AutoDAN attack on JBB-100"
echo "========================================================================"
echo "Step: ${STEP_ARG}"
echo ""

# Load environment from .env
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
    echo "✓ Loaded environment from .env"
fi

# Verify JBB behaviors and targets exist
if [[ ! -f "$JBB_BEHAVIORS" ]]; then
    echo "ERROR: JBB behaviors CSV not found: $JBB_BEHAVIORS"
    exit 1
fi

if [[ ! -f "$JBB_CONFIG" ]]; then
    echo "ERROR: JBB AutoDAN config not found: $JBB_CONFIG"
    exit 1
fi

echo "✓ JBB behaviors CSV found"
echo "✓ JBB AutoDAN config found"

# Create directories
mkdir -p "$BASE_SAVE_DIR/AutoDAN"
mkdir -p "$BASE_LOG_DIR"

# Change to HarmBench directory
cd "$HARMBENCH_ROOT"

# Function to swap config (for Step 1 only)
swap_config() {
    echo ""
    echo "→ Swapping AutoDAN config to use JBB targets..."
    if [[ -f "$BACKUP_CONFIG" ]]; then
        echo "WARNING: Backup config already exists, skipping backup"
    else
        cp "$ORIGINAL_CONFIG" "$BACKUP_CONFIG"
    fi
    cp "$JBB_CONFIG" "$ORIGINAL_CONFIG"
    echo "✓ Config swapped"
}

# Function to restore config
restore_config() {
    echo ""
    echo "→ Restoring original AutoDAN config..."
    if [[ -f "$BACKUP_CONFIG" ]]; then
        mv "$BACKUP_CONFIG" "$ORIGINAL_CONFIG"
        echo "✓ Config restored"
    else
        echo "WARNING: No backup config found"
    fi
}

# Trap to ensure config is restored on exit
trap restore_config EXIT

# Create model list string
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

# Load environment
echo ""
echo "Loading HarmBench environment..."
module purge
module load external
module load anaconda
conda activate harmbench_env

# Function to run a step
run_step() {
    local STEP=$1
    
    echo ""
    echo "========================================"
    echo "Running AutoDAN Step $STEP"
    echo "========================================"
    
    case "$STEP" in
        "1")
            # Swap config for Step 1
            swap_config
            
            # Step 1: Generate test cases (requires 2 GPUs)
            echo "Generating evolved jailbreak prompts with AutoDAN..."
            python scripts/run_pipeline.py \
                --methods "$ATTACK_METHOD" \
                --models "$MODEL_LIST" \
                --behaviors_path "$JBB_BEHAVIORS" \
                --base_save_dir "$BASE_SAVE_DIR" \
                --base_log_dir "$BASE_LOG_DIR" \
                --step "1" \
                --mode "slurm" \
                --partition "$PARTITION_STEP1" \
                --overwrite
            
            # Restore config after submission
            restore_config
            trap - EXIT  # Remove trap since we manually restored
            ;;
        "1.5")
            # Step 1.5: Merge test cases (runs locally, no config needed)
            echo "Merging test cases..."
            python scripts/run_pipeline.py \
                --methods "$ATTACK_METHOD" \
                --models "$MODEL_LIST" \
                --behaviors_path "$JBB_BEHAVIORS" \
                --base_save_dir "$BASE_SAVE_DIR" \
                --base_log_dir "$BASE_LOG_DIR" \
                --step "1.5" \
                --mode "local"
            ;;
        "2")
            # Step 2: Generate completions (no config needed)
            echo "Generating completions from target models..."
            python scripts/run_pipeline.py \
                --methods "$ATTACK_METHOD" \
                --models "$MODEL_LIST" \
                --behaviors_path "$JBB_BEHAVIORS" \
                --base_save_dir "$BASE_SAVE_DIR" \
                --base_log_dir "$BASE_LOG_DIR" \
                --step "2" \
                --mode "slurm" \
                --partition "$PARTITION" \
                --max_new_tokens 512
            ;;
        "3")
            # Step 3: Evaluate with Llama-2 classifier (no config needed)
            echo "Evaluating with Llama-2 classifier..."
            python scripts/run_pipeline.py \
                --methods "$ATTACK_METHOD" \
                --models "$MODEL_LIST" \
                --behaviors_path "$JBB_BEHAVIORS" \
                --base_save_dir "$BASE_SAVE_DIR" \
                --base_log_dir "$BASE_LOG_DIR" \
                --step "3" \
                --mode "slurm" \
                --partition "$PARTITION" \
                --cls_path "cais/HarmBench-Llama-2-13b-cls"
            ;;
        *)
            echo "ERROR: Unknown step: $STEP"
            exit 1
            ;;
    esac
    
    echo "✓ AutoDAN Step $STEP submitted"
}

# Run steps based on argument
if [[ "$STEP_ARG" == "all" ]]; then
    echo ""
    echo "Running all steps sequentially..."
    echo "NOTE: You must wait for each step to complete before the next can run!"
    echo ""
    
    for STEP in "1" "1.5" "2" "3"; do
        run_step "$STEP"
        
        if [[ "$STEP" != "3" && "$STEP" != "1.5" ]]; then
            echo ""
            echo "⏳ Wait for Step $STEP jobs to complete before proceeding."
            echo "   Monitor: squeue -u \$USER | grep AutoDAN"
            echo ""
            read -p "Press Enter when Step $STEP is complete to continue..."
        fi
    done
else
    run_step "$STEP_ARG"
fi

echo ""
echo "========================================================================"
echo "AutoDAN on JBB - Complete"
echo "========================================================================"
