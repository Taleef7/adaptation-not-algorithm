#!/bin/bash
#===============================================================================
# Original grid: HarmBench attacks (PAIR, ArtPrompt, AutoDAN) on the JBB-100 behaviors
#
# This script runs HarmBench attacks on JBB (100 behaviors) dataset.
# Each attack follows the HarmBench 3-step pipeline:
#   Step 1: Generate test cases (jailbreak prompts)
#   Step 1.5: Merge test cases (if needed)
#   Step 2: Generate completions from target models
#   Step 3: Evaluate with Llama-2 classifier
#
# Usage:
#   bash run_harmbench_attacks_jbb.sh [ATTACK] [STEP]
#
# Arguments:
#   ATTACK: PAIR, ArtPrompt, AutoDAN, or all (default: all)
#   STEP: 1, 1.5, 2, 3, or all (default: all)
#
# Examples:
#   bash run_harmbench_attacks_jbb.sh all all      # Run all attacks, all steps
#   bash run_harmbench_attacks_jbb.sh PAIR 1       # Run PAIR Step 1 only
#   bash run_harmbench_attacks_jbb.sh AutoDAN 2    # Run AutoDAN Step 2 only
#===============================================================================

set -e

# Arguments
ATTACK_ARG=${1:-all}
STEP_ARG=${2:-all}

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv"

# All 25 model configurations
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
    "qwen25_14b_base_fp16_custom"
    "qwen25_14b_base_4bit_custom"
    "qwen25_14b_lora_custom"
    "qwen25_14b_qlora_custom"
    "qwen25_14b_fft_custom"
    "phi4_base_fp16_custom"
    "phi4_base_4bit_custom"
    "phi4_lora_custom"
    "phi4_qlora_custom"
    "phi4_fft_custom"
)

# Attack list
if [[ "$ATTACK_ARG" == "all" ]]; then
    ATTACKS=("PAIR" "ArtPrompt" "AutoDAN")
else
    ATTACKS=("$ATTACK_ARG")
fi

# Step list
if [[ "$STEP_ARG" == "all" ]]; then
    STEPS=("1" "1.5" "2" "3")
else
    STEPS=("$STEP_ARG")
fi

echo "========================================================================"
echo "HarmBench attacks on JBB-100"
echo "========================================================================"
echo "Attacks: ${ATTACKS[@]}"
echo "Steps: ${STEPS[@]}"
echo "Models: ${#MODELS[@]} configurations"
echo "Behaviors: ${JBB_BEHAVIORS}"
echo ""

# Verify JBB behaviors CSV exists
if [[ ! -f "$JBB_BEHAVIORS" ]]; then
    echo "ERROR: JBB behaviors CSV not found: $JBB_BEHAVIORS"
    echo "Run: python evaluation/original_grid/jbb/convert_jbb_to_harmbench_format.py"
    exit 1
fi

echo "✓ JBB behaviors CSV found ($(wc -l < "$JBB_BEHAVIORS") lines)"
echo ""

# Change to HarmBench directory
cd "$HARMBENCH_ROOT"

# Create model list string for run_pipeline.py
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

# Function to run a step
run_step() {
    local ATTACK=$1
    local STEP=$2
    local SAVE_DIR="results_jbb/${ATTACK}"
    local LOG_DIR="slurm_logs_jbb/${ATTACK}"
    
    mkdir -p "$SAVE_DIR"
    mkdir -p "$LOG_DIR"
    
    echo "----------------------------------------"
    echo "Running $ATTACK Step $STEP"
    echo "Save dir: $SAVE_DIR"
    echo "Log dir: $LOG_DIR"
    echo "----------------------------------------"
    
    if [[ "$STEP" == "1.5" ]]; then
        # Step 1.5 runs locally (merge test cases)
        python scripts/run_pipeline.py \
            --methods "$ATTACK" \
            --models "$MODEL_LIST" \
            --behaviors_path "$JBB_BEHAVIORS" \
            --base_save_dir "$SAVE_DIR" \
            --base_log_dir "$LOG_DIR" \
            --step "1.5" \
            --mode "local"
    elif [[ "$STEP" == "3" ]]; then
        # Step 3 uses Llama-2 classifier
        python scripts/run_pipeline.py \
            --methods "$ATTACK" \
            --models "$MODEL_LIST" \
            --behaviors_path "$JBB_BEHAVIORS" \
            --base_save_dir "$SAVE_DIR" \
            --base_log_dir "$LOG_DIR" \
            --step "3" \
            --mode "slurm" \
            --partition "a30" \
            --cls_path "cais/HarmBench-Llama-2-13b-cls"
    else
        # Steps 1 and 2 use SLURM
        python scripts/run_pipeline.py \
            --methods "$ATTACK" \
            --models "$MODEL_LIST" \
            --behaviors_path "$JBB_BEHAVIORS" \
            --base_save_dir "$SAVE_DIR" \
            --base_log_dir "$LOG_DIR" \
            --step "$STEP" \
            --mode "slurm" \
            --partition "a30" \
            --overwrite
    fi
    
    echo "✓ $ATTACK Step $STEP submitted"
    echo ""
}

# Activate HarmBench environment
echo "Loading HarmBench environment..."
module purge
module load external
module load anaconda
conda activate harmbench_env

# Run each attack and step
for ATTACK in "${ATTACKS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "Processing: $ATTACK"
    echo "========================================================================"
    
    for STEP in "${STEPS[@]}"; do
        run_step "$ATTACK" "$STEP"
        
        # Add delay between job submissions
        sleep 2
    done
done

echo ""
echo "========================================================================"
echo "Jobs submitted"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "Check results:"
for ATTACK in "${ATTACKS[@]}"; do
    echo "  ls results_jbb/${ATTACK}/*/test_cases/"
    echo "  ls results_jbb/${ATTACK}/*/completions/"
    echo "  ls results_jbb/${ATTACK}/*/results/"
done
echo ""
echo "Note: Wait for each step to complete before running the next step!"
echo "  Step 1: Generates test cases (jailbreak prompts)"
echo "  Step 1.5: Merges test cases (run locally after Step 1)"
echo "  Step 2: Generates completions from models"
echo "  Step 3: Evaluates with Llama-2 classifier"
echo ""
