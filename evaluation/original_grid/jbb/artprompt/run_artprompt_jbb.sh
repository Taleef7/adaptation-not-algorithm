#!/bin/bash
#===============================================================================
# ArtPrompt Attack on JBB Dataset (100 behaviors)
#
# ArtPrompt uses ASCII art representations to bypass LLM safety filters.
# It converts sensitive words into ASCII art that models can still interpret.
#
# Key Characteristics:
#   - No GPU required for test case generation (uses pre-defined ASCII patterns)
#   - Works with both open and closed source models
#   - behavior_chunk_size: 5
#
# HarmBench Pipeline Steps:
#   Step 1: Generate test cases (ASCII art jailbreak prompts)
#   Step 1.5: Merge test cases (if needed)
#   Step 2: Generate completions from target models
#   Step 3: Evaluate with Llama-2 classifier
#
# Usage:
#   bash run_artprompt_jbb.sh [STEP]
#   
#   STEP: 1, 1.5, 2, 3, or all (default: all)
#
# Directory Structure:
#   Results: toolkits/HarmBench/results/ArtPrompt_JBB/{model}/
#   Logs: toolkits/HarmBench/slurm_logs/ArtPrompt_JBB/
#===============================================================================

set -e

STEP_ARG=${1:-all}

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv"

# Attack-specific settings
ATTACK_METHOD="ArtPrompt"

# JBB results use separate base directories from the HarmBench-400 results
# Existing results: results/ArtPrompt/{model}/ (harmbench dataset)
# JBB results:      results_jbb/ArtPrompt/{model}/ (jbb dataset)
BASE_SAVE_DIR="${HARMBENCH_ROOT}/results_jbb"
BASE_LOG_DIR="${HARMBENCH_ROOT}/slurm_logs_jbb"

# Convenience shortcuts for checking results
RESULTS_DIR="${BASE_SAVE_DIR}/ArtPrompt"
LOG_DIR="${BASE_LOG_DIR}"

# ArtPrompt config: 0 GPU for step 1, 1 GPU for steps 2-3
# Using a30 partition as it has good availability
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
echo "ArtPrompt Attack on JBB Dataset"
echo "========================================================================"
echo "Step: ${STEP_ARG}"
echo "Models: ${#MODELS[@]} configurations"
echo "Behaviors: ${JBB_BEHAVIORS}"
echo "Base Save Dir: ${BASE_SAVE_DIR}"
echo "Base Log Dir: ${BASE_LOG_DIR}"
echo "Results will be in: ${RESULTS_DIR}/{model}/"
echo ""
echo "NOTE: JBB results are SEPARATE from existing HarmBench results in results/"
echo ""

# Load OpenAI API key from .env
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
    echo "✓ Loaded OPENAI_API_KEY from .env"
else
    echo "WARNING: .env file not found. ArtPrompt Step 1 requires OPENAI_API_KEY."
fi

# Verify JBB behaviors CSV exists
if [[ ! -f "$JBB_BEHAVIORS" ]]; then
    echo "ERROR: JBB behaviors CSV not found: $JBB_BEHAVIORS"
    echo "Run: python evaluation/original_grid/jbb/convert_jbb_to_harmbench_format.py"
    exit 1
fi

echo "✓ JBB behaviors CSV found ($(wc -l < "$JBB_BEHAVIORS") lines)"

# Create directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

# Change to HarmBench directory
cd "$HARMBENCH_ROOT"

# Create model list string
MODEL_LIST=$(IFS=,; echo "${MODELS[*]}")

# Load environment
echo ""
echo "Loading HarmBench environment..."
module purge
module load external
module load anaconda
conda activate harmbench_env

# Verify environment
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found. Is harmbench_env activated?"
    exit 1
fi

# Function to run a step
run_step() {
    local STEP=$1
    
    echo ""
    echo "========================================"
    echo "Running ArtPrompt Step $STEP"
    echo "========================================"
    
    case "$STEP" in
        "1")
            # Step 1: Generate test cases (can run locally since no GPU needed)
            echo "Generating ASCII art jailbreak prompts..."
            echo "Note: ArtPrompt Step 1 doesn't require GPU, running locally..."
            python scripts/run_pipeline.py \
                --methods "$ATTACK_METHOD" \
                --models "$MODEL_LIST" \
                --behaviors_path "$JBB_BEHAVIORS" \
                --base_save_dir "$BASE_SAVE_DIR" \
                --base_log_dir "$BASE_LOG_DIR" \
                --step "1" \
                --mode "local" \
                --overwrite
            ;;
        "1.5")
            # Step 1.5: Merge test cases (runs locally)
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
            # Step 2: Generate completions (requires GPU)
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
            # Step 3: Evaluate with Llama-2 classifier
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
    
    echo "✓ ArtPrompt Step $STEP completed/submitted"
}

# Run steps based on argument
if [[ "$STEP_ARG" == "all" ]]; then
    echo ""
    echo "Running all steps..."
    echo "NOTE: Steps 1 and 1.5 run locally. Steps 2-3 require waiting for SLURM jobs."
    echo ""
    
    # Steps 1 and 1.5 can run immediately (no GPU needed)
    run_step "1"
    run_step "1.5"
    
    # Step 2 requires SLURM
    run_step "2"
    echo ""
    echo "⏳ Wait for Step 2 jobs to complete before running Step 3."
    echo "   Monitor: squeue -u \$USER | grep ArtPrompt"
    echo ""
    read -p "Press Enter when Step 2 is complete to continue..."
    
    run_step "3"
else
    run_step "$STEP_ARG"
fi

echo ""
echo "========================================================================"
echo "ArtPrompt on JBB - Jobs Submitted"
echo "========================================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER | grep ArtPrompt"
echo ""
echo "Check logs:"
echo "  ls -la ${LOG_DIR}/"
echo ""
echo "Check results:"
echo "  ls ${RESULTS_DIR}/*/test_cases/"
echo "  ls ${RESULTS_DIR}/*/completions/"
echo "  ls ${RESULTS_DIR}/*/results/"
echo ""
