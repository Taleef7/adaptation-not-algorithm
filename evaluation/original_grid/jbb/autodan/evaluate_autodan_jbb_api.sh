#!/bin/bash
#===============================================================================
# Evaluate AutoDAN JBB Completions with GPT-4o-mini API Judge
#
# This script evaluates the completions from AutoDAN attack on JBB dataset
# using GPT-4o-mini as the secondary judge (primary is Llama-2).
#
# Results Structure:
#   Primary (Llama-2):   results_jbb/AutoDAN/{model}/results/results.json
#   Secondary (GPT-4o):  results_jbb/AutoDAN/{model}/results_api/results.json
#
# Usage:
#   bash evaluate_autodan_jbb_api.sh [check|submit|all]
#
#   check  - Check completion status
#   submit - Submit SLURM jobs for evaluation
#   all    - Check status then submit missing evaluations
#===============================================================================

ACTION=${1:-all}

# Configuration
PROJECT_ROOT="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/jbb_behaviors.csv"

# Directory paths
COMPLETIONS_BASE="${HARMBENCH_ROOT}/results_jbb/AutoDAN"
LOGS_BASE="${HARMBENCH_ROOT}/slurm_logs_jbb/evaluate_completions_api/AutoDAN"

# SLURM settings
PARTITION="a30"  # Use a30 partition with gres for API calls
QOS="standby"
ACCOUNT="YOUR_ACCOUNT"

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
echo "AutoDAN JBB - GPT-4o-mini API Evaluation"
echo "========================================================================"
echo "Action: ${ACTION}"
echo "Models: ${#MODELS[@]} configurations"
echo "Behaviors: ${JBB_BEHAVIORS}"
echo ""

# Function to check completion status
check_status() {
    echo "Checking completion status..."
    local completed=0
    local missing=0
    
    for model in "${MODELS[@]}"; do
        local result_file="${COMPLETIONS_BASE}/${model}/results_api/results.json"
        if [[ -f "$result_file" ]]; then
            echo "✓ $model"
            ((completed++))
        else
            echo "✗ $model - MISSING"
            ((missing++))
        fi
    done
    
    echo ""
    echo "Status: $completed/$((completed + missing)) completed"
    echo ""
    
    return $missing
}

# Function to submit evaluation jobs
submit_jobs() {
    echo "Submitting GPT-4o-mini evaluation jobs..."
    local submitted=0
    
    # Load OpenAI API key
    if [[ -f "${PROJECT_ROOT}/.env" ]]; then
        export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
        echo "✓ Loaded OPENAI_API_KEY from .env"
    else
        echo "ERROR: .env file not found at ${PROJECT_ROOT}/.env"
        exit 1
    fi
    
    cd "${HARMBENCH_ROOT}"
    
    for model in "${MODELS[@]}"; do
        local completions_path="${COMPLETIONS_BASE}/${model}/completions/completions.json"
        local result_file="${COMPLETIONS_BASE}/${model}/results_api/results.json"
        local log_dir="${LOGS_BASE}/${model}"
        local log_file="${log_dir}/${model}.log"
        
        # Existing results are overwritten
        if [[ -f "$result_file" ]]; then
            echo "Overwriting existing results for $model"
        fi
        
        # Check if completions exist
        if [[ ! -f "$completions_path" ]]; then
            echo "⚠️  Skipping $model (completions not found)"
            continue
        fi
        
        # Create directories
        mkdir -p "${COMPLETIONS_BASE}/${model}/results_api"
        mkdir -p "$log_dir"
        
        # Submit SLURM job
        sbatch \
            --partition="$PARTITION" \
            --account="$ACCOUNT" \
            --qos="$QOS" \
            --job-name="eval_api_autodan_jbb_${model}" \
            --nodes=1 \
            --gres=gpu:1 \
            --mem=8G \
            --time=1:00:00 \
            --output="$log_file" \
            --wrap="module load external anaconda && conda activate harmbench_env && python evaluate_attacks_api.py \
                --behaviors_path '$JBB_BEHAVIORS' \
                --completions_path '$completions_path' \
                --save_path '$result_file' \
                --num_tokens 512"
        
        echo "✓ Submitted: $model"
        ((submitted++))
    done
    
    echo ""
    echo "Submitted $submitted evaluation jobs"
    echo ""
    echo "Monitor progress:"
    echo "  squeue -u \$USER | grep eval_api_autodan"
    echo ""
    echo "Check logs:"
    echo "  ls -la ${LOGS_BASE}/"
}

# Main execution
case "$ACTION" in
    "check")
        check_status
        ;;
    "submit")
        submit_jobs
        ;;
    "all")
        check_status
        missing_count=$?
        if [[ $missing_count -gt 0 ]]; then
            echo "Submitting missing evaluations..."
            submit_jobs
        else
            echo "All evaluations already completed!"
        fi
        ;;
    *)
        echo "ERROR: Unknown action: $ACTION"
        echo "Usage: $0 [check|submit|all]"
        exit 1
        ;;
esac

echo "========================================================================"
echo "Done"
echo "========================================================================"
