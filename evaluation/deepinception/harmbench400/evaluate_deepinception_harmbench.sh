#!/bin/bash
# HB-Cls (cais/HarmBench-Llama-2-13b-cls) scoring of DeepInception HarmBench-400 completions via
# harmbench/evaluate_deepinception_harmbench.py (run from toolkits/HarmBench). GPT-4o-mini scoring
# uses harmbench/evaluate_deepinception_harmbench_api.py (scripts/run_deepinception_harmbench_eval_api.sh
# in harmbench.patch); LlamaGuard-3 uses scripts/submit_llamaguard_deepinception_hb.sh.
#
# Usage:
#     bash evaluate_deepinception_harmbench.sh           # status
#     bash evaluate_deepinception_harmbench.sh submit    # submit missing evaluations
# Required env: PROJECT_ROOT

set -e

# Model configurations
MODELS=(
    # Gemma-2 9B
    "gemma2_base_fp16_custom"
    "gemma2_base_4bit_custom"
    "gemma2_lora_custom"
    "gemma2_qlora_custom"
    "gemma2_fft_custom"
    
    # Llama 3.1 8B
    "llama31_base_fp16_custom"
    "llama31_base_4bit_custom"
    "llama31_lora_custom"
    "llama31_qlora_custom"
    "llama31_fft_custom"
    
    # Qwen3 4B
    "qwen3_base_fp16_custom"
    "qwen3_base_4bit_custom"
    "qwen3_lora_custom"
    "qwen3_qlora_custom"
    "qwen3_fft_custom"
    
    # Phi-4 14B
    "phi4_base_fp16_custom"
    "phi4_base_4bit_custom"
    "phi4_lora_custom"
    "phi4_qlora_custom"
    "phi4_fft_custom"

    # Qwen2.5 14B
    "qwen25_14b_base_fp16_custom"
    "qwen25_14b_base_4bit_custom"
    "qwen25_14b_lora_custom"
    "qwen25_14b_qlora_custom"
    "qwen25_14b_fft_custom"
)

BEHAVIORS_PATH="toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv"
RESULTS_BASE="toolkits/HarmBench/results/DeepInception_HarmBench"

echo "Evaluating DeepInception-HarmBench with Llama-2 Classifier"
echo "==========================================================="

# Change to project directory
cd "${PROJECT_ROOT:?set PROJECT_ROOT}"

# Create log directory
mkdir -p toolkits/HarmBench/slurm_logs/DeepInception_HarmBench/evaluate

# Generate and submit evaluation jobs
if [[ "$1" == "submit" ]]; then
    echo "Submitting evaluation jobs..."
    echo "=============================="
    
    for model in "${MODELS[@]}"; do
        COMPLETIONS_PATH="${RESULTS_BASE}/${model}/completions/${model}.json"
        SAVE_PATH="${RESULTS_BASE}/${model}/results/${model}.json"
        
        # Check if completions exist
        if [[ ! -f "$COMPLETIONS_PATH" ]]; then
            echo "  ⚠ Skipping $model: completions not found"
            continue
        fi
        
        # Check if already evaluated
        if [[ -f "$SAVE_PATH" ]]; then
            echo "  ✓ Skipping $model: already evaluated"
            continue
        fi
        
        # Submit evaluation job
        JOB_ID=$(sbatch <<EOF | awk '{print $4}'
#!/bin/bash
#SBATCH --job-name=eval_deepinc_${model}
#SBATCH --output=toolkits/HarmBench/slurm_logs/DeepInception_HarmBench/evaluate/${model}_%j.log
#SBATCH --partition=a30
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=2:00:00

echo "========================================"
echo "Evaluating: ${model}"
echo "Job ID: \$SLURM_JOB_ID"
echo "Started: \$(date)"
echo "========================================"

module load anaconda/2025.06-py313
source activate harmbench_env

cd "${PROJECT_ROOT:?set PROJECT_ROOT}"/toolkits/HarmBench

python evaluate_deepinception_harmbench.py --behaviors_path ${PROJECT_ROOT}/${BEHAVIORS_PATH} --completions_path ${PROJECT_ROOT}/${COMPLETIONS_PATH} --save_path ${PROJECT_ROOT}/${SAVE_PATH} --cls_path cais/HarmBench-Llama-2-13b-cls

echo "========================================"
echo "Completed: \$(date)"
echo "========================================"
EOF
)
        
        echo "  $model: Job $JOB_ID"
    done
    
    echo ""
    echo "✓ Submitted evaluation jobs"
    echo "✓ Monitor with: squeue -u \$USER"
    
else
    # Just check status
    echo "Checking evaluation status..."
    echo "=============================="
    
    TOTAL=0
    COMPLETED=0
    PENDING=0
    MISSING=0
    
    for model in "${MODELS[@]}"; do
        TOTAL=$((TOTAL + 1))
        
        COMPLETIONS_PATH="${RESULTS_BASE}/${model}/completions/${model}.json"
        SAVE_PATH="${RESULTS_BASE}/${model}/results/${model}.json"
        
        if [[ ! -f "$COMPLETIONS_PATH" ]]; then
            echo "  ⚠ $model: completions missing"
            MISSING=$((MISSING + 1))
        elif [[ -f "$SAVE_PATH" ]]; then
            echo "  ✓ $model: evaluated"
            COMPLETED=$((COMPLETED + 1))
        else
            echo "  ⏳ $model: pending"
            PENDING=$((PENDING + 1))
        fi
    done
    
    echo ""
    echo "Status Summary:"
    echo "  Total models: $TOTAL"
    echo "  Completed:    $COMPLETED"
    echo "  Pending:      $PENDING"
    echo "  Missing:      $MISSING"
    echo ""
    echo "To submit evaluation jobs, run:"
    echo "  bash evaluation/deepinception/harmbench400/evaluate_deepinception_harmbench.sh submit"
fi
