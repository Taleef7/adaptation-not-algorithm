#!/bin/bash
# Submit Llama Guard evaluations for Benign dataset (DirectRequest)
# Calculates False Positive Rate (FPR)

PROJECT_ROOT="${PROJECT_ROOT}"
HARMBENCH_ROOT="${PROJECT_ROOT}/toolkits/HarmBench"
BENIGN_BEHAVIORS="${HARMBENCH_ROOT}/data/behavior_datasets/data/benign-behaviors.csv"
MODEL_PATH="${PROJECT_ROOT}/models/llama-guard-3-8b"
BENIGN_RESULTS_DIR="${HARMBENCH_ROOT}/results/Benign/DirectRequest/default/results"

cd "$HARMBENCH_ROOT"

# Find all JSON files in the benign results directory
# Each file represents a model
submitted=0

echo "=============================================="
echo "Submitting Llama Guard evals for Benign Dataset"
echo "Results Dir: $BENIGN_RESULTS_DIR"
echo "=============================================="

for completion_file in ${BENIGN_RESULTS_DIR}/*.json; do
    model_file=$(basename "$completion_file")
    model="${model_file%.json}"
    
    # Define save path: results/Benign/DirectRequest/default/results_llamaguard/{model}.json
    SAVE_DIR="${HARMBENCH_ROOT}/results/Benign/DirectRequest/default/results_llamaguard"
    SAVE_PATH="${SAVE_DIR}/${model}.json"
    LOG_DIR="${HARMBENCH_ROOT}/slurm_logs/evaluate_benign_llamaguard/${model}"
    
    mkdir -p "$SAVE_DIR" "$LOG_DIR"
    rm -f "${LOG_DIR}/${model}.log"
    
    WRAP_CMD="module load external anaconda && conda activate harmbench_env && python evaluate_benign_llamaguard.py --behaviors_path '$BENIGN_BEHAVIORS' --completions_path '$completion_file' --save_path '$SAVE_PATH' --model_path '$MODEL_PATH' --num_tokens 512"
    
    # Use v100 partition
    output=$(sbatch --partition=v100 --account=YOUR_ACCOUNT --qos=standby --nodes=1 --gres=gpu:1 --mem=32G --time=1:00:00 --job-name="LG-Benign-${model:0:6}" --output="${LOG_DIR}/${model}.log" --error="${LOG_DIR}/${model}.log" --wrap="$WRAP_CMD" 2>&1)
    
    if [[ $output =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
        : $((submitted++))
        echo "  [$submitted] ${model} (ID: ${BASH_REMATCH[1]})"
    else
        echo "  [ERROR] ${model}: $output"
    fi
done

echo ""
echo "=============================================="
echo "Total submitted: $submitted jobs"
echo "=============================================="
