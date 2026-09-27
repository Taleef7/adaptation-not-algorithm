#!/bin/bash

# Script to run benign over-refusal evaluation for Qwen3 and Phi-4 models
# This follows the same pattern used for Gemma-2 and Llama 3.1

set -e

# Models to evaluate
QWEN3_MODELS=(
    "qwen3_base_fp16_custom"
    "qwen3_base_4bit_custom"
    "qwen3_lora_custom"
    "qwen3_qlora_custom"
    "qwen3_fft_custom"
)

PHI4_MODELS=(
    "phi4_base_fp16_custom"
    "phi4_base_4bit_custom"
    "phi4_lora_custom"
    "phi4_qlora_custom"
    "phi4_fft_custom"
)

# Combine all models
ALL_MODELS=("${QWEN3_MODELS[@]}" "${PHI4_MODELS[@]}")

# Paths
BENIGN_BEHAVIORS="./data/behavior_datasets/data/benign-behaviors.csv"
METHOD="DirectRequest"
VARIANT="default"

# Base directories (matches existing structure for gemma2/llama31)
COMPLETIONS_BASE="./results/Benign_Temp/${METHOD}/${VARIANT}/completions"
RESULTS_BASE="./results/Benign/${METHOD}/${VARIANT}/results"
COMPLETIONS_LOG_BASE="./slurm_logs/generate_completions/Benign/${METHOD}/${VARIANT}"
EVALUATION_LOG_BASE="./slurm_logs/evaluate_completions/Benign/${METHOD}/${VARIANT}"

# Create directories
mkdir -p "${COMPLETIONS_BASE}"
mkdir -p "${RESULTS_BASE}"
mkdir -p "${COMPLETIONS_LOG_BASE}"
mkdir -p "${EVALUATION_LOG_BASE}"

# Function to submit completion generation job
submit_completion_job() {
    local model=$1
    local completion_path="${COMPLETIONS_BASE}/${model}.json"
    local log_path="${COMPLETIONS_LOG_BASE}/${model}.log"
    
    echo "Submitting completion generation for ${model}..."
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=benign_gen_${model}
#SBATCH --output=${log_path}
#SBATCH --partition=training
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00

echo "Generating benign completions for ${model}"
echo "Model: ${model}"
echo "Behaviors: ${BENIGN_BEHAVIORS}"
echo "Output: ${completion_path}"

module load anaconda/2025.06-py313
source activate harmbench_env

cd ${PROJECT_ROOT}/toolkits/HarmBench

python generate_completions.py \\
    --model_name ${model} \\
    --behaviors_path ${BENIGN_BEHAVIORS} \\
    --save_path ${completion_path} \\
    --max_new_tokens 512

echo "Completion generation finished for ${model}"
EOF
}

# Function to submit evaluation job
submit_evaluation_job() {
    local model=$1
    local completion_path="${COMPLETIONS_BASE}/${model}.json"
    local result_path="${RESULTS_BASE}/${model}.json"
    local log_path="${EVALUATION_LOG_BASE}/${model}_eval_api_%j.log"
    
    echo "Submitting evaluation for ${model}..."
    
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${model}_eval_api
#SBATCH --output=${log_path}
#SBATCH --partition=a30
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=standby
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

echo "Evaluating refusals for ${model}"
echo "Behaviors: ${BENIGN_BEHAVIORS}"
echo "Completions: ${completion_path}"
echo "Results: ${result_path}"

module load anaconda/2025.06-py313
source activate harmbench_env

cd ${PROJECT_ROOT}/toolkits/HarmBench

python -u evaluate_refusals_api.py \\
    --behaviors_path ${BENIGN_BEHAVIORS} \\
    --completions_path ${completion_path} \\
    --save_path ${result_path}

echo "Evaluation finished for ${model}"
EOF
}

# Main execution
case "$1" in
    "generate")
        echo "=== Submitting completion generation jobs for Qwen3 and Phi-4 ==="
        for model in "${ALL_MODELS[@]}"; do
            submit_completion_job "$model"
        done
        echo "All completion generation jobs submitted!"
        echo "Monitor with: squeue -u \$USER"
        ;;
    
    "evaluate")
        echo "=== Submitting evaluation jobs for Qwen3 and Phi-4 ==="
        for model in "${ALL_MODELS[@]}"; do
            completion_file="${COMPLETIONS_BASE}/${model}.json"
            if [ -f "$completion_file" ]; then
                submit_evaluation_job "$model"
            else
                echo "Warning: Completion file not found for ${model}, skipping evaluation"
            fi
        done
        echo "All evaluation jobs submitted!"
        echo "Monitor with: squeue -u \$USER"
        ;;
    
    "calculate")
        echo "=== Calculating Over-Refusal Rates ==="
        cd ${PROJECT_ROOT}/toolkits/HarmBench
        
        # Update calculate_orr.py to include our models
        python3 << 'PYEOF'
import json
import os

results_dir = './results/Benign/DirectRequest/default/results/'
model_configs = [
    'qwen3_base_fp16_custom',
    'qwen3_base_4bit_custom',
    'qwen3_lora_custom',
    'qwen3_qlora_custom',
    'qwen3_fft_custom',
    'phi4_base_fp16_custom',
    'phi4_base_4bit_custom',
    'phi4_lora_custom',
    'phi4_qlora_custom',
    'phi4_fft_custom'
]

over_refusal_rates = {}

print("Calculating Over-Refusal Rates (ORR) for Qwen3 and Phi-4...")
print("=" * 70)

for model_name in model_configs:
    file_path = os.path.join(results_dir, f"{model_name}.json")
    
    if not os.path.exists(file_path):
        print(f"Warning: Results file not found for {model_name}")
        continue
    
    try:
        with open(file_path, 'r') as f:
            results_data = json.load(f)
        
        refusal_count = 0
        helpful_count = 0
        total_behaviors = 0
        
        for behavior_index, results_list in results_data.items():
            if not results_list:
                continue
            
            result_item = results_list[0]
            total_behaviors += 1
            
            # advbench_label: 0 = Refusal, 1 = Helpful
            if result_item.get("advbench_label") == 0:
                refusal_count += 1
            elif result_item.get("advbench_label") == 1:
                helpful_count += 1
        
        if total_behaviors == 0:
            print(f"Warning: No valid results in {file_path}")
            orr = None
        else:
            orr = refusal_count / total_behaviors
            over_refusal_rates[model_name] = orr
            print(f"{model_name:<30} | Total: {total_behaviors:3d} | Refused: {refusal_count:3d} | Helpful: {helpful_count:3d} | ORR: {orr:.4f} ({orr*100:.2f}%)")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

print("\n" + "=" * 70)
print("SUMMARY - Over-Refusal Rates")
print("=" * 70)

# Group by model family
qwen3_models = {k: v for k, v in over_refusal_rates.items() if k.startswith('qwen3')}
phi4_models = {k: v for k, v in over_refusal_rates.items() if k.startswith('phi4')}

if qwen3_models:
    print("\nQwen3 Models:")
    for model_name in sorted(qwen3_models.keys()):
        print(f"  {model_name:<30}: {qwen3_models[model_name]:.4f} ({qwen3_models[model_name]*100:.2f}%)")

if phi4_models:
    print("\nPhi-4 Models:")
    for model_name in sorted(phi4_models.keys()):
        print(f"  {model_name:<30}: {phi4_models[model_name]:.4f} ({phi4_models[model_name]*100:.2f}%)")

print("=" * 70)
PYEOF
        ;;
    
    "check")
        echo "=== Checking completion and result files ==="
        echo ""
        echo "Completion files (temporary):"
        for model in "${ALL_MODELS[@]}"; do
            file="${COMPLETIONS_BASE}/${model}.json"
            if [ -f "$file" ]; then
                size=$(du -h "$file" | cut -f1)
                echo "  ✓ ${model}: ${size}"
            else
                echo "  ✗ ${model}: NOT FOUND"
            fi
        done
        
        echo ""
        echo "Result files (permanent):"
        for model in "${ALL_MODELS[@]}"; do
            file="${RESULTS_BASE}/${model}.json"
            if [ -f "$file" ]; then
                size=$(du -h "$file" | cut -f1)
                echo "  ✓ ${model}: ${size}"
            else
                echo "  ✗ ${model}: NOT FOUND"
            fi
        done
        
        echo ""
        echo "Log files:"
        echo "  Completions: ${COMPLETIONS_LOG_BASE}/"
        if [ -d "${COMPLETIONS_LOG_BASE}" ]; then
            comp_count=$(find "${COMPLETIONS_LOG_BASE}" -name "*.log" 2>/dev/null | wc -l)
            echo "    Found ${comp_count} log files"
        else
            echo "    Directory not found"
        fi
        echo "  Evaluations: ${EVALUATION_LOG_BASE}/"
        if [ -d "${EVALUATION_LOG_BASE}" ]; then
            eval_count=$(find "${EVALUATION_LOG_BASE}" -name "*.log" 2>/dev/null | wc -l)
            echo "    Found ${eval_count} log files"
        else
            echo "    Directory not found"
        fi
        ;;
    
    "cleanup")
        echo "=== Cleaning up temporary completion files ==="
        if [ -d "./results/Benign_Temp" ]; then
            echo "Removing ./results/Benign_Temp/"
            rm -rf ./results/Benign_Temp
            echo "✓ Temporary completions cleaned up (matches gemma2/llama31 pattern)"
        else
            echo "No temporary directory found (already cleaned or never created)"
        fi
        ;;
    
    *)
        echo "Usage: $0 {generate|evaluate|calculate|check|cleanup}"
        echo ""
        echo "Commands:"
        echo "  generate  - Submit jobs to generate completions for benign behaviors"
        echo "  evaluate  - Submit jobs to evaluate completions for over-refusals"
        echo "  calculate - Calculate and display Over-Refusal Rates (ORR)"
        echo "  check     - Check which completion and result files exist"
        echo "  cleanup   - Remove temporary completion files (do after evaluation)"
        echo ""
        echo "Workflow:"
        echo "  1. $0 generate    # Generate completions (takes ~1-2 hours)"
        echo "  2. $0 check       # Verify completions were created"
        echo "  3. $0 evaluate    # Evaluate refusals (takes ~30-60 min)"
        echo "  4. $0 check       # Verify results were created"
        echo "  5. $0 calculate   # Calculate and display ORRs"
        echo "  6. $0 cleanup     # Remove temporary completions (optional)"
        exit 1
        ;;
esac
