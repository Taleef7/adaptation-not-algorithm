#!/bin/bash
#SBATCH --job-name=orr_llama2
#SBATCH --output=logs/evaluate_completions/Benign_Llama2/DirectRequest/default/%x_%A_%a.log
#SBATCH --partition=training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=training
#SBATCH --array=0-19

module load anaconda/2025.06-py313
source activate harmbench_env

# Model configurations
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
    "phi4_base_fp16_custom"
    "phi4_base_4bit_custom"
    "phi4_lora_custom"
    "phi4_qlora_custom"
    "phi4_fft_custom"
    "qwen3_base_fp16_custom"
    "qwen3_base_4bit_custom"
    "qwen3_lora_custom"
    "qwen3_qlora_custom"
    "qwen3_fft_custom"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "================================================================"
echo "Evaluating ORR with Llama-2 for: $MODEL"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================================================"

cd ${PROJECT_ROOT}/toolkits/HarmBench

# Paths
BEHAVIORS="./data/behavior_datasets/data/benign-behaviors.csv"
GPT_RESULTS="./results/Benign/DirectRequest/default/results/${MODEL}.json"
RESULTS_DIR="./results/Benign/DirectRequest/default/results_llama2"
SAVE_PATH="${RESULTS_DIR}/${MODEL}.json"

# Check if GPT-4o-mini results exist (which contain the completions)
if [ ! -f "$GPT_RESULTS" ]; then
    echo "ERROR: GPT-4o-mini results file not found: $GPT_RESULTS"
    exit 1
fi

echo ""
echo "Input files:"
echo "  Behaviors: $BEHAVIORS"
echo "  GPT-4o-mini Results (contains completions): $GPT_RESULTS"
echo "  Output: $SAVE_PATH"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Run evaluation
echo "Running Llama-2 ORR evaluation..."
echo "================================================================"
python evaluate_benign_llama2.py \
    --behaviors_path "$BEHAVIORS" \
    --completions_path "$GPT_RESULTS" \
    --save_path "$SAVE_PATH" \
    --cls_path "cais/HarmBench-Llama-2-13b-cls" \
    --device "cuda:0" \
    --num_tokens 512 \
    --clip_tokenizer_path "gpt2"

EXIT_CODE=$?

echo "================================================================"
echo "Evaluation complete for: $MODEL"
echo "Exit code: $EXIT_CODE"
echo "Results saved to: $SAVE_PATH"
echo "End Time: $(date)"
echo "================================================================"

exit $EXIT_CODE
