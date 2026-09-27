#!/bin/bash
#SBATCH --job-name=orbg_family
#SBATCH --output=logs/or_bench/%x_%A_%a.log
#SBATCH --partition=a10
#SBATCH --qos=standby
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-4

set -eo pipefail

module --force purge
module load external
module load anaconda/2025.06-py313
source activate harmbench_env
set -u

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_DIR="$ROOT_DIR/toolkits/HarmBench"
cd "$HARMBENCH_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

FAMILY="${FAMILY:-}"
SUBSET_NAME="${SUBSET_NAME:-Hard-1K}"

case "$FAMILY" in
  "gemma2")
    MODELS=(
      "gemma2_base_fp16_custom"
      "gemma2_base_4bit_custom"
      "gemma2_lora_custom"
      "gemma2_qlora_custom"
      "gemma2_fft_custom"
    )
    ;;
  "llama31")
    MODELS=(
      "llama31_base_fp16_custom"
      "llama31_base_4bit_custom"
      "llama31_lora_custom"
      "llama31_qlora_custom"
      "llama31_fft_custom"
    )
    ;;
  "phi4")
    MODELS=(
      "phi4_base_fp16_custom"
      "phi4_base_4bit_custom"
      "phi4_lora_custom"
      "phi4_qlora_custom"
      "phi4_fft_custom"
    )
    ;;
  "qwen25_14b")
    MODELS=(
      "qwen25_14b_base_fp16_custom"
      "qwen25_14b_base_4bit_custom"
      "qwen25_14b_lora_custom"
      "qwen25_14b_qlora_custom"
      "qwen25_14b_fft_custom"
    )
    ;;
  "qwen3")
    MODELS=(
      "qwen3_base_fp16_custom"
      "qwen3_base_4bit_custom"
      "qwen3_lora_custom"
      "qwen3_qlora_custom"
      "qwen3_fft_custom"
    )
    ;;
  *)
    echo "Unknown FAMILY: $FAMILY"
    exit 1
    ;;
esac

MODEL_NAME="${MODELS[$SLURM_ARRAY_TASK_ID]}"

case "$SUBSET_NAME" in
  "Hard-1K")
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_hard_1k.csv"
    SAVE_DIR="$HARMBENCH_DIR/results/ORBench/Hard-1K/DirectRequest/default/completions"
    ;;
  "Toxic")
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_toxic.csv"
    SAVE_DIR="$HARMBENCH_DIR/results/ORBench/Toxic/DirectRequest/default/completions"
    ;;
  *)
    echo "Unknown SUBSET_NAME: $SUBSET_NAME"
    exit 1
    ;;
esac

mkdir -p "$SAVE_DIR"

echo "================================================================"
echo "OR-Bench generation"
echo "Family: $FAMILY"
echo "Subset: $SUBSET_NAME"
echo "Model: $MODEL_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================================================"

python generate_completions.py \
  --model_name "$MODEL_NAME" \
  --behaviors_path "$BEHAVIORS_PATH" \
  --save_path "$SAVE_DIR/${MODEL_NAME}.json" \
  --max_new_tokens 512 \
  --incremental_update

echo "Completed OR-Bench generation for $MODEL_NAME on $SUBSET_NAME"
