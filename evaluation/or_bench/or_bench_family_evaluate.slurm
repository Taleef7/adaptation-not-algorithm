#!/bin/bash
#SBATCH --job-name=orbe_family
#SBATCH --output=logs/or_bench/%x_%A_%a.log
#SBATCH --partition=a30
#SBATCH --qos=standby
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
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

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ROOT_DIR/.env"
  set +a
fi

export OR_BENCH_API_MAX_RETRY="${OR_BENCH_API_MAX_RETRY:-60}"
export OR_BENCH_API_RETRY_SLEEP="${OR_BENCH_API_RETRY_SLEEP:-30}"
export OR_BENCH_API_TIMEOUT="${OR_BENCH_API_TIMEOUT:-240}"
export OR_BENCH_API_MAX_SLEEP="${OR_BENCH_API_MAX_SLEEP:-600}"
export OR_BENCH_API_RETRY_JITTER="${OR_BENCH_API_RETRY_JITTER:-2.0}"

FAMILY="${FAMILY:-}"
SUBSET_NAME="${SUBSET_NAME:-Hard-1K}"
JUDGE_MODEL="${OR_BENCH_JUDGE_MODEL:-gpt-4o-mini}"

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
    PROMPT_TYPE="safe"
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_hard_1k.csv"
    COMPLETIONS_DIR="$HARMBENCH_DIR/results/ORBench/Hard-1K/DirectRequest/default/completions"
    RESULTS_DIR="$HARMBENCH_DIR/results/ORBench/Hard-1K/DirectRequest/default/results"
    ;;
  "Toxic")
    PROMPT_TYPE="toxic"
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_toxic.csv"
    COMPLETIONS_DIR="$HARMBENCH_DIR/results/ORBench/Toxic/DirectRequest/default/completions"
    RESULTS_DIR="$HARMBENCH_DIR/results/ORBench/Toxic/DirectRequest/default/results"
    ;;
  *)
    echo "Unknown SUBSET_NAME: $SUBSET_NAME"
    exit 1
    ;;
esac

COMPLETIONS_PATH="$COMPLETIONS_DIR/${MODEL_NAME}.json"
SAVE_PATH="$RESULTS_DIR/${MODEL_NAME}.json"

if [ ! -f "$COMPLETIONS_PATH" ]; then
  echo "Missing completions file: $COMPLETIONS_PATH"
  exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "OR-Bench evaluation"
echo "Family: $FAMILY"
echo "Subset: $SUBSET_NAME"
echo "Prompt type: $PROMPT_TYPE"
echo "Model: $MODEL_NAME"
echo "Judge: $JUDGE_MODEL"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "================================================================"

python evaluate_or_bench.py \
  --behaviors_path "$BEHAVIORS_PATH" \
  --completions_path "$COMPLETIONS_PATH" \
  --save_path "$SAVE_PATH" \
  --subset_name "$SUBSET_NAME" \
  --prompt_type "$PROMPT_TYPE" \
  --judge_model "$JUDGE_MODEL" \
  --incremental_update

echo "Completed OR-Bench evaluation for $MODEL_NAME on $SUBSET_NAME"
