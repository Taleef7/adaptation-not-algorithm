#!/bin/bash
#SBATCH --job-name=orbe_4bit_regen
#SBATCH --output=logs/or_bench/%x_%j.log
#SBATCH --partition=a30
#SBATCH --qos=standby
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00

# GPT-4o-mini judging (OR-Bench three-way prompt) of the Base 4-bit completions in the regen4bit
# namespace. Same as or_bench_family_evaluate.slurm, one model per submission.

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

FAMILY="${FAMILY:?set FAMILY}"
SUBSET_NAME="${SUBSET_NAME:?set SUBSET_NAME}"
JUDGE_MODEL="${OR_BENCH_JUDGE_MODEL:-gpt-4o-mini}"
MODEL_NAME="${FAMILY}_base_4bit_custom"

case "$SUBSET_NAME" in
  "Hard-1K")
    PROMPT_TYPE="safe"
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_hard_1k.csv"
    NS_DIR="$HARMBENCH_DIR/results/ORBench/Hard-1K/DirectRequest/regen4bit"
    ;;
  "Toxic")
    PROMPT_TYPE="toxic"
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_toxic.csv"
    NS_DIR="$HARMBENCH_DIR/results/ORBench/Toxic/DirectRequest/regen4bit"
    ;;
  *)
    echo "Unknown SUBSET_NAME: $SUBSET_NAME"; exit 1 ;;
esac

COMPLETIONS_PATH="$NS_DIR/completions/${MODEL_NAME}.json"
SAVE_PATH="$NS_DIR/results/${MODEL_NAME}.json"

if [ ! -f "$COMPLETIONS_PATH" ]; then
  echo "Missing completions file: $COMPLETIONS_PATH"
  exit 1
fi

mkdir -p "$NS_DIR/results"

echo "================================================================"
echo "OR-Bench base_4bit regen EVALUATION"
echo "Family: $FAMILY  Subset: $SUBSET_NAME  Model: $MODEL_NAME  Judge: $JUDGE_MODEL"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "================================================================"

python evaluate_or_bench.py \
  --behaviors_path "$BEHAVIORS_PATH" \
  --completions_path "$COMPLETIONS_PATH" \
  --save_path "$SAVE_PATH" \
  --subset_name "$SUBSET_NAME" \
  --prompt_type "$PROMPT_TYPE" \
  --judge_model "$JUDGE_MODEL" \
  --incremental_update

echo "Completed base_4bit regen evaluation for $FAMILY on $SUBSET_NAME"
