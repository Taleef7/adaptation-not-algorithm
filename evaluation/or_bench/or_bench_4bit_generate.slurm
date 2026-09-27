#!/bin/bash
#SBATCH --job-name=orbg_4bit_regen
#SBATCH --output=logs/or_bench/%x_%j.log
#SBATCH --partition=a10
#SBATCH --qos=standby
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --time=04:00:00

# OR-Bench completions for the Base 4-bit cells. These are generated through the vLLM path
# (--generate_with_vllm), which applies the models.yaml `quantization: bitsandbytes` setting;
# HarmBench's HF path (baselines/model_utils.py load_model_and_tokenizer) does not use that key.
# The attack completions are likewise generated with --generate_with_vllm.
# Output goes to the regen4bit results namespace (DirectRequest/regen4bit/).

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

FAMILY="${FAMILY:?set FAMILY}"
SUBSET_NAME="${SUBSET_NAME:?set SUBSET_NAME}"
MODEL_NAME="${FAMILY}_base_4bit_custom"

case "$SUBSET_NAME" in
  "Hard-1K")
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_hard_1k.csv"
    SAVE_DIR="$HARMBENCH_DIR/results/ORBench/Hard-1K/DirectRequest/regen4bit/completions"
    ;;
  "Toxic")
    BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/or_bench/or_bench_toxic.csv"
    SAVE_DIR="$HARMBENCH_DIR/results/ORBench/Toxic/DirectRequest/regen4bit/completions"
    ;;
  *)
    echo "Unknown SUBSET_NAME: $SUBSET_NAME"; exit 1 ;;
esac

SAVE_PATH="$SAVE_DIR/${MODEL_NAME}.json"
FP16_COMPLETIONS="$HARMBENCH_DIR/results/ORBench/$SUBSET_NAME/DirectRequest/default/completions/${FAMILY}_base_fp16_custom.json"

mkdir -p "$SAVE_DIR"

echo "================================================================"
echo "OR-Bench base_4bit generation (vLLM, bitsandbytes 4-bit)"
echo "Family: $FAMILY  Subset: $SUBSET_NAME  Model: $MODEL_NAME"
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  Start: $(date)"
echo "================================================================"

# models_4bit_regen.yaml is models.yaml without `use_vllm: false` on the phi4 and qwen25_14b
# base_4bit entries (that setting would route those two models to the HF path).
python generate_completions.py \
  --model_name "$MODEL_NAME" \
  --models_config_file ./configs/model_configs/models_4bit_regen.yaml \
  --behaviors_path "$BEHAVIORS_PATH" \
  --save_path "$SAVE_PATH" \
  --max_new_tokens 512 \
  --generate_with_vllm \
  --incremental_update

# Check that the 4-bit output differs from the fp16 output (i.e. quantization was applied).
if [ -f "$FP16_COMPLETIONS" ]; then
  a=$(md5sum "$SAVE_PATH" | cut -d' ' -f1)
  b=$(md5sum "$FP16_COMPLETIONS" | cut -d' ' -f1)
  if [ "$a" = "$b" ]; then
    echo "ERROR: base_4bit output is byte-identical to base_fp16 (md5 $a); quantization not applied."
    exit 1
  fi
  echo "Check: base_4bit output differs from base_fp16 (md5 $a vs $b). OK."
fi

echo "Completed base_4bit generation for $FAMILY on $SUBSET_NAME"
