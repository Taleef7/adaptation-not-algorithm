#!/bin/bash
#SBATCH --job-name=merge_rank_layer
#SBATCH --output=logs/merge_rank_layer_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=90G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby

# Merge step for the Llama-3.1 rank (r=4/8/32/64) and layer-specific (attn-only,
# mlp-only) LoRA adapters into the bf16 Llama-3.1-8B-Instruct base. This is the
# merge stage of the post-training pipeline used for these ablations; the
# subsequent AutoDAN generation/evaluation stage lives with the evaluation code.
# Run after rank/slurm/run_rank_ablation.slurm and layer/slurm/run_layer_specific.slurm.

set -eo pipefail

module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env
set -u

cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
BASE_MODEL_PATH="models/llama3.1/llama-3.1-8b-instruct-base"

if [ ! -d "$BASE_MODEL_PATH" ] || [ ! -f "$BASE_MODEL_PATH/config.json" ]; then
    echo "ERROR: Local base model missing or incomplete: $BASE_MODEL_PATH"
    exit 1
fi

MODELS_TO_MERGE=(
    "llama-3.1-8b-lora-alpaca-r4:llama-3.1-8b-instruct-lora-r4-merged"
    "llama-3.1-8b-lora-alpaca-r8:llama-3.1-8b-instruct-lora-r8-merged"
    "llama-3.1-8b-lora-alpaca-r32:llama-3.1-8b-instruct-lora-r32-merged"
    "llama-3.1-8b-lora-alpaca-r64:llama-3.1-8b-instruct-lora-r64-merged"
    "llama-3.1-8b-lora-alpaca-attn-only:llama-3.1-8b-instruct-lora-attn-only-merged"
    "llama-3.1-8b-lora-alpaca-mlp-only:llama-3.1-8b-instruct-lora-mlp-only-merged"
)

for MODEL_PAIR in "${MODELS_TO_MERGE[@]}"; do
    ADAPTER_NAME="${MODEL_PAIR%%:*}"
    MERGED_NAME="${MODEL_PAIR##*:}"
    ADAPTER_PATH="models/llama3.1/$ADAPTER_NAME"
    MERGED_PATH="models/llama3.1/$MERGED_NAME"

    if [ ! -d "$ADAPTER_PATH" ]; then
        echo "ERROR: Missing adapter directory: $ADAPTER_PATH"
        exit 1
    fi

    if [ -d "$MERGED_PATH" ]; then
        echo "  SKIP: $MERGED_NAME (already merged)"
        continue
    fi

    echo "  Merging $ADAPTER_NAME -> $MERGED_NAME..."
    python -c "
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained('$BASE_MODEL_PATH', torch_dtype=torch.bfloat16, device_map='auto')
tok = AutoTokenizer.from_pretrained('$BASE_MODEL_PATH')
model = PeftModel.from_pretrained(base, '$ADAPTER_PATH')
merged = model.merge_and_unload()
merged.save_pretrained('$MERGED_PATH')
tok.save_pretrained('$MERGED_PATH')
print('Merged successfully!')
"
    if [ ! -d "$MERGED_PATH" ] || [ ! -f "$MERGED_PATH/config.json" ]; then
        echo "ERROR: Merge did not produce expected model files: $MERGED_PATH"
        exit 1
    fi
done
