#!/bin/bash
#SBATCH --job-name=safe_lora_faithful
#SBATCH --output=logs/safe_lora_faithful_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=90G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby

# Faithful IBM-style SafeLoRA pipeline for Llama-3.1-8B / Llama-3.1-8B-Instruct.
#
# This pipeline processes a single faithful SafeLoRA variant per job.
# Use `SAFE_LORA_VARIANT=main|seed0|seed123` when submitting, or the
# companion submit script to launch the broader three-job sweep.
#
# Methodology:
#   - unaligned base model: meta-llama/Llama-3.1-8B
#   - aligned model:       local Llama-3.1-8B-Instruct checkpoint
#   - adapter-level projection following IBM SafeLoRA layer selection semantics
#   - selection mode:      number
#   - projected layers:    10
#
# Fair-comparison protocol:
#   - all faithful SafeLoRA variants reuse the existing main LoRA AutoDAN test cases
#
# Outputs:
#   - merged models under models/llama3.1/*safe-faithful*
#   - HarmBench results under toolkits/HarmBench/results/AutoDAN/llama31_safe_lora_faithful*

set -eo pipefail

echo "=============================================="
echo "Faithful SafeLoRA Pipeline"
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "GPU:      $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"
echo "=============================================="

module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env
set -u

ROOT_DIR="${PROJECT_ROOT}"
cd "$ROOT_DIR"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="$HF_HOME/hub"

HARMBENCH_DIR="$ROOT_DIR/toolkits/HarmBench"
ALIGN_MODEL="$ROOT_DIR/models/llama3.1/llama-3.1-8b-instruct-base"
BASE_MODEL_REPO="meta-llama/Llama-3.1-8B"
LLAMAGUARD_MODEL="$ROOT_DIR/models/llama-guard-3-8b"
LLAMA2_CLS="cais/HarmBench-Llama-2-13b-cls"
BEHAVIORS_PATH="$HARMBENCH_DIR/data/behavior_datasets/harmbench_behaviors_text_all.csv"
AUTODAN_TC="$HARMBENCH_DIR/results/AutoDAN/llama31_lora_custom/test_cases/test_cases.json"

if [ ! -d "$ALIGN_MODEL" ] || [ ! -f "$ALIGN_MODEL/config.json" ]; then
    echo "ERROR: aligned instruct model not found: $ALIGN_MODEL"
    exit 1
fi

if [ ! -f "$AUTODAN_TC" ]; then
    echo "ERROR: expected main LoRA AutoDAN test cases not found: $AUTODAN_TC"
    exit 1
fi

if [ ! -d "$LLAMAGUARD_MODEL" ]; then
    echo "ERROR: LlamaGuard model not found: $LLAMAGUARD_MODEL"
    exit 1
fi

if command -v hf >/dev/null 2>&1; then
    echo ""
    echo "==> Hugging Face auth"
    hf auth whoami || true
fi

SAFE_LORA_VARIANT="${SAFE_LORA_VARIANT:-main}"

case "$SAFE_LORA_VARIANT" in
  main)
    VARIANT_SPEC="llama31_safe_lora_faithful_custom|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-alpaca|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-safe-faithful-adapter|$ROOT_DIR/models/llama3.1/llama-3.1-8b-instruct-lora-safe-faithful-merged|Base FP16|llama31_base_fp16_custom|LoRA r=16|llama31_lora_custom|FFT|llama31_fft_custom"
    ;;
  seed0)
    VARIANT_SPEC="llama31_safe_lora_faithful_seed0_custom|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-alpaca-seed0|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-safe-faithful-adapter-seed0|$ROOT_DIR/models/llama3.1/llama-3.1-8b-instruct-lora-safe-faithful-merged-seed0|Base Seed0|llama31_base_fp16_seed0_custom|LoRA Seed0|llama31_lora_seed0_custom|FFT Seed0|llama31_fft_seed0_custom"
    ;;
  seed123)
    VARIANT_SPEC="llama31_safe_lora_faithful_seed123_custom|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-alpaca-seed123|$ROOT_DIR/models/llama3.1/llama-3.1-8b-lora-safe-faithful-adapter-seed123|$ROOT_DIR/models/llama3.1/llama-3.1-8b-instruct-lora-safe-faithful-merged-seed123|Base Seed123|llama31_base_fp16_seed123_custom|LoRA Seed123|llama31_lora_seed123_custom|FFT Seed123|llama31_fft_seed123_custom"
    ;;
  *)
    echo "ERROR: SAFE_LORA_VARIANT must be one of: main, seed0, seed123"
    exit 1
    ;;
esac

copy_legacy_aliases() {
    local canonical_path="$1"
    local alias_path="$2"
    if [ -f "$canonical_path" ]; then
        cp "$canonical_path" "$alias_path"
    fi
}

IFS='|' read -r MODEL_NAME ADAPTER_PATH OUTPUT_ADAPTER OUTPUT_MERGED BASE_LABEL BASE_MODEL LORA_LABEL LORA_MODEL FFT_LABEL FFT_MODEL <<< "$VARIANT_SPEC"

echo ""
echo "============================================================"
echo "Processing variant: $MODEL_NAME"
echo "Variant key:        $SAFE_LORA_VARIANT"
echo "Adapter:            $ADAPTER_PATH"
echo "Output merged:      $OUTPUT_MERGED"
echo "============================================================"

if [ ! -d "$ADAPTER_PATH" ] || [ ! -f "$ADAPTER_PATH/adapter_config.json" ]; then
    echo "ERROR: adapter not found: $ADAPTER_PATH"
    exit 1
fi

METADATA_FILE="$OUTPUT_MERGED/safe_lora_faithful_metadata.json"
if [ -f "$METADATA_FILE" ] && [ -f "$OUTPUT_MERGED/config.json" ]; then
    echo "  SKIP STEP 1: faithful SafeLoRA model already exists"
else
    echo "  STEP 1: applying faithful SafeLoRA projection"
    python -u training/original_grid/ablations/safelora/safe_lora_faithful.py \
        --base_model_path "$BASE_MODEL_REPO" \
        --aligned_model_path "$ALIGN_MODEL" \
        --adapter_path "$ADAPTER_PATH" \
        --output_adapter_path "$OUTPUT_ADAPTER" \
        --output_merged_path "$OUTPUT_MERGED" \
        --select_layers_type number \
        --num_proj_layers 10 \
        --dtype bfloat16
fi

if [ ! -f "$METADATA_FILE" ]; then
    echo "ERROR: faithful SafeLoRA metadata missing after projection: $METADATA_FILE"
    exit 1
fi

COMP_DIR="$HARMBENCH_DIR/results/AutoDAN/$MODEL_NAME/completions"
COMP_FILE="$COMP_DIR/${MODEL_NAME}.json"
COMP_ALIAS="$COMP_DIR/completions.json"
RESULT_DIR="$HARMBENCH_DIR/results/AutoDAN/$MODEL_NAME/results"
RESULT_FILE="$RESULT_DIR/${MODEL_NAME}.json"
RESULT_ALIAS="$RESULT_DIR/results.json"
API_DIR="$HARMBENCH_DIR/results/AutoDAN/$MODEL_NAME/results_api"
API_FILE="$API_DIR/${MODEL_NAME}_api_eval.json"
API_ALIAS="$API_DIR/results.json"
LG_DIR="$HARMBENCH_DIR/results/AutoDAN/$MODEL_NAME/results_llamaguard"
LG_FILE="$LG_DIR/${MODEL_NAME}.json"

echo "  STEP 2: generating completions"
if [ -f "$COMP_FILE" ]; then
    echo "    SKIP: completions already exist"
else
    mkdir -p "$COMP_DIR"
    cd "$HARMBENCH_DIR"
    python -u generate_completions.py \
        --model_name "$MODEL_NAME" \
        --models_config_file configs/model_configs/models.yaml \
        --behaviors_path "$BEHAVIORS_PATH" \
        --test_cases_path "$AUTODAN_TC" \
        --save_path "$COMP_FILE" \
        --max_new_tokens 512
    cd "$ROOT_DIR"
fi
copy_legacy_aliases "$COMP_FILE" "$COMP_ALIAS"

echo "  STEP 3: evaluating with Llama-2-13b-cls"
if [ -f "$RESULT_FILE" ]; then
    echo "    SKIP: Llama-2 results already exist"
else
    mkdir -p "$RESULT_DIR"
    cd "$HARMBENCH_DIR"
    python -u evaluate_completions.py \
        --cls_path "$LLAMA2_CLS" \
        --behaviors_path "$BEHAVIORS_PATH" \
        --completions_path "$COMP_FILE" \
        --save_path "$RESULT_FILE"
    cd "$ROOT_DIR"
fi
copy_legacy_aliases "$RESULT_FILE" "$RESULT_ALIAS"

echo "  STEP 4: evaluating with GPT-4o-mini"
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    if [ -f "$API_FILE" ]; then
        echo "    SKIP: GPT-4o-mini results already exist"
    else
        mkdir -p "$API_DIR"
        cd "$HARMBENCH_DIR"
        python -u evaluate_attacks_api.py \
            --behaviors_path "$BEHAVIORS_PATH" \
            --completions_path "$COMP_FILE" \
            --save_path "$API_FILE" \
            --num_tokens 512
        cd "$ROOT_DIR"
    fi
    copy_legacy_aliases "$API_FILE" "$API_ALIAS"
else
    echo "    SKIP: .env missing, so GPT-4o-mini evaluation was not run"
fi

echo "  STEP 5: evaluating with LlamaGuard-3"
if [ -f "$LG_FILE" ]; then
    echo "    SKIP: LlamaGuard results already exist"
else
    mkdir -p "$LG_DIR"
    cd "$HARMBENCH_DIR"
    python -u evaluate_completions_llamaguard.py \
        --behaviors_path "$BEHAVIORS_PATH" \
        --completions_path "$COMP_FILE" \
        --save_path "$LG_FILE" \
        --model_path "$LLAMAGUARD_MODEL" \
        --num_tokens 512
    cd "$ROOT_DIR"
fi


echo ""
echo "=============================================="
echo "Faithful SafeLoRA pipeline complete!"
echo "Finished: $(date)"
echo "=============================================="
echo "Variant complete: $SAFE_LORA_VARIANT"
echo "Model name:       $MODEL_NAME"
echo "=============================================="
