#!/bin/bash
# Seed-42 controlled grid: submit one training job on a SLURM cluster.
#   PEFT (lora/qlora): train_canonical.py -> merge_adapter.py
#   FFT (<=9B):        train_fft_canonical.py (saves the full model directly)
#   FFT (14B phi4/qwen25): use train_fft_14b_<family>.slurm instead.
# Usage: ./submit_train.sh <family> <method>   e.g. ./submit_train.sh llama31 lora
#   families: llama31 gemma2 qwen3 phi4 qwen25 llama31_mixat   methods: lora qlora fft
# Required env: PROJECT_ROOT (repository root with models/ and data/), VENV_DIR (python venv).
set -euo pipefail
FAMILY=$1; METHOD=$2
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
VENV="${VENV_DIR:?set VENV_DIR to the training virtualenv}"
LOGDIR=$REPO/logs/training
mkdir -p "$LOGDIR"

# Resource classes used for the paper's runs (6,470 optimizer steps, effective batch 8):
#   <=9B LoRA/QLoRA  -> 1x A100-40GB, ~2-4h
#   14B LoRA/QLoRA   -> 1x A100-80GB, ~5-6h
#   <=9B FFT         -> 1x A100-80GB, ~2.5-4h
# Adjust partition names to your cluster.
KEY="${FAMILY}_${METHOD}"
case "$KEY" in
  qwen3_lora|llama31_lora|gemma2_lora|llama31_mixat_lora)  PART=a100-40gb; TIME=4:00:00; MEM=80G  ;;
  qwen3_qlora|llama31_qlora|gemma2_qlora)                  PART=a100-40gb; TIME=4:00:00; MEM=80G  ;;
  phi4_lora|qwen25_lora|phi4_qlora|qwen25_qlora)           PART=a100-80gb; TIME=8:00:00; MEM=100G ;;
  llama31_fft|qwen3_fft|gemma2_fft|llama31_mixat_fft)      PART=a100-80gb; TIME=6:00:00; MEM=120G ;;
  *)                                                       PART=a100-80gb; TIME=8:00:00; MEM=120G ;;
esac

# Software stack for the seed-42 runs: torch 2.8.0 (cu128), transformers 4.56.2, trl 0.23.0,
# peft 0.17.1, datasets 3.6.0, bitsandbytes 0.47.0.
if [ "$METHOD" = "fft" ]; then
  RUN="$VENV/bin/python -u training/seed42_grid/train_fft_canonical.py --family $FAMILY"
else
  RUN="$VENV/bin/python -u training/seed42_grid/train_canonical.py --family $FAMILY --method $METHOD && \
$VENV/bin/python -u training/seed42_grid/merge_adapter.py --family $FAMILY --method $METHOD"
fi
sbatch --account=YOUR_ACCOUNT --partition=$PART \
  --gres=gpu:1 --cpus-per-task=6 --mem=$MEM --time=$TIME \
  --job-name="seed42_${FAMILY}_${METHOD}" \
  --output="$LOGDIR/${FAMILY}_${METHOD}_%j.log" \
  --wrap="export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && $RUN && \
echo DONE_${FAMILY}_${METHOD}"
