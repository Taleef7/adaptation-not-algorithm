#!/bin/bash
# Recipe robustness (track 2), combined arm: Dolly-15k for 3 epochs.
# Submits one training job (train -> merge for LoRA; FFT saves the full model directly).
# Recipe otherwise identical to the seed-42 controlled grid.
# Usage: ./submit_track2.sh <family> <method>   e.g. ./submit_track2.sh llama31 lora
#   families: llama31 gemma2 qwen3   methods: lora fft
# Required env: PROJECT_ROOT (repository root with models/ and data/), VENV_DIR (python venv).
# Run prep_dolly.py once first to create $PROJECT_ROOT/data/dolly15k_dataset.
set -euo pipefail
FAMILY=$1; METHOD=$2
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
VENV="${VENV_DIR:?set VENV_DIR to the training virtualenv}"
T2=$REPO/training/track2
LOGDIR=$REPO/logs/training
mkdir -p "$LOGDIR"

# Dolly 15,011 examples x 3 epochs / effective batch 8 = ~5,629 optimizer steps.
KEY="${FAMILY}_${METHOD}"
case "$KEY" in
  llama31_lora|gemma2_lora|qwen3_lora)   PART=a100-40gb; TIME=4:00:00; MEM=80G  ;;
  llama31_fft|gemma2_fft|qwen3_fft)      PART=a100-80gb; TIME=4:00:00; MEM=120G ;;
  *) echo "unsupported cell $KEY (llama31|gemma2|qwen3 x lora|fft)"; exit 1 ;;
esac

# Software stack for the track-2 runs: torch 2.8.0 (cu128), transformers 4.56.1, trl 0.22.2,
# peft 0.17.1, datasets 3.6.0, bitsandbytes 0.47.0.
if [ "$METHOD" = "fft" ]; then
  RUN="$VENV/bin/python -u $T2/train_track2_fft.py --family $FAMILY"
else
  RUN="$VENV/bin/python -u $T2/train_track2_peft.py --family $FAMILY --method $METHOD && \
$VENV/bin/python -u $T2/merge_track2.py --family $FAMILY --method $METHOD"
fi
sbatch --account=YOUR_ACCOUNT --partition=$PART \
  --gres=gpu:1 --cpus-per-task=6 --mem=$MEM --time=$TIME \
  --job-name="t2_${FAMILY}_${METHOD}" \
  --output="$LOGDIR/t2_${FAMILY}_${METHOD}_%j.log" \
  --wrap="export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && $RUN && \
echo TRACK2_DONE_${FAMILY}_${METHOD}"
