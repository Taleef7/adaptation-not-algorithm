#!/bin/bash
# Recipe robustness (track 2): submit one training job for a single-factor arm.
#   dolly1ep   = Dolly-15k @ 1 epoch                      (dataset arm, vs Alpaca/1 epoch)
#   dollystep  = Dolly-15k @ exactly 6,470 optimizer steps (step-matched arm: same number of
#                steps as Alpaca @ 1 epoch = 51,760 / effective batch 8; Dolly @ 1 epoch is
#                only ~1,876 steps)
#   alpaca3ep  = Alpaca-cleaned @ 3 epochs                 (training-length arm)
# The combined Dolly @ 3 epochs arm is submitted with submit_track2.sh.
# Recipe otherwise identical to the seed-42 controlled grid (seed 42, lora_dropout 0.0,
# adamw_8bit, effective batch 8, bf16; lr 2e-4 PEFT / 2e-5 FFT).
# LoRA trains an adapter then merges; FFT saves the merged model directly.
# Outputs: $PROJECT_ROOT/models/track2/*_<arm>_seed42*.
# Usage: ./submit_arm.sh <family> <method> <arm> [partition] [time]
# Required env: PROJECT_ROOT (repository root with models/ and data/), VENV_DIR (python venv).
set -euo pipefail
family=$1; method=$2; arm=$3
# FFT (full fine-tune of 8-9B) needs 80GB; PEFT fits 40GB.
if [ "$method" = "fft" ]; then DEFPART=a100-80gb; else DEFPART=a100-40gb; fi
PART=${4:-$DEFPART}; TIME=${5:-4:00:00}
REPO="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
VENV="${VENV_DIR:?set VENV_DIR to the training virtualenv}"
T2=$REPO/training/track2
LOG=$REPO/logs/training; mkdir -p "$LOG"
DOLLY=$REPO/data/dolly15k_dataset; ALPACA=$REPO/data/alpaca_cleaned_dataset
MAXSTEPS=-1
case "$arm" in
  dolly1ep)   DS=$DOLLY;  EP=1; PLR=2e-4; FLR=2e-5 ;;
  alpaca3ep)  DS=$ALPACA; EP=3; PLR=2e-4; FLR=2e-5 ;;
  # EP is ignored when max_steps > 0.
  dollystep)  DS=$DOLLY;  EP=99; PLR=2e-4; FLR=2e-5; MAXSTEPS=6470 ;;
  *) echo "unknown arm $arm (dolly1ep|dollystep|alpaca3ep)"; exit 1 ;;
esac
if [ "$method" = "fft" ]; then
  RUN="$VENV/bin/python -u $T2/train_arm_fft.py --family $family --dataset $DS --epochs $EP --max_steps $MAXSTEPS --lr $FLR --suffix $arm"
else
  RUN="$VENV/bin/python -u $T2/train_arm_peft.py --family $family --method $method --dataset $DS --epochs $EP --max_steps $MAXSTEPS --lr $PLR --suffix $arm && \
$VENV/bin/python -u $T2/merge_arm.py --family $family --method $method --suffix $arm"
fi
sbatch --parsable --account=YOUR_ACCOUNT --partition=$PART \
  --gres=gpu:1 --cpus-per-task=6 --mem=100G --time=$TIME \
  --job-name="arm_${family}_${method}_${arm}" --output="$LOG/arm_${family}_${method}_${arm}_%j.log" \
  --wrap="export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && cd $REPO && $RUN && echo ARM_DONE_${family}_${method}_${arm}"
