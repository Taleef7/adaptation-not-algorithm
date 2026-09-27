#!/bin/bash
#SBATCH --job-name=seed42_fft_phi4
#SBATCH --partition=a100-80gb
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/training/fft14b_phi4_%j.log

# Seed-42 controlled grid: phi4 full fine-tune on a single 80GB GPU
# (seed 42, paged_adamw_8bit, lr 2e-5, 1 epoch Alpaca-cleaned, 6,470 optimizer steps).
# The script checkpoints every 500 steps and resumes automatically if resubmitted.
# Submit from anywhere with PROJECT_ROOT and VENV_DIR exported.

set -e
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${VENV_DIR:?set VENV_DIR to the training virtualenv}/bin/python" -u training/seed42_grid/train_fft_14b_canonical.py --family phi4
