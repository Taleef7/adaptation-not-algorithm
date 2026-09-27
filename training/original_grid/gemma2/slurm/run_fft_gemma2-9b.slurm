#!/bin/bash

#SBATCH --job-name=fft_gemma2-9b_alpaca
#SBATCH --output=logs/finetune/gemma2-9b/fft_unsloth_gemma2-9b_finetune_%j.log

# FFT is the most demanding, so we request the full A100 and high memory
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=120G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

module --force purge
module load external
module load anaconda
# Unsloth runs in a separate environment (gemma_fft_env)
conda activate gemma_fft_env

echo "Conda environment activated."
echo "Running Gemma-2-9B Unsloth FFT script..."

python -u training/original_grid/gemma2/finetune_fft_unsloth_gemma2-9b.py

echo "Python script finished."
echo "Job finished at $(date)"