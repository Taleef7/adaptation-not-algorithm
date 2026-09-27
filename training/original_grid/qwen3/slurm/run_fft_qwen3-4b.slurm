#!/bin/bash

#SBATCH --job-name=fft_qwen3_alpaca
#SBATCH --output=logs/finetune/qwen3-4b/fft_finetune_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='a100|h100'
#SBATCH --time=10:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

# Activate conda environment
module --force purge
module load external anaconda
conda activate thesis_env

echo "Running FULL FINE-TUNING script for Qwen3..."

# Run the python script with unbuffered output
python -u training/original_grid/qwen3/finetune_fft_qwen3-4b.py

echo "Python script finished."
echo "Job finished at $(date)"
