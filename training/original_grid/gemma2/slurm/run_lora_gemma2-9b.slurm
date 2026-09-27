#!/bin/bash

#SBATCH --job-name=lora_gemma2-9b_alpaca
#SBATCH --output=logs/finetune/gemma2-9b/lora_gemma2-9b_finetune_%j.log
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

module --force purge
module load external
module load anaconda
conda activate thesis_env

echo "Conda environment activated."
echo "Running Gemma-2-9B LoRA script..."

python -u training/original_grid/gemma2/finetune_lora_gemma2-9b.py

echo "Python script finished."
echo "Job finished at $(date)"