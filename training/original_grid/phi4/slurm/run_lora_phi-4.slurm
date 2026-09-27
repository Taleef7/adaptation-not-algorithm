#!/bin/bash

#SBATCH --job-name=lora_phi4_alpaca
#SBATCH --output=logs/finetune/phi-4/lora_finetune_%j.txt 
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=150G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

# Create directories if they don't exist
mkdir -p logs models outputs scripts

# Activate conda environment
module --force purge
module load anaconda/2025.06-py313
source activate thesis_env

echo "Conda environment activated."
echo "Running LoRA fine-tuning script for Phi-4..."

# Run the python script with unbuffered output
python -u training/original_grid/phi4/finetune_lora_phi-4.py

echo "Python script finished."
echo "Job finished at $(date)"
