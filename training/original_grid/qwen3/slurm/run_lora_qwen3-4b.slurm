#!/bin/bash

#SBATCH --job-name=lora_qwen3_alpaca
#SBATCH --output=logs/finetune/qwen3-4b/lora_finetune_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

# Create directories if they don't exist
mkdir -p logs models outputs scripts

# Activate conda environment
module --force purge
module load external anaconda
conda activate thesis_env

echo "Conda environment activated."
echo "Running Python script..."

# Run the python script with unbuffered output
python -u training/original_grid/qwen3/finetune_lora_qwen3-4b.py

echo "Python script finished."
echo "Job finished at $(date)"
