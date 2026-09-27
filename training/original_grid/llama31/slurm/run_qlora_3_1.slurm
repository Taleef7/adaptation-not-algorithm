#!/bin/bash

#SBATCH --job-name=qlora_llama3.1_alpaca
#SBATCH --output=logs/finetune/llama3.1/qlora_3_1_finetune_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=90G
#SBATCH --qos=standby

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

module --force purge
module load external
module load anaconda/2025.06-py313
conda activate thesis_env

echo "Conda environment activated."
echo "Running Llama 3.1 QLoRA script..."

# HF QLoRA script
python -u training/original_grid/llama31/finetune_qlora_llama3.1.py

echo "Python script finished."
echo "Job finished at $(date)"