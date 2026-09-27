#!/bin/bash

#SBATCH --job-name=lora_llama3.1_seed0
#SBATCH --output=logs/finetune/llama3.1/lora_seed0_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --qos=normal
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"
echo "Multi-seed validation: LoRA seed=0"

mkdir -p logs/finetune/llama3.1 models/llama3.1 outputs/finetune/llama3.1

module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env

echo "Conda environment activated."
echo "Running LoRA fine-tuning (seed=0)..."

python -u training/original_grid/ablations/multi_seed/finetune_lora_fp16_llama3.1_seed0.py

echo "Python script finished."
echo "Job finished at $(date)"
