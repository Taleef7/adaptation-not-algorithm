#!/bin/bash

#SBATCH --job-name=fft_HF_llama3.1
#SBATCH --output=logs/finetune/llama3.1/fft_hf_finetune_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

# Load the anaconda module and the training environment
module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env

echo "Running PURE HUGGING FACE script for FFT..."

python -u ${PROJECT_ROOT}/training/original_grid/llama31/finetune_fft_hf_llama3.1.py

echo "Python script finished."
echo "Job finished at $(date)"