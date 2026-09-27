#!/bin/bash

#SBATCH --job-name=fft_llama3.1_seed0
#SBATCH --output=logs/finetune/llama3.1/fft_seed0_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=90G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"
echo "Multi-seed validation: FFT seed=0"

module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env

echo "Running PURE HUGGING FACE script for FFT (seed=0)..."

# FFT of the 8B model plus AdamW optimizer states nearly fills an 80GB GPU;
# expandable_segments reduces allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u ${PROJECT_ROOT}/training/original_grid/ablations/multi_seed/finetune_fft_hf_llama3.1_seed0.py

echo "Python script finished."
echo "Job finished at $(date)"
