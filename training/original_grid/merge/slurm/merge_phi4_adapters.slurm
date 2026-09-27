#!/bin/bash
#SBATCH --job-name=merge_phi4
#SBATCH --partition=a100-80gb
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=standby
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=90G
#SBATCH --time=4:00:00
#SBATCH --output=logs/merge_phi4_%j.log
#SBATCH --error=logs/merge_phi4_%j.err

# Load conda environment
module load anaconda/2025.06-py313
source activate thesis_env

# Set PyTorch memory allocation configuration for better memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the merge script
python ${PROJECT_ROOT}/training/original_grid/merge/merge_phi4_adapters.py
