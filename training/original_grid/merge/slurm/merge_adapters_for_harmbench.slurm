#!/bin/bash
#SBATCH --job-name=merge_adapters
#SBATCH --partition=a10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=90G
#SBATCH --output=logs/merge_adapters_%j.log
#SBATCH --error=logs/merge_adapters_%j.err
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=standby

# Merge PEFT adapters for HarmBench compatibility
# This creates merged model directories needed for vLLM loading

module load anaconda/2025.06-py313
source activate thesis_env

echo "=========================================="
echo "Merging PEFT Adapters for HarmBench"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Change to project directory
cd ${PROJECT_ROOT}

# Run merge script for all models and methods
python training/original_grid/merge/merge_adapters_for_harmbench.py \
    --models-dir ${PROJECT_ROOT}/models \
    --model both \
    --method both

echo ""
echo "=========================================="
echo "Merging Complete"
echo "=========================================="
