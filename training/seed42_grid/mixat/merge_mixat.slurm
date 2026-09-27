#!/bin/bash
#SBATCH --job-name=mixat_merge
#SBATCH --output=logs/training/mixat_merge_%j.log
#SBATCH --error=logs/training/mixat_merge_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION

# Merge the released MixAT LoRA adapter into its base model (CPU-side, device_map="cpu";
# ~16GB of bf16 weights out). The paper uses the Llama-3.1 arm: FAMILY=llama31 (default).
# Expects the adapter at $PROJECT_ROOT/models/mixat/llama31-mixat-adapter
# (INSAIT-Institute/Llama3.1-8B-MixAT).
set -eo pipefail
cd "${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
export PYTHONNOUSERSITE=1
echo "node=$(hostname) started=$(date)"
python -u training/seed42_grid/mixat/merge_mixat_base.py --family "${FAMILY:-llama31}"
echo "finished=$(date)"
