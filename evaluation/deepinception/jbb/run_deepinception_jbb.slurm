#!/bin/bash
#SBATCH --job-name=di_jbb
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=logs/deepinception/di_jbb_%j.log
# Original-grid DeepInception on JBB-100: runs one per-cell script from deepinception/jbb/<family>/.
# One job per (family, configuration). Submit from ${PROJECT_ROOT}; the scripts use paths relative to it
# (models/..., outputs/deepinception/...).
# Usage: sbatch --export=ALL,DI_SCRIPT=evaluation/deepinception/jbb/llama3.1/deep_inception_llama3.1_lora_alpaca.py \
#            evaluation/deepinception/jbb/run_deepinception_jbb.slurm
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${DI_SCRIPT:?set DI_SCRIPT}"
cd "$PROJECT_ROOT"
module load external anaconda
conda activate "${CONDA_ENV:-harmbench_env}"
export PYTHONNOUSERSITE=1
echo "Starting DeepInception: $DI_SCRIPT"
python -u "$DI_SCRIPT"
echo "Script finished."
