#!/bin/bash
#SBATCH --job-name=s42_hbcls
#SBATCH --output=logs/seed42_grid/hbcls_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
# HB-Cls (cais/HarmBench-Llama-2-13b-cls) scoring job for the controlled grid.
# The 13B classifier needs a >= 40 GB GPU. Submit from ${PROJECT_ROOT}.
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$PROJECT_ROOT"
export PYTHONNOUSERSITE=1
echo "node=$(hostname) started=$(date)"
bash evaluation/seed42_grid/score_seed42_hbcls.sh
echo "finished=$(date)"
