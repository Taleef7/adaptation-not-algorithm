#!/bin/bash
#SBATCH --job-name=s42sec
#SBATCH --output=logs/seed42_secondary_%x_%j.log
#SBATCH --error=logs/seed42_secondary_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-40gb
#SBATCH --qos=standby
# Secondary-evaluator scoring of seed-42 controlled-grid completions (secondary_score.py).
# Pass EVAL and MANIFEST via: sbatch --export=ALL,EVAL=llamaguard,MANIFEST=<path>
# Resumes from partially written outputs. Submit from ${PROJECT_ROOT}.
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
HB=$PROJECT_ROOT
VENV=$VENV_DIR
export PYTHONNOUSERSITE=1
cd "$HB"
: "${EVAL:?set EVAL=gpt4omini|llamaguard}"
: "${MANIFEST:?set MANIFEST=<manifest path>}"
echo "node=$(hostname) EVAL=$EVAL MANIFEST=$MANIFEST started=$(date)"

EXTRA=""
if [ "$EVAL" = gpt4omini ]; then EXTRA="--max_retry 40 --retry_sleep 30 --req_interval 0.5"; fi

"$VENV/bin/python" evaluation/seed42_grid/secondary_score.py \
    --manifest "$MANIFEST" --evaluator "$EVAL" $EXTRA

echo "DONE EVAL=$EVAL finished=$(date)"
