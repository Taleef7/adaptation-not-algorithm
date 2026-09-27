#!/bin/bash
# Score all multi-rollout DeepInception files in one GPU job (classifier loaded once).
# Usage: ./submit_score_rollouts.sh [partition] [qos] [time]
set -euo pipefail
PART=${1:-a100-40gb}; QOS=${2:-standby}; TIME=${3:-4:00:00}
# Required env: PROJECT_ROOT, VENV_DIR
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
HBENV=${VENV_DIR:?set VENV_DIR}
CLS=cais/HarmBench-Llama-2-13b-cls
RD=$REPO/evaluation/seed42_grid/rollout
IN=$REPO/toolkits/HarmBench/results_rollout/deepinception
OUTCSV=$REPO/artifacts/rollout_di_asr.csv
LABELS=$REPO/toolkits/HarmBench/results_rollout/scored
LOG=$REPO/logs/rollout; mkdir -p "$LOG"
sbatch --parsable --account=YOUR_ACCOUNT --partition=$PART --qos=$QOS \
  --gres=gpu:1 --cpus-per-task=6 --mem=100G --time=$TIME \
  --job-name="score_dirol" --output="$LOG/score_dirol_%j.log" \
  --wrap="source $REPO/evaluation/slurm/env.sh && cd $REPO && \
$HBENV/bin/python -u $RD/score_rollouts.py --in_dir $IN --out_csv $OUTCSV --labels_dir $LABELS --cls $CLS && \
echo SCORE_DIROL_DONE"