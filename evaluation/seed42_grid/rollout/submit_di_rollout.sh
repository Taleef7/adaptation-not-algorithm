#!/bin/bash
# Multi-rollout DeepInception on the seed-42 controlled grid: DeepInception (do_sample, T=0.6,
# top_p=0.9) repeated with 5 rollout seeds to measure sampling variability of DeepInception ASR.
# One job per (family, method) runs 5 rollouts x {jbb, harmbench}; complete rollouts are skipped.
# Usage: ./submit_di_rollout.sh <family> <method> [partition] [qos] [time] [one_seed] [one_bench]
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
family=$1; method=$2; PART=${3:-a100-40gb}; QOS=${4:-standby}; TIME=${5:-4:00:00}; ONESEED=${6:-}; ONEBENCH=${7:-}
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
HBENV=${VENV_DIR:?set VENV_DIR}
HB=$REPO/toolkits/HarmBench
LOG=$REPO/logs/rollout
OUT=$HB/results_rollout/deepinception
HB_CSV=$HB/data/behavior_datasets/harmbench_behaviors_text_all.csv
mkdir -p "$LOG" "$OUT"
model="$REPO/models/phase5/${family}_${method}_seed42_merged"
DIpy="$HBENV/bin/python -u $REPO/evaluation/seed42_grid/rollout/deepinception_rollout.py"
SEEDS="${ONESEED:-0 1 2 3 4}"
CMDS=""
BENCHES="${ONEBENCH:-jbb harmbench}"
for s in $SEEDS; do
  for bench in $BENCHES; do
    out="$OUT/${family}_${method}_seed42_${bench}_roll${s}.jsonl"
    exp=100; [ "$bench" = harmbench ] && exp=400
    # skip rollouts that are already complete
    if [ -f "$out" ] && [ "$(wc -l < "$out")" -eq "$exp" ]; then continue; fi
    csv=""; [ "$bench" = harmbench ] && csv="--behaviors_csv $HB_CSV"
    CMDS="${CMDS}${CMDS:+ && }$DIpy --model_path $model --benchmark $bench $csv --seed $s --out $out"
  done
done
if [ -z "$CMDS" ]; then echo "${family}_${method}: all rollouts complete"; exit 0; fi
sbatch --parsable --account=YOUR_ACCOUNT --partition=$PART --qos=$QOS \
  --gres=gpu:1 --cpus-per-task=6 --mem=100G --time=$TIME \
  --job-name="dirol_${family}_${method}${ONESEED:+_s$ONESEED}${ONEBENCH:+_$ONEBENCH}" --output="$LOG/dirol_${family}_${method}${ONESEED:+_s$ONESEED}${ONEBENCH:+_$ONEBENCH}_%j.log" \
  --wrap="source $REPO/evaluation/slurm/env.sh && cd $REPO && $CMDS && echo DIROLL_DONE_${family}_${method}"
