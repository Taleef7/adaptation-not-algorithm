#!/bin/bash
# GCG-JBB steps 1.5-3 for one cell: merge per-behavior test cases -> generate completions -> score
# with HarmBench-Llama-2-13b-cls. Run this only after submit_gcg_jbb.sh has banked all 100 behaviors
# (check: ls results_jbb/GCG/<cell>/test_cases/test_cases_individual_behaviors | wc -l).
#
# The three steps run in one job (merge <1 min, completions ~6 min, scoring ~2 min). The classifier
# loads only after the generation process has exited, so peak GPU memory is the larger of the two.
#
# Every step overwrites its own output, so the job can be resubmitted as a whole.
#
# generate_completions.sh always passes --generate_with_vllm; generate_completions.py downgrades that
# to HF generation for cells whose models.yaml entry sets use_vllm: False (the phi4 family). Do not
# change the flag here; the per-model setting keeps phi4 consistent with qwen25_14b.
#
# Usage: ./submit_gcg_jbb_score.sh <cell> <comma_partitions>
#   e.g. ./submit_gcg_jbb_score.sh qwen3_lora_custom a10,a30,a100-40gb,a100-80gb
set -euo pipefail
cell=$1; PARTS=$2
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
HB=$REPO/toolkits/HarmBench
LOG=$REPO/logs/gcg_jbb_score; mkdir -p "$LOG"
JBB=./data/behavior_datasets/jbb_behaviors.csv
DIR=./results_jbb/GCG/${cell}
CLS=cais/HarmBench-Llama-2-13b-cls

# Only score a cell whose attack phase covers all 100 behaviors.
n=$(ls "$HB/${DIR#./}/test_cases/test_cases_individual_behaviors" 2>/dev/null | wc -l)
[ "$n" -eq 100 ] || { echo "ABORT: $cell has $n/100 behaviors banked"; exit 1; }

JID=$(sbatch --parsable --account=YOUR_ACCOUNT --partition="$PARTS" --qos=standby --gres=gpu:1 \
  --cpus-per-task=6 --mem=80G --time=2:00:00 \
  --job-name="gcgScore_${cell}" --output="$LOG/${cell}.log" \
  --wrap="cd $HB \
    && bash scripts/merge_test_cases_env.sh GCG $DIR/test_cases \
    && bash scripts/generate_completions.sh ${cell} $JBB $DIR/test_cases/test_cases.json $DIR/completions/${cell}.json 512 False \
    && bash scripts/evaluate_completions.sh $CLS $JBB $DIR/completions/${cell}.json $DIR/results/${cell}.json")
echo "$cell: submitted GCG-JBB score job $JID on [$PARTS]"
