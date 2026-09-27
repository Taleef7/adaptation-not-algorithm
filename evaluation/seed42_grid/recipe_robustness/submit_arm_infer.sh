#!/bin/bash
# Recipe-robustness arms (second dataset / training length): DeepInception + ArtPrompt on
# JBB-100 and HarmBench-400 for one re-trained cell <family>_<method>_<arm>, using the same
# drivers and settings as the controlled grid (deepinception_phase5.py, artprompt_phase5.py with
# the original-grid ArtPrompt test cases of <family>_<method>). Only slices whose output is
# missing or incomplete (jbb=100, harmbench=400) are generated.
#
# ArtPrompt prompt format is taken from AP_TEMPLATE (default: alpaca, as in the controlled grid).
# The Qwen-3 dolly3ep arm cells were generated with AP_TEMPLATE=tokenizer.
#
# Usage: ./submit_arm_infer.sh <family> <method> <arm> [partition] [qos] [time]
#   e.g. AP_TEMPLATE=alpaca ./submit_arm_infer.sh gemma2 lora dolly3ep a100-80gb normal 16:00:00
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
family=$1; method=$2; ARM=$3; PART=${4:-a100-40gb}; QOS=${5:-standby}; TIME=${6:-4:00:00}
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
HBENV=${VENV_DIR:?set VENV_DIR}
AP_TEMPLATE=${AP_TEMPLATE:-alpaca}
HB=$REPO/toolkits/HarmBench
LOG=$REPO/logs/recipe_robustness
DI=$HB/results_track2/deepinception; AP=$HB/results_track2/artprompt
JBB_CSV=$HB/data/behavior_datasets/jbb_behaviors.csv
HB_CSV=$HB/data/behavior_datasets/harmbench_behaviors_text_all.csv
cell="${family}_${method}_${ARM}"
model="$REPO/models/track2/${family}_${method}_${ARM}_seed42_merged"
orig="${family}_${method}"
DIpy="$HBENV/bin/python -u $REPO/evaluation/seed42_grid/deepinception_phase5.py"
APpy="$HBENV/bin/python -u $REPO/evaluation/seed42_grid/artprompt_phase5.py --template $AP_TEMPLATE"

need() { # need <file> <expected_n>  -> true if missing or wrong row count
  [ ! -f "$1" ] && return 0
  [ "$(wc -l < "$1")" -ne "$2" ] && return 0
  return 1
}
CMDS=()
need "$DI/${cell}_jbb.jsonl" 100 && CMDS+=("$DIpy --model_path $model --benchmark jbb --out $DI/${cell}_jbb.jsonl")
need "$DI/${cell}_harmbench.jsonl" 400 && CMDS+=("$DIpy --model_path $model --benchmark harmbench --behaviors_csv $HB_CSV --out $DI/${cell}_harmbench.jsonl")
need "$AP/${cell}_jbb.jsonl" 100 && CMDS+=("$APpy --model_path $model --test_cases $HB/results_jbb/ArtPrompt/${orig}_custom/test_cases/test_cases.json --behaviors_csv $JBB_CSV --benchmark jbb --out $AP/${cell}_jbb.jsonl")
need "$AP/${cell}_harmbench.jsonl" 400 && CMDS+=("$APpy --model_path $model --test_cases $HB/results/ArtPrompt/${orig}_custom/test_cases/test_cases.json --behaviors_csv $HB_CSV --benchmark harmbench --out $AP/${cell}_harmbench.jsonl")

if [ ${#CMDS[@]} -eq 0 ]; then echo "$cell: ALL 4 SLICES COMPLETE, nothing to submit"; exit 0; fi
RUN=$(printf ' %s &&' "${CMDS[@]}"); RUN="${RUN% &&}"
echo "$cell: submitting ${#CMDS[@]} incomplete slice(s) on $PART/$QOS/$TIME"
sbatch --parsable --account=YOUR_ACCOUNT --partition=$PART --qos=$QOS \
  --gres=gpu:1 --cpus-per-task=6 --mem=100G --time=$TIME \
  --job-name="arm_inf_${cell}" --output="$LOG/infer_${cell}_%j.log" \
  --wrap="source $REPO/evaluation/slurm/env.sh && cd $REPO && $RUN && echo ARM_INFER_DONE_${cell}"
