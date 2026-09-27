#!/bin/bash
# Submits the three evaluators for one model's converted ActorAttack completions:
#   HB-Cls (HarmBench evaluate_completions.py, cais/HarmBench-Llama-2-13b-cls) -> results/
#   LlamaGuard-3 (evaluate_completions_llamaguard.py)                           -> results_llamaguard/
#   GPT-4o-mini (evaluate_attacks_api.py)                                        -> results_api/
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_ROOT="$ROOT_DIR/toolkits/HarmBench"

ATTACK_METHOD="${ATTACK_METHOD:-ActorAttack}"
MODEL_NAME="${MODEL_NAME:-llama31_base_fp16_custom}"
FORCE_RESUBMIT="${FORCE_RESUBMIT:-0}"
MIN_COMPLETION_BEHAVIORS="${MIN_COMPLETION_BEHAVIORS:-0}"
MAX_FALLBACK_BEHAVIORS="${MAX_FALLBACK_BEHAVIORS:-0}"

RESULTS_ROOT="${RESULTS_ROOT:-$HARMBENCH_ROOT/results}"
LOG_ROOT="${LOG_ROOT:-$HARMBENCH_ROOT/slurm_logs}"
BEHAVIORS_PATH="${BEHAVIORS_PATH:-$HARMBENCH_ROOT/data/behavior_datasets/harmbench_behaviors_text_all.csv}"
COMPLETIONS_PATH="${COMPLETIONS_PATH:-$RESULTS_ROOT/$ATTACK_METHOD/$MODEL_NAME/completions/$MODEL_NAME.json}"

SAVE_PATH_L2="${SAVE_PATH_L2:-$RESULTS_ROOT/$ATTACK_METHOD/$MODEL_NAME/results/$MODEL_NAME.json}"
SAVE_PATH_LG="${SAVE_PATH_LG:-$RESULTS_ROOT/$ATTACK_METHOD/$MODEL_NAME/results_llamaguard/$MODEL_NAME.json}"
SAVE_PATH_API="${SAVE_PATH_API:-$RESULTS_ROOT/$ATTACK_METHOD/$MODEL_NAME/results_api/${MODEL_NAME}_api_eval.json}"

LOG_BASE="${LOG_BASE:-$LOG_ROOT/evaluate_completions/$ATTACK_METHOD/$MODEL_NAME}"
LOG_L2_DIR="$LOG_BASE/eval_llama2_logs"
LOG_LG_DIR="$LOG_BASE/eval_llamaguard_logs"
LOG_API_DIR="$LOG_BASE/eval_api_logs"

L2_PARTITION="${ACTORATTACK_EVAL_L2_PARTITION:-a30}"
L2_QOS="${ACTORATTACK_EVAL_L2_QOS:-standby}"
L2_MEM="${ACTORATTACK_EVAL_L2_MEM:-64G}"
L2_TIME="${ACTORATTACK_EVAL_L2_TIME:-02:00:00}"

LG_PARTITION="${ACTORATTACK_EVAL_LG_PARTITION:-a30}"
LG_QOS="${ACTORATTACK_EVAL_LG_QOS:-standby}"
LG_MEM="${ACTORATTACK_EVAL_LG_MEM:-48G}"
LG_TIME="${ACTORATTACK_EVAL_LG_TIME:-02:00:00}"

API_PARTITION="${ACTORATTACK_EVAL_API_PARTITION:-a30}"
API_QOS="${ACTORATTACK_EVAL_API_QOS:-standby}"
API_MEM="${ACTORATTACK_EVAL_API_MEM:-32G}"
API_TIME="${ACTORATTACK_EVAL_API_TIME:-02:00:00}"

# Note: L2_TIME / LG_TIME / API_TIME are overridden by the sweep script
# to 03:45:00 / 03:00:00 / 03:00:00 for the full 400-behavior run.
ACTORATTACK_API_FRESH_EVAL="${ACTORATTACK_API_FRESH_EVAL:-1}"

CLASSIFIER_PATH="${CLASSIFIER_PATH:-cais/HarmBench-Llama-2-13b-cls}"
LLAMAGUARD_MODEL_PATH="${LLAMAGUARD_MODEL_PATH:-$ROOT_DIR/models/llama-guard-3-8b}"

mkdir -p "$(dirname "$SAVE_PATH_L2")" "$(dirname "$SAVE_PATH_LG")" "$(dirname "$SAVE_PATH_API")"
mkdir -p "$LOG_L2_DIR" "$LOG_LG_DIR" "$LOG_API_DIR"

if [ ! -f "$COMPLETIONS_PATH" ]; then
  echo "ERROR: Completions not found: $COMPLETIONS_PATH"
  echo "Run conversion first: python $ROOT_DIR/evaluation/actorattack/convert_actorattack_to_harmbench.py"
  exit 1
fi

if [ "$MIN_COMPLETION_BEHAVIORS" -gt 0 ] || [ "$MAX_FALLBACK_BEHAVIORS" -ge 0 ]; then
  read -r COMPLETION_COUNT FALLBACK_COUNT CONVERSION_FALLBACK_COUNT RUNTIME_FALLBACK_COUNT <<<"$(python - "$COMPLETIONS_PATH" <<'PY'
import json
import sys

path = sys.argv[1]
fallback_marker = "[ActorAttack conversion fallback]"
runtime_fallback_marker = "runtime_fallback_injected"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

if not isinstance(data, dict):
    print("0 0 0 0")
    raise SystemExit(0)

conversion_fallback_count = 0
runtime_fallback_count = 0
for items in data.values():
    if not items or not isinstance(items, list):
        conversion_fallback_count += 1
        continue
    first_item = items[0] if isinstance(items[0], dict) else {}
    generation = first_item.get("generation", "")
    test_case = first_item.get("test_case", "")
    if isinstance(generation, str) and generation.startswith(fallback_marker):
        conversion_fallback_count += 1
    if isinstance(test_case, str) and runtime_fallback_marker in test_case:
        runtime_fallback_count += 1

fallback_count = conversion_fallback_count + runtime_fallback_count
print(f"{len(data)} {fallback_count} {conversion_fallback_count} {runtime_fallback_count}")
PY
)"

  echo "Completion quality summary: total=$COMPLETION_COUNT fallback_total=$FALLBACK_COUNT conversion_fallback=$CONVERSION_FALLBACK_COUNT runtime_fallback=$RUNTIME_FALLBACK_COUNT"

  if [ "$MIN_COMPLETION_BEHAVIORS" -gt 0 ] && [ "$COMPLETION_COUNT" -lt "$MIN_COMPLETION_BEHAVIORS" ]; then
    echo "ERROR: Completion coverage gate failed for $MODEL_NAME"
    echo "Found $COMPLETION_COUNT behaviors, required at least $MIN_COMPLETION_BEHAVIORS"
    exit 1
  fi

  if [ "$FALLBACK_COUNT" -gt "$MAX_FALLBACK_BEHAVIORS" ]; then
    echo "ERROR: Completion fallback-quality gate failed for $MODEL_NAME"
    echo "Found $FALLBACK_COUNT fallback behaviors, allowed at most $MAX_FALLBACK_BEHAVIORS"
    exit 1
  fi

  echo "Completion quality gates passed."
fi

if [ ! -f "$BEHAVIORS_PATH" ]; then
  echo "ERROR: Behaviors CSV not found: $BEHAVIORS_PATH"
  exit 1
fi

if [ ! -d "$LLAMAGUARD_MODEL_PATH" ]; then
  echo "ERROR: LlamaGuard model path not found: $LLAMAGUARD_MODEL_PATH"
  exit 1
fi

job_exists() {
  local job_name="$1"
  squeue -u "$USER" -h -n "$job_name" | grep -q .
}

submit_job() {
  local name="$1"
  local partition="$2"
  local qos="$3"
  local mem="$4"
  local time_limit="$5"
  local out_log="$6"
  local err_log="$7"
  local wrap_cmd="$8"

  if [ "$FORCE_RESUBMIT" != "1" ] && job_exists "$name"; then
    echo "SKIP: Existing queued/running job with name '$name' (set FORCE_RESUBMIT=1 to override)" >&2
    return 1
  fi

  sbatch --parsable \
    --partition="$partition" \
    --account=YOUR_ACCOUNT \
    --qos="$qos" \
    --nodes=1 \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem="$mem" \
    --time="$time_limit" \
    --job-name="$name" \
    --output="$out_log" \
    --error="$err_log" \
    --wrap="$wrap_cmd"
}

L2_JOB_NAME="L2-AA-${MODEL_NAME}"
LG_JOB_NAME="LG-AA-${MODEL_NAME}"
API_JOB_NAME="API-AA-${MODEL_NAME}"

API_EXTRA_ARGS=""
if [ "$ACTORATTACK_API_FRESH_EVAL" = "1" ]; then
  API_EXTRA_ARGS=" --no_resume"
  echo "API evaluator freshness mode: enabled (--no_resume)"
else
  echo "API evaluator freshness mode: disabled (resume checkpoint if present)"
fi

L2_WRAP="module load external anaconda && conda activate harmbench_env && cd $HARMBENCH_ROOT && python -u evaluate_completions.py --cls_path $CLASSIFIER_PATH --behaviors_path $BEHAVIORS_PATH --completions_path $COMPLETIONS_PATH --save_path $SAVE_PATH_L2 --num_tokens 512"
LG_WRAP="module load external anaconda && conda activate harmbench_env && cd $HARMBENCH_ROOT && python -u evaluate_completions_llamaguard.py --behaviors_path $BEHAVIORS_PATH --completions_path $COMPLETIONS_PATH --save_path $SAVE_PATH_LG --model_path $LLAMAGUARD_MODEL_PATH --num_tokens 512"
API_WRAP="module load external anaconda && conda activate harmbench_env && cd $HARMBENCH_ROOT && python -u evaluate_attacks_api.py --behaviors_path $BEHAVIORS_PATH --completions_path $COMPLETIONS_PATH --save_path $SAVE_PATH_API --num_tokens 512$API_EXTRA_ARGS"

echo "Submitting ActorAttack evaluator jobs for model: $MODEL_NAME"
echo "Completions: $COMPLETIONS_PATH"

echo "Submitting Llama-2 classifier job..."
if L2_JOB_ID=$(submit_job "$L2_JOB_NAME" "$L2_PARTITION" "$L2_QOS" "$L2_MEM" "$L2_TIME" "$LOG_L2_DIR/$MODEL_NAME.log" "$LOG_L2_DIR/$MODEL_NAME.err" "$L2_WRAP"); then
  echo "Llama-2 job id: $L2_JOB_ID"
fi

echo "Submitting LlamaGuard-3 classifier job..."
if LG_JOB_ID=$(submit_job "$LG_JOB_NAME" "$LG_PARTITION" "$LG_QOS" "$LG_MEM" "$LG_TIME" "$LOG_LG_DIR/$MODEL_NAME.log" "$LOG_LG_DIR/$MODEL_NAME.err" "$LG_WRAP"); then
  echo "LlamaGuard-3 job id: $LG_JOB_ID"
fi

echo "Submitting GPT-4o-mini API evaluator job..."
if API_JOB_ID=$(submit_job "$API_JOB_NAME" "$API_PARTITION" "$API_QOS" "$API_MEM" "$API_TIME" "$LOG_API_DIR/$MODEL_NAME.log" "$LOG_API_DIR/$MODEL_NAME.err" "$API_WRAP"); then
  echo "GPT-4o-mini API job id: $API_JOB_ID"
fi

echo
echo "Current queue entries for this model:"
squeue -u "$USER" -h -n "$L2_JOB_NAME","$LG_JOB_NAME","$API_JOB_NAME" -o "%.18i %.20j %.8T %.10P %.8Q %.20M %.20R" || true
