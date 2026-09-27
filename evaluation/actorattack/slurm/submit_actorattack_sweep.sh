#!/bin/bash
# ActorAttack full-sweep submission: 1 shared pre-attack + 25 in-attack jobs.
#
# Pipeline design (mirrors HarmBench run_pipeline.py structure):
#   Phase 1 — pre-attack (1 job):  generates social actors + queries for all
#             400 behaviors using the attacker model ONLY (Mistral-7B). The
#             target model is not involved. Output is a single shared JSON file.
#   Phase 2 — in-attack (25 jobs): each loads one target model and runs the
#             actual multi-round attacks using Phase 1 output. All 25 jobs
#             start after Phase 1 completes (afterok dependency).
#
# Academic rationale for shared pre-attack:
#   Pre-attack is purely behavior-driven; it has no knowledge of the target
#   model. Sharing it means all 25 models face identical attack inputs, which
#   eliminates attacker stochasticity as a confound in cross-model comparisons.
#   This is MORE rigorous than 25 independent pre-attack runs.
#
# Cluster settings: in-attack jobs use standby QOS on a100-80gb nodes; the shared pre-attack
# job uses normal QOS on a100-80gb (8h wall). Adjust partitions/QOS for your cluster.
#
# Usage:
#   bash evaluation/actorattack/slurm/submit_actorattack_sweep.sh              # full sweep
#   DRY_RUN=1 bash evaluation/actorattack/slurm/submit_actorattack_sweep.sh    # preview
#   MODELS_FILTER=llama31 bash evaluation/actorattack/slurm/submit_actorattack_sweep.sh
#   SKIP_PREATTACK=1 bash evaluation/actorattack/slurm/submit_actorattack_sweep.sh  # reuse existing
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$ROOT_DIR"

# ---------------------------------------------------------------------------
# Config (override via env)
# ---------------------------------------------------------------------------
ATTACK_METHOD="${ATTACK_METHOD:-ActorAttack}"
BEHAVIOR_LIMIT="${BEHAVIOR_LIMIT:-400}"
ACTOR_COUNT="${ACTOR_COUNT:-2}"
BEHAVIOR_CSV="${BEHAVIOR_CSV:-$ROOT_DIR/toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv}"
ATTACKER_MODEL="${ATTACKER_MODEL:-mistral_7b_v3_attacker_custom}"
MAX_FALLBACK="${MAX_FALLBACK:-40}"
DRY_RUN="${DRY_RUN:-0}"
SUBMISSION_DELAY="${SUBMISSION_DELAY:-5}"
MODELS_FILTER="${MODELS_FILTER:-}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/toolkits/HarmBench/results}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/toolkits/HarmBench/slurm_logs}"
ACTORATTACK_OUTPUT_ROOT="${ACTORATTACK_OUTPUT_ROOT:-}"
PREATTACK_RESULT_ROOT="${PREATTACK_RESULT_ROOT:-$ROOT_DIR/toolkits/ActorAttack/pre_attack_result}"

# Canonical pre-attack output path — deterministic, no timestamp.
# Set SKIP_PREATTACK=1 to skip Phase 1 and reuse this file if it already exists.
CANONICAL_PRE_ATTACK_PATH="${ACTORATTACK_PRE_ATTACK_OUTPUT_PATH:-$PREATTACK_RESULT_ROOT/shared_${ATTACKER_MODEL}_${BEHAVIOR_LIMIT}.json}"
SKIP_PREATTACK="${SKIP_PREATTACK:-0}"

# Phase 1 (pre-attack): only the attacker model, no target — a100-80gb normal QOS.
# Normal QOS used here because hp_a100-80gb=1 allows it, and the 8h wall gives safe
# headroom. Checkpointing means a killed job resumes rather than restarting from scratch.
PREATTACK_PARTITION="${PREATTACK_PARTITION:-a100-80gb}"
PREATTACK_QOS="${PREATTACK_QOS:-normal}"
PREATTACK_TIME="${PREATTACK_TIME:-08:00:00}"
PREATTACK_MEM="${PREATTACK_MEM:-60G}"

# Phase 2 (in-attack): target model + attacker — a100-80gb standby.
# With shared pre-attack, each job only runs Phase 2 (~1.5-2h), fits in 4h.
GEN_QOS="${GEN_QOS:-standby}"
GEN_TIME="${GEN_TIME:-04:00:00}"
GEN_PARTITION="a100-80gb"
GEN_MEM="90G"

# Post-processing and evaluation — a30 standby (classifier only, <4h each).
POST_PARTITION="a30"
POST_QOS="standby"
POST_MEM="16G"
POST_TIME="00:45:00"

EVAL_L2_PARTITION="a30"
EVAL_L2_QOS="standby"
EVAL_L2_TIME="03:45:00"
EVAL_LG_PARTITION="a30"
EVAL_LG_QOS="standby"
EVAL_LG_TIME="03:00:00"
EVAL_API_PARTITION="a30"
EVAL_API_QOS="standby"
EVAL_API_TIME="03:00:00"

# ---------------------------------------------------------------------------
# Model list: 5 families × 5 configs = 25 models
# ---------------------------------------------------------------------------
ALL_MODELS=(
  qwen3_base_fp16_custom
  qwen3_base_4bit_custom
  qwen3_lora_custom
  qwen3_qlora_custom
  qwen3_fft_custom

  llama31_base_fp16_custom
  llama31_base_4bit_custom
  llama31_lora_custom
  llama31_qlora_custom
  llama31_fft_custom

  gemma2_base_fp16_custom
  gemma2_base_4bit_custom
  gemma2_lora_custom
  gemma2_qlora_custom
  gemma2_fft_custom

  phi4_base_fp16_custom
  phi4_base_4bit_custom
  phi4_lora_custom
  phi4_qlora_custom
  phi4_fft_custom

  qwen25_14b_base_fp16_custom
  qwen25_14b_base_4bit_custom
  qwen25_14b_lora_custom
  qwen25_14b_qlora_custom
  qwen25_14b_fft_custom
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "========================================"
echo "ActorAttack HarmBench sweep (shared pre-attack)"
echo "  behaviors     : $BEHAVIOR_LIMIT"
echo "  CSV           : $BEHAVIOR_CSV"
echo "  attacker      : $ATTACKER_MODEL"
echo "  pre-attack out: $CANONICAL_PRE_ATTACK_PATH"
echo "  results root  : $RESULTS_ROOT"
echo "  log root      : $LOG_ROOT"
echo "  output root   : ${ACTORATTACK_OUTPUT_ROOT:-<default>}"
echo "  max fallback  : $MAX_FALLBACK / $BEHAVIOR_LIMIT"
echo "  DRY_RUN       : $DRY_RUN"
[ -n "$MODELS_FILTER" ] && echo "  filter        : *${MODELS_FILTER}*"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Phase 1: submit pre-attack job (or reuse existing file)
# ---------------------------------------------------------------------------
PREATTACK_DEP=""

if [ "$SKIP_PREATTACK" = "1" ] && [ -f "$CANONICAL_PRE_ATTACK_PATH" ]; then
  echo "[Phase 1] Reusing existing pre-attack file:"
  echo "  $CANONICAL_PRE_ATTACK_PATH"
  echo "  (set SKIP_PREATTACK=0 to force a fresh run)"
elif squeue -u "$USER" -h -n "AA-preattack" | grep -q .; then
  EXISTING_PA_JOB=$(squeue -u "$USER" -h -n "AA-preattack" -o "%i" | head -1)
  echo "[Phase 1] Pre-attack job already in queue: $EXISTING_PA_JOB — in-attack jobs will depend on it"
  PREATTACK_DEP="afterok:${EXISTING_PA_JOB}"
else
  PA_LOG_DIR="$LOG_ROOT/generate_completions/$ATTACK_METHOD/shared_preattack"
  mkdir -p "$PA_LOG_DIR"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[Phase 1] DRY_RUN: would submit pre-attack job"
    echo "  partition=$PREATTACK_PARTITION qos=$PREATTACK_QOS time=$PREATTACK_TIME mem=$PREATTACK_MEM"
  else
    PA_JOB_ID=$(sbatch --parsable \
      --partition="$PREATTACK_PARTITION" \
      --qos="$PREATTACK_QOS" \
      --gpus-per-node=1 \
      --time="$PREATTACK_TIME" \
      --mem="$PREATTACK_MEM" \
      --job-name="AA-preattack" \
      --output="$PA_LOG_DIR/preattack_%j.log" \
      --account=YOUR_ACCOUNT \
      --export=ALL,ACTORATTACK_BEHAVIOR_LIMIT="$BEHAVIOR_LIMIT",ACTORATTACK_ACTOR_COUNT="$ACTOR_COUNT",ACTORATTACK_BEHAVIOR_CSV="$BEHAVIOR_CSV",ACTORATTACK_ATTACKER_MODEL_NAME="$ATTACKER_MODEL",ACTORATTACK_PRE_ATTACK_OUTPUT_PATH="$CANONICAL_PRE_ATTACK_PATH",ACTORATTACK_CHECKPOINT_DIR="$PREATTACK_RESULT_ROOT/checkpoints" \
      evaluation/actorattack/slurm/actorattack_preattack.slurm)
    echo "[Phase 1] Submitted pre-attack job: $PA_JOB_ID"
    PREATTACK_DEP="afterok:${PA_JOB_ID}"
  fi
fi

echo ""

# ---------------------------------------------------------------------------
# Phase 2: submit 25 in-attack jobs (each depends on pre-attack completing)
# ---------------------------------------------------------------------------
echo "[Phase 2] Submitting in-attack jobs..."
echo ""

SUBMITTED=0
SKIPPED=0
IDX=0

for MODEL in "${ALL_MODELS[@]}"; do
  ((IDX++)) || true

  if [ -n "$MODELS_FILTER" ] && [[ "$MODEL" != *"$MODELS_FILTER"* ]]; then
    continue
  fi

  echo "[$IDX/${#ALL_MODELS[@]}] $MODEL → partition=$GEN_PARTITION qos=$GEN_QOS mem=$GEN_MEM"

  if [ -n "$ACTORATTACK_OUTPUT_ROOT" ]; then
    ACTORATTACK_OUTPUT_DIR="$ACTORATTACK_OUTPUT_ROOT/$MODEL"
  else
    ACTORATTACK_OUTPUT_DIR=""
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY_RUN: skipping submission"
    ((SKIPPED++)) || true
    continue
  fi

  if squeue -u "$USER" -h -o "%j" | grep -Eq "^AA-gen-${MODEL}(-|$)"; then
    echo "  SKIP: AA-gen-${MODEL} chunk(s) already in queue"
    ((SKIPPED++)) || true
    continue
  fi

  MODEL_NAME="$MODEL" \
  ATTACK_METHOD="$ATTACK_METHOD" \
  RESULTS_ROOT="$RESULTS_ROOT" \
  LOG_ROOT="$LOG_ROOT" \
  ACTORATTACK_BEHAVIOR_LIMIT="$BEHAVIOR_LIMIT" \
  ACTORATTACK_ACTOR_COUNT="$ACTOR_COUNT" \
  ACTORATTACK_BEHAVIOR_CSV="$BEHAVIOR_CSV" \
  ACTORATTACK_ATTACKER_MODEL_NAME="$ATTACKER_MODEL" \
  ACTORATTACK_PRE_ATTACK_DATA_PATH="$CANONICAL_PRE_ATTACK_PATH" \
  ACTORATTACK_OUTPUT_ROOT="$ACTORATTACK_OUTPUT_ROOT" \
  ACTORATTACK_OUTPUT_DIR="$ACTORATTACK_OUTPUT_DIR" \
  ACTORATTACK_PARTITION="$GEN_PARTITION" \
  ACTORATTACK_QOS="$GEN_QOS" \
  ACTORATTACK_TIME="$GEN_TIME" \
  ACTORATTACK_MEM="$GEN_MEM" \
  ACTORATTACK_SUBMIT_POST_CHAIN="1" \
  ACTORATTACK_MAX_FALLBACK_BEHAVIORS="$MAX_FALLBACK" \
  ACTORATTACK_POST_PARTITION="$POST_PARTITION" \
  ACTORATTACK_POST_QOS="$POST_QOS" \
  ACTORATTACK_POST_MEM="$POST_MEM" \
  ACTORATTACK_POST_TIME="$POST_TIME" \
  ACTORATTACK_EVAL_L2_PARTITION="$EVAL_L2_PARTITION" \
  ACTORATTACK_EVAL_L2_QOS="$EVAL_L2_QOS" \
  ACTORATTACK_EVAL_L2_TIME="$EVAL_L2_TIME" \
  ACTORATTACK_EVAL_LG_PARTITION="$EVAL_LG_PARTITION" \
  ACTORATTACK_EVAL_LG_QOS="$EVAL_LG_QOS" \
  ACTORATTACK_EVAL_LG_TIME="$EVAL_LG_TIME" \
  ACTORATTACK_EVAL_API_PARTITION="$EVAL_API_PARTITION" \
  ACTORATTACK_EVAL_API_QOS="$EVAL_API_QOS" \
  ACTORATTACK_EVAL_API_TIME="$EVAL_API_TIME" \
  ACTORATTACK_PREATTACK_DEP="$PREATTACK_DEP" \
  JOB_SCRIPT="evaluation/actorattack/slurm/actorattack_sweep.slurm" \
    bash evaluation/actorattack/slurm/submit_actorattack_model.sh

  ((SUBMITTED++)) || true
  sleep "$SUBMISSION_DELAY"
done

echo ""
echo "========================================"
echo "Sweep submission done: $SUBMITTED submitted, $SKIPPED skipped."
echo "Phase 1 dependency: ${PREATTACK_DEP:-none (reusing existing file)}"
echo "Monitor: squeue -u \$USER | grep AA-"
echo "========================================"
