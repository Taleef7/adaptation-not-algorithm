#!/bin/bash
# ActorAttack submission for one target model (called by submit_actorattack_sweep.sh with the
# sweep settings in the environment). Submits in-attack generation in behavior chunks of
# ACTORATTACK_CHUNK_SIZE (actorattack_sweep.slurm), then, when ACTORATTACK_SUBMIT_POST_CHAIN=1,
# a dependent job that converts the dialogues to HarmBench completions
# (convert_actorattack_to_harmbench.py) and submits the three evaluators
# (submit_actorattack_evals.sh). Chunks with a complete checkpoint are skipped.
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$ROOT_DIR"

ATTACK_METHOD="${ATTACK_METHOD:-ActorAttack}"
MODEL_NAME="${MODEL_NAME:-llama31_base_fp16_custom}"
RESULTS_ROOT="${RESULTS_ROOT:-$ROOT_DIR/toolkits/HarmBench/results}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/toolkits/HarmBench/slurm_logs}"
ACTORATTACK_OUTPUT_ROOT="${ACTORATTACK_OUTPUT_ROOT:-}"
ACTORATTACK_RAW_RESULT_ROOT="${ACTORATTACK_RAW_RESULT_ROOT:-${ACTORATTACK_OUTPUT_ROOT:-$ROOT_DIR/toolkits/ActorAttack/attack_result}}"
if [ -z "${ACTORATTACK_OUTPUT_DIR:-}" ] && [ -n "$ACTORATTACK_OUTPUT_ROOT" ]; then
    ACTORATTACK_OUTPUT_DIR="$ACTORATTACK_OUTPUT_ROOT/$MODEL_NAME"
fi
if [ -z "${ACTORATTACK_CHECKPOINT_DIR:-}" ] && [ -n "$ACTORATTACK_OUTPUT_ROOT" ]; then
    ACTORATTACK_CHECKPOINT_DIR="$ACTORATTACK_OUTPUT_ROOT/checkpoints/$MODEL_NAME"
fi
ACTORATTACK_BEHAVIOR_LIMIT="${ACTORATTACK_BEHAVIOR_LIMIT:-2}"
ACTORATTACK_ACTOR_COUNT="${ACTORATTACK_ACTOR_COUNT:-2}"
ACTORATTACK_LOCAL_MAX_NEW_TOKENS="${ACTORATTACK_LOCAL_MAX_NEW_TOKENS:-256}"
ACTORATTACK_GPUS_PER_NODE="${ACTORATTACK_GPUS_PER_NODE:-1}"
ACTORATTACK_BACKEND="${ACTORATTACK_BACKEND:-hf_local}"
ACTORATTACK_TIME="${ACTORATTACK_TIME:-04:00:00}"
ACTORATTACK_MEM="${ACTORATTACK_MEM:-90G}"
JOB_SCRIPT="${JOB_SCRIPT:-evaluation/actorattack/slurm/actorattack_sweep.slurm}"
ACTORATTACK_PARTITION="${ACTORATTACK_PARTITION:-a100-40gb}"
ACTORATTACK_QOS="${ACTORATTACK_QOS:-standby}"
ACTORATTACK_SUBMIT_POST_CHAIN="${ACTORATTACK_SUBMIT_POST_CHAIN:-0}"
# Post-processing only (no generation): set ACTORATTACK_ALLOW_POST_ONLY=1 with ACTORATTACK_BEHAVIOR_LIMIT=400.

ACTORATTACK_POST_PARTITION="${ACTORATTACK_POST_PARTITION:-a30}"
ACTORATTACK_POST_QOS="${ACTORATTACK_POST_QOS:-standby}"
ACTORATTACK_POST_MEM="${ACTORATTACK_POST_MEM:-16G}"
ACTORATTACK_POST_TIME="${ACTORATTACK_POST_TIME:-00:45:00}"
ACTORATTACK_ALLOW_POST_ONLY="${ACTORATTACK_ALLOW_POST_ONLY:-0}"
HARMBENCH_ROOT="$ROOT_DIR/toolkits/HarmBench"
# Generation logs go under LOG_ROOT so JBB can keep a separate log tree.
GEN_LOG_DIR="${GEN_LOG_DIR:-$LOG_ROOT/generate_completions/$ATTACK_METHOD/$MODEL_NAME}"
# Post-processing logs (conversion + eval submit) sit alongside generation logs
ACTORATTACK_POST_LOG_DIR="${ACTORATTACK_POST_LOG_DIR:-$GEN_LOG_DIR}"

ACTORATTACK_BEHAVIOR_CSV="${ACTORATTACK_BEHAVIOR_CSV:-$ROOT_DIR/toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv}"
ACTORATTACK_MAX_FALLBACK_BEHAVIORS="${ACTORATTACK_MAX_FALLBACK_BEHAVIORS:-0}"
ACTORATTACK_MIN_COMPLETION_BEHAVIORS="${ACTORATTACK_MIN_COMPLETION_BEHAVIORS:-$(python - "$ACTORATTACK_BEHAVIOR_CSV" <<'PY'
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
texts = set()
with csv_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        goal = (row.get("Goal") or row.get("Behavior") or "").strip()
        if goal:
            texts.add(goal)
print(len(texts))
PY
)}"
ACTORATTACK_EVAL_L2_PARTITION="${ACTORATTACK_EVAL_L2_PARTITION:-a30}"
ACTORATTACK_EVAL_L2_QOS="${ACTORATTACK_EVAL_L2_QOS:-standby}"
ACTORATTACK_EVAL_L2_TIME="${ACTORATTACK_EVAL_L2_TIME:-02:00:00}"
ACTORATTACK_EVAL_LG_PARTITION="${ACTORATTACK_EVAL_LG_PARTITION:-a30}"
ACTORATTACK_EVAL_LG_QOS="${ACTORATTACK_EVAL_LG_QOS:-standby}"
ACTORATTACK_EVAL_LG_TIME="${ACTORATTACK_EVAL_LG_TIME:-02:00:00}"
ACTORATTACK_EVAL_API_PARTITION="${ACTORATTACK_EVAL_API_PARTITION:-a30}"
ACTORATTACK_EVAL_API_QOS="${ACTORATTACK_EVAL_API_QOS:-standby}"
ACTORATTACK_EVAL_API_TIME="${ACTORATTACK_EVAL_API_TIME:-02:00:00}"
ACTORATTACK_API_FRESH_EVAL="${ACTORATTACK_API_FRESH_EVAL:-1}"
# Optional pre-attack job dependency (set by submit_actorattack_sweep.sh).
# When set, the gen job will not start until the pre-attack job succeeds.
ACTORATTACK_PREATTACK_DEP="${ACTORATTACK_PREATTACK_DEP:-}"
# Path to shared pre-attack result file (passed through to the SLURM job env).
ACTORATTACK_PRE_ATTACK_DATA_PATH="${ACTORATTACK_PRE_ATTACK_DATA_PATH:-}"

if [ "$ACTORATTACK_BACKEND" != "hf_local" ]; then
	echo "ERROR: only the hf_local backend is supported."
	echo "Set ACTORATTACK_BACKEND=hf_local (or leave it unset)."
	exit 1
fi

ACTORATTACK_CHUNK_SIZE="${ACTORATTACK_CHUNK_SIZE:-100}"

# Build optional dependency flag for the gen job sbatch call.
DEP_FLAG=""
if [ -n "$ACTORATTACK_PREATTACK_DEP" ]; then
        DEP_FLAG="--dependency=$ACTORATTACK_PREATTACK_DEP"
fi

mkdir -p "$GEN_LOG_DIR"
GEN_JOB_IDS=""
GEN_MSG=""

for START_IDX in $(seq 0 $ACTORATTACK_CHUNK_SIZE $(($ACTORATTACK_BEHAVIOR_LIMIT - 1))); do
    END_IDX=$(($START_IDX + $ACTORATTACK_CHUNK_SIZE))
    if [ $END_IDX -gt $ACTORATTACK_BEHAVIOR_LIMIT ]; then
        END_IDX=$ACTORATTACK_BEHAVIOR_LIMIT
    fi

    JNAME="AA-gen-${MODEL_NAME}-${START_IDX}"
    if [ -n "$ACTORATTACK_OUTPUT_ROOT" ]; then
        CKPT_FILE="$ACTORATTACK_OUTPUT_ROOT/checkpoints/$MODEL_NAME/inattack_${MODEL_NAME}_ckpt_${START_IDX}-${END_IDX}.jsonl"
    else
        CKPT_FILE="$ROOT_DIR/toolkits/ActorAttack/attack_result/checkpoints/$MODEL_NAME/inattack_${MODEL_NAME}_ckpt_${START_IDX}-${END_IDX}.jsonl"
    fi
    EXPECTED_LINES=$(($END_IDX - $START_IDX))

    if squeue -u "$USER" -h -o "%j" | grep -Eq "^${JNAME}(-|$)"; then
        echo "SKIP: $JNAME already in queue"
        continue
    fi

    if [ -f "$CKPT_FILE" ]; then
        CKPT_LINES=$(wc -l < "$CKPT_FILE" | awk '{print $1}')
        if [ "$CKPT_LINES" -ge "$EXPECTED_LINES" ]; then
            echo "SKIP: $JNAME checkpoint already complete ($CKPT_LINES/$EXPECTED_LINES lines)"
            continue
        fi

        echo "RESUME: $JNAME checkpoint has $CKPT_LINES/$EXPECTED_LINES lines"
    fi

    GEN_JOB_ID=$(sbatch --parsable \
        --partition="$ACTORATTACK_PARTITION" \
        --qos="$ACTORATTACK_QOS" \
        --gpus-per-node="$ACTORATTACK_GPUS_PER_NODE" \
        --time="$ACTORATTACK_TIME" \
        --mem="$ACTORATTACK_MEM" \
        --job-name="$JNAME" \
        --output="$GEN_LOG_DIR/${MODEL_NAME}_${START_IDX}-${END_IDX}_%j.log" \
        --export=ALL,MODEL_NAME="$MODEL_NAME",ACTORATTACK_LOCAL_MAX_NEW_TOKENS="$ACTORATTACK_LOCAL_MAX_NEW_TOKENS",ACTORATTACK_BEHAVIOR_START="$START_IDX",ACTORATTACK_BEHAVIOR_END="$END_IDX",ACTORATTACK_ACTOR_COUNT="$ACTORATTACK_ACTOR_COUNT",ACTORATTACK_BACKEND="$ACTORATTACK_BACKEND",ACTORATTACK_BEHAVIOR_CSV="$ACTORATTACK_BEHAVIOR_CSV",ACTORATTACK_PRE_ATTACK_DATA_PATH="$ACTORATTACK_PRE_ATTACK_DATA_PATH",ACTORATTACK_OUTPUT_DIR="$ACTORATTACK_OUTPUT_DIR",ACTORATTACK_CHECKPOINT_DIR="$ACTORATTACK_CHECKPOINT_DIR" \
        ${DEP_FLAG:+"$DEP_FLAG"} \
        "$JOB_SCRIPT")

    if [ -z "$GEN_JOB_IDS" ]; then
        GEN_JOB_IDS="$GEN_JOB_ID"
    else
        GEN_JOB_IDS="$GEN_JOB_IDS:$GEN_JOB_ID"
    fi
    GEN_MSG="$GEN_MSG $GEN_JOB_ID"
done

echo "Submitted ActorAttack generation chunks: $GEN_MSG"

if [ "$ACTORATTACK_SUBMIT_POST_CHAIN" = "1" ]; then
        if [ -z "$GEN_JOB_IDS" ]; then
                if [ "$ACTORATTACK_ALLOW_POST_ONLY" = "1" ]; then
                        echo "No generation chunks were submitted for $MODEL_NAME; submitting post-processing only."
                else
                        echo "No generation chunks were submitted for $MODEL_NAME; skipping dependent post job."
                        exit 0
                fi
        fi
        mkdir -p "$ACTORATTACK_POST_LOG_DIR"
        POST_JNAME="AA-post-${MODEL_NAME}"
        if squeue -u "$USER" -h -o "%j" | grep -Eq "^${POST_JNAME}(-|$)"; then
                echo "SKIP: $POST_JNAME already in queue"
                exit 0
        fi
        POST_LOG="$ACTORATTACK_POST_LOG_DIR/post_${MODEL_NAME}_%j.log"
        POST_WRAP="module --force purge && module load external anaconda/2025.06-py313 && source activate harmbench_env && cd $ROOT_DIR && ACTORATTACK_RAW_RESULT_ROOT='$ACTORATTACK_RAW_RESULT_ROOT' ACTORATTACK_RESULTS_ROOT='$RESULTS_ROOT' ACTORATTACK_ATTACK_NAME='$ATTACK_METHOD' python evaluation/actorattack/convert_actorattack_to_harmbench.py --model-config $MODEL_NAME --min-behaviors $ACTORATTACK_MIN_COMPLETION_BEHAVIORS --behavior-csv '$ACTORATTACK_BEHAVIOR_CSV' && MODEL_NAME=$MODEL_NAME ATTACK_METHOD=$ATTACK_METHOD RESULTS_ROOT=$RESULTS_ROOT LOG_ROOT=$LOG_ROOT MIN_COMPLETION_BEHAVIORS=$ACTORATTACK_MIN_COMPLETION_BEHAVIORS MAX_FALLBACK_BEHAVIORS=$ACTORATTACK_MAX_FALLBACK_BEHAVIORS FORCE_RESUBMIT=0 ACTORATTACK_EVAL_L2_PARTITION=$ACTORATTACK_EVAL_L2_PARTITION ACTORATTACK_EVAL_L2_QOS=$ACTORATTACK_EVAL_L2_QOS ACTORATTACK_EVAL_L2_TIME=$ACTORATTACK_EVAL_L2_TIME ACTORATTACK_EVAL_LG_PARTITION=$ACTORATTACK_EVAL_LG_PARTITION ACTORATTACK_EVAL_LG_QOS=$ACTORATTACK_EVAL_LG_QOS ACTORATTACK_EVAL_LG_TIME=$ACTORATTACK_EVAL_LG_TIME ACTORATTACK_EVAL_API_PARTITION=$ACTORATTACK_EVAL_API_PARTITION ACTORATTACK_EVAL_API_QOS=$ACTORATTACK_EVAL_API_QOS ACTORATTACK_EVAL_API_TIME=$ACTORATTACK_EVAL_API_TIME ACTORATTACK_API_FRESH_EVAL=$ACTORATTACK_API_FRESH_EVAL bash evaluation/actorattack/slurm/submit_actorattack_evals.sh"
        POST_DEP_FLAG=""
        if [ -n "$GEN_JOB_IDS" ]; then
                POST_DEP_FLAG="--dependency=afterok:$GEN_JOB_IDS"
        fi
        POST_JOB_ID=$(sbatch --parsable \
                ${POST_DEP_FLAG:+"$POST_DEP_FLAG"} \
                --partition="$ACTORATTACK_POST_PARTITION" \
                --account=YOUR_ACCOUNT \
                --qos="$ACTORATTACK_POST_QOS" \
                --nodes=1 \
                --gres=gpu:1 \
                --cpus-per-task=2 \
                --mem="$ACTORATTACK_POST_MEM" \
                --time="$ACTORATTACK_POST_TIME" \
                --job-name="AA-post-${MODEL_NAME}" \
                --output="$POST_LOG" \
                --wrap="$POST_WRAP")

        if [ -n "$POST_DEP_FLAG" ]; then
                echo "Submitted dependent post-processing job: $POST_JOB_ID (afterok:$GEN_JOB_IDS)"
        else
                echo "Submitted post-only post-processing job: $POST_JOB_ID"
        fi
fi
