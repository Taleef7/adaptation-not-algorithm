#!/bin/bash
# In-attack generation SLURM script for the full ActorAttack sweep (400 HarmBench behaviors).
# Partition, QOS, time, and mem are injected by submit_actorattack_sweep.sh via sbatch flags.
# Phase 1 (pre-attack) is always skipped here — ACTORATTACK_PRE_ATTACK_DATA_PATH points to
# the shared canonical file produced by the AA-preattack job.
#
# Checkpointing: each behavior result is written to a per-model JSONL checkpoint as it
# completes. If this job is killed and resubmitted, already-done behaviors are skipped.
# Monitor progress via [X/N Y%] log lines and SUCCESS/no-jailbreak per behavior.
#SBATCH --job-name=actorattack_sweep
#SBATCH --output=logs/actorattack_gen_%x_%j.log
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --account=YOUR_ACCOUNT

set -eo pipefail

module --force purge
module load external
module load anaconda/2025.06-py313
source activate harmbench_env
set -u

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  . ".env"
  set +a
fi

export MODEL_NAME="${MODEL_NAME:-llama31_base_fp16_custom}"
export ACTORATTACK_BEHAVIOR_START="${ACTORATTACK_BEHAVIOR_START:-0}"
export ACTORATTACK_BEHAVIOR_END="${ACTORATTACK_BEHAVIOR_END:--1}"
export ACTORATTACK_ACTOR_COUNT="${ACTORATTACK_ACTOR_COUNT:-2}"
# Attacker: Mistral-7B-Instruct-v0.3.
export ACTORATTACK_ATTACKER_MODEL_NAME="${ACTORATTACK_ATTACKER_MODEL_NAME:-mistral_7b_v3_attacker_custom}"
export ACTORATTACK_ATTACKER_MODEL_PATH="${ACTORATTACK_ATTACKER_MODEL_PATH:-}"
export ACTORATTACK_BEHAVIOR_CSV="${ACTORATTACK_BEHAVIOR_CSV:-$ROOT_DIR/toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv}"
export ACTORATTACK_BACKEND="${ACTORATTACK_BACKEND:-hf_local}"
export ACTORATTACK_LOCAL_MAX_NEW_TOKENS="${ACTORATTACK_LOCAL_MAX_NEW_TOKENS:-512}"
# Shared pre-attack result path (set by submit_actorattack_sweep.sh).
# When present, Phase 1 is skipped and this file is used directly.
export ACTORATTACK_PRE_ATTACK_DATA_PATH="${ACTORATTACK_PRE_ATTACK_DATA_PATH:-}"

# Checkpoint dir — one subdir per model so parallel jobs don't collide.
# JSONL written per-behavior; a resubmitted job picks up from where it left off.
CKPT_DIR="${ACTORATTACK_CHECKPOINT_DIR:-$ROOT_DIR/toolkits/ActorAttack/attack_result/checkpoints/$MODEL_NAME}"
mkdir -p "$CKPT_DIR"
export ACTORATTACK_CHECKPOINT_DIR="$CKPT_DIR"

if [ "$ACTORATTACK_BACKEND" != "hf_local" ]; then
  echo "ERROR: only the hf_local backend is supported."
  exit 1
fi

echo "========================================="
echo "ActorAttack in-attack sweep"
echo "  SLURM job   : $SLURM_JOB_ID"
echo "  model       : $MODEL_NAME"
echo "  start_idx   : $ACTORATTACK_BEHAVIOR_START"
echo "  end_idx     : $ACTORATTACK_BEHAVIOR_END"
echo "  actors      : $ACTORATTACK_ACTOR_COUNT"
echo "  attacker    : $ACTORATTACK_ATTACKER_MODEL_NAME"
echo "  csv         : $ACTORATTACK_BEHAVIOR_CSV"
echo "  checkpoint  : $CKPT_DIR"
if [ -n "$ACTORATTACK_PRE_ATTACK_DATA_PATH" ]; then
  echo "  pre-attack  : $ACTORATTACK_PRE_ATTACK_DATA_PATH (shared, Phase 1 skipped)"
fi
echo "========================================="

python -u -m evaluation.actorattack.run_actorattack \
  --backend "$ACTORATTACK_BACKEND" \
  --questions "-1" \
  --behavior_start "$ACTORATTACK_BEHAVIOR_START" \
  --behavior_end "$ACTORATTACK_BEHAVIOR_END" \
  --actors "$ACTORATTACK_ACTOR_COUNT" \
  --behavior "$ACTORATTACK_BEHAVIOR_CSV" \
  --attack_model_name "$ACTORATTACK_ATTACKER_MODEL_NAME" \
  --target_model_name "$MODEL_NAME" \
  --early_stop True \
  --dynamic_modify True \
  --checkpoint_dir "$CKPT_DIR" \
  ${ACTORATTACK_PRE_ATTACK_DATA_PATH:+--pre_attack_data_path "$ACTORATTACK_PRE_ATTACK_DATA_PATH"}

echo "ActorAttack sweep generation completed for $MODEL_NAME."
