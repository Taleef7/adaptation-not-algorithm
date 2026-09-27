#!/bin/bash
# Phase 1 (pre-attack) only — generates social actors + opening queries for all
# behaviors using the attacker model (Mistral). Target model is NOT loaded.
# Output is written to a canonical fixed path so all 25 in-attack jobs can
# reference it without needing to know the timestamped filename.
#
# Resources: one a100-80gb GPU, 8h wall for 400 behaviors (set by submit_actorattack_sweep.sh).
#
# Checkpointing: each behavior result is written to a JSONL checkpoint file as it
# completes. If this job is killed and resubmitted, already-done behaviors are
# skipped automatically. Monitor progress via [X/N Y%] log lines.
#
# This job must complete before any in-attack jobs start (enforced by afterok
# dependencies in submit_actorattack_sweep.sh).
#SBATCH --job-name=AA-preattack
#SBATCH --output=logs/actorattack_preattack_%j.log
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
  set -a; . ".env"; set +a
fi

export MODEL_NAME="${MODEL_NAME:-preattack_shared}"
export ACTORATTACK_BEHAVIOR_LIMIT="${ACTORATTACK_BEHAVIOR_LIMIT:-400}"
export ACTORATTACK_ACTOR_COUNT="${ACTORATTACK_ACTOR_COUNT:-2}"
export ACTORATTACK_ATTACKER_MODEL_NAME="${ACTORATTACK_ATTACKER_MODEL_NAME:-mistral_7b_v3_attacker_custom}"
export ACTORATTACK_BEHAVIOR_CSV="${ACTORATTACK_BEHAVIOR_CSV:-$ROOT_DIR/toolkits/HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv}"

# Canonical output path — stable, no timestamp — referenced by all 25 in-attack jobs.
# Allow callers to override the root so JBB can write into its own result tree.
CANONICAL_PATH="${ACTORATTACK_PRE_ATTACK_OUTPUT_PATH:-$ROOT_DIR/toolkits/ActorAttack/pre_attack_result/shared_${ACTORATTACK_ATTACKER_MODEL_NAME}_${ACTORATTACK_BEHAVIOR_LIMIT}.json}"
export ACTORATTACK_PRE_ATTACK_OUTPUT_PATH="$CANONICAL_PATH"

# Checkpoint dir — JSONL written per-behavior; survives job restarts.
# Allow callers to override the root so JBB can keep its own checkpoints.
CKPT_DIR="${ACTORATTACK_CHECKPOINT_DIR:-$ROOT_DIR/toolkits/ActorAttack/pre_attack_result/checkpoints}"
export ACTORATTACK_CHECKPOINT_DIR="$CKPT_DIR"
mkdir -p "$CKPT_DIR"

echo "========================================="
echo "ActorAttack pre-attack phase (shared)"
echo "  SLURM job   : $SLURM_JOB_ID"
echo "  behaviors   : $ACTORATTACK_BEHAVIOR_LIMIT"
echo "  actors      : $ACTORATTACK_ACTOR_COUNT"
echo "  attacker    : $ACTORATTACK_ATTACKER_MODEL_NAME"
echo "  csv         : $ACTORATTACK_BEHAVIOR_CSV"
echo "  output      : $CANONICAL_PATH"
echo "  checkpoint  : $CKPT_DIR"
echo "========================================="

python -u -m evaluation.actorattack.run_actorattack \
  --backend hf_local \
  --questions "$ACTORATTACK_BEHAVIOR_LIMIT" \
  --actors "$ACTORATTACK_ACTOR_COUNT" \
  --behavior "$ACTORATTACK_BEHAVIOR_CSV" \
  --attack_model_name "$ACTORATTACK_ATTACKER_MODEL_NAME" \
  --target_model_name "none" \
  --pre_attack_only \
  --pre_attack_output_path "$CANONICAL_PATH" \
  --checkpoint_dir "$CKPT_DIR"

echo "Pre-attack phase complete. Canonical output: $CANONICAL_PATH"
