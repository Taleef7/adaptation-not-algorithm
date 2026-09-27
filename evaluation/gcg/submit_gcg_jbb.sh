#!/bin/bash
# Fan out GCG step-1 (generate_test_cases) for one cell across JBB-100, one behavior per SLURM array task.
# Each task calls HarmBench scripts/generate_test_cases.sh for one behavior (1 GPU, standby), tasks 0..99.
# - PARTS is a comma-separated partition list; SLURM runs each task on whichever is free first.
#   Match PARTS to what the model fits: llama31 (8B, ~16GB) -> a10,a30,a100-40gb ; qwen25_14b (14B, ~28GB)
#   -> a100-40gb,a100-80gb.
# - overwrite=False => generate_test_cases.py skips behaviors whose test_cases.json already exists, so
#   submitting again only fills in missing behaviors.
# - 1 behavior/task keeps every task under the 4h standby wall on any of the listed GPUs.
# Usage: ./submit_gcg_jbb.sh <cell> <comma_partitions> [throttle]
#   e.g. ./submit_gcg_jbb.sh llama31_lora_custom a10,a30,a100-40gb 24
set -euo pipefail
cell=$1; PARTS=$2; THROTTLE=${3:-24}
REPO=${PROJECT_ROOT:?set PROJECT_ROOT}
HB=$REPO/toolkits/HarmBench
LOG=$REPO/logs/gcg_jbb; mkdir -p "$LOG"
JBB=./data/behavior_datasets/jbb_behaviors.csv          # 100 behaviors
SAVE=./results_jbb/GCG/${cell}/test_cases
N=100
# sanity: cell must be a resolvable GCG experiment
grep -qE "^${cell}:" "$HB/configs/method_configs/GCG_config.yaml" || { echo "ABORT: $cell not in GCG_config.yaml"; exit 1; }
JID=$(sbatch --parsable --account=YOUR_ACCOUNT --partition="$PARTS" --qos=standby --gres=gpu:1 \
  --cpus-per-task=6 --mem=64G --time=4:00:00 --array=0-$((N-1))%${THROTTLE} \
  --job-name="gcgJ_${cell}" --output="$LOG/${cell}_%a.log" \
  --wrap="cd $HB && i=\$SLURM_ARRAY_TASK_ID && bash scripts/generate_test_cases.sh GCG ${cell} $JBB $SAVE \$i \$((i+1)) '' '' False True")
echo "$cell: submitted GCG-JBB array $JID (0-$((N-1))%${THROTTLE}) on [$PARTS]"
