#!/bin/bash
# Seed-42 PAIR step 1 (generate_test_cases) for the 9 controlled-grid cells of Llama-3.1,
# Gemma-2 and Qwen-3 on JBB-100.
# Uses PAIR_config_jbb.yaml (explicit entries) + jbb_behaviors.csv (100 behaviors).
# 10 behaviors/chunk = 10 jobs/cell x 9 cells = 90 jobs, 2 GPUs each, 4h wall.
#
# After all jobs finish, run merge_pair_seed42.py, then submit_pair_seed42_step2.sh.
# Required env: PROJECT_ROOT, VENV_DIR, HF_HOME

set -euo pipefail

HB_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
JBB_BEHAVIORS="$HB_DIR/data/behavior_datasets/jbb_behaviors.csv"
PAIR_CONFIG="$HB_DIR/configs/method_configs/PAIR_config_jbb.yaml"
LOG_ROOT="$HB_DIR/slurm_logs_jbb/generate_test_cases/PAIR_seed42"
SAVE_ROOT="$HB_DIR/results_phase5/PAIR_jbb"

EXPERIMENTS=(
  "llama31_lora_seed42_pair_experiment"
  "llama31_qlora_seed42_pair_experiment"
  "llama31_fft_seed42_pair_experiment"
  "gemma2_lora_seed42_pair_experiment"
  "gemma2_qlora_seed42_pair_experiment"
  "gemma2_fft_seed42_pair_experiment"
  "qwen3_lora_seed42_pair_experiment"
  "qwen3_qlora_seed42_pair_experiment"
  "qwen3_fft_seed42_pair_experiment"
)

cd "$HB_DIR"
mkdir -p "$LOG_ROOT"

total=0
for exp in "${EXPERIMENTS[@]}"; do
  LOG_DIR="$LOG_ROOT/$exp"
  mkdir -p "$LOG_DIR"
  SAVE_DIR="$SAVE_ROOT/$exp/test_cases"
  mkdir -p "$SAVE_DIR"

  for start_idx in $(seq 0 10 90); do
    end_idx=$((start_idx + 10))

    sbatch \
      --partition=a100-40gb \
      --account=YOUR_ACCOUNT \
      --qos=standby \
      --nodes=1 \
      --gres=gpu:2 \
      --mem=90G \
      --time=4:00:00 \
      --job-name="ps42j1_${exp:0:20}_${start_idx}" \
      --output="$LOG_DIR/start_${start_idx}.log" \
      --error="$LOG_DIR/start_${start_idx}.log" \
      --wrap="
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME}
cd $HB_DIR
${VENV_DIR}/bin/python -u generate_test_cases.py \
  --method_name PAIR \
  --experiment_name $exp \
  --method_config_file $PAIR_CONFIG \
  --behaviors_path $JBB_BEHAVIORS \
  --save_dir $SAVE_DIR \
  --behavior_start_idx $start_idx \
  --behavior_end_idx $end_idx \
  --overwrite
"
    total=$((total + 1))
  done
  echo "[SUBMITTED] $exp (10 jobs)"
done

echo ""
echo "Total jobs submitted: $total  (${#EXPERIMENTS[@]} cells × 10 chunks)"
echo "Monitor: squeue -u $USER | grep ps42j1"
echo ""
echo "After all complete, check coverage:"
echo "  for exp in ${EXPERIMENTS[*]}; do"
echo "    echo \"\$exp: \$(ls $SAVE_ROOT/\$exp/test_cases/test_cases_individual_behaviors/ 2>/dev/null | wc -l)/100\""
echo "  done"
