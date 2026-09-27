#!/bin/bash
# Seed-42 PAIR step 2: generate target-model completions on the merged PAIR test cases.
# 18 jobs (9 cells x {JBB-100, HB-400}), 1 GPU each, 4h wall.
# Same settings as HarmBench scripts/generate_completions.sh used for the original-grid PAIR
# cells: --generate_with_vllm --max_new_tokens 512.
# --incremental_update lets a job resume (behaviors already completed are skipped).
# Prerequisite: merge_pair_seed42.py (test_cases.json present per experiment).
# Required env: PROJECT_ROOT, VENV_DIR, HF_HOME
set -euo pipefail
HB_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
VENV=${VENV_DIR}
JBB_BEH="$HB_DIR/data/behavior_datasets/jbb_behaviors.csv"
HB_BEH="$HB_DIR/data/behavior_datasets/harmbench_behaviors_text_all.csv"
LOG_ROOT="$HB_DIR/slurm_logs/PAIR_seed42_step2"
cd "$HB_DIR"; mkdir -p "$LOG_ROOT"

BASES=(llama31_lora llama31_qlora llama31_fft gemma2_lora gemma2_qlora gemma2_fft \
       qwen3_lora qwen3_qlora qwen3_fft)

submit_one () {
  local model_key="$1" exp_dir="$2" beh="$3" tag="$4"
  local tc="$exp_dir/test_cases/test_cases.json"
  local save="$exp_dir/completions/${model_key}.json"
  if [ ! -f "$tc" ]; then echo "SKIP_NO_TESTCASES $tag ($tc missing)"; return; fi
  mkdir -p "$exp_dir/completions"
  sbatch \
    --partition=a100-40gb --account=YOUR_ACCOUNT --qos=standby \
    --nodes=1 --gres=gpu:1 --mem=64G --time=4:00:00 \
    --job-name="ps42s2_${tag}" \
    --output="$LOG_ROOT/${tag}.log" --error="$LOG_ROOT/${tag}.log" \
    --wrap="
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME}
cd $HB_DIR
$VENV/bin/python -u generate_completions.py \
  --model_name $model_key \
  --behaviors_path $beh \
  --test_cases_path $tc \
  --save_path $save \
  --max_new_tokens 512 \
  --generate_with_vllm \
  --incremental_update
"
  echo "[SUBMITTED] step2 $tag -> $save"
}

for b in "${BASES[@]}"; do
  submit_one "${b}_seed42_custom" "results_phase5/PAIR_jbb/${b}_seed42_pair_experiment" "$JBB_BEH" "jbb_${b}"
done
for b in "${BASES[@]}"; do
  submit_one "${b}_seed42_custom" "results_phase5/PAIR_hb/test_${b}_seed42_custom" "$HB_BEH" "hb_${b}"
done
echo "Monitor: squeue -u $USER | grep ps42s2"
echo "After all complete, verify counts (JBB=100, HB=400) then run submit_pair_seed42_step3.sh"
