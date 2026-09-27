#!/bin/bash
# Seed-42 PAIR step 3: HB-Cls (cais/HarmBench-Llama-2-13b-cls) scoring of the step-2 completions.
# 18 jobs (9 cells x {JBB-100, HB-400}), 1 GPU each, 4h wall.
# Same call as HarmBench scripts/evaluate_completions.sh (--include_advbench_metric,
# num_tokens 512); writes results/<key>.json with per-behavior labels and an ASR summary.
# A completions file is scored only if it covers every behavior (JBB=100, HB=400).
# Prerequisite: step 2 complete.
# Required env: PROJECT_ROOT, VENV_DIR, HF_HOME
set -euo pipefail
HB_DIR="${PROJECT_ROOT}/toolkits/HarmBench"
VENV=${VENV_DIR}
CLS="cais/HarmBench-Llama-2-13b-cls"
JBB_BEH="$HB_DIR/data/behavior_datasets/jbb_behaviors.csv"
HB_BEH="$HB_DIR/data/behavior_datasets/harmbench_behaviors_text_all.csv"
LOG_ROOT="$HB_DIR/slurm_logs/PAIR_seed42_step3"
cd "$HB_DIR"; mkdir -p "$LOG_ROOT"

BASES=(llama31_lora llama31_qlora llama31_fft gemma2_lora gemma2_qlora gemma2_fft \
       qwen3_lora qwen3_qlora qwen3_fft)

submit_one () {
  local model_key="$1" exp_dir="$2" beh="$3" expected="$4" tag="$5"
  local comp="$exp_dir/completions/${model_key}.json"
  local save="$exp_dir/results/${model_key}.json"
  if [ ! -f "$comp" ]; then echo "SKIP_NO_COMPLETIONS $tag ($comp missing)"; return; fi
  local n
  n=$("$VENV/bin/python" -c "import json;print(len(json.load(open('$comp'))))" 2>/dev/null || echo -1)
  if [ "$n" -ne "$expected" ]; then
    echo "SKIP_INCOMPLETE $tag (completions n=$n expected=$expected)"; return
  fi
  mkdir -p "$exp_dir/results"
  sbatch \
    --partition=a100-40gb --account=YOUR_ACCOUNT --qos=standby \
    --nodes=1 --gres=gpu:1 --mem=48G --time=4:00:00 \
    --job-name="ps42s3_${tag}" \
    --output="$LOG_ROOT/${tag}.log" --error="$LOG_ROOT/${tag}.log" \
    --wrap="
export PYTHONNOUSERSITE=1
export HF_HOME=${HF_HOME}
cd $HB_DIR
$VENV/bin/python -u evaluate_completions.py \
  --cls_path $CLS \
  --behaviors_path $beh \
  --completions_path $comp \
  --save_path $save \
  --include_advbench_metric
"
  echo "[SUBMITTED] step3 $tag ($n behaviors) -> $save"
}

for b in "${BASES[@]}"; do
  submit_one "${b}_seed42_custom" "results_phase5/PAIR_jbb/${b}_seed42_pair_experiment" "$JBB_BEH" 100 "jbb_${b}"
done
for b in "${BASES[@]}"; do
  submit_one "${b}_seed42_custom" "results_phase5/PAIR_hb/test_${b}_seed42_custom" "$HB_BEH" 400 "hb_${b}"
done
echo "Monitor: squeue -u $USER | grep ps42s3"
echo "Results land in <exp>/results/<key>.json (per-behavior label + ASR summary)."
