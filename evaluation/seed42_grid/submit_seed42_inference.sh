#!/bin/bash
# Controlled-grid (seed-42) inference: DeepInception + ArtPrompt on JBB-100 and HarmBench-400
# for the 15 re-trained {LoRA, QLoRA, FFT} cells of the five families.
#
#   DeepInception : deepinception_phase5.py (chat template, T=0.6, top_p=0.9, 1024 new tokens)
#   ArtPrompt     : artprompt_phase5.py, reusing each original-grid cell's ArtPrompt test cases
#                   (target-independent), Alpaca template, greedy, 512 new tokens
#
# One SLURM job per cell (8B-14B bf16 fits a single 40 GB A100). Scoring is run separately
# (score_seed42_hbcls.sh, build_secondary_manifests.sh + secondary_eval.slurm).
#
# Usage: bash submit_seed42_inference.sh [cell ...]      (default: all 15 cells)
# Required env: PROJECT_ROOT, VENV_DIR (HarmBench/vLLM environment)
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
HB=$PROJECT_ROOT/toolkits/HarmBench
LOG=$PROJECT_ROOT/logs/seed42_grid
DI=$HB/results_phase5/deepinception
AP=$HB/results_phase5/artprompt
JBB_CSV=$HB/data/behavior_datasets/jbb_behaviors.csv
HB_CSV=$HB/data/behavior_datasets/harmbench_behaviors_text_all.csv
DRIVERS=$PROJECT_ROOT/evaluation/seed42_grid
mkdir -p "$DI" "$AP" "$LOG"

# cell -> original-grid HarmBench experiment name whose ArtPrompt test cases are reused
declare -A TCNAME=(
  [llama31_lora]=llama31_lora [llama31_qlora]=llama31_qlora [llama31_fft]=llama31_fft
  [gemma2_lora]=gemma2_lora   [gemma2_qlora]=gemma2_qlora   [gemma2_fft]=gemma2_fft
  [qwen3_lora]=qwen3_lora     [qwen3_qlora]=qwen3_qlora     [qwen3_fft]=qwen3_fft
  [phi4_lora]=phi4_lora       [phi4_qlora]=phi4_qlora       [phi4_fft]=phi4_fft
  [qwen25_lora]=qwen25_14b_lora [qwen25_qlora]=qwen25_14b_qlora [qwen25_fft]=qwen25_14b_fft
)
ALL_CELLS="llama31_lora llama31_qlora llama31_fft gemma2_lora gemma2_qlora gemma2_fft \
qwen3_lora qwen3_qlora qwen3_fft phi4_lora phi4_qlora phi4_fft qwen25_lora qwen25_qlora qwen25_fft"
CELLS="${*:-$ALL_CELLS}"

submit_cell() {
  local cell=$1
  local model="$PROJECT_ROOT/models/phase5/${cell}_seed42_merged"
  local tc="${TCNAME[$cell]}"
  local DIpy="$VENV_DIR/bin/python -u $DRIVERS/deepinception_phase5.py"
  local APpy="$VENV_DIR/bin/python -u $DRIVERS/artprompt_phase5.py --template alpaca"
  local RUN="\
$DIpy --model_path $model --benchmark jbb       --out $DI/${cell}_seed42_jbb.jsonl && \
$DIpy --model_path $model --benchmark harmbench --behaviors_csv $HB_CSV --out $DI/${cell}_seed42_harmbench.jsonl && \
$APpy --model_path $model --test_cases $HB/results_jbb/ArtPrompt/${tc}_custom/test_cases/test_cases.json --behaviors_csv $JBB_CSV --benchmark jbb       --out $AP/${cell}_seed42_jbb.jsonl && \
$APpy --model_path $model --test_cases $HB/results/ArtPrompt/${tc}_custom/test_cases/test_cases.json     --behaviors_csv $HB_CSV  --benchmark harmbench --out $AP/${cell}_seed42_harmbench.jsonl && \
echo SEED42_INFER_DONE_${cell}"
  sbatch --parsable --account=YOUR_ACCOUNT --partition=a100-40gb --qos=standby \
    --gres=gpu:1 --cpus-per-task=6 --mem=80G --time=4:00:00 \
    --job-name="s42_inf_${cell}" --output="$LOG/infer_${cell}_%j.log" \
    --wrap="export PYTHONNOUSERSITE=1 && cd $PROJECT_ROOT && $RUN"
}

for cell in $CELLS; do
  echo "submitted seed-42 inference $cell: $(submit_cell "$cell")"
done
