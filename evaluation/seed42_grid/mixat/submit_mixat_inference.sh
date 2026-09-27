#!/bin/bash
# MixAT defended-base comparison (Llama-3.1, JBB-100): completions for the six-cell ladder.
#
#   level   undefended                                defended (MixAT merged into the base)
#   ------  ----------------------------------------  -----------------------------------------
#   base    llama31_base  (Llama-3.1-8B-Instruct)     llama31_mixat_base
#   +LoRA   llama31_lora  (seed-42 controlled grid)   llama31_mixat_lora (same seed-42 recipe)
#   +FFT    llama31_fft   (seed-42 controlled grid)   llama31_mixat_fft  (same seed-42 recipe)
#
# DeepInception: deepinception_phase5.py for the four cells that are not already part of the
#   controlled grid (the undefended LoRA/FFT cells reuse results_phase5/deepinception/*_seed42_jbb).
# ArtPrompt: artprompt_phase5.py with --template tokenizer (native Llama-3.1 chat template) for all
#   six cells, written to results_phase5/artprompt_tokchat/. Each defended cell reuses the test
#   cases of its undefended counterpart, so the attack prompts are identical within each pair.
#
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
REPO=$PROJECT_ROOT
VENV=$VENV_DIR
HB=$REPO/toolkits/HarmBench
DRIVERS=$REPO/evaluation/seed42_grid
LOG=$REPO/logs/mixat
DI=$HB/results_phase5/deepinception
APTOK=$HB/results_phase5/artprompt_tokchat
JBB_CSV=$HB/data/behavior_datasets/jbb_behaviors.csv
mkdir -p "$DI" "$APTOK" "$LOG"

declare -A MODEL=(
  [llama31_base]=$REPO/models/llama3.1/llama-3.1-8b-instruct-base
  [llama31_mixat_base]=$REPO/models/mixat/llama31_mixat_base
  [llama31_lora]=$REPO/models/phase5/llama31_lora_seed42_merged
  [llama31_mixat_lora]=$REPO/models/phase5/llama31_mixat_lora_seed42_merged
  [llama31_fft]=$REPO/models/phase5/llama31_fft_seed42_merged
  [llama31_mixat_fft]=$REPO/models/phase5/llama31_mixat_fft_seed42_merged
)
# cell -> original-grid experiment whose ArtPrompt test cases are reused (the undefended counterpart)
declare -A TC=(
  [llama31_base]=llama31_base_fp16      [llama31_mixat_base]=llama31_base_fp16
  [llama31_lora]=llama31_lora           [llama31_mixat_lora]=llama31_lora
  [llama31_fft]=llama31_fft             [llama31_mixat_fft]=llama31_fft
)
# DeepInception is generated here only for cells outside the controlled grid.
declare -A NEEDS_DI=( [llama31_base]=1 [llama31_mixat_base]=1 [llama31_mixat_lora]=1 [llama31_mixat_fft]=1 )

for cell in llama31_base llama31_mixat_base llama31_lora llama31_mixat_lora llama31_fft llama31_mixat_fft; do
  model="${MODEL[$cell]}"
  tc="$HB/results_jbb/ArtPrompt/${TC[$cell]}_custom/test_cases/test_cases.json"
  RUN=""
  if [ -n "${NEEDS_DI[$cell]:-}" ]; then
    RUN="$VENV/bin/python -u $DRIVERS/deepinception_phase5.py --model_path $model --benchmark jbb --out $DI/${cell}_jbb.jsonl && "
  fi
  RUN="${RUN}$VENV/bin/python -u $DRIVERS/artprompt_phase5.py --model_path $model --test_cases $tc \
--behaviors_csv $JBB_CSV --template tokenizer --benchmark jbb --out $APTOK/${cell}_jbb.jsonl && \
echo MIXAT_INFER_DONE_${cell}"
  # 8B bf16 (~16 GB) fits a 24 GB A30; larger partitions are listed as alternatives.
  jid=$(sbatch --parsable --account=YOUR_ACCOUNT --partition=a30,a100-40gb,a100-80gb --qos=standby \
    --gres=gpu:1 --cpus-per-task=6 --mem=64G --time=2:00:00 \
    --job-name="mixat_inf_${cell}" --output="$LOG/infer_${cell}_%j.log" \
    --wrap="export PYTHONNOUSERSITE=1 && cd $REPO && $RUN")
  echo "submitted MixAT inference ${cell}: $jid"
done
