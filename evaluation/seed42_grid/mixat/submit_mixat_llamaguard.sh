#!/bin/bash
# LlamaGuard-3 scoring (secondary_score.py --evaluator llamaguard) for the MixAT ladder:
# 6 cells {base, lora, fft} x {undefended, MixAT-defended} x {ArtPrompt, DeepInception} = 12 files,
# scored in one job in --manifest mode so the 8B judge is loaded once. ArtPrompt files are the
# tokenizer-template completions (results_phase5/artprompt_tokchat/) for both arms; undefended
# LoRA/FFT DeepInception files are the controlled-grid completions.
# Required env: PROJECT_ROOT, VENV_DIR
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
REPO=$PROJECT_ROOT
VENV=$VENV_DIR
HB=$REPO/toolkits/HarmBench
LOG=$REPO/logs/mixat
AP=$HB/results_phase5/artprompt_tokchat
DI=$HB/results_phase5/deepinception
OUT=$HB/results_phase5/scored/mixat_llamaguard
MAN=$LOG/mixat_llamaguard_manifest.txt
mkdir -p "$OUT" "$LOG"

# cell -> ArtPrompt (tokenizer template) file, DeepInception file
declare -A APF=(
  [base]=$AP/llama31_base_jbb.jsonl              [mixat_base]=$AP/llama31_mixat_base_jbb.jsonl
  [lora]=$AP/llama31_lora_jbb.jsonl              [mixat_lora]=$AP/llama31_mixat_lora_jbb.jsonl
  [fft]=$AP/llama31_fft_jbb.jsonl                [mixat_fft]=$AP/llama31_mixat_fft_jbb.jsonl
)
declare -A DIF=(
  [base]=$DI/llama31_base_jbb.jsonl              [mixat_base]=$DI/llama31_mixat_base_jbb.jsonl
  [lora]=$DI/llama31_lora_seed42_jbb.jsonl       [mixat_lora]=$DI/llama31_mixat_lora_jbb.jsonl
  [fft]=$DI/llama31_fft_seed42_jbb.jsonl         [mixat_fft]=$DI/llama31_mixat_fft_jbb.jsonl
)

: > "$MAN"
for cell in base mixat_base lora mixat_lora fft mixat_fft; do
  for attack in artprompt deepinception; do
    if [ "$attack" = artprompt ]; then src="${APF[$cell]}"; else src="${DIF[$cell]}"; fi
    [ -s "$src" ] || { echo "FATAL missing completions: $src" >&2; exit 1; }
    echo "$src $OUT/${cell}_${attack}_jbb_llamaguard.jsonl" >> "$MAN"
  done
done
echo "manifest ($(wc -l < "$MAN") files):"; cat "$MAN"

# LlamaGuard-3-8B bf16 (~16 GB) fits a 24 GB A30. Greedy; resumes from partial outputs.
jid=$(sbatch --parsable --account=YOUR_ACCOUNT --partition=a30,a100-40gb,a100-80gb --qos=standby \
  --gres=gpu:1 --cpus-per-task=6 --mem=48G --time=01:30:00 \
  --job-name="mixat_lg" --output="$LOG/mixat_llamaguard_%j.log" \
  --wrap="export PYTHONNOUSERSITE=1 && cd $REPO && \
$VENV/bin/python -u $REPO/evaluation/seed42_grid/secondary_score.py \
  --manifest $MAN --evaluator llamaguard --lg_model $REPO/models/llama-guard-3-8b && \
echo MIXAT_LLAMAGUARD_DONE")
echo "submitted MixAT LlamaGuard batch: $jid"
