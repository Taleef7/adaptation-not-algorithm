#!/bin/bash
# GPT-4o-mini scoring (secondary_score.py --evaluator gpt4omini) for the MixAT ladder: the same
# 12 completion files as submit_mixat_llamaguard.sh (6 cells x 2 attacks), responses clipped to
# 512 gpt2 tokens. API calls only (no GPU); run on a node with outbound HTTPS and OPENAI_API_KEY set.
# Required env: PROJECT_ROOT, VENV_DIR, OPENAI_API_KEY
set -euo pipefail
: "${PROJECT_ROOT:?set PROJECT_ROOT}"; : "${VENV_DIR:?set VENV_DIR}"
REPO=$PROJECT_ROOT
VENV=$VENV_DIR
HB=$REPO/toolkits/HarmBench
AP=$HB/results_phase5/artprompt_tokchat
DI=$HB/results_phase5/deepinception
OUT=$HB/results_phase5/scored/mixat_gpt4omini
MAN=$REPO/logs/mixat/mixat_gpt4omini_manifest.txt
mkdir -p "$(dirname "$MAN")"
mkdir -p "$OUT"

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
    echo "$src $OUT/${cell}_${attack}_jbb_gpt4omini.jsonl" >> "$MAN"
  done
done
echo "manifest ($(wc -l < "$MAN") files) written to $MAN"

export PYTHONNOUSERSITE=1
cd "$REPO"
$VENV/bin/python -u "$REPO/evaluation/seed42_grid/secondary_score.py" \
  --manifest "$MAN" --evaluator gpt4omini
echo MIXAT_GPT4OMINI_DONE
