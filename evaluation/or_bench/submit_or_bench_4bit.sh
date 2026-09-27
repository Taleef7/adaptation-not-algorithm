#!/bin/bash
# Submits the Base 4-bit OR-Bench cells: 5 families x {Hard-1K, Toxic}, generation (vLLM,
# bitsandbytes 4-bit) followed by GPT-4o-mini evaluation (afterok dependency).
# Outputs go to the regen4bit results namespace. Required env: PROJECT_ROOT

set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
LOG_DIR="${ROOT_DIR}/logs/or_bench"
mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

submit_cell() {
    local family="$1"
    local partition="$2"
    local subset="$3"

    local gen_job
    gen_job=$(sbatch --parsable \
        --partition="${partition}" \
        --job-name="orbg4b_${family}_${subset}" \
        --export=ALL,FAMILY="${family}",SUBSET_NAME="${subset}" \
        evaluation/or_bench/or_bench_4bit_generate.slurm)

    local eval_job
    eval_job=$(sbatch --parsable \
        --job-name="orbe4b_${family}_${subset}" \
        --dependency="afterok:${gen_job}" \
        --export=ALL,FAMILY="${family}",SUBSET_NAME="${subset}" \
        evaluation/or_bench/or_bench_4bit_evaluate.slurm)

    printf '%-12s %-8s gen=%s eval=%s\n' "${family}" "${subset}" "${gen_job}" "${eval_job}"
}

echo "Submitting base_4bit OR-Bench generation + evaluation (regen4bit namespace)"
submit_cell "gemma2"     "a10"       "Hard-1K"
submit_cell "gemma2"     "a10"       "Toxic"
submit_cell "llama31"    "a30"       "Hard-1K"
submit_cell "llama31"    "a30"       "Toxic"
submit_cell "phi4"       "a30"       "Hard-1K"
submit_cell "phi4"       "a30"       "Toxic"
submit_cell "qwen25_14b" "a100-80gb" "Hard-1K"
submit_cell "qwen25_14b" "a100-80gb" "Toxic"
submit_cell "qwen3"      "a10"       "Hard-1K"
submit_cell "qwen3"      "a10"       "Toxic"
echo "Done."
