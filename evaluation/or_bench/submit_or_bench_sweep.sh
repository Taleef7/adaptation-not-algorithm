#!/bin/bash
# OR-Bench sweep: for each family, one generation array job (5 configurations, HarmBench
# generate_completions.py, 512 new tokens) followed by one GPT-4o-mini evaluation array job
# (HarmBench evaluate_or_bench.py, OR-Bench three-way response-check prompt), for Hard-1K and Toxic.
# Required env: PROJECT_ROOT
set -euo pipefail

ROOT_DIR="${PROJECT_ROOT:?set PROJECT_ROOT}"
HARMBENCH_DIR="${ROOT_DIR}/toolkits/HarmBench"
LOG_DIR="${ROOT_DIR}/logs/or_bench"
cd "${ROOT_DIR}"

mkdir -p "${LOG_DIR}"

submit_family() {
    local family="$1"
    local partition="$2"
    local subset="$3"
    local gen_job_name="orbg_${family}_${subset}"
    local eval_job_name="orbe_${family}_${subset}"

    local gen_job
    gen_job=$(sbatch --parsable \
        --partition="${partition}" \
        --account=YOUR_ACCOUNT \
        --qos=standby \
        --export=ALL,FAMILY="${family}",SUBSET_NAME="${subset}" \
        evaluation/or_bench/or_bench_family_generate.slurm)

    local eval_job
    eval_job=$(sbatch --parsable \
        --partition=a30 \
        --account=YOUR_ACCOUNT \
        --qos=standby \
        --dependency="afterok:${gen_job}" \
        --export=ALL,FAMILY="${family}",SUBSET_NAME="${subset}" \
        evaluation/or_bench/or_bench_family_evaluate.slurm)

    printf '%-10s %-10s gen=%s eval=%s\n' "${family}" "${subset}" "${gen_job}" "${eval_job}"
}

echo "Submitting OR-Bench sweep"
submit_family "gemma2" "a10" "Hard-1K"
submit_family "gemma2" "a10" "Toxic"
submit_family "llama31" "a30" "Hard-1K"
submit_family "llama31" "a30" "Toxic"
submit_family "phi4" "a30" "Hard-1K"
submit_family "phi4" "a30" "Toxic"
submit_family "qwen25_14b" "a100-80gb" "Hard-1K"
submit_family "qwen25_14b" "a100-80gb" "Toxic"
submit_family "qwen3" "a10" "Hard-1K"
submit_family "qwen3" "a10" "Toxic"

echo "Done. Generation jobs are dependency parents for their paired evaluation jobs."
