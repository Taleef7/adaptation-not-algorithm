#!/bin/bash
set -euo pipefail

ROOT="${PROJECT_ROOT:?set PROJECT_ROOT to the repository root}"
cd "$ROOT"

for variant in main seed0 seed123; do
  echo "Submitting faithful SafeLoRA variant: $variant"
  sbatch --export=ALL,SAFE_LORA_VARIANT="$variant" \
    --job-name="safelora_${variant}" \
    training/original_grid/ablations/safelora/slurm/safe_lora_faithful_pipeline.slurm
done
