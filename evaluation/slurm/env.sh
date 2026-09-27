#!/bin/bash
# Common environment for the SLURM jobs in evaluation/. Source it at the start of a job.
# Set these for your cluster before use:
#   PROJECT_ROOT : checkout of the research repository (contains toolkits/HarmBench, models/, ...)
#   VENV_DIR     : Python environment with HarmBench requirements, vLLM and transformers
#   HF_HOME      : Hugging Face cache (models, datasets)
: "${PROJECT_ROOT:?set PROJECT_ROOT}"
: "${VENV_DIR:?set VENV_DIR}"
export PYTHONNOUSERSITE=1            # keep user site-packages out of the job environment
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME"
