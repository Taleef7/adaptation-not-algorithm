#!/bin/bash
#SBATCH --job-name=qlora_qwen25_14b_alpaca
#SBATCH --output=logs/finetune/qwen2.5-14b/qlora_finetune_%j.log
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --mem=150G

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOBID"

module --force purge
module load external
module load anaconda
conda activate thesis_env

echo "Conda environment activated."
echo "Running Qwen2.5-14B QLoRA script..."

cd ${PROJECT_ROOT}
python -u training/original_grid/qwen25/finetune_qlora_qwen2.5-14b.py

echo "Python script finished."
echo "Job finished at $(date)"
