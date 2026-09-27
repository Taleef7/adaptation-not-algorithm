#!/bin/bash
#SBATCH --job-name=merge_qwen25_14b
#SBATCH --output=logs/finetune/qwen2.5-14b/merge_%j.log
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
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
echo "Merging Qwen2.5-14B adapters..."

cd ${PROJECT_ROOT}
python -u training/original_grid/qwen25/merge_adapters.py --adapter_type both

echo "Python script finished."
echo "Job finished at $(date)"
