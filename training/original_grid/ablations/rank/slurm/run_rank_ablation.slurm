#!/bin/bash
#SBATCH --job-name=lora_rank_ablation
#SBATCH --output=logs/finetune/llama3.1/rank_ablation_%j.txt 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00 
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=a100-80gb
#SBATCH --qos=standby 
#SBATCH --mem=90G

set -eo pipefail

echo "=============================================="
echo "LoRA Rank Ablation: Training r={4, 8, 32, 64}"
echo "Job ID: $SLURM_JOBID"
echo "Node: $(hostname)"
echo "=============================================="

mkdir -p logs/finetune/llama3.1 models/llama3.1 outputs/finetune/llama3.1

module --force purge
module load external
module load anaconda/2025.06-py313
source activate thesis_env
set -u

cd ${PROJECT_ROOT}

require_adapter_dir() {
	local adapter_dir="$1"
	if [ ! -d "$adapter_dir" ] || [ ! -f "$adapter_dir/adapter_config.json" ] || [ ! -f "$adapter_dir/adapter_model.safetensors" ]; then
		echo "ERROR: Expected adapter output missing or incomplete: $adapter_dir"
		exit 1
	fi
}

run_stage_if_needed() {
	local label="$1"
	local train_cmd="$2"
	local adapter_dir="$3"

	if [ -f "$adapter_dir/adapter_config.json" ] && [ -f "$adapter_dir/adapter_model.safetensors" ]; then
		echo "==> SKIP: $label already complete ($adapter_dir)"
		return
	fi

	echo "==> Training $label..."
	rm -rf "$adapter_dir"
	eval "$train_cmd"
	require_adapter_dir "$adapter_dir"
	echo "==> $label complete."
}

echo ""
run_stage_if_needed "r=4" "python -u training/original_grid/ablations/rank/finetune_lora_r4.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-r4"

echo ""
run_stage_if_needed "r=8" "python -u training/original_grid/ablations/rank/finetune_lora_r8.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-r8"

echo ""
run_stage_if_needed "r=32" "python -u training/original_grid/ablations/rank/finetune_lora_r32.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-r32"

echo ""
run_stage_if_needed "r=64" "python -u training/original_grid/ablations/rank/finetune_lora_r64.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-r64"

echo ""
echo "=============================================="
echo "All rank ablation training complete!"
echo "Job finished at $(date)"
echo "=============================================="
