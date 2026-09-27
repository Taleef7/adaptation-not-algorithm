#!/bin/bash
#SBATCH --job-name=lora_layer_specific
#SBATCH --output=logs/finetune/llama3.1/layer_specific_%j.txt 
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
echo "Layer-Specific LoRA: Attention-Only + MLP-Only"
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
run_stage_if_needed "Attention-Only LoRA (q,k,v,o_proj)" "python -u training/original_grid/ablations/layer/finetune_lora_attn_only.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-attn-only"

echo ""
run_stage_if_needed "MLP-Only LoRA (gate,up,down_proj)" "python -u training/original_grid/ablations/layer/finetune_lora_mlp_only.py" "models/llama3.1/llama-3.1-8b-lora-alpaca-mlp-only"

echo ""
echo "=============================================="
echo "Layer-specific training complete!"
echo "Job finished at $(date)"
echo "=============================================="
