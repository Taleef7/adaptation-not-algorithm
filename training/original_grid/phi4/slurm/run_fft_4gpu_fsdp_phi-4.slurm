#!/bin/bash
#SBATCH --job-name=phi4-fft-fsdp
#SBATCH --partition=training
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --qos=training
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=logs/finetune/phi4_fft_fsdp_%j.out
#SBATCH --error=logs/finetune/phi4_fft_fsdp_%j.err

echo "================================================================================"
echo "Phi-4 14B Full Fine-Tuning with FSDP (4x A100-80GB)"
echo "================================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x A100-80GB"
echo "Time: $(date)"
echo "================================================================================"

# Load required modules
module purge
module load anaconda/2025.06-py313
module list

# Activate conda environment
echo -e "\n==> Activating conda environment: thesis_env"
source activate thesis_env

# Verify environment
echo -e "\n==> Python version:"
python --version
echo -e "\n==> PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo -e "\n==> Transformers version:"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Set environment variables for memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Set CUDA architecture to avoid compilation overhead
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture

# Change to project directory
cd ${PROJECT_ROOT}

echo -e "\n================================================================================"
echo "Starting training with FSDP (torchrun launcher)..."
echo "================================================================================"

# Run training with torchrun (PyTorch native distributed launcher)
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    training/original_grid/phi4/finetune_fft_4gpu_fsdp_phi-4.py

exit_code=$?

echo -e "\n================================================================================"
if [ $exit_code -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code: $exit_code"
fi
echo "Time: $(date)"
echo "================================================================================"

exit $exit_code
