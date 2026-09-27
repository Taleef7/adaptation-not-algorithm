#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # No GPU needed for API calls
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4  # Request a few CPUs for processing
#SBATCH --mem=16G          # Modest memory requirement
#SBATCH --time=02:00:00    # Adjust time based on API speed/rate limits
#SBATCH --job-name=eval_refusals
#SBATCH --output=slurm_logs/evaluate_completions/Benign/%x_%A_%a.log # Log file name
#SBATCH --account=YOUR_ACCOUNT    # Your account
#SBATCH --partition=a30       # A CPU partition is fine, or default
#SBATCH --qos=standby

# --- Script Arguments ---
# $1: behaviors_path (e.g., ./data/behavior_datasets/data/benign-behaviors.csv)
# $2: completions_path (e.g., ./results/Benign_Temp/DirectRequest/default/completions/MODEL.json)
# $3: save_path (e.g., ./results/Benign/DirectRequest/default/results/MODEL.json)

echo "Evaluating refusals using API Judge"
echo "Behaviors Path: $1"
echo "Completions Path: $2"
echo "Save Path: $3"

# Activate Conda environment
module load external anaconda
conda activate harmbench_env # Make sure this is your correct env name

# Ensure output directory exists
mkdir -p "$(dirname "$3")"

# Run the Python script
# Using python -u for unbuffered output
python -u evaluate_refusals_api.py \
    --behaviors_path "$1" \
    --completions_path "$2" \
    --save_path "$3"

echo "Evaluation finished."