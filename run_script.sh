#!/bin/bash
#SBATCH -J medgemma_run          # Job name
#SBATCH -p mit_normal_gpu        # Partition
#SBATCH -N 1                     # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --gres=gpu:h200:1        # Request 1 H200 GPU
#SBATCH --cpus-per-task=8        # CPU cores per task (adjust as needed)
#SBATCH --mem=64G                # Memory (adjust as needed)
#SBATCH --time=06:00:00          # Walltime (hh:mm:ss)
#SBATCH -o logs/%x-%j.out        # Stdout
#SBATCH -e logs/%x-%j.err        # Stderr

# ===== Load .env file for configuration =====
# Load environment variables from .env file
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
    echo "Loaded configuration from .env file"
else
    echo "Warning: .env file not found"
fi

# ===== Load modules (adjust to your Engaging setup) =====
module purge
module load miniforge/24.3.0-0
module load cuda/12.4.0

# ===== Activate your Conda / virtualenv =====
# Example with conda:
conda activate ~/conda/envs/promptenv

# Sanity check: show GPU info
echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Python version ====="
python --version

# ===== Run your script =====
# Assumes script.py is in the submission directory
echo "===== Starting script.py ====="
python script.py
echo "===== script.py finished ====="