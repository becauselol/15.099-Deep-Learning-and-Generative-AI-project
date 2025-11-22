#!/bin/bash
#SBATCH -J ann_training            # Job name
#SBATCH -p mit_normal_gpu          # Partition
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:l40s:1          # Request 1 L40S GPU
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --mem=32G                  # Memory
#SBATCH --time=02:00:00            # Walltime (2 hours for 20 epochs)
#SBATCH -o logs/ann-%j.out         # Stdout
#SBATCH -e logs/ann-%j.err         # Stderr

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
echo "===== Loading modules ====="
module purge
module load miniforge/24.3.0-0
module load cuda/12.4.0

# Activate conda environment
echo "===== Activating conda environment ====="
conda activate ~/conda/envs/promptenv

# Sanity check: show GPU info
echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Python version ====="
python --version

# Navigate to Methods directory
cd Methods

# Run ANN training script
echo "===== Starting ANN training ====="
python ann_model.py
echo "===== ANN training finished ====="
