#!/bin/bash
#SBATCH -J train_all_ann       # Job name
#SBATCH -p mit_normal_gpu      # Partition (GPU partition)
#SBATCH -N 1                   # Number of nodes
#SBATCH --ntasks=1             # Number of tasks
#SBATCH --gres=gpu:l40s:1      # Request 1 L40S GPU
#SBATCH --cpus-per-task=8      # CPU cores per task
#SBATCH --mem=32G              # Memory
#SBATCH --time=06:00:00        # Walltime (6 hours for all 6 combinations)
#SBATCH -o logs/train_all_ann-%j.out   # Stdout
#SBATCH -e logs/train_all_ann-%j.err   # Stderr

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
echo "===== Loading modules ====="
module purge
module load miniforge/24.3.0-0
module load cuda/12.4.0

# Activate conda environment
echo "===== Activating conda environment ====="
conda activate ~/.conda/envs/promptenv

# Sanity check
echo "===== nvidia-smi ====="
nvidia-smi

echo "===== Python version ====="
python --version

# Navigate to Methods directory
cd Methods

# Run training for all combinations
echo "===== Starting ANN training for all combinations ====="
bash train_all_combinations.sh ann

echo "===== All ANN training finished ====="
