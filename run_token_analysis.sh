#!/bin/bash
#SBATCH -J token_analysis          # Job name
#SBATCH -p mit_normal              # Partition (CPU is sufficient)
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=16G                  # Memory
#SBATCH --time=00:30:00            # Walltime (30 minutes)
#SBATCH -o logs/token_analysis-%j.out        # Stdout
#SBATCH -e logs/token_analysis-%j.err        # Stderr

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules
echo "===== Loading modules ====="
module purge
module load miniforge/24.3.0-0

# Activate conda environment
echo "===== Activating conda environment ====="
conda activate ~/.conda/envs/promptenv

echo "===== Python version ====="
python --version

# Navigate to Methods directory
cd Methods

# Run token analysis script
echo "===== Starting token length analysis ====="
python token_length_analysis.py
echo "===== Token analysis finished ====="
