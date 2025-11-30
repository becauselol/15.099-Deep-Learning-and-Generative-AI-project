#!/bin/bash
#SBATCH -J xgboost_embeddings      # Job name
#SBATCH -p mit_normal_gpu          # Partition (GPU for BERT embedding extraction)
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:l40s:1          # Request 1 L40S GPU (only for embedding extraction)
#SBATCH --cpus-per-task=16         # More CPUs for XGBoost training
#SBATCH --mem=64G                  # More memory for embedding storage
#SBATCH --time=01:30:00            # Walltime (1.5 hours)
#SBATCH -o logs/xgboost-%j.out     # Stdout
#SBATCH -e logs/xgboost-%j.err     # Stderr

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

# Run XGBoost training script
echo "===== Starting XGBoost with BERT embeddings ====="
python xgboost_model.py
echo "===== XGBoost training finished ====="
