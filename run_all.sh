#!/bin/bash
#SBATCH -J all_models              # Job name
#SBATCH -p mit_normal_gpu          # Partition
#SBATCH -N 1                       # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:l40s:1          # Request 1 L40S GPU
#SBATCH --cpus-per-task=16         # CPU cores per task
#SBATCH --mem=64G                  # Memory
#SBATCH --time=06:00:00            # Walltime (6 hours for all models)
#SBATCH -o logs/all_models-%j.out  # Stdout
#SBATCH -e logs/all_models-%j.err  # Stderr

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

# Run all models sequentially
echo "===== Starting BERT finetuning ====="
python bert.py
echo "===== BERT training finished ====="

echo "===== Starting XGBoost with BERT embeddings ====="
python xgboost_model.py
echo "===== XGBoost training finished ====="

echo "===== Starting ANN training ====="
python ann_model.py
echo "===== ANN training finished ====="

echo "===== All models trained successfully ====="
