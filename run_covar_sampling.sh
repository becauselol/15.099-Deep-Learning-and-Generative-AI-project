#!/bin/bash
#SBATCH -J covar_samp_5x0625
#SBATCH -p pi_dbertsim
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH -o logs/covar_sampling_%j.out
#SBATCH -e logs/covar_sampling_%j.err

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

cd ~/15.099-Deep-Learning-and-Generative-AI-project/Methods

python sampling_covariance_pipeline.py \
  --sampling-method diverse \
  --num-samples 5 \
  --sample-fraction 0.625 \
  --include-bert