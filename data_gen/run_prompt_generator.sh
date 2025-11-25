#!/bin/bash
#SBATCH --job-name=prompt_gen_gpu
#SBATCH --partition=mit_normal
#SBATCH -N 1                     # Number of nodes
#SBATCH --ntasks=1               # Number of tasks
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%j.out        # Stdout
#SBATCH -e logs/%x-%j.err        # Stderr

set -euo pipefail

# === Environment setup ===
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate ~/conda/envs/promptenv

cd ~/promptopt

echo "==== Job started on $(hostname) at $(date) ===="
echo "==== GPU info ===="
nvidia-smi || echo "No GPU info available"
echo "==================="

# === Run your model ===
srun python prompt_generator.py
echo "==== Job finished at $(date) ===="