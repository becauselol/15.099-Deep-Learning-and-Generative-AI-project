#!/bin/bash
#SBATCH -J countdown_llama_run          # Job name
#SBATCH -p mit_normal_gpu               # GPU partition
#SBATCH -N 1                             # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --gres=gpu:h100:1               # Request 1 H100 GPU
#SBATCH --cpus-per-task=16              # CPU threads for llama.cpp
#SBATCH --mem=120G                      # Ample RAM
#SBATCH --time=06:00:00                 # Job time (hh:mm:ss)
#SBATCH -o logs/%x-%j.out               # Stdout
#SBATCH -e logs/%x-%j.err               # Stderr

# ===== Optional: HuggingFace token =====
export HF_TOKEN="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"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

# ===== Load modules =====
module purge
module load miniforge/24.3.0-0
module load cuda/12.4.0

# ===== Activate conda environment =====
conda activate ~/conda/envs/promptenv

# Sanity check
echo "===== GPU info ====="
nvidia-smi

echo "===== Python version ====="
python --version

# ===== Run Countdown Llama Script =====
echo "===== Starting run_countdown_llama_pairs.py ====="

python run_countdown_llama_pairs.py \
    --prompts_file transformed_countdown_prompts_gemini.csv \
    --instances_file countdown_features.csv \
    --pairs_file countdown_prompt_instance_results.csv \
    --repo_id TheBloke/Llama-2-7B-Chat-GGUF \
    --filename llama-2-7b-chat.Q4_K_M.gguf \
    --num_pairs 50000 \
    --max_calls 1000 \
    --max_tokens 512 \
    --temp 0.2 \
    --n_ctx 4096 \
    --n_gpu_layers -1 \
    --seed_local 0 \
    --verbose

echo "===== run_countdown_llama_pairs.py finished ====="