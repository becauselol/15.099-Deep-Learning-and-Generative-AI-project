# Countdown Evaluation Pipeline

This repository contains the full pipeline for evaluating countdown-style math puzzles using locally run LLMs.

## Contents

- **`transformed_countdown_prompts_gemini.csv`**  
  A dataset of rewritten / transformed prompts for the countdown task.

- **`countdown_features.csv`**  
  Extracted numerical and structural features for each countdown instance.

- **`run_countdown_llama_pairs.py`**  
  Main script that:
  - loads prompts and instance features  
  - generates random prompt–instance pairs  
  - queries the local LLaMA model  
  - checks correctness of the model’s answer  

- **`run_countdown_llama_pairs.sh`**  
  SLURM batch script to run the evaluation pipeline on MIT Engaging.

## Running on Engaging

Submit the job with:

```bash
sbatch run_countdown_llama_pairs.sh