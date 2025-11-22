# Running Models on MIT Engaging Cluster

This guide explains how to run your ML models on the MIT Engaging cluster.

## Prerequisites

1. **SSH into Engaging:**
   ```bash
   ssh your_kerberos@orcd-login001.mit.edu
   ```

2. **Set up your conda environment (one-time setup):**
   ```bash
   # Load modules
   module load miniforge/24.3.0-0
   module load cuda/12.4.0

   # Create environment
   conda create -n promptenv python=3.10
   conda activate promptenv

   # Install dependencies
   pip install torch transformers datasets wandb scikit-learn xgboost pandas numpy tqdm python-dotenv
   ```

3. **Transfer your project files:**
   ```bash
   # From your local machine (not on cluster)
   scp -r "15.099 Homework/Project" your_kerberos@orcd-login001.mit.edu:/home/your_kerberos/
   ```

## Available Scripts

### Individual Model Scripts

| Script | Model | GPU | CPUs | Memory | Time | Use Case |
|--------|-------|-----|------|--------|------|----------|
| `run_bert.sh` | BERT Fine-tuning | L40S (1) | 8 | 32GB | 2h | End-to-end BERT training |
| `run_xgboost.sh` | XGBoost + BERT | L40S (1) | 16 | 64GB | 1.5h | XGBoost with embeddings |
| `run_ann.sh` | Simple ANN | L40S (1) | 8 | 32GB | 2h | Neural network training |
| `run_all.sh` | All Models | L40S (1) | 16 | 64GB | 6h | Train all models sequentially |

### GPU Options

If you need different GPUs, modify the `--gres` line:
```bash
#SBATCH --gres=gpu:h200:1    # H200: 140GB VRAM (for very large models)
#SBATCH --gres=gpu:l40s:1    # L40S: 44GB VRAM (recommended for BERT)
#SBATCH --gres=gpu:a100:1    # A100: 40/80GB VRAM
```

## How to Submit Jobs

### 1. Navigate to Project Directory
```bash
cd /home/your_kerberos/15.099\ Homework/Project
```

### 2. Make Scripts Executable
```bash
chmod +x run_bert.sh run_xgboost.sh run_ann.sh run_all.sh
```

### 3. Submit a Job
```bash
# Submit individual models
sbatch run_bert.sh
sbatch run_xgboost.sh
sbatch run_ann.sh

# Or train all models in one job
sbatch run_all.sh
```

### 4. Monitor Your Jobs
```bash
# Check job status
squeue --me

# Check detailed job info
scontrol show job <jobid>

# View real-time output
tail -f logs/bert-<jobid>.out

# Check for errors
tail -f logs/bert-<jobid>.err
```

### 5. Cancel a Job
```bash
scancel <jobid>
```

## After Job Completion

### View Results
```bash
# Check training logs
cat logs/bert-<jobid>.out

# View JSON summaries
cat Methods/training_summary.json
cat Methods/xgboost_training_summary.json
cat Methods/ann_training_summary.json
```

### Download Results to Local Machine
```bash
# From your local machine
scp your_kerberos@orcd-login001.mit.edu:/home/your_kerberos/15.099\ Homework/Project/Methods/*.json ./

# Download trained models
scp -r your_kerberos@orcd-login001.mit.edu:/home/your_kerberos/15.099\ Homework/Project/Methods/bert_finetuned_model ./
```

### Check Resource Usage
```bash
# After job completes
sacct -j <jobid> -o JobID,JobName,State,Elapsed,MaxRSS,ReqMem --units=G
```

## Troubleshooting

### Out of Memory Error
Increase memory in SBATCH directive:
```bash
#SBATCH --mem=128G  # Instead of 64G
```

### Job Timeout
Increase time limit:
```bash
#SBATCH --time=12:00:00  # 12 hours
```

### Conda Environment Not Found
Make sure the path matches your setup:
```bash
conda activate ~/conda/envs/promptenv
# Or if you created it elsewhere:
conda activate /path/to/your/env
```

### CUDA Out of Memory
Reduce batch size in `.env`:
```env
BERT_BATCH_SIZE=8   # Instead of 16
ANN_BATCH_SIZE=16   # Instead of 32
```

### Missing Dependencies
```bash
# On login node
module load miniforge/24.3.0-0
conda activate promptenv
pip install python-dotenv datasets wandb
```

## Best Practices

1. **Test Locally First:** Run a few epochs locally before submitting to cluster
2. **Use Preemptable for Long Jobs:** For jobs >6 hours, consider `mit_preemptable` partition
3. **Monitor Resource Usage:** Check `sacct` after first run to optimize future requests
4. **Check Logs Early:** Look at logs 5-10 minutes after submission to catch early errors
5. **Use .env for Configuration:** Modify hyperparameters in `.env` instead of code

## Interactive Testing

For quick testing/debugging:
```bash
# Request interactive session
srun --pty --partition=mit_normal_gpu --time=01:00:00 \
     --gres=gpu:l40s:1 --cpus-per-task=8 --mem=32G bash

# Load modules and activate environment
module load miniforge/24.3.0-0
module load cuda/12.4.0
conda activate promptenv

# Navigate and test
cd Methods
python bert.py
```

## Wandb Integration

Your models automatically log to Wandb. View results at:
- https://wandb.ai/your-username/bert-prompt-correctness

To login (first time only):
```bash
wandb login
# Enter your API key when prompted
```

## Support

- **ORCD Help:** orcd-help-engaging@mit.edu
- **Documentation:** https://orcd-docs.mit.edu/
- **Check logs directory:** `logs/` for detailed error messages
